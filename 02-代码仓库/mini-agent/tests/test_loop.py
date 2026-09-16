"""阶段1 单元测试：不访问网络，使用脚本化模型适配器。"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from mini_agent.core.loop import AgentCore
from mini_agent.core.types import Message, ModelResponse, RunStatus, ToolCall, Usage
from mini_agent.tools.builtin import calculator
from mini_agent.tools.registry import execute_tool


class ScriptedAdapter:
    """按剧本返回模型输出，并记录每轮看到的完整消息。"""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = deque(responses)
        self.seen_messages: list[Message] = []

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.seen_messages = list(messages)
        if not self.responses:
            raise AssertionError("script has no more responses")
        return self.responses.popleft()


def call_tool(
    name: str, arguments: dict, call_id: str = "call_1"
) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def response_with_tool(name: str, arguments: dict, call_id: str = "call_1") -> ModelResponse:
    return ModelResponse(
        content="我需要先调用工具。",
        tool_calls=[call_tool(name, arguments, call_id)],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def final_answer(answer: str) -> ModelResponse:
    return ModelResponse(content=answer, usage=Usage(prompt_tokens=20, completion_tokens=10))


def test_success_after_tool_call(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        [
            response_with_tool("calculator", {"expression": "23*47+11"}),
            final_answer("23*47+11 的结果是 1092。"),
        ]
    )
    result = AgentCore(adapter, max_iterations=3).run("计算 23*47+11", tmp_path)

    assert result.status is RunStatus.SUCCESS
    assert result.answer == "23*47+11 的结果是 1092。"
    assert result.iterations == 2
    assert result.usage.prompt_tokens == 30
    assert result.usage.completion_tokens == 15
    assert [step.state for step in result.steps] == ["THINK", "ACT", "OBSERVE", "THINK", "END"]
    assert any(step.tool_name == "calculator" and step.ok for step in result.steps)


def test_max_iterations_when_model_never_answers(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        [
            response_with_tool("calculator", {"expression": "1+1"}, "call_1"),
            response_with_tool("calculator", {"expression": "2+2"}, "call_2"),
        ]
    )
    result = AgentCore(adapter, max_iterations=2).run("不停调用工具", tmp_path)

    assert result.status is RunStatus.MAX_ITER
    assert result.iterations == 2
    assert "最大迭代次数" in (result.termination_reason or "")
    assert result.answer is not None


def test_model_exception_terminates_as_parse_error(tmp_path: Path) -> None:
    class BrokenAdapter:
        def chat(self, messages, tools):
            raise RuntimeError("network unavailable")

    result = AgentCore(BrokenAdapter(), max_iterations=3).run("测试异常", tmp_path)
    assert result.status is RunStatus.PARSE_ERROR
    assert result.iterations == 1
    assert "network unavailable" in (result.termination_reason or "")
    assert result.answer is not None


def test_tool_error_is_fed_back_to_model(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        [
            response_with_tool("read_text_file", {"path": "missing.txt"}),
            final_answer("文件不存在，无法读取。"),
        ]
    )
    result = AgentCore(adapter, max_iterations=3).run("读取 missing.txt", tmp_path)

    assert result.status is RunStatus.SUCCESS
    tool_messages = [message for message in adapter.seen_messages if message.role == "tool"]
    assert tool_messages
    payload = json.loads(tool_messages[-1].content or "{}")
    assert payload["ok"] is False
    assert "file not found" in payload["error"]
    assert result.answer == "文件不存在，无法读取。"


def test_unknown_tool_returns_structured_error(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        [
            response_with_tool("web_search", {"query": "test"}),
            final_answer("当前没有 web_search 工具。"),
        ]
    )
    result = AgentCore(adapter, max_iterations=3).run("调用不存在工具", tmp_path)

    assert result.status is RunStatus.SUCCESS
    act_step = next(step for step in result.steps if step.state == "ACT")
    assert act_step.ok is False
    assert "unknown tool" in (act_step.error or "")


def test_calculator_supports_basic_operations() -> None:
    assert calculator("23*47+11") == 1092
    assert calculator("-7 + 2") == -5
    assert calculator("7 / 2") == 3.5


@pytest.mark.parametrize("expression", ["2**10000", "__import__('os')", "'abc' + 'd'"])
def test_calculator_rejects_unsafe_or_huge_expressions(expression: str) -> None:
    with pytest.raises(ValueError):
        calculator(expression)


def test_read_file_is_limited_to_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    result = execute_tool(
        call_tool("read_text_file", {"path": "../outside.txt"}), sandbox
    )
    assert result.ok is False
    assert "escapes sandbox" in (result.error or "")


def test_read_file_success(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "notes.txt").write_text("hello", encoding="utf-8")

    result = execute_tool(call_tool("read_text_file", {"path": "notes.txt"}), sandbox)
    assert result.ok is True
    assert result.data == {
        "path": "notes.txt",
        "content": "hello",
        "truncated": False,
    }