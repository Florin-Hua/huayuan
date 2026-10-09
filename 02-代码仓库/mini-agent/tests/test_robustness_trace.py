"""阶段6 健壮性与 Trace 验：全部离线，使用 scripted adapter 和失败注入工具。"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini_agent.cli import build_parser
from mini_agent.config import Settings
from mini_agent.core.agent import Agent
from mini_agent.core.loop import AgentCore
from mini_agent.core.types import (
    Message,
    ModelResponse,
    ModelResponseError,
    RunStatus,
    ToolCall,
    Usage,
)
from mini_agent.tools.registry import ToolRegistry, tool
from mini_agent.trace import TraceStore


class CrashInput(BaseModel):
    value: str = Field(description="任意输入")


class SlowInput(BaseModel):
    seconds: float = Field(description="睡眠秒数")


@tool(CrashInput)
def crash_tool(value: str) -> str:
    """故意抛出异常，测试工具错误归一化。"""
    raise RuntimeError(f"boom: {value}")


@tool(SlowInput, timeout_seconds=0.01)
def timeout_tool(seconds: float) -> str:
    """故意超时。"""
    time.sleep(seconds)
    return "done"


class ScriptedAdapter:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = deque(responses)
        self.seen_messages: list[list[Message]] = []
        self.calls = 0

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.calls += 1
        self.seen_messages.append(list(messages))
        if not self.responses:
            raise AssertionError("script has no more responses")
        return self.responses.popleft()


def tool_response(
    name: str,
    arguments: dict[str, Any],
    call_id: str = "call_1",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> ModelResponse:
    return ModelResponse(
        content="需要调用工具。",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def final_response(answer: str) -> ModelResponse:
    return ModelResponse(
        content=answer,
        usage=Usage(prompt_tokens=20, completion_tokens=10),
    )


def invalid_response(call_id: str = "call_bad") -> ModelResponse:
    return ModelResponse(
        content=" malformed ",
        tool_calls=[
            ToolCall(id=call_id, name="calculator", arguments={"_raw": "{bad json"})
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def test_invalid_model_output_recovers_by_feedback_retry(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        [
            invalid_response(),
            tool_response("calculator", {"expression": "1+1"}),
            final_response("结果是 2。"),
        ]
    )
    result = AgentCore(adapter, max_iterations=3).run("计算 1+1", tmp_path)

    assert result.status is RunStatus.SUCCESS
    assert result.parse_retries == 1
    assert any(step.state == "RETRY" for step in result.steps)
    assert adapter.calls == 3
    feedback = [m for m in adapter.seen_messages[1] if m.role == "user"][-1]
    assert "模型输出非法" in (feedback.content or "")
    assert "JSON object" in (feedback.content or "")


def test_persistent_invalid_output_terminates_after_two_retries(tmp_path: Path) -> None:
    adapter = ScriptedAdapter([invalid_response() for _ in range(3)])
    result = AgentCore(adapter, max_iterations=3).run("测试解析失败", tmp_path)

    assert result.status is RunStatus.PARSE_ERROR
    assert result.parse_retries == 2
    assert adapter.calls == 3
    assert "已重试 2 次" in (result.termination_reason or "")


def test_tool_exception_is_structured_and_fed_back(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(crash_tool)
    adapter = ScriptedAdapter(
        [
            tool_response("crash_tool", {"value": "x"}),
            final_response("工具执行失败，任务无法完成。"),
        ]
    )
    core = AgentCore(adapter, max_iterations=2, registry=registry)
    result = core.run("触发工具异常", tmp_path)

    assert result.status is RunStatus.SUCCESS
    act = next(step for step in result.steps if step.state == "ACT")
    assert act.ok is False
    assert "RuntimeError" in (act.error or "")
    tool_messages = [m for m in adapter.seen_messages[-1] if m.role == "tool"]
    assert tool_messages and "RuntimeError" in (tool_messages[-1].content or "")


def test_tool_timeout_is_structured_and_fed_back(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(timeout_tool)
    adapter = ScriptedAdapter(
        [
            tool_response("timeout_tool", {"seconds": 0.2}),
            final_response("工具超时，任务无法完成。"),
        ]
    )
    core = AgentCore(adapter, max_iterations=2, registry=registry)
    result = core.run("触发工具超时", tmp_path)

    assert result.status is RunStatus.SUCCESS
    act = next(step for step in result.steps if step.state == "ACT")
    assert act.ok is False
    assert act.error is not None
    assert act.error.startswith("ToolTimeoutError")


def test_token_budget_guard_terminates_gracefully(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        [tool_response("calculator", {"expression": "1+1"}, prompt_tokens=80, completion_tokens=40)]
    )
    core = AgentCore(adapter, max_iterations=2, token_budget=100)
    result = core.run("触发 token 预算", tmp_path)

    assert result.status is RunStatus.TOKEN_BUDGET
    assert result.usage.total == 120
    assert "token 预算超限" in (result.termination_reason or "")
    assert not any(step.state == "ACT" for step in result.steps)


def test_oscillation_guard_stops_second_identical_call(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        [
            tool_response("calculator", {"expression": "1+1"}, "call_1"),
            tool_response("calculator", {"expression": "1+1"}, "call_2"),
        ]
    )
    core = AgentCore(adapter, max_iterations=3)
    result = core.run("重复调用同一个工具", tmp_path)

    assert result.status is RunStatus.OSCILLATION
    assert sum(step.state == "ACT" for step in result.steps) == 1
    assert "完全相同的工具与参数" in (result.termination_reason or "")


def test_trace_store_persists_run_and_steps(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.sqlite3"
    adapter = ScriptedAdapter(
        [
            tool_response("calculator", {"expression": "2*3"}),
            final_response("结果是 6。"),
        ]
    )
    with TraceStore(trace_path) as tracer:
        result = AgentCore(adapter, max_iterations=2, tracer=tracer).run(
            "计算 2*3", tmp_path
        )

    with TraceStore(trace_path) as tracer:
        run = tracer.get_run(result.run_id)
        steps = tracer.get_steps(result.run_id)

    assert run is not None
    assert run["status"] == "success"
    assert run["iterations"] == 2
    assert run["total_tokens"] == 45
    assert len(steps) == len(result.steps)
    act = next(step for step in steps if step["state"] == "ACT")
    assert act["tool_name"] == "calculator"
    assert act["tool_args"] == '{"expression": "2*3"}'
    assert act["result_status"] == "ok"


def test_agent_integrates_trace_store(tmp_path: Path) -> None:
    trace_path = tmp_path / "agent-trace.sqlite3"
    settings = Settings(
        memory_enabled=False,
        trace_enabled=True,
        trace_db_path=str(trace_path),
        sandbox_dir=str(tmp_path / "sandbox"),
    )
    adapter = ScriptedAdapter([final_response("hello")])
    with Agent(adapter=adapter, settings=settings) as agent:
        result = agent.run("打招呼")

    with TraceStore(trace_path) as tracer:
        run = tracer.get_run(result.run_id)
    assert run is not None
    assert run["task"] == "打招呼"
    assert run["status"] == "success"


def test_cli_parser_supports_trace_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["trace", "abc123"])
    assert args.command == "trace"
    assert args.run_id == "abc123"


class ParseErrorThenSuccessAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            raise ModelResponseError("tool calculator arguments is not a JSON object")
        return final_response("解析错误已恢复。")


def test_adapter_model_response_error_recovers_by_retry(tmp_path: Path) -> None:
    adapter = ParseErrorThenSuccessAdapter()
    result = AgentCore(adapter, max_iterations=2).run("测试 adapter 抛解析错误", tmp_path)

    assert result.status is RunStatus.SUCCESS
    assert result.parse_retries == 1
    assert adapter.calls == 2
    assert result.answer == "解析错误已恢复。"
