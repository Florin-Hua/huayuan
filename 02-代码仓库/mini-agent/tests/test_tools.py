"""阶段2 工具系统测试：schema、注册、超时、沙箱与多步调度。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini_agent.core.loop import AgentCore
from mini_agent.core.types import ModelResponse, RunStatus, ToolCall, Usage
from mini_agent.tools.builtin import read_text_file, write_text_file
from mini_agent.tools.registry import ToolRegistry, tool


class EchoInput(BaseModel):
    message: str = Field(description="要原样返回的文本")


class SleepInput(BaseModel):
    seconds: float = Field(description="睡眠秒数")


@tool(EchoInput)
def echo_tool(message: str) -> dict[str, Any]:
    """返回输入文本。"""
    return {"message": message}


@tool
def upper_case(text: str) -> str:
    """把文本转为大写。"""
    return text.upper()


@tool(SleepInput, timeout_seconds=0.01)
def slow_tool(seconds: float) -> str:
    """故意睡眠，用于测试超时。"""
    time.sleep(seconds)
    return "done"


class ScriptedAdapter:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)

    def chat(self, messages, tools):
        return self.responses.pop(0)


def response_with_tool(
    name: str, arguments: dict[str, Any], call_id: str
) -> ModelResponse:
    return ModelResponse(
        content="继续执行工具。",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def test_schema_generation_from_explicit_pydantic_model() -> None:
    registry = ToolRegistry()
    registry.register(echo_tool)

    assert registry.names == ["echo_tool"]
    schema = registry.schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo_tool"
    assert schema["function"]["description"] == "返回输入文本。"
    assert schema["function"]["parameters"]["properties"]["message"] == {
        "type": "string",
        "description": "要原样返回的文本",
    }
    assert schema["function"]["parameters"]["required"] == ["message"]


def test_schema_generation_from_function_signature() -> None:
    schema = upper_case.schema
    assert schema["function"]["name"] == "upper_case"
    assert schema["function"]["parameters"]["properties"]["text"]["type"] == "string"
    assert schema["function"]["parameters"]["required"] == ["text"]


def test_registry_executes_validated_arguments() -> None:
    registry = ToolRegistry()
    registry.register(echo_tool)

    result = registry.execute(
        ToolCall(id="call_1", name="echo_tool", arguments={"message": "hello"})
    )
    assert result.ok is True
    assert result.data == {"message": "hello"}


def test_invalid_arguments_return_structured_error() -> None:
    registry = ToolRegistry()
    registry.register(echo_tool)

    result = registry.execute(
        ToolCall(id="call_1", name="echo_tool", arguments={"wrong": "argument"})
    )
    assert result.ok is False
    assert result.error is not None
    assert "ToolInputError" in result.error


def test_hallucinated_tool_returns_structured_error() -> None:
    registry = ToolRegistry()
    result = registry.execute(
        ToolCall(id="call_1", name="does_not_exist", arguments={})
    )
    assert result.ok is False
    assert result.error == "unknown tool: does_not_exist"


def test_tool_timeout_is_normalized() -> None:
    registry = ToolRegistry()
    registry.register(slow_tool)

    result = registry.execute(
        ToolCall(id="call_slow", name="slow_tool", arguments={"seconds": 0.2})
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.startswith("ToolTimeoutError")
    assert result.duration_ms < 100


def test_duplicate_tool_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(echo_tool)
    try:
        registry.register(echo_tool)
    except ValueError as exc:
        assert "duplicate tool" in str(exc)
    else:
        raise AssertionError("duplicate tool was accepted")


def test_sandbox_write_and_read_roundtrip(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    write_result = write_text_file("result.txt", "42", sandbox)
    assert write_result == {
        "path": "result.txt",
        "bytes_written": 2,
        "overwritten": False,
    }

    read_result = read_text_file("result.txt", sandbox)
    assert read_result == {
        "path": "result.txt",
        "content": "42",
        "truncated": False,
    }


def test_read_and_write_sandbox_escape_are_rejected(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside.txt"

    registry = ToolRegistry()
    registry.register(write_text_file)
    registry.register(read_text_file)
    write_call = registry.execute(
        ToolCall(id="w", name="write_text_file", arguments={"path": "../outside.txt", "content": "x"}),
        sandbox,
    )
    read_call = registry.execute(
        ToolCall(id="r", name="read_text_file", arguments={"path": "../outside.txt"}),
        sandbox,
    )
    assert write_call.ok is False
    assert "escapes sandbox" in write_call.error
    assert read_call.ok is False
    assert "escapes sandbox" in read_call.error
    assert not outside.exists()


def test_multistep_read_write_and_verify(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "notes.txt").write_text("a\nb\nc\n", encoding="utf-8")

    adapter = ScriptedAdapter(
        [
            response_with_tool("read_text_file", {"path": "notes.txt"}, "call_read"),
            response_with_tool(
                "write_text_file",
                {"path": "result.txt", "content": "3"},
                "call_write",
            ),
            response_with_tool("read_text_file", {"path": "result.txt"}, "call_verify"),
            ModelResponse(
                content="notes.txt 有 3 行；result.txt 已写入并验证为 3。",
                usage=Usage(prompt_tokens=30, completion_tokens=20),
            ),
        ]
    )
    result = AgentCore(adapter, max_iterations=5).run(
        "读取 notes.txt，统计行数，写入 result.txt，再读回验证", sandbox
    )

    assert result.status is RunStatus.SUCCESS
    assert result.iterations == 4
    assert (sandbox / "result.txt").read_text(encoding="utf-8") == "3"
    tool_sequence = [
        step.tool_name for step in result.steps if step.state == "ACT"
    ]
    assert tool_sequence == ["read_text_file", "write_text_file", "read_text_file"]