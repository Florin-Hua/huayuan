"""阶段7 MCP 兼容测试：真实 stdio 验收 + mock 协议结果。"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from mcp.types import CallToolResult, TextContent

from mini_agent.cli import build_parser
from mini_agent.config import Settings
from mini_agent.core.agent import Agent
from mini_agent.core.types import ModelResponse, RunStatus, ToolCall, Usage
from mini_agent.mcp_tools import McpToolAdapter
from mini_agent.tools.registry import ToolRegistry


class ScriptedAdapter:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)

    def chat(self, messages, tools):
        return self.responses.pop(0)


def tool_response(name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(
        content="调用 MCP 工具。",
        tool_calls=[ToolCall(id="mcp_call", name=name, arguments=arguments)],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def final_response(answer: str) -> ModelResponse:
    return ModelResponse(
        content=answer,
        usage=Usage(prompt_tokens=20, completion_tokens=10),
    )


def test_mcp_adapter_discovers_and_calls_real_stdio_server() -> None:
    adapter = McpToolAdapter([sys.executable, "-m", "mini_agent.mcp_server"])
    try:
        registry = ToolRegistry()
        definitions = adapter.register_tools(registry)

        assert [item.name for item in definitions] == ["mcp_echo", "mcp_add"]
        schema = registry.get("mcp_add").schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mcp_add"
        assert schema["function"]["parameters"]["required"] == ["a", "b"]

        result = registry.execute(
            ToolCall(
                id="call_add",
                name="mcp_add",
                arguments={"a": 20, "b": 22},
            )
        )
        assert result.ok is True
        assert result.data == 42
    finally:
        adapter.close()


def test_agent_calls_mcp_tool_like_builtin_tool(tmp_path) -> None:
    settings = Settings(
        memory_enabled=False,
        trace_enabled=False,
        sandbox_dir=str(tmp_path),
        context_compression_enabled=False,
    )
    adapter = ScriptedAdapter(
        [
            tool_response("mcp_add", {"a": 19, "b": 23}),
            final_response("19 + 23 = 42"),
        ]
    )
    with Agent(
        adapter=adapter,
        settings=settings,
        sandbox_dir=tmp_path,
        mcp_server_command=[sys.executable, "-m", "mini_agent.mcp_server"],
    ) as agent:
        result = agent.run("用 MCP 计算 19 + 23")

    assert result.status is RunStatus.SUCCESS
    assert result.answer == "19 + 23 = 42"
    act = next(step for step in result.steps if step.state == "ACT")
    assert act.tool_name == "mcp_add"
    assert act.ok is True


class FakeMcpClient:
    def __init__(self, result: CallToolResult):
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="mcp_json",
                    description="Return JSON payload",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "value": {"type": "integer", "description": "Input value"}
                        },
                        "required": ["value"],
                    },
                )
            ]
        )

    def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return self.result

    def close(self) -> None:
        return None


def test_mcp_json_result_is_parsed_and_forwarded() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps({"value": 7}))
        ]
    )
    fake = FakeMcpClient(result)
    adapter = McpToolAdapter(client=fake)
    registry = ToolRegistry()
    adapter.register_tools(registry)

    tool_result = registry.execute(
        ToolCall(id="json", name="mcp_json", arguments={"value": 7})
    )
    assert tool_result.ok is True
    assert tool_result.data == {"value": 7}
    assert fake.calls == [("mcp_json", {"value": 7})]


def test_mcp_error_result_is_normalized() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="mcp server exploded")],
        is_error=True,
    )
    adapter = McpToolAdapter(client=FakeMcpClient(result))
    registry = ToolRegistry()
    adapter.register_tools(registry)

    tool_result = registry.execute(
        ToolCall(id="error", name="mcp_json", arguments={"value": 1})
    )
    assert tool_result.ok is False
    assert tool_result.error is not None
    assert "McpToolError" in tool_result.error
    assert "mcp server exploded" in tool_result.error


def test_cli_parser_supports_mcp_server_option() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "task", "--mcp-server", "python -m mini_agent.mcp_server"])
    assert args.command == "run"
    assert args.mcp_server == "python -m mini_agent.mcp_server"
