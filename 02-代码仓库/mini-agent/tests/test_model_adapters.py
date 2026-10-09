"""阶段3 多模型适配测试：OpenAI 与 Anthropic 格式互转、工厂与统一执行。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mini_agent.adapters.anthropic import AnthropicAdapter
from mini_agent.adapters.factory import create_adapter
from mini_agent.adapters.openai import OpenAIAdapter
from mini_agent.config import Settings
from mini_agent.core.loop import AgentCore
from mini_agent.core.types import ModelResponseError, Message, RunStatus, ToolCall

FIXTURES = Path(__file__).parent / "fixtures"


def object_from_json(data: Any) -> Any:
    """把 JSON fixture 转成 SDK 响应类似的属性对象。

    Anthropic 的 tool_use.input 在 SDK 中保持 dict，不能递归转成 namespace。
    """
    if isinstance(data, dict):
        return SimpleNamespace(
            **{
                key: value if key == "input" else object_from_json(value)
                for key, value in data.items()
            }
        )
    if isinstance(data, list):
        return [object_from_json(item) for item in data]
    return data


def load_fixture(name: str) -> Any:
    return object_from_json(json.loads((FIXTURES / name).read_text(encoding="utf-8")))


def base_settings(provider: str) -> Settings:
    return Settings(
        model_provider=provider,
        model_name="fixture-model",
        anthropic_api_key="test-key",
    )


def internal_messages() -> list[Message]:
    return [
        Message(role="system", content="system prompt"),
        Message(role="user", content="计算 23*47+11"),
        Message(
            role="assistant",
            content="我需要先调用工具。",
            tool_calls=[
                ToolCall(
                    id="call_001",
                    name="calculator",
                    arguments={"expression": "23*47+11"},
                )
            ],
        ),
        Message(
            role="tool",
            content='{"ok": true, "data": 1092, "error": null}',
            tool_call_id="call_001",
            name="calculator",
        ),
    ]


def test_openai_internal_to_provider_messages() -> None:
    adapter = OpenAIAdapter(base_settings("openai"), client=object())
    provider_messages = [adapter._to_openai_message(m) for m in internal_messages()]

    assert provider_messages[0] == {"role": "system", "content": "system prompt"}
    assert provider_messages[1] == {"role": "user", "content": "计算 23*47+11"}

    assistant = provider_messages[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_001"
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "expression": "23*47+11"
    }

    tool = provider_messages[3]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_001"
    assert tool["name"] == "calculator"


def test_openai_recorded_response_to_model_response() -> None:
    response = load_fixture("openai_response.json")
    result = OpenAIAdapter.to_model_response(response)

    assert result.content is None
    assert result.tool_calls == [
        ToolCall(
            id="call_fixture_001",
            name="calculator",
            arguments={"expression": "23*47+11"},
        )
    ]
    assert result.usage is not None
    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 35


def test_openai_invalid_tool_arguments_raise_model_response_error() -> None:
    raw = object_from_json(
        {
            "id": "call_bad",
            "function": {
                "name": "calculator",
                "arguments": "{invalid json",
            }
        }
    )
    with pytest.raises(ModelResponseError, match="not a JSON object"):
        OpenAIAdapter._to_tool_call(raw)


def test_anthropic_internal_to_provider_messages() -> None:
    system, provider_messages = AnthropicAdapter.to_anthropic_messages(
        internal_messages()
    )

    assert system == "system prompt"
    assert provider_messages[0] == {"role": "user", "content": "计算 23*47+11"}

    assistant = provider_messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {
        "type": "text",
        "text": "我需要先调用工具。",
    }
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "call_001",
        "name": "calculator",
        "input": {"expression": "23*47+11"},
    }

    tool_result_message = provider_messages[2]
    assert tool_result_message["role"] == "user"
    block = tool_result_message["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_001"
    assert block["is_error"] is False
    assert json.loads(block["content"])["data"] == 1092


def test_anthropic_error_tool_result_is_marked() -> None:
    messages = [
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="call_err", name="read_text_file", arguments={})],
        ),
        Message(
            role="tool",
            content='{"ok": false, "data": null, "error": "file not found"}',
            tool_call_id="call_err",
            name="read_text_file",
        ),
    ]
    _, provider_messages = AnthropicAdapter.to_anthropic_messages(messages)
    block = provider_messages[-1]["content"][0]
    assert block["is_error"] is True
    assert "file not found" in block["content"]


def test_anthropic_tool_schema_conversion() -> None:
    openai_schema = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算表达式",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]
    result = AnthropicAdapter.to_anthropic_tools(openai_schema)
    assert result == [
        {
            "name": "calculator",
            "description": "计算表达式",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]


def test_anthropic_recorded_response_to_model_response() -> None:
    response = load_fixture("anthropic_response.json")
    result = AnthropicAdapter.to_model_response(response)

    assert result.content == "我需要先计算表达式。"
    assert result.tool_calls == [
        ToolCall(
            id="toolu_fixture_001",
            name="calculator",
            arguments={"expression": "23*47+11"},
        )
    ]
    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 40


def test_factory_selects_provider() -> None:
    openai_adapter = create_adapter(
        base_settings("openai"), model="gpt-fixture", client=object()
    )
    anthropic_adapter = create_adapter(
        base_settings("anthropic"), model="claude-fixture", client=object()
    )
    assert isinstance(openai_adapter, OpenAIAdapter)
    assert isinstance(anthropic_adapter, AnthropicAdapter)


def test_factory_rejects_unknown_provider() -> None:
    settings = base_settings("openai")
    settings.model_provider = "invalid"
    with pytest.raises(ValueError):
        create_adapter(settings)


class FakeOpenAIClient:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeAnthropicClient:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_same_task_runs_on_both_provider_formats(tmp_path: Path) -> None:
    task = "计算 23*47+11 并告诉我结果"
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    openai_client = FakeOpenAIClient(
        [
            object_from_json(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_openai",
                                        "function": {
                                            "name": "calculator",
                                            "arguments": "{\"expression\": \"23*47+11\"}",
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                }
            ),
            object_from_json(
                {
                    "choices": [{"message": {"content": "结果是 1092。", "tool_calls": None}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 12},
                }
            ),
        ]
    )
    anthropic_client = FakeAnthropicClient(
        [
            object_from_json(
                {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_anthropic",
                            "name": "calculator",
                            "input": {"expression": "23*47+11"},
                        }
                    ],
                    "usage": {"input_tokens": 25, "output_tokens": 11},
                }
            ),
            object_from_json(
                {
                    "content": [{"type": "text", "text": "结果是 1092。"}],
                    "usage": {"input_tokens": 35, "output_tokens": 13},
                }
            ),
        ]
    )

    cases = [
        (base_settings("openai"), openai_client, "fixture-model"),
        (base_settings("anthropic"), anthropic_client, "fixture-model"),
    ]
    for settings, client, model in cases:
        adapter = create_adapter(settings, model=model, client=client)
        result = AgentCore(adapter, max_iterations=3).run(task, sandbox)
        assert result.status is RunStatus.SUCCESS
        assert result.answer == "结果是 1092。"
        assert result.iterations == 2
        assert client.calls
        assert client.calls[0]["tools"]