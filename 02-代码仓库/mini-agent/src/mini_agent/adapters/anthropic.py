"""Anthropic Messages API 适配器：内部统一消息格式与 tool_use/tool_result 互转。"""
from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic

from mini_agent.config import Settings
from mini_agent.core.types import Message, ModelResponse, ToolCall, Usage


class AnthropicAdapter:
    """把 MiniAgent 内部格式转换成 Anthropic 原生 Messages API 格式。"""

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or settings.resolved_model
        self.max_tokens = settings.anthropic_max_tokens
        self._client = client or Anthropic(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
        )

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        """调用 Anthropic Messages API；供应商异常由核心循环统一处理。"""
        system, provider_messages = self.to_anthropic_messages(messages)
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=provider_messages,
            tools=self.to_anthropic_tools(tools),
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return self.to_model_response(response)

    @staticmethod
    def to_anthropic_messages(
        messages: list[Message],
    ) -> tuple[str, list[dict[str, Any]]]:
        """内部消息 -> Anthropic messages。

        Anthropic 的 system 是独立参数；tool_result 必须放在 user 消息中，
        并且要跟在上一条 assistant tool_use 消息之后。
        """
        system_parts: list[str] = []
        provider_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                provider_messages.append(
                    {"role": "user", "content": list(pending_tool_results)}
                )
                pending_tool_results.clear()

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            if message.role == "tool":
                pending_tool_results.append(
                    AnthropicAdapter._to_tool_result_block(message)
                )
                continue

            flush_tool_results()

            if message.role == "assistant":
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for call in message.tool_calls or []:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                provider_messages.append(
                    {
                        "role": "assistant",
                        "content": content or [{"type": "text", "text": ""}],
                    }
                )
            elif message.role == "user":
                provider_messages.append(
                    {
                        "role": "user",
                        "content": message.content or "",
                    }
                )
            else:
                raise ValueError(f"unsupported message role: {message.role}")

        flush_tool_results()
        return "\n\n".join(system_parts), provider_messages

    @staticmethod
    def _to_tool_result_block(message: Message) -> dict[str, Any]:
        """内部 tool 消息 -> Anthropic tool_result content block。"""
        content = message.content or ""
        is_error = False
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                is_error = payload.get("ok") is False
        except json.JSONDecodeError:
            pass
        return {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id,
            "content": content,
            "is_error": is_error,
        }

    @staticmethod
    def to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """OpenAI function schema -> Anthropic tool schema。"""
        result: list[dict[str, Any]] = []
        for item in tools:
            if item.get("type") != "function" or "function" not in item:
                raise ValueError("tool schema must be OpenAI function format")
            function = item["function"]
            result.append(
                {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "input_schema": function["parameters"],
                }
            )
        return result

    @staticmethod
    def to_model_response(response: Any) -> ModelResponse:
        """Anthropic response -> MiniAgent 统一 ModelResponse。"""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                arguments = getattr(block, "input", {})
                if not isinstance(arguments, dict):
                    arguments = {"_raw": arguments}
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=arguments,
                    )
                )

        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = Usage(
                prompt_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
            )
        return ModelResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            usage=usage,
        )