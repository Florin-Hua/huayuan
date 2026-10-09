"""OpenAI Chat Completions 适配器：把框架消息转成供应商格式，再转回来。"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from mini_agent.config import Settings
from mini_agent.core.types import (
    Message,
    ModelResponse,
    ModelResponseError,
    ToolCall,
    Usage,
)


class OpenAIAdapter:
    """OpenAI 兼容接口适配器，覆盖 GPT/DeepSeek/Qwen 等兼容服务。"""

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or settings.resolved_model
        self._client = client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        """调用一次模型；供应商异常由调用方统一转为 PARSE_ERROR。"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[self._to_openai_message(message) for message in messages],
            tools=tools,
            temperature=0.2,
        )
        return self.to_model_response(response)

    @staticmethod
    def to_model_response(response: Any) -> ModelResponse:
        """OpenAI response -> MiniAgent 统一 ModelResponse。"""
        if not response.choices:
            raise ModelResponseError("model response has no choices")
        choice = response.choices[0]
        message = choice.message
        tool_calls = [OpenAIAdapter._to_tool_call(call) for call in message.tool_calls or []]
        usage = None
        if response.usage is not None:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        return ModelResponse(content=message.content, tool_calls=tool_calls, usage=usage)

    def _to_openai_message(self, message: Message) -> dict[str, Any]:
        if message.role == "assistant":
            data: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                data["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments, ensure_ascii=False
                            ),
                        },
                    }
                    for call in message.tool_calls
                ]
            return data
        if message.role == "tool":
            return {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
                "name": message.name,
            }
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _to_tool_call(raw: Any) -> ToolCall:
        function = raw.function
        arguments: dict[str, Any]
        try:
            parsed = json.loads(function.arguments or "{}")
            if not isinstance(parsed, dict):
                raise ValueError("tool arguments must be a JSON object")
            arguments = parsed
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelResponseError(
                f"tool {function.name} arguments is not a JSON object: {exc}"
            ) from exc
        return ToolCall(id=raw.id, name=function.name, arguments=arguments)