"""模型适配器工厂：根据配置选择 provider，核心代码零改动切换。"""
from __future__ import annotations

from typing import Protocol

from mini_agent.adapters.anthropic import AnthropicAdapter
from mini_agent.adapters.openai import OpenAIAdapter
from mini_agent.config import Settings


class ChatAdapter(Protocol):
    """供应商适配器必须实现的统一接口。"""

    def chat(self, messages, tools):
        """调用一次模型并返回统一 ModelResponse。"""
        ...


def create_adapter(
    settings: Settings,
    model: str | None = None,
    client: object | None = None,
) -> ChatAdapter:
    """按 MODEL_PROVIDER 创建 OpenAI 或 Anthropic adapter。"""
    provider = settings.model_provider.lower()
    if provider == "openai":
        return OpenAIAdapter(settings, model, client=client)
    if provider == "anthropic":
        return AnthropicAdapter(settings, model, client=client)
    raise ValueError(
        f"unsupported MODEL_PROVIDER: {settings.model_provider}; "
        "expected openai or anthropic"
    )