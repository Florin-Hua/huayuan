"""LLM 抽象层 — 工厂模式 + 多 Provider + tool_calls 解析"""
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 统一响应"""
    content: str = ""
    tool_calls: List[Dict[str, Any]] = None
    finish_reason: str = ""
    tokens_used: int = 0

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class BaseLLMProvider(ABC):
    """LLM 提供商基类"""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """对话接口 — 返回 LLMResponse（含 tool_calls）"""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """流式对话接口 — 逐 token 产出文本"""
        pass

    def get_token_count(self, text: str) -> int:
        """计算 token 数量（粗略估算）"""
        return len(text) // 3


# ====================================================================== #
#  OpenAI Provider
# ====================================================================== #

class OpenAIProvider(BaseLLMProvider):
    """OpenAI / 兼容 API (DeepSeek, Moonshot, etc.)"""

    def __init__(self, api_key: str, base_url: str = None,
                 model: str = "gpt-4o-mini"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._tiktoken_encoding = None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # 解析 tool_calls
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments  # JSON 字符串
                })

        tokens = 0
        if response.usage:
            tokens = response.usage.total_tokens

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "",
            tokens_used=tokens
        )

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        # 注意：OpenAI 流式也支持 tools，但此处简化为仅流式文本
        # 工具调用走非流式 chat() 路径
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def get_token_count(self, text: str) -> int:
        try:
            if self._tiktoken_encoding is None:
                import tiktoken
                self._tiktoken_encoding = tiktoken.encoding_for_model(self.model)
            return len(self._tiktoken_encoding.encode(text))
        except Exception:
            return len(text) // 3


# ====================================================================== #
#  Ollama Provider (本地模型)
# ====================================================================== #

class OllamaProvider(BaseLLMProvider):
    """Ollama 本地模型"""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        import httpx

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message", {})
        tool_calls = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append({
                    "id": f"call_{hash(func.get('name',''))}",
                    "name": func.get("name", ""),
                    "arguments": json.dumps(func.get("arguments", {}))
                })

        return LLMResponse(
            content=msg.get("content", ""),
            tool_calls=tool_calls,
            tokens_used=data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        )

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


# ====================================================================== #
#  DeepSeek Provider (兼容 OpenAI 协议)
# ====================================================================== #

class DeepSeekProvider(OpenAIProvider):
    """DeepSeek — 复用 OpenAI 协议"""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model=model
        )


# ====================================================================== #
#  LLM 工厂
# ====================================================================== #

# Provider 注册表
PROVIDERS: Dict[str, type] = {
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
    "deepseek": DeepSeekProvider,
}


class LLMFactory:
    """LLM 工厂 — 根据 provider 名称创建对应实例"""

    @staticmethod
    def create(
        provider: str,
        api_key: str = "",
        base_url: str = None,
        model: str = None
    ) -> BaseLLMProvider:
        cls = PROVIDERS.get(provider)
        if cls is None:
            raise ValueError(
                f"不支持的 LLM 提供商: {provider}。"
                f"可用: {', '.join(PROVIDERS.keys())}"
            )

        if provider == "ollama":
            return OllamaProvider(
                base_url=base_url or "http://localhost:11434",
                model=model or "qwen2.5:7b"
            )
        elif provider == "deepseek":
            return DeepSeekProvider(api_key=api_key, model=model or "deepseek-chat")
        else:
            return OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
                model=model or "gpt-4o-mini"
            )
