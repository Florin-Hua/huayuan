"""配置：从 .env 读取（密钥只放 .env，禁止入库）。"""
from __future__ import annotations

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MiniAgent 配置。

    环境变量名与 .env.example 一致。MODEL_NAME 是新名称，LLM_MODEL 为兼容旧配置保留。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 模型与 provider
    model_provider: Literal["openai", "anthropic"] = "openai"
    model_name: str | None = None
    llm_model: str = "gpt-4o-mini"

    # OpenAI 兼容：GPT / DeepSeek / Qwen 等均可通过 base_url 切换
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # Anthropic 原生接口
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_key: str = ""
    anthropic_max_tokens: int = 1024

    # Agent
    max_iterations: int = 10
    sandbox_dir: str = "data"

    # Context compression (Stage 4)
    context_compression_enabled: bool = True
    context_token_budget: int = 8192
    context_recent_turns: int = 5
    context_tool_output_max_chars: int = 2000
    context_tool_output_head_chars: int = 1000
    context_tool_output_tail_chars: int = 500

    # Tools
    tool_timeout_seconds: float = 30.0

    # Memory (Stage 5)
    memory_enabled: bool = True
    memory_db_path: str = "data/memory.sqlite3"
    memory_top_k: int = 3
    memory_dimension: int = 256

    # Robustness & Trace (Stage 6)
    token_budget: int = 100_000
    parse_retry_limit: int = 2
    trace_enabled: bool = True
    trace_db_path: str = "data/trace.sqlite3"

    @field_validator("model_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def resolved_model(self) -> str:
        """优先使用 MODEL_NAME，兼容旧 LLM_MODEL。"""
        return self.model_name or self.llm_model


def load_settings() -> Settings:
    """加载配置（当前工作目录下的 .env 优先）。"""
    return Settings()