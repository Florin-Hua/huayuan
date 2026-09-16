"""配置：从 .env 读取（密钥只放 .env，禁止入库）。"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MiniAgent 配置。

    环境变量名与 .env.example 一致，可直接复用已有凭证配置。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 模型
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # Agent
    max_iterations: int = 10
    sandbox_dir: str = "data"

    # Tools
    tool_timeout_seconds: float = 30.0


def load_settings() -> Settings:
    """加载配置（当前工作目录下的 .env 优先）。"""
    return Settings()