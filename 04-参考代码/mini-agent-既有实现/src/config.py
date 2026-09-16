"""Mini-Agent 配置管理"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # ===== LLM 配置 =====
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # ===== Embedding 配置 =====
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ===== RAG 配置 =====
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # ===== 记忆配置 =====
    short_term_max_turns: int = 20
    summary_interval: int = 10
    memory_search_top_k: int = 3

    # ===== 服务配置 =====
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = ["*"]

    # ===== Redis =====
    redis_url: str = "redis://localhost:6379/0"

    # ===== 数据路径 =====
    vector_db_path: str = "./data/vector_db"
    db_path: str = "./data/sqlite/agent.db"
    upload_dir: str = "./data/uploads"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
