"""Agent：MiniAgent 的用户入口门面。"""
from __future__ import annotations

from pathlib import Path

from mini_agent.adapters.openai import OpenAIAdapter
from mini_agent.config import Settings
from mini_agent.core.loop import AgentCore
from mini_agent.core.types import AgentRun


class Agent:
    """组装模型适配器、核心循环和工具沙箱。"""

    def __init__(
        self,
        model: str | None = None,
        max_iterations: int | None = None,
        sandbox_dir: str | Path | None = None,
        adapter: object | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.model = model or self.settings.llm_model
        self.max_iterations = max_iterations or self.settings.max_iterations

        self.sandbox = Path(
            sandbox_dir if sandbox_dir is not None else self.settings.sandbox_dir
        ).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)

        self.adapter = adapter or OpenAIAdapter(self.settings, self.model)
        self.core = AgentCore(self.adapter, self.max_iterations)

    def run(self, task: str) -> AgentRun:
        """执行一次任务并返回结构化运行结果。"""
        return self.core.run(task, self.sandbox)