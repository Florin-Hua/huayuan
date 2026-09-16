"""Agent：MiniAgent 的用户入口门面。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from mini_agent.adapters.openai import OpenAIAdapter
from mini_agent.config import Settings
from mini_agent.core.loop import AgentCore, ChatAdapter
from mini_agent.core.types import AgentRun
from mini_agent.tools.builtin import BUILTIN_TOOLS
from mini_agent.tools.registry import ToolDefinition, ToolRegistry


class Agent:
    """组装模型适配器、核心循环、工具注册表和工具沙箱。"""

    def __init__(
        self,
        model: str | None = None,
        max_iterations: int | None = None,
        sandbox_dir: str | Path | None = None,
        tools: Iterable[ToolDefinition | object] | None = None,
        adapter: ChatAdapter | None = None,
        registry: ToolRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.model = model or self.settings.llm_model
        self.max_iterations = (
            self.settings.max_iterations
            if max_iterations is None
            else max_iterations
        )

        self.sandbox = Path(
            sandbox_dir if sandbox_dir is not None else self.settings.sandbox_dir
        ).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)

        if registry is not None:
            self.registry = registry
        else:
            self.registry = ToolRegistry(
                timeout_seconds=self.settings.tool_timeout_seconds
            )
            candidates = tools if tools is not None else BUILTIN_TOOLS
            for candidate in candidates:
                self.registry.register(candidate)

        self.adapter = adapter or OpenAIAdapter(self.settings, self.model)
        self.core = AgentCore(
            self.adapter,
            self.max_iterations,
            registry=self.registry,
        )

    def run(self, task: str) -> AgentRun:
        """执行一次任务并返回结构化运行结果。"""
        return self.core.run(task, self.sandbox)