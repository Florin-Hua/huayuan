"""Agent：MiniAgent 的用户入口门面。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from mini_agent.adapters.factory import ChatAdapter, create_adapter
from mini_agent.config import Settings
from mini_agent.core.context import ContextManager
from mini_agent.core.loop import AgentCore
from mini_agent.core.types import AgentRun, Message
from mini_agent.memory import (
    MemoryExtractor,
    MemoryStore,
    create_save_memory_tool,
    memory_to_message,
)
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
        context_manager: ContextManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.memory_store = (
            MemoryStore(
                self.settings.memory_db_path,
                dimension=self.settings.memory_dimension,
            )
            if self.settings.memory_enabled
            else None
        )
        self.model = model or self.settings.resolved_model
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

        if self.memory_store is not None and self.registry.get("save_memory") is None:
            self.registry.register(create_save_memory_tool(self.memory_store))

        self.adapter = adapter or create_adapter(self.settings, self.model)
        self.core = AgentCore(
            self.adapter,
            self.max_iterations,
            registry=self.registry,
            context_manager=context_manager or ContextManager(
                token_budget=self.settings.context_token_budget,
                recent_turns=self.settings.context_recent_turns,
                enabled=self.settings.context_compression_enabled,
                tool_output_max_chars=self.settings.context_tool_output_max_chars,
                tool_output_head_chars=self.settings.context_tool_output_head_chars,
                tool_output_tail_chars=self.settings.context_tool_output_tail_chars,
            ),
        )

    def run(self, task: str) -> AgentRun:
        """执行一次任务，并按配置检索/抽取长期记忆。"""
        memory: list[Message] = []
        if self.memory_store is not None:
            matches = self.memory_store.search(
                task, limit=self.settings.memory_top_k
            )
            message = memory_to_message(matches)
            if message is not None:
                memory.append(message)

        result = self.core.run(task, self.sandbox, memory=memory)

        if self.memory_store is not None and result.status.value == "success":
            try:
                result.memories_saved = MemoryExtractor(
                    self.adapter, self.memory_store
                ).extract(task, result.answer or "", result.run_id)
            except Exception:
                # 记忆抽取失败不影响主任务结果；阶段6 Trace 再记录失败原因。
                result.memories_saved = []
        return result

    def close(self) -> None:
        """释放数据库连接。"""
        if self.memory_store is not None:
            self.memory_store.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
