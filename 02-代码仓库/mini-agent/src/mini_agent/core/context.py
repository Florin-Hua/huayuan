"""ContextManager：上下文 token 预算、滚动摘要与工具输出截断。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import tiktoken

from mini_agent.core.types import CompressionStats, Message


class ContextBudgetExceeded(RuntimeError):
    """上下文超过预算且无法继续压缩。"""

    def __init__(self, before_tokens: int, after_tokens: int | None = None) -> None:
        after = f"，压缩后仍为 {after_tokens}" if after_tokens is not None else ""
        super().__init__(f"上下文 token 超过预算：{before_tokens}{after}")
        self.before_tokens = before_tokens
        self.after_tokens = after_tokens


_SHARED_ENCODING: tiktoken.Encoding | None = None


class TokenEstimator:
    """基于 tiktoken cl100k_base 的 token 估算器。"""

    def __init__(self) -> None:
        global _SHARED_ENCODING
        if _SHARED_ENCODING is None:
            _SHARED_ENCODING = tiktoken.get_encoding("cl100k_base")
        self._encoding = _SHARED_ENCODING

    def estimate_text(self, text: str | None) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def estimate_message(self, message: Message) -> int:
        parts: list[str] = [message.role]
        if message.content:
            parts.append(message.content)
        if message.name:
            parts.append(message.name)
        if message.tool_calls:
            parts.append(json.dumps([call.model_dump() for call in message.tool_calls], ensure_ascii=False))
        # 消息结构本身有少量开销，按 4 token 估算。
        return self.estimate_text(" ".join(parts)) + 4

    def estimate_messages(self, messages: Iterable[Message]) -> int:
        return sum(self.estimate_message(message) for message in messages)


@dataclass(slots=True)
class PreparedContext:
    """一次上下文准备结果。"""

    messages: list[Message]
    compression: CompressionStats | None = None


Summarizer = Callable[[list[Message]], str]


def truncate_tool_output(
    content: str,
    max_chars: int = 2000,
    head_chars: int = 1000,
    tail_chars: int = 500,
) -> str:
    """超过长度时保留头尾并插入省略提示。"""
    if len(content) <= max_chars:
        return content
    omitted = len(content) - head_chars - tail_chars
    marker = f"\n...[MiniAgent truncated {omitted} characters]...\n"
    return content[:head_chars] + marker + content[-tail_chars:]


class ContextManager:
    """维护上下文预算，并把较早轮次替换为摘要。"""

    def __init__(
        self,
        token_budget: int = 8192,
        recent_turns: int = 5,
        enabled: bool = True,
        summarizer: Summarizer | None = None,
        tool_output_max_chars: int = 2000,
        tool_output_head_chars: int = 1000,
        tool_output_tail_chars: int = 500,
        estimator: TokenEstimator | None = None,
    ) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        if recent_turns < 1:
            raise ValueError("recent_turns must be >= 1")
        self.token_budget = token_budget
        self.recent_turns = recent_turns
        self.enabled = enabled
        self.summarizer = summarizer
        self.tool_output_max_chars = tool_output_max_chars
        self.tool_output_head_chars = tool_output_head_chars
        self.tool_output_tail_chars = tool_output_tail_chars
        self.estimator = estimator or TokenEstimator()

    def truncate_payload(self, payload: Any) -> tuple[Any, bool]:
        """截断工具返回值，并保持外层 JSON 可解析。"""
        if payload is None:
            return None, False
        if isinstance(payload, str):
            truncated = truncate_tool_output(
                payload,
                self.tool_output_max_chars,
                self.tool_output_head_chars,
                self.tool_output_tail_chars,
            )
            return truncated, truncated != payload
        serialized = json.dumps(payload, ensure_ascii=False)
        if len(serialized) <= self.tool_output_max_chars:
            return payload, False
        truncated = truncate_tool_output(
            serialized,
            self.tool_output_max_chars,
            self.tool_output_head_chars,
            self.tool_output_tail_chars,
        )
        return truncated, True

    def prepare(
        self,
        history: list[Message],
        memory: list[Message] | None = None,
    ) -> PreparedContext:
        """按 system -> memory -> 摘要 -> 当前输入 -> 近期原文的顺序构造上下文。"""
        memory = memory or []
        before_tokens = self.estimator.estimate_messages(history + memory)
        if before_tokens <= self.token_budget:
            return PreparedContext(list(history))

        if not self.enabled:
            raise ContextBudgetExceeded(before_tokens)

        if len(history) < 2 or history[0].role != "system" or history[1].role != "user":
            raise ValueError("history must start with system and user messages")

        system = history[0]
        current_input = history[1]
        rounds = self._split_rounds(history[2:])
        if len(rounds) <= self.recent_turns:
            raise ContextBudgetExceeded(before_tokens)

        split = len(rounds) - self.recent_turns
        older_rounds = rounds[:split]
        recent_rounds = rounds[split:]

        summary = self._summarize([message for round_ in older_rounds for message in round_])
        summary_message = Message(
            role="system",
            content=f"[MiniAgent compressed history]\n{summary}",
        )
        messages = [system, *memory, summary_message, current_input]
        messages.extend(message for round_ in recent_rounds for message in round_)

        after_tokens = self.estimator.estimate_messages(messages)
        stats = CompressionStats(
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summarized_rounds=len(older_rounds),
            summarized_messages=sum(len(round_) for round_ in older_rounds),
            enabled=True,
        )
        if after_tokens > self.token_budget:
            raise ContextBudgetExceeded(before_tokens, after_tokens)
        return PreparedContext(messages, stats)

    def _split_rounds(self, messages: list[Message]) -> list[list[Message]]:
        rounds: list[list[Message]] = []
        current: list[Message] = []
        for message in messages:
            if message.role == "assistant" and current:
                rounds.append(current)
                current = []
            current.append(message)
        if current:
            rounds.append(current)
        return rounds

    def _summarize(self, messages: list[Message]) -> str:
        if self.summarizer is None:
            counted = len(messages)
            return f"已完成 {counted} 条早期消息，详细原文因上下文预算被压缩。"
        return self.summarizer(messages)
