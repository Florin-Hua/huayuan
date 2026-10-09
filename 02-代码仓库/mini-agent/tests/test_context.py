"""阶段4单元测试：token 估算、工具输出截断、注入顺序与预算压缩。"""
from __future__ import annotations

from collections import deque
from pathlib import Path

from mini_agent.core.context import (
    ContextManager,
    TokenEstimator,
    truncate_tool_output,
)
from mini_agent.core.loop import AgentCore
from mini_agent.core.types import Message, ModelResponse, RunStatus, ToolCall, Usage
from mini_agent.tools.registry import ToolRegistry, tool


@tool
def long_echo(text: str) -> str:
    """原样返回输入文本，用于制造大工具输出。"""
    return text


class ScriptedAdapter:
    """按剧本返回模型输出，并记录最后一次看到的上下文。"""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = deque(responses)
        self.seen_messages: list[Message] = []

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.seen_messages = list(messages)
        if not self.responses:
            raise AssertionError("script has no more responses")
        return self.responses.popleft()


def tool_response(index: int, text: str) -> ModelResponse:
    return ModelResponse(
        content=f"第 {index} 轮，需要调用工具。",
        tool_calls=[
            ToolCall(id=f"call_{index}", name="long_echo", arguments={"text": text})
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def final_answer(answer: str) -> ModelResponse:
    return ModelResponse(
        content=answer,
        usage=Usage(prompt_tokens=20, completion_tokens=10),
    )


def build_history(rounds: int, text: str) -> list[Message]:
    messages = [
        Message(role="system", content="MiniAgent system prompt"),
        Message(role="user", content="处理长上下文任务"),
    ]
    for index in range(1, rounds + 1):
        messages.append(
            Message(
                role="assistant",
                content=f"第 {index} 轮",
                tool_calls=[
                    ToolCall(
                        id=f"call_{index}",
                        name="long_echo",
                        arguments={"text": text},
                    )
                ],
            )
        )
        messages.append(
            Message(
                role="tool",
                content=f"{{\"ok\": true, \"data\": \"{text}\"}}",
                tool_call_id=f"call_{index}",
                name="long_echo",
            )
        )
    return messages


def test_token_estimator_counts_text_and_message_overhead() -> None:
    estimator = TokenEstimator()
    text_tokens = estimator.estimate_text("hello world")
    message_tokens = estimator.estimate_message(
        Message(role="user", content="hello world")
    )
    assert text_tokens > 0
    assert message_tokens > text_tokens


def test_tool_output_truncation_keeps_head_and_tail() -> None:
    content = "A" * 1000 + "B" * 1000 + "C" * 500
    result = truncate_tool_output(content)

    assert len(result) < len(content)
    assert result.startswith("A" * 1000)
    assert result.endswith("C" * 500)
    assert "MiniAgent truncated" in result


def test_context_manager_truncates_payload_but_keeps_json_serializable() -> None:
    manager = ContextManager(token_budget=100)
    payload = "A" * 2500
    truncated, was_truncated = manager.truncate_payload(payload)

    assert was_truncated is True
    assert len(truncated) < len(payload)
    assert truncated.startswith("A" * 1000)


def test_context_injection_order_is_system_memory_summary_input_recent() -> None:
    estimator = TokenEstimator()
    text = "A" * 3000
    history = build_history(7, text)
    memory = [Message(role="system", content="用户偏好：回答保持简洁。")]
    summary = "早期任务与工具结果已压缩。"
    base_messages = [
        history[0],
        *memory,
        Message(role="system", content=f"[MiniAgent compressed history]\n{summary}"),
        history[1],
    ]
    budget = 2000
    manager = ContextManager(
        token_budget=budget,
        recent_turns=2,
        summarizer=lambda messages: summary,
    )

    prepared = manager.prepare(history, memory=memory)

    assert prepared.compression is not None
    assert prepared.compression.before_tokens > prepared.compression.after_tokens
    assert prepared.compression.summarized_rounds == 5
    assert [message.role for message in prepared.messages[:4]] == [
        "system",
        "system",
        "system",
        "user",
    ]
    assert "[MiniAgent compressed history]" in prepared.messages[2].content
    assert prepared.messages[3].content == "处理长上下文任务"
    assert len([message for message in prepared.messages if message.role == "tool"]) == 2


def test_long_task_completes_with_context_compression(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(long_echo)
    text = "A" * 6000
    adapter = ScriptedAdapter(
        [tool_response(index, text) for index in range(1, 13)]
        + [final_answer("长任务已完成，早期上下文已压缩。")]
    )
    manager = ContextManager(
        token_budget=4000,
        recent_turns=2,
        summarizer=lambda messages: "前五轮工具调用已完成，关键结果保留。",
    )
    core = AgentCore(
        adapter,
        max_iterations=13,
        registry=registry,
        context_manager=manager,
    )

    result = core.run("连续调用工具并完成长任务", tmp_path)

    assert result.status is RunStatus.SUCCESS
    assert result.compressions
    compression = result.compressions[-1]
    assert compression.before_tokens > compression.after_tokens
    assert any(
        message.role == "system"
        and "[MiniAgent compressed history]" in (message.content or "")
        for message in adapter.seen_messages
    )
    assert len([message for message in adapter.seen_messages if message.role == "tool"]) == 2
    assert any(step.state == "COMPRESS" for step in result.steps)


def test_context_budget_exceeded_when_compression_disabled(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(long_echo)
    text = "A" * 6000
    adapter = ScriptedAdapter(
        [
            tool_response(1, text),
            tool_response(2, text),
            final_answer("不应该到达这里。"),
        ]
    )
    manager = ContextManager(
        token_budget=1000,
        recent_turns=1,
        enabled=False,
    )
    core = AgentCore(
        adapter,
        max_iterations=3,
        registry=registry,
        context_manager=manager,
    )

    result = core.run("测试关闭压缩时的预算终止", tmp_path)

    assert result.status is RunStatus.CONTEXT_LIMIT
    assert result.iterations == 2
    assert "上下文预算终止" in (result.termination_reason or "")
    assert result.compressions == []
