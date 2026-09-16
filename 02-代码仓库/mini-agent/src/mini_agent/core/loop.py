"""AgentCore：手写 ReAct 循环（阶段1 不依赖任何 agent 框架）。

状态机：THINK -> ACT -> OBSERVE -> ... -> END。
工具执行失败不会终止任务，而是把结构化错误回喂给模型，让它自行调整。
"""
from __future__ import annotations

import json
import uuid
from typing import Protocol

from mini_agent.core.types import (
    AgentRun,
    Message,
    ModelResponse,
    RunStatus,
    StepLog,
    Usage,
)
from mini_agent.tools.builtin import TOOL_SCHEMAS, execute_tool


class ChatAdapter(Protocol):
    """模型适配器的最小接口；阶段3会扩展更多模型供应商。"""

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        """调用一次模型并返回统一结构。"""
        ...


class AgentCore:
    """执行一次任务的核心循环。"""

    SYSTEM_PROMPT = (
        "你是 MiniAgent，一个使用工具完成任务的智能体。\n"
        "工作规则：\n"
        "1. 需要精确计算或读取文件时，必须调用工具，不要凭空猜测。\n"
        "2. 每轮可以调用多个工具；工具结果会以 tool 消息返回给你。\n"
        "3. 工具错误是正常反馈：读取错误后应调整路径或明确说明文件不存在，不要重复同一个失败调用。\n"
        "4. 已获得完成任务所需信息后，直接输出面向用户的最终答案，不要输出 JSON。\n"
        "5. 若无法完成任务，说明缺少什么信息以及你尝试过什么。"
    )

    def __init__(self, adapter: ChatAdapter, max_iterations: int) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.adapter = adapter
        self.max_iterations = max_iterations

    def run(self, task: str, sandbox) -> AgentRun:
        """运行任务并返回完整轨迹（不抛业务异常）。"""
        run = AgentRun(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            status=RunStatus.MAX_ITER,
            iterations=0,
        )
        messages: list[Message] = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=task),
        ]

        for iteration in range(1, self.max_iterations + 1):
            run.iterations = iteration
            run.steps.append(StepLog(iteration=iteration, state="THINK", note="调用模型"))

            # THINK：模型决定最终回答还是发起工具调用
            try:
                response = self.adapter.chat(messages, TOOL_SCHEMAS)
            except Exception as exc:
                reason = f"模型调用失败：{type(exc).__name__}: {exc}"
                run.steps.append(
                    StepLog(iteration=iteration, state="END", error=reason, note="终止")
                )
                self._finish(run, RunStatus.PARSE_ERROR, "模型调用失败，任务未能继续。", reason)
                return run

            run.usage.add(response.usage)
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls or None,
                )
            )

            if not response.tool_calls:
                answer = (response.content or "").strip() or "（模型未返回文本答案）"
                run.steps.append(
                    StepLog(
                        iteration=iteration,
                        state="END",
                        note="模型返回最终答案",
                    )
                )
                self._finish(run, RunStatus.SUCCESS, answer, "模型返回最终答案")
                return run

            # ACT：执行本轮全部工具调用
            for call in response.tool_calls:
                result = execute_tool(call, sandbox)
                run.steps.append(
                    StepLog(
                        iteration=iteration,
                        state="ACT",
                        tool_name=call.name,
                        tool_args=call.arguments,
                        ok=result.ok,
                        error=result.error,
                        note=f"耗时 {result.duration_ms}ms",
                    )
                )

                # OBSERVE：把成功数据和失败错误都回喂模型
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(
                            {
                                "ok": result.ok,
                                "data": result.data,
                                "error": result.error,
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

            run.steps.append(
                StepLog(
                    iteration=iteration,
                    state="OBSERVE",
                    ok=all(step.ok for step in run.steps if step.state == "ACT" and step.iteration == iteration),
                    note="工具结果已回喂模型",
                )
            )
        else:
            reason = f"达到最大迭代次数 {self.max_iterations}，未能得到最终答案"
            run.steps.append(
                StepLog(
                    iteration=self.max_iterations,
                    state="END",
                    error=reason,
                    note="终止",
                )
            )
            self._finish(run, RunStatus.MAX_ITER, "已达到最大迭代次数，任务未完成。", reason)
            return run

        return run

    @staticmethod
    def _finish(
        run: AgentRun,
        status: RunStatus,
        answer: str | None,
        reason: str | None,
    ) -> None:
        from datetime import datetime

        run.status = status
        run.answer = answer
        run.termination_reason = reason
        run.finished_at = datetime.now()