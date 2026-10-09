"""AgentCore：手写 ReAct 循环、解析自愈、护栏与 Trace。"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Protocol

from mini_agent.core.context import ContextBudgetExceeded, ContextManager
from mini_agent.core.types import (
    AgentRun,
    Message,
    ModelResponse,
    ModelResponseError,
    RunStatus,
    StepLog,
)
from mini_agent.tools.registry import ToolRegistry, default_registry


class ChatAdapter(Protocol):
    """模型适配器的最小接口。"""

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        """调用一次模型并返回统一结构。"""
        ...


class Tracer(Protocol):
    """Trace 存储的最小接口，方便测试替换。"""

    def save_run(self, run: AgentRun) -> None:
        """保存一次 run 的汇总与步骤。"""
        ...


class AgentCore:
    """执行一次任务的核心循环。"""

    SYSTEM_PROMPT = (
        "你是 MiniAgent，一个使用工具完成任务的智能体。\n"
        "工作规则：\n"
        "1. 需要精确计算、读写文件或搜索时，必须调用工具，不要凭空猜测。\n"
        "2. 每轮可以调用多个工具；工具结果会以 tool 消息返回给你。\n"
        "3. 工具错误是正常反馈：读取错误后应调整路径或明确说明文件不存在，不要重复同一个失败调用。\n"
        "4. 已获得完成任务所需信息后，直接输出面向用户的最终答案，不要输出 JSON。\n"
        "5. 若无法完成任务，说明缺少什么信息以及你尝试过什么。\n"
        "6. 工具调用参数必须是 JSON object，工具 ID 和工具名不能为空。"
    )
    TOOL_NAME_RE = re.compile(r"[a-zA-Z0-9_-]{1,64}")

    def __init__(
        self,
        adapter: ChatAdapter,
        max_iterations: int,
        registry: ToolRegistry | None = None,
        context_manager: ContextManager | None = None,
        token_budget: int = 100_000,
        parse_retry_limit: int = 2,
        tracer: Tracer | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        if parse_retry_limit < 0:
            raise ValueError("parse_retry_limit must be >= 0")
        self.adapter = adapter
        self.max_iterations = max_iterations
        self.registry = registry or default_registry()
        self.context_manager = context_manager or ContextManager(
            summarizer=self._summarize_history
        )
        self.token_budget = token_budget
        self.parse_retry_limit = parse_retry_limit
        self.tracer = tracer

    def run(
        self,
        task: str,
        sandbox,
        memory: list[Message] | None = None,
    ) -> AgentRun:
        """运行任务并返回完整轨迹；Trace 保存失败不影响返回结果。"""
        run = AgentRun(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            status=RunStatus.MAX_ITER,
            iterations=0,
        )
        try:
            self._execute(run, task, sandbox, memory)
        finally:
            if self.tracer is not None:
                try:
                    self.tracer.save_run(run)
                except Exception as exc:
                    run.steps.append(
                        StepLog(
                            iteration=max(run.iterations, 1),
                            state="END",
                            error=f"Trace保存失败：{type(exc).__name__}: {exc}",
                            note="Trace 保存失败，但任务结果已返回",
                        )
                    )
        return run

    def _execute(
        self,
        run: AgentRun,
        task: str,
        sandbox,
        memory: list[Message] | None,
    ) -> None:
        history: list[Message] = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=task),
        ]
        previous_tool_signature: list[str] | None = None

        for iteration in range(1, self.max_iterations + 1):
            run.iterations = iteration
            prepared = self._prepare(run, history, memory, iteration)
            if prepared is None:
                return

            response, parse_error = self._chat_with_parse_retry(
                run, history, memory, iteration, prepared
            )
            if response is None:
                return

            history.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls or None,
                )
            )

            if run.usage.total > self.token_budget:
                reason = (
                    f"单 run token 预算超限：{run.usage.total} > {self.token_budget}"
                )
                run.steps.append(
                    StepLog(
                        iteration=iteration,
                        state="GUARD",
                        error=reason,
                        note="Token Guard",
                        tokens=run.usage.total,
                    )
                )
                self._finish(
                    run,
                    RunStatus.TOKEN_BUDGET,
                    "单次任务 token 预算超限，已强制终止。",
                    reason,
                )
                return

            if not response.tool_calls:
                answer = (response.content or "").strip()
                run.steps.append(
                    StepLog(
                        iteration=iteration,
                        state="END",
                        note="模型返回最终答案",
                        tokens=response.usage.total if response.usage else 0,
                    )
                )
                self._finish(run, RunStatus.SUCCESS, answer, "模型返回最终答案")
                return

            current_signature = [
                json.dumps(
                    [call.name, call.arguments],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for call in response.tool_calls
            ]
            if current_signature == previous_tool_signature:
                reason = "检测到连续两轮完全相同的工具与参数，判定为循环震荡"
                run.steps.append(
                    StepLog(
                        iteration=iteration,
                        state="GUARD",
                        tool_name=response.tool_calls[0].name,
                        tool_args=response.tool_calls[0].arguments,
                        error=reason,
                        note="Oscillation Guard",
                    )
                )
                self._finish(
                    run,
                    RunStatus.OSCILLATION,
                    "检测到重复工具调用循环，已强制终止。",
                    reason,
                )
                return
            previous_tool_signature = current_signature

            for call in response.tool_calls:
                result = self.registry.execute(call, sandbox)
                run.steps.append(
                    StepLog(
                        iteration=iteration,
                        state="ACT",
                        tool_name=call.name,
                        tool_args=call.arguments,
                        ok=result.ok,
                        error=result.error,
                        note="工具执行完成",
                        duration_ms=result.duration_ms,
                    )
                )

                data, data_truncated = self.context_manager.truncate_payload(
                    result.data
                )
                error, _ = self.context_manager.truncate_payload(result.error)
                history.append(
                    Message(
                        role="tool",
                        content=json.dumps(
                            {
                                "ok": result.ok,
                                "data": data,
                                "error": error,
                                "truncated": data_truncated,
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
                    ok=all(
                        step.ok
                        for step in run.steps
                        if step.state == "ACT" and step.iteration == iteration
                    ),
                    note="工具结果已回喂模型",
                )
            )

        reason = f"达到最大迭代次数 {self.max_iterations}，未能得到最终答案"
        run.steps.append(
            StepLog(
                iteration=self.max_iterations,
                state="END",
                error=reason,
                note="终止",
                tokens=run.usage.total,
            )
        )
        self._finish(
            run,
            RunStatus.MAX_ITER,
            "已达到最大迭代次数，任务未完成。",
            reason,
        )

    def _prepare(
        self,
        run: AgentRun,
        history: list[Message],
        memory: list[Message] | None,
        iteration: int,
    ):
        try:
            prepared = self.context_manager.prepare(history, memory=memory)
        except ContextBudgetExceeded as exc:
            reason = f"上下文预算终止：{exc}"
            run.steps.append(
                StepLog(
                    iteration=iteration,
                    state="END",
                    error=reason,
                    note="上下文超过预算",
                )
            )
            self._finish(
                run,
                RunStatus.CONTEXT_LIMIT,
                "上下文超过预算，任务未能继续。",
                reason,
            )
            return None

        if prepared.compression is not None:
            stats = prepared.compression
            run.compressions.append(stats)
            run.steps.append(
                StepLog(
                    iteration=iteration,
                    state="COMPRESS",
                    note=(
                        f"上下文压缩：{stats.before_tokens} -> "
                        f"{stats.after_tokens} tokens；"
                        f"摘要 {stats.summarized_rounds} 轮"
                    ),
                )
            )
        return prepared

    def _chat_with_parse_retry(
        self,
        run: AgentRun,
        history: list[Message],
        memory: list[Message] | None,
        iteration: int,
        prepared,
    ) -> tuple[ModelResponse | None, str | None]:
        """调用模型；解析失败时最多回喂错误重试两次。"""
        while True:
            retry_note = (
                "调用模型"
                if run.parse_retries == 0
                else f"解析失败后第 {run.parse_retries} 次重试"
            )
            run.steps.append(
                StepLog(iteration=iteration, state="THINK", note=retry_note)
            )

            try:
                response = self.adapter.chat(
                    prepared.messages, self.registry.schemas
                )
            except ModelResponseError as exc:
                response = None
                parse_error = f"模型输出格式错误：{exc.detail}"
            except Exception as exc:
                reason = f"模型调用失败：{type(exc).__name__}: {exc}"
                run.steps.append(
                    StepLog(
                        iteration=iteration,
                        state="END",
                        error=reason,
                        note="终止",
                    )
                )
                self._finish(
                    run,
                    RunStatus.PARSE_ERROR,
                    "模型调用失败，任务未能继续。",
                    reason,
                )
                return None, reason
            else:
                run.usage.add(response.usage)
                parse_error = self._format_response_errors(response)

            if parse_error is None:
                return response, None

            if run.parse_retries >= self.parse_retry_limit:
                reason = (
                    f"模型输出解析失败，已重试 {run.parse_retries} 次仍失败："
                    f"{parse_error}"
                )
                run.steps.append(
                    StepLog(
                        iteration=iteration,
                        state="END",
                        error=reason,
                        note="解析自愈失败",
                        tokens=run.usage.total,
                    )
                )
                self._finish(
                    run,
                    RunStatus.PARSE_ERROR,
                    "模型输出格式持续非法，任务未能继续。",
                    reason,
                )
                return None, reason

            run.parse_retries += 1
            run.steps.append(
                StepLog(
                    iteration=iteration,
                    state="RETRY",
                    ok=False,
                    error=parse_error,
                    note=f"错误回喂模型，第 {run.parse_retries} 次解析重试",
                )
            )
            # 不把非法 tool_calls 写入正式历史，避免下一轮携带坏结构；
            # 只保留文本并追加一条明确的纠错指令。
            history.append(
                Message(
                    role="assistant",
                    content=(response.content if response is not None else None) or "",
                )
            )
            history.append(
                Message(
                    role="user",
                    content=(
                        "上一轮模型输出非法："
                        f"{parse_error}。"
                        "请重新输出：要么给最终文本答案，要么给合法 tool_calls；"
                        "工具 arguments 必须是 JSON object。"
                    ),
                )
            )
            prepared = self._prepare(run, history, memory, iteration)
            if prepared is None:
                return None, "上下文超过预算"

    def _format_response_errors(self, response: ModelResponse) -> str | None:
        errors: list[str] = []
        if not (response.content or "").strip() and not response.tool_calls:
            errors.append("content 与 tool_calls 均为空")

        seen_ids: set[str] = set()
        for call in response.tool_calls:
            if not call.id:
                errors.append(f"tool call id 为空：{call.name}")
            elif call.id in seen_ids:
                errors.append(f"tool call id 重复：{call.id}")
            else:
                seen_ids.add(call.id)

            if not call.name or not self.TOOL_NAME_RE.fullmatch(call.name):
                errors.append(f"tool name 非法：{call.name!r}")
            if "_raw" in call.arguments:
                errors.append(f"工具 {call.name} 的 arguments 不是 JSON object")

        if errors:
            return "；".join(errors)
        return None

    def _summarize_history(self, messages: list[Message]) -> str:
        """用当前模型适配器生成滚动摘要；摘要调用不计入主任务 usage。"""
        transcript = "\n".join(
            f"{message.role}: {message.content or ''}" for message in messages
        )
        prompt = [
            Message(
                role="system",
                content=(
                    "你是上下文压缩器。请把早期 ReAct 历史压缩成不超过 300 字的摘要，"
                    "保留任务目标、已尝试工具、关键结果、错误与仍未完成事项。"
                ),
            ),
            Message(role="user", content=transcript),
        ]
        response = self.adapter.chat(prompt, [])
        return (response.content or "").strip() or "早期上下文已压缩，但摘要为空。"

    @staticmethod
    def _finish(
        run: AgentRun,
        status: RunStatus,
        answer: str | None,
        reason: str | None,
    ) -> None:
        run.status = status
        run.answer = answer
        run.termination_reason = reason
        run.finished_at = datetime.now()
