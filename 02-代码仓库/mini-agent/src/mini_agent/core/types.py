"""核心数据结构（第0阶段设计的阶段1精简落地）。

所有模块只通过这里定义的数据结构通信（依赖规则见 01-阶段进程/第0阶段-总体设计.md）。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """模型发起的一次工具调用请求（未执行）。"""

    id: str = Field(description="调用唯一ID，用于把 tool 消息关联回 assistant 消息")
    name: str = Field(description="工具名；幻觉调用会在执行层收到结构化错误")
    arguments: dict[str, Any] = Field(default_factory=dict, description="实参")


class Message(BaseModel):
    """框架内部唯一的消息表示（模型无关）。"""

    role: Role
    content: str | None = Field(default=None, description="文本内容；tool 消息为结果 JSON")
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="仅 assistant 消息：本轮请求的工具调用"
    )
    tool_call_id: str | None = Field(
        default=None, description="仅 tool 消息：对应的 ToolCall.id"
    )
    name: str | None = Field(default=None, description="仅 tool 消息：工具名")


class Usage(BaseModel):
    """token 用量统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage | None") -> None:
        if other is None:
            return
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelResponse(BaseModel):
    """一次模型输出的统一表示（最终答案 or 工具调用）。"""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None


class ToolResult(BaseModel):
    """一次工具执行的统一结果（含失败，错误必须回喂模型）。"""

    call: ToolCall
    ok: bool
    data: Any | None = None
    error: str | None = None
    duration_ms: int = 0


class RunStatus(str, Enum):
    """运行终态：阶段1支持三种，其余终态在阶段6补全。"""

    SUCCESS = "success"
    MAX_ITER = "max_iterations"
    PARSE_ERROR = "parse_error"


class StepLog(BaseModel):
    """一步执行的日志（阶段6将升级为 Tracer 落库）。"""

    iteration: int
    state: Literal["THINK", "ACT", "OBSERVE", "END"]
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    ok: bool | None = None
    error: str | None = None
    note: str | None = None


class AgentRun(BaseModel):
    """run() 的返回值：一次任务的完整结果与执行轨迹。"""

    run_id: str
    task: str
    status: RunStatus
    answer: str | None = Field(
        default=None, description="SUCCESS 时为最终答案，异常终止时为善后说明"
    )
    iterations: int = 0
    steps: list[StepLog] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    termination_reason: str | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = None