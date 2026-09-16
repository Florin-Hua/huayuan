"""ToolRegistry：@tool 注册、schema 生成、统一调度与错误归一化。"""
from __future__ import annotations

import inspect
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError, create_model

from mini_agent.core.types import ToolCall, ToolResult

_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass
class ToolDefinition:
    """一个已注册工具的完整定义。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    args_model: type[BaseModel]
    timeout_seconds: float = 30.0
    requires_sandbox: bool = False

    @property
    def schema(self) -> dict[str, Any]:
        """OpenAI function calling 格式的工具 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """保留直接调用能力，便于独立测试工具本体。"""
        return self.func(*args, **kwargs)


def _schema_without_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """清理 Pydantic 生成的 title，让 schema 简洁且对模型友好。"""
    if isinstance(schema, dict):
        return {
            key: _schema_without_titles(value)
            for key, value in schema.items()
            if key != "title"
        }
    if isinstance(schema, list):
        return [_schema_without_titles(item) for item in schema]
    return schema


def _model_from_signature(func: Callable[..., Any]) -> type[BaseModel]:
    """没有显式参数模型时，从函数签名自动生成 Pydantic 模型。"""
    signature = inspect.signature(func)
    fields: dict[str, Any] = {}
    type_hints = getattr(func, "__annotations__", {})

    for name, parameter in signature.parameters.items():
        if name == "sandbox":
            continue
        annotation = type_hints.get(name, Any)
        if parameter.default is inspect.Parameter.empty:
            default = ...
        else:
            default = parameter.default
        fields[name] = (
            annotation,
            Field(default, description=f"{name} parameter"),
        )
    return create_model(f"{func.__name__}Input", **fields)


def tool(
    _func_or_model: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_seconds: float = 30.0,
) -> Callable[[Callable[..., Any]], ToolDefinition] | ToolDefinition:
    """把普通函数注册成 MiniAgent 标准工具。

    用法一：@tool
    用法二：@tool(CalculatorInput)
    显式参数模型优先；未提供时从函数签名自动生成。
    """

    def build(func: Callable[..., Any]) -> ToolDefinition:
        nonlocal _func_or_model

        explicit_model = None
        if _func_or_model is not None and isinstance(_func_or_model, type):
            if not issubclass(_func_or_model, BaseModel):
                raise TypeError("args model must be a pydantic.BaseModel subclass")
            explicit_model = _func_or_model

        args_model = explicit_model or _model_from_signature(func)
        tool_name = name or func.__name__
        if not _TOOL_NAME_RE.fullmatch(tool_name):
            raise ValueError(
                "tool name must match [a-zA-Z0-9_-] and be 1-64 characters"
            )

        doc = inspect.getdoc(func) or ""
        tool_description = description or next(
            (line.strip() for line in doc.splitlines() if line.strip()),
            f"Tool {tool_name}",
        )
        signature = inspect.signature(func)
        requires_sandbox = "sandbox" in signature.parameters

        return ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=_schema_without_titles(args_model.model_json_schema()),
            func=func,
            args_model=args_model,
            timeout_seconds=timeout_seconds,
            requires_sandbox=requires_sandbox,
        )

    if callable(_func_or_model) and not isinstance(_func_or_model, type):
        return build(_func_or_model)
    return build


class ToolRegistry:
    """注册、查找、执行工具；所有失败都归一化为 ToolResult。"""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self.timeout_seconds = timeout_seconds
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition | Callable[..., Any]) -> ToolDefinition:
        """注册一个 ToolDefinition，或自动包装普通函数。"""
        if not isinstance(definition, ToolDefinition):
            definition = tool()(definition)
        if definition.name in self._tools:
            raise ValueError(f"duplicate tool: {definition.name}")
        self._tools[definition.name] = definition
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema for name in sorted(self._tools)]

    def execute(
        self,
        call: ToolCall,
        sandbox: Path | None = None,
    ) -> ToolResult:
        """执行一次工具调用：幻觉、参数错误、异常、超时都不抛出。"""
        import time

        started = time.perf_counter()
        ok = False
        data: Any = None
        error: str | None = None
        definition = self._tools.get(call.name)

        if definition is None:
            error = f"unknown tool: {call.name}"
        else:
            try:
                validated = definition.args_model.model_validate(call.arguments)
                arguments = validated.model_dump()
                if definition.requires_sandbox and sandbox is None:
                    raise ValueError("tool requires a sandbox, but sandbox is None")

                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    future: Future = executor.submit(
                        definition.func,
                        **arguments,
                        **({"sandbox": sandbox} if definition.requires_sandbox else {}),
                    )
                    data = future.result(timeout=definition.timeout_seconds)
                    ok = True
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
            except ValidationError as exc:
                error = f"ToolInputError: {exc.error_count()} invalid argument(s): {exc}"
            except TimeoutError as exc:
                error = (
                    f"ToolTimeoutError: {call.name} exceeded "
                    f"{definition.timeout_seconds}s"
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        duration_ms = round((time.perf_counter() - started) * 1000)
        return ToolResult(
            call=call,
            ok=ok,
            data=data,
            error=error,
            duration_ms=duration_ms,
        )


_default_registry: ToolRegistry | None = None


def default_registry() -> ToolRegistry:
    """内置工具注册表（懒加载，避免循环导入）。"""
    global _default_registry
    if _default_registry is None:
        from mini_agent.tools.builtin import BUILTIN_TOOLS

        registry = ToolRegistry()
        for definition in BUILTIN_TOOLS:
            registry.register(definition)
        _default_registry = registry
    return _default_registry


def execute_tool(
    call: ToolCall,
    sandbox: Path,
    registry: ToolRegistry | None = None,
) -> ToolResult:
    """阶段1兼容入口；新代码应直接使用 ToolRegistry.execute。"""
    return (registry or default_registry()).execute(call, sandbox)