"""阶段1 内置工具：calculator 与 read_text_file。

这里刻意不用 eval/exec，文件读取也被限制在沙箱内，先建立最小安全边界。
"""
from __future__ import annotations

import ast
import operator
import time
from pathlib import Path
from typing import Any

from mini_agent.core.types import ToolCall, ToolResult

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MAX_TEXT_CHARS = 8000
_MAX_POWER = 1000


def calculator(expression: str) -> int | float:
    """用 AST 白名单安全计算整数/浮点表达式，不执行任意代码。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc.msg}") from exc

    _validate_ast(tree)
    return _evaluate_node(tree.body)


def _validate_ast(node: ast.AST) -> None:
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)

    if isinstance(node, ast.Expression):
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or isinstance(node.value, (int, float)):
            return
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return
    if isinstance(node, (ast.operator, ast.unaryop)) and (
        type(node) in _BIN_OPS or type(node) in _UNARY_OPS
    ):
        return
    raise ValueError(f"disallowed AST node: {type(node).__name__}")


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPS.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported unary operator")
        return operation(_evaluate_node(node.operand))
    if isinstance(node, ast.BinOp):
        operation = _BIN_OPS.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported binary operator")
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if operation is operator.pow and abs(right) > _MAX_POWER:
            raise ValueError(f"exponent exceeds limit {_MAX_POWER}")
        if operation is operator.truediv and right == 0:
            raise ValueError("division by zero")
        return operation(left, right)
    raise ValueError(f"disallowed AST node: {type(node).__name__}")


def read_text_file(path: str, sandbox: Path) -> dict[str, Any]:
    """读取沙箱内文本文件；路径逃逸、不存在、编码错误都会返回异常。"""
    requested = Path(path)
    target = requested if requested.is_absolute() else sandbox / requested
    target = target.resolve()
    sandbox_resolved = sandbox.resolve()

    if not target.is_relative_to(sandbox_resolved):
        raise ValueError(f"path escapes sandbox: {requested}")

    if not target.is_file():
        raise FileNotFoundError(f"file not found: {requested}")

    content = target.read_text(encoding="utf-8")
    truncated = len(content) > _MAX_TEXT_CHARS
    return {
        "path": str(requested),
        "content": content[:_MAX_TEXT_CHARS],
        "truncated": truncated,
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "安全计算一个数字表达式，支持 + - * / // % ** 和一元正负号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的表达式，例如 23*47+11",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_text_file",
            "description": "读取沙箱内一个 UTF-8 文本文件；path 是相对沙箱的路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径，例如 notes.txt 或 data/notes.txt",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


def execute_tool(call: ToolCall, sandbox: Path) -> ToolResult:
    """统一工具入口：未知工具和执行错误都变成结构化结果回喂模型。"""
    started = time.perf_counter()
    ok = False
    data: Any = None
    error: str | None = None

    try:
        if call.name == "calculator":
            data = calculator(**call.arguments)
            ok = True
        elif call.name == "read_text_file":
            data = read_text_file(**call.arguments, sandbox=sandbox)
            ok = True
        else:
            error = f"unknown tool: {call.name}"
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