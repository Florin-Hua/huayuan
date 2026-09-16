"""MiniAgent 内置工具：calculator、文件读写、mock_web_search。"""
from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini_agent.tools.registry import ToolDefinition, tool

_MAX_TEXT_CHARS = 8000
_MAX_POWER = 1000

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


class CalculatorInput(BaseModel):
    expression: str = Field(description="要计算的数字表达式，例如 23*47+11")


class ReadTextFileInput(BaseModel):
    path: str = Field(description="沙箱内文件路径，例如 notes.txt")


class WriteTextFileInput(BaseModel):
    path: str = Field(description="沙箱内目标文件路径")
    content: str = Field(description="要写入的 UTF-8 文本内容")


class MockWebSearchInput(BaseModel):
    query: str = Field(description="搜索关键词")


@tool(CalculatorInput)
def calculator(expression: str) -> int | float:
    """安全计算一个数字表达式，支持 + - * / // % ** 和一元正负号。"""
    if not expression.strip():
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


def _resolve_sandbox_path(path: str, sandbox: Path) -> Path:
    requested = Path(path)
    target = requested if requested.is_absolute() else sandbox / requested
    target = target.resolve()
    sandbox_resolved = sandbox.resolve()
    if not target.is_relative_to(sandbox_resolved):
        raise ValueError(f"path escapes sandbox: {requested}")
    return target


@tool(ReadTextFileInput)
def read_text_file(path: str, sandbox: Path) -> dict[str, Any]:
    """读取沙箱内一个 UTF-8 文本文件，超过 8000 字符自动截断。"""
    target = _resolve_sandbox_path(path, sandbox)
    if not target.is_file():
        raise FileNotFoundError(f"file not found: {path}")

    content = target.read_text(encoding="utf-8")
    truncated = len(content) > _MAX_TEXT_CHARS
    return {
        "path": path,
        "content": content[:_MAX_TEXT_CHARS],
        "truncated": truncated,
    }


@tool(WriteTextFileInput)
def write_text_file(path: str, content: str, sandbox: Path) -> dict[str, Any]:
    """把 UTF-8 文本写入沙箱内文件，用于保存任务结果。"""
    target = _resolve_sandbox_path(path, sandbox)
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "path": path,
        "bytes_written": len(content.encode("utf-8")),
        "overwritten": existed,
    }


@tool(MockWebSearchInput)
def mock_web_search(query: str) -> dict[str, Any]:
    """返回固定 fixture 的模拟搜索结果，不访问网络。"""
    return {
        "query": query,
        "results": [
            {
                "title": "MiniAgent mock result 1",
                "url": "https://example.com/mock-1",
                "snippet": f"Mock search snippet about {query}.",
            },
            {
                "title": "MiniAgent mock result 2",
                "url": "https://example.com/mock-2",
                "snippet": "This result is deterministic and safe for tests.",
            },
        ],
    }


BUILTIN_TOOLS: list[ToolDefinition] = [
    calculator,
    read_text_file,
    write_text_file,
    mock_web_search,
]

TOOL_SCHEMAS = [definition.schema for definition in BUILTIN_TOOLS]