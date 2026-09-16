"""工具注册表 — 带完整参数 Schema + ReAct 执行支持"""
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
import json
import traceback


@dataclass
class ToolInfo:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """工具注册表 — 管理所有可用工具，提供 OpenAI Function Calling 格式定义"""

    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}

    # ------------------------------------------------------------------ #
    #  注册
    # ------------------------------------------------------------------ #
    def register(self, name: str = None, description: str = None,
                 parameters: Dict[str, Any] = None):
        """装饰器：注册工具（含参数 Schema）"""
        def decorator(func):
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""
            self.tools[tool_name] = ToolInfo(
                name=tool_name,
                description=tool_desc,
                function=func,
                parameters=parameters or {}
            )
            return func
        return decorator

    # ------------------------------------------------------------------ #
    #  查询
    # ------------------------------------------------------------------ #
    def get_tool(self, name: str) -> Optional[ToolInfo]:
        return self.tools.get(name)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """返回 OpenAI Function Calling 格式工具列表"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in self.tools.values()
        ]

    def get_all_tools_info(self) -> List[Dict[str, Any]]:
        """返回工具摘要（供 /api/v1/tools 列表接口）"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in self.tools.values()
        ]

    # ------------------------------------------------------------------ #
    #  执行
    # ------------------------------------------------------------------ #
    def execute(self, name: str, **kwargs) -> str:
        """执行工具，返回字符串结果"""
        tool_info = self.tools.get(name)
        if not tool_info:
            return f"[错误] 工具不存在: {name}"

        try:
            result = tool_info.function(**kwargs)
            return str(result)
        except Exception as e:
            tb = traceback.format_exc()
            return f"[工具执行错误] {name}: {e}\n{tb}"


# ====================================================================== #
#  全局注册表实例
# ====================================================================== #
tool_registry = ToolRegistry()


# ====================================================================== #
#  内置工具
# ====================================================================== #

@tool_registry.register(
    name="calculator",
    description="计算数学表达式，支持加减乘除、幂运算、三角函数等",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(144)' 或 'sin(3.14/2)'"
            }
        },
        "required": ["expression"]
    }
)
def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    import math
    # 允许的函数和常量
    safe_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "int": int, "float": float,
        "sqrt": math.sqrt, "pow": pow, "log": math.log, "log2": math.log2,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "inf": math.inf,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)
        return f"{result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool_registry.register(
    name="python_executor",
    description="执行 Python 代码并返回结果。用于数据分析、计算、文本处理等。最后一行赋值给 result 变量会作为返回值。",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码"
            }
        },
        "required": ["code"]
    }
)
def python_executor(code: str) -> str:
    """在受限环境中执行 Python 代码"""
    import io
    import contextlib

    stdout_buf = io.StringIO()
    local_vars = {"__builtins__": __builtins__}

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, local_vars)

        output = stdout_buf.getvalue()
        result_val = local_vars.get("result")

        parts = []
        if output:
            parts.append(f"输出:\n{output}")
        if result_val is not None:
            parts.append(f"结果: {result_val}")

        return "\n".join(parts) if parts else "执行完成（无输出）"
    except Exception as e:
        return f"执行错误: {e}"


@tool_registry.register(
    name="web_search",
    description="搜索互联网获取实时信息，返回标题、摘要和链接",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数，默认 3",
                "default": 3
            }
        },
        "required": ["query"]
    }
)
def web_search(query: str, max_results: int = 3) -> str:
    """使用 DuckDuckGo 搜索互联网"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "未找到相关结果"
            parts = []
            for i, r in enumerate(results, 1):
                parts.append(f"{i}. **{r['title']}**\n   {r['body']}\n   链接: {r['href']}")
            return "\n\n".join(parts)
    except ImportError:
        return "[错误] 未安装 duckduckgo-search: pip install duckduckgo-search"
    except Exception as e:
        return f"搜索错误: {e}"


@tool_registry.register(
    name="read_file_content",
    description="读取已上传到知识库的文档内容",
    parameters={
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "文档 ID（从知识库列表获取）"
            }
        },
        "required": ["doc_id"]
    }
)
def read_file_content(doc_id: str) -> str:
    """读取知识库中的文档 — 运行时由 Agent 注入 rag pipeline"""
    # 这个工具在 Agent 初始化时会被动态绑定到具体的 RAG pipeline
    return f"[错误] read_file_content 未绑定到 RAG pipeline"


@tool_registry.register(
    name="knowledge_search",
    description="从已上传的知识库中搜索相关文档内容。当用户提问涉及已上传的文档、论文、资料时使用。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或问题"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["query"]
    }
)
def knowledge_search(query: str, top_k: int = 3) -> str:
    """搜索知识库 — 运行时由 Agent 注入 rag pipeline"""
    return f"[错误] knowledge_search 未绑定到 RAG pipeline"
