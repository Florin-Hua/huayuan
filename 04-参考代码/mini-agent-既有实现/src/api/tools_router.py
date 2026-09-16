"""工具管理 API — 列表 / 手动执行"""
from fastapi import APIRouter, HTTPException
from src.tools.registry import tool_registry

router = APIRouter()


@router.get("/tools")
async def list_tools():
    """列出所有可用工具"""
    return {
        "tools": tool_registry.get_all_tools_info(),
        "total": len(tool_registry.tools)
    }


@router.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, params: dict = None):
    """手动执行工具（调试用）"""
    if params is None:
        params = {}

    tool = tool_registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"工具不存在: {tool_name}")

    result = tool_registry.execute(tool_name, **params)
    return {
        "tool": tool_name,
        "params": params,
        "result": result
    }
