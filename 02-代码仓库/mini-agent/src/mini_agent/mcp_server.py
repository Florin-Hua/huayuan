"""最小 MCP server：两个示例工具，stdio transport。"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="mini-agent-example",
    title="MiniAgent Example MCP Server",
    description="MiniAgent Stage 7 local MCP example server",
    version="0.1.0",
)


@server.tool(name="mcp_echo", description="Echo the input text to verify MCP forwarding.")
def mcp_echo(text: str) -> str:
    """MCP echo 示例工具。"""
    return text


@server.tool(name="mcp_add", description="Add two integers.")
def mcp_add(a: int, b: int) -> int:
    """MCP add 示例工具。"""
    return a + b


def main() -> None:
    """以 stdio transport 启动 MCP server。"""
    server.run("stdio")


if __name__ == "__main__":
    main()
