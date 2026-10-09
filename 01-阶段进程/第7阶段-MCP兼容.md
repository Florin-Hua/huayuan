# 第7阶段 · MCP 兼容（可选加分）

> 完成日期：2026-10-09
> 状态：MCP Python SDK 最小 server、同步适配层、ToolRegistry 集成与 pytest 验收完成；`mini run` 的真实模型调用仍受当前网络 / 密钥条件限制，未伪造结果。

## 1. 阶段目标

1. 用 MCP Python SDK 写一个最小 stdio server，暴露 2 个示例工具；
2. 实现 `McpToolAdapter`，把 MCP tools 转成 MiniAgent 标准 `ToolDefinition`；
3. 做 schema 转换、调用转发与错误归一化；
4. 让模型在 Agent 循环中像调用内置工具一样调用 MCP 工具；
5. 用 pytest 覆盖工具发现、调用转发和错误处理；
6. 在交付物中记录 MCP 工具与本地工具的安全差异。

## 2. 交付文件

| 文件 | 作用 |
|---|---|
| `src/mini_agent/mcp_server.py` | MCP Python SDK 2.x 最小 stdio server |
| `src/mini_agent/mcp_tools.py` | `McpClient`、`McpToolAdapter`、schema / result / error 转换 |
| `src/mini_agent/core/agent.py` | Agent 门面接入 MCP server |
| `src/mini_agent/cli.py` | `mini run --mcp-server "..."` |
| `src/mini_agent/config.py` / `.env.example` | MCP 配置 |
| `tests/test_mcp.py` | 5 个 MCP 测试 |
| `03-交付物/interview_notes.md` | MCP 与本地工具的安全差异 |
| `pyproject.toml` | 新增 `mcp>=2.3` 依赖与 `mini-mcp-server` 命令 |

## 3. MCP server 设计

使用 MCP Python SDK 2.x：

```python
server = MCPServer(name="mini-agent-example")

@server.tool(name="mcp_echo")
def mcp_echo(text: str) -> str:
    return text

@server.tool(name="mcp_add")
def mcp_add(a: int, b: int) -> int:
    return a + b

server.run("stdio")
```

启动方式：

```powershell
.venv\Scripts\python.exe -m mini_agent.mcp_server
# 或安装包后：
mini-mcp-server
```

## 4. McpToolAdapter 设计

### 4.1 同步 / 异步桥接

MCP Python SDK 的 `ClientSession` 是 async，而 MiniAgent 的 `ToolRegistry.execute()` 是同步并在工具线程池中执行。为了不重写整个框架，第7阶段采用：

```text
后台线程 + 独立 asyncio event loop
    └─ stdio_client + ClientSession 长连接
同步工具函数
    └─ asyncio.run_coroutine_threadsafe(...) 转发 tools/list / tools/call
```

优点：`AgentCore`、`ToolRegistry`、CLI 全部保持同步模型，改动小。  
代价：每个 MCP server 对应一个后台线程；如果未来接入大量 server，应改成共享事件循环或整体 async 化。

### 4.2 schema 转换

MCP `Tool.input_schema`（JSON Schema）转换为：

1. 去掉 `title`，减少模型噪声；
2. 根据 `properties` / `required` 动态生成 Pydantic 参数模型；
3. 包装成 `ToolDefinition.parameters`；
4. `ToolDefinition.schema` 继续输出 OpenAI function calling 格式。

因此模型看到的 MCP 工具与本地工具格式一致。

### 4.3 调用与结果转换

```text
ToolRegistry.execute(call)
  → 参数 Pydantic 校验
  → McpToolAdapter.call_tool(name, arguments)
  → ClientSession.call_tool(...)
  → structured_content / content JSON 解析
  → ToolResult
```

错误处理：

- MCP `is_error=true` → `McpToolError`；
- `ToolRegistry` 捕获后归一化为 `ToolResult.ok=false`；
- 错误文本回喂模型，与本地工具失败路径一致。

### 4.4 Agent / CLI 集成

CLI：

```powershell
mini run "用 MCP 计算 19 + 23" --mcp-server ".venv\Scripts\python.exe -m mini_agent.mcp_server"
```

`.env`：

```env
MCP_SERVER_COMMAND=python -m mini_agent.mcp_server
MCP_TOOL_TIMEOUT_SECONDS=30
```

安全约束：`--mcp-server` 只能由用户显式配置，模型不能自行启动任意 MCP server。

## 5. 验收结果

### 单元测试

```powershell
.venv\Scripts\python.exe -m pytest
```

结果：

```text
59 passed in 13.48s
```

新增第7阶段测试覆盖：

1. 真实 stdio 子进程：发现 `mcp_echo` / `mcp_add`；
2. schema 转换为 OpenAI function calling 格式；
3. 通过 `ToolRegistry` 调用 `mcp_add(20,22)`，返回 `42`；
4. Agent 循环中模型调用 `mcp_add`，与内置工具路径一致；
5. MCP 文本 JSON 结果解析；
6. MCP error result 归一化为 `ToolResult.ok=false`；
7. CLI parser 支持 `--mcp-server`。

### 真实模型验收

`mini run --mcp-server ...` 的真实模型链路尚未执行：

- OpenAI key 存在，但当前 `api.openai.com` DNS / TCP 异常；
- Anthropic 网络可达，但没有有效 `ANTHROPIC_API_KEY`。

已完成的替代验收是：使用 scripted model adapter + 真实 MCP stdio 子进程，验证 Agent 从工具 schema、模型 tool call、MCP 转发到最终回答的完整链路。该结果不冒充真实供应商调用。

## 6. 关键决策

| 决策 | 理由 |
|---|---|
| 使用 MCP Python SDK 2.x 而非手写 JSON-RPC | 保证协议初始化、分帧、版本协商符合生态标准 |
| 后台线程包装 async ClientSession | 保持 MiniAgent 核心 API 简单，避免阶段7被迫全框架 async 化 |
| 动态生成 Pydantic 参数模型 | 调用前校验外部 server schema，避免畸形参数直接出进程 |
| MCP server 只启用 tools 能力 | 不自动暴露 resources / sampling / roots，降低权限面 |
| server 命令只能由用户配置 | 防止模型或任务动态启动任意可执行文件 |

## 7. 代码规模

- 运行时代码：约 2,369 行（预算 3,000 行内）
- 测试代码：约 1,232 行
- 当前版本：`0.7.0`

## 8. 已知局限

- 每个 MCP server 使用一个后台线程，未做多 server 连接池；
- schema 只映射常见基础类型，复杂 `anyOf` / nested schema 会退化为 `Any`；
- 客户端没有 OAuth / 远程 HTTP transport，当前只支持 stdio；
- MCP server 侧文件权限需要 server 自己实现，客户端沙箱不能延伸到外部进程。

下一阶段为第8阶段：评测与交付。
