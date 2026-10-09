# MiniAgent 面试笔记

> 本文件按阶段逐步补充。第8阶段会扩展成 10 个完整面试问答；本节先记录阶段7新增的 MCP 安全设计。

## MCP 工具与本地工具在权限 / 安全上的差异

### 1. 信任边界不同

本地工具运行在 MiniAgent 同一个 Python 进程或本进程创建的线程池里，代码来源、依赖版本和异常行为都由仓库直接控制。MCP tool 是一个独立进程，甚至可能是远程服务；MiniAgent 只信任它通过 `tools/list` 暴露的声明和 schema，不能假设它的实现是安全的。

因此 MiniAgent 的处理是：MCP tool 只被适配成 `ToolDefinition`，调用统一进入 `ToolRegistry` 的超时、参数校验和错误归一化，但不会自动继承本地文件沙箱。

### 2. 文件沙箱不自动延伸

本地 `read_text_file` / `write_text_file` 接收 `sandbox` 参数，框架能在执行前拦截 `../` 路径逃逸。MCP tool 的文件访问发生在 server 进程内，MiniAgent 看不到它内部路径，也无法强制它遵守客户端沙箱。

如果 MCP server 需要文件能力，正确做法是在 server 侧实现自己的根目录、路径校验和权限模型；客户端只能把该 server 视为有能力边界的外部服务，并在部署配置中限制它。

### 3. 权限模型不同

本地工具的权限 = 当前进程权限，风险主要来自代码缺陷和沙箱逃逸。MCP 工具的权限 = server 进程权限 + 协议能力协商结果，还可能涉及 OAuth、token、远程资源访问。把一个高权限 MCP server 接进 Agent，等于把模型的可调用能力扩展到该 server 的全部授权范围。

MiniAgent 当前实现保持克制：只做 stdio 子进程、只接入 `tools/list` / `tools/call`，没有自动启用 sampling、roots、resource 写入等高级能力。这是降低攻击面的有意选择。

### 4. 输入输出契约不同

本地工具用 Pydantic 模型生成 schema，类型和必填约束由源码保证。MCP tool 的 schema 来自外部 server，客户端必须先转换再验证。MiniAgent 用 JSON schema 生成动态 Pydantic 参数模型，调用前先校验，避免把畸形参数直接送到外部进程。

输出侧同样不能信任：MCP 可能返回文本、结构化内容或 error result。MiniAgent 会解析 JSON 文本、优先使用 `structured_content`，并把 `is_error` 转成 `McpToolError`，再由 `ToolRegistry` 归一化为 `ToolResult.ok=false` 回喂模型。

### 5. 生命周期和故障域不同

本地工具调用结束即释放，失败大多是单次异常。MCP 连接是长生命周期：子进程可能崩溃、stdio 队列可能阻塞、server 可能在启动后改变行为。MiniAgent 为每个 MCP server 建立后台 asyncio 线程，同步工具调用通过 `run_coroutine_threadsafe` 转发，并设置启动 / 请求超时和显式 close。

这也带来一个 trade-off：线程模型更简单，但并发接入多个 server 时会占用更多资源；后续如果框架整体异步化，可以改成共享事件循环。

### 6. 供应链风险不同

本地工具依赖由 `pyproject.toml` 锁定，代码可审计。MCP server 可能由第三方分发，命令行启动意味着引入它的依赖、环境变量和更新策略。生产环境应在配置层白名单 server 命令，不允许模型或用户任务动态指定任意命令。

MiniAgent 当前 `--mcp-server` 是用户显式传入的 CLI 配置，模型不能自行启动 MCP server，这是安全边界的关键。
