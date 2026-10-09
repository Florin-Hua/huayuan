# MiniAgent 面试笔记

> 用途：把项目经历转成可回答面试官追问的知识点。每题按“实际设计 → trade-off → 真实踩坑”展开。
> 口径提醒：阶段8使用确定性 scripted adapter 做“框架机制评测”，结果不能冒充商业模型真实准确率；真实 OpenAI 调用因网络超时未完成，Anthropic 缺少有效 key，这些都如实记录。

## 1. 为什么要自研，而不用 LangChain / LangGraph？

### 我的实际设计

项目目标是做一个约 3000 行以内的 MiniAgent，把 Agent 的关键机制拆成可解释模块：`AgentCore` 负责 THINK→ACT→OBSERVE→REFLECT 状态机，`ToolRegistry` 管工具契约，`ModelAdapter` 隔离供应商，`ContextManager` 管上下文，`MemoryStore` / `TraceStore` 管持久状态。所有核心行为都有单元测试，不依赖 Agent 编排框架。

### trade-off

自研放弃了成熟生态、持久化图执行、复杂 human-in-the-loop 和现成集成，换来三点收益：第一，每一行代码都能解释设计动机；第二，失败路径不会被框架黑盒吞掉；第三，可以在固定预算内控制复杂度。团队生产项目里，我会优先评估 LangGraph/CrewAI/AutoGen，而不是为了自研而自研。

### 真实踩坑

最初我也想直接在既有实现上加功能，但发现工具契约、循环状态、配置和测试耦合在一起，很难讲清“哪些是我写的”。后来把旧实现归档到只读参考目录，从零重建。短期内进度变慢，但测试和阶段文档变得非常清晰。

## 2. ReAct 循环具体怎么实现？

### 我的实际设计

`AgentCore.run()` 为每次任务创建 `AgentRun`，维护 `run_id`、状态、迭代数、usage、steps 和终止原因。每轮先用 `ContextManager.prepare()` 构造模型输入，再调用 adapter 得到统一 `ModelResponse`。如果有 `tool_calls`，就交给 `ToolRegistry.execute()`；工具结果格式化成 tool message 回填 history，进入下一轮。如果模型返回纯文本且没有 tool call，则置为 success 并生成最终答案。

### trade-off

我采用同步循环，没有做 async 流式执行。优点是状态转移和测试都很简单；缺点是不能边生成边展示 token，后续多 MCP server 并发时也需要事件循环或线程桥接。最大迭代默认 10 次，防止 Agent 无限思考。

### 真实踩坑

解析失败时不能把非法 tool call 直接写入正式历史，否则下一轮会继续携带坏结构。我把非法输出转成明确的纠错 user message，并限制解析重试次数，默认 2 次。这样既能自愈，又不会被模型持续输出坏 JSON 拖垮。

## 3. 工具系统如何保证参数安全？

### 我的实际设计

工具用 `@tool` 装饰器注册。参数来源有两种：显式 Pydantic 模型或函数签名自动生成。`ToolDefinition` 保存 JSON Schema、执行函数、args model、超时和是否需要 sandbox。执行前用 Pydantic 校验参数；计算器不是 `eval`，而是 AST 白名单；文件工具通过 `Path.resolve()` 检查目标必须位于 sandbox 内；工具名也用正则限制在 1-64 个安全字符。

### trade-off

这层校验会增加代码和一次模型 schema 序列化，但能拦截幻觉工具、畸形 JSON、路径逃逸、非法表达式和过长指数。工具执行放在线程池中做超时控制，但 Python 线程无法被强制杀死，超时后底层调用可能继续运行。

### 真实踩坑

`eval("999 ** 10000")` 会造成巨大的计算消耗。最终实现 AST 遍历，只允许数值常量和白名单运算符，并限制指数。阶段8里 SB028 专门注入这个失败，strong 策略收到错误后改算 `999 ** 2`，验证了错误回喂的价值。

## 4. 多模型适配如何隔离 provider 差异？

### 我的实际设计

框架内部只有自己的 `Message / ToolCall / ModelResponse / Usage`。OpenAI adapter 转换 Chat Completions 的 `tool_calls` 与 JSON arguments；Anthropic adapter 处理独立 system 参数、`tool_use` blocks、`tool_result` 必须紧跟 assistant tool_use 的消息顺序，以及 `input_tokens/output_tokens` 到统一 Usage 的映射。`create_adapter()` 根据 `MODEL_PROVIDER` 选择实现。

### trade-off

统一抽象减少核心循环对 SDK 的依赖，但新供应商能力会被压缩到最小公共接口，流式、多模态、缓存等能力暂未暴露。为了可测试性，adapter 构造函数允许注入 fake client 或录制 fixture。

### 真实踩坑

Anthropic 的 tool result 不是独立 role，而是 user message 里的 content block，还要求 `tool_use_id` 正确关联。如果内部消息没有保存 `tool_call_id`，转换时就会断链。因此统一消息结构必须包含 provider 间转换所需的字段，而不是只存 text。

## 5. 上下文压缩的触发与代价是什么？

### 我的实际设计

`ContextManager` 用 tiktoken cl100k_base 估算消息 token。超过预算时，保留最近若干轮原文，更早历史交给当前 adapter 生成不超过 300 字的滚动摘要，再按 system → memory → summary → 当前输入 → recent turns 注入。工具输出超过 2000 字符时保留头 1000、尾 500 和省略标记，避免长文件撑爆上下文。

### trade-off

压缩能救长任务，但摘要调用本身消耗 token，也可能丢失细节。评测中 strong 策略开启压缩后平均 token 从 325.0 增至 417.5，但 4 条长上下文任务从 0/4 变成 4/4；关闭压缩时均以 `context_limit` 受控终止。这个数字清楚说明它是准确性换成本的机制。

### 真实踩坑

不能只截断 JSON 字符串，否则 tool message 会变成不可解析文本。实现时先序列化判断长度，截断后标记为 truncated 字符串，保证下一轮模型收到的是明确文本而不是坏 JSON。

## 6. 记忆系统为什么选 SQLite + sqlite-vec？

### 我的实际设计

`MemoryStore` 用 SQLite 保存内容、source run id 和创建时间，用 sqlite-vec 保存向量并按距离检索 top-K。默认向量器是本地 HashingEmbedder，任务开始注入相关记忆，任务结束后由 `MemoryExtractor` 判断是否保存长期事实，模型也可调用 `save_memory` 工具主动保存。记忆和 Trace 分库保存，生命周期不同。

### trade-off

SQLite 单文件部署简单、离线可测、便于审计；sqlite-vec 提供轻量向量检索。默认哈希向量语义能力弱于真实 embedding，多进程并发也受 SQLite 限制。Embedder 保留接口，后续可替换成语义模型或外部向量库。

### 真实踩坑

评测里 SB021 保存“项目总结必须中文且简短”，SB022 依赖这个偏好。记忆开启时 30/30 成功；关闭后 SB022 失败，成功率降到 96.67%。样本只有 1 条，不能说泛化收益很大，但足以证明检索和注入链路有效。

## 7. 工具错误如何自愈？

### 我的实际设计

参数错误、执行异常、超时、未知工具都会被 `ToolRegistry` 归一化为 `ToolResult(ok=false)`，错误信息作为 observation 回喂模型。system prompt 明确说明“工具错误是正常反馈，读取后调整，不要重复失败调用”。评测设置 8 条失败恢复任务，包括读错文件、除零、路径逃逸、幻觉工具和指数超限，strong 策略恢复率 100%。

### trade-off

直接抛异常实现最简单，但模型没有机会修正；全部回喂错误又可能鼓励重复调用。因此需要配合最大迭代和震荡检测。另一个边界是线程池超时无法真正杀死线程，只能让调用方先拿到错误。

### 真实踩坑

SB026 先写 `../escape.txt` 被沙箱拒绝，再写 `safe.txt` 成功。这个例子说明错误文本必须包含足够语义，只返回“工具失败”模型无法知道应该换路径。

## 8. Guard 如何防无限循环和 token 爆炸？

### 我的实际设计

运行前校验 `max_iterations >= 1`、token budget `>= 1`、parse retry limit 非负。运行中每轮累计 usage，超过默认 100,000 返回预算终止；达到最大迭代默认 10 次返回 max iterations；连续两轮完全相同的工具调用和参数判定为震荡。上下文超限且无法压缩时返回 context limit。所有终止都有状态、答案和 reason。

### trade-off

严格 Guard 可能提前终止一个看似慢但其实有进展的任务；宽松 Guard 又会烧钱甚至死循环。我选择“宁可受控失败，也不无限等待”，并用 Trace 记录原因，方便事后调参。

### 真实踩坑

只按“工具名相同”判震荡会误伤多次合法重试，因此签名包含完整 arguments。但如果模型每轮加一个无效字符，仍然可以绕过简单签名检测；生产上需要更强的进度评估或人工介入。

## 9. Trace 如何帮助调试？

### 我的实际设计

`TraceStore` 用 SQLite 保存 run 汇总和逐步日志，包括状态、工具名、参数、结果是否成功、token、耗时、错误、note 等字段。CLI 提供 `mini trace <run_id>`。Trace 保存失败不会影响用户拿到结果，会在 run steps 里追加一条 END 记录说明存储失败。

### trade-off

逐步落库会带来 I/O 和少量 schema 维护成本，但它把“模型为什么这么做”变成可查询事实，比打印日志更利于复盘。敏感参数需要脱敏和访问控制，本项目把 Trace 放在 `data/` 下的本地运行数据，不入 git。

### 真实踩坑

阶段8最早统计的 model calls 和 token 是 adapter 累计值，跨任务会不断变大。后来改为每个任务先记录 before，再用差值计算单任务增量；token 口径也补上了上下文摘要和记忆抽取调用。这个问题正是靠检查逐任务 records 发现的。

## 10. MCP 工具与本地工具的安全差异是什么？

### 我的实际设计

本地工具在 MiniAgent 进程或其线程池中执行，代码和依赖由仓库控制，文件工具继承统一 sandbox。MCP 工具来自独立 stdio server，MiniAgent 只信任 `tools/list` 的声明，把 JSON schema 动态转换为 Pydantic 参数模型，调用结果与 error result 统一转成 `ToolResult`。MCP server 命令只能由用户在 CLI/env 中显式配置，模型不能自行启动。

### trade-off

MCP 带来生态兼容，但信任边界扩大：server 进程权限、供应链、协议能力和生命周期都由外部决定。本地文件沙箱无法约束 server 内部路径；如果 server 需要文件能力，必须在 server 侧实现自己的根目录和权限模型。当前实现只做 stdio、tools/list、tools/call，不自动启用 sampling、roots、resource 写入等高级能力。

### 真实踩坑

MCP SDK 的 `ClientSession` 是 async，而 MiniAgent 的核心 API 是同步。为了让阶段7不被迫整体 async 化，我用后台线程和独立事件循环，通过 `run_coroutine_threadsafe` 转发调用。代价是每个 server 一个线程；真实 stdio 子进程测试通过，但如果接入大量 server，应改成共享事件循环或整体异步化。
