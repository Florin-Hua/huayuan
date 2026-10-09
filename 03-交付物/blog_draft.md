# 为什么我不用框架手写了一个 Agent

大四实习季，我最想做的不是又一个“调 API 的聊天壳”，而是一个能真正使用工具、有记忆、能失败恢复的个人 Agent。市面上的框架很多，LangChain 能拼链，LangGraph 能画图，CrewAI 能组织角色，AutoGen 能让多个智能体对话。它们都很好，但我一直有一个疑问：把 Agent 的核心逻辑交给框架之后，我还能说清楚工具调用、上下文、记忆和失败恢复到底是怎么发生的吗？

于是我给自己定了一个约束：不用重型 Agent 框架，从零写一个 MiniAgent，核心代码控制在约 3000 行以内，所有关键机制必须能被测试。这个选择不是为了证明框架不好，而是为了把抽象背后的工程代价亲手摸一遍。

## 第一决策：先定义最小闭环

我没有从“个人助理”这个大词开始，而是从最小 ReAct 闭环开始：模型思考，发起工具调用，框架执行工具，把 observation 回喂，模型再决定继续还是输出最终答案。实现上只有四个状态：THINK、ACT、OBSERVE、REFLECT。看似简单，但真正的难点在边界：模型输出空内容怎么办？tool arguments 不是 JSON object 怎么办？工具不存在怎么办？模型一直重复调用怎么办？

这些问题的答案决定了 Agent 是 demo 还是工程。我让 run 返回 `AgentRun` 而不是字符串，里面有状态、步骤、token、终止原因和完整轨迹。字符串只能演示，结构化结果才能测试、评测和复盘。

## 工具系统：先写契约，再谈能力

第二阶段我把工具系统抽成 `ToolRegistry`。一个工具必须有名称、描述、JSON Schema、参数模型、执行函数和超时。用 `@tool` 装饰器可以注册普通函数；参数可以显式声明 Pydantic 模型，也可以从签名生成。模型看到的 schema 和执行前校验使用同一份定义，避免“给模型看的类型”和“运行时类型”漂移。

安全是工具系统的第一原则。计算器没有用 `eval`，而是解析 AST，只允许数字常量和白名单运算符，并限制指数；文件工具要求所有路径 resolve 后必须位于 sandbox 内；未知工具、参数错误、执行异常和超时都被归一化为 `ToolResult(ok=false)`。错误不是异常栈，而是给模型看的 observation。阶段8评测里，8 条失败恢复任务全部在收到错误后调整策略，包括文件不存在、除零、路径逃逸和幻觉工具。

这个设计的代价是繁琐。每加一个工具都要认真声明参数和错误，但换来的是可审计。后来接 MCP 时，这套契约救了我：外部工具 schema 只要能转换成同一个 `ToolDefinition`，核心循环就不用知道它是本地函数还是外部 server。

## 多模型适配：把供应商差异压到边界

第三阶段接了 OpenAI 兼容接口和 Anthropic 原生 Messages API。框架内部只有自己的 `Message / ToolCall / ModelResponse / Usage`。OpenAI 的 `tool_calls`、JSON arguments、`tool_call_id`，Anthropic 的独立 system、`tool_use` block、放在 user content 里的 `tool_result`，都在 adapter 内完成转换。

这件事让我第一次真切理解“抽象不是少写代码，而是选公共语义”。如果内部消息结构只保存文本，Anthropic 的 tool result 关联就会断；如果核心循环直接使用某一家 SDK 的对象，换供应商时所有测试都会跟着变脆。我的选择是 adapter 层复杂，核心层稳定。

## 上下文：压缩是有成本的

Agent 最容易被忽略的问题是上下文无限膨胀。读几个长文件，搜索几次，模型下一轮的输入就变成历史垃圾堆。第四阶段实现 `ContextManager`：用 tiktoken 估算 token，超过预算时保留最近 5 轮原文，更早历史交给当前模型压缩成摘要；工具输出超过 2000 字符时保留头尾并打上省略标记。

评测给了很明确的数字：4 条长上下文任务，压缩开启时 4/4 成功；关闭后全部 `context_limit` 终止。但压缩开启后平均 token 从 325.0 增至 417.5，因为摘要本身也是一次模型调用。压缩不是魔法优化，而是“保留更多细节”和“控制上下文”的 trade-off。

## 记忆：先解决可复现，再追求聪明

第五阶段做长期记忆。我选 SQLite + sqlite-vec：内容表保存事实和来源 run，向量表保存 embedding。默认用本地 HashingEmbedder，不依赖网络。任务开始检索 top-3 相关记忆并注入 system prompt；任务结束后一次模型调用判断是否有长期事实；模型也可以主动调用 `save_memory`。

默认哈希向量当然不如语义 embedding 聪明，但它让测试离线可复现。评测里 SB021 保存了“项目总结必须中文且简短”，SB022 依赖这条偏好。记忆开启时 30/30 成功，关闭后 SB022 失败。样本很小，我不会夸大成泛化结论，但链路证明了。

## 健壮性与 Trace：让失败变成数据

第六阶段加了解析自愈和三类护栏。模型输出非法时，不把坏 tool call 写进正式历史，而是追加纠错指令，最多重试 2 次。最大迭代默认 10 次，token 预算默认 100,000，连续两轮相同工具和参数判定为震荡。Trace 用 SQLite 保存 run 汇总和逐步记录，CLI 可以查 `mini trace <run_id>`。

这一步最大的收获是调试方式变了。以前我盯 print，现在我看状态、工具参数、返回、token 和耗时。评测初期我发现 model calls 和 token 是累计值，跨任务越来越大，就是通过逐任务 records 检查出来的，后来改成单任务差值口径。

## MCP：兼容生态，不交出安全边界

第七阶段接 MCP。MCP Python SDK 的 server 和 client 是 async，我的核心 API 是同步，于是用后台线程和独立事件循环桥接。`McpToolAdapter` 把 `tools/list` 转成标准工具 schema，把 `tools/call` 结果和 error result 归一化后交给 ToolRegistry。

这里的安全边界必须清醒：本地工具在仓库和 sandbox 约束内，MCP tool 是独立进程，客户端沙箱管不到它内部访问了什么文件。因此 server 命令只能由用户显式传入，模型不能自己启动；MCP JSON schema 会动态生成 Pydantic 模型先校验；当前只实现 stdio、tools/list 和 tools/call，不自动打开 sampling、roots 或 resource 写入。

## 评测：先证明框架链路，再谈模型能力

最后阶段我做了 30 条自建任务，覆盖计算、文件、搜索、长上下文、记忆和失败恢复。为了避免烧 API 和网络不稳定，模型侧用确定性 scripted adapter：strong 按任务脚本调用工具，baseline 不调用工具。评测的是框架机制，不是 GPT 或 Claude 的真实准确率。

结果是：strong + 压缩 + 记忆 30/30；关闭压缩降到 86.67%；关闭记忆降到 96.67%；8 条失败恢复任务恢复率 100%；baseline 只有 6.67%。这组数字的价值在于消融，不在炫分。GAIA 是 gated dataset，我没有授权 token，所以只留下授权访问清单和选择脚本，不伪造指标。

## 反思：什么时候应该用框架

写完之后，我更尊重框架，也更知道自己要什么。如果明天要做团队生产系统，我会直接评估 LangGraph 的 StateGraph、checkpoint 和 interrupt；要做多角色协作，CrewAI 更快；要研究多智能体对话，AutoGen 更合适；要一个开箱即用的个人助理运行时，OpenClaw 这类产品比我的小框架完整得多。

但如果你和我一样处在学习阶段，想搞懂 Agent 的每一步为什么这样设计，自研一次非常值得。工具 schema、错误回喂、上下文预算、记忆注入、MCP 信任边界，这些词在文档里都很好读，只有亲手写过，才知道每个抽象背后都是一次取舍。

MiniAgent 还很小，没有流式输出、checkpoint 恢复、真实语义 embedding 和并发调度。但它有 64 个单元测试，有 Trace，有消融评测，有每一阶段的设计记录。对我而言，这比一个看起来更庞大的 demo 更像一个工程项目。
