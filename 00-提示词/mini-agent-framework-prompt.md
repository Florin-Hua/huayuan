# 自研 Mini Agent 框架 · 完整项目提示词

> 目标：从零手写一个 mini agent 框架（不用任何重型框架），做到可评测、可对比、可面试深聊。
> 核心卖点：自研 ReAct 循环、工具系统、多模型适配、上下文压缩、记忆、评测闭环 + 主流框架源码研读对照。
> 与前两个项目的区别：这个项目最大的敌人不是技术难度，而是 scope 蔓延——所以本提示词内置了"代码量预算"和"明确不做什么"。

---

## 一、主提示词（Master Prompt，整体投喂给 AI 编程助手）

```markdown
# 角色
你是一位资深 Agent 框架工程师，读过 LangGraph / CrewAI / AutoGen / OpenClaw 的源码，主导过生产级 agent loop 的设计。
你的任务是全程指导并实现以下项目。遇到技术决策时，先给出 2-3 个方案的 trade-off 对比，再给出推荐和理由，然后动手实现。

# 项目目标
从零自研一个 mini agent 框架（代号 MiniAgent）：不借助任何重型 agent 框架，亲手实现 ReAct 循环、
工具注册与调度、多模型适配、上下文压缩、长期记忆与评测闭环。
定位：代码量克制、每个组件都为面试深聊服务的"教学级但工程级质量"框架。

# "mini" 的硬性边界（防 scope 蔓延，这是本项目的生死线）
- 核心代码（core/ + adapters/，不含 tests/、eval/、示例）不超过 3000 行
- 明确不做：多 agent 编排、Web UI、模型微调、流式前端、分布式执行
- 原则：一个真正能跑、能讲清楚的 ReAct loop，价值大于十个半成品功能
- 超出预算时先砍功能，不许改预算

# 核心能力（用户故事，必须全部支持）
1. 极简入口：Agent(tools, model).run("任务描述")，一条 API 跑多步任务
2. @tool 装饰器注册工具，自动从 Pydantic 参数模型生成 JSON Schema
3. 同一套循环支持 GPT-4o / Claude / DeepSeek / Qwen，切换只改 .env
4. 多步任务自主拆解：如"读取 notes.txt，统计行数，把结果写入 result.txt"
5. 上下文超预算时自动滚动摘要压缩，任务仍能完成
6. 工具失败 / 模型输出格式错误能自愈：错误信息回喂、自动重试
7. 每次运行产出完整 trace：每步的 prompt、工具调用、token、耗时
8.（加分项）兼容 MCP：能加载一个外部 MCP server 的工具当自己的工具用

# 技术栈（硬性约束，不得随意更换）
- Python 3.11+，用 uv 管理依赖
- 模型层：OpenAI 兼容接口 + Anthropic 原生接口两个 adapter，内部统一消息格式
- 存储：SQLite（trace / 会话 / 记忆）+ sqlite-vec（长期记忆向量检索）
- 依赖白名单（核心代码只允许这些）：openai、anthropic、pydantic、pydantic-settings、rich、tiktoken、pytest
- 禁止导入：langchain、langgraph、crewai、autogen、llama-index、dify SDK——一律手工实现
- 入口：Python API + CLI 双入口；不需要 Docker 部署，pip install -e . 即用

# 系统架构（分层，单一职责）
Agent 入口 → AgentCore 状态机（THINK → ACT → OBSERVE → REFLECT → END）
- ToolRegistry：@tool 注册、schema 生成、执行调度、错误归一化
- ModelAdapter：统一内部消息格式 ↔ 各家 API 格式互转
- ContextManager：token 预算、滚动摘要压缩、注入顺序管理
- MemoryStore：长期记忆向量检索与写入
- Tracer：逐步落库（prompt、工具、结果、token、耗时）
- Guard：最大迭代数、token 预算、单工具超时
模块之间只通过明确定义的数据结构（Message / ToolCall / ToolResult / TraceStep）通信，禁止互相直接读内部状态。

# Agent Loop 设计规范（框架的心脏）
- 状态机：THINK（模型决策）→ ACT（执行工具）→ OBSERVE（结果回填）→ REFLECT（决定继续/终止）→ END
- 最大迭代默认 10 次（可配置）；防止震荡：连续 2 次调用完全相同的工具+参数时强制终止
- 终止条件：模型给出最终答案 / 达到最大迭代 / 预算耗尽 / 连续相同调用
- 工具结果以固定格式回填给模型，模型必须能看到错误信息并有机会自行修正
- 模型输出解析失败：最多重试 2 次（把格式错误信息喂回去），仍失败则终止并完整保存 trace——这是评测重点用例

# 工具系统规范
- @tool 装饰器：从函数名 + docstring + Pydantic 参数模型自动生成 tool schema（名字/描述/参数 JSON Schema）
- 工具执行统一返回 {"ok": true, "data": ...} 或 {"ok": false, "error": ...}，错误信息必须可读且回喂模型
- 幻觉调用（模型调用了不存在的工具）：返回结构化错误让模型改，而不是抛异常崩溃
- 内置示例工具 3 个：calculator（安全表达式求值）、read_text_file（限制在沙箱目录）、mock_web_search（返回固定 fixture，可 mock）
- 额外写 1 个示例工具 write_text_file（同样沙箱限制），用于演示多步文件任务

# 上下文管理（核心亮点，面试必深挖）
- token 预算器：用 tiktoken 按模型估算；系统提示 + 历史 + 工具结果总和超预算触发压缩
- 滚动摘要：保留最近 K 轮原文 + 更早对话的 LLM 摘要；工具长输出采用"头 N 字符 + 尾 M 字符 + 中间省略"截断策略
- 注入顺序固定：系统角色 → 长期记忆 → 压缩后的历史 → 当前输入
- 压缩功能必须有开关：消融实验要对比 开/关 的成功率与 token 成本

# 记忆系统
- 显式工具 save_memory（模型可主动调用）+ 任务结束后自动抽取"值得记的事实"（用一次 LLM 调用）
- 每次任务开始前检索 top-k 相关记忆注入系统提示词
- 记忆记录来源（哪次 run 写入），支持删除与人工审核（CLI 命令 mini memory list/delete）

# 观测与预算（Guard + Tracer）
- Tracer 落 SQLite：run_id、step、prompt 摘要、工具名、参数、结果状态、token、耗时
- Guard：最大迭代 10、单 run token 预算上限（默认 100k）、单工具执行超时 30s；任何触发都要在最终输出里说明原因
- CLI 命令：mini run "任务"、mini trace <run_id>（rich 表格展示执行轨迹）、mini memory list

# 评测体系（面试核心弹药，必须量化）
- 主基准：GAIA validation 子集，选 30-50 条 text 类（不依赖真实浏览器的）任务
- 辅助：自建 30 条多步任务（文件操作 / 计算 / mock 搜索组合），必须包含失败恢复用例（工具故意报错，看 agent 能否自愈）
- 指标：任务成功率、平均迭代步数、平均 token 成本、自愈率（工具失败后恢复并完成任务的比例）
- 消融至少 2 组：上下文压缩开/关；长期记忆开/关；外加至少 2 个模型横向对比
- 输出 eval_report.md + 失败案例逐条归因（至少 5 条真实失败案例，禁止只报平均分）

# 开发节奏（严格分阶段推进，每阶段验收通过才进入下一阶段）
- 阶段 0 设计（2 天）：状态机图、模块边界、公共 API 与核心数据结构、各模块行数预算分配
- 阶段 1 最小循环（3 天）：手写 ReAct loop + 2 个硬编码工具 + 单模型 + CLI，跑通一个多步任务
- 阶段 2 工具系统（4 天）：@tool 装饰器、schema 自动生成、注册表、错误归一化、幻觉调用处理
- 阶段 3 多模型适配（3 天）：OpenAI 兼容 + Anthropic 两个 adapter、统一消息格式、function calling 差异抹平
- 阶段 4 上下文压缩（4 天）：token 预算、滚动摘要、工具输出截断、消融开关
- 阶段 5 记忆系统（3 天）：sqlite-vec 检索、save_memory、任务后自动抽取、注入
- 阶段 6 健壮性与 Trace（3 天）：解析失败重试、Guard、Tracer 落库、mini trace 命令
- 阶段 7 MCP 兼容（3 天，可选加分）：加载一个外部 MCP server 的工具，适配进 ToolRegistry
- 阶段 8 评测与交付（5 天）：GAIA 子集 + 自建任务、消融、报告、README、博客、面试笔记

# 源码研读任务（贯穿全程，产出 framework_comparison.md）
- OpenClaw：重点看它的 agent loop、持久化与记忆系统设计
- LangGraph：图编排 vs 你的状态机——各自适用什么场景
- CrewAI：role-based 多 agent 与单 agent 循环的能力边界
- AutoGen：它的消息传递抽象设计
每个项目写约 200 字：核心抽象是什么 / 你借鉴了什么 / 你为什么不那样做。禁止空话，必须引用具体源码位置或机制。

# 交付物清单
1. 可安装的包：pip install -e . 后即可 mini run "..." 与 Python API 调用
2. README.md：架构图（Mermaid）、10 行快速上手 API 示例、与主流框架对比表、评测指标表、已知局限（至少 3 条）
3. framework_comparison.md：源码研读对照笔记
4. eval_report.md：GAIA + 自建任务评测 + 消融实验 + 失败案例归因
5. interview_notes.md：10 个面试深挖问题的答案（见下方清单）
6. 一篇技术博客《为什么我不用框架手写了一个 Agent》的完整初稿

# 面试深挖问题（interview_notes.md 必须覆盖）
1. Agent 和普通 LLM 应用的本质区别是什么？你的 loop 怎么体现？
2. 为什么不用 LangChain/LangGraph？自研的合理边界在哪里？
3. ReAct 循环怎么终止？死循环/震荡（反复调用同一工具）怎么办？
4. 工具 schema 怎么自动生成？模型幻觉调用不存在的工具怎么处理？
5. 上下文压缩会丢信息吗？你怎么验证压缩没有伤害成功率？
6. 多模型的 function calling 格式差异怎么抹平？你的内部消息格式怎么设计？
7. 为什么用 GAIA 做评测？怎么防止评测集被训练数据污染？
8. MCP 和你的框架是什么关系？为什么值得兼容它？
9. 3000 行的代码预算你是怎么守住的？砍掉了哪些功能？
10. 如果要生产化，你最先补什么？（并发安全 / 工具沙箱 / 成本控制 / 可观测性）

# 禁止事项
- 禁止导入 langchain / langgraph / crewai / autogen / llama-index / dify 等任何框架
- 禁止一次性写完所有代码再给我：必须每阶段先给设计、我确认后再实现
- 禁止写任何一行你解释不了其用途的代码
- 禁止伪造评测数据和指标
- 禁止跳过任何阶段验收
- 禁止用"以后再补"掩盖核心功能缺失——先砍需求，不留烂尾
```

---

## 二、分阶段执行提示词（每阶段开始时发送）

### 阶段 0：设计

```markdown
我们开始阶段 0：总体设计。只输出设计文档，不写实现代码。
请依次给出：
1. AgentCore 状态机 Mermaid 图：THINK/ACT/OBSERVE/REFLECT/END 的转移条件，包括所有终止路径
2. 模块依赖图：Agent / ToolRegistry / ModelAdapter / ContextManager / MemoryStore / Tracer / Guard
3. 核心数据结构定义（Pydantic 草稿）：Message、ToolCall、ToolResult、TraceStep、AgentRun
4. 公共 API 草稿：Agent(tools, model).run() 的完整签名与返回值设计
5. 3000 行的预算分配表：每个模块分配多少行，超支时的砍功能优先级
6. 两个关键决策的 trade-off 分析：a) 显式状态机 vs while 循环 + if；b) 内部统一消息格式各字段怎么定
完成后列出本阶段验收清单，等我确认再进入阶段 1。
```

### 阶段 1：最小循环

```markdown
进入阶段 1：最小闭环。目标是亲手写一遍 ReAct loop，禁止提前做工具系统。
1. 初始化 uv 项目与包结构（src/mini_agent/），实现 pyproject.toml 可 pip install -e .
2. 硬编码两个工具（不用装饰器）：calculator、read_text_file（沙箱目录限制）
3. 手写 AgentCore：单模型（OpenAI 兼容）、最大迭代 10、工具结果回填、最终答案终止
4. CLI 入口 mini run "..."
验收任务（必须全部通过）：
a) "计算 23*47+11，再读取 data/notes.txt，把两个结果分别告诉我"
b) 故意让 read_text_file 指向不存在的文件，观察模型是否看到错误信息并调整
5. 补 pytest：mock 模型响应，测试循环的终止条件（最终答案 / 最大迭代）
先给出文件清单和每个文件的职责与预计行数，再写代码。
```

### 阶段 2：工具系统

```markdown
进入阶段 2：把硬编码工具升级为正式工具系统。
1. @tool 装饰器：从函数名 + docstring + Pydantic 参数模型自动生成 tool schema
2. ToolRegistry：注册、按名查找、并发执行先不做（单线程即可）、执行超时 30s
3. 统一返回 {"ok":true,"data":...}/{"ok":false,"error":...}；错误信息必须回喂模型
4. 幻觉调用处理：模型调用了不存在的工具时返回结构化错误，模型有机会改
5. 内置工具集：calculator、read_text_file、write_text_file（都限制沙箱目录）、mock_web_search（fixture 返回）
6. pytest：schema 生成正确性、幻觉调用、超时、沙箱越界拦截（读写 ../ 必须被拒）
完成后用一个多步任务验收："读取 notes.txt 统计行数，把行数写入 result.txt，再读回来验证"。
```

### 阶段 3：多模型适配

```markdown
进入阶段 3：多模型适配层。
1. 定义内部统一消息格式（阶段 0 的设计），模型无关
2. OpenAI 兼容 adapter：覆盖 GPT-4o/DeepSeek/Qwen（同一实现，base_url 可配）
3. Anthropic adapter：原生 SDK，把内部格式转成 Anthropic 的 tool_use/tool_result 格式
4. .env 切换：MODEL_PROVIDER=openai/anthropic、MODEL_NAME、base_url、api_key
5. 两个 adapter 各配一组"录制的真实响应"做 fixture（手工 mock），pytest 验证格式互转双向正确
验收：同一个多步任务在两个不同 provider 上都能完成，代码零改动只改 .env。
```

### 阶段 4：上下文压缩

```markdown
进入阶段 4：上下文预算与压缩，本项目第一个核心亮点。
1. ContextManager：tiktoken 估算当前上下文 token；超过预算（默认 8k，可配）触发压缩
2. 滚动摘要：最近 K=5 轮原文保留，更早内容用 LLM 生成摘要替换
3. 工具输出截断：超过 2000 字符时保留头 1000 + 尾 500 + 省略提示
4. 注入顺序：系统 → 记忆 → 压缩历史 → 当前输入
5. 压缩开关（.env），每次压缩记录统计：压缩前后 token、被摘要的轮数
6. 验收：构造一个必然超预算的长任务（10+ 轮工具调用），验证开压缩能完成、关压缩会触发 Guard 的预算终止
7. pytest：token 估算、截断逻辑、注入顺序
```

### 阶段 5：记忆系统

```markdown
进入阶段 5：长期记忆。
1. MemoryStore：SQLite + sqlite-vec，记忆字段：内容、embedding、来源 run_id、创建时间
2. save_memory 工具：模型可主动调用保存事实
3. 任务结束后自动抽取：用一次 LLM 调用判断本次会话是否有"值得长期记住的事实"，有则写入
4. 每次任务开始检索 top-3 相关记忆注入系统提示词（有开关）
5. CLI：mini memory list / mini memory delete <id>
6. pytest：写入、检索相关性排序、删除
验收：第一次告诉 agent"我的导师姓张"，新会话里问"我导师姓什么"能靠记忆答对。
```

### 阶段 6：健壮性与 Trace

```markdown
进入阶段 6：健壮性 + 可观测性，这是"工程级"和"玩具"的分界线。
1. 输出解析失败自愈：捕获非法 JSON/工具调用格式 → 把错误信息回喂重试（最多 2 次）→ 仍失败保存 trace 并优雅终止
2. Guard：最大迭代 10、单 run token 预算 100k、震荡检测（连续 2 次完全相同的工具+参数 → 强制终止）
3. Tracer 落 SQLite：run_id、step、工具名、参数、结果状态、token、耗时；mini trace <run_id> 用 rich 表格展示
4. 失败注入 pytest：非法模型输出、工具抛异常、工具超时、震荡调用——全部验证不崩溃且有正确终止原因
5. 跑三个真实任务，附 trace 截图/文本存 docs/ 目录
```

### 阶段 7：MCP 兼容（可选加分）

```markdown
进入阶段 7（可选）：MCP 兼容，把框架接入 2026 年生态。
1. 用 MCP Python SDK 写一个最小 MCP server（暴露 2 个示例工具，stdio transport）
2. 在 MiniAgent 里实现 McpToolAdapter：连接该 server、把它的工具适配成 ToolRegistry 里的标准工具（schema 转换、调用转发、错误归一）
3. 验收：mini run 里模型能像调用内置工具一样调用 MCP 工具
4. pytest：mock stdio 消息，验证工具发现与调用转发
5. 在 interview_notes.md 里补一节：MCP 工具与本地工具在权限/安全上的差异
```

### 阶段 8：评测与交付

```markdown
进入阶段 8：评测与交付。禁止伪造任何数据。
1. eval/gaia_subset.json：从 GAIA validation 挑 30-50 条不依赖真实浏览器的 text 任务，说明筛选标准
2. eval/selfbuilt.json：自建 30 条多步任务，含 5 条失败恢复用例（工具故意报错）
3. eval/runner.py：批量执行、记录成功率/迭代步数/token 成本/自愈率；支持按配置开关记忆与压缩
4. 消融：压缩开/关、记忆开/关、至少 2 个模型横向——输出对比表
5. eval_report.md：总指标 + 分类指标 + 至少 5 个失败案例逐条归因
6. README.md：架构图、快速上手、与 LangGraph/CrewAI/OpenClaw 的对比表（引用 framework_comparison.md）、指标表、已知局限至少 3 条
7. framework_comparison.md：按主提示词的源码研读任务完成（OpenClaw/LangGraph/CrewAI/AutoGen 各约 200 字，必须具体）
8. interview_notes.md：10 个面试问题逐一作答（我的设计 + trade-off + 真实踩坑例子）
9. 博客初稿《为什么我不用框架手写了一个 Agent》：1500 字以上，核心讲决策而不是流水账
10. 终审：统计核心代码行数（必须在预算内）、删除死代码、确认无硬编码密钥、跑全量测试
```

---

## 三、使用方式

1. **整体投喂**：把「主提示词」粘贴给 Codex / Claude Code / Cursor 作为长期任务说明，之后每轮只说"进入阶段 X"。
2. **分阶段投喂**：每轮把「主提示词 + 当前阶段提示词」一起发送（推荐，长项目更稳）。
3. **人在回路**：每个阶段的产出先自己看懂、能讲出来再放行。这个项目的简历价值 = 你能解释的每一行设计。
4. **成本控制**：开发期用 DeepSeek/Qwen；评测期至少跑一次 GPT-4o 或 Claude 做横向对比。
5. **行数纪律**：每阶段结束看一次核心代码行数，逼近 3000 行预算就砍功能——"mini 的克制"本身就是面试故事。
6. **源码对照**：每完成一个模块，就去读对应框架（LangGraph/CrewAI/OpenClaw）的同模块源码，把差异写进 framework_comparison.md——这是本项目区别于"又一个玩具框架"的关键。
