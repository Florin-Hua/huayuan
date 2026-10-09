# MiniAgent

> MiniAgent 0.8.0：从零自研的 mini agent 框架。当前完成第0-8阶段，包含 ReAct 循环、工具系统、多模型适配、上下文压缩、记忆、健壮性与 Trace、MCP 兼容和 scripted 框架评测。

## 架构

```text
CLI / Python API
      │
   Agent 门面
      │
   AgentCore ReAct 状态机
 THINK → ACT → OBSERVE → REFLECT → THINK / END
      │              │              │
 ModelAdapter   ToolRegistry     Guard
 OpenAI/Anthropic  本地工具       迭代 / token / 震荡
                   MCP Tool       │
 ContextManager  MemoryStore   TraceStore
 滚动摘要+截断   SQLite+vec    SQLite trace
```

## 快速开始

```powershell
uv venv .venv
uv pip install -e ".[dev]"
copy .env.example .env   # 填入模型配置
mini run "读取 data/notes.txt，统计行数，把结果写入 data/result.txt" --sandbox data
```

也可以直接运行模块：

```powershell
.venv\Scripts\python.exe -m mini_agent.cli run "读取 notes.txt 并告诉我内容" --sandbox data
```

## 多模型配置

MiniAgent 内部使用统一消息格式，供应商差异被限制在 adapter 层。当前支持：

- `openai`：OpenAI / DeepSeek / Qwen 等 OpenAI 兼容接口；
- `anthropic`：Anthropic Messages API 原生接口。

`.env` 示例：

```env
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini

OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-key

# 切换 Anthropic：
# MODEL_PROVIDER=anthropic
# MODEL_NAME=claude-sonnet-4-5
# ANTHROPIC_API_KEY=sk-ant-your-key
# ANTHROPIC_BASE_URL=https://api.anthropic.com
# ANTHROPIC_MAX_TOKENS=1024
```

CLI 可以临时覆盖 provider：

```powershell
.venv\Scripts\python.exe -m mini_agent.cli run "读取 notes.txt，统计行数，把结果写入 result.txt" --provider anthropic --sandbox data
```

## 测试

```powershell
.venv\Scripts\python.exe -m pytest
```

当前结果：`64 passed in 15.37s`。测试使用 fake client、录制 fixture、scripted adapter、失败注入与真实 stdio MCP 子进程；除 MCP 子进程外不访问网络、不消耗 API key。

## 评测

阶段8评测定位为 **框架机制评测**，不是商业模型能力评测。`scripted-strong` 按数据集脚本调用工具，`scripted-baseline` 不调用工具，二者均通过真实 Agent、工具、上下文与记忆链路执行。

```powershell
.venv\Scripts\python.exe eval\runner.py
```

输出写入 `eval/results/selfbuilt_scripted.json`：

| 模型策略 | 压缩 | 记忆 | 成功率 | 平均迭代 | 平均 token | 失败恢复率 |
|---|---:|---:|---:|---:|---:|---:|
| scripted-strong | ON | ON | 100.00% | 2.40 | 417.5 | 100.00% |
| scripted-strong | ON | OFF | 96.67% | 2.40 | 326.4 | 100.00% |
| scripted-strong | OFF | ON | 86.67% | 2.27 | 325.0 | 100.00% |
| scripted-strong | OFF | OFF | 83.33% | 2.27 | 245.7 | 100.00% |
| scripted-baseline | ON | ON | 6.67% | 1.00 | 173.8 | 0.00% |
| scripted-baseline | ON | OFF | 3.33% | 1.00 | 84.6 | 0.00% |
| scripted-baseline | OFF | ON | 6.67% | 1.00 | 173.8 | 0.00% |
| scripted-baseline | OFF | OFF | 3.33% | 1.00 | 84.6 | 0.00% |

自建数据集 `eval/selfbuilt.json` 共 30 条，覆盖 arithmetic、file_io、mock_search、context_compression、memory、failure_recovery。其中 8 条失败恢复任务全部恢复；关闭压缩后 4 条长上下文任务全部 `context_limit`；关闭记忆后 SB022 失败。

### GAIA 状态

GAIA `gaia-benchmark/GAIA` 是 HuggingFace gated dataset。本机没有授权 token，`eval/gaia_subset.json` 仅保存 gated 状态和筛选标准，`tasks` 为空，不包含未授权数据。获得授权并自行下载 validation JSONL 后可运行：

```powershell
.venv\Scripts\python.exe eval\select_gaia.py --input data\gaia_validation.jsonl --output eval\gaia_subset.json --limit 30
```

## 上下文压缩

默认开启，预算 8192 token，最近 5 轮保留原文，更早轮次滚动摘要；工具输出超过 2000 字符时保留头 1000 + 尾 500。

```env
CONTEXT_COMPRESSION_ENABLED=true
CONTEXT_TOKEN_BUDGET=8192
CONTEXT_RECENT_TURNS=5
CONTEXT_TOOL_OUTPUT_MAX_CHARS=2000
CONTEXT_TOOL_OUTPUT_HEAD_CHARS=1000
CONTEXT_TOOL_OUTPUT_TAIL_CHARS=500
```

长任务运行时，CLI 会显示 `上下文压缩：N 次`；关闭压缩后超预算会以 `context_limit` 优雅终止。

## 长期记忆

记忆默认保存在 `data/memory.sqlite3`，使用 SQLite + sqlite-vec 做向量检索。任务开始注入 top-3 相关记忆，任务结束后自动抽取值得保存的事实，模型也可以调用 `save_memory` 工具主动保存。

```env
MEMORY_ENABLED=true
MEMORY_DB_PATH=data/memory.sqlite3
MEMORY_TOP_K=3
MEMORY_DIMENSION=256
```

CLI：

```powershell
mini memory list
mini memory search "导师" --top-k 3
mini memory delete 1
```

默认向量器是离线 HashingEmbedder，适合测试与无网络环境；后续可通过 Embedder 协议替换为语义 embedding。

## 健壮性与 Trace

AgentCore 内置三层护栏：

- 最大迭代次数：默认 10；
- 单 run token 预算：默认 100,000，超限返回 `token_budget`；
- 震荡检测：连续两轮完全相同的工具与参数，返回 `oscillation`。

模型输出解析失败时回喂错误并重试，默认最多 2 次；工具异常和超时仍结构化回喂模型，不直接崩溃。

```env
MAX_ITERATIONS=10
TOKEN_BUDGET=100000
PARSE_RETRY_LIMIT=2
TRACE_ENABLED=true
TRACE_DB_PATH=data/trace.sqlite3
```

Trace 保存 run 汇总与每一步状态、工具名、参数、结果、token 与耗时：

```powershell
mini trace <run_id>
```

## MCP 兼容

内置最小 MCP server 暴露两个示例工具：

```powershell
.venv\Scripts\python.exe -m mini_agent.mcp_server
# 或
mini-mcp-server
```

在 run 时接入：

```powershell
mini run "用 MCP 计算 19 + 23" --mcp-server ".venv\Scripts\python.exe -m mini_agent.mcp_server"
```

也可以在 `.env` 中配置：

```env
MCP_SERVER_COMMAND=python -m mini_agent.mcp_server
MCP_TOOL_TIMEOUT_SECONDS=30
```

`McpToolAdapter` 会把 MCP `tools/list` 转成标准 `ToolDefinition`，模型看到的 schema 与本地工具一致；`tools/call` 的结果和错误也会归一化后回喂模型。安全边界：server 命令只能由用户显式配置，模型不能自行启动 MCP server。

## 自定义工具

```python
from mini_agent import Agent, tool

@tool
def shout(text: str) -> str:
    """把文本转为大写。"""
    return text.upper()

agent = Agent(tools=[shout], sandbox_dir="data")
result = agent.run("把 hello 转成大写")
print(result.answer)
```

工具参数可用 Pydantic 模型显式声明，也可以从函数签名自动生成。执行失败、参数错误、超时和幻觉工具都会被归一化为结构化错误并回喂模型。

## 已实现范围

### 阶段1 · 最小 ReAct 闭环

- `AgentCore`：THINK / ACT / OBSERVE / REFLECT 状态机
- 统一 `AgentRun` 返回状态、答案、usage、steps 与终止原因
- 模型错误和工具错误均不直接崩溃
- CLI `mini run`

### 阶段2 · 工具系统

- `@tool` 装饰器与 Pydantic schema 自动生成
- `ToolRegistry`：注册、查找、schema 输出、统一执行
- 默认 30 秒工具超时
- 内置 calculator、read_text_file、write_text_file、mock_web_search
- 沙箱路径逃逸拦截与 AST 白名单计算器

### 阶段3 · 多模型适配

- `OpenAIAdapter`：OpenAI 兼容接口
- `AnthropicAdapter`：Messages API 与 tool_use/tool_result 转换
- `create_adapter()` 按 `MODEL_PROVIDER` 选择 provider
- CLI `--provider openai/anthropic`

### 阶段4 · 上下文压缩

- `ContextManager`：tiktoken 估算与上下文预算
- 滚动摘要：最近 5 轮原文，早期历史交给当前模型压缩
- 工具输出截断：头 1000 + 尾 500 + 省略标记
- 注入顺序：system → memory → compressed history → current input → recent turns

### 阶段5 · 记忆系统

- `MemoryStore`：SQLite + sqlite-vec，保存内容、embedding、来源 run_id
- 任务开始检索 top-3 相关记忆并注入 system prompt
- `save_memory` 工具与任务结束自动抽取
- CLI：`mini memory list / search / delete`

### 阶段6 · 健壮性与Trace

- `ModelResponseError` 与解析失败错误回喂重试
- Guard：最大迭代、token 预算、连续相同工具调用震荡
- `TraceStore`：SQLite 保存 run 汇总与逐步 trace
- CLI：`mini trace <run_id>`

### 阶段7 · MCP兼容

- MCP Python SDK 2.x 最小 stdio server：`mcp_echo`、`mcp_add`
- `McpToolAdapter`：工具发现、schema 转换、调用转发、错误归一化
- 后台 asyncio 线程桥接同步 ToolRegistry
- CLI：`mini run --mcp-server "..."`

### 阶段8 · 评测与交付

- 30 条自建框架机制评测数据集
- 8 组 scripted 消融实验：strong/baseline × 压缩 ON/OFF × 记忆 ON/OFF
- 逐任务 records、分类指标、失败恢复率
- GAIA gated 状态与授权后本地抽样脚本
- 5 个新增评测测试，全量 64/64 通过

## 代码规模与验收

| 范围 | 文件数 | 行数 |
|---|---:|---:|
| `src/mini_agent/` | 19 | 2369 |
| `tests/` | 8 | 1262 |
| `eval/` | 2 | 295 |

验收命令：

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe eval\runner.py
.venv\Scripts\python.exe -m compileall -q src tests eval
```

## 已知局限

1. 阶段8是 scripted 框架评测，真实商业模型准确率未测；
2. OpenAI 当前网络超时，Anthropic 缺少有效 key；
3. GAIA 是 gated dataset，未授权前没有公开基准成绩；
4. 默认 HashingEmbedder 语义能力有限，记忆泛化样本量小；
5. 无 checkpoint 恢复、流式输出与并发调度；
6. MCP 仅实现 stdio `tools/list` / `tools/call`；
7. Python 线程池超时无法强制终止底层调用。

完整设计见仓库根 `01-阶段进程/`，交付报告见 `03-交付物/`。
