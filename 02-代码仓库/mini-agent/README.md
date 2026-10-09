# MiniAgent

从零自研的 mini agent 框架。当前进度：阶段5（记忆系统）。

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
.venv\Scripts\python.exe -m mini_agent.cli run "读取 notes.txt，统计行数，把结果写入 data/result.txt" --provider anthropic --sandbox data
```

## 测试

```powershell
.venv\Scripts\python.exe -m pytest
```

当前结果：`44 passed`。测试使用 fake client、录制 fixture 与 scripted adapter，不访问网络、不消耗 API key。

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

记忆默认保存在 `data/memory.sqlite3`，使用 SQLite + sqlite-vec 做向量检索。任务开始会注入 top-3 相关记忆，任务结束后自动抽取值得保存的事实，模型也可以调用 `save_memory` 工具主动保存。

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

### 阶段2 · 工具系统

- `@tool` 装饰器与 Pydantic schema 自动生成
- `ToolRegistry`：注册、查找、schema 输出、统一执行
- 默认 30 秒工具超时，可通过 `.env` 配置
- 内置工具：calculator、read_text_file、write_text_file、mock_web_search
- 沙箱路径逃逸拦截
- 多步任务：读取 -> 写入 -> 读回验证

### 阶段3 · 多模型适配

- `OpenAIAdapter`：OpenAI 兼容接口
- `AnthropicAdapter`：Anthropic 原生 Messages API 与 tool_use/tool_result 转换
- `create_adapter()`：按 `MODEL_PROVIDER` 选择 provider
- CLI `--provider openai/anthropic`
- AgentCore / ReAct / ToolRegistry 不感知供应商格式

### 阶段5 · 记忆系统

- `MemoryStore`：SQLite + sqlite-vec，保存内容、embedding、来源 run_id、创建时间
- 任务开始检索 top-3 相关记忆并注入 system prompt
- `save_memory` 工具支持模型主动保存事实
- 任务结束后用一次模型调用自动抽取长期事实
- CLI：`mini memory list / search / delete`

### 阶段4 · 上下文压缩

- `ContextManager`：tiktoken token 估算与上下文预算
- 滚动摘要：最近 5 轮原文，早期历史交给当前模型压缩
- 工具输出截断：头 1000 + 尾 500 + 省略标记
- 注入顺序：system → memory → compressed history → current input → recent turns
- 超预算关闭压缩时返回 `context_limit`

## 当前验收说明

单元测试 44/44 通过。真实 OpenAI 调用当前因 `api.openai.com` 网络超时被阻断；Anthropic 网络可达但缺少有效 `ANTHROPIC_API_KEY`。待网络或密钥配置后按根仓库 `01-阶段进程/第3阶段-多模型适配.md` 与 `第5阶段-记忆系统.md` 中的命令重试。

完整设计见仓库根 `01-阶段进程/` 各阶段文档。
