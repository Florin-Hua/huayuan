# 第6阶段 · 健壮性与 Trace

> 完成日期：2026-10-09
> 状态：工程实现与失败注入测试完成；真实模型三任务验收因当前 OpenAI 网络 / Anthropic 密钥条件未满足，待恢复后补充。

## 1. 阶段目标

本阶段把 MiniAgent 从「功能可跑」推进到「工程级可观测、可恢复、可终止」：

1. 模型输出解析失败不直接崩溃，而是把错误回喂模型重试；
2. 加入最大迭代、token 预算、震荡调用三类 Guard；
3. 把 run 汇总与逐步轨迹落 SQLite；
10. CLI 可按 `run_id` 查看 trace；
5. 用失败注入测试验证非法输出、工具异常、工具超时和循环震荡。

## 2. 交付文件

| 文件 | 作用 |
|---|---|
| `src/mini_agent/core/types.py` | 新增 `ModelResponseError`、`TOKEN_BUDGET`、`OSCILLATION`、Trace 字段 |
| `src/mini_agent/core/loop.py` | 解析自愈、Guard、执行轨迹、Tracer 接入 |
| `src/mini_agent/trace.py` | SQLite `TraceStore`：runs + trace_steps |
| `src/mini_agent/core/agent.py` | Agent 门面组装 TraceStore 与护栏配置 |
| `src/mini_agent/config.py` / `.env.example` | 新增 Stage 6 配置 |
| `src/mini_agent/cli.py` | 新增 `mini trace <run_id>` |
| `tests/test_robustness_trace.py` | 9 个失败注入 / Trace / CLI 测试 |
| `README.md` | 使用方式与阶段说明更新 |

## 3. 核心设计

### 3.1 输出解析失败自愈

新增 `ModelResponseError`，表示「模型响应结构坏掉，但可以尝试纠正」：

- OpenAI 兼容接口：tool arguments 不是合法 JSON object 时抛出；
- Anthropic 原生接口：tool input 不是 object 时抛出；
- AgentCore 校验 `content` 与 `tool_calls` 至少有一个非空、tool id 不为空且不重复、tool name 合法。

处理流程：

```text
adapter.chat()
  ├─ ModelResponseError / 结构校验失败
  │    ├─ 第 1/2 次：追加 user 纠错消息，重新调用模型
  │    └─ 第 3 次仍失败：RunStatus.PARSE_ERROR，保存 trace，优雅终止
  └─ 网络 / provider 异常：RunStatus.PARSE_ERROR，不重试
```

关键决策：非法 `tool_calls` 不写入正式 assistant 历史，只保留文本并追加纠错 user 消息，避免下一轮继续携带坏结构。

### 3.2 Guard

| Guard | 默认值 | 终态 |
|---|---:|---|
| 最大迭代 | 10 | `max_iterations` |
| 单 run token 预算 | 100,000 | `token_budget` |
| 震荡检测 | 连续两轮完全相同工具+参数 | `oscillation` |
| 上下文预算 | 8192 | `context_limit` |

震荡检测比较的是规范化 JSON 签名：

```python
json.dumps([call.name, call.arguments], sort_keys=True, separators=(",", ":"))
```

第二轮与上一轮完全相同即终止，且不会执行第二次重复工具。

### 3.3 TraceStore

Trace 与 Memory 分库：

- Memory：用户长期事实，生命周期跟随用户；
- Trace：运行事实，用于调试、评测与失败归因，可随时清理。

表结构：

```text
runs
  run_id, task, status, answer, iterations,
  prompt_tokens, completion_tokens, total_tokens,
  parse_retries, termination_reason, started_at, finished_at

trace_steps
  run_id, step, iteration, state,
  tool_name, tool_args, result_status,
  error, note, tokens, duration_ms, created_at
```

`AgentCore.run()` 在 `finally` 中保存 trace；即使后续记忆抽取修改了 `AgentRun`，Agent 门面会再次保存最新结果。Trace 保存失败不会影响任务结果返回。

### 3.4 CLI

```powershell
mini trace <run_id>
```

输出 rich 表格，包含 step、iteration、state、tool、args、result、tokens、耗时与错误详情。

## 4. 配置

```env
MAX_ITERATIONS=10
TOKEN_BUDGET=100000
PARSE_RETRY_LIMIT=2
TRACE_ENABLED=true
TRACE_DB_PATH=data/trace.sqlite3
```

`data/trace.sqlite3` 属于运行数据，已被 `.gitignore` 忽略。

## 5. 验收结果

### 单元测试

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m compileall -q src tests
```

结果：

```text
54 passed in 3.72s
COMPILE_OK
```

新增第6阶段测试覆盖：

1. 非法模型输出 → 错误回喂 → 下一轮修正 → 成功；
2. adapter 直接抛 `ModelResponseError` → 回喂重试 → 成功；
3. 连续非法输出 → 重试 2 次 → `parse_error` 优雅终止；
10. 工具抛异常 → `ToolResult.ok=false` → 错误回喂模型；
4. 工具超时 → `ToolTimeoutError` → 错误回喂模型；
10. token 预算超限 → `token_budget`，不执行后续工具；
10. 连续两轮相同工具+参数 → `oscillation`，第二次不执行；
10. TraceStore 保存 run 与 step，含参数 JSON、token、耗时；
10. Agent 门面集成 TraceStore；
9. CLI parser 支持 `trace` 子命令。

### 真实任务验收

尚未执行，不伪造结果。当前限制：

- OpenAI / OpenAI 兼容 endpoint 网络超时；
- Anthropic 网络可达但缺少有效 `ANTHROPIC_API_KEY`。

恢复后执行三类任务并保存文本 trace 到 `docs/`：

1. 多步文件读取/写入/验证；
2. 工具失败恢复（先读不存在文件，再根据错误说明终止）；
3. 长上下文 + 压缩 + trace 查询。

## 6. 关键决策

| 决策 | 理由 |
|---|---|
| 解析错误与网络错误分开处理 | 前者可回喂纠错，后者重试大概率仍失败且可能浪费时间 |
| 非法 tool calls 不进入正式历史 | 避免坏结构被下一轮模型继续放大 |
| 震荡按规范化 JSON 签名比较 | 避免字典顺序导致的漏判，同时保持实现简单 |
| Trace 使用独立 SQLite 文件 | 运行数据与用户记忆的生命周期、清理策略不同 |
| Trace 保存失败不影响任务结果 | 可观测性不能反过来破坏主流程可用性 |

## 7. 代码规模

- 运行时代码：约 2,099 行（预算 3,000 行内）
- 测试代码：约 1,081 行
- 当前版本：`0.6.0`

## 8. 已知局限与下一步

- 震荡检测只比较相邻两轮完整工具调用序列，尚未覆盖更复杂循环模式；
- Token Guard 使用供应商返回 usage，缺少本地估算兜底；
- Trace 目前只支持单机 SQLite，没有并发 run 聚合查询；
- 真实任务 trace 待网络 / 密钥恢复后补充。

下一阶段为可选的第7阶段：MCP 兼容，实现最小 MCP server 与 `McpToolAdapter`。
