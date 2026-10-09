# SuperAgentHua · MiniAgent 项目主仓库

> 项目：自研 Mini Agent 框架（大四实习个人 Agent 项目）
> 主目录：`D:\SuperAgentHua\`（git 仓库根目录，main 分支）
> 远程仓库：<https://github.com/Florin-Hua/huayuan>
> 当前版本：MiniAgent 0.9.0
> 目标：不依赖重型 Agent 框架，从 0 实现 ReAct 循环、工具系统、多模型适配、上下文压缩、记忆、健壮性、MCP 兼容、评测闭环与本地可视化控制台

## 项目结论

MiniAgent 已完成第0-9阶段。Agent 运行时代码 2373 非空行（另有 826 非空行可视化层），单元测试 69/69 通过，评测 runner 复现成功，并新增本地 Web 控制台。阶段8使用确定性 scripted policy 评测框架机制，`scripted-strong + 压缩 ON + 记忆 ON` 在 30 条自建任务中成功 30/30，失败恢复 8/8；关闭压缩后成功率降至 86.67%，关闭记忆后降至 96.67%，baseline 仅 6.67%。

这些数字证明的是 MiniAgent 的 ReAct 循环、工具执行、错误回喂、上下文压缩、记忆注入和评测统计链路有效，**不能外推为 GPT / Claude / DeepSeek 的真实准确率**。真实 OpenAI 调用当前因网络超时被阻断，Anthropic 缺少有效 key；GAIA 是 gated dataset，本项目未伪造公开基准结果。

## 架构总览

```text
CLI / Web UI / Python API
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

## 目录结构（按序号排序）

| 序号 | 路径 | 内容 | 当前状态 |
|---|---|---|---|
| 00 | 00-提示词/ | 项目主提示词（所有阶段开发总纲） | ✅ 已归档 |
| 01 | 01-阶段进程/ | 每阶段设计+验收+决策记录 | ✅ 第0-9阶段完成 |
| 02 | 02-代码仓库/ | mini-agent 从零自研源代码 | ✅ 0.9.0，测试 69/69 |
| 03 | 03-交付物/ | 评测报告、框架对比、面试笔记、博客 | ✅ 已完成 |
| 04 | 04-参考代码/ | 既有实现归档，仅参考不修改 | ✅ 已归档 |

## 阶段进度

| 阶段 | 进程文件 | 状态 | 完成日期 |
|---|---|---|---|
| 0 总体设计 | 第0阶段-总体设计.md | ✅ 已完成 | 2026-09-16 |
| 1 最小闭环 | 第1阶段-最小闭环.md | ✅ 已完成（单测 11/11；真实验收待网络恢复重试） | 2026-09-16 |
| 2 工具系统 | 第2阶段-工具系统.md | ✅ 已完成（单测 21/21；真实验收待网络恢复重试） | 2026-09-16 |
| 3 多模型适配 | 第3阶段-多模型适配.md | ✅ 已完成（单测 31/31；真实双 provider 待网络/密钥） | 2026-10-09 |
| 4 上下文压缩 | 第4阶段-上下文压缩.md | ✅ 已完成（单测 37/37；长任务压缩与预算终止验收通过） | 2026-10-09 |
| 5 记忆系统 | 第5阶段-记忆系统.md | ✅ 已完成（单测 44/44；跨会话记忆 mock 验收通过） | 2026-10-09 |
| 6 健壮性与Trace | 第6阶段-健壮性与Trace.md | ✅ 已完成（单测 54/54；真实任务 trace 待网络/密钥） | 2026-10-09 |
| 7 MCP兼容 | 第7阶段-MCP兼容.md | ✅ 已完成（单测 59/59；真实 stdio 子进程验收通过） | 2026-10-09 |
| 8 评测与交付 | 第8阶段-评测与交付.md | ✅ 已完成（单测 64/64；评测与交付文档完成） | 2026-10-09 |
| 9 可视化界面 | 第9阶段-可视化界面.md | ✅ 已完成（单测 69/69；本地 Web smoke test 通过） | 2026-10-09 |

## 关键决策记录

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-09-16 | 从零自研，既有代码只归档参考 | 保持交付叙事干净，每一行代码都能解释设计动机 |
| 2026-09-16 | 工具错误结构化回喂，不直接终止 | 让模型有机会基于 observation 自愈，也便于评测失败恢复 |
| 2026-10-09 | 统一内部消息格式 + provider adapter | 供应商差异限制在 adapter 层，核心循环与测试不依赖 SDK |
| 2026-10-09 | 超预算时滚动摘要，保留近期原文 | 控制长任务上下文膨胀，同时保留工具推理连续性 |
| 2026-10-09 | SQLite + sqlite-vec + 默认哈希向量 | 单文件部署、离线可测，Embedder 接口保留替换空间 |
| 2026-10-09 | 解析错误回喂重试 + 三类 Guard | 先给模型纠错机会，再由护栏兜底，避免无限循环和 token 爆炸 |
| 2026-10-09 | MCP 只做 stdio 最小工具能力 | 兼容生态的同时收窄信任边界，模型不能自行启动 server |
| 2026-10-09 | 阶段8使用 scripted 框架评测 | 离线可复现，并明确区分框架机制与商业模型能力 |
| 2026-10-09 | Web UI 使用标准库 HTTP server + 原生前端 | 零新增依赖、零构建链，可视化层不侵入 Agent 核心 |

## 快速导航

- 主提示词：[00-提示词/mini-agent-framework-prompt.md](00-提示词/mini-agent-framework-prompt.md)
- 总体设计：[01-阶段进程/第0阶段-总体设计.md](01-阶段进程/第0阶段-总体设计.md)
- 最终阶段记录：[01-阶段进程/第9阶段-可视化界面.md](01-阶段进程/第9阶段-可视化界面.md)
- 评测报告：[03-交付物/eval_report.md](03-交付物/eval_report.md)
- 框架对比：[03-交付物/framework_comparison.md](03-交付物/framework_comparison.md)
- 面试笔记：[03-交付物/interview_notes.md](03-交付物/interview_notes.md)
- 博客初稿：[03-交付物/blog_draft.md](03-交付物/blog_draft.md)
- 代码仓库：[02-代码仓库/mini-agent/README.md](02-代码仓库/mini-agent/README.md)
- 可视化界面：[02-代码仓库/mini-agent/src/mini_agent/web/index.html](02-代码仓库/mini-agent/src/mini_agent/web/index.html)
- 参考代码：`04-参考代码/`（只读，不属于交付叙事）

## 阶段8指标表

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

## 快速验收命令

```powershell
cd D:\SuperAgentHua\02-代码仓库\mini-agent
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe eval\runner.py
.venv\Scripts\python.exe -m compileall -q src tests eval
.venv\Scripts\python.exe -m mini_agent.webapp
```

预期：

```text
69 passed
8 组实验输出与 eval_report.md 一致
COMPILE_OK
浏览器打开 http://127.0.0.1:8765
```

## 已知局限

1. 阶段8是 scripted 框架机制评测，真实商业模型准确率未测；
2. OpenAI 当前网络超时，Anthropic 缺少有效 key；
3. GAIA 是 gated dataset，未授权前没有公开基准成绩；
4. 默认 HashingEmbedder 语义能力有限，记忆泛化样本量小；
5. 无 checkpoint 恢复、流式输出与并发调度；
6. MCP 仅实现 stdio `tools/list` / `tools/call`；
7. Python 线程池超时无法强制终止底层调用；
8. Web UI 面向本机使用，无登录鉴权，不适合直接暴露公网。

## 命名与提交规范

- 阶段进程文件：`第X阶段-名称.md`，统一存放在 `01-阶段进程/`
- git 提交信息格式：`feat/fix/test/docs(scope): 描述`
- 每完成一个阶段：写进程文件 → 更新本 README → git 提交并推送
- `.env`、`data/`、`.venv/`、`__pycache__/`、`.pytest_cache/`、`*.db`、`*.sqlite3` 严禁入库

## Git 状态说明

- 本地分支：`main`
- 远程：`origin` → <https://github.com/Florin-Hua/huayuan>
- 第7阶段提交 `cb6925d` 当时因 GitHub 网络连接失败未能推送；
- 第8阶段完成后已重试推送，实际结果以最终终端输出为准，本仓库不伪造推送状态。
