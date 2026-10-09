# MiniAgent 阶段8评测报告

> 评测时间：2026-10-09
> 评测对象：MiniAgent 0.8.0
> 数据：`02-代码仓库/mini-agent/eval/selfbuilt.json`
> 原始结果：`02-代码仓库/mini-agent/eval/results/selfbuilt_scripted.json`

## 1.评测结论

- 自建 30 条任务中，开启压缩与记忆的 `scripted-strong` 策略成功 30/30。
- 关闭上下文压缩后，4 条长上下文任务全部进入 `context_limit`，总成功率降至 86.67%。
- 关闭记忆后，跨任务偏好任务失败，总成功率降至 96.67%。
- 8 条失败恢复任务全部在工具错误回喂后恢复，恢复率 100%。
- 无工具调用的 `scripted-baseline` 策略成功率仅 6.67%，说明结果主要来自工具执行而非答案猜测。

**重要口径说明**：本轮是“框架机制评测”，不是商业模型能力评测。模型侧使用两个确定性 scripted policy（`strong` / `baseline`），用于控制是否调用工具、是否根据记忆作答。因此这些数字只能证明 MiniAgent 的 ReAct 循环、工具错误回喂、上下文压缩、记忆注入和评测统计链路有效，不能外推为 GPT / Claude / DeepSeek 的真实准确率。

## 2.总指标

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

`token` 为 runner 内 adapter 的确定性 usage 估算，包含主循环、上下文摘要与记忆抽取调用，不代表供应商计费 token。

## 3.分类指标（scripted-strong，压缩 ON，记忆 ON）

| 类别 | 数量 | 成功率 | 平均迭代 | 平均 token | 压缩次数 |
|---|---:|---:|---:|---:|---:|
| arithmetic | 7 | 100% | 2.00 | 276.0 | 0 |
| file_io | 4 | 100% | 2.00 | 293.3 | 0 |
| mock_search | 5 | 100% | 2.00 | 315.4 | 0 |
| context_compression | 4 | 100% | 4.00 | 1089.0 | 8 |
| memory | 2 | 100% | 1.00 | 174.0 | 0 |
| failure_recovery | 8 | 100% | 2.75 | 392.3 | 0 |

## 4.消融结论

1. **上下文压缩**：对 4 条长上下文任务是决定性能力。压缩开启时 4/4 成功；关闭后 0/4 成功，均以 `context_limit` 终止。副作用是平均 token 从 325.0 增至 417.5，因为摘要调用本身有成本。
2. **记忆系统**：影响跨任务偏好任务 SB022。记忆开启时检索到 SB021 保存的偏好并成功；关闭后无法确定语言偏好。当前数据集只有 1 条敏感任务，结论是链路有效，但样本量不足以评估泛化收益。
3. **工具使用**：`scripted-baseline` 不调用工具，绝大多数任务失败；`scripted-strong` 调用工具后在计算、文件、搜索和失败恢复类别全部成功，说明最终答案来自工具 observation。
4. **失败恢复**：8 条失败注入任务全部经历 `ACT(ok=false)` 后由模型调整策略，最终成功。包括文件不存在、除零、路径逃逸、未知工具、指数超限等错误类型。

## 5.至少五个失败案例归因

| 案例 | 配置 | 表现 | 归因 |
|---|---|---|---|
| SB017 | strong，压缩 OFF | `context_limit` | 三个长文件读取结果超过 650 token 预算；关闭压缩后没有滚动摘要，Guard 正确终止，避免无限膨胀。 |
| SB022 | strong，记忆 OFF | 输出“无法确定语言偏好” | SB021 的偏好没有持久化，任务开始时无 memory system prompt；这证明记忆检索是跨任务成功的必要条件。 |
| SB001 | baseline，任意配置 | 输出猜测值 41 | 模型策略未调用 calculator，Agent 无法获得 observation；说明 prompt 不等于执行，工具调用是可靠结果来源。 |
| SB024 | baseline，任意配置 | 除零错误后直接失败 | baseline 不读取 tool error，也没有发起第二轮 corrected call；对比 strong 的 100% 恢复率，验证错误回喂机制的价值。 |
| SB027 | baseline，任意配置 | 未知工具后失败 | baseline 不根据 unknown tool error 切换到 `mock_web_search`；strong 在同一错误反馈后完成替代调用。 |
| SB018 | strong，压缩 OFF | `context_limit` | 与 SB017 同类，长上下文无压缩时预算护栏生效；这不是框架崩溃，而是有状态的受控终止。 |

## 6.GAIA 子集状态

`eval/gaia_subset.json` 目前是授权访问清单，而不是评测结果：

- GAIA 数据集 `gaia-benchmark/GAIA` 是 gated dataset。
- 本机没有 HuggingFace 授权 token，镜像请求返回 `GatedRepo`。
- 项目禁止复制未授权 validation 内容，也禁止伪造 GAIA 指标。
- 获得授权后，可用 `eval/select_gaia.py` 从本地已下载的 validation JSONL 生成 30 条 text-only 子集，再补充真实模型评测。

## 7.复现命令

```powershell
cd D:\SuperAgentHua\02-代码仓库\mini-agent
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe eval\runner.py
```

期望结果：全量测试通过，runner 输出 8 组实验指标。
