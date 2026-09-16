# MiniAgent

从零自研的 mini agent 框架。当前进度：阶段1（最小 ReAct 循环）。

## 快速开始

```powershell
uv venv .venv
uv pip install -e ".[dev]"
copy .env.example .env   # 填入模型配置
mini run "计算 23*47+11，再读取 data/notes.txt，把两个结果分别告诉我" --sandbox .
```

也可以直接运行模块：

```powershell
.venv\Scripts\python.exe -m mini_agent.cli run "读取 missing_file.txt 并告诉我内容" --sandbox data
```

## 测试

```powershell
.venv\Scripts\python.exe -m pytest
```

阶段1结果：`11 passed`。测试使用 ScriptedAdapter，不访问网络。

## 阶段1范围

- 手写 ReAct 循环：THINK -> ACT -> OBSERVE，默认最大迭代 10
- 硬编码工具：calculator（AST 安全求值）、read_text_file（沙箱限制）
- 工具错误结构化回喂，模型可继续自愈
- 三种终态：SUCCESS / MAX_ITER / PARSE_ERROR
- CLI：rich 表格展示执行轨迹

## 当前验收说明

单元测试全部通过；真实 OpenAI 调用当前出现 `APITimeoutError`，已按 `PARSE_ERROR` 处理。网络恢复后按根仓库 `01-阶段进程/第1阶段-最小闭环.md` 中的命令重试。

完整设计见仓库根 `01-阶段进程/` 各阶段文档。