# MiniAgent

从零自研的 mini agent 框架。当前进度：阶段2（工具系统）。

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

## 测试

```powershell
.venv\Scripts\python.exe -m pytest
```

当前结果：`21 passed`。测试使用 ScriptedAdapter，不访问网络。

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

## 阶段2范围

- `@tool` 装饰器与 Pydantic schema 自动生成
- `ToolRegistry`：注册、查找、schema 输出、统一执行
- 默认 30 秒工具超时，可通过 `.env` 配置
- 内置工具：calculator、read_text_file、write_text_file、mock_web_search
- 沙箱路径逃逸拦截
- 多步任务：读取 -> 写入 -> 读回验证

## 当前验收说明

单元测试全部通过；真实 OpenAI 调用当前因 `api.openai.com` 网络超时被阻断。框架将该异常处理为 `PARSE_ERROR`。网络恢复后按根仓库 `01-阶段进程/第2阶段-工具系统.md` 中的命令重试。

完整设计见仓库根 `01-阶段进程/` 各阶段文档。