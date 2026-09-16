# 🤖 Mini-Agent

> 一个轻量级 AI 智能体系统，支持知识库 + RAG + 记忆 + 代码执行

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 功能特性

- 🧠 **智能对话** - 基于 LLM 的多轮对话
- 📚 **RAG 知识库** - 支持 PDF/TXT/MD/DOCX 文档检索增强生成
- 💾 **记忆系统** - 短期记忆 + 长期记忆 + 对话摘要
- 🔧 **工具调用** - 代码执行、数学计算、网页搜索
- 🐳 **Docker 部署** - 一键启动，开箱即用
- 🔌 **可扩展** - 模块化设计，易于添加新功能

## 🚀 快速开始

### 方式一：Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/yourname/mini-agent.git
cd mini-agent

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY

# 3. 一键启动
docker-compose up -d --build

# 4. 访问
# API 文档: http://localhost:8000/docs
# 健康检查: http://localhost:8000/health
```

### 方式二：本地开发

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY

# 4. 启动服务
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 项目结构

```
mini-agent/
├── src/                    # 源代码
│   ├── main.py            # FastAPI 应用入口
│   ├── config.py          # 配置管理
│   ├── api/               # API 路由
│   ├── agent/             # Agent 核心
│   ├── llm/               # LLM 抽象层
│   ├── rag/               # RAG 检索系统
│   ├── memory/            # 记忆系统
│   ├── tools/             # 工具系统
│   ├── storage/           # 存储层
│   └── models/            # 数据模型
├── data/                   # 数据目录
├── tests/                  # 测试
├── Dockerfile             # Docker 镜像
├── docker-compose.yml     # Docker 编排
└── requirements.txt       # 依赖
```

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat` | 发送消息 |
| `POST` | `/api/v1/chat/stream` | 流式对话 |
| `POST` | `/api/v1/knowledge/upload` | 上传文档 |
| `GET` | `/api/v1/knowledge/list` | 列出文档 |
| `DELETE` | `/api/v1/knowledge/{id}` | 删除文档 |
| `GET` | `/api/v1/memory/{session_id}` | 获取记忆 |
| `DELETE` | `/api/v1/memory/{session_id}` | 清除记忆 |
| `GET` | `/api/v1/health` | 健康检查 |

## 💬 使用示例

```python
import requests

# 发送消息
response = requests.post("http://localhost:8000/api/v1/chat", json={
    "session_id": "user-123",
    "message": "你好，帮我解释一下什么是 RAG？",
    "use_memory": True,
    "enable_tools": True
})

print(response.json())
```

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI |
| Agent 框架 | LangChain / LangGraph |
| 向量数据库 | ChromaDB |
| LLM | OpenAI API |
| 缓存 | Redis |
| 容器化 | Docker + Compose |

## 📖 文档

- [架构设计](docs/architecture.md)
- [部署指南](docs/deployment.md)
- [API 参考](docs/api_reference.md)

## 🤝 贡献

欢迎贡献代码、报告 Bug 或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目使用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🔗 相关链接

- [LangChain 文档](https://docs.langchain.com/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [ChromaDB 文档](https://docs.trychroma.com/)
