# Mini-Agent — 轻量级 AI 智能体系统

> 一个可本地运行、支持知识库 + RAG + 记忆 + 代码执行的 AI Agent，Docker 一键部署，GitHub 开箱即用。

---

## 项目定位

| 维度 | 说明 |
|------|------|
| **目标** | 构建一个轻量但功能完整的 AI 智能体，具备 RAG 知识检索、多轮记忆、代码执行、工具调用等核心能力 |
| **定位** | 面向个人开发者/小团队的「可学习、可扩展」Agent 脚手架，非生产级平台 |
| **技术栈** | Python 3.11 + FastAPI + LangChain/LangGraph + ChromaDB + OpenAI API + Docker |
| **差异化** | 单仓部署、一键启动、模块清晰、易于二次开发 |

---

## 系统架构总览

```
+--------------------------------------------------------------+
|                        用户界面层                              |
|         Web UI (Gradio / Streamlit / React)                  |
|              或 CLI 命令行交互                                 |
+----------------------------+---------------------------------+
                             | HTTP / WebSocket
                             v
+--------------------------------------------------------------+
|                     API 网关层 (FastAPI)                       |
|  /chat   /rag   /upload   /memory/history   /health          |
+----------------------------+---------------------------------+
                             |
                             v
+--------------------------------------------------------------+
|                    Agent 编排层 (LangGraph)                    |
|                                                              |
|  意图识别 --> 路由分发 --> 任务执行                               |
|                                                              |
|         ReAct 循环 (推理 --> 行动 --> 观察)                     |
+-------+----------+----------+--------------------------------+
        |          |          |
        v          v          v
+--------------------------------------------------------------+
|  记忆系统       RAG 检索      工具系统       LLM 推理核心        |
|  Memory       Retriever     Tools        (OpenAI / 本地)     |
+-------+----------+----------+--------------------------------+
        |          |          |
        v          v          v
+--------------------------------------------------------------+
|                      数据存储层                                |
|  ChromaDB(向量)  SQLite(元数据)  文件系统  Redis(会话缓存)      |
+--------------------------------------------------------------+
```

---

## 核心模块详细设计

### 模块 1：LLM 推理核心

- 使用 LangChain BaseChatModel 做统一抽象
- 支持 streaming=True 流式返回
- Function Calling 解析为结构化工具调用
- 可通过 .env 切换不同 LLM 提供商
- Token 计数与预算控制

支持的 LLM 提供商：
- OpenAI GPT-4o / GPT-4o-mini
- 本地模型 Ollama (Qwen, Llama, etc.)
- 其他兼容 API (DeepSeek, Moonshot, etc.)

---

### 模块 2：RAG 知识检索系统

Pipeline 流程：
文档上传 --> 文件解析(PDF/TXT/MD/DOCX/URL) --> 文本分块(Chunking) --> 向量化(Embedding) --> ChromaDB 存储

查询流程：
用户问题 --> 问题向量化 --> 向量相似度 Top-K 检索 --> 重排序(Rerank) --> 构造增强上下文 --> LLM 生成

分块策略：
- 固定长度分块: chunk_size=512, overlap=50 (通用文本)
- 语义分块: 基于句子相似度断句 (结构化文档)
- 递归字符分块: 按段落->句子->字符递归 (混合内容)
- Markdown感知分块: 按标题层级切分 (Markdown文档)

技术细节：
- 文档解析: PyPDF2/pdfplumber(PDF), python-docx(Word), markdown(MD)
- 向量化模型: text-embedding-3-small(OpenAI) 或 bge-small-zh(本地)
- 检索策略: 混合检索(BM25 + Dense)、MMR 多样性去重
- 元数据过滤: 按文档来源、上传时间、标签过滤

---

### 模块 3：记忆系统

短期记忆 (Working Memory):
- 当前对话上下文 (最近 N 轮)
- 当前任务状态
- 存储位置: 内存 (List[Message])

长期记忆 (Long-term Memory):
- 用户画像/偏好 (JSON)
- 历史对话摘要 (向量化存储)
- 重要事实提取 (Entity Memory)
- 存储位置: SQLite + ChromaDB

会话记忆 (Session Memory):
- Session ID -> 对话历史映射
- TTL 过期清理
- 存储位置: Redis / SQLite

记忆更新策略:
- 每 N 轮对话自动摘要压缩
- 关键事实提取存入长期记忆
- 相似问题自动关联历史记忆

---

### 模块 4：工具系统

内置工具：
- 代码执行工具: Python 沙箱, 文件操作, 超时控制
- 搜索工具: DuckDuckGo, 自定义 API, 知识库搜索
- 计算工具: 数学表达式, 单位换算, 数据分析
- 文件处理工具: PDF 生成, CSV 解析, 图片处理

工具注册与发现:
- @tool decorator 自动注册到工具注册表
- 支持: 本地工具 / MCP 协议工具 / HTTP API 工具

---

### 模块 5：Agent 编排层 (LangGraph)

状态定义:
- messages: List[Message]  消息历史
- context: str             RAG 上下文
- memory: dict             记忆状态
- tools_used: List[str]    已使用工具
- iteration: int           当前迭代次数

节点:
- parse_input: 解析输入
- classify_intent: 意图分类
- rag_retrieve: RAG 检索
- tool_call: 工具调用
- generate: 生成回复
- update_memory: 更新记忆

条件路由:
- 意图分类后路由到 chat / rag / tool 三个分支
- 最终都汇聚到 generate --> update_memory --> END

---

## 项目目录结构

```
mini-agent/
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── main.py               # FastAPI 应用入口
│   ├── config.py             # 配置管理
│   │
│   ├── api/                  # API 路由层
│   │   ├── chat.py           # /chat 聊天接口
│   │   ├── knowledge.py      # /knowledge 知识库管理
│   │   ├── memory.py         # /memory 记忆管理
│   │   ├── health.py         # /health 健康检查
│   │   └── websocket.py      # WebSocket 实时通信
│   │
│   ├── agent/                # Agent 核心
│   │   ├── core.py           # Agent 主循环 (ReAct)
│   │   ├── graph.py          # LangGraph 状态图定义
│   │   ├── nodes.py          # 图节点实现
│   │   ├── edges.py          # 边/路由逻辑
│   │   └── prompts.py        # 系统提示词管理
│   │
│   ├── llm/                  # LLM 抽象层
│   │   ├── base.py           # 基类
│   │   ├── openai_provider.py
│   │   ├── ollama_provider.py
│   │   └── factory.py        # LLM 工厂模式
│   │
│   ├── rag/                  # RAG 检索系统
│   │   ├── pipeline.py       # RAG 管道主逻辑
│   │   ├── embeddings.py     # Embedding 模型封装
│   │   ├── chunker.py        # 文档分块策略
│   │   ├── retriever.py      # 检索器
│   │   ├── reranker.py       # 重排序器
│   │   └── ingest.py         # 文档入库流程
│   │
│   ├── memory/               # 记忆系统
│   │   ├── manager.py        # 记忆管理器
│   │   ├── short_term.py     # 短期记忆
│   │   ├── long_term.py      # 长期记忆
│   │   ├── session.py        # 会话管理
│   │   └── summarizer.py     # 对话摘要压缩
│   │
│   ├── tools/                # 工具系统
│   │   ├── registry.py       # 工具注册表
│   │   ├── python_executor.py
│   │   ├── web_search.py
│   │   ├── calculator.py
│   │   └── file_ops.py
│   │
│   ├── storage/              # 存储层
│   │   ├── vector_store.py   # 向量数据库封装
│   │   ├── document_store.py # 文档存储
│   │   └── cache.py          # 缓存层
│   │
│   └── models/               # 数据模型
│       ├── schemas.py        # Pydantic 数据模型
│       ├── database.py       # 数据库连接
│       └── entities.py       # ORM 模型
│
├── web/                      # 前端 (可选)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── notebooks/                # Jupyter Notebook 教程
│   ├── 01_quickstart.ipynb
│   ├── 02_rag_demo.ipynb
│   └── 03_agent_custom.ipynb
│
├── tests/                    # 测试
│   ├── conftest.py
│   ├── test_agent.py
│   ├── test_rag.py
│   ├── test_memory.py
│   └── test_tools.py
│
├── docs/                     # 文档
│   ├── architecture.md
│   ├── deployment.md
│   └── api_reference.md
│
├── scripts/                  # 脚本工具
│   ├── setup.sh
│   ├── seed_data.py
│   └── benchmark.py
│
└── data/                     # 数据目录 (gitignore)
    ├── uploads/
    ├── vector_db/
    └── sqlite/
```

---

## Docker 部署设计

### Dockerfile

```dockerfile
# 阶段 1: 构建
FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 阶段 2: 运行
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY src/ ./src/
COPY web/ ./web/
RUN mkdir -p /app/data/{uploads,vector_db,sqlite}
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: "3.8"
services:
  agent:
    build: .
    container_name: mini-agent
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env:ro
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-3-small}
      - VECTOR_DB_PATH=/app/data/vector_db
      - DB_PATH=/app/data/sqlite/agent.db
    restart: unless-stopped
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    container_name: mini-agent-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

volumes:
  redis_data:
```

### .env.example

```bash
# LLM 配置
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1

# Embedding 配置
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# RAG 配置
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=5

# 记忆配置
SHORT_TERM_MAX_TURNS=20
SUMMARY_INTERVAL=10
MEMORY_SEARCH_TOP_K=3

# 服务配置
HOST=0.0.0.0
PORT=8000
DEBUG=false
CORS_ORIGINS=["http://localhost:3000"]

# Redis
REDIS_URL=redis://redis:6379/0

# 数据路径
VECTOR_DB_PATH=/app/data/vector_db
DB_PATH=/app/data/sqlite/agent.db
UPLOAD_DIR=/app/data/uploads
```

---

## 技术栈清单

| 层次 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 语言 | Python | 3.11+ | 核心开发语言 |
| Web框架 | FastAPI | 0.115+ | API 服务 |
| ASGI服务器 | Uvicorn | 0.30+ | 高性能 ASGI |
| Agent框架 | LangGraph | 0.2+ | Agent 状态图编排 |
| LLM封装 | LangChain | 0.3+ | LLM 统一接口 |
| 向量数据库 | ChromaDB | 0.5+ | 向量存储与检索 |
| Embedding | OpenAI / BGE | - | 文本向量化 |
| 会话缓存 | Redis | 7+ | 会话与缓存 |
| 数据库 | SQLite | 3+ | 元数据持久化 |
| 文档解析 | PyPDF2 / pdfplumber | - | PDF 解析 |
| 文档分块 | LangChain TextSplitters | - | 文本分块 |
| 前端 | Gradio / Streamlit | - | 可视化界面 |
| 容器化 | Docker + Compose | - | 部署与编排 |
| 环境管理 | Pydantic Settings | - | 配置管理 |

---

## 快速启动流程

```bash
# 1. 克隆项目
git clone https://github.com/yourname/mini-agent.git
cd mini-agent

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 3. 一键启动 (Docker)
docker-compose up -d --build

# 4. 访问
# API 文档: http://localhost:8000/docs
# 健康检查: http://localhost:8000/health

# 5. (可选) 本地开发模式
pip install -r requirements.txt
uvicorn src.main:app --reload
```

---

## API 接口设计

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/chat | 发送消息获取回复 |
| POST | /api/v1/chat/stream | 流式对话 |
| POST | /api/v1/knowledge/upload | 上传文档到知识库 |
| GET | /api/v1/knowledge/list | 列出知识库文档 |
| DELETE | /api/v1/knowledge/{id} | 删除知识库文档 |
| GET | /api/v1/memory/{session_id} | 获取会话记忆 |
| DELETE | /api/v1/memory/{session_id} | 清除会话记忆 |
| GET | /api/v1/health | 健康检查 |
| WS | /api/v1/ws/{session_id} | WebSocket 实时对话 |

请求/响应示例：

```json
// POST /api/v1/chat
// Request
{
  "session_id": "user-123",
  "message": "帮我分析一下这个PDF里的关键数据",
  "context": {
    "knowledge_base": "default",
    "use_memory": true,
    "enable_tools": true
  }
}

// Response
{
  "session_id": "user-123",
  "reply": "根据文档分析，关键数据如下：...",
  "sources": [
    {"doc_id": "doc-001", "page": 3, "score": 0.92},
    {"doc_id": "doc-001", "page": 7, "score": 0.87}
  ],
  "tools_used": ["knowledge_search"],
  "memory_updated": true,
  "tokens_used": 1520,
  "latency_ms": 2340
}
```

---

## 开发路线图

### Phase 1: MVP (核心功能) - 2 周
- 项目骨架搭建
- FastAPI 基础服务
- LLM 调用封装 (OpenAI)
- 基础对话接口
- Docker 容器化

### Phase 2: RAG 知识库 - 1 周
- 文档上传与解析
- 文本分块策略
- ChromaDB 向量存储
- 检索与生成

### Phase 3: 记忆系统 - 1 周
- 短期记忆管理
- 长期记忆持久化
- 对话摘要压缩
- 会话管理

### Phase 4: 工具与Agent - 1 周
- 工具注册系统
- 代码执行沙箱
- ReAct Agent 循环
- LangGraph 编排

### Phase 5: 优化与发布 - 1 周
- 前端界面 (Gradio)
- 测试覆盖
- 文档编写
- GitHub 发布

---

## 关键依赖 (requirements.txt)

```txt
# Web 框架
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
websockets>=12.0

# LLM & Agent
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-core>=0.3.0
langgraph>=0.2.0
openai>=1.50.0

# RAG & 向量数据库
chromadb>=0.5.0
sentence-transformers>=3.0.0
pypdf2>=3.0.0
pdfplumber>=0.11.0
python-docx>=1.0.0
markdown>=3.6

# 记忆 & 缓存
redis>=5.0.0
aiosqlite>=0.20.0

# 工具
duckduckgo-search>=6.0.0

# 配置 & 数据模型
pydantic>=2.8.0
pydantic-settings>=2.4.0
python-dotenv>=1.0.0

# Token 计数
tiktoken>=0.7.0
aiofiles>=24.0.0

# 前端 (可选)
gradio>=4.40.0
```

---

## 扩展方向

| 方向 | 说明 |
|------|------|
| 多模态 | 支持图片/音频输入，集成 Whisper + DALL-E |
| 多Agent | 多角色协作，任务委派 |
| MCP协议 | 接入 MCP 服务器，支持外部工具生态 |
| RAG优化 | GraphRAG、Agentic RAG、Self-RAG |
| 评估体系 | RAGAS 评估、Agent 可观测性 |
| 微调支持 | 基于对话数据微调私有模型 |
| 权限控制 | 用户认证、角色权限、数据隔离 |

---

*本文档为 Mini-Agent 项目的完整技术规划，涵盖架构设计、模块拆解、技术选型、部署方案等全部内容。*
