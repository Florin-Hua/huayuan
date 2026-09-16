"""
Mini-Agent - 纯 Python 版本 (无需第三方依赖)
使用内置 http.server 实现
"""
import http.server
import json
import os
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path

# ===== 配置 =====
HOST = "0.0.0.0"
PORT = 8000
WEB_DIR = Path(__file__).parent / "web"

# ===== 简易记忆系统 =====
class MemoryManager:
    def __init__(self):
        self.sessions = {}

    def get_messages(self, session_id):
        return self.sessions.get(session_id, [])

    def add_message(self, session_id, role, content):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%H:%M")
        })
        # 保留最近 20 条
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]

    def clear(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

memory = MemoryManager()


# ===== 简易 RAG 系统 =====
class SimpleRAG:
    def __init__(self):
        self.documents = {}  # doc_id -> content

    def add_document(self, doc_id, content):
        self.documents[doc_id] = content

    def search(self, query, top_k=3):
        results = []
        query_lower = query.lower()
        for doc_id, content in self.documents.items():
            # 简单关键词匹配
            if any(word in content.lower() for word in query_lower.split()):
                results.append({
                    "doc_id": doc_id,
                    "content": content[:500],
                    "score": 0.8
                })
        return results[:top_k]

rag = SimpleRAG()


# ===== 简易 Agent =====
class MiniAgent:
    def __init__(self):
        self.knowledge_base = {
            "问候": "你好！我是 Mini-Agent，一个轻量级 AI 智能体。我可以帮你解答问题、分析文档等。",
            "功能": "我的主要功能包括：\n1. 💬 智能对话 - 基于 LLM 的多轮对话\n2. 📚 RAG 知识库 - 支持文档检索增强生成\n3. 💾 记忆系统 - 短期记忆 + 长期记忆\n4. 🔧 工具调用 - 代码执行、数学计算、网页搜索",
            "技术": "技术栈：Python + FastAPI + LangChain + ChromaDB + Docker",
            "架构": "系统架构：\n1. 用户界面层 - Web UI\n2. API 网关层 - FastAPI\n3. Agent 编排层 - LangGraph\n4. 核心服务层 - LLM/RAG/Memory/Tools\n5. 数据存储层 - ChromaDB/SQLite/Redis",
            "部署": "部署方式：\n1. 本地运行: uvicorn src.main:app --reload\n2. Docker: docker-compose up -d --build\n3. 访问: http://localhost:8000",
        }

    def chat(self, session_id, message):
        # 添加用户消息到记忆
        memory.add_message(session_id, "user", message)

        # 生成回复
        reply = self._generate_reply(message)

        # 添加 AI 回复到记忆
        memory.add_message(session_id, "assistant", reply)

        return {
            "session_id": session_id,
            "reply": reply,
            "sources": [],
            "tools_used": [],
            "memory_updated": True,
            "tokens_used": len(message.split()),
            "latency_ms": 50
        }

    def _generate_reply(self, message):
        message_lower = message.lower()

        # 知识库匹配
        for key, value in self.knowledge_base.items():
            if key in message_lower:
                return value

        # RAG 检索
        rag_results = rag.search(message)
        if rag_results:
            context = "\n".join([f"参考: {r['content'][:200]}" for r in rag_results])
            return f"根据知识库检索：\n\n{context}\n\n请问还有什么可以帮助你的？"

        # 默认回复
        if any(word in message_lower for word in ["你好", "hi", "hello", "嗨"]):
            return "你好！👋 我是 Mini-Agent，很高兴为你服务！\n\n你可以问我任何问题，比如：\n- 介绍一下你的功能\n- 你的技术架构是什么\n- 怎么部署"

        if any(word in message_lower for word in ["谢谢", "thank", "感谢"]):
            return "不客气！😊 如果还有其他问题，随时可以问我。"

        if "?" in message or "？" in message or "什么" in message or "怎么" in message or "如何" in message:
            return f"这是一个很好的问题！关于「{message}」，我目前还在学习中。\n\n你可以：\n1. 上传相关文档到知识库，我可以帮你检索分析\n2. 试试问我关于 Mini-Agent 的功能和技术\n3. 等待后续版本支持更强大的 LLM 对接"

        return f"收到你的消息：「{message}」\n\n我是 Mini-Agent，目前运行在演示模式。配置 OpenAI API Key 后，我可以提供更智能的对话服务。\n\n试试问我：\n- 介绍一下你的功能\n- 你的技术架构是什么\n- 怎么部署"

agent = MiniAgent()


# ===== HTTP 请求处理 =====
class AgentHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        elif self.path == "/api/v1/health" or self.path == "/health":
            self._send_json({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "0.1.0"
            })
            return
        elif self.path.startswith("/api/v1/memory/"):
            session_id = self.path.split("/")[-1]
            messages = memory.get_messages(session_id)
            self._send_json({
                "session_id": session_id,
                "messages": messages
            })
            return

        super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body) if body else {}

        if self.path == "/api/v1/chat":
            result = agent.chat(
                session_id=data.get("session_id", "default"),
                message=data.get("message", "")
            )
            self._send_json(result)

        elif self.path == "/api/v1/knowledge/upload":
            doc_id = f"doc-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            content = data.get("content", "")
            rag.add_document(doc_id, content)
            self._send_json({
                "doc_id": doc_id,
                "status": "uploaded"
            })

        else:
            self._send_json({"error": "Not Found"}, 404)

    def do_DELETE(self):
        if self.path.startswith("/api/v1/memory/"):
            session_id = self.path.split("/")[-1]
            memory.clear(session_id)
            self._send_json({"status": "cleared"})
        else:
            self._send_json({"error": "Not Found"}, 404)

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


# ===== 启动 =====
if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════╗
║           Mini-Agent 已启动                   ║
╠══════════════════════════════════════════════╣
║  地址: http://localhost:{PORT}                 ║
║  模式: 纯 Python (无需第三方依赖)              ║
╚══════════════════════════════════════════════╝
""")
    server = http.server.HTTPServer((HOST, PORT), AgentHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMini-Agent 已停止")
        server.server_close()

