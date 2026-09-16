"""Mini-Agent 应用入口 — FastAPI (lifespan 模式)"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from src.config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================================================== #
#  Lifespan (替代 deprecated on_event)
# ====================================================================== #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理"""
    # ---- 启动 ----
    from src.storage.database import db
    await db.init()

    logger.info(
        f"Mini-Agent v0.2.0 已启动 | "
        f"http://{settings.host}:{settings.port} | "
        f"LLM: {settings.llm_provider}/{settings.llm_model}"
    )
    yield
    # ---- 关闭 ----
    logger.info("Mini-Agent 关闭")


# ====================================================================== #
#  FastAPI 应用
# ====================================================================== #

app = FastAPI(
    title="Mini-Agent API",
    description="轻量级 AI 智能体系统 — 支持 RAG 知识检索 / 多轮记忆 / 工具调用",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================================== #
#  注册 API 路由
# ====================================================================== #
from src.api import health, chat, knowledge, memory, sessions, tools_router

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(knowledge.router, prefix="/api/v1", tags=["Knowledge"])
app.include_router(memory.router, prefix="/api/v1", tags=["Memory"])
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
app.include_router(tools_router.router, prefix="/api/v1", tags=["Tools"])


# ====================================================================== #
#  前端静态文件
# ====================================================================== #

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """提供前端页面"""
    html_path = os.path.join(os.path.dirname(__file__), "..", "web", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Mini-Agent</h1><p>前端未找到。访问 <a href='/docs'>/docs</a> 查看 API。</p>",
        status_code=404
    )


# ====================================================================== #
#  直接运行
# ====================================================================== #
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
