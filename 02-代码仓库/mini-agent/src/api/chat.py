"""聊天 API — 对接 ReAct Agent + SSE 流式"""
import json
import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.models.schemas import ChatRequest, ChatResponse
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局 Agent 实例 (延迟初始化)
_agent = None


def get_agent():
    """获取或初始化 Agent"""
    global _agent
    if _agent is None:
        from src.agent.core import MiniAgent
        _agent = MiniAgent(
            llm_provider=settings.llm_provider,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
        )
        logger.info(f"Agent 初始化完成: {settings.llm_provider}/{settings.llm_model}")
    return _agent


# ====================================================================== #
#  非流式对话
# ====================================================================== #

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    发送消息并获取 AI 回复（非流式）

    - session_id: 会话 ID
    - message: 用户消息
    - use_memory: 是否启用记忆
    - enable_tools: 是否启用工具
    """
    start_time = time.time()

    try:
        agent = get_agent()
        result = await agent.chat(
            session_id=request.session_id,
            message=request.message,
            use_memory=request.use_memory,
            enable_tools=request.enable_tools,
        )
        return ChatResponse(**result)

    except Exception as e:
        logger.exception("chat error")
        latency_ms = (time.time() - start_time) * 1000
        # 降级：无 API Key 或 LLM 错误时返回提示
        return ChatResponse(
            session_id=request.session_id,
            reply=(
                f"⚠️ 处理消息时出错: {e}\n\n"
                f"请检查 .env 中的 OPENAI_API_KEY 是否已配置。"
            ),
            sources=[],
            tools_used=[],
            memory_updated=False,
            tokens_used=0,
            latency_ms=latency_ms,
        )


# ====================================================================== #
#  流式对话 (SSE)
# ====================================================================== #

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口 — Server-Sent Events

    事件类型：
    - content:     {"type": "content", "content": "..."}         流式文本
    - tool_call:   {"type": "tool_call", "tool": "...", "args": {...}}  工具调用
    - tool_result: {"type": "tool_result", "tool": "...", "result": "..."}  工具结果
    - done:        {"type": "done", "reply": "...", "tools_used": [...]}  完成
    - error:       {"type": "error", "message": "..."}          错误
    """

    async def event_generator():
        try:
            agent = get_agent()
            async for event_str in agent.chat_stream(
                session_id=request.session_id,
                message=request.message,
                use_memory=request.use_memory,
                enable_tools=request.enable_tools,
            ):
                yield f"data: {event_str}\n\n"

        except Exception as e:
            logger.exception("chat_stream error")
            error_event = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )
