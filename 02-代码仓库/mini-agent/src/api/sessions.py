"""会话管理 API — 创建 / 列表 / 删除 / 重命名"""
import uuid
from fastapi import APIRouter, HTTPException
from src.storage.repositories import SessionRepository, MessageRepository

router = APIRouter()


@router.get("/sessions")
async def list_sessions():
    """获取所有会话列表"""
    sessions = await SessionRepository.list_all()
    return {"sessions": sessions, "total": len(sessions)}


@router.post("/sessions")
async def create_session(title: str = "新对话"):
    """创建新会话"""
    session_id = str(uuid.uuid4())[:8]
    session = await SessionRepository.create(session_id, title)
    return session


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取单个会话详情（含消息）"""
    session = await SessionRepository.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = await MessageRepository.get_messages(session_id, limit=100)
    return {
        "session": session,
        "messages": messages,
        "message_count": len(messages)
    }


@router.put("/sessions/{session_id}/title")
async def update_session_title(session_id: str, title: str):
    """重命名会话"""
    session = await SessionRepository.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    await SessionRepository.update_title(session_id, title)
    return {"status": "updated", "title": title}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话及其所有消息"""
    await SessionRepository.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.delete("/sessions")
async def delete_all_sessions():
    """删除所有会话（危险操作）"""
    await SessionRepository.delete_all()
    return {"status": "all_deleted"}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 50):
    """获取会话消息列表"""
    messages = await MessageRepository.get_messages(session_id, limit=limit)
    return {"session_id": session_id, "messages": messages}
