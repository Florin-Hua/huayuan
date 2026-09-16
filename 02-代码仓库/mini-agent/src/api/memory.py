"""记忆管理 API — 查询 / 清除 / 事实管理"""
from fastapi import APIRouter, HTTPException
from src.config import settings

router = APIRouter()

# 全局 Agent 引用
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from src.api.chat import get_agent
        _agent = get_agent()
    return _agent


@router.get("/memory/{session_id}")
async def get_memory(session_id: str):
    """获取会话记忆（短期 + 长期事实 + 摘要）"""
    try:
        agent = _get_agent()
        session = agent.memory.get_session(session_id)
        stats = agent.memory.get_stats(session_id)

        return {
            "session_id": session_id,
            "summary": session.get("summary", ""),
            "facts": session.get("facts", []),
            "message_count": len(session.get("messages", [])),
            "stats": stats
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "summary": "",
            "facts": [],
            "message_count": 0,
            "stats": {},
            "error": str(e)
        }


@router.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """清除会话记忆"""
    try:
        agent = _get_agent()
        agent.memory.clear_session(session_id)
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{session_id}/summary")
async def get_memory_summary(session_id: str):
    """获取会话摘要"""
    agent = _get_agent()
    session = agent.memory.get_session(session_id)
    return {
        "session_id": session_id,
        "summary": session.get("summary", "暂无摘要")
    }


@router.post("/memory/{session_id}/facts")
async def add_fact(session_id: str, fact: str):
    """手动添加长期记忆事实"""
    agent = _get_agent()
    agent.memory.add_fact(session_id, fact)
    return {"status": "added", "fact": fact}


@router.get("/memory/stats")
async def get_memory_stats():
    """获取全局记忆统计"""
    agent = _get_agent()
    return {
        "total_sessions": len(agent.memory.sessions),
        "sessions": {
            sid: agent.memory.get_stats(sid)
            for sid in list(agent.memory.sessions.keys())[:20]
        }
    }
