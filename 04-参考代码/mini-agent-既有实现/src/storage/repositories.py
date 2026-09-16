"""数据访问层 — 会话 / 消息 / 文档 CRUD"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.storage.database import db

logger = logging.getLogger(__name__)


# ====================================================================== #
#  会话仓库
# ====================================================================== #

class SessionRepository:

    @staticmethod
    async def create(session_id: str, title: str = "新对话") -> Dict:
        now = datetime.now().isoformat()
        async with await db.get_connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now)
            )
            await conn.commit()
        return {"id": session_id, "title": title, "created_at": now}

    @staticmethod
    async def get(session_id: str) -> Optional[Dict]:
        async with await db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row:
                return dict(row)
            return None

    @staticmethod
    async def list_all() -> List[Dict]:
        async with await db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT s.*, "
                "  (SELECT COUNT(*) FROM messages WHERE session_id = s.id) as message_count "
                "FROM sessions s ORDER BY s.updated_at DESC"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    async def update_title(session_id: str, title: str):
        async with await db.get_connection() as conn:
            await conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, datetime.now().isoformat(), session_id)
            )
            await conn.commit()

    @staticmethod
    async def delete(session_id: str):
        async with await db.get_connection() as conn:
            await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await conn.commit()

    @staticmethod
    async def delete_all():
        async with await db.get_connection() as conn:
            await conn.execute("DELETE FROM messages")
            await conn.execute("DELETE FROM sessions")
            await conn.commit()


# ====================================================================== #
#  消息仓库
# ====================================================================== #

class MessageRepository:

    @staticmethod
    async def add(session_id: str, role: str, content: str,
                  tool_calls: List = None, tokens_used: int = 0) -> int:
        now = datetime.now().isoformat()
        async with await db.get_connection() as conn:
            cursor = await conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tokens_used, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, json.dumps(tool_calls or []), tokens_used, now)
            )
            # 更新 session 的 updated_at
            await conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
            await conn.commit()
            return cursor.lastrowid

    @staticmethod
    async def get_messages(session_id: str, limit: int = 50) -> List[Dict]:
        async with await db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT role, content, tool_calls, tokens_used, created_at "
                "FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
            rows = await cursor.fetchall()
            messages = []
            for r in reversed(rows):
                msg = dict(r)
                msg["tool_calls"] = json.loads(msg["tool_calls"]) if msg["tool_calls"] else []
                messages.append(msg)
            return messages

    @staticmethod
    async def get_context_messages(session_id: str, limit: int = 20) -> List[Dict]:
        """获取用于 LLM 上下文的消息格式"""
        messages = await MessageRepository.get_messages(session_id, limit=limit)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    @staticmethod
    async def count(session_id: str) -> int:
        async with await db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    @staticmethod
    async def clear(session_id: str):
        async with await db.get_connection() as conn:
            await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            await conn.commit()


# ====================================================================== #
#  文档仓库
# ====================================================================== #

class DocumentRepository:

    @staticmethod
    async def add(doc_id: str, filename: str, file_path: str = "",
                  file_size: int = 0, file_type: str = "",
                  chunk_count: int = 0, text_length: int = 0,
                  tags: str = "") -> Dict:
        now = datetime.now().isoformat()
        async with await db.get_connection() as conn:
            await conn.execute(
                "INSERT INTO documents (id, filename, file_path, file_size, file_type, "
                "chunk_count, text_length, tags, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, filename, file_path, file_size, file_type,
                 chunk_count, text_length, tags, now)
            )
            await conn.commit()
        return {"id": doc_id, "filename": filename}

    @staticmethod
    async def list_all() -> List[Dict]:
        async with await db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    async def delete(doc_id: str):
        async with await db.get_connection() as conn:
            await conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await conn.commit()
