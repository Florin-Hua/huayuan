"""长期记忆：SQLite + sqlite-vec 向量检索与任务后事实抽取。"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from array import array
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import sqlite_vec

from mini_agent.core.types import Message, ModelResponse
from mini_agent.tools.registry import ToolDefinition, tool


class Embedder(Protocol):
    """文本向量接口。"""

    def embed(self, text: str) -> array:
        """把文本转换成固定维度 float32 向量。"""
        ...


class HashingEmbedder:
    """确定性本地向量器：词 + 中文字符/bigram 哈希。"""

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("embedding dimension must be >= 8")
        self.dimension = dimension

    def embed(self, text: str) -> array:
        vector = array("f", [0.0] * self.dimension)
        normalized = text.lower().strip()
        tokens: list[str] = re.findall(r"[a-z0-9_]+", normalized)
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        tokens.extend(cjk)
        tokens.extend(cjk[index : index + 2] for index in range(max(len(cjk) - 1, 0)))

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self.dimension] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm > 0:
            for index, value in enumerate(vector):
                vector[index] = value / norm
        return vector


class MemoryRecord:
    """一条长期记忆的数据库记录。"""

    def __init__(
        self,
        memory_id: int,
        content: str,
        source_run_id: str | None,
        created_at: str,
    ) -> None:
        self.id = memory_id
        self.content = content
        self.source_run_id = source_run_id
        self.created_at = created_at


class MemoryMatch:
    """一条检索命中的记忆。"""

    def __init__(self, record: MemoryRecord, distance: float) -> None:
        self.record = record
        self.distance = distance
        self.score = 1.0 / (1.0 + distance)


class MemoryStore:
    """SQLite 存储记忆正文，sqlite-vec 存储 embedding 并做 KNN 检索。"""

    def __init__(
        self,
        db_path: str | Path,
        embedder: Embedder | None = None,
        dimension: int = 256,
    ) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder(dimension)
        self.dimension = dimension

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source_run_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors "
            f"USING vec0(embedding float[{self.dimension}])"
        )
        self._conn.commit()

    def save(self, content: str, source_run_id: str | None = None) -> MemoryRecord:
        """保存一条记忆并建立向量索引。"""
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        created_at = datetime.now().isoformat(timespec="seconds")
        cursor = self._conn.execute(
            "INSERT INTO memories(content, source_run_id, created_at) VALUES (?, ?, ?)",
            (content, source_run_id, created_at),
        )
        memory_id = int(cursor.lastrowid)
        vector = self.embedder.embed(content)
        self._conn.execute(
            "INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
            (memory_id, sqlite_vec.serialize_float32(vector)),
        )
        self._conn.commit()
        return MemoryRecord(memory_id, content, source_run_id, created_at)

    def list(self) -> list[MemoryRecord]:
        rows = self._conn.execute(
            "SELECT id, content, source_run_id, created_at FROM memories ORDER BY id DESC"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def search(self, query: str, limit: int = 3) -> list[MemoryMatch]:
        """按向量距离检索 top-K 相关记忆。"""
        if limit < 1:
            raise ValueError("memory search limit must be >= 1")
        vector = sqlite_vec.serialize_float32(self.embedder.embed(query))
        rows = self._conn.execute(
            """
            SELECT m.id, m.content, m.source_run_id, m.created_at, v.distance
            FROM memory_vectors v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (vector, limit),
        ).fetchall()
        return [MemoryMatch(self._to_record(row[:-1]), float(row[-1])) for row in rows]

    def delete(self, memory_id: int) -> bool:
        """删除记忆与对应向量。"""
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cursor.rowcount > 0
        self._conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (memory_id,))
        self._conn.commit()
        return deleted

    @staticmethod
    def _to_record(row: tuple[Any, ...]) -> MemoryRecord:
        return MemoryRecord(row[0], row[1], row[2], row[3])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def create_save_memory_tool(memory_store: MemoryStore) -> ToolDefinition:
    """生成绑定当前 MemoryStore 的 save_memory 工具。"""

    @tool
    def save_memory(content: str) -> dict:
        """保存一条需要跨会话记住的事实。"""
        record = memory_store.save(content, source_run_id="model-tool")
        return {"id": record.id, "content": record.content, "saved": True}

    return save_memory


def memory_to_message(matches: list[MemoryMatch]) -> Message | None:
    """把 top-K 记忆转换成注入 system prompt 的消息。"""
    if not matches:
        return None
    lines = [f"- {match.record.content}" for match in matches]
    return Message(
        role="system",
        content="长期记忆（按相关性排序）：\n" + "\n".join(lines),
    )


class MemoryExtractor:
    """任务结束后用一次模型调用判断是否保存长期事实。"""

    def __init__(self, adapter: Any, memory_store: MemoryStore) -> None:
        self.adapter = adapter
        self.memory_store = memory_store

    def extract(self, task: str, answer: str, run_id: str) -> list[str]:
        prompt = [
            Message(
                role="system",
                content=(
                    "你是长期记忆抽取器。请判断本次任务是否包含用户偏好、身份、"
                    "长期事实或后续仍需记住的信息。只输出 JSON，格式："
                    '{"save": true, "memories": ["..."]}'
                ),
            ),
            Message(role="user", content=f"任务：{task}\n最终回答：{answer}"),
        ]
        response: ModelResponse = self.adapter.chat(prompt, [])
        payload = self._parse_response(response.content or "")
        if not payload.get("save"):
            return []
        memories = payload.get("memories", [])
        if not isinstance(memories, list):
            return []
        saved: list[str] = []
        for memory in memories:
            if not isinstance(memory, str) or not memory.strip():
                continue
            record = self.memory_store.save(memory.strip(), source_run_id=run_id)
            saved.append(record.content)
        return saved

    @staticmethod
    def _parse_response(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("memory extractor response has no JSON object")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("memory extractor response must be a JSON object")
        return payload
