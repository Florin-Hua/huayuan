"""SQLite 数据库连接管理"""
import os
import aiosqlite
import logging
from src.config import settings

logger = logging.getLogger(__name__)

# 数据库初始化 SQL
INIT_SQL = """
-- 会话表
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT DEFAULT '新对话',
    model TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 消息表
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT DEFAULT '[]',
    tokens_used INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

-- 知识库文档元数据
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT,
    file_size INTEGER DEFAULT 0,
    file_type TEXT DEFAULT '',
    chunk_count INTEGER DEFAULT 0,
    text_length INTEGER DEFAULT 0,
    tags TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

-- 长期记忆事实
CREATE TABLE IF NOT EXISTS memory_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    created_at TEXT NOT NULL
);
"""


class Database:
    """异步 SQLite 数据库管理"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.db_path
        self._initialized = False

    async def init(self):
        """初始化数据库（创建表）"""
        if self._initialized:
            return
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(INIT_SQL)
            await db.commit()
        self._initialized = True
        logger.info(f"数据库初始化完成: {self.db_path}")

    async def get_connection(self) -> aiosqlite.Connection:
        """获取数据库连接"""
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        return db


# 全局数据库实例
db = Database()
