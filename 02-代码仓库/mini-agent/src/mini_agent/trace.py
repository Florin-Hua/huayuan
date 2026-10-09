"""SQLite Trace 存储：run 汇总与逐步执行轨迹。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mini_agent.core.types import AgentRun


class TraceStore:
    """把 AgentRun / StepLog 持久化为可查询的执行轨迹。

    Trace 与 Memory 分库保存：前者是运行事实，后者是用户长期事实，
    生命周期和清理策略不同。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT,
                    iterations INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    parse_retries INTEGER NOT NULL,
                    termination_reason TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS trace_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    iteration INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    tool_name TEXT,
                    tool_args TEXT,
                    result_status TEXT,
                    error TEXT,
                    note TEXT,
                    tokens INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_trace_steps_run_id
                    ON trace_steps(run_id, step);
                """
            )

    def save_run(self, run: AgentRun) -> None:
        """保存或覆盖一次 run；重复调用以最新 AgentRun 为准。"""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, task, status, answer, iterations,
                    prompt_tokens, completion_tokens, total_tokens,
                    parse_retries, termination_reason, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    task=excluded.task,
                    status=excluded.status,
                    answer=excluded.answer,
                    iterations=excluded.iterations,
                    prompt_tokens=excluded.prompt_tokens,
                    completion_tokens=excluded.completion_tokens,
                    total_tokens=excluded.total_tokens,
                    parse_retries=excluded.parse_retries,
                    termination_reason=excluded.termination_reason,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at
                """,
                (
                    run.run_id,
                    run.task,
                    run.status.value,
                    run.answer,
                    run.iterations,
                    run.usage.prompt_tokens,
                    run.usage.completion_tokens,
                    run.usage.total,
                    run.parse_retries,
                    run.termination_reason,
                    datetime.now(timezone.utc).isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                ),
            )
            self._conn.execute("DELETE FROM trace_steps WHERE run_id = ?", (run.run_id,))
            self._conn.executemany(
                """
                INSERT INTO trace_steps (
                    run_id, step, iteration, state, tool_name, tool_args,
                    result_status, error, note, tokens, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._step_row(run.run_id, index, step) for index, step in enumerate(run.steps, start=1)],
            )

    def _step_row(self, run_id: str, step: int, log: Any) -> tuple[Any, ...]:
        result_status = None if log.ok is None else ("ok" if log.ok else "error")
        return (
            run_id,
            step,
            log.iteration,
            log.state,
            log.tool_name,
            json.dumps(log.tool_args, ensure_ascii=False, sort_keys=True)
            if log.tool_args is not None
            else None,
            result_status,
            log.error,
            log.note,
            log.tokens,
            log.duration_ms,
            datetime.now(timezone.utc).isoformat(),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT step, iteration, state, tool_name, tool_args,
                   result_status, error, note, tokens, duration_ms, created_at
            FROM trace_steps
            WHERE run_id = ?
            ORDER BY step
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT run_id, task, status, iterations, total_tokens, finished_at
            FROM runs
            ORDER BY started_at DESC, run_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TraceStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
