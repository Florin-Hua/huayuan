"""阶段5单元测试：SQLite + sqlite-vec 记忆写入、检索、删除与跨会话注入。"""
from __future__ import annotations

from collections import deque
from pathlib import Path

from mini_agent.cli import build_parser
from mini_agent.config import Settings
from mini_agent.core.agent import Agent
from mini_agent.core.types import ModelResponse, ToolCall, Usage
from mini_agent.memory import (
    MemoryExtractor,
    MemoryStore,
    create_save_memory_tool,
    memory_to_message,
)
from mini_agent.tools.registry import ToolRegistry


class ScriptedAdapter:
    """按剧本返回模型输出，并记录最后一次看到的上下文。"""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = deque(responses)
        self.seen_messages = []
        self.all_calls = []

    def chat(self, messages, tools):
        self.seen_messages = list(messages)
        self.all_calls.append(list(messages))
        if not self.responses:
            raise AssertionError("script has no more responses")
        return self.responses.popleft()


def test_memory_store_save_list_and_relevance_order(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save("用户的导师姓张", source_run_id="run_1")
        store.save("用户喜欢喝拿铁咖啡", source_run_id="run_2")
        store.save("MiniAgent 使用 Python 实现", source_run_id="run_3")

        matches = store.search("我的导师姓什么", limit=3)

        assert matches
        assert matches[0].record.content == "用户的导师姓张"
        assert matches[0].distance <= matches[-1].distance
        assert len(store.list()) == 3


def test_memory_store_delete_removes_record_and_vector(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        record = store.save("这条记忆将被删除")

        assert store.delete(record.id) is True
        assert store.list() == []
        assert store.search("这条记忆将被删除", limit=1) == []
        assert store.delete(record.id) is False


def test_save_memory_tool_writes_to_store(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        save_memory = create_save_memory_tool(store)
        registry = ToolRegistry()
        registry.register(save_memory)

        result = registry.execute(
            ToolCall(
                id="call_memory",
                name="save_memory",
                arguments={"content": "用户的项目仓库是 huayuan"},
            )
        )

        assert result.ok is True
        assert result.data["saved"] is True
        assert store.list()[0].content == "用户的项目仓库是 huayuan"
        assert store.list()[0].source_run_id == "model-tool"


def test_memory_extractor_parses_fenced_json_and_saves_fact(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        [
            ModelResponse(
                content='```json\n{"save": true, "memories": ["用户的导师姓张"]}\n```',
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            )
        ]
    )
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        saved = MemoryExtractor(adapter, store).extract(
            "请记住我的导师姓张", "已记住。", "run_extractor"
        )

        assert saved == ["用户的导师姓张"]
        assert store.list()[0].source_run_id == "run_extractor"
        assert "长期记忆抽取器" in adapter.seen_messages[0].content


def test_memory_to_message_orders_relevant_memories(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save("用户的导师姓张")
        store.save("用户喜欢喝拿铁咖啡")
        matches = store.search("我导师姓什么", limit=2)
        message = memory_to_message(matches)

        assert message is not None
        assert message.role == "system"
        assert "长期记忆" in message.content
        assert "用户的导师姓张" in message.content


def test_agent_remembers_fact_across_sessions(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    settings = Settings(
        memory_enabled=True,
        memory_db_path=str(db_path),
        sandbox_dir=str(tmp_path / "sandbox"),
        context_token_budget=8192,
    )

    first_adapter = ScriptedAdapter(
        [
            ModelResponse(
                content="已记住。",
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            ),
            ModelResponse(
                content='{"save": true, "memories": ["用户的导师姓张"]}',
                usage=Usage(prompt_tokens=8, completion_tokens=4),
            ),
        ]
    )
    with Agent(adapter=first_adapter, settings=settings) as agent:
        first = agent.run("请记住我的导师姓张")

    assert first.memories_saved == ["用户的导师姓张"]

    second_adapter = ScriptedAdapter(
        [
            ModelResponse(
                content="你的导师姓张。",
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            ),
            ModelResponse(
                content='{"save": false, "memories": []}',
                usage=Usage(prompt_tokens=8, completion_tokens=4),
            ),
        ]
    )
    with Agent(adapter=second_adapter, settings=settings) as agent:
        second = agent.run("我导师姓什么")

    assert second.answer == "你的导师姓张。"
    assert any(
        message.role == "system" and "用户的导师姓张" in message.content
        for message in second_adapter.all_calls[0]
    )


def test_memory_cli_parser_supports_list_delete_and_search() -> None:
    parser = build_parser()

    assert parser.parse_args(["memory", "list"]).memory_command == "list"
    delete_args = parser.parse_args(["memory", "delete", "3"])
    assert delete_args.memory_command == "delete"
    assert delete_args.memory_id == 3
    search_args = parser.parse_args(["memory", "search", "导师", "--top-k", "2"])
    assert search_args.query == "导师"
    assert search_args.top_k == 2
