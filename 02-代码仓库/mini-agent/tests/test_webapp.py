"""Stage 9 Web UI tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mini_agent.config import Settings
from mini_agent.core.agent import Agent
from mini_agent.core.types import ModelResponse, Usage
from mini_agent.memory import MemoryStore
from mini_agent.webapp import MiniAgentWebApp, serialize_run


class FakeAdapter:
    def chat(self, messages: list[Any], tools: list[dict]) -> ModelResponse:
        return ModelResponse(content="42", usage=Usage(prompt_tokens=10, completion_tokens=5))


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        memory_enabled=False,
        trace_enabled=True,
        sandbox_dir=str(tmp_path / "sandbox"),
        memory_db_path=str(tmp_path / "memory.sqlite3"),
        trace_db_path=str(tmp_path / "trace.sqlite3"),
    )


def make_app(settings: Settings) -> MiniAgentWebApp:
    def factory(**kwargs: Any) -> Agent:
        return Agent(adapter=FakeAdapter(), settings=kwargs["settings"])

    return MiniAgentWebApp(settings=settings, agent_factory=factory)


def test_config_does_not_expose_secrets() -> None:
    app = MiniAgentWebApp(Settings(_env_file=None, memory_enabled=False, trace_enabled=False))
    payload = app.config()
    assert payload["provider"] in ("openai", "anthropic")
    assert "apiKey" not in payload
    assert "openai_api_key" not in payload


def test_run_task_returns_trace_ready_payload(tmp_path: Path) -> None:
    app = make_app(make_settings(tmp_path))
    status, payload = app.route_api(
        "POST", "/api/run", {}, {"task": "计算 19 + 23", "maxIterations": 3}
    )
    assert status == 200
    run = payload["run"]
    assert run["status"] == "success"
    assert run["answer"] == "42"
    assert run["usage"]["total"] == 15
    assert serialize_run.__name__ == "serialize_run"


def test_run_task_validates_payload(tmp_path: Path) -> None:
    app = make_app(make_settings(tmp_path))
    assert app.route_api("POST", "/api/run", {}, {"task": ""})[0] == 400
    assert app.route_api("POST", "/api/run", {}, {"task": "x", "provider": "bad"})[0] == 400
    assert app.route_api("POST", "/api/run", {}, {"task": "x", "maxIterations": 21})[0] == 400


def test_memory_routes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path).model_copy(deep=True)
    settings.memory_enabled = True
    app = MiniAgentWebApp(settings=settings)
    with MemoryStore(settings.memory_db_path, dimension=settings.memory_dimension) as store:
        record = store.save("项目总结使用中文")

    status, payload = app.route_api("GET", "/api/memories", {}, {})
    assert status == 200
    assert payload["memories"][0]["content"] == record.content

    status, payload = app.route_api(
        "POST", "/api/memories/search", {}, {"query": "项目", "topK": 3}
    )
    assert status == 200
    assert payload["matches"][0]["id"] == record.id

    status, payload = app.route_api("DELETE", f"/api/memories/{record.id}", {}, {})
    assert status == 200
    assert payload["deleted"] == record.id


def test_trace_and_static_routes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = make_app(settings)
    _, payload = app.route_api("POST", "/api/run", {}, {"task": "读取 notes.txt"})
    run_id = payload["run"]["run_id"]

    status, payload = app.route_api("GET", "/api/runs", {}, {})
    assert status == 200
    assert payload["runs"][0]["runId"] == run_id

    status, payload = app.route_api("GET", f"/api/runs/{run_id}", {}, {})
    assert status == 200
    assert payload["run"]["answer"] == "42"

    status, content, content_type = app.route_static("/")
    assert status.value == 200
    assert "MiniAgent 可视化控制台" in content
    assert content_type == "text/html; charset=utf-8"

    status, _, _ = app.route_static("/..")
    assert status.value == 403
