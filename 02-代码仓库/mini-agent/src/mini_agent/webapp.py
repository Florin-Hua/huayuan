"""MiniAgent 本地可视化 Web UI：零前端构建、零新增依赖。"""
from __future__ import annotations

import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

from mini_agent.config import Settings, load_settings
from mini_agent.core.types import AgentRun
from mini_agent.memory import MemoryStore
from mini_agent.trace import TraceStore


_STATIC_ROOT = Path(__file__).parent / "web"
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
_AGENT_FACTORY = Callable[..., Any]


def serialize_run(run: AgentRun) -> dict[str, Any]:
    """把 AgentRun 转成 JSON 安全数据，并补齐 UI 需要的 total token。"""
    payload = run.model_dump(mode="json")
    payload["usage"]["total"] = run.usage.total
    return payload


class MiniAgentWebApp:
    """纯 Python API 路由层，方便单元测试与 HTTP 层解耦。"""

    def __init__(
        self,
        settings: Settings | None = None,
        agent_factory: _AGENT_FACTORY | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._agent_factory = agent_factory

    def config(self) -> dict[str, Any]:
        return {
            "provider": self.settings.model_provider,
            "model": self.settings.resolved_model,
            "memoryEnabled": self.settings.memory_enabled,
            "traceEnabled": self.settings.trace_enabled,
            "sandboxDir": str(Path(self.settings.sandbox_dir).resolve()),
            "maxIterations": self.settings.max_iterations,
            "tokenBudget": self.settings.token_budget,
            "contextCompression": self.settings.context_compression_enabled,
            "tools": [
                "calculator",
                "read_text_file",
                "write_text_file",
                "mock_web_search",
                "save_memory",
                "mcp_echo",
                "mcp_add",
            ],
        }

    def run_task(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        task = str(payload.get("task", "")).strip()
        if not task:
            return 400, {"error": "task is required"}
        if len(task) > 4000:
            return 400, {"error": "task is too long"}

        provider = payload.get("provider")
        if provider is not None and provider not in ("openai", "anthropic"):
            return 400, {"error": "provider must be openai or anthropic"}
        model = payload.get("model")
        if model is not None:
            model = str(model).strip()[:100] or None
        max_iterations = payload.get("maxIterations")
        if max_iterations is not None:
            try:
                max_iterations = int(max_iterations)
            except (TypeError, ValueError):
                return 400, {"error": "maxIterations must be an integer"}
            if not 1 <= max_iterations <= 20:
                return 400, {"error": "maxIterations must be between 1 and 20"}

        settings = self.settings.model_copy(deep=True)
        if provider is not None:
            settings.model_provider = provider
        if model is not None:
            settings.model_name = model
        if max_iterations is not None:
            settings.max_iterations = max_iterations
        if payload.get("useBuiltinMcp") is True:
            # UI 只允许启用项目内置 MCP server，不允许传任意命令。
            settings.mcp_server_command = [sys.executable, "-m", "mini_agent.mcp_server"]

        factory = self._agent_factory
        if factory is None:
            from mini_agent.core.agent import Agent

            factory = Agent
        try:
            with factory(settings=settings) as agent:
                result = agent.run(task)
        except Exception as exc:
            return 500, {"error": f"{type(exc).__name__}: {exc}"}
        return 200, {"run": serialize_run(result)}

    def list_memories(self) -> tuple[int, dict[str, Any]]:
        if not self.settings.memory_enabled:
            return 400, {"error": "memory is disabled"}
        with MemoryStore(
            self.settings.memory_db_path, dimension=self.settings.memory_dimension
        ) as store:
            records = [
                {
                    "id": record.id,
                    "content": record.content,
                    "sourceRunId": record.source_run_id,
                    "createdAt": record.created_at,
                }
                for record in store.list()
            ]
        return 200, {"memories": records}

    def search_memories(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if not self.settings.memory_enabled:
            return 400, {"error": "memory is disabled"}
        query = str(payload.get("query", "")).strip()
        if not query:
            return 400, {"error": "query is required"}
        try:
            limit = int(payload.get("topK", self.settings.memory_top_k))
        except (TypeError, ValueError):
            return 400, {"error": "topK must be an integer"}
        if not 1 <= limit <= 20:
            return 400, {"error": "topK must be between 1 and 20"}
        with MemoryStore(
            self.settings.memory_db_path, dimension=self.settings.memory_dimension
        ) as store:
            matches = [
                {
                    "id": match.record.id,
                    "content": match.record.content,
                    "distance": match.distance,
                }
                for match in store.search(query, limit=limit)
            ]
        return 200, {"matches": matches}

    def delete_memory(self, memory_id: int) -> tuple[int, dict[str, Any]]:
        if not self.settings.memory_enabled:
            return 400, {"error": "memory is disabled"}
        with MemoryStore(
            self.settings.memory_db_path, dimension=self.settings.memory_dimension
        ) as store:
            if not store.delete(memory_id):
                return 404, {"error": f"memory {memory_id} not found"}
        return 200, {"deleted": memory_id}

    def list_runs(self) -> tuple[int, dict[str, Any]]:
        if not self.settings.trace_enabled:
            return 400, {"error": "trace is disabled"}
        with TraceStore(self.settings.trace_db_path) as store:
            runs = [
                {
                    "runId": row["run_id"],
                    "task": row["task"],
                    "status": row["status"],
                    "iterations": row["iterations"],
                    "totalTokens": row["total_tokens"],
                    "finishedAt": row["finished_at"],
                }
                for row in store.list_runs()
            ]
        return 200, {"runs": runs}

    def get_run(self, run_id: str) -> tuple[int, dict[str, Any]]:
        if not self.settings.trace_enabled:
            return 400, {"error": "trace is disabled"}
        with TraceStore(self.settings.trace_db_path) as store:
            run = store.get_run(run_id)
            if run is None:
                return 404, {"error": f"run {run_id} not found"}
            steps = store.get_steps(run_id)
        for step in steps:
            if step.get("tool_args"):
                try:
                    step["tool_args"] = json.loads(step["tool_args"])
                except json.JSONDecodeError:
                    pass
        return 200, {"run": run, "steps": steps}

    def route_api(
        self, method: str, path: str, query: dict[str, list[str]], body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        if path == "/api/config" and method == "GET":
            return 200, self.config()
        if path == "/api/run" and method == "POST":
            return self.run_task(body)
        if path == "/api/memories" and method == "GET":
            return self.list_memories()
        if path == "/api/memories/search" and method == "POST":
            return self.search_memories(body)
        memory_prefix = "/api/memories/"
        if path.startswith(memory_prefix) and method == "DELETE":
            try:
                memory_id = int(path[len(memory_prefix) :])
            except ValueError:
                return 400, {"error": "invalid memory id"}
            return self.delete_memory(memory_id)
        if path == "/api/runs" and method == "GET":
            return self.list_runs()
        runs_prefix = "/api/runs/"
        if path.startswith(runs_prefix) and method == "GET":
            return self.get_run(unquote(path[len(runs_prefix) :]))
        return 404, {"error": "not found"}

    def route_static(self, path: str) -> tuple[HTTPStatus, bytes | str, str | None]:
        relative = "index.html" if path == "/" else path.lstrip("/")
        if "/" in relative or ".." in relative:
            return HTTPStatus.FORBIDDEN, "forbidden", "text/plain; charset=utf-8"
        target = _STATIC_ROOT / relative
        if not target.is_file():
            return HTTPStatus.NOT_FOUND, "not found", "text/plain; charset=utf-8"
        content_type = _STATIC_TYPES.get(target.suffix.lower())
        if content_type is None:
            return HTTPStatus.FORBIDDEN, "forbidden", "text/plain; charset=utf-8"
        return HTTPStatus.OK, target.read_text(encoding="utf-8"), content_type


class MiniAgentRequestHandler(BaseHTTPRequestHandler):
    """绑定 MiniAgentWebApp 的 HTTP handler。"""

    app: MiniAgentWebApp

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[mini-web] {self.address_string()} {format % args}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, path: str) -> None:
        status, content, content_type = self.app.route_static(path)
        if content_type is None:
            self._send_json(500, {"error": "invalid static response"})
            return
        data = content.encode("utf-8") if isinstance(content, str) else content
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0:
            return {}
        if length > 64 * 1024:
            raise ValueError("request body too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _handle(self, method: str) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/"):
                body = self._read_json() if method in ("POST", "PUT", "PATCH") else {}
                status, payload = self.app.route_api(
                    method, path, parse_qs(parsed.query), body
                )
                self._send_json(status, payload)
                return
            if method == "GET":
                self._send_static(path)
                return
            self._send_json(405, {"error": "method not allowed"})
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_DELETE(self) -> None:
        self._handle("DELETE")


def create_server(
    host: str = "127.0.0.1", port: int = 8765, app: MiniAgentWebApp | None = None
) -> ThreadingHTTPServer:
    """创建只绑定本机的 HTTP 服务。"""
    handler = type("BoundMiniAgentHandler", (MiniAgentRequestHandler,), {"app": app or MiniAgentWebApp()})
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="mini-web", description="MiniAgent 可视化界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = MiniAgentWebApp()
    server = create_server(args.host, args.port, app)
    print(f"MiniAgent Web UI: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止。密钥保留在本地 .env，不会通过 /api/config 返回。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMiniAgent Web UI 已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
