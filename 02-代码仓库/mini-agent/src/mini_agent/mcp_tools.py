"""MCP 工具适配：把 MCP server 的 tools 接入 MiniAgent ToolRegistry。"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, create_model

from mini_agent.tools.registry import ToolDefinition, ToolRegistry


class McpToolError(RuntimeError):
    """MCP tool 返回 error result。"""


class McpClientProtocol(Protocol):
    """McpToolAdapter 需要的最小客户端接口，便于测试 mock。"""

    def list_tools(self) -> Any:
        ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        ...

    def close(self) -> None:
        ...


class McpClient:
    """用后台 asyncio 线程包装 MCP stdio ClientSession。

    MiniAgent 的 ToolRegistry 是同步执行模型；这里把 async SDK 放在
    专用事件循环线程中，工具函数通过 run_coroutine_threadsafe 转发。
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        startup_timeout: float = 10.0,
        request_timeout: float = 30.0,
    ) -> None:
        self.params = StdioServerParameters(
            command=command, args=args or [], cwd=cwd, env=env
        )
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run_worker, name="mini-agent-mcp-client", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(self.startup_timeout):
            self.close()
            raise TimeoutError(f"MCP server startup timeout: {command}")
        if self._startup_error is not None:
            self.close()
            raise RuntimeError(f"MCP server startup failed: {self._startup_error}") from self._startup_error

    def _run_worker(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._worker_main())
        except BaseException as exc:
            self._startup_error = exc
        finally:
            try:
                loop.close()
            finally:
                self._loop = None
                self._ready.set()

    async def _worker_main(self) -> None:
        assert self._loop is not None
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(self.params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            self._ready.set()
            # stop.wait 是同步阻塞，放到线程池避免阻塞事件循环。
            await asyncio.to_thread(self._stop.wait)
            self._session = None

    def _submit(self, coroutine: Any) -> Any:
        loop = self._loop
        session = self._session
        if loop is None or session is None or not self._thread.is_alive():
            if self._startup_error is not None:
                raise RuntimeError(f"MCP client stopped: {self._startup_error}") from self._startup_error
            raise RuntimeError("MCP client is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(
            self.request_timeout
        )

    def list_tools(self) -> Any:
        assert self._session is not None
        return self._submit(self._session.list_tools())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert self._session is not None
        return self._submit(self._session.call_tool(name, arguments))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.request_timeout)
        if self._thread.is_alive():
            raise RuntimeError("MCP client thread did not stop cleanly")


def _without_titles(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_titles(item)
            for key, item in value.items()
            if key != "title"
        }
    if isinstance(value, list):
        return [_without_titles(item) for item in value]
    return value


def _annotation(schema: dict[str, Any]) -> type:
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list[Any],
        "object": dict[str, Any],
    }
    return mapping.get(schema.get("type"), Any)


def _args_model(schema: dict[str, Any], tool_name: str) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    required = set(schema.get("required", []))
    for name, spec in schema.get("properties", {}).items():
        spec = spec if isinstance(spec, dict) else {}
        annotation = _annotation(spec)
        description = spec.get("description")
        if name in required:
            fields[name] = (annotation, Field(description=description))
        else:
            fields[name] = (annotation | None, None)
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", tool_name)
    return create_model(f"{safe_name}McpInput", **fields)


def _content_payload(content: Any) -> Any:
    if not content:
        return None
    payloads: list[Any] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None:
            payloads.append(block)
            continue
        try:
            payloads.append(json.loads(text))
        except (TypeError, ValueError, json.JSONDecodeError):
            payloads.append(text)
    return payloads[0] if len(payloads) == 1 else payloads


def _call_result_payload(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    if structured is not None:
        return structured
    return _content_payload(getattr(result, "content", None))


def _definition(client: McpClientProtocol, tool: Any, timeout_seconds: float) -> ToolDefinition:
    schema = _without_titles(tool.input_schema)
    model = _args_model(schema, tool.name)

    def invoke(**arguments: Any) -> Any:
        result = client.call_tool(tool.name, arguments)
        if getattr(result, "is_error", False):
            raise McpToolError(f"MCP tool {tool.name} failed: {_call_result_payload(result)}")
        return _call_result_payload(result)

    return ToolDefinition(
        name=tool.name,
        description=tool.description or f"MCP tool {tool.name}",
        parameters=schema,
        func=invoke,
        args_model=model,
        timeout_seconds=timeout_seconds,
        requires_sandbox=False,
    )


class McpToolAdapter:
    """发现 MCP tools，并转换成 ToolRegistry 的标准 ToolDefinition。"""

    def __init__(
        self,
        command: str | list[str] | None = None,
        client: McpClientProtocol | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if client is None:
            if not command:
                raise ValueError("McpToolAdapter requires command or client")
            parts = [command] if isinstance(command, str) else list(command)
            if not parts:
                raise ValueError("MCP server command is empty")
            self.owns_client = True
            self.client: McpClientProtocol = McpClient(command=parts[0], args=parts[1:])
        else:
            self.owns_client = False
            self.client = client
        self.timeout_seconds = timeout_seconds

    def definitions(self) -> list[ToolDefinition]:
        result = self.client.list_tools()
        tools = list(getattr(result, "tools", []))
        return [
            _definition(self.client, tool, self.timeout_seconds) for tool in tools
        ]

    def register_tools(self, registry: ToolRegistry) -> list[ToolDefinition]:
        definitions = self.definitions()
        for definition in definitions:
            registry.register(definition)
        return definitions

    def close(self) -> None:
        if self.owns_client:
            self.client.close()
