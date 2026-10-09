"""MiniAgent 命令行入口：mini run / mini memory。"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mini_agent.config import Settings, load_settings
from mini_agent.core.types import AgentRun, RunStatus
from mini_agent.memory import MemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini", description="MiniAgent 命令行")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="运行一个任务")
    run.add_argument("task", help="要完成的任务描述")
    run.add_argument("--model", help="覆盖 .env 中的 MODEL_NAME / LLM_MODEL")
    run.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="覆盖 .env 中的 MODEL_PROVIDER",
    )
    run.add_argument(
        "--max-iterations", type=int, default=None, help="覆盖最大迭代次数"
    )
    run.add_argument(
        "--sandbox",
        default=None,
        help="工具沙箱目录，默认读取 SANDBOX_DIR",
    )

    memory = sub.add_parser("memory", help="管理长期记忆")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    memory_sub.add_parser("list", help="列出全部记忆")
    delete = memory_sub.add_parser("delete", help="删除一条记忆")
    delete.add_argument("memory_id", type=int, help="记忆 ID")
    search = memory_sub.add_parser("search", help="按相关性检索记忆")
    search.add_argument("query", help="检索文本")
    search.add_argument("--top-k", type=int, default=None, help="返回条数")
    return parser


def _print_run(console: Console, result: AgentRun) -> None:
    table = Table(title=f"MiniAgent run {result.run_id}")
    table.add_column("Iter", justify="right")
    table.add_column("State")
    table.add_column("Tool")
    table.add_column("OK")
    table.add_column("Detail")
    for step in result.steps:
        detail = step.error or step.note or ""
        table.add_row(
            str(step.iteration),
            step.state,
            step.tool_name or "-",
            "-" if step.ok is None else ("yes" if step.ok else "no"),
            detail,
        )
    console.print(table)
    console.print(f"状态：[bold]{result.status.value}[/bold]")
    console.print(f"迭代次数：{result.iterations}；tokens：{result.usage.total}")
    if result.compressions:
        console.print(f"上下文压缩：{len(result.compressions)} 次")
    if result.memories_saved:
        console.print(f"新增长期记忆：{len(result.memories_saved)} 条")
    if result.answer:
        console.print(f"最终输出：{result.answer}")


def _print_memories(console: Console, records, distances=None) -> None:
    table = Table(title="MiniAgent memories")
    table.add_column("ID", justify="right")
    table.add_column("Content")
    table.add_column("Run ID")
    table.add_column("Created At")
    if distances is not None:
        table.add_column("Distance", justify="right")
    for index, record in enumerate(records):
        row = [
            str(record.id),
            record.content,
            record.source_run_id or "-",
            record.created_at,
        ]
        if distances is not None:
            row.append(f"{distances[index]:.4f}")
        table.add_row(*row)
    console.print(table)


def _handle_memory(args: argparse.Namespace, console: Console, settings: Settings) -> int:
    if not settings.memory_enabled:
        console.print("[red]MEMORY_ENABLED=false，记忆系统未启用。[/red]")
        return 1

    with MemoryStore(
        settings.memory_db_path, dimension=settings.memory_dimension
    ) as store:
        if args.memory_command == "list":
            records = store.list()
            if not records:
                console.print("暂无长期记忆。")
                return 0
            _print_memories(console, records)
            return 0

        if args.memory_command == "delete":
            if store.delete(args.memory_id):
                console.print(f"已删除记忆 {args.memory_id}。")
                return 0
            console.print(f"[red]未找到记忆 {args.memory_id}。[/red]")
            return 1

        if args.memory_command == "search":
            limit = args.top_k or settings.memory_top_k
            matches = store.search(args.query, limit=limit)
            if not matches:
                console.print("没有匹配的长期记忆。")
                return 0
            _print_memories(
                console,
                [match.record for match in matches],
                [match.distance for match in matches],
            )
            return 0

    raise ValueError(f"unsupported memory command: {args.memory_command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    settings = load_settings()

    if args.command == "memory":
        return _handle_memory(args, console, settings)

    if args.provider:
        settings.model_provider = args.provider

    model = args.model or settings.resolved_model
    max_iterations = (
        settings.max_iterations if args.max_iterations is None else args.max_iterations
    )
    sandbox = Path(args.sandbox or settings.sandbox_dir).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        from mini_agent.core.agent import Agent

        with Agent(
            model=model,
            max_iterations=max_iterations,
            sandbox_dir=sandbox,
            settings=settings,
        ) as agent:
            result = agent.run(args.task)
    except Exception as exc:
        console.print(f"[red]启动失败：{type(exc).__name__}: {exc}[/red]")
        return 1

    _print_run(console, result)
    return 0 if result.status == RunStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
