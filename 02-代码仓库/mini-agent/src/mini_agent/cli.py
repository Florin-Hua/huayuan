"""MiniAgent 命令行入口：mini run "task"。"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mini_agent.config import load_settings
from mini_agent.core.types import AgentRun, RunStatus


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
    if result.answer:
        console.print(f"最终输出：{result.answer}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    settings = load_settings()
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

        agent = Agent(
            model=model,
            max_iterations=max_iterations,
            sandbox_dir=sandbox,
            settings=settings,
        )
        result = agent.run(args.task)
    except Exception as exc:
        console.print(f"[red]启动失败：{type(exc).__name__}: {exc}[/red]")
        return 1

    _print_run(console, result)
    return 0 if result.status == RunStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())