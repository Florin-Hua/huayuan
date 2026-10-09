"""MiniAgent Stage 8 evaluation runner.

Runs the selfbuilt benchmark with deterministic scripted adapters so the
framework mechanisms (tools, recovery, compression, memory) can be measured
without claiming commercial-model accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mini_agent.config import Settings
from mini_agent.core.agent import Agent
from mini_agent.core.types import Message, ModelResponse, ToolCall, Usage


class ScriptedEvalAdapter:
    """Deterministic policy adapter used only for framework evaluation."""

    def __init__(self, tasks: list[dict[str, Any]], model: str) -> None:
        self.tasks = {task["task"]: task for task in tasks}
        self.model = model
        self.calls = 0
        self.tokens = 0
        self._call_seq = 0
        self._main_call_index: dict[str, int] = {}

    def chat(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.calls += 1
        self._call_seq += 1
        system_texts = [message.content or "" for message in messages if message.role == "system"]
        user_texts = [message.content or "" for message in messages if message.role == "user"]

        if any(text.startswith("你是上下文压缩器") for text in system_texts):
            usage = Usage(prompt_tokens=90, completion_tokens=35)
            self.tokens += usage.total
            return ModelResponse(content="早期多轮工具结果已压缩，共同关键词为 compression。", usage=usage)

        if any(text.startswith("你是长期记忆抽取器") for text in system_texts):
            task = self._find_task("\n".join(user_texts))
            save = bool(task and task.get("memory_seed"))
            content = json.dumps({"save": save, "memories": ["项目总结必须使用中文，并且保持简短。"] if save else []}, ensure_ascii=False)
            usage = Usage(prompt_tokens=70, completion_tokens=18)
            self.tokens += usage.total
            return ModelResponse(content=content, usage=usage)

        task = self._find_task("\n".join(user_texts))
        if task is None:
            raise RuntimeError(f"unknown evaluation task: {user_texts[-1][:120]}")

        prompt_tokens = 30 + sum(max(len(message.content or "") // 8, 4) for message in messages)
        completion_tokens = 18
        response: ModelResponse

        if self.model == "baseline":
            if task.get("memory_sensitive"):
                has_memory = any(text.startswith("长期记忆") for text in system_texts)
                content = task["final"] if has_memory else task["baseline_final"]
            else:
                content = task["baseline_final"]
            response = ModelResponse(content=content)
        elif self.model == "strong":
            if task.get("memory_sensitive"):
                has_memory = any(text.startswith("长期记忆") for text in system_texts)
                content = task["final"] if has_memory else task["baseline_final"]
                response = ModelResponse(content=content)
            else:
                round_index = self._main_call_index.get(task["id"], 0)
                self._main_call_index[task["id"]] = round_index + 1
                rounds = task.get("tool_rounds", [])
                if round_index < len(rounds):
                    calls = [
                        ToolCall(
                            id=f"eval_{self._call_seq}_{index}",
                            name=item["name"],
                            arguments=item["arguments"],
                        )
                        for index, item in enumerate(rounds[round_index], start=1)
                    ]
                    completion_tokens += 9 * len(calls)
                    response = ModelResponse(content="我需要调用工具。", tool_calls=calls)
                else:
                    response = ModelResponse(content=task["final"])
        else:
            raise ValueError(f"unknown scripted model: {self.model}")

        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self.tokens += usage.total
        response.usage = usage
        return response

    def _find_task(self, text: str) -> dict[str, Any] | None:
        for task_text, task in self.tasks.items():
            if task_text in text:
                return task
        return None


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", text).strip().lower()
    value = re.sub(r"[\s，。！？；：,.!?;:]+", "", value)
    return value


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    if not tasks:
        raise ValueError(f"dataset has no tasks: {path}")
    return tasks


def setup_task_sandbox(root: Path, task: dict[str, Any]) -> Path:
    sandbox = root / "sandbox" / task["id"]
    sandbox.mkdir(parents=True, exist_ok=True)
    for relative_path, content in (task.get("setup_files") or {}).items():
        target = sandbox / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return sandbox


def run_experiment(
    tasks: list[dict[str, Any]],
    model: str,
    compression: bool,
    memory: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    selected = tasks if limit is None else tasks[:limit]
    with tempfile.TemporaryDirectory(prefix="miniagent-eval-") as temp:
        temp_path = Path(temp)
        adapter = ScriptedEvalAdapter(selected, model)
        memory_db = temp_path / "memory.sqlite3"
        records: list[dict[str, Any]] = []

        for task in selected:
            calls_before = adapter.calls
            tokens_before = adapter.tokens
            sandbox = setup_task_sandbox(temp_path, task)
            settings = Settings(
                memory_enabled=memory,
                memory_db_path=str(memory_db),
                memory_top_k=5,
                trace_enabled=False,
                context_compression_enabled=compression,
                context_token_budget=650,
                context_recent_turns=1,
            )
            agent = Agent(
                adapter=adapter,
                sandbox_dir=sandbox,
                max_iterations=8,
                settings=settings,
            )
            try:
                result = agent.run(task["task"])
            finally:
                agent.close()

            answer_ok = normalize_answer(result.answer) == normalize_answer(task["expected"])
            failed_tools = [step for step in result.steps if step.state == "ACT" and step.ok is False]
            records.append(
                {
                    "id": task["id"],
                    "category": task["category"],
                    "success": result.status.value == "success" and answer_ok,
                    "status": result.status.value,
                    "answer": result.answer,
                    "expected": task["expected"],
                    "iterations": result.iterations,
                    "tokens": adapter.tokens - tokens_before,
                    "core_tokens": result.usage.total,
                    "model_calls": adapter.calls - calls_before,
                    "parse_retries": result.parse_retries,
                    "compressions": len(result.compressions),
                    "memories_saved": len(result.memories_saved),
                    "failed_tool_calls": len(failed_tools),
                    "recovered": bool(task.get("failure_recovery")) and answer_ok and bool(failed_tools),
                    "termination_reason": result.termination_reason,
                }
            )

        return summarize(records, model, compression, memory)


def summarize(
    records: list[dict[str, Any]], model: str, compression: bool, memory: bool
) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)

    for category, items in sorted(grouped.items()):
        by_category[category] = metric_block(items)

    failure_items = [record for record in records if record["category"] == "failure_recovery"]
    return {
        "model": f"scripted-{model}",
        "compression": compression,
        "memory": memory,
        "total": metric_block(records),
        "recovery_rate": (
            sum(record["recovered"] for record in failure_items) / len(failure_items)
            if failure_items
            else 0.0
        ),
        "by_category": by_category,
        "records": records,
    }


def metric_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    return {
        "count": count,
        "success_rate": sum(record["success"] for record in records) / count if count else 0.0,
        "avg_iterations": sum(record["iterations"] for record in records) / count if count else 0.0,
        "avg_tokens": sum(record["tokens"] for record in records) / count if count else 0.0,
        "avg_model_calls": sum(record["model_calls"] for record in records) / count if count else 0.0,
        "parse_retries": sum(record["parse_retries"] for record in records),
        "compressions": sum(record["compressions"] for record in records),
        "memories_saved": sum(record["memories_saved"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/selfbuilt.json", type=Path)
    parser.add_argument("--output", default=Path("eval/results/selfbuilt_scripted.json"), type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", nargs="+", default=["strong", "baseline"])
    args = parser.parse_args()

    tasks = load_dataset(args.dataset)
    experiments = []
    started = time.perf_counter()
    for model in args.models:
        for compression in (True, False):
            for memory in (True, False):
                experiments.append(
                    run_experiment(tasks, model, compression, memory, args.limit)
                )
    elapsed = time.perf_counter() - started
    payload = {
        "benchmark": str(args.dataset),
        "mode": "scripted-framework-eval",
        "note": "scripted adapters test framework mechanisms; results are not commercial model accuracy",
        "task_count": len(tasks) if args.limit is None else args.limit,
        "elapsed_seconds": round(elapsed, 3),
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for experiment in experiments:
        total = experiment["total"]
        print(
            f"{experiment['model']}: compression={experiment['compression']} "
            f"memory={experiment['memory']} success={total['success_rate']:.2%} "
            f"iterations={total['avg_iterations']:.2f} tokens={total['avg_tokens']:.1f} "
            f"recovery={experiment['recovery_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
