"""从已授权下载的 GAIA validation 文件生成 text-only 子集。

用法：
  python eval/select_gaia.py --input data/gaia_validation.jsonl --output eval/gaia_subset.json --limit 30

本脚本不访问网络、不绕过 HuggingFace gating；输入文件必须由已授权用户自行下载。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_text_only(item: dict) -> bool:
    task_type = str(item.get("task_type", "")).lower()
    if task_type == "web":
        return False
    question = str(item.get("Question", item.get("question", "")))
    answer = str(item.get("Final answer", item.get("final_answer", "")))
    return bool(question.strip() and answer.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("eval/gaia_subset.json"), type=Path)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if is_text_only(row)][: args.limit]
    manifest = {
        "name": "GAIA validation text-only subset",
        "source": {
            "dataset": "gaia-benchmark/GAIA",
            "split": "validation",
            "access": "authorized-local-copy",
        },
        "target_size": args.limit,
        "selection_criteria": [
            "排除 task_type=web 或需要真实浏览器的任务",
            "排除缺失 question / final answer 的任务",
            "保留可通过文本与本地工具完成的任务",
        ],
        "tasks": [
            {
                "id": str(row.get("task_id", row.get("Task_id", ""))),
                "task": str(row.get("Question", row.get("question", ""))),
                "expected": str(row.get("Final answer", row.get("final_answer", ""))),
                "category": str(row.get("task_type", row.get("task_type", "text"))),
                "level": row.get("Level", row.get("level")),
            }
            for row in selected
        ],
    }
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected {len(selected)} tasks -> {args.output}")


if __name__ == "__main__":
    main()
