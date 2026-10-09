"""Stage 8 evaluation tests."""
from __future__ import annotations

import json
from pathlib import Path

from eval.runner import load_dataset, normalize_answer, run_experiment

EVAL_DIR = Path(__file__).parents[1] / "eval"


def test_selfbuilt_dataset_has_required_coverage() -> None:
    tasks = load_dataset(EVAL_DIR / "selfbuilt.json")
    assert len(tasks) == 30
    assert sum(bool(task.get("failure_recovery")) for task in tasks) >= 5
    assert sum(bool(task.get("compression_sensitive")) for task in tasks) >= 4
    assert sum(bool(task.get("memory_seed") or task.get("memory_sensitive")) for task in tasks) >= 2


def test_normalize_answer_ignores_punctuation_and_case() -> None:
    assert normalize_answer(" The answer: 42。 ") == normalize_answer("the answer 42")


def test_strong_policy_recovers_all_failures() -> None:
    tasks = load_dataset(EVAL_DIR / "selfbuilt.json")
    result = run_experiment(tasks, "strong", compression=True, memory=True)
    assert result["total"]["success_rate"] == 1.0
    assert result["recovery_rate"] == 1.0
    assert result["by_category"]["context_compression"]["success_rate"] == 1.0


def test_compression_off_fails_long_context_tasks() -> None:
    tasks = load_dataset(EVAL_DIR / "selfbuilt.json")
    result = run_experiment(tasks, "strong", compression=False, memory=True)
    assert result["by_category"]["context_compression"]["success_rate"] == 0.0
    assert result["total"]["success_rate"] < 1.0


def test_gaia_subset_does_not_fake_gated_data() -> None:
    payload = json.loads((EVAL_DIR / "gaia_subset.json").read_text(encoding="utf-8"))
    assert payload["source"]["access"] == "gated"
    assert payload["source"]["download_status"] == "not_available_without_huggingface_authorization"
    assert payload["tasks"] == []
