#!/usr/bin/env python3
"""Private JSONL problem runner for the Rust ``code-speedup`` verifier.

The bridge is the only component that materializes edge inputs, benchmark
variants, and hidden generators. Its stdout contains a deliberately small result
schema; it never returns a record, inputs, expected outputs, seeds, or a known
fast implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datagen.gen_problems import problem_from_record

from harness.perf import evaluate

_REQUIRED_RECORD_FIELDS = {
    "id", "family_id", "function_name", "reference_source", "input_generator",
    "edge_inputs", "benchmark_variants", "verification",
}


def load_problem_record(dataset: Path, problem_id: str) -> dict[str, Any]:
    """Select one verifier-gated private record without exposing other rows."""
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") != problem_id:
            continue
        missing = _REQUIRED_RECORD_FIELDS - set(row)
        if missing:
            raise ValueError(f"record missing fields: {sorted(missing)}")
        if row["verification"].get("status") != "passed":
            raise ValueError("selected record is not verifier-gated")
        return row
    raise ValueError("problem id not found")


def _first_count(value: Any) -> int | None:
    return value[0] if isinstance(value, (list, tuple)) and value else None


def safe_result(result: Any) -> dict[str, Any]:
    """Map harness output to the model-safe bridge response contract."""
    get = result.get if isinstance(result, dict) else lambda name, default=None: getattr(result, name, default)
    correct = bool(get("correct", False))
    reward = get("reward")
    status = str(get("status", "bridge_error"))
    passed = correct and status == "ok" and isinstance(reward, (int, float)) and reward > 0
    if correct and status == "ok" and not passed:
        status = "no_speedup"
    return {
        "passed": passed,
        "status": "passed" if passed else status,
        "correct": correct,
        "reward": reward if isinstance(reward, (int, float)) else None,
        "reference_instruction_count": _first_count(get("reference_counts", ())),
        "candidate_instruction_count": _first_count(get("candidate_counts", ())),
    }


def model_context(record: dict[str, Any], candidate_source: str, result: dict[str, Any]) -> str:
    """Build the exact public context shape for tests and downstream adapters."""
    return (
        f"family: {record['family_id']}\n"
        f"function: {record['function_name']}\n"
        f"status: {result['status']}\n"
        f"correct: {result['correct']}\n"
        f"reward: {result['reward']}\n"
        f"reference_instruction_count: {result['reference_instruction_count']}\n"
        f"candidate_instruction_count: {result['candidate_instruction_count']}\n"
        "--- slow reference ---\n"
        f"{record['reference_source']}\n"
        "--- candidate.py ---\n"
        f"{candidate_source}\n"
        "Rewrite candidate.py to be correct and use fewer instruction references."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random-cases", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    try:
        record = load_problem_record(args.dataset, args.problem_id)
        candidate = args.candidate.read_text(encoding="utf-8")
        result = evaluate(problem_from_record(record), candidate, seed=args.seed,
                          random_cases=args.random_cases, repeats=args.repeats)
        print(json.dumps(safe_result(result), sort_keys=True, allow_nan=False))
        return 0
    except Exception:  # noqa: BLE001 - never put private oracle details on stdout/stderr
        print(json.dumps({"passed": False, "status": "bridge_error", "correct": False,
                          "reward": None, "reference_instruction_count": None,
                          "candidate_instruction_count": None}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
