#!/usr/bin/env python3
"""Generate verified code-speed teacher trajectories with Gemini API-key auth.

This script only accepts the training split. For every case it writes a private
``candidate.py`` initialized from the slow reference, invokes the same Rust
controller/verifier that the local model uses, then retains only success
trajectories whose final private bridge measurement has a positive reward.
Neither eval split is accepted or read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gemini-3.8-flash"
SAFE_VERIFIER_FIELDS = {
    "passed", "status", "correct", "reward", "reference_instruction_count",
    "candidate_instruction_count",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


TRAINING_SPLIT_HINTS = {"train_dev", "train_v2", "train_v6"}


def load_train_cases(path: Path) -> list[dict[str, Any]]:
    """Load only verifier-gated training rows; reject any evaluation-shaped input.

    Accepts the v1 ``train_speedup_*`` split (``split_hint`` ``train_dev``) and the
    v2/v6 training-only corpus (``corpus_speedup_v*``, split hints
    ``train_v2``/``train_v6``).
    Any evaluation-shaped file is refused before a single record is read: the
    frozen v1 evals and the ``eval_speedup_v2_dev_behavioral`` split both carry
    ``eval`` in the name, so the teacher can never be pointed at a measurement set.
    """
    name = path.name.lower()
    if "eval" in name or "dev_behavioral" in name:
        raise ValueError("teacher refuses an evaluation split; training corpus only")
    if "train" not in name and "corpus" not in name:
        raise ValueError("teacher requires a train/corpus JSONL input, never an eval split")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("training JSONL is empty")
    seen = set()
    for row in rows:
        if row.get("id") in seen:
            raise ValueError("duplicate training record ID")
        seen.add(row.get("id"))
        if row.get("split_hint") not in TRAINING_SPLIT_HINTS:
            raise ValueError("training input contains a non-train record")
        if row.get("verification", {}).get("status") != "passed":
            raise ValueError("training input must contain only verifier-gated records")
        if not isinstance(row.get("reference_source"), str):
            raise TypeError("training record lacks reference source")
    return rows


def _completed_case_ids(state: Path, out: Path, *, retry_failed: bool) -> set[str]:
    completed = set()
    if state.exists():
        for line in state.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("outcome") == "success" or not retry_failed:
                completed.add(row["case_id"])
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["case_id"])
    return completed


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _safe_bridge_result(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) - SAFE_VERIFIER_FIELDS:
        raise ValueError("bridge response contained an unexpected field")
    if not SAFE_VERIFIER_FIELDS <= set(payload):
        raise ValueError("bridge response omitted a required safe field")
    return {field: payload[field] for field in sorted(SAFE_VERIFIER_FIELDS)}


def _failure_diagnostic(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Keep enough non-oracle metadata to diagnose rejected teacher episodes."""
    return {
        "attempts": trajectory.get("attempts"),
        "steps": [
            {
                "attempt": step.get("attempt"),
                "action": step.get("proposal", {}).get("action"),
                "reason": step.get("proposal", {}).get("reason"),
                "patch_applied": step.get("patch_applied"),
                "reverted": step.get("reverted"),
                "primary_code": step.get("primary_code"),
                "error_count_after": step.get("error_count_after"),
            }
            for step in trajectory.get("steps", [])
        ],
    }


def _run_bridge(python: Path, bridge: Path, recipe_root: Path, train: Path,
                case_id: str, candidate: Path,
                bridge_container_image: str | None = None) -> dict[str, Any] | None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(recipe_root) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        if bridge_container_image:
            root = recipe_root.resolve()
            bridge_rel = bridge.resolve().relative_to(root)
            train_rel = train.resolve().relative_to(root)
            command = [
                "docker", "run", "--rm", "--privileged", "--network", "none",
                "-e", "PYTHONPATH=/repo", "-v", f"{root}:/repo:ro",
                "-v", f"{candidate.parent.resolve()}:/candidate:ro",
                bridge_container_image, "python3", f"/repo/{bridge_rel}",
                "--dataset", f"/repo/{train_rel}", "--problem-id", case_id,
                "--candidate", f"/candidate/{candidate.name}",
            ]
        else:
            command = [str(python), str(bridge), "--dataset", str(train),
                       "--problem-id", case_id, "--candidate", str(candidate)]
        result = subprocess.run(
            command,
            env=env, capture_output=True, text=True, timeout=600, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        return _safe_bridge_result(json.loads(result.stdout))
    except (json.JSONDecodeError, ValueError):
        return None


def run_case(*, binary: Path, bridge: Path, python: Path, recipe_root: Path,
             train: Path, case: dict[str, Any], backend: str, model: str,
             vertex_project: str | None, vertex_location: str,
             max_attempts: int, bridge_container_image: str | None = None) -> dict[str, Any]:
    """Run one controller episode; returned values are safe to persist in state."""
    case_id = case["id"]
    with tempfile.TemporaryDirectory(prefix="code_speed_teacher_") as temp:
        workspace = Path(temp)
        candidate = workspace / "candidate.py"
        candidate.write_text(case["reference_source"], encoding="utf-8")
        trajectory_log = workspace / "trajectory.jsonl"
        command = [
            str(binary), "--workspace", str(workspace), "--dataset", str(train),
            "--problem-id", case_id, "--backend", backend, "--model", model,
            "--max-attempts", str(max_attempts), "--log", str(trajectory_log),
        ]
        if backend == "gemini-vertex":
            command.extend(["--vertex-project", vertex_project or "",
                            "--vertex-location", vertex_location])
        if bridge_container_image:
            command.extend(["--bridge-container-image", bridge_container_image])
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
        except subprocess.TimeoutExpired:
            return {"case_id": case_id, "outcome": "timeout"}
        except OSError:
            return {"case_id": case_id, "outcome": "controller_error"}
        if not trajectory_log.exists():
            return {"case_id": case_id, "outcome": "controller_error"}
        lines = [line for line in trajectory_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return {"case_id": case_id, "outcome": "controller_error"}
        try:
            trajectory = json.loads(lines[-1])
        except json.JSONDecodeError:
            return {"case_id": case_id, "outcome": "controller_error"}
        outcome = trajectory.get("outcome", "controller_error")
        if outcome != "success":
            return {
                "case_id": case_id,
                "outcome": outcome,
                "diagnostic": _failure_diagnostic(trajectory),
            }
        result = _run_bridge(python, bridge, recipe_root, train, case_id, candidate,
                             bridge_container_image)
        if result is None or not (result["passed"] and result["correct"]
                                  and isinstance(result["reward"], (int, float))
                                  and result["reward"] > 0):
            return {"case_id": case_id, "outcome": "verification_rejected"}
        return {
            "case_id": case_id,
            "outcome": "success",
            "trajectory": trajectory,
            "verifier": result,
        }


def _stats_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".stats.json")


def _run_pending(task, pending: list[dict[str, Any]], workers: int):
    if workers == 1:
        for case in pending:
            yield task(case)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task, case) for case in pending]
        for future in as_completed(futures):
            yield future.result()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True,
                        help="built code-speedup controller binary")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--bridge", type=Path, default=Path(__file__).parents[1] / "harness/bridge.py")
    parser.add_argument("--python", type=Path, default=Path("python3"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("gemini", "gemini-vertex"), default="gemini")
    parser.add_argument("--vertex-project")
    parser.add_argument("--vertex-location", default="global")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all remaining cases")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--bridge-container-image", default=None,
                        help="network-disabled verifier image for OCI-hosted runs")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_attempts < 1 or args.max_cases < 0:
        raise ValueError("workers/max-attempts must be positive and max-cases nonnegative")
    if not args.binary.is_file():
        raise ValueError(f"controller binary not found: {args.binary}")
    if not args.bridge.is_file():
        raise ValueError(f"bridge not found: {args.bridge}")
    if args.backend == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for the Gemini teacher")
    if args.backend == "gemini-vertex" and not args.vertex_project:
        raise ValueError("--vertex-project is required for the Gemini Vertex teacher")
    train = args.train.resolve()
    cases = load_train_cases(train)
    out = args.out
    state = args.state or out.with_suffix(out.suffix + ".state.jsonl")
    completed = _completed_case_ids(state, out, retry_failed=args.retry_failed)
    pending = [case for case in cases if case["id"] not in completed]
    if args.max_cases:
        pending = pending[:args.max_cases]
    recipe_root = Path(__file__).parents[1]
    task = lambda case: run_case(binary=args.binary, bridge=args.bridge, python=args.python,
                                 recipe_root=recipe_root, train=train, case=case,
                                 backend=args.backend, model=args.model,
                                 vertex_project=args.vertex_project,
                                 vertex_location=args.vertex_location,
                                 max_attempts=args.max_attempts,
                                 bridge_container_image=args.bridge_container_image)
    successes = 0
    outcomes: dict[str, int] = {}
    attempted = 0
    for index, result in enumerate(_run_pending(task, pending, args.workers), start=1):
        attempted = index
        outcome = result["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == "success":
            # Trajectory contexts are model-facing; the original private record
            # and its benchmark/edge/generator data are never copied here.
            _append_jsonl(out, {
                "schema_version": 1,
                "case_id": result["case_id"],
                "teacher": {"backend": args.backend, "model": args.model},
                "trajectory": result["trajectory"],
                "verifier": result["verifier"],
            })
            successes += 1
        # Success rows reach the durable output before state so an interruption
        # cannot mark a trajectory complete without retaining it. Existing
        # output rows also participate in resume de-duplication.
        state_row = {"case_id": result["case_id"], "outcome": outcome}
        if "diagnostic" in result:
            state_row["diagnostic"] = result["diagnostic"]
        _append_jsonl(state, state_row)
        print(f"[{index}/{len(pending)}] case={result['case_id']} outcome={outcome} kept={successes}", flush=True)
    stats = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_train_sha256": _sha256_file(train),
        "controller_binary_sha256": _sha256_file(args.binary),
        "bridge_sha256": _sha256_file(args.bridge),
        "teacher_backend": args.backend,
        "teacher_model": args.model,
        "requested_cases": len(cases),
        "resumed_completed": len(completed),
        "attempted": attempted,
        "successes_kept": successes,
        "outcomes": outcomes,
        "output": str(out),
    }
    stats_path = args.stats or _stats_path(out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
