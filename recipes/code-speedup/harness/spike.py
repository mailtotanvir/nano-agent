"""CPU-only reward spike. Persist every completed pair before starting the next."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .correctness import evaluate_correctness
from .perf import evaluate
from .problems import SPIKE_PAIRS


def run_spike(output: Path, *, seed: int = 314159, pairs=None) -> dict:
    """Run real measurements and retain partial evidence if interrupted."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace an earlier experiment, even if that experiment failed.
    with output.open("x") as stream:
        stream.write("{}\n")
    if pairs is None:
        pairs = SPIKE_PAIRS
    binary = os.environ.get("VALGRIND") or shutil.which("valgrind")
    runtime = {"kernel": platform.release(), "machine": platform.machine()}
    for name, command in [("python", ["/usr/bin/python3", "--version"]),
                          ("valgrind", [binary, "--version"] if binary else None),
                          ("bubblewrap", ["/usr/bin/bwrap", "--version"])]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False) if command else None
            runtime[name] = result.stdout.strip() if result and result.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            runtime[name] = "unavailable"
    source_dir = Path(__file__).parent
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "seed": seed, "repeats": 2, "runtime": runtime,
              "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                                for name in ["sandbox.py", "worker.py", "correctness.py",
                                             "problems.py", "perf.py", "spike.py"]},
              "pairs": [], "attacks": [], "summary": {"gate": "INCOMPLETE"}}

    def persist():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    persist()
    for problem, fast in pairs:
        started = time.monotonic()
        result = evaluate(problem, fast, seed=seed)
        row = {"id": problem.id, "family": problem.family, **asdict(result),
               "elapsed_seconds": time.monotonic() - started,
               "reference_sha256": hashlib.sha256(problem.reference.encode()).hexdigest(),
               "candidate_sha256": hashlib.sha256(fast.encode()).hexdigest(),
               "benchmark_sha256": hashlib.sha256(json.dumps(problem.benchmark_inputs).encode()).hexdigest()}
        report["pairs"].append(row)
        persist()
        print(f"{problem.id}: {result.status}, reward={result.reward}", flush=True)
    problem, _ = SPIKE_PAIRS[0]
    attacks = {
        "visible_example_hardcode": "def solve(xs):\n    return False\n",
        "benchmark_size_hardcode": problem.reference.replace(
            "def solve(xs):\n", "def solve(xs):\n    if len(xs) == 800:\n        return False\n"),
    }
    for name, source in attacks.items():
        result = evaluate_correctness(problem, source, seed=seed)
        report["attacks"].append({"id": name, **asdict(result)})
        persist()
    report["summary"] = summarize(report["pairs"], [problem.id for problem, _ in pairs], report["attacks"])
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    persist()
    return report


def summarize(rows: list[dict], expected_ids: list[str], attacks: list[dict]) -> dict:
    """Fail closed on missing/duplicate pairs, noise, wrong code or attack success."""
    ids = [row["id"] for row in rows]
    complete = bool(expected_ids) and sorted(ids) == sorted(set(expected_ids))
    wins = []
    for row in rows:
        reference = row.get("reference_counts", [])
        candidate = row.get("candidate_counts", [])
        stable = (len(reference) >= 2 and len(candidate) >= 2
                  and len(set(reference)) == len(set(candidate)) == 1)
        if (row.get("correct") and row.get("status") == "ok" and stable
                and reference[0] > candidate[0] > 0 and (row.get("reward") or 0) > 0):
            wins.append(row["id"])
    rejected = bool(attacks) and all(not row["passed"] and row["status"] == "mismatch"
                                    for row in attacks)
    go = complete and len(wins) == len(expected_ids) and rejected
    return {"gate": "GO_REWARD_SIGNAL" if go else "NO_GO",
            "expected_pairs": len(expected_ids), "collected_pairs": len(rows),
            "stable_positive_pairs": len(wins), "attack_rejections": rejected,
            "scope": "finite local evidence only; not a proof of security or equivalence"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new evidence JSON file (never overwritten)")
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    report = run_spike(args.out, seed=args.seed)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["gate"] == "GO_REWARD_SIGNAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
