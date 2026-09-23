"""Cachegrind Ir measurement and correctness-gated log-speedup reward.

These are simulated instruction references, not hardware retired instructions or
elapsed time. Whole-process counts include fixed Python startup/harness costs.
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .problems import Problem


@dataclass(frozen=True)
class RewardResult:
    correct: bool
    reward: float | None
    status: str
    tested: int
    reference_counts: tuple[int, ...] = ()
    candidate_counts: tuple[int, ...] = ()
    detail: str = ""


def evaluate(problem: Problem, candidate: str, *, seed=None, random_cases: int = 64,
             repeats: int = 2, config=None) -> RewardResult:
    """Correctness first. A missing reward denotes an unusable measurement.

    A reference failure is an infrastructure error, never a model failure.
    Hidden values and expected outputs are not returned to a model caller.
    """
    from .correctness import evaluate_correctness

    correctness = evaluate_correctness(problem, candidate, seed=seed,
                                       random_cases=random_cases)
    if not correctness.passed:
        reward = None if correctness.status == "infrastructure_error" else 0.0
        return RewardResult(False, reward, correctness.status, correctness.tested,
                            detail=correctness.detail)
    reference = measure(problem.reference, problem.function_name, problem.benchmark_inputs,
                        repeats=repeats, config=config)
    if not reference.ok:
        return RewardResult(True, None, "reference_" + reference.status, correctness.tested,
                            reference.counts, detail=reference.detail)
    measured = measure(candidate, problem.function_name, problem.benchmark_inputs,
                       repeats=repeats, config=config)
    if not measured.ok:
        return RewardResult(True, None, "candidate_" + measured.status, correctness.tested,
                            reference.counts, measured.counts, measured.detail)
    return RewardResult(True, log_speedup(reference.counts[0], measured.counts[0]),
                        "ok", correctness.tested, reference.counts, measured.counts)


@dataclass(frozen=True)
class Measurement:
    ok: bool
    counts: tuple[int, ...] = ()
    status: str = "ok"
    detail: str = ""

    @property
    def count(self) -> int | None:
        return self.counts[0] if self.ok else None


def measure(source: str, function_name: str, inputs: list[list], *,
            repeats: int = 2, config=None) -> Measurement:
    """Require exact repeatability in fresh isolated processes, never round it.

    VALGRIND selects the executable; VALGRIND_LIB optionally selects the tool
    directory for a locally extracted Valgrind package. No tool is installed or
    downloaded automatically. Counts from failed/unstable runs are not rewards.
    """
    from .sandbox import CachegrindConfig, run_candidate

    if repeats < 2:
        raise ValueError("at least two independent measurements are required")
    if config is None:
        binary = os.environ.get("VALGRIND") or shutil.which("valgrind")
        if not binary:
            return Measurement(False, status="tool_unavailable", detail="Valgrind is required")
        lib_dir = os.environ.get("VALGRIND_LIB")
        config = CachegrindConfig(Path(binary), Path(lib_dir) if lib_dir else None)
    counts = []
    for _ in range(repeats):
        result = run_candidate(source, function_name, inputs, cachegrind=config,
                               timeout_s=30, cpu_seconds=20)
        if not result.ok:
            return Measurement(False, tuple(counts), result.status, result.detail)
        try:
            counts.append(parse_ir(result.profile or ""))
        except ValueError as exc:
            return Measurement(False, tuple(counts), "profile_error", str(exc))
    if len(set(counts)) != 1:
        return Measurement(False, tuple(counts), "unstable", "instruction counts differ")
    return Measurement(True, tuple(counts))


def parse_ir(profile: str) -> int:
    """Read exactly one complete Cachegrind event summary; never infer success."""
    events = [line.split()[1:] for line in profile.splitlines()
              if line.startswith("events:")]
    summaries = [line.split()[1:] for line in profile.splitlines()
                 if line.startswith("summary:")]
    if len(events) != 1 or len(summaries) != 1 or events[0].count("Ir") != 1:
        raise ValueError("missing or ambiguous Cachegrind summary")
    if len(events[0]) != len(summaries[0]):
        raise ValueError("incomplete Cachegrind summary")
    values = [int(value) for value in summaries[0]]
    if any(value < 0 for value in values):
        raise ValueError("negative instruction/cache count")
    count = values[events[0].index("Ir")]
    if count <= 0:
        raise ValueError("nonpositive instruction count")
    return count


def log_speedup(reference: int, candidate: int) -> float:
    """Raw log ratio; slower correct candidates are negative, equal cost is zero.

    Log compresses large ratios but is not mathematically bounded. Training must
    separately handle invalid candidates: a zero reward beats a negative one.
    """
    if reference <= 0 or candidate <= 0:
        raise ValueError("instruction counts must be positive")
    return math.log(reference / candidate)
