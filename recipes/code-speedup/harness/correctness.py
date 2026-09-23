"""Parent-side finite correctness battery; never a proof of equivalence."""
import math
import secrets
from copy import deepcopy
from dataclasses import dataclass

from .problems import Problem
from .sandbox import run_candidate


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    status: str
    tested: int
    detail: str


def _valid_json(value):
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_valid_json(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _valid_json(item) for key, item in value.items())
    return False


def _same_json(expected, observed):
    if type(expected) is not type(observed):
        return False
    if type(expected) is list:
        return len(expected) == len(observed) and all(
            _same_json(a, b) for a, b in zip(expected, observed))
    if type(expected) is dict:
        return expected.keys() == observed.keys() and all(
            _same_json(expected[key], observed[key]) for key in expected)
    return expected == observed


def evaluate_correctness(problem: Problem, candidate: str, *, seed: int | None = None,
                         random_cases: int = 64) -> CorrectnessResult:
    """Check private edges, random cases, and benchmarks in separate workers.

    A supplied seed is for reproducible trusted tests; omit it in scoring.
    Infrastructure errors must not become negative training examples.
    ``tested`` counts the complete compared batch (zero on execution failure).
    Details intentionally omit inputs, outputs, seeds, case IDs, and exceptions.
    """
    if type(random_cases) is not int or random_cases < 0:
        raise ValueError("random_cases must be a nonnegative integer")
    if seed is None:
        seed = secrets.randbits(128)
    try:
        inputs = problem.hidden_inputs(seed, random_cases) + deepcopy(problem.benchmark_inputs)
    except Exception:  # noqa: BLE001 - trusted generator failures must fail closed
        return CorrectnessResult(False, "infrastructure_error", 0, "Input generation failed.")
    if not inputs:
        return CorrectnessResult(False, "infrastructure_error", 0, "Empty correctness battery.")
    try:
        reference = run_candidate(problem.reference, problem.function_name, deepcopy(inputs))
    except Exception:  # noqa: BLE001 - isolate unexpected runner failures
        return CorrectnessResult(False, "infrastructure_error", 0, "Reference isolation failed.")
    if not reference.ok or not _valid_json(reference.outputs):
        return CorrectnessResult(False, "infrastructure_error", 0, "Reference execution failed.")
    if type(reference.outputs) is not list or len(reference.outputs) != len(inputs):
        return CorrectnessResult(False, "infrastructure_error", 0, "Reference output protocol failed.")
    try:
        actual = run_candidate(candidate, problem.function_name, deepcopy(inputs))
    except Exception:  # noqa: BLE001 - isolate unexpected runner failures
        return CorrectnessResult(False, "infrastructure_error", 0, "Candidate isolation failed.")
    if actual.status == "sandbox_error":
        return CorrectnessResult(False, "infrastructure_error", 0, "Candidate isolation failed.")
    if not actual.ok or not _valid_json(actual.outputs):
        return CorrectnessResult(False, "candidate_error", 0, "Candidate execution failed.")
    if type(actual.outputs) is not list or len(actual.outputs) != len(inputs):
        return CorrectnessResult(False, "infrastructure_error", 0, "Candidate output protocol failed.")
    for expected, observed in zip(reference.outputs, actual.outputs):
        if not _same_json(expected, observed):
            return CorrectnessResult(False, "mismatch", len(inputs),
                                     "Candidate output differs from the reference.")
    return CorrectnessResult(True, "passed", len(inputs),
                             "Finite test battery passed; semantic equivalence is not proven.")
