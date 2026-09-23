"""Instruction-count reward tests, including real sandboxed Cachegrind runs."""
import math

import pytest


def test_ir_parser_uses_event_order_not_a_fixed_column():
    from harness.perf import parse_ir

    assert parse_ir("events: Dr Ir Dw\nsummary: 9 12345 8\n") == 12345


@pytest.mark.parametrize("profile", [
    "", "events: Dr\nsummary: 12\n", "events: Ir\nsummary: 0\n",
    "events: Ir\nsummary: -1\n", "events: Ir Dr\nsummary: 7\n",
    "events: Ir\nsummary: nan\n", "events: Ir\nsummary: 1\nsummary: 2\n",
])
def test_ir_parser_fails_closed(profile):
    from harness.perf import parse_ir

    with pytest.raises(ValueError):
        parse_ir(profile)


def test_log_speedup_orders_improvements_and_does_not_reward_regression():
    from harness.perf import log_speedup

    assert log_speedup(100, 25) == pytest.approx(math.log(4))
    assert log_speedup(100, 25) > log_speedup(100, 50) > log_speedup(100, 100)
    assert log_speedup(100, 200) < 0
    with pytest.raises(ValueError):
        log_speedup(100, 0)


@pytest.mark.cachegrind
def test_real_instruction_counts_are_bit_identical():
    from harness.perf import measure

    result = measure("def solve(xs):\n    return len(set(xs))\n", "solve", [[list(range(800))]])
    assert result.ok, result
    assert len(result.counts) == 2
    assert result.counts[0] == result.counts[1] > 0
    assert result.count == result.counts[0]


def test_incorrect_candidate_gets_zero_without_calling_the_profiler():
    from pathlib import Path

    from harness.perf import evaluate
    from harness.problems import SPIKE_PAIRS
    from harness.sandbox import CachegrindConfig

    problem, _ = SPIKE_PAIRS[0]
    result = evaluate(problem, "def solve(xs):\n    return False\n", seed=31,
                      config=CachegrindConfig(Path("/nonexistent/valgrind")))
    assert not result.correct
    assert result.reward == 0.0
    assert result.status == "mismatch"
    assert result.reference_counts == result.candidate_counts == ()


@pytest.mark.cachegrind
def test_correct_set_rewrite_has_positive_repeatable_reward():
    from harness.perf import evaluate
    from harness.problems import SPIKE_PAIRS

    problem, fast = SPIKE_PAIRS[0]
    result = evaluate(problem, fast, seed=31)
    assert result.correct, result
    assert result.status == "ok", result
    assert result.reward > 0
    assert result.reference_counts[0] == result.reference_counts[1]
    assert result.candidate_counts[0] == result.candidate_counts[1]
    assert result.reference_counts[0] > result.candidate_counts[0]


def test_missing_profiler_is_not_zero_reward_training_data():
    from pathlib import Path

    from harness.perf import evaluate
    from harness.problems import SPIKE_PAIRS
    from harness.sandbox import CachegrindConfig

    problem, fast = SPIKE_PAIRS[0]
    result = evaluate(problem, fast, seed=31,
                      config=CachegrindConfig(Path("/nonexistent/valgrind")))
    assert result.correct
    assert result.reward is None
    assert result.status == "reference_sandbox_error"


def test_unstable_profiles_are_not_rounded_into_a_reward(monkeypatch):
    from harness import sandbox
    from harness.perf import measure

    profiles = iter(["events: Ir\nsummary: 100\n", "events: Ir\nsummary: 101\n"])
    monkeypatch.setattr(sandbox, "run_candidate", lambda *a, **kw:
                        sandbox.SandboxResult(True, profile=next(profiles)))
    result = measure("unused", "solve", [], config=object())
    assert not result.ok
    assert result.status == "unstable"
    assert result.counts == (100, 101)
    assert result.count is None
