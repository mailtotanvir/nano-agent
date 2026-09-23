"""Correctness-gate integration tests exercise real isolated subprocesses."""
from dataclasses import replace

from harness.problems import SPIKE_PAIRS


def test_genuine_fast_rewrite_passes_hidden_and_benchmark_inputs():
    from harness.correctness import evaluate_correctness
    problem, fast = SPIKE_PAIRS[0]
    result = evaluate_correctness(problem, fast, seed=77, random_cases=12)
    assert result.passed, result
    assert result.status == "passed"
    assert result.tested == len(problem.edge_inputs) + 12 + len(problem.benchmark_inputs)
    assert result.detail == "Finite test battery passed; semantic equivalence is not proven."



def test_visible_example_hardcoder_fails_without_leaking_hidden_data():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    result = evaluate_correctness(problem, "def solve(xs):\n    return False\n", seed=912345, random_cases=12)
    assert not result.passed
    assert result.status == "mismatch"
    assert 0 < result.tested <= len(problem.edge_inputs) + 12 + len(problem.benchmark_inputs)
    assert result.detail == "Candidate output differs from the reference."
    assert "912345" not in result.detail


def test_candidate_runtime_failure_is_not_a_pass_or_infrastructure_error():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    result = evaluate_correctness(problem, "def solve(xs):\n    return 1 // 0\n", seed=1)
    assert not result.passed
    assert result.status == "candidate_error"
    assert result.tested == 0
    assert result.detail == "Candidate execution failed."


def test_reference_failure_is_infrastructure_error_even_for_wrong_candidate():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    broken = replace(problem, reference="def solve(xs):\n    return 1 // 0\n")
    result = evaluate_correctness(broken, "def solve(xs):\n    return False\n", seed=1)
    assert not result.passed
    assert result.status == "infrastructure_error"
    assert result.tested == 0
    assert result.detail == "Reference execution failed."


def test_nested_json_bool_is_not_an_integer():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    typed = replace(problem, reference='def solve(xs):\n    return {"nested": [True, None, "x"]}\n')
    result = evaluate_correctness(typed, 'def solve(xs):\n    return {"nested": [1, None, "x"]}\n', seed=1)
    assert not result.passed
    assert result.status == "mismatch"


def test_nonfinite_reference_is_infrastructure_error():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    broken = replace(problem, reference="def solve(xs):\n    return 1e309\n")
    result = evaluate_correctness(broken, "def solve(xs):\n    return 0\n", seed=1)
    assert not result.passed
    assert result.status == "infrastructure_error"


def test_nonfinite_candidate_is_candidate_error():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    result = evaluate_correctness(problem, 'def solve(xs):\n    return {"bad": [1e309 - 1e309]}\n', seed=1)
    assert not result.passed
    assert result.status == "candidate_error"


def test_benchmark_size_special_case_cheat_fails_large_hidden_inputs():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    cheat = "def solve(xs):\n    if len(xs) == 800:\n        return False\n    return len(set(xs)) != len(xs)\n"
    result = evaluate_correctness(problem, cheat, seed=77, random_cases=0)
    assert not result.passed
    assert result.status == "mismatch"


def test_candidate_side_sandbox_failure_is_infrastructure_error():
    from unittest.mock import patch

    from harness import correctness
    from harness.sandbox import SandboxResult
    problem, fast = SPIKE_PAIRS[0]
    real_run = correctness.run_candidate
    calls = []
    def fail_candidate_only(source, function_name, inputs):
        calls.append(source)
        if len(calls) == 2:
            return SandboxResult(False, status="sandbox_error", detail="PRIVATE TRANSPORT ERROR")
        return real_run(source, function_name, inputs)
    # Fault injection only at the unavailable-transport boundary.
    with patch.object(correctness, "run_candidate", side_effect=fail_candidate_only):
        result = correctness.evaluate_correctness(problem, fast, seed=1, random_cases=0)
    assert result.status == "infrastructure_error"
    assert not result.passed
    assert "PRIVATE" not in result.detail


def test_truncated_worker_outputs_cannot_false_pass():
    from unittest.mock import patch

    from harness import correctness
    problem, fast = SPIKE_PAIRS[0]
    real_run = correctness.run_candidate
    for truncate_call in (1, 2):
        calls = []
        def truncate_transport(source, function_name, inputs, *, _calls=calls,
                               _truncate_call=truncate_call):
            result = real_run(source, function_name, inputs)
            _calls.append(source)
            if len(_calls) == _truncate_call:
                return replace(result, outputs=result.outputs[:-1])
            return result
        with patch.object(correctness, "run_candidate", side_effect=truncate_transport):
            result = correctness.evaluate_correctness(problem, fast, seed=1, random_cases=0)
        assert not result.passed
        assert result.status == "infrastructure_error"
        assert result.tested == 0


def test_invalid_random_case_counts_are_rejected():
    import pytest

    from harness.correctness import evaluate_correctness
    problem, fast = SPIKE_PAIRS[0]
    for invalid in (-1, True, 1.5, "4"):
        with pytest.raises(ValueError, match="random_cases"):
            evaluate_correctness(problem, fast, seed=1, random_cases=invalid)


def test_default_seed_comes_from_secure_parent_entropy():
    from unittest.mock import patch

    from harness import correctness
    problem, fast = SPIKE_PAIRS[0]
    with patch("harness.correctness.secrets.randbits", return_value=83) as entropy:
        result = correctness.evaluate_correctness(problem, fast, random_cases=3)
    entropy.assert_called_once_with(128)
    assert result.passed


def test_empty_battery_is_infrastructure_error_not_vacuous_pass():
    from harness.correctness import evaluate_correctness
    problem, fast = SPIKE_PAIRS[0]
    empty = replace(problem, edge_inputs=[], benchmark_inputs=[])
    result = evaluate_correctness(empty, fast, seed=1, random_cases=0)
    assert not result.passed
    assert result.status == "infrastructure_error"


def test_broken_private_generator_is_sanitized_infrastructure_error():
    from harness.correctness import evaluate_correctness
    problem, fast = SPIKE_PAIRS[0]
    def broken(rng):
        raise RuntimeError("PRIVATE SEED AND INPUT")
    invalid = replace(problem, _generator=broken)
    result = evaluate_correctness(invalid, fast, seed=1, random_cases=1)
    assert not result.passed
    assert result.status == "infrastructure_error"
    assert "PRIVATE" not in result.detail


def test_every_spike_fast_source_passes_real_isolated_battery():
    from harness.correctness import evaluate_correctness
    for problem, fast in SPIKE_PAIRS:
        result = evaluate_correctness(problem, fast, seed=2026)
        assert result.passed, (problem.id, result)
        assert result.tested == len(problem.hidden_inputs(2026, 64)) + len(problem.benchmark_inputs)


def test_wrong_benchmark_output_fails_even_when_hidden_outputs_are_right():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    benchmark = replace(problem, benchmark_inputs=[[[918273645, 918273645]]])
    cheat = "def solve(xs):\n    if xs == [918273645, 918273645]:\n        return False\n    return len(set(xs)) != len(xs)\n"
    result = evaluate_correctness(benchmark, cheat, seed=7, random_cases=0)
    assert not result.passed
    assert result.status == "mismatch"
    assert "918273645" not in result.detail


def test_candidate_input_mutation_cannot_poison_reference_or_parent_fixtures():
    from copy import deepcopy

    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    original = deepcopy(problem.benchmark_inputs)
    before = problem.hidden_inputs(9, 0)
    mutator = "def solve(xs):\n    xs.clear()\n    return False\n"
    first = evaluate_correctness(problem, mutator, seed=9, random_cases=0)
    second = evaluate_correctness(problem, mutator, seed=9, random_cases=0)
    assert first == second
    assert first.status == "mismatch"
    assert problem.benchmark_inputs == original
    assert problem.hidden_inputs(9, 0) == before


def test_reference_and_candidate_receive_separate_payloads_without_secrets():
    from unittest.mock import patch

    from harness import correctness
    problem, fast = SPIKE_PAIRS[0]
    real_run = correctness.run_candidate
    calls = []
    def record_real_run(source, function_name, inputs):
        calls.append((source, function_name, inputs))
        return real_run(source, function_name, inputs)
    with patch.object(correctness, "run_candidate", side_effect=record_real_run):
        result = correctness.evaluate_correctness(problem, fast, seed=937462, random_cases=0)
    assert result.passed
    assert len(calls) == 2
    assert calls[0][0] == problem.reference
    assert calls[1][0] == fast
    assert calls[0][2] == calls[1][2]
    assert calls[0][2] is not calls[1][2]
    assert calls[0][2][0] is not calls[1][2][0]


def test_json_comparison_preserves_container_structure_and_scalar_types():
    from harness.correctness import _same_json, _valid_json
    assert _same_json({"a": [1, None], "b": False}, {"b": False, "a": [1, None]})
    for expected, actual in [(True, 1), (1, 1.0), ([1], [1, 2]), ({"a": 1}, {"b": 1}), ([], {})]:
        assert not _same_json(expected, actual)
    for invalid in [(1,), {1: "x"}, {1}, float("inf"), float("nan")]:
        assert not _valid_json(invalid)
    assert _valid_json({"nested": [1.5, True, 1, None, "x"]})


def test_runner_exception_is_sanitized_infrastructure_error():
    from unittest.mock import patch

    from harness import correctness
    problem, fast = SPIKE_PAIRS[0]
    real_run = correctness.run_candidate
    for failure_call in (1, 2):
        calls = []
        def unavailable_runner(source, function_name, inputs, *, _calls=calls,
                               _failure_call=failure_call):
            _calls.append(source)
            if len(_calls) == _failure_call:
                raise OSError("PRIVATE FILESYSTEM PATH")
            return real_run(source, function_name, inputs)
        with patch.object(correctness, "run_candidate", side_effect=unavailable_runner):
            result = correctness.evaluate_correctness(problem, fast, seed=1, random_cases=0)
        assert result.status == "infrastructure_error"
        assert not result.passed
        assert "PRIVATE" not in result.detail


def test_non_json_return_values_are_rejected_before_lossy_serialization():
    from harness.correctness import evaluate_correctness
    problem, _ = SPIKE_PAIRS[0]
    for expression in ("(1, 2)", "{1: 2}", "{1, 2}"):
        candidate = "def solve(xs):\n    return " + expression + "\n"
        result = evaluate_correctness(problem, candidate, seed=1, random_cases=0)
        assert not result.passed
        assert result.status == "candidate_error", (expression, result)
        broken = replace(problem, reference=candidate)
        result = evaluate_correctness(broken, "def solve(xs):\n    return False\n", seed=1, random_cases=0)
        assert result.status == "infrastructure_error", (expression, result)
