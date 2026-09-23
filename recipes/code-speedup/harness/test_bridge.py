import json


def test_bridge_loads_only_selected_verified_record_and_sanitizes_model_context(tmp_path):
    from harness.bridge import load_problem_record, model_context, safe_result

    record = {
        "id": "fixture-1",
        "family_id": "hash-membership",
        "function_name": "solve",
        "reference_source": "def solve(xs):\n    return False\n",
        "input_generator": {"kind": "int_list", "version": 1},
        "edge_inputs": [[[]]],
        "benchmark_variants": [{"id": "n", "inputs": [[1, 2, 3]]}],
        "verification": {"status": "passed"},
        "private_seed_policy": "private",
        "known_fast_source": "must never reach a model",
    }
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(json.dumps(record) + "\n")
    selected = load_problem_record(dataset, "fixture-1")
    response = safe_result({
        "correct": True, "reward": 0.4, "status": "ok",
        "reference_counts": [120], "candidate_counts": [80],
    })
    context = model_context(selected, "def solve(xs):\n    return len(set(xs)) != len(xs)\n", response)
    assert "reference_source" not in context
    assert "return False" in context
    assert "len(set(xs))" in context
    assert "benchmark_variants" not in context
    assert "edge_inputs" not in context
    assert "input_generator" not in context
    assert "private" not in context
    assert "must never" not in context
    assert "reference_instruction_count: 120" in context
    assert "candidate_instruction_count: 80" in context


def test_safe_result_requires_positive_correct_reward_and_hides_detail():
    from harness.bridge import safe_result

    no_speedup = safe_result({
        "correct": True, "reward": 0.0, "status": "ok", "detail": "private input [1, 2]",
        "reference_counts": [10], "candidate_counts": [10],
    })
    assert no_speedup == {
        "passed": False, "status": "no_speedup", "correct": True, "reward": 0.0,
        "reference_instruction_count": 10, "candidate_instruction_count": 10,
    }
    failed = safe_result({
        "correct": False, "reward": 5.0, "status": "mismatch", "detail": "private"})
    assert failed["passed"] is False
    assert failed["status"] == "mismatch"
    assert "detail" not in failed
