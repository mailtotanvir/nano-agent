import json
from itertools import pairwise

import pytest


def test_normalized_hash_ignores_layout_and_local_identifier_spelling():
    from datagen.gen_problems import normalized_ast_hash

    left = """def solve(values):\n    total = 0\n    for item in values:\n        total += item\n    return total\n"""
    right = """def solve(xs):\n\tacc=0\n\tfor value in xs:\n\t\tacc += value\n\treturn acc\n"""
    changed_literal = right.replace("return acc", "return acc + 1")
    assert normalized_ast_hash(left) == normalized_ast_hash(right)
    assert normalized_ast_hash(left) != normalized_ast_hash(changed_literal)


def test_seed_corpus_is_deterministic_and_never_serializes_fast_source():
    from datagen.gen_problems import generate_pairs, public_record

    first = generate_pairs(n=250, seed=20260907)
    second = generate_pairs(n=250, seed=20260907)
    assert [pair.record["id"] for pair in first] == [pair.record["id"] for pair in second]
    assert len(first) == 250
    assert len({pair.record["id"] for pair in first}) == 250
    assert all(pair.record["normalized_ast_sha256"] for pair in first)
    assert all(pair.record["family_id"] and pair.record["template_id"] for pair in first)
    serialized = "\n".join(json.dumps(public_record(pair)) for pair in first)
    assert "known_fast_source" not in serialized
    assert "fast_source" not in serialized
    assert all(pair.fast_source not in serialized for pair in first)


@pytest.mark.parametrize(
    ("family", "args"),
    [
        ("hash-membership", [[0, 4]]),
        ("distinct-cardinality", [[0, 4]]),
        ("stable-deduplication", [[0, 4]]),
        ("frequency-index", [[0, 4]]),
        ("complement-lookup", [[0, 4], 0]),
        ("running-aggregate", [[0]]),
        ("prefix-range-query", [[0], [[0, 1]]]),
        ("fixed-sliding-window", [[0], 1]),
        ("string-concatenation", [["x"]]),
        ("sort-selection", [[0]]),
        ("indexed-lookup", [[[0, 0]], [0]]),
    ],
)
def test_semantic_instance_literals_change_observable_behavior_not_only_hashes(family, args):
    from datagen.gen_problems import _source_pair

    one, one_fast, _ = _source_pair(family, 0, 2)
    two, two_fast, _ = _source_pair(family, 0, 3)
    one_ns, one_fast_ns, two_ns, two_fast_ns = {}, {}, {}, {}
    exec(one, one_ns)  # noqa: S102 - trusted generated fixture
    exec(one_fast, one_fast_ns)  # noqa: S102 - trusted generated fixture
    exec(two, two_ns)  # noqa: S102 - trusted generated fixture
    exec(two_fast, two_fast_ns)  # noqa: S102 - trusted generated fixture
    assert one_ns["solve"](*args) == one_fast_ns["solve"](*args)
    assert two_ns["solve"](*args) == two_fast_ns["solve"](*args)
    assert one_ns["solve"](*args) != two_ns["solve"](*args)
    assert "marker" not in one


def test_records_supply_private_generators_and_adversarial_benchmark_neighbours():
    from datagen.gen_problems import generate_pairs, problem_from_record

    pair = generate_pairs(n=250, seed=20260907)[0]
    problem = problem_from_record(pair.record)
    assert problem.hidden_inputs(5, 4) == problem.hidden_inputs(5, 4)
    variants = pair.record["benchmark_variants"]
    names = {variant["id"] for variant in variants}
    assert {"n_minus_1", "n", "n_plus_1", "values_changed", "order_changed"} <= names
    assert len(problem.benchmark_inputs) == len(variants)


def test_model_prompt_record_is_an_exact_allowlist_without_oracle_metadata():
    from datagen.gen_problems import generate_pairs, model_prompt_record

    record = generate_pairs(n=250, seed=20260907)[0].record
    prompt = model_prompt_record(record)
    assert set(prompt) == {"id", "family_id", "template_id", "function_name", "reference_source"}
    assert not ({"benchmark_variants", "edge_inputs", "input_generator", "verification",
                 "private_seed_policy", "semantic_variant"} & set(prompt))


def test_verify_uses_fast_source_only_in_memory_and_rejects_nonpositive_results():
    from datagen.gen_problems import generate_pairs, verify_pairs

    pairs = generate_pairs(n=250, seed=20260907)[:2]
    calls = []

    class Result:
        correct = True
        reward = 0.4
        status = "ok"
        tested = 7
        reference_counts = (100, 100)
        candidate_counts = (60, 60)

    def evaluator(problem, candidate, **kwargs):
        calls.append((problem, candidate, kwargs))
        return Result()

    verified = verify_pairs(pairs, evaluator=evaluator)
    assert len(calls) == 2
    assert all(item.record["verification"]["status"] == "passed" for item in verified)
    assert all("fast_source" not in item.record for item in verified)


def test_verification_checkpoint_resumes_only_when_content_and_config_match(tmp_path):
    import copy

    from datagen.gen_problems import (
        GeneratedPair,
        generate_pairs,
        verification_key,
        verify_pairs,
    )

    pairs = generate_pairs(n=250, seed=20260907)[:2]
    checkpoint = tmp_path / "verification.jsonl"
    calls, progress = [], []

    class Result:
        correct = True
        reward = 0.4
        status = "ok"
        tested = 7
        reference_counts = (100, 100)
        candidate_counts = (60, 60)

    def evaluator(problem, candidate, **kwargs):
        calls.append((problem.id, candidate, kwargs))
        return Result()

    verify_pairs(pairs, evaluator=evaluator, checkpoint_path=checkpoint,
                 progress=lambda *args: progress.append(args))
    assert len(calls) == 2
    assert [entry[3] for entry in progress] == ["verified", "verified"]
    assert len(checkpoint.read_text().splitlines()) == 2
    # The cache keys the fast implementation by hash, never by source text.
    assert all(pair.fast_source not in checkpoint.read_text() for pair in pairs)

    def unexpected(*args, **kwargs):
        raise AssertionError("matching checkpoint rows must not be evaluated again")

    resumed = []
    verify_pairs(pairs, evaluator=unexpected, checkpoint_path=checkpoint,
                 progress=lambda *args: resumed.append(args))
    assert [entry[3] for entry in resumed] == ["cached", "cached"]
    assert verification_key(pairs[0]) != verification_key(pairs[0], {
        "schema_version": 1, "random_cases": 3, "repeats": 2, "seed": 314159,
        "python_executable": "test", "valgrind": "", "valgrind_lib": "",
    })
    original_key = verification_key(pairs[0])
    for field, mutate in [
        ("benchmark_variants", lambda value: value[0]["inputs"][0].append(999)),
        ("edge_inputs", lambda value: value[0][0].append(999)),
        ("input_generator", lambda value: value.__setitem__("kind", "strings")),
    ]:
        changed = copy.deepcopy(pairs[0].record)
        mutate(changed[field])
        assert verification_key(GeneratedPair(changed, pairs[0].fast_source)) != original_key


def test_failed_verification_reports_failed_before_raising(tmp_path):
    import pytest

    from datagen.gen_problems import generate_pairs, verify_pairs

    class Failed:
        correct = True
        reward = -0.01
        status = "ok"
        tested = 7
        reference_counts = (100, 100)
        candidate_counts = (101, 101)

    states = []
    with pytest.raises(RuntimeError, match="verification failed"):
        verify_pairs(generate_pairs(n=250, seed=20260907)[:1],
                     evaluator=lambda *args, **kwargs: Failed(),
                     checkpoint_path=tmp_path / "failed.jsonl",
                     progress=lambda *args: states.append(args))
    assert [state[3] for state in states] == ["failed"]


def test_all_heldout_sort_rows_match_fast_source_on_every_benchmark_variant():
    from datagen.gen_problems import generate_pairs

    pairs = [pair for pair in generate_pairs(n=250, seed=20260907)
             if pair.record["family_id"] == "sort-selection"]
    assert len(pairs) == 10
    for pair in pairs:
        slow_namespace, fast_namespace = {}, {}
        exec(pair.record["reference_source"], slow_namespace)  # noqa: S102 - trusted fixture
        exec(pair.fast_source, fast_namespace)  # noqa: S102 - trusted fixture
        for variant in pair.record["benchmark_variants"]:
            args = variant["inputs"]
            assert slow_namespace["solve"](*args) == fast_namespace["solve"](*args), pair.record["id"]
        variants = {variant["id"]: variant["inputs"][0] for variant in pair.record["benchmark_variants"]}
        assert {len(variants[name]) for name in ("n_minus_1", "n", "n_plus_1")} == {19_999, 20_000, 20_001}
        # These must not accidentally be monotonic Timsort best cases.
        assert all(0 < sum(left < right for left, right in pairwise(values)) < len(values) - 1
                   for values in variants.values())


def test_all_public_sources_are_valid_python_and_have_distinct_source_hashes_within_splits():
    import ast

    from datagen.gen_problems import generate_pairs

    pairs = generate_pairs(n=250, seed=20260907)
    for pair in pairs:
        ast.parse(pair.record["reference_source"])
    assert len({pair.record["normalized_ast_sha256"] for pair in pairs}) == len(pairs)


def test_every_generated_known_fast_rewrite_matches_its_slow_reference_on_trusted_cases():
    """Fast code is checked before serialization; this is not semantic proof."""
    import copy

    from datagen.gen_problems import generate_pairs, problem_from_record

    for pair in generate_pairs(n=250, seed=20260907):
        problem = problem_from_record(pair.record)
        reference, candidate = {}, {}
        exec(pair.record["reference_source"], reference)  # noqa: S102 - trusted generator output
        exec(pair.fast_source, candidate)  # noqa: S102 - trusted generator output
        for args in problem.hidden_inputs(418, 3):
            assert reference["solve"](*copy.deepcopy(args)) == candidate["solve"](*copy.deepcopy(args)), pair.record["id"]
