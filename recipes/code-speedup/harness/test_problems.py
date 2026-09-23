"""Trusted-parent tests for the hand-built spike examples."""
import copy
import json
import random


def pair(problem_id):
    from harness.problems import SPIKE_PAIRS
    found = [item for item in SPIKE_PAIRS if item[0].id == problem_id]
    assert len(found) == 1, f"Missing unique spike pair: {problem_id}"
    return found[0]


def trusted_call(source, args):
    # Only hand-authored fixtures, never model-generated candidate source.
    namespace = {}
    exec(source, namespace)  # noqa: S102 - trusted, hand-authored fixture only
    return namespace["solve"](*copy.deepcopy(args))


def check_pair(problem_id, oracle, explicit):
    problem, fast = pair(problem_id)
    state = random.getstate()
    cases = problem.hidden_inputs(1729, 40)
    assert random.getstate() == state
    assert cases == problem.hidden_inputs(1729, 40)
    assert cases != problem.hidden_inputs(1730, 40)
    assert len(cases) == len(problem.edge_inputs) + 40
    assert len(problem.hidden_inputs(1729, 0)) == len(problem.edge_inputs)
    for args in explicit + cases + problem.benchmark_inputs:
        expected = oracle(*args)
        assert trusted_call(problem.reference, args) == expected
        assert trusted_call(fast, args) == expected
        json.dumps(args, allow_nan=False)
    assert 300 <= max(len(args[0]) for args in problem.benchmark_inputs) <= 1000


def test_has_duplicate_pair_and_seeded_hidden_battery():
    check_pair("has_duplicate", lambda xs: len(xs) != len(set(xs)),
               [[[]], [[1]], [[1, 2, 1]], [[-1, 0, -1]]])



def test_count_distinct_pair():
    check_pair("count_distinct", lambda xs: len(set(xs)), [[[]], [[5, 1, 5, 1, -1]]])


def test_stable_unique_pair():
    check_pair("stable_unique", lambda xs: list(dict.fromkeys(xs)), [[[]], [[4, 2, 4, -1, 2]]])


def test_frequency_counts_pair():
    from collections import Counter
    check_pair("frequency_counts", lambda xs: [Counter(xs)[x] for x in xs],
               [[[]], [[4, 3, 4, 2, 3, 4]]])


def test_ordered_intersection_pair():
    check_pair("ordered_intersection", lambda xs, ys: list(dict.fromkeys(x for x in xs if x in ys)),
               [[[3, 2, 1, 2], [1, 2]], [[0, -1, 0], [-1, 0]]])


def test_pair_sum_pair():
    from itertools import combinations
    check_pair("pair_sum", lambda xs, target: any(a + b == target for a, b in combinations(xs, 2)),
               [[[3], 6], [[3, 3], 6], [[], 3]])


def test_prefix_sums_pair():
    from itertools import accumulate
    check_pair("prefix_sums", lambda xs: list(accumulate(xs)), [[[]], [[3, -5, 2, 7]]])


def test_range_sums_pair():
    check_pair("range_sums", lambda xs, queries: [sum(xs[i] for i in range(l, r)) for l, r in queries],
               [[[1, -2, 4], [[0, 3], [1, 1], [1, 3]]]])


def test_first_unique_pair():
    from collections import Counter
    check_pair("first_unique", lambda xs: next((x for x in xs if Counter(xs)[x] == 1), None),
               [[[]], [[5, 2, 5, 3]], [[0, 0, -1]]])


def test_max_window_sum_pair():
    def oracle(xs, k):
        if k <= 0 or k > len(xs):
            return None
        return max(sum(xs[j] for j in range(i, i + k)) for i in range(len(xs) - k + 1))
    check_pair("max_window_sum", oracle, [[[-3, -2, -8], 2], [[1, 2], 2], [[], 0]])


def test_subarray_sum_count_pair():
    from itertools import accumulate
    def oracle(xs, target):
        prefix = [0] + list(accumulate(xs))
        return sum(prefix[r] - prefix[l] == target
                   for l in range(len(xs)) for r in range(l + 1, len(xs) + 1))
    check_pair("subarray_sum_count", oracle, [[[0, 0], 0], [[-2, 1, -2], -3]])


def test_longest_unique_span_pair():
    def oracle(xs):
        best = 0
        for left in range(len(xs)):
            window = []
            for x in xs[left:]:
                if x in window:
                    break
                window.append(x)
            best = max(best, len(window))
        return best
    check_pair("longest_unique_span", oracle,
               [[[1, 2, 2, 1]], [[1, 2, 3, 1, 4]], [[]]])


def test_random_hidden_battery_includes_large_diverse_benchmark_neighbours():
    problem, _ = pair("has_duplicate")
    random_only = problem.hidden_inputs(2026, 64)[len(problem.edge_inputs):]
    large = [args[0] for args in random_only if len(args[0]) >= 300]
    assert large, "Random hidden cases must include benchmark-sized inputs"
    assert {len(xs) for xs in large} >= {799, 800, 801}
    assert any(len(set(xs)) == len(xs) for xs in large)
    assert any(len(set(xs)) < len(xs) for xs in large)


def test_all_pairs_have_fixed_adversarial_benchmark_neighbours():
    from harness.problems import SPIKE_PAIRS
    assert len(SPIKE_PAIRS) == 12
    assert len({problem.id for problem, _ in SPIKE_PAIRS}) == 12
    for problem, _ in SPIKE_PAIRS:
        sizes = {len(args[0]) for args in problem.edge_inputs}
        assert sizes >= {799, 800, 801}, problem.id


def test_source_hash_and_case_ids_are_stable_content_addressed_metadata():
    import hashlib
    from dataclasses import replace
    problem, _ = pair("has_duplicate")
    assert problem.source_sha256 == hashlib.sha256(problem.reference.encode()).hexdigest()
    assert replace(problem, reference=problem.reference + "\n").source_sha256 != problem.source_sha256
    args = [[1, 2, 3]]
    assert problem.case_id(args) == problem.case_id(copy.deepcopy(args))
    assert problem.case_id(args) != problem.case_id([[1, 2, 4]])
    assert problem.case_id(args).startswith(problem.id + ":")
