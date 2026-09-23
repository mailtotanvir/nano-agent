"""Trusted-parent spike fixtures, NOT a frozen training/evaluation corpus.

Never pass this module, the reference, generators, or seeds to a candidate.
The candidate worker receives only its own source, function name, and inputs.
Integer-list domains keep all inputs JSON-compatible. Finite generated tests
provide evidence of correctness, not a proof of semantic equivalence.
"""
import hashlib
import json
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Problem:
    id: str
    family: str
    reference: str
    function_name: str
    benchmark_inputs: list[list]
    edge_inputs: list[list] = field(repr=False)
    _generator: Callable[[random.Random], list] = field(repr=False, compare=False)

    @property
    def source_sha256(self) -> str:
        """Exact source hash; this metadata does not freeze a corpus or split."""
        return hashlib.sha256(self.reference.encode("utf-8")).hexdigest()

    def case_id(self, args: list) -> str:
        """Stable private content ID for a JSON positional-input case."""
        payload = json.dumps(args, allow_nan=False, sort_keys=True, separators=(",", ":"))
        return self.id + ":" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def hidden_inputs(self, seed: int, n: int) -> list[list]:
        """Return independent copies of every edge case plus n seeded cases."""
        rng = random.Random(seed)
        return deepcopy(self.edge_inputs) + [self._generator(rng) for _ in range(n)]


def _integer_list(rng):
    # Most cases stay small; a bounded minority probe benchmark-sized inputs.
    size = rng.choice([rng.randint(0, 48)] * 12 + [799, 800, 801])
    if rng.randrange(3) == 0:
        values = rng.sample(range(-size * 5 - 1, size * 5 + 1), size)
        if size > 1 and rng.randrange(2):
            values[-1] = values[0]
        return values
    return [rng.randint(-20, 20) for _ in range(size)]


def _unary(rng):
    return [_integer_list(rng)]


SPIKE_PAIRS: list[tuple[Problem, str]] = [
    (Problem(
        id="has_duplicate", family="hash-membership", function_name="solve",
        reference='''def solve(xs):
    for i in range(len(xs)):
        for j in range(i):
            if xs[i] == xs[j]:
                return True
    return False
''',
        benchmark_inputs=[[list(range(800))]],
        edge_inputs=[[[]], [[0]], [[1, 1]], [[-1, 0, -1]], [[1, 2, 3]]]
                    + [[list(range(n))] for n in (799, 800, 801)]
                    + [[list(range(n - 1)) + [0]] for n in (799, 800, 801)],
        _generator=_unary,
    ), '''def solve(xs):
    return len(set(xs)) != len(xs)
'''),
]


SPIKE_PAIRS.append((Problem(
    id='count_distinct', family='distinct-cardinality', function_name="solve",
    reference='''def solve(xs):
    unique = []
    for x in xs:
        if x not in unique:
            unique.append(x)
    return len(unique)
''',
    benchmark_inputs=[[list(range(800))]],
    edge_inputs=[[[]], [[0]], [[2, 2, 1, 2]], [[-4, -3, 0]]],
    _generator=_unary,
), '''def solve(xs):
    return len(set(xs))
'''))


SPIKE_PAIRS.append((Problem(
    id='stable_unique', family='order-preserving-deduplication', function_name="solve",
    reference='''def solve(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out
''',
    benchmark_inputs=[[list(range(800, 0, -1))]],
    edge_inputs=[[[]], [[0]], [[2, 1, 2, 3, 1]], [[-1, -1, 0]]],
    _generator=_unary,
), '''def solve(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
'''))


SPIKE_PAIRS.append((Problem(
    id='frequency_counts', family='frequency-index', function_name="solve",
    reference='''def solve(xs):
    return [xs.count(x) for x in xs]
''',
    benchmark_inputs=[[list(range(800))]],
    edge_inputs=[[[]], [[0]], [[3, 3, 2, 3]], [[-1, 0, -1]]],
    _generator=_unary,
), '''def solve(xs):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return [counts[x] for x in xs]
'''))

def _two_lists(rng):
    return [_integer_list(rng), _integer_list(rng)]

SPIKE_PAIRS.append((Problem(
    id='ordered_intersection', family='indexed-intersection', function_name="solve",
    reference='''def solve(xs, ys):
    out = []
    for x in xs:
        if x in ys and x not in out:
            out.append(x)
    return out
''',
    benchmark_inputs=[[list(range(800)), list(range(400, 1200))]],
    edge_inputs=[[[], []], [[1], []], [[], [1]], [[3, 1, 3, 2], [2, 3, 3]]],
    _generator=_two_lists,
), '''def solve(xs, ys):
    remaining = set(ys)
    out = []
    for x in xs:
        if x in remaining:
            remaining.remove(x)
            out.append(x)
    return out
'''))

def _with_target(rng):
    return [_integer_list(rng), rng.randint(-40, 40)]

SPIKE_PAIRS.append((Problem(
    id='pair_sum', family='complement-lookup', function_name="solve",
    reference='''def solve(xs, target):
    for i in range(len(xs)):
        for j in range(i):
            if xs[i] + xs[j] == target:
                return True
    return False
''',
    benchmark_inputs=[[list(range(800)), -1]],
    edge_inputs=[[[], 0], [[2], 4], [[2, 2], 4], [[-2, 5, 0], 3], [[0, 0], 0]],
    _generator=_with_target,
), '''def solve(xs, target):
    seen = set()
    for x in xs:
        if target - x in seen:
            return True
        seen.add(x)
    return False
'''))


SPIKE_PAIRS.append((Problem(
    id='prefix_sums', family='running-aggregate', function_name="solve",
    reference='''def solve(xs):
    return [sum(xs[:i + 1]) for i in range(len(xs))]
''',
    benchmark_inputs=[[list(range(800))]],
    edge_inputs=[[[]], [[0]], [[1, -1, 2]], [[-4, -3]]],
    _generator=_unary,
), '''def solve(xs):
    total = 0
    out = []
    for x in xs:
        total += x
        out.append(total)
    return out
'''))

def _ranges(rng):
    xs = _integer_list(rng)
    queries = [sorted([rng.randint(0, len(xs)), rng.randint(0, len(xs))])
               for _ in range(rng.randint(0, 48))]
    return [xs, queries]

SPIKE_PAIRS.append((Problem(
    id='range_sums', family='prefix-range-queries', function_name="solve",
    reference='''def solve(xs, queries):
    return [sum(xs[left:right]) for left, right in queries]
''',
    benchmark_inputs=[[list(range(800)), [[i % 200, 800 - i % 200] for i in range(800)]]],
    edge_inputs=[[[], [[0, 0]]], [[4], [[0, 0], [0, 1], [1, 1]]], [[-1, 3, -2], [[0, 3], [1, 2], [1, 1]]]],
    _generator=_ranges,
), '''def solve(xs, queries):
    prefix = [0]
    total = 0
    for x in xs:
        total += x
        prefix.append(total)
    return [prefix[right] - prefix[left] for left, right in queries]
'''))


SPIKE_PAIRS.append((Problem(
    id='first_unique', family='unique-frequency-selection', function_name="solve",
    reference='''def solve(xs):
    for x in xs:
        if xs.count(x) == 1:
            return x
    return None
''',
    benchmark_inputs=[[list(range(400)) + list(range(400)) + [999]]],
    edge_inputs=[[[]], [[0]], [[1, 1]], [[2, 1, 2, 3]], [[-1, -1, 0]]],
    _generator=_unary,
), '''def solve(xs):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    for x in xs:
        if counts[x] == 1:
            return x
    return None
'''))

def _window(rng):
    xs = _integer_list(rng)
    return [xs, rng.randint(-1, len(xs) + 2)]

SPIKE_PAIRS.append((Problem(
    id='max_window_sum', family='fixed-sliding-window', function_name="solve",
    reference='''def solve(xs, k):
    if k <= 0 or k > len(xs):
        return None
    return max(sum(xs[i:i + k]) for i in range(len(xs) - k + 1))
''',
    benchmark_inputs=[[list(range(800)), 400]],
    edge_inputs=[[[], 1], [[1], 0], [[1], -1], [[1], 2], [[1], 1], [[-4, -2, -3], 2]],
    _generator=_window,
), '''def solve(xs, k):
    if k <= 0 or k > len(xs):
        return None
    total = sum(xs[:k])
    best = total
    for i in range(k, len(xs)):
        total += xs[i] - xs[i - k]
        best = max(best, total)
    return best
'''))


SPIKE_PAIRS.append((Problem(
    id='subarray_sum_count', family='prefix-sum-multiplicity', function_name="solve",
    reference='''def solve(xs, target):
    count = 0
    for left in range(len(xs)):
        total = 0
        for right in range(left, len(xs)):
            total += xs[right]
            if total == target:
                count += 1
    return count
''',
    benchmark_inputs=[[[i % 7 - 3 for i in range(800)], 0]],
    edge_inputs=[[[], 0], [[0], 0], [[0, 0, 0], 0], [[1, -1, 1], 1], [[-1, -2], -3]],
    _generator=_with_target,
), '''def solve(xs, target):
    counts = {0: 1}
    total = 0
    answer = 0
    for x in xs:
        total += x
        answer += counts.get(total - target, 0)
        counts[total] = counts.get(total, 0) + 1
    return answer
'''))


SPIKE_PAIRS.append((Problem(
    id='longest_unique_span', family='variable-sliding-window', function_name="solve",
    reference='''def solve(xs):
    best = 0
    for left in range(len(xs)):
        seen = set()
        for right in range(left, len(xs)):
            if xs[right] in seen:
                break
            seen.add(xs[right])
            best = max(best, right - left + 1)
    return best
''',
    benchmark_inputs=[[list(range(800))]],
    edge_inputs=[[[]], [[0]], [[1, 1, 1]], [[1, 2, 3, 2, 4]], [[1, 2, 2, 1]]],
    _generator=_unary,
), '''def solve(xs):
    last = {}
    left = 0
    best = 0
    for right, x in enumerate(xs):
        left = max(left, last.get(x, -1) + 1)
        last[x] = right
        best = max(best, right - left + 1)
    return best
'''))


def _large_edges(problem_id):
    cases = []
    for n in (799, 800, 801):
        xs = list(range(n)) if n != 800 else list(range(n - 1)) + [0]
        if problem_id == "ordered_intersection":
            cases.append([xs, list(range(n // 2, n + 20))])
        elif problem_id == "pair_sum":
            if n == 800:
                cases.append([[-i - 1 for i in range(n)], -2 * n + 1])
            else:
                cases.append([xs, 0 if n == 801 else -1])
        elif problem_id == "range_sums":
            cases.append([xs, [[0, n], [0, 0], [n, n], [n // 2, n]]])
        elif problem_id == "max_window_sum":
            cases.append([[-x for x in xs], n // 2 if n == 800 else n])
        elif problem_id == "subarray_sum_count":
            cases.append([[0] * n if n == 800 else [i % 3 - 1 for i in range(n)],
                          0 if n == 800 else -1])
        elif problem_id == "first_unique":
            cases.append([list(range(n // 2)) * 2 + ([n] if n % 2 else [])])
        else:
            cases.append([xs])
    return cases


# These are private correctness probes, not extra performance benchmarks.
# In particular, equal dimensions must include different correct answers.
for _problem, _fast in SPIKE_PAIRS:
    if _problem.id != "has_duplicate":
        _problem.edge_inputs.extend(_large_edges(_problem.id))
