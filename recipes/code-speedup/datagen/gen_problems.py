#!/usr/bin/env python3
"""Build a deterministic, verifier-gated corpus of pure-Python speedup tasks.

The generated JSONL contains only the slow reference and trusted evaluation
metadata.  The known fast implementation exists only in the in-memory
``GeneratedPair`` used while verifying a row; it is deliberately omitted from
all public artifacts and model prompts.

``--verify`` is intentionally expensive: every row is passed through the same
correctness gate and Cachegrind reward path used by the training harness.  It is
safe to generate an *unverified staging corpus* without that switch, but such a
file must never be used for SFT, GRPO, or frozen evaluation.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import random
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from harness.problems import Problem

SCHEMA_VERSION = 1
DEFAULT_SEED = 20_260_907
SEEN_FAMILIES = (
    "hash-membership", "distinct-cardinality", "stable-deduplication",
    "frequency-index", "complement-lookup", "running-aggregate",
    "prefix-range-query", "fixed-sliding-window",
)
HELDOUT_FAMILIES = ("string-concatenation", "sort-selection", "indexed-lookup")


@dataclass(frozen=True)
class GeneratedPair:
    """Trusted construction object.  ``fast_source`` is never serialized."""

    record: dict
    fast_source: str


class _AlphaRename(ast.NodeTransformer):
    """Canonicalize local spelling while retaining literals and operation shape."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def _name(self, name: str) -> str:
        if name == "solve" or name in {"len", "set", "range", "sum", "max", "min", "sorted",
                                        "enumerate", "list", "str", "dict", "any", "all"}:
            return name
        if name not in self.names:
            self.names[name] = f"v{len(self.names)}"
        return self.names[name]

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._name(node.arg)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._name(node.id)
        return node


def normalized_ast_hash(source: str) -> str:
    """Hash parsed structure, ignoring whitespace and local identifier spelling.

    This is a leakage detector, not a semantic-equivalence oracle.  Attribute
    names and literals are retained because changing either can change behavior.
    """
    tree = ast.parse(source)
    tree = _AlphaRename().visit(tree)
    ast.fix_missing_locations(tree)
    payload = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _int_list(rng: random.Random) -> list[int]:
    length = rng.randint(0, 50)
    return [rng.randint(-50, 50) for _ in range(length)]


def _domain_case(kind: str, rng: random.Random) -> list:
    if kind == "int_list":
        return [_int_list(rng)]
    if kind == "two_int_lists":
        return [_int_list(rng), _int_list(rng)]
    if kind == "target":
        return [_int_list(rng), rng.randint(-80, 80)]
    if kind == "ranges":
        xs = _int_list(rng)
        queries = []
        for _ in range(rng.randint(0, 40)):
            left, right = sorted((rng.randint(0, len(xs)), rng.randint(0, len(xs))))
            queries.append([left, right])
        return [xs, queries]
    if kind == "window":
        xs = _int_list(rng)
        return [xs, rng.randint(-1, len(xs) + 2)]
    if kind == "strings":
        alphabet = ("a", "b", "c", "xy", "", "z")
        return [[rng.choice(alphabet) for _ in range(rng.randint(0, 50))]]
    if kind == "lookup":
        items = [[key, key * 3 - 7] for key in rng.sample(range(-120, 121), rng.randint(0, 50))]
        return [items, [rng.randint(-120, 120) for _ in range(rng.randint(0, 50))]]
    raise ValueError(f"unknown generator kind: {kind}")


def _edge_cases(kind: str) -> list[list]:
    if kind == "int_list":
        return [[[]], [[0]], [[1, 1]], [[-2, 1, -2, 3]], [[3, 2, 1]]]
    if kind == "two_int_lists":
        return [[[], []], [[1], []], [[], [1]], [[3, 1, 3], [2, 3]]]
    if kind == "target":
        return [[[], 0], [[2], 4], [[2, 2], 4], [[-2, 5, 0], 3]]
    if kind == "ranges":
        return [[[], [[0, 0]]], [[4], [[0, 0], [0, 1]]], [[-1, 3, -2], [[0, 3], [1, 2]]]]
    if kind == "window":
        return [[[], 1], [[1], 0], [[1], 1], [[1, -3, 2], 2]]
    if kind == "strings":
        return [[[]], [[""]], [["a", "b", "c"]], [["x", "", "y"]]]
    if kind == "lookup":
        return [[[], []], [[[1, 4]], []], [[[1, 4]], [1, 2]], [[[2, 3], [2, 4]], [2]]]
    raise ValueError(kind)


def _benchmark_variants(kind: str, salt: int, family: str) -> list[dict]:
    """Return five same-domain probes, including size and value neighbours."""
    def labelled(ident: str, inputs: list) -> dict:
        return {"id": ident, "inputs": inputs}

    if family == "sort-selection":
        # ``sorted(range(n))`` and ``sorted(reversed(range(n)))`` are nearly
        # linear for Timsort, while process startup dominates at n=800. These
        # disordered 20k neighbours exercise the actual O(n log n) selection
        # anti-pattern against O(n) min/max without relying on wall-clock time.
        n = 20_000

        def permutation(size: int, multiplier: int) -> list[int]:
            # 37, 53 and 71 are coprime to each neighbouring benchmark size.
            return [(index * multiplier) % size for index in range(size)]

        return [
            labelled("n_minus_1", [permutation(n - 1, 37)]),
            labelled("n", [permutation(n, 37)]),
            labelled("n_plus_1", [permutation(n + 1, 37)]),
            labelled("values_changed", [[(index * 53 + salt) % (n * 2)
                                          for index in range(n)]]),
            labelled("order_changed", [permutation(n, 71)]),
        ]

    n = 800
    if kind == "int_list":
        base = list(range(n))
        return [
            labelled("n_minus_1", [list(range(n - 1))]),
            labelled("n", [base]),
            labelled("n_plus_1", [list(range(n + 1))]),
            labelled("values_changed", [[(i * 17 + salt) % 1901 - 950 for i in range(n)]]),
            labelled("order_changed", [list(reversed(base))]),
        ]
    if kind == "two_int_lists":
        return [
            labelled("n_minus_1", [list(range(n - 1)), list(range(n // 2, n + 17))]),
            labelled("n", [list(range(n)), list(range(n // 2, n + 17))]),
            labelled("n_plus_1", [list(range(n + 1)), list(range(n // 2, n + 17))]),
            labelled("values_changed", [[i * 2 + salt for i in range(n)], [i * 2 + salt for i in range(n // 2, n + 17)]]),
            labelled("order_changed", [list(reversed(range(n))), list(reversed(range(n // 2, n + 17)))]),
        ]
    if kind == "target":
        return [
            labelled("n_minus_1", [list(range(n - 1)), -1]),
            labelled("n", [list(range(n)), -1]),
            labelled("n_plus_1", [list(range(n + 1)), -1]),
            labelled("values_changed", [[i * 3 + salt for i in range(n)], salt * 2 + 3]),
            labelled("order_changed", [list(reversed(range(n))), -1]),
        ]
    if kind == "ranges":
        def case(size: int, values: list[int]) -> list:
            return [values, [[i % 200, size - i % 200] for i in range(size)]]
        return [
            labelled("n_minus_1", case(n - 1, list(range(n - 1)))),
            labelled("n", case(n, list(range(n)))),
            labelled("n_plus_1", case(n + 1, list(range(n + 1)))),
            labelled("values_changed", case(n, [i * 13 + salt for i in range(n)])),
            labelled("order_changed", case(n, list(reversed(range(n))))),
        ]
    if kind == "window":
        return [
            labelled("n_minus_1", [list(range(n - 1)), (n - 1) // 2]),
            labelled("n", [list(range(n)), n // 2]),
            labelled("n_plus_1", [list(range(n + 1)), (n + 1) // 2]),
            labelled("values_changed", [[(i * 7 + salt) % 401 - 200 for i in range(n)], n // 2]),
            labelled("order_changed", [list(reversed(range(n))), n // 2]),
        ]
    if kind == "strings":
        return [
            labelled("n_minus_1", [[str(i % 10) for i in range(n - 1)]]),
            labelled("n", [[str(i % 10) for i in range(n)]]),
            labelled("n_plus_1", [[str(i % 10) for i in range(n + 1)]]),
            labelled("values_changed", [[chr(97 + (i + salt) % 26) * (1 + i % 3) for i in range(n)]]),
            labelled("order_changed", [[str(i % 10) for i in reversed(range(n))]]),
        ]
    if kind == "lookup":
        def case(size: int, reverse: bool = False, changed: bool = False) -> list:
            items = [[i, (i * 11 + salt) if changed else i * 3] for i in range(size)]
            if reverse:
                items.reverse()
            keys = list(range(size))
            if reverse:
                keys.reverse()
            return [items, keys]
        return [
            labelled("n_minus_1", case(n - 1)), labelled("n", case(n)),
            labelled("n_plus_1", case(n + 1)), labelled("values_changed", case(n, changed=True)),
            labelled("order_changed", case(n, reverse=True)),
        ]
    raise ValueError(kind)


def _semantic_prelude(family: str, domain: str, value: int, structural_variant: int) -> str:
    """A task-defining transform applied identically to slow and fast programs.

    Unlike a cosmetic nonce, this changes at least one observable result in the
    supported domain.  It gives each row a meaningful source/input contract while
    keeping the optimization family and source template fixed.
    """
    equality_families = {
        "hash-membership", "distinct-cardinality", "stable-deduplication",
        "frequency-index", "complement-lookup",
    }
    if family in equality_families:
        # The corpus draws unique, bounded values, so this literal changes the
        # key space rather than acting as a cosmetic nonce.
        modulus = 2 + value
        if structural_variant:
            return f"    xs = [({modulus} - x) % {modulus} for x in xs]\n"
        return f"    xs = [x % {modulus} for x in xs]\n"
    if domain == "int_list":
        if structural_variant:
            return f"    xs = [{value} - x for x in xs]\n"
        return f"    xs = [x + {value} for x in xs]\n"
    if domain == "target":
        if structural_variant:
            return f"    xs = [{value} - x for x in xs]\n"
        return f"    xs = [x + {value} for x in xs]\n"
    if domain == "ranges":
        if structural_variant:
            return f"    xs = [{value} - x for x in xs]\n"
        return f"    xs = [x + {value} for x in xs]\n"
    if domain == "window":
        if structural_variant:
            return f"    xs = [{value} - x for x in xs]\n"
        return f"    xs = [x + {value} for x in xs]\n"
    if domain == "strings":
        return f"    xs = [x + str({value}) for x in xs]\n"
    if domain == "lookup":
        return f"    items = [[item[0], item[1] + {value}] for item in items]\n"
    raise ValueError(domain)


def _with_prelude(source: str, family: str, domain: str, value: int, structural_variant: int) -> str:
    header, newline, body = source.partition("\n")
    assert newline and header.startswith("def solve(")
    return header + "\n" + _semantic_prelude(family, domain, value, structural_variant) + body


def _finish_pair(slow: str, fast: str, domain: str, value: int, pattern: int,
                 family: str) -> tuple[str, str, str]:
    transform = 1 if pattern >= 5 else 0
    return (_with_prelude(slow, family, domain, value, transform),
            _with_prelude(fast, family, domain, value, transform), domain)


def _source_pair(family: str, pattern: int, semantic_value: int) -> tuple[str, str, str]:
    """Return slow source, fast source and input generator kind for one template.

    ``semantic_value`` is applied to the actual input/output contract by a
    domain transform. Templates, rather than instance values, define split
    boundaries.
    """
    if family == "hash-membership":
        bodies = [
            "    for i in range(len(xs)):\n        for j in range(i):\n            if xs[i] == xs[j]:\n                return True\n    return False\n",
            "    for i, x in enumerate(xs):\n        for y in xs[:i]:\n            if x == y:\n                return True\n    return False\n",
            "    for i in range(len(xs)):\n        for j in range(i + 1, len(xs)):\n            if xs[i] == xs[j]:\n                return True\n    return False\n",
            "    for x in xs:\n        if xs.count(x) > 1:\n            return True\n    return False\n",
            "    seen = []\n    for x in xs:\n        if x in seen:\n            return True\n        seen.append(x)\n    return False\n",
        ]
        return _finish_pair("def solve(xs):\n" + bodies[pattern % 5], "def solve(xs):\n    return len(set(xs)) != len(xs)\n", "int_list", semantic_value, pattern, family)
    if family == "distinct-cardinality":
        bodies = [
            "    unique = []\n    for x in xs:\n        if x not in unique:\n            unique.append(x)\n    return len(unique)\n",
            "    count = 0\n    for i in range(len(xs)):\n        if xs[i] not in xs[:i]:\n            count += 1\n    return count\n",
            "    out = []\n    for x in xs:\n        found = False\n        for y in out:\n            if x == y:\n                found = True\n        if not found:\n            out.append(x)\n    return len(out)\n",
            "    return sum(1 for i in range(len(xs)) if xs.index(xs[i]) == i)\n",
            "    result = []\n    for x in xs:\n        if result.count(x) == 0:\n            result.append(x)\n    return len(result)\n",
        ]
        return _finish_pair("def solve(xs):\n" + bodies[pattern % 5], "def solve(xs):\n    return len(set(xs))\n", "int_list", semantic_value, pattern, family)
    if family == "stable-deduplication":
        bodies = [
            "    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n",
            "    out = []\n    for i in range(len(xs)):\n        if xs[i] not in xs[:i]:\n            out.append(xs[i])\n    return out\n",
            "    out = []\n    for x in xs:\n        present = False\n        for y in out:\n            if x == y:\n                present = True\n        if not present:\n            out.append(x)\n    return out\n",
            "    out = []\n    for x in xs:\n        if out.count(x) == 0:\n            out.append(x)\n    return out\n",
            "    return [xs[i] for i in range(len(xs)) if xs[i] not in xs[:i]]\n",
        ]
        fast = "def solve(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n"
        return _finish_pair("def solve(xs):\n" + bodies[pattern % 5], fast, "int_list", semantic_value, pattern, family)
    if family == "frequency-index":
        bodies = [
            "    return [xs.count(x) for x in xs]\n",
            "    out = []\n    for x in xs:\n        count = 0\n        for y in xs:\n            if x == y:\n                count += 1\n        out.append(count)\n    return out\n",
            "    return [sum(1 for y in xs if y == x) for x in xs]\n",
            "    out = []\n    for i in range(len(xs)):\n        count = 0\n        for j in range(len(xs)):\n            if xs[i] == xs[j]:\n                count += 1\n        out.append(count)\n    return out\n",
            "    out = []\n    for x in xs:\n        count = 0\n        for y in reversed(xs):\n            if x == y:\n                count += 1\n        out.append(count)\n    return out\n",
        ]
        fast = "def solve(xs):\n    counts = {}\n    for x in xs:\n        counts[x] = counts.get(x, 0) + 1\n    return [counts[x] for x in xs]\n"
        return _finish_pair("def solve(xs):\n" + bodies[pattern % 5], fast, "int_list", semantic_value, pattern, family)
    if family == "complement-lookup":
        bodies = [
            "    for i in range(len(xs)):\n        for j in range(i):\n            if xs[i] + xs[j] == target:\n                return True\n    return False\n",
            "    for i, x in enumerate(xs):\n        for y in xs[:i]:\n            if x + y == target:\n                return True\n    return False\n",
            "    seen = []\n    for x in xs:\n        if target - x in seen:\n            return True\n        seen.append(x)\n    return False\n",
            "    for i in range(len(xs)):\n        for j in range(i + 1, len(xs)):\n            if xs[i] + xs[j] == target:\n                return True\n    return False\n",
            "    for i, x in enumerate(xs):\n        for y in xs[i + 1:]:\n            if x + y == target:\n                return True\n    return False\n",
        ]
        fast = "def solve(xs, target):\n    seen = set()\n    for x in xs:\n        if target - x in seen:\n            return True\n        seen.add(x)\n    return False\n"
        return _finish_pair("def solve(xs, target):\n" + bodies[pattern % 5], fast, "target", semantic_value, pattern, family)
    if family == "running-aggregate":
        bodies = [
            "    return [sum(xs[:i + 1]) for i in range(len(xs))]\n",
            "    out = []\n    for i in range(len(xs)):\n        total = 0\n        for j in range(i + 1):\n            total += xs[j]\n        out.append(total)\n    return out\n",
            "    out = []\n    for i, x in enumerate(xs):\n        out.append(sum(xs[:i + 1]))\n    return out\n",
            "    return [sum(xs[0:i + 1]) for i in range(len(xs))]\n",
            "    out = []\n    for i in range(len(xs)):\n        out.append(sum(xs[:i] + [xs[i]]))\n    return out\n",
        ]
        fast = "def solve(xs):\n    total = 0\n    out = []\n    for x in xs:\n        total += x\n        out.append(total)\n    return out\n"
        return _finish_pair("def solve(xs):\n" + bodies[pattern % 5], fast, "int_list", semantic_value, pattern, family)
    if family == "prefix-range-query":
        bodies = [
            "    return [sum(xs[left:right]) for left, right in queries]\n",
            "    out = []\n    for left, right in queries:\n        total = 0\n        for i in range(left, right):\n            total += xs[i]\n        out.append(total)\n    return out\n",
            "    return [sum(xs[i] for i in range(left, right)) for left, right in queries]\n",
            "    out = []\n    for query in queries:\n        left = query[0]\n        right = query[1]\n        out.append(sum(xs[left:right]))\n    return out\n",
            "    out = []\n    for left, right in queries:\n        values = []\n        for i in range(left, right):\n            values.append(xs[i])\n        out.append(sum(values))\n    return out\n",
        ]
        fast = "def solve(xs, queries):\n    prefix = [0]\n    total = 0\n    for x in xs:\n        total += x\n        prefix.append(total)\n    return [prefix[right] - prefix[left] for left, right in queries]\n"
        return _finish_pair("def solve(xs, queries):\n" + bodies[pattern % 5], fast, "ranges", semantic_value, pattern, family)
    if family == "fixed-sliding-window":
        prefix = "    if k <= 0 or k > len(xs):\n        return None\n"
        bodies = [
            prefix + "    return max(sum(xs[i:i + k]) for i in range(len(xs) - k + 1))\n",
            prefix + "    best = None\n    for i in range(len(xs) - k + 1):\n        total = 0\n        for j in range(i, i + k):\n            total += xs[j]\n        if best is None or total > best:\n            best = total\n    return best\n",
            prefix + "    values = []\n    for i in range(len(xs) - k + 1):\n        values.append(sum(xs[i:i + k]))\n    return max(values)\n",
            prefix + "    best = sum(xs[:k])\n    for i in range(1, len(xs) - k + 1):\n        current = sum(xs[i:i + k])\n        if current > best:\n            best = current\n    return best\n",
            prefix + "    return max(sum(xs[j] for j in range(i, i + k)) for i in range(len(xs) - k + 1))\n",
        ]
        fast = "def solve(xs, k):\n    if k <= 0 or k > len(xs):\n        return None\n    total = sum(xs[:k])\n    best = total\n    for i in range(k, len(xs)):\n        total += xs[i] - xs[i - k]\n        if total > best:\n            best = total\n    return best\n"
        return _finish_pair("def solve(xs, k):\n" + bodies[pattern % 5], fast, "window", semantic_value, pattern, family)
    if family == "string-concatenation":
        bodies = [
            "    out = ''\n    for x in xs:\n        out = out + x\n    return out\n",
            "    out = ''\n    for x in xs:\n        out += x\n    return out\n",
            "    out = ''\n    for i in range(len(xs)):\n        out = out + xs[i]\n    return out\n",
            "    pieces = ''\n    for x in reversed(xs):\n        pieces = x + pieces\n    return pieces\n",
            "    out = ''\n    for x in xs:\n        out = '{}{}'.format(out, x)\n    return out\n",
        ]
        # ``format`` is forbidden by the restricted worker; template 4 remains
        # syntactically valid but is intentionally excluded below by remapping.
        if pattern % 5 == 4:
            bodies[4] = "    out = ''\n    for x in xs:\n        out = out + ''.join([x])\n    return out\n"
        return _finish_pair("def solve(xs):\n" + bodies[pattern % 5], "def solve(xs):\n    return ''.join(xs)\n", "strings", semantic_value, pattern, family)
    if family == "sort-selection":
        direction = pattern % 2
        pick = "0" if direction == 0 else "-1"
        builtin = "min" if direction == 0 else "max"
        bodies = [
            f"    if not xs:\n        return None\n    return sorted(xs)[{pick}] + {semantic_value}\n",
            f"    values = sorted(xs)\n    if not values:\n        return None\n    return values[{pick}] + {semantic_value}\n",
            f"    if len(xs) == 0:\n        return None\n    ordered = list(xs)\n    ordered.sort()\n    return ordered[{pick}] + {semantic_value}\n",
            f"    if not xs:\n        return None\n    return list(sorted(xs))[{pick}] + {semantic_value}\n",
            f"    values = []\n    for x in xs:\n        values.append(x)\n    if not values:\n        return None\n    values.sort()\n    return values[{pick}] + {semantic_value}\n",
        ]
        # Applying the semantic literal after selection avoids a shared O(n)
        # list transform that can erase the Cachegrind gap between ``sorted``
        # and ``min``/``max``.
        fast = f"def solve(xs):\n    if not xs:\n        return None\n    return {builtin}(xs) + {semantic_value}\n"
        return "def solve(xs):\n" + bodies[pattern % 5], fast, "int_list"
    if family == "indexed-lookup":
        bodies = [
            "    out = []\n    for key in keys:\n        value = None\n        for item in items:\n            if item[0] == key:\n                value = item[1]\n                break\n        out.append(value)\n    return out\n",
            "    out = []\n    for key in keys:\n        found = None\n        for i in range(len(items)):\n            if items[i][0] == key:\n                found = items[i][1]\n                break\n        out.append(found)\n    return out\n",
            "    out = []\n    for key in keys:\n        matches = [item[1] for item in items if item[0] == key]\n        out.append(matches[0] if matches else None)\n    return out\n",
            "    out = []\n    for key in keys:\n        value = None\n        for item in reversed(items):\n            if item[0] == key:\n                value = item[1]\n        out.append(value)\n    return out\n",
            "    out = []\n    for key in keys:\n        value = None\n        for item in items:\n            if key == item[0] and value is None:\n                value = item[1]\n        out.append(value)\n    return out\n",
        ]
        # The reversed scan is only equivalent where keys are unique, which is
        # part of this template's documented input domain.
        fast = "def solve(items, keys):\n    index = {}\n    for item in items:\n        if item[0] not in index:\n            index[item[0]] = item[1]\n    return [index.get(key) for key in keys]\n"
        return _finish_pair("def solve(items, keys):\n" + bodies[pattern % 5], fast, "lookup", semantic_value, pattern, family)
    raise ValueError(f"unknown family: {family}")


def _template_specs() -> list[tuple[str, str, int, str]]:
    """List real source structures; instances never mint cosmetic templates."""
    specs = []
    for family_index, family in enumerate(SEEN_FAMILIES):
        eval_patterns = range(4) if family_index < 6 else range(3)
        for pattern in range(9):
            hint = "seen_eval" if pattern in eval_patterns else "train_dev"
            specs.append((family, f"{family}:template-{pattern:02d}", pattern, hint))
    for family in HELDOUT_FAMILIES:
        # Five structural templates per excluded family, with two independently
        # semantic task instances per structure.  This track does not claim
        # template-disjointness; family exclusion is the experimental boundary.
        for pattern in range(5):
            specs.append((family, f"{family}:template-{pattern:02d}", pattern, "heldout_eval"))
    return specs


def _record(*, ident: str, family: str, template: str, source: str, domain: str,
            semantic_value: int, split_hint: str) -> dict:
    variants = _benchmark_variants(domain, semantic_value, family)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": ident,
        "family_id": family,
        "template_id": template,
        "semantic_variant": semantic_value,
        "split_hint": split_hint,
        "function_name": "solve",
        "reference_source": source,
        "normalized_ast_sha256": normalized_ast_hash(source),
        "input_generator": {"kind": domain, "version": 1},
        "edge_inputs": _edge_cases(domain),
        "benchmark_variants": variants,
        "benchmark_variant_ids": [variant["id"] for variant in variants],
        "private_seed_policy": "draw fresh entropy after candidate receipt; never include a seed in a model prompt",
        "verification": {"status": "unverified", "method": "harness.evaluate"},
    }


def generate_pairs(*, n: int = 250, seed: int = DEFAULT_SEED) -> list[GeneratedPair]:
    """Generate at least the 250 rows needed by the frozen v1 split.

    The base design is 190 train/development rows plus both 30-row frozen tracks.
    Larger requested corpora add only train/development instances, preserving the
    frozen evaluation definition.
    """
    if n < 250:
        raise ValueError("n must be at least 250 to preserve both frozen 30-case tracks")
    rng = random.Random(seed)
    pairs: list[GeneratedPair] = []
    templates = _template_specs()
    train_templates = [item for item in templates if item[3] == "train_dev"]
    eval_templates = [item for item in templates if item[3] != "train_dev"]
    # 42 genuine train/development structures get four instances each, then 22
    # get a fifth: 168 + 22 = 190 source-distinct semantic tasks.
    staged: list[tuple[str, str, int, str, int]] = []
    for template_index, (family, template, pattern, hint) in enumerate(train_templates):
        for instance in range(4 + (template_index < 22)):
            staged.append((family, template, pattern, hint, instance))
    for family, template, pattern, hint in eval_templates:
        instances = 2 if hint == "heldout_eval" else 1
        for instance in range(instances):
            staged.append((family, template, pattern, hint, instance))
    extra = n - len(staged)
    for instance in range(extra):
        family, template, pattern, hint = train_templates[instance % len(train_templates)]
        staged.append((family, template, pattern, hint, 19 + instance // len(train_templates)))
    semantic_values = rng.sample(range(1, 10_001), len(staged))
    for ordinal, (family, template, pattern, hint, instance) in enumerate(staged):
        semantic_value = semantic_values[ordinal]
        source, fast, domain = _source_pair(family, pattern, semantic_value)
        ident = f"cs-v1-{ordinal:04d}-{hashlib.sha256((template + ':' + str(instance)).encode()).hexdigest()[:10]}"
        record = _record(ident=ident, family=family, template=template, source=source,
                         domain=domain, semantic_value=semantic_value, split_hint=hint)
        pairs.append(GeneratedPair(record, fast))
    hashes = [pair.record["normalized_ast_sha256"] for pair in pairs]
    if len(hashes) != len(set(hashes)):
        raise AssertionError("generated normalized-AST collision; refuse ambiguous corpus")
    return pairs


V2_SEED = 20_260_909
# v2 semantic literals live in a disjoint range from v1 (1..10_000) so that a
# v2 slow reference can never share a normalized-AST fingerprint with any v1
# instance even if two structures happen to coincide.
V2_VALUE_RANGE = (10_001, 40_000)
# v6 starts a new, source-disjoint training run.  Its literal range is disjoint
# from both v1 and v2, so normalized-AST fingerprints cannot collide merely
# because a slow structure is reused as curriculum context.
V6_SEED = 20_260_921
V6_VALUE_RANGE = (40_001, 70_000)


def _v2_template_specs():
    """Enumerate v2 training templates: (family, template_id, body, distractor).

    Surface diversity comes from (1) eight-to-five structurally distinct slow
    bodies per family and (2) an optional benign distractor snippet, doubling
    the template count. The underlying optimization per family is unchanged.
    """
    from datagen import _v2_bank as bank

    specs = []
    ordered = list(bank.SEEN_FAMILIES) + list(bank.NEW_FAMILIES)
    for family in ordered:
        for body_index, body in enumerate(bank.TRAIN_BODIES[family]):
            for distractor_index, distractor in enumerate(bank.DISTRACTORS):
                template = f"{family}:v2-b{body_index:02d}-d{distractor_index}"
                specs.append((family, template, body, distractor, distractor_index))
    return specs


def _v2_dev_template_specs():
    """Enumerate the fresh dev-behavioral templates (seen families only).

    These bodies are structurally disjoint from every training body and carry
    their own ``v2-dev`` template ids, so no dev template appears in training.
    """
    from datagen import _v2_bank as bank

    specs = []
    for family in bank.SEEN_FAMILIES:
        for body_index, body in enumerate(bank.DEV_BODIES[family]):
            template = f"{family}:v2-dev-b{body_index:02d}"
            specs.append((family, template, body))
    return specs


def _v2_record(*, ident, family, template, slow, domain, value, split_hint):
    from datagen import _v2_bank as bank

    return _record(ident=ident, family=family, template=template, source=slow,
                   domain=bank.DOMAIN[family], semantic_value=value,
                   split_hint=split_hint)


def generate_v2_pairs(*, seen_instances: int = 3, new_instances: int = 3,
                      seed: int = V2_SEED) -> list[GeneratedPair]:
    """Generate the deterministic v2 training-only corpus.

    Every problem is built through the same schema/record path as v1 and carries
    a ``train_v2`` split hint. Fast sources stay in memory only. The number of
    semantic instances per template is configurable so the corpus size lands in
    the planned ~450-600 range; templates, not instances, define diversity.
    """
    from datagen import _v2_bank as bank

    rng = random.Random(seed)
    specs = _v2_template_specs()
    staged = []
    for family, template, body, distractor, _distractor_index in specs:
        instances = seen_instances if family in bank.SEEN_FAMILIES else new_instances
        for instance in range(instances):
            staged.append((family, template, body, distractor, instance))
    values = rng.sample(range(*V2_VALUE_RANGE), len(staged))
    pairs: list[GeneratedPair] = []
    for ordinal, (family, template, body, distractor, instance) in enumerate(staged):
        value = values[ordinal]
        slow = bank.slow_source(family, body, distractor, value)
        fast = bank.fast_source(family, value)
        digest = hashlib.sha256((template + ":" + str(instance)).encode()).hexdigest()[:10]
        ident = f"cs-v2-{ordinal:04d}-{digest}"
        record = _v2_record(ident=ident, family=family, template=template, slow=slow,
                            domain=bank.DOMAIN[family], value=value, split_hint="train_v2")
        pairs.append(GeneratedPair(record, fast))
    _assert_distinct_ast(pairs)
    return pairs


def generate_v6_pairs(*, seen_instances: int = 3, new_instances: int = 3,
                      seed: int = V6_SEED) -> list[GeneratedPair]:
    """Generate the v6 training-only corpus with fresh source fingerprints.

    This is intentionally not a relabelled v5 corpus: every source carries a
    literal from ``V6_VALUE_RANGE`` and a v6-only record id.  It includes the
    transformed residue-count family added after the v5 diagnostic, while the
    frozen v1 evaluation families remain excluded by the bank invariant.
    """
    from datagen import _v2_bank as bank

    rng = random.Random(seed)
    staged = []
    for family, template, body, distractor, _distractor_index in _v2_template_specs():
        instances = seen_instances if family in bank.SEEN_FAMILIES else new_instances
        for instance in range(instances):
            staged.append((family, template, body, distractor, instance))
    values = rng.sample(range(*V6_VALUE_RANGE), len(staged))
    pairs: list[GeneratedPair] = []
    for ordinal, (family, template, body, distractor, instance) in enumerate(staged):
        value = values[ordinal]
        slow = bank.slow_source(family, body, distractor, value)
        fast = bank.fast_source(family, value)
        digest = hashlib.sha256((template + ":" + str(instance)).encode()).hexdigest()[:10]
        ident = f"cs-v6-{ordinal:04d}-{digest}"
        record = _v2_record(ident=ident, family=family, template=f"{template}:v6",
                            slow=slow, domain=bank.DOMAIN[family], value=value,
                            split_hint="train_v6")
        pairs.append(GeneratedPair(record, fast))
    _assert_distinct_ast(pairs)
    return pairs


def generate_v6_heldout_pairs(*, instances: int = 1,
                              seed: int = V6_SEED + 1) -> list[GeneratedPair]:
    """Create the one-time, template-disjoint v6 generalization measurement.

    The split is restricted to fresh ``DEV_BODIES`` of the transformed bucket
    task.  It must not be used for teacher generation or checkpoint selection.
    """
    from datagen import _v2_bank as bank

    family = "residue-count-index"
    rng = random.Random(seed)
    staged = [(body_index, body, instance)
              for body_index, body in enumerate(bank.DEV_BODIES[family])
              for instance in range(instances)]
    values = rng.sample(range(70_001, 90_000), len(staged))
    pairs: list[GeneratedPair] = []
    for ordinal, (body_index, body, instance) in enumerate(staged):
        value = values[ordinal]
        template = f"{family}:v6-heldout-b{body_index:02d}"
        slow = bank.slow_source(family, body, "", value)
        fast = bank.fast_source(family, value)
        digest = hashlib.sha256((template + ":" + str(instance)).encode()).hexdigest()[:10]
        record = _v2_record(ident=f"cs-v6heldout-{ordinal:04d}-{digest}", family=family,
                            template=template, slow=slow, domain=bank.DOMAIN[family],
                            value=value, split_hint="heldout_v6_unobserved")
        pairs.append(GeneratedPair(record, fast))
    _assert_distinct_ast(pairs)
    return pairs


def generate_dev_behavioral_pairs(*, instances: int = 1,
                                  seed: int = V2_SEED + 1) -> list[GeneratedPair]:
    """Generate the fresh behavioral dev split (3 per seen family by default).

    Marked ``dev_behavioral``: this is training infrastructure for checkpoint
    selection, explicitly distinct from the frozen v1 evaluation files. Its
    templates and normalized-AST fingerprints are disjoint from all training
    data (asserted by the leakage checker).
    """
    from datagen import _v2_bank as bank

    rng = random.Random(seed)
    specs = _v2_dev_template_specs()
    staged = []
    for family, template, body in specs:
        for instance in range(instances):
            staged.append((family, template, body, instance))
    values = rng.sample(range(*V2_VALUE_RANGE), len(staged))
    pairs: list[GeneratedPair] = []
    for ordinal, (family, template, body, instance) in enumerate(staged):
        value = values[ordinal]
        slow = bank.slow_source(family, body, "", value)
        fast = bank.fast_source(family, value)
        digest = hashlib.sha256((template + ":" + str(instance)).encode()).hexdigest()[:10]
        ident = f"cs-v2dev-{ordinal:04d}-{digest}"
        record = _v2_record(ident=ident, family=family, template=template, slow=slow,
                            domain=bank.DOMAIN[family], value=value,
                            split_hint="dev_behavioral")
        pairs.append(GeneratedPair(record, fast))
    _assert_distinct_ast(pairs)
    return pairs


def _assert_distinct_ast(pairs: list[GeneratedPair]) -> None:
    hashes = [pair.record["normalized_ast_sha256"] for pair in pairs]
    if len(hashes) != len(set(hashes)):
        raise AssertionError("generated normalized-AST collision; refuse ambiguous corpus")


def problem_from_record(record: dict) -> Problem:
    """Materialize a trusted runtime problem; generator metadata stays out of prompts."""
    kind = record["input_generator"]["kind"]
    return Problem(
        id=record["id"], family=record["family_id"], reference=record["reference_source"],
        function_name=record["function_name"],
        benchmark_inputs=[copy.deepcopy(item["inputs"]) for item in record["benchmark_variants"]],
        edge_inputs=copy.deepcopy(record["edge_inputs"]),
        _generator=lambda rng: _domain_case(kind, rng),
    )


def _result_dict(result) -> dict:
    return {
        "status": "passed" if result.correct and result.status == "ok" and result.reward is not None and result.reward > 0 else "failed",
        "method": "harness.evaluate",
        "harness_status": result.status,
        "correct": bool(result.correct),
        "positive_log_speedup": result.reward,
        "tested": result.tested,
        "reference_counts": list(result.reference_counts),
        "candidate_counts": list(result.candidate_counts),
    }


_HARNESS_FILES = ("correctness.py", "perf.py", "sandbox.py", "worker.py", "problems.py")
_EXECUTION_RECORD_FIELDS = (
    "family_id", "function_name", "reference_source", "input_generator",
    "edge_inputs", "benchmark_variants",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def harness_source_hashes() -> dict[str, str]:
    """Content-address the exact verifier code that made a cached decision."""
    root = Path(__file__).resolve().parent.parent / "harness"
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _HARNESS_FILES}


def verification_config(*, random_cases: int, repeats: int, seed: int) -> dict:
    """Capture every local setting that can alter a verifier decision."""
    return {
        "schema_version": 1,
        "random_cases": random_cases,
        "repeats": repeats,
        "seed": seed,
        "python_executable": sys.executable,
        "valgrind": os.environ.get("VALGRIND", ""),
        "valgrind_lib": os.environ.get("VALGRIND_LIB", ""),
    }


def execution_metadata_hash(record: dict) -> str:
    """Hash every record field consumed by the correctness/performance oracle."""
    missing = set(_EXECUTION_RECORD_FIELDS) - set(record)
    if missing:
        raise ValueError(f"record missing execution fields: {sorted(missing)}")
    payload = {field: record[field] for field in _EXECUTION_RECORD_FIELDS}
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def verification_key(pair: GeneratedPair, config: dict | None = None) -> str:
    """Key a reusable result to sources, settings, and every verifier module."""
    if config is None:
        config = verification_config(random_cases=64, repeats=2, seed=314159)
    payload = {
        "source_sha256": _sha256_text(pair.record["reference_source"]),
        "fast_source_sha256": _sha256_text(pair.fast_source),
        "execution_metadata_sha256": execution_metadata_hash(pair.record),
        "verifier_config": config,
        "harness_source_sha256": harness_source_hashes(),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _load_checkpoint(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    cache = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = row["verification_key"]
            verification = row["verification"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid verification checkpoint at line {line_number}") from exc
        if not isinstance(key, str) or not isinstance(verification, dict):
            raise TypeError(f"invalid verification checkpoint at line {line_number}")
        cache[key] = row
    return cache


def _append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def verify_pairs(pairs: Iterable[GeneratedPair], *, evaluator: Callable | None = None,
                 checkpoint_path: Path | None = None, random_cases: int = 64,
                 repeats: int = 2, seed: int = 314159,
                 progress: Callable[[int, int, str, str], None] | None = None) -> list[GeneratedPair]:
    """Verifier-gate every pair with durable, content-addressed resumption.

    A row is reusable only when slow and fast sources, evaluator configuration,
    and every harness module hash match.  Both passing and failing completed rows
    are checkpointed before any exception, so a late failure is reproducible and
    does not require replaying earlier Cachegrind work.
    """
    if evaluator is None:
        from harness.perf import evaluate as evaluator
    pairs = list(pairs)
    config = verification_config(random_cases=random_cases, repeats=repeats, seed=seed)
    checkpoint = _load_checkpoint(checkpoint_path)
    verified = []
    for index, pair in enumerate(pairs, start=1):
        record = copy.deepcopy(pair.record)
        key = verification_key(pair, config)
        cached = checkpoint.get(key)
        if cached is not None:
            record["verification"] = copy.deepcopy(cached["verification"])
            state = "cached"
        else:
            result = evaluator(problem_from_record(pair.record), pair.fast_source,
                               seed=seed, random_cases=random_cases, repeats=repeats)
            record["verification"] = _result_dict(result)
            checkpoint_row = {
                "schema_version": 1,
                "id": record["id"],
                "verification_key": key,
                "source_sha256": _sha256_text(pair.record["reference_source"]),
                "fast_source_sha256": _sha256_text(pair.fast_source),
                "execution_metadata_sha256": execution_metadata_hash(pair.record),
                "verifier_config": config,
                "harness_source_sha256": harness_source_hashes(),
                "verification": record["verification"],
            }
            if checkpoint_path is not None:
                _append_checkpoint(checkpoint_path, checkpoint_row)
            checkpoint[key] = checkpoint_row
            state = "verified"
        record["verification"]["cache_key"] = key
        if record["verification"]["status"] != "passed":
            if progress is not None:
                progress(index, len(pairs), record["id"],
                         "cached-failed" if cached is not None else "failed")
            raise RuntimeError(f"verification failed for {record['id']}: "
                               f"{record['verification']['harness_status']}")
        if progress is not None:
            progress(index, len(pairs), record["id"], state)
        verified.append(GeneratedPair(record, pair.fast_source))
    return verified


def public_record(pair: GeneratedPair) -> dict:
    """Copy only model-safe/public corpus metadata; guard against future leakage."""
    record = copy.deepcopy(pair.record)
    forbidden = {"fast_source", "known_fast_source", "candidate", "solution"}
    if forbidden & set(record):
        raise AssertionError("fast implementation leaked into a public record")
    return record


_MODEL_PROMPT_FIELDS = ("id", "family_id", "template_id", "function_name", "reference_source")


def model_prompt_record(record: dict) -> dict:
    """Return the only corpus fields that may reach a model completion prompt.

    Exact benchmark inputs, edge cases, generator metadata and verifier outcomes
    are evaluator-private.  This function is the required boundary used by
    future SFT/GRPO prompt construction rather than a convention callers must
    remember.
    """
    missing = set(_MODEL_PROMPT_FIELDS) - set(record)
    if missing:
        raise ValueError(f"record missing prompt fields: {sorted(missing)}")
    return {field: copy.deepcopy(record[field]) for field in _MODEL_PROMPT_FIELDS}


def write_jsonl(path: Path, pairs: Iterable[GeneratedPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(public_record(pair), sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v1", "v2", "v6", "v6-heldout", "dev-behavioral"), default="v1",
                        help="v1: frozen 250-row design; v2: training-only corpus; "
                             "v6: refreshed training-only corpus; v6-heldout: one-time "
                             "template-heldout measurement; dev-behavioral: fresh behavioral dev split")
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seen-instances", type=int, default=3,
                        help="v2: semantic instances per seen-family template")
    parser.add_argument("--new-instances", type=int, default=3,
                        help="v2: semantic instances per new-family template")
    parser.add_argument("--dev-instances", type=int, default=1,
                        help="dev-behavioral: instances per fresh dev template")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="run hidden correctness + Cachegrind on every known-fast pair")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="append-only per-row verifier checkpoint (default: alongside --out)")
    parser.add_argument("--verification-seed", type=int, default=314159,
                        help="deterministic hidden-battery seed for corpus gating")
    args = parser.parse_args()
    if args.variant == "v1":
        pairs = generate_pairs(n=args.n, seed=args.seed if args.seed is not None else DEFAULT_SEED)
    elif args.variant == "v2":
        pairs = generate_v2_pairs(seen_instances=args.seen_instances,
                                  new_instances=args.new_instances,
                                  seed=args.seed if args.seed is not None else V2_SEED)
    elif args.variant == "v6":
        pairs = generate_v6_pairs(seen_instances=args.seen_instances,
                                  new_instances=args.new_instances,
                                  seed=args.seed if args.seed is not None else V6_SEED)
    elif args.variant == "v6-heldout":
        pairs = generate_v6_heldout_pairs(instances=args.dev_instances,
                                          seed=args.seed if args.seed is not None else V6_SEED + 1)
    else:
        pairs = generate_dev_behavioral_pairs(instances=args.dev_instances,
                                              seed=args.seed if args.seed is not None else V2_SEED + 1)
    if args.verify:
        checkpoint = args.checkpoint or args.out.with_suffix(args.out.suffix + ".verify-checkpoint.jsonl")

        def report(index: int, total: int, ident: str, state: str) -> None:
            print(f"[{index}/{total}] {state}: {ident}", flush=True)

        pairs = verify_pairs(pairs, checkpoint_path=checkpoint, seed=args.verification_seed,
                             progress=report)
    write_jsonl(args.out, pairs)
    print(json.dumps({"rows": len(pairs), "verified": args.verify, "output": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
