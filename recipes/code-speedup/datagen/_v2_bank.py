"""Private v2 template bank for the code-speedup training-only corpus.

This module holds the slow reference *body* structures and the known-fast
rewrite for every v2 family, plus the per-family input-transform prelude that
gives each generated instance a distinct source and normalized-AST fingerprint.

Like the v1 ``_source_pair`` data embedded in ``gen_problems.py``, the known-fast
sources here are trusted construction inputs. They are consumed only in-process
by the verifier and are never serialized into a public corpus record or a model
prompt (see ``gen_problems.public_record`` / ``gen_problems.model_prompt_record``).

Design rules honored by every entry:

* Each slow body computes *exactly* the same function as its family fast source
  over the supported integer-list domain (checked in ``test_gen_problems_v2``).
* Each slow body is a genuinely slower kernel (O(n^2) / O(n*m) scans, repeated
  ``sum``/``count``/membership) so Cachegrind sees an instruction-count gap
  against the O(n) fast rewrite.
* Bodies obey the restricted worker AST/builtin policy (no imports, no
  disallowed calls) - this is the *pure-subset* idiom the v1 student failed to
  reach for (13/49 disallowed-import reflex failures).
* Structures are surface-diverse: for / while / comprehension / enumerate /
  reversed / sorted-scan shapes, varied accumulator patterns and names.
"""
from __future__ import annotations

# The three FROZEN held-out families. Nothing in this bank may replicate them.
FROZEN_HELDOUT_FAMILIES = ("string-concatenation", "sort-selection", "indexed-lookup")

# Eight seen families (identical ids to v1) - template diversity track.
SEEN_FAMILIES = (
    "hash-membership", "distinct-cardinality", "stable-deduplication",
    "frequency-index", "complement-lookup", "running-aggregate",
    "prefix-range-query", "fixed-sliding-window",
)

# New training-only families (Phase 2.3). Each targets a diagnosed weakness and
# is structurally distinct from all three frozen held-out families.
NEW_FAMILIES = (
    "position-index",        # build first-occurrence index dict (index-building)
    "count-index",           # build occurrence-count dict (index-building)
    "membership-fusion",     # set-membership fusion replacing nested in-scan
    "ordered-intersection",  # set + seen-guard replacing O(n*m) in/append scan
    "mode-bucket",           # counting-bucket selection replacing repeated count
    "residue-count-index",   # transformed bucket-count queries (v6 repair)
)

# One-line transformation rationale per new family (surfaced in docs + report).
NEW_FAMILY_RATIONALE = {
    "position-index":
        "Precompute a value->first-index dict, then answer each query by O(1) "
        "lookup instead of re-scanning xs per query; directly teaches building "
        "the index the v1 model named but never constructed.",
    "count-index":
        "Precompute a value->count dict in one pass, then answer count queries "
        "by O(1) lookup instead of a full re-count per query; a second, "
        "structurally different index-building transform.",
    "membership-fusion":
        "Hash the second list into a set once and probe it, replacing the "
        "nested `x in ys` linear scan - the pure-subset idiom the v1 model "
        "skipped when it reached for disallowed imports.",
    "ordered-intersection":
        "Fuse a membership set with a seen-guard set to keep first-occurrence "
        "order in one pass, replacing the O(n*m) `x in ys and x not in out` scan.",
    "mode-bucket":
        "Tally counts into a dict once (bucket/counting selection) then pick the "
        "min-value mode, replacing the repeated `xs.count(x)` re-scan.",
    "residue-count-index":
        "Build a transformed residue->count index once, then answer many bucket "
        "queries in O(1); this teaches mapping after a value transform without "
        "reusing the frozen item/key lookup contract.",
}

# Function signature and hidden-input generator/benchmark domain per family.
SIGNATURE = {
    "hash-membership": "xs", "distinct-cardinality": "xs",
    "stable-deduplication": "xs", "frequency-index": "xs",
    "complement-lookup": "xs, target", "running-aggregate": "xs",
    "prefix-range-query": "xs, queries", "fixed-sliding-window": "xs, k",
    "position-index": "xs, queries", "count-index": "xs, queries",
    "membership-fusion": "xs, ys", "ordered-intersection": "xs, ys",
    "mode-bucket": "xs", "residue-count-index": "xs, queries",
}
DOMAIN = {
    "hash-membership": "int_list", "distinct-cardinality": "int_list",
    "stable-deduplication": "int_list", "frequency-index": "int_list",
    "complement-lookup": "target", "running-aggregate": "int_list",
    "prefix-range-query": "ranges", "fixed-sliding-window": "window",
    "position-index": "two_int_lists", "count-index": "two_int_lists",
    "membership-fusion": "two_int_lists", "ordered-intersection": "two_int_lists",
    "mode-bucket": "int_list", "residue-count-index": "two_int_lists",
}

# Known-fast rewrite (body only, after the `def solve(<sig>):` header). Never
# serialized. Header is added by the generator so the prelude can be injected.
FAST_BODY = {
    "hash-membership": "    return len(set(xs)) != len(xs)\n",
    "distinct-cardinality": "    return len(set(xs))\n",
    "stable-deduplication":
        "    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n"
        "            seen.add(x)\n            out.append(x)\n    return out\n",
    "frequency-index":
        "    counts = {}\n    for x in xs:\n        counts[x] = counts.get(x, 0) + 1\n"
        "    return [counts[x] for x in xs]\n",
    "complement-lookup":
        "    seen = set()\n    for x in xs:\n        if target - x in seen:\n"
        "            return True\n        seen.add(x)\n    return False\n",
    "running-aggregate":
        "    total = 0\n    out = []\n    for x in xs:\n        total += x\n"
        "        out.append(total)\n    return out\n",
    "prefix-range-query":
        "    prefix = [0]\n    total = 0\n    for x in xs:\n        total += x\n"
        "        prefix.append(total)\n    return [prefix[right] - prefix[left] "
        "for left, right in queries]\n",
    "fixed-sliding-window":
        "    if k <= 0 or k > len(xs):\n        return None\n    total = sum(xs[:k])\n"
        "    best = total\n    for i in range(k, len(xs)):\n        total += xs[i] - xs[i - k]\n"
        "        if total > best:\n            best = total\n    return best\n",
    "position-index":
        "    index = {}\n    for i, x in enumerate(xs):\n        if x not in index:\n"
        "            index[x] = i\n    return [index.get(q, -1) for q in queries]\n",
    "count-index":
        "    counts = {}\n    for x in xs:\n        counts[x] = counts.get(x, 0) + 1\n"
        "    return [counts.get(q, 0) for q in queries]\n",
    "membership-fusion":
        "    present = set(ys)\n    count = 0\n    for x in xs:\n        if x in present:\n"
        "            count += 1\n    return count\n",
    "ordered-intersection":
        "    present = set(ys)\n    out = []\n    used = set()\n    for x in xs:\n"
        "        if x in present and x not in used:\n            used.add(x)\n"
        "            out.append(x)\n    return out\n",
    "mode-bucket":
        "    if not xs:\n        return None\n    counts = {}\n    for x in xs:\n"
        "        counts[x] = counts.get(x, 0) + 1\n    best = None\n    best_count = -1\n"
        "    for value in counts:\n        if counts[value] > best_count or "
        "(counts[value] == best_count and value < best):\n            best = value\n"
        "            best_count = counts[value]\n    return best\n",
    "residue-count-index":
        "    counts = {}\n    for x in xs:\n        bucket = x % bucket_size\n"
        "        counts[bucket] = counts.get(bucket, 0) + 1\n"
        "    return [counts.get(q % bucket_size, 0) for q in queries]\n",
}

# Benign distractor snippets inserted (in the slow reference only) between the
# prelude and the real body. They reference only ``xs`` (present in every
# signature), never affect the output, and double the surface-form templates
# with "mixed benign distractor code" per the plan.
DISTRACTORS = (
    "",
    "    warmup = 0\n    for probe in xs:\n        warmup += 1\n",
)

# ----------------------------------------------------------------------------
# Slow body banks. TRAIN_BODIES[family] -> training-only structures.
# DEV_BODIES[family] -> fresh structures reserved for the behavioral dev split,
# structurally disjoint from every training body (and from v1).
# ----------------------------------------------------------------------------
TRAIN_BODIES: dict[str, list[str]] = {}
DEV_BODIES: dict[str, list[str]] = {}

TRAIN_BODIES["hash-membership"] = [
    "    i = 0\n    while i < len(xs):\n        j = i + 1\n        while j < len(xs):\n            if xs[i] == xs[j]:\n                return True\n            j = j + 1\n        i = i + 1\n    return False\n",
    "    return any(xs[i] == xs[j] for i in range(len(xs)) for j in range(i + 1, len(xs)))\n",
    "    duplicate = False\n    for i in range(len(xs)):\n        for j in range(i + 1, len(xs)):\n            if xs[i] == xs[j]:\n                duplicate = True\n                break\n        if duplicate:\n            break\n    return duplicate\n",
    "    for i in range(len(xs)):\n        if xs[i] in xs[i + 1:]:\n            return True\n    return False\n",
    "    for x in xs:\n        total = 0\n        for y in xs:\n            if x == y:\n                total += 1\n        if total > 1:\n            return True\n    return False\n",
    "    seen = []\n    for x in xs:\n        found = False\n        for y in seen:\n            if y == x:\n                found = True\n                break\n        if found:\n            return True\n        seen.append(x)\n    return False\n",
    "    matches = 0\n    for i, a in enumerate(xs):\n        for b in xs[i + 1:]:\n            if a == b:\n                matches += 1\n    return matches > 0\n",
    "    n = len(xs)\n    for i in range(n - 1, 0, -1):\n        for j in range(i):\n            if xs[i] == xs[j]:\n                return True\n    return False\n",
]
DEV_BODIES["hash-membership"] = [
    "    duplicate_pairs = 0\n    for i in range(len(xs)):\n        for j in range(len(xs)):\n            if i < j and xs[i] == xs[j]:\n                duplicate_pairs += 1\n    return duplicate_pairs != 0\n",
    "    for first in range(len(xs)):\n        target_value = xs[first]\n        for second in range(first + 1, len(xs)):\n            if xs[second] == target_value:\n                return True\n    return False\n",
    "    ordered = sorted(xs)\n    for i in range(1, len(ordered)):\n        if ordered[i] == ordered[i - 1]:\n            return True\n    return False\n",
]

TRAIN_BODIES["distinct-cardinality"] = [
    "    total = 0\n    for i in range(len(xs)):\n        seen_before = False\n        for j in range(i):\n            if xs[j] == xs[i]:\n                seen_before = True\n        if not seen_before:\n            total += 1\n    return total\n",
    "    uniques = []\n    for x in xs:\n        if x in uniques:\n            continue\n        uniques.append(x)\n    return len(uniques)\n",
    "    return sum(1 for i in range(len(xs)) if xs[i] not in xs[i + 1:])\n",
    "    kept = []\n    i = 0\n    while i < len(xs):\n        present = False\n        for value in kept:\n            if value == xs[i]:\n                present = True\n                break\n        if not present:\n            kept.append(xs[i])\n        i += 1\n    return len(kept)\n",
    "    firsts = 0\n    for i in range(len(xs)):\n        earliest = True\n        for j in range(i):\n            if xs[j] == xs[i]:\n                earliest = False\n                break\n        if earliest:\n            firsts += 1\n    return firsts\n",
    "    seen = []\n    distinct = 0\n    for x in xs:\n        if x not in seen:\n            distinct += 1\n            seen.append(x)\n    return distinct\n",
    "    firsts = [i for i in range(len(xs)) if xs.index(xs[i]) == i]\n    return len(firsts)\n",
    "    ordered = sorted(xs)\n    total = len(ordered)\n    for i in range(1, len(ordered)):\n        if ordered[i] == ordered[i - 1]:\n            total -= 1\n    return total\n",
]
DEV_BODIES["distinct-cardinality"] = [
    "    tally = 0\n    for position, value in enumerate(xs):\n        earlier = xs[:position]\n        if value not in earlier:\n            tally += 1\n    return tally\n",
    "    bag = []\n    for x in reversed(xs):\n        if x not in bag:\n            bag.append(x)\n    return len(bag)\n",
    "    return len([x for i, x in enumerate(xs) if x not in xs[:i]])\n",
]

TRAIN_BODIES["stable-deduplication"] = [
    "    out = []\n    for i in range(len(xs)):\n        keep = True\n        for j in range(i):\n            if xs[j] == xs[i]:\n                keep = False\n                break\n        if keep:\n            out.append(xs[i])\n    return out\n",
    "    out = []\n    for x in xs:\n        if x in out:\n            continue\n        out.append(x)\n    return out\n",
    "    return [xs[i] for i in range(len(xs)) if xs[i] not in xs[:i]]\n",
    "    result = []\n    i = 0\n    while i < len(xs):\n        already = False\n        for value in result:\n            if value == xs[i]:\n                already = True\n                break\n        if not already:\n            result.append(xs[i])\n        i += 1\n    return result\n",
    "    out = []\n    for index, value in enumerate(xs):\n        if value not in xs[:index]:\n            out.append(value)\n    return out\n",
    "    out = []\n    for x in xs:\n        seen_count = 0\n        for y in out:\n            if y == x:\n                seen_count += 1\n        if seen_count == 0:\n            out.append(x)\n    return out\n",
    "    kept = []\n    for x in xs:\n        duplicate = any(x == y for y in kept)\n        if not duplicate:\n            kept.append(x)\n    return kept\n",
    "    out = []\n    for i in range(len(xs)):\n        if xs.index(xs[i]) == i:\n            out.append(xs[i])\n    return out\n",
]
DEV_BODIES["stable-deduplication"] = [
    "    order = []\n    for position in range(len(xs)):\n        earlier = xs[:position]\n        if xs[position] not in earlier:\n            order.append(xs[position])\n    return order\n",
    "    out = []\n    for x in xs:\n        found = False\n        j = 0\n        while j < len(out):\n            if out[j] == x:\n                found = True\n                break\n            j += 1\n        if not found:\n            out.append(x)\n    return out\n",
    "    picks = []\n    for x in xs:\n        if x not in picks:\n            picks = picks + [x]\n    return picks\n",
]

TRAIN_BODIES["frequency-index"] = [
    "    out = []\n    for x in xs:\n        total = 0\n        for y in xs:\n            if y == x:\n                total = total + 1\n        out.append(total)\n    return out\n",
    "    return [sum(1 for y in xs if y == x) for x in xs]\n",
    "    result = []\n    for i in range(len(xs)):\n        c = 0\n        for j in range(len(xs)):\n            if xs[j] == xs[i]:\n                c += 1\n        result.append(c)\n    return result\n",
    "    out = []\n    i = 0\n    while i < len(xs):\n        total = 0\n        for y in xs:\n            if y == xs[i]:\n                total += 1\n        out.append(total)\n        i += 1\n    return out\n",
    "    counts = []\n    for x in xs:\n        counts.append(xs.count(x))\n    return counts\n",
    "    out = []\n    for x in xs:\n        matches = [y for y in xs if y == x]\n        out.append(len(matches))\n    return out\n",
    "    out = []\n    for i, x in enumerate(xs):\n        total = 0\n        for j, y in enumerate(xs):\n            if x == y:\n                total += 1\n        out.append(total)\n    return out\n",
    "    result = []\n    for x in xs:\n        seen = 0\n        for y in reversed(xs):\n            if x == y:\n                seen += 1\n        result.append(seen)\n    return result\n",
]
DEV_BODIES["frequency-index"] = [
    "    out = []\n    for position in range(len(xs)):\n        value = xs[position]\n        total = 0\n        for other in xs:\n            if other == value:\n                total += 1\n        out.append(total)\n    return out\n",
    "    return [len([y for y in xs if y == x]) for x in xs]\n",
    "    tallies = []\n    for x in xs:\n        total = 0\n        k = 0\n        while k < len(xs):\n            if xs[k] == x:\n                total += 1\n            k += 1\n        tallies.append(total)\n    return tallies\n",
]

TRAIN_BODIES["complement-lookup"] = [
    "    n = len(xs)\n    for i in range(n):\n        for j in range(i + 1, n):\n            if xs[i] + xs[j] == target:\n                return True\n    return False\n",
    "    for i, a in enumerate(xs):\n        for b in xs[i + 1:]:\n            if a + b == target:\n                return True\n    return False\n",
    "    found = False\n    for i in range(len(xs)):\n        for j in range(i):\n            if xs[i] + xs[j] == target:\n                found = True\n    return found\n",
    "    i = 0\n    while i < len(xs):\n        j = i + 1\n        while j < len(xs):\n            if xs[i] + xs[j] == target:\n                return True\n            j += 1\n        i += 1\n    return False\n",
    "    return any(xs[i] + xs[j] == target for i in range(len(xs)) for j in range(i + 1, len(xs)))\n",
    "    for i, a in enumerate(xs):\n        for k in range(i):\n            if a + xs[k] == target:\n                return True\n    return False\n",
    "    hit = False\n    for i in range(len(xs)):\n        for j in range(i + 1, len(xs)):\n            if xs[i] + xs[j] == target:\n                hit = True\n                break\n        if hit:\n            break\n    return hit\n",
    "    for a in range(len(xs)):\n        complement = target - xs[a]\n        for b in range(a + 1, len(xs)):\n            if xs[b] == complement:\n                return True\n    return False\n",
]
DEV_BODIES["complement-lookup"] = [
    "    hits = 0\n    for i in range(len(xs)):\n        for j in range(i + 1, len(xs)):\n            if xs[i] + xs[j] == target:\n                hits += 1\n    return hits > 0\n",
    "    for first in range(len(xs)):\n        for second in range(first + 1, len(xs)):\n            if target == xs[first] + xs[second]:\n                return True\n    return False\n",
    "    pairs = [(xs[i], xs[j]) for i in range(len(xs)) for j in range(i + 1, len(xs))]\n    for a, b in pairs:\n        if a + b == target:\n            return True\n    return False\n",
]

TRAIN_BODIES["running-aggregate"] = [
    "    out = []\n    for i in range(len(xs)):\n        total = 0\n        for j in range(i + 1):\n            total += xs[j]\n        out.append(total)\n    return out\n",
    "    return [sum(xs[:i + 1]) for i in range(len(xs))]\n",
    "    out = []\n    i = 0\n    while i < len(xs):\n        total = 0\n        for j in range(i + 1):\n            total = total + xs[j]\n        out.append(total)\n        i += 1\n    return out\n",
    "    out = []\n    for i, x in enumerate(xs):\n        out.append(sum(xs[:i + 1]))\n    return out\n",
    "    out = []\n    for i in range(len(xs)):\n        window = xs[0:i + 1]\n        out.append(sum(window))\n    return out\n",
    "    prefixes = []\n    for i in range(len(xs)):\n        total = 0\n        for value in xs[:i + 1]:\n            total += value\n        prefixes.append(total)\n    return prefixes\n",
    "    return [sum(xs[j] for j in range(i + 1)) for i in range(len(xs))]\n",
    "    out = []\n    for i in range(len(xs)):\n        out.append(sum(xs[k] for k in range(0, i + 1)))\n    return out\n",
]
DEV_BODIES["running-aggregate"] = [
    "    result = []\n    for position in range(len(xs)):\n        acc = 0\n        for other in range(position + 1):\n            acc += xs[other]\n        result.append(acc)\n    return result\n",
    "    out = []\n    for i in range(len(xs)):\n        head = xs[:i + 1]\n        total = 0\n        for value in head:\n            total = total + value\n        out.append(total)\n    return out\n",
    "    return [sum([xs[j] for j in range(i + 1)]) for i in range(len(xs))]\n",
]

TRAIN_BODIES["prefix-range-query"] = [
    "    out = []\n    for left, right in queries:\n        total = 0\n        for i in range(left, right):\n            total += xs[i]\n        out.append(total)\n    return out\n",
    "    return [sum(xs[left:right]) for left, right in queries]\n",
    "    out = []\n    for query in queries:\n        left = query[0]\n        right = query[1]\n        total = 0\n        for i in range(left, right):\n            total = total + xs[i]\n        out.append(total)\n    return out\n",
    "    return [sum(xs[i] for i in range(left, right)) for left, right in queries]\n",
    "    out = []\n    for left, right in queries:\n        values = []\n        for i in range(left, right):\n            values.append(xs[i])\n        out.append(sum(values))\n    return out\n",
    "    answers = []\n    q = 0\n    while q < len(queries):\n        left, right = queries[q]\n        total = 0\n        for i in range(left, right):\n            total += xs[i]\n        answers.append(total)\n        q += 1\n    return answers\n",
    "    out = []\n    for left, right in queries:\n        out.append(sum(xs[left:right]))\n    return out\n",
    "    return [sum(value for value in xs[left:right]) for left, right in queries]\n",
]
DEV_BODIES["prefix-range-query"] = [
    "    result = []\n    for pair in queries:\n        lo, hi = pair[0], pair[1]\n        acc = 0\n        for idx in range(lo, hi):\n            acc += xs[idx]\n        result.append(acc)\n    return result\n",
    "    out = []\n    for left, right in queries:\n        segment = xs[left:right]\n        total = 0\n        for value in segment:\n            total = total + value\n        out.append(total)\n    return out\n",
    "    return [sum(xs[k] for k in range(pair[0], pair[1])) for pair in queries]\n",
]

TRAIN_BODIES["fixed-sliding-window"] = [
    "    if k <= 0 or k > len(xs):\n        return None\n    best = None\n    for i in range(len(xs) - k + 1):\n        total = 0\n        for j in range(i, i + k):\n            total += xs[j]\n        if best is None or total > best:\n            best = total\n    return best\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    return max(sum(xs[i:i + k]) for i in range(len(xs) - k + 1))\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    values = []\n    for i in range(len(xs) - k + 1):\n        values.append(sum(xs[i:i + k]))\n    return max(values)\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    best = sum(xs[:k])\n    for i in range(1, len(xs) - k + 1):\n        current = sum(xs[i:i + k])\n        if current > best:\n            best = current\n    return best\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    return max(sum(xs[j] for j in range(i, i + k)) for i in range(len(xs) - k + 1))\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    best = None\n    i = 0\n    while i <= len(xs) - k:\n        window = xs[i:i + k]\n        total = sum(window)\n        if best is None or total > best:\n            best = total\n        i += 1\n    return best\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    sums = [sum(xs[i:i + k]) for i in range(len(xs) - k + 1)]\n    best = sums[0]\n    for value in sums[1:]:\n        if value > best:\n            best = value\n    return best\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    best = None\n    for i in range(len(xs) - k + 1):\n        total = 0\n        for value in xs[i:i + k]:\n            total += value\n        if best is None or total > best:\n            best = total\n    return best\n",
]
DEV_BODIES["fixed-sliding-window"] = [
    "    if k <= 0 or k > len(xs):\n        return None\n    top = None\n    for start in range(len(xs) - k + 1):\n        acc = 0\n        for offset in range(k):\n            acc += xs[start + offset]\n        if top is None or acc > top:\n            top = acc\n    return top\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    candidates = []\n    for i in range(len(xs) - k + 1):\n        segment = xs[i:i + k]\n        total = 0\n        for value in segment:\n            total = total + value\n        candidates.append(total)\n    return max(candidates)\n",
    "    if k <= 0 or k > len(xs):\n        return None\n    best = sum(xs[0:k])\n    for i in range(len(xs) - k + 1):\n        total = sum(xs[i:i + k])\n        if total > best:\n            best = total\n    return best\n",
]

TRAIN_BODIES["position-index"] = [
    "    out = []\n    for q in queries:\n        found = -1\n        for i in range(len(xs)):\n            if xs[i] == q:\n                found = i\n                break\n        out.append(found)\n    return out\n",
    "    out = []\n    for q in queries:\n        position = -1\n        for i, x in enumerate(xs):\n            if x == q and position == -1:\n                position = i\n        out.append(position)\n    return out\n",
    "    result = []\n    for q in queries:\n        matches = [i for i in range(len(xs)) if xs[i] == q]\n        result.append(matches[0] if matches else -1)\n    return result\n",
    "    out = []\n    j = 0\n    while j < len(queries):\n        q = queries[j]\n        found = -1\n        i = 0\n        while i < len(xs):\n            if xs[i] == q:\n                found = i\n                break\n            i += 1\n        out.append(found)\n        j += 1\n    return out\n",
    "    out = []\n    for q in queries:\n        answer = -1\n        for i in range(len(xs) - 1, -1, -1):\n            if xs[i] == q:\n                answer = i\n        out.append(answer)\n    return out\n",
]
TRAIN_BODIES["count-index"] = [
    "    out = []\n    for q in queries:\n        total = 0\n        for x in xs:\n            if x == q:\n                total += 1\n        out.append(total)\n    return out\n",
    "    return [xs.count(q) for q in queries]\n",
    "    out = []\n    for q in queries:\n        matches = [x for x in xs if x == q]\n        out.append(len(matches))\n    return out\n",
    "    result = []\n    j = 0\n    while j < len(queries):\n        q = queries[j]\n        total = 0\n        i = 0\n        while i < len(xs):\n            if xs[i] == q:\n                total += 1\n            i += 1\n        result.append(total)\n        j += 1\n    return result\n",
    "    return [sum(1 for x in xs if x == q) for q in queries]\n",
]
TRAIN_BODIES["membership-fusion"] = [
    "    count = 0\n    for x in xs:\n        if x in ys:\n            count += 1\n    return count\n",
    "    count = 0\n    for x in xs:\n        found = False\n        for y in ys:\n            if x == y:\n                found = True\n                break\n        if found:\n            count += 1\n    return count\n",
    "    return sum(1 for x in xs if x in ys)\n",
    "    count = 0\n    i = 0\n    while i < len(xs):\n        j = 0\n        hit = False\n        while j < len(ys):\n            if xs[i] == ys[j]:\n                hit = True\n                break\n            j += 1\n        if hit:\n            count += 1\n        i += 1\n    return count\n",
    "    matched = [x for x in xs if x in ys]\n    return len(matched)\n",
]
TRAIN_BODIES["ordered-intersection"] = [
    "    out = []\n    for x in xs:\n        if x in ys and x not in out:\n            out.append(x)\n    return out\n",
    "    out = []\n    for x in xs:\n        inside = False\n        for y in ys:\n            if x == y:\n                inside = True\n                break\n        if inside and x not in out:\n            out.append(x)\n    return out\n",
    "    out = []\n    for x in xs:\n        if x not in ys:\n            continue\n        already = False\n        for kept in out:\n            if kept == x:\n                already = True\n                break\n        if not already:\n            out.append(x)\n    return out\n",
    "    seen = []\n    for x in xs:\n        if x in ys and x not in seen:\n            seen.append(x)\n    return seen\n",
    "    out = []\n    i = 0\n    while i < len(xs):\n        x = xs[i]\n        if x in ys and x not in out:\n            out.append(x)\n        i += 1\n    return out\n",
]
TRAIN_BODIES["mode-bucket"] = [
    "    if not xs:\n        return None\n    best = None\n    best_count = -1\n    for x in xs:\n        c = xs.count(x)\n        if c > best_count or (c == best_count and x < best):\n            best = x\n            best_count = c\n    return best\n",
    "    if len(xs) == 0:\n        return None\n    best = None\n    best_count = -1\n    for x in xs:\n        c = 0\n        for y in xs:\n            if y == x:\n                c += 1\n        if c > best_count or (c == best_count and x < best):\n            best = x\n            best_count = c\n    return best\n",
    "    if not xs:\n        return None\n    best = None\n    best_count = -1\n    for x in xs:\n        c = len([y for y in xs if y == x])\n        if c > best_count or (c == best_count and x < best):\n            best = x\n            best_count = c\n    return best\n",
    "    if not xs:\n        return None\n    best = None\n    best_count = -1\n    i = 0\n    while i < len(xs):\n        x = xs[i]\n        c = xs.count(x)\n        if c > best_count or (c == best_count and x < best):\n            best = x\n            best_count = c\n        i += 1\n    return best\n",
    "    if not xs:\n        return None\n    best = None\n    best_count = -1\n    for x in xs:\n        c = sum(1 for y in xs if y == x)\n        if c > best_count or (c == best_count and x < best):\n            best = x\n            best_count = c\n    return best\n",
]

TRAIN_BODIES["residue-count-index"] = [
    ("    out = []\n    for q in queries:\n        total = 0\n        for x in xs:\n"
    "            if x % bucket_size == q % bucket_size:\n                total += 1\n"
    "        out.append(total)\n    return out\n"),
    "    return [sum(1 for x in xs if x % bucket_size == q % bucket_size) for q in queries]\n",
    ("    answers = []\n    for query_index in range(len(queries)):\n        matches = 0\n"
    "        for value in xs:\n            if value % bucket_size == queries[query_index] % bucket_size:\n"
    "                matches += 1\n        answers.append(matches)\n    return answers\n"),
    ("    out = []\n    i = 0\n    while i < len(queries):\n        count = 0\n"
    "        for x in xs:\n            if (x - queries[i]) % bucket_size == 0:\n                count += 1\n"
    "        out.append(count)\n        i += 1\n    return out\n"),
    ("    result = []\n    for q in queries:\n        matching = []\n        for x in xs:\n"
    "            if x % bucket_size == q % bucket_size:\n                matching.append(x)\n"
    "        result.append(len(matching))\n    return result\n"),
    ("    result = []\n    for q in queries:\n        found = 0\n        for x in reversed(xs):\n"
    "            if q % bucket_size == x % bucket_size:\n                found = found + 1\n"
    "        result.append(found)\n    return result\n"),
]
DEV_BODIES["residue-count-index"] = [
    ("    answers = []\n    for query in queries:\n        tally = 0\n        for position in range(len(xs)):\n"
    "            if xs[position] % bucket_size == query % bucket_size:\n                tally += 1\n"
    "        answers.append(tally)\n    return answers\n"),
    ("    return [len([value for value in xs if (value - query) % bucket_size == 0]) "
    "for query in queries]\n"),
    ("    out = []\n    for query in reversed(queries):\n        count = 0\n        for value in xs:\n"
    "            if value % bucket_size == query % bucket_size:\n                count += 1\n"
    "        out.append(count)\n    out.reverse()\n    return out\n"),
]


def prelude(family: str, value: int) -> str:
    """Input transform applied identically to slow and fast programs.

    Gives each generated instance a distinct source/AST fingerprint (the literal
    is retained by the normalized-AST hash) and a meaningful input contract,
    while leaving the optimization family and slow structure fixed.
    """
    modulus = 2 + value
    equality_int = {
        "hash-membership", "distinct-cardinality", "stable-deduplication",
        "frequency-index", "mode-bucket",
    }
    if family in equality_int:
        return f"    xs = [x % {modulus} for x in xs]\n"
    if family == "complement-lookup":
        return f"    xs = [x + {value} for x in xs]\n"
    if family in {"running-aggregate", "prefix-range-query", "fixed-sliding-window"}:
        return f"    xs = [x + {value} for x in xs]\n"
    if family in {"position-index", "count-index"}:
        return (f"    xs = [x % {modulus} for x in xs]\n"
                f"    queries = [q % {modulus} for q in queries]\n")
    if family == "residue-count-index":
        return f"    bucket_size = {modulus}\n"
    if family in {"membership-fusion", "ordered-intersection"}:
        return (f"    xs = [x % {modulus} for x in xs]\n"
                f"    ys = [y % {modulus} for y in ys]\n")
    raise ValueError(f"no v2 prelude for family: {family}")


def slow_source(family: str, body: str, distractor: str, value: int) -> str:
    """Assemble the public slow reference: header + prelude + distractor + body."""
    return (f"def solve({SIGNATURE[family]}):\n"
            + prelude(family, value) + distractor + body)


def fast_source(family: str, value: int) -> str:
    """Assemble the in-memory known-fast reference (never serialized)."""
    return (f"def solve({SIGNATURE[family]}):\n"
            + prelude(family, value) + FAST_BODY[family])
