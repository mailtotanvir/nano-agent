#!/usr/bin/env python3
"""Freeze the evaluation set and split corpus into eval / train pools.

The eval set is frozen FIRST (spec section 4.6). Its case_ids (content hashes of
the broken source) are recorded so the teacher/training pipeline can EXCLUDE them
and guarantee zero train/test leakage. Splitting is deterministic (seeded, sorted
by case_id) so the frozen set is reproducible.

Usage:
    python split_eval.py --corpus corpus.jsonl --eval-frac 0.35 --seed 7 \
        --out-eval datasets/eval_v0.jsonl --out-train datasets/train_pool_v0.jsonl \
        --out-manifest datasets/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def stratified_split(cases: list[dict], eval_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratify by category so each error class appears in both splits."""
    import random
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c)
    eval_set, train_set = [], []
    for cat, items in sorted(by_cat.items()):
        items = sorted(items, key=lambda c: c["case_id"])  # deterministic
        rng = random.Random(f"{seed}:{cat}")
        rng.shuffle(items)
        n_eval = max(1, round(len(items) * eval_frac)) if len(items) > 1 else 0
        eval_set.extend(items[:n_eval])
        train_set.extend(items[n_eval:])
    return eval_set, train_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()

    cases = load(Path(args.corpus))
    # Dedup by case_id (content hash of broken source).
    uniq: dict[str, dict] = {}
    for c in cases:
        uniq[c["case_id"]] = c
    cases = list(uniq.values())

    eval_set, train_set = stratified_split(cases, args.eval_frac, args.seed)
    eval_ids = {c["case_id"] for c in eval_set}
    train_ids = {c["case_id"] for c in train_set}
    assert eval_ids.isdisjoint(train_ids), "LEAK: eval/train case_id overlap"

    for path, rows in [(args.out_eval, eval_set), (args.out_train, train_set)]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def cat_hist(rows: list[dict]) -> dict[str, int]:
        h: dict[str, int] = defaultdict(int)
        for r in rows:
            h[r["category"]] += 1
        return dict(sorted(h.items()))

    manifest = {
        "corpus_size": len(cases),
        "eval_size": len(eval_set),
        "train_size": len(train_set),
        "eval_frac": args.eval_frac,
        "seed": args.seed,
        "eval_by_category": cat_hist(eval_set),
        "train_by_category": cat_hist(train_set),
        "eval_ids_sha": hashlib.sha1(
            "".join(sorted(eval_ids)).encode()
        ).hexdigest(),
        "eval_case_ids": sorted(eval_ids),
    }
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=2))
    print(f"eval={len(eval_set)} train={len(train_set)} corpus={len(cases)}")
    print("eval by category:", json.dumps(manifest["eval_by_category"]))
    print("frozen eval_ids_sha:", manifest["eval_ids_sha"])


if __name__ == "__main__":
    main()
