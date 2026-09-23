#!/usr/bin/env python3
"""Freeze the layered v1 evaluation and audit leakage invariants.

The script accepts a verifier-gated corpus produced by ``gen_problems.py``.
It writes the training and development pools separately from the two frozen eval
tracks.  Neither eval file should be used for prompt tuning, checkpoint choice,
or reward-rule changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_NAMES = ("train", "development", "seen_family_eval", "family_heldout_eval")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stable_shuffle(rows: list[dict], seed: int, label: str) -> list[dict]:
    rows = sorted(rows, key=lambda row: row["id"])
    rng = random.Random(f"{seed}:{label}")
    rng.shuffle(rows)
    return rows


def build_split(corpus: list[dict], *, seed: int = 73) -> dict[str, list[dict]]:
    """Build the fixed 152/38/30/30 layout from the v1 250-row design."""
    identifiers = [row["id"] for row in corpus]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate record IDs")
    by_hint: dict[str, list[dict]] = defaultdict(list)
    for row in corpus:
        by_hint[row.get("split_hint")].append(row)
    seen = by_hint["seen_eval"]
    heldout = by_hint["heldout_eval"]
    train_dev = by_hint["train_dev"]
    if len(seen) != 30 or len(heldout) != 30 or len(train_dev) < 190:
        raise ValueError("corpus does not contain the required v1 30/30/190 split hints")
    # Eval entries are one record per deliberately withheld source template.
    if len({row["template_id"] for row in seen}) != 30:
        raise ValueError("seen-family eval must contain 30 distinct withheld templates")
    heldout_families = {row["family_id"] for row in heldout}
    if len(heldout_families) < 3:
        raise ValueError("family-held-out evaluation needs at least three excluded families")
    # Extra train/dev instances from an enlarged corpus remain in train; the
    # original 190 are partitioned deterministically into 152/38.
    by_template: dict[str, list[dict]] = defaultdict(list)
    for row in train_dev:
        by_template[row["template_id"]].append(row)
    development, train = [], []
    for index, (template, rows) in enumerate(sorted(by_template.items())):
        rows = _stable_shuffle(rows, seed, template)
        # Development is source-disjoint and spans 38 of the 42 train/dev
        # structures.  It is for tuning; neither frozen evaluation file is.
        dev_count = 1 if index < 38 else 0
        development.extend(rows[:dev_count])
        train.extend(rows[dev_count:])
    split = {
        "train": sorted(train, key=lambda row: row["id"]),
        "development": sorted(development, key=lambda row: row["id"]),
        "seen_family_eval": sorted(seen, key=lambda row: row["id"]),
        "family_heldout_eval": sorted(heldout, key=lambda row: row["id"]),
    }
    check_invariants(split)
    return split


def _hashes(rows: list[dict], key: str) -> set[str]:
    return {row[key] for row in rows}


def check_invariants(split: dict[str, list[dict]]) -> None:
    """Fail closed if a public source, template, family, or variant leaks."""
    if set(split) != set(SPLIT_NAMES):
        raise ValueError("unexpected split names")
    all_rows = [row for name in SPLIT_NAMES for row in split[name]]
    ids = [row["id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("record ID overlap")
    hashes = {name: _hashes(split[name], "normalized_ast_sha256") for name in SPLIT_NAMES}
    for left in SPLIT_NAMES:
        for right in SPLIT_NAMES:
            if left < right and not hashes[left].isdisjoint(hashes[right]):
                raise ValueError(f"normalized AST hash leakage: {left}/{right}")
    train_dev_templates = _hashes(split["train"] + split["development"], "template_id")
    seen_templates = _hashes(split["seen_family_eval"], "template_id")
    if len(seen_templates) != 30 or not train_dev_templates.isdisjoint(seen_templates):
        raise ValueError("seen-family template leakage")
    heldout_families = _hashes(split["family_heldout_eval"], "family_id")
    train_dev_families = _hashes(split["train"] + split["development"], "family_id")
    if len(heldout_families) < 3 or not heldout_families.isdisjoint(train_dev_families):
        raise ValueError("family-held-out leakage")
    for name in ("seen_family_eval", "family_heldout_eval"):
        for row in split[name]:
            variants = set(row["benchmark_variant_ids"])
            if not {"n_minus_1", "n", "n_plus_1", "values_changed", "order_changed"} <= variants:
                raise ValueError(f"missing adversarial benchmark neighbours in {row['id']}")
            if "private_seed_policy" not in row:
                raise ValueError(f"missing private seed policy in {row['id']}")


def _family_hist(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["family_id"] for row in rows).items()))


def _canonical_record_hash(rows: list[dict]) -> str:
    """Hash complete records in stable ID order, including verifier-private data."""
    encoded = [json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
               for row in sorted(rows, key=lambda row: row["id"])]
    return hashlib.sha256("\n".join(encoded).encode("utf-8")).hexdigest()


def _normalized_ast_set_hash(rows: list[dict]) -> str:
    """Hash only AST fingerprints; this is a leakage signal, not provenance."""
    return hashlib.sha256(
        "\n".join(sorted(row["normalized_ast_sha256"] for row in rows)).encode("utf-8")
    ).hexdigest()


def manifest_for(split: dict[str, list[dict]], *, corpus_seed: int, split_seed: int) -> dict:
    check_invariants(split)
    all_rows = [row for name in SPLIT_NAMES for row in split[name]]
    verification_states = {row.get("verification", {}).get("status") for row in all_rows}
    verified = verification_states == {"passed"}
    heldout = sorted(_hashes(split["family_heldout_eval"], "family_id"))
    return {
        "schema_version": 2,
        "corpus_seed": corpus_seed,
        "split_seed": split_seed,
        "corpus_verification": "verified" if verified else "unverified",
        "sizes": {name: len(split[name]) for name in SPLIT_NAMES},
        "family_inventory": _family_hist(all_rows),
        "track_family_histograms": {
            "train": _family_hist(split["train"]),
            "development": _family_hist(split["development"]),
            "seen_family_eval": _family_hist(split["seen_family_eval"]),
            "family_heldout_eval": _family_hist(split["family_heldout_eval"]),
        },
        "heldout_family_ids": heldout,
        "eval_policy": {
            "frozen": True,
            "forbidden_uses": ["prompt_tuning", "checkpoint_selection", "reward_revision"],
            "private_correctness": "fresh entropy is drawn after candidate receipt",
            "performance": "aggregate log-speedup over n-1/n/n+1 plus changed values and order",
        },
        "invariants": {
            "normalized_ast_disjoint": True,
            "seen_eval_template_disjoint": True,
            "heldout_families_excluded_from_train_and_development": True,
            "private_seed_isolation_declared": True,
            "adversarial_benchmark_neighbours_present": True,
        },
        "split_full_record_sha256": {
            name: _canonical_record_hash(split[name]) for name in SPLIT_NAMES
        },
        "full_corpus_content_sha256": _canonical_record_hash(all_rows),
        "full_corpus_normalized_ast_sha256": _normalized_ast_set_hash(all_rows),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def v2_manifest(train_rows: list[dict], dev_rows: list[dict], *,
                corpus_seed: int, dev_seed: int) -> dict:
    """Additive manifest for the v2 training-only corpus + behavioral dev split.

    This is deliberately a SEPARATE artifact from the frozen v1 ``manifest.json``:
    the v1 manifest entries are never modified. ``train`` here is training-only
    (all rows carry ``train_v2``); ``dev_behavioral`` is training-infrastructure
    that MAY be used for checkpoint selection, unlike the frozen v1 evals.
    """
    from datagen import _v2_bank as bank

    def by_family(rows):
        return _family_hist(rows)

    def templates(rows):
        return sorted({row["template_id"] for row in rows})

    train_verified = {row.get("verification", {}).get("status") for row in train_rows} == {"passed"}
    dev_verified = {row.get("verification", {}).get("status") for row in dev_rows} == {"passed"}
    seen = [r for r in train_rows if r["family_id"] in bank.SEEN_FAMILIES]
    new = [r for r in train_rows if r["family_id"] in bank.NEW_FAMILIES]
    return {
        "schema_version": 1,
        "role": "training-only corpus + behavioral dev split (v2)",
        "relationship_to_v1": "additive; the frozen v1 manifest.json and eval "
                              "files are never modified or trained on",
        "corpus_seed": corpus_seed,
        "dev_seed": dev_seed,
        "train_verification": "verified" if train_verified else "unverified",
        "dev_verification": "verified" if dev_verified else "unverified",
        "sizes": {
            "train_v2": len(train_rows),
            "train_v2_seen_families": len(seen),
            "train_v2_new_families": len(new),
            "dev_behavioral": len(dev_rows),
        },
        "template_counts": {
            "train_v2_total": len(templates(train_rows)),
            "train_v2_seen_families": len(templates(seen)),
            "train_v2_new_families": len(templates(new)),
            "dev_behavioral": len(templates(dev_rows)),
        },
        "seen_family_ids": list(bank.SEEN_FAMILIES),
        "new_family_ids": list(bank.NEW_FAMILIES),
        "new_family_rationale": bank.NEW_FAMILY_RATIONALE,
        "frozen_heldout_family_ids_excluded": list(bank.FROZEN_HELDOUT_FAMILIES),
        "train_family_histogram": by_family(train_rows),
        "dev_family_histogram": by_family(dev_rows),
        "dev_policy": {
            "frozen": False,
            "permitted_uses": ["checkpoint_selection", "iteration_metric"],
            "note": "template- and AST-disjoint from all training data and from "
                    "both frozen v1 eval tracks",
        },
        "train_full_record_sha256": _canonical_record_hash(train_rows),
        "dev_full_record_sha256": _canonical_record_hash(dev_rows),
        "train_normalized_ast_sha256": _normalized_ast_set_hash(train_rows),
        "dev_normalized_ast_sha256": _normalized_ast_set_hash(dev_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-seed", type=int, default=20_260_907)
    parser.add_argument("--split-seed", type=int, default=73)
    args = parser.parse_args()
    split = build_split(load_jsonl(args.corpus), seed=args.split_seed)
    out = args.out_dir
    verified = all(row.get("verification", {}).get("status") == "passed"
                   for rows in split.values() for row in rows)
    suffix = "" if verified else ".unverified"
    write_jsonl(out / f"train_speedup_v1{suffix}.jsonl", split["train"])
    write_jsonl(out / f"development_speedup_v1{suffix}.jsonl", split["development"])
    # The two tracks intentionally remain separate files to prevent an easier
    # seen-family headline from concealing cross-family performance.
    write_jsonl(out / f"eval_speedup_v1_seen_family{suffix}.jsonl", split["seen_family_eval"])
    write_jsonl(out / f"eval_speedup_v1_family_heldout{suffix}.jsonl", split["family_heldout_eval"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest_for(split, corpus_seed=args.corpus_seed,
                                                     split_seed=args.split_seed), indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in split.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
