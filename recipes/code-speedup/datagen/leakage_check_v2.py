#!/usr/bin/env python3
"""Airtight leakage audit for the SFT v2 training-only corpus and dev split.

Phase 2 of the SFT v2 plan expands training data with (a) template diversity in
the eight seen families and (b) new training-only families. This checker asserts
that none of that new data leaks into any evaluation set:

* the frozen v1 seen-family eval (``eval_speedup_v1_seen_family.jsonl``),
* the frozen v1 family-held-out eval (``eval_speedup_v1_family_heldout.jsonl``),
* the new v2 behavioral dev split (``eval_speedup_v2_dev_behavioral.jsonl``).

For every (training source, eval set) pair it requires:

* zero ``template_id`` overlap,
* zero ``normalized_ast_sha256`` overlap,
* zero record ``id`` overlap.

It additionally asserts that the new v2 families are structurally distinct from
the three FROZEN held-out families (``string-concatenation``, ``sort-selection``,
``indexed-lookup``): zero ``family_id`` overlap. The v2 dev split is itself
checked for disjointness against the frozen v1 evals and against all training
data - it may be used for checkpoint selection, so it must never intersect the
frozen measurement or memorize a training template.

Run it and commit the JSON report as evidence; ``main`` exits non-zero if any
invariant fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FROZEN_HELDOUT_FAMILIES = ("string-concatenation", "sort-selection", "indexed-lookup")
OVERLAP_KEYS = ("id", "template_id", "normalized_ast_sha256")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _values(rows: list[dict], key: str) -> set:
    return {row[key] for row in rows}


def _pair_overlap(source_rows: list[dict], eval_rows: list[dict]) -> dict:
    """Return the overlap counts and offending values for one source/eval pair."""
    report = {}
    for key in OVERLAP_KEYS:
        shared = sorted(_values(source_rows, key) & _values(eval_rows, key))
        report[key] = {"overlap_count": len(shared), "overlap_values": shared[:20]}
    return report


def audit(*, train_rows: list[dict], new_family_rows: list[dict],
          eval_sets: dict[str, list[dict]],
          dev_rows: list[dict] | None = None,
          sft_case_ids: set | None = None) -> dict:
    """Compute the full leakage report. ``train_rows`` is all v2 training data.

    ``new_family_rows`` is the subset belonging to the new training-only families
    (used for the family_id exclusion check). ``eval_sets`` maps a name to its
    rows; every set is checked against ``train_rows``. If ``dev_rows`` is given,
    the dev split is also checked against training data. If ``sft_case_ids`` is
    given, the FINAL SFT v2 corpus case_ids are asserted disjoint from every
    frozen v1 eval split AND the v2 dev split (by record ``id``) — the corpus is
    what the student actually trains on, so this is the airtight final gate.
    """
    checks: list[dict] = []
    ok = True

    for eval_name, eval_rows in eval_sets.items():
        pair = _pair_overlap(train_rows, eval_rows)
        clean = all(pair[key]["overlap_count"] == 0 for key in OVERLAP_KEYS)
        ok = ok and clean
        checks.append({
            "source": "train_v2",
            "against": eval_name,
            "clean": clean,
            "overlap": pair,
        })

    # Family exclusion: new families must not touch the frozen held-out families.
    new_families = sorted(_values(new_family_rows, "family_id")) if new_family_rows else []
    forbidden = sorted(set(new_families) & set(FROZEN_HELDOUT_FAMILIES))
    family_clean = len(forbidden) == 0
    ok = ok and family_clean
    checks.append({
        "source": "train_v2_new_families",
        "against": "frozen_heldout_families",
        "clean": family_clean,
        "new_family_ids": new_families,
        "frozen_heldout_family_ids": list(FROZEN_HELDOUT_FAMILIES),
        "family_id_overlap": forbidden,
    })

    if dev_rows is not None:
        # The dev split must be disjoint from the two frozen evals ...
        for eval_name, eval_rows in eval_sets.items():
            if eval_name == "v2_dev_behavioral":
                continue
            pair = _pair_overlap(dev_rows, eval_rows)
            clean = all(pair[key]["overlap_count"] == 0 for key in OVERLAP_KEYS)
            ok = ok and clean
            checks.append({
                "source": "v2_dev_behavioral",
                "against": eval_name,
                "clean": clean,
                "overlap": pair,
            })
        # ... and from every training row (no memorized training template).
        pair = _pair_overlap(dev_rows, train_rows)
        clean = all(pair[key]["overlap_count"] == 0 for key in OVERLAP_KEYS)
        ok = ok and clean
        checks.append({
            "source": "v2_dev_behavioral",
            "against": "train_v2",
            "clean": clean,
            "overlap": pair,
        })

    if sft_case_ids is not None:
        # Final gate: the SFT v2 corpus case_ids must not intersect any frozen
        # v1 eval split or the v2 dev split (by record id). This is the set the
        # student is actually trained on.
        for eval_name, eval_rows in eval_sets.items():
            eval_ids = _values(eval_rows, "id")
            shared = sorted(sft_case_ids & eval_ids)
            clean = len(shared) == 0
            ok = ok and clean
            checks.append({
                "source": "sft_speedup_v2",
                "against": eval_name,
                "clean": clean,
                "id_overlap_count": len(shared),
                "id_overlap_values": shared[:20],
            })

    return {
        "schema_version": 1,
        "all_clean": ok,
        "overlap_keys_checked": list(OVERLAP_KEYS),
        "counts": {
            "train_v2": len(train_rows),
            "train_v2_new_families": len(new_family_rows),
            "v2_dev_behavioral": len(dev_rows) if dev_rows is not None else 0,
            "sft_speedup_v2": len(sft_case_ids) if sft_case_ids is not None else 0,
            **{name: len(rows) for name, rows in eval_sets.items()},
        },
        "checks": checks,
    }


def build_report(datasets_dir: Path, sft_path: Path | None = None) -> dict:
    """Load the committed v2 corpus + dev split and every eval set, then audit.

    New-family rows are identified by ``family_id`` membership in the v2 bank's
    ``NEW_FAMILIES`` rather than by a brittle id prefix. If ``sft_path`` is given,
    the final SFT v2 corpus case_ids are also asserted disjoint from every eval
    split.
    """
    from datagen import _v2_bank as bank

    train_rows = load_jsonl(datasets_dir / "corpus_speedup_v2.verified.jsonl")
    dev_rows = load_jsonl(datasets_dir / "eval_speedup_v2_dev_behavioral.jsonl")
    new_family_rows = [row for row in train_rows if row["family_id"] in bank.NEW_FAMILIES]
    eval_sets = {
        "v1_seen_family": load_jsonl(datasets_dir / "eval_speedup_v1_seen_family.jsonl"),
        "v1_family_heldout": load_jsonl(datasets_dir / "eval_speedup_v1_family_heldout.jsonl"),
        "v2_dev_behavioral": dev_rows,
    }
    sft_case_ids = None
    if sft_path is not None:
        sft_rows = load_jsonl(sft_path)
        sft_case_ids = {row["meta"]["case_id"] for row in sft_rows if "meta" in row}
    return audit(train_rows=train_rows, new_family_rows=new_family_rows,
                 eval_sets=eval_sets, dev_rows=dev_rows, sft_case_ids=sft_case_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sft", type=Path, default=None,
                        help="final SFT v2 corpus JSONL to check case_id disjointness")
    args = parser.parse_args()
    report = build_report(args.datasets_dir, sft_path=args.sft)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_clean": report["all_clean"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
