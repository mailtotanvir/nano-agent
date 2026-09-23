#!/usr/bin/env python3
"""Build the SFT v2 corpus from verified v2 teacher trajectories + v1 examples.

Composition (Phase 2.5 of the SFT v2 plan):

* **v2 single-turn examples** — one per accepted, applied, non-reverted patch
  step of every kept v2 success trajectory. Legacy unified diffs are converted
  mechanically to the canonical SEARCH/REPLACE contract without changing the
  teacher's edit.
* **v1 examples** — the frozen 115 ``sft_speedup_v1.jsonl`` rows, carried over,
  deduplicated by ``case_id`` against the v2 set (v1 ids are ``cs-v1-*`` and v2
  ``cs-v2-*``; the dedup is belt-and-suspenders). Their system message is
  replaced with the same checked-in prompt used by the production verifier.
* **repair traces** — for every kept v2 success whose teacher needed a retry
  (a reverted attempt before the accepted one, i.e. ``attempt >= 1`` on the
  accepted step), the FULL multi-turn conversation: failed proposal -> the
  public controller feedback the retry actually consumed -> corrected proposal.
  The controller feedback loop guarantees each step's recorded ``context`` is
  exactly the user message that attempt was shown, so the trace is faithful.

HARD RULE honored by construction: every user ``context`` is model-facing text
that was literally shown to the teacher (no private benchmark/edge inputs, no
known-fast reference). Assistant edits are semantically identical mechanical
normalizations of the teacher proposals; provenance records that transformation.
Nothing private is copied here.

Writes ``sft_speedup_v2.jsonl`` and a provenance JSON mirroring v1's pattern,
extended with per-source / repair-trace / teacher-model breakdowns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from patch_contract import canonicalize_patch, load_system_prompt, reconstruct_candidate

SYSTEM_PROMPT = load_system_prompt()
LEGACY_TASK_SUFFIX = "\nRewrite candidate.py to be correct and use fewer instruction references."
TASK_SUFFIX = (
    "\nReturn one JSON proposal with action \"patch\" and exact SEARCH/REPLACE blocks. "
    "Rewrite candidate.py to be correct and use fewer instruction references."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _patch_format(patch: str | None) -> str:
    text = patch or ""
    if "<<<<<<< SEARCH" in text:
        return "search_replace"
    if "@@" in text:
        return "unified_diff"
    return "other"


def _candidate_from_context(context: str) -> str:
    marker = "--- candidate.py ---\n"
    suffix = LEGACY_TASK_SUFFIX if context.endswith(LEGACY_TASK_SUFFIX) else TASK_SUFFIX
    if marker not in context or not context.endswith(suffix):
        raise ValueError("teacher context does not contain the candidate.py boundary")
    return context.split(marker, 1)[1][:-len(suffix)]


def canonical_user_context(context: str) -> str:
    """Replace the old ambiguous task cue without changing model-visible code."""
    if context.endswith(TASK_SUFFIX):
        return context
    if not context.endswith(LEGACY_TASK_SUFFIX):
        raise ValueError("teacher context does not end with the expected task suffix")
    return context[:-len(LEGACY_TASK_SUFFIX)] + TASK_SUFFIX


def normalized_proposal_json(proposal: dict[str, Any], context: str) -> str:
    original_patch = proposal.get("patch")
    canonical_patch = canonicalize_patch(original_patch)
    source = _candidate_from_context(context)
    original_candidate, _, original_applied = reconstruct_candidate(source, original_patch)
    canonical_candidate, _, canonical_applied = reconstruct_candidate(source, canonical_patch)
    if not original_applied or not canonical_applied or canonical_candidate != original_candidate:
        raise ValueError("patch normalization changed or could not reconstruct the candidate")
    return json.dumps({
        "action": proposal["action"],
        "patch": canonical_patch,
        "reason": proposal.get("reason") or "",
        "confidence": proposal.get("confidence", 0.0),
    }, ensure_ascii=False)


def _accepted_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Applied, non-reverted patch steps — the v1 make_sft selection rule."""
    return [
        step for step in trajectory["steps"]
        if step["proposal"].get("action") == "patch"
        and step.get("patch_applied") and not step.get("reverted")
    ]


def build_v2_examples(trajectories: list[dict[str, Any]]) -> tuple[list, list]:
    """Return (single_turn_examples, repair_trace_examples) for kept v2 successes."""
    single: list[dict[str, Any]] = []
    repair: list[dict[str, Any]] = []
    for record in trajectories:
        trajectory = record["trajectory"]
        if (trajectory.get("outcome") != "success"
                or not record.get("verifier", {}).get("passed")):
            continue
        model = record["teacher"]["model"]
        reward = record["verifier"]["reward"]
        accepted = _accepted_steps(trajectory)
        for step in accepted:
            single.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": canonical_user_context(step["context"])},
                    {"role": "assistant", "content": normalized_proposal_json(
                        step["proposal"], step["context"])},
                ],
                "meta": {
                    "case_id": record["case_id"],
                    "teacher_model": model,
                    "attempt": step["attempt"],
                    "reward": reward,
                    "source": "teacher_v2",
                    "kind": "single_turn",
                },
            })
        # Repair trace: the accepted step arrived on attempt >= 1 AND a prior
        # step was reverted (a genuine failed proposal -> feedback -> fix).
        steps = trajectory["steps"]
        had_revert = any(s.get("reverted") for s in steps)
        final_attempt = accepted[-1]["attempt"] if accepted else 0
        if had_revert and final_attempt >= 1:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for step in steps:
                messages.append({"role": "user", "content": canonical_user_context(step["context"])})
                messages.append({"role": "assistant", "content": normalized_proposal_json(
                    step["proposal"], step["context"])})
            repair.append({
                "messages": messages,
                "meta": {
                    "case_id": record["case_id"],
                    "teacher_model": model,
                    "attempts": final_attempt + 1,
                    "reward": reward,
                    "source": "teacher_v2",
                    "kind": "repair_trace",
                },
            })
    return single, repair


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True,
                        help="kept v2 teacher trajectory JSONL")
    parser.add_argument("--v1-sft", type=Path, required=True,
                        help="frozen sft_speedup_v1.jsonl to carry over")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--corpus", type=Path,
                        help="verified training-only corpus used only to map case IDs to families")
    parser.add_argument("--focus-family", action="append", default=[],
                        help="training-only family whose accepted teacher examples are repeated")
    parser.add_argument("--focus-weight", type=int, default=1,
                        help="total copies for each focused teacher example (default: 1)")
    args = parser.parse_args()

    trajectories = _load_jsonl(args.trajectories)
    v1_examples = _load_jsonl(args.v1_sft)
    if args.focus_weight < 1:
        raise ValueError("--focus-weight must be at least 1")
    if args.focus_family and args.corpus is None:
        raise ValueError("--focus-family requires --corpus")
    family_by_case = {
        row["id"]: row["family_id"] for row in _load_jsonl(args.corpus)
    } if args.corpus else {}
    focus_families = set(args.focus_family)
    unknown_families = focus_families - set(family_by_case.values())
    if unknown_families:
        raise ValueError(f"focused families not present in corpus: {sorted(unknown_families)}")

    input_patch_formats = Counter()
    for record in trajectories:
        for step in _accepted_steps(record["trajectory"]):
            input_patch_formats[_patch_format(step["proposal"].get("patch"))] += 1
    for example in v1_examples:
        proposal = json.loads(example["messages"][-1]["content"])
        input_patch_formats[_patch_format(proposal.get("patch"))] += 1

    single, repair = build_v2_examples(trajectories)
    v2_case_ids = {ex["meta"]["case_id"] for ex in single}

    # Dedup the v1 carry-over by case_id against the v2 set.
    v1_kept: list[dict[str, Any]] = []
    v1_dropped = 0
    for ex in v1_examples:
        case_id = ex.get("meta", {}).get("case_id")
        if case_id in v2_case_ids:
            v1_dropped += 1
            continue
        ex = {**ex, "messages": [dict(message) for message in ex["messages"]]}
        ex["messages"][0]["content"] = SYSTEM_PROMPT
        assistant = json.loads(ex["messages"][-1]["content"])
        assistant_json = normalized_proposal_json(assistant, ex["messages"][-2]["content"])
        ex["messages"][-1]["content"] = assistant_json
        ex["messages"][-2]["content"] = canonical_user_context(ex["messages"][-2]["content"])
        ex["meta"] = {**ex.get("meta", {}), "source": "sft_v1"}
        v1_kept.append(ex)

    focused_copies: list[dict[str, Any]] = []
    if focus_families and args.focus_weight > 1:
        for example in single:
            family = family_by_case.get(example["meta"]["case_id"])
            if family not in focus_families:
                continue
            for copy_index in range(1, args.focus_weight):
                focused_copies.append({
                    "messages": [dict(message) for message in example["messages"]],
                    "meta": {**example["meta"], "source": "teacher_v2_focused",
                             "family_id": family, "focus_copy": copy_index},
                })

    examples = single + focused_copies + repair + v1_kept

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")

    source_counts = Counter(ex["meta"].get("source") for ex in examples)
    kind_counts = Counter(ex["meta"].get("kind", "single_turn") for ex in examples)
    teacher_model_counts = Counter(
        ex["meta"].get("teacher_model") for ex in examples
        if ex["meta"].get("source") == "teacher_v2")
    # Per-family from the v2 corpus family_id, keyed by case_id.
    kept_successes = sum(
        1 for r in trajectories
        if r["trajectory"].get("outcome") == "success" and r.get("verifier", {}).get("passed"))

    provenance = {
        "schema_version": 2,
        "sources": {
            "teacher_v2": {
                "path": str(args.trajectories),
                "sha256": _sha256(args.trajectories),
                "kept_success_trajectories": kept_successes,
            },
            "sft_v1": {
                "path": str(args.v1_sft),
                "sha256": _sha256(args.v1_sft),
                "examples": len(v1_examples),
            },
        },
        "sft_examples": len(examples),
        "counts_by_source": dict(source_counts),
        "counts_by_kind": dict(kind_counts),
        "v2_single_turn": len(single),
        "focused_teacher_copies": len(focused_copies),
        "focus_families": sorted(focus_families),
        "focus_weight": args.focus_weight,
        "repair_traces": len(repair),
        "v1_carried_over": len(v1_kept),
        "v1_dropped_by_dedup": v1_dropped,
        "teacher_model_breakdown": dict(teacher_model_counts),
        "patch_contract": "search_replace_v1",
        "canonicalized_examples": len(examples),
        "input_patch_formats": dict(sorted(input_patch_formats.items())),
        "output_patch_formats": {"search_replace": len(examples)},
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "output": str(args.out),
        "output_sha256": _sha256(args.out),
    }
    provenance_path = args.provenance or args.out.with_suffix(args.out.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(json.dumps({
        "sft_examples": len(examples),
        "v2_single_turn": len(single),
        "repair_traces": len(repair),
        "v1_carried_over": len(v1_kept),
        "teacher_model_breakdown": dict(teacher_model_counts),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
