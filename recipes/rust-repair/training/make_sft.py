#!/usr/bin/env python3
"""Convert verified teacher trajectories into SFT training examples.

Each trajectory step becomes one (prompt -> completion) pair in chat format:
  - system: the repair-agent contract
  - user:   the step's context (diagnostic + code window + prior attempts)
  - assistant: the exact JSON proposal the teacher emitted (which the compiler
    then verified as progress)

Only steps from SUCCESS trajectories are used. Multi-step trajectories yield
multiple turns, teaching recovery. The assistant target is the serialized JSON
proposal, matching the contract the tiny model must learn.

Output: JSONL with {"messages": [...]} per line (TRL/axolotl SFT format).

Usage:
  python make_sft.py --trajectories ../datasets/teacher_trajectories_v1.jsonl \
      --out ../datasets/sft_v1.jsonl --system-from-controller
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Mirror of nano_controller::DEFAULT_SYSTEM_PROMPT so training and inference agree.
SYSTEM_PROMPT = (
    "You are a Rust compiler-error repair agent. You are given one compiler error "
    "and the relevant source. Respond with ONLY a single JSON object matching this "
    'contract: {"action": "patch"|"no_fix"|"escalate", "patch": "<SEARCH/REPLACE '
    'blocks or null>", "reason": "<short>", "confidence": <0.0-1.0>}. For a patch, '
    'the "patch" field must contain one or more blocks in this exact format:\n'
    "file: <path>\n<<<<<<< SEARCH\n<exact existing lines>\n=======\n"
    "<replacement lines>\n>>>>>>> REPLACE\nMake the smallest change that fixes the "
    'error. If you cannot fix it, use action "no_fix" or "escalate".'
)


def proposal_json(step: dict) -> str:
    p = step["proposal"]
    obj = {
        "action": p["action"],
        "patch": p.get("patch"),
        "reason": p.get("reason") or "",
        "confidence": p.get("confidence", 0.0),
    }
    return json.dumps(obj, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-unapplied", action="store_true",
                    help="include steps whose patch did NOT apply or was reverted "
                         "(default: excluded, since they teach wrong/failed formats)")
    args = ap.parse_args()

    only_applied = not args.include_unapplied

    trajs = [json.loads(l) for l in Path(args.trajectories).read_text().splitlines() if l.strip()]
    examples = []
    kept_multi = 0
    for t in trajs:
        if t.get("outcome") != "success":
            continue
        steps = t["steps"]
        if len(steps) > 1:
            kept_multi += 1
        for step in steps:
            if only_applied and (not step.get("patch_applied") or step.get("reverted")):
                continue
            if step["proposal"]["action"] != "patch":
                continue
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": step["context"]},
                    {"role": "assistant", "content": proposal_json(step)},
                ],
                "meta": {
                    "case_id": t["case_id"],
                    "category": t.get("category"),
                    "intended_code": t.get("intended_code"),
                    "attempt": step["attempt"],
                },
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"trajectories(success)={sum(1 for t in trajs if t.get('outcome')=='success')} "
          f"multi_step={kept_multi} -> {len(examples)} SFT examples")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
