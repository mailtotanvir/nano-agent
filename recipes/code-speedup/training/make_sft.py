#!/usr/bin/env python3
"""Convert successful, verifier-gated code-speed trajectories to chat SFT JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datagen.build_sft_v2 import canonical_user_context, normalized_proposal_json
from patch_contract import load_system_prompt

SYSTEM_PROMPT = load_system_prompt()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()

    trajectories = [json.loads(line) for line in args.trajectories.read_text().splitlines()
                    if line.strip()]
    examples = []
    for record in trajectories:
        trajectory = record["trajectory"]
        if trajectory.get("outcome") != "success" or not record.get("verifier", {}).get("passed"):
            continue
        for step in trajectory["steps"]:
            if (step["proposal"].get("action") != "patch"
                    or not step.get("patch_applied") or step.get("reverted")):
                continue
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": canonical_user_context(step["context"])},
                    {"role": "assistant", "content": normalized_proposal_json(
                        step["proposal"], step["context"])},
                ],
                "meta": {
                    "case_id": record["case_id"],
                    "teacher_model": record["teacher"]["model"],
                    "attempt": step["attempt"],
                    "reward": record["verifier"]["reward"],
                },
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    provenance = args.provenance or args.out.with_suffix(args.out.suffix + ".provenance.json")
    provenance.write_text(json.dumps({
        "schema_version": 1,
        "source": str(args.trajectories),
        "source_sha256": _sha256(args.trajectories),
        "successful_trajectories": sum(
            t["trajectory"].get("outcome") == "success" for t in trajectories),
        "sft_examples": len(examples),
        "output_sha256": _sha256(args.out),
    }, indent=2) + "\n")
    print(f"successful_trajectories={len(trajectories)} sft_examples={len(examples)}")


if __name__ == "__main__":
    main()
