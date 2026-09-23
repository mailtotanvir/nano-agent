#!/usr/bin/env python3
"""Verify one deterministic shard of the v6 trusted corpus.

The known-fast rewrite remains in-process and is never serialized.  Sharding is
only an operational accelerator for the authoritative CPU gate: each shard has
its own append-only checkpoint and public JSONL output, which can be combined
only after every shard exits successfully.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from harness.perf import evaluate

from datagen.gen_problems import generate_v6_pairs, verify_pairs, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verification-seed", type=int, default=314159)
    parser.add_argument("--unstable-retries", type=int, default=3,
                        help="retry only an exact-repeatability failure; never accept it")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count)")
    if args.unstable_retries < 0:
        parser.error("unstable retries must be nonnegative")

    pairs = generate_v6_pairs()
    shard = [pair for ordinal, pair in enumerate(pairs)
             if ordinal % args.shard_count == args.shard_index]

    def report(index: int, total: int, ident: str, state: str) -> None:
        print(f"[{args.shard_index}:{index}/{total}] {state}: {ident}", flush=True)

    def stable_evaluator(problem, candidate, *, seed, random_cases, repeats):
        """Require an exactly repeatable result, tolerating profiler jitter only.

        A retry is permitted solely for the verifier's ``*_unstable`` status.
        Every accepted result still contains a fresh pair of identical
        Cachegrind measurements; semantic, sandbox, timeout, and no-speedup
        failures return immediately and remain gate failures.
        """
        result = evaluate(problem, candidate, seed=seed, random_cases=random_cases,
                          repeats=repeats)
        for _ in range(args.unstable_retries):
            if result.status not in {"reference_unstable", "candidate_unstable"}:
                break
            result = evaluate(problem, candidate, seed=seed, random_cases=random_cases,
                              repeats=repeats)
        return result

    verified = verify_pairs(shard, evaluator=stable_evaluator, checkpoint_path=args.checkpoint,
                            seed=args.verification_seed, progress=report)
    write_jsonl(args.out, verified)
    print(f"shard {args.shard_index}/{args.shard_count}: {len(verified)} verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
