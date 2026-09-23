#!/usr/bin/env python3
"""Verifier-gate one v6 ordinal in a fresh pinned container process.

Used only to recover from an ``*_unstable`` Cachegrind result in a long-lived
shard process.  It calls the same trusted generator and ``verify_pairs`` gate;
the operational distinction is a fresh container/Python process, not a weaker
acceptance criterion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datagen.gen_problems import generate_v6_pairs, verify_pairs, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    pairs = generate_v6_pairs()
    if not 0 <= args.ordinal < len(pairs):
        parser.error(f"ordinal must be in [0, {len(pairs)})")
    verified = verify_pairs([pairs[args.ordinal]], checkpoint_path=args.checkpoint)
    write_jsonl(args.out, verified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
