#!/usr/bin/env python3
"""OOD eval runner: compare the trained tiny model vs frontier models on the
hand-authored out-of-distribution set (eval_ood_v1.jsonl).

Every model runs through the SAME rust-repair controller (same prompt, same
SEARCH/REPLACE parser, same cargo verifier) so the comparison is apples-to-apples.

Arms:
  - tiny_loop : trained 0.5B via llama.cpp server, up to N attempts (the loop)
  - <frontier>: one-shot (1 attempt) via Azure backend, per model name

Usage:
  python run_ood_eval.py --binary ../../../target/release/rust-repair \
      --eval ../datasets/eval_ood_v1.jsonl \
      --tiny-url http://127.0.0.1:8090 --tiny-model qwen05-sft-v2 \
      --azure-models grok-4.6,DeepSeek-V4-Flash,gpt-5.6-luna \
      --out results/ood_v1.json

Reads AZURE_API_BASE / AZURE_API_KEY from env (or parses ~/.bashrc if absent).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

CARGO_TOML = ('[package]\nname="c"\nversion="0.1.0"\nedition="2021"\n\n'
              '[[bin]]\nname="c"\npath="src/main.rs"\n')


def materialize(broken: str, root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(CARGO_TOML)
    (root / "src" / "main.rs").write_text(broken)


def last_traj(log: Path) -> dict | None:
    if not log.exists():
        return None
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def ensure_azure_env() -> None:
    if os.environ.get("AZURE_API_BASE") and os.environ.get("AZURE_API_KEY"):
        return
    txt = Path("~/.bashrc").expanduser().read_text()
    for var in ("AZURE_API_BASE", "AZURE_API_KEY"):
        m = re.search(rf'{var}="([^"]+)"', txt)
        if m:
            os.environ[var] = m.group(1)


def run_case(binary: str, case: dict, arm: str, cfg: dict, workdir: Path) -> dict | None:
    crate = workdir / f"{arm}_{case['case_id']}"
    materialize(case["broken"], crate)
    log = crate / "traj.jsonl"

    if arm == "tiny_loop":
        cmd = [binary, "--path", str(crate), "--backend", "llama",
               "--model", cfg["tiny_model"], "--max-attempts", str(cfg["max_attempts"]),
               "--llama-url", cfg["tiny_url"], "--log", str(log), "--case-id", case["case_id"]]
    else:  # frontier one-shot via azure; arm == model name
        cmd = [binary, "--path", str(crate), "--backend", "azure",
               "--model", arm, "--max-attempts", "1",
               "--log", str(log), "--case-id", case["case_id"]]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    return last_traj(log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--tiny-url", default="http://127.0.0.1:8090")
    ap.add_argument("--tiny-model", default="qwen05-sft-v2")
    ap.add_argument("--azure-models", default="grok-4.6,DeepSeek-V4-Flash,gpt-5.6-luna")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ensure_azure_env()
    cases = [json.loads(l) for l in Path(args.eval).read_text().splitlines() if l.strip()]
    azure_models = [m.strip() for m in args.azure_models.split(",") if m.strip()]
    arms = ["tiny_loop"] + azure_models
    cfg = {"tiny_model": args.tiny_model, "tiny_url": args.tiny_url,
           "max_attempts": args.max_attempts}

    summary = {}
    per_case = {}
    with tempfile.TemporaryDirectory(prefix="ood_") as td:
        workdir = Path(td)
        for arm in arms:
            t0 = time.time()
            solved = 0
            rows = []
            for i, case in enumerate(cases, 1):
                traj = run_case(args.binary, case, arm, cfg, workdir)
                ok = bool(traj and traj.get("outcome") == "success")
                solved += ok
                rows.append({
                    "case_id": case["case_id"], "category": case.get("category"),
                    "solved": ok,
                    "attempts": (traj or {}).get("attempts"),
                    "latency_ms": (traj or {}).get("latency_ms"),
                })
                print(f"[{arm} {i}/{len(cases)}] {case['case_id']:22} "
                      f"{'OK ' if ok else '.  '} solved={solved}")
            n = len(cases)
            summary[arm] = {
                "solved": solved, "n": n, "rate": round(solved / n, 4),
                "elapsed_s": round(time.time() - t0, 1),
            }
            per_case[arm] = rows
            print(f"=== {arm}: {solved}/{n} = {summary[arm]['rate']*100:.1f}% "
                  f"({summary[arm]['elapsed_s']}s) ===\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_case": per_case,
                               "eval": args.eval, "n_cases": len(cases)}, indent=2))
    print("\n=== OOD SUMMARY ===")
    for arm in arms:
        s = summary[arm]
        print(f"  {arm:22} {s['solved']:>2}/{s['n']}  {s['rate']*100:5.1f}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
