#!/usr/bin/env python3
"""4-arm evaluation harness for rust-repair (spec section 9).

Runs the frozen eval set through up to four arms and reports repair rate,
attempts, and latency, broken down PER ERROR CODE so the competence envelope is
an explicit table.

Arms:
  1. frontier_direct : one-shot frontier (no loop) -- upper bound
  2. tiny_direct     : one-shot tiny model (no loop, max_attempts=1)
  3. tiny_loop       : tiny model + compiler loop (THE recipe; max_attempts=N)
  4. recipe_escalate : tiny loop, then frontier fallback on failure

Each arm runs the `rust-repair` binary per case in an isolated temp crate and
reads back the trajectory. The verifier (cargo) decides success, never the model.

Usage:
  python run_eval.py --eval ../datasets/eval_v1.jsonl \
      --binary ../../../target/release/rust-repair \
      --arms tiny_loop,frontier_direct \
      --tiny-model qwen2.5-coder-0.5b-q8_0 --llama-url http://127.0.0.1:8080 \
      --frontier-model gemini-3.6-flash \
      --out results/eval_v1_qwen05.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

CARGO_TOML = (
    '[package]\nname = "case"\nversion = "0.1.0"\nedition = "2021"\n\n'
    '[[bin]]\nname = "case"\npath = "src/main.rs"\n'
)


def materialize(broken: str, root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(CARGO_TOML)
    (root / "src" / "main.rs").write_text(broken)


def run_arm(binary: Path, case: dict, arm: str, cfg: dict, workdir: Path) -> dict | None:
    """Run one case under one arm; return the trajectory dict (or None)."""
    crate = workdir / f"{arm}_{case['case_id']}"
    materialize(case["broken"], crate)
    log = crate / "traj.jsonl"

    if arm == "frontier_direct":
        backend, model, attempts = cfg["frontier_backend"], cfg["frontier_model"], 1
    elif arm == "tiny_direct":
        backend, model, attempts = "llama", cfg["tiny_model"], 1
    elif arm == "tiny_loop":
        backend, model, attempts = "llama", cfg["tiny_model"], cfg["max_attempts"]
    elif arm == "recipe_escalate":
        # handled specially below (tiny loop then frontier)
        backend, model, attempts = "llama", cfg["tiny_model"], cfg["max_attempts"]
    else:
        raise ValueError(arm)

    cmd = [str(binary), "--path", str(crate), "--backend", backend, "--model", model,
           "--max-attempts", str(attempts), "--log", str(log), "--case-id", case["case_id"]]
    if backend == "llama":
        cmd += ["--llama-url", cfg["llama_url"]]
    if backend == "vertex":
        cmd += ["--gcp-project", cfg["gcp_project"], "--gcp-location", cfg["gcp_location"]]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    traj = _last_traj(log)

    # recipe_escalate: if tiny loop failed, run frontier on the (reset) case.
    if arm == "recipe_escalate" and traj and traj.get("outcome") != "success":
        crate2 = workdir / f"{arm}_esc_{case['case_id']}"
        materialize(case["broken"], crate2)
        log2 = crate2 / "traj.jsonl"
        cmd2 = [str(binary), "--path", str(crate2), "--backend", cfg["frontier_backend"],
                "--model", cfg["frontier_model"], "--max-attempts", str(cfg["max_attempts"]),
                "--log", str(log2), "--case-id", case["case_id"]]
        if cfg["frontier_backend"] == "vertex":
            cmd2 += ["--gcp-project", cfg["gcp_project"], "--gcp-location", cfg["gcp_location"]]
        try:
            subprocess.run(cmd2, capture_output=True, text=True, timeout=300)
            esc = _last_traj(log2)
            if esc:
                esc["escalated"] = True
                # combine attempt counts
                esc["attempts"] = (traj.get("attempts", 0) + esc.get("attempts", 0))
                traj = esc
        except subprocess.TimeoutExpired:
            pass
    return traj


def _last_traj(log: Path) -> dict | None:
    if not log.exists():
        return None
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def evaluate(eval_cases: list[dict], binary: Path, arms: list[str], cfg: dict) -> dict:
    results = {arm: [] for arm in arms}
    with tempfile.TemporaryDirectory(prefix="eval_") as td:
        workdir = Path(td)
        for arm in arms:
            t0 = time.time()
            for i, case in enumerate(eval_cases, 1):
                traj = run_arm(binary, case, arm, cfg, workdir)
                success = bool(traj and traj.get("outcome") == "success")
                results[arm].append({
                    "case_id": case["case_id"],
                    "intended_code": case.get("intended_code"),
                    "category": case.get("category"),
                    "success": success,
                    "attempts": (traj or {}).get("attempts", 0),
                    "latency_ms": (traj or {}).get("total_latency_ms", 0),
                    "escalated": (traj or {}).get("escalated", False),
                })
                if i % 10 == 0:
                    solved = sum(r["success"] for r in results[arm])
                    print(f"[{arm} {i}/{len(eval_cases)}] solved={solved} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)
    return results


def summarize(results: dict) -> dict:
    summary = {}
    for arm, rows in results.items():
        n = len(rows)
        solved = sum(r["success"] for r in rows)
        lat = sorted(r["latency_ms"] for r in rows)
        by_code = defaultdict(lambda: [0, 0])  # code -> [solved, total]
        for r in rows:
            c = r["intended_code"] or "?"
            by_code[c][1] += 1
            by_code[c][0] += int(r["success"])
        envelope = {c: {"solved": s, "total": t, "rate": round(s / t, 3)}
                    for c, (s, t) in sorted(by_code.items())}
        summary[arm] = {
            "n": n,
            "solved": solved,
            "repair_rate": round(solved / n, 3) if n else 0,
            "avg_attempts": round(sum(r["attempts"] for r in rows) / n, 2) if n else 0,
            "median_latency_ms": lat[len(lat) // 2] if lat else 0,
            "p95_latency_ms": lat[int(len(lat) * 0.95)] if lat else 0,
            "escalated": sum(r["escalated"] for r in rows),
            "envelope_by_code": envelope,
        }
    return summary


def print_envelope(summary: dict) -> None:
    print("\n=== competence envelope (repair rate per error code) ===")
    arms = list(summary.keys())
    codes = sorted({c for a in arms for c in summary[a]["envelope_by_code"]})
    header = f"{'code':10}" + "".join(f"{a:>18}" for a in arms)
    print(header)
    for code in codes:
        row = f"{code:10}"
        for a in arms:
            e = summary[a]["envelope_by_code"].get(code)
            row += f"{(str(e['solved'])+'/'+str(e['total'])):>18}" if e else f"{'-':>18}"
        print(row)
    print("\n=== overall ===")
    for a in arms:
        s = summary[a]
        print(f"{a:18} rate={s['repair_rate']:.1%} solved={s['solved']}/{s['n']} "
              f"avg_attempts={s['avg_attempts']} median_lat={s['median_latency_ms']}ms "
              f"escalated={s['escalated']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--arms", default="tiny_loop,frontier_direct")
    ap.add_argument("--tiny-model", default="qwen2.5-coder-0.5b-q8_0")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--frontier-model", default="gemini-3.6-flash")
    ap.add_argument("--frontier-backend", default="vertex", choices=["vertex", "gemini"])
    ap.add_argument("--gcp-project", default=os.environ.get("GCP_PROJECT", "your-gcp-project"))
    ap.add_argument("--gcp-location", default="global")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eval_cases = [json.loads(l) for l in Path(args.eval).read_text().splitlines() if l.strip()]
    if args.max_cases:
        eval_cases = eval_cases[: args.max_cases]
    arms = args.arms.split(",")
    cfg = {
        "tiny_model": args.tiny_model, "llama_url": args.llama_url,
        "frontier_model": args.frontier_model, "frontier_backend": args.frontier_backend,
        "gcp_project": args.gcp_project, "gcp_location": args.gcp_location,
        "max_attempts": args.max_attempts,
    }
    binary = Path(args.binary).resolve()

    results = evaluate(eval_cases, binary, arms, cfg)
    summary = summarize(results)
    print_envelope(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "summary": summary, "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
