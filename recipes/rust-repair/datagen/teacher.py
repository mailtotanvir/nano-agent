#!/usr/bin/env python3
"""Teacher trajectory pipeline.

Runs the frontier teacher (Gemini) through the SAME Rust controller the tiny
model will use, so teacher trajectories have identical shape to tiny-model runs
(spec section 8). Every step is compiler-verified by the controller itself; we
only keep episodes the teacher actually fixed (Outcome::Success) plus their full
step history (including any failed-then-recovered intermediate attempts, which are
valuable multi-turn recovery data).

Eval cases are EXCLUDED by case_id to guarantee no train/test leakage.

For each case we:
  1. materialize the broken crate in a temp dir,
  2. invoke the `rust-repair` binary with --backend gemini,
  3. read back the appended trajectory line,
  4. keep it if outcome == success.

Usage:
    GEMINI_API_KEY=... python teacher.py \
        --train-pool datasets/train_pool_v0.jsonl \
        --binary ../../../target/release/rust-repair \
        --model gemini-3.6-flash \
        --out datasets/teacher_trajectories_v0.jsonl \
        --max-cases 200 --max-attempts 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CARGO_TOML = (
    '[package]\nname = "case"\nversion = "0.1.0"\nedition = "2021"\n\n'
    '[[bin]]\nname = "case"\npath = "src/main.rs"\n'
)


def materialize(broken: str, root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(CARGO_TOML)
    (root / "src" / "main.rs").write_text(broken)


def run_case(binary: Path, case: dict, model: str, max_attempts: int, workdir: Path,
             backend: str = "vertex", gcp_project: str = os.environ.get("GCP_PROJECT", "your-gcp-project"),
             gcp_location: str = "global") -> dict | None:
    """Run one teacher episode; return the trajectory dict or None on hard error."""
    crate = workdir / case["case_id"]
    materialize(case["broken"], crate)
    log = crate / "traj.jsonl"
    env = dict(os.environ)
    cmd = [
        str(binary), "--path", str(crate), "--backend", backend,
        "--model", model, "--max-attempts", str(max_attempts),
        "--log", str(log), "--case-id", case["case_id"],
    ]
    if backend == "vertex":
        cmd += ["--gcp-project", gcp_project, "--gcp-location", gcp_location]
    try:
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if not log.exists():
        return None
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    traj = json.loads(lines[-1])
    # annotate with the ground-truth label from the corpus
    traj["intended_code"] = case.get("intended_code")
    traj["category"] = case.get("category")
    return traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pool", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--exclude-eval", default=None,
                    help="manifest.json whose eval_case_ids will be skipped")
    ap.add_argument("--backend", default="vertex", choices=["vertex", "gemini", "azure"],
                    help="vertex = Vertex AI via ADC (billed to GCP project); "
                         "gemini = AI Studio API key (free-tier limited); "
                         "azure = Azure OpenAI-compatible endpoint")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel independent cases (recommended: 12 for Azure reasoning models)")
    ap.add_argument("--gcp-project", default=os.environ.get("GCP_PROJECT", "your-gcp-project"))
    ap.add_argument("--gcp-location", default="global")
    args = ap.parse_args()

    if args.backend == "gemini" and not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GEMINI_API_KEY / GOOGLE_API_KEY not set for gemini backend")
    if args.backend == "azure":
        bashrc = Path("~/.bashrc").expanduser()
        for name in ("AZURE_API_BASE", "AZURE_API_KEY"):
            if os.environ.get(name):
                continue
            match = re.search(rf'^\s*(?:export\s+)?{name}=["\']?([^"\'\n]+)',
                              bashrc.read_text(), re.MULTILINE) if bashrc.exists() else None
            if match:
                os.environ[name] = match.group(1).strip()
        missing = [name for name in ("AZURE_API_BASE", "AZURE_API_KEY") if not os.environ.get(name)]
        if missing:
            raise SystemExit(f"Azure credentials unavailable: {', '.join(missing)}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    binary = Path(args.binary).resolve()
    if not binary.exists():
        raise SystemExit(f"binary not found: {binary}")

    cases = [json.loads(l) for l in Path(args.train_pool).read_text().splitlines() if l.strip()]

    excluded: set[str] = set()
    if args.exclude_eval:
        manifest = json.loads(Path(args.exclude_eval).read_text())
        excluded = set(manifest.get("eval_case_ids", []))
    cases = [c for c in cases if c["case_id"] not in excluded]
    if args.max_cases:
        cases = cases[: args.max_cases]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    stats = {"success": 0, "escalate": 0, "no_fix": 0, "other": 0, "hard_error": 0}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="teacher_") as td:
        workdir = Path(td)

        def run(case: dict) -> dict | None:
            return run_case(binary, case, args.model, args.max_attempts, workdir,
                            backend=args.backend, gcp_project=args.gcp_project,
                            gcp_location=args.gcp_location)

        if args.workers == 1:
            trajectories = [run(case) for case in cases]
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                trajectories = list(executor.map(run, cases))

        with out.open("w") as f:
            for i, traj in enumerate(trajectories, 1):
                if traj is None:
                    stats["hard_error"] += 1
                else:
                    outcome = traj.get("outcome", "other")
                    stats[outcome if outcome in stats else "other"] += 1
                    # Keep successful episodes for SFT positives.
                    if outcome == "success":
                        f.write(json.dumps(traj) + "\n")
                        kept += 1
                if i % 10 == 0 or i == len(cases):
                    print(f"[{i}/{len(cases)}] kept={kept} stats={stats} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"done: {len(cases)} cases, kept {kept} success trajectories -> {out}")
    print("outcome stats:", json.dumps(stats))
    solve_rate = stats["success"] / max(1, len(cases))
    print(f"teacher solve rate: {solve_rate:.1%}")


if __name__ == "__main__":
    main()
