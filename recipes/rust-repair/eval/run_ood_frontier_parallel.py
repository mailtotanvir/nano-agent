#!/usr/bin/env python3
"""Run the 3 frontier models on the OOD set IN PARALLEL via the rust-repair
Azure backend. Each (model, case) is an independent subprocess; we fan out with a
thread pool so slow reasoning models don't serialize. Writes one JSON per model.
"""
import json, os, re, subprocess, tempfile, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BIN = os.environ.get("RUST_REPAIR_BIN") or str(Path(__file__).resolve().parents[3] / "target/release/rust-repair")
EVAL = str(Path(__file__).resolve().parents[1] / "datasets/eval_ood_v1.jsonl")
OUT = str(Path(__file__).resolve().parents[1] / "eval/results/ood_frontier.json")
MODELS = ["grok-4.6", "DeepSeek-V4-Flash", "gpt-5.6-luna"]
CARGO_TOML = '[package]\nname="c"\nversion="0.1.0"\nedition="2021"\n\n[[bin]]\nname="c"\npath="src/main.rs"\n'

# load azure creds
txt = Path("~/.bashrc").expanduser().read_text()
for v in ("AZURE_API_BASE", "AZURE_API_KEY"):
    os.environ[v] = re.search(rf'{v}="([^"]+)"', txt).group(1)

cases = [json.loads(l) for l in Path(EVAL).read_text().splitlines() if l.strip()]

def run_one(model, case):
    d = Path(tempfile.mkdtemp(prefix="oodf_"))
    try:
        (d/"src").mkdir()
        (d/"Cargo.toml").write_text(CARGO_TOML)
        (d/"src/main.rs").write_text(case["broken"])
        log = d/"traj.jsonl"
        cmd = [BIN, "--path", str(d), "--backend", "azure", "--model", model,
               "--max-attempts", "1", "--log", str(log), "--case-id", case["case_id"]]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return model, case["case_id"], case.get("category"), False, None
        traj = None
        if log.exists():
            lines = [l for l in log.read_text().splitlines() if l.strip()]
            if lines: traj = json.loads(lines[-1])
        ok = bool(traj and traj.get("outcome") == "success")
        lat = (traj or {}).get("latency_ms")
        return model, case["case_id"], case.get("category"), ok, lat
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)

tasks = [(m, c) for m in MODELS for c in cases]
results = {m: [] for m in MODELS}
t0 = time.time()
done = 0
# 12-way concurrency: API-bound, safe
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = [ex.submit(run_one, m, c) for m, c in tasks]
    for f in as_completed(futs):
        model, cid, cat, ok, lat = f.result()
        results[model].append({"case_id": cid, "category": cat, "solved": ok, "latency_ms": lat})
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)

summary = {}
for m in MODELS:
    solved = sum(r["solved"] for r in results[m])
    summary[m] = {"solved": solved, "n": len(cases), "rate": round(solved/len(cases), 4)}

Path(OUT).write_text(json.dumps({"summary": summary, "per_case": results}, indent=2))
print("\n=== FRONTIER OOD ===")
for m in MODELS:
    s = summary[m]; print(f"  {m:20} {s['solved']:>2}/{s['n']}  {s['rate']*100:5.1f}%")
print("wrote", OUT, f"({time.time()-t0:.0f}s)")
