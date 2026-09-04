#!/usr/bin/env python3
"""Get Gemini Flash's score on the frozen OOD eval set via the rust-repair
Vertex backend (one-shot frontier). This is the BAR THE NANO MODEL MUST BEAT
(the teacher's own OOD accuracy). API inference only, no new cloud resources.
Runs cases in parallel through the same controller/parser/verifier.
"""
import json, os, subprocess, tempfile, time, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BIN = os.environ.get("RUST_REPAIR_BIN") or str(Path(__file__).resolve().parents[3] / "target/release/rust-repair")
EVAL = str(Path(__file__).resolve().parents[1] / "datasets/eval_ood_v1.jsonl")
OUT = str(Path(__file__).resolve().parents[1] / "eval/results/ood_gemini.json")
MODEL = "gemini-3.6-flash"
CARGO = '[package]\nname="c"\nversion="0.1.0"\nedition="2021"\n\n[[bin]]\nname="c"\npath="src/main.rs"\n'

cases = [json.loads(l) for l in Path(EVAL).read_text().splitlines() if l.strip()]

def run(case):
    d = Path(tempfile.mkdtemp(prefix="oodg_"))
    try:
        (d/"src").mkdir()
        (d/"Cargo.toml").write_text(CARGO)
        (d/"src/main.rs").write_text(case["broken"])
        log = d/"traj.jsonl"
        cmd = [BIN, "--path", str(d), "--backend", "vertex", "--model", MODEL,
               "--gcp-project", os.environ.get("GCP_PROJECT", "your-gcp-project"), "--gcp-location", "global",
               "--max-attempts", "1", "--log", str(log), "--case-id", case["case_id"]]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return case["case_id"], case["category"], False
        traj = None
        if log.exists():
            ls = [l for l in log.read_text().splitlines() if l.strip()]
            if ls: traj = json.loads(ls[-1])
        return case["case_id"], case["category"], bool(traj and traj.get("outcome")=="success")
    finally:
        shutil.rmtree(d, ignore_errors=True)

t0=time.time(); rows=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    for f in as_completed([ex.submit(run,c) for c in cases]):
        cid,cat,ok=f.result(); rows.append({"case_id":cid,"category":cat,"solved":ok})
solved=sum(r["solved"] for r in rows)
json.dump({"model":MODEL,"solved":solved,"n":len(cases),"rate":round(solved/len(cases),4),
           "per_case":rows}, open(OUT,"w"), indent=2)
print(f"GEMINI-FLASH OOD: {solved}/{len(cases)} = {solved/len(cases)*100:.1f}%  ({time.time()-t0:.0f}s)")
