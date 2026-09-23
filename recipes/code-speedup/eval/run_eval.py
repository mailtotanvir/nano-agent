#!/usr/bin/env python3
"""Evaluate a local model through the production controller and private verifier.

Beyond the frozen v1 pass/fail row (``case_id``/``family_id``/``success``/
``outcome``/``attempts``/``latency_ms``), this runner can capture rich failure
detail for diagnosis (Phase 1.1): the model's parsed proposal, whether the JSON
proposal and the SEARCH/REPLACE blocks parsed, and an authoritative re-run of the
private bridge on the model's *own* reconstructed candidate (safe JSON status,
correctness, reward, instruction counts).

Data boundary: only the model-produced proposal/candidate and the bridge's
sanitized JSON response are recorded. Private benchmark inputs, edge inputs,
generators, and the known-fast reference solution are never read or written here;
they stay inside the bridge process. Diagnostic detail lands only in the local
JSON artifact, never in a model-visible prompt or log.
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

from patch_contract import reconstruct_candidate


def container_bridge_command(image: str, pythonpath: Path, bridge: Path,
                             dataset: Path, case_id: str, candidate: Path) -> list[str]:
    """Build an isolated verifier command while inference stays on the host."""
    root = pythonpath.resolve()
    bridge_rel = bridge.resolve().relative_to(root)
    dataset_rel = dataset.resolve().relative_to(root)
    return [
        "docker", "run", "--rm", "--privileged", "--network", "none",
        "-e", "PYTHONPATH=/repo",
        "-v", f"{root}:/repo:ro",
        "-v", f"{candidate.parent.resolve()}:/candidate:ro",
        image, "python3", f"/repo/{bridge_rel}",
        "--dataset", f"/repo/{dataset_rel}", "--problem-id", case_id,
        "--candidate", f"/candidate/{candidate.name}",
    ]


def run_bridge(python: str, bridge: Path, pythonpath: Path, dataset: Path,
               case_id: str, candidate_source: str,
               bridge_container_image: str | None = None) -> dict:
    """Re-run the private bridge on the model's candidate; safe JSON only.

    The bridge never returns private inputs or the known-fast solution.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(candidate_source)
        candidate_path = Path(handle.name)
    try:
        if bridge_container_image:
            command = container_bridge_command(
                bridge_container_image, pythonpath, bridge, dataset, case_id,
                candidate_path,
            )
            env = os.environ.copy()
        else:
            command = [python, str(bridge), "--dataset", str(dataset),
                       "--problem-id", case_id, "--candidate", candidate_path]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(pythonpath)
        completed = subprocess.run(
            command,
            capture_output=True, text=True, env=env, timeout=600, check=False,
        )
        try:
            return json.loads(completed.stdout)
        except (ValueError, json.JSONDecodeError):
            return {"passed": False, "status": "bridge_error", "correct": False,
                    "reward": None, "reference_instruction_count": None,
                    "candidate_instruction_count": None}
    finally:
            os.unlink(candidate_path)


def classify_failure(step: dict, blocks_parsed: bool, blocks_applied: bool,
                     bridge: dict | None) -> str:
    """Assign one taxonomy category to a non-successful final step.

    formatting            unparseable / contract-violating JSON proposal
    invalid_patch         JSON parsed but patch blocks rejected by the controller
                          (unparseable blocks, not-found, ambiguous, size cap) --
                          nothing applied to the workspace
    semantic_failure      patch applied but ran wrong (mismatch / candidate_error)
    correct_but_not_faster  correct, but no Cachegrind speedup (no_speedup)
    missed_optimization   other: model declined (no_fix/escalate), or the bridge
                          reported an infrastructure/bridge error
    """
    proposal = step.get("proposal", {})
    action = proposal.get("action", "")
    reason = proposal.get("reason") or ""
    # A model whose output could not be parsed to a Proposal is recorded by the
    # controller as a synthetic no_fix with reason "model_error: ...".
    if reason.startswith("model_error"):
        return "formatting"
    if action in ("no_fix", "escalate"):
        return "missed_optimization"
    # action == "patch"
    if not step.get("patch_applied", False):
        # JSON parsed, but the patch body was malformed or the blocks did not
        # apply cleanly -> controller rejected the patch.
        return "invalid_patch"
    status = (bridge or {}).get("status", "")
    if status == "passed":
        return "correct_but_not_faster"  # should not reach here for a failure
    if status == "no_speedup":
        return "correct_but_not_faster"
    if status in ("mismatch", "candidate_error"):
        return "semantic_failure"
    return "missed_optimization"


def evaluate(dataset: Path, binary: Path, model: str, url: str,
             max_attempts: int, max_cases: int, out: Path, capture: bool,
             python: Path, bridge_container_image: str | None = None) -> dict:
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if max_cases:
        cases = cases[:max_cases]
    # recipes/code-speedup root: the tree holding harness/bridge.py. Walk up from
    # the dataset so out-of-tree dataset copies (smoke tests) still resolve.
    speedup_root = dataset.resolve().parents[1]
    for parent in [dataset.resolve(), *dataset.resolve().parents]:
        if (parent / "harness" / "bridge.py").is_file():
            speedup_root = parent
            break
    bridge_py = speedup_root / "harness" / "bridge.py"
    references = {case["id"]: case["reference_source"] for case in cases}

    rows_path = out.with_suffix(out.suffix + ".rows.jsonl") if capture else None
    if rows_path and rows_path.exists():
        rows_path.unlink()  # fresh diagnostic run; never resume stale rows

    rows = []
    with tempfile.TemporaryDirectory(prefix="code_speed_eval_") as temp:
        root = Path(temp)
        for index, case in enumerate(cases, start=1):
            workspace = root / case["id"]
            workspace.mkdir()
            (workspace / "candidate.py").write_text(case["reference_source"])
            log = workspace / "trajectory.jsonl"
            started = time.monotonic()
            command = [
                str(binary), "--workspace", str(workspace), "--dataset", str(dataset),
                "--problem-id", case["id"], "--backend", "llama", "--model", model,
                "--llama-url", url, "--max-attempts", str(max_attempts), "--log", str(log),
                "--python", str(python),
            ]
            if bridge_container_image:
                command.extend(["--bridge-container-image", bridge_container_image])
            process = subprocess.run(command, capture_output=True, text=True, timeout=1800,
                                     check=False, env=os.environ.copy())
            trajectory = json.loads(log.read_text().splitlines()[-1]) if log.exists() else {}
            success = trajectory.get("outcome") == "success" and process.returncode == 0
            row = {
                "case_id": case["id"], "family_id": case["family_id"],
                "success": success, "outcome": trajectory.get("outcome", "error"),
                "attempts": trajectory.get("attempts", 0),
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
            if capture:
                row.update(capture_detail(
                    trajectory, references[case["id"]], success, python, bridge_py,
                    speedup_root, dataset, case["id"], process, bridge_container_image))
            rows.append(row)
            if rows_path:
                with rows_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            tag = row.get("failure_category", "-") if capture else ""
            print(f"[{index}/{len(cases)}] {case['id']} success={success} {tag}", flush=True)
    return {
        "dataset": str(dataset),
        "model": model,
        "max_attempts": max_attempts,
        "max_cases": max_cases,
        "rows": rows,
    }


def capture_detail(trajectory: dict, reference_source: str, success: bool,
                   python: Path, bridge_py: Path, speedup_root: Path,
                   dataset: Path, case_id: str, process,
                   bridge_container_image: str | None = None) -> dict:
    """Rich, model-safe failure/success detail for one case (diagnostic only)."""
    steps = trajectory.get("steps", [])
    final = steps[-1] if steps else {}
    proposal = final.get("proposal", {}) if final else {}
    patch_text = proposal.get("patch")
    reason = proposal.get("reason") or ""

    # Parser outcome: did the JSON proposal and its patch blocks parse?
    json_parsed = not reason.startswith("model_error")
    candidate, blocks_parsed, blocks_applied = reconstruct_candidate(
        reference_source, patch_text) if json_parsed else (None, False, False)

    bridge = None
    if candidate is not None:
        bridge = run_bridge(str(python), bridge_py, speedup_root, dataset,
                            case_id, candidate, bridge_container_image)

    detail = {
        # raw model proposal (parsed contract fields; for unparseable output the
        # controller stores a truncated sample of the bad text in `reason`).
        "proposal": {
            "action": proposal.get("action"),
            "patch": patch_text,
            "reason": reason,
            "confidence": proposal.get("confidence"),
        },
        "parser_outcome": {
            "json_proposal_parsed": json_parsed,
            "patch_blocks_parsed": blocks_parsed,
            "patch_applied": bool(final.get("patch_applied", False)),
            "patch_fuzzy": bool(final.get("patch_fuzzy", False)),
            "reverted": bool(final.get("reverted", False)),
        },
        # controller/verifier severity from the in-loop re-verify (0 pass, 1
        # no_speedup, 2 mismatch/candidate_error, 3 infra/bridge error).
        "controller_error_count_after": final.get("error_count_after"),
        # authoritative bridge re-run on the model's own candidate (safe JSON).
        "verifier": bridge,
        "correct": (bridge or {}).get("correct") if bridge else None,
        "reward": (bridge or {}).get("reward") if bridge else None,
        "reference_instruction_count": (bridge or {}).get("reference_instruction_count") if bridge else None,
        "candidate_instruction_count": (bridge or {}).get("candidate_instruction_count") if bridge else None,
        "candidate_source": candidate,
        "cli_returncode": process.returncode,
        "tokens_out": final.get("tokens_out"),
    }
    if not success:
        detail["failure_category"] = classify_failure(
            final, blocks_parsed, blocks_applied, bridge)
    else:
        detail["failure_category"] = None
    return detail


def summarize(report: dict) -> None:
    rows = report["rows"]
    by_family = defaultdict(lambda: [0, 0])
    for row in rows:
        by_family[row["family_id"]][1] += 1
        by_family[row["family_id"]][0] += int(row["success"])
    report["summary"] = {
        "total": len(rows),
        "successful": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else 0,
        "by_family": {name: {"successful": value[0], "total": value[1]}
                      for name, value in sorted(by_family.items())},
    }
    if any("failure_category" in row for row in rows):
        categories = defaultdict(int)
        for row in rows:
            category = row.get("failure_category")
            if category:
                categories[category] += 1
        report["summary"]["failure_categories"] = dict(sorted(categories.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--python", type=Path, default=Path("python3"),
                        help="interpreter for the private bridge (needs Valgrind on PATH/env)")
    parser.add_argument("--capture-detail", action="store_true",
                        help="record rich, model-safe failure detail for diagnosis")
    parser.add_argument("--bridge-container-image",
                        help="replay verifier in this privileged, network-disabled Docker image")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.dataset.resolve(), args.binary.resolve(), args.model, args.url,
                      args.max_attempts, args.max_cases, args.out.resolve(),
                      args.capture_detail, args.python, args.bridge_container_image)
    summarize(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
