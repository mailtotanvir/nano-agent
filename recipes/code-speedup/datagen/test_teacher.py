import json
import os
from pathlib import Path


def _record(case_id="case-1", verified=True):
    return {
        "id": case_id,
        "split_hint": "train_dev",
        "family_id": "hash-membership",
        "function_name": "solve",
        "reference_source": "def solve(xs):\n    return False\n",
        "verification": {"status": "passed" if verified else "unverified"},
        "edge_inputs": [[[1, 2, 3]]],
        "benchmark_variants": [{"id": "n", "inputs": [[1, 2, 3]]}],
        "input_generator": {"kind": "int_list"},
    }


def _fake_binary(path: Path):
    path.write_text("""#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
assert args[args.index('--backend') + 1] == 'gemini'
assert args[args.index('--model') + 1] == 'gemini-3.8-flash'
workspace = pathlib.Path(args[args.index('--workspace') + 1])
workspace.joinpath('candidate.py').write_text('def solve(xs):\\n    return True\\n')
log = pathlib.Path(args[args.index('--log') + 1])
log.write_text(json.dumps({'outcome': 'success', 'steps': [{'context': 'safe context only'}]}) + '\\n')
""")
    path.chmod(0o755)


def _fake_bridge(path: Path):
    path.write_text("""import json
print(json.dumps({'passed': True, 'status': 'passed', 'correct': True, 'reward': 0.7, 'reference_instruction_count': 100, 'candidate_instruction_count': 50}))
""")


def test_teacher_keeps_successes_only_and_is_idempotent(tmp_path, monkeypatch):
    from datagen.teacher import main

    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps(_record()) + "\n")
    binary, bridge = tmp_path / "fake-controller", tmp_path / "fake_bridge.py"
    _fake_binary(binary)
    _fake_bridge(bridge)
    out = tmp_path / "teacher.jsonl"
    stats = tmp_path / "stats.json"
    args = [
        "--train", str(train), "--binary", str(binary), "--bridge", str(bridge),
        "--python", os.sys.executable, "--out", str(out), "--stats", str(stats),
    ]
    assert main(args) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["case_id"] == "case-1"
    assert rows[0]["verifier"]["reward"] == 0.7
    serialized = out.read_text()
    assert "benchmark_variants" not in serialized
    assert "edge_inputs" not in serialized
    assert "input_generator" not in serialized
    assert "return False" not in serialized
    report = json.loads(stats.read_text())
    assert report["successes_kept"] == 1
    assert main(args) == 0
    assert len(out.read_text().splitlines()) == 1
    report = json.loads(stats.read_text())
    assert report["successes_kept"] == 0
    assert report["resumed_completed"] == 1


def test_teacher_rejects_non_train_or_unverified_records_before_invocation(tmp_path):
    from datagen.teacher import load_train_cases

    eval_path = tmp_path / "eval_speedup_v1.jsonl"
    eval_path.write_text(json.dumps(_record()) + "\n")
    try:
        load_train_cases(eval_path)
    except ValueError as exc:
        assert "evaluation" in str(exc) or "train" in str(exc)
    else:
        raise AssertionError("evaluation input must be rejected")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps(_record(verified=False)) + "\n")
    try:
        load_train_cases(train)
    except ValueError as exc:
        assert "verifier-gated" in str(exc)
    else:
        raise AssertionError("unverified training record must be rejected")


def _v2_record(case_id="cs-v2-0000-abcd", verified=True):
    row = _record(case_id=case_id, verified=verified)
    row["split_hint"] = "train_v2"
    return row


def test_teacher_accepts_v2_corpus_and_still_refuses_dev_behavioral(tmp_path):
    from datagen.teacher import load_train_cases

    # The v2 training-only corpus (split_hint train_v2) is accepted.
    corpus = tmp_path / "corpus_speedup_v2.verified.jsonl"
    corpus.write_text(json.dumps(_v2_record()) + "\n")
    rows = load_train_cases(corpus)
    assert len(rows) == 1 and rows[0]["split_hint"] == "train_v2"

    # The v2 behavioral dev split must be refused even though it is not a v1 eval.
    dev = tmp_path / "eval_speedup_v2_dev_behavioral.jsonl"
    dev.write_text(json.dumps(_v2_record()) + "\n")
    try:
        load_train_cases(dev)
    except ValueError as exc:
        assert "evaluation" in str(exc)
    else:
        raise AssertionError("v2 dev-behavioral split must be rejected")

    # A non-training split_hint inside a corpus file is still rejected.
    bad = tmp_path / "corpus_speedup_v2.bad.jsonl"
    row = _v2_record()
    row["split_hint"] = "dev_behavioral"
    bad.write_text(json.dumps(row) + "\n")
    try:
        load_train_cases(bad)
    except ValueError as exc:
        assert "non-train" in str(exc)
    else:
        raise AssertionError("non-train split_hint must be rejected")


def test_teacher_accepts_v6_training_corpus(tmp_path):
    from datagen.teacher import load_train_cases

    row = _v2_record(case_id="cs-v6-0000-abcd")
    row["split_hint"] = "train_v6"
    corpus = tmp_path / "corpus_speedup_v6.verified.jsonl"
    corpus.write_text(json.dumps(row) + "\n")
    rows = load_train_cases(corpus)
    assert len(rows) == 1 and rows[0]["split_hint"] == "train_v6"


def test_failure_diagnostic_excludes_context_and_patch():
    from datagen.teacher import _failure_diagnostic

    diagnostic = _failure_diagnostic({
        "attempts": 1,
        "steps": [{
            "attempt": 0,
            "context": "private model context",
            "proposal": {"action": "patch", "patch": "secret source"},
            "patch_applied": False,
            "reverted": False,
            "primary_code": "SPEEDUP_no_speedup",
            "error_count_after": 1,
        }],
    })
    serialized = json.dumps(diagnostic)
    assert "private model context" not in serialized
    assert "secret source" not in serialized
    assert diagnostic["steps"][0]["action"] == "patch"
