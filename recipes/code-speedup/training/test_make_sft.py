import json


def test_make_sft_keeps_only_verified_successful_patches(tmp_path):
    from training.make_sft import main

    source = tmp_path / "teacher.jsonl"
    kept = {
        "case_id": "case-1",
        "teacher": {"model": "teacher-model"},
        "verifier": {"passed": True, "reward": 0.5},
        "trajectory": {
            "outcome": "success",
            "steps": [{
                "attempt": 0,
                "context": (
                    "family: test\n--- candidate.py ---\n"
                    "return slow(xs)\n"
                    "\nRewrite candidate.py to be correct and use fewer instruction references."
                ),
                "proposal": {"action": "patch", "patch": (
                    "--- candidate.py\n+++ candidate.py\n@@ -1 +1 @@\n"
                    "-return slow(xs)\n+return fast(xs)\n"
                ), "reason": "faster",
                             "confidence": 0.9},
                "patch_applied": True,
                "reverted": False,
            }],
        },
    }
    rejected = {
        **kept,
        "case_id": "case-2",
        "trajectory": {**kept["trajectory"], "outcome": "escalate"},
    }
    source.write_text("\n".join(map(json.dumps, [kept, rejected])) + "\n")
    output = tmp_path / "sft.jsonl"

    main_args = ["make_sft.py", "--trajectories", str(source), "--out", str(output)]
    import sys
    old_argv = sys.argv
    try:
        sys.argv = main_args
        main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["meta"]["case_id"] == "case-1"
    proposal = json.loads(rows[0]["messages"][-1]["content"])
    assert proposal["action"] == "patch"
    assert "<<<<<<< SEARCH" in proposal["patch"]
    assert "@@ -" not in proposal["patch"]
