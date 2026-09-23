from pathlib import Path

from eval.run_eval import container_bridge_command, reconstruct_candidate


def test_reconstruct_candidate_accepts_unified_diff_fallback():
    source = "def solve(xs):\n    return slow(xs)\n"
    patch = (
        "--- candidate.py\n"
        "+++ candidate.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def solve(xs):\n"
        "-    return slow(xs)\n"
        "+    return fast(xs)\n"
    )

    candidate, parsed, applied = reconstruct_candidate(source, patch)

    assert candidate == "def solve(xs):\n    return fast(xs)\n"
    assert parsed is True
    assert applied is True


def test_reconstruct_candidate_rejects_nonmatching_unified_diff():
    source = "def solve(xs):\n    return slow(xs)\n"
    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def solve(xs):\n"
        "-    return missing(xs)\n"
        "+    return fast(xs)\n"
    )

    candidate, parsed, applied = reconstruct_candidate(source, patch)

    assert candidate is None
    assert parsed is True
    assert applied is False


def test_container_bridge_command_is_networkless_and_read_only():
    command = container_bridge_command(
        "code-speed-verifier:test", Path("/repo").resolve(),
        Path("/repo/harness/bridge.py").resolve(), Path("/repo/datasets/dev.jsonl").resolve(),
        "case-1", Path("/tmp/candidate.py").resolve(),
    )

    assert command[:6] == ["docker", "run", "--rm", "--privileged", "--network", "none"]
    assert command[6:8] == ["-e", "PYTHONPATH=/repo"]
    assert command[8] == "-v"
    assert "/repo:/repo:ro" in command
    assert "/tmp:/candidate:ro" in command
    assert command[-6:] == ["--dataset", "/repo/datasets/dev.jsonl", "--problem-id", "case-1", "--candidate", "/candidate/candidate.py"]
