"""Decision gate aggregation tests. Fixture measurements here are unit data."""
import pytest


@pytest.mark.cachegrind
def test_real_spike_persists_measurements_and_refuses_to_overwrite(tmp_path):
    import json

    from harness.problems import SPIKE_PAIRS
    from harness.spike import run_spike

    output = tmp_path / "spike.json"
    run_spike(output, seed=31, pairs=SPIKE_PAIRS[:1])
    report = json.loads(output.read_text())
    assert report["summary"]["gate"] == "GO_REWARD_SIGNAL", report
    assert len(report["pairs"]) == 1
    assert len(report["attacks"]) == 2
    assert report["runtime"]["python"].startswith("Python ")
    assert report["source_sha256"]["sandbox.py"]
    with pytest.raises(FileExistsError):
        run_spike(output, seed=31, pairs=SPIKE_PAIRS[:1])



def test_spike_gate_requires_every_unique_pair_and_rejects_noise():
    from harness.spike import summarize

    rows = [{"id": "a", "correct": True, "status": "ok", "reward": 1.0,
             "reference_counts": [200, 200], "candidate_counts": [100, 100]}]
    attacks = [{"id": "hardcode", "status": "mismatch", "passed": False}]
    assert summarize(rows, ["a"], attacks)["gate"] == "GO_REWARD_SIGNAL"
    assert summarize([], ["a"], attacks)["gate"] == "NO_GO"
    assert summarize(rows * 2, ["a"], attacks)["gate"] == "NO_GO"
    rows[0]["candidate_counts"] = [100, 101]
    assert summarize(rows, ["a"], attacks)["gate"] == "NO_GO"


def test_spike_gate_cannot_pass_without_attack_rejection():
    from harness.spike import summarize

    rows = [{"id": "a", "correct": True, "status": "ok", "reward": 1.0,
             "reference_counts": [200, 200], "candidate_counts": [100, 100]}]
    assert summarize(rows, ["a"], [])["gate"] == "NO_GO"
    assert summarize(rows, ["a"], [{"id": "cheat", "passed": True,
                                   "status": "passed"}])["gate"] == "NO_GO"
