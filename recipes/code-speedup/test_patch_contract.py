from patch_contract import canonicalize_patch, load_system_prompt, reconstruct_candidate


def test_unified_diff_is_canonicalized_without_changing_candidate():
    source = "def solve(xs):\n    return slow(xs)\n"
    unified = (
        "--- candidate.py\n"
        "+++ candidate.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def solve(xs):\n"
        "-    return slow(xs)\n"
        "+    return fast(xs)\n"
    )

    original_candidate, _, original_applied = reconstruct_candidate(source, unified)
    canonical = canonicalize_patch(unified)
    canonical_candidate, _, canonical_applied = reconstruct_candidate(source, canonical)

    assert original_applied is True
    assert canonical_applied is True
    assert canonical_candidate == original_candidate
    assert "<<<<<<< SEARCH" in canonical
    assert "@@ -" not in canonical


def test_shared_prompt_declares_the_canonical_contract():
    prompt = load_system_prompt()
    assert "ONLY one JSON object" in prompt
    assert "<<<<<<< SEARCH" in prompt
    assert ">>>>>>> REPLACE" in prompt
    assert "All imports" in prompt and "are forbidden" in prompt
