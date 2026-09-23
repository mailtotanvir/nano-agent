import json

from patch_contract import load_system_prompt

from datagen.build_sft_v2 import (
    SYSTEM_PROMPT,
    TASK_SUFFIX,
    build_v2_examples,
    canonical_user_context,
)


def test_v2_builder_normalizes_teacher_unified_diff_and_uses_shared_prompt():
    record = {
        "case_id": "cs-v2-test",
        "teacher": {"model": "teacher"},
        "verifier": {"passed": True, "reward": 1.0},
        "trajectory": {
            "outcome": "success",
            "steps": [{
                "attempt": 0,
                "context": (
                    "family: test\n--- candidate.py ---\n"
                    "return slow(xs)\n"
                    "\nRewrite candidate.py to be correct and use fewer instruction references."
                ),
                "proposal": {
                    "action": "patch",
                    "patch": (
                        "--- candidate.py\n+++ candidate.py\n@@ -1 +1 @@\n"
                        "-return slow(xs)\n+return fast(xs)\n"
                    ),
                    "reason": "faster",
                    "confidence": 1.0,
                },
                "patch_applied": True,
                "reverted": False,
            }],
        },
    }

    single, repair = build_v2_examples([record])

    assert repair == []
    assert single[0]["messages"][0]["content"] == load_system_prompt() == SYSTEM_PROMPT
    patch = json.loads(single[0]["messages"][-1]["content"])["patch"]
    assert "<<<<<<< SEARCH" in patch
    assert "@@ -" not in patch
    assert single[0]["messages"][1]["content"].endswith(TASK_SUFFIX)


def test_builder_accepts_a_teacher_context_already_using_current_suffix():
    context = "family: test\n--- candidate.py ---\nreturn slow(xs)\n" + TASK_SUFFIX
    assert canonical_user_context(context) == context
