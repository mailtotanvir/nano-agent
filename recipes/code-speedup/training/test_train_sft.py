from training.contract_probe import (
    DEFAULT_CHAT_TEMPLATE,
    parse_action,
    select_rows,
    supervision_report,
)
from training.train_sft import apply_v1_replay, split_by_case


def _row(case_id, source, family):
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"family: {family}\ncontext"},
            {"role": "assistant", "content": "answer"},
        ],
        "meta": {"case_id": case_id, "source": source},
    }


def test_case_group_split_is_disjoint_and_deterministic():
    rows = []
    for source in ("teacher_v2", "sft_v1"):
        for family in ("a", "b"):
            for index in range(10):
                row = _row(f"{source}-{family}-{index}", source, family)
                rows.extend([row, dict(row)])

    train_a, eval_a = split_by_case(rows, 0.2, 7)
    train_b, eval_b = split_by_case(rows, 0.2, 7)
    train_ids = {row["meta"]["case_id"] for row in train_a}
    eval_ids = {row["meta"]["case_id"] for row in eval_a}

    assert train_ids.isdisjoint(eval_ids)
    assert [row["meta"]["case_id"] for row in train_a] == [
        row["meta"]["case_id"] for row in train_b]
    assert [row["meta"]["case_id"] for row in eval_a] == [
        row["meta"]["case_id"] for row in eval_b]


def test_v1_replay_reaches_requested_effective_ratio():
    rows = ([_row(f"new-{index}", "teacher_v2", "a") for index in range(60)]
            + [_row(f"old-{index}", "sft_v1", "a") for index in range(10)])

    replayed = apply_v1_replay(rows, 0.4, 7)
    legacy = sum(row["meta"]["source"] == "sft_v1" for row in replayed)

    assert len(replayed) == 100
    assert legacy == 40


def test_contract_probe_selects_source_family_strata_and_parses_action():
    rows = [
        _row("v1-a", "sft_v1", "a"),
        _row("v2-b", "teacher_v2", "b"),
        _row("v2-c", "teacher_v2", "c"),
    ]
    chosen = select_rows(rows, 3)

    assert {row["meta"]["case_id"] for row in chosen} == {"v1-a", "v2-b", "v2-c"}
    assert parse_action('prefix {"action":"patch","patch":"x"} suffix') == "patch"
    assert parse_action('{"action":"rewrite"}') == "rewrite"
    assert parse_action("not json") is None


def test_checked_in_template_declares_assistant_generation_boundaries():
    template = DEFAULT_CHAT_TEMPLATE.read_text(encoding="utf-8")

    assert "generation %}" in template
    assert "endgeneration %}" in template


class _MultiTurnMaskTokenizer:
    """Small tokenizer double proving repair traces retain all assistant spans."""

    chat_template = "test"

    def apply_chat_template(self, _messages, **_kwargs):
        # Two contiguous assistant spans, each decoding to the canonical action.
        return {"input_ids": [0, 1, 2, 3, 4], "assistant_masks": [0, 1, 1, 0, 1]}

    def decode(self, ids):
        return '{"action": "patch"' if ids and ids[0] in (1, 4) else "x"


def test_contract_probe_accepts_multi_turn_repair_trace():
    report = supervision_report(_MultiTurnMaskTokenizer(), [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": '{"action": "patch"}'},
        {"role": "user", "content": "feedback"},
        {"role": "assistant", "content": '{"action": "patch"}'},
    ], max_length=128)

    assert report["assistant_turns"] == 2
    assert report["assistant_tokens"] == 3
