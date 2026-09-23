"""Tests for the SFT v2 training-only corpus generator and leakage checker.

Follows ``test_gen_problems.py`` patterns: deterministic construction, no
fast-source serialization, schema parity with v1, correctness of every known-fast
rewrite against its slow reference over trusted cases, and airtight leakage
invariants against the frozen v1 evals plus the new behavioral dev split.
"""
import ast
import copy
import json

import pytest

SEEN_FAMILIES = (
    "hash-membership", "distinct-cardinality", "stable-deduplication",
    "frequency-index", "complement-lookup", "running-aggregate",
    "prefix-range-query", "fixed-sliding-window",
)
NEW_FAMILIES = (
    "position-index", "count-index", "membership-fusion",
    "ordered-intersection", "mode-bucket", "residue-count-index",
)
FROZEN_HELDOUT_FAMILIES = ("string-concatenation", "sort-selection", "indexed-lookup")


def test_v2_corpus_is_deterministic_and_hits_template_and_size_targets():
    from datagen.gen_problems import generate_v2_pairs

    first = generate_v2_pairs()
    second = generate_v2_pairs()
    assert [p.record["id"] for p in first] == [p.record["id"] for p in second]
    # ~450-600 training-only problems.
    assert 450 <= len(first) <= 600
    templates = {p.record["template_id"] for p in first}
    assert len(templates) >= 120
    # >=120 distinct templates across the eight seen families alone (~3x v1's 42).
    seen_templates = {p.record["template_id"] for p in first
                      if p.record["family_id"] in SEEN_FAMILIES}
    assert len(seen_templates) >= 120
    assert len({p.record["id"] for p in first}) == len(first)


def test_v2_records_share_the_v1_schema_and_never_serialize_fast_source():
    from datagen.gen_problems import generate_v2_pairs, public_record

    pairs = generate_v2_pairs()
    v1_fields = {
        "schema_version", "id", "family_id", "template_id", "semantic_variant",
        "split_hint", "function_name", "reference_source", "normalized_ast_sha256",
        "input_generator", "edge_inputs", "benchmark_variants",
        "benchmark_variant_ids", "private_seed_policy", "verification",
    }
    for pair in pairs[:20]:
        assert set(pair.record) == v1_fields
        assert pair.record["split_hint"] == "train_v2"
    serialized = "\n".join(json.dumps(public_record(pair)) for pair in pairs)
    assert "fast_source" not in serialized and "known_fast_source" not in serialized
    assert all(pair.fast_source not in serialized for pair in pairs)


def test_v2_includes_training_only_new_families_none_replicating_frozen_heldout():
    from datagen.gen_problems import generate_v2_pairs

    families = {p.record["family_id"] for p in generate_v2_pairs()}
    assert set(NEW_FAMILIES) <= families
    assert set(SEEN_FAMILIES) <= families
    assert families.isdisjoint(FROZEN_HELDOUT_FAMILIES)


def test_v2_new_families_are_index_building_or_pure_subset_transforms():
    """The new families exist to answer diagnosed v1 failure modes."""
    from datagen import _v2_bank as bank

    # index-building families materialize a dict/precomputed lookup.
    for family in ("position-index", "count-index"):
        assert "{}" in bank.FAST_BODY[family]
        assert ".get(" in bank.FAST_BODY[family]
    # pure-subset membership fusion builds a set instead of a nested scan.
    for family in ("membership-fusion", "ordered-intersection"):
        assert "set(" in bank.FAST_BODY[family]
    # transformed count queries build their own residue index rather than an
    # item/key-value map like the frozen indexed-lookup benchmark.
    assert "bucket = x % bucket_size" in bank.FAST_BODY["residue-count-index"]
    assert "counts.get(q % bucket_size, 0)" in bank.FAST_BODY["residue-count-index"]
    # every new family carries a one-line rationale.
    assert set(bank.NEW_FAMILY_RATIONALE) == set(NEW_FAMILIES)


@pytest.mark.parametrize("family", SEEN_FAMILIES + NEW_FAMILIES)
def test_every_v2_known_fast_matches_its_slow_reference_over_trusted_cases(family):
    from datagen.gen_problems import generate_v2_pairs, problem_from_record

    pairs = [p for p in generate_v2_pairs() if p.record["family_id"] == family]
    assert pairs, family
    for pair in pairs[:6]:
        problem = problem_from_record(pair.record)
        reference, candidate = {}, {}
        exec(pair.record["reference_source"], reference)  # noqa: S102 - trusted generator output
        exec(pair.fast_source, candidate)  # noqa: S102 - trusted generator output
        for args in problem.hidden_inputs(2024, 4):
            assert (reference["solve"](*copy.deepcopy(args))
                    == candidate["solve"](*copy.deepcopy(args))), pair.record["id"]


def test_all_v2_public_sources_parse_and_have_distinct_ast_fingerprints():
    from datagen.gen_problems import generate_v2_pairs

    pairs = generate_v2_pairs()
    for pair in pairs:
        ast.parse(pair.record["reference_source"])
    assert len({p.record["normalized_ast_sha256"] for p in pairs}) == len(pairs)


def test_v2_semantic_instances_change_observable_behavior_within_a_template():
    from datagen.gen_problems import generate_v2_pairs

    by_template = {}
    for pair in generate_v2_pairs():
        by_template.setdefault(pair.record["template_id"], []).append(pair)
    changed = 0
    for pairs in by_template.values():
        if len(pairs) < 2:
            continue
        assert pairs[0].record["normalized_ast_sha256"] != pairs[1].record["normalized_ast_sha256"]
        changed += 1
    assert changed > 0


def test_dev_behavioral_split_is_fresh_template_disjoint_and_marked():
    from datagen.gen_problems import (
        generate_dev_behavioral_pairs,
        generate_v2_pairs,
    )

    dev = generate_dev_behavioral_pairs()
    # ~24 cases, 3 per seen family.
    assert len(dev) == 24
    assert all(p.record["split_hint"] == "dev_behavioral" for p in dev)
    by_family = {}
    for pair in dev:
        by_family.setdefault(pair.record["family_id"], []).append(pair)
    assert set(by_family) == set(SEEN_FAMILIES)
    assert all(len(rows) == 3 for rows in by_family.values())
    # dev templates and AST fingerprints appear in NO training data.
    train = generate_v2_pairs()
    train_templates = {p.record["template_id"] for p in train}
    train_hashes = {p.record["normalized_ast_sha256"] for p in train}
    dev_templates = {p.record["template_id"] for p in dev}
    dev_hashes = {p.record["normalized_ast_sha256"] for p in dev}
    assert dev_templates.isdisjoint(train_templates)
    assert dev_hashes.isdisjoint(train_hashes)


def test_v2_training_is_disjoint_from_the_frozen_v1_evals():
    from datagen.gen_problems import generate_pairs, generate_v2_pairs

    v1 = generate_pairs(n=250, seed=20260907)
    frozen = [p for p in v1 if p.record["split_hint"] in ("seen_eval", "heldout_eval")]
    v2 = generate_v2_pairs()
    for key in ("id", "template_id", "normalized_ast_sha256"):
        assert {p.record[key] for p in v2}.isdisjoint({p.record[key] for p in frozen})


def test_v6_is_fresh_and_has_a_template_disjoint_one_time_heldout_split():
    from datagen.gen_problems import generate_v6_heldout_pairs, generate_v6_pairs

    train = generate_v6_pairs()
    heldout = generate_v6_heldout_pairs()
    assert all(p.record["split_hint"] == "train_v6" for p in train)
    assert all(p.record["split_hint"] == "heldout_v6_unobserved" for p in heldout)
    assert {p.record["family_id"] for p in train}.isdisjoint(FROZEN_HELDOUT_FAMILIES)
    for key in ("id", "template_id", "normalized_ast_sha256"):
        assert {p.record[key] for p in train}.isdisjoint({p.record[key] for p in heldout})
    assert {p.record["semantic_variant"] for p in train}.isdisjoint(
        {p.record["semantic_variant"] for p in heldout})


def test_v6_heldout_fast_rewrites_match_its_fresh_slow_references():
    from datagen.gen_problems import generate_v6_heldout_pairs, problem_from_record

    for pair in generate_v6_heldout_pairs():
        problem = problem_from_record(pair.record)
        reference, candidate = {}, {}
        exec(pair.record["reference_source"], reference)  # noqa: S102 - trusted generator output
        exec(pair.fast_source, candidate)  # noqa: S102 - trusted generator output
        for args in problem.hidden_inputs(2026, 4):
            assert reference["solve"](*copy.deepcopy(args)) == candidate["solve"](*copy.deepcopy(args))


def test_leakage_audit_flags_and_passes_correctly():
    from datagen.leakage_check_v2 import audit

    train = [
        {"id": "a", "family_id": "position-index", "template_id": "t1", "normalized_ast_sha256": "h1"},
        {"id": "b", "family_id": "hash-membership", "template_id": "t2", "normalized_ast_sha256": "h2"},
    ]
    new_family_rows = [train[0]]
    clean_eval = {
        "v1_seen_family": [
            {"id": "z", "family_id": "hash-membership", "template_id": "e1", "normalized_ast_sha256": "he1"},
        ],
    }
    report = audit(train_rows=train, new_family_rows=new_family_rows, eval_sets=clean_eval)
    assert report["all_clean"] is True

    leaky_eval = {
        "v1_seen_family": [
            {"id": "z", "family_id": "hash-membership", "template_id": "t2", "normalized_ast_sha256": "he1"},
        ],
    }
    leaky = audit(train_rows=train, new_family_rows=new_family_rows, eval_sets=leaky_eval)
    assert leaky["all_clean"] is False

    forbidden_family = [
        {"id": "c", "family_id": "indexed-lookup", "template_id": "t3", "normalized_ast_sha256": "h3"},
    ]
    bad = audit(train_rows=forbidden_family, new_family_rows=forbidden_family, eval_sets=clean_eval)
    assert bad["all_clean"] is False
