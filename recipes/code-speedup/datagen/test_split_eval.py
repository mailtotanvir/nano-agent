from collections import Counter


def test_layered_split_is_frozen_and_obeys_all_leakage_invariants():
    from datagen.gen_problems import generate_pairs, public_record
    from datagen.split_eval import build_split, check_invariants

    corpus = [public_record(pair) for pair in generate_pairs(n=250, seed=20260907)]
    split = build_split(corpus, seed=73)
    check_invariants(split)
    assert len(split["train"]) == 152
    assert len(split["development"]) == 38
    assert len(split["seen_family_eval"]) == 30
    assert len(split["family_heldout_eval"]) == 30
    assert len({row["template_id"] for row in split["seen_family_eval"]}) == 30
    heldout = {row["family_id"] for row in split["family_heldout_eval"]}
    assert len(heldout) >= 3
    assert heldout.isdisjoint({row["family_id"] for row in split["train"] + split["development"]})
    assert {row["normalized_ast_sha256"] for row in split["seen_family_eval"]}.isdisjoint(
        {row["normalized_ast_sha256"] for row in split["train"] + split["development"]})


def test_template_cardinalities_describe_real_structures_not_instance_nonces():
    from datagen.gen_problems import generate_pairs, public_record
    from datagen.split_eval import build_split

    split = build_split([public_record(pair) for pair in generate_pairs(n=250, seed=20260907)], seed=73)
    train_dev = split["train"] + split["development"]
    assert len({row["template_id"] for row in train_dev}) == 42
    assert len({row["template_id"] for row in split["seen_family_eval"]}) == 30
    assert len({row["template_id"] for row in split["family_heldout_eval"]}) == 15
    # Multiple heldout rows per template are semantic task instances, while the
    # template source shape itself remains fixed and the family is absent in train.
    assert len(split["family_heldout_eval"]) == 30


def test_manifest_records_inventory_and_auditable_invariants():
    from datagen.gen_problems import generate_pairs, public_record
    from datagen.split_eval import build_split, manifest_for

    split = build_split([public_record(pair) for pair in generate_pairs(n=250, seed=20260907)], seed=73)
    manifest = manifest_for(split, corpus_seed=20260907, split_seed=73)
    assert manifest["schema_version"] == 2
    assert manifest["corpus_verification"] == "unverified"
    assert manifest["sizes"] == {"train": 152, "development": 38,
                                 "seen_family_eval": 30, "family_heldout_eval": 30}
    assert manifest["invariants"]["normalized_ast_disjoint"] is True
    assert manifest["invariants"]["seen_eval_template_disjoint"] is True
    assert manifest["invariants"]["heldout_families_excluded_from_train_and_development"] is True
    assert all(count > 0 for count in manifest["family_inventory"].values())
    assert Counter(manifest["heldout_family_ids"]).most_common(1)[0][1] == 1
    assert set(manifest["split_full_record_sha256"]) == set(split)
    assert len(manifest["full_corpus_content_sha256"]) == 64
    assert len(manifest["full_corpus_normalized_ast_sha256"]) == 64


def test_full_record_provenance_hash_changes_for_benchmark_metadata_without_ast_change():
    import copy

    from datagen.gen_problems import generate_pairs, public_record
    from datagen.split_eval import build_split, manifest_for

    split = build_split([public_record(pair) for pair in generate_pairs(n=250, seed=20260907)], seed=73)
    before = manifest_for(split, corpus_seed=20260907, split_seed=73)
    changed = copy.deepcopy(split)
    changed["seen_family_eval"][0]["benchmark_variants"][0]["inputs"][0].append(123456)
    after = manifest_for(changed, corpus_seed=20260907, split_seed=73)
    assert before["full_corpus_normalized_ast_sha256"] == after["full_corpus_normalized_ast_sha256"]
    assert before["full_corpus_content_sha256"] != after["full_corpus_content_sha256"]
    assert (before["split_full_record_sha256"]["seen_family_eval"]
            != after["split_full_record_sha256"]["seen_family_eval"])


def test_v2_manifest_is_additive_and_records_training_only_and_dev_roles():
    from datagen.gen_problems import (
        generate_dev_behavioral_pairs,
        generate_v2_pairs,
        public_record,
    )
    from datagen.split_eval import v2_manifest

    train = [public_record(p) for p in generate_v2_pairs()]
    dev = [public_record(p) for p in generate_dev_behavioral_pairs()]
    manifest = v2_manifest(train, dev, corpus_seed=20260909, dev_seed=20260910)
    assert manifest["sizes"]["train_v2"] == len(train)
    assert manifest["sizes"]["dev_behavioral"] == len(dev)
    assert manifest["template_counts"]["train_v2_seen_families"] >= 120
    assert manifest["dev_policy"]["frozen"] is False
    assert "checkpoint_selection" in manifest["dev_policy"]["permitted_uses"]
    # never claims the frozen heldout families as training families
    assert set(manifest["new_family_ids"]).isdisjoint(
        manifest["frozen_heldout_family_ids_excluded"])
    assert set(manifest["frozen_heldout_family_ids_excluded"]) == {
        "string-concatenation", "sort-selection", "indexed-lookup"}
    # unverified before the harness gate runs
    assert manifest["train_verification"] == "unverified"
