# rust-repair datagen

Produces the compiler-verified failure corpus and teacher trajectories that
Phase 1/2 consume. Everything here is deterministic (seeded) and every case is
confirmed by the real `cargo check`.

## Pipeline

```text
seeds.py (compiling programs)
   │  break_it.py   (labeled mutations + parallel cargo verify)
   ▼
corpus.jsonl  (broken cases, each tagged with intended error code + category)
   │  split_eval.py  (stratified, deterministic; freezes eval set first)
   ├──► eval_vN.jsonl        (frozen; never used for training)
   ├──► train_pool_vN.jsonl
   └──► manifest.json        (sizes, per-category histograms, frozen eval_ids)
                │  teacher.py  (Gemini through the SAME Rust controller)
                ▼
        teacher_trajectories_vN.jsonl  (verified SFT positives, eval excluded)
```

## Commands

```bash
# 1. generate a balanced, cargo-verified corpus
python break_it.py --out /tmp/corpus_v0.jsonl --count 500 --seed 0 --verify --workers 4

# 2. freeze eval set + split (stratified by category, leak-checked)
python split_eval.py --corpus /tmp/corpus_v0.jsonl --eval-frac 0.35 --seed 7 \
    --out-eval ../datasets/eval_v0.jsonl \
    --out-train ../datasets/train_pool_v0.jsonl \
    --out-manifest ../datasets/manifest.json

# 3. teacher trajectories (frontier through the controller; eval excluded)
GEMINI_API_KEY=... python teacher.py \
    --train-pool ../datasets/train_pool_v0.jsonl \
    --binary ../../../target/release/rust-repair \
    --model gemini-3.6-flash \
    --out ../datasets/teacher_trajectories_v0.jsonl \
    --exclude-eval ../datasets/manifest.json \
    --max-attempts 4
```

## Design notes

- **Labeled mutations.** Each mutation targets a specific error class
  (E0308 type mismatch, E0433 missing import, E0599 unknown method, E0425
  undeclared ident, E0277 missing derive/trait bound, E0596 missing mut, E0282
  missing type annotation, E0560 wrong field, plus syntax). The intended code is
  the ground truth for the per-error-code competence envelope.
- **Verify-or-discard.** A candidate is kept only if `cargo check` actually fails
  with the intended code. Mutations that happen not to break (or break
  differently) are dropped.
- **Category balancing.** A per-category cap keeps any single mutation from
  dominating the corpus, so the envelope table stays meaningful.
- **Eval frozen first, leak-checked.** `split_eval.py` asserts eval/train
  case_id disjointness (case_id = SHA1 of the broken source) and records the
  frozen eval ids in the manifest so `teacher.py` excludes them.
- **Teacher = controller.** The teacher runs through the exact `rust-repair`
  controller the tiny model uses, so teacher trajectories are shape-identical to
  tiny-model runs and every step is compiler-verified. Only `success` episodes
  are kept as SFT positives (their full multi-step history included).

## Current status (v2)

Corpus scaled with a **parametric seed generator** (`seed_gen.py`): 14 templates
synthesize diverse compiling Rust programs by varying identifiers, types, values,
method chains, struct shapes, collections, and control flow. Static hand-written
seeds (`seeds.py`) can be combined via `--include-static`.

```bash
# 450-case verified corpus from 299 seeds (49 static + 250 generated)
python break_it.py --out corpus_v2.jsonl --count 450 --seed 0 --verify \
    --workers 4 --gen-seeds 250 --include-static
```

- 299 seed programs -> 450 cargo-verified cases across all 12 categories
- Distribution balanced (largest category ~21%, smallest single-digit)
- **eval_v1 stays frozen**: the v2 train pool excludes the 45 frozen eval
  case_ids (23 overlaps removed) -> `train_pool_v2.jsonl` = 427 leak-free cases.
  Baselines remain directly comparable.

To grow further, raise `--gen-seeds` or add templates to `seed_gen.py`.

### v1 (historical)

49 static seeds x 12 site-enumerating mutations -> 168 candidates -> 131 verified,
split 45 eval / 86 train. eval_v1 remains the frozen evaluation set.

## Datasets are not committed

`datasets/*.jsonl` are gitignored (they live on the OCI disk / HF Hub). Only
`manifest.json` is tracked, so the split is reproducible from the seed + corpus.
