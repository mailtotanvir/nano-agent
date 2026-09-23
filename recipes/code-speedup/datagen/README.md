# Code-speedup corpus and frozen evaluation

`gen_problems.py` constructs the deterministic v1 corpus of pure-Python,
single-function rewrite tasks. Each row has a slow reference implementation,
an explicit optimization family and structural template ID, edge-case metadata,
and five performance probes: `n_minus_1`, `n`, `n_plus_1`, `values_changed`,
and `order_changed`.

The known fast implementation is held only in process while the trusted parent
verifies a row. It is never written to JSONL or passed to a model. Build the
temporary staging corpus with:

```bash
PYTHONPATH=recipes/code-speedup python3 -m datagen.gen_problems \
  --n 250 --seed 20260907 \
  --out recipes/code-speedup/datasets/corpus_speedup_v1.unverified.jsonl
```

Add `--verify` only on a host where the Phase 0 Bubblewrap and Cachegrind gate
works. It runs the real harness correctness battery and two deterministic
Cachegrind measurements for every known-fast pair. An unverified file is a
staging artifact and must not be used for SFT, GRPO, or headline evaluation.
The verifier is finite evidence of equivalence over the supported input domain,
not a proof of semantic equivalence.

Verification writes an append-only checkpoint after every completed row and
prints `[completed/total]` progress. The cache key includes the slow source,
known-fast source hash, every execution-relevant problem field (hidden-input
generator, edges, and benchmark variants), verifier configuration, and hashes
of every harness module. Resume a stopped host run with the same command and
checkpoint:

```bash
PYTHONPATH=recipes/code-speedup python3 -m datagen.gen_problems \
  --n 250 --seed 20260907 --verify \
  --out recipes/code-speedup/datasets/corpus_speedup_v1.verified.jsonl \
  --checkpoint recipes/code-speedup/datasets/corpus_speedup_v1.verify-checkpoint.jsonl
```

Changing a source, known-fast reference, oracle inputs, Cachegrind
path/configuration, or the harness automatically makes old checkpoint rows
ineligible for reuse.

Freeze the split from a verifier-gated corpus:

```bash
PYTHONPATH=recipes/code-speedup python3 -m datagen.split_eval \
  --corpus recipes/code-speedup/datasets/corpus_speedup_v1.verified.jsonl \
  --out-dir recipes/code-speedup/datasets \
  --manifest recipes/code-speedup/datasets/manifest.json
```

The v1 layout is 152 train, 38 development, 30 template-disjoint seen-family
evaluation, and 30 family-held-out evaluation. The entirely excluded families
are `string-concatenation`, `sort-selection`, and `indexed-lookup`. The two
evaluation tracks are stored separately and must be reported separately.

Use `model_prompt_record(record)` for all model-facing construction. It permits
only `id`, `family_id`, `template_id`, `function_name`, and `reference_source`.
It deliberately strips exact benchmarks, edge cases, generator metadata, the
private-seed policy, semantic-instance metadata, and verification results.

## Gemini teacher trajectories

`teacher.py` is the teacher-data path. It accepts a verifier-gated
`train_speedup_*.jsonl` file, refuses evaluation-shaped inputs, and invokes the
same Rust `code-speedup` controller with `--backend gemini --model
gemini-3.8-flash`. It starts each private workspace with the slow reference in
`candidate.py`; only success trajectories whose final bridge result is correct
and has positive reward are retained. It writes append-only output/state JSONL
and a provenance/stats JSON file, so resuming does not duplicate completed
cases. Its output copies only controller trajectory data and the sanitized
bridge result, never the source JSONL record or private oracle fields.

For higher-quota Vertex access through Application Default Credentials, pass
`--backend gemini-vertex --vertex-project YOUR_GCP_PROJECT --vertex-location
global`. This path obtains an ADC bearer token through `gcloud`; it does not use
or expose the AI Studio API key.

## SFT v2 training-only corpus (Phase 2)

Phase 2 of the SFT v2 plan expands the data **without touching any frozen v1
artifact**. The v1 manifest, both frozen eval files, and the v1 train/dev files
are never modified or trained on. Two new artifacts are produced:

* `corpus_speedup_v2.verified.jsonl` — training-only problems (`split_hint`
  `train_v2`), and
* `eval_speedup_v2_dev_behavioral.jsonl` — a fresh behavioral dev split
  (`split_hint` `dev_behavioral`) used for checkpoint selection.

Both are generated through the same schema, `public_record` boundary, and
Bubblewrap + Cachegrind verifier as v1. Known-fast sources live only in memory
(in `datagen/_v2_bank.py`) and are never serialized or shown to a model.

Build and verify the v2 corpus:

```bash
PYTHONPATH=recipes/code-speedup python3 -m datagen.gen_problems \
  --variant v2 --verify \
  --out recipes/code-speedup/datasets/corpus_speedup_v2.verified.jsonl \
  --checkpoint recipes/code-speedup/datasets/corpus_speedup_v2.verify-checkpoint.jsonl
```

Build and verify the behavioral dev split:

```bash
PYTHONPATH=recipes/code-speedup python3 -m datagen.gen_problems \
  --variant dev-behavioral --verify \
  --out recipes/code-speedup/datasets/eval_speedup_v2_dev_behavioral.jsonl \
  --checkpoint recipes/code-speedup/datasets/eval_speedup_v2_dev_behavioral.verify-checkpoint.jsonl
```

### Template diversity (Phase 2.2)

The eight seen families reach **128 distinct training templates** (~3x the v1
42): each family carries eight structurally distinct slow bodies (for / while /
comprehension / `enumerate` / `reversed` / sorted-scan shapes with varied
accumulator patterns and names), each optionally combined with a benign
distractor snippet. The underlying optimization per family is unchanged from v1;
only the surface form and the input-transform prelude literal vary, so every
instance has a distinct `normalized_ast_sha256`.

### New training-only families (Phase 2.3)

Five NEW families target the v1 failure taxonomy. **None replicates the three
frozen held-out families** (`string-concatenation`, `sort-selection`,
`indexed-lookup`); the leakage check asserts zero `family_id` overlap.

| family | transformation it teaches | answers |
| --- | --- | --- |
| `position-index` | build a value→first-index dict, then O(1) lookup per query | the "names the dict but never builds the index" failure |
| `count-index` | build a value→count dict in one pass, then O(1) lookup per query | a second, structurally distinct index-building transform |
| `membership-fusion` | hash `ys` into a set once and probe it, replacing the nested `x in ys` scan | the disallowed-import reflex (pure-subset idiom) |
| `ordered-intersection` | fuse a membership set with a seen-guard set in one pass | O(n·m) `x in ys and x not in out` scan removal |
| `mode-bucket` | tally counts into a dict (bucket/counting selection) then pick the min-value mode | repeated `xs.count(x)` re-scan removal |

`position-index` and `count-index` are index-building but structurally distinct
from the frozen `indexed-lookup` family: they operate on a single value list with
a separate query list (not `(key, value)` item pairs), build the index over `xs`
rather than over `items`, and return positions/counts rather than mapped values.

### Leakage audit (Phase 2 gate)

`datagen/leakage_check_v2.py` asserts, for every training source against the
frozen v1 seen-family eval, the frozen v1 family-held-out eval, and the new v2
behavioral dev split: zero `id`, `template_id`, and `normalized_ast_sha256`
overlap, plus zero `family_id` overlap between the new families and the three
frozen held-out families. The dev split is additionally checked disjoint from all
training data.

```bash
PYTHONPATH=recipes/code-speedup python3 -m datagen.leakage_check_v2 \
  --datasets-dir recipes/code-speedup/datasets \
  --out recipes/code-speedup/datasets/leakage_check_v2.json
```

The additive v2 manifest (`manifest_v2.json`, built via
`datagen.split_eval.v2_manifest`) records sizes, per-family and per-template
counts, the new-family rationale, and the dev-split policy — it never edits the
frozen v1 `manifest.json`.
