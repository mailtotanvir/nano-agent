# Code-speed 1.5B claim–evidence matrix

Release-candidate control document. This follows the claim/scope discipline
of the integrated-study paper supplied as a presentation reference. Do not
upgrade any claim without new evidence and an explicit release review.

| Proposed public claim | Evidence | Scope and publication status |
| --- | --- | --- |
| v5 passed 23/24 development; v6 passed 12/24; v6.1 passed 21/24 | `experiments/V5_BEHAVIORAL_EVALUATION_2026-09-21.md`, `V6_BEHAVIORAL_EVALUATION_2026-09-23.md`, `V6P1_BEHAVIORAL_EVALUATION_2026-09-23.md` | One greedy proposal/case through the production verifier. Development was used for checkpoint choice. Supported as a development result, not generalization. |
| v6's regression coincided with removal of focused invariant copies; v6.1 restored focused weight and recovered most development successes | Corpus composition and training-result files; behavioral reports above | Observational comparison. Corpora and training runs differ; no causal claim that weighting alone caused the change. |
| v5 and v6.1 each passed 24/30 frozen seen-family cases | `experiments/V5_BEHAVIORAL_EVALUATION_2026-09-21.md` (v5 report SHA `90b68a23e8c57f8b40f5c08a2ddd16fe78554cc208cba6eb7d50906e6a80ce82`); `experiments/V6P1_FROZEN_EVALUATION_2026-09-23.md` (v6.1 full-report SHA `00a41526db5fac9e963e6c41a4bd59a1888ba2fcd3ce2e2266a7d384f65b7dd2`) | Different case-level successes despite equal totals. Measurement-only after v5 development selection. |
| v5 passed 14/30, v6.1 16/30 frozen family-heldout | Same experiment records; v5 report SHA `200eaf01e049e87843fb9426044d57f8684c7c9e73d5dea5121c15df5b9cabf4`, v6.1 full-report SHA `adc0066bde045e9622654de6dca8bc16d2c7c42ea30f9d2257eb64e5a09ec383` | Two-case descriptive gain. v6.1 8/10 string concatenation, 8/10 sort selection, 0/10 indexed lookup; v5 7/10, 7/10, 0/10. Not broad transfer. |
| v6.1 failed 0/3 fresh residue-count-index cases after dropping `bucket_size` | `experiments/V6P1_FROZEN_EVALUATION_2026-09-23.md`; full-report SHA `05a66dab3d26b7fbe2266e41eb1913b5faac74d59df5fb68fe4e08874d68b1c1` | Exploratory diagnostic only; v5 not measured. All three `candidate_error`, grouped as semantic failure. Do not treat 0/3 as a stable rate. |
| The verifier requires hidden-input correctness and positive Cachegrind instruction-reference reward | Controller/verifier source and frozen protocol | Finite tests, restricted pure Python. `$Ir$` is simulated instruction references, not runtime or proof of equivalence. |
| Two adapters are available as equal experiment variants | Pre-upload manifest with adapter SHA values; public HF hashes to be checked after upload | Release intent only until public upload. Equal visibility does not imply equal measured performance. |
| Training artifacts were extracted and GCP VMs torn down | Training-result records and archive hashes | Verified local extraction/teardown. Price `$0.856856553/hour` is a quote; actual billed total pending. |

## Claims deliberately excluded

- No claim of optimizing arbitrary Python or other languages.
- No claim of formal equivalence or real-world wall-clock speedup.
- No claim that v6.1 is globally better than v5, or that frozen totals select a new checkpoint.
- No claim that focused repetition alone caused the v6 regression/recovery.
- No aggregate obtained by pooling the two frozen 30-case tracks or the three-case diagnostic.
- No claim of a measured total cloud bill, public DOI, or released HF model before verification.

## Figure data provenance

| Publication visual | Plotted values | Source and guardrail |
| --- | --- | --- |
| Blog Figure 2 / paper Figure `development` | v5 23/24, v6 12/24, v6.1 21/24 | Three behavioral development reports above. Same 24 cases, one proposal each; one training run/configuration. |
| Paper Figure `frozen` | Seen-family 24/30 and 24/30; family-heldout 14/30 and 16/30 | v5 and v6.1 frozen experiment reports. Two different splits; no pooled 60-case claim. |
| Blog Figure 3 / paper Figure `heldoutfamilies` | String 7/10 and 8/10; sort 7/10 and 8/10; indexed 0/10 and 0/10 | Frozen family-heldout tables in `V6P1_FROZEN_EVALUATION_2026-09-23.md`. Bars are case counts, not confidence intervals. |
| Paper Figure `failures` | Seen: v5 4 semantic + 2 no-speedup; v6.1 5 + 1. Heldout: v5 15 + 1; v6.1 13 + 1 | v5 behavioral report and v6.1 frozen report. Failure categories are evaluator labels; candidate errors are included under semantic failure. |
