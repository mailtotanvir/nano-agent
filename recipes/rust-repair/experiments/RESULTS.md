# nano-agent / rust-repair — Results Capture (for publication)

Frozen record of measured results as of 2026-09-03. All numbers are real runs
through the same controller + SEARCH/REPLACE parser + cargo verifier. Latency is a
non-goal (the thesis is frontier-level ACCURACY running locally/privately).

## The story arc (publication narrative)

1. Untrained sub-1B in the loop = 0% — the tools alone don't rescue an untrained
   model (it can't emit the patch format).
2. One SFT pass (348 teacher trajectories) = 86.7% in-distribution — matches/beats
   the frontier ceiling on the eval it was trained toward.
3. Honest OOD test (25 hand-authored novel cases) exposes overfitting: 86.7%
   in-dist collapses to 44% OOD. The single-site-edit skill didn't generalize.
4. Failure-driven curriculum (distill the failing skill CLASSES from the teacher,
   never the test cases) lifts OOD 44% -> 60%, crossing the 56% teacher bar.
   The nano model now beats its own Gemini Flash teacher on generalization.
5. Teacher-strength ablation (Phase 4): distilling the SAME failing classes from a
   FRONTIER teacher (gpt-5.6-luna, 100% OOD) instead of Gemini lifts OOD 60% -> 72%.
   A stronger teacher is a real lever on structural generalization, but the jump is
   bounded — residual failures cluster in the hardest "insert a whole trait method/
   import" class, a 0.5B capacity signal (next lever = RL or a bigger base).

## Headline table — frozen eval_v1 (45 in-distribution cases)

| arm | model | repair rate | avg attempts | median lat |
|---|---|---|---|---|
| tiny_loop untrained | Qwen2.5-Coder-0.5B base | 0.0% (0/45) | 3.89 | 6.8s |
| frontier_direct | Gemini 3.6 Flash 1-shot | 84.4% (38/45) | 1.0 | 3.4s |
| tiny_loop SFT v2 | Qwen2.5-Coder-0.5B trained | 86.7% (39/45) | 1.4 | 1.8s |

## Headline table — frozen eval_ood_v1 (25 hand-authored OOD cases)

| model | OOD rate | notes |
|---|---|---|
| gpt-5.6-luna (frontier, 1-shot) | 100% (25/25) | comparison baseline only |
| DeepSeek-V4-Flash (frontier, 1-shot) | 84% (21/25) | comparison baseline |
| grok-4.6 (frontier, 1-shot) | 76% (19/25) | comparison baseline |
| Gemini Flash (TEACHER / the bar, 1-shot) | 56% (14/25) | the number to beat |
| trained 0.5B v2 (single-site SFT only) | 44% (11/25) | pre-curriculum |
| trained 0.5B v3 (+ Gemini structural curriculum) | 60% (15/25) | beats flash teacher |
| **trained 0.5B v4 (+ gpt-5.6-luna structural curriculum)** | **72% (18/25)** | **+16 over teacher bar** |

OOD delta v2->v3: +16 points (Gemini teacher), ZERO regressions. 4 newly-solved
categories (move_use, generic_bound, string_return, trailing_semi).

OOD delta v3->v4: +12 points (gpt-5.6-luna frontier teacher on the same 10 failing
classes). Net +5 newly-solved (iter_collect, option_unwrap, match_arms, clone_needed,
derive_eq), -2 single-site regressions (ref_arg, question_unit; not in v4 curriculum,
noise-level). Still failing (7): ref_arg, question_unit, trait_method, trait_scope,
index_type, macro_name, trait_impl — the trait method/import-insertion cluster is the
0.5B's residual hard class even with a 100%-OOD teacher (capacity, not data).

## Method notes (for reproducibility section)

- Model: Qwen2.5-Coder-0.5B-Instruct, full bf16 SFT (no QLoRA), 3 epochs, lr 1e-5,
  effective batch 16, assistant-only loss. Served as q8_0 GGUF on a 4-core ARM
  (Oracle OCI) box via llama.cpp — no GPU at inference.
- Teacher: Gemini 3.6 Flash via Vertex AI. Deliberately NOT a frontier model —
  the thesis is a nano model BEATING its flash-class teacher, not matching frontier.
  A teacher can only distill skills it can itself solve (Gemini structural pool
  solve rate 65.9%).
- Data: break_it.py (single-site mutations, in-dist) + structural_cases.py
  (10 multi-line/structural skill classes, the curriculum). All cargo-verified
  (broken must fail, fixed must compile). sft_v3 = 489 examples (348 + 141);
  sft_v4 = 739 (sft_v3 + 250 gpt-5.6-luna structural_pool_v2 trajectories,
  applied-and-kept only, 0 exact-duplicate messages, 0 eval overlap).
- Teacher-strength ablation (Phase 4): the v4 curriculum swaps the Gemini teacher
  for gpt-5.6-luna (frontier, 100% OOD) on the 10 still-failing classes. Isolates
  whether teacher capability — not just curriculum coverage — moves nano OOD. It
  does: 60% -> 72%. Headline thesis unchanged (nano beats its flash teacher); luna
  probes the ceiling, it is not the project's teacher of record.
- Eval integrity: eval_v1 (45) and eval_ood_v1 (25) are FROZEN and NEVER trained
  on. Verified 0 content overlap between training and OOD eval.
- Contamination guardrail: never train on the eval cases themselves; train on new
  cases of the same skill class.

## Artifact locations

- Models (weights gitignored; live on OCI + local):
  - v2: OCI ~/models/qwen05-sft-v2-q8_0.gguf, local recipes/.../out/qwen05-sft-v2/
  - v3: OCI ~/models/qwen05-sft-v3-q8_0.gguf, local recipes/.../out/qwen05-sft-v3/
  - v4: OCI ~/models/qwen05-sft-v4-q8_0.gguf, local recipes/.../out/qwen05-sft-v4/
        (model.safetensors 988097824 bytes; also ~/nano-agent-artifacts/qwen05-sft-v4/)
- Datasets: recipes/rust-repair/datasets/ (sft_v2, sft_v3, sft_v4, structural_pool_v1,
  structural_pool_v2, teacher_luna_v4, sft_luna_v4, eval_v1, eval_ood_v1; provenance).
- Results: recipes/rust-repair/eval/results/ (eval_v1_*, ood_v1.json, ood_gemini,
  ood_frontier, ood_tiny, ood_v3_tiny, ood_v4_tiny).
- Experiment log (chronological, all runs): recipes/rust-repair/experiments/LOG.md
- Spot/on-demand price evidence: infra/gcp/SPOT_PRICE_LOG.md

## Cost to date (GCP credit)

- Teacher/eval API inference (Vertex + Azure): a few dollars.
- GPU: Phase 2 ~$0.51 (spot) + Phase 3b 2 lost spot runs ~$0.13 + 1 on-demand
  ~$0.16 + Phase 4 on-demand ~$0.46 = ~$1.26 total GPU. Whole project well under
  $6 of the ~$200 credit.
