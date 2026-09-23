# v6.1 invariant weighted Qwen 1.5B LoRA — training result (2026-09-23)

The approved one epoch SFT run completed on one on demand L4 in `us-east4-a`.
The 1,170 example corpus SHA-256 was
`fc9f92f2a3dcd5a2a0cf3dd2667ebdfa47b0cc842a73314a2d9364cc9f1ad4d2`.
The remote contract probe passed all rows before training.

| Metric | Value |
| --- | ---: |
| Optimizer steps | 99 |
| Trainer runtime | 726.5452 s |
| Train loss | 0.2547310320 |
| Evaluation loss | 0.1823733300 |
| Evaluation token accuracy | 0.9518498108 |
| Unique / effective training examples | 1,047 / 1,575 |
| Evaluation examples | 123 |

The artifact archive and its integrity files are gitignored under
`artifacts/code-speed-sft-v6p1-20260923/`. The remote and local archive
SHA-256 matched:
`9db729bee2f2e52489424216a68f1d9b1cb2cc29f2dd2e84a1a9850d252e67a3`.
The final adapter file SHA-256 is
`f1414066654c7728c3b8629b610f82bb6d428fba9892a38001a037be14916a85`.

After verified transfer, the training VM was
deleted. Final GCP checks reported `not-found` for the VM and `[]` for the
instance, disk, and reserved address inventories. The quote and authorization
ceiling are summarized in the paper; final billing export is pending.

Training metrics are not the promotion gate. The same 24 case behavioral
development evaluation used for v5 and v6 completed on the existing OCI CPU
host: v6.1 scored **21/24**, compared with v6 **12/24** and v5 **23/24**.
The temporary OCI model server was stopped after the report was copied and
verified. See `V6P1_BEHAVIORAL_EVALUATION_2026-09-23.md` for the development
failures. The user subsequently authorized measurement-only frozen evaluation;
see `V6P1_FROZEN_EVALUATION_2026-09-23.md` for those results. v5 remains the
development-selected checkpoint.
