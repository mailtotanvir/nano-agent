# v6 teacher-expanded Qwen 1.5B LoRA — training result (2026-09-22)

## Outcome

The approved one-epoch v6 SFT run completed successfully on an on-demand GCP
L4. This records training metrics only; it does not establish behavioral or
frozen-evaluation performance and therefore does not promote the adapter.

## Inputs and method

- Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct`; method: LoRA; seed `7`.
- Corpus: 666 examples, SHA-256
  `21bad5b2e3597d7e83b2fd8d05d23d8811e831f77f1869a44b848e23cfe3b717`.
- Split/replay: 595 unique training examples plus 329 v1 replay examples
  (822 effective training examples), 71 evaluation examples, requested v1
  replay ratio `0.4`.
- Hardware: one NVIDIA L4 on `g2-standard-8`, `us-east4-a`.

## Result

| Metric | Value |
| --- | ---: |
| Optimizer steps | 52 |
| Trainer runtime | 366.0971 s |
| Final train loss | 0.3361113014 |
| Final evaluation loss | 0.2585840225 |
| Final evaluation token accuracy | 0.9367560943 |

The initially installed image had an ABI-incompatible optional `torchaudio`
package. It failed before any optimizer step; the unused package was removed,
imports were verified (`torch 2.9.1+cu129`, `transformers 5.17.0`, `peft
0.21.0`, `trl 1.13.0`), and the run was restarted successfully.

## Artifact and teardown

- Local archive:
  `artifacts/code-speed-sft-v6-20260922/out_code-speed-sft-v6-qwen15b.tar`.
- Remote and local archive SHA-256 match:
  `9940ddbbaf486250eac2c72c7e1f3b7f2e7cc85387df94bfc8f21aeaf4f50205`.
- The archive contains the final adapter, tokenizer, metrics, and checkpoints
  20, 40, and 52. Local integrity evidence (`input.sha256`, `artifact.sha256`,
  `archive.sha256`, contract report, and train log) is stored alongside it.
- The training VM was deleted after transfer. Final GCP
  checks returned `not-found` for the VM and empty instance, disk, and reserved
  address inventories.

The $1.75 authorization ceiling and the quoted `$0.856856553/hour` rate are
recorded in the public paper. A billing export is not yet available,
so this report intentionally does not claim a final billed amount.

## Next gate

The behavioral-development evaluation is recorded in
`V6_BEHAVIORAL_EVALUATION_2026-09-23.md`. It scored 12/24 versus v5's 23/24;
v6 was not promoted or evaluated on the frozen sets. The next experiment is a
v6.1 mixture restoring the v5 invariant family weights while retaining the
verified v6 teacher trajectories.
