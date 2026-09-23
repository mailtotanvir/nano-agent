# v5 and v6.1 adapter release manifest

The user chose **two equal experiment variants in one new code-speed model
repository**. Both are PEFT LoRA adapters for the same
`Qwen/Qwen2.5-Coder-1.5B-Instruct` base, not merged base weights. The exact
repository is `mailtotanvir/nano-agent-code-speed-1.5b`. Stage the adapters in explicit `v5/` and `v6p1/`
subdirectories; do not put either adapter at the repository root in a way
that makes it look like the default or hides the other. Use a joint root
`README.md` based on `JOINT_MODEL_CARD_DRAFT.md` with a balanced results table
and precise loading instructions.

| Repository path | SHA-256 |
| --- | --- |
| `v5/adapter_config.json` | `2c94cf237609aaee50af8bd309c876eafa2449466ae61df38a6022f00d423d1f` |
| `v5/adapter_model.safetensors` | `55e80adb158cfc1f382b8b06862e25f6b9755d423b46647d6fad9bfde261bcc3` |
| `v6p1/adapter_config.json` | `af345afe8728c861c7161678aa1d41d50ea65e7f3e9db901c28224e731b92de6` |
| `v6p1/adapter_model.safetensors` | `f1414066654c7728c3b8629b610f82bb6d428fba9892a38001a037be14916a85` |
| `chat_template.jinja` | `14add2c91e5ccf42b33faa79ef1c0afbd41e9408d5168f7d0ac846ed2ee932c5` |
| `tokenizer.json` | `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` |
| `tokenizer_config.json` | `8b6f658e2435ec8da6866013cb485a1d2a6559bed4bf76ed99ea68b031854b41` |

The local sources are the gitignored extractions at
`artifacts/code-speed-sft-v5-20260920/extracted/code-speed-sft-v5-qwen15b/`
and
`artifacts/code-speed-sft-v6p1-20260923/extracted/code-speed-sft-v6p1-qwen15b/`.
The shared chat template and tokenizer files have identical hashes in both
extractions, so one root copy is sufficient. Do not upload either
trainer-generated `README.md`: each has an unrelated sample prompt, a `None`
model path, and an invalid license field. Do not upload intermediate
checkpoints or `training_args.bin`. Verify uploaded file hashes independently.

Training archive SHA-256 values: v5
`a4a402fe8812bd50d808be0ee40c4e3a58e78892fd48d235904fac013eecdb6b`;
v6.1
`9db729bee2f2e52489424216a68f1d9b1cb2cc29f2dd2e84a1a9850d252e67a3`.
Both archives are retained locally. Both GCP training VMs and their boot
disks were deleted after archive transfer and hash verification.
