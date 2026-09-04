# Recipe: rust-repair

A tiny verified agent that repairs Rust **compilation** failures.

## Problem

`cargo check` fails. A bounded class of these errors (type mismatches, missing
imports, borrow/move issues, missing derives/bounds, trivial syntax) are
structurally identifiable and fixable without frontier-level reasoning — given
the structured diagnostic, the relevant source, and compiler feedback after each
attempt.

## Competence envelope (measured, not claimed)

The eval harness reports repair rate **per error code**, so the envelope is an
explicit table of which `E####` codes the tiny agent owns vs. must escalate.
Populated in Phase 1+. Until then this section is intentionally empty — we do not
publish headline numbers before the eval set is trustworthy.

## Model

- Primary: `Qwen2.5-Coder-0.5B-Instruct` (Apache-2.0, code-pretrained, GGUF on
  llama.cpp / ARM64).
- Sweep: SmolLM2-135M, SmolLM2-360M, Qwen3-0.6B, Qwen2.5-Coder-1.5B (control).
- Teacher / fallback: Gemini 2.5/3.x Flash.

## How it runs

1. `cargo check --message-format=json` → structured diagnostics.
2. Build context: primary diagnostic (rendered) + ±20-line source window.
3. Tiny model proposes a JSON `{action, patch, reason, confidence}`, with the
   patch as SEARCH/REPLACE blocks (GBNF-constrained so JSON is always valid).
4. Apply, re-verify. Regression guard reverts any patch that increases the error
   count. Up to `--max-attempts` (default 4), then escalate.

## Layout

```text
recipes/rust-repair/
  crates/verifier/   cargo/rustc JSON → diagnostics (implements Verifier)
  crates/cli/        `rust-repair` binary + e2e tests
  proposal.gbnf      grammar constraining model output
  datagen/           synthetic breakage + teacher trajectory pipeline
  training/          SFT configs per base model
  eval/              4-arm harness + per-E-code envelope tables
  datasets/          versioned manifests (data on HF Hub / OCI disk)
  experiments/       one config + results.json per run
```

## Run

```bash
cargo build --release -p rust-repair-cli
./target/release/rust-repair --path <crate-dir> --backend llama \
    --model qwen2.5-coder-0.5b-instruct --log trajectories.jsonl
# exit 0 = fixed, 2 = not fixed (escalate/no_fix/budget)
```

## Status

Phase 0 complete: harness (controller, verifier, patch engine, trajectory log,
CLI) builds and passes unit + end-to-end tests that repair a real broken crate.
Next: datagen + Gemini baseline, then the tiny-model sweep.
