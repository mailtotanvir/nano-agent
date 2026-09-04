# Recipe template

Skeleton for a new nano-agent recipe (e.g. `sql-fix`, `tf-repair`). Copy this
directory to `recipes/<name>/` and implement the pieces that are recipe-specific.

## What you implement

Only two things are recipe-specific; everything else is reused from `core/`:

1. **A `Verifier`** (implement `nano_controller::Verifier`): how to run your
   tool, parse its diagnostics into `VerifyState`, and read/write workspace files.
   Model this on `recipes/rust-repair/crates/verifier`.
2. **A context builder**: turn the primary diagnostic + relevant source into the
   string the model sees (in your CLI's `Verifier::verify`).

## What you reuse unchanged

- `core/controller` — the bounded verify→propose→apply→escalate loop.
- `core/patch` — the SEARCH/REPLACE patch engine.
- `core/model-client` — llama.cpp + Gemini backends + the GBNF JSON grammar.
- `core/trajectory` — the JSONL episode schema and logger.

## Checklist

- [ ] `crates/verifier` implementing `Verifier` for your tool
- [ ] `crates/cli` binary `nano-agent <name> fix <path>`
- [ ] `datagen/` producing labeled, verifier-confirmed broken cases
- [ ] `eval/` 4-arm harness with a per-error-class envelope table
- [ ] `README.md` recipe card (problem, envelope, model, how-to-run)
- [ ] add crates to the root `Cargo.toml` workspace members
