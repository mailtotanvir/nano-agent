# Recipe template

A **compilable, runnable** skeleton for a new nano-agent recipe (e.g. `sql-fix`,
`tf-repair`). It builds and runs a no-op recipe out of the box, so you always have
a working baseline to edit — you fill in TODOs, you don't wire from scratch.

## Quick start

```bash
# 1. copy the skeleton
cp -r recipes/_template recipes/my-recipe

# 2. rename the crates (edit each Cargo.toml: nano-template-* -> my-recipe-*)
#    and add the two crates to the root Cargo.toml [workspace] members.

# 3. it already builds and runs (no-op verifier => immediate success):
cargo run -p nano-template-cli -- --path /some/dir
# outcome=Success attempts=0 escalated=false
```

## What you implement (4 TODOs, all marked in the code)

Everything else is reused from `core/` unchanged.

1. **`TemplateVerifier::verify`** (`crates/verifier/src/lib.rs`, TODO 1) — run YOUR
   tool against the workspace, parse its diagnostics into a `VerifyState`, and
   build the model-facing `context` string from the primary failure. This is the
   only substantial code you write.
2. **`Verifier::command`** (TODO 2) — the human-readable tool name.
3. **`Verifier::default_file`** (TODO 3) — the file a patch targets when a block
   omits its own hint (usually the primary diagnostic's file).
4. **`RecipeSpec::system_prompt`** (`crates/cli/src/main.rs`, TODO 4) — a
   domain-specific system prompt (specialize the generic contract's first line for
   your tool).

The reference implementation to model yours on is
[`recipes/rust-repair`](../rust-repair/).

## What you reuse unchanged (from `core/`)

- `core/controller` — the bounded verify→propose→apply→re-verify→escalate loop,
  the `Verifier` and `RecipeSpec` traits, and the generic `drive()` entrypoint.
- `core/patch` — the SEARCH/REPLACE patch engine.
- `core/model-client` — llama.cpp (tiny) + Gemini/Azure (frontier) backends + the
  GBNF JSON grammar.
- `core/trajectory` — the JSONL episode schema and logger.

## The `RecipeSpec` contract

A recipe is one trait implementation:

```rust
impl RecipeSpec for MyRecipe {
    fn name(&self) -> &str { "my-recipe" }
    fn verifier(&self, ws: &Path) -> Result<Box<dyn Verifier>> { /* your tool */ }
    fn system_prompt(&self) -> String { /* domain prompt */ }
    fn escalation_policy(&self) -> EscalationPolicy { /* default: off */ }
    fn base_config(&self) -> Config { /* max_attempts, etc. */ }
}
```

Then `drive(&recipe, workspace, tiny_model, escalation_model, case_id, traj_id)`
runs the whole loop.

## Checklist for a production recipe

- [x] `crates/verifier` implementing `Verifier` (skeleton provided)
- [x] `crates/cli` binary implementing `RecipeSpec` + calling `drive()` (skeleton provided)
- [ ] Replace the 4 TODOs with your tool's logic
- [ ] `datagen/` producing labeled, verifier-confirmed broken cases
- [ ] `eval/` harness with a per-error-class competence table
- [ ] `README.md` recipe card (problem, envelope, model, how-to-run)
- [ ] add crates to the root `Cargo.toml` workspace members
