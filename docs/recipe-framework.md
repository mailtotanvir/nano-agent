# How to add a recipe

nano-agent is a **recipe framework**: a generic core (`core/`) plus per-domain
recipes (`recipes/<name>/`). A recipe teaches the framework how to talk to one
verifier (a compiler, linter, type checker, test runner, schema/policy validator,
...). Everything else — the repair loop, the patch engine, the model backends, the
trajectory schema, and frontier escalation — is provided for you.

This guide walks through adding a recipe. The reference implementation is
[`recipes/rust-repair`](../recipes/rust-repair/); the copy-me skeleton is
[`recipes/_template`](../recipes/_template/).

## The two-piece contract

A recipe implements exactly two things:

1. **A `Verifier`** (`nano_controller::Verifier`) — how to run your tool, parse its
   diagnostics into a `VerifyState`, read/write workspace files, and pick the
   default patch target.
2. **A `RecipeSpec`** (`nano_controller::RecipeSpec`) — metadata that bundles the
   verifier factory, the system prompt, the escalation policy, and the base config.

Then one call runs the whole loop:

```rust
let traj = nano_controller::drive(
    &recipe,            // your RecipeSpec
    workspace,          // &Path to the broken workspace
    tiny.as_ref(),      // the tiny ModelClient (proposer)
    escalation.as_deref(), // Option<&dyn ModelClient> frontier fallback
    case_id,
    traj_id,
)?;
```

## Step 1: implement the `Verifier`

The verifier is the only substantial code you write. It turns your tool's output
into the two things the loop needs: a pass/fail signal and a model-facing context
string.

```rust
use nano_controller::{Verifier, VerifyState};

impl Verifier for MyVerifier {
    fn command(&self) -> &str { "my-tool check" }

    fn verify(&self) -> anyhow::Result<VerifyState> {
        let result = run_my_tool(&self.root)?;
        Ok(VerifyState {
            passed: result.ok,
            error_count: result.errors.len(),
            primary_code: result.primary().map(|e| e.code.clone()),
            codes: result.error_codes(),
            // The context is what the model SEES. Put the rendered primary
            // diagnostic + a code excerpt here. Quality of this string is the
            // single biggest lever on repair rate.
            context: build_context(&result),
        })
    }

    fn read_file(&self, rel: &str) -> anyhow::Result<String> { /* ... */ }
    fn write_file(&self, rel: &str, content: &str) -> anyhow::Result<()> { /* ... */ }
    fn default_file(&self) -> Option<String> { /* primary diagnostic's file */ }
}
```

**Context builder tips** (from rust-repair): include the rendered primary
diagnostic verbatim, a windowed code excerpt around the error span, and a one-line
note if there are multiple errors ("fix this one first"). The tool has already
localized the fault; your job is to hand the model that localization cleanly.

## Step 2: implement the `RecipeSpec`

```rust
use nano_controller::{Config, EscalationPolicy, RecipeSpec, Verifier};

struct MyRecipe { max_attempts: u32, escalate: bool }

impl RecipeSpec for MyRecipe {
    fn name(&self) -> &str { "my-recipe" }

    fn verifier(&self, ws: &std::path::Path) -> anyhow::Result<Box<dyn Verifier>> {
        Ok(Box::new(MyVerifier::new(ws.to_path_buf())))
    }

    fn system_prompt(&self) -> String {
        // Specialize the generic contract's first line for your domain.
        MY_SYSTEM_PROMPT.to_string()
    }

    fn escalation_policy(&self) -> EscalationPolicy {
        EscalationPolicy { enabled: self.escalate, max_frontier_attempts: 1 }
    }

    fn base_config(&self) -> Config {
        Config { max_attempts: self.max_attempts, ..Default::default() }
    }
}
```

### The system prompt

`core/controller` ships a **domain-neutral** `GENERIC_SYSTEM_PROMPT` describing only
the JSON contract and the SEARCH/REPLACE block format. Your recipe overrides it
with a domain-specific prompt — typically the generic text with a specialized
first sentence ("You are a SQL query repair agent." etc.). The core never assumes
a language; the recipe owns the domain wording.

## Step 3: escalation (optional)

The framework implements the full recipe: the tiny model owns the routine cases,
and when its attempt budget is exhausted the loop can escalate to a frontier model
for the hard tail. Enable it via the recipe's `escalation_policy()` and pass a
frontier `ModelClient` as the fourth argument to `drive()`. Each frontier attempt
is recorded in the trajectory as an `Actor::Frontier` step and `trajectory.escalated`
is set true. If you pass no escalation model (or the policy is disabled), an
exhausted loop simply returns `Outcome::Escalate`.

## Step 4: wire the CLI

Copy `recipes/_template/crates/cli` — it already parses a `--path`, builds a
llama.cpp tiny model, and calls `drive()`. Add your backend/escalation flags as
needed. Model the full CLI on `recipes/rust-repair/crates/cli/src/main.rs`.

## Step 5: register in the workspace

Add your two crates to the root `Cargo.toml` `[workspace] members`, then:

```bash
cargo build --workspace
cargo run -p my-recipe-cli -- --path /some/broken/workspace
```

## Step 6: datagen + eval (for a real recipe)

To go beyond a skeleton, add:

- `datagen/` — generate labeled broken cases, each **confirmed by your real tool**
  (broken must fail with the intended error class; fixed must pass). Freeze an eval
  set first and verify it is content-disjoint from training by hash.
- `eval/` — a harness measuring per-error-class repair rate, with the tiny model
  in the loop compared against the frontier teacher and untrained baselines.

See `recipes/rust-repair/datagen/` and `recipes/rust-repair/eval/` for the pattern.

## What you never touch

The whole point of the framework: these are provided and stay unchanged across
recipes.

| Crate | What it gives you |
|---|---|
| `core/controller` | the loop, `Verifier` + `RecipeSpec` traits, `drive()`, escalation |
| `core/patch` | the SEARCH/REPLACE patch engine (fuzzy matching, apply/revert) |
| `core/model-client` | llama.cpp / Gemini / Azure backends + GBNF grammar |
| `core/trajectory` | the JSONL episode schema (research logging + SFT data) |
