# nano-agent — Architecture

This document describes how nano-agent is put together: the generic core, the
recipe boundary, and how one repair episode flows end to end. It reflects the
codebase as of v0.1.x, with a final section on the v2.0 recipe-framework direction
(see [`docs/v2-plan.md`](docs/v2-plan.md)).

## The thesis, in one line

A large fraction of bounded, structurally identifiable failures can be repaired by
a **tiny model inside a verified loop**, at a fraction of the cost and latency of a
frontier call. The model is one component; the recipe — tool + verifier + patch
engine + bounded retry + escalation — is the product.

The compiler (or linter, or type checker, or policy engine) is the **verifier**.
The loop supplies the reliability the tiny model lacks. The frontier model is the
teacher (offline, for training data) and the fallback (online, for the hard tail)
— not the runtime dependency for ordinary cases.

## Workspace layout

```
core/
  controller     the generic verify -> propose -> apply -> re-verify loop
  patch          language-agnostic SEARCH/REPLACE patch engine
  model-client   ModelClient trait + llama.cpp / Gemini / Azure backends + GBNF grammar
  trajectory     the JSONL episode schema (research logging AND SFT data)
recipes/
  rust-repair    the reference recipe: cargo check as the verifier
  _template       skeleton to copy for a new recipe
```

Cargo workspace members are declared in the root `Cargo.toml`. Everything under
`core/` is domain-neutral; everything domain-specific lives under `recipes/<name>/`.

## The core crates

### `core/controller` — the loop

Owns the deterministic control loop and defines the recipe boundary. It never
knows which language or tool it is repairing.

- **`trait Verifier`** — the recipe boundary. A recipe implements:
  - `command() -> &str` — human-readable tool name (e.g. `"cargo check"`).
  - `verify() -> VerifyState` — run the tool, summarize pass/fail + primary error
    + a model-facing context string.
  - `read_file(rel)` / `write_file(rel, content)` — workspace file access.
  - `default_file()` — fallback target when a patch block omits its file hint.
- **`struct VerifyState`** — `{ passed, error_count, primary_code, codes, context }`.
  The `context` string is what the model actually sees.
- **`struct Config`** — `{ max_attempts, patch_limits, regression_guard,
  system_prompt, actor }`.
- **`fn run_episode(verifier, model, cfg, case_id, traj_id) -> Trajectory`** —
  runs one bounded episode, mutating the workspace through the verifier.

### `core/patch` — the edit engine

- **`parse_blocks(text, default_file)`** — parse SEARCH/REPLACE blocks out of the
  model's `patch` field into `EditBlock`s. Chosen over unified diff because sub-1B
  models mangle diff headers and line arithmetic.
- **`apply_block(orig, block, limits) -> ApplyOutcome`** — apply one block, with
  bounded fuzzy matching (`ApplyOutcome.fuzzy` records whether fuzz was used).
- **`PatchLimits`** — guards on block size / fuzz radius.

### `core/model-client` — the proposer

- **`trait ModelClient`** — `propose(&[Message]) -> ModelResponse`, `name()`.
- **`parse_proposal(text) -> Proposal`** — parse the contract JSON
  `{action, patch, reason, confidence}`.
- Backends: **`LlamaCppClient`** (tiny, local, CPU, OpenAI-compatible server, with
  optional GBNF grammar to force valid contract JSON), **`GeminiClient`**
  (AI Studio key or Vertex ADC), **`AzureClient`** (OpenAI-compatible frontier).
- **`PROPOSAL_GBNF`** — the grammar that constrains sub-1B output to valid JSON.

### `core/trajectory` — the episode record

- **`struct Trajectory`** — the full episode: initial error codes, an ordered list
  of `Step`s, outcome, attempts, total latency.
- **`struct Step`** — one attempt: `{ attempt, actor, primary_code, error_count,
  context, proposal, patch_applied, patch_fuzzy, error_count_after, reverted,
  model_latency_ms, tokens_out }`.
- **`enum Outcome`** — `Success | NoFix | Escalate`.
- **`enum Actor`** — `Tiny | Frontier` (provenance: who produced this step).
- **`TrajectoryLog`** — append-only JSONL writer. The same schema is used for
  research logging AND as the source format for SFT data.

## One repair episode, end to end

```
                        +-----------------------------+
  broken workspace ---> |  Verifier.verify()          |
                        |  (e.g. `cargo check`)       |
                        +--------------+--------------+
                                       |
                          passed? -----+----- yes ---> Outcome::Success
                                       | no
                                       v
                        +-----------------------------+
                        |  build context from the     |
                        |  primary diagnostic + code  |
                        +--------------+--------------+
                                       v
                        +-----------------------------+
                        |  ModelClient.propose()      |
                        |  -> {action, patch, ...}    |
                        +--------------+--------------+
                                       v
              action == "escalate" ----+---- Outcome::Escalate
              action == "no_fix"   ----+---- Outcome::NoFix
                                       | action == "patch"
                                       v
                        +-----------------------------+
                        |  parse_blocks + apply_block |
                        |  (snapshot files first)     |
                        +--------------+--------------+
                                       v
                        +-----------------------------+
                        |  Verifier.verify() again    |
                        |  regressed? -> revert       |
                        +--------------+--------------+
                                       |
                     passed? --- yes --+--> Outcome::Success
                                       | no, budget left -> loop
                                       | no, budget spent -> Outcome::Escalate
                                       v
                                (escalation: see v2.0)
```

Key invariants:

- **Snapshot-and-revert.** Every touched file is snapshotted before a patch is
  applied; if the patch fails to apply or *increases* the error count
  (`regression_guard`), the workspace is reverted. The loop never leaves a
  workspace worse than it found it.
- **Same loop for teacher and tiny.** The frontier teacher that generates SFT data
  runs through the *exact* same controller + parser + verifier as the tiny model,
  so teacher trajectories are shape-identical to tiny-model runs and every step is
  compiler-verified.
- **Bounded budget.** `max_attempts` caps the tiny loop; exhaustion yields
  `Outcome::Escalate`.

## The recipe boundary

A recipe supplies exactly two things and reuses everything in `core/` unchanged:

1. **A `Verifier`** — how to run the tool, parse its diagnostics into
   `VerifyState`, and read/write workspace files.
2. **A context builder** — how to turn the primary diagnostic + relevant source
   into the string the model sees (implemented inside the recipe's `Verifier::verify`).

Everything else — the loop, the patch engine, the model backends, the trajectory
schema — is provided by `core/`.

The reference implementation is `recipes/rust-repair`: a `Verifier` wrapping
`cargo check`, a per-recipe CLI (`rust-repair`) that wires the verifier + a chosen
model backend into `run_episode`, and `datagen/` + `eval/` for producing training
data and measuring per-error-class competence.

## Design decisions worth knowing

- **SEARCH/REPLACE, not unified diff.** Tiny models reliably corrupt diff headers
  and miscount hunk line numbers. A "find this exact block, replace with that
  block" format removes line arithmetic entirely.
- **Grammar-constrained output.** A GBNF grammar forces the tiny model to emit
  valid contract JSON, so the parser can never be handed a malformed shape.
- **The verifier is a free, perfect oracle.** It cannot write a fix but can judge
  one instantly without hallucinating. Pairing a cheap proposer with a perfect
  judge is the core leverage.
- **Frozen evals, hash-checked.** Eval sets are frozen and verified content-disjoint
  from training data by case-id hash. Training never touches the test.

---

## v2.0 direction — the recipe framework

v0.1.x proved the thesis with one recipe. v2.0 generalizes the codebase into a
genuine **recipe framework**: a provably-generic core where adding a recipe is
turnkey, without yet building a second domain recipe. Full task breakdown in
[`docs/v2-plan.md`](docs/v2-plan.md). The four changes:

1. **Domain-neutral core prompt.** Today `core/controller` ships a
   `DEFAULT_SYSTEM_PROMPT` that literally says "You are a Rust compiler-error
   repair agent" — the one real Rust leak in the "recipe-agnostic" core. v2.0 makes
   the generic prompt domain-neutral (JSON contract + SEARCH/REPLACE format only)
   and moves the Rust wording into the recipe.

2. **Recipe-configurable escalation, actually wired.** Today `run_episode` returns
   `Outcome::Escalate` with `escalated = false` and a comment that the caller
   "wires actual frontier fallback" — but no caller does. v2.0 adds an
   `EscalationPolicy` (enabled + max frontier attempts) and
   `run_episode_with_escalation`, so an exhausted tiny loop runs a real frontier
   attempt, recorded truthfully as `Actor::Frontier` steps with
   `trajectory.escalated = true`. This completes the "tiny owns the routine 95%,
   frontier for the hard 5%" story end to end.

3. **A `RecipeSpec` trait.** Today each recipe hand-wires verifier + model + config
   + actor in its own `main.rs`. v2.0 introduces one trait a recipe implements —
   `name`, `verifier`, `system_prompt`, `escalation_policy`, `base_config` — plus a
   generic `drive(spec, workspace, tiny, escalation_model, ...)` entrypoint. Adding
   a recipe becomes "implement one trait," not "copy a `main.rs`."

4. **A compilable `_template`.** Today `recipes/_template` is a README only. v2.0
   ships a skeleton that *builds and runs a no-op recipe out of the box*, so
   `cp -r recipes/_template recipes/foo` (+ add to the workspace) gives a newcomer
   working stubs to edit rather than prose to interpret.

Per-recipe binaries stay (no unified dispatcher). No second domain recipe, no
retraining, no packaging — those are separate follow-ups. The regression bar for
v2.0 is strict: rust-repair's frozen-eval outcomes must stay byte-identical, since
this is a refactor of structure, not a change of behavior.
