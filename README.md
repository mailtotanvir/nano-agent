# nano-agent

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22293159.svg)](https://doi.org/10.5281/zenodo.22293159)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-nano--agent--rust--repair--0.5b-yellow)](https://huggingface.co/mailtotanvir/nano-agent-rust-repair-0.5b)
[![Blog](https://img.shields.io/badge/Read-the%20write--up-46E39B)](https://mailtotanvir.github.io/nano-agent/blog.html)

A cookbook of **tiny, verified repair agents**. Each recipe pairs a small
open-weight model (0.1B–1.5B) with a deterministic controller that loops a real
tool's verifier — compile, diagnose, patch, re-verify — and escalates to a
frontier model only when its bounded budget is exhausted.

The thesis (see [`docs/rust-tiny-agent-spec.md`](docs/rust-tiny-agent-spec.md)):

> A large fraction of bounded, structurally identifiable failures can be repaired
> by a tiny model **inside a verified loop**, at a fraction of the cost and
> latency of a frontier call. The model is one component; the recipe (tool +
> verifier + patch engine + bounded retry + escalation) is the product.

The compiler (or linter, or policy engine) is the verifier. The loop supplies the
extra intelligence. The frontier model is the teacher and the fallback — not the
runtime dependency for ordinary cases.

## Recipes

| Recipe | Domain | Verifier | Status |
|---|---|---|---|
| [`rust-repair`](recipes/rust-repair/) | Rust compile errors | `cargo check` | Phase 0 (harness) |
| `_template` | — | — | skeleton for new recipes |

Planned: `sql-fix` (query validation), `tf-repair` (Terraform validate/plan).

## Architecture

```text
                 Repair Recipe
                      │
     ┌────────────────┼────────────────┐
  Tiny Model      Tool Verifier     Controller
     │                │                │
     └────────────────┼────────────────┘
                 Bounded Retry Loop
                      │
                  Escalation
```

Shared, recipe-agnostic machinery lives in [`core/`](core/):

- `core/controller` — the generic bounded verify→propose→apply loop + the
  `Verifier` / `ModelClient` traits. It does not know which language it repairs.
- `core/patch` — language-agnostic SEARCH/REPLACE patch engine (chosen over
  unified diff because tiny models mangle diff headers).
- `core/model-client` — `ModelClient` trait with llama.cpp (tiny, CPU) and
  Gemini (frontier teacher/fallback) backends; ships a GBNF grammar that forces
  valid contract JSON out of sub-1B models.
- `core/trajectory` — the JSONL episode schema used for both research logging and
  SFT data.

Each recipe supplies only its verifier + context builder and reuses everything
above unchanged.

## Quick start (rust-repair)

```bash
# build
cargo build --release

# fix a broken crate with the tiny model (llama.cpp server on :8080)
./target/release/rust-repair --path ./broken-crate --backend llama \
    --model qwen2.5-coder-0.5b-instruct

# or use the Gemini teacher / fallback
GEMINI_API_KEY=... ./target/release/rust-repair --path ./broken-crate \
    --backend gemini --model gemini-2.5-flash
```

Every run appends a full trajectory to `trajectories.jsonl` (initial diagnostics,
each proposal + applied patch + re-verify result, terminal outcome).

## Development

```bash
cargo test --workspace   # unit + e2e (e2e needs cargo on PATH)
cargo fmt --check
cargo clippy --workspace -- -D warnings
```

## Paper & citation

The accompanying technical report is in [`paper/nano-agent.pdf`](paper/nano-agent.pdf)
(read it in your browser: [mailtotanvir.github.io/nano-agent/paper/nano-agent.pdf](https://mailtotanvir.github.io/nano-agent/paper/nano-agent.pdf)).
The write-up is at [mailtotanvir.github.io/nano-agent](https://mailtotanvir.github.io/nano-agent/blog.html),
and the trained model is on Hugging Face:
[`mailtotanvir/nano-agent-rust-repair-0.5b`](https://huggingface.co/mailtotanvir/nano-agent-rust-repair-0.5b).

An archived, citable snapshot of this repository is on Zenodo:

> Ahmed, T. (2026). *nano-agent: A Tiny Model Beats Its Teacher — Verified-Loop
> Repair with a Sub-1B Language Model.* Zenodo. https://doi.org/10.5281/zenodo.22293159

DOI (all versions): [`10.5281/zenodo.22293159`](https://doi.org/10.5281/zenodo.22293159).
See [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
