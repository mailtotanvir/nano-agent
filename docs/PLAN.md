# Rust Tiny Agent - Implementation Plan & Design

Status: DRAFT for review. No implementation begins until this plan is approved.
Spec reference: `rust-tiny-agent-spec.md`
Date: 2026-09-01

---

## 1. Executive Summary

We build a compiler-error repair primitive: a tiny open-weight model (0.1B to 1B params) wrapped in a deterministic controller that loops `cargo check` -> diagnose -> patch -> re-verify, escalating to a frontier model only when the bounded budget is exhausted.

Key decisions proposed in this plan:

| Decision | Proposal |
|---|---|
| Primary base model | Qwen2.5-Coder-0.5B-Instruct (code-pretrained, best fit at target size) |
| Model sweep | SmolLM2-135M, SmolLM2-360M, Qwen2.5-Coder-0.5B, Qwen3-0.6B, Qwen2.5-Coder-1.5B (control) |
| Training GPU | GCP NVIDIA L4 24GB (g2-standard-8), spot where possible. GPU needed ONLY for Phase 2/3 training, roughly 30-60 GPU-hours total |
| Inference | CPU on the OCI ARM box via llama.cpp (GGUF Q8_0/Q4_K_M). No GPU needed at serve time |
| Harness language | Rust for the controller + Python for training/eval pipelines |
| Dataset | Synthetic broken-Rust generator + real crates.io regressions + frontier-teacher repairs, all compiler-verified |

The OCI ARM machine (4 vCPU, 24GB RAM) is the permanent home of everything except training: dataset generation, the cargo verifier farm, tiny-model inference, evaluation, and logging. GCP GPU is a short burst resource used inside the 20-day credit window.

---

## 2. Do We Need a GPU? (Direct Answer)

Yes, but only for training, and only briefly.

| Workload | Hardware | Why |
|---|---|---|
| Controller / verifier / cargo check farm | OCI ARM CPU | Compilation is CPU-bound; 4 vCPU handles a serial repair loop fine |
| Dataset generation (breaking code, teacher calls) | OCI ARM CPU | API calls + rustc runs, no GPU math |
| Tiny model inference (0.1B-0.6B) | OCI ARM CPU, llama.cpp | A 0.5B model at Q8_0 is ~600MB and runs 15-40 tok/s on 4 ARM cores with NEON. Patches are short (typically <300 tokens), so per-attempt latency is seconds. This is the whole point of the project: cheap CPU inference |
| Phase 1 baseline eval (untrained models) | OCI ARM CPU | Same as above |
| Phase 2 SFT training | GCP GPU (L4 24GB) | Even a 0.5B full fine-tune wants bf16 tensor cores. CPU training is not viable |
| Phase 3 GRPO/RL (conditional) | GCP GPU (L4, possibly A100 40GB if rollout throughput demands it) | Online rollouts need fast batched generation |
| Frontier teacher / fallback | External API | Existing credits |

### GCP GPU sizing and budget (verified against project your-gcp-project)

Target: g2-standard-8 (1x L4 24GB, 8 vCPU, 32GB RAM), SPOT, any North America region with capacity (us-central1 / us-east1 / us-east4 / us-west1 checked).

Verified quota state (2026-09-01, via gcloud):
- Per-region NVIDIA_L4_GPUS = 1 and PREEMPTIBLE_NVIDIA_L4_GPUS = 1 in all five NA regions: enough for our single-GPU plan.
- BLOCKER: project-wide GPUS_ALL_REGIONS = 0. A quota increase to 1 must be requested (Console > IAM > Quotas > "GPUs (all regions)") before ANY GPU VM can launch, spot included. Usually approved in minutes-to-hours for 1 GPU on a billing-enabled project; do this on day 1, not day 8.

Pricing (list, us-central1, subject to the live estimate we will pull before launch):
- g2-standard-8 spot: roughly $0.30-0.40/hr all-in (spot L4 ~ $0.22/hr + spot vCPU/RAM). On-demand fallback: ~$0.85/hr.
- Spot usage IS billable Compute Engine usage and therefore IS covered by credits, same as on-demand, just cheaper. We will confirm on the first run by checking the billing report shows the charge offset by the promotion credit.

Budget fit ($200 total credit, expires 2026-09-20, shared with the Gemini teacher):
- SFT sweep 30-60 GPU-hours on spot = $12-25.
- Optional Phase 3 RL 40-80 GPU-hours = $15-30.
- Teacher datagen on Gemini 3.6 Flash: flash-class pricing makes 20-50k trajectory steps a few dollars; reserve $30.
- Worst case total ~ $85, half the credit pool. Comfortable.

Compressed timeline (user wants launch in 3-4 days, credits end Sept 20):
- Day 1: request GPUS_ALL_REGIONS quota; OCI box setup; start harness.
- Days 1-3: harness + dataset v0 + teacher trajectories on OCI.
- Day 3-4: Phase 1 untrained sweep on OCI; first SFT run on GCP spot.
- Remaining window to Sept 20: iterate SFT, optional RL, eval, packaging. Checkpoints sync to the OCI box immediately after every run.

Explicitly NOT using: H100/A100 by default, TPUs, multi-GPU. The spec forbids it until measurements demand it.

---

## 3. Open-Weight Model Selection

### Recommendation: Qwen2.5-Coder-0.5B-Instruct as primary

Rationale:
- Only sub-1B open model pretrained heavily on code (5.5T tokens with large code fraction), including Rust.
- Apache 2.0 license, no usage restrictions.
- 32k context (we need ~2-4k for diagnostic + file excerpt + patch).
- Strong tokenizer for code; GGUF conversion and llama.cpp support are mature, including ARM64.
- 0.5B sits at the spec's ~0.3-0.6B sweet spot: big enough to learn Rust patch syntax, small enough for fast CPU serving.

### Sweep matrix (per spec section 7)

| Size class | Model | License | Notes |
|---|---|---|---|
| ~0.1B | SmolLM2-135M-Instruct | Apache 2.0 | Lower bound; likely fails, valuable data point |
| ~0.3B | SmolLM2-360M-Instruct | Apache 2.0 | Cheapest plausible candidate |
| ~0.5B | Qwen2.5-Coder-0.5B-Instruct | Apache 2.0 | PRIMARY |
| ~0.6B | Qwen3-0.6B | Apache 2.0 | Newer base, general-purpose; tests whether code pretraining matters at this scale |
| ~1.5B | Qwen2.5-Coder-1.5B-Instruct | Apache 2.0 | Control / upper bound, and fallback primary if 0.5B underperforms badly |

All five run Phase 1 (untrained, in-loop baseline). Only the best 2-3 proceed to SFT to conserve the GPU window.

Memory check for OCI serving: 1.5B at Q4_K_M is ~1.1GB, 0.5B at Q8_0 is ~0.6GB. All fit trivially in 24GB RAM alongside cargo builds.

---

## 4. Solution Design

### 4.1 Component architecture

```text
+--------------------------------------------------------------+
|                    rust-repair-recipe (OCI ARM)               |
|                                                              |
|  +----------------+     +-----------------+                  |
|  |  Controller    |---->|  Sandbox        |                  |
|  |  (Rust binary) |     |  workspace/git  |                  |
|  +-------+--------+     +--------+--------+                  |
|          |                       |                           |
|          | context pack          | cargo check --message-    |
|          v                       v        format=json        |
|  +----------------+     +-----------------+                  |
|  | Model Client   |     |  Verifier       |                  |
|  | (trait)        |     |  (rustc JSON    |                  |
|  |  - llama.cpp   |     |   parser +      |                  |
|  |    (tiny)      |     |   success/fail  |                  |
|  |  - OpenAI-compat|    |   classifier)   |                  |
|  |    (frontier)  |     +-----------------+                  |
|  +----------------+                                          |
|          |                                                   |
|          v                                                   |
|  +----------------+     +-----------------+                  |
|  | Patch Engine   |     | Trajectory Log  |                  |
|  | (unified diff /|     | (JSONL, every   |                  |
|  |  search-replace|     |  step recorded) |                  |
|  |  apply+validate|     +-----------------+                  |
|  +----------------+                                          |
+--------------------------------------------------------------+
```

The Model Client is a trait with two implementations (llama.cpp HTTP server for the tiny model, OpenAI-compatible client for the frontier teacher/fallback). Swapping models never touches the controller, satisfying spec section 12.

### 4.2 Controller loop (deterministic)

```text
attempt = 0
snapshot = git stash of workspace
while attempt < MAX_ATTEMPTS (default 4):
    diags = cargo check --message-format=json
    if no errors: return SUCCESS(trajectory)
    primary = select_primary_error(diags)        # first error, dedup by code
    context = build_context(primary)             # error text + code excerpt
                                                 # (window around span, +/- 20
                                                 # lines) + error-code docs
                                                 # (rustc --explain, truncated)
    proposal = model.propose(context, history)   # JSON: action/patch/reason/conf
    if action == escalate or no_fix: break
    if !patch_applies_cleanly(proposal):
        log(invalid_patch); attempt += 1; continue
    apply(proposal)
    attempt += 1
return ESCALATE(trajectory)                       # frontier gets full history
```

Safety rails:
- Workspace is a throwaway git-tracked copy; every attempt is a commit, rollback is `git reset`.
- Regression guard: if error COUNT increases after a patch, revert that patch before the next attempt.
- Hard caps: wall-clock timeout per repair (default 120s), max patch size (default 60 changed lines), no file creation/deletion in MVP.

### 4.3 Patch format

Search/replace blocks, not unified diff. Tiny models produce malformed diff headers constantly; exact-match search/replace with a fuzzy fallback (whitespace-normalized) is far more reliable and trivially validated:

```text
<<<<<<< SEARCH
fn main() {
    let x: i32 = "5";
=======
fn main() {
    let x: i32 = 5;
>>>>>>> REPLACE
```

The patch engine validates: search text exists exactly once in the named file, replace differs from search, size cap respected. Invalid patches are logged as negative signal and cost one attempt.

### 4.4 Model I/O contract

Prompt (chat template of the base model):

```text
SYSTEM: You are a Rust compiler-error repair agent... respond ONLY with
        the JSON contract...
USER:
  file: src/main.rs
  error[E0308]: mismatched types  ... (full primary diagnostic)
  --- code (lines 10-52 of src/main.rs) ---
  ...
  --- previous attempts (if any): patch + resulting diagnostic ---
```

Output: the spec's JSON contract (action, patch, reason, confidence). Enforced via llama.cpp GBNF grammar so the tiny model physically cannot emit malformed JSON. This is a major reliability lever for sub-1B models.

### 4.5 Repository layout (cookbook style, APPROVED direction)

`nano-agent` is a public cookbook of tiny verified-repair agents. Each recipe is a self-contained vertical (Rust repair first, SQL fixer / Terraform later); genuinely shared machinery lives in `core/`. Nothing recipe-specific leaks into core.

```text
nano-agent/                          (GitHub-published mono-repo)
  README.md                          cookbook index: what a nano agent is, recipe list
  LICENSE                            Apache 2.0
  docs/
    rust-tiny-agent-spec.md          research spec (moved from root)
    PLAN.md                          this plan (moved from root)
    ADRs/                            architecture decision records
  core/                              shared, recipe-agnostic
    controller/  (Rust crate)        generic bounded verify->propose->apply loop,
                                     Verifier / ModelClient / PatchEngine traits
    model-client/ (Rust crate)       llama.cpp + OpenAI-compatible + Gemini impls
    patch/       (Rust crate)        search/replace engine (language-agnostic)
    trajectory/  (Rust crate)        JSONL trajectory schema + logging
    py/nano_common/                  shared Python: dataset schema, experiment
                                     registry, eval report generation
  recipes/
    rust-repair/
      README.md                      recipe card: scope, envelope, usage, results
      crates/verifier/               cargo/rustc JSON diagnostics -> Verifier trait
      crates/cli/                    `nano-agent rust-repair fix <path>` + HTTP endpoint
      datagen/                       corpus building, mutation rules, teacher loop
      training/                      SFT configs per base model (TRL)
      eval/                          4-arm harness, per-E-code envelope tables
      datasets/                      versioned manifests (data itself on OCI disk,
                                     published sets via HF Hub, not git)
      experiments/                   one YAML config + results.json per run
    _template/                       skeleton for the next recipe (SQL, Terraform)
  infra/
    gcp/                             training VM launch/teardown scripts, spot handling
    oci/                             box setup (rustup, llama.cpp build, sccache)
  Cargo.toml                         workspace root (core crates + recipe crates)
```

Conventions for publishability: Apache 2.0, CI (cargo fmt/clippy/test + ruff), recipe READMEs follow a fixed "recipe card" format (problem, envelope, model, metrics, how to run), no datasets or checkpoints in git (HF Hub + manifests).

### 4.6 Dataset design (Phase 0/2 input)

Three sources, all verified by rustc before entering the dataset:

1. Synthetic breakage: take compiling code (small crates from crates.io with permissive licenses, plus generated snippets), apply mutation rules that produce the target error distribution: E0308 (type mismatch), E0382/E0502/E0499 (borrow/move), E0433/E0412 (unresolved path/type), E0599 (missing method), E0106 (lifetimes), missing trait bounds, missing derives, missing imports, semicolon/syntax classes. Mutations are deterministic and labeled, giving ground-truth error taxonomy for the competence-envelope analysis.
2. Real-world failures: top-N crates.io packages built at intermediate git commits that fail `cargo check`; plus community datasets if licensing checks pass.
3. Teacher trajectories: frontier model runs INSIDE the same controller loop on the failure corpus. Every step is compiler-verified. Passing trajectories become SFT positives; failing attempts with their follow-up diagnostics become multi-turn recovery examples.

Targets: ~5k distinct failure cases for eval candidates, ~20-50k teacher trajectory steps for SFT. Eval set (~500 cases, stratified by error code) is frozen FIRST and its failure cases are excluded from all training data (dedup by content hash of the broken file set).

### 4.7 Evaluation harness

Runs the four spec-mandated arms on the frozen eval set:

1. Frontier direct (one-shot, no loop)
2. Tiny model alone (one-shot, no loop)
3. Tiny model + loop (the recipe)
4. Recipe + frontier escalation

Metrics per spec section 9, reported per error-code stratum so the "bounded competence envelope" is an actual table: which E-codes the tiny agent owns, which it must escalate. Latency measured on the OCI box (the real deployment target). Cost model: measured tokens x published API prices for frontier vs measured watt-hours/instance-cost for CPU inference.

---

## 5. Phase Plan with Infra Assignment

| Phase | Work | Where | Est. effort |
|---|---|---|---|
| 0 | Harness: controller, verifier, patch engine, logging. Frontier baseline on eval set | OCI ARM | Days 1-5 |
| 0.5 | Dataset v0: synthetic breakage + eval-set freeze + teacher trajectory generation | OCI ARM + frontier API | Days 3-7 (overlaps 0) |
| 1 | Untrained tiny models in the loop (5-model sweep, GGUF on llama.cpp) | OCI ARM | Days 6-8 |
| 2 | SFT on verified trajectories, 2-3 best models, full fine-tune bf16 | GCP L4 (credit window) | Days 8-14 |
| 2.5 | Convert checkpoints to GGUF, re-run Phase 1 eval with trained models | OCI ARM | Days 13-15 |
| 3 | GRPO/RL only if SFT leaves headroom AND credits/time remain | GCP L4/A100 | Conditional |
| 4 | Escalation wiring + arm-4 measurement | OCI ARM | Days 15-17 |
| 5 | Recipe packaging: `rust-repair` CLI + HTTP endpoint + report | OCI ARM | Days 17-20 |

Hard rule from spec: nothing trains before Phase 0 baseline numbers exist.

---

## 6. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| GCP credits expire mid-training | All training front-loaded to days 8-14; checkpoints synced off-GCP after every run; Azure credits as the named fallback (spec priority order) |
| Sub-1B models emit garbage patches | GBNF grammar-constrained JSON, search/replace format, invalid-patch-as-signal, 1.5B control as fallback primary |
| OCI 4 vCPU too slow for cargo farm | Pre-warm target dirs, `cargo check` not `build`, sccache, small crates in corpus; datagen parallelism capped at 3 workers |
| Eval contamination (train/test leak) | Eval set frozen first, content-hash dedup, mutation seeds disjoint |
| Teacher trajectories too easy (teacher fixes in 1 shot, no recovery data) | Deliberately sample harder mutations; keep failed-then-recovered multi-turn traces; cap teacher context to mimic tiny-model view |
| ARM llama.cpp throughput disappoints | Q4_K_M quantization, 4-thread NEON; if still slow, patches are short so even 10 tok/s = ~30s/attempt, acceptable; measure in Phase 1 before concluding |

---

## 7. Decisions Confirmed (2026-09-01)

1. Primary model: Qwen2.5-Coder-0.5B-Instruct + sweep list — CONFIRMED.
2. GCP: project `your-gcp-project`, billing enabled, any NA region with spot capacity. $200 credit (covers GPU spot + Gemini teacher), expires 2026-09-20. Per-region L4 quota exists; project-wide GPUS_ALL_REGIONS=0 must be raised to 1 first (day-1 action). Spot L4 ~ $0.30-0.40/hr all-in and is credit-eligible. Launch target: 3-4 days.
3. Teacher/fallback: Gemini 3.6 Flash. IMPORTANT (discovered 2026-09-01): the
   AI Studio GEMINI_API_KEY in the environment is FREE-TIER (20 requests/day,
   HTTP 429 beyond that) and is NOT billed to the $200 Cloud credit. The teacher
   therefore runs through VERTEX AI on project your-gcp-project via ADC
   (`gcloud auth application-default print-access-token`), model
   `gemini-3.6-flash` on the `global` endpoint. This bills the Cloud credit, has
   real quota, and is ~2-3s/call (vs 15s+ on the throttled free tier). The
   `rust-repair` CLI supports `--backend vertex --gcp-project --gcp-location`;
   each teacher case is a fresh process so ADC tokens never expire mid-run.
4. Serving box verified: Ubuntu 24.04 aarch64, 4 vCPU, 23GB RAM, 143GB free disk, Docker installed, Rust via rustup. Accessed over SSH.
5. Repo: cookbook/recipe mono-repo per section 4.5, published to GitHub, Apache 2.0, CI, recipe-card READMEs.

Remaining pre-flight items (day 1 of implementation):
- Submit GPUS_ALL_REGIONS quota increase (0 -> 1) and confirm approval.
- Verify on the first GPU hour that the spot charge is offset by the promotional credit in the billing report.

PLAN APPROVED PENDING FINAL USER SIGN-OFF. Phase 0 starts on the OCI box; no GPU spend until Phase 2.
