# Rust Tiny Agent — Research & Build Specification

## 1. Objective

Build a **tiny specialized Rust repair agent** that replaces frontier-model calls for a bounded class of Rust compilation failures.

The research question is not simply whether a small model can repair Rust.

> **How much frontier intelligence can be replaced by a tiny model embedded in a verified repair loop?**

The model is one component. The **agent recipe + compiler + verifier + retry loop** is the product/research unit.

---

## 2. Target Workflow

```text
Rust code
   ↓
 rustc
   ↓
compiler diagnostic
   ↓
Rust Tiny Agent
   ├── diagnose
   ├── propose patch
   ├── apply patch
   ├── rustc
   └── repeat
        ↓
 verified build
```

Expose the capability as a reusable **Solari-style recipe**:

```text
Developer / coding agent
        ↓
   Rust compile failure
        ↓
   Rust Repair Recipe
        ↓
 ┌─────────────────────────┐
 │ tiny model              │
 │ rustc diagnostics       │
 │ patch/apply tool        │
 │ compiler verifier       │
 │ bounded retry loop      │
 └─────────────────────────┘
        ↓
     fixed code
```

If the recipe cannot resolve the failure within a bounded budget, it should **escalate to a frontier model** rather than pretending to be universal.

---

## 3. Core Hypothesis

A large fraction of Rust compilation failures are bounded, structurally identifiable problems where a tiny model does not need frontier-level general reasoning if it has:

1. structured `rustc` diagnostics,
2. relevant source context,
3. compiler feedback after each patch,
4. a constrained action space,
5. iterative verification.

Therefore:

> **Tiny model + tools + verification + iteration can outperform a much larger model on cost/latency for a bounded Rust-repair distribution.**

---

## 4. Scope — MVP

Start with **Rust compilation repair**, not general Rust software engineering.

### In scope

- compilation errors
- compiler diagnostics
- source/context extraction
- patch generation
- patch application
- `cargo check` / `rustc` verification
- bounded retry
- repair success/failure classification
- escalation to frontier model
- trajectory collection for later training

### Explicitly out of scope initially

- arbitrary repo-level feature implementation
- architecture redesign
- performance optimization
- security auditing
- test-driven feature development
- large multi-file refactors unless naturally required by a compiler error
- autonomous production deployment

---

## 5. Agent Contract

Input:

```json
{
  "repository": "...",
  "command": "cargo check",
  "diagnostic": "...",
  "relevant_files": ["..."],
  "attempt": 0
}
```

Output:

```json
{
  "action": "patch",
  "patch": "...",
  "reason": "...",
  "confidence": 0.0
}
```

Possible terminal actions:

- `patch`
- `no_fix`
- `escalate`

The verifier, not the model, determines whether a repair succeeded.

---

## 6. Closed-Loop Controller

Implement a deterministic controller around the model.

```text
while attempts < MAX_ATTEMPTS:
    diagnostic = run_cargo_check()

    if build_passes:
        SUCCESS

    context = extract_relevant_context(diagnostic)
    proposal = tiny_model(context)
    apply_patch(proposal)

    if patch_is_invalid:
        record_failure()
        continue

ESCALATE
```

The controller must log every trajectory:

```text
initial code
→ diagnostic
→ model proposal
→ applied patch
→ compiler result
→ next diagnostic
→ ...
→ success/failure
```

---

## 7. Model Strategy

Do **not** begin by deciding that a particular parameter count is optimal.

Test a small range of existing open-weight bases, approximately:

- ~0.1B
- ~0.3B
- ~0.6B
- optionally ~1B as a control

Primary training methods to investigate:

1. SFT baseline
2. SFT + verifier-generated trajectories
3. RL/GRPO-style verifier optimization if justified
4. Other preference/trajectory optimization only if the baseline warrants it

The first milestone is **not** maximum model quality. It is establishing whether specialization + verification produces a meaningful frontier-call reduction.

---

## 8. Data Generation

Use frontier models as **teachers**, because existing user credits make teacher-token cost effectively negligible for this experiment.

Generate or collect Rust failures, then have the teacher propose repairs.

Every proposed repair must be validated by the real compiler.

Preferred data pipeline:

```text
Rust failure
    ↓
frontier teacher
    ↓
candidate patch
    ↓
rustc/cargo check
    ├── PASS → positive trajectory
    └── FAIL → diagnostic + another attempt
```

Failed attempts are valuable training data.

Do not train on unverified teacher answers as if they were ground truth.

---

## 9. Evaluation

The primary metric is **frontier calls avoided**, not merely model benchmark accuracy.

Evaluate at minimum:

### Repair quality

- initial repair success rate
- success after N attempts
- exact/build-valid repair rate
- regression rate

### Agent efficiency

- frontier calls avoided
- average attempts per successful repair
- median latency
- p95 latency
- tokens generated
- inference cost

### Model efficiency

- parameter count
- memory footprint
- throughput
- CPU vs GPU inference

### Comparison

Compare:

1. frontier model directly
2. tiny model alone
3. tiny model + compiler loop
4. tiny model + compiler loop + escalation

The key experiment is #3 vs #1, with #4 showing practical deployment value.

---

## 10. Success Criterion

The MVP is successful if it demonstrates a **clear bounded competence envelope** where the tiny agent:

- achieves useful verified repair rates,
- is materially faster/cheaper than frontier inference,
- avoids a significant fraction of frontier calls,
- and fails safely by escalating when outside its competence.

A hypothetical result such as:

> “The 300M Rust Repair Agent resolves X% of compiler failures with verification and avoids Y% of frontier calls at Z× lower latency/cost.”

would be a strong research/product result even if it does not solve general Rust problems.

Do not optimize the headline numbers before establishing a trustworthy evaluation set.

---

## 11. Infrastructure Strategy

### CPU / orchestration

Use the existing OCI ARM64 machine for:

- dataset processing
- compiler/verifier workloads where practical
- experiment orchestration
- storage
- logging
- evaluation
- serving if the chosen inference stack supports the target architecture

### GPU priority

Use cloud credits before real money:

1. **GCP first** — credits expire in ~20 days
2. **Azure second** — existing credits
3. **AWS only if applicable credits are confirmed**
4. **Vast.ai last** — real personal cash

Start with a modest cloud GPU. Do not use H100/A100-class hardware unless measurements demonstrate that the experiment requires it.

For initial SFT/QLoRA experiments on sub-billion-parameter models, target a single ~24GB GPU class first.

---

## 12. Architecture Principle

Keep the model and recipe separable.

```text
                  Rust Repair Recipe
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   Tiny Model        Rust Tooling      Controller
        │                │                │
        └────────────────┼────────────────┘
                         │
                    Verifier Loop
                         │
                    Escalation
```

The same recipe should eventually support different tiny model backends without changing the verifier/controller.

---

## 13. Research Logging

Every experiment must record:

- model/base checkpoint
- parameter count
- training method
- dataset version
- number of examples/tokens
- GPU type
- GPU hours
- training configuration
- evaluation set version
- repair success
- attempts
- latency
- frontier calls
- failures

Make experiments reproducible.

---

## 14. Phase Plan

### Phase 0 — Baseline

Build the Rust repair harness before training anything.

Measure frontier model directly on the same failures.

### Phase 1 — Tiny baseline

Run a small open-weight model inside the repair loop without specialized training.

Establish whether the tools/loop alone create a meaningful improvement.

### Phase 2 — SFT

Train on verified repair trajectories.

Measure improvement.

### Phase 3 — Verifier-driven optimization

Only if SFT leaves substantial headroom, investigate RL/trajectory optimization using compilation success as the reward signal.

### Phase 4 — Escalation

Add frontier fallback.

Measure the percentage of failures resolved without frontier inference.

### Phase 5 — Recipe packaging

Turn the harness into a reusable agent/recipe callable by a coding agent.

---

## 15. Important Design Decision

**Do not build a generic “Rust AI agent.”**

Build:

> **A compiler-error repair primitive that happens to be implemented as an agent recipe.**

The bounded environment is what allows the tiny model to be competitive.

The compiler is the verifier.

The loop supplies additional intelligence.

The frontier model is the teacher and fallback—not the runtime dependency for ordinary successful cases.

---

## 16. Future Extension: Terraform

After Rust is validated, repeat the same architecture for Terraform/IaC:

```text
Terraform failure
    ↓
Tiny IaC Agent
    ↓
terraform validate / plan / policy scanners
    ↓
verified repair
    ↓
escalate if necessary
```

Do not implement Terraform in the MVP. It is the second recipe and serves as evidence that the approach generalizes beyond one compiler ecosystem.

---

## 17. Deliverable

The final MVP should contain:

- reproducible Rust failure dataset
- Rust repair harness
- tiny model inference wrapper
- training pipeline
- compiler verifier
- bounded retry controller
- frontier teacher/fallback interface
- evaluation harness
- cost/latency/frontier-call comparison
- experiment report
- runnable recipe/API for external coding agents

The central artifact is not merely the trained checkpoint.

**It is the verified Rust Repair Recipe.**
