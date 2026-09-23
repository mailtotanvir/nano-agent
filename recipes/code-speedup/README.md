# code-speedup: verified tiny-model program optimization

This recipe tests whether a restricted, pure Python function can be rewritten
with fewer Cachegrind instruction references while passing an independent
hidden-input battery. It now includes the original Qwen2.5-Coder-0.5B SFT
baseline and a 1.5B LoRA sequence (v5, v6, v6.1). v5 remains the
development-selected checkpoint at 23/24; v6 scored 12/24, and v6.1 recovered
to 21/24. On frozen measurement, v6.1 tied v5 on seen-family cases at 24/30,
scored 16/30 on family-heldout cases (v5: 14/30), and scored 0/3 on a separate
small v6 heldout diagnostic. It remains 0/10 on frozen indexed lookup, so it
is not a general code optimizer. See `experiments/RESULTS.md` for the original
baseline, `experiments/V6P1_BEHAVIORAL_EVALUATION_2026-09-23.md` for
development, and `experiments/V6P1_FROZEN_EVALUATION_2026-09-23.md` for the
full split-by-split report. GRPO has not run.

The [paper](https://mailtotanvir.github.io/nano-agent/paper/code-speedup-paper.pdf),
[technical story](https://mailtotanvir.github.io/nano-agent/code-speedup-blog.html),
and [two adapter variants](https://huggingface.co/mailtotanvir/nano-agent-code-speed-1.5b)
give the complete release context. The original experiment plan is
`../../docs/phase5-code-speedup-plan.md`. No VM or service is created by these
tools. Cloud launches require separate explicit approval.

## Phase 1 corpus status

`datagen/` defines a deterministic 250-row corpus with a separate development
pool and two frozen evaluation tracks. The checked-in verified splits passed the
same Bubblewrap correctness and Cachegrind measurement gate used by training.
See `datagen/README.md` for the invocation and the model-prompt data boundary.

## Local Phase 2.1 loop

The `code-speedup` binary runs only against a local llama.cpp-compatible server.
An episode workspace must contain `candidate.py`; start it with the slow
reference source, then select a verifier-gated record by ID:

```bash
cargo run -p code-speedup-cli -- \
  --workspace /path/to/episode \
  --dataset recipes/code-speedup/datasets/corpus_speedup_v1.verified.jsonl \
  --problem-id cs-v1-0000-d728f57a4f
```

The Rust controller sends the model only the slow source, current `candidate.py`,
and sanitized status, reward, and instruction counts. The Python bridge keeps
the JSONL benchmark inputs, edge cases, generators, private test randomness,
and known-fast implementation outside the model-facing context.

## Run the local checks

Required host: Linux with unprivileged user/PID/network namespaces, bubblewrap,
`setarch`, `/usr/bin/python3`, and Valgrind with Cachegrind. The host test runner
needs pytest. The harness uses Python's standard library, not a model service.
Missing or blocked isolation is an error; there is no unsandboxed fallback.

From the repository root:

    python3 -m pytest recipes/code-speedup/harness -v
    python3 -m ruff check recipes/code-speedup/harness

If Valgrind was extracted locally instead of installed on PATH, explicitly set:

    export VALGRIND=/absolute/path/to/usr/bin/valgrind
    export VALGRIND_LIB=/absolute/path/to/usr/libexec/valgrind

These are tool paths, not credentials. Tests marked `cachegrind` exercise real
profiling. Passing only the parser/unit tests does not satisfy the spike gate.

## What the score means

The oracle compares independent reference and candidate executions on edge,
random, and benchmark inputs. Wrong outputs receive zero reward. Correct,
repeatable executions receive `log(reference_Ir / candidate_Ir)`. Infrastructure
errors and unstable counts must not be converted to zero-reward training data.

Cachegrind counts simulated instruction references (`Ir`), not hardware retired
instructions and not nanoseconds. Counts cover the Python process, including
startup, source compilation, harness and serialization. Fixed overhead dilutes
speedups. Comparisons are meaningful only under the same pinned interpreter,
Valgrind, inputs, environment and sandbox policy. Exact repeatability is checked,
not assumed from Cachegrind's name.

The raw logarithm is not bounded. A correct slower candidate has a negative raw
score, so zero for invalid output can be attractive to GRPO. The training phase
must explicitly resolve that reward-ranking issue before optimization; do not
silently treat the spike formula as a validated training objective.

## What the sandbox and tests do not prove

Only one function is allowed, using a documented subset of Python with safe
builtins and container methods. Imports, filesystem/network APIs, private names,
introspection, decorators and nested functions are outside this recipe's scope.
Namespace isolation and resource limits complement the language policy; the AST
filter alone is not a security boundary. Inputs and outputs must be JSON values.

Finite hidden tests are evidence, not a proof of equivalence or immunity to
reward hacking. A predictable benchmark can still be special-cased. The spike
must include adversarial cases at benchmark-sized inputs, and later RL runs need
fresh hidden batteries and attack monitoring. Do not run arbitrary hostile code
or expose this harness as a public multi-tenant service based on these tests.

The hand-authored spike fixtures are not a frozen evaluation set and are not
training data. Corpus verification and a content-disjoint frozen eval are later,
separately gated steps. No SFT plateau or RL improvement is claimed here.
