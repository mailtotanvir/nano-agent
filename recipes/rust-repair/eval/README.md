# rust-repair eval

The 4-arm evaluation harness that produces the competence envelope (spec §9).

## Arms

| Arm | What | Role |
|---|---|---|
| `frontier_direct` | one-shot frontier, no loop | upper bound |
| `tiny_direct` | one-shot tiny model, no loop | shows what the loop adds |
| `tiny_loop` | tiny model + cargo loop (the recipe) | THE experiment |
| `recipe_escalate` | tiny loop, then frontier fallback on failure | deployment value |

The key comparison is `tiny_loop` vs `frontier_direct`: how much frontier-level
accuracy the tiny agent recovers by running inside the verified loop. Latency is
reported but is NOT the objective -- the goal is frontier-level accuracy running
locally/privately, even if slower.

## Output

Per-error-code table (the envelope) plus overall repair rate, avg attempts,
median/p95 latency, and escalation count. Written as JSON to `--out` and printed.

```bash
python run_eval.py --eval ../datasets/eval_v1.jsonl \
    --binary ../../../target/release/rust-repair \
    --arms tiny_loop,frontier_direct \
    --tiny-model qwen2.5-coder-0.5b-q8_0 --llama-url http://127.0.0.1:8080 \
    --frontier-model gemini-3.6-flash --frontier-backend vertex \
    --out results/eval_v1_qwen05.json
```

The tiny arms require a running llama.cpp server (see `infra/oci`). The frontier
arms use Vertex AI via ADC (billed to the GCP project).

## Notes

- The verifier (cargo), not the model, decides success.
- Runs each case in an isolated temp crate; nothing touches the frozen dataset.
- Eval cases were frozen and excluded from all teacher/training data
  (see `datagen/split_eval.py` + `datasets/manifest.json`).
