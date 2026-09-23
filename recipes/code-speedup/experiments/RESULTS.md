# code-speed SFT v1 results

## Checkpoint

- Base model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Training: full-parameter bf16 SFT, 3 epochs, 115 examples
- Train loss: 0.5926
- Development loss: 0.4275
- Development token accuracy: 0.9095
- Local artifact: `../artifacts/code-speed-sft-v1-20260908/`
- Archive SHA-256: `00aa9d714f6c1e308ac7840a944f359fbcd94ed7c5ac270d4e1b64adb5534290`

The training run used an on-demand L4 VM. The artifact was copied to this
workstation and verified before the VM and boot disk were deleted. The recorded
list-price estimate is $0.233.

## Behavioral evaluation

The evaluation uses greedy decoding and one proposal per problem. Every reported
success passed the existing private correctness battery and the Cachegrind speed
gate through the Rust controller. The model sees the slow source and public
controller feedback; it does not see hidden inputs or the known-fast solution.

| Split | Success | Rate | Interpretation |
| --- | ---: | ---: | --- |
| Training-distribution probe | 5/5 | 100.0% | The checkpoint can reproduce a taught hash-membership transformation. |
| Template-disjoint, seen families | 3/30 | 10.0% | Success is confined to distinct-cardinality (3/4). |
| Family-heldout | 8/30 | 26.7% | String-concatenation is 6/10, sort-selection 2/10, indexed-lookup 0/10. |

Raw reports:

- `eval_sft_v1_train_5.json`
- `eval_sft_v1_seen_family.json`
- `eval_sft_v1_family_heldout.json`

These results show that the tiny model learned valid optimization behavior, but
not broad program optimization. The 5/5 training probe alongside 3/30 on new
templates is strong evidence of narrow template learning. The held-out-family
successes demonstrate some transfer, especially for string concatenation, but
the zero score on indexed lookup and low overall sample count prevent a general
capability claim.

The local Hugging Face server does not enforce the llama.cpp grammar parameter.
The Rust controller still parses the response and the verifier accepts only
valid patches, so successes remain valid; the rates also include formatting
failures that grammar-constrained deployment may avoid. Latency is CPU-local and
is not a model-speed benchmark.

## Next gate

Before GRPO, inspect failure categories and add verified teacher demonstrations
for weak transformations while preserving both frozen evaluation splits. Then
repeat this exact one-shot evaluation and report a separate multi-attempt score.
Only after SFT improves across families should the project test verifier-driven
GRPO with a reward where invalid output is strictly worse than a correct but
slower candidate.
