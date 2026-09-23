# v5 invariant-weighted 1.5B behavioral evaluation (2026-09-21)

## Decision

**Retain the v5 adapter as the best seen-family checkpoint; do not promote it
as a general code-speed model.** It substantially improves the development and
frozen seen-family result, but it does not improve the frozen heldout aggregate.
Frozen reports are measurement-only and will not be used in corpus construction
or checkpoint selection.

## Training and artifact integrity

The on-demand L4 training completed 97 steps on 1,147 SFT rows. Final training
loss was `0.27260894627915216`; evaluation loss was `0.22524704039096832`; and
assistant-token accuracy was `0.9443444728851318`. The extracted adapter is
present under `artifacts/code-speed-sft-v5-20260920/extracted/`. Its transferred
archive SHA-256 is
`a4a402fe8812bd50d808be0ee40c4e3a58e78892fd48d235904fac013eecdb6b`.

The archive was verified before teardown. The GCP VM and its auto-delete boot
disk were then confirmed absent. The quoted hourly rate, not a final bill,
is reported in the paper.

## One-shot behavioral results

| Split | Success | Change from v3.1 |
|---|---:|---:|
| Behavioral development | 23 / 24 (95.8%) | +37.5 points |
| Frozen seen-family | 24 / 30 (80.0%) | +20.0 points |
| Frozen family-heldout | 14 / 30 (46.7%) | unchanged |

Seen-family successes were perfect for frequency indexing, hash membership,
prefix ranges, running aggregates, and stable deduplication. The remaining
seen-family weaknesses were complement lookup (1/4), fixed sliding windows
(1/3), and distinct cardinality (3/4). The heldout split had no successful
cases in its indexed-lookup family, while the other two heldout families each
reached 7/10. Failures were predominantly semantic: development 1; frozen seen
4 semantic plus 2 correct-but-not-faster; frozen heldout 15 semantic plus 1
correct-but-not-faster.

## Evidence files

The raw reports are intentionally gitignored alongside the artifact. Their
SHA-256 values are:

- development: `1c4c815524a1fc5e62722034087df8b8c91dbcad0df47b941c87551be5479ca7`
- frozen seen-family: `90b68a23e8c57f8b40f5c08a2ddd16fe78554cc208cba6eb7d50906e6a80ce82`
- frozen family-heldout: `200eaf01e049e87843fb9426044d57f8684c7c9e73d5dea5121c15df5b9cabf4`

## Next corrective experiment

v6 adds training-only transformed residue-count queries: it builds a
residue-to-count map and answers many queries after a deterministic value
transform. This preserves a deliberately different `xs, queries` contract from
the frozen item/key lookup benchmark. v6 also defines a new source-template-
disjoint, one-time heldout measurement. Before GPU training, every v6 pair must
pass the authoritative verifier, the final SFT corpus must pass the leakage and
assistant-token contract gates, and a CPU smoke must pass. A fresh GCP VM quote
and explicit approval remain required for any v6 GPU run.
