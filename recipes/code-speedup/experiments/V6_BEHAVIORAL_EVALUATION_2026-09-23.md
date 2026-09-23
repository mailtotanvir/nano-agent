# v6 1.5B behavioral development result (2026-09-23)

## Decision

Keep v5 as the best validated adapter. The v6 final adapter scored **12/24
(50.0%)** on the same 24 case behavioral development set where v5 scored
**23/24 (95.8%)**. Do not promote v6 or measure it on the frozen sets. The
24 case one attempt run completed on the existing OCI CPU host; its temporary
inference server and evaluator were stopped afterward.

## Development result

| Family | v5 | v6 |
| --- | ---: | ---: |
| Hash membership | 3/3 | 3/3 |
| Distinct cardinality | 3/3 | 3/3 |
| Stable deduplication | 3/3 | 2/3 |
| Frequency index | 3/3 | 2/3 |
| Complement lookup | 3/3 | 0/3 |
| Running aggregate | 3/3 | 0/3 |
| Prefix range query | 3/3 | 2/3 |
| Fixed sliding window | 2/3 | 0/3 |
| **Total** | **23/24** | **12/24** |

All 12 v6 failures are semantic: nine verifier mismatches and three candidate
errors. There were no formatting, patch application, or verifier infrastructure
failures. Representative patches drop the `x + offset` preprocessing in
complement lookup, count the wrong frequency output shape, or mishandle prefix
and window boundaries. The evaluation used one greedy proposal per case through
the unchanged production controller and the network isolated verifier image.

## Evidence and likely cause

The complete raw JSON and per case JSONL reports are gitignored under
`artifacts/code-speed-sft-v6-20260922/`. Their SHA-256 values are:

- Full report: `1b4e9953839bfc69cd57d95590ae352ddb7922fd6a7a30f90444170c004eba0a`
- Per case rows: `5fb1271f6660b1aaf2ed9ea95cedb9f6a01b2445a80934622c5cbb70702d025f`

The dominant change in the SFT mixture is a plausible explanation, not a
proven causal result: v5 used 530 teacher turns plus 502 extra copies of six
training only invariant families, while v6 used 550 teacher turns and one
repair trace with no focused copies. Both included 115 v1 examples and used
40% effective v1 replay during training. The new verified v6 family contributed
36 teacher examples. Thus v6 added new coverage while sharply reducing the
relative weight of invariant preservation that v5 needed.

## Next experiment

Build v6.1 from the already verified v6 teacher trajectories and the same six
v5 focus families at weight three. Run leakage, supervision contract, and CPU
training smoke gates before seeking approval for another GPU run. The frozen
sets and the new v6 heldout set remain untouched by this diagnosis.
