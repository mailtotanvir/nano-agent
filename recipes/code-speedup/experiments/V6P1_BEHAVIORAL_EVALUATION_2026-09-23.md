# v6.1 1.5B behavioral development result (2026-09-23)

## Decision

Keep v5 as the best validated adapter. v6.1 scored **21/24 (87.5%)** on the
same 24-case behavioral development set where v5 scored **23/24 (95.8%)** and
v6 scored **12/24 (50.0%)**. The v6.1 invariant-weighting repair recovered
9 cases versus v6 but did not clear the v5 promotion bar. At this point, the
existing development-first gate did not call for frozen evaluation. The user
subsequently authorized a one-time, measurement-only frozen comparison; see
`V6P1_FROZEN_EVALUATION_2026-09-23.md`. v5 remains development-selected.

## Development result

| Family | v5 | v6 | v6.1 |
| --- | ---: | ---: | ---: |
| Hash membership | 3/3 | 3/3 | 3/3 |
| Distinct cardinality | 3/3 | 3/3 | 2/3 |
| Stable deduplication | 3/3 | 2/3 | 2/3 |
| Frequency index | 3/3 | 2/3 | 3/3 |
| Complement lookup | 3/3 | 0/3 | 3/3 |
| Running aggregate | 3/3 | 0/3 | 3/3 |
| Prefix range query | 3/3 | 2/3 | 3/3 |
| Fixed sliding window | 2/3 | 0/3 | 2/3 |
| **Total** | **23/24** | **12/24** | **21/24** |

The run used one greedy proposal per case through the unchanged production
controller and network-isolated verifier image on the existing OCI CPU host.
All three failures were semantic verifier mismatches; no evaluator or patch
application failures occurred.

The failed v6.1 cases show distinct, actionable errors:

- `cs-v2dev-0003-6e944d2fcc` (distinct cardinality): after modulo
  normalization, the candidate counted values occurring **exactly once**
  instead of counting **distinct** values.
- `cs-v2dev-0006-8d68c1c798` (stable deduplication): the candidate retained
  only values with global count one, which drops all repeated values rather
  than retaining their first occurrence. It also counted against unnormalized
  input while emitting modulo-normalized values.
- `cs-v2dev-0022-6f3143792c` (fixed sliding window): the running-window
  update was valid, but the candidate selected the **minimum** window where
  the reference selected the **maximum**.

## Artifacts and teardown

The raw report and per-case JSONL are gitignored under
`artifacts/code-speed-sft-v6p1-20260923/`. Local and OCI SHA-256 matched:

- Full report: `5c5aeeb3da206d359339caec6f8b2892483198fac01e027ad37d139235b844b0`
- Per-case rows: `b506579e731843ac6545ced71cefc0edc585189a3b56fcc912815462ee5bb6b2`

The OCI evaluator exited after writing the report, and its temporary model
server was stopped. The GCP training VM had already been deleted after the
training archive was copied and verified; a fresh check found no compute
instances, disks, or reserved addresses in the project. The price quote and
teardown outcome are summarized in the public paper.

## Next option

Keep v5 for publication and any existing held-out claim. For another model
experiment, add verified counterexamples targeting the three failed semantic
invariants and test the next adapter on the same development gate before any
frozen-set evaluation. This is an option, not an approved GPU run.
