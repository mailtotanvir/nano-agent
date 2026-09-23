# v6.1 one-shot frozen evaluation (2026-09-23)

## Decision and reporting boundary

v6.1 scored **24/30 (80.0%)** on the frozen seen-family split, tying v5, and
**16/30 (53.3%)** on the frozen family-heldout split, two cases above v5's
14/30. The gain was limited to string concatenation and sort selection; both
v5 and v6.1 remained **0/10 on indexed lookup**. On the new three-case v6
residue-count-index diagnostic, v6.1 scored **0/3**. These three splits answer
different questions and must not be combined into a single headline.

v5 remains the checkpoint selected by the higher behavioral development
score (23/24 versus v6.1's 21/24). The user separately authorized a one-time
v6.1 frozen measurement for publication comparison. These frozen results are
measurement-only: they do not justify checkpoint selection or training another
model against these exact cases. The user chose to release v5 and v6.1 as
**equal experiment variants** in a new code-speed model repository; that is a
presentation and artifact decision, not a claim of equal measured performance
or general code-optimization ability.

## Protocol and results

Every case used one greedy proposal through the unchanged production Rust
controller and the isolated `code-speed-verifier:20260919` image. A success
requires hidden-input correctness and positive Cachegrind instruction-reference
reward. The model did not see private inputs or known-fast sources.

| Split | v5 | v6.1 |
| --- | ---: | ---: |
| Behavioral development, 24 | 23/24 | 21/24 |
| Frozen seen-family, 30 | 24/30 | 24/30 |
| Frozen family-heldout, 30 | 14/30 | 16/30 |
| Fresh v6 heldout, 3 | Not measured | 0/3 |

The seen-family tie masks different strengths:

| Frozen seen family | v5 | v6.1 |
| --- | ---: | ---: |
| Complement lookup | 1/4 | 2/4 |
| Distinct cardinality | 3/4 | 2/4 |
| Fixed sliding window | 1/3 | 2/3 |
| Frequency index | 4/4 | 4/4 |
| Hash membership | 4/4 | 4/4 |
| Prefix range query | 3/3 | 3/3 |
| Running aggregate | 4/4 | 4/4 |
| Stable deduplication | 4/4 | 3/4 |

| Frozen family | v5 | v6.1 |
| --- | ---: | ---: |
| String concatenation | 7/10 | 8/10 |
| Sort selection | 7/10 | 8/10 |
| Indexed lookup | 0/10 | 0/10 |

v6.1's seen-family six failures were five semantic and one
correct-but-not-faster. Its 14 family-heldout failures were 13 semantic and
one correct-but-not-faster. The fresh v6 heldout had three `candidate_error`
verifier statuses, all grouped by the evaluator under `semantic_failure`:
each proposed rewrite dropped the reference's in-function `bucket_size`
assignment, leaving an undefined variable. One also reversed query order.
The three-case result is a diagnostic of that missed invariant, not a stable
estimate of generalization.

## Evidence and teardown

Frozen input SHA-256 values, identical locally and on OCI before evaluation:

- Seen-family: `f8cf59cf2a850009e09e35e019b860f8135c5b9f9d503c27e002b8fb7e894128`
- Family-heldout: `1e1f0ae7ad6e0f097bc42a4316f5c7f02574c9cf6996c0a568748cba08e705ce`
- Fresh v6 heldout: `75bd543d56cc87846e4b6d8e9310ac4c393a91376478eaf69e36e4edc0132c2f`

The six raw report files are retained locally under the gitignored
`artifacts/code-speed-sft-v6p1-20260923/` directory. Each local SHA-256
matched the corresponding OCI source after transfer:

| Report | Full JSON SHA-256 | Per-case JSONL SHA-256 |
| --- | --- | --- |
| Frozen seen-family | `00a41526db5fac9e963e6c41a4bd59a1888ba2fcd3ce2e2266a7d384f65b7dd2` | `840435ae3e522b27759c1498c53a9092cb428f491afff7c6b10d3816ee464d18` |
| Frozen family-heldout | `adc0066bde045e9622654de6dca8bc16d2c7c42ea30f9d2257eb64e5a09ec383` | `29299948d9197dde80d408652670927ac92c16ae6b0a2d05fadb7577b892c7d9` |
| Fresh v6 heldout | `05a66dab3d26b7fbe2266e41eb1913b5faac74d59df5fb68fe4e08874d68b1c1` | `66f321e52a0f0b9894fcd47000a794c89aec704152c09858f7f15a1b375775da` |

The OCI evaluator and temporary model server exited after the third report;
a process check found neither running. The GCP v6.1 training VM and boot disk
had already been deleted after archive hash verification. The training quote
was `$0.856856553/hour`, under a `$1.75` two-hour authorization ceiling;
actual billing export remains pending.
