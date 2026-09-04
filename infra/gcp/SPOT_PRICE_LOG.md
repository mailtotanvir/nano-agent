# GCP Spot Price Evidence Log - nano-agent Phase 2 SFT

SOURCE: GCP Cloud Billing Catalog API (authoritative public list price)
  service: 6F81-5844-456A (Compute Engine)
  endpoint: https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus
CAPTURED: 2026-09-02T23:08:53.663277-04:00
PROJECT: your-gcp-project
BILLING ACCOUNT: 0176EA-61085B-B7643A
REGION: us-east1 (zone us-east1-c)
MACHINE: g2-standard-8 (8 vCPU, 32 GiB RAM) + 1x NVIDIA L4, provisioning-model=SPOT

SPOT SKU UNIT PRICES (USD, us-east1, Americas):
  Spot G2 Instance Core .............. $0.014990 / vCPU-hour
  Spot G2 Instance RAM ............... $0.001756 / GiB-hour
  Nvidia L4 GPU attached to Spot VMs .. $0.335900 / GPU-hour

COMPUTED g2-standard-8 + 1x L4 SPOT TOTAL:
  cpu:  8 vCPU  x $0.014990   = $0.119920/hr
  ram:  32 GiB   x $0.001756 = $0.056192/hr
  gpu:  1 L4    x $0.335900   = $0.335900/hr
  --------------------------------------------------
  TOTAL SPOT PRICE = $0.512012 / hour  (~$0.512/hr)

CLAIM BASIS: If billed above this rate for a SPOT g2-standard-8+L4 in us-east1,
this is the documented list spot price at capture time. On-demand equivalent is
~4-6x higher (~$0.85-1.00/hr for the same shape), so any charge near on-demand
indicates the SPOT provisioning-model did not apply and is disputable.

NOTE: Spot prices float within a capped band; GCP may charge at or below this
figure. This is the published catalog rate, not a locked quote.

---

## Run 2 - Phase 3b retrain (sft_v3), captured 2026-09-03T01:38:47-04:00

SOURCE: GCP Cloud Billing Catalog API (service 6F81-5844-456A)
REGION: us-central1 (zone us-central1-a), SPOT g2-standard-8 + 1x L4

  Spot G2 Instance Core .............. $0.014990 / vCPU-hour  x8  = $0.1199/hr
  Spot G2 Instance RAM ............... $0.001844 / GiB-hour   x32 = $0.0590/hr
  Nvidia L4 GPU attached to Spot VMs .. $0.335900 / GPU-hour   x1  = $0.3359/hr
  --------------------------------------------------
  TOTAL SPOT PRICE = $0.514828 / hour

No GCS this run (bucket deleted per user); artifacts pulled via SSH/SCP.

## Run 3 - Phase 3b retrain RETRY (sft_v3), captured 2026-09-03T01:54:13-04:00

Reason: run 2 artifact was truncated (SCP'd before tar finished; VM deleted).
REGION: us-central1-a, SPOT g2-standard-8 + 1x L4
  core $0.014990/vCPU-h x8 + ram $0.001844/GiB-h x32 + L4 $0.335900/GPU-h x1
  TOTAL SPOT PRICE = $0.514828 / hour
Capture fix: wait for STATUS=DONE + verify tar contains model.safetensors BEFORE teardown.

## Run 4 - Phase 3b retrain, ON-DEMAND (no preemption), captured 2026-09-03T02:03-04:00

Switched to ON-DEMAND after 2 spot preemptions destroyed the artifact mid-capture.
SOURCE: GCP Cloud Billing Catalog API (service 6F81-5844-456A), us-central1, standard g2-standard-8 + 1x L4:
  G2 Instance Core ..... $0.024988 / vCPU-hour  x8  = $0.199904/hr
  G2 Instance RAM ...... $0.002927 / GiB-hour   x32 = $0.093664/hr
  Nvidia L4 GPU ........ $0.560040 / GPU-hour    x1  = $0.560040/hr
  --------------------------------------------------
  TOTAL ON-DEMAND = $0.853608 / hour  (~$0.85/hr)
On-demand won't preempt -> reliable capture. ~15 min run = ~$0.21.

## Run 5 - Phase 4 retrain (sft_v4), ON-DEMAND, captured 2026-09-03T22:42:07-04:00

SOURCE: GCP Cloud Billing Catalog API (service 6F81-5844-456A), queried live;
pricing effective 2026-09-03T07:00:00Z. REGION: us-central1 (zone us-central1-a),
standard g2-standard-8 + 1x L4. No spot/provisioning-model flag will be used.
  G2 Instance Core ..... $0.024988212 / vCPU-hour x8  = $0.199905696/hr
  G2 Instance RAM ...... $0.002927448 / GiB-hour  x32 = $0.093678336/hr
  Nvidia L4 GPU ........ $0.560040239 / GPU-hour  x1  = $0.560040239/hr
  --------------------------------------------------
  TOTAL ON-DEMAND = $0.853624271 / hour (~$0.85/hr)

Authorized scope: create ONLY this single training VM; do not create any other
paid or free GCP service, instance, bucket, disk, or API resource without a new
explicit user authorization. Artifact transfer is SSH/SCP only; no GCS.
