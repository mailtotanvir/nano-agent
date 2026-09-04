#!/usr/bin/env bash
# Launch a spot L4 GPU VM on GCP for Phase 2 SFT, run training, sync results back.
#
# Prereqs (verified for your GCP project):
#   - GPUS_ALL_REGIONS quota >= 1
#   - per-region NVIDIA_L4 + PREEMPTIBLE_NVIDIA_L4 quota >= 1 (us-east1 etc.)
#   - gcloud authenticated, billing enabled
#
# This is a scaffold: review before running. It creates the VM with a deep-learning
# CUDA image, but training is driven manually (or via --startup) so we control cost.
set -euo pipefail

PROJECT="${PROJECT:-your-gcp-project}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-nano-sft-l4}"
MACHINE="${MACHINE:-g2-standard-8}"     # 1x L4 24GB, 8 vCPU, 32GB RAM

case "${1:-}" in
  up)
    echo "Creating SPOT ${MACHINE} (1x L4) in ${ZONE}..."
    gcloud compute instances create "$VM" \
      --project="$PROJECT" --zone="$ZONE" \
      --machine-type="$MACHINE" \
      --provisioning-model=SPOT \
      --instance-termination-action=DELETE \
      --maintenance-policy=TERMINATE \
      --accelerator=type=nvidia-l4,count=1 \
      --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=100GB --boot-disk-type=pd-balanced \
      --metadata="install-nvidia-driver=True"
    echo "SSH in with: gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE"
    ;;
  price)
    echo "Estimating spot price for ${MACHINE} + L4 in ${ZONE}..."
    echo "List spot (approx): g2-standard-8 ~ \$0.30-0.40/hr all-in."
    echo "Confirm actual on the billing report after the first hour."
    ;;
  status)
    gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" \
      --format="value(status,scheduling.provisioningModel)" 2>/dev/null || echo "not found"
    ;;
  down)
    echo "Deleting $VM (stops all billing)..."
    gcloud compute instances delete "$VM" --project="$PROJECT" --zone="$ZONE" --quiet
    ;;
  *)
    echo "usage: gpu_vm.sh {up|price|status|down}"
    echo "  up     - create spot L4 VM"
    echo "  price  - price note"
    echo "  status - VM status"
    echo "  down   - DELETE VM (do this the moment training finishes!)"
    exit 1
    ;;
esac
