#!/usr/bin/env bash
# Unattended Phase 4 SFT runbook. The artifact is pulled only after DONE and
# remote archive validation; no GCS or other cloud service is used.
set -uo pipefail

RUN="qwen05-sft-v4"
WORK="$HOME/nano"
mkdir -p "$WORK/out"
cd "$WORK"

echo "STARTING" > "$WORK/STATUS"
echo "[$(date -u +%H:%M:%S)] deps" | tee -a "$WORK/run.log"
sudo pip uninstall -y torchaudio >/dev/null 2>&1 || true
pip install -q --upgrade 'transformers>=4.46' 'trl>=0.12' 'datasets>=3.0' \
    'accelerate>=1.0' 'peft>=0.13' 'jinja2>=3.1.0' >/dev/null 2>&1

echo "TRAINING" > "$WORK/STATUS"
echo "[$(date -u +%H:%M:%S)] train start" | tee -a "$WORK/run.log"
python3 train_sft.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --data sft_v4.jsonl --out out/$RUN --epochs 3 --lr 1e-5 --bs 8 \
    >> "$WORK/train.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "FAILED_TRAIN" > "$WORK/STATUS"
    echo "train rc=$rc" >> "$WORK/run.log"
    exit 1
fi

echo "[$(date -u +%H:%M:%S)] train done" | tee -a "$WORK/run.log"
# Archive ONLY the final model dir; exclude intermediate checkpoints, which are
# full model copies that bloat the tar, can fill the disk, and wedge sshd.
tar czf "$WORK/out_${RUN}.tgz" --exclude="checkpoint-*" -C "$WORK/out" "$RUN"
echo "DONE" > "$WORK/STATUS"
echo "[$(date -u +%H:%M:%S)] artifact ready: out_${RUN}.tgz" | tee -a "$WORK/run.log"
