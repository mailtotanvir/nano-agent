#!/usr/bin/env bash
# Unattended Phase 2 SFT runbook - runs entirely on the GPU VM at boot.
# Trains 0.5B, converts to GGUF, uploads artifacts to GCS (survives spot preemption).
# Progress + status are streamed to the GCS bucket so we can watch without SSH.
set -uo pipefail

BUCKET="${BUCKET:-gs://your-sft-bucket}"
RUN="qwen05-sft-v2"
WORK=/opt/nano
mkdir -p "$WORK/out"
cd "$WORK"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; gsutil -q cp "$WORK/run.log" "$BUCKET/$RUN/run.log" 2>/dev/null || true; }
status() { echo "$1" | gsutil -q cp - "$BUCKET/$RUN/STATUS" 2>/dev/null || true; }

status "STARTING"
log "boot: fetching training assets from GCS"
gsutil -q cp "$BUCKET/$RUN/inputs/train_sft.py" "$WORK/train_sft.py"
gsutil -q cp "$BUCKET/$RUN/inputs/sft_v2.jsonl"  "$WORK/sft_v2.jsonl"

log "fixing python deps (torchaudio ABI, trl/transformers/jinja2)"
sudo pip uninstall -y torchaudio >/dev/null 2>&1 || true
pip install -q --upgrade 'transformers>=4.46' 'trl>=0.12' 'datasets>=3.0' \
    'accelerate>=1.0' 'peft>=0.13' 'jinja2>=3.1.0' >/dev/null 2>&1

status "TRAINING"
log "training start"
python3 train_sft.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --data sft_v2.jsonl --out out/$RUN --epochs 3 --lr 1e-5 --bs 8 \
    >> "$WORK/train.log" 2>&1
rc=$?
gsutil -q cp "$WORK/train.log" "$BUCKET/$RUN/train.log" || true
if [ $rc -ne 0 ]; then log "TRAINING FAILED rc=$rc"; status "FAILED_TRAIN"; exit 1; fi
log "training done; uploading HF checkpoint"
gsutil -q -m cp -r "out/$RUN" "$BUCKET/$RUN/hf_checkpoint" || true
status "CONVERTING"

# Convert to GGUF via llama.cpp (clone + build minimal, CPU convert is fine)
log "cloning llama.cpp for GGUF conversion"
if [ ! -d llama.cpp ]; then git clone -q --depth 1 https://github.com/ggerganov/llama.cpp; fi
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt >/dev/null 2>&1 || true
python3 llama.cpp/convert_hf_to_gguf.py "out/$RUN" \
    --outfile "out/${RUN}-f16.gguf" --outtype f16 >> "$WORK/convert.log" 2>&1
# Build quantize tool
if [ ! -f llama.cpp/build/bin/llama-quantize ]; then
  cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF >> "$WORK/convert.log" 2>&1
  cmake --build llama.cpp/build --target llama-quantize -j4 >> "$WORK/convert.log" 2>&1
fi
llama.cpp/build/bin/llama-quantize "out/${RUN}-f16.gguf" "out/${RUN}-q8_0.gguf" q8_0 >> "$WORK/convert.log" 2>&1

gsutil -q cp "$WORK/convert.log" "$BUCKET/$RUN/convert.log" || true
if [ -f "out/${RUN}-q8_0.gguf" ]; then
  gsutil -q cp "out/${RUN}-q8_0.gguf" "$BUCKET/$RUN/${RUN}-q8_0.gguf"
  log "GGUF uploaded: $BUCKET/$RUN/${RUN}-q8_0.gguf"
  status "DONE"
else
  log "GGUF conversion failed (HF checkpoint still saved in GCS)"; status "DONE_HF_ONLY"
fi
log "ALL DONE"
