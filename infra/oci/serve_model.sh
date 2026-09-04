#!/usr/bin/env bash
# Serve a GGUF model with llama.cpp on the OCI box for the rust-repair tiny arms.
# Usage: serve_model.sh <model-file.gguf> [port]
set -euo pipefail

MODEL="${1:?usage: serve_model.sh <model.gguf> [port]}"
PORT="${2:-8090}"  # 8080 is often taken on the OCI box; default to 8090
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
LLAMA="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"

path="$MODELS_DIR/$MODEL"
[[ -f "$path" ]] || { echo "model not found: $path"; exit 1; }
[[ -x "$LLAMA" ]] || { echo "llama-server not found: $LLAMA"; exit 1; }

# 4 threads = OCI core count. --jinja enables the model's chat template.
# Context 4096 is plenty for diagnostic + code window + patch.
# To run persistently after SSH closes, launch via:
#   setsid bash -c 'nohup infra/oci/serve_model.sh MODEL PORT > /tmp/llama.log 2>&1 &'
exec "$LLAMA" \
  --model "$path" \
  --host 127.0.0.1 --port "$PORT" \
  --ctx-size 4096 \
  --threads 4 \
  --jinja \
  --alias "$(basename "$MODEL" .gguf)"
