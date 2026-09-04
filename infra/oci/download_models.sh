#!/usr/bin/env bash
# Download the rust-repair GGUF model sweep onto the OCI box.
# Direct HF resolve URLs (no huggingface-cli needed). Idempotent.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# repo|file|localname  (Q8_0 for <=0.6B, Q4_K_M for the 1.5B control)
DOWNLOADS=(
  "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF|qwen2.5-coder-0.5b-instruct-q8_0.gguf|qwen2.5-coder-0.5b-q8_0.gguf"
  "Qwen/Qwen2.5-0.5B-Instruct-GGUF|qwen2.5-0.5b-instruct-q8_0.gguf|qwen2.5-0.5b-q8_0.gguf"
  "bartowski/SmolLM2-360M-Instruct-GGUF|SmolLM2-360M-Instruct-Q8_0.gguf|smollm2-360m-q8_0.gguf"
  "bartowski/SmolLM2-135M-Instruct-GGUF|SmolLM2-135M-Instruct-Q8_0.gguf|smollm2-135m-q8_0.gguf"
  "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF|qwen2.5-coder-1.5b-instruct-q4_k_m.gguf|qwen2.5-coder-1.5b-q4_k_m.gguf"
  "ggml-org/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q8_0.gguf|qwen3-0.6b-q8_0.gguf"
)

for entry in "${DOWNLOADS[@]}"; do
  IFS='|' read -r repo file local <<< "$entry"
  if [[ -f "$local" ]]; then
    echo "have $local ($(du -h "$local" | cut -f1))"
    continue
  fi
  url="https://huggingface.co/${repo}/resolve/main/${file}"
  echo "downloading $local from $repo ..."
  if curl -fL --retry 3 -o "${local}.part" "$url"; then
    mv "${local}.part" "$local"
    echo "  done: $(du -h "$local" | cut -f1)"
  else
    echo "  FAILED: $url (skipping)"
    rm -f "${local}.part"
  fi
done

echo "=== models present ==="
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "none"
