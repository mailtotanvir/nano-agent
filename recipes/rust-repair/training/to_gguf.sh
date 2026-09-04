#!/usr/bin/env bash
# Convert a trained HF checkpoint to GGUF and quantize, for serving on the OCI box.
# Run on the GPU VM (or any box with llama.cpp checked out) after train_sft.py.
#
# Usage: to_gguf.sh <hf_checkpoint_dir> <out_name> [quant]
#   quant defaults to q8_0 (best for <=0.6B); use q4_k_m for 1.5B.
set -euo pipefail

CKPT="${1:?usage: to_gguf.sh <hf_checkpoint_dir> <out_name> [quant]}"
NAME="${2:?output base name, e.g. qwen05-sft-v1}"
QUANT="${3:-q8_0}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
OUT_DIR="${OUT_DIR:-$HOME/models}"

mkdir -p "$OUT_DIR"
f16="$OUT_DIR/${NAME}-f16.gguf"
final="$OUT_DIR/${NAME}-${QUANT}.gguf"

echo "1/2 convert HF -> f16 GGUF"
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$CKPT" --outfile "$f16" --outtype f16

echo "2/2 quantize -> $QUANT"
"$LLAMA_DIR/build/bin/llama-quantize" "$f16" "$final" "$QUANT"

echo "done: $final ($(du -h "$final" | cut -f1))"
echo "copy to OCI: scp $final ubuntu@nexus-oci:~/models/"
