#!/usr/bin/env bash
set -euo pipefail
source ./pipeline.env

python3 -m pip install -q --disable-pip-version-check huggingface_hub hf_transfer || true
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$FP8_DIR"

echo "[INFO] Downloading ${TOKENIZER_PATH} to ${FP8_DIR} (FP8)"
huggingface-cli download "${TOKENIZER_PATH}" \
  --include "*.safetensors" "*.json" "tokenizer*" \
  --local-dir "${FP8_DIR}" \
  --local-dir-use-symlinks False
echo "[INFO] Done."
