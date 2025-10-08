#!/usr/bin/env bash
set -euo pipefail
source ./pipeline.env

echo "[orbax] HF BF16 -> scanned @ ${MODEL_BUCKET}/${IDX}"
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_family_ckpt \
  --base_model_path    "${CHKPT_BUCKET}" \
  --maxtext_model_path "${MODEL_BUCKET}/${IDX}" \
  --model_size         "${MODEL_NAME}"

echo "[orbax] scanned -> unscanned @ ${MODEL_BUCKET}/${IDX}/unscanned"
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_family_unscanned_ckpt \
  --base_model_path    "${CHKPT_BUCKET}" \
  --maxtext_model_path "${MODEL_BUCKET}/${IDX}/unscanned" \
  --model_size         "${MODEL_NAME}"
