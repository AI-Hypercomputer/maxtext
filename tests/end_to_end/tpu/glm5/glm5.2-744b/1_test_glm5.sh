#!/bin/bash

# This file is documentation for how to get started with GLM-5.2 (Cross-Layer IndexShare).
#
# Step 1 (this script, runs on CPU):
# Convert the HuggingFace bf16 checkpoint to MaxText-compatible bf16 checkpoints:
# - Scanned format (`scan_layers=true`): suitable for training and forward-pass logit checks.
# - Unscanned format (`scan_layers=false`): suitable for autoregressive decoding (prunes shared-layer indexers).
#
# Step 2 (`2_test_glm5.sh`, runs on TPU):
# Run forward-pass golden logit verification, distributed pre-training, and autoregressive decoding.

set -ex
set -o pipefail

export MODEL_NAME="${MODEL_NAME:-glm5.2-744b}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-zai-org/GLM-5.2}"

if [ -z "${BASE_OUTPUT_PATH:-}" ]; then
  # Non-Googlers please remember to point `BASE_OUTPUT_PATH` to GCS buckets that you own;
  # this script uses internal buckets for testing.
  export BASE_OUTPUT_PATH="gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)"
  echo "BASE_OUTPUT_PATH is not set; using ${BASE_OUTPUT_PATH}"
fi
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH%/}"
echo "Using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"

# Step 1: Checkpoint conversion
# Official HuggingFace checkpoint: https://huggingface.co/zai-org/GLM-5.2
# Standard MaxText GCS bucket convention for pre-uploaded bf16 HF weights:
#   gs://maxtext-model-checkpoints/glm5.2-744b/hf-bf16
# Provide either a local directory (`BF16_LOCAL_PATH` or `CKPT_DISK_LOCATION`)
# or `BF16_HF_PATH` (GCS URI or local path).
if [ -n "${CKPT_DISK_LOCATION:-}" ] && [ -z "${BF16_LOCAL_PATH:-}" ]; then
  export BF16_LOCAL_PATH="${CKPT_DISK_LOCATION}"
fi
if [ -z "${BF16_LOCAL_PATH:-}" ]; then
  : "${BF16_HF_PATH:?Please set BF16_HF_PATH (e.g. gs://maxtext-model-checkpoints/glm5.2-744b/hf-bf16) or BF16_LOCAL_PATH}"
  BF16_HF_PATH="${BF16_HF_PATH%/}"
  if [[ "${BF16_HF_PATH}" =~ ^gs:// ]]; then
    export BF16_LOCAL_PATH="/tmp/$(basename "${BF16_HF_PATH}")"
    mkdir -p "${BF16_LOCAL_PATH}"
    gcloud storage cp -r "${BF16_HF_PATH}/*" "${BF16_LOCAL_PATH}/"
  else
    export BF16_LOCAL_PATH="${BF16_HF_PATH}"
  fi
fi

# Install CPU torch for HuggingFace safetensors / checkpoint conversion
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

CONFIG_FILE="${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml"

# Convert checkpoint to `scanned` format (suitable for training and forward-pass logit verification)
python3 -m maxtext.checkpoint_conversion.to_maxtext "${CONFIG_FILE}" \
  model_name="${MODEL_NAME}" \
  scan_layers=true \
  attention=dot_product \
  base_output_directory="${BASE_OUTPUT_PATH}/scanned" \
  hardware=cpu \
  skip_jax_distributed_system=True \
  checkpoint_storage_concurrent_gb=1024 \
  --hf_model_path="${BF16_LOCAL_PATH}" \
  --lazy_load_tensors="${LAZY_LOAD_TENSORS:-False}" \
  --eager_load_method=safetensors \
  --save_dtype=bfloat16
export SCANNED_CKPT_PATH="${BASE_OUTPUT_PATH}/scanned/0/items"
echo "Scanned checkpoint path: ${SCANNED_CKPT_PATH}"

# Convert checkpoint to `unscanned` format (suitable for decoding)
python3 -m maxtext.checkpoint_conversion.to_maxtext "${CONFIG_FILE}" \
  model_name="${MODEL_NAME}" \
  scan_layers=false \
  attention=dot_product \
  base_output_directory="${BASE_OUTPUT_PATH}/unscanned" \
  hardware=cpu \
  skip_jax_distributed_system=True \
  checkpoint_storage_concurrent_gb=1024 \
  --hf_model_path="${BF16_LOCAL_PATH}" \
  --lazy_load_tensors="${LAZY_LOAD_TENSORS:-False}" \
  --eager_load_method=safetensors \
  --save_dtype=bfloat16
export UNSCANNED_CKPT_PATH="${BASE_OUTPUT_PATH}/unscanned/0/items"
echo "Unscanned checkpoint path: ${UNSCANNED_CKPT_PATH}"
