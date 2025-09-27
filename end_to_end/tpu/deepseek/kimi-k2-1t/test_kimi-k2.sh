#!/bin/bash
set -euo pipefail

# This file tests the implementation of Kimi-K2.

# The flow of this file is as follows:
# 1. Download the checkpoint from HuggingFace (fp8 weights).
# 2. Convert the checkpoint from FP8 to BF16 in HuggingFace format.
# 3. Upload the BF16 HuggingFace checkpoint to your GCS bucket.
# 4. Convert the BF16 HuggingFace checkpoint to a MaxText scanned Orbax checkpoint.
# 5. Convert the scanned checkpoint to an unscanned checkpoint for efficient decoding.
# 6. Run logits check test between HuggingFace and MaxText using the unscanned checkpoint.

export MODEL_NAME='kimi-k2-1t'
export TOKENIZER_PATH='moonshotai/Kimi-K2-Instruct'

export CHKPT_BUCKET='gs://maxtext-deepseek/kimi-k2-1t/hf'
export MODEL_BUCKET='gs://maxtext-deepseek/kimi-k2-1t'

# Local working dirs for HF weights
export HF_LOCAL_FP8_DIR="${PWD}/kimi-k2-fp8"
export HF_LOCAL_BF16_DIR="${PWD}/kimi-k2-bf16"

export BASE_CFG='src/MaxText/configs/base.yml'

# Environment / deps
echo "[setup] Installing dependencies..."
python3 -m pip install -q --disable-pip-version-check \
  torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
  safetensors==0.4.5 \
  transformers \
  huggingface_hub \
  jsonlines \
  google-cloud-storage

# Step 1: Download FP8 weights from Hugging Face
if [[ ! -d "${HF_LOCAL_FP8_DIR}" ]]; then
  echo "[step 1] Downloading ${TOKENIZER_PATH} into ${HF_LOCAL_FP8_DIR}"
  huggingface-cli download "${TOKENIZER_PATH}" \
    --local-dir "${HF_LOCAL_FP8_DIR}" \
    --local-dir-use-symlinks False
else
  echo "[step 1] Skipping download; ${HF_LOCAL_FP8_DIR} already exists"
fi

# Step 2: Convert FP8 -> BF16 in HuggingFace format
if [[ ! -d "${HF_LOCAL_BF16_DIR}" ]]; then
  echo "[step 2] Converting FP8 -> BF16 HF checkpoint"
  python3 -m MaxText.deepseek_fp8_to_bf16 \
    --input-fp8-hf-path  "${HF_LOCAL_FP8_DIR}" \
    --output-bf16-hf-path "${HF_LOCAL_BF16_DIR}"
else
  echo "[step 2] Skipping FP8->BF16; ${HF_LOCAL_BF16_DIR} already exists"
fi

# Step 3: Upload BF16 HF weights to GCS
# After downloading and converting checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Non-Googlers please remember to use separate GCS paths for uploading model weights from HuggingFace ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
echo "[step 3] Syncing BF16 HF checkpoint to ${CHKPT_BUCKET}"
gsutil -m rsync -r "${HF_LOCAL_BF16_DIR}" "${CHKPT_BUCKET}"

# Step 4: Convert HF (BF16) -> MaxText scanned Orbax
echo "[step 4] HF BF16 -> MaxText scanned Orbax"
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_family_ckpt \
  --base_model_path    "${CHKPT_BUCKET}" \
  --maxtext_model_path "${MODEL_BUCKET}/${idx}" \
  --model_size         "${MODEL_NAME}"

# Step 5: Convert scanned -> unscanned Orbax
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_family_unscanned_ckpt \
  --base_model_path    "${CHKPT_BUCKET}" \
  --maxtext_model_path "${MODEL_BUCKET}/${idx}/unscanned" \
  --model_size         "${MODEL_NAME}"

export UNSCANNED_CKPT_PATH="${MODEL_BUCKET}/${idx}/unscanned/0/items"

# Step 6: Logit check (MaxText vs HF) using unscanned checkpoint
echo "[step 6] Running forward_pass_logit_checker (unscanned ckpt, bf16 dtype)"
python3 -m tests.forward_pass_logit_checker \
  "${BASE_CFG}" \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  load_parameters_path="${UNSCANNED_CKPT_PATH}" \
  run_name="forward_pass_test_${MODEL_NAME}_hf_live_unscanned" \
  per_device_batch_size=1 \
  model_name="${MODEL_NAME}" \
  max_prefill_predict_length=16 \
  max_target_length=16 \
  dataset_type=synthetic \
  scan_layers=false \
  sparse_matmul=False \
  dtype=float32 \
  activations_in_float32=true \
  matmul_precision=high \
  --run_hf_model=true \
  --hf_model_path="${TOKENIZER_PATH}" \
  --max_kl_div=2e-4

echo "[done] Cross-check completed."
