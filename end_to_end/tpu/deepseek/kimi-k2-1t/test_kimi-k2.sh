#!/bin/bash
set -euo pipefail

# This file tests the implementation of Kimi-K2.

# The flow of this file is as follows:
# 1. Convert the checkpoint downloaded from HuggingFace to make it compatible with MaxText.
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.
# 3. Run logits check test between Huggingface and MaxText.

export MODEL_NAME='kimi-k2-1t'
export TOKENIZER_PATH='moonshotai/Kimi-K2-Instruct'

export CHKPT_BUCKET='gs://maxtext-deepseek/kimi-k2-1t/hf'
export MODEL_BUCKET='gs://maxtext-deepseek/kimi-k2-1t'
export idx=0

export BASE_CFG='src/MaxText/configs/base.yml'

# Environment / deps
echo "[setup] Installing minimal torch wheel for forward_pass_logit_checker deps..."
python3 -m pip install -q --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu

# Step 1:
echo "[convert] Converting HF checkpoint to MaxText scanned Orbax..."
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_family_ckpt \
  --base_model_path "${CHKPT_BUCKET}" \
  --maxtext_model_path "${MODEL_BUCKET}/${idx}" \
  --model_size "${MODEL_NAME}"

# Step 2:
echo "[convert] Creating unscanned Orbax..."
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_family_unscanned_ckpt \
  --base_model_path "${CHKPT_BUCKET}" \
  --maxtext_model_path "${MODEL_BUCKET}/${idx}/unscanned" \
  --model_size "${MODEL_NAME}"

# Step 3:
export SCANNED_CKPT_PATH="${MODEL_BUCKET}/${idx}/0/items"

echo "[check] Running forward_pass_logit_checker"
python3 -m tests.forward_pass_logit_checker \
  "${BASE_CFG}" \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  load_parameters_path="${UNSCANNED_CKPT_PATH}" \
  run_name="forward_pass_test_${MODEL_NAME}_hf_live" \
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
