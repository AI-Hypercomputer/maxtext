#!/usr/bin/env bash
set -euo pipefail
source ./pipeline.env

UNSCANNED="${MODEL_BUCKET}/${IDX}/unscanned/0/items"
echo "[check] Using unscanned checkpoint: ${UNSCANNED}"

python3 -m tests.forward_pass_logit_checker \
  "${BASE_CFG}" \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  load_parameters_path="${UNSCANNED}" \
  run_name="forward_pass_test_${MODEL_NAME}_hf_live_unscanned" \
  per_device_batch_size=1 \
  model_name="${MODEL_NAME}" \
  max_prefill_predict_length=16 \
  max_target_length=16 \
  dataset_type=synthetic \
  scan_layers=false \
  sparse_matmul=False \
  dtype=bfloat16 \
  activations_in_float32=true \
  matmul_precision=high \
  --run_hf_model=true \
  --hf_model_path="${TOKENIZER_PATH}" \
  --max_kl_div=2e-4

echo "[check] Done."
