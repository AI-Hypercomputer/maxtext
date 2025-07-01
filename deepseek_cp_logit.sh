#!/bin/bash

# This file is documentation for how to get started with DeepSeek v2-Lite on v5p-8.

# The flow of this file is as follows:
# 1. Convert the checkpoint downloaded from HuggingFace to make it compatible with MaxText.
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.
# 3. Run logits check test between Huggingface and MaxText.
# 4. Run pre-training, fine-tuning, and decoding.

set -ex
idx=$(date +%Y-%m-%d-%H-%M)
export MODEL_NAME='deepseek2-16b'
export TOKENIZER_PATH='deepseek-ai/DeepSeek-V2-Lite'

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 1:
# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Non-Googlers please remember to use separate GCS paths for uploading model weights from HuggingFace ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
# You can use the HuggingFace checkpoint at https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
export LOCAL_HF_CHKPT_DIR=~/deepseek_v2_lite_hf_checkpoints/hf # <--- THIS IS YOUR LOCAL DOWNLOAD DIRECTORY
export CHKPT_BUCKET=gs://maxtext-deepseek/deepseek2-16b/hf
export MODEL_BUCKET=gs://pb-checkpoints-bucket-123456

# Step 0: Download the HuggingFace checkpoint locally (already done in Part 1)
# You should have already run: gcloud storage cp --recursive ${CHKPT_BUCKET}/ ${LOCAL_HF_CHKPT_DIR}/
# Add a check to ensure the directory exists and has files, or download if not present
if [ ! -d "${LOCAL_HF_CHKPT_DIR}" ] || [ -z "$(ls -A ${LOCAL_HF_CHKPT_DIR})" ]; then
  echo "Local HuggingFace checkpoint directory is empty or does not exist. Downloading now..."
  gcloud storage cp --recursive "${CHKPT_BUCKET}/" "${LOCAL_HF_CHKPT_DIR}/"
else
  echo "Local HuggingFace checkpoint directory found: ${LOCAL_HF_CHKPT_DIR}"
fi


JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_ckpt --base_model_path ${LOCAL_HF_CHKPT_DIR} --maxtext_model_path ${MODEL_BUCKET}/${idx} --model_size ${MODEL_NAME}

# Step 2:
# Note that the `CONVERTED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_unscanned_ckpt --base_model_path ${LOCAL_HF_CHKPT_DIR} --maxtext_model_path ${MODEL_BUCKET}/${idx}/unscanned --model_size ${MODEL_NAME}

# Step 3:
export UNSCANNED_CKPT_PATH=${MODEL_BUCKET}/${idx}/unscanned/0/items
python3 -m MaxText.tests.forward_pass_logit_checker MaxText/configs/base.yml tokenizer_type=huggingface tokenizer_path=deepseek-ai/DeepSeek-V2-Lite load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=forward_pass_test_${MODEL_NAME} per_device_batch_size=1 model_name=${MODEL_NAME} max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic scan_layers=false sparse_matmul=False dtype=float32 activations_in_float32=true matmul_precision=high ici_context_parallelism=4 --max_kl_div=2e-4
