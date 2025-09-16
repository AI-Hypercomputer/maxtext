#!/bin/bash

# This file is documentation for how to get started with DeepSeek v3 on v5p-256.

# The flow of this file is as follows:
# 1. Convert the checkpoint downloaded from HuggingFace to make it compatible with MaxText.
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.
# 3. Run pre-training, fine-tuning, and decoding.
# export HF_TOKEN

set -ex

export MODEL_NAME='gpt-oss-20b'
export TOKENIZER_PATH='openai/gpt-oss-20b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 1:
# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Non-Googlers please remember to use separate GCS paths for uploading model weights from HuggingFace ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
# You can use the HuggingFace checkpoint at TODO
export CHKPT_BUCKET=gs://maxtext-model-checkpoints/gpt-oss-20b/hf-bf16
JAX_PLATFORMS=cpu python3 -m MaxText.convert_gpt_oss_ckpt --base_model_path ${CHKPT_BUCKET} --maxtext_model_path ${BASE_OUTPUT_PATH}/scanned --model_size ${MODEL_NAME}

# Step 2:
# Note that the `SCANNED_CKPT_PATH` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
JAX_PLATFORMS=cpu python3 -m MaxText.convert_gpt_oss_unscanned_ckpt --base_model_path ${CHKPT_BUCKET} --maxtext_model_path ${BASE_OUTPUT_PATH}/unscanned --model_size ${MODEL_NAME}

# Step 3:
# We define `SCANNED_CKPT_PATH` to refer to the checkpoint subdirectory. This way it is easier to use this path in the `train.py` and `decode.py` commands
export SCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/scanned/0/items
export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/unscanned/0/items
# Non-Googlers please remember to point `DATASET_PATH` to the GCS bucket where you have your training data
export DATASET_PATH=gs://maxtext-dataset


# Run pre-training - megablox implementation
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml \
base_output_directory=${BASE_OUTPUT_DIRECTORY} run_name=megablox_pre_training model_name=${MODEL_NAME} \
tokenizer_type=huggingface tokenizer_path=${TOKENIZER_PATH} hf_access_token=${HF_TOKEN} dataset_type=synthetic \
enable_checkpointing=false \
attention=flash sparse_matmul=True megablox=True dtype=bfloat16 weight_dtype=bfloat16 \
per_device_batch_size=4 steps=5 max_target_length=1024 ici_fsdp_parallelism=4

# Run fine-tuning - megablox implementation
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml \
base_output_directory=${BASE_OUTPUT_DIRECTORY} run_name=megablox_fine_tuning model_name=${MODEL_NAME} \
tokenizer_type=huggingface tokenizer_path=${TOKENIZER_PATH} hf_access_token=${HF_TOKEN} dataset_path=${DATASET_PATH} \
enable_checkpointing=true async_checkpointing=false load_parameters_path=${SCANNED_CKPT_PATH} scan_layers=True \
attention=flash sparse_matmul=True megablox=True dtype=bfloat16 weight_dtype=bfloat16 \
per_device_batch_size=4 steps=5 max_target_length=1024 ici_fsdp_parallelism=4 

# Run supervised fine-tuning - megablox implementation
python3 -m MaxText.sft_trainer "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/sft.yml \
base_output_directory=${BASE_OUTPUT_DIRECTORY} run_name=megablox_supervised_fine_tuning model_name=${MODEL_NAME} \
tokenizer_type=huggingface tokenizer_path=${TOKENIZER_PATH} hf_access_token=${HF_TOKEN} dataset_type=hf \
enable_checkpointing=true async_checkpointing=false load_parameters_path=${SCANNED_CKPT_PATH} scan_layers=True \
attention=flash sparse_matmul=True megablox=True dtype=bfloat16 weight_dtype=bfloat16 \
per_device_batch_size=4 steps=5 max_target_length=1024 ici_fsdp_parallelism=1 ici_expert_parallelism=4 

# Run decoding - megablox implementation
python3 -m MaxText.decode "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml \
base_output_directory=${BASE_OUTPUT_DIRECTORY} run_name=decode model_name=${MODEL_NAME} \
tokenizer_type=huggingface tokenizer_path=${TOKENIZER_PATH} hf_access_token=${HF_TOKEN} \
load_parameters_path=${UNSCANNED_CKPT_PATH} scan_layers=False \
attention=dot_product sparse_matmul=True megablox=True dtype=bfloat16 weight_dtype=bfloat16 \
per_device_batch_size=1 max_prefill_predict_length=64 max_target_length=128 prompt="I love to" \
ici_fsdp_parallelism=1 ici_tensor_parallelism=4 

# Run forward logit, default golden_logits_path=/deps/src/MaxText/test_assets/golden_data_{model_name}.jsonl, copied from gs://maxtext-test-assets/golden_data_{model_name}.jsonl
python3 -m MaxText.tests.forward_pass_logit_checker "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} run_name=forward_logits_check model_name=${MODEL_NAME} \
load_parameters_path=${UNSCANNED_CKPT_PATH} scan_layers=false \
attention=dot_product sparse_matmul=True megablox=True per_device_batch_size=1 max_target_length=4 dtype=float32 \
--atol=0.1 --rtol=0.1 --max_kl_div=3e-4 --golden_logits_path=${GOLD}