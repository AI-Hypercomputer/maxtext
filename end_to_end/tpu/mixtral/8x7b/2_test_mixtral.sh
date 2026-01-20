#!/bin/bash

# This file, combined with step 1 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Mixtral PyTorch checkpoint to MaxText(orbax) format using a CPU VM.
# 2. Takes the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.

# The flow of this file is to take the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.
# Please make sure you have run end_to_end/tpu/mixtral/8x7b/1_test_mixtral.sh before running commands from this file.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mixtral/8x7b/2_test_mixtral.sh
# Use the same BASE_OUTPUT_PATH for both 1_test_mixtral.sh & 2_test_mixtral.sh.

set -ex

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

MODEL_VARIATION='8x7b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
  export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)
  echo "BASE_OUTPUT_PATH is not set"
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}

export DATASET_PATH=gs://maxtext-dataset

# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py`
export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt/0/items

# `UNSCANNED_CHECKPOINT` refers to run decoding
export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/unscanned_ckpt/checkpoints/0/items

# Run decoding with converted ckpt - matmul implementation
# TODO(ranran): add decoding test for megablox implementation
python3 -m MaxText.decode "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=unscanned_decoding per_device_batch_size=1 model_name=mixtral-8x7b async_checkpointing=false tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.mistral-v1 ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 max_prefill_predict_length=11 max_target_length=24 prompt='"[INST] I love to [/INST]"' megablox=False sparse_matmul=False scan_layers=false

# Run decoding with converted ckpt - dropping implementation
python3 -m MaxText.decode "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=unscanned_decoding per_device_batch_size=1 model_name=mixtral-8x7b async_checkpointing=false tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.mistral-v1 ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 max_prefill_predict_length=11 max_target_length=24 prompt='"[INST] I love to [/INST]"' megablox=False sparse_matmul=False scan_layers=false capacity_factor=1.25

# Test whether the forward pass logits match the golden logits - matmul implementation
python3 -m tests.forward_pass_logit_checker "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=matmul_forward_pass_test per_device_batch_size=1 model_name=mixtral-8x7b tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.mistral-v1 ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 max_prefill_predict_length=11 max_target_length=11 dtype=float32 megablox=False sparse_matmul=False scan_layers=false --token_size=4 --max_kl_div=3e-3

# To repeat duplicate tests, we have MoE unit test to verify outputs matching for matmul, megablox, and ragged_dot implementation at https://github.com/AI-Hypercomputer/maxtext/blob/5c4090b8d5713a1a25cab85df89b0ec9c9862635/MaxText/tests/moe_test.py#L338-L411

# Run pre-training - megablox implementation
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} run_name=megablox_pre_training per_device_batch_size=4 enable_checkpointing=false model_name=mixtral-8x7b ici_fsdp_parallelism=-1 steps=5 max_target_length=1024 async_checkpointing=false tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.mistral-v1 attention=flash dtype=bfloat16 weight_dtype=bfloat16

# Run pre-training - matmul implementation
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} run_name=matmul_pre_training per_device_batch_size=4 enable_checkpointing=false model_name=mixtral-8x7b ici_fsdp_parallelism=-1 steps=5 max_target_length=1024 async_checkpointing=false tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.mistral-v1 attention=flash dtype=bfloat16 weight_dtype=bfloat16 megablox=False sparse_matmul=False

# Run pre-training - dropping implementation
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} run_name=dropping_pre_training per_device_batch_size=4 enable_checkpointing=false model_name=mixtral-8x7b ici_fsdp_parallelism=-1 steps=5 max_target_length=1024 async_checkpointing=false tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.mistral-v1 attention=flash dtype=bfloat16 weight_dtype=bfloat16 megablox=False sparse_matmul=False capacity_factor=1
