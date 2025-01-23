#!/bin/bash

# This file, combined with step 1 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Mistral PyTorch checkpoint to MaxText(orbax) format using a CPU VM.
# 2. Takes the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.

# The flow of this file is to take the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.
# Please make sure you have run end_to_end/tpu/mixtral/8x22b/1_test_mixtral.sh before running commands from this file.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mixtral/8x22b/2_test_mixtral.sh
# Use the same BASE_OUTPUT_PATH for both 1_test_mixtral.sh & 2_test_mixtral.sh.

set -ex
MODEL_VARIATION='8x22b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

export DATASET_PATH=gs://maxtext-dataset
export TOKENIZER_PATH=assets/tokenizer.mistral-v3

# Run pre-training without load_parameters_path - megablox implementation
python3 MaxText/train.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
  run_name=pre_training_megablox per_device_batch_size=4 enable_checkpointing=false \
  model_name=mixtral-8x22b ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 \
  steps=5 max_target_length=128 async_checkpointing=false \
  tokenizer_path=${TOKENIZER_PATH} attention=flash dtype=bfloat16 \
  weight_dtype=bfloat16 megablox=True sparse_matmul=True

# Run pre-training without load_parameters_path - matmul implementation
python3 MaxText/train.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
  run_name=pre_training_matmul per_device_batch_size=4 enable_checkpointing=false \
  model_name=mixtral-8x22b ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 \
  steps=5 max_target_length=128 async_checkpointing=false \
  tokenizer_path=${TOKENIZER_PATH} attention=flash dtype=bfloat16 \
  weight_dtype=bfloat16 megablox=False sparse_matmul=False

# Run pre-training without load_parameters_path - dropping implementation
python3 MaxText/train.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
  run_name=pre_training_dropping per_device_batch_size=4 enable_checkpointing=false \
  model_name=mixtral-8x22b ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 \
  steps=5 max_target_length=128 async_checkpointing=false \
  tokenizer_path=${TOKENIZER_PATH} attention=flash dtype=bfloat16 \
  weight_dtype=bfloat16 megablox=False sparse_matmul=False capacity_factor=1.25

# TODO(ranran): Add decoding, fine-tuning, and forward_pass_logit_checker tests after b/384580048
