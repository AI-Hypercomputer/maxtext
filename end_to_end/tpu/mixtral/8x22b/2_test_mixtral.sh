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
PREDICT_LEN=7
ATOL=60.0
RTOL=10.0

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

export DATASET_PATH=gs://maxtext-dataset

# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py` 
# export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt/0/items
export SCANNED_CHECKPOINT=gs://maxtext-external/Mixtral-8x22B-Instruct-v0.1-maxtext/0/items

export TOKENIZER_PATH=assets/tokenizer.mistral-v3

# Run decoding with converted ckpt - matmul implementation
python3 MaxText/decode.py MaxText/configs/base.yml \
  load_parameters_path=${SCANNED_CHECKPOINT} run_name=scanned_decoding \
  per_device_batch_size=1 model_name=mixtral-8x22b async_checkpointing=false \
  tokenizer_path=${TOKENIZER_PATH} ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=-1 max_prefill_predict_length=64 max_target_length=64 \
  prompt="[INST] I love to [/INST]" megablox=False weight_dtype=float16
 
# TODO(rdyro): add decoding test for megablox implementation
#python3 MaxText/decode.py MaxText/configs/base.yml \
#    load_parameters_path=${SCANNED_CHECKPOINT} run_name=scanned_decoding \
#    per_device_batch_size=1 model_name=mixtral-8x22b async_checkpointing=false \
#    tokenizer_path=${TOKENIZER_PATH} ici_tensor_parallelism=1 \
#    ici_fsdp_parallelism=-1 max_prefill_predict_length=16 max_target_length=24 \
#    prompt="[INST] I love to [/INST]" megablox=True weight_dtype=float16
 
# Test whether the forward pass logits match the golden logits - matmul implementation
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  load_parameters_path=${SCANNED_CHECKPOINT} run_name=matmul_forward_pass_test \
  per_device_batch_size=1 model_name=mixtral-8x22b \
  tokenizer_path=${TOKENIZER_PATH} ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=-1 max_prefill_predict_length=$PREDICT_LEN max_target_length=$PREDICT_LEN \
  dataset_type=synthetic dtype=bfloat16 weight_dtype=float16 megablox=False \
  --atol=$ATOL --rtol=$RTOL --token_size=$PREDICT_LEN
  # TODO(rdyro): figure out the reason for numerical mismatch for some tokens

python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  load_parameters_path=${SCANNED_CHECKPOINT} run_name=matmul_forward_pass_test \
  per_device_batch_size=1 model_name=mixtral-8x22b \
  tokenizer_path=${TOKENIZER_PATH} ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=-1 max_prefill_predict_length=$PREDICT_LEN max_target_length=$PREDICT_LEN \
  dataset_type=synthetic dtype=bfloat16 weight_dtype=bfloat16 megablox=False \
  --atol=$ATOL --rtol=$RTOL --token_size=$PREDICT_LEN
  # TODO(rdyro): figure out the reason for numerical mismatch for some tokens

# Test whether the forward pass logits match the golden logits - megablox implementation
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  load_parameters_path=${SCANNED_CHECKPOINT} run_name=megablox_forward_pass_test \
  per_device_batch_size=1 model_name=mixtral-8x22b \
  tokenizer_path=${TOKENIZER_PATH} ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=-1 max_prefill_predict_length=$PREDICT_LEN max_target_length=$PREDICT_LEN \
  dataset_type=synthetic dtype=bfloat16 weight_dtype=bfloat16 megablox=True \
  --atol=$ATOL --rtol=$RTOL --token_size=$PREDICT_LEN
  # TODO(rdyro): figure out the reason for numerical mismatch for some tokens

python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  load_parameters_path=${SCANNED_CHECKPOINT} run_name=megablox_forward_pass_test \
  per_device_batch_size=1 model_name=mixtral-8x22b \
  tokenizer_path=${TOKENIZER_PATH} ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=-1 max_prefill_predict_length=$PREDICT_LEN max_target_length=$PREDICT_LEN \
  dataset_type=synthetic dtype=bfloat16 weight_dtype=float16 megablox=True \
  --atol=$ATOL --rtol=$RTOL --token_size=$PREDICT_LEN
  # TODO(rdyro): figure out the reason for numerical mismatch for some tokens
 
# training

# Run pre-training without load_parameters_path - megablox implementation
python3 MaxText/train.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
  run_name=pre_training per_device_batch_size=1 enable_checkpointing=false \
  model_name=mixtral-8x22b ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 \
  steps=5 max_target_length=1024 async_checkpointing=false \
  tokenizer_path=${TOKENIZER_PATH} attention=flash dtype=bfloat16 \
  weight_dtype=bfloat16 megablox=True

# Run fine-tuning - megablox implementation
python3 MaxText/train.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
  load_parameters_path=${SCANNED_CHECKPOINT} run_name=fine_tuning \
  per_device_batch_size=1 model_name=mixtral-8x22b ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=-1 steps=10 max_target_length=1024 \
  async_checkpointing=false tokenizer_path=${TOKENIZER_PATH} checkpoint_period=100 \
  attention=flash dtype=bfloat16 weight_dtype=bfloat16 megablox=False
