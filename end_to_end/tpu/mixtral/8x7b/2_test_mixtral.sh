#!/bin/bash

# This file, combined with step 1 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Mistral PyTorch checkpoint to MaxText(orbax) format using a CPU VM.
# 2. Takes the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.

# The flow of this file is to take the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM. 
# Please make sure you have run end_to_end/tpu/mixtral/8x7b/1_test_mixtral.sh before running commands from this file. 

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mixtral/8x7b/2_test_mixtral.sh
# Use the same BASE_OUTPUT_PATH for both 1_test_mixtral.sh & 2_test_mixtral.sh.

set -ex
MODEL_VARIATION='8x7b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

export DATASET_PATH=gs://maxtext-dataset

# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py` 
export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}${MODEL_VARIATION}/scanned_ckpt/0/items

# Run decoding with converted ckpt - matmul implementation
# TODO(ranran): add decoding test for megablox implementation
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${SCANNED_CHECKPOINT} run_name=scanned_decoding per_device_batch_size=1 model_name=mixtral-8x7b async_checkpointing=false tokenizer_path=assets/tokenizer.mistral ici_tensor_parallelism=4 ici_fsdp_parallelism=16 max_prefill_predict_length=11 max_target_length=24 prompt="[INST] I love to [/INST]" megablox=False

# Test whether the forward pass logits match the golden logits - matmul implementation
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${SCANNED_CHECKPOINT} run_name=matmul_forward_pass_test per_device_batch_size=8 model_name=mixtral-8x7b tokenizer_path=assets/tokenizer.mistral ici_tensor_parallelism=4 ici_fsdp_parallelism=16 max_prefill_predict_length=11 max_target_length=11 dataset_type=synthetic dtype=float32 megablox=False --atol=3 --rtol=1 --token_size=4

# Test whether the forward pass logits match the golden logits - megablox implementation
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${SCANNED_CHECKPOINT} run_name=megablox_forward_pass_test per_device_batch_size=8 model_name=mixtral-8x7b tokenizer_path=assets/tokenizer.mistral ici_fsdp_parallelism=64 max_prefill_predict_length=11 max_target_length=11 dataset_type=synthetic dtype=bfloat16 weight_dtype=bfloat16 --atol=4 --rtol=1 --token_size=4

# Run fine-tuning - megablox implementation
python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} load_parameters_path=${SCANNED_CHECKPOINT} run_name=fine_tuning per_device_batch_size=8 model_name=mixtral-8x7b ici_fsdp_parallelism=64 steps=10 max_target_length=1024 async_checkpointing=false tokenizer_path=assets/tokenizer.mistral checkpoint_period=5 attention=flash dtype=bfloat16 weight_dtype=bfloat16

# Run pre-training without load_parameters_path - megablox implementation
python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} run_name=pre_training per_device_batch_size=8 enable_checkpointing=false model_name=mixtral-8x7b ici_fsdp_parallelism=64 steps=5 max_target_length=1024 async_checkpointing=false tokenizer_path=assets/tokenizer.mistral attention=flash dtype=bfloat16 weight_dtype=bfloat16

# TODO(ranran): Run decoding with unscanned ckpt
