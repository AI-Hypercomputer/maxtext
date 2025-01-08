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

# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py` 
export SCANNED_CHECKPOINT=gs://ranran-multipod-dev/xlml/8x22b/origin/8x22b/scanned_ckpt/0/items

export TOKENIZER_PATH=assets/tokenizer.mistral-v3

# TODO(ranran): enable the fine-tuning, decoding, and forward_pass_logit_checker tests once b/380148614 has been fixed

# Run pre-training without load_parameters_path - megablox implementation
python3 MaxText/train.py MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
  run_name=pre_training per_device_batch_size=1 enable_checkpointing=True \
  model_name=mixtral-8x22b ici_tensor_parallelism=4 ici_fsdp_parallelism=16 \
  steps=5 max_target_length=1024 async_checkpointing=True \
  tokenizer_path=${TOKENIZER_PATH} attention=flash dtype=bfloat16 \
  weight_dtype=bfloat16 megablox=False


python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${CONVERTED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=True model_name=${MODEL_VARIATION} attention=dot_product prompt="I love to"


# TODO(ranran): add decoding test for megablox implementation
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${SCANNED_CHECKPOINT} run_name=unscanned_decoding_1 per_device_batch_size=1 model_name=mixtral-8x22b async_checkpointing=false tokenizer_path=assets/tokenizer.mistral-v3 ici_tensor_parallelism=4 ici_fsdp_parallelism=16 max_prefill_predict_length=11 max_target_length=24 prompt="[INST] I love to [/INST]" megablox=False

# Run decoding with converted ckpt - dropping implementation
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${SCANNED_CHECKPOINT} run_name=unscanned_decoding_2 per_device_batch_size=1 model_name=mixtral-8x22b async_checkpointing=false tokenizer_path=assets/tokenizer.mistral-v3 ici_tensor_parallelism=4 ici_fsdp_parallelism=16 max_prefill_predict_length=11 max_target_length=24 prompt="[INST] I love to [/INST]" megablox=False capacity_factor=1.25

