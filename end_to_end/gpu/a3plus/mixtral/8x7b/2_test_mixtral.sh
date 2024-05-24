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

export M_ENABLE_CHECKPOINTING=true
export M_BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_PATH}${MODEL_VARIATION}
export M_DATASET_PATH=gs://maxtext-dataset
export M_ASYNC_CHECKPOINTING=false


# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py` 
export SCANNED_CHECKPOINT=${M_BASE_OUTPUT_DIRECTORY}/scanned_ckpt/0/items
export UNSCANNED_CHECKPOINT=${M_BASE_OUTPUT_DIRECTORY}/unscanned_ckpt/0/items

# Verify model correctness
# python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml dataset_type=synthetic hardware=gpu dataset_path=gs://maxtext-dataset base_output_directory=${BASE_OUTPUT_PATH}${MODEL_VARIATION} load_parameters_path=${SCANNED_CHECKPOINT} run_name=scanned_decoding per_device_batch_size=1 model_name=mixtral-8x7b tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral max_prefill_predict_length=11 max_target_length=24 prompt="[INST] I love to [/INST]" autoregressive_decode_assert="That's great to hear! I love to learn new things" attention=dot_product

# Run decoding with converted ckpt
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${SCANNED_CHECKPOINT} \
    run_name=scanned_decoding per_device_batch_size=1 model_name=mixtral-8x7b \
    tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral \
    max_prefill_predict_length=11 max_target_length=24 prompt="[INST] I love to [/INST]" \
    autoregressive_decode_assert="That's great to hear! I love to learn new things" attention=dot_product

# Run fine-tuning
# python3 MaxText/train.py MaxText/configs/base.yml load_parameters_path=${SCANNED_CHECKPOINT} run_name=fine_tuning per_device_batch_size=1 model_name=mixtral-8x7b steps=10 max_target_length=1024 tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral checkpoint_period=5 attention=dot_product

# Run pre-training without load_parameters_path
# python3 MaxText/train.py MaxText/configs/base.yml run_name=pre_training per_device_batch_size=1 model_name=mixtral-8x7b steps=5 max_target_length=1024 tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral attention=dot_product

# TODO(ranran): Run decoding with unscanned ckpt

# Test whether the forward pass logits match the golden logits - megablox implementation
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${MATMUL_SCANNED_CHECKPOINT} \
    run_name=megablox_forward_pass_test per_device_batch_size=1 model_name=mixtral-8x7b \
    tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral \
    ici_tensor_parallelism=4 ici_fsdp_parallelism=16 \
    max_prefill_predict_length=11 max_target_length=11 \
    dataset_type=synthetic dtype=float32 moe_matmul=True megablox=True --atol=3 --rtol=1 --token_size=4
