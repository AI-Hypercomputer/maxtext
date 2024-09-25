#!/bin/bash

# This file runs on daily basis (on a v4-8 cluster) and demonstrates:
# 1. Converts the Mistral PyTorch checkpoint to MaxText(orbax) format.
# 2. Loads the MaxText(orbax) checkpoint to run inference, and runs one forward pass on a given input.
# 3. Compares the logits to pre-computed logits obtained by running the HF checkpoint directly,
#    see scratch_code/golden-mistral-7b_export.ipynb and the resulting test_assets/golden_data_mistral-7b.jsonl

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mistral/7b/test_mistral-7b.sh

set -ex
MODEL_VARIATION='7b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

# Download checkpoint
pip3 install torch
gcloud storage cp -r gs://maxtext-external/mistral-7B-v0.1 /tmp

# Convert it to MaxText(orbax) format - scanned ckpt
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path=/tmp/mistral-7B-v0.1 --model-size=mistral-7b --maxtext-model-path=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt/
echo "Wrote MaxText compatible scanned checkpoint to ${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt"

# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py`
export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt/0/items

# Generate unscanned ckpt for efficient decoding test
export RUN_NAME=unscanned_ckpt
JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${SCANNED_CHECKPOINT} run_name=${RUN_NAME} model_name='mistral-7b' force_unroll=true
echo "Wrote MaxText compatible unscanned checkpoint to ${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints"

export DATASET_PATH=gs://maxtext-dataset

# Run decoding with converted ckpt - matmul implementation
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${SCANNED_CHECKPOINT} run_name=scanned_decoding per_device_batch_size=1 model_name=mistral-7b async_checkpointing=false tokenizer_path=assets/tokenizer.mistral-v1 max_prefill_predict_length=11 max_target_length=16 prompt="[INST] I love to [/INST]" attention=dot_product megablox=False

# Test whether the forward pass logits match the golden logits - matmul implementation
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${SCANNED_CHECKPOINT} run_name=matmul_forward_pass_test per_device_batch_size=1 model_name=mistral-7b tokenizer_path=assets/tokenizer.mistral-v1 max_prefill_predict_length=11 max_target_length=11 dataset_type=synthetic dtype=float32 megablox=False --atol=3 --rtol=1 --token_size=4
