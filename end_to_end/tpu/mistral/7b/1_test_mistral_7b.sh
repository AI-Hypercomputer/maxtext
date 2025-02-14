#!/bin/bash

# This file runs on daily basis (on a v4-8 cluster) and demonstrates:
# 1. Converts the Mistral PyTorch checkpoint to MaxText(orbax) format.
# 2. Loads the MaxText(orbax) checkpoint to run inference, and runs one forward pass on a given input.
# 3. Compares the logits to pre-computed logits obtained by running the HF checkpoint directly,
#    see scratch_code/golden-mistral-7b_export.ipynb and the resulting test_assets/golden_data_mistral-7b.jsonl

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mistral/7b/test_mistral-7b.sh

set -ex
RUN_ID=$(date +%Y-%m-%d-%H-%M)

export MODEL='mistral-7b'
export ASYNC_CHECKPOINTING=false
export CKPT_BUCKET=gs://maxtext-model-checkpoints
# `SCANNED_CHECKPOINT` is the path to the GCS bucket where we want to save our converted (Orbax) checkpoint. Non-Googlers please remember to point `SCANNED_CHECKPOINT` to a GCS bucket that you own
export SCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/scanned
export UNSCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}

# Installing torch for deps in forward_pass_logit_chekcker.py
pip install torch --index-url https://download.pytorch.org/whl/cpu

gcloud storage cp -r gs://maxtext-external/mistral-7B-v0.1 /tmp

# Convert it to MaxText(orbax) format - scanned ckpt
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path=/tmp/mistral-7B-v0.1 --model-size=${MODEL} --maxtext-model-path=${SCANNED_CHECKPOINT}

# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py`
export SCANNED_CHECKPOINT=${SCANNED_CHECKPOINT}/0/items

# Generate unscanned ckpt for efficient decoding test
JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml async_checkpointing=false base_output_directory=${UNSCANNED_CHECKPOINT} load_parameters_path=${SCANNED_CHECKPOINT} run_name=unscanned model_name='mistral-7b' force_unroll=true

