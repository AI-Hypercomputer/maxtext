#!/bin/bash

# This file runs once a day on a CPU and has follows: 

# The flow of this file is as follows:
# 1. Convert the checkpoint downloaded from Kaggle to make it compatible with MaxText
# 2. Create MaxText compatible unscanned orbax checkpoint

set -ex
RUN_ID=$(date +%Y-%m-%d-%H-%M)

export MODEL='gemma-2b'
export ASYNC_CHECKPOINTING=false
export CKPT_BUCKET=gs://maxtext-model-checkpoints
# `SCANNED_CHECKPOINT` is the path to the GCS bucket where we want to save our converted (Orbax) checkpoint. Non-Googlers please remember to point `SCANNED_CHECKPOINT` to a GCS bucket that you own
export SCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/scanned
export UNSCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}
export HF_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/huggingface

# Installing torch for deps in forward_pass_logit_chekcker.py
pip install torch --index-url https://download.pytorch.org/whl/cpu

# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Non-Googlers please remember to use separate GCS paths for uploading model weights from kaggle ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
export CHKPT_BUCKET=gs://maxtext-gemma/flax

JAX_PLATFORMS=cpu python MaxText/convert_gemma_chkpt.py --base_model_path ${CHKPT_BUCKET}/2b --maxtext_model_path ${SCANNED_CHECKPOINT} --model_size 2b

# We define `SCANNED_CHECKPOINT` to refer to the checkpoint subdirectory exactly inside `SCANNED_CHECKPOINT`. This way it is easier to use this path in future commands
export SCANNED_CHECKPOINT=${SCANNED_CHECKPOINT}/0/items

# Note that the `SCANNED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
# We can do this by running `MaxText/generate_param_only_checkpoint.py` on `SCANNED_CHECKPOINT` with `force_unroll=true`. 
JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml base_output_directory=${UNSCANNED_CHECKPOINT} load_parameters_path=${SCANNED_CHECKPOINT} async_checkpointing=${ASYNC_CHECKPOINTING} run_name=unscanned model_name=${MODEL} force_unroll=true
