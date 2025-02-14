#!/bin/bash

# This file, combined with step 2 in the same directory, demonstrates converting a Gemma checkpoint from Kaggle and running various MaxText operations on it.
# This step is tested nightly on an ordinary CPU VM.

# The flow of this file is as follows:
# 1. Pull the checkpoint from a GCS bucket and uploads the new MaxText compatible checkpoint to destination GCS bucket.
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/gemma/7b/1_test_gemma.sh
# Use the same BASE_OUTPUT_PATH as end_to_end/tpu/gemma/7b/2_test_gemma.sh.
# Please note that in these two scripts (1_test_gemma.sh and 2_test_gemma.sh) BASE_OUTPUT_PATH is assumed to be already a unique path across multiple runs and 
# the subfolders names aka RUN_NAMEs are static. Please remember to change BASE_OUTPUT_PATH across different runs.

set -ex
RUN_ID=$(date +%Y-%m-%d-%H-%M)

export MODEL='gemma-7b'
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

JAX_PLATFORMS=cpu python MaxText/convert_gemma_chkpt.py --base_model_path ${CHKPT_BUCKET}/7b --maxtext_model_path ${SCANNED_CHECKPOINT} --model_size 7b

# We define `SCANNED_CHECKPOINT` to refer to the checkpoint subdirectory exactly inside `SCANNED_CHECKPOINT`. This way it is easier to use this path in future commands
export SCANNED_CHECKPOINT=${SCANNED_CHECKPOINT}/0/items

# Note that the `SCANNED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
# We can do this by running `MaxText/generate_param_only_checkpoint.py` on `SCANNED_CHECKPOINT` with `force_unroll=true`. 
python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml base_output_directory=${UNSCANNED_CHECKPOINT} load_parameters_path=${SCANNED_CHECKPOINT} run_name=unscanned model_name=${MODEL} force_unroll=true

# TODO(b/369726357) Convert MaxText Gemma checkpoint to HF
