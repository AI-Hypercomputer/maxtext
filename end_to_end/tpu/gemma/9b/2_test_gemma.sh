#!/bin/bash

# This file, combined with step 2 in the same directory, demonstrates converting a Gemma checkpoint from Kaggle and running various MaxText operations on it.
# This step is tested nightly on an ordinary CPU VM.

# The flow of this file is as follows:
# 1. Pull the checkpoint from a GCS bucket and uploads the new MaxText compatible checkpoint to destination GCS bucket.
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/gemma/9b/1_test_gemma.sh
# Use the same BASE_OUTPUT_PATH as end_to_end/tpu/gemma/9b/2_test_gemma.sh.
# Please note that in these two scripts (1_test_gemma.sh and 2_test_gemma.sh) BASE_OUTPUT_PATH is assumed to be already a unique path across multiple runs and 
# the subfolders names aka RUN_NAMEs are static. Please remember to change BASE_OUTPUT_PATH across different runs.

set -ex
export MODEL_VARIATION='9b'


# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Please use separate GCS paths for uploading model weights from kaggle ($CHKPT_BUCKET) and MaxText compatible weights ($BASE_OUTPUT_PATH).
# Non-Googlers please remember to point CHKPT_BUCKET to GCS buckets that you own
export CHKPT_BUCKET=gs://maxtext-gemma/flax

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    # Use the same BASE_OUTPUT_PATH as end_to_end/tpu/gemma/7b/2_test_gemma.sh
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)/
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

# Non-Googlers please remember to point `DATASET_PATH` to the GCS bucket where you have your training data
export DATASET_PATH=gs://maxtext-dataset


# We define `CONVERTED_CHECKPOINT` to refer to the checkpoint subdirectory. This way it is easier to use this path in the `train.py` and `decode.py` commands
export CONVERTED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt/0/items
export RUN_NAME=unscanned_chkpt
# We defined path to unscanned checkpoint created in 1_test_gemma.sh
export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints/0/items

# We run decoding on the `UNSCANNED_CKPT_PATH` for efficient decoding on the unscanned version of the checkpoint. Note that this checkpoint only has parameters and no optimizer state. 
# So, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${UNSCANNED_CKPT_PATH} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false scan_layers=false model_name=gemma-9b attention=dot_product prompt="I love to"