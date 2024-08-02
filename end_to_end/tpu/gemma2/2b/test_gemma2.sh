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
export MODEL_VARIATION='2b'


# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Please use separate GlsS paths for uploading model weights from kaggle ($CHKPT_BUCKET) and MaxText compatible weights ($BASE_OUTPUT_PATH).
# Non-Googlers please remember to point CHKPT_BUCKET to GCS buckets that you own
export CHKPT_BUCKET=gs://zyc_maxtext/gemma/gemma2-2b/ckpt
export BASE_OUTPUT_PATH=gs://zyc_maxtext/runner-maxtext-logs/gemma2-2b

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    # Use the same BASE_OUTPUT_PATH as end_to_end/tpu/gemma/7b/2_test_gemma.sh
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)/
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

echo "Converted checkpoints are stored at ${BASE_OUTPUT_PATH}"


JAX_PLATFORMS=cpu python MaxText/convert_gemma2_chkpt.py --base_model_path ${CHKPT_BUCKET} --maxtext_model_path ${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt --model_size ${MODEL_VARIATION}
echo "Wrote MaxText compatible checkpoint to ${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt"

# We define `CONVERTED_CHECKPOINT` to refer to the checkpoint subdirectory.
export CONVERTED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt/0/items
# Note that the `CONVERTED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
# We can do this by running `MaxText/generate_param_only_checkpoint.py` on `CONVERTED_CHECKPOINT` with `force_unroll=true`. 
export RUN_NAME=unscanned_chkpt
JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${CONVERTED_CHECKPOINT} run_name=${RUN_NAME} model_name='gemma2-2b' force_unroll=true
echo "Written MaxText compatible unscanned checkpoint to ${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints/0/items"


############# test decode
export MODEL_VARIATION='2b'
# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Please use separate GlsS paths for uploading model weights from kaggle ($CHKPT_BUCKET) and MaxText compatible weights ($BASE_OUTPUT_PATH).
# Non-Googlers please remember to point CHKPT_BUCKET to GCS buckets that you own
export CHKPT_BUCKET=gs://zyc_maxtext/gemma/gemma2-2b/ckpt
export BASE_OUTPUT_PATH=gs://zyc_maxtext/runner-maxtext-logs/gemma2-2b
export RUN_NAME=unscanned_chkpt

export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints/0/items
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${UNSCANNED_CKPT_PATH} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false scan_layers=false model_name=gemma2-2b prompt="I love to"

python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${CONVERTED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma2-2b prompt="I love to"
############## test 2b config
# export BASE_OUTPUT_PATH=gs://zyc_maxtext/runner-maxtext-logs/gemma-2b/
# export DATASET_PATH=gs://maxtext-dataset

# python MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} tokenizer_path=assets/tokenizer.gemma per_device_batch_size=1 run_name=runner_pretrain_1 max_target_length=8192 steps=5 enable_checkpointing=false model_name=gemma-2b

# compare with golden logits

python3 MaxText/tests/forward_pass_logit_checker.py  MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path='gs://zyc_maxtext/runner-maxtext-logs/gemma2-2b/unscanned_chkpt/checkpoints/0/items' run_name=forward_pass_test_gemma2-2b per_device_batch_size=1 model_name=gemma2-2b max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic scan_layers=false 
