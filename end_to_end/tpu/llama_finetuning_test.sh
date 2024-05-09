#!/bin/bash

# This script is designed for internal use within Google. External users can adapt it by:
#  - Updating GCS paths (gs://) to your accessible locations.
#  - Using the checkpoint generated from train.py or available one in open source (https://llama.meta.com/llama-downloads/).

set -e
idx=$(date +%Y-%m-%d-%H-%M)

base_ckpt_path=gs://maxtext-llama/test/2024-01-15-06-49/decode-ckpt-maxtext/0/items
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
DATASET_PATH=gs://maxtext-dataset

export LOSS_THRESHOLD=2.5

python3 MaxText/train.py MaxText/configs/base.yml run_name=runner_direct_${idx} base_output_directory=${BASE_OUTPUT_DIRECTORY} load_parameters_path=${base_ckpt_path} model_name='llama2-7b' dataset_path=${DATASET_PATH} async_checkpointing=false  model_name='llama2-7b' ici_tensor_parallelism=4 steps=10 per_device_batch_size=.25 metrics_file='metrics.txt'

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/tpu/eval_assert.py final_loss metrics.txt $LOSS_THRESHOLD