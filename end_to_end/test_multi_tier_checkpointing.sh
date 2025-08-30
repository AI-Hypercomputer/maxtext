#!/bin/bash
set -ex

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}

export TPU_PREMAPPED_BUFFER_SIZE=20000014336
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=20000014336

# Train and save checkpoint
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml remat_policy=full base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH \
steps=100 enable_emergency_checkpoint=true checkpoint_period=200 local_checkpoint_directory=/local local_checkpoint_period=20 run_name=$RUN_NAME metrics_file='saved_metrics.txt'

# Retrieve checkpoint
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml remat_policy=full base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH \
steps=110 enable_emergency_checkpoint=true checkpoint_period=200 local_checkpoint_directory=/local local_checkpoint_period=20 run_name=$RUN_NAME metrics_file='restored_metrics.txt'


python3 end_to_end/tpu/eval_assert.py checkpoint_save_restore metrics.txt learning/loss

# Clean up ramdisk
rm -rf /local/*
