#!/bin/bash
set -ex

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}

export TPU_PREMAPPED_BUFFER_SIZE=20000014336
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=20000014336

# Train and save checkpoint
python3 -m MaxText.train MaxText/configs/base.yml remat_policy=full base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH \
steps=100 checkpoint_period=200 run_name=$RUN_NAME enable_emergency_checkpoint=true local_checkpoint_directory=/local local_checkpoint_period=20 use_replicator_service=True replicator_backup_interval_minutes=5 metrics_file='saved_metrics.txt'
