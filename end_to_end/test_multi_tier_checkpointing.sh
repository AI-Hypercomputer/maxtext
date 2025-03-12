#!/bin/bash
set -ex

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}
CHECKPOINT_PATH=gs://$OUTPUT_PATH/$RUN_NAME/checkpoints/

export TPU_PREMAPPED_BUFFER_SIZE=20000014336
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=20000014336

# Check if the destination directory already exists
if gsutil ls "${CHECKPOINT_PATH}" 2> /dev/null 1> /dev/null; then
    echo "${CHECKPOINT_PATH} already exists. Skipping copy."
else
    echo "Copying ${RUN_NAME} to ${OUTPUT_PATH}"
    mkdir -p $RUN_NAME/checkpoints/
    touch $RUN_NAME/checkpoints/$RUN_NAME.txt

    gsutil cp -r $RUN_NAME gs://$OUTPUT_PATH/
    rm -rf $RUN_NAME
fi

# Train and save checkpoint
python3 MaxText/train.py MaxText/configs/base.yml remat_policy=full base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH \
steps=100 enable_emergency_checkpoint=true checkpoint_period=200 local_checkpoint_directory=/local local_checkpoint_period=20 run_name=$RUN_NAME metrics_file='saved_metrics.txt'

# Retrieve checkpoint
python3 MaxText/train.py MaxText/configs/base.yml remat_policy=full base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH \
steps=110 enable_emergency_checkpoint=true checkpoint_period=200 local_checkpoint_directory=/local local_checkpoint_period=20 run_name=$RUN_NAME metrics_file='restored_metrics.txt'


python3 end_to_end/tpu/eval_assert.py checkpoint_save_restore metrics.txt learning/loss

# Clean up ramdisk

rm -rf /local/*
