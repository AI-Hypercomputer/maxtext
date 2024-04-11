#!/bin/bash
set -ex

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}
DATASET_TYPE=${4}

if [ "$DATASET_TYPE" == "c4-array_record" ]
then
    EVAL_METRICS=grain_checkpoint_save_restore
    echo "Using c4-array_record dataset type"
    echo "Mounting $DATASET_PATH to /tmp/gcsfuse/"
    bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$DATASET_PATH MOUNT_PATH=/tmp/gcsfuse/
    DATASET_PATH=/tmp/gcsfuse/
    CMD_DATA=" dataset_type=c4-array_record dataset_name=array-record/c4/en/3.0.1 eval_dataset_name=array-record/c4/en/3.0.1"
fi

#Train
CMD1="python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME}_1 steps=5 metrics_file=run_1_metrics.txt\
    enable_checkpointing=False enable_data_shuffling=True enable_dropout=False base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH"
CMD1+=$CMD_DATA


CMD2="python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME}_2 steps=5 metrics_file=run_2_metrics.txt\
    enable_checkpointing=False enable_data_shuffling=True enable_dropout=False base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH"
CMD2+=$CMD_DATA

$CMD1
$CMD2
python3 end_to_end/tpu/eval_assert.py determinism metrics.txt learning/loss 
