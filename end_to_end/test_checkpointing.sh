#!/bin/bash
set -e

RUN_NAME=${1}-${4}-$(date +%Y-%m-%d-%H-%M)
OUTPUT_PATH=${2}
DATASET_PATH=${3}
COLLECT_STACK_TRACE=${4}
DATASET_TYPE=${5}
EVAL_METRICS=checkpoint_save_restore

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
CMD1="python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=7\
    metrics_file=saved_metrics.txt checkpoint_period=5 base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
    collect_stack_trace=$COLLECT_STACK_TRACE"
CMD1+=$CMD_DATA

CMD2="python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=7\
    metrics_file=restored_metrics.txt base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
    collect_stack_trace=$COLLECT_STACK_TRACE"
CMD2+=$CMD_DATA


$CMD1
# Wait for first train to finish
process_id=$!
wait $process_id

$CMD2

python3 end_to_end/eval_assert.py $EVAL_METRICS metrics.txt learning/loss
