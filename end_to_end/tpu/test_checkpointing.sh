#!/bin/bash
set -ex

if [ -f "saved_metrics.txt" ]; then
    rm saved_metrics.txt
    echo "removed existing saved_metrics.txt"
fi

if [ -f "restored_metrics.txt" ]; then
    rm restored_metrics.txt
    echo "removed existing restored_metrics.txt"
fi

RUN_NAME=${1}-${4}-$(date +%Y-%m-%d-%H-%M)
OUTPUT_PATH=${2}
DATASET_PATH=${3}
COLLECT_STACK_TRACE=${4}
DATASET_TYPE=${5}
eval_metrics=checkpoint_save_restore
model_params=" base_emb_dim=384 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=192 base_num_decoder_layers=8 head_dim=128"
CMD_DATA=""

if [ "$DATASET_TYPE" == "c4-array_record" ]
then
    eval_metrics=grain_checkpoint_save_restore
    echo "Using c4-array_record dataset type"
    echo "Mounting $DATASET_PATH to /tmp/gcsfuse/"
    bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$DATASET_PATH MOUNT_PATH=/tmp/gcsfuse/
    DATASET_PATH=/tmp/gcsfuse/
    CMD_DATA=" grain_worker_count=0 dataset_type=c4-array_record dataset_name=array-record/c4/en/3.0.1 eval_dataset_name=array-record/c4/en/3.0.1"
fi

#Train
CMD1="python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=5 max_target_length=128 per_device_batch_size=1\
    metrics_file=saved_metrics.txt checkpoint_period=3 base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
    async_checkpointing=false collect_stack_trace=$COLLECT_STACK_TRACE"
CMD1+=$model_params
CMD1+=$CMD_DATA

CMD2="python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=5 max_target_length=128 per_device_batch_size=1\
    metrics_file=restored_metrics.txt base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
    async_checkpointing=false collect_stack_trace=$COLLECT_STACK_TRACE"
CMD2+=$model_params
CMD2+=$CMD_DATA

echo
echo "Start the first training run"
echo "Command is:"
echo $CMD1

$CMD1
# Wait for first train to finish
# process_id=$!
# wait $process_id
echo
echo "First training run done"
echo "Start the second training run"
echo "Command is:"
echo $CMD2

$CMD2

python3 end_to_end/tpu/eval_assert.py $eval_metrics metrics.txt learning/loss
