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

RUN_NAME=${1}-${4}
OUTPUT_PATH=${2}
DATASET_PATH=${3}
COLLECT_STACK_TRACE=${4}
DATASET_TYPE=${5}
ATTENTION=${6}
if [ -z "${6}" ]; then
    ATTENTION='autoselected'
fi
ASYNC_CHECKPOINTING=${7:-true}
eval_metrics=checkpoint_save_restore
model_params=" base_emb_dim=384 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=192 base_num_decoder_layers=8 head_dim=128"
CMD_DATA=""

if [ "$DATASET_TYPE" == "grain" ]
then
    eval_metrics=grain_checkpoint_save_restore
    echo "Using grain dataset type"
    echo "Mounting $DATASET_PATH to /tmp/gcsfuse/"
    bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$DATASET_PATH MOUNT_PATH=/tmp/gcsfuse/
    DATASET_PATH=/tmp/gcsfuse/
    CMD_DATA=" grain_worker_count=0 dataset_type=grain grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record*"
fi

# This command runs training for some steps and saves a checkpoint.
CMD1="python3 -m MaxText.train ${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/configs/base.yml run_name=$RUN_NAME steps=5 max_target_length=128 per_device_batch_size=1\
    metrics_file=saved_metrics.txt checkpoint_period=3 base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
    async_checkpointing=$ASYNC_CHECKPOINTING collect_stack_trace=$COLLECT_STACK_TRACE attention=$ATTENTION"
CMD1+=$model_params
CMD1+=$CMD_DATA

# This command restores the checkpoint from the previous run and continue training from the restored checkpoint.
# This ensures actual new training steps are executed after restoring checkpoint from the above training run.
CMD2="python3 -m MaxText.train ${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/configs/base.yml run_name=$RUN_NAME steps=10 max_target_length=128 per_device_batch_size=1\
    metrics_file=restored_metrics.txt base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
    async_checkpointing=$ASYNC_CHECKPOINTING collect_stack_trace=$COLLECT_STACK_TRACE attention=$ATTENTION"
CMD2+=$model_params
CMD2+=$CMD_DATA

echo
echo "Start the first training run"
echo "Command is:"
echo $CMD1

$CMD1

echo
echo "First training run done"
echo "Start the second training run"
echo "Command is:"
echo $CMD2

$CMD2

python3 end_to_end/tpu/eval_assert.py $eval_metrics metrics.txt learning/loss
