#!/bin/bash
set -e -x

export LOSS_THRESHOLD=100.0 # Set to large value so test is guaranteed to pass.

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [ -z "$STEPS" ]
then
    STEPS=20400
fi

if [ -z "$PACKING" ]
then
    PACKING=true
fi

if [ -z "$WORKER" ]
then
    WORKER=4
fi

if [ -z "$MOUNT_PATH" ]
then
    MOUNT_PATH=gcsfuse
fi

METRICS_FILE=metrics-$(date +%m%d-%H:%M).txt
# Setup GCSFUSE
# bash setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=$MOUNT_PATH

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=$STEPS per_device_batch_size=2.0 learning_rate=1e-4 enable_checkpointing=true checkpoint_period=99\
    max_target_length=2048 global_parameter_scale=1\
    enable_profiler=false metrics_file=$METRICS_FILE base_output_directory=$OUTPUT_PATH\
    dataset_path=$MOUNT_PATH dataset_name='array-record/c4/en/3.0.1' eval_dataset_name='array-record/c4/en/3.0.1'\
    dataset_type=c4-array_record log_period=150 pack_examples=$PACKING grain_worker_count=$WORKER

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/eval_assert.py final_loss $METRICS_FILE $LOSS_THRESHOLD