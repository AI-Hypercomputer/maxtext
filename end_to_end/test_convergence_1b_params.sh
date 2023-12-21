#!/bin/bash

echo "Running test_convergence_1b_params.sh"
# Run this on 256 chips to achieve a loss value of ~2.6
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# LOSS_THRESHOLD (Optional, default is 100.0 )
#
# Example to invoke this script:
# bash end_to_end/test_convergence_1b_params.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" LOSS_THRESHOLD=100.0

# Stop execution if any command exits with error
set -e

export LOSS_THRESHOLD=100.0 # Set to large value so test is guaranteed to pass.

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=20400 per_device_batch_size=2.0 learning_rate=1e-4 enable_checkpointing=false\
    max_target_length=2048 global_parameter_scale=1\
    enable_profiler=false metrics_file='metrics.txt' base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH log_period=150

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/eval_assert.py final_loss metrics.txt $LOSS_THRESHOLD
