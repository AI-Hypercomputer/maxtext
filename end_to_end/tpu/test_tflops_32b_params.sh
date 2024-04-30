#!/bin/bash
echo "Running test_tflops_32b_params.sh"

# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
# TFLOP_THRESHOLD (Optional, default is 0 )
#
# Example to invoke this script:
# bash end_to_end/tpu/test_tflops_32b_params.sh RUN_NAME="<your_run_name>"" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke" TFLOP_THRESHOLD=0

# Stop execution if any command exits with error
set -ex

export TFLOP_THRESHOLD=0
export PLATFORM="gce"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=150 per_device_batch_size=4 enable_checkpointing=false\
    enable_profiler=false remat_policy=full\
    max_target_length=2048 metrics_file='metrics.txt' base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH log_period=150 global_parameter_scale=32

# Assert TFLOP/s
python3 end_to_end/tpu/eval_assert.py metrics_average metrics.txt $TFLOP_THRESHOLD perf/per_device_tflops_per_sec
