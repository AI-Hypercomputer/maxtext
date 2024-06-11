# GPT-3 175B model.
# This config will work out of the box for any number of v5e-256 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#
# Example to invoke this script:
# bash MaxText/configs/v5e/gpt3_175b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke"
#
# Example to AOT compile:
# bash MaxText/configs/v5e/gpt3_175b.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY=v5e-256 M_COMPILE_TOPOLOGY_NUM_SLICES=8


# Stop execution if any command exits with error
set -e

export PLATFORM="gce" # Can be "gke" or "gce"
export EXECUTABLE="train.py" # or train_compile.py

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# The setup accommodates two cases:
# 1) Passing the 'RUN_NAME' variable at runtime
# 2) Propagating the 'M_RUN_NAME' variable within an Airflow sweeping workflow
if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

# Set up network
bash preflight.sh PLATFORM=$PLATFORM

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_spmd_rng_bit_generator_unsafe=true"

python3 MaxText/$EXECUTABLE MaxText/configs/base.yml model_name=gpt3-175b\
  steps=30 per_device_batch_size=0.5 enable_checkpointing=false\
  remat_policy=full ici_fsdp_parallelism=-1 ici_tensor_parallelism=16\
  max_target_length=2048 base_output_directory=$OUTPUT_PATH\
  reuse_example_batch=1 dataset_type=synthetic gcs_metrics=true\
  attention='flash'
