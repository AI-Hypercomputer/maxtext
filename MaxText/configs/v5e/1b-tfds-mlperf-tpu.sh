echo "Running 1b.sh"
pip freeze
# 16B parameter model.
# This config will work out of the box for any number of v5e-256 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#
# Example to invoke this script:
# bash MaxText/configs/v5e/16b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke"
#
# Example to AOT compile:
# bash MaxText/configs/v5e/16b.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY=v5e-256 M_COMPILE_TOPOLOGY_NUM_SLICES=2


# Stop execution if any command exits with error
set -e

export PLATFORM="gce"
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

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

# Use these setting for gcs
DATASET_PATH=gs://mlperf-exp-us-east1-cp0
DATASET_NAME=c4/en:3.0.7

# Use these setting for HF Hub
# DATASET_NAME=allenai/c4
# DATASET_DIR=en
# DATASET_PATH=''

# uncommment this if using gcsfuse
# echo "Mounting bucket to /tmp/gcsfuse/"
# bash setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=/tmp/gcsfuse
# DATASET_PATH=/tmp/gcsfuse/hf/c4/c4-train-*.parquet

# Train
#SMALL_MODEL="base_num_decoder_layers=4 base_mlp_dim=128 base_emb_dim=128 base_num_query_heads=4 base_num_kv_heads=4"

export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

JAX_PLATFORMS=tpu python3 MaxText/standalone_dataloader.py MaxText/configs/base.yml \
    steps=1000 per_device_batch_size=2 \
    enable_checkpointing=False \
    max_target_length=2048 global_parameter_scale=1 \
    base_output_directory=gs://aireenmei-multipod/tfds_scale \
    dataset_path=gs://mlperf-exp-us-east1-cp0 \
    dataset_name='c4/en:3.0.7' \
    eval_dataset_name='' \
    eval_split='validation_24567exp' \
    dataset_type=tfds \
    run_name=$(date +%m%d-%H%M)
    #model_name=gpt3-175b
    #run_name=$(date +%m%d-%H%M)
    #expansion_factor_real_data=4
