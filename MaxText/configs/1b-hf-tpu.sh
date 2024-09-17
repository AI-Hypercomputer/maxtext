echo "Running 1b.sh"
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
set -e -x
# dig google.com
# pip freeze

export PLATFORM="gce"
export EXECUTABLE="standalone_dataloader.py" # or train_compile.py

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

# copy tokenizer
gsutil cp -r gs://maxtext-dataset/hf/llama2-tokenizer assets/

# Use these setting for gcs
DATASET_FILES=gs://aireenmei-us-central1/maxtext-dataset/hf/c4_2048/c4-train-*-of-02048.parquet
DATASET_PATH=parquet
DATASET_DIR=''

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
JAX_PLATFORMS=tpu python3 MaxText/train.py MaxText/configs/base.yml \
    enable_checkpointing=True \
    steps=10 \
    base_output_directory=gs://aireenmei-multipod/cpu \
    hf_path=parquet \
    hf_train_files=gs://aireenmei-us-central1/maxtext-dataset/hf/c4_2048/c4-train-00000-of-02048.parquet \
    dataset_type=hf \
    tokenizer_path="assets/llama2-tokenizer" \
    run_name=$(date +%m%d-%H%M) \
    checkpoint_period=5
