# Mixtral-8x7b model.
# This config will work out of the box for any number of trillium-256 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
#
# Example to invoke this script:
# bash MaxText/configs/trillium/mixtral_8x7b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>"
#


# Stop execution if any command exits with error
set -e

export EXECUTABLE="train.py" # or train_compile.py
export RUN_PREFLIGHT="true"

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
if [ "$RUN_PREFLIGHT" = "true" ]; then
    bash preflight.sh
fi

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=81920 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"

python3 MaxText/$EXECUTABLE MaxText/configs/base.yml model_name=mixtral-8x7b\
  steps=15 per_device_batch_size=32 enable_checkpointing=false\
  remat_policy=full ici_fsdp_parallelism=-1\
  max_target_length=1024 base_output_directory=$OUTPUT_PATH\
  reuse_example_batch=1 dataset_type=synthetic gcs_metrics=true\
  attention='flash' sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048
