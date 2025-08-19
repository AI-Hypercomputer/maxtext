echo "Running 64b.sh"
# 64B parameter model.
# This config will work out of the box for any number of v5e-256 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
#
# Example to invoke this script:
# bash MaxText/configs/v5e/64b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>"
#
# Example to AOT compile:
# bash MaxText/configs/v5e/64b.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY=v5e-256 M_COMPILE_TOPOLOGY_NUM_SLICES=2


# Stop execution if any command exits with error
set -e

export EXECUTABLE="train.py" # or train_compile.py
export RUN_PREFLIGHT="true"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Use 64b parameters if not set.
PARAMETERS="${PARAMETERS:-64}"
echo "Using ${PARAMETERS}b parameters"

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
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
JAX_PLATFORMS=cpu python3 -m MaxText.$EXECUTABLE MaxText/configs/base.yml\
    steps=$STEPS checkpoint_period=$CHECKPOINT_PERIOD per_device_batch_size=1 enable_checkpointing=true\
    jax_distributed_initialization_timeout=1800\
    async_checkpointing=false\
    remat_policy=full global_parameter_scale=$PARAMETERS\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    hardware=$HARDWARE\
    use_iota_embed=true reuse_example_batch=1\
    dataset_type=synthetic attention='flash' gcs_metrics=true\
    gcs_metrics_bucket=$GCS_METRICS_BUCKET\
    per_step_interval=$PER_STEP_INTERVAL\
    checkpoint_storage_target_data_file_size_bytes=$OCDBT_TARGET_DATA_FILE_SIZE\
    max_ckpts_to_keep=$MAX_CKPTS_TO_KEEP\
    enable_background_delete=$ENABLE_BACKGROUND_DELETE\
    final_ckpts_deletion_timeout_in_s=$FINAL_CKPTS_DELETION_TIMEOUT_IN_S\
    enable_single_replica_ckpt_restoring=$ENABLE_SINGLE_REPLICA_CKPT_RESTORING\
    use_replica_parallel=$USE_REPLICA_PARALLEL