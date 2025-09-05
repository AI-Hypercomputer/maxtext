# Llama2 7B model.
# The batch size is set to achieve 4M token global batch size on 1 x v5p-512.
# If running on a different slice size, modify the per_device_batch_size field
# accordingly to achieve desired batch size.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml or using synthetic dataset_type)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
#
# Example to invoke this script:
# bash src/MaxText/configs/v5p/llama2_7b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>"
#
# Example to AOT compile:
# bash src/MaxText/configs/v5p/llama2_7b.sh EXECUTABLE=train_compile M_COMPILE_TOPOLOGY=v5p-512 M_COMPILE_TOPOLOGY_NUM_SLICES=1


# Stop execution if any command exits with error
set -e

export EXECUTABLE="train" # or train_compile
export DATASET_TYPE="synthetic" # synthetic data by default
export REUSE_EXAMPLE_BATCH=1
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
export LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 -m MaxText.$EXECUTABLE "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml model_name=llama2-7b\
  base_output_directory=$OUTPUT_PATH dataset_path=${DATASET_PATH}\
  tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/assets}"/tokenizer.llama2 remat_policy=minimal per_device_batch_size=4\
  steps=30 enable_checkpointing=false use_iota_embed=true max_target_length=4096\
  profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=5 gcs_metrics=true\
  dataset_type=$DATASET_TYPE reuse_example_batch=$REUSE_EXAMPLE_BATCH

