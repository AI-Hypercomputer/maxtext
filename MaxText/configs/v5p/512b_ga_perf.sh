echo "Running 512b.sh"
# 512B parameter model.
# This config will work out of the box for any number of v5p-1024 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#
# Example to invoke this script:
# bash MaxText/configs/v5p/512b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke"
#
# Example to AOT compile:
# bash MaxText/configs/v5p/512b.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY=v5p-1024 M_COMPILE_TOPOLOGY_NUM_SLICES=2


# Stop execution if any command exits with error
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

# debug
export TPU_STDERR_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0

# hlo dump
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_pass_re=spmd|sharding"

WORK_ID=$(grep '^WORKER_ID' /tmp/tpu-env | cut -d "'" -f 2)
echo "WORK_ID=${WORK_ID}"

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_experimental_fusion_cost_model=false --xla_tpu_dot_dot_fusion_duplicated=false --xla_tpu_dot_dot_fusion=false --xla_jf_conv_input_fusion=true --xla_jf_conv_output_fusion=false --xla_tpu_rwb_fusion=false  --xla_tpu_copy_fusion_pad_unpad_ratio=300 --xla_tpu_enable_aggressive_loop_fusion_layout_opt=false --xla_tpu_enable_copy_fusion=false --xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=false --xla_tpu_scavenge_vmem_for_fusions=false --xla_tpu_vector_load_fusion_window=256 --xla_tpu_vector_store_fusion_window=64 --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_megacore_fusion=true --xla_enable_async_all_gather=true --xla_enable_async_collective_permute=true --xla_always_enable_all_gather_2d_asymmetric=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_dcn_max_overlap_estimation=32"
python3 MaxText/$EXECUTABLE MaxText/configs/base.yml\
    skip_first_n_steps_for_profiler=15 profiler=xplane\
    steps=20 per_device_batch_size=2 enable_checkpointing=false\
    remat_policy=save_qkv_proj global_parameter_scale=512\
    ici_fsdp_parallelism=-1 ici_tensor_parallelism=8\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH use_iota_embed=true reuse_example_batch=1\
    dataset_type=synthetic gcs_metrics=true attention='flash' quantization="int8"

if [[ $WORK_ID == '0' ]];then
  gsutil -m cp -r /tmp/xla_dump_file "$OUTPUT_PATH/$RUN_NAME/xla"
fi

gsutil -m cp -r /tmp/tpu_logs "$OUTPUT_PATH/$RUN_NAME/tpu_logs"
