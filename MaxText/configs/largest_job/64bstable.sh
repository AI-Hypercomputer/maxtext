
#bash docker_build_dependency_image.sh MODE=stable LIBTPU_GCS_PATH=gs://libtpu_internal/raymondzou/viperlite/2023-09-27-22:32:47-libtpu.so  (won't work for non-Googlers, contact Googlers for access)

echo "Running 64bstable.sh 26"
# Example command to invoke this script
# bash MaxText/configs/largest_job/64bstable.sh 

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod-useast"
export DATASET_PATH="gs://max-datasets-rogue-useast/"

# Set environment variables
#for ARGUMENT in "$@"; do
#    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
#    export "$KEY"="$VALUE"
#done

# Set up network
bash gke_rto_setup.sh

# For DNS lookup when running on large number of VMs
echo '142.250.123.95 www.googleapis.com' | tee -a /etc/hosts
echo '142.251.4.128 storage.googleapis.com' | tee -a /etc/hosts

export TPU_STDERR_LOG_LEVEL=0
export TPU_LOG_DIR=0

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=10000 per_device_batch_size=1 enable_checkpointing=true\
    enable_profiler=false remat_policy=full global_parameter_scale=64\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH use_iota_embed=true\
    reuse_example_batch=1\
    expansion_factor_real_data=16 enable_data_shuffling=false log_period=1000000 save_period=100\
    collect_stack_trace=false load_from_other_directory=gs://maxtext-experiments-multipod-useast/mattdavidow-o-save-scale64-slices1-a1/checkpoints
