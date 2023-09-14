echo "Running 32b.sh"
# Example command to invoke this script
# bash MaxText/configs/largest_job/32b.sh

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod"
export DATASET_PATH="gs://maxtext-dataset/"
export TPU_LIBRARY_PATH=/root/custom_libtpu/libtpu.so

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Set up network
bash gke_rto_setup.sh

# For DNS lookup when running on large number of VMs
echo '142.250.123.95 www.googleapis.com' | tee -a /etc/hosts
echo '142.251.4.128 storage.googleapis.com' | tee -a /etc/hosts

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=30 per_device_batch_size=4 enable_checkpointing=false\
    enable_profiler=false remat_policy=full global_parameter_scale=32\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH use_iota_embed=true reuse_example_batch=1\
    dataset_type=synthetic enable_flash_attention=True
