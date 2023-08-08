#!/bin/bash
set -e

lr=$1
int8=$2
scale=$3
num_slice=$4
init_key=$5
run_name=$6


base_steps=20000 # number of Chinchilla steps for 1B model on 1 pod for per_device_batch of 4 token length 1k
#steps=$(($base_steps * $scale / $num_slice))
steps=4205
echo "Running for $steps steps to hit Chinchilla number of tokens"

echo "GCS_RESOLVE_REFRESH_SECS=60" >> /etc/environment
echo "GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300" >> /etc/environment
echo "GCS_METADATA_REQUEST_TIMEOUT_SECS=300" >> /etc/environment
echo "GCS_READ_REQUEST_TIMEOUT_SECS=300" >> /etc/environment
echo "GCS_WRITE_REQUEST_TIMEOUT_SECS=600" >> /etc/environment

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
export TPU_VMODULE="retrying_utils=3,gcs_dns_cache=3"
export TF_CPP_VMODULE="retrying_utils=3,gcs_dns_cache=3"

command="python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${steps} per_device_batch_size=4 learning_rate=${lr} enable_profiler=false enable_checkpointing=true \
    save_period=600 enable_dropout=false run_name=${run_name}\
    base_output_directory=gs://maxtext-experiments-multipod\
    dataset_path=gs://max-datasets-rogue\
    use_int8_training=${int8}\
    remat_policy=full init_weights_seed=${init_key}\
    global_parameter_scale=${scale}"

echo "Starting run (${run_name}) with command: ${command}"
eval ${command}
echo "Finished command"
