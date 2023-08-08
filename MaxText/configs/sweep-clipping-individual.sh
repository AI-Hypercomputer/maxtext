#!/bin/bash
set -e

lr=$1
int8=$2
scale=$3
num_slice=$4
clipping=$5
init_key=$6
run_name=$7


base_steps=20000 # number of Chinchilla steps for 1B model on 1 pod for per_device_batch of 4 token length 1k
steps=$(($base_steps * $scale / $num_slice))
echo "Running for $steps steps to hit Chinchilla number of tokens"

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"


command="python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${steps} per_device_batch_size=4 learning_rate=${lr} enable_profiler=false enable_checkpointing=true \
    save_period=2000 enable_dropout=false run_name=${run_name}\
    base_output_directory=gs://maxtext-experiments-multipod\
    dataset_path=gs://max-datasets-rogue\
    use_int8_training=${int8}\
    remat_policy=full init_weights_seed=${init_key}\
    global_parameter_scale=${scale}
    gradient_clipping_threshold=${clipping}"

echo "Starting run (${run_name}) with command: ${command}"
eval ${command}
echo "Finished command"
