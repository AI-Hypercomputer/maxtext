#!/bin/bash
set -e

int8_training=$1
global_parameter_scale=$2
remat_policy=$3
per_device_batch_size=$4
run_name=$5


STEPS=5

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"


python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${STEPS} per_device_batch_size=${per_device_batch_size} learning_rate=0.001 warmup_steps=2000 enable_profiler=true enable_checkpointing=false \
    enable_dropout=false enable_data_shuffling=false run_name=${run_name}\
    base_output_directory=gs://maxtext-experiments-multipod\
    dataset_path=gs://max-datasets-rogue\
    int8_training=${int8_training}\
    remat_policy=${remat_policy}\
    global_parameter_scale=${global_parameter_scale}
