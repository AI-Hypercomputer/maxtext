#!/bin/bash
set -e

fwd_int8=$1
bwd_int8=$2
global_parameter_scale=$3
prng_key=$4
run_name=$5

steps=3001

output_file=gs://mattdavidow-maxtext-br/${run_name}.txt

remat_policy=full

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"


command="python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${steps} per_device_batch_size=4 learning_rate=0.001 warmup_steps=1000 enable_profiler=false enable_checkpointing=true \
    enable_dropout=false enable_data_shuffling=false run_name=${run_name}\
    base_output_directory=gs://maxtext-experiments-tpem\
    dataset_path=gs://max-datasets-rogue\
    int8_training=true metrics_file=metrics.txt\
    remat_policy=${remat_policy} init_prng_key=${prng_key}\
    fwd_int8=${fwd_int8} bwd_int8=${bwd_int8}\
    global_parameter_scale=${global_parameter_scale}"

echo "Starting run (${run_name}) with command: ${command}"
eval ${command}
echo "Finished command"
echo "Now writing to ${output_file}"
if [[ ${SLICE_ID} -eq 0 && ${WORKER_ID} -eq 0 ]]; then
    gsutil cp metrics.txt ${output_file}
fi
echo "Done writing to ${output_file}"