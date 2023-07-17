#!/bin/bash
set -e

STEPS=40000

USE_INT8_TRAINING=$1
USE_FWD_QUANT=$2

RUN_NAME=rwitten-20230715_2_2B-useint8_${USE_INT8_TRAINING}_usefwd_${USE_FWD_QUANT}


export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

command="python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${STEPS} per_device_batch_size=4 learning_rate=0.001 warmup_steps=2000 enable_profiler=false \
    enable_dropout=false enable_data_shuffling=false run_name=${RUN_NAME} base_emb_dim=4096\
    use_fwd_quant=${USE_FWD_QUANT} use_int8_training=${USE_INT8_TRAINING} metrics_file=metrics.txt"

echo "Starting run (${RUN_NAME}) with command: ${command}"
eval ${command}
echo "Finished command"
if [[ ${SLICE_ID} -eq 0 && ${WORKER_ID} -eq 0 ]]; then
    gsutil cp metrics.txt gs://mattdavidow-maxtext-br/${RUN_NAME}.txt
fi

