#!/bin/bash
set -e



fwd_int8=$1
bwd_int8=$2
global_parameter_scale=$3
PRNG_KEY=$4
RUN_NAME=$5

OUTPUT_FILE=gs://mattdavidow-maxtext-br/${RUN_NAME}.txt

BASE_STEPS=2
STEPS=$(($BASE_STEPS * $global_parameter_scale))

REMAT_POLICY=full

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"


command="python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${STEPS} per_device_batch_size=4 learning_rate=0.001 warmup_steps=2000 enable_profiler=false enable_checkpointing=false \
    enable_dropout=false enable_data_shuffling=false run_name=${RUN_NAME}\
    base_output_directory=gs://max-experiments\
    dataset_path=gs://maxtext-dataset\
    int8_training=true metrics_file=metrics.txt\
    remat_policy=${REMAT_POLICY} init_prng_key=${PRNG_KEY}\
    fwd_int8=${fwd_int8} bwd_int8=${bwd_int8}\
    global_parameter_scale=${global_parameter_scale}"

echo "Starting run (${RUN_NAME}) with command: ${command}"
eval ${command}
echo "Finished command"
echo "Now writing to ${OUTPUT_FILE}"
if [[ ${SLICE_ID} -eq 0 && ${WORKER_ID} -eq 0 ]]; then
    gsutil cp metrics.txt ${OUTPUT_FILE}
fi
echo "Done writing to ${OUTPUT_FILE}"