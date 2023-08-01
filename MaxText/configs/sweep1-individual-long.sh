#!/bin/bash
set -e



int8_training=$1
global_parameter_scale=$2
PRNG_KEY=$3
RUN_NAME=$4


BASE_STEPS=60000
STEPS=$(($BASE_STEPS * $global_parameter_scale))

OUTPUT_FILE=gs://mattdavidow-maxtext-br/${RUN_NAME}.txt

REMAT_POLICY=full

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"


command="python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${STEPS} per_device_batch_size=4 learning_rate=0.001 warmup_steps=2000 enable_profiler=false enable_checkpointing=false \
    enable_dropout=false enable_data_shuffling=false run_name=${RUN_NAME}\
    base_output_directory=gs://maxtext-experiments-multipod\
    dataset_path=gs://max-datasets-rogue\
    int8_training=${int8_training} metrics_file=metrics.txt\
    remat_policy=${REMAT_POLICY} init_prng_key=${PRNG_KEY}\
    global_parameter_scale=${global_parameter_scale}"

echo "Starting run (${RUN_NAME}) with command: ${command}"
eval ${command}
echo "Finished command"
echo "Now writing to ${OUTPUT_FILE}"
if [[ ${SLICE_ID} -eq 0 && ${WORKER_ID} -eq 0 ]]; then
    gsutil cp metrics.txt ${OUTPUT_FILE}
fi
echo "Done writing to ${OUTPUT_FILE}"