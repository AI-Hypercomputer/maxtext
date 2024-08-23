#!/bin/bash
set -euox pipefail

# BRANCH=${BRANCH:-main}
BRANCH=${BRANCH:-lizhiyu/flash_attention_check}

if [[ ! -d "maxtext" ]]; then
  git clone https://github.com/google/maxtext.git
fi

# switch branch
cd maxtext
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"

bash preflight.sh PLATFORM=gke
sleep 60

# flags set as default

# hlo dump
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_pass_re=spmd|sharding"

# debug
export TPU_STDERR_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0

# tunable parameter
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_enable_transpose_trace=true"
# export LIBTPU_INIT_ARGS='--xla_enable_transpose_trace=true --xla_enable_trace=true --xla_jf_debug_level=2'
# export LIBTPU_INIT_ARGS="--xla_enable_transpose_trace=true"



BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY:-"gs://mlperf-exp/${USER}"}
RUNNAME="${WORKLOAD_NAME}/${TIMESTAMP}"

# tunable parameters: ici_tensor_parallelism, per_device_batch_size, remat_policy, attention, int8_training
#  ici_tensor_parallelism is tunable and should be compatibility to topology
python3 MaxText/attentions_test.py MaxText/configs/base.yml run_name="${RUNNAME}"\
        per_device_batch_size=4 model_name=mixtral-8x7b\
        max_target_length=4096\
        base_output_directory="${BASE_OUTPUT_DIRECTORY}"\
        enable_checkpointing=false\
        skip_first_n_steps_for_profiler=90\
        steps=100 dtype=bfloat16 weight_dtype=bfloat16 max_target_length=4096\
        megablox=True profiler=xplane attention=flash dataset_type=synthetic\
        tokenizer_path=assets/tokenizer.mistral 2>&1 | tee /tmp/large_scale_multislice_test_log

EXP_FOLDER="${BASE_OUTPUT_DIRECTORY}/${RUNNAME}"
if [[ ${MEGASCALE_SLICE_ID} == "0" ]]; then
  if [[ ${TPU_WORKER_ID} == "0" ]]; then
    gsutil -m cp -r /tmp/xla_dump_file "${EXP_FOLDER}/xla"
    gsutil -m cp /tmp/large_scale_multislice_test_log "${EXP_FOLDER}/large_scale_multislice_test_log"
  fi
fi

