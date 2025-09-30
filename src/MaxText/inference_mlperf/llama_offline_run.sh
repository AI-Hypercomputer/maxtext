#!/usr/bin/env bash

# Example:
# bash llama_offline_run.sh -r test_int8_kv_216-108-54  -n

# enable profiling using -p option and capture using
# tensorboard --logdir /tmp/tensorboard/

run_name="llama_offline_benchmarks"
dry_run=false
skip_warmup=false
test_run=false
enable_profiler=false
performance=true
audit=false
accuracy=false
fast_eval=false
enable_batch_prefill=false

for arg in "$@"; do
  case $arg in
    -n) dry_run=true ;;
    -t) test_run=true ;;
    -s) skip_warmup=true ;;
    -p) enable_profiler=true ;;
    -c) enable_batch_prefill=true ;;
    -d) audit=true ;;
    -a) accuracy=true ;;
    -f) fast_eval=true ;;
    -r=*|--run=*) run_name="${arg#*=}" ;;
    -r|--run)
      shift
      run_name="$1"
      ;;
  esac
  shift
done


if "$dry_run"; then
    CMD=echo
else
    CMD=''
fi

SKIP_WARMUP_OPTION=""
if "$skip_warmup"; then
    SKIP_WARMUP_OPTION="--skip_warmup"
fi

PROFILER_OPTION=""
if "$enable_profiler"; then
    PROFILER_OPTION="--enable_profile"
fi

BATCH_PREFILL_OPTION=""
if "$enable_batch_prefill"; then
    BATCH_PREFILL_OPTION="--enable_batch_prefill"
fi

if [ -z "$TOKENIZER_PATH" ]; then
  TOKENIZER_PATH="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.llama2
fi

BATCH_STR=""
if [ -z "$PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES" ];
then
  PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES="256,216|512,108|1024,54"
fi

if [ -z "$TOK_OUTLEN_MULTIPLIER" ];
then
  TOK_OUTLEN_MULTIPLIER="2.5"
fi

if [ -z "$MODEL_NAME" ];
then
  MODEL_NAME="llama2-70b"
fi

if [ -z "$HF_CKPT" ];
then
  HF_CKPT="meta-llama/Llama-2-70b-chat-hf"
fi

if [[ -z ${MAXENGINE_CONFIG_FILEPATH} ]] ; then
    MAXENGINE_CONFIG_FILEPATH="${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/configs/base.yml"
fi


if [ -z "$MAXENGINE_ARGS" ];
then
  CHECKPOINT="gs://msingh-bkt/checkpoints/quant_${MODEL_NAME}-chat/mlperf_070924/int8_"
  BASE_CFG="model_name=${MODEL_NAME} tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
  QUANT_CFG="quantization=int8 quantize_kvcache=True checkpoint_is_quantized=True skip_jax_distributed_system=true"
  MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG}"
fi

if [ -z "$BASEDIR" ];
then
  BASEDIR=/home/${USER}/inference
fi

if [ -z "$DATA_DISK_DIR" ];
then
  DATA_DISK_DIR=/home/${USER}/loadgen_run_data
fi

export LOADGEN_RUN_TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S%Z)
export API_URL=0.0.0.0:9000
if "$test_run"; then
  export DATASET_TYPE=test
  export DATASET_PATH=${DATA_DISK_DIR}/processed-data.pkl
  export TOTAL_SAMPLE_COUNT=1000
  export USER_CONFIG=user${TOTAL_SAMPLE_COUNT}.conf
else
  export DATASET_TYPE=full
  export DATASET_PATH=${DATA_DISK_DIR}/processed-data.pkl
  export TOTAL_SAMPLE_COUNT=24576
  export USER_CONFIG=user.conf # NOTE: you may need to change this path(e.g. `src/MaxText/inference_mlperf/user.conf`)
fi

# LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# makes subsequent runs faster
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS

# Ensure working directory is at repository root.
cd $(dirname $0)/../../

run_loadgen() {
  OUTPUT_LOG_ID=llama70b-${run_name}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TYPE}_${LOADGEN_RUN_TIMESTAMP}
  OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
  mkdir -p ${OUTPUT_LOG_DIR} && cp ${USER_CONFIG} ${OUTPUT_LOG_DIR}
  OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json

  echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
  echo "DATASET_PATH: ${DATASET_PATH}"
  echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
  echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
  echo "USER_CONFIG: ${USER_CONFIG}"
  echo "PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES: ${PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES}"
  echo "MAXENGINE_ARGS: ${MAXENGINE_ARGS}"
  echo
  ${CMD} python3 -m MaxText.inference_mlperf.offline_mode \
    --maxengine_config_filepath=${MAXENGINE_CONFIG_FILEPATH} \
    --mlperf_test_mode=${TEST_MODE} \
    --input_mode tokenized \
    --output_mode tokenized \
    --mlperf_conf $BASEDIR/mlperf.conf \
    --user_conf ${USER_CONFIG} \
    --audit_conf ${AUDIT_CONF}  \
    --total_sample_count ${TOTAL_SAMPLE_COUNT} \
    --dataset_path ${DATASET_PATH} \
    --prefill_lengths_and_per_device_batch_sizes ${PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES} \
    --maxengine_args "${MAXENGINE_ARGS}" \
    --output_log_dir ${OUTPUT_LOG_DIR} \
    --tok_outlen_multiplier ${TOK_OUTLEN_MULTIPLIER} \
    ${SKIP_WARMUP_OPTION} ${PROFILER_OPTION} ${BATCH_PREFILL_OPTION} 2>&1 | tee ${OUTPUT_LOG_DIR}/${LOADGEN_RUN_TYPE}_log.log
}

run_loadgen_performance () {
  LOADGEN_RUN_TYPE=offline-performance
  TEST_MODE="performance"
  AUDIT_CONF="no_audit"
  run_loadgen
}

run_loadgen_audit () {
  LOADGEN_RUN_TYPE=offline-audit
  TEST_MODE="performance"
  AUDIT_CONF="$BASEDIR/compliance/nvidia/TEST06/audit.config"
  run_loadgen
}

run_loadgen_accuracy () {
  LOADGEN_RUN_TYPE=offline-accuracy
  TEST_MODE="accuracy"
  AUDIT_CONF="no_audit"
  run_loadgen

  if [ $dry_run ] ; then
    touch ${OUTPUT_ACCURACY_JSON_PATH}
  fi

  # Eval Run
  if [ -e ${OUTPUT_ACCURACY_JSON_PATH} ]; then
    if [ "${FAST_EVAL:-false}" = "true" ] || "$fast_eval"; then
      EVAL_SCRIPT="evaluate-accuracy-fast"
    else
      EVAL_SCRIPT="evaluate-accuracy"
    fi
    echo
    ${CMD} python3 -m MaxText.inference_mlperf.${EVAL_SCRIPT} \
      --checkpoint-path ${HF_CKPT} \
      --mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
      --dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_offline_accuracy_log.log
  fi
}

performance=true
if "$audit"; then
  performance=false
  echo
  echo "Starting loadgen audit"
  run_loadgen_audit
fi

if "$accuracy"; then
  performance=false
  echo
  echo "Starting loadgen accuracy"
  run_loadgen_accuracy
fi

if "$performance"; then
  echo
  echo "Starting loadgen performance run"
  run_loadgen_performance
fi
