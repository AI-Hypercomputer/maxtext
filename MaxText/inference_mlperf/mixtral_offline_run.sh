#!/usr/bin/env bash

# Example:
# bash mixtral_offline_run.sh -r test_int8_kv_216-108-54  -n

# enable profiling using -p option and capture using
# tensorboard --logdir /tmp/tensorboard/

run_name="test_int8_kv_bs_216-108-54"
dry_run=false
skip_warmup=false
test_run=false
enable_profiler=false
performance=true
audit=false
accuracy=false
fast_eval=false

for arg in "$@"; do
  case $arg in
    -n) dry_run=true ;;
    -t) test_run=true ;;
    -s) skip_warmup=true ;;
    -p) enable_profiler=true ;;
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
    cmd=echo
else
    cmd=''
fi

SKIP_WARMUP_OPTION=""
if "$skip_warmup"; then
    SKIP_WARMUP_OPTION="--skip_warmup"
fi

PROFILER_OPTION=""
if "$enable_profiler"; then
    PROFILER_OPTION="--enable_profile"
fi

if [ -z "$TOKENIZER_PATH" ]; then
  TOKENIZER_PATH=/home/${USER}/maxtext/assets/tokenizer.mistral-v1
fi

BATCH_STR=""
if [ -z "$PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES" ];
then
  PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES="256,144|512,72|2048,18"
fi

if [ -z "$TOK_OUTLEN_MULTIPLIER" ];
then
  TOK_OUTLEN_MULTIPLIER="2.5"
fi

if [ -z "$MAXENGINE_ARGS" ];
then
  CHECKPOINT="gs://vipannalla-bkt/checkpoints/quantized/mixtral-8x7b-instruct/11-06-24"
  BASE_CFG="model_name=mixtral-8x7b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
  QUANT_CFG="quantization=int8 quantize_kvcache=True checkpoint_is_quantized=True"
  LAYOUT_CFG="compute_axis_order=0,2,1,3 ar_cache_axis_order=0,2,1,3"
  MOE_CFG="megablox=False sparse_matmul=False capacity_factor=1 model_call_mode=inference"
  MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG} ${LAYOUT_CFG} ${MOE_CFG}"
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
  export DATASET_PATH=${DATA_DISK_DIR}/mixtral-processed-data.pkl
  export TOTAL_SAMPLE_COUNT=100
  export USER_CONFIG=user${TOTAL_SAMPLE_COUNT}.conf
else
  export DATASET_TYPE=full
  export DATASET_PATH=${DATA_DISK_DIR}/mixtral-processed-data.pkl
  export TOTAL_SAMPLE_COUNT=15000
  export USER_CONFIG=user.conf
fi

# LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# makes subsequent runs faster
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_windowed_einsum_for_reduce_scatter=false --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000"

run_loadgen() {
  OUTPUT_LOG_ID=mixtral-8x7b-${run_name}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TYPE}_${LOADGEN_RUN_TIMESTAMP}
  OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
  mkdir -p ${OUTPUT_LOG_DIR} && cp ${USER_CONFIG} ${OUTPUT_LOG_DIR}
  OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json
  MIXTRAL_COLS_RENAME="{\"tok_input_len\": \"tok_input_length\", \"tok_ref_output_len\": \"tok_output_length\"}"

  echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
  echo "DATASET_PATH: ${DATASET_PATH}"
  echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
  echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
  echo "USER_CONFIG: ${USER_CONFIG}"
  echo "PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES: ${PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES}"
  echo "MAXENGINE_ARGS: ${MAXENGINE_ARGS}"

  ${cmd} python3 -m offline_mode \
    --mlperf_test_mode=${TEST_MODE} \
    --input_mode tokenized \
    --output_mode tokenized \
    --mlperf_conf $BASEDIR/mlperf.conf \
    --user_conf ${USER_CONFIG} \
    --audit_conf ${AUDIT_CONF}  \
    --total_sample_count ${TOTAL_SAMPLE_COUNT} \
    --dataset_path ${DATASET_PATH} \
    --prefill_lengths_and_batch_sizes ${PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES} \
    --maxengine_args "${MAXENGINE_ARGS}" \
    --output_log_dir ${OUTPUT_LOG_DIR} \
    --tok_outlen_multiplier ${TOK_OUTLEN_MULTIPLIER} \
    --rename_dataset_cols "${MIXTRAL_COLS_RENAME}" \
    ${SKIP_WARMUP_OPTION} ${PROFILER_OPTION} 2>&1 | tee ${OUTPUT_LOG_DIR}/${LOADGEN_RUN_TYPE}_log.log
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

# TODO: currently only checks Q&A and Math samples and fails on CodeGen. Will fix that in future PR.
run_loadgen_accuracy () {
  LOADGEN_RUN_TYPE=offline-accuracy
  TEST_MODE="accuracy"
  AUDIT_CONF="no_audit"
  run_loadgen

  # Eval Run
  if [ -e ${OUTPUT_ACCURACY_JSON_PATH} ]; then
    if [ "${FAST_EVAL:-false}" = "true" ] || "$fast_eval"; then
      EVAL_SCRIPT="evaluate-accuracy-fast.py"
    else
      EVAL_SCRIPT="evaluate-accuracy.py"
    fi

    ${CMD} python3 ${EVAL_SCRIPT} \
      --tokenizer-path ${TOKENIZER_PATH} \
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
