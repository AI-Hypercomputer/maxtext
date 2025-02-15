#!/usr/bin/env bash

# Example:
# bash llama_offline_run.sh -r test_int8_kv_216-108-54  -n

# enable profiling using -p option and capture using
# tensorboard --logdir /tmp/tensorboard/

dry_run=false
performance=false
accuracy=false
enable_batch_prefill=true
#enable_batch_prefill=false
audit=true
enable_profiler=false
test_run=false

BATCH_SIZE=64
KV="int4"
run_name="trillium_${BATCH_SIZE}_${KV}_pack_${enable_batch_prefill}"

if "$dry_run"; then
    export CMD=echo
else
    export CMD=''
fi

BATCH_PREFILL_OPTION=""
if "$enable_batch_prefill"; then
  BATCH_PREFILL_OPTION="--enable_batch_prefill"
fi
export PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES="1024,${BATCH_SIZE}"
export TOKENIZER_PATH=/home/${USER}/maxtext/assets/tokenizer.llama2
export TOK_OUTLEN_MULTIPLIER="2.5"
export MODEL_NAME="llama2-70b"
export CHECKPOINT="gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_"
export BASE_CFG="model_name=${MODEL_NAME} tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
export QUANT_CFG="quantization=int8 checkpoint_is_quantized=True"
export KV_QUANT_CFG="quantize_kvcache=True kv_quant_dtype=${KV}"
MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG} ${KV_QUANT_CFG} optimize_mesh_for_tpu_v6e=True"

TEST_FLAGS=$(python3 trillium/select_xla_flags.py)
export LIBTPU_INIT_ARGS=${TEST_FLAGS}

SKIP_WARMUP_OPTION=""
if "$skip_warmup"; then
    SKIP_WARMUP_OPTION="--skip_warmup"
fi

PROFILER_OPTION=""
if "$enable_profiler"; then
    PROFILER_OPTION="--enable_profile"
fi


export BASEDIR=/home/${USER}/inference
export DATA_DISK_DIR=/home/${USER}/loadgen_run_data
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
  export USER_CONFIG=user.conf
fi

export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS

run_loadgen() {
  OUTPUT_LOG_ID=${run_name}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TYPE}_${LOADGEN_RUN_TIMESTAMP}
  OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
  mkdir -p ${OUTPUT_LOG_DIR} && cp ${USER_CONFIG} ${OUTPUT_LOG_DIR}
  OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json

  echo XLA_FLAGS:
  echo $LIBTPU_INIT_ARGS
  echo
  echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
  echo "DATASET_PATH: ${DATASET_PATH}"
  echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
  echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
  echo "USER_CONFIG: ${USER_CONFIG}"
  echo "PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES: ${PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES}"
  echo
  echo "MAXENGINE_ARGS: ${MAXENGINE_ARGS}"
  echo
  
  ${CMD} python -m offline_mode \
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
  echo
  echo eval_accuracy
  echo
  EVAL_SCRIPT="evaluate-accuracy.py"
  
  ${CMD} python3 ${EVAL_SCRIPT} \
      --checkpoint-path "meta-llama/Llama-2-70b-chat-hf" \
      --mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
      --dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_offline_accuracy_log.log
}

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
