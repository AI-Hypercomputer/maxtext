#!/usr/bin/env bash
me=$(basename "$0")

if [ -z "$BASEDIR"];
then
  BASEDIR=/home/vipannalla/inference_mlperf4.1
fi

USER_CONFIG=$BASEDIR/language/llama2-70b/tpu/user.conf

if [ -z "$DATA_DISK_DIR"];
then
  DATA_DISK_DIR=/home/vipannalla/loadgen_run_data
fi

DATASET_PATH=${DATA_DISK_DIR}/processed-data.pkl
TOTAL_SAMPLE_COUNT=1000
LOG_INTERVAL=200

# HF model id
TOKENIZER_PATH="meta-llama/Llama-2-70b-chat-hf"
LOADGEN_RUN_TYPE=offline-accuracy
MODEL_NAME=llama70b
DATASET_TYPE=full

LOADGEN_RUN_TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S%Z)
OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}
OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}

mkdir -p ${OUTPUT_LOG_DIR} && cp ${USER_CONFIG} ${OUTPUT_LOG_DIR}

OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json

# LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# makes subsequent runs faster
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS

echo "LOADGEN_RUN_TYPE: ${LOADGEN_RUN_TYPE}"
echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
echo "BATCH_SIZE_EXP: ${BATCH_SIZE_EXP}"
echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
echo "USER_CONFIG: ${USER_CONFIG}"

python -m offline_mode \
        --mlperf_test_mode=accuracy \
	--input_mode tokenized \
        --output_mode tokenized \
	--mlperf_conf $BASEDIR/mlperf.conf \
	--user_conf ${USER_CONFIG} \
	--audit_conf no_audit \
	--total_sample_count ${TOTAL_SAMPLE_COUNT} \
	--dataset_path ${DATASET_PATH} \
	--output_log_dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/offline_accuracy_log.log

# Eval Run
if [ -e ${OUTPUT_ACCURACY_JSON_PATH} ]; then
        python3 evaluate-accuracy.py \
                --checkpoint-path meta-llama/Llama-2-70b-chat-hf \
                --mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
                --dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_offline_accuracy_log.log
fi

