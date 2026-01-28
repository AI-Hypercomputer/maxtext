#!/bin/bash

# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

run_name="microbenchmark_llama2-70b_h100-8_$(date +%Y-%m-%d-%H-%M)"
enable_profiler=false
stages=prefill,generate

helpFunction()
{
   echo ""
   echo "Usage: $0 [-p] [-s stages] [-r run_name]"
   echo -e "\t-p Enable profiler"
   echo -e "\t-r Specify run name"
   echo -e "\t-s Specify comma-separated benchmark stages [prefill|prefill-multisampling|generate] (default: prefill,generate)"
   exit 1
}

for arg in "$@"; do
    case $arg in
        -p) enable_profiler=true ;;
        -r=*|--run=*) run_name="${arg#*=}" ;;
        -r|--run) shift; run_name="$1" ;;
        -s=*|--stages=*) stages="${arg#*=}" ;;
        -s|--stages) shift; stages="$1" ;;
        -h|--help) helpFunction ;;
    esac
    shift
done

# Validate benchmark stages
IFS=',' read -ra stage <<< "$stages"
for i in "${stage[@]}"; do
    case "$i" in
        prefill|prefill-multisampling|generate) ;;
        *) echo "Invalid benchmark stage '$i'. Must be: prefill, prefill-multisampling, or generate."; exit 1 ;;
    esac
done

# Default parameters
if [[ -z ${BASE_OUTPUT_DIRECTORY} ]] ; then
    export BASE_OUTPUT_DIRECTORY="/tmp/maxtext"
fi
if [[ -z ${INFERENCE_LOG_FILE_PATH} ]] ; then
    export INFERENCE_LOG_FILE_PATH="${BASE_OUTPUT_DIRECTORY}/microbenchmark_llama2-70b_h100-8_results.txt"
fi
if [[ -z ${MAXENGINE_CONFIG_FILEPATH} ]] ; then
    MAXENGINE_CONFIG_FILEPATH="$(dirname $0)/../../configs/inference.yml"
fi
if [[ -z ${QUANTIZATION} ]] ; then
    QUANTIZATION="aqt_fp8"
fi
if [[ -z ${KV_QUANT_DTYPE} ]] ; then
    KV_QUANT_DTYPE="fp8"
    QUANTIZE_KVCACHE=True
fi
if [[ -z ${PREFILL_LENGTHS} ]] ; then
    PREFILL_LENGTHS=1024
fi
if [[ -z ${PER_DEVICE_BATCH_SIZE} ]] ; then
    PER_DEVICE_BATCH_SIZE=190
fi
if [[ -z ${ATTENTION} ]] ; then
    ATTENTION=dot_product
fi
if [[ -z ${AUTOREGRESSIVE_PARALLELISM} ]] ; then
    AUTOREGRESSIVE_PARALLELISM=1
fi
if [[ -z ${TENSOR_PARALLELISM} ]] ; then
    TENSOR_PARALLELISM=-1
fi
PROFILER_STR=""
if [[ "$enable_profiler" = true ]] ; then
    PROFILER_STR=" profiler=xplane"
fi
if [[ -z ${GCS_METRICS} ]] ; then
    GCS_METRICS=False
fi

# Get max prefill length
IFS=',' read -r -a prefill_lengths_arr <<< "$PREFILL_LENGTHS"
max_prefill_predict_length=${prefill_lengths_arr[0]}
for n in "${prefill_lengths_arr[@]}" ; do
    ((n > max_prefill_predict_length)) && max_prefill_predict_length=$n
done

# Execute from repository root
cd $(dirname $0)/../../../

XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_command_buffer=FUSION --xla_disable_hlo_passes=rematerialization" \
TF_FORCE_GPU_ALLOW_GROWTH=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.94 \
python3 -m maxtext.inference.inference_microbenchmark $MAXENGINE_CONFIG_FILEPATH  \
    base_output_directory=$BASE_OUTPUT_DIRECTORY  \
    tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.llama2 \
    model_name='llama2-70b' \
    max_prefill_predict_length=$max_prefill_predict_length  \
    max_target_length=2048  \
    attention=$ATTENTION \
    scan_layers=false \
    hardware=gpu \
    async_checkpointing=false \
    per_device_batch_size=$PER_DEVICE_BATCH_SIZE \
    inference_microbenchmark_prefill_lengths=$PREFILL_LENGTHS  \
    inference_microbenchmark_stages=$stages \
    inference_microbenchmark_loop_iters=64 \
    inference_microbenchmark_log_file_path=$INFERENCE_LOG_FILE_PATH \
    run_name=$run_name \
    ici_fsdp_parallelism=1 \
    ici_autoregressive_parallelism=$AUTOREGRESSIVE_PARALLELISM \
    ici_tensor_parallelism=$TENSOR_PARALLELISM \
    weight_dtype=bfloat16 \
    kv_quant_dtype=$KV_QUANT_DTYPE \
    quantize_kvcache=$QUANTIZE_KVCACHE \
    quantization=$QUANTIZATION$PROFILER_STR \
    gcs_metrics=$GCS_METRICS
