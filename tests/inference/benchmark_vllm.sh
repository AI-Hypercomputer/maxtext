#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generic benchmark script for MaxText models using vLLM on TPU.
#
# Usage:
#   bash tests/inference/benchmark_vllm.sh <MODEL_NAME> [ATTENTION] [TP_SIZE]
#
# Arguments:
#   $1 (MODEL_NAME) : MaxText model name (e.g., "qwen3-0.6b", "llama3.1-8b") [Required]
#   $2 (ATTENTION)  : Attention kernel to benchmark (default: "vllm_rpa")
#   $3 (TP_SIZE)    : Tensor parallelism size (default: 4)
#
# Environment Variable Overrides:
#   TOKENIZER      : HuggingFace tokenizer/model path (default: "Qwen/Qwen3-0.6B")
#
# Examples:
#   bash tests/inference/benchmark_vllm.sh qwen3-0.6b
#   bash tests/inference/benchmark_vllm.sh qwen3-0.6b vllm_batched_rpa
#   TOKENIZER=meta-llama/Llama-3.1-8B bash tests/inference/benchmark_vllm.sh llama3.1-8b vllm_batched_rpa 4

set -e

if [ -z "$1" ]; then
  echo "Error: MODEL_NAME is required as the first argument."
  echo "Usage: bash tests/inference/benchmark_vllm.sh <MODEL_NAME> [ATTENTION] [TP_SIZE]"
  echo "Example: bash tests/inference/benchmark_vllm.sh qwen3-0.6b"
  exit 1
fi

MODEL_NAME=$1
ATTENTION=${2:-${ATTENTION:-"vllm_rpa"}}
TP_SIZE=${3:-${TP_SIZE:-4}}
TOKENIZER=${TOKENIZER:-"Qwen/Qwen3-0.6B"}

INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-32}
LOAD_FORMAT=${LOAD_FORMAT:-"dummy"}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
PYTHON_EXEC=${PYTHON_EXEC:-"python3"}

echo "================================================================="
echo "Running vLLM Benchmark for MaxText"
echo "Model Name       : $MODEL_NAME"
echo "Tokenizer Path   : $TOKENIZER"
echo "Attention Kernel : $ATTENTION"
echo "Tensor Parallel  : $TP_SIZE"
echo "Input / Output   : $INPUT_LEN in / $OUTPUT_LEN out ($NUM_PROMPTS prompts)"
echo "Load Format      : $LOAD_FORMAT"
echo "================================================================="

# Set standard TPU & vLLM environment variables
export PYTHONPATH=$(pwd)/src:${PYTHONPATH}
export SKIP_JAX_PRECOMPILE=1
export NEW_MODEL_DESIGN=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

if [ "$ATTENTION" = "vllm_batched_rpa" ]; then
  export USE_BATCHED_RPA_KERNEL=1
else
  export USE_BATCHED_RPA_KERNEL=0
fi

# Run offline throughput benchmark
$PYTHON_EXEC -m vllm.entrypoints.cli.main bench throughput \
  --model "$TOKENIZER" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --load-format "$LOAD_FORMAT" \
  --hf-overrides '{"architectures": ["MaxTextForCausalLM"]}' \
  --additional-config "{\"maxtext_config\": {\"model_name\": \"$MODEL_NAME\", \"weight_dtype\": \"bfloat16\", \"attention\": \"$ATTENTION\", \"allow_split_physical_axes\": true, \"scan_layers\": true, \"enable_nnx\": true, \"pure_nnx_decoder\": true}}" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS"

echo "================================================================="
echo "Benchmark completed successfully for $MODEL_NAME ($ATTENTION)!"
echo "================================================================="
