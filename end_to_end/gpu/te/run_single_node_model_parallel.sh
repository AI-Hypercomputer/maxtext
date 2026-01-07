# Copyright 2023–2026 Google LLC
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

# Run MaxText with Transformer Engine (TE) across different parallelization strategies and quantization recipes
# Usage: bash run_single_node_model_parallel.sh --model MODEL --output-dir-tag OUTPUT_DIR_TAG --trace true|false --steps STEPS --single-gpu-run true|false --num-decoder-layers N_LAYERS

#!/bin/bash
set -euo pipefail

# Default values
MODEL="llama3.1-8b"
OUTPUT_DIR_TAG=""
STEPS=50
TRACE=false
SINGLE_GPU_RUNS=true
NUM_DECODER_LAYERS="" # unset

# Parse keyword-style arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir-tag)
            OUTPUT_DIR_TAG="$2"
            shift 2
            ;;
        --trace)
            TRACE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --single-gpu-run)
            SINGLE_GPU_RUNS="$2"
            shift 2
            ;;
        --num-decoder-layers)
            NUM_DECODER_LAYERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model MODEL] [--output-dir-tag OUTPUT_DIR_TAG] [--trace true|false] [--steps STEPS] [--single-gpu-run true|false] [--num-decoder-layers N_LAYERS]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--model MODEL] [--output-dir-tag OUTPUT_DIR_TAG] [--trace true|false] [--steps STEPS] [--single-gpu-run true|false] [--num-decoder-layers N_LAYERS]"
            exit 1
            ;;
    esac
done

if [[ "$TRACE" == "true" ]]; then
  OUTPUT_DIR_TAG="trace${OUTPUT_DIR_TAG:+_$OUTPUT_DIR_TAG}"
fi

# Now your variables are set as needed
echo "MODEL=$MODEL"
echo "OUTPUT_DIR_TAG=$OUTPUT_DIR_TAG"
echo "TRACE=$TRACE"
echo "STEPS=$STEPS"
echo "SINGLE_GPU_RUNS=$SINGLE_GPU_RUNS"

WARMUP_STEPS=10
if (( STEPS <= WARMUP_STEPS )); then
    echo "ERROR: STEPS ($STEPS) must be greater than WARMUP_STEPS ($WARMUP_STEPS)"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAXTEXT_DIR="$(realpath "$SCRIPT_DIR/../../../")"
OUTPUT_DIR="${SCRIPT_DIR}/output/${MODEL}${NUM_DECODER_LAYERS:+_${NUM_DECODER_LAYERS}_layers}${OUTPUT_DIR_TAG:+_$OUTPUT_DIR_TAG}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

n_gpus=$(nvidia-smi -L | wc -l)
half_gpus=$((n_gpus / 2))
# List of experiments: <DP> <TPSP> <FSDP>
experiments=(
  "1        1           1"        # Single GPU
  "$n_gpus  1           1"        # Full DP
  "1        $n_gpus     1"        # Full TPSP
  "2        $half_gpus  1"        # DP=2, TPSP=half GPUs
  "1        1           $n_gpus"  # Full FSDP
  "1        $half_gpus  2"        # FSDP=2, TPSP=half GPUs
)

CSV="$OUTPUT_DIR/raw_results.csv"
echo -e "test\tdp\ttpsp\tfsdp\tmean\tstddev" > "$CSV"

run_and_parse() {
  local test="$1"
  local dp="$2"
  local tpsp="$3"
  local fsdp="$4"
  set +e
  local cmd="$5"
  set -e
  local stdout="$OUTPUT_DIR/run_${test}_dp${dp}_tpsp${tpsp}_fsdp${fsdp}.log"
  echo "===== Executing ${test}\t${dp}\t${tpsp}\t${fsdp} ====="
  eval "$cmd" 2>&1 | tee "$stdout"
  # Exclude the warning steps for warning up and last step for tracing
  std=$(grep 'Tokens/s/device:' "$stdout" | sed '1,'"${WARMUP_STEPS}"'d;$d' | awk -F'Tokens/s/device: ' '{print $2}' | awk -F',' '{print $1}')

  if [ -z "$std" ]; then
    mean="NA"
    stddev="NA"
  else
    mean_stddev=$(echo "$std" | python3 -c "import sys; import numpy as np
arr = [float(l.strip()) for l in sys.stdin if l.strip()]
if arr:
  print(f'{np.mean(arr):.2f}\t{np.std(arr, ddof=1):.2f}')
else:
  print('NA\tNA')
"
    )
    mean=$(echo "$mean_stddev" | cut -f1)
    stddev=$(echo "$mean_stddev" | cut -f2)
  fi
  echo -e "${test}\t${dp}\t${tpsp}\t${fsdp}\t${mean}\t${stddev}" >> "$CSV"

  if [[ "$TRACE" == "true" ]]; then
    TRACE_SRC=$(grep -oE '/tmp/tmp\.[^ ]+' "$stdout" | head -n1)
    if [[ -n "$TRACE_SRC" && -e "$TRACE_SRC" ]]; then
      TRACE_DEST="${OUTPUT_DIR}/trace_${test}_dp${dp}_tpsp${tpsp}_fsdp${fsdp}"
      mv "$TRACE_SRC" "$TRACE_DEST"
      echo " === Trace moved: $TRACE_SRC -> $TRACE_DEST"
    else
      echo "=== No trace file found for $test, dp=$dp, tpsp=$tpsp, fsdp=$fsdp"
    fi
  fi
}

PROFILE_SKIP_STEPS=$(($STEPS-1))
PROFILE_ARG=""
original_num_decoder_layers=1
if [[ "$TRACE" == "true" ]]; then
  PROFILE_ARG="profiler=xplane skip_first_n_steps_for_profiler=${PROFILE_SKIP_STEPS} profiler_steps=1"
fi
# Updating the model config file as we can't pass base_num_decoder_layers=1 in additional-args
if [ -n "$NUM_DECODER_LAYERS" ]; then
  MODEL_CONFIG="$MAXTEXT_DIR/MaxText/configs/models/$MODEL.yml"
  original_num_decoder_layers=$(grep "base_num_decoder_layers" "$MODEL_CONFIG" | awk -F': ' '{print $2}')
  sed -i "s/base_num_decoder_layers: .*/base_num_decoder_layers: $NUM_DECODER_LAYERS/" "$MODEL_CONFIG"
  echo "=== Setting base_num_decoder_layers=$NUM_DECODER_LAYERS in $MODEL_CONFIG"
fi

# Updating the model config file back if modified
restore_model_config_file() {
  if [ -n "$NUM_DECODER_LAYERS" ]; then
    sed -i "s/base_num_decoder_layers: .*/base_num_decoder_layers: ${original_num_decoder_layers}/" "$MODEL_CONFIG"
    echo "=== Restoring base_num_decoder_layers back to ${original_num_decoder_layers} in $MODEL_CONFIG"
  fi
}
trap restore_model_config_file EXIT

BASE_ARGS="--model $MODEL --steps $STEPS"
# Need to be with four escape quotes
OTHER_ARGS="--additional-args=\"\"\"${PROFILE_ARG}\"\"\""
TRAINING_RECIPES=("fp8" "te_fp8_delayedscaling" "te_fp8_currentscaling" "te_mxfp8" "te_nvfp4")   # fp8 is the MaxText baseline

export NVTE_JAX_CUSTOM_CALLS='NormFwdPrimitive=false,NormBwdPrimitive=false'

if [[ "$SINGLE_GPU_RUNS" == "false" ]]; then
    start_index=1
else
    start_index=0
fi
for ((i = start_index; i < ${#experiments[@]}; i++)); do
  exp="${experiments[$i]}"
  echo "Running experiment: $exp"
  read dp tpsp fsdp <<< "$exp"

  n_used_gpus=$((dp * tpsp * fsdp))
  if (( n_used_gpus > n_gpus )); then
    echo "Error: requested $n_used_gpus GPUs, but only $n_gpus are available."
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n_used_gpus - 1)))
  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  echo "=== Using GPUs: $CUDA_VISIBLE_DEVICES"

  args="--data-parallel=$dp --tensor-sequence-parallel=$tpsp --fsdp=$fsdp"

  for recipe in "${TRAINING_RECIPES[@]}"; do
    test="${recipe}"
    run_and_parse "$test" "$dp" "$tpsp" "$fsdp" \
      "MAXTEXT_DIR=${MAXTEXT_DIR} bash test-maxtext.sh $args --quantization=${recipe} $BASE_ARGS ${OTHER_ARGS}"
  done
done


OUTPUT_FORMAT="txt" # txt or csv
echo "=== Experiments finished. Raw CSV at $CSV"
python3 $SCRIPT_DIR/normalize.py "$CSV" "${OUTPUT_DIR}/summary.$OUTPUT_FORMAT" "$OUTPUT_FORMAT"
cat "${OUTPUT_DIR}/summary.$OUTPUT_FORMAT"
