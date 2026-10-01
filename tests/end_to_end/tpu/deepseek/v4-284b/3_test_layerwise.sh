#!/bin/bash

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

# End-to-end real-weight layer-subgroup forward and backward (VJP + dW + routing)
# verification for DeepSeek-V4-Flash (deepseek4-284b) on TPU (or CPU).
#
# Subgroups verified against official PyTorch reference bundles at seq_len=4096:
#   S1: layers 0..2 (unrolled prefix: SWA+hash layers 0..1, CSA+indexer+hash layer 2)
#   S2: layers 3..6 (2 scanned HCA+CSA blocks with top-k routed MoE)
#   S3: layers 41..42 + hc_head + decoder_norm + logits_dense + masked cross-entropy
#   B<b>: layers 3+2b..4+2b (scanned block b); S1 + B0..B18 + S3 covers all 43 layers + head.
# Each unit starts from the reference boundary state (teacher forcing), so one bad layer cannot hide or smear.
# LAYERWISE_OUTPUTS_DIR saves each unit's h_out / dh_in for a direct CPU-vs-TPU comparison.

set -ex
set -o pipefail

python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install transformers==4.57.3
python3 -m pip install "pathwaysutils==0.1.11"
python3 -m pip install "qwix==0.1.8"
python3 -m pip install "orbax-checkpoint==0.12.4"
python3 -m pip install safetensors

UNSCANNED_CKPT_PATH="${UNSCANNED_CKPT_PATH:-gs://maxtext-deepseek/deepseek4-284b/2026-09-17/unscanned/0/items}"
TID2EID_PATH="${TID2EID_PATH:-gs://maxtext-deepseek/deepseek4-284b/2026-09-17/tid2eid.safetensors}"
LAYERWISE_BUNDLE_DIR="${LAYERWISE_BUNDLE_DIR:-gs://maxtext-test-assets/deepseek4-284b/layerwise/2026-09-25/fp32_train}"
LAYERWISE_DTYPES="${LAYERWISE_DTYPES:-bfloat16 float32}"
LAYERWISE_SUBGROUPS="${LAYERWISE_SUBGROUPS:-S1 $(seq -s ' ' -f 'B%g' 0 18) S3}"
LAYERWISE_NUM_COTANGENTS="${LAYERWISE_NUM_COTANGENTS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/layerwise_results}"
LAYERWISE_OUTPUTS_DIR="${LAYERWISE_OUTPUTS_DIR:-}"

mkdir -p "${OUTPUT_DIR}"

failed=()
for dtype in ${LAYERWISE_DTYPES}; do
  for sg in ${LAYERWISE_SUBGROUPS}; do
    echo "=== LAYERWISE VERIFICATION: subgroup=${sg} dtype=${dtype} ==="
    save_args=()
    if [[ -n "${LAYERWISE_OUTPUTS_DIR}" ]]; then
      save_args=(--save_outputs="${LAYERWISE_OUTPUTS_DIR}/${dtype}/${sg}.safetensors")
    fi
    # fp32 GMM tiles exceed the default 16 MiB scoped VMEM on v5p.
    libtpu_args="${LIBTPU_INIT_ARGS:-}"
    if [[ "${dtype}" == "float32" ]]; then
      libtpu_args="${libtpu_args} --xla_tpu_scoped_vmem_limit_kib=65536"
    fi
    # Keep sweeping after a failing unit so one run reports every layer.
    LIBTPU_INIT_ARGS="${libtpu_args}" python3 tests/end_to_end/tpu/deepseek/v4-284b/verify_layerwise.py \
      --subgroup="${sg}" \
      --bundle_dir="${LAYERWISE_BUNDLE_DIR}" \
      --dtype="${dtype}" \
      --num_cotangents="${LAYERWISE_NUM_COTANGENTS}" \
      --unscanned_ckpt="${UNSCANNED_CKPT_PATH}" \
      --tid2eid_path="${TID2EID_PATH}" \
      --out_json="${OUTPUT_DIR}/${dtype}_${sg}.json" \
      "${save_args[@]}" \
      --assert_pass || failed+=("${dtype}/${sg}")
    echo "=== LAYERWISE UNIT DONE: subgroup=${sg} dtype=${dtype} ==="
  done
done

if (( ${#failed[@]} )); then
  echo "LAYERWISE FAILED UNITS: ${failed[*]}"
  exit 1
fi
echo "DeepSeek V4 layer-subgroup forward & backward verification completed successfully!"
