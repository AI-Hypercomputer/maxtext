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

# Chained full-model layerwise replacement for the two 2_test_deepseek.sh full-model stages on a small slice:
#   logit:    chain_layerwise.py, the forward-logit stage (same tokens/config) -> KL vs golden and vs e2e logits.
#   pretrain: chain_pretrain.py, step 1 of the pretrain stage (same synthetic batch/config) -> losses and grad norm.
# Each unit restores only its own weights and consumes the previous unit's output.

set -ex
set -o pipefail

python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install "orbax-checkpoint==0.12.4" safetensors

UNSCANNED_CKPT_PATH="${UNSCANNED_CKPT_PATH:-gs://maxtext-deepseek/deepseek4-284b/2026-09-30/unscanned/0/items}"
TID2EID_PATH="${TID2EID_PATH:-gs://maxtext-deepseek/deepseek4-284b/2026-09-17/tid2eid.safetensors}"
CHAIN_STAGES="${CHAIN_STAGES:-logit pretrain}"
CHAIN_LOGIT_ARGS="${CHAIN_LOGIT_ARGS:-}"
CHAIN_PRETRAIN_ARGS="${CHAIN_PRETRAIN_ARGS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/chain_results}"
D=tests/end_to_end/tpu/deepseek/v4-284b
mkdir -p "${OUTPUT_DIR}"

for stage in ${CHAIN_STAGES}; do
  echo "=== CHAIN STAGE: ${stage} $(date -u +%T) ==="
  if [[ "${stage}" == "logit" ]]; then
    # shellcheck disable=SC2086
    python3 "${D}/chain_layerwise.py" --unscanned_ckpt="${UNSCANNED_CKPT_PATH}" --tid2eid_path="${TID2EID_PATH}" \
      --out_json="${OUTPUT_DIR}/logit.json" ${CHAIN_LOGIT_ARGS}
  else
    # shellcheck disable=SC2086
    python3 "${D}/chain_pretrain.py" --unscanned_ckpt="${UNSCANNED_CKPT_PATH}" --tid2eid_path="${TID2EID_PATH}" \
      --out_json="${OUTPUT_DIR}/pretrain.json" ${CHAIN_PRETRAIN_ARGS}
  fi
  echo "=== CHAIN STAGE DONE: ${stage} $(date -u +%T) ==="
done
