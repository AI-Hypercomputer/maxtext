#!/bin/bash
# Copyright 2026 Google LLC
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

set -euo pipefail

# ==============================================================================
# 256K Recipe (S = 262,144, PDB = 0.0625, GBS = 8, FSDP = 4, EP = 32)
# Inherits from qwen35_64c_128k.sh and overrides the 256K mesh/batch dimensions
# plus the 256K HBM remat policy:
#   - save_decoder_input_hbm = false, offload_decoder_input = true
#   - save_moe_gate_up_sublayers = [1, 3], offload_moe_gate_up_sublayers = [0, 2]
#   - save_key_value_proj = true, sa_block_kv_dkv_compute = 1024
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-qwen35-397b-256k-cf125}"

export SEQ_LEN="${SEQ_LEN:-262144}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export FSDP_SIZE="${FSDP_SIZE:-4}"
export EP_SIZE="${EP_SIZE:-32}"

export SAVE_DECODER_INPUT_HBM="${SAVE_DECODER_INPUT_HBM:-false}"
export OFFLOAD_DECODER_INPUT="${OFFLOAD_DECODER_INPUT:-true}"
export SAVE_MOE_GATE_UP_SUBLAYERS="${SAVE_MOE_GATE_UP_SUBLAYERS:-[1, 3]}"
export OFFLOAD_MOE_GATE_UP_SUBLAYERS="${OFFLOAD_MOE_GATE_UP_SUBLAYERS:-[0, 2]}"
export SAVE_KEY_VALUE_PROJ="${SAVE_KEY_VALUE_PROJ:-true}"
export SA_BLOCK_KV_DKV_COMPUTE="${SA_BLOCK_KV_DKV_COMPUTE:-1024}"

# shellcheck source=/dev/null
source "${DIR}/qwen35_64c_128k.sh" "$@"
