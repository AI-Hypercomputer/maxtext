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
# 256K PDB = 1/32 Recipe (S = 262,144, PDB = 0.03125, GBS = 4, FSDP = 4, EP = 32)
# Inherits from qwen35_64c_256k.sh and runs with schedule_mode="unsplit"
# (1 local sequence of 8,192 tokens/TC) and all 4 MoE gate_up sublayers saved in
# HBM with zero PCIe host offload.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-qwen35-397b-256k-cf125-pdb003125}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export SCHEDULE_MODE="${SCHEDULE_MODE:-unsplit}"
export SAVE_DECODER_INPUT_HBM="${SAVE_DECODER_INPUT_HBM:-true}"
export OFFLOAD_DECODER_INPUT="${OFFLOAD_DECODER_INPUT:-false}"
export SAVE_MOE_GATE_UP_SUBLAYERS="${SAVE_MOE_GATE_UP_SUBLAYERS:-[0, 1, 2, 3]}"
export OFFLOAD_MOE_GATE_UP_SUBLAYERS="${OFFLOAD_MOE_GATE_UP_SUBLAYERS:-[]}"
export SA_BLOCK_KV_DKV_COMPUTE="${SA_BLOCK_KV_DKV_COMPUTE:-512}"

# shellcheck source=/dev/null
source "${DIR}/qwen35_64c_256k.sh" "$@"
