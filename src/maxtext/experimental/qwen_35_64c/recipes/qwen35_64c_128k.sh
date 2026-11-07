#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# ==============================================================================
# 128K Recipe (S = 131,072, PDB = 0.125, GBS = 16, FSDP = 8, EP = 16)
# Inherits from qwen35_64c_base.sh and overrides only the 128K topology & remat.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-qwen35-397b-128k-cf125}"

export SEQ_LEN="${SEQ_LEN:-131072}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export FSDP_SIZE="${FSDP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-16}"
export DECOUPLE_EP_D2D="${DECOUPLE_EP_D2D:-true}"

export MOE_BWD_STASH_OVERFLOW="${MOE_BWD_STASH_OVERFLOW:-false}"
export SAVE_DECODER_INPUT_HBM="${SAVE_DECODER_INPUT_HBM:-false}"
export OFFLOAD_DECODER_INPUT="${OFFLOAD_DECODER_INPUT:-true}"
export SAVE_MOE_GATE_UP_SUBLAYERS="${SAVE_MOE_GATE_UP_SUBLAYERS:-[]}"
export OFFLOAD_MOE_GATE_UP_SUBLAYERS="${OFFLOAD_MOE_GATE_UP_SUBLAYERS:-[0, 1, 2, 3]}"
export SAVE_KEY_VALUE_PROJ="${SAVE_KEY_VALUE_PROJ:-false}"
export OFFLOAD_GDN_QKV="${OFFLOAD_GDN_QKV:-true}"

# shellcheck source=/dev/null
source "${DIR}/qwen35_64c_base.sh" "$@"
