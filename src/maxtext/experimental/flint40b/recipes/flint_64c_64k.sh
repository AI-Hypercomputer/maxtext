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

# Flint-40B 64-chip TPU v7x recipe:
#   S=65536, GBS=128 (PDB=1), GA=1, FSDP=128, CP=1, EP=1, remat=auto.
# Enables the long-context kernel stack (sdiag softmax gate, transposed global
# forward, bf16 dKV softmax, fused gmm_v2 dense FP8, saved first sliding-layer
# attention output, and dKV-only FP8 Q @ K^T; see README.md).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DIR

export JOB_NAME="${JOB_NAME:-flint-40b-64c-64k}"
export SEQ_LEN="${SEQ_LEN:-65536}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export FSDP_SIZE="${FSDP_SIZE:-128}"
export CP_SIZE="${CP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-1}"
export REMAT_POLICY="${REMAT_POLICY:-auto}"
export SPLASH_GLOBAL_MAX_LOGIT_CONST="${SPLASH_GLOBAL_MAX_LOGIT_CONST:-none}"
export SPLASH_LOCAL_MAX_LOGIT_CONST="${SPLASH_LOCAL_MAX_LOGIT_CONST:-none}"
export SPLASH_GLOBAL_SDIAG_GATE="${SPLASH_GLOBAL_SDIAG_GATE:-true}"
export SPLASH_LOCAL_SDIAG_GATE="${SPLASH_LOCAL_SDIAG_GATE:-true}"
export SPLASH_GLOBAL_FWD_TRANSPOSED_SV="${SPLASH_GLOBAL_FWD_TRANSPOSED_SV:-true}"
export SPLASH_GLOBAL_DKV_BF16_SOFTMAX="${SPLASH_GLOBAL_DKV_BF16_SOFTMAX:-true}"
export DENSE_FP8_GMM="${DENSE_FP8_GMM:-true}"
export SAVE_LOCAL_ATTN_OUT_LAYERS="${SAVE_LOCAL_ATTN_OUT_LAYERS:-1}"
export SPLASH_GLOBAL_DKV_FP8_QK="${SPLASH_GLOBAL_DKV_FP8_QK:-true}"

# shellcheck source=/dev/null
source "${DIR}/flint_64c_base.sh" "$@"
