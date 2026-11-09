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
#   S=16384, GBS=64 (PDB=0.5), GA=1, FSDP=64, CP=2, EP=1, remat=selective.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DIR

export JOB_NAME="${JOB_NAME:-flint-40b-64c-16k-pdb05}"
export SEQ_LEN="${SEQ_LEN:-16384}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export FSDP_SIZE="${FSDP_SIZE:-64}"
export CP_SIZE="${CP_SIZE:-2}"
export EP_SIZE="${EP_SIZE:-1}"
export REMAT_POLICY="${REMAT_POLICY:-selective}"

# shellcheck source=/dev/null
source "${DIR}/flint_64c_base.sh" "$@"
