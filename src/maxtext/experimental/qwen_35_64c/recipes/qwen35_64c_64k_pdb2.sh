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
# 64K PDB=2.0 Recipe (S = 65,536, PDB = 2.0, GBS = 256, GA = 4, FSDP = 16, EP = 8)
# Inherits from qwen35_64c_64k_pdb1.sh and overrides only GBS = 256 and GA = 4.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-qwen35-397b-64k-cf125-pdb2}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

# shellcheck source=/dev/null
source "${DIR}/qwen35_64c_64k_pdb1.sh" "$@"
