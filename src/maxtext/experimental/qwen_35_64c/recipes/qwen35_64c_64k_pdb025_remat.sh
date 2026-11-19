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
# 64K PDB=0.25 Aggressive Remat Recipe (S = 65,536, PDB = 0.25, GBS = 32)
# Inherits from qwen35_64c_base.sh and packs 3 of 4 MoE gate_up sublayers + KV
# projections into HBM while halving PCIe offload.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-qwen35-397b-64k-cf125-pdb025-remat-3gu-v2}"

export SAVE_MOE_GATE_UP_SUBLAYERS="${SAVE_MOE_GATE_UP_SUBLAYERS:-[0, 1, 3]}"
export OFFLOAD_MOE_GATE_UP_SUBLAYERS="${OFFLOAD_MOE_GATE_UP_SUBLAYERS:-[2]}"
export SAVE_KEY_VALUE_PROJ="${SAVE_KEY_VALUE_PROJ:-true}"

# shellcheck source=/dev/null
source "${DIR}/qwen35_64c_base.sh" "$@"
