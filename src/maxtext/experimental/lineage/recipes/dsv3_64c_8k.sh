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

# Lineage DeepSeek-V3 671B (+ 1 MTP layer) 64-chip TPU v7x 8K recipe:
#   SEQ_LEN=8192, PDB=1.0, GBS=128 (1,048,576 tokens/step), GA=1,
#   mesh ici=[1,4,4,4,2] (ep8_fsdp64, CP=1, EP=8).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DIR

export RUN_NAME="${RUN_NAME:-dsv3-671b-64c-8k}"
export SEQ_LEN="${SEQ_LEN:-8192}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export LINEAGE_LAYOUT_4X4X4="${LINEAGE_LAYOUT_4X4X4:-ep8_fsdp64}"
export LINEAGE_LOSS_CHUNK_TOKENS="${LINEAGE_LOSS_CHUNK_TOKENS:-2048}"
export BF16_PROPAGATION="${BF16_PROPAGATION:-true}"
export PIPELINE_MEMORY_HOST_TRANSFERS="${PIPELINE_MEMORY_HOST_TRANSFERS:-auto}"

# shellcheck source=/dev/null
source "${DIR}/dsv3_64c_base.sh" "$@"
