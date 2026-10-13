#!/usr/bin/env bash
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

# AOT memory check (CPU, no TPU) of the e2e trainer program (Tunix GRPO loss, packed micro-batch with router replay,
# fwd_bwd / accumulate / update) with aot_trainer.py, inside an image whose MaxText and Tunix match the e2e run.
#
#   E2E_RENDER=<file> AOT_IMAGE=<image> [TRAINER_SLICE=4x8x8] [AOT_COMPILE_TOPOLOGY=tpu7x-512]
#   [EXTRA_MAXTEXT_FLAGS=... REPLACE_MAXTEXT_FLAGS=... LIBTPU_EXTRA=... LIBTPU_DROP=... COMPUTE_LOGPS_CHUNK_SIZE=...]
#   [LOG=<file>] bash aot_e2e.sh [extra MaxText key=value ...]
#
# The MaxText flags, --xla_* libtpu flags and mesh come from the e2e render (render_e2e.py --print-aot-env, with the
# same variant knobs as render_e2e.sh), so the compile matches what the bench would run. AOT_COMPILE_TOPOLOGY must
# name the slice: tpu7x-256 for 4x4x8 (the default), tpu7x-512 for 4x8x8. Summarize with aot_summarize.py <log>.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${E2E_RENDER:?set E2E_RENDER to the DRY_RUN output of your e2e launch}"
: "${AOT_IMAGE:?set AOT_IMAGE to an image with the MaxText and Tunix of the e2e run}"
_aot_env=$(python3 "$HERE/render_e2e.py" --print-aot-env --e2e-render "$E2E_RENDER" \
  --extra-maxtext-flags "${EXTRA_MAXTEXT_FLAGS:-}" --replace-maxtext-flags "${REPLACE_MAXTEXT_FLAGS:-}" \
  --libtpu-extra="${LIBTPU_EXTRA:-}" --libtpu-drop="${LIBTPU_DROP:-}" --trainer-slice "${TRAINER_SLICE:-}" \
  --compute-logps-chunk-size "${COMPUTE_LOGPS_CHUNK_SIZE:-}")
eval "$_aot_env"
case "${TRAINER_SLICE:-}" in 4x8x8) _topo=tpu7x-512 ;; *) _topo=tpu7x-256 ;; esac
TOPO=${AOT_COMPILE_TOPOLOGY:-$_topo}
LOG=${LOG:-logs/aot-e2e-$(date -u +%Y%m%dT%H%M%SZ).log}
mkdir -p "$(dirname "$LOG")"
echo "log: $LOG"
{ echo "=== $(date -u -Is) aot_e2e TOPO=$TOPO FSDP=$AOT_MESH_FSDP TP=$AOT_MESH_TP EXPERT=$AOT_MESH_EXPERT CONTEXT=$AOT_MESH_CONTEXT IMAGE=$AOT_IMAGE"
  echo "MAXTEXT_EXTRA_FLAGS=$MAXTEXT_EXTRA_FLAGS"; echo "LIBTPU_INIT_ARGS=$LIBTPU_INIT_ARGS"; } | tee -a "$LOG"
docker run --rm --entrypoint python -v "$HERE":/bench:ro \
  -e JAX_PLATFORMS=cpu,tpu -e LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS" -e MAXTEXT_EXTRA_FLAGS="$MAXTEXT_EXTRA_FLAGS" \
  -e AOT_MESH_FSDP="$AOT_MESH_FSDP" -e AOT_MESH_TP="$AOT_MESH_TP" -e AOT_MESH_EXPERT="$AOT_MESH_EXPERT" \
  -e AOT_MESH_CONTEXT="$AOT_MESH_CONTEXT" -e AOT_COMPILE_TOPOLOGY="$TOPO" \
  -e AOT_LOGPS_CHUNK_SIZE="${AOT_LOGPS_CHUNK_SIZE:-512}" -e TF_CPP_MIN_LOG_LEVEL=1 \
  "$AOT_IMAGE" /bench/aot_trainer.py "$@" 2>&1 | tee -a "$LOG"
