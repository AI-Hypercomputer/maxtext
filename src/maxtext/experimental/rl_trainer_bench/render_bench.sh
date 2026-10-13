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

# Renders the RL trainer benchmark JobSet (bench_driver.py on the trainer JobSet of an e2e DRY_RUN render).
# Writes one YAML file and nothing else; it never talks to a cluster.
#
# Required:
#   LAUNCH_RENDER=<file>     DRY_RUN output of the e2e launcher (see README)
#   NAME=<name>              JobSet name, worker id and output folder (lowercase, at most 30 characters)
#   K8S_NAMESPACE=<namespace>
#   OUTPUT_PREFIX=<prefix>   the bench writes to <prefix>/<NAME> (e.g. a GCS folder you own)
# Optional:
#   OUT=<file>               default ./rendered/<NAME>.yaml
#   IMAGE=<image>            client image; default keeps the e2e render's
#   STEPS=3                  optimizer steps
#   MICRO_STEPS=<n>          fwd_bwd calls per step; default 9 (5 with TRAINER_SLICE=4x8x8), see README
#   EXTRA_MAXTEXT_FLAGS="k=v ..."      appended to MAXTEXT_EXTRA_FLAGS (last wins)
#   REPLACE_MAXTEXT_FLAGS="k=v ..."    replace existing MAXTEXT_EXTRA_FLAGS keys in place
#   SET_TRAINER_FLAGS="k=v ..."        replace existing trainer flags in place (e.g. train_micro_batch_size=128)
#   SET_ORCH_FLAGS="k=v ..."           replace existing orchestrator flags in place
#   LIBTPU_EXTRA="--flag=v ..."        add or override libtpu flags (client, proxy and workers)
#   LIBTPU_DROP="--flag ..."           remove libtpu flags by name
#   COMPUTE_LOGPS_CHUNK_SIZE=<n>       trainer --compute_logps_chunk_size in place; empty keeps the render's
#   TRAINER_SLICE=<topology>           e.g. 4x8x8: re-target the trainer slice, FSDP fills it
#   PROFILER_SKIP=<n> PROFILER_STEPS=<m>  trace fwd_bwd calls n .. n+m-1 of the run (see README)
#   DRIVER_ENV="K=V ..."               bench_driver.py knobs: PAYLOAD ROUTING ROUTING_SCALE ROUTING_TARGETS
#                                      PAD_FRACTION FINAL_PAD_FRACTION PAYLOAD_SEED EP_SIZE ADVANTAGES LOGP_GATES
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LAUNCH_RENDER:?set LAUNCH_RENDER to the DRY_RUN output of your e2e launch}"
: "${NAME:?set NAME (lowercase, <=30 chars)}"
: "${K8S_NAMESPACE:?set K8S_NAMESPACE}"
: "${OUTPUT_PREFIX:?set OUTPUT_PREFIX (the bench writes to OUTPUT_PREFIX/NAME)}"
OUT=${OUT:-rendered/$NAME.yaml}
mkdir -p "$(dirname "$OUT")"
# Default micro-steps: an example calibration from the 397B recipe, where one optimizer step of about 1,152
# trajectories packed into 9 micro-batches of 64 rows on a 4x4x8 trainer, and the same token mass into 5 micro-batches
# of 128 rows on 4x8x8. Set MICRO_STEPS to the fwd_bwd count per step of your own e2e run.
case "${TRAINER_SLICE:-}" in 4x8x8) _ms=5 ;; *) _ms=9 ;; esac
python3 "$HERE/render_bench.py" \
  --launch-render "$LAUNCH_RENDER" --driver "$HERE/bench_driver.py" --out "$OUT" \
  --name "$NAME" --namespace "$K8S_NAMESPACE" --output-prefix "$OUTPUT_PREFIX" \
  --steps "${STEPS:-3}" --micro-steps "${MICRO_STEPS:-$_ms}" \
  --extra-maxtext-flags "${EXTRA_MAXTEXT_FLAGS:-}" --replace-maxtext-flags "${REPLACE_MAXTEXT_FLAGS:-}" \
  --set-trainer-flags "${SET_TRAINER_FLAGS:-}" --set-orch-flags "${SET_ORCH_FLAGS:-}" \
  --libtpu-extra="${LIBTPU_EXTRA:-}" --libtpu-drop="${LIBTPU_DROP:-}" \
  --image "${IMAGE:-}" --driver-env "${DRIVER_ENV:-}" --trainer-slice "${TRAINER_SLICE:-}" \
  --compute-logps-chunk-size "${COMPUTE_LOGPS_CHUNK_SIZE:-}" \
  --profiler-skip "${PROFILER_SKIP:-}" --profiler-steps "${PROFILER_STEPS:-}"
