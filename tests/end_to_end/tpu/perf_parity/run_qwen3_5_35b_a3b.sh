#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Trainer-vs-trainer on MaxText qwen3.5-35b-a3b. Produces the tables in
# RESULTS-qwen35-35b-20260902.md; run on 8 devices (measured on a v7-8, 4 Ironwood chips).
#
# Three things about this model are not free choices, and each cost a failed run to learn:
#
#   * `--tp 2`, not 8. The model has 2 KV heads, and `_validate_kv_head_sharding` requires
#     num_kv_heads % tp == 0, so tp=8 and tp=4 are both rejected outright. fsdp fills the
#     rest of the mesh: fsdp=4 x tp=2.
#   * `--scan`. Unscanned, 40 layers x (256 routed + 1 shared) experts OOMs regardless of
#     remat policy or batch size.
#   * Only the two MaxText-side arms exist. `qwen3_0p6b_tunix_profile.py` cannot run this
#     model -- tunix implements one architecture -- so the comparison is engine vs
#     PeftTrainer over the *same* MaxText model, which is the pairing that isolates the
#     trainer anyway.
#
# GA coverage is deliberately asymmetric. PeftTrainer runs at GA=1 only: every GA>1 dies
# with an identical `jit__update_step` needing 161.94 G of HLO temporaries against 94.74 G
# available, because off its single-microstep fast path `GradientAccumulator` allocates a
# full fp32 copy of the parameter tree. The figure is identical at GA=2, 4 and 8, which is
# the tell that it is parameter-shaped and not GA-depth-shaped -- no batch or remat knob
# moves it. The GA=2/4 runs below are kept precisely to re-demonstrate that.
#
# Usage: ./run_qwen3_5_35b_a3b.sh [output-dir]   # default: ./results-qwen3-5-35b-a3b
#
# Runs serially: every arm wants the whole host. Budget ~90 min for the full sweep.

set -u

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$BENCH/results-qwen3-5-35b-a3b}"
mkdir -p "$OUT"

SHAPE="--model qwen3.5-35b-a3b --tp 2 --scan --seq 1024"

run() { # <log-name> <script> <args...>
  local name=$1; shift
  echo "=== $name: $* ===" >&2
  { time "$@"; } > "$OUT/$name.log" 2>&1
  echo "EXIT=$?" >> "$OUT/$name.log"
}

cd "$BENCH" || exit 1

# Wall clock, untraced. Tracing overhead is negligible on this model (<=0.08%, unlike
# qwen3-0.6b) but the untraced run is still the one to quote.
run engine-ga1       python engine_profile.py       $SHAPE --no-trace --ga 1
run peft-maxtext-ga1 python peft_trainer_profile.py $SHAPE --no-trace --ga 1

# Engine GA scaling. Expect linear, with per-micro cost improving slightly as the update
# amortizes over more micro-batches.
for ga in 2 4 8; do
  run "engine-ga$ga"       python engine_profile.py       $SHAPE --no-trace --ga "$ga"
  # Expected to OOM; the log is the evidence for the GA-independent 161.94 G figure.
  run "peft-maxtext-ga$ga" python peft_trainer_profile.py $SHAPE --no-trace --ga "$ga"
done

# Traces for the device-side table. --steps 6 keeps each xplane near ~1 GiB at this shape.
#
# The GA=8 device trace *truncates*: the buffer holds a bounded number of module events and
# captures only 9 of 48 fwd_bwd executions. Build the step from the per-execution column
# (`ms/exec`), never from totals/steps, which understates it by ~5x.
for ga in 1 8; do
  PERF_PARITY_PROFILE_ROOT="$OUT/traces" \
    python engine_profile.py       $SHAPE --steps 6 --ga "$ga" > "$OUT/traced-engine-ga$ga.log" 2>&1
done
PERF_PARITY_PROFILE_ROOT="$OUT/traces" \
  python peft_trainer_profile.py   $SHAPE --steps 6 --ga 1     > "$OUT/traced-peft-maxtext-ga1.log" 2>&1

echo "=== steady-state step times ===" >&2
grep -H "steady state" "$OUT"/*.log >&2
echo "=== PeftTrainer GA>1 OOM (expected) ===" >&2
grep -H "exceeds available HBM" "$OUT"/*.log >&2
echo "=== traces ===" >&2
find "$OUT/traces" -name '*.xplane.pb' >&2
echo >&2
echo "TPU-busy per arm:  python xplane_device_summary.py --steps 3 <path>.xplane.pb" >&2
echo "  read TPU-busy off the XLA Modules line, not XLA Ops -- on this scanned MoE the" >&2
echo "  Ops line reads ~2x the module time and ~2x the wall step." >&2
