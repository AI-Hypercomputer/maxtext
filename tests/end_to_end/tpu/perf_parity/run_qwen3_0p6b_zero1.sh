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
# Zero-1 on qwen3-0.6b: what `shard_optimizer_over_data` costs and what it buys.
#
# Zero-1 shards the optimizer's parameter-shaped state over the `data` axis, so each
# replica keeps and updates 1/N of the moments and all-gathers the new parameters at the
# end of `update()`. It needs `shard_mode=explicit` (under `auto` the reshards are hints
# GSPMD may ignore), it is mutually exclusive with FSDP, and it is vacuous under `sgd` --
# hence `--dp 8 --opt adamw`, which is not the shape the other two runners use.
#
# Six arms, each at GA=1 and GA=8. The arms differ in one thing each, so the cost of the mesh
# mode and the cost of the feature come apart:
#
#   engine   dp=8 auto      adamw            the baseline: replicated optimizer
#   engine   dp=8 explicit  adamw            the same run, Explicit axes -- isolates the mesh mode
#   engine   dp=8 explicit  adamw  zero1     the feature
#   peft     dp=8 auto      adamw            PeftTrainer on the same mesh; it has no Zero-1
#   peft     dp=8 explicit  adamw            the same mesh mode on the other trainer
#   peft     fsdp=8 auto    adamw            what tunix would do instead to save the memory
#
# The last three are the "(or fsdp/tp for the tunix side)" half of the comparison: PeftTrainer
# cannot shard an optimizer over a data axis, so its way out of a replicated optimizer is to
# shard the parameters instead. Both answers cost a collective; the arms say which is cheaper.
#
# `peft dp=8 explicit` is the arm that keeps the headline honest. `--explicit` is what lets the
# engine defer its all-reduce, and without running the same flag on the other trainer there is
# no way to tell an engine capability from a property of Explicit axes. It is the latter that
# the result rules out: the same flag makes the engine 1.50x faster at GA=8 and PeftTrainer
# 1.31x slower.
#
# Engine-side Zero-1 support is newer than this rig. Every engine arm prints a `zero1:` line
# -- ACTIVE, DECLINED with a reason, or UNSUPPORTED on a build without it -- and a DECLINED
# run is the baseline wearing the Zero-1 arm's name. The summary at the end greps for it.
#
# For the sgd/tp shape the other three arms share, see run_qwen3_0p6b.sh.
#
# Usage: ./run_qwen3_0p6b_zero1.sh [output-dir]   # default: ./results-qwen3-0p6b-zero1
#
# Runs serially: every arm wants the whole host.

set -u

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$BENCH/results-qwen3-0p6b-zero1}"
mkdir -p "$OUT"

# Unscanned, to stay comparable with the qwen3-0.6b figures already recorded elsewhere.
SHAPE="--model qwen3-0.6b --seq 1024 --opt adamw"
DP="--dp 8"

run() { # <log-name> <script> <args...>
  local name=$1; shift
  echo "=== $name: $* ===" >&2
  { time "$@"; } > "$OUT/$name.log" 2>&1
  echo "EXIT=$?" >> "$OUT/$name.log"
}

cd "$BENCH" || exit 1

# Wall clock. --no-trace matters: the profiler charges per dispatch, so a traced A/B
# flatters whichever arm dispatches less.
for ga in 1 8; do
  run "engine-base-ga$ga"     python engine_profile.py       $SHAPE $DP            --no-trace --ga "$ga"
  run "engine-explicit-ga$ga" python engine_profile.py       $SHAPE $DP --explicit --no-trace --ga "$ga"
  run "engine-zero1-ga$ga"    python engine_profile.py       $SHAPE $DP --zero1    --no-trace --ga "$ga"
  run "peft-dp-ga$ga"         python peft_trainer_profile.py $SHAPE $DP            --no-trace --ga "$ga"
  run "peft-explicit-ga$ga"   python peft_trainer_profile.py $SHAPE $DP --explicit --no-trace --ga "$ga"
  run "peft-fsdp-ga$ga"       python peft_trainer_profile.py $SHAPE --fsdp 8       --no-trace --ga "$ga"
done

# Traces, for where the time went. The explicit control is traced too: `shard_mode` is not
# only a layout choice on this engine. Explicit axes are what let the gradients come out
# `unreduced`, which moves the data-parallel all-reduce out of every micro-batch and into
# `update()` -- so the control and the baseline compile to visibly different kernels, and
# without it there is no way to say how much of the Zero-1 arm's time is Zero-1.
for ga in 1 8; do
  for arm in "engine-base:engine_profile.py:" \
             "engine-explicit:engine_profile.py:--explicit" \
             "engine-zero1:engine_profile.py:--zero1" \
             "peft-dp:peft_trainer_profile.py:" \
             "peft-explicit:peft_trainer_profile.py:--explicit" ; do
    IFS=: read -r name script extra <<< "$arm"
    PERF_PARITY_PROFILE_ROOT="$OUT/traces" \
      python "$script" $SHAPE $DP $extra --steps 6 --ga "$ga" > "$OUT/traced-$name-ga$ga.log" 2>&1
  done
  PERF_PARITY_PROFILE_ROOT="$OUT/traces" \
    python peft_trainer_profile.py $SHAPE --fsdp 8 --steps 6 --ga "$ga" > "$OUT/traced-peft-fsdp-ga$ga.log" 2>&1
done

echo "=== zero1 gate (every engine arm must say what it did) ===" >&2
grep -H "^zero1:" "$OUT"/*.log >&2
echo "=== steady-state step times ===" >&2
grep -H "steady state" "$OUT"/*.log >&2
echo "=== peak HBM ===" >&2
grep -H "peak HBM" "$OUT"/*.log >&2
echo "=== traces ===" >&2
find "$OUT/traces" -name '*.xplane.pb' >&2
echo >&2
echo "TPU-busy per arm:  python xplane_device_summary.py --steps 3 <path>.xplane.pb" >&2
