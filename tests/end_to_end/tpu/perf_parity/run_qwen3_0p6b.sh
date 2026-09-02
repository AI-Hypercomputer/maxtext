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
# All three arms on qwen3-0.6b: unscanned, tp=8, at GA=1 and GA=8. This is the shape
# where the tunix-model arm exists, so it is the only model that answers both questions
# at once -- how much of the gap is the model (`peft_trainer_profile` vs
# `qwen3_0p6b_tunix_profile`, trainer fixed) and how much is the trainer (`engine_profile`
# vs `peft_trainer_profile`, model fixed).
#
# For the 35b counterpart, which has no tunix-model arm, see run_qwen3_5_35b_a3b.sh.
#
# Usage: ./run_qwen3_0p6b.sh [output-dir]        # default: ./results-qwen3-0p6b
#
# Runs serially: every arm wants the whole host.

set -u

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$BENCH/results-qwen3-0p6b}"
mkdir -p "$OUT"

# Unscanned because tunix builds its 28 decoder layers as a ModuleList and runs a Python
# loop over them; --scan here would compare compilation strategies, not implementations.
SHAPE="--model qwen3-0.6b --tp 8 --seq 1024"

run() { # <log-name> <script> <args...>
  local name=$1; shift
  echo "=== $name: $* ===" >&2
  # `time` is what the end-to-end process row is read from; the arm's own `train() total`
  # covers the loop only.
  { time "$@"; } > "$OUT/$name.log" 2>&1
  echo "EXIT=$?" >> "$OUT/$name.log"
}

cd "$BENCH" || exit 1

# Wall clock. --no-trace matters: the profiler charges per dispatch, so a traced A/B
# flatters whichever arm dispatches less. Quote these for time, the traced pass for where
# the time went.
for ga in 1 8; do
  run "engine-ga$ga"       python engine_profile.py            $SHAPE --no-trace --ga "$ga"
  run "peft-maxtext-ga$ga" python peft_trainer_profile.py      $SHAPE --no-trace --ga "$ga"
  run "peft-tunix-ga$ga"   python qwen3_0p6b_tunix_profile.py  --tp 8 --seq 1024 --no-trace --ga "$ga"
done

# Traces, for TPU-busy and utilization. --steps 6 keeps each xplane manageable; the
# steady-state window still has 3 steps in it after the 3 warmup steps are dropped.
for ga in 1 8; do
  PERF_PARITY_PROFILE_ROOT="$OUT/traces" \
    python engine_profile.py       $SHAPE --steps 6 --ga "$ga" > "$OUT/traced-engine-ga$ga.log" 2>&1
  PERF_PARITY_PROFILE_ROOT="$OUT/traces" \
    python peft_trainer_profile.py $SHAPE --steps 6 --ga "$ga" > "$OUT/traced-peft-maxtext-ga$ga.log" 2>&1
done

echo "=== steady-state step times ===" >&2
grep -H "steady state" "$OUT"/*.log >&2
echo "=== traces ===" >&2
find "$OUT/traces" -name '*.xplane.pb' >&2
echo >&2
echo "TPU-busy per arm:  python xplane_device_summary.py --steps 3 <path>.xplane.pb" >&2
