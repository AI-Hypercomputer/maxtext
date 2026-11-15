#!/bin/bash
# Copyright 2023-2026 Google LLC
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
# One-step SGD gradient twins with the pre-training trainer (pre_train/train.py), for comparing two configs pairwise.
#
#   1. p0:   one step of SGD at learning_rate=0 from init_weights_seed. Its checkpoint "0" (MaxText labels the state after
#            the first update step 0) is exactly the initial params, written by the same trainer and model code.
#   2. twins: each config loads p0 (load_parameters_path) and takes one SGD step at LR=1e8 on the same synthetic batch
#            (dataset_type=synthetic is a fixed batch; same seeds): checkpoint "0" holds p1 = p0 - LR * g.
#   3. compare.py p0 p1_a p1_b: g = (p0 - p1) / LR, whole-model and per-tensor |g_a - g_b| / |g_a|.
# Loading p0 in both twins (instead of re-initializing at the same seed) keeps init out of the comparison: two configs
# whose param layout or init code differ would otherwise start from different p0.
#
# Usage: run.sh OUT_DIR "<common args>" "<config A args>" "<config B args>"
#   e.g. run.sh /tmp/gt "model_name=olmo35-tiny ..." "kda_chunked_impl=subblock" "kda_chunked_impl=pairwise"
# On TPU set OUT_DIR to gs://... and run the same commands on every host; locally it runs one process.
set -euo pipefail
OUT=$1
COMMON=$2
A=$3
B=$4
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../../../../.." && pwd)}
cd "$REPO"
# shellcheck disable=SC2086
TRAIN() { python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml base_output_directory="$OUT" \
  dataset_type=synthetic steps=1 learning_rate_schedule_steps=1 warmup_steps_fraction=0.0 lr_schedule_type=cosine \
  opt_type=sgd gradient_clipping_threshold=0.0 weight_dtype=float32 init_weights_seed=0 data_shuffle_seed=0 \
  enable_checkpointing=true checkpoint_period=1 save_checkpoint_on_start=true async_checkpointing=false \
  enable_goodput_recording=false skip_jax_distributed_system=true $COMMON "$@"; }

# shellcheck disable=SC2086
[ -d "$OUT/p0/checkpoints/0/items" ] || TRAIN run_name=p0 learning_rate=0.0 $A
P0="$OUT/p0/checkpoints/0/items"
# shellcheck disable=SC2086
TRAIN run_name=twin_a learning_rate=1e8 load_parameters_path="$P0" $A
# shellcheck disable=SC2086
TRAIN run_name=twin_b learning_rate=1e8 load_parameters_path="$P0" $B
python3 "$HERE/compare.py" "$P0" "$OUT/twin_a/checkpoints/0/items" "$OUT/twin_b/checkpoints/0/items" \
  --json "$OUT/grad_twin.json" | tee "$OUT/grad_twin.txt"
