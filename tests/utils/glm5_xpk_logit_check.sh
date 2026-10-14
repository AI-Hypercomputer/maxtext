#!/bin/bash
# Copyright 2023–2026 Google LLC
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

# Runs tests/utils/forward_pass_logit_checker.py for one or more golden-logit cases inside an XPK workload,
# fetching each golden file from GCS first (the checker only reads local files).
#
# Args: common MaxText key=value overrides and forward_pass_logit_checker flags, plus one or more
#   --case=<golden_gcs_path>@<comma-separated extra overrides>
# Each case waits (up to 4h) until its golden file exists in GCS, downloads it, and runs the checker.
#
# Example (GLM-5.3-Flash full scale, HF checkpoint converted on the fly):
#   xpk_run.sh --run-name glm53-logits --script tests/utils/glm5_xpk_logit_check.sh --device-type v5p-64 \
#     --base-docker-image .../maxtext_post_training_nightly:<date> -- \
#     model_name=glm5.3-flash override_model_config=true source_checkpoint_layout=safetensors_dynamic \
#     load_parameters_path=gs://snehalv-data/glm-5.3-flash/hf/full scan_layers=true per_device_batch_size=1 \
#     max_target_length=512 max_prefill_predict_length=512 --max_kl_div=0.5 --clip_logits_epsilon=1e-8 \
#     "--case=gs://snehalv-data/glm-5.3-flash/golden_logits/golden_glm5.3-flash_topk8.jsonl@dtype=float32,indexer_topk=8"
set -o pipefail
COMMON=()
CASES=()
for a in "$@"; do
  case "$a" in
    --case=*) CASES+=("${a#--case=}") ;;
    *) COMMON+=("$a") ;;
  esac
done
export PYTHONPATH=$PWD/src:$PWD
STATUS=0
i=0
for c in "${CASES[@]}"; do
  GOLDEN_GCS="${c%%@*}"
  EXTRA="${c#*@}"
  [ "$EXTRA" = "$c" ] && EXTRA=""
  GOLDEN_LOCAL=/tmp/golden_$i.jsonl
  echo "=== case $i: golden=$GOLDEN_GCS extra=$EXTRA"
  python3 - "$GOLDEN_GCS" "$GOLDEN_LOCAL" <<'EOF'
import sys, time
from google.cloud import storage
src, dst = sys.argv[1], sys.argv[2]
bucket, name = src[len("gs://"):].split("/", 1)
blob = storage.Client().bucket(bucket).blob(name)
t = time.time()
while not blob.exists():
  if time.time() - t > 4 * 3600:
    sys.exit(f"Timed out waiting for {src}")
  print(f"Waiting for {src} ...", flush=True)
  time.sleep(120)
blob.download_to_filename(dst)
print(f"Downloaded {src} -> {dst} after {time.time() - t:.0f}s", flush=True)
EOF
  IFS=',' read -r -a EXTRA_ARGS <<< "$EXTRA"
  python3 -m tests.utils.forward_pass_logit_checker "${COMMON[@]}" "${EXTRA_ARGS[@]}" \
    --golden_logits_path=$GOLDEN_LOCAL
  rc=$?
  echo "=== case $i exit code: $rc"
  [ $rc -ne 0 ] && STATUS=$rc
  i=$((i + 1))
done
exit $STATUS
