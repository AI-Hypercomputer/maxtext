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

set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTHONPATH=/app/src:/app:$PYTHONPATH

# Resolve slice ID and host addresses from MEGASCALE environment variables before unsetting
SLICE_ID="${MEGASCALE_SLICE_ID:-0}"
JOB_NAME="${MEGASCALE_COORDINATOR_ADDRESS#*.}"
SRC_HOST=$(getent hosts "${MEGASCALE_COORDINATOR_ADDRESS}" | awk '{print $1}')
DST_HOST=$(getent hosts "${JOB_NAME}-slice-job-1-0.${JOB_NAME}" | awk '{print $1}')

if [[ -z "${SRC_HOST}" ]]; then
  SRC_HOST="127.0.0.1"
fi
if [[ -z "${DST_HOST}" ]]; then
  DST_HOST="127.0.0.1"
fi
CTRL_HOST="${SRC_HOST}"

echo "=== [Setup Info] SliceID: ${SLICE_ID}, JobName: ${JOB_NAME}, Hostname: $(hostname), SrcHost: ${SRC_HOST}, DstHost: ${DST_HOST}, CtrlHost: ${CTRL_HOST} ==="

# Set standalone single-slice TPU environment so each slice runs independent JAX on local TPUs
unset MEGASCALE_COORDINATOR_ADDRESS
unset MEGASCALE_NUM_SLICES
unset MEGASCALE_SLICE_ID
export TPU_WORKER_HOSTNAMES="localhost"
export TPU_WORKER_ID=0
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="1,1,1"
export TPU_TOPOLOGY="2x2x1"
export TPU_SKIP_MDS_QUERY="true"

if [[ "${SLICE_ID}" == "0" ]]; then
  echo "=== [Slice 0] Starting Central Controller on ${CTRL_HOST}:29500 ==="
  python3 -u /app/src/maxtext/experimental/weight_transfer/transfer_weights_raiden_multi_client.py \
      --role=controller \
      --controller_port=29500 \
      --iterations=${ITERATIONS:-5} \
      --warmup_iterations=${WARMUP_ITERATIONS:-3} &
  CTRL_PID=$!

  # Wait for controller port to be open
  for i in {1..30}; do
    if python3 -c "import socket; s = socket.socket(); s.connect(('127.0.0.1', 29500)); s.close()" 2>/dev/null; then
      echo "=== [Slice 0] Central Controller is UP and listening on 29500 ==="
      break
    fi
    sleep 1
  done

  echo "=== [Slice 0] Starting Source Worker ==="
  python3 -u /app/src/maxtext/experimental/weight_transfer/transfer_weights_raiden_multi_client.py \
      --role=source \
      --controller_address="${CTRL_HOST}:29500" \
      --local_ip="${SRC_HOST}"
  wait $CTRL_PID
else
  echo "=== [Slice 1] Waiting for Controller at ${CTRL_HOST}:29500 ==="
  for i in {1..60}; do
    if python3 -c "import socket; s = socket.socket(); s.settimeout(2.0); s.connect(('${CTRL_HOST}', 29500)); s.close()" 2>/dev/null; then
      echo "=== [Slice 1] Connected to Controller at ${CTRL_HOST}:29500 ==="
      break
    fi
    sleep 1
  done

  echo "=== [Slice 1] Starting Destination Worker ==="
  python3 -u /app/src/maxtext/experimental/weight_transfer/transfer_weights_raiden_multi_client.py \
      --role=destination \
      --controller_address="${CTRL_HOST}:29500" \
      --local_ip="${DST_HOST}" \
      --verify=true
fi
