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

# Pathways on a single-host TPU VM's own chips, in docker: the resource manager, one worker (the VM's
# chips) and the IFRT proxy, run the way xpk's Pathways head and worker pods run them on GKE, all
# --net=host on 127.0.0.1. A local loop for programs that compile only on Pathways, such as Shardy
# MPMD's IFRT IR:
#
#   bash local_pathways.sh up --gcs_scratch_location=gs://BUCKET/DIR --instance_type=tpu7x:2x2x1
#   JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
#       python3 -m maxtext.experimental.mpmd.partir.train --pp=4 --fsdp=2 --num_microbatches=4 \
#       --per_device_batch_size=4 --model_name=simple
#   bash local_pathways.sh status   # containers, worker restarts, slice state
#   bash local_pathways.sh down     # frees the TPU again
#
# up's arguments:
#   --gcs_scratch_location  (required) gs:// location for Pathways' scratch files and its persistent
#                           compilation cache. The containers authenticate as the VM's service
#                           account, which needs write access to it.
#   --instance_type         (required) the VM's slice as the resource manager names it: tpu7x:2x2x1
#                           on a TPU7x VM with 4 chips (tpu7x-8), tpuv6e:2x2 on a v6e-4.
#   --tag                   Pathways image tag; it must match the client's jax (default
#                           20261117-jax_0.12.0.1; the 20261118, 20261120 and 20261124 jax_0.12.0.1
#                           images reject the IFRT IR of the Shardy MPMD programs jax 0.12.0.1 lowers).
#   --xla_flags_file        File of XLA / libtpu flags, one --key=value per line (# comments), for the
#                           proxy (which compiles), the resource manager and the worker: launch.py's
#                           --xla_flags_file.
# Environment: DOCKER, the docker command (default: sudo docker).
#
# While the worker is up it owns the TPU, so plain-JAX runs on this VM fail until 'down'. The worker
# exits on purpose each time a client disconnects, so it runs under --restart=on-failure (as xpk's
# Pathways JobSet restarts it on GKE; see the README about gcluster); the next client's first
# execution waits about 10 s for it to rejoin. Profiling: under Pathways, jax.profiler.start_trace
# accepts only gs:// directories, and the worker's TPU tracer starts late: work dispatched within
# about a second of start_trace is missing from the trace.
set -euo pipefail

TAG=20261117-jax_0.12.0.1
GCS=
INSTANCE_TYPE=
XLA_FLAGS_FILE=
IMG=us-docker.pkg.dev/cloud-tpu-v2-images/pathways
DOCKER=${DOCKER:-sudo docker}
PREFIX=maxtext-pathways
NAMES=("$PREFIX-rm" "$PREFIX-worker" "$PREFIX-proxy")
LOGS=(--log-opt max-size=50m --log-opt max-file=2)

usage() {
  echo "usage: $0 up --gcs_scratch_location=gs://BUCKET/DIR --instance_type=TYPE [--tag=TAG]" \
    "[--xla_flags_file=FILE] | down | status"
  exit 2
}

# (not grep -q: under pipefail, grep -q exiting early fails 'docker logs' with SIGPIPE on a long log)
slice_state() { $DOCKER logs "$PREFIX-rm" 2>&1 | grep -o '[0-9]*/[0-9]* Pathways Slices Now Ready' | tail -1; }
proxy_ready() { $DOCKER logs "$PREFIX-proxy" 2>&1 | grep 'IFRT proxy server started with status OK' >/dev/null; }

up() {
  for arg in "$@"; do
    case $arg in
      --gcs_scratch_location=*) GCS=${arg#*=} ;;
      --instance_type=*) INSTANCE_TYPE=${arg#*=} ;;
      --tag=*) TAG=${arg#*=} ;;
      --xla_flags_file=*) XLA_FLAGS_FILE=${arg#*=} ;;
      *) echo "unknown argument: $arg"; usage ;;
    esac
  done
  if [[ $GCS != gs://* ]]; then echo "--gcs_scratch_location=gs://... is required"; usage; fi
  if [[ -z $INSTANCE_TYPE ]]; then echo "--instance_type is required"; usage; fi
  local flags=() flag
  if [[ -n $XLA_FLAGS_FILE ]]; then
    if [[ ! -r $XLA_FLAGS_FILE ]]; then echo "cannot read --xla_flags_file=$XLA_FLAGS_FILE"; exit 1; fi
    mapfile -t flags < <(sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$XLA_FLAGS_FILE")
    for flag in "${flags[@]}"; do
      if [[ $flag != --?*=* || $flag == *[[:space:]]* ]]; then
        echo "$XLA_FLAGS_FILE: expected one --key=value a line, got '$flag'"; exit 1
      fi
    done
  fi
  if [[ -n "$($DOCKER ps -aq --filter name="^$PREFIX-")" ]]; then
    echo "$PREFIX-* containers exist already: run '$0 down' first"; exit 1
  fi
  if sudo lsof /dev/vfio/[0-9]* >/dev/null 2>&1; then
    echo "the TPU is in use:"; sudo lsof /dev/vfio/[0-9]*; exit 1
  fi
  $DOCKER run -d --name "$PREFIX-rm" --net=host "${LOGS[@]}" \
      -e TPU_SKIP_MDS_QUERY=true -e JOBSET_NAME="$PREFIX" -e REPLICATED_JOB_NAME=pathways-head \
      -e HOST_ADDRESS=127.0.0.1 \
      "$IMG/server:$TAG" --server_port=29001 --gcs_scratch_location="$GCS" \
      --node_type=resource_manager --instance_count=1 --instance_type="$INSTANCE_TYPE" "${flags[@]}" >/dev/null
  $DOCKER run -d --name "$PREFIX-worker" --net=host --privileged --restart=on-failure "${LOGS[@]}" \
      -e TPU_MIN_LOG_LEVEL=0 -e TF_CPP_MIN_LOG_LEVEL=0 -e XCLOUD_ENVIRONMENT=GCP \
      -e MEGASCALE_GRPC_ENABLE_XOR_TRACER=false -e MEGASCALE_NUM_SLICES=1 -e MEGASCALE_SLICE_ID=0 \
      -e JOBSET_NAME="$PREFIX" -e REPLICATED_JOB_NAME=worker -e PATHWAYS_HEAD=127.0.0.1 \
      -e MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1 \
      "$IMG/server:$TAG" --server_port=29005 --resource_manager_address=127.0.0.1:29001 \
      --gcs_scratch_location="$GCS" "${flags[@]}" >/dev/null
  $DOCKER run -d --name "$PREFIX-proxy" --net=host "${LOGS[@]}" -e PATHWAYS_HEAD=127.0.0.1 \
      "$IMG/proxy_server:$TAG" --server_port=29000 --resource_manager_address=127.0.0.1:29001 \
      --gcs_scratch_location="$GCS" "${flags[@]}" >/dev/null
  for _ in $(seq 90); do
    if [[ "$(slice_state)" == "1/1 Pathways Slices Now Ready" ]] && proxy_ready; then
      echo "Pathways up ($INSTANCE_TYPE, $TAG, ${#flags[@]} XLA flags):" \
        "JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000"
      return
    fi
    if [[ "$($DOCKER inspect -f '{{.State.Running}}' "$PREFIX-rm" 2>/dev/null)" != true ]]; then
      echo "the resource manager exited:"; status; $DOCKER logs --tail 30 "$PREFIX-rm" 2>&1; exit 1
    fi
    sleep 2
  done
  echo "not ready after 180 s:"; status; exit 1
}

down() {
  for n in "${NAMES[@]}"; do $DOCKER rm -f "$n" >/dev/null 2>&1 && echo "removed $n" || true; done
}

status() {
  $DOCKER ps -a --filter name="^$PREFIX-" --format '{{.Names}}\t{{.Status}}'
  if $DOCKER inspect "$PREFIX-worker" >/dev/null 2>&1; then
    echo "worker restarts: $($DOCKER inspect "$PREFIX-worker" --format '{{.RestartCount}}')"
  fi
  if $DOCKER inspect "$PREFIX-rm" >/dev/null 2>&1; then echo "slices: $(slice_state)"; fi
  if $DOCKER inspect "$PREFIX-proxy" >/dev/null 2>&1; then
    if proxy_ready; then echo "proxy: ready"; else echo "proxy: not ready"; fi
  fi
}

case "${1:-}" in
  up) shift; up "$@" ;;
  down | status) "$1" ;;
  *) usage ;;
esac
