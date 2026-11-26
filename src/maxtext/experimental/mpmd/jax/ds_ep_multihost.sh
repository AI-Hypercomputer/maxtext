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

# One host of a multi-host MPMD DeepSeek run: every pod of the slice runs this
# (xpk / GKE, one process per host).  The pipeline is ds_ep_pipeline
# --distributed: each process runs the stage holding its chips and takes part
# in every boundary copy.
#
#   RUN_DIR=gs://bucket/run bash src/maxtext/experimental/mpmd/jax/ds_ep_multihost.sh --stages 4 --fsdp 64 --ep 1 \
#       --layout physical --transport xfer --bench --profile '$RUN_DIR/xprof' [pipeline flags...]
#
# Across v6e or v5p hosts pass --transport xfer (or ppermute; across slices, with --layout slices, relay): with the
# default, native (jax's cross-host copy), every receiving host aborts (std::bad_alloc in the receive, on a v6e-256
# and on a v5p-128, 2026-10, jax 0.11.1, libtpu 0.0.46).
#
# It runs `$PYTHON -u -m maxtext.experimental.mpmd.jax.ds_ep_pipeline --distributed --json <record> "$@"` from the
# directory the package imports from, under a timeout, with LIBTPU_INIT_ARGS built as ds_ep_run.sh builds it (on GPU,
# XLA_FLAGS: below) and no hash randomization (every process must build the same plan; the run checks).  Then it
# copies this host's log and run records to RUN_DIR, and exits with the run's status.  They live in a directory made
# for this invocation alone under $TMPDIR (default /tmp), so that no file an earlier run or another user left there is
# taken for this run's; the log's first lines name it.  Once they are copied the run records go and the log stays, as
# a host's log always has: a watcher that reads these logs finds it there (a failed copy leaves the records too, and
# names the directory).  Where no such directory can be made (a full, read-only or missing $TMPDIR), the host runs
# all the same, with no log file or run record: its peers would wait for it.
#
# Environment (PYTHON, MPMD_FLAG_SET, LIBTPU_INIT_ARGS_OVERRIDE and EXTRA_LIBTPU_INIT_ARGS as in ds_ep_run.sh):
#   RUN_DIR     where this invocation's files go (gs:// or a directory; default /tmp/mpmd_run; trailing '/'s dropped):
#               each host's log to $RUN_DIR/$STEP_TAG/logs/ (mpmd_<tag>_host<id>.log, its first line the host's id
#               and place, its second, under a step agent whose WID is not that id, a warning: below), the run
#               records (run.json) of process 0 and the stage leaders to $RUN_DIR/$STEP_TAG/json/.  '$RUN_DIR' in an
#               argument is replaced by $RUN_DIR/$STEP_TAG, this invocation's directory.
#   STEP_TAG    a name for this invocation inside RUN_DIR (default: run).
#   JAX_CACHE_DIR  the persistent compile cache (default /tmp/jax_cache).  Under jax.distributed
#               jax writes entries from process 0 only, i.e. for its own stage's programs: every
#               other host compiles its programs afresh on every run.
#   MODULE      another multi-host entry point to run the same way, in place of ds_ep_pipeline
#               (given only "$@"; no --distributed / --json).
#   TIMEOUT     seconds before the run is killed (default 3600): a host that dies
#               mid-step leaves its peers waiting on copies that never come.
#
# The host id (<id> above) is slice x hosts per slice + worker: MEGASCALE_SLICE_ID x the number of entries of
# TPU_WORKER_HOSTNAMES + TPU_WORKER_ID, so that the hosts of several slices never share one (TPU_WORKER_ID numbers the
# hosts of one slice, and repeats in every slice).
# It is the formula of jax.distributed's cluster detection, slice x len(worker list) + worker
# (jax._src.clusters.cloud_tpu_cluster), though jax may read its worker list elsewhere (TPU_PROCESS_ADDRESSES) and
# counts empty entries: on a GKE pod (one process a host) the two agree.  The id names the host's log and nothing else,
# so it may differ from jax.process_index(), the TPU runtime's numbering (TPU_WORKER_ID 7 of a v5p-128 ran as process
# 8), but no two hosts may share it: in a gs:// RUN_DIR one's log would replace the other's.  On one slice (no
# MEGASCALE_*) the id is TPU_WORKER_ID, else JOB_COMPLETION_INDEX.  Where none of these names a worker (a GCE TPU VM,
# whose environment holds none of them) the script asks the metadata server, as jax does there: the worker is
# agent-worker-number, the slice's hosts the entries of worker-network-endpoints, and MEGASCALE_* come from tpu-env
# where unset (GCE_METADATA_IP and TPU_SKIP_MDS_QUERY as for jax); without a numeric answer within 2 s (off GCE) the
# worker is 0.  MAXTEXT_MPMD_SLICE_ID, MAXTEXT_MPMD_WORKER_ID, MAXTEXT_MPMD_NUM_SLICES and
# MAXTEXT_MPMD_HOSTS_PER_SLICE, where set, take the place of MEGASCALE_SLICE_ID, TPU_WORKER_ID, MEGASCALE_NUM_SLICES
# and that number of entries, for processes that stand in for hosts without a TPU job's environment (several
# processes on one machine).  A value that is not a number stops the script before anything runs (exit 2), and so do
# a slice id past the slice count and, on several slices, a worker id past the slice's hosts or no slice id at all
# (every slice would be slice 0): the ids would pass slices x hosts, or collide.
#
# Under Pathways ('proxy' JAX_PLATFORMS's first platform, as pathways.is_proxy reads it) it starts nothing (exit 2):
# one client process drives every device there, so there is no host of a multi-controller run to be -- run that one
# client through ds_ep_run.sh.
set -uo pipefail
FIRST_PLATFORM=${JAX_PLATFORMS:-}
FIRST_PLATFORM=${FIRST_PLATFORM%%,*}
FIRST_PLATFORM=${FIRST_PLATFORM//[[:space:]]/}
if [[ ${FIRST_PLATFORM,,} == proxy ]]; then
  echo "ds_ep_multihost.sh: JAX_PLATFORMS=$JAX_PLATFORMS is Pathways: one client process drives every device, so" \
       "there is no host of a multi-controller run (--distributed) to start: run the one client through ds_ep_run.sh" >&2
  exit 2
fi
# This script's directory and the directory the package imports from, every symlink resolved, without `cd` (as in
# ds_ep_run.sh).
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="${SELF%/*}"
# The package, and ROOT, the directory it imports from: HERE one level up for each component of the package's name.
PKG=maxtext.experimental.mpmd.jax
ROOT=$HERE
for _ in ${PKG//./ }; do ROOT=${ROOT%/*}; done
ROOT=${ROOT:-/}
PYTHON=${PYTHON:-python3}
# GPU (PLATFORM=gpu, as a GPU job's entry script sets it, or JAX_PLATFORMS=cuda...): XLA_FLAGS from
# gpu.xla_flags (XLA_FLAGS_OVERRIDE replaces them) plus EXTRA_XLA_FLAGS, no LIBTPU_INIT_ARGS, and no libtpu flag set
# read.  Said, not guessed from an nvidia-smi on the host: a TPU job's step must keep its libtpu flags wherever it runs
# (ds_ep_aot_test's Launchers.test_multihost_flag_composition).
if [[ ${PLATFORM:-} == gpu || ${JAX_PLATFORMS:-} == cuda* || ${JAX_PLATFORMS:-} == gpu* ]]; then
  GPU=1
  export XLA_FLAGS="${XLA_FLAGS_OVERRIDE-$(grep -v '^#' "$HERE/gpu.xla_flags" | tr '\n' ' ')} ${EXTRA_XLA_FLAGS:-}"
else
  GPU=
  # LIBTPU_INIT_ARGS, as every launcher builds it (ds_ep_run.sh and ds_ep_aot.sh hold the same lines).
  FLAG_SET=${MPMD_FLAG_SET:-tpu7x}
  SET_FLAGS=""
  if [[ $FLAG_SET != none ]]; then
    SET_FILE="$HERE/libtpu_flags/$FLAG_SET.flags"
    if [[ $FLAG_SET == */* || ! -f $SET_FILE ]]; then
      SETS="none"
      for f in "$HERE"/libtpu_flags/*.flags; do
        if [[ -f $f ]]; then f=${f##*/}; SETS+=", ${f%.flags}"; fi
      done
      echo "ds_ep_multihost.sh: MPMD_FLAG_SET=$FLAG_SET: no such libtpu flag set ($SETS; the files of" \
           "$HERE/libtpu_flags)" >&2
      exit 1
    fi
    SET_TEXT="$(sed -e 's/#.*//' "$SET_FILE")" || exit 1     # '#' starts a comment
    read -r -d '' -a SET <<< "$SET_TEXT" || true             # one flag a word, unglobbed
    SET_FLAGS="${SET[*]-}"                                     # joined by single spaces
  fi
  MPMD_LIBTPU_INIT_ARGS="--xla_max_concurrent_async_collective_permutes=64"
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS_OVERRIDE-$SET_FLAGS} $MPMD_LIBTPU_INIT_ARGS ${EXTRA_LIBTPU_INIT_ARGS:-}"
fi
# Every process must build the same plan (the run checks); no hash randomization.
export PYTHONHASHSEED=0
export JAX_CACHE_DIR=${JAX_CACHE_DIR:-/tmp/jax_cache}
export PYTHONUNBUFFERED=1
RUN_DIR=${RUN_DIR:-/tmp/mpmd_run}
# No trailing '/': gs://bucket/runs/ would put '//' in every object name under it (the '//' of gs:// stays).
while [[ $RUN_DIR == */ && $RUN_DIR != *:// ]]; do RUN_DIR=${RUN_DIR%/}; done
TAG=${STEP_TAG:-run}

# The metadata server's value of the instance attribute $1, where a GCE TPU VM keeps what a GKE pod has in its
# environment, and jax's cluster detection reads it; status 1 under TPU_SKIP_MDS_QUERY (as for jax), without curl, or
# without an answer: no such name (off GCE), a 404 (a VM that is no TPU VM), or no answer within 2 s three times over
# (curl retries a timeout and a 408, 429 or 5xx answer, as jax retries).
metadata() {
  if [[ -n ${TPU_SKIP_MDS_QUERY+set} ]] || ! command -v curl > /dev/null; then return 1; fi
  curl -sf -m 2 --retry 2 --retry-delay 1 --noproxy '*' -H 'Metadata-Flavor: Google' \
    "http://${GCE_METADATA_IP:-metadata.google.internal}/computeMetadata/v1/instance/attributes/$1"
}

# "slice worker slices hosts_per_slice" of this host, as decimal numbers; or a message on stderr, and status 1.
host_ids() {
  local given=${MAXTEXT_MPMD_SLICE_ID:-${MEGASCALE_SLICE_ID:-}}
  local worker=${MAXTEXT_MPMD_WORKER_ID:-${TPU_WORKER_ID:-${JOB_COMPLETION_INDEX:-}}}
  local slices=${MAXTEXT_MPMD_NUM_SLICES:-${MEGASCALE_NUM_SLICES:-}}
  local hosts=${MAXTEXT_MPMD_HOSTS_PER_SLICE:-} workers=${TPU_WORKER_HOSTNAMES:-} h v
  if [[ -z $worker ]]; then
    worker=$(metadata agent-worker-number) || worker=""
    if [[ $worker =~ ^[0-9]+$ ]]; then     # a GCE TPU VM: what the environment lacks, from the metadata as well
      if [[ -z $workers ]]; then workers=$(metadata worker-network-endpoints) || workers=""; fi
      local key val env_slice="" env_slices=""
      while IFS=: read -r key val; do       # tpu-env's lines are KEY: 'value'
        val=${val//[[:space:]\']/}
        case ${key//[[:space:]]/} in
          MEGASCALE_SLICE_ID) env_slice=$val ;;
          MEGASCALE_NUM_SLICES) env_slices=$val ;;
        esac
      done <<< "$(metadata tpu-env || true)"
      given=${given:-$env_slice} slices=${slices:-$env_slices}
    else
      if [[ -n $worker ]]; then
        echo "ds_ep_multihost.sh: the metadata server's agent-worker-number is '$worker', not a number: worker 0" >&2
      fi
      worker=0
    fi
  fi
  local slice=${given:-0}
  slices=${slices:-1}
  if [[ -z $hosts ]]; then
    local -a names
    IFS=, read -ra names <<< "$workers"
    hosts=0
    for h in "${names[@]}"; do
      if [[ -n $h ]]; then hosts=$((hosts + 1)); fi
    done
    if (( hosts == 0 )); then hosts=1; fi
  fi
  for v in "$slice" "$worker" "$slices" "$hosts"; do
    if ! [[ $v =~ ^[0-9]+$ ]]; then
      echo "ds_ep_multihost.sh: not a number: '$v' (slice '$slice', worker '$worker', slices '$slices'," \
           "hosts per slice '$hosts')" >&2
      return 1
    fi
  done
  if [[ -z $given ]] && (( 10#$slices > 1 )); then
    echo "ds_ep_multihost.sh: no slice id (MEGASCALE_SLICE_ID) in a job of $slices slices: every slice would be" \
         "slice 0, and the ids of their hosts would collide" >&2
    return 1
  fi
  if (( 10#$slice >= 10#$slices )); then
    echo "ds_ep_multihost.sh: slice $slice of a job of $slices (MEGASCALE_NUM_SLICES; 1 when unset): its ids would" \
         "pass the job's $((10#$slices * 10#$hosts)) hosts" >&2
    return 1
  fi
  if (( 10#$slices > 1 && 10#$worker >= 10#$hosts )); then
    echo "ds_ep_multihost.sh: worker $worker of a slice of $hosts hosts (TPU_WORKER_HOSTNAMES, or a GCE TPU VM's" \
         "worker-network-endpoints): the ids of $slices slices would collide" >&2
    return 1
  fi
  echo "$((10#$slice)) $((10#$worker)) $((10#$slices)) $((10#$hosts))"
}
IDS=$(host_ids) || exit 2
read -r SLICE WORKER SLICES HOSTS <<< "$IDS"
ID=$((SLICE * HOSTS + WORKER))
# A step agent (a process on each host that runs a job's steps in turn, each through this script) may export WID,
# the id it took for this host, to its steps.  One that took TPU_WORKER_ID, which repeats in every slice, has the
# slices' step logs overwrite each other, and a late or restarted one skips steps the other slices' agents already
# marked done under the same id: the log's second line says so.
STALE=""
if [[ -n ${WID:-} ]] && ! { [[ $WID =~ ^[0-9]+$ ]] && (( 10#$WID == ID )); }; then
  STALE="warning: the agent's WID $WID is not this host's id $ID (slice x hosts per slice + worker): an agent that"
  STALE+=" numbers hosts by TPU_WORKER_ID -- the slices' step logs collide, and a late or restarted agent skips steps"
fi
WID=$ID
WHERE=""
if (( SLICES > 1 )); then WHERE="slice $SLICE of $SLICES, worker $WORKER of $HOSTS"; fi

cd "$ROOT"
args=()
# The replacement is quoted: unquoted, bash 5.2's patsub_replacement turns an '&' in it into the match, '$RUN_DIR'.
for x in "$@"; do args+=("${x//\$RUN_DIR/"$RUN_DIR/$TAG"}"); done
# This host's log and run records (process 0 writes $JSON, a stage leader ${JSON%.json}.p<process>.json, both at the
# end) go to a directory made for this invocation alone: a host keeps /tmp between runs (a TPU VM; an agent pod
# between steps), and a file an earlier run, or another user, left there under the same name must neither stop this
# run nor be copied as its own.  So each host's --json names a file of its own (the plan check leaves --json out).
# This is best effort: no directory, no log file or record, but the run goes on.
json=()
if WORK=$(mktemp -d "${TMPDIR:-/tmp}/mpmd.XXXXXXXX"); then
  LOG=$WORK/mpmd_${TAG}_host${WID}.log
  JSON=$WORK/mpmd_${TAG}.json
  json=(--json "$JSON")
else
  echo "ds_ep_multihost.sh: no directory under ${TMPDIR:-/tmp} for this host's log and run records: running without" \
       "them" >&2
  WORK="" LOG=/dev/null JSON=""
fi
{ echo "host $WID (${WHERE:+$WHERE, }$(hostname)): ${MODULE:-$PKG.ds_ep_pipeline --distributed} ${args[*]}"
  [[ -z $STALE ]] || echo "$STALE"; } | tee "$LOG"
if [[ -n $GPU ]]; then FLAGS_LINE="XLA_FLAGS=$XLA_FLAGS"; else FLAGS_LINE="LIBTPU_INIT_ARGS=$LIBTPU_INIT_ARGS"; fi
echo "$FLAGS_LINE" | tee -a "$LOG"
echo "log and run records: ${WORK:-none} (copied to $RUN_DIR/$TAG at the end)" | tee -a "$LOG"
if [[ -n "${MODULE:-}" ]]; then
  cmd=("$PYTHON" -u -m "$MODULE" "${args[@]}")
else
  cmd=("$PYTHON" -u -m "$PKG.ds_ep_pipeline" --distributed "${json[@]}" "${args[@]}")
fi
timeout -s KILL "${TIMEOUT:-3600}" "${cmd[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "host $WID exit $rc" | tee -a "$LOG"
# Copy this host's log and whatever run records it wrote (process 0's, a stage leader's) to RUN_DIR; then remove the
# records, unless the copy failed (no access to a gs:// RUN_DIR, say): they stay there too.  The log stays either way.
if [[ -z $WORK ]]; then
  echo "ds_ep_multihost.sh: nothing to copy to $RUN_DIR/$TAG (no directory for the log and records)" >&2
elif "$PYTHON" - "$RUN_DIR/$TAG" "$LOG" "$JSON" "${JSON%.json}".p[0-9]*.json <<'EOF'; then
import os, shutil, sys
dst, files = sys.argv[1], [f for f in sys.argv[2:] if os.path.exists(f)]
if dst.startswith('gs://'):
  from google.cloud import storage
  bucket, _, prefix = dst[5:].partition('/')
  b = storage.Client().bucket(bucket)
  for f in files:
    sub = 'logs' if f.endswith('.log') else 'json'
    b.blob(f'{prefix}/{sub}/{os.path.basename(f)}').upload_from_filename(f)
else:
  for f in files:
    sub = 'logs' if f.endswith('.log') else 'json'
    os.makedirs(f'{dst}/{sub}', exist_ok=True)
    shutil.copy(f, f'{dst}/{sub}')
print(f'uploaded {len(files)} files to {dst}')
EOF
  rm -f "$JSON" "${JSON%.json}".p[0-9]*.json
else
  echo "ds_ep_multihost.sh: copying to $RUN_DIR/$TAG failed: this host's log and run records stay in $WORK" >&2
fi
exit $rc
