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

# AOT-compile the multi-host MPMD DeepSeek pipeline for a slice (ds_ep_aot.py) under
# the libtpu flags the run itself gets (ds_ep_multihost.sh / ds_ep_run.sh): they
# change what compiles -- e.g. the scoped VMEM limit the megablox tiles need.  So
# give it the run's MPMD_FLAG_SET, LIBTPU_INIT_ARGS_OVERRIDE and EXTRA_LIBTPU_INIT_ARGS.
#
#   bash src/maxtext/experimental/mpmd/jax/ds_ep_aot.sh --topology v6e-256 --stages 4 --fsdp 64 --ep 1 --layout physical ...
#
# Environment: PYTHON (default python3); MAXTEXT_DIR as in ds_ep_run.sh (that checkout's
# src/ first on PYTHONPATH, so the programs compile the MaxText kernels -- ragged sort,
# megablox -- a run with the same MAXTEXT_DIR calls; stops if `maxtext` resolves anywhere
# else); MPMD_FLAG_SET (default tpu7x), LIBTPU_INIT_ARGS_OVERRIDE / EXTRA_LIBTPU_INIT_ARGS
# as in ds_ep_run.sh, and so is LIBTPU_INIT_ARGS.
set -euo pipefail
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
if [[ -n "${MAXTEXT_DIR:-}" ]]; then
  MAXTEXT_DIR="$(CDPATH='' cd "$MAXTEXT_DIR" && pwd)"
  export PYTHONPATH="$MAXTEXT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi
# LIBTPU_INIT_ARGS, as every launcher builds it (ds_ep_run.sh and ds_ep_multihost.sh hold the same lines).
FLAG_SET=${MPMD_FLAG_SET:-tpu7x}
SET_FLAGS=""
if [[ $FLAG_SET != none ]]; then
  SET_FILE="$HERE/libtpu_flags/$FLAG_SET.flags"
  if [[ $FLAG_SET == */* || ! -f $SET_FILE ]]; then
    SETS="none"
    for f in "$HERE"/libtpu_flags/*.flags; do
      if [[ -f $f ]]; then f=${f##*/}; SETS+=", ${f%.flags}"; fi
    done
    echo "ds_ep_aot.sh: MPMD_FLAG_SET=$FLAG_SET: no such libtpu flag set ($SETS; the files of $HERE/libtpu_flags)" >&2
    exit 1
  fi
  SET_TEXT="$(sed -e 's/#.*//' "$SET_FILE")" || exit 1     # '#' starts a comment
  read -r -d '' -a SET <<< "$SET_TEXT" || true             # one flag a word, unglobbed
  SET_FLAGS="${SET[*]-}"                                     # joined by single spaces
fi
MPMD_LIBTPU_INIT_ARGS="--xla_max_concurrent_async_collective_permutes=64"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS_OVERRIDE-$SET_FLAGS} $MPMD_LIBTPU_INIT_ARGS ${EXTRA_LIBTPU_INIT_ARGS:-}"
export PYTHONHASHSEED=0
cd "$ROOT"
if [[ -n "${MAXTEXT_DIR:-}" ]]; then
  # Found, not imported (cheap), from the directory the AOT runs in.
  "$PYTHON" - "$MAXTEXT_DIR/src" <<'EOF'
import importlib.util
import os
import sys

spec = importlib.util.find_spec('maxtext')
path = spec and (spec.origin or next(iter(spec.submodule_search_locations or ()), None))
if not path or not os.path.realpath(path).startswith(os.path.realpath(sys.argv[1]) + os.sep):
  sys.exit(f'ds_ep_aot.sh: `maxtext` resolves to {path}, not under {sys.argv[1]}')
EOF
fi
exec "$PYTHON" -m "$PKG.ds_ep_aot" "$@"
