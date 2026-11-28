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

# MPMD DeepSeek sparse-layer pipeline (PP x FSDP x EP) under a libtpu flag set (libtpu_flags/: by default tpu7x,
# the flags both sides of the TPU7x measurements ran), plus the one flag below.
#
#   bash src/maxtext/experimental/mpmd/jax/ds_ep_run.sh --bench --profile /tmp/xprof/mpmd [pipeline flags...]
#   MPMD_FLAG_SET=v5p bash src/maxtext/experimental/mpmd/jax/ds_ep_run.sh --bench  # MaxText's v5p DeepSeek-V3 recipe flags
#
# Environment (PYTHON, MPMD_FLAG_SET, LIBTPU_INIT_ARGS_OVERRIDE and EXTRA_LIBTPU_INIT_ARGS alike in ds_ep_aot.sh and
# ds_ep_multihost.sh; MAXTEXT_DIR in ds_ep_aot.sh too):
#   PYTHON         the interpreter (default python3).
#   MAXTEXT_DIR    if set, that checkout's src/ goes first on PYTHONPATH, so the MPMD programs call that checkout's
#                  MaxText kernels (ragged sort, megablox), and the run stops if `maxtext` resolves anywhere else.
#   MPMD_FLAG_SET  the libtpu flag set: the name of a file in libtpu_flags/ without .flags -- tpu7x (the default),
#                  v5p (MaxText's v5p DeepSeek-V3 recipe set), v5p_fsdp_extra (additions to v5p, given through
#                  EXTRA_LIBTPU_INIT_ARGS: see its header) -- or none.  A name without a file is an error, and
#                  nothing runs.
#   LIBTPU_INIT_ARGS_OVERRIDE  replaces the set's flags, even when set to "".
#   EXTRA_LIBTPU_INIT_ARGS     appended after every other flag (the last mention of a flag wins).
#   JAX_CACHE_DIR  the persistent compile cache (default ~/jax_cache).
# LIBTPU_INIT_ARGS becomes the set's flags (or LIBTPU_INIT_ARGS_OVERRIDE's), then the MPMD flag below, then
# EXTRA_LIBTPU_INIT_ARGS; a value LIBTPU_INIT_ARGS had before is not used.
set -euo pipefail
# This script's directory and the directory the package imports from, with every symlink resolved (a link to the
# script or to its directory still finds this checkout), and without `cd`, which an exported CDPATH makes print.
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
# LIBTPU_INIT_ARGS, as every launcher builds it (ds_ep_aot.sh and ds_ep_multihost.sh hold the same lines).
FLAG_SET=${MPMD_FLAG_SET:-tpu7x}
SET_FLAGS=""
if [[ $FLAG_SET != none ]]; then
  SET_FILE="$HERE/libtpu_flags/$FLAG_SET.flags"
  if [[ $FLAG_SET == */* || ! -f $SET_FILE ]]; then
    SETS="none"
    for f in "$HERE"/libtpu_flags/*.flags; do
      if [[ -f $f ]]; then f=${f##*/}; SETS+=", ${f%.flags}"; fi
    done
    echo "ds_ep_run.sh: MPMD_FLAG_SET=$FLAG_SET: no such libtpu flag set ($SETS; the files of $HERE/libtpu_flags)" >&2
    exit 1
  fi
  SET_TEXT="$(sed -e 's/#.*//' "$SET_FILE")" || exit 1     # '#' starts a comment
  read -r -d '' -a SET <<< "$SET_TEXT" || true             # one flag a word, unglobbed
  SET_FLAGS="${SET[*]-}"                                     # joined by single spaces
fi
# Plus one flag the MPMD programs need: the FSDP gather / reduce-scatter are
# collective-permutes folded into forward / backward programs, and libtpu's
# default cap on concurrent async permutes serializes them (measured: 19.4 ms
# of 22.4 ms exposed -> 8.0 ms with the cap raised).  MaxText's step time was
# unchanged by it (2.986 s/step with, 2.987 without), measured in the first
# configuration (32 experts, 8 layers) only, not re-measured at 128 experts.
MPMD_LIBTPU_INIT_ARGS="--xla_max_concurrent_async_collective_permutes=64"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS_OVERRIDE-$SET_FLAGS} $MPMD_LIBTPU_INIT_ARGS ${EXTRA_LIBTPU_INIT_ARGS:-}"
cd "$ROOT"
if [[ -n "${MAXTEXT_DIR:-}" ]]; then
  # Found, not imported (cheap), from the directory the pipeline runs in.
  "$PYTHON" - "$MAXTEXT_DIR/src" <<'EOF'
import importlib.util
import os
import sys

spec = importlib.util.find_spec('maxtext')
path = spec and (spec.origin or next(iter(spec.submodule_search_locations or ()), None))
if not path or not os.path.realpath(path).startswith(os.path.realpath(sys.argv[1]) + os.sep):
  sys.exit(f'ds_ep_run.sh: `maxtext` resolves to {path}, not under {sys.argv[1]}')
EOF
fi
exec "$PYTHON" -m "$PKG.ds_ep_pipeline" "$@"
