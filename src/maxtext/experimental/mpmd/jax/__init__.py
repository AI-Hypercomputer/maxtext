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

"""Pure-JAX MPMD pipeline: one jitted program set per pipeline stage, driven from Python."""

import os
import sys

# A module of this package run by file path (`python3 src/maxtext/experimental/mpmd/jax/ds_ep_pipeline.py`), or a
# Python started in this directory, has this directory on sys.path, where the package's modules shadow top-level ones
# of the same name: xprof.py hides the `xprof` package, which `import tokamax` imports (tokamax 0.0.17).  The
# package's modules import one another by their full names and need no such entry: take it off before any of them
# imports tokamax.
_here = globals().get("__file__")  # a loader may set none (frozen, in-memory): it has no directory on sys.path then
if _here:
  _here = os.path.dirname(os.path.realpath(_here))
  sys.path[:] = [p for p in sys.path if not (isinstance(p, str) and os.path.realpath(p or os.curdir) == _here)]
del os, sys, _here
