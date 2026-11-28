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

"""The MPMD pipeline's profiling defaults: one TPU, one SparseCore and one SparseCore tile a traced host, and the
hosts named in pipeline-stage order.

Every profile the pipeline takes for XProf traces the first CHIPS chips of each traced process (one TPU a slice where
one host a slice traces), with SPARSECORE_TRIM; a MaxText run gets the same trace from the config overrides
MAXTEXT_ARGS.  --profile-chips N traces more: a process holding several pipeline stages shows only its first chip's
otherwise (under Pathways, where a traced worker host holds several stages, every chip is the default: EVERY_CHIP).
XProf lays a session's hosts out sorted by name, so trace_label zero-pads the stage and process numbers.  The module
imports no other module of the package, and jax only inside profile_options.
"""

from __future__ import annotations

import argparse

# one TPU a traced process: a host's four tpu7x chips made a 1.5-1.7 GB xplane of two DeepSeek-V3 steps
CHIPS = 1
# every chip of a traced process (no tpu_num_chips_to_profile_per_task): the default under Pathways, where no process
# is a stage's own and a traced worker host holds several stages' chips
EVERY_CHIP = "all"
# one SparseCore and one tile a chip: their traces fill the device trace buffer in ~0.6 s otherwise
SPARSECORE_TRIM = {"tpu_num_sparse_cores_to_trace": 1, "tpu_num_sparse_core_tiles_to_trace": 1}
# the same trace from MaxText's profiler (they act only with profiler=xplane)
MAXTEXT_ARGS = (
    "enable_tpu_profiling_options=true",
    f"tpu_num_chips_to_profile_per_task={CHIPS}",
    *(f"{k}={v}" for k, v in SPARSECORE_TRIM.items()),
)


def padded(i, n):
  """i zero-padded to the width of n - 1 (no padding below ten)."""
  return f"{i:0{len(str(max(n - 1, 0)))}d}"


def trace_label(stage, process, n_stages, n_processes, host):
  """s<stage>-p<process>-<host>, padded; sidle for stage None (a process holding none)."""
  s = "idle" if stage is None else padded(stage, n_stages)
  return f"s{s}-p{padded(process, n_processes)}-{host}"


def positive_int(s):
  n = int(s)
  if n < 1:
    raise argparse.ArgumentTypeError(f"at least 1, not {n}")
  return n


def chip_count(s):
  """--profile-chips's type: a count of one or more, or 'all' (EVERY_CHIP)."""
  return EVERY_CHIP if s == EVERY_CHIP else positive_int(s)


# A GPU trace's CUPTI caps: jax's defaults drop the kernels of a long pipeline step (a DeepSeek-V3 stage launches
# ~10^5 a step), and an event dropped is a gap in the timeline xprof.py reads.
GPU_EVENT_CAPS = {"gpu_max_activity_api_events": 1 << 24, "gpu_max_callback_api_events": 1 << 24}
GPU_PLATFORMS = ("gpu", "cuda", "rocm")


def profile_options(chips=None, platform=None, **attrs):
  """jax.profiler.ProfileOptions for the first `chips` chips (None: CHIPS; EVERY_CHIP: all of them) of the traced
  process; `attrs` set other fields (host_tracer_level=2, session_id=...).

  TPU (platform None or 'tpu'): libtpu's tpu_num_chips_to_profile_per_task, trimmed by SPARSECORE_TRIM.  GPU
  (`platform` 'gpu' / 'cuda' / 'rocm', a device's .platform): CUPTI's own keys -- gpu_num_chips_to_profile_per_task
  and GPU_EVENT_CAPS -- and no Python tracer (it times every Python call of the host-bound MPMD driver).  The GPU
  tracer refuses any key it does not know, the TPU ones among them, and then records no device at all (jax's
  raise_error_on_start_failure is off by default), so the platform must be right."""
  chips = CHIPS if chips is None else chips
  if chips != EVERY_CHIP and chips < 1:
    raise ValueError(f"at least one chip, not {chips}")
  import jax

  o = jax.profiler.ProfileOptions()
  if platform in GPU_PLATFORMS:
    n = {} if chips == EVERY_CHIP else {"gpu_num_chips_to_profile_per_task": chips}
    o.advanced_configuration = {**n, **GPU_EVENT_CAPS}
    o.python_tracer_level = 0
  else:
    n = {} if chips == EVERY_CHIP else {"tpu_num_chips_to_profile_per_task": chips}
    o.advanced_configuration = {**n, **SPARSECORE_TRIM}
  for k, v in attrs.items():
    setattr(o, k, v)
  return o
