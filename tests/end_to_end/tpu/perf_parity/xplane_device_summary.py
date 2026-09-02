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

"""Device-side counterpart to `xplane_host_summary.py`: TPU-busy time and module launches.

The host summary skips TPU planes because it is looking for Python overhead. This one
looks at exactly those planes, because the question here is the opposite one -- how much
of the step the core was actually executing, which is the denominator for utilization.

Two lines on each core plane carry the answer:

  * `XLA Modules` -- one event per module execution. Summing its durations gives TPU-busy
                     for that core, and counting the events gives the launch count, which
                     is what separates a folded-in accumulation from an eager one. This is
                     the figure to quote, *not* xprof's Steps line: the engine's eager
                     metric dispatches split a step into hundreds of fake steps and make
                     the Steps summary meaningless.
  * `XLA Ops`     -- one event per fused op. Printed for comparison only. On a scanned
                     model it reads roughly 2x both the module time and the wall step,
                     because ops inside the scan are emitted per iteration underneath a
                     fusion that already accounts for them. Summing it overstates busy
                     time and can push utilization above 100%.

Cores run the same SPMD program in lockstep, so per-core busy time -- not the sum over
cores -- is what compares against wall clock. The mean across cores is reported, with the
spread shown so lockstep can be verified rather than assumed.

Usage: python xplane_device_summary.py [--steps N] <a.xplane.pb> [<b.xplane.pb> ...]

The argument is the `.xplane.pb` itself, not the run directory -- an arm writing to
`$PERF_PARITY_PROFILE_ROOT/<arm>` leaves it at
`<arm>/plugins/profile/<timestamp>/<host>.xplane.pb`.
"""

import collections
import sys

from xplane_host_summary import _str, event_metadata, events, lines, planes


def _line_name(line):
  """XLine.name = 2."""
  return _str(line, 2)


def summarize(path, steps=None, top=12):
  """Prints per-core TPU-busy, module launches and the costliest modules in one xplane."""
  with open(path, "rb") as fh:
    space = memoryview(fh.read())

  print(f"\n===== {path}")
  busy_per_core, launches_per_core = {}, {}
  module_totals = collections.defaultdict(lambda: [0, 0])
  for plane in planes(space):
    plane_name = _str(plane, 2)
    # Core planes only: '/device:TPU:0 (pid N)' and the like. The Python-tracing plane
    # carries 'TPU' in its name too but holds host events.
    if "device:TPU" not in plane_name or "Python" in plane_name:
      continue
    names = event_metadata(plane)
    for line in lines(plane):
      lname = _line_name(line)
      if lname == "XLA Ops":
        busy_per_core[plane_name] = sum(d for _, d in events(line)) / 1e9
      elif lname == "XLA Modules":
        evs = list(events(line))
        launches_per_core[plane_name] = len(evs)
        for meta_id, dur_ps in evs:
          agg = module_totals[names.get(meta_id, f"<{meta_id}>")]
          agg[0] += 1
          agg[1] += dur_ps

  if not busy_per_core:
    print("   no TPU core planes with an 'XLA Ops' line -- device trace is empty")
    return

  cores = sorted(busy_per_core)
  vals = [busy_per_core[c] for c in cores]
  mean_ops = sum(vals) / len(vals)
  n_launch = sum(launches_per_core.values())
  # TPU-busy is taken from XLA Modules, not XLA Ops. A module execution is unambiguously
  # "the core was running a compiled program", which is what utilization is a fraction of.
  # The XLA Ops line double-counts on a scanned model -- it reads ~2x the module time and
  # ~2x the wall step, because ops inside the scan are emitted per iteration underneath the
  # fusion that already covers them. Ops are still printed, to keep that discrepancy visible.
  mean_busy = sum(v[1] for v in module_totals.values()) / 1e9 / len(cores)
  print(f"   cores            : {len(cores)}")
  print(f"   TPU-busy / core  : mean {mean_busy:9.1f} ms  (XLA Modules)")
  # Only flagged when it actually exceeds the module time. Unscanned, the Ops line sits just
  # under it (gaps between modules are not covered by any op) and is unremarkable; scanned,
  # it runs to ~2x and the warning is the point.
  flag = "  [OVERCOUNTS -- ignore, see docstring]" if mean_ops > mean_busy else ""
  print(f"   XLA Ops / core   : mean {mean_ops:9.1f} ms   min {min(vals):9.1f}   max {max(vals):9.1f}{flag}")
  print(f"   module launches  : {n_launch} total, {n_launch // len(cores)} per core")
  if steps:
    print(
        f"   per step (n={steps:<3d}) : TPU-busy {mean_busy / steps:8.1f} ms   launches {n_launch // len(cores) / steps:.1f}"
    )
  # Per-execution cost is the figure to build a step out of. The device trace buffer holds a
  # bounded number of module events, so on a long step (GA=8 here captures 9 of 48 fwd_bwd
  # executions) the totals cover only part of the run and totals/steps understates the step.
  # A single execution's duration is unaffected by how many of them were recorded.
  print(f"   costliest modules (summed over {len(cores)} cores):")
  for name, (count, dur_ps) in sorted(module_totals.items(), key=lambda kv: -kv[1][1])[:top]:
    per_core_ms = dur_ps / 1e9 / len(cores)
    per_exec = dur_ps / 1e9 / count if count else 0.0
    print(
        f"     {per_core_ms:9.2f} ms/core  n={count:<5d} ({count // len(cores):>4d}/core)"
        f"  {per_exec:9.2f} ms/exec  {name[:66]}"
    )


if __name__ == "__main__":
  argv = sys.argv[1:]
  n_steps = None
  if argv and argv[0] == "--steps":
    n_steps = int(argv[1])
    argv = argv[2:]
  for p in argv:
    summarize(p, steps=n_steps)
