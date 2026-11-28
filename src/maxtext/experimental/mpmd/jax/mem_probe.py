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

"""Compile a ds_ep_pipeline run's programs on the devices it will run on and price its HBM, without allocating any
state: each program's memory (temporaries, arguments, outputs, aliased; largest temporaries first), then each local
stage's estimated peak -- ds_ep_aot's hbm_walk, what is live when the stage's heaviest program runs, the boundary
inputs waiting on the stage included, plus the programs' code -- against the device's memory (bytes_limit: what the
allocator was given), and what it leaves free.  ds_ep_aot does the same for a TPU slice this host does not have,
against a compile-only topology; on GPU a run's own devices compile it.  Takes ds_ep_pipeline's flags (with
--distributed, one process a host: each prices its own stages) and --headroom.  Under Pathways (JAX_PLATFORMS=proxy)
the Pathways server compiles every stage's programs and the devices report no bytes_limit (memory_stats raises
through the proxy), so the HBM is the device kind's (pathways.HBM_BYTES), or --hbm-gib's; there one client prices
every stage, and the flags only multi-controller JAX takes are refused (ds_ep_pipeline.multi_controller_only).

    python3 -m maxtext.experimental.mpmd.jax.mem_probe --model qwen3.8-2.4t-a95b --stages 2 --ep 4 ... [--distributed]
"""

import sys

from maxtext.experimental.mpmd.jax import ds_ep_pipeline as pipe
from maxtext.experimental.mpmd.jax import pathways


def stage_line(r):
  """A stage's row of ds_ep_aot.report as mem_probe prints it: the estimated peak against the device's memory (less
  the headroom asked for), whether it fits and by how much -- what it leaves free, or how far over it is -- and at
  which program it peaks, of what."""
  parts = ", ".join(
      f'{k[5:-4] if k.startswith("live_") else k[:-4]} {r[k]:.1f}'
      for k in (
          "persistent_gib",
          "live_weights_gib",
          "live_gacc_gib",
          "live_grads_gib",
          "live_residuals_gib",
          "live_inputs_gib",
          "live_temp_gib",
          "code_gib",
      )
  )
  free = r["hbm_gib"] - r["est_peak_gib"]
  room = f' less {r["headroom_gib"]:.1f} headroom' if r["headroom_gib"] else ""
  margin = f"{free:.1f} GiB free" if free >= 0 else f"{-free:.1f} GiB over"
  return (
      f'stage {r["stage"]}: est. peak {r["est_peak_gib"]:.1f} of {r["hbm_gib"]:.1f} GiB{room} '
      f'({"fits" if r["fits"] else "DOES NOT FIT"}, {margin}) at {r["peak_at"]}: {parts}'
  )


def device_hbm(devices, given_gib=None):
  """The HBM to price against, bytes: `given_gib` (--hbm-gib) if given, else the least bytes_limit of `devices` (0
  where a device keeps no statistics: CPU), or where their backend cannot report any (through the IFRT proxy
  memory_stats raises) their kind's (pathways.hbm_bytes) -> (bytes, where it came from)."""
  if given_gib is not None:
    return int(given_gib * 2**30), "--hbm-gib"
  stats = [pathways.memory_stats(d) for d in devices]
  if not any(why for _, why in stats):
    return min(st.get("bytes_limit", 0) for st, _ in stats), "bytes_limit"
  kinds = {d.device_kind for d in devices}
  table = [pathways.hbm_bytes(k) for k in kinds]
  if table and all(table):
    return min(table), f"the device kind's ({', '.join(sorted(map(str, kinds)))}: pathways.HBM_BYTES)"
  raise SystemExit(
      f"mem_probe: the devices ({', '.join(sorted(map(str, kinds)))}) report no bytes_limit and pathways.HBM_BYTES "
      "has no entry for their kind: give --hbm-gib"
  )


def main(argv=None):
  proxy = pathways.initialize()  # under JAX_PLATFORMS=proxy, before any backend starts
  ap = pipe.make_parser()
  ap.add_argument(
      "--headroom",
      type=float,
      default=0.0,
      help="share of the device's memory the estimated peak must leave free to fit (default %(default)s: "
      "it fits under the whole bytes_limit).  ds_ep_aot keeps 10%% free on TPU for what its walk "
      "leaves out there: run-ahead (the next step's first programs), which this estimate leaves out "
      "too, and the boundary inputs, which it prices",
  )
  ap.add_argument(
      "--hbm-gib",
      type=float,
      default=None,
      help="price against this much HBM per device (GiB) instead of the devices' bytes_limit (or, where they report "
      "none -- through the IFRT proxy, Pathways -- their kind's, pathways.HBM_BYTES)",
  )
  a = ap.parse_args(argv)
  if not 0 <= a.headroom < 1:
    sys.exit(f"--headroom is a share of the device memory in [0, 1), not {a.headroom}")
  if proxy and (bad := pipe.multi_controller_only(a)):
    # one client under Pathways: said before the backend starts (and before --distributed's init), in main's words --
    # relay / xfer first, pathways.check_transport's message
    raise SystemExit("\n".join(bad))
  if a.distributed:
    pipe.init_distributed()
  import jax

  p = pipe.build(a)
  # each line tagged with its process under multi-controller JAX (each prices its own stages); one controller's not
  tag = f"[p{jax.process_index()}] " if p.multi else ""
  n, dt = p.precompile()
  G = 2**30
  rows = []
  for name, c in p.compiled:
    m = c.memory_analysis()
    if m is not None:
      rows.append((m.temp_size_in_bytes, m.argument_size_in_bytes, m.output_size_in_bytes, m.alias_size_in_bytes, name))
  print(f"{tag}{n} programs compiled in {dt:.1f}s; GiB per device:", flush=True)
  for t, ar, o, al, name in sorted(rows, reverse=True):
    print(
        f"{tag}  {name:30s} temp {t / G:7.2f}  args {ar / G:7.2f}  out {o / G:7.2f}  alias {al / G:7.2f}",
        flush=True,
    )
  from maxtext.experimental.mpmd.jax import ds_ep_aot

  hbm, src = device_hbm(jax.local_devices(), a.hbm_gib)
  if src != "bytes_limit":
    print(f"{tag}HBM per device {hbm / G:.2f} GiB, {src}", flush=True)
  out = ds_ep_aot.report(p, hbm, a.headroom, inputs=True)
  for r in out:
    print(f"{tag}{stage_line(r)}", flush=True)
  return out


if __name__ == "__main__":
  main(sys.argv[1:])
