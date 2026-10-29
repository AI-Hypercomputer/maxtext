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

"""Wave clustering and straggler attribution for TPU ICI-SDC and sflag logs.

Implements:
- Approach 1 (ICI-SDC active probe): wave clustering by `D + 1` hop step,
  `S0` SEND vs. RECV classification, 1-hop torus/mesh neighbor coverage,
  `silent_at_S1` TensorCore disambiguation, and optional `(S0 + 1) % P == 0`
  cross-checking.
- Approach 2 (passive sflag wait timeout): `<no-hlo-instruction>` launch-barrier
  filtering, per-launch collective grouping, `N - 1` unsatisfied peer timeout
  complement (`all_cores - {c | latest_val < for_val}`), self-satisfied
  (`latest_val == for_val`) delay-injection detection, and multi-step
  consistency filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import re
from typing import Any

from maxtext.diagnostics.step_triage import events as events_mod


K_STEPS_FOR_DELAYS: tuple[int, ...] = (
    5,
    10,
    25,
    50,
    75,
    100,
    101,
    107,
    113,
    131,
    139,
    150,
    151,
    163,
    173,
    191,
    197,
    200,
    211,
    227,
    233,
    241,
    250,
    257,
    269,
    281,
    293,
    300,
    311,
    317,
    337,
    349,
    359,
    373,
    389,
    400,
    401,
    419,
    431,
    439,
    449,
    461,
    467,
    491,
    500,
    503,
    521,
    541,
    557,
    569,
    577,
    593,
    607,
    617,
    631,
    643,
    653,
    661,
    677,
    691,
    719,
    733,
    743,
    750,
    757,
    769,
    787,
    809,
    821,
    829,
    853,
    859,
    877,
    883,
    907,
    919,
    937,
    1000,
)


@dataclass
class CoreInfo:
  """Aggregated metadata and per-run `delay_iterations` (`P`) for one core."""

  coord: str
  chip_coord: tuple[int, int, int] | None = None
  tc_idx: int | None = None
  global_idx: int | None = None
  host: str = ""
  run_id_offset: int = 0
  delay_p_by_run: dict[int | None, int] = field(default_factory=dict)
  inferred_p_by_run: dict[int | None, int] = field(default_factory=dict)
  delay_events: list[events_mod.Event] = field(default_factory=list)

  @property
  def chip_label(self) -> str:
    if self.chip_coord is None:
      return self.coord
    return f"{self.chip_coord[0]}x{self.chip_coord[1]}x{self.chip_coord[2]}"

  def get_logged_p(self, norm_run_id: int | None = None) -> int | None:
    """Returns logged `delay_iterations` (`P`) for `norm_run_id`, if present."""
    if norm_run_id in self.delay_p_by_run:
      return self.delay_p_by_run[norm_run_id]
    has_explicit_runs = any(k is not None for k in self.delay_p_by_run)
    if norm_run_id is not None and has_explicit_runs:
      return None
    if None in self.delay_p_by_run:
      return self.delay_p_by_run[None]
    pos = [p for p in self.delay_p_by_run.values() if p > 0]
    if len(set(pos)) == 1:
      return pos[0]
    if self.delay_p_by_run:
      return next(iter(self.delay_p_by_run.values()))
    return None

  def get_effective_p(self, norm_run_id: int | None = None) -> int | None:
    """Returns logged `P`, falling back to wave-inferred `P` if unlogged."""
    logged = self.get_logged_p(norm_run_id)
    if logged is not None:
      return logged
    if norm_run_id in self.inferred_p_by_run:
      return self.inferred_p_by_run[norm_run_id]
    if None in self.inferred_p_by_run:
      return self.inferred_p_by_run[None]
    pos_inf = [p for p in self.inferred_p_by_run.values() if p > 0]
    if len(set(pos_inf)) == 1:
      return pos_inf[0]
    if self.inferred_p_by_run:
      return next(iter(self.inferred_p_by_run.values()))
    return None

  def get_delay_event(self, norm_run_id: int | None = None) -> events_mod.Event | None:
    """Returns the `DELAY_ITER` event for `norm_run_id`."""
    for ev in self.delay_events:
      if ev.norm_run_id == norm_run_id:
        return ev
    return self.delay_events[0] if self.delay_events else None


@dataclass
class TopologyModel:
  """3D chip grid and per-core 1-hop neighbor adjacency model."""

  bounds: tuple[int, int, int]
  cores_per_chip: int
  max_distance: int
  wrap_mode: str
  wrap_axes: tuple[bool, bool, bool]
  all_cores: list[str] = field(default_factory=list)
  neighbors_torus: dict[str, list[str]] = field(default_factory=dict)
  neighbors_mesh: dict[str, list[str]] = field(default_factory=dict)
  chip_to_cores: dict[tuple[int, int, int], list[str]] = field(default_factory=dict)


def parse_topology_str(val: str | None) -> tuple[int, int, int] | None:
  """Parses `WxH` or `WxHxZ` topology string into a 3-tuple of ints."""
  if not val:
    return None
  parts = [int(x) for x in re.split(r"[xX,]", val.strip()) if x.strip()]
  if len(parts) == 2:
    return (parts[0], parts[1], 1)
  if len(parts) == 3:
    return (parts[0], parts[1], parts[2])
  raise ValueError(f"Invalid topology '{val}', expected WxH or WxHxZ")


def chip_manhattan_distance(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    bounds: tuple[int, int, int],
    wrap_axes: tuple[bool, bool, bool],
) -> int:
  """Computes 3D Manhattan distance between chips `a` and `b`."""
  dist = 0
  for i in range(3):
    diff = abs(a[i] - b[i])
    if wrap_axes[i] and bounds[i] > 1:
      diff = min(diff, bounds[i] - diff)
    dist += diff
  return dist


def build_core_catalog(parsed: events_mod.ParsedLogSet) -> dict[str, CoreInfo]:
  """Builds a map of `coord -> CoreInfo` from all events in `parsed`."""
  cores: dict[str, CoreInfo] = {}

  def _ensure_core(coord: str) -> CoreInfo:
    if coord not in cores:
      chip, tc = events_mod.parse_coord_parts(coord)
      cores[coord] = CoreInfo(
          coord=coord,
          chip_coord=chip,
          tc_idx=tc,
          global_idx=parsed.core_to_global.get(coord),
          host=parsed.core_to_host.get(coord, ""),
          run_id_offset=parsed.run_id_offsets.get(coord, 0),
      )
    return cores[coord]

  for c in parsed.core_to_global:
    _ensure_core(c)
  for c in parsed.core_to_host:
    _ensure_core(c)

  for ev in parsed.events:
    for c_cand in (ev.coord, ev.src_coord, ev.dst_coord):
      if c_cand and not c_cand.startswith("core_"):
        ci = _ensure_core(c_cand)
        if c_cand == ev.coord and ev.host and not ci.host:
          ci.host = ev.host
        if c_cand == ev.coord and ev.global_idx is not None and ci.global_idx is None:
          ci.global_idx = ev.global_idx
    if ev.kind == "DELAY_ITER" and ev.coord:
      ci = _ensure_core(ev.coord)
      if ev.delay_p is not None:
        ci.delay_p_by_run[ev.norm_run_id] = ev.delay_p
      ci.delay_events.append(ev)

  return cores


def build_topology_model(
    parsed: events_mod.ParsedLogSet,
    cores: dict[str, CoreInfo],
    topology_override: tuple[int, int, int] | None = None,
    cores_per_chip_override: int | None = None,
    max_distance: int = 1,
    wrap_mode: str = "auto",
) -> TopologyModel | None:
  """Constructs the 3D chip grid and 1-hop neighbor sets for all cores."""
  coord_cores = [ci for ci in cores.values() if ci.chip_coord is not None]

  if not coord_cores and topology_override is None:
    max_g = max(
        (g for ev in parsed.events for g in (ev.global_idx, ev.src_global_idx, ev.dst_global_idx) if g is not None),
        default=-1,
    )
    if max_g >= 0:
      if max_g < 8:
        topology_override = (2, 2, 1)
        cores_per_chip_override = cores_per_chip_override or 2
      elif max_g < 16:
        topology_override = (2, 2, 2)
        cores_per_chip_override = cores_per_chip_override or 2
      elif max_g < 128:
        topology_override = (4, 4, 4)
        cores_per_chip_override = cores_per_chip_override or 2
      elif max_g < 256:
        topology_override = (4, 4, 8)
        cores_per_chip_override = cores_per_chip_override or 2
    if topology_override is None:
      return None

  if topology_override is not None:
    w, h, z = topology_override
  elif "TPU_TOPOLOGY" in parsed.env_vars:
    try:
      w, h, z = parse_topology_str(parsed.env_vars["TPU_TOPOLOGY"]) or (1, 1, 1)
    except ValueError:
      w = max(ci.chip_coord[0] for ci in coord_cores if ci.chip_coord) + 1
      h = max(ci.chip_coord[1] for ci in coord_cores if ci.chip_coord) + 1
      z = max(ci.chip_coord[2] for ci in coord_cores if ci.chip_coord) + 1
  else:
    w = max(ci.chip_coord[0] for ci in coord_cores if ci.chip_coord) + 1
    h = max(ci.chip_coord[1] for ci in coord_cores if ci.chip_coord) + 1
    z = max(ci.chip_coord[2] for ci in coord_cores if ci.chip_coord) + 1

  if cores_per_chip_override is not None:
    cpc = cores_per_chip_override
  elif coord_cores:
    cpc = max((ci.tc_idx or 0) for ci in coord_cores) + 1
  else:
    cpc = 2

  bounds = (w, h, z)

  eff_wrap_mode = wrap_mode
  env_wrap_axes: tuple[bool, bool, bool] | None = None
  if wrap_mode == "auto" and "TPU_TOPOLOGY_WRAP" in parsed.env_vars:
    parts = [p.strip().lower() == "true" for p in parsed.env_vars["TPU_TOPOLOGY_WRAP"].split(",") if p.strip()]
    if len(parts) == 3:
      env_wrap_axes = (parts[0], parts[1], parts[2])
      if not any(parts):
        eff_wrap_mode = "false"
      elif all(parts):
        eff_wrap_mode = "true"

  chip_to_cores: dict[tuple[int, int, int], list[str]] = {}
  all_cores: list[str] = []
  sample_coords = [ci.coord for ci in coord_cores]
  is_2d = bool(sample_coords) and all(c.startswith("(") for c in sample_coords)
  has_tc = not sample_coords or any("_TC" in c for c in sample_coords)

  for gz, gy, gx in itertools.product(range(z), range(h), range(w)):
    chip = (gx, gy, gz)
    chip_to_cores[chip] = []
    for tc in range(cpc):
      cname = events_mod.format_coord(gx, gy, gz, tc, is_2d_format=is_2d, has_tc_suffix=has_tc)
      g_formula = (gx + w * (gy + h * gz)) * cpc + tc
      if cname not in cores:
        cores[cname] = CoreInfo(
            coord=cname,
            chip_coord=chip,
            tc_idx=tc,
            global_idx=parsed.core_to_global.get(cname, g_formula),
            host=parsed.core_to_host.get(cname, ""),
            run_id_offset=parsed.run_id_offsets.get(cname, 0),
        )
      elif cores[cname].global_idx is None:
        cores[cname].global_idx = g_formula
      parsed.global_to_core.setdefault(g_formula, cname)
      parsed.core_to_global.setdefault(cname, cores[cname].global_idx or g_formula)
      chip_to_cores[chip].append(cname)
      all_cores.append(cname)

  # Resolve any compact-code `core_<G>` placeholders now that grid bijection is complete.
  for ev in parsed.events:
    if ev.kind == "XFER_TIMEOUT_COMPACT":
      if ev.coord and ev.coord.startswith("core_") and ev.global_idx in parsed.global_to_core:
        ev.coord = parsed.global_to_core[ev.global_idx]
        ev.chip_coord, ev.tc_idx = events_mod.parse_coord_parts(ev.coord)
      if ev.src_coord and ev.src_coord.startswith("core_") and ev.src_global_idx in parsed.global_to_core:
        ev.src_coord = parsed.global_to_core[ev.src_global_idx]
      if ev.dst_coord and ev.dst_coord.startswith("core_") and ev.dst_global_idx in parsed.global_to_core:
        ev.dst_coord = parsed.global_to_core[ev.dst_global_idx]

  if env_wrap_axes is not None:
    wrap_torus = env_wrap_axes
  else:
    wrap_torus = (w >= 3, h >= 3, z >= 3)
  wrap_mesh = (False, False, False)

  neighbors_torus: dict[str, list[str]] = {c: [] for c in cores}
  neighbors_mesh: dict[str, list[str]] = {c: [] for c in cores}
  all_chips = list(chip_to_cores.keys())

  for i, chip_a in enumerate(all_chips):
    for j in range(i + 1, len(all_chips)):
      chip_b = all_chips[j]
      d_torus = chip_manhattan_distance(chip_a, chip_b, bounds, wrap_torus)
      d_mesh = chip_manhattan_distance(chip_a, chip_b, bounds, wrap_mesh)
      if 0 < d_torus <= max_distance:
        for ca in chip_to_cores[chip_a]:
          for cb in chip_to_cores[chip_b]:
            neighbors_torus[ca].append(cb)
            neighbors_torus[cb].append(ca)
      if 0 < d_mesh <= max_distance:
        for ca in chip_to_cores[chip_a]:
          for cb in chip_to_cores[chip_b]:
            neighbors_mesh[ca].append(cb)
            neighbors_mesh[cb].append(ca)

  for c in neighbors_torus:
    neighbors_torus[c].sort()
    neighbors_mesh[c].sort()

  return TopologyModel(
      bounds=bounds,
      cores_per_chip=cpc,
      max_distance=max_distance,
      wrap_mode=eff_wrap_mode,
      wrap_axes=wrap_torus if eff_wrap_mode != "false" else wrap_mesh,
      all_cores=sorted(all_cores),
      neighbors_torus=neighbors_torus,
      neighbors_mesh=neighbors_mesh,
      chip_to_cores=chip_to_cores,
  )


# ---------------------------------------------------------------------------
# Approach 1: ICI-SDC Wave Clustering & Attribution
# ---------------------------------------------------------------------------


def _cores_adjacent_or_same_chip(
    cores_prev: set[str],
    cores_curr: set[str],
    cores: dict[str, CoreInfo],
    topo: TopologyModel | None,
) -> bool:
  """Returns True if `cores_curr` touches or shares a chip with `cores_prev`."""
  if topo is None:
    return True
  nbrs = topo.neighbors_mesh if topo.wrap_mode == "false" else topo.neighbors_torus
  for u in cores_prev:
    u_chip = cores[u].chip_coord if u in cores else None
    for v in nbrs.get(u, []):
      if v in cores_curr:
        return True
    if u_chip is not None:
      for v in topo.chip_to_cores.get(u_chip, []):
        if v in cores_curr:
          return True
  return False


def _is_clear_new_wave_origin(
    events_at_w: list[events_mod.Event],
    cores: dict[str, CoreInfo],
    topo: TopologyModel | None,
) -> bool:
  """Checks blindly (without P) if `events_at_w` starts a fresh wave."""
  if any(ev.direction == "SEND" for ev in events_at_w):
    return True
  if topo is None:
    return False
  recv_cores = {ev.coord for ev in events_at_w if ev.direction == "RECV" and ev.coord}
  if len(recv_cores) < 2:
    return False
  nbr_map = topo.neighbors_mesh if topo.wrap_mode == "false" else topo.neighbors_torus
  for c_name in cores:
    if c_name in recv_cores:
      continue
    nbrs = nbr_map.get(c_name, [])
    if nbrs and len(nbrs) >= 2 and all(n in recv_cores for n in nbrs):
      return True
  return False


def _find_matching_wave(
    w: int,
    evs_w: list[events_mod.Event],
    cores_w: set[str],
    active_waves: list[dict[str, Any]],
    step: int,
    cores: dict[str, CoreInfo],
    topo: TopologyModel | None,
) -> dict[str, Any] | None:
  """Finds an existing wave in `active_waves` that `evs_w` continues."""
  for cand in reversed(active_waves):
    if w - cand["last_hop_w"] == step and cores_w != cand["last_cores"]:
      if _cores_adjacent_or_same_chip(cand["last_cores"], cores_w, cores, topo):
        return cand

  if not _is_clear_new_wave_origin(evs_w, cores, topo):
    for cand in reversed(active_waves):
      if 0 < w - cand["last_hop_w"] <= 2:
        if (cores_w & cand["last_cores"]) or _cores_adjacent_or_same_chip(cand["last_cores"], cores_w, cores, topo):
          return cand
  return None


def cluster_xfer_events_into_waves(
    xfer_events: list[events_mod.Event],
    pipeline_depth: int,
    cores: dict[str, CoreInfo],
    topo: TopologyModel | None,
) -> list[list[events_mod.Event]]:
  """Clusters transfer timeout events into waves separated by `norm_run_id` and `D + 1`."""
  if not xfer_events:
    return []

  step = pipeline_depth + 1
  by_run: dict[int | None, list[events_mod.Event]] = {}
  for ev in xfer_events:
    by_run.setdefault(ev.norm_run_id, []).append(ev)

  all_waves: list[list[events_mod.Event]] = []
  for run_id in sorted(by_run.keys(), key=lambda x: (-1 if x is None else x)):
    run_evs = sorted(
        by_run[run_id],
        key=lambda e: (
            e.iteration or 0,
            0 if e.direction == "SEND" else 1,
            e.coord or "",
        ),
    )
    by_iter: dict[int, list[events_mod.Event]] = {}
    for ev in run_evs:
      if ev.iteration is not None:
        by_iter.setdefault(ev.iteration, []).append(ev)

    active_waves: list[dict[str, Any]] = []
    for w in sorted(by_iter.keys()):
      evs_w = by_iter[w]
      cores_w = {ev.coord for ev in evs_w if ev.coord}
      best_wave = _find_matching_wave(w, evs_w, cores_w, active_waves, step, cores, topo)

      if best_wave is not None:
        if w - best_wave["last_hop_w"] >= step - 1:
          best_wave["last_hop_w"] = w
          best_wave["last_cores"] = cores_w
        else:
          best_wave["last_cores"] = best_wave["last_cores"] | cores_w
        best_wave["seen_cores"].update(cores_w)
        best_wave["events"].extend(evs_w)
      else:
        active_waves.append(
            {
                "s0": w,
                "last_hop_w": w,
                "last_cores": set(cores_w),
                "seen_cores": set(cores_w),
                "events": list(evs_w),
            }
        )

    for wv in active_waves:
      all_waves.append(wv["events"])

  return all_waves


def infer_missing_p_from_waves(
    raw_waves: list[list[events_mod.Event]],
    pipeline_depth: int,
    cores: dict[str, CoreInfo],
    topo: TopologyModel | None,
) -> None:
  """Infers `P` from clean periodic S0 waves when `DELAY_ITER` lines were not flushed."""
  if topo is None:
    return
  step = pipeline_depth + 1
  nbr_map = topo.neighbors_mesh if topo.wrap_mode == "false" else topo.neighbors_torus
  waves_by_run: dict[int | None, list[list[events_mod.Event]]] = {}
  for w in raw_waves:
    if w:
      waves_by_run.setdefault(w[0].norm_run_id, []).append(w)

  for run_id, run_waves in waves_by_run.items():
    missing_cores = [c for c, ci in cores.items() if ci.get_logged_p(run_id) is None]
    if not missing_cores:
      continue

    core_s0s: dict[str, list[int]] = {c: [] for c in missing_cores}
    for w_evs in run_waves:
      s0 = min(ev.iteration for ev in w_evs if ev.iteration is not None)
      if any(ev.direction == "SEND" for ev in w_evs if ev.iteration == s0):
        continue
      s0_c = {ev.coord for ev in w_evs if ev.iteration == s0 and ev.direction == "RECV" and ev.coord}
      s1_c = {ev.coord for ev in w_evs if ev.iteration == s0 + step and ev.coord}
      cands = [
          c
          for c in missing_cores
          if c not in s0_c and nbr_map.get(c) and all(n in s0_c for n in nbr_map[c]) and (not s1_c or c not in s1_c)
      ]
      if len(cands) == 1:
        core_s0s[cands[0]].append(s0)

    for c, s0s in core_s0s.items():
      if len(s0s) < 2:
        continue
      matches = [
          pr for pr in K_STEPS_FOR_DELAYS if sum(1 for s in s0s if (s + 1) % pr == 0) >= max(2, int(len(s0s) * 0.8))
      ]
      if len(matches) == 1:
        cores[c].inferred_p_by_run[run_id] = matches[0]


def _attribute_bad_links(
    send_s0_events: list[events_mod.Event],
    send_s0_cores: list[str],
    all_s0_set: set[str],
    s1_cores_set: set[str],
    cores: dict[str, CoreInfo],
    topo: TopologyModel | None,
) -> list[dict[str, Any]]:
  """Attributes `SUSPECTED_BAD_LINK` endpoints from `S0` `SEND` timeouts."""
  suspected_links: list[dict[str, Any]] = []
  seen_core_links: set[tuple[str, str]] = set()
  send_s0_set = set(send_s0_cores)

  explicit_pairs: dict[str, list[str]] = {}
  for ev in send_s0_events:
    if ev.src_coord and ev.dst_coord and ev.src_coord != ev.dst_coord:
      lst = explicit_pairs.setdefault(ev.src_coord, [])
      if ev.dst_coord not in lst:
        lst.append(ev.dst_coord)

  for a in send_s0_cores:
    ci_a = cores.get(a)
    chip_a = ci_a.chip_label if ci_a else a
    host_a = ci_a.host if ci_a else ""
    nbrs_a = set(topo.neighbors_torus.get(a, [])) | set(topo.neighbors_mesh.get(a, [])) if topo else (all_s0_set - {a})
    matched_b = explicit_pairs.get(a, [])
    if not matched_b:
      matched_b = [b for b in sorted(all_s0_set) if b != a and b in nbrs_a]
    if not matched_b and topo is not None:
      matched_b = [b for b in sorted(s1_cores_set) if b != a and b in nbrs_a]

    if not matched_b:
      suspected_links.append(
          {
              "src_core": a,
              "dst_core": "unknown",
              "src_chip": chip_a,
              "dst_chip": "unknown",
              "src_host": host_a,
              "dst_host": "",
              "bidirectional": False,
              "link_label": f"{a} -> <outbound>",
              "chip_link_label": f"{chip_a} -> <outbound>",
          }
      )
      continue

    for b in matched_b:
      if (a, b) in seen_core_links:
        continue
      seen_core_links.add((a, b))
      ci_b = cores.get(b)
      chip_b = ci_b.chip_label if ci_b else b
      host_b = ci_b.host if ci_b else ""
      if chip_a == chip_b:
        continue
      bidir = b in send_s0_set and (a in explicit_pairs.get(b, []) or a in nbrs_a)
      suspected_links.append(
          {
              "src_core": a,
              "dst_core": b,
              "src_chip": chip_a,
              "dst_chip": chip_b,
              "src_host": host_a,
              "dst_host": host_b,
              "bidirectional": bidir,
              "link_label": f"{a} <-> {b}" if bidir else f"{a} -> {b}",
              "chip_link_label": f"{chip_a} <-> {chip_b}" if bidir else f"{chip_a} -> {chip_b}",
          }
      )
  return suspected_links


def _best_neighbor_coverage(
    c_name: str,
    recv_s0_set: set[str],
    topo: TopologyModel | None,
) -> tuple[float, list[str], list[str], str]:
  """Returns `(best_cov, best_nbrs, best_hit_nbrs, best_mode)` for `c_name`."""
  best_cov = 0.0
  best_nbrs: list[str] = []
  best_hit_nbrs: list[str] = []
  best_mode = "torus"
  if topo is None:
    return best_cov, best_nbrs, best_hit_nbrs, best_mode

  modes_to_try = []
  if topo.wrap_mode in ("auto", "true"):
    modes_to_try.append(("torus", topo.neighbors_torus))
  if topo.wrap_mode in ("auto", "false"):
    modes_to_try.append(("mesh", topo.neighbors_mesh))
  for m_name, nbr_map in modes_to_try:
    nbrs = nbr_map.get(c_name, [])
    if not nbrs:
      continue
    hits = [n for n in nbrs if n in recv_s0_set]
    cov = len(hits) / len(nbrs)
    if cov > best_cov or (cov == best_cov and len(hits) > len(best_hit_nbrs)):
      best_cov = cov
      best_nbrs = nbrs
      best_hit_nbrs = hits
      best_mode = m_name
  return best_cov, best_nbrs, best_hit_nbrs, best_mode


def analyze_single_wave(
    wave_idx: int,
    wave_events: list[events_mod.Event],
    pipeline_depth: int,
    cores: dict[str, CoreInfo],
    topo: TopologyModel | None,
) -> dict[str, Any]:
  """Analyzes one wave and returns S0 stats, hop histogram, and culprit attribution."""
  step = pipeline_depth + 1
  s0 = min(ev.iteration for ev in wave_events if ev.iteration is not None)
  run_id = wave_events[0].norm_run_id

  first_by_core: dict[str, events_mod.Event] = {}
  for ev in sorted(
      wave_events,
      key=lambda x: (x.iteration or 0, 0 if x.direction == "SEND" else 1, x.coord or ""),
  ):
    if ev.coord and ev.coord not in first_by_core:
      first_by_core[ev.coord] = ev

  s0_events = [ev for ev in wave_events if ev.iteration == s0]
  send_s0_events = [ev for ev in s0_events if ev.direction == "SEND"]
  recv_s0_events = [ev for ev in s0_events if ev.direction == "RECV"]

  send_s0_cores = sorted({ev.coord for ev in send_s0_events if ev.coord})
  recv_s0_cores = sorted({ev.coord for ev in recv_s0_events if ev.coord})
  send_s0_set = set(send_s0_cores)
  recv_s0_set = set(recv_s0_cores)
  all_s0_set = send_s0_set | recv_s0_set

  hop_counts: dict[int, int] = {}
  hop_cores: dict[int, list[str]] = {}
  hop_iterations: dict[int, list[int]] = {}
  exact_step_matches = 0
  for c, ev in sorted(first_by_core.items()):
    delta = (ev.iteration or s0) - s0
    h = round(delta / step)
    if delta == h * step:
      exact_step_matches += 1
    hop_counts[h] = hop_counts.get(h, 0) + 1
    hop_cores.setdefault(h, []).append(c)
    if (ev.iteration or s0) not in hop_iterations.setdefault(h, []):
      hop_iterations[h].append(ev.iteration or s0)

  hop_histogram = {h: hop_counts[h] for h in sorted(hop_counts)}
  hop_histogram_labeled = {f"S{h}": hop_counts[h] for h in sorted(hop_counts)}
  s1_cores_set = set(hop_cores.get(1, []))

  verdict = "NOISE"
  suspected_links: list[dict[str, Any]] = []
  suspected_slow_cores: list[dict[str, Any]] = []
  blind_suspected_slow_cores: list[dict[str, Any]] = []

  if send_s0_cores:
    verdict = "SUSPECTED_BAD_LINK"
    suspected_links = _attribute_bad_links(
        send_s0_events=send_s0_events,
        send_s0_cores=send_s0_cores,
        all_s0_set=all_s0_set,
        s1_cores_set=s1_cores_set,
        cores=cores,
        topo=topo,
    )
  else:
    candidates: list[dict[str, Any]] = []
    blind_candidates: list[dict[str, Any]] = []
    for c_name, ci in sorted(cores.items()):
      if c_name in recv_s0_set:
        continue
      logged_p = ci.get_logged_p(run_id)
      effective_p = ci.get_effective_p(run_id)
      p_matches = (s0 + 1) % effective_p == 0 if (effective_p is not None and effective_p > 0) else None
      logged_p_matches = (s0 + 1) % logged_p == 0 if (logged_p is not None and logged_p > 0) else None
      silent_at_s1 = c_name not in s1_cores_set

      best_cov, best_nbrs, best_hit_nbrs, best_mode = _best_neighbor_coverage(c_name, recv_s0_set, topo)

      blind_qualifies = False
      if topo is not None and best_nbrs:
        if best_cov == 1.0 and len(best_nbrs) >= 2:
          blind_qualifies = True
        elif best_cov >= 0.75 and len(best_hit_nbrs) >= 3 and recv_s0_set.issubset(set(best_nbrs)):
          blind_qualifies = True

      qualifies = False
      if topo is not None and best_nbrs:
        same_chip_cores = topo.chip_to_cores.get(ci.chip_coord, []) if ci.chip_coord else []
        other_tc_matches = any(
            (cores[oc].get_effective_p(run_id) or 0) > 0 and (s0 + 1) % (cores[oc].get_effective_p(run_id) or 1) == 0
            for oc in same_chip_cores
            if oc != c_name and oc in cores
        )
        if (
            best_cov == 1.0
            and len(best_nbrs) >= 2
            and (p_matches is not False or (not other_tc_matches and silent_at_s1))
        ):
          qualifies = True
        elif best_cov >= 0.5 and len(best_hit_nbrs) >= 2 and p_matches is True:
          qualifies = True
        elif best_cov >= 0.75 and len(best_hit_nbrs) >= 3 and p_matches is None and recv_s0_set.issubset(set(best_nbrs)):
          qualifies = True
      elif topo is None and p_matches is True and len(recv_s0_set) >= 2:
        qualifies = True

      if qualifies or blind_qualifies:
        delay_ev = ci.get_delay_event(run_id)
        cand_obj = {
            "core": c_name,
            "chip": ci.chip_label,
            "coords": list(ci.chip_coord) if ci.chip_coord else None,
            "tc_index": ci.tc_idx,
            "global_core_index": ci.global_idx,
            "host": ci.host,
            "topology_mode": best_mode,
            "silent_at_s0": True,
            "silent_at_s1": silent_at_s1,
            "recv_neighbor_fraction": f"{len(best_hit_nbrs)}/{len(best_nbrs)}",
            "neighbor_recv_s0_cores": best_hit_nbrs,
            "neighbor_all_cores": best_nbrs,
            "neighbor_recv_s0_count": len(best_hit_nbrs),
            "neighbor_total_count": len(best_nbrs),
            "neighbor_coverage": round(best_cov, 4),
            "logged_p": logged_p,
            "logged_p_divides_s0_plus_1": logged_p_matches,
            "inferred_p": ci.inferred_p_by_run.get(run_id),
            "effective_p": effective_p,
            "p_divides_s0_plus_1": p_matches,
            "bw_log_ref": f"{delay_ev.file}:{delay_ev.line_no}" if delay_ev else None,
        }
        if qualifies:
          candidates.append(cand_obj)
        if blind_qualifies:
          blind_candidates.append(cand_obj)

    if s1_cores_set and any(c["silent_at_s1"] for c in blind_candidates):
      blind_candidates = [c for c in blind_candidates if c["silent_at_s1"]]
    blind_candidates.sort(
        key=lambda x: (
            1 if x["silent_at_s1"] else 0,
            x["neighbor_coverage"],
            x["neighbor_recv_s0_count"],
        ),
        reverse=True,
    )
    blind_suspected_slow_cores = blind_candidates

    if any(c["p_divides_s0_plus_1"] is True for c in candidates):
      candidates = [c for c in candidates if c["p_divides_s0_plus_1"] is True]
    if s1_cores_set and any(c["silent_at_s1"] for c in candidates):
      candidates = [c for c in candidates if c["silent_at_s1"]]
    candidates.sort(
        key=lambda x: (
            1 if x["p_divides_s0_plus_1"] is True else 0,
            1 if x["silent_at_s1"] else 0,
            x["neighbor_coverage"],
            x["neighbor_recv_s0_count"],
        ),
        reverse=True,
    )
    if candidates:
      verdict = "SUSPECTED_SLOW_CORE"
      suspected_slow_cores = candidates

  return {
      "wave_id": wave_idx,
      "run_id": run_id,
      "s0_iteration": s0,
      "pipeline_depth": pipeline_depth,
      "expected_hop_step": step,
      "total_events": len(wave_events),
      "affected_cores_count": len(first_by_core),
      "s0_send_cores": send_s0_cores,
      "s0_recv_cores": recv_s0_cores,
      "hop_histogram": hop_histogram,
      "hop_histogram_labeled": hop_histogram_labeled,
      "hop_cores": {f"S{h}": hop_cores[h] for h in sorted(hop_cores)},
      "hop_iterations": {f"S{h}": hop_iterations[h] for h in sorted(hop_iterations)},
      "exact_hop_step_matches": exact_step_matches,
      "verdict": verdict,
      "suspected_links": suspected_links,
      "suspected_slow_cores": suspected_slow_cores,
      "blind_suspected_slow_cores": blind_suspected_slow_cores,
      "blind_top_core": (blind_suspected_slow_cores[0]["core"] if blind_suspected_slow_cores else None),
      "p_checked_top_core": (suspected_slow_cores[0]["core"] if suspected_slow_cores else None),
  }


# ---------------------------------------------------------------------------
# Approach 2: Passive Collective sflag Wait-Timeout Attribution
# ---------------------------------------------------------------------------


def analyze_sflag_timeouts(
    sflag_events: list[events_mod.Event],
    cores: dict[str, CoreInfo],
    topo: TopologyModel | None,
) -> dict[str, Any]:
  """Analyzes Approach-2 passive sflag wait timeout events.

  Implements the 3 attribution rules from Section 5.2 of the design spec:
  1. Filter `<no-hlo-instruction>` launch-barrier noise from collective HLOs.
  2. Compute the silent complement `all_cores - {c | latest_val < for_val}` on
     the earliest collective HLO per launch/step.
  3. Detect self-satisfied timeouts (`latest_val >= for_val`) emitted by the
     delay-injected core itself, and require step consistency across launches.
  """
  raw_count = len(sflag_events)
  if raw_count == 0:
    return {
        "raw_event_count": 0,
        "unique_event_count": 0,
        "launch_barrier_count": 0,
        "collective_event_count": 0,
        "collective_launch_count": 0,
        "verdict": "CLEAN",
        "suspected_straggler_cores": [],
        "self_satisfied_cores": [],
        "per_launch_collectives": [],
        "by_hlo": {},
    }

  # Deduplicate identical timeout lines across log files.
  dedup_map: dict[tuple[Any, ...], events_mod.Event] = {}
  for ev in sflag_events:
    step_k = ev.launch_id if ev.launch_id else f"run_{ev.norm_run_id}"
    key = (ev.coord, step_k, ev.alloc_label, ev.hlo_op, ev.for_val, ev.latest_val)
    if key not in dedup_map:
      dedup_map[key] = ev
  uniq_events = list(dedup_map.values())

  all_grid_cores = set(topo.all_cores) if topo and topo.all_cores else set(cores.keys())
  total_cores = len(all_grid_cores)

  launch_barrier_events = [ev for ev in uniq_events if not ev.hlo_op or ev.hlo_op == "<no-hlo-instruction>"]
  collective_events = [ev for ev in uniq_events if ev.hlo_op and ev.hlo_op != "<no-hlo-instruction>"]

  # Group collective events by launch/step key, preserving appearance order.
  by_launch: dict[str, list[events_mod.Event]] = {}
  for ev in collective_events:
    step_k = f"launch:{ev.launch_id}" if ev.launch_id else f"run:{ev.norm_run_id}"
    by_launch.setdefault(step_k, []).append(ev)

  per_launch_collectives: list[dict[str, Any]] = []
  by_hlo_summary: dict[str, dict[str, Any]] = {}
  self_satisfied_counts: dict[str, int] = {}

  # Track per-HLO frequency across launches so we can distinguish single-step
  # startup `device_put` programs from multi-step training `jit_step` launches.
  hlo_launch_freq: dict[str, int] = {}
  for step_k, step_evs in by_launch.items():
    seen_hlos_in_step = {ev.hlo_op for ev in step_evs if ev.hlo_op}
    for h in seen_hlos_in_step:
      hlo_launch_freq[h] = hlo_launch_freq.get(h, 0) + 1

  max_hlo_freq = max(hlo_launch_freq.values(), default=0)

  for step_k, step_evs in by_launch.items():
    # Group within this step by hlo_op in order of first appearance.
    hlo_groups: dict[str, list[events_mod.Event]] = {}
    for ev in step_evs:
      if ev.hlo_op:
        hlo_groups.setdefault(ev.hlo_op, []).append(ev)

    step_records: list[dict[str, Any]] = []
    for hlo_idx, (hlo_op, h_evs) in enumerate(hlo_groups.items()):
      unsat_cores = sorted(
          {
              ev.coord
              for ev in h_evs
              if ev.coord and ev.for_val is not None and ev.latest_val is not None and ev.latest_val < ev.for_val
          }
      )
      self_sat_cores = sorted(
          {
              ev.coord
              for ev in h_evs
              if ev.coord and ev.for_val is not None and ev.latest_val is not None and ev.latest_val >= ev.for_val
          }
      )
      for sc in self_sat_cores:
        self_satisfied_counts[sc] = self_satisfied_counts.get(sc, 0) + 1

      complement = sorted(all_grid_cores - set(unsat_cores)) if all_grid_cores else []
      # Valid peer-waiting straggler signature requires N - 1 (single slow core)
      # or N - 2 (slow dual-core chip) of the N cores to time out with
      # `latest_val < for_val`, leaving a 1-to-2 core silent complement.
      max_comp = 2 if total_cores >= 4 else 1
      is_valid_complement = (
          total_cores >= 2 and len(unsat_cores) >= max(2, total_cores - max_comp) and 1 <= len(complement) <= max_comp
      )
      is_startup_only = max_hlo_freq >= 3 and hlo_launch_freq.get(hlo_op, 0) == 1

      sample_ev = h_evs[0]
      rec = {
          "launch_key": step_k,
          "hlo_op": hlo_op,
          "hlo_order_in_launch": hlo_idx,
          "alloc_label": sample_ev.alloc_label,
          "unsatisfied_count": len(unsat_cores),
          "total_cores": total_cores,
          "silent_complement": complement,
          "self_satisfied_cores": self_sat_cores,
          "is_valid_complement": is_valid_complement,
          "is_startup_only": is_startup_only,
          "sample_ref": f"{sample_ev.file}:{sample_ev.line_no}",
      }
      step_records.append(rec)

      h_sum = by_hlo_summary.setdefault(
          hlo_op,
          {
              "hlo_op": hlo_op,
              "launch_count": 0,
              "valid_complement_launches": 0,
              "complement_counts": {},
              "self_satisfied_cores": [],
          },
      )
      h_sum["launch_count"] += 1
      if is_valid_complement:
        h_sum["valid_complement_launches"] += 1
        for c in complement:
          h_sum["complement_counts"][c] = h_sum["complement_counts"].get(c, 0) + 1
      for sc in self_sat_cores:
        if sc not in h_sum["self_satisfied_cores"]:
          h_sum["self_satisfied_cores"].append(sc)

    per_launch_collectives.extend(step_records)

  # Attribution across launches:
  # For each steady-state launch, take the earliest collective HLO that has a
  # valid non-empty complement (`is_valid_complement` and not `is_startup_only`).
  primary_complement_steps: dict[str, int] = {}
  primary_hlo_by_core: dict[str, str] = {}
  primary_alloc_by_core: dict[str, str] = {}
  primary_unsat_by_core: dict[str, int] = {}
  valid_steady_launches = 0

  by_launch_records: dict[str, list[dict[str, Any]]] = {}
  for rec in per_launch_collectives:
    by_launch_records.setdefault(rec["launch_key"], []).append(rec)

  for step_k, recs in by_launch_records.items():
    valid_recs = [r for r in recs if r["is_valid_complement"] and not r["is_startup_only"]]
    if not valid_recs:
      continue
    valid_steady_launches += 1
    # Prefer the earliest collective HLO in the launch with the smallest complement size.
    valid_recs.sort(key=lambda r: (len(r["silent_complement"]), r["hlo_order_in_launch"]))
    best_rec = valid_recs[0]
    for c in best_rec["silent_complement"]:
      primary_complement_steps[c] = primary_complement_steps.get(c, 0) + 1
      primary_hlo_by_core.setdefault(c, best_rec["hlo_op"])
      primary_alloc_by_core.setdefault(c, best_rec["alloc_label"] or "")
      primary_unsat_by_core[c] = max(primary_unsat_by_core.get(c, 0), best_rec["unsatisfied_count"])

  min_req_steps = 1 if valid_steady_launches < 3 else min(3, valid_steady_launches)
  suspected_stragglers: list[dict[str, Any]] = []
  for c, step_cnt in sorted(
      primary_complement_steps.items(),
      key=lambda kv: (-kv[1], -self_satisfied_counts.get(kv[0], 0), kv[0]),
  ):
    if step_cnt < min_req_steps:
      continue
    ci = cores.get(c)
    suspected_stragglers.append(
        {
            "core": c,
            "chip": ci.chip_label if ci else c,
            "global_core_index": ci.global_idx if ci else None,
            "host": ci.host if ci else "",
            "consistent_steps": step_cnt,
            "total_steady_launches": valid_steady_launches,
            "earliest_hlo_op": primary_hlo_by_core.get(c, ""),
            "alloc_label": primary_alloc_by_core.get(c, ""),
            "peer_Wait_sflag_timeouts": primary_unsat_by_core.get(c, 0),
            "total_cores": total_cores,
            "self_satisfied_timeouts": self_satisfied_counts.get(c, 0),
        }
    )

  if suspected_stragglers:
    verdict = "SUSPECTED_SLOW_CORE"
  elif launch_barrier_events or collective_events:
    verdict = "LAUNCH_BARRIER_NOISE_ONLY"
  else:
    verdict = "CLEAN"

  return {
      "raw_event_count": raw_count,
      "unique_event_count": len(uniq_events),
      "launch_barrier_count": len(launch_barrier_events),
      "collective_event_count": len(collective_events),
      "collective_launch_count": len(by_launch),
      "valid_steady_launches": valid_steady_launches,
      "verdict": verdict,
      "suspected_straggler_cores": suspected_stragglers,
      "self_satisfied_cores": sorted(self_satisfied_counts.keys()),
      "per_launch_collectives": per_launch_collectives,
      "by_hlo": by_hlo_summary,
  }


# ---------------------------------------------------------------------------
# Top-level combined analysis
# ---------------------------------------------------------------------------


def analyze_parsed_logs(
    parsed: events_mod.ParsedLogSet,
    pipeline_depth: int | None = None,
    topology_override: tuple[int, int, int] | None = None,
    cores_per_chip_override: int | None = None,
    wrap_mode: str = "auto",
) -> dict[str, Any]:
  """Runs both Approach-1 and Approach-2 attribution on `parsed`."""
  events_mod.normalize_parsed_log_set(parsed)
  cores = build_core_catalog(parsed)

  emitter_cfgs = [ev for ev in parsed.events if ev.kind == "EMITTER_CFG"]
  if pipeline_depth is not None:
    eff_depth = pipeline_depth
  elif emitter_cfgs and emitter_cfgs[0].pipeline_depth is not None:
    eff_depth = emitter_cfgs[0].pipeline_depth
  else:
    eff_depth = 13

  max_distance = emitter_cfgs[0].max_distance if emitter_cfgs and emitter_cfgs[0].max_distance is not None else 1

  topo = build_topology_model(
      parsed=parsed,
      cores=cores,
      topology_override=topology_override,
      cores_per_chip_override=cores_per_chip_override,
      max_distance=max_distance,
      wrap_mode=wrap_mode,
  )

  raw_xfer_events = [ev for ev in parsed.events if ev.kind in ("XFER_TIMEOUT", "XFER_TIMEOUT_COMPACT")]
  has_native_xfer = any("llo_log_report_handler" in ev.raw_line or ev.launch_id for ev in raw_xfer_events)
  has_delay_iter = any(ev.kind == "DELAY_ITER" for ev in parsed.events)
  xfer_dedup: dict[tuple[Any, ...], events_mod.Event] = {}
  for ev in raw_xfer_events:
    if has_native_xfer and has_delay_iter and "llo_log_report_handler" not in ev.raw_line and ev.run_id in (0, 1, 2):
      continue
    k = (ev.coord, ev.norm_run_id, ev.direction, ev.iteration, ev.src_coord, ev.dst_coord)
    if k not in xfer_dedup:
      xfer_dedup[k] = ev
  xfer_events = list(xfer_dedup.values())
  sflag_events = [ev for ev in parsed.events if ev.kind == "SFLAG_TIMEOUT"]

  raw_waves = cluster_xfer_events_into_waves(
      xfer_events=xfer_events,
      pipeline_depth=eff_depth,
      cores=cores,
      topo=topo,
  )
  infer_missing_p_from_waves(
      raw_waves=raw_waves,
      pipeline_depth=eff_depth,
      cores=cores,
      topo=topo,
  )

  waves_out: list[dict[str, Any]] = []
  for idx, w_evs in enumerate(raw_waves, start=1):
    waves_out.append(
        analyze_single_wave(
            wave_idx=idx,
            wave_events=w_evs,
            pipeline_depth=eff_depth,
            cores=cores,
            topo=topo,
        )
    )

  signal_waves = [w for w in waves_out if w["verdict"] != "NOISE"]
  noise_waves = [w for w in waves_out if w["verdict"] == "NOISE"]

  # Aggregate Approach-1 slow cores and bad links across signal waves.
  culprit_links: list[str] = []
  suspected_links_agg: list[dict[str, Any]] = []
  slow_core_wave_counts: dict[str, int] = {}
  blind_slow_core_wave_counts: dict[str, int] = {}
  slow_core_s0s: dict[str, list[int]] = {}
  slow_core_covs: dict[str, list[float]] = {}

  for w in signal_waves:
    for sl in w["suspected_links"]:
      lbl = sl["link_label"]
      if lbl not in culprit_links:
        culprit_links.append(lbl)
        suspected_links_agg.append(sl)
    for sc in w["suspected_slow_cores"]:
      cname = sc["core"]
      slow_core_wave_counts[cname] = slow_core_wave_counts.get(cname, 0) + 1
      slow_core_s0s.setdefault(cname, []).append(w["s0_iteration"])
      slow_core_covs.setdefault(cname, []).append(sc["neighbor_coverage"])
    for bsc in w["blind_suspected_slow_cores"]:
      bname = bsc["core"]
      blind_slow_core_wave_counts[bname] = blind_slow_core_wave_counts.get(bname, 0) + 1

  suspected_slow_cores_agg: list[dict[str, Any]] = []
  for cname in sorted(slow_core_wave_counts.keys(), key=lambda x: (-slow_core_wave_counts[x], x)):
    ci = cores.get(cname)
    cov_list = slow_core_covs.get(cname, [1.0])
    suspected_slow_cores_agg.append(
        {
            "core": cname,
            "wave_count": slow_core_wave_counts[cname],
            "blind_wave_count": blind_slow_core_wave_counts.get(cname, 0),
            "chip": ci.chip_label if ci else cname,
            "global_core_index": ci.global_idx if ci else None,
            "host": ci.host if ci else "",
            "logged_p": ci.get_logged_p() if ci else None,
            "effective_p": ci.get_effective_p() if ci else None,
            "s0_iterations": slow_core_s0s.get(cname, []),
            "mean_neighbor_coverage": round(sum(cov_list) / len(cov_list), 4),
        }
    )

  sflag_summary = analyze_sflag_timeouts(
      sflag_events=sflag_events,
      cores=cores,
      topo=topo,
  )

  culprit_slow_cores = [sc["core"] for sc in suspected_slow_cores_agg]
  for a2_sc in sflag_summary["suspected_straggler_cores"]:
    if a2_sc["core"] not in culprit_slow_cores:
      culprit_slow_cores.append(a2_sc["core"])

  if culprit_links and not culprit_slow_cores:
    overall_verdict = "SUSPECTED_BAD_LINK"
  elif culprit_slow_cores and not culprit_links:
    overall_verdict = "SUSPECTED_SLOW_CORE"
  elif culprit_links and culprit_slow_cores:
    overall_verdict = "SUSPECTED_BAD_LINK_AND_SLOW_CORE"
  elif noise_waves:
    overall_verdict = "NOISE_ONLY"
  elif sflag_summary["raw_event_count"] > 0:
    overall_verdict = "LAUNCH_BARRIER_NOISE_ONLY"
  else:
    overall_verdict = "CLEAN"

  ground_truth_p: dict[str, Any] = {}
  for cname, ci in sorted(cores.items()):
    if ci.delay_events or ci.inferred_p_by_run:
      ground_truth_p[cname] = {
          "core": cname,
          "chip": ci.chip_label,
          "global_core_index": ci.global_idx,
          "host": ci.host,
          "logged_p": ci.get_logged_p(),
          "effective_p": ci.get_effective_p(),
          "delay_p_by_run": {("default" if k is None else str(k)): v for k, v in ci.delay_p_by_run.items()},
          "inferred_p_by_run": {("default" if k is None else str(k)): v for k, v in ci.inferred_p_by_run.items()},
          "bw_gibs": [ev.bw_gibs for ev in ci.delay_events if ev.bw_gibs is not None],
          "time_ms": [ev.time_ms for ev in ci.delay_events if ev.time_ms is not None],
      }

  return {
      "files_scanned": parsed.files_scanned,
      "pipeline_depth": eff_depth,
      "expected_hop_step": eff_depth + 1,
      "run_id_offsets": parsed.run_id_offsets,
      "topology": (
          {
              "bounds": list(topo.bounds),
              "cores_per_chip": topo.cores_per_chip,
              "max_distance": topo.max_distance,
              "wrap_mode": topo.wrap_mode,
              "wrap_axes": list(topo.wrap_axes),
              "total_cores": len(topo.all_cores),
          }
          if topo
          else None
      ),
      "total_xfer_timeout_events": len(xfer_events),
      "send_timeout_events": sum(1 for ev in xfer_events if ev.direction == "SEND"),
      "recv_timeout_events": sum(1 for ev in xfer_events if ev.direction == "RECV"),
      "sflag_timeout_raw_events": sflag_summary["raw_event_count"],
      "sflag_timeout_unique_events": sflag_summary["unique_event_count"],
      "approach2_sflag_summary": sflag_summary,
      "total_waves_clustered": len(waves_out),
      "signal_wave_count": len(signal_waves),
      "noise_wave_count": len(noise_waves),
      "verdict": overall_verdict,
      "overall_verdict": overall_verdict,
      "culprit_links": culprit_links,
      "suspected_links": suspected_links_agg,
      "culprit_slow_cores": culprit_slow_cores,
      "suspected_slow_cores": suspected_slow_cores_agg,
      "ground_truth_p": ground_truth_p,
      "waves": waves_out,
      "noise_waves": noise_waves,
  }
