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

"""Torus-aligned core-minor [ICI, core] TPU7x (64c/128c/256c/512c) and CPU mesh construction."""

from __future__ import annotations

import jax
from jax.experimental import topologies
from jax.sharding import Mesh
import numpy as np

from maxtext.experimental.lineage.qwen.config import QwenConfig


def create_mesh(
    cfg: QwenConfig,
    *,
    devices: list[jax.Device] | None = None,
    aot_topology: str | None = "gf=4x8x8_untwisted",
) -> Mesh:
  """Builds a core-minor 3D `('fsdp', 'expert', 'ep_d2d')` or 2D `('fsdp', 'expert')` Mesh.

  Supports Ghostfish (`tpu7x`) slices across:
  - `64c` (`gf=4x4x4`, `128` TensorCores): `FSDP=16` `(y=4, x=4)`, `expert=4`
  `(z=4)`, `ep_d2d=2` `(core=2)`.
  - `128c` (`gf=4x4x8_untwisted`, `256` TensorCores): `FSDP=16` `(y=4, x=4)`,
  `expert=8` `(z=8)`, `ep_d2d=2` `(core=2)`.
  - `256c` (`gf=4x8x8_untwisted`, `512` TensorCores): `FSDP=32` `(y=8, x=4)`,
  `expert=8` `(z=8)`, `ep_d2d=2` `(core=2)`.
  - `512c` (`gf=8x8x8_untwisted`, `1024` TensorCores): `FSDP=64` `(y=8, x=8)`,
  `expert=8` `(z=8)`, `ep_d2d=2` `(core=2)`.

  When `cfg.decouple_ep_d2d=True` and `cfg.ep_core_minor=True` (default):
  `grid[x, y, z, core].transpose(1, 0, 2, 3)` places `core=2` (`ep_d2d`) as the
  minor-most axis, ensuring both `('expert', 'ep_d2d')` EP collectives (`SC0`)
  and `('fsdp', 'ep_d2d')` dense Stage-2 collectives (`SC1`) keep adjacent
  chiplets `[2k, 2k+1]` consecutive in every ring (`~186-200 GB/s`).

  Args:
    cfg: Qwen model and sharding configuration.
    devices: Optional explicit device list; overrides `aot_topology`.
    aot_topology: Offline TPU topology descriptor name when `devices` is None.

  Returns:
    Constructed JAX `Mesh` matching `cfg`.
  """
  if devices is None:
    if aot_topology is not None:
      last_err: Exception | None = None
      for cand in dict.fromkeys(
          [aot_topology, aot_topology.removesuffix("_untwisted")]
      ):
        try:
          topo = topologies.get_topology_desc(
              platform="tpu", topology_name=cand
          )
          devices = list(topo.devices)
          break
        except Exception as err:  # pylint: disable=broad-exception-caught
          last_err = err
      if devices is None:
        raise ValueError(
            f"Unsupported aot_topology={aot_topology!r}: {last_err}"
        ) from last_err
    else:
      devices = list(jax.devices())

  if len(devices) != cfg.num_devices:
    raise ValueError(
        f"Expected {cfg.num_devices} devices (fsdp={cfg.fsdp_size} x"
        f" ep={cfg.ep_size}), got {len(devices)}."
    )

  has_coords = (
      bool(devices)
      and hasattr(devices[0], "coords")
      and hasattr(devices[0], "core_on_chip")
      and len(getattr(devices[0], "coords", ())) == 3
  )
  if has_coords:
    xs = max(int(d.coords[0]) for d in devices) + 1
    ys = max(int(d.coords[1]) for d in devices) + 1
    zs = max(int(d.coords[2]) for d in devices) + 1
    cores = max(int(d.core_on_chip) for d in devices) + 1
    if xs * ys * zs * cores == len(devices) and cores == 2:
      grid = np.empty((xs, ys, zs, cores), dtype=object)
      for d in devices:
        x, y, z = d.coords
        grid[x, y, z, d.core_on_chip] = d

      if cfg.decouple_ep_d2d:
        if cfg.mesh_mode == "x1e_yx_core":
          # Pick a single closed physical torus axis of length `ep_ici_size`
          # for `'expert'` so EP collectives run on an unbroken 1D torus ring
          # and `'fsdp'` spans the remaining 2D sub-torus. Priority `(z, y, x)`
          # keeps exact `(y, x, z, core)` order across all canonical slices:
          # `64c` (`4x4x4`), `128c` (`4x4x8_untwisted`), `256c`
          # (`4x8x8_untwisted`), and `512c` (`8x8x8_untwisted`, where `fsdp=64`
          # owns `(y=8, x=8)` and `expert=8` owns `z=8`).
          if zs == cfg.ep_ici_size:
            fsdp_a, fsdp_b, ep_ax = 1, 0, 2
          elif ys == cfg.ep_ici_size:
            fsdp_a, fsdp_b, ep_ax = 2, 0, 1
          elif xs == cfg.ep_ici_size:
            fsdp_a, fsdp_b, ep_ax = 2, 1, 0
          elif xs * ys == cfg.ep_ici_size and zs == cfg.fsdp_size:
            # 2D `4x4x1` EP torus `(y, x)` + 1D `1x1x4` FSDP ring `z` (e.g.
            # `64c` `4x4x4` proxy with `EP_ICI=16, FSDP_ICI=4`).
            if cfg.ep_core_minor:
              transposed = grid.transpose(2, 1, 0, 3)
              arr = transposed.reshape(cfg.fsdp_size, cfg.ep_ici_size, 2)
              return Mesh(arr, ("fsdp", "expert", "ep_d2d"))
            transposed = grid.transpose(2, 3, 1, 0)
            arr = transposed.reshape(cfg.fsdp_size, 2, cfg.ep_ici_size)
            return Mesh(arr, ("fsdp", "ep_d2d", "expert"))
          else:
            fsdp_a, fsdp_b, ep_ax = 1, 0, 2
          if cfg.ep_core_minor:
            transposed = grid.transpose(fsdp_a, fsdp_b, ep_ax, 3)
            arr = transposed.reshape(cfg.fsdp_size, cfg.ep_ici_size, 2)
            return Mesh(arr, ("fsdp", "expert", "ep_d2d"))
          transposed = grid.transpose(fsdp_a, fsdp_b, 3, ep_ax)
          arr = transposed.reshape(cfg.fsdp_size, 2, cfg.ep_ici_size)
          return Mesh(arr, ("fsdp", "ep_d2d", "expert"))
        if cfg.mesh_mode == "ep4_x_fsdp128_yz_core":
          transposed = grid.transpose(1, 2, 0, 3)
          arr = transposed.reshape(cfg.fsdp_size, cfg.ep_ici_size, 2)
          return Mesh(arr, ("fsdp", "expert", "ep_d2d"))
        raise ValueError(
            f"Unsupported mesh_mode={cfg.mesh_mode!r} with decouple_ep_d2d=True"
        )

      if cfg.mesh_mode == "x1e_yx_core":
        transposed = grid.transpose(1, 0, 3, 2)
      elif cfg.mesh_mode == "ep4_x_fsdp128_yz_core":
        transposed = grid.transpose(1, 2, 3, 0)
      elif cfg.mesh_mode == "xcore_ep8_yz_fsdp64":
        transposed = grid.transpose(1, 2, 0, 3)
      else:
        raise ValueError(f"Unknown mesh_mode={cfg.mesh_mode!r}")
      return Mesh(
          transposed.reshape(cfg.fsdp_size, cfg.ep_size), ("fsdp", "expert")
      )

  if cfg.decouple_ep_d2d:
    if cfg.ep_core_minor:
      arr = np.asarray(devices, dtype=object).reshape(
          cfg.fsdp_size, cfg.ep_ici_size, 2
      )
      return Mesh(arr, ("fsdp", "expert", "ep_d2d"))
    arr = np.asarray(devices, dtype=object).reshape(
        cfg.fsdp_size, 2, cfg.ep_ici_size
    )
    return Mesh(arr, ("fsdp", "ep_d2d", "expert"))
  arr = np.asarray(devices, dtype=object).reshape(cfg.fsdp_size, cfg.ep_size)
  return Mesh(arr, ("fsdp", "expert"))
