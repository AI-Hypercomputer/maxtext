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

"""Torus-aligned TPU7x (4x4x4 / 4x8x8) and CPU mesh construction for Qwen3.5."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
from jax.experimental import topologies
from jax.sharding import Mesh
import numpy as np

from maxtext.experimental.qwen_35_64c.config import Qwen35Config


def create_mesh(
    cfg: Qwen35Config,
    *,
    devices: list[jax.Device] | None = None,
    aot_topology: str | None = "tpu7x:4x4x4",
) -> Mesh:
  """Builds a 2D ('fsdp', 'expert') Mesh matching MaxText physical torus axes.

  On `tpu7x:4x4x4` (`128` TensorCores across `64` chips), `x1e_yx_core` gives
  `fsdp=16` on `(y=4, x=4)`, `ep_d2d=2` on `core`, and `expert=4` on the `z=4`
  torus ring.

  On `tpu7x:4x8x8` (`512` logical TensorCores across `256` TPU v7x chips):
  - `x1e_yx_core`: `grid[x, y, z, core].transpose(1, 0, 3, 2)` ->
    `fsdp=64` spans `(y=8, x=4, core=2)` and `expert=8` spans the wrap-around
    `z=8` torus ring (`1` TensorCore per chip along `z`), giving disjoint
    physical ICI axes for `SC0` (EP along `z`) vs `SC1` (FSDP along `y, x,
    core`).
  - `ep4_x_fsdp128_yz_core`: `grid[x, y, z, core].transpose(1, 2, 3, 0)` ->
    `fsdp=128` on `(y=8, z=8, core=2)` and `expert=4` on `x=4`.
  - `xcore_ep8_yz_fsdp64`: `grid[x, y, z, core].transpose(1, 2, 0, 3)` ->
    `fsdp=64` on `(y=8, z=8)` and `expert=8` on `(x=4, core=2)`.
  """
  if devices is None:
    if aot_topology is not None:
      topo = topologies.get_topology_desc(platform="tpu", topology_name=aot_topology)
      devices = list(topo.devices)
    else:
      devices = list(jax.devices())

  arr, axis_names = arrange_mesh_devices(cfg, devices)
  return Mesh(arr, axis_names)


def arrange_mesh_devices(
    cfg: Qwen35Config,
    devices: Sequence[Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
  """Arranges `devices` into the physical torus `(arr, axis_names)` grid."""
  if len(devices) != cfg.num_devices:
    raise ValueError(
        f"Expected {cfg.num_devices} devices (fsdp={cfg.fsdp_size} x" f" ep={cfg.ep_size}), got {len(devices)}."
    )

  has_coords = hasattr(devices[0], "coords") and hasattr(devices[0], "core_on_chip")
  if has_coords:
    dims = [1 + max(d.coords[i] for d in devices) for i in range(3)]
    num_cores = 1 + max(d.core_on_chip for d in devices)
    grid = np.empty((*dims, num_cores), dtype=object)
    for d in devices:
      x, y, z = d.coords
      grid[x, y, z, d.core_on_chip] = d
    if any(g is None for g in grid.flat):
      raise ValueError(f"Devices do not fill a dense {dims}x{num_cores} torus grid.")

    if cfg.decouple_ep_d2d:
      if num_cores != 2:
        raise ValueError(f"decouple_ep_d2d=True requires 2 cores/chip, got {num_cores}.")
      if cfg.mesh_mode == "x1e_yx_core":
        chip_ordered = grid.transpose(1, 0, 2, 3)  # (y, x, z, core)
      elif cfg.mesh_mode in ("ep4_x_fsdp128_yz_core", "xcore_ep8_yz_fsdp64"):
        chip_ordered = grid.transpose(1, 2, 0, 3)  # (y, z, x, core)
      else:
        raise ValueError(f"Unknown mesh_mode={cfg.mesh_mode!r}")
      arr = chip_ordered.reshape(cfg.fsdp_size, cfg.ep_ici_size, 2).transpose(0, 2, 1)
      return arr, ("fsdp", "ep_d2d", "expert")

    if cfg.mesh_mode == "x1e_yx_core":
      transposed = grid.transpose(1, 0, 3, 2)
    elif cfg.mesh_mode == "ep4_x_fsdp128_yz_core":
      transposed = grid.transpose(1, 2, 3, 0)
    elif cfg.mesh_mode == "xcore_ep8_yz_fsdp64":
      transposed = grid.transpose(1, 2, 0, 3)
    else:
      raise ValueError(f"Unknown mesh_mode={cfg.mesh_mode!r}")

    arr = transposed.reshape(cfg.fsdp_size, cfg.ep_size)
  else:
    if cfg.decouple_ep_d2d:
      arr = np.asarray(devices, dtype=object).reshape(cfg.fsdp_size, 2, cfg.ep_ici_size)
      return arr, ("fsdp", "ep_d2d", "expert")
    arr = np.asarray(devices, dtype=object).reshape(cfg.fsdp_size, cfg.ep_size)

  return arr, ("fsdp", "expert")
