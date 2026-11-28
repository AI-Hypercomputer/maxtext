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
"""Mesh, topology and layer-placement helpers for MPMD pipeline parallelism.

An MPMD pipeline is described by two pieces of data:

  * a *topology*: a mapping from mesh name (``m0``, ``m1``, ...) to a
    :class:`jax.sharding.Mesh` holding the devices of that pipeline stage, and
  * a *name-to-mesh assignment*: a mapping from the name of each
    ``mpmd.named_computation`` (``embed``, ``layer0``, ..., ``unembed``) to the
    mesh it must execute on.

This module builds both, and nothing else: it has no JIT, model or MaxText
dependency, which keeps it cheap to unit test on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any

import jax
from jax.sharding import Mesh
import numpy as np

# Mesh axis names used inside a single pipeline stage. They are deliberately
# the names MaxText's `logical_axis_rules` refer to (`fsdp`, `tensor`,
# `expert`), so MaxText's own layers resolve their activation and weight
# shardings against a stage mesh without any translation.
FSDP_AXIS = "fsdp"
TENSOR_AXIS = "tensor"
EXPERT_AXIS = "expert"

# Mesh axes the (micro)batch dimension is sharded over. MaxText maps
# `activation_batch` onto `('data', 'fsdp', 'fsdp_transpose', 'expert')`: with
# expert parallelism the batch is still split over *every* non-tensor device
# of the stage, and the `expert` axis additionally partitions the MoE experts
# so that tokens are exchanged with an all-to-all instead of every device
# all-gathering the full expert weights.
DATA_PARALLEL_AXES = (FSDP_AXIS, EXPERT_AXIS)

# Names of the non-layer named computations.
EMBED_NAME = "embed"
UNEMBED_NAME = "unembed"


def mesh_name(stage_index: int) -> str:
  """Returns the canonical mesh name for `stage_index`."""
  return f"m{stage_index}"


def layer_name(layer_index: int) -> str:
  """Returns the canonical named-computation name for `layer_index`."""
  return f"layer{layer_index}"


def pre_stage_name(stage_index: int) -> str:
  """Returns the canonical pre-loop named-computation name for `stage_index`."""
  return f"pre_m{stage_index}"


def post_stage_name(stage_index: int) -> str:
  """Returns the canonical post-loop named-computation name for `stage_index`."""
  return f"post_m{stage_index}"


def data_parallel_axes(mesh: Any) -> tuple[str, ...]:
  """Returns the axes of `mesh` that the batch dimension is sharded over.

  Args:
    mesh: A concrete or abstract mesh with ``axis_names``.

  Returns:
    The subset of :data:`DATA_PARALLEL_AXES` present in `mesh`, in mesh order.
  """
  names = tuple(getattr(mesh, "axis_names", ()))
  return tuple(name for name in names if name in DATA_PARALLEL_AXES)


def data_parallel_size(mesh: Any) -> int:
  """Returns the number of ways the batch dimension is sharded on `mesh`."""
  shape = dict(getattr(mesh, "shape", {}))
  size = 1
  for name in data_parallel_axes(mesh):
    size *= int(shape.get(name, 1))
  return size


@dataclasses.dataclass(frozen=True)
class StageTopology:
  """A pipeline topology together with the metadata used to build it.

  Attributes:
    meshes: Mapping from mesh name to the stage's device mesh.
    num_stages: Number of pipeline stages.
    devices_per_stage: Number of devices backing each stage.
    fsdp_size: FSDP degree inside each stage.
    tensor_size: Tensor-parallel degree inside each stage.
    expert_size: Expert-parallel degree inside each stage.
  """

  meshes: dict[str, Mesh]
  num_stages: int
  devices_per_stage: int
  fsdp_size: int
  tensor_size: int
  expert_size: int = 1

  @property
  def axis_names(self) -> tuple[str, ...]:
    """Returns the mesh axis names common to every stage mesh."""
    return self.meshes[mesh_name(0)].axis_names

  @property
  def data_parallel_size(self) -> int:
    """Returns how many ways a microbatch is sharded inside a stage."""
    return self.fsdp_size * self.expert_size

  def describe(self) -> str:
    """Returns a one-line human readable summary."""
    return (
        f"{self.num_stages} stages x {self.devices_per_stage} devices"
        f" (fsdp={self.fsdp_size}, expert={self.expert_size},"
        f" tensor={self.tensor_size}), axes={self.axis_names}"
    )


# The 3D TPU sub-cube shapes (in chip coordinates) that XLA:TPU's
# supported-slice-shape check accepts.
_SUPPORTED_3D_SUBSLICES = frozenset(
    {
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 4),
        (2, 4, 4),
    }
)


def _is_supported_3d_subslice(dx: int, dy: int, dz: int) -> bool:
  """Returns whether ``(dx, dy, dz)`` is a valid XLA 3D TPU subslice shape.

  When ``grid_z > 1`` (3D TPU topologies such as TPU v7x ``tpu7x-4x4x4`` or
  v5p), XLA:TPU's collective verification applies its supported-slice-shape
  check to the bounding box of every stage's collective replica group. A
  subslice is supported iff all dimensions are multiples of 4 (multi-cube
  slice) or the ``(dx, dy, dz)`` chip bounds match one of the sub-cube
  topologies above.

  Args:
    dx: Sub-box chip extent along X.
    dy: Sub-box chip extent along Y.
    dz: Sub-box chip extent along Z.
  """
  if dx <= 0 or dy <= 0 or dz <= 0:
    return False
  if dx % 4 == 0 and dy % 4 == 0 and dz % 4 == 0:
    return True
  return (dx, dy, dz) in _SUPPORTED_3D_SUBSLICES


def _order_subbox_coords(dx: int, dy: int, dz: int) -> list[tuple[int, int, int]]:
  """Orders a ``(dx, dy, dz)`` sub-box in row-major ``(x, y, z)`` order.

  XLA:TPU's ring reordering for limited ICI routing constructs hardware
  Hamiltonian cycles over 2D torus slices automatically, under the
  precondition that every collective's ``replica_group`` -- including
  power-of-2 contiguous or strided subgroups when a tensor's batch or sequence
  dimension is smaller than ``fsdp`` -- forms a Cartesian product
  ``X_subset x Y_subset``. Row-major ordering inside each stage's contiguous
  rectangular ``(dx, dy, dz)`` sub-box guarantees that both the full stage and
  every power-of-2 contiguous or strided subgroup form rectangular 2D slices.

  Args:
    dx: Sub-box size along the X coordinate axis.
    dy: Sub-box size along the Y coordinate axis.
    dz: Sub-box size along the Z coordinate axis.

  Returns:
    Row-major list of ``(x, y, z)`` coordinate offsets inside the sub-box.
  """
  if dx <= 0 or dy <= 0 or dz <= 0:
    return []
  coords: list[tuple[int, int, int]] = []
  for z in range(dz):
    for x in range(dx):
      for y in range(dy):
        coords.append((x, y, z))
  return coords


def _snake_order_blocks(px: int, py: int, pz: int) -> list[tuple[int, int, int]]:
  """Orders a ``(px, py, pz)`` grid of stage blocks as a boustrophedon path.

  Planes go along z, rows along x with the x direction reversed on every other
  plane, and blocks along y with the y direction reversed on every other row,
  so each block is a face neighbour of the one before it: consecutive pipeline
  stages share a face of their sub-boxes (their nearest chips are one ICI hop
  apart). Row-major order instead puts, e.g., stages 1 and 2 of a 4x4 grid cut
  into 2x2 blocks only diagonally next to each other.

  Args:
    px: Number of blocks along X.
    py: Number of blocks along Y.
    pz: Number of blocks along Z.

  Returns:
    The ``(x, y, z)`` block coordinates in path order.
  """
  coords: list[tuple[int, int, int]] = []
  row = 0
  for z in range(pz):
    xs = range(px) if z % 2 == 0 else range(px - 1, -1, -1)
    for x in xs:
      ys = range(py) if row % 2 == 0 else range(py - 1, -1, -1)
      coords.extend((x, y, z) for y in ys)
      row += 1
  return coords


def _partition_grid_into_stages(
    flat_devices: np.ndarray,
    num_stages: int,
) -> list[np.ndarray] | None:
  """Partitions a dense 3D TPU device grid into contiguous rectangular stages.

  JAX's default ``devices()`` list enumerates chips host-by-host (``2x2`` tiles
  per host on v5e/v6e), so slicing ``flat_devices[s * n : (s + 1) * n]`` neither
  guarantees rectangular sub-slices nor preserves rectangular 2D slices for
  subgroup collectives. When the devices carry ``(x, y, z)`` coordinates that
  tile a dense ``(X, Y, Z)`` box and ``num_stages`` factors into
  ``(P_x, P_y, P_z)`` dividing ``(X, Y, Z)``, this function:

    1. Picks a stage sub-box ``(dx, dy, dz)`` that passes XLA:TPU's
       supported-slice-shape check on 3D topologies, preserves whole ``2x2``
       hosts whenever possible, and keeps each stage compact.
    2. Orders the ``(P_x, P_y, P_z)`` stage blocks as a snake
       (:func:`_snake_order_blocks`), so adjacent pipeline stages ``m_s`` and
       ``m_{s+1}`` are face neighbours on the ICI torus.
    3. Orders the devices inside each stage's ``(dx, dy, dz)`` sub-box using
       :func:`_order_subbox_coords` so both full-stage and subgroup collectives
       satisfy XLA's rectangular slice requirement for limited ICI routing.

  Args:
    flat_devices: 1D array of JAX devices.
    num_stages: Number of pipeline stages.

  Returns:
    A list of ``num_stages`` 1D device arrays, or ``None`` if the devices do
    not form a factorable 3D coordinate grid.
  """
  coord_to_devices: dict[tuple[int, int, int], list[Any]] = {}
  for dev in flat_devices:
    coords = getattr(dev, "coords", None)
    if coords is None or len(coords) != 3:
      return None
    key = (int(coords[0]), int(coords[1]), int(coords[2]))
    coord_to_devices.setdefault(key, []).append(dev)

  if not coord_to_devices:
    return None

  num_devices = flat_devices.size
  num_chips = len(coord_to_devices)
  if num_devices % num_chips != 0 or num_chips % num_stages != 0:
    return None
  cores_per_chip = num_devices // num_chips
  for key, devs in coord_to_devices.items():
    if len(devs) != cores_per_chip:
      return None
    coord_to_devices[key] = sorted(
        devs,
        key=lambda d: (getattr(d, "core_on_chip", 0), getattr(d, "id", 0)),
    )

  xs = [c[0] for c in coord_to_devices]
  ys = [c[1] for c in coord_to_devices]
  zs = [c[2] for c in coord_to_devices]
  if min(xs) != 0 or min(ys) != 0 or min(zs) != 0:
    return None
  grid_x, grid_y, grid_z = max(xs) + 1, max(ys) + 1, max(zs) + 1
  if grid_x * grid_y * grid_z != num_chips:
    return None

  candidates: list[tuple[tuple[int, ...], tuple[int, int, int]]] = []
  for px in range(1, num_stages + 1):
    if num_stages % px != 0 or grid_x % px != 0:
      continue
    rem_yz = num_stages // px
    for py in range(1, rem_yz + 1):
      if rem_yz % py != 0 or grid_y % py != 0:
        continue
      pz = rem_yz // py
      if grid_z % pz != 0:
        continue
      dx, dy, dz = grid_x // px, grid_y // py, grid_z // pz
      # First require 3D subslice bounds that XLA:TPU's supported-slice-shape
      # check accepts, then prefer multiples of 2x2 hosts in (X, Y), the
      # canonical sub-cube orientation (dx <= dy <= dz, e.g. 2x2x4 on 4x4x4),
      # and compact sub-boxes.
      supported_subslice = int(grid_z == 1 or _is_supported_3d_subslice(dx, dy, dz))
      whole_hosts = int((dx % 2 == 0 or grid_x < 2) and (dy % 2 == 0 or grid_y < 2))
      canonical_subcube_order = int(dx <= dy <= dz) if grid_z > 1 else 1
      is_2d_slice = int(sum(d > 1 for d in (dx, dy, dz)) <= 2)
      has_cycle = int((dx >= 2 and dy >= 2 and (dx % 2 == 0 or dy % 2 == 0)) or dx * dy <= 2)
      score = (
          -supported_subslice,
          -whole_hosts,
          -canonical_subcube_order,
          -is_2d_slice,
          -has_cycle,
          max(dx, dy, dz),
          dx + dy + dz,
          -dz,
      )
      candidates.append((score, (px, py, pz)))

  if not candidates:
    return None

  candidates.sort(key=lambda item: item[0])
  px, py, pz = candidates[0][1]
  dx, dy, dz = grid_x // px, grid_y // py, grid_z // pz

  stage_block_coords = _snake_order_blocks(px, py, pz)
  intra_stage_coords = _order_subbox_coords(dx, dy, dz)

  stages: list[np.ndarray] = []
  for bx, by, bz in stage_block_coords:
    base_x, base_y, base_z = bx * dx, by * dy, bz * dz
    stage_devs = []
    for rx, ry, rz in intra_stage_coords:
      stage_devs.extend(coord_to_devices[(base_x + rx, base_y + ry, base_z + rz)])
    stages.append(np.asarray(stage_devs, dtype=object))
  return stages


def _devices_by_slice(flat_devices: np.ndarray) -> list[np.ndarray]:
  """Groups devices by ``slice_index``, in slice order.

  Returns one group holding every device when there is a single slice, or when
  the devices carry no ``slice_index`` (CPU devices).
  """
  slice_ids = [getattr(dev, "slice_index", None) for dev in flat_devices]
  if any(sid is None for sid in slice_ids) or len(set(slice_ids)) <= 1:
    return [flat_devices]
  by_slice: dict[int, list[Any]] = {}
  for dev, sid in zip(flat_devices, slice_ids):
    by_slice.setdefault(int(sid), []).append(dev)
  return [np.asarray(by_slice[sid], dtype=object) for sid in sorted(by_slice)]


def _partition_devices_into_stages(flat_devices: np.ndarray, num_stages: int) -> list[np.ndarray]:
  """Splits one slice's devices into stages: grid sub-boxes, else contiguous."""
  stages = _partition_grid_into_stages(flat_devices, num_stages)
  if stages is not None:
    return stages
  per_stage = flat_devices.size // num_stages
  return [flat_devices[stage * per_stage : (stage + 1) * per_stage] for stage in range(num_stages)]


def _partition_slices_into_stages(slices: list[np.ndarray], num_stages: int) -> list[np.ndarray]:
  """Splits a multi-slice device set into stages that never straddle slices.

  Every slice's chips have the same coordinates, so partitioning the devices of
  all slices as one coordinate grid would put devices of every slice into every
  stage: each stage's FSDP collectives would cross DCN while the stage-boundary
  transfers stayed on ICI, the opposite of pipelining over DCN. Instead no stage
  straddles a slice: each slice holds ``num_stages / num_slices`` consecutive
  stages (cut as in a single slice), so only the boundaries between slices
  cross DCN; or, with more slices than stages, each stage holds ``num_slices /
  num_stages`` whole slices, slice-major, so the outermost axis of its mesh
  (``fsdp``; ``tensor`` or ``expert`` with ``--fsdp=1``) runs over DCN.

  Args:
    slices: Each slice's devices, in slice order.
    num_stages: Number of pipeline stages.

  Returns:
    ``num_stages`` 1D device arrays.

  Raises:
    ValueError: If the slices differ in size, or if neither of ``num_stages``
      and the number of slices divides the other.
  """
  num_slices = len(slices)
  sizes = {group.size for group in slices}
  if len(sizes) != 1:
    raise ValueError(f"Slices hold different numbers of devices: {sorted(sizes)}.")
  if num_stages % num_slices == 0:
    stages: list[np.ndarray] = []
    for group in slices:
      stages += _partition_devices_into_stages(group, num_stages // num_slices)
    return stages
  if num_slices % num_stages == 0:
    per_stage = num_slices // num_stages
    return [
        np.concatenate(
            [_partition_devices_into_stages(group, 1)[0] for group in slices[stage * per_stage : (stage + 1) * per_stage]]
        )
        for stage in range(num_stages)
    ]
  raise ValueError(
      f"num_stages ({num_stages}) and the number of slices ({num_slices}) must"
      " divide one another, so that no stage straddles a slice."
  )


def create_stage_meshes(
    devices: Sequence[Any] | np.ndarray,
    num_stages: int,
    fsdp_size: int = -1,
    expert_size: int = 1,
) -> StageTopology:
  """Splits `devices` into one device mesh per pipeline stage.

  When `devices` carry 3D TPU chip coordinates forming a dense grid, the grid
  is partitioned into contiguous rectangular stage sub-boxes (of whole ``2x2``
  hosts where the grid allows; a v5p-16 at four stages gets half a host each),
  ordered in a snake across stages so consecutive stages are neighbours,
  with each stage's devices in row-major order (rectangular collective
  subgroups; see :func:`_partition_grid_into_stages`). Otherwise, devices are
  partitioned contiguously in the order supplied by the caller: stage ``i``
  receives ``devices[i * n : (i + 1) * n]``. Devices of several slices
  (``slice_index``) are grouped by slice first, so that no stage straddles a
  slice: each slice holds whole stages, or each stage whole slices (see
  :func:`_partition_slices_into_stages`).

  Each stage mesh is ``fsdp x tensor x expert`` (axes of size 1 are omitted,
  except that a stage always keeps at least the ``fsdp`` axis so that shardings
  can be expressed uniformly). The axis order mirrors MaxText's ``mesh_axes``
  (``fsdp`` before ``tensor`` before ``expert``), so a batch sharded over
  ``('fsdp', 'expert')`` is laid out exactly like MaxText lays it out.

  Args:
    devices: Devices to distribute, either a flat sequence or an ndarray.
    num_stages: Number of pipeline stages to create.
    fsdp_size: FSDP degree inside each stage. -1 (the default) uses every device
      of the stage not already consumed by `expert_size`, i.e. no tensor
      parallelism.
    expert_size: Expert-parallel degree inside each stage. MoE expert weights
      are partitioned (not replicated or FSDP-gathered) across this axis, and
      the batch is additionally sharded over it.

  Returns:
    The resulting :class:`StageTopology`.

  Raises:
    ValueError: If the devices cannot be split evenly into the requested
      stage/FSDP/expert shape.
  """
  if num_stages <= 0:
    raise ValueError(f"num_stages must be positive, got {num_stages}.")
  if expert_size <= 0:
    raise ValueError(f"expert_size must be positive, got {expert_size}.")

  flat_devices = np.asarray(devices).flatten()
  num_devices = flat_devices.size
  if num_devices < num_stages:
    raise ValueError(f"Not enough devices ({num_devices}) for {num_stages} pipeline stages.")
  if num_devices % num_stages:
    raise ValueError(f"Number of devices ({num_devices}) must be divisible by num_stages" f" ({num_stages}).")

  devices_per_stage = num_devices // num_stages
  if devices_per_stage % expert_size:
    raise ValueError(f"devices_per_stage ({devices_per_stage}) must be divisible by" f" expert_size ({expert_size}).")
  if fsdp_size <= 0:
    fsdp_size = devices_per_stage // expert_size
  if fsdp_size * expert_size > devices_per_stage:
    raise ValueError(
        f"fsdp_size x expert_size ({fsdp_size} x {expert_size}) exceeds the"
        f" number of devices available per stage ({devices_per_stage})."
        " Reduce --fsdp, --expert or --num_stages."
    )
  if devices_per_stage % (fsdp_size * expert_size):
    raise ValueError(
        f"devices_per_stage ({devices_per_stage}) must be divisible by"
        f" fsdp_size x expert_size ({fsdp_size} x {expert_size})."
    )

  tensor_size = devices_per_stage // (fsdp_size * expert_size)
  mesh_shape: list[int] = []
  axis_names: list[str] = []
  for name, size in (
      (FSDP_AXIS, fsdp_size),
      (TENSOR_AXIS, tensor_size),
      (EXPERT_AXIS, expert_size),
  ):
    if size > 1:
      mesh_shape.append(size)
      axis_names.append(name)
  if not axis_names:
    mesh_shape, axis_names = [fsdp_size], [FSDP_AXIS]

  slices = _devices_by_slice(flat_devices)
  if len(slices) > 1:
    stage_device_lists = _partition_slices_into_stages(slices, num_stages)
  else:
    stage_device_lists = _partition_devices_into_stages(flat_devices, num_stages)

  meshes = {}
  for stage, stage_devices in enumerate(stage_device_lists):
    meshes[mesh_name(stage)] = Mesh(stage_devices.reshape(tuple(mesh_shape)), tuple(axis_names))

  return StageTopology(
      meshes=meshes,
      num_stages=num_stages,
      devices_per_stage=devices_per_stage,
      fsdp_size=fsdp_size,
      tensor_size=tensor_size,
      expert_size=expert_size,
  )


def layer_stage_indices(
    num_layers: int,
    num_stages: int,
    num_repeats: int = 1,
    stage_layer_counts: Sequence[int] = (),
) -> list[int]:
  """Returns the logical chunk index for each decoder layer ``0..num_layers-1``.

  Args:
    num_layers: Total number of decoder layers.
    num_stages: Number of pipeline stages.
    num_repeats: Number of times the pipeline is traversed per forward pass.
    stage_layer_counts: Optional per-chunk layer counts. When empty, layers are
      divided uniformly into ``num_repeats * num_stages`` chunks.

  Returns:
    A list of length ``num_layers`` mapping each layer index to its chunk index
    in ``[0, num_repeats * num_stages)``.

  Raises:
    ValueError: If the layer counts or stage counts are invalid.
  """
  if num_stages <= 0:
    raise ValueError(f"num_stages must be positive, got {num_stages}.")
  if num_repeats <= 0:
    raise ValueError(f"num_repeats must be positive, got {num_repeats}.")

  num_chunks = num_repeats * num_stages
  if stage_layer_counts:
    counts = tuple(int(c) for c in stage_layer_counts)
    if len(counts) != num_chunks:
      raise ValueError(
          f"stage_layer_counts ({counts}) must have length"
          f" num_repeats * num_stages ({num_chunks}), got {len(counts)}."
      )
    if any(c <= 0 for c in counts):
      raise ValueError(f"Every entry in stage_layer_counts ({counts}) must be positive.")
    if sum(counts) != num_layers:
      raise ValueError(f"sum(stage_layer_counts) ({sum(counts)}) must equal" f" num_layers ({num_layers}).")
    assignments: list[int] = []
    for chunk_index, count in enumerate(counts):
      assignments.extend([chunk_index] * count)
    return assignments

  if num_layers < num_chunks or num_layers % num_chunks:
    raise ValueError(
        f"num_layers ({num_layers}) must be a positive multiple of"
        f" num_repeats * num_stages ({num_repeats} * {num_stages} ="
        f" {num_chunks})."
    )
  layers_per_chunk = num_layers // num_chunks
  return [layer_index // layers_per_chunk for layer_index in range(num_layers)]


def create_name_to_mesh_assignment(
    num_layers: int,
    num_stages: int,
    num_repeats: int = 1,
    stage_layer_counts: Sequence[int] = (),
) -> dict[str, str]:
  """Assigns every named computation of the model to a stage mesh.

  With ``num_repeats == 1`` the assignment is contiguous: the first stage's
  layers live on ``m0``, the next stage's on ``m1`` and so on. With
  ``num_repeats > 1`` the assignment is *circular* (the layout
  interleaved/looped pipelines need): the model is cut into
  ``num_repeats * num_stages`` chunks which are dealt round-robin across the
  stages, so stage ``i`` owns chunks ``i``, ``i + num_stages``, ...

  The embedding is pinned to the first stage and the unembedding to the last,
  matching the data flow of the forward pass.

  Args:
    num_layers: Total number of decoder layers.
    num_stages: Number of pipeline stages.
    num_repeats: Number of times the pipeline is traversed per forward pass.
    stage_layer_counts: Optional per-stage (or per-chunk) layer counts.

  Returns:
    Mapping from named-computation name to mesh name.

  Raises:
    ValueError: If the layers cannot be divided into valid chunks.
  """
  chunk_indices = layer_stage_indices(
      num_layers=num_layers,
      num_stages=num_stages,
      num_repeats=num_repeats,
      stage_layer_counts=stage_layer_counts,
  )

  name_to_mesh = {}
  for layer_index, chunk_index in enumerate(chunk_indices):
    stage_index = chunk_index % num_stages
    name_to_mesh[layer_name(layer_index)] = mesh_name(stage_index)

  name_to_mesh[EMBED_NAME] = mesh_name(0)
  name_to_mesh[UNEMBED_NAME] = mesh_name(num_stages - 1)
  for stage_index in range(num_stages):
    name_to_mesh[post_stage_name(stage_index)] = mesh_name(stage_index)
    name_to_mesh[pre_stage_name(stage_index)] = mesh_name(stage_index)
  return name_to_mesh


def create_name_to_stage_assignment(
    num_layers: int,
    num_stages: int,
    num_repeats: int = 1,
    stage_layer_counts: Sequence[int] = (),
) -> dict[str, int]:
  """Assigns every named computation to a *pipeline stage index*.

  For circular schedules PartIR needs to know the position of a computation in
  the logical pipeline, which is distinct from the mesh it runs on once a stage
  is visited more than once.

  Args:
    num_layers: Total number of decoder layers.
    num_stages: Number of pipeline stages.
    num_repeats: Number of times the pipeline is traversed per forward pass.
    stage_layer_counts: Optional per-stage (or per-chunk) layer counts.

  Returns:
    Mapping from named-computation name to its logical stage index.
  """
  chunk_indices = layer_stage_indices(
      num_layers=num_layers,
      num_stages=num_stages,
      num_repeats=num_repeats,
      stage_layer_counts=stage_layer_counts,
  )
  num_chunks = num_repeats * num_stages

  name_to_stage = {}
  for layer_index, chunk_index in enumerate(chunk_indices):
    name_to_stage[layer_name(layer_index)] = chunk_index
  name_to_stage[EMBED_NAME] = 0
  name_to_stage[UNEMBED_NAME] = num_chunks - 1
  for stage_index in range(num_stages):
    name_to_stage[post_stage_name(stage_index)] = num_chunks + stage_index
    name_to_stage[pre_stage_name(stage_index)] = num_chunks + num_stages + stage_index
  return name_to_stage


def resolve_devices(
    topology_name: str = "",
    num_slices: int = 1,
) -> Sequence[Any]:
  """Returns the devices to build a topology from.

  Args:
    topology_name: An AOT topology name such as ``v6e:4x4``. When empty (or
      ``cpu``) the locally attached devices are returned instead, which is what
      a real training run wants.
    num_slices: Number of slices for the virtual topology.

  Returns:
    A sequence of (possibly virtual) JAX devices.
  """
  if not topology_name or topology_name == "cpu":
    return jax.devices()

  # Imported lazily: pulling in the AOT topology machinery costs a non-trivial
  # amount of start-up time and is useless during a real run.
  from jax.experimental.topologies import get_topology_desc  # pylint: disable=import-outside-toplevel

  topology = get_topology_desc(
      platform="tpu",
      topology_name=topology_name,
      num_slices=num_slices,
  )
  return topology.devices
