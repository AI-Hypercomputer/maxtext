"""Unit tests and verification for explicit core sharding on TPU v7x."""

import functools
from typing import Any, Sequence
import numpy as np
import jax
from jax._src import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.experimental.topologies import get_topology_desc
import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def get_v7x_devices(topology_name: str = "tpu7x:2x2x2") -> Sequence[Any]:
  """Simulate TPU v7x devices via XAOT topology description.

  `tpu7x:2x2x2` defines an 8-chip grid (2x2x2) with 2 cores per chip = 16 TPU v7x devices.
  """
  topology = get_topology_desc(
      platform="tpu",
      topology_name=topology_name,
      chip_config_name="default",
      chips_per_host_bounds=(2, 2, 1),
      num_slices=1,
      wrap=(False, False, False),
  )
  return topology.devices


def _create_device_mesh_for_nd_torus_v7x(
    physical_mesh: np.ndarray,
    mesh_shape: Sequence[int],
    core_mesh_shape: Sequence[int] | None = None,
    *,
    allow_split_physical_axes: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
  """Assigns logical parallelism axes to physical axes on TPU v7x with explicit core sharding.

  Matches the JAX `_create_device_mesh_for_nd_torus` API structure while adding
  explicit control over the on-chip core dimension on TPU v7x.

  Args:
    physical_mesh: 4D np.ndarray of shape (dim_x, dim_y, dim_z, cores_per_chip)
      representing the physical TPU topology (from `_get_physical_tpu_mesh`).
    mesh_shape: Sequence[int] representing the ICI mesh shape (chips per logical axis).
    core_mesh_shape: Optional Sequence[int] specifying the explicit core sharding
      degrees per logical axis (must have product == 2 for v7x, e.g. [2, 1] or [1, 2]).
    allow_split_physical_axes: bool, whether physical axes can be split.

  Returns:
    device_mesh: np.ndarray of devices shaped in network intensity order.
    assignment_array: 2D array [physical_axis, logical_axis] -> assigned size.
    ordered_axes_indices: tuple of logical axis indices in network intensity order.
  """
  if core_mesh_shape is None:
    core_mesh_shape = (1,) * len(mesh_shape)

  ici_mesh_shape = tuple(mesh_shape)
  core_mesh_shape = tuple(core_mesh_shape)

  if len(ici_mesh_shape) != len(core_mesh_shape):
    raise ValueError(
        f"Length of ici_mesh_shape ({len(ici_mesh_shape)}) must equal"
        f" length of core_mesh_shape ({len(core_mesh_shape)})"
    )
  if physical_mesh.ndim != 4 or physical_mesh.shape[3] != 2:
    raise ValueError(
        "Expected 4D physical mesh with 2 cores per chip for v7x, got"
        f" shape {physical_mesh.shape}"
    )

  num_chips = int(np.prod(physical_mesh.shape[:3]))
  cores_per_chip = physical_mesh.shape[3]

  if int(np.prod(ici_mesh_shape)) != num_chips:
    raise ValueError(
        f"Product of ici_mesh_shape {ici_mesh_shape} ({np.prod(ici_mesh_shape)})"
        f" must match number of chips ({num_chips})"
    )
  if int(np.prod(core_mesh_shape)) != cores_per_chip:
    raise ValueError(
        f"Product of core_mesh_shape {core_mesh_shape} ({np.prod(core_mesh_shape)})"
        f" must match cores per chip ({cores_per_chip})"
    )

  # 1. Order axes by network intensity:
  # - Pure ICI axes (core == 1) have lower network intensity -> placed first.
  # - Core-sharded axes (core > 1) use the fast on-chip interconnect -> placed last (stride 1).
  num_axes = len(ici_mesh_shape)
  perm = tuple(sorted(range(num_axes), key=lambda i: (core_mesh_shape[i] > 1, ici_mesh_shape[i])))

  ordered_ici_shape = tuple(ici_mesh_shape[i] for i in perm)
  ordered_core_shape = tuple(core_mesh_shape[i] for i in perm)
  ordered_total_shape = tuple(ordered_ici_shape[i] * ordered_core_shape[i] for i in range(num_axes))

  # 2. Build device mesh array using JAX mesh_utils
  if allow_split_physical_axes:
    ordered_device_mesh, assignment = mesh_utils._create_device_mesh_for_nd_torus_splitting_axes(
        physical_mesh, ordered_total_shape
    )
  else:
    ordered_device_mesh, assignment = mesh_utils._create_device_mesh_for_nd_torus(
        physical_mesh, ordered_total_shape
    )

  return ordered_device_mesh, assignment, perm


def create_mesh_v7x(
    devices: Sequence[Any],
    axis_names: Sequence[str],
    ici_mesh_shape: Sequence[int],
    core_mesh_shape: Sequence[int] | None = None,
    *,
    allow_split_physical_axes: bool = True,
) -> Mesh:
  """High-level helper to create a JAX Mesh on TPU v7x with explicit core sharding."""
  physical_mesh = mesh_utils._get_physical_tpu_mesh(devices)
  ordered_device_mesh, _, perm = _create_device_mesh_for_nd_torus_v7x(
      physical_mesh,
      ici_mesh_shape,
      core_mesh_shape,
      allow_split_physical_axes=allow_split_physical_axes,
  )
  ordered_axis_names = tuple(axis_names[i] for i in perm)
  return Mesh(ordered_device_mesh, ordered_axis_names)


def test_case_1_cores_sharded_by_ag_dim():
  """Case 1: Cores ARE sharded along the All-Gather dimension ('row').

  Here:
    - 'row': ici=4, core=2 -> degree = 8
    - 'col': ici=2, core=1 -> degree = 2
  All-gathering along 'row' gathers both ICI chips and the 2 cores on each chip.
  Result: Replica groups MUST contain core pairs (e.g. 0 and 1, 2 and 3, etc.).
  """
  print("=" * 80)
  print("CASE 1: Cores ARE sharded by the All-Gather dimension ('row')")
  print("  - ici_mesh_shape = (4, 2)  ['row'=4, 'col'=2]")
  print("  - core_mesh_shape = (2, 1) ['row'=2, 'col'=1]")
  print("  - Total shape = (8, 2)     ['row'=8, 'col'=2]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:2x2x2")
  mesh = create_mesh_v7x(
      devices=devices,
      axis_names=("row", "col"),
      ici_mesh_shape=(4, 2),
      core_mesh_shape=(2, 1),
  )
  print(f"Constructed Mesh: {mesh}")
  print(f"Mesh shape: {mesh.shape}")
  print("Device IDs in Mesh array:")
  print(np.vectorize(lambda d: d.id)(mesh.devices))

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("row", "col"),
      out_specs=P(None, "col"),
      check_rep=False,
  )
  def all_gather_row(x):
    # Local shape (4, 32) -> All-gather across 'row' -> Local shape (32, 32)
    return lax.all_gather(x, axis_name="row", axis=0, tiled=True)

  x_sharding = NamedSharding(mesh, P("row", "col"))
  x_abstract = jax.ShapeDtypeStruct((32, 64), jnp.float32, sharding=x_sharding)

  lowered = all_gather_row.lower(x_abstract)
  hlo_text = lowered.as_text("hlo")

  print("\n--- Lowered HLO Module (Case 1) ---")
  print(hlo_text)

  # Verification: replica_groups must group core pairs together
  assert "replica_groups={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}}" in hlo_text, (
      "Expected replica groups with core pairs {{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}}"
  )
  print("\n[PASS] Case 1: Core pairs (0,1), (2,3), etc. are in the same replica group as expected!")


def test_case_2_cores_not_sharded_by_ag_dim():
  """Case 2: Cores are NOT sharded along the All-Gather dimension ('row').

  Here:
    - 'row': ici=4, core=1 -> degree = 4 (pure ICI)
    - 'col': ici=2, core=2 -> degree = 4 (cores on 'col')
  All-gathering along 'row' gathers across ICI chips only for a fixed core.
  Result: Replica groups do NOT group cores on the same chip together; each group has identical core index.
  """
  print("\n" + "=" * 80)
  print("CASE 2: Cores are NOT sharded by the All-Gather dimension ('row')")
  print("  - ici_mesh_shape = (4, 2)  ['row'=4, 'col'=2]")
  print("  - core_mesh_shape = (1, 2) ['row'=1, 'col'=2]  (cores are on 'col')")
  print("  - Total shape = (4, 4)     ['row'=4, 'col'=4]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:2x2x2")
  mesh = create_mesh_v7x(
      devices=devices,
      axis_names=("row", "col"),
      ici_mesh_shape=(4, 2),
      core_mesh_shape=(1, 2),
  )
  print(f"Constructed Mesh: {mesh}")
  print(f"Mesh shape: {mesh.shape}")
  print("Device IDs in Mesh array:")
  print(np.vectorize(lambda d: d.id)(mesh.devices))

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("row", "col"),
      out_specs=P(None, "col"),
      check_rep=False,
  )
  def all_gather_row(x):
    # Local shape (8, 16) -> All-gather across 'row' -> Local shape (32, 16)
    return lax.all_gather(x, axis_name="row", axis=0, tiled=True)

  x_sharding = NamedSharding(mesh, P("row", "col"))
  x_abstract = jax.ShapeDtypeStruct((32, 64), jnp.float32, sharding=x_sharding)

  lowered = all_gather_row.lower(x_abstract)
  hlo_text = lowered.as_text("hlo")

  print("\n--- Lowered HLO Module (Case 2) ---")
  print(hlo_text)

  # Verification: replica_groups gather across chips for each individual core
  assert "replica_groups={{0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15}}" in hlo_text, (
      "Expected replica groups {{0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15}}"
  )
  print("\n[PASS] Case 2: Cores on the same chip (0 and 1) are in separate replica groups as expected!")


if __name__ == "__main__":
  test_case_1_cores_sharded_by_ag_dim()
  test_case_2_cores_not_sharded_by_ag_dim()
  print("\n" + "=" * 80)
  print("ALL TESTS PASSED SUCCESSFULLY!")
  print("=" * 80)
