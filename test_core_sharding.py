"""Unit tests and verification for explicit core sharding across 1D, 2D, 3D, and 4D meshes on TPU v7x."""

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

  - `tpu7x:2x2x2`: 8 chips (2x2x2) x 2 cores/chip = 16 TPU v7x devices.
  - `tpu7x:4x4x4`: 64 chips (4x4x4) x 2 cores/chip = 128 TPU v7x devices (v7x cube).
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
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
  """Assigns logical parallelism axes to physical axes on TPU v7x with explicit core sharding.

  Supports 1D, 2D, 3D, and 4D logical shardings. Core sharding can either be
  shared with an ICI axis (e.g. ici=4, core=2 -> size 8) or assigned to a unique
  core-only axis not shared with any ICI dimension (e.g. ici=1, core=2 -> size 2).

  Args:
    physical_mesh: 4D np.ndarray of shape (dim_x, dim_y, dim_z, cores_per_chip)
      representing the physical TPU topology (from `_get_physical_tpu_mesh`).
    mesh_shape: Sequence[int] representing the ICI mesh shape (chips per logical axis).
    core_mesh_shape: Optional Sequence[int] specifying the explicit core sharding
      degrees per logical axis (product must equal 2 for v7x).
    allow_split_physical_axes: bool, whether physical axes can be split.

  Returns:
    device_mesh: np.ndarray of devices shaped in network intensity order.
    assignment_array: 2D array [physical_axis, logical_axis] -> assigned size.
    perm: tuple of logical axis indices in network intensity order.
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

  # Order axes by network intensity:
  # - Pure ICI axes (core == 1) have lower network intensity -> placed first.
  # - Core-sharded axes (core > 1) use the fast on-chip interconnect -> placed last (stride 1).
  num_axes = len(ici_mesh_shape)
  perm = tuple(sorted(range(num_axes), key=lambda i: (core_mesh_shape[i] > 1, ici_mesh_shape[i])))

  ordered_ici_shape = tuple(ici_mesh_shape[i] for i in perm)
  ordered_core_shape = tuple(core_mesh_shape[i] for i in perm)
  ordered_total_shape = tuple(ordered_ici_shape[i] * ordered_core_shape[i] for i in range(num_axes))

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


# ==============================================================================
# 1D SHARDING TESTS
# ==============================================================================

def test_1d_shared_core_and_ici():
  """1D Sharding: 128 devices on a v7x cube (64 chips x 2 cores/chip).

  - 'data': ici=64, core=2 -> degree = 128
  All-gather across 'data' gathers all 128 devices into 1 replica group.
  """
  print("\n" + "=" * 80)
  print("TEST 1D: Shared Core and ICI on 1D Mesh (128 devices)")
  print("  - ici_mesh_shape = (64,)   ['data'=64]")
  print("  - core_mesh_shape = (2,)   ['data'=2]")
  print("  - Total shape = (128,)     ['data'=128]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:4x4x4")
  mesh = create_mesh_v7x(devices, ("data",), (64,), (2,))
  print(f"Constructed Mesh: {mesh}")

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("data"),
      out_specs=P(None),
      check_rep=False,
  )
  def ag_1d(x):
    return lax.all_gather(x, axis_name="data", axis=0, tiled=True)

  x = jax.ShapeDtypeStruct((128, 16), jnp.float32, sharding=NamedSharding(mesh, P("data")))
  lowered = ag_1d.lower(x)
  hlo_text = lowered.as_text("hlo")
  print("\n--- Lowered HLO (1D Shared) ---")
  for line in hlo_text.splitlines():
    if "all-gather" in line or "replica_groups" in line:
      print(line)

  assert "replica_groups={{0,1,2,3,4,5,6,7,8,9,10" in hlo_text
  print("[PASS] 1D Shared Core + ICI gathers all 128 devices into 1 replica group.")


# ==============================================================================
# 2D SHARDING TESTS
# ==============================================================================

def test_2d_shared_core_and_ici():
  """2D Sharding: Core shared with 'row' on 16 devices (8 chips x 2 cores).

  - 'row': ici=4, core=2 -> degree = 8
  - 'col': ici=2, core=1 -> degree = 2
  All-gather over 'row' gathers core pairs together.
  """
  print("\n" + "=" * 80)
  print("TEST 2D (Case A): Core shared with 'row' on 2D Mesh (16 devices)")
  print("  - ici_mesh_shape = (4, 2)  ['row'=4, 'col'=2]")
  print("  - core_mesh_shape = (2, 1) ['row'=2, 'col'=1]")
  print("  - Total shape = (8, 2)     ['row'=8, 'col'=2]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:2x2x2")
  mesh = create_mesh_v7x(devices, ("row", "col"), (4, 2), (2, 1))
  print(f"Constructed Mesh: {mesh}")

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("row", "col"),
      out_specs=P(None, "col"),
      check_rep=False,
  )
  def ag_2d_shared(x):
    return lax.all_gather(x, axis_name="row", axis=0, tiled=True)

  x = jax.ShapeDtypeStruct((32, 64), jnp.float32, sharding=NamedSharding(mesh, P("row", "col")))
  lowered = ag_2d_shared.lower(x)
  hlo_text = lowered.as_text("hlo")
  print("\n--- Lowered HLO (2D Shared) ---")
  for line in hlo_text.splitlines():
    if "all-gather" in line or "replica_groups" in line:
      print(line)

  assert "replica_groups={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}}" in hlo_text
  print("[PASS] 2D Shared Core: Core pairs are in the same replica group.")


def test_2d_unique_core_axis():
  """2D Sharding: Core on a unique axis 'core_axis' not shared with ICI.

  - 'core_axis': ici=1, core=2 -> degree = 2 (pure core axis!)
  - 'ici_axis':  ici=8, core=1 -> degree = 8 (pure ICI axis!)
  All-gather over 'core_axis' gathers strictly the 2 cores of each chip (size 2 groups).
  """
  print("\n" + "=" * 80)
  print("TEST 2D (Case B): Unique Core axis 'core_axis' not shared with ICI (16 devices)")
  print("  - ici_mesh_shape = (1, 8)  ['core_axis'=1, 'ici_axis'=8]")
  print("  - core_mesh_shape = (2, 1) ['core_axis'=2, 'ici_axis'=1]")
  print("  - Total shape = (2, 8)     ['core_axis'=2, 'ici_axis'=8]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:2x2x2")
  mesh = create_mesh_v7x(devices, ("core_axis", "ici_axis"), (1, 8), (2, 1))
  print(f"Constructed Mesh: {mesh}")

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("core_axis", "ici_axis"),
      out_specs=P(None, "ici_axis"),
      check_rep=False,
  )
  def ag_2d_unique(x):
    return lax.all_gather(x, axis_name="core_axis", axis=0, tiled=True)

  x = jax.ShapeDtypeStruct((16, 64), jnp.float32, sharding=NamedSharding(mesh, P("core_axis", "ici_axis")))
  lowered = ag_2d_unique.lower(x)
  hlo_text = lowered.as_text("hlo")
  print("\n--- Lowered HLO (2D Unique Core Axis) ---")
  for line in hlo_text.splitlines():
    if "all-gather" in line or "replica_groups" in line:
      print(line)

  assert "replica_groups={{0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15}}" in hlo_text
  print("[PASS] 2D Unique Core: Gathers strictly within on-chip core pairs.")


# ==============================================================================
# 3D SHARDING TESTS
# ==============================================================================

def test_3d_shared_core_and_ici():
  """3D Sharding: Core shared with 'x' on 128 devices (64 chips x 2 cores).

  - 'x': ici=4, core=2 -> degree = 8 (2x over cores and 4x over ICI)
  - 'y': ici=4, core=1 -> degree = 4 (pure ICI)
  - 'z': ici=4, core=1 -> degree = 4 (pure ICI)
  All-gather over 'x' gathers 4 chips x 2 cores = 8 devices per replica group.
  """
  print("\n" + "=" * 80)
  print("TEST 3D (Case A): Core shared with 'x' (8x4x4 sharding on 128-device cube)")
  print("  - ici_mesh_shape = (4, 4, 4)  ['x'=4, 'y'=4, 'z'=4]")
  print("  - core_mesh_shape = (2, 1, 1) ['x'=2, 'y'=1, 'z'=1]")
  print("  - Total shape = (8, 4, 4)     ['x'=8, 'y'=4, 'z'=4]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:4x4x4")
  mesh = create_mesh_v7x(devices, ("x", "y", "z"), (4, 4, 4), (2, 1, 1))
  print(f"Constructed Mesh: {mesh}")

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("x", "y", "z"),
      out_specs=P(None, "y", "z"),
      check_rep=False,
  )
  def ag_3d_shared(t):
    return lax.all_gather(t, axis_name="x", axis=0, tiled=True)

  x = jax.ShapeDtypeStruct((16, 16, 16), jnp.float32, sharding=NamedSharding(mesh, P("x", "y", "z")))
  lowered = ag_3d_shared.lower(x)
  hlo_text = lowered.as_text("hlo")
  print("\n--- Lowered HLO (3D Shared) ---")
  for line in hlo_text.splitlines():
    if "all-gather" in line or "replica_groups" in line:
      print(line)

  # Group 0 should contain 4 core pairs: (0, 1), (2, 3), (4, 5), (6, 7)
  assert "replica_groups={{0,1,2,3,4,5,6,7}" in hlo_text
  print("[PASS] 3D Shared Core: Gathers 8 devices per group containing core pairs.")


def test_3d_unique_core_axis():
  """3D Sharding: Core on a unique axis 'core_axis' on 128 devices (64 chips x 2 cores).

  - 'core_axis': ici=1, core=2 -> degree = 2 (pure core axis!)
  - 'plane_x':   ici=8, core=1 -> degree = 8 (pure ICI)
  - 'plane_y':   ici=8, core=1 -> degree = 8 (pure ICI)
  All-gather over 'core_axis' gathers strictly within on-chip core pairs (64 groups of 2).
  """
  print("\n" + "=" * 80)
  print("TEST 3D (Case B): Unique Core axis on 3D Mesh (2x8x8 sharding on 128 devices)")
  print("  - ici_mesh_shape = (1, 8, 8)  ['core_axis'=1, 'plane_x'=8, 'plane_y'=8]")
  print("  - core_mesh_shape = (2, 1, 1) ['core_axis'=2, 'plane_x'=1, 'plane_y'=1]")
  print("  - Total shape = (2, 8, 8)     ['core_axis'=2, 'plane_x'=8, 'plane_y'=8]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:4x4x4")
  mesh = create_mesh_v7x(devices, ("core_axis", "plane_x", "plane_y"), (1, 8, 8), (2, 1, 1))
  print(f"Constructed Mesh: {mesh}")

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("core_axis", "plane_x", "plane_y"),
      out_specs=P(None, "plane_x", "plane_y"),
      check_rep=False,
  )
  def ag_3d_unique(t):
    return lax.all_gather(t, axis_name="core_axis", axis=0, tiled=True)

  x = jax.ShapeDtypeStruct((16, 16, 16), jnp.float32, sharding=NamedSharding(mesh, P("core_axis", "plane_x", "plane_y")))
  lowered = ag_3d_unique.lower(x)
  hlo_text = lowered.as_text("hlo")
  print("\n--- Lowered HLO (3D Unique Core Axis) ---")
  for line in hlo_text.splitlines():
    if "all-gather" in line or "replica_groups" in line:
      print(line)

  assert "replica_groups={{0,1},{2,3},{4,5}" in hlo_text
  print("[PASS] 3D Unique Core: 64 replica groups of 2 devices (on-chip core pairs).")


# ==============================================================================
# 4D SHARDING TESTS
# ==============================================================================

def test_4d_unique_core_axis():
  """4D Sharding: Unique 2x4x4x4 sharding on 128-device cube.

  - 'core_axis': ici=1, core=2 -> degree = 2 (pure core axis!)
  - 'x':         ici=4, core=1 -> degree = 4 (pure ICI)
  - 'y':         ici=4, core=1 -> degree = 4 (pure ICI)
  - 'z':         ici=4, core=1 -> degree = 4 (pure ICI)
  All-gather over 'core_axis' gathers strictly within on-chip core pairs (64 groups of 2).
  """
  print("\n" + "=" * 80)
  print("TEST 4D (Case A): 4D sharding with unique core axis (2x4x4x4 on 128-device cube)")
  print("  - ici_mesh_shape = (1, 4, 4, 4)  ['core_axis'=1, 'x'=4, 'y'=4, 'z'=4]")
  print("  - core_mesh_shape = (2, 1, 1, 1) ['core_axis'=2, 'x'=1, 'y'=1, 'z'=1]")
  print("  - Total shape = (2, 4, 4, 4)     ['core_axis'=2, 'x'=4, 'y'=4, 'z'=4]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:4x4x4")
  mesh = create_mesh_v7x(devices, ("core_axis", "x", "y", "z"), (1, 4, 4, 4), (2, 1, 1, 1))
  print(f"Constructed Mesh: {mesh}")

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("core_axis", "x", "y", "z"),
      out_specs=P(None, "x", "y", "z"),
      check_rep=False,
  )
  def ag_4d_unique(t):
    return lax.all_gather(t, axis_name="core_axis", axis=0, tiled=True)

  x = jax.ShapeDtypeStruct((4, 8, 8, 8), jnp.float32, sharding=NamedSharding(mesh, P("core_axis", "x", "y", "z")))
  lowered = ag_4d_unique.lower(x)
  hlo_text = lowered.as_text("hlo")
  print("\n--- Lowered HLO (4D Unique Core Axis) ---")
  for line in hlo_text.splitlines():
    if "all-gather" in line or "replica_groups" in line:
      print(line)

  assert "replica_groups={{0,1},{2,3},{4,5}" in hlo_text
  print("[PASS] 4D Unique Core: Gathers strictly within on-chip core pairs across all 64 chips.")


def test_4d_shared_core_and_ici():
  """4D Sharding: Core shared with 'expert' axis in 4D (data, fsdp, tensor, expert).

  - 'data':   ici=2, core=1 -> degree = 2
  - 'fsdp':   ici=4, core=1 -> degree = 4
  - 'tensor': ici=4, core=1 -> degree = 4
  - 'expert': ici=2, core=2 -> degree = 4 (expert has 2x ICI and 2x cores!)
  All-gather over 'expert' gathers 2 chips x 2 cores = 4 devices per group (32 groups of 4).
  """
  print("\n" + "=" * 80)
  print("TEST 4D (Case B): Core shared with 'expert' in 4D Mesh (2x4x4x4 on 128 devices)")
  print("  - ici_mesh_shape = (2, 4, 4, 2)  ['data'=2, 'fsdp'=4, 'tensor'=4, 'expert'=2]")
  print("  - core_mesh_shape = (1, 1, 1, 2) ['data'=1, 'fsdp'=1, 'tensor'=1, 'expert'=2]")
  print("  - Total shape = (2, 4, 4, 4)     ['data'=2, 'fsdp'=4, 'tensor'=4, 'expert'=4]")
  print("=" * 80)

  devices = get_v7x_devices("tpu7x:4x4x4")
  mesh = create_mesh_v7x(devices, ("data", "fsdp", "tensor", "expert"), (2, 4, 4, 2), (1, 1, 1, 2))
  print(f"Constructed Mesh: {mesh}")

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("data", "fsdp", "tensor", "expert"),
      out_specs=P("data", "fsdp", "tensor", None),
      check_rep=False,
  )
  def ag_4d_shared(t):
    return lax.all_gather(t, axis_name="expert", axis=3, tiled=True)

  x = jax.ShapeDtypeStruct((4, 8, 8, 8), jnp.float32, sharding=NamedSharding(mesh, P("data", "fsdp", "tensor", "expert")))
  lowered = ag_4d_shared.lower(x)
  hlo_text = lowered.as_text("hlo")
  print("\n--- Lowered HLO (4D Shared Core Axis) ---")
  for line in hlo_text.splitlines():
    if "all-gather" in line or "replica_groups" in line:
      print(line)

  # Replica groups should gather 4 devices containing 2 core pairs: e.g. {0, 1, 2, 3}
  assert "replica_groups={{0,1,2,3},{4,5,6,7}" in hlo_text
  print("[PASS] 4D Shared Core: Gathers 4 devices per group containing core pairs.")


def main():
  print("\n" + "#" * 80)
  print("RUNNING ALL 1D, 2D, 3D, 4D TPU v7x CORE SHARDING TESTS")
  print("#" * 80)

  test_1d_shared_core_and_ici()
  test_2d_shared_core_and_ici()
  test_2d_unique_core_axis()
  test_3d_shared_core_and_ici()
  test_3d_unique_core_axis()
  test_4d_unique_core_axis()
  test_4d_shared_core_and_ici()

  print("\n" + "=" * 80)
  print("ALL 1D, 2D, 3D, AND 4D TESTS PASSED SUCCESSFULLY!")
  print("=" * 80)


if __name__ == "__main__":
  main()
