"""Prototype demonstrating explicit core and ICI mesh control for multi-core TPU v7x with XAOT compilation."""

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


def get_v7x_devices(topology_name: str = "tpu7x:4x4x4") -> Sequence[Any]:
  """Simulate TPU v7x multi-core devices (2 cores per chip) via XAOT topology description.

  For example:
  - `tpu7x:2x2x2`: 8 chips (2x2x2) x 2 cores/chip = 16 total v7x devices.
  - `tpu7x:4x4x4`: 64 chips (4x4x4) x 2 cores/chip = 128 total v7x devices (v7x cube).
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
  """Creates a JAX Mesh on TPU v7x with explicit core sharding."""
  physical_mesh = mesh_utils._get_physical_tpu_mesh(devices)
  ordered_device_mesh, _, perm = _create_device_mesh_for_nd_torus_v7x(
      physical_mesh,
      ici_mesh_shape,
      core_mesh_shape,
      allow_split_physical_axes=allow_split_physical_axes,
  )
  ordered_axis_names = tuple(axis_names[i] for i in perm)
  return Mesh(ordered_device_mesh, ordered_axis_names)


def main():
  print("=" * 80)
  print("Simulating TPU v7x cube (topology: tpu7x:4x4x4 -> 128 devices) via XAOT")
  print("=" * 80)
  devices = get_v7x_devices("tpu7x:4x4x4")
  print(f"Total simulated devices: {len(devices)} (Device kind: {devices[0].device_kind})")

  print("\n" + "=" * 80)
  print("Creating 4D Mesh: unique core axis not shared with ICI (2x4x4x4)")
  print("  - 'core_axis': ici=1, core=2 -> degree = 2 (pure core axis)")
  print("  - 'x':         ici=4, core=1 -> degree = 4")
  print("  - 'y':         ici=4, core=1 -> degree = 4")
  print("  - 'z':         ici=4, core=1 -> degree = 4")
  print("=" * 80)

  mesh = create_mesh_v7x(
      devices=devices,
      axis_names=("core_axis", "x", "y", "z"),
      ici_mesh_shape=(1, 4, 4, 4),
      core_mesh_shape=(2, 1, 1, 1),
  )
  print(f"Constructed Mesh: {mesh}")

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("core_axis", "x", "y", "z"),
      out_specs=P(None, "x", "y", "z"),
      check_rep=False,
  )
  def all_gather_core_axis(x):
    return lax.all_gather(x, axis_name="core_axis", axis=0, tiled=True)

  x_sharding = NamedSharding(mesh, P("core_axis", "x", "y", "z"))
  x_abstract = jax.ShapeDtypeStruct((2, 4, 4, 4), jnp.float32, sharding=x_sharding)

  lowered = all_gather_core_axis.lower(x_abstract)
  print("\n--- Lowered HLO Module ---")
  for line in lowered.as_text("hlo").splitlines():
    if "all-gather" in line or "replica_groups" in line:
      print(line)

  compiled = lowered.compile()
  print("\n--- XAOT Compilation Successful! ---")
  print(f"Compiled executable: {compiled}")


if __name__ == "__main__":
  main()
