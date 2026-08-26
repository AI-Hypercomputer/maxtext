"""Standalone prototype demonstrating explicit core and ICI mesh control for multi-core TPUs (e.g., v7x) with XAOT compilation."""

import functools
from typing import Any, Mapping, Sequence
import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.experimental.topologies import get_topology_desc
import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def get_simulated_v7x_devices(topology_name: str = "v4:2x2x1") -> Sequence[Any]:
  """Simulate a multi-core TPU (such as v7x with 2 cores per chip) via XAOT topology description."""
  topology = get_topology_desc(platform="tpu", topology_name=topology_name)
  return topology.devices


def create_mesh(
    devices: Sequence[Any],
    axis_names: Sequence[str] = ("row", "col"),
    ici_parallelism: Mapping[str, int] | None = None,
    core_parallelism: Mapping[str, int] | None = None,
    **kwargs: Any,
) -> Mesh:
  """Creates a JAX Mesh with explicit control over ICI vs Core parallelism per logical axis.

  Args:
    devices: Sequence of JAX devices (real or simulated via get_topology_desc).
    axis_names: Tuple/List of logical axis names (e.g., ('row', 'col')).
    ici_parallelism: Mapping of axis name -> ICI parallelism degree.
    core_parallelism: Mapping of axis name -> Core parallelism degree.
    **kwargs: Can also accept kwargs in the form `ici_<axis>_parallelism` and
      `core_<axis>_parallelism`.

  Returns:
    A jax.sharding.Mesh where each axis has size ici_parallelism[axis] *
    core_parallelism[axis].
  """
  axis_names = tuple(axis_names)
  ici_map = dict(ici_parallelism or {})
  core_map = dict(core_parallelism or {})

  # Parse additional kwargs like ici_row_parallelism=2, core_row_parallelism=2
  for k, v in kwargs.items():
    if k.startswith("ici_") and k.endswith("_parallelism"):
      axis = k[4:-12]
      ici_map[axis] = v
    elif k.startswith("core_") and k.endswith("_parallelism"):
      axis = k[5:-12]
      core_map[axis] = v

  # Identify hardware topology dimensions: chips and cores per chip
  coords_set = sorted(list(set(tuple(d.coords) for d in devices)))
  cores_set = sorted(list(set(d.core_on_chip for d in devices)))
  num_chips = len(coords_set)
  cores_per_chip = len(cores_set)

  # Validate products
  ici_total = int(np.prod([ici_map.get(ax, 1) for ax in axis_names]))
  core_total = int(np.prod([core_map.get(ax, 1) for ax in axis_names]))

  if ici_total != num_chips:
    raise ValueError(
        f"Total ICI parallelism ({ici_total}) must equal number of chips ({num_chips})."
    )
  if core_total != cores_per_chip:
    raise ValueError(
        f"Total Core parallelism ({core_total}) must equal cores per chip ({cores_per_chip})."
    )

  # Map chips to ICI logical dimensions
  # Pick representative device per chip (core 0)
  chip_rep_devices = [d for d in devices if d.core_on_chip == cores_set[0]]
  ici_shape = tuple(ici_map.get(ax, 1) for ax in axis_names)
  ici_chip_mesh = mesh_utils.create_device_mesh(
      ici_shape, chip_rep_devices, allow_split_physical_axes=True
  )

  # Mapping from (chip_coords, core_id) -> device
  dev_map = {(tuple(d.coords), d.core_on_chip): d for d in devices}

  # Map local cores to core logical dimensions
  core_shape = tuple(core_map.get(ax, 1) for ax in axis_names)
  core_mesh = np.array(cores_set).reshape(core_shape)

  # Combine ICI and Core mapping for each logical coordinate in the final mesh
  final_shape = tuple(
      ici_map.get(ax, 1) * core_map.get(ax, 1) for ax in axis_names
  )
  device_array = np.empty(final_shape, dtype=object)

  for final_idx in np.ndindex(*final_shape):
    ici_idx = []
    core_idx = []
    for dim_i, ax in enumerate(axis_names):
      core_dim = core_map.get(ax, 1)
      ici_idx.append(final_idx[dim_i] // core_dim)
      core_idx.append(final_idx[dim_i] % core_dim)

    rep_dev = ici_chip_mesh[tuple(ici_idx)]
    core_val = core_mesh[tuple(core_idx)]
    device_array[final_idx] = dev_map[(tuple(rep_dev.coords), core_val)]

  return Mesh(device_array, axis_names)


def main():
  print("=" * 60)
  print("1. Simulating multi-core TPU (v7x-like with 2 cores/chip) via XAOT")
  print("=" * 60)
  # v4:2x2x1 simulates 4 chips (2x2x1) with 2 cores per chip = 8 devices
  devices = get_simulated_v7x_devices(topology_name="v4:2x2x1")
  print(f"Total simulated devices: {len(devices)}")
  for d in devices:
    print(f"  Device {d.id}: coords={d.coords}, core_on_chip={d.core_on_chip}")

  print("\n" + "=" * 60)
  print("2. Creating 2D Mesh with explicit Core and ICI controls")
  print("   - 'row': ici=2, core=2 -> total degree = 4")
  print("   - 'col': ici=2, core=1 -> total degree = 2")
  print("=" * 60)

  mesh = create_mesh(
      devices=devices,
      axis_names=("row", "col"),
      ici_row_parallelism=2,
      core_row_parallelism=2,
      ici_col_parallelism=2,
      core_col_parallelism=1,
  )
  print(f"Constructed Mesh: {mesh}")
  print("Physical Device Layout in Mesh [row, col]:")
  for r in range(mesh.devices.shape[0]):
    for c in range(mesh.devices.shape[1]):
      d = mesh.devices[r, c]
      print(f"  mesh[row={r}, col={c}] -> ID {d.id} (coords={d.coords}, core={d.core_on_chip})")

  print("\n" + "=" * 60)
  print("3. Defining All-Gather over 'row' axis (ShardMap + XAOT)")
  print("=" * 60)

  @jax.jit
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=P("row", "col"),
      out_specs=P(None, "col"),
      check_rep=False,
  )
  def all_gather_row(x):
    # x is sharded across ('row', 'col'); all_gather across 'row' produces shape sharded only on 'col'
    return lax.all_gather(x, axis_name="row", axis=0, tiled=True)

  # Abstract input tensor of shape (16, 32)
  x_sharding = NamedSharding(mesh, P("row", "col"))
  x_abstract = jax.ShapeDtypeStruct((16, 32), jnp.float32, sharding=x_sharding)

  # Generate and print JAXPR
  jaxpr = jax.make_jaxpr(all_gather_row)(x_abstract)
  print("\n--- JAXPR ---")
  print(jaxpr)

  # Lower & XAOT Compile to HLO
  lowered = all_gather_row.lower(x_abstract)
  print("\n--- Lowered HLO Module ---")
  print(lowered.as_text("hlo"))

  compiled = lowered.compile()
  print("\n--- XAOT Compilation Successful! ---")
  print(f"Compiled executable: {compiled}")


if __name__ == "__main__":
  main()
