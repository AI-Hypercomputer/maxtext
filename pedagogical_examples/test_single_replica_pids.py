import argparse
import jax
import numpy as np
import os
import orbax.checkpoint as ocp
import sys

from etils import epath
from etils import epath
from jax import sharding
from jax import numpy as jnp
from jax._src import sharding_impls
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import type_handlers
from typing import cast


def setup_sharded_pytree(
    replicate_data: bool = False,
    data_dim: int = 2,
    num_slices: int = 1
):
  """Creates a PyTree of sharded arrays for testing."""

  devices = jax.devices()
  num_devices = len(devices)
  devices = np.asarray(devices)

  data_axis_name = 'x'

  mesh_2d = jax.sharding.Mesh(
      devices.reshape(data_dim, num_devices // data_dim),
      (data_axis_name, 'y')
  )
  if replicate_data:
    mesh_axes_2d = jax.sharding.PartitionSpec('y')
  else:
    mesh_axes_2d = jax.sharding.PartitionSpec(data_axis_name, 'y')

  return mesh_2d, mesh_axes_2d


def get_single_replica_pids(sharding):
  """ Get device IDs that correspond to 0th replica_axis_name """
  replica_id_map = sharding_impls.device_replica_id_map(sharding, sharding.mesh.devices.shape)
  # print('replica_id_map', replica_id_map)
  rep0_pids = set([d.process_index for d, rep_id in replica_id_map.items() if rep_id == 0])
  print('===== rep 0 IDS =====', rep0_pids)
  return rep0_pids


def main(args):

  mesh_2d, mesh_axes_2d = setup_sharded_pytree(
    replicate_data=args.replicate_data,
    data_dim=args.data_dim,
    # num_slices=args.num_slices  # can be found with jax.devices though
    )
  sharding = jax.sharding.NamedSharding(mesh_2d, mesh_axes_2d)
  print('***** mesh shape ****', mesh_2d.devices.shape)
  get_single_replica_pids(sharding)

  return


def parser(args):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--replicate-data',
    type=bool,
    default=True,
    help='whether to replicate or shard data axis'
    )

  parser.add_argument(
    '--data-dim',
    type=int,
    default=2,
    help='dimension of data axis'
  )

  parser.add_argument(
    '--num-slices',
    type=int,
    default=1,
    help='dimension of data axis'
  )
  return parser.parse_args(args)

if __name__ == '__main__':
  args = parser(sys.argv[1:])
  main(args)
