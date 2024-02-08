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

  # mesh_2d = jax.sharding.Mesh(
  #     devices.reshape(data_dim, num_devices // data_dim),
  #     (data_axis_name, 'y')
  # )
  mesh_4d = jax.sharding.Mesh(
      devices.reshape(2, 4, 4, 4),
      (data_axis_name, 'y', 'z', 'w')
  )

  if replicate_data:
    mesh_axes_2d = jax.sharding.PartitionSpec('y')
  else:
    mesh_axes_2d = jax.sharding.PartitionSpec(data_axis_name, 'y')

  # return mesh_2d, mesh_axes_2d
  return mesh_4d, mesh_axes_2d


# def get_single_replica_pids(sharding):
#   """ Get device IDs that correspond to 0th replica_axis_name """
#   replica_id_map = sharding_impls.device_replica_id_map(sharding, sharding.mesh.devices.shape)
#   # print('replica_id_map', replica_id_map)
#   rep0_pids = set([d.process_index for d, rep_id in replica_id_map.items() if rep_id == 0])
#   print('===== rep 0 IDS =====', rep0_pids)
#   return rep0_pids

# def is_single_data_replica(rep0_pids):
#   return jax.local_devices()[0].process_index in rep0_pids

def is_single_data_replica(rep0_dev_ids):
  for d in jax.local_devices():
    if d.id in rep0_dev_ids:
      print(f'I am host {jax.process_index()} in replica 0')
      return True
      # rep0_pids.append(jax.process_index())
  print(f'host {jax.process_index()} NOT in replica 0')
  return False

def main(args):

  mesh_2d, mesh_axes_2d = setup_sharded_pytree(
    replicate_data=args.replicate_data,
    data_dim=args.data_dim,
    # num_slices=args.num_slices  # can be found with jax.devices though
    )
  print('***** mesh shape ****', mesh_2d.devices.shape)

  replica0_devices = np.take(mesh_2d.devices, 0, axis=0)
  replica0_devices_flattened = replica0_devices.flatten()
  print('replica0_devices shape',
        replica0_devices.shape,
        type(replica0_devices)
        )
  print('replica0_devices_flattened shape',
        replica0_devices_flattened.shape,
        type(replica0_devices_flattened)
        )

  print('0th element', replica0_devices[0][0][0])
  print('0th element flattened', replica0_devices_flattened[0])
  # devs_flatten = [dev_id for dev_id in np.nditer(replica0_devices, flags=['refs_ok'])]
  # print('replica0_devices shape', replica0_devices.shape, type(replica0_devices))
  # print('devs_flatten shape', devs_flatten)
  # flatten_ids = [d.id for d in devs_flatten]
  # print(flatten_ids)
  # replica0_pids = set([d.process_index for d in replica0_devices.tolist()])
  # replica0_pids = set([d.process_index for d in
  #                      np.nditer(replica0_devices, flags=['refs_ok'])])
  # print('====== replica0 host IDs', replica0_pids)
  # print('______ replica0 iter host IDs', replica0_pids_iter)
  # print(f'host id {jax.process_index()} is in replica0',
  #       is_single_data_replica(devs_flatten)
  #       )

  # get_single_replica_pids(sharding)

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
