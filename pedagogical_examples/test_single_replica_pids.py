import argparse
import jax
import numpy as np
import string
import os
import orbax.checkpoint as ocp
import sys

from etils import epath
from etils import epath
from functools import reduce
from jax import sharding
from jax import numpy as jnp
from jax._src import sharding_impls
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import type_handlers
from typing import cast

LETTERS = list(string.ascii_lowercase)

def setup_sharded_pytree(
    shape
):
  """Creates a mesh with given shapes."""

  devices = jax.devices()
  num_devices = len(devices)
  assert num_devices == reduce(lambda x, y: x * y, shape, 1)

  devices = np.asarray(devices)

  dim = len(shape)
  # import pdb; pdb.set_trace()
  mesh = jax.sharding.Mesh(
      devices.reshape(shape),
      LETTERS[-dim:]
  )
  return mesh


def is_single_data_replica(rep0_pids):
  return jax.local_devices()[0].process_index in rep0_pids


def get_replica_pids(rep_id, mesh):
  replica_devices = np.take(mesh.devices, rep_id, axis=0).flatten()
  # replica_devices_flattened = replica_devices.flatten()
  return set([d.process_index for d in replica_devices])


def main(args):

  mesh = setup_sharded_pytree(
    shape=args.mesh_shape,
    )

  print('***** mesh shape ****', mesh.devices.shape)
  replica0_pids = get_replica_pids(0, mesh)

  print('--- replica0_pids ---', replica0_pids)
  print(f'host id {jax.process_index()} is in replica0',
        # is_single_data_replica(replica0_devices)
        is_single_data_replica(replica0_pids)
        )

  return


def parser(args):
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--mesh-shape',
    type=int,
    nargs="+",
    default=None,
    help='dimension of data mesh'
  )

  return parser.parse_args(args)

if __name__ == '__main__':
  args = parser(sys.argv[1:])
  main(args)
