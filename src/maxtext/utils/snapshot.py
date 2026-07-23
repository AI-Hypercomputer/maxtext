"""Alias for Orbax Snapshotter with MaxText resiliency fixes."""

import logging
import threading
import sys
import jax

from etils import epath
from orbax.checkpoint.experimental.v1 import training
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types
from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import Snapshotter as BaseSnapshotter
from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import is_shardable_array, _unpack_if_prng_key, _wrap_if_prng_key, _is_prng_key

from pathwaysutils.experimental import concatenate_by_mesh_axis
from pathwaysutils.experimental import split_by_mesh_axis


class Snapshotter(BaseSnapshotter):
  def join(self) -> None:
    self._queue.join()

  @jax.jit(donate_argnums=0)
  def identity(array):
    return array

  def load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool = True,
  ) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices.

    Uses `abstract_state.sharding` to properly re-partition onto the new mesh.

    Args:
      abstract_state: An abstract representation of the state, used to provide
        the target shardings for the restored arrays on the TPU devices.
      reset_snapshot_state: If True, clears snapshot history and resets it to
        contain only the returned restored state (in host-pinned memory).

    Returns:
      The restored array state.

    Raises:
      RuntimeError: If no snapshots are available to restore from.
    """
    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    def is_replica_active(array):
      try:
        identity(_unpack_if_prng_key(array)).block_until_ready()
      except jax.errors.JaxRuntimeError:
        return False
      else:
        return True


    def get_active_pytree(x):
      mesh_axis_name = x.sharding.mesh.axis_names[self.replica_axis_index]
      data = _unpack_if_prng_key(x)
      all_replicas = split_by_mesh_axis.split_by_mesh_axis(
          data,
          mesh_axis_name,
      )

      active_replicas = [
          replica for replica in all_replicas if is_replica_active(replica)
      ]

      if not active_replicas:
        raise RuntimeError(
            "No active replicas found."
        )

      reconstructed_state = concatenate_by_mesh_axis.concatenate_by_mesh_axis(
          active_replicas,
          mesh_axis_name,
      )
      return _wrap_if_prng_key(reconstructed_state, x)

    pinned_state = jax.tree.map(
        lambda x: get_active_pytree(x) if is_shardable_array(x) else x,
        pinned_state,
    )

    def _device_put_pinned(x, abs_x):
      if is_shardable_array(x):
        data = _unpack_if_prng_key(x)
        put_x = jax.device_put(
            data, abs_x.sharding.with_memory_kind("pinned_host")
        )
        return _wrap_if_prng_key(put_x, x)
      return x

    # Re-shard on host to the target device mesh
    host_target_state = jax.tree.map(
        _device_put_pinned,
        pinned_state,
        abstract_state,
    )

    def _device_put_to_device(x, abs_x):
      if is_shardable_array(x):
        data = _unpack_if_prng_key(x)
        put_x = jax.device_put(data, abs_x.sharding.with_memory_kind(None))
        return _wrap_if_prng_key(put_x, x)
      return x

    # Move from host back to device (TPU) memory.
    restored_state = jax.tree.map(
        _device_put_to_device,
        host_target_state,
        abstract_state,
    )
    unpacked_restored = jax.tree.map(_unpack_if_prng_key, restored_state)
    jax.block_until_ready(unpacked_restored)

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state


