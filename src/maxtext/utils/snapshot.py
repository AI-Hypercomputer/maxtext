"""Manages asynchronous backups of JAX array states to pinned host memory."""

import logging
from typing import Any

import jax
from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import Snapshotter as BaseSnapshotter
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types
from pathwaysutils.experimental import concatenate_by_mesh_axis
from pathwaysutils.experimental import split_by_mesh_axis

_logger = logging.getLogger(__name__)

_identity_jit = jax.jit(lambda x: x)


def _is_prng_key(x: Any) -> bool:
  """Returns True if x has a JAX PRNGKeyArray dtype."""
  return (
      hasattr(x, "dtype")
      and hasattr(x, "shape")
      and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
  )


def _unpack_if_prng_key(x: Any) -> Any:
  """Extracts the underlying key data buffer if x is a PRNGKeyArray."""
  return jax.random.key_data(x) if _is_prng_key(x) else x


def _wrap_if_prng_key(x: Any, orig_x: Any) -> Any:
  """Wraps raw array x into a PRNGKeyArray if orig_x was a PRNGKeyArray."""
  if _is_prng_key(orig_x):
    return jax.random.wrap_key_data(x, dtype=orig_x.dtype)
  return x


def is_shardable_array(x: Any) -> bool:
  """Returns True if x is a concrete shardable array."""
  return isinstance(x, jax.Array)


class Snapshotter(BaseSnapshotter):
  """Extends Orbax Snapshotter using identity_jit for active replica validation."""

  def load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool = True,
  ) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices with identity_jit replica validation."""
    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    def is_replica_active(arr):
      try:
        data = _unpack_if_prng_key(arr)
        _identity_jit(data).block_until_ready()
        return True
      except (jax.errors.JaxRuntimeError, RuntimeError) as _:
        return False

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
        raise RuntimeError("No active replicas found.")

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

    restored_state = jax.tree.map(
        _device_put_to_device,
        host_target_state,
        abstract_state,
    )
    unpacked_restored = jax.tree.map(_unpack_if_prng_key, restored_state)
    _identity_jit(unpacked_restored).block_until_ready()

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state

  def join(self) -> None:
    self._queue.join()
