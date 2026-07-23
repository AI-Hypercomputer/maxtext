"""Manages asynchronous backups of JAX array states to pinned host memory."""

import logging
import queue
import threading
from typing import Any

from etils import epath
import jax
from orbax.checkpoint.experimental.v1 import training
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types
from pathwaysutils.experimental import concatenate_by_mesh_axis
from pathwaysutils.experimental import split_by_mesh_axis

_logger = logging.getLogger(__name__)

_identity_jit = jax.jit(lambda x: x)


def is_shardable_array(x: Any) -> bool:
  """Returns True if x is a concrete shardable array."""
  return isinstance(x, jax.Array)


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


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0):
    self._latest_snapshot: tuple[tree_types.PyTree, int] | None = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self.replica_axis_index = replica_axis_index

    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    """Processes background snapshot requests from the queue."""
    while True:
      pinned_state, step = self._queue.get()
      try:
        unpacked_state = jax.tree.map(_unpack_if_prng_key, pinned_state)
        _identity_jit(unpacked_state).block_until_ready()
      except (jax.errors.JaxRuntimeError, RuntimeError) as e:
        _logger.warning("Failed to snapshot state at step %d: %s", step, e)
      else:
        with self._lock:
          self._latest_snapshot = (pinned_state, step)
      finally:
        self._queue.task_done()

  def save(self, step: int, state: tree_types.PyTree) -> None:
    """Backs up JAX array states to pinned host memory, asynchronously."""
    if self._queue.full():
      _logger.warning("Snapshotter busy. Skipping snapshot for step %d", step)
      return

    def _pin_leaf(x):
      if not is_shardable_array(x):
        return x
      data = _unpack_if_prng_key(x)
      pinned = jax.device_put(
          data, data.sharding.with_memory_kind("pinned_host")
      )
      return _wrap_if_prng_key(pinned, x)

    pinned_state = jax.tree.map(_pin_leaf, state)
    self._queue.put((pinned_state, step))

  def load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool = True,
  ) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices."""
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
    unpacked_restored = jax.tree.map(_unpack_if_prng_key, unpacked_restored if 'unpacked_restored' in locals() else restored_state)
    _identity_jit(unpacked_restored).block_until_ready()

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state

  def save_pytree(self, step: int, state: Any) -> None:
    self.save(step, state)

  def load_pytree(self, abstract_state: Any, *, reset_snapshot_state: bool = True) -> Any:
    return self.load(abstract_state, reset_snapshot_state=reset_snapshot_state)

  def join(self) -> None:
    self._queue.join()

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step = self._latest_snapshot
    return training.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )
