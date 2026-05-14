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
    while True:
      pinned_state, step = self._queue.get()
      try:
        _logger.info(
            "[*] [Snapshot Thread] Waiting for snapshot at step %d to be ready...",
            step,
        )
        jax.block_until_ready(pinned_state)
        _logger.info(
            "[*] [Snapshot Thread] Snapshot at step %d is ready and secured.",
            step,
        )
        with self._lock:
          self._latest_snapshot = (pinned_state, step)
      except Exception as e:  # pylint: disable=broad-except
        _logger.warning(
            "[*] [Snapshot Thread] Failed to secure snapshot at step %d: %s.",
            step,
            e,
        )
      finally:
        self._queue.task_done()

  def save_pytree(
      self, step: int, state: tree_types.PyTreeOf[jax.Array]
  ) -> None:
    """Move arrays onto CPU worker devices."""
    if self._queue.full():
      _logger.warning("Snapshotter busy. Skipping snapshot for step %d", step)
      return

    pinned_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host"), state
    )

    pinned_state = jax.device_put(state, pinned_shardings)

    self._queue.put((pinned_state, step))

  def load_pytree(
      self,
      abstract_state: tree_types.PyTreeOf[jax.Array],
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

    def is_replica_active(arr):
      try:
        jax.block_until_ready(arr)
        return True
      except jax.errors.JaxRuntimeError as _:
        return False

    def get_active_pytree(x):
      mesh_axis_name = x.sharding.mesh.axis_names[self.replica_axis_index]
      all_replicas = split_by_mesh_axis.split_by_mesh_axis(
          x,
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
      return reconstructed_state

    _logger.info("Restoring from snapshot at step %d...", step)
    pinned_state = jax.tree.map(get_active_pytree, pinned_state)

    # Re-shard on host to the target device mesh
    host_target_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host"), abstract_state
    )

    host_target_state = jax.device_put(
        pinned_state, host_target_shardings
    )

    # Move from host back to device (TPU) memory.
    restored_state = jax.device_put(
        host_target_state, jax.tree.map(lambda x: x.sharding, abstract_state)
    )
    jax.block_until_ready(restored_state)

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state

  def join(self) -> None:
    """Blocks until all snapshots in the queue are ready and secured."""
    self._queue.join()

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step = self._latest_snapshot
    return training.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )
