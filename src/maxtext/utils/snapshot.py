"""Manages asynchronous backups of JAX array states to pinned host memory."""

import contextlib
import logging
import threading
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import Snapshotter as BaseSnapshotter

_logger = logging.getLogger(__name__)

_identity_jit = jax.jit(lambda x: x)
_original_block_until_ready = jax.block_until_ready
_thread_local = threading.local()


def get_live_devices() -> list[Any]:
  """Constructs list containing one device per live device."""
  return [d for d in jax.devices() if d is not None]


def _ensure_mesh_sharding(x: Any) -> Any:
  """Shards single device arrays onto a mesh constructed with one device per live device."""
  if isinstance(x, jax.Array) and not hasattr(x.sharding, "mesh"):
    live_devs = get_live_devices()
    mesh = Mesh(np.array(live_devs), ("replica",))
    sharding = NamedSharding(mesh, PartitionSpec("replica"))
    return jax.device_put(x, sharding)
  return x


def shard_single_device_arrays(pytree: Any) -> Any:
  """Converts single device arrays in a PyTree to NamedSharding on a one-device-per-live-device mesh."""
  return jax.tree.map(_ensure_mesh_sharding, pytree)


def _custom_block_until_ready(x: Any) -> Any:
  """Thread-local override calling identity_jit(x).block_until_ready() during load."""
  if hasattr(_thread_local, "use_identity_jit") and _thread_local.use_identity_jit:
    try:
      return _identity_jit(x).block_until_ready()
    except Exception:
      return _original_block_until_ready(x)
  return _original_block_until_ready(x)


# Install thread-safe wrapper once at module import
jax.block_until_ready = _custom_block_until_ready


@contextlib.contextmanager
def _identity_jit_block_until_ready_context():
  """Context manager enabling thread-local identity_jit for block_until_ready."""
  _thread_local.use_identity_jit = True
  try:
    yield
  finally:
    _thread_local.use_identity_jit = False


class Snapshotter(BaseSnapshotter):
  """Extends Orbax Snapshotter using thread-safe identity_jit during load."""

  def save(self, state: Any, step: int | None = None) -> bool:
    """Saves state after ensuring all single device arrays are sharded onto one-device-per-live-device mesh."""
    state = shard_single_device_arrays(state)
    return super().save(state, step=step)

  def load(
      self,
      abstract_state: Any,
      *,
      reset_snapshot_state: bool = True,
  ) -> Any:
    """Move arrays from workers onto TPU devices with thread-local identity_jit validation."""
    abstract_state = shard_single_device_arrays(abstract_state)
    with _identity_jit_block_until_ready_context():
      return super().load(
          abstract_state, reset_snapshot_state=reset_snapshot_state
      )

  def join(self) -> None:
    self._queue.join()
