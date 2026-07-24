"""Manages asynchronous backups of JAX array states to pinned host memory."""

import contextlib
import logging
import threading
from typing import Any

import jax
from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import Snapshotter as BaseSnapshotter
import orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter as orbax_snapshotter

_logger = logging.getLogger(__name__)

# Monkeypatch orbax snapshotter is_shardable_array to require mesh attribute on sharding
_original_is_shardable_array = getattr(orbax_snapshotter, "is_shardable_array", lambda x: isinstance(x, jax.Array))

def _fixed_is_shardable_array(x: Any) -> bool:
  return _original_is_shardable_array(x) and hasattr(getattr(x, "sharding", None), "mesh")

orbax_snapshotter.is_shardable_array = _fixed_is_shardable_array

_identity_jit = jax.jit(lambda x: x)
_original_block_until_ready = jax.block_until_ready
_thread_local = threading.local()


def _custom_block_until_ready(x: Any) -> Any:
  """Thread-local override calling identity_jit(x).block_until_ready() during load."""
  if getattr(_thread_local, "use_identity_jit", False):
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

  def load(
      self,
      abstract_state: Any,
      *,
      reset_snapshot_state: bool = True,
  ) -> Any:
    """Move arrays from workers onto TPU devices with thread-local identity_jit validation."""
    with _identity_jit_block_until_ready_context():
      return super().load(abstract_state, reset_snapshot_state=reset_snapshot_state)

  def join(self) -> None:
    self._queue.join()
