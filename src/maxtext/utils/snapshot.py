"""Manages asynchronous backups of JAX array states to pinned host memory."""

import logging
from typing import Any
from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import Snapshotter as BaseSnapshotter

_logger = logging.getLogger(__name__)


class Snapshotter(BaseSnapshotter):
  """Extends Orbax Snapshotter with convenience methods for MaxText training."""

  def save_pytree(self, step: int, state: Any) -> None:
    self.save(step, state)

  def load_pytree(self, abstract_state: Any, *, reset_snapshot_state: bool = True) -> Any:
    return self.load(abstract_state, reset_snapshot_state=reset_snapshot_state)

  def join(self) -> None:
    self._queue.join()
