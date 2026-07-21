# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grain utility functions for checkpointing."""

import asyncio
import dataclasses
import json
from typing import Any, Optional

from etils import epath
import grain
from grain import experimental as grain_experimental
from grain import python
import jax
from maxtext.input_pipeline import multihost_dataloading
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import v1 as ocp_v1


PyGrainCheckpointHandler = python.PyGrainCheckpointHandler
Composite = ocp.args.Composite
RemoteIteratorWrapper = multihost_dataloading.RemoteIteratorWrapper

ElasticIterator = grain_experimental.ElasticIterator


class GrainCheckpointHandler(PyGrainCheckpointHandler, ocp.CheckpointHandler):
  """A CheckpointHandler that allows specifying process_index and process_count."""

  def save(
      self,
      directory: epath.Path,
      # `item` is for backwards compatibility with older Orbax API, see
      # https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html.
      item: Optional[Any] = None,
      args: Any = None,
  ):
    """Saves the given iterator to the checkpoint in `directory`."""
    item = item or args.item  # pytype:disable=attribute-error

    # RemoteIteratorWrapper handles checkpointing via colocated python
    if isinstance(item, RemoteIteratorWrapper):
      step = int(directory.parent.name)
      item.save_state(step)
      return

    # ElasticIterator state is a single global scalar shared by all shards,
    # so we write one fixed `process_0.json` from process 0 only. This file
    # layout survives changes in `jax.process_count()`.
    if isinstance(item, ElasticIterator):
      if jax.process_index() == 0:
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / "process_0.json"
        filename.write_text(json.dumps(item.get_state(), indent=4))
      return

    def save_single_process(item, process_index, process_count):
      filename = directory / f"process_{process_index}-of-{process_count}.json"
      if isinstance(item, grain.DatasetIterator):
        state = json.dumps(item.get_state(), indent=4)
      else:
        state = item.get_state().decode()
      filename.write_text(state)

    if isinstance(item, list):
      for local_iterator, process_index, process_count in item:
        save_single_process(local_iterator, process_index, process_count)
    else:
      process_index, process_count = jax.process_index(), jax.process_count()
      save_single_process(item, process_index, process_count)

  def restore(
      self,
      directory: epath.Path,
      item: Optional[Any] = None,
      args: Any = None,
  ) -> Any:
    """Restores the given iterator from the checkpoint in `directory`."""
    item = item or args.item
    process_index = getattr(args, "process_index", None)
    process_count = getattr(args, "process_count", None)

    # In Pathways + colocated_python environment, RemoteIteratorWrapper handles checkpointing
    if isinstance(item, RemoteIteratorWrapper):
      step = int(directory.parent.name)
      item.restore_state(step)
      return item

    # McJax and Pathways through controller cases
    # ElasticIterator: every process reads the same shared `process_0.json`.
    if isinstance(item, ElasticIterator):
      filename = directory / "process_0.json"
      if not filename.exists():
        raise ValueError(f"File {filename} does not exist.")
      item.set_state(json.loads(filename.read_text()))
      return item

    def restore_single_process(item, process_index, process_count):
      filename = directory / f"process_{process_index}-of-{process_count}.json"
      if not filename.exists():
        raise ValueError(f"File {filename} does not exist.")
      state = filename.read_text()
      if isinstance(item, grain.DatasetIterator):
        state = json.loads(state)
      else:
        state = state.encode()
      item.set_state(state)
      return item

    if isinstance(item, list):
      restored_items = []
      for data_iter, process_idx in zip(item, process_index):  # pyrefly: ignore[bad-argument-type]
        restored_items.append(restore_single_process(data_iter, process_idx, process_count))
      return restored_items
    else:
      if process_index is None or process_count is None:
        process_index, process_count = jax.process_index(), jax.process_count()
      return restore_single_process(item, process_index, process_count)


@ocp.args.register_with_handler(GrainCheckpointHandler, for_save=True)
@dataclasses.dataclass
class GrainCheckpointSave(ocp.args.CheckpointArgs):
  item: Any


@ocp.args.register_with_handler(GrainCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class GrainCheckpointRestore(ocp.args.CheckpointArgs):
  item: Any
  process_index: Optional[int | list[int]] = None
  process_count: Optional[int] = None


class GrainCheckpointable(ocp_v1.StatefulCheckpointable):
  """Adapts `GrainCheckpointHandler` to Orbax v1's `StatefulCheckpointable`."""

  def __init__(
      self,
      *,
      save_args: GrainCheckpointSave | None = None,
      restore_args: GrainCheckpointRestore | None = None,
  ):
    self._handler = GrainCheckpointHandler()
    self._save_args = save_args
    self._restore_args = restore_args

  async def save(self, directory):
    """Saves the Grain iterator state to the given directory."""
    # `GrainCheckpointHandler.save` snapshots iterator state (`get_state`) AND
    # writes it; both must happen in this (blocking) save phase, NOT in the
    # returned background coroutine.
    path = await directory.await_creation()
    self._handler.save(path, args=self._save_args)

    async def _committed():  # nothing left for the background commit phase
      return None

    return _committed()

  async def load(self, directory: epath.Path):
    """Loads the Grain iterator state from the given directory."""
    handler, args = self._handler, self._restore_args

    # This will be ran to completion so asynchronous portion is just for
    # compatibility with Orbax v1 API.
    async def _background_load():
      await asyncio.to_thread(handler.restore, directory, args=args)

    return _background_load()


# TODO(b/534897901): Remove find_idx + replica_devices once Orbax exposes multislice.
def find_idx(array: np.ndarray, replica_axis_idx: int):
  """Returns the index along given dimension that the current host belongs to."""
  idx = None
  for idx, val in np.ndenumerate(array):
    if val.process_index == jax.process_index():
      break
  return idx[replica_axis_idx]


def replica_devices(device_array: np.ndarray, replica_axis_idx: int):
  """Returns the devices from the replica that current host belongs to.

  Replicas are assumed to be restricted to the first axis.

  Args:
    device_array: devices of the mesh that can be obtained by mesh.devices()
    replica_axis_idx: axis dimension along which replica is taken

  Returns:
    devices inside the replica that current host is in
  """
  idx = find_idx(device_array, replica_axis_idx)
  replica_result = np.take(device_array, idx, axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)


def prepare_scaled_down_grain_restore_args(
    data_iterator: list, process_count_jax: int, process_count_stored: int, directory: epath.Path
) -> GrainCheckpointRestore:
  """
  Prepares the restore arguments for a scaled-up (list) data iterator.

  This is used when restoring a checkpoint saved with more processes than
  the current run (e.g., 64 files onto 32 JAX processes).
  """
  # 1. Validation Assertions
  assert isinstance(data_iterator, list), (
      f"{process_count_stored} processes found in Grain checkpoint directory {directory}, but only "
      f"{process_count_jax} jax processes in this run, please set expansion_factor_real_data accordingly."
  )

  scaling_factor = len(data_iterator)
  expected_process_count = process_count_stored / process_count_jax
  assert scaling_factor == expected_process_count, (
      f"Found {process_count_stored} processes in checkpoint and {process_count_jax} "
      f"JAX processes, implying a scaling factor of {expected_process_count}. "
      f"However, the data_iterator list has {scaling_factor} items."
  )

  # 2. Prepare Arguments
  local_iterator_list = [x.local_iterator for x in data_iterator]
  # Each JAX process calculates the global indices it's responsible for.
  # e.g., process 0 with scaling_factor=2 handles checkpoints from processes [0, 32]
  # e.g., process 1 with scaling_factor=2 handles checkpoints from processes [1, 33]
  process_index_list = [jax.process_index() + i * process_count_jax for i in range(scaling_factor)]

  return GrainCheckpointRestore(local_iterator_list, process_index=process_index_list, process_count=process_count_stored)


def restore_grain_iterator(
    checkpoint_manager,
    step: int,
    data_iterator,
    checkpoint_args,
    expansion_factor_real_data: int,  # This must be defined in the outer scope
) -> tuple[Any, None]:
  """
  Handles the complex logic for restoring a Grain data iterator checkpoint.
  This function dispatches to the correct restore strategy based on
  the number of stored checkpoint files vs. current JAX processes.
  """
  if isinstance(data_iterator, RemoteIteratorWrapper):
    grain_restore_args = GrainCheckpointRestore(item=data_iterator)
    restored_state = checkpoint_manager.restore(step, args=Composite(items=checkpoint_args, iter=grain_restore_args))
    return (restored_state, None)

  # ElasticIterator: one shared `process_0.json` regardless of shard count.
  if not isinstance(data_iterator, list) and isinstance(data_iterator.local_iterator, ElasticIterator):
    grain_restore_args = GrainCheckpointRestore(item=data_iterator.local_iterator)
    restored_state = checkpoint_manager.restore(step, args=Composite(items=checkpoint_args, iter=grain_restore_args))
    return (restored_state, None)

  directory = checkpoint_manager.directory / str(step) / "iter"
  process_count_jax = jax.process_count()

  # Count the number of checkpoint files
  process_count_stored = len(list(directory.glob("process_*-of-*.json")))

  grain_restore_args = None

  if process_count_stored > process_count_jax:
    # Scaling down from a larger number of hosts. (e.g., 128 files -> 64 processes)
    # In this case, each host restores a list of data iterators.
    grain_restore_args = prepare_scaled_down_grain_restore_args(
        data_iterator, process_count_jax, process_count_stored, directory
    )

  elif process_count_stored == process_count_jax:
    # Normal case: number of hosts is the same. (e.g., 64 files -> 64 processes)
    assert not isinstance(data_iterator, list), (
        f"{process_count_stored} processes found in Grain checkpoint directory {directory}, matching the number of "
        "jax process, please do not set expansion_factor_real_data."
    )
    grain_restore_args = GrainCheckpointRestore(data_iterator.local_iterator)

  elif expansion_factor_real_data > 1 and process_count_stored == process_count_jax // expansion_factor_real_data:
    # Scaling up to a larger number of hosts.(e.g., 32 files -> 64 processes)
    # In this case, a subset of hosts restore the data iterator.
    assert not isinstance(
        data_iterator, list
    ), "when expansion_factor_real_data > 1, the data iterator should not be a list."
    grain_restore_args = GrainCheckpointRestore(
        data_iterator.local_iterator, process_index=jax.process_index(), process_count=process_count_stored
    )

  else:
    # Case 4: Mismatch
    raise ValueError(
        f"Error restoring Grain checkpoint in {directory}: "
        f"The number of stored checkpoint files ({process_count_stored}) "
        f"is incompatible with the number of JAX processes ({process_count_jax}). "
        "If you are resuming training with a different number of chips, see instructions in "
        "https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline/"
        "data_input_grain.md#using-grain"
    )

  # Call restore once with the composed arguments
  restored_state = checkpoint_manager.restore(step, args=Composite(items=checkpoint_args, iter=grain_restore_args))
  return (restored_state, None)
