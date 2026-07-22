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
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import v1 as ocp

PyGrainCheckpointHandler = python.PyGrainCheckpointHandler
Composite = ocp_v0.args.Composite
RemoteIteratorWrapper = multihost_dataloading.RemoteIteratorWrapper

ElasticIterator = grain_experimental.ElasticIterator
_PROCESS_0_FILENAME = "process_0.json"


def _process_filename(process_index: int, process_count: int) -> str:
  return f"process_{process_index}-of-{process_count}.json"


def _state_to_text(iterator) -> str:
  """Serializes an iterator's state (grain DatasetIterator -> json, else bytes)."""
  if isinstance(iterator, grain.DatasetIterator):
    return json.dumps(iterator.get_state(), indent=4)
  return iterator.get_state().decode()


def _set_state_from_text(iterator, text: str) -> None:
  """Inverse of :func:`_state_to_text`."""
  if isinstance(iterator, grain.DatasetIterator):
    iterator.set_state(json.loads(text))
  else:
    iterator.set_state(text.encode())


async def no_op():
  return None


class GrainCheckpointable_v1(ocp.StatefulCheckpointable):
  """Orbax v1 `StatefulCheckpointable` for MaxText grain data iterators.

  This is the v1 port of the `GrainCheckpointHandler`: a single object that
  dispatches on the held item, kept here (rather than in grain) only for the
  cases grain itself does not cover. The case-dispatch lives in this class —
  callers just wrap the item — matching the original handler.

  `save` snapshots iterator state synchronously and returns a coroutine that
  does the file IO in the background (the Orbax two-phase contract); `load`
  returns a coroutine that reads and re-applies the state.

  Cases:
    * **standard** grain iterator -> delegate to grain's own
      `StatefulCheckpointable` (`process_{index}-of-{count}.json`). This
      functionality is folded into grain.
    * **ElasticIterator** -> one shared `process_0.json` (state is a single
      global scalar; the fixed name survives a change in `jax.process_count()`).
      Lift target: grain, which owns ``ElasticIterator``.
    * **list** of `(iterator, process_index, process_count)` -> explicit
      per-file write/read for a host-count change (`expansion_factor_real_data`).
      The index/count arithmetic is computed by the caller.
    * **RemoteIteratorWrapper** (Pathways colocated-python) -> the wrapper
      persists/restores its own state keyed by `step`; identified structurally
      (it exposes `save_state`/`restore_state`) so this module stays
      independent of the input pipeline.
  """

  def __init__(self, item, *, restore_process_index=None, restore_process_count=None, step=None):
    """Initializes a GrainCheckpointable_v1.

    Args:
      item: a grain iterator, a grain `ElasticIterator`, a
        `RemoteIteratorWrapper`, or a list of `(iterator, process_index,
        process_count)` (scaled save).
      restore_process_index: restore-only; an int (scale-up) or list of ints
        (scale-down, paired with the items in `item`). `None` uses the
        current process index.
      restore_process_count: restore-only; the stored host count for the file name.
        `None` uses the current process count (the standard, grain-native
        path).
      step: required only for the `RemoteIteratorWrapper` case.
    """
    self._item = item
    self._restore_process_index = restore_process_index
    self._restore_process_count = restore_process_count
    self._step = step

  async def save(self, directory):
    """Snapshots the wrapped iterator's state and returns a coroutine that writes it to ``directory``."""
    item = self._item

    # RemoteIteratorWrapper handles checkpointing via colocated python
    if isinstance(item, RemoteIteratorWrapper):
      item.save_state(self._step)
      return no_op()

    # ElasticIterator state is a single global scalar shared by all shards,
    # so we write one fixed `process_0.json` from process 0 only. This file
    # layout survives changes in `jax.process_count()`.
    if isinstance(item, ElasticIterator):
      state = item.get_state()  # snapshot in the blocking phase

      async def _write_elastic():
        path = await directory.await_creation()
        if jax.process_index() == 0:  # one shared file written by process 0
          await asyncio.to_thread(
              (path / _PROCESS_0_FILENAME).write_text,
              json.dumps(state, indent=4),
          )

      return _write_elastic()

    if isinstance(item, list):
      snapshots = [(_state_to_text(it), idx, count) for it, idx, count in item]

      async def _write_list():
        path = await directory.await_creation()
        for text, idx, count in snapshots:
          await asyncio.to_thread((path / _process_filename(idx, count)).write_text, text)

      return _write_list()

    if hasattr(item, "save"):
      # Standard: delegate to grain's own StatefulCheckpointable.
      return await item.save(directory)

    # Custom fallback for iterators without .save()
    state_text = _state_to_text(item)

    async def _write_single_fallback():
      path = await directory.await_creation()
      filename = path / _process_filename(jax.process_index(), jax.process_count())
      await asyncio.to_thread(filename.write_text, state_text)

    return _write_single_fallback()

  async def load(self, directory):
    """Restores the wrapped iterator's state from ``directory`` (or returns a coroutine that does)."""
    item = self._item

    # In Pathways + colocated_python environment, RemoteIteratorWrapper handles checkpointing
    if isinstance(item, RemoteIteratorWrapper):
      item.restore_state(self._step)
      return no_op()

    # McJax and Pathways through controller cases
    # ElasticIterator: every process reads the same shared `process_0.json`.
    if isinstance(item, ElasticIterator):

      async def _read_elastic():
        filename = directory / _PROCESS_0_FILENAME
        if not await asyncio.to_thread(filename.exists):
          raise ValueError(f"File {filename} does not exist.")
        item.set_state(json.loads(await asyncio.to_thread(filename.read_text)))

      return _read_elastic()

    if isinstance(item, list):
      # Scale-down: each held iterator reads its own stored shard file.
      specs = list(zip(item, self._restore_process_index))  # pyrefly: ignore[bad-argument-type]

      async def _read_list():
        for iterator, idx in specs:
          filename = directory / _process_filename(idx, self._restore_process_count)  # pyrefly: ignore[bad-argument-type]
          if not await asyncio.to_thread(filename.exists):
            raise ValueError(f"File {filename} does not exist.")
          _set_state_from_text(iterator, await asyncio.to_thread(filename.read_text))

      return _read_list()

    if self._restore_process_count is not None:
      # Single iterator with an explicit stored host count (scale-up).
      index = self._restore_process_index if self._restore_process_index is not None else jax.process_index()

      async def _read_single():
        filename = directory / _process_filename(index, self._restore_process_count)
        if not await asyncio.to_thread(filename.exists):
          raise ValueError(f"File {filename} does not exist.")
        _set_state_from_text(item, await asyncio.to_thread(filename.read_text))

      return _read_single()

    if hasattr(item, "load"):
      # Standard: delegate to grain's own StatefulCheckpointable.
      return await item.load(directory)

    # Custom fallback for iterators without .load()
    async def _read_single_fallback():
      filename = directory / _process_filename(jax.process_index(), jax.process_count())
      if not await asyncio.to_thread(filename.exists):
        raise ValueError(f"File {filename} does not exist.")
      _set_state_from_text(item, await asyncio.to_thread(filename.read_text))

    return _read_single_fallback()


def for_save(step: int, data_iterator: Any, expansion_factor_real_data: int) -> GrainCheckpointable_v1:
  """Builds the v1 ``GrainCheckpointable_v1`` for saving the grain iterator."""
  if isinstance(data_iterator, RemoteIteratorWrapper):
    return GrainCheckpointable_v1(data_iterator, step=step)

  if (
      not isinstance(data_iterator, list)
      and hasattr(data_iterator, "local_iterator")
      and isinstance(data_iterator.local_iterator, ElasticIterator)
  ):
    return GrainCheckpointable_v1(data_iterator.local_iterator)

  iterators = data_iterator if isinstance(data_iterator, list) else [data_iterator]
  process_count_total = jax.process_count() * len(iterators)
  if expansion_factor_real_data > 1:
    process_count_total = process_count_total // expansion_factor_real_data

  if len(iterators) == 1 and process_count_total == jax.process_count():
    return GrainCheckpointable_v1(
        iterators[0].local_iterator if hasattr(iterators[0], "local_iterator") else iterators[0]
    )

  specs = [
      (
          di.local_iterator if hasattr(di, "local_iterator") else di,
          jax.process_index() + i * jax.process_count(),
          process_count_total,
      )
      for i, di in enumerate(iterators)
  ]
  return GrainCheckpointable_v1(specs)


def for_restore(
    checkpoint_manager: Any, step: int, data_iterator: Any, expansion_factor_real_data: int
) -> GrainCheckpointable_v1:
  """Builds the v1 ``GrainCheckpointable_v1`` for restoring the grain iterator."""
  if isinstance(data_iterator, RemoteIteratorWrapper):
    return GrainCheckpointable_v1(data_iterator, step=step)

  if (
      not isinstance(data_iterator, list)
      and hasattr(data_iterator, "local_iterator")
      and isinstance(data_iterator.local_iterator, ElasticIterator)
  ):
    return GrainCheckpointable_v1(data_iterator.local_iterator)

  directory = checkpoint_manager.directory / str(step) / "iter"
  process_count_jax = jax.process_count()
  process_count_stored = len(list(directory.glob("process_*-of-*.json")))

  if process_count_stored > process_count_jax:
    assert isinstance(data_iterator, list), (
        f"{process_count_stored} processes found in Grain checkpoint directory {directory}, but only "
        f"{process_count_jax} jax processes in this run, please set expansion_factor_real_data accordingly."
    )
    scaling_factor = len(data_iterator)
    expected = process_count_stored / process_count_jax
    assert scaling_factor == expected, (
        f"Found {process_count_stored} processes in checkpoint and {process_count_jax} JAX processes, "
        f"implying a scaling factor of {expected}, but the data_iterator list has {scaling_factor} items."
    )
    local_iterators = [x.local_iterator if hasattr(x, "local_iterator") else x for x in data_iterator]
    restore_process_index = [jax.process_index() + i * process_count_jax for i in range(scaling_factor)]
    return GrainCheckpointable_v1(
        local_iterators, restore_process_index=restore_process_index, restore_process_count=process_count_stored
    )

  if process_count_stored == process_count_jax:
    assert not isinstance(data_iterator, list), (
        f"{process_count_stored} processes found in Grain checkpoint directory {directory}, matching the number of "
        "jax processes, please do not set expansion_factor_real_data."
    )
    return GrainCheckpointable_v1(
        data_iterator.local_iterator if hasattr(data_iterator, "local_iterator") else data_iterator
    )

  if expansion_factor_real_data > 1 and process_count_stored == process_count_jax // expansion_factor_real_data:
    assert not isinstance(
        data_iterator, list
    ), "when expansion_factor_real_data > 1, the data iterator should not be a list."
    return GrainCheckpointable_v1(
        data_iterator.local_iterator if hasattr(data_iterator, "local_iterator") else data_iterator,
        restore_process_index=jax.process_index(),
        restore_process_count=process_count_stored,
    )

  raise ValueError(
      f"Error restoring Grain checkpoint in {directory}: "
      f"The number of stored checkpoint files ({process_count_stored}) "
      f"is incompatible with the number of JAX processes ({process_count_jax}). "
      "If you are resuming training with a different number of chips, see instructions in "
      "https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline/"
      "data_input_grain.md#using-grain"
  )


# ------------------------------------------------------------------------------
# TODO(b/532274266): Remove everything below this line once distillation_utils
# supports the new GrainCheckpointHandler.
# ------------------------------------------------------------------------------


class GrainCheckpointHandler(PyGrainCheckpointHandler, ocp_v0.CheckpointHandler):
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


@ocp_v0.args.register_with_handler(GrainCheckpointHandler, for_save=True)
@dataclasses.dataclass
class GrainCheckpointSave(ocp_v0.args.CheckpointArgs):
  item: Any


@ocp_v0.args.register_with_handler(GrainCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class GrainCheckpointRestore(ocp_v0.args.CheckpointArgs):
  item: Any
  process_index: Optional[int | list[int]] = None
  process_count: Optional[int] = None


class GrainCheckpointable(ocp.StatefulCheckpointable):
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
