# Copyright 2023–2026 Google LLC
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

"""Orbax v1 `StatefulCheckpointable` for MaxText grain data iterators.

This is the v1 port of the old `GrainCheckpointHandler`: a single object that
dispatches on the held item, kept here (rather than in grain) only for the cases
grain itself does not cover. The case-dispatch lives in this class — callers
just wrap the item — matching the original handler.

`save` snapshots iterator state synchronously and returns a coroutine that
does the file IO in the background (the Orbax two-phase contract); `load`
returns a coroutine that reads and re-applies the state.

Cases:
  * **standard** grain iterator -> delegate to grain's own 
    `StatefulCheckpointable` (`process_{index}-of-{count}.json`). This
    functionality is folded into grain.
  * **ElasticIterator** -> one shared `process_0.json` (state is a single global
    scalar; the fixed name survives a change in `jax.process_count()`). Lift
    target: grain, which owns ``ElasticIterator``.
  * **list** of `(iterator, process_index, process_count)` -> explicit per-file
    write/read for a host-count change (`expansion_factor_real_data`). The
    index/count arithmetic is computed by the caller.
  * **RemoteIteratorWrapper** (Pathways colocated-python) -> the wrapper
    persists/restores its own state keyed by `step`; identified structurally
    (it exposes `save_state`/`restore_state`) so this module stays independent
    of the input pipeline.
"""

import asyncio
import json
from typing import Any

import grain
from grain import experimental as grain_experimental
import jax
from maxtext.input_pipeline import multihost_dataloading
from orbax.checkpoint import v1 as ocp_v1


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


class GrainCheckpointable(ocp_v1.StatefulCheckpointable):
  """v1 checkpointable for a grain iterator (or the non-standard variants)."""

  def __init__(
      self, item, *, process_index=None, process_count=None, step=None
  ):
    """Initializes a GrainCheckpointable.

    Args:
      item: a grain iterator, a grain ``ElasticIterator``, a
        ``RemoteIteratorWrapper``, or a list of ``(iterator, process_index,
        process_count)`` (scaled save).
      process_index: restore-only; an int (scale-up) or list of ints
        (scale-down, paired with the items in ``item``). ``None`` uses the
        current process index.
      process_count: restore-only; the stored host count for the file name.
        ``None`` uses the current process count (the standard, grain-native
        path).
      step: required only for the ``RemoteIteratorWrapper`` case.
    """
    self._item = item
    self._process_index = process_index
    self._process_count = process_count
    self._step = step

  async def save(self, directory):
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
      specs = list(zip(item, self._process_index))

      async def _read_list():
        for iterator, idx in specs:
          filename = directory / _process_filename(idx, self._process_count)
          if not await asyncio.to_thread(filename.exists):
            raise ValueError(f"File {filename} does not exist.")
          _set_state_from_text(iterator, await asyncio.to_thread(filename.read_text))

      return _read_list()

    if self._process_count is not None:
      # Single iterator with an explicit stored host count (scale-up).
      index = self._process_index if self._process_index is not None else jax.process_index()

      async def _read_single():
        filename = directory / _process_filename(index, self._process_count)
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


def for_save(checkpoint_manager: Any, step: int, data_iterator: Any, expansion_factor_real_data: int) -> GrainCheckpointable:
  """Builds the v1 ``GrainCheckpointable`` for saving the grain iterator."""
  if isinstance(data_iterator, RemoteIteratorWrapper):
    return GrainCheckpointable(data_iterator, step=step)

  if not isinstance(data_iterator, list) and hasattr(data_iterator, "local_iterator") and isinstance(data_iterator.local_iterator, ElasticIterator):
    return GrainCheckpointable(data_iterator.local_iterator)

  iterators = data_iterator if isinstance(data_iterator, list) else [data_iterator]
  process_count_total = jax.process_count() * len(iterators)
  if expansion_factor_real_data > 1:
    process_count_total = process_count_total // expansion_factor_real_data
    
  if len(iterators) == 1 and process_count_total == jax.process_count():
    return GrainCheckpointable(iterators[0].local_iterator if hasattr(iterators[0], "local_iterator") else iterators[0])
    
  specs = [
      (di.local_iterator if hasattr(di, "local_iterator") else di, jax.process_index() + i * jax.process_count(), process_count_total)
      for i, di in enumerate(iterators)
  ]
  return GrainCheckpointable(specs)


def for_restore(checkpoint_manager: Any, step: int, data_iterator: Any, expansion_factor_real_data: int) -> GrainCheckpointable:
  """Builds the v1 ``GrainCheckpointable`` for restoring the grain iterator."""
  if isinstance(data_iterator, RemoteIteratorWrapper):
    return GrainCheckpointable(data_iterator, step=step)

  if not isinstance(data_iterator, list) and hasattr(data_iterator, "local_iterator") and isinstance(data_iterator.local_iterator, ElasticIterator):
    return GrainCheckpointable(data_iterator.local_iterator)

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
    process_index = [jax.process_index() + i * process_count_jax for i in range(scaling_factor)]
    return GrainCheckpointable(
        local_iterators, process_index=process_index, process_count=process_count_stored
    )

  if process_count_stored == process_count_jax:
    assert not isinstance(data_iterator, list), (
        f"{process_count_stored} processes found in Grain checkpoint directory {directory}, matching the number of "
        "jax processes, please do not set expansion_factor_real_data."
    )
    return GrainCheckpointable(data_iterator.local_iterator if hasattr(data_iterator, "local_iterator") else data_iterator)

  if expansion_factor_real_data > 1 and process_count_stored == process_count_jax // expansion_factor_real_data:
    assert not isinstance(
        data_iterator, list
    ), "when expansion_factor_real_data > 1, the data iterator should not be a list."
    return GrainCheckpointable(
        data_iterator.local_iterator if hasattr(data_iterator, "local_iterator") else data_iterator,
        process_index=jax.process_index(),
        process_count=process_count_stored
    )

  raise ValueError(
      f"Error restoring Grain checkpoint in {directory}: "
      f"The number of stored checkpoint files ({process_count_stored}) "
      f"is incompatible with the number of JAX processes ({process_count_jax}). "
      "If you are resuming training with a different number of chips, see instructions in "
      "https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline/"
      "data_input_grain.md#using-grain"
  )
