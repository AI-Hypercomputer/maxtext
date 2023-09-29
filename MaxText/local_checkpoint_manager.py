# Copyright 2023 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A class providing functionalities for managing multiple checkpoints."""
print("Matts local checkpoint manager")
import concurrent.futures
import contextlib
import dataclasses
import datetime
import numpy as np
import threading
from typing import Any, Callable, Container, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization
from orbax.checkpoint import utils
from orbax.checkpoint.abstract_checkpointer import AbstractCheckpointer
from orbax.checkpoint.async_checkpointer import AsyncCheckpointer
from orbax.checkpoint.checkpointer import Checkpointer
from orbax.checkpoint.json_checkpoint_handler import JsonCheckpointHandler
from orbax.checkpoint.proto_checkpoint_handler import ProtoCheckpointHandler

PyTree = Any
CheckpointDirs = Tuple[str, str]
SaveParams = Mapping[str, Any]
RestoreParams = SaveParams
CheckpointersDict = Mapping[str, AbstractCheckpointer]

DEFAULT_ITEM_NAME = 'default'
DESCRIPTOR_ITEM_NAME = 'descriptor'
METRIC_ITEM_NAME = 'metrics'
METADATA_ITEM_NAME = 'metadata'

RESERVED_ITEM_NAMES = [DESCRIPTOR_ITEM_NAME, METRIC_ITEM_NAME]

_INIT_TIME = datetime.datetime.now(tz=datetime.timezone.utc)


def _metrics_file_exists(metrics_item_path: epath.Path) -> bool:
  """True if item directory AND actual file both exist."""
  return (
      metrics_item_path.exists()
      and (metrics_item_path / METRIC_ITEM_NAME).exists()
  )


def _descriptor_file_exists(descriptor_item_path: epath.Path) -> bool:
  """True if item directory AND actual file both exist."""
  return (
      descriptor_item_path.exists()
      and (descriptor_item_path / f'{DESCRIPTOR_ITEM_NAME}.pbtxt').exists()
  )


# TODO(b/268051457) Clean up when no longer depended upon by internal users.
def is_async_checkpointer(checkpointer: AbstractCheckpointer):
  return isinstance(checkpointer, AsyncCheckpointer) or isinstance(
      checkpointer,
      serialization.GlobalAsyncCheckpointManagerBase,
  )


@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  save_interval_steps:
    The interval at which checkpoints should be saved.
    Ensures checkpoints will only be saved every n steps. Defaults to 1.
  max_to_keep:
    If provided, specifies the maximum number of checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints. Must be None or non-negative. When set, checkpoints
    may be considered for deletion when there are more than `max_to_keep`
    checkpoints present. Checkpoints are kept if they meet any of the conditions
    below, such as `keep_time_interval`, `keep_period`, etc. Any remaining
    checkpoints that do not meet these conditions are garbage-collected.
  keep_time_interval:
    When more than max_to_keep checkpoints are present,
    an older checkpoint that would ordinarily be deleted will be preserved if it
    has been at least `keep_time_interval` since the previous preserved
    checkpoint. The default setting of `None` does not preserve any checkpoints
    in this way. For example, this may be used to ensure checkpoints are
    retained at a frequency of approximately than one per hour.
  keep_period:
    If set, will not delete any checkpoint where checkpoint_step %
    keep_period == 0.
  best_fn:
    If set, maintains checkpoints based on the quality of given
    metrics rather than recency. The function should accept a PyTree of metrics,
    and return a scalar value that can be used to determine the quality score
    of the checkpoint. If `max_to_keep` is also set, then the retained
    checkpoints will be kept based on their quality, as measured by this
    function.
  best_mode:
    One of ['max', 'min']. The best metric is determine on the basis of this
    value.
  keep_checkpoints_without_metrics:
    If False, checkpoints without metrics present
    are eligible for cleanup. Otherwise, they will never be deleted.
  step_prefix:
    If provided, step directories will take the form
    f'{step_prefix}_<step>'. Otherwise, they will simply be an integer <step>.
  step_format_fixed_length:
    If set, formats step with n digits (leading zeros).
    This makes sorting steps easier. Otherwise, step has no leading zeros.
  create:
    If True, creates the top-level directory if it does not already exist.
  cleanup_tmp_directories:
    If True, cleans up any existing temporary directories
    on CheckpointManager creation.
  save_on_steps:
    Optional set of steps at which checkpoints should be saved.
    Useful to save checkpoints on a fixed set of steps that are not multiple of
    `save_interval_steps`.
  """
  save_interval_steps: int = 1
  max_to_keep: Optional[int] = None
  keep_time_interval: Optional[datetime.timedelta] = None
  keep_period: Optional[int] = None
  best_fn: Optional[Callable[[PyTree], float]] = None
  best_mode: str = 'max'
  keep_checkpoints_without_metrics: bool = True
  step_prefix: Optional[str] = None
  step_format_fixed_length: Optional[int] = None
  create: bool = True
  cleanup_tmp_directories: bool = False
  save_on_steps: Optional[Container[int]] = None

  def __post_init__(self):
    if self.best_mode not in ('min', 'max'):
      msg = ("`CheckpointManagerOptions.best_mode` must be one of None, 'min' "
             "or 'max'. Got {self.dtype}.")
      raise ValueError(msg)
    self.save_on_steps = frozenset(self.save_on_steps or ())
    if self.max_to_keep is not None and self.max_to_keep < 0:
      raise ValueError('Setting of `max_to_keep` must be None or non-negative.')


@dataclasses.dataclass
class CheckpointInfo:
  """Metadata about a checkpoint."""
  step: int
  time: datetime.datetime
  metrics: Optional[PyTree]
  is_locked: Optional[bool] = None

  def __str__(self) -> str:
    return f'Checkpoint[step={self.step} | time={self.time}]'

  def __eq__(self, other: 'CheckpointInfo') -> bool:
    return self.step == other.step and self.time == other.time


class MyCheckpointManager:
  """A generic, synchronous CheckpointManager implementation.

  Allows a user to save and restore objects for which a Checkpointer
  implementation exists (e.g. PyTreeCheckpointer for PyTrees). The class
  keeps track of multiple checkpointable objects in the following structure::

    path/to/directory/    (top-level directory)
      0/    (step)
        params/    (first saveable)
          ...
        metadata/    (second saveable)
          ...
      1/    (step)
        ...
      2/    (step)
        ...
      ...
  """

  def __init__(
      self,
      directory: epath.PathLike,
      checkpointers: Union[AbstractCheckpointer, CheckpointersDict],
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[Mapping[str, Any]] = None,
  ):
    """CheckpointManager constructor.

    Example::

      CheckpointManager(
        'path/to/dir/',
        # Multiple items.
        checkpointers = {
            'train_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
            'dataset': Checkpointer(CustomTFDatasetCheckpointHandler()),
        },
        metadata={'version': 1.1, 'lang': 'en'},
      )

      CheckpointManager(
        'path/to/dir/',
        # Single item.
        checkpointers = AsyncCheckpointer(PyTreeCheckpointHandler()),
        options = CheckpointManagerOptions(max_to_keep=5, ...),
      )

    Args:
      directory: the top level directory in which to save all files.
      checkpointers: a mapping of object name to Checkpointer object. For
        example, `items` provided to `save` below should have keys matching the
        keys in this argument. Alternatively, a single Checkpointer may be
        provided, in which case `save` and `restore` should always be called
        with a single item rather than a dictionary of items. See below for more
        details.
     options: CheckpointManagerOptions. May be provided to specify additional
       arguments. If None, uses default values of CheckpointManagerOptions.
     metadata: High-level metadata that does not depend on step number, and only
       needs to be saved once.
    """
    jax.monitoring.record_event('/jax/orbax/checkpoint_manager/init')
    self._single_item = False
    if isinstance(checkpointers, AbstractCheckpointer):
      self._single_item = True
      checkpointers = {DEFAULT_ITEM_NAME: checkpointers}
    elif isinstance(checkpointers, dict):
      for item in [k for k in checkpointers if k in RESERVED_ITEM_NAMES]:
        raise ValueError(
            f'Found {item} in `checkpointers`; this is a reserved'
            ' key.'
        )
    else:
      raise ValueError(
          f'Invalid type for `checkpointers`. Found {checkpointers}.')

    self._checkpointers = checkpointers
    self._options = options or CheckpointManagerOptions()
    if self._options.best_mode not in ['min', 'max']:
      raise ValueError('`best_mode` must be one of: "min", "max"')
    if self._track_best:
      self._checkpointers[METRIC_ITEM_NAME] = Checkpointer(
          JsonCheckpointHandler(filename=METRIC_ITEM_NAME)
      )

    self._directory = epath.Path(directory)
    if self._options.create:
      if jax.process_index() == 0 and not self._directory.exists():
        self._directory.mkdir(parents=True)
      utils.sync_global_devices('CheckpointManager:create_directory')

    # Cleanup directories from previous runs that may not have been finalized.
    if self._options.cleanup_tmp_directories:
      self._cleanup_tmp_directories()
    self._checkpoints = self._create_checkpoints()
    self._interval_preserved_checkpoints = (
        self._get_interval_preserved_checkpoints(self._checkpoints)
    )
    if self._checkpoints:
      self._last_checkpoint = self._checkpoints[-1]
    else:
      self._last_checkpoint = None

    self._metadata = None
    if metadata is not None:
      self._save_metadata(metadata)

    self._finalize_thread = None
    # Steps that get cleaned up during finalize.
    self._steps_to_remove = []

  @property
  def directory(self) -> epath.Path:
    return self._directory

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    print("\n\n We are in all steps \n\n")
    if read:
      max_steps = 10
      padded_step_list = np.array([-1] * max_steps)
      if jax.process_index() == 0:
        steps = np.array(utils.checkpoint_steps(self.directory))
        print(f"\n\n {steps=} \n\n")
        padded_step_list[0:len(steps)] = steps
        print(f"\n\n {padded_step_list=} \n\n")
      padded_step_list = multihost_utils.broadcast_one_to_all(padded_step_list)
      print(f"\n\n {padded_step_list=} \n\n")

      def unpad_step_list(padded_step_list):
        return [step for step in padded_step_list if step>=0]
      steps = unpad_step_list(padded_step_list)
      print(f"{steps=}")
      steps = list(steps)
      return steps
        
      
      
    return [ckpt.step for ckpt in self._checkpoints]

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    steps = self.all_steps(read=False)
    return max(steps) if steps else None

  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as defined by `options.best_fn`.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    if not self._track_best:
      return self.latest_step()
    if not self._checkpoints:
      return None
    _, sorted_checkpoints = self._sort_checkpoints_by_metrics(self._checkpoints)
    if not sorted_checkpoints:
      return None
    return sorted_checkpoints[-1].step

  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""
    return multihost_utils.reached_preemption_sync_point(step)

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    if self.reached_preemption(step):
      return True
    last_checkpoint_step = (
        self._last_checkpoint.step if self._last_checkpoint else None)
    # Ensure current step is between the last step and next step (accounting for
    # save interval). The `last_checkpoint_step` may not be initialized, in
    # which case we should save. Otherwise, step must fall on the specified
    # save interval. This condition accounts for the possibility of saving
    # on preemption, in which case we want to maintain the same save period as
    # if preemption had not happened.
    return last_checkpoint_step is None or (
        last_checkpoint_step < step and
        (step % self._options.save_interval_steps == 0 or
         step in self._options.save_on_steps))

  def _get_save_directory(
      self,
      step: int,
      directory: epath.Path,
      key_name: Optional[str] = None,
      tmp_directory: Optional[epath.Path] = None,
  ) -> epath.Path:
    """Returns the standardized path to a save directory for a single item."""
    return utils.get_save_directory(
        step,
        directory,
        name=key_name,
        step_prefix=self._options.step_prefix,
        override_directory=tmp_directory,
        step_format_fixed_length=self._options.step_format_fixed_length,
    )

  def _create_tmp_directory(self, directory: epath.Path) -> epath.Path:
    """Creates a tmp directory based on the given directory."""
    return utils.create_tmp_directory(directory)

  def delete(self, step: int):
    """Deletes a step checkpoint."""
    if step not in self.all_steps():
      raise ValueError(f'Requested deleting a non-existent step: {step}.')
    if jax.process_index() == 0:
      step_dir = self._get_save_directory(step, self._directory)
      # Erase files, but not in-memory record of past checkpoints.
      step_dir.rmtree()
    utils.sync_global_devices('CheckpointManager:deleted_step')
    for i, info in enumerate(self._checkpoints):
      if info.step == step:
        self._checkpoints.pop(i)

  def save(self,
           step: int,
           items: Union[Any, Mapping[str, Any]],
           save_kwargs: Optional[Union[SaveParams, Mapping[str,
                                                           SaveParams]]] = None,
           metrics: Optional[PyTree] = None,
           force: Optional[bool] = False) -> bool:
    """Saves the provided items.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    Items and save_kwargs must have a top-level structure matching that of
    self._checkpointers, meaning that for every key in items and save_kwargs, a
    corresponding key must be present in self._checkpointers.

    Items takes a form similar to the following::

      {
        'params': PyTree(),
        'metadata': <nested k/v>,
        ...
      }
      Similarly, save_kwargs takes the form:
      {
        'params': {
          <kwargs for PyTreeCheckpointHandler.save>
        },
        'metadata': {
          <kwargs for JsonCheckpointHandler.save>
        }
        ...
      }

    The kwargs under 'params' correspond to PyTreeCheckpointHandler.save. If a
    key is not present in save_kwargs, it is assumed that no kwargs are needed
    for saving that item. If not provided at all, it is assumed that no items
    need extra kwargs for saving.

    Note that if a single Checkpointer was provided at construction time,
    `items` must be a singular saveable object, and `save_kwargs` must be the
    kwargs needed by a single Checkpointer.

    Args:
      step: current step, int
      items: a savable object, or a dictionary of object name to savable object.
      save_kwargs: save kwargs for a single Checkpointer, or a dictionary of
        object name to kwargs needed by the Checkpointer implementation to save
        the object.
      metrics: a dictionary of metric name (string) to numeric value to be
        tracked along with this checkpoint. Required if `options.best_fn` is
        set. Allows users to specify a metric value to determine which
        checkpoints are best and should be kept (in conjunction with
        `options.max_to_keep`).
      force: if `True`, this method will attempt to save a checkpoint
        regardless of the result of `CheckpointManager.should_save(step)`. By
        default, `save` will only write a checkpoint to disk when the options
        permit, e.g. when `step` is in `options.save_interval_steps` or
        `options.save_on_steps`.
        Setting `force=True` will not overwrite existing checkpoints.

    Returns:
      bool indicating whether a save operation was performed.
    Raises:
      ValueError: if `track_best` was indicated but `metrics` is not provided.
      ValueError: directory creation failed.
      ValueError: if an item is provided for which no `Checkpointer` is
      found.
      ValueError: if the checkpoint already exists.
    """
    if not force and not self.should_save(step):
      return False
    if self.reached_preemption(step):
      logging.info('Saving checkpoint at step %d due to preemption.', step)

    # Wait for ongoing saves to complete. Only applicable if some of the
    # checkpointers are AsyncCheckpointers.
    self.wait_until_finished()

    if step in self.all_steps():
      raise ValueError(f'Checkpoint for step {step} already exists.')

    if save_kwargs is None:
      save_kwargs = {}
    if self._single_item:
      items = {DEFAULT_ITEM_NAME: items}
      save_kwargs = {DEFAULT_ITEM_NAME: save_kwargs}
    else:
      items = dict(items)

    if self._track_best:
      if metrics is None:
        logging.warning('Requested `tracked_metric`; did not provide metrics.')
      else:
        items[METRIC_ITEM_NAME] = metrics
    tmp_step_dir = self._create_tmp_directory(
        self._get_save_directory(step, self.directory)
    )
    for k, item in items.items():
      # Gets save dirs given top directory, step number, and a "collection". All
      # files from the same input object should be saved under this collection.
      item_dir = self._get_save_directory(
          step, self.directory, k, tmp_directory=tmp_step_dir
      )
      if k not in self._checkpointers:
        raise ValueError(f'Checkpointer for item "{k}" not found')
      kwargs = save_kwargs.get(k, {})
      self._checkpointers[k].save(item_dir, item, **kwargs)

    self._add_checkpoint_info(step, metrics)
    self._get_old_steps_to_remove()
    # Sync needed to ensure that old steps to remove are retrieved before
    # actually deleting them during finalize, since retrieval can involve
    # looking at the directory.
    utils.sync_global_devices('CheckpointManager:old_steps_to_remove')

    assert self._finalize_thread is None
    if self._all_checkpointers_are_sync:
      self._finalize(tmp_step_dir)
      utils.sync_global_devices('CheckpointManager:finalize')
    else:
      self._finalize_thread = threading.Thread(
          target=self._finalize, args=(tmp_step_dir,)
      )
      self._finalize_thread.start()
    return True

  def restore(
      self,
      step: int,
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      restore_kwargs: Optional[Union[RestoreParams,
                                     Mapping[str, RestoreParams]]] = None,
      directory: Optional[epath.PathLike] = None
  ) -> Union[Any, Mapping[str, Any]]:
    """Restores from the given step and provided items.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    Items and restore_kwargs must have a top-level structure matching that of
    self._checkpointers, meaning that for every key in items and restore_kwargs,
    a
    corresponding key must be present in self._checkpointers.

    Items takes a form similar to the following::

      {
        'params': PyTree(),
        'metadata': <nested k/v>,
        ...
      }

    Items may not be provided at all, in which case it the items restored are
    those specified in self._checkpointers, and item=None is provided to
    Checkpointer.restore. Similarly, an item may be omitted from `items`,
    in
    which case item=None will be provided to Checkpointer.restore.

    Similarly, restore_kwargs takes the form::

      {
        'params': {
          'meshes': PyTree(),
          'mesh_axes': PyTree(),
        },
        'metadata': {
          <kwargs for JsonCheckpointHandler.save>
        }
        ...
      }

    The kwargs under 'params' correspond to PyTreeCheckpointHandler.restore. If
    a key is not present in restore_kwargs, it is assumed that no kwargs are
    needed for restoring that item. If not provided at all, it is assumed that
    no items need extra kwargs for restoring.

    Note that if a single Checkpointer was provided at construction time,
    `items` must be a singular saveable object, and `restore_kwargs` must be the
    kwargs needed by a single Checkpointer.

    Args:
      step: current step, int
      items: a restoreable object, or a dictionary of object name to restorable
        object.
      restore_kwargs: restore kwargs for a single Checkpointer, or a dictionary
        of object name to kwargs needed by the Checkpointer implementation to
        restore the object.
      directory: if provided, uses the given directory rather than the
        `directory` property of this class. Can be used to restore checkpoints
        from an independent location.

    Returns:
      A dictionary matching the structure of self._checkpointers, with one
      object returned for each Checkpointer, or a single restored object,
      if a
      single item is being tracked by this manager.
    """
    if items is None:
      items = {}
    elif self._single_item:
      items = {DEFAULT_ITEM_NAME: items}
    if restore_kwargs is None:
      restore_kwargs = {}
    elif self._single_item:
      restore_kwargs = {DEFAULT_ITEM_NAME: restore_kwargs}

    restored_items = self._restore_impl(
        step, items, restore_kwargs, directory=directory)

    if self._single_item:
      return restored_items[DEFAULT_ITEM_NAME]
    return restored_items

  def _restore_impl(
      self,
      step: int,
      items: Mapping[str, Any],
      restore_kwargs: Mapping[str, RestoreParams],
      directory: Optional[epath.PathLike] = None) -> Mapping[str, Any]:
    """Restores only the provided items, or all items if empty."""
    if directory is None:
      directory = self.directory
    else:
      directory = epath.Path(directory)
    restored = {}
    item_keys_to_restore = items.keys() or self._checkpointers.keys()
    for item_name in item_keys_to_restore:
      path = self._get_save_directory(step, directory, item_name)
      if item_name == METRIC_ITEM_NAME:
        assert self._track_best
        # No metrics file present: not an error.
        if not _metrics_file_exists(path):
          logging.warning('Missing metrics for step %d', step)
          continue
      if item_name not in self._checkpointers:
        raise ValueError(f'Checkpointer for item "{item_name}" not found')
      item = items.get(item_name, None)
      kwargs = restore_kwargs.get(item_name, {})
      restored[item_name] = self._checkpointers[item_name].restore(
          path, item=item, **kwargs)

    return restored

  def item_metadata(self, step: int) -> Union[Any, Mapping[str, Optional[Any]]]:
    """For all Checkpointers, returns any metadata associated with the item.

    Calls the `metadata` method for each Checkpointer and returns a
    mapping of each item name to the restored metadata. If the manager only
    manages a single item, a single metadata will be returned instead.

    Metadata may be None for an individual item.

    Args:
      step: Step for which to retrieve metadata.

    Returns:
      A dictionary mapping name to item metadata, or a single item metadata.
    """
    result = {}
    for name, checkpointer in self._checkpointers.items():
      path = self._get_save_directory(step, self.directory, name)
      if name == METRIC_ITEM_NAME:
        assert self._track_best
        # No metrics file present: not an error.
        if not _metrics_file_exists(path):
          logging.warning('Missing metrics for step %d', step)
          continue
      metadata = checkpointer.metadata(path)
      result[name] = metadata
    if self._single_item:
      return result[DEFAULT_ITEM_NAME]
    return result

  @property
  def _track_best(self):
    """Returns true if we should track the best checkpoints by given metric."""
    return self._options.best_fn is not None

  @property
  def _all_checkpointers_are_sync(self):
    return all(not is_async_checkpointer(checkpointer)
               for checkpointer in self._checkpointers.values())

  def _create_checkpoints(self) -> List[CheckpointInfo]:
    """Create a list of CheckpointInfo for existing checkpoints.

    If none are present, returns empty list.

    Returns:
      a list of CheckpointInfo, sorted by increasing step.
    """
    steps = sorted(self.all_steps(read=True))
    if not steps:
      return []

    def checkpoint_info(step: int) -> CheckpointInfo:
      time = datetime.datetime.fromtimestamp(
          self._get_save_directory(step, self.directory).stat().mtime,
          tz=datetime.timezone.utc,
      )

      metrics = None
      if self._track_best:
        restored = self._restore_impl(step, {METRIC_ITEM_NAME: None}, {})
        if METRIC_ITEM_NAME in restored:
          metrics = restored[METRIC_ITEM_NAME]
      return CheckpointInfo(step=step, time=time, metrics=metrics)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = {step: executor.submit(checkpoint_info, step) for step in steps}
      return [futures[step].result() for step in steps]

  def _get_interval_preserved_checkpoints(
      self, checkpoints: List[CheckpointInfo]
  ) -> List[CheckpointInfo]:
    """Gets which checkpoints should be kept based on keep_time_interval."""
    if not checkpoints:
      return []
    interval_preserved_checkpoints = [checkpoints[0]]
    if self._options.keep_time_interval is not None:
      for info in checkpoints[1:]:
        if (
            info.time
            >= interval_preserved_checkpoints[-1].time
            + self._options.keep_time_interval
        ):
          interval_preserved_checkpoints.append(info)
    return interval_preserved_checkpoints

  def _add_checkpoint_info(self, step: int, metrics: Optional[PyTree]):
    self._checkpoints.append(
        CheckpointInfo(step, datetime.datetime.now(tz=datetime.timezone.utc),
                       metrics))
    self._last_checkpoint = self._checkpoints[-1]
    # Only empty if this is the very first checkpoint. First checkpoint is
    # always preserved based on save_time_interval.
    if not self._interval_preserved_checkpoints:
      self._interval_preserved_checkpoints.append(self._checkpoints[-1])

  def _save_metadata(self, metadata: Mapping[str, Any]):
    checkpointer = Checkpointer(JsonCheckpointHandler())
    path = self.directory / METADATA_ITEM_NAME
    if not path.exists():  # May have been created by a previous run.
      checkpointer.save(path, metadata)

  def metadata(self) -> Mapping[str, Any]:
    if self._metadata is None:
      checkpointer = Checkpointer(JsonCheckpointHandler())
      path = self.directory / METADATA_ITEM_NAME
      self._metadata = checkpointer.restore(path)
    return self._metadata

  def _sort_checkpoints_by_metrics(
      self, checkpoints: List[CheckpointInfo]
  ) -> Tuple[List[CheckpointInfo], List[CheckpointInfo]]:
    """Sorts `checkpoints` in order of increasing metric quality.

    Checkpoints without corresponding metrics set will be at the beginning.

    Args:
      checkpoints: a list of CheckpointInfo.

    Returns:
      Tuple of CheckpointInfo lists:
      (checkpoints_without_metrics, checkpoints_sorted_by_metrics)
    """
    without_metrics = [info for info in checkpoints if info.metrics is None]
    with_metrics = [info for info in checkpoints if info.metrics is not None]

    return without_metrics, sorted(
        with_metrics,
        key=lambda info: self._options.best_fn(info.metrics),
        reverse=(self._options.best_mode == 'min'))

  def _cleanup_tmp_directories(self):
    utils.cleanup_tmp_directories(self.directory)

  def _delete_directory(self, step: int):
    if jax.process_index() == 0:
      # TODO(cpgaffney) Optimize tree removal if possible.
      self._get_save_directory(step, self.directory).rmtree()

  def _get_old_steps_to_remove(self):
    """Collects checkpoints that should be deleted later."""
    # Must have set max_to_keep in order to remove any checkpoints.
    if self._options.max_to_keep is None:
      return
    # Not enough checkpoints accumulated to consider deletion.
    if len(self._checkpoints) <= self._options.max_to_keep:
      return

    # Exclude the latest checkpoint, since it is not finalized.
    are_locked = utils.are_locked(
        self.directory,
        tuple([info.step for info in self._checkpoints[:-1]]),
        self._options.step_prefix,
        self._options.step_format_fixed_length,
    )
    self._checkpoints[:-1] = [
        dataclasses.replace(info, is_locked=is_locked)
        for info, is_locked in zip(self._checkpoints, are_locked)
    ]

    if self._track_best:
      # Best steps (to keep) are at the end, after sorting.
      (
          checkpoints_without_metrics,
          sorted_checkpoints,
      ) = self._sort_checkpoints_by_metrics(self._checkpoints)
    else:
      # checkpoints already sorted by ascending step
      checkpoints_without_metrics = []
      sorted_checkpoints = self._checkpoints

    keep = int(self._options.max_to_keep)
    if self._options.keep_checkpoints_without_metrics:
      maybe_delete = (
          sorted_checkpoints[:-keep] if keep > 0 else sorted_checkpoints
      )
      active_checkpoints = (
          checkpoints_without_metrics + sorted_checkpoints[-keep:]
          if keep > 0
          else []
      )
    else:
      all_checkpoints = checkpoints_without_metrics + sorted_checkpoints
      maybe_delete = all_checkpoints[:-keep] if keep > 0 else sorted_checkpoints
      active_checkpoints = all_checkpoints[-keep:] if keep > 0 else []

    kept_checkpoints = []
    self._steps_to_remove = []
    for info in maybe_delete:
      if info.is_locked:
        logging.info(
            'Preserving %s: (Reason: checkpoint is locked).',
            info,
        )
        kept_checkpoints.append(info)
        continue
      if (
          self._options.keep_time_interval is not None
          and self._interval_preserved_checkpoints
      ):
        if info in self._interval_preserved_checkpoints:
          logging.info(
              'Preserving %s: (Reason: older falling on keep_time_interval).',
              info,
          )
          kept_checkpoints.append(info)
          continue
        elif (
            info.time
            >= self._interval_preserved_checkpoints[-1].time
            + self._options.keep_time_interval
        ):
          self._interval_preserved_checkpoints.append(info)
          logging.info(
              'Preserving %s: (Reason: latest falling on keep_time_interval).',
              info,
          )
          kept_checkpoints.append(info)
          continue

      if (
          self._options.keep_period is not None
          and info.step % self._options.keep_period == 0
      ):
        logging.info('Preserving %s: (Reason: on keep_period).', info)
        kept_checkpoints.append(info)
        continue

      reason = 'worse metric' if self._track_best else 'old checkpoint'
      logging.info('Deleting %s: (Reason: %s).', info, reason)
      self._steps_to_remove.append(info.step)

    kept_checkpoints += active_checkpoints
    if self._track_best:
      # Maintain in ascending step order.
      self._checkpoints = sorted(kept_checkpoints, key=lambda info: info.step)
    else:
      self._checkpoints = kept_checkpoints

  def wait_until_finished(self, join_finalize_thread=True):
    """Blocks until any incomplete save operations are completed.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished.

    Args:
      join_finalize_thread: Whether to join the _finalize_thread. This should
        always be True for external callers.
    """
    for checkpointer in self._checkpointers.values():
      if is_async_checkpointer(checkpointer):
        checkpointer.wait_until_finished()  # pytype: disable=attribute-error
    if join_finalize_thread:
      if self._finalize_thread is not None:
        self._finalize_thread.join()
        self._finalize_thread = None
        # Additional work is being done on process 0 of the finalize threads.
        # When joining the threads, we must wait for all threads to complete
        # before proceeding.
        utils.sync_global_devices('CheckpointManager:join_finalize_thread')

  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """
    for checkpointer in self._checkpointers.values():
      if is_async_checkpointer(checkpointer):
        checkpointer.check_for_errors()  # pytype: disable=attribute-error

  def _finalize_checkpoint(
      self, temp_ckpt_dir: epath.Path
  ) -> Optional[epath.Path]:
    """Moves tmp step checkpoint to final.

    Args:
      temp_ckpt_dir: The temporary checkpoint directory. If not None, only
        finalize the checkpoints in `temp_ckpt_dir`. If None, it will iterate
        through all temp checkpoints in `self.directory` and finalize them all.

    Returns:
      the final checkpoint dir
    """
    final_ckpt_dir = None
    if jax.process_index() == 0:
      try:
        self.check_for_errors()
      except Exception as e:  # pylint: disable=broad-except
        logging.error(
            (
                'Received error: %s from Checkpointer. One or more items may'
                ' not be finalized. Skipping finalization of step checkpoint.'
            ),
            e,
        )
        return None
      step = utils.step_from_checkpoint_name(temp_ckpt_dir.name)
      # If at a preemption step, record the time since the previous checkpoint.
      # This represents training time that would otherwise have been wasted.
      # If another checkpoint has not been previously saved, measures the time
      # since program start.
      if self.reached_preemption(step):
        if len(self._checkpoints) > 1:
          previous_time = self._checkpoints[-2].time
        else:
          previous_time = _INIT_TIME
        assert self._last_checkpoint is not None
        duration = self._last_checkpoint.time - previous_time
        jax.monitoring.record_event_duration_secs(
            '/jax/checkpoint/write/preempt/duration_saved_secs',
            duration.total_seconds(),
        )
      final_ckpt_dir = self._get_save_directory(step, self.directory)
      utils.ensure_atomic_save(temp_ckpt_dir, final_ckpt_dir)
    return final_ckpt_dir

  def _finalize(self, temp_ckpt_dir: epath.Path):
    """Cleans up old checkpoints and synchronizes hosts."""
    if not self._all_checkpointers_are_sync:
      self.wait_until_finished(join_finalize_thread=False)
    final_ckpt_dir = self._finalize_checkpoint(temp_ckpt_dir)
    for step in self._steps_to_remove:
      self._delete_directory(step)

  def close(self):
    """Waits for outstanding operations to finish and closes Checkpointers."""
    self.wait_until_finished()
    for c in self._checkpointers.values():
      c.close()


@contextlib.contextmanager
def checkpoint_manager_context(*args, **kwargs):
  """Context manager for CheckpointManager.

  Initializes CheckpointManager and closes the object when the context is
  exited.

  Args:
    *args: Arguments to initialize CheckpointManager.
    **kwargs: Keyword arguments to initialize CheckpointManager.

  Usage::

    with checkpoint_manager_context(
        directory, checkpointers, options) as mngr:
      mngr.save(...)
      mngr.all_steps()

  Yields:
    CheckpointManager
  """
  manager = CheckpointManager(*args, **kwargs)
  try:
    yield manager
  finally:
    manager.close()