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

"""In-memory snapshots of the train state for elastic training with `elastic_backup_kind: snapshot`.

In snapshot mode MaxText keeps a recent copy of the train state in pinned host memory, using the Orbax
`Snapshotter`, so that the attempt after an elastic event can resume from that copy instead of from the last
persistent checkpoint.

The module holds no state of its own. The caller owns the `Snapshotter` and keeps it alive across attempts; in
MaxText that is the `snapshotter` slot of `elastic_utils.RetryCache`, which `get_train_func` fills, the elastic
event callback resets, `training_loop_iteration` saves into and `checkpointing.load_state_if_possible` restores
from.

A snapshot holds the train state in the checkpoint layout plus the state of the grain `ElasticIterator`, so that a
restore continues with the next batch rather than replaying data. `ElasticIterator.get_state()` is one global example
index, valid for any number of hosts; Orbax passes non-array leaves through the snapshotter unchanged. Snapshot mode
therefore requires `grain_use_elastic_iterator` (or synthetic data); the config validation enforces that.

The state is split along the data axis of the mesh, which spans the slices; on restore Orbax rebuilds it from the
replicas that are still reachable and reshards it onto the shardings it is given, which may be on a different mesh.
So the same snapshot serves pause/resume (same slices), a slice-down (fewer slices) and a scale-up (more slices).
"""

from flax import nnx
from grain import experimental as grain_experimental
import jax
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.utils import elastic_utils
from maxtext.utils import max_logging
from orbax.checkpoint import v1 as ocp

Snapshotter = ocp.training.pathways.Snapshotter


def create_snapshotter(config) -> Snapshotter | None:
  """Returns a new snapshotter in snapshot mode, and None otherwise.

  Args:
    config: MaxText config. `elastic_backup_kind` selects snapshot mode and `mesh_axes` locates the data axis.

  Returns:
    A `Snapshotter` that splits the state along the data axis of the mesh, which spans the slices, or None when
    `elastic_backup_kind` is not `snapshot` or elastic training is off.
  """
  if not elastic_utils.elastic_snapshot(config):
    return None
  if config.shard_optimizer_over_data:
    # TODO(b/574652982): Support in-memory snapshot recovery when `shard_optimizer_over_data=True`.
    # The snapshotter rebuilds an array from the replicas along the data axis that survived. State that is sharded
    # along that axis has no surviving replica once a slice is gone, so every restore after a slice loss falls back
    # to the checkpoint, and the snapshots cost host memory for nothing.
    max_logging.log(
        "WARNING: elastic_backup_kind=snapshot with shard_optimizer_over_data=True: the optimizer state is sharded"
        " across slices, so a snapshot cannot be restored after a slice is lost and the checkpoint is used instead."
    )
  return Snapshotter(replica_axis_index=config.mesh_axes.index("data"))


def reset_snapshotter(snapshotter: Snapshotter) -> Snapshotter:
  """Returns a new snapshotter that holds only the latest completed snapshot of `snapshotter`.

  For the slice-down event callback (before a scale-up nothing is in flight: `maybe_save_snapshot` waited for the
  last save). A save that was running when the slice died is in `block_until_ready` on buffers of that slice. If
  that call raises, Orbax logs it and frees the queue slot, and the old snapshotter would be fine to keep. If it
  hangs, or raises something the worker doesn't catch, the single queue slot stays taken for ever: every later save
  is skipped as "busy" and a later `Snapshotter.join` would never return. A new snapshotter, with an empty queue and
  a fresh worker, is the only recovery from those cases; it costs one parked thread per event. The old one is left
  without a snapshot so that it holds no pinned memory.

  Args:
    snapshotter: The snapshotter of the interrupted attempt.

  Returns:
    The snapshotter for the next attempt.
  """
  new_snapshotter = Snapshotter(replica_axis_index=snapshotter.replica_axis_index)
  # TODO(abhinavsing): Ask Orbax for a public API to reset and reuse a `Snapshotter` in place (e.g.
  # `Snapshotter.reset()`), and use it here. `Snapshotter.latest` only exposes the step, not the pinned arrays, and
  # `load()` consumes them onto devices, so the only way today to hand the last completed snapshot to a fresh worker
  # is to copy its private `_latest_snapshot` under its `_lock`. Both are private fields of an experimental module:
  # if Orbax renames them this raises `AttributeError` in the elastic event callback.
  # pylint: disable=protected-access
  with snapshotter._lock:
    new_snapshotter._latest_snapshot = snapshotter._latest_snapshot
    # The old snapshotter is not collected: its worker thread holds a reference to it and blocks on its queue for
    # ever. Without this it would keep a full pinned-host copy of the train state alive per elastic event.
    snapshotter._latest_snapshot = None
  # pylint: enable=protected-access
  return new_snapshotter


def latest_snapshot_step(snapshotter: Snapshotter | None) -> int | None:
  """Returns the step of the latest completed snapshot, or None if there is no snapshotter or no snapshot yet."""
  if snapshotter is None:
    return None
  if latest := snapshotter.latest:
    return latest.step
  return None


def wait_for_snapshot(snapshotter: Snapshotter, step: int) -> bool:
  """Waits for the snapshotter's queue to drain and reports whether the snapshot of `step` is the latest one.

  A save that failed or was skipped still drains the queue, so this returns promptly with False in those cases.

  TODO(abhinavsing): `Snapshotter.join` has no timeout. A worker blocked in `block_until_ready` on buffers of a
  slice that went away would hold this for ever if that call hangs rather than raises; ask Orbax for
  `join(timeout=...)`, and for the worker to survive exceptions other than `JaxRuntimeError`/`RuntimeError`.

  Args:
    snapshotter: The snapshotter of the run.
    step: The step whose snapshot to wait for.

  Returns:
    Whether the snapshot of `step` is the latest completed one once the queue is empty.
  """
  snapshotter.join()
  return latest_snapshot_step(snapshotter) == step


def _elastic_iterator(data_iterator) -> grain_experimental.ElasticIterator | None:
  """Returns the `ElasticIterator` inside `data_iterator`, or None for synthetic data.

  Raises:
    ValueError: If `data_iterator` is a non-synthetic iterator that does not wrap `grain.experimental.ElasticIterator`.
  """
  if data_iterator is None or isinstance(
      data_iterator,
      (synthetic_data_processing.SyntheticDataIterator, synthetic_data_processing.PlaceHolderDataIterator),
  ):
    return None
  if isinstance(data_iterator, multihost_dataloading.MultiHostDataLoadIterator):
    for candidate in (data_iterator.local_iterator, getattr(data_iterator, "dataloader", None)):
      if isinstance(candidate, grain_experimental.ElasticIterator):
        return candidate
  raise ValueError(
      "Snapshot mode requires a MultiHostDataLoadIterator wrapping grain's ElasticIterator (or synthetic data),"
      f" got {type(data_iterator).__name__}."
  )


def _mesh_of(pytree):
  """Returns the mesh of the first leaf with a `NamedSharding`, or None."""
  for leaf in jax.tree.leaves(pytree):
    if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) and isinstance(leaf.sharding, jax.sharding.NamedSharding):
      return leaf.sharding.mesh
  return None


def _replicate_single_device_sharded_arrays(pytree):
  """Replicates any single-device sharded arrays across the whole mesh.

  The snapshotter splits every array along the replica axis of its mesh, which a single-device array doesn't have.

  Args:
    pytree: A pytree of arrays or `jax.ShapeDtypeStruct`s.

  Returns:
    `pytree`, with each single-device sharded leaf replicated across the mesh of its other leaves.
  """
  mesh = _mesh_of(pytree)
  if mesh is None:
    return pytree
  replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def _replicate(x):
    if isinstance(x, (jax.Array, jax.ShapeDtypeStruct)) and isinstance(x.sharding, jax.sharding.SingleDeviceSharding):
      if isinstance(x, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=replicated_sharding)
      return jax.device_put(x, replicated_sharding)
    return x

  return jax.tree.map(_replicate, pytree)


def _restore_original_shardings(restored, target):
  """Moves restored arrays back onto the shardings and memory kinds of `target`.

  `Snapshotter.load` returns every array in device memory, and single-device arrays replicated by
  `_replicate_single_device_sharded_arrays`. This undoes both, e.g. for host-offloaded optimizer state.

  Args:
    restored: The arrays returned by `Snapshotter.load`.
    target: A pytree with the same structure, whose leaves carry the shardings to restore onto.

  Returns:
    `restored`, with each array on the sharding and memory kind of the matching leaf of `target`.
  """

  def _put(restored_leaf, target_leaf):
    if not isinstance(restored_leaf, jax.Array) or not isinstance(target_leaf, (jax.Array, jax.ShapeDtypeStruct)):
      return restored_leaf
    sharding = target_leaf.sharding
    if sharding is None:
      return restored_leaf
    if restored_leaf.sharding == sharding and restored_leaf.sharding.memory_kind == sharding.memory_kind:
      return restored_leaf
    if jax.dtypes.issubdtype(restored_leaf.dtype, jax.dtypes.prng_key):
      # Move PRNG keys through their key data, as the snapshotter does.
      key_data = jax.device_put(jax.random.key_data(restored_leaf), sharding)
      return jax.random.wrap_key_data(key_data, dtype=restored_leaf.dtype)
    return jax.device_put(restored_leaf, sharding)

  return jax.tree.map(_put, restored, target)


def save_snapshot(snapshotter: Snapshotter, state: nnx.State, step: int, data_iterator=None) -> None:
  """Starts copying `state` and the data iterator state to pinned host memory.

  The copy runs on the snapshotter's worker thread; Orbax skips this snapshot if an earlier one is still being
  copied. Call it with the index of the step that produced `state`, so that a snapshot and a checkpoint saved after
  the same step carry the same step.

  Args:
    snapshotter: The snapshotter to save with.
    state: The train state to snapshot.
    step: The index of the step that produced `state`.
    data_iterator: The training data iterator. The state of its `ElasticIterator` goes into the snapshot; with
      synthetic data there is none.
  """
  iterator = _elastic_iterator(data_iterator)
  if iterator is None:
    iterator_state = None
  else:
    iterator_state = iterator.get_state()
  snapshot = _replicate_single_device_sharded_arrays(
      {"items": train_state_nnx.to_checkpoint_dict(state), "data_iterator": iterator_state}
  )
  max_logging.log(f"Saving in-memory snapshot at step {step}...")
  snapshotter.save(step, snapshot)


def should_save_snapshot(config, step: int, checkpoint_manager) -> bool:
  """Whether `training_loop_iteration` should snapshot after `step`.

  Snapshots are taken every `elastic_snapshot_interval` steps, except while a checkpoint save is still running:
  copying to the host at the same time contends for host memory and bandwidth.

  Args:
    config: MaxText config.
    step: The index of the step that just finished.
    checkpoint_manager: The checkpoint manager, or None if checkpointing is disabled.
  """
  if step % config.elastic_snapshot_interval != 0:
    return False
  # TODO(abhinavsing): Support snapshotting and checkpointing concurrently so that background checkpoint saves do not
  # block periodic snapshots.
  if checkpoint_manager is not None and checkpointing.is_saving_in_progress(checkpoint_manager):
    max_logging.log(f"Skipping the in-memory snapshot at step {step} because a checkpoint save is in progress.")
    return False
  return True


def maybe_save_snapshot(snapshotter: Snapshotter | None, config, state, step: int, checkpoint_manager, data_iterator):
  """Snapshots after `step` when it is due, and always before interrupting for a scale-up.

  Call this at the end of each train step, after the checkpoint save. Does nothing outside snapshot mode.

  When new slices are available, `maybe_elastic_scale_up` interrupts the attempt at the start of the next step.
  Taking a snapshot right before that, and waiting for it to complete, means the attempt on the larger mesh resumes
  from this very step instead of from the last periodic snapshot or checkpoint. The waits block until the snapshotter
  is idle; a snapshot that failed is logged here, so that the steps the next attempt loses are explained.

  Args:
    snapshotter: The snapshotter of the run, or None outside snapshot mode.
    config: MaxText config.
    state: The train state after step `step`.
    step: The index of the step that produced `state`.
    checkpoint_manager: The checkpoint manager, or None if checkpointing is disabled.
    data_iterator: The training data iterator.
  """
  if snapshotter is None:
    return
  if elastic_utils.is_scale_up_event(config):
    max_logging.log(f"New slices are available. Snapshotting step {step} before scaling up.")
    if checkpoint_manager is not None:
      checkpointing.wait_until_finished(checkpoint_manager)
    # Let an earlier snapshot finish first, so that this one isn't skipped as "busy". This blocks the main thread,
    # by design: the attempt is interrupted at the top of the next step either way, and the wait is what lets the
    # next attempt resume from this step rather than from the last periodic snapshot.
    snapshotter.join()
    save_snapshot(snapshotter, state, step, data_iterator)
    if not wait_for_snapshot(snapshotter, step):
      max_logging.log(
          f"WARNING: the in-memory snapshot of step {step} did not complete before the scale-up. The next attempt"
          f" resumes from the latest snapshot (step {latest_snapshot_step(snapshotter)}) or checkpoint."
      )
    return
  if should_save_snapshot(config, step, checkpoint_manager):
    save_snapshot(snapshotter, state, step, data_iterator)


def should_restore_snapshot(snapshotter: Snapshotter | None, checkpoint_manager) -> bool:
  """Whether the next attempt should restore from the snapshot rather than from the checkpoint.

  The snapshot wins unless the latest checkpoint is at a later step, which happens when a periodic snapshot was
  skipped (snapshotter busy, or a checkpoint save in progress) and a checkpoint landed after it.

  Args:
    snapshotter: The snapshotter of the run, or None outside snapshot mode.
    checkpoint_manager: The checkpoint manager of this attempt, or None if checkpointing is disabled.
  """
  snapshot_step = latest_snapshot_step(snapshotter)
  if snapshot_step is None:
    return False
  if checkpoint_manager is None:
    return True
  checkpoint_step = checkpointing.latest_step(checkpoint_manager)
  # TODO(abhinavsing): Revisit whether a slightly older snapshot (e.g. a few steps behind the latest checkpoint) is
  # still faster to restore and replay than loading the checkpoint from storage.
  if checkpoint_step is not None and checkpoint_step > snapshot_step:
    max_logging.log(
        f"Restoring the checkpoint at step {checkpoint_step} rather than the in-memory snapshot at step"
        f" {snapshot_step}, which is older."
    )
    return False
  return True


def _load_snapshot(snapshotter: Snapshotter, abstract_state: nnx.State, config, abstract_iterator_state):
  """Restores the latest snapshot onto the shardings of `abstract_state`, which may be on a different mesh.

  Args:
    snapshotter: The snapshotter holding the snapshot.
    abstract_state: The abstract train state of this attempt.
    config: MaxText config.
    abstract_iterator_state: A state dict with the same tree structure as `ElasticIterator.get_state()`, or None
      without an `ElasticIterator`. The restore target must have the same structure as the saved snapshot.

  Returns:
    The restored NNX state and the restored `ElasticIterator` state dict (or None without an iterator).
  """
  target_items = train_state_nnx.to_checkpoint_dict(abstract_state)
  target = {
      "items": _replicate_single_device_sharded_arrays(target_items),
      "data_iterator": abstract_iterator_state,
  }
  restored = snapshotter.load(target)
  items = _restore_original_shardings(restored["items"], target_items)
  state = checkpointing.restored_linen_to_nnx(items, abstract_state, config=config)
  return state, restored["data_iterator"]


def load_snapshot(
    snapshotter: Snapshotter | None, abstract_state: nnx.State, config, data_iterator=None
) -> nnx.State | None:
  """Restores the train state and the data iterator state from the latest completed snapshot, if there is one.

  Only a previous attempt in this process can have taken a snapshot, so this returns None on the first attempt
  and outside snapshot mode. Use `should_restore_snapshot` first to decide between the snapshot and the checkpoint.

  Args:
    snapshotter: The snapshotter of the run, or None outside snapshot mode.
    abstract_state: The abstract train state of this attempt, with the shardings to restore onto.
    config: MaxText config, used to check the restored weights against the model.
    data_iterator: The training data iterator of this attempt. Its `ElasticIterator` is positioned where the
      snapshot was taken.

  Returns:
    The restored state, in which the leaves a snapshot doesn't hold (e.g. caches) are still `jax.ShapeDtypeStruct`s,
    or None if there is no snapshot or the restore failed in a way a checkpoint restore can recover from. In both
    None cases the caller should restore the checkpoint instead.

  Raises:
    jax.errors.JaxRuntimeError: If a slice went down during the restore. `elastic_retry` retries it.
  """
  if snapshotter is None:
    return None
  step = latest_snapshot_step(snapshotter)
  if step is None:
    return None
  iterator = _elastic_iterator(data_iterator)
  if iterator is None:
    abstract_iterator_state = None
  else:
    abstract_iterator_state = iterator.get_state()
  max_logging.log(f"Restoring the in-memory snapshot at step {step}...")
  try:
    state, iterator_state = _load_snapshot(snapshotter, abstract_state, config, abstract_iterator_state)
  except (jax.errors.JaxRuntimeError, RuntimeError) as e:
    if isinstance(e, jax.errors.JaxRuntimeError) and elastic_utils.is_slice_down_error(e):
      raise
    # No replica of some array survived (`RuntimeError`), or the restore hit a non-slice-down JAX error;
    # return None so the caller can fall back.
    max_logging.log(f"Restoring the in-memory snapshot at step {step} failed: {e}")
    return None
  if iterator is not None:
    iterator.set_state(iterator_state)
    max_logging.log(f"Restored the data iterator state: {iterator_state}.")
  max_logging.log(f"Restored the in-memory snapshot at step {step}.")
  return state
