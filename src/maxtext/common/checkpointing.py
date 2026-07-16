# Copyright 2023–2025 Google LLC
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


"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

import time
from typing import Any, Optional

from absl import flags
import datetime
from etils import epath
from flax import nnx
from flax.training import train_state
import jax
from maxtext.utils.globals import DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator
from maxtext.input_pipeline.multihost_dataloading import RemoteIteratorWrapper
from maxtext.input_pipeline.synthetic_data_processing import PlaceHolderDataIterator
from maxtext.common import train_state_nnx
from maxtext.utils import exceptions
from maxtext.utils import max_logging
from maxtext.utils import gcs_utils
from maxtext.utils import elastic_utils
from maxtext.checkpoint_conversion.utils.load_dynamic import load_safetensors_dynamic_state

import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager
# pylint: disable=too-many-positional-arguments
import dataclasses
import json

import grain
from grain.python import PyGrainCheckpointHandler
from grain.experimental import ElasticIterator

CheckpointManager = ocp.CheckpointManager
CheckpointManagerOptions = ocp.CheckpointManagerOptions
Composite = ocp.args.Composite
PyTreeCheckpointHandler = ocp.PyTreeCheckpointHandler
EmergencyCheckpointManager = emergency_checkpoint_manager.CheckpointManager
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = emergency_checkpoint_manager.PersistentCheckpointOptions
EmergencyReplicatorCheckpointManager = emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager


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


def _weight_mismatches(want, have, path=()):
  """Returns `(path, problem)` for each weight in `want` that `have` didn't restore faithfully.

  A weight is wrong if the checkpoint didn't carry it -- absent, or left by Orbax as an
  unmaterialized ShapeDtypeStruct -- or carried it at a different shape. Only the shape can
  disagree: Orbax casts a restored array to the target's dtype.
  """
  if isinstance(want, dict):
    out = []
    for k, v in want.items():
      out.extend(_weight_mismatches(v, have.get(k) if isinstance(have, dict) else None, path + (k,)))
    return out
  name = "/".join(str(p) for p in path)
  if have is None or isinstance(have, jax.ShapeDtypeStruct):
    return [(name, f"missing (model expects {getattr(want, 'shape', '?')} {getattr(want, 'dtype', '?')})")]
  want_shape, got_shape = getattr(want, "shape", None), getattr(have, "shape", None)
  if want_shape is not None and got_shape is not None and tuple(want_shape) != tuple(got_shape):
    return [(name, f"shape {tuple(got_shape)} but the model expects {tuple(want_shape)}")]
  return []


def _expected_and_restored_params(abstract_nnx_state, restored_linen):
  """Returns the model's expected weights and the checkpoint's restored weights, as pure dicts.

  Splits the abstract by Variable type (nnx.Param) so only real weights are compared --
  rngs/dropout/batch stats live in `nnx_aux` and are restored separately.
  """
  want = nnx.split_state(abstract_nnx_state, nnx.Param, ...)[0].to_pure_dict().get("model", {})
  have = restored_linen.get("params", {}).get("params", {})
  return want, have


def _raise_on_weight_mismatch(want, have):
  """Raises if the restored weights (`have`) don't match what the model expects (`want`).

  Both are pure dicts, so this works for any structure. `partial_restore` returns a weight the
  checkpoint doesn't carry as an unmaterialized ShapeDtypeStruct, and Orbax restores a stored
  array at its own shape rather than the target's. Either way it reaches the model as an
  untrained init value (a silent accuracy loss) or fails much later, deep in the first step,
  without naming the weight.
  """
  problems = _weight_mismatches(want, have)
  if not problems:
    return
  lines = "\n".join(f"  - '{p}': {why}" for p, why in problems)
  raise ValueError(
      "Checkpoint does not match the model:\n"
      f"{lines}\n"
      "Verify the checkpoint matches the model architecture (emb_dim, mlp_dim, num layers, scan_layers)."
  )


def _linen_items_to_nnx(restored_linen, abstract_nnx_state):
  """Reshapes a restored Linen-layout `items` dict into an NNX state.

  The inverse of `to_checkpoint_dict`, over the same `split_for_checkpoint` partition. The Linen
  weights + optimizer fill `linen_state`; the `nnx_aux` state (rngs/dropout, batch stats, custom
  variables) fills `aux`; the two are recombined with `nnx.merge_state`. The split copies, so the
  caller's abstract is untouched. Leaves the checkpoint didn't carry -- including the caches it
  never stores -- stay unmaterialized `ShapeDtypeStruct`s; the caller fills them from a fresh init.
  """
  linen_state, aux_state, ephemeral = train_state_nnx.split_for_checkpoint(abstract_nnx_state)
  weights = train_state_nnx.from_linen_checkpoint_dict(restored_linen)
  if "model" in weights:
    nnx.replace_by_pure_dict(linen_state, {"model": weights["model"]})
  if "optimizer" in weights:
    nnx.replace_by_pure_dict(linen_state, {"optimizer": weights["optimizer"]})

  nnx_aux = restored_linen.get("nnx_aux")
  if nnx_aux:
    nnx.replace_by_pure_dict(aux_state, nnx_aux)

  return nnx.merge_state(linen_state, aux_state, ephemeral)


def _load_linen_checkpoint_into_nnx(
    path,
    abstract_nnx_state,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
):
  """Restores a Linen-layout checkpoint into an NNX state (pure_nnx resume).

  Restores a Linen-shape target that includes `nnx_aux`, then reshapes back via
  `_linen_items_to_nnx`. rngs/dropout/batch stats come from `items/nnx_aux` when
  present, else keep their fresh init value. A genuinely-missing weight raises.
  """
  max_logging.log(f"Restoring Linen-layout checkpoint into NNX state at {path}")
  linen_abstract = train_state_nnx.to_checkpoint_dict(abstract_nnx_state)
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=checkpoint_storage_concurrent_gb,
          save_concurrent_gb=checkpoint_storage_concurrent_gb,
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
      )
  )
  restore_args = ocp.checkpoint_utils.construct_restore_args(linen_abstract)
  restored = ocp.args.PyTreeRestore(item=linen_abstract, restore_args=restore_args, partial_restore=True)
  restored = ckptr.restore(epath.Path(path), args=restored)
  _raise_on_weight_mismatch(*_expected_and_restored_params(abstract_nnx_state, restored))
  return _linen_items_to_nnx(restored, abstract_nnx_state)


def _restore_emergency_linen_checkpoint_into_nnx(
    checkpoint_manager,
    step,
    abstract_nnx_state,
    map_to_pspec,
):
  """Restores an emergency Linen-layout checkpoint into an NNX state.

  The `nnx_aux` subtree is stored inside `items`, so an emergency checkpoint
  carries it too; it's restored when present and otherwise kept at its fresh
  init value. A genuinely-missing weight raises.
  """
  max_logging.log(f"Restoring emergency Linen-layout checkpoint into NNX state at step {step}")
  linen_abstract = train_state_nnx.to_checkpoint_dict(abstract_nnx_state)
  restore_args = jax.tree_util.tree_map(map_to_pspec, linen_abstract)
  checkpoint_args = ocp.args.PyTreeRestore(
      item=linen_abstract,
      restore_args=restore_args,
      partial_restore=True,
  )
  restored = checkpoint_manager.restore(step, args=Composite(state=checkpoint_args)).state
  _raise_on_weight_mismatch(*_expected_and_restored_params(abstract_nnx_state, restored))
  return _linen_items_to_nnx(restored, abstract_nnx_state)


def _load_full_state_from_path(
    path,
    abstract_unboxed_pre_state,
    enable_orbax_v1,
    checkpoint_conversion_fn,
    source_checkpoint_layout,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
):
  """Load full state from checkpoint at specified path.

  Args:
    path: path to checkpoint
    abstract_unboxed_pre_state: an abstract state that Orbax matches type
      against.
    enable_orbax_v1: whether to use orbax v1 or the previously supported v0.
    checkpoint_conversion_fn: user-provided function to convert checkpoint to
      maxtext-supported state.
    source_checkpoint_layout: String representation of the checkpoint layout of
      the source checkpoint.
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    use_ocdbt: Whether to use OCDBT format.
    use_zarr3: Whether to use Zarr3 format.

  Returns:
    The loaded state.
  """

  if enable_orbax_v1:
    if source_checkpoint_layout == "orbax":
      # pure_nnx saves in the Linen on-disk layout; reshape it back into the NNX state.
      if isinstance(abstract_unboxed_pre_state, nnx.State):
        return _load_linen_checkpoint_into_nnx(
            path, abstract_unboxed_pre_state, checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3
        )
      context = ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.ORBAX)
      with context:
        return ocp_v1.load_pytree(path, abstract_unboxed_pre_state)
    elif source_checkpoint_layout == "safetensors":
      context = ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS)
      with context:
        metadata = ocp_v1.pytree_metadata(path)
        simple_abstract_state = metadata.metadata
        shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

        def combine_sharding(sds, shardings):
          return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings)

        sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
        pre_transformed_state = ocp_v1.load_pytree(path, sharded_abstract_state)
      state = checkpoint_conversion_fn(pre_transformed_state)
      return state
    else:
      raise ocp_v1.errors.InvalidLayoutError(f"Unknown checkpoint layout: {source_checkpoint_layout}")
  else:
    # pure_nnx saves in the Linen on-disk layout; reshape it back into the NNX state.
    if isinstance(abstract_unboxed_pre_state, nnx.State):
      return _load_linen_checkpoint_into_nnx(
          path,
          abstract_unboxed_pre_state,
          checkpoint_storage_concurrent_gb,
          use_ocdbt,
          use_zarr3,
      )

    # Original v0 logic.
    p = epath.Path(path)
    handler = ocp.PyTreeCheckpointHandler(
        restore_concurrent_gb=checkpoint_storage_concurrent_gb,
        save_concurrent_gb=checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )
    # Only Linen TrainState reaches here; nnx.State returned above.
    restore_target = abstract_unboxed_pre_state
    # Provide sharding info to ensure restoration returns JAX arrays (not NumPy arrays).
    restore_args = jax.tree_util.tree_map(
        lambda x: ocp.type_handlers.ArrayRestoreArgs(sharding=x.sharding),
        restore_target,
    )
    return ocp.Checkpointer(handler).restore(p, restore_target, restore_args=restore_args)


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: None | str = None,
    orbax_logger: Any = None,  # pytype: disable=attribute-error
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    enable_continuous_checkpointing: bool = False,
    max_num_checkpoints_to_keep: int = 10,
    checkpoint_storage_concurrent_gb: int = 96,
    enable_single_controller: bool = False,
    colocated_python_checkpointing: bool = False,
    enable_single_replica_ckpt_restoring: bool = False,
    enable_autocheckpoint: bool = False,
    todelete_subdir: str | None = None,
    todelete_full_path: str | None = None,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None

  max_logging.log(f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}")

  # Base configuration for all dataset types
  item_names = ("items",)
  # we need to use ocdbt and zarr3 to control max file size in the checkpoint
  item_handlers = {
      "items": PyTreeCheckpointHandler(
          restore_concurrent_gb=checkpoint_storage_concurrent_gb,
          save_concurrent_gb=checkpoint_storage_concurrent_gb,
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
      )
  }

  if dataset_type is not None and dataset_type == "grain":
    item_names += ("iter",)
    item_handlers["iter"] = GrainCheckpointHandler()  # pyrefly: ignore[bad-assignment]

  # local storage checkpoint needs parent directory created
  p = gcs_utils.mkdir_and_check_permissions(checkpoint_dir)
  if enable_continuous_checkpointing:
    max_logging.log("Enabling policy for continuous checkpointing.")
    save_decision_policy = save_decision_policy_lib.ContinuousCheckpointingPolicy()
  elif enable_autocheckpoint:
    max_logging.log("Enabling policy for autocheckpoint.")
    save_decision_policy = save_decision_policy_lib.AnySavePolicy(
        [
            save_decision_policy_lib.PreemptionCheckpointingPolicy(),
            save_decision_policy_lib.FixedIntervalPolicy(save_interval_steps),
        ]
    )
  else:
    max_logging.log("Enabling policy for fixed interval checkpointing.")
    save_decision_policy = save_decision_policy_lib.FixedIntervalPolicy(interval=save_interval_steps)
  preservation_policy = preservation_policy_lib.LatestN(max_num_checkpoints_to_keep)

  async_options = None
  if enable_continuous_checkpointing:
    async_options = ocp.AsyncOptions(
        timeout_secs=int(datetime.timedelta(minutes=60).total_seconds()),
    )
  manager = CheckpointManager(
      p,
      item_names=item_names,
      item_handlers=item_handlers,
      options=CheckpointManagerOptions(
          create=True,
          enable_async_checkpointing=use_async,
          save_decision_policy=save_decision_policy,
          preservation_policy=preservation_policy,
          async_options=async_options,
          todelete_subdir=todelete_subdir,
          todelete_full_path=todelete_full_path,
      ),
      logger=orbax_logger,
  )

  max_logging.log("Checkpoint manager created!")
  return manager


def create_orbax_emergency_checkpoint_manager(
    local_checkpoint_dir: str,
    persistent_checkpoint_dir: str,
    global_mesh: jax.sharding.Mesh,
    abstract_state: Any,
    local_save_interval_steps: int,
    persistent_save_interval_steps: int,
    orbax_logger: Any = None,  # pytype: disable=attribute-error
):
  """Returns an emergency checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency checkpoint manager...")

  # Only create local directories if running on GPUs as the previous directory structure might be assumed by TPUs.
  if global_mesh.devices.flatten()[0].platform == "gpu":
    # pylint: disable=protected-access
    local_checkpoint_dir = f"{local_checkpoint_dir}/{jax._src.distributed.global_state.process_id}"
    local_p = epath.Path(local_checkpoint_dir)
    local_p.mkdir(exist_ok=True, parents=True)

  persistent_p = gcs_utils.mkdir_and_check_permissions(persistent_checkpoint_dir)

  # pure_nnx saves via to_checkpoint_dict (Linen params/opt_state/step plus an nnx_aux
  # subtree), but the emergency manager restores against the abstract it is built with.
  # Convert it the same way so it matches what is on disk; restore reshapes back to NNX.
  if isinstance(abstract_state, nnx.State):
    abstract_state = train_state_nnx.to_checkpoint_dict(abstract_state)

  manager = EmergencyCheckpointManager(
      local_checkpoint_dir,
      persistent_p,
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=emergency_checkpoint_manager.CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=local_save_interval_steps),
          persistent=PersistentCheckpointOptions(save_interval_steps=persistent_save_interval_steps),
      ),
      logger=orbax_logger,
  )

  max_logging.log("Emergency checkpoint manager created!")
  return manager


def create_orbax_emergency_replicator_checkpoint_manager(
    local_checkpoint_dir: str,
    save_interval_steps: int,
    global_mesh: jax.sharding.Mesh,
    colocated_python_checkpointing: bool = False,
):
  """Returns an emergency replicator checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency replicator checkpoint manager...")

  manager = EmergencyReplicatorCheckpointManager(
      epath.Path(local_checkpoint_dir),
      options=emergency_replicator_checkpoint_manager.ReplicatorCheckpointManagerOptions(
          save_interval_steps=save_interval_steps,
          use_colocated_python=colocated_python_checkpointing,
      ),
      global_mesh=global_mesh,
  )

  max_logging.log("Emergency replicator checkpoint manager created!")
  return manager


def replicator_error_handler(config: Any):
  """Replicator error handler to handle errors in replicator service."""
  if config.enable_multi_tier_checkpointing:
    local_dir = config.local_checkpoint_directory
    replicator_errors_file = f"{local_dir}/replicator.errors"
    replicator_failed_file = f"{local_dir}/replicator.failed"
    process_replicator_error_file(replicator_errors_file)

    # if the replicator.failed file exists, then we have a fatal error
    is_fatal = process_replicator_error_file(replicator_failed_file)
    if is_fatal:
      raise ValueError("Replicator fatal error found in replicator.failed file.")


def process_replicator_error_file(error_file: str) -> bool:
  """Handles replicator errors by reading, logging, cleaning the error file."""
  error_file_path_exists = epath.Path(error_file).exists()
  if error_file_path_exists:
    max_logging.log(f"replicator_error_handler: file found: {error_file}.")
    read_replicator_error_file(error_file)
    cleanup_replicator_error_file(error_file)

  return error_file_path_exists


def read_replicator_error_file(error_file: str):
  """Read replicator errors file."""
  try:
    error_data = epath.Path(error_file).read_text()
    max_logging.log(f"Contents of replicator error file:\n{error_data}")
  except (OSError, ValueError) as e:
    max_logging.log("replicator_error_handler: Failed to read contents of failed" f" file: {e}")


def cleanup_replicator_error_file(error_file: str):
  """Clean up replicator errors file."""
  try:
    epath.Path(error_file).unlink()
  except (OSError, ValueError) as e:
    max_logging.log("replicator_error_handler: Failed to remove replicator errors file:" f" {e}")


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


def _find_idx(array: np.ndarray, replica_axis_idx: int):
  """Returns the index along given dimension that the current host belongs to."""
  idx = None
  for idx, val in np.ndenumerate(array):
    if val.process_index == jax.process_index():
      break
  return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
  """Returns the devices from the replica that current host belongs to.

  Replicas are assumed to be restricted to the first axis.

  Args:
    device_array: devices of the mesh that can be obtained by mesh.devices()
    replica_axis_idx: axis dimension along which replica is taken

  Returns:
    devices inside the replica that current host is in
  """
  idx = _find_idx(device_array, replica_axis_idx)
  replica_result = np.take(device_array, idx, axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)


def _prepare_scaled_down_grain_restore_args(
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


def _restore_grain_iterator(
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
    grain_restore_args = _prepare_scaled_down_grain_restore_args(
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


def load_state_if_possible(
    checkpoint_manager: CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    checkpoint_storage_concurrent_gb: int,
    abstract_unboxed_pre_state: train_state.TrainState | nnx.State,
    enable_single_replica_ckpt_restoring: bool | None = False,
    dataset_type: str | None = "tfds",
    step: int = -1,  # -1 means latest
    use_ocdbt=True,
    use_zarr3=True,
    enable_orbax_v1=False,
    checkpoint_conversion_fn=None,
    source_checkpoint_layout="orbax",
    expansion_factor_real_data: int = -1,
    maxtext_config: Any | None = None,
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint
      manager, load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint
      manager, load full state from a full state checkpoint at this path.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitn
      with SingleReplicaArrayHandler
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    enable_orbax_v1: bool flag for enabling Orbax v1.
    checkpoint_conversion_fn: function for converting checkpoint to Orbax v1.
    source_checkpoint_layout: Optional checkpoint context to use for loading,
    provided in string format with the default being "orbax".

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    step = checkpoint_manager.latest_step() if step < 0 else step  # pyrefly: ignore[bad-assignment]
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      def map_to_pspec(data):
        if not enable_single_replica_ckpt_restoring:
          return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)
        pspec = data.sharding.spec
        mesh = data.sharding.mesh
        replica_axis_index = 0
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

        return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(mesh, pspec),
            single_replica_sharding=single_replica_sharding,
            global_shape=data.shape,
            dtype=data.dtype,
        )

      if enable_single_replica_ckpt_restoring:
        array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
            replica_axis_index=0,
            broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
        )
        ocp.type_handlers.register_type_handler(jax.Array, array_handler, override=True)

      # pure_nnx saves in the Linen on-disk layout; restore that layout (weights +
      # opt_state + step + nnx_aux), restoring the grain iterator in place when
      # present, then reshape it back into the NNX state.
      # (Emergency managers use their own restore path below.)
      if isinstance(abstract_unboxed_pre_state, nnx.State) and not isinstance(
          checkpoint_manager,
          (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager),
      ):
        linen_abstract = train_state_nnx.to_checkpoint_dict(abstract_unboxed_pre_state)
        restore_args = jax.tree_util.tree_map(map_to_pspec, linen_abstract)
        checkpoint_args = ocp.args.PyTreeRestore(item=linen_abstract, restore_args=restore_args, partial_restore=True)
        if (
            dataset_type == "grain"
            and data_iterator
            and not isinstance(data_iterator, PlaceHolderDataIterator)
            and (checkpoint_manager.directory / str(step) / "iter").exists()
        ):
          restored, _ = _restore_grain_iterator(
              checkpoint_manager, step, data_iterator, checkpoint_args, expansion_factor_real_data
          )
        else:
          restored = checkpoint_manager.restore(step, args=Composite(items=checkpoint_args))
        _raise_on_weight_mismatch(*_expected_and_restored_params(abstract_unboxed_pre_state, restored["items"]))
        restored_nnx = _linen_items_to_nnx(restored["items"], abstract_unboxed_pre_state)
        return ({"items": restored_nnx}, None)

      if isinstance(abstract_unboxed_pre_state, nnx.State) and isinstance(
          checkpoint_manager,
          (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager),
      ):
        restored = _restore_emergency_linen_checkpoint_into_nnx(
            checkpoint_manager,
            step,
            abstract_unboxed_pre_state,
            map_to_pspec,
        )
        return (
            restored,
            None,
        )

      # Only Linen TrainState reaches here; the NNX cases returned above.
      restore_target = abstract_unboxed_pre_state
      restore_args = jax.tree_util.tree_map(map_to_pspec, restore_target)
      checkpoint_args = ocp.args.PyTreeRestore(
          item=restore_target,
          restore_args=restore_args,
          partial_restore=True,
      )

      match (checkpoint_manager, dataset_type, data_iterator):
        # Case 1: Matches if 'checkpoint_manager' is an instance of either EmergencyCheckpointManager
        # or EmergencyReplicatorCheckpointManager. The '_' indicates that 'dataset_type' and
        # 'data_iterator' can be any value and aren't used in this pattern.
        case (checkpoint_manager, _, _) if isinstance(
            checkpoint_manager,
            (
                EmergencyCheckpointManager,
                EmergencyReplicatorCheckpointManager,
            ),
        ):
          restored = checkpoint_manager.restore(step, args=Composite(state=checkpoint_args)).state
          return (
              restored,
              None,
          )
        # Case 2: Matches if dataset type is "grain" and the data iterator is not a
        # PlaceHolderDataIterator and a specific checkpoint file exists for the iterator
        case (
            checkpoint_manager,
            dataset_type,
            data_iterator,
        ) if (
            dataset_type == "grain"
            and data_iterator
            and not isinstance(data_iterator, PlaceHolderDataIterator)
            and (checkpoint_manager.directory / str(step) / "iter").exists()
        ):
          return _restore_grain_iterator(
              checkpoint_manager,
              step,
              data_iterator,
              checkpoint_args,
              expansion_factor_real_data,
          )
        # Case 3: Default/Fallback case.
        # This case acts as a wildcard ('_') and matches if none of the preceding cases were met.
        case _:
          restored = checkpoint_manager.restore(step, args=Composite(items=checkpoint_args))
          return (restored, None)

  if source_checkpoint_layout == "safetensors_dynamic":
    path = load_parameters_from_path or load_full_state_from_path
    max_logging.log(f"Dynamic On-the-Fly Formatting: Loading SafeTensors from {path}")

    return load_safetensors_dynamic_state(path, abstract_unboxed_pre_state, maxtext_config)
  elif load_parameters_from_path != "":
    if isinstance(abstract_unboxed_pre_state, nnx.State):
      _, params, _ = nnx.split(abstract_unboxed_pre_state.model, nnx.Param, ...)
    else:
      params = abstract_unboxed_pre_state.params

    restored_params = load_params_from_path(
        load_parameters_from_path,
        params,
        checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"Loading full state from path: {load_full_state_from_path}")
    restored_state = _load_full_state_from_path(
        path=load_full_state_from_path,
        abstract_unboxed_pre_state=abstract_unboxed_pre_state,
        enable_orbax_v1=enable_orbax_v1,
        checkpoint_conversion_fn=checkpoint_conversion_fn,
        source_checkpoint_layout=source_checkpoint_layout,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )
    return {"items": restored_state}, None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> Any | None:  # pytype: disable=attribute-error
  """Setup checkpoint logger.
  Args:
    config
  Returns:
    CloudLogger
  """
  orbax_cloud_logger = None
  max_logging.log("Setting up checkpoint logger...")
  if config.enable_checkpoint_cloud_logger:
    logger_name = f"goodput_{config.run_name}"
    orbax_cloud_logger = ocp.logging.CloudLogger(
        options=ocp.logging.CloudLoggerOptions(job_name=config.run_name, logger_name=logger_name)
    )
    max_logging.log("Successfully set up checkpoint cloud logger.")

  return orbax_cloud_logger


def load_params_from_path(
    load_parameters_from_path,
    abstract_unboxed_params,
    checkpoint_storage_concurrent_gb,
    use_ocdbt=True,
    use_zarr3=True,
):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")

  # On disk the weights live at `params/params/...`: an outer key naming the item, and Flax's
  # `params` collection inside it. A Linen TrainState.params is that collection; an NNX params
  # state sits one level below it (bare weights), so wrap it going in and unwrap it coming out.
  is_nnx = isinstance(abstract_unboxed_params, nnx.State)
  want = abstract_unboxed_params.to_pure_dict() if is_nnx else abstract_unboxed_params
  params_collection = {"params": want} if is_nnx else want

  # *_concurrent_gb should be set for large models, the default is 96.
  max_logging.log(f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}")
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=checkpoint_storage_concurrent_gb,
          save_concurrent_gb=checkpoint_storage_concurrent_gb,
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
      )
  )

  # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
  # Rather than pass the entire abstract state, which could unnecessarily restore opt_state and such and waste
  # memory, we instead specify here that we are just restoring the params field of the checkpoint
  # (which itself may be a dictionary containing a key named 'params').
  restore_args = ocp.checkpoint_utils.construct_restore_args(params_collection)
  restored = ckptr.restore(
      epath.Path(load_parameters_from_path),
      item={"params": params_collection},
      transforms={},
      restore_args={"params": restore_args},
  )
  restored_collection = restored["params"]
  # `transforms={}` lets Orbax return an unmaterialized leaf for a weight the checkpoint lacks,
  # and a stored array at its own shape rather than the target's. Either reaches the model and
  # fails much later without naming the weight, so check here -- the params-only load
  # (load_parameters_path, e.g. SFT) has no init state to fall back on.
  _raise_on_weight_mismatch(want, restored_collection["params"] if is_nnx else restored_collection)
  if is_nnx:
    nnx.replace_by_pure_dict(abstract_unboxed_params, restored_collection["params"])
    return abstract_unboxed_params
  return restored_collection


def save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  print(f"Saving quantized params checkpoint with use_ocdbt = {use_ocdbt} and use_zarr3 = {use_zarr3}")
  orbax_checkpointer = ocp.PyTreeCheckpointer(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
  orbax_checkpointer.save(checkpoint_dir, {"params": params}, force=True)
  print(f"Quantized params checkpoint saved at: {checkpoint_dir}")


def load_checkpoint_metadata(checkpoint_dir_path: str) -> dict[str, Any]:
  """Loads custom metadata from an Orbax checkpoint.

  Args:
    checkpoint_dir_path: Path to the checkpoint directory.

  Returns:
    A dictionary containing custom metadata, or an empty dictionary if none is
    present or loading fails.
  """
  checkpoint_dir = epath.Path(checkpoint_dir_path)
  try:
    ckptr = ocp.StandardCheckpointer()
    metadata = ckptr.metadata(checkpoint_dir)
    return metadata.custom_metadata or {}
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Warning: Failed to load checkpoint metadata: {e}")
    return {}


def _uses_local_checkpoint_period(config):
  return config.enable_emergency_checkpoint or config.enable_multi_tier_checkpointing


def _should_save_checkpoint_at_step(checkpoint_manager, step, config, force):
  """Returns whether MaxText should build and dispatch checkpoint args."""
  if force:
    return True
  if step == 0 and not config.save_checkpoint_on_start:
    # if step = 0, `step % config.checkpoint_period == 0` is always true, force skip
    return False
  if config.enable_continuous_checkpointing:
    base_checkpoint_due = bool(checkpoint_manager.should_save(step))
  else:
    base_checkpoint_due = step % config.checkpoint_period == 0
  local_checkpoint_due = _uses_local_checkpoint_period(config) and step % config.local_checkpoint_period == 0
  autocheckpoint_due = config.enable_autocheckpoint and checkpoint_manager.reached_preemption(step)
  return base_checkpoint_due or local_checkpoint_due or autocheckpoint_due


def _handle_post_checkpoint_preemption(checkpoint_manager, step, force_ckpt_save):
  """Waits on final/preemption saves and raises if preempted."""
  reached_preemption = checkpoint_manager.reached_preemption(step)
  if force_ckpt_save or reached_preemption:
    checkpoint_manager.wait_until_finished()
  if reached_preemption:
    raise exceptions.StopTraining("Job is preempted.")


def maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step=None):
  """Save checkpoint if checkpointing is enabled."""
  if checkpoint_manager is None:
    return

  # Determine the effective step for saving a checkpoint.
  # If 'step' is not provided, this call is for a potential final checkpoint
  # and use the last completed step from the state.
  if step is not None:
    actual_step = int(step)
  else:
    if config.pure_nnx:
      # Under DiLoCo the step lives on the DiLoCoTrainState; otherwise on the optimizer.
      actual_step = int(state.step if config.enable_diloco else state.optimizer.step) - 1
    else:
      # Linen TrainState has .step attribute
      actual_step = int(state.step) - 1

  # Determine if a checkpoint save should be forced, overriding the usual
  # `config.checkpoint_period` logic.
  # This occurs if this function was called:
  # without an explicit 'step' (implying it's a checkpoint save for final step),
  # AND the 'actual_step' is a valid step,
  # AND it's not a step that would normally trigger a checkpoint save.
  force_ckpt_save = step is None and actual_step != -1 and (actual_step % config.checkpoint_period != 0)

  if not _should_save_checkpoint_at_step(checkpoint_manager, actual_step, config, force_ckpt_save):
    _handle_post_checkpoint_preemption(checkpoint_manager, actual_step, force_ckpt_save)
    return

  if checkpoint_manager.latest_step() == actual_step:
    max_logging.log(f"Checkpoint for step {actual_step} already exists, skipping save.")
    return

  if config.pure_nnx:
    # Save in the Linen on-disk layout so pure_nnx and Linen checkpoints are interchangeable.
    if config.enable_diloco:
      # DiLoCoTrainState: persist the synchronized global model (outer params).
      # The per-replica inner optimizer / outer-momentum state is not checkpointed.
      step_value = state.step.get_value() if hasattr(state.step, "get_value") else state.step
      state = train_state_nnx.to_linen_checkpoint_dict({"model": state.params, "optimizer": {"step": step_value}})
    else:
      # rngs/dropout/batch-stats are packed under items/nnx_aux so the RNG/dropout
      # stream continues across resumes instead of resetting to a base key.
      state = train_state_nnx.to_checkpoint_dict(state)

  try:
    checkpoint_saved = save_checkpoint(checkpoint_manager, actual_step, state, config, data_iterator, force_ckpt_save)
    if checkpoint_saved:
      print_save_message(actual_step, config.async_checkpointing)
      if config.elastic_enabled:
        elastic_utils.maybe_elastic_scale_up(config, checkpoint_manager)
  except elastic_utils.manager.ScaleUpSignalError as e:
    if config.elastic_enabled:
      max_logging.log(f"Elastic event detected, letting exception bubble up: {e}")
      raise
    else:
      raise exceptions.StopTraining("Job is preempted.") from e
  except jax.errors.JaxRuntimeError as e:
    if config.elastic_enabled:
      max_logging.log(f"Elastic event detected, letting exception bubble up: {e}")
      raise
    else:
      raise exceptions.StopTraining("Job is preempted.") from e
  except Exception as e:
    raise exceptions.StopTraining(f"Checkpointing failed. {str(e)}") from e

  # Wait for any pending checkpoint save to finish during preemption or final
  # step save, then raise upon preemption.
  _handle_post_checkpoint_preemption(checkpoint_manager, actual_step, force_ckpt_save)


def save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False):
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (
        force
        or (step % config.checkpoint_period == 0 and not config.enable_continuous_checkpointing)
        or (_uses_local_checkpoint_period(config) and step % config.local_checkpoint_period == 0)
        or (config.enable_autocheckpoint and checkpoint_manager.reached_preemption(step))
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )

  # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
  chunk_byte_size = (
      config.checkpoint_storage_target_data_file_size_bytes if config else DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  )

  checkpoint_args = ocp.args.PyTreeSave(
      item=state,
      save_args=jax.tree.map(lambda _: ocp.SaveArgs(chunk_byte_size=chunk_byte_size), state),
      ocdbt_target_data_file_size=chunk_byte_size,
  )
  save_args_composite = {"items": checkpoint_args}

  if config and config.dataset_type == "grain" and not isinstance(data_iterator, PlaceHolderDataIterator):
    if isinstance(data_iterator, RemoteIteratorWrapper):
      # Pass the wrapper directly; GrainCheckpointHandler will call save_state with the step
      save_args_composite["iter"] = GrainCheckpointSave(item=data_iterator)  # pyrefly: ignore[bad-assignment]
    elif not isinstance(data_iterator, list) and isinstance(
        data_iterator.local_iterator, ElasticIterator
    ):  # pyrefly: ignore[missing-attribute]
      # ElasticIterator checkpoints a single global scalar shared by all shards.
      save_args_composite["iter"] = GrainCheckpointSave(
          item=data_iterator.local_iterator
      )  # pyrefly: ignore[bad-assignment]
    else:
      if not isinstance(data_iterator, list):
        data_iterator = [data_iterator]
      grain_iters_to_save = []
      process_count_total = jax.process_count() * len(data_iterator)
      if config.expansion_factor_real_data > 1:
        process_count_total = process_count_total // config.expansion_factor_real_data
      for i, data_iter in enumerate(data_iterator):
        process_index = jax.process_index() + i * jax.process_count()
        grain_iters_to_save.append(
            (data_iter.local_iterator, process_index, process_count_total)
        )  # pyrefly: ignore[missing-attribute]
      save_args_composite["iter"] = GrainCheckpointSave(item=grain_iters_to_save)  # pyrefly: ignore[bad-assignment]

  custom_metadata = {}
  if config:
    if hasattr(config, "scan_layers"):
      custom_metadata["scan_layers"] = config.scan_layers
    if hasattr(config, "lora") and config.lora and getattr(config.lora, "lora_rank", 0) > 0:
      custom_metadata["lora"] = config.lora.model_dump()

  match (checkpoint_manager, config, data_iterator):
    case (checkpoint_manager, _, _) if isinstance(
        checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)
    ):
      replicator_error_handler(config)
      return checkpoint_manager.save(step, args=Composite(state=checkpoint_args), force=force)
    case _:
      return checkpoint_manager.save(
          step, args=Composite(**save_args_composite), force=force, custom_metadata=custom_metadata
      )
