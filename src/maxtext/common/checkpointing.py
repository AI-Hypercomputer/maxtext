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

import datetime
import importlib
import time
from typing import Any

from etils import epath
from flax import nnx
from flax.training import train_state
from grain.experimental import ElasticIterator
import jax
from maxtext.checkpoint_conversion.utils.load_dynamic import load_safetensors_dynamic_state
from maxtext.common import emergency_checkpointing
from maxtext.common import grain_utility
from maxtext.common import train_state_nnx
from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator
from maxtext.input_pipeline.multihost_dataloading import RemoteIteratorWrapper
from maxtext.input_pipeline.synthetic_data_processing import PlaceHolderDataIterator
from maxtext.utils import elastic_utils
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils.globals import DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
import orbax.checkpoint as ocp
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib


CheckpointManagerOptions = ocp.CheckpointManagerOptions
Composite = ocp.args.Composite
PyTreeCheckpointHandler = ocp.PyTreeCheckpointHandler
# Backward compatibility aliases for v0 emergency managers.
EmergencyCheckpointManager = emergency_checkpointing.CheckpointManager
EmergencyReplicatorCheckpointManager = emergency_checkpointing.ReplicatorCheckpointManager
create_orbax_emergency_checkpoint_manager = emergency_checkpointing.create_emergency_checkpoint_manager
create_orbax_emergency_replicator_checkpoint_manager = emergency_checkpointing.create_replicator_checkpoint_manager

# Union of CheckpointManager / the emergency factories return; used in type hints.
CheckpointManager = ocp.CheckpointManager | EmergencyCheckpointManager | EmergencyReplicatorCheckpointManager


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
  return _restored_linen_to_nnx(restored, abstract_nnx_state)


def _restored_linen_to_nnx(restored_linen, abstract_nnx_state):
  """Reshapes a restored Linen-layout tree into the NNX state.

  Raises if the checkpoint is missing a weight. Every NNX restore path ends here: the load
  itself is the Linen one, since pure_nnx reads and writes the Linen on-disk layout.
  """
  _raise_on_weight_mismatch(*_expected_and_restored_params(abstract_nnx_state, restored_linen))
  return _linen_items_to_nnx(restored_linen, abstract_nnx_state)


def _abstract_params(abstract_unboxed_pre_state):
  """Returns the state's weights: the NNX Param subtree, or Linen's `params` collection."""
  if isinstance(abstract_unboxed_pre_state, nnx.State):
    return nnx.split_state(abstract_unboxed_pre_state.model, nnx.Param, ...)[0]
  return abstract_unboxed_pre_state.params


def _bare_weights(tree):
  """Strips the Flax `params` collection wrapper so weights compare at the same depth.

  A Linen params tree is the collection, an NNX one the bare weights; the dynamic
  safetensors loader always returns the collection.
  """
  return tree["params"] if isinstance(tree, dict) and len(tree) == 1 and "params" in tree else tree


def _resolve_conversion_fn(checkpoint_conversion_fn):
  """Returns `checkpoint_conversion_fn` as a callable.

  Config carries it as a dotted string ("my_pkg.my_module.my_fn"), so it has to be imported
  before it can be called. A callable is used as is.
  """
  if checkpoint_conversion_fn is None:
    raise ValueError(
        "source_checkpoint_layout='safetensors' needs `checkpoint_conversion_fn` to map the "
        "checkpoint's weights onto the model's, e.g. checkpoint_conversion_fn=my_pkg.my_module.my_fn."
    )
  if callable(checkpoint_conversion_fn):
    return checkpoint_conversion_fn
  module_name, _, fn_name = str(checkpoint_conversion_fn).rpartition(".")
  if not module_name:
    raise ValueError(f"`checkpoint_conversion_fn` must be a dotted path to a function, got {checkpoint_conversion_fn!r}.")
  try:
    fn = getattr(importlib.import_module(module_name), fn_name, None)
  except ImportError as e:
    raise ValueError(f"Could not import `checkpoint_conversion_fn` {checkpoint_conversion_fn!r}: {e}") from e
  if not callable(fn):
    raise ValueError(f"`checkpoint_conversion_fn` {checkpoint_conversion_fn!r} is not a function.")
  return fn


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
      # Resolved first, so a bad config fails before the weights are read.
      conversion_fn = _resolve_conversion_fn(checkpoint_conversion_fn)
      context = ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS)
      with context:
        metadata = ocp_v1.pytree_metadata(path)
        simple_abstract_state = metadata.metadata
        shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

        def combine_sharding(sds, shardings):
          return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings)

        sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
        pre_transformed_state = ocp_v1.load_pytree(path, sharded_abstract_state)
      state = conversion_fn(pre_transformed_state)
      # The conversion fn returns MaxText's on-disk (Linen) layout, which is what pure_nnx reads,
      # so NNX needs the same reshape as every other restore. An NNX state passes through.
      if isinstance(abstract_unboxed_pre_state, nnx.State) and not isinstance(state, nnx.State):
        state = _restored_linen_to_nnx(state, abstract_unboxed_pre_state)
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
    item_handlers["iter"] = grain_utility.GrainCheckpointHandler()  # pyrefly: ignore[bad-assignment]

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
  manager = ocp.CheckpointManager(
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


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


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

  # pure_nnx saves in the Linen on-disk layout, so every branch below loads the same tree Linen
  # does: the NNX abstract is converted to that layout going in, and what comes back is reshaped
  # into the NNX state on the way out.
  is_nnx = isinstance(abstract_unboxed_pre_state, nnx.State)

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
        replica_devices = grain_utility.replica_devices(mesh.devices, replica_axis_index)
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

      restore_target = (
          train_state_nnx.to_checkpoint_dict(abstract_unboxed_pre_state) if is_nnx else abstract_unboxed_pre_state
      )
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
          if is_nnx:
            restored = _restored_linen_to_nnx(restored, abstract_unboxed_pre_state)
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
          restored, iterator = grain_utility.restore_grain_iterator(
              checkpoint_manager,
              step,
              data_iterator,
              checkpoint_args,
              expansion_factor_real_data,
          )
          if is_nnx:
            restored = {"items": _restored_linen_to_nnx(restored["items"], abstract_unboxed_pre_state)}
          return (restored, iterator)
        # Case 3: Default/Fallback case.
        # This case acts as a wildcard ('_') and matches if none of the preceding cases were met.
        case _:
          restored = checkpoint_manager.restore(step, args=Composite(items=checkpoint_args))
          if is_nnx:
            restored = {"items": _restored_linen_to_nnx(restored["items"], abstract_unboxed_pre_state)}
          return (restored, None)

  if source_checkpoint_layout == "safetensors_dynamic":
    path = load_parameters_from_path or load_full_state_from_path
    max_logging.log(f"Dynamic On-the-Fly Formatting: Loading SafeTensors from {path}")

    # Weights-only for both paths, so the loader gets the weights rather than the whole state:
    # the HF param mappings name weights, and an NNX state hides them under `model`.
    params = _abstract_params(abstract_unboxed_pre_state)
    restored, restored_params = load_safetensors_dynamic_state(path, params, maxtext_config)
    # A weight no HF mapping covered comes back unmaterialized and would reach the model as an
    # untrained init value. Same check the Orbax weights-only load makes.
    _raise_on_weight_mismatch(_bare_weights(params.to_pure_dict() if is_nnx else params), _bare_weights(restored_params))
    if is_nnx:
      # The loader returns the Linen `params` collection; NNX holds bare weights, so unwrap it
      # back into the params state, the shape load_params_from_path returns.
      nnx.replace_by_pure_dict(params, restored_params["params"])
      return restored, params
    return restored, restored_params
  elif load_parameters_from_path != "":
    params = _abstract_params(abstract_unboxed_pre_state)
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
      save_args_composite["iter"] = grain_utility.GrainCheckpointSave(
          item=data_iterator
      )  # pyrefly: ignore[bad-assignment]
    elif not isinstance(data_iterator, list) and isinstance(
        data_iterator.local_iterator, ElasticIterator
    ):  # pyrefly: ignore[missing-attribute]
      # ElasticIterator checkpoints a single global scalar shared by all shards.
      save_args_composite["iter"] = grain_utility.GrainCheckpointSave(
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
      save_args_composite["iter"] = grain_utility.GrainCheckpointSave(
          item=grain_iters_to_save
      )  # pyrefly: ignore[bad-assignment]

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
      emergency_checkpointing.replicator_error_handler(config)
      return checkpoint_manager.save(step, args=Composite(state=checkpoint_args), force=force)
    case _:
      return checkpoint_manager.save(
          step, args=Composite(**save_args_composite), force=force, custom_metadata=custom_metadata
      )
