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


"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

import contextlib
import importlib
import os
import time
from typing import Any

from etils import epath
from flax import nnx
from flax import struct


from flax.training import train_state
import jax
from jax.experimental import multihost_utils
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.common import checkpoint_context
from maxtext.common import emergency_checkpointing
from maxtext.common import grain_utility
from maxtext.common import train_state_nnx
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.trainers.diloco.utils import spmd_diloco_checkpointing as diloco_checkpoint_utils
from maxtext.utils import elastic_utils
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import max_logging
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.arrays import sharding as sharding_utils

load_safetensors_dynamic_state = load_dynamic.load_safetensors_dynamic_state
PlaceHolderDataIterator = synthetic_data_processing.PlaceHolderDataIterator
MultiHostDataLoadIterator = multihost_dataloading.MultiHostDataLoadIterator
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = maxtext_globals.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE

# Backward compatibility aliases for v0 emergency managers.
EmergencyCheckpointManager = emergency_checkpointing.CheckpointManager
EmergencyReplicatorCheckpointManager = emergency_checkpointing.ReplicatorCheckpointManager
create_orbax_emergency_checkpoint_manager = emergency_checkpointing.create_emergency_checkpoint_manager
create_orbax_emergency_replicator_checkpoint_manager = emergency_checkpointing.create_replicator_checkpoint_manager

# Union of v1 Checkpointer / the emergency factories return; used in type hints.
CheckpointManager = ocp.training.Checkpointer | EmergencyCheckpointManager | EmergencyReplicatorCheckpointManager


def _weight_mismatches(want, have, path=(), check_missing: bool = True, is_quantized_param: bool = False):
  """Returns `(path, problem)` for each weight in `want` that `have` didn't restore.

  If check_missing is True, this reports absent weights (post-load check). If check_missing is
  False, missing weights are ignored and only shapes of matching keys are checked (the pre-load
  check against checkpoint metadata, whose leaves carry shapes but no values). LoRA adapter
  weights, rng streams, and quantized-param subtrees are allowed to be absent from the
  checkpoint. Only shapes and structure can disagree: Orbax casts restored array dtypes.
  """
  if isinstance(want, dict):
    out = []
    is_quant = is_quantized_param or any(k in want for k in ("qvalue", "qarray"))
    for k, v in want.items():
      nested = have.get(k) if isinstance(have, dict) else None
      if check_missing or nested is not None:
        out.extend(_weight_mismatches(v, nested, path + (k,), check_missing=check_missing, is_quantized_param=is_quant))
    return out
  name = "/".join(str(p) for p in path)
  is_missing = have is None or (check_missing and isinstance(have, jax.ShapeDtypeStruct))
  if is_missing and ("lora_a" in name or "lora_b" in name or "rngs" in path or "rng" in path):
    return []
  if is_missing and is_quantized_param:
    return []
  if is_missing:
    return (
        [(name, f"missing (model expects {getattr(want, 'shape', '?')} {getattr(want, 'dtype', '?')})")]
        if check_missing
        else []
    )
  want_shape, got_shape = getattr(want, "shape", None), getattr(have, "shape", None)
  if want_shape is not None and got_shape is None:
    return [(name, f"structural mismatch: model expects a tensor but checkpoint provides {type(have).__name__}")]
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


def _raise_weight_problems(problems):
  """Raises a ValueError naming each mismatched weight; returns if there are none."""
  # Ignore the weight mismatches in the custom projector so it can stay randomly initialized
  if not problems:
    return
  lines = "\n".join(f"  - '{p}': {why}" for p, why in problems)
  raise ValueError(
      "Checkpoint does not match the model:\n"
      f"{lines}\n"
      "Verify the checkpoint matches the model architecture (emb_dim, mlp_dim, num layers, scan_layers)."
  )


def _is_custom_projector_problem(path: str, want: dict) -> bool:
  """Returns True if a weight mismatch belongs to a newly attached custom vision projector."""
  parts = [p for p in path.replace(".", "/").split("/") if p and p != "params"]
  if len(parts) >= 2 and parts[0] == "vision_encoder":
    proj_name = parts[1]
    proj_dict = want.get("vision_encoder", {}).get(proj_name, {})
    if isinstance(proj_dict, dict) and any("custom_linear" in k for k in proj_dict.keys()):
      max_logging.warning(
          f"===Warning: weight mismatch found in custom vision projector: {proj_name}.\n"
          f"Path: {path}\n"
          "This custom vision projector will be initialized with random weights.==="
      )
      return True
  return False


def _raise_on_weight_mismatch(want, have, config=None):
  """Raises if the restored weights (`have`) don't match what the model expects (`want`).

  Both are pure dicts, so this works for any structure. `partial_restore` returns a weight the
  checkpoint doesn't carry as an unmaterialized ShapeDtypeStruct, and Orbax restores a stored
  array at its own shape rather than the target's. Either way it reaches the model as an
  untrained init value (a silent accuracy loss) or fails much later, deep in the first step,
  without naming the weight.
  """
  if config and getattr(getattr(config, "lora", None), "enable_lora", False):
    want = _filter_lora_trainable_state(want)

  problems = _weight_mismatches(want, have)
  problems = [(p, why) for p, why in problems if not _is_custom_projector_problem(p, want)]
  _raise_weight_problems(problems)


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
    enable_single_replica_ckpt_restoring: bool = False,
    config=None,
):
  """Restores a Linen-layout checkpoint into an NNX state (pure_nnx resume).

  Restores a Linen-shape target that includes `nnx_aux`, then reshapes back via
  `_restored_linen_to_nnx`. rngs/dropout/batch stats come from `items/nnx_aux` when
  present, else keep their fresh init value. A genuinely-missing weight raises.
  """
  max_logging.log(f"Restoring Linen-layout checkpoint into NNX state at {path}")
  if config and getattr(config, "enable_diloco", False):
    return diloco_checkpoint_utils.restore_diloco_checkpoint(
        path,
        abstract_nnx_state,
        checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        config=config,
    )

  linen_abstract = train_state_nnx.to_checkpoint_dict(abstract_nnx_state)
  if config and getattr(getattr(config, "lora", None), "enable_lora", False):
    linen_abstract = _filter_lora_trainable_state(linen_abstract)
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True,
      enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
  )
  # Orbax v1 refuses to read an item subdirectory directly (the step root carries the
  # checkpoint indicator); normalize the documented ".../<step>/items" form to its root
  # and load the checkpointable by name. A v0-written flat pytree dir has no "items"
  # child and is read directly.
  root = epath.Path(_normalize_checkpoint_root(path))
  with context:
    checkpointable_name = "items" if (root / "items").exists() else None
    restored = ocp.load(
        root, linen_abstract, checkpointable_name=checkpointable_name
    )  # pyrefly: ignore[bad-argument-type]
  return _restored_linen_to_nnx(restored, abstract_nnx_state, config=config)


def _restored_linen_to_nnx(restored_linen, abstract_nnx_state, config=None):
  """Reshapes a restored Linen-layout tree into the NNX state.

  Raises if the checkpoint is missing a weight. Every NNX restore path ends here: the load
  itself is the Linen one, since pure_nnx reads and writes the Linen on-disk layout.
  """
  _raise_on_weight_mismatch(*_expected_and_restored_params(abstract_nnx_state, restored_linen), config=config)
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
    checkpoint_conversion_fn,
    source_checkpoint_layout,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
    enable_single_replica_ckpt_restoring: bool = False,
    maxtext_config=None,
):
  """Load full state from checkpoint at specified path.

  Args:
    path: path to checkpoint
    abstract_unboxed_pre_state: an abstract state that Orbax matches type
      against.
    checkpoint_conversion_fn: user-provided function to convert checkpoint to
      maxtext-supported state.
    source_checkpoint_layout: String representation of the checkpoint layout of
      the source checkpoint.
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    use_ocdbt: Whether to use OCDBT format.
    use_zarr3: Whether to use Zarr3 format.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoint
      with load-and-broadcast (single replica). Supported for Orbax format only.
    maxtext_config: Optional configuration dictionary/object.

  Returns:
    The loaded state.
  """
  if source_checkpoint_layout == "orbax":
    # pure_nnx checkpoints are stored in the Linen on-disk layout; reshape to NNX.
    if isinstance(abstract_unboxed_pre_state, nnx.State):
      return _load_linen_checkpoint_into_nnx(
          path,
          abstract_unboxed_pre_state,
          checkpoint_storage_concurrent_gb,
          use_ocdbt,
          use_zarr3,
          enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
          config=maxtext_config,
      )
    context = checkpoint_context.build_context(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.ORBAX,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )
    with context:
      return ocp.load(path, abstract_unboxed_pre_state)

  if source_checkpoint_layout == "safetensors":
    if enable_single_replica_ckpt_restoring:
      max_logging.warning("enable_single_replica_ckpt_restoring is not supported for safetensors layout.")
    # Resolved first, so a bad config fails before the weights are read.
    conversion_fn = _resolve_conversion_fn(checkpoint_conversion_fn)
    context = checkpoint_context.build_context(
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS,
    )
    with context:
      metadata = ocp.metadata(path)
      simple_abstract_state = metadata.metadata
      shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

      def combine_sharding(sds, shardings):
        return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings)

      sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
      pre_transformed_state = ocp.load(path, sharded_abstract_state)
    state = conversion_fn(pre_transformed_state)
    # The conversion fn returns MaxText's on-disk (Linen) layout, which is what pure_nnx reads,
    # so NNX needs the same reshape as every other restore. An NNX state passes through.
    if isinstance(abstract_unboxed_pre_state, nnx.State) and not isinstance(state, nnx.State):
      state = _restored_linen_to_nnx(state, abstract_unboxed_pre_state, config=maxtext_config)
    return state

  raise ocp.errors.InvalidLayoutError(f"Unknown checkpoint layout: {source_checkpoint_layout}")


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
    enable_autocheckpoint: bool = False,
    todelete_subdir: str | None = None,
    todelete_full_path: str | None = None,
    ocdbt_target_data_file_size_bytes: int | None = None,
):
  """Returns an Orbax v1 training ``Checkpointer``, or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None

  # TODO: b/529622681 - Remove deprecated settings.
  if orbax_logger is not None:
    max_logging.warning(
        "Cloud logging (enable_checkpoint_cloud_logger) is disabled because"
        " Orbax v1 now configures its own logger internally. This config"
        " setting is ignored and will be removed."
    )
  if dataset_type is not None:
    max_logging.warning(
        "Specifying dataset_type upon checkpointer creation is deprecated and"
        " will be removed soon, this is now handled dynamically by Orbax"
        " Checkpointer."
    )

  max_logging.log(f"Creating checkpointer with ocdbt={use_ocdbt} and zarr3={use_zarr3}")

  validated_path = gcs_utils.mkdir_and_check_permissions(checkpoint_dir)

  if ocdbt_target_data_file_size_bytes is None:
    ocdbt_target_data_file_size_bytes = DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE

  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      ocdbt_target_data_file_size_bytes=ocdbt_target_data_file_size_bytes,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      enable_continuous_checkpointing=enable_continuous_checkpointing,
      todelete_full_path=todelete_full_path,
      todelete_subdir=todelete_subdir,
      partial_load=True,
  )

  manager = ocp.training.Checkpointer(
      validated_path,
      context=context,
      save_decision_policy=checkpoint_context.build_save_decision_policy(
          save_interval_steps=save_interval_steps,
          enable_continuous_checkpointing=enable_continuous_checkpointing,
          enable_autocheckpoint=enable_autocheckpoint,
      ),
      preservation_policy=checkpoint_context.build_preservation_policy(
          max_to_keep=max_num_checkpoints_to_keep,
      ),
  )
  # Necessary bridge to support v0 backward compatibility.
  manager.use_async = use_async  # pyrefly: ignore[missing-attribute]

  max_logging.log("Checkpoint manager created!")
  return manager


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


def latest_step(checkpoint_manager):
  """Latest saved step or None, across the v0 emergency manager and the v1 Checkpointer."""
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return checkpoint_manager.latest_step()
  else:
    latest = checkpoint_manager.latest
    return latest.step if latest is not None else None


def all_steps(checkpoint_manager):
  """All saved steps, across the v0 emergency manager and the v1 Checkpointer."""
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return checkpoint_manager.all_steps()
  return [checkpoint.step for checkpoint in checkpoint_manager.checkpoints]


def wait_until_finished(checkpoint_manager):
  """Blocks until pending saves finish, across the v0 emergency manager and the v1 Checkpointer."""
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    checkpoint_manager.wait_until_finished()
  else:
    checkpoint_manager.wait()


def reached_preemption(checkpoint_manager, step: int) -> bool:
  """Whether a preemption sync point has been reached at ``step`` (manager-agnostic)."""
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return checkpoint_manager.reached_preemption(step)
  else:
    return multihost_utils.reached_preemption_sync_point(step)


def _normalize_checkpoint_root(path_str):
  """Lifts a v0-convention pytree path (".../<step>/items") to its checkpoint root."""
  path_str = str(path_str).rstrip("/")
  if path_str == "items":
    return "."
  return path_str.removesuffix("/items")


def load_state_if_possible(
    checkpoint_manager: CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    checkpoint_storage_concurrent_gb: int,
    abstract_unboxed_pre_state: train_state.TrainState | nnx.State,
    enable_single_replica_ckpt_restoring: bool | None = False,
    dataset_type: str | None = "synthetic",
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
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoint
      with load-and-broadcast (single replica). Supported for Orbax format only.
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

  # TODO: b/529622681 - Remove deprecated settings.
  if enable_orbax_v1:
    max_logging.warning(
        "enable_orbax_v1 is deprecated and will be removed, as Orbax v1 is now the default checkpointing API."
    )

  if load_parameters_from_path:
    load_parameters_from_path = _normalize_checkpoint_root(load_parameters_from_path)
  if load_full_state_from_path:
    load_full_state_from_path = _normalize_checkpoint_root(load_full_state_from_path)
  # pure_nnx saves in the Linen on-disk layout, so every branch below loads the same tree Linen
  # does: the NNX abstract is converted to that layout going in, and what comes back is reshaped
  # into the NNX state on the way out.
  is_nnx = isinstance(abstract_unboxed_pre_state, (nnx.State, train_state_nnx.TrainStateNNX))

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    step = latest_step(checkpoint_manager) if step < 0 else step  # pyrefly: ignore[bad-assignment]
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      is_diloco = bool(maxtext_config and getattr(maxtext_config, "enable_diloco", False))

      # Map the expected training state to the on-disk checkpoint dictionary layout:
      # - DiLoCo: DiLoCoTrainState (wrapping NNX or Linen inner state + outer params + opt state).
      # - Standard non-DiLoCo NNX: TrainStateNNX (converted to Linen collection layout for storage).
      # - Standard non-DiLoCo Linen: TrainState dataclass (used directly).
      if is_diloco:
        restore_target = diloco_checkpoint_utils.to_diloco_checkpoint_dict(
            abstract_unboxed_pre_state, config=maxtext_config
        )
      elif is_nnx:
        restore_target = train_state_nnx.to_checkpoint_dict(abstract_unboxed_pre_state)
      else:
        restore_target = abstract_unboxed_pre_state

      if maxtext_config and getattr(getattr(maxtext_config, "lora", None), "enable_lora", False):
        restore_target = _filter_lora_trainable_state(restore_target)

      # Case 1: emergency / replicator managers restore via their own v0 path.
      if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
        restored = emergency_checkpointing.restore(checkpoint_manager, step, restore_target)
        if is_diloco:
          restored = diloco_checkpoint_utils.from_diloco_checkpoint_dict(
              restored, abstract_unboxed_pre_state, config=maxtext_config
          )
        elif is_nnx:
          restored = _restored_linen_to_nnx(restored, abstract_unboxed_pre_state, config=maxtext_config)
        return (restored, None)

      # Case 2: standard v1 Checkpointer, restoring the grain iterator in place when
      # a "grain" dataset iterator was checkpointed alongside the state.
      assert isinstance(checkpoint_manager, ocp.training.Checkpointer)
      abstract_checkpointables = {"items": restore_target}
      if (
          dataset_type == "grain"
          and data_iterator
          and not isinstance(data_iterator, PlaceHolderDataIterator)
          and (checkpoint_manager.directory / str(step) / "iter").exists()
      ):
        abstract_checkpointables["iter"] = grain_utility.for_restore(
            checkpoint_manager, step, data_iterator, expansion_factor_real_data
        )
      restored = checkpoint_manager.load_checkpointables(step, abstract_checkpointables)
      if is_diloco:
        restored_items = diloco_checkpoint_utils.from_diloco_checkpoint_dict(
            restored["items"], abstract_unboxed_pre_state, config=maxtext_config
        )
        restored = {"items": restored_items}
      elif is_nnx:
        restored_items = _restored_linen_to_nnx(restored["items"], abstract_unboxed_pre_state, config=maxtext_config)
        restored = {"items": restored_items}
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
        enable_single_replica_ckpt_restoring=bool(enable_single_replica_ckpt_restoring),
    )
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"Loading full state from path: {load_full_state_from_path}")
    restored_state = _load_full_state_from_path(
        path=load_full_state_from_path,
        abstract_unboxed_pre_state=abstract_unboxed_pre_state,
        checkpoint_conversion_fn=checkpoint_conversion_fn,
        source_checkpoint_layout=source_checkpoint_layout,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        enable_single_replica_ckpt_restoring=bool(enable_single_replica_ckpt_restoring),
        maxtext_config=maxtext_config,
    )
    return {"items": restored_state}, None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> Any | None:  # pytype: disable=attribute-error
  """DEPRECATED: Setup checkpoint logger."""
  # TODO: b/529622681 - Remove this config option entirely.
  if config.enable_checkpoint_cloud_logger:
    max_logging.warning(
        "Cloud logging (enable_checkpoint_cloud_logger) is disabled because"
        " Orbax v1 now configures its own logger internally. This config"
        " setting is ignored and will be removed."
    )


def load_params_from_path(
    load_parameters_from_path,
    abstract_unboxed_params,
    checkpoint_storage_concurrent_gb,
    use_ocdbt=True,
    use_zarr3=True,
    enable_single_replica_ckpt_restoring: bool = False,
):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")

  # Orbax v1 refuses to read an item subdirectory directly; normalize the documented
  # ".../<step>/items" form to its checkpoint root and load it by name below.
  path = epath.Path(_normalize_checkpoint_root(load_parameters_from_path))

  # On disk the weights live at `params/params/...`: an outer key naming the item, and Flax's
  # `params` collection inside it. A Linen TrainState.params is that collection; an NNX params
  # state sits one level below it (bare weights), so wrap it going in and unwrap it coming out.
  is_nnx = isinstance(abstract_unboxed_params, nnx.State)
  want = abstract_unboxed_params.to_pure_dict() if is_nnx else abstract_unboxed_params

  # Determine the restore key based on the leaf directory name to support native and custom SFT
  restore_key = os.path.basename(str(load_parameters_from_path).rstrip("/"))
  if restore_key not in ("model_params", "model"):
    restore_key = "params"

  if restore_key in ("model_params", "model"):
    params_collection = want
  else:
    params_collection = {"params": want} if is_nnx else want

  # Memory optimization: restore only the "params" key (the checkpoint may also hold opt_state/step);
  # partial_load drops the rest. The abstract carries shape/dtype/sharding directly.
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True,
      enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
  )
  # Dispatch on the on-disk layout instead of assuming a step root: callers pass step roots,
  # v0-style pytree dirs (normalized above), and v0 flat params-only checkpoints
  # (save_params_to_path wrote the pytree directly at the directory).
  with context:
    checkpointable_name = "items" if (path / "items").exists() else None
    # Orbax v1 fails a mid-load shape mismatch itself, with an error that reports the
    # shapes but not which weight; compare the stored metadata first so the error names
    # it. A metadata read failure falls through to the load (worst case: Orbax's error).
    try:
      stored = ocp.metadata(path, checkpointable_name=checkpointable_name).metadata
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"Skipping pre-load shape check, checkpoint metadata unreadable: {e}")
      stored = None
    if isinstance(stored, dict):
      stored_collection = stored.get(restore_key)
      if restore_key == "params" and is_nnx and isinstance(stored_collection, dict):
        stored_collection = stored_collection.get("params")
      _raise_weight_problems(_weight_mismatches(want, stored_collection, check_missing=False))
    restored = ocp.load(
        path,
        {restore_key: params_collection},
        checkpointable_name=checkpointable_name,  # pyrefly: ignore[bad-argument-type]
    )
  restored_collection = restored[restore_key]  # pyrefly: ignore[bad-index]
  # partial_load lets Orbax return an unmaterialized leaf for a weight the checkpoint lacks,
  # and a stored array at its own shape rather than the target's. Either reaches the model and
  # fails much later without naming the weight, so check here -- the params-only load
  # (load_parameters_path, e.g. SFT) has no init state to fall back on.

  if restore_key in ("model_params", "model"):
    restored_weights = restored_collection
  else:
    restored_weights = restored_collection["params"] if is_nnx else restored_collection

  _raise_on_weight_mismatch(want, restored_weights)
  if is_nnx:
    nnx.replace_by_pure_dict(abstract_unboxed_params, restored_weights)
    return abstract_unboxed_params
  return restored_collection


def save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  max_logging.log(f"Saving params checkpoint with use_ocdbt={use_ocdbt} and" f" use_zarr3={use_zarr3}")
  context = checkpoint_context.build_context(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
  with context:
    ocp.save(
        checkpoint_dir,
        {"params": params},
        checkpointable_name="items",
        overwrite=True,  # pyrefly: ignore[bad-argument-type]
    )
  max_logging.log(f"Params checkpoint saved at: {checkpoint_dir}")


def load_checkpoint_metadata(checkpoint_dir_path: str) -> Any:
  """Loads custom metadata from an Orbax checkpoint.

  Args:
    checkpoint_dir_path: Path to the checkpoint directory.

  Returns:
    A dictionary containing custom metadata, or an empty dictionary if none is
    present or loading fails.
  """
  checkpoint_dir = epath.Path(_normalize_checkpoint_root(checkpoint_dir_path))
  try:
    metadata = ocp.checkpointables_metadata(checkpoint_dir)
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
  autocheckpoint_due = config.enable_autocheckpoint and reached_preemption(checkpoint_manager, step)
  return base_checkpoint_due or local_checkpoint_due or autocheckpoint_due


def _handle_post_checkpoint_preemption(checkpoint_manager, step, force_ckpt_save):
  """Waits on final/preemption saves and raises if preempted."""
  # Named is_preempted (not reached_preemption) so it doesn't shadow the module-level
  # reached_preemption dispatcher we call below.
  is_preempted = reached_preemption(checkpoint_manager, step)
  if force_ckpt_save or is_preempted:
    wait_until_finished(checkpoint_manager)
  if is_preempted:
    raise exceptions.StopTraining("Job received termination signal (SIGTERM).")


@contextlib.contextmanager
def checkpoint_exception_guard(config, checkpoint_manager, handler_fn=None):
  """Context manager that wraps checkpointing save Exception handling.

  On block success (checkpoint written without errors): runs the scale-up check
  if elastic training is active.
  On block failure: bubbles up JAX/ScaleUp errors if elastic training is active;
  otherwise delegates to `handler_fn`.

  Args:
    config: maxtext configuration object.
    checkpoint_manager: The CheckpointManager instance.
    handler_fn: Optional callback function(Exception) that handles/wraps
      non-elastic exceptions. If this handler raises a new exception, that new
      exception is propagated. If it returns normally (returns None) or is not
      provided, the original exception is re-raised (preserving its traceback).
  """
  try:
    yield
    elastic_utils.maybe_elastic_scale_up(config, checkpoint_manager)
  except Exception as e:  # pylint: disable=broad-except
    elastic_utils.maybe_bubble_elastic_exception(config, e)
    if handler_fn:
      handler_fn(e)
    else:
      raise


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

  # Skip if step directory already exists (e.g. step 0 or prior checkpoints in all_steps())
  # to prevent Orbax OCDBT UUID collisions during auto-resume / continuation runs for DiLoCo.
  if latest_step(checkpoint_manager) == actual_step or actual_step in all_steps(checkpoint_manager):
    max_logging.log(f"Checkpoint for step {actual_step} already exists, skipping save.")
    return

  def _checkpoint_error_handler(err):
    """Handles checkpointing errors."""
    raise RuntimeError(f"Checkpointing failed. {str(err)}") from err

  with checkpoint_exception_guard(config, checkpoint_manager, _checkpoint_error_handler):
    checkpoint_saved = save_checkpoint(
        checkpoint_manager,
        actual_step,
        state,
        config,
        data_iterator,
        force_ckpt_save,
    )
  if checkpoint_saved:
    print_save_message(actual_step, config.async_checkpointing)

  # Wait for any pending checkpoint save to finish during preemption or final
  # step save, then raise upon preemption.
  _handle_post_checkpoint_preemption(checkpoint_manager, actual_step, force_ckpt_save)


def _filter_lora_trainable_state(state):
  """Filters state representation to keep only LoRA weights and opt_state using Flax NNX filter."""

  def _lora_filter(path, val):
    path_str = "/".join(str(p) for p in path).lower()
    return isinstance(val, nnx.LoRAParam) or "lora" in path_str or "step" in path_str

  if isinstance(state, (nnx.State, nnx.Module)):
    return nnx.state(state, _lora_filter)

  def _filter_dict(val, path=()):
    if isinstance(val, dict):
      res = {}
      for k, v in val.items():
        curr_path = path + (str(k),)
        filtered = _filter_dict(v, curr_path)
        if filtered is not None:
          res[k] = filtered
      return res if res else None

    path_str = "/".join(path).lower()
    if "lora" in path_str or "step" in path_str:
      return val
    return None

  return _filter_dict(state)


def save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False):
  """Wrapper for saving checkpoint."""
  # Allow struct.PyTreeNode so Flax dataclass states (e.g. DiLoCoTrainState) aren't cleared to empty dicts ({})
  if not isinstance(state, (dict, nnx.State, train_state.TrainState, struct.PyTreeNode)):
    if isinstance(state, train_state_nnx.TrainStateNNX):
      state = nnx.state(state)
    elif not isinstance(state, (dict, nnx.State)):
      state = {}

  if config and getattr(config, "enable_diloco", False):
    state = diloco_checkpoint_utils.to_diloco_checkpoint_dict(state, config)
  elif config and getattr(config, "pure_nnx", False):
    # Save in the Linen on-disk layout so pure_nnx and Linen checkpoints are interchangeable.
    if isinstance(state, nnx.State):
      state = train_state_nnx.to_checkpoint_dict(state)

  if config and getattr(config, "enable_checkpointing", False):
    if (
        force
        or (step % config.checkpoint_period == 0 and not getattr(config, "enable_continuous_checkpointing", False))
        or (_uses_local_checkpoint_period(config) and step % config.local_checkpoint_period == 0)
        or (getattr(config, "enable_autocheckpoint", False) and reached_preemption(checkpoint_manager, step))
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

  # LoRA training persists only the adapter weights (plus step/opt_state).
  if config and getattr(getattr(config, "lora", None), "enable_lora", False):
    filtered = _filter_lora_trainable_state(state)
    if filtered:
      state = filtered

  # Emergency / replicator managers keep the v0 save path.
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return emergency_checkpointing.save(checkpoint_manager, step, state, config, force)

  # Record config properties needed to validate compatibility at load time
  # (e.g. proactive scan_layers verification, LoRA restore).
  custom_metadata = {}
  if config:
    if hasattr(config, "scan_layers"):
      custom_metadata["scan_layers"] = config.scan_layers
    if hasattr(config, "lora") and config.lora and getattr(config.lora, "lora_rank", 0) > 0:
      custom_metadata["lora"] = config.lora.model_dump()

  # Standard path: Orbax v1 Checkpointer. Storage/chunk options live on the manager's Context.
  checkpointables = {"items": state}
  if (
      config
      and getattr(config, "dataset_type", None) == "grain"
      and not isinstance(data_iterator, PlaceHolderDataIterator)
  ):
    checkpointables["iter"] = grain_utility.for_save(step, data_iterator, config.expansion_factor_real_data)
  # The v1 Checkpointer raises for an already-existing step BEFORE consulting the
  # save decision policy; v0 silently skipped such saves (should_save ran first).
  # Preserve v0 semantics: e.g. resuming from a non-latest step into a directory
  # that still holds later/off-interval checkpoints must not kill training.
  try:
    if getattr(checkpoint_manager, "use_async", False):
      # Async save returns once the blocking device-to-host copy is done and writes in
      # the background (v0 enable_async_checkpointing parity); a None response means the
      # save decision policy declined. Background errors surface on the next save/wait.
      response = checkpoint_manager.save_checkpointables_async(
          step, checkpointables, force=force, custom_metadata=custom_metadata
      )
      return response is not None
    return checkpoint_manager.save_checkpointables(step, checkpointables, force=force, custom_metadata=custom_metadata)
  except FileExistsError as e:  # ocp.training StepAlreadyExistsError subclasses FileExistsError
    max_logging.log(f"Checkpoint for step {step} already exists, skipping save. ({e})")
    return False
