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
from maxtext.utils import lora_utils
from maxtext.utils import max_logging
import jax.numpy as jnp
import numpy as np

from qwix import QArray
from qwix._src.providers.ptq import WithAux

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
import flax

CheckpointManager = ocp.CheckpointManager | EmergencyCheckpointManager | EmergencyReplicatorCheckpointManager


def _tree_to_dict(tree):
  """Recursively converts NNX State or PyTree to pure python dict."""
  if hasattr(tree, "to_pure_dict"):
    return _tree_to_dict(tree.to_pure_dict())
  if isinstance(tree, dict):
    res = {k: _tree_to_dict(v) for k, v in tree.items()}
    if len(res) == 1 and ("value" in res or "raw_value" in res):
      return res.get("value", res.get("raw_value"))
    return res
  if hasattr(tree, "get_value"):
    return tree.get_value()
  if hasattr(tree, "value"):
    return tree.value
  return tree


def _norm_path_key(path):
  """Normalizes a PyTree path key or string path to a canonical string representation."""
  if isinstance(path, str):
    parts = path.split("/")
  elif isinstance(path, (tuple, list)):
    parts = [str(p) for p in path]
  else:
    parts = [str(path)]
  if len(parts) > 1 and parts[-1] in ("value", "raw_value"):
    parts = parts[:-1]
  norm_parts = []
  for p in parts:
    if p in ("scanned_blocks", "layers", "layers_remainder", "model", "params"):
      continue
    if p.isdigit():
      norm_parts.append(f"layers_{p}")
    else:
      norm_parts.append(p)
  return "/".join(norm_parts)


def _is_dict_leaf(x):
  if not isinstance(x, (dict, nnx.State)):
    return True
  if ("qvalue" in x and "scale" in x) or ("array" in x and "how" in x):
    return True
  if "array" in x and isinstance(
      x["array"],
      (dict, nnx.State)) and ("qvalue" in x["array"] and "scale" in x["array"]):
    return True
  return False


def _flatten_and_norm_dict(d):
  """Flattens dict d with flax.traverse_util.flatten_dict and normalizes path keys with _norm_path_key."""
  if d is None:
    return {}
  d = _tree_to_dict(d)
  flat = flax.traverse_util.flatten_dict(
      d,
      is_leaf=lambda *path: _is_dict_leaf(path[-1]),
  )
  return {_norm_path_key(k): v for k, v in flat.items()}


def _weight_mismatches(want, have):
  """Returns `(path, problem)` for each weight in `want` that `have` didn't restore faithfully.

  A weight is wrong if the checkpoint didn't carry it -- absent, or left by Orbax as an
  unmaterialized ShapeDtypeStruct -- or carried it at a different shape. Only the shape can
  disagree: Orbax casts a restored array to the target's dtype.

  For PEFT/LoRA and QLoRA compatibility:
  - Runtime-injected LoRA parameters (`kernel_lora_a`, `kernel_lora_b`) and RNG states are skipped.
  - Dynamically quantized QLoRA weights expecting `qvalue` / `scale` leaves are matched against
    unquantized base parameter keys (e.g. `kernel`).
  """
  if want is None:
    return []
  flat_want = flax.traverse_util.flatten_dict(_tree_to_dict(want))
  norm_have = _flatten_and_norm_dict(have)

  problems = []
  for path, target_val in flat_want.items():
    while isinstance(target_val, nnx.Variable):
      target_val = target_val.get_value() if hasattr(
          target_val, "get_value") else target_val.value

    name = "/".join(str(p) for p in path)
    path_parts = [str(p) for p in path]
    norm_p = _norm_path_key(path)
    restored_val = norm_have.get(norm_p)

    if "lora_a" in name or "lora_b" in name or "rngs" in path_parts or "rng" in path_parts:
      if restored_val is None or isinstance(restored_val, jax.ShapeDtypeStruct):
        continue

    # If quantized QArray leaf is missing, check if unquantized base parameter exists in restored checkpoint
    if (restored_val is None or
        isinstance(restored_val, jax.ShapeDtypeStruct)) and any(
            k in name for k in ("qvalue", "scale", "qarray", "zero_point")):
      base_parts = [
          p for p in path_parts if p not in ("array", "qvalue", "scale",
                                             "qarray", "zero_point", "bits")
      ]
      base_norm_p = _norm_path_key(base_parts)
      if base_norm_p in norm_have and not isinstance(norm_have[base_norm_p],
                                                     jax.ShapeDtypeStruct):
        continue

    if restored_val is None or isinstance(restored_val, jax.ShapeDtypeStruct):
      target_shape = getattr(target_val, "shape", "?")
      target_dtype = getattr(target_val, "dtype", "?")
      problems.append(
          (name, f"missing (model expects {target_shape} {target_dtype})"))
    else:
      want_shape, got_shape = getattr(target_val, "shape",
                                      None), getattr(restored_val, "shape",
                                                     None)
      if want_shape is not None and got_shape is not None and tuple(
          want_shape) != tuple(got_shape):
        problems.append((
            name,
            f"shape {tuple(got_shape)} but the model expects {tuple(want_shape)}"
        ))
  return problems


def _set_nested_leaf(target, path, leaf_val):
  """Sets a nested leaf in target given a path."""
  curr = target
  for key_obj in path[:-1]:
    k = getattr(key_obj, "key",
                str(key_obj)) if not isinstance(key_obj,
                                                (str, int)) else key_obj
    if isinstance(curr, (dict, nnx.State)) or hasattr(curr, "__getitem__"):
      curr = curr[k]
    elif hasattr(curr, str(k)):
      curr = getattr(curr, str(k))
    else:
      return
  last_k = getattr(path[-1], "key", str(
      path[-1])) if not isinstance(path[-1], (str, int)) else path[-1]
  if isinstance(curr, (dict, nnx.State)) or hasattr(curr, "__setitem__"):
    curr[last_k] = leaf_val
  elif hasattr(curr, str(last_k)):
    setattr(curr, str(last_k), leaf_val)


def _rebuild_qwix_types(val):
  """Recursively reconstructs QArray or WithAux objects from dict/State representations restored from Orbax."""
  if hasattr(val, "to_pure_dict"):
    val = val.to_pure_dict()
  if isinstance(val, (dict, nnx.State)):
    d = {k: _rebuild_qwix_types(v) for k, v in val.items()}
    if "qvalue" in d and "scale" in d:
      qval = d["qvalue"]
      scale = d["scale"]
      qval = qval.get_value() if hasattr(qval, "get_value") else getattr(
          qval, "value", qval)
      scale = scale.get_value() if hasattr(scale, "get_value") else getattr(
          scale, "value", scale)
      if hasattr(qval, "dtype") and jnp.issubdtype(qval.dtype, jnp.floating):
        qval = qval.astype(jnp.int8)
      zp = d.get("zero_point")
      if zp is not None:
        zp = zp.get_value() if hasattr(zp, "get_value") else getattr(
            zp, "value", zp)
      qtype = d.get("qtype", "nf4")
      if isinstance(qtype, (jax.Array, np.ndarray)) or hasattr(qtype, "item"):
        qtype = str(qtype.item()) if hasattr(qtype, "item") else str(qtype)
      qarr = QArray(qvalue=qval, scale=scale, zero_point=zp, qtype=qtype)
      return WithAux(array=qarr, how=d.get("how", "ptq"))
    if "array" in d:
      arr = d["array"]
      arr = arr.get_value() if hasattr(arr, "get_value") else getattr(
          arr, "value", arr)
      how = d.get("how", "ptq")
      how = how.get_value() if hasattr(how, "get_value") else getattr(
          how, "value", how)
      return WithAux(array=arr, how=how)
    return d
  if hasattr(val, "get_value"):
    return _rebuild_qwix_types(val.get_value())
  if hasattr(val, "value"):
    return _rebuild_qwix_types(val.value)
  return val


def _update_leaf_var(var, leaf_val, target_root, path):
  """Updates an NNX Variable or target leaf with the given leaf value."""
  leaf_val = _rebuild_qwix_types(leaf_val)
  target_sharding = getattr(var, "sharding", None)
  if target_sharding is None and hasattr(var, "get_value"):
    target_sharding = getattr(var.get_value(), "sharding", None)

  if (target_sharding is not None and
      isinstance(target_sharding, jax.sharding.Sharding) and
      not isinstance(leaf_val, jax.ShapeDtypeStruct)):
    leaf_val = jax.device_put(leaf_val, target_sharding)

  if isinstance(leaf_val, (jax.Array, np.ndarray)):
    leaf_val = jnp.copy(leaf_val)

  if hasattr(var, "set_value"):
    var.set_value(leaf_val)
  elif hasattr(var, "value"):
    var.value = leaf_val
  else:
    _set_nested_leaf(target_root, path, leaf_val)


def _norm_nnx_path_key(path_tuple, root_state=None):
  """Normalize NNX path key tuple into a canonical slash-delimited string key."""
  parts = [
      getattr(k, "key", str(k)) if not isinstance(k, (str, int)) else str(k)
      for k in path_tuple
  ]
  if "layers_remainder" in parts:
    num_scanned = 0
    if root_state is not None:
      pure_s = root_state.to_pure_dict() if hasattr(
          root_state, "to_pure_dict") else root_state
      if isinstance(pure_s, dict):

        def _count_scanned(d):
          if not isinstance(d, dict):
            return 0
          if "scanned_blocks" in d and isinstance(d["scanned_blocks"], dict):
            sb = d["scanned_blocks"]
            if "layers" in sb and isinstance(sb["layers"], dict):
              return len(sb["layers"])
            return len([
                k for k in sb.keys()
                if k.startswith("layers_") or str(k).isdigit()
            ])
          for v in d.values():
            if isinstance(v, dict):
              c = _count_scanned(v)
              if c > 0:
                return c
          return 0

        num_scanned = _count_scanned(pure_s)

    rem_idx = parts.index("layers_remainder")
    new_parts = list(parts[:rem_idx])
    for p in parts[rem_idx + 1:]:
      if p == "layers":
        continue
      if p.startswith("layers_") and p[7:].isdigit():
        global_idx = num_scanned + int(p[7:])
        new_parts.append(f"layers_{global_idx}")
      elif p.isdigit():
        global_idx = num_scanned + int(p)
        new_parts.append(f"layers_{global_idx}")
      else:
        new_parts.append(p)
    return _norm_path_key(new_parts)

  return _norm_path_key(parts)


def _update_nnx_state_from_pure_dict(nnx_state, pure_dict):
  """Overlays pure dictionary parameters onto target NNX state variables."""
  if isinstance(nnx_state, dict):
    target_state = nnx_state
  elif hasattr(nnx_state, "to_pure_dict"):
    target_state = nnx_state
  else:
    return

  pure_dict = pure_dict.to_pure_dict() if hasattr(pure_dict,
                                                  "to_pure_dict") else pure_dict
  if not isinstance(pure_dict, (dict, nnx.State)):
    return

  pure_dict = train_state_nnx._rename_nnx_to_linen_layers(pure_dict)  # pylint: disable=protected-access

  # Convert pure_dict to a flat mapping of norm_path -> raw_array
  flat_updates = {}
  if isinstance(pure_dict, (dict, nnx.State)):
    pure_flat = _flatten_and_norm_dict(pure_dict)
    for k, v in pure_flat.items():
      raw_val = _rebuild_qwix_types(v)
      if isinstance(raw_val, nnx.State):
        continue
      flat_updates[k] = raw_val

  leaves, _ = jax.tree_util.tree_flatten_with_path(
      nnx_state, is_leaf=lambda *args: isinstance(args[-1], nnx.Variable))
  for path, var in leaves:
    path_tuple = tuple(getattr(k, "key", str(k)) for k in path)
    norm_p = _norm_nnx_path_key(path_tuple, target_state)
    if norm_p in flat_updates:
      val = flat_updates[norm_p]
      if val is not None and not isinstance(val, jax.ShapeDtypeStruct):
        _update_leaf_var(var, val, target_state, path)


def _expected_and_restored_params(abstract_nnx_state, restored_linen):
  """Returns the model's expected weights and the checkpoint's restored weights, as pure dicts.

  Splits the abstract by Variable type (nnx.Param) so only real weights are compared --
  rngs/dropout/batch stats live in `nnx_aux` and are restored separately.
  """
  lora_params = nnx.filter_state(abstract_nnx_state, nnx.LoRAParam)
  if lora_params:
    want = lora_params.to_pure_dict().get("model", {})
  else:
    want = nnx.split_state(abstract_nnx_state, nnx.Param,
                           ...)[0].to_pure_dict().get("model", {})
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
  linen_state, aux_state, ephemeral = train_state_nnx.split_for_checkpoint(
      abstract_nnx_state)
  weights = train_state_nnx.from_linen_checkpoint_dict(restored_linen)
  if "model" in weights:
    _update_nnx_state_from_pure_dict(linen_state, {"model": weights["model"]})
  if "optimizer" in weights:
    _update_nnx_state_from_pure_dict(linen_state,
                                     {"optimizer": weights["optimizer"]})

  nnx_aux = restored_linen.get("nnx_aux")
  if nnx_aux:
    _update_nnx_state_from_pure_dict(aux_state, nnx_aux)

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
      ))
  restore_args = ocp.checkpoint_utils.construct_restore_args(linen_abstract)
  restored = ocp.args.PyTreeRestore(item=linen_abstract,
                                    restore_args=restore_args,
                                    partial_restore=True)
  restored = ckptr.restore(epath.Path(path), args=restored)
  _raise_on_weight_mismatch(
      *_expected_and_restored_params(abstract_nnx_state, restored))
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
  max_logging.log(
      f"Restoring emergency Linen-layout checkpoint into NNX state at step {step}"
  )
  linen_abstract = train_state_nnx.to_checkpoint_dict(abstract_nnx_state)
  restore_args = jax.tree_util.tree_map(map_to_pspec, linen_abstract)
  checkpoint_args = ocp.args.PyTreeRestore(
      item=linen_abstract,
      restore_args=restore_args,
      partial_restore=True,
  )
  restored = checkpoint_manager.restore(
      step, args=Composite(state=checkpoint_args)).state
  _raise_on_weight_mismatch(
      *_expected_and_restored_params(abstract_nnx_state, restored))
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
            path, abstract_unboxed_pre_state, checkpoint_storage_concurrent_gb,
            use_ocdbt, use_zarr3)
      context = ocp_v1.Context(
          checkpoint_layout=ocp_v1.options.CheckpointLayout.ORBAX)
      with context:
        return ocp_v1.load_pytree(path, abstract_unboxed_pre_state)
    elif source_checkpoint_layout == "safetensors":
      context = ocp_v1.Context(
          checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS)
      with context:
        metadata = ocp_v1.pytree_metadata(path)
        simple_abstract_state = metadata.metadata
        shardings = sharding_utils.construct_maximal_shardings(
            simple_abstract_state)

        def combine_sharding(sds, shardings):
          return jax.ShapeDtypeStruct(shape=sds.shape,
                                      dtype=sds.dtype,
                                      sharding=shardings)

        sharded_abstract_state = jax.tree.map(combine_sharding,
                                              simple_abstract_state, shardings)
        pre_transformed_state = ocp_v1.load_pytree(path, sharded_abstract_state)
      state = checkpoint_conversion_fn(pre_transformed_state)
      return state
    else:
      raise ocp_v1.errors.InvalidLayoutError(
          f"Unknown checkpoint layout: {source_checkpoint_layout}")
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
    return ocp.Checkpointer(handler).restore(p,
                                             restore_target,
                                             restore_args=restore_args)


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

  max_logging.log(
      f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}"
  )

  # Base configuration for all dataset types
  item_names = ("items",)
  # we need to use ocdbt and zarr3 to control max file size in the checkpoint
  item_handlers = {
      "items":
          PyTreeCheckpointHandler(
              restore_concurrent_gb=checkpoint_storage_concurrent_gb,
              save_concurrent_gb=checkpoint_storage_concurrent_gb,
              use_ocdbt=use_ocdbt,
              use_zarr3=use_zarr3,
          )
  }

  if dataset_type is not None and dataset_type == "grain":
    item_names += ("iter",)
    item_handlers["iter"] = grain_utility.GrainCheckpointHandler(
    )  # pyrefly: ignore[bad-assignment]

  # local storage checkpoint needs parent directory created
  p = gcs_utils.mkdir_and_check_permissions(checkpoint_dir)
  if enable_continuous_checkpointing:
    max_logging.log("Enabling policy for continuous checkpointing.")
    save_decision_policy = save_decision_policy_lib.ContinuousCheckpointingPolicy(
    )
  elif enable_autocheckpoint:
    max_logging.log("Enabling policy for autocheckpoint.")
    save_decision_policy = save_decision_policy_lib.AnySavePolicy([
        save_decision_policy_lib.PreemptionCheckpointingPolicy(),
        save_decision_policy_lib.FixedIntervalPolicy(save_interval_steps),
    ])
  else:
    max_logging.log("Enabling policy for fixed interval checkpointing.")
    save_decision_policy = save_decision_policy_lib.FixedIntervalPolicy(
        interval=save_interval_steps)
  preservation_policy = preservation_policy_lib.LatestN(
      max_num_checkpoints_to_keep)

  async_options = None
  if enable_continuous_checkpointing:
    async_options = ocp.AsyncOptions(timeout_secs=int(
        datetime.timedelta(minutes=60).total_seconds()),)
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
    data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] |
    None,
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
    max_logging.log(
        "checkpoint manager exists so trying to load this run's existing checkpoint"
    )

    step = checkpoint_manager.latest_step(
    ) if step < 0 else step  # pyrefly: ignore[bad-assignment]
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      def map_to_pspec(data):
        if not enable_single_replica_ckpt_restoring:
          return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)
        pspec = data.sharding.spec
        mesh = data.sharding.mesh
        replica_axis_index = 0
        replica_devices = grain_utility.replica_devices(mesh.devices,
                                                        replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(
            replica_mesh, pspec)

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
        ocp.type_handlers.register_type_handler(jax.Array,
                                                array_handler,
                                                override=True)

      # pure_nnx saves in the Linen on-disk layout; restore that layout (weights +
      # opt_state + step + nnx_aux), restoring the grain iterator in place when
      # present, then reshape it back into the NNX state.
      # (Emergency managers use their own restore path below.)
      if isinstance(abstract_unboxed_pre_state, nnx.State) and not isinstance(
          checkpoint_manager,
          (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager),
      ):
        linen_abstract = train_state_nnx.to_checkpoint_dict(
            abstract_unboxed_pre_state)
        restore_args = jax.tree_util.tree_map(map_to_pspec, linen_abstract)
        checkpoint_args = ocp.args.PyTreeRestore(item=linen_abstract,
                                                 restore_args=restore_args,
                                                 partial_restore=True)
        if (dataset_type == "grain" and data_iterator and
            not isinstance(data_iterator, PlaceHolderDataIterator) and
            (checkpoint_manager.directory / str(step) / "iter").exists()):
          restored, _ = grain_utility.restore_grain_iterator(
              checkpoint_manager, step, data_iterator, checkpoint_args,
              expansion_factor_real_data)
        else:
          restored = checkpoint_manager.restore(
              step, args=Composite(items=checkpoint_args))
        _raise_on_weight_mismatch(*_expected_and_restored_params(
            abstract_unboxed_pre_state, restored["items"]))
        restored_nnx = _linen_items_to_nnx(restored["items"],
                                           abstract_unboxed_pre_state)
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
          restored = checkpoint_manager.restore(
              step, args=Composite(state=checkpoint_args)).state
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
        ) if (dataset_type == "grain" and data_iterator and
              not isinstance(data_iterator, PlaceHolderDataIterator) and
              (checkpoint_manager.directory / str(step) / "iter").exists()):
          return grain_utility.restore_grain_iterator(
              checkpoint_manager,
              step,
              data_iterator,
              checkpoint_args,
              expansion_factor_real_data,
          )
        # Case 3: Default/Fallback case.
        # This case acts as a wildcard ('_') and matches if none of the preceding cases were met.
        case _:
          restored = checkpoint_manager.restore(
              step, args=Composite(items=checkpoint_args))
          return (restored, None)

  if source_checkpoint_layout == "safetensors_dynamic":
    path = load_parameters_from_path or load_full_state_from_path
    max_logging.log(
        f"Dynamic On-the-Fly Formatting: Loading SafeTensors from {path}")

    return load_safetensors_dynamic_state(path, abstract_unboxed_pre_state,
                                          maxtext_config)
  elif load_parameters_from_path != "":
    if isinstance(abstract_unboxed_pre_state, nnx.State):
      params = (abstract_unboxed_pre_state.model if hasattr(
          abstract_unboxed_pre_state, "model") else
                abstract_unboxed_pre_state["model"])
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
    max_logging.log(
        f"Loading full state from path: {load_full_state_from_path}")
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
        options=ocp.logging.CloudLoggerOptions(job_name=config.run_name,
                                               logger_name=logger_name))
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
  is_nnx = isinstance(abstract_unboxed_params, (nnx.State, nnx.Module))
  if is_nnx:
    unquant = lora_utils.restore_qlora_base_weights(abstract_unboxed_params)
    pure_want = unquant.to_pure_dict() if hasattr(unquant,
                                                  "to_pure_dict") else unquant
    candidate_wants = [
        pure_want,
        train_state_nnx._rename_nnx_to_linen_layers(pure_want)
    ]  # pylint: disable=protected-access
    max_logging.log(
        f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}"
    )
    ckptr = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(
            restore_concurrent_gb=checkpoint_storage_concurrent_gb,
            save_concurrent_gb=checkpoint_storage_concurrent_gb,
            use_ocdbt=use_ocdbt,
            use_zarr3=use_zarr3,
        ))
    last_err = None
    for want in candidate_wants:
      params_collection = {"params": want}
      restore_args = ocp.checkpoint_utils.construct_restore_args(
          params_collection)
      restored = ckptr.restore(
          epath.Path(load_parameters_from_path),
          item={"params": params_collection},
          transforms={},
          restore_args={"params": restore_args},
      )
      restored_collection = restored["params"]
      try:
        _raise_on_weight_mismatch(want, restored_collection["params"])
        max_logging.log(
            "load_params_from_path: successfully restored parameters.")
        _update_nnx_state_from_pure_dict(abstract_unboxed_params,
                                         restored_collection["params"])
        return abstract_unboxed_params
      except ValueError as e:
        last_err = e
    if last_err is not None:
      raise last_err
  else:
    want = abstract_unboxed_params
    params_collection = want
    max_logging.log(
        f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}"
    )
    ckptr = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(
            restore_concurrent_gb=checkpoint_storage_concurrent_gb,
            save_concurrent_gb=checkpoint_storage_concurrent_gb,
            use_ocdbt=use_ocdbt,
            use_zarr3=use_zarr3,
        ))
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        params_collection)
    restored = ckptr.restore(
        epath.Path(load_parameters_from_path),
        item={"params": params_collection},
        transforms={},
        restore_args={"params": restore_args},
    )
    restored_collection = restored["params"]
    _raise_on_weight_mismatch(want, restored_collection)
    return restored_collection
  return restored_collection


def save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  print(
      f"Saving quantized params checkpoint with use_ocdbt = {use_ocdbt} and use_zarr3 = {use_zarr3}"
  )
  orbax_checkpointer = ocp.PyTreeCheckpointer(use_ocdbt=use_ocdbt,
                                              use_zarr3=use_zarr3)
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
  local_checkpoint_due = _uses_local_checkpoint_period(
      config) and step % config.local_checkpoint_period == 0
  autocheckpoint_due = config.enable_autocheckpoint and checkpoint_manager.reached_preemption(
      step)
  return base_checkpoint_due or local_checkpoint_due or autocheckpoint_due


def _handle_post_checkpoint_preemption(checkpoint_manager, step,
                                       force_ckpt_save):
  """Waits on final/preemption saves and raises if preempted."""
  reached_preemption = checkpoint_manager.reached_preemption(step)
  if force_ckpt_save or reached_preemption:
    checkpoint_manager.wait_until_finished()
  if reached_preemption:
    raise exceptions.StopTraining("Job is preempted.")


def maybe_save_checkpoint(checkpoint_manager,
                          state,
                          config,
                          data_iterator,
                          step=None):
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
      actual_step = int(
          state.step if config.enable_diloco else state.optimizer.step) - 1
    else:
      # Linen TrainState has .step attribute
      actual_step = int(state.step) - 1

  # Determine if a checkpoint save should be forced, overriding the usual
  # `config.checkpoint_period` logic.
  # This occurs if this function was called:
  # without an explicit 'step' (implying it's a checkpoint save for final step),
  # AND the 'actual_step' is a valid step,
  # AND it's not a step that would normally trigger a checkpoint save.
  force_ckpt_save = step is None and actual_step != -1 and (
      actual_step % config.checkpoint_period != 0)

  if not _should_save_checkpoint_at_step(checkpoint_manager, actual_step,
                                         config, force_ckpt_save):
    _handle_post_checkpoint_preemption(checkpoint_manager, actual_step,
                                       force_ckpt_save)
    return

  if checkpoint_manager.latest_step() == actual_step:
    max_logging.log(
        f"Checkpoint for step {actual_step} already exists, skipping save.")
    return

  if config.pure_nnx:
    # Save in the Linen on-disk layout so pure_nnx and Linen checkpoints are interchangeable.
    if config.enable_diloco:
      # DiLoCoTrainState: persist the synchronized global model (outer params).
      # The per-replica inner optimizer / outer-momentum state is not checkpointed.
      step_value = state.step.get_value() if hasattr(
          state.step, "get_value") else state.step
      state = train_state_nnx.to_linen_checkpoint_dict({
          "model": state.params,
          "optimizer": {
              "step": step_value
          }
      })
    else:
      # rngs/dropout/batch-stats are packed under items/nnx_aux so the RNG/dropout
      # stream continues across resumes instead of resetting to a base key.
      state = train_state_nnx.to_checkpoint_dict(state)

  try:
    checkpoint_saved = save_checkpoint(checkpoint_manager, actual_step, state,
                                       config, data_iterator, force_ckpt_save)
    if checkpoint_saved:
      print_save_message(actual_step, config.async_checkpointing)
    if config.elastic_enabled:
      elastic_utils.maybe_elastic_scale_up(config, checkpoint_manager)
  except elastic_utils.manager.ScaleUpSignalError as e:
    if config.elastic_enabled:
      max_logging.log(
          f"Elastic event detected, letting exception bubble up: {e}")
      raise
    else:
      raise exceptions.StopTraining("Job is preempted.") from e
  except jax.errors.JaxRuntimeError as e:
    if config.elastic_enabled:
      max_logging.log(
          f"Elastic event detected, letting exception bubble up: {e}")
      raise
    else:
      raise exceptions.StopTraining("Job is preempted.") from e
  except Exception as e:
    raise exceptions.StopTraining(f"Checkpointing failed. {str(e)}") from e

  # Wait for any pending checkpoint save to finish during preemption or final
  # step save, then raise upon preemption.
  _handle_post_checkpoint_preemption(checkpoint_manager, actual_step,
                                     force_ckpt_save)


def save_checkpoint(checkpoint_manager,
                    step,
                    state,
                    config=None,
                    data_iterator=None,
                    force=False):
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (force or (step % config.checkpoint_period == 0 and
                  not config.enable_continuous_checkpointing) or
        (_uses_local_checkpoint_period(config) and
         step % config.local_checkpoint_period == 0) or
        (config.enable_autocheckpoint and
         checkpoint_manager.reached_preemption(step))):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing.")

  # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
  chunk_byte_size = (config.checkpoint_storage_target_data_file_size_bytes
                     if config else DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE)

  checkpoint_args = ocp.args.PyTreeSave(
      item=state,
      save_args=jax.tree.map(
          lambda _: ocp.SaveArgs(chunk_byte_size=chunk_byte_size), state),
      ocdbt_target_data_file_size=chunk_byte_size,
  )

  save_args_composite = {"items": checkpoint_args}

  if config and config.dataset_type == "grain" and not isinstance(
      data_iterator, PlaceHolderDataIterator):
    if isinstance(data_iterator, RemoteIteratorWrapper):
      # Pass the wrapper directly; GrainCheckpointHandler will call save_state with the step
      save_args_composite["iter"] = grain_utility.GrainCheckpointSave(
          item=data_iterator)  # pyrefly: ignore[bad-assignment]
    elif not isinstance(data_iterator, list) and isinstance(
        data_iterator.local_iterator,
        ElasticIterator):  # pyrefly: ignore[missing-attribute]
      # ElasticIterator checkpoints a single global scalar shared by all shards.
      save_args_composite["iter"] = grain_utility.GrainCheckpointSave(
          item=data_iterator.local_iterator)  # pyrefly: ignore[bad-assignment]
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
            (data_iter.local_iterator, process_index,
             process_count_total))  # pyrefly: ignore[missing-attribute]
      save_args_composite["iter"] = grain_utility.GrainCheckpointSave(
          item=grain_iters_to_save)  # pyrefly: ignore[bad-assignment]

  custom_metadata = {}
  if config:
    if hasattr(config, "scan_layers"):
      custom_metadata["scan_layers"] = config.scan_layers
    if hasattr(config, "lora") and config.lora and getattr(
        config.lora, "lora_rank", 0) > 0:
      custom_metadata["lora"] = config.lora.model_dump()

  match (checkpoint_manager, config, data_iterator):
    case (checkpoint_manager, _, _) if isinstance(
        checkpoint_manager,
        (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
      emergency_checkpointing.replicator_error_handler(config)
      return checkpoint_manager.save(step,
                                     args=Composite(state=checkpoint_args),
                                     force=force)
    case _:
      return checkpoint_manager.save(step,
                                     args=Composite(**save_args_composite),
                                     force=force,
                                     custom_metadata=custom_metadata)
