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

"""Common LoRA utils needed to support LoRA adapters."""

from collections.abc import Mapping
from functools import partial
import json
import os
import re
from typing import Any, Optional

from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
import qwix

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

# NNX-only imports (`flax.nnx`, `train_state_nnx`, `model_creation_utils`) are
# loaded lazily inside the NNX dispatch branches so the Linen-only flow doesn't
# pull them in.


def apply_lora_on_base_params(base_params, lora_params, lora_scale_factor=1.0):
  """
  Apply the LoRA weights on the base weights of the model using formula:
                W_new = W + BA, where
                    W_new is the new weights with LoRA applied
                    W is the base model weights
                    B is lora_b adapter weights
                    A is lora_a adapter weights

  Here both the base_params and lora_params are PyTrees of same structure depending
  on the base model. The leaf nodes of lora_params are only not-None if it is the target
  module for lora in its config.
  """

  def lora_update_or_base(base_weight, lora_a, lora_b):
    if lora_a is not None and lora_b is not None:
      return base_weight + jnp.einsum("br,rnd->bnd", lora_b, lora_a) * lora_scale_factor
    else:
      return base_weight  # Keep the base weight if no Lora update

  def apply_lora_recursively(base_params, lora_params, module_name):
    for name, param in lora_params.items():
      if isinstance(param, dict):
        apply_lora_recursively(base_params[name], param, f"{module_name}.{name}")
      elif param is not None:
        if name not in ["lora_a.kernel", "lora_b.kernel"]:
          raise ValueError(f"Unexpected non-lora specific weights ({module_name}.{name}) found in the lora_params")

        lora_b = lora_params["lora_a.kernel"]
        lora_a = lora_params["lora_b.kernel"]

        base = base_params["kernel"]

        base_params["kernel"] = lora_update_or_base(base, lora_a, lora_b)
        break

  apply_lora_recursively(base_params, lora_params, "")


def unapply_lora_from_base_params(base_params, lora_params, lora_scale_factor=1.0):
  """
  Unapply the LoRA weights from the base weights of the model using formula:
                W_org = W - BA, where
                    W is the premerged weights of base and LoRA
                    W_org is the original base model weights
                    B is lora_b adapter weights
                    A is lora_a adapter weights

  Here both the base_params and lora_params are PyTrees of same structure depending
  on the base model. The leaf nodes of lora_params are only not-None if it is the target
  module for lora in its config.
  """

  def lora_update_or_base(base_weight, lora_a, lora_b):
    if lora_a is not None and lora_b is not None:
      return base_weight - jnp.einsum("br,rnd->bnd", lora_b, lora_a) * lora_scale_factor
    else:
      return base_weight  # Keep the base weight if no Lora update

  def unapply_lora_recursively(base_params, lora_params, module_name):
    for name, param in lora_params.items():
      if isinstance(param, dict):
        unapply_lora_recursively(base_params[name], param, f"{module_name}.{name}")
      elif param is not None:
        if name not in ["lora_a.kernel", "lora_b.kernel"]:
          raise ValueError(f"Unexpected non-lora specific weights ({module_name}.{name}) found in the lora_params")

        lora_b = lora_params["lora_a.kernel"]
        lora_a = lora_params["lora_b.kernel"]

        base_kernel = base_params["kernel"]

        base_params["kernel"] = lora_update_or_base(base_kernel, lora_a, lora_b)
        break

  unapply_lora_recursively(base_params, lora_params, "")


def load_adapter(config, base_abstract_state_params, adapter_config_path, adapter_weights_path):
  """Load LoRA weights into a PyTree and return it.

  On the NNX path, `base_abstract_state_params` and the returned `lora_params`
  are `nnx.State`-shaped (no outer `{"params": ...}` wrap).
  """
  # Load LoRA weights
  lora_params = None
  lora_config = None
  if adapter_config_path:
    if adapter_config_path.startswith("gs://"):
      lora_config = gcs_utils.read_json_from_gcs(adapter_config_path)
    else:
      with open(adapter_config_path, "rt", encoding="utf8") as f:
        lora_config = json.load(f)

    if lora_config is None:
      raise FileNotFoundError(f"Failed to read lora_config from {adapter_config_path}.")

    if not gcs_utils.gcs_path_exists(f"{adapter_weights_path}/commit_success.txt"):
      raise FileNotFoundError(f"Failed to read lora_weights from {adapter_weights_path}.")

    if config.pure_nnx:
      lora_state, _ = get_lora_abstract_state_nnx(base_abstract_state_params, lora_config)
    else:
      lora_state, _ = get_lora_abstract_state(base_abstract_state_params, lora_config)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      lora_params = checkpointing.load_params_from_path(
          adapter_weights_path,
          lora_state.params,
          config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt,
          config.checkpoint_storage_use_zarr3,
      )

  return lora_params, lora_config


def setup_initial_lora_state(model, data_iterator, tx, config, rng, mesh, checkpoint_manager, lora_adapter_path):
  """Initialize the LoRA train state and optionally load weights from disk.

  Returns `(lora_config, lora_state, lora_state_annotations)`. On the NNX path
  `model` is unused (the NNX abstract state is built via
  `model_creation_utils.create_nnx_abstract_model`) and `lora_state.params`
  is `nnx.State`-shaped; on Linen it is the original `{"params": ...}` tree.
  """

  lora_state = None
  lora_state_annotations = None
  lora_config = None

  if lora_adapter_path:
    max_logging.log(f"Setting initial state of LoRA with lora_adapter_path = {lora_adapter_path}")
    if config.pure_nnx:
      # pylint: disable=import-outside-toplevel
      from maxtext.layers import train_state_nnx
      from maxtext.utils import model_creation_utils

      _create_model_partial, _ = model_creation_utils.create_nnx_abstract_model(config, mesh)

      def create_train_state_fn():
        nnx_model = _create_model_partial()
        optimizer = nnx.Optimizer(nnx_model, tx, wrt=nnx.Param)
        return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

      init_state_fn = create_train_state_fn
    else:
      init_state_fn = partial(maxtext_utils.init_initial_state, model, tx, config, True, rng)
    unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, True)

    lora_config_path = lora_adapter_path + "adapter_config.json"

    lora_config = gcs_utils.read_json_from_gcs(lora_config_path)

    if config.pure_nnx:
      base_abstract_params = _nnx_param_subtree(unboxed_abstract_state)
      lora_state, lora_state_annotations = get_lora_abstract_state_nnx(base_abstract_params, lora_config)
    else:
      lora_state, lora_state_annotations = get_lora_abstract_state(unboxed_abstract_state.params, lora_config)

    lora_weights_path = f"{lora_adapter_path}/0/items"

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      restored_lora, raw_lora_params = checkpointing.load_state_if_possible(
          checkpoint_manager,
          data_iterator,
          lora_weights_path,
          config.load_full_state_path,
          config.checkpoint_storage_concurrent_gb,
          lora_state,
          config.enable_single_replica_ckpt_restoring,
          config.dataset_type,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
      )

      if restored_lora:
        raise NotImplementedError("This codepath is not implemented for LoRA adapters yet.")
      else:
        lora_state = lora_state.replace(params=raw_lora_params)
        lora_state = max_utils.unbox_logicallypartioned(lora_state)

  return lora_config, lora_state, lora_state_annotations


def get_lora_abstract_state(base_abstract_params, lora_config):
  """
  Generates an abstract state representing only the LoRA parameters,
  inferring sharding information from the base parameters.

  Args:
    base_abstract_params: A PyTree containing jax.ShapeDtypeStruct objects
                            representing the abstract state of the base model
                            parameters. This includes sharding information.
    lora_config: A config of the Lora adapter that includes details about the
                 Lora like rank, or target_modules on which lora is implemented.

  Returns:
    A TrainState object representing the abstract state of the LoRA parameters, including
    inferred sharding information.
  """
  other_lora_format_to_jax_format = {
      "q_proj": "self_attention.query",
      "k_proj": "self_attention.key",
      "v_proj": "self_attention.value",
      "o_proj": "self_attention.out",
  }

  lora_target_modules = lora_config["target_modules"]
  lora_target_modules = [other_lora_format_to_jax_format.get(s, s) for s in lora_target_modules]

  lora_rank = int(lora_config["r"])

  lora_abstract_params = {}

  def get_lora_param_shape(base_array_shape, lora_rank, lora_module):
    base_array_dimensions = len(base_array_shape)

    if base_array_dimensions > 4:
      raise ValueError(
          f"Encountered unexpected shape={base_array_shape} of array in base params. Array dimensions > 4 not supported."
      )

    if lora_module in [
        "self_attention.query",
        "self_attention.key",
        "self_attention.value",
    ]:
      lora_a_shape = base_array_shape[:-2] + (lora_rank,)
      lora_b_shape = (lora_rank,) + base_array_shape[1:]
    elif lora_module in ["self_attention.out"]:
      lora_a_shape = base_array_shape[:-1] + (lora_rank,)
      if base_array_dimensions == 4:
        lora_b_shape = (lora_rank, base_array_shape[1], base_array_shape[-1])
      else:
        lora_b_shape = (lora_rank, base_array_shape[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")

    return lora_a_shape, lora_b_shape

  def get_lora_param_sharding(base_param_sharding, lora_module):
    if base_param_sharding is None:  # Base parameter is replicated
      return None, None  # Replicate LoRA parameters as well

    base_sharding_pspec_size = len(base_param_sharding.spec)

    if base_sharding_pspec_size > 4:
      raise ValueError("Encountered unexpected size of PartitionSpec in sharding. Size > 4 is not supported")

    base_mesh = base_param_sharding.mesh
    base_memory_kind = base_param_sharding.memory_kind
    base_pspec = base_param_sharding.spec

    if lora_module in [
        "self_attention.query",
        "self_attention.key",
        "self_attention.value",
    ]:
      lora_a_pspec_tuple = base_pspec[:-2] + ((),)
      lora_a_pspec = jax.sharding.PartitionSpec(*lora_a_pspec_tuple)

      lora_b_pspec_tuple = ((),) + base_pspec[1:]
      lora_b_pspec = jax.sharding.PartitionSpec(*lora_b_pspec_tuple)

    elif lora_module in ["self_attention.out"]:
      lora_a_pspec_tuple = base_pspec[:-1] + ((),)
      lora_a_pspec = jax.sharding.PartitionSpec(*lora_a_pspec_tuple)
      if base_sharding_pspec_size == 4:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[1], base_pspec[-1])
      else:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")

    lora_a_sharding = jax.sharding.NamedSharding(mesh=base_mesh, spec=lora_a_pspec, memory_kind=base_memory_kind)
    lora_b_sharding = jax.sharding.NamedSharding(mesh=base_mesh, spec=lora_b_pspec, memory_kind=base_memory_kind)

    return lora_a_sharding, lora_b_sharding

  def module_is_target_module(module, target_modules):
    """Checks if any of the target_modules is part of the current module which represents an array.

    Args:
      module: A string where nested dictionary keys are concatenated to make a path of the internal most kernel/scale arrays.
      target_modules: A list of strings which represents the target_modules on which lora is applied.

    Return:
      The matched target_module, if that is found in the current module path, None otherwise.
    """
    for target_module in target_modules:
      if target_module in module:
        return target_module
    return None

  def add_lora_params(lora_params, module_name, base_params, lora_rank, lora_target_modules):
    for name, param in base_params.items():
      if isinstance(param, dict):
        lora_params[name] = {}
        add_lora_params(
            lora_params[name],
            f"{module_name}.{name}",
            param,
            lora_rank,
            lora_target_modules,
        )
      else:
        if name not in ["kernel", "scale", "embedding"]:
          raise ValueError(f"Unexpected key={name} exists in the abstract params of base model.")

        if not isinstance(param, jax.ShapeDtypeStruct):
          raise ValueError("Unexpected type found in the abstract params of the base model.")

        lora_a_key = "lora_a.kernel"
        lora_b_key = "lora_b.kernel"

        target_module = module_is_target_module(module_name, lora_target_modules)

        if target_module is not None:
          lora_a_shape, lora_b_shape = get_lora_param_shape(param.shape, lora_rank, target_module)
          base_dtype = param.dtype
          lora_a_sharding, lora_b_sharding = get_lora_param_sharding(param.sharding, target_module)

          lora_params[lora_a_key] = jax.ShapeDtypeStruct(shape=lora_a_shape, dtype=base_dtype, sharding=lora_a_sharding)

          lora_params[lora_b_key] = jax.ShapeDtypeStruct(shape=lora_b_shape, dtype=base_dtype, sharding=lora_b_sharding)
        else:
          lora_params[name] = None

  def get_lora_annotations(lora_abstract_params):
    return jax.tree_util.tree_map(lambda x: x.sharding.spec, lora_abstract_params)

  add_lora_params(lora_abstract_params, "", base_abstract_params, lora_rank, lora_target_modules)

  unboxed_abstract_lora_state = train_state.TrainState(
      step=0, apply_fn=None, params=lora_abstract_params, tx=None, opt_state={}  # type: ignore
  )

  lora_state_mesh_annotations = train_state.TrainState(
      step=0, apply_fn=None, params=get_lora_annotations(lora_abstract_params), tx=None, opt_state={}  # type: ignore
  )

  return unboxed_abstract_lora_state, lora_state_mesh_annotations


# --- Qwix LoRA Utils ---


def _get_lora_module_path(mt_config: pyconfig.HyperParameters) -> str:
  """Gets the regex for modules to apply LoRA on from config, architecture map, or fallback."""
  if mt_config.lora.lora_module_path:
    return mt_config.lora.lora_module_path

  config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "lora_module_path.yml")
  lora_configs = pyconfig._load_config(config_path)  # pylint: disable=protected-access
  model_name = mt_config.model_name.lower()

  # Find the first matching architecture prefix or use 'default'
  matched_key = next((k for k in lora_configs if k != "default" and model_name.startswith(k)), "default")

  if matched_key == "default":
    max_logging.log(f"Warning: Model '{model_name}' is unverified; falling back to default LoRA path.")
  else:
    max_logging.log(f"Auto-detected lora_module_path for model '{model_name}' (matched: '{matched_key}')")

  raw_path = lora_configs.get(matched_key, "decoder/layers/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))")

  # This regex makes the layer index optional, matching both scanned and unscanned layer paths
  # (e.g. 'layers/0/mlp/...' vs 'layers/mlp/...').
  optional_layer_index = "(?:[0-9]+/)?"
  final_path = str(raw_path).replace("layers/", f"layers/{optional_layer_index}")

  max_logging.log(f"Using lora_module_path: {final_path}")
  return final_path


def _build_lora_provider(mt_config: pyconfig.HyperParameters) -> qwix.LoraProvider:
  """Builds a Qwix LoRA provider from MaxText LoRA settings."""
  lora_module_path = _get_lora_module_path(mt_config)
  lora_kwargs = {
      "module_path": lora_module_path,
      "rank": mt_config.lora.lora_rank,
      "alpha": mt_config.lora.lora_alpha,
      "dropout": 0.0,
  }
  max_logging.log(
      f"LoRA configured: module_path={lora_module_path} "
      f"rank={mt_config.lora.lora_rank} alpha={mt_config.lora.lora_alpha}"
  )
  return qwix.LoraProvider(**lora_kwargs)


def _prepare_dummy_inputs() -> tuple[jnp.ndarray, jnp.ndarray]:
  """Builds dummy decoder inputs used to materialize LoRA parameters."""
  # Keep LoRA warmup as small as possible to minimize compile/memory overhead.
  dummy_bs = 1
  seq_len = 1
  decoder_input_tokens = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  decoder_positions = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  return decoder_input_tokens, decoder_positions


def is_lora_enabled(model: nnx.Module) -> bool:
  """Checks if the model has LoRA parameters."""
  for _, value in nnx.iter_graph(model):
    if isinstance(value, nnx.LoRAParam):
      return True
  return False


def _verify_lora_parameters(lora_model: nnx.Module, mt_config: pyconfig.HyperParameters):
  """Validates that LoRA is active or that target modules were matched."""

  if is_lora_enabled(lora_model):
    return

  lora_module_path = _get_lora_module_path(mt_config)
  compiled_module_path = re.compile(lora_module_path)
  matched_module_paths = []
  sample_module_paths = []

  for path, _ in nnx.iter_modules(lora_model):
    module_path = "/".join(str(p) for p in path)
    if len(sample_module_paths) < 100:
      sample_module_paths.append(module_path)
    if compiled_module_path.search(module_path):
      matched_module_paths.append(module_path)

  if not matched_module_paths:
    max_logging.log(
        f"LoRA module_path='{lora_module_path}' did not match any weights. " f"Sample module paths: {sample_module_paths}"
    )
    raise ValueError("LoRA enabled but no LoRA parameters found in decoder/model state.")

  raise ValueError(
      "LoRA module path matched target modules, but nnx.LoRAParam is still "
      "missing. For Tunix PeftTrainer, LoRA params must be materialized before "
      "trainer initialization, otherwise it falls back to full-model training. "
      f"Sample matches: {matched_module_paths[:10]}"
  )


def apply_lora_to_model(
    model: nnx.Module,
    mesh: Optional[jax.sharding.Mesh],
    mt_config: pyconfig.HyperParameters,
) -> nnx.Module:
  """Optionally applies LoRA/QLoRA to a MaxText model using Qwix."""
  # Skip Qwix LoRA if MaxText LoRA adapters are loaded
  if mt_config.lora_input_adapters_path:
    max_logging.log("MaxText LoRA adapters loaded, skipping Qwix LoRA application")
    return model

  if not mt_config.lora.enable_lora:
    return model

  # Dynamically detect and set LoRA rank before model creation if restoring

  lora_provider = _build_lora_provider(mt_config)

  model_rngs = getattr(model.decoder, "rngs", None)
  decoder_input_tokens, decoder_positions = _prepare_dummy_inputs()

  lora_model = qwix.apply_lora_to_model(
      model,
      lora_provider,
      decoder_input_tokens=decoder_input_tokens,
      decoder_positions=decoder_positions,
      rngs=model_rngs,
  )

  if mesh is not None:
    with mesh, nn_partitioning.axis_rules(mt_config.logical_axis_rules):
      graph_def, state = nnx.split(lora_model)

      # We handle explicit replication for LoRA to ensure safety and efficiency.
      state = jax.tree_util.tree_map(
          lambda x: x.replace(sharding=jax.sharding.PartitionSpec(), out_sharding=None, sharding_names=None)
          if isinstance(x, nnx.LoRAParam)
          else x,
          state,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )

      # Use logical_to_mesh_sharding to correctly map logical axes like 'embed'
      # to physical mesh axes.
      dst_shardings = sharding.logical_to_mesh_sharding(
          nnx.get_partition_spec(state), mesh, rules=mt_config.logical_axis_rules
      )

      from tunix.rl import reshard  # pylint: disable=import-outside-toplevel

      state = reshard.reshard_pytree(state, dst_shardings)
      lora_model = nnx.merge(graph_def, state)

  _verify_lora_parameters(lora_model, mt_config)

  return lora_model


def restore_lora_from_path(trainer: Any, mt_config: pyconfig.HyperParameters) -> Any:
  """Restores LoRA parameter weights from an external Orbax checkpoint for a fresh run."""
  lora_restore_path = mt_config.lora.lora_restore_path

  train_steps = getattr(trainer, "train_steps", 0)
  if train_steps > 0:
    max_logging.log(
        f"PeftTrainer restored current run at step {train_steps}; " f"ignoring lora_restore_path '{lora_restore_path}'."
    )
    return trainer

  if not is_lora_enabled(trainer.model):
    lora_module_path = _get_lora_module_path(mt_config)
    if not mt_config.lora.enable_lora:
      raise ValueError(
          "lora_restore_path is set but LoRA is not enabled on the model. "
          f"Set lora.enable_lora=True and verify lora_module_path ('{lora_module_path}') matches model modules."
      )

  abstract_lora_params = nnx.state(trainer.model, nnx.LoRAParam)

  target_for_restore = jax.tree.map(
      lambda v: {"value": v.value},
      abstract_lora_params,
      is_leaf=lambda n: isinstance(n, nnx.Variable),
  )

  sharding_tree = jax.tree.map(lambda x: x.sharding if hasattr(x, "sharding") else None, target_for_restore)
  restore_args_tree = ocp.checkpoint_utils.construct_restore_args(target_for_restore, sharding_tree)

  try:
    restore_args = ocp.args.PyTreeRestore(
        item=target_for_restore,
        restore_args=restore_args_tree,
        partial_restore=True,
    )
    restored_lora_params = ocp.PyTreeCheckpointer().restore(
        lora_restore_path,
        args=restore_args,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Guided restore failed: {e}. Falling back to basic restore.")
    restored_lora_params = ocp.PyTreeCheckpointer().restore(lora_restore_path)

  # Post processing
  def _map_to_state(path, variable):
    if not isinstance(variable, nnx.Variable):
      return

    str_path = [str(k.key if hasattr(k, "key") else (k.name if hasattr(k, "name") else k)) for k in path]

    curr = restored_lora_params
    for p in str_path:
      if isinstance(curr, dict) and p in curr:
        curr = curr[p]
      elif hasattr(curr, p):
        curr = getattr(curr, p)
      else:
        return

    if isinstance(curr, dict) and "value" in curr:
      matched_val = curr["value"]
    elif hasattr(curr, "value"):
      matched_val = getattr(curr, "value")
    else:
      matched_val = curr

    variable.value = matched_val

  jax.tree_util.tree_map_with_path(
      _map_to_state,
      abstract_lora_params,
      is_leaf=lambda n: isinstance(n, nnx.Variable),
  )

  nnx.update(trainer.model, abstract_lora_params)
  max_logging.log(f"LoRA restore complete from '{lora_restore_path}'.")
  return trainer


# NNX-shaped LoRA helpers. The Linen walkers above key on `isinstance(x, dict)`
# and bare leaves; NNX trees use `nnx.State` (Mapping but not dict) and
# Variable-wrapped leaves, so we need separate mirrors. The math (W += B @ A * s)
# is identical.


def _is_nnx_branch(x):
  return isinstance(x, Mapping)


def _nnx_param_subtree(unboxed_abstract_state):
  """Drop the outer TrainStateNNX wrapping and return the model substate."""
  return unboxed_abstract_state["model"] if "model" in unboxed_abstract_state else unboxed_abstract_state


def apply_lora_on_base_params_nnx(base_params, lora_params, lora_scale_factor=1.0):
  """NNX variant of `apply_lora_on_base_params`. Mutates `base_params` in place."""

  def lora_update_or_base(base_weight, lora_a, lora_b):
    if lora_a is not None and lora_b is not None:
      return base_weight + jnp.einsum("br,rnd->bnd", lora_b, lora_a) * lora_scale_factor
    return base_weight

  def recurse(base_node, lora_node, path):
    for name, lora_child in lora_node.items():
      if _is_nnx_branch(lora_child):
        recurse(base_node[name], lora_child, f"{path}.{name}")
      elif lora_child is not None:
        if name not in ("lora_a.kernel", "lora_b.kernel"):
          raise ValueError(f"Unexpected non-lora key ({path}.{name}) in lora_params")
        lora_b = lora_node["lora_a.kernel"]
        lora_a = lora_node["lora_b.kernel"]
        base_node["kernel"] = lora_update_or_base(base_node["kernel"], lora_a, lora_b)
        return

  recurse(base_params, lora_params, "")


def unapply_lora_from_base_params_nnx(base_params, lora_params, lora_scale_factor=1.0):
  """NNX-shaped variant of `unapply_lora_from_base_params`. Mutates `base_params`."""

  def lora_update_or_base(base_weight, lora_a, lora_b):
    if lora_a is not None and lora_b is not None:
      return base_weight - jnp.einsum("br,rnd->bnd", lora_b, lora_a) * lora_scale_factor
    return base_weight

  def recurse(base_node, lora_node, path):
    for name, lora_child in lora_node.items():
      if _is_nnx_branch(lora_child):
        recurse(base_node[name], lora_child, f"{path}.{name}")
      elif lora_child is not None:
        if name not in ("lora_a.kernel", "lora_b.kernel"):
          raise ValueError(f"Unexpected non-lora key ({path}.{name}) in lora_params")
        lora_b = lora_node["lora_a.kernel"]
        lora_a = lora_node["lora_b.kernel"]
        base_node["kernel"] = lora_update_or_base(base_node["kernel"], lora_a, lora_b)
        return

  recurse(base_params, lora_params, "")


def get_lora_abstract_state_nnx(base_abstract_params, lora_config):
  """`get_lora_abstract_state` for the NNX path.

  Walks the abstract `state.model` substate and emits a parallel tree with
  `lora_a.kernel` / `lora_b.kernel` leaves at target attention paths and
  `None` elsewhere.
  """
  other_lora_format_to_jax_format = {
      "q_proj": "self_attention.query",
      "k_proj": "self_attention.key",
      "v_proj": "self_attention.value",
      "o_proj": "self_attention.out",
  }

  lora_target_modules = [other_lora_format_to_jax_format.get(s, s) for s in lora_config["target_modules"]]
  lora_rank = int(lora_config["r"])

  def get_lora_param_shape(base_array_shape, lora_module):
    if len(base_array_shape) > 4:
      raise ValueError(f"Unsupported base array shape {base_array_shape} (>4D)")
    if lora_module in ("self_attention.query", "self_attention.key", "self_attention.value"):
      lora_a_shape = base_array_shape[:-2] + (lora_rank,)
      lora_b_shape = (lora_rank,) + base_array_shape[1:]
    elif lora_module == "self_attention.out":
      lora_a_shape = base_array_shape[:-1] + (lora_rank,)
      if len(base_array_shape) == 4:
        lora_b_shape = (lora_rank, base_array_shape[1], base_array_shape[-1])
      else:
        lora_b_shape = (lora_rank, base_array_shape[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")
    return lora_a_shape, lora_b_shape

  def get_lora_param_sharding(base_param_sharding, lora_module):
    if base_param_sharding is None:
      return None, None
    base_pspec = base_param_sharding.spec
    if len(base_pspec) > 4:
      raise ValueError("PartitionSpec size > 4 not supported")
    if lora_module in ("self_attention.query", "self_attention.key", "self_attention.value"):
      lora_a_pspec = jax.sharding.PartitionSpec(*(base_pspec[:-2] + ((),)))
      lora_b_pspec = jax.sharding.PartitionSpec(*(((),) + base_pspec[1:]))
    elif lora_module == "self_attention.out":
      lora_a_pspec = jax.sharding.PartitionSpec(*(base_pspec[:-1] + ((),)))
      if len(base_pspec) == 4:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[1], base_pspec[-1])
      else:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")
    mesh = base_param_sharding.mesh
    mem_kind = base_param_sharding.memory_kind
    return (
        jax.sharding.NamedSharding(mesh=mesh, spec=lora_a_pspec, memory_kind=mem_kind),
        jax.sharding.NamedSharding(mesh=mesh, spec=lora_b_pspec, memory_kind=mem_kind),
    )

  def module_is_target(module_path):
    for tgt in lora_target_modules:
      if tgt in module_path:
        return tgt
    return None

  def add_lora(out_node, base_node, path):
    for name, child in base_node.items():
      if _is_nnx_branch(child):
        out_node[name] = {}
        add_lora(out_node[name], child, f"{path}.{name}")
      else:
        if name not in ("kernel", "scale", "embedding"):
          raise ValueError(f"Unexpected key={name} in base abstract params at {path}")
        if not isinstance(child, jax.ShapeDtypeStruct):
          raise ValueError(f"Unexpected leaf type {type(child).__name__} at {path}.{name}")
        target_module = module_is_target(path)
        if target_module is not None:
          a_shape, b_shape = get_lora_param_shape(child.shape, target_module)
          a_sharding, b_sharding = get_lora_param_sharding(child.sharding, target_module)
          out_node["lora_a.kernel"] = jax.ShapeDtypeStruct(shape=a_shape, dtype=child.dtype, sharding=a_sharding)
          out_node["lora_b.kernel"] = jax.ShapeDtypeStruct(shape=b_shape, dtype=child.dtype, sharding=b_sharding)
        else:
          out_node[name] = None

  lora_abstract_params = {}
  add_lora(lora_abstract_params, base_abstract_params, "")

  unboxed_abstract_lora_state = train_state.TrainState(
      step=0, apply_fn=None, params=lora_abstract_params, tx=None, opt_state={}  # type: ignore
  )
  lora_state_mesh_annotations = train_state.TrainState(
      step=0,
      apply_fn=None,
      params=jax.tree_util.tree_map(
          lambda x: x.sharding.spec if x.sharding is not None else None,
          lora_abstract_params,
      ),
      tx=None,  # type: ignore
      opt_state={},
  )
  return unboxed_abstract_lora_state, lora_state_mesh_annotations
