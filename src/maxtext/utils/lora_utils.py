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

""" Common LoRA utils needed to support LoRA adapters."""
import inspect
import json
import os
from typing import Any, Optional

import omegaconf
import jax
import jax.numpy as jnp

from flax.training import train_state
from flax.linen import partitioning as nn_partitioning

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.utils import gcs_utils
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import max_logging

import math
import re

from flax import nnx
from orbax import checkpoint as ocp

import qwix
import qwix._src.flax_util as qwix_flax_util
import qwix._src.providers.lora as qwix_lora
import qwix._src.providers.ptq as qwix_ptq


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
  """
  Load the LoRA weights into a PyTree and return it.
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
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    lora_adapter_path: Path of the LoRA adapter which is expected to have
        `adapter_config.json` and adapter weights

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  lora_state = None
  lora_state_annotations = None
  lora_config = None

  if lora_adapter_path:
    max_logging.log(f"Setting initial state of LoRA with lora_adapter_path = {lora_adapter_path}")
    unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(model, tx, config, rng, mesh, True)

    lora_config_path = lora_adapter_path + "adapter_config.json"

    lora_config = gcs_utils.read_json_from_gcs(lora_config_path)

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

    if lora_module in ["self_attention.query", "self_attention.key", "self_attention.value"]:
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

    if lora_module in ["self_attention.query", "self_attention.key", "self_attention.value"]:
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
        add_lora_params(lora_params[name], f"{module_name}.{name}", param, lora_rank, lora_target_modules)
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
  """Gets the regex for modules to apply LoRA on based on the model name."""
  if mt_config.lora_module_path:
    return mt_config.lora_module_path

  config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "post_train", "lora_module_path.yml")
  lora_configs = omegaconf.OmegaConf.load(config_path)
  model_name = mt_config.model_name.lower()

  for key, module_path in lora_configs.items():
    if key != "default" and model_name.startswith(key):
      max_logging.log(f"Auto-detected lora_module_path for model '{model_name}': {module_path}")
      return str(module_path)

  default_path = lora_configs.get("default", "decoder/layers/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))")
  max_logging.log(
      f"Warning: Model '{model_name}' is not in the list of verified LoRA models. "
      "Auto-detection might not work. Please provide an explicit `lora_module_path` in your config if training fails."
  )
  max_logging.log(f"Falling back to default lora_module_path: {default_path}")
  return str(default_path)


def _build_lora_provider(mt_config: pyconfig.HyperParameters) -> qwix.LoraProvider:
  """Builds a Qwix LoRA provider from MaxText LoRA settings."""
  lora_module_path = _get_lora_module_path(mt_config)
  lora_kwargs = {
      "module_path": lora_module_path,
      "rank": mt_config.lora_rank,
      "alpha": mt_config.lora_alpha,
      "dropout": 0.0,
  }
  if mt_config.lora_tile_size is not None:
    lora_kwargs["tile_size"] = mt_config.lora_tile_size
  if mt_config.lora_weight_qtype is not None:
    lora_kwargs["weight_qtype"] = mt_config.lora_weight_qtype
    max_logging.log(
        f"QLoRA configured: module_path={lora_module_path} "
        f"rank={mt_config.lora_rank} alpha={mt_config.lora_alpha} "
        f"weight_qtype={mt_config.lora_weight_qtype} "
        f"tile_size={mt_config.lora_tile_size}"
    )
  else:
    max_logging.log(
        f"LoRA configured: module_path={lora_module_path} "
        f"rank={mt_config.lora_rank} alpha={mt_config.lora_alpha} "
        f"tile_size={mt_config.lora_tile_size}"
    )
  return qwix.LoraProvider(**lora_kwargs)


def _patch_nnx_decoder_apply_layers_sequentially(model: nnx.Module) -> None:
  """Patches the NNX decoder's _apply_layers_sequentially to include Qwix specific logic."""

  def _apply_layers_sequentially_with_qwix(self, layers, x_in, *args, length: int, **kwargs):
    """Runs the layer stack using nnx.scan with Qwix specific graph init and VJP downcasting."""
    policy = self.get_remat_policy()
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
    graphdef, params, state = nnx.split(
        layers, nnx.Param, ...
    )  # state: the mutable state we carry (KV cache, RNGs, etc.)

    scan_axis = self.config.param_scan_axis
    if scan_axis != 0:
      # Move scan_axis to 0 so scan can iterate over it
      params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)

    layer_cls = layers.__class__
    sig = inspect.signature(layer_cls.__call__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}
    # During Qwix init (disable_quant_stats_update=True), params may be lazily
    # created and the layer graphdef can grow. Allow graphdef refresh in that
    # phase only. Keep scanned training path static for remat purity.
    dynamic_graph_init = bool(getattr(self, "disable_quant_stats_update", False))
    updated_graphdef = [graphdef]

    def layer_fn(carry, scanned_vars):
      # Unpack the sliced variables for THIS layer
      current_params, current_state = scanned_vars

      if self.config.parameter_memory_host_offload:
        current_params = jax.tree.map(lambda x: jax.device_put(x, max_utils.device_space()), current_params)

      # Merge using the SLICED state
      layer = nnx.merge(graphdef, current_params, current_state)

      # Run the layer (Filter kwargs if using the solution from previous turn)
      layer_out = layer(carry, *args, **valid_kwargs)

      new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out

      # Qwix init: return updated params so graphdef can grow.
      # In normal training, keep params unchanged to avoid extra memory use.
      new_graphdef, updated_params, updated_state = nnx.split(layer, nnx.Param, ...)
      if dynamic_graph_init:
        updated_graphdef[0] = new_graphdef
        returned_params = updated_params
      else:
        returned_params = current_params
      return new_carry, (returned_params, updated_state)

    layer_fn_wrapped = jax.checkpoint(layer_fn, policy=policy, prevent_cse=prevent_cse)

    def _ensure_scan_leading_axis(x):
      # Promote scalars for scan axis compatibility.
      if not hasattr(x, "shape"):
        return x
      if len(x.shape) == 0:
        return jnp.broadcast_to(x, (length,))
      return x

    params = jax.tree.map(_ensure_scan_leading_axis, params)
    state = jax.tree.map(_ensure_scan_leading_axis, state)

    final_carry, (scanned_params, scanned_other) = jax.lax.scan(layer_fn_wrapped, x_in, (params, state))

    if scan_axis != 0:
      scanned_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), scanned_params)

    if dynamic_graph_init:
      return final_carry, nnx.merge(updated_graphdef[0], scanned_params, scanned_other)
    else:
      nnx.update(layers, nnx.State.merge(scanned_params, scanned_other))
      return final_carry, layers

  # IMPORTANT: Patch the class so nnx.merge doesn't lose the patch
  model.decoder.__class__._apply_layers_sequentially = _apply_layers_sequentially_with_qwix  # pylint: disable=protected-access


def _prepare_dummy_inputs() -> tuple[jnp.ndarray, jnp.ndarray]:
  """Builds dummy decoder inputs used to materialize LoRA parameters."""
  # Keep LoRA warmup as small as possible to minimize compile/memory overhead.
  dummy_bs = 1
  seq_len = 1
  decoder_input_tokens = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  decoder_positions = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  return decoder_input_tokens, decoder_positions


def _verify_lora_parameters(lora_model: nnx.Module, mt_config: pyconfig.HyperParameters) -> None:
  """Validates that LoRA is active or that target modules were matched."""
  from tunix.sft import utils as tunix_sft_utils  # pylint: disable=import-outside-toplevel

  if tunix_sft_utils.is_lora_enabled(lora_model):
    max_logging.log("LoRA verification: is_lora_enabled=True")
    return

  lora_module_path = _get_lora_module_path(mt_config)
  compiled_module_path = re.compile(lora_module_path)
  matched_module_paths = []
  sample_module_paths = []

  for path, _ in nnx.iter_modules(lora_model):
    module_path = "/".join(str(p) for p in path)
    if len(sample_module_paths) < 50:
      sample_module_paths.append(module_path)
    if compiled_module_path.search(module_path):
      matched_module_paths.append(module_path)

  if not matched_module_paths:
    max_logging.log(
        f"LoRA module_path='{lora_module_path}' did not match any weights. "
        f"Sample module paths: {sample_module_paths}"
    )
    raise ValueError("LoRA enabled but no LoRA parameters found in decoder/model state.")

  raise ValueError(
      "LoRA module path matched target modules, but nnx.LoRAParam is still "
      "missing. For Tunix PeftTrainer, LoRA params must be materialized before "
      "trainer initialization, otherwise it falls back to full-model training. "
      f"Sample matches: {matched_module_paths[:10]}"
  )


def apply_lora_to_model(
    model: nnx.Module, mesh: Optional[jax.sharding.Mesh], mt_config: pyconfig.HyperParameters
) -> nnx.Module:
  """Optionally applies LoRA/QLoRA to a MaxText model using Qwix."""
  # Skip Qwix LoRA if MaxText LoRA adapters are loaded
  if getattr(mt_config, "lora_input_adapters_path", None):
    max_logging.log("MaxText LoRA adapters loaded, skipping Qwix LoRA application")
    return model

  if not getattr(mt_config, "enable_lora", False):
    return model

  lora_provider = _build_lora_provider(mt_config)

  # Core Qwix patches are now integrated into Qwix HEAD.
  # We still patch the NNX decoder to handle Qwix dynamic graph init.
  _patch_nnx_decoder_apply_layers_sequentially(model)

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
      default_memory_kind = jax.devices()[0].default_memory().kind
      dst_shardings = jax.tree_util.tree_map(
          lambda x: jax.sharding.NamedSharding(mesh, x, memory_kind=default_memory_kind) if x is not None else None,
          nnx.get_partition_spec(state),
      )
      from tunix.rl import reshard  # pylint: disable=import-outside-toplevel
      state = reshard.reshard_pytree(state, dst_shardings)
      lora_model = nnx.merge(graph_def, state)

  # Warm up once outside jax.jit so any remaining lazy LoRA params are
  # materialized before train_step tracing.
  lora_model.set_attributes(disable_quant_stats_update=True, qwix_rngs=model_rngs)
  try:
    lora_model(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
    )
  finally:
    lora_model.set_attributes(disable_quant_stats_update=False, qwix_rngs=None)

  _verify_lora_parameters(lora_model, mt_config)

  return lora_model


def restore_lora_from_path(trainer: Any, mt_config: pyconfig.HyperParameters) -> Any:
  """Optionally restores LoRA params from an external checkpoint item path."""
  lora_restore_path = getattr(mt_config, "lora_restore_path", "")
  if not lora_restore_path:
    return trainer

  train_steps = getattr(trainer, "train_steps", 0)
  if train_steps > 0:
    max_logging.log(
        f"PeftTrainer restored current run at step {train_steps}; " f"ignoring lora_restore_path '{lora_restore_path}'."
    )
    return trainer

  from tunix.sft import utils as tunix_sft_utils  # pylint: disable=import-outside-toplevel

  if not tunix_sft_utils.is_lora_enabled(trainer.model):
    lora_module_path = _get_lora_module_path(mt_config)
    raise ValueError(
        "lora_restore_path is set but LoRA is not enabled on the model. "
        f"Set enable_lora=True and verify lora_module_path ('{lora_module_path}') matches model modules."
    )

  abstract_lora_params = nnx.state(trainer.model, nnx.LoRAParam)
  restored_lora_params = ocp.StandardCheckpointer().restore(
      lora_restore_path,
      target=abstract_lora_params,
  )
  nnx.update(trainer.model, restored_lora_params)
  max_logging.log(f"LoRA restore complete from '{lora_restore_path}'. " "Trainer step remains at 0 for this run.")
  return trainer
