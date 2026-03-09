#  Copyright 2023-2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
SFT training script that calls a trainer in Tunix to run SFT on a MaxText model
using `HuggingFaceH4/ultrachat_200k` dataset. The configurations for the dataset
are defined inside `src/MaxText/configs/sft.yml`.

Example command:
Training & Evaluation:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml \
    run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL_NAME?} load_parameters_path=${CHECKPOINT_PATH?} \
    hf_access_token=${HF_ACCESS_TOKEN?} tokenizer_path=${TOKENIZER_PATH?} \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=10 profiler=xplane weight_dtype=bfloat16

Training:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml \
    run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL_NAME?} load_parameters_path=${CHECKPOINT_PATH?} \
    hf_access_token=${HF_ACCESS_TOKEN?} tokenizer_path=${TOKENIZER_PATH?} \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=-1 steps=10 profiler=xplane weight_dtype=bfloat16
"""

from typing import Sequence

from absl import app
import math
import os
import re
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import pathwaysutils

from flax.linen import partitioning as nn_partitioning

from orbax import checkpoint as ocp

from tunix.sft import metrics_logger, peft_trainer, profiler
from tunix.sft import utils as tunix_sft_utils
from tunix.rl import reshard

from maxtext.optimizers import optimizers
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train.train import loss_fn
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.trainers.post_train.sft import hooks
from maxtext.utils import max_utils
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


def get_tunix_config(mt_config):
  """Gets the Tunix training configurations from the MaxText config.

  Args:
    mt_config: MaxText config.

  Returns:
    A Tunix `TrainingConfig` object.
  """
  # Checkpointing configurations
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=mt_config.checkpoint_period,
      enable_async_checkpointing=mt_config.async_checkpointing,
  )

  # Metrics configurations
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(log_dir=mt_config.tensorboard_dir)

  # Profiler configurations
  profiler_options = None
  if mt_config.profiler:
    set_profile_options = True
    platform_version = jax.extend.backend.get_backend().platform_version.strip()
    if platform_version.startswith("Pathways"):
      max_logging.log("Pathways backend detected. Disabling setting profile options.")
      set_profile_options = False
    profiler_options = profiler.ProfilerOptions(
        log_dir=mt_config.tensorboard_dir,
        skip_first_n_steps=mt_config.skip_first_n_steps_for_profiler,
        profiler_steps=mt_config.profiler_steps,
        set_profile_options=set_profile_options,
    )

  return peft_trainer.TrainingConfig(
      eval_every_n_steps=mt_config.eval_interval,
      max_steps=mt_config.steps,
      gradient_accumulation_steps=mt_config.gradient_accumulation_steps,
      checkpoint_root_directory=mt_config.checkpoint_dir,
      checkpointing_options=checkpointing_options,
      metrics_logging_options=metrics_logging_options,
      profiler_options=profiler_options,
  )


def use_maxtext_loss_function(trainer, mt_config):
  """Configures the trainer to use MaxText's loss function.

  This function creates a wrapper around MaxText's `loss_fn` to make it
  compatible with the Tunix trainer's expected loss function signature.

  Args:
    trainer: The PeftTrainer instance.
    mt_config: MaxText config.

  Returns:
    The trainer configured with the MaxText loss function.
  """

  def loss_func(
      model,
      inputs,
      inputs_position,
      inputs_segmentation,
      targets,
      targets_position,
      targets_segmentation,
  ):
    data = {
        "inputs": inputs,
        "inputs_position": inputs_position,
        "inputs_segmentation": inputs_segmentation,
        "targets": targets,
        "targets_position": targets_position,
        "targets_segmentation": targets_segmentation,
    }
    return loss_fn(model, mt_config, data, dropout_rng=None, params=None, is_train=True)

  trainer = trainer.with_loss_fn(loss_func, has_aux=True)
  return trainer


def _validate_lora_config(mt_config):
  """Validates required LoRA configuration fields."""
  if mt_config.lora_rank <= 0:
    raise ValueError("enable_lora is True but lora_rank is not set to a positive value.")
  if not mt_config.lora_module_path:
    raise ValueError("enable_lora is True but lora_module_path is empty.")


def _build_lora_provider(mt_config, qwix):
  """Builds a Qwix LoRA provider from MaxText LoRA settings."""
  lora_kwargs = {
      "module_path": mt_config.lora_module_path,
      "rank": mt_config.lora_rank,
      "alpha": mt_config.lora_alpha,
  }
  if mt_config.lora_tile_size is not None:
    lora_kwargs["tile_size"] = mt_config.lora_tile_size
  if mt_config.lora_weight_qtype is not None:
    lora_kwargs["weight_qtype"] = mt_config.lora_weight_qtype
    max_logging.log(
        f"QLoRA configured: module_path={mt_config.lora_module_path} "
        f"rank={mt_config.lora_rank} alpha={mt_config.lora_alpha} "
        f"weight_qtype={mt_config.lora_weight_qtype} "
        f"tile_size={mt_config.lora_tile_size}"
    )
  else:
    max_logging.log(
        f"LoRA configured: module_path={mt_config.lora_module_path} "
        f"rank={mt_config.lora_rank} alpha={mt_config.lora_alpha} "
        f"tile_size={mt_config.lora_tile_size}"
    )
  return qwix.LoraProvider(**lora_kwargs)


def _patch_qwix_dot_general_with_3d(lora_provider, qwix_flax_util, qwix_lora, qwix_ptq, types):
  """Patches Qwix LoRA dot_general to support selected 3D-kernel paths."""

  def _dot_general_with_3d(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision=None,
      preferred_element_type=None,
      out_sharding=None,
  ):
    def _fallback_dot_general():
      return qwix_lora.LoraProvider.dot_general(
          self,
          lhs,
          rhs,
          dimension_numbers,
          precision,
          preferred_element_type,
          out_sharding=out_sharding,
      )

    res = qwix_ptq.PtqProvider.dot_general(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type,
        out_sharding=out_sharding,
    )

    rule, _ = self._get_current_rule_and_op_id("dot_general", repeated_call=True)  # pylint: disable=protected-access
    if not isinstance(rule, qwix_lora.LoraRule):
      return res

    weight_name = qwix_flax_util.find_param(rhs, qwix_lora.ptq.WithAux)
    if weight_name is None:
      return res

    if len(rhs.shape) == 3 and tuple(dimension_numbers[0][1]) == (0,) and not dimension_numbers[1][1]:
      lora_params = qwix_lora._get_or_create_lora_params(  # pylint: disable=protected-access
          name=weight_name,
          rule=rule,
          a_shape=(rhs.shape[0], rule.rank),
          b_shape=(rule.rank, rhs.shape[1] * rhs.shape[2]),
          a_sharding_transpose=(0, None),
          b_sharding_transpose=(None, 1),
      )
      lora_a, lora_b = lora_params[:2]
      if rule.dropout > 0:
        lhs = nnx.Dropout(rule.dropout)(lhs, rngs=qwix_flax_util.make_rng("dropout"))
      lora_b = jnp.reshape(lora_b, (rule.rank, rhs.shape[1], rhs.shape[2]))
      delta = jnp.einsum("...k,kr->...r", lhs, lora_a)
      delta = jnp.einsum("...r,rnm->...nm", delta, lora_b)
      return res + delta * (rule.alpha / rule.rank)

    if len(rhs.shape) == 3 and tuple(dimension_numbers[0][1]) == (0, 1) and not dimension_numbers[1][1]:
      k = rhs.shape[0] * rhs.shape[1]
      lora_params = qwix_lora._get_or_create_lora_params(  # pylint: disable=protected-access
          name=weight_name,
          rule=rule,
          a_shape=(k, rule.rank),
          b_shape=(rule.rank, rhs.shape[2]),
          a_sharding_transpose=(0, None),
          b_sharding_transpose=(None, 1),
      )
      lora_a, lora_b = lora_params[:2]
      if rule.dropout > 0:
        lhs = nnx.Dropout(rule.dropout)(lhs, rngs=qwix_flax_util.make_rng("dropout"))
      contract_axes = tuple(dimension_numbers[0][0])
      lhs_perm = [i for i in range(lhs.ndim) if i not in contract_axes] + list(contract_axes)
      lhs_trans = jnp.transpose(lhs, lhs_perm)
      lhs_shape = lhs_trans.shape
      lhs_flat = jnp.reshape(lhs_trans, lhs_shape[: -len(contract_axes)] + (k,))
      if lora_a.shape[0] != k:
        return _fallback_dot_general()
      delta = jnp.einsum("...k,kr->...r", lhs_flat, lora_a)
      delta = jnp.einsum("...r,rm->...m", delta, lora_b)
      return res + delta * (rule.alpha / rule.rank)

    return _fallback_dot_general()

  lora_provider.dot_general = types.MethodType(_dot_general_with_3d, lora_provider)

def _prepare_dummy_inputs(mt_config, mesh):
  """Builds dummy decoder inputs used to materialize LoRA parameters."""
  batch_size = getattr(mt_config, "per_device_batch_size", 1)
  seq_len = getattr(mt_config, "max_target_length", 1)
  if batch_size <= 0 or seq_len <= 0:
    raise ValueError("per_device_batch_size and max_target_length must be positive when LoRA is enabled.")

  devices_data_fsdp = 1
  if mesh is not None:
    devices_data_fsdp = mesh.shape.get("data", 1) * mesh.shape.get("fsdp", 1)

  dummy_bs = (max(batch_size, devices_data_fsdp) + devices_data_fsdp - 1) // devices_data_fsdp
  dummy_bs *= devices_data_fsdp

  decoder_input_tokens = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  decoder_positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (dummy_bs, seq_len))
  return decoder_input_tokens, decoder_positions


def _precreate_lora_params(lora_model, lora_provider, mt_config, qwix_flax_util, qwix_lora, types):
  """Pre-creates LoRA parameter tensors for modules matching the target regex."""
  rules = [rule for rule in getattr(lora_provider, "_rules", []) if isinstance(rule, qwix_lora.LoraRule)]
  if not rules:
    max_logging.log("LoRA precreate: no LoRA rules found on provider, skipping.")
    return

  # MaxText uses a single LoRA rule from the provided module_path regex.
  rule = rules[0]
  compiled_module_path = re.compile(mt_config.lora_module_path)
  num_decoder_layers = getattr(mt_config, "num_decoder_layers", None)
  if num_decoder_layers is None:
    num_decoder_layers = getattr(mt_config, "base_num_decoder_layers", None)
  param_scan_axis = int(getattr(mt_config, "param_scan_axis", 0))

  def _with_layer_axis(base_shape_or_transpose, layer_value):
    axis = max(0, min(param_scan_axis, len(base_shape_or_transpose)))
    values = list(base_shape_or_transpose)
    values.insert(axis, layer_value)
    return tuple(values)

  matched_modules = 0
  precreated_modules = 0
  skipped_modules = []
  precreated_shapes = []

  for path, module in nnx.iter_modules(lora_model):
    module_path = "/".join(str(p) for p in path)
    if not compiled_module_path.search(module_path):
      continue

    matched_modules += 1
    kernel = getattr(module, "kernel", None)
    if kernel is None:
      if len(skipped_modules) < 10:
        skipped_modules.append(f"{module_path}: no kernel")
      continue

    try:
      kernel_value = qwix_flax_util.unbox(kernel)
    except Exception:  # pylint: disable=broad-exception-caught
      if len(skipped_modules) < 10:
        skipped_modules.append(f"{module_path}: cannot unbox kernel")
      continue

    kernel_shape = getattr(kernel_value, "shape", None)
    if kernel_shape is None and hasattr(kernel_value, "array"):
      kernel_shape = getattr(kernel_value.array, "shape", None)
      if kernel_shape is None and hasattr(kernel_value.array, "qvalue"):
        kernel_shape = getattr(kernel_value.array.qvalue, "shape", None)
    if kernel_shape is None or len(kernel_shape) < 2:
      if len(skipped_modules) < 10:
        skipped_modules.append(f"{module_path}: unsupported kernel shape {kernel_shape}")
      continue

    is_scanned_decoder_module = (
        "decoder/layers/" in module_path and isinstance(num_decoder_layers, int) and num_decoder_layers > 1
    )

    if is_scanned_decoder_module:
      layer_axis = None
      if 0 <= param_scan_axis < len(kernel_shape):
        layer_axis = int(param_scan_axis)
      elif len(kernel_shape) > 1 and int(kernel_shape[1]) == int(num_decoder_layers):
        layer_axis = 1
      else:
        for axis, dim in enumerate(kernel_shape):
          if int(dim) == int(num_decoder_layers):
            layer_axis = axis
            break
      if layer_axis is None:
        if len(skipped_modules) < 10:
          skipped_modules.append(f"{module_path}: unable to infer layer axis from kernel shape {kernel_shape}")
        continue

      effective_shape = tuple(int(dim) for i, dim in enumerate(kernel_shape) if i != layer_axis)
      if len(effective_shape) < 2:
        if len(skipped_modules) < 10:
          skipped_modules.append(f"{module_path}: unsupported effective shape {effective_shape}")
        continue

      if "decoder/layers/self_attention/out" in module_path and len(effective_shape) >= 3:
        in_dim = int(math.prod(effective_shape[:-1]))
        out_dim = int(effective_shape[-1])
      else:
        in_dim = int(effective_shape[0])
        out_dim = int(math.prod(effective_shape[1:]))
      if in_dim <= 0 or out_dim <= 0:
        if len(skipped_modules) < 10:
          skipped_modules.append(f"{module_path}: non-positive dims in={in_dim} out={out_dim}")
        continue

      in_axis = next(i for i in range(len(kernel_shape)) if i != layer_axis)
      out_axis = next(i for i in range(len(kernel_shape) - 1, -1, -1) if i != layer_axis)

      a_shape = _with_layer_axis((in_dim, rule.rank), num_decoder_layers)
      b_shape = _with_layer_axis((rule.rank, out_dim), num_decoder_layers)
      a_sharding_transpose = _with_layer_axis((in_axis, None), layer_axis)
      b_sharding_transpose = _with_layer_axis((None, out_axis), layer_axis)
    else:
      prefix_shape = tuple(kernel_shape[:-2])
      in_dim = int(kernel_shape[-2])
      out_dim = int(kernel_shape[-1])
      if in_dim <= 0 or out_dim <= 0:
        if len(skipped_modules) < 10:
          skipped_modules.append(f"{module_path}: non-positive dims in={in_dim} out={out_dim}")
        continue

      full_prefix_shape = prefix_shape
      a_shape = full_prefix_shape + (in_dim, rule.rank)
      b_shape = full_prefix_shape + (rule.rank, out_dim)

      prefix_rank = len(full_prefix_shape)
      a_sharding_transpose = tuple(range(prefix_rank)) + (prefix_rank, None)
      b_sharding_transpose = tuple(range(prefix_rank)) + (None, prefix_rank + 1)

    def _init_for_module(
        self,  # pylint: disable=unused-argument
        a_shape=a_shape,
        b_shape=b_shape,
        a_sharding_transpose=a_sharding_transpose,
        b_sharding_transpose=b_sharding_transpose,
    ):
      qwix_lora._get_or_create_lora_params(  # pylint: disable=protected-access
          name="kernel",
          rule=rule,
          a_shape=a_shape,
          b_shape=b_shape,
          a_sharding_transpose=a_sharding_transpose,
          b_sharding_transpose=b_sharding_transpose,
      )

    types.MethodType(_init_for_module, module)()
    precreated_modules += 1
    if len(precreated_shapes) < 10:
      precreated_shapes.append((module_path, a_shape, b_shape))

  max_logging.log(
      f"LoRA precreate: matched_modules={matched_modules} "
      f"precreated_modules={precreated_modules} "
      f"skipped_sample={skipped_modules} "
      f"shape_sample={precreated_shapes}"
  )


def _verify_lora_parameters(lora_model, mt_config):
  """Validates that LoRA is active or that target modules were matched."""
  compiled_module_path = re.compile(mt_config.lora_module_path)
  matched_module_paths = []
  sample_module_paths = []

  for path, _ in nnx.iter_modules(lora_model):
    module_path = "/".join(str(p) for p in path)
    if len(sample_module_paths) < 50:
      sample_module_paths.append(module_path)
    if compiled_module_path.search(module_path):
      matched_module_paths.append(module_path)

  is_lora_enabled = tunix_sft_utils.is_lora_enabled(lora_model)
  if is_lora_enabled:
    max_logging.log("LoRA verification: tunix_sft_utils.is_lora_enabled=True")
    return

  if not matched_module_paths:
    max_logging.log(
        f"LoRA module_path='{mt_config.lora_module_path}' did not match any weights. "
        f"Sample module paths: {sample_module_paths}"
    )
    raise ValueError("LoRA enabled but no LoRA parameters found in decoder/model state.")

  max_logging.log(
      f"LoRA verification: matched {len(matched_module_paths)} target modules but "
      "LoRA params are not yet materialized; continuing with lazy LoRA initialization. "
      f"Sample matches: {matched_module_paths[:10]}"
  )


def maybe_apply_lora(model, mesh, mt_config):
  """Optionally applies LoRA/QLoRA to a MaxText model using Qwix."""
  # Skip Qwix LoRA if MaxText LoRA adapters are loaded
  if hasattr(mt_config, "lora_input_adapters_path") and mt_config.lora_input_adapters_path:
    max_logging.log("MaxText LoRA adapters loaded, skipping Qwix LoRA application")
    return model

  if not getattr(mt_config, "enable_lora", False):
    return model

  import qwix  # pylint: disable=import-outside-toplevel
  import qwix._src.flax_util as qwix_flax_util  # pylint: disable=import-outside-toplevel
  import qwix._src.providers.lora as qwix_lora  # pylint: disable=import-outside-toplevel
  import qwix._src.providers.ptq as qwix_ptq  # pylint: disable=import-outside-toplevel
  import types  # pylint: disable=import-outside-toplevel

  _validate_lora_config(mt_config)
  lora_provider = _build_lora_provider(mt_config, qwix)

  _patch_qwix_dot_general_with_3d(lora_provider, qwix_flax_util, qwix_lora, qwix_ptq, types)

  decoder_input_tokens, decoder_positions = _prepare_dummy_inputs(mt_config, mesh)
  lora_model = qwix.apply_lora_to_model(
      model,
      lora_provider,
      decoder_input_tokens=decoder_input_tokens,
      decoder_positions=decoder_positions,
      skip_nnx_init=True,
  )

  # Materialize LoRA parameters. Qwix 0.1.5+ unsets RNGs after apply_lora_to_model,
  lora_model.set_attributes(qwix_rngs=nnx.Rngs(10003))
  _precreate_lora_params(lora_model, lora_provider, mt_config, qwix_flax_util, qwix_lora, types)
  lora_model.set_attributes(qwix_rngs=None)

  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)

  _verify_lora_parameters(lora_model, mt_config)
  return lora_model


def maybe_restore_lora_from_path(model, mt_config, mesh=None):
  """Optionally restores LoRA params from a dedicated adapter checkpoint path.

  If `lora_restore_path` is set and LoRA params have not yet been materialized on
  the model, this function attempts to apply LoRA first (when enabled) before
  restoring adapter weights.
  """
  lora_restore_path = getattr(mt_config, "lora_restore_path", "")
  if not lora_restore_path:
    return model

  if not tunix_sft_utils.is_lora_enabled(model):
    if getattr(mt_config, "enable_lora", False):
      max_logging.log("lora_restore_path is set but model has no LoRA params yet; " "applying LoRA before restore.")
      model = maybe_apply_lora(model, mesh, mt_config)

    if not tunix_sft_utils.is_lora_enabled(model):
      raise ValueError(
          "lora_restore_path is set but LoRA is not enabled on the model. "
          "Set enable_lora=True and verify lora_module_path matches model modules."
      )

  if not os.path.exists(lora_restore_path):
    raise ValueError(f"lora_restore_path does not exist: {lora_restore_path}")

  max_logging.log(f"Restoring LoRA params from: {lora_restore_path}")

  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=mt_config.checkpoint_storage_concurrent_gb,
          save_concurrent_gb=mt_config.checkpoint_storage_concurrent_gb,
          use_ocdbt=mt_config.checkpoint_storage_use_ocdbt,
          use_zarr3=mt_config.checkpoint_storage_use_zarr3,
      )
  )

  lora_state = nnx.state(model, nnx.LoRAParam)
  metadata = ckptr.metadata(lora_restore_path)

  # Restore is target-driven from the currently materialized `lora_state`.
  # Checkpoint adapter paths that do not match these LoRA params are not
  # remapped automatically by Orbax during restore.

  # LoRA restore path is NNX-only.
  if "params" in metadata.item_metadata.tree.keys() and "params" in metadata.item_metadata.tree.get("params", {}).keys():
    raise ValueError("lora_restore_path must point to an NNX LoRA checkpoint (not Linen format).")

  target_for_restore = jax.tree.map(
      lambda v: {"value": v.value},
      lora_state,
      is_leaf=lambda n: isinstance(n, nnx.Variable),
  )
  item_to_restore = target_for_restore
  restore_args = ocp.checkpoint_utils.construct_restore_args(target_for_restore)

  restored = ckptr.restore(
      lora_restore_path,
      item=item_to_restore,
      transforms={},
      restore_args=restore_args,
  )

  restored_lora = jax.tree.map(
      lambda v: v["value"],
      restored,
      is_leaf=lambda x: isinstance(x, dict) and "value" in x and not isinstance(x.get("value"), dict),
  )

  if restored_lora:
    nnx.update(model, restored_lora)
    max_logging.log("LoRA restore complete.")

  return model


def setup_trainer_state(mt_config, goodput_recorder=None):
  """Set up prerequisites for training loop."""
  tunix_config = get_tunix_config(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    model, mesh = model_creation_utils.create_nnx_model(mt_config)
    model = maybe_apply_lora(model, mesh, mt_config)
    model = maybe_restore_lora_from_path(model, mt_config, mesh)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    # pass in model for muon
    optimizer = optimizers.get_optimizer(mt_config, learning_rate_schedule, model)

    if mt_config.gradient_clipping_threshold > 0:
      optimizer = optax.chain(
          optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold),
          optimizer,
      )

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    training_hooks = hooks.SFTTrainingHooks(mt_config, mesh, learning_rate_schedule, goodput_recorder)
    data_hooks = hooks.SFTDataHooks(mt_config, mesh, goodput_recorder)

    trainer = peft_trainer.PeftTrainer(model, optimizer, tunix_config)
    trainer.with_training_hooks(training_hooks)
    trainer.with_data_hooks(data_hooks)
    trainer = use_maxtext_loss_function(trainer, mt_config)

  return trainer, mesh


def train_model(mt_config, trainer, mesh):
  """Runs the SFT training loop in Tunix."""
  with mesh, nn_partitioning.axis_rules(mt_config.logical_axis_rules):
    trainer.train(
        trainer.data_hooks.train_data_iterator,
        trainer.data_hooks.eval_data_iterator,
    )
  return trainer


def train(mt_config, goodput_recorder=None):
  """Main method for SFT training.

  Args:
    mt_config: MaxText config.
    goodput_recorder: An optional GoodputRecorder to record performance metrics.
  """
  trainer, mesh = setup_trainer_state(mt_config, goodput_recorder)
  _job_completed_gracefully = False
  try:
    trainer = train_model(mt_config, trainer, mesh)
    _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(goodput_recorder, RECORD_JOB_END_TIME)
  return trainer, mesh


def main(argv: Sequence[str]) -> None:
  """Main function to run SFT training.

  Args:
    argv: Command-line arguments.
  """
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  mt_config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  goodput_recorder = create_goodput_recorder(mt_config)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(mt_config):
    train(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
