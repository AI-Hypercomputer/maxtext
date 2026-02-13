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
  python3 -m maxtext.trainers.post_train.sft.train_sft src/MaxText/configs/sft.yml \
    run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY \
    model_name=$MODEL_NAME load_parameters_path=$CHECKPOINT_PATH \
    hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=10 profiler=xplane weight_dtype=bfloat16

Training:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/MaxText/configs/sft.yml \
    run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY \
    model_name=$MODEL_NAME load_parameters_path=$CHECKPOINT_PATH \
    hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=-1 steps=10 profiler=xplane weight_dtype=bfloat16
"""

from typing import Sequence

from absl import app
import os
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

from MaxText import optimizers
from MaxText import pyconfig
from MaxText.train import loss_fn
from maxtext.common.goodput import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
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

  def loss_func(model, inputs, inputs_position, inputs_segmentation, targets, targets_position, targets_segmentation):
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
  if mt_config.lora_rank <= 0:
    raise ValueError("enable_lora is True but lora_rank is not set to a positive value.")
  if not mt_config.lora_module_path:
    raise ValueError("enable_lora is True but lora_module_path is empty.")


def _build_lora_provider(mt_config, qwix):
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
      "QLoRA configured: module_path=%s rank=%s alpha=%s weight_qtype=%s tile_size=%s"
      % (
        mt_config.lora_module_path,
        mt_config.lora_rank,
        mt_config.lora_alpha,
        mt_config.lora_weight_qtype,
        mt_config.lora_tile_size,
      )
    )
  else:
    max_logging.log(
      "LoRA configured: module_path=%s rank=%s alpha=%s tile_size=%s"
      % (
        mt_config.lora_module_path,
        mt_config.lora_rank,
        mt_config.lora_alpha,
        mt_config.lora_tile_size,
      )
    )
  return qwix.LoraProvider(**lora_kwargs)


def _patch_qwix_dot_general_with_3d(lora_provider, qwix_flax_util, qwix_lora, qwix_ptq, types):
  def _dot_general_with_3d(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision=None,
      preferred_element_type=None,
      out_sharding=None,
  ):
    res = qwix_ptq.PtqProvider.dot_general(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type,
        out_sharding=out_sharding,
    )

    rule, _ = self._get_current_rule_and_op_id("dot_general", repeated_call=True)
    if not isinstance(rule, qwix_lora.LoraRule):
      return res

    weight_name = qwix_flax_util.find_param(rhs, qwix_lora.ptq.WithAux)
    if weight_name is None:
      return res

    if (
        len(rhs.shape) == 3
        and tuple(dimension_numbers[0][1]) == (0,)
        and not dimension_numbers[1][1]
    ):
      lora_params = qwix_lora._get_or_create_lora_params(
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

    if (
        len(rhs.shape) == 3
        and tuple(dimension_numbers[0][1]) == (0, 1)
        and not dimension_numbers[1][1]
    ):
      k = rhs.shape[0] * rhs.shape[1]
      lora_params = qwix_lora._get_or_create_lora_params(
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
      lhs_flat = jnp.reshape(lhs_trans, lhs_shape[:-len(contract_axes)] + (k,))
      delta = jnp.einsum("...k,kr->...r", lhs_flat, lora_a)
      delta = jnp.einsum("...r,rm->...m", delta, lora_b)
      return res + delta * (rule.alpha / rule.rank)

    return qwix_lora.LoraProvider.dot_general(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type,
        out_sharding=out_sharding,
    )

  lora_provider.dot_general = types.MethodType(_dot_general_with_3d, lora_provider)


def _patch_qwix_find_param(qwix_flax_util):
  if getattr(qwix_flax_util, "_maxtext_find_param_patched", False):
    return

  original_find_param = qwix_flax_util.find_param

  def _safe_find_param(x, ptq_array_type=None):
    module = qwix_flax_util.get_current_module()
    candidates = {}

    # 1) Pure NNX: scan attributes for nnx.Params / ptq arrays.
    if isinstance(module, nnx.Module):
      array_types = (nnx.Param,) if ptq_array_type is None else (nnx.Param, ptq_array_type)
      for name, node in module.__dict__.items():
        if isinstance(node, array_types):
          value = getattr(node, "value", None)
          if value is None:
            try:
              value = qwix_flax_util.unbox(node)
            except Exception:
              continue
          candidates[name] = value

      

    else:
      return original_find_param(x, ptq_array_type)

    candidates_by_id = {id(c): n for n, c in candidates.items()}

    if id(x) in candidates_by_id:
      return candidates_by_id[id(x)]

    if isinstance(x, jax.core.Tracer) and hasattr(x, "parent"):
      while True:
        if id(x) in candidates_by_id:
          return candidates_by_id[id(x)]
        if x.parent and len(x.parent.in_tracers) == 1:
          x = x.parent.in_tracers[0]
        elif id(const := x.get_const()) in candidates_by_id:
          return candidates_by_id[id(const)]
        else:
          return None

    if not hasattr(x, "shape"):
      return None
    candidates = {n: c for n, c in candidates.items() if getattr(c, "shape", None) == x.shape}
    if len(candidates) > 2:
      raise ValueError(f"Multiple candidate params found: {candidates.keys()}")
    if len(candidates) == 1:
      return list(candidates.keys())[0]

    return None

  qwix_flax_util.find_param = _safe_find_param
  qwix_flax_util._maxtext_find_param_patched = True


def _patch_with_sharding_constraint():
  if getattr(jax.lax, "_maxtext_with_sharding_constraint_patched", False):
    return

  jax.lax._original_with_sharding_constraint = jax.lax.with_sharding_constraint

  def _safe_with_sharding_constraint(x, sharding, *args, **kwargs):
    def _safe_leaf_fn(x_leaf, s_leaf):
      try:
        spec = getattr(s_leaf, "spec", s_leaf)
        if hasattr(spec, "__len__"):
          ndim = getattr(x_leaf, "ndim", None)
          if ndim is not None and len(spec) > ndim:
            return x_leaf
      except Exception:
        pass
      return jax.lax._original_with_sharding_constraint(x_leaf, s_leaf, *args, **kwargs)

    return jax.tree_util.tree_map(_safe_leaf_fn, x, sharding)

  jax.lax.with_sharding_constraint = _safe_with_sharding_constraint
  jax.lax._maxtext_with_sharding_constraint_patched = True


def _prepare_dummy_inputs(mt_config, mesh):
  batch_size = getattr(mt_config, "per_device_batch_size", 1)
  seq_len = getattr(mt_config, "max_target_length", 1)
  if batch_size <= 0 or seq_len <= 0:
    raise ValueError(
        "per_device_batch_size and max_target_length must be positive when LoRA is enabled."
    )

  devices_data_fsdp = 1
  if mesh is not None:
    devices_data_fsdp = mesh.shape.get("data", 1) * mesh.shape.get("fsdp", 1)

  dummy_bs = (max(batch_size, devices_data_fsdp) + devices_data_fsdp - 1) // devices_data_fsdp
  dummy_bs *= devices_data_fsdp

  decoder_input_tokens = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  decoder_positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (dummy_bs, seq_len))
  return decoder_input_tokens, decoder_positions

def _precreate_lora_params(target_model, lora_provider, qwix_flax_util, qwix_lora, math, re, types):
  rules = getattr(lora_provider, "_rules", [])
  if not rules:
    return

  for path, module in target_model.iter_modules():
    module_path = "/".join(map(str, path))
    
    for rule in rules:
      if rule.op_names and "dot_general" not in rule.op_names:
        continue

      kernel_tensor = None
      in_rank = 0
      out_rank = 0
      found_param_name = "kernel"

      # Case A: Pure NNX (Standard attributes)
      if hasattr(module, "kernel") and hasattr(module, "in_features_shape"):
        try:
          kernel_tensor = qwix_flax_util.unbox(module.kernel)
          in_rank = len(module.in_features_shape)
          out_rank = len(module.out_features_shape)
        except Exception:
          kernel_tensor = None

      

      if kernel_tensor is None:
        continue

      # Closure to define LoRA A and B shapes and sharding
      def _init_for_module(self, k_tensor=kernel_tensor, i_r=in_rank, o_r=out_rank, p_name=found_param_name):
        kernel_shape = getattr(k_tensor, "shape", ())
        extra_rank = max(0, len(kernel_shape) - (i_r + o_r))
        
        prefix_shape = kernel_shape[:extra_rank]
        in_shape = kernel_shape[extra_rank : extra_rank + i_r]
        out_shape = kernel_shape[extra_rank + i_r :]

        in_size = int(math.prod(in_shape))
        out_size = int(math.prod(out_shape))
        
        if in_size <= 0 or out_size <= 0:
          return

        a_shape = prefix_shape + (in_size, rule.rank)
        b_shape = prefix_shape + (rule.rank, out_size)
        
        prefix_axes = tuple(range(extra_rank))
        a_sharding_transpose = prefix_axes + (None,)
        b_sharding_transpose = prefix_axes + (None,)

        qwix_lora._get_or_create_lora_params(
            name=p_name,
            rule=rule,
            a_shape=a_shape,
            b_shape=b_shape,
            a_sharding_transpose=a_sharding_transpose,
            b_sharding_transpose=b_sharding_transpose,
        )

      types.MethodType(_init_for_module, module)()


def _verify_lora_parameters(lora_model, mt_config):
  is_lora_enabled = tunix_sft_utils.is_lora_enabled(lora_model)
  if not is_lora_enabled:
    module_paths = []
    for path, _ in lora_model.iter_modules():
      module_paths.append("/".join(str(p) for p in path))
      if len(module_paths) >= 50:
        break
    max_logging.log(
      f"LoRA module_path='{mt_config.lora_module_path}' did not match any weights. "
      f"Sample module paths: {module_paths}"
    )
    raise ValueError("LoRA enabled but no LoRA parameters found in decoder/model state.")
  max_logging.log("LoRA verification: tunix_sft_utils.is_lora_enabled=True")


def maybe_apply_lora(model, mesh, mt_config):
  """Optionally applies LoRA/QLoRA to a MaxText model using Qwix."""
  # Skip Qwix LoRA if MaxText LoRA adapters are loaded
  if hasattr(mt_config, 'lora_input_adapters_path') and mt_config.lora_input_adapters_path:
    max_logging.log("MaxText LoRA adapters loaded, skipping Qwix LoRA application")
    return model
    
  if not getattr(mt_config, "enable_lora", False):
    return model

  import qwix
  import math
  import re
  import qwix._src.flax_util as qwix_flax_util
  import qwix._src.providers.lora as qwix_lora
  import qwix._src.providers.ptq as qwix_ptq
  import types

  _validate_lora_config(mt_config)
  lora_provider = _build_lora_provider(mt_config, qwix)

  _patch_qwix_dot_general_with_3d(lora_provider, qwix_flax_util, qwix_lora, qwix_ptq, types)
  _patch_qwix_find_param(qwix_flax_util)
  _patch_with_sharding_constraint()

  decoder_input_tokens, decoder_positions = _prepare_dummy_inputs(mt_config, mesh)
  lora_model = qwix.apply_lora_to_model(
    model,
    lora_provider,
    decoder_input_tokens=decoder_input_tokens,
    decoder_positions=decoder_positions,
    skip_nnx_init=True,
  )

  _precreate_lora_params(lora_model, lora_provider, qwix_flax_util, qwix_lora, math, re, types)

  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)

  _verify_lora_parameters(lora_model, mt_config)
  return lora_model


def setup_trainer_state(mt_config, goodput_recorder=None):
  """Set up prerequisites for training loop."""
  tunix_config = get_tunix_config(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    model, mesh = model_creation_utils.create_nnx_model(mt_config)
    model = maybe_apply_lora(model, mesh, mt_config)
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
    trainer.train(trainer.data_hooks.train_data_iterator, trainer.data_hooks.eval_data_iterator)
  return trainer


def train(mt_config, goodput_recorder=None):
  """Main method for SFT training.

  Args:
    mt_config: MaxText config.
    goodput_recorder: An optional GoodputRecorder to record performance metrics.
  """
  trainer, mesh = setup_trainer_state(mt_config, goodput_recorder)
  trainer = train_model(mt_config, trainer, mesh)
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

  with maybe_record_goodput(goodput_recorder, GoodputEvent.JOB), maybe_monitor_goodput(mt_config):
    train(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
