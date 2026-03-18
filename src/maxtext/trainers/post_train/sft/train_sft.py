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
      "dropout": 0.0,
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


def _patch_qwix_dot_general_with_3d(qwix_flax_util, qwix_lora, qwix_ptq, mt_config=None):
  """Patches Qwix LoRA dot_general to support selected 3D-kernel paths."""

  original_dot_general = qwix_lora.LoraProvider.dot_general

  def _dot_general_with_3d(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision=None,
      preferred_element_type=None,
      out_sharding=None,
  ):
    if len(rhs.shape) <= 2:
      return original_dot_general(
          self, lhs, rhs, dimension_numbers, precision, preferred_element_type, out_sharding=out_sharding
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

    rule, _ = self._get_current_rule_and_op_id("dot_general", repeated_call=True)
    if not isinstance(rule, qwix_lora.LoraRule):
      return res

    weight_name = qwix_flax_util.find_param(rhs, qwix_lora.ptq.WithAux)
    if weight_name is None:
      return res

    contract_axes_lhs = tuple(dimension_numbers[0][0])
    contract_axes_rhs = tuple(dimension_numbers[0][1])
    batch_axes_lhs = tuple(dimension_numbers[1][0])
    batch_axes_rhs = tuple(dimension_numbers[1][1])
    if not contract_axes_rhs:
      return res

    batch_shape = tuple(rhs.shape[axis] for axis in batch_axes_rhs)
    contract_shape = tuple(rhs.shape[axis] for axis in contract_axes_rhs)
    k = math.prod(contract_shape)

    non_contracting_axes_rhs = tuple(i for i in range(rhs.ndim) if i not in contract_axes_rhs and i not in batch_axes_rhs)
    if not non_contracting_axes_rhs:
      return res
    out_shape = tuple(rhs.shape[axis] for axis in non_contracting_axes_rhs)
    out_dim = math.prod(out_shape)

    a_shape = batch_shape + (k, rule.rank)
    b_shape = batch_shape + (rule.rank, out_dim)
    a_sharding_transpose = batch_axes_rhs + (contract_axes_rhs[0], None)
    b_sharding_transpose = batch_axes_rhs + (None, non_contracting_axes_rhs[0])

    try:
      lora_a, lora_b = qwix_lora._get_or_create_lora_params(  # pylint: disable=protected-access
          name=weight_name,
          rule=rule,
          a_shape=a_shape,
          b_shape=b_shape,
          a_sharding_transpose=a_sharding_transpose,
          b_sharding_transpose=b_sharding_transpose,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      max_logging.log(f"LoRA param init failed for '{weight_name}': {exc}")
      return res

    lora_a = lora_a[...] if isinstance(lora_a, nnx.Variable) else lora_a
    lora_b = lora_b[...] if isinstance(lora_b, nnx.Variable) else lora_b

    lhs_lora = lhs
    if rule.dropout > 0:
      lhs_lora = nnx.Dropout(rule.dropout)(lhs_lora, rngs=qwix_flax_util.make_rng("dropout"))

    lora_a_reshaped = jnp.reshape(lora_a, batch_shape + contract_shape + (rule.rank,))
    lora_a_batch_axes = tuple(range(len(batch_shape)))
    lora_a_contract_axes = tuple(range(len(batch_shape), len(batch_shape) + len(contract_shape)))
    with jax.named_scope("lora_a"):
      delta_a = jax.lax.dot_general(
          lhs_lora,
          lora_a_reshaped,
          (((contract_axes_lhs), lora_a_contract_axes), ((batch_axes_lhs), lora_a_batch_axes)),
      )

    lora_b_reshaped = jnp.reshape(lora_b, batch_shape + (rule.rank,) + out_shape)
    delta_a_batch_axes = tuple(range(len(batch_shape)))
    with jax.named_scope("lora_b"):
      delta = jax.lax.dot_general(
          delta_a,
          lora_b_reshaped,
          (
              ((delta_a.ndim - 1,), (len(batch_shape),)),
              (delta_a_batch_axes, tuple(range(len(batch_shape)))),
          ),
      )

    if delta.shape != res.shape:
      delta = jnp.reshape(delta, res.shape)
    return res + delta * (rule.alpha / rule.rank)

  qwix_lora.LoraProvider.dot_general = _dot_general_with_3d


def _patch_qwix_update_boxed(qwix_flax_util):
  """Patches Qwix flax_util.update_boxed to handle PartitionSpec."""
  original_update_boxed = qwix_flax_util.update_boxed

  def patched_update_boxed(
      boxed,
      *,
      value=None,
      split=None,
      merge=None,
      transpose=None,
  ):
    import jax
    from flax import nnx

    if isinstance(boxed, nnx.Variable):
      if value is not None:
        boxed = boxed.replace(value)
      shape = boxed.shape
      metadata = boxed.get_metadata()
      sharding_key = "out_sharding" if "out_sharding" in metadata else "sharding_names"
      axes = metadata.get(sharding_key, None)
      if isinstance(axes, (list, tuple, jax.sharding.PartitionSpec)):
        updated_axes = qwix_flax_util.update_sharding(
            axes, shape=shape, split=split, merge=merge, transpose=transpose
        )
        if not isinstance(updated_axes, jax.sharding.PartitionSpec):
          updated_axes = jax.sharding.PartitionSpec(*updated_axes)

        # Avoid mutating metadata unless sharding actually changed.
        current_axes = axes if isinstance(axes, jax.sharding.PartitionSpec) else jax.sharding.PartitionSpec(*axes)
        if current_axes != updated_axes:
          boxed.set_metadata(sharding_key, updated_axes)
      return boxed
    return original_update_boxed(
        boxed, value=value, split=split, merge=merge, transpose=transpose
    )

  qwix_flax_util.update_boxed = patched_update_boxed


def _patch_qwix_lora_param_sharding(qwix_flax_util, qwix_lora, qwix_ptq):
  """Patches Qwix LoRA param init to inherit sharding from the target weight."""
  original_get_or_create_lora_params = qwix_lora._get_or_create_lora_params  # pylint: disable=protected-access

  def _get_canonical_named_sharding(maybe_boxed):
    value = qwix_flax_util.unbox(maybe_boxed)
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
      return None
    padded_pspec = sharding.spec + (None,) * (value.ndim - len(sharding.spec))
    return sharding.update(spec=padded_pspec)

  def _copy_sharding_to_lora_param(module, param_name, named_sharding):
    lora_param = getattr(module, param_name, None)
    if not isinstance(lora_param, nnx.Variable):
      return None

    lora_value = qwix_flax_util.unbox(lora_param)
    lora_value = jax.device_put(lora_value, named_sharding)
    lora_param = lora_param.replace(lora_value)

    metadata = lora_param.get_metadata()
    sharding_key = "out_sharding" if "out_sharding" in metadata else "sharding_names"
    lora_param.set_metadata(sharding_key, named_sharding.spec)
    setattr(module, param_name, lora_param)
    return qwix_flax_util.unbox(lora_param)

  def patched_get_or_create_lora_params(
      *,
      name,
      rule,
      a_shape,
      b_shape,
      a_sharding_transpose,
      b_sharding_transpose,
  ):
    module = None
    had_existing_lora_params = False
    try:
      module = qwix_flax_util.get_current_module()
      if isinstance(module, nnx.Module):
        had_existing_lora_params = isinstance(getattr(module, name + "_lora_a", None), nnx.Variable) and isinstance(
            getattr(module, name + "_lora_b", None), nnx.Variable
        )
    except Exception:  # pylint: disable=broad-exception-caught
      module = None

    lora_a, lora_b = original_get_or_create_lora_params(
        name=name,
        rule=rule,
        a_shape=a_shape,
        b_shape=b_shape,
        a_sharding_transpose=a_sharding_transpose,
        b_sharding_transpose=b_sharding_transpose,
    )

    # Avoid repeated Python-side setattr/device_put mutation inside forward
    # paths once LoRA params are already present.
    if had_existing_lora_params:
      return lora_a, lora_b

    try:
      if not isinstance(module, nnx.Module):
        return lora_a, lora_b

      if isinstance(qwix_flax_util.unbox(lora_a), jax.core.Tracer) or isinstance(qwix_flax_util.unbox(lora_b), jax.core.Tracer):
        return lora_a, lora_b

      target_param = getattr(module, name, None)
      if target_param is None:
        return lora_a, lora_b

      base_boxed = target_param.array.qvalue if isinstance(target_param, qwix_ptq.WithAux) else target_param
      base_sharding = _get_canonical_named_sharding(base_boxed)
      if base_sharding is None:
        return lora_a, lora_b

      lora_a_spec = qwix_flax_util.update_sharding(base_sharding.spec, transpose=a_sharding_transpose)
      lora_b_spec = qwix_flax_util.update_sharding(base_sharding.spec, transpose=b_sharding_transpose)

      lora_a_sharding = base_sharding.update(spec=lora_a_spec)
      lora_b_sharding = base_sharding.update(spec=lora_b_spec)

      updated_lora_a = _copy_sharding_to_lora_param(module, name + "_lora_a", lora_a_sharding)
      updated_lora_b = _copy_sharding_to_lora_param(module, name + "_lora_b", lora_b_sharding)
      if updated_lora_a is not None:
        lora_a = updated_lora_a
      if updated_lora_b is not None:
        lora_b = updated_lora_b
    except Exception as exc:  # pylint: disable=broad-exception-caught
      max_logging.log(f"LoRA sharding patch failed for '{name}': {exc}")

    return lora_a, lora_b

  qwix_lora._get_or_create_lora_params = patched_get_or_create_lora_params  # pylint: disable=protected-access

def _prepare_dummy_inputs():
  """Builds dummy decoder inputs used to materialize LoRA parameters."""
  # Keep LoRA warmup as small as possible to minimize compile/memory overhead.
  dummy_bs = 1
  seq_len = 1
  decoder_input_tokens = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  decoder_positions = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  return decoder_input_tokens, decoder_positions

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

  raise ValueError(
      "LoRA module path matched target modules, but nnx.LoRAParam is still "
      "missing. For Tunix PeftTrainer, LoRA params must be materialized before "
      "trainer initialization, otherwise it falls back to full-model training. "
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

  _validate_lora_config(mt_config)
  lora_provider = _build_lora_provider(mt_config, qwix)

  _patch_qwix_dot_general_with_3d(qwix_flax_util, qwix_lora, qwix_ptq, mt_config)
  _patch_qwix_update_boxed(qwix_flax_util)
  _patch_qwix_lora_param_sharding(qwix_flax_util, qwix_lora, qwix_ptq)

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
    from flax.linen import partitioning as nn_partitioning
    with mesh, nn_partitioning.axis_rules(mt_config.logical_axis_rules):
      graph_def, state = nnx.split(lora_model)
      default_memory_kind = jax.devices()[0].default_memory().kind
      dst_shardings = jax.tree_util.tree_map(
          lambda x: jax.sharding.NamedSharding(mesh, x, memory_kind=default_memory_kind) if x is not None else None,
          nnx.get_partition_spec(state),
      )
      lora_model = nnx.merge(graph_def, reshard.reshard_pytree(state, dst_shardings))

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


def _maybe_restore_lora_from_path(trainer, lora_restore_path):
  """Optionally restores LoRA params from an external checkpoint item path."""
  if not lora_restore_path:
    return trainer

  if getattr(trainer, "_train_steps", 0) > 0:
    max_logging.log(
        f"PeftTrainer restored current run at step {trainer._train_steps}; "
        f"ignoring lora_restore_path '{lora_restore_path}'."
    )
    return trainer

  if not tunix_sft_utils.is_lora_enabled(trainer.model):
    raise ValueError(
        "lora_restore_path is set but LoRA is not enabled on the model. "
        "Set enable_lora=True and verify lora_module_path matches model modules."
    )

  abstract_lora_params = nnx.state(trainer.model, nnx.LoRAParam)
  restored_lora_params = ocp.StandardCheckpointer().restore(
      lora_restore_path,
      target=abstract_lora_params,
  )
  nnx.update(trainer.model, restored_lora_params)
  max_logging.log(
      f"LoRA restore complete from '{lora_restore_path}'. "
      "Trainer step remains at 0 for this run."
  )
  return trainer

def setup_trainer_state(mt_config, goodput_recorder=None):
  """Set up prerequisites for training loop."""
  tunix_config = get_tunix_config(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    model, mesh = model_creation_utils.create_nnx_model(mt_config)
    model = maybe_apply_lora(model, mesh, mt_config)
    lora_restore_path = getattr(mt_config, "lora_restore_path", "")
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
    trainer = _maybe_restore_lora_from_path(trainer, lora_restore_path)

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
