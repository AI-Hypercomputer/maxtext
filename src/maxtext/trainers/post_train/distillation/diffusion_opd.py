# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaxText model-aware preparation for Tunix diffusion OPD."""

import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.diffusion import denoise
from maxtext.diffusion import scoring
from maxtext.trainers.post_train.distillation import distillation_utils
from maxtext.utils import max_logging
from tunix.diffusion import types as diffusion_types
from tunix.distillation import diffusion as diffusion_distillation
from tunix.distillation import diffusion_opd as tunix_diffusion_opd
from tunix.sft import peft_trainer


def _resolve_logit_alignment(config) -> str:
  if getattr(config, "training_objective", "causal_lm") == "block_diffusion":
    return config.block_diffusion_logit_alignment
  return "shifted"


def create_target_aligned_logits_fn(config, *, enable_dropout: bool):
  """Builds a MaxText forward adapter satisfying Tunix's logits contract."""
  alignment = _resolve_logit_alignment(config)

  def logits_fn(model, model_inputs):
    logits = model(
        decoder_input_tokens=model_inputs["input_tokens"],
        decoder_positions=model_inputs["positions"],
        decoder_segment_ids=model_inputs["decoder_segment_ids"],
        enable_dropout=enable_dropout,
        decoder_target_tokens=model_inputs["targets"],
        decoder_target_mask=model_inputs["targets_segmentation"],
    )
    return scoring.align_logits_to_targets(
        logits,
        alignment,
        model_inputs["positions"],
        model_inputs["targets_segmentation"] != 0,
    )

  return logits_fn


def create_rollout_fn(config):
  """Builds one compiled student rollout with deterministic dropout settings."""
  logits_fn = create_target_aligned_logits_fn(config, enable_dropout=False)
  max_denoise_steps = config.distill_rollout_max_denoise_steps
  if max_denoise_steps == -1:
    max_denoise_steps = config.block_diffusion_block_size

  @nnx.jit
  def rollout(model, initial_tokens, positions, decoder_segment_ids, completion_mask):
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    validity_mask = decoder_segment_ids != 0

    def canvas_logits(canvas):
      # The denoising state machine uses nested JAX control flow. Reconstructing
      # a local model at that trace level avoids mutating outer NNX Variables;
      # rollout uses dropout=False, so its transient non-parameter state is
      # intentionally discarded after each proposal forward.
      local_model = nnx.merge(graphdef, params, rest, copy=True)
      return logits_fn(
          local_model,
          {
              "input_tokens": canvas,
              "positions": positions,
              "decoder_segment_ids": decoder_segment_ids,
              "targets": canvas,
              "targets_segmentation": decoder_segment_ids,
          },
      )

    return denoise.low_confidence_generate(
        canvas_logits,
        initial_tokens,
        positions,
        validity_mask,
        completion_mask,
        block_size=config.block_diffusion_block_size,
        mask_id=config.block_diffusion_mask_id,
        logit_alignment=config.block_diffusion_logit_alignment,
        canvas_policy=config.block_diffusion_canvas_policy,
        confidence_threshold=config.distill_rollout_confidence_threshold,
        temperature=config.distill_rollout_temperature,
        max_denoise_steps=max_denoise_steps,
    )

  return rollout


def create_teacher_score_fn(config):
  """Builds a compiled frozen-teacher scoring function."""
  logits_fn = create_target_aligned_logits_fn(config, enable_dropout=False)

  @nnx.jit
  def score(model, model_inputs):
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    local_model = nnx.merge(graphdef, params, rest, copy=True)
    return logits_fn(local_model, model_inputs)

  return score


def _concrete_numpy(value):
  if isinstance(value, jax.core.Tracer):
    return None
  if isinstance(value, jax.Array) and not value.is_fully_addressable:
    return None
  return np.asarray(value)


def _validate_input_masks(
    positions,
    validity_mask,
    completion_mask,
    corruption_mask,
    loss_mask,
    *,
    logit_alignment,
    block_size,
):
  """Validates role and loss ownership before launching model computation."""
  shapes = {
      "positions": tuple(positions.shape),
      "validity_mask": tuple(validity_mask.shape),
      "completion_mask": tuple(completion_mask.shape),
      "corruption_mask": tuple(corruption_mask.shape),
      "targets_loss_mask": tuple(loss_mask.shape),
  }
  if len(set(shapes.values())) != 1:
    raise ValueError(f"diffusion OPD masks must have identical shapes; received {shapes}")
  concrete = [_concrete_numpy(value) for value in (positions, validity_mask, completion_mask, corruption_mask, loss_mask)]
  if any(value is None for value in concrete):
    return
  concrete_positions = np.asarray(concrete[0])
  validity, completion, corruption, weights = (np.asarray(value, dtype=bool) for value in concrete[1:])
  if np.any(completion & ~validity):
    raise ValueError("completion_mask must be a subset of valid target positions")
  if np.any(corruption & ~completion):
    raise ValueError("corruption_mask must be a subset of completion_mask")
  if np.any(weights & ~completion):
    raise ValueError("targets_loss_mask must be a subset of completion_mask")
  if np.any(corruption & ~weights):
    raise ValueError("every corrupted position must carry a positive target loss weight")
  safe_clean_targets = np.zeros_like(weights)
  if logit_alignment == "shifted":
    safe_clean_targets = (concrete_positions > 0) & (concrete_positions % block_size == 0)
  if np.any(weights & ~corruption & ~safe_clean_targets):
    raise ValueError(
        "weighted clean student targets are allowed only at shifted logical block anchors; "
        "same-position targets must be corrupted"
    )
  if np.any(weights & (concrete_positions == 0)):
    raise ValueError("causal teacher distillation must not weight logical position zero")


def _safe_loss_weights(
    positions, validity_mask, completion_mask, corruption_mask, loss_weights, *, alignment, block_size
):
  """Masks unsafe rows and target positions using device-side invariants."""
  sequence_length = positions.shape[1]
  batch_size = positions.shape[0]
  validity_mask = jnp.asarray(validity_mask, dtype=jnp.bool_)
  completion_mask = jnp.asarray(completion_mask, dtype=jnp.bool_)
  corruption_mask = jnp.asarray(corruption_mask, dtype=jnp.bool_)
  loss_weights = jnp.asarray(loss_weights)
  positions = jnp.asarray(positions, dtype=jnp.int32)
  weighted_positions = loss_weights > 0
  completion_is_valid = jnp.all(~completion_mask | validity_mask, axis=1)
  corruption_is_completion = jnp.all(~corruption_mask | completion_mask, axis=1)
  loss_is_completion = jnp.all(~weighted_positions | completion_mask, axis=1)
  corruption_is_weighted = jnp.all(~corruption_mask | weighted_positions, axis=1)
  positions_in_range = jnp.all(~validity_mask | ((positions >= 0) & (positions < sequence_length)), axis=1)
  valid_completion_mask = completion_mask & validity_mask
  valid_corruption_mask = corruption_mask & valid_completion_mask
  safe_positions = jnp.clip(positions, 0, sequence_length - 1)
  row_indices = jnp.broadcast_to(jnp.arange(batch_size, dtype=jnp.int32)[:, None], positions.shape)

  logical_counts = jnp.zeros_like(positions).at[row_indices, safe_positions].add(validity_mask.astype(jnp.int32))
  valid_count = jnp.sum(validity_mask, axis=1)
  expected_counts = jnp.arange(sequence_length, dtype=jnp.int32)[None, :] < valid_count[:, None]
  positions_are_canonical = jnp.all(logical_counts == expected_counts.astype(jnp.int32), axis=1)

  logical_completion = jnp.zeros_like(valid_completion_mask).at[row_indices, safe_positions].max(valid_completion_mask)
  seen_completion = jax.lax.associative_scan(jnp.logical_or, logical_completion, axis=1)
  invalid_future_context = seen_completion & expected_counts & ~logical_completion
  has_prompt_origin = expected_counts[:, 0] & ~logical_completion[:, 0]
  has_completion = jnp.any(logical_completion, axis=1)
  row_is_safe = (
      positions_in_range
      & positions_are_canonical
      & completion_is_valid
      & corruption_is_completion
      & loss_is_completion
      & corruption_is_weighted
      & ~jnp.any(invalid_future_context, axis=1)
      & has_prompt_origin
      & has_completion
  )
  numeric_weights_are_safe = jnp.all(jnp.isfinite(loss_weights) & (loss_weights >= 0), axis=1)
  row_is_safe &= numeric_weights_are_safe

  allowed_positions = valid_completion_mask & (positions != 0)
  if alignment == "same_position":
    allowed_positions &= valid_corruption_mask
  else:
    shifted_anchors = (positions > 0) & (positions % block_size == 0)
    allowed_positions &= valid_corruption_mask | shifted_anchors
  row_is_safe &= jnp.all(~weighted_positions | allowed_positions, axis=1)
  return jnp.where(row_is_safe[:, None] & allowed_positions, loss_weights, 0.0)


def _completion_through_first_stop(generated_tokens, completion_mask, stop_token_ids):
  """Keeps completion positions through the first generated stop token, inclusive."""
  if not stop_token_ids:
    return completion_mask
  stop_positions = completion_mask & jnp.isin(generated_tokens, jnp.asarray(stop_token_ids, generated_tokens.dtype))
  seen_stop = jax.lax.associative_scan(jnp.logical_or, stop_positions, axis=1)
  seen_stop_before = jnp.concatenate([jnp.zeros_like(seen_stop[:, :1]), seen_stop[:, :-1]], axis=1)
  return completion_mask & ~seen_stop_before


def prepare_diffusion_opd_batch(
    input_data: distillation_utils.MaxTextTrainingInput,
    student_model: nnx.Module,
    teacher_model: nnx.Module,
    *,
    rollout_fn,
    teacher_score_fn,
    mask_id: int,
    stop_token_ids: tuple[int, ...] = (),
    shifted_seed: bool = False,
    logit_alignment: str = "same_position",
    block_size: int = 1,
) -> diffusion_distillation.DiffusionDistillationBatch:
  """Creates a fresh rollout and teacher scores for one OPD step."""
  required_fields = {
      "targets": input_data.targets,
      "positions": input_data.positions,
      "completion_mask": input_data.completion_mask,
      "corruption_mask": input_data.corruption_mask,
      "targets_loss_mask": input_data.targets_loss_mask,
  }
  missing = sorted(name for name, value in required_fields.items() if value is None)
  if missing:
    raise ValueError(f"diffusion OPD requires explicit batch fields; missing {missing}")
  targets = input_data.targets
  positions = input_data.positions
  decoder_segment_ids = input_data.decoder_segment_ids
  if decoder_segment_ids is None:
    decoder_segment_ids = jnp.asarray(input_data.input_mask, dtype=jnp.int32)
  validity_mask = decoder_segment_ids != 0
  completion_mask = jnp.asarray(input_data.completion_mask, dtype=jnp.bool_)
  corruption_mask = jnp.asarray(input_data.corruption_mask, dtype=jnp.bool_)
  loss_weights = jnp.asarray(input_data.targets_loss_mask, dtype=jnp.float32)
  _validate_input_masks(
      positions,
      validity_mask,
      completion_mask,
      corruption_mask,
      loss_weights,
      logit_alignment=logit_alignment,
      block_size=block_size,
  )
  completion_mask &= validity_mask
  corruption_mask &= completion_mask
  denoise.validate_completion_suffix(
      positions,
      validity_mask,
      completion_mask,
      shifted_seed=shifted_seed,
  )

  generated_tokens = rollout_fn(
      student_model,
      targets,
      positions,
      decoder_segment_ids,
      completion_mask,
  )
  completion_mask = _completion_through_first_stop(generated_tokens, completion_mask, stop_token_ids)
  validity_mask &= ~jnp.asarray(input_data.completion_mask, dtype=jnp.bool_) | completion_mask
  decoder_segment_ids = jnp.where(validity_mask, decoder_segment_ids, 0)
  corruption_mask &= completion_mask
  loss_weights = jnp.where(completion_mask, loss_weights, 0.0)
  student_inputs = jnp.where(corruption_mask, jnp.asarray(mask_id, generated_tokens.dtype), generated_tokens)
  loss_weights = _safe_loss_weights(
      positions,
      validity_mask,
      completion_mask,
      corruption_mask,
      loss_weights,
      alignment=logit_alignment,
      block_size=block_size,
  )
  model_inputs = {
      "input_tokens": student_inputs,
      "positions": positions,
      "decoder_segment_ids": decoder_segment_ids,
      "targets": generated_tokens,
      "targets_segmentation": decoder_segment_ids,
  }
  student_batch = diffusion_types.DiffusionTokenBatch.create(
      model_inputs=model_inputs,
      target_ids=generated_tokens,
      loss_weights=loss_weights,
  )
  teacher_inputs = dict(model_inputs)
  teacher_inputs["input_tokens"] = generated_tokens
  teacher_logits = teacher_score_fn(teacher_model, teacher_inputs)
  return diffusion_distillation.DiffusionDistillationBatch.create(
      student_batch=student_batch,
      teacher_logits=teacher_logits,
  )


def validate_diffusion_opd_configs(student_config, teacher_config, *, is_offline: bool) -> bool:
  """Validates OPD-only constraints and returns whether OPD is enabled."""
  if student_config.distill_data_source != "student_rollout":
    return False
  if is_offline:
    raise ValueError("diffusion OPD requires an online teacher; offline_data_dir is not supported")
  if student_config.training_objective != "block_diffusion":
    raise ValueError("diffusion OPD requires student training_objective='block_diffusion'")
  if not student_config.use_sft or not student_config.sft_train_on_completion_only:
    raise ValueError("diffusion OPD requires completion-only SFT data with an explicit prompt/completion boundary")
  if teacher_config.training_objective != "causal_lm" or teacher_config.attention_type in {
      "block_diffusion",
      "full",
      "compressed",
  }:
    raise ValueError("diffusion OPD currently requires a causal teacher to score clean student rollouts")
  if getattr(teacher_config, "mtp_num_layers", 0) > 0:
    raise ValueError("diffusion OPD does not support an MTP teacher")
  if student_config.vocab_size != teacher_config.vocab_size:
    raise ValueError("diffusion OPD requires matching student and teacher vocabularies")
  student_tokenizer = (
      str(getattr(student_config, "tokenizer_path", "")),
      getattr(getattr(student_config, "tokenizer_type", ""), "value", getattr(student_config, "tokenizer_type", "")),
  )
  teacher_tokenizer = (
      str(getattr(teacher_config, "tokenizer_path", "")),
      getattr(getattr(teacher_config, "tokenizer_type", ""), "value", getattr(teacher_config, "tokenizer_type", "")),
  )
  if student_tokenizer != teacher_tokenizer:
    raise ValueError(
        "diffusion OPD requires identical student and teacher tokenizer_path/tokenizer_type; "
        "token ID remapping is not supported"
    )
  if student_tokenizer[1] != "huggingface":
    raise ValueError("diffusion OPD requires tokenizer_type='huggingface' to match the HF input pipeline")
  if student_config.distill_beta != 0.0 or student_config.distill_beta_end is not None:
    raise ValueError("diffusion OPD does not support feature distillation")
  if student_config.distill_alpha_schedule != "constant" or student_config.distill_temperature_schedule != "constant":
    raise ValueError("diffusion OPD currently requires constant alpha and temperature")
  if student_config.distill_alpha_end is not None or student_config.distill_temperature_end is not None:
    raise ValueError("diffusion OPD does not accept scheduled alpha or temperature end values")
  if not math.isfinite(student_config.distill_alpha) or not 0.0 <= student_config.distill_alpha <= 1.0:
    raise ValueError("distill_alpha must be finite and in [0, 1] for diffusion OPD")
  if not math.isfinite(student_config.distill_temperature) or student_config.distill_temperature <= 0.0:
    raise ValueError("distill_temperature must be finite and positive for diffusion OPD")
  if not math.isfinite(student_config.distill_rollout_temperature):
    raise ValueError("distill_rollout_temperature must be finite for diffusion OPD")
  if not math.isfinite(student_config.distill_rollout_confidence_threshold):
    raise ValueError("distill_rollout_confidence_threshold must be finite for diffusion OPD")
  if student_config.eval_interval > 0:
    raise ValueError("diffusion OPD evaluation is not supported yet; set eval_interval=-1")
  if student_config.steps <= 0:
    raise ValueError("diffusion OPD requires a positive finite number of training steps")
  if student_config.gradient_accumulation_steps > 1 and not student_config.use_tunix_gradient_accumulation:
    raise ValueError(
        "diffusion OPD with gradient accumulation requires use_tunix_gradient_accumulation=True "
        "so the HF batch is divided into Tunix microbatches exactly once"
    )
  if getattr(student_config, "generate_padding_batch_train", False):
    raise ValueError("diffusion OPD does not support generate_padding_batch_train")
  if getattr(student_config, "elastic_enabled", False):
    raise ValueError("diffusion OPD does not yet support elastic data loading or topology changes")
  if getattr(student_config, "expansion_factor_real_data", -1.0) != -1.0:
    raise ValueError("diffusion OPD does not support expansion_factor_real_data")
  if getattr(student_config, "enable_rampup_batch_size", False):
    raise ValueError("diffusion OPD does not support ramp-up batch sizing")
  if student_config.learn_to_init_mode:
    raise ValueError("diffusion OPD is not compatible with learn_to_init_mode")
  if getattr(student_config, "student_params_to_update", None):
    raise ValueError("diffusion OPD does not yet support student_params_to_update filtering")
  max_steps = student_config.distill_rollout_max_denoise_steps
  if max_steps != -1 and max_steps < student_config.block_diffusion_block_size:
    raise ValueError("distill_rollout_max_denoise_steps must be -1 or at least block_diffusion_block_size")
  stop_token_ids = student_config.distill_rollout_stop_token_ids
  if len(set(stop_token_ids)) != len(stop_token_ids) or any(
      token_id < 0 or token_id >= student_config.vocab_size for token_id in stop_token_ids
  ):
    raise ValueError("distill_rollout_stop_token_ids must contain unique IDs in the student vocabulary")
  return True


def _checkpoint_contract(student_config, teacher_config, stop_token_ids=()):
  """Builds the semantic identity that must remain stable across resume."""
  return {
      "version": 1,
      "student_model": str(getattr(student_config, "model_name", "")),
      "student_tokenizer": str(getattr(student_config, "tokenizer_path", "")),
      "student_tokenizer_type": str(
          getattr(getattr(student_config, "tokenizer_type", ""), "value", getattr(student_config, "tokenizer_type", ""))
      ),
      "teacher_model": str(getattr(teacher_config, "model_name", "")),
      "teacher_checkpoint": str(getattr(teacher_config, "load_parameters_path", "")),
      "teacher_tokenizer": str(getattr(teacher_config, "tokenizer_path", "")),
      "teacher_tokenizer_type": str(
          getattr(getattr(teacher_config, "tokenizer_type", ""), "value", getattr(teacher_config, "tokenizer_type", ""))
      ),
      "vocab_size": int(student_config.vocab_size),
      "block_size": int(student_config.block_diffusion_block_size),
      "mask_id": int(student_config.block_diffusion_mask_id),
      "stop_token_ids": [int(token_id) for token_id in stop_token_ids],
      "min_noise": float(student_config.block_diffusion_min_noise),
      "logit_alignment": student_config.block_diffusion_logit_alignment,
      "canvas_policy": student_config.block_diffusion_canvas_policy,
      "rollout_algorithm": student_config.distill_rollout_algorithm,
      "rollout_confidence_threshold": float(student_config.distill_rollout_confidence_threshold),
      "rollout_temperature": float(student_config.distill_rollout_temperature),
      "rollout_max_denoise_steps": int(student_config.distill_rollout_max_denoise_steps),
      "enable_dropout": bool(student_config.enable_dropout),
      "dropout_rate": float(getattr(student_config, "dropout_rate", 0.0)),
      "steps": int(student_config.steps),
      "optimizer_type": str(student_config.opt_type),
      "learning_rate": float(student_config.learning_rate),
      "warmup_steps_fraction": float(student_config.warmup_steps_fraction),
      "learning_rate_final_fraction": float(student_config.learning_rate_final_fraction),
      "gradient_clipping_threshold": float(student_config.gradient_clipping_threshold),
      "adam_b1": float(getattr(student_config, "adam_b1", 0.0)),
      "adam_b2": float(getattr(student_config, "adam_b2", 0.0)),
      "adam_eps": float(getattr(student_config, "adam_eps", 0.0)),
      "adam_eps_root": float(getattr(student_config, "adam_eps_root", 0.0)),
      "adam_weight_decay": float(getattr(student_config, "adam_weight_decay", 0.0)),
      "adamw_mask": str(getattr(student_config, "adamw_mask", "")),
      "mu_dtype": str(getattr(student_config, "mu_dtype", "")),
      "skip_step_on_spikes": bool(getattr(student_config, "skip_step_on_spikes", False)),
      "skip_step_interval": int(getattr(student_config, "skip_step_interval", 0)),
      "skip_step_scaling_factor": float(getattr(student_config, "skip_step_scaling_factor", 0.0)),
      "trainable_parameters_mask": str(getattr(student_config, "trainable_parameters_mask", "")),
      "distill_temperature": float(student_config.distill_temperature),
      "soft_loss_weight": float(student_config.distill_alpha),
      "hard_loss_weight": float(1.0 - student_config.distill_alpha),
      "dataset_type": str(student_config.dataset_type),
      "hf_path": str(getattr(student_config, "hf_path", "")),
      "hf_name": str(getattr(student_config, "hf_name", "")),
      "hf_data_dir": str(getattr(student_config, "hf_data_dir", "")),
      "hf_train_files": str(getattr(student_config, "hf_train_files", "")),
      "train_split": str(getattr(student_config, "train_split", "")),
      "train_data_columns": str(getattr(student_config, "train_data_columns", "")),
      "tokenize_train_data": bool(getattr(student_config, "tokenize_train_data", True)),
      "enable_data_shuffling": bool(getattr(student_config, "enable_data_shuffling", False)),
      "data_shuffle_seed": int(getattr(student_config, "data_shuffle_seed", 0)),
      "num_epoch": int(getattr(student_config, "num_epoch", 1)),
      "max_target_length": int(student_config.max_target_length),
      "add_bos": bool(getattr(student_config, "add_bos", True)),
      "add_eos": bool(getattr(student_config, "add_eos", True)),
      "packing": bool(getattr(student_config, "packing", False)),
      "generate_padding_batch_train": bool(getattr(student_config, "generate_padding_batch_train", False)),
      "chat_template_path": str(getattr(student_config, "chat_template_path", "")),
      "chat_template": str(getattr(student_config, "chat_template", "")),
      "formatting_func_path": str(getattr(student_config, "formatting_func_path", "")),
      "formatting_func_kwargs": str(getattr(student_config, "formatting_func_kwargs", "")),
      "gradient_accumulation_steps": int(student_config.gradient_accumulation_steps),
      "use_tunix_gradient_accumulation": bool(student_config.use_tunix_gradient_accumulation),
      "global_batch_size_to_load": int(student_config.global_batch_size_to_load),
      "global_batch_size_to_train_on": int(getattr(student_config, "global_batch_size_to_train_on", 0)),
      "data_sharding": str(getattr(student_config, "data_sharding", "")),
      "expansion_factor_real_data": float(getattr(student_config, "expansion_factor_real_data", -1.0)),
      "process_count": int(jax.process_count()),
      "local_device_count": int(jax.local_device_count()),
  }


def replay_and_bound_iterator(
    iterator,
    *,
    iter_steps: int,
    train_steps: int,
    max_steps: int,
    accumulation_steps: int,
    replay_iterator=None,
):
  """Replays a deterministic HF stream and bounds it to remaining updates.

  `replay_iterator` may be the host-local source underlying `iterator`. Advancing
  it avoids materializing and device-putting global arrays for skipped batches.
  """
  if iter_steps < 0 or train_steps < 0 or accumulation_steps < 1:
    raise ValueError("restored iterator and training step counts must be nonnegative")
  if train_steps > max_steps:
    raise ValueError(f"restored train_steps ({train_steps}) exceeds configured max_steps ({max_steps})")
  if iter_steps != train_steps * accumulation_steps:
    raise ValueError("restored iter_steps must equal train_steps * accumulation_steps")
  remaining_updates = max_steps - train_steps
  if remaining_updates == 0:
    return iter(())
  replay_source = iterator if replay_iterator is None else replay_iterator
  for replayed_steps in range(iter_steps):
    try:
      next(replay_source)
    except StopIteration as exc:
      raise ValueError(
          f"diffusion OPD HF replay exhausted after {replayed_steps} of {iter_steps} required microbatches"
      ) from exc
  remaining_microbatches = remaining_updates * accumulation_steps

  def take_remaining_exactly():
    for consumed_steps in range(remaining_microbatches):
      try:
        yield next(iterator)
      except StopIteration as exc:
        raise ValueError(
            "diffusion OPD HF stream exhausted after "
            f"{consumed_steps} of {remaining_microbatches} required remaining microbatches"
        ) from exc

  return take_remaining_exactly()


class MaxTextDiffusionOPDTrainer(peft_trainer.PeftTrainer):
  """Tunix trainer that prepares fresh MaxText rollouts before every JIT step."""

  def __init__(
      self,
      *,
      student_model,
      teacher_model,
      optimizer,
      training_config,
      student_config,
      teacher_config,
      eos_id,
  ):
    super().__init__(model=student_model, optimizer=optimizer, training_config=training_config)
    if teacher_model is None:
      raise ValueError("diffusion OPD requires a teacher model")
    configured_stop_ids = tuple(student_config.distill_rollout_stop_token_ids)
    if not configured_stop_ids and (eos_id is None or eos_id < 0 or eos_id >= student_config.vocab_size):
      raise ValueError("diffusion OPD requires a valid tokenizer EOS token ID")
    self.teacher_model = teacher_model
    self.student_config = student_config
    self.stop_token_ids = configured_stop_ids or (int(eos_id),)
    self.checkpoint_contract = _checkpoint_contract(student_config, teacher_config, self.stop_token_ids)
    self.rollout_fn = create_rollout_fn(student_config)
    self.teacher_score_fn = create_teacher_score_fn(teacher_config)
    student_logits_fn = create_target_aligned_logits_fn(student_config, enable_dropout=student_config.enable_dropout)
    tunix_diffusion_opd.configure_prepared_diffusion_opd(
        self,
        lambda batch: batch,
        student_logits_fn,
        temperature=student_config.distill_temperature,
        soft_loss_weight=student_config.distill_alpha,
        hard_loss_weight=1.0 - student_config.distill_alpha,
    )

  def _prepare_inputs(self, input_data):
    return prepare_diffusion_opd_batch(
        input_data,
        self.model,
        self.teacher_model,
        rollout_fn=self.rollout_fn,
        teacher_score_fn=self.teacher_score_fn,
        mask_id=self.student_config.block_diffusion_mask_id,
        stop_token_ids=self.stop_token_ids,
        shifted_seed=self.student_config.block_diffusion_logit_alignment == "shifted",
        logit_alignment=self.student_config.block_diffusion_logit_alignment,
        block_size=self.student_config.block_diffusion_block_size,
    )

  def setup_checkpoint_manager_and_restore(self, raw_train_iter, config):
    """Restores model state; HF stream position is replayed deterministically."""
    if self.checkpoint_manager is not None:
      self.checkpoint_manager.close()
    self.checkpoint_manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=None,
        root_directory=config.checkpoint_dir,
        student_config=config,
        options=self.config.checkpointing_options,
    )
    self._train_steps, self._restored_custom_metadata = self.checkpoint_manager.maybe_restore(
        self.model,
        self.optimizer,
        restore_only_lora_params=getattr(self, "_lora_enabled", False),
    )
    if self._train_steps > 0:
      restored_contract = self._restored_custom_metadata.get("diffusion_opd_contract")
      if restored_contract is None:
        raise ValueError("diffusion OPD checkpoint is missing its semantic training contract")
      if restored_contract != self.checkpoint_contract:
        raise ValueError(
            "diffusion OPD checkpoint contract does not match the current student, teacher, tokenizer, or objective"
        )
    self._iter_steps = self._train_steps * self.config.get_with_default("gradient_accumulation_steps", 1)
    if self._iter_steps:
      max_logging.log(f"Diffusion OPD will replay {self._iter_steps} deterministic HF microbatches after restore.")
    return raw_train_iter

  def custom_checkpoint_metadata(self):
    """Returns the immutable semantic contract persisted with each checkpoint."""
    return {"diffusion_opd_contract": self.checkpoint_contract}
