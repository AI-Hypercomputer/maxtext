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

"""Utility classes for MaxText Distillation with Tunix.

This module contains adapter classes that bridge MaxText's data loading and
model structures with Tunix's training interfaces.
"""

import abc
import dataclasses
import functools
from typing import Any, Callable, Iterator, List, Literal, NamedTuple, Optional, Sequence

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint

from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.common import grain_utility
from tunix.sft import checkpoint_manager as tunix_checkpoint_manager
from tunix.sft import peft_trainer

# -----------------------------------------------------------------------------
# Custom Data Structures
# -----------------------------------------------------------------------------


@flax.struct.dataclass(frozen=True)
class DistillationForwardOutput:
  """Dataclass to carry MaxText-specific output fields."""

  #: Logits [batch, seq, vocab]. With `distill_fused_lm_head` (training mode), a
  #: `(hidden_states [batch, seq, emb], embedding_table [vocab, emb])` tuple instead;
  #: the strategy then projects to logits one sequence chunk at a time.
  logits: jax.Array | tuple[jax.Array, jax.Array]
  #: Distillation features [num_layers, batch, seq, emb]: per-layer attention out-projection activations, or the
  #: decoder block-boundary activations with unscanned layers.
  out_projection_activations: jax.Array | None = None
  #: True if `out_projection_activations` already holds only the `distill_layer_indices` layers (in that order),
  #: so the strategy must not select layers again.
  features_preselected: bool = flax.struct.field(pytree_node=False, default=False)
  #: moe load balance loss
  moe_lb_loss: jax.Array | None = None
  #: top-k indices for sparse offline distillation
  top_k_indices: jax.Array | None = None


@dataclasses.dataclass(frozen=True)
class LMHeadSpec:
  """Per-model settings needed to turn (hidden_states, embedding_table) into logits.

  Mirrors the tied-embedding branch of the decoder's `apply_output_head`.
  """

  #: Divide the logits by sqrt(emb_dim) (`normalize_embedding_logits`).
  normalize_embedding_logits: bool
  #: tanh soft cap applied to the logits (`final_logits_soft_cap`); None or 0 disables it.
  final_logits_soft_cap: float | None
  #: Accumulate the logits matmul in float32 (`logits_dot_in_fp32`).
  logits_dot_in_fp32: bool


def project_to_logits(hidden: jax.Array, embedding_table: jax.Array, head: LMHeadSpec) -> jax.Array:
  """Computes tied-embedding logits [batch, seq, vocab] like the decoder's `apply_output_head`."""
  dot_dtype = jnp.float32 if head.logits_dot_in_fp32 else hidden.dtype
  logits = jnp.einsum(
      "...e,ve->...v", hidden, jnp.asarray(embedding_table, jnp.bfloat16), preferred_element_type=dot_dtype
  )
  if head.normalize_embedding_logits:
    logits = logits / jnp.sqrt(hidden.shape[-1])
  if head.final_logits_soft_cap:
    logits = jnp.tanh(logits / head.final_logits_soft_cap) * head.final_logits_soft_cap
  return logits


class Labels(NamedTuple):
  """Hard-label targets returned by `DistillationStrategy.create_labels`.

  Integer targets plus a mask are used instead of masked one-hot vectors so that
  no [batch, seq, vocab] label tensor is allocated.
  """

  #: Integer target token ids, [batch, seq]. Masked-out positions hold 0.
  targets: jax.Array
  #: Float32 validity mask, [batch, seq]: 1.0 for positions that contribute to the loss.
  mask: jax.Array


@flax.struct.dataclass(frozen=True)
class MaxTextTrainingInput(peft_trainer.TrainingInput):
  """Extended TrainingInput dataclass to carry MaxText-specific fields."""

  #: Position indices for the tokens (for RoPE).
  positions: jax.Array | None = None
  #: Segment IDs for packed sequences (0=padding, 1+=examples).
  decoder_segment_ids: jax.Array | None = None
  #: Ground truth target tokens (used for loss calculation and logging).
  targets: jax.Array | None = None
  #: Position indices for the target tokens.
  targets_position: jax.Array | None = None
  #: Segment IDs for packed target tokens.
  targets_segmentation: jax.Array | None = None
  #: Top-K logits from the teacher model.
  top_k_logits: jax.Array | None = None
  top_k_indices: jax.Array | None = None


# -----------------------------------------------------------------------------
# Data Loading Adapter
# -----------------------------------------------------------------------------


class MaxTextToTunixIterator:
  """Adapts the raw dictionary output of MaxText's data loader to Tunix objects.

  MaxText's `input_pipeline_interface.create_data_iterator` yields a dictionary.
  Tunix expects an object with specific attributes (input_tokens, etc.).
  """

  def __init__(self, maxtext_iterator: Iterator):
    """Initializes the adapter.

    Args:
      maxtext_iterator: The upstream iterator created by MaxText's input pipeline.
    """
    self._iterator = maxtext_iterator

  def __iter__(self):
    """Returns self as the iterator."""
    return self

  def __next__(self) -> MaxTextTrainingInput:
    """Fetches the next batch and converts it to the Tunix data class.

    Returns:
      A MaxTextTrainingInput object containing the batch data.

    Raises:
      StopIteration: If the upstream iterator is exhausted.
    """
    batch = next(self._iterator)

    # Ensure segmentation exists, default to ones if missing (standard non-packed)
    if "inputs_segmentation" in batch:
      input_mask = batch["inputs_segmentation"] != 0
      seg_ids = batch["inputs_segmentation"]
    else:
      # Fallback for non-packed datasets
      input_mask = jnp.ones_like(batch["inputs"], dtype=bool)
      seg_ids = None

    # If in SFT-mode, 'targets' contains prompts which should be masked out when computing the loss.
    # If using with packing the targets_segmentation mask is supposed to be a combined target+packing mask
    targets_segmentation = batch.get("targets_segmentation", jnp.ones_like(batch["targets"]))
    targets_position = batch.get("targets_position", batch.get("inputs_position"))

    # pylint: disable=unexpected-keyword-arg
    return MaxTextTrainingInput(
        input_tokens=batch["inputs"],
        input_mask=input_mask,
        positions=batch["inputs_position"],
        decoder_segment_ids=seg_ids,
        targets=batch["targets"],
        targets_position=targets_position,
        targets_segmentation=targets_segmentation,
        top_k_logits=batch.get("top_k_logits"),
        top_k_indices=batch.get("top_k_indices"),
    )


# -----------------------------------------------------------------------------
# Distillation Strategy
# -----------------------------------------------------------------------------

# Clamp CE before exp() so a divergence spike doesn't poison PPL averages
# with inf. 20 nats is well above plausible CE (Llama random-init ~11.76)
# and far below fp32 exp overflow (~88).
_PPL_CE_CAP = 20.0

METRIC_TOTAL_LOSS = "distill/total_loss"
METRIC_HARD_LOSS = "distill/hard_loss"
METRIC_SOFT_LOSS = "distill/soft_loss"
METRIC_TEACHER_LOSS = "distill/teacher_loss"
METRIC_KL_DIV_T1 = "distill/kl_div_T1"
METRIC_KL_DIV_AT_T = "distill/kl_div_at_T"
METRIC_STUDENT_PERPLEXITY = "distill/student_perplexity"
METRIC_TEACHER_PERPLEXITY = "distill/teacher_perplexity"
METRIC_OUT_PROJ_FEATURE_LOSS = "distill/out_proj_feature_loss"
METRIC_MOE_LB_LOSS = "distill/moe_lb_loss"
METRIC_TEACHER_MOE_LB_LOSS = "distill/teacher_moe_lb_loss"
METRIC_TEMPERATURE = "distill/temperature"
METRIC_ALPHA = "distill/alpha"
METRIC_BETA_FEATURE = "distill/beta_feature"


def compute_schedule(
    step: jax.Array,
    max_steps: int,
    start_value: float,
    end_value: float | None,
    schedule_type: str,
) -> jax.Array:
  """Computes a scheduled value based on training progress.

  Args:
    step: Current training step as a JAX array.
    max_steps: Total number of training steps.
    start_value: Value at the beginning of training.
    end_value: Value at the end of training. If None, returns start_value.
    schedule_type: One of "constant", "linear", or "cosine".

  Returns:
    The scheduled value as a JAX scalar.
  """
  if end_value is None or schedule_type == "constant":
    return jnp.array(start_value, dtype=jnp.float32)

  progress = jnp.clip(step.astype(jnp.float32) / max_steps, 0.0, 1.0)

  if schedule_type == "linear":
    return start_value + (end_value - start_value) * progress
  elif schedule_type == "cosine":
    return end_value + (start_value - end_value) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
  else:
    raise ValueError(f"Unsupported schedule_type: {schedule_type!r}. Must be 'constant', 'linear', or 'cosine'.")


def weighted_mean(sum_count_pairs: Sequence[tuple[Any, Any]] | np.ndarray) -> float:
  """Aggregates `(sum, count)` pairs into a single token-weighted mean.

  Used as the aggregation function for metrics emitted by `compute_loss` and
  `compute_eval_loss`. Robust to per-host imbalance and to varying valid-token
  counts across logging steps:
    final_value = sum(sums) / sum(counts)

  Accepts either a list of (sum, count) tuples or an ndarray of shape (N, 2).
  Tunix's metrics writer can pass either form, so we normalize here.

  Returns 0.0 for an empty input or when total count is non-positive.
  """
  arr = np.asarray(sum_count_pairs, dtype=np.float32)
  if arr.size == 0:
    return 0.0
  # Normalize shape. Single pair -> (1, 2); list of pairs -> (N, 2).
  if arr.ndim == 1:
    arr = arr.reshape(1, -1)
  if arr.ndim != 2 or arr.shape[1] != 2:
    return 0.0
  total = float(arr[:, 1].sum())
  if total <= 0.0:
    return 0.0
  return float(arr[:, 0].sum() / total)


def calculate_distillation_tflops_per_device(
    student_config,
    teacher_config,
    is_offline: bool = False,
) -> tuple[float, float, float]:
  """Per-step per-device TFLOPs for online distillation.

  Student counts full forward + backward (matches the standard `* 3` accounting
  in `calculate_tflops_training_per_device`). Teacher counts forward only, so we
  divide its standard total by 3. In offline mode the teacher is not run, so its
  contribution is zero.

  Caller contract: `teacher_config`'s `per_device_batch_size`, `max_target_length`,
  and `gradient_accumulation_steps` must already match `student_config`'s, since
  the teacher runs on batches produced by the shared input pipeline. The
  distillation entry point (`train_distill.main`) sets this up.

  Returns:
    (combined, student, teacher) per-step per-device TFLOPs.
  """
  student_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(student_config, log=False)
  if is_offline or teacher_config is None:
    teacher_tflops = 0.0
  else:
    teacher_full, _, _ = maxtext_utils.calculate_tflops_training_per_device(teacher_config, log=False)
    teacher_tflops = teacher_full / 3.0
  return student_tflops + teacher_tflops, student_tflops, teacher_tflops


def _sum_over_seq_chunks(
    fn: Callable[..., tuple[jax.Array, ...]],
    num_chunks: int,
    seq_arrays: Sequence[jax.Array],
    *extra_args: Any,
) -> tuple[jax.Array, ...]:
  """Sums the tuple of scalars returned by `fn` over equal chunks of the sequence axis.

  `fn(*seq_chunks, *extra_args)` is called on `num_chunks` equal slices of axis 1
  of every array in `seq_arrays` and is rematerialized (`jax.checkpoint`) per
  chunk, so the backward pass only keeps one chunk's intermediates (e.g. float32
  [batch, chunk, vocab] softmax tensors) live at a time.

  Args:
    fn: Function returning a tuple of scalars (e.g. masked loss sums) for one chunk.
    num_chunks: Number of chunks; must divide the sequence length.
    seq_arrays: Arrays of shape [batch, seq, ...] that are sliced along axis 1.
    *extra_args: Arguments passed unchanged to every call of `fn`.

  Returns:
    The element-wise sum of `fn`'s outputs over all chunks.
  """
  seq_len = seq_arrays[0].shape[1]
  if num_chunks <= 0 or seq_len % num_chunks != 0:
    raise ValueError(f"Sequence length {seq_len} must be divisible by the number of loss chunks ({num_chunks}).")
  ckpt_fn = jax.checkpoint(fn)
  if num_chunks == 1:
    return ckpt_fn(*seq_arrays, *extra_args)
  chunk_len = seq_len // num_chunks

  def body(acc, idx):
    chunks = [jax.lax.dynamic_slice_in_dim(a, idx * chunk_len, chunk_len, axis=1) for a in seq_arrays]
    return jax.tree.map(jnp.add, acc, ckpt_fn(*chunks, *extra_args)), None

  chunk_shapes = [jax.ShapeDtypeStruct((a.shape[0], chunk_len) + a.shape[2:], a.dtype) for a in seq_arrays]
  out_shapes = jax.eval_shape(fn, *chunk_shapes, *extra_args)
  init = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), out_shapes)
  acc, _ = jax.lax.scan(body, init, jnp.arange(num_chunks))
  return acc


def _logit_loss_sums(
    s_logits: jax.Array,
    t_logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    temperature: jax.Array,
    *,
    unit_temperature: bool,
    zero_hard_loss_weight: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Masked sums of (KL at T, KL at T=1, student CE, teacher CE) over full-vocabulary logits.

  Args:
    s_logits: Student logits, [batch, seq, vocab].
    t_logits: Teacher logits, [batch, seq, vocab]. No gradient flows into them.
    targets: Integer targets, [batch, seq].
    mask: Validity mask, [batch, seq].
    temperature: Softmax temperature for the soft loss.
    unit_temperature: True if the temperature is 1 for the whole run; the T=1 KL
      metric then equals the soft KL and is not recomputed.
    zero_hard_loss_weight: True if the hard (student CE) loss has weight 0 for the
      whole run; the student CE is then a metric only and gets no backward pass.

  Returns:
    (kl_at_T_sum, kl_T1_sum, student_ce_sum, teacher_ce_sum).
  """
  s = s_logits.astype(jnp.float32)
  t = jax.lax.stop_gradient(t_logits).astype(jnp.float32)
  kl_soft_pos = optax.kl_divergence(
      jax.nn.log_softmax(s / temperature, axis=-1), jax.nn.softmax(t / temperature, axis=-1)
  )
  kl_soft_sum = jnp.sum(kl_soft_pos * mask)
  if unit_temperature:
    kl_t1_sum = jax.lax.stop_gradient(kl_soft_sum)
  else:
    log_s_1 = jax.nn.log_softmax(jax.lax.stop_gradient(s), axis=-1)
    kl_t1_sum = jnp.sum(optax.kl_divergence(log_s_1, jax.nn.softmax(t, axis=-1)) * mask)
  s_for_ce = jax.lax.stop_gradient(s) if zero_hard_loss_weight else s
  ce_student_sum = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=s_for_ce, labels=targets) * mask)
  ce_teacher_sum = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=t, labels=targets) * mask)
  return kl_soft_sum, kl_t1_sum, ce_student_sum, ce_teacher_sum


# Positional arguments follow the `_sum_over_seq_chunks` protocol: fn(*seq_chunks, *extra_args).
def _fused_logit_loss_sums(  # pylint: disable=too-many-positional-arguments
    s_hidden: jax.Array,
    t_hidden: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    s_embedding: jax.Array,
    t_embedding: jax.Array,
    temperature: jax.Array,
    *,
    student_head: LMHeadSpec,
    teacher_head: LMHeadSpec,
    unit_temperature: bool,
    zero_hard_loss_weight: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Like `_logit_loss_sums`, but first projects hidden states to logits with each model's LM head."""
  s_logits = project_to_logits(s_hidden, s_embedding, student_head)
  t_logits = project_to_logits(jax.lax.stop_gradient(t_hidden), jax.lax.stop_gradient(t_embedding), teacher_head)
  return _logit_loss_sums(
      s_logits,
      t_logits,
      targets,
      mask,
      temperature,
      unit_temperature=unit_temperature,
      zero_hard_loss_weight=zero_hard_loss_weight,
  )


def _top_k_logit_loss_sums(
    s_logits: jax.Array,
    t_top_k_logits: jax.Array,
    top_k_indices: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    *,
    temperature: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Offline-mode loss sums when only the teacher's top-K logits are stored.

  KL(T || S) is computed with both distributions normalized over the teacher's
  top-K vocabulary subset. Full teacher logits are unavailable, so the teacher CE
  and the T=1 KL metrics are reported as 0.

  Args:
    s_logits: Student logits, [batch, seq, vocab].
    t_top_k_logits: Teacher top-K logits, [batch, seq, K].
    top_k_indices: Vocabulary indices of `t_top_k_logits`, [batch, seq, K].
    targets: Integer targets, [batch, seq].
    mask: Validity mask, [batch, seq].
    temperature: Softmax temperature for the soft loss.

  Returns:
    (kl_at_T_sum, 0, student_ce_sum, 0).
  """
  s = s_logits.astype(jnp.float32)
  t = jax.lax.stop_gradient(t_top_k_logits).astype(jnp.float32)
  t_p = jax.nn.softmax(t / temperature, axis=-1)
  log_t_p = jax.nn.log_softmax(t / temperature, axis=-1)
  log_s_p = jax.nn.log_softmax(jnp.take_along_axis(s, top_k_indices, axis=-1) / temperature, axis=-1)
  kl_soft_sum = jnp.sum(jnp.sum(t_p * (log_t_p - log_s_p), axis=-1) * mask)
  ce_student_sum = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=s, labels=targets) * mask)
  zero = jnp.zeros((), dtype=jnp.float32)
  return kl_soft_sum, zero, ce_student_sum, zero


class DistillationStrategy(abc.ABC):
  """Abstract base class for MaxText Distillation Strategies."""

  def __init__(
      self, student_forward_fn: Callable, teacher_forward_fn: Callable, vocab_size: int, pad_id: int = 0, **kwargs
  ):
    """Initializes the generic distillation strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        vocab_size: The size of the model's vocabulary.
        pad_id: The ID used for padding tokens.
    """
    self.student_forward_fn = student_forward_fn
    self.teacher_forward_fn = teacher_forward_fn
    self.vocab_size = vocab_size
    self.pad_id = pad_id

  @abc.abstractmethod
  def compute_loss(
      self,
      student_output: "DistillationForwardOutput",
      teacher_output: "DistillationForwardOutput",
      labels: Labels,
      step: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, tuple[jax.Array, jax.Array]]]:
    """Computes the distillation loss.

    Args:
        student_output: The forward pass output of the student model.
        teacher_output: The forward pass output of the frozen teacher model.
        labels: The ground truth labels from `create_labels`.
        step: Current training step for dynamic scheduling. If None, uses fixed values.

    Returns:
        A tuple containing the scalar loss and a dictionary of auxiliary metrics
        (e.g., {"distill/soft_loss": ..., "distill/total_loss": ...})
    """
    raise NotImplementedError

  @abc.abstractmethod
  def compute_eval_loss(
      self,
      student_output: "DistillationForwardOutput",
      labels: Labels,
  ) -> tuple[jax.Array, dict[str, tuple[jax.Array, jax.Array]]]:
    """Computes the evaluation loss (typically just the task loss).

    Args:
        student_output: The forward pass output of the student model.
        labels: The ground truth labels from `create_labels`.

    Returns:
        A tuple containing the scalar loss and an empty (or auxiliary) dict.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def create_labels(self, targets: jax.Array, targets_segmentation: Optional[jax.Array] = None, **kwargs) -> Labels:
    """
    Creates the labels used to compute the loss.
    """
    raise NotImplementedError


class CombinedDistillationStrategy(DistillationStrategy):
  """Strategy that returns detailed metrics for TensorBoard."""

  def __init__(
      self,
      student_forward_fn: Callable[..., DistillationForwardOutput],
      teacher_forward_fn: Callable[..., DistillationForwardOutput],
      pad_id: int = 0,
      temperature: float = 2.0,
      alpha: float = 0.5,
      beta_feature: float = 0.0,
      layer_indices: Optional[List[int]] = None,
      feature_loss_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
      feature_loss_type: Literal["cosine", "l2"] = "cosine",
      cosine_distance_axis: int | tuple[int, ...] = -1,
      vocab_size: int = 0,
      alpha_end: float | None = None,
      alpha_schedule: Literal["constant", "linear", "cosine"] = "constant",
      temperature_end: float | None = None,
      temperature_schedule: Literal["constant", "linear", "cosine"] = "constant",
      beta_end: float | None = None,
      beta_schedule: Literal["constant", "linear", "cosine"] = "constant",
      max_steps: int = 1,
      logit_loss_chunks: int | None = None,
      fused_lm_head_chunks: int | None = None,
      student_lm_head: LMHeadSpec | None = None,
      teacher_lm_head: LMHeadSpec | None = None,
  ):
    """Initializes the Combined distillation strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        temperature: Temperature for softening probabilities (> 0).
        alpha: Weight to balance distillation loss and task loss (0.0 to 1.0).
        beta_feature: Weight to balance feature loss (0.0 to 1.0). 0.0 disables feature loss.
        layer_indices: Layer indices to apply feature loss.
        feature_loss_type: The type of feature loss to use if `feature_loss_fn` is None.
          Can be "cosine" (default) or "l2".
        feature_loss_fn: A function (student_features, teacher_features, mask) -> scalar
          where features are [L, B, T, D] and mask is [B, T] (1.0 for valid tokens).
          Defaults to a masked cosine distance with epsilon-floored safe-norm.
        cosine_distance_axis: The axis to use for cosine distance computation if
          feature_loss_fn is not provided. Defaults to -1.
        alpha_end: Target alpha value at end of training. None keeps alpha fixed.
        alpha_schedule: Schedule type for alpha annealing.
        temperature_end: Target temperature at end of training. None keeps temperature fixed.
        temperature_schedule: Schedule type for temperature annealing.
        beta_end: Target beta_feature value at end of training. None keeps beta fixed.
        beta_schedule: Schedule type for beta annealing.
        max_steps: Total training steps, used for schedule computation.
        logit_loss_chunks: Number of sequence chunks used to compute the online
          (full-vocabulary) logit losses. Each chunk is rematerialized in the
          backward pass, which bounds the float32 softmax intermediates to one
          chunk. Must divide the sequence length. None picks 4 chunks when the
          sequence length is a multiple of 4 and at least 512, else 1.
        fused_lm_head_chunks: Number of sequence chunks for models that return
          `(hidden_states, embedding_table)` (`distill_fused_lm_head`). Logits are
          computed per chunk with `student_lm_head` / `teacher_lm_head`, so full
          sequence logits are never materialized. Must divide the sequence length.
          None means the models return logits.
        student_lm_head: LM head settings of the student; required with `fused_lm_head_chunks`.
        teacher_lm_head: LM head settings of the teacher; required with `fused_lm_head_chunks`.
    """

    super().__init__(
        student_forward_fn=student_forward_fn,
        teacher_forward_fn=teacher_forward_fn,
        vocab_size=vocab_size,
        pad_id=pad_id,
    )

    self.temperature = temperature
    self.alpha = alpha
    self.beta_feature = beta_feature
    self.layer_indices = jnp.array(layer_indices) if layer_indices is not None else None

    # Schedule parameters
    self.alpha_end = alpha_end
    self.alpha_schedule = alpha_schedule
    self.temperature_end = temperature_end
    self.temperature_schedule = temperature_schedule
    self.beta_end = beta_end
    self.beta_schedule = beta_schedule
    self.max_steps = max_steps
    if logit_loss_chunks is not None and logit_loss_chunks <= 0:
      raise ValueError(f"logit_loss_chunks must be positive, got {logit_loss_chunks}")
    self.logit_loss_chunks = logit_loss_chunks
    if fused_lm_head_chunks is not None:
      if fused_lm_head_chunks <= 0:
        raise ValueError(f"fused_lm_head_chunks must be positive, got {fused_lm_head_chunks}")
      if student_lm_head is None or teacher_lm_head is None:
        raise ValueError("fused_lm_head_chunks requires student_lm_head and teacher_lm_head.")
    self.fused_lm_head_chunks = fused_lm_head_chunks
    self.student_lm_head = student_lm_head
    self.teacher_lm_head = teacher_lm_head

    # Validate schedule parameter ranges
    if alpha_end is not None and not 0.0 <= alpha_end <= 1.0:
      raise ValueError(f"alpha_end must be in [0, 1], got {alpha_end}")
    if temperature_end is not None and temperature_end <= 0.0:
      raise ValueError(f"temperature_end must be > 0, got {temperature_end}")
    if beta_end is not None and beta_end < 0.0:
      raise ValueError(f"beta_end must be >= 0, got {beta_end}")
    if beta_feature == 0.0 and beta_end is not None and beta_end > 0.0:
      raise ValueError(
          f"distill_beta=0.0 but distill_beta_end={beta_end}. Feature extraction is disabled when "
          "distill_beta starts at 0.0 (the model does not sow intermediate activations). "
          "Set distill_beta to a small positive value (e.g., 1e-6) to enable feature extraction."
      )
    for param_name, schedule, end_value in [
        ("alpha", alpha_schedule, alpha_end),
        ("temperature", temperature_schedule, temperature_end),
        ("beta", beta_schedule, beta_end),
    ]:
      if schedule != "constant" and end_value is None:
        raise ValueError(
            f"{param_name}_schedule is '{schedule}' but {param_name}_end is None. "
            f"Set {param_name}_end to a target value or use schedule='constant'."
        )

    # Mask keeps zero-norm pad activations out of the cosine denominator to avoid 0/0 NaN.
    self.feature_loss_fn = feature_loss_fn
    if feature_loss_fn is None:
      if feature_loss_type == "cosine":

        def _masked_cosine(student_features, teacher_features, mask):
          # epsilon>0 floors the safe-norm so an all-zero row can't divide by zero.
          cd = optax.cosine_distance(
              student_features, teacher_features, axis=cosine_distance_axis, epsilon=1e-6
          )  # [L, B, T]
          mask_b = mask.astype(cd.dtype)
          num_valid_terms = jnp.maximum(jnp.sum(mask_b), 1.0) * cd.shape[0]
          return jnp.sum(cd * mask_b[None, :, :]) / num_valid_terms

        self.feature_loss_fn = _masked_cosine
      elif feature_loss_type == "l2":

        def _masked_l2(student_features, teacher_features, mask):
          sq = jnp.mean(jnp.square(student_features - teacher_features), axis=-1)  # [L, B, T]
          mask_b = mask.astype(sq.dtype)
          num_valid_terms = jnp.maximum(jnp.sum(mask_b), 1.0) * sq.shape[0]
          return jnp.sum(sq * mask_b[None, :, :]) / num_valid_terms

        self.feature_loss_fn = _masked_l2
      else:
        raise ValueError(f"Unsupported feature_loss_type: {feature_loss_type!r}")

  def _get_scheduled_weights(self, step: jax.Array | None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Resolves the current alpha, temperature, and beta values from schedules.

    Args:
      step: Current training step. If None, returns the fixed initial values.

    Returns:
      A tuple of (alpha, temperature, beta_feature) as JAX scalars.
    """
    if step is None:
      return (
          jnp.array(self.alpha, dtype=jnp.float32),
          jnp.array(self.temperature, dtype=jnp.float32),
          jnp.array(self.beta_feature, dtype=jnp.float32),
      )
    alpha = compute_schedule(step, self.max_steps, self.alpha, self.alpha_end, self.alpha_schedule)
    temperature = compute_schedule(
        step, self.max_steps, self.temperature, self.temperature_end, self.temperature_schedule
    )
    beta_feature = compute_schedule(step, self.max_steps, self.beta_feature, self.beta_end, self.beta_schedule)
    return alpha, temperature, beta_feature

  def _select_feature_layers(self, features: jax.Array, preselected: bool) -> jax.Array:
    """Selects the `layer_indices` layers of [num_layers, ...] features unless the model already did."""
    if self.layer_indices is None or preselected:
      return features
    return jnp.take(features, self.layer_indices, axis=0)

  def _is_fixed_at(self, start: float, end: float | None, schedule: str, value: float) -> bool:
    """Returns True if a scheduled weight equals `value` for the whole run."""
    return start == value and (end is None or schedule == "constant" or end == value)

  def _logit_loss_sums(
      self,
      student_output: DistillationForwardOutput,
      teacher_output: DistillationForwardOutput,
      targets: jax.Array,
      mask: jax.Array,
      temperature: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns masked sums of (KL at T, KL at T=1, student CE, teacher CE)."""
    fixed_weights = {
        "unit_temperature": self._is_fixed_at(self.temperature, self.temperature_end, self.temperature_schedule, 1.0),
        "zero_hard_loss_weight": self._is_fixed_at(self.alpha, self.alpha_end, self.alpha_schedule, 1.0),
    }
    s_logits = student_output.logits
    t_logits = teacher_output.logits
    num_fused_chunks = self.fused_lm_head_chunks
    student_head, teacher_head = self.student_lm_head, self.teacher_lm_head
    if isinstance(s_logits, tuple) or isinstance(t_logits, tuple):
      if (
          not isinstance(s_logits, tuple)
          or not isinstance(t_logits, tuple)
          or num_fused_chunks is None
          or student_head is None
          or teacher_head is None
      ):
        raise ValueError(
            "Fused LM-head outputs (hidden_states, embedding_table) require both the student and the teacher to use "
            "distill_fused_lm_head and the strategy to be built with fused_lm_head_chunks."
        )
      s_hidden, s_embedding = s_logits
      t_hidden, t_embedding = t_logits
      loss_fn = functools.partial(
          _fused_logit_loss_sums, student_head=student_head, teacher_head=teacher_head, **fixed_weights
      )
      kl_sum, kl_t1_sum, ce_student_sum, ce_teacher_sum = _sum_over_seq_chunks(
          loss_fn,
          num_fused_chunks,
          (s_hidden, t_hidden, targets, mask),
          s_embedding,
          t_embedding,
          temperature,
      )
      return kl_sum, kl_t1_sum, ce_student_sum, ce_teacher_sum
    if teacher_output.top_k_indices is not None:
      return _top_k_logit_loss_sums(
          s_logits, t_logits, teacher_output.top_k_indices, targets, mask, temperature=temperature
      )
    loss_fn = functools.partial(_logit_loss_sums, **fixed_weights)
    seq_len = s_logits.shape[1]
    num_chunks = self.logit_loss_chunks
    if num_chunks is None:
      num_chunks = 4 if seq_len >= 512 and seq_len % 4 == 0 else 1
    kl_sum, kl_t1_sum, ce_student_sum, ce_teacher_sum = _sum_over_seq_chunks(
        loss_fn, num_chunks, (s_logits, t_logits, targets, mask), temperature
    )
    return kl_sum, kl_t1_sum, ce_student_sum, ce_teacher_sum

  def compute_loss(
      self,
      student_output: DistillationForwardOutput,
      teacher_output: DistillationForwardOutput,
      labels: Labels,
      step: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, tuple[jax.Array, jax.Array]]]:
    """Computes Loss and Auxiliary Metrics.

    Metrics are emitted as (sum, count) pairs so that they can be aggregated
    across hosts and across logging windows in a token-weighted (unbiased) way:
    final_value = sum(sums) / sum(counts).
    """
    # Resolve scheduled weights for this step
    alpha, temperature, beta_feature = self._get_scheduled_weights(step)

    s_features = student_output.out_projection_activations
    t_features = teacher_output.out_projection_activations

    if (s_features is None or t_features is None) and self.beta_feature > 0.0:
      raise ValueError(
          "Features extracted from student or teacher model are None, but distill_beta > 0.0. "
          "Ensure the model architecture supports feature extraction (e.g., 'out_projection_activations' is sowed)."
      )

    mask = labels.mask.astype(jnp.float32)  # [B, T]
    valid_count = jnp.sum(mask)
    safe_count = jnp.maximum(valid_count, 1.0)

    kl_softened_sum, kl_t1_sum, ce_student_sum, ce_teacher_sum = self._logit_loss_sums(
        student_output, teacher_output, labels.targets, mask, temperature
    )
    soft_loss_sum_scaled = kl_softened_sum * (temperature**2)
    soft_loss_mean = soft_loss_sum_scaled / safe_count
    hard_loss_mean = ce_student_sum / safe_count

    base_logit_loss = (alpha * soft_loss_mean) + ((1.0 - alpha) * hard_loss_mean)

    feature_loss = jnp.array(0.0, dtype=jnp.float32)
    if self.beta_feature > 0.0:
      assert s_features is not None and t_features is not None
      s_features_sliced = self._select_feature_layers(s_features, student_output.features_preselected)
      t_features_sliced = self._select_feature_layers(t_features, teacher_output.features_preselected)

      feature_loss_fn = self.feature_loss_fn

      # Rematerialized so the float32 copies of the [L, B, T, D] features are not kept for the backward pass.
      @jax.checkpoint
      def _feature_loss(s_feat, t_feat, mask_in):
        s_f = s_feat.astype(jnp.float32)
        t_f = jax.lax.stop_gradient(t_feat).astype(jnp.float32)
        return feature_loss_fn(s_f, t_f, mask_in)  # pyrefly: ignore[not-callable]

      feature_loss = beta_feature * _feature_loss(s_features_sliced, t_features_sliced, mask)

    total_loss = base_logit_loss + feature_loss

    moe_lb_loss = jnp.array(0.0)
    if student_output.moe_lb_loss is not None:
      # The moe_lb_loss collected from the model is already scaled by load_balance_loss_weight
      # within the MoE layer itself (see load_balance_loss in moe.py).
      moe_lb_loss = student_output.moe_lb_loss
      total_loss += moe_lb_loss

    teacher_moe_lb_loss = jnp.array(0.0)
    if teacher_output.moe_lb_loss is not None:
      teacher_moe_lb_loss = teacher_output.moe_lb_loss

    # Per-step next-token perplexity. Note: this is mean(exp(per-step CE)), not
    # exp(window-CE-mean) — close to true perplexity in steady state. For the exact
    # perplexity over a logging window compute exp(distill/hard_loss) on the TB side.
    teacher_loss_mean = ce_teacher_sum / safe_count
    student_perplexity_step = jnp.exp(jnp.minimum(hard_loss_mean, _PPL_CE_CAP))
    teacher_perplexity_step = jnp.exp(jnp.minimum(teacher_loss_mean, _PPL_CE_CAP))

    one = jnp.array(1.0, dtype=jnp.float32)
    metrics: dict[str, tuple[jax.Array, jax.Array]] = {
        # Token-weighted: emit (sum, valid_count) so multi-host averaging is unbiased.
        METRIC_SOFT_LOSS: (soft_loss_sum_scaled, valid_count),
        METRIC_HARD_LOSS: (ce_student_sum, valid_count),
        METRIC_TEACHER_LOSS: (ce_teacher_sum, valid_count),
        # Next-token prediction perplexity (per-step approximation of exp(hard_loss)).
        # The headline `_train_perplexity` Tunix prints is exp(total_loss) which for
        # distillation is exp(α·soft + (1-α)·hard + β·feature) and NOT next-token PPL.
        METRIC_STUDENT_PERPLEXITY: (student_perplexity_step, one),
        METRIC_TEACHER_PERPLEXITY: (teacher_perplexity_step, one),
        # KL at the current (scheduled) temperature T, without the T^2 scaling
        # that soft_loss applies. Pair with kl_div_T1 to compare T vs T=1.
        METRIC_KL_DIV_AT_T: (kl_softened_sum, valid_count),
        # KL at T=1: comparable across runs / annealing schedules.
        METRIC_KL_DIV_T1: (kl_t1_sum, valid_count),
        # Per-step quantities: (value, 1.0) so the aggregator yields a simple mean over steps.
        METRIC_OUT_PROJ_FEATURE_LOSS: (feature_loss, one),
        METRIC_MOE_LB_LOSS: (moe_lb_loss, one),
        METRIC_TEACHER_MOE_LB_LOSS: (teacher_moe_lb_loss, one),
        METRIC_TOTAL_LOSS: (total_loss, one),
        METRIC_TEMPERATURE: (temperature, one),
        METRIC_ALPHA: (alpha, one),
        METRIC_BETA_FEATURE: (beta_feature, one),
    }
    return total_loss, metrics

  def compute_eval_loss(
      self,
      student_output: DistillationForwardOutput,
      labels: Labels,
  ) -> tuple[jax.Array, dict[str, tuple[jax.Array, jax.Array]]]:
    """Computes Eval Loss. Returns (loss, metrics) with (sum, count) metric pairs."""
    mask = labels.mask.astype(jnp.float32)
    valid_count = jnp.sum(mask)
    safe_count = jnp.maximum(valid_count, 1.0)

    s_out_logits = student_output.logits
    if isinstance(s_out_logits, tuple):
      num_fused_chunks = self.fused_lm_head_chunks
      head = self.student_lm_head
      if num_fused_chunks is None or head is None:
        raise ValueError("Fused LM-head student outputs require the strategy to be built with fused_lm_head_chunks.")
      s_hidden, s_embedding = s_out_logits

      def _ce_sum(hidden, targets, mask_in, embedding, head=head):
        logits = project_to_logits(hidden, embedding, head).astype(jnp.float32)
        return (jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets) * mask_in),)

      (ce_sum,) = _sum_over_seq_chunks(_ce_sum, num_fused_chunks, (s_hidden, labels.targets, mask), s_embedding)
    else:
      s_logits = s_out_logits.astype(jnp.float32)
      ce_per_pos = optax.softmax_cross_entropy_with_integer_labels(logits=s_logits, labels=labels.targets)
      ce_sum = jnp.sum(ce_per_pos * mask)
    task_loss = ce_sum / safe_count

    metrics = {
        "eval/hard_loss": (ce_sum, valid_count),
        "eval/student_perplexity": (jnp.exp(jnp.minimum(task_loss, _PPL_CE_CAP)), jnp.array(1.0, dtype=jnp.float32)),
    }
    return task_loss, metrics

  def create_labels(self, targets, targets_segmentation=None, **kwargs) -> Labels:
    """Returns integer targets and a validity mask (pad and segmentation-0 positions are masked out)."""
    del kwargs  # Unused
    mask = jnp.not_equal(targets, self.pad_id).astype(jnp.float32)
    if targets_segmentation is not None:
      mask = mask * (targets_segmentation != 0).astype(jnp.float32)
    safe_targets = jnp.where(mask > 0, targets, 0).astype(jnp.int32)
    return Labels(targets=safe_targets, mask=mask)


# -----------------------------------------------------------------------------
# Checkpoint Manager
# -----------------------------------------------------------------------------


class MaxTextCheckpointManager(tunix_checkpoint_manager.CheckpointManager):
  """Custom CheckpointManager that uses MaxText's native handlers.

  Model and optimizer are delegated to Tunix's v1 ``Checkpointer`` unchanged.
  The Grain input pipeline is added as an extra ``"iter"`` checkpointable via
  ``GrainCheckpointable``, which implements Orbax's ``StatefulCheckpointable``.
  """

  def __init__(
      self,
      raw_iterator: Any | None,
      root_directory: str | None,
      student_config: Any,
      options: checkpoint.CheckpointManagerOptions | None = None,
  ):
    super().__init__(root_directory=root_directory, options=options)
    self.student_config = student_config
    self._iterator = raw_iterator

  def save(
      self,
      step,
      model,
      optimizer=None,
      save_only_lora_params=False,
      force=False,
      custom_metadata=None,
  ):
    """Saves model, optimizer and the Grain input pipeline state."""
    if self._checkpointer is None:
      return False

    # Standard Tunix Logic for Model/Optimizer.
    # Accept either a ModelBundle (common path) or a plain nnx module.
    target_model = getattr(model, "student_model", model)
    if save_only_lora_params:
      params = nnx.state(target_model, nnx.LoRAParam)
    else:
      params = nnx.state(target_model)

    checkpointables: dict[str, Any] = {"model_params": params}
    # Exclude optimizer state when learn_to_init_mode is active.
    exclude_opt = self.student_config.learn_to_init_mode

    if optimizer is not None and not exclude_opt:
      checkpointables["optimizer_state"] = nnx.state(optimizer, nnx.optimizer.OptState)

    if self._iterator is not None:
      # Follow MaxText's logic to handle multi-process saving
      # Logic extracted from src/maxtext/common/checkpointing.py:save_checkpoint
      data_iterator = self._iterator
      if not isinstance(data_iterator, list):
        data_iterator = [data_iterator]

      grain_iters_to_save = []
      process_count_total = jax.process_count() * len(data_iterator)

      for i, data_iter in enumerate(data_iterator):
        process_index = jax.process_index() + i * jax.process_count()
        # MaxText iterators (MultiHostDataLoadIterator) wrap the actual Grain iterator in .local_iterator
        local_iter = data_iter.local_iterator if hasattr(data_iter, "local_iterator") else data_iter
        grain_iters_to_save.append((local_iter, process_index, process_count_total))

      checkpointables["iter"] = grain_utility.GrainCheckpointable(grain_iters_to_save)

    return self._save_checkpointables(step, checkpointables, force, custom_metadata)

  def maybe_restore(  # pyrefly: ignore[bad-override]
      self,
      model: Any,
      optimizer: Any = None,
      restore_only_lora_params: bool = False,
  ) -> tuple[int, dict[str, Any]]:
    """Restores model + optimizer by delegating to upstream Tunix.

    Unwraps `ModelBundle` if present (we only restore `student_model`).

    Returns:
      (restored step, custom_metadata dict). Step is 0 if no checkpoint exists.
    """
    if self._checkpointer is None:
      return 0, {}

    target_model = getattr(model, "student_model", model)

    step, custom_metadata = super().maybe_restore(
        model=target_model,
        optimizer=optimizer,
        restore_only_lora_params=restore_only_lora_params,
    )
    if step == 0:
      return 0, {}

    max_logging.log(f"Restored from checkpoint step {step}.")

    return step, dict(custom_metadata or {})

  def restore_iterator(self):
    """Restores the iterator using MaxText's logic."""
    if self._checkpointer is None or self._iterator is None:
      return None

    step = self.latest_step()
    if step is None:
      return None

    try:
      # MaxText logic for restoration (simplified for standard case)
      # We assume 1-to-1 process mapping for now (no elasticity logic here yet)
      data_iter = self._iterator
      local_iter = data_iter.local_iterator if hasattr(data_iter, "local_iterator") else data_iter

      self._checkpointer.load_checkpointables(
          step,
          {"iter": grain_utility.GrainCheckpointable(local_iter)},
      )
      # Since Grain restores in-place via set_state(), we return the original object
      return self._iterator

    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Warning: Could not restore input pipeline: {e}")
      return None

  def wait_until_finished(self):
    """Blocks until all outstanding checkpoint operations are complete."""
    if self._checkpointer is not None:
      self._checkpointer.wait()

  def close(self) -> None:
    """Closes the checkpoint manager."""
    super().close()
    if getattr(self, "_checkpointer", None) is not None:
      if hasattr(self._checkpointer, "close"):
        self._checkpointer.close()
