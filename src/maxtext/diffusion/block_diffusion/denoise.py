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

"""Model-independent block-diffusion denoising transitions."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


_SUPPORTED_CONTRACTS = {
    ("same_position", "all_masked"),
    ("shifted", "seed_and_mask"),
}


def _validate_shapes(initial_tokens, positions, validity_mask, completion_mask):
  """Checks that all token-level arrays share a batch-major shape."""
  expected_shape = tuple(initial_tokens.shape)
  if len(expected_shape) != 2:
    raise ValueError(f"initial_tokens must have shape [batch, length]; received {expected_shape}")
  for name, value in (
      ("positions", positions),
      ("validity_mask", validity_mask),
      ("completion_mask", completion_mask),
  ):
    if tuple(value.shape) != expected_shape:
      raise ValueError(f"{name} must match initial_tokens shape; received {tuple(value.shape)} and {expected_shape}")


def _concrete_numpy(value):
  """Returns a NumPy view only for values already resident on the host."""
  if isinstance(value, (jax.Array, jax.core.Tracer, jax.ShapeDtypeStruct)):
    return None
  return np.asarray(value)


def _validate_logical_positions(positions, validity_mask, completion_mask, *, shifted_seed):
  """Checks eager, host-addressable logical sequence invariants."""
  concrete_positions = _concrete_numpy(positions)
  concrete_validity = _concrete_numpy(validity_mask)
  concrete_completion = _concrete_numpy(completion_mask)
  if concrete_positions is None or concrete_validity is None or concrete_completion is None:
    return
  concrete_validity = np.asarray(concrete_validity, dtype=bool)
  concrete_completion = np.asarray(concrete_completion, dtype=bool)
  if np.any(concrete_completion & ~concrete_validity):
    raise ValueError("completion_mask must be a subset of valid token positions")
  concrete_completion &= concrete_validity
  sequence_length = concrete_positions.shape[1]
  for row in range(concrete_positions.shape[0]):
    valid_positions = np.asarray(concrete_positions[row, concrete_validity[row]])
    expected_positions = np.arange(valid_positions.size, dtype=valid_positions.dtype)
    if valid_positions.size and not np.array_equal(np.sort(valid_positions), expected_positions):
      raise ValueError(
          "valid logical positions must be unique and contiguous from zero within the physical sequence length"
      )
    if np.any(valid_positions < 0) or np.any(valid_positions >= sequence_length):
      raise ValueError("valid logical positions must be nonnegative and smaller than the physical sequence length")
    if shifted_seed:
      position_zero = concrete_validity[row] & (concrete_positions[row] == 0)
      if np.count_nonzero(position_zero) != 1 or np.any(concrete_completion[row] & position_zero):
        raise ValueError("shifted seed-and-mask generation requires exactly one prompt token at logical position zero")


def validate_completion_suffix(positions, validity_mask, completion_mask, *, shifted_seed=False):
  """Validates one prompt followed by one generated completion per row."""
  _validate_logical_positions(positions, validity_mask, completion_mask, shifted_seed=shifted_seed)
  concrete_positions = _concrete_numpy(positions)
  concrete_validity = _concrete_numpy(validity_mask)
  concrete_completion = _concrete_numpy(completion_mask)
  if concrete_positions is None or concrete_validity is None or concrete_completion is None:
    return
  concrete_validity = np.asarray(concrete_validity, dtype=bool)
  concrete_completion = np.asarray(concrete_completion, dtype=bool) & concrete_validity
  for row in range(concrete_positions.shape[0]):
    valid_indices = np.flatnonzero(concrete_validity[row])
    if valid_indices.size == 0:
      continue
    ordered_indices = valid_indices[np.argsort(concrete_positions[row, valid_indices])]
    ordered_completion = concrete_completion[row, ordered_indices]
    completion_indices = np.flatnonzero(ordered_completion)
    if completion_indices.size and not np.all(ordered_completion[completion_indices[0] :]):
      raise ValueError("block-diffusion generation requires completion_mask to be a contiguous suffix")


def low_confidence_generate(
    logits_fn: Callable[[jax.Array], jax.Array],
    initial_tokens: jax.Array,
    positions: jax.Array,
    validity_mask: jax.Array,
    completion_mask: jax.Array,
    *,
    block_size: int,
    mask_id: int,
    logit_alignment: str,
    canvas_policy: str,
    confidence_threshold: float = 0.9,
    temperature: float = 1.0,
    max_denoise_steps: int | None = None,
) -> jax.Array:
  """Generates a completion block by block with confidence-based commits.

  ``logits_fn`` returns target-aligned logits for the current token canvas.
  Each step commits every token at or above the confidence threshold. If a row
  has no such token, its highest-confidence unresolved token is committed to
  guarantee progress for heterogeneous and partial blocks.
  """
  _validate_shapes(initial_tokens, positions, validity_mask, completion_mask)
  if (logit_alignment, canvas_policy) not in _SUPPORTED_CONTRACTS:
    raise ValueError(
        "generation supports only same_position/all_masked or shifted/seed_and_mask; "
        f"received {logit_alignment}/{canvas_policy}"
    )
  if block_size <= 0:
    raise ValueError(f"block_size must be positive; received {block_size}")
  if canvas_policy == "seed_and_mask" and block_size < 2:
    raise ValueError("seed_and_mask requires block_size to be at least 2")
  if mask_id < 0:
    raise ValueError(f"mask_id must be nonnegative; received {mask_id}")
  if not 0.0 <= confidence_threshold <= 1.0:
    raise ValueError(f"confidence_threshold must be in [0, 1]; received {confidence_threshold}")
  if temperature <= 0.0:
    raise ValueError(f"temperature must be positive; received {temperature}")
  if max_denoise_steps is None:
    max_denoise_steps = block_size
  if max_denoise_steps < block_size:
    raise ValueError(
        f"max_denoise_steps must be at least block_size ({block_size}) to guarantee completion; "
        f"received {max_denoise_steps}"
    )

  shifted_seed = logit_alignment == "shifted"
  validate_completion_suffix(positions, validity_mask, completion_mask, shifted_seed=shifted_seed)
  validity_mask = jnp.asarray(validity_mask, dtype=jnp.bool_)
  completion_mask = jnp.asarray(completion_mask, dtype=jnp.bool_) & validity_mask
  positions = jnp.asarray(positions, dtype=jnp.int32)
  canvas = jnp.where(completion_mask, jnp.asarray(mask_id, initial_tokens.dtype), initial_tokens)
  block_ids = positions // block_size
  num_blocks = (initial_tokens.shape[1] + block_size - 1) // block_size

  def propose(current_canvas):
    logits = logits_fn(current_canvas)
    expected_prefix = tuple(current_canvas.shape)
    if len(logits.shape) != 3 or tuple(logits.shape[:2]) != expected_prefix:
      raise ValueError(
          "logits_fn must return [batch, length, vocab] target-aligned logits; "
          f"received {tuple(logits.shape)} for canvas {expected_prefix}"
      )
    vocab_size = logits.shape[-1]
    if vocab_size < 2:
      raise ValueError("logits_fn must expose at least two vocabulary entries so the mask token can be excluded")
    if not 0 <= mask_id < vocab_size:
      raise ValueError(f"mask_id must satisfy 0 <= mask_id < vocab_size ({vocab_size}); received {mask_id}")
    scaled_logits = jnp.asarray(logits, dtype=jnp.float32) / temperature
    scaled_logits = scaled_logits.at[..., mask_id].set(-jnp.inf)
    probabilities = jax.nn.softmax(scaled_logits, axis=-1)
    proposed_tokens = jnp.argmax(scaled_logits, axis=-1).astype(initial_tokens.dtype)
    return proposed_tokens, jnp.max(probabilities, axis=-1)

  def generate_block(block_id, current_canvas):
    in_block = completion_mask & (block_ids == block_id)

    def run_active_block(active_canvas):
      anchors = in_block & (positions % block_size == 0) if shifted_seed else jnp.zeros_like(in_block)

      def generate_anchors(anchor_canvas):
        anchor_tokens, _ = propose(anchor_canvas)
        return jnp.where(anchors, anchor_tokens, anchor_canvas)

      if shifted_seed:
        active_canvas = jax.lax.cond(jnp.any(anchors), generate_anchors, lambda value: value, active_canvas)
      unresolved = in_block & ~anchors

      def continue_denoising(state):
        step, _, remaining = state
        return (step < max_denoise_steps) & jnp.any(remaining)

      def denoise_step(state):
        step, step_canvas, remaining = state
        proposed_tokens, confidence = propose(step_canvas)
        commits = remaining & (confidence >= confidence_threshold)
        row_needs_fallback = jnp.any(remaining, axis=1) & ~jnp.any(commits, axis=1)
        finite_confidence = jnp.where(jnp.isfinite(confidence), confidence, -jnp.inf)
        fallback_confidence = jnp.max(jnp.where(remaining, finite_confidence, -jnp.inf), axis=1)
        tied_for_fallback = remaining & (finite_confidence == fallback_confidence[:, None])
        fallback_positions = jnp.where(tied_for_fallback, positions, positions.shape[1])
        fallback_indices = jnp.argmin(fallback_positions, axis=1)
        fallback = jax.nn.one_hot(fallback_indices, remaining.shape[1], dtype=jnp.bool_)
        commits |= fallback & row_needs_fallback[:, None]
        step_canvas = jnp.where(commits, proposed_tokens, step_canvas)
        return step + 1, step_canvas, remaining & ~commits

      _, active_canvas, _ = jax.lax.while_loop(
          continue_denoising,
          denoise_step,
          (jnp.asarray(0, dtype=jnp.int32), active_canvas, unresolved),
      )
      return active_canvas

    return jax.lax.cond(jnp.any(in_block), run_active_block, lambda value: value, current_canvas)

  return jax.lax.fori_loop(0, num_blocks, generate_block, canvas)
