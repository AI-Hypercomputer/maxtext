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

"""Model-independent block-diffusion rollout state transitions."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class DenoiseTrace(NamedTuple):
  """Compact sampled trajectory for replaying diffusion policy scores."""

  tokens: jax.Array
  action_steps: jax.Array
  action_logps: jax.Array


def _validate_shapes(initial_tokens, positions, validity_mask, completion_mask):
  """Checks that all token-level rollout arrays share a batch-major shape."""
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
  if isinstance(value, jax.core.Tracer):
    return None
  if isinstance(value, jax.Array) and not value.is_fully_addressable:
    return None
  return np.asarray(value)


def _per_row_rngs(rng, batch_size):
  """Normalizes one or per-row typed/legacy JAX keys to per-row keys."""
  rng = jnp.asarray(rng)
  is_typed_key = jax.dtypes.issubdtype(rng.dtype, jax.dtypes.prng_key)
  scalar_shape = () if is_typed_key else tuple(jax.random.PRNGKey(0).shape)
  if tuple(rng.shape) == scalar_shape:
    return jax.vmap(lambda row: jax.random.fold_in(rng, row))(jnp.arange(batch_size, dtype=jnp.int32))
  if tuple(rng.shape) == (batch_size, *scalar_shape):
    return rng
  raise ValueError(
      "rng must be one PRNG key or one key per batch row; "
      f"received shape {tuple(rng.shape)} for key shape {scalar_shape}"
  )


def _select_row_rngs(row_mask, new_rngs, old_rngs):
  selector = row_mask.reshape((row_mask.shape[0],) + (1,) * (old_rngs.ndim - 1))
  return jnp.where(selector, new_rngs, old_rngs)


def _validate_logical_positions(positions, validity_mask, completion_mask, *, shifted_seed):
  """Checks logical sequence invariants on eager, host-addressable inputs."""
  concrete_positions = _concrete_numpy(positions)
  concrete_validity = _concrete_numpy(validity_mask)
  concrete_completion = _concrete_numpy(completion_mask)
  if concrete_positions is None or concrete_validity is None or concrete_completion is None:
    return
  concrete_validity = np.asarray(concrete_validity, dtype=bool)
  concrete_completion = np.asarray(concrete_completion, dtype=bool) & concrete_validity
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
        raise ValueError("shifted seed-and-mask rollout requires exactly one prompt token at logical position zero")


def validate_completion_suffix(positions, validity_mask, completion_mask, *, shifted_seed=False):
  """Validates the initial OPD rollout scope when eager values are available.

  The first integration intentionally supports one prompt followed by one
  completion per sequence. This prevents clean future turns from appearing in
  the bidirectional attention block of an earlier generated completion.
  """
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
      raise ValueError("diffusion OPD rollout requires completion_mask to be a contiguous suffix")


def low_confidence_rollout(
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
    rng: jax.Array | None = None,
) -> DenoiseTrace:
  """Samples a completion and records each token's pre-commit action step.

  ``logits_fn`` must return target-aligned logits for the current token canvas.
  Each denoising step commits every token at or above the confidence threshold;
  if a row has no such token, its highest-confidence unresolved token is forced
  to commit. ``action_steps`` compactly identifies the exact masked canvas for
  replay: token ``p`` was scored while every token with step >= its step was
  still masked. Prompt, padding, and inactive completion tokens use step -1.
  """
  _validate_shapes(initial_tokens, positions, validity_mask, completion_mask)
  valid_contracts = {("same_position", "all_masked"), ("shifted", "seed_and_mask")}
  if (logit_alignment, canvas_policy) not in valid_contracts:
    raise ValueError(
        "rollout supports only same_position/all_masked or shifted/seed_and_mask; "
        f"received {logit_alignment}/{canvas_policy}"
    )
  if block_size <= 0:
    raise ValueError(f"block_size must be positive; received {block_size}")
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
  action_steps = jnp.full(initial_tokens.shape, -1, dtype=jnp.int32)
  action_logps = jnp.zeros(initial_tokens.shape, dtype=jnp.float32)
  sample_tokens = rng is not None
  rng = jax.random.PRNGKey(0) if rng is None else jnp.asarray(rng)
  row_rngs = _per_row_rngs(rng, initial_tokens.shape[0])
  block_ids = positions // block_size
  num_blocks = (initial_tokens.shape[1] + block_size - 1) // block_size

  def propose(current_canvas, proposal_keys):
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
    log_probabilities = jax.nn.log_softmax(scaled_logits, axis=-1)
    if sample_tokens:
      proposed_tokens = jax.vmap(lambda key, row_logits: jax.random.categorical(key, row_logits, axis=-1))(
          proposal_keys, scaled_logits
      ).astype(initial_tokens.dtype)
    else:
      proposed_tokens = jnp.argmax(scaled_logits, axis=-1).astype(initial_tokens.dtype)
    proposed_logps = jnp.take_along_axis(log_probabilities, proposed_tokens[..., None], axis=-1)[..., 0]
    confidence = jnp.max(jnp.exp(log_probabilities), axis=-1)
    return proposed_tokens, confidence, proposed_logps

  def generate_block(block_id, rollout_state):
    current_canvas, current_steps, current_logps, global_step, current_rng = rollout_state
    in_block = completion_mask & (block_ids == block_id)

    def run_active_block(active_state):
      active_canvas, active_steps, active_logps, active_global_step, active_rng = active_state
      anchors = in_block & (positions % block_size == 0) if shifted_seed else jnp.zeros_like(in_block)

      def generate_anchors(anchor_state):
        anchor_canvas, anchor_steps, anchor_logps, anchor_global_step, anchor_rng = anchor_state
        split_keys = jax.vmap(jax.random.split)(anchor_rng)
        next_rng = split_keys[:, 0]
        proposal_keys = split_keys[:, 1]
        anchor_tokens, _, proposed_logps = propose(anchor_canvas, proposal_keys)
        anchor_canvas = jnp.where(anchors, anchor_tokens, anchor_canvas)
        anchor_steps = jnp.where(anchors, anchor_global_step[:, None], anchor_steps)
        anchor_logps = jnp.where(anchors, proposed_logps, anchor_logps)
        rows_with_anchors = jnp.any(anchors, axis=1)
        anchor_rng = _select_row_rngs(rows_with_anchors, next_rng, anchor_rng)
        return (
            anchor_canvas,
            anchor_steps,
            anchor_logps,
            anchor_global_step + rows_with_anchors.astype(jnp.int32),
            anchor_rng,
        )

      if shifted_seed:
        active_canvas, active_steps, active_logps, active_global_step, active_rng = jax.lax.cond(
            jnp.any(anchors),
            generate_anchors,
            lambda value: value,
            (active_canvas, active_steps, active_logps, active_global_step, active_rng),
        )
      unresolved = in_block & ~anchors

      def continue_denoising(state):
        step, _, _, _, _, _, remaining = state
        return (step < max_denoise_steps) & jnp.any(remaining)

      def denoise_step(state):
        step, step_canvas, step_ids, step_logps, step_global, step_rng, remaining = state
        split_keys = jax.vmap(jax.random.split)(step_rng)
        next_rng = split_keys[:, 0]
        proposal_keys = split_keys[:, 1]
        proposed_tokens, confidence, proposed_logps = propose(step_canvas, proposal_keys)
        commits = remaining & (confidence >= confidence_threshold)
        row_needs_fallback = jnp.any(remaining, axis=1) & ~jnp.any(commits, axis=1)
        fallback_confidence = jnp.max(jnp.where(remaining, confidence, -jnp.inf), axis=1)
        tied_for_fallback = remaining & (confidence == fallback_confidence[:, None])
        fallback_positions = jnp.where(tied_for_fallback, positions, positions.shape[1])
        fallback_indices = jnp.argmin(fallback_positions, axis=1)
        fallback = jax.nn.one_hot(fallback_indices, remaining.shape[1], dtype=jnp.bool_)
        commits |= fallback & row_needs_fallback[:, None]
        step_canvas = jnp.where(commits, proposed_tokens, step_canvas)
        step_ids = jnp.where(commits, step_global[:, None], step_ids)
        step_logps = jnp.where(commits, proposed_logps, step_logps)
        rows_with_commits = jnp.any(commits, axis=1)
        step_rng = _select_row_rngs(rows_with_commits, next_rng, step_rng)
        return (
            step + 1,
            step_canvas,
            step_ids,
            step_logps,
            step_global + rows_with_commits.astype(jnp.int32),
            step_rng,
            remaining & ~commits,
        )

      _, active_canvas, active_steps, active_logps, active_global_step, active_rng, _ = jax.lax.while_loop(
          continue_denoising,
          denoise_step,
          (
              jnp.asarray(0, dtype=jnp.int32),
              active_canvas,
              active_steps,
              active_logps,
              active_global_step,
              active_rng,
              unresolved,
          ),
      )
      return active_canvas, active_steps, active_logps, active_global_step, active_rng

    return jax.lax.cond(
        jnp.any(in_block),
        run_active_block,
        lambda value: value,
        (current_canvas, current_steps, current_logps, global_step, current_rng),
    )

  canvas, action_steps, action_logps, _, _ = jax.lax.fori_loop(
      0,
      num_blocks,
      generate_block,
      (
          canvas,
          action_steps,
          action_logps,
          jnp.zeros((initial_tokens.shape[0],), dtype=jnp.int32),
          row_rngs,
      ),
  )
  return DenoiseTrace(tokens=canvas, action_steps=action_steps, action_logps=action_logps)


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
  """Greedily generates tokens while preserving the original public API."""

  return low_confidence_rollout(
      logits_fn,
      initial_tokens,
      positions,
      validity_mask,
      completion_mask,
      block_size=block_size,
      mask_id=mask_id,
      logit_alignment=logit_alignment,
      canvas_policy=canvas_policy,
      confidence_threshold=confidence_threshold,
      temperature=temperature,
      max_denoise_steps=max_denoise_steps,
  ).tokens
