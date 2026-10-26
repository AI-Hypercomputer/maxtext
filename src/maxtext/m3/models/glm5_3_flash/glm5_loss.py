# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loss function and optimizer mask for GLM-5.3-Flash training.

Note on Sequence Packing (F16):
Host-side sequence packing validation is enforced by:
- Glm5Config.__post_init__ raising ValueError when packing=True
  (src/maxtext/m3/models/glm5_3_flash/glm5_config.py:136-137)
- glm5_config.validate_no_packing(decoder_segment_ids)
  (src/maxtext/m3/models/glm5_3_flash/glm5_config.py:35-41)
- MaxText data processing pipelines honoring config.packing
  (src/maxtext/input_pipeline/data_processing_utils.py:93,
   src/maxtext/input_pipeline/tfds_data_processing.py:241,
   src/maxtext/input_pipeline/hf_data_processing.py:550)
"""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.glm5_3_flash import glm5_moe
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
import optax


def compute_glm5_cross_entropy_loss(
    logits: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes masked token cross-entropy loss strictly in float32.

  Args:
    logits: Model logits of shape [batch_size, seq_len, vocab_size].
    decoder_target_tokens: Next-token shifted target token ids of shape
      [batch_size, seq_len] (`tokens[:, 1:]` aligned with `tokens[:, :-1]`).
    decoder_target_mask: Next-token shifted target loss mask of shape
      [batch_size, seq_len] (`tokens[:, 1:]` aligned with `tokens[:, :-1]`).

  Returns:
    (ce_loss, xent_sum, total_weights) tuple strictly in float32.
  """
  logits_f32 = logits.astype(jnp.float32)
  targets_arr = jnp.asarray(decoder_target_tokens)
  mask_f32 = (jnp.asarray(decoder_target_mask) != 0).astype(jnp.float32)

  token_losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_f32, labels=targets_arr)
  masked_losses = token_losses * mask_f32
  xent_sum = jnp.sum(masked_losses).astype(jnp.float32)
  total_weights = jnp.sum(mask_f32).astype(jnp.float32)
  ce_loss = (xent_sum / jnp.maximum(total_weights, 1.0)).astype(jnp.float32)
  return ce_loss, xent_sum, total_weights


def _compute_tiled_cross_entropy(
    model: modeling_glm5_3_flash.Glm5Model,
    hidden_states: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array,
    num_vocab_tiling: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes cross-entropy loss tiled across batch-sequence tokens.

  Args:
    model: The Glm5Model instance exposing
      `logits_from_hidden_states_for_vocab_tiling`.
    hidden_states: Post-final-norm hidden states of shape [batch_size, seq_len,
      hidden_size].
    decoder_target_tokens: Target tokens of shape [batch_size, seq_len].
    decoder_target_mask: Target mask of shape [batch_size, seq_len].
    num_vocab_tiling: Number of tiles along the batch*seq_len token dimension.

  Returns:
    (ce_loss, xent_sum, total_weights) tuple strictly in float32.
  """
  if num_vocab_tiling <= 1:
    logits = model.logits_from_hidden_states_for_vocab_tiling(hidden_states)
    return compute_glm5_cross_entropy_loss(logits, decoder_target_tokens, decoder_target_mask)

  b, s, emb_dim = hidden_states.shape
  total_tokens = b * s
  if total_tokens % num_vocab_tiling != 0:
    raise ValueError(
        f"Batch tokens ({b} * {s} = {total_tokens}) must be divisible by" f" num_vocab_tiling={num_vocab_tiling}."
    )
  tile_size = total_tokens // num_vocab_tiling
  hidden_tiles = hidden_states.reshape((num_vocab_tiling, tile_size, emb_dim))
  targets_tiles = jnp.asarray(decoder_target_tokens).reshape((num_vocab_tiling, tile_size))
  mask_tiles = (jnp.asarray(decoder_target_mask) != 0).astype(jnp.float32).reshape((num_vocab_tiling, tile_size))

  total_weights = jnp.sum(mask_tiles).astype(jnp.float32)

  @jax.checkpoint
  def _tile_xent_sum(
      h_tile: jax.Array,
      t_tile: jax.Array,
      m_tile: jax.Array,
  ) -> jax.Array:
    chunk_logits = model.logits_from_hidden_states_for_vocab_tiling(h_tile).astype(jnp.float32)
    chunk_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=chunk_logits,
        labels=t_tile,
    )
    return jnp.sum(chunk_loss * m_tile).astype(jnp.float32)

  def _scan_step(
      acc: jax.Array,
      tile_inputs: tuple[jax.Array, jax.Array, jax.Array],
  ) -> tuple[jax.Array, None]:
    h_tile, t_tile, m_tile = tile_inputs
    tile_sum = _tile_xent_sum(h_tile, t_tile, m_tile)
    return acc + tile_sum, None

  xent_sum, _ = jax.lax.scan(
      _scan_step,
      jnp.zeros((), dtype=jnp.float32),
      (hidden_tiles, targets_tiles, mask_tiles),
  )

  ce_loss = (xent_sum / jnp.maximum(total_weights, 1.0)).astype(jnp.float32)
  return ce_loss, xent_sum, total_weights


def compute_glm5_loss(
    model: modeling_glm5_3_flash.Glm5Model,
    tokens: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array,
    decoder_segment_ids: jax.Array | None = None,
    decoder_positions: jax.Array | None = None,
    *,
    is_train: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes cross-entropy and load-balancing loss for GLM-5.3-Flash.

  Contract:
    `decoder_target_tokens` and `decoder_target_mask` are next-token shifted
    targets (`tokens[:, 1:]` aligned with `tokens[:, :-1]`), matching
    MaxText's `loss_fn` convention.
    In CL7, `is_train` indicates training mode (selecting MTP inclusion added
    in CL8); `total_loss = ce_loss + moe_lb_loss_weight * moe_lb_loss` in both
    modes.

  Args:
    model: The Glm5Model instance being trained.
    tokens: Input token ids of shape [batch_size, seq_len].
    decoder_target_tokens: Next-token shifted targets [batch_size, seq_len].
    decoder_target_mask: Next-token shifted loss mask [batch_size, seq_len].
    decoder_segment_ids: Optional input segmentation [batch_size, seq_len].
    decoder_positions: Optional input positions [batch_size, seq_len].
    is_train: Required boolean flag gating training-specific loss components.

  Returns:
    (total_loss, aux_metrics) tuple strictly in float32.
  """
  if not isinstance(is_train, bool):
    raise TypeError(f"is_train must be a bool, got {type(is_train)}")
  if tokens is None:
    raise ValueError("tokens must be provided as a non-None array.")
  if decoder_target_tokens is None:
    raise ValueError("decoder_target_tokens must be provided as a non-None array.")
  if decoder_target_mask is None:
    raise ValueError("decoder_target_mask must be provided as a non-None array.")

  tokens_arr = jnp.asarray(tokens)
  target_tokens_arr = jnp.asarray(decoder_target_tokens)
  target_mask_arr = jnp.asarray(decoder_target_mask)

  if target_tokens_arr.shape != tokens_arr.shape:
    raise ValueError(f"decoder_target_tokens shape {target_tokens_arr.shape} != tokens shape" f" {tokens_arr.shape}.")
  if target_mask_arr.shape != tokens_arr.shape:
    raise ValueError(f"decoder_target_mask shape {target_mask_arr.shape} != tokens shape" f" {tokens_arr.shape}.")

  seg_ids = jnp.asarray(decoder_segment_ids) if decoder_segment_ids is not None else None
  positions = jnp.asarray(decoder_positions) if decoder_positions is not None else None

  num_vocab_tiling = model.config.num_vocab_tiling
  if num_vocab_tiling > 1:
    out = model.forward_with_aux(
        tokens_arr,
        decoder_positions=positions,
        decoder_segment_ids=seg_ids,
        compute_logits=False,
    )
    ce_loss, xent_sum, total_weights = _compute_tiled_cross_entropy(
        model=model,
        hidden_states=out.hidden_states,
        decoder_target_tokens=target_tokens_arr,
        decoder_target_mask=target_mask_arr,
        num_vocab_tiling=num_vocab_tiling,
    )
  else:
    out = model.forward_with_aux(
        tokens_arr,
        decoder_positions=positions,
        decoder_segment_ids=seg_ids,
        compute_logits=True,
    )
    if out.logits is None:
      raise RuntimeError("out.logits is None but compute_logits=True was requested.")
    ce_loss, xent_sum, total_weights = compute_glm5_cross_entropy_loss(out.logits, target_tokens_arr, target_mask_arr)

  moe_lb_loss = out.aux["moe_lb_loss"].astype(jnp.float32)
  total_loss = (ce_loss + jnp.float32(model.config.moe_lb_loss_weight) * moe_lb_loss).astype(jnp.float32)

  aux_metrics = {
      "ce_loss": ce_loss,
      "moe_lb_loss": moe_lb_loss,
      "xent_sum": xent_sum,
      "total_weights": total_weights,
  }
  return total_loss, aux_metrics


def glm5_loss_fn(
    model: modeling_glm5_3_flash.Glm5Model,
    *,
    inputs: jax.Array,
    targets: jax.Array,
    targets_segmentation: jax.Array,
    is_train: bool,
    inputs_segmentation: jax.Array | None = None,
    inputs_position: jax.Array | None = None,
    targets_position: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes masked token cross-entropy loss and folds MoE load-balancing loss.

  Contract:
    `targets` and `targets_segmentation` are next-token shifted targets
    (`inputs[:, 1:]` aligned with `inputs[:, :-1]`), matching MaxText's
    `loss_fn` convention.
    In CL7, `is_train` indicates training mode (selecting MTP inclusion added
    in CL8); `total_loss = ce_loss + moe_lb_loss_weight * moe_lb_loss` in both
    modes.

  Args:
    model: The Glm5Model instance being trained.
    inputs: Input token ids of shape [batch_size, seq_len].
    targets: Target token ids of shape [batch_size, seq_len].
    targets_segmentation: Required segmentation mask [batch_size, seq_len].
    is_train: Required boolean flag gating training-specific loss components.
    inputs_segmentation: Optional input segmentation [batch_size, seq_len].
    inputs_position: Optional input positions [batch_size, seq_len].
    targets_position: Optional target positions [batch_size, seq_len].

  Returns:
    (total_loss, aux_metrics) tuple.
  """
  if not isinstance(is_train, bool):
    raise TypeError(f"is_train must be a bool, got {type(is_train)}")
  del targets_position
  return compute_glm5_loss(
      model=model,
      tokens=inputs,
      decoder_target_tokens=targets,
      decoder_target_mask=targets_segmentation,
      decoder_segment_ids=inputs_segmentation,
      decoder_positions=inputs_position,
      is_train=is_train,
  )


def trainable_mask(model_or_params: Any) -> Any:
  """Returns boolean PyTree (False on indexer.* and e_score_correction_bias)."""
  tree = (
      nnx.state(model_or_params, (nnx.Param, glm5_moe.Glm5RouterBias))
      if isinstance(model_or_params, nnx.Module)
      else model_or_params
  )

  def _is_trainable(path: tuple[Any, ...], leaf: Any) -> bool:
    del leaf
    path_str = jax.tree_util.keystr(path)
    if "indexer" in path_str:
      return False
    if "e_score_correction_bias" in path_str:
      return False
    return True

  return jax.tree_util.tree_map_with_path(
      _is_trainable,
      tree,
      is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.Array)),
  )
