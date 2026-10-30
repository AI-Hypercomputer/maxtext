# Copyright 2026 Google LLC
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

"""Token-weighted cross-entropy, `jax.checkpoint` vocabulary tiling, z-loss, and MoE auxiliary loss."""

from collections.abc import Mapping
import dataclasses
import inspect
import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.m3.core.interfaces import CausalLMOutput


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LossBreakdown:
  """Pure PyTree breakdown of scalar training/evaluation losses and MoE router counts."""

  total_loss: jax.Array
  lm_loss: jax.Array
  xent_sum: jax.Array
  z_loss_sum: jax.Array
  aux_loss: jax.Array
  valid_tokens: jax.Array
  router_token_counts: jax.Array | None = None


def compute_dense_xent_and_z_loss_sums(
    logits: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    *,
    z_loss_multiplier: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
  """Computes unnormalized float32 token-masked cross-entropy sum and z-loss sum from dense logits."""
  logits_f32 = logits.astype(jnp.float32)
  weights_f32 = weights.astype(jnp.float32)

  max_logit = jax.lax.stop_gradient(jnp.max(logits_f32, axis=-1))
  log_z = max_logit + jnp.log(jnp.sum(jnp.exp(logits_f32 - max_logit[..., None]), axis=-1))
  target_logit = jnp.take_along_axis(logits_f32, targets[..., None].astype(jnp.int32), axis=-1)[..., 0]

  xent_sum = jnp.sum((log_z - target_logit) * weights_f32)
  if z_loss_multiplier > 0.0:
    z_loss_sum = jnp.asarray(z_loss_multiplier, dtype=jnp.float32) * jnp.sum(jnp.square(log_z) * weights_f32)
  else:
    z_loss_sum = jnp.zeros((), dtype=jnp.float32)
  return xent_sum, z_loss_sum


def _extract_normed_hidden_and_vocab_weight(
    model: nnx.Module,
    hidden_states: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Applies final decoder norm and extracts the `[D, V]` output projection matrix."""
  decoder = getattr(model, "decoder", None)
  if decoder is not None and hasattr(decoder, "decoder_norm"):
    normed = decoder.decoder_norm(hidden_states)
  elif hasattr(model, "norm") and callable(getattr(model, "norm")):
    normed = model.norm(hidden_states)
  else:
    normed = hidden_states

  if decoder is not None and hasattr(decoder, "logits_dense"):
    vocab_weight = decoder.logits_dense.kernel[...]
  elif hasattr(model, "lm_head") and hasattr(model.lm_head, "kernel"):
    vocab_weight = model.lm_head.kernel[...]
  elif hasattr(model, "token_embedder") and hasattr(model.token_embedder, "embedding"):
    vocab_weight = model.token_embedder.embedding[...].T
  else:
    raise ValueError(
        f"Model {type(model).__name__} does not expose `decoder.logits_dense`, `lm_head`, or "
        "`token_embedder` required for vocabulary-tiled cross-entropy."
    )
  return normed, vocab_weight


def compute_vocab_tiled_xent_sum(
    model: nnx.Module,
    hidden_states: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    *,
    num_vocab_tiles: int,
    z_loss_multiplier: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
  """Computes exact cross-entropy and z-loss sums by scanning `jax.checkpoint` vocabulary tiles.

  Avoids materializing or caching dense `[B, S, V]` logits across forward and backward passes: peak
  logit memory scales as `O(B * S * (V / num_vocab_tiles))` while matching `num_vocab_tiles == 1`
  loss and parameter gradients to float32 precision.
  """
  normed, vocab_weight = _extract_normed_hidden_and_vocab_weight(model, hidden_states)
  cfg = getattr(model, "config", None)
  logits_dot_in_fp32 = bool(getattr(cfg, "logits_dot_in_fp32", False))
  normalize_logits = bool(getattr(cfg, "normalize_embedding_logits", False))
  soft_cap = getattr(cfg, "final_logits_soft_cap", None)

  hidden_dim, vocab_size = int(vocab_weight.shape[0]), int(vocab_weight.shape[1])
  num_tiles = max(1, min(int(num_vocab_tiles), vocab_size))
  tile_size = math.ceil(vocab_size / num_tiles)
  padded_vocab = num_tiles * tile_size
  if padded_vocab > vocab_size:
    vocab_weight_padded = jnp.pad(vocab_weight, ((0, 0), (0, padded_vocab - vocab_size)))
  else:
    vocab_weight_padded = vocab_weight

  # Shape `[T, D, V_tile]` so `jax.lax.scan` slices one vocabulary tile per iteration.
  weight_tiles = jnp.swapaxes(vocab_weight_padded.reshape(hidden_dim, num_tiles, tile_size), 0, 1)
  tile_starts = jnp.arange(num_tiles, dtype=jnp.int32) * tile_size

  num_tokens = int(targets.size)
  normed_flat = normed.reshape(num_tokens, hidden_dim)
  targets_flat = targets.reshape(num_tokens).astype(jnp.int32)
  weights_flat = weights.reshape(num_tokens).astype(jnp.float32)
  attend_dtype = jnp.float32 if logits_dot_in_fp32 else normed.dtype

  @jax.checkpoint
  def _tile_step(
      carry: tuple[jax.Array, jax.Array, jax.Array],
      xs: tuple[jax.Array, jax.Array],
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    running_max, running_sum_exp, target_logit = carry
    w_tile, tile_start = xs

    tile_logits = jnp.dot(normed_flat.astype(attend_dtype), w_tile.astype(attend_dtype))
    if normalize_logits:
      tile_logits = tile_logits / jnp.sqrt(float(hidden_dim))
    if soft_cap:
      tile_logits = jnp.tanh(tile_logits / soft_cap) * soft_cap
    tile_logits = tile_logits.astype(jnp.float32)

    vocab_ids = tile_start + jnp.arange(tile_size, dtype=jnp.int32)
    valid_vocab = vocab_ids < vocab_size

    masked_for_max = jnp.where(valid_vocab[None, :], tile_logits, -jnp.inf)
    tile_max = jax.lax.stop_gradient(jnp.max(masked_for_max, axis=-1))
    new_max = jax.lax.stop_gradient(jnp.maximum(running_max, tile_max))
    safe_new_max = jnp.where(jnp.isfinite(new_max), new_max, 0.0)

    # Double-where guards ensure `jnp.exp` inputs are finite on both initial carry and padded columns.
    prev_has_val = jnp.isfinite(running_max)
    prev_scale = jnp.where(
        prev_has_val,
        jnp.exp(jnp.where(prev_has_val, running_max - safe_new_max, 0.0)),
        0.0,
    )
    safe_shifted = jnp.where(valid_vocab[None, :], tile_logits - safe_new_max[:, None], 0.0)
    tile_exp = jnp.where(valid_vocab[None, :], jnp.exp(safe_shifted), 0.0)
    new_sum_exp = running_sum_exp * prev_scale + jnp.sum(tile_exp, axis=-1)

    in_tile = (targets_flat >= tile_start) & (targets_flat < jnp.minimum(tile_start + tile_size, vocab_size))
    local_idx = jnp.where(in_tile, targets_flat - tile_start, 0)
    gathered = jnp.take_along_axis(tile_logits, local_idx[:, None], axis=-1)[:, 0]
    new_target_logit = target_logit + jnp.where(in_tile, gathered, 0.0)

    return (new_max, new_sum_exp, new_target_logit), None

  init_carry = (
      jnp.full((num_tokens,), -jnp.inf, dtype=jnp.float32),
      jnp.zeros((num_tokens,), dtype=jnp.float32),
      jnp.zeros((num_tokens,), dtype=jnp.float32),
  )
  (final_max, final_sum_exp, final_target_logit), _ = jax.lax.scan(
      _tile_step,
      init_carry,
      (weight_tiles, tile_starts),
  )
  log_z = final_max + jnp.log(final_sum_exp)
  xent_sum = jnp.sum((log_z - final_target_logit) * weights_flat)
  if z_loss_multiplier > 0.0:
    z_loss_sum = jnp.asarray(z_loss_multiplier, dtype=jnp.float32) * jnp.sum(jnp.square(log_z) * weights_flat)
  else:
    z_loss_sum = jnp.zeros((), dtype=jnp.float32)
  return xent_sum, z_loss_sum


def _invoke_model_forward(
    model: nnx.Module,
    batch: Mapping[str, Any],
    *,
    return_logits: bool,
) -> CausalLMOutput:
  """Calls `model` with `batch` and normalizes raw array outputs into `CausalLMOutput`."""
  sig = inspect.signature(model.__call__)
  params = sig.parameters
  has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

  kwargs: dict[str, Any] = {}
  if "decoder_segment_ids" in params or has_var_kw:
    kwargs["decoder_segment_ids"] = batch.get("segment_ids")
  elif "segment_ids" in params:
    kwargs["segment_ids"] = batch.get("segment_ids")

  if "return_output" in params or has_var_kw:
    kwargs["return_output"] = True
  if "return_logits" in params or has_var_kw:
    kwargs["return_logits"] = return_logits

  extras = batch.get("extras")
  if isinstance(extras, Mapping):
    for k, v in extras.items():
      if k in params or has_var_kw:
        kwargs[k] = v

  raw_out = model(batch["inputs"], batch["positions"], **kwargs)
  if isinstance(raw_out, CausalLMOutput):
    return raw_out
  return CausalLMOutput(logits=raw_out)


def compute_batch_loss_sums(
    model: nnx.Module,
    batch: Mapping[str, Any],
    *,
    num_vocab_tiles: int = 1,
    z_loss_multiplier: float = 0.0,
    aux_loss_weight: float = 1.0,
) -> LossBreakdown:
  """Computes `LossBreakdown` for one microbatch or full batch."""
  use_vocab_tiling = int(num_vocab_tiles) > 1
  out = _invoke_model_forward(model, batch, return_logits=not use_vocab_tiling)

  targets = batch["targets"]
  weights = batch.get("weights")
  if weights is None:
    seg = batch.get("segment_ids")
    weights = jnp.ones_like(targets, dtype=jnp.float32) if seg is None else (seg != 0).astype(jnp.float32)
  else:
    weights = weights.astype(jnp.float32)

  if use_vocab_tiling:
    if out.hidden_states is None:
      raise ValueError(
          "Vocabulary-tiled cross-entropy requires `CausalLMOutput.hidden_states` when `return_logits=False`."
      )
    xent_sum, z_loss_sum = compute_vocab_tiled_xent_sum(
        model,
        out.hidden_states,
        targets,
        weights,
        num_vocab_tiles=num_vocab_tiles,
        z_loss_multiplier=z_loss_multiplier,
    )
  else:
    if out.logits is None:
      raise ValueError("Dense cross-entropy requires `CausalLMOutput.logits` when `num_vocab_tiles == 1`.")
    xent_sum, z_loss_sum = compute_dense_xent_and_z_loss_sums(
        out.logits,
        targets,
        weights,
        z_loss_multiplier=z_loss_multiplier,
    )

  valid_tokens = jnp.maximum(jnp.sum(weights), 1.0)
  lm_loss = xent_sum / valid_tokens
  if out.aux_loss is not None and aux_loss_weight > 0.0:
    aux_loss = jnp.asarray(aux_loss_weight, dtype=jnp.float32) * out.aux_loss.astype(jnp.float32)
  else:
    aux_loss = jnp.zeros((), dtype=jnp.float32)

  total_loss = (xent_sum + z_loss_sum) / valid_tokens + aux_loss
  counts = None if out.router_token_counts is None else out.router_token_counts.astype(jnp.float32)
  return LossBreakdown(
      total_loss=total_loss,
      lm_loss=lm_loss,
      xent_sum=xent_sum,
      z_loss_sum=z_loss_sum,
      aux_loss=aux_loss,
      valid_tokens=valid_tokens,
      router_token_counts=counts,
  )
