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

"""Custom-VJP 2D vocab-and-token chunked FP32 cross-entropy loss for Flint."""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import ops

_ALL_AXES: tuple[str, str, str] = ("fsdp", "cp", "ep")
_MIN_TILE_FLOOR: int = 128


def _validate_lm_head_inputs(
    normed_x_local: jax.Array,
    lm_head_shard: jax.Array,
    cfg: flint_config.FlintConfig,
) -> tuple[tuple[int, ...], int, int, int]:
  """Validates `normed_x_local` and `lm_head_shard` local shapes."""
  hidden_size = cfg.model.hidden_size
  vocab_size = cfg.model.vocab_size
  d_loc = hidden_size // cfg.sharding.num_devices
  if normed_x_local.ndim < 1 or normed_x_local.shape[-1] != hidden_size:
    raise ValueError(f"Expected normed_x_local[..., {hidden_size}], got" f" {normed_x_local.shape}.")
  if lm_head_shard.shape != (d_loc, vocab_size):
    raise ValueError(f"Expected lm_head_shard shape {(d_loc, vocab_size)}, got" f" {lm_head_shard.shape}.")
  lead_shape = normed_x_local.shape[:-1]
  n_loc = math.prod(lead_shape)
  if n_loc < 1:
    raise ValueError(f"normed_x_local must contain >= 1 token, got {normed_x_local.shape}.")
  return lead_shape, n_loc, d_loc, vocab_size


def _compute_token_tile_geometry(
    n_loc: int,
    vocab_token_chunk_size: int,
) -> tuple[int, int, int]:
  """Returns `(num_token_tiles, tile_tokens, pad_tokens)`."""
  if n_loc <= vocab_token_chunk_size:
    return 1, n_loc, 0
  gcd_tile = math.gcd(n_loc, vocab_token_chunk_size)
  min_ok = min(_MIN_TILE_FLOOR, vocab_token_chunk_size)
  if gcd_tile >= min_ok:
    return n_loc // gcd_tile, gcd_tile, 0
  tile_tokens = vocab_token_chunk_size
  num_tiles = math.ceil(n_loc / tile_tokens)
  pad_tokens = num_tiles * tile_tokens - n_loc
  return num_tiles, tile_tokens, pad_tokens


def compute_logits_f32(
    normed_x_local: jax.Array,
    lm_head_shard: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    upcast_weight_f32: bool = True,
) -> jax.Array:
  """Computes local `float32[..., vocab_size]` logits by scanning vocab chunks.

  Args:
    normed_x_local: Local final-RMSNorm activations `[..., hidden_size]`.
    lm_head_shard: Local `lm_head` shard `[hidden_size // num_devices, vocab]`.
    cfg: Validated `FlintConfig`.
    upcast_weight_f32: When `True` (default), gathers `lm_head` chunks in
      `float32` and contracts `normed_x_local.astype(float32)` at
      `Precision.HIGHEST`, matching `TorchFlintForCausalLM` (`F.linear` in
      FP32). When `False`, uses `cfg.train.compute_jnp_dtype`.

  Returns:
    Local `float32` logits with shape `(*normed_x_local.shape[:-1], vocab)`.
  """
  lead_shape, n_loc, d_loc, vocab_size = _validate_lm_head_inputs(normed_x_local, lm_head_shard, cfg)
  num_chunks = cfg.kernels.num_vocab_chunks
  v_chunk = vocab_size // num_chunks
  comp_dt = cfg.train.compute_jnp_dtype

  if upcast_weight_f32:
    h_mat = normed_x_local.astype(jnp.float32).reshape(n_loc, cfg.model.hidden_size)
    gather_dt = jnp.dtype(jnp.float32)
    prec = jax.lax.Precision.HIGHEST
  else:
    h_mat = normed_x_local.astype(comp_dt).reshape(n_loc, cfg.model.hidden_size)
    gather_dt = comp_dt
    prec = jax.lax.Precision.HIGHEST if comp_dt == jnp.float32 else jax.lax.Precision.DEFAULT

  w_chunks_l = jnp.swapaxes(lm_head_shard.reshape(d_loc, num_chunks, v_chunk), 0, 1)

  def _logit_chunk_step(carry: None, w_c_l: jax.Array) -> tuple[None, jax.Array]:
    w_c = ops.gather_dense_weight(w_c_l, cfg, sharded_axis=0, dtype=gather_dt)
    logits_c = jnp.matmul(
        h_mat,
        w_c,
        precision=prec,
        preferred_element_type=jnp.float32,
    )
    return carry, logits_c

  _, logits_c_n_v = jax.lax.scan(_logit_chunk_step, None, w_chunks_l)
  logits_n_v = jnp.swapaxes(logits_c_n_v, 0, 1).reshape(*lead_shape, vocab_size)
  return logits_n_v.astype(jnp.float32)


def dense_ce_loss_sum(
    normed_x_local: jax.Array,
    lm_head_shard: jax.Array,
    targets_local: jax.Array,
    loss_mask_local: jax.Array,
    cfg: flint_config.FlintConfig,
) -> jax.Array:
  """Computes the reference unchunked local FP32 cross-entropy loss sum.

  Used in parity tests to verify exact forward and backward agreement with
  `chunked_lm_head_ce_sum`.

  Args:
    normed_x_local: Local final-RMSNorm activations `[..., hidden_size]`.
    lm_head_shard: Local `lm_head` shard `[hidden_size // num_devices, vocab]`.
    targets_local: Local integer target IDs matching
      `normed_x_local.shape[:-1]`.
    loss_mask_local: Local validity mask (`> 0` marks valid tokens).
    cfg: Validated `FlintConfig`.

  Returns:
    Scalar `float32` unnormalized sum of token NLL over local valid tokens.

  Raises:
    ValueError: If `targets_local` or `loss_mask_local` shapes mismatch.
  """
  lead_shape, _, _, _ = _validate_lm_head_inputs(normed_x_local, lm_head_shard, cfg)
  if targets_local.shape != lead_shape or loss_mask_local.shape != lead_shape:
    raise ValueError(
        f"Expected targets/loss_mask shape {lead_shape}, got" f" {targets_local.shape} and {loss_mask_local.shape}."
    )
  comp_dt = cfg.train.compute_jnp_dtype
  prec = jax.lax.Precision.HIGHEST if comp_dt == jnp.float32 else jax.lax.Precision.DEFAULT
  w_full = ops.gather_dense_weight(lm_head_shard, cfg, sharded_axis=0, bwd_use_sc=False)
  logits_f32 = jnp.matmul(
      normed_x_local.astype(comp_dt),
      w_full,
      precision=prec,
      preferred_element_type=jnp.float32,
  )
  valid = loss_mask_local > 0
  msk_f32 = valid.astype(jnp.float32)
  tgt_i32 = targets_local.astype(jnp.int32)
  safe_t = jnp.where(valid & (tgt_i32 >= 0), tgt_i32, 0)
  log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
  picked_lp = jnp.take_along_axis(log_probs, safe_t[..., None], axis=-1)[..., 0]
  return jnp.sum(-picked_lp * msk_f32)


def _chunked_ce_forward_core(
    cfg: flint_config.FlintConfig,
    x_2d: jax.Array,
    lm_head_shard: jax.Array,
    targets_1d: jax.Array,
    valid_mask_1d: jax.Array,
) -> tuple[jax.Array, tuple[Any, ...]]:
  """Runs 2D-tiled streaming `logsumexp` + target gather in `float32`."""
  n_loc, hidden_size = x_2d.shape
  d_loc, vocab_size = lm_head_shard.shape
  num_chunks = cfg.kernels.num_vocab_chunks
  v_chunk = vocab_size // num_chunks
  comp_dt = cfg.train.compute_jnp_dtype
  prec = jax.lax.Precision.HIGHEST if comp_dt == jnp.float32 else jax.lax.Precision.DEFAULT

  num_tiles, tile_tokens, pad_tokens = _compute_token_tile_geometry(n_loc, cfg.kernels.vocab_token_chunk_size)
  valid_bool = valid_mask_1d > 0
  msk_1d = valid_bool.astype(jnp.float32)
  safe_t_1d = jnp.where(valid_bool & (targets_1d >= 0), targets_1d, 0).astype(jnp.int32)

  if pad_tokens > 0:
    h_work = jnp.pad(x_2d, ((0, pad_tokens), (0, 0)))
    t_work = jnp.pad(safe_t_1d, (0, pad_tokens))
    msk_work = jnp.pad(msk_1d, (0, pad_tokens))
  else:
    h_work = x_2d
    t_work = safe_t_1d
    msk_work = msk_1d

  h_tiles = h_work.reshape(num_tiles, tile_tokens, hidden_size)
  t_tiles = t_work.reshape(num_tiles, tile_tokens)
  msk_tiles = msk_work.reshape(num_tiles, tile_tokens)

  w_chunks_l = jnp.swapaxes(lm_head_shard.reshape(d_loc, num_chunks, v_chunk), 0, 1)
  m_init = jnp.full((num_tiles, tile_tokens), -jnp.inf, dtype=jnp.float32)
  s_init = jnp.zeros((num_tiles, tile_tokens), dtype=jnp.float32)
  picked_init = jnp.zeros((num_tiles, tile_tokens), dtype=jnp.float32)

  def _vocab_fwd_step(
      carry: tuple[jax.Array, jax.Array, jax.Array],
      c_and_w: tuple[jax.Array, jax.Array],
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    m_prev, s_prev, picked_prev = carry
    c_idx, w_c_l = c_and_w
    w_c = ops.gather_dense_weight(w_c_l, cfg, sharded_axis=0)
    chunk_start = c_idx * v_chunk

    def _tile_fwd(
        tile_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      h_ti, t_ti, m_ti, s_ti, p_ti = tile_in
      logits_ti = jnp.matmul(
          h_ti,
          w_c,
          precision=prec,
          preferred_element_type=jnp.float32,
      )
      m_c = jnp.max(logits_ti, axis=-1)
      m_new = jnp.maximum(m_ti, m_c)
      exp_prev = jnp.where(jnp.isfinite(m_ti), jnp.exp(m_ti - m_new), 0.0)
      s_c = jnp.sum(jnp.exp(logits_ti - m_new[:, None]), axis=-1)
      s_new = s_ti * exp_prev + s_c

      rel_t = t_ti - chunk_start
      in_chunk = (rel_t >= 0) & (rel_t < v_chunk)
      safe_rel = jnp.where(in_chunk, rel_t, 0)
      chunk_pick = jnp.take_along_axis(logits_ti, safe_rel[:, None], axis=-1)[:, 0]
      p_new = p_ti + jnp.where(in_chunk, chunk_pick, 0.0)
      return m_new, s_new, p_new

    if num_tiles == 1:
      m0, s0, p0 = _tile_fwd((h_tiles[0], t_tiles[0], m_prev[0], s_prev[0], picked_prev[0]))
      return (m0[None, :], s0[None, :], p0[None, :]), None

    _, (m_next, s_next, p_next) = jax.lax.scan(
        lambda _, inp: (None, _tile_fwd(inp)),
        None,
        (h_tiles, t_tiles, m_prev, s_prev, picked_prev),
    )
    return (m_next, s_next, p_next), None

  (m_fin, s_fin, picked_fin), _ = jax.lax.scan(
      _vocab_fwd_step,
      (m_init, s_init, picked_init),
      (jnp.arange(num_chunks, dtype=jnp.int32), w_chunks_l),
  )
  lse_tiles = m_fin + jnp.log(s_fin)
  loss_sum = jnp.sum((lse_tiles - picked_fin) * msk_tiles)
  residuals = (h_tiles, lm_head_shard, t_tiles, msk_tiles, lse_tiles)
  return loss_sum, residuals


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _chunked_ce_2d(
    cfg: flint_config.FlintConfig,
    x_2d: jax.Array,
    lm_head_shard: jax.Array,
    targets_1d: jax.Array,
    valid_mask_1d: jax.Array,
) -> jax.Array:
  """Custom-VJP 2D-tiled CE sum over flattened `[n_loc, hidden_size]`."""
  loss_sum, _ = _chunked_ce_forward_core(cfg, x_2d, lm_head_shard, targets_1d, valid_mask_1d)
  return loss_sum


def _chunked_ce_2d_fwd(
    cfg: flint_config.FlintConfig,
    x_2d: jax.Array,
    lm_head_shard: jax.Array,
    targets_1d: jax.Array,
    valid_mask_1d: jax.Array,
) -> tuple[jax.Array, tuple[Any, ...]]:
  """Forward pass saving only `(h_tiles, lm_head_shard, t, msk, lse)`."""
  return _chunked_ce_forward_core(cfg, x_2d, lm_head_shard, targets_1d, valid_mask_1d)


def _chunked_ce_2d_bwd(
    cfg: flint_config.FlintConfig,
    residuals: tuple[Any, ...],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None]:
  """Backward pass recomputing chunked softmax probabilities per 2D tile."""
  h_tiles, lm_head_shard, t_tiles, msk_tiles, lse_tiles = residuals
  num_tiles, tile_tokens, hidden_size = h_tiles.shape
  d_loc, vocab_size = lm_head_shard.shape
  num_chunks = cfg.kernels.num_vocab_chunks
  v_chunk = vocab_size // num_chunks
  comp_dt = cfg.train.compute_jnp_dtype
  prec = jax.lax.Precision.HIGHEST if comp_dt == jnp.float32 else jax.lax.Precision.DEFAULT

  w_chunks_l = jnp.swapaxes(lm_head_shard.reshape(d_loc, num_chunks, v_chunk), 0, 1)
  scale_tiles = g_out.astype(jnp.float32) * msk_tiles
  dh_init_f32 = jnp.zeros((num_tiles, tile_tokens, hidden_size), jnp.float32)
  col_idx = jnp.arange(v_chunk, dtype=jnp.int32)[None, :]

  def _vocab_bwd_step(
      dh_acc_f32: jax.Array,
      c_and_w: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    c_idx, w_c_l = c_and_w
    w_c = ops.gather_dense_weight(w_c_l, cfg, sharded_axis=0)
    w_c_t = w_c.T
    chunk_start = c_idx * v_chunk
    dw_init_f32 = jnp.zeros((hidden_size, v_chunk), dtype=jnp.float32)

    def _tile_bwd(
        dw_acc_f32: jax.Array,
        tile_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
      h_ti, t_ti, lse_ti, scale_ti, dh_prev_ti = tile_in
      logits_ti = jnp.matmul(
          h_ti,
          w_c,
          precision=prec,
          preferred_element_type=jnp.float32,
      )
      probs = jnp.exp(logits_ti - lse_ti[:, None])
      rel_t = t_ti - chunk_start
      diff_probs = jnp.where(rel_t[:, None] == col_idx, probs - 1.0, probs)
      grad_logits = (diff_probs * scale_ti[:, None]).astype(comp_dt)
      dh_delta = jnp.matmul(
          grad_logits,
          w_c_t,
          precision=prec,
          preferred_element_type=jnp.float32,
      )
      dw_delta = jnp.matmul(
          h_ti.T,
          grad_logits,
          precision=prec,
          preferred_element_type=jnp.float32,
      )
      return dw_acc_f32 + dw_delta, dh_prev_ti + dh_delta

    if num_tiles == 1:
      dw_full_f32, dh_next_0 = _tile_bwd(
          dw_init_f32,
          (
              h_tiles[0],
              t_tiles[0],
              lse_tiles[0],
              scale_tiles[0],
              dh_acc_f32[0],
          ),
      )
      dh_next_f32 = dh_next_0[None, :, :]
    else:
      dw_full_f32, dh_next_f32 = jax.lax.scan(
          _tile_bwd,
          dw_init_f32,
          (h_tiles, t_tiles, lse_tiles, scale_tiles, dh_acc_f32),
      )

    if cfg.sharding.num_devices > 1:
      dw_shard_f32 = jax.lax.psum_scatter(
          dw_full_f32,
          _ALL_AXES,
          scatter_dimension=0,
          tiled=True,
      )
    else:
      dw_shard_f32 = dw_full_f32
    return dh_next_f32, dw_shard_f32.astype(lm_head_shard.dtype)

  dh_final_f32, dw_chunks_l = jax.lax.scan(
      _vocab_bwd_step,
      dh_init_f32,
      (jnp.arange(num_chunks, dtype=jnp.int32), w_chunks_l),
  )
  _, _, pad_tokens = _compute_token_tile_geometry(num_tiles * tile_tokens, cfg.kernels.vocab_token_chunk_size)
  del pad_tokens  # Original `n_loc` slice is handled by caller if padded.
  dh_flat = dh_final_f32.reshape(num_tiles * tile_tokens, hidden_size).astype(h_tiles.dtype)
  dw_shard = jnp.swapaxes(dw_chunks_l, 0, 1).reshape(d_loc, vocab_size)
  return dh_flat, dw_shard, None, None


_chunked_ce_2d.defvjp(_chunked_ce_2d_fwd, _chunked_ce_2d_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _fused_vocab_ce(
    cfg: flint_config.FlintConfig,
    lm_head_dtype: jnp.dtype,
    x_2d: jax.Array,
    lm_head_shard: jax.Array,
    targets_1d: jax.Array,
    valid_mask_1d: jax.Array,
) -> jax.Array:
  """Single-scan token-tiled CE (`num_vocab_chunks == 1`)."""
  del lm_head_dtype
  loss_sum, _ = _chunked_ce_forward_core(cfg, x_2d, lm_head_shard, targets_1d, valid_mask_1d)
  return loss_sum


def _fused_vocab_ce_fwd(
    cfg: flint_config.FlintConfig,
    lm_head_dtype: jnp.dtype,
    x_2d: jax.Array,
    lm_head_shard: jax.Array,
    targets_1d: jax.Array,
    valid_mask_1d: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Computes CE loss + `dh` + `dw_shard` in a single forward token scan."""
  del lm_head_dtype
  n_work, hidden_size = x_2d.shape
  _, vocab_size = lm_head_shard.shape
  comp_dt = cfg.train.compute_jnp_dtype
  prec = jax.lax.Precision.HIGHEST if comp_dt == jnp.float32 else jax.lax.Precision.DEFAULT
  num_tiles, tile_tokens, _ = _compute_token_tile_geometry(n_work, cfg.kernels.vocab_token_chunk_size)
  valid_bool = valid_mask_1d > 0
  msk_1d = valid_bool.astype(jnp.float32)
  safe_t_1d = jnp.where(valid_bool & (targets_1d >= 0), targets_1d, 0).astype(jnp.int32)
  h_tiles = x_2d.reshape(num_tiles, tile_tokens, hidden_size)
  t_tiles = safe_t_1d.reshape(num_tiles, tile_tokens)
  msk_tiles = msk_1d.reshape(num_tiles, tile_tokens)

  w_full = ops.gather_dense_weight(
      jax.lax.stop_gradient(lm_head_shard),
      cfg,
      sharded_axis=0,
      bwd_use_sc=False,
  )
  w_full_t = w_full.T
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tile_tokens, vocab_size), 1)

  def _tile_step(
      carry: tuple[jax.Array, jax.Array],
      tile_in: tuple[jax.Array, jax.Array, jax.Array],
  ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    loss_acc_f32, dw_acc_f32 = carry
    h_ti, t_ti, msk_ti = tile_in
    logits_ti = jnp.matmul(
        h_ti,
        w_full,
        precision=prec,
        preferred_element_type=jnp.float32,
    )
    m_ti = jnp.max(logits_ti, axis=-1, keepdims=True)
    exp_shifted = jnp.exp(logits_ti - m_ti)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    lse_ti = m_ti[:, 0] + jnp.log(sum_exp[:, 0])
    picked_ti = jnp.take_along_axis(logits_ti, t_ti[:, None], axis=-1)[:, 0]
    tile_loss = jnp.sum((lse_ti - picked_ti) * msk_ti)

    probs = exp_shifted / sum_exp
    diff_probs = jnp.where(t_ti[:, None] == col_idx, probs - 1.0, probs)
    grad_logits = (diff_probs * msk_ti[:, None]).astype(comp_dt)
    dh_ti = jnp.matmul(
        grad_logits,
        w_full_t,
        precision=prec,
        preferred_element_type=jnp.float32,
    ).astype(x_2d.dtype)
    dw_delta = jnp.matmul(
        h_ti.T,
        grad_logits,
        precision=prec,
        preferred_element_type=jnp.float32,
    )
    return (loss_acc_f32 + tile_loss, dw_acc_f32 + dw_delta), dh_ti

  loss_init = jnp.zeros((), dtype=jnp.float32)
  dw_init = jnp.zeros((hidden_size, vocab_size), dtype=jnp.float32)
  if num_tiles == 1:
    (loss_sum, dw_full_f32), dh_0 = _tile_step(
        (loss_init, dw_init),
        (h_tiles[0], t_tiles[0], msk_tiles[0]),
    )
    dh_flat = dh_0
  else:
    (loss_sum, dw_full_f32), dh_tiles = jax.lax.scan(
        _tile_step,
        (loss_init, dw_init),
        (h_tiles, t_tiles, msk_tiles),
    )
    dh_flat = dh_tiles.reshape(num_tiles * tile_tokens, hidden_size)
  if cfg.sharding.num_devices > 1:
    dw_shard_f32 = jax.lax.psum_scatter(
        dw_full_f32,
        _ALL_AXES,
        scatter_dimension=0,
        tiled=True,
    )
  else:
    dw_shard_f32 = dw_full_f32
  return loss_sum, (dh_flat, dw_shard_f32)


def _fused_vocab_ce_bwd(
    cfg: flint_config.FlintConfig,
    lm_head_dtype: jnp.dtype,
    residuals: tuple[jax.Array, jax.Array],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None]:
  """Scales precomputed `(dh_flat, dw_shard_f32)` by replicated `g_out`."""
  del cfg
  dh_flat, dw_shard_f32 = residuals
  g_f32 = g_out.astype(jnp.float32)
  dh_scaled = (dh_flat.astype(jnp.float32) * g_f32).astype(dh_flat.dtype)
  dw_scaled = (dw_shard_f32 * g_f32).astype(lm_head_dtype)
  return dh_scaled, dw_scaled, None, None


_fused_vocab_ce.defvjp(_fused_vocab_ce_fwd, _fused_vocab_ce_bwd)


def chunked_lm_head_ce_sum(
    normed_x_local: jax.Array,
    lm_head_shard: jax.Array,
    targets_local: jax.Array,
    loss_mask_local: jax.Array,
    cfg: flint_config.FlintConfig,
) -> jax.Array:
  """Computes local FP32 CE sum via 2D vocab-and-token chunked `custom_vjp`.

  Never materializes `[N_loc, vocab_size]` logits or the full `[hidden_size,
  vocab_size]` gathered `lm_head` at once, saving only `float32[N_loc]`
  `logsumexp` across the forward-backward boundary.

  Args:
    normed_x_local: Local final-RMSNorm activations `[..., hidden_size]`.
    lm_head_shard: Local `lm_head` shard `[hidden_size // num_devices, vocab]`.
    targets_local: Local integer target IDs matching
      `normed_x_local.shape[:-1]`.
    loss_mask_local: Local validity mask (`> 0` marks valid tokens).
    cfg: Validated `FlintConfig`.

  Returns:
    Scalar `float32` unnormalized sum of token NLL over local valid tokens.

  Raises:
    ValueError: If input shapes violate config or sharding invariants.
  """
  lead_shape, n_loc, _, _ = _validate_lm_head_inputs(normed_x_local, lm_head_shard, cfg)
  if targets_local.shape != lead_shape or loss_mask_local.shape != lead_shape:
    raise ValueError(
        f"Expected targets/loss_mask shape {lead_shape}, got" f" {targets_local.shape} and {loss_mask_local.shape}."
    )
  hidden_size = cfg.model.hidden_size
  comp_dt = cfg.train.compute_jnp_dtype
  x_2d = normed_x_local.astype(comp_dt).reshape(n_loc, hidden_size)
  targets_1d = targets_local.reshape(n_loc).astype(jnp.int32)
  valid_1d = (loss_mask_local.reshape(n_loc) > 0).astype(jnp.int32)

  _, _, pad_tokens = _compute_token_tile_geometry(n_loc, cfg.kernels.vocab_token_chunk_size)
  if pad_tokens > 0:
    x_2d = jnp.pad(x_2d, ((0, pad_tokens), (0, 0)))
    targets_1d = jnp.pad(targets_1d, (0, pad_tokens))
    valid_1d = jnp.pad(valid_1d, (0, pad_tokens))
  if cfg.kernels.use_fused_vocab_vjp_scan:
    return _fused_vocab_ce(cfg, lm_head_shard.dtype, x_2d, lm_head_shard, targets_1d, valid_1d)
  return _chunked_ce_2d(cfg, x_2d, lm_head_shard, targets_1d, valid_1d)
