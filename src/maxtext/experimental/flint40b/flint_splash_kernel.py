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

"""Flint SplashAttention kernels with GQA head folding and dKV tuning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import json
import math
from typing import Any

import jax
from jax import ad_checkpoint
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import splash_attention_kernel as tk
from tokamax._src.ops.splash_attention import splash_attention_mask_info as mask_info_lib

LOG2E: float = tk.LOG2E
QKVLayout = tk.QKVLayout
MaskInfo = mask_info_lib.MaskInfo
MaskFunctionType = tk.MaskFunctionType

_NUM_LANES: int = tk.NUM_LANES
_NUM_SUBLANES: int = tk.NUM_SUBLANES
_NN_DIM_NUMBERS = tk.NN_DIM_NUMBERS
_NT_DIM_NUMBERS = tk.NT_DIM_NUMBERS
_VMEM_LIMIT_64_MIB: int = 64 * 1024 * 1024


def _is_fp8(x: jax.Array) -> bool:
  """Returns whether `x` uses `float8_e4m3fn`."""
  return x.dtype == jnp.float8_e4m3fn


def _unquantized_dtype(x: jax.Array) -> jnp.dtype:
  """Returns `bfloat16` for FP8 inputs and `x.dtype` otherwise."""
  return jnp.dtype(jnp.bfloat16) if _is_fp8(x) else x.dtype


def _not(x: jax.Array | bool) -> jax.Array | bool:
  """Computes logical negation on a traced array or Python boolean."""
  if isinstance(x, jax.Array):
    return jnp.logical_not(x)
  return not x


def _div(dividend: int | jax.Array, divisor: int) -> int | jax.Array:
  """Divides `dividend` by `divisor`, eliding `lax.div` when `divisor == 1`."""
  if divisor == 1:
    return dividend
  return lax.div(dividend, divisor)


def _bytes(
    x: jax.Array | np.ndarray | jax.ShapeDtypeStruct | None,
) -> int:
  """Returns the byte footprint of `x`, or 0 when `x` is `None`."""
  if x is None:
    return 0
  if jnp.issubdtype(x.dtype, jnp.floating):
    bits = jnp.finfo(x.dtype).bits
  elif jnp.issubdtype(x.dtype, jnp.integer):
    bits = jnp.iinfo(x.dtype).bits
  else:
    raise ValueError(f"Unsupported dtype: {x.dtype}")
  return math.ceil(math.prod(x.shape) * bits / 8)


def _above_diag(q_band: int, q_size: int, kv_band: int, kv_size: int) -> bool:
  """Returns whether a diagonal-block sub-tile lies strictly above `kv <= q`."""
  return kv_band * kv_size >= (q_band + 1) * q_size


def _diag_grid_pair(grid: int | tuple[int, int]) -> tuple[int, int]:
  """Normalizes a diagonal-skip grid spec to `(q_splits, kv_splits)`."""
  return (grid, grid) if isinstance(grid, int) else grid


def _make_unvmap_primitive(name: str, reduce_fn: Callable[..., jax.Array]) -> jax.extend.core.Primitive:
  """Creates a primitive that reduces `vmap` dims so `MaskInfo` stays shared."""
  prim = jax.extend.core.Primitive(name)
  prim.def_impl(lambda x: x)
  prim.def_abstract_eval(lambda x: jax.core.ShapedArray(x.shape, x.dtype))
  jax.interpreters.batching.primitive_batchers[prim] = lambda xs, bdims: (
      prim.bind(xs[0] if bdims[0] is None else reduce_fn(xs[0], axis=bdims[0])),
      None,
  )
  jax.interpreters.mlir.register_lowering(prim, jax.interpreters.mlir.lower_fun(lambda x: x, multiple_results=False))
  return prim


_unvmap_any_p = _make_unvmap_primitive("flint_splash_unvmap_any", jnp.any)


def _onehot_lookup(table: jax.Array, idx: jax.Array) -> jax.Array:
  """Returns `table[idx]` for in-range `idx` without emitting an XLA gather."""
  onehot = idx[..., None] == jnp.arange(table.shape[-1], dtype=idx.dtype)
  return jnp.sum(jnp.where(onehot, table, jnp.zeros_like(table)), axis=-1)


def _segment_block_overlap(
    segment_ids: base.SegmentIds,
    q_idx: jax.Array,
    kv_idx: jax.Array,
    *,
    q_blocks: int,
    kv_blocks: int,
    bq: int,
    bkv: int,
) -> jax.Array:
  """Returns whether each `(q_idx, kv_idx)` tile can share any segment ID."""
  q_segs = jnp.asarray(segment_ids.q).reshape(q_blocks, bq)
  kv_segs = jnp.asarray(segment_ids.kv).reshape(kv_blocks, bkv)
  q_lo = _onehot_lookup(jnp.min(q_segs, axis=-1), q_idx)
  q_hi = _onehot_lookup(jnp.max(q_segs, axis=-1), q_idx)
  kv_lo = _onehot_lookup(jnp.min(kv_segs, axis=-1), kv_idx)
  kv_hi = _onehot_lookup(jnp.max(kv_segs, axis=-1), kv_idx)
  return _unvmap_any_p.bind((q_hi >= kv_lo) & (kv_hi >= q_lo))


def _refine_mask_info_with_segments(
    mask_info: MaskInfo,
    segment_ids: base.SegmentIds,
    *,
    q_blocks: int,
    kv_blocks: int,
    bq: int,
    bkv: int,
    is_dkv: bool,
) -> tuple[
    MaskInfo,
    jax.Array | np.ndarray | None,
    jax.Array | np.ndarray | None,
]:
  """Prunes `mask_info` tiles whose Q and KV segment ID ranges are disjoint."""
  num_rows, num_cols = (kv_blocks, q_blocks) if is_dkv else (q_blocks, kv_blocks)
  if mask_info.active_rows is None:
    m_size = num_rows * num_cols
    rows, cols = jnp.unravel_index(jnp.arange(m_size, dtype=jnp.int32), (num_rows, num_cols))
    num_active = jnp.array([m_size], dtype=jnp.int32)
  else:
    rows = jnp.asarray(mask_info.active_rows, dtype=jnp.int32)
    cols = jnp.asarray(mask_info.active_cols, dtype=jnp.int32)
    m_size = rows.size
    num_active = jnp.asarray(mask_info.num_active_blocks, dtype=jnp.int32)

  block_mask = (
      jnp.full((m_size,), 2, dtype=jnp.int8)
      if mask_info.block_mask is None
      else jnp.asarray(mask_info.block_mask).reshape(m_size)
  )
  idx = jnp.arange(m_size, dtype=jnp.int32)
  valid_static = (block_mask > 0) & (idx < num_active[0])

  rows_c = jnp.clip(rows, 0, num_rows - 1)
  cols_c = jnp.clip(cols, 0, num_cols - 1)
  seg_any = _segment_block_overlap(
      segment_ids,
      cols_c if is_dkv else rows_c,
      rows_c if is_dkv else cols_c,
      q_blocks=q_blocks,
      kv_blocks=kv_blocks,
      bq=bq,
      bkv=bkv,
  )

  active = valid_static & seg_any
  row_eq = rows_c[:, None] == jnp.arange(num_rows, dtype=jnp.int32)[None, :]
  row_has_active = jnp.any(row_eq & active[:, None], axis=0)
  first_in_row = jnp.argmax(row_eq, axis=0).astype(jnp.int32)
  keep = active | (
      (is_dkv | jnp.any(active))
      & ~jnp.any(row_eq & row_has_active[None, :], axis=1)
      & (idx == jnp.sum(jnp.where(row_eq, first_in_row[None, :], 0), axis=1))
  )

  block_mask = jnp.where(active, block_mask, 0).astype(block_mask.dtype)
  mask_next = None if mask_info.mask_next is None else jnp.asarray(mask_info.mask_next).reshape(m_size)

  new_num_active = jnp.sum(keep.astype(jnp.int32), keepdims=True)
  payload = (rows_c, cols_c, block_mask) + (() if mask_next is None else (mask_next,))
  _, *payload = lax.sort(((~keep).astype(jnp.int32), *payload), num_keys=1, is_stable=True)
  is_active_slot = idx < new_num_active[0]
  rows_s, cols_s, block_mask, *mask_next_s = [jnp.where(is_active_slot, x, 0).astype(x.dtype) for x in payload]
  bounds_start, bounds_end = mask_info_lib.find_bounds(jnp.where(is_active_slot, payload[0], -1))
  row_dtype = jnp.int32 if mask_info.active_rows is None else mask_info.active_rows.dtype
  col_dtype = jnp.int32 if mask_info.active_cols is None else mask_info.active_cols.dtype
  return (
      mask_info._replace(
          active_rows=rows_s.astype(row_dtype),
          active_cols=cols_s.astype(col_dtype),
          block_mask=block_mask,
          mask_next=mask_next_s[0] if mask_next_s else None,
          num_active_blocks=new_num_active,
      ),
      bounds_start,
      bounds_end,
  )


def _generate_blockwise_dropout_mask(
    prng_key: Any,
    head_idx: jax.Array | int,
    q_block_idx: jax.Array | int,
    kv_block_idx: jax.Array | int,
    q_block_size: int,
    kv_block_size: int,
    dropout_rate: float,
    n_q_blocks: int | jax.Array,
    n_kv_blocks: int | jax.Array,
) -> jax.Array:
  """Generates the dropout mask of a single `(head, q_block, kv_block)` tile."""
  tile_id = (head_idx * n_q_blocks + q_block_idx) * n_kv_blocks + kv_block_idx
  sub_key = jax.random.fold_in(prng_key[...], tile_id)
  threshold = np.uint32(dropout_rate * (2**32))
  bits = jax.random.bits(sub_key, (q_block_size, kv_block_size), dtype=jnp.uint32)
  dropped = bits < threshold
  scale = jnp.float32(1.0 / (1.0 - dropout_rate))
  return jnp.where(dropped, 0.0, scale)


def _dropout_mask_tile(
    prng_key: Any,
    *,
    head_idx: jax.Array | int,
    q_block_idx: jax.Array | int,
    kv_block_idx: jax.Array | int,
    q_block_size: int,
    kv_block_size: int,
    canonical_q: int,
    canonical_kv: int,
    dropout_rate: float,
    n_q_blocks: int | jax.Array,
    n_kv_blocks: int | jax.Array,
) -> jax.Array:
  """Builds the `[q_block_size, kv_block_size]` dropout mask of a tile."""
  nq, rem = divmod(q_block_size, canonical_q)
  assert rem == 0, f"{q_block_size=} must be a multiple of {canonical_q=}"
  nkv, rem = divmod(kv_block_size, canonical_kv)
  assert rem == 0, f"{kv_block_size=} must be a multiple of {canonical_kv=}"

  if nq == 1 and nkv == 1:
    return _generate_blockwise_dropout_mask(
        prng_key,
        head_idx=head_idx,
        q_block_idx=q_block_idx,
        kv_block_idx=kv_block_idx,
        q_block_size=canonical_q,
        kv_block_size=canonical_kv,
        dropout_rate=dropout_rate,
        n_q_blocks=n_q_blocks,
        n_kv_blocks=n_kv_blocks,
    )

  q_base = q_block_idx * nq
  kv_base = kv_block_idx * nkv
  rows = []
  for a in range(nq):
    cols = [
        _generate_blockwise_dropout_mask(
            prng_key,
            head_idx=head_idx,
            q_block_idx=q_base + a,
            kv_block_idx=kv_base + b,
            q_block_size=canonical_q,
            kv_block_size=canonical_kv,
            dropout_rate=dropout_rate,
            n_q_blocks=n_q_blocks,
            n_kv_blocks=n_kv_blocks,
        )
        for b in range(nkv)
    ]
    rows.append(cols[0] if nkv == 1 else jnp.concatenate(cols, axis=1))
  return rows[0] if nq == 1 else jnp.concatenate(rows, axis=0)


def _check_dropout_args(
    config: tk.SplashConfig,
    prng_key: jax.Array | None,
) -> jax.Array | None:
  """Validates dropout arguments and converts `prng_key` for Pallas."""
  if not config.dropout_rate:
    return None
  if prng_key is None:
    raise ValueError("A prng_key is required when config.dropout_rate > 0; got None.")
  return pltpu.to_pallas_key(prng_key)


@dataclasses.dataclass(frozen=True, slots=True)
class FlintSplashConfig(tk.SplashConfig):
  """SplashAttention configuration with dKV Q-head folding and scheduler."""

  num_stacked_q_heads_dkv: int = 1
  use_experimental_scheduler_dkv: bool | None = None
  fwd_strip_schedule: str = "off"
  fwd_strip_hoist_q_scale: bool | None = None
  compact_lse_residual: bool = False
  dkv_head_stitch: bool = False

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.fwd_strip_schedule not in ("off", "hoisted", "per_strip", "auto"):
      raise ValueError(
          "fwd_strip_schedule must be 'off', 'hoisted', 'per_strip', or" f" 'auto', got {self.fwd_strip_schedule!r}."
      )
    if self.num_stacked_q_heads < 1:
      raise ValueError(f"num_stacked_q_heads must be >= 1, got {self.num_stacked_q_heads}.")
    if self.num_stacked_q_heads_dkv < 1:
      raise ValueError("num_stacked_q_heads_dkv must be >= 1, got" f" {self.num_stacked_q_heads_dkv}.")
    if self.dkv_head_stitch:
      if (
          self.q_layout != QKVLayout.HEAD_DIM_MINOR
          or self.k_layout != QKVLayout.HEAD_DIM_MINOR
          or self.v_layout != QKVLayout.HEAD_DIM_MINOR
      ):
        raise ValueError("dkv_head_stitch requires HEAD_DIM_MINOR layouts.")
      if self.dropout_rate or self.qk_diag_skip or getattr(self, "diag_band_skip", False):
        raise ValueError("dkv_head_stitch does not support dropout, qk_diag_skip, or" " diag_band_skip.")
    if self.num_stacked_q_heads > 1 and (self.qk_diag_skip or self.sv_diag_skip):
      raise ValueError("num_stacked_q_heads > 1 is not supported with qk_diag_skip or" " sv_diag_skip.")
    if self.num_stacked_q_heads_dkv > 1:
      if self.q_layout != QKVLayout.HEAD_DIM_MINOR:
        raise ValueError("num_stacked_q_heads_dkv > 1 requires q_layout == HEAD_DIM_MINOR.")
      if self.dropout_rate:
        raise ValueError("num_stacked_q_heads_dkv > 1 is not supported with" " dropout_rate > 0.")
      if self.qk_diag_skip:
        raise ValueError("num_stacked_q_heads_dkv > 1 is not supported with qk_diag_skip.")

  @classmethod
  def get_default(cls) -> FlintSplashConfig:
    """Returns a default `FlintSplashConfig` instance."""
    return cls(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
        fuse_reciprocal=True,
    )


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    mask_ref: Any,
    q_sequence_ref: Any,
    q_segment_ids_ref: Any,
    kv_segment_ids_ref: Any,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes: bool = True,
    mask_function: MaskFunctionType | None = None,
    has_partial_mask: bool = False,
) -> jax.Array:
  """Applies logit soft-capping, partial block masks, and segment masks."""
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks: list[jax.Array] = []
  if has_partial_mask:
    if mask_ref is not None:
      mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
      masks.append(mask.astype(jnp.bool_))
    elif mask_function is not None:
      assert q_sequence_ref is not None
      if k_in_lanes:
        assert q_sequence_ref.shape == (bq, _NUM_LANES)
        k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (bq, k_slice.size), 1)
        repeats, rem = divmod(k_slice.size, _NUM_LANES)
        assert rem == 0
        q_sequence = jnp.tile(q_sequence_ref[...], (1, repeats))
      else:
        assert q_sequence_ref.shape == (_NUM_SUBLANES, bq)
        k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (k_slice.size, bq), 0)
        q_sequence = q_sequence_ref[:1, :]
        q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

      assert q_sequence.shape == k_sequence.shape
      computed_mask = mask_function(q_sequence, k_sequence)
      if computed_mask.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("Mask function must return a boolean-valued array, but got:" f" {computed_mask.dtype}")
      masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], _NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {_NUM_LANES}")
      q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, _NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {_NUM_LANES}")
      kv_ids = jnp.tile(kv_segment_ids_ref[k_slice, :], (1, repeats))
      q_ids = q_segment_ids_ref[:1, :]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits: jax.Array) -> jax.Array:
    if attn_logits_soft_cap is not None:
      capped = jnp.tanh(logits / attn_logits_soft_cap)
      return capped * attn_logits_soft_cap
    return logits

  if masks:
    bool_masks = [m.astype(jnp.bool_) for m in masks]
    combined_mask = functools.reduce(jnp.logical_and, bool_masks)
    qk = cap_logits(qk)
    if combined_mask.ndim == 2 and qk.ndim == 3:
      combined_mask = jnp.expand_dims(combined_mask, axis=0)
    elif combined_mask.ndim == 2 and qk.ndim == 2 and combined_mask.shape[1] != qk.shape[1]:
      combined_mask = jnp.tile(combined_mask, (1, qk.shape[1] // combined_mask.shape[1]))
    qk = jnp.where(combined_mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def _use_fwd_strip_schedule(
    config: tk.SplashConfig,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    num_stacked_q_heads: int,
    has_max_logit_estimate: bool,
) -> bool:
  """Returns whether the forward kernel runs `config.fwd_strip_schedule`."""
  sched = getattr(config, "fwd_strip_schedule", "off")
  return (
      sched != "off"
      and not config.dropout_rate
      and not has_max_logit_estimate
      and not config.sv_diag_skip
      and bkv_compute % _NUM_LANES == 0
      and (bkv // bkv_compute > 1 or num_stacked_q_heads * bq > 1024)
  )


def _resolve_fwd_strip_schedule(schedule: str, num_iters: int) -> str:
  """Resolves `fwd_strip_schedule='auto'` for kv blocks of `num_iters` tiles."""
  if schedule == "auto":
    return "per_strip" if num_iters >= 4 else "hoisted"
  return schedule


def _flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref: Any,
    active_cols_ref: Any,
    mask_next_ref: Any,
    bounds_start_ref: Any,
    bounds_end_ref: Any,
    block_mask_ref: Any,
    prng_key_ref: Any,
    # Inputs
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    q_segment_ids_ref: Any,
    kv_segment_ids_ref: Any,
    sinks_ref: Any,
    mask_ref: Any,
    q_sequence_ref: Any,
    max_logit_value_ref: Any,
    # Outputs
    o_ref: Any,
    logsumexp_ref: Any,
    l_linear_ref: Any,
    max_logits_ref: Any,
    # Scratch
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    o_scratch_ref: Any,
    *q_scaled_scratch_refs: Any,
    mask_value: float,
    q_steps: int,
    kv_steps: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    num_stacked_q_heads: int,
    mask_function: MaskFunctionType | None,
    fuse_reciprocal: bool,
    config: tk.SplashConfig,
) -> None:
  """Forward Pallas kernel supporting stacked Q heads with attention sinks."""
  del mask_next_ref
  q_scaled_ref: Any = None
  if q_scaled_scratch_refs:
    (q_scaled_ref,) = q_scaled_scratch_refs
  float32 = jnp.float32
  head_dim_minor = QKVLayout.HEAD_DIM_MINOR
  dropout_rate = config.dropout_rate
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  head_dim_v_repeats = pl.cdiv(head_dim_v, _NUM_LANES)

  grid_idx = pl.program_id(1)
  h = pl.program_id(0)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) > 0
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
    i = active_rows_ref[grid_idx].astype(jnp.int32) if dropout_rate else 0
  else:
    should_not_mask = False
    should_run = True
    j = grid_idx % kv_steps
    should_initialize = j == 0
    should_write = j == kv_steps - 1
    i = grid_idx // kv_steps

  max_logit_estimate: float | jax.Array | None = config.max_logit_const
  if max_logit_value_ref is not None:
    assert num_stacked_q_heads == 1
    max_logit_estimate = max_logit_value_ref[0, h]

  if config.use_base2_exp and max_logit_estimate is not None:
    max_logit_estimate = max_logit_estimate * LOG2E

  @pl.when(should_initialize)
  def init() -> None:
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    if q_scaled_ref is not None:
      q_init = q_ref[...] if config.q_layout == head_dim_minor else q_ref[...].mT
      q_scaled_ref[...] = q_init * LOG2E

    sink: jax.Array | None = None
    if sinks_ref is not None:
      if num_stacked_q_heads == 1:
        s_0 = sinks_ref[0, h].astype(m_scratch_ref.dtype)
        sink = s_0 * LOG2E if config.use_base2_exp else s_0
      else:
        head_sinks = []
        for head in range(num_stacked_q_heads):
          s_h = sinks_ref[0, h * num_stacked_q_heads + head].astype(m_scratch_ref.dtype)
          if config.use_base2_exp:
            s_h = s_h * LOG2E
          head_sinks.append(jnp.full((1, bq, _NUM_LANES), s_h, dtype=m_scratch_ref.dtype))
        sink = jnp.concatenate(head_sinks, axis=0)

    if sinks_ref is None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is None and max_logit_estimate is not None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is not None and max_logit_estimate is None:
      assert sink is not None
      if num_stacked_q_heads == 1:
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sink)
      else:
        m_scratch_ref[...] = sink
      l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
    else:
      assert sink is not None and max_logit_estimate is not None
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      m_est = jnp.full_like(m_scratch_ref, max_logit_estimate)
      m_scratch_ref[...] = m_est
      l_scratch_ref[...] = exp(sink - m_est)

  def qk_matmul(
      q_flat: jax.Array,
      k: jax.Array,
      qk_dims: Any,
      *,
      diag: bool,
  ) -> jax.Array:
    if not diag:
      return lax.dot_general(
          q_flat,
          k,
          qk_dims,
          preferred_element_type=jnp.float32,
      )
    gq, gk = _diag_grid_pair(config.qk_diag_grid)
    sq, sk = bkv_compute // gq, bkv_compute // gk
    if config.k_layout == head_dim_minor:
      k_parts = [k[t * sk : (t + 1) * sk, :] for t in range(gk)]
    else:
      k_parts = [k[:, t * sk : (t + 1) * sk] for t in range(gk)]
    qk_rows = []
    for qi in range(gq):
      q_part = q_flat[qi * sq : (qi + 1) * sq, :]
      cols = []
      for kj in range(gk):
        if _above_diag(qi, sq, kj, sk):
          cols.append(jnp.full((sq, sk), mask_value, dtype=float32))
        else:
          cols.append(
              lax.dot_general(
                  q_part,
                  k_parts[kj],
                  qk_dims,
                  preferred_element_type=jnp.float32,
              )
          )
      qk_rows.append(jnp.concatenate(cols, axis=1))
    if q_flat.shape[0] > bkv_compute:
      qk_rows.append(
          lax.dot_general(
              q_flat[bkv_compute:, :],
              k,
              qk_dims,
              preferred_element_type=float32,
          )
      )
    return jnp.concatenate(qk_rows, axis=0)

  def sv_matmul(
      s_flat: jax.Array,
      v: jax.Array,
      sv_dims: Any,
      *,
      has_partial_mask: bool,
      diag: bool,
  ) -> jax.Array:
    gq, gk = _diag_grid_pair(config.qk_diag_grid)
    if diag:
      sv_bands = 1
      while sv_bands * 2 <= gq and bkv_compute // (sv_bands * 2) >= 256:
        sv_bands *= 2
      band = bkv_compute // sv_bands
      o_rows = []
      for b in range(sv_bands):
        kv_end = (b + 1) * band
        if config.v_layout == head_dim_minor:
          v_pre = v[:kv_end, :]
        else:
          v_pre = v[:, :kv_end]
        o_rows.append(
            lax.dot_general(
                s_flat[b * band : (b + 1) * band, :kv_end],
                v_pre,
                sv_dims,
                preferred_element_type=float32,
            )
        )
      if s_flat.shape[0] > bkv_compute:
        o_rows.append(
            lax.dot_general(
                s_flat[bkv_compute:, :],
                v,
                sv_dims,
                preferred_element_type=float32,
            )
        )
      return jnp.concatenate(o_rows, axis=0)
    if (
        config.sv_diag_skip
        and has_partial_mask
        and num_stacked_q_heads == 1
        and config.v_layout == head_dim_minor
        and bq % gq == 0
        and bkv_compute % gk == 0
    ):
      sq = bq // gq
      sk = bkv_compute // gk
      v_parts = [v[t * sk : (t + 1) * sk, :] for t in range(gk)]
      o_rows = []
      for qi in range(gq):
        o_row: jax.Array | None = None
        for kj in range(gk):
          if _above_diag(qi, sq, kj, sk):
            continue
          part = lax.dot_general(
              s_flat[qi * sq : (qi + 1) * sq, kj * sk : (kj + 1) * sk],
              v_parts[kj],
              sv_dims,
              preferred_element_type=float32,
          )
          o_row = part if o_row is None else o_row + part
        assert o_row is not None
        o_rows.append(o_row)
      return jnp.concatenate(o_rows, axis=0)
    return lax.dot_general(s_flat, v, sv_dims, preferred_element_type=float32)

  def body(
      kv_compute_index: int | jax.Array,
      _: Any,
      has_partial_mask: bool = False,
      diag: bool = False,
  ) -> None:
    r0 = kv_compute_index * bkv_compute if diag else 0
    assert isinstance(r0, int)
    nq = bq - r0
    rows = (slice(None), slice(r0, bq), slice(None)) if diag else ...
    row_view = lambda ref: ref.at[r0:bq] if diag and ref is not None else ref
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[rows], l_scratch_ref[rows]
    assert m_prev.shape == (num_stacked_q_heads, nq, _NUM_LANES)
    assert l_prev.shape == (num_stacked_q_heads, nq, _NUM_LANES)

    if config.q_layout == head_dim_minor:
      q = q_ref[rows]
    else:
      q = q_ref[(slice(None), slice(None), slice(r0, bq)) if diag else ...].mT
    if config.use_base2_exp and not config.q_prescaled_log2e:
      if q.dtype == jnp.float8_e4m3fn:
        q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
      else:
        q = q * LOG2E

    head_dim_qk = q.shape[-1]
    q_flat = q.reshape((num_stacked_q_heads * nq, head_dim_qk))

    if config.k_layout == head_dim_minor:
      k = k_ref[slice_k, :]
      qk_dims = _NT_DIM_NUMBERS
    else:
      k = k_ref[:, slice_k]
      qk_dims = _NN_DIM_NUMBERS

    qk_flat = qk_matmul(q_flat, k, qk_dims, diag=diag)
    qk = qk_flat.reshape((num_stacked_q_heads, nq, bkv_compute))

    qk = _apply_mask_and_soft_cap(
        qk,
        mask_value,
        row_view(mask_ref),
        row_view(q_sequence_ref),
        row_view(q_segment_ids_ref),
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=j * bkv + kv_compute_index * bkv_compute,
        bq=nq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )

    bkv_repeats, rem = divmod(bkv_compute, _NUM_LANES)
    if rem != 0:
      raise NotImplementedError(f"{bkv_compute=} should be a multiple of {_NUM_LANES}")

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[..., None]
      assert m_curr.shape == (num_stacked_q_heads, nq, 1)
      m_next = jnp.maximum(m_prev, m_curr)
      assert m_next.shape == (num_stacked_q_heads, nq, _NUM_LANES)
      s_curr = exp(qk - jnp.tile(m_next, (1, 1, bkv_repeats)))
    else:
      m_next = None
      s_curr = exp(qk - max_logit_estimate)
    assert s_curr.shape == (num_stacked_q_heads, nq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0, 1))
    assert l_curr.shape == (num_stacked_q_heads, nq, _NUM_LANES)

    if max_logit_estimate is None:
      assert m_next is not None
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      m_scratch_ref[rows], l_scratch_ref[rows] = m_next, l_next
    else:
      alpha = None
      l_scratch_ref[rows] = l_curr + l_prev

    if dropout_rate:
      global_kv_block_idx = j * (bkv // bkv_compute) + kv_compute_index
      canonical_q = config.active_dropout_block_q
      canonical_kv = config.active_dropout_block_kv
      n_q_blocks = q_steps * (bq // canonical_q)
      n_kv_blocks = kv_steps * (bkv // canonical_kv)
      head_masks = []
      for head in range(num_stacked_q_heads):
        head_mask = _dropout_mask_tile(
            prng_key_ref,
            head_idx=h * num_stacked_q_heads + head,
            q_block_idx=i,
            kv_block_idx=global_kv_block_idx,
            q_block_size=bq,
            kv_block_size=bkv_compute,
            canonical_q=canonical_q,
            canonical_kv=canonical_kv,
            dropout_rate=dropout_rate,
            n_q_blocks=n_q_blocks,
            n_kv_blocks=n_kv_blocks,
        )
        head_masks.append(head_mask[r0:, :] if diag else head_mask)
      dropout_scale = jnp.stack(head_masks, axis=0)
      s_curr = s_curr * dropout_scale

    s_curr_flat = s_curr.reshape((num_stacked_q_heads * nq, bkv_compute))

    if config.v_layout == head_dim_minor:
      v = v_ref[slice_k, :]
      sv_dims = _NN_DIM_NUMBERS
    else:
      v = v_ref[:, slice_k]
      sv_dims = _NT_DIM_NUMBERS

    o_curr_flat = sv_matmul(s_curr_flat, v, sv_dims, has_partial_mask=has_partial_mask, diag=diag)
    o_curr = o_curr_flat.reshape((num_stacked_q_heads, nq, head_dim_v))

    if max_logit_estimate is None:
      assert alpha is not None
      alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[rows] = alpha_o * o_scratch_ref[rows] + o_curr
    else:
      o_scratch_ref[rows] = o_scratch_ref[rows] + o_curr

  assert bkv % bkv_compute == 0
  num_iters = k_ref.shape[0 if config.k_layout == head_dim_minor else 1] // bkv_compute
  use_diag = config.qk_diag_skip and block_mask_ref is not None and num_stacked_q_heads == 1 and bq == bkv
  use_strip_schedule = _use_fwd_strip_schedule(
      config,
      bq=bq,
      bkv=num_iters * bkv_compute,
      bkv_compute=bkv_compute,
      num_stacked_q_heads=num_stacked_q_heads,
      has_max_logit_estimate=max_logit_estimate is not None,
  )
  if q_scaled_ref is not None:
    assert use_strip_schedule and config.use_base2_exp
    assert not config.q_prescaled_log2e
    assert q_scaled_ref.shape[:2] == (num_stacked_q_heads, bq)

  def run_strip_schedule(has_partial_mask: bool) -> None:
    """Runs online softmax of one KV block across independent Q strips."""
    raw_sched = getattr(config, "fwd_strip_schedule", "off")
    schedule = _resolve_fwd_strip_schedule(raw_sched, num_iters)
    max_strips, min_strip_rows = 4, 1024
    h_strips = 1
    for cand_h in range(min(num_stacked_q_heads, max_strips), 0, -1):
      if num_stacked_q_heads % cand_h != 0:
        continue
      rows_per_h = (num_stacked_q_heads // cand_h) * bq
      if rows_per_h >= min_strip_rows and rows_per_h % 256 == 0:
        h_strips = cand_h
        break
    rem_strips = max(1, max_strips // h_strips)
    q_strips = 1
    while (
        q_strips * 2 <= rem_strips
        and bq % (q_strips * 2) == 0
        and (bq // (q_strips * 2)) % _NUM_SUBLANES == 0
        and ((num_stacked_q_heads // h_strips) * (bq // (q_strips * 2))) >= min_strip_rows
        and ((num_stacked_q_heads // h_strips) * (bq // (q_strips * 2))) % 256 == 0
    ):
      q_strips *= 2
    h_step = num_stacked_q_heads // h_strips
    q_step = bq // q_strips
    n_strips = h_strips * q_strips
    bkv_repeats = bkv_compute // _NUM_LANES
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    qk_dims = _NT_DIM_NUMBERS if config.k_layout == head_dim_minor else _NN_DIM_NUMBERS
    sv_dims = _NN_DIM_NUMBERS if config.v_layout == head_dim_minor else _NT_DIM_NUMBERS

    def strip_q_rows(g: int) -> slice:
      if q_strips == 1:
        return slice(None)
      qi = g % q_strips
      return slice(qi * q_step, (qi + 1) * q_step)

    def strip_slices(g: int) -> tuple[Any, Any]:
      if n_strips == 1:
        return ..., ...
      hi = g // q_strips
      hs = slice(hi * h_step, (hi + 1) * h_step) if h_strips > 1 else slice(None)
      qs = strip_q_rows(g)
      rows_s = (hs, qs, slice(None))
      if config.q_layout == head_dim_minor:
        return rows_s, rows_s
      return rows_s, (hs, slice(None), qs)

    def load_scaled_q_flat(g: int) -> jax.Array:
      rows_s, q_rows_s = strip_slices(g)
      if q_scaled_ref is not None:
        q = q_scaled_ref[rows_s]
        return q.reshape((h_step * q_step, q.shape[-1]))
      if config.q_layout == head_dim_minor:
        q = q_ref[q_rows_s]
      else:
        q = q_ref[q_rows_s].mT
      if config.use_base2_exp and not config.q_prescaled_log2e:
        if q.dtype == jnp.float8_e4m3fn:
          q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
        else:
          q = q * LOG2E
      return q.reshape((h_step * q_step, q.shape[-1]))

    def load_k(c: int) -> jax.Array:
      sk = pl.ds(c * bkv_compute, bkv_compute)
      return k_ref[sk, :] if config.k_layout == head_dim_minor else k_ref[:, sk]

    def load_v(c: int) -> jax.Array:
      sk = pl.ds(c * bkv_compute, bkv_compute)
      return v_ref[sk, :] if config.v_layout == head_dim_minor else v_ref[:, sk]

    def build_mask(c: int, qs: slice, n_rows: int) -> jax.Array | None:
      sk = pl.ds(c * bkv_compute, bkv_compute)
      masks: list[jax.Array] = []
      if has_partial_mask:
        if mask_ref is not None:
          masks.append(mask_ref[qs, sk].astype(jnp.bool_))
        elif mask_function is not None:
          k_seq = (j * bkv + c * bkv_compute) + jax.lax.broadcasted_iota(jnp.int32, (n_rows, bkv_compute), 1)
          q_seq = jnp.tile(q_sequence_ref[qs, :], (1, bkv_repeats))
          computed_mask = mask_function(q_seq, k_seq)
          if computed_mask.dtype != jnp.dtype(jnp.bool_):
            raise ValueError("Mask function must return a boolean-valued array, but got:" f" {computed_mask.dtype}")
          masks.append(computed_mask)
      if q_segment_ids_ref is not None:
        kv_ids = kv_segment_ids_ref[:1, sk]
        q_ids = jnp.tile(q_segment_ids_ref[qs, :], (1, bkv_repeats))
        masks.append((q_ids == kv_ids).astype(jnp.bool_))
      if not masks:
        return None
      return jnp.expand_dims(functools.reduce(jnp.logical_and, masks), axis=0)

    def compute_qk(q_flat: jax.Array, k: jax.Array, mask: jax.Array | None, g: int) -> jax.Array:
      qk = lax.dot_general(q_flat, k, qk_dims, preferred_element_type=float32)
      qk = qk.reshape((h_step, q_step, bkv_compute))
      if attn_logits_soft_cap is not None:
        qk = jnp.tanh(qk / attn_logits_soft_cap) * attn_logits_soft_cap
      if mask is not None:
        if mask.shape[1] != q_step:
          qi = g % q_strips
          mask = mask[:, qi * q_step : (qi + 1) * q_step, :]
        qk = jnp.where(mask, qk, mask_value)
      return qk

    def softmax_and_sv(
        qk: jax.Array,
        m_prev: jax.Array,
        l_prev: jax.Array,
        o_prev: jax.Array,
        v: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      m_curr = qk.max(axis=-1)[..., None]
      m_next = jnp.maximum(m_prev, m_curr)
      s_curr = exp(qk - jnp.tile(m_next, (1, 1, bkv_repeats)))
      l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0, 1))
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      s_curr_flat = s_curr.reshape((h_step * q_step, bkv_compute))
      o_curr_flat = lax.dot_general(s_curr_flat, v, sv_dims, preferred_element_type=float32)
      o_curr = o_curr_flat.reshape((h_step, q_step, head_dim_v))
      alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))
      alpha_o = alpha_o[..., : o_prev.shape[-1]]
      o_next = alpha_o * o_prev + o_curr
      return m_next, l_next, o_next

    def run_strip(
        g: int,
        get_k: Callable[[int], jax.Array],
        get_v: Callable[[int], jax.Array],
        get_mask: Callable[[int], jax.Array | None],
    ) -> None:
      rows_s, _ = strip_slices(g)
      q_flat = load_scaled_q_flat(g)
      m_s = m_scratch_ref[rows_s]
      l_s = l_scratch_ref[rows_s]
      o_s = o_scratch_ref[rows_s]
      for c in range(num_iters):
        qk = compute_qk(q_flat, get_k(c), get_mask(c), g)
        m_s, l_s, o_s = softmax_and_sv(qk, m_s, l_s, o_s, get_v(c))
      m_scratch_ref[rows_s] = m_s
      l_scratch_ref[rows_s] = l_s
      o_scratch_ref[rows_s] = o_s

    if schedule == "per_strip":
      for g in range(n_strips):
        qs = strip_q_rows(g)
        run_strip(g, load_k, load_v, lambda c, qs=qs: build_mask(c, qs, q_step))
    else:
      k_tiles = [load_k(c) for c in range(num_iters)]
      v_tiles = [load_v(c) for c in range(num_iters)]
      mask_tiles = [build_mask(c, slice(None), bq) for c in range(num_iters)]
      for g in range(n_strips):
        run_strip(
            g,
            k_tiles.__getitem__,
            v_tiles.__getitem__,
            mask_tiles.__getitem__,
        )

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _run_unmasked() -> None:
    if use_strip_schedule:
      run_strip_schedule(has_partial_mask=False)
    else:
      lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
  def _run_partial() -> None:
    if use_diag:
      for c in range(num_iters):
        body(c, None, has_partial_mask=True, diag=True)
    elif use_strip_schedule:
      run_strip_schedule(has_partial_mask=True)
    else:
      lax.fori_loop(
          0,
          num_iters,
          functools.partial(body, has_partial_mask=True),
          None,
          unroll=True,
      )

  def _store_scalar_stat(ref: Any, val: jax.Array) -> None:
    val = val.astype(ref.dtype)
    if ref.shape == (num_stacked_q_heads, bq, _NUM_LANES):
      ref[...] = val
    elif ref.ndim == 2:
      assert ref.shape == (num_stacked_q_heads, bq)
      assert num_stacked_q_heads % _NUM_SUBLANES == 0
      sublane_idx = jax.lax.broadcasted_iota(jnp.int32, (_NUM_SUBLANES, bq), 0)
      blocks = []
      for blk_start in range(0, num_stacked_q_heads, _NUM_SUBLANES):
        packed = jnp.broadcast_to(val[blk_start].T[:1, :], (_NUM_SUBLANES, bq))
        for h_off in range(1, _NUM_SUBLANES):
          packed = jnp.where(
              sublane_idx == h_off,
              jnp.broadcast_to(val[blk_start + h_off].T[:1, :], (_NUM_SUBLANES, bq)),
              packed,
          )
        blocks.append(packed)
      ref[...] = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, 0)
    else:
      assert ref.shape == (num_stacked_q_heads, _NUM_SUBLANES, bq)
      heads = [
          jnp.expand_dims(jnp.broadcast_to(val[h_i].T[:1, :], (_NUM_SUBLANES, bq)), axis=0)
          for h_i in range(num_stacked_q_heads)
      ]
      ref[...] = heads[0] if len(heads) == 1 else jnp.concatenate(heads, axis=0)

  @pl.when(should_write)
  def end() -> None:
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    if fuse_reciprocal:
      l_inv = jnp.tile(jnp.where(l == 0.0, 0.0, 1.0 / l), (1, 1, head_dim_v_repeats))
      l_inv = l_inv[..., : o_scratch_ref.shape[-1]]
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    else:
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if logsumexp_ref is not None:
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m + log(l)
      _store_scalar_stat(logsumexp_ref, logsumexp)
    if l_linear_ref is not None:
      _store_scalar_stat(l_linear_ref, l)
    if max_logits_ref is not None:
      _store_scalar_stat(max_logits_ref, m)


def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: tk.SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
    prng_key: jax.Array | None = None,
    return_max_logits: bool = False,
) -> base.SplashCustomReturnType:
  """Launches the forward SplashAttention kernel."""
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  assert bkv_compute is not None
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals
  num_stacked_q_heads = config.num_stacked_q_heads
  prng_key = _check_dropout_args(config, prng_key)

  if num_stacked_q_heads < 1:
    raise ValueError(f"num_stacked_q_heads must be >= 1, got {num_stacked_q_heads}.")
  if num_stacked_q_heads > 1 and (config.qk_diag_skip or config.sv_diag_skip):
    raise ValueError("num_stacked_q_heads > 1 is not supported with qk_diag_skip or" " sv_diag_skip.")
  if num_stacked_q_heads > 1 and max_logit_value is not None:
    raise ValueError("Stacked heads are not supported with max_logit_value.")

  if is_mqa:
    expected_kv_rank = 2
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    num_kv_heads = k.shape[0]

  if len(k.shape) != expected_kv_rank:
    raise ValueError(f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a" f" {len(k.shape)}-dim one.")
  if k.shape[-1] != head_dim_qk:
    raise ValueError(f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:" f" {k.shape[-1]}.")
  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to divide"
        f" the number of 'query' heads ({num_q_heads})."
    )
  if num_q_heads % num_stacked_q_heads != 0:
    raise ValueError(f"{num_q_heads=} must be a multiple of {num_stacked_q_heads=}")

  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if q_heads_per_kv_head % num_stacked_q_heads != 0:
    raise ValueError(f"{q_heads_per_kv_head=} must be a multiple of {num_stacked_q_heads=}")
  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same" " leading dimensions.")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % _NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {_NUM_LANES}.")

  kv_seq_len = k.shape[-2]
  kv_steps = kv_seq_len // bkv

  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)
    assert isinstance(segment_ids.kv, jax.Array)
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(f"Invalid shape for q segment_ids: {segment_ids.q.shape}. Expected:" f" {(q_seq_len,)}")
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(f"Invalid shape for kv segment_ids: {segment_ids.kv.shape}." f" Expected: {(kv_seq_len,)}")
    mask_info, bounds_start, bounds_end = _refine_mask_info_with_segments(
        mask_info,
        segment_ids,
        q_blocks=q_seq_len // bq,
        kv_blocks=kv_steps,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
    )
  elif mask_info.active_rows is not None:
    bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
  else:
    bounds_start, bounds_end = None, None
  dynamic_grid = mask_info.active_rows is not None

  if config.max_logit_const is not None and max_logit_value is not None:
    raise ValueError(f"Only one of {config.max_logit_const=} and" f" {max_logit_value=} can be set.")
  if max_logit_value is not None:
    if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
      raise ValueError(
          "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
          f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
      )
    max_logit_value = jnp.broadcast_to(jnp.atleast_1d(max_logit_value), (num_q_heads,))

  q_layout = config.q_layout
  k_layout = config.k_layout
  v_layout = config.v_layout

  def unravel(f: Callable[..., Any]) -> Callable[..., Any]:
    def index_map(h_block: Any, grid_idx: Any, rows_ref: Any, cols_ref: Any, *_: Any) -> Any:
      if dynamic_grid:
        i = tk.to_i32(rows_ref[grid_idx])
        j = tk.to_i32(cols_ref[grid_idx])
      else:
        i = grid_idx // kv_steps
        j = grid_idx % kv_steps
      return f(h_block, i, j)

    return index_map

  def create_kv_index_map(layout: QKVLayout) -> Callable[..., Any]:
    def index_map(h_block: Any, i: Any, j: Any) -> Any:
      del i
      first_h_in_block = h_block * num_stacked_q_heads
      prefix = () if is_mqa else (_div(first_h_in_block, q_heads_per_kv_head),)
      return tk.from_head_minor((*prefix, j, 0), layout)

    return index_map

  q_index_map = unravel(lambda h_block, i, j: tk.from_head_minor((h_block, i, 0), q_layout))
  out_index_map = unravel(lambda h_block, i, j: (h_block, i, 0))
  k_index_map = unravel(create_kv_index_map(k_layout))
  v_index_map = unravel(create_kv_index_map(v_layout))

  def mask_index_map(
      h_block: Any,
      grid_idx: Any,
      rows_ref: Any,
      cols_ref: Any,
      mask_next_ref: Any = None,
      *_: Any,
  ) -> tuple[Any, int, int]:
    del h_block, rows_ref, cols_ref
    next_m = tk.to_i32(mask_next_ref[grid_idx])
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h_block, i, j: (i, 0))
  kv_segment_ids_index_map = unravel(lambda h_block, i, j: (0, j))

  in_specs: list[pl.BlockSpec | None] = [
      pl.BlockSpec(
          tk.from_head_minor((num_stacked_q_heads, bq, head_dim_qk), q_layout),
          q_index_map,
      ),
      pl.BlockSpec(
          tk.from_head_minor(
              (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
              k_layout,
          ),
          k_index_map,
      ),
      pl.BlockSpec(
          tk.from_head_minor((bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout),
          v_index_map,
      ),
  ]
  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)
    assert isinstance(segment_ids.kv, jax.Array)
    in_specs += [
        pl.BlockSpec((bq, _NUM_LANES), q_segment_ids_index_map),
        pl.BlockSpec((_NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(segment_ids.q, (q_seq_len, _NUM_LANES), (0,))
    kv_segment_ids = jax.lax.broadcast_in_dim(segment_ids.kv, (_NUM_SUBLANES, kv_seq_len), (1,))
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,), f"{sinks.shape=} != {num_q_heads=}"
    in_specs.append(
        pl.BlockSpec(
            (_NUM_SUBLANES, num_q_heads),
            lambda h, i, j, *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    )
    sinks = jnp.broadcast_to(sinks.astype(jnp.float32)[None, :], (_NUM_SUBLANES, num_q_heads))
  else:
    in_specs.append(None)

  if mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(jnp.asarray(mask_info.q_sequence), (q_seq_len, _NUM_LANES), (0,))
    in_specs.append(pl.BlockSpec((bq, _NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  if max_logit_value is not None:
    max_logit_value = jnp.broadcast_to(
        max_logit_value.astype(jnp.float32)[None, :],
        (_NUM_SUBLANES, num_q_heads),
    )
    in_specs.append(
        pl.BlockSpec(
            (_NUM_SUBLANES, num_q_heads),
            lambda *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    )
  else:
    in_specs.append(None)

  out_shapes: list[jax.ShapeDtypeStruct | None] = [
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), _unquantized_dtype(q)),
  ]
  out_specs: list[pl.BlockSpec | None] = [
      pl.BlockSpec((num_stacked_q_heads, bq, head_dim_v), out_index_map),
  ]
  need_max_logits = False
  compact_lse = bool(getattr(config, "compact_lse_residual", False))
  if save_residuals:
    need_max_logits = return_max_logits or not config.fuse_reciprocal
    if compact_lse and num_stacked_q_heads % _NUM_SUBLANES == 0 and bq % _NUM_LANES == 0:
      lse_compact_mode = "2d"
      residual_struct = jax.ShapeDtypeStruct((num_q_heads, q_seq_len), jnp.float32)
      residual_spec = pl.BlockSpec(
          (num_stacked_q_heads, bq),
          unravel(lambda h_block, i, j, *_: (h_block, i)),
      )
    elif compact_lse and bq % _NUM_LANES == 0:
      lse_compact_mode = "sublane"
      residual_struct = jax.ShapeDtypeStruct((num_q_heads, _NUM_SUBLANES, q_seq_len), jnp.float32)
      residual_spec = pl.BlockSpec(
          (num_stacked_q_heads, _NUM_SUBLANES, bq),
          unravel(lambda h_block, i, j, *_: (h_block, 0, i)),
      )
    else:
      lse_compact_mode = "lanes"
      residual_struct = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, _NUM_LANES), jnp.float32)
      residual_spec = pl.BlockSpec(
          (num_stacked_q_heads, bq, _NUM_LANES),
          unravel(lambda h_block, i, j, *_: (h_block, i, 0)),
      )
    out_shapes += [
        residual_struct if fuse_reciprocal else None,
        residual_struct if not fuse_reciprocal else None,
        residual_struct if need_max_logits else None,
    ]
    out_specs += [
        residual_spec if fuse_reciprocal else None,
        residual_spec if not fuse_reciprocal else None,
        residual_spec if need_max_logits else None,
    ]
  else:
    lse_compact_mode = "lanes"
    out_shapes += [None, None, None]
    out_specs += [None, None, None]

  kernel_name = tk.get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      is_segmented=segment_ids is not None,
      phase="fwd",
  )
  metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

  def _fwd_cost_estimate(
      q_in: jax.Array,
      k_in: jax.Array,
      v_in: jax.Array,
      q_seg: jax.Array | None,
      kv_seg: jax.Array | None,
      partial_mask_blocks: jax.Array | np.ndarray | None,
      shapes_out: Sequence[jax.ShapeDtypeStruct | None],
      mask_sparsity: float,
  ) -> pl.CostEstimate:
    n_q_heads, q_len, d_qk = q_in.shape
    kv_len, d_v = v_in.shape[-2:]
    matmul_flops = 2 * q_len * kv_len * d_qk + 2 * q_len * kv_len * d_v
    total_flops = n_q_heads * matmul_flops * mask_sparsity
    transcendentals = n_q_heads * q_len * kv_len * mask_sparsity
    inputs_ = [q_in, k_in, v_in, q_seg, kv_seg, partial_mask_blocks]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, shapes_out))
    return pl.CostEstimate(
        flops=int(total_flops),
        transcendentals=int(transcendentals),
        bytes_accessed=int(input_bytes + output_bytes),
    )

  cost_estimate = config.fwd_cost_estimate or _fwd_cost_estimate(
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      mask_info.partial_mask_blocks,
      out_shapes,
      fwd_mask_sparsity,
  )

  grid_size_h = num_q_heads // num_stacked_q_heads
  if dynamic_grid:
    assert mask_info.num_active_blocks is not None
    num_active_blocks = mask_info.num_active_blocks[0]
    grid = (grid_size_h, num_active_blocks)
    is_empty_attention_block = num_active_blocks == 0
  else:
    grid = (grid_size_h, kv_steps * (q_seq_len // bq))
    is_empty_attention_block = False

  fwd_vmem_limit_bytes = _VMEM_LIMIT_64_MIB if (num_stacked_q_heads * bq >= 4096 or bkv >= 8192) else None

  fwd_scratch_shapes: list[Any] = [
      pltpu.VMEM((num_stacked_q_heads, bq, _NUM_LANES), jnp.float32),
      pltpu.VMEM((num_stacked_q_heads, bq, _NUM_LANES), jnp.float32),
      pltpu.VMEM((num_stacked_q_heads, bq, head_dim_v), jnp.float32),
  ]
  hoist_q_scale: bool | None = getattr(config, "fwd_strip_hoist_q_scale", None)
  if hoist_q_scale is None:
    hoist_q_scale = (
        _resolve_fwd_strip_schedule(getattr(config, "fwd_strip_schedule", "off"), bkv // bkv_compute) == "per_strip"
    )
  if (
      hoist_q_scale
      and config.use_base2_exp
      and not config.q_prescaled_log2e
      and q.dtype in (jnp.bfloat16, jnp.float32)
      and _use_fwd_strip_schedule(
          config,
          bq=bq,
          bkv=bkv,
          bkv_compute=bkv_compute,
          num_stacked_q_heads=num_stacked_q_heads,
          has_max_logit_estimate=(config.max_logit_const is not None or max_logit_value is not None),
      )
  ):
    fwd_scratch_shapes.append(pltpu.VMEM((num_stacked_q_heads, bq, head_dim_qk), q.dtype))

  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        functools.partial(
            _flash_attention_kernel,
            mask_value=mask_value,
            q_steps=q_seq_len // bq,
            kv_steps=kv_steps,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            num_stacked_q_heads=num_stacked_q_heads,
            fuse_reciprocal=fuse_reciprocal,
            config=config,
            mask_function=mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=fwd_scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=fwd_vmem_limit_bytes,
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": (config.use_experimental_scheduler)},
        ),
        out_shape=out_shapes,
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(
        mask_info.active_rows,
        mask_info.active_cols,
        mask_info.mask_next,
        bounds_start,
        bounds_end,
        mask_info.block_mask,
        prng_key,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        mask_info.partial_mask_blocks,
        q_sequence,
        max_logit_value,
    )
  out, logsumexp, l_linear, max_logits = all_out

  def init_if_empty(x: jax.Array, value: float) -> jax.Array:
    if not dynamic_grid:
      return x
    return jnp.where(is_empty_attention_block, value, x)

  out = init_if_empty(out, 0.0)

  def _unpack_stat(x: jax.Array) -> jax.Array:
    if lse_compact_mode == "2d":
      return x
    if lse_compact_mode == "sublane":
      return x[:, 0, :]
    return x[..., 0]

  if save_residuals:
    if need_max_logits:
      assert max_logits is not None
      max_logits = init_if_empty(_unpack_stat(max_logits), mask_value)

    if fuse_reciprocal:
      assert logsumexp is not None
      logsumexp = init_if_empty(_unpack_stat(logsumexp), mask_value)
    else:
      assert l_linear is not None
      assert max_logits is not None
      log = jnp.log2 if config.use_base2_exp else jnp.log
      l = _unpack_stat(l_linear)
      logsumexp = max_logits + log(l)
      out = (out / l[..., None]).astype(out.dtype)
  else:
    assert fuse_reciprocal

  if config.residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(out, name=config.residual_checkpoint_name)
    if logsumexp is not None:
      logsumexp = ad_checkpoint.checkpoint_name(logsumexp, name=config.residual_checkpoint_name)
  if save_residuals:
    assert logsumexp is not None
    stats: dict[str, jax.Array] = {"logsumexp": logsumexp}
    if max_logits is not None:
      stats["max_logits"] = max_logits
    stats = jax.tree.map(jax.lax.stop_gradient, stats)
    return out, stats
  return out


def _flash_attention_dkv_kernel(
    # Prefetched inputs
    active_rows_ref: Any,
    active_cols_ref: Any,
    mask_next_ref: Any,
    bounds_start_ref: Any,
    bounds_end_ref: Any,
    block_mask_ref: Any,
    prng_key_ref: Any,
    # Inputs
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    q_segment_ids_ref: Any,
    kv_segment_ids_ref: Any,
    logsumexp_ref: Any,
    do_ref: Any,
    di_ref: Any,
    mask_ref: Any,
    q_sequence_ref: Any,
    # Aliases
    dq_alias: Any,
    dk_alias: Any,
    dv_alias: Any,
    # Outputs
    dq_ref: Any,
    dk_ref: Any,
    dv_ref: Any,
    # Scratch
    dq_scratch_ref: Any,
    dk_scratch_ref: Any,
    dv_scratch_ref: Any,
    *,
    mask_value: float,
    q_steps: int,
    kv_steps: int,
    bq: int,
    bkv_compute: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    q_heads_per_kv_head: int,
    config: tk.SplashConfig,
    num_stacked_q_heads_dkv: int = 1,
    use_vmem_dq_reduction: bool = False,
) -> None:
  """Fused backward dQ/dK/dV Pallas kernel with Q-head group folding."""
  del mask_next_ref
  head_dim_minor = QKVLayout.HEAD_DIM_MINOR
  dropout_rate = config.dropout_rate
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  should_initialize_dq: jax.Array | bool = True
  should_write_dq: jax.Array | bool = True
  if active_rows_ref is not None:
    assert bounds_start_ref is not None
    assert bounds_end_ref is not None
    grid_idx = pl.program_id(1)
    kv_index = active_rows_ref[grid_idx].astype(jnp.int32)
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    if use_vmem_dq_reduction:
      col = lambda g: active_cols_ref[g].astype(jnp.int32)
      last = pl.num_programs(1) - 1
      should_initialize_dq = jnp.logical_or(grid_idx == 0, col(jnp.maximum(grid_idx - 1, 0)) != col(grid_idx))
      should_write_dq = jnp.logical_or(
          grid_idx == last,
          col(jnp.minimum(grid_idx + 1, last)) != col(grid_idx),
      )
    if dropout_rate:
      q_head = pl.program_id(0)
      q_index = active_cols_ref[grid_idx].astype(jnp.int32)
    else:
      q_head = 0
      q_index = 0
  else:
    kv_index, q_head, q_index = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
    )
    grid_idx = (kv_index * q_steps) + q_index
    should_initialize = q_index == 0
    should_write = True if q_steps <= 2 else q_index == q_steps - 1
    if q_heads_per_kv_head > 1:
      q_head_index_per_kv_head = lax.rem(q_head, q_heads_per_kv_head)
      should_initialize = jnp.logical_and(should_initialize, q_head_index_per_kv_head == 0)
      should_write = jnp.logical_and(should_write, q_head_index_per_kv_head == q_heads_per_kv_head - 1)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) > 0
  else:
    should_not_mask = False
    should_run = True

  @pl.when(should_initialize)
  def init() -> None:
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  def body(i: int | jax.Array, _: Any, has_partial_mask: bool = False) -> None:
    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]
    if num_stacked_q_heads_dkv > 1:
      q = q.reshape((num_stacked_q_heads_dkv * bq, q.shape[-1]))
    if config.use_base2_exp and not config.q_prescaled_log2e:
      if q.dtype == jnp.float8_e4m3fn:
        scaled_q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
      else:
        scaled_q = q * LOG2E
    else:
      scaled_q = q

    def _load_kv(ref: Any, layout: QKVLayout) -> jax.Array:
      if layout == head_dim_minor:
        return ref[slice_k, :]
      return ref[:, slice_k].T

    k = _load_kv(k_ref, config.k_layout)
    v = _load_kv(v_ref, config.v_layout)
    if num_stacked_q_heads_dkv == 1:
      logsumexp = logsumexp_ref[:1, :]
      do = do_ref[...]
      di = di_ref[:1, :]
    else:
      logsumexp_parts = []
      di_parts = []
      for head in range(num_stacked_q_heads_dkv):
        logsumexp_parts.append(logsumexp_ref[head, :1, :])
        di_parts.append(di_ref[head, :1, :])
      logsumexp = jnp.concatenate(logsumexp_parts, axis=1)
      di = jnp.concatenate(di_parts, axis=1)
      do = do_ref[...].reshape((num_stacked_q_heads_dkv * bq, do_ref.shape[-1]))

    qk_dims = _NT_DIM_NUMBERS if config.q_layout == head_dim_minor else _NN_DIM_NUMBERS
    gq, gk = _diag_grid_pair(config.active_qk_diag_grid_dkv)
    sk, sq = bkv_compute // gk, bq // gq
    if (
        config.qk_diag_skip
        and config.diag_band_skip
        and num_stacked_q_heads_dkv == 1
        and gq == gk
        and has_partial_mask
        and not dropout_rate
        and (dq_scratch_ref is not None or dq_ref is None)
        and bkv_compute % gk == 0
        and bq % gq == 0
        and sk % _NUM_LANES == 0
        and sq % _NUM_LANES == 0
    ):
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      dk_dims = _NN_DIM_NUMBERS if config.q_layout == head_dim_minor else _NT_DIM_NUMBERS
      q_view = lambda ref, s: None if ref is None else ref.at[:, s]
      for ki in range(gk):
        q0, k0 = ki * sq, ki * sk
        qs = pl.ds(q0, bq - q0)
        if config.q_layout == head_dim_minor:
          q_b, scaled_q_b = q[q0:, :], scaled_q[q0:, :]
        else:
          q_b, scaled_q_b = q[:, q0:], scaled_q[:, q0:]
        k_b, v_b, do_b = k[k0 : k0 + sk, :], v[k0 : k0 + sk, :], do[q0:, :]
        qk_b = lax.dot_general(k_b, scaled_q_b, qk_dims, preferred_element_type=jnp.float32)
        p_b = exp(
            _apply_mask_and_soft_cap(
                qk_b,
                mask_value,
                q_view(mask_ref, qs),
                q_view(q_sequence_ref, qs),
                q_view(q_segment_ids_ref, qs),
                kv_segment_ids_ref,
                attn_logits_soft_cap=attn_logits_soft_cap,
                k_slice=pl.ds(i * bkv_compute + k0, sk),
                k_offset=kv_index * bkv + i * bkv_compute + k0,
                bq=bq - q0,
                k_in_lanes=False,
                mask_function=mask_function,
                has_partial_mask=True,
            )
            - logsumexp[:, q0:]
        )
        dp_b = lax.dot_general(v_b, do_b, _NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
        ds_b = (dp_b - di[:, q0:]) * p_b
        if attn_logits_soft_cap is not None:
          d = jnp.tanh(qk_b / attn_logits_soft_cap)
          ds_b = ds_b * (1 - d * d)
        kv_rows = pl.ds(i * bkv_compute + k0, sk)
        dv_scratch_ref[kv_rows, :] += lax.dot(p_b.astype(do.dtype), do_b, preferred_element_type=jnp.float32).astype(
            dv_scratch_ref.dtype
        )
        dk_b = lax.dot_general(
            ds_b.astype(do.dtype),
            scaled_q_b if config.dkv_reuse_scaled_q else q_b,
            dk_dims,
            preferred_element_type=jnp.float32,
        )
        if config.use_base2_exp and (config.dkv_reuse_scaled_q or config.q_prescaled_log2e):
          dk_b = dk_b / LOG2E
        dk_scratch_ref[kv_rows, :] += dk_b.astype(dk_scratch_ref.dtype)
        if dq_scratch_ref is not None:
          dq_b = lax.dot_general(
              ds_b.T if _is_fp8(k) else ds_b.T.astype(k.dtype),
              k_b,
              _NN_DIM_NUMBERS,
              preferred_element_type=jnp.float32,
          )
          if config.q_prescaled_log2e:
            dq_b = dq_b / LOG2E
          dq_scratch_ref[q0:, :] += dq_b
      return

    if (
        config.qk_diag_skip
        and num_stacked_q_heads_dkv == 1
        and has_partial_mask
        and bkv_compute % gk == 0
        and bq % gq == 0
    ):
      k_parts = [k[t * sk : (t + 1) * sk, :] for t in range(gk)]
      if config.q_layout == head_dim_minor:
        q_parts = [scaled_q[t * sq : (t + 1) * sq, :] for t in range(gq)]
      else:
        q_parts = [scaled_q[:, t * sq : (t + 1) * sq] for t in range(gq)]
      mm_fn = lambda kk, qq: lax.dot_general(kk, qq, qk_dims, preferred_element_type=jnp.float32)
      rows = []
      for ki in range(gk):
        cols = []
        for qj in range(gq):
          if _above_diag(qj, sq, ki, sk):
            cols.append(jnp.full((sk, sq), mask_value, dtype=jnp.float32))
          else:
            cols.append(mm_fn(k_parts[ki], q_parts[qj]))
        rows.append(jnp.concatenate(cols, axis=1))
      qk_uncapped = jnp.concatenate(rows, axis=0)
    else:
      qk_uncapped = lax.dot_general(k, scaled_q, qk_dims, preferred_element_type=jnp.float32)

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)

    dropout_scale: jax.Array | None = None
    if dropout_rate:
      global_kv_block_idx = kv_index * (bkv // bkv_compute) + i
      canonical_q = config.active_dropout_block_q
      canonical_kv = config.active_dropout_block_kv
      n_q_blocks = q_steps * (bq // canonical_q)
      n_kv_blocks = kv_steps * (bkv // canonical_kv)
      dropout_scale = _dropout_mask_tile(
          prng_key_ref,
          head_idx=q_head,
          q_block_idx=q_index,
          kv_block_idx=global_kv_block_idx,
          q_block_size=bq,
          kv_block_size=bkv_compute,
          canonical_q=canonical_q,
          canonical_kv=canonical_kv,
          dropout_rate=dropout_rate,
          n_q_blocks=n_q_blocks,
          n_kv_blocks=n_kv_blocks,
      ).T
      pr = p * dropout_scale
    else:
      pr = p

    dv = lax.dot(pr.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
    dv_scratch_ref[slice_k, :] = dv

    dp = lax.dot_general(
        v,
        do,
        _NT_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    if dropout_rate:
      assert dropout_scale is not None
      dp = dp * dropout_scale
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)
    dk_dims = _NN_DIM_NUMBERS if config.q_layout == head_dim_minor else _NT_DIM_NUMBERS
    dk = lax.dot_general(
        ds.astype(do.dtype),
        scaled_q if config.dkv_reuse_scaled_q else q,
        dk_dims,
        preferred_element_type=jnp.float32,
    )
    if config.use_base2_exp and (config.dkv_reuse_scaled_q or config.q_prescaled_log2e):
      dk = dk / LOG2E
    dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
    dk_scratch_ref[slice_k, :] = dk
    if dq_scratch_ref is not None or dq_ref is not None:
      dq = lax.dot_general(
          ds.T if _is_fp8(k) else ds.T.astype(k.dtype),
          k,
          _NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      if config.q_prescaled_log2e:
        dq = dq / LOG2E
      if num_stacked_q_heads_dkv > 1:
        dq = dq.reshape((num_stacked_q_heads_dkv, bq, dq.shape[-1]))
      if dq_scratch_ref is not None:
        dq_scratch_ref[...] += dq
      else:
        if dq_alias is not None:
          dq_ref[...] = dq_alias[...] + dq.astype(dq_ref.dtype)
        else:
          dq_ref[...] = dq.astype(dq_ref.dtype)

  if dq_scratch_ref is not None:

    @pl.when(should_initialize_dq)
    def _init_dq_scratch() -> None:
      dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  elif dq_alias is not None:
    dq_ref[...] = dq_alias[...]
  else:
    dq_ref[...] = jnp.zeros_like(dq_ref)

  num_iters = k_ref.shape[0 if config.k_layout is head_dim_minor else 1] // bkv_compute

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _run_unmasked() -> None:
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
  def _run_partial() -> None:
    lax.fori_loop(
        0,
        num_iters,
        functools.partial(body, has_partial_mask=True),
        None,
        unroll=True,
    )

  if dq_scratch_ref is not None:

    @pl.when(should_write_dq)
    def _write_dq_scratch() -> None:
      if dq_alias is not None:
        dq_ref[...] = dq_alias[...] + dq_scratch_ref[...].astype(dq_ref.dtype)
      else:
        dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

  if dk_alias is None:
    assert dv_alias is None

    @pl.when(should_write)
    def _write_dkv() -> None:
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

  else:
    q_head_group = pl.program_id(0)
    first_q_head_in_kv_group = lax.rem(q_head_group, q_heads_per_kv_head) == 0

    @pl.when(jnp.logical_and(should_write, first_q_head_in_kv_group))
    def _write_first_dkv() -> None:
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

    @pl.when(jnp.logical_and(should_write, _not(first_q_head_in_kv_group)))
    def _accum_dkv() -> None:
      dk_ref[...] = dk_alias[...] + dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_alias[...] + dv_scratch_ref[...].astype(dv_ref.dtype)


def _flash_attention_dkv_kernel_head_stitch(
    # Prefetched inputs
    active_rows_ref: Any,
    active_cols_ref: Any,
    mask_next_ref: Any,
    bounds_start_ref: Any,
    bounds_end_ref: Any,
    block_mask_ref: Any,
    prng_key_ref: Any,
    # Inputs
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    q_segment_ids_ref: Any,
    kv_segment_ids_ref: Any,
    logsumexp_ref: Any,
    do_ref: Any,
    di_ref: Any,
    mask_ref: Any,
    q_sequence_ref: Any,
    # Aliases
    dq_alias: Any,
    dk_alias: Any,
    dv_alias: Any,
    # Outputs
    dq_ref: Any,
    dk_ref: Any,
    dv_ref: Any,
    # Scratch
    dq_scratch_ref: Any,
    dkt_scratch_ref: Any,
    dvt_scratch_ref: Any,
    qt_scratch_ref: Any,
    dot_scratch_ref: Any,
    lse_scratch_ref: Any,
    di_scratch_ref: Any,
    qseq_scratch_ref: Any,
    *,
    mask_value: float,
    q_steps: int,
    kv_steps: int,
    bq: int,
    bkv_compute: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    q_heads_per_kv_head: int,
    config: tk.SplashConfig,
    num_stacked_q_heads_dkv: int = 1,
    use_vmem_dq_reduction: bool = False,
) -> None:
  """Fused dKV kernel accumulating `dK^T` and `dV^T` for 256x256 MXU fill."""
  del mask_next_ref, prng_key_ref, mask_ref, q_segment_ids_ref
  del kv_segment_ids_ref, kv_steps
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E
  stacked = num_stacked_q_heads_dkv
  rows = stacked * bq
  bkv_repeats = bkv_compute // _NUM_LANES

  should_initialize_dq: bool | jax.Array = True
  should_write_dq: bool | jax.Array = True
  if active_rows_ref is not None:
    assert bounds_start_ref is not None
    assert bounds_end_ref is not None
    grid_idx = pl.program_id(1)
    kv_index = active_rows_ref[grid_idx].astype(jnp.int32)
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    if use_vmem_dq_reduction:
      col = lambda g: active_cols_ref[g].astype(jnp.int32)
      last = pl.num_programs(1) - 1
      should_initialize_dq = jnp.logical_or(grid_idx == 0, col(jnp.maximum(grid_idx - 1, 0)) != col(grid_idx))
      should_write_dq = jnp.logical_or(
          grid_idx == last,
          col(jnp.minimum(grid_idx + 1, last)) != col(grid_idx),
      )
  else:
    kv_index, q_head, q_index = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
    )
    grid_idx = (kv_index * q_steps) + q_index
    should_initialize = q_index == 0
    should_write = True if q_steps <= 2 else q_index == q_steps - 1
    if q_heads_per_kv_head > 1:
      q_head_index_per_kv_head = lax.rem(q_head, q_heads_per_kv_head)
      should_initialize = jnp.logical_and(should_initialize, q_head_index_per_kv_head == 0)
      should_write = jnp.logical_and(should_write, q_head_index_per_kv_head == q_heads_per_kv_head - 1)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) > 0
  else:
    should_not_mask = False
    should_run = True

  @pl.when(should_initialize)
  def init() -> None:
    dkt_scratch_ref[...] = jnp.zeros_like(dkt_scratch_ref)
    dvt_scratch_ref[...] = jnp.zeros_like(dvt_scratch_ref)

  def load_q() -> tuple[jax.Array, jax.Array]:
    q = q_ref[...]
    if stacked > 1:
      q = q.reshape((rows, q.shape[-1]))
    if config.use_base2_exp and not config.q_prescaled_log2e:
      scaled_q = q * LOG2E
    else:
      scaled_q = q
    return q, scaled_q

  def load_do() -> jax.Array:
    do = do_ref[...]
    if stacked > 1:
      do = do.reshape((rows, do.shape[-1]))
    return do

  def setup() -> None:
    q, scaled_q = load_q()
    q_dk = scaled_q if config.dkv_reuse_scaled_q else q
    qt_scratch_ref[...] = q_dk.astype(jnp.float32).T.astype(qt_scratch_ref.dtype)
    dot_scratch_ref[...] = load_do().astype(jnp.float32).T.astype(dot_scratch_ref.dtype)
    if stacked == 1:
      lse_row = logsumexp_ref[:1, :]
      di_row = di_ref[:1, :]
    else:
      lse_row = jnp.concatenate([logsumexp_ref[h, :1, :] for h in range(stacked)], axis=1)
      di_row = jnp.concatenate([di_ref[h, :1, :] for h in range(stacked)], axis=1)
    lse_scratch_ref[...] = jnp.broadcast_to(lse_row, (_NUM_LANES, rows)).T
    di_scratch_ref[...] = jnp.broadcast_to(di_row, (_NUM_LANES, rows)).T
    if q_sequence_ref is not None:
      qseq_scratch_ref[...] = jnp.broadcast_to(q_sequence_ref[:1, :], (_NUM_LANES, bq)).T

  def body(i: int | jax.Array, _: Any, has_partial_mask: bool = False) -> None:
    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    _, scaled_q = load_q()
    do = load_do()
    k = k_ref[slice_k, :]
    v = v_ref[slice_k, :]

    qk_uncapped = lax.dot_general(scaled_q, k, _NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
    qk3 = qk_uncapped.reshape((stacked, bq, bkv_compute)) if stacked > 1 else qk_uncapped
    qk3 = _apply_mask_and_soft_cap(
        qk3,
        mask_value,
        None,
        qseq_scratch_ref if q_sequence_ref is not None else None,
        None,
        None,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=True,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    qk = qk3.reshape((rows, bkv_compute)) if stacked > 1 else qk3
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - jnp.tile(lse_scratch_ref[...], (1, bkv_repeats)))

    dp = lax.dot_general(do, v, _NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
    ds = (dp - jnp.tile(di_scratch_ref[...], (1, bkv_repeats))) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)

    dvt = lax.dot_general(
        dot_scratch_ref[...],
        p.astype(do.dtype),
        _NN_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    dvt_scratch_ref[i] = dvt.astype(dvt_scratch_ref.dtype) + dvt_scratch_ref[i]

    dkt = lax.dot_general(
        qt_scratch_ref[...],
        ds.astype(do.dtype),
        _NN_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    if config.use_base2_exp and (config.dkv_reuse_scaled_q or config.q_prescaled_log2e):
      dkt = dkt / LOG2E
    dkt_scratch_ref[i] = dkt.astype(dkt_scratch_ref.dtype) + dkt_scratch_ref[i]

    if dq_scratch_ref is not None or dq_ref is not None:
      dq = lax.dot_general(
          ds if _is_fp8(k) else ds.astype(k.dtype),
          k,
          _NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      if config.q_prescaled_log2e:
        dq = dq / LOG2E
      if stacked > 1:
        dq = dq.reshape((stacked, bq, dq.shape[-1]))
      if dq_scratch_ref is not None:
        dq_scratch_ref[...] += dq
      else:
        if dq_alias is not None:
          dq_ref[...] = dq_alias[...] + dq.astype(dq_ref.dtype)
        else:
          dq_ref[...] = dq.astype(dq_ref.dtype)

  if dq_scratch_ref is not None:

    @pl.when(should_initialize_dq)
    def _init_dq_scratch() -> None:
      dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  elif dq_alias is not None:
    dq_ref[...] = dq_alias[...]
  else:
    dq_ref[...] = jnp.zeros_like(dq_ref)

  num_iters = k_ref.shape[0] // bkv_compute
  assert dkt_scratch_ref.shape[0] == num_iters

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _run_unmasked() -> None:
    setup()
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
  def _run_partial() -> None:
    setup()
    lax.fori_loop(
        0,
        num_iters,
        functools.partial(body, has_partial_mask=True),
        None,
        unroll=True,
    )

  if dq_scratch_ref is not None:

    @pl.when(should_write_dq)
    def _write_dq_scratch() -> None:
      if dq_alias is not None:
        dq_ref[...] = dq_alias[...] + dq_scratch_ref[...].astype(dq_ref.dtype)
      else:
        dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

  def untranspose(scratch_ref: Any) -> jax.Array:
    return jnp.concatenate([scratch_ref[c].T for c in range(num_iters)], axis=0)

  if dk_alias is None:
    assert dv_alias is None

    @pl.when(should_write)
    def _write_dkv() -> None:
      dk_ref[...] = untranspose(dkt_scratch_ref).astype(dk_ref.dtype)
      dv_ref[...] = untranspose(dvt_scratch_ref).astype(dv_ref.dtype)

  else:
    q_head_group = pl.program_id(0)
    first_q_head_in_kv_group = lax.rem(q_head_group, q_heads_per_kv_head) == 0

    @pl.when(jnp.logical_and(should_write, first_q_head_in_kv_group))
    def _write_first_dkv() -> None:
      dk_ref[...] = untranspose(dkt_scratch_ref).astype(dk_ref.dtype)
      dv_ref[...] = untranspose(dvt_scratch_ref).astype(dv_ref.dtype)

    @pl.when(jnp.logical_and(should_write, _not(first_q_head_in_kv_group)))
    def _accum_dkv() -> None:
      dk_ref[...] = dk_alias[...] + untranspose(dkt_scratch_ref).astype(dk_ref.dtype)
      dv_ref[...] = dv_alias[...] + untranspose(dvt_scratch_ref).astype(dv_ref.dtype)


def _splash_attention_bwd_dkv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    logsumexp: jax.Array,
    do: jax.Array,
    di: jax.Array,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: MaskInfo,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    config: tk.SplashConfig,
    dkv_mask_sparsity: float,
    prng_key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Launches the fused backward dQ/dK/dV SplashAttention kernel."""
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]
  num_stacked_q_heads_dkv: int = getattr(config, "num_stacked_q_heads_dkv", 1)
  prng_key = _check_dropout_args(config, prng_key)
  kv_steps = kv_seq_len // bkv
  q_steps = q_seq_len // bq
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if num_stacked_q_heads_dkv < 1:
    raise ValueError(f"num_stacked_q_heads_dkv must be >= 1, got {num_stacked_q_heads_dkv}.")
  if num_stacked_q_heads_dkv > 1:
    if config.q_layout != QKVLayout.HEAD_DIM_MINOR:
      raise ValueError("num_stacked_q_heads_dkv > 1 requires q_layout == HEAD_DIM_MINOR.")
    if config.dropout_rate:
      raise ValueError("num_stacked_q_heads_dkv > 1 is not supported with dropout_rate > 0.")
    if config.qk_diag_skip:
      raise ValueError("num_stacked_q_heads_dkv > 1 is not supported with qk_diag_skip.")

  if segment_ids is not None:
    mask_info, bounds_start, bounds_end = _refine_mask_info_with_segments(
        mask_info,
        segment_ids,
        q_blocks=q_steps,
        kv_blocks=kv_steps,
        bq=bq,
        bkv=bkv,
        is_dkv=True,
    )
  elif mask_info.active_rows is not None:
    bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
  else:
    bounds_start, bounds_end = None, None
  dynamic_grid = mask_info.active_rows is not None

  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")
  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to divide"
        f" the number of 'query' heads ({num_q_heads})."
    )
  if num_q_heads % num_stacked_q_heads_dkv != 0:
    raise ValueError(f"{num_q_heads=} must be a multiple of {num_stacked_q_heads_dkv=}")
  if q_heads_per_kv_head % num_stacked_q_heads_dkv != 0:
    raise ValueError(f"{q_heads_per_kv_head=} must be a multiple of" f" {num_stacked_q_heads_dkv=}")
  if num_stacked_q_heads_dkv > 1 and bq % _NUM_LANES != 0:
    raise ValueError(f"{bq=} must be a multiple of {_NUM_LANES} when" " num_stacked_q_heads_dkv > 1.")
  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same" " leading dimensions.")

  num_q_head_groups = num_q_heads // num_stacked_q_heads_dkv
  q_head_groups_per_kv_head = q_heads_per_kv_head // num_stacked_q_heads_dkv
  head_block_dim = None if num_stacked_q_heads_dkv == 1 else num_stacked_q_heads_dkv

  if dynamic_grid:

    def unravel(f: Callable[..., Any]) -> Callable[..., Any]:
      def index_map(h: Any, grid_idx: Any, rows_ref: Any, cols_ref: Any, *_: Any) -> Any:
        j = tk.to_i32(rows_ref[grid_idx])
        i = tk.to_i32(cols_ref[grid_idx])
        return f(h, i, j)

      return index_map

    assert mask_info.num_active_blocks is not None
    grid_size = mask_info.num_active_blocks[0]
    grid: tuple[Any, ...] = (num_q_head_groups, grid_size)

    def mask_index_map(
        h: Any,
        grid_idx: Any,
        rows_ref: Any,
        cols_ref: Any,
        mask_next_ref: Any = None,
        *_: Any,
    ) -> tuple[Any, int, int]:
      del h, rows_ref, cols_ref
      next_m = tk.to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  else:
    unravel = lambda f: lambda j, h, i, *_: f(h, i, j)
    grid = (kv_steps, num_q_head_groups, q_steps)

    def mask_index_map(
        j: Any,
        h: Any,
        i: Any,
        rows_ref: Any,
        cols_ref: Any,
        mask_next_ref: Any = None,
        *_: Any,
    ) -> tuple[Any, int, int]:
      del h, rows_ref, cols_ref
      grid_idx = j * q_steps + i
      next_m = tk.to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  q_index_map = unravel(lambda h, i, j: tk.from_head_minor((h, i, 0), config.q_layout))
  o_index_map = unravel(lambda h, i, j: (h, i, 0))

  def create_kv_index_map(layout: QKVLayout) -> Callable[..., Any]:
    def index_map(h: Any, i: Any, j: Any, *_: Any) -> Any:
      del i
      first_h = h * num_stacked_q_heads_dkv
      prefix = () if is_mqa else (_div(first_h, q_heads_per_kv_head),)
      return tk.from_head_minor((*prefix, j, 0), layout)

    return index_map

  k_index_map = unravel(create_kv_index_map(config.k_layout))
  v_index_map = unravel(create_kv_index_map(config.v_layout))

  q_spec = pl.BlockSpec(
      tk.from_head_minor((head_block_dim, bq, head_dim_qk), config.q_layout),
      q_index_map,
  )
  o_spec = pl.BlockSpec((head_block_dim, bq, head_dim_v), o_index_map)
  k_spec = pl.BlockSpec(
      tk.from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          config.k_layout,
      ),
      k_index_map,
  )
  v_spec = pl.BlockSpec(
      tk.from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          config.v_layout,
      ),
      v_index_map,
  )

  reduce_dkv_over_q_heads = dynamic_grid and q_head_groups_per_kv_head != 1 and kv_steps <= 2
  alias_dkv = dynamic_grid and q_head_groups_per_kv_head != 1 and not reduce_dkv_over_q_heads

  def create_dkv_index_map(h: Any, i: Any, j: Any, *_: Any) -> tuple[Any, ...]:
    del i
    if reduce_dkv_over_q_heads:
      prefix: tuple[Any, ...] = (h,)
    elif is_mqa:
      prefix = ()
    else:
      prefix = (_div(h * num_stacked_q_heads_dkv, q_heads_per_kv_head),)
    return (*prefix, j, 0)

  dkv_index_map = unravel(create_dkv_index_map)

  dk_spec = pl.BlockSpec(
      (bkv, head_dim_qk) if (is_mqa and not reduce_dkv_over_q_heads) else (None, bkv, head_dim_qk),
      dkv_index_map,
  )
  dv_spec = pl.BlockSpec(
      (bkv, head_dim_v) if (is_mqa and not reduce_dkv_over_q_heads) else (None, bkv, head_dim_v),
      dkv_index_map,
  )
  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  q_segment_ids_index_map = unravel(lambda h, i, j: (0, i))
  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)
    assert isinstance(segment_ids.kv, jax.Array)
    kv_segment_ids_index_map = unravel(lambda h, i, j: (j, 0))
    q_segment_spec: pl.BlockSpec | None = pl.BlockSpec((_NUM_SUBLANES, bq), q_segment_ids_index_map)
    kv_segment_spec: pl.BlockSpec | None = pl.BlockSpec((bkv, _NUM_LANES), kv_segment_ids_index_map)
    q_segment_ids: jax.Array | None = jax.lax.broadcast_in_dim(segment_ids.q, (_NUM_SUBLANES, q_seq_len), (1,))
    kv_segment_ids: jax.Array | None = jax.lax.broadcast_in_dim(segment_ids.kv, (kv_seq_len, _NUM_LANES), (0,))
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  do_spec = o_spec
  logsumexp_index_map = unravel(lambda h, i, j: (h, 0, i))

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  logsumexp_shape = (num_q_heads, _NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((head_block_dim, _NUM_SUBLANES, bq), logsumexp_index_map)

  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((head_block_dim, _NUM_SUBLANES, bq), logsumexp_index_map)

  in_specs: list[pl.BlockSpec | None] = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((_NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(jnp.asarray(mask_info.q_sequence), (_NUM_SUBLANES, q_seq_len), (1,))
  else:
    q_sequence = None
    in_specs.append(None)

  use_vmem_dq_reduction = dynamic_grid and config.dq_reduction_steps == 1
  dq_reduction_steps = 3 if dynamic_grid else config.dq_reduction_steps
  if use_vmem_dq_reduction:
    dq_reduction_steps = None
  if not dynamic_grid and kv_steps <= 3 and dq_reduction_steps == 3:
    dq_reduction_steps = None

  dq: jax.Array | None = None
  dq_alias_spec: pl.BlockSpec | None = None
  if use_vmem_dq_reduction:
    dq_spec = pl.BlockSpec((head_block_dim, bq, head_dim_qk), o_index_map)
    dq_shape = jax.ShapeDtypeStruct(q.shape, _unquantized_dtype(q))
  elif dq_reduction_steps == 3:
    dq_index_map = (
        (lambda h, g, r, c, *_: (g % 3, h, tk.to_i32(c[g]), 0))
        if dynamic_grid
        else unravel(lambda h, i, j: (j % 3, h, i, 0))
    )
    dq_spec = pl.BlockSpec((None, head_block_dim, bq, head_dim_qk), dq_index_map)
    dq_alias_spec = dq_spec
    dq_shape = jax.ShapeDtypeStruct((3, *q.shape), _unquantized_dtype(q))
    dq = jnp.zeros_like(dq_shape)
  else:
    dq_index_map = unravel(lambda h, i, j: (j, h, i, 0))
    dq_spec = pl.BlockSpec((None, head_block_dim, bq, head_dim_qk), dq_index_map)
    if _is_fp8(q):
      q_dtype = jnp.dtype(jnp.bfloat16)
    else:
      q_dtype = q.dtype if kv_steps <= 4 else jnp.dtype(jnp.float32)
    dq_shape = jax.ShapeDtypeStruct((kv_steps, *q.shape), q_dtype)

  in_specs.append(dq_alias_spec)

  if bkv == bkv_compute and not use_vmem_dq_reduction:
    dq_scratch = None
  else:
    dq_scratch_shape = (bq, head_dim_qk) if num_stacked_q_heads_dkv == 1 else (num_stacked_q_heads_dkv, bq, head_dim_qk)
    dq_scratch = pltpu.VMEM(dq_scratch_shape, jnp.float32)

  if alias_dkv:
    in_specs += [dk_spec, dv_spec]
    dk: jax.Array | None = lax.empty(k.shape, dtype=jnp.float32)
    dv: jax.Array | None = lax.empty(v.shape, dtype=jnp.float32)
    dk_type = dv_type = jnp.dtype(jnp.float32)
  else:
    in_specs += [None, None]
    dk, dv = None, None
    dk_type = jnp.dtype(jnp.float32) if reduce_dkv_over_q_heads else _unquantized_dtype(k)
    dv_type = jnp.dtype(jnp.float32) if reduce_dkv_over_q_heads else _unquantized_dtype(v)

  dk_shape = (num_q_head_groups, kv_seq_len, head_dim_qk) if reduce_dkv_over_q_heads else k.shape
  dv_shape = (num_q_head_groups, kv_seq_len, head_dim_v) if reduce_dkv_over_q_heads else v.shape
  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct(dk_shape, dk_type),
      jax.ShapeDtypeStruct(dv_shape, dv_type),
  ]
  out_specs = [dq_spec, dk_spec, dv_spec]

  use_head_stitch = (
      bool(getattr(config, "dkv_head_stitch", False))
      and segment_ids is None
      and mask_info.partial_mask_blocks is None
      and not (_is_fp8(q) or _is_fp8(k) or _is_fp8(v))
      and config.q_layout == QKVLayout.HEAD_DIM_MINOR
      and config.k_layout == QKVLayout.HEAD_DIM_MINOR
      and config.v_layout == QKVLayout.HEAD_DIM_MINOR
      and not config.dropout_rate
      and not config.qk_diag_skip
      and head_dim_qk % _NUM_LANES == 0
      and head_dim_v % _NUM_LANES == 0
      and bkv_compute % _NUM_LANES == 0
      and bq % _NUM_LANES == 0
  )

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      mask_value=mask_value,
      q_steps=q_steps,
      kv_steps=kv_steps,
      bq=bq,
      bkv_compute=bkv_compute,
      config=config,
      bkv=bkv,
      mask_function=mask_function,
      q_heads_per_kv_head=q_head_groups_per_kv_head,
      num_stacked_q_heads_dkv=num_stacked_q_heads_dkv,
      use_vmem_dq_reduction=use_vmem_dq_reduction,
  )

  kernel_name = tk.get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dkv",
  )
  if use_head_stitch:
    kernel_name = f"{kernel_name}_head_stitch"
  sched_dkv_override: bool | None = getattr(config, "use_experimental_scheduler_dkv", None)
  use_sched_dkv = config.use_experimental_scheduler if sched_dkv_override is None else sched_dkv_override
  metadata = {
      "xprof_metadata": json.dumps(
          dict(
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_kv_dkv_compute=bkv_compute,
              num_stacked_q_heads_dkv=num_stacked_q_heads_dkv,
              dkv_head_stitch=use_head_stitch,
              q_layout=config.q_layout,
              k_layout=config.k_layout,
              v_layout=config.v_layout,
              use_experimental_scheduler=use_sched_dkv,
          ),
      )
  }
  args = [
      mask_info.active_rows,
      mask_info.active_cols,
      mask_info.mask_next,
      bounds_start,
      bounds_end,
      mask_info.block_mask,
      prng_key,
      q if config.q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
      k if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
      v if config.v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
  ]
  num_args = sum(1 for x in args if x is not None)
  input_output_aliases: dict[int, int] = {}
  if dq_reduction_steps == 3:
    if alias_dkv:
      input_output_aliases = {num_args: 0, num_args + 1: 1, num_args + 2: 2}
    else:
      input_output_aliases = {num_args: 0}
  elif alias_dkv:
    input_output_aliases = {num_args: 1, num_args + 1: 2}

  scratch_shapes: list[Any] = [
      dq_scratch,
      pltpu.VMEM((bkv, head_dim_qk), jnp.float32),
      pltpu.VMEM((bkv, head_dim_v), jnp.float32),
  ]
  if use_head_stitch:
    stitch_rows = num_stacked_q_heads_dkv * bq
    stitch_iters = bkv // bkv_compute
    scratch_shapes = [
        dq_scratch,
        pltpu.VMEM((stitch_iters, head_dim_qk, bkv_compute), jnp.float32),
        pltpu.VMEM((stitch_iters, head_dim_v, bkv_compute), jnp.float32),
        pltpu.VMEM((head_dim_qk, stitch_rows), q.dtype),
        pltpu.VMEM((head_dim_v, stitch_rows), do.dtype),
        pltpu.VMEM((stitch_rows, _NUM_LANES), jnp.float32),
        pltpu.VMEM((stitch_rows, _NUM_LANES), jnp.float32),
        None if q_sequence is None else pltpu.VMEM((bq, _NUM_LANES), q_sequence.dtype),
    ]
    kernel = functools.partial(_flash_attention_dkv_kernel_head_stitch, **kernel.keywords)

  def _bwd_cost_estimate(
      q_in: jax.Array,
      k_in: jax.Array,
      v_in: jax.Array,
      q_seg: jax.Array | None,
      kv_seg: jax.Array | None,
      lse_in: jax.Array,
      do_in: jax.Array,
      di_in: jax.Array,
      partial_mask_blocks: jax.Array | np.ndarray | None,
      q_seq: jax.Array | None,
      shapes_out: Sequence[jax.ShapeDtypeStruct],
      mask_sparsity_factor: float,
  ) -> pl.CostEstimate:
    n_q_heads, q_len, d_qk = q_in.shape
    kv_len, d_v = v_in.shape[-2:]
    total_matmul_flops_per_head = (
        2 * q_len * kv_len * d_qk
        + 2 * q_len * kv_len * d_v
        + 2 * q_len * kv_len * d_v
        + 2 * q_len * kv_len * d_qk
        + 2 * q_len * kv_len * d_qk
    )
    estimated_flops = int(total_matmul_flops_per_head * n_q_heads * mask_sparsity_factor)
    exp_flops = n_q_heads * q_len * kv_len * mask_sparsity_factor
    tanh_flops = 0 if config.attn_logits_soft_cap is None else 2 * n_q_heads * q_len * kv_len * mask_sparsity_factor
    estimated_transcendentals = int(exp_flops + tanh_flops)
    inputs_ = [
        q_in,
        k_in,
        v_in,
        q_seg,
        kv_seg,
        lse_in,
        do_in,
        di_in,
        partial_mask_blocks,
        q_seq,
    ]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, shapes_out))
    return pl.CostEstimate(
        flops=estimated_flops,
        transcendentals=estimated_transcendentals,
        bytes_accessed=int(input_bytes + output_bytes),
    )

  cost_estimate = config.bwd_cost_estimate or _bwd_cost_estimate(
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
      out_shapes,
      dkv_mask_sparsity,
  )

  needs_expanded_dkv_vmem = (
      use_head_stitch
      or use_vmem_dq_reduction
      or bkv_compute >= 2048
      or bkv >= 8192
      or num_stacked_q_heads_dkv * bq >= 4096
  )
  dkv_vmem_limit_bytes = _VMEM_LIMIT_64_MIB if needs_expanded_dkv_vmem else None

  with jax.named_scope(kernel_name):
    dq_unreduced, dk_out, dv_out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",) * len(grid),
            vmem_limit_bytes=dkv_vmem_limit_bytes,
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": use_sched_dkv},
        ),
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(*args, dq, dk, dv)
  dq_final = dq_unreduced if use_vmem_dq_reduction else dq_unreduced.sum(axis=0)
  if reduce_dkv_over_q_heads:
    dk_out = dk_out.reshape(num_kv_heads, q_head_groups_per_kv_head, kv_seq_len, head_dim_qk).sum(axis=1)
    dv_out = dv_out.reshape(num_kv_heads, q_head_groups_per_kv_head, kv_seq_len, head_dim_v).sum(axis=1)
    if is_mqa:
      dk_out = jnp.squeeze(dk_out, axis=0)
      dv_out = jnp.squeeze(dv_out, axis=0)
  dq_final = dq_final.astype(_unquantized_dtype(q))
  dk_out = dk_out.astype(_unquantized_dtype(k))
  dv_out = dv_out.astype(_unquantized_dtype(v))
  return dq_final, dk_out, dv_out


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(7, 8, 9, 10, 11, 12, 13),
)
def splash_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None = None,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    is_mqa: bool = False,
    config: tk.SplashConfig | None = None,
    mask_function: MaskFunctionType | None = None,
    fwd_mask_sparsity: float = 1.0,
    dkv_mask_sparsity: float = 1.0,
    max_logit_value: jax.Array | None = None,
    prng_key: jax.Array | None = None,
) -> base.SplashCustomReturnType:
  """Custom-VJP SplashAttention entry point supporting GQA head folding.

  Args:
    fwd_mask_info: Preprocessed forward block mask metadata.
    dkv_mask_info: Optional backward `dKV` block mask metadata (`None` when only
      forward evaluation is required).
    q: Query tensor of shape `[num_q_heads, q_seq_len, head_dim_qk]`.
    k: Key tensor of shape `[num_kv_heads, kv_seq_len, head_dim_qk]` (or 2D when
      `is_mqa=True`).
    v: Value tensor of shape `[num_kv_heads, kv_seq_len, head_dim_v]` (or 2D
      when `is_mqa=True`).
    segment_ids: Optional `(q, kv)` packed document segment IDs (`None` disables
      cross-segment masking).
    sinks: Optional per-query-head attention sink logits (`None` omits sinks).
    save_residuals: Whether to return `(out, stats)` with `logsumexp` and
      `max_logits` alongside the attention output.
    mask_value: Additive mask fill constant for masked-out logit positions.
    is_mqa: Whether `k` and `v` are single-head MQA tensors without a head axis.
    config: Optional `SplashConfig` (`None` uses `FlintSplashConfig` defaults).
    mask_function: Optional coordinate mask callable (`None` uses `MaskInfo`
      partial mask blocks when present).
    fwd_mask_sparsity: Fraction of active forward mask blocks for cost model.
    dkv_mask_sparsity: Fraction of active backward `dKV` blocks for cost model.
    max_logit_value: Optional per-head max-logit estimate (`None` computes exact
      online softmax maxima).
    prng_key: Optional PRNG key for attention dropout (`None` disables dropout).

  Returns:
    Attention output `out`, or `(out, stats)` when `save_residuals=True`.
  """
  del dkv_mask_info, dkv_mask_sparsity
  if config is None:
    config = FlintSplashConfig.get_default()

  ret = _splash_attention_forward(
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=save_residuals,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
      prng_key=prng_key,
      return_max_logits=save_residuals,
  )
  if save_residuals:
    assert isinstance(ret, tuple)
    out, stats = ret
    if config.use_base2_exp:
      stats = {
          "logsumexp": stats["logsumexp"] / LOG2E,
          "max_logits": stats["max_logits"] / LOG2E,
      }
    return out, stats
  return ret


def splash_attention_fwd(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None = None,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    is_mqa: bool = False,
    config: tk.SplashConfig | None = None,
    mask_function: MaskFunctionType | None = None,
    fwd_mask_sparsity: float = 1.0,
    dkv_mask_sparsity: float = 1.0,
    max_logit_value: jax.Array | None = None,
    prng_key: jax.Array | None = None,
) -> tuple[base.SplashCustomReturnType, base.SplashResidualsType]:
  """Forward rule for `splash_attention_custom` omitting dead `max_logits`.

  Args:
    fwd_mask_info: Preprocessed forward block mask metadata.
    dkv_mask_info: Optional backward `dKV` block mask metadata (`None` skips
      `dKV` mask specialization).
    q: Query tensor of shape `[num_q_heads, q_seq_len, head_dim_qk]`.
    k: Key tensor of shape `[num_kv_heads, kv_seq_len, head_dim_qk]` (or 2D).
    v: Value tensor of shape `[num_kv_heads, kv_seq_len, head_dim_v]` (or 2D).
    segment_ids: Optional `(q, kv)` packed document segment IDs (`None` disables
      cross-segment masking).
    sinks: Optional per-query-head attention sink logits (`None` omits sinks).
    save_residuals: Whether to return `(out, stats)` alongside VJP residuals.
    mask_value: Additive mask fill constant for masked-out logit positions.
    is_mqa: Whether `k` and `v` are single-head MQA tensors without a head axis.
    config: Optional `SplashConfig` (`None` uses `FlintSplashConfig` defaults).
    mask_function: Optional coordinate mask callable (`None` uses `MaskInfo`).
    fwd_mask_sparsity: Fraction of active forward mask blocks for cost model.
    dkv_mask_sparsity: Unused in forward rule; kept for `custom_vjp` arity.
    max_logit_value: Optional per-head max-logit estimate (`None` computes exact
      online softmax maxima).
    prng_key: Optional PRNG key for attention dropout (`None` disables dropout).

  Returns:
    Tuple `(output, residuals)` for `splash_attention_bwd`.
  """
  del dkv_mask_sparsity
  if config is None:
    config = FlintSplashConfig.get_default()

  ret = _splash_attention_forward(
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=True,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
      prng_key=prng_key,
      return_max_logits=save_residuals,
  )
  assert isinstance(ret, tuple)
  out, stats = ret
  logsumexp = stats["logsumexp"]
  residuals: base.SplashResidualsType = (
      q,
      k,
      v,
      segment_ids,
      sinks,
      out,
      logsumexp,
      dkv_mask_info,
      prng_key,
  )
  if save_residuals:
    if config.use_base2_exp:
      stats = {
          "logsumexp": stats["logsumexp"] / LOG2E,
          "max_logits": stats["max_logits"] / LOG2E,
      }
    return (out, stats), residuals
  return out, residuals


def splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: tk.SplashConfig | None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    res: base.SplashResidualsType,
    grads: jax.Array | tuple[jax.Array, dict[str, jax.Array]],
) -> tuple[
    MaskInfo | None,
    MaskInfo | None,
    jax.Array,
    jax.Array,
    jax.Array,
    base.SegmentIds | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
  """Backward rule for `splash_attention_custom`.

  Args:
    save_residuals: Whether forward returned `(out, stats)` tuple output.
    mask_value: Additive mask fill constant for masked-out logit positions.
    is_mqa: Whether `k` and `v` are single-head MQA tensors without a head axis.
    config: Optional `SplashConfig` (`None` uses `FlintSplashConfig` defaults).
    mask_function: Optional coordinate mask callable (`None` uses `MaskInfo`).
    fwd_mask_sparsity: Unused in backward rule; kept for `custom_vjp` arity.
    dkv_mask_sparsity: Fraction of active backward `dKV` blocks for cost model.
    res: Saved VJP residuals from `splash_attention_fwd`.
    grads: Upstream cotangent `do` (or `(do, dstats)` if `save_residuals=True`).

  Returns:
    Tuple of cotangents matching the differentiable arguments of
    `splash_attention_custom`.
  """
  if config is None:
    config = FlintSplashConfig.get_default()
  if save_residuals:
    assert isinstance(grads, tuple)
    do, _ = grads
  else:
    assert isinstance(grads, jax.Array)
    do = grads
  del fwd_mask_sparsity
  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")
  bq_dkv = config.block_q_dkv
  bkv_dkv_memory = config.block_kv_dkv
  bkv_dkv_compute = config.block_kv_dkv_compute
  assert bq_dkv is not None
  assert bkv_dkv_memory is not None
  assert bkv_dkv_compute is not None

  q, k, v, segment_ids, sinks, o, logsumexp, dkv_mask_info, prng_key = res
  assert dkv_mask_info is not None

  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))
  dq, dk, dv = _splash_attention_bwd_dkv(
      q,
      k,
      v,
      segment_ids,
      logsumexp,
      do,
      di,
      bq=bq_dkv,
      bkv=bkv_dkv_memory,
      bkv_compute=bkv_dkv_compute,
      is_mqa=is_mqa,
      mask_info=dkv_mask_info,
      mask_value=mask_value,
      mask_function=mask_function,
      config=config,
      dkv_mask_sparsity=dkv_mask_sparsity,
      prng_key=prng_key,
  )
  dsinks: jax.Array | None = None
  if sinks is not None:
    logsumexp_nat = (logsumexp / LOG2E) if config.use_base2_exp else logsumexp
    sinks_exp = -jnp.exp(sinks[..., None, None].astype(jnp.float32) - logsumexp_nat[..., None].astype(jnp.float32))
    dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))
  return (
      None,
      None,
      dq,
      dk,
      dv,
      None,
      dsinks,
      None,
      None,
  )


splash_attention_custom.defvjp(splash_attention_fwd, splash_attention_bwd)
