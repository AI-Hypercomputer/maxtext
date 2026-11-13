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

"""Hybrid GQA attention with QK-RMSNorm, split-half RoPE, CP, and Splash."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.splash_attention import base as tokamax_splash_base
from tokamax._src.ops.splash_attention import splash_attention_mask as tokamax_splash_mask
from tokamax._src.ops.splash_attention import splash_attention_mask_info as tokamax_splash_mask_info

from maxtext.experimental.flint40b import flint_attention_prologue
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_splash_kernel
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops

_SPLASH_NUM_LANES: int = 128
_REF_MASK_VALUE: float = -1.0e30


@functools.lru_cache(maxsize=32)
def make_fast_tokamax_mask_info(
    mask: tokamax_splash_mask.CausalMask | tokamax_splash_mask.LocalMask,
    block_shape: tuple[int, int],
    *,
    is_dkv: bool,
    return_dynamic_grid: bool,
    q_seq_shards: int,
) -> tuple[tokamax_splash_mask_info.MaskInfo, Any]:
  """Vectorized `process_mask` for `CausalMask` and causal `LocalMask`.

  Computes block sparsity states analytically from the min/max query indices of
  each contiguous `q_sequence` block without materializing dense `[bq, bkv]`
  boolean tiles, matching `tokamax_splash_mask_info._process_mask` bit-for-bit.

  Args:
    mask: `CausalMask(offset=0)` or `LocalMask(window_size=(L, 0), offset=0)`
      with unit-stride monotonic `q_sequence` within each query block.
    block_shape: `(q_block_size, kv_block_size)` tile sizes.
    is_dkv: Whether to transpose block grids for the dKV backward kernel.
    return_dynamic_grid: Whether to compact active block coordinates.
    q_seq_shards: Number of query sequence shards (`cp_size`).

  Returns:
    Tuple `(mask_info, mask_function)` matching `_process_mask`.

  Raises:
    ValueError: If `mask`, `block_shape`, `q_seq_shards`, or `q_sequence`
      violate Splash invariants.
  """
  if len(mask.shape) != 2:
    raise ValueError(f"Expected a 2-dim mask, instead got: {mask.shape=}.")

  q_seq_len, kv_seq_len = mask.shape
  q_block_size, kv_block_size = block_shape
  if q_block_size <= 0 or kv_block_size <= 0:
    raise ValueError(f"Block sizes must be positive, got {block_shape=}.")
  if q_seq_shards < 1:
    raise ValueError(f"q_seq_shards must be >= 1, got {q_seq_shards=}.")

  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  if q_mod != 0:
    raise ValueError(f"{q_block_size=} should divide {q_seq_len=}.")
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)
  if kv_mod != 0:
    raise ValueError(f"{kv_block_size=} should divide {kv_seq_len=}.")
  q_seq_len_per_shard, qs_mod = divmod(q_seq_len, q_seq_shards)
  if qs_mod != 0:
    raise ValueError(f"{q_seq_shards=} should divide {q_seq_len=}.")
  q_blocks_per_shard, qbs_mod = divmod(q_seq_len_per_shard, q_block_size)
  if qbs_mod != 0:
    raise ValueError(f"{q_block_size=} should divide {q_seq_len_per_shard=}.")

  q_sequence = mask.q_sequence
  mask_function = mask.mask_function
  q_blocks_2d = q_sequence.reshape(q_blocks_count, q_block_size)
  if not np.all(q_blocks_2d[:, -1] - q_blocks_2d[:, 0] == q_block_size - 1):
    raise ValueError(
        "Every query block in q_sequence must be contiguous monotonic"
        " (q_blocks_2d[:, -1] - q_blocks_2d[:, 0] == q_block_size - 1)."
    )
  if q_block_size > 1 and not np.all(np.diff(q_blocks_2d, axis=1) == 1):
    raise ValueError("Every query block in q_sequence must have unit stride (+1).")

  q_min = q_blocks_2d[:, 0:1].astype(np.int64)
  q_max = q_blocks_2d[:, -1:].astype(np.int64)
  kv_idx = np.arange(kv_blocks_count, dtype=np.int64)[None, :]
  kv_min = kv_idx * kv_block_size
  kv_max = (kv_idx + 1) * kv_block_size - 1

  state_grid = np.ones((q_blocks_count, kv_blocks_count), dtype=np.int32)
  if isinstance(mask, tokamax_splash_mask.CausalMask):
    if mask.offset != 0:
      raise ValueError(f"CausalMask offset must be 0, got {mask.offset}.")
    state_grid[q_max < kv_min] = 0
    state_grid[q_min >= kv_max] = 2
  elif isinstance(mask, tokamax_splash_mask.LocalMask):
    left_size, right_size = mask.window_size
    if mask.offset != 0 or left_size is None or left_size < 0 or right_size != 0:
      raise ValueError(
          "LocalMask must have window_size=(L, 0) with L >= 0 and offset=0,"
          f" got window_size={mask.window_size}, offset={mask.offset}."
      )
    state_grid[(q_max < kv_min) | (q_min - left_size > kv_max)] = 0
    state_grid[(q_min >= kv_max) & (q_max - left_size <= kv_min)] = 2
  else:
    raise ValueError(f"Unsupported mask type: {type(mask)!r}.")

  if (state_grid == 2).all():
    return (
        tokamax_splash_mask_info.MaskInfo(
            mask_next=None,
            active_rows=None,
            active_cols=None,
            block_mask=None,
            num_active_blocks=None,
            partial_mask_blocks=None,
            q_sequence=q_sequence,
        ),
        None,
    )

  partial_id_grid = np.full((q_blocks_count, kv_blocks_count), -1, dtype=np.int32)
  all_shards_metadata = []
  for q_shard_idx in range(q_seq_shards):
    q_slice = slice(
        q_shard_idx * q_blocks_per_shard,
        (q_shard_idx + 1) * q_blocks_per_shard,
    )
    all_shards_metadata.append(
        tokamax_splash_mask_info._generate_shard_metadata(  # pylint: disable=protected-access
            state_grid[q_slice, :],
            partial_id_grid[q_slice, :],
            is_dkv,
            return_dynamic_grid,
        )
    )

  (
      active_rows_slices,
      active_cols_slices,
      _,
      block_mask_slices,
      num_active_blocks_list,
  ) = zip(*all_shards_metadata)

  if return_dynamic_grid:
    max_size = max(num_active_blocks_list)

    def pad_idx(arr: np.ndarray) -> np.ndarray:
      return np.pad(
          arr,
          (0, max_size - arr.shape[0]),
          mode="constant",
          constant_values=-1,
      )

    active_rows = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
        np.concatenate(list(map(pad_idx, active_rows_slices)), axis=0)
    )
    active_cols = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
        np.concatenate(list(map(pad_idx, active_cols_slices)), axis=0)
    )
    block_mask_slices = list(map(pad_idx, block_mask_slices))
    num_active_blocks = np.array(num_active_blocks_list, dtype=np.int32)
  else:
    active_rows = None
    active_cols = None
    num_active_blocks = None

  block_mask = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
      np.concatenate(block_mask_slices, axis=0)
  )
  return (
      tokamax_splash_mask_info.MaskInfo(
          mask_next=None,
          active_rows=active_rows,
          active_cols=active_cols,
          block_mask=block_mask,
          num_active_blocks=num_active_blocks,
          partial_mask_blocks=None,
          q_sequence=q_sequence,
      ),
      mask_function,
  )


@dataclasses.dataclass(frozen=True)
class _LocalHaloMaskFn:
  """Hashable mask callable for 1-neighbor halo local sliding attention."""

  window_left: int

  def __call__(self, q_ids: Any, kv_ids: Any) -> Any:
    return (q_ids >= kv_ids) & ((q_ids - kv_ids) <= self.window_left)


@functools.lru_cache(maxsize=32)
def make_local_halo_mask_info(
    local_q_len: int,
    halo_len: int,
    window_left: int,
    cp_size: int,
    block_shape: tuple[int, int],
    *,
    is_dkv: bool,
    return_dynamic_grid: bool,
    load_balanced: bool = False,
) -> tuple[tokamax_splash_mask_info.MaskInfo, Callable[[Any, Any], Any]]:
  """Builds per-CP-shard `MaskInfo` for natural or 2-stripe LB halo SWA.

  Args:
    local_q_len: Local query sequence length per CP shard.
    halo_len: Number of overlapping left-neighbor halo KV tokens per stripe.
    window_left: Left attention context window length (`sliding_window - 1`).
    cp_size: Size of the context-parallelism mesh axis (`cp`).
    block_shape: SplashAttention `(block_q, block_kv)` tile dimensions.
    is_dkv: Whether building metadata for the backward `dKV` kernel.
    return_dynamic_grid: Whether to compact active block coordinates.
    load_balanced: When `True`, builds 2-stripe `[r, 2*cp_size - 1 - r]` halo
      metadata with `kv_len_halo = 2 * (halo_len + local_q_len // 2)`.

  Returns:
    Tuple `(mask_info, mask_fn)` for local-halo `flint_splash_kernel`.
  """
  q_block_size, kv_block_size = block_shape
  halo_blocks = halo_len // kv_block_size
  if load_balanced:
    if local_q_len % 2 != 0:
      raise ValueError(f"load_balanced halo requires even local_q_len={local_q_len}.")
    stripe_len = local_q_len // 2
    q_blocks_per_stripe = stripe_len // q_block_size
    q_blocks = 2 * q_blocks_per_stripe
    kv_len_halo = 2 * (halo_len + stripe_len)
    kv_blocks = kv_len_halo // kv_block_size
    q_seq_local = np.concatenate(
        [
            np.arange(halo_len, halo_len + stripe_len, dtype=np.int32),
            np.arange(
                2 * halo_len + stripe_len,
                2 * halo_len + 2 * stripe_len,
                dtype=np.int32,
            ),
        ],
        axis=0,
    )
    q_iota = np.arange(q_blocks_per_stripe, dtype=np.int32) * q_block_size
    q_min = np.concatenate([halo_len + q_iota, 2 * halo_len + stripe_len + q_iota], axis=0)[:, None]
  else:
    q_blocks = local_q_len // q_block_size
    q_blocks_per_stripe = q_blocks
    kv_len_halo = halo_len + local_q_len
    kv_blocks = kv_len_halo // kv_block_size
    q_seq_local = np.arange(halo_len, halo_len + local_q_len, dtype=np.int32)
    q_min = (halo_len + np.arange(q_blocks, dtype=np.int32) * q_block_size)[:, None]
  q_max = q_min + (q_block_size - 1)
  kv_min = (np.arange(kv_blocks, dtype=np.int32) * kv_block_size)[None, :]
  kv_max = kv_min + (kv_block_size - 1)

  is_zero = (q_max < kv_min) | ((q_min - kv_max) > window_left)
  is_full = (q_min >= kv_max) & ((q_max - kv_min) <= window_left)
  base_state_grid = np.where(is_zero, 0, np.where(is_full, 2, 1)).astype(np.int32)
  c0_state_grid = base_state_grid.copy()
  c0_state_grid[:q_blocks_per_stripe, :halo_blocks] = 0
  partial_id_grid = np.full((q_blocks, kv_blocks), -1, dtype=np.int32)

  all_shards_metadata = []
  for shard_idx in range(cp_size):
    sg = c0_state_grid if shard_idx == 0 else base_state_grid
    all_shards_metadata.append(
        tokamax_splash_mask_info._generate_shard_metadata(  # pylint: disable=protected-access
            sg,
            partial_id_grid,
            is_dkv,
            return_dynamic_grid,
        )
    )

  (
      active_rows_slices,
      active_cols_slices,
      _,
      block_mask_slices,
      num_active_blocks_list,
  ) = zip(*all_shards_metadata)

  if return_dynamic_grid:
    max_size = max(num_active_blocks_list)

    def pad_halo_idx(arr: np.ndarray) -> np.ndarray:
      return np.pad(
          arr,
          (0, max_size - arr.shape[0]),
          mode="constant",
          constant_values=-1,
      )

    active_rows = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
        np.concatenate(list(map(pad_halo_idx, active_rows_slices)), axis=0)
    )
    active_cols = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
        np.concatenate(list(map(pad_halo_idx, active_cols_slices)), axis=0)
    )
    block_mask_slices = list(map(pad_halo_idx, block_mask_slices))
    num_active_blocks = np.array(num_active_blocks_list, dtype=np.int32)
  else:
    active_rows = None
    active_cols = None
    num_active_blocks = None

  block_mask = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
      np.concatenate(block_mask_slices, axis=0)
  )
  q_sequence = np.tile(q_seq_local, cp_size)
  mask_fn = _LocalHaloMaskFn(window_left=window_left)
  return (
      tokamax_splash_mask_info.MaskInfo(
          mask_next=None,
          active_rows=active_rows,
          active_cols=active_cols,
          block_mask=block_mask,
          num_active_blocks=num_active_blocks,
          partial_mask_blocks=None,
          q_sequence=q_sequence,
      ),
      mask_fn,
  )


def _shard_tokamax_arr(
    arr: jax.Array | np.ndarray | None,
    rank: jax.Array,
    num_shards: int,
) -> Any:
  if arr is None or num_shards == 1:
    return arr
  jax_arr = jnp.asarray(arr)
  shard_len = jax_arr.shape[0] // num_shards
  return jax.lax.dynamic_slice_in_dim(jax_arr, rank * shard_len, shard_len, axis=0)


def _local_tokamax_mask_info(
    info: tokamax_splash_mask_info.MaskInfo | None,
    rank: jax.Array,
    num_shards: int,
) -> tokamax_splash_mask_info.MaskInfo | None:
  if info is None or num_shards == 1:
    return info
  return tokamax_splash_mask_info.MaskInfo(
      mask_next=_shard_tokamax_arr(info.mask_next, rank, num_shards),
      active_rows=_shard_tokamax_arr(info.active_rows, rank, num_shards),
      active_cols=_shard_tokamax_arr(info.active_cols, rank, num_shards),
      block_mask=_shard_tokamax_arr(info.block_mask, rank, num_shards),
      num_active_blocks=_shard_tokamax_arr(info.num_active_blocks, rank, num_shards),
      partial_mask_blocks=info.partial_mask_blocks,
      q_sequence=_shard_tokamax_arr(info.q_sequence, rank, num_shards),
  )


def _clamp_splash_tile(desired: int, limit: int) -> int:
  clamped = min(desired, limit)
  if limit % clamped == 0:
    return clamped
  return math.gcd(clamped, limit)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def _splash_batched(
    config: flint_splash_kernel.FlintSplashConfig,
    mask_value: float,
    mask_function: Any,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    is_global: bool,
    fwd_mask_info: tokamax_splash_mask_info.MaskInfo,
    dkv_mask_info: tokamax_splash_mask_info.MaskInfo | None,
    q_bhsd: jax.Array,
    k_bhsd: jax.Array,
    v_bhsd: jax.Array,
    seg_q_bs: jax.Array | None,
    seg_kv_bs: jax.Array | None,
    sinks_f32: jax.Array,
) -> jax.Array:
  """Runs batched Flint SplashAttention with exact FP32 `dsinks` VJP."""
  del is_global
  seg_ax = None if seg_q_bs is None else 0

  def _one_fwd(
      q_i: jax.Array,
      k_i: jax.Array,
      v_i: jax.Array,
      sq_i: jax.Array | None,
      skv_i: jax.Array | None,
  ) -> jax.Array:
    seg = None if sq_i is None or skv_i is None else tokamax_splash_base.SegmentIds(q=sq_i, kv=skv_i)
    save_res = not config.fuse_reciprocal
    ret = flint_splash_kernel.splash_attention_custom(
        fwd_mask_info,
        dkv_mask_info,
        q_i,
        k_i,
        v_i,
        seg,
        sinks_f32,
        save_residuals=save_res,
        mask_value=mask_value,
        is_mqa=False,
        config=config,
        mask_function=mask_function,
        fwd_mask_sparsity=fwd_mask_sparsity,
        dkv_mask_sparsity=dkv_mask_sparsity,
    )
    return ret[0] if isinstance(ret, tuple) else ret

  return jax.vmap(_one_fwd, in_axes=(0, 0, 0, seg_ax, seg_ax))(q_bhsd, k_bhsd, v_bhsd, seg_q_bs, seg_kv_bs)


def _splash_batched_fwd(
    config: flint_splash_kernel.FlintSplashConfig,
    mask_value: float,
    mask_function: Any,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    is_global: bool,
    fwd_mask_info: tokamax_splash_mask_info.MaskInfo,
    dkv_mask_info: tokamax_splash_mask_info.MaskInfo | None,
    q_bhsd: jax.Array,
    k_bhsd: jax.Array,
    v_bhsd: jax.Array,
    seg_q_bs: jax.Array | None,
    seg_kv_bs: jax.Array | None,
    sinks_f32: jax.Array,
) -> tuple[jax.Array, tuple[Any, ...]]:
  """Forward pass saving `logsumexp` for fused backward and FP32 `dsinks`."""
  seg_ax = None if seg_q_bs is None else 0

  def _one_fwd_with_res(
      q_i: jax.Array,
      k_i: jax.Array,
      v_i: jax.Array,
      sq_i: jax.Array | None,
      skv_i: jax.Array | None,
  ) -> tuple[jax.Array, jax.Array]:
    seg = None if sq_i is None or skv_i is None else tokamax_splash_base.SegmentIds(q=sq_i, kv=skv_i)
    out_i, res_i = flint_splash_kernel.splash_attention_fwd(
        fwd_mask_info,
        dkv_mask_info,
        q_i,
        k_i,
        v_i,
        seg,
        sinks_f32,
        save_residuals=False,
        mask_value=mask_value,
        is_mqa=False,
        config=config,
        mask_function=mask_function,
        fwd_mask_sparsity=fwd_mask_sparsity,
        dkv_mask_sparsity=dkv_mask_sparsity,
    )
    lse_i = res_i[6]
    assert isinstance(out_i, jax.Array)
    assert isinstance(lse_i, jax.Array)
    return out_i, lse_i

  out, logsumexp = jax.vmap(_one_fwd_with_res, in_axes=(0, 0, 0, seg_ax, seg_ax))(
      q_bhsd, k_bhsd, v_bhsd, seg_q_bs, seg_kv_bs
  )
  out_lse_name = "attn_global_out_lse" if is_global else "attn_local_out_lse"
  q_res = jax.ad_checkpoint.checkpoint_name(q_bhsd, "attn_q_res")
  out_res = jax.ad_checkpoint.checkpoint_name(out, out_lse_name)
  lse_res = jax.ad_checkpoint.checkpoint_name(logsumexp, out_lse_name)
  residuals = (
      q_res,
      k_bhsd,
      v_bhsd,
      seg_q_bs,
      seg_kv_bs,
      sinks_f32,
      out_res,
      lse_res,
      dkv_mask_info,
  )
  return out_res, residuals


def _splash_batched_bwd(
    config: flint_splash_kernel.FlintSplashConfig,
    mask_value: float,
    mask_function: Any,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    is_global: bool,
    residuals: tuple[Any, ...],
    do: jax.Array,
) -> tuple[
    None,
    None,
    jax.Array,
    jax.Array,
    jax.Array,
    None,
    None,
    jax.Array,
]:
  """Backward pass passing through `(dq, dk, dv)` with exact FP32 `dsinks`."""
  del is_global
  (
      q_bhsd,
      k_bhsd,
      v_bhsd,
      seg_q_bs,
      seg_kv_bs,
      sinks_f32,
      out,
      logsumexp,
      dkv_mask_info,
  ) = residuals
  seg_ax = None if seg_q_bs is None else 0

  def _one_bwd(
      q_i: jax.Array,
      k_i: jax.Array,
      v_i: jax.Array,
      sq_i: jax.Array | None,
      skv_i: jax.Array | None,
      out_i: jax.Array,
      lse_i: jax.Array,
      do_i: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    seg = None if sq_i is None or skv_i is None else tokamax_splash_base.SegmentIds(q=sq_i, kv=skv_i)
    res_i = (
        q_i,
        k_i,
        v_i,
        seg,
        None,
        out_i,
        lse_i,
        dkv_mask_info,
        None,
    )
    _, _, dq_i, dk_i, dv_i, _, _, _, _ = flint_splash_kernel.splash_attention_bwd(
        save_residuals=False,
        mask_value=mask_value,
        is_mqa=False,
        config=config,
        mask_function=mask_function,
        fwd_mask_sparsity=fwd_mask_sparsity,
        dkv_mask_sparsity=dkv_mask_sparsity,
        res=res_i,
        grads=do_i,
    )
    assert dq_i is not None and dk_i is not None and dv_i is not None
    return dq_i, dk_i, dv_i

  dq_bhsd, dk_bhsd, dv_bhsd = jax.vmap(_one_bwd, in_axes=(0, 0, 0, seg_ax, seg_ax, 0, 0, 0))(
      q_bhsd, k_bhsd, v_bhsd, seg_q_bs, seg_kv_bs, out, logsumexp, do
  )
  di_f32 = jnp.einsum(
      "bhsd,bhsd->bhs",
      out.astype(jnp.float32),
      do.astype(jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )
  lse_nat = (
      logsumexp.astype(jnp.float32) / flint_splash_kernel.LOG2E if config.use_base2_exp else logsumexp.astype(jnp.float32)
  )
  dsinks_f32 = -jnp.sum(
      jnp.exp(sinks_f32.astype(jnp.float32)[None, :, None] - lse_nat) * di_f32,
      axis=(0, 2),
  ).astype(jnp.float32)
  return (None, None, dq_bhsd, dk_bhsd, dv_bhsd, None, None, dsinks_f32)


_splash_batched.defvjp(_splash_batched_fwd, _splash_batched_bwd)


def dot_product_attention_ref(
    q_lb: jax.Array,
    k_full: jax.Array,
    v_full: jax.Array,
    seg_q_lb: jax.Array,
    seg_kv_full: jax.Array,
    sinks_f32: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
) -> jax.Array:
  """Exact segmented/windowed GQA dot-product attention with FP32 sinks.

  Args:
    q_lb: Load-balanced local queries `[B_loc, S_loc, H_q, head_dim]`.
    k_full: Natural-order full keys `[B_loc, S_full, H_kv, head_dim]`.
    v_full: Natural-order full values `[B_loc, S_full, H_kv, head_dim]`.
    seg_q_lb: Load-balanced query segment IDs `int32[B_loc, S_loc]`.
    seg_kv_full: Natural-order KV segment IDs `int32[B_loc, S_full]`.
    sinks_f32: Per-head sink logits `float32[H_q]`.
    cfg: Validated `FlintConfig`.
    is_global: If `True`, global causal mask; otherwise causal sliding window of
      width `cfg.model.sliding_window`.

  Returns:
    Load-balanced attention context `[B_loc, S_loc, H_q, head_dim]`.
  """
  b_loc, s_loc, h_q, head_dim = q_lb.shape
  s_full = k_full.shape[1]
  h_kv = k_full.shape[2]
  gqa = h_q // h_kv
  cp_size = cfg.sharding.cp_size

  q_grp = q_lb.astype(jnp.float32).reshape(b_loc, s_loc, h_kv, gqa, head_dim)
  k_f32 = k_full.astype(jnp.float32)
  scale = float(head_dim) ** -0.5
  scores = (
      jnp.einsum(
          "bshgd,bthd->bhgst",
          q_grp,
          k_f32,
          precision=jax.lax.Precision.HIGHEST,
      ).reshape(b_loc, h_q, s_loc, s_full)
      * scale
  )

  q_idx = ops.cp_lb_global_positions(s_full, cp_size)
  kv_idx = jnp.arange(s_full, dtype=jnp.int32)
  valid = (seg_q_lb[:, None, :, None] == seg_kv_full[:, None, None, :]) & (
      q_idx[None, None, :, None] >= kv_idx[None, None, None, :]
  )
  if not is_global:
    valid = valid & ((q_idx[None, None, :, None] - kv_idx[None, None, None, :]) < cfg.model.sliding_window)

  masked_scores = jnp.where(valid, scores, jnp.asarray(_REF_MASK_VALUE, dtype=jnp.float32))
  sink_col = jnp.broadcast_to(
      sinks_f32.astype(jnp.float32)[None, :, None, None],
      (b_loc, h_q, s_loc, 1),
  )
  logits_with_sink = jnp.concatenate([masked_scores, sink_col], axis=-1)
  probs = jax.nn.softmax(logits_with_sink, axis=-1)[..., :-1].astype(v_full.dtype)

  probs_grp = probs.reshape(b_loc, h_kv, gqa, s_loc, s_full)
  v_prec = jax.lax.Precision.HIGHEST if v_full.dtype == jnp.float32 else jax.lax.Precision.DEFAULT
  return jnp.einsum(
      "bhgst,bthd->bshgd",
      probs_grp,
      v_full,
      precision=v_prec,
  ).reshape(b_loc, s_loc, h_q, head_dim)


def _make_flint_splash_config(
    k_cfg: flint_config.FlintKernelConfig,
    *,
    is_global: bool,
    bq: int,
    bkv: int,
    bkv_c: int,
    bq_dkv: int,
    bkv_dkv: int,
    bkv_dkv_c: int,
    eff_dq_steps: int | None,
) -> flint_splash_kernel.FlintSplashConfig:
  """Builds a `FlintSplashConfig` for global causal or local SWA layers."""
  if is_global:
    st_q = k_cfg.splash_global_num_stacked_q_heads or k_cfg.splash_num_stacked_q_heads
    st_dkv = k_cfg.splash_global_num_stacked_q_heads_dkv or k_cfg.splash_num_stacked_q_heads_dkv
    use_sched = (
        k_cfg.splash_use_scheduler if k_cfg.splash_global_use_scheduler is None else k_cfg.splash_global_use_scheduler
    )
    use_sched_dkv = (
        k_cfg.splash_use_scheduler_dkv
        if k_cfg.splash_global_use_scheduler_dkv is None
        else k_cfg.splash_global_use_scheduler_dkv
    )
    reuse_sq = (
        k_cfg.splash_dkv_reuse_scaled_q
        if k_cfg.splash_global_dkv_reuse_scaled_q is None
        else k_cfg.splash_global_dkv_reuse_scaled_q
    )
    dkv_stitch = (
        k_cfg.splash_dkv_head_stitch
        if k_cfg.splash_global_dkv_head_stitch is None
        else k_cfg.splash_global_dkv_head_stitch
    )
  else:
    st_q = k_cfg.splash_num_stacked_q_heads
    st_dkv = k_cfg.splash_num_stacked_q_heads_dkv
    use_sched = k_cfg.splash_use_scheduler
    use_sched_dkv = k_cfg.splash_use_scheduler_dkv
    reuse_sq = k_cfg.splash_dkv_reuse_scaled_q
    dkv_stitch = k_cfg.splash_dkv_head_stitch

  return flint_splash_kernel.FlintSplashConfig(
      block_q=bq,
      block_kv=bkv,
      block_kv_compute=bkv_c,
      block_q_dkv=bq_dkv,
      block_kv_dkv=bkv_dkv,
      block_kv_dkv_compute=bkv_dkv_c,
      use_fused_bwd_kernel=True,
      dq_reduction_steps=eff_dq_steps,
      use_base2_exp=k_cfg.splash_use_base2_exp,
      fuse_reciprocal=k_cfg.splash_fuse_reciprocal,
      qk_diag_skip=k_cfg.splash_qk_diag_skip if is_global else False,
      sv_diag_skip=k_cfg.splash_sv_diag_skip if is_global else False,
      num_stacked_q_heads=st_q,
      num_stacked_q_heads_dkv=st_dkv,
      use_experimental_scheduler=use_sched,
      use_experimental_scheduler_dkv=use_sched_dkv,
      dkv_reuse_scaled_q=reuse_sq,
      fwd_strip_schedule=k_cfg.splash_fwd_strip_schedule,
      fwd_strip_hoist_q_scale=k_cfg.splash_fwd_strip_hoist_q_scale,
      compact_lse_residual=k_cfg.splash_compact_lse_residual,
      dkv_head_stitch=dkv_stitch,
      residual_checkpoint_name="context",
  )


def _splash_local_halo_attention(
    q_bhsd: jax.Array,
    k_bhsd: jax.Array,
    v_bhsd: jax.Array,
    seg_local: jax.Array | None,
    sinks_f32: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    load_balanced: bool = False,
) -> jax.Array:
  """Runs 1-neighbor `ppermute` halo SplashAttention on natural or LB shards."""
  s_loc = q_bhsd.shape[2]
  cp_size = cfg.sharding.cp_size
  k_cfg = cfg.kernels
  window_left = cfg.model.sliding_window - 1
  stripe_len = (s_loc // 2) if load_balanced else s_loc

  bq = _clamp_splash_tile(k_cfg.splash_block_q, stripe_len)
  bkv = _clamp_splash_tile(k_cfg.splash_block_kv, stripe_len)
  bkv_c = _clamp_splash_tile(k_cfg.splash_block_kv_compute, bkv)
  bq_dkv = _clamp_splash_tile(k_cfg.splash_block_q_dkv, stripe_len)
  bkv_dkv = _clamp_splash_tile(k_cfg.splash_block_kv_dkv, stripe_len)
  bkv_dkv_c = _clamp_splash_tile(k_cfg.splash_block_kv_dkv_compute, bkv_dkv)
  for tile_name, tile_val in (
      ("block_q", bq),
      ("block_kv", bkv),
      ("block_kv_compute", bkv_c),
      ("block_q_dkv", bq_dkv),
      ("block_kv_dkv", bkv_dkv),
      ("block_kv_dkv_compute", bkv_dkv_c),
  ):
    if tile_val < _SPLASH_NUM_LANES or tile_val % _SPLASH_NUM_LANES != 0:
      raise ValueError(
          f"Clamped halo Splash {tile_name}={tile_val} must be >= 128 and"
          f" divisible by 128 (stripe_len={stripe_len}, cp_size={cp_size})."
      )

  kv_lcm = math.lcm(bq, bkv, bq_dkv, bkv_dkv)
  halo_len = math.ceil(max(1, window_left) / kv_lcm) * kv_lcm
  if stripe_len < halo_len or stripe_len % kv_lcm != 0:
    raise ValueError(
        f"Local halo SWA requires stripe_len={stripe_len} >=" f" halo_len={halo_len} and divisible by kv_lcm={kv_lcm}."
    )

  cp_rank = jax.lax.axis_index("cp")
  perm_fwd = [(r, (r + 1) % cp_size) for r in range(cp_size)]
  if load_balanced:
    perm_bwd = [(r, (r - 1) % cp_size) for r in range(cp_size)]
    is_last_cp = cp_rank == (cp_size - 1)
    k_s0 = k_bhsd[:, :, :stripe_len, :]
    k_s1 = k_bhsd[:, :, stripe_len:, :]
    v_s0 = v_bhsd[:, :, :stripe_len, :]
    v_s1 = v_bhsd[:, :, stripe_len:, :]
    kv_tail_s0 = jnp.stack([k_s0[:, :, -halo_len:, :], v_s0[:, :, -halo_len:, :]], axis=0)
    kv_tail_s1 = jnp.stack([k_s1[:, :, -halo_len:, :], v_s1[:, :, -halo_len:, :]], axis=0)
    kv_halo_s0 = jax.lax.ppermute(kv_tail_s0, "cp", perm=perm_fwd)
    kv_halo_s1_recv = jax.lax.ppermute(kv_tail_s1, "cp", perm=perm_bwd)
    kv_halo_s1 = jnp.where(is_last_cp, kv_tail_s0, kv_halo_s1_recv)
    k_ext_bhsd = jnp.concatenate([kv_halo_s0[0], k_s0, kv_halo_s1[0], k_s1], axis=2)
    v_ext_bhsd = jnp.concatenate([kv_halo_s0[1], v_s0, kv_halo_s1[1], v_s1], axis=2)
    if seg_local is None:
      seg_q_in = None
      seg_kv_in = None
    else:
      seg_s0 = seg_local[:, :stripe_len]
      seg_s1 = seg_local[:, stripe_len:]
      seg_tail_s0 = seg_s0[:, -halo_len:]
      seg_tail_s1 = seg_s1[:, -halo_len:]
      seg_halo_s0 = jax.lax.ppermute(seg_tail_s0, "cp", perm=perm_fwd)
      seg_halo_s0 = jnp.where(cp_rank == 0, jnp.int32(-1), seg_halo_s0)
      seg_halo_s1_recv = jax.lax.ppermute(seg_tail_s1, "cp", perm=perm_bwd)
      seg_halo_s1 = jnp.where(is_last_cp, seg_tail_s0, seg_halo_s1_recv)
      seg_q_in = seg_local
      seg_kv_in = jnp.concatenate([seg_halo_s0, seg_s0, seg_halo_s1, seg_s1], axis=1)
  else:
    kv_tail = jnp.stack([k_bhsd[:, :, -halo_len:, :], v_bhsd[:, :, -halo_len:, :]], axis=0)
    kv_halo = jax.lax.ppermute(kv_tail, "cp", perm=perm_fwd)
    k_ext_bhsd = jnp.concatenate([kv_halo[0], k_bhsd], axis=2)
    v_ext_bhsd = jnp.concatenate([kv_halo[1], v_bhsd], axis=2)
    if seg_local is None:
      seg_q_in = None
      seg_kv_in = None
    else:
      seg_halo = jax.lax.ppermute(seg_local[:, -halo_len:], "cp", perm=perm_fwd)
      seg_halo = jnp.where(cp_rank == 0, jnp.int32(-1), seg_halo)
      seg_q_in = seg_local
      seg_kv_in = jnp.concatenate([seg_halo, seg_local], axis=1)

  fwd_info_np, mask_fn_fwd = make_local_halo_mask_info(
      s_loc,
      halo_len,
      window_left,
      cp_size,
      (bq, bkv),
      is_dkv=False,
      return_dynamic_grid=True,
      load_balanced=load_balanced,
  )
  fwd_sparsity = 1.0 if fwd_info_np.block_mask is None else float(np.mean(fwd_info_np.block_mask > 0))

  req_dq_steps = (
      k_cfg.splash_dq_reduction_steps
      if k_cfg.splash_local_dq_reduction_steps is None
      else k_cfg.splash_local_dq_reduction_steps
  )
  dkv_info_np, _ = make_local_halo_mask_info(
      s_loc,
      halo_len,
      window_left,
      cp_size,
      (bq_dkv, bkv_dkv),
      is_dkv=True,
      return_dynamic_grid=req_dq_steps in (1, 3),
      load_balanced=load_balanced,
  )
  q_blocks_per_shard = s_loc // bq_dkv
  if req_dq_steps == 1 and not (tokamax_splash_mask_info.dkv_q_blocks_are_contiguous(dkv_info_np, q_blocks_per_shard)):
    eff_dq_steps: int | None = 3
  else:
    eff_dq_steps = req_dq_steps

  sa_config = _make_flint_splash_config(
      k_cfg,
      is_global=False,
      bq=bq,
      bkv=bkv,
      bkv_c=bkv_c,
      bq_dkv=bq_dkv,
      bkv_dkv=bkv_dkv,
      bkv_dkv_c=bkv_dkv_c,
      eff_dq_steps=eff_dq_steps,
  )
  dkv_sparsity = 1.0 if dkv_info_np.block_mask is None else float(np.mean(dkv_info_np.block_mask > 0))

  fwd_mask_info = jax.tree.map(jnp.asarray, fwd_info_np)
  dkv_mask_info = jax.tree.map(jnp.asarray, dkv_info_np)
  local_fwd_info = _local_tokamax_mask_info(fwd_mask_info, cp_rank, cp_size)
  local_dkv_info = _local_tokamax_mask_info(dkv_mask_info, cp_rank, cp_size)
  assert local_fwd_info is not None

  q_res_bhsd = jax.ad_checkpoint.checkpoint_name(q_bhsd, "attn_q_res")
  ctx_bhsd = _splash_batched(
      sa_config,
      tokamax_splash_base.DEFAULT_MASK_VALUE,
      mask_fn_fwd,
      fwd_sparsity,
      dkv_sparsity,
      False,
      local_fwd_info,
      local_dkv_info,
      q_res_bhsd,
      k_ext_bhsd,
      v_ext_bhsd,
      seg_q_in,
      seg_kv_in,
      sinks_f32,
  )
  return jnp.swapaxes(ctx_bhsd, 1, 2)


def _splash_attention(
    q_lb_bhsd: jax.Array,
    k_full_bhsd: jax.Array,
    v_full_bhsd: jax.Array,
    seg_q_lb: jax.Array | None,
    seg_kv_full: jax.Array | None,
    sinks_f32: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
) -> jax.Array:
  """Runs Flint SplashAttention on pre-scaled `BHSD` LB CP query shards."""
  s_full = k_full_bhsd.shape[2]
  cp_size = cfg.sharding.cp_size
  k_cfg = cfg.kernels

  half_q = s_full // (2 * cp_size) if cp_size > 1 else s_full
  des_bq = (k_cfg.splash_global_block_q or k_cfg.splash_block_q) if is_global else k_cfg.splash_block_q
  des_bkv = (k_cfg.splash_global_block_kv or k_cfg.splash_block_kv) if is_global else k_cfg.splash_block_kv
  des_bkv_c = (
      (k_cfg.splash_global_block_kv_compute or k_cfg.splash_block_kv_compute)
      if is_global
      else k_cfg.splash_block_kv_compute
  )
  des_bq_dkv = (k_cfg.splash_global_block_q_dkv or k_cfg.splash_block_q_dkv) if is_global else k_cfg.splash_block_q_dkv
  des_bkv_dkv = (
      (k_cfg.splash_global_block_kv_dkv or k_cfg.splash_block_kv_dkv) if is_global else k_cfg.splash_block_kv_dkv
  )
  des_bkv_dkv_c = (
      (k_cfg.splash_global_block_kv_dkv_compute or k_cfg.splash_block_kv_dkv_compute)
      if is_global
      else k_cfg.splash_block_kv_dkv_compute
  )
  use_diag_skip = is_global and (k_cfg.splash_qk_diag_skip or k_cfg.splash_sv_diag_skip)
  kv_limit = half_q if use_diag_skip else s_full
  bq = _clamp_splash_tile(des_bq, half_q)
  bkv = _clamp_splash_tile(des_bkv, kv_limit)
  bkv_c = _clamp_splash_tile(des_bkv_c, bkv)
  bq_dkv = _clamp_splash_tile(des_bq_dkv, half_q)
  bkv_dkv = _clamp_splash_tile(des_bkv_dkv, kv_limit)
  bkv_dkv_c = _clamp_splash_tile(des_bkv_dkv_c, bkv_dkv)

  for tile_name, tile_val in (
      ("block_q", bq),
      ("block_kv", bkv),
      ("block_kv_compute", bkv_c),
      ("block_q_dkv", bq_dkv),
      ("block_kv_dkv", bkv_dkv),
      ("block_kv_dkv_compute", bkv_dkv_c),
  ):
    if tile_val < _SPLASH_NUM_LANES or tile_val % _SPLASH_NUM_LANES != 0:
      raise ValueError(
          f"Clamped Splash {tile_name}={tile_val} must be >= 128 and divisible"
          f" by 128 (s_full={s_full}, cp_size={cp_size})."
      )

  mask: tokamax_splash_mask.CausalMask | tokamax_splash_mask.LocalMask
  if is_global:
    mask = tokamax_splash_mask.CausalMask(shape=(s_full, s_full), offset=0)
  else:
    left_window = cfg.model.sliding_window - 1
    mask = tokamax_splash_mask.LocalMask(
        shape=(s_full, s_full),
        window_size=(left_window, 0),
        offset=0,
    )
  mask.q_sequence = ops.lb_q_sequence_np(s_full, cp_size)

  fwd_info_np, mask_fn_fwd = make_fast_tokamax_mask_info(
      mask,
      (bq, bkv),
      is_dkv=False,
      return_dynamic_grid=True,
      q_seq_shards=cp_size,
  )
  fwd_sparsity = 1.0 if fwd_info_np.block_mask is None else float(np.mean(fwd_info_np.block_mask > 0))

  req_dq_steps = (
      k_cfg.splash_dq_reduction_steps
      if (is_global or k_cfg.splash_local_dq_reduction_steps is None)
      else k_cfg.splash_local_dq_reduction_steps
  )
  dkv_info_np, _ = make_fast_tokamax_mask_info(
      mask,
      (bq_dkv, bkv_dkv),
      is_dkv=True,
      return_dynamic_grid=req_dq_steps in (1, 3),
      q_seq_shards=cp_size,
  )
  q_blocks_per_shard = s_full // cp_size // bq_dkv
  if req_dq_steps == 1 and not (tokamax_splash_mask_info.dkv_q_blocks_are_contiguous(dkv_info_np, q_blocks_per_shard)):
    eff_dq_steps: int | None = 3
  else:
    eff_dq_steps = req_dq_steps

  sa_config = _make_flint_splash_config(
      k_cfg,
      is_global=is_global,
      bq=bq,
      bkv=bkv,
      bkv_c=bkv_c,
      bq_dkv=bq_dkv,
      bkv_dkv=bkv_dkv,
      bkv_dkv_c=bkv_dkv_c,
      eff_dq_steps=eff_dq_steps,
  )
  dkv_sparsity = 1.0 if dkv_info_np.block_mask is None else float(np.mean(dkv_info_np.block_mask > 0))

  fwd_mask_info = jax.tree.map(jnp.asarray, fwd_info_np)
  dkv_mask_info = jax.tree.map(jnp.asarray, dkv_info_np)
  if cp_size > 1:
    cp_rank = jax.lax.axis_index("cp")
    local_fwd_info = _local_tokamax_mask_info(fwd_mask_info, cp_rank, cp_size)
    local_dkv_info = _local_tokamax_mask_info(dkv_mask_info, cp_rank, cp_size)
  else:
    local_fwd_info = fwd_mask_info
    local_dkv_info = dkv_mask_info
  assert local_fwd_info is not None

  q_bhsd = jax.ad_checkpoint.checkpoint_name(q_lb_bhsd, "attn_q_res")
  ctx_bhsd = _splash_batched(
      sa_config,
      tokamax_splash_base.DEFAULT_MASK_VALUE,
      mask_fn_fwd,
      fwd_sparsity,
      dkv_sparsity,
      is_global,
      local_fwd_info,
      local_dkv_info,
      q_bhsd,
      k_full_bhsd,
      v_full_bhsd,
      seg_q_lb,
      seg_kv_full,
      sinks_f32,
  )
  return jnp.swapaxes(ctx_bhsd, 1, 2)


def _gather_global_kv_bhsd(
    k_bhsd: jax.Array,
    v_bhsd: jax.Array,
    cp_size: int,
    *,
    lb_layout: bool,
    use_pallas_reorder: bool,
) -> tuple[jax.Array, jax.Array]:
  """All-gathers local `BHSD` KV shards into full natural sequence order."""
  if cp_size <= 1:
    return k_bhsd, v_bhsd
  if not lb_layout:
    return (
        ops.cp_all_gather_natural(k_bhsd, cp_size, axis=2),
        ops.cp_all_gather_natural(v_bhsd, cp_size, axis=2),
    )
  if use_pallas_reorder:
    k_ag = jax.lax.all_gather(k_bhsd, "cp", axis=2, tiled=True)
    v_ag = jax.lax.all_gather(v_bhsd, "cp", axis=2, tiled=True)
    return (
        flint_attention_prologue.reorder_lb_kv_to_contiguous(k_ag, cp_size),
        flint_attention_prologue.reorder_lb_kv_to_contiguous(v_ag, cp_size),
    )
  return (
      ops.cp_all_gather_lb_to_natural(k_bhsd, cp_size, axis=2),
      ops.cp_all_gather_lb_to_natural(v_bhsd, cp_size, axis=2),
  )


def _prepare_qk_pair_bhsd(
    q_bshd: jax.Array,
    k_bshd: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    positions_local: jax.Array,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
    pre_attn_scalar: float | None,
) -> tuple[jax.Array, jax.Array]:
  """Applies Q/K RMSNorm, optional RoPE, optional Q scalar, and `BSHD->BHSD`."""
  m = cfg.model
  q_in = jax.ad_checkpoint.checkpoint_name(q_bshd, "attn_qk_pre_norm")
  k_in = jax.ad_checkpoint.checkpoint_name(k_bshd, "attn_qk_pre_norm")
  if cfg.kernels.use_fused_qk_prologue:
    cos_2d: jax.Array | None = None
    sin_2d: jax.Array | None = None
    if not is_global:
      if rope_cos_sin is not None:
        cos_loc, sin_loc = rope_cos_sin
      else:
        cos_loc, sin_loc = ops.compute_split_half_rope_cos_sin(
            positions_local,
            m.head_dim,
            rope_theta=m.rope_theta,
            dtype=cfg.train.compute_jnp_dtype,
        )
      cos_2d = cos_loc[:, :, 0, :]
      sin_2d = sin_loc[:, :, 0, :]
    q_bhsd = flint_attention_prologue.fused_qk_prologue_local(
        q_in,
        q_norm_scale,
        epsilon=m.rms_norm_eps,
        pre_attn_scalar=pre_attn_scalar,
        cos_2d=cos_2d,
        sin_2d=sin_2d,
    )
    k_bhsd = flint_attention_prologue.fused_qk_prologue_local(
        k_in,
        k_norm_scale,
        epsilon=m.rms_norm_eps,
        pre_attn_scalar=None,
        cos_2d=cos_2d,
        sin_2d=sin_2d,
    )
    return q_bhsd, k_bhsd

  q = ops.flint_rms_norm(q_in, q_norm_scale, m.rms_norm_eps)
  k = ops.flint_rms_norm(k_in, k_norm_scale, m.rms_norm_eps)
  if not is_global:
    if rope_cos_sin is not None:
      cos_loc, sin_loc = rope_cos_sin
      q, k = ops.apply_precomputed_split_half_rope(q, k, cos_loc, sin_loc)
    else:
      q, k = ops.apply_split_half_rope(q, k, positions_local, m.head_dim, rope_theta=m.rope_theta)
  if pre_attn_scalar is not None and pre_attn_scalar != 1.0:
    q = (q.astype(jnp.float32) * pre_attn_scalar).astype(q.dtype)
  return jnp.swapaxes(q, 1, 2), jnp.swapaxes(k, 1, 2)


def _prepare_single_nope_bhsd(
    x_bshd: jax.Array,
    norm_scale: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    pre_attn_scalar: float | None = None,
) -> jax.Array:
  """Applies NoPE RMSNorm (+ optional Q scalar) and `BSHD->BHSD`."""
  x_in = jax.ad_checkpoint.checkpoint_name(x_bshd, "attn_qk_pre_norm")
  if cfg.kernels.use_fused_qk_prologue:
    return flint_attention_prologue.fused_qk_prologue_local(
        x_in,
        norm_scale,
        epsilon=cfg.model.rms_norm_eps,
        pre_attn_scalar=pre_attn_scalar,
    )
  y = ops.flint_rms_norm(x_in, norm_scale, cfg.model.rms_norm_eps)
  if pre_attn_scalar is not None and pre_attn_scalar != 1.0:
    y = (y.astype(jnp.float32) * pre_attn_scalar).astype(y.dtype)
  return jnp.swapaxes(y, 1, 2)


def flint_attention_forward(
    x_local: jax.Array,
    weights_shard: flint_types.FlintAttnWeights[jax.Array],
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Runs Flint hybrid GQA attention inside `shard_map(..., check_vma=False)`.

  Args:
    x_local: Local input activations `[B_loc, S_loc, hidden_size]` in natural CP
      order (or 2-stripe LB CP order when `cfg.kernels.use_lb_cp_layout` and
      `cp_size > 1`).
    weights_shard: Single-layer `FlintAttnWeights` shard (`q/k/v_proj` sharded
      on axis 0, `o_proj` on axis 1, `q_norm/k_norm/sinks` replicated `P()`).
    positions_local: Token positions `int32[B_loc, S_loc]` matching `x_local`.
    segment_ids_local: Document segment IDs `int32[B_loc, S_loc]` matching
      `x_local`.
    cfg: Validated `FlintConfig`.
    is_global: If `True`, applies global causal NoPE attention; if `False`,
      applies split-half RoPE and causal sliding-window (`W=513`) attention.
    rope_cos_sin: Optional precomputed `(cos, sin)` tables of shape `[B_loc,
      S_loc, 1, head_dim]` for sliding-window RoPE layers.

  Returns:
    Local attention output `[B_loc, S_loc, hidden_size]` in `compute_dtype` and
    the same CP sequence order as `x_local`.

  Raises:
    ValueError: If input shapes violate config or CP sharding invariants.
  """
  m = cfg.model
  cp_size = cfg.sharding.cp_size
  if x_local.ndim != 3 or x_local.shape[-1] != m.hidden_size:
    raise ValueError(f"Expected x_local shape [B_loc, S_loc, {m.hidden_size}], got" f" {x_local.shape}.")
  b_loc, s_loc, _ = x_local.shape
  if positions_local.shape != (b_loc, s_loc):
    raise ValueError(f"positions_local shape {positions_local.shape} must match" f" {(b_loc, s_loc)}.")
  if segment_ids_local.shape != (b_loc, s_loc):
    raise ValueError(f"segment_ids_local shape {segment_ids_local.shape} must match" f" {(b_loc, s_loc)}.")
  if cp_size > 1 and s_loc % 2 != 0:
    raise ValueError(f"Local CP sequence length s_loc={s_loc} must be even when" f" cp_size={cp_size}.")
  h_q = m.num_attention_heads
  h_kv = m.num_key_value_heads
  head_dim = m.head_dim
  if weights_shard.sinks.shape != (h_q,):
    raise ValueError(f"Expected sinks shape {(h_q,)}, got {weights_shard.sinks.shape}.")
  if weights_shard.q_norm.shape != (head_dim,) or (weights_shard.k_norm.shape != (head_dim,)):
    raise ValueError(
        f"Expected q_norm/k_norm shape {(head_dim,)}, got" f" {weights_shard.q_norm.shape}, {weights_shard.k_norm.shape}."
    )

  compute_dt = cfg.train.compute_jnp_dtype
  matmul_prec = jax.lax.Precision.HIGHEST if compute_dt == jnp.float32 else jax.lax.Precision.DEFAULT
  lb_layout = cfg.kernels.use_lb_cp_layout and cp_size > 1
  use_early_kv = cfg.kernels.early_global_kv_proj and is_global and cp_size > 1 and cfg.kernels.use_splash
  pre_q_scalar = (float(head_dim) ** -0.5) if cfg.kernels.use_splash else None
  sinks_f32 = weights_shard.sinks.astype(jnp.float32)
  ignore_seg = cfg.kernels.use_splash and cfg.kernels.splash_ignore_segment_ids

  x_cast = x_local.astype(compute_dt)
  o_proj_t: jax.Array | None = None
  o_proj: jax.Array | None = None
  o_shard_in: jax.Array | None = None
  pre_o_fp8: tuple[jax.Array, jax.Array] | None = None
  o_sharded_axis = 0 if cfg.kernels.transpose_dense_rhs_ag else 1
  q_dim = h_q * head_dim
  kv_dim = h_kv * head_dim

  if use_early_kv:
    if cfg.kernels.use_fp8:
      if cfg.kernels.fuse_layer_projections:
        kv_shard = jnp.concatenate([weights_shard.k_proj, weights_shard.v_proj], axis=-1)
        kv = ops.fp8_dense_dot_sharded(x_cast, kv_shard, cfg, sharded_axis=0)
        k = kv[..., :kv_dim].reshape(b_loc, s_loc, h_kv, head_dim)
        v = kv[..., kv_dim:].reshape(b_loc, s_loc, h_kv, head_dim)
      else:
        k = ops.fp8_dense_dot_sharded(x_cast, weights_shard.k_proj, cfg, sharded_axis=0).reshape(
            b_loc, s_loc, h_kv, head_dim
        )
        v = ops.fp8_dense_dot_sharded(x_cast, weights_shard.v_proj, cfg, sharded_axis=0).reshape(
            b_loc, s_loc, h_kv, head_dim
        )
    else:
      if cfg.kernels.fuse_layer_projections:
        kv_shard = jnp.concatenate([weights_shard.k_proj, weights_shard.v_proj], axis=-1)
        kv_proj = ops.gather_dense_weight(kv_shard, cfg, sharded_axis=0)
        kv = jnp.matmul(x_cast, kv_proj, precision=matmul_prec)
        k = kv[..., :kv_dim].reshape(b_loc, s_loc, h_kv, head_dim)
        v = kv[..., kv_dim:].reshape(b_loc, s_loc, h_kv, head_dim)
      else:
        k_proj = ops.gather_dense_weight(weights_shard.k_proj, cfg, sharded_axis=0)
        v_proj = ops.gather_dense_weight(weights_shard.v_proj, cfg, sharded_axis=0)
        k = jnp.matmul(x_cast, k_proj, precision=matmul_prec).reshape(b_loc, s_loc, h_kv, head_dim)
        v = jnp.matmul(x_cast, v_proj, precision=matmul_prec).reshape(b_loc, s_loc, h_kv, head_dim)
    k_bhsd = _prepare_single_nope_bhsd(k, weights_shard.k_norm, cfg)
    v_bhsd = jnp.swapaxes(v, 1, 2)
    k_bhsd = jax.ad_checkpoint.checkpoint_name(k_bhsd, "attn_global_kv")
    v_bhsd = jax.ad_checkpoint.checkpoint_name(v_bhsd, "attn_global_kv")
    k_full_bhsd, v_full_bhsd = _gather_global_kv_bhsd(
        k_bhsd,
        v_bhsd,
        cp_size,
        lb_layout=lb_layout,
        use_pallas_reorder=cfg.kernels.use_pallas_lb_kv_reorder,
    )
    if cfg.kernels.use_fp8:
      q = ops.fp8_dense_dot_sharded(x_cast, weights_shard.q_proj, cfg, sharded_axis=0).reshape(
          b_loc, s_loc, h_q, head_dim
      )
      o_shard_in = (
          jnp.swapaxes(weights_shard.o_proj, 0, 1) if cfg.kernels.transpose_dense_rhs_ag else weights_shard.o_proj
      )
      pre_o_fp8 = ops.prefetch_dense_weight_fp8(
          o_shard_in,
          cfg,
          sharded_axis=o_sharded_axis,
          rhs_is_transposed=cfg.kernels.transpose_dense_rhs_ag,
      )
    else:
      q_proj = ops.gather_dense_weight(weights_shard.q_proj, cfg, sharded_axis=0)
      q = jnp.matmul(x_cast, q_proj, precision=matmul_prec).reshape(b_loc, s_loc, h_q, head_dim)
      if cfg.kernels.transpose_dense_rhs_ag:
        o_proj_t = ops.gather_dense_weight(jnp.swapaxes(weights_shard.o_proj, 0, 1), cfg, sharded_axis=0)
      else:
        o_proj = ops.gather_dense_weight(weights_shard.o_proj, cfg, sharded_axis=1)
    q_bhsd = _prepare_single_nope_bhsd(q, weights_shard.q_norm, cfg, pre_attn_scalar=pre_q_scalar)
    q_lb_bhsd = q_bhsd if lb_layout else ops.cp_natural_to_lb(q_bhsd, cp_size, axis=2)
    if ignore_seg:
      seg_q_lb = None
      seg_kv_full = None
    else:
      seg_i32 = segment_ids_local.astype(jnp.int32)
      if lb_layout:
        seg_q_lb = seg_i32
        seg_kv_full = ops.cp_all_gather_lb_to_natural(seg_i32, cp_size, axis=1)
      else:
        seg_q_lb = ops.cp_natural_to_lb(seg_i32, cp_size, axis=1)
        seg_kv_full = ops.cp_all_gather_natural(seg_i32, cp_size, axis=1)
    ctx_lb = _splash_attention(
        q_lb_bhsd,
        k_full_bhsd,
        v_full_bhsd,
        seg_q_lb,
        seg_kv_full,
        sinks_f32,
        cfg,
        is_global=True,
    )
    ctx = ctx_lb if lb_layout else ops.cp_lb_to_natural(ctx_lb, cp_size, axis=1)
  else:
    if cfg.kernels.use_fp8:
      if cfg.kernels.fuse_layer_projections:
        qkv_shard = jnp.concatenate(
            [weights_shard.q_proj, weights_shard.k_proj, weights_shard.v_proj],
            axis=-1,
        )
        qkv = ops.fp8_dense_dot_sharded(x_cast, qkv_shard, cfg, sharded_axis=0)
        q = qkv[..., :q_dim].reshape(b_loc, s_loc, h_q, head_dim)
        k = qkv[..., q_dim : q_dim + kv_dim].reshape(b_loc, s_loc, h_kv, head_dim)
        v = qkv[..., q_dim + kv_dim :].reshape(b_loc, s_loc, h_kv, head_dim)
      else:
        q = ops.fp8_dense_dot_sharded(x_cast, weights_shard.q_proj, cfg, sharded_axis=0).reshape(
            b_loc, s_loc, h_q, head_dim
        )
        k = ops.fp8_dense_dot_sharded(x_cast, weights_shard.k_proj, cfg, sharded_axis=0).reshape(
            b_loc, s_loc, h_kv, head_dim
        )
        v = ops.fp8_dense_dot_sharded(x_cast, weights_shard.v_proj, cfg, sharded_axis=0).reshape(
            b_loc, s_loc, h_kv, head_dim
        )
      o_shard_in = (
          jnp.swapaxes(weights_shard.o_proj, 0, 1) if cfg.kernels.transpose_dense_rhs_ag else weights_shard.o_proj
      )
      pre_o_fp8 = ops.prefetch_dense_weight_fp8(
          o_shard_in,
          cfg,
          sharded_axis=o_sharded_axis,
          rhs_is_transposed=cfg.kernels.transpose_dense_rhs_ag,
      )
    else:
      if cfg.kernels.fuse_layer_projections:
        qkv_shard = jnp.concatenate(
            [weights_shard.q_proj, weights_shard.k_proj, weights_shard.v_proj],
            axis=-1,
        )
        qkv_proj = ops.gather_dense_weight(qkv_shard, cfg, sharded_axis=0)
        qkv = jnp.matmul(x_cast, qkv_proj, precision=matmul_prec)
        q = qkv[..., :q_dim].reshape(b_loc, s_loc, h_q, head_dim)
        k = qkv[..., q_dim : q_dim + kv_dim].reshape(b_loc, s_loc, h_kv, head_dim)
        v = qkv[..., q_dim + kv_dim :].reshape(b_loc, s_loc, h_kv, head_dim)
      else:
        q_proj = ops.gather_dense_weight(weights_shard.q_proj, cfg, sharded_axis=0)
        k_proj = ops.gather_dense_weight(weights_shard.k_proj, cfg, sharded_axis=0)
        v_proj = ops.gather_dense_weight(weights_shard.v_proj, cfg, sharded_axis=0)
        q = jnp.matmul(x_cast, q_proj, precision=matmul_prec).reshape(b_loc, s_loc, h_q, head_dim)
        k = jnp.matmul(x_cast, k_proj, precision=matmul_prec).reshape(b_loc, s_loc, h_kv, head_dim)
        v = jnp.matmul(x_cast, v_proj, precision=matmul_prec).reshape(b_loc, s_loc, h_kv, head_dim)
      if cfg.kernels.transpose_dense_rhs_ag:
        o_proj_t = ops.gather_dense_weight(jnp.swapaxes(weights_shard.o_proj, 0, 1), cfg, sharded_axis=0)
      else:
        o_proj = ops.gather_dense_weight(weights_shard.o_proj, cfg, sharded_axis=1)

    q_bhsd, k_bhsd = _prepare_qk_pair_bhsd(
        q,
        k,
        weights_shard.q_norm,
        weights_shard.k_norm,
        positions_local,
        rope_cos_sin,
        cfg,
        is_global=is_global,
        pre_attn_scalar=pre_q_scalar,
    )
    v_bhsd = jnp.swapaxes(v, 1, 2)
    kv_ckpt_name = "attn_global_kv" if is_global else "attn_local_kv"
    k_bhsd = jax.ad_checkpoint.checkpoint_name(k_bhsd, kv_ckpt_name)
    v_bhsd = jax.ad_checkpoint.checkpoint_name(v_bhsd, kv_ckpt_name)

    use_halo_cp = cp_size > 1 and not is_global and cfg.kernels.use_splash and cfg.kernels.splash_use_local_halo_cp
    if use_halo_cp:
      ctx = _splash_local_halo_attention(
          q_bhsd,
          k_bhsd,
          v_bhsd,
          None if ignore_seg else segment_ids_local.astype(jnp.int32),
          sinks_f32,
          cfg,
          load_balanced=lb_layout,
      )
    else:
      k_full_bhsd, v_full_bhsd = _gather_global_kv_bhsd(
          k_bhsd,
          v_bhsd,
          cp_size,
          lb_layout=lb_layout,
          use_pallas_reorder=cfg.kernels.use_pallas_lb_kv_reorder,
      )
      q_lb_bhsd = q_bhsd if lb_layout else ops.cp_natural_to_lb(q_bhsd, cp_size, axis=2)
      if ignore_seg:
        seg_q_lb = None
        seg_kv_full = None
      else:
        seg_i32 = segment_ids_local.astype(jnp.int32)
        if lb_layout:
          seg_q_lb = seg_i32
          seg_kv_full = ops.cp_all_gather_lb_to_natural(seg_i32, cp_size, axis=1)
        else:
          seg_q_lb = ops.cp_natural_to_lb(seg_i32, cp_size, axis=1)
          seg_kv_full = ops.cp_all_gather_natural(seg_i32, cp_size, axis=1)

      if cfg.kernels.use_splash:
        ctx_lb = _splash_attention(
            q_lb_bhsd,
            k_full_bhsd,
            v_full_bhsd,
            seg_q_lb,
            seg_kv_full,
            sinks_f32,
            cfg,
            is_global=is_global,
        )
      else:
        assert seg_q_lb is not None and seg_kv_full is not None
        ctx_lb = dot_product_attention_ref(
            jnp.swapaxes(q_lb_bhsd, 1, 2),
            jnp.swapaxes(k_full_bhsd, 1, 2),
            jnp.swapaxes(v_full_bhsd, 1, 2),
            seg_q_lb,
            seg_kv_full,
            sinks_f32,
            cfg,
            is_global=is_global,
        )
      ctx = ctx_lb if lb_layout else ops.cp_lb_to_natural(ctx_lb, cp_size, axis=1)
  ctx_flat = ctx.reshape(b_loc, s_loc, q_dim).astype(compute_dt)
  if cfg.kernels.use_fp8:
    assert pre_o_fp8 is not None and o_shard_in is not None
    return ops.fp8_dense_dot_sharded(
        ctx_flat,
        o_shard_in,
        cfg,
        sharded_axis=o_sharded_axis,
        rhs_is_transposed=cfg.kernels.transpose_dense_rhs_ag,
        pre_gathered=pre_o_fp8,
    )
  if o_proj_t is not None:
    return jax.lax.dot_general(ctx_flat, o_proj_t, (((2,), (1,)), ((), ())), precision=matmul_prec)
  assert o_proj is not None
  return jnp.matmul(ctx_flat, o_proj, precision=matmul_prec)
