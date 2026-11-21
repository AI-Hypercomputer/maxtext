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

"""Dynamic FP8 (`e4m3fn` fwd / `e5m2` bwd) grouped matmuls and collective QAG for Muse Spark MoE.

Provides:
- `0.0-ULP` bit-exact pre-all-gather dynamic FP8 FSDP expert weight QAG (`512 MiB` vs `1.0 GiB` BF16),
  supporting both `wo_fsdp_axis1=True` (`gather_axis=1`, batched `48 KiB` `pmax`) and `False` (`gather_axis=2`).
- `ragged_dot_fp8`: `gmm_v2` / `tgmm_v2` `custom_vjp` supporting 2D `[M, D]` and 3D `[M, D // 128, 128]`
  token layouts, `valid_mask` tail NaN/Inf guarding for `mask_padding=False`, decoupled overflow VJP
  modes (`need_lhs_grad`, `need_rhs_grad`), and multi-chunk weight/activation gradient accumulation.
- `fp8_ep_token_all_gather` (Stretch L3): `0.0-ULP`-forward idempotent `512`-block dynamic `f8e4m3fn`
  Ring-of-Experts token `all-gather` matching `gmm_v2`'s in-VMEM `lhs` quantizer bit-for-bit.
- `fp8_ep_combine_psum_scatter` (Stretch L4): post-`post_expert_norm` combine `psum_scatter` whose
  backward `all-gather` sends `512`-block dynamic `f8e4m3fn` cotangents (`31.5 dB` SQNR, `2x` wire reduction).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Any

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
import qwix
from qwix import pallas as qpl

_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0
_GMM_LHS_QUANT_BLOCK = 512
_FP8_TM512_MIN_ROWS_PER_EXPERT = 1792


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PreGatheredFp8Weight:
  """Container for pre-quantized, FSDP-gathered FP8 expert weights reusable across token chunks."""

  qvalue: jax.Array  # f8e4m3fn[G, K, N]
  scale: jax.Array  # f32[1 | G, 1, N]


def is_fp8_moe_active(config: Any, rule: qwix.QtRule | None = None) -> bool:
  """Returns True when dynamic FP8 expert matmuls should run for `MuseSparkMoE`."""
  if config is None:
    return rule is not None
  if getattr(config, "quantization", "") == "fp8_full":
    return True
  if rule is not None and getattr(rule, "weight_qtype", None) == jnp.float8_e4m3fn:
    return True
  active_rule = qpl.get_current_rule("gmm")
  return bool(active_rule is not None and getattr(active_rule, "weight_qtype", None) == jnp.float8_e4m3fn)


def precompute_expert_weight_scales(
    weights: Sequence[jax.Array],
    gather_axes: Sequence[int],
    fsdp_axis: Any,
    *,
    per_expert_scale: bool = False,
) -> tuple[jax.Array, ...]:
  """Computes exact 0-ULP global f8e4m3fn weight scales across FSDP.

  Args:
    weights: Local expert weight shards (`[G, K_shard, N]` or `[G, K, N_shard]`).
    gather_axes: FSDP gather dimension (`1` for contracting-sharded, `2` for output-sharded) per weight.
    fsdp_axis: FSDP mesh axis name (or tuple of names), or `None` when unsharded.
    per_expert_scale: If True, keep per-expert `[G, 1, N]` scales; otherwise `[1, 1, N]` (`RoutedMoE` default).

  Returns:
    Tuple of `float32` scale tensors aligned with `weights`.
  """
  if len(weights) != len(gather_axes):
    raise ValueError(f"weights ({len(weights)}) and gather_axes ({len(gather_axes)}) must have equal length.")

  reduce_axis: int | tuple[int, int] = 1 if per_expert_scale else (0, 1)
  f8_max = jnp.float32(_E4M3_MAX)
  local_amaxes = [
      jnp.max(jnp.abs(jax.lax.stop_gradient(w).astype(jnp.float32)), axis=reduce_axis, keepdims=True)
      for w in weights
  ]

  if fsdp_axis is None:
    return tuple(jnp.where(am == 0.0, jnp.float32(1.0), am / f8_max) for am in local_amaxes)

  reduce_pmax = functools.partial(jax.lax.pmax, axis_name=fsdp_axis)

  # Group contracting-sharded (gather_axis == 1) amaxes by shape so (wi_0, wi_1[, wo]) share a single pmax.
  shape_groups: dict[tuple[int, ...], list[int]] = {}
  for idx, (am, g_axis) in enumerate(zip(local_amaxes, gather_axes)):
    if g_axis == 1:
      shape_groups.setdefault(am.shape, []).append(idx)
    elif g_axis != 2:
      raise ValueError(f"Unsupported expert weight gather_axis={g_axis}; expected 1 or 2.")

  resolved_amaxes = list(local_amaxes)
  for indices in shape_groups.values():
    if len(indices) == 1:
      i = indices[0]
      resolved_amaxes[i] = reduce_pmax(local_amaxes[i])
    else:
      stacked = jnp.stack([local_amaxes[i] for i in indices], axis=0)
      reduced = reduce_pmax(stacked)
      for pos, i in enumerate(indices):
        resolved_amaxes[i] = reduced[pos]

  return tuple(jnp.where(am == 0.0, jnp.float32(1.0), am / f8_max) for am in resolved_amaxes)


def _quantize_and_gather_rhs_impl(
    rhs_sharded: jax.Array,
    rhs_scale: jax.Array,
    *,
    fsdp_axis: Any,
    gather_axis: int,
) -> tuple[jax.Array, jax.Array]:
  """Internal primal helper: quantizes local weight shard to f8e4m3fn and all-gathers across FSDP."""
  f8_max = jnp.float32(_E4M3_MAX)
  rhs_q_local = jnp.clip(rhs_sharded.astype(jnp.float32) / rhs_scale, -f8_max, f8_max).astype(jnp.float8_e4m3fn)
  if fsdp_axis is None:
    return rhs_q_local, rhs_scale

  rhs_q_full = jax.lax.all_gather(rhs_q_local, axis_name=fsdp_axis, axis=gather_axis, tiled=True)

  if gather_axis == 2 and rhs_scale.shape[-1] != rhs_q_full.shape[2]:
    rhs_scale_full = jax.lax.all_gather(rhs_scale, axis_name=fsdp_axis, axis=2, tiled=True)
  else:
    rhs_scale_full = rhs_scale
  return rhs_q_full, rhs_scale_full


def _flattened_minor_dim(x: jax.Array) -> int:
  return int(math.prod(x.shape[1:]))


def _mask_valid_rows(x: jax.Array, valid_mask: jax.Array | None) -> jax.Array:
  """Zeroes invalid `[count : M]` rows via `jnp.where` so uninitialized NaN/Inf tails never propagate."""
  if valid_mask is None:
    return x
  m = x.shape[0]
  mask = valid_mask[:m].reshape(m, *([1] * (x.ndim - 1)))
  return jnp.where(mask, x, jnp.zeros((), dtype=x.dtype))


def _maybe_pad_rows(x: jax.Array, align_m: int) -> tuple[jax.Array, int]:
  m = x.shape[0]
  if align_m <= 1 or m % align_m == 0:
    return x, 0
  pad_m = align_m - (m % align_m)
  pad_spec = [(0, pad_m)] + [(0, 0)] * (x.ndim - 1)
  return jnp.pad(x, pad_spec), pad_m


def _fp8_clamped_gmm_tiles(
    tm: int,
    tk: int,
    tn: int,
    size_m: int,
    size_group: int,
    size_k: int,
    size_n: int,
    *,
    expected_active_rows: int | None = None,
) -> gmm_v2.TileSizes:
  """Selects spill-free FP8 `gmm_v2` tiles (`tm=512` at active `M/G >= 1792` `chunks=1`, `tm=256` at `chunks=2`)."""
  if (tm, tk, tn) == (512, 1024, 1024) and size_k >= 2048 and size_n >= 2048:
    if 0 < size_group < 16:
      tm, tk, tn = 1024, 4096, 1024
    else:
      # `tn=4096` runs out of VMEM. Gate `tm` on `expected_active_rows` when provided, so a buffer that is only
      # partly filled still selects `tm=256`.
      eff_m = int(expected_active_rows) if expected_active_rows is not None else size_m
      opt_tm = 512 if eff_m >= _FP8_TM512_MIN_ROWS_PER_EXPERT * max(size_group, 1) else 256
      tm, tk, tn = opt_tm, 4096, 2048
  elif tn >= 4096 and size_k >= 2048:
    tn = 2048
  return gmm_v2.TileSizes(
      tile_m=tm,
      tile_k=min(tk, gmm_v2.align_to(size_k, 128)),
      tile_n=min(tn, gmm_v2.align_to(size_n, 128)),
  )


def _fp8_clamped_tgmm_tiles(
    tm: int,
    tk: int,
    tn: int,
    size_m: int,
    size_group: int,
    size_k: int,
    size_n: int,
    *,
    expected_active_rows: int | None = None,
) -> gmm_v2.TileSizes:
  """Selects spill-free FP8 `tgmm_v2` tiles (`tm=512` at active `M/G >= 1792` `chunks=1`, `tm=256` at `chunks=2`)."""
  if (tm, tk, tn) == (512, 1024, 1024) and size_k >= 2048 and size_n >= 2048:
    eff_m = int(expected_active_rows) if expected_active_rows is not None else size_m
    opt_tm = 512 if eff_m >= _FP8_TM512_MIN_ROWS_PER_EXPERT * max(size_group, 1) else 256
    tm, tk, tn = opt_tm, 2048, 2048
  elif (tk >= 4096 or tn >= 4096) and size_k >= 2048 and size_n >= 2048:
    tk, tn = min(tk, 2048), min(tn, 2048)
  return gmm_v2.TileSizes(
      tile_m=tm,
      tile_k=min(tk, gmm_v2.align_to(size_k, 128)),
      tile_n=min(tn, gmm_v2.align_to(size_n, 128)),
  )


def _can_use_native_3d_fp8(dim: int) -> bool:
  """`relayout.py` packs 4 8-bit rows per 32b word, requiring `(dim // 128) % 4 == 0` (`dim % 512 == 0`)."""
  return dim >= 512 and dim % 512 == 0


def _simulate_in_vmem_lhs_quant_dequant(lhs_2d: jax.Array) -> jax.Array:
  """CPU reference matching XLA/Mosaic f32-VREG 512-block `gmm_v2` `lhs` quantizer."""
  m, k = lhs_2d.shape
  blk = math.gcd(k, _GMM_LHS_QUANT_BLOCK)
  x_blk = lhs_2d.astype(jnp.bfloat16).astype(jnp.float32).reshape(m, k // blk, blk)
  block_abs_max = jnp.max(jnp.abs(x_blk), axis=-1, keepdims=True)
  block_scale = block_abs_max / jnp.float32(_E4M3_MAX)
  block_scale_inv = jnp.where(block_scale == 0.0, jnp.float32(0.0), jnp.float32(1.0) / block_scale)
  block_q = (x_blk * block_scale_inv).astype(jnp.float8_e4m3fn)
  return (block_q.astype(jnp.float32) * block_scale).reshape(m, k)


def quantize_lhs_rowwise(lhs: jax.Array, valid_mask: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
  """Rowwise dynamic `f8e4m3fn` activation quantizer (usable for `drhs` and pre-quantized `lhs` A/B)."""
  lhs_clean = _mask_valid_rows(lhs, valid_mask)
  lhs_f32 = lhs_clean.astype(jnp.float32)
  reduce_axes = tuple(range(1, lhs_f32.ndim))
  f8_max = jnp.float32(_E4M3_MAX)
  amax = jnp.max(jnp.abs(lhs_f32), axis=reduce_axes, keepdims=True)
  scale = jnp.where(amax == 0.0, jnp.float32(1.0), amax / f8_max)
  # Barrier on unmasked `lhs` + `scale` prevents XLA from spilling `lhs_f32` (`f32[M, K]`) or an extra
  # masked `bf16[M, K]` buffer to HBM across the `amax` reduction while preserving `HloCSE` (#317/#318).
  lhs_b, scale_b = jax.lax.optimization_barrier((lhs, scale))
  lhs_b_f32 = _mask_valid_rows(lhs_b, valid_mask).astype(jnp.float32)
  lhs_q = jnp.clip(lhs_b_f32 / scale_b, -f8_max, f8_max).astype(jnp.float8_e4m3fn)
  return lhs_q, scale


def ragged_dot_fp8(
    lhs: jax.Array,
    rhs_sharded: jax.Array,
    rhs_scale: jax.Array,
    group_sizes: jax.Array,
    *,
    fsdp_axis: Any,
    gather_axis: int,
    tiling: tuple[int, ...],
    use_gmm_v2: bool,
    use_heuristic_tiling: bool = False,
    valid_mask: jax.Array | None = None,
    out_is_3d: bool = False,
    need_lhs_grad: bool = True,
    need_rhs_grad: bool = True,
    pre_gathered_rhs_q: PreGatheredFp8Weight | jax.Array | None = None,
    dlhs_partial_sum: jax.Array | None = None,
    drhs_partial_sum: jax.Array | None = None,
    scatter_drhs: bool = True,
    return_lhs: bool = False,
    return_rhs: bool = False,
    bwd_lhs_override: jax.Array | None = None,
    mask_dlhs_output: bool = True,
    expected_active_rows: int | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
  """Dynamic FP8 (`e4m3fn` fwd & in-VMEM `dlhs` / `e4m3fn`x`e5m2` `drhs`) grouped matmul with `0-ULP` FSDP QAG."""
  zero_init = valid_mask is None
  has_pre_gathered = pre_gathered_rhs_q is not None
  pre_q_is_struct = isinstance(pre_gathered_rhs_q, PreGatheredFp8Weight)
  pre_q_scale_shape = pre_gathered_rhs_q.scale.shape if pre_q_is_struct else ()
  pre_q_scale_dtype = pre_gathered_rhs_q.scale.dtype if pre_q_is_struct else jnp.float32
  use_reuse_bwd_rhs = bool(fsdp_axis is not None or has_pre_gathered or return_rhs)
  lhs_dtype = lhs.dtype
  rhs_dtype = rhs_sharded.dtype
  rhs_shape = rhs_sharded.shape
  orig_m = lhs.shape[0]
  lhs_ndim = lhs.ndim

  def _resolve_rhs(rhs_in: jax.Array, scale_in: jax.Array, pre_q_in: PreGatheredFp8Weight | jax.Array | None):
    if pre_q_in is not None:
      if isinstance(pre_q_in, PreGatheredFp8Weight):
        return pre_q_in.qvalue, pre_q_in.scale
      return pre_q_in, scale_in
    return _quantize_and_gather_rhs_impl(
        rhs_in,
        scale_in,
        fsdp_axis=fsdp_axis,
        gather_axis=gather_axis,
    )

  def _run_fwd_kernel(
      lhs_in: jax.Array,
      rhs_q_full: jax.Array,
      rhs_scale_full: jax.Array,
      grp_in: jax.Array,
      vmask_in: jax.Array | None,
  ):
    num_groups, size_k, size_n = rhs_q_full.shape
    if out_is_3d and size_n % 128 != 0:
      raise ValueError(f"out_is_3d=True requires size_n ({size_n}) % 128 == 0.")

    if not use_gmm_v2:
      lhs_2d = lhs_in.reshape(orig_m, size_k)
      lhs_deq = _simulate_in_vmem_lhs_quant_dequant(lhs_2d)
      rhs_deq = rhs_q_full.astype(jnp.float32) * rhs_scale_full
      out_2d = jax.lax.ragged_dot(
          lhs_deq,
          rhs_deq,
          grp_in,
          precision=jax.lax.Precision.HIGHEST,
      ).astype(lhs_dtype)
      out = out_2d.reshape(orig_m, size_n // 128, 128) if out_is_3d else out_2d
      return out

    if size_k % 128 != 0 or size_n % 128 != 0:
      raise ValueError(
          f"MuseSparkMoE FP8 gmm_v2 requires K, N multiples of 128; got ({num_groups}, {size_k}, {size_n})."
      )
    fwd_tile = (
        gmm_v2.calculate_tiling
        if use_heuristic_tiling
        else _fp8_clamped_gmm_tiles(
            tiling[0],
            tiling[1],
            tiling[2],
            orig_m,
            num_groups,
            size_k,
            size_n,
            expected_active_rows=expected_active_rows,
        )
    )
    pad_align_m = max(fwd_tile.tile_m if isinstance(fwd_tile, gmm_v2.TileSizes) else tiling[0], 16)
    use_kern_3d_in = lhs_in.ndim == 3 and _can_use_native_3d_fp8(size_k)
    use_kern_3d_out = out_is_3d and _can_use_native_3d_fp8(size_n)
    lhs_for_kern = lhs_in if use_kern_3d_in else lhs_in.reshape(orig_m, size_k)
    lhs_pad, pad_m = _maybe_pad_rows(lhs_for_kern, pad_align_m)
    rhs_scale_gmm = jnp.broadcast_to(rhs_scale_full[:, None, :, :], (num_groups, 1, 1, size_n))
    out_pad = gmm_v2.gmm_v2(
        lhs=lhs_pad,
        rhs=rhs_q_full,
        group_sizes=grp_in,
        rhs_scale=rhs_scale_gmm,
        lhs_scale=None,
        maybe_quantize_lhs=True,
        tile_info=fwd_tile,
        preferred_element_type=lhs_dtype,
        zero_initialize=zero_init,
        out_is_3d=use_kern_3d_out,
    )
    out = out_pad[:orig_m] if pad_m > 0 else out_pad
    if out_is_3d and not use_kern_3d_out:
      out = out.reshape(orig_m, size_n // 128, 128)
    return out

  def _compute_dlhs(
      rhs_q_full: jax.Array,
      rhs_scale_full: jax.Array,
      grp_in: jax.Array,
      g: jax.Array,
      vmask: jax.Array | None,
      dlhs_ps: jax.Array | None,
  ) -> jax.Array:
    num_groups, size_k, size_n = rhs_q_full.shape
    g_clean = _mask_valid_rows(g, vmask)
    s_max = jnp.max(jnp.abs(rhs_scale_full))
    s_max_safe = jnp.where(s_max == 0.0, jnp.float32(1.0), s_max)
    s_norm = rhs_scale_full / s_max_safe
    if s_norm.shape[0] == 1:
      s_n_norm = s_norm.reshape(1, *g_clean.shape[1:])
      g_pre = (g_clean.astype(jnp.float32) * s_n_norm).astype(lhs_dtype)
    else:
      s_rep_bf16 = jnp.repeat(
          s_norm.squeeze(1).astype(lhs_dtype), grp_in, axis=0, total_repeat_length=orig_m
      ).reshape(orig_m, *g_clean.shape[1:])
      g_pre = (g_clean.astype(jnp.float32) * s_rep_bf16.astype(jnp.float32)).astype(lhs_dtype)
    if not use_gmm_v2:
      g_deq = _simulate_in_vmem_lhs_quant_dequant(g_pre.reshape(orig_m, size_n))
      rhs_t = jnp.swapaxes(rhs_q_full, 1, 2).astype(jnp.float32) * s_max_safe
      d_lhs_2d = jax.lax.ragged_dot(g_deq, rhs_t, grp_in, precision=jax.lax.Precision.HIGHEST)
      d_lhs = d_lhs_2d.reshape(orig_m, size_k // 128, 128) if lhs_ndim == 3 else d_lhs_2d
      d_lhs = _mask_valid_rows(d_lhs.astype(lhs_dtype), vmask)
      if dlhs_ps is not None:
        d_lhs = d_lhs + dlhs_ps.astype(lhs_dtype)
      return d_lhs

    if use_heuristic_tiling:
      dlhs_tile: Any = gmm_v2.calculate_tiling
    else:
      dlhs_tile = _fp8_clamped_gmm_tiles(
          tiling[3],
          tiling[4],
          tiling[5],
          orig_m,
          num_groups,
          size_n,
          size_k,
          expected_active_rows=expected_active_rows,
      )
    pad_align_dlhs = max(dlhs_tile.tile_m if isinstance(dlhs_tile, gmm_v2.TileSizes) else tiling[3], 16)
    use_dlhs_3d_in = g_pre.ndim == 3 and _can_use_native_3d_fp8(size_n)
    use_dlhs_3d_out = (lhs_ndim == 3) and _can_use_native_3d_fp8(size_k)
    g_kern = g_pre if use_dlhs_3d_in else g_pre.reshape(orig_m, size_n)
    g_pad, pad_dlhs = _maybe_pad_rows(g_kern, pad_align_dlhs)
    ps_kern = None
    can_fuse_ps = (
        dlhs_ps is not None
        and ((not use_dlhs_3d_out) or dlhs_ps.ndim == 3)
        and (not (lhs_ndim == 3 and not use_dlhs_3d_out))
    )
    if can_fuse_ps:
      assert dlhs_ps is not None
      ps_aligned = dlhs_ps.astype(lhs_dtype)
      if not use_dlhs_3d_out:
        ps_aligned = ps_aligned.reshape(orig_m, size_k)
      ps_kern, _ = _maybe_pad_rows(ps_aligned, pad_align_dlhs)
    dlhs_rhs_scale = jnp.broadcast_to(s_max_safe.reshape(1, 1, 1, 1), (num_groups, 1, 1, size_k))
    d_lhs_pad = gmm_v2.gmm_v2(
        lhs=g_pad,
        rhs=rhs_q_full,
        group_sizes=grp_in,
        rhs_scale=dlhs_rhs_scale,
        partial_sum=ps_kern,
        lhs_scale=None,
        maybe_quantize_lhs=True,
        tile_info=dlhs_tile,
        preferred_element_type=lhs_dtype,
        transpose_rhs=True,
        zero_initialize=zero_init,
        out_is_3d=use_dlhs_3d_out,
    )
    d_lhs = d_lhs_pad[:orig_m] if pad_dlhs > 0 else d_lhs_pad
    if lhs_ndim == 3 and not use_dlhs_3d_out:
      d_lhs = d_lhs.reshape(orig_m, size_k // 128, 128)
    if mask_dlhs_output:
      d_lhs = _mask_valid_rows(d_lhs, vmask)
    if dlhs_ps is not None and not can_fuse_ps:
      d_lhs = d_lhs + dlhs_ps.astype(lhs_dtype)
    return d_lhs

  def _compute_drhs_full(
      lhs_in: jax.Array,
      num_groups: int,
      size_k: int,
      size_n: int,
      grp_in: jax.Array,
      g: jax.Array,
      vmask: jax.Array | None,
      drhs_ps: jax.Array | None,
  ) -> jax.Array:
    lhs_q, lhs_row_scale = quantize_lhs_rowwise(lhs_in.reshape(orig_m, size_k), vmask)
    g_clean = _mask_valid_rows(g, vmask).reshape(orig_m, size_n)
    s_row = lhs_row_scale.reshape(orig_m, *([1] * (g_clean.ndim - 1)))
    drhs_g = g_clean.astype(jnp.float32) * s_row

    e5_max = jnp.float32(_E5M2_MAX)
    drhs_amax = jnp.max(jnp.abs(drhs_g), axis=0, keepdims=True)
    drhs_scale = jnp.where(drhs_amax == 0.0, jnp.float32(1.0), drhs_amax / e5_max)
    # Barrier on `(g_clean, s_row, drhs_scale)` eliminates the `f32[M, N]` (`805 MiB`) `drhs_g` HBM spill
    # across the column `drhs_amax` reduction at strict `0.0-ULP` parity (#317/#318).
    g_b, s_row_b, drhs_scale_b = jax.lax.optimization_barrier((g_clean, s_row, drhs_scale))
    drhs_q = jnp.clip((g_b.astype(jnp.float32) * s_row_b) / drhs_scale_b, -e5_max, e5_max).astype(jnp.float8_e5m2)
    tgmm_rhs_scale = drhs_scale.reshape(1, 1, size_n)

    if not use_gmm_v2:
      lhs_2d = lhs_q.reshape(orig_m, size_k).astype(jnp.float32)
      rhs_2d = drhs_q.reshape(orig_m, size_n).astype(jnp.float32)
      dim_nums = jax.lax.RaggedDotDimensionNumbers(
          dot_dimension_numbers=(((0,), (0,)), ((), ())),
          lhs_ragged_dimensions=(0,),
          rhs_group_dimensions=(),
      )
      d_rhs_f32 = jax.lax.ragged_dot_general(
          lhs=lhs_2d,
          rhs=rhs_2d,
          group_sizes=grp_in,
          ragged_dot_dimension_numbers=dim_nums,
          precision=jax.lax.Precision.HIGHEST,
      )[:num_groups]
      d_rhs_full = (d_rhs_f32 * tgmm_rhs_scale).astype(rhs_dtype)
      if drhs_ps is not None:
        d_rhs_full = (d_rhs_full.astype(jnp.float32) + drhs_ps.astype(jnp.float32)).astype(rhs_dtype)
      return d_rhs_full

    drhs_tile = (
        tgmm_v2.calculate_tgmm_tiling
        if use_heuristic_tiling
        else _fp8_clamped_tgmm_tiles(
            tiling[6],
            tiling[7],
            tiling[8],
            orig_m,
            num_groups,
            size_k,
            size_n,
            expected_active_rows=expected_active_rows,
        )
    )
    pad_align_drhs = max(drhs_tile.tile_m if isinstance(drhs_tile, gmm_v2.TileSizes) else tiling[6], 16)
    lhs_q_kern = lhs_q if (lhs_q.ndim == 3 and _can_use_native_3d_fp8(size_k)) else lhs_q.reshape(orig_m, size_k)
    drhs_q_kern = drhs_q if (drhs_q.ndim == 3 and _can_use_native_3d_fp8(size_n)) else drhs_q.reshape(orig_m, size_n)
    lhs_q_pad, _ = _maybe_pad_rows(lhs_q_kern, pad_align_drhs)
    drhs_q_pad, _ = _maybe_pad_rows(drhs_q_kern, pad_align_drhs)
    ps_arg = None if drhs_ps is None else drhs_ps.astype(rhs_dtype)
    return tgmm_v2.tgmm_v2(
        lhs=lhs_q_pad,
        rhs=drhs_q_pad,
        group_sizes=grp_in,
        num_actual_groups=num_groups,
        rhs_scale=tgmm_rhs_scale,
        partial_sum=ps_arg,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=rhs_dtype,
        tile_info=drhs_tile,
    )

  def _maybe_scatter_drhs(d_rhs_full: jax.Array) -> jax.Array:
    if not scatter_drhs or fsdp_axis is None:
      return d_rhs_full.astype(rhs_dtype)
    return jax.lax.psum_scatter(
        d_rhs_full, axis_name=fsdp_axis, scatter_dimension=gather_axis, tiled=True
    ).astype(rhs_dtype)

  scale_const = jax.lax.stop_gradient(rhs_scale)

  # Mode 1: Activation cotangent only (e.g. `decouple_overflow_bwd` fast activation pass).
  # Pass all array operands explicitly (zero closed-over tracers under `scan_layers` + `jax.checkpoint`).
  if need_lhs_grad and not need_rhs_grad:
    rhs_const = jax.lax.stop_gradient(rhs_sharded)
    pre_q_const = jax.lax.stop_gradient(pre_gathered_rhs_q) if has_pre_gathered else None

    @jax.custom_vjp
    def _dot_lhs_fn(lhs_in, rhs_in, scale_in, pre_q_in, grp_in, vmask_in, dlhs_ps_in):
      del dlhs_ps_in
      _, grp_b = jax.lax.optimization_barrier((lhs_in, grp_in))
      rhs_q_full, rhs_scale_full = _resolve_rhs(rhs_in, scale_in, pre_q_in)
      out = _run_fwd_kernel(lhs_in, rhs_q_full, rhs_scale_full, grp_b, vmask_in)
      return (out, lhs_in) if return_lhs else out

    def _dot_lhs_fwd(lhs_in, rhs_in, scale_in, pre_q_in, grp_in, vmask_in, dlhs_ps_in):
      _, grp_b = jax.lax.optimization_barrier((lhs_in, grp_in))
      rhs_q_full, rhs_scale_full = _resolve_rhs(rhs_in, scale_in, pre_q_in)
      out = _run_fwd_kernel(lhs_in, rhs_q_full, rhs_scale_full, grp_b, vmask_in)
      rhs_res = (rhs_q_full, rhs_scale_full) if use_reuse_bwd_rhs else (rhs_in, scale_in)
      primals_out = (out, lhs_in) if return_lhs else out
      return primals_out, (rhs_res, grp_b, vmask_in, dlhs_ps_in)

    def _dot_lhs_bwd(res, grads):
      rhs_res, grp_b, vmask_in, dlhs_ps_in = res
      g_out, g_lhs_ret = grads if return_lhs else (grads, None)
      _, grp_b = jax.lax.optimization_barrier((g_out, grp_b))
      if use_reuse_bwd_rhs:
        rhs_q_full, rhs_scale_full = rhs_res
      else:
        rhs_q_full, rhs_scale_full = _resolve_rhs(rhs_res[0], rhs_res[1], None)
      combined_dlhs_ps = g_lhs_ret if dlhs_ps_in is None else (
          dlhs_ps_in if g_lhs_ret is None else dlhs_ps_in + g_lhs_ret
      )
      d_lhs = _compute_dlhs(rhs_q_full, rhs_scale_full, grp_b, g_out, vmask_in, combined_dlhs_ps)
      d_ps = jnp.zeros_like(dlhs_ps_in) if dlhs_ps_in is not None else None
      return d_lhs, None, None, None, None, None, d_ps

    _dot_lhs_fn.defvjp(_dot_lhs_fwd, _dot_lhs_bwd)
    return _dot_lhs_fn(lhs, rhs_const, scale_const, pre_q_const, group_sizes, valid_mask, dlhs_partial_sum)

  # Mode 2: Weight cotangent only (e.g. `decouple_overflow_bwd` weight pass).
  lhs_bwd_const = jax.lax.stop_gradient(bwd_lhs_override) if bwd_lhs_override is not None else None

  if need_rhs_grad and not need_lhs_grad:
    lhs_const = jax.lax.stop_gradient(lhs)

    @jax.custom_vjp
    def _dot_rhs_fn(lhs_in, rhs_in, scale_in, pre_q_in, grp_in, vmask_in, drhs_ps_in, lhs_bwd_in):
      del drhs_ps_in, lhs_bwd_in
      _, grp_b = jax.lax.optimization_barrier((lhs_in, grp_in))
      rhs_q_full, rhs_scale_full = _resolve_rhs(rhs_in, scale_in, pre_q_in)
      out = _run_fwd_kernel(lhs_in, rhs_q_full, rhs_scale_full, grp_b, vmask_in)
      rhs_ret = PreGatheredFp8Weight(qvalue=rhs_q_full, scale=rhs_scale_full)
      return (out, rhs_ret) if return_rhs else out

    def _dot_rhs_fwd(lhs_in, rhs_in, scale_in, pre_q_in, grp_in, vmask_in, drhs_ps_in, lhs_bwd_in):
      _, grp_b = jax.lax.optimization_barrier((lhs_in, grp_in))
      rhs_q_full, rhs_scale_full = _resolve_rhs(rhs_in, scale_in, pre_q_in)
      out = _run_fwd_kernel(lhs_in, rhs_q_full, rhs_scale_full, grp_b, vmask_in)
      shape_meta = rhs_q_full.shape
      rhs_ret = PreGatheredFp8Weight(qvalue=rhs_q_full, scale=rhs_scale_full)
      primals_out = (out, rhs_ret) if return_rhs else out
      saved_lhs = lhs_bwd_in if lhs_bwd_in is not None else lhs_in
      return primals_out, (saved_lhs, shape_meta, grp_b, vmask_in, drhs_ps_in)

    def _dot_rhs_bwd(res, grads):
      lhs_saved, (num_groups, size_k, size_n), grp_b, vmask_in, drhs_ps_in = res
      g_out, g_rhs_ret = grads if return_rhs else (grads, None)
      _, grp_b = jax.lax.optimization_barrier((g_out, grp_b))
      ps_from_ret = g_rhs_ret.qvalue if isinstance(g_rhs_ret, PreGatheredFp8Weight) else g_rhs_ret
      combined_drhs_ps = ps_from_ret if drhs_ps_in is None else (
          drhs_ps_in if ps_from_ret is None else drhs_ps_in + ps_from_ret
      )
      d_rhs_full = _compute_drhs_full(
          lhs_saved, num_groups, size_k, size_n, grp_b, g_out, vmask_in, combined_drhs_ps
      )
      if has_pre_gathered:
        if pre_q_is_struct:
          d_pre_q = PreGatheredFp8Weight(
              qvalue=d_rhs_full, scale=jnp.zeros(pre_q_scale_shape, dtype=pre_q_scale_dtype)
          )
        else:
          d_pre_q = d_rhs_full
        return None, jnp.zeros(rhs_shape, dtype=rhs_dtype), None, d_pre_q, None, None, None, None
      d_rhs = _maybe_scatter_drhs(d_rhs_full)
      return None, d_rhs, None, None, None, None, None, None

    _dot_rhs_fn.defvjp(_dot_rhs_fwd, _dot_rhs_bwd)
    rhs_arg = jax.lax.stop_gradient(rhs_sharded) if has_pre_gathered else rhs_sharded
    return _dot_rhs_fn(
        lhs_const, rhs_arg, scale_const, pre_gathered_rhs_q, group_sizes, valid_mask, drhs_partial_sum,
        lhs_bwd_const,
    )

  # Mode 3 (Default): Both activation and weight cotangents.
  @jax.custom_vjp
  def _dot_fn(lhs_in, rhs_in, scale_in, pre_q_in, grp_in, vmask_in, dlhs_ps_in, drhs_ps_in, lhs_bwd_in):
    del dlhs_ps_in, drhs_ps_in, lhs_bwd_in
    _, grp_b = jax.lax.optimization_barrier((lhs_in, grp_in))
    rhs_q_full, rhs_scale_full = _resolve_rhs(rhs_in, scale_in, pre_q_in)
    out = _run_fwd_kernel(lhs_in, rhs_q_full, rhs_scale_full, grp_b, vmask_in)
    rhs_ret = PreGatheredFp8Weight(qvalue=rhs_q_full, scale=rhs_scale_full)
    if return_lhs and return_rhs:
      return out, lhs_in, rhs_ret
    if return_lhs:
      return out, lhs_in
    if return_rhs:
      return out, rhs_ret
    return out

  def _dot_fwd(lhs_in, rhs_in, scale_in, pre_q_in, grp_in, vmask_in, dlhs_ps_in, drhs_ps_in, lhs_bwd_in):
    _, grp_b = jax.lax.optimization_barrier((lhs_in, grp_in))
    rhs_q_full, rhs_scale_full = _resolve_rhs(rhs_in, scale_in, pre_q_in)
    out = _run_fwd_kernel(lhs_in, rhs_q_full, rhs_scale_full, grp_b, vmask_in)
    rhs_res = (rhs_q_full, rhs_scale_full) if use_reuse_bwd_rhs else (rhs_in, scale_in)
    rhs_ret = PreGatheredFp8Weight(qvalue=rhs_q_full, scale=rhs_scale_full)
    if return_lhs and return_rhs:
      primals_out = (out, lhs_in, rhs_ret)
    elif return_lhs:
      primals_out = (out, lhs_in)
    elif return_rhs:
      primals_out = (out, rhs_ret)
    else:
      primals_out = out
    saved_lhs = lhs_bwd_in if lhs_bwd_in is not None else lhs_in
    return primals_out, (saved_lhs, rhs_res, grp_b, vmask_in, dlhs_ps_in, drhs_ps_in)

  def _dot_bwd(res, grads):
    lhs_saved, rhs_res, grp_b, vmask_in, dlhs_ps_in, drhs_ps_in = res
    if return_lhs and return_rhs:
      g_out, g_lhs_ret, g_rhs_ret = grads
    elif return_lhs:
      g_out, g_lhs_ret = grads
      g_rhs_ret = None
    elif return_rhs:
      g_out, g_rhs_ret = grads
      g_lhs_ret = None
    else:
      g_out, g_lhs_ret, g_rhs_ret = grads, None, None

    _, grp_b = jax.lax.optimization_barrier((g_out, grp_b))
    if use_reuse_bwd_rhs:
      rhs_q_full, rhs_scale_full = rhs_res
    else:
      rhs_q_full, rhs_scale_full = _resolve_rhs(rhs_res[0], rhs_res[1], None)

    num_groups, size_k, size_n = rhs_q_full.shape
    combined_dlhs_ps = g_lhs_ret if dlhs_ps_in is None else (
        dlhs_ps_in if g_lhs_ret is None else dlhs_ps_in + g_lhs_ret
    )
    d_lhs = _compute_dlhs(rhs_q_full, rhs_scale_full, grp_b, g_out, vmask_in, combined_dlhs_ps)

    ps_from_ret = g_rhs_ret.qvalue if isinstance(g_rhs_ret, PreGatheredFp8Weight) else g_rhs_ret
    combined_drhs_ps = ps_from_ret if drhs_ps_in is None else (
        drhs_ps_in if ps_from_ret is None else drhs_ps_in + ps_from_ret
    )
    d_rhs_full = _compute_drhs_full(
        lhs_saved, num_groups, size_k, size_n, grp_b, g_out, vmask_in, combined_drhs_ps
    )

    d_dlhs_ps = jnp.zeros_like(dlhs_ps_in) if dlhs_ps_in is not None else None

    if has_pre_gathered:
      if pre_q_is_struct:
        d_pre_q = PreGatheredFp8Weight(
            qvalue=d_rhs_full, scale=jnp.zeros(pre_q_scale_shape, dtype=pre_q_scale_dtype)
        )
      else:
        d_pre_q = d_rhs_full
      return d_lhs, jnp.zeros(rhs_shape, dtype=rhs_dtype), None, d_pre_q, None, None, d_dlhs_ps, None, None

    d_rhs = _maybe_scatter_drhs(d_rhs_full)
    return d_lhs, d_rhs, None, None, None, None, d_dlhs_ps, None, None

  _dot_fn.defvjp(_dot_fwd, _dot_bwd)
  rhs_arg = jax.lax.stop_gradient(rhs_sharded) if has_pre_gathered else rhs_sharded
  return _dot_fn(
      lhs, rhs_arg, scale_const, pre_gathered_rhs_q, group_sizes, valid_mask, dlhs_partial_sum, drhs_partial_sum,
      lhs_bwd_const,
  )


def fp8_ep_token_all_gather(
    x_local: jax.Array,
    ep_axis: Any,
    *,
    block_size: int = _GMM_LHS_QUANT_BLOCK,
) -> jax.Array:
  """Stretch L3: `512`-block dynamic `f8e4m3fn` EP token `all-gather` (`512 -> 258 MiB` on `EP=8`).

  Args:
    x_local: Local token activations (`bf16[T_local, D]` or `bf16[T_local, D // 128, 128]`).
    ep_axis: Expert-parallel mesh axis name(s), or `None` when `EP == 1`.
    block_size: Sub-channel block size matching `gmm_v2` (`512`).

  Returns:
    Gathered `bfloat16` (or `x_local.dtype`) token activations across `ep_axis`.
  """
  if ep_axis is None:
    return x_local

  ag_fn = functools.partial(jax.lax.all_gather, axis_name=ep_axis, axis=0, tiled=True)
  rs_fn = functools.partial(jax.lax.psum_scatter, axis_name=ep_axis, scatter_dimension=0, tiled=True)

  orig_dtype = x_local.dtype
  tail_shape = x_local.shape[1:]
  dim_k = _flattened_minor_dim(x_local)
  eff_block = math.gcd(dim_k, block_size)
  num_blocks = dim_k // eff_block

  x_sg = jax.lax.stop_gradient(x_local)
  t_local = x_sg.shape[0]
  x_blk = x_sg.astype(jnp.bfloat16).astype(jnp.float32).reshape(t_local, num_blocks, eff_block)
  block_abs_max = jnp.max(jnp.abs(x_blk), axis=-1, keepdims=True)
  block_scale = block_abs_max / jnp.float32(_E4M3_MAX)
  block_scale_inv = jnp.where(block_scale == 0.0, jnp.float32(0.0), jnp.float32(1.0) / block_scale)
  x_q_local = (x_blk * block_scale_inv).astype(jnp.float8_e4m3fn).reshape(t_local, *tail_shape)

  # Tag gathered FP8 tokens (`256 MiB/L`) and lane-packed scales (`2 MiB/L`) outside `custom_vjp` (`C3`, #278)
  # so `te_moe_dispatch_remat_location=device` DCEs all `2_REMAT` SC0 EP `all-gather`s while `=remat` is a no-op.
  ep_x_q = checkpoint_name(ag_fn(x_q_local), "te_moe_dispatch_remat_location")
  ep_scale_raw = ag_fn(block_scale.squeeze(-1))
  t_ep = ep_x_q.shape[0]
  if ep_scale_raw.size > 0 and ep_scale_raw.size % 128 == 0:
    ep_scale = checkpoint_name(ep_scale_raw.reshape(-1, 128), "te_moe_dispatch_remat_location").reshape(
        t_ep, num_blocks
    )
  else:
    ep_scale = checkpoint_name(ep_scale_raw, "te_moe_dispatch_remat_location")
  ep_x_bf16 = (ep_x_q.reshape(t_ep, num_blocks, eff_block).astype(jnp.float32) * ep_scale[:, :, None]).astype(
      jnp.bfloat16
  )
  ep_x_fwd = ep_x_bf16.reshape(t_ep, *tail_shape).astype(orig_dtype)

  @jax.custom_vjp
  def _attach_ste_rs_bwd(fwd_val: jax.Array, x_in: jax.Array) -> jax.Array:
    del x_in
    return fwd_val

  def _attach_fwd(fwd_val: jax.Array, x_in: jax.Array):
    del x_in
    return fwd_val, ()

  def _attach_bwd(_, g_ep: jax.Array):
    return (None, rs_fn(g_ep).astype(orig_dtype))

  _attach_ste_rs_bwd.defvjp(_attach_fwd, _attach_bwd)
  return _attach_ste_rs_bwd(jax.lax.stop_gradient(ep_x_fwd), x_local)


def fp8_ep_combine_psum_scatter(
    combined_ep: jax.Array,
    ep_axis: Any,
    *,
    block_size: int = _GMM_LHS_QUANT_BLOCK,
) -> jax.Array:
  """Stretch L4: `0.0-ULP` fwd combine `psum_scatter` whose VJP all-gathers `512`-block `f8e4m3fn` (`2x` reduction)."""
  if ep_axis is None:
    return combined_ep

  rs_fn = functools.partial(jax.lax.psum_scatter, axis_name=ep_axis, scatter_dimension=0, tiled=True)
  ag_fn = functools.partial(jax.lax.all_gather, axis_name=ep_axis, axis=0, tiled=True)

  orig_dtype = combined_ep.dtype
  tail_shape = combined_ep.shape[1:]
  dim_n = _flattened_minor_dim(combined_ep)
  eff_block = math.gcd(dim_n, block_size)
  num_blocks = dim_n // eff_block

  @jax.custom_vjp
  def _ep_rs(x_ep: jax.Array) -> jax.Array:
    return rs_fn(x_ep)

  def _ep_rs_fwd(x_ep: jax.Array):
    return _ep_rs(x_ep), ()

  def _ep_rs_bwd(_, g_local: jax.Array):
    f8_max = jnp.float32(_E4M3_MAX)
    t_local = g_local.shape[0]
    g_f32 = g_local.astype(jnp.float32).reshape(t_local, num_blocks, eff_block)
    g_amax = jnp.max(jnp.abs(g_f32), axis=-1, keepdims=True)
    g_scale = jnp.where(g_amax == 0.0, jnp.float32(1.0), g_amax / f8_max)
    g_q_local = jnp.clip(g_f32 / g_scale, -f8_max, f8_max).astype(jnp.float8_e4m3fn).reshape(t_local, *tail_shape)

    g_q_ep = ag_fn(g_q_local)
    g_scale_ep = ag_fn(g_scale.squeeze(-1))
    t_ep = g_q_ep.shape[0]
    g_ep = (
        g_q_ep.reshape(t_ep, num_blocks, eff_block).astype(jnp.float32) * g_scale_ep[:, :, None]
    ).reshape(t_ep, *tail_shape).astype(orig_dtype)
    return (g_ep,)

  _ep_rs.defvjp(_ep_rs_fwd, _ep_rs_bwd)
  return _ep_rs(combined_ep)
