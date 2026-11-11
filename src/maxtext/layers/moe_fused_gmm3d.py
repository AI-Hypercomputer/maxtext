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

"""Fused 3D-layout routed-expert MLP for EP=1 (``moe_use_fused_gmm3d``).

Ports the Lineage DSv3 fused GMMv2 expert path (``experimental/lineage/gmmv2``)
to the MaxText ``RoutedMoE`` sparse-matmul layer when there is no expert
parallelism (pure FSDP). The tokens of a layer are the only "chunk", so there
is no activation bank across layers and no chunk loop: the routing permutation,
the fp8 quantization of the routed tokens, the SwiGLU, the routing coefficient
and the top-k combine are all fused into the expert GMM kernels.

Forward (per device, ``N`` local tokens, ``C = N * top_k`` sorted slots)::

  x_q3d [N, D // 128, 128] fp8      per-tensor static quantization of the input
  w_gate_q [E, D, 2F], wo_q [E, F, D] fp8   quantized + FSDP all-gathered weights
  (bank [C, 2F] bf16, act [C, F] fp8) = gather_gate_gmm(x_q3d, token_indices, ...)
      the kernel DMAs the routed rows straight from x_q3d (no sort / gather in
      HBM) and runs silu(g0) * g1 * coeff + fp8 quantization in its epilogue.
  y [N, D // 128, 128] bf16 = linear_combine_fwd(act, wo_q, token_indices, ...)
      the linear GMM rows are summed straight into their tokens (top-k combine).

Backward::

  routed [C, D // 128, 128] fp8 = ragged_gather_tc(x_q3d, token_indices)
  (dgate_q, dcoeffs, dwo) = linear_bwd_dgrad / linear_bwd_wgrad(dy, token_indices,
      wo_q, bank, coeffs)   gathers dy rows in-kernel, e5m2 per-tile scales,
      SiLU backward (with the fixed-calibration straight-through mask)
  (dx_routed, dw_gate) = gate_bwd_fused(routed, w_gate_q, dgate_q)
  dx [N, D] = ragged_gather_reduce_tc(dx_routed, ...)   top-k sum back to tokens
  dw_gate, dwo are reduce-scattered back to the FSDP weight shards.

``FusedGmm3dParams.use_reference`` selects a pure-JAX emulation of the kernels
(``jax.lax.ragged_dot`` + dense XLA ops) with the same quantization points. It
runs on CPU and is what the parity tests compare against the unfused path.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
from typing import Any, Sequence

import jax
from jax.experimental import xla_metadata
import jax.numpy as jnp

LANES = 128
_SUBLANE_FP8 = 32


def _finfo_max(qtype: Any) -> float:
  return float(jnp.finfo(qtype).max)


@dataclasses.dataclass(frozen=True)
class FusedGmm3dParams:
  """Static parameters of ``fused_moe_experts``.

  Attributes:
    num_experts: Number of experts E (all local at EP=1).
    top_k: Experts per token.
    emb_dim: Model dim D (multiple of 128).
    mlp_dim: Expert hidden dim F (multiple of 128).
    act_qtype: fp8 dtype of the tokens / activations, or None for an
      unquantized (bf16 / f32) reference run (``use_reference`` only).
    weight_qtype: fp8 dtype of the expert weights (None when ``act_qtype`` is).
    bwd_qtype: fp8 dtype of the output gradients in the backward pass.
    act_scale: Static per-tensor quantization scale of tokens and activations.
    weight_scale: Static per-tensor quantization scale of the weights.
    compute_dtype: Rounding dtype of the GMM outputs (bf16 in production).
    grad_dtype: Dtype of the weight gradients.
    wi_gather_axes: Mesh axes over which dim 0 of the flat ``[E * D, 2F]`` gate
      weight shard is all-gathered (and its gradient reduce-scattered).
    wo_gather_axes: Same for the flat ``[E * F, D]`` linear weight shard.
    gate_tile_m: ``tile_m`` of the gather + gate GMM.
    gate_tile_k: ``tile_k`` of the gather + gate GMM (multiple of 1024).
    use_reference: Run the pure-JAX emulation instead of the Pallas kernels.
  """

  num_experts: int
  top_k: int
  emb_dim: int
  mlp_dim: int
  act_qtype: Any
  weight_qtype: Any
  bwd_qtype: Any
  act_scale: float
  weight_scale: float
  compute_dtype: Any = jnp.bfloat16
  grad_dtype: Any = jnp.bfloat16
  wi_gather_axes: tuple[str, ...] = ()
  wo_gather_axes: tuple[str, ...] = ()
  gate_tile_m: int = 768
  gate_tile_k: int = 1024
  dgrad_tile_rows: int = 256
  dgrad_bucket_rows: int = 128
  dgrad_tile_n: int = 512
  dgrad_tile_k: int = 1024
  dx_tile_m: int = 512
  dx_tile_k: int = 3072
  dx_tile_n: int = 2048
  dw_tile_k: int = 1024
  use_gmm_linear_combine: bool = True
  split_gate_bwd: bool = True
  split_dw_gate: bool = True
  prequant_dy: bool = False
  early_dwo_rs: bool = False
  sc_routed_gather: bool = False
  exact_topk_combine: bool = False
  exact_topk_out_2d: bool = False
  inv_perm_combine: bool = False
  relayout_x_2d_to_3d: bool = False
  fused_quant_relayout: bool = False
  overlap_dcoeffs_scatter: bool = False
  wgrad_schedule_mode: int = 0
  wi_ag_latency_ns: int = 0
  use_reference: bool = False

  @property
  def quantized(self) -> bool:
    return self.act_qtype is not None

  def validate(self):
    """Validates fused 3D GMM MoE shape, quantization, tile, and schedule parameters."""
    if self.emb_dim % LANES or self.mlp_dim % LANES:
      raise ValueError(f"emb_dim ({self.emb_dim}) and mlp_dim ({self.mlp_dim}) must be multiples of {LANES}.")
    if self.quantized and self.weight_qtype is None:
      raise ValueError("weight_qtype must not be None when act_qtype is set.")
    if self.wgrad_schedule_mode not in range(18):
      raise ValueError(f"wgrad_schedule_mode ({self.wgrad_schedule_mode}) must be in 0..17.")
    if not self.use_reference:
      if not self.quantized or self.weight_qtype is None or self.bwd_qtype is None:
        raise ValueError("The fused gmm3d kernels require fp8 act / weight / bwd quantization.")
      if self.gate_tile_k <= 0 or self.emb_dim % self.gate_tile_k or self.gate_tile_k % 1024:
        raise ValueError(
            f"gate_tile_k ({self.gate_tile_k}) must be a positive multiple of 1024 dividing emb_dim ({self.emb_dim})."
        )
      if (
          self.mlp_dim % 256
          or self.dgrad_tile_n <= 0
          or self.dgrad_tile_n % LANES
          or self.dgrad_tile_k <= 0
          or self.dgrad_tile_k % LANES
      ):
        raise ValueError(
            f"mlp_dim ({self.mlp_dim}) must be a multiple of 256 and dgrad_tile_n ({self.dgrad_tile_n}),"
            f" dgrad_tile_k ({self.dgrad_tile_k}) positive multiples of {LANES}."
        )
      if (
          self.dx_tile_m <= 0
          or self.dx_tile_m % LANES
          or self.dx_tile_n <= 0
          or (self.emb_dim > 1024 and math.gcd(self.dx_tile_n, self.emb_dim) % 1024)
      ):
        raise ValueError(
            f"dx_tile_m ({self.dx_tile_m}) must be a positive multiple of {LANES} and dx_tile_n ({self.dx_tile_n})"
            f" must give a 1024-aligned tile dividing emb_dim ({self.emb_dim})."
        )


def _scale(value: float) -> jax.Array:
  return jnp.full((1, 1), value, jnp.float32)


def static_quantize(x: jax.Array, qtype: Any, scale: float) -> jax.Array:
  """Returns ``clip(x / scale)`` cast to ``qtype`` (saturating)."""
  fmax = _finfo_max(qtype)
  return jnp.clip(x.astype(jnp.float32) / scale, -fmax, fmax).astype(qtype)


def _varying_axes(x: jax.Array) -> tuple[str, ...]:
  """Mesh axes `x` is typed varying over (empty outside shard_map / check_vma)."""
  try:
    return tuple(sorted(jax.typeof(x).manual_axis_type.varying))
  except (AttributeError, TypeError):
    return ()


def _all_gather_rows(x: jax.Array, axes: Sequence[str]) -> jax.Array:
  for axis_name in axes:
    x = jax.lax.all_gather(x, axis_name, axis=0, tiled=True)
  return x


def _reduce_scatter_rows(x: jax.Array, axes: Sequence[str]) -> jax.Array:
  for axis_name in reversed(tuple(axes)):
    x = jax.lax.psum_scatter(x, axis_name, scatter_dimension=0, tiled=True)
  return x


@contextlib.contextmanager
def _ag_latency_scope(latency_ns: int):
  if latency_ns <= 0:
    yield
    return
  with xla_metadata.set_xla_metadata(latency_metadata=str(int(latency_ns))):
    yield


def quantize_and_gather_weight(
    w_local: jax.Array,
    qtype: Any,
    scale: float,
    gather_axes: Sequence[str],
    gathered_shape: tuple[int, int, int],
    *,
    latency_ns: int = 0,
) -> jax.Array:
  """Quantizes a flat ``[rows_local, cols]`` weight shard and all-gathers it (QAG).

  Returns the ``[E, rows, cols]`` view of the gathered weight (a free bitcast of
  the flat matrix), quantized to ``qtype`` when given, else in ``w_local``'s
  dtype.
  """
  if qtype is not None:
    w_local = static_quantize(w_local, qtype, scale)
  with _ag_latency_scope(latency_ns):
    gathered = _all_gather_rows(w_local, gather_axes)
  return gathered.reshape(gathered_shape)


def reduce_scatter_weight_grad(dw_full: jax.Array, gather_axes: Sequence[str], w_local: jax.Array) -> jax.Array:
  """Inverse of ``quantize_and_gather_weight`` for the weight gradient (in `w_local`'s dtype / shape)."""
  dw = dw_full.reshape(-1, dw_full.shape[-1])
  dw = _reduce_scatter_rows(dw, gather_axes)
  if extra := set(_varying_axes(dw)) - set(_varying_axes(w_local)):
    dw = jax.lax.psum(dw, tuple(sorted(extra)))
  return dw.reshape(w_local.shape).astype(w_local.dtype)


def reduce_scatter_split_wi_grad(
    dw_0: jax.Array,
    dw_1: jax.Array,
    gather_axes: Sequence[str],
    wi_local: jax.Array,
) -> jax.Array:
  """Reduce-scatters `[E, D, F]` gate halves separately on dim 0 so each RS overlaps a TC wgrad kernel."""
  dwi_0 = _reduce_scatter_rows(dw_0.reshape(-1, dw_0.shape[-1]), gather_axes)
  dwi_1 = _reduce_scatter_rows(dw_1.reshape(-1, dw_1.shape[-1]), gather_axes)
  dwi = jnp.concatenate([dwi_0, dwi_1], axis=-1)
  if extra := set(_varying_axes(dwi)) - set(_varying_axes(wi_local)):
    dwi = jax.lax.psum(dwi, tuple(sorted(extra)))
  return dwi.reshape(wi_local.shape).astype(wi_local.dtype)


def dense_topk_sorted_tokens(num_tokens: int, top_k: int) -> jax.Array:
  """Tokens of the sorted slots in ascending order when every token has top_k slots."""
  return jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)


# -----------------------------------------------------------------------------
# Pure-JAX reference emulation of the kernels.
# -----------------------------------------------------------------------------


def _ragged_dot(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
  return jax.lax.ragged_dot(
      lhs.astype(jnp.float32),
      rhs.astype(jnp.float32),
      group_sizes.astype(jnp.int32),
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )


def _ragged_dot_drhs(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, num_experts: int) -> jax.Array:
  """``out[e] = lhs[rows of e]^T @ rhs[rows of e]`` for ``[C, K]`` lhs and ``[C, N]`` rhs."""
  del num_experts
  return jax.lax.ragged_dot_general(
      lhs.astype(jnp.float32),
      rhs.astype(jnp.float32),
      group_sizes.astype(jnp.int32),
      ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
          dot_dimension_numbers=(((0,), (0,)), ((), ())),
          lhs_ragged_dimensions=(0,),
          rhs_group_dimensions=(),
      ),
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )


def _dequant(x: jax.Array, scale: float | jax.Array) -> jax.Array:
  return x.astype(jnp.float32) * scale


def _pow2_absmax_quantize(x: jax.Array, qtype: Any) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x` with one power-of-two absmax scale (like the dgrad tiles)."""
  fmax = _finfo_max(qtype)
  amax = jnp.max(jnp.abs(x.astype(jnp.float32)))
  exp = jnp.ceil(jnp.log2(jnp.maximum(amax / fmax, 1e-30)))
  scale = jnp.where(amax > 0, jnp.exp2(exp), 1.0).astype(jnp.float32)
  xq = jnp.clip(x.astype(jnp.float32) / scale, -fmax, fmax).astype(qtype)
  return xq, scale


def _silu(x):
  return x * jax.nn.sigmoid(x)


def _dsilu(x):
  s = jax.nn.sigmoid(x)
  return s * (1.0 + x * (1.0 - s))


def _ref_fwd(p: FusedGmm3dParams, x_q, w_gate_q, wo_q, coeffs, token_indices, group_sizes):
  """Reference of gather_gate_gmm + linear_combine_fwd; returns (y2d, bank, act_q)."""
  num_tokens, size_d = x_q.shape
  size_f = p.mlp_dim
  cdt = p.compute_dtype
  sa = p.act_scale if p.quantized else 1.0
  sw = p.weight_scale if p.quantized else 1.0
  routed = _dequant(x_q[token_indices], sa)  # [C, D]
  gate = _ragged_dot(routed, _dequant(w_gate_q, sw), group_sizes)  # [C, 2F] f32
  bank = gate.astype(cdt)
  g0 = bank[:, :size_f].astype(jnp.float32)
  g1 = bank[:, size_f:].astype(jnp.float32)
  act = (_silu(g0) * g1 * coeffs.astype(jnp.float32)).astype(cdt)
  if p.quantized:
    act_q = static_quantize(act, p.act_qtype, p.act_scale)
  else:
    act_q = act
  out = _ragged_dot(_dequant(act_q, sa), _dequant(wo_q, sw), group_sizes).astype(cdt)  # [C, D]
  y = jnp.zeros((num_tokens, size_d), jnp.float32).at[token_indices].add(out.astype(jnp.float32))
  return y.astype(cdt), bank, act_q


def _ref_bwd(p: FusedGmm3dParams, x_q, w_gate_q, wo_q, coeffs, token_indices, group_sizes, bank, dy2d):
  """Reference of the fused backward; returns (dx2d, dw_gate [E, D, 2F], dwo [E, F, D], dcoeffs [C, 1])."""
  num_tokens, size_d = x_q.shape
  size_f = p.mlp_dim
  cdt = p.compute_dtype
  sa = p.act_scale if p.quantized else 1.0
  sw = p.weight_scale if p.quantized else 1.0
  bwd_quantized = p.bwd_qtype is not None
  coeffs_f32 = coeffs.astype(jnp.float32)
  dy_rows = dy2d.astype(cdt)[token_indices]  # [C, D]
  if bwd_quantized:
    dy_q, s_dy = _pow2_absmax_quantize(dy_rows, p.bwd_qtype)
    dy_dq = _dequant(dy_q, s_dy)
  else:
    dy_dq = dy_rows.astype(jnp.float32)
  # Rematerialize act like the dgrad kernel does from the bank.
  g0 = bank[:, :size_f].astype(jnp.float32)
  g1 = bank[:, size_f:].astype(jnp.float32)
  h = _silu(g0) * g1
  act = (h * coeffs_f32).astype(cdt)
  if p.quantized:
    act_q = static_quantize(act, p.act_qtype, p.act_scale)
    act_max = p.act_scale * _finfo_max(p.act_qtype)
    mask = jnp.abs(act.astype(jnp.float32)) <= act_max
  else:
    act_q = act
    mask = None
  # dact = dy @ wo^T  (bf16-rounded like the kernel's output).
  dact = _ragged_dot(dy_dq, jnp.swapaxes(_dequant(wo_q, sw), 1, 2), group_sizes).astype(cdt).astype(jnp.float32)
  if mask is not None:
    dact = jnp.where(mask, dact, 0.0)
  dh = dact * coeffs_f32
  dcoeffs = jnp.sum(dact * h, axis=-1, keepdims=True)  # [C, 1] f32
  dg0 = dh * g1 * _dsilu(g0)
  dg1 = dh * _silu(g0)
  dgate = jnp.concatenate([dg0, dg1], axis=-1).astype(cdt)  # [C, 2F]
  if bwd_quantized:
    dgate_q, s_dg = _pow2_absmax_quantize(dgate, p.bwd_qtype)
    dgate_dq = _dequant(dgate_q, s_dg)
  else:
    dgate_dq = dgate.astype(jnp.float32)
  # Weight gradients.
  dwo = _ragged_dot_drhs(_dequant(act_q, sa), dy_dq, group_sizes, p.num_experts).astype(p.grad_dtype)  # [E, F, D]
  routed = _dequant(x_q[token_indices], sa)
  dw_gate = _ragged_dot_drhs(routed, dgate_dq, group_sizes, p.num_experts).astype(p.grad_dtype)  # [E, D, 2F]
  # Token gradients: dx_routed = dgate @ w_gate^T, summed back over the top-k slots.
  dx_routed = _ragged_dot(dgate_dq, jnp.swapaxes(_dequant(w_gate_q, sw), 1, 2), group_sizes).astype(cdt)
  dx = jnp.zeros((num_tokens, size_d), jnp.float32).at[token_indices].add(dx_routed.astype(jnp.float32))
  return dx.astype(cdt), dw_gate, dwo, dcoeffs


# -----------------------------------------------------------------------------
# Pallas (TPU) path: Lineage gmmv2 kernels.
# -----------------------------------------------------------------------------


def _inv_perm_src_rows(sort_indices: jax.Array) -> jax.Array:
  """Inverts the routing bijection ``sort_indices`` without an HLO ``sort``."""
  idx = sort_indices.astype(jnp.int32)
  zero_vma = idx * jnp.int32(0)
  positions = zero_vma + jnp.arange(idx.shape[0], dtype=jnp.int32)
  return zero_vma.at[idx].set(positions, mode="promise_in_bounds", unique_indices=True)


def _guard_array_with_tensors(target: jax.Array, tensors: Sequence[jax.Array | None]) -> jax.Array:
  """Zero-FLOP HLO data-dependency barrier forcing ``tensors`` to complete before ``target``."""
  target_vma = set(_varying_axes(target))
  guard = None
  for t in tensors:
    if t is None or not set(_varying_axes(t)).issubset(target_vma):
      continue
    elem = t[(0,) * t.ndim] if t.ndim > 0 else t
    term = jnp.isnan(elem)
    guard = term if guard is None else (guard | term)
  if guard is None:
    return target
  return jnp.where(guard, jnp.asarray(0, dtype=target.dtype), target)


def _lineage():
  """Imports the Lineage kernels lazily (TPU only; keeps CPU-only users light)."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.experimental.lineage import ops as lineage_ops
  from maxtext.experimental.lineage.gmmv2 import expert_mlp
  from maxtext.experimental.lineage.gmmv2 import gate_bwd_fused
  from maxtext.experimental.lineage.gmmv2 import gmmv2_kernel
  from maxtext.experimental.lineage.gmmv2 import linear_bwd_split
  from maxtext.experimental.lineage.gmmv2 import linear_combine_fwd
  from maxtext.experimental.lineage.gmmv2 import relayout
  from maxtext.experimental.lineage.gmmv2 import tgmm_kernel

  return (
      lineage_ops,
      expert_mlp,
      gate_bwd_fused,
      gmmv2_kernel,
      linear_bwd_split,
      linear_combine_fwd,
      relayout,
      tgmm_kernel,
  )


def _kernel_cfg(p: FusedGmm3dParams, expert_mlp, gmmv2_kernel):
  return expert_mlp.ExpertMlpConfig(
      gate_fwd=gmmv2_kernel.TileSizes(
          tile_m=p.gate_tile_m,
          tile_k=p.gate_tile_k,
          tile_n=2 * p.mlp_dim,  # glu_coeffs needs a single n tile.
          bucket_base=256,
      ),
      linear_fwd=gmmv2_kernel.TileSizes(
          tile_m=512,
          tile_k=p.mlp_dim,
          tile_n=math.gcd(2048, p.emb_dim),
          bucket_base=256,
      ),
      disable_multi_core_mode=True,
      zero_initialize=False,
      flat_weights=True,
  )


def _kernel_fwd(
    p: FusedGmm3dParams,
    x_q,
    w_gate_q,
    wo_q,
    coeffs,
    token_indices,
    group_sizes,
    *,
    need_bank: bool = True,
    sort_indices: jax.Array | None = None,
):
  """gather_gate_gmm + (linear_gmm_fp8 + ragged_gather_reduce_tc | linear_combine_fwd); returns (y2d, bank, reduce_meta)."""
  (
      lineage_ops,
      expert_mlp,
      gate_bwd_fused,
      gmmv2_kernel,
      linear_bwd_split,
      linear_combine_fwd,
      relayout,
      tgmm_kernel,
  ) = _lineage()
  del gate_bwd_fused, linear_bwd_split, tgmm_kernel
  num_tokens = x_q.shape[0]
  size_d = p.emb_dim
  size_c = token_indices.shape[0]
  sa, sw = p.act_scale, p.weight_scale
  cfg = _kernel_cfg(p, expert_mlp, gmmv2_kernel)
  if x_q.ndim == 3:
    x_q3d = x_q
  elif p.relayout_x_2d_to_3d:
    x_q3d = relayout.relayout_2d_to_3d_pallas(x_q)
  else:
    x_q3d = x_q.reshape(num_tokens, size_d // LANES, LANES)
  with jax.named_scope("fused_gmm3d_gather_gate"):
    gate_res = expert_mlp.gather_gate_gmm(
        cfg,
        x_q3d,
        token_indices,
        0,
        w_gate_q,
        group_sizes,
        out=None,
        out_offset=0,
        out_scale=_scale(sa * sw),
        size_m=size_c,
        glu_coeffs=coeffs,
        glu_out_dtype=p.act_qtype,
        glu_out_scale=_scale(sa),
        glu_only=not need_bank,
        preferred_element_type=p.compute_dtype,
    )
    if need_bank:
      bank, act = gate_res
    else:
      bank, act = None, gate_res
  with jax.named_scope("fused_gmm3d_linear_combine"):
    if p.use_gmm_linear_combine:
      out3d = expert_mlp.linear_gmm_fp8(
          cfg,
          act,
          wo_q,
          group_sizes,
          act_scale=_scale(sa),
          act_qtype=p.act_qtype,
          out_scale=_scale(sw),
      )
      if (
          p.exact_topk_combine
          and size_c == num_tokens * p.top_k
          and lineage_ops.can_use_exact_topk_gather_reduce_tc(
              num_tokens,
              p.top_k,
              out3d.shape[1],
              out3d.shape[2],
              out3d.dtype,
              out_2d=p.exact_topk_out_2d,
          )
      ):
        if p.inv_perm_combine and sort_indices is not None:
          reduce_meta: Any = _inv_perm_src_rows(sort_indices)
        else:
          reduce_meta = lineage_ops.exact_topk_gather_reduce_tc_src_rows(token_indices)
        y3d = lineage_ops.exact_topk_gather_reduce_tc(
            out3d,
            reduce_meta,
            num_out_tokens=num_tokens,
            top_k=p.top_k,
            out_2d=p.exact_topk_out_2d,
        )
      else:
        reduce_meta = lineage_ops.ragged_gather_reduce_tc_metadata(
            token_indices, 0, size_c, top_k=p.top_k, max_num_tokens=size_c
        )
        y3d = lineage_ops.ragged_gather_reduce_tc(out3d, reduce_meta, num_out_tokens=num_tokens, top_k=p.top_k)
    else:
      reduce_meta = None
      y3d = linear_combine_fwd.linear_combine_fwd(
          act,
          wo_q,
          group_sizes,
          token_indices,
          0,
          dense_topk_sorted_tokens(num_tokens, p.top_k),
          out_scale=_scale(sa * sw),
          num_out_tokens=num_tokens,
      )
  return y3d.reshape(num_tokens, size_d), bank, reduce_meta


def _kernel_bwd(
    p: FusedGmm3dParams,
    x_q,
    w_gate_q,
    wo_q,
    coeffs,
    token_indices,
    group_sizes,
    bank,
    reduce_meta,
    dy2d,
    *,
    wi_local=None,
    wo_local=None,
    sort_indices: jax.Array | None = None,
):
  """linear_bwd_dgrad/wgrad + (split gmm_v2 dx + dual tgmm_v2 dw | gate_bwd_fused) + TC gather-reduce."""
  (
      lineage_ops,
      expert_mlp,
      gate_bwd_fused,
      gmmv2_kernel,
      linear_bwd_split,
      linear_combine_fwd,
      relayout,
      tgmm_kernel,
  ) = _lineage()
  del expert_mlp, linear_combine_fwd
  num_tokens = x_q.shape[0]
  size_d = p.emb_dim
  size_f = p.mlp_dim
  size_c = token_indices.shape[0]
  sa, sw = p.act_scale, p.weight_scale

  def _combine_dx(dx_routed):
    if (
        p.exact_topk_combine
        and size_c == num_tokens * p.top_k
        and lineage_ops.can_use_exact_topk_gather_reduce_tc(
            num_tokens,
            p.top_k,
            dx_routed.shape[1],
            dx_routed.shape[2],
            dx_routed.dtype,
            out_2d=p.exact_topk_out_2d,
        )
    ):
      if isinstance(reduce_meta, jax.Array):
        src_rows = reduce_meta
      elif p.inv_perm_combine and sort_indices is not None:
        src_rows = _inv_perm_src_rows(sort_indices)
      else:
        src_rows = lineage_ops.exact_topk_gather_reduce_tc_src_rows(token_indices)
      return lineage_ops.exact_topk_gather_reduce_tc(
          dx_routed,
          src_rows,
          num_out_tokens=num_tokens,
          top_k=p.top_k,
          out_2d=p.exact_topk_out_2d,
      )
    meta = (
        reduce_meta
        if isinstance(reduce_meta, lineage_ops.RaggedGatherReduceMetadata)
        else lineage_ops.ragged_gather_reduce_tc_metadata(token_indices, 0, size_c, top_k=p.top_k, max_num_tokens=size_c)
    )
    return lineage_ops.ragged_gather_reduce_tc(dx_routed, meta, num_out_tokens=num_tokens, top_k=p.top_k)

  dgrad_tk = size_d if (p.dgrad_tile_k <= 0 or p.dgrad_tile_k >= size_d) else math.gcd(p.dgrad_tile_k, size_d)
  tile_d0 = dgrad_tk // LANES
  use_prequant_dy = (
      p.prequant_dy and p.bwd_qtype is not None and tile_d0 % 4 == 0 and (tile_d0 % 8 == 0 or dgrad_tk == size_d)
  )
  if use_prequant_dy:
    dy_q2d, dy_scale_1d = _pow2_absmax_quantize(dy2d.astype(jnp.bfloat16), p.bwd_qtype)
    dy_scale = dy_scale_1d.reshape(1, 1)
    dy3d = dy_q2d.reshape(num_tokens, size_d // LANES, LANES)
  else:
    dy_scale = None
    if tile_d0 % 2 != 0 or (tile_d0 % 8 != 0 and dgrad_tk != size_d):
      dgrad_tk = math.gcd(1024, size_d)
    dy3d = dy2d.astype(jnp.bfloat16).reshape(num_tokens, size_d // LANES, LANES)
  dgrad_tiles = linear_bwd_split.DgradTiles(
      tile_rows=p.dgrad_tile_rows,
      bucket_rows=p.dgrad_bucket_rows,
      tile_n=math.gcd(p.dgrad_tile_n, size_f),
      tile_k=dgrad_tk,
  )
  if p.split_gate_bwd:
    if p.sc_routed_gather:
      with jax.named_scope("fused_gmm3d_gather_routed"):
        x_q2d = x_q.reshape(num_tokens, size_d) if x_q.ndim == 3 else x_q
        routed = x_q2d.at[token_indices].get(mode="promise_in_bounds")
    else:
      routed = None
    with jax.named_scope("fused_gmm3d_linear_dgrad"):
      dgrad = linear_bwd_split.linear_bwd_dgrad(
          dy3d,
          token_indices,
          0,
          wo_q,
          group_sizes,
          bank,
          0,
          coeffs,
          act_scale=sa,
          w_scale=sw,
          act_qtype=p.act_qtype,
          dy_qtype=p.bwd_qtype,
          dy_scale=dy_scale,
          grad_gate_qtype=None,
          tiles=dgrad_tiles,
      )

    def _compute_dwo(tile_exps_override: jax.Array | None = None):
      with jax.named_scope("fused_gmm3d_linear_wgrad"):
        dwo_raw_ = linear_bwd_split.linear_bwd_wgrad(
            dgrad.act_q,
            dgrad.dy_q,
            dgrad.tile_exps if tile_exps_override is None else tile_exps_override,
            group_sizes,
            act_scale=sa,
            dgrad_tile_rows=dgrad_tiles.tile_rows,
            dy_scale=dy_scale,
            dw_acc=None,
            dw_dtype=p.grad_dtype,
        )
        dwo_ = reduce_scatter_weight_grad(dwo_raw_, p.wo_gather_axes, wo_local) if wo_local is not None else dwo_raw_
        return dwo_raw_, dwo_

    if (
        (p.early_dwo_rs and p.wgrad_schedule_mode < 15)
        or (1 <= p.wgrad_schedule_mode <= 14)
        or p.wgrad_schedule_mode == 17
    ):
      dwo_raw, dwo = _compute_dwo()
    else:
      dwo_raw, dwo = None, None
    fmax = _finfo_max(p.bwd_qtype)
    exp = jnp.ceil(jnp.log2(jnp.maximum(dgrad.grad_gate_amax / fmax, 1e-30)))
    dgate_scale = jnp.where(dgrad.grad_gate_amax > 0, jnp.exp2(exp), 1.0).astype(jnp.float32)
    if (1 <= p.wgrad_schedule_mode <= 14 or p.wgrad_schedule_mode == 17) and dwo_raw is not None:
      dgate_scale = _guard_array_with_tensors(dgate_scale, (dwo_raw,))
    # The [C, 1] coefficient gradient; schedule modes 13/14 hang zero-copy reduce-scatter guards on it.
    grad_coeffs = dgrad.grad_coeffs
    dx_tk = 2 * size_f if (p.dx_tile_k <= 0 or p.dx_tile_k >= 2 * size_f) else math.gcd(p.dx_tile_k, 2 * size_f)
    dx_tn = size_d if size_d <= 1024 else math.gcd(p.dx_tile_n, size_d)
    dx_tiles = gmmv2_kernel.TileSizes(
        tile_m=p.dx_tile_m,
        tile_k=dx_tk,
        tile_n=dx_tn,
        bucket_base=min(256, p.dx_tile_m),
    )

    def _run_gate_dx(lhs_scale_dx: jax.Array) -> jax.Array:
      with jax.named_scope("fused_gmm3d_gate_dx"):
        dx_routed_ = gmmv2_kernel.gmm_v2(
            dgrad.grad_gate,
            w_gate_q,
            group_sizes,
            lhs_scale=lhs_scale_dx,
            lhs_quant_dtype=p.bwd_qtype,
            out_scale=_scale(sw),
            acc_dtype=jnp.float32,
            preferred_element_type=jnp.bfloat16,
            tile_info=dx_tiles,
            transpose_rhs=True,
            out_is_3d=True,
            flat_rhs=True,
            zero_initialize=False,
            disable_multi_core_mode=True,
        )
      with jax.named_scope("fused_gmm3d_combine_dx"):
        return _combine_dx(dx_routed_)

    def _rs_wi_half(dw_half: jax.Array) -> jax.Array | None:
      if wi_local is None:
        return None
      dwi_half = _reduce_scatter_rows(dw_half.reshape(-1, dw_half.shape[-1]), p.wi_gather_axes)
      if extra := set(_varying_axes(dwi_half)) - set(_varying_axes(wi_local)):
        dwi_half = jax.lax.psum(dwi_half, tuple(sorted(extra)))
      return dwi_half

    def _ensure_routed(routed_in: jax.Array | None) -> jax.Array:
      if routed_in is not None:
        return routed_in
      with jax.named_scope("fused_gmm3d_gather_routed"):
        x_q3d_ = (
            x_q
            if x_q.ndim == 3
            else (
                relayout.relayout_2d_to_3d_pallas(x_q)
                if p.relayout_x_2d_to_3d
                else x_q.reshape(num_tokens, size_d // LANES, LANES)
            )
        )
        return lineage_ops.ragged_gather_tc(x_q3d_, token_indices, 0, size_c, max_out_tokens=size_c)

    def _run_gate_wgrad(routed_in: jax.Array, scale_in: jax.Array, chain_halves: bool):
      with jax.named_scope("fused_gmm3d_gate_wgrad"):
        dw_tile_k = math.gcd(p.dw_tile_k, size_d)
        if p.split_dw_gate:
          dw_tile_n = 1536 if size_f % 1536 == 0 else (size_f if size_f <= 2048 else math.gcd(2048, size_f))
          dw_tiles = gmmv2_kernel.TileSizes(
              tile_m=768,
              tile_k=dw_tile_k,
              tile_n=dw_tile_n,
              bucket_base=256,
          )
          dw_0_ = tgmm_kernel.tgmm_v2(
              routed_in,
              dgrad.grad_gate,
              group_sizes,
              p.num_experts,
              tile_info=dw_tiles,
              rhs_quant_scale=scale_in,
              rhs_quant_dtype=p.bwd_qtype,
              out_scale=_scale(sa),
              acc_dtype=jnp.float32,
              preferred_element_type=p.grad_dtype,
              disable_multi_core_mode=True,
              rhs_col_offset=0,
              rhs_num_cols=size_f,
          )
          dwi_0_ = _rs_wi_half(dw_0_)
          scale_1 = _guard_array_with_tensors(scale_in, (dw_0_,)) if chain_halves else scale_in
          dw_1_ = tgmm_kernel.tgmm_v2(
              routed_in,
              dgrad.grad_gate,
              group_sizes,
              p.num_experts,
              tile_info=dw_tiles,
              rhs_quant_scale=scale_1,
              rhs_quant_dtype=p.bwd_qtype,
              out_scale=_scale(sa),
              acc_dtype=jnp.float32,
              preferred_element_type=p.grad_dtype,
              disable_multi_core_mode=True,
              rhs_col_offset=size_f,
              rhs_num_cols=size_f,
          )
          if wi_local is not None and dwi_0_ is not None:
            dwi_1_ = _rs_wi_half(dw_1_)
            assert dwi_1_ is not None
            dwi_ = jnp.concatenate([dwi_0_, dwi_1_], axis=-1)
            dw_gate_: Any = dwi_.reshape(wi_local.shape).astype(wi_local.dtype)
          else:
            dwi_1_ = None
            dw_gate_ = (dw_0_, dw_1_)
          return dw_gate_, dw_0_, dw_1_, dwi_0_, dwi_1_
        dw_full_n = 2 * size_f
        dw_tile_n = 1536 if dw_full_n % 1536 == 0 else (dw_full_n if dw_full_n <= 2048 else math.gcd(2048, dw_full_n))
        dw_tiles = gmmv2_kernel.TileSizes(
            tile_m=768,
            tile_k=dw_tile_k,
            tile_n=dw_tile_n,
            bucket_base=256,
        )
        dw_gate_raw_ = tgmm_kernel.tgmm_v2(
            routed_in,
            dgrad.grad_gate,
            group_sizes,
            p.num_experts,
            tile_info=dw_tiles,
            rhs_quant_scale=scale_in,
            rhs_quant_dtype=p.bwd_qtype,
            out_scale=_scale(sa),
            acc_dtype=jnp.float32,
            preferred_element_type=p.grad_dtype,
            disable_multi_core_mode=True,
        )
        dw_gate_ = (
            reduce_scatter_weight_grad(dw_gate_raw_, p.wi_gather_axes, wi_local) if wi_local is not None else dw_gate_raw_
        )
        return dw_gate_, dw_gate_raw_, None, None, None

    if p.wgrad_schedule_mode == 4:
      routed = _ensure_routed(routed)
      dw_gate, dw_0, dw_1, _, dwi_1 = _run_gate_wgrad(routed, dgate_scale, chain_halves=True)
      dgate_scale_dx = _guard_array_with_tensors(dgate_scale, (dw_1 if dw_1 is not None else dw_0,))
      dx3d = _run_gate_dx(dgate_scale_dx)
      dx3d = _guard_array_with_tensors(
          dx3d,
          (
              dwo,
              dw_gate if isinstance(dw_gate, jax.Array) else dw_0,
              dwi_1 if not isinstance(dw_gate, jax.Array) else None,
          ),
      )
    elif p.wgrad_schedule_mode in (10, 11, 12, 13, 14, 17):
      # Scalar-only guard schedule (ZERO copy on 512 MB dx3d!):
      #   linear_bwd_dgrad -> linear_bwd_wgrad -> tgmm_v2 dw_0 -> tgmm_v2 dw_1 -> gate_dx -> combine_dx -> Attn bwd.
      # Hides sc_routed_gather behind linear_bwd_dgrad + linear_bwd_wgrad (~14.4 ms), dwo RS behind dw_0,
      # dwi_0 RS behind dw_1, and dwi_1 RS behind gate_dx + combine_dx + Attention bwd (~53.2 ms)!
      #   13: mode 12 + the dwi_1 RS must also finish before the layer's dx leaves the block: guards the [C, 1]
      #       grad_coeffs (2 MB, feeds the router bwd and hence the attention bwd) instead of the 512 MB dx3d, so
      #       XLA starts that RS right after dw_1 and it overlaps gate_dx + combine_dx rather than the next layer.
      #   14: mode 10 + all three RSs (dwo, dwi_0, dwi_1) guarded on grad_coeffs only (none forced before
      #       gate_dx), letting XLA spread them across dw_0 + dw_1 + gate_dx + combine_dx (~20 ms of TC work).
      #   17: mode 10 + all three RSs guarded on the combined dx3d instead of grad_coeffs: grad_coeffs feeds the
      #       SparseCore dcoeffs scatter, which the scheduler issues early and thereby pulls the RS dones before
      #       gate_dx; dx3d is consumed only by the attention backward, so the RSs may finish anywhere during
      #       gate_dx + combine_dx (the dx3d select costs one 512 MiB pass).
      routed = _ensure_routed(routed)
      dw_gate, dw_0, dw_1, dwi_0, dwi_1 = _run_gate_wgrad(routed, dgate_scale, chain_halves=True)
      last_dw = dw_1 if dw_1 is not None else dw_0
      post_rs_0 = dwi_0 if dwi_0 is not None else dw_0
      post_rs_1 = dwi_1 if dwi_1 is not None else last_dw
      if p.wgrad_schedule_mode in (12, 13):
        dx_deps: tuple[jax.Array | None, ...] = (dwo, post_rs_0, last_dw)
      elif p.wgrad_schedule_mode == 11:
        dx_deps = (dwo, last_dw)
      else:
        dx_deps = (last_dw,)
      dgate_scale_dx = _guard_array_with_tensors(dgate_scale, dx_deps)
      dx3d = _run_gate_dx(dgate_scale_dx)
      if p.wgrad_schedule_mode == 13:
        grad_coeffs = _guard_array_with_tensors(grad_coeffs, (post_rs_1,))
      elif p.wgrad_schedule_mode == 14:
        grad_coeffs = _guard_array_with_tensors(grad_coeffs, (dwo, post_rs_0, post_rs_1))
      elif p.wgrad_schedule_mode == 17:
        wi_rs = (dw_gate,) if (dwi_0 is None and isinstance(dw_gate, jax.Array)) else (post_rs_0, post_rs_1)
        dx3d = _guard_array_with_tensors(dx3d, (dwo, *wi_rs))
    elif p.wgrad_schedule_mode in (15, 16):
      # Un-delayed gate_dx + combine_dx (Attention bwd runs immediately), while forcing
      # linear_bwd_wgrad (dwo_raw, pre-RS, does NOT need `routed` and frees 2.75 GiB act_q + dy3d)
      # BEFORE chained tgmm_v2 (dw_0 -> dw_1): hides `gather_routed` behind linear_bwd_wgrad,
      # hides `dwo` RS behind `dw_0`, and hides `dwi_0` RS behind `dw_1`!
      dx3d = _run_gate_dx(dgate_scale)
      tile_exps_dwo = _guard_array_with_tensors(dgrad.tile_exps, (dx3d,)) if p.wgrad_schedule_mode == 16 else None
      dwo_raw, dwo = _compute_dwo(tile_exps_override=tile_exps_dwo)
      routed = _ensure_routed(routed)
      scale_wgrad = _guard_array_with_tensors(dgate_scale, (dwo_raw,))
      dw_gate, _, _, _, _ = _run_gate_wgrad(routed, scale_wgrad, chain_halves=True)
    elif p.wgrad_schedule_mode == 5:
      # 1) dwo first (frees act_q + dy_q, starts dwo RS + gather_offload_async),
      # 2) gate_dx + combine_dx (hides gather_offload_async + dwo RS on SparseCore),
      # 3) chained tgmm_v2 after dx3d (hides dwi_0 RS behind dw_1, starts dwi_1 RS before Attention bwd).
      dx3d = _run_gate_dx(dgate_scale)
      routed = _ensure_routed(routed)
      scale_wgrad = _guard_array_with_tensors(dgate_scale, (dx3d,))
      dw_gate, dw_0, dw_1, dwi_0, _ = _run_gate_wgrad(routed, scale_wgrad, chain_halves=True)
      dx3d = _guard_array_with_tensors(
          dx3d,
          (
              dwo,
              dwi_0 if dwi_0 is not None else dw_0,
              dw_1,
          ),
      )
    else:
      # Modes 0..3 and 6..9:
      #   6: dwo_raw before gate_dx; tgmm_v2 after dx3d (no dx3d hold).
      #   7: dwo_raw before gate_dx; dwo RS must complete before tgmm_v2 (overlaps gate_dx + Attention bwd).
      #   8: dwo_raw before gate_dx; dwo RS completes right after gate_dx + combine_dx (frees 1.5 GiB dwo_raw
      #      before Splash bwd so SparseCore gather_offload can overlap Attention bwd); tgmm_v2 after dx3d.
      #   9: mode 8 + chain tgmm_v2 halves so dwi_0 RS overlaps dw_1 compute.
      dx3d = _run_gate_dx(dgate_scale)
      if dwo is None:
        _, dwo = _compute_dwo()
      if p.wgrad_schedule_mode in (8, 9):
        dx3d = _guard_array_with_tensors(dx3d, (dwo,))
      routed = _ensure_routed(routed)
      if p.wgrad_schedule_mode == 7:
        scale_wgrad = _guard_array_with_tensors(dgate_scale, (dwo, dx3d))
      elif p.wgrad_schedule_mode in (6, 8, 9):
        scale_wgrad = _guard_array_with_tensors(dgate_scale, (dx3d,))
      else:
        scale_wgrad = dgate_scale
      dw_gate, dw_0, dw_1, dwi_0, _ = _run_gate_wgrad(routed, scale_wgrad, chain_halves=p.wgrad_schedule_mode in (3, 9))
      if p.wgrad_schedule_mode == 2:
        dx3d = _guard_array_with_tensors(dx3d, (dw_0, dw_1))
      elif p.wgrad_schedule_mode == 3:
        dx3d = _guard_array_with_tensors(
            dx3d,
            (
                dwo,
                dwi_0 if dwi_0 is not None else dw_0,
                dw_1,
            ),
        )
    dx2d = dx3d.reshape(num_tokens, size_d).astype(p.compute_dtype)
    return dx2d, dw_gate, dwo, grad_coeffs

  with jax.named_scope("fused_gmm3d_gather_routed"):
    # The routed fp8 tokens for the gate dW TGMM (TensorCore gather, 1 B/elem).
    x_q3d = (
        x_q
        if x_q.ndim == 3
        else (
            relayout.relayout_2d_to_3d_pallas(x_q)
            if p.relayout_x_2d_to_3d
            else x_q.reshape(num_tokens, size_d // LANES, LANES)
        )
    )
    routed = lineage_ops.ragged_gather_tc(x_q3d, token_indices, 0, size_c, max_out_tokens=size_c)
  with jax.named_scope("fused_gmm3d_linear_bwd"):
    dgrad = linear_bwd_split.linear_bwd_dgrad(
        dy3d,
        token_indices,
        0,
        wo_q,
        group_sizes,
        bank,
        0,
        coeffs,
        act_scale=sa,
        w_scale=sw,
        act_qtype=p.act_qtype,
        dy_qtype=p.bwd_qtype,
        dy_scale=dy_scale,
        grad_gate_qtype=p.bwd_qtype,
        tiles=dgrad_tiles,
    )
    dwo = linear_bwd_split.linear_bwd_wgrad(
        dgrad.act_q,
        dgrad.dy_q,
        dgrad.tile_exps,
        group_sizes,
        act_scale=sa,
        dgrad_tile_rows=dgrad_tiles.tile_rows,
        dy_scale=dy_scale,
        dw_acc=None,
        dw_dtype=p.grad_dtype,
    )
    if wo_local is not None:
      dwo = reduce_scatter_weight_grad(dwo, p.wo_gather_axes, wo_local)
  with jax.named_scope("fused_gmm3d_gate_bwd"):
    if not gate_bwd_fused.is_supported(routed, w_gate_q, group_sizes, dgrad.grad_gate, dw_dtype=p.grad_dtype):
      raise ValueError(
          f"gate_bwd_fused does not support routed={routed.shape}/{routed.dtype}, w_gate={w_gate_q.shape},"
          f" dgate={dgrad.grad_gate.shape}/{dgrad.grad_gate.dtype}."
      )
    dx_routed, dw_gate = gate_bwd_fused.gate_bwd_fused(
        routed,
        w_gate_q,
        group_sizes,
        dgrad.grad_gate,
        dx_scale=_scale(sw),
        dw_scale=_scale(sa),
        dgate_tile_exps=dgrad.grad_gate_tile_exps,
        dgrad_tile_rows=dgrad_tiles.tile_rows,
        dw_acc=None,
        dw_dtype=p.grad_dtype,
    )
    if wi_local is not None:
      dw_gate = reduce_scatter_weight_grad(dw_gate, p.wi_gather_axes, wi_local)
  with jax.named_scope("fused_gmm3d_combine_dx"):
    dx3d = _combine_dx(dx_routed)
  dx2d = dx3d.reshape(num_tokens, size_d).astype(p.compute_dtype)
  return dx2d, dw_gate, dwo, dgrad.grad_coeffs


# -----------------------------------------------------------------------------
# custom_vjp wrapper.
# -----------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def fused_moe_experts(
    p: FusedGmm3dParams,
    x2d: jax.Array,  # [N, D] compute dtype
    wi_local: jax.Array,  # [E * D / fsdp, 2F] flat gate weight shard (bf16)
    wo_local: jax.Array,  # [E * F / fsdp, D] flat linear weight shard (bf16)
    sorted_coeffs: jax.Array,  # [C, 1] routing coefficients (sorted-slot order, or unsorted when overlap_dcoeffs_scatter)
    sort_indices: jax.Array,  # int32 [C] flat (token * top_k + k) index of each sorted slot
    group_sizes: jax.Array,  # int32 [E] slots per expert
) -> jax.Array:  # [N, D] compute dtype
  """Routed experts of all local tokens with the fused 3D-layout GMM kernels.

  Equivalent to: gather the rows ``x2d[sort_indices // top_k]``, run
  ``silu(x @ wi[:, :F]) * (x @ wi[:, F:]) * coeff`` per expert, ``@ wo`` per expert
  and sum each token's top_k rows, with fp8 per-tensor quantization of the
  tokens, weights and activations and e5m2 output gradients in the backward.
  """
  return _primal_fwd(p, x2d, wi_local, wo_local, sorted_coeffs, sort_indices, group_sizes)


def _prepare(
    p: FusedGmm3dParams,
    x2d,
    wi_local,
    wo_local,
    sorted_coeffs,
    sort_indices,
    *,
    allow_3d_xq: bool = True,
):
  """Quantizes / relays out the fused-MoE inputs; returns `(x_q, w_gate_q, wo_q, coeffs, token_indices)`."""
  p.validate()
  size_e, size_d, size_f = p.num_experts, p.emb_dim, p.mlp_dim
  if x2d.shape[1] != size_d or wi_local.shape[-1] != 2 * size_f or wo_local.shape[-1] != size_d:
    raise ValueError(f"Bad shapes: x {x2d.shape}, wi {wi_local.shape}, wo {wo_local.shape} for {p}.")
  size_c = sort_indices.shape[0]
  if size_c != x2d.shape[0] * p.top_k:
    raise ValueError(f"Expected {x2d.shape[0] * p.top_k} sorted slots, got {size_c}.")
  if not p.use_reference and size_c % _SUBLANE_FP8:
    raise ValueError(f"The fused kernels need the slot count ({size_c}) to be a multiple of {_SUBLANE_FP8}.")
  if p.quantized:
    if p.fused_quant_relayout and allow_3d_xq and not p.use_reference:
      relayout = _lineage()[6]
      x_q = relayout.quantize_relayout_2d_to_3d_pallas(x2d, p.act_qtype, p.act_scale)
    else:
      x_q = static_quantize(x2d, p.act_qtype, p.act_scale)
  else:
    x_q = x2d.astype(p.compute_dtype)
  w_gate_q = quantize_and_gather_weight(
      wi_local,
      p.weight_qtype,
      p.weight_scale,
      p.wi_gather_axes,
      (size_e, size_d, 2 * size_f),
      latency_ns=p.wi_ag_latency_ns,
  )
  wo_q = quantize_and_gather_weight(
      wo_local,
      p.weight_qtype,
      p.weight_scale,
      p.wo_gather_axes,
      (size_e, size_f, size_d),
      latency_ns=p.wi_ag_latency_ns // 2,
  )
  token_indices = (sort_indices // p.top_k).astype(jnp.int32)
  coeffs_in = sorted_coeffs.reshape(size_c, 1).astype(jnp.float32)
  coeffs = (
      coeffs_in.at[sort_indices].get(mode="promise_in_bounds", unique_indices=True)
      if p.overlap_dcoeffs_scatter
      else coeffs_in
  )
  return x_q, w_gate_q, wo_q, coeffs, token_indices


def _primal_fwd(p: FusedGmm3dParams, x2d, wi_local, wo_local, sorted_coeffs, sort_indices, group_sizes):
  """Forward of the fused MoE experts without residuals (the primal of the custom VJP)."""
  group_sizes = group_sizes.astype(jnp.int32)
  x_q, w_gate_q, wo_q, coeffs, token_indices = _prepare(
      p, x2d, wi_local, wo_local, sorted_coeffs, sort_indices, allow_3d_xq=True
  )
  if p.use_reference:
    y2d, _, _ = _ref_fwd(p, x_q, w_gate_q, wo_q, coeffs, token_indices, group_sizes)
  else:
    y2d, _, _ = _kernel_fwd(
        p, x_q, w_gate_q, wo_q, coeffs, token_indices, group_sizes, need_bank=False, sort_indices=sort_indices
    )
  return y2d.astype(x2d.dtype)


def _fused_fwd(p: FusedGmm3dParams, x2d, wi_local, wo_local, sorted_coeffs, sort_indices, group_sizes):
  """Custom-VJP forward of the fused MoE experts; returns the output and the backward residuals."""
  group_sizes = group_sizes.astype(jnp.int32)
  # Keep x_q 2D when SparseCore gathers routed rows (`sc_routed_gather`) so gather_offload + tgmm_v2 stay 2D.
  allow_3d_xq = not (p.split_gate_bwd and p.sc_routed_gather)
  x_q, w_gate_q, wo_q, coeffs, token_indices = _prepare(
      p, x2d, wi_local, wo_local, sorted_coeffs, sort_indices, allow_3d_xq=allow_3d_xq
  )
  if p.use_reference:
    y2d, bank, _ = _ref_fwd(p, x_q, w_gate_q, wo_q, coeffs, token_indices, group_sizes)
    reduce_meta = None
  else:
    y2d, bank, reduce_meta = _kernel_fwd(
        p, x_q, w_gate_q, wo_q, coeffs, token_indices, group_sizes, need_bank=True, sort_indices=sort_indices
    )
  # The local weight shards are only kept for the shape / dtype of their gradients (they are inputs, so
  # saving them costs nothing); the gathered fp8 weights are recomputed under remat like the gmm_v2 path.
  residuals = (
      x_q,
      w_gate_q,
      wo_q,
      coeffs,
      token_indices,
      group_sizes,
      bank,
      reduce_meta,
      wi_local,
      wo_local,
      sorted_coeffs,
      sort_indices,
  )
  return y2d.astype(x2d.dtype), residuals


def _fused_bwd(p: FusedGmm3dParams, residuals, dy2d):
  """Custom-VJP backward of the fused MoE experts."""
  (
      x_q,
      w_gate_q,
      wo_q,
      coeffs,
      token_indices,
      group_sizes,
      bank,
      reduce_meta,
      wi_local,
      wo_local,
      sorted_coeffs,
      sort_indices,
  ) = residuals
  if p.use_reference:
    dx2d, dw_gate, dwo, dcoeffs = _ref_bwd(p, x_q, w_gate_q, wo_q, coeffs, token_indices, group_sizes, bank, dy2d)
    if p.split_gate_bwd and p.split_dw_gate:
      dwi_local = reduce_scatter_split_wi_grad(
          dw_gate[..., : p.mlp_dim],
          dw_gate[..., p.mlp_dim :],
          p.wi_gather_axes,
          wi_local,
      )
    else:
      dwi_local = reduce_scatter_weight_grad(dw_gate, p.wi_gather_axes, wi_local)
    dwo_local = reduce_scatter_weight_grad(dwo, p.wo_gather_axes, wo_local)
  else:
    dx2d, dwi_local, dwo_local, dcoeffs = _kernel_bwd(
        p,
        x_q,
        w_gate_q,
        wo_q,
        coeffs,
        token_indices,
        group_sizes,
        bank,
        reduce_meta,
        dy2d,
        wi_local=wi_local,
        wo_local=wo_local,
        sort_indices=sort_indices,
    )
  dx_out = dx2d.astype(dy2d.dtype)
  if p.overlap_dcoeffs_scatter:
    dcoeffs_1d = dcoeffs.reshape(-1)
    dweights_1d = (
        jnp.zeros_like(dcoeffs_1d).at[sort_indices].set(dcoeffs_1d, mode="promise_in_bounds", unique_indices=True)
    )
    if not p.use_reference and p.wgrad_schedule_mode >= 10:
      dweights_1d = _guard_array_with_tensors(dweights_1d, (dx_out,))
    dcoeffs_out = dweights_1d.reshape(sorted_coeffs.shape).astype(sorted_coeffs.dtype)
  else:
    dcoeffs_out = dcoeffs.reshape(sorted_coeffs.shape).astype(sorted_coeffs.dtype)
  return (
      dx_out,
      dwi_local,
      dwo_local,
      dcoeffs_out,
      None,
      None,
  )


fused_moe_experts.defvjp(_fused_fwd, _fused_bwd, optimize_remat=True)
