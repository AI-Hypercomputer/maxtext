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

"""Pure-JAX no-drop forked gmmv2 3D FP8 routed MoE & SC1 FP8 STE weight gather."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from maxtext.experimental.qwen_35_64c import ops
from maxtext.experimental.qwen_35_64c import ragged_ops
from maxtext.experimental.qwen_35_64c.config import Qwen35Config
from maxtext.experimental.qwen_35_64c.ext_kernels.ragged import ragged_gather_tc as tc
from maxtext.experimental.qwen_35_64c.gmmv2 import gate_bwd_fused
from maxtext.experimental.qwen_35_64c.gmmv2 import gmmv2_kernel
from maxtext.experimental.qwen_35_64c.gmmv2 import linear_bwd_split
from maxtext.experimental.qwen_35_64c.gmmv2 import linear_combine_fwd

_FP8_E4M3FN_NEG_ZERO_BITS = 0x80

_FWD_RAGGED_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [1]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)
_DLHS_RAGGED_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [2]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)
_DRHS_RAGGED_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Fp8SteWeight:
  """Pure-JAX FP8 weight carrier for in-kernel VMEM FP8 dequantization.

  - `w_ste`: zero-broadcast `[E_local, K, N]` (or 4D tiled `[E_local, K//8, 8,
  N]`)
    bf16 straight-through carrier.
  - `codes`: `[E_local, K, N]` (or 4D tiled `[E_local, K//8, 8, N]`)
    float8_e4m3fn quantized weights.
  - `scale`: `[E_local, 1, N]` or `[1, 1]` float32 weight scale.
  """

  w_ste: Any
  codes: jax.Array
  scale: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    if self.codes.ndim == 4:
      return (
          self.codes.shape[0],
          self.codes.shape[1] * self.codes.shape[2],
          self.codes.shape[3],
      )
    return self.codes.shape

  @property
  def dtype(self) -> jnp.dtype:
    return jnp.bfloat16 if self.w_ste is None else self.w_ste.dtype

  @property
  def ndim(self) -> int:
    return 3 if self.codes.ndim == 4 else self.codes.ndim


def compute_hoisted_moe_absmax(w_stacked_local: jax.Array, *, per_expert_scalar: bool = False) -> jax.Array:
  """Computes local MoE weight absmax across stacked superblocks before `scan`.

  `w_stacked_local` has shape `[num_superblocks, 1, in_dim, out_dim]` on each
  TensorCore (`P(None, ('expert', 'fsdp'), None, None)`), so reducing over
  `in_dim` (axis 2) or `(in_dim, out_dim)` (axes 2, 3) is 100% local with zero
  cross-device collectives!
  When `per_expert_scalar=True` (used for native FP8 MXU `gmm_v2` in both fwd
  and `transpose_rhs=True` `dlhs`), returns shape `[num_superblocks, 1, 1, 1]`.
  """
  red_axis = (2, 3) if per_expert_scalar else 2
  absmax = jnp.max(jnp.abs(w_stacked_local), axis=red_axis, keepdims=True).astype(jnp.float32)
  return jax.lax.stop_gradient(absmax)


def quantize_moe_codes_from_absmax(w_local: jax.Array, absmax_local: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Quantizes `[1, in_dim, out_dim]` weight shard to `(codes_fp8, scale_f32)`."""
  qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
  scale = jnp.where(absmax_local > 0.0, absmax_local / qmax, jnp.ones_like(absmax_local))
  scale_div = scale[None, :, :] if scale.ndim == 2 else scale
  q = jnp.clip(w_local.astype(jnp.float32) / scale_div, -qmax, qmax).astype(jnp.float8_e4m3fn)
  bits = jax.lax.bitcast_convert_type(q, jnp.uint8)
  bits = jnp.where(bits == _FP8_E4M3FN_NEG_ZERO_BITS, jnp.zeros_like(bits), bits)
  codes = jax.lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
  codes = jax.lax.optimization_barrier(codes)
  return codes, scale


def gather_routed_weight_sc1(
    w_local: jax.Array,
    absmax_local: jax.Array | None,
    cfg: Qwen35Config,
    *,
    start_after: jax.Array | None = None,
    is_wi: bool = False,
) -> jax.Array | Fp8SteWeight:
  """Gathers routed expert weight over `'fsdp'` on SC1; reduces grad on balanced SC!

  - Forward:
    - When `cfg.moe_fp8_weights`: quantizes `w_local` `[1, K, N]` to
    `float8_e4m3fn`
      `codes_local` (`1 B/el`), reshapes via `routed_w_to_tiles` (`[1, K//8, 8,
      N]`)
      so SparseCore all-gathers directly in `T(8, 128)` without relayout copies,
      all-gathers `codes` (`1/2` bytes!) on **SC1 (`core_ids=[1]`)**,
      and returns `Fp8SteWeight(w_ste, codes_full, scale_full)`.
    - When `cfg.moe_use_forked_3d_gmmv2` is enabled, keeps `codes_f` + `w_ste`
    in
      4D tiled `[64, K//8, 8, N]` across carries/residuals (`flat_rhs=True`
      bitcast
      at kernel entry -> ZERO HBM relayout copies!) and skips `scale` SC1
      all-gather
      when `amax` is already a hoisted `[1, 1]` per-EP scalar!
  - Backward:
    - Receives `dw_full` (`[64, K, N]` or 4D `[64, K//8, 8, N]` bf16 accumulated
      across `mb1 + mb0`!) and runs `sc_psum_scatter` over `'fsdp'` on
      `rs_core`.
  """
  ag_core = ops._fsdp_ag_core_id(cfg, is_bwd_remat=False)  # pylint: disable=protected-access
  rs_core = ops._moe_rs_core_id(cfg, is_wi=is_wi)  # pylint: disable=protected-access
  min_sc = cfg.sc_min_collective_bytes
  use_tiles = cfg.routed_w_tiled_layout
  keep_tiled_carrier = cfg.moe_use_forked_3d_gmmv2 and use_tiles and (w_local.shape[1] % 8 == 0)
  use_fp8 = cfg.moe_fp8_weights and absmax_local is not None
  e_full = w_local.shape[0] * cfg.fsdp_size
  full_shape = (e_full, w_local.shape[1], w_local.shape[2])

  if use_fp8:
    assert absmax_local is not None
    amax_in = ops.order_after(absmax_local, start_after) if start_after is not None else absmax_local

    @jax.custom_vjp
    def _gather_fp8(w: jax.Array, amax: jax.Array) -> Fp8SteWeight:
      codes_l, scale_l = quantize_moe_codes_from_absmax(w, amax)
      if use_tiles:
        codes_l = ops.routed_w_to_tiles(codes_l)
      codes_f = ops.sc_all_gather(codes_l, "fsdp", core_id=ag_core, axis=0, min_sc_bytes=min_sc)
      if use_tiles and not keep_tiled_carrier:
        codes_f = ops.routed_w_from_tiles(codes_f)
      if scale_l.ndim == 2:
        scale_f = scale_l
      else:
        scale_f = ops.sc_all_gather(scale_l, "fsdp", core_id=ag_core, axis=0, min_sc_bytes=min_sc)
      ste_shape = codes_f.shape if keep_tiled_carrier else full_shape
      w_ste = jax.lax.broadcast(jnp.zeros((), dtype=w.dtype), ste_shape)
      return Fp8SteWeight(w_ste=w_ste, codes=codes_f, scale=scale_f)

    def _gather_fp8_fwd(w: jax.Array, amax: jax.Array):
      return _gather_fp8(w, amax), None

    def _gather_fp8_bwd(_, g_carrier: Fp8SteWeight):
      dw_full = g_carrier.w_ste
      if dw_full.ndim == 4:
        dw_local_tiled = ops.sc_psum_scatter(
            dw_full,
            "fsdp",
            core_id=rs_core,
            scatter_dimension=0,
            min_sc_bytes=min_sc,
        )
        dw_local = ops.routed_w_from_tiles(dw_local_tiled)
      elif use_tiles:
        dw_tiled = ops.routed_w_to_tiles(dw_full)
        dw_local_tiled = ops.sc_psum_scatter(
            dw_tiled,
            "fsdp",
            core_id=rs_core,
            scatter_dimension=0,
            min_sc_bytes=min_sc,
        )
        dw_local = ops.routed_w_from_tiles(dw_local_tiled)
      else:
        dw_local = ops.sc_psum_scatter(
            dw_full,
            "fsdp",
            core_id=rs_core,
            scatter_dimension=0,
            min_sc_bytes=min_sc,
        )
      return dw_local, None

    _gather_fp8.defvjp(_gather_fp8_fwd, _gather_fp8_bwd)
    return _gather_fp8(w_local, amax_in)

  w_sharded_in = ops.order_after(w_local, start_after) if start_after is not None else w_local

  @jax.custom_vjp
  def _gather_bf16(w: jax.Array) -> jax.Array:
    w_in = ops.routed_w_to_tiles(w) if use_tiles else w
    w_f = ops.sc_all_gather(w_in, "fsdp", core_id=ag_core, axis=0, min_sc_bytes=min_sc)
    return ops.routed_w_from_tiles(w_f) if use_tiles else w_f

  def _gather_bf16_fwd(w: jax.Array):
    return _gather_bf16(w), None

  def _gather_bf16_bwd(_, dw_full: jax.Array):
    if use_tiles:
      dw_tiled = ops.routed_w_to_tiles(dw_full)
      dw_local = ops.routed_w_from_tiles(
          ops.sc_psum_scatter(
              dw_tiled,
              "fsdp",
              core_id=rs_core,
              scatter_dimension=0,
              min_sc_bytes=min_sc,
          )
      )
    else:
      dw_local = ops.sc_psum_scatter(
          dw_full,
          "fsdp",
          core_id=rs_core,
          scatter_dimension=0,
          min_sc_bytes=min_sc,
      )
    return (dw_local,)

  _gather_bf16.defvjp(_gather_bf16_fwd, _gather_bf16_bwd)
  return _gather_bf16(w_sharded_in)


def bwd_rw3_prefetch_shapes(
    cfg: Qwen35Config, amax_ndim: int = 2
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  """Returns `(wi_codes_shape, wi_scale_shape, wo_codes_shape, wo_scale_shape)` for `attn_3`."""
  e_full = cfg.local_num_experts
  wi_k, wi_n = cfg.emb_dim, 2 * cfg.moe_mlp_dim
  wo_k, wo_n = cfg.moe_mlp_dim, cfg.emb_dim
  if cfg.routed_w_tiled_layout and (wi_k % 8 == 0) and (wo_k % 8 == 0):
    wi_cs = (e_full, wi_k // 8, 8, wi_n)
    wo_cs = (e_full, wo_k // 8, 8, wo_n)
  else:
    wi_cs = (e_full, wi_k, wi_n)
    wo_cs = (e_full, wo_k, wo_n)
  scale_s = (1, 1) if amax_ndim == 2 else (e_full, 1, 1)
  return wi_cs, scale_s, wo_cs, scale_s


def gather_routed_weight_codes_bwd_only(
    w_local: jax.Array,
    absmax_local: jax.Array,
    cfg: Qwen35Config,
    *,
    start_after: Any = None,
) -> tuple[jax.Array, jax.Array]:
  """Gathers `float8_e4m3fn` `(codes_f, scale_f)` on SC1 in backward."""
  ag_core = ops._fsdp_ag_core_id(cfg, is_bwd_remat=False)  # pylint: disable=protected-access
  min_sc = cfg.sc_min_collective_bytes
  use_tiles = cfg.routed_w_tiled_layout
  keep_tiled = cfg.moe_use_forked_3d_gmmv2 and use_tiles and (w_local.shape[1] % 8 == 0)
  amax_in = ops.order_after(absmax_local, start_after) if start_after is not None else absmax_local
  codes_l, scale_l = quantize_moe_codes_from_absmax(w_local, amax_in)
  if use_tiles:
    codes_l = ops.routed_w_to_tiles(codes_l)
  codes_f = ops.sc_all_gather(codes_l, "fsdp", core_id=ag_core, axis=0, min_sc_bytes=min_sc)
  if use_tiles and not keep_tiled:
    codes_f = ops.routed_w_from_tiles(codes_f)
  if scale_l.ndim == 2:
    scale_f = scale_l
  else:
    scale_f = ops.sc_all_gather(scale_l, "fsdp", core_id=ag_core, axis=0, min_sc_bytes=min_sc)
  return codes_f, scale_f


def prequantize_routed_weight_local(
    w_local: jax.Array,
    absmax_local: jax.Array,
    cfg: Qwen35Config,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes local FSDP-sharded `w_local` to `float8_e4m3fn` `(codes_l, scale_l)`."""
  codes_l, scale_l = quantize_moe_codes_from_absmax(w_local, absmax_local)
  if cfg.routed_w_tiled_layout:
    codes_l = ops.routed_w_to_tiles(codes_l)
  return codes_l, scale_l


def gather_prequantized_routed_weight_bwd(
    codes_l: jax.Array,
    scale_l: jax.Array,
    cfg: Qwen35Config,
    *,
    raw_k_dim: int,
    start_after: Any = None,
) -> tuple[jax.Array, jax.Array]:
  """Gathers pre-quantized local `(codes_l, scale_l)` on SC1 in backward."""
  ag_core = ops._fsdp_ag_core_id(cfg, is_bwd_remat=False)  # pylint: disable=protected-access
  min_sc = cfg.sc_min_collective_bytes
  use_tiles = cfg.routed_w_tiled_layout
  keep_tiled = cfg.moe_use_forked_3d_gmmv2 and use_tiles and (raw_k_dim % 8 == 0)
  c_in = ops.order_after(codes_l, start_after) if start_after is not None else codes_l
  codes_f = ops.sc_all_gather(c_in, "fsdp", core_id=ag_core, axis=0, min_sc_bytes=min_sc)
  if use_tiles and not keep_tiled:
    codes_f = ops.routed_w_from_tiles(codes_f)
  if scale_l.ndim == 2:
    scale_f = scale_l
  else:
    s_in = ops.order_after(scale_l, start_after) if start_after is not None else scale_l
    scale_f = ops.sc_all_gather(s_in, "fsdp", core_id=ag_core, axis=0, min_sc_bytes=min_sc)
  return codes_f, scale_f


def produce_prev_bwd_rw3(
    tok_ep,
    carry_pref_in: ops.BwdRoutedWeightPrefetch,
    prev_rw3_local: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    cfg: Qwen35Config,
):
  """Prefetches `attn_3[sb_idx - 1]` on SC1 in backward without closing over any tracers."""

  @jax.custom_vjp
  def _produce(
      tok_in,
      pref_in: ops.BwdRoutedWeightPrefetch,
      wi_c_l: jax.Array,
      wi_s_l: jax.Array,
      wo_c_l: jax.Array,
      wo_s_l: jax.Array,
  ):
    del pref_in, wi_c_l, wi_s_l, wo_c_l, wo_s_l
    return tok_in

  def _produce_fwd(
      tok_in,
      pref_in: ops.BwdRoutedWeightPrefetch,
      wi_c_l: jax.Array,
      wi_s_l: jax.Array,
      wo_c_l: jax.Array,
      wo_s_l: jax.Array,
  ):
    del pref_in
    return tok_in, (wi_c_l, wi_s_l, wo_c_l, wo_s_l)

  def _produce_bwd(res, g_tok_ep):
    wi_c_l, wi_s_l, wo_c_l, wo_s_l = res
    anchor = g_tok_ep.ste_carrier if isinstance(g_tok_ep, ops.Fp8DispatchedTokens) else g_tok_ep
    wi_c, wi_s = gather_prequantized_routed_weight_bwd(wi_c_l, wi_s_l, cfg, raw_k_dim=cfg.emb_dim, start_after=anchor)
    wo_c, wo_s = gather_prequantized_routed_weight_bwd(wo_c_l, wo_s_l, cfg, raw_k_dim=cfg.moe_mlp_dim, start_after=anchor)
    pref = ops.BwdRoutedWeightPrefetch(wi_codes=wi_c, wi_scale=wi_s, wo_codes=wo_c, wo_scale=wo_s)
    return g_tok_ep, pref, None, None, None, None

  _produce.defvjp(_produce_fwd, _produce_bwd)
  wi_c_l, wi_s_l, wo_c_l, wo_s_l = prev_rw3_local
  return _produce(tok_ep, carry_pref_in, wi_c_l, wi_s_l, wo_c_l, wo_s_l)


def attach_last_bwd_rw3(
    x_out: jax.Array,
    final_pref_slot: ops.BwdRoutedWeightPrefetch,
    last_rw3_local: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    cfg: Qwen35Config,
) -> jax.Array:
  """Prefetches final superblock's `attn_3` weights on SC1 during `lm_head` backward."""

  @jax.custom_vjp
  def _attach(
      x: jax.Array,
      pref_slot: ops.BwdRoutedWeightPrefetch,
      wi_c_l: jax.Array,
      wi_s_l: jax.Array,
      wo_c_l: jax.Array,
      wo_s_l: jax.Array,
  ) -> jax.Array:
    del pref_slot, wi_c_l, wi_s_l, wo_c_l, wo_s_l
    return x

  def _attach_fwd(
      x: jax.Array,
      pref_slot: ops.BwdRoutedWeightPrefetch,
      wi_c_l: jax.Array,
      wi_s_l: jax.Array,
      wo_c_l: jax.Array,
      wo_s_l: jax.Array,
  ):
    del pref_slot
    return x, (wi_c_l, wi_s_l, wo_c_l, wo_s_l)

  def _attach_bwd(res, g_x: jax.Array):
    wi_c_l, wi_s_l, wo_c_l, wo_s_l = res
    wi_c, wi_s = gather_prequantized_routed_weight_bwd(wi_c_l, wi_s_l, cfg, raw_k_dim=cfg.emb_dim, start_after=g_x)
    wo_c, wo_s = gather_prequantized_routed_weight_bwd(wo_c_l, wo_s_l, cfg, raw_k_dim=cfg.moe_mlp_dim, start_after=g_x)
    pref = ops.BwdRoutedWeightPrefetch(wi_codes=wi_c, wi_scale=wi_s, wo_codes=wo_c, wo_scale=wo_s)
    return g_x, pref, None, None, None, None

  _attach.defvjp(_attach_fwd, _attach_bwd)
  wi_c_l, wi_s_l, wo_c_l, wo_s_l = last_rw3_local
  return _attach(x_out, final_pref_slot, wi_c_l, wi_s_l, wo_c_l, wo_s_l)


# ---------------------------------------------------------------------------
# No-drop capacity-factor chunking of the sorted local-expert slots
# ---------------------------------------------------------------------------


def chunk_num_tokens(count: jax.Array, start: jax.Array | int, capacity: int) -> jax.Array:
  """Returns the number of local-expert slots in the chunk starting at sorted slot `start`."""
  remaining = count - start
  return jnp.where(remaining >= capacity, capacity, remaining).astype(jnp.int32)


def chunk_group_sizes(local_gs: jax.Array, start: jax.Array | int, num_tokens: jax.Array) -> jax.Array:
  """Returns the per-local-expert sizes of sorted slots `[start, start + num_tokens)`."""
  ends = jnp.cumsum(local_gs)
  begins = ends - local_gs
  stop = start + num_tokens
  return (jnp.clip(ends, start, stop) - jnp.clip(begins, start, stop)).astype(local_gs.dtype)


def chunk_reduce_metadata(
    token_ids_window: jax.Array,
    num_tokens: jax.Array,
    cfg: Qwen35Config,
    num_out_tokens: int,
) -> tc.RaggedGatherReduceMetadata:
  """Gather-reduce routing of one chunk from its `[capacity]` window of sorted token ids."""
  return tc.ragged_gather_reduce_tc_metadata(
      token_ids_window,
      0,
      num_tokens,
      top_k=cfg.num_experts_per_tok,
      block_size=cfg.tc_reduce_block_size,
      num_out_tokens=num_out_tokens,
      packed_sort=True,
  )


def _rowwise_pow2_fp8_roundtrip(x: jax.Array, qtype: jax.typing.DTypeLike) -> jax.Array:
  """Pure-JAX per-row power-of-two FP8 quantize -> dequantize of `[M, ...]` rows (f32 `[M, K]`).

  Stand-in for the Pallas kernels' per-tile power-of-two scales; being per-row,
  the result does not depend on how the sorted slots are split into chunks.
  """
  x2 = x.reshape(x.shape[0], -1).astype(jnp.float32)
  fmax = float(jnp.finfo(jnp.dtype(qtype)).max)
  amax = jnp.max(jnp.abs(x2), axis=1, keepdims=True)
  scale = jnp.where(amax > 0.0, jnp.exp2(jnp.ceil(jnp.log2(amax / fmax))), 1.0)
  codes = jnp.clip(x2 / scale, -fmax, fmax).astype(qtype)
  return codes.astype(jnp.float32) * scale


def _scalar_w_scale(scale: jax.Array) -> jax.Array:
  return scale.reshape(1, 1) if scale.size == 1 else jnp.max(scale).reshape(1, 1)


def routed_moe_forked_3d_fp8(
    disp: ops.Fp8DispatchedTokens,
    w_rows0: jax.Array,
    topk_weights_flat: jax.Array,
    wi: Fp8SteWeight,
    wo: Fp8SteWeight,
    local_gs_full: jax.Array,
    sort_indices_full: jax.Array,
    count_total: jax.Array,
    reduce_meta0: Any,
    cfg: Qwen35Config,
    *,
    capacity: int,
    return_rhs: bool = False,
    forward_tie: Any = None,
    wrap_partial_fp8_bwd: bool = False,
    sublayer_idx: int = 0,
    use_external_bwd_rw: bool = False,
):
  """No-drop zero-relayout forked `gmmv2` 3D FP8 MoE, chunked by `capacity` sorted slots.

  Routing covers ALL `num_slots = T_ep * top_k` slots, sorted with this rank's
  `count_total = sum(local_gs_full)` local-expert slots first:
  `sort_indices_full[s]` is the unsorted slot of sorted slot `s` (EP token
  `sort_indices_full[s] // top_k`, routing weight
  `topk_weights_flat[sort_indices_full[s]]`). Chunk `k` processes sorted slots
  `[k * capacity, min((k + 1) * capacity, count_total))`, so no slot is ever
  dropped. `reduce_meta0` is chunk 0's `chunk_reduce_metadata` (Pallas only).

  Routing weights cost O(processed windows), never O(num_slots): `w_rows0` is
  chunk 0's `[capacity]` weights (zero past `count_total`), gathered up front by
  the caller as a NON-differentiable view (`stop_gradient`) of
  `topk_weights_flat`; overflow chunks gather their `[capacity]` window from
  `topk_weights_flat` inside the loops. The backward returns the full
  `d(topk_weights_flat)` (chunk 0 + overflow windows scatter-added with unique
  indices) and a zero cotangent for `w_rows0`.

  - Forward: chunk 0 runs the 2 fused Pallas calls
    (`gmmv2_kernel.gmm_v2(..., lhs_gather_indices=..., glu_coeffs=w_rows)` ->
    `[capacity, 2F]` bank + fp8 `act`, then `linear_combine_fwd` reducing top-k
    rows
    into `bf16[T_ep, D // 128, 128]`) and saves ONLY its bank as the
    `moe_gate_up_{sublayer_idx}` residual. A `lax.while_loop` then runs chunks
    `>= 1` (if any), accumulating into the output via
    `linear_combine_fwd(out=...)`.
  - Backward: chunk 0 `linear_bwd_dgrad` -> `linear_bwd_wgrad` (`dwo`), then
  loop A
    over chunks `>= 1` recomputes each chunk's bank + dgrad exactly once,
    finishes `dwo`
    and `d(topk_weights_flat)` (so the `dwo` FSDP reduce-scatter still launches
    before
    chunk 0's gate backward) and stashes the chunk's fp8 `grad_gate` (+ tile
    exponents)
    in an uninitialized `[ceil(num_slots / capacity) - 1, capacity, 2F]` buffer,
    then
    chunk 0 `ragged_gather_tc` -> `gate_bwd_fused` -> `ragged_gather_reduce_tc`,
    then
    loop B over chunks `>= 1` runs the gate backward from the stash and
    accumulates
    `dwi` and `dx`. Both loops run zero iterations when `count_total <=
    capacity`.
  """
  use_pallas = cfg.use_pallas_kernels
  topk = cfg.num_experts_per_tok
  num_tokens_ep = disp.codes_ep.shape[0]
  num_slots = sort_indices_full.shape[0]
  if num_slots != num_tokens_ep * topk or topk_weights_flat.shape != (num_slots,):
    raise ValueError(
        f"Full-length routing must cover T_ep * top_k = {num_tokens_ep * topk}"
        f" slots; got sort_indices_full {sort_indices_full.shape},"
        f" topk_weights_flat {topk_weights_flat.shape}."
    )
  if not 0 < capacity <= num_slots:
    raise ValueError(f"capacity must be in (0, {num_slots}], got {capacity}.")
  if w_rows0.shape != (capacity,):
    raise ValueError(f"w_rows0 must be chunk 0's [{capacity}] weights, got {w_rows0.shape}.")
  if use_pallas and reduce_meta0 is None:
    raise ValueError("Pallas routed MoE requires chunk-0 reduce_meta0.")
  ep_tokens_shape = disp.codes_ep.shape
  ste_dtype = disp.ste_carrier.dtype
  wi_codes_shape = wi.codes.shape
  wi_codes_dtype = wi.codes.dtype
  wi_ste_dtype = wi.dtype
  wi_scale_shape = wi.scale.shape
  wo_codes_shape = wo.codes.shape
  wo_codes_dtype = wo.codes.dtype
  wo_ste_dtype = wo.dtype
  wo_scale_shape = wo.scale.shape
  gate_tiles = gmmv2_kernel.TileSizes(
      tile_m=cfg.gate_fwd_tiles[0],
      tile_k=cfg.gate_fwd_tiles[1],
      tile_n=cfg.gate_fwd_tiles[2],
      bucket_base=cfg.gate_fwd_tiles[3],
  )

  def _window(full_padded: jax.Array, start: jax.Array) -> jax.Array:
    return jax.lax.dynamic_slice_in_dim(full_padded, start, capacity, axis=0)

  def _pad_slots(full: jax.Array) -> jax.Array:
    # Windows of `capacity` slots start below `count_total <= num_slots`.
    return jnp.pad(full, (0, capacity))

  def _more_chunks(start: jax.Array, cnt: jax.Array) -> jax.Array:
    return start + capacity < cnt

  def _window_weight_indices(slot_win: jax.Array, start: jax.Array, cnt: jax.Array) -> jax.Array:
    # Valid rows are distinct slots and rows past `cnt` get distinct out-of-range
    # indices, so window gathers/scatters are unique_indices (no scatter-sort).
    iota = jnp.arange(capacity, dtype=jnp.int32)
    return jnp.where(start + iota < cnt, slot_win, num_slots + iota)

  def _window_weights(
      w_flat_f32: jax.Array,
      slot_win: jax.Array,
      start: jax.Array,
      cnt: jax.Array,
  ) -> jax.Array:
    # Routing weights of sorted slots `[start, start + capacity)`, zero past `cnt`.
    return w_flat_f32.at[_window_weight_indices(slot_win, start, cnt)].get(
        mode="fill", fill_value=0.0, unique_indices=True
    )

  def _gate_gmm_pallas(codes_ep, wi_rhs, gate_out_scale, gs_k, tok_win, gather_start, **glu):
    return gmmv2_kernel.gmm_v2(
        lhs=codes_ep,
        rhs=wi_rhs,
        group_sizes=gs_k,
        lhs_scale=None,
        rhs_scale=None,
        out_scale=gate_out_scale,
        tile_info=gate_tiles,
        preferred_element_type=jnp.bfloat16,
        acc_dtype=jnp.float32,
        group_offset=jnp.int32(0),
        maybe_quantize_lhs=False,
        zero_initialize=False,
        out_is_3d=False,
        flat_rhs=True,
        disable_multi_core_mode=True,
        out_offset=0,
        lhs_gather_indices=tok_win,
        lhs_gather_start=gather_start,
        lhs_gather_size_m=capacity,
        **glu,
    )

  def _gate_gmm_jax(codes_ep, codes_scale, wi_rhs, wi_s, gs_k, n_k, tok_win, gather_start):
    valid = jnp.arange(capacity, dtype=jnp.int32) < n_k
    safe_src = jnp.where(valid, tok_win + gather_start, 0)
    sorted_fp8 = jnp.where(
        valid[:, None, None],
        jnp.take(codes_ep, safe_src, axis=0),
        jnp.zeros((), dtype=jnp.float8_e4m3fn),
    )
    x_deq_2d = sorted_fp8.reshape(capacity, -1).astype(jnp.float32) * codes_scale.reshape(1, 1)
    wi_f32 = wi_rhs.astype(jnp.float32) * wi_s.reshape(1, 1, 1)
    gate_up = jax.lax.ragged_dot_general(
        x_deq_2d,
        wi_f32,
        gs_k,
        _FWD_RAGGED_DIM_NUMS,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], gate_up, jnp.zeros((), dtype=jnp.bfloat16))

  def _fwd_impl(
      disp_in: ops.Fp8DispatchedTokens,
      w0_in: jax.Array,
      w_flat_in: jax.Array,
      wi_in: Fp8SteWeight,
      wo_in: Fp8SteWeight,
      gs_full_in: jax.Array,
      slot_full_in: jax.Array,
      cnt_in: jax.Array,
      rmeta0_in: Any,
      tie_in: jax.Array,
  ):
    scale_tie_i32 = jax.lax.stop_gradient((ops.first_scalar(disp_in.scale) * 0.0).astype(jnp.int32))
    wi_s = _scalar_w_scale(wi_in.scale)
    wo_s = _scalar_w_scale(wo_in.scale)
    act_s_init = jnp.full((1, 1), cfg.moe_act_fp8_scale, dtype=jnp.float32)
    act_s_lc = act_s_init + tie_in
    gate_out_scale = (disp_in.scale.reshape(1, 1) * wi_s).reshape(1, 1)
    w_flat_f32 = w_flat_in.astype(jnp.float32)
    lc_tiles = linear_combine_fwd.Tiles(
        tile_m=cfg.lc_fwd_tiles[0],
        tile_n=cfg.lc_fwd_tiles[1],
        bucket_m=cfg.lc_fwd_tiles[2],
    )

    def _chunk(start, tok_win, w_win, gather_start, rmeta, out_acc, bank_init):
      # Weights stay in `[E, rows // 8, 8, cols]` tile form until the use site, so
      # while-loop carries keep `T(8, 128)` and XLA never relayouts the FP8 weight.
      wi_rhs = ops.routed_w_from_tiles(wi_in.codes)
      wo_rhs = ops.routed_w_from_tiles(wo_in.codes)
      n_k = chunk_num_tokens(cnt_in, start, capacity)
      gs_k = chunk_group_sizes(gs_full_in, start, n_k)
      w_col = w_win[:, None]
      if use_pallas:
        bank_2d, act_fp8 = _gate_gmm_pallas(
            disp_in.codes_ep,
            wi_rhs,
            gate_out_scale,
            gs_k,
            tok_win,
            gather_start,
            out=bank_init,
            glu_coeffs=w_col,
            glu_out_dtype=jnp.float8_e4m3fn,
            glu_out_scale=act_s_init,
        )
        partial = linear_combine_fwd.linear_combine_fwd(
            act_fp8,
            wo_rhs,
            gs_k,
            tok_win,
            0,
            None if out_acc is not None else rmeta.tokens,
            out_scale=(act_s_lc * wo_s).reshape(1, 1),
            num_out_tokens=num_tokens_ep,
            out=out_acc,
            tiles=lc_tiles,
        )
        return partial, bank_2d
      bank_2d = _gate_gmm_jax(
          disp_in.codes_ep,
          disp_in.scale,
          wi_rhs,
          wi_s,
          gs_k,
          n_k,
          tok_win,
          gather_start,
      )
      act_fp8 = ragged_ops.ragged_silu_mul_fwd(
          bank_2d,
          0,
          w_col,
          n_k,
          out_dtype=jnp.float8_e4m3fn,
          out_scale=act_s_init,
          use_pallas=False,
      )
      act_deq = act_fp8.astype(jnp.float32) * act_s_lc
      wo_f32 = wo_rhs.astype(jnp.float32) * wo_s.reshape(1, 1, 1)
      out_2d = jax.lax.ragged_dot_general(
          act_deq,
          wo_f32,
          gs_k,
          _FWD_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(ste_dtype)
      weighted_down_3d = out_2d.reshape(capacity, out_2d.shape[1] // 128, 128)
      valid = jnp.arange(capacity, dtype=jnp.int32) < n_k
      safe_dst = jnp.where(valid, tok_win, num_tokens_ep)
      padded_out = jnp.zeros((num_tokens_ep + 1, *weighted_down_3d.shape[1:]), dtype=ste_dtype)
      partial = padded_out.at[safe_dst].add(jnp.where(valid[:, None, None], weighted_down_3d, 0))[:num_tokens_ep]
      if out_acc is not None:
        partial = out_acc + partial
      return partial, bank_2d

    # Chunk 0: sorted slots [0, capacity); its bank is the only saved residual.
    bank_init = (
        ops.alloc_activation_bank(capacity, 2 * cfg.moe_mlp_dim, dtype=jnp.bfloat16, constrain=True)
        if use_pallas and not cfg.moe_gmm_no_alias_out
        else None
    )
    partial_ep_3d, bank_2d = _chunk(
        0,
        slot_full_in[:capacity] // topk,
        w0_in.astype(jnp.float32),
        scale_tie_i32,
        rmeta0_in,
        None,
        bank_init,
    )
    if use_pallas and cfg.moe_save_bank_in_superblock:
      bank_tiles = ops.bank_to_tiles(ops.constrain_activation_bank(bank_2d, enabled=True))
    elif use_pallas:
      bank_tiles = bank_2d
    else:
      bank_tiles = ops.bank_to_tiles(bank_2d)

    # Chunks >= 1 (overflow past `capacity`): accumulate, save nothing.
    slot_pad = _pad_slots(slot_full_in)

    def _fwd_body(carry):
      start, acc = carry
      start = start + capacity
      slot_win = _window(slot_pad, start)
      acc, _ = _chunk(
          start,
          slot_win // topk,
          _window_weights(w_flat_f32, slot_win, start, cnt_in),
          jnp.int32(0),
          None,
          acc,
          None,
      )
      return start, acc

    _, partial_ep_3d = jax.lax.while_loop(
        lambda carry: _more_chunks(carry[0], cnt_in),
        _fwd_body,
        (jnp.int32(0), partial_ep_3d),
    )

    if wrap_partial_fp8_bwd:
      partial_out: jax.Array | ops.MoePartialTokens = ops.MoePartialTokens(
          y_ep=partial_ep_3d,
          dy_codes=jax.lax.broadcast(jnp.zeros((), dtype=jnp.float8_e5m2), ep_tokens_shape),
          dy_scale=jnp.zeros((1, 1), dtype=jnp.float32),
      )
    else:
      partial_out = partial_ep_3d

    wi_res = None if use_external_bwd_rw else dataclasses.replace(wi_in, w_ste=None)
    wo_res = None if use_external_bwd_rw else dataclasses.replace(wo_in, w_ste=None)
    disp_codes_res = jax.ad_checkpoint.checkpoint_name(disp_in.codes_ep, f"moe_disp_ep_{sublayer_idx}")
    disp_scale_res = jax.ad_checkpoint.checkpoint_name(disp_in.scale, f"moe_disp_ep_{sublayer_idx}")
    bank_tiles_res = jax.ad_checkpoint.checkpoint_name(bank_tiles, f"moe_gate_up_{sublayer_idx}")
    res = (
        disp_codes_res,
        disp_scale_res,
        bank_tiles_res,
        w0_in,
        w_flat_in,
        wi_res,
        wo_res,
        gs_full_in,
        slot_full_in,
        cnt_in,
        rmeta0_in,
    )
    return partial_out, res

  def _bwd_impl(
      res,
      g_partial_ep: jax.Array | ops.MoePartialTokens,
      ps_wi_carrier: Fp8SteWeight | None,
      ps_wo_carrier: Fp8SteWeight | None,
      ext_bwd_rw: ops.BwdRoutedWeightPrefetch | None = None,
  ):
    (
        disp_codes_ep,
        disp_scale,
        bank_tiles,
        w0_in,
        w_flat_in,
        wi_res,
        wo_res,
        gs_full_in,
        slot_full_in,
        cnt_in,
        rmeta0_in,
    ) = res
    ps_wi_raw = ps_wi_carrier.w_ste if isinstance(ps_wi_carrier, Fp8SteWeight) else ps_wi_carrier
    ps_wo_raw = ps_wo_carrier.w_ste if isinstance(ps_wo_carrier, Fp8SteWeight) else ps_wo_carrier
    ps_wi_3d = ops.routed_w_from_tiles(ps_wi_raw) if ps_wi_raw is not None else None
    ps_wo_3d = ops.routed_w_from_tiles(ps_wo_raw) if ps_wo_raw is not None else None

    if ext_bwd_rw is not None:
      wi_codes = ext_bwd_rw.wi_codes
      wi_scale = ext_bwd_rw.wi_scale
      wo_codes = ext_bwd_rw.wo_codes
      wo_scale = ext_bwd_rw.wo_scale
    else:
      assert wi_res is not None and wo_res is not None
      wi_codes = wi_res.codes
      wi_scale = wi_res.scale
      wo_codes = wo_res.codes
      wo_scale = wo_res.scale

    wi_s = _scalar_w_scale(wi_scale)
    wo_s = _scalar_w_scale(wo_scale)
    act_s = jnp.full((1, 1), cfg.moe_act_fp8_scale, dtype=jnp.float32)
    w_flat_f32 = w_flat_in.astype(jnp.float32)

    if isinstance(g_partial_ep, ops.MoePartialTokens):
      dy_ag = g_partial_ep.dy_codes
      dy_scale = g_partial_ep.dy_scale
    else:
      dy_ag = g_partial_ep.astype(jnp.bfloat16) if use_pallas else g_partial_ep
      dy_scale = None

    if use_pallas:
      dgrad_tiles = linear_bwd_split.DgradTiles(*cfg.lbwd_dgrad_tiles)
      dgrad_tr = min(dgrad_tiles.tile_rows, capacity)
      wgrad_tiles = linear_bwd_split.WgradTiles(*cfg.lbwd_wgrad_tiles)
      gbwd_tiles = gate_bwd_fused.FusedTiles(*cfg.gbwd_fused_tiles)
      gate_out_scale = (disp_scale.reshape(1, 1) * wi_s).reshape(1, 1)

      def _recompute_bank(tok_win, gs_k, n_k):
        del n_k
        bank = _gate_gmm_pallas(
            disp_codes_ep,
            ops.routed_w_from_tiles(wi_codes),
            gate_out_scale,
            gs_k,
            tok_win,
            jnp.int32(0),
            out=None,
        )
        return ops.constrain_activation_bank(bank, enabled=True)

      def _dgrad(tok_win, gs_k, n_k, bank_2d, w_col):
        del n_k
        # Fused `linear_bwd_dgrad` (3D dy gather + wo dgrad + dSwiGLU + dw_rows + act_q + e5m2 grad_gate)!
        return linear_bwd_split.linear_bwd_dgrad(
            dy_ag,
            tok_win,
            0,
            ops.routed_w_from_tiles(wo_codes),
            gs_k,
            bank_2d,
            0,
            w_col,
            act_scale=act_s,
            w_scale=wo_s,
            act_qtype=jnp.float8_e4m3fn,
            dy_qtype=jnp.float8_e5m2,
            dy_scale=dy_scale,
            grad_gate_qtype=jnp.float8_e5m2,
            tiles=dgrad_tiles,
        )

      def _dw_rows(dg):
        return dg.grad_coeffs[:, 0].astype(jnp.float32)

      def _wgrad(dg, gs_k, dw_acc):
        return linear_bwd_split.linear_bwd_wgrad(
            dg.act_q,
            dg.dy_q,
            dg.tile_exps,
            gs_k,
            act_scale=act_s,
            dgrad_tile_rows=dgrad_tr,
            dy_scale=dy_scale,
            dw_acc=dw_acc,
            dw_dtype=wo_ste_dtype,
            tiles=wgrad_tiles,
        )

      def _gather_anchor(dg):
        return dy_scale if dy_scale is not None else dg.grad_gate_amax

      def _gate_inputs(dg):
        return (dg.grad_gate, dg.grad_gate_tile_exps)

      def _gate_bwd(gate_in, tok_win, gs_k, n_k, gather_start, dw_acc):
        grad_gate, grad_gate_tile_exps = gate_in
        sorted_fp8 = tc.ragged_gather_tc(
            disp_codes_ep,
            tok_win,
            start=gather_start,
            num_tokens=n_k,
            max_out_tokens=capacity,
            block_size=cfg.tc_gather_block_size,
        )
        # Fused `gate_bwd_fused` (`dx_sorted_3d` in 3D via VMEM relayout + `dwi` in one pass)!
        return gate_bwd_fused.gate_bwd_fused(
            sorted_fp8,
            ops.routed_w_from_tiles(wi_codes),
            gs_k,
            grad_gate,
            dx_scale=wi_s,
            dw_scale=disp_scale.reshape(1, 1),
            dgate_tile_exps=grad_gate_tile_exps,
            dgrad_tile_rows=dgrad_tr,
            dw_acc=dw_acc,
            dw_dtype=wi_ste_dtype,
            tiles=gbwd_tiles,
        )

      def _reduce(dx_sorted_3d, tok_win, n_k, rmeta):
        del tok_win, n_k
        return tc.ragged_gather_reduce_tc(dx_sorted_3d, rmeta, num_out_tokens=num_tokens_ep, top_k=topk)

      bank0_2d = ops.bank_from_tiles(bank_tiles, constrain=True) if bank_tiles.ndim == 3 else bank_tiles
    else:
      bwd_tie_f32 = jax.lax.stop_gradient(ops.first_scalar(dy_scale) * 0.0) if dy_scale is not None else 0.0
      wo_f32 = ops.routed_w_from_tiles(wo_codes).astype(jnp.float32) * wo_s.reshape(1, 1, 1)
      wi_f32 = ops.routed_w_from_tiles(wi_codes).astype(jnp.float32) * wi_s.reshape(1, 1, 1)

      def _recompute_bank(tok_win, gs_k, n_k):
        return _gate_gmm_jax(
            disp_codes_ep,
            disp_scale,
            ops.routed_w_from_tiles(wi_codes),
            wi_s,
            gs_k,
            n_k,
            tok_win,
            jnp.int32(0),
        )

      def _dgrad(tok_win, gs_k, n_k, bank_2d, w_col):
        valid = jnp.arange(capacity, dtype=jnp.int32) < n_k
        gathered_dy = jnp.where(
            valid[:, None, None],
            jnp.take(dy_ag, jnp.where(valid, tok_win, 0), axis=0),
            jnp.zeros((), dtype=dy_ag.dtype),
        )
        if dy_scale is not None:
          g_deq_2d = gathered_dy.reshape(capacity, -1).astype(jnp.float32) * (dy_scale.reshape(1, 1) + bwd_tie_f32)
        else:
          g_deq_2d = _rowwise_pow2_fp8_roundtrip(gathered_dy, jnp.float8_e5m2)
        act_fp8_bwd = ragged_ops.ragged_silu_mul_fwd(
            bank_2d,
            0,
            w_col,
            n_k,
            out_dtype=jnp.float8_e4m3fn,
            out_scale=act_s,
            use_pallas=False,
        )
        act_deq_2d = act_fp8_bwd.astype(jnp.float32) * act_s
        d_act = jax.lax.ragged_dot_general(
            g_deq_2d,
            wo_f32,
            gs_k,
            _DLHS_RAGGED_DIM_NUMS,
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        d_gate_up, dw_rows_2d = ragged_ops.ragged_silu_mul_bwd(
            d_act, bank_2d, 0, w_col, n_k, with_absmax=False, use_pallas=False
        )
        d_gate_deq = _rowwise_pow2_fp8_roundtrip(d_gate_up, jnp.float8_e5m2)
        return act_deq_2d, g_deq_2d, d_gate_deq, dw_rows_2d

      def _dw_rows(dg):
        return dg[3][:, 0].astype(jnp.float32)

      def _wgrad(dg, gs_k, dw_acc):
        act_deq_2d, g_deq_2d, _, _ = dg
        dwo_3d = jax.lax.ragged_dot_general(
            act_deq_2d,
            g_deq_2d,
            gs_k,
            _DRHS_RAGGED_DIM_NUMS,
            preferred_element_type=jnp.float32,
        ).astype(wo_ste_dtype)
        return dwo_3d if dw_acc is None else dw_acc + dwo_3d

      def _gather_anchor(dg):
        del dg
        return None

      def _gate_inputs(dg):
        return (dg[2],)

      def _gate_bwd(gate_in, tok_win, gs_k, n_k, gather_start, dw_acc):
        del gather_start
        (d_gate_deq,) = gate_in
        valid = jnp.arange(capacity, dtype=jnp.int32) < n_k
        sorted_fp8 = jnp.where(
            valid[:, None, None],
            jnp.take(disp_codes_ep, jnp.where(valid, tok_win, 0), axis=0),
            jnp.zeros((), dtype=jnp.float8_e4m3fn),
        )
        x_deq_2d = sorted_fp8.reshape(capacity, -1).astype(jnp.float32) * disp_scale.reshape(1, 1)
        dwi_3d = jax.lax.ragged_dot_general(
            x_deq_2d,
            d_gate_deq,
            gs_k,
            _DRHS_RAGGED_DIM_NUMS,
            preferred_element_type=jnp.float32,
        ).astype(wi_ste_dtype)
        if dw_acc is not None:
          dwi_3d = dw_acc + dwi_3d
        dx_2d = jax.lax.ragged_dot_general(
            d_gate_deq,
            wi_f32,
            gs_k,
            _DLHS_RAGGED_DIM_NUMS,
            preferred_element_type=jnp.float32,
        ).astype(ste_dtype)
        return dx_2d.reshape(capacity, dx_2d.shape[1] // 128, 128), dwi_3d

      def _reduce(dx_sorted_3d, tok_win, n_k, rmeta):
        del rmeta
        valid = jnp.arange(capacity, dtype=jnp.int32) < n_k
        safe_dst = jnp.where(valid, tok_win, num_tokens_ep)
        padded_dx = jnp.zeros((num_tokens_ep + 1, *dx_sorted_3d.shape[1:]), dtype=ste_dtype)
        return padded_dx.at[safe_dst].add(jnp.where(valid[:, None, None], dx_sorted_3d, 0))[:num_tokens_ep]

      bank0_2d = ops.bank_from_tiles(bank_tiles, constrain=False)

    slot_pad = _pad_slots(slot_full_in)

    def _chunk_routing(start):
      n_k = chunk_num_tokens(cnt_in, start, capacity)
      gs_k = chunk_group_sizes(gs_full_in, start, n_k)
      slot_win = _window(slot_pad, start)
      return n_k, gs_k, slot_win, slot_win // topk

    def _scatter_dw_rows(dw_flat_acc, dg, slot_win, start):
      # Rows past `cnt` (kernel garbage) map to out-of-range indices and are dropped;
      # every valid slot is written exactly once across all windows.
      idx = _window_weight_indices(slot_win, start, cnt_in)
      rows = _dw_rows(dg)
      if cfg.moe_dw_scatter_set:
        return dw_flat_acc.at[idx].set(rows, mode="drop", unique_indices=True)
      return dw_flat_acc.at[idx].add(rows, mode="drop", unique_indices=True)

    def _chunk_dgrad(start):
      n_k, gs_k, slot_win, tok_win = _chunk_routing(start)
      w_col = _window_weights(w_flat_f32, slot_win, start, cnt_in)[:, None]
      bank_k = _recompute_bank(tok_win, gs_k, n_k)
      return gs_k, slot_win, _dgrad(tok_win, gs_k, n_k, bank_k, w_col)

    # Overflow chunks `k >= 1` (at most `max_overflow`) each recompute their bank and
    # dgrad exactly once, in loop A.
    max_overflow = -(-num_slots // capacity) - 1

    # 1. Chunk 0 dgrad + `dwo` IMMEDIATELY from the saved bank.
    n0 = chunk_num_tokens(cnt_in, 0, capacity)
    gs0 = chunk_group_sizes(gs_full_in, 0, n0)
    slot0 = slot_full_in[:capacity]
    tok0 = slot0 // topk
    dg0 = _dgrad(tok0, gs0, n0, bank0_2d, w0_in.astype(jnp.float32)[:, None])
    dwo_3d = _wgrad(dg0, gs0, ps_wo_3d)
    dw_flat = _scatter_dw_rows(jnp.zeros((num_slots,), jnp.float32), dg0, slot0, 0)

    gate_in0 = _gate_inputs(dg0)

    # 2. Loop A: chunks >= 1 recompute bank + dgrad once, finish `dwo` +
    #    `d(topk_weights_flat)` (so the `dwo` FSDP RS launches early) and stash the
    #    chunk's gate-backward inputs (`grad_gate` + tile exponents) for loop B. Stash
    #    slot `k - 1` is fully overwritten by chunk `k`'s dgrad outputs before loop B
    #    reads it, and slots past the last overflow chunk are never read, so loop B sees
    #    exactly the dgrad outputs a recompute would produce (rows past `n_k` are masked
    #    in both `gate_bwd_fused` dW operands and never DMA'd into `dx`).
    if max_overflow > 0 and cfg.moe_bwd_stash_overflow:
      gate_stash = jax.tree.map(
          lambda x: jax.lax.empty((max_overflow, *x.shape), dtype=x.dtype),
          gate_in0,
      )

      def _body_a(carry):
        start, dwo_acc, dw_flat_acc, stash = carry
        start = start + capacity
        gs_k, slot_win, dg_k = _chunk_dgrad(start)
        dwo_acc = _wgrad(dg_k, gs_k, dwo_acc)
        dw_flat_acc = _scatter_dw_rows(dw_flat_acc, dg_k, slot_win, start)
        slot = start // capacity - 1
        stash = jax.tree.map(
            lambda st, x: jax.lax.dynamic_update_index_in_dim(st, x, slot, axis=0),
            stash,
            _gate_inputs(dg_k),
        )
        return start, dwo_acc, dw_flat_acc, stash

      _, dwo_3d, dw_flat, gate_stash = jax.lax.while_loop(
          lambda carry: _more_chunks(carry[0], cnt_in),
          _body_a,
          (jnp.int32(0), dwo_3d, dw_flat, gate_stash),
      )
    elif max_overflow > 0:
      gate_stash = None

      def _body_a_nostash(carry):
        start, dwo_acc, dw_flat_acc = carry
        start = start + capacity
        gs_k, slot_win, dg_k = _chunk_dgrad(start)
        dwo_acc = _wgrad(dg_k, gs_k, dwo_acc)
        dw_flat_acc = _scatter_dw_rows(dw_flat_acc, dg_k, slot_win, start)
        return start, dwo_acc, dw_flat_acc

      _, dwo_3d, dw_flat = jax.lax.while_loop(
          lambda carry: _more_chunks(carry[0], cnt_in),
          _body_a_nostash,
          (jnp.int32(0), dwo_3d, dw_flat),
      )

    # 3. Chunk 0 gate backward. Gate the `sorted_fp8` gather on `dy_scale`/`grad_gate_amax`
    #    so XLA never hoists a `[capacity, D // 128, 128]` allocation across the B4 -> B3 barrier!
    anchor = _gather_anchor(dg0)
    gather_start_i32 = (
        jax.lax.stop_gradient((ops.first_scalar(anchor) * 0.0).astype(jnp.int32)) if anchor is not None else jnp.int32(0)
    )
    if use_pallas and cfg.moe_bwd_tie_wgrad_before_gate:
      dwo_tie_i32 = jax.lax.stop_gradient((ops.first_scalar(dwo_3d).astype(jnp.float32) * 0.0).astype(jnp.int32))
      gather_start_i32 = gather_start_i32 + dwo_tie_i32
    dx_sorted0, dwi_3d = _gate_bwd(gate_in0, tok0, gs0, n0, gather_start_i32, ps_wi_3d)
    dx_ep_3d = _reduce(dx_sorted0, tok0, n0, rmeta0_in)

    # 4. Loop B: chunks >= 1 run gate backward from the stashed dgrad (no bank, dgrad or
    #    routing-weight recompute) and accumulate `dwi` + `dx`.
    if max_overflow > 0:

      def _body_b(carry):
        start, dwi_acc, dx_acc = carry
        start = start + capacity
        n_k, gs_k, _, tok_win = _chunk_routing(start)
        if gate_stash is not None:
          slot = start // capacity - 1
          gate_in_k = jax.tree.map(
              lambda st: jax.lax.dynamic_index_in_dim(st, slot, axis=0, keepdims=False),
              gate_stash,
          )
        else:
          _, _, dg_k = _chunk_dgrad(start)
          gate_in_k = _gate_inputs(dg_k)
        dx_sorted_k, dwi_acc = _gate_bwd(gate_in_k, tok_win, gs_k, n_k, jnp.int32(0), dwi_acc)
        rmeta_k = chunk_reduce_metadata(tok_win, n_k, cfg, num_tokens_ep) if use_pallas else None
        dx_acc = dx_acc + _reduce(dx_sorted_k, tok_win, n_k, rmeta_k)
        return start, dwi_acc, dx_acc

      _, dwi_3d, dx_ep_3d = jax.lax.while_loop(
          lambda carry: _more_chunks(carry[0], cnt_in),
          _body_b,
          (jnp.int32(0), dwi_3d, dx_ep_3d),
      )

    dwi_ste = ops.routed_w_to_tiles(dwi_3d) if len(wi_codes_shape) == 4 else dwi_3d
    dwo_ste = ops.routed_w_to_tiles(dwo_3d) if len(wo_codes_shape) == 4 else dwo_3d
    d_disp = ops.Fp8DispatchedTokens(
        ste_carrier=dx_ep_3d.astype(ste_dtype),
        codes_ep=jax.lax.broadcast(jnp.zeros((), dtype=jnp.float8_e4m3fn), ep_tokens_shape),
        scale=jnp.zeros_like(disp_scale),
    )
    dwi = Fp8SteWeight(
        w_ste=dwi_ste,
        codes=jax.lax.broadcast(jnp.zeros((), dtype=wi_codes_dtype), wi_codes_shape),
        scale=jnp.zeros(wi_scale_shape, dtype=jnp.float32),
    )
    dwo = Fp8SteWeight(
        w_ste=dwo_ste,
        codes=jax.lax.broadcast(jnp.zeros((), dtype=wo_codes_dtype), wo_codes_shape),
        scale=jnp.zeros(wo_scale_shape, dtype=jnp.float32),
    )
    return (
        d_disp,
        None,  # `w_rows0` is a stop_gradient view of `topk_weights_flat`.
        dw_flat.astype(w_flat_in.dtype),
        dwi,
        dwo,
        None,
        None,
        None,
        None,
        None,
    )

  args = (
      disp,
      w_rows0,
      topk_weights_flat,
      wi,
      wo,
      local_gs_full,
      sort_indices_full,
      count_total,
      reduce_meta0,
      # `forward_tie` enters as an explicit zero-valued scalar operand (a custom_vjp
      # must not close over a traced value).
      ops.order_after(jnp.zeros((1, 1), jnp.float32), forward_tie)
      if forward_tie is not None
      else jnp.zeros((1, 1), jnp.float32),
  )

  def _zero_slot():
    return ops.zero_bwd_rw_prefetch(wi_codes_shape, wi_scale_shape, wo_codes_shape, wo_scale_shape)

  if use_external_bwd_rw:
    if return_rhs:

      @jax.custom_vjp
      def _op_ret_ext(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in):
        out, _ = _fwd_impl(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in)
        return out, wi_in, wo_in, _zero_slot()

      def _op_ret_ext_fwd(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in):
        out, res = _fwd_impl(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in)
        return (out, wi_in, wo_in, _zero_slot()), res

      def _op_ret_ext_bwd(res, grads):
        g_out, g_wi_ps, g_wo_ps, g_ext_rw = grads
        return _bwd_impl(res, g_out, g_wi_ps, g_wo_ps, ext_bwd_rw=g_ext_rw)

      _op_ret_ext.defvjp(_op_ret_ext_fwd, _op_ret_ext_bwd)
      return _op_ret_ext(*args)

    @jax.custom_vjp
    def _op_ext(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in):
      out, _ = _fwd_impl(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in)
      return out, _zero_slot()

    def _op_ext_fwd(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in):
      out, res = _fwd_impl(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in)
      return (out, _zero_slot()), res

    def _op_ext_bwd(res, grads):
      g_out, g_ext_rw = grads
      return _bwd_impl(res, g_out, None, None, ext_bwd_rw=g_ext_rw)

    _op_ext.defvjp(_op_ext_fwd, _op_ext_bwd)
    return _op_ext(*args)

  if return_rhs:

    @jax.custom_vjp
    def _op_ret(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in):
      out, _ = _fwd_impl(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in)
      return out, wi_in, wo_in

    def _op_ret_fwd(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in):
      out, res = _fwd_impl(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in)
      return (out, wi_in, wo_in), res

    def _op_ret_bwd(res, grads):
      g_out, g_wi_ps, g_wo_ps = grads
      return _bwd_impl(res, g_out, g_wi_ps, g_wo_ps)

    _op_ret.defvjp(_op_ret_fwd, _op_ret_bwd)
    return _op_ret(*args)

  @jax.custom_vjp
  def _op(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in):
    out, _ = _fwd_impl(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in)
    return out

  def _op_fwd(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in):
    return _fwd_impl(d_in, w0_in, wf_in, wi_in, wo_in, gs_in, si_in, n_in, rm_in, tie_in)

  def _op_bwd(res, g_out):
    return _bwd_impl(res, g_out, None, None)

  _op.defvjp(_op_fwd, _op_bwd)
  return _op(*args)
