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

"""Pure-JAX MoE Pallas gmmv2/tgmm_v2 wrappers & SC1 FP8 STE weight gather.

Empirically measured Ghostfish (`gf=1x1x1`, Qwen MAX `EP=16`, `D=8192`,
`F=2048`, `g=32`, `buf_size=43008`, `2`-mb, `xm_hlo_runner` `--num_repeats=50`,
single-TC `chip_no:0 & tc.core_no:0`) `lbwd_wgrad_tiles` sweep (`has_ps=False` /
`has_ps=True` / step wall):
  - `(1024, 256, 2048)`: `1.204 ms` / `1.477 ms` (`2.681 ms`), `19.01 ms` step
    (fastest overall; `58.80M` / `64.00M` AOT scoped VMEM) -> recommended
    `QwenConfig.lbwd_wgrad_tiles` for Qwen MAX `EP=16`.
  - `(768, 256, 2048)`: `1.173 ms` / `1.646 ms` (`2.819 ms`), `19.15 ms` step
    (`56.28M` / `64.00M` AOT scoped VMEM; default baseline).
  - `(1024, 512, 1024)`: `1.291 ms` / `1.728 ms` (`3.019 ms`), `19.31 ms` step
    (`56.01M` / `64.00M` AOT scoped VMEM).
  - `(1280, 256, 2048)`: `4.909 ms` / `5.072 ms` (`9.981 ms`), `26.28 ms` step
    (`61.33M` / `64.00M` AOT scoped VMEM; severe slowdown from VMEM/spill).
  - `(1536, 256, 2048)`: AOT scoped VMEM OOM (`63.33M > 62.72M`).
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

import jax
import jax.numpy as jnp

from maxtext.experimental.lineage.qwen import ops
from maxtext.experimental.lineage.qwen import ragged_ops
from maxtext.experimental.lineage.qwen.config import QwenConfig
from maxtext.experimental.lineage.qwen.ext_kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.experimental.lineage.qwen.ext_kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.kernels.ragged import ragged_gather_tc as tc
from maxtext.experimental.lineage.qwen.gmmv2 import gate_bwd_fused
from maxtext.experimental.lineage.qwen.gmmv2 import gmmv2_kernel
from maxtext.experimental.lineage.qwen.gmmv2 import linear_bwd_split
from maxtext.experimental.lineage.qwen.gmmv2 import linear_combine_fwd

# Keep in-kernel FP8 MXU mode active globally; W8A16 software dequant is
# strictly disallowed by the Global Mixed-Precision Invariant.
os.environ.setdefault("MAXTEXT_GMM_V2_IN_KERNEL_FP8_DEQUANT", "1")

_FP8_E4M3FN_NEG_ZERO_BITS = 0x80
_ALLOWED_REDUCTION_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))

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


def assert_global_mixed_precision_contract(cfg: QwenConfig) -> None:
  """Enforces native FP8 MXU compute and BF16+ mandatory reductions."""
  if cfg.moe_fp8_weights and not cfg.moe_fp8_gmm_compute:
    raise ValueError(
        "Global Mixed-Precision Invariant violation: W8A16 software dequant"
        " (`moe_fp8_weights=True, moe_fp8_gmm_compute=False`) is forbidden;"
        " native `fp8_e4m3fn x fp8_e4m3fn` MXU compute is required."
    )
  if cfg.moe_fp8_combine_sc0:
    raise ValueError(
        "Global Mixed-Precision Invariant violation: `moe_fp8_combine_sc0=True`"
        " is forbidden; EP Combine ReduceScatter MUST run in bfloat16 or"
        " float32."
    )


def _assert_bf16_plus_dtype(
    dtype: jax.typing.DTypeLike, op_name: str
) -> jnp.dtype:
  """Validates `dtype` is bfloat16 or float32 before gradient reductions."""
  dt = jnp.dtype(dtype)
  if dt not in _ALLOWED_REDUCTION_DTYPES:
    raise ValueError(
        f"{op_name} reduction dtype must be bfloat16 or float32, got {dt}."
    )
  return dt


def _largest_pow2_divisor(n: int) -> int:
  return max(1, n & -n)


def _clamp_mult128_tile(
    requested: int, full_dim: int, *, min_tile: int = 128
) -> int:
  """Clamps a tile size so it divides `full_dim` and respects alignment."""
  if full_dim <= min_tile:
    return full_dim
  cap = min(max(requested, min_tile), full_dim)
  step = min_tile if full_dim % min_tile == 0 else 128
  for cand in range((cap // step) * step, step - 1, -step):
    if full_dim % cand == 0:
      return cand
  return full_dim


def _clamp_3d_k_tile(requested: int, full_dim: int) -> int:
  """Clamps a 3D-relayout tile (`tile_d0 % 8 == 0` or `tile == full_dim`)."""
  if full_dim < 1024 or full_dim % 1024 != 0:
    return full_dim
  cap = min(max(requested, 1024), full_dim)
  for cand in range((cap // 1024) * 1024, 1023, -1024):
    if full_dim % cand == 0:
      return cand
  return full_dim


def _clamp_triple_tiles(
    req: tuple[int, int, int], m_dim: int, k_dim: int, n_dim: int
) -> tuple[int, int, int]:
  """Clamps `(tile_m, tile_k, tile_n)` cleanly to `(m_dim, k_dim, n_dim)`."""
  tm = min(req[0], m_dim)
  tk = _clamp_mult128_tile(req[1], k_dim, min_tile=128)
  tn = _clamp_mult128_tile(req[2], n_dim, min_tile=128)
  return (tm, tk, tn)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Fp8SteWeight:
  """Pure-JAX FP8 weight carrier for native FP8 MXU compute with BF16+ STE.

  Attributes:
    w_ste: Zero-broadcast `[E_local, K, N]` (or 4D tiled `[E_local, K//8, 8,
      N]`) bfloat16/float32 straight-through gradient carrier.
    codes: `[E_local, K, N]` (or 4D tiled) `float8_e4m3fn` quantized weights.
    scale: `[E_local, 1, N]` or `[1, 1]` `float32` weight scale.
  """

  w_ste: Any
  codes: jax.Array
  scale: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    """Unpacked logical 3D weight shape `[E_local, K, N]`."""
    if self.codes.ndim == 4:
      return (
          self.codes.shape[0],
          self.codes.shape[1] * self.codes.shape[2],
          self.codes.shape[3],
      )
    return self.codes.shape

  @property
  def dtype(self) -> jnp.dtype:
    """STE gradient dtype (`bfloat16` or `float32`)."""
    return jnp.bfloat16 if self.w_ste is None else self.w_ste.dtype

  @property
  def ndim(self) -> int:
    """Logical tensor rank (`3` for `[E_local, K, N]`)."""
    return 3 if self.codes.ndim == 4 else self.codes.ndim


def compute_hoisted_moe_absmax(
    w_stacked_local: jax.Array,
    cfg: QwenConfig | None = None,
    *,
    per_expert_scalar: bool | None = None,
) -> jax.Array:
  """Computes local (or EP-group `pmax`) MoE weight absmax across superblocks.

  `w_stacked_local` has shape `[num_superblocks, E_shard, in_dim, out_dim]` on
  each TensorCore. For the forked 3D pipeline (`cfg.moe_use_forked_3d_gmmv2`),
  all kernels take a single `(1, 1)` scalar weight scale per EP group, so we
  reduce over `(1, 2, 3)` and `pmax` over `'fsdp'` before `scan` to avoid
  mismatching per-expert scales and eliminate SC1 scale all-gathers inside
  `scan`.

  Args:
    w_stacked_local: Stacked superblock MoE weight shards.
    cfg: Optional `QwenConfig` to select scalar EP-wide vs per-expert absmax.
    per_expert_scalar: Explicit override. When False and `cfg is None`, reduces
      per output channel (`axis=2`).

  Returns:
    Stop-gradient `float32` absmax tensor (`[num_superblocks, 1, 1]` for forked
    3D, `[num_superblocks, E_shard, 1, 1]` for per-expert scalar, or
    `[num_superblocks, E_shard, 1, out_dim]` for per-channel).
  """
  forked_ep_scalar = (
      cfg is not None
      and per_expert_scalar is None
      and cfg.moe_use_forked_3d_gmmv2
      and cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and cfg.moe_use_ragged_silu_3d_fp8
      and cfg.moe_fp8_token_dispatch
      and cfg.moe_fp8_gmm_compute
  )
  if forked_ep_scalar:
    amax_local = jax.lax.stop_gradient(
        jnp.max(
            jnp.abs(w_stacked_local),
            axis=tuple(range(1, w_stacked_local.ndim)),
        ).astype(jnp.float32)
    )
    amax_ep = (
        jax.lax.pmax(amax_local, axis_name="fsdp")
        if (cfg is not None and cfg.fsdp_size > 1)
        else amax_local
    )
    return jax.lax.stop_gradient(amax_ep[:, None, None])

  use_expert_scalar = (
      bool(cfg.moe_fp8_gmm_compute)
      if (per_expert_scalar is None and cfg is not None)
      else bool(per_expert_scalar)
  )
  red_axis = (2, 3) if use_expert_scalar else 2
  absmax = jnp.max(
      jnp.abs(w_stacked_local), axis=red_axis, keepdims=True
  ).astype(jnp.float32)
  return jax.lax.stop_gradient(absmax)


def _prepare_gmm_rhs_scale(scale: jax.Array, out_cols: int) -> jax.Array:
  """Formats `[G, 1, S]` (`S` in `{1, out_cols}`) into `(G, 1, 1, out_cols)`."""
  s4 = jnp.expand_dims(scale, axis=1)
  if s4.shape[-1] == out_cols:
    return s4
  return jnp.broadcast_to(s4, (s4.shape[0], s4.shape[1], 1, out_cols))


def quantize_moe_codes_from_absmax(
    w_local: jax.Array, absmax_local: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `[E_shard, in_dim, out_dim]` weight shard to `(codes_fp8, scale_f32)`."""
  qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
  scale = jnp.where(
      absmax_local > 0.0, absmax_local / qmax, jnp.ones_like(absmax_local)
  )
  scale_div = scale[None, :, :] if scale.ndim == 2 else scale
  q = jnp.clip(w_local.astype(jnp.float32) / scale_div, -qmax, qmax).astype(
      jnp.float8_e4m3fn
  )
  bits = jax.lax.bitcast_convert_type(q, jnp.uint8)
  bits = jnp.where(
      bits == _FP8_E4M3FN_NEG_ZERO_BITS, jnp.zeros_like(bits), bits
  )
  codes = jax.lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
  codes = jax.lax.optimization_barrier(codes)
  return codes, scale


def _k_sharded_across_fsdp(cfg: QwenConfig) -> bool:
  """True when `E < num_devices` (`512c = 1024 TCs`) so `K` is FSDP-sharded."""
  return cfg.num_experts % cfg.num_devices != 0


def gather_routed_weight_sc1(
    w_local: jax.Array,
    absmax_local: jax.Array | None,
    cfg: QwenConfig,
    *,
    start_after: jax.Array | None = None,
    is_wi: bool = False,
) -> jax.Array | Fp8SteWeight:
  """Gathers routed expert weights over `'fsdp'` on SC1; scatters BF16+ grads.

  Supports both standard expert-sharded layouts (`64c`, `128c`, `256c`,
  sharded along axis `0` over `'fsdp'`) and `K`-sharded layouts (`512c = 1024`
  TensorCores, sharded along axis `1` over `'fsdp'`).

  Args:
    w_local: Local sharded routed weight `[E_shard, K, N]` or `[E_local,
      K_shard, N]`.
    absmax_local: Optional precomputed `float32` absmax tensor for FP8 weight
      gather.
    cfg: Model & sharding config.
    start_after: Optional forward-only barrier dependency.
    is_wi: Whether this weight is `wi` (used for SC0/SC1 backward RS balance).

  Returns:
    Gathered `Fp8SteWeight` when `cfg.moe_fp8_weights` and `absmax_local` is
    provided, otherwise gathered BF16/FP32 weight array.
  """
  assert_global_mixed_precision_contract(cfg)
  _assert_bf16_plus_dtype(w_local.dtype, "gather_routed_weight_sc1")
  ag_core = ops._fsdp_ag_core_id(cfg, is_bwd_remat=False)  # pylint: disable=protected-access
  rs_core = ops._moe_rs_core_id(cfg, is_wi=is_wi)  # pylint: disable=protected-access
  k_shards = cfg.moe_k_shards
  flat_k_sharded = k_shards > 1
  legacy_k_sharded = k_shards == 0
  k_sharded = flat_k_sharded or legacy_k_sharded
  gather_axis = 1 if legacy_k_sharded else 0
  use_tiles = cfg.routed_w_tiled_layout and (w_local.shape[1] % 8 == 0)
  if flat_k_sharded:
    full_k = w_local.shape[1] * k_shards
    e_full = (w_local.shape[0] * cfg.fsdp_size) // k_shards
  elif legacy_k_sharded:
    full_k = w_local.shape[1] * cfg.fsdp_size
    e_full = w_local.shape[0]
  else:
    full_k = w_local.shape[1]
    e_full = w_local.shape[0] * cfg.fsdp_size
  keep_tiled_carrier = (
      cfg.moe_use_forked_3d_gmmv2 and use_tiles and (full_k % 8 == 0)
  )
  use_fp8 = cfg.moe_fp8_weights and absmax_local is not None
  min_b = cfg.sc_min_collective_bytes
  full_shape = (e_full, full_k, w_local.shape[2])

  if use_fp8:
    assert absmax_local is not None
    amax_in = (
        ops.order_after(absmax_local, start_after)
        if start_after is not None
        else absmax_local
    )
    if k_sharded:
      amax_in = jax.lax.pmax(amax_in, axis_name="fsdp")

    @jax.custom_vjp
    def _gather_fp8(w: jax.Array, amax: jax.Array) -> Fp8SteWeight:
      codes_l, scale_l = quantize_moe_codes_from_absmax(w, amax)
      if use_tiles:
        codes_l = ops.routed_w_to_tiles(codes_l)
      codes_f = ops.sc_all_gather(
          codes_l,
          "fsdp",
          core_id=ag_core,
          axis=gather_axis,
          min_collective_bytes=min_b,
      )
      if flat_k_sharded:
        if use_tiles and keep_tiled_carrier:
          codes_f = codes_f.reshape(e_full, full_k // 8, 8, codes_f.shape[-1])
        elif use_tiles:
          codes_f = ops.routed_w_from_tiles(codes_f).reshape(full_shape)
        else:
          codes_f = codes_f.reshape(full_shape)
      elif use_tiles and not keep_tiled_carrier:
        codes_f = ops.routed_w_from_tiles(codes_f)
      if scale_l.ndim == 2:
        scale_f = scale_l
      elif flat_k_sharded:
        scale_f = jnp.broadcast_to(scale_l, (e_full, 1, scale_l.shape[-1]))
      elif legacy_k_sharded:
        scale_f = scale_l
      else:
        scale_f = ops.sc_all_gather(
            scale_l,
            "fsdp",
            core_id=ag_core,
            axis=0,
            min_collective_bytes=min_b,
        )
      ste_shape = codes_f.shape if keep_tiled_carrier else full_shape
      w_ste = jax.lax.broadcast(jnp.zeros((), dtype=w.dtype), ste_shape)
      return Fp8SteWeight(w_ste=w_ste, codes=codes_f, scale=scale_f)

    def _gather_fp8_fwd(w: jax.Array, amax: jax.Array):
      return _gather_fp8(w, amax), None

    def _gather_fp8_bwd(_, g_carrier: Fp8SteWeight):
      dw_full = g_carrier.w_ste
      ops.assert_bf16_or_fp32_reduction(dw_full, "gather_routed_weight_sc1_bwd")
      if dw_full.ndim == 4:
        dw_rs = (
            dw_full.reshape(
                dw_full.shape[0] * k_shards,
                dw_full.shape[1] // k_shards,
                dw_full.shape[2],
                dw_full.shape[3],
            )
            if flat_k_sharded
            else dw_full
        )
        dw_local_tiled = ops.sc_psum_scatter(
            dw_rs,
            "fsdp",
            core_id=rs_core,
            scatter_dimension=gather_axis,
            min_collective_bytes=min_b,
        )
        dw_local = ops.routed_w_from_tiles(dw_local_tiled)
      elif use_tiles:
        dw_in = (
            dw_full.reshape(
                dw_full.shape[0] * k_shards,
                dw_full.shape[1] // k_shards,
                dw_full.shape[2],
            )
            if flat_k_sharded
            else dw_full
        )
        dw_tiled = ops.routed_w_to_tiles(dw_in)
        dw_local_tiled = ops.sc_psum_scatter(
            dw_tiled,
            "fsdp",
            core_id=rs_core,
            scatter_dimension=gather_axis,
            min_collective_bytes=min_b,
        )
        dw_local = ops.routed_w_from_tiles(dw_local_tiled)
      else:
        dw_rs = (
            dw_full.reshape(
                dw_full.shape[0] * k_shards,
                dw_full.shape[1] // k_shards,
                dw_full.shape[2],
            )
            if flat_k_sharded
            else dw_full
        )
        dw_local = ops.sc_psum_scatter(
            dw_rs,
            "fsdp",
            core_id=rs_core,
            scatter_dimension=gather_axis,
            min_collective_bytes=min_b,
        )
      return dw_local, None

    _gather_fp8.defvjp(_gather_fp8_fwd, _gather_fp8_bwd)
    return _gather_fp8(w_local, amax_in)

  w_sharded_in = (
      ops.order_after(w_local, start_after)
      if start_after is not None
      else w_local
  )

  @jax.custom_vjp
  def _gather_bf16(w: jax.Array) -> jax.Array:
    w_in = ops.routed_w_to_tiles(w) if use_tiles else w
    w_f = ops.sc_all_gather(
        w_in,
        "fsdp",
        core_id=ag_core,
        axis=gather_axis,
        min_collective_bytes=min_b,
    )
    w_out = ops.routed_w_from_tiles(w_f) if use_tiles else w_f
    return w_out.reshape(full_shape) if flat_k_sharded else w_out

  def _gather_bf16_fwd(w: jax.Array):
    return _gather_bf16(w), None

  def _gather_bf16_bwd(_, dw_full: jax.Array):
    ops.assert_bf16_or_fp32_reduction(
        dw_full, "gather_routed_weight_sc1_bf16_bwd"
    )
    dw_in = (
        dw_full.reshape(
            dw_full.shape[0] * k_shards,
            dw_full.shape[1] // k_shards,
            dw_full.shape[2],
        )
        if flat_k_sharded
        else dw_full
    )
    if use_tiles:
      dw_tiled = ops.routed_w_to_tiles(dw_in)
      dw_local = ops.routed_w_from_tiles(
          ops.sc_psum_scatter(
              dw_tiled,
              "fsdp",
              core_id=rs_core,
              scatter_dimension=gather_axis,
              min_collective_bytes=min_b,
          )
      )
    else:
      dw_local = ops.sc_psum_scatter(
          dw_in,
          "fsdp",
          core_id=rs_core,
          scatter_dimension=gather_axis,
          min_collective_bytes=min_b,
      )
    return (dw_local,)

  _gather_bf16.defvjp(_gather_bf16_fwd, _gather_bf16_bwd)
  return _gather_bf16(w_sharded_in)


def bwd_rw3_prefetch_shapes(
    cfg: QwenConfig, amax_ndim: int = 2
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  """Returns `(wi_codes_shape, wi_scale_shape, wo_codes_shape, wo_scale_shape)`."""
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
    cfg: QwenConfig,
    *,
    start_after: Any = None,
) -> tuple[jax.Array, jax.Array]:
  """Gathers `float8_e4m3fn` `(codes_f, scale_f)` on SC1 in backward."""
  ag_core = ops._fsdp_ag_core_id(cfg, is_bwd_remat=False)  # pylint: disable=protected-access
  k_shards = cfg.moe_k_shards
  flat_k_sharded = k_shards > 1
  legacy_k_sharded = k_shards == 0
  k_sharded = flat_k_sharded or legacy_k_sharded
  gather_axis = 1 if legacy_k_sharded else 0
  use_tiles = cfg.routed_w_tiled_layout and (w_local.shape[1] % 8 == 0)
  keep_tiled = cfg.moe_use_forked_3d_gmmv2 and use_tiles
  min_b = cfg.sc_min_collective_bytes
  e_full = (
      (w_local.shape[0] * cfg.fsdp_size) // k_shards
      if flat_k_sharded
      else (
          w_local.shape[0]
          if legacy_k_sharded
          else w_local.shape[0] * cfg.fsdp_size
      )
  )
  full_k = (
      w_local.shape[1] * k_shards
      if flat_k_sharded
      else (
          w_local.shape[1] * cfg.fsdp_size
          if legacy_k_sharded
          else w_local.shape[1]
      )
  )
  amax_in = (
      ops.order_after(absmax_local, start_after)
      if start_after is not None
      else absmax_local
  )
  if k_sharded:
    amax_in = jax.lax.pmax(amax_in, axis_name="fsdp")
  codes_l, scale_l = quantize_moe_codes_from_absmax(w_local, amax_in)
  if use_tiles:
    codes_l = ops.routed_w_to_tiles(codes_l)
  codes_f = ops.sc_all_gather(
      codes_l,
      "fsdp",
      core_id=ag_core,
      axis=gather_axis,
      min_collective_bytes=min_b,
  )
  if flat_k_sharded:
    if use_tiles and keep_tiled:
      codes_f = codes_f.reshape(e_full, full_k // 8, 8, codes_f.shape[-1])
    elif use_tiles:
      codes_f = ops.routed_w_from_tiles(codes_f).reshape(
          e_full, full_k, codes_f.shape[-1]
      )
    else:
      codes_f = codes_f.reshape(e_full, full_k, codes_f.shape[-1])
  elif use_tiles and not keep_tiled:
    codes_f = ops.routed_w_from_tiles(codes_f)
  if scale_l.ndim == 2:
    scale_f = scale_l
  elif flat_k_sharded:
    scale_f = jnp.broadcast_to(scale_l, (e_full, 1, scale_l.shape[-1]))
  elif legacy_k_sharded:
    scale_f = scale_l
  else:
    scale_f = ops.sc_all_gather(
        scale_l, "fsdp", core_id=ag_core, axis=0, min_collective_bytes=min_b
    )
  return codes_f, scale_f


def prequantize_routed_weight_local(
    w_local: jax.Array,
    absmax_local: jax.Array,
    cfg: QwenConfig,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes local FSDP-sharded `w_local` to `float8_e4m3fn` `(codes_l, scale_l)`."""
  amax_in = (
      jax.lax.pmax(absmax_local, axis_name="fsdp")
      if _k_sharded_across_fsdp(cfg)
      else absmax_local
  )
  codes_l, scale_l = quantize_moe_codes_from_absmax(w_local, amax_in)
  if cfg.routed_w_tiled_layout and (w_local.shape[1] % 8 == 0):
    codes_l = ops.routed_w_to_tiles(codes_l)
  return codes_l, scale_l


def gather_prequantized_routed_weight_bwd(
    codes_l: jax.Array,
    scale_l: jax.Array,
    cfg: QwenConfig,
    *,
    raw_k_dim: int,
    start_after: Any = None,
) -> tuple[jax.Array, jax.Array]:
  """Gathers pre-quantized local `(codes_l, scale_l)` on SC1 in backward."""
  ag_core = ops._fsdp_ag_core_id(cfg, is_bwd_remat=False)  # pylint: disable=protected-access
  k_shards = cfg.moe_k_shards
  flat_k_sharded = k_shards > 1
  legacy_k_sharded = k_shards == 0
  gather_axis = 1 if legacy_k_sharded else 0
  use_tiles = cfg.routed_w_tiled_layout and (codes_l.ndim == 4)
  keep_tiled = (
      cfg.moe_use_forked_3d_gmmv2 and use_tiles and (raw_k_dim % 8 == 0)
  )
  min_b = cfg.sc_min_collective_bytes
  e_full = (
      (codes_l.shape[0] * cfg.fsdp_size) // k_shards
      if flat_k_sharded
      else (
          codes_l.shape[0]
          if legacy_k_sharded
          else codes_l.shape[0] * cfg.fsdp_size
      )
  )
  c_in = (
      ops.order_after(codes_l, start_after)
      if start_after is not None
      else codes_l
  )
  codes_f = ops.sc_all_gather(
      c_in,
      "fsdp",
      core_id=ag_core,
      axis=gather_axis,
      min_collective_bytes=min_b,
  )
  if flat_k_sharded:
    if use_tiles and keep_tiled:
      codes_f = codes_f.reshape(e_full, raw_k_dim // 8, 8, codes_f.shape[-1])
    elif use_tiles:
      codes_f = ops.routed_w_from_tiles(codes_f).reshape(
          e_full, raw_k_dim, codes_f.shape[-1]
      )
    else:
      codes_f = codes_f.reshape(e_full, raw_k_dim, codes_f.shape[-1])
  elif use_tiles and not keep_tiled:
    codes_f = ops.routed_w_from_tiles(codes_f)
  if scale_l.ndim == 2:
    scale_f = scale_l
  elif flat_k_sharded:
    scale_f = jnp.broadcast_to(scale_l, (e_full, 1, scale_l.shape[-1]))
  elif legacy_k_sharded:
    scale_f = scale_l
  else:
    s_in = (
        ops.order_after(scale_l, start_after)
        if start_after is not None
        else scale_l
    )
    scale_f = ops.sc_all_gather(
        s_in, "fsdp", core_id=ag_core, axis=0, min_collective_bytes=min_b
    )
  return codes_f, scale_f


def produce_prev_bwd_rw3(
    tok_ep: Any,
    carry_pref_in: ops.BwdRoutedWeightPrefetch,
    prev_rw3_local: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    cfg: QwenConfig,
):
  """Prefetches `attn_3[sb_idx - 1]` on SC1 in backward without closing over tracers."""

  @jax.custom_vjp
  def _produce(
      tok_in: Any,
      pref_in: ops.BwdRoutedWeightPrefetch,
      wi_c_l: jax.Array,
      wi_s_l: jax.Array,
      wo_c_l: jax.Array,
      wo_s_l: jax.Array,
  ):
    del pref_in, wi_c_l, wi_s_l, wo_c_l, wo_s_l
    return tok_in

  def _produce_fwd(
      tok_in: Any,
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
    anchor = (
        g_tok_ep.ste_carrier
        if isinstance(g_tok_ep, ops.Fp8DispatchedTokens)
        else g_tok_ep
    )
    wi_c, wi_s = gather_prequantized_routed_weight_bwd(
        wi_c_l, wi_s_l, cfg, raw_k_dim=cfg.emb_dim, start_after=anchor
    )
    wo_c, wo_s = gather_prequantized_routed_weight_bwd(
        wo_c_l, wo_s_l, cfg, raw_k_dim=cfg.moe_mlp_dim, start_after=anchor
    )
    pref = ops.BwdRoutedWeightPrefetch(
        wi_codes=wi_c, wi_scale=wi_s, wo_codes=wo_c, wo_scale=wo_s
    )
    return g_tok_ep, pref, None, None, None, None

  _produce.defvjp(_produce_fwd, _produce_bwd)
  wi_c_l, wi_s_l, wo_c_l, wo_s_l = prev_rw3_local
  return _produce(tok_ep, carry_pref_in, wi_c_l, wi_s_l, wo_c_l, wo_s_l)


def attach_last_bwd_rw3(
    x_out: jax.Array,
    final_pref_slot: ops.BwdRoutedWeightPrefetch,
    last_rw3_local: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    cfg: QwenConfig,
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
    wi_c, wi_s = gather_prequantized_routed_weight_bwd(
        wi_c_l, wi_s_l, cfg, raw_k_dim=cfg.emb_dim, start_after=g_x
    )
    wo_c, wo_s = gather_prequantized_routed_weight_bwd(
        wo_c_l, wo_s_l, cfg, raw_k_dim=cfg.moe_mlp_dim, start_after=g_x
    )
    pref = ops.BwdRoutedWeightPrefetch(
        wi_codes=wi_c, wi_scale=wi_s, wo_codes=wo_c, wo_scale=wo_s
    )
    return g_x, pref, None, None, None, None

  _attach.defvjp(_attach_fwd, _attach_bwd)
  wi_c_l, wi_s_l, wo_c_l, wo_s_l = last_rw3_local
  return _attach(x_out, final_pref_slot, wi_c_l, wi_s_l, wo_c_l, wo_s_l)


# ---------------------------------------------------------------------------
# Pure-JAX SwiGLU up-projection (`wi`) and fused dSwiGLU down-projection (`wo`)
# ---------------------------------------------------------------------------


def pre_act_tile_n(tile_fwd: tuple[int, int, int], two_n: int) -> int:
  """Computes per-half `tile_n` matching `gmm_v2(..., fuse_act='silu')`."""
  half_n = two_n // 2
  cand = tile_fwd[2] // 2
  return min(max(cand, 128), half_n)


def wi_gmm_silu(
    lhs: jax.Array,
    wi: jax.Array | Fp8SteWeight,
    group_sizes_local: jax.Array,
    cfg: QwenConfig,
    *,
    return_rhs: bool = False,
) -> tuple[jax.Array, ...]:
  """Fused `wi` SwiGLU up-projection returning `(act, gate_up)` or `(act, gate_up, wi)`."""
  assert_global_mixed_precision_contract(cfg)
  use_pallas = cfg.use_pallas_kernels
  m_rows = lhs.shape[0]
  k_dim, two_f = wi.shape[1], wi.shape[2]
  f_dim = two_f // 2
  tile_fwd = _clamp_triple_tiles(cfg.wi_tile_fwd, m_rows, k_dim, two_f)
  tile_dlhs = _clamp_triple_tiles(cfg.wi_tile_dlhs, m_rows, two_f, k_dim)
  tile_drhs = _clamp_triple_tiles(cfg.wi_tile_drhs, m_rows, k_dim, two_f)
  rhs_buf_cnt = cfg.wi_fwd_rhs_buffer_count

  def _fwd_impl(x: jax.Array, w: jax.Array | Fp8SteWeight, gs: jax.Array):
    is_ste = isinstance(w, Fp8SteWeight)
    fp8_compute = is_ste and cfg.moe_fp8_gmm_compute
    if is_ste and not fp8_compute:
      raise ValueError(
          "Fp8SteWeight requires cfg.moe_fp8_gmm_compute=True; W8A16 software"
          " dequant is forbidden."
      )
    if use_pallas:
      tn = pre_act_tile_n(tile_fwd, w.shape[-1])
      t_info = gmm_v2.TileSizes(
          tile_m=tile_fwd[0], tile_k=tile_fwd[1], tile_n=tn
      )
      extra: dict[str, Any] = {}
      if rhs_buf_cnt is not None:
        extra["rhs_buffer_count"] = rhs_buf_cnt
      if is_ste:
        extra["rhs_scale"] = _prepare_gmm_rhs_scale(w.scale, w.shape[2])
        extra["dequant_rhs_in_vmem"] = False
      act, gate_up = gmm_v2.gmm_v2(
          lhs=x,
          rhs=w.codes if is_ste else w,
          group_sizes=gs,
          tile_info=t_info,
          preferred_element_type=jnp.bfloat16,
          group_offset=jnp.int32(0),
          maybe_quantize_lhs=True,
          fuse_act="silu",
          return_pre_act="concat",
          **extra,
      )
    else:
      w_bf16 = (
          (w.codes.astype(jnp.float32) * w.scale).astype(w.dtype)
          if is_ste
          else w
      )
      x_2d = x.reshape(x.shape[0], -1)
      if fp8_compute:
        x_q, x_s = ops.quantize_rows_fp8(x_2d)
        x_2d = ops.dequantize_rows_fp8(x_q, x_s, x.dtype)
      gate_up = jax.lax.ragged_dot_general(
          x_2d,
          w_bf16,
          gs,
          _FWD_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(x.dtype)
      gate, up = gate_up[:, :f_dim], gate_up[:, f_dim:]
      act = (
          jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
      ).astype(x.dtype)

    w_res = (
        dataclasses.replace(w, w_ste=None) if isinstance(w, Fp8SteWeight) else w
    )
    return act, gate_up, w_res

  def _bwd_impl(
      res, d_gate_up: jax.Array, ps_carrier: jax.Array | Fp8SteWeight | None
  ):
    x, w_res, gs = res
    _assert_bf16_plus_dtype(w_res.dtype, "wi_gmm_silu_wgrad")
    dout = d_gate_up.astype(jnp.bfloat16) if use_pallas else d_gate_up
    is_ste = isinstance(w_res, Fp8SteWeight)
    fp8_compute = is_ste and cfg.moe_fp8_gmm_compute
    if is_ste and not fp8_compute:
      raise ValueError(
          "Fp8SteWeight requires cfg.moe_fp8_gmm_compute=True; W8A16 software"
          " dequant is forbidden."
      )
    ps = (
        ps_carrier.w_ste if isinstance(ps_carrier, Fp8SteWeight) else ps_carrier
    )
    if ps is not None:
      ops.assert_bf16_or_fp32_reduction(ps, "wi_gmm_silu_partial_sum")

    if use_pallas:
      if fp8_compute:
        lhs_q, lhs_s = ops.quantize_rows_fp8(x)
        dout_scaled = dout.astype(jnp.float32) * lhs_s.reshape(
            -1, *([1] * (dout.ndim - 1))
        )
        rhs_q, rhs_s = ops.quantize_cols_fp8_for_tgmm(dout_scaled)
        drhs_t = _clamp_triple_tiles(cfg.wi_tile_drhs_fp8, m_rows, k_dim, two_f)
        tn_drhs = min(drhs_t[2], 1024) if ps is not None else drhs_t[2]
        drhs_tiles = gmm_v2.TileSizes(
            tile_m=drhs_t[0], tile_k=drhs_t[1], tile_n=tn_drhs
        )
        dw_raw = tgmm_v2.tgmm_v2(
            lhs=lhs_q,
            rhs=rhs_q,
            group_sizes=gs,
            num_actual_groups=w_res.shape[0],
            rhs_scale=rhs_s,
            partial_sum=ps,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=w_res.dtype,
            group_offset=jnp.int32(0),
            tile_info=drhs_tiles,
        )
      else:
        tn_drhs = min(tile_drhs[2], 2048) if ps is not None else tile_drhs[2]
        drhs_tiles = gmm_v2.TileSizes(
            tile_m=tile_drhs[0], tile_k=tile_drhs[1], tile_n=tn_drhs
        )
        dw_raw = tgmm_v2.tgmm_v2(
            lhs=x,
            rhs=dout,
            group_sizes=gs,
            num_actual_groups=w_res.shape[0],
            rhs_scale=None,
            partial_sum=ps,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=w_res.dtype,
            group_offset=jnp.int32(0),
            tile_info=drhs_tiles,
        )
      dlhs_t = (
          _clamp_triple_tiles(cfg.wi_tile_dlhs_fp8, m_rows, two_f, k_dim)
          if fp8_compute
          else tile_dlhs
      )
      dlhs_tiles = gmm_v2.TileSizes(
          tile_m=dlhs_t[0], tile_k=dlhs_t[1], tile_n=dlhs_t[2]
      )
      rhs_s_dlhs = (
          _prepare_gmm_rhs_scale(w_res.scale, w_res.shape[1])
          if is_ste
          else None
      )
      dx = gmm_v2.gmm_v2(
          lhs=dout,
          rhs=w_res.codes if is_ste else w_res,
          group_sizes=gs,
          rhs_scale=rhs_s_dlhs,
          partial_sum=None,
          tile_info=dlhs_tiles,
          preferred_element_type=x.dtype,
          group_offset=jnp.int32(0),
          transpose_rhs=True,
          maybe_quantize_lhs=True,
          zero_initialize=True,
          out_is_3d=(x.ndim == 3),
          dequant_rhs_in_vmem=False,
      )
    else:
      w_bf16 = (
          (w_res.codes.astype(jnp.float32) * w_res.scale).astype(w_res.dtype)
          if is_ste
          else w_res
      )
      x_2d = x.reshape(x.shape[0], -1)
      if fp8_compute:
        lhs_q, lhs_s = ops.quantize_rows_fp8(x_2d)
        rhs_q, rhs_s = ops.quantize_cols_fp8_for_tgmm(
            dout.astype(jnp.float32) * lhs_s
        )
        dw_raw = (
            jax.lax.ragged_dot_general(
                lhs_q.astype(jnp.float32),
                rhs_q.astype(jnp.float32),
                gs,
                _DRHS_RAGGED_DIM_NUMS,
                preferred_element_type=jnp.float32,
            )
            * rhs_s
        ).astype(w_res.dtype)
        dout_q, dout_s = ops.quantize_rows_fp8(dout)
        dout_in = ops.dequantize_rows_fp8(dout_q, dout_s, dout.dtype)
      else:
        dw_raw = jax.lax.ragged_dot_general(
            x_2d,
            dout,
            gs,
            _DRHS_RAGGED_DIM_NUMS,
            preferred_element_type=jnp.float32,
        ).astype(w_res.dtype)
        dout_in = dout
      if ps is not None:
        dw_raw = dw_raw + ps
      dx_2d = jax.lax.ragged_dot_general(
          dout_in,
          w_bf16,
          gs,
          _DLHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(x.dtype)
      dx = dx_2d.reshape(x.shape) if x.ndim == 3 else dx_2d

    ops.assert_bf16_or_fp32_reduction(dw_raw, "wi_gmm_silu_dw_out")
    if is_ste:
      dw: Any = Fp8SteWeight(
          w_ste=dw_raw,
          codes=jnp.zeros_like(w_res.codes),
          scale=jnp.zeros_like(w_res.scale),
      )
    else:
      dw = dw_raw
    return dx, dw

  if return_rhs:

    @jax.custom_vjp
    def _op_ret(x, w, gs):
      act, gate_up, _ = _fwd_impl(x, w, gs)
      return act, gate_up, w

    def _op_ret_fwd(x, w, gs):
      act, gate_up, w_res = _fwd_impl(x, w, gs)
      return (act, gate_up, w), (x, w_res, gs)

    def _op_ret_bwd(res, grads):
      _, d_gate_up, g_w_ps = grads
      dx, dw = _bwd_impl(res, d_gate_up, g_w_ps)
      return dx, dw, None

    _op_ret.defvjp(_op_ret_fwd, _op_ret_bwd)
    return _op_ret(lhs, wi, group_sizes_local)

  @jax.custom_vjp
  def _op(x, w, gs):
    act, gate_up, _ = _fwd_impl(x, w, gs)
    return act, gate_up

  def _op_fwd(x, w, gs):
    act, gate_up, w_res = _fwd_impl(x, w, gs)
    return (act, gate_up), (x, w_res, gs)

  def _op_bwd(res, grads):
    _, d_gate_up = grads
    dx, dw = _bwd_impl(res, d_gate_up, None)
    return dx, dw, None

  _op.defvjp(_op_fwd, _op_bwd)
  return _op(lhs, wi, group_sizes_local)


def _dswiglu_and_dot_from_u(
    u: jax.Array,
    gate_up: jax.Array,
    row_scale: jax.Array,
    *,
    dact_tn: int | None,
    interleaved_gate_up: bool,
) -> tuple[jax.Array, jax.Array]:
  """Computes `(d_gate_up, dot)` in fp32 de-interleaving `gate_up`."""
  if interleaved_gate_up:
    gate, up = gmm_v2.split_dact_gate_up(gate_up, dact_tn)
  else:
    half = gate_up.shape[-1] // 2
    gate, up = gate_up[:, :half], gate_up[:, half:]
  u_f32 = u.astype(jnp.float32)
  g_f32 = gate.astype(jnp.float32)
  up_f32 = up.astype(jnp.float32)
  sig_g = jax.nn.sigmoid(g_f32)
  silu_g = g_f32 * sig_g
  h_f32 = silu_g * up_f32
  dot = jnp.sum(u_f32 * h_f32, axis=-1)
  rs = row_scale.astype(jnp.float32).reshape(-1, 1)
  dh = u_f32 * rs
  dsilu_dg = sig_g * (1.0 + g_f32 * (1.0 - sig_g))
  d_gate = dh * up_f32 * dsilu_dg
  d_up = dh * silu_g
  d_gate_up = jnp.concatenate([d_gate, d_up], axis=-1).astype(gate_up.dtype)
  return d_gate_up, dot


def make_wo_callbacks(cfg: QwenConfig):
  """Returns `(down_projection, down_projection_bwd_dswiglu)` callbacks."""
  assert_global_mixed_precision_contract(cfg)
  use_pallas = cfg.use_pallas_kernels
  dact_tn = pre_act_tile_n(cfg.wi_tile_fwd, 2 * cfg.moe_mlp_dim)

  def down_projection(
      h: jax.Array, wo: jax.Array | Fp8SteWeight, gs_local: jax.Array
  ) -> jax.Array:
    is_ste = isinstance(wo, Fp8SteWeight)
    fp8_compute = is_ste and cfg.moe_fp8_gmm_compute
    if is_ste and not fp8_compute:
      raise ValueError(
          "Fp8SteWeight requires cfg.moe_fp8_gmm_compute=True; W8A16 software"
          " dequant is forbidden."
      )
    if use_pallas:
      tf = _clamp_triple_tiles(
          cfg.wo_tile_fwd, h.shape[0], wo.shape[1], wo.shape[2]
      )
      fwd_tiles = gmm_v2.TileSizes(tile_m=tf[0], tile_k=tf[1], tile_n=tf[2])
      return gmm_v2.gmm_v2(
          lhs=h,
          rhs=wo.codes if is_ste else wo,
          group_sizes=gs_local,
          rhs_scale=_prepare_gmm_rhs_scale(wo.scale, wo.shape[2])
          if is_ste
          else None,
          tile_info=fwd_tiles,
          preferred_element_type=jnp.bfloat16,
          partial_sum=None,
          group_offset=jnp.int32(0),
          lhs_scale=None,
          maybe_quantize_lhs=True,
          out_is_3d=False,
          dequant_rhs_in_vmem=False,
      )
    w_bf16 = (
        (wo.codes.astype(jnp.float32) * wo.scale).astype(wo.dtype)
        if is_ste
        else wo
    )
    h_in = (
        ops.dequantize_rows_fp8(*ops.quantize_rows_fp8(h), h.dtype)
        if fp8_compute
        else h
    )
    return jax.lax.ragged_dot_general(
        h_in,
        w_bf16,
        gs_local,
        _FWD_RAGGED_DIM_NUMS,
        preferred_element_type=jnp.float32,
    ).astype(h.dtype)

  def down_projection_bwd_dswiglu(
      gathered_g: jax.Array,
      weighted_h: jax.Array,
      wo: jax.Array | Fp8SteWeight,
      gs_local: jax.Array,
      *,
      dact_gate_up: jax.Array,
      dact_row_scale: jax.Array,
      drhs_partial_sum: jax.Array | Fp8SteWeight | None = None,
  ):
    _assert_bf16_plus_dtype(wo.dtype, "wo_bwd_wgrad")
    is_ste = isinstance(wo, Fp8SteWeight)
    fp8_compute = is_ste and cfg.moe_fp8_gmm_compute
    if is_ste and not fp8_compute:
      raise ValueError(
          "Fp8SteWeight requires cfg.moe_fp8_gmm_compute=True; W8A16 software"
          " dequant is forbidden."
      )
    ps = (
        drhs_partial_sum.w_ste
        if isinstance(drhs_partial_sum, Fp8SteWeight)
        else drhs_partial_sum
    )
    if ps is not None:
      ops.assert_bf16_or_fp32_reduction(ps, "wo_bwd_partial_sum")
    m_rows, k_dim, n_dim = weighted_h.shape[0], wo.shape[1], wo.shape[2]

    if use_pallas:
      if fp8_compute:
        lhs_q, lhs_s = ops.quantize_rows_fp8(weighted_h)
        g_scaled = gathered_g.astype(jnp.float32) * lhs_s.reshape(
            -1, *([1] * (gathered_g.ndim - 1))
        )
        rhs_q, rhs_s = ops.quantize_cols_fp8_for_tgmm(g_scaled)
        tdr = _clamp_triple_tiles(cfg.wo_tile_drhs_fp8, m_rows, k_dim, n_dim)
        tk_drhs = min(tdr[1], 1024) if ps is not None else tdr[1]
        drhs_tiles = gmm_v2.TileSizes(
            tile_m=tdr[0], tile_k=tk_drhs, tile_n=tdr[2]
        )
        dwo_raw = tgmm_v2.tgmm_v2(
            lhs=lhs_q,
            rhs=rhs_q,
            group_sizes=gs_local,
            num_actual_groups=wo.shape[0],
            rhs_scale=rhs_s,
            partial_sum=ps,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=wo.dtype,
            group_offset=jnp.int32(0),
            tile_info=drhs_tiles,
        )
        tdl = _clamp_triple_tiles(cfg.wo_tile_dlhs_fp8, m_rows, n_dim, k_dim)
        dlhs_tiles = gmm_v2.TileSizes(*tdl)
        u = gmm_v2.gmm_v2(
            lhs=gathered_g,
            rhs=wo.codes,
            group_sizes=gs_local,
            rhs_scale=_prepare_gmm_rhs_scale(wo.scale, wo.shape[1]),
            partial_sum=None,
            tile_info=dlhs_tiles,
            preferred_element_type=jnp.bfloat16,
            group_offset=jnp.int32(0),
            lhs_scale=None,
            transpose_rhs=True,
            maybe_quantize_lhs=True,
            zero_initialize=True,
            out_is_3d=False,
            dequant_rhs_in_vmem=False,
        )
        d_gate_up, dot = _dswiglu_and_dot_from_u(
            u,
            dact_gate_up,
            dact_row_scale,
            dact_tn=dact_tn,
            interleaved_gate_up=True,
        )
      else:
        tdr = _clamp_triple_tiles(cfg.wo_tile_drhs, m_rows, k_dim, n_dim)
        tn_drhs = min(tdr[2], 2048) if ps is not None else tdr[2]
        drhs_tiles = gmm_v2.TileSizes(
            tile_m=tdr[0], tile_k=tdr[1], tile_n=tn_drhs
        )
        dwo_raw = tgmm_v2.tgmm_v2(
            lhs=weighted_h,
            rhs=gathered_g,
            group_sizes=gs_local,
            num_actual_groups=wo.shape[0],
            rhs_scale=None,
            partial_sum=ps,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=wo.dtype,
            group_offset=jnp.int32(0),
            tile_info=drhs_tiles,
        )
        tdl = _clamp_triple_tiles(cfg.wo_tile_dlhs, m_rows, n_dim, k_dim)
        dlhs_tiles = gmm_v2.TileSizes(
            tile_m=tdl[0], tile_k=tdl[1], tile_n=tdl[2]
        )
        d_gate_up, dot = gmm_v2.gmm_v2(
            lhs=gathered_g,
            rhs=wo,
            group_sizes=gs_local,
            rhs_scale=None,
            partial_sum=None,
            tile_info=dlhs_tiles,
            preferred_element_type=jnp.bfloat16,
            group_offset=jnp.int32(0),
            lhs_scale=None,
            transpose_rhs=True,
            maybe_quantize_lhs=True,
            zero_initialize=True,
            out_is_3d=False,
            fuse_dact="swiglu",
            dact_gate_up=dact_gate_up,
            dact_row_scale=dact_row_scale,
            return_dact_dot=True,
            dact_pre_act_tile_n=dact_tn,
            dequant_rhs_in_vmem=False,
        )
    else:
      w_bf16 = (
          (wo.codes.astype(jnp.float32) * wo.scale).astype(wo.dtype)
          if is_ste
          else wo
      )
      g_2d = gathered_g.reshape(gathered_g.shape[0], -1)
      if fp8_compute:
        lhs_q, lhs_s = ops.quantize_rows_fp8(weighted_h)
        rhs_q, rhs_s = ops.quantize_cols_fp8_for_tgmm(
            g_2d.astype(jnp.float32) * lhs_s
        )
        dwo_raw = (
            jax.lax.ragged_dot_general(
                lhs_q.astype(jnp.float32),
                rhs_q.astype(jnp.float32),
                gs_local,
                _DRHS_RAGGED_DIM_NUMS,
                preferred_element_type=jnp.float32,
            )
            * rhs_s
        ).astype(wo.dtype)
        g_in = ops.dequantize_rows_fp8(*ops.quantize_rows_fp8(g_2d), g_2d.dtype)
      else:
        dwo_raw = jax.lax.ragged_dot_general(
            weighted_h,
            g_2d,
            gs_local,
            _DRHS_RAGGED_DIM_NUMS,
            preferred_element_type=jnp.float32,
        ).astype(wo.dtype)
        g_in = g_2d
      if ps is not None:
        dwo_raw = dwo_raw + ps
      u = jax.lax.ragged_dot_general(
          g_in,
          w_bf16,
          gs_local,
          _DLHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      )
      d_gate_up, dot = _dswiglu_and_dot_from_u(
          u,
          dact_gate_up,
          dact_row_scale,
          dact_tn=None,
          interleaved_gate_up=False,
      )

    ops.assert_bf16_or_fp32_reduction(dwo_raw, "wo_bwd_dwo_out")
    if is_ste:
      dwo: Any = Fp8SteWeight(
          w_ste=dwo_raw,
          codes=jnp.zeros_like(wo.codes),
          scale=jnp.zeros_like(wo.scale),
      )
    else:
      dwo = dwo_raw
    return d_gate_up, dot, dwo

  return down_projection, down_projection_bwd_dswiglu


def wo_weighted_gmm(
    act_sg: jax.Array,
    gate_up: jax.Array,
    w_rows: jax.Array,
    wo: jax.Array | Fp8SteWeight,
    gs_local: jax.Array,
    cfg: QwenConfig,
    *,
    return_rhs: bool = False,
    out_is_3d: bool = False,
    forward_tie: jax.Array | None = None,
):
  """Pure-JAX weighted `wo` GMM for `ring_ragged_unsort_tc(..., prescaled=True)`."""
  assert_global_mixed_precision_contract(cfg)
  use_pallas = cfg.use_pallas_kernels
  dact_tn = pre_act_tile_n(cfg.wi_tile_fwd, 2 * cfg.moe_mlp_dim)
  _, down_proj_bwd_fn = make_wo_callbacks(cfg)

  def _fwd_impl(
      act_in: jax.Array,
      gate_up_in: jax.Array,
      w_rows_in: jax.Array,
      wo_in: jax.Array | Fp8SteWeight,
      gs_in: jax.Array,
  ):
    del gate_up_in
    weighted_h = (
        act_in.astype(jnp.float32) * w_rows_in.astype(jnp.float32)[:, None]
    ).astype(act_in.dtype)
    if forward_tie is not None:
      weighted_h = ops.order_after(weighted_h, forward_tie)
    is_ste = isinstance(wo_in, Fp8SteWeight)
    fp8_compute = is_ste and cfg.moe_fp8_gmm_compute
    if is_ste and not fp8_compute:
      raise ValueError(
          "Fp8SteWeight requires cfg.moe_fp8_gmm_compute=True; W8A16 software"
          " dequant is forbidden."
      )
    if use_pallas:
      tf = _clamp_triple_tiles(
          cfg.wo_tile_fwd, weighted_h.shape[0], wo_in.shape[1], wo_in.shape[2]
      )
      fwd_tiles = gmm_v2.TileSizes(tile_m=tf[0], tile_k=tf[1], tile_n=tf[2])
      out = gmm_v2.gmm_v2(
          lhs=weighted_h,
          rhs=wo_in.codes if is_ste else wo_in,
          group_sizes=gs_in,
          rhs_scale=_prepare_gmm_rhs_scale(wo_in.scale, wo_in.shape[2])
          if is_ste
          else None,
          tile_info=fwd_tiles,
          preferred_element_type=jnp.bfloat16,
          partial_sum=None,
          group_offset=jnp.int32(0),
          lhs_scale=None,
          maybe_quantize_lhs=True,
          out_is_3d=out_is_3d,
          dequant_rhs_in_vmem=False,
      )
    else:
      w_bf16 = (
          (wo_in.codes.astype(jnp.float32) * wo_in.scale).astype(wo_in.dtype)
          if is_ste
          else wo_in
      )
      if fp8_compute:
        h_in = ops.dequantize_rows_fp8(
            *ops.quantize_rows_fp8(weighted_h), weighted_h.dtype
        )
      else:
        h_in = weighted_h
      out_2d = jax.lax.ragged_dot_general(
          h_in,
          w_bf16,
          gs_in,
          _FWD_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(act_in.dtype)
      out = (
          out_2d.reshape(out_2d.shape[0], out_2d.shape[1] // 128, 128)
          if out_is_3d
          else out_2d
      )

    wo_res = (
        dataclasses.replace(wo_in, w_ste=None)
        if isinstance(wo_in, Fp8SteWeight)
        else wo_in
    )
    return out, wo_res

  def _bwd_impl(
      res,
      gathered_g: jax.Array,
      ps_carrier: jax.Array | Fp8SteWeight | None,
  ):
    gate_up_in, w_rows_in, wo_res, gs_in = res
    g_in = gathered_g.astype(jnp.bfloat16) if use_pallas else gathered_g
    if use_pallas:
      gate, up = gmm_v2.split_dact_gate_up(gate_up_in, dact_tn)
    else:
      half = gate_up_in.shape[-1] // 2
      gate, up = gate_up_in[:, :half], gate_up_in[:, half:]
    h_f32 = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    weighted_h = (h_f32 * w_rows_in.astype(jnp.float32)[:, None]).astype(
        gate_up_in.dtype
    )

    d_gate_up, dw_rows, dwo = down_proj_bwd_fn(
        g_in,
        weighted_h,
        wo_res,
        gs_in,
        dact_gate_up=gate_up_in,
        dact_row_scale=w_rows_in,
        drhs_partial_sum=ps_carrier,
    )
    zero_dact = jax.lax.broadcast(
        jnp.zeros((), dtype=gate_up_in.dtype),
        (gate_up_in.shape[0], gate_up_in.shape[1] // 2),
    )
    return zero_dact, d_gate_up, dw_rows.astype(w_rows_in.dtype), dwo, None

  if return_rhs:

    @jax.custom_vjp
    def _op_ret(a, gu, wr, w, gs):
      out, _ = _fwd_impl(a, gu, wr, w, gs)
      return out, w

    def _op_ret_fwd(a, gu, wr, w, gs):
      out, w_res = _fwd_impl(a, gu, wr, w, gs)
      return (out, w), (gu, wr, w_res, gs)

    def _op_ret_bwd(res, grads):
      g_out, g_w_ps = grads
      return _bwd_impl(res, g_out, g_w_ps)

    _op_ret.defvjp(_op_ret_fwd, _op_ret_bwd)
    return _op_ret(act_sg, gate_up, w_rows, wo, gs_local)

  @jax.custom_vjp
  def _op(a, gu, wr, w, gs):
    out, _ = _fwd_impl(a, gu, wr, w, gs)
    return out

  def _op_fwd(a, gu, wr, w, gs):
    out, w_res = _fwd_impl(a, gu, wr, w, gs)
    return out, (gu, wr, w_res, gs)

  def _op_bwd(res, g_out):
    return _bwd_impl(res, g_out, None)

  _op.defvjp(_op_fwd, _op_bwd)
  return _op(act_sg, gate_up, w_rows, wo, gs_local)


def routed_moe_3d_fp8_silu(
    disp: ops.Fp8DispatchedTokens,
    w_rows: jax.Array,
    wi: Fp8SteWeight,
    wo: Fp8SteWeight,
    local_gs: jax.Array,
    token_ids_sorted: jax.Array,
    count: jax.Array,
    reduce_meta: Any,
    cfg: QwenConfig,
    *,
    buf_size: int,
    return_rhs: bool = False,
    forward_tie: jax.Array | None = None,
):
  """End-to-end 3D FP8 + Ragged SiLU MoE pipeline (non-forked fallback)."""
  assert_global_mixed_precision_contract(cfg)
  use_pallas = cfg.use_pallas_kernels
  num_tokens_ep = disp.codes_ep.shape[0]
  ep_tokens_shape = disp.codes_ep.shape
  ste_dtype = _assert_bf16_plus_dtype(
      disp.ste_carrier.dtype, "routed_moe_3d_fp8_silu_dx"
  )
  wi_codes_shape = wi.codes.shape
  wi_codes_dtype = wi.codes.dtype
  wi_ste_dtype = _assert_bf16_plus_dtype(wi.dtype, "routed_moe_3d_fp8_silu_dwi")
  wi_scale_shape = wi.scale.shape
  wo_codes_shape = wo.codes.shape
  wo_codes_dtype = wo.codes.dtype
  wo_ste_dtype = _assert_bf16_plus_dtype(wo.dtype, "routed_moe_3d_fp8_silu_dwo")
  wo_scale_shape = wo.scale.shape

  def _fwd_impl(
      disp_in: ops.Fp8DispatchedTokens,
      w_rows_in: jax.Array,
      wi_in: Fp8SteWeight,
      wo_in: Fp8SteWeight,
      gs_in: jax.Array,
      tok_idx_in: jax.Array,
      cnt_in: jax.Array,
      rmeta_in: Any,
  ):
    scale_tie_i32 = jax.lax.stop_gradient(
        (ops.first_scalar(disp_in.scale) * 0.0).astype(jnp.int32)
    )
    if use_pallas:
      sorted_fp8 = tc.ragged_gather_tc(
          disp_in.codes_ep,
          tok_idx_in,
          start=scale_tie_i32,
          num_tokens=cnt_in,
          max_out_tokens=buf_size,
          block_size=cfg.tc_gather_block_size,
      )
      wi_t = (
          min(cfg.wi_tile_fwd_3d_fp8[0], buf_size),
          _clamp_3d_k_tile(cfg.wi_tile_fwd_3d_fp8[1], cfg.emb_dim),
          _clamp_mult128_tile(
              cfg.wi_tile_fwd_3d_fp8[2], 2 * cfg.moe_mlp_dim, min_tile=128
          ),
      )
      gate_up = gmm_v2.gmm_v2(
          lhs=sorted_fp8,
          rhs=wi_in.codes,
          group_sizes=gs_in,
          lhs_scale=disp_in.scale,
          rhs_scale=_prepare_gmm_rhs_scale(wi_in.scale, wi_in.shape[2]),
          tile_info=gmm_v2.TileSizes(*wi_t),
          preferred_element_type=jnp.bfloat16,
          group_offset=jnp.int32(0),
          maybe_quantize_lhs=False,
          zero_initialize=False,
          out_is_3d=False,
          rhs_buffer_count=2,
          dequant_rhs_in_vmem=False,
      )
      act_bf16 = ragged_ops.ragged_silu_mul_fwd(
          gate_up,
          0,
          w_rows_in[:, None],
          cnt_in,
          block_size_tokens=math.gcd(1024, buf_size),
          block_size_hidden=math.gcd(2048, cfg.moe_mlp_dim),
          use_pallas=True,
      )
      act_fp8, s_act = ragged_ops.ragged_absmax_quantize(
          act_bf16,
          jnp.float8_e4m3fn,
          cnt_in,
          use_pallas=True,
      )
      if forward_tie is not None:
        s_act = ops.order_after(s_act, forward_tie)
      wo_t = _clamp_triple_tiles(
          cfg.wo_tile_fwd_3d_fp8, buf_size, cfg.moe_mlp_dim, cfg.emb_dim
      )
      wo_t = (
          wo_t[0],
          wo_t[1],
          _clamp_3d_k_tile(wo_t[2], cfg.emb_dim),
      )
      weighted_down_3d = gmm_v2.gmm_v2(
          lhs=act_fp8,
          rhs=wo_in.codes,
          group_sizes=gs_in,
          lhs_scale=s_act,
          rhs_scale=_prepare_gmm_rhs_scale(wo_in.scale, wo_in.shape[2]),
          tile_info=gmm_v2.TileSizes(*wo_t),
          preferred_element_type=jnp.bfloat16,
          group_offset=jnp.int32(0),
          maybe_quantize_lhs=False,
          zero_initialize=False,
          out_is_3d=True,
          rhs_buffer_count=2,
          dequant_rhs_in_vmem=False,
      )
      partial_ep = tc.ragged_gather_reduce_tc(
          weighted_down_3d,
          rmeta_in,
          num_out_tokens=num_tokens_ep,
          top_k=cfg.num_experts_per_tok,
      )
    else:
      valid = jnp.arange(buf_size, dtype=jnp.int32) < cnt_in
      safe_src = jnp.where(valid, tok_idx_in + scale_tie_i32, 0)
      sorted_fp8 = jnp.where(
          valid[:, None, None],
          jnp.take(disp_in.codes_ep, safe_src, axis=0),
          jnp.zeros((), dtype=jnp.float8_e4m3fn),
      )
      x_deq_2d = (
          sorted_fp8.reshape(buf_size, -1).astype(jnp.float32) * disp_in.scale
      )
      wi_f32 = wi_in.codes.astype(jnp.float32) * wi_in.scale
      gate_up = jax.lax.ragged_dot_general(
          x_deq_2d,
          wi_f32,
          gs_in,
          _FWD_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(jnp.bfloat16)
      act_bf16 = ragged_ops.ragged_silu_mul_fwd(
          gate_up, 0, w_rows_in[:, None], cnt_in, use_pallas=False
      )
      act_fp8, s_act = ragged_ops.ragged_absmax_quantize(
          act_bf16, jnp.float8_e4m3fn, cnt_in, use_pallas=False
      )
      if forward_tie is not None:
        s_act = ops.order_after(s_act, forward_tie)
      act_deq = act_fp8.astype(jnp.float32) * s_act
      wo_f32 = wo_in.codes.astype(jnp.float32) * wo_in.scale
      out_2d = jax.lax.ragged_dot_general(
          act_deq,
          wo_f32,
          gs_in,
          _FWD_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(ste_dtype)
      weighted_down_3d = out_2d.reshape(buf_size, out_2d.shape[1] // 128, 128)
      safe_dst = jnp.where(valid, tok_idx_in, num_tokens_ep)
      padded_out = jnp.zeros(
          (num_tokens_ep + 1, *weighted_down_3d.shape[1:]), dtype=ste_dtype
      )
      partial_ep = padded_out.at[safe_dst].add(
          jnp.where(valid[:, None, None], weighted_down_3d, 0)
      )[:num_tokens_ep]

    wi_res = dataclasses.replace(wi_in, w_ste=None)
    wo_res = dataclasses.replace(wo_in, w_ste=None)
    res = (
        sorted_fp8,
        disp_in.scale,
        s_act,
        gate_up,
        w_rows_in,
        wi_res,
        wo_res,
        gs_in,
        tok_idx_in,
        cnt_in,
        rmeta_in,
    )
    return partial_ep, res

  def _bwd_impl(
      res,
      g_partial_ep: jax.Array,
      ps_wi_carrier: Fp8SteWeight | None,
      ps_wo_carrier: Fp8SteWeight | None,
  ):
    (
        sorted_fp8,
        disp_scale,
        s_act,
        gate_up,
        w_rows_in,
        wi_res,
        wo_res,
        gs_in,
        tok_idx_in,
        cnt_in,
        rmeta_in,
    ) = res
    ps_wi = (
        ps_wi_carrier.w_ste
        if isinstance(ps_wi_carrier, Fp8SteWeight)
        else ps_wi_carrier
    )
    ps_wo = (
        ps_wo_carrier.w_ste
        if isinstance(ps_wo_carrier, Fp8SteWeight)
        else ps_wo_carrier
    )
    if ps_wi is not None:
      ops.assert_bf16_or_fp32_reduction(ps_wi, "routed_moe_3d_ps_wi")
    if ps_wo is not None:
      ops.assert_bf16_or_fp32_reduction(ps_wo, "routed_moe_3d_ps_wo")

    if use_pallas:
      gathered_g_3d = tc.ragged_gather_tc(
          g_partial_ep.astype(jnp.bfloat16),
          tok_idx_in,
          start=jnp.int32(0),
          num_tokens=cnt_in,
          max_out_tokens=buf_size,
          block_size=cfg.tc_gather_block_size,
      )
      g_fp8, s_g = ragged_ops.ragged_absmax_quantize(
          gathered_g_3d, jnp.float8_e4m3fn, cnt_in, use_pallas=True
      )
      act_fp8_bwd = ragged_ops.ragged_silu_mul_fwd(
          gate_up,
          0,
          w_rows_in[:, None],
          cnt_in,
          block_size_tokens=math.gcd(1024, buf_size),
          block_size_hidden=math.gcd(2048, cfg.moe_mlp_dim),
          out_dtype=jnp.float8_e4m3fn,
          out_scale=s_act,
          use_pallas=True,
      )
      wo_drhs_t = _clamp_triple_tiles(
          cfg.wo_tile_drhs_3d_fp8, buf_size, cfg.moe_mlp_dim, cfg.emb_dim
      )
      wo_drhs_t = (
          wo_drhs_t[0],
          wo_drhs_t[1],
          _clamp_3d_k_tile(wo_drhs_t[2], cfg.emb_dim),
      )
      wo_rhs_s = jnp.broadcast_to(
          (s_act * s_g).reshape(1, 1, 1), (1, 1, wo_res.shape[2])
      )
      dwo_raw = tgmm_v2.tgmm_v2(
          lhs=act_fp8_bwd,
          rhs=g_fp8,
          group_sizes=gs_in,
          num_actual_groups=wo_res.shape[0],
          rhs_scale=wo_rhs_s,
          partial_sum=ps_wo,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=wo_ste_dtype,
          group_offset=jnp.int32(0),
          tile_info=gmm_v2.TileSizes(*wo_drhs_t),
      )
      wo_dlhs_t = (
          min(cfg.wo_tile_dlhs_3d_fp8[0], buf_size),
          _clamp_3d_k_tile(cfg.wo_tile_dlhs_3d_fp8[1], cfg.emb_dim),
          _clamp_mult128_tile(cfg.wo_tile_dlhs_3d_fp8[2], cfg.moe_mlp_dim),
      )
      d_act = gmm_v2.gmm_v2(
          lhs=g_fp8,
          rhs=wo_res.codes,
          group_sizes=gs_in,
          lhs_scale=s_g,
          rhs_scale=_prepare_gmm_rhs_scale(wo_res.scale, wo_res.shape[1]),
          partial_sum=None,
          tile_info=gmm_v2.TileSizes(*wo_dlhs_t),
          preferred_element_type=jnp.bfloat16,
          group_offset=jnp.int32(0),
          transpose_rhs=True,
          maybe_quantize_lhs=False,
          zero_initialize=False,
          out_is_3d=False,
          rhs_buffer_count=2,
          dequant_rhs_in_vmem=False,
      )
      d_gate_up, dw_rows_2d, d_gate_amax = ragged_ops.ragged_silu_mul_bwd(
          d_act,
          gate_up,
          0,
          w_rows_in[:, None],
          cnt_in,
          block_size_tokens=math.gcd(512, buf_size),
          block_size_hidden=math.gcd(1024, cfg.moe_mlp_dim),
          with_absmax=True,
          use_pallas=True,
      )
      s_dgate = ragged_ops.absmax_to_scale(d_gate_amax, jnp.float8_e4m3fn)
      d_gate_fp8 = ragged_ops.ragged_quantize(
          d_gate_up, s_dgate, jnp.float8_e4m3fn, cnt_in, use_pallas=True
      )
      wi_drhs_t = (
          min(cfg.wi_tile_drhs_3d_fp8[0], buf_size),
          _clamp_3d_k_tile(cfg.wi_tile_drhs_3d_fp8[1], cfg.emb_dim),
          _clamp_mult128_tile(cfg.wi_tile_drhs_3d_fp8[2], 2 * cfg.moe_mlp_dim),
      )
      wi_rhs_s = jnp.broadcast_to(
          (disp_scale * s_dgate).reshape(1, 1, 1), (1, 1, wi_res.shape[2])
      )
      dwi_raw = tgmm_v2.tgmm_v2(
          lhs=sorted_fp8,
          rhs=d_gate_fp8,
          group_sizes=gs_in,
          num_actual_groups=wi_res.shape[0],
          rhs_scale=wi_rhs_s,
          partial_sum=ps_wi,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=wi_ste_dtype,
          group_offset=jnp.int32(0),
          tile_info=gmm_v2.TileSizes(*wi_drhs_t),
      )
      wi_dlhs_t = (
          min(cfg.wi_tile_dlhs_3d_fp8[0], buf_size),
          _clamp_mult128_tile(cfg.wi_tile_dlhs_3d_fp8[1], 2 * cfg.moe_mlp_dim),
          _clamp_3d_k_tile(cfg.wi_tile_dlhs_3d_fp8[2], cfg.emb_dim),
      )
      dx_sorted_3d = gmm_v2.gmm_v2(
          lhs=d_gate_fp8,
          rhs=wi_res.codes,
          group_sizes=gs_in,
          lhs_scale=s_dgate,
          rhs_scale=_prepare_gmm_rhs_scale(wi_res.scale, wi_res.shape[1]),
          partial_sum=None,
          tile_info=gmm_v2.TileSizes(*wi_dlhs_t),
          preferred_element_type=jnp.bfloat16,
          group_offset=jnp.int32(0),
          transpose_rhs=True,
          maybe_quantize_lhs=False,
          zero_initialize=False,
          out_is_3d=True,
          rhs_buffer_count=2,
          dequant_rhs_in_vmem=False,
      )
      dx_ep_3d = tc.ragged_gather_reduce_tc(
          dx_sorted_3d,
          rmeta_in,
          num_out_tokens=num_tokens_ep,
          top_k=cfg.num_experts_per_tok,
      )
    else:
      valid = jnp.arange(buf_size, dtype=jnp.int32) < cnt_in
      gathered_g_3d = jnp.where(
          valid[:, None, None],
          jnp.take(g_partial_ep, jnp.where(valid, tok_idx_in, 0), axis=0),
          0,
      )
      g_fp8, s_g = ragged_ops.ragged_absmax_quantize(
          gathered_g_3d, jnp.float8_e4m3fn, cnt_in, use_pallas=False
      )
      act_fp8_bwd = ragged_ops.ragged_silu_mul_fwd(
          gate_up,
          0,
          w_rows_in[:, None],
          cnt_in,
          out_dtype=jnp.float8_e4m3fn,
          out_scale=s_act,
          use_pallas=False,
      )
      g_deq_2d = g_fp8.reshape(buf_size, -1).astype(jnp.float32) * s_g
      act_deq_2d = act_fp8_bwd.astype(jnp.float32) * s_act
      dwo_raw = jax.lax.ragged_dot_general(
          act_deq_2d,
          g_deq_2d,
          gs_in,
          _DRHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(wo_ste_dtype)
      if ps_wo is not None:
        dwo_raw = dwo_raw + ps_wo
      wo_f32 = wo_res.codes.astype(jnp.float32) * wo_res.scale
      d_act = jax.lax.ragged_dot_general(
          g_deq_2d,
          wo_f32,
          gs_in,
          _DLHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(jnp.bfloat16)
      d_gate_up, dw_rows_2d, d_gate_amax = ragged_ops.ragged_silu_mul_bwd(
          d_act,
          gate_up,
          0,
          w_rows_in[:, None],
          cnt_in,
          with_absmax=True,
          use_pallas=False,
      )
      s_dgate = ragged_ops.absmax_to_scale(d_gate_amax, jnp.float8_e4m3fn)
      d_gate_fp8 = ragged_ops.ragged_quantize(
          d_gate_up, s_dgate, jnp.float8_e4m3fn, cnt_in, use_pallas=False
      )
      d_gate_deq = d_gate_fp8.astype(jnp.float32) * s_dgate
      x_deq_2d = (
          sorted_fp8.reshape(buf_size, -1).astype(jnp.float32) * disp_scale
      )
      dwi_raw = jax.lax.ragged_dot_general(
          x_deq_2d,
          d_gate_deq,
          gs_in,
          _DRHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(wi_ste_dtype)
      if ps_wi is not None:
        dwi_raw = dwi_raw + ps_wi
      wi_f32 = wi_res.codes.astype(jnp.float32) * wi_res.scale
      dx_2d = jax.lax.ragged_dot_general(
          d_gate_deq,
          wi_f32,
          gs_in,
          _DLHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(ste_dtype)
      dx_sorted_3d = dx_2d.reshape(buf_size, dx_2d.shape[1] // 128, 128)
      safe_dst = jnp.where(valid, tok_idx_in, num_tokens_ep)
      padded_dx = jnp.zeros(
          (num_tokens_ep + 1, *dx_sorted_3d.shape[1:]), dtype=ste_dtype
      )
      dx_ep_3d = padded_dx.at[safe_dst].add(
          jnp.where(valid[:, None, None], dx_sorted_3d, 0)
      )[:num_tokens_ep]

    ops.assert_bf16_or_fp32_reduction(dwi_raw, "routed_moe_3d_dwi_out")
    ops.assert_bf16_or_fp32_reduction(dwo_raw, "routed_moe_3d_dwo_out")
    d_disp = ops.Fp8DispatchedTokens(
        ste_carrier=dx_ep_3d.astype(ste_dtype),
        codes_ep=jax.lax.broadcast(
            jnp.zeros((), dtype=jnp.float8_e4m3fn), ep_tokens_shape
        ),
        scale=jnp.zeros_like(disp_scale),
    )
    dwi = Fp8SteWeight(
        w_ste=dwi_raw,
        codes=jax.lax.broadcast(
            jnp.zeros((), dtype=wi_codes_dtype), wi_codes_shape
        ),
        scale=jnp.zeros(wi_scale_shape, dtype=jnp.float32),
    )
    dwo = Fp8SteWeight(
        w_ste=dwo_raw,
        codes=jax.lax.broadcast(
            jnp.zeros((), dtype=wo_codes_dtype), wo_codes_shape
        ),
        scale=jnp.zeros(wo_scale_shape, dtype=jnp.float32),
    )
    return (
        d_disp,
        dw_rows_2d[:, 0].astype(w_rows_in.dtype),
        dwi,
        dwo,
        None,
        None,
        None,
        None,
    )

  if return_rhs:

    @jax.custom_vjp
    def _op_ret(d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in):
      out, _ = _fwd_impl(
          d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
      )
      return out, wi_in, wo_in

    def _op_ret_fwd(
        d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
    ):
      out, res = _fwd_impl(
          d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
      )
      return (out, wi_in, wo_in), res

    def _op_ret_bwd(res, grads):
      g_out, g_wi_ps, g_wo_ps = grads
      return _bwd_impl(res, g_out, g_wi_ps, g_wo_ps)

    _op_ret.defvjp(_op_ret_fwd, _op_ret_bwd)
    return _op_ret(
        disp, w_rows, wi, wo, local_gs, token_ids_sorted, count, reduce_meta
    )

  @jax.custom_vjp
  def _op(d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in):
    out, _ = _fwd_impl(
        d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
    )
    return out

  def _op_fwd(d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in):
    return _fwd_impl(
        d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
    )

  def _op_bwd(res, g_out):
    return _bwd_impl(res, g_out, None, None)

  _op.defvjp(_op_fwd, _op_bwd)
  return _op(
      disp, w_rows, wi, wo, local_gs, token_ids_sorted, count, reduce_meta
  )


def _scalar_w_scale(scale: jax.Array) -> jax.Array:
  return (
      scale.reshape(1, 1) if scale.size == 1 else jnp.max(scale).reshape(1, 1)
  )


def _resolve_forked_3d_tiles(cfg: QwenConfig, buf_size: int):
  """Clamps all 5 forked 3D kernel tile configs cleanly to `(buf_size, D, F)`."""
  emb_dim = cfg.emb_dim
  mlp_dim = cfg.moe_mlp_dim
  two_f = 2 * mlp_dim

  gate_tm = min(cfg.gate_fwd_tiles[0], buf_size)
  gate_tk = _clamp_3d_k_tile(cfg.gate_fwd_tiles[1], emb_dim)
  gate_bb = math.gcd(gate_tm, cfg.gate_fwd_tiles[3])
  gate_tiles = gmmv2_kernel.TileSizes(
      tile_m=gate_tm, tile_k=gate_tk, tile_n=two_f, bucket_base=gate_bb
  )

  lc_tm = min(cfg.lc_fwd_tiles[0], buf_size)
  lc_tn = _clamp_mult128_tile(cfg.lc_fwd_tiles[1], emb_dim, min_tile=256)
  lc_bm = max(32, _largest_pow2_divisor(math.gcd(lc_tm, cfg.lc_fwd_tiles[2])))
  if lc_bm > lc_tm or lc_tm % lc_bm != 0:
    lc_bm = _largest_pow2_divisor(lc_tm)
  lc_tiles = linear_combine_fwd.Tiles(
      tile_m=lc_tm, tile_n=lc_tn, bucket_m=lc_bm
  )

  dg_tr = min(cfg.lbwd_dgrad_tiles[0], buf_size)
  dg_br = math.gcd(dg_tr, cfg.lbwd_dgrad_tiles[1])
  dg_tn = _clamp_mult128_tile(cfg.lbwd_dgrad_tiles[2], mlp_dim, min_tile=128)
  dg_tk = _clamp_3d_k_tile(cfg.lbwd_dgrad_tiles[3], emb_dim)
  dgrad_tiles = linear_bwd_split.DgradTiles(
      tile_rows=dg_tr, bucket_rows=dg_br, tile_n=dg_tn, tile_k=dg_tk
  )

  wg_tr_req = min(cfg.lbwd_wgrad_tiles[0], buf_size)
  wg_tr = max(dg_tr, (wg_tr_req // dg_tr) * dg_tr)
  wg_br = math.gcd(wg_tr, cfg.lbwd_wgrad_tiles[1])
  wg_td = _clamp_mult128_tile(cfg.lbwd_wgrad_tiles[2], emb_dim, min_tile=128)
  wgrad_tiles = linear_bwd_split.WgradTiles(
      tile_rows=wg_tr, bucket_rows=wg_br, tile_d=wg_td
  )

  gb_tr_req = min(cfg.gbwd_fused_tiles[0], buf_size)
  gb_tr = max(dg_tr, (gb_tr_req // dg_tr) * dg_tr)
  gb_br = math.gcd(gb_tr, cfg.gbwd_fused_tiles[1])
  gb_td = _clamp_3d_k_tile(cfg.gbwd_fused_tiles[2], emb_dim)
  gbwd_tiles = gate_bwd_fused.FusedTiles(
      tile_rows=gb_tr, bucket_rows=gb_br, tile_d=gb_td
  )
  return gate_tiles, lc_tiles, dgrad_tiles, wgrad_tiles, gbwd_tiles


def routed_moe_forked_3d_fp8(
    disp: ops.Fp8DispatchedTokens,
    w_rows: jax.Array,
    wi: Fp8SteWeight,
    wo: Fp8SteWeight,
    local_gs: jax.Array,
    token_ids_sorted: jax.Array,
    count: jax.Array,
    reduce_meta: Any,
    cfg: QwenConfig,
    *,
    buf_size: int,
    return_rhs: bool = False,
    forward_tie: Any = None,
    wrap_partial_fp8_bwd: bool = False,
    sublayer_idx: int = 0,
    use_external_bwd_rw: bool = False,
    use_external_bwd_disp: bool = False,
) -> Any:
  """Zero-relayout forked 3D VMEM-relayout FP8 `gmmv2` MoE pipeline.

  Works uniformly across both Qwen MAX (`D=8192, E_dim=2048`) and Qwen 3.5
  (`D=4096, E_dim=1024`), across `EP=16` (`32` local experts) and `EP=8`
  (`64` local experts). Forward uses `gmm_v2(..., glu_coeffs=w_rows,
  lhs_gather_indices=...)` + `linear_combine_fwd`; backward uses
  `linear_bwd_dgrad` + `linear_bwd_wgrad` + `ragged_gather_tc` +
  `gate_bwd_fused` + `ragged_gather_reduce_tc`.

  Args:
    disp: Dispatched FP8 tokens (`Fp8DispatchedTokens`).
    w_rows: Routing weights aligned to sorted tokens.
    wi: Gate/up routed expert weights (`Fp8SteWeight`).
    wo: Down-projection routed expert weights (`Fp8SteWeight`).
    local_gs: Local expert group sizes `[E_local]`.
    token_ids_sorted: Sorted EP token indices `[buf_size]`.
    count: Scalar active routed token count.
    reduce_meta: `ragged_gather_reduce_tc_metadata` structure.
    cfg: Qwen model & kernel configuration.
    buf_size: Padded sorted token buffer capacity.
    return_rhs: If True, also return `(wi, wo)` carriers for bwd chaining.
    forward_tie: Optional forward dependency tie scalar/array.
    wrap_partial_fp8_bwd: If True, wrap output in `MoePartialTokens`.
    sublayer_idx: Superblock sublayer index (`0..3`) for remat naming.
    use_external_bwd_rw: If True, attach external prefetched bwd weight slot.

  Returns:
    Partial EP output (plus optional `(wi, wo)` or `rw_slot` carriers).
  """
  assert_global_mixed_precision_contract(cfg)
  if use_external_bwd_disp and not use_external_bwd_rw:
    raise ValueError(
        "use_external_bwd_disp=True requires use_external_bwd_rw=True."
    )
  use_pallas = cfg.use_pallas_kernels
  num_tokens_ep = disp.codes_ep.shape[0]
  ep_tokens_shape = disp.codes_ep.shape
  disp_scale_shape = disp.scale.shape
  ste_dtype = _assert_bf16_plus_dtype(
      disp.ste_carrier.dtype, "routed_moe_forked_3d_fp8_dx"
  )
  wi_codes_shape = wi.codes.shape
  wi_codes_dtype = wi.codes.dtype
  wi_ste_dtype = _assert_bf16_plus_dtype(
      wi.dtype, "routed_moe_forked_3d_fp8_dwi"
  )
  wi_scale_shape = wi.scale.shape
  wo_codes_shape = wo.codes.shape
  wo_codes_dtype = wo.codes.dtype
  wo_ste_dtype = _assert_bf16_plus_dtype(
      wo.dtype, "routed_moe_forked_3d_fp8_dwo"
  )
  wo_scale_shape = wo.scale.shape

  gate_tiles, lc_tiles, dgrad_tiles, wgrad_tiles, gbwd_tiles = (
      _resolve_forked_3d_tiles(cfg, buf_size)
  )

  def _fwd_impl(
      disp_in: ops.Fp8DispatchedTokens,
      w_rows_in: jax.Array,
      wi_in: Fp8SteWeight,
      wo_in: Fp8SteWeight,
      gs_in: jax.Array,
      tok_idx_in: jax.Array,
      cnt_in: jax.Array,
      rmeta_in: Any,
  ):
    scale_tie_i32 = jax.lax.stop_gradient(
        (ops.first_scalar(disp_in.scale) * 0.0).astype(jnp.int32)
    )
    wi_s = _scalar_w_scale(wi_in.scale)
    wo_s = _scalar_w_scale(wo_in.scale)
    act_s_init = jnp.full((1, 1), cfg.moe_act_fp8_scale, dtype=jnp.float32)
    w_rows_col = w_rows_in.astype(jnp.float32)[:, None]

    if use_pallas:
      if cfg.moe_gmm_no_alias_out:
        bank_init = None
      else:
        bank_init = ops.alloc_activation_bank(
            buf_size, 2 * cfg.moe_mlp_dim, dtype=jnp.bfloat16, constrain=True
        )
      bank_2d, act_fp8 = gmmv2_kernel.gmm_v2(
          lhs=disp_in.codes_ep,
          rhs=ops.routed_w_from_tiles(wi_in.codes),
          group_sizes=gs_in,
          lhs_scale=None,
          rhs_scale=None,
          out_scale=(disp_in.scale.reshape(1, 1) * wi_s).reshape(1, 1),
          tile_info=gate_tiles,
          preferred_element_type=jnp.bfloat16,
          acc_dtype=jnp.float32,
          group_offset=jnp.int32(0),
          maybe_quantize_lhs=False,
          zero_initialize=False,
          out_is_3d=False,
          flat_rhs=True,
          disable_multi_core_mode=True,
          out=bank_init,
          out_offset=0,
          glu_coeffs=w_rows_col,
          glu_out_dtype=jnp.float8_e4m3fn,
          glu_out_scale=act_s_init,
          lhs_gather_indices=tok_idx_in,
          lhs_gather_start=scale_tie_i32,
          lhs_gather_size_m=buf_size,
      )
      act_s_lc = (
          ops.order_after(act_s_init, forward_tie)
          if forward_tie is not None
          else act_s_init
      )
      partial_ep_3d = linear_combine_fwd.linear_combine_fwd(
          act_fp8,
          ops.routed_w_from_tiles(wo_in.codes),
          gs_in,
          tok_idx_in,
          0,
          rmeta_in.tokens,
          out_scale=(act_s_lc * wo_s).reshape(1, 1),
          num_out_tokens=num_tokens_ep,
          tiles=lc_tiles,
      )
      if cfg.moe_save_bank_in_superblock:
        bank_tiles = ops.bank_to_tiles(
            ops.constrain_activation_bank(bank_2d, enabled=True)
        )
      else:
        bank_tiles = bank_2d
    else:
      valid = jnp.arange(buf_size, dtype=jnp.int32) < cnt_in
      safe_src = jnp.where(valid, tok_idx_in + scale_tie_i32, 0)
      sorted_fp8 = jnp.where(
          valid[:, None, None],
          jnp.take(disp_in.codes_ep, safe_src, axis=0),
          jnp.zeros((), dtype=jnp.float8_e4m3fn),
      )
      x_deq_2d = sorted_fp8.reshape(buf_size, -1).astype(
          jnp.float32
      ) * disp_in.scale.reshape(1, 1)
      wi_f32 = ops.routed_w_from_tiles(wi_in.codes).astype(
          jnp.float32
      ) * wi_s.reshape(1, 1, 1)
      gate_up = jax.lax.ragged_dot_general(
          x_deq_2d,
          wi_f32,
          gs_in,
          _FWD_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(jnp.bfloat16)
      bank_2d = jnp.where(
          valid[:, None], gate_up, jnp.zeros((), dtype=jnp.bfloat16)
      )
      act_fp8 = ragged_ops.ragged_silu_mul_fwd(
          bank_2d,
          0,
          w_rows_col,
          cnt_in,
          out_dtype=jnp.float8_e4m3fn,
          out_scale=act_s_init,
          use_pallas=False,
      )
      act_s_lc = (
          ops.order_after(act_s_init, forward_tie)
          if forward_tie is not None
          else act_s_init
      )
      act_deq = act_fp8.astype(jnp.float32) * act_s_lc
      wo_f32 = ops.routed_w_from_tiles(wo_in.codes).astype(
          jnp.float32
      ) * wo_s.reshape(1, 1, 1)
      out_2d = jax.lax.ragged_dot_general(
          act_deq,
          wo_f32,
          gs_in,
          _FWD_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(ste_dtype)
      weighted_down_3d = out_2d.reshape(buf_size, out_2d.shape[1] // 128, 128)
      safe_dst = jnp.where(valid, tok_idx_in, num_tokens_ep)
      padded_out = jnp.zeros(
          (num_tokens_ep + 1, *weighted_down_3d.shape[1:]), dtype=ste_dtype
      )
      partial_ep_3d = padded_out.at[safe_dst].add(
          jnp.where(valid[:, None, None], weighted_down_3d, 0)
      )[:num_tokens_ep]
      bank_tiles = ops.bank_to_tiles(bank_2d)

    if wrap_partial_fp8_bwd:
      partial_out: jax.Array | ops.MoePartialTokens = ops.MoePartialTokens(
          y_ep=partial_ep_3d,
          dy_codes=jax.lax.broadcast(
              jnp.zeros((), dtype=jnp.float8_e5m2), ep_tokens_shape
          ),
          dy_scale=jnp.zeros((1, 1), dtype=jnp.float32),
      )
    else:
      partial_out = partial_ep_3d

    wi_res = (
        None if use_external_bwd_rw else dataclasses.replace(wi_in, w_ste=None)
    )
    wo_res = (
        None if use_external_bwd_rw else dataclasses.replace(wo_in, w_ste=None)
    )
    disp_codes_res = (
        None
        if use_external_bwd_disp
        else jax.ad_checkpoint.checkpoint_name(
            disp_in.codes_ep, f"moe_disp_ep_{sublayer_idx}"
        )
    )
    disp_scale_res = (
        None
        if use_external_bwd_disp
        else jax.ad_checkpoint.checkpoint_name(
            disp_in.scale, f"moe_disp_ep_{sublayer_idx}"
        )
    )
    bank_tiles_res = jax.ad_checkpoint.checkpoint_name(
        bank_tiles, f"moe_gate_up_{sublayer_idx}"
    )
    res = (
        disp_codes_res,
        disp_scale_res,
        bank_tiles_res,
        w_rows_in,
        wi_res,
        wo_res,
        gs_in,
        tok_idx_in,
        cnt_in,
        rmeta_in,
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
        w_rows_in,
        wi_res,
        wo_res,
        gs_in,
        tok_idx_in,
        cnt_in,
        rmeta_in,
    ) = res
    ps_wi_raw = (
        ps_wi_carrier.w_ste
        if isinstance(ps_wi_carrier, Fp8SteWeight)
        else ps_wi_carrier
    )
    ps_wo_raw = (
        ps_wo_carrier.w_ste
        if isinstance(ps_wo_carrier, Fp8SteWeight)
        else ps_wo_carrier
    )
    ps_wi_3d = (
        ops.routed_w_from_tiles(ps_wi_raw) if ps_wi_raw is not None else None
    )
    ps_wo_3d = (
        ops.routed_w_from_tiles(ps_wo_raw) if ps_wo_raw is not None else None
    )
    if ps_wi_3d is not None:
      ops.assert_bf16_or_fp32_reduction(ps_wi_3d, "forked_3d_bwd_ps_wi")
    if ps_wo_3d is not None:
      ops.assert_bf16_or_fp32_reduction(ps_wo_3d, "forked_3d_bwd_ps_wo")

    if ext_bwd_rw is not None:
      wi_codes = ext_bwd_rw.wi_codes
      wi_scale = ext_bwd_rw.wi_scale
      wo_codes = ext_bwd_rw.wo_codes
      wo_scale = ext_bwd_rw.wo_scale
      if ext_bwd_rw.disp_codes_ep is not None:
        disp_codes_ep = ext_bwd_rw.disp_codes_ep
        disp_scale = ext_bwd_rw.disp_scale
    else:
      assert wi_res is not None and wo_res is not None
      wi_codes = wi_res.codes
      wi_scale = wi_res.scale
      wo_codes = wo_res.codes
      wo_scale = wo_res.scale
    assert disp_codes_ep is not None and disp_scale is not None

    wi_s = _scalar_w_scale(wi_scale)
    wo_s = _scalar_w_scale(wo_scale)
    act_s = jnp.full((1, 1), cfg.moe_act_fp8_scale, dtype=jnp.float32)
    w_rows_col = w_rows_in.astype(jnp.float32)[:, None]

    if isinstance(g_partial_ep, ops.MoePartialTokens):
      dy_ag = g_partial_ep.dy_codes
      dy_scale = g_partial_ep.dy_scale
    else:
      dy_ag = g_partial_ep.astype(jnp.bfloat16) if use_pallas else g_partial_ep
      dy_scale = None

    if use_pallas:
      bank_2d = (
          ops.bank_from_tiles(bank_tiles, constrain=True)
          if bank_tiles.ndim == 3
          else bank_tiles
      )
      dgrad_tr = dgrad_tiles.tile_rows
      dgrad_res = linear_bwd_split.linear_bwd_dgrad(
          dy_ag,
          tok_idx_in,
          0,
          ops.routed_w_from_tiles(wo_codes),
          gs_in,
          bank_2d,
          0,
          w_rows_col,
          act_scale=act_s,
          w_scale=wo_s,
          act_qtype=jnp.float8_e4m3fn,
          dy_qtype=jnp.float8_e5m2,
          dy_scale=dy_scale,
          grad_gate_qtype=jnp.float8_e5m2,
          tiles=dgrad_tiles,
      )
      dwo_3d = linear_bwd_split.linear_bwd_wgrad(
          dgrad_res.act_q,
          dgrad_res.dy_q,
          dgrad_res.tile_exps,
          gs_in,
          act_scale=act_s,
          dgrad_tile_rows=dgrad_tr,
          dy_scale=dy_scale,
          dw_acc=ps_wo_3d,
          dw_dtype=wo_ste_dtype,
          tiles=wgrad_tiles,
      )
      bwd_anchor = (
          dy_scale if dy_scale is not None else dgrad_res.grad_gate_amax
      )
      gather_start_i32 = jax.lax.stop_gradient(
          (ops.first_scalar(bwd_anchor) * 0.0).astype(jnp.int32)
      )
      if cfg.moe_bwd_tie_wgrad_before_gate:
        dwo_tie_i32 = jax.lax.stop_gradient(
            (ops.first_scalar(dwo_3d).astype(jnp.float32) * 0.0).astype(
                jnp.int32
            )
        )
        gather_start_i32 = gather_start_i32 + dwo_tie_i32
      sorted_fp8 = tc.ragged_gather_tc(
          disp_codes_ep,
          tok_idx_in,
          start=gather_start_i32,
          num_tokens=cnt_in,
          max_out_tokens=buf_size,
          block_size=cfg.tc_gather_block_size,
      )
      dx_sorted_3d, dwi_3d = gate_bwd_fused.gate_bwd_fused(
          sorted_fp8,
          ops.routed_w_from_tiles(wi_codes),
          gs_in,
          dgrad_res.grad_gate,
          dx_scale=wi_s,
          dw_scale=disp_scale.reshape(1, 1),
          dgate_tile_exps=dgrad_res.grad_gate_tile_exps,
          dgrad_tile_rows=dgrad_tr,
          dw_acc=ps_wi_3d,
          dw_dtype=wi_ste_dtype,
          tiles=gbwd_tiles,
      )
      dx_ep_3d = tc.ragged_gather_reduce_tc(
          dx_sorted_3d,
          rmeta_in,
          num_out_tokens=num_tokens_ep,
          top_k=cfg.num_experts_per_tok,
      )
      dw_rows_2d = dgrad_res.grad_coeffs
    else:
      bank_2d = ops.bank_from_tiles(bank_tiles, constrain=False)
      valid = jnp.arange(buf_size, dtype=jnp.int32) < cnt_in
      bwd_tie_f32 = (
          jax.lax.stop_gradient(ops.first_scalar(dy_scale) * 0.0)
          if dy_scale is not None
          else 0.0
      )
      if dy_scale is not None:
        gathered_dy_q = jnp.where(
            valid[:, None, None],
            jnp.take(dy_ag, jnp.where(valid, tok_idx_in, 0), axis=0),
            jnp.zeros((), dtype=dy_ag.dtype),
        )
        g_deq_2d = gathered_dy_q.reshape(buf_size, -1).astype(jnp.float32) * (
            dy_scale.reshape(1, 1) + bwd_tie_f32
        )
      else:
        gathered_g_3d = jnp.where(
            valid[:, None, None],
            jnp.take(dy_ag, jnp.where(valid, tok_idx_in, 0), axis=0),
            0,
        )
        g_fp8, s_g = ragged_ops.ragged_absmax_quantize(
            gathered_g_3d, jnp.float8_e5m2, cnt_in, use_pallas=False
        )
        g_deq_2d = g_fp8.reshape(buf_size, -1).astype(jnp.float32) * s_g

      act_fp8_bwd = ragged_ops.ragged_silu_mul_fwd(
          bank_2d,
          0,
          w_rows_col,
          cnt_in,
          out_dtype=jnp.float8_e4m3fn,
          out_scale=act_s,
          use_pallas=False,
      )
      act_deq_2d = act_fp8_bwd.astype(jnp.float32) * act_s
      dwo_3d = jax.lax.ragged_dot_general(
          act_deq_2d,
          g_deq_2d,
          gs_in,
          _DRHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(wo_ste_dtype)
      if ps_wo_3d is not None:
        dwo_3d = dwo_3d + ps_wo_3d
      wo_f32 = ops.routed_w_from_tiles(wo_codes).astype(
          jnp.float32
      ) * wo_s.reshape(1, 1, 1)
      d_act = jax.lax.ragged_dot_general(
          g_deq_2d,
          wo_f32,
          gs_in,
          _DLHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(jnp.bfloat16)
      d_gate_up, dw_rows_2d, d_gate_amax = ragged_ops.ragged_silu_mul_bwd(
          d_act,
          bank_2d,
          0,
          w_rows_col,
          cnt_in,
          with_absmax=True,
          use_pallas=False,
      )
      s_dgate = ragged_ops.absmax_to_scale(d_gate_amax, jnp.float8_e5m2)
      d_gate_fp8 = ragged_ops.ragged_quantize(
          d_gate_up, s_dgate, jnp.float8_e5m2, cnt_in, use_pallas=False
      )
      d_gate_deq = d_gate_fp8.astype(jnp.float32) * s_dgate
      sorted_fp8 = jnp.where(
          valid[:, None, None],
          jnp.take(disp_codes_ep, jnp.where(valid, tok_idx_in, 0), axis=0),
          jnp.zeros((), dtype=jnp.float8_e4m3fn),
      )
      x_deq_2d = sorted_fp8.reshape(buf_size, -1).astype(
          jnp.float32
      ) * disp_scale.reshape(1, 1)
      dwi_3d = jax.lax.ragged_dot_general(
          x_deq_2d,
          d_gate_deq,
          gs_in,
          _DRHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(wi_ste_dtype)
      if ps_wi_3d is not None:
        dwi_3d = dwi_3d + ps_wi_3d
      wi_f32 = ops.routed_w_from_tiles(wi_codes).astype(
          jnp.float32
      ) * wi_s.reshape(1, 1, 1)
      dx_2d = jax.lax.ragged_dot_general(
          d_gate_deq,
          wi_f32,
          gs_in,
          _DLHS_RAGGED_DIM_NUMS,
          preferred_element_type=jnp.float32,
      ).astype(ste_dtype)
      dx_sorted_3d = dx_2d.reshape(buf_size, dx_2d.shape[1] // 128, 128)
      safe_dst = jnp.where(valid, tok_idx_in, num_tokens_ep)
      padded_dx = jnp.zeros(
          (num_tokens_ep + 1, *dx_sorted_3d.shape[1:]), dtype=ste_dtype
      )
      dx_ep_3d = padded_dx.at[safe_dst].add(
          jnp.where(valid[:, None, None], dx_sorted_3d, 0)
      )[:num_tokens_ep]

    ops.assert_bf16_or_fp32_reduction(dwi_3d, "forked_3d_dwi_out")
    ops.assert_bf16_or_fp32_reduction(dwo_3d, "forked_3d_dwo_out")
    ops.assert_bf16_or_fp32_reduction(dx_ep_3d, "forked_3d_dx_ep_out")

    dwi_ste = (
        ops.routed_w_to_tiles(dwi_3d) if len(wi_codes_shape) == 4 else dwi_3d
    )
    dwo_ste = (
        ops.routed_w_to_tiles(dwo_3d) if len(wo_codes_shape) == 4 else dwo_3d
    )
    d_disp = ops.Fp8DispatchedTokens(
        ste_carrier=dx_ep_3d.astype(ste_dtype),
        codes_ep=jax.lax.broadcast(
            jnp.zeros((), dtype=jnp.float8_e4m3fn), ep_tokens_shape
        ),
        scale=jnp.zeros_like(disp_scale),
    )
    dwi = Fp8SteWeight(
        w_ste=dwi_ste,
        codes=jax.lax.broadcast(
            jnp.zeros((), dtype=wi_codes_dtype), wi_codes_shape
        ),
        scale=jnp.zeros(wi_scale_shape, dtype=jnp.float32),
    )
    dwo = Fp8SteWeight(
        w_ste=dwo_ste,
        codes=jax.lax.broadcast(
            jnp.zeros((), dtype=wo_codes_dtype), wo_codes_shape
        ),
        scale=jnp.zeros(wo_scale_shape, dtype=jnp.float32),
    )
    return (
        d_disp,
        dw_rows_2d[:, 0].astype(w_rows_in.dtype),
        dwi,
        dwo,
        None,
        None,
        None,
        None,
    )

  if use_external_bwd_rw:
    ext_disp_codes_shape = ep_tokens_shape if use_external_bwd_disp else None
    ext_disp_scale_shape = disp_scale_shape if use_external_bwd_disp else None
    if return_rhs:

      @jax.custom_vjp
      def _op_ret_ext(
          d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
      ):
        out, _ = _fwd_impl(
            d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
        )
        z_slot = ops.zero_bwd_rw_prefetch(
            wi_codes_shape,
            wi_scale_shape,
            wo_codes_shape,
            wo_scale_shape,
            disp_codes_shape=ext_disp_codes_shape,
            disp_scale_shape=ext_disp_scale_shape,
        )
        return out, wi_in, wo_in, z_slot

      def _op_ret_ext_fwd(
          d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
      ):
        out, res = _fwd_impl(
            d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
        )
        z_slot = ops.zero_bwd_rw_prefetch(
            wi_codes_shape,
            wi_scale_shape,
            wo_codes_shape,
            wo_scale_shape,
            disp_codes_shape=ext_disp_codes_shape,
            disp_scale_shape=ext_disp_scale_shape,
        )
        return (out, wi_in, wo_in, z_slot), res

      def _op_ret_ext_bwd(res, grads):
        g_out, g_wi_ps, g_wo_ps, g_ext_rw = grads
        return _bwd_impl(res, g_out, g_wi_ps, g_wo_ps, ext_bwd_rw=g_ext_rw)

      _op_ret_ext.defvjp(_op_ret_ext_fwd, _op_ret_ext_bwd)
      return _op_ret_ext(
          disp, w_rows, wi, wo, local_gs, token_ids_sorted, count, reduce_meta
      )

    @jax.custom_vjp
    def _op_ext(d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in):
      out, _ = _fwd_impl(
          d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
      )
      z_slot = ops.zero_bwd_rw_prefetch(
          wi_codes_shape,
          wi_scale_shape,
          wo_codes_shape,
          wo_scale_shape,
          disp_codes_shape=ext_disp_codes_shape,
          disp_scale_shape=ext_disp_scale_shape,
      )
      return out, z_slot

    def _op_ext_fwd(
        d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
    ):
      out, res = _fwd_impl(
          d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
      )
      z_slot = ops.zero_bwd_rw_prefetch(
          wi_codes_shape,
          wi_scale_shape,
          wo_codes_shape,
          wo_scale_shape,
          disp_codes_shape=ext_disp_codes_shape,
          disp_scale_shape=ext_disp_scale_shape,
      )
      return (out, z_slot), res

    def _op_ext_bwd(res, grads):
      g_out, g_ext_rw = grads
      return _bwd_impl(res, g_out, None, None, ext_bwd_rw=g_ext_rw)

    _op_ext.defvjp(_op_ext_fwd, _op_ext_bwd)
    return _op_ext(
        disp, w_rows, wi, wo, local_gs, token_ids_sorted, count, reduce_meta
    )

  if return_rhs:

    @jax.custom_vjp
    def _op_ret(d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in):
      out, _ = _fwd_impl(
          d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
      )
      return out, wi_in, wo_in

    def _op_ret_fwd(
        d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
    ):
      out, res = _fwd_impl(
          d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
      )
      return (out, wi_in, wo_in), res

    def _op_ret_bwd(res, grads):
      g_out, g_wi_ps, g_wo_ps = grads
      return _bwd_impl(res, g_out, g_wi_ps, g_wo_ps)

    _op_ret.defvjp(_op_ret_fwd, _op_ret_bwd)
    return _op_ret(
        disp, w_rows, wi, wo, local_gs, token_ids_sorted, count, reduce_meta
    )

  @jax.custom_vjp
  def _op(d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in):
    out, _ = _fwd_impl(
        d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
    )
    return out

  def _op_fwd(d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in):
    return _fwd_impl(
        d_in, wr_in, wi_in, wo_in, gs_in, tidx_in, cnt_in, rmeta_in
    )

  def _op_bwd(res, g_out):
    return _bwd_impl(res, g_out, None, None)

  _op.defvjp(_op_fwd, _op_bwd)
  return _op(
      disp, w_rows, wi, wo, local_gs, token_ids_sorted, count, reduce_meta
  )
