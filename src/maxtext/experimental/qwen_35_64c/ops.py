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

"""Pure-JAX Dual-SparseCore collectives, FP8 helpers, norms, and barriers."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
from jax.experimental.compute_on import compute_on
from jax.experimental.layout import Layout, with_layout_constraint
from jax.experimental.xla_metadata import set_xla_metadata
import jax.numpy as jnp

from maxtext.experimental.qwen_35_64c.config import Qwen35Config
from maxtext.experimental.qwen_35_64c.ext_kernels import rmsnorm_tc

# Matches `--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=2000000`
# so `< 2.0 MB` routing collectives (`1.31 MB` `topk_weights`/`topk_ids`) run on
# fast TensorCore ICI (`~0.01 ms`, zero SC0 contention) while `>= 4.19 MB`
# dense Stage-1/2 FSDP collectives (`in_proj_ba`, `out_proj`) stay on SC1!
_SC_MIN_COLLECTIVE_BYTES = 2_000_000


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Fp8DispatchedTokens:
  """Pre-quantized 3D/2D FP8 dispatched EP tokens with zero-byte bf16 STE carrier.

  - `ste_carrier`: 0-byte lazy `broadcast(0, [T_ep, ...])` carrying `dx_ep` in
  bwd.
  - `codes_ep`: `[T_ep, D // 128, 128]` (or `[T_ep, D]`) `float8_e4m3fn`
  gathered on SC0.
  - `scale`: `[1, 1]` `float32` per-EP-group dynamic quantization scale.
  """

  ste_carrier: Any
  codes_ep: jax.Array
  scale: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    return self.codes_ep.shape

  @property
  def dtype(self) -> jnp.dtype:
    return jnp.bfloat16 if self.ste_carrier is None else self.ste_carrier.dtype

  @property
  def ndim(self) -> int:
    return self.codes_ep.ndim


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoePartialTokens:
  """Partial EP output tokens carrying pre-quantized `fp8_e5m2` `dy` in backward.

  - `y_ep`: `[T_ep, D // 128, 128]` `bfloat16` partial EP output in forward
    (0-byte broadcast in backward).
  - `dy_codes`: `[T_ep, D // 128, 128]` `float8_e5m2` pre-quantized cotangent in
    backward (0-byte broadcast in forward).
  - `dy_scale`: `[1, 1]` `float32` per-EP-group `dy` scale in backward.
  """

  y_ep: jax.Array
  dy_codes: jax.Array
  dy_scale: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    return self.y_ep.shape

  @property
  def dtype(self) -> jnp.dtype:
    return self.y_ep.dtype

  @property
  def ndim(self) -> int:
    return self.y_ep.ndim


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class BwdRoutedWeightPrefetch:
  """Zero-forward-byte carrier transporting prefetched `attn_3` FP8 weights across backward scan."""

  wi_codes: jax.Array
  wi_scale: jax.Array
  wo_codes: jax.Array
  wo_scale: jax.Array


def zero_bwd_rw_prefetch(
    wi_codes_shape: tuple[int, ...],
    wi_scale_shape: tuple[int, ...],
    wo_codes_shape: tuple[int, ...],
    wo_scale_shape: tuple[int, ...],
) -> BwdRoutedWeightPrefetch:
  """Creates 0-byte lazy broadcast placeholders for `BwdRoutedWeightPrefetch` in forward."""
  z_f8 = jnp.zeros((), dtype=jnp.float8_e4m3fn)
  z_f32 = jnp.zeros((), dtype=jnp.float32)
  return BwdRoutedWeightPrefetch(
      wi_codes=jax.lax.broadcast(z_f8, wi_codes_shape),
      wi_scale=jax.lax.broadcast(z_f32, wi_scale_shape),
      wo_codes=jax.lax.broadcast(z_f8, wo_codes_shape),
      wo_scale=jax.lax.broadcast(z_f32, wo_scale_shape),
  )


def merge_bwd_rw3_slots(
    slot_mb0: BwdRoutedWeightPrefetch,
    slot_mb1: BwdRoutedWeightPrefetch,
) -> BwdRoutedWeightPrefetch:
  """Zero-residual fanout sharing prefetched `attn_3` weights across `mb1` (`B4`) and `mb0` (`B3`)."""
  wi_cs = slot_mb0.wi_codes.shape
  wi_ss = slot_mb0.wi_scale.shape
  wo_cs = slot_mb0.wo_codes.shape
  wo_ss = slot_mb0.wo_scale.shape

  @jax.custom_vjp
  def _merge(s0: BwdRoutedWeightPrefetch, s1: BwdRoutedWeightPrefetch) -> BwdRoutedWeightPrefetch:
    del s0, s1
    return zero_bwd_rw_prefetch(wi_cs, wi_ss, wo_cs, wo_ss)

  def _merge_fwd(s0: BwdRoutedWeightPrefetch, s1: BwdRoutedWeightPrefetch):
    del s0, s1
    return zero_bwd_rw_prefetch(wi_cs, wi_ss, wo_cs, wo_ss), ()

  def _merge_bwd(_, g_pref: BwdRoutedWeightPrefetch):
    return g_pref, g_pref

  _merge.defvjp(_merge_fwd, _merge_bwd)
  return _merge(slot_mb0, slot_mb1)


def _ep_core_id(cfg: Qwen35Config) -> int | None:
  return 0 if (cfg.use_pallas_kernels and cfg.ep_sc_pinning == "sc0") else None


def _fsdp_ag_core_id(cfg: Qwen35Config, *, is_bwd_remat: bool = False) -> int | None:
  if not cfg.use_pallas_kernels:
    return None
  if cfg.fsdp_ag_sc_pinning == "sc1":
    return 1
  if cfg.fsdp_ag_sc_pinning == "sc1_fwd_only" and not is_bwd_remat:
    return 1
  return None


def _fsdp_rs_core_id(cfg: Qwen35Config) -> int | None:
  return 1 if (cfg.use_pallas_kernels and cfg.fsdp_rs_sc_pinning == "sc1") else None


def _dense_rs_core_id(cfg: Qwen35Config) -> int | None:
  if not cfg.use_pallas_kernels or cfg.fsdp_rs_sc_pinning == "unpinned":
    return None
  if cfg.bwd_sc_balance_mode == "wi_dense_rs_sc0":
    return 0
  return 1


def _moe_rs_core_id(cfg: Qwen35Config, *, is_wi: bool = False) -> int | None:
  if not cfg.use_pallas_kernels or cfg.fsdp_rs_sc_pinning == "unpinned":
    return None
  if is_wi and cfg.bwd_sc_balance_mode in ("wi_rs_sc0", "wi_dense_rs_sc0"):
    return 0
  if (not is_wi) and cfg.bwd_sc_balance_mode == "dwo_sc0":
    return 0
  return 1


def sc_all_gather(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None,
    axis: int = 0,
    min_sc_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
) -> jax.Array:
  """Single-array all-gather pinned to a specific SparseCore (`0` or `1`).

  Uses a single-output `@compute_on` wrapper so both XLA's SparseCore scheduler
  and `hybrid_sim` (`GetCollectiveFromAsyncSparseCoreOp`) resolve the exact
  inner `kAllGather` and `core_ids` cleanly. Falls back to unannotated
  `all_gather` when gathered size `< min_sc_bytes`.
  """
  axis_size = (
      jax.lax.axis_size(axis_name)
      if isinstance(axis_name, str)
      else functools.reduce(lambda a, b: a * b, (jax.lax.axis_size(n) for n in axis_name), 1)
  )
  gathered_bytes = x.size * x.dtype.itemsize * axis_size
  if core_id is None or gathered_bytes < min_sc_bytes:
    return jax.lax.all_gather(x, axis_name=axis_name, axis=axis, tiled=True)

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )
  def _ag(inp: jax.Array) -> jax.Array:
    return jax.lax.all_gather(inp, axis_name=axis_name, axis=axis, tiled=True)

  return _ag(x)


def sc_psum_scatter(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None,
    scatter_dimension: int = 0,
    min_sc_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
) -> jax.Array:
  """Single-array psum_scatter pinned to SparseCore (`0` or `1`)."""
  input_bytes = x.size * x.dtype.itemsize
  if core_id is None or input_bytes < min_sc_bytes:
    return jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=scatter_dimension, tiled=True)

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )
  def _rs(inp: jax.Array) -> jax.Array:
    return jax.lax.psum_scatter(
        inp,
        axis_name=axis_name,
        scatter_dimension=scatter_dimension,
        tiled=True,
    )

  return _rs(x)


def sc_psum(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None,
    min_sc_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
) -> jax.Array:
  """Single-array psum all-reduce pinned to SparseCore (`0` or `1`)."""
  input_bytes = x.size * x.dtype.itemsize
  if core_id is None or input_bytes < min_sc_bytes:
    return jax.lax.psum(x, axis_name=axis_name)

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )
  def _ar(inp: jax.Array) -> jax.Array:
    return jax.lax.psum(inp, axis_name=axis_name)

  return _ar(x)


# ---------------------------------------------------------------------------
# Lineage-style T(8, 128) zero-copy routed weight tiling across SparseCore
# ---------------------------------------------------------------------------


def routed_w_to_tiles(w: jax.Array) -> jax.Array:
  """Reshapes `[..., E, rows, cols]` -> `[..., E, rows // 8, 8, cols]`."""
  if w.shape[-2] % 8 != 0:
    return w
  *prefix, rows, cols = w.shape
  return w.reshape(*prefix, rows // 8, 8, cols)


def routed_w_from_tiles(w: jax.Array) -> jax.Array:
  """Restores `[..., E, rows // 8, 8, cols]` -> `[..., E, rows, cols]`."""
  if w.ndim < 4 or w.shape[-2] != 8:
    return w
  *prefix, r_tiles, sub, cols = w.shape
  return w.reshape(*prefix, r_tiles * sub, cols)


def constrain_activation_bank(bank: jax.Array, *, enabled: bool = True) -> jax.Array:
  """Pins `[R, 2F]` or `[R // 8, 8, 2F]` bf16 activation bank to `T(8, 128), (2, 1)`."""
  if not enabled or bank.shape[-1] % 128 != 0:
    return bank
  if bank.ndim == 2 and bank.shape[0] % 8 != 0:
    return bank
  tiling = ((8, 128), (2, 1)) if jnp.dtype(bank.dtype).itemsize == 2 else ((8, 128),)
  layout = Layout(major_to_minor=tuple(range(bank.ndim)), tiling=tiling)
  return with_layout_constraint(bank, layout)


def bank_to_tiles(bank_2d: jax.Array) -> jax.Array:
  """Views `[R, 2F]` activation bank as `[R // 8, 8, 2F]` so `T(8, 128)` survives carries/VJPs."""
  if bank_2d.ndim != 2 or bank_2d.shape[0] % 8 != 0:
    return bank_2d
  return bank_2d.reshape(bank_2d.shape[0] // 8, 8, bank_2d.shape[1])


def bank_from_tiles(bank_tiles: jax.Array, *, constrain: bool = True) -> jax.Array:
  """Restores `[R // 8, 8, 2F]` -> `[R, 2F]` with `T(8, 128)` layout constraint."""
  b2d = bank_tiles.reshape(-1, bank_tiles.shape[-1]) if bank_tiles.ndim == 3 else bank_tiles
  return constrain_activation_bank(b2d, enabled=constrain)


def alloc_activation_bank(
    rows: int,
    cols: int,
    *,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    constrain: bool = True,
) -> jax.Array:
  """Allocates uninitialized `[rows, cols]` activation bank in `T(8, 128)` tiling."""
  raw = jax.lax.empty((rows, cols), dtype=dtype)
  return constrain_activation_bank(raw, enabled=constrain)


# ---------------------------------------------------------------------------
# Forward-only layout & optimization barrier primitives
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def forward_layout_constraint(x: jax.Array, major_to_minor: tuple[int, ...]) -> jax.Array:
  """Pins `with_layout_constraint` in forward only; identity in backward."""
  return with_layout_constraint(x, Layout(major_to_minor=major_to_minor))


def _fwd_layout_fwd(x: jax.Array, major_to_minor: tuple[int, ...]):
  return with_layout_constraint(x, Layout(major_to_minor=major_to_minor)), None


def _fwd_layout_bwd(major_to_minor: tuple[int, ...], _, cotangent: jax.Array):
  del major_to_minor
  return (cotangent,)


forward_layout_constraint.defvjp(_fwd_layout_fwd, _fwd_layout_bwd)


def pin_row_major(x: jax.Array) -> jax.Array:
  """Pins `x` to the default row-major layout so XLA never picks token-minor EP layouts."""
  return with_layout_constraint(x, Layout(major_to_minor=tuple(range(x.ndim))))


def first_scalar(arr: jax.Array) -> jax.Array:
  """Reads element `(0, ..., 0)` directly without flattening (`reshape(-1)` copies tiled 4D arrays!)."""
  return arr[(0,) * arr.ndim] if arr.ndim > 0 else arr


def order_after(x: Any, dep: Any) -> Any:
  """Forces consumers of `x` after `dep` in forward; zero-cotangent identity in backward.

  Uses a zero-multiply data edge (`first_scalar(dep) * 0`) marked with
  `checkpoint_name(..., 'fwd_tie_scalar')` (2-4 bytes saved per tie) so:
  1. XLA cannot strip the forward dependency or block unrelated collectives.
  2. `jax.checkpoint` never saves `dep` (e.g. `512 MiB` `unperm_mb1_in`) across
     remat boundaries and never serializes backward remat on forward-only ties.
  3. Indexing `(0, ..., 0)` directly avoids flattening tiled 4D tensors
  (`copy.9462`!).
  4. When `x` is `Fp8DispatchedTokens`, ties the 4-byte `x.scale` scalar instead
     of copying a `268-537 MB` token tensor!
  """
  if isinstance(dep, Fp8DispatchedTokens):
    z_c = first_scalar(dep.codes_ep).astype(jnp.float32) * 0.0
    z_s = first_scalar(dep.scale).astype(jnp.float32) * 0.0
    z_f32 = jax.lax.stop_gradient(z_c + z_s)
  elif isinstance(dep, MoePartialTokens):
    z_f32 = jax.lax.stop_gradient(first_scalar(dep.y_ep).astype(jnp.float32) * 0.0)
  else:
    z_f32 = jax.lax.stop_gradient(first_scalar(dep).astype(jnp.float32) * 0.0)
  z_f32 = jax.ad_checkpoint.checkpoint_name(z_f32, "fwd_tie_scalar")
  if isinstance(x, Fp8DispatchedTokens):
    return Fp8DispatchedTokens(
        ste_carrier=x.ste_carrier,
        codes_ep=x.codes_ep,
        scale=x.scale + z_f32.astype(x.scale.dtype),
    )
  if isinstance(x, MoePartialTokens):
    return MoePartialTokens(
        y_ep=x.y_ep + z_f32.astype(x.y_ep.dtype),
        dy_codes=x.dy_codes,
        dy_scale=x.dy_scale,
    )
  return x + z_f32.astype(x.dtype)


# Alias kept for backward compatibility
barrier_tie = order_after


def _is_tieable_grad_leaf(v: Any) -> bool:
  return (
      hasattr(v, "dtype")
      and jnp.issubdtype(v.dtype, jnp.floating)
      and v.dtype not in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  )


def _is_custom_token_carrier(v: Any) -> bool:
  return isinstance(v, (Fp8DispatchedTokens, MoePartialTokens))


@jax.custom_vjp
def backward_barrier_tie(a_tree, b_tree):
  """No-op in forward; in backward, forces `g_a` after `g_b` via zero-multiply data edge.

  When `b_tree` holds sharded sub-layer parameters (`~48 MiB` MoE shards) and
  `a_tree` holds `(x_mb0, x_mb1)`, `g_b` is the post-`reduce-scatter` parameter
  cotangent, forcing sub-layer `j`'s SC1 FSDP `psum_scatter` to retire before
  sub-layer `j - 1`'s backward begins and immediately freeing `~6 GiB` of
  unreduced `[64, K, N]` `drhs` buffers!
  When a leaf in `a_tree` is `MoePartialTokens`, ties ONLY its 4-byte `dy_scale`
  instead of copying `536.9 MB` of token cotangents!
  """
  return a_tree, b_tree


def _bwd_barrier_tie_fwd(a_tree, b_tree):
  return (a_tree, b_tree), None


def _bwd_barrier_tie_bwd(_, cotangents):
  g_a, g_b = cotangents
  b_terms: list[jax.Array] = []
  for leaf in jax.tree_util.tree_leaves(g_b, is_leaf=_is_custom_token_carrier):
    if isinstance(leaf, Fp8DispatchedTokens):
      if _is_tieable_grad_leaf(leaf.ste_carrier):
        b_terms.append((first_scalar(leaf.ste_carrier) * 0.0).astype(jnp.float32))
    elif isinstance(leaf, MoePartialTokens):
      b_terms.append(
          (first_scalar(leaf.dy_codes).astype(jnp.float32) * 0.0)
          + (first_scalar(leaf.dy_scale).astype(jnp.float32) * 0.0)
      )
    elif _is_tieable_grad_leaf(leaf):
      b_terms.append((first_scalar(leaf) * 0.0).astype(jnp.float32))
  if not b_terms:
    return g_a, g_b
  zero_dep = sum(b_terms)

  def _tie_a_leaf(v: Any) -> Any:
    if isinstance(v, Fp8DispatchedTokens):
      return Fp8DispatchedTokens(
          ste_carrier=v.ste_carrier + zero_dep.astype(v.ste_carrier.dtype),
          codes_ep=v.codes_ep,
          scale=v.scale,
      )
    if isinstance(v, MoePartialTokens):
      return MoePartialTokens(
          y_ep=v.y_ep,
          dy_codes=v.dy_codes,
          dy_scale=v.dy_scale + zero_dep.astype(v.dy_scale.dtype),
      )
    if _is_tieable_grad_leaf(v):
      return v + zero_dep.astype(v.dtype)
    return v

  g_a_tied = jax.tree_util.tree_map(_tie_a_leaf, g_a, is_leaf=_is_custom_token_carrier)
  return g_a_tied, g_b


backward_barrier_tie.defvjp(_bwd_barrier_tie_fwd, _bwd_barrier_tie_bwd)


@jax.custom_vjp
def tie_bwd_after_partial_dy_scale(a: jax.Array, partial_in: Any):
  """No-op in forward; in backward, orders `g_a` (`B1(j)`) after `g_partial.dy_scale` (`AR_1`).

  Prevents XLA `AllReduceCombiner` from sinking `mb1(j-1)`'s early `dy` `pmax`
  (`AR_1`)
  past `B1(j)` to merge with `mb0(j-1)`'s `dy` `pmax` (`AR_2`), allowing
  `dy_codes_ep`
  (`f8e5m2[32768, 64, 128]`) to launch on SC0 BEFORE `B1(j)` and overlap `B1(j)`
  +
  `dwi/dwo` RS without making `B1(j)` wait on `dy_codes_ep` `call-done`!
  """
  return a, partial_in


def _tie_bwd_dy_scale_fwd(a: jax.Array, partial_in: Any):
  return (a, partial_in), None


def _tie_bwd_dy_scale_bwd(_, cotangents):
  g_a, g_partial = cotangents
  if isinstance(g_partial, MoePartialTokens) and _is_tieable_grad_leaf(g_a):
    z = (first_scalar(g_partial.dy_scale).astype(jnp.float32) * 0.0).astype(g_a.dtype)
    return g_a + z, g_partial
  return g_a, g_partial


tie_bwd_after_partial_dy_scale.defvjp(_tie_bwd_dy_scale_fwd, _tie_bwd_dy_scale_bwd)


# ---------------------------------------------------------------------------
# FP8 row quantization & SC0-pinned EP collectives
# ---------------------------------------------------------------------------


def quantize_rows_fp8(x: jax.Array, qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn) -> tuple[jax.Array, jax.Array]:
  """Per-token-row dynamic FP8 quantization for 2D `[T, D]` or 3D `[T, D0, 128]`."""
  qmax = float(jnp.finfo(qtype).max)
  red_axes = tuple(range(1, x.ndim))
  absmax = jnp.max(jnp.abs(x), axis=red_axes, keepdims=True).astype(jnp.float32)
  scale = jnp.where(absmax > 0.0, absmax / qmax, jnp.ones_like(absmax))
  codes = jnp.clip(x.astype(jnp.float32) / scale, -qmax, qmax).astype(qtype)
  return codes, scale


def quantize_cols_fp8_for_tgmm(
    x: jax.Array, qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn
) -> tuple[jax.Array, jax.Array]:
  """Per-output-channel (`N`) FP8 quantization for 2D `[M, N]` or 3D `[M, N0, 128]` `tgmm_v2` RHS.

  Preserves 3D `[M, N0, 128]` layout without relayout and returns `rhs_scale` of
  shape `(1, 1, N)` matching `tgmm_v2`'s per-N scale contract.
  """
  qmax = float(jnp.finfo(qtype).max)
  x_f32 = x.astype(jnp.float32)
  absmax = jnp.max(jnp.abs(x_f32), axis=0)
  scale = jnp.where(absmax > 0.0, absmax / qmax, jnp.ones_like(absmax))
  codes = jnp.clip(x_f32 / scale[None, ...], -qmax, qmax).astype(qtype)
  return codes, scale.reshape(1, 1, -1)


def dequantize_rows_fp8(codes: jax.Array, scale: jax.Array, dtype: jax.typing.DTypeLike) -> jax.Array:
  """Inverse of `quantize_rows_fp8`."""
  return (codes.astype(jnp.float32) * scale).astype(dtype)


def ep_axis_index(cfg: Qwen35Config) -> jax.Array:
  """Returns flat EP rank `0 ..

  cfg.ep_size - 1` matching `(*cfg.ep_axis_tuple, 'fsdp')`.
  """
  if cfg.decouple_ep_d2d:
    return jax.lax.axis_index("ep_d2d") * cfg.ep_ici_size + jax.lax.axis_index("expert")
  return jax.lax.axis_index("expert")


def ep_token_dispatch(tokens_local: jax.Array, cfg: Qwen35Config) -> jax.Array:
  """SC0-pinned EP token all-gather (FP8 forward option, bf16 SC0 RS backward)."""
  core_id = _ep_core_id(cfg)
  use_fp8 = cfg.moe_fp8_token_dispatch
  ep_ax = cfg.ep_axis_names
  min_sc = cfg.sc_min_collective_bytes

  @jax.custom_vjp
  def _dispatch(z: jax.Array) -> jax.Array:
    if use_fp8:
      qval, scale = quantize_rows_fp8(z, jnp.float8_e4m3fn)
      qval_ep = sc_all_gather(qval, ep_ax, core_id=core_id, axis=0, min_sc_bytes=min_sc)
      scale_ep = jax.lax.all_gather(scale, axis_name=ep_ax, axis=0, tiled=True)
      return dequantize_rows_fp8(qval_ep, scale_ep, z.dtype)
    return sc_all_gather(z, ep_ax, core_id=core_id, axis=0, min_sc_bytes=min_sc)

  def _dispatch_fwd(z: jax.Array):
    return _dispatch(z), None

  def _dispatch_bwd(_, g_ep: jax.Array):
    return (
        sc_psum_scatter(
            g_ep,
            ep_ax,
            core_id=core_id,
            scatter_dimension=0,
            min_sc_bytes=min_sc,
        ),
    )

  _dispatch.defvjp(_dispatch_fwd, _dispatch_bwd)
  return _dispatch(tokens_local)


def _ep_scalar_pmax(amax_local: jax.Array, cfg: Qwen35Config) -> jax.Array:
  """Computes `pmax` over EP axes without cross-microbatch/phase combining."""
  ep_ax = cfg.ep_axis_names
  if cfg.fp8_amax_no_combine:
    # `TpuAllReduceCombiner` skips `all-reduce`s carrying `_scheduling_group_id`
    # while `LegalizeSchedulingAnnotations` treats `"noop"` as a pure no-op!
    with set_xla_metadata(_scheduling_group_id="noop"):
      return jax.lax.pmax(amax_local, axis_name=ep_ax)
  return jax.lax.pmax(amax_local, axis_name=ep_ax)


def ep_token_dispatch_3d_fp8(tokens_local: jax.Array, cfg: Qwen35Config) -> Fp8DispatchedTokens:
  """SC0-pinned 3D FP8 EP token all-gather keeping tokens in `float8_e4m3fn` through `wi`!

  - Forward: computes per-EP-group dynamic `scale` (`[1, 1]` fp32 via 4-byte TC
  ICI `pmax`),
    quantizes local `[T_local, 64, 128]` tokens to `float8_e4m3fn`, all-gathers
    `codes_ep`
    (`[T_ep, 64, 128]`, `268.4 MB`) on **SC0**, and skips `dequantize_rows_fp8`
    completely!
  - Backward: receives `Fp8DispatchedTokens(ste_carrier=dx_ep, ...)` and runs
  SC0
    `psum_scatter` on `dx_ep`.
  """
  core_id = _ep_core_id(cfg)
  ep_ax = cfg.ep_axis_names
  min_sc = cfg.sc_min_collective_bytes
  ep_shape = (tokens_local.shape[0] * cfg.ep_size, *tokens_local.shape[1:])
  qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)

  @jax.custom_vjp
  def _dispatch_fp8(z: jax.Array) -> Fp8DispatchedTokens:
    amax_local = jax.lax.stop_gradient(jnp.max(jnp.abs(z.astype(jnp.float32))))
    amax_ep = _ep_scalar_pmax(amax_local, cfg).reshape(1, 1)
    scale = jax.lax.stop_gradient(jnp.where(amax_ep > 0.0, amax_ep / qmax, jnp.ones_like(amax_ep)))
    inv_scale = (1.0 / scale).reshape(1, *([1] * (z.ndim - 1)))
    codes_l = jax.lax.stop_gradient(jnp.clip(z.astype(jnp.float32) * inv_scale, -qmax, qmax).astype(jnp.float8_e4m3fn))
    codes_ep = pin_row_major(
        sc_all_gather(
            pin_row_major(codes_l),
            ep_ax,
            core_id=core_id,
            axis=0,
            min_sc_bytes=min_sc,
        )
    )
    ste = jax.lax.broadcast(jnp.zeros((), dtype=z.dtype), ep_shape)
    return Fp8DispatchedTokens(ste_carrier=ste, codes_ep=codes_ep, scale=scale)

  def _dispatch_fp8_fwd(z: jax.Array):
    return _dispatch_fp8(z), None

  def _dispatch_fp8_bwd(_, g_disp: Fp8DispatchedTokens):
    return (
        pin_row_major(
            sc_psum_scatter(
                pin_row_major(g_disp.ste_carrier),
                ep_ax,
                core_id=core_id,
                scatter_dimension=0,
                min_sc_bytes=min_sc,
            )
        ),
    )

  _dispatch_fp8.defvjp(_dispatch_fp8_fwd, _dispatch_fp8_bwd)
  return _dispatch_fp8(tokens_local)


def ep_routing_weights_dispatch(topk_weights_local: jax.Array, cfg: Qwen35Config) -> jax.Array:
  """TensorCore ICI EP all-gather of `topk_weights` (optional SC backward RS)."""
  ep_ax = cfg.ep_axis_names
  ep_size = cfg.ep_size
  min_sc = cfg.sc_min_collective_bytes
  bwd_core_id: int | None = None
  if cfg.use_pallas_kernels:
    if cfg.ep_routing_weights_bwd_sc == "sc1":
      bwd_core_id = 1
    elif cfg.ep_routing_weights_bwd_sc == "sc0":
      bwd_core_id = 0

  @jax.custom_vjp
  def _ag_w(w: jax.Array) -> jax.Array:
    return jax.lax.all_gather(w, axis_name=ep_ax, axis=0, tiled=True)

  def _ag_w_fwd(w: jax.Array):
    return _ag_w(w), None

  def _ag_w_bwd(_, gw_ep: jax.Array):
    if cfg.ep_routing_weights_bwd_sc == "none":
      return (jax.lax.psum_scatter(gw_ep, axis_name=ep_ax, scatter_dimension=0, tiled=True),)
    t_local = gw_ep.shape[0] // ep_size
    k = gw_ep.shape[1]
    local_elems = t_local * k
    if local_elems % 1024 == 0:
      m_local = local_elems // 128
      gw_aligned = pin_row_major(gw_ep.reshape(ep_size * m_local, 128))
      out_aligned = pin_row_major(
          sc_psum_scatter(
              gw_aligned,
              ep_ax,
              core_id=bwd_core_id,
              scatter_dimension=0,
              min_sc_bytes=min_sc,
          )
      )
      return (out_aligned.reshape(t_local, k),)
    return (
        pin_row_major(
            sc_psum_scatter(
                pin_row_major(gw_ep),
                ep_ax,
                core_id=bwd_core_id,
                scatter_dimension=0,
                min_sc_bytes=min_sc,
            )
        ),
    )

  _ag_w.defvjp(_ag_w_fwd, _ag_w_bwd)
  return _ag_w(topk_weights_local)


def ep_routing_ids_dispatch(topk_ids_local: jax.Array, cfg: Qwen35Config) -> jax.Array:
  """Non-differentiable TensorCore ICI EP all-gather of integer `topk_ids` (`1.31-2.62 MB`, zero SC0 contention)."""
  ids_i32 = jax.lax.stop_gradient(topk_ids_local.astype(jnp.int32))
  return jax.lax.all_gather(ids_i32, axis_name=cfg.ep_axis_names, axis=0, tiled=True)


def ep_token_combine_bwd_only(
    g_local_shaped: jax.Array,
    cfg: Qwen35Config,
    *,
    as_partial_tokens: bool = False,
) -> jax.Array | MoePartialTokens:
  """Exact transpose of `ep_token_combine` (`[B_mb, S, D]` or 3D -> `[T_ep, D]` or 3D) pinned to SC0."""
  core_id = _ep_core_id(cfg)
  ep_ax = cfg.ep_axis_names
  min_sc = cfg.sc_min_collective_bytes
  emb_dim = cfg.emb_dim
  use_3d = cfg.moe_use_tc_ragged and cfg.moe_3d_ep_tokens and (emb_dim % 128 == 0)
  g_in = g_local_shaped.reshape(-1, emb_dim // 128, 128) if use_3d else g_local_shaped.reshape(-1, emb_dim)
  if as_partial_tokens:
    qtype = jnp.float8_e5m2
    qmax = float(jnp.finfo(qtype).max)
    amax_l = jax.lax.stop_gradient(jnp.max(jnp.abs(g_in.astype(jnp.float32))))
    amax_ep = _ep_scalar_pmax(amax_l, cfg).reshape(1, 1)
    dy_scale = jax.lax.stop_gradient(jnp.where(amax_ep > 0.0, amax_ep / qmax, jnp.ones_like(amax_ep)))
    inv_scale = (1.0 / dy_scale).reshape(1, *([1] * (g_in.ndim - 1)))
    dy_codes_l = jnp.clip(g_in.astype(jnp.float32) * inv_scale, -qmax, qmax).astype(qtype)
    dy_codes_ep = pin_row_major(
        sc_all_gather(
            pin_row_major(dy_codes_l),
            ep_ax,
            core_id=core_id,
            axis=0,
            min_sc_bytes=min_sc,
        )
    )
    ep_shape = (g_in.shape[0] * cfg.ep_size, *g_in.shape[1:])
    y_zero = jax.lax.broadcast(jnp.zeros((), dtype=g_in.dtype), ep_shape)
    return MoePartialTokens(y_ep=y_zero, dy_codes=dy_codes_ep, dy_scale=dy_scale)

  if cfg.moe_fp8_bwd_dispatch:
    if cfg.moe_use_forked_3d_gmmv2 and cfg.moe_bwd_dy_fp8_e5m2:
      qtype = jnp.float8_e5m2
      qmax = float(jnp.finfo(qtype).max)
      amax_l = jax.lax.stop_gradient(jnp.max(jnp.abs(g_in.astype(jnp.float32))))
      amax_ep = _ep_scalar_pmax(amax_l, cfg).reshape(1, 1)
      dy_scale = jax.lax.stop_gradient(jnp.where(amax_ep > 0.0, amax_ep / qmax, jnp.ones_like(amax_ep)))
      inv_scale = (1.0 / dy_scale).reshape(1, *([1] * (g_in.ndim - 1)))
      dy_codes_l = jnp.clip(g_in.astype(jnp.float32) * inv_scale, -qmax, qmax).astype(qtype)
      dy_codes_ep = pin_row_major(
          sc_all_gather(
              pin_row_major(dy_codes_l),
              ep_ax,
              core_id=core_id,
              axis=0,
              min_sc_bytes=min_sc,
          )
      )
      return (dy_codes_ep.astype(jnp.float32) * dy_scale.reshape(1, *([1] * (g_in.ndim - 1)))).astype(g_in.dtype)
    qval, scale = quantize_rows_fp8(g_in, jnp.float8_e4m3fn)
    qval_ep = sc_all_gather(qval, ep_ax, core_id=core_id, axis=0, min_sc_bytes=min_sc)
    scale_ep = jax.lax.all_gather(scale, axis_name=ep_ax, axis=0, tiled=True)
    return dequantize_rows_fp8(qval_ep, scale_ep, g_in.dtype)
  return sc_all_gather(g_in, ep_ax, core_id=core_id, axis=0, min_sc_bytes=min_sc)


def combine_bwd_bridge(
    cfg: Qwen35Config,
    unperm_prev: jax.Array | MoePartialTokens,
    base_prev: jax.Array,
    x_mb1_val: jax.Array,
) -> jax.Array:
  """Zero-residual cross-layer combine bridge routing `dx_mb1` -> `(combine_bwd(dx_mb1), dx_mb1, 0)`."""
  as_pt = isinstance(unperm_prev, MoePartialTokens)

  @jax.custom_vjp
  def _bridge(u_prev, b_prev, x_val: jax.Array) -> jax.Array:
    del u_prev, b_prev
    return x_val

  def _bridge_fwd(u_prev, b_prev, x_val: jax.Array):
    del u_prev, b_prev
    return x_val, ()

  def _bridge_bwd(_, g_x_mb1: jax.Array):
    return (
        ep_token_combine_bwd_only(g_x_mb1, cfg, as_partial_tokens=as_pt),
        g_x_mb1,
        jnp.zeros_like(g_x_mb1),
    )

  _bridge.defvjp(_bridge_fwd, _bridge_bwd)
  return _bridge(unperm_prev, base_prev, x_mb1_val)


def ep_token_combine(partial_tokens_ep: jax.Array | MoePartialTokens, cfg: Qwen35Config) -> jax.Array:
  """SC0-pinned EP token combine `psum_scatter` (`[EP * T_local, 32, 128] -> [T_local, 32, 128]`, 33.5 MB)."""
  fwd_core_id = None if cfg.ep_combine_fwd_unpinned else _ep_core_id(cfg)
  ep_ax = cfg.ep_axis_names
  min_sc = cfg.sc_min_collective_bytes
  as_pt = isinstance(partial_tokens_ep, MoePartialTokens)

  @jax.custom_vjp
  def _combine(p_ep: jax.Array | MoePartialTokens) -> jax.Array:
    y_ep = p_ep.y_ep if isinstance(p_ep, MoePartialTokens) else p_ep
    return pin_row_major(
        sc_psum_scatter(
            pin_row_major(y_ep),
            ep_ax,
            core_id=fwd_core_id,
            scatter_dimension=0,
            min_sc_bytes=min_sc,
        )
    )

  def _combine_fwd(p_ep: jax.Array | MoePartialTokens):
    return _combine(p_ep), None

  def _combine_bwd(_, g_local: jax.Array):
    return (ep_token_combine_bwd_only(g_local, cfg, as_partial_tokens=as_pt),)

  _combine.defvjp(_combine_fwd, _combine_bwd)
  return _combine(partial_tokens_ep)


# ---------------------------------------------------------------------------
# SC1-pinned FSDP dense weight gather & gradient reduction
# ---------------------------------------------------------------------------


def gather_dense_weight(
    w_sharded: jax.Array,
    cfg: Qwen35Config,
    *,
    sharded_axis: int = 0,
    is_fsdp_expert: bool = False,
    start_after: jax.Array | None = None,
) -> jax.Array:
  """Gathers a dense weight on SC1 (`fsdp`) with SC1 `psum_scatter` backward.

  When `is_fsdp_expert=True` (`512`-way sharded over `all_sharded_axes`),
  performs a lineage-style 2-stage decoupled gather/reduce with BOTH stages
  pinned to **SC1** (`core_id=ag_core` / `rs_core`) so **SC0** remains 100%
  exclusively dedicated to EP token/routing dispatch & combine!
  When `cfg.decouple_ep_d2d=True`, Stage 1 spans `'expert'` (`z=8` ICI) and
  Stage 2 spans `('fsdp', 'ep_d2d')` (`64`-way `y,x,core`,
  `chiplet_scale_factor=1`).
  """
  ag_core = _fsdp_ag_core_id(cfg, is_bwd_remat=False)
  rs_core = _dense_rs_core_id(cfg)
  s2_ax = cfg.dense_stage2_axis_names
  min_sc = cfg.sc_min_collective_bytes
  w_in = order_after(w_sharded, start_after) if start_after is not None else w_sharded

  def _do_gather(w: jax.Array, sc1_core: int | None) -> jax.Array:
    if is_fsdp_expert:
      # Major axis in all_sharded_axes is s2_ax ('fsdp' or ('fsdp', 'ep_d2d')),
      # minor is 'expert'. Gathering minor ('expert') first yields contiguous
      # per-Stage-2 blocks, then gathering s2_ax reconstructs the global axis!
      w_ep = sc_all_gather(w, "expert", core_id=sc1_core, axis=sharded_axis, min_sc_bytes=min_sc)
      return sc_all_gather(w_ep, s2_ax, core_id=sc1_core, axis=sharded_axis, min_sc_bytes=min_sc)
    return sc_all_gather(w, s2_ax, core_id=sc1_core, axis=sharded_axis, min_sc_bytes=min_sc)

  @jax.custom_vjp
  def _gather_fn(w: jax.Array) -> jax.Array:
    return _do_gather(w, ag_core)

  def _gather_fwd(w: jax.Array):
    return _do_gather(w, ag_core), None

  def _gather_bwd(_, gw_full: jax.Array):
    # Reduce over major s2_ax on SC1 first, then minor ('expert') on SC1 / TC ICI.
    gw_fsdp = sc_psum_scatter(
        gw_full,
        s2_ax,
        core_id=rs_core,
        scatter_dimension=sharded_axis,
        min_sc_bytes=min_sc,
    )
    if is_fsdp_expert:
      gw_sharded = sc_psum_scatter(
          gw_fsdp,
          "expert",
          core_id=rs_core,
          scatter_dimension=sharded_axis,
          min_sc_bytes=min_sc,
      )
      return (gw_sharded,)
    gw_sharded = sc_psum(gw_fsdp, "expert", core_id=rs_core, min_sc_bytes=min_sc)
    return (gw_sharded,)

  _gather_fn.defvjp(_gather_fwd, _gather_bwd)
  return _gather_fn(w_in)


def gather_or_replicate_dense_weight(
    w: jax.Array,
    cfg: Qwen35Config,
    *,
    param_name: str,
    sharded_axis: int = 0,
    start_after: jax.Array | None = None,
) -> jax.Array:
  """Dispatches per-parameter sharding tier (`replicated` / `fsdp_only` / `fsdp_expert`)."""
  kind = cfg.dense_param_sharding_kind(param_name)
  if kind == "replicated":
    return replicate_small_param(w, cfg)
  return gather_dense_weight(
      w,
      cfg,
      sharded_axis=sharded_axis,
      is_fsdp_expert=(kind == "fsdp_expert"),
      start_after=start_after,
  )


def replicate_small_param(w_rep: jax.Array, cfg: Qwen35Config | None = None) -> jax.Array:
  """Identity in forward; all-reduces replicated parameter gradient over `all_sharded_axes`."""
  axes = cfg.all_sharded_axes if cfg is not None else ("fsdp", "expert")

  @jax.custom_vjp
  def _rep(w: jax.Array) -> jax.Array:
    return w

  def _rep_fwd(w: jax.Array):
    return w, None

  def _rep_bwd(_, gw: jax.Array):
    gw_sum = jax.lax.psum(gw, axis_name=axes)
    return (gw_sum.astype(gw.dtype),)

  _rep.defvjp(_rep_fwd, _rep_bwd)
  return _rep(w_rep)


# ---------------------------------------------------------------------------
# Qwen3Next Layer Normalizations (Pure JAX)
# ---------------------------------------------------------------------------


def rms_norm_zero_centered(
    x: jax.Array,
    scale: jax.Array,
    eps: float = 1e-6,
    *,
    use_pallas: bool = False,
) -> jax.Array:
  """Qwen3NextRMSNorm: `(x / rms(x)) * (1.0 + scale)` accumulated in fp32."""
  if use_pallas:
    return rmsnorm_tc.rms_norm_zero_centered_pallas(x, scale, eps)
  orig_dtype = x.dtype
  x_f32 = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
  normed = x_f32 * inv_rms * (1.0 + scale.astype(jnp.float32))
  return normed.astype(orig_dtype)


def rms_norm_gated(
    x: jax.Array,
    gate_z: jax.Array,
    scale: jax.Array,
    eps: float = 1e-6,
    *,
    use_pallas: bool = False,
) -> jax.Array:
  """Qwen3NextRMSNormGated: `rms_norm(x, scale) * silu(gate_z)` in fp32."""
  if use_pallas:
    return rmsnorm_tc.rms_norm_gated_pallas(x, gate_z, scale, eps)
  orig_dtype = x.dtype
  x_f32 = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
  normed = x_f32 * inv_rms * scale.astype(jnp.float32)
  gated = normed * jax.nn.silu(gate_z.astype(jnp.float32))
  return gated.astype(orig_dtype)
