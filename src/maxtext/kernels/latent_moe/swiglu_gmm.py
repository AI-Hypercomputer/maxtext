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

"""Ragged grouped matmuls with a fused SwiGLU epilogue for MoE expert FFNs.

The routed expert FFN of a (latent) MoE layer is

    a = x @ wi_0[e],  b = x @ wi_1[e],  h = silu(a) * b,  y = h @ wo[e]

for the rows `x` that are sorted by expert `e` (ragged `group_sizes`). The
unfused path runs three grouped matmuls (GMMs) plus an elementwise SwiGLU
fusion, and round-trips `a`, `b` and `h` (each `[M, d_ff]`) through HBM; its
backward adds another elementwise pass for the SwiGLU derivative.

`latent_expert_ffn` keeps `a` and `b` in VMEM:

  Forward
    1. `gate_up_swiglu`: one ragged GMM over (`wi_0`, `wi_1`) whose epilogue
       writes only `h = silu(a) * b` (bf16) to HBM.
    2. `y = h @ wo` with the Megablox ragged GMM.

  Backward (custom VJP, residuals are the primal inputs only)
    1. `swiglu_bwd`: per tile, recompute `a`, `b` (from `x`, `wi_0`, `wi_1`)
       and `dh = dy @ wo^T`, all in VMEM, and apply the SwiGLU derivative in
       the epilogue. Writes `[da; db]` (stacked) and `h`.
    2. `dx = da @ wi_0^T + db @ wi_1^T` as a single ragged GMM whose reduction
       runs over both halves of the stacked operand.
    3. `dwi_0`, `dwi_1` = one ragged transposed GMM (TGMM) over (`x`, `[da; db]`).
    4. `dwo` = TGMM over (`dy`, `h`).

All matmuls take bf16 (or the input dtype) operands and accumulate in fp32.

Rows past `sum(group_sizes)` (ragged-buffer padding) are never computed: like
the Megablox kernels, the corresponding output rows (`y`, `dx`) are left
unspecified and must be ignored by the caller. They never leak into the weight
gradients.
"""

import dataclasses
import functools
import math
import json
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from maxtext.kernels.megablox import backend as mblx_backend


Tile3 = tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class LatentFFNTiling:
  """Tile sizes (tm, tk, tn) for each kernel of `latent_expert_ffn`.

  m is the ragged (row) dimension. For every kernel k is the contracting
  dimension and n the output-column dimension:

    gate_up : x[M, D] @ wi_{0,1}[E, D, F]     -> h[M, F]       (k over D, n over F)
    wo_fwd  : h[M, F] @ wo[E, F, D]          -> y[M, D]       (k over F, n over D)
    swiglu_bwd : like gate_up                                  (k over D, n over F)
    dx      : [da; db][2, M, F] @ wi^T       -> dx[M, D]      (k over F, n over D)
    dwi     : x^T @ [da; db]                 -> dwi[E, 2, D, F] (k over D, n over F)
    dwo     : dy^T @ h                       -> dwo^T[E, D, F] (k over D, n over F)

  The first grid axis of every kernel walks the n tiles and is marked
  "parallel", so on chips with two TensorCores per device (TPU v4 MegaCore)
  an even number of n tiles keeps both cores busy.
  """

  gate_up: Tile3 = (512, 768, 896)
  wo_fwd: Tile3 = (512, 1792, 384)
  swiglu_bwd: Tile3 = (256, 768, 896)
  dx: Tile3 = (512, 1792, 384)
  dwi: Tile3 = (512, 768, 896)
  dwo: Tile3 = (512, 768, 896)
  vmem_limit_bytes: int | None = None
  # Feed the backward kernels pre-transposed weight copies (wo^T [E, D, F], wi^T [E, F, D]; a few MB each) so
  # every in-kernel matmul is a plain [m, k] @ [k, n] instead of a transposed-rhs one.
  pretranspose_weights: bool = False
  # TGMM contraction over rows: "swap" transposes the lhs tile explicitly, "dot_general" lets Mosaic contract
  # dim 0 of both operands.
  tgmm_lhs_mode: str = "swap"
  # swiglu_bwd weight-block pipeline buffers. 1 frees VMEM for a larger tm (e.g. (384, 768, 896) on a
  # 16 MiB-VMEM core); measured within 3% of the default on TPU v4.
  swiglu_bwd_weight_buffers: int = 2

  def clamp(self, m: int, d: int, f: int) -> "LatentFFNTiling":
    """Clamps every tile to the problem size (m must be a multiple of every tm)."""

    def c(t, km, kk, kn):
      return (min(t[0], km), min(t[1], kk), min(t[2], kn))

    return dataclasses.replace(
        self,
        gate_up=c(self.gate_up, m, d, f),
        wo_fwd=c(self.wo_fwd, m, f, d),
        swiglu_bwd=c(self.swiglu_bwd, m, d, f),
        dx=c(self.dx, m, f, d),
        dwi=c(self.dwi, m, d, f),
        dwo=c(self.dwo, m, d, f),
    )

  def row_multiple(self) -> int:
    tms = (self.gate_up[0], self.wo_fwd[0], self.swiglu_bwd[0], self.dx[0], self.dwi[0], self.dwo[0])
    return functools.reduce(math.lcm, tms)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _num_tiles(x: int, t: int, what: str) -> int:
  if x % t:
    raise ValueError(f"{what}={x} must be divisible by its tile size {t}.")
  return x // t


def _extract_vma(tensor: Any) -> tuple[str, ...]:
  """Varying manual axes of `tensor` inside shard_map (empty outside / without check_vma)."""
  type_str = str(jax.typeof(tensor))
  if "{V:" in type_str:
    start = type_str.index("{V:") + 3
    end = type_str.index("}", start)
    return tuple(sorted(a.strip() for a in type_str[start:end].strip("()").split(",") if a.strip()))
  return tuple()


def _sds(shape, dtype, *tensors) -> jax.ShapeDtypeStruct:
  vma = frozenset(ax for t in tensors for ax in _extract_vma(t))
  if vma:
    return jax.ShapeDtypeStruct(shape, dtype, manual_axis_type=jax.sharding.ManualAxisType(varying=vma))
  return jax.ShapeDtypeStruct(shape, dtype)


def _group_metadata(group_sizes, m, tm, group_offset, num_local_groups, visit_empty_groups):
  return mblx_backend.make_group_metadata(  # pylint: disable=unbalanced-tuple-unpacking
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      start_group=group_offset[0],
      num_nonzero_groups=num_local_groups,
      visit_empty_groups=visit_empty_groups,
  )


def _row_mask(grid_id, group_metadata, tm, width):
  """[tm, width] mask of the rows of the current m tile that belong to the current group."""
  return mblx_backend._get_store_mask(  # pylint: disable=protected-access
      grid_id=grid_id, group_metadata=group_metadata, tm=tm, tn=width
  )


def _compiler_params(semantics, vmem_limit_bytes):
  kwargs = {"dimension_semantics": semantics}
  if vmem_limit_bytes is not None:
    kwargs["vmem_limit_bytes"] = vmem_limit_bytes
  return pltpu.CompilerParams(**kwargs)


def _dot_t(lhs, rhs):
  """lhs[m, k] . rhs[n, k]^T -> [m, n] in fp32 (MXU transposed-rhs matmul)."""
  return lax.dot_general(lhs, rhs, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _silu_parts(a):
  sig = jax.nn.sigmoid(a)
  return sig, a * sig


# ------------------------------------------------------------------------------
# Forward: gate/up GMM with SwiGLU epilogue
# ------------------------------------------------------------------------------


def gate_up_swiglu(
    x: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    tiling: Tile3,
    *,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """h[M, F] = silu(x @ wi_0[g]) * (x @ wi_1[g]) per group, without writing a or b.

  Args:
    x: [M, D] rows sorted by group; M must be a multiple of tm.
    wi_0, wi_1: [E_local, D, F].
    group_sizes: int32[G] (G >= E_local); groups [group_offset, group_offset + E_local) are local.
    group_offset: int32[1].
    tiling: (tm, tk, tn).
    vmem_limit_bytes: Optional Mosaic scoped VMEM limit.
    interpret: Run the Pallas kernel in interpret mode (CPU tests).

  Returns:
    h: [M, F] in x.dtype. Rows outside the local groups are unspecified.
  """
  m, d = x.shape
  e_local, _, f = wi_0.shape
  tm, tk, tn = tiling
  tiles_k = _num_tiles(d, tk, "D")
  tiles_n = _num_tiles(f, tn, "F")
  _num_tiles(m, tm, "M")
  out_dtype = x.dtype
  group_metadata, num_active_tiles = _group_metadata(group_sizes, m, tm, group_offset, e_local, False)

  def kernel(group_metadata, group_offset, x_ref, w0_ref, w1_ref, h_ref, *scratch):
    del group_offset
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)
    xv = x_ref[...]
    pa = jnp.dot(xv, w0_ref[...], preferred_element_type=jnp.float32)
    pb = jnp.dot(xv, w1_ref[...], preferred_element_type=jnp.float32)

    def epilogue(a, b):
      _, silu = _silu_parts(a)
      h = (silu * b).astype(out_dtype)
      mask = _row_mask(grid_id, group_metadata, tm, tn)
      h_ref[...] = jnp.where(mask, h, h_ref[...])

    if tiles_k == 1:
      epilogue(pa, pb)
    else:
      acc_a, acc_b = scratch

      @pl.when(k_i == 0)
      def _():
        acc_a[...] = jnp.zeros_like(acc_a)
        acc_b[...] = jnp.zeros_like(acc_b)

      acc_a[...] += pa
      acc_b[...] += pb

      @pl.when(k_i == tiles_k - 1)
      def _():
        epilogue(acc_a[...], acc_b[...])

  def x_map(n_i, g_i, k_i, gm, go):
    del n_i, go
    return gm[2][g_i], k_i

  def w_map(n_i, g_i, k_i, gm, go):
    return gm[1][g_i] - go[0], k_i, n_i

  def h_map(n_i, g_i, k_i, gm, go):
    del k_i, go
    return gm[2][g_i], n_i

  scratch_shapes = [] if tiles_k == 1 else [pltpu.VMEM((tm, tn), jnp.float32)] * 2
  bytes_accessed = x.size * x.dtype.itemsize * tiles_n + 2 * wi_0.size * wi_0.dtype.itemsize + m * f * 2
  call = pl.pallas_call(
      kernel,
      out_shape=_sds((m, f), out_dtype, x, wi_0),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec((tm, tk), x_map),
              pl.BlockSpec((None, tk, tn), w_map),
              pl.BlockSpec((None, tk, tn), w_map),
          ],
          out_specs=pl.BlockSpec((tm, tn), h_map),
          grid=(tiles_n, num_active_tiles, tiles_k),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=_compiler_params(("parallel", "arbitrary", "arbitrary"), vmem_limit_bytes),
      interpret=interpret,
      cost_estimate=pl.CostEstimate(flops=4 * m * d * f, transcendentals=m * f, bytes_accessed=bytes_accessed),
      metadata={"xprof_metadata": json.dumps({"kernel": "gate_up_swiglu", "tiling": list(tiling)})},
  )
  return call(group_metadata, group_offset, x, wi_0, wi_1)


# ------------------------------------------------------------------------------
# Backward 1: recompute a, b and dh in VMEM, SwiGLU derivative epilogue
# ------------------------------------------------------------------------------


def swiglu_bwd(
    x: jax.Array,
    dy: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    tiling: Tile3,
    *,
    wo_transposed: bool = False,
    weight_buffers: int = 2,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Fused SwiGLU backward.

  Per tile computes, in VMEM and fp32,
    a = x @ wi_0, b = x @ wi_1, dh = dy @ wo^T,
    h = silu(a) * b, da = dh * b * silu'(a), db = dh * silu(a).

  Args:
    x: [M, D] forward input rows.
    dy: [M, D] cotangent of the FFN output.
    wi_0, wi_1: [E_local, D, F].
    wo: [E_local, F, D], or wo^T [E_local, D, F] if `wo_transposed`.
    group_sizes, group_offset: see `gate_up_swiglu`.
    tiling: (tm, tk, tn), k over D and n over F.
    wo_transposed: `wo` is given as wo^T.
    weight_buffers: Pipeline buffer count for the weight blocks (1 = single-buffered; weights only change
      at group / column-tile boundaries).
    vmem_limit_bytes: Optional Mosaic scoped VMEM limit.
    interpret: Pallas interpret mode.

  Returns:
    dadb: [2, M, F] (dadb[0] = da, dadb[1] = db) in x.dtype.
    h: [M, F] in x.dtype.
  """
  m, d = x.shape
  e_local, _, f = wi_0.shape
  tm, tk, tn = tiling
  tiles_k = _num_tiles(d, tk, "D")
  tiles_n = _num_tiles(f, tn, "F")
  _num_tiles(m, tm, "M")
  out_dtype = x.dtype
  group_metadata, num_active_tiles = _group_metadata(group_sizes, m, tm, group_offset, e_local, False)
  def kernel(group_metadata, group_offset, x_ref, dy_ref, w0_ref, w1_ref, wo_ref, dadb_ref, h_ref, *scratch):
    del group_offset
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)
    xv = x_ref[...]
    pa = jnp.dot(xv, w0_ref[...], preferred_element_type=jnp.float32)
    pb = jnp.dot(xv, w1_ref[...], preferred_element_type=jnp.float32)
    if wo_transposed:
      pdh = jnp.dot(dy_ref[...], wo_ref[...], preferred_element_type=jnp.float32)
    else:
      pdh = _dot_t(dy_ref[...], wo_ref[...])

    def epilogue(a, b, dh):
      sig, silu = _silu_parts(a)
      h = silu * b
      db = dh * silu
      da = dh * b * (sig * (1.0 + a * (1.0 - sig)))
      mask = _row_mask(grid_id, group_metadata, tm, tn)
      dadb_ref[0] = jnp.where(mask, da.astype(out_dtype), dadb_ref[0])
      dadb_ref[1] = jnp.where(mask, db.astype(out_dtype), dadb_ref[1])
      h_ref[...] = jnp.where(mask, h.astype(out_dtype), h_ref[...])

    if tiles_k == 1:
      epilogue(pa, pb, pdh)
    else:
      acc_a, acc_b, acc_dh = scratch

      @pl.when(k_i == 0)
      def _():
        acc_a[...] = jnp.zeros_like(acc_a)
        acc_b[...] = jnp.zeros_like(acc_b)
        acc_dh[...] = jnp.zeros_like(acc_dh)

      acc_a[...] += pa
      acc_b[...] += pb
      acc_dh[...] += pdh

      @pl.when(k_i == tiles_k - 1)
      def _():
        epilogue(acc_a[...], acc_b[...], acc_dh[...])

  def row_map(n_i, g_i, k_i, gm, go):
    del n_i, go
    return gm[2][g_i], k_i

  def w_map(n_i, g_i, k_i, gm, go):
    return gm[1][g_i] - go[0], k_i, n_i

  def wo_map(n_i, g_i, k_i, gm, go):
    return gm[1][g_i] - go[0], n_i, k_i

  def dadb_map(n_i, g_i, k_i, gm, go):
    del k_i, go
    return 0, gm[2][g_i], n_i

  def h_map(n_i, g_i, k_i, gm, go):
    del k_i, go
    return gm[2][g_i], n_i

  scratch_shapes = [] if tiles_k == 1 else [pltpu.VMEM((tm, tn), jnp.float32)] * 3
  w_kw = {} if weight_buffers == 2 else {"pipeline_mode": pl.Buffered(weight_buffers)}
  isz = x.dtype.itemsize
  bytes_accessed = 2 * m * d * isz * tiles_n + 3 * wi_0.size * wi_0.dtype.itemsize + 3 * m * f * isz
  call = pl.pallas_call(
      kernel,
      out_shape=(
          _sds((2, m, f), out_dtype, x, dy, wi_0),
          _sds((m, f), out_dtype, x, dy, wi_0),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec((tm, tk), row_map),
              pl.BlockSpec((tm, tk), row_map),
              pl.BlockSpec((None, tk, tn), w_map, **w_kw),
              pl.BlockSpec((None, tk, tn), w_map, **w_kw),
              pl.BlockSpec((None, tk, tn), w_map, **w_kw)
              if wo_transposed
              else pl.BlockSpec((None, tn, tk), wo_map, **w_kw),
          ],
          out_specs=(
              pl.BlockSpec((2, tm, tn), dadb_map),
              pl.BlockSpec((tm, tn), h_map),
          ),
          grid=(tiles_n, num_active_tiles, tiles_k),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=_compiler_params(("parallel", "arbitrary", "arbitrary"), vmem_limit_bytes),
      interpret=interpret,
      cost_estimate=pl.CostEstimate(flops=6 * m * d * f, transcendentals=m * f, bytes_accessed=bytes_accessed),
      metadata={"xprof_metadata": json.dumps({"kernel": "swiglu_bwd", "tiling": list(tiling)})},
  )
  return call(group_metadata, group_offset, x, dy, wi_0, wi_1, wo)


# ------------------------------------------------------------------------------
# Backward 2: dx = da @ wi_0^T + db @ wi_1^T
# ------------------------------------------------------------------------------


def stacked_dlhs(
    dadb: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    tiling: Tile3,
    *,
    out_dtype: jnp.dtype,
    w_transposed: bool = False,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """dx[M, D] = dadb[0] @ wi_0[g]^T + dadb[1] @ wi_1[g]^T, one fp32 accumulator over both halves.

  Args:
    dadb: [2, M, F].
    wi_0, wi_1: [E_local, D, F] (used as transposed right-hand sides; no copy), or wi^T [E_local, F, D] if
      `w_transposed`.
    group_sizes, group_offset: see `gate_up_swiglu`.
    tiling: (tm, tk, tn), k over F and n over D.
    out_dtype: dtype of dx.
    w_transposed: weights are given as wi^T.
    vmem_limit_bytes: Optional Mosaic scoped VMEM limit.
    interpret: Pallas interpret mode.

  Returns:
    dx: [M, D]. Rows outside the local groups are unspecified.
  """
  _, m, f = dadb.shape
  e_local = wi_0.shape[0]
  d = wi_0.shape[2] if w_transposed else wi_0.shape[1]
  tm, tk, tn = tiling
  tiles_kh = _num_tiles(f, tk, "F")
  tiles_n = _num_tiles(d, tn, "D")
  _num_tiles(m, tm, "M")
  group_metadata, num_active_tiles = _group_metadata(group_sizes, m, tm, group_offset, e_local, False)

  def kernel(group_metadata, group_offset, dadb_ref, w0_ref, w1_ref, out_ref, acc_ref):
    del group_offset
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    @pl.when(k_i == 0)
    def _():
      acc_ref[...] = jnp.zeros_like(acc_ref)

    def mm(lhs, rhs):
      if w_transposed:
        return jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
      return _dot_t(lhs, rhs)

    @pl.when(k_i < tiles_kh)
    def _():
      acc_ref[...] += mm(dadb_ref[...], w0_ref[...])

    @pl.when(k_i >= tiles_kh)
    def _():
      acc_ref[...] += mm(dadb_ref[...], w1_ref[...])

    @pl.when(k_i == 2 * tiles_kh - 1)
    def _():
      mask = _row_mask(grid_id, group_metadata, tm, tn)
      out_ref[...] = jnp.where(mask, acc_ref[...].astype(out_dtype), out_ref[...])

  def lhs_map(n_i, g_i, k_i, gm, go):
    del n_i, go
    return k_i // tiles_kh, gm[2][g_i], k_i % tiles_kh

  # Each weight's block index is clamped while the other half is being reduced, so the
  # Pallas pipeline does not refetch it.
  def w0_map(n_i, g_i, k_i, gm, go):
    k_blk = jnp.minimum(k_i, tiles_kh - 1)
    return (gm[1][g_i] - go[0], k_blk, n_i) if w_transposed else (gm[1][g_i] - go[0], n_i, k_blk)

  def w1_map(n_i, g_i, k_i, gm, go):
    k_blk = jnp.maximum(k_i - tiles_kh, 0)
    return (gm[1][g_i] - go[0], k_blk, n_i) if w_transposed else (gm[1][g_i] - go[0], n_i, k_blk)

  w_block = (None, tk, tn) if w_transposed else (None, tn, tk)

  def out_map(n_i, g_i, k_i, gm, go):
    del k_i, go
    return gm[2][g_i], n_i

  isz = dadb.dtype.itemsize
  bytes_accessed = dadb.size * isz * tiles_n + 2 * wi_0.size * wi_0.dtype.itemsize + m * d * jnp.dtype(out_dtype).itemsize
  call = pl.pallas_call(
      kernel,
      out_shape=_sds((m, d), out_dtype, dadb, wi_0),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec((None, tm, tk), lhs_map),
              pl.BlockSpec(w_block, w0_map),
              pl.BlockSpec(w_block, w1_map),
          ],
          out_specs=pl.BlockSpec((tm, tn), out_map),
          grid=(tiles_n, num_active_tiles, 2 * tiles_kh),
          scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
      ),
      compiler_params=_compiler_params(("parallel", "arbitrary", "arbitrary"), vmem_limit_bytes),
      interpret=interpret,
      cost_estimate=pl.CostEstimate(flops=4 * m * d * f, transcendentals=0, bytes_accessed=bytes_accessed),
      metadata={"xprof_metadata": json.dumps({"kernel": "stacked_dlhs", "tiling": list(tiling)})},
  )
  return call(group_metadata, group_offset, dadb, wi_0, wi_1)


# ------------------------------------------------------------------------------
# Backward 3: weight gradients, out[g, s] = lhs[g rows]^T @ rhs[s][g rows]
# ------------------------------------------------------------------------------


def stacked_tgmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    num_local_groups: int,
    tiling: Tile3,
    *,
    out_dtype: jnp.dtype,
    lhs_mode: str = "swap",
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """out[e, s] = sum_{rows r in group e} lhs[r]^T rhs[s, r] for each local group e and stack index s.

  Args:
    lhs: [M, K].
    rhs: [S, M, N].
    group_sizes, group_offset: see `gate_up_swiglu`.
    num_local_groups: E_local.
    tiling: (tm, tk, tn).
    out_dtype: output dtype (fp32 accumulation inside).
    lhs_mode: "swap" (explicit lhs tile transpose) or "dot_general" (contract dim 0 of both tiles).
    vmem_limit_bytes: Optional Mosaic scoped VMEM limit.
    interpret: Pallas interpret mode.

  Returns:
    [E_local, S, K, N]. Empty groups get zeros.
  """
  m, k = lhs.shape
  s, _, n = rhs.shape
  tm, tk, tn = tiling
  tiles_k = _num_tiles(k, tk, "K")
  tiles_nh = _num_tiles(n, tn, "N")
  _num_tiles(m, tm, "M")
  group_metadata, num_active_tiles = _group_metadata(group_sizes, m, tm, group_offset, num_local_groups, True)

  def kernel(group_metadata, group_offset, lhs_ref, rhs_ref, out_ref, acc_ref):
    del group_offset
    grid_id = pl.program_id(2)
    group_offsets, group_ids, _ = group_metadata
    group = group_ids[grid_id]
    prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
    group_has_changed = jnp.logical_or(grid_id == 0, group_ids[prev_grid_id] != group)

    @pl.when(group_has_changed)
    def _():
      acc_ref[...] = jnp.zeros_like(acc_ref)

    group_size = group_offsets[group + 1] - group_offsets[group]

    @pl.when(group_size > 0)
    def _():
      lhs_v = lhs_ref[...]
      rhs_v = rhs_ref[...]
      lhs_v = jnp.where(_row_mask(grid_id, group_metadata, tm, tk), lhs_v, jnp.zeros_like(lhs_v))
      rhs_v = jnp.where(_row_mask(grid_id, group_metadata, tm, tn), rhs_v, jnp.zeros_like(rhs_v))
      if lhs_mode == "dot_general":
        acc_ref[...] += lax.dot_general(lhs_v, rhs_v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
      else:
        acc_ref[...] += jnp.dot(lhs_v.swapaxes(0, 1), rhs_v, preferred_element_type=jnp.float32)

    is_end = grid_id == pl.num_programs(2) - 1
    next_grid_id = jnp.where(is_end, grid_id, grid_id + 1)
    group_is_changing = jnp.logical_or(is_end, group_ids[next_grid_id] != group)

    @pl.when(group_is_changing)
    def _():
      out_ref[...] = acc_ref[...].astype(out_dtype)

  def lhs_map(j_i, k_i, g_i, gm, go):
    del j_i, go
    return gm[2][g_i], k_i

  def rhs_map(j_i, k_i, g_i, gm, go):
    del k_i, go
    return j_i // tiles_nh, gm[2][g_i], j_i % tiles_nh

  def out_map(j_i, k_i, g_i, gm, go):
    return gm[1][g_i] - go[0], j_i // tiles_nh, k_i, j_i % tiles_nh

  bytes_accessed = (
      lhs.size * lhs.dtype.itemsize * s * tiles_nh
      + rhs.size * rhs.dtype.itemsize * tiles_k
      + num_local_groups * s * k * n * jnp.dtype(out_dtype).itemsize
  )
  call = pl.pallas_call(
      kernel,
      out_shape=_sds((num_local_groups, s, k, n), out_dtype, lhs, rhs),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec((tm, tk), lhs_map),
              pl.BlockSpec((None, tm, tn), rhs_map),
          ],
          out_specs=pl.BlockSpec((None, None, tk, tn), out_map),
          grid=(s * tiles_nh, tiles_k, num_active_tiles),
          scratch_shapes=[pltpu.VMEM((tk, tn), jnp.float32)],
      ),
      compiler_params=_compiler_params(("parallel", "arbitrary", "arbitrary"), vmem_limit_bytes),
      interpret=interpret,
      cost_estimate=pl.CostEstimate(flops=2 * m * k * n * s, transcendentals=0, bytes_accessed=bytes_accessed),
      metadata={"xprof_metadata": json.dumps({"kernel": "stacked_tgmm", "tiling": list(tiling)})},
  )
  return call(group_metadata, group_offset, lhs, rhs)


# ------------------------------------------------------------------------------
# Public API: latent_expert_ffn with custom VJP
# ------------------------------------------------------------------------------


def _wo_fwd(h, wo, group_sizes, group_offset, tiling: LatentFFNTiling, interpret):
  out = mblx_backend.gmm(
      h,
      wo,
      group_sizes,
      h.dtype,
      tiling.wo_fwd,
      group_offset[0],
      interpret=interpret,
  )
  return out


def _ffn_fwd_impl(x, wi_0, wi_1, wo, group_sizes, group_offset, tiling: LatentFFNTiling, interpret):
  h = gate_up_swiglu(
      x,
      wi_0,
      wi_1,
      group_sizes,
      group_offset,
      tiling.gate_up,
      vmem_limit_bytes=tiling.vmem_limit_bytes,
      interpret=interpret,
  )
  return _wo_fwd(h, wo, group_sizes, group_offset, tiling, interpret)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _latent_ffn(x, wi_0, wi_1, wo, group_sizes, group_offset, tiling, interpret):
  return _ffn_fwd_impl(x, wi_0, wi_1, wo, group_sizes, group_offset, tiling, interpret)


def _latent_ffn_fwd(x, wi_0, wi_1, wo, group_sizes, group_offset, tiling, interpret):
  y = _ffn_fwd_impl(x, wi_0, wi_1, wo, group_sizes, group_offset, tiling, interpret)
  # Only the primal inputs are residuals: nothing [M, F]-sized is kept for the backward.
  return y, (x, wi_0, wi_1, wo, group_sizes, group_offset)


def _latent_ffn_bwd(tiling: LatentFFNTiling, interpret, res, dy):
  x, wi_0, wi_1, wo, group_sizes, group_offset = res
  e_local = wi_0.shape[0]
  vmem = tiling.vmem_limit_bytes
  dy = dy.astype(x.dtype)
  pre_t = tiling.pretranspose_weights
  dadb, h = swiglu_bwd(
      x,
      dy,
      wi_0,
      wi_1,
      wo.swapaxes(1, 2) if pre_t else wo,
      group_sizes,
      group_offset,
      tiling.swiglu_bwd,
      wo_transposed=pre_t,
      weight_buffers=tiling.swiglu_bwd_weight_buffers,
      vmem_limit_bytes=vmem,
      interpret=interpret,
  )
  dx = stacked_dlhs(
      dadb,
      wi_0.swapaxes(1, 2) if pre_t else wi_0,
      wi_1.swapaxes(1, 2) if pre_t else wi_1,
      group_sizes,
      group_offset,
      tiling.dx,
      out_dtype=x.dtype,
      w_transposed=pre_t,
      vmem_limit_bytes=vmem,
      interpret=interpret,
  )
  dwi = stacked_tgmm(
      x,
      dadb,
      group_sizes,
      group_offset,
      e_local,
      tiling.dwi,
      out_dtype=wi_0.dtype,
      lhs_mode=tiling.tgmm_lhs_mode,
      vmem_limit_bytes=vmem,
      interpret=interpret,
  )
  # dwo^T[e] = dy[e rows]^T @ h[e rows]: the narrower operand (dy, D wide) is the one transposed in-kernel.
  dwo_t = stacked_tgmm(
      dy,
      h[None],
      group_sizes,
      group_offset,
      e_local,
      tiling.dwo,
      out_dtype=wo.dtype,
      lhs_mode=tiling.tgmm_lhs_mode,
      vmem_limit_bytes=vmem,
      interpret=interpret,
  )
  dwo = dwo_t[:, 0].swapaxes(1, 2)
  return dx, dwi[:, 0], dwi[:, 1], dwo, None, None


_latent_ffn.defvjp(_latent_ffn_fwd, _latent_ffn_bwd)


def latent_expert_ffn(
    x_sorted: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    group_sizes: jax.Array,
    *,
    group_offset: jax.Array | int | None = None,
    tiling: LatentFFNTiling | None = None,
    interpret: bool = False,
) -> jax.Array:
  """SwiGLU expert FFN over ragged, expert-sorted rows with VMEM-fused SwiGLU (fwd and bwd).

  Computes, for every row r of group e (rows sorted by group),

      y[r] = (silu(x[r] @ wi_0[e]) * (x[r] @ wi_1[e])) @ wo[e]

  with Pallas TPU kernels that never write the gate/up projections to HBM. The
  custom VJP keeps no [M, d_ff] residuals: the backward recomputes the gate/up
  projections tile by tile. Numerics: operands in `x_sorted.dtype` (cast weights
  to it first), fp32 accumulation, SwiGLU and its derivative in fp32; `h`,
  `[da; db]` are rounded to `x_sorted.dtype` before the next matmul (as in the
  unfused path), weight gradients are returned in the weight dtype.

  Args:
    x_sorted: [M, D] rows sorted by expert. Rows past `sum(group_sizes)` are
      padding (e.g. the ragged all-to-all buffer); their outputs are
      unspecified and they contribute nothing to the weight gradients.
    wi_0: [E_local, D, F] gate weights (silu branch).
    wi_1: [E_local, D, F] up weights (linear branch).
    wo: [E_local, F, D] down weights.
    group_sizes: int32[G], rows per group; G >= E_local.
    group_offset: First local group in `group_sizes` (expert sharding). Default 0.
    tiling: `LatentFFNTiling`; tiles are clamped to the problem size and M is
      zero-padded to a multiple of every tm.
    interpret: Run the Pallas kernels in interpret mode (CPU testing).

  Returns:
    y: [M, D] in `x_sorted.dtype`.
  """
  tiling = tiling or LatentFFNTiling()
  m, d = x_sorted.shape
  e_local, d_w, f = wi_0.shape
  if d_w != d or wi_1.shape != wi_0.shape or wo.shape != (e_local, f, d):
    raise ValueError(f"Shape mismatch: x {x_sorted.shape}, wi_0 {wi_0.shape}, wi_1 {wi_1.shape}, wo {wo.shape}.")
  if group_offset is None:
    group_offset = 0
  group_offset = jnp.asarray(group_offset, dtype=jnp.int32).reshape((1,))
  group_sizes = group_sizes.astype(jnp.int32)
  dtype = x_sorted.dtype
  wi_0, wi_1, wo = (w.astype(dtype) for w in (wi_0, wi_1, wo))

  tiling = tiling.clamp(m, d, f)
  row_multiple = tiling.row_multiple()
  pad = (-m) % row_multiple
  x_p = jnp.pad(x_sorted, ((0, pad), (0, 0))) if pad else x_sorted
  if pad:
    tiling = tiling.clamp(m + pad, d, f)
  y = _latent_ffn(x_p, wi_0, wi_1, wo, group_sizes, group_offset, tiling, interpret)
  return y[:m] if pad else y
