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

"""GPU kernels the MPMD pipeline's tokamax path needs beyond tokamax (gmm_impl 'tokamax': tokamax_gmm 'mosaic' runs
grouped_matmul and ragged_dw's kernels, 'tokamax_mosaic' and 'triton' ragged_dw's Triton one only; combine_impl
'triton' runs gather_sum).

`grouped_matmul(lhs, rhs, group_sizes)` -- the routed experts' forward and dlhs grouped matmuls on Blackwell
(sm_100), reading the weight as it lies.  tokamax's Mosaic GPU kernel takes its weight N-major and transposes the whole
`[E / EP, D, F]` weight into a copy on every forward call (half of a 1.59 ms DeepSeek-V3 forward product at EP 4 on a
GB200); this is that kernel with the weight's tiles loaded as stored and handed to the tensor core as an MN-major
operand, and tiles tuned at the pipeline's shapes (0.82 ms; dlhs, whose transposes cancel, 1.0 -> 0.85 ms).

`ragged_dw(lhs, g, group_sizes, acc)` is, per group e of the rows (sorted by group, `group_sizes` the groups' sizes,
rows past them ignored), `acc[e] + lhs[rows of e]^T @ g[rows of e]` -- megablox tgmm's existing_out, which TPU's
pipeline uses to accumulate a weight gradient inside its dW kernel (ds_ep named_gmm's acc).  tokamax's grouped matmuls
take no starting value, so without it a chunk's backward writes each layer's routed-expert dW into a buffer of its own
(3 x [E / EP, D, F]: 12 GiB a Qwen3.8 layer at EP 4) and adds it to the step's accumulator in a pass of its own (a
read of both and a write).  Here the kernel loads the accumulator's tile, adds its fp32 partial sum and stores the tile
back into the same buffer (a jax.Ref: jax_triton aliases it to the call's output, so a donated accumulator is updated
where it lies), one rounding to its dtype.  On sm_100 it is a tcgen05 (Mosaic GPU) kernel (`_dw_mosaic`, into an
fp32 accumulator too, DW_MOSAIC_F32; ds_ep runs it under tokamax_gmm 'mosaic' only): 0.94 ms at DeepSeek-V3's shapes,
against 2.0 for tokamax's dW and a bf16 add (2.7 with the add in fp32); elsewhere, and for shapes the tcgen05 kernel
does not take, tokamax's Triton ragged-contracting kernel (tokamax/_src/ops/ragged_dot/triton.py,
`_ragged_contracting_dim_dot_kernel`) with the accumulator's tile loaded first (1.74 ms).  Off GPU (CPU tests), and
under tokamax_gmm 'xla', it is lax.ragged_dot_general plus the add, in fp32.

`gather_sum(y, pos, w)` -- the EP combine's weighted sum over a token's K choices, and the dispatch's gradient
(ds_ep combine_impl 'triton'): a Triton kernel that loads only the rows that exist, where XLA's fill-gather walks
all T x K of them (its section below).
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax import lax

_CONTRACTING = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())), lhs_ragged_dimensions=[0], rhs_group_dimensions=[]
)
# Tuned on a GB200 at DeepSeek-V3's and Qwen3.8's EP 4 shapes (tokamax's heuristic tiles, 128 / 128 / 32, 4 warps,
# 4 stages, ran 6-12% slower here; 64-wide k at 4 stages 40% slower)
TILES = dict(block_m=128, block_n=128, block_k=64, num_warps=4, num_stages=3)


@functools.cache
def _kernel():
  import triton
  import triton.language as tl

  @triton.jit
  def dw_acc_kernel(
      a_ptr,
      b_ptr,
      cum_ptr,
      acc_ptr,
      m: tl.constexpr,
      n: tl.constexpr,
      stride_ak: tl.constexpr,
      stride_am: tl.constexpr,
      stride_bk: tl.constexpr,
      stride_bn: tl.constexpr,
      stride_oe: tl.constexpr,
      stride_om: tl.constexpr,
      stride_on: tl.constexpr,
      block_m: tl.constexpr,
      block_n: tl.constexpr,
      block_k: tl.constexpr,
  ):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2).to(tl.int64)
    lo = tl.load(cum_ptr + pid_e)
    hi = tl.load(cum_ptr + pid_e + 1)
    offs_m = (pid_m * block_m).to(tl.int64) + tl.arange(0, block_m)
    offs_n = (pid_n * block_n).to(tl.int64) + tl.arange(0, block_n)
    offs_k = lo.to(tl.int64) + tl.arange(0, block_k)
    # no m / n masks where the tiles divide the weight (tokamax's: a masked load vectorizes worse)
    m_mask = True if m % block_m == 0 else (offs_m[None, :] < m)
    n_mask = True if n % block_n == 0 else (offs_n[None, :] < n)
    o_ptrs = acc_ptr + pid_e * stride_oe + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    o_mask = (True if m % block_m == 0 else (offs_m[:, None] < m)) & n_mask
    # the accumulator's tile is loaded before the contraction, not after it: its latency then hides behind the
    # loop instead of stalling the store (1.74 against 1.98 ms at DeepSeek-V3's shapes on a GB200)
    prev = tl.load(o_ptrs, mask=o_mask, other=0.0)
    a_ptrs = a_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for start_k in range(lo, hi, block_k):
      k_mask = (start_k + tl.arange(0, block_k))[:, None] < hi
      a = tl.load(a_ptrs, mask=k_mask & m_mask, other=0.0)
      b = tl.load(b_ptrs, mask=k_mask & n_mask, other=0.0)
      acc = tl.dot(tl.trans(a), b, acc=acc)
      a_ptrs += block_k * stride_ak
      b_ptrs += block_k * stride_bk
    tl.store(o_ptrs, (acc + prev.to(tl.float32)).to(acc_ptr.dtype.element_ty), mask=o_mask)

  return dw_acc_kernel


def _strides(shape):
  out, s = [], 1
  for d in reversed(shape):
    out.append(s)
    s *= d
  return tuple(reversed(out))


# ------------------------------------------------- gather_sum: the EP combine's and the dispatch backward's reduction
#
# out[t] = sum_k w[t, k] y[pos[t, k]] over a token's K choices, pos[t, k] = len(y) for a choice whose row is not in
# y (another EP shard's expert, or one past the buffer).  XLA emits it as a gather of T x K rows (zeros for the
# missing ones) reduced over K: it walks all T x K x D elements, though at EP 8 seven choices in eight have no row.
# This kernel loads only the rows that exist and accumulates in fp32 in registers.  B200, DeepSeek-V3's EP 8 shapes
# (a batch-split half: 32768 tokens x 8 choices x 7168): the combine 0.34 against XLA's 0.51 ms, the dispatch's
# gradient 0.25 against 0.41; A100, dsv3-mini's (65536 x 6 x 4096): 0.68 against 0.97, 0.64 against 0.84.
GATHER_SUM_TILES = dict(block_t=8, block_d=512, num_warps=8)


@functools.cache
def _gather_sum_kernel():
  import triton
  import triton.language as tl

  @triton.jit
  def gather_sum_kernel(
      y_ptr,
      pos_ptr,
      w_ptr,
      out_ptr,
      t: tl.constexpr,
      r: tl.constexpr,
      k: tl.constexpr,
      d: tl.constexpr,
      block_t: tl.constexpr,
      block_d: tl.constexpr,
      weighted: tl.constexpr,
  ):
    offs_t = (tl.program_id(0) * block_t + tl.arange(0, block_t)).to(tl.int64)
    offs_d = (tl.program_id(1) * block_d + tl.arange(0, block_d)).to(tl.int64)
    t_mask = offs_t < t
    d_mask = True if d % block_d == 0 else (offs_d[None, :] < d)
    acc = tl.zeros((block_t, block_d), dtype=tl.float32)
    for j in tl.static_range(k):
      p = tl.load(pos_ptr + offs_t * k + j, mask=t_mask, other=r)
      has = p < r
      rows = tl.where(has, p, 0).to(tl.int64)
      v = tl.load(y_ptr + rows[:, None] * d + offs_d[None, :], mask=has[:, None] & d_mask, other=0.0)
      v = v.to(tl.float32)
      if weighted:
        v = v * tl.load(w_ptr + offs_t * k + j, mask=t_mask, other=0.0).to(tl.float32)[:, None]
      acc += v
    tl.store(
        out_ptr + offs_t[:, None] * d + offs_d[None, :], acc.to(out_ptr.dtype.element_ty), mask=t_mask[:, None] & d_mask
    )

  return gather_sum_kernel


def gather_sum(y, pos, w=None, *, out_dtype=None, tiles=None):
  """sum_k w[t, k] y[pos[t, k]] for every t -> [T, D] in `out_dtype` (y's by default), accumulated in fp32.

  y [R, D]; pos [T, K] int32, R for a missing row (it adds nothing); w [T, K] (None: every weight 1).  On GPU a
  Triton kernel that reads only the rows that exist (see above; `tiles` overrides GATHER_SUM_TILES); elsewhere the
  same sum in XLA (a fill-gather of T x K rows)."""
  T, K = pos.shape
  R, D = y.shape
  out_dtype = jnp.dtype(out_dtype or y.dtype)
  if jax.default_backend() != "gpu":
    yk = jnp.take(y, pos.reshape(-1), axis=0, mode="fill", fill_value=0).reshape(T, K, D).astype(jnp.float32)
    if w is None:
      return yk.sum(axis=1).astype(out_dtype)
    return jnp.einsum("tkd,tk->td", yk, w.astype(jnp.float32)).astype(out_dtype)
  import jax_triton as jt

  t = dict(GATHER_SUM_TILES, **(tiles or {}))
  bd = min(t["block_d"], 1 << (D - 1).bit_length())
  w_arg = pos if w is None else w  # an unread placeholder when unweighted
  return jt.triton_call(
      y,
      pos.astype(jnp.int32),
      w_arg,
      kernel=_gather_sum_kernel(),
      out_type=jax.ShapeDtypeStruct((T, D), out_dtype),
      name="gather_sum",
      grid=((T + t["block_t"] - 1) // t["block_t"], (D + bd - 1) // bd),
      num_warps=t["num_warps"],
      num_stages=1,
      t=T,
      r=R,
      k=K,
      d=D,
      block_t=t["block_t"],
      block_d=bd,
      weighted=w is not None,
  )


# ------------------------------------------------- the forward on the weight as stored ([G, K, N])
#
# tokamax's Blackwell grouped matmul (tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100.py,
# `ragged_dot_gpu_non_quant_blackwell_kernel`, Apache 2.0) takes its weight N-major ([G, N, K]) and makes it so with
# `rhs.mT` -- a transposed copy of the whole weight on every call: on a GB200 1.3 ms of a Qwen3.8 layer's 2.7 ms
# forward product at EP 4 (128 x 8192 x 2048 bf16), 0.66 of DeepSeek-V3's 1.59.  The dlhs product's weight is the
# transposed view already (the two transposes cancel).  This is that kernel with the weight's tiles loaded as they
# lie ([block_k, block_n] of the stored [G, K, N]) and handed to the tensor core transposed (an MN-major SMEM
# operand), which tcgen05 reads as fast; in the collective (two-CTA) form each CTA loads its half of the tile's
# columns.  Everything else -- the group schedule, the TMEM double buffer, the ragged store -- is tokamax's.
_COMPUTE_WG, _MMA_WARP, _TMA_WARP, _EPILOGUE_WG = 0, 1, 0, 1


def mosaic_config(**kw):
  """A tokamax Mosaic GPU ragged-dot Config from plain values (grid_minor_dim 'M' / 'N')."""
  from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common

  kw = dict(kw)
  kw["grid_minor_dim"] = getattr(common.MatmulDimension, kw.get("grid_minor_dim", "M"))
  return common.Config(split_k=1, persistent=False, **kw)


def _mosaic_kernel(lhs, rhs, group_sizes, out_dtype, config, rhs_kn):
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import mosaic_gpu as plgpu
  from tokamax._src import mosaic_gpu as mgpu_lib
  from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common

  tcgen05, tcgen05_t = plgpu.Layout.TCGEN05, plgpu.Layout.TCGEN05_TRANSPOSED

  block_m, block_n, block_k = config.block_m, config.block_n, config.block_k
  collective = config.collective
  cluster_block_m = block_m * 2 if collective else block_m
  cluster_block_n = block_n * 2 if collective else block_n
  m, k = lhs.shape
  g, n = rhs.shape[0], rhs.shape[2] if rhs_kn else rhs.shape[1]
  m_iters = pl.cdiv(m, cluster_block_m) + g - 1
  n_iters = pl.cdiv(n, cluster_block_n)
  k_iters = pl.cdiv(k, block_k)
  num_stages = min(config.num_stages, k_iters)
  align_tile = 8

  def kernel(
      x_gmem,
      w_gmem,
      group_id_gmem,
      start_within_block_gmem,
      actual_size_gmem,
      block_start_gmem,
      out_gmem,
      x_smem,
      w_smem,
      acc_smem,
      acc_tmem,
      xw_barrier,
      xw_consumed_barrier,
      acc_barrier,
      acc_consumed_barrier,
  ):
    cluster_idx = lax.axis_index("cluster")

    @plgpu.dynamic_scheduling_loop(
        ("mn",), thread_axis="wg", cluster_axes=("cluster",) if collective else (), init_carry=0
    )
    def mn_loop(loop_info, carry):
      (idx,) = loop_info.index
      tid_m, ni = plgpu.planar_snake(idx, (m_iters, n_iters), config.grid_minor_dim, config.grid_tile_width)
      wg = lax.axis_index("wg")
      group_id = group_id_gmem[tid_m]
      start_within_block = start_within_block_gmem[tid_m]
      actual_size = actual_size_gmem[tid_m]
      block_start = pl.multiple_of(block_start_gmem[tid_m], align_tile)
      ms = pl.ds(block_start, cluster_block_m)
      ns = pl.ds(ni * cluster_block_n, cluster_block_n)

      @pl.when(actual_size > 0)
      def compute():
        @mgpu_lib.warp_map_when(wg == _COMPUTE_WG)
        def compute_wg_warp(warp_id):
          cluster_axis = "cluster" if collective else None

          @pl.when(warp_id == _TMA_WARP)
          def tma_warp():
            @pl.loop(0, k_iters)
            def tma_loop(ki):
              ks = pl.ds(ki * block_k, block_k)
              step = carry * k_iters + ki
              si = lax.rem(step, num_stages)

              @pl.when(step >= num_stages)
              def _():
                plgpu.barrier_wait(xw_consumed_barrier.at[si])

              plgpu.copy_gmem_to_smem(
                  x_gmem.at[ms, ks],
                  x_smem.at[si],
                  xw_barrier.at[si],
                  leader_tracked=plgpu.CopyPartition.PARTITIONED(0) if collective else None,
                  collective_axes=cluster_axis,
              )
              # [block_k, cluster_block_n] of a [G, K, N] weight (a CTA takes its columns), or tokamax's
              # [cluster_block_n, block_k] of a [G, N, K] one (its rows)
              plgpu.copy_gmem_to_smem(
                  w_gmem.at[group_id, ks, ns] if rhs_kn else w_gmem.at[group_id, ns, ks],
                  w_smem.at[si],
                  xw_barrier.at[si],
                  leader_tracked=plgpu.CopyPartition.PARTITIONED(int(rhs_kn)) if collective else None,
                  collective_axes=cluster_axis,
              )

          @pl.when((warp_id == _MMA_WARP) & (cluster_idx == 0))
          def mma_warp():
            si_acc = lax.rem(carry, jnp.int32(2))
            ms_acc = pl.ds(si_acc * cluster_block_m, cluster_block_m)

            @pl.when(carry > 1)
            def _():
              plgpu.barrier_wait(acc_consumed_barrier.at[si_acc])

            @pl.loop(0, k_iters)
            def mma_loop_body(ki):
              si_tma = lax.rem(carry * k_iters + ki, num_stages)
              plgpu.barrier_wait(xw_barrier.at[si_tma])
              w_tile = w_smem.at[si_tma].T if rhs_kn else w_smem.at[si_tma]  # [block_n, block_k] either way
              plgpu.tcgen05_mma(
                  acc_tmem.at[:, ms_acc],
                  w_tile,
                  x_smem.at[si_tma].T,
                  xw_consumed_barrier.at[si_tma],
                  accumulate=(ki > 0),
                  collective_axis=cluster_axis,
              )

            plgpu.tcgen05_commit_arrive(acc_barrier.at[si_acc], collective_axis=cluster_axis)

        @pl.when(wg == _EPILOGUE_WG)
        def epilogue_wg():
          si = lax.rem(carry, jnp.int32(2))
          plgpu.barrier_wait(acc_barrier.at[si])
          ms_acc = pl.ds(si * cluster_block_m, cluster_block_m)
          acc = plgpu.async_load_tmem(acc_tmem.at[:, ms_acc], layout=tcgen05)
          acc = acc.astype(acc_smem.dtype)
          acc_smem.T[...] = plgpu.layout_cast(acc, tcgen05_t)
          plgpu.wait_load_tmem()
          plgpu.barrier_arrive(acc_consumed_barrier.at[si])
          plgpu.commit_smem()
          common.copy_ragged_smem_to_gmem(
              acc_smem,
              out_gmem,
              block_start=block_start,
              start_within_block=start_within_block,
              actual_size=actual_size,
              ns=pl.ds(ni * cluster_block_n + cluster_idx * block_n, block_n),
              max_rows=min(cluster_block_m, m),
          )

      return carry + (actual_size > 0)

  if collective:
    acc_consumed_barrier = plgpu.ClusterBarrier(collective_axes=("cluster",), num_barriers=2, orders_tensor_core=True)
  else:
    acc_consumed_barrier = plgpu.Barrier(num_barriers=2, orders_tensor_core=True)
  scratch_shapes = dict(
      x_smem=common.tiled_swizzled_smem((num_stages, block_m, block_k), lhs.dtype, "x"),
      w_smem=common.tiled_swizzled_smem(
          (num_stages, block_k, block_n) if rhs_kn else (num_stages, block_n, block_k), rhs.dtype, "w"
      ),
      acc_tmem=plgpu.TMEM((block_n, cluster_block_m * 2), jnp.float32, collective=collective),
      acc_smem=common.tiled_swizzled_smem((cluster_block_m, block_n), out_dtype, "acc", tiling_prefix=(1,)),
      xw_barrier=plgpu.Barrier(num_arrivals=2, num_barriers=num_stages),
      xw_consumed_barrier=plgpu.Barrier(num_barriers=num_stages, orders_tensor_core=True),
      acc_barrier=plgpu.Barrier(num_barriers=2, orders_tensor_core=True),
      acc_consumed_barrier=acc_consumed_barrier,
  )
  f = plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, n), out_dtype),
      scratch_types=scratch_shapes,
      num_threads=2,
      thread_name="wg",
      grid=(m_iters * n_iters,),
      grid_names=("mn",),
      cluster=(1 + collective,),
      cluster_names=("cluster",),
      kernel_name="grouped_matmul_kn_sm100" if rhs_kn else "grouped_matmul_nk_sm100",
      compiler_params=plgpu.CompilerParams(
          approx_math=True, unsafe_no_auto_barriers=True, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
      ),
  )
  info = common.GroupInfo.create_aligned(group_sizes, cluster_block_m, m_iters, align_tile)
  return f(lhs, rhs, info.group_id, info.start_within_block, info.actual_size, info.block_start)


def mosaic_tiles(rows, groups):
  """The tuned tiles for a buffer of `rows` rows over `groups` groups: 128-row tiles, 4 stages where the groups
  average 768 rows or more of the buffer (DeepSeek-V3's 65536 over 64), else 64-row tiles, 6 stages (Qwen3.8's
  61440 over 128) -- on a GB200 each is the other's shape's runner-up within 2-7%."""
  big = rows >= 768 * groups
  return dict(
      block_m=128 if big else 64,
      block_n=128,
      block_k=64,
      num_stages=4 if big else 6,
      collective=True,
      grid_minor_dim="M",
      grid_tile_width=1,
  )


def mosaic_supported(device=None):
  """Blackwell datacenter GPUs (sm_100 / sm_103: tcgen05), where `grouped_matmul` runs."""
  device = device or jax.devices()[0]
  try:
    return device.platform == "gpu" and int(float(getattr(device, "compute_capability", 0))) == 10
  except (TypeError, ValueError):
    return False


def grouped_matmul_supported(lhs, rhs, device=None):
  """Can `grouped_matmul` take this product: an sm_100 device, bf16 (or f16) both sides, and the weight's two
  dims multiples of 64 (its SMEM tiling).  Any row count and group sizes go: rows are padded to the tiling's 8
  where they are not a multiple of it (a copy of lhs, which the pipeline's buffers -- dropless, or a factor of 1.5
  or 2 -- never need)."""
  return (
      lhs.dtype == rhs.dtype
      and lhs.dtype in (jnp.bfloat16, jnp.float16)
      and rhs.ndim == 3
      and rhs.shape[1] % 64 == 0
      and rhs.shape[2] % 64 == 0
      and mosaic_supported(device)
  )


def grouped_matmul(lhs, rhs, group_sizes, *, transpose_rhs=False, out_dtype=None, config=None):
  """lhs[rows of group e] @ rhs[e] for every group e -> [M, N], reading rhs as it lies: [G, K, N], or with
  transpose_rhs a [G, N, K] weight contracted on its last dim (lhs @ rhs[e]^T: the dlhs product).

  lhs [M, K] holds the rows sorted by group, `group_sizes` [G] int32 (sum <= M); the rows past the groups are not
  written (tokamax's kernel's contract: callers mask them).  bf16 x bf16 (or f16) -> out_dtype (lhs's) from an fp32
  accumulator, on sm_100 only (mosaic_supported); `config` a tokamax Mosaic Config (default: mosaic_tiles)."""
  M, K = lhs.shape
  G = rhs.shape[0]
  K2, N = (rhs.shape[2], rhs.shape[1]) if transpose_rhs else rhs.shape[1:]
  if rhs.ndim != 3 or K2 != K or group_sizes.shape != (G,):
    raise ValueError(
        f"grouped_matmul: lhs {lhs.shape}, rhs {rhs.shape} (transpose_rhs={transpose_rhs}), "
        f"group_sizes {group_sizes.shape}"
    )
  if lhs.dtype != rhs.dtype or lhs.dtype not in (jnp.bfloat16, jnp.float16):
    raise ValueError(f"grouped_matmul takes bf16 x bf16 (or f16), got {lhs.dtype} x {rhs.dtype}")
  config = config if config is not None else mosaic_config(**mosaic_tiles(M, G))
  # The kernel's [block_m, block_k] SMEM tiling (8, 64) applies to the whole [M, K] operand, which Mosaic GPU then
  # wants a multiple of 8 rows long (or at most 8): pad, and drop the rows again (they lie past the groups, so the
  # kernel neither reads them into a group nor writes them).
  pad = -M % 8 if M > 8 else 0
  x = jnp.pad(lhs, ((0, pad), (0, 0))) if pad else lhs
  out = _mosaic_kernel(
      x, rhs, group_sizes.astype(jnp.int32), jnp.dtype(out_dtype or lhs.dtype), config, rhs_kn=not transpose_rhs
  )
  return out[:M] if pad else out


# ------------------------------------------------- the in-place dW on Blackwell's tensor cores (tcgen05)
#
# The ragged contraction -- acc[e] += lhs[rows of e]^T g[rows of e] -- as a warp-specialized Mosaic GPU kernel in
# the forward kernel's mould: a dynamic scheduling loop over (group, cluster tile of acc[e]) work items, group-major
# (a group's rows stay in L2 while its tiles run), a TMA warp streaming [block_k rows, block_m] tiles of lhs and
# [block_k, block_n] of g through num_stages SMEM slots, an MMA warp accumulating lhs^T g in TMEM (lhs's tile as an
# MN-major A operand, g's as an MN-major B; two-CTA collective MMA: a cluster tile is 256 x 256), double-buffered so
# that the epilogue warpgroup -- the accumulator's tile TMA-loaded beside the MMAs, + the fp32 sum, one rounding,
# a TMA store back into the same buffer -- overlaps the next tile's MMAs.  The contraction is ragged and Mosaic's
# 8-row SMEM tiling wants every TMA to start on a row 8 divides, so each group runs as full block_k-row steps from
# its first 8-aligned row (its body) and its other rows -- < 8 before the body, < block_k after it -- are gathered,
# zero-padded, into a side buffer of 2 x block_k rows a group (_dw_sides) whose one or two steps run like any other:
# no step ever reads another group's row.  On a GB200, side buffers included: 0.92 ms at DeepSeek-V3's EP 4 shapes
# (the Triton kernel below: 1.74; tokamax's Triton dW writing a fresh buffer: 1.22), 1.91 ms at Qwen3.8's (3.29;
# 2.09).
DW_MOSAIC = dict(
    block_m=128,
    block_n=128,
    block_k=64,
    num_stages=5,
    collective=True,
    grid_minor_dim="M",
    grid_tile_width=1,
    epi_chunk=128,
)
# An fp32 accumulator (Megatron's main_grad): its epilogue tile is 128 KB of SMEM, which leaves room for 3 stages of
# A / B tiles (32 KB each) within sm_100's 227 KB.  B200, DeepSeek-V3's wi at EP 8 ([32, 7168, 2048] from 65536 rows):
# 1.77 ms and exact against an fp32 product, against 2.69 for the Triton kernel and 1.34 with a bf16 accumulator.
# MAXTEXT_MPMD_DW_MOSAIC_F32=0 puts fp32 accumulators back on the Triton kernel.
DW_MOSAIC_F32 = dict(DW_MOSAIC, num_stages=3)
MOSAIC_F32_ACC = os.environ.get("MAXTEXT_MPMD_DW_MOSAIC_F32", "1") != "0"


def _dw_sides(lhs, g, group_sizes, bk):
  """Per group: the first row of its body (8-aligned: the full bk-row steps), the body's step count and the step
  count in all; and the group's other rows (< 8 before the body, < bk after it) zero-padded into side buffers of
  2 x bk rows a group, [E * 2 * bk, m] / [E * 2 * bk, n]."""
  R = lhs.shape[0]
  cum = jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True)
  lo, hi = cum[:-1], cum[1:]
  body = (lo + 7) // 8 * 8  # 8-aligned always (the kernel's TMA rows), past a group shorter than that
  nbody = jnp.maximum(hi - body, 0) // bk
  body_end = body + nbody * bk
  head, tail = jnp.minimum(body, hi) - lo, jnp.maximum(hi - body_end, 0)
  nsteps = nbody + (head + tail + bk - 1) // bk
  j = jnp.arange(2 * bk, dtype=jnp.int32)[None, :]
  idx = jnp.where(j < head[:, None], lo[:, None] + j, body_end[:, None] + j - head[:, None])
  ok = (j < (head + tail)[:, None]).reshape(-1)[:, None]
  idx = jnp.clip(idx, 0, R - 1).reshape(-1)
  side_l = jnp.where(ok, jnp.take(lhs, idx, axis=0), jnp.zeros((), lhs.dtype))
  side_g = jnp.where(ok, jnp.take(g, idx, axis=0), jnp.zeros((), g.dtype))
  return body, nbody, nsteps, side_l, side_g


def _dw_mosaic(lhs, g, group_sizes, acc_ref, cfg):
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import mosaic_gpu as plgpu
  from tokamax._src import mosaic_gpu as mgpu_lib
  from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common

  bm, bn, bk, stages = cfg["block_m"], cfg["block_n"], cfg["block_k"], cfg["num_stages"]
  collective = cfg["collective"]
  cbm, cbn = (2 * bm, 2 * bn) if collective else (bm, bn)  # the cluster's tile: A split by rows, B by columns
  E, m, n = acc_ref.shape
  m_iters, n_iters = m // cbm, n // cbn
  tiles_per_group = m_iters * n_iters
  minor = getattr(common.MatmulDimension, cfg["grid_minor_dim"])
  ec = cfg["epi_chunk"]
  if cbn % ec:  # the epilogue adds cbn // ec whole chunks: the columns past them would be stored back without dW
    raise ValueError(f"ragged_dw: epi_chunk {ec} does not divide the cluster tile's {cbn} columns")
  tcgen05 = plgpu.Layout.TCGEN05
  body, nbody, nsteps, side_l, side_g = _dw_sides(lhs, g, group_sizes, bk)

  def kernel(
      x_gmem,
      y_gmem,
      xs_gmem,
      ys_gmem,
      body_gmem,
      nbody_gmem,
      nsteps_gmem,
      acc_gmem,
      x_smem,
      y_smem,
      e_smem,
      acc_tmem,
      xw_barrier,
      xw_consumed,
      acc_barrier,
      acc_consumed,
      prev_barrier,
  ):
    cluster_idx = lax.axis_index("cluster")

    @plgpu.dynamic_scheduling_loop(
        ("w",), thread_axis="wg", cluster_axes=("cluster",) if collective else (), init_carry=(jnp.int32(0), jnp.int32(0))
    )
    def loop(info, carry):
      steps, tiles = carry  # k-steps and (non-empty) tiles this CTA ran: SMEM / TMEM slots
      (idx,) = info.index
      e = lax.div(idx, jnp.int32(tiles_per_group))
      mi, ni = plgpu.planar_snake(
          lax.rem(idx, jnp.int32(tiles_per_group)), (m_iters, n_iters), minor, cfg["grid_tile_width"]
      )
      start = pl.multiple_of(body_gmem[e], 8)
      side0 = pl.multiple_of(e * (2 * bk), 8)
      nb, nst = nbody_gmem[e], nsteps_gmem[e]
      wg = lax.axis_index("wg")
      ms, ns = pl.ds(mi * cbm, cbm), pl.ds(ni * cbn, cbn)

      @pl.when(nst > 0)  # an empty group's tiles keep their accumulator: nothing to do
      def work():
        @mgpu_lib.warp_map_when(wg == _COMPUTE_WG)
        def compute_wg_warp(warp_id):
          cluster_axis = "cluster" if collective else None
          part = plgpu.CopyPartition.PARTITIONED(1) if collective else None

          @pl.when(warp_id == _TMA_WARP)
          def tma_warp():
            @pl.loop(0, nst)
            def tma_loop(ki):
              step = steps + ki
              si = lax.rem(step, stages)

              @pl.when(step >= stages)
              def _():
                plgpu.barrier_wait(xw_consumed.at[si])

              def load(xsrc, ysrc, row):
                rows = pl.ds(row, bk)
                plgpu.copy_gmem_to_smem(
                    xsrc.at[rows, ms], x_smem.at[si], xw_barrier.at[si], leader_tracked=part, collective_axes=cluster_axis
                )
                plgpu.copy_gmem_to_smem(
                    ysrc.at[rows, ns], y_smem.at[si], xw_barrier.at[si], leader_tracked=part, collective_axes=cluster_axis
                )

              pl.when(ki < nb)(lambda: load(x_gmem, y_gmem, start + ki * bk))
              pl.when(ki >= nb)(lambda: load(xs_gmem, ys_gmem, side0 + (ki - nb) * bk))

          @pl.when((warp_id == _MMA_WARP) & (cluster_idx == 0))
          def mma_warp():
            slot = lax.rem(tiles, 2)

            @pl.when(tiles > 1)
            def _():
              plgpu.barrier_wait(acc_consumed.at[slot])

            @pl.loop(0, nst)
            def mma_loop(ki):
              si = lax.rem(steps + ki, stages)
              plgpu.barrier_wait(xw_barrier.at[si])
              plgpu.tcgen05_mma(
                  acc_tmem.at[:, pl.ds(slot * cbn, cbn)],
                  x_smem.at[si].T,
                  y_smem.at[si],
                  xw_consumed.at[si],
                  accumulate=(ki > 0),
                  collective_axis=cluster_axis,
              )

            plgpu.tcgen05_commit_arrive(acc_barrier.at[slot], collective_axis=cluster_axis)

        @pl.when(wg == _EPILOGUE_WG)
        def epilogue_wg():
          slot = lax.rem(tiles, 2)
          rows = pl.ds(mi * cbm + cluster_idx * bm, bm)  # this CTA's rows of the cluster tile
          plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the last tile's store has read e_smem
          plgpu.copy_gmem_to_smem(acc_gmem.at[e, rows, ns], e_smem, prev_barrier.at[0])
          plgpu.barrier_wait(acc_barrier.at[slot])
          plgpu.barrier_wait(prev_barrier.at[0])
          for c in range(cbn // ec):
            cols = pl.ds(c * ec, ec)
            part_sum = plgpu.async_load_tmem(acc_tmem.at[:, pl.ds(slot * cbn + c * ec, ec)], layout=tcgen05)
            prev = plgpu.load(e_smem.at[:, cols], layout=tcgen05, optimized=False)
            plgpu.wait_load_tmem()
            e_smem.at[:, cols][...] = (part_sum + prev.astype(jnp.float32)).astype(e_smem.dtype)
          plgpu.barrier_arrive(acc_consumed.at[slot])
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(e_smem, acc_gmem.at[e, rows, ns])

      return steps + nst, tiles + (nst > 0).astype(jnp.int32)

    plgpu.wait_smem_to_gmem(0)

  if collective:
    acc_consumed = plgpu.ClusterBarrier(collective_axes=("cluster",), num_barriers=2, orders_tensor_core=True)
  else:
    acc_consumed = plgpu.Barrier(num_barriers=2, orders_tensor_core=True)
  scratch = dict(
      x_smem=common.tiled_swizzled_smem((stages, bk, bm), lhs.dtype, "x"),
      y_smem=common.tiled_swizzled_smem((stages, bk, bn), g.dtype, "y"),
      e_smem=common.tiled_swizzled_smem((bm, cbn), acc_ref.dtype, "e"),
      acc_tmem=plgpu.TMEM((bm, 2 * cbn), jnp.float32, collective=collective),
      xw_barrier=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
      xw_consumed=plgpu.Barrier(num_barriers=stages, orders_tensor_core=True),
      acc_barrier=plgpu.Barrier(num_barriers=2, orders_tensor_core=True),
      acc_consumed=acc_consumed,
      prev_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=1),
  )
  f = plgpu.kernel(
      kernel,
      out_type=(),
      scratch_types=scratch,
      num_threads=2,
      thread_name="wg",
      grid=(E * tiles_per_group,),
      grid_names=("w",),
      cluster=(1 + collective,),
      cluster_names=("cluster",),
      kernel_name="ragged_dw_sm100",
      compiler_params=plgpu.CompilerParams(
          approx_math=True, unsafe_no_auto_barriers=True, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
      ),
  )
  f(lhs, g, side_l, side_g, body, nbody, nsteps, acc_ref)


def _dw_mosaic_cfg(acc_dtype, mosaic=None):
  return dict(DW_MOSAIC_F32 if jnp.dtype(acc_dtype) == jnp.float32 else DW_MOSAIC, **(mosaic or {}))


def ragged_dw_mosaic_supported(lhs, g, acc, device=None, cfg=None):
  """Does ragged_dw run the tcgen05 kernel here: an sm_100 device, bf16 (or f16) lhs and g, an accumulator of
  their dtype or fp32 (DW_MOSAIC_F32's tiles; MOSAIC_F32_ACC), acc's two dims multiples of the cluster tile
  (256 x 256 by default; rows of any count, padded to the tiling's 8), and the tile's columns a multiple of the
  epilogue's epi_chunk (it adds whole chunks)."""
  cfg = cfg or _dw_mosaic_cfg(acc.dtype)
  k = 2 if cfg["collective"] else 1
  acc_ok = acc.dtype == lhs.dtype or (acc.dtype == jnp.float32 and MOSAIC_F32_ACC)
  return (
      lhs.dtype == g.dtype
      and lhs.dtype in (jnp.bfloat16, jnp.float16)
      and acc_ok
      and acc.ndim == 3
      and acc.shape[1] % (k * cfg["block_m"]) == 0
      and acc.shape[2] % (k * cfg["block_n"]) == 0
      and (k * cfg["block_n"]) % cfg["epi_chunk"] == 0
      and mosaic_supported(device)
  )


def _sm80_dw_tiles(acc_dtype):
  """The Triton kernel's tiles for an fp32 accumulator on an sm80 GPU (A100): 64-row tiles -- the 128 x 128 default
  holds a 128 x 128 fp32 accumulator tile in registers beside the contraction's, and spills (A100, dsv3-mini expert
  shapes [64, 4096, 1536] from 49152 rows: 5.66 ms against 6.70 at the default tiles, 6.06-10.4 at the others tried).
  {} elsewhere (TILES: tuned on GB200, where the bf16 kernel runs on tcgen05 anyway)."""
  try:
    cc = float(getattr(jax.devices()[0], "compute_capability", 0) or 0)
  except Exception:  # no GPU backend
    return {}
  return dict(block_m=64, block_n=128) if 8 <= cc < 9 and jnp.dtype(acc_dtype).itemsize == 4 else {}


def ragged_dw(lhs, g, group_sizes, acc, *, tiles=None, mosaic=None, xla=False):
  """acc[e] + lhs[rows of group e]^T @ g[rows of group e] for every group e -> acc's shape and dtype.

  lhs [R, m] and g [R, n] hold the rows sorted by group, `group_sizes` [E] int32 (sum <= R; rows past them ignored),
  acc [E, m, n].  On GPU acc's buffer is updated in place (jax.Ref; donated, no copy) with one rounding: on sm_100
  by the tcgen05 kernel where it takes the shapes (ragged_dw_mosaic_supported; `mosaic` a dict of tiles over
  DW_MOSAIC's -- DW_MOSAIC_F32's for an fp32 acc -- or False for the Triton kernel), else by the Triton one (`tiles`:
  its TILES); elsewhere, or with `xla` (tokamax_gmm 'xla'), the product is lax's, added in fp32.  ds_ep tells it
  which per tokamax_gmm (ds_ep._ragged_dw)."""
  E, m, n = acc.shape
  if lhs.shape[1] != m or g.shape[1] != n or lhs.shape[0] != g.shape[0] or group_sizes.shape != (E,):
    raise ValueError(f"ragged_dw: lhs {lhs.shape}, g {g.shape}, group_sizes {group_sizes.shape} against acc {acc.shape}")
  if xla or jax.default_backend() != "gpu":
    dw = lax.ragged_dot_general(lhs, g, group_sizes, _CONTRACTING, preferred_element_type=jnp.float32)
    return (acc.astype(jnp.float32) + dw).astype(acc.dtype)
  cfg = None if mosaic is False or tiles is not None else _dw_mosaic_cfg(acc.dtype, mosaic)
  if cfg is not None and ragged_dw_mosaic_supported(lhs, g, acc, cfg=cfg):
    R = lhs.shape[0]
    pad = -R % 8 if R > 8 else 0  # the 8-row SMEM tiling applies to the whole [R, .] operands
    if pad:
      lhs, g = jnp.pad(lhs, ((0, pad), (0, 0))), jnp.pad(g, ((0, pad), (0, 0)))
    ref = jax.new_ref(acc)
    _dw_mosaic(lhs, g, group_sizes.astype(jnp.int32), ref, cfg)
    return ref[...]
  import jax_triton as jt

  t = dict(TILES, **(_sm80_dw_tiles(acc.dtype) if tiles is None else {}), **(tiles or {}))
  cum = jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True)
  ref = jax.new_ref(acc)
  sa, sb, so = _strides(lhs.shape), _strides(g.shape), _strides(acc.shape)
  cdiv = lambda a, b: (a + b - 1) // b
  jt.triton_call(
      lhs,
      g,
      cum,
      ref,
      kernel=_kernel(),
      out_type=(),
      grid=(cdiv(m, t["block_m"]), cdiv(n, t["block_n"]), E),
      name="ragged_dw_acc",
      num_warps=t["num_warps"],
      num_stages=t["num_stages"],
      m=m,
      n=n,
      stride_ak=sa[0],
      stride_am=sa[1],
      stride_bk=sb[0],
      stride_bn=sb[1],
      stride_oe=so[0],
      stride_om=so[1],
      stride_on=so[2],
      block_m=t["block_m"],
      block_n=t["block_n"],
      block_k=t["block_k"],
  )
  return ref[...]
