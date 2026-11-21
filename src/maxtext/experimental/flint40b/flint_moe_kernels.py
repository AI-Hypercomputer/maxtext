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

"""SparseCore gather-reduce, counting sort, and Pallas router top-k kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
import math
from typing import Any

import jax
from jax.experimental import layout as jax_layout
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp

ReduceTree = Any

B9_COMBINE_REDUCE_TREE: ReduceTree = (((0, 4), (2, 6)), ((1, 5), (3, 7)))
B9_UNROUTE_REDUCE_TREE: ReduceTree = (
    ((0, 4, True), (2, 6, True), True),
    ((1, 5, True), (3, 7, True), True),
)


def linear_reduce_tree(n: int) -> ReduceTree:
  """Builds left-associated `((((0 + 1) + 2) + ...) + n-1)` reduction tree.

  Args:
    n: Number of leaf slots (`n >= 1`).

  Returns:
    Left-associated nested tuple reduction tree covering `0..n-1`.
  """
  tree: ReduceTree = 0
  for i in range(1, n):
    tree = (tree, i)
  return tree


def reduce_tree_leaves(tree: ReduceTree) -> list[int]:
  """Extracts leaf slot indices from `tree` in traversal order.

  Args:
    tree: Reduction tree specification (`int` leaf or `(left, right[, round])`).

  Returns:
    Ordered list of integer leaf slot indices.
  """
  if isinstance(tree, int):
    return [tree]
  if not (isinstance(tree, tuple) and len(tree) in (2, 3)):
    raise ValueError(f"Invalid reduce tree node: {tree!r}.")
  return reduce_tree_leaves(tree[0]) + reduce_tree_leaves(tree[1])


def _validate_reduce_tree(tree: ReduceTree, k: int) -> bool:
  leaves = reduce_tree_leaves(tree)
  return len(leaves) >= k and sorted(leaves) == list(range(len(leaves)))


def eval_reduce_tree(
    tree: ReduceTree,
    vals: Sequence[Any],
    zero: Any,
    round_fn: Callable[[Any], Any] | None = None,
) -> Any:
  """Evaluates `tree` over `vals` (`slot >= len(vals)` maps to `zero`).

  Args:
    tree: Reduction tree specification.
    vals: Sequence of leaf values.
    zero: Additive identity value for empty slots.
    round_fn: Optional rounding function applied at intermediate nodes.

  Returns:
    Reduced value resulting from tree evaluation.
  """
  if isinstance(tree, int):
    return vals[tree] if tree < len(vals) else zero
  out = eval_reduce_tree(tree[0], vals, zero, round_fn) + eval_reduce_tree(tree[1], vals, zero, round_fn)
  if len(tree) == 3 and tree[2] and round_fn is not None:
    out = round_fn(out)
  return out


def with_default_layout(x: jax.Array) -> jax.Array:
  """Pins `x` to row-major layout on TPU without altering tensor values.

  Args:
    x: Input array.

  Returns:
    Array constrained to major-to-minor `(0, 1, ...)` layout on TPU.
  """
  if not pltpu.is_tpu_device() or x.ndim == 0:
    return x
  return jax_layout.with_layout_constraint(x, jax_layout.Layout(major_to_minor=tuple(range(x.ndim))))


def checkpoint_router_map(x: jax.Array, *, compact: bool) -> jax.Array:
  """Checkpoints `x` under 'moe_router_maps' with optional (-1, 128) pack.

  Args:
    x: Input tensor to checkpoint.
    compact: Whether to pack trailing dimensions into `(-1, 128)` row-major.

  Returns:
    Checkpointed tensor with the original shape preserved.
  """
  if compact and x.size >= 128 and x.size % 128 == 0:
    orig_shape = x.shape
    packed = with_default_layout(x.reshape(-1, 128))
    packed = jax.ad_checkpoint.checkpoint_name(packed, "moe_router_maps")
    return with_default_layout(packed).reshape(orig_shape)
  return jax.ad_checkpoint.checkpoint_name(x, "moe_router_maps")


def round_f32_to_bf16_in_f32(x: jax.Array) -> jax.Array:
  """Rounds finite `float32` values to `bfloat16` RNE, returned as `float32`.

  Args:
    x: Finite `float32` array.

  Returns:
    `float32` array rounded to `bfloat16` round-to-nearest-even precision.
  """
  u = jax.lax.bitcast_convert_type(x, jnp.int32)
  lsb = jax.lax.shift_right_logical(u, jnp.int32(16)) & jnp.int32(1)
  u = (u + jnp.int32(0x7FFF) + lsb) & jnp.int32(-65536)
  return jax.lax.bitcast_convert_type(u, jnp.float32)


def dense_gather_reduce_sc(
    operand: jax.Array,
    gather_indices: jax.Array,
    topk_weights: jax.Array | None = None,
    *,
    reduce_group_size: int,
    reduce_tree: ReduceTree,
    col_chunk_size: int = 1024,
    single_sc: bool = False,
) -> jax.Array | None:
  """Runs all-valid fixed-k Dual-SparseCore gather-reduce (including `k=6`).

  Args:
    operand: Source `[num_in_rows, hidden_dim]` matrix (`bfloat16`/`float32`).
    gather_indices: Gather index vector `int32[num_out * reduce_group_size]`.
    topk_weights: Optional per-slot weights `[num_out * reduce_group_size]`;
      when `None`, performs an unweighted sum reduction.
    reduce_group_size: Number of slots `k` reduced into each output row.
    reduce_tree: Association and intermediate-rounding tree over `0..n-1`.
    col_chunk_size: Preferred hidden-dimension tile size.
    single_sc: When True, restricts execution to 1 SparseCore.

  Returns:
    Reduced `[num_out, hidden_dim]` array in `operand.dtype`, or `None` when not
    on a SparseCore TPU or when dimensions do not align with subcore wave
    tiling.
  """
  if not pltpu.is_tpu_device():
    return None
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return None
  if operand.ndim != 2 or gather_indices.ndim != 1 or gather_indices.dtype != jnp.int32:
    return None
  if operand.dtype not in (jnp.bfloat16, jnp.float32):
    return None
  if not _validate_reduce_tree(reduce_tree, reduce_group_size):
    raise ValueError(f"reduce_tree {reduce_tree!r} must cover slots 0..n-1" f" (n >= {reduce_group_size}).")

  num_gathered = gather_indices.shape[0]
  num_in_rows, hidden_dim = operand.shape
  if num_gathered <= 0 or num_in_rows <= 0 or hidden_dim <= 0:
    return None
  if num_gathered % reduce_group_size != 0 or hidden_dim % 128 != 0:
    return None

  packing = 32 // jax.dtypes.itemsize_bits(operand.dtype)
  if num_in_rows % packing != 0:
    return None

  lanes = sc_info.num_lanes
  out_rows_per_step = math.lcm(packing, lanes // math.gcd(lanes, reduce_group_size))
  row_subchunk_size = out_rows_per_step * reduce_group_size
  row_chunk_size = math.lcm(row_subchunk_size, 128)

  num_sc_cores = 1 if single_sc else sc_info.num_cores
  row_wave_size = row_chunk_size * num_sc_cores * sc_info.num_subcores
  if num_gathered % row_wave_size != 0:
    if not single_sc and sc_info.num_cores > 1:
      single_wave = row_chunk_size * sc_info.num_subcores
      if num_gathered % single_wave == 0:
        num_sc_cores = 1
        row_wave_size = single_wave
      else:
        return None
    else:
      return None

  generation = pltpu.get_tpu_info().generation
  vmem_budget = int((512 if generation >= 7 else 256) * 1024 * 0.90)
  out_i32_rows = out_rows_per_step // packing
  chosen_col_chunk = 0
  for cand in (col_chunk_size, 1024, 512, 256, 128):
    if cand < 128 or cand % 128 != 0 or hidden_dim % cand != 0:
      continue
    buf_bytes = 2 * (row_subchunk_size + out_i32_rows) * cand * 4
    if buf_bytes <= vmem_budget:
      chosen_col_chunk = cand
      break
  if chosen_col_chunk == 0:
    return None

  num_out = num_gathered // reduce_group_size
  op_sg = jax.lax.stop_gradient(operand)
  idx_sg = jax.lax.stop_gradient(gather_indices)
  weights_f32 = None if topk_weights is None else jax.lax.stop_gradient(topk_weights).reshape(-1).astype(jnp.float32)
  if operand.dtype == jnp.bfloat16:
    round_fn = round_f32_to_bf16_in_f32
  else:
    round_fn = None
  emit_pipeline: Any = pltpu.emit_pipeline

  def _kernel(
      in_hbm_ref: Any,
      idx_hbm_ref: Any,
      weights_hbm_ref: Any,
      out_hbm_ref: Any,
  ) -> None:
    num_row_chunks = num_gathered // row_wave_size
    num_col_chunks = hidden_dim // chosen_col_chunk
    subcore_first_row_chunk = jax.lax.axis_index(("core", "subcore")) * num_row_chunks
    in_spec = pl.BlockSpec((row_chunk_size,), lambda i: (subcore_first_row_chunk + i,))
    in_specs = (in_spec,) * (1 + (weights_hbm_ref is not None))

    @functools.partial(
        emit_pipeline,
        grid=(num_row_chunks,),
        in_specs=in_specs,
        _explicit_indices=True,
    )
    def idx_pipeline(
        outer_step: Any,
        idx_ref: Any,
        weights_ref: Any = None,
    ) -> None:
      row_chunk_idx = subcore_first_row_chunk + outer_step.index[0]
      num_row_subchunks = row_chunk_size // row_subchunk_size

      @functools.partial(
          emit_pipeline,
          grid=(num_row_subchunks, num_col_chunks),
          in_specs=pl.BlockSpec(
              (pl.Indirect(row_subchunk_size), chosen_col_chunk),
              lambda r, c: (
                  jax.lax.div(
                      idx_ref[pl.ds(r * row_subchunk_size, row_subchunk_size)],
                      packing,
                  ),
                  c,
              ),
          ),
          out_specs=pl.BlockSpec(
              (out_i32_rows, chosen_col_chunk),
              lambda r, c: (row_chunk_idx * num_row_subchunks + r, c),
          ),
          _explicit_indices=True,
      )
      def data_pipeline(
          inner_step: Any,
          gather_ref: Any,
          out_ref: Any,
      ) -> None:
        gather_typed = gather_ref.bitcast(operand.dtype)
        out_typed = out_ref.bitcast(operand.dtype)
        row_slice = pl.ds(inner_step.index[0] * row_subchunk_size, row_subchunk_size)
        subchunk_idxs = idx_ref[row_slice]
        w_sub = None if weights_ref is None else weights_ref[row_slice].astype(jnp.float32)
        unpack_col_chunk = 32

        @plsc.parallel_loop(0, chosen_col_chunk, step=unpack_col_chunk)
        def _(col_base: Any) -> None:
          accs = []
          for g in range(out_rows_per_step):
            group_vals = []
            for s in range(reduce_group_size):
              row = g * reduce_group_size + s
              row_data = gather_typed[
                  pl.ds(row * packing, packing),
                  pl.ds(col_base, unpack_col_chunk),
              ].astype(jnp.float32)
              if packing == 1:
                val = row_data[0]
              else:
                val = jnp.where(
                    jax.lax.bitwise_and(subchunk_idxs[row], 1) == 0,
                    row_data[0],
                    row_data[1],
                )
              if w_sub is not None:
                val = val * w_sub[row]
              group_vals.append(val)
            zero = jnp.zeros_like(group_vals[0])
            accs.append(eval_reduce_tree(reduce_tree, group_vals, zero, round_fn))
          out_typed[:, pl.ds(col_base, unpack_col_chunk)] = jnp.stack(accs, axis=0).astype(operand.dtype)

      data_pipeline(in_hbm_ref.bitcast(jnp.int32), out_hbm_ref.bitcast(jnp.int32))

    if weights_hbm_ref is not None:
      idx_pipeline(idx_hbm_ref, weights_hbm_ref)
    else:
      idx_pipeline(idx_hbm_ref)

  run_kernel: Any = pl.kernel(
      _kernel,
      out_type=jax.ShapeDtypeStruct((num_out, hidden_dim), operand.dtype),
      mesh=plsc.VectorSubcoreMesh(
          core_axis_name="core",
          subcore_axis_name="subcore",
          num_cores=num_sc_cores,
      ),
      compiler_params=pltpu.CompilerParams(needs_layout_passes=True, use_tc_tiling_on_sc=True),
      name=f"sc_dense_gather_reduce_k{reduce_group_size}",
  )
  return run_kernel(op_sg, idx_sg, weights_f32)


def _router_topk_fwd_kernel(
    x_ref: Any,
    w_ref: Any,
    i_ref: Any,
    *,
    topk: int,
    num_experts: int,
) -> None:
  """Exact VMEM k-pass max-and-mask matching `jax.lax.top_k` bit-for-bit."""
  x = x_ref[...]
  avail = jnp.ones(x.shape, dtype=jnp.bool_)
  iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
  for j in range(topk):
    x_cur = jnp.where(avail, x, -jnp.inf)
    m = jnp.max(x_cur, axis=1, keepdims=True)
    cand = avail & (x_cur == m)
    idx = jnp.min(jnp.where(cand, iota, num_experts - 1), axis=1)
    w_ref[:, j] = m[:, 0]
    i_ref[:, j] = idx.astype(jnp.int32)
    avail = avail & (iota != idx[:, None])


def _router_topk_bwd_kernel(
    idx_ref: Any,
    g_ref: Any,
    dx_ref: Any,
    *,
    topk: int,
    num_experts: int,
) -> None:
  """Exact VMEM k-pass dense VJP for distinct top-k indices (`-0.0 -> +0.0`)."""
  idx = idx_ref[...]
  g = g_ref[...]
  g_bits = jax.lax.bitcast_convert_type(g, jnp.int32)
  g = jax.lax.bitcast_convert_type(jnp.where(g_bits == jnp.int32(-(2**31)), jnp.int32(0), g_bits), g.dtype)
  iota = jax.lax.broadcasted_iota(jnp.int32, (idx.shape[0], num_experts), 1)
  acc = jnp.zeros((idx.shape[0], num_experts), dtype=g.dtype)
  for j in range(topk):
    acc = jnp.where(iota == idx[:, j : j + 1], g[:, j : j + 1], acc)
  dx_ref[...] = acc


def is_tpu_device() -> bool:
  """Returns True when the active JAX backend is TPU."""
  return pltpu.is_tpu_device()


def _select_router_topk_block(rows: int, block_rows: int) -> int | None:
  """Selects an exact VMEM-aligned row tile dividing `rows`, or `None`."""
  if rows <= 0 or block_rows <= 0:
    return None
  if rows <= block_rows:
    return rows
  if rows % 8 != 0:
    return None
  block = math.gcd(rows, block_rows)
  if block < 8 or block % 8 != 0 or rows % block != 0:
    return None
  return block


def router_topk_fwd_tpu(
    scores: jax.Array,
    topk: int,
    block_rows: int = 512,
) -> tuple[jax.Array, jax.Array] | None:
  """Runs exact Pallas TensorCore VMEM top-k, or returns `None` to fall back.

  Args:
    scores: Router logits/scores of shape `[..., num_experts]` in `float32`.
    topk: Number of highest scores to select (`1 <= topk <= 16`).
    block_rows: Row block size for Pallas VMEM tiling.

  Returns:
    Tuple `(values, indices)` of shape `[..., topk]`, or `None` when not on a
    TPU device or when dimensions do not satisfy VMEM alignment requirements.
  """
  if not pltpu.is_tpu_device():
    return None
  num_experts = scores.shape[-1]
  if scores.dtype != jnp.float32 or num_experts % 128 != 0 or topk <= 0 or topk > 16 or topk > num_experts:
    return None
  orig_shape = scores.shape
  rows = scores.size // num_experts
  block = _select_router_topk_block(rows, block_rows)
  if block is None:
    return None
  scores_2d = jax.lax.stop_gradient(scores.reshape(rows, num_experts))
  vals_2d, idx_2d = pl.pallas_call(
      functools.partial(_router_topk_fwd_kernel, topk=topk, num_experts=num_experts),
      grid=(rows // block,),
      in_specs=[pl.BlockSpec((block, num_experts), lambda i: (i, 0))],
      out_specs=[
          pl.BlockSpec((block, topk), lambda i: (i, 0)),
          pl.BlockSpec((block, topk), lambda i: (i, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((rows, topk), jnp.float32),
          jax.ShapeDtypeStruct((rows, topk), jnp.int32),
      ],
      name=f"router_topk_fwd_k{topk}_e{num_experts}",
  )(scores_2d)
  out_shape = orig_shape[:-1] + (topk,)
  return vals_2d.reshape(out_shape), idx_2d.reshape(out_shape)


def router_topk_bwd_tpu(
    indices: jax.Array,
    grads: jax.Array,
    num_experts: int,
    block_rows: int = 512,
) -> jax.Array:
  """Runs exact Pallas TensorCore VMEM dense VJP for distinct top-k indices.

  Args:
    indices: Selected expert indices of shape `[..., topk]`.
    grads: Upstream gradients wrt top-k values of shape `[..., topk]`.
    num_experts: Total number of experts in the full score dimension.
    block_rows: Row block size for Pallas VMEM tiling.

  Returns:
    Dense gradient wrt router scores of shape `[..., num_experts]`.
  """
  orig_prefix = indices.shape[:-1]
  topk = indices.shape[-1]
  rows = indices.size // topk
  block = _select_router_topk_block(rows, block_rows)
  if pltpu.is_tpu_device():
    if grads.dtype != jnp.float32 or num_experts % 128 != 0 or block is None:
      raise ValueError(
          "router_topk_bwd_tpu failed TPU preconditions for grads"
          f" shape={grads.shape} ({grads.dtype}), num_experts={num_experts},"
          f" block_rows={block_rows}."
      )
    dx_2d = pl.pallas_call(
        functools.partial(_router_topk_bwd_kernel, topk=topk, num_experts=num_experts),
        grid=(rows // block,),
        in_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block, num_experts), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, num_experts), jnp.float32),
        name=f"router_topk_bwd_k{topk}_e{num_experts}",
    )(
        indices.reshape(rows, topk).astype(jnp.int32),
        grads.reshape(rows, topk),
    )
    return dx_2d.reshape(orig_prefix + (num_experts,))

  ids_2d = indices.reshape(rows, topk).astype(jnp.int32)
  g_2d = grads.astype(jnp.float32).reshape(rows, topk)
  ar_e = jax.lax.broadcasted_iota(jnp.int32, (rows, num_experts), 1)
  dx_2d = jnp.zeros((rows, num_experts), dtype=jnp.float32)
  for slot in range(topk):
    dx_2d = dx_2d + jnp.where(
        ar_e == ids_2d[:, slot : slot + 1],
        g_2d[:, slot : slot + 1],
        jnp.float32(0.0),
    )
  return dx_2d.reshape(orig_prefix + (num_experts,)).astype(grads.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def attach_topk_vjp(
    scores: jax.Array,
    precomputed_vals: jax.Array,
    indices: jax.Array,
    num_experts: int,
) -> jax.Array:
  """Attaches exact distinct-top-k VJP wrt `scores` to `precomputed_vals`.

  Args:
    scores: Full router scores `[..., num_experts]`.
    precomputed_vals: Selected top-k score values `[..., topk]`.
    indices: Selected top-k expert indices `[..., topk]`.
    num_experts: Total number of experts.

  Returns:
    `precomputed_vals` wired with custom VJP wrt `scores`.
  """
  del scores, indices, num_experts
  return precomputed_vals


def _attach_topk_vjp_fwd(
    scores: jax.Array,
    precomputed_vals: jax.Array,
    indices: jax.Array,
    num_experts: int,
) -> tuple[jax.Array, jax.Array]:
  """Forward rule for `attach_topk_vjp`."""
  del scores, num_experts
  return precomputed_vals, indices


def _attach_topk_vjp_bwd(
    num_experts: int,
    indices: jax.Array,
    grad_output: jax.Array,
) -> tuple[jax.Array, None, None]:
  """Backward rule for `attach_topk_vjp`."""
  dx = router_topk_bwd_tpu(indices, grad_output, num_experts)
  return dx, None, None


attach_topk_vjp.defvjp(_attach_topk_vjp_fwd, _attach_topk_vjp_bwd)


def counting_sort_permute_sc(
    flat_experts: jax.Array,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array] | None:
  """Runs 2-pass Dual-SparseCore 32-subcore stable counting sort for MoE.

  Returns `(perm, inv_perm, group_sizes)` bit-for-bit identical to
  `(argsort(flat_experts, stable=True), argsort(perm),`
  `bincount(flat_experts))` when dimensions align, or `None` to fall back.

  Args:
    flat_experts: 1D `int32` expert IDs in `[0, num_experts)`.
    num_experts: Total number of experts (must be a multiple of 128 `<= 1024`).

  Returns:
    Tuple `(perm, inv_perm, group_sizes)` of `int32` arrays, or `None` when not
    on a SparseCore TPU or when dimensions do not align with wave chunk tiling.
  """
  if not pltpu.is_tpu_device():
    return None
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return None
  if flat_experts.ndim != 1 or num_experts <= 0 or num_experts % 128 != 0 or num_experts > 1024:
    return None

  total = flat_experts.shape[0]
  num_sc_cores = sc_info.num_cores
  num_subcores = sc_info.num_subcores
  total_subcores = num_sc_cores * num_subcores
  lanes = sc_info.num_lanes
  if total <= 0 or total % (total_subcores * 128) != 0:
    return None
  chunk = total // total_subcores
  generation = pltpu.get_tpu_info().generation
  max_chunk = 32768 if generation >= 7 else 16384
  if chunk > max_chunk or chunk % lanes != 0:
    return None

  flat_i32 = jax.lax.stop_gradient(flat_experts).astype(jnp.int32)
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=num_sc_cores,
      num_subcores=num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  def _pass1_kernel(
      exp_hbm_ref: Any,
      hist_hbm_ref: Any,
      rank_hbm_ref: Any,
      exp_vmem: Any,
      rank_vmem: Any,
      hist_vmem: Any,
  ) -> None:
    sub_id = jax.lax.axis_index("core") * num_subcores + jax.lax.axis_index("subcore")
    base_offset = sub_id * chunk
    pltpu.sync_copy(exp_hbm_ref.at[pl.ds(base_offset, chunk)], exp_vmem)

    @pl.loop(0, num_experts // lanes)
    def _(b: Any) -> None:
      hist_vmem[pl.ds(b * lanes, lanes)] = jnp.zeros((lanes,), dtype=jnp.int32)

    @pl.loop(0, chunk // lanes)
    def _(i: Any) -> None:
      sl = pl.ds(i * lanes, lanes)
      e = exp_vmem[sl]
      dup, last = plsc.scan_count(e)
      cur = plsc.load_gather(hist_vmem, (e,))
      rank_vmem[sl] = cur + dup - 1
      plsc.addupdate_scatter(hist_vmem, (e,), dup, mask=last)

    pltpu.sync_copy(hist_vmem, hist_hbm_ref.at[pl.ds(sub_id * num_experts, num_experts)])
    pltpu.sync_copy(rank_vmem, rank_hbm_ref.at[pl.ds(base_offset, chunk)])

  run_pass1: Any = pl.kernel(
      _pass1_kernel,
      out_type=(
          jax.ShapeDtypeStruct((total_subcores * num_experts,), jnp.int32),
          jax.ShapeDtypeStruct((total,), jnp.int32),
      ),
      scratch_types=[
          pltpu.VMEM((chunk,), jnp.int32),
          pltpu.VMEM((chunk,), jnp.int32),
          pltpu.VMEM((num_experts,), jnp.int32),
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=True, use_tc_tiling_on_sc=True),
      mesh=vector_mesh,
      name="sc_counting_sort_pass1",
  )
  hist_flat, local_rank = run_pass1(flat_i32)

  hist_2d = hist_flat.reshape(total_subcores, num_experts)
  group_size = jnp.sum(hist_2d, axis=0, dtype=jnp.int32)
  expert_offsets = jnp.cumsum(group_size, dtype=jnp.int32) - group_size
  subcore_offsets = (expert_offsets[None, :] + jnp.cumsum(hist_2d, axis=0, dtype=jnp.int32) - hist_2d).reshape(-1)

  def _pass2_kernel(
      exp_hbm_ref: Any,
      rank_hbm_ref: Any,
      off_hbm_ref: Any,
      sorted_hbm_ref: Any,
      inv_hbm_ref: Any,
      exp_vmem: Any,
      pos_vmem: Any,
      src_vmem: Any,
      off_vmem: Any,
  ) -> None:
    sub_id = jax.lax.axis_index("core") * num_subcores + jax.lax.axis_index("subcore")
    base_offset = sub_id * chunk
    pltpu.sync_copy(exp_hbm_ref.at[pl.ds(base_offset, chunk)], exp_vmem)
    pltpu.sync_copy(rank_hbm_ref.at[pl.ds(base_offset, chunk)], pos_vmem)
    pltpu.sync_copy(off_hbm_ref.at[pl.ds(sub_id * num_experts, num_experts)], off_vmem)
    lane_iota = jnp.arange(lanes, dtype=jnp.int32)

    @pl.loop(0, chunk // lanes)
    def _(i: Any) -> None:
      sl = pl.ds(i * lanes, lanes)
      e = exp_vmem[sl]
      r = pos_vmem[sl]
      pos_vmem[sl] = plsc.load_gather(off_vmem, (e,)) + r
      src_vmem[sl] = base_offset + i * lanes + lane_iota

    pltpu.sync_copy(pos_vmem, inv_hbm_ref.at[pl.ds(base_offset, chunk)])
    pltpu.sync_copy(src_vmem, sorted_hbm_ref.at[pos_vmem])

  run_pass2: Any = pl.kernel(
      _pass2_kernel,
      out_type=(
          jax.ShapeDtypeStruct((total,), jnp.int32),
          jax.ShapeDtypeStruct((total,), jnp.int32),
      ),
      scratch_types=[
          pltpu.VMEM((chunk,), jnp.int32),
          pltpu.VMEM((chunk,), jnp.int32),
          pltpu.VMEM((chunk,), jnp.int32),
          pltpu.VMEM((num_experts,), jnp.int32),
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=True, use_tc_tiling_on_sc=True),
      mesh=vector_mesh,
      name="sc_counting_sort_pass2",
  )
  perm, inv_perm = run_pass2(flat_i32, local_rank, subcore_offsets)
  return perm, inv_perm, group_size


def _unroute_fallback(
    grads: jax.Array,
    inv_perm: jax.Array,
    top_k: int,
) -> jax.Array:
  """8-sublane-padded gather-sum fallback matching `B9_UNROUTE_REDUCE_TREE`."""
  num_tokens = inv_perm.shape[0] // top_k
  if num_tokens == 0 or grads.shape[0] == 0:
    return jnp.zeros((num_tokens, grads.shape[-1]), dtype=grads.dtype)
  pad_k = (-top_k) % 8
  if pad_k > 0:
    padded_k = top_k + pad_k
    valid_2d = jnp.broadcast_to((jnp.arange(padded_k) < top_k)[None, :], (num_tokens, padded_k))
    inds_pad = jnp.pad(inv_perm.reshape(num_tokens, top_k), ((0, 0), (0, pad_k))).reshape(-1)
    filler = jnp.arange(inds_pad.shape[0], dtype=inds_pad.dtype) % grads.shape[0]
    safe_inds = jnp.where(valid_2d.reshape(-1), inds_pad, filler)
    gathered = grads[safe_inds].reshape(num_tokens, padded_k, grads.shape[-1])
    gathered = jnp.where(valid_2d[:, :, None], gathered, jnp.zeros((), dtype=grads.dtype))
    return jnp.sum(gathered, axis=1)
  return jnp.sum(grads[inv_perm].reshape(num_tokens, top_k, grads.shape[-1]), axis=1)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def route_tokens_direct(
    tokens: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array,
    top_k: int,
    use_sc: bool,
) -> jax.Array:
  """Gathers `tokens[perm // top_k]` with Dual-SC `dense_gather_reduce_sc` VJP.

  Args:
    tokens: Input activations `[num_tokens, hidden_dim]`.
    perm: Forward permutation indices `int32[num_tokens * top_k]`.
    inv_perm: Inverse permutation indices `int32[num_tokens * top_k]`.
    top_k: Number of routed experts per token.
    use_sc: Whether to attempt backward gather-reduce on Dual-SparseCore.

  Returns:
    Expert-sorted token activations `[num_tokens * top_k, hidden_dim]`.
  """
  del inv_perm, use_sc
  return tokens[perm // top_k]


def _route_tokens_direct_fwd(
    tokens: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array,
    top_k: int,
    use_sc: bool,
) -> tuple[jax.Array, tuple[jax.Array]]:  # pylint: disable=g-one-element-tuple
  """Forward rule for `route_tokens_direct`."""
  del use_sc
  return tokens[perm // top_k], (inv_perm,)


def _route_tokens_direct_bwd(
    top_k: int,
    use_sc: bool,
    residuals: tuple[jax.Array],  # pylint: disable=g-one-element-tuple
    grads: jax.Array,
) -> tuple[jax.Array, None, None]:
  """Backward rule for `route_tokens_direct`."""
  (inv_perm,) = residuals
  if use_sc:
    tree = B9_UNROUTE_REDUCE_TREE if top_k == 6 else linear_reduce_tree(top_k)
    sc_out = dense_gather_reduce_sc(
        grads,
        inv_perm,
        None,
        reduce_group_size=top_k,
        reduce_tree=tree,
    )
    if sc_out is not None:
      return sc_out, None, None
    if pltpu.is_tpu_device():
      raise ValueError(
          "use_sc_moe_gather_reduce=True failed TPU unroute preconditions for"
          f" grads={grads.shape} ({grads.dtype}), inv_perm={inv_perm.shape},"
          f" top_k={top_k}."
      )
  return _unroute_fallback(grads, inv_perm, top_k), None, None


route_tokens_direct.defvjp(_route_tokens_direct_fwd, _route_tokens_direct_bwd)


def _eval_weighted_combine(
    sorted_out: jax.Array,
    inv_perm: jax.Array,
    w_1d: jax.Array,
    top_k: int,
    use_sc: bool,
) -> jax.Array:
  """Evaluates weighted `k`-way combine on Dual-SC or 8-padded fallback."""
  if use_sc:
    tree = B9_COMBINE_REDUCE_TREE if top_k == 6 else linear_reduce_tree(top_k)
    sc_out = dense_gather_reduce_sc(
        sorted_out,
        inv_perm,
        w_1d,
        reduce_group_size=top_k,
        reduce_tree=tree,
    )
    if sc_out is not None:
      return sc_out
    if pltpu.is_tpu_device():
      raise ValueError(
          "use_sc_moe_gather_reduce=True failed TPU combine preconditions for"
          f" sorted_out={sorted_out.shape} ({sorted_out.dtype}),"
          f" inv_perm={inv_perm.shape}, top_k={top_k}."
      )

  num_tokens = inv_perm.shape[0] // top_k
  if num_tokens == 0 or sorted_out.shape[0] == 0:
    return jnp.zeros((num_tokens, sorted_out.shape[-1]), dtype=sorted_out.dtype)
  pad_k = (-top_k) % 8
  padded_k = top_k + pad_k
  w_f32 = w_1d.astype(jnp.float32).reshape(num_tokens, top_k)
  if pad_k > 0:
    valid_2d = jnp.broadcast_to((jnp.arange(padded_k) < top_k)[None, :], (num_tokens, padded_k))
    inds_pad = jnp.pad(inv_perm.reshape(num_tokens, top_k), ((0, 0), (0, pad_k))).reshape(-1)
    w_f32 = jnp.pad(w_f32, ((0, 0), (0, pad_k)))
    filler = jnp.arange(inds_pad.shape[0], dtype=inds_pad.dtype) % sorted_out.shape[0]
    safe_inds = jnp.where(valid_2d.reshape(-1), inds_pad, filler)
    gathered = sorted_out[safe_inds].reshape(num_tokens, padded_k, sorted_out.shape[-1])
    gathered = jnp.where(valid_2d[:, :, None], gathered, jnp.zeros((), dtype=sorted_out.dtype))
    w_f32 = jnp.where(valid_2d, w_f32, 0.0)
  else:
    gathered = sorted_out[inv_perm].reshape(num_tokens, top_k, sorted_out.shape[-1])
  out = gathered.astype(jnp.float32) * w_f32[:, :, None]
  return jnp.sum(out, axis=1).astype(sorted_out.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def weighted_combine_direct(
    sorted_out: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array,
    w_1d: jax.Array,
    top_k: int,
    use_sc: bool,
) -> jax.Array:
  """Combines `sorted_out` with `w_1d` via Dual-SC gather-reduce + gather VJP.

  Args:
    sorted_out: Expert outputs `[num_tokens * top_k, hidden_dim]`.
    perm: Forward permutation indices `int32[num_tokens * top_k]`.
    inv_perm: Inverse permutation indices `int32[num_tokens * top_k]`.
    w_1d: Flattened routing weights `[num_tokens * top_k]`.
    top_k: Number of routed experts per token.
    use_sc: Whether to attempt forward gather-reduce on Dual-SparseCore.

  Returns:
    Weighted token-order expert output `[num_tokens, hidden_dim]`.
  """
  del perm
  return _eval_weighted_combine(sorted_out, inv_perm, w_1d, top_k, use_sc)


def _weighted_combine_direct_fwd(
    sorted_out: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array,
    w_1d: jax.Array,
    top_k: int,
    use_sc: bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
  """Forward rule for `weighted_combine_direct`."""
  out = _eval_weighted_combine(sorted_out, inv_perm, w_1d, top_k, use_sc)
  return out, (sorted_out, perm, inv_perm, w_1d)


def _weighted_combine_direct_bwd(
    top_k: int,
    use_sc: bool,
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    dout: jax.Array,
) -> tuple[jax.Array, None, None, jax.Array]:
  """Backward rule for `weighted_combine_direct`."""
  del use_sc
  sorted_out, perm, inv_perm, w_1d = residuals
  dout_tok = dout[perm // top_k]
  dout_f32 = dout_tok.astype(jnp.float32)
  w_sorted = w_1d[perm].astype(jnp.float32)
  dsorted = (dout_f32 * w_sorted[:, None]).astype(sorted_out.dtype)
  dw_sorted = jnp.sum(dout_f32 * sorted_out.astype(jnp.float32), axis=-1)
  dw_1d = dw_sorted[inv_perm].astype(w_1d.dtype)
  return dsorted, None, None, dw_1d


weighted_combine_direct.defvjp(_weighted_combine_direct_fwd, _weighted_combine_direct_bwd)
