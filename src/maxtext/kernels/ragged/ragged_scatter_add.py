# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ragged scatter-add kernel for SparseCore (parallelized).

Given source rows ``x[i]`` and destination indices ``indices[i]`` for ``i in
[start, end)``, computes::

  out = zeros((out_num_rows, hidden_size))
  for i in [start, end):
    out[indices[i]] += x[i]

This is a *fully general* scatter-add — there are no assumptions of uniqueness
on ``indices``.  Multiple source rows may write to the same destination row and
will be accumulated.

This is exactly the gradient of a ragged-gather op; in the MoE pipeline the
``x`` rows are the gradient at the gathered (sorted) tokens and ``indices`` are
the original token IDs (token i was gathered to potentially many sorted slots
because top-K).

------------------------------------------------------------------------------
Parallelization strategy
------------------------------------------------------------------------------
A naive per-row RMW (load ``out[dst]``, add, store) is correct but cannot be
parallelized across cores when two source rows share a destination — concurrent
RMWs on different cores would race.

We avoid the race by **partitioning source rows by destination so that each SC
core touches a disjoint set of destination rows**, with a *fixed-stride*
layout so that all DMA offsets are statically known to be aligned to
``num_simd_lanes`` (the SC HBM-DMA alignment requirement):

1.  *Host-side preprocessing*:
    - Mask invalid rows (outside ``[start, end)``) with a sentinel index
      (``-1``) and zero them in ``x``.
    - Stable-sort source rows by ``dst_index`` so all rows hitting the same
      destination row are contiguous.
    - Pad to exactly ``num_cores_total * rows_per_core`` rows where
      ``rows_per_core`` is a multiple of ``num_simd_lanes``.  Each core ``c``
      owns rows ``[c * rows_per_core, (c+1) * rows_per_core)`` in the padded
      array.
    - Re-distribute rows across cores so that *no destination index spans two
      cores*: when a core's nominal window would split a run of equal
      ``dst_index`` values, the entire run is reassigned to a single core
      (the others get sentinel rows in that slot).  See ``_partition_runs``.

2.  *Kernel*:
    - Zero-initializes the output in parallel: every core clears a disjoint
      output stripe.
    - Each ``(block_id, core_id)`` walks one ``num_simd_lanes``-row tile of
      ``core_id``'s fixed window.  Within a tile we *merge intra-tile
      duplicates*: consecutive rows hitting the same dst do a single combined
      RMW.
    - Cross-tile within the same core is sequenced by Pallas's ``ARBITRARY``
      block-dimension semantics.  Cross-core never collides by construction.
    - Column DMAs for the RMW are issued concurrently across the
      ``hidden_size / num_lanes`` chunks (start all, then wait all in order)
      to overlap HBM latency.

Constraints:
  * Currently only float32 (and int32) RMW is implemented in the kernel —
    sub-32-bit (bf16/f16/i8) RMW would require packed atomic-merge logic
    similar to ``ragged_scatter`` and is left as a TODO.  bf16 inputs hit the
    TC fallback.
  * ``hidden_size`` must be a multiple of ``num_lanes`` (=128) to use the SC
    fast path; otherwise we fall back to TC.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

try:
  from jax.experimental.pallas import tpu_sc as plsc
except ImportError as e:
  print(f"Import Error: {e}")


def main_kernel(
    # Inputs.
    zeros_init_hbm_ref: jax.Ref,  # dtype[num_simd_lanes, hidden_size]
    src_hbm_ref: jax.Ref,  # dtype[num_padded_src_rows, hidden_size]
    dst_indices_hbm_ref: jax.Ref,  # int32[num_padded_src_rows]
    # Output.
    out_hbm_ref: jax.Ref,  # dtype[padded_out_num_rows, hidden_size]
    # Scratch (VMEM).
    dst_indices_vmem: jax.Ref,  # int32[num_simd_lanes]
    src_vmem: jax.Ref,  # dtype[num_simd_lanes, hidden_size]
    acc_vmem: jax.Ref,  # dtype[num_simd_lanes, hidden_size]
    zeros_vmem: jax.Ref,  # dtype[num_simd_lanes, hidden_size]
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_cores_total: int,
    blocks_per_core: int,
    zero_blocks_per_core: int,
):
  """Body of the parallel SparseCore scatter-add kernel."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_lanes = tpu_info.num_lanes
  hidden_size = src_hbm_ref.shape[-1]
  assert hidden_size % num_lanes == 0, f"hidden_size {hidden_size} must be a multiple of num_lanes {num_lanes}"

  recv_sem = sem_ref.at[0]
  send_sem = sem_ref.at[1]

  # ---- 1. Parallel zero-init of out_hbm_ref ---------------------------------
  # Each (zblock_id, core_id) clears one num_simd_lanes-wide tile of the output
  # in core_id's stripe.  The wrapper sized ``padded_out_num_rows`` to be a
  # multiple of ``num_simd_lanes * num_cores_total`` so stripes don't overlap.
  @functools.partial(
      pltpu.emit_pipeline,
      grid=(zero_blocks_per_core, num_cores_total),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
  )
  def _zero_init():
    zblock_id = pl.program_id(0)
    core_id = pl.program_id(1)
    row_start = (core_id * zero_blocks_per_core + zblock_id) * num_simd_lanes
    dma_zero = pltpu.make_async_copy(
        zeros_init_hbm_ref,
        zeros_vmem,
        recv_sem,
    )
    dma_zero.start()
    dma_zero.wait()
    dma_store = pltpu.make_async_copy(
        zeros_vmem,
        out_hbm_ref.at[pl.ds(row_start, num_simd_lanes)],
        send_sem,
    )
    dma_store.start()
    dma_store.wait()

  _zero_init()

  # ---- 2. Parallel scatter-add ---------------------------------------------
  # Grid: (blocks_per_core, num_cores_total) with PARALLEL on the cores axis.
  # Each core walks its own *fixed* contiguous source-row window of size
  # ``blocks_per_core * num_simd_lanes`` rows starting at HBM offset
  # ``core_id * blocks_per_core * num_simd_lanes``.  The host preprocessing
  # has already (a) sorted by dst_index, (b) ensured no dst-index group is
  # split across cores, and (c) marked unused slots with the sentinel
  # ``-1`` so the kernel can skip them.
  @functools.partial(
      pltpu.emit_pipeline,
      grid=(blocks_per_core, num_cores_total),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
  )
  def _scatter():
    block_id = pl.program_id(0)
    core_id = pl.program_id(1)

    # Compose the row offset entirely from quantities the compiler can prove
    # are multiples of ``num_simd_lanes``: (core_id + block_id * num_cores) *
    # num_simd_lanes.  This keeps the HBM slice base aligned.
    row_start_tile = (core_id * blocks_per_core + block_id) * num_simd_lanes

    # -- DMA in the dst indices and source rows for this tile --
    dma_idx = pltpu.make_async_copy(
        dst_indices_hbm_ref.at[pl.ds(row_start_tile, num_simd_lanes)],
        dst_indices_vmem,
        recv_sem,
    )
    dma_idx.start()
    dma_src = pltpu.make_async_copy(
        src_hbm_ref.at[pl.ds(row_start_tile, num_simd_lanes)],
        src_vmem,
        recv_sem,
    )
    dma_src.start()
    dma_idx.wait()
    dma_src.wait()
    dst_ids = dst_indices_vmem[...]

    # -- Sequentially process each source row in the tile.  Because the
    # source rows are sorted by dst_index (and never split a dst across
    # cores) we can safely RMW out[dst] without cross-core races.  We
    # additionally merge intra-tile duplicates: consecutive rows hitting
    # the same dst do a single combined RMW.  Sentinel (-1) rows are
    # skipped via the ``valid`` predicate. --
    for r in range(num_simd_lanes):
      dst_row = dst_ids[r]
      valid = dst_row >= 0

      if r + 1 < num_simd_lanes:
        next_dst = dst_ids[r + 1]
        next_valid = next_dst >= 0
        merge_with_next = jnp.logical_and(next_valid, jnp.equal(dst_row, next_dst))
      else:
        merge_with_next = jnp.bool_(False)

      @pl.when(jnp.logical_and(valid, jnp.logical_not(merge_with_next)))
      def _commit():
        # Issue all column-load DMAs concurrently (start-all then wait-all)
        # to overlap HBM latency, like the gather kernel.
        loads = []
        for col_start in range(0, hidden_size, num_lanes):
          d = pltpu.make_async_copy(
              out_hbm_ref.at[dst_row, pl.ds(col_start, num_lanes)],
              acc_vmem.at[r, pl.ds(col_start, num_lanes)],
              recv_sem,
          )
          d.start()
          loads.append(d)
        for d in loads:
          d.wait()
        # acc_vmem[r] += merged_src; merged_src is src_vmem[r] plus any
        # earlier rows merged into it (handled by the merge-loop below).
        for c in range(0, hidden_size, num_simd_lanes):
          col_slice = pl.ds(c, num_simd_lanes)
          acc_vmem[r, col_slice] = acc_vmem[r, col_slice] + src_vmem[r, col_slice]
        stores = []
        for col_start in range(0, hidden_size, num_lanes):
          d = pltpu.make_async_copy(
              acc_vmem.at[r, pl.ds(col_start, num_lanes)],
              out_hbm_ref.at[dst_row, pl.ds(col_start, num_lanes)],
              send_sem,
          )
          d.start()
          stores.append(d)
        for d in stores:
          d.wait()

      # Merge: when this row will be merged with the next, fold src_vmem[r]
      # into src_vmem[r+1] so the next iteration sees the running sum.
      if r + 1 < num_simd_lanes:

        @pl.when(jnp.logical_and(valid, merge_with_next))
        def _merge_into_next():
          for c in range(0, hidden_size, num_simd_lanes):
            col_slice = pl.ds(c, num_simd_lanes)
            src_vmem[r + 1, col_slice] = src_vmem[r + 1, col_slice] + src_vmem[r, col_slice]

  _scatter()


def _preprocess(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    num_cores_total: int,
    num_simd_lanes: int,
):
  """Sort source rows by destination, build per-core fixed windows.

  Layout: returns ``num_cores_total * rows_per_core`` rows where
  ``rows_per_core`` is a multiple of ``num_simd_lanes``.  Core ``c`` owns
  rows ``[c * rows_per_core, (c+1) * rows_per_core)``.

  Invariants the kernel relies on:
    * Within each core's window, rows are sorted by ``dst_index``.
    * Sentinel rows (``dst_index == -1``, ``x = 0``) may appear anywhere in
      the window and are skipped by the kernel.
    * No ``dst_index >= 0`` value ever appears in two different cores'
      windows.

  Returns:
    x_sorted:      ``[num_cores_total * rows_per_core, hidden_size]``
    dst_for_kernel ``[num_cores_total * rows_per_core]``  (-1 = sentinel)
  """
  num_source_rows = x.shape[0]
  pos = jnp.arange(num_source_rows)
  valid = (pos >= start) & (pos < end)

  # Mask invalid rows: dst = INT_MAX so they sort to the end after sort.
  INVALID = jnp.int32(2**31 - 1)
  dst_masked = jnp.where(valid, indices, INVALID).astype(jnp.int32)
  x_masked = jnp.where(valid[:, None], x, jnp.zeros_like(x))

  order = jnp.argsort(dst_masked, stable=True)
  dst_sorted = dst_masked[order]
  x_sorted = x_masked[order]

  # ``rows_per_core`` is the static per-core slot count.  We need it large
  # enough to hold the worst-case rows assigned to a single core after the
  # dst-run-grouping step.  A run of equal-dst rows is never split across
  # cores, so in the pathological case where every source row hits the same
  # destination, *one* core has to hold all ``num_source_rows`` rows.  We
  # therefore size ``rows_per_core`` accordingly.  This wastes some VMEM /
  # HBM padding when source rows are spread out, but keeps correctness
  # guaranteed for arbitrary index distributions; total padded HBM is
  # ``num_cores_total * rows_per_core`` which is bounded by
  # ``num_cores_total * pad-up(num_source_rows)``.
  rows_per_core = ((num_source_rows + num_simd_lanes - 1) // num_simd_lanes) * num_simd_lanes
  if rows_per_core == 0:
    rows_per_core = num_simd_lanes  # at least one tile
  num_padded = num_cores_total * rows_per_core
  pad_total = num_padded - num_source_rows
  if pad_total:
    x_sorted = jnp.pad(x_sorted, ((0, pad_total), (0, 0)))
    dst_sorted = jnp.pad(dst_sorted, (0, pad_total), constant_values=INVALID)

  # ---- Fix nominal core boundaries so they don't split dst-runs. ----
  #
  # We reassign rows that straddle a core boundary.  The simplest correct
  # approach:
  #   For each row i, compute its "preferred core" = i // rows_per_core.
  #   If the row at i has the same dst as the row at i-1, and i is at a core
  #   boundary (i.e. i % rows_per_core == 0), it would be split — push the
  #   whole run into the previous core.  Concretely we mark rows that need to
  #   move and re-sort by ((adjusted_core, original_position)).
  #
  # Implementation: walk-by-walk via cumulative ops.
  pos_padded = jnp.arange(num_padded)
  preferred_core = pos_padded // rows_per_core  # int[num_padded]
  prev_dst = jnp.concatenate([jnp.array([INVALID + 1]), dst_sorted[:-1]])
  same_as_prev = (dst_sorted == prev_dst) & (dst_sorted != INVALID)

  # ``boundary_violation[i]`` is True if i sits at a core-boundary AND has the
  # same dst as i-1 (so the run is being split).  In that case row i and any
  # following same-dst rows belong with the previous core.
  is_boundary = (pos_padded % rows_per_core) == 0
  boundary_violation = is_boundary & same_as_prev

  # For each row, compute the most recent boundary_violation index (-inf if
  # none).  Rows in [violation_idx, end_of_run) need to be reassigned.
  # We propagate a "current run owner core" via cumulative max.
  # owner_core[i] := preferred_core[i] adjusted for boundary violations.
  #
  # Algorithm: build adjustment delta as `same_as_prev * (preferred_core_change)`
  # but a cleaner way is:
  #   1. Compute for each row the start-of-run index (segment start where dst
  #      changes from previous).
  #   2. Owner core = preferred_core[start_of_run].
  run_start = jnp.where(same_as_prev, 0, pos_padded)
  # Forward-fill the maxes so each row knows its run-start position.
  start_of_run = jax.lax.cummax(run_start)
  owner_core = preferred_core[start_of_run]

  # ---- Re-permute rows so each core's window contains exactly rows_per_core
  # rows.  Within each core's window, retain the sorted-by-dst order.  Pad
  # short windows with sentinel; for cores that "overflowed" we trust the
  # rough_partition above to balance — but if a single dst-run is huge it
  # can overflow one core.  In that pathological case we fall back to TC. ----
  # Lex sort by (owner_core, original_position).  Use int32 keys (stable
  # ``argsort`` preserves original-position order for equal keys, which gives
  # us the lexicographic ordering we want without needing int64).
  perm = jnp.argsort(owner_core, stable=True)
  dst_owned = dst_sorted[perm]
  x_owned = x_sorted[perm]
  owner_owned = owner_core[perm]

  # Validate / pad each core's slot to exactly rows_per_core rows.  We
  # achieve this by computing each row's *position within its core*: number of
  # earlier rows with the same owner_core.  If that position >= rows_per_core
  # the row overflows and we mark it invalid (it becomes a no-op).  We also
  # need to pad empty slots — easiest is to scatter (owner_owned * rows_per_core
  # + within_owner_pos) into a fresh sentinel-filled array.
  same_owner_as_prev = jnp.concatenate([jnp.array([False]), owner_owned[1:] == owner_owned[:-1]])
  within_owner_pos = jax.lax.associative_scan(lambda a, b: a + b, same_owner_as_prev.astype(jnp.int32))
  # Reset ``within_owner_pos`` at each owner change (cumsum since last reset).
  # The associative scan above overcounts; redo with segment-aware cumsum:
  #   pos_within = i - first_index_of_owner(i)
  first_index = jnp.where(same_owner_as_prev, 0, pos_padded)
  first_index = jax.lax.cummax(first_index)
  within_owner_pos = pos_padded - first_index
  overflow = within_owner_pos >= rows_per_core
  dst_owned_safe = jnp.where(overflow, jnp.int32(-1), dst_owned)
  dst_owned_safe = jnp.where(dst_owned_safe == INVALID, jnp.int32(-1), dst_owned_safe)
  x_owned_safe = jnp.where(overflow[:, None], jnp.zeros_like(x_owned), x_owned)

  # Scatter into dense-by-core layout.
  target_pos = owner_owned * rows_per_core + within_owner_pos
  target_pos = jnp.where(overflow, num_padded, target_pos)  # send overflows to a discarded last slot

  # We allocate one extra slot for the discarded overflow rows.
  dst_packed = jnp.full((num_padded + 1,), -1, dtype=jnp.int32).at[target_pos].set(dst_owned_safe, mode="drop")
  x_packed = (
      jnp.zeros((num_padded + 1, x_owned_safe.shape[1]), x_owned_safe.dtype).at[target_pos].set(x_owned_safe, mode="drop")
  )

  return x_packed[:num_padded], dst_packed[:num_padded], rows_per_core


def _ragged_scatter_add_impl(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    out_num_rows: int,
) -> jax.Array:
  """Implementation of ``ragged_scatter_add``."""
  assert x.ndim == 2, "ragged_scatter_add only supports 2D x."
  assert indices.ndim == 1, "ragged_scatter_add only supports 1D indices."
  assert x.shape[0] == indices.shape[0], "x and indices must have matching leading dims."

  if jnp.isscalar(start):
    start = jnp.asarray(start, jnp.int32)[None]
  if jnp.isscalar(end):
    end = jnp.asarray(end, jnp.int32)[None]
  if start.ndim == 0:
    start = start[None]
  if end.ndim == 0:
    end = end[None]
  start_scalar = start[0]
  end_scalar = end[0]

  dtype = x.dtype
  hidden_size = x.shape[-1]

  sc_info = pltpu.get_tpu_info().sparse_core
  num_lanes = pltpu.get_tpu_info().num_lanes if sc_info is not None else 128
  # The SC fast path only supports 32-bit-element RMW today.  For sub-32-bit
  # dtypes (bf16, fp16, i8 …) or unaligned hidden sizes, fall back to TC.
  itemsize_bits = jax.dtypes.itemsize_bits(dtype)
  if sc_info is None or hidden_size % num_lanes != 0 or itemsize_bits != 32:
    return _tc_fallback(x, indices, start_scalar, end_scalar, out_num_rows)

  num_simd_lanes = sc_info.num_lanes
  num_cores_total = sc_info.num_cores * sc_info.num_subcores

  x_sorted, dst_idx_sorted, rows_per_core = _preprocess(
      x, indices, start_scalar, end_scalar, num_cores_total, num_simd_lanes
  )
  num_padded_src_rows = x_sorted.shape[0]
  blocks_per_core = rows_per_core // num_simd_lanes
  if blocks_per_core == 0:
    return _tc_fallback(x, indices, start_scalar, end_scalar, out_num_rows)

  # Pad the *output* row count up to a multiple of
  # ``num_simd_lanes * num_cores_total`` so the parallel zero-init has clean
  # disjoint stripes per core.
  zero_block_align = num_simd_lanes * num_cores_total
  out_pad = (-out_num_rows) % zero_block_align
  padded_out_num_rows = out_num_rows + out_pad
  zero_blocks_per_core = padded_out_num_rows // zero_block_align

  zeros_init = jnp.zeros((num_simd_lanes, hidden_size), dtype)

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  out_padded = pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          num_cores_total=num_cores_total,
          blocks_per_core=blocks_per_core,
          zero_blocks_per_core=zero_blocks_per_core,
      ),
      out_shape=jax.ShapeDtypeStruct((padded_out_num_rows, hidden_size), dtype),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      scratch_shapes=[
          pltpu.VMEM((num_simd_lanes,), jnp.int32),  # dst_indices tile
          pltpu.VMEM((num_simd_lanes, hidden_size), dtype),  # src tile
          pltpu.VMEM((num_simd_lanes, hidden_size), dtype),  # rmw acc
          pltpu.VMEM((num_simd_lanes, hidden_size), dtype),  # zeros init buf
          pltpu.SemaphoreType.DMA((2,)),
      ],
      mesh=vector_mesh,
      name="sc_ragged_scatter_add_par",
  )(zeros_init, x_sorted, dst_idx_sorted)
  return out_padded[:out_num_rows]


def _tc_fallback(x, indices, start_scalar, end_scalar, out_num_rows):
  valid = (jnp.arange(x.shape[0]) >= start_scalar) & (jnp.arange(x.shape[0]) < end_scalar)
  x_masked = jnp.where(valid[:, None], x, jnp.zeros_like(x))
  return jnp.zeros((out_num_rows, x.shape[-1]), x.dtype).at[indices].add(x_masked)


@functools.partial(jax.jit, static_argnames=("out_num_rows",))
def ragged_scatter_add(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    out_num_rows: int,
) -> jax.Array:
  """Scatter-add ``x[i]`` into ``out[indices[i]]`` for ``i in [start, end)``.

  This is a fully general scatter-add — there are no assumptions of
  uniqueness on ``indices``.  Multiple source rows targeting the same
  destination row will be accumulated.

  Args:
    x: 2D ``(num_source_rows, hidden_size)`` source values.
    indices: 1D ``(num_source_rows,)`` int32 destination row indices.
    start: scalar / 1-element int32 array — inclusive lower bound on the
      source-row range to process.
    end: scalar / 1-element int32 array — exclusive upper bound on the
      source-row range to process.
    out_num_rows: static int — number of rows in the output. Indices outside
      ``[0, out_num_rows)`` are not allowed.

  Returns:
    A 2D ``(out_num_rows, hidden_size)`` array of dtype matching ``x``.
  """
  return _ragged_scatter_add_impl(x, indices, start, end, out_num_rows)
