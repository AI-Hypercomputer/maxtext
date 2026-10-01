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

"""Ring-of-experts ragged sort / unsort on TensorCore.

Drop-in alternative to `ring_ragged_sort` / `ring_ragged_unsort` (truncated-buffer
case) that replaces the SparseCore ragged_gather / ragged_gather_reduce kernels
with the TensorCore DMA kernels in `ragged_gather_tc.py`.

Layout: token rows are reshaped (N, D) -> (N, D // 128, 128) around the kernels
so each token is one contiguous row DMA.

Routing (shared by sort and unsort of one MoE chunk):
  token_ids_sorted[j]: source token of the j-th slot in global expert order.
  slot_order[j]:       flat (token * topk + k) slot of the j-th sorted slot.
  start, count:        this shard's window [start, start + count) of sorted slots,
                       count clamped to the buffer size.
  reduce_meta:         routing of the TC gather-reduce, derived once.

Forward sort:    buf[i] = x[token_ids_sorted[start + i]], i < count (else 0).
Forward unsort:  y[t] = sum_{i < count, token_ids_sorted[start+i] == t} w_i * buf[i].
Each op's transpose is the other op, implemented with the other TC kernel.

keep_3d (moe_tc_ragged_3d_gmm): the sorted buffer stays in the kernels' 3D layout
(cap, D // 128, 128) on both sides (sort output / unsort input and their
cotangents), for GMM kernels that consume / produce that layout directly. Buffer
rows past `count` are then left uninitialized (even with mask_padding): the 3D
gmm_v2 / tgmm_v2 consumers only use rows < sum(group_sizes) == count (gmm
outputs are masked per row and zero-initialized past it, tgmm masks its inputs).
"""

import dataclasses
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from maxtext.kernels.ragged import ragged_gather_tc as tc


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TcRouting:
  token_ids_sorted: jax.Array
  slot_order: jax.Array
  start: jax.Array
  count: jax.Array
  reduce_meta: tc.RaggedGatherReduceMetadata
  # Optional [cap] routing weight of each buffer row (differentiated), sorted with the experts.
  row_weights: jax.Array | None = None


def _to3d(x):
  n, d = x.shape
  if d % 128:
    raise ValueError(f"TC ragged sort needs hidden % 128 == 0, got {d}.")
  return x.reshape(n, d // 128, 128)


def _to2d(x):
  return x.reshape(x.shape[0], x.shape[1] * x.shape[2])


# --- Pallas relayout (N, D) <-> (N, D // 128, 128), after lineage ragged_flatten ---
#
# XLA lowers these reshapes to full-size layout copies. The kernels reshape in
# VMEM and only visit blocks overlapping the first `num_rows` rows; rows past
# that are left uninitialized unless `zero_fill`.


def _manual_axis_type(x):
  return getattr(jax.typeof(x), "manual_axis_type", None)


def _flatten_kernel(n_ref, x_ref, o_ref, *, block, zero_fill):
  if not zero_fill:
    o_ref[...] = x_ref[...].reshape(o_ref.shape)
    return
  start = pl.program_id(0) * block
  n = n_ref[0]

  @pl.when(start + block <= n)
  def _():
    o_ref[...] = x_ref[...].reshape(o_ref.shape)

  @pl.when(start >= n)
  def _():
    o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

  @pl.when(jnp.logical_and(start < n, start + block > n))
  def _():
    y = x_ref[...].reshape(o_ref.shape)
    rows = start + jax.lax.broadcasted_iota(jnp.int32, o_ref.shape, 0)
    o_ref[...] = jnp.where(rows < n, y, jnp.zeros((), y.dtype))


def _tc_flatten(x, num_rows, block, zero_fill=False):
  """(R, D0, D1) -> (R, D0 * D1) over the first num_rows rows."""
  rows, d0, d1 = x.shape
  d = d0 * d1
  block = min(block, rows)
  n = jnp.reshape(jnp.asarray(num_rows, jnp.int32), (1,))
  if zero_fill:
    # Every block is written (zeros past num_rows); inputs past num_rows are
    # clamped to the last valid block so they are not re-fetched.
    grid = (pl.cdiv(rows, block),)
    in_map = lambda i, n_ref: (jnp.minimum(i, jnp.maximum(pl.cdiv(n_ref[0], block) - 1, 0)), 0, 0)
  else:
    grid = (pl.cdiv(n[0], block),)
    in_map = lambda i, n_ref: (i, 0, 0)
  return pl.pallas_call(
      functools.partial(_flatten_kernel, block=block, zero_fill=zero_fill),
      out_shape=jax.ShapeDtypeStruct((rows, d), x.dtype, manual_axis_type=_manual_axis_type(x)),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=grid,
          in_specs=[pl.BlockSpec((block, d0, d1), in_map)],
          out_specs=pl.BlockSpec((block, d), lambda i, n_ref: (i, 0)),
      ),
      name="ragged_flatten_tc",
  )(n, x)


def _unflatten_kernel(n_ref, x_ref, o_ref):
  del n_ref
  o_ref[...] = x_ref[...].reshape(o_ref.shape)


def _tc_unflatten(x, num_rows, block):
  """(R, D) -> (R, D // 128, 128) over the first num_rows rows."""
  rows, d = x.shape
  if d % 128:
    raise ValueError(f"TC ragged sort needs hidden % 128 == 0, got {d}.")
  d0, d1 = d // 128, 128
  block = min(block, rows)
  n = jnp.reshape(jnp.asarray(num_rows, jnp.int32), (1,))
  return pl.pallas_call(
      _unflatten_kernel,
      out_shape=jax.ShapeDtypeStruct((rows, d0, d1), x.dtype, manual_axis_type=_manual_axis_type(x)),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(pl.cdiv(n[0], block),),
          in_specs=[pl.BlockSpec((block, d), lambda i, n_ref: (i, 0))],
          out_specs=pl.BlockSpec((block, d0, d1), lambda i, n_ref: (i, 0, 0)),
      ),
      name="ragged_unflatten_tc",
  )(n, x)


def _keep_3d(blocks):
  return len(blocks) > 2 and bool(blocks[2])


def _gather(x2d, routing, cap, blocks, mask_padding):
  block_size, flat_block = blocks[:2]
  if flat_block:
    x3d = _tc_unflatten(x2d, x2d.shape[0], flat_block)
  else:
    x3d = _to3d(x2d)
  out = tc.ragged_gather_tc(
      x3d,
      routing.token_ids_sorted,
      start=routing.start,
      num_tokens=routing.count,
      max_out_tokens=cap,
      block_size=block_size,
  )
  if _keep_3d(blocks):
    return out
  if flat_block:
    return _tc_flatten(out, routing.count, flat_block, zero_fill=mask_padding)
  out = _to2d(out)
  if mask_padding:
    valid = jnp.arange(cap, dtype=jnp.int32) < routing.count
    out = jnp.where(valid[:, None], out, jnp.zeros((), out.dtype))
  return out


def _gather_reduce(buf2d, routing, num_out_tokens, topk, blocks):
  flat_block = blocks[1]
  if buf2d.ndim == 3:
    buf3d = buf2d
  elif flat_block:
    buf3d = _tc_unflatten(buf2d, routing.count, flat_block)
  else:
    buf3d = _to3d(buf2d)
  out = tc.ragged_gather_reduce_tc(
      buf3d,
      routing.reduce_meta,
      num_out_tokens=num_out_tokens,
      top_k=topk,
  )
  if flat_block:
    return _tc_flatten(out, num_out_tokens, flat_block)
  return _to2d(out)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def tc_sort_gather(x2d, routing, cap, topk, blocks, mask_padding):
  """buf[i] = x[token_ids_sorted[start + i]] for i < count, zero-padded to cap."""
  return _gather(x2d, routing, cap, blocks, mask_padding)


def _tc_sort_gather_fwd(x2d, routing, cap, topk, blocks, mask_padding):
  return _gather(x2d, routing, cap, blocks, mask_padding), (routing, x2d.shape[0])


def _tc_sort_gather_bwd(cap, topk, blocks, mask_padding, res, g):
  del cap, mask_padding
  routing, n_tokens = res
  return _gather_reduce(g, routing, n_tokens, topk, blocks), None


tc_sort_gather.defvjp(_tc_sort_gather_fwd, _tc_sort_gather_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def tc_unsort_reduce(buf2d, routing, num_out_tokens, topk, blocks, mask_padding):
  """y[t] = sum of valid buffer rows whose source token is t."""
  return _gather_reduce(buf2d, routing, num_out_tokens, topk, blocks)


def _tc_unsort_reduce_fwd(buf2d, routing, num_out_tokens, topk, blocks, mask_padding):
  return _gather_reduce(buf2d, routing, num_out_tokens, topk, blocks), (routing, buf2d.shape[0])


def _tc_unsort_reduce_bwd(num_out_tokens, topk, blocks, mask_padding, res, g):
  del num_out_tokens, topk
  routing, cap = res
  return _gather(g, routing, cap, blocks, mask_padding), None


tc_unsort_reduce.defvjp(_tc_unsort_reduce_fwd, _tc_unsort_reduce_bwd)


def ring_ragged_sort_tc(
    hidden_states_local,
    topk_indices_local,
    num_experts,
    topk,
    ep_name,
    ep_size,
    buffer_size,
    gather_block_size=1024,
    reduce_block_size=896,
    mask_padding=True,
    flatten_block_size=0,
    topk_weights_local=None,
    keep_3d=False,
):
  """TC version of `ring_ragged_sort` for the truncated-buffer case.

  If `topk_weights_local` ([N, topk] or flat) is given, the weights ride along as a payload of the
  expert sort, so each buffer row's weight (`TcRouting.row_weights`) needs no separate gather.

  Returns:
    (sorted buffer [buffer_size, hidden], group_sizes [num_experts],
     topk_argsort_revert_indices [N * topk], TcRouting).
  """
  num_tokens_local = hidden_states_local.shape[0]
  topk_indices_flat = topk_indices_local.flatten().astype(jnp.int32)
  n = topk_indices_flat.shape[0]
  w_sorted = None
  if topk_weights_local is None:
    topk_argsort_indices = jnp.argsort(topk_indices_flat, stable=True).astype(jnp.int32)
  else:
    _, topk_argsort_indices, w_sorted = jax.lax.sort(
        (
            topk_indices_flat,
            jnp.arange(n, dtype=jnp.int32),
            jnp.ravel(topk_weights_local).astype(jnp.float32),
        ),
        num_keys=1,
        is_stable=True,
    )
  token_ids_sorted = topk_argsort_indices // topk
  group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)
  topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices).astype(jnp.int32)

  shard_idx = jax.lax.axis_index(ep_name) if ep_size > 1 else 0
  local_num_experts = num_experts // ep_size
  group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
  start = group_offsets[shard_idx * local_num_experts].astype(jnp.int32)
  end = group_offsets[(shard_idx + 1) * local_num_experts].astype(jnp.int32)
  cap = n if buffer_size is None else min(buffer_size, n)
  count = jnp.minimum(end - start, cap).astype(jnp.int32)

  token_ids_sorted = jax.lax.stop_gradient(token_ids_sorted)
  row_weights = None
  if w_sorted is not None:
    rows = jax.lax.dynamic_slice_in_dim(jnp.pad(w_sorted, (0, cap)), start, cap, axis=0)
    row_weights = jnp.where(jnp.arange(cap, dtype=jnp.int32) < count, rows, 0.0)
  reduce_meta = tc.ragged_gather_reduce_tc_metadata(
      token_ids_sorted, start, count, top_k=topk, block_size=reduce_block_size
  )
  routing = TcRouting(
      token_ids_sorted=token_ids_sorted,
      slot_order=topk_argsort_indices,
      start=start,
      count=count,
      reduce_meta=reduce_meta,
      row_weights=row_weights,
  )
  del num_tokens_local
  x = tc_sort_gather(
      hidden_states_local, routing, cap, topk, (gather_block_size, flatten_block_size, keep_3d), mask_padding
  )
  return x, group_sizes_local, topk_argsort_revert_indices, routing


def tc_buffer_row_weights(routing, topk_weights, cap):
  """Routing weight of each buffer row: w[slot_order[start + i]] for i < count, else 0.

  Args:
    routing: TcRouting from `ring_ragged_sort_tc`.
    topk_weights: flat [N * topk] routing weights (differentiated).
    cap: buffer rows.

  Returns:
    [cap] float32 weights.
  """
  if routing.row_weights is not None:
    return routing.row_weights
  w_flat = topk_weights.astype(jnp.float32)
  padded_order = jnp.pad(routing.slot_order, (0, cap))
  rows = jax.lax.dynamic_slice_in_dim(padded_order, routing.start, cap, axis=0)
  valid = jnp.arange(cap, dtype=jnp.int32) < routing.count
  return jnp.where(valid, jnp.take(w_flat, jnp.where(valid, rows, 0), axis=0), 0.0)


def ring_ragged_unsort_tc(
    sorted_tokens_local,
    routing,
    topk,
    topk_weights,
    gather_block_size=1024,
    mask_padding=True,
    flatten_block_size=0,
    prescaled=False,
):
  """TC version of `ring_ragged_unsort`: weighted top-k combine of this shard's slots.

  Args:
    sorted_tokens_local: [buffer_size, hidden] expert outputs, or [buffer_size, hidden // 128, 128]
      (moe_tc_ragged_3d_gmm), which is fed to the TC gather-reduce without a relayout.
    routing: TcRouting from `ring_ragged_sort_tc`.
    topk: routing top-k.
    topk_weights: flat [N * topk] routing weights (differentiated). Unused if prescaled.
    prescaled: rows are already multiplied by their routing weight (e.g. on the expert
      activation, see `tc_buffer_row_weights`), so only the unweighted combine is done.

  Returns:
    [N, hidden] partial combine for this shard's experts.
  """
  cap = sorted_tokens_local.shape[0]
  n = routing.slot_order.shape[0]
  num_out_tokens = n // topk
  if prescaled:
    scaled = sorted_tokens_local
  else:
    w_rows = tc_buffer_row_weights(routing, topk_weights, cap)
    w_rows = w_rows.reshape(-1, *([1] * (sorted_tokens_local.ndim - 1)))
    scaled = (sorted_tokens_local.astype(jnp.float32) * w_rows).astype(sorted_tokens_local.dtype)
  keep_3d = sorted_tokens_local.ndim == 3
  return tc_unsort_reduce(
      scaled, routing, num_out_tokens, topk, (gather_block_size, flatten_block_size, keep_3d), mask_padding
  )
