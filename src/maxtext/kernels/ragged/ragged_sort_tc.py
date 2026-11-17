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
cotangents), for GMM kernels that consume / produce that layout directly. As in
the 2D path, mask_padding zeros the buffer rows past `count` (the sort output and
the unsort-input cotangent). Without it they are left uninitialized: the 3D
gmm_v2 / tgmm_v2 consumers only use rows < sum(group_sizes) == count, but
whole-tensor statistics (e.g. absmax quantization of the gmm lhs or of the
backward gradient) would see the uninitialized rows.
"""

import dataclasses
import functools
from typing import Callable

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
  num_out_tokens: int = jax.tree.static()
  # slot_order / token_ids_sorted hold only this shard's [cap] window (start == 0), see `window_routing`.
  windowed: bool = jax.tree.static()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LocalEpRouting:
  """Compact post-EP-AG routing state safe to checkpoint across remat."""

  tc_routing: TcRouting
  group_sizes: jax.Array
  raw_group_sizes: jax.Array
  raw_count: jax.Array
  local_expert_of_row: jax.Array


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
  """Relayouts one (block, D0, D1) block to (block, D0 * D1); with zero_fill, rows >= n are zeroed."""
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

    def in_map(i, n_ref):
      return (jnp.minimum(i, jnp.maximum(pl.cdiv(n_ref[0], block) - 1, 0)), 0, 0)

  else:
    grid = (pl.cdiv(n[0], block),)

    def in_map(i, n_ref):
      del n_ref
      return (i, 0, 0)

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


def _tokens_3d(blocks):
  """Token-side tensors (sort input / unsort output) are already in the (N, D // 128, 128) layout."""
  return len(blocks) > 3 and bool(blocks[3])


def _is_tpu(blocks):
  """Explicit target-platform flag (`False` selects pure-JAX CPU fallback without calling `jax.default_backend()`)."""
  return len(blocks) <= 4 or bool(blocks[4])


def _gather(x2d, routing, cap, blocks, mask_padding):
  """TC ragged gather of the shard's sorted slots into a [cap, D] buffer ([cap, D // 128, 128] with keep_3d)."""
  if not _is_tpu(blocks):
    x3d = x2d if x2d.ndim == 3 else _to3d(x2d)
    valid = jnp.arange(cap, dtype=jnp.int32) < routing.count
    idx_slice = jax.lax.dynamic_slice_in_dim(
        jnp.pad(routing.token_ids_sorted, (0, cap)), routing.start, cap, axis=0
    )
    safe_idx = jnp.where(valid, idx_slice, 0)
    out = jnp.where(valid[:, None, None], x3d[safe_idx], jnp.zeros((), x3d.dtype))
    return out if _keep_3d(blocks) else _to2d(out)

  block_size, flat_block = blocks[:2]
  if x2d.ndim == 3:
    x3d = x2d
  elif flat_block:
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
    if mask_padding:
      valid = jnp.arange(cap, dtype=jnp.int32) < routing.count
      out = jnp.where(valid[:, None, None], out, jnp.zeros((), out.dtype))
    return out
  if flat_block:
    return _tc_flatten(out, routing.count, flat_block, zero_fill=mask_padding)
  out = _to2d(out)
  if mask_padding:
    valid = jnp.arange(cap, dtype=jnp.int32) < routing.count
    out = jnp.where(valid[:, None], out, jnp.zeros((), out.dtype))
  return out


def _gather_reduce(buf2d, routing, num_out_tokens, topk, blocks):
  """TC ragged gather-reduce: sums each token's valid buffer rows into a [num_out_tokens, D] output."""
  if not _is_tpu(blocks):
    buf3d = buf2d if buf2d.ndim == 3 else _to3d(buf2d)
    if num_out_tokens == 0 or buf3d.shape[0] == 0:
      out = jnp.zeros((num_out_tokens, buf3d.shape[1], buf3d.shape[2]), buf3d.dtype)
    else:
      slot_iota = jnp.arange(routing.reduce_meta.tokens.shape[0], dtype=jnp.int32)
      valid = (slot_iota < routing.count) & (routing.reduce_meta.tokens < num_out_tokens)
      safe_src = jnp.clip(routing.reduce_meta.src_rows, 0, buf3d.shape[0] - 1)
      rows_f32 = jnp.where(valid[:, None, None], buf3d[safe_src].astype(jnp.float32), 0.0)
      safe_tok = jnp.where(valid, routing.reduce_meta.tokens, num_out_tokens)
      out_pad = (
          jnp.zeros((num_out_tokens + 1, buf3d.shape[1], buf3d.shape[2]), jnp.float32)
          .at[safe_tok]
          .add(rows_f32)[:num_out_tokens]
      )
      out = out_pad.astype(buf3d.dtype)
    return out if _tokens_3d(blocks) else _to2d(out)

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
  if _tokens_3d(blocks):
    return out
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
    tag_routing_fn=None,
    keep_3d=False,
    window_routing=False,
):
  """TC version of `ring_ragged_sort` for the truncated-buffer case.

  `tag_routing_fn` (as in `ring_ragged_sort`, e.g. a checkpoint_name tagger for moe_routing_maps) is applied to
  each integer routing array (the TcRouting leaves, group sizes and the inverse permutation), so they can be
  saved across the remat boundary instead of recomputed in the backward pass.

  With `window_routing` (moe_tc_window_routing), the expert sort is a packed uint32 sort, the routing keeps only
  this shard's [start, start + cap) window of the sorted slots (so the reduce metadata sorts cap instead of
  N * topk slots), and the inverse permutation is not computed: the third output is then the forward
  permutation, which TC callers only use for its shape.

  Returns:
    (sorted buffer [buffer_size, hidden], group_sizes [num_experts],
     topk_argsort_revert_indices [N * topk], TcRouting).
  """
  num_tokens_local = hidden_states_local.shape[0]
  topk_indices_flat = topk_indices_local.flatten().astype(jnp.int32)
  n = topk_indices_flat.shape[0]
  if window_routing:
    _, topk_argsort_indices = tc.packed_stable_sort(topk_indices_flat, num_experts - 1)
  else:
    topk_argsort_indices = jnp.argsort(topk_indices_flat, stable=True).astype(jnp.int32)
  group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)

  shard_idx = jax.lax.axis_index(ep_name) if ep_size > 1 else 0
  local_num_experts = num_experts // ep_size
  group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
  start = group_offsets[shard_idx * local_num_experts].astype(jnp.int32)
  end = group_offsets[(shard_idx + 1) * local_num_experts].astype(jnp.int32)
  cap = n if buffer_size is None else min(buffer_size, n)
  count = jnp.minimum(end - start, cap).astype(jnp.int32)

  if window_routing:
    topk_argsort_revert_indices = topk_argsort_indices
    slot_order = jax.lax.dynamic_slice_in_dim(jnp.pad(topk_argsort_indices, (0, cap)), start, cap, axis=0)
    token_ids_sorted = jax.lax.stop_gradient(slot_order // topk)
    reduce_meta = tc.ragged_gather_reduce_tc_metadata(
        token_ids_sorted,
        0,
        count,
        top_k=topk,
        block_size=reduce_block_size,
        num_out_tokens=num_tokens_local,
        packed_sort=True,
    )
    start = jnp.int32(0)
  else:
    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices).astype(jnp.int32)
    slot_order = topk_argsort_indices
    token_ids_sorted = jax.lax.stop_gradient(topk_argsort_indices // topk)
    reduce_meta = tc.ragged_gather_reduce_tc_metadata(
        token_ids_sorted, start, count, top_k=topk, block_size=reduce_block_size
    )
  routing = TcRouting(
      token_ids_sorted=token_ids_sorted,
      slot_order=slot_order,
      start=start,
      count=count,
      reduce_meta=reduce_meta,
      num_out_tokens=num_tokens_local,
      windowed=window_routing,
  )
  if tag_routing_fn is not None:
    routing = jax.tree.map(tag_routing_fn, routing)
    group_sizes_local = tag_routing_fn(group_sizes_local)
    topk_argsort_revert_indices = tag_routing_fn(topk_argsort_revert_indices)
  # A 3D (N, D // 128, 128) input (moe_tc_ragged_3d_dispatch) is gathered without a relayout, and its
  # cotangent is produced in the same layout.
  tokens_3d = hidden_states_local.ndim == 3
  x = tc_sort_gather(
      hidden_states_local,
      routing,
      cap,
      topk,
      (gather_block_size, flatten_block_size, keep_3d, tokens_3d),
      mask_padding,
  )
  return x, group_sizes_local, topk_argsort_revert_indices, routing


def local_ep_ragged_sort_tc(
    ep_x: jax.Array,
    ep_expert_ids: jax.Array,
    *,
    num_local_experts: int,
    expert_start: jax.Array | int,
    topk: int,
    buffer_size: int | None,
    ep_valid: jax.Array | None = None,
    gather_block_size: int = 1024,
    reduce_block_size: int = 896,
    mask_padding: bool = False,
    flatten_block_size: int = 0,
    keep_3d: bool = False,
    return_full_perm: bool = False,
    is_tpu: bool = True,
    tag_routing_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, LocalEpRouting, jax.Array | None]:
  """Muse-tuned local-bucket TC ragged dispatch avoiding global 256-expert `one_hot`.

  Buckets `[expert_start, expert_start + num_local_experts)` into `0..num_local_experts-1`
  and clamps all non-local (`< 0` or `>= num_local_experts`) or invalid (`~ep_valid`) slots
  to positive sentinel `num_local_experts` before `packed_stable_sort`. Local slots therefore
  always occupy `[0, raw_count)` (`start == 0`), eliminating global `jax.nn.one_hot` and
  unused `topk_argsort_revert_indices` remat buffers while optionally returning `perm` for
  dropless overflow windows.

  Args:
    ep_x: Input token activations (`[N, D]` or `[N, D // 128, 128]`).
    ep_expert_ids: Global expert assignments (`[N, topk]`).
    num_local_experts: Number of experts hosted on this EP shard.
    expert_start: Global expert ID of local expert 0 on this shard.
    topk: Number of experts selected per token.
    buffer_size: Optional ragged capacity bound (`None` uses `N * topk`).
    ep_valid: Optional boolean validity mask (`[N, topk]`).
    gather_block_size: Rows per Pallas gather tile.
    reduce_block_size: Slots per Pallas gather-reduce tile.
    mask_padding: Whether to zero-fill rows `[count, cap)` in the forward buffer.
    flatten_block_size: Optional flatten block size (`0` disables flattening).
    keep_3d: Whether to return `(cap, D // 128, 128)` output.
    return_full_perm: Whether to return the full `[N * topk]` sort permutation.
    is_tpu: Explicit target-platform flag (`False` uses pure-JAX CPU fallback).
    tag_routing_fn: Optional `checkpoint_name` tagger applied to integer routing leaves.

  Returns:
    Tuple `(x_sorted, local_routing, full_perm)`.
  """
  num_tokens_local = ep_x.shape[0]
  flat_ids = ep_expert_ids.reshape(-1).astype(jnp.int32)
  n = flat_ids.shape[0]
  local_ids = flat_ids - jnp.asarray(expert_start, dtype=jnp.int32)
  is_local = (local_ids >= 0) & (local_ids < num_local_experts)
  if ep_valid is not None:
    is_local = is_local & ep_valid.reshape(-1).astype(jnp.bool_)
  bucket_ids = jnp.where(is_local, local_ids, jnp.int32(num_local_experts))

  sorted_bucket_ids, perm = tc.packed_stable_sort(bucket_ids, num_local_experts)
  raw_group_sizes = jnp.sum(
      bucket_ids[:, None] == jnp.arange(num_local_experts, dtype=jnp.int32)[None, :],
      axis=0,
      dtype=jnp.int32,
  )
  raw_count = jnp.sum(raw_group_sizes, dtype=jnp.int32)

  cap = n if buffer_size is None else min(buffer_size, n)
  cum_groups = jnp.minimum(jnp.cumsum(raw_group_sizes, dtype=jnp.int32), cap)
  group_sizes = jnp.concatenate([cum_groups[:1], cum_groups[1:] - cum_groups[:-1]]).astype(jnp.int32)
  count = cum_groups[-1].astype(jnp.int32) if num_local_experts > 0 else jnp.int32(0)

  slot_order = perm[:cap]
  max_local_id = jnp.int32(max(num_local_experts - 1, 0))
  local_expert_of_row = jnp.minimum(sorted_bucket_ids[:cap], max_local_id).astype(jnp.int32)
  token_ids_sorted = jax.lax.stop_gradient(jax.lax.div(slot_order, jnp.int32(topk)))
  reduce_meta = tc.ragged_gather_reduce_tc_metadata(
      token_ids_sorted,
      0,
      count,
      top_k=topk,
      block_size=reduce_block_size,
      num_out_tokens=num_tokens_local,
      packed_sort=True,
  )
  tc_routing = TcRouting(
      token_ids_sorted=token_ids_sorted,
      slot_order=slot_order,
      start=jnp.int32(0),
      count=count,
      reduce_meta=reduce_meta,
      num_out_tokens=num_tokens_local,
      windowed=True,
  )
  local_routing = LocalEpRouting(
      tc_routing=tc_routing,
      group_sizes=group_sizes,
      raw_group_sizes=raw_group_sizes,
      raw_count=raw_count,
      local_expert_of_row=local_expert_of_row,
  )
  full_perm = perm if return_full_perm else None
  if tag_routing_fn is not None:
    local_routing = jax.tree.map(tag_routing_fn, local_routing)
    if full_perm is not None:
      full_perm = tag_routing_fn(full_perm)

  tokens_3d = ep_x.ndim == 3
  x_sorted = tc_sort_gather(
      ep_x,
      local_routing.tc_routing,
      cap,
      topk,
      (gather_block_size, flatten_block_size, keep_3d, tokens_3d, is_tpu),
      mask_padding,
  )
  return x_sorted, local_routing, full_perm


def tc_buffer_row_weights(routing, topk_weights, cap):
  """Routing weight of each buffer row: w[slot_order[start + i]] for i < count, else 0.

  Args:
    routing: TcRouting from `ring_ragged_sort_tc`.
    topk_weights: flat [N * topk] routing weights (differentiated).
    cap: buffer rows.

  Returns:
    [cap] float32 weights.
  """
  w_flat = topk_weights.astype(jnp.float32)
  iota = jnp.arange(cap, dtype=jnp.int32)
  valid = iota < routing.count
  if routing.windowed:
    # Valid rows are distinct slots and padding rows get distinct out-of-range indices (filled with 0), so the
    # gather is unique_indices and its transpose needs no scatter-sort.
    rows = jnp.where(valid, routing.slot_order, w_flat.shape[0] + iota)
    return w_flat.at[rows].get(mode="fill", fill_value=0.0, unique_indices=True)
  padded_order = jnp.pad(routing.slot_order, (0, cap))
  rows = jax.lax.dynamic_slice_in_dim(padded_order, routing.start, cap, axis=0)
  return jnp.where(valid, jnp.take(w_flat, jnp.where(valid, rows, 0), axis=0), 0.0)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _windowed_row_weights(
    w_flat: jax.Array,
    slot_order: jax.Array,
    count: jax.Array,
    cap: int,
    num_slots: int,
    is_tpu: bool,
) -> jax.Array:
  """Gathers unique `[0, count)` slot weights without saving `w_flat` in VJP residuals."""
  iota = jnp.arange(cap, dtype=jnp.int32)
  valid = iota < count
  rows = jnp.where(valid, slot_order, num_slots + iota)
  if is_tpu and num_slots >= 4096:
    # Under `--xla_tpu_offload_gather_to_sparsecore=false`, a 1D HBM gather emits `cap`
    # scalar 4B DMAs (`2.49 ms` at `cap=98304`). VMEM scatter + sort runs ~4.5x faster
    # on TensorCore with `0-ULP` identical results because `[0, count)` keys are unique.
    dst_of_slot = jnp.full((num_slots,), cap, dtype=jnp.int32).at[rows].set(
        iota, mode="drop", unique_indices=True
    )
    _, w_sorted = jax.lax.sort((dst_of_slot, w_flat), num_keys=1, is_stable=False)
    return jnp.where(valid, w_sorted[:cap], 0.0)
  return w_flat.at[rows].get(mode="fill", fill_value=0.0, unique_indices=True)


def _windowed_row_weights_fwd(
    w_flat: jax.Array,
    slot_order: jax.Array,
    count: jax.Array,
    cap: int,
    num_slots: int,
    is_tpu: bool,
):
  out = _windowed_row_weights(w_flat, slot_order, count, cap, num_slots, is_tpu)
  # Save ONLY `(slot_order, count)` — never `w_flat` (`ep_weights`) — so checkpointing
  # `w_rows` across remat eliminates the backward remat `ep_weights` all-gather.
  return out, (slot_order, count)


def _windowed_row_weights_bwd(cap: int, num_slots: int, is_tpu: bool, res, g: jax.Array):
  del is_tpu
  slot_order, count = res
  iota = jnp.arange(cap, dtype=jnp.int32)
  valid = iota < count
  g_valid = jnp.where(valid, g.astype(jnp.float32), 0.0)
  rows = jnp.where(valid, slot_order, num_slots + iota)
  dw = jnp.zeros((num_slots,), jnp.float32).at[rows].set(g_valid, mode="drop", unique_indices=True)
  return dw, None, None


_windowed_row_weights.defvjp(_windowed_row_weights_fwd, _windowed_row_weights_bwd)


def tc_local_ep_row_weights(routing: TcRouting, topk_weights: jax.Array, cap: int, *, is_tpu: bool = True):
  """Routing weight of each buffer row of a `local_ep_ragged_sort_tc` routing: `w[slot_order[i]]` for `i < count`.

  Unlike `tc_buffer_row_weights`, the VJP saves only `(slot_order, count)`, never the weights, so a checkpointed
  result does not keep the gathered `[N * topk]` weights alive across rematerialization.

  Args:
    routing: `LocalEpRouting.tc_routing`.
    topk_weights: `[N * topk]` or `[N, topk]` routing weights (differentiated).
    cap: buffer rows.
    is_tpu: True uses a scatter and sort that stay in VMEM on TPU.

  Returns:
    `[cap]` float32 weights, zero for rows at or beyond `count`.
  """
  w_flat = topk_weights.reshape(-1).astype(jnp.float32)
  return _windowed_row_weights(w_flat, routing.slot_order, routing.count, cap, w_flat.shape[0], is_tpu)


def ring_ragged_unsort_tc(
    sorted_tokens_local,
    routing,
    topk,
    topk_weights,
    gather_block_size=1024,
    mask_padding=True,
    flatten_block_size=0,
    prescaled=False,
    out_3d=False,
    is_tpu: bool = True,
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
    out_3d: return the combine in the TC kernels' (N, hidden // 128, 128) layout (no relayout);
      the cotangent is then expected in that layout too.
    is_tpu: False runs a pure-JAX fallback instead of the TensorCore kernel.

  Returns:
    [N, hidden] (or [N, hidden // 128, 128] if out_3d) partial combine for this shard's experts.
  """
  cap = sorted_tokens_local.shape[0]
  num_out_tokens = routing.num_out_tokens
  if prescaled:
    scaled = sorted_tokens_local
  else:
    w_rows = tc_buffer_row_weights(routing, topk_weights, cap)
    w_rows = w_rows.reshape(-1, *([1] * (sorted_tokens_local.ndim - 1)))
    scaled = (sorted_tokens_local.astype(jnp.float32) * w_rows).astype(sorted_tokens_local.dtype)
  keep_3d = sorted_tokens_local.ndim == 3
  return tc_unsort_reduce(
      scaled,
      routing,
      num_out_tokens,
      topk,
      (gather_block_size, flatten_block_size, keep_3d, out_3d, is_tpu),
      mask_padding,
  )
