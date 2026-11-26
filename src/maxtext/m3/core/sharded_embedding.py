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

"""Index-routed `ragged_all_to_all` lookup for row-sharded embedding tables.

Large row-sharded embedding tables (such as Qwen3.8-Flash-Next's 320M-row PLE
n-gram table, ~102 GB in bf16) are sharded along the vocabulary axis across
devices (`P(table_axes, None)`). If lookup indices are replicated across all
devices, every shard gathers and scatters over the entire global batch of
indices even though only `1 / P` of them fall in its local row range.

This module replaces replicated gather/scatter with a shard-local index-routed
collective:

  1. Each device computes (or receives) only its local token slice's indices.
  2. Indices are bucketed by destination table shard `dst = idx // V_loc` and
     exchanged as compact `int32` local row offsets via `ragged_all_to_all`
     inside capacity-bounded multi-round overflow loops (preventing OOM under
     Zipfian hot-shard skew while executing a single round on balanced inputs).
  3. Each shard gathers only the rows belonging to its local `V_loc` slice and
     returns embeddings to the requesting token shards via reverse
     `ragged_all_to_all`.
  4. The custom VJP routes output gradients back to the owning table shards and
     scatter-adds only the received rows into `d_table_local` (`jax.shard_map`'s
     VJP transpose automatically sums across any extra token-sharded axes on
     which the table is replicated, e.g. DCN `"data"`).
"""

from collections.abc import Callable
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.m3.core import kernels as m3_kernels


def _ragged_exchange(
    operand: jax.Array,
    out_buf: jax.Array,
    in_off: jax.Array,
    send_sz: jax.Array,
    out_off: jax.Array,
    recv_sz: jax.Array,
    *,
    axis_name: tuple[str, ...],
    is_tpu: bool,
) -> jax.Array:
  """Exchanges ragged leading-axis slices across `axis_name`, updating `out_buf` in place.

  Uses hardware `jax.lax.ragged_all_to_all` on TPU and an exact disjoint-slice
  `all_gather` reference on CPU/GPU hosts so multi-axis `axis_name` tuples and
  multi-round partial updates into `out_buf` behave identically across backends.
  """
  safe_in_off = jnp.where(send_sz > 0, in_off, jnp.int32(0))
  safe_out_off = jnp.where(send_sz > 0, out_off, jnp.int32(0))
  if is_tpu:
    return jax.lax.ragged_all_to_all(
        operand,
        out_buf,
        safe_in_off,
        send_sz,
        safe_out_off,
        recv_sz,
        axis_name=axis_name,
    )

  rank = jax.lax.axis_index(axis_name)
  in_len = operand.shape[0]
  out_len = out_buf.shape[0]
  all_operands = jax.lax.all_gather(operand, axis_name, axis=0, tiled=False)
  all_in_off = jax.lax.all_gather(safe_in_off, axis_name, axis=0, tiled=False)
  all_out_off = jax.lax.all_gather(safe_out_off, axis_name, axis=0, tiled=False)

  starts = all_out_off[:, rank]
  ends = starts + recv_sz
  row_idx = jnp.arange(out_len, dtype=jnp.int32)[:, None]
  hit = (row_idx >= starts[None, :]) & (row_idx < ends[None, :])
  has_hit = jnp.any(hit, axis=1)
  src_idx = jnp.argmax(hit.astype(jnp.int32), axis=1)
  in_row = jnp.clip(all_in_off[src_idx, rank] + (row_idx[:, 0] - starts[src_idx]), 0, max(in_len - 1, 0))
  fetched = all_operands[src_idx, in_row]
  return jnp.where(has_hit[:, None], fetched, out_buf)


def _compute_round_schedule(
    counts_total: jax.Array,
    sent_so_far: jax.Array,
    base_in_off_all: jax.Array,
    *,
    recv_buf_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes greedy per-round send/receive counts and offsets capped at `recv_buf_len`."""
  rem = counts_total - sent_so_far
  cum_end = jnp.cumsum(rem, axis=0)
  cum_start = cum_end - rem
  round_counts = jnp.clip(jnp.int32(recv_buf_len) - cum_start, jnp.int32(0), rem)
  recv_off_all = jnp.cumsum(round_counts, axis=0) - round_counts
  src_in_off_all = base_in_off_all + sent_so_far
  return round_counts, recv_off_all, src_in_off_all, sent_so_far + round_counts


def _local_row_sharded_lookup(
    table_local: jax.Array,
    ids_local: jax.Array,
    *,
    table_axes: tuple[str, ...],
    table_shards: int,
    out_dtype: Any,
    buffer_factor: float,
    is_tpu: bool,
) -> jax.Array:
  """Per-device index-routed lookup with capacity-bounded multi-round overflow and custom VJP."""
  v_loc, d_dim = table_local.shape
  n_loc = ids_local.shape[0]
  total_rows = table_shards * v_loc
  out_dt = jnp.dtype(out_dtype)
  recv_buf_len = min(table_shards * n_loc, max(1, int(math.ceil(buffer_factor * n_loc))))

  ids_i32 = ids_local.astype(jnp.int32)
  valid = (ids_i32 >= 0) & (ids_i32 < total_rows)
  safe_ids = jnp.where(valid, ids_i32, jnp.int32(0))
  dst = safe_ids // jnp.int32(v_loc)
  local_row = safe_ids - dst * jnp.int32(v_loc)

  send_perm = jnp.argsort(dst)
  inv_send_perm = jnp.argsort(send_perm)
  sorted_local_row = local_row[send_perm][:, None]
  send_counts = jnp.bincount(dst, length=table_shards).astype(jnp.int32)
  counts_total = jax.lax.all_gather(send_counts, table_axes, axis=0, tiled=False)
  tbl_dtype = table_local.dtype

  @jax.custom_vjp
  def _lookup_sorted(
      tbl: jax.Array,
      s_rows: jax.Array,
      c_tot: jax.Array,
      s_perm: jax.Array,
      inv_perm: jax.Array,
  ) -> jax.Array:
    out, _ = _lookup_sorted_fwd(tbl, s_rows, c_tot, s_perm, inv_perm)
    return out

  def _lookup_sorted_fwd(
      tbl: jax.Array,
      s_rows: jax.Array,
      c_tot: jax.Array,
      s_perm: jax.Array,
      inv_perm: jax.Array,
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    base_in_off_all = jnp.cumsum(c_tot, axis=1) - c_tot
    rank = jax.lax.axis_index(table_axes)

    def _cond(state: tuple[jax.Array, jax.Array]) -> jax.Array:
      sent, _ = state
      return jnp.any(c_tot > sent)

    def _body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
      sent, out_sorted = state
      r_counts, recv_off_all, src_in_off_all, next_sent = _compute_round_schedule(
          c_tot, sent, base_in_off_all, recv_buf_len=recv_buf_len
      )
      recv_rows = _ragged_exchange(
          s_rows,
          jnp.zeros((recv_buf_len, 1), dtype=jnp.int32),
          src_in_off_all[rank],
          r_counts[rank],
          recv_off_all[rank],
          r_counts[:, rank],
          axis_name=table_axes,
          is_tpu=is_tpu,
      )
      gathered = jnp.take(tbl, recv_rows[:, 0], axis=0).astype(out_dt)
      next_out = _ragged_exchange(
          gathered,
          out_sorted,
          recv_off_all[:, rank],
          r_counts[:, rank],
          src_in_off_all[:, rank],
          r_counts[rank],
          axis_name=table_axes,
          is_tpu=is_tpu,
      )
      return next_sent, next_out

    init_sent = jnp.zeros((table_shards, table_shards), dtype=jnp.int32)
    init_out = jnp.zeros((n_loc, d_dim), dtype=out_dt)
    _, out_sorted = jax.lax.while_loop(_cond, _body, (init_sent, init_out))
    return out_sorted[inv_perm], (s_rows, c_tot, s_perm)

  def _lookup_sorted_bwd(
      res: tuple[jax.Array, jax.Array, jax.Array],
      g_out: jax.Array,
  ) -> tuple[jax.Array, None, None, None, None]:
    s_rows, c_tot, s_perm = res
    base_in_off_all = jnp.cumsum(c_tot, axis=1) - c_tot
    rank = jax.lax.axis_index(table_axes)
    # Transpose of `out_sorted[inv_perm]` is the forward gather `g_out[s_perm]` (no local scatter).
    g_cast = g_out[s_perm].astype(out_dt)

    def _cond(state: tuple[jax.Array, jax.Array]) -> jax.Array:
      sent, _ = state
      return jnp.any(c_tot > sent)

    def _body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
      sent, d_tbl = state
      r_counts, recv_off_all, src_in_off_all, next_sent = _compute_round_schedule(
          c_tot, sent, base_in_off_all, recv_buf_len=recv_buf_len
      )
      recv_rows = _ragged_exchange(
          s_rows,
          jnp.zeros((recv_buf_len, 1), dtype=jnp.int32),
          src_in_off_all[rank],
          r_counts[rank],
          recv_off_all[rank],
          r_counts[:, rank],
          axis_name=table_axes,
          is_tpu=is_tpu,
      )
      recv_grads = _ragged_exchange(
          g_cast,
          jnp.zeros((recv_buf_len, d_dim), dtype=out_dt),
          src_in_off_all[rank],
          r_counts[rank],
          recv_off_all[rank],
          r_counts[:, rank],
          axis_name=table_axes,
          is_tpu=is_tpu,
      )
      total_recv = jnp.sum(r_counts[:, rank])
      valid_recv = jnp.arange(recv_buf_len, dtype=jnp.int32) < total_recv
      # Route invalid padding slots out-of-bounds (`v_loc`) with `mode="drop"` so hardware drops them
      # instead of creating atomic write collisions on row 0 during SparseCore scatter-add.
      safe_rows = jnp.where(valid_recv, recv_rows[:, 0], jnp.int32(v_loc))
      masked_grads = jnp.where(valid_recv[:, None], recv_grads, jnp.zeros((), dtype=out_dt)).astype(tbl_dtype)
      next_d_tbl = d_tbl.at[safe_rows].add(masked_grads, mode="drop")
      return next_sent, next_d_tbl

    init_sent = jnp.zeros((table_shards, table_shards), dtype=jnp.int32)
    init_d_tbl = jnp.zeros((v_loc, d_dim), dtype=tbl_dtype)
    _, d_tbl = jax.lax.while_loop(_cond, _body, (init_sent, init_d_tbl))
    return d_tbl, None, None, None, None

  _lookup_sorted.defvjp(_lookup_sorted_fwd, _lookup_sorted_bwd)
  embeds = _lookup_sorted(table_local, sorted_local_row, counts_total, send_perm, inv_send_perm)
  return jnp.where(valid[:, None], embeds, jnp.zeros((), dtype=out_dt))


def _left_halo_exchange(
    tokens_local: jax.Array,
    *,
    halo: int,
    eos_token_id: int,
    length_axes: tuple[str, ...],
    length_shards: int,
) -> jax.Array:
  """Exchanges the trailing `halo` tokens from sequence shard `r - 1` to shard `r`."""
  b_loc = tokens_local.shape[0]
  eos_ctx = jnp.full((b_loc, halo), eos_token_id, dtype=tokens_local.dtype)
  if length_shards <= 1 or halo <= 0:
    return eos_ctx
  tail = tokens_local[:, -halo:]
  perm = [(i, (i + 1) % length_shards) for i in range(length_shards)]
  prev_tail = jax.lax.ppermute(tail, axis_name=length_axes, perm=perm)
  seq_rank = jax.lax.axis_index(length_axes)
  return jnp.where(seq_rank == 0, eos_ctx, prev_tail)


def row_sharded_embedding_lookup(
    table: jax.Array,
    ids: jax.Array,
    *,
    table_axes: tuple[str, ...],
    ids_specs: P,
    out_dtype: Any | None = None,
    buffer_factor: float = 2.0,
    mesh: Any = None,
) -> jax.Array:
  """Index-routed lookup into a row-sharded `(V, D)` embedding table over `mesh`.

  Args:
    table: Row-sharded embedding table of shape `(V, D)` sharded along axis 0 over `table_axes`.
    ids: Sharded integer indices of arbitrary shape `(*prefix,)` with `PartitionSpec` `ids_specs`.
    table_axes: Non-empty tuple of mesh axis names sharding `table` axis 0.
    ids_specs: `PartitionSpec` for `ids` inside `jax.shard_map`. Every axis in `table_axes` must
      appear in `ids_specs` so each table shard holds a distinct slice of tokens.
    out_dtype: Output embedding dtype (defaults to `table.dtype`).
    buffer_factor: Capacity multiplier (`recv_buf_len = ceil(buffer_factor * N_local)`) for
      multi-round overflow protection under skewed/Zipfian index distributions.
    mesh: Target JAX mesh (defaults to active abstract mesh).

  Returns:
    Embeddings of shape `(*prefix, D)` sharded as `P(*ids_specs, None)`.
  """
  if buffer_factor <= 0.0:
    raise ValueError(f"Expected buffer_factor > 0.0, got {buffer_factor}.")
  active_mesh = mesh if mesh is not None else jax.sharding.get_abstract_mesh()
  if active_mesh.empty:
    raise ValueError("row_sharded_embedding_lookup requires a non-empty JAX mesh.")
  table_shards = math.prod(int(active_mesh.shape[ax]) for ax in table_axes)
  if table_shards <= 1:
    raise ValueError(f"Expected table_axes product > 1, got {table_axes} (size {table_shards}).")
  if table.shape[0] % table_shards != 0:
    raise ValueError(f"Table vocab size {table.shape[0]} must be divisible by table_shards={table_shards}.")

  ids_axis_set: list[str] = []
  for entry in ids_specs:
    if entry is None:
      continue
    names = (entry,) if isinstance(entry, str) else tuple(entry)
    for n in names:
      if int(active_mesh.shape[n]) > 1 and n not in ids_axis_set:
        ids_axis_set.append(n)
  missing = set(table_axes) - set(ids_axis_set)
  if missing:
    raise ValueError(f"All table_axes {table_axes} must be sharded on ids_specs {ids_specs}; missing {sorted(missing)}.")

  eff_dtype = table.dtype if out_dtype is None else jnp.dtype(out_dtype)
  is_tpu = m3_kernels.is_tpu_mesh(active_mesh)
  table_spec = P(table_axes if len(table_axes) > 1 else table_axes[0], None)
  out_spec = P(*ids_specs, None)

  def _per_device(tbl_loc: jax.Array, ids_loc: jax.Array) -> jax.Array:
    prefix_shape = ids_loc.shape
    flat_embeds = _local_row_sharded_lookup(
        tbl_loc,
        ids_loc.reshape(-1),
        table_axes=table_axes,
        table_shards=table_shards,
        out_dtype=eff_dtype,
        buffer_factor=buffer_factor,
        is_tpu=is_tpu,
    )
    return flat_embeds.reshape(*prefix_shape, tbl_loc.shape[1])

  tbl_in = m3_kernels.ensure_explicit_spec(table, table_spec, active_mesh)
  ids_in = m3_kernels.ensure_explicit_spec(ids, ids_specs, active_mesh)
  return jax.shard_map(
      _per_device,
      mesh=active_mesh,
      in_specs=(table_spec, ids_specs),
      out_specs=out_spec,
      check_vma=False,
  )(tbl_in, ids_in)


def index_routed_ngram_embedding(
    table: jax.Array,
    input_ids: jax.Array,
    compute_ngram_ids_fn: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    eos_token_id: int,
    ngram_size: int,
    ple_embed_dim: int,
    table_axes: tuple[str, ...],
    batch_axes: tuple[str, ...],
    length_axes: tuple[str, ...],
    out_dtype: Any,
    buffer_factor: float = 2.0,
    mesh: Any = None,
) -> jax.Array:
  """Computes shard-local n-gram hashes (with CP halo) and index-routed PLE table lookup.

  Args:
    table: Row-sharded PLE n-gram table of shape `(V_pad, head_dim)`.
    input_ids: Sharded token IDs of shape `(B, S)` on `P(batch_axes, length_axes)`.
    compute_ngram_ids_fn: Callable `(input_ids_local, left_context) -> ngram_ids_local`
      returning `(B_local, S_local, ngram_heads)`.
    eos_token_id: EOS token ID used at document/sequence boundaries.
    ngram_size: N-gram window size (`halo = ngram_size - 1`).
    ple_embed_dim: Flattened PLE embedding width (`ngram_heads * head_dim`).
    table_axes: Active mesh axes sharding `table` axis 0.
    batch_axes: Active mesh axes sharding `input_ids` axis 0 (`()` if unsharded).
    length_axes: Active mesh axes sharding `input_ids` axis 1 (`()` if unsharded).
    out_dtype: Output embedding dtype.
    buffer_factor: Capacity-bounded overflow buffer multiplier for `ragged_all_to_all`.
    mesh: Active JAX mesh.

  Returns:
    PLE embeddings of shape `(B, S, ple_embed_dim)` sharded as `P(batch_axes, length_axes, None)`.
  """
  if buffer_factor <= 0.0:
    raise ValueError(f"Expected buffer_factor > 0.0, got {buffer_factor}.")
  active_mesh = mesh if mesh is not None else jax.sharding.get_abstract_mesh()
  if active_mesh.empty:
    raise ValueError("index_routed_ngram_embedding requires a non-empty JAX mesh.")
  table_shards = math.prod(int(active_mesh.shape[ax]) for ax in table_axes)
  if table_shards <= 1:
    raise ValueError(f"Expected table_axes product > 1, got {table_axes} (size {table_shards}).")
  if table.shape[0] % table_shards != 0:
    raise ValueError(f"Table vocab size {table.shape[0]} must be divisible by table_shards={table_shards}.")
  length_shards = math.prod(int(active_mesh.shape[ax]) for ax in length_axes) if length_axes else 1
  if set(batch_axes) & set(length_axes):
    raise ValueError(f"batch_axes {batch_axes} and length_axes {length_axes} must be disjoint.")
  token_axes = tuple(dict.fromkeys((*batch_axes, *length_axes)))
  missing = set(table_axes) - set(token_axes)
  if missing:
    raise ValueError(
        f"All table_axes {table_axes} must be covered by batch_axes {batch_axes} or length_axes {length_axes}; "
        f"missing {sorted(missing)}."
    )

  def _spec_entry(axes: tuple[str, ...]) -> str | tuple[str, ...] | None:
    if not axes:
      return None
    return axes[0] if len(axes) == 1 else axes

  table_spec = P(_spec_entry(table_axes), None)
  ids_spec = P(_spec_entry(batch_axes), _spec_entry(length_axes))
  out_spec = P(_spec_entry(batch_axes), _spec_entry(length_axes), None)
  is_tpu = m3_kernels.is_tpu_mesh(active_mesh)
  eff_dtype = jnp.dtype(out_dtype)
  halo = ngram_size - 1

  def _per_device(tbl_loc: jax.Array, tok_loc: jax.Array) -> jax.Array:
    b_loc, s_loc = tok_loc.shape
    left_ctx = _left_halo_exchange(
        tok_loc,
        halo=halo,
        eos_token_id=eos_token_id,
        length_axes=length_axes,
        length_shards=length_shards,
    )
    ngram_ids_loc = compute_ngram_ids_fn(tok_loc, left_ctx)
    flat_embeds = _local_row_sharded_lookup(
        tbl_loc,
        ngram_ids_loc.reshape(-1),
        table_axes=table_axes,
        table_shards=table_shards,
        out_dtype=eff_dtype,
        buffer_factor=buffer_factor,
        is_tpu=is_tpu,
    )
    return flat_embeds.reshape(b_loc, s_loc, ple_embed_dim)

  tbl_in = m3_kernels.ensure_explicit_spec(table, table_spec, active_mesh)
  tok_in = m3_kernels.ensure_explicit_spec(input_ids, ids_spec, active_mesh)
  return jax.shard_map(
      _per_device,
      mesh=active_mesh,
      in_specs=(table_spec, ids_spec),
      out_specs=out_spec,
      check_vma=False,
  )(tbl_in, tok_in)
