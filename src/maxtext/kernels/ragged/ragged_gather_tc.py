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

"""TensorCore (Pallas, DMA-driven) ragged gather / gather-reduce kernels.

Ported from Lineage's ops.py (ragged_gather_tc,
ragged_gather_reduce_tc and its metadata). Tokens are laid out as
(num_tokens, d0, d1) with d1 = 128 so that each token row is one contiguous DMA.
"""

import dataclasses
import functools
import logging
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# --- Pallas Kernels for Ragged Gather on TensorCore ---

_RAGGED_GATHER_TC_NUM_BUFFERS = 2


def ragged_gather_tc(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    num_tokens: jax.Array | int,
    max_out_tokens: int,
    block_size: int = 2048,
) -> jax.Array:
  """Executes a ragged gather on TensorCore using a Pallas kernel.

  Gathers `out[i] = x[indices[start + i]]` for `i < num_tokens`. Output rows
  `[num_tokens, max_out_tokens)` are left untouched (garbage from the
  uninitialized output buffer).

  The caller must guarantee `0 <= num_tokens <= max_out_tokens` and
  `0 <= start` with `start + num_tokens <= num_indices`; these are not checked
  at runtime and violating them is undefined behavior.

  Args:
    x: Input array of shape (num_input_tokens, d0, d1).
    indices: Indices into `x` of shape (num_indices,).
    start: Offset into `indices` of the first index to gather. May be a traced
      scalar; the kernel does not recompile per value.
    num_tokens: Number of indices to gather. May be a traced scalar; the kernel
      does not recompile per value.
    max_out_tokens: Length of the output buffer along axis 0.
    block_size: Number of output rows gathered per grid step, i.e. the number of
      concurrent row DMAs in flight per buffer. Clamped to `max_out_tokens`.

  Returns:
    Gathered array in an output buffer of shape (max_out_tokens, d0, d1).
  """
  if max_out_tokens <= 0:
    raise ValueError(f"max_out_tokens must be positive, got {max_out_tokens}.")
  if isinstance(num_tokens, int) and not 0 <= num_tokens <= max_out_tokens:
    raise ValueError(
        f"num_tokens must be in [0, {max_out_tokens}], got {num_tokens}."
    )
  if block_size <= 0:
    raise ValueError(f"block_size must be positive, got {block_size}.")
  num_indices = indices.shape[0]
  _, dim0, dim1 = x.shape
  if num_indices == 0 or (isinstance(num_tokens, int) and num_tokens == 0):
    return jnp.empty((max_out_tokens, dim0, dim1), dtype=x.dtype)

  block_size = min(block_size, max_out_tokens)
  num_buffers = _RAGGED_GATHER_TC_NUM_BUFFERS
  num_full_blocks = max_out_tokens // block_size
  rem = max_out_tokens % block_size
  num_blocks = num_full_blocks + (1 if rem > 0 else 0)

  # `indices` may exceed SMEM capacity, so each block stages only its window of
  # indices in SMEM. The window starts at an offset aligned to the HBM tile
  # (128 x i32) so the dynamic DMA slice needs no retiling, and has room for the
  # misalignment. It is clamped to `max_win_start` to stay in bounds, which
  # requires the (padded) indices length to be aligned and at least one window
  # long.
  idx_align = 128
  idx_window = pl.cdiv(block_size, idx_align) * idx_align + idx_align
  idx_pad = max((-num_indices) % idx_align, idx_window - num_indices)
  if idx_pad > 0:
    indices = jnp.pad(indices, (0, idx_pad))
  max_win_start = indices.shape[0] - idx_window

  def _ragged_gather_kernel(
      scalars_smem_ref,
      idx_hbm_ref,
      x_hbm_ref,
      o_hbm_ref,
      vmem_ref,
      idx_smem_ref,
      idx_sem,
      data_recv_sem,
      data_send_sem,
  ):
    block_idx = pl.program_id(0)
    buf_idx = block_idx % num_buffers
    start_idx = scalars_smem_ref[0]
    num_valid = scalars_smem_ref[1]

    def _run_step(step_idx, b_idx, action: str):
      base = step_idx * block_size
      row0 = start_idx + base
      win_start = jnp.minimum((row0 // idx_align) * idx_align, max_win_start)
      idx_off = row0 - win_start

      def _fetch_indices():
        # Synchronous; the row gathers read the indices as scalars when issued,
        # so a single SMEM buffer suffices.
        copy = pltpu.make_async_copy(
            idx_hbm_ref.at[
                pl.ds(pl.multiple_of(win_start, idx_align), idx_window)
            ],
            idx_smem_ref,
            idx_sem,
        )
        copy.start()
        copy.wait()

      def _chunk(off, size: int, chunk_action: str):
        """Issues `chunk_action` for block rows [off, off + size)."""
        if chunk_action == "data_start":
          step_unroll = math.gcd(size, 128)

          @pl.loop(0, size // step_unroll)
          def _(chunk_i):
            base_i = off + chunk_i * step_unroll
            src_indices = [
                idx_smem_ref[idx_off + base_i + u] for u in range(step_unroll)
            ]
            for u in range(step_unroll):
              pltpu.make_async_copy(
                  x_hbm_ref.at[pl.ds(src_indices[u], 1), :, :],
                  vmem_ref.at[b_idx, pl.ds(base_i + u, 1), :, :],
                  data_recv_sem.at[b_idx],
              ).start()

        elif chunk_action == "recv_wait":
          # Only the byte count matters for the wait; `o_hbm_ref` always has
          # at least `block_size` rows, whereas `x` may have fewer.
          pltpu.make_async_copy(
              o_hbm_ref.at[pl.ds(0, size), :, :],
              vmem_ref.at[b_idx, pl.ds(0, size), :, :],
              data_recv_sem.at[b_idx],
          ).wait()
        else:
          copy = pltpu.make_async_copy(
              vmem_ref.at[b_idx, pl.ds(off, size), :, :],
              o_hbm_ref.at[pl.ds(base + off, size), :, :],
              data_send_sem.at[b_idx],
          )
          if chunk_action == "send_start":
            copy.start()
          else:
            copy.wait()

      def _do_len(length: int):
        step_valid = jnp.minimum(jnp.maximum(num_valid - base, 0), length)

        def _over_valid_rows(chunk_action: str):
          # Fast path: the whole block is valid, so use one static-size chunk.
          @pl.when(step_valid == length)
          def _():
            _chunk(0, length, chunk_action)

          # Partially valid block: DMA sizes must be static, so split the
          # dynamic row count into power-of-two chunks, one per set bit.
          @pl.when(step_valid < length)
          def _():
            off = 0
            for k in reversed(range(length.bit_length())):
              size = 1 << k
              bit = step_valid & size

              def _chunk_if_bit_set(off=off, size=size):
                _chunk(off, size, chunk_action)

              pl.when(bit != 0)(_chunk_if_bit_set)

              off = off + bit

        if action == "data_start":

          @pl.when(step_valid > 0)
          def _():
            _fetch_indices()

          _over_valid_rows("data_start")
        elif action == "copy_out_start":
          # All recv waits must finish before any send starts, because the
          # per-row gather DMAs share one semaphore and complete out of order.
          _over_valid_rows("recv_wait")
          _over_valid_rows("send_start")
        elif action == "copy_out_wait":
          _over_valid_rows("send_wait")

      if rem == 0:
        _do_len(block_size)
      elif isinstance(step_idx, int):
        _do_len(rem if step_idx == num_blocks - 1 else block_size)
      else:
        jax.lax.cond(
            step_idx == num_blocks - 1,
            lambda: _do_len(rem),
            lambda: _do_len(block_size),
        )

    @pl.when(block_idx == 0)
    def _prologue():
      _run_step(0, 0, "data_start")

    _run_step(block_idx, buf_idx, "copy_out_start")

    # The grid covers only blocks overlapping `[0, num_tokens)`.
    num_steps = pl.num_programs(0)
    next_data_block = block_idx + 1

    @pl.when(next_data_block < num_steps)
    def _queue_next():
      next_data_buf = next_data_block % num_buffers

      @pl.when(block_idx >= 1)
      def _wait_prev_out():
        _run_step(block_idx - 1, next_data_buf, "copy_out_wait")

      _run_step(next_data_block, next_data_buf, "data_start")

    @pl.when(block_idx == num_steps - 1)
    def _epilogue():
      @pl.when(block_idx >= 1)
      def _wait_prev_out():
        prev_block = block_idx - 1
        _run_step(prev_block, prev_block % num_buffers, "copy_out_wait")

      _run_step(block_idx, buf_idx, "copy_out_wait")

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_shape = jax.ShapeDtypeStruct(
      (max_out_tokens, dim0, dim1),
      x.dtype,
      manual_axis_type=manual_axis_type,
  )
  scalars = jnp.stack([
      jnp.asarray(start, dtype=jnp.int32),
      jnp.asarray(num_tokens, dtype=jnp.int32),
  ])
  # Only launch grid steps for blocks overlapping `[0, num_tokens)`. When
  # `num_tokens` is traced, this is a dynamic grid size.
  num_steps = pl.cdiv(
      num_tokens if isinstance(num_tokens, int) else scalars[1], block_size
  )

  return pl.pallas_call(
      _ragged_gather_kernel,
      out_shape=out_shape,
      grid=(num_steps,),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
      scratch_shapes=[
          pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
          pltpu.SMEM((idx_window,), indices.dtype),
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA((num_buffers,)),
          pltpu.SemaphoreType.DMA((num_buffers,)),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY,),
      ),
      name="ragged_gather_tc",
  )(scalars, indices, x)


# --- Ragged Gather Reduce on TensorCore ---


# While block i is reduced, the row DMAs of block i + 2 are issued in the same
# loop and block i + 1's are in flight, and block i - 1 is being stored.
_RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS = 4
_RAGGED_GATHER_REDUCE_TC_LOOKAHEAD = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS - 2
# Static bodies per trip of the row DMA issue and reduce loops.
_RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL = 64
_RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL = 8
# Alignment of dynamic 1D int32 HBM slices.
_RAGGED_GATHER_REDUCE_TC_ALIGN = 128
# Slots per chunk of the two-level search for block boundaries.
_RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK = 64
# Block table entries per block: first slot, slot count, first token, token
# count.
_RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS = 4
# VMEM of the block buffers, within Ghostfish's max valid 65,472 KiB limit.
_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT = 65472 * 1024
_RAGGED_GATHER_REDUCE_TC_VMEM_BUDGET = 56 * 1024 * 1024


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RaggedGatherReduceMetadata:
  """Routing of `ragged_gather_reduce_tc`.

  Built by `ragged_gather_reduce_tc_metadata`. The valid slots of `indices` are
  compacted so that each output token's valid rows are contiguous and in buffer
  order, and output tokens are split into blocks of consecutive tokens whose
  valid rows plus output rows fit in `block_size` VMEM rows.

  Attributes:
    src_rows: Buffer row of each compacted slot, padded to an aligned length of
      at least one SMEM slot window.
    tokens: Output token of each compacted slot, padded like `src_rows`.
    blocks: Number of blocks, followed by the first slot, slot count, first
      token and token count of each block.
    block_size: Maximum VMEM rows per block (valid input rows plus output
      tokens), which also bounds the row DMAs in flight per block.
    num_indices: Length of the `indices` the metadata was derived from
      (`num_out_tokens * top_k`, or the buffer cap for windowed indices).
  """

  src_rows: jax.Array
  tokens: jax.Array
  blocks: jax.Array
  block_size: int = jax.tree.static()
  num_indices: int = jax.tree.static()


def _ragged_gather_reduce_tc_layout(
    num_indices: int,
    top_k: int,
    block_size: int,
    num_out_tokens: int | None = None,
) -> tuple[int, int, int]:
  """Returns (max_blocks, window, slots_len) of the metadata layout.

  A token costs its valid rows plus its output row. Block `b` takes the tokens
  whose exclusive cost prefix is in `[b * span, (b + 1) * span)`, where
  `span = block_size - top_k`, so it costs at most `block_size` VMEM rows. Each
  block stages a `window` of the compacted slots in SMEM, starting at an aligned
  offset, so the slot arrays are padded to `slots_len`.

  Args:
    num_indices: Length of the forward gather's indices.
    top_k: Number of occurrences of each token id in the indices.
    block_size: Maximum VMEM rows per block.
    num_out_tokens: Number of output tokens; defaults to `num_indices // top_k`.
  """
  if num_out_tokens is None:
    num_out_tokens = num_indices // top_k
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  max_blocks = pl.cdiv(num_indices + num_out_tokens, block_size - top_k)
  window = pl.cdiv(block_size, align) * align + align
  slots_len = pl.cdiv(max(num_indices, window), align) * align
  return max_blocks, window, slots_len


def _check_ragged_gather_reduce_tc_args(top_k: int, block_size: int):
  if top_k <= 0:
    raise ValueError(f"top_k must be positive, got {top_k}.")
  if block_size <= top_k:
    raise ValueError(
        f"block_size must exceed top_k ({top_k}) so that a block fits a token"
        f" and its rows, got {block_size}."
    )


@functools.cache
def _warn_packed_sort_fallback(
    n: int, max_key: int, pos_bits: int, key_bits: int
) -> None:
  """Logs, once per shape at trace time, that packed_stable_sort took the slower tuple sort."""
  logging.warning(
      "packed_stable_sort: %d position bits (n=%d) + %d key bits (max_key=%d) >"
      " 32; falling back to the (key, position) tuple sort.",
      pos_bits,
      n,
      key_bits,
      max_key,
  )


def packed_stable_sort(
    key: jax.Array, max_key: int
) -> tuple[jax.Array, jax.Array]:
  """Stable sort of `arange(len(key))` by non-negative int `key` <= `max_key`.

  Sorts the single uint32 array `key << pos_bits | position` when it fits in 32
  bits, or `key * n + position` when `(max_key + 1) * n <= 2**32` (e.g. EP=16
  `max_key=65536, n=43008`), else falls back to the `(key, position)` tuple
  sort. The positions are distinct, so every packed value is unique and the
  unstable single-array sort gives the stable order.

  Args:
    key: 1-D non-negative integer array of sort keys.
    max_key: Upper bound on values in `key`.

  Returns:
    Tuple of `(sorted_key, sorted_positions)` sorted stably by `key`.
  """
  n = key.shape[0]
  positions = jnp.arange(n, dtype=jnp.int32)
  pos_bits = max(1, (n - 1).bit_length())
  key_bits = max(1, int(max_key).bit_length())
  if pos_bits + key_bits <= 32:
    shift = jnp.uint32(pos_bits)
    packed_key = (key.astype(jnp.uint32) << shift) | positions.astype(
        jnp.uint32
    )
    sorted_packed = jax.lax.sort(packed_key, is_stable=False)
    sorted_positions = (sorted_packed & jnp.uint32((1 << pos_bits) - 1)).astype(
        jnp.int32
    )
    return (sorted_packed >> shift).astype(jnp.int32), sorted_positions
  if n > 0 and (int(max_key) + 1) * n <= (1 << 32):
    radix = jnp.uint32(n)
    packed_key = key.astype(jnp.uint32) * radix + positions.astype(jnp.uint32)
    sorted_packed = jax.lax.sort(packed_key, is_stable=False)
    sorted_key_u32 = sorted_packed // radix
    sorted_positions = (sorted_packed - sorted_key_u32 * radix).astype(
        jnp.int32
    )
    return sorted_key_u32.astype(jnp.int32), sorted_positions
  _warn_packed_sort_fallback(n, int(max_key), pos_bits, key_bits)
  sorted_key, sorted_positions = jax.lax.sort(
      (key.astype(jnp.int32), positions), num_keys=1, is_stable=True
  )
  return sorted_key, sorted_positions


def _packed_stable_sort_with_weights_impl(
    key: jax.Array, max_key: int, weights_flat: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Sorts `weights_flat` as a payload alongside `packed_stable_sort(key, max_key)`."""
  n = key.shape[0]
  positions = jnp.arange(n, dtype=jnp.int32)
  if n == 0:
    return key.astype(jnp.int32), positions, weights_flat
  pos_bits = max(1, (n - 1).bit_length())
  key_bits = max(1, int(max_key).bit_length())
  if pos_bits + key_bits <= 32:
    shift = jnp.uint32(pos_bits)
    packed_key = (key.astype(jnp.uint32) << shift) | positions.astype(
        jnp.uint32
    )
    sorted_packed, sorted_weights = jax.lax.sort(
        (packed_key, weights_flat), dimension=0, is_stable=False, num_keys=1
    )
    sorted_positions = (sorted_packed & jnp.uint32((1 << pos_bits) - 1)).astype(
        jnp.int32
    )
    return (
        (sorted_packed >> shift).astype(jnp.int32),
        sorted_positions,
        sorted_weights,
    )
  if (int(max_key) + 1) * n <= (1 << 32):
    radix = jnp.uint32(n)
    packed_key = key.astype(jnp.uint32) * radix + positions.astype(jnp.uint32)
    sorted_packed, sorted_weights = jax.lax.sort(
        (packed_key, weights_flat), dimension=0, is_stable=False, num_keys=1
    )
    sorted_key_u32 = sorted_packed // radix
    sorted_positions = (sorted_packed - sorted_key_u32 * radix).astype(
        jnp.int32
    )
    return sorted_key_u32.astype(jnp.int32), sorted_positions, sorted_weights
  _warn_packed_sort_fallback(n, int(max_key), pos_bits, key_bits)
  sorted_key, sorted_positions, sorted_weights = jax.lax.sort(
      (key.astype(jnp.int32), positions, weights_flat),
      dimension=0,
      num_keys=1,
      is_stable=True,
  )
  return sorted_key, sorted_positions, sorted_weights


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def packed_stable_sort_with_weights(
    key: jax.Array, max_key: int, weights_flat: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Stable sort of `(arange(n), weights_flat)` by non-negative int `key`.

  Avoids both the forward `gather_fusion` (`w_flat[slot_order]`) and the
  backward `scatter` by carrying `weights_flat` directly as a sort payload in
  forward and unsorting its cotangent via inverse sort on the bijection
  `sorted_positions` in backward.

  Args:
    key: Non-negative integer keys `[N]` with values `<= max_key`.
    max_key: Static upper bound on values in `key`.
    weights_flat: Differentiable 1D routing weights `[N]` (`bf16` or `f32`).

  Returns:
    Tuple `(sorted_key, sorted_positions, sorted_weights)`.
  """
  return _packed_stable_sort_with_weights_impl(key, max_key, weights_flat)


def _packed_stable_sort_with_weights_fwd(
    key: jax.Array, max_key: int, weights_flat: jax.Array
):
  sorted_key, sorted_positions, sorted_weights = (
      _packed_stable_sort_with_weights_impl(key, max_key, weights_flat)
  )
  return (sorted_key, sorted_positions, sorted_weights), (sorted_positions,)


def _packed_stable_sort_with_weights_bwd(
    max_key: int, res: tuple[jax.Array], g_out
):
  del max_key
  (sorted_positions,) = res
  _, _, g_sorted_weights = g_out
  if g_sorted_weights.shape[0] == 0:
    return (None, g_sorted_weights)
  # `sorted_positions` is a bijection on `{0, ..., N - 1}`, so sorting
  # `(sorted_positions, g_sorted_weights)` by `sorted_positions` places every
  # cotangent back at its exact source slot with zero scatter and zero gather.
  _, g_weights = jax.lax.sort(
      (sorted_positions.astype(jnp.uint32), g_sorted_weights),
      dimension=0,
      is_stable=False,
      num_keys=1,
  )
  return (None, g_weights)


packed_stable_sort_with_weights.defvjp(
    _packed_stable_sort_with_weights_fwd,
    _packed_stable_sort_with_weights_bwd,
)


# --- TC-only 1D Unique Gather / Scatter for MoE Routing Weights ---

_TC_1D_TILE_SUBLANES = 8
_TC_1D_TILE_LANES = 128
_TC_1D_CHUNK_SIZE = _TC_1D_TILE_SUBLANES * _TC_1D_TILE_LANES  # 1024
_TC_1D_BWD_CHUNK_SUBLANES = 256
_TC_1D_BWD_CHUNK_LANES = 256
_TC_1D_BWD_LANE_SHIFT = 8  # log2(256)
_TC_1D_BWD_CHUNK_SHIFT = 16  # log2(256 * 256)
_TC_1D_BWD_CHUNK_SIZE = (
    _TC_1D_BWD_CHUNK_SUBLANES * _TC_1D_BWD_CHUNK_LANES
)  # 65536 = 1 << 16
_TC_1D_BWD_K_STEP = 256
_TC_1D_BWD_M_CHUNK_ELEMS = _TC_1D_TILE_SUBLANES * _TC_1D_BWD_K_STEP  # 2048
_TC_1D_FWD_MAX_VMEM_BYTES = 32 * 1024 * 1024


def _tc_gather_1d_sorted_pallas(
    w_flat_f32: jax.Array,
    s_sorted: jax.Array,
    n_orig: int,
    *,
    interpret: bool = False,
) -> jax.Array:
  """Gathers `w_flat_f32[s_sorted]` on TC via VMEM `take_along_axis` (0 scatter/gather HLO)."""
  n = w_flat_f32.shape[0]
  m = s_sorted.shape[0]
  chunk_sz = _TC_1D_CHUNK_SIZE
  num_w_chunks = pl.cdiv(n, chunk_sz)
  n_pad = num_w_chunks * chunk_sz
  num_m_chunks = pl.cdiv(m, chunk_sz)
  m_pad = num_m_chunks * chunk_sz

  w_bytes = n_pad * 4
  if w_bytes > _TC_1D_FWD_MAX_VMEM_BYTES:
    raise ValueError(
        f"tc_gather_1d_unique: padded w_flat ({w_bytes} B) exceeds VMEM"
        f" budget ({_TC_1D_FWD_MAX_VMEM_BYTES} B)."
    )

  w_3d = jnp.pad(w_flat_f32, (0, n_pad - n)).reshape(
      num_w_chunks, _TC_1D_TILE_SUBLANES, _TC_1D_TILE_LANES
  )
  s_3d = jnp.pad(
      s_sorted, (0, m_pad - m), constant_values=jnp.int32(n_pad)
  ).reshape(num_m_chunks, _TC_1D_TILE_SUBLANES, _TC_1D_TILE_LANES)

  s_first = s_3d[:, 0, 0]
  has_valid = s_first < n_orig
  s_valid_max = jnp.max(
      jnp.where(s_3d < n_orig, s_3d, jnp.int32(0)), axis=(1, 2)
  )
  c_starts = jnp.where(
      has_valid, s_first >> 10, jnp.int32(num_w_chunks)
  ).astype(jnp.int32)
  c_ends = jnp.where(has_valid, (s_valid_max >> 10) + 1, c_starts).astype(
      jnp.int32
  )

  def _gather_kernel(w_vmem, s_vmem, c_starts_smem, c_ends_smem, out_vmem):
    t = pl.program_id(0)
    c_start = c_starts_smem[t]
    c_end = c_ends_smem[t]
    s_tile = s_vmem[0, :, :]
    row_tile = s_tile >> 7
    lane_tile = s_tile & 127

    def _chunk_body(c, acc):
      w_chunk = w_vmem[c, :, :]
      base_row = c * _TC_1D_TILE_SUBLANES
      for r in range(_TC_1D_TILE_SUBLANES):
        r_idx = jnp.full(
            (_TC_1D_TILE_SUBLANES, _TC_1D_TILE_LANES), r, dtype=jnp.int32
        )
        row_bcast = jnp.take_along_axis(
            w_chunk,
            r_idx,
            axis=-2,
            mode="promise_in_bounds",
            wrap_negative_indices=False,
        )
        gathered = jnp.take_along_axis(
            row_bcast,
            lane_tile,
            axis=-1,
            mode="promise_in_bounds",
            wrap_negative_indices=False,
        )
        acc = jnp.where(row_tile == (base_row + r), gathered, acc)
      return acc

    acc = jax.lax.fori_loop(
        c_start,
        c_end,
        _chunk_body,
        jnp.zeros(
            (_TC_1D_TILE_SUBLANES, _TC_1D_TILE_LANES), dtype=jnp.float32
        ),
    )
    out_vmem[0, :, :] = acc

  manual_axis_type = getattr(jax.typeof(w_flat_f32), "manual_axis_type", None)
  out_shape = jax.ShapeDtypeStruct(
      (num_m_chunks, _TC_1D_TILE_SUBLANES, _TC_1D_TILE_LANES),
      jnp.float32,
      manual_axis_type=manual_axis_type,
  )
  out_3d = pl.pallas_call(
      _gather_kernel,
      out_shape=out_shape,
      grid=(num_m_chunks,),
      in_specs=[
          pl.BlockSpec(
              (num_w_chunks, _TC_1D_TILE_SUBLANES, _TC_1D_TILE_LANES),
              lambda t: (0, 0, 0),
          ),
          pl.BlockSpec(
              (1, _TC_1D_TILE_SUBLANES, _TC_1D_TILE_LANES),
              lambda t: (t, 0, 0),
          ),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
      ],
      out_specs=pl.BlockSpec(
          (1, _TC_1D_TILE_SUBLANES, _TC_1D_TILE_LANES),
          lambda t: (t, 0, 0),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY,),
      ),
      interpret=interpret,
      name="tc_gather_1d_unique",
  )(w_3d, s_3d, c_starts, c_ends)
  return out_3d.reshape(m_pad)[:m]


def _tc_scatter_1d_sorted_pallas(
    s_sorted: jax.Array,
    gw_sorted_f32: jax.Array,
    num_out: int,
    *,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    interpret: bool = False,
) -> jax.Array:
  """Scatters sorted unique `(s_sorted, gw_sorted_f32)` into `zeros(num_out)` on TC.

  Keeps `(s_sorted, gw_sorted_f32)` in VMEM (`~336 KiB` at `M=43,008`) as
  `(num_m_chunks, 8, 256)` (`2,048` elems/chunk) and scatters each `65,536`-elem
  output chunk `(256, 256)` via MXU one-hot outer-product accumulation:
  `acc[r, c] += sum_k (sub[k]==r ? gw[k] : 0) * (lane[k]==c)`. Mosaic fuses the
  one-hot predicate directly into MXU push (`vmatpush1.bf16.xpose.msra`),
  eliminating serial XLU binary search (`16.7 us` on TPU7x). `bf16` output runs
  in 1 pass; `f32` (or other) output uses a 3-pass `bf16` mantissa split
  (`hi + mid + lo`) for 0-ULP bit-exact `f32` accumulation.
  """
  out_dt = jnp.dtype(out_dtype)
  m = s_sorted.shape[0]
  chunk_sublanes = _TC_1D_BWD_CHUNK_SUBLANES
  chunk_lanes = _TC_1D_BWD_CHUNK_LANES
  chunk_sz = _TC_1D_BWD_CHUNK_SIZE
  k_step = _TC_1D_BWD_K_STEP
  m_tile_elems = _TC_1D_BWD_M_CHUNK_ELEMS

  num_w_chunks = pl.cdiv(num_out, chunk_sz)
  n_pad = num_w_chunks * chunk_sz
  num_m_chunks = pl.cdiv(max(m, 1), m_tile_elems)
  m_pad = num_m_chunks * m_tile_elems

  vmem_bytes = 2 * m_pad * 4
  if vmem_bytes > _TC_1D_FWD_MAX_VMEM_BYTES:
    raise ValueError(
        f"tc_scatter_1d_unique: padded (s_sorted, gw_sorted) ({vmem_bytes} B)"
        f" exceeds VMEM budget ({_TC_1D_FWD_MAX_VMEM_BYTES} B)."
    )

  s_3d = jnp.pad(
      s_sorted, (0, m_pad - m), constant_values=jnp.int32(n_pad)
  ).reshape(num_m_chunks, _TC_1D_TILE_SUBLANES, k_step)
  gw_3d = jnp.pad(
      gw_sorted_f32, (0, m_pad - m), constant_values=jnp.float32(0.0)
  ).reshape(num_m_chunks, _TC_1D_TILE_SUBLANES, k_step)

  s_mc = s_3d.reshape(num_m_chunks, m_tile_elems)
  mc_first = s_mc[:, 0]
  mc_valid_max = jnp.max(
      jnp.where(s_mc < n_pad, s_mc, jnp.int32(-1)), axis=1
  )
  mc_has_valid = mc_valid_max >= 0
  mc_first_chunk = jnp.where(
      mc_has_valid,
      mc_first >> _TC_1D_BWD_CHUNK_SHIFT,
      jnp.int32(num_w_chunks),
  )
  mc_last_chunk = jnp.where(
      mc_has_valid,
      mc_valid_max >> _TC_1D_BWD_CHUNK_SHIFT,
      jnp.int32(num_w_chunks),
  )
  c_ids = jnp.arange(num_w_chunks, dtype=jnp.int32)
  mc_starts = jnp.sum(
      (mc_last_chunk[None, :] < c_ids[:, None]).astype(jnp.int32), axis=1
  )
  mc_ends = jnp.sum(
      (mc_first_chunk[None, :] <= c_ids[:, None]).astype(jnp.int32), axis=1
  )

  r_rep = chunk_sublanes // _TC_1D_TILE_SUBLANES
  c_rep = chunk_lanes // _TC_1D_TILE_SUBLANES
  is_bf16_out = out_dt == jnp.dtype(jnp.bfloat16)

  def _scatter_kernel(
      s_vmem, gw_vmem, mc_starts_smem, mc_ends_smem, out_vmem
  ):
    c = pl.program_id(0)
    mc_start = mc_starts_smem[c]
    mc_end = mc_ends_smem[c]
    base = c * chunk_sz
    row_ids = jnp.broadcast_to(
        jnp.arange(chunk_sublanes, dtype=jnp.int32)[:, None],
        (chunk_sublanes, k_step),
    )
    col_ids = jnp.broadcast_to(
        jnp.arange(chunk_lanes, dtype=jnp.int32)[:, None],
        (chunk_lanes, k_step),
    )

    def _mc_body(mc, acc):
      s_tile = s_vmem[mc, :, :]
      gw_tile = gw_vmem[mc, :, :]
      local_s = s_tile - base
      in_chunk = (local_s >= 0) & (local_s < chunk_sz)
      sub_tile = jnp.where(
          in_chunk, local_s >> _TC_1D_BWD_LANE_SHIFT, jnp.int32(-1)
      )
      lane_tile = local_s & (chunk_lanes - 1)
      gw_tile_clean = jnp.where(in_chunk, gw_tile, jnp.float32(0.0))

      for u in range(_TC_1D_TILE_SUBLANES):
        u_idx = jnp.full((_TC_1D_TILE_SUBLANES, k_step), u, dtype=jnp.int32)
        sub_8 = jnp.take_along_axis(
            sub_tile,
            u_idx,
            axis=-2,
            mode="promise_in_bounds",
            wrap_negative_indices=False,
        )
        lane_8 = jnp.take_along_axis(
            lane_tile,
            u_idx,
            axis=-2,
            mode="promise_in_bounds",
            wrap_negative_indices=False,
        )
        gw_8 = jnp.take_along_axis(
            gw_tile_clean,
            u_idx,
            axis=-2,
            mode="promise_in_bounds",
            wrap_negative_indices=False,
        )
        sub_mask = jnp.tile(sub_8, (r_rep, 1)) == row_ids
        b_t = (jnp.tile(lane_8, (c_rep, 1)) == col_ids).astype(jnp.bfloat16)
        if is_bf16_out:
          a_mat = jnp.where(
              sub_mask,
              jnp.tile(gw_8.astype(jnp.bfloat16), (r_rep, 1)),
              jnp.bfloat16(0.0),
          )
          acc = acc + jax.lax.dot_general(
              a_mat,
              b_t,
              (((1,), (1,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
        else:
          hi_8 = gw_8.astype(jnp.bfloat16)
          rem1_8 = gw_8 - hi_8.astype(jnp.float32)
          mid_8 = rem1_8.astype(jnp.bfloat16)
          lo_8 = (rem1_8 - mid_8.astype(jnp.float32)).astype(jnp.bfloat16)
          for part_8 in (hi_8, mid_8, lo_8):
            a_mat = jnp.where(
                sub_mask, jnp.tile(part_8, (r_rep, 1)), jnp.bfloat16(0.0)
            )
            acc = acc + jax.lax.dot_general(
                a_mat,
                b_t,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
      return acc

    acc = jax.lax.fori_loop(
        mc_start,
        mc_end,
        _mc_body,
        jnp.zeros((chunk_sublanes, chunk_lanes), dtype=jnp.float32),
    )
    out_vmem[0, :, :] = acc.astype(out_dt)

  manual_axis_type = getattr(
      jax.typeof(gw_sorted_f32), "manual_axis_type", None
  )
  out_shape = jax.ShapeDtypeStruct(
      (num_w_chunks, chunk_sublanes, chunk_lanes),
      out_dt,
      manual_axis_type=manual_axis_type,
  )
  out_3d = pl.pallas_call(
      _scatter_kernel,
      out_shape=out_shape,
      grid=(num_w_chunks,),
      in_specs=[
          pl.BlockSpec(
              (num_m_chunks, _TC_1D_TILE_SUBLANES, k_step),
              lambda c: (0, 0, 0),
          ),
          pl.BlockSpec(
              (num_m_chunks, _TC_1D_TILE_SUBLANES, k_step),
              lambda c: (0, 0, 0),
          ),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
      ],
      out_specs=pl.BlockSpec(
          (1, chunk_sublanes, chunk_lanes),
          lambda c: (c, 0, 0),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.PARALLEL,),
      ),
      interpret=interpret,
      name="tc_scatter_1d_unique",
  )(s_3d, gw_3d, mc_starts, mc_ends)
  return out_3d.reshape(n_pad)[:num_out]


def tc_scatter_1d_unique(
    g_w_rows: jax.Array,
    slot_order: jax.Array,
    count: jax.Array | int,
    *,
    num_out: int,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    use_pallas: bool = True,
    interpret: bool = False,
) -> jax.Array:
  """Zero-scatter TC transpose of `tc_gather_1d_unique` for unique `slot_order[:count]`.

  Args:
    g_w_rows: 1D gradient weights of shape `[M]`.
    slot_order: 1D int32 destination indices of shape `[M]`, unique on
      `[:count]` in `[0, num_out)`.
    count: Scalar number of valid entries in `slot_order`.
    num_out: Static output length `N`.
    out_dtype: Output dtype (defaults to `bfloat16`; `float32` uses 0-ULP
      3-pass mantissa split).
    use_pallas: Use Pallas TC MXU kernel (`False` for CPU fallback).
    interpret: Run Pallas kernel in CPU interpret mode (for unit tests).

  Returns:
    Scattered 1D array of shape `[num_out]` and dtype `out_dtype`.
  """
  m = slot_order.shape[0]
  if num_out == 0 or m == 0:
    return jnp.zeros((num_out,), dtype=out_dtype)
  valid = jnp.arange(m, dtype=jnp.int32) < jnp.asarray(count, dtype=jnp.int32)
  if not use_pallas:
    safe_idx = jnp.where(valid, slot_order, jnp.int32(0))
    vals = jnp.where(valid, g_w_rows.astype(jnp.float32), jnp.float32(0.0))
    return (
        jnp.zeros((num_out,), dtype=jnp.float32)
        .at[safe_idx]
        .add(vals)
        .astype(out_dtype)
    )
  n_pad = pl.cdiv(num_out, _TC_1D_BWD_CHUNK_SIZE) * _TC_1D_BWD_CHUNK_SIZE
  s_masked = jnp.where(
      valid, slot_order.astype(jnp.int32), jnp.int32(n_pad)
  )
  gw_masked = jnp.where(
      valid, g_w_rows.astype(jnp.float32), jnp.float32(0.0)
  )
  s_sorted, gw_sorted = jax.lax.sort(
      (s_masked, gw_masked), dimension=0, is_stable=False, num_keys=1
  )
  return _tc_scatter_1d_sorted_pallas(
      s_sorted, gw_sorted, num_out, out_dtype=out_dtype, interpret=interpret
  )


def _tc_gather_1d_unique_impl(
    w_flat: jax.Array,
    slot_order: jax.Array,
    count: jax.Array,
    out_dtype: Any,
    use_pallas: bool,
    interpret: bool,
) -> jax.Array:
  """Forward body of `tc_gather_1d_unique`."""
  n = w_flat.shape[0]
  m = slot_order.shape[0]
  if n == 0 or m == 0:
    return jnp.zeros((m,), dtype=out_dtype)
  valid = jnp.arange(m, dtype=jnp.int32) < count.astype(jnp.int32)
  if not use_pallas:
    safe_idx = jnp.where(valid, slot_order, jnp.int32(0))
    return jnp.where(
        valid,
        w_flat[safe_idx].astype(out_dtype),
        jnp.asarray(0, dtype=out_dtype),
    )
  n_pad = pl.cdiv(n, _TC_1D_CHUNK_SIZE) * _TC_1D_CHUNK_SIZE
  s_masked = jnp.where(
      valid, slot_order.astype(jnp.int32), jnp.int32(n_pad)
  )
  pos = jnp.arange(m, dtype=jnp.int32)
  s_sorted, pos_sorted = jax.lax.sort(
      (s_masked, pos), dimension=0, is_stable=False, num_keys=1
  )
  w_sorted_f32 = _tc_gather_1d_sorted_pallas(
      w_flat.astype(jnp.float32), s_sorted, n, interpret=interpret
  )
  if w_flat.dtype == jnp.bfloat16 and m <= (1 << 16):
    # Pack `(pos_sorted << 16) | bf16_bits` for a 1-array u32 inverse sort.
    w_u16 = jax.lax.bitcast_convert_type(
        w_sorted_f32.astype(jnp.bfloat16), jnp.uint16
    )
    packed = (pos_sorted.astype(jnp.uint32) << jnp.uint32(16)) | w_u16.astype(
        jnp.uint32
    )
    unsorted = jax.lax.sort(packed, dimension=0, is_stable=False)
    w_rows_bf16 = jax.lax.bitcast_convert_type(
        (unsorted & jnp.uint32(0xFFFF)).astype(jnp.uint16), jnp.bfloat16
    )
    return w_rows_bf16.astype(out_dtype)
  _, w_rows_f32 = jax.lax.sort(
      (pos_sorted.astype(jnp.uint32), w_sorted_f32),
      dimension=0,
      is_stable=False,
      num_keys=1,
  )
  return w_rows_f32.astype(out_dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _tc_gather_1d_unique_core(
    w_flat: jax.Array,
    slot_order: jax.Array,
    count: jax.Array,
    num_out: int,
    in_dtype: Any,
    out_dtype: Any,
    use_pallas: bool,
    interpret: bool,
) -> jax.Array:
  del num_out, in_dtype
  return _tc_gather_1d_unique_impl(
      w_flat, slot_order, count, out_dtype, use_pallas, interpret
  )


def _tc_gather_1d_unique_fwd(
    w_flat: jax.Array,
    slot_order: jax.Array,
    count: jax.Array,
    num_out: int,
    in_dtype: Any,
    out_dtype: Any,
    use_pallas: bool,
    interpret: bool,
):
  del num_out, in_dtype
  w_rows = _tc_gather_1d_unique_impl(
      w_flat, slot_order, count, out_dtype, use_pallas, interpret
  )
  return w_rows, (slot_order, count)


def _tc_gather_1d_unique_bwd(
    num_out: int,
    in_dtype: Any,
    out_dtype: Any,
    use_pallas: bool,
    interpret: bool,
    res: tuple[jax.Array, jax.Array],
    g_w_rows: jax.Array,
):
  del out_dtype
  slot_order, count = res
  g_w_flat = tc_scatter_1d_unique(
      g_w_rows,
      slot_order,
      count,
      num_out=num_out,
      out_dtype=in_dtype,
      use_pallas=use_pallas,
      interpret=interpret,
  )
  return (g_w_flat, None, None)


_tc_gather_1d_unique_core.defvjp(
    _tc_gather_1d_unique_fwd, _tc_gather_1d_unique_bwd
)


def tc_gather_1d_unique(
    w_flat: jax.Array,
    slot_order: jax.Array,
    count: jax.Array | int,
    *,
    out_dtype: jax.typing.DTypeLike = jnp.float32,
    use_pallas: bool = True,
    interpret: bool = False,
) -> jax.Array:
  """Zero-gather, zero-scatter TC 1D weight gather/scatter for unique `slot_order[:count]`.

  Computes `out[i] = w_flat[slot_order[i]]` for `i < count` and `0` otherwise,
  where `slot_order[:count]` contains strictly unique indices in `[0, len(w_flat))`.
  Avoids carrying `w_flat` through the full `N`-element expert sort while
  emitting zero HLO `gather` and zero HLO `scatter` ops on TPU.

  Args:
    w_flat: 1D routing weights of shape `[N]`.
    slot_order: 1D int32 indices of shape `[M]`, unique on `[:count]`.
    count: Scalar number of valid entries in `slot_order`.
    out_dtype: Output dtype (defaults to `float32`).
    use_pallas: Use Pallas TC kernels (`False` for CPU fallback).
    interpret: Run Pallas kernels in CPU interpret mode (for unit tests).

  Returns:
    Gathered weights of shape `[M]` and dtype `out_dtype`.
  """
  return _tc_gather_1d_unique_core(
      w_flat,
      slot_order,
      jnp.asarray(count, dtype=jnp.int32),
      w_flat.shape[0],
      w_flat.dtype,
      jnp.dtype(out_dtype),
      use_pallas,
      interpret,
  )


def ragged_gather_reduce_tc_metadata(
    indices: jax.Array,
    start: jax.Array | int,
    num_tokens: jax.Array | int,
    *,
    top_k: int,
    block_size: int = 896,
    num_out_tokens: int | None = None,
    packed_sort: bool = False,
) -> RaggedGatherReduceMetadata:
  """Derives the routing consumed by `ragged_gather_reduce_tc`.

  The result depends only on the routing, not on the token data, so it can be
  computed once and shared by every `ragged_gather_reduce_tc` call with the
  same `indices`, `start` and `num_tokens`.

  The valid slots, those in the `[start, start + num_tokens)` window, are
  compacted by a stable sort on their token, so a token's valid rows are
  contiguous and in buffer order. Output tokens are then split into blocks of
  consecutive tokens that each fit `block_size` VMEM rows of input rows and
  output tokens, so sparse windows get blocks of many tokens and dense windows
  blocks of many rows.

  Each token id must appear exactly `top_k` times in `indices` (e.g.
  `indices = sort_indices // top_k`). The caller must guarantee `0 <= start`,
  `0 <= num_tokens` and `start + num_tokens <= num_indices`; these are not
  checked at runtime and violating them is undefined behavior.

  Args:
    indices: Indices used by the forward ragged gather, of shape (num_out_tokens
      * top_k,), with values in [0, num_out_tokens).
    start: Offset into `indices` of the index that formed the first buffer row.
      May be a traced scalar.
    num_tokens: Number of valid rows in the buffer. May be a traced scalar.
    top_k: Number of occurrences of each token id in `indices`.
    block_size: Maximum VMEM rows (valid input rows plus output tokens) per
      block of the kernel. Must exceed `top_k`, and `ragged_gather_reduce_tc`
      requires its block buffers to fit in VMEM for the token shape.
    num_out_tokens: Number of output tokens. Defaults to `num_indices // top_k`;
      set it when `indices` is already the `[start, start + cap)` window of the
      full indices (then pass `start=0`).
    packed_sort: Compact the slots with `packed_stable_sort` (same result, a
      single-array sort) instead of the (key, position) tuple sort.

  Returns:
    The compacted slots and block table.
  """
  _check_ragged_gather_reduce_tc_args(top_k, block_size)
  num_indices = indices.shape[0]
  if num_out_tokens is None:
    if num_indices % top_k:
      raise ValueError(
          f"indices length ({num_indices}) must be divisible by top_k"
          f" ({top_k})."
      )
    num_out_tokens = num_indices // top_k
  if isinstance(num_tokens, int) and num_tokens < 0:
    raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
  if isinstance(start, int) and start < 0:
    raise ValueError(f"start must be non-negative, got {start}.")
  if (
      isinstance(start, int)
      and isinstance(num_tokens, int)
      and start + num_tokens > num_indices
  ):
    raise ValueError(
        f"start + num_tokens ({start + num_tokens}) must not exceed indices"
        f" length ({num_indices})."
    )
  max_blocks, _, slots_len = _ragged_gather_reduce_tc_layout(
      num_indices, top_k, block_size, num_out_tokens=num_out_tokens
  )
  if num_out_tokens == 0:
    empty_slots = jnp.zeros((slots_len,), jnp.int32)
    return RaggedGatherReduceMetadata(
        src_rows=empty_slots,
        tokens=empty_slots,
        blocks=jnp.zeros((1,), jnp.int32),
        block_size=block_size,
        num_indices=num_indices,
    )

  start = jnp.asarray(start, dtype=jnp.int32)
  num_tokens = jnp.asarray(num_tokens, dtype=jnp.int32)
  positions = jnp.arange(num_indices, dtype=jnp.int32)
  rel = positions - start
  valid = (rel >= 0) & (rel < num_tokens)
  # Valid slots first, by token and then buffer order; invalid slots last.
  key = jnp.where(valid, indices, num_out_tokens).astype(jnp.int32)
  if packed_sort:
    sorted_key, sorted_positions = packed_stable_sort(key, num_out_tokens)
  else:
    sorted_key, sorted_positions = jax.lax.sort(
        (key, positions), num_keys=1, is_stable=True
    )
  pad = slots_len - num_indices
  src_rows = jnp.pad(sorted_positions - start, (0, pad))
  tokens = jnp.pad(sorted_key, (0, pad))

  # A token costs its valid rows plus its output row, so the tokens before
  # slot i's token cost run_start[i] + sorted_key[i], where run_start[i] is
  # the first slot of that token. This cost increases over the valid slots.
  is_run_start = jnp.concatenate(
      [jnp.ones((1,), bool), sorted_key[1:] != sorted_key[:-1]]
  )
  run_start = jax.lax.cummax(jnp.where(is_run_start, positions, 0))
  int_max = jnp.iinfo(jnp.int32).max
  slot_cost = jnp.where(
      sorted_key < num_out_tokens, run_start + sorted_key, int_max
  )
  # Block b starts at the first token whose tokens before it cost at least
  # threshold[b]. The slots of the tokens before it, slot_start[b], are the
  # slots with a smaller cost. Searches over all slots are slow on TPU, so
  # slot_start is counted in whole chunks of slots, then in the one chunk that
  # straddles the threshold.
  chunk = _RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK
  num_chunks = pl.cdiv(num_indices, chunk)
  cost_chunks = jnp.pad(
      slot_cost, (0, num_chunks * chunk - num_indices), constant_values=int_max
  ).reshape(num_chunks, chunk)
  threshold = jnp.arange(max_blocks + 1, dtype=jnp.int32) * (block_size - top_k)
  full_chunks = jnp.sum(
      cost_chunks[:, -1] < threshold[:, None], axis=1, dtype=jnp.int32
  )
  row = jnp.minimum(full_chunks, num_chunks - 1)
  in_chunk = jnp.sum(
      cost_chunks[row] < threshold[:, None], axis=1, dtype=jnp.int32
  )
  slot_start = row * chunk + in_chunk
  # The tokens between slot_start - 1's token and slot_start's token have no
  # valid slots and cost 1 each, so block b starts at the first of them whose
  # tokens before it reach the threshold, or else at slot_start's token.
  prev_key = jnp.where(
      slot_start > 0, sorted_key[jnp.maximum(slot_start - 1, 0)], -1
  )
  next_key = jnp.where(
      slot_start < num_indices,
      sorted_key[jnp.minimum(slot_start, num_indices - 1)],
      num_out_tokens,
  )
  token_start = jnp.clip(threshold - slot_start, prev_key + 1, next_key)
  table = jnp.stack(
      [
          slot_start[:-1],
          jnp.diff(slot_start),
          token_start[:-1],
          jnp.diff(token_start),
      ],
      axis=1,
  )
  num_blocks = jnp.sum(token_start[:-1] < num_out_tokens, dtype=jnp.int32)
  blocks = jnp.concatenate([num_blocks[None], table.reshape(-1)])
  return RaggedGatherReduceMetadata(
      src_rows=src_rows,
      tokens=tokens,
      blocks=blocks.astype(jnp.int32),
      block_size=block_size,
      num_indices=num_indices,
  )


def ragged_gather_reduce_tc(
    x: jax.Array,
    metadata: RaggedGatherReduceMetadata,
    *,
    num_out_tokens: int,
    top_k: int,
) -> jax.Array:
  """Transpose of `ragged_gather_tc`, reducing top-k duplicates by sum.

  `x` is a token buffer formed by `ragged_gather_tc(src, indices, start,
  num_tokens, ...)`, i.e. `x[i] = src[indices[start + i]]` for
  `i < num_tokens`. Each valid row is sent back to its source token, and rows
  that land on the same token are summed:
  `out[t] = sum_{i < num_tokens, indices[start + i] == t} x[i]`. Rows of `x` at
  or past `num_tokens` are never read. Accumulation is done in float32 in
  buffer order. An output token with no valid rows is zero.

  The routing is given by `metadata`, from
  `ragged_gather_reduce_tc_metadata(indices, start, num_tokens, top_k=top_k,
  ...)`. The caller must also guarantee `num_tokens <= max_input_tokens`; this
  is not checked at runtime and violating it is undefined behavior.

  Args:
    x: Ragged token buffer of shape (max_input_tokens, d0, d1).
    metadata: Routing from `ragged_gather_reduce_tc_metadata`.
    num_out_tokens: Number of output tokens the metadata was derived for.
    top_k: Number of occurrences of each token id in `indices`.

  Returns:
    Reduced tokens of shape (num_out_tokens, d0, d1).

  Raises:
    ValueError: If the metadata was not derived for `num_out_tokens * top_k`
      indices and `top_k`, or its block buffers do not fit in VMEM.
  """
  max_input_tokens, dim0, dim1 = x.shape
  block_size = metadata.block_size
  _check_ragged_gather_reduce_tc_args(top_k, block_size)
  if num_out_tokens < 0:
    raise ValueError(
        f"num_out_tokens must be non-negative, got {num_out_tokens}."
    )
  num_buffers = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  fields = _RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS
  max_blocks, window, slots_len = _ragged_gather_reduce_tc_layout(
      metadata.num_indices, top_k, block_size, num_out_tokens=num_out_tokens
  )
  expected_shapes = ((slots_len,), (slots_len,), (1 + fields * max_blocks,))
  shapes = (
      metadata.src_rows.shape,
      metadata.tokens.shape,
      metadata.blocks.shape,
  )
  if shapes != expected_shapes:
    raise ValueError(
        "metadata (src_rows, tokens, blocks) must have shapes"
        f" {expected_shapes}, got {shapes}; was it derived for"
        f" num_out_tokens={num_out_tokens} with top_k={top_k}?"
    )
  # A (d0, d1) token is padded to whole (sublanes, 128) tiles in VMEM.
  itemsize = jnp.dtype(x.dtype).itemsize
  sublanes = 8 * max(1, 4 // itemsize)
  token_vmem_bytes = (
      pl.cdiv(dim0, sublanes) * sublanes * pl.cdiv(dim1, 128) * 128 * itemsize
  )
  vmem_budget = _RAGGED_GATHER_REDUCE_TC_VMEM_BUDGET
  if num_buffers * block_size * token_vmem_bytes > vmem_budget:
    raise ValueError(
        f"block_size {block_size} does not fit {num_buffers} VMEM buffers of"
        f" ({dim0}, {dim1}) {x.dtype} tokens in {vmem_budget} bytes; use"
        f" block_size <= {vmem_budget // (num_buffers * token_vmem_bytes)}."
    )
  if num_out_tokens == 0 or max_input_tokens == 0:
    return jnp.zeros((num_out_tokens, dim0, dim1), x.dtype)

  lookahead = _RAGGED_GATHER_REDUCE_TC_LOOKAHEAD
  # A block's slot window is loaded two steps before its rows are issued and
  # is live through its reduce, lookahead steps later, so windows cycle
  # through a ring of lookahead + 2 SMEM slots.
  ring = lookahead + 2

  def _ragged_gather_reduce_kernel(
      blocks_smem_ref,
      src_hbm_ref,
      tok_hbm_ref,
      x_hbm_ref,
      o_hbm_ref,
      vmem_ref,
      src_smem_ref,
      tok_smem_ref,
      recv_sem,
      send_sem,
      slots_sem,
  ):
    num_blocks = blocks_smem_ref[0]

    # Block blk reduces compacted slots [slot_start, slot_start + num_slots)
    # into output tokens [token_start, token_start + num_tokens). In its VMEM
    # buffer, rows [0, num_tokens) hold the output tokens and rows
    # [num_tokens, num_tokens + num_slots) the gathered input rows.
    def _block(blk):
      entry = 1 + fields * blk
      return tuple(blocks_smem_ref[entry + f] for f in range(fields))

    def _window_start(slot_start):
      return jnp.minimum((slot_start // align) * align, slots_len - window)

    def _slots_loads(blk):
      r = blk % ring
      window_start = pl.multiple_of(_window_start(_block(blk)[0]), align)
      return [
          pltpu.make_async_copy(
              hbm_ref.at[pl.ds(window_start, window)],
              smem_ref.at[pl.ds(r * window, window)],
              slots_sem.at[r],
          )
          for hbm_ref, smem_ref in (
              (src_hbm_ref, src_smem_ref),
              (tok_hbm_ref, tok_smem_ref),
          )
      ]

    def _slot_base(blk):
      """Returns the SMEM offset of block blk's first slot."""
      slot_start = _block(blk)[0]
      return (blk % ring) * window + slot_start - _window_start(slot_start)

    def _pow2_chunks(n, fn):
      """Calls fn(off, size) on static power-of-two sizes summing to n."""
      off = 0
      for k in reversed(range(block_size.bit_length())):
        size = 1 << k
        bit = n & size

        def _fn_if_bit_set(off=off, size=size):
          fn(off, size)

        pl.when(bit != 0)(_fn_if_bit_set)

        off = off + bit

    def _wait_rows(b, n, sem):
      def _wait(off, size):
        del off
        rows = vmem_ref.at[b, pl.ds(0, size), :, :]
        pltpu.make_async_copy(rows, rows, sem.at[b]).wait()

      _pow2_chunks(n, _wait)

    def _dynamic_fori(lo, hi, unroll, body, carry):
      """fori_loop over [lo, hi) with `unroll` static bodies per trip."""

      def _chunk(c, carry):
        for u in range(unroll):
          carry = body(lo + c * unroll + u, carry)
        return carry

      num_full = (hi - lo) // unroll
      carry = jax.lax.fori_loop(0, num_full, _chunk, carry)
      return jax.lax.fori_loop(lo + num_full * unroll, hi, body, carry)

    def _issuer(blk):
      """Returns a loop body that starts block blk's i-th row DMA."""
      b = blk % num_buffers
      num_tokens = _block(blk)[3]
      base = _slot_base(blk)

      def _issue(i, carry):
        pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(src_smem_ref[base + i], 1), :, :],
            vmem_ref.at[b, pl.ds(num_tokens + i, 1), :, :],
            recv_sem.at[b],
        ).start()
        return carry

      return _issue

    def _in_start(blk):
      _dynamic_fori(
          0,
          _block(blk)[1],
          _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL,
          _issuer(blk),
          (),
      )

    def _in_wait(blk):
      _wait_rows(blk % num_buffers, _block(blk)[1], recv_sem)

    def _reducer(blk):
      """Zeroes block blk's tokens and returns its row reduce loop body."""
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)

      # Tokens without valid rows are zero.
      @pl.loop(0, num_tokens)
      def _zero_token(t):
        vmem_ref[b, t, :, :] = jnp.zeros((dim0, dim1), x.dtype)

      # A token's valid rows are contiguous in the compacted order, so sum
      # them in a running float32 total that restarts when the token changes.
      # Every row stores the running total; the token's last row leaves the
      # final sum.
      base = _slot_base(blk)

      def _reduce(i, carry):
        prev_t, total = carry
        t = tok_smem_ref[base + i] - token_start
        row = vmem_ref[b, num_tokens + i, :, :].astype(jnp.float32)
        total = jnp.where(t == prev_t, total, 0.0) + row
        vmem_ref[b, t, :, :] = total.astype(x.dtype)
        return t, total

      return _reduce

    def _out_start(blk):
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)

      def _store(off, size):
        pltpu.make_async_copy(
            vmem_ref.at[b, pl.ds(off, size), :, :],
            o_hbm_ref.at[pl.ds(token_start + off, size), :, :],
            send_sem.at[b],
        ).start()

      _pow2_chunks(num_tokens, _store)

    def _out_wait(blk):
      _wait_rows(blk % num_buffers, _block(blk)[3], send_sem)

    def _slots_wait(blk):
      for load in _slots_loads(blk):
        load.wait()

    for blk in range(ring):

      @pl.when(blk < num_blocks)
      def _load_first_slots(blk=blk):
        for load in _slots_loads(blk):
          load.start()

    for blk in range(lookahead):

      @pl.when(blk < num_blocks)
      def _start_first_rows(blk=blk):
        _slots_wait(blk)
        _in_start(blk)

    @pl.loop(0, num_blocks)
    def _step(i):
      _in_wait(i)
      nxt = i + lookahead
      has_nxt = nxt < num_blocks

      # Block nxt reuses the buffer of block nxt - num_buffers, whose store
      # must be done.
      @pl.when(nxt >= num_buffers)
      def _free_buffer():
        _out_wait(nxt - num_buffers)

      @pl.when(has_nxt)
      def _wait_next_slots():
        _slots_wait(nxt)

      # Block nxt's row DMAs are started in block i's reduce loop, so the
      # scalar DMA issue shares bundles with the vector reduce.
      nxt = jnp.minimum(nxt, num_blocks - 1)
      num_reduce = _block(i)[1]
      num_issue = jnp.where(has_nxt, _block(nxt)[1], 0)
      num_both = jnp.minimum(num_reduce, num_issue)
      issue = _issuer(nxt)
      reduce = _reducer(i)

      def _issue_and_reduce(j, carry):
        issue(j, ())
        return reduce(j, carry)

      unroll = _RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL
      carry = (jnp.int32(-1), jnp.zeros((dim0, dim1), jnp.float32))
      carry = _dynamic_fori(0, num_both, unroll, _issue_and_reduce, carry)
      _dynamic_fori(num_both, num_reduce, unroll, reduce, carry)
      _dynamic_fori(
          num_both,
          num_issue,
          _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL,
          issue,
          (),
      )
      _out_start(i)

      # Block i's slot window is free now that its reduce is done.
      @pl.when(i + ring < num_blocks)
      def _load_later_slots():
        for load in _slots_loads(i + ring):
          load.start()

    # The last num_buffers - lookahead stores are still pending.
    for k in range(num_buffers - lookahead):

      @pl.when(num_blocks - 1 - k >= 0)
      def _wait_last_stores(k=k):
        _out_wait(num_blocks - 1 - k)

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_shape = jax.ShapeDtypeStruct(
      (num_out_tokens, dim0, dim1), x.dtype, manual_axis_type=manual_axis_type
  )
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  dma_sems = pltpu.SemaphoreType.DMA((num_buffers,))
  return pl.pallas_call(
      _ragged_gather_reduce_kernel,
      out_shape=out_shape,
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          hbm_spec,
          hbm_spec,
          hbm_spec,
      ],
      out_specs=hbm_spec,
      scratch_shapes=[
          pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
          pltpu.SMEM((ring * window,), jnp.int32),
          pltpu.SMEM((ring * window,), jnp.int32),
          dma_sems,
          dma_sems,
          pltpu.SemaphoreType.DMA((ring,)),
      ],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT,
      ),
      name="ragged_gather_reduce_tc",
  )(metadata.blocks, metadata.src_rows, metadata.tokens, x)
