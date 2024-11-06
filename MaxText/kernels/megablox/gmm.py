# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grouped matrix multiplication kernels for TPU written in Pallas."""

import functools
from typing import Any, Callable, Optional, Union

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu.megablox import common
import jax.numpy as jnp
import numpy as np

partial = functools.partial


def _validate_args(
    *,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    expected_rhs_dims: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.dtype]:
  """Validates the arguments for the gmm function."""
  # Validate 'lhs'.
  if lhs.ndim != 2:
    raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")
  common.assert_is_supported_dtype(lhs.dtype)

  # Validate 'rhs'.
  if rhs.ndim != expected_rhs_dims:
    raise ValueError(
        f"Expected {expected_rhs_dims}-tensor for 'rhs' but got"
        f" {rhs.ndim}-tensor."
    )
  common.assert_is_supported_dtype(rhs.dtype)

  # Validate 'group_sizes'.
  if group_sizes.dtype != jnp.int32:
    raise ValueError(
        f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}."
    )

  return lhs, group_sizes, common.select_input_dtype(lhs, rhs)


def _calculate_num_tiles(x: int, tx: int) -> int:
  tiles, rem = divmod(x, tx)
  if rem:
    raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
  return tiles


def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
  tiles, rem = divmod(x, tx)
  if rem:
    tiles += 1
  return tiles, rem


GroupMetadata = Any  # TODO(enriqueps): Clean this up and use a namedtuple


def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata:
  """Create the metadata needed for grouped matmul computation.

  Args:
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    m: The number of rows in lhs.
    tm: The m-dimension tile size being used.
    start_group: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_nonzero_groups: Number of groups in group sizes to compute on. Useful in
      combination with group_offset.
    visit_empty_groups: If True, do not squeeze tiles for empty groups out of
      the metadata. This is necessary for tgmm, where we at least need to zero
      the output for each group.

  Returns:
    tuple of:
      group_offsets: A 1d, jnp.ndarray with shape [num_groups+1] and jnp.int32
        dtype. group_offsets[i] indicates the row at which group [i] starts in
        the lhs matrix and group_offsets[i-1] = m.
      group_ids: A 1d, jnp.ndarray with shape [m_tiles + num_groups] and
        jnp.int32 dtype. group_ids[i] indicates which group grid index 'i' will
        work on.
      m_tile_ids: A 1d, jnp.ndarray with shape [m_tiles + num_groups] and
        jnp.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
        will work on.
    num_tiles: The number of m-dimension tiles to execute.
  """

  num_groups = group_sizes.shape[0]
  np_start_group = np.zeros(start_group.shape, dtype=start_group.dtype)
  end_group = np_start_group + num_nonzero_groups - 1
  # Calculate the offset of each group, starting at zero. This metadata is
  # similar to row offsets in a CSR matrix. The following properties hold:
  #
  # group_offsets.shape = [num_groups + 1]
  # group_offsets[0] = 0
  # group_offsets[num_groups] = m
  #
  # The row at which group 'i' starts is group_offsets[i].
  group_ends = jnp.cumsum(group_sizes)
  group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])
  # Assign a group id to each grid index.
  #
  # If a group starts somewhere other than the start of a tile or ends somewhere
  # other than the end of a tile we need to compute that full tile. Calculate
  # the number of tiles for each group by rounding their end up to the nearest
  # 'tm' and their start down to the nearest 'tm'.

  # (1) Round the group_ends up to the nearest multiple of 'tm'.
  #
  # NOTE: This does not change group_offsets[num_groups], which is m
  # (because we enforce m is divisible by tm).
  rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
  # (2) Round the group_starts down to the nearest multiple of 'tm'.
  group_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
  )
  rounded_group_starts = group_starts // tm * tm

  # (3) Calculate the number of rows in each group.
  #
  # NOTE: Handle zero-sized groups as a special case. If the start for a
  # zero-sized group is not divisible by 'tm' its start will be rounded down and
  # its end will be rounded up such that its size will become 1 tile here.
  rounded_group_sizes = rounded_group_ends - rounded_group_starts
  rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)
  # (4) Convert the group sizes from units of rows to unit of 'tm' sized tiles.
  #
  # An m-dimension tile is 'owned' by group 'i' if the first row of the tile
  # belongs to group 'i'. In addition to owned tiles, each group can have 0 or 1
  # initial partial tiles if it's first row does not occur in the first row of a
  # tile. The '0-th' group never has a partial tile because it always starts at
  # the 0-th row.
  #
  # If no group has a partial tile, the total number of tiles is equal to
  # 'm // tm'. If every group has a partial except the 0-th group, the total
  # number of tiles is equal to 'm // tm + num_groups - 1'. Thus we know that
  #
  # tiles_m <= group_tiles.sum() <= tiles_m + num_groups - 1
  #
  # Where tiles_m = m // tm.
  #
  # NOTE: All group sizes are divisible by 'tm' because of the rounding in steps
  # (1) and (2) so this division is exact.
  group_tiles = rounded_group_sizes // tm
  if visit_empty_groups:
    # Insert one tile for empty groups.
    group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

  # Create the group ids for each grid index based on the tile counts for each
  # group.
  #
  # NOTE: This repeat(...) will pad group_ids with the final group id if
  # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
  # such that we only execute the necessary number of tiles.
  tiles_m = _calculate_num_tiles(m, tm)
  group_ids = jnp.repeat(
      jnp.arange(num_groups, dtype=jnp.int32),
      group_tiles,
      total_repeat_length=tiles_m + num_groups - 1,
  )
  # Assign an m-dimension tile id to each grid index.
  #
  # NOTE: Output tiles can only be re-visited consecutively. The following
  # procedure guarantees that m-dimension tile indices respect this.

  # (1) Calculate how many times each m-dimension tile will be visited.
  #
  # Each tile is guaranteed to be visited once by the group that owns the tile.
  # The remaining possible visits occur when a group starts inside of a tile at
  # a position other than the first row. We can calculate which m-dimension tile
  # each group starts in by floor-dividing its offset with `tm` and then count
  # tile visits with a histogram.
  #
  # To avoid double counting tile visits from the group that owns the tile,
  # filter these out by assigning their tile id to `tile_m` (one beyond the max)
  # such that they're ignored by the subsequent histogram. Also filter out any
  # group which is empty.
  #
  # TODO(tgale): Invert the 'partial_tile_mask' predicates to be more clear.
  partial_tile_mask = jnp.logical_or(
      (group_offsets[:-1] % tm) == 0, group_sizes == 0
  )

  # Explicitly enable tiles for zero sized groups, if specified. This covers
  # zero sized groups that start on a tile-aligned row and those that do not.
  if visit_empty_groups:
    partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

  partial_tile_ids = jnp.where(
      partial_tile_mask, tiles_m, group_offsets[:-1] // tm
  )

  tile_visits = (
      jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0]
      + 1
  )

  # Create the m-dimension tile ids for each grid index based on the visit
  # counts for each tile.
  m_tile_ids = jnp.repeat(
      jnp.arange(tiles_m, dtype=jnp.int32),
      tile_visits.astype(jnp.int32),
      total_repeat_length=tiles_m + num_groups - 1,
  )
  # Account for sharding.
  #
  # Find the start of the groups owned by our shard and shift the group_ids and
  # m_tile_ids s.t. the metadata for our tiles are at the front of the arrays.
  #
  # TODO(tgale): Move this offset into the kernel to avoid these rolls.
  first_tile_in_shard = (group_ids < start_group).sum()
  group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
  m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)


  # Calculate the number of tiles we need to compute for our shard.
  #
  # Remove tile visits that belong to a group not in our shard.
  iota = jnp.arange(num_groups, dtype=jnp.int32)
  active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
  group_tiles = jnp.where(active_group_mask, group_tiles, 0)
  num_tiles = group_tiles.sum()
  return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_group_size(
    *, grid_id: jnp.ndarray, group_metadata: GroupMetadata
) -> jnp.ndarray:
  """Calculate the number of rows in the current group."""
  group_offsets, group_ids = group_metadata[:2]
  group_id = group_ids[grid_id]
  group_start = group_offsets[group_id]
  group_end = group_offsets[group_id + 1]
  return group_end - group_start


def _get_store_mask(
    *,
    grid_id: jnp.ndarray,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jnp.ndarray:
  """Mask for rows that belong to the current group in the current tile."""
  group_offsets, group_ids, m_tile_ids = group_metadata[:3]
  group_id = group_ids[grid_id]
  group_start = group_offsets[group_id]
  group_end = group_offsets[group_id + 1]
  m_id = m_tile_ids[grid_id] * tm
  iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
  return jnp.logical_and(iota >= group_start, iota < group_end)


def _zero_uninitialized_memory(
    out: jnp.ndarray,
    *,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> jnp.ndarray:
  """Zero out uninitialized memory from output."""
  group_offsets = group_metadata[0]
  group_start = group_offsets[start_group]
  group_end = group_offsets[start_group + num_nonzero_groups]
  valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
  valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
  return jnp.where(valid_mask[:, None], out, 0)


LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]



def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: Optional[Union[tuple[int, int, int], LutFn]] = (128, 128, 128),
    group_offset: Optional[jnp.ndarray] = None,
    existing_out: Optional[jnp.ndarray] = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray:
  
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if group_offset.shape:
      raise ValueError(
          f"group_offset must be a ()-shaped array. Got: {group_offset.shape}."
      )
    group_offset = group_offset[None]

  group_metadata, num_active_tiles = make_group_metadata(  # pylint: disable=unbalanced-tuple-unpacking
      group_sizes=group_sizes,
      m=lhs.shape[0],
      tm=tiling[0],
      start_group=group_offset[0],
      num_nonzero_groups=rhs.shape[0],
      visit_empty_groups=False,
  )
  return _gmm(
    lhs=lhs, 
    rhs=rhs,
    group_metadata=group_metadata,
    num_total_groups=group_sizes.shape[0],
    num_active_tiles=num_active_tiles.item(),
    preferred_element_type=preferred_element_type,
    tiling=tiling,
    group_offset=group_offset,
    existing_out=existing_out,
    transpose_rhs=transpose_rhs,
    interpret=interpret,
  )

@functools.partial(
    jax.jit,
    static_argnames=[
        "num_active_tiles",
        "num_total_groups",
        "preferred_element_type",
        "tiling",
        "transpose_rhs",
        "interpret",
    ],
)
def _gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_metadata: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    num_total_groups: int,
    num_active_tiles: int,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: Optional[Union[tuple[int, int, int], LutFn]] = (128, 128, 128),
    group_offset: Optional[jnp.ndarray] = None,
    existing_out: Optional[jnp.ndarray] = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jnp.ndarray with shape [m, k].
    rhs: A 3d, jnp.ndarray with shape [num_groups, k, n].
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    existing_out: Existing output to write to.
    transpose_rhs: True if the rhs needs to be transposed.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.

  Returns:
    A 2d, jnp.ndarray with shape [m, n].
  """
  preferred_element_type = jnp.bfloat16

  if existing_out is not None:
    assert isinstance(existing_out, jax.Array)
    expected_dtype = existing_out.dtype
    if expected_dtype != preferred_element_type:
      raise ValueError(
          "Existing output dtype must match preferred_element_type."
      )
  num_current_groups = rhs.shape[0]
  input_dtype = common.select_input_dtype(lhs, rhs)


  # Gather shape information.
  m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
  if transpose_rhs:
    n = rhs.shape[1]

  # If tiling is callable, look up the problem dimensions in the LUT. If no tuned
  # tile dimensions are available throw an error.
  if callable(tiling):
    tiling = tiling(m, k, n)

  if tiling is None:
    raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

  tm, tk, tn = tiling
  tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
  tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
  del n_rem

  def kernel(
      lhs,
      rhs,
      existing_out,
      group_metadata_ref,
      group_ids_ref,
      m_tile_ids_ref,
      group_offset_ref,
      acc_ref,
      out,
  ):

    n_i = pl.program_id(0)
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    m_i = m_tile_ids_ref[grid_id]
    rhs_group_id = group_ids_ref[grid_id] - group_offset_ref[0]
    rhs_k_i, rhs_n_i, rhs_tk, rhs_tn = k_i, n_i, tk, tn
    if transpose_rhs:
      rhs_k_i, rhs_n_i, rhs_tk, rhs_tn = n_i, k_i, tn, tk


    @pl.when(k_i == 0)
    def _zero_acc():
      acc_ref[...] = jnp.zeros_like(acc_ref)

      if existing_out is not None:
        prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
        is_first_processed_group = grid_id == 0
        m_tile_changed = m_i != m_tile_ids_ref[prev_grid_id]
        first_time_seeing_out = jnp.logical_or(
            is_first_processed_group, m_tile_changed
        )

        @pl.when(first_time_seeing_out)
        def _init_out():
          out[m_i:m_i+tm, n_i:n_i+tn] = existing_out[m_i:m_i+tm, n_i:n_i+tn]

    def mask_k_rem(x, *, dim):
      if k_rem == 0:
        return x

      orig_dtype = x.dtype
      iota = lax.broadcasted_iota(jnp.int32, x.shape, dim)
      x = x.astype(jnp.float32)
      return jnp.where(iota < k_rem, x, 0).astype(orig_dtype)

    def _store_accum():
      mask = _get_store_mask(
          grid_id=grid_id,
          group_metadata=group_metadata_ref,
          tm=tm,
          tn=tn,
      )
      to_store = acc_ref[...]
      pl.store(
        out, 
        (pl.dslice(m_i, tm), pl.dslice(n_i, tn)), 
        to_store.astype(preferred_element_type), 
        mask=mask,
      )


    def _accum(is_last_k_tile):
      if is_last_k_tile:
        mask_k_rem_lhs = partial(mask_k_rem, dim=1)
        mask_k_rem_rhs = partial(mask_k_rem, dim=int(transpose_rhs))
      else:
        mask_k_rem_lhs = lambda x: x
        mask_k_rem_rhs = lambda x: x

      if transpose_rhs:
        dot_general_dims = (((1,), (1,)), ((), ()))
      else:
        dot_general_dims = (((1,), (0,)), ((), ()))

      loaded_lhs = pl.load(lhs, (pl.ds(m_i, tm), pl.ds(k_i, tk)))
      loaded_rhs = pl.load(rhs, (rhs_group_id, pl.ds(rhs_k_i, rhs_tk), pl.ds(rhs_n_i, rhs_tn)))
      acc_ref[...] += lax.dot_general(
          mask_k_rem_lhs(loaded_lhs).astype(input_dtype),
          mask_k_rem_rhs(loaded_rhs).astype(input_dtype),
          preferred_element_type=jnp.float32,
          dimension_numbers=dot_general_dims,
      )

      if is_last_k_tile:
        _store_accum()

    lax.cond(
      k_i == pl.num_programs(2) - 1,
      partial(_accum, True),
      partial(_accum, False),
    )

  if existing_out is None:
    input_output_aliases = {}
  else:
    input_output_aliases = {6: 0}

  call_gmm = pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
      grid=(tiles_n, num_active_tiles, tiles_k),
      input_output_aliases=input_output_aliases,
      interpret=interpret,
      debug=True,
  )
  acc = jnp.zeros((tm, tn), dtype=jnp.float32)
  out = call_gmm(
      lhs,
      rhs,
      existing_out,
      group_metadata,
      group_metadata[1],
      group_metadata[2],
      group_offset,
      acc,
  )
  if existing_out is None and num_current_groups < num_total_groups:
    out = _zero_uninitialized_memory(
        out,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        group_metadata=group_metadata,
    )
  return out
