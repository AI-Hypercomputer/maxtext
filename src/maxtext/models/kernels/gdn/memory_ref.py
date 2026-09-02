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
# ==============================================================================

"""Weight and state reference dataclasses for VMEM."""

import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

try:
  from maxtext.models.kernels.gdn import config
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn import config
  except (ImportError, ModuleNotFoundError):
    from . import config


def _flat_pos(shape: tuple[int, ...], indices: tuple[Any, ...]) -> Any:
  """Row-major flat offset of `indices` into a logical array of `shape`."""
  strides = pl.strides_from_shape(shape)
  assert len(strides) == len(indices)

  pos = 0
  for stride, idx in zip(strides, indices):
    pos += stride * idx
  return pos


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvWeightsRef:
  weight: Any
  bias: Any | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNWeightsRef:
  a_log: Any
  dt_bias: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightRefs:
  conv: ConvWeightsRef
  gdn: GDNWeightsRef


class FieldOffset:
  """Descriptor returning the record field at ``data[pos + offset]``.

  Reads a single dynamically-indexed element rather than a slice, since JAX
  can't slice a range with traced indices. Read-only: metadata is never
  written.
  """

  def __init__(self, offset: int):
    self.offset = offset

  def __get__(self, obj, objtype=None):
    if obj is None:
      return self
    return obj.data[obj.pos + self.offset]


# Per-p_id metadata is an array of structs: each p_id's fields sit contiguously
# and FieldOffset(k) reads the k-th word of its struct.
#
# Packed struct: [r_base, packed_word].
# Fields share packed_word to save SMEM: is_first_tile(0), is_last_tile(1), r_size(2..15), s_idx(16..31).
@dataclasses.dataclass(frozen=True)
class PackedPIdRecord:
  """Packed struct [r_base, packed_word]; the four small fields bit-slice word.

  Each bit field masks after shifting, which also clears the sign bits that
  ``>>`` extends on the signed int32 word.
  """

  STRUCT_SIZE = 2
  FIRST_TILE_SHIFT = 0
  LAST_TILE_SHIFT = 1
  R_SIZE_SHIFT = 2
  S_IDX_SHIFT = 16
  FLAG_MASK = 1
  R_SIZE_MASK = (1 << (S_IDX_SHIFT - R_SIZE_SHIFT)) - 1
  S_IDX_MASK = (1 << (32 - S_IDX_SHIFT)) - 1
  MAX_SEQS = S_IDX_MASK + 1

  data: Any
  pos: Any
  r_base = FieldOffset(0)
  word = FieldOffset(1)

  @property
  def s_idx(self):
    return (self.word >> self.S_IDX_SHIFT) & self.S_IDX_MASK

  @property
  def r_size(self):
    return (self.word >> self.R_SIZE_SHIFT) & self.R_SIZE_MASK

  @property
  def is_first_tile(self):
    return (self.word & self.FLAG_MASK) != 0

  @property
  def is_last_tile(self):
    return ((self.word >> self.LAST_TILE_SHIFT) & self.FLAG_MASK) != 0

  @classmethod
  def pack(
      cls,
      s_idx: jax.Array,
      r_size: jax.Array,
      is_first_tile: jax.Array,
      is_last_tile: jax.Array,
  ) -> jax.Array:
    """Packs s_idx, row size and two tile-state flags into one int32 word."""

    s_idx = s_idx.reshape(-1).astype(jnp.int32)
    r_size = r_size.reshape(-1).astype(jnp.int32)
    is_first_tile = is_first_tile.reshape(-1).astype(jnp.int32)
    is_last_tile = is_last_tile.reshape(-1).astype(jnp.int32)
    word = s_idx << cls.S_IDX_SHIFT
    word |= r_size << cls.R_SIZE_SHIFT
    word |= is_last_tile << cls.LAST_TILE_SHIFT
    word |= is_first_tile << cls.FIRST_TILE_SHIFT
    return word


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  num_tiles: Any
  # Array of structs holding every p_id's metadata
  records: Any
  s_idx_has_initial_state: Any
  s_idx_to_state_indices: Any
  shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  def get_record(self, p_id, idx) -> PackedPIdRecord:
    """View of one p_id's metadata: .r_base / .s_idx / .r_size / .is_*_tile."""
    record_idx = _flat_pos(self.shape, (p_id, idx))
    return PackedPIdRecord(
        self.records, record_idx * PackedPIdRecord.STRUCT_SIZE
    )

  @classmethod
  def create(  # pyrefly: ignore[bad-override]
      cls,
      cfgs: config.GDNConfig,
      num_tiles: jax.Array,
      p_id_to_s_idx: jax.Array,
      p_id_to_r_base: jax.Array,
      p_id_to_r_size: jax.Array,
      p_id_is_first_tile: jax.Array,
      p_id_is_last_tile: jax.Array,
      s_idx_has_initial_state: jax.Array,
      s_idx_to_state_indices: jax.Array,
  ):
    # NOTE: First dim does not matter when it comes to calculating stride.
    shape = (1, cfgs.seq_tile_size)
    assert s_idx_has_initial_state.shape[0] <= PackedPIdRecord.MAX_SEQS, (
        f"Number of sequences ({s_idx_has_initial_state.shape[0]}) exceeds"
        f" PackedPIdRecord limit ({PackedPIdRecord.MAX_SEQS})."
    )
    assert cfgs.tile_size <= PackedPIdRecord.R_SIZE_MASK, (
        f"Tile size ({cfgs.tile_size}) exceeds PackedPIdRecord limit"
        f" ({PackedPIdRecord.R_SIZE_MASK})."
    )

    r_base = p_id_to_r_base.reshape(-1).astype(jnp.int32)
    word = PackedPIdRecord.pack(
        p_id_to_s_idx, p_id_to_r_size, p_id_is_first_tile, p_id_is_last_tile
    )
    fields = [r_base, word]
    # Interleave fields into one array of structs: [rec0_f0, rec0_f1, ...].
    records = jnp.stack(fields, axis=-1).reshape(-1)

    return cls(
        num_tiles=num_tiles,
        records=records,
        s_idx_has_initial_state=s_idx_has_initial_state,
        s_idx_to_state_indices=s_idx_to_state_indices,
        shape=shape,
    )

  def __len__(self) -> int:
    return len(jax.tree_util.tree_leaves(self))


@dataclasses.dataclass(frozen=True, kw_only=True)
class BaseBufferedRef(pltpu.BufferedRef):

  cfg: config.GDNConfig = dataclasses.field(metadata=dict(static=True))
  # NOTE: Despite being ref, metadata_ref should be set to static. This is
  # because the memory will be allocated outside of kernel and metadata_ref
  # merely points to the reference.
  metadata_ref: MetadataRef = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pyrefly: ignore[bad-override]
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: jax.Array,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfg: config.GDNConfig,
      metadata_ref: MetadataRef,
  ):
    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
    )
    return cls(
        cfg=cfg,
        metadata_ref=metadata_ref,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class InBufferedRef(BaseBufferedRef):

  def copy_in(self, src_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      record = self.metadata_ref.get_record(p_id, idx)
      r_base = record.r_base
      dma_size = record.r_size
      pltpu.make_async_copy(
          src_ref.at[pl.ds(r_base, dma_size)],
          vmem_ref.at[idx, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
          sem,
      ).start()

  def wait_in(self, src_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      dma_size += self.metadata_ref.get_record(p_id, idx).r_size

    pltpu.make_async_copy(
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutBufferedRef(BaseBufferedRef):

  def copy_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      record = self.metadata_ref.get_record(p_id, idx)
      r_base = record.r_base
      dma_size = record.r_size
      pltpu.make_async_copy(
          vmem_ref.at[idx, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
          dst_ref.at[pl.ds(r_base, dma_size)],
          sem,
      ).start()

  def wait_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      dma_size += self.metadata_ref.get_record(p_id, idx).r_size

    pltpu.make_async_copy(
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TInvBufferedRef(BaseBufferedRef):
  """DMA buffer for triangular inverse matrix caching (t_inv)."""

  def copy_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      pltpu.make_async_copy(
          vmem_ref.at[idx],
          dst_ref.at[p_id + idx],
          sem,
      ).start()

  def wait_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]

    for idx in range(self.cfg.seq_tile_size):
      pltpu.make_async_copy(
          vmem_ref.at[idx],
          vmem_ref.at[idx],
          sem,
      ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ChunkStatesBufferedRef(BaseBufferedRef):
  """DMA buffer for caching intermediate recurrent chunk states."""

  def copy_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      pltpu.make_async_copy(
          vmem_ref.at[idx],
          dst_ref.at[p_id + idx],
          sem,
      ).start()

  def wait_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]

    for idx in range(self.cfg.seq_tile_size):
      pltpu.make_async_copy(
          vmem_ref.at[idx],
          vmem_ref.at[idx],
          sem,
      ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class StateBufferedRef(BaseBufferedRef):

  def copy_in(self, src_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      record = self.metadata_ref.get_record(p_id, idx)
      is_first_tile = record.is_first_tile
      s_idx = record.s_idx
      state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]
      has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
      should_read = jnp.logical_and(is_first_tile, has_initial_state)
      dma_size = jnp.where(should_read, 1, 0)

      pltpu.make_async_copy(
          src_ref.at[pl.ds(state_idx, dma_size)],
          vmem_ref.at[pl.ds(idx, dma_size)],  # pyrefly: ignore[missing-attribute]
          sem,
      ).start()

  def wait_in(self, src_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      record = self.metadata_ref.get_record(p_id, idx)
      is_first_tile = record.is_first_tile
      s_idx = record.s_idx
      has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
      should_read = jnp.logical_and(is_first_tile, has_initial_state)
      dma_size += jnp.where(should_read, 1, 0)

    pltpu.make_async_copy(
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()

  def copy_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      record = self.metadata_ref.get_record(p_id, idx)
      is_last_tile = record.is_last_tile
      s_idx = record.s_idx
      state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]
      dma_size = jnp.where(is_last_tile, 1, 0)

      pltpu.make_async_copy(
          vmem_ref.at[pl.ds(idx, dma_size)],  # pyrefly: ignore[missing-attribute]
          dst_ref.at[pl.ds(state_idx, dma_size)],
          sem,
      ).start()

  def wait_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      is_last_tile = self.metadata_ref.get_record(p_id, idx).is_last_tile
      dma_size += jnp.where(is_last_tile, 1, 0)

    pltpu.make_async_copy(
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()


def create_allocs(
    metadata_ref: MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    out_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    cfg: config.GDNConfig,
    t_inv_ref: jax.Array | None = None,
    chunk_states_ref: jax.Array | None = None,
) -> tuple[Any, ...]:
  qkv_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.dim_size)
  ba_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.aligned_num_v_heads)

  out_shape = (
      cfg.seq_tile_size,
      cfg.chunk_size,
      cfg.num_v_heads,
      cfg.v_head_dim,
  )
  conv_shape = (cfg.seq_tile_size, cfg.prev_kernel_size, 1, cfg.dim_size)
  recurrent_shape = (
      cfg.seq_tile_size,
      cfg.num_v_heads,
      cfg.kq_head_dim,
      cfg.v_head_dim,
  )

  pipeline_mode = pl.Buffered(buffer_count=cfg.num_buffers, use_lookahead=False)

  block_spec_partial = functools.partial(
      pl.BlockSpec,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pipeline_mode,
  )

  qkv_spec = block_spec_partial(block_shape=qkv_shape)
  ba_spec = block_spec_partial(block_shape=ba_shape)
  in_buffered_partial = functools.partial(
      InBufferedRef.input,
      buffer_count=pipeline_mode.buffer_count,
      use_lookahead=pipeline_mode.use_lookahead,
      cfg=cfg,
      metadata_ref=metadata_ref,
  )
  qkv_alloc = in_buffered_partial(spec=qkv_spec, dtype_or_type=qkv_ref)
  b_alloc = in_buffered_partial(spec=ba_spec, dtype_or_type=b_ref)
  a_alloc = in_buffered_partial(spec=ba_spec, dtype_or_type=a_ref)

  out_alloc = OutBufferedRef.output(
      spec=block_spec_partial(block_shape=out_shape),
      dtype_or_type=out_ref,
      buffer_count=pipeline_mode.buffer_count,
      use_lookahead=pipeline_mode.use_lookahead,
      cfg=cfg,
      metadata_ref=metadata_ref,
  )

  conv_spec = block_spec_partial(block_shape=conv_shape)
  recurrent_spec = block_spec_partial(block_shape=recurrent_shape)
  state_buffered_partial = functools.partial(
      StateBufferedRef.input_output,
      buffer_count=pipeline_mode.buffer_count,
      use_lookahead=pipeline_mode.use_lookahead,
      cfg=cfg,
      metadata_ref=metadata_ref,
  )
  conv_alloc = state_buffered_partial(
      spec=conv_spec, dtype_or_type=conv_state_ref
  )
  recurrent_alloc = state_buffered_partial(
      spec=recurrent_spec, dtype_or_type=recurrent_state_ref
  )

  allocs = [
      qkv_alloc,
      b_alloc,
      a_alloc,
      conv_alloc,
      recurrent_alloc,
      out_alloc,
  ]

  if t_inv_ref is not None:
    t_inv_shape = (
        cfg.seq_tile_size,
        cfg.num_v_heads,
        cfg.chunk_size,
        cfg.chunk_size,
    )
    t_inv_alloc = TInvBufferedRef.output(
        spec=block_spec_partial(block_shape=t_inv_shape),
        dtype_or_type=t_inv_ref,
        buffer_count=pipeline_mode.buffer_count,
        use_lookahead=pipeline_mode.use_lookahead,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )
    allocs.append(t_inv_alloc)

  if chunk_states_ref is not None:
    chunk_states_shape = (
        cfg.seq_tile_size,
        cfg.num_v_heads,
        cfg.kq_head_dim,
        cfg.v_head_dim,
    )
    chunk_states_alloc = ChunkStatesBufferedRef.output(
        spec=block_spec_partial(block_shape=chunk_states_shape),
        dtype_or_type=chunk_states_ref,
        buffer_count=pipeline_mode.buffer_count,
        use_lookahead=pipeline_mode.use_lookahead,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )
    allocs.append(chunk_states_alloc)

  return tuple(allocs)
