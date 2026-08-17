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

import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.causal_conv1d_gated_delta_rule import config


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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
  """Maps physical 1-D data into logical N-D representation."""

  data: Any
  shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  def _get_pos(self, indices: tuple[Any, ...]) -> Any:
    strides = pl.strides_from_shape(self.shape)
    assert len(strides) == len(indices)

    pos = 0
    for stride, idx in zip(strides, indices):
      pos += stride * idx
    return pos

  def __getitem__(self, indices: tuple[Any, ...]) -> Any:
    return self.data[self._get_pos(indices)]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  num_tiles: Any
  p_id_to_s_idx: SmemWrapper
  p_id_to_r_base: SmemWrapper
  p_id_to_r_size: SmemWrapper
  p_id_is_first_tile: SmemWrapper
  p_id_is_last_tile: SmemWrapper
  s_idx_has_initial_state: Any
  s_idx_to_state_indices: Any

  @classmethod
  def create(
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
  ) -> 'MetadataRef':
    # NOTE: First dim does not matter when it comes to calculating stride.
    shape = (1, cfgs.seq_tile_size)
    return cls(
        num_tiles=num_tiles,
        p_id_to_s_idx=SmemWrapper(p_id_to_s_idx, shape),
        p_id_to_r_base=SmemWrapper(p_id_to_r_base, shape),
        p_id_to_r_size=SmemWrapper(p_id_to_r_size, shape),
        p_id_is_first_tile=SmemWrapper(p_id_is_first_tile, shape),
        p_id_is_last_tile=SmemWrapper(p_id_is_last_tile, shape),
        s_idx_has_initial_state=s_idx_has_initial_state,
        s_idx_to_state_indices=s_idx_to_state_indices,
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

  def copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      r_base = self.metadata_ref.p_id_to_r_base[p_id, idx]
      dma_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
      pltpu.make_async_copy(
          src_ref.at[pl.ds(r_base, dma_size)],
          vmem_ref.at[idx, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
          sem,
      ).start()

  def wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      dma_size += self.metadata_ref.p_id_to_r_size[p_id, idx]

    pltpu.make_async_copy(
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutBufferedRef(BaseBufferedRef):

  def copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      r_base = self.metadata_ref.p_id_to_r_base[p_id, idx]
      dma_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
      pltpu.make_async_copy(
          vmem_ref.at[idx, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
          dst_ref.at[pl.ds(r_base, dma_size)],
          sem,
      ).start()

  def wait_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      dma_size += self.metadata_ref.p_id_to_r_size[p_id, idx]

    pltpu.make_async_copy(
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class StateBufferedRef(BaseBufferedRef):

  def copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):

      is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
      s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
      state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]
      has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
      should_read = jnp.logical_and(is_first_tile, has_initial_state)
      dma_size = jnp.where(should_read, 1, 0)

      pltpu.make_async_copy(
          src_ref.at[pl.ds(state_idx, dma_size)],
          vmem_ref.at[pl.ds(idx, dma_size)],  # pyrefly: ignore[missing-attribute]
          sem,
      ).start()

  def wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
      s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
      has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
      should_read = jnp.logical_and(is_first_tile, has_initial_state)
      dma_size += jnp.where(should_read, 1, 0)

    pltpu.make_async_copy(
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()

  def copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
      s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
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
      is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
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
) -> tuple[
    InBufferedRef,
    InBufferedRef,
    InBufferedRef,
    StateBufferedRef,
    StateBufferedRef,
    OutBufferedRef,
]:
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

  return qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc
