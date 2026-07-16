# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
# Forked from:
# https://github.com/openxla/tokamax/blob/a1105e7513c4cc8604bad5627d099dcf09430ca1/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py
"""GMM kernel implemented using Pallas."""

from abc import ABC, abstractmethod
import dataclasses
import functools
from typing import Any, Callable, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Util.


def swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0) -> jax.Array:
  """Activation used in some models such as GPT-OSS."""

  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  glu = gate * jax.nn.sigmoid(alpha * gate)
  return (up + 1.0) * glu


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
  """Applies a fused activation function to the accumulator.

  This function is used when an activation function is fused with the matrix
  multiplication. The input accumulator `acc` is expected to contain
  concatenated results for both the 'gate' and 'up' projections.

  Args:
    acc: The accumulator array, with the last dimension being 2 * tile_n.
    fuse_act: The name of the activation function to apply. Supported values are
      "silu", "gelu", and "swigluoai". If None, no activation is applied.

  Returns:
    The result of applying the activation function.

  Raises:
    NotImplementedError: If an unsupported `fuse_act` is provided.
  """

  if fuse_act is None:
    return acc

  acc_gate, acc_up = jnp.split(acc, 2, -1)
  match fuse_act:
    case "silu":
      return jax.nn.silu(acc_gate) * acc_up
    case "gelu":
      return jax.nn.gelu(acc_gate) * acc_up
    case "swigluoai":
      return swigluoai(acc_gate, acc_up)
    case _:
      raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
  return pl.cdiv(x, a) * a


# Define data classes.


class RhsRef(ABC):
  """Abstract class that defines interfaces for rhs values."""

  @abstractmethod
  def get_weight(self) -> jax.Array:
    ...

  @abstractmethod
  def get_scale(self) -> jax.Array:
    ...

  @abstractmethod
  def get_bias(self) -> jax.Array:
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
  """Dataclass for a single weights."""

  weight: Any
  scale: Any | None
  bias: Any | None

  def get_weight(self) -> jax.Array:
    return self.weight[...]

  def get_scale(self) -> jax.Array:
    assert self.scale is not None
    return self.scale[...]

  def get_bias(self) -> jax.Array:
    assert self.bias is not None
    return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
  """Dataclass for gate and up weights used in fused activation."""

  gate: WeightsRef
  up: WeightsRef

  def get_weight(self) -> jax.Array:
    w_gate = self.gate.get_weight()
    w_up = self.up.get_weight()
    return jnp.concatenate([w_gate, w_up], axis=-1)

  def get_scale(self) -> jax.Array:
    s_gate = self.gate.get_scale()
    s_up = self.up.get_scale()
    return jnp.concatenate([s_gate, s_up], axis=-1)

  def get_bias(self) -> jax.Array:
    b_gate = self.gate.get_bias()
    b_up = self.up.get_bias()
    return jnp.concatenate([b_gate, b_up], axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LhsRef:
  """Dataclass for the lhs value and its optional quantization scale.

  Unlike `rhs`, the lhs is passed to the kernel *unquantized*. When
  `scale` is provided, the kernel uses it to quantize the lhs (i.e.
  `qvalue = clip(lhs / scale)` and the result is multiplied back by `scale`).
  The scale's shape encodes the granularity (per-tensor `[1, 1]`; extensible to
  per-channel `[M, 1]` and sub-channel `[M, num_blocks]`).
  """

  value: Any
  scale: Any | None

  def get_value(self) -> jax.Array:
    return self.value[...]

  def get_scale(self) -> jax.Array:
    assert self.scale is not None
    return self.scale[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  gm_id_to_group_id: jax.Array
  gm_id_to_m_offset: jax.Array


@dataclasses.dataclass(frozen=True)
class TileSizes:
  tile_m: int
  tile_k: int
  tile_n: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
  size_m: int
  size_k: int
  size_n: int
  size_group: int
  size_lhs_group: int
  size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
  """Configuration parameters for input tensors."""

  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  dtype: jnp.dtype
  has_bias: bool = False
  # Whether a scale array accompanies this input. The *direction* is inferred
  # from the dtype relationship: when the input already arrives quantized
  # (dtype == quant_dtype) the scale dequantizes it (rhs); when it arrives
  # unquantized (dtype != quant_dtype) the scale quantizes it online (lhs).
  has_scale: bool = False

  @property
  def should_use_external_scale(self) -> bool:
    # A scale is present but the input is not yet quantized
    # (dtype != quant_dtype). The kernel uses it to quantize the input online
    # and multiply the result by the scale after. This differs from an already
    # quantized input (dtype == quant_dtype), whose scale only dequantizes after
    # the matmul.
    return (
        self.has_scale
        and self.quant_dtype is not None
        and self.dtype != self.quant_dtype
    )

  @property
  def should_bitcast(self) -> bool:
    bits = jax.dtypes.itemsize_bits(self.dtype)
    return bits < 8

  @property
  def should_dequantize_before_matmul(self) -> bool:
    if not self.has_scale:
      return False
    assert self.quant_block_size is not None
    mxu_size = pltpu.get_tpu_info().mxu_column_size
    return self.quant_block_size < mxu_size

  @property
  def should_dequantize_after_matmul(self) -> bool:
    return self.has_scale and not self.should_dequantize_before_matmul


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
  """Full configuration details for GMM execution."""

  tiles: TileSizes
  dims: Dimensions
  lhs_cfgs: InputConfigs
  rhs_cfgs: InputConfigs
  out_dtype: jnp.dtype
  acc_dtype: jnp.dtype
  has_partial_sum: bool
  zero_init: bool
  fuse_act: str | None

  @property
  def num_quant_blocks_per_tile_k(self) -> int:
    return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)  # pyrefly: ignore[no-matching-overload]

  @property
  def out_size_n(self) -> int:
    if self.fuse_act is None:
      return self.dims.size_n
    else:
      return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int | None, str | None, bool], TileSizes]


class IndexMaps:
  """Index maps for GMM kernel."""

  def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs

  def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start

    return (pl.ds(row_start, row_size), 0, k_id)

  def lhs_scale_index_map(
      self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array
  ):
    # Per-tensor scale: a single [1, 1] value shared across every tile, so the
    # block always reads index 0. Extension point: when the scale is per-channel
    # or sub-channel, tile the row axis like `lhs_index_map` (using gm_id) and
    # index the K-block axis from `k_id`.
    del gm_id, k_id
    return (0, 0)

  def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    return (group_id, k_id, n_id)

  def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    return (group_id, 0, n_id)

  def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    # Simply multiplying k_id by num_quant_blocks_per_tile_k will not work
    # since a single quant block could be shared along multiple k tile.
    k_row = k_id * self.cfgs.tiles.tile_k
    b_row = k_row // self.cfgs.rhs_cfgs.quant_block_size  # pyrefly: ignore[unsupported-operation]
    b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
    return (group_id, b_tile_id, 0, n_id)

  def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    """Calculates index map for the output tensor."""
    is_last_gm = gm_id == (pl.num_programs(1) - 1)
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
    last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
    row_size = row_end - row_start

    return (pl.ds(row_start, row_size), 0, n_id)

  def ps_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start

    return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> Tuple[Tuple[LhsRef, WeightsRef, pl.BlockSpec | None], pl.BlockSpec]:
  """Generates block specs for the given lhs, rhs, and out refs."""

  index_map = IndexMaps(metadata_ref, cfgs)
  bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

  lhs_value_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
      index_map.lhs_index_map,
  )
  lhs_scale_spec = None
  if cfgs.lhs_cfgs.has_scale:
    lhs_scale_spec = pl.BlockSpec(
        (1, 1),
        index_map.lhs_scale_index_map,
    )
  lhs_block_spec = LhsRef(value=lhs_value_spec, scale=lhs_scale_spec)

  tile_k_rhs = cfgs.tiles.tile_k
  if cfgs.rhs_cfgs.should_bitcast:
    packing = pl.cdiv(32, jax.dtypes.itemsize_bits(cfgs.rhs_cfgs.dtype))
    tile_k_rhs //= packing

  rhs_weight_spec = pl.BlockSpec(
      (None, tile_k_rhs, cfgs.tiles.tile_n),
      index_map.rhs_weight_index_map,
      pipeline_mode=pl.Buffered(buffer_count=3),
  )
  rhs_scale_block_spec = rhs_bias_block_spec = ps_block_spec = None
  if cfgs.rhs_cfgs.has_bias:
    rhs_bias_block_spec = pl.BlockSpec(
        (None, 1, cfgs.tiles.tile_n),
        index_map.rhs_bias_index_map,
    )
  if cfgs.rhs_cfgs.has_scale:
    rhs_scale_block_spec = pl.BlockSpec(
        (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
        index_map.rhs_scale_index_map,
    )

  if cfgs.has_partial_sum:
    ps_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.ps_index_map,
    )

  rhs_block_spec = WeightsRef(
      weight=rhs_weight_spec,
      scale=rhs_scale_block_spec,
      bias=rhs_bias_block_spec,
  )

  out_block_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
      index_map.out_index_map,
  )

  return (lhs_block_spec, rhs_block_spec, ps_block_spec), out_block_spec


# Define kernels.


def inner_kernel(
    # In
    tiled_lhs_ref: LhsRef,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n]
    # Partial Sum
    tiled_ps_ref: jax.Array | None,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
):
  """Inner kernel invoked by emit_pipeline to perform matmul.

  tiled_lhs_ref and tiled_out_ref points to rows [m_start:m_end] of lhs and out.
  Additionally, m_start and m_end does not have to align with tile boundaries
  [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
  [m_end:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
  invalid data and needs to be masked out.

  Args:
    tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
    tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end]. where
      g_id is the group associated with lhs[m_start:m_end, :]
    tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
    partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
      Will be initialized to zero if this is first tile for grid[n_id, :, :].
    acc_ref: Reference to the accumulator.
    metadata_ref: Reference to the metadata.
    cfgs: GmmConfigs.
  """

  def _matmul(is_first_k_step: bool, is_last_k_step: bool):
    tpu_info = pltpu.get_tpu_info()
    mxu_size = tpu_info.mxu_column_size

    # Step 1: Input pre-processing.
    tiled_lhs = tiled_lhs_ref.get_value().reshape(-1, cfgs.tiles.tile_k)[...]
    tiled_rhs = tiled_rhs_ref.get_weight()
    # When rhs is packed (quantized dtype packed into uint32), unpack it
    # back to the original dtype using pltpu.bitcast which operates on K
    # axis. This expands the K dimension back to tile_k.
    if cfgs.rhs_cfgs.should_bitcast:
      tiled_rhs = pltpu.bitcast(tiled_rhs, cfgs.rhs_cfgs.dtype)
    rhs_tile_n = tiled_rhs.shape[1]

    # This should only be taken in the case where we don't requantize
    # the scales and thus we need to dequantize inside VMEM to avoid small
    # contracting dimensions
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
      rhs_qbs = cfgs.rhs_cfgs.quant_block_size
      tiled_rhs_scale = tiled_rhs_ref.get_scale().astype(acc_ref.dtype)
      num_blocks = cfgs.num_quant_blocks_per_tile_k
      tiled_rhs_dequant = tiled_rhs.astype(acc_ref.dtype).reshape(num_blocks, rhs_qbs, rhs_tile_n)
      tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
      tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)

    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    if is_last_k_step and valid_k != 0:
      mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
      tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

    # Step 2: Matmul.
    acc_list = []
    if cfgs.lhs_cfgs.quant_dtype is None:
      # Unquantized matmul path.
      rhs_qbs = cfgs.rhs_cfgs.quant_block_size

      for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n

        acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)
        for b_id in range(cfgs.num_quant_blocks_per_tile_k):
          start_k = b_id * rhs_qbs  # pyrefly: ignore[unsupported-operation]
          end_k = start_k + rhs_qbs  # pyrefly: ignore[unsupported-operation]

          block_acc = jnp.matmul(
              tiled_lhs[:, start_k:end_k],
              tiled_rhs[start_k:end_k, start_n:end_n],
              preferred_element_type=jnp.float32,
          ).astype(acc_ref.dtype)

          if cfgs.rhs_cfgs.should_dequantize_after_matmul:
            tiled_rhs_scale = tiled_rhs_ref.get_scale()
            block_acc *= tiled_rhs_scale[b_id, :, start_n:end_n].astype(acc_ref.dtype)

          acc_n += block_acc
        acc_list.append(acc_n)
    else:
      # Quantized matmul path.
      lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
      q_block_size = cfgs.lhs_cfgs.quant_block_size

      if jnp.issubdtype(lhs_q_dtype, jnp.floating):
        dtype_max = float(jnp.finfo(lhs_q_dtype).max)
        preferred_element_type = jnp.float32
      else:
        dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
        preferred_element_type = jnp.int32

      # When the caller supplies a quantization scale, use it directly instead
      # of computing a dynamic per-block absmax.
      lhs_scale = lhs_scale_inv = None
      should_use_external_scale = cfgs.lhs_cfgs.should_use_external_scale
      if should_use_external_scale:
        lhs_scale = tiled_lhs_ref.get_scale().astype(acc_ref.dtype)
        lhs_scale_inv = 1.0 / lhs_scale

      # Without n outer loop, result of quantized matmul becomes available only
      # at the last iteration of the loop. This means [tile_m, tile_n] value
      # needs to be stored until the last iteration. By adding n outer loop,
      # result of [tile_m, mxu_size] becomes available at the end of every k
      # inner loop which can be used to pipeline subsequent VPU or VST ops with
      # MXU ops for the next [tile_m, mxu_size].
      for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n

        acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)
        for start_k in range(0, cfgs.tiles.tile_k, q_block_size):  # pyrefly: ignore[bad-argument-type]
          end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)  # pyrefly: ignore[unsupported-operation]

          block_lhs = tiled_lhs[:, start_k:end_k]
          block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

          # Perform lhs quantization. Note that for every block_lhs,
          # same computation will be performed tiles_n//mxu_size times.
          # But we can let compiler perform CSE and avoid recomputation.
          if should_use_external_scale:
            assert lhs_scale is not None
            assert lhs_scale_inv is not None
            block_lhs_q = jnp.clip(
                block_lhs * lhs_scale_inv, -dtype_max, dtype_max
            ).astype(lhs_q_dtype)
            block_scale = lhs_scale  # [1, 1]
          else:
            block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
            block_scale = block_abs_max / dtype_max

            # If block_scale=0, it will cause division by zero and return either
            # NaN or Inf. Since this can cause numeric issue when downcasting to
            # quantized value, we convert them into 0.
            block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
            # Convert lhs into quantized dtype.
            block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)

          # Unlike unquantized path, compiler may not perform implicit type
          # conversion due to numeric concerns. As this can cause unsupported
          # matmul error, explicit type conversion is performed.
          if not tpu_info.is_matmul_supported(lhs_q_dtype, block_rhs.dtype):
            block_rhs = block_rhs.astype(lhs_q_dtype)

          block_acc = jnp.matmul(
              block_lhs_q,
              block_rhs,
              preferred_element_type=preferred_element_type,
          ).astype(acc_ref.dtype)

          block_acc *= block_scale.astype(acc_ref.dtype)

          # Apply rhs subchannel scale per quant block.
          if cfgs.rhs_cfgs.should_dequantize_after_matmul:
            b_id = start_k // cfgs.rhs_cfgs.quant_block_size  # pyrefly: ignore[unsupported-operation]
            rhs_scale_slice = tiled_rhs_ref.get_scale()
            block_acc *= rhs_scale_slice[b_id, :, start_n:end_n].astype(acc_ref.dtype)

          acc_n += block_acc
        acc_list.append(acc_n)
    acc = jnp.concatenate(acc_list, axis=1)

    # Step 3: Output post-processing.
    if not is_first_k_step:
      acc += acc_ref[...]

    if is_last_k_step:
      if cfgs.rhs_cfgs.has_bias:
        tiled_rhs_bias = tiled_rhs_ref.get_bias()
        acc += tiled_rhs_bias.astype(acc.dtype)
      if cfgs.has_partial_sum:
        ps_tile = tiled_ps_ref[...].reshape(acc.shape)  # pyrefly: ignore[unsupported-operation]
        acc += ps_tile.astype(acc.dtype)

      acc = apply_act_fn(acc, cfgs.fuse_act)

      gm_id = pl.program_id(1)

      # Mask out rows that does not belong to the current group.
      m_start = metadata_ref.gm_id_to_m_offset[gm_id]
      m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
      m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

      m_start_local = m_start - m_offset
      m_end_local = m_end - m_offset

      iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
      mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
      acc_masked = jnp.where(mask, acc, 0).reshape(tiled_out_ref.shape)

      # Write the final output to the output ref.
      tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)

      # If this is the first tile for grid[n_id, :, :], we initialize the
      # partial out to zeros. Otherwise, partial out from last tile of
      # grid[n_id-1, :, :] can be used and cause numeric issues.
      partial_out_zeros = jnp.zeros_like(partial_out_ref)

      # Accumulate the partial output from the previous step.
      tiled_out_ref[0] += jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])

      # Consider following case where size_lhs_sublane = 4, number denotes group
      # id and | denotes boundaries between sublanes:
      # | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 |
      #
      # Assuming group id of current step is 1, current step will not completely
      # fill size_lhs_sublane rows and will be revisited at the next step. By
      # storing the partial rows into the partial_out_ref, the next step can
      # read them and accumulate to them.  Additionally, for group id of 2,
      # since it completely fills the size_lhs_sublane rows, we need to zero out
      # partial_out_ref to avoid numeric error for group 3.
      last_row = m_end_local // cfgs.dims.size_lhs_sublane
      partial_out_ref[...] = jnp.where(
          m_end_local % cfgs.dims.size_lhs_sublane == 0,
          partial_out_zeros,
          tiled_out_ref[last_row],
      )
    else:
      acc_ref[...] = acc

  # Define matmul wrapper functions.
  @jax.named_scope("matmul_first_last")
  def matmul_first_last():
    _matmul(is_first_k_step=True, is_last_k_step=True)

  @jax.named_scope("matmul_first")
  def matmul_first():
    _matmul(is_first_k_step=True, is_last_k_step=False)

  @jax.named_scope("matmul")
  def matmul():
    _matmul(is_first_k_step=False, is_last_k_step=False)

  @jax.named_scope("matmul_last")
  def matmul_last():
    _matmul(is_first_k_step=False, is_last_k_step=True)

  # Select and execute matmul function based on the current step.
  num_k = pl.num_programs(2)
  k_id = pl.program_id(2)

  is_first_k_step = k_id == 0
  is_last_k_step = k_id == (num_k - 1)

  lax.cond(
      is_first_k_step,
      lambda: lax.cond(
          is_last_k_step,
          matmul_first_last,
          matmul_first,
      ),
      lambda: lax.cond(
          is_last_k_step,
          matmul_last,
          matmul,
      ),
  )


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
  """Fills the metadata for the given lhs group sizes and group offset.

  Iterates over the lhs group sizes and if the group id is valid, determines
  the number of gm tiles that are needed to process the current group. Then,
  it fills starting and ending offset (gm_id_to_m_offset), and the group id
  (gm_id_to_group_id) for each gm tile.

  Args:
    lhs_group_sizes_ref: The group sizes of lhs.
    group_offset_ref: Offset of the first group to process.
    metadata_ref: Metadata that is used to determine the group id and m offsets
      for each gmm tile.
    cfgs: GmmConfigs.

  Returns:
      The number of gm tiles to process lhs with given group offset.
  """

  group_offset = group_offset_ref[0]
  max_num_group = group_offset + cfgs.dims.size_group
  metadata_ref.gm_id_to_m_offset[0] = 0

  @jax.named_scope("inner_tm_loop")
  def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
    local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
    tm_size = jnp.minimum(cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset)

    metadata_ref.gm_id_to_group_id[tm_id] = group_id

    next_m_offset = curr_m_offset + tm_size
    metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
    metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset

    return next_m_offset

  @jax.named_scope("outer_group_loop")
  def outer_group_loop(lhs_group_id, carry):
    num_gm, start_m_offset = carry

    group_id = lhs_group_id - group_offset
    group_size = lhs_group_sizes_ref[lhs_group_id]
    end_m_offset = start_m_offset + group_size

    # Assume following arguments:
    # - size_lhs_sublane & tile_m = 4
    # - group_size = 3
    # - start_m_offset = 7
    #
    # If we visualize it, it will look like this where:
    # - |: denotes boundaries between sublanes
    # - 0: denotes values for other groups
    # - 1: denotes values for the current group
    # | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 |
    #
    # In this example, we see that we require processing 2 m tiles.
    # But, performing a naive cdiv(group_size, tile_m) will return 1.
    # Instead, adding local_offset will give us the correct value.
    local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
    aligned_group_size = group_size + local_offset
    curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

    # We need to handle cases where we should not process the group.
    # 1. Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
    # 2. If group comes before the group_offset, we should not process it.
    should_process = jnp.logical_and(group_size > 0, group_id >= 0)
    curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
    next_num_gm = num_gm + curr_num_gm

    tm_loop_fn = functools.partial(
        inner_tm_loop,
        end_m_offset=end_m_offset,
        group_id=group_id,
    )
    lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)

    return next_num_gm, end_m_offset

  num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
  return num_gm


def zero_out_start(
    out_ref: jax.Array,  # [size_m, size_n]
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
):
  """Zero out output rows that are not used in the computation."""

  num_lanes = pltpu.get_tpu_info().num_lanes
  assert num_lanes == zero_ref.shape[-1]
  zero_ref[...] = jnp.zeros_like(zero_ref)

  zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
  out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
  row_size = zero_dma.shape[0]

  compute_start = metadata_ref.gm_id_to_m_offset[0]
  compute_end = metadata_ref.gm_id_to_m_offset[num_gm]

  left_zero_start = 0
  left_zero_end = compute_start // dims.size_lhs_sublane
  left_zero_size = left_zero_end - left_zero_start
  left_num_loops = pl.cdiv(left_zero_size, row_size)

  right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
  right_zero_end = out_dma.shape[0]
  right_zero_size = right_zero_end - right_zero_start
  right_num_loops = pl.cdiv(right_zero_size, row_size)

  def fill_zero(i, zero_size, *, start, end):
    dma_start = start + i * row_size
    dma_end = jnp.minimum(dma_start + row_size, end)
    dma_size = dma_end - dma_start

    # Static loop. Will be unrolled during compile time.
    for n_start in range(0, out_dma.shape[-1], num_lanes):
      n_end = n_start + num_lanes
      pltpu.make_async_copy(
          src_ref=zero_dma.at[pl.ds(0, dma_size)],
          dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :, n_start:n_end],
          sem=semaphore_ref.at[0],
      ).start(priority=1)

    return zero_size + dma_size

  @jax.named_scope("left_fill_zero")
  def left_fill_zero(i, zero_size):
    return fill_zero(i, zero_size, start=left_zero_start, end=left_zero_end)

  @jax.named_scope("right_fill_zero")
  def right_fill_zero(i, zero_size):
    return fill_zero(i, zero_size, start=right_zero_start, end=right_zero_end)

  zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
  zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
  return zero_size


def zero_out_end(
    out_ref: jax.Array,  # [size_m, size_n]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
  out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
  pltpu.make_async_copy(
      src_ref=out_dma.at[pl.ds(0, zero_size)],
      dst_ref=out_dma.at[pl.ds(0, zero_size)],
      sem=semaphore_ref.at[0],
  ).wait()


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: LhsRef,  # value: [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    partial_sum_ref: jax.Array,  # [size_m, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array | None,  # [1]
    *,
    cfgs: GmmConfigs,
):
  """Entry point for GMM kernel.

  Computes metadata to determine which rows of lhs needs processing and how
  they will be tiled. And then, invoke inner kernel using metadata.

  Uses the following notation:
  - g: rhs group dimension
  - m: Batch dimension
  - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
    of tile_m. Skips over empty groups and accounts for revisited tiles.
  - k: in dimension
  - n: out dimension

  Args:
    lhs_group_sizes_ref: Reference to the group sizes of lhs.
    group_offset_ref: Reference to the group offset.
    lhs_ref: Reference to the lhs.
    rhs_ref: Reference to the rhs.
    out_ref: Reference to the out.
    partial_out_ref: Reference to the partial output.
    acc_ref: Reference to the accumulator.
    metadata_ref: Reference to the metadata.
    zero_ref: Scratch memory for storing zero values used in initialization.
    semaphore_ref: Semaphore for zero initialization DMAs.
    cfgs: GmmConfigs.
  """

  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

  # Pack along K (2nd minor dim) so that pltpu.bitcast can unpack inside the
  # kernel.
  # [G, K, N] -> [G, K//packing, N] uint32
  if cfgs.rhs_cfgs.should_bitcast:
    rhs_weight = rhs_ref.weight.bitcast(jnp.uint32)
    rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

  # Fill metadata buffer and return number of group & m iterations.
  num_gm = fill_metadata(
      lhs_group_sizes_ref,
      group_offset_ref,
      metadata_ref,
      cfgs=cfgs,
  )

  if cfgs.zero_init:
    zero_size = zero_out_start(
        out_ref,
        zero_ref,  # pyrefly: ignore[bad-argument-type]
        semaphore_ref,  # pyrefly: ignore[bad-argument-type]
        metadata_ref,
        num_gm,
        dims=cfgs.dims,
    )

  (lhs_spec, rhs_spec, ps_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

  if cfgs.fuse_act is not None:
    rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
    rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)  # pyrefly: ignore[bad-assignment]

    rhs_spec = FusedWeightsRef(
        gate=rhs_spec,
        up=rhs_spec,
    )

  # Execute the inner kernel.
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(inner_kernel, cfgs=cfgs),
      grid=(num_n, num_gm, num_k),
      in_specs=(lhs_spec, rhs_spec, ps_spec),
      out_specs=out_spec,
  )

  # Bounded slice requires second last dim to be aligned to the sublane size.

  # rhs_ref uses static tiling thus reshape is not needed. The lhs quant scale
  # (when present) is small and statically tiled, so it is passed through as-is.
  lhs_value_in = lhs_ref.value.reshape(
    -1, cfgs.dims.size_lhs_sublane, lhs_ref.value.shape[-1]
  )
  lhs_in = LhsRef(value=lhs_value_in, scale=lhs_ref.scale)

  ps_in = None
  if cfgs.has_partial_sum:
    ps_in = partial_sum_ref.reshape(-1, cfgs.dims.size_lhs_sublane, partial_sum_ref.shape[-1])
  out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
  scratches = [partial_out_ref, acc_ref, metadata_ref]
  pipeline_fn(lhs_in, rhs_ref, ps_in, out_in, scratches=scratches)

  if cfgs.zero_init:
    zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)  # pyrefly: ignore[bad-argument-type, unbound-name]


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
    has_partial_sum: bool = False,
) -> TileSizes:
  """Calculate optimal tile sizes for GMM kernel."""

  lhs_dtype = lhs_cfgs.quant_dtype or lhs_cfgs.dtype
  rhs_dtype = rhs_cfgs.dtype

  lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
  rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

  # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
  # safe to use for most scenarios. But if lower bitwidth is used, we need
  # to tweak tile_m to account for using faster hardware unit.
  # TODO: Account for different TPU hardware specs.
  bf16_bf16_tile_m = 128
  lhs_mod = min(pl.cdiv(16, lhs_bits), 2)
  rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
  tile_m = bf16_bf16_tile_m * lhs_mod // rhs_mod
  tile_m = min(tile_m, dims.size_m)

  # To avoid stalling MXU, we add some buffer room where tile_n cannot go
  # smaller than 2x of mxu_column_size.
  tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
  tile_n_limit = min(tile_n_limit, dims.size_n)

  size_n_per_rhs = dims.size_n
  fuse_act_factor = 1
  if fuse_act is not None:
    # When computing activation function, rhs is concatenated along dim n.
    fuse_act_factor = 2
    size_n_per_rhs //= fuse_act_factor
    tile_n_limit //= fuse_act_factor

  def _is_tile_k_quant_block_compatible(tk: int) -> bool:
    if (
        tk % rhs_cfgs.quant_block_size != 0 and rhs_cfgs.quant_block_size % tk != 0
    ):  # pyrefly: ignore[unsupported-operation]
      return False
    return True

  # Initialize tile_k and tile_n to their maximum valid values.
  num_k_tiles = num_n_tiles = 1
  num_lanes = pltpu.get_tpu_info().num_lanes
  tile_k = align_to(dims.size_k, num_lanes)
  tile_n = align_to(size_n_per_rhs, num_lanes)

  def _gmm_vmem_estimate(tn: int, tk: int) -> int:
    # 1. LHS tile (double-buffered)
    lhs_tile_bytes = lhs_bits // 8
    lhs_vmem = 2 * tile_m * tk * lhs_tile_bytes

    # 2. RHS tile (triple-buffered, includes scale and bias if present)
    # If fuse_act is enabled, we have both gate and up weights,
    # so RHS memory is doubled.
    rhs_weight_vmem = tk * tn * rhs_bits // 8
    rhs_scale_vmem = 0
    if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
      num_quant_blocks_per_tile_k = pl.cdiv(tk, rhs_cfgs.quant_block_size)
      rhs_scale_vmem = num_quant_blocks_per_tile_k * tn * 4
    rhs_bias_vmem = 0
    if rhs_cfgs.has_bias:
      rhs_bias_vmem = tn * 4
    rhs_vmem = fuse_act_factor * (3 * rhs_weight_vmem + 2 * rhs_scale_vmem + 2 * rhs_bias_vmem)

    # 3. Accumulator
    acc_cols = fuse_act_factor * tn
    acc_dtype_bytes = 2 if lhs_cfgs.quant_dtype is not None else 4
    acc_vmem = tile_m * acc_cols * acc_dtype_bytes

    # 4. Output tile (double-buffered) and partial sum buffer
    out_dtype_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    out_vmem = 2 * tile_m * tn * out_dtype_bytes
    ps_vmem = 2 * tile_m * tn * out_dtype_bytes if has_partial_sum else 0
    partial_out_vmem = dims.size_lhs_sublane * tn * out_dtype_bytes

    return lhs_vmem + rhs_vmem + acc_vmem + out_vmem + ps_vmem + partial_out_vmem

  # Multiple k tiles will introduce accumulation overhead. Thus, we first try
  # to fit the tensors into vmem by only adjusting tile_n.

  # Decrease tile_n until total memory fits in vmem limit.
  while _gmm_vmem_estimate(tile_n, tile_k) > vmem_limit_bytes and tile_n > tile_n_limit:
    num_n_tiles += 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

  # If decreasing tile_n is no longer possible, we decrease tile_k instead.
  if tile_n < tile_n_limit:
    num_n_tiles -= 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

    # Decrease tile_k until total memory fits in vmem limit and tile_k is valid.
    while _gmm_vmem_estimate(tile_n, tile_k) > vmem_limit_bytes or not _is_tile_k_quant_block_compatible(tile_k):
      num_k_tiles += 1
      tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

  if tile_n == 0 or tile_k == 0:
    final_estimate = _gmm_vmem_estimate(tile_n, tile_k)
    raise ValueError(
        f"Could not find valid tile sizes for {dims=} and" f" {final_estimate=} (limit: {vmem_limit_bytes})."
    )

  return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    partial_sum: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
    maybe_quantize_lhs: bool = True,
    lhs_scale: jax.Array | None = None,
) -> Dimensions:
  """Validates the inputs for the GMM kernel."""

  size_m = lhs.shape[0]
  size_group, size_k, size_n = rhs.shape
  size_lhs_group = group_sizes.shape[0]

  assert size_group <= size_lhs_group
  assert lhs.shape == (size_m, size_k)
  assert rhs.shape == (size_group, size_k, size_n)
  if rhs_bias is not None:
    assert rhs_bias.shape == (size_group, 1, size_n)
  if partial_sum is not None:
    assert partial_sum.shape[-1] == size_n
    # lhs's m dimension can sometimes be padded to wi_tile_fwd_batch_seq
    assert partial_sum.shape[0] <= size_m
  if rhs_scale is not None:
    num_quant_blocks = rhs_scale.shape[1]
    assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n), (
        f"rhs_scale shape {rhs_scale.shape}. Expecting ({size_group},"
        f" {num_quant_blocks}, 1, {size_n})"
    )
    assert size_k % num_quant_blocks == 0

  if lhs_scale is not None:
    assert maybe_quantize_lhs, (
        "lhs_scale requires maybe_quantize_lhs=True."
    )
    # Only per-tensor scales are supported for now. The current implementation generalizes to per-channel [M, 1] and
    # sub-channel [M, num_k_blocks]; extend the validation and the block spec /
    # index map together when adding those.
    assert lhs_scale.shape == (1, 1), (
        "Only per-tensor lhs_scale of shape (1, 1) is supported, got "
        f"{lhs_scale.shape}."
    )

  assert group_offset.shape == (1,)

  size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
  size_lhs_sublane = min(size_lhs_sublane, size_m)
  if fuse_act is not None:
    num_lanes = pltpu.get_tpu_info().num_lanes
    if size_n % (2 * num_lanes) != 0:
      raise ValueError(
          f"{size_n=} should be divisible by 2 * num_lanes when fuse_act is "
          "enabled since we need to split n dimension for gate and up."
      )

  return Dimensions(
      size_m=size_m,
      size_k=size_k,
      size_n=size_n,
      size_group=size_group,
      size_lhs_group=size_lhs_group,
      size_lhs_sublane=size_lhs_sublane,
  )


def get_cost_estimate(cfgs: GmmConfigs):
  """Returns the cost estimate for the GMM kernel."""

  dims = cfgs.dims
  lhs_dtype = cfgs.lhs_cfgs.quant_dtype or cfgs.lhs_cfgs.dtype
  rhs_dtype = cfgs.rhs_cfgs.dtype

  # We use bits for rhs since it could sub-byte dtype like int4.
  rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
  fp32_bytes = jnp.dtype(jnp.float32).itemsize

  # TODO: Add compute flops for quant, dequant, and bias.
  flops = 2 * dims.size_m * dims.size_k * dims.size_n

  lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

  rhs_size = dims.size_group * dims.size_k * dims.size_n
  rhs_bytes = rhs_size * rhs_bits // 8
  if cfgs.rhs_cfgs.has_scale:
    num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)  # pyrefly: ignore[no-matching-overload]
    rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes
  if cfgs.rhs_cfgs.has_bias:
    rhs_bytes += dims.size_group * dims.size_n * fp32_bytes

  out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize

  total_bytes = lhs_bytes + rhs_bytes + out_bytes

  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=total_bytes,
      transcendentals=0,
  )


def get_scope_name(cfgs: GmmConfigs) -> str:
  dims = cfgs.dims
  tiles = cfgs.tiles
  return (
      f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    partial_sum: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
    fuse_act: str | None = None,
    lhs_scale: jax.Array | None = None,
):
  """Fills the GMM config for the GMM kernel."""

  dims = validate_inputs(
      lhs,
      rhs,
      rhs_scale,
      rhs_bias,
      group_sizes,
      group_offset,
      fuse_act,
      maybe_quantize_lhs,
      lhs_scale,
  )

  if rhs_scale is not None:
    has_scale = True
    rhs_quant_dtype = rhs.dtype
    num_blocks = rhs_scale.shape[1]
    block_size = dims.size_k // num_blocks
  else:
    has_scale = False
    rhs_quant_dtype = None
    block_size = dims.size_k

  rhs_cfgs = InputConfigs(
      quant_dtype=rhs_quant_dtype,
      quant_block_size=block_size,
      dtype=rhs.dtype,
      has_bias=rhs_bias is not None,
      has_scale=has_scale,
  )

  lhs_q_dtype = None
  if maybe_quantize_lhs and rhs_cfgs.should_dequantize_after_matmul:
    # Choose lhs quantization dtype based on TPU hardware support.
    is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)  # pyrefly: ignore[bad-argument-type]
    tpu_info = pltpu.get_tpu_info()
    # Check if there is hardware compute support for rhs dtype group.
    if tpu_info.fp8_ops_per_second > 0:
      # Special handling for 4-bit integer rhs as it can be converted to fp8
      # without a numeric issues. Note that this is not the case for 4-bit
      # floating rhs as conversion to int8 will cause numeric issues.
      is_rhs_4bits = jax.dtypes.itemsize_bits(rhs_quant_dtype) == 4  # pyrefly: ignore[bad-argument-type]
      if is_rhs_float or is_rhs_4bits:
        lhs_q_dtype = jnp.float8_e4m3fn.dtype
    if tpu_info.int8_ops_per_second > 0:
      if not is_rhs_float:
        lhs_q_dtype = jnp.int8.dtype

  if lhs_scale is not None:
    assert lhs_q_dtype is not None, (
        "lhs_scale requires lhs quantization to engage, but no lhs quant "
        "dtype was selected. Ensure rhs is quantized and the hardware supports "
        "fp8/int8 matmul."
    )
  has_lhs_scale = lhs_scale is not None and lhs_q_dtype is not None

  lhs_cfgs = InputConfigs(
      quant_dtype=lhs_q_dtype,
      # Input quantization involves reading all elements in a block to compute
      # scale value. Since this operation is very memory intensive, we use a
      # block size that is small enough to minimize memory overhead but large
      # enough to minimize compute overhead of quantization.
      quant_block_size=512,
      dtype=lhs.dtype,
      has_scale=has_lhs_scale,
  )

  if out_dtype is None:
    out_dtype = lhs.dtype

  if acc_dtype is None:
    if lhs_cfgs.quant_dtype is None:
      acc_dtype = jnp.float32.dtype
    else:
      # Input quantization requires elementwise ops which can put pressure on
      # VPUs. Using faster bf16 hardware during accumulation can help offset the
      # pressure.
      acc_dtype = jnp.bfloat16.dtype

  if isinstance(tile_info, TileSizes):
    tiles = tile_info
  else:
    tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act, partial_sum is not None)

  return GmmConfigs(
      dims=dims,
      tiles=tiles,
      lhs_cfgs=lhs_cfgs,
      rhs_cfgs=rhs_cfgs,
      out_dtype=jnp.dtype(out_dtype),
      acc_dtype=jnp.dtype(acc_dtype),
      has_partial_sum=partial_sum is not None,
      zero_init=zero_initialize,
      fuse_act=fuse_act,
  )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str | int | float]:
  cfgs_dict = dataclasses.asdict(cfgs)
  ret = {}
  for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
    key = jax.tree_util.keystr(path, simple=True, separator=".")
    if not isinstance(val, str | int | float):
      val = str(val)
    ret[key] = val
  return ret


@jax.jit(
    static_argnames=[
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
        "maybe_quantize_lhs",
        "zero_initialize",
        "fuse_act",
    ]
)
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    partial_sum: jax.Array | None = None,  # [size_m, size_n]
    group_offset: jax.Array | None = None,  # int32[1]
    lhs_scale: jax.Array | None = None,  # [1, 1] (per-tensor)
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,  # pyrefly: ignore[bad-function-definition]
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
) -> jax.Array:
  """GMM kernel implemented with emit_pipeline.

  Dynamically calculate offset lhs/out tiles to reduce redundant computations.
  Additionally, it adjusts dma size based on number of valid rows and utilize
  triple buffering on weights to better utilize memory.

  Args:
    lhs: lhs with shape [size_m, size_k].
    rhs: rhs with shape [size_group, size_k, size_n].
    group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
    rhs_scale: The rhs scale of shape [size_group, num_blocks, 1, out_size].
    rhs_bias: The rhs bias of shape [size_group, 1, out_size].
    partial_sum: Optional. Per-token partial sums of shape [size_m, size_n].
    group_offset: Optional. The group offset of shape [1,].
    lhs_scale: Optional scale used to quantize the (unquantized) lhs
      inside the kernel and the result is multiplied back by `scale`. The shape
      encodes granularity; currently only per-tensor `[1, 1]` is supported. When
      None, a quantized lhs uses the default dynamic per-block absmax
      calibration. Only takes effect when maybe_quantize_lhs is True and rhs is
      quantized.
    tile_info: The tile sizes or tile function to use.
    vmem_limit_bytes: Optional vmem limit in bytes.
    precision: Unused. Exists for compatibility reasons.
    preferred_element_type: Optional jnp.dtype for the output matrix.
    acc_dtype: Optional jnp.dtype for the accumulator.
    maybe_quantize_lhs: Quantize lhs if set to True and rhs is quantized.
    zero_initialize: Whether to initialize unvisited output elements to zero.
    fuse_act: Activation function to fuse with GMM, None if no fusion.

  Returns:
    Output of shape [size_m, size_n].
  """

  del precision

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

  cfgs = make_gmm_configs(
      lhs,
      rhs,
      rhs_scale,
      rhs_bias,
      partial_sum,
      group_sizes,
      group_offset,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      out_dtype=preferred_element_type,
      acc_dtype=acc_dtype,
      maybe_quantize_lhs=maybe_quantize_lhs,
      zero_initialize=zero_initialize,
      fuse_act=fuse_act,
      lhs_scale=lhs_scale,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles

  # Prepare block specs.
  lhs_scale_spec = None
  if cfgs.lhs_cfgs.has_scale:
    assert lhs_scale is not None
    lhs_scale = lhs_scale.astype(jnp.float32)
    lhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  else:
    lhs_scale = None

  rhs_scale_spec = rhs_bias_spec = None
  if rhs_scale is not None:
    rhs_scale = rhs_scale.astype(jnp.float32)
    rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  if rhs_bias is not None:
    rhs_bias = rhs_bias.astype(jnp.float32)
    rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

  # Initialize scratch shapes.
  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
  acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
  scratch_shapes = [
      # partial_out_ref
      pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
      # acc_ref
      pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
      # metadata_ref
      MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]

  num_lanes = pltpu.get_tpu_info().num_lanes
  if cfgs.zero_init:
    # TODO: Create better heuristics for determining this value.
    target_zero_ref_bytes = 2 * 1024 * 1024

    # Zero initialization is done by tiling size_m dim where each tile invokes
    # zero initializing DMA for up-to tile_zero_m rows. This means larger
    # tile_zero_m will result in fewer number of tiles and lead to smaller
    # overhead. However, in order to invoke DMA call up-to tile_zero_m rows, we
    # need to store equivalent sized memory in VMEM buffer for the duration of
    # DMA. Storing [tile_zero_m, size_n] in buffer will trigger OOM if
    # tile_zero_m is too large. Instead, if we set column size as num_lanes
    # (which is smallest allowed column size for DMA) and reuse the buffer by
    # size_n//num_lanes times in a single tile, we can significantly increase
    # tile_zero_m without triggering OOM.
    out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
    tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
    tile_zero_m = min(tile_zero_m, dims.size_m)

    scratch_shapes += [
        pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
        pltpu.SemaphoreType.DMA((1,)),
    ]
  else:
    scratch_shapes += [None, None]

  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
  lhs_in = LhsRef(value=lhs, scale=lhs_scale)
  rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)
  in_specs = [
      pl.BlockSpec(memory_space=pltpu.HBM),
      WeightsRef(
          weight=pl.BlockSpec(memory_space=pltpu.HBM),
          scale=rhs_scale_spec,
          bias=rhs_bias_spec,
      ),
  ]

  partial_sum_spec = None
  if partial_sum is not None:
    in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
    partial_sum_spec = pl.BlockSpec(memory_space=pltpu.HBM)

  in_specs = [
      LhsRef(
          value=pl.BlockSpec(memory_space=pltpu.HBM),
          scale=lhs_scale_spec,
      ),
      WeightsRef(
          weight=pl.BlockSpec(memory_space=pltpu.HBM),
          scale=rhs_scale_spec,
          bias=rhs_bias_spec,
      ),  # rhs_weights
      partial_sum_spec,  # partial_sum
  ]

  input_output_aliases = {}
  if partial_sum is not None:
    flat_args_preceding = (group_sizes, group_offset, lhs, rhs_weights)
    leaves = jax.tree_util.tree_leaves(flat_args_preceding)
    partial_sum_idx = sum(1 for x in leaves if x is not None)
    input_output_aliases = {partial_sum_idx: 0}

  return pl.pallas_call(
      functools.partial(kernel_main, cfgs=cfgs),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=in_specs,
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          scratch_shapes=scratch_shapes,  # pyrefly: ignore[bad-argument-type]
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=get_scope_name(cfgs),
      cost_estimate=get_cost_estimate(cfgs),
      metadata=get_metadata(cfgs),
      input_output_aliases=input_output_aliases,
  )(group_sizes, group_offset, lhs_in, rhs_weights, partial_sum)[:, : cfgs.out_size_n]
