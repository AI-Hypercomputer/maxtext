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
"""Pallas TPU grouped matrix multiplication (gmm_v2) kernel."""

import abc
import dataclasses
import functools
import math
from typing import Any, Callable, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from maxtext.experimental.lineage import ops
from maxtext.experimental.lineage.gmmv2 import relayout

# Util.


def swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0) -> jax.Array:
  """Activation used in some models such as GPT-OSS."""

  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  glu = gate * jax.nn.sigmoid(alpha * gate)
  return (up + 1.0) * glu


def silu_and_mul_with_clamp(gate: jax.Array, up: jax.Array, limit: float = 10.0) -> jax.Array:
  """Activation used in some models DeepSeek V4."""
  # The limit value is from DSV4's config.
  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  return jax.nn.silu(gate) * up


def situ_and_mul(gate: jax.Array, up: jax.Array, beta: float, linear_beta: float | None) -> jax.Array:
  gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
  if linear_beta is not None:
    up = linear_beta * jnp.tanh(up / linear_beta)
  return gate * up


def interleave_lane(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """Interleaves two arrays along lane dim at zero-cost."""
  assert lhs.shape == rhs.shape
  chunk_size = pltpu.get_tpu_info().num_lanes
  num_chunks = lhs.shape[-1] // chunk_size
  lhs_chunks = jnp.split(lhs, num_chunks, axis=-1)
  rhs_chunks = jnp.split(rhs, num_chunks, axis=-1)
  interleaved = []
  for i in range(num_chunks):
    interleaved += [lhs_chunks[i], rhs_chunks[i]]
  return jnp.concat(interleaved, axis=-1)


def deinterleave_lane(val: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Deinterleaves an array along lane dim at zero-cost."""
  chunk_size = pltpu.get_tpu_info().num_lanes
  num_chunks = val.shape[-1] // chunk_size
  chunks = jnp.split(val, num_chunks, axis=-1)
  lhs = jnp.concat(chunks[0::2], axis=-1)
  rhs = jnp.concat(chunks[1::2], axis=-1)
  return lhs, rhs


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
  """Applies a fused activation function to the accumulator.

  This function is used when an activation function is fused with the matrix
  multiplication. The input accumulator `acc` is expected to contain
  concatenated results for both the 'gate' and 'up' projections.

  Args:
    acc: The accumulator array, with the last dimension being 2 * tile_n.
    fuse_act: The name of the activation function to apply.

  Returns:
    The result of applying the activation function.

  Raises:
    NotImplementedError: If an unsupported `fuse_act` is provided.
  """

  if fuse_act is None:
    return acc

  acc_gate, acc_up = deinterleave_lane(acc)
  match fuse_act:
    case "silu":
      return jax.nn.silu(acc_gate) * acc_up
    case "gelu":
      return jax.nn.gelu(acc_gate) * acc_up
    case "gelu_tanh":
      return jax.nn.gelu(acc_gate, approximate=True) * acc_up
    case "swigluoai":
      return swigluoai(acc_gate, acc_up)
    case "silu_and_mul_with_clamp":
      return silu_and_mul_with_clamp(acc_gate, acc_up)
    case str() if fuse_act.startswith("situ:"):
      _, beta, linear_beta = fuse_act.split(":")
      linear_beta = None if linear_beta == "none" else float(linear_beta)
      return situ_and_mul(acc_gate, acc_up, float(beta), linear_beta)
    case _:
      raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
  return pl.cdiv(x, a) * a


def _get_lhs_sublane_size(dtype: jnp.dtype, size_m: int) -> int:
  """Sublane block size the kernel processes lhs rows in."""
  size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(dtype)
  size_lhs_sublane = min(size_lhs_sublane, size_m)
  return size_lhs_sublane


def get_packing_factor(storage_dtype: jnp.dtype, quant_dtype: jnp.dtype | None) -> int:
  """Returns the number of quantized elements packed per storage element."""
  if quant_dtype is None or quant_dtype == storage_dtype:
    return 1
  storage_bits = jax.dtypes.itemsize_bits(storage_dtype)
  quant_bits = jax.dtypes.itemsize_bits(quant_dtype)
  packing_factor, remainder = divmod(storage_bits, quant_bits)
  if remainder != 0:
    raise ValueError(f"Storage dtype {storage_dtype} is not divisible by " f"quant dtype {quant_dtype}")
  return packing_factor


# Define data classes.


class RhsRef(abc.ABC):
  """Abstract class that defines interfaces for rhs values."""

  @abc.abstractmethod
  def get_weight(self) -> jax.Array:
    ...

  @abc.abstractmethod
  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    """Returns scale array, optionally replicated across sublanes."""
    ...

  @abc.abstractmethod
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

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    assert self.scale is not None
    if replicate_size is not None:
      # Perform zero-stride load for efficient broadcasting across sublanes.
      return self.scale[:, pl.ds(0, replicate_size, 0), :]
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
    return interleave_lane(w_gate, w_up)

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    s_gate = self.gate.get_scale(replicate_size)
    s_up = self.up.get_scale(replicate_size)
    return interleave_lane(s_gate, s_up)

  def get_bias(self) -> jax.Array:
    b_gate = self.gate.get_bias()
    b_up = self.up.get_bias()
    return interleave_lane(b_gate, b_up)


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
  glu_coeffs: Any | None = None

  def get_value(self) -> jax.Array:
    return self.value[...]

  def get_scale(self) -> jax.Array:
    assert self.scale is not None
    return self.scale[...]

  def get_glu_coeffs(self) -> jax.Array:
    assert self.glu_coeffs is not None
    return self.glu_coeffs[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OutRef:
  """Bundles the main GMM output ref and optional fused GLU activation ref."""

  out: Any
  act: Any | None = None


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
  bucket_base: int


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
  """Configuration for an LHS or RHS input operand."""

  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  dtype: jnp.dtype
  has_bias: bool = False
  # Whether a scale array accompanies this input. The *direction* is inferred
  # from the dtype relationship: when the input already arrives quantized
  # (dtype == quant_dtype) the scale dequantizes it (rhs); when it arrives
  # unquantized (dtype != quant_dtype) the scale quantizes it online (lhs).
  has_scale: bool = False
  is_transposed: bool = False
  # lhs only: the operand is stored as [size_m, size_k // 128, 128] (one
  # contiguous [D0, 128] slab per row) instead of [size_m, size_k]. The kernel
  # DMAs 4D blocks and relayouts them to [tile_m, tile_k] inside VMEM.
  is_3d: bool = False

  @property
  def should_unpack(self) -> bool:
    """True if rhs weights are sub-byte (e.g.

    INT4) packed in INT32/UINT32 carriers.
    """
    return get_packing_factor(self.dtype, self.quant_dtype) > 1

  @property
  def should_use_external_scale(self) -> bool:
    # A scale is present but the input is not yet quantized
    # (dtype != quant_dtype). The kernel uses it to quantize the input online
    # and multiply the result by the scale after. This differs from an already
    # quantized input (dtype == quant_dtype), whose scale only dequantizes after
    # the matmul.
    return self.has_scale and self.quant_dtype is not None and self.dtype != self.quant_dtype

  @property
  def should_dequantize_before_matmul(self) -> bool:
    """Dequantize rhs before matmul if block size limits MXU utilization."""
    if not self.has_scale:
      return False
    assert self.quant_block_size is not None
    mxu_size = pltpu.get_tpu_info().mxu_column_size
    return self.quant_block_size < mxu_size

  @property
  def should_dequantize_after_matmul(self) -> bool:
    return self.has_scale and not self.should_dequantize_before_matmul

  @property
  def should_quantize(self) -> bool:
    if self.quant_dtype is None:
      return False
    return self.quant_dtype != self.dtype


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
  """Aggregated configuration for the GMM/TGMM Pallas kernel."""

  tiles: TileSizes
  dims: Dimensions
  lhs_cfgs: InputConfigs
  rhs_cfgs: InputConfigs
  out_dtype: jnp.dtype
  acc_dtype: jnp.dtype
  zero_init: bool
  fuse_act: str | None
  transpose_rhs: bool = False
  disable_multi_core_mode: bool = False
  # Output is stored as [size_m, out_size_n // 128, 128] (one contiguous slab
  # per row); tiles are relayouted from [tile_m, tile_n] inside VMEM.
  out_is_3d: bool = False
  # The rhs weight reaches the kernel as a 2D [size_group * rows, cols] view of
  # the [size_group, rows, cols] array (see `gmm_v2`'s `flat_rhs`).
  flat_rhs: bool = False
  # A per-tensor f32 scale (SMEM) multiplies the accumulator on the last k step
  # (see `gmm_v2`'s `out_scale`).
  has_out_scale: bool = False
  # Fused dual-output GLU epilogue: writes both gate_out (to `out`) and
  # `(silu(g0) * g1 * c)` (optionally fp8-quantized) to `act`.
  has_glu_coeffs: bool = False
  glu_out_dtype: jnp.dtype | None = None
  has_glu_out_scale: bool = False

  @property
  def num_quant_blocks_per_tile_k(self) -> int:
    return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)  # pyrefly: ignore[no-matching-overload]

  @property
  def out_size_n(self) -> int:
    if self.fuse_act is None:
      return self.dims.size_n
    else:
      return self.dims.size_n // 2

  @property
  def rhs_row_blocks_per_group(self) -> int:
    """Row blocks per group in the flat [size_group * rows, cols] rhs view."""
    if self.transpose_rhs:
      return self.dims.size_n // self.tiles.tile_n
    return self.dims.size_k // self.tiles.tile_k

  @property
  def lhs_is_3d(self) -> bool:
    return self.lhs_cfgs.is_3d

  @property
  def lhs_tile_d0(self) -> int:
    """Second-minor block size of a 3D lhs tile ([.., tile_d0, 128])."""
    assert self.lhs_is_3d
    return self.tiles.tile_k // pltpu.get_tpu_info().num_lanes

  @property
  def out_tile_d0(self) -> int:
    """Second-minor block size of a 3D out tile ([.., tile_d0, 128])."""
    assert self.out_is_3d
    return self.tiles.tile_n // pltpu.get_tpu_info().num_lanes


# (dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act, *, out_is_3d=False)
# -> TileSizes. Declared with `...` so the optional `out_is_3d` keyword type
# checks.
TileFn = Callable[..., TileSizes]


class IndexMaps:
  """Index maps for GMM kernel."""

  def __init__(
      self,
      metadata_ref: MetadataRef,
      cfgs: GmmConfigs,
      out_offset_ref: jax.Array | None = None,
  ):
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs
    self.out_offset_ref = out_offset_ref

  def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    """Computes block indices for LHS."""
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start

    if self.cfgs.lhs_is_3d:
      # Block is [rows, sublane, tile_d0, 128]: the K axis is blocked along
      # D0 (tile_d0 = tile_k // 128) and the 128-lane axis is always whole.
      return (pl.ds(row_start, row_size), 0, k_id, 0)
    return (pl.ds(row_start, row_size), 0, k_id)

  def lhs_scale_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    # Per-tensor scale: a single [1, 1] value shared across every tile, so the
    # block always reads index 0. Extension point: when the scale is per-channel
    # or sub-channel, tile the row axis like `lhs_index_map` (using gm_id) and
    # index the K-block axis from `k_id`.
    del gm_id, k_id
    return (0, 0)

  def glu_coeffs_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    del n_id, k_id
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    return (pl.ds(row_start, row_size), 0, 0)

  def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    row_id, col_id = (n_id, k_id) if self.cfgs.transpose_rhs else (k_id, n_id)
    if self.cfgs.flat_rhs:
      # Group g owns block rows [g * blocks_per_group, (g + 1) *
      # blocks_per_group) of the [size_group * rows, cols] view.
      return (group_id * self.cfgs.rhs_row_blocks_per_group + row_id, col_id)
    return (group_id, row_id, col_id)

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
    """Computes block indices for the output tensor."""
    is_last_gm = gm_id == (pl.num_programs(1) - 1)
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
    last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
    row_size = row_end - row_start
    if self.out_offset_ref is not None:
      # Rows go to out[out_offset + m]; out_offset is sublane aligned.
      row_start += self.out_offset_ref[0] // self.cfgs.dims.size_lhs_sublane

    if self.cfgs.out_is_3d:
      return (pl.ds(row_start, row_size), 0, n_id, 0)
    return (pl.ds(row_start, row_size), 0, n_id)

  def act_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    """Computes block indices for the fused GLU activation output tensor."""
    is_last_gm = gm_id == (pl.num_programs(1) - 1)
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
    last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
    row_size = row_end - row_start
    return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef,
    cfgs: GmmConfigs,
    out_offset_ref: jax.Array | None = None,
) -> Tuple[Tuple[LhsRef, WeightsRef], OutRef]:
  """Generates block specs for the given lhs, rhs, and out refs."""

  index_map = IndexMaps(metadata_ref, cfgs, out_offset_ref)
  bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

  if cfgs.lhs_is_3d:
    lhs_value_spec = pl.BlockSpec(
        (
            bounded_slice_gm,
            cfgs.dims.size_lhs_sublane,
            cfgs.lhs_tile_d0,
            pltpu.get_tpu_info().num_lanes,
        ),
        index_map.lhs_index_map,
    )
  else:
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
  glu_coeffs_spec = None
  if cfgs.has_glu_coeffs:
    glu_coeffs_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, 1),
        index_map.glu_coeffs_index_map,
    )
  lhs_block_spec = LhsRef(value=lhs_value_spec, scale=lhs_scale_spec, glu_coeffs=glu_coeffs_spec)

  if cfgs.transpose_rhs:
    rhs_tile = (cfgs.tiles.tile_n, cfgs.tiles.tile_k)
  else:
    rhs_tile = (cfgs.tiles.tile_k, cfgs.tiles.tile_n)
  rhs_weight_spec = pl.BlockSpec(
      rhs_tile if cfgs.flat_rhs else (None, *rhs_tile),
      index_map.rhs_weight_index_map,
      pipeline_mode=pl.Buffered(buffer_count=3),
  )
  rhs_scale_block_spec = rhs_bias_block_spec = None
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

  rhs_block_spec = WeightsRef(
      weight=rhs_weight_spec,
      scale=rhs_scale_block_spec,
      bias=rhs_bias_block_spec,
  )

  if cfgs.out_is_3d:
    out_block_spec = pl.BlockSpec(
        (
            bounded_slice_gm,
            cfgs.dims.size_lhs_sublane,
            cfgs.out_tile_d0,
            pltpu.get_tpu_info().num_lanes,
        ),
        index_map.out_index_map,
    )
  else:
    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )

  act_block_spec = None
  if cfgs.has_glu_coeffs:
    act_block_spec = pl.BlockSpec(
        (
            bounded_slice_gm,
            cfgs.dims.size_lhs_sublane,
            cfgs.tiles.tile_n // 2,
        ),
        index_map.act_index_map,
    )

  return (lhs_block_spec, rhs_block_spec), OutRef(out=out_block_spec, act=act_block_spec)


# Define kernels.


def inner_kernel(
    # In
    tiled_lhs_ref: LhsRef,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: OutRef,
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
    out_scale_ref: jax.Array | None = None,  # f32[1] in SMEM
    glu_out_scale_ref: jax.Array | None = None,  # f32[1] in SMEM
    partial_act_ref: jax.Array | None = None,  # [size_lhs_sublane, tile_n // 2]
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
    out_scale_ref: Optional per-tensor scale the accumulator is multiplied by on
      the last k step.
    glu_out_scale_ref: Optional per-tensor scale for quantizing the fused GLU
      activation output.
    partial_act_ref: Scratch buffer for sublane-boundary partial rows of the
      fused GLU activation output.
  """

  gm_id = pl.program_id(1)
  m_start = metadata_ref.gm_id_to_m_offset[gm_id]
  m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
  m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

  m_start_local = m_start - m_offset
  m_end_local = m_end - m_offset

  def _matmul(is_first_k_step: bool, is_last_k_step: bool, bucket_m: int):
    tpu_info = pltpu.get_tpu_info()
    mxu_size = tpu_info.mxu_column_size

    # Step 1: Input pre-processing.
    if cfgs.lhs_is_3d:
      # Block is [tile_m // sublane, sublane, tile_d0, 128]; relayout the first
      # bucket_m rows into [bucket_m, tile_k] with tokens on sublanes.
      tiled_lhs = relayout.load_3d_as_2d(tiled_lhs_ref.value, bucket_m, cfgs.lhs_tile_d0)
    else:
      tiled_lhs = tiled_lhs_ref.get_value().reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
    tiled_rhs = tiled_rhs_ref.get_weight()
    if cfgs.transpose_rhs:
      tiled_rhs = jnp.transpose(tiled_rhs)

    # This should only be taken in the case where we don't requantize
    # the scales and thus we need to dequantize inside VMEM to avoid small
    # contracting dimensions
    rhs_tile_n = tiled_rhs.shape[1]
    rhs_qbs = cfgs.rhs_cfgs.quant_block_size
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
      if cfgs.transpose_rhs:
        raise NotImplementedError("should_dequantize_before_matmul is not supported with" " transpose_rhs.")
      tiled_rhs_scale = tiled_rhs_ref.get_scale(replicate_size=rhs_qbs).astype(cfgs.lhs_cfgs.dtype)
      num_blocks = cfgs.num_quant_blocks_per_tile_k
      tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(num_blocks, rhs_qbs, rhs_tile_n)
      tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
      tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)
      rhs_qbs = cfgs.tiles.tile_k

    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    if is_last_k_step and valid_k != 0:
      # `lhs` needs to be masked over k-axis iff `lhs` contains `jnp.nan`s
      # Otherwise, only masking `rhs` to zeros is sufficient.
      mask_lhs = lax.broadcasted_iota(jnp.int32, tiled_lhs.shape, 1) < valid_k
      tiled_lhs = jnp.where(mask_lhs, tiled_lhs, 0)

      mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
      tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

    if cfgs.lhs_cfgs.should_use_external_scale and not cfgs.rhs_cfgs.has_scale:
      assert cfgs.lhs_cfgs.quant_dtype is not None
      inv_scale = 1.0 / tiled_lhs_ref.get_scale().astype(jnp.float32)
      tiled_lhs = ops.quantize_tile(tiled_lhs, inv_scale, cfgs.lhs_cfgs.quant_dtype)

    # Step 2: Matmul.
    acc_list = []
    if not cfgs.lhs_cfgs.should_quantize or (cfgs.lhs_cfgs.should_use_external_scale and not cfgs.rhs_cfgs.has_scale):
      # Unquantized / pre-quantized / external-lhs-scale matmul path.
      for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n

        acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
        for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):  # pyrefly: ignore[bad-argument-type]
          end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)  # pyrefly: ignore[unsupported-operation]

          # dot_general (unlike jnp.matmul) accepts mixed operand dtypes, e.g.
          # an e5m2 lhs with e4m3 weights.
          block_acc = lax.dot_general(
              tiled_lhs[:, start_k:end_k],
              tiled_rhs[start_k:end_k, start_n:end_n],
              (((1,), (0,)), ((), ())),
              preferred_element_type=jnp.float32,
          ).astype(acc_ref.dtype)

          if cfgs.rhs_cfgs.should_dequantize_after_matmul:
            if cfgs.transpose_rhs:
              raise NotImplementedError("should_dequantize_after_matmul is not supported with" " transpose_rhs.")
            b_id = start_k // rhs_qbs  # pyrefly: ignore[unsupported-operation]
            rhs_scale_replicated = tiled_rhs_ref.get_scale(replicate_size=bucket_m)[b_id, :, start_n : start_n + col_size]
            block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)

          acc_n += block_acc
        acc_list.append(acc_n)
    else:
      # Quantized matmul path.
      lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
      q_block_size = cfgs.lhs_cfgs.quant_block_size

      if jnp.issubdtype(lhs_q_dtype, jnp.floating):  # pyrefly: ignore[bad-argument-type]
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
      # With a k-invariant lhs scale, sub-block results accumulate directly
      # into acc_n and the scale is applied once after the k loop.
      ext_lhs_scale_const_k = should_use_external_scale and lhs_scale is not None and lhs_scale.shape[-1] == 1

      # Without n outer loop, result of quantized matmul becomes available only
      # at the last iteration of the loop. This means [tile_m, tile_n] value
      # needs to be stored until the last iteration. By adding n outer loop,
      # result of [tile_m, mxu_size] becomes available at the end of every k
      # inner loop which can be used to pipeline subsequent VPU or VST ops with
      # MXU ops for the next [tile_m, mxu_size].
      step_k = (
          math.gcd(q_block_size, rhs_qbs)  # pyrefly: ignore[bad-argument-type]
          if cfgs.rhs_cfgs.should_dequantize_after_matmul
          else q_block_size
      )
      for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n

        acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
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
            block_lhs_q = jnp.clip(block_lhs * lhs_scale_inv, -dtype_max, dtype_max).astype(lhs_q_dtype)
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

          block_len = end_k - start_k
          # Initialize to None rather than jnp.zeros to avoid emitting an extra
          # VPU add instruction on the first sub-block in Pallas/Mosaic lowering
          block_acc = acc_n if ext_lhs_scale_const_k else None
          for sub_k in range(0, block_len, step_k):  # pyrefly: ignore[bad-argument-type]
            sub_end_k = min(block_len, sub_k + step_k)  # pyrefly: ignore[unsupported-operation]
            sub_acc = jnp.matmul(
                block_lhs_q[:, sub_k:sub_end_k],
                block_rhs[sub_k:sub_end_k, :],
                preferred_element_type=preferred_element_type,
            ).astype(acc_ref.dtype)

            # Apply rhs subchannel scale per quant block.
            if cfgs.rhs_cfgs.should_dequantize_after_matmul:
              if cfgs.transpose_rhs:
                raise NotImplementedError("should_dequantize_after_matmul is not supported with" " transpose_rhs.")
              b_id = (start_k + sub_k) // rhs_qbs  # pyrefly: ignore[unsupported-operation]
              rhs_scale_replicated = tiled_rhs_ref.get_scale(replicate_size=bucket_m)[
                  b_id, :, start_n : start_n + col_size
              ]
              sub_acc *= rhs_scale_replicated.astype(acc_ref.dtype)

            block_acc = sub_acc if block_acc is None else block_acc + sub_acc

          assert block_acc is not None
          if ext_lhs_scale_const_k:
            acc_n = block_acc
          else:
            block_acc *= block_scale.astype(acc_ref.dtype)
            acc_n += block_acc
        if ext_lhs_scale_const_k:
          assert lhs_scale is not None
          acc_n *= lhs_scale
        acc_list.append(acc_n)

    acc = jnp.concatenate(acc_list, axis=1)

    # Step 3: Output post-processing.
    if not is_first_k_step:
      acc += acc_ref[:bucket_m]
    acc_m = acc.shape[0]

    if is_last_k_step:
      if out_scale_ref is not None:
        acc *= out_scale_ref[0].astype(acc.dtype)
      if cfgs.rhs_cfgs.has_bias:
        tiled_rhs_bias = tiled_rhs_ref.get_bias()
        acc += tiled_rhs_bias.astype(acc.dtype)

      acc = apply_act_fn(acc, cfgs.fuse_act)

      # Mask out rows that does not belong to the current group.
      iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
      mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
      acc_masked = jnp.where(mask, acc, 0)

      # Write the final output to the output ref.
      out_val_ref = tiled_out_ref.out
      acc_out = acc_masked.astype(out_val_ref.dtype)
      if cfgs.out_is_3d:
        # Block is [tile_m // sublane, sublane, tile_d0, 128]; scatter the
        # [acc_m, tile_n] rows back into per-token [tile_d0, 128] slabs.
        relayout.store_2d_as_3d(
            out_val_ref,
            acc_out,
            acc_m,
            cfgs.out_tile_d0,
        )
      else:
        tiled_out_2d_ref = out_val_ref.reshape(-1, cfgs.tiles.tile_n)
        tiled_out_2d_ref[:acc_m] = acc_out

      # If this is the first tile for grid[n_id, :, :], we initialize the
      # partial out to zeros. Otherwise, partial out from last tile of
      # grid[n_id-1, :, :] can be used and cause numeric issues.
      partial_out_zeros = jnp.zeros_like(partial_out_ref)

      # Accumulate the partial output from the previous step.
      out_val_ref[0] += jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])

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
          out_val_ref[last_row],
      )

      if cfgs.has_glu_coeffs:
        assert tiled_out_ref.act is not None
        assert partial_act_ref is not None
        act_val_ref = tiled_out_ref.act
        half_n = cfgs.tiles.tile_n // 2
        c_tile = jnp.where(
            mask[:, :1],
            tiled_lhs_ref.get_glu_coeffs().reshape(-1, 1)[:acc_m],
            0.0,
        )
        act_tile = ops.silu_mul_tile(acc_out[:, :half_n], acc_out[:, half_n:], c_tile, out_val_ref.dtype)
        if glu_out_scale_ref is not None:
          inv_scale = 1.0 / glu_out_scale_ref[0].astype(jnp.float32)
          act_out = ops.quantize_tile(act_tile, inv_scale, act_val_ref.dtype)
        else:
          act_out = act_tile.astype(act_val_ref.dtype)
        tiled_act_2d_ref = act_val_ref.reshape(-1, half_n)
        tiled_act_2d_ref[:acc_m] = act_out

        partial_act_zeros = jnp.zeros_like(partial_act_ref)
        prev_partial_act = jnp.where(gm_id == 0, partial_act_zeros, partial_act_ref[...])
        sublane_iota = lax.broadcasted_iota(jnp.int32, partial_act_ref.shape, 0)
        act_val_ref[0] = jnp.where(
            sublane_iota < m_start_local,
            prev_partial_act,
            act_val_ref[0],
        )
        partial_act_ref[...] = jnp.where(
            m_end_local % cfgs.dims.size_lhs_sublane == 0,
            partial_act_zeros,
            act_val_ref[last_row],
        )
    else:
      acc_ref[:acc_m] = acc

  def run_matmul_step(bucket_idx: int):
    bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)

    @jax.named_scope(f"bm{bucket_m}_first_last")
    def matmul_first_last():
      _matmul(is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m)

    @jax.named_scope(f"bm{bucket_m}_first")
    def matmul_first():
      _matmul(is_first_k_step=True, is_last_k_step=False, bucket_m=bucket_m)

    @jax.named_scope(f"bm{bucket_m}_mid")
    def matmul_mid():
      _matmul(is_first_k_step=False, is_last_k_step=False, bucket_m=bucket_m)

    @jax.named_scope(f"bm{bucket_m}_last")
    def matmul_last():
      _matmul(is_first_k_step=False, is_last_k_step=True, bucket_m=bucket_m)

    num_k = pl.num_programs(2)
    k_id = pl.program_id(2)
    is_first_k_step = k_id == 0
    is_last_k_step = k_id == (num_k - 1)

    lax.cond(
        is_first_k_step,
        lambda: lax.cond(is_last_k_step, matmul_first_last, matmul_first),
        lambda: lax.cond(is_last_k_step, matmul_last, matmul_mid),
    )

  branches = []
  for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base):
    branches.append(functools.partial(run_matmul_step, bucket_idx=bucket_idx))
  bucket_idx = m_end_local // cfgs.tiles.bucket_base
  lax.switch(bucket_idx, branches)


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
    out_ref: jax.Array,  # [size_m, size_n] or [size_m, D0, 128]
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes] or [tile_zero_m, D0, 128]
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

  out_is_3d = out_ref.ndim == 3
  if out_is_3d:
    # [size_m, D0, 128] -> [size_m // sublane, sublane, D0, 128]. Each DMA
    # zeroes whole [D0, 128] slabs; slicing single D0 rows would not be
    # HBM-tile aligned.
    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, *zero_ref.shape[1:])
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, *out_ref.shape[1:])
  else:
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

    if out_is_3d:
      pltpu.make_async_copy(
          src_ref=zero_dma.at[pl.ds(0, dma_size)],
          dst_ref=out_dma.at[pl.ds(dma_start, dma_size)],
          sem=semaphore_ref.at[0],
      ).start(priority=1)
    else:
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
    out_ref: jax.Array,  # [size_m, size_n] or [size_m, D0, 128]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
  out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, *out_ref.shape[1:])
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
    # [size_group, size_k, size_n] (N and K swapped when `cfgs.transpose_rhs`).
    rhs_ref: WeightsRef,
    out_scale_ref: jax.Array | None,  # f32[1] in SMEM
    glu_out_scale_ref: jax.Array | None,  # f32[1] in SMEM
    # Out
    out_ref: OutRef,  # out: [size_m, size_n], act: [size_m, size_n // 2] | None
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    partial_act_ref: jax.Array | None,  # [size_lhs_sublane, tile_n // 2]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array | None,  # [1]
    *,
    cfgs: GmmConfigs,
    out_offset_ref: jax.Array | None = None,  # int32[1]
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
    out_scale_ref: Optional per-tensor scale of the output (see `gmm_v2`).
    glu_out_scale_ref: Optional per-tensor scale of the fused GLU activation.
    out_ref: Reference to the out bundle.
    partial_out_ref: Reference to the partial output.
    partial_act_ref: Optional reference to the partial GLU activation output.
    acc_ref: Reference to the accumulator.
    metadata_ref: Reference to the metadata.
    zero_ref: Scratch memory for storing zero values used in initialization.
    semaphore_ref: Semaphore for zero initialization DMAs.
    cfgs: GmmConfigs.
    out_offset_ref: Optional sublane-aligned row of `out_ref` that lhs row 0 is
      written to.
  """
  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

  if cfgs.rhs_cfgs.should_unpack:
    rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
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
        out_ref.out,
        zero_ref,  # pyrefly: ignore[bad-argument-type]
        semaphore_ref,  # pyrefly: ignore[bad-argument-type]
        metadata_ref,
        num_gm,
        dims=cfgs.dims,
    )

  (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs, out_offset_ref)

  if cfgs.fuse_act is not None:
    rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
    rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)  # pyrefly: ignore[bad-assignment]

    rhs_spec = FusedWeightsRef(
        gate=rhs_spec,
        up=rhs_spec,
    )

  # Partition output tiles across TCs in MegaCore mode over both parallel
  # dimensions.
  # TODO(b/549337409): Revert temporary fallback to unblock vmap on older JAX.
  # DO NOT EDIT THIS BLOCK: This path is a temporary fallback to unblock
  # until Tokamax updates its JAX version.
  core_axis_name = None if cfgs.disable_multi_core_mode else "core"
  dimension_semantics = None if cfgs.disable_multi_core_mode else (pltpu.PARALLEL, pltpu.ARBITRARY, pltpu.ARBITRARY)
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(
          inner_kernel,
          cfgs=cfgs,
          out_scale_ref=out_scale_ref,
          glu_out_scale_ref=glu_out_scale_ref,
          partial_act_ref=partial_act_ref,
      ),
      grid=(num_n, num_gm, num_k),
      in_specs=(lhs_spec, rhs_spec),
      out_specs=out_spec,
      core_axis_name=core_axis_name,
      dimension_semantics=dimension_semantics,
  )

  # Bounded slice requires second last dim to be aligned to the sublane size.
  # rhs_ref uses static tiling thus reshape is not needed. The lhs quant scale
  # (when present) is small and statically tiled, so it is passed through as-is.
  if cfgs.lhs_is_3d:
    # [size_m, D0, 128] -> [size_m // sublane, sublane, D0, 128].
    lhs_value_in = lhs_ref.value.reshape(-1, cfgs.dims.size_lhs_sublane, *lhs_ref.value.shape[-2:])
  else:
    lhs_value_in = lhs_ref.value.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.value.shape[-1])
  glu_coeffs_in = None
  if lhs_ref.glu_coeffs is not None:
    glu_coeffs_in = lhs_ref.glu_coeffs.reshape(-1, cfgs.dims.size_lhs_sublane, 1)
  lhs_in = LhsRef(value=lhs_value_in, scale=lhs_ref.scale, glu_coeffs=glu_coeffs_in)
  # Works for both [size_m, size_n] and the 3D [size_m, D0, 128] output.
  out_val_in = out_ref.out.reshape(-1, cfgs.dims.size_lhs_sublane, *out_ref.out.shape[1:])
  act_val_in = None
  if out_ref.act is not None:
    act_val_in = out_ref.act.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.act.shape[-1])
  out_in = OutRef(out=out_val_in, act=act_val_in)
  scratches = [partial_out_ref, acc_ref, metadata_ref]
  pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)

  if cfgs.zero_init:
    zero_out_end(
        out_ref.out, semaphore_ref, zero_size, dims=cfgs.dims
    )  # pyrefly: ignore[bad-argument-type, unbound-name]


def _kernel_main_into_out(
    lhs_group_sizes_ref: jax.Array,
    group_offset_ref: jax.Array,
    out_offset_ref: jax.Array,
    lhs_ref: LhsRef,
    rhs_ref: WeightsRef,
    out_scale_ref: jax.Array | None,
    glu_out_scale_ref: jax.Array | None,
    aliased_out_ref: jax.Array,
    out_ref: OutRef,
    *scratch_refs,
    cfgs: GmmConfigs,
):
  """`kernel_main` writing into an existing `out` aliased with the output."""
  del aliased_out_ref  # Same buffer as `out_ref.out`.
  kernel_main(
      lhs_group_sizes_ref,
      group_offset_ref,
      lhs_ref,
      rhs_ref,
      out_scale_ref,
      glu_out_scale_ref,
      out_ref,
      *scratch_refs,
      cfgs=cfgs,
      out_offset_ref=out_offset_ref,
  )


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
    *,
    out_is_3d: bool = False,
    flat_rhs: bool = False,
    has_glu_coeffs: bool = False,
) -> TileSizes:
  """Calculate optimal tile sizes for GMM kernel."""

  lhs_dtype = lhs_cfgs.dtype
  rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype

  lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
  rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

  # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
  # safe to use for most scenarios. But if lower bitwidth is used, we need
  # to tweak tile_m to account for using faster hardware unit.
  # TODO(kyuyeunk): Account for different TPU hardware specs.
  bf16_bf16_tile_m = 256 if rhs_cfgs.is_transposed else 128
  rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
  tile_m = bf16_bf16_tile_m // rhs_mod
  if lhs_cfgs.should_quantize:
    tile_m *= 2
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
        tk % rhs_cfgs.quant_block_size != 0  # pyrefly: ignore[unsupported-operation]
        and rhs_cfgs.quant_block_size % tk != 0  # pyrefly: ignore[unsupported-operation]
    ):
      return False
    return True

  # Initialize tile_k and tile_n to their maximum valid values.
  num_k_tiles = num_n_tiles = 1
  num_lanes = pltpu.get_tpu_info().num_lanes

  # A 3D lhs / out is relayouted with sublane-strided loads / stores, which
  # requires tile_d0 = tile // num_lanes to be a multiple of 8 (see
  # relayout.py), i.e. the tile must be a multiple of 1024 -- or cover the
  # whole (small) dim, in which case the full dim is the only legal tile.
  def _align_3d(size: int) -> int:
    return 8 * num_lanes if size >= 8 * num_lanes else size

  k_align = _align_3d(dims.size_k) if lhs_cfgs.is_3d else num_lanes
  n_align = _align_3d(size_n_per_rhs) if out_is_3d else num_lanes
  tile_k = align_to(dims.size_k, k_align)
  tile_n = align_to(size_n_per_rhs, n_align)
  sublane_tiling = pltpu.get_tpu_info().get_sublane_tiling(lhs_dtype)

  def _slab_padded(tile: int) -> int:
    # VMEM tiles the [tile_d0, 128] slab per row, so tile_d0 pads up to the
    # sublane tiling (e.g. 56 -> 64 for bf16).
    return align_to(tile // num_lanes, sublane_tiling) * num_lanes

  def _gmm_vmem_estimate(tm: int, tn: int, tk: int) -> int:
    # 1. LHS tile (double-buffered HBM load)
    lhs_tile_bytes = lhs_bits // 8
    tk_vmem = _slab_padded(tk) if lhs_cfgs.is_3d else tk
    lhs_vmem = 2 * tm * tk_vmem * lhs_tile_bytes
    if lhs_cfgs.is_3d or out_is_3d:
      # Scratch the estimate does not model: the zero-init buffer
      # (target_zero_ref_bytes = 2 MiB) and the relayout's temporaries. The 2D
      # path fits without this only because of slack; with the padded 3D
      # tiles the compiler overflowed by <1 MiB, so account for it here.
      lhs_vmem += 2 * 1024 * 1024
    # If LHS is quantized on-the-fly, we need an extra single-buffered cast
    # buffer in VMEM.
    if lhs_cfgs.should_quantize:
      lhs_quant_bits = jax.dtypes.itemsize_bits(lhs_cfgs.quant_dtype)  # pyrefly: ignore[bad-argument-type]
      lhs_vmem += tm * tk * (lhs_quant_bits // 8)

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
    acc_vmem = tm * acc_cols * acc_dtype_bytes

    # 4. Output tile (double-buffered)
    out_dtype_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    tn_vmem = _slab_padded(tn) if out_is_3d else tn
    out_vmem = 2 * tm * tn_vmem * out_dtype_bytes
    if has_glu_coeffs:
      out_vmem += 2 * tm * (tn // 2) * 2 + 2 * tm * num_lanes * 2

    return lhs_vmem + rhs_vmem + acc_vmem + out_vmem

  # Multiple k tiles will introduce accumulation overhead. Thus, we first try
  # to fit the tensors into vmem by only adjusting tile_n.

  def _is_tile_k_valid(tk: int) -> bool:
    if not _is_tile_k_quant_block_compatible(tk):
      return False
    if flat_rhs and not rhs_cfgs.is_transposed and dims.size_k % tk != 0:
      return False
    return True

  def _is_tile_n_valid(tn: int) -> bool:
    if flat_rhs and rhs_cfgs.is_transposed and size_n_per_rhs % tn != 0:
      return False
    return True

  def _shrink_tile_k():
    nonlocal tile_k, num_k_tiles
    # Decrease tile_k until total memory fits in vmem limit and tile_k is valid.
    while (
        _gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes or not _is_tile_k_valid(tile_k)
    ) and tile_k > k_align:
      num_k_tiles += 1
      tile_k = align_to(dims.size_k, num_k_tiles * k_align) // num_k_tiles

  # Decrease tile_n until total memory fits in vmem limit. A 3D output cannot
  # go below n_align (1024), and glu_coeffs requires the full size_n.
  tile_n_floor = dims.size_n if has_glu_coeffs else max(tile_n_limit, n_align)
  while (
      _gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes or not _is_tile_n_valid(tile_n)
  ) and tile_n > tile_n_floor:
    num_n_tiles += 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * n_align) // num_n_tiles

  # If decreasing tile_n is no longer possible, we decrease tile_k instead.
  if tile_n < tile_n_limit:
    num_n_tiles -= 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * n_align) // num_n_tiles
    _shrink_tile_k()
  elif (out_is_3d or has_glu_coeffs) and _gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes:
    # tile_n bottomed out at its floor (1024 for out_is_3d, size_n for
    # has_glu_coeffs) but still does not fit.
    _shrink_tile_k()

  if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None and not _is_tile_k_quant_block_compatible(tile_k):
    tile_k = rhs_cfgs.quant_block_size

  if tile_n == 0 or tile_k == 0:
    final_estimate = _gmm_vmem_estimate(tile_m, tile_n, tile_k)
    raise ValueError(
        f"Could not find valid tile sizes for {dims=} and" f" {final_estimate=} (limit: {vmem_limit_bytes})."
    )

  # TODO(alynie, kyuyeunk): max number of bucket was chosen empirically.
  # Revisit the value to be based on number of instruction memory size.
  max_num_buckets = 4
  bucket_base = tile_m
  for _ in range(1, max_num_buckets):
    new_tile_m = tile_m + bucket_base
    if new_tile_m > dims.size_m:
      break
    if _gmm_vmem_estimate(new_tile_m, tile_n, tile_k) > vmem_limit_bytes:
      break
    tile_m = new_tile_m

  return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base)


def is_fp8(dtype: jax.typing.DTypeLike) -> bool:
  return jnp.dtype(dtype) in (
      jnp.dtype(jnp.float8_e4m3fn),
      jnp.dtype(jnp.float8_e5m2),
  )


def is_manually_cast_matmul_dtype_combo(lhs_dtype: jax.typing.DTypeLike, rhs_dtype: jax.typing.DTypeLike) -> bool:
  """Some dtype combos are not implicitly supported/promoted by JAX but we support them in the kernel through explicit casting."""
  manual_cast_dtype_combo = {
      # (lhs_dtype, rhs_dtype)
      (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.int4)),
      (jnp.dtype(jnp.float8_e5m2), jnp.dtype(jnp.int4)),
      (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.bfloat16)),
      (jnp.dtype(jnp.float8_e5m2), jnp.dtype(jnp.bfloat16)),
  }
  return (jnp.dtype(lhs_dtype), jnp.dtype(rhs_dtype)) in manual_cast_dtype_combo


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
    maybe_quantize_lhs: bool = True,
    lhs_scale: jax.Array | None = None,
    transpose_rhs: bool = False,
    lhs_quant_dtype: jnp.dtype | None = None,
    packing_factor: int = 1,
) -> Dimensions:
  """Validates the inputs for the GMM kernel."""

  size_m = lhs.shape[0]
  if transpose_rhs:
    size_group, size_n, size_k = rhs.shape
  else:
    size_group, size_k, size_n = rhs.shape
  size_k *= packing_factor
  size_lhs_group = group_sizes.shape[0]

  assert size_group <= size_lhs_group
  if lhs.ndim == 3:
    num_lanes = pltpu.get_tpu_info().num_lanes
    if lhs.dtype != jnp.bfloat16 and not is_fp8(lhs.dtype):
      raise ValueError(
          "3D lhs [size_m, size_k // 128, 128] is only supported for bfloat16"
          " and fp8 (the in-VMEM relayout packs 16 / 8-bit rows into words);"
          f" got {lhs.dtype}."
      )
    if lhs.shape != (size_m, size_k // num_lanes, num_lanes):
      raise ValueError(
          f"3D lhs must have shape ({size_m}, {size_k} // {num_lanes},"
          f" {num_lanes}) to match rhs contracting dim {size_k}; got"
          f" {lhs.shape}."
      )
    if size_k % num_lanes != 0:
      raise ValueError(f"3D lhs requires {size_k=} % {num_lanes} == 0.")
  else:
    assert lhs.shape == (size_m, size_k)
  if transpose_rhs:
    if rhs_scale is not None:
      raise NotImplementedError("transpose_rhs does not support quantized RHS (rhs_scale is not" " None).")
    if rhs_bias is not None:
      raise NotImplementedError("transpose_rhs does not support RHS bias (rhs_bias is not None).")
    if fuse_act is not None:
      raise NotImplementedError("transpose_rhs does not support fuse_act.")
    assert rhs.shape == (size_group, size_n, size_k // packing_factor)
  else:
    assert rhs.shape == (size_group, size_k // packing_factor, size_n)
  if rhs_bias is not None:
    assert rhs_bias.shape == (size_group, 1, size_n)
  if rhs_scale is not None:
    num_quant_blocks = rhs_scale.shape[1]
    assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n), (
        f"rhs_scale shape {rhs_scale.shape}. Expecting ({size_group}," f" {num_quant_blocks}, 1, {size_n})"
    )
    assert size_k % num_quant_blocks == 0

  if lhs_scale is not None:
    assert maybe_quantize_lhs, "lhs_scale requires maybe_quantize_lhs=True."
    # Only per-tensor scales are supported for now. The current implementation
    # generalizes to per-channel [M, 1] and
    # sub-channel [M, num_k_blocks]; extend the validation and the block spec /
    # index map together when adding those.
    assert lhs_scale.shape == (1, 1), "Only per-tensor lhs_scale of shape (1, 1) is supported, got " f"{lhs_scale.shape}."

  assert group_offset.shape == (1,)

  if lhs_quant_dtype is not None:
    if not maybe_quantize_lhs:
      raise ValueError("lhs_quant_dtype cannot be set when maybe_quantize_lhs is False.")
    tpu_info = pltpu.get_tpu_info()
    if (
        not tpu_info.is_matmul_supported(lhs_quant_dtype, rhs.dtype)
        and not is_manually_cast_matmul_dtype_combo(lhs_quant_dtype, rhs.dtype)
        and not (is_fp8(lhs_quant_dtype) and is_fp8(rhs.dtype))
    ):
      raise ValueError(
          f"lhs_quant_dtype ({lhs_quant_dtype}) and rhs dtype ({rhs.dtype}) " "are not a supported combination."
      )

  size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
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
  rhs_dtype = cfgs.rhs_cfgs.quant_dtype or cfgs.rhs_cfgs.dtype

  # We use bits for rhs since it could sub-byte dtype like int4.
  rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
  fp32_bytes = jnp.dtype(jnp.float32).itemsize

  # TODO(kyuyeunk): Add compute flops for quant, dequant, and bias.
  flops = 2 * dims.size_m * dims.size_k * dims.size_n

  lhs_bytes = dims.size_m * dims.size_k * jnp.dtype(lhs_dtype).itemsize

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
  name = (
      f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )
  if cfgs.has_glu_coeffs:
    name += "-glu"
  return name


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
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
    transpose_rhs: bool = False,
    lhs_quant_dtype: jnp.dtype | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
    disable_multi_core_mode: bool = False,
    out_is_3d: bool = False,
    flat_rhs: bool = False,
    has_out_scale: bool = False,
    has_glu_coeffs: bool = False,
    glu_out_dtype: jnp.dtype | None = None,
    has_glu_out_scale: bool = False,
):
  """Fills the GMM config for the GMM kernel."""
  packing_factor = get_packing_factor(rhs.dtype, rhs_quant_dtype)

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
      transpose_rhs=transpose_rhs,
      lhs_quant_dtype=lhs_quant_dtype,
      packing_factor=packing_factor,
  )

  if rhs_scale is not None:
    has_scale = True
    rhs_quant_dtype = rhs_quant_dtype or rhs.dtype
    num_blocks = rhs_scale.shape[1]
    block_size = dims.size_k // num_blocks
  else:
    has_scale = False
    block_size = dims.size_k

  rhs_cfgs = InputConfigs(
      quant_dtype=rhs_quant_dtype,
      quant_block_size=block_size,
      dtype=rhs.dtype,
      has_bias=rhs_bias is not None,
      has_scale=has_scale,
      is_transposed=transpose_rhs,
  )

  lhs_q_dtype = None
  if maybe_quantize_lhs and rhs_cfgs.should_dequantize_after_matmul:
    if lhs_quant_dtype is not None:
      lhs_q_dtype = lhs_quant_dtype
    else:
      # Choose lhs quantization dtype based on TPU hardware support
      # only if lhs_quant_dtype is not provided.
      assert rhs_quant_dtype is not None
      is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)  # pyrefly: ignore[bad-argument-type]
      tpu_info = pltpu.get_tpu_info()
      # Check if there is hardware compute support for rhs dtype group.
      if tpu_info.fp8_ops_per_second > 0:
        # Special handling for 4-bit integer rhs as it can be converted to fp8
        # without a numeric issues. Note that this is not the case for 4-bit
        # floating rhs as conversion to int8 will cause numeric issues.
        # see is_manually_cast_matmul_dtype_combo()
        is_rhs_4bits = jax.dtypes.itemsize_bits(rhs_quant_dtype) == 4  # pyrefly: ignore[bad-argument-type]
        if is_rhs_float or is_rhs_4bits:
          lhs_q_dtype = jnp.float8_e4m3fn.dtype
      if tpu_info.int8_ops_per_second > 0:
        if not is_rhs_float:
          lhs_q_dtype = jnp.int8.dtype
  elif maybe_quantize_lhs and lhs_scale is not None and lhs_quant_dtype is not None:
    # Static per-tensor lhs quantization against an unscaled (e.g. fp8) rhs,
    # whose scale the caller folds into `out_scale`.
    lhs_q_dtype = lhs_quant_dtype

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
      # enough to minimize compute overhead of quantization. When an external
      # static scale is provided (`has_lhs_scale`), no per-block reduction is
      # needed so we use the full contracting dimension.
      quant_block_size=dims.size_k if has_lhs_scale else 512,
      dtype=lhs.dtype,
      has_scale=has_lhs_scale,
      is_3d=lhs.ndim == 3,
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
    if out_is_3d:
      tile_info = functools.partial(tile_info, out_is_3d=True)
    if flat_rhs:
      tile_info = functools.partial(tile_info, flat_rhs=True)
    if has_glu_coeffs:
      tile_info = functools.partial(tile_info, has_glu_coeffs=True)
    tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)  # pyrefly: ignore[bad-argument-type]

  num_lanes = pltpu.get_tpu_info().num_lanes
  if lhs_cfgs.is_3d:
    if tiles.tile_k % num_lanes != 0:
      raise ValueError(f"3D lhs requires tile_k % {num_lanes} == 0; got {tiles.tile_k=}.")
    # Raises a descriptive error if tile_k // 128 is neither a multiple of 8
    # nor the full D0 (small K).
    relayout.check_tile_d0(
        tiles.tile_k // num_lanes,
        full_d0=dims.size_k // num_lanes,
        pack=relayout.packing(lhs.dtype),
    )

  if out_is_3d:
    if jnp.dtype(out_dtype) != jnp.bfloat16:
      raise ValueError(
          "3D output [size_m, out_size_n // 128, 128] is only supported for"
          f" bfloat16 (the in-VMEM relayout packs bf16 pairs); got {out_dtype}."
      )
    out_size_n = dims.size_n if fuse_act is None else dims.size_n // 2
    if out_size_n % num_lanes != 0:
      raise ValueError(f"3D output requires out_size_n ({out_size_n}) % {num_lanes} == 0.")
    if tiles.tile_n % num_lanes != 0:
      raise ValueError(f"3D output requires tile_n % {num_lanes} == 0; got {tiles.tile_n=}.")
    # Raises a descriptive error if tile_n // 128 is neither a multiple of 8
    # nor the full D0 (small N).
    relayout.check_tile_d0(tiles.tile_n // num_lanes, full_d0=out_size_n // num_lanes)

  if flat_rhs:
    if packing_factor > 1 or has_scale or fuse_act is not None:
      raise ValueError(
          "flat_rhs supports only unquantized, unpacked rhs without fuse_act;"
          f" got {rhs.dtype=}, {rhs_quant_dtype=}, {has_scale=}, {fuse_act=}."
      )
    # A block of the flat view must not straddle two groups, so the tile must
    # evenly split each group's rows.
    if transpose_rhs:
      rows, tile_rows = dims.size_n, tiles.tile_n
    else:
      rows, tile_rows = dims.size_k, tiles.tile_k
    if rows % tile_rows != 0:
      raise ValueError(
          f"flat_rhs requires the rhs row tile ({tile_rows}) to divide the rhs"
          f" rows per group ({rows}); {transpose_rhs=}."
      )

  if has_glu_coeffs:
    if fuse_act is not None or out_is_3d:
      raise ValueError("glu_coeffs is not supported with fuse_act or out_is_3d.")
    if tiles.tile_n != dims.size_n or dims.size_n % (2 * num_lanes) != 0:
      raise ValueError(
          f"glu_coeffs requires tile_n ({tiles.tile_n}) == size_n"
          f" ({dims.size_n}) and size_n divisible by {2 * num_lanes}."
      )

  return GmmConfigs(
      dims=dims,
      tiles=tiles,
      lhs_cfgs=lhs_cfgs,
      rhs_cfgs=rhs_cfgs,
      out_dtype=jnp.dtype(out_dtype),
      acc_dtype=jnp.dtype(acc_dtype),
      zero_init=zero_initialize,
      fuse_act=fuse_act,
      transpose_rhs=transpose_rhs,
      disable_multi_core_mode=disable_multi_core_mode,
      out_is_3d=out_is_3d,
      flat_rhs=flat_rhs,
      has_out_scale=has_out_scale,
      has_glu_coeffs=has_glu_coeffs,
      glu_out_dtype=(jnp.dtype(glu_out_dtype) if glu_out_dtype is not None else None),
      has_glu_out_scale=has_glu_out_scale,
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
        "lhs_quant_dtype",
        "rhs_quant_dtype",
        "zero_initialize",
        "fuse_act",
        "transpose_rhs",
        "disable_multi_core_mode",
        "out_is_3d",
        "flat_rhs",
        "glu_out_dtype",
    ],
    donate_argnames=["out"],
)
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n] (or [.., size_n, size_k])
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    group_offset: jax.Array | None = None,  # int32[1]
    lhs_scale: jax.Array | None = None,  # [1, 1] (per-tensor)
    out_scale: jax.Array | None = None,  # [1, 1] (per-tensor)
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    lhs_quant_dtype: jnp.dtype | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    transpose_rhs: bool = False,
    disable_multi_core_mode: bool = False,
    out_is_3d: bool = False,
    flat_rhs: bool = False,
    out: jax.Array | None = None,  # [out_rows, size_n]
    out_offset: jax.Array | int = 0,  # int32 scalar
    glu_coeffs: jax.Array | None = None,  # [size_m, 1]
    glu_out_dtype: jnp.dtype | None = None,
    glu_out_scale: jax.Array | None = None,  # [1, 1]
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """GMM kernel implemented with emit_pipeline.

  Dynamically calculate offset lhs/out tiles to reduce redundant computations.
  Additionally, it adjusts dma size based on number of valid rows and utilize
  triple buffering on weights to better utilize memory.

  Args:
    lhs: lhs with shape [size_m, size_k], or (bfloat16 / fp8 only) the 3D layout
      [size_m, size_k // 128, 128] where each row is stored as a contiguous
      [size_k // 128, 128] slab. The 3D layout is consumed directly (no HBM
      reshape) and relayouted to [tile_m, tile_k] inside VMEM; it requires
      tile_k to be a multiple of 1024. An fp8 (e4m3 / e5m2) lhs is multiplied as
      is (with an fp8 or bf16 rhs); fold its scale into `out_scale`.
    rhs: rhs with shape [size_group, size_k, size_n] (or [size_group, size_n,
      size_k] if transpose_rhs) if native else [size_group, size_k // 8, size_n]
      if packed.
    group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
    rhs_scale: The rhs scale of shape [size_group, num_blocks, 1, out_size].
    rhs_bias: The rhs bias of shape [size_group, 1, out_size].
    group_offset: Optional. The group offset of shape [1,].
    lhs_scale: Optional scale used to quantize the (unquantized) lhs inside the
      kernel and the result is multiplied back by `scale`. The shape encodes
      granularity; currently only per-tensor `[1, 1]` is supported. When None, a
      quantized lhs uses the default dynamic per-block absmax calibration. Only
      takes effect when maybe_quantize_lhs is True and rhs is quantized, or when
      `lhs_quant_dtype` is given (static quantization against an unscaled rhs).
    out_scale: Optional f32 per-tensor scale `[1, 1]` that the accumulator is
      multiplied by once, on the last k step (before bias / activation). Use it
      for the combined scales of pre-quantized operands; it needs `acc_dtype`
      float32 to apply to an f32 accumulator.
    lhs_quant_dtype: Optional jnp.dtype to use for quantizing lhs
    rhs_quant_dtype: Optional jnp.dtype to use for quantizing rhs
    tile_info: The tile sizes or tile function to use. When `out_is_3d` and a
      tile function is given, it is called with `out_is_3d=True`.
    vmem_limit_bytes: Optional vmem limit in bytes.
    precision: Unused. Exists for compatibility reasons.
    preferred_element_type: Optional jnp.dtype for the output matrix.
    acc_dtype: Optional jnp.dtype for the accumulator.
    maybe_quantize_lhs: Quantize lhs if set to True and rhs is quantized.
    zero_initialize: Whether to initialize unvisited output elements to zero.
    fuse_act: Activation function to fuse with GMM, None if no fusion.
    transpose_rhs: Whether to transpose rhs in-core without HBM transposition.
    out_is_3d: If True, the (bfloat16) output is written directly in the 3D
      layout [size_m, out_size_n // 128, 128] (no HBM reshape); tiles are
      relayouted from [tile_m, tile_n] inside VMEM. Requires out_size_n % 128 ==
      0 and tile_n a multiple of 1024.
    flat_rhs: If True, the kernel reads rhs through its 2D [size_group * rows,
      cols] view (a free bitcast) instead of the 3D array. XLA keeps a bitcast
      operand's HBM tiling (e.g. bf16 T(16,128)) for the kernel, while a 3D
      operand computed elsewhere (loop-carried, all-gathered) gets a relayout
      copy to the default tiling on every call. Requires an unquantized rhs
      without fuse_act, and the rhs row tile (tile_k, or tile_n if
      transpose_rhs) must divide the per-group rows.
    out: Optional existing 2D output buffer (donated). If given, lhs row m is
      written to out[out_offset + m] in place and `out` is returned; rows past
      the last group are neither written nor zeroed (`zero_initialize` is
      ignored), except that rows up to the end of the last sublane block
      (`size_lhs_sublane` rows) may be clobbered. Requires
      `disable_multi_core_mode` and a 2D output.
    out_offset: Row of `out` that lhs row 0 is written to. Must be a multiple of
      the lhs sublane tiling (16 for bfloat16).
    glu_coeffs: Optional `[size_m, 1]` routing coefficients. When provided,
      enables the fused dual-output GLU epilogue returning `(out, act)` where
      `act = silu(out[:, :N//2]) * out[:, N//2:] * glu_coeffs`.
    glu_out_dtype: Optional output dtype of `act` when `glu_coeffs` is provided
      (e.g. `jnp.float8_e4m3fn`). Defaults to `out`'s dtype.
    glu_out_scale: Optional `[1, 1]` f32 static quantization scale for `act`
      when `glu_out_dtype` is fp8.

  Returns:
    Output of shape [size_m, size_n], or [size_m, size_n // 128, 128] if
    `out_is_3d`, or the updated `out` (or `(out, act)` when `glu_coeffs` is
    provided).
  """

  del precision

  if out is not None:
    if not disable_multi_core_mode:
      raise ValueError("`out` requires disable_multi_core_mode=True (`pl.kernel` has no" " input/output aliasing).")
    if out_is_3d or out.ndim != 2:
      raise ValueError(f"`out` must be a 2D output; got {out.shape=}.")
    # Zero-filling rows outside the computed range would wipe `out`.
    zero_initialize = False

  # The kernel reshapes lhs by its sublane block, so pad the rows up to it.
  # The extra rows sit past every group, so they add no work, and they are
  # sliced back off the output below.
  size_m = lhs.shape[0]
  if glu_coeffs is not None and glu_coeffs.shape != (size_m, 1):
    raise ValueError(f"glu_coeffs must have shape ({size_m}, 1); got {glu_coeffs.shape}.")
  size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
  padded_size_m = align_to(size_m, size_lhs_sublane)
  if padded_size_m != size_m:
    lhs = jnp.pad(lhs, ((0, padded_size_m - size_m),) + ((0, 0),) * (lhs.ndim - 1))
    if glu_coeffs is not None:
      glu_coeffs = jnp.pad(glu_coeffs, ((0, padded_size_m - size_m), (0, 0)))

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
      transpose_rhs=transpose_rhs,
      lhs_quant_dtype=lhs_quant_dtype,
      rhs_quant_dtype=rhs_quant_dtype,
      disable_multi_core_mode=disable_multi_core_mode,
      out_is_3d=out_is_3d,
      flat_rhs=flat_rhs,
      has_out_scale=out_scale is not None,
      has_glu_coeffs=glu_coeffs is not None,
      glu_out_dtype=glu_out_dtype,
      has_glu_out_scale=glu_out_scale is not None,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles
  if cfgs.lhs_cfgs.should_use_external_scale and not cfgs.rhs_cfgs.has_scale:
    assert lhs_scale is not None
    out_scale = lhs_scale if out_scale is None else lhs_scale * out_scale
    cfgs = dataclasses.replace(cfgs, has_out_scale=True)
  if out_scale is not None:
    if out_scale.shape != (1, 1):
      raise ValueError(f"out_scale must be [1, 1]; got {out_scale.shape}.")
    # A scalar in SMEM, read by the kernel on the last k step.
    out_scale = out_scale.astype(jnp.float32).reshape(1)
  if glu_out_scale is not None:
    if glu_out_scale.shape != (1, 1):
      raise ValueError(f"glu_out_scale must be [1, 1]; got {glu_out_scale.shape}.")
    glu_out_scale = glu_out_scale.astype(jnp.float32).reshape(1)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  out_scale_spec = smem if out_scale is not None else None
  glu_out_scale_spec = smem if glu_out_scale is not None else None

  # Prepare block specs.
  if cfgs.lhs_cfgs.has_scale:
    assert lhs_scale is not None
    lhs_scale = lhs_scale.astype(jnp.float32)
  else:
    lhs_scale = None

  if rhs_scale is not None:
    rhs_scale = rhs_scale.astype(jnp.float32)
  if rhs_bias is not None:
    rhs_bias = rhs_bias.astype(jnp.float32)

  num_lanes = pltpu.get_tpu_info().num_lanes
  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  out_d0 = aligned_n // num_lanes

  # Initialize scratch shapes.
  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
  acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
  if cfgs.out_is_3d:
    partial_out_shape = (dims.size_lhs_sublane, cfgs.out_tile_d0, num_lanes)
  else:
    partial_out_shape = (dims.size_lhs_sublane, tiles.tile_n)
  act_dtype = cfgs.glu_out_dtype or cfgs.out_dtype
  partial_act_scratch = pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n // 2), act_dtype) if cfgs.has_glu_coeffs else None
  scratch_shapes = [
      # partial_out_ref
      pltpu.VMEM(partial_out_shape, cfgs.out_dtype),
      # partial_act_ref
      partial_act_scratch,
      # acc_ref
      pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
      # metadata_ref
      MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]

  if cfgs.zero_init:
    # TODO(kyuyeunk): Create better heuristics for determining this value.
    target_zero_ref_bytes = 2 * 1024 * 1024
    out_bytes = jnp.dtype(cfgs.out_dtype).itemsize

    if cfgs.out_is_3d:
      # In the 3D layout a row is a whole [D0, 128] slab and single D0 rows are
      # not HBM-tile aligned, so the zero buffer holds full slabs:
      # [tile_zero_m, D0, 128], with tile_zero_m a multiple of the sublane
      # block (zero_out_start reshapes it by size_lhs_sublane). VMEM pads D0
      # up to the sublane tiling, so budget with the padded size.
      sublane = dims.size_lhs_sublane
      sublane_tiling = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
      row_bytes = align_to(out_d0, sublane_tiling) * num_lanes * out_bytes
      tile_zero_m = target_zero_ref_bytes // row_bytes // sublane * sublane
      tile_zero_m = max(sublane, min(tile_zero_m, dims.size_m))
      zero_shape = (tile_zero_m, out_d0, num_lanes)
    else:
      # Zero initialization is done by tiling size_m dim where each tile
      # invokes zero initializing DMA for up-to tile_zero_m rows. This means
      # larger tile_zero_m will result in fewer number of tiles and lead to
      # smaller overhead. However, in order to invoke DMA call up-to
      # tile_zero_m rows, we need to store equivalent sized memory in VMEM
      # buffer for the duration of DMA. Storing [tile_zero_m, size_n] in
      # buffer will trigger OOM if tile_zero_m is too large. Instead, if we set
      # column size as num_lanes (which is smallest allowed column size for
      # DMA) and reuse the buffer by size_n//num_lanes times in a single tile,
      # we can significantly increase tile_zero_m without triggering OOM.
      tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
      tile_zero_m = min(tile_zero_m, dims.size_m)
      zero_shape = (tile_zero_m, num_lanes)

    scratch_shapes += [
        pltpu.VMEM(zero_shape, cfgs.out_dtype),
        pltpu.SemaphoreType.DMA((1,)),
    ]
  else:
    scratch_shapes += [None, None]

  if cfgs.out_is_3d:
    out_shape = (dims.size_m, out_d0, num_lanes)
    out_slice = (slice(None, size_m),)
  else:
    out_shape = (dims.size_m, aligned_n)
    out_slice = (slice(None, size_m), slice(None, cfgs.out_size_n))
  # Propagate the manual-axis type so the kernel works under
  # shard_map(check_vma=True) (the output inherits the lhs' manual axes).
  out_init = jax.ShapeDtypeStruct(
      out_shape,
      cfgs.out_dtype,
      manual_axis_type=jax.typeof(lhs).manual_axis_type,
  )
  act_init = None
  if cfgs.has_glu_coeffs:
    act_init = jax.ShapeDtypeStruct(
        (dims.size_m, dims.size_n // 2),
        act_dtype,
        manual_axis_type=jax.typeof(lhs).manual_axis_type,
    )
  lhs_in = LhsRef(value=lhs, scale=lhs_scale, glu_coeffs=glu_coeffs)
  if cfgs.flat_rhs:
    # Merging the leading dims is a bitcast, so the kernel operand keeps the
    # producer's HBM tiling (see `flat_rhs` above). Keep this the last op on
    # rhs before the kernel call.
    rhs = rhs.reshape(-1, rhs.shape[-1])
  rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

  if out is not None:
    if out.shape[1] != aligned_n or out.shape[0] % dims.size_lhs_sublane != 0 or out.dtype != cfgs.out_dtype:
      raise ValueError(
          f"`out` must be [k * {dims.size_lhs_sublane}, {aligned_n}] of dtype"
          f" {cfgs.out_dtype}; got {out.shape=}, {out.dtype=}."
      )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out_offset = jnp.asarray(out_offset, jnp.int32)[None]
    scalars = (group_sizes, group_offset, out_offset)
    inputs = (lhs_in, rhs_weights, out_scale, glu_out_scale, out)
    res = pl.pallas_call(
        functools.partial(_kernel_main_into_out, cfgs=cfgs),
        out_shape=OutRef(
            out=jax.ShapeDtypeStruct(
                out.shape,
                out.dtype,
                manual_axis_type=jax.typeof(out).manual_axis_type,
            ),
            act=act_init,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            in_specs=[
                LhsRef(
                    value=hbm,
                    scale=hbm if cfgs.lhs_cfgs.has_scale else None,
                    glu_coeffs=hbm if cfgs.has_glu_coeffs else None,
                ),
                WeightsRef(
                    weight=hbm,
                    scale=hbm if rhs_scale is not None else None,
                    bias=hbm if rhs_bias is not None else None,
                ),
                out_scale_spec,
                glu_out_scale_spec,
                hbm,
            ],
            out_specs=OutRef(out=hbm, act=hbm if cfgs.has_glu_coeffs else None),
            # pyrefly: ignore[bad-argument-type]
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        # `out` is the last flattened operand.
        input_output_aliases={len(jax.tree.leaves((scalars, inputs))) - 1: 0},
        name=get_scope_name(cfgs),
        cost_estimate=get_cost_estimate(cfgs),
        metadata=get_metadata(cfgs),  # pyrefly: ignore[bad-argument-type]
    )(*scalars, *inputs)
    if cfgs.has_glu_coeffs:
      assert res.act is not None
      return res.out, res.act[:size_m]
    return res.out

  # TODO(b/549337409): Revert temporary fallback to unblock vmap on older JAX.
  # DO NOT EDIT THIS BLOCK: This path is a temporary fallback to unblock
  # until Tokamax updates its JAX version.
  if cfgs.disable_multi_core_mode:
    lhs_scale_spec = None
    if cfgs.lhs_cfgs.has_scale:
      lhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    glu_coeffs_spec = None
    if cfgs.has_glu_coeffs:
      glu_coeffs_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
      rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
      rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    res = pl.pallas_call(
        functools.partial(kernel_main, cfgs=cfgs),
        out_shape=OutRef(out=out_init, act=act_init),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                LhsRef(
                    value=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=lhs_scale_spec,
                    glu_coeffs=glu_coeffs_spec,
                ),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=rhs_scale_spec,
                    bias=rhs_bias_spec,
                ),
                out_scale_spec,
                glu_out_scale_spec,
            ],
            out_specs=OutRef(
                out=pl.BlockSpec(memory_space=pltpu.HBM),
                act=(pl.BlockSpec(memory_space=pltpu.HBM) if cfgs.has_glu_coeffs else None),
            ),
            # pyrefly: ignore[bad-argument-type]
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(cfgs),
        cost_estimate=get_cost_estimate(cfgs),
        metadata=get_metadata(cfgs),  # pyrefly: ignore[bad-argument-type]
    )(
        group_sizes,
        group_offset,
        lhs_in,
        rhs_weights,
        out_scale,
        glu_out_scale,
    )
    if cfgs.has_glu_coeffs:
      assert res.act is not None
      return res.out[out_slice], res.act[:size_m]
    return res.out[out_slice]

  group_sizes = pltpu.with_memory_space_constraint(group_sizes, pltpu.SMEM)
  group_offset = pltpu.with_memory_space_constraint(group_offset, pltpu.SMEM)
  if out_scale is not None:
    out_scale = pltpu.with_memory_space_constraint(out_scale, pltpu.SMEM)
  if glu_out_scale is not None:
    glu_out_scale = pltpu.with_memory_space_constraint(glu_out_scale, pltpu.SMEM)

  # Configure per-core execution over TensorCore mesh for MegaCore scaling.
  res = pl.kernel(
      functools.partial(kernel_main, cfgs=cfgs),
      out_type=OutRef(out=out_init, act=act_init),
      mesh=pltpu.TensorCoreMesh(axis_name="core"),
      scratch_types=scratch_shapes,  # pyrefly: ignore[bad-argument-type]
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=get_scope_name(cfgs),
      cost_estimate=get_cost_estimate(cfgs),
      metadata=get_metadata(cfgs),  # pyrefly: ignore[bad-argument-type]
  )(group_sizes, group_offset, lhs_in, rhs_weights, out_scale, glu_out_scale)
  if cfgs.has_glu_coeffs:
    assert res.act is not None
    return res.out[out_slice], res.act[:size_m]
  return res.out[out_slice]


tokamax_gmmv2 = gmm_v2
gmmv2 = gmm_v2
