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

from maxtext.kernels.megablox import relayout

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


def apply_dact_fn(
    acc: jax.Array,
    gate_tile: jax.Array,
    up_tile: jax.Array,
    row_scale_tile: jax.Array | None,
    fuse_dact: str | None,
    *,
    return_dot: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Applies the backward of a gated activation (fuse_dact) to the accumulator.

  This is the epilogue of the DLHS GMM of the projection that follows a gated activation (the MoE down
  projection): `acc` holds u = dout @ rhs^T, the cotangent of the activation output, for the rows of one
  tile, and `gate_tile` / `up_tile` are the matching rows of the activation's input [gate | up]. For
  fuse_dact="swiglu", act = silu(gate) * up:

    d_gate = s * u * up * silu'(gate),   d_up = s * u * silu(gate),
    silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))),

  with s the optional per-row scale `row_scale_tile` (e.g. the routing weight). All of it runs in float32
  on the float32 accumulator: u is not rounded to the output dtype first. With return_dot it also returns
  <u, act> per row, with act = silu(gate) * up in float32: the gradient of a per-row scale applied to the
  projection's output (the routing weight).

  Args:
    acc: The accumulator, [tile_m, tile_n].
    gate_tile: The gate rows, [tile_m, tile_n].
    up_tile: The up rows, [tile_m, tile_n].
    row_scale_tile: An optional scale per row, [tile_m, 1].
    fuse_dact: The activation to differentiate. Supported: "swiglu". If None, `acc` is returned unchanged.
    return_dot: Whether to also return <u, act> per row.

  Returns:
    (float32 [d_gate | d_up] of shape [tile_m, 2 * tile_n], and the [tile_m, 1] dot or None).

  Raises:
    NotImplementedError: If an unsupported `fuse_dact` is provided.
  """
  if fuse_dact is None:
    return acc, None

  dot_tile = None
  match fuse_dact:
    case "swiglu":
      u = acc.astype(jnp.float32)
      g = gate_tile.astype(jnp.float32)
      up = up_tile.astype(jnp.float32)
      sig = jax.nn.sigmoid(g)
      silu_g = g * sig
      if return_dot:
        h = silu_g * up
        dot_tile = jnp.sum(u * h, axis=-1, keepdims=True)
      da = u * row_scale_tile.astype(jnp.float32) if row_scale_tile is not None else u
      dsilu = sig * (1.0 + g * (1.0 - sig))
      dgate = da * up * dsilu
      dup = da * silu_g
      return jnp.concatenate([dgate, dup], axis=-1), dot_tile
    case _:
      raise NotImplementedError(f"Unsupported backward activation function: {fuse_dact}")


def split_dact_gate_up(gate_up_tile: jax.Array, pre_act_tile_n: int | None) -> tuple[jax.Array, jax.Array]:
  """The gate and up halves of a [tile_m, 2 * tile_n] block of a fuse_dact epilogue's gate_up input.

  return_pre_act="concat" stores each forward n tile's accumulator [gate_j | up_j] as one block, so a
  row of the fused forward's [gate | up] output is laid out [gate_0 | up_0 | gate_1 | up_1 | ...], with
  pre_act_tile_n (the forward's tile_n, per half) columns per piece. The fuse_dact epilogue spans the whole
  row (one n tile), so its block is that row: the gate is the concatenation of the even pieces and up of the
  odd ones. Static, lane-aligned slices (pre_act_tile_n is a multiple of 128): a selection of whole vregs in
  VMEM, no extra HBM traffic. With None (or one piece per half: the [gate | up] of an unfused GMM output, or
  of a forward with one n tile per half) this is the plain split in two.

  Args:
    gate_up_tile: The block, [tile_m, 2 * tile_n].
    pre_act_tile_n: The width of the forward's pieces, or None for the contiguous [gate | up] layout.

  Returns:
    (gate, up), each [tile_m, tile_n].
  """
  tile_n = gate_up_tile.shape[-1] // 2
  if pre_act_tile_n is None or pre_act_tile_n == tile_n:
    gate, up = jnp.split(gate_up_tile, 2, axis=-1)
    return gate, up
  if pre_act_tile_n <= 0 or tile_n % pre_act_tile_n:
    raise ValueError(f"pre_act_tile_n {pre_act_tile_n} must divide the block's half width {tile_n}.")
  p = pre_act_tile_n
  pieces = range(tile_n // p)
  gate = jnp.concatenate([gate_up_tile[:, 2 * j * p : (2 * j + 1) * p] for j in pieces], axis=-1)
  up = jnp.concatenate([gate_up_tile[:, (2 * j + 1) * p : (2 * j + 2) * p] for j in pieces], axis=-1)
  return gate, up


def align_to(x, a):
  return pl.cdiv(x, a) * a


# Define data classes.


class RhsRef(ABC):
  """Abstract class that defines interfaces for rhs values."""

  @abstractmethod
  def get_weight(self, transpose_rhs: bool = False) -> jax.Array:
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

  def get_weight(self, transpose_rhs: bool = False) -> jax.Array:
    del transpose_rhs
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

  def get_weight(self, transpose_rhs: bool = False) -> jax.Array:
    w_gate = self.gate.get_weight()
    w_up = self.up.get_weight()
    concat_axis = -2 if transpose_rhs else -1
    return jnp.concatenate([w_gate, w_up], axis=concat_axis)

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
class DactInputRef:
  """The inputs of the fuse_dact epilogue, passed in the partial_sum slot (fuse_dact excludes partial_sum).

  gate_up is the [size_m, 2 * size_n] input [gate | up] of the gated activation; row_scale is None or the
  optional per-row scale, broadcast to [size_m, 128] float32 (one DMA-able lane row per row).
  """

  gate_up: Any
  row_scale: Any | None = None


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
  # The operand is stored as [size_m, size // 128, 128] (one contiguous slab per
  # row) instead of [size_m, size]. The kernel DMAs 4D blocks and relayouts them
  # to 2D tiles inside VMEM (see relayout.py).
  is_3d: bool = False

  @property
  def should_use_external_scale(self) -> bool:
    # A scale is present but the input is not yet quantized
    # (dtype != quant_dtype). The kernel uses it to quantize the input online
    # and multiply the result by the scale after. This differs from an already
    # quantized input (dtype == quant_dtype), whose scale only dequantizes after
    # the matmul.
    return self.has_scale and self.quant_dtype is not None and self.dtype != self.quant_dtype

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
  transpose_rhs: bool
  # Output is stored as [size_m, out_size_n // 128, 128] (one contiguous slab
  # per row); tiles are relayouted from [tile_m, tile_n] inside VMEM.
  out_is_3d: bool = False
  # With fuse_act: also write the two halves of the accumulator before the
  # activation (gate and up), each [size_m, out_size_n] in out_dtype, next to
  # the activation output. The backward of the activation needs them.
  # "concat": write them as one [size_m, 2 * out_size_n] output (the input of a
  # fuse_dact epilogue): each n tile's [gate_j | up_j] is one block, so the
  # layout is [gate_0 | up_0 | gate_1 | up_1 | ...] with tile_n columns per
  # piece, i.e. [gate | up] with one n tile per half.
  return_pre_act: bool | str = False
  # The backward of a gated activation (apply_dact_fn) in the epilogue: the
  # accumulator is the activation output's cotangent, dact_gate_up
  # [size_m, 2 * size_n] (read in the partial_sum slot) the activation's input,
  # and the output is [d_gate | d_up], [size_m, 2 * size_n]. Needs a single
  # n tile. has_dact_row_scale: a per-row scale is read too.
  # return_dact_dot: a second, float32 [size_m, 128] output holds <u, act> per
  # row, broadcast across the lanes.
  fuse_dact: str | None = None
  has_dact_row_scale: bool = False
  return_dact_dot: bool = False
  # With fuse_dact: dact_gate_up is a fused forward's return_pre_act="concat"
  # output with this many columns per [gate_j | up_j] piece (that forward's
  # tile_n), read with split_dact_gate_up. None: the contiguous [gate | up].
  dact_pre_act_tile_n: int | None = None
  # VMEM buffers of the rhs (weight) tile pipeline. None: the default (3; 2 for a
  # chunked 3D lhs, see generate_block_specs). With 2, each weight block's DMA
  # starts one grid step before its first use instead of two, and one weight tile
  # of VMEM is freed (with fuse_act, a gate and an up tile). Values are unchanged.
  rhs_buffer_count: int | None = None

  @property
  def lhs_is_3d(self) -> bool:
    return self.lhs_cfgs.is_3d

  @property
  def lhs_tile_d0(self) -> int:
    assert self.lhs_is_3d
    return self.tiles.tile_k // pltpu.get_tpu_info().num_lanes

  @property
  def out_tile_d0(self) -> int:
    assert self.out_is_3d
    return self.tiles.tile_n // pltpu.get_tpu_info().num_lanes

  @property
  def lhs_window(self) -> bool:
    """3D lhs whose K tile is not a legal D0 block: DMA full-D0 slabs, relayout the K window in VMEM."""
    return self.lhs_is_3d and relayout.needs_window(self.lhs_tile_d0, self.dims.size_k // pltpu.get_tpu_info().num_lanes)

  @property
  def out_window(self) -> bool:
    """3D out whose N tile is not a legal D0 block: a full-D0 out slab stays resident across N.

    The grid is then (gm, n, k) instead of (n, gm, k), so every N tile of a gm
    tile is stored into the same VMEM slab before it is written back once.
    """
    return self.out_is_3d and relayout.needs_window(self.out_tile_d0, self.out_size_n // pltpu.get_tpu_info().num_lanes)

  @property
  def gm_axis(self) -> int:
    return 0 if self.out_window else 1

  @property
  def n_axis(self) -> int:
    return 1 if self.out_window else 0

  @property
  def lhs_chunk_d0(self) -> int | None:
    """D0 rows per in-VMEM relayout chunk for a large 3D lhs K tile (None: relayout the whole tile).

    Relayouting e.g. a [512, 7168] fp8 tile at once materializes it in VMEM; chunks
    of 1024 keep the relayout buffer small (the matmul accumulates across chunks).
    """
    if not self.lhs_is_3d or self.lhs_cfgs.quant_dtype is not None:
      return None
    if self.num_quant_blocks_per_tile_k != 1 or self.rhs_cfgs.should_dequantize_before_matmul:
      return None
    # 8-bit only: there the chunked accumulation matches one K-wide matmul (bit-exact
    # on some TPU generations, within ~1 bf16 ulp on others); bf16 matmuls round
    # noticeably differently across chunk boundaries.
    if jnp.dtype(self.lhs_cfgs.dtype).itemsize != 1 or jnp.dtype(self.rhs_cfgs.dtype).itemsize != 1:
      return None
    d0 = self.lhs_tile_d0
    if d0 <= 16 or d0 % 8:
      return None
    return 8

  @property
  def num_quant_blocks_per_tile_k(self) -> int:
    return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)  # pyrefly: ignore[no-matching-overload]

  @property
  def out_size_n(self) -> int:
    if self.fuse_act is not None:
      return self.dims.size_n // 2
    if self.fuse_dact is not None:
      return self.dims.size_n * 2
    return self.dims.size_n

  @property
  def out_tile_n(self) -> int:
    """Width of an output tile: with fuse_dact, [d_gate | d_up] is twice the matmul's tile_n."""
    if self.fuse_dact is not None:
      return self.tiles.tile_n * 2
    return self.tiles.tile_n


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int | None, str | None, bool], TileSizes]


class IndexMaps:
  """Index maps for GMM kernel."""

  def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs

  def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    """Block index of the lhs tile (rows of the group tile, K block; 3D lhs blocks K along D0)."""
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start

    if self.cfgs.lhs_is_3d:
      # Block is [rows, sublane, tile_d0, 128]: K is blocked along D0. A window
      # block covers the full D0 (the K window is relayouted from VMEM).
      return (pl.ds(row_start, row_size), 0, 0 if self.cfgs.lhs_window else k_id, 0)
    return (pl.ds(row_start, row_size), 0, k_id)

  def lhs_scale_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    # Per-tensor scale: a single [1, 1] value shared across every tile, so the
    # block always reads index 0. Extension point: when the scale is per-channel
    # or sub-channel, tile the row axis like `lhs_index_map` (using gm_id) and
    # index the K-block axis from `k_id`.
    del gm_id, k_id
    return (0, 0)

  def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    if self.cfgs.transpose_rhs:
      return (group_id, n_id, k_id)
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
    is_last_gm = gm_id == (pl.num_programs(self.cfgs.gm_axis) - 1)
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
    last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
    row_size = row_end - row_start

    if self.cfgs.out_is_3d:
      return (pl.ds(row_start, row_size), 0, 0 if self.cfgs.out_window else n_id, 0)
    return (pl.ds(row_start, row_size), 0, n_id)

  def ps_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    """Block index of the partial_sum tile; a 3D partial_sum (with a 3D out) is blocked like the out."""
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start

    if self.cfgs.out_is_3d:
      return (pl.ds(row_start, row_size), 0, 0 if self.cfgs.out_window else n_id, 0)
    return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> Tuple[Tuple[LhsRef, WeightsRef, pl.BlockSpec | None], pl.BlockSpec]:
  """Generates block specs for the given lhs, rhs, and out refs."""

  index_map = IndexMaps(metadata_ref, cfgs)
  bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

  num_lanes = pltpu.get_tpu_info().num_lanes

  # Index maps take (n_id, gm_id, k_id); with out_window the grid is (gm, n, k).
  def _ordered(fn):
    if not cfgs.out_window:
      return fn
    return lambda gm_id, n_id, k_id: fn(n_id, gm_id, k_id)

  if cfgs.lhs_is_3d:
    lhs_d0 = cfgs.dims.size_k // num_lanes if cfgs.lhs_window else cfgs.lhs_tile_d0
    lhs_value_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, lhs_d0, num_lanes),
        _ordered(index_map.lhs_index_map),
    )
  else:
    lhs_value_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        _ordered(index_map.lhs_index_map),
    )
  lhs_scale_spec = None
  if cfgs.lhs_cfgs.has_scale:
    lhs_scale_spec = pl.BlockSpec(
        (1, 1),
        _ordered(index_map.lhs_scale_index_map),
    )
  lhs_block_spec = LhsRef(value=lhs_value_spec, scale=lhs_scale_spec)

  tile_k_rhs = cfgs.tiles.tile_k
  if cfgs.rhs_cfgs.should_bitcast:
    packing = pl.cdiv(32, jax.dtypes.itemsize_bits(cfgs.rhs_cfgs.dtype))
    tile_k_rhs //= packing

  rhs_tile_shape = (None, cfgs.tiles.tile_n, tile_k_rhs) if cfgs.transpose_rhs else (None, tile_k_rhs, cfgs.tiles.tile_n)
  if cfgs.rhs_buffer_count is not None:
    rhs_buffer_count = cfgs.rhs_buffer_count
  else:
    # A full-K 3D lhs tile (e.g. a 512 x 7168 fp8 wi forward tile) needs VMEM for
    # the relayout; double-buffer the (large) rhs tile there.
    rhs_buffer_count = 2 if cfgs.lhs_chunk_d0 is not None else 3
  rhs_weight_spec = pl.BlockSpec(
      rhs_tile_shape,
      _ordered(index_map.rhs_weight_index_map),
      pipeline_mode=pl.Buffered(buffer_count=rhs_buffer_count),
  )
  rhs_scale_block_spec = rhs_bias_block_spec = ps_block_spec = None
  if cfgs.rhs_cfgs.has_bias:
    rhs_bias_block_spec = pl.BlockSpec(
        (None, 1, cfgs.tiles.tile_n),
        _ordered(index_map.rhs_bias_index_map),
    )
  if cfgs.rhs_cfgs.has_scale:
    rhs_scale_block_spec = pl.BlockSpec(
        (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
        _ordered(index_map.rhs_scale_index_map),
    )

  if cfgs.has_partial_sum and cfgs.out_is_3d:
    ps_d0 = cfgs.out_size_n // num_lanes if cfgs.out_window else cfgs.out_tile_d0
    ps_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, ps_d0, num_lanes),
        _ordered(index_map.ps_index_map),
    )
  elif cfgs.has_partial_sum:
    ps_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        _ordered(index_map.ps_index_map),
    )
  elif cfgs.fuse_dact is not None:
    gate_up_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, 2 * cfgs.tiles.tile_n),
        _ordered(index_map.ps_index_map),
    )
    row_scale_spec = None
    if cfgs.has_dact_row_scale:
      row_scale_spec = pl.BlockSpec(
          (bounded_slice_gm, cfgs.dims.size_lhs_sublane, num_lanes),
          _ordered(index_map.ps_index_map),
      )
    ps_block_spec = DactInputRef(gate_up=gate_up_spec, row_scale=row_scale_spec)

  rhs_block_spec = WeightsRef(
      weight=rhs_weight_spec,
      scale=rhs_scale_block_spec,
      bias=rhs_bias_block_spec,
  )

  if cfgs.out_is_3d:
    out_d0 = cfgs.out_size_n // num_lanes if cfgs.out_window else cfgs.out_tile_d0
    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, out_d0, num_lanes),
        _ordered(index_map.out_index_map),
    )
  else:
    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.out_tile_n),
        _ordered(index_map.out_index_map),
    )

  return (lhs_block_spec, rhs_block_spec, ps_block_spec), out_block_spec


# Define kernels.


def inner_kernel(
    # In
    tiled_lhs_ref: LhsRef,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n] or [tile_n, tile_k]
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
    tiled_rhs_ref: Contains the rhs tile for g_id. The tile is [tile_k, tile_n]
      normally and [tile_n, tile_k] when transpose_rhs is enabled.
    tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
    partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
      Will be initialized to zero if this is first tile for grid[n_id, :, :].
    acc_ref: Reference to the accumulator.
    metadata_ref: Reference to the metadata.
    cfgs: GmmConfigs.

  With cfgs.return_pre_act, tiled_out_ref and partial_out_ref are (act, gate,
  up) triples ((act, [gate | up]) pairs for "concat"). The activation takes the
  regular store path; gate and up are stored from the same accumulator before
  the activation is applied. With cfgs.fuse_dact, tiled_ps_ref is a
  DactInputRef and the accumulator becomes [d_gate | d_up] before the regular
  store path; with cfgs.return_dact_dot the out refs are (out, dot) pairs.
  """

  pre_act_refs = ()
  if cfgs.return_pre_act or cfgs.return_dact_dot:
    (tiled_out_ref, *tiled_pre_act_refs), (partial_out_ref, *partial_pre_act_refs) = tiled_out_ref, partial_out_ref
    pre_act_refs = tuple(zip(tiled_pre_act_refs, partial_pre_act_refs))

  def _matmul(is_first_k_step: bool, is_last_k_step: bool):
    tpu_info = pltpu.get_tpu_info()
    mxu_size = tpu_info.mxu_column_size

    # Step 1: Input pre-processing.
    lhs_chunk_d0 = cfgs.lhs_chunk_d0
    if cfgs.lhs_window:
      # Block is [tile_m // sublane, sublane, D0, 128]; relayout the K window
      # k_id of the full-D0 slab into [tile_m, tile_k].
      tiled_lhs = relayout.load_3d_window_as_2d(
          tiled_lhs_ref.value, cfgs.tiles.tile_m, cfgs.lhs_tile_d0, pl.program_id(2)
      )
    elif lhs_chunk_d0 is not None:
      tiled_lhs: Any = None  # Relayouted chunk by chunk in the matmul loop below.
    elif cfgs.lhs_is_3d:
      # Block is [tile_m // sublane, sublane, tile_d0, 128]; relayout into
      # [tile_m, tile_k] with tokens on sublanes.
      tiled_lhs = relayout.load_3d_as_2d(tiled_lhs_ref.value, cfgs.tiles.tile_m, cfgs.lhs_tile_d0)
    else:
      tiled_lhs = tiled_lhs_ref.get_value().reshape(-1, cfgs.tiles.tile_k)[...]
    tiled_rhs = tiled_rhs_ref.get_weight(transpose_rhs=cfgs.transpose_rhs)
    # When rhs is packed (quantized dtype packed into uint32), unpack it
    # back to the original dtype using pltpu.bitcast which operates on K
    # axis. This expands the K dimension back to tile_k.
    if cfgs.rhs_cfgs.should_bitcast:
      tiled_rhs = pltpu.bitcast(tiled_rhs, cfgs.rhs_cfgs.dtype)
    rhs_tile_n = tiled_rhs.shape[0 if cfgs.transpose_rhs else 1]

    # This should only be taken in the case where we don't requantize
    # the scales and thus we need to dequantize inside VMEM to avoid small
    # contracting dimensions
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
      rhs_qbs = cfgs.rhs_cfgs.quant_block_size
      tiled_rhs_scale = tiled_rhs_ref.get_scale().astype(acc_ref.dtype)
      num_blocks = cfgs.num_quant_blocks_per_tile_k
      if cfgs.transpose_rhs:
        tiled_rhs_dequant = tiled_rhs.astype(acc_ref.dtype).reshape(rhs_tile_n, num_blocks, rhs_qbs)
        tiled_rhs_dequant = tiled_rhs_dequant * jnp.moveaxis(tiled_rhs_scale, -1, 0)
        tiled_rhs = tiled_rhs_dequant.reshape(rhs_tile_n, cfgs.tiles.tile_k)
      else:
        tiled_rhs_dequant = tiled_rhs.astype(acc_ref.dtype).reshape(num_blocks, rhs_qbs, rhs_tile_n)
        tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
        tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)

    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    if is_last_k_step and valid_k != 0:
      k_axis = 1 if cfgs.transpose_rhs else 0
      mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, k_axis) < valid_k
      tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)
      # The lhs block past size_k is stale VMEM too: zeroing only rhs still lets a NaN / Inf there poison the
      # accumulator (NaN * 0 = NaN) and the in-kernel lhs quantization's block absmax. Only the 2D lhs path gets
      # here: 3D lhs requires size_k % tile_k == 0 (make_gmm_configs).
      if not cfgs.lhs_is_3d:
        mask_lhs = lax.broadcasted_iota(jnp.int32, tiled_lhs.shape, 1) < valid_k
        tiled_lhs = jnp.where(mask_lhs, tiled_lhs, 0)

    def rhs_block(start_k: int, end_k: int, start_n: int, end_n: int):
      if cfgs.transpose_rhs:
        return tiled_rhs[start_n:end_n, start_k:end_k]
      return tiled_rhs[start_k:end_k, start_n:end_n]

    # Step 2: Matmul.
    acc_list = []
    if lhs_chunk_d0 is not None:
      # Unquantized, single quant block: relayout the 3D lhs K tile in chunks and
      # accumulate each [tile_m, mxu] output column block across the chunks.
      num_lanes = pltpu.get_tpu_info().num_lanes
      n_starts = list(range(0, rhs_tile_n, mxu_size))
      acc_ns: list[Any] = [None] * len(n_starts)
      for c0 in range(0, cfgs.lhs_tile_d0, lhs_chunk_d0):
        lhs_c = relayout.load_3d_as_2d(tiled_lhs_ref.value, cfgs.tiles.tile_m, lhs_chunk_d0, c0)
        start_k, end_k = c0 * num_lanes, (c0 + lhs_chunk_d0) * num_lanes
        for i, start_n in enumerate(n_starts):
          end_n = min(rhs_tile_n, start_n + mxu_size)
          rhs_b = rhs_block(start_k, end_k, start_n, end_n)
          if cfgs.transpose_rhs:
            part = lax.dot_general(lhs_c, rhs_b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
          else:
            part = jnp.matmul(lhs_c, rhs_b, preferred_element_type=jnp.float32)
          acc_ns[i] = part if acc_ns[i] is None else acc_ns[i] + part
      for i, start_n in enumerate(n_starts):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        block_acc = acc_ns[i].astype(acc_ref.dtype)
        if cfgs.rhs_cfgs.should_dequantize_after_matmul:
          tiled_rhs_scale = tiled_rhs_ref.get_scale()
          block_acc *= tiled_rhs_scale[0, :, start_n:end_n].astype(acc_ref.dtype)
        acc_n = jnp.zeros((cfgs.tiles.tile_m, end_n - start_n), dtype=acc_ref.dtype)
        acc_n += block_acc
        acc_list.append(acc_n)
    elif cfgs.lhs_cfgs.quant_dtype is None:
      # Unquantized matmul path.
      rhs_qbs = cfgs.rhs_cfgs.quant_block_size

      for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n

        acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)
        for b_id in range(cfgs.num_quant_blocks_per_tile_k):
          start_k = b_id * rhs_qbs  # pyrefly: ignore[unsupported-operation]
          end_k = start_k + rhs_qbs  # pyrefly: ignore[unsupported-operation]

          block_lhs = tiled_lhs[:, start_k:end_k]
          block_rhs = rhs_block(start_k, end_k, start_n, end_n)
          if cfgs.transpose_rhs:
            block_acc = lax.dot_general(
                block_lhs,
                block_rhs,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(acc_ref.dtype)
          else:
            block_acc = jnp.matmul(
                block_lhs,
                block_rhs,
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
          block_rhs = rhs_block(start_k, end_k, start_n, end_n)

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

          if cfgs.transpose_rhs:
            block_acc = lax.dot_general(
                block_lhs_q,
                block_rhs,
                (((1,), (1,)), ((), ())),
                preferred_element_type=preferred_element_type,
            ).astype(acc_ref.dtype)
          else:
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
      if cfgs.lhs_cfgs.has_scale and cfgs.lhs_cfgs.quant_dtype is None:
        acc *= tiled_lhs_ref.get_scale().astype(acc.dtype)
      if cfgs.rhs_cfgs.has_bias:
        tiled_rhs_bias = tiled_rhs_ref.get_bias()
        acc += tiled_rhs_bias.astype(acc.dtype)
      if cfgs.has_partial_sum:
        if cfgs.out_window:
          ps_tile = relayout.load_3d_window_as_2d(
              tiled_ps_ref, cfgs.tiles.tile_m, cfgs.out_tile_d0, pl.program_id(cfgs.n_axis)
          )
        elif cfgs.out_is_3d:
          ps_tile = relayout.load_3d_as_2d(tiled_ps_ref, cfgs.tiles.tile_m, cfgs.out_tile_d0)
        else:
          ps_tile = tiled_ps_ref[...].reshape(acc.shape)  # pyrefly: ignore[unsupported-operation]
        acc += ps_tile.astype(acc.dtype)

      if cfgs.return_pre_act:
        # acc is [tile_m, 2 * tile_n] = gate | up. Store before the activation, rounded to the
        # output dtype exactly as the unfused kernel rounds its output; the activation below stays in f32.
        if cfgs.return_pre_act == "concat":
          # One [gate_j | up_j] block per n tile (the whole row [gate | up] with one n tile per half); a
          # fuse_dact reader de-interleaves it with split_dact_gate_up(..., tile_n).
          tiled_pre_ref, partial_pre_ref = pre_act_refs[0]
          _store_group_rows(tiled_pre_ref, partial_pre_ref, acc, metadata_ref, cfgs)
        else:
          for (tiled_pre_ref, partial_pre_ref), acc_half in zip(pre_act_refs, jnp.split(acc, 2, -1)):
            _store_group_rows(tiled_pre_ref, partial_pre_ref, acc_half, metadata_ref, cfgs)

      acc = apply_act_fn(acc, cfgs.fuse_act)

      if cfgs.fuse_dact is not None:
        # acc is the activation output's cotangent u for this tile's rows; the gate | up block covers the
        # whole row (make_gmm_configs requires a single n tile), in the layout the fused forward wrote it
        # (split_dact_gate_up). The [d_gate | d_up] it becomes takes the regular store path below (masking,
        # rounding to out_dtype, partial-row carry), contiguous whatever that layout.
        tile_m, tile_n = cfgs.tiles.tile_m, cfgs.tiles.tile_n
        gate_up_tile = tiled_ps_ref.gate_up[...].reshape(tile_m, 2 * tile_n)  # pyrefly: ignore[missing-attribute]
        gate_tile, up_tile = split_dact_gate_up(gate_up_tile, cfgs.dact_pre_act_tile_n)
        row_scale_tile = None
        if cfgs.has_dact_row_scale:
          row_scale_tile = tiled_ps_ref.row_scale[...].reshape(tile_m, -1)[:, :1]  # pyrefly: ignore[missing-attribute]
        acc, dot_tile = apply_dact_fn(
            acc,
            gate_tile,
            up_tile,
            row_scale_tile,
            cfgs.fuse_dact,
            return_dot=cfgs.return_dact_dot,
        )
        if cfgs.return_dact_dot:
          # The dot is one float32 per row: broadcast to a lane row and stored through the extra output.
          tiled_dot_ref, partial_dot_ref = pre_act_refs[0]
          dot_tile_padded = jnp.broadcast_to(dot_tile, (cfgs.tiles.tile_m, tpu_info.num_lanes))
          _store_group_rows(tiled_dot_ref, partial_dot_ref, dot_tile_padded, metadata_ref, cfgs)

      # Write the final output to the output ref.
      if not cfgs.out_is_3d:
        _store_group_rows(tiled_out_ref, partial_out_ref, acc, metadata_ref, cfgs)
        return
      acc_masked, gm_id, m_end_local = _mask_group_rows(acc, metadata_ref, cfgs)
      if cfgs.out_window:
        # Full-D0 out slab resident across the N tiles of this gm tile. The
        # cross-gm partial-row carry is kept per N tile, in the 2D domain.
        sub = cfgs.dims.size_lhs_sublane
        n_id = pl.program_id(cfgs.n_axis)
        out2d = acc_masked.astype(tiled_out_ref.dtype)
        prev = partial_out_ref[n_id]
        prev = jnp.where(gm_id == 0, jnp.zeros_like(prev), prev)
        out2d = jnp.concatenate([out2d[:sub] + prev, out2d[sub:]], axis=0)
        # Carry the (possibly shared) last sublane row to the next gm tile.
        acc_ref[...] = out2d.astype(acc_ref.dtype)
        last_row = m_end_local // sub
        last = acc_ref[pl.ds(pl.multiple_of(last_row * sub, sub), sub), :].astype(out2d.dtype)
        partial_out_ref[n_id] = jnp.where(m_end_local % sub == 0, jnp.zeros_like(last), last)
        relayout.store_2d_as_3d_window(tiled_out_ref, out2d, out2d.shape[0], cfgs.out_tile_d0, n_id)
        return
      # Block is [tile_m // sublane, sublane, tile_d0, 128]; scatter the
      # [tile_m, tile_n] rows back into per-token [tile_d0, 128] slabs.
      relayout.store_2d_as_3d(
          tiled_out_ref, acc_masked.astype(tiled_out_ref.dtype), acc_masked.shape[0], cfgs.out_tile_d0
      )
      _carry_partial_rows(tiled_out_ref, partial_out_ref, gm_id, m_end_local, cfgs)
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


def _mask_group_rows(acc: jax.Array, metadata_ref: MetadataRef, cfgs: GmmConfigs):
  """Zeroes the rows of `acc` that do not belong to the current gm tile's group.

  Returns the masked `acc`, the gm tile id and the group's end row relative to
  the tile's first sublane row (what _carry_partial_rows needs).
  """
  gm_id = pl.program_id(cfgs.gm_axis)

  # Mask out rows that does not belong to the current group.
  m_start = metadata_ref.gm_id_to_m_offset[gm_id]
  m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
  m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

  m_start_local = m_start - m_offset
  m_end_local = m_end - m_offset

  iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
  mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
  return jnp.where(mask, acc, 0), gm_id, m_end_local


def _carry_partial_rows(
    tiled_out_ref: jax.Array,
    partial_out_ref: jax.Array,
    gm_id: jax.Array,
    m_end_local: jax.Array,
    cfgs: GmmConfigs,
):
  """Adds the sublane rows the previous gm tile left in `partial_out_ref`; leaves this tile's unfinished ones there."""
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


def _store_group_rows(
    tiled_ref: jax.Array,  # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    partial_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    cfgs: GmmConfigs,
):
  """The 2D store of inner_kernel's last k step: the current group's rows of `acc`, with the partial-row carry.

  Zeroes the rows of other groups, rounds to the output dtype, stores the tile
  and carries its unfinished last sublane rows to the next gm tile. Used for the
  plain 2D output and for the extra (return_pre_act) outputs written from the
  same accumulator.
  """
  acc_masked, gm_id, m_end_local = _mask_group_rows(acc, metadata_ref, cfgs)
  tiled_ref[...] = acc_masked.reshape(tiled_ref.shape).astype(tiled_ref.dtype)
  _carry_partial_rows(tiled_ref, partial_ref, gm_id, m_end_local, cfgs)


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

  out_is_3d = out_ref.ndim == 3
  if out_is_3d:
    # [size_m, D0, 128] -> [size_m // sublane, sublane, D0, 128]. Each DMA
    # zeroes whole [D0, 128] slabs.
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
    out_ref: jax.Array,  # [size_m, size_n]
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
    pre_act_out_refs: tuple[jax.Array, ...] = (),  # return_pre_act: (gate, up) outputs, each shaped like out_ref
    pre_act_partial_refs: tuple[jax.Array, ...] = (),  # return_pre_act: their partial-row scratch
    # Per extra output, the VMEM zero buffer of its zero-init DMAs when its dtype is not out_ref's (None: zero_ref).
    extra_zero_refs: tuple[jax.Array | None, ...] = (),
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
  num_n = pl.cdiv(cfgs.out_size_n, cfgs.out_tile_n)

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
    # The extra outputs (pre-activation gate and up, [gate | up], or the fuse_dact dot) zero the same rows on
    # the same semaphore, with DMAs over their own widths; zero_out_end below waits for them one output at a
    # time. The float32 dot needs a float32 zero buffer (extra_zero_refs).
    for i, pre_act_out_ref in enumerate(pre_act_out_refs):
      z_ref = extra_zero_refs[i] if i < len(extra_zero_refs) and extra_zero_refs[i] is not None else zero_ref
      zero_out_start(
          pre_act_out_ref,
          z_ref,  # pyrefly: ignore[bad-argument-type]
          semaphore_ref,  # pyrefly: ignore[bad-argument-type]
          metadata_ref,
          num_gm,
          dims=cfgs.dims,
      )

  (lhs_spec, rhs_spec, ps_spec), out_spec = generate_block_specs(metadata_ref, cfgs)
  if pre_act_out_refs:
    # The extra outputs' tiles share the out index map; inner_kernel receives them as one tuple with out's.
    if cfgs.return_pre_act == "concat":
      pre_spec = dataclasses.replace(
          out_spec,
          block_shape=(out_spec.block_shape[0], out_spec.block_shape[1], 2 * cfgs.out_tile_n),
      )
    elif cfgs.return_dact_dot:
      pre_spec = dataclasses.replace(
          out_spec,
          block_shape=(out_spec.block_shape[0], out_spec.block_shape[1], pltpu.get_tpu_info().num_lanes),
      )
    else:
      pre_spec = out_spec
    out_spec = ((out_spec,) + (pre_spec,) * len(pre_act_out_refs),)

  if cfgs.fuse_act is not None:
    if cfgs.transpose_rhs:
      rhs_up_weight = rhs_ref.weight.at[:, cfgs.out_size_n :, :]
    else:
      rhs_up_weight = rhs_ref.weight.at[..., cfgs.out_size_n :]
    rhs_up_scale = None if rhs_ref.scale is None else rhs_ref.scale.at[..., cfgs.out_size_n :]
    rhs_up_bias = None if rhs_ref.bias is None else rhs_ref.bias.at[..., cfgs.out_size_n :]
    rhs_up_ref = WeightsRef(weight=rhs_up_weight, scale=rhs_up_scale, bias=rhs_up_bias)
    rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)  # pyrefly: ignore[bad-assignment]

    rhs_spec = FusedWeightsRef(
        gate=rhs_spec,
        up=rhs_spec,
    )

  # Execute the inner kernel.
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(inner_kernel, cfgs=cfgs),
      grid=(num_gm, num_n, num_k) if cfgs.out_window else (num_n, num_gm, num_k),
      in_specs=(lhs_spec, rhs_spec, ps_spec),
      out_specs=out_spec,
  )

  # Bounded slice requires second last dim to be aligned to the sublane size.
  # rhs_ref uses static tiling thus reshape is not needed. The lhs quant scale
  # (when present) is small and statically tiled, so it is passed through as-is.
  if cfgs.lhs_is_3d:
    # [size_m, D0, 128] -> [size_m // sublane, sublane, D0, 128].
    lhs_value_in = lhs_ref.value.reshape(-1, cfgs.dims.size_lhs_sublane, *lhs_ref.value.shape[-2:])
  else:
    lhs_value_in = lhs_ref.value.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.value.shape[-1])
  lhs_in = LhsRef(value=lhs_value_in, scale=lhs_ref.scale)
  ps_in = None
  if cfgs.has_partial_sum:
    # Works for both [size_m, size_n] and the 3D [size_m, D0, 128] partial_sum.
    ps_in = partial_sum_ref.reshape(-1, cfgs.dims.size_lhs_sublane, *partial_sum_ref.shape[1:])
  elif cfgs.fuse_dact is not None:
    gate_up_in = partial_sum_ref.gate_up.reshape(-1, cfgs.dims.size_lhs_sublane, *partial_sum_ref.gate_up.shape[1:])
    row_scale_in = None
    if cfgs.has_dact_row_scale:
      row_scale_in = partial_sum_ref.row_scale.reshape(
          -1, cfgs.dims.size_lhs_sublane, *partial_sum_ref.row_scale.shape[1:]
      )
    ps_in = DactInputRef(gate_up=gate_up_in, row_scale=row_scale_in)
  # Works for both [size_m, size_n] and the 3D [size_m, D0, 128] output.
  out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, *out_ref.shape[1:])
  scratches = [partial_out_ref, acc_ref, metadata_ref]
  if pre_act_out_refs:
    out_in = (out_in, *(r.reshape(-1, cfgs.dims.size_lhs_sublane, *r.shape[1:]) for r in pre_act_out_refs))
    scratches[0] = (partial_out_ref, *pre_act_partial_refs)
  pipeline_fn(lhs_in, rhs_ref, ps_in, out_in, scratches=scratches)

  if cfgs.zero_init:
    zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)  # pyrefly: ignore[bad-argument-type, unbound-name]
    for pre_act_out_ref in pre_act_out_refs:
      # pyrefly: ignore[bad-argument-type, unbound-name]
      zero_out_end(pre_act_out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


def kernel_main_pre_act(
    lhs_group_sizes_ref: jax.Array,
    group_offset_ref: jax.Array,
    lhs_ref: LhsRef,
    rhs_ref: WeightsRef,
    partial_sum_ref: jax.Array,
    out_ref: jax.Array,  # act: [size_m, size_n // 2]
    gate_out_ref: jax.Array,  # gate before the activation: [size_m, size_n // 2]
    up_out_ref: jax.Array,  # up before the activation: [size_m, size_n // 2]
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,
    semaphore_ref: jax.Array | None,
    partial_gate_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    partial_up_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    *,
    cfgs: GmmConfigs,
):
  """kernel_main for return_pre_act: pallas_call passes the gate and up outputs right after act."""
  kernel_main(
      lhs_group_sizes_ref,
      group_offset_ref,
      lhs_ref,
      rhs_ref,
      partial_sum_ref,
      out_ref,
      partial_out_ref,
      acc_ref,
      metadata_ref,
      zero_ref,
      semaphore_ref,
      cfgs=cfgs,
      pre_act_out_refs=(gate_out_ref, up_out_ref),
      pre_act_partial_refs=(partial_gate_ref, partial_up_ref),
  )


def kernel_main_single_extra_out(
    lhs_group_sizes_ref: jax.Array,
    group_offset_ref: jax.Array,
    lhs_ref: LhsRef,
    rhs_ref: WeightsRef,
    partial_sum_ref: jax.Array,
    out_ref: jax.Array,
    extra_out_ref: jax.Array,
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,
    semaphore_ref: jax.Array | None,
    partial_extra_ref: jax.Array,
    extra_zero_ref: jax.Array | None,
    *,
    cfgs: GmmConfigs,
):
  """kernel_main for one extra output: [gate | up] (return_pre_act="concat") or the dot (return_dact_dot).

  pallas_call passes the extra output right after out, and its partial-row scratch and (for the float32 dot) its
  zero buffer last.
  """
  kernel_main(
      lhs_group_sizes_ref,
      group_offset_ref,
      lhs_ref,
      rhs_ref,
      partial_sum_ref,
      out_ref,
      partial_out_ref,
      acc_ref,
      metadata_ref,
      zero_ref,
      semaphore_ref,
      cfgs=cfgs,
      pre_act_out_refs=(extra_out_ref,),
      pre_act_partial_refs=(partial_extra_ref,),
      extra_zero_refs=(extra_zero_ref,),
  )


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
        # pyrefly: ignore[unsupported-operation]
        tk % rhs_cfgs.quant_block_size != 0
        and rhs_cfgs.quant_block_size % tk != 0  # pyrefly: ignore[unsupported-operation]
    ):
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
    transpose_rhs: bool = False,
) -> Dimensions:
  """Validates the inputs for the GMM kernel."""

  size_m = lhs.shape[0]
  if transpose_rhs:
    size_group, size_n, size_k = rhs.shape
  else:
    size_group, size_k, size_n = rhs.shape
  size_lhs_group = group_sizes.shape[0]

  assert size_group <= size_lhs_group
  if lhs.ndim == 3:
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert lhs.shape == (size_m, size_k // num_lanes, num_lanes), (lhs.shape, size_k)
  else:
    assert lhs.shape == (size_m, size_k)
  expected_rhs_shape = (size_group, size_n, size_k) if transpose_rhs else (size_group, size_k, size_n)
  assert rhs.shape == expected_rhs_shape
  if transpose_rhs and jax.dtypes.itemsize_bits(rhs.dtype) < 8:
    raise ValueError("transpose_rhs is not supported for sub-byte rhs dtypes.")
  if rhs_bias is not None:
    assert rhs_bias.shape == (size_group, 1, size_n)
  if partial_sum is not None:
    if partial_sum.ndim == 3:
      # 3D partial_sum [m, size_n // 128, 128] (requires a 3D out; checked in make_gmm_configs).
      num_lanes = pltpu.get_tpu_info().num_lanes
      assert partial_sum.shape[1:] == (size_n // num_lanes, num_lanes), (partial_sum.shape, size_n)
    else:
      assert partial_sum.shape[-1] == size_n
    # lhs's m dimension can sometimes be padded to wi_tile_fwd_batch_seq
    assert partial_sum.shape[0] <= size_m
  if rhs_scale is not None:
    num_quant_blocks = rhs_scale.shape[1]
    assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n), (
        f"rhs_scale shape {rhs_scale.shape}. Expecting ({size_group}," f" {num_quant_blocks}, 1, {size_n})"
    )
    assert size_k % num_quant_blocks == 0

  if lhs_scale is not None:
    # Only per-tensor scales are supported for now. The current implementation generalizes to per-channel [M, 1] and
    # sub-channel [M, num_k_blocks]; extend the validation and the block spec /
    # index map together when adding those.
    assert lhs_scale.shape == (1, 1), "Only per-tensor lhs_scale of shape (1, 1) is supported, got " f"{lhs_scale.shape}."

  assert group_offset.shape == (1,)

  size_lhs_sublane = max(pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype), 16)
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
  if cfgs.return_pre_act:
    out_bytes *= 3  # act, gate and up
  if cfgs.fuse_dact is not None:
    out_bytes *= 2  # dact_gate_up read + d_gate_up write

  total_bytes = lhs_bytes + rhs_bytes + out_bytes

  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=total_bytes,
      transcendentals=0,
  )


def get_scope_name(cfgs: GmmConfigs) -> str:
  """The kernel's named scope: its dims, tiles and the flags that change the kernel."""
  dims = cfgs.dims
  tiles = cfgs.tiles
  scope_name = (
      f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )
  if cfgs.transpose_rhs:
    scope_name += "-transpose_rhs"
  if cfgs.return_pre_act == "concat":
    scope_name += "-pre_act_concat"
  elif cfgs.return_pre_act:
    scope_name += "-pre_act"
  if cfgs.fuse_dact is not None:
    scope_name += f"-dact_{cfgs.fuse_dact}"
    if cfgs.dact_pre_act_tile_n is not None:
      scope_name += f"-preact_tn_{cfgs.dact_pre_act_tile_n}"
  if cfgs.rhs_buffer_count is not None:
    scope_name += f"-rhsbuf_{cfgs.rhs_buffer_count}"
  return scope_name


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
    transpose_rhs: bool = False,
    out_is_3d: bool = False,
    return_pre_act: bool | str = False,
    fuse_dact: str | None = None,
    dact_gate_up: jax.Array | None = None,
    dact_row_scale: jax.Array | None = None,
    return_dact_dot: bool = False,
    dact_pre_act_tile_n: int | None = None,
    rhs_buffer_count: int | None = None,
):
  """Fills the GMM config for the GMM kernel."""

  dims = validate_inputs(
      lhs,
      rhs,
      rhs_scale,
      rhs_bias,
      partial_sum,
      group_sizes,
      group_offset,
      fuse_act=fuse_act,
      maybe_quantize_lhs=maybe_quantize_lhs,
      lhs_scale=lhs_scale,
      transpose_rhs=transpose_rhs,
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

  if lhs_scale is not None and maybe_quantize_lhs:
    assert lhs_q_dtype is not None, (
        "lhs_scale requires lhs quantization to engage, but no lhs quant "
        "dtype was selected. Ensure rhs is quantized and the hardware supports "
        "fp8/int8 matmul."
    )
  has_lhs_scale = lhs_scale is not None

  lhs_cfgs = InputConfigs(
      quant_dtype=lhs_q_dtype,
      # Input quantization involves reading all elements in a block to compute
      # scale value. Since this operation is very memory intensive, we use a
      # block size that is small enough to minimize memory overhead but large
      # enough to minimize compute overhead of quantization.
      quant_block_size=512,
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
    tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act, partial_sum is not None)

  num_lanes = pltpu.get_tpu_info().num_lanes
  if lhs_cfgs.is_3d:
    if tiles.tile_k % num_lanes or dims.size_k % tiles.tile_k:
      raise ValueError(f"3D lhs requires tile_k % 128 == 0 and size_k % tile_k == 0; got {tiles=}, {dims=}.")
    lhs_tile_d0, lhs_full_d0 = tiles.tile_k // num_lanes, dims.size_k // num_lanes
    if relayout.needs_window(lhs_tile_d0, lhs_full_d0):
      relayout.check_window(lhs_tile_d0, lhs_full_d0, lhs.dtype, store=False)
    else:
      relayout.check_tile_d0(lhs_tile_d0, full_d0=lhs_full_d0, dtype=lhs.dtype)
    if tiles.tile_m % dims.size_lhs_sublane:
      raise ValueError(f"3D lhs requires tile_m % sublane == 0; got {tiles=}, {dims=}.")
  if out_is_3d:
    out_size_n = dims.size_n // 2 if fuse_act is not None else dims.size_n
    if tiles.tile_n % num_lanes or out_size_n % tiles.tile_n:
      raise ValueError(f"3D out requires tile_n % 128 == 0 and out_size_n % tile_n == 0; got {tiles=}, {dims=}.")
    out_tile_d0, out_full_d0 = tiles.tile_n // num_lanes, out_size_n // num_lanes
    if relayout.needs_window(out_tile_d0, out_full_d0):
      relayout.check_window(out_tile_d0, out_full_d0, out_dtype, store=True)
      if fuse_act is not None:
        raise NotImplementedError("A 3D out window does not support fuse_act.")
    else:
      relayout.check_tile_d0(out_tile_d0, full_d0=out_full_d0, dtype=out_dtype)
    if partial_sum is not None:
      if partial_sum.ndim != 3 or partial_sum.shape[0] != dims.size_m:
        raise ValueError(
            f"A 3D output requires a 3D partial_sum of shape [size_m, out_size_n // 128, 128]; got {partial_sum.shape}."
        )
      if fuse_act is not None:
        raise NotImplementedError("A 3D partial_sum does not support fuse_act.")
      ps_d0 = out_full_d0 if relayout.needs_window(out_tile_d0, out_full_d0) else out_tile_d0
      relayout.check_tile_d0(ps_d0, full_d0=out_full_d0, dtype=partial_sum.dtype)
      if relayout.needs_window(out_tile_d0, out_full_d0):
        relayout.check_window(out_tile_d0, out_full_d0, partial_sum.dtype, store=False)
    if tiles.tile_m % 2:
      raise ValueError(f"3D out requires an even tile_m; got {tiles=}.")
  elif partial_sum is not None and partial_sum.ndim == 3:
    raise ValueError("A 3D partial_sum requires out_is_3d=True.")
  if return_pre_act not in (False, True, "concat"):
    raise ValueError(f'return_pre_act must be False, True or "concat"; got {return_pre_act!r}.')
  if return_pre_act and (fuse_act is None or out_is_3d or partial_sum is not None or fuse_dact is not None):
    raise NotImplementedError("return_pre_act needs fuse_act and a 2D output, and no partial_sum.")
  if return_pre_act == "concat" and (dims.size_n // 2) % tiles.tile_n:
    # Each n tile's accumulator is [gate tile | up tile], stored as one block per n tile: the layout is
    # [gate_0 | up_0 | gate_1 | up_1 | ...] (split_dact_gate_up reads it back), which needs whole tiles.
    raise NotImplementedError(
        f'return_pre_act="concat" requires tile_n to divide size_n // 2 = {dims.size_n // 2}; got {tiles=}.'
    )
  if fuse_dact is not None:
    if fuse_dact != "swiglu":
      raise NotImplementedError(f'fuse_dact supports "swiglu" only; got {fuse_dact!r}.')
    if fuse_act is not None or out_is_3d or partial_sum is not None:
      raise NotImplementedError("fuse_dact does not support fuse_act, out_is_3d, or partial_sum.")
    if rhs_scale is not None or rhs_bias is not None or lhs_scale is not None:
      # The epilogue reads the float32 accumulator of a plain (unscaled, unquantized) matmul.
      raise NotImplementedError("fuse_dact does not support rhs_scale, rhs_bias or lhs_scale.")
    if dact_gate_up is None or dact_gate_up.shape != (dims.size_m, dims.size_n * 2):
      raise ValueError(
          f"fuse_dact requires dact_gate_up of shape ({dims.size_m}, {dims.size_n * 2}); "
          f"got {None if dact_gate_up is None else dact_gate_up.shape}."
      )
    if tiles.tile_n != dims.size_n:
      # The [d_gate | d_up] tile of an n tile is stored as one block, as with return_pre_act="concat": it is
      # the contiguous [d_gate | d_up] the next GMM's backward reads only with one tile. (The gate | up block
      # it reads is then the whole row, in whatever layout dact_pre_act_tile_n describes.)
      raise NotImplementedError(f"fuse_dact requires a single n-tile (tile_n == size_n); got {tiles=}, {dims=}.")
    if dact_pre_act_tile_n == dims.size_n:
      dact_pre_act_tile_n = None  # one piece per half: the contiguous [gate | up], the same kernel
    if dact_pre_act_tile_n is not None and (
        dact_pre_act_tile_n <= 0 or dims.size_n % dact_pre_act_tile_n or dact_pre_act_tile_n % num_lanes
    ):
      # The single n tile spans size_n, so a forward tile wider than size_n cannot exist; one that does not
      # divide it would leave a partial piece, and one off the lane grid could not be selected in whole vregs.
      raise NotImplementedError(
          f"dact_pre_act_tile_n (the fused forward's tile_n per half) must be a multiple of {num_lanes} that"
          f" divides size_n = {dims.size_n}; got {dact_pre_act_tile_n}."
      )
    if dact_row_scale is not None and dact_row_scale.shape not in (
        (dims.size_m,),
        (dims.size_m, 1),
        (dims.size_m, num_lanes),
    ):
      raise ValueError(f"dact_row_scale must have shape ({dims.size_m}, ...); got {dact_row_scale.shape}.")
  elif dact_gate_up is not None or dact_row_scale is not None or return_dact_dot or dact_pre_act_tile_n is not None:
    raise ValueError("dact_gate_up, dact_row_scale, return_dact_dot and dact_pre_act_tile_n require fuse_dact to be set.")
  if rhs_buffer_count is not None and (not isinstance(rhs_buffer_count, int) or rhs_buffer_count < 2):
    # One buffer would serialize every weight DMA with the matmul that reads it.
    raise ValueError(f"rhs_buffer_count must be None or an int >= 2; got {rhs_buffer_count!r}.")

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
      transpose_rhs=transpose_rhs,
      out_is_3d=out_is_3d,
      return_pre_act=return_pre_act,
      fuse_dact=fuse_dact,
      has_dact_row_scale=dact_row_scale is not None,
      return_dact_dot=return_dact_dot,
      dact_pre_act_tile_n=dact_pre_act_tile_n,
      rhs_buffer_count=rhs_buffer_count,
  )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str | int | float]:
  """The kernel config flattened to the pallas_call metadata dict."""
  cfgs_dict = dataclasses.asdict(cfgs)
  if not cfgs.return_pre_act:
    del cfgs_dict["return_pre_act"]  # keeps the metadata of every other gmm_v2 call unchanged
  if cfgs.fuse_dact is None:
    del cfgs_dict["fuse_dact"]
    del cfgs_dict["has_dact_row_scale"]
    del cfgs_dict["return_dact_dot"]
  if cfgs.dact_pre_act_tile_n is None:
    del cfgs_dict["dact_pre_act_tile_n"]  # the contiguous [gate | up] reader keeps its metadata too
  if cfgs.rhs_buffer_count is None:
    del cfgs_dict["rhs_buffer_count"]  # the default rhs buffering keeps its metadata too
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
        "transpose_rhs",
        "out_is_3d",
        "return_pre_act",
        "fuse_dact",
        "return_dact_dot",
        "dact_pre_act_tile_n",
        "rhs_buffer_count",
    ]
)
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n], or [size_group, size_n, size_k] if transposed
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
    transpose_rhs: bool = False,
    out_is_3d: bool = False,
    return_pre_act: bool | str = False,
    fuse_dact: str | None = None,
    dact_gate_up: jax.Array | None = None,
    dact_row_scale: jax.Array | None = None,
    return_dact_dot: bool = False,
    dact_pre_act_tile_n: int | None = None,
    rhs_buffer_count: int | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
  """GMM kernel implemented with emit_pipeline.

  Dynamically calculate offset lhs/out tiles to reduce redundant computations.
  Additionally, it adjusts dma size based on number of valid rows and utilize
  triple buffering on weights to better utilize memory.

  Args:
    lhs: lhs with shape [size_m, size_k].
    rhs: rhs with shape [size_group, size_k, size_n], or
      [size_group, size_n, size_k] when transpose_rhs=True.
    group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
    rhs_scale: The rhs scale of shape [size_group, num_blocks, 1, out_size].
    rhs_bias: The rhs bias of shape [size_group, 1, out_size].
    partial_sum: Optional. Per-token partial sums of shape [size_m, size_n].
    group_offset: Optional. The group offset of shape [1,].
    lhs_scale: Optional per-tensor `[1, 1]` lhs scale. With maybe_quantize_lhs
      and a quantized rhs, it is used to quantize the (unquantized) lhs inside
      the kernel and the result is multiplied back by `scale`; when None, a
      quantized lhs uses the default dynamic per-block absmax calibration.
      Otherwise the accumulator is multiplied by `scale` after the matmul,
      before rhs_bias and partial_sum are added.
    tile_info: The tile sizes or tile function to use.
    vmem_limit_bytes: Optional vmem limit in bytes.
    precision: Unused. Exists for compatibility reasons.
    preferred_element_type: Optional jnp.dtype for the output matrix.
    acc_dtype: Optional jnp.dtype for the accumulator.
    maybe_quantize_lhs: Quantize lhs if set to True and rhs is quantized.
    zero_initialize: Whether to initialize unvisited output elements to zero.
    fuse_act: Activation function to fuse with GMM, None if no fusion.
    transpose_rhs: Interpret rhs as [size_group, size_n, size_k] and contract
      its last dimension without materializing a full rhs transpose.
    out_is_3d: If True, the output is written directly in the 3D layout
      [size_m, out_size_n // 128, 128], relayouted from [tile_m, tile_n] inside
      VMEM. A 3D lhs ([size_m, size_k // 128, 128]) is detected from lhs.ndim.
      Both require tile_k / tile_n to be a multiple of 128 that divides the
      dim. Tiles whose D0 (tile // 128) is not a multiple of 8 use full-D0
      blocks and relayout the tile window in VMEM.
    return_pre_act: With fuse_act, also return the gate and up halves of the
      matmul before the activation (what the activation's backward needs), each
      rounded to the output dtype as the unfused kernel rounds them. The
      activation itself is computed from the f32 accumulator. 2D output only,
      no partial_sum. "concat" returns them as one [size_m, size_n] array
      instead (the input of a fuse_dact epilogue), one [gate_j | up_j] block
      per n tile: [gate | up] with one tile per half (tile_n == size_n // 2),
      else interleaved by tile (a fuse_dact reader passes this tile_n as its
      dact_pre_act_tile_n). tile_n must divide size_n // 2.
    fuse_dact: The backward of a gated activation in the epilogue ("swiglu",
      see apply_dact_fn): the matmul computes the activation output's cotangent
      u = lhs @ rhs^T, and the kernel writes [d_gate | d_up] of shape
      [size_m, 2 * size_n] instead of u, computed in f32 from the accumulator.
      Needs dact_gate_up and a single n tile (tile_n == size_n); 2D output, no
      fuse_act, partial_sum, scales or bias.
    dact_gate_up: With fuse_dact, the activation's input [gate | up] of shape
      [size_m, 2 * size_n].
    dact_row_scale: With fuse_dact, an optional scale per row ([size_m] or
      [size_m, 1]) that multiplies u before the derivative.
    return_dact_dot: With fuse_dact, also return <u, silu(gate) * up> per row in
      f32, shape [size_m].
    dact_pre_act_tile_n: With fuse_dact, the layout of dact_gate_up when it is a
      fused forward's return_pre_act="concat" output: that forward's tile_n (per
      half), i.e. the width of its [gate_j | up_j] pieces. None (or size_n):
      the contiguous [gate | up]. The output is [d_gate | d_up] either way.
    rhs_buffer_count: VMEM buffers of the rhs (weight) tile pipeline, >= 2.
      None: the default (3, or 2 for a chunked 3D lhs). 2 frees one weight
      tile of VMEM (with fuse_act, one gate and one up tile) and starts each
      weight block's DMA one grid step ahead instead of two. With a single K
      tile (tile_k == size_k) the weight block changes only at group and n-tile
      boundaries. The values are unchanged.

  Returns:
    Output of shape [size_m, size_n] (or [size_m, size_n // 128, 128]); with
    fuse_act the output width is size_n // 2. With return_pre_act, the tuple
    (act, gate, up), each of shape [size_m, size_n // 2] ((act, [gate | up])
    with "concat"). With fuse_dact, [d_gate | d_up] of shape
    [size_m, 2 * size_n], or ([d_gate | d_up], dot) with return_dact_dot.
    Rows outside the computed groups are zero in every output (with
    zero_initialize).
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
      transpose_rhs=transpose_rhs,
      out_is_3d=out_is_3d,
      return_pre_act=return_pre_act,
      fuse_dact=fuse_dact,
      dact_gate_up=dact_gate_up,
      dact_row_scale=dact_row_scale,
      return_dact_dot=return_dact_dot,
      dact_pre_act_tile_n=dact_pre_act_tile_n,
      rhs_buffer_count=rhs_buffer_count,
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
  num_lanes = pltpu.get_tpu_info().num_lanes
  if cfgs.out_window:
    partial_out_shape = (pl.cdiv(cfgs.out_size_n, tiles.tile_n), dims.size_lhs_sublane, tiles.tile_n)
  elif cfgs.out_is_3d:
    partial_out_shape = (dims.size_lhs_sublane, cfgs.out_tile_d0, num_lanes)
  else:
    partial_out_shape = (dims.size_lhs_sublane, cfgs.out_tile_n)
  scratch_shapes = [
      # partial_out_ref
      pltpu.VMEM(partial_out_shape, cfgs.out_dtype),
      # acc_ref
      pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
      # metadata_ref
      MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]

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
    if cfgs.out_is_3d:
      # A row is a whole [D0, 128] slab; the zero buffer holds full slabs,
      # tile_zero_m a multiple of the sublane block. VMEM pads D0 up to the
      # sublane tiling, so budget with the padded size.
      out_d0 = align_to(cfgs.out_size_n, num_lanes) // num_lanes
      sublane = dims.size_lhs_sublane
      sublane_tiling = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
      row_bytes = align_to(out_d0, sublane_tiling) * num_lanes * out_bytes
      tile_zero_m = target_zero_ref_bytes // row_bytes // sublane * sublane
      tile_zero_m = max(sublane, min(tile_zero_m, dims.size_m))
      zero_shape = (tile_zero_m, out_d0, num_lanes)
    else:
      tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
      tile_zero_m = min(tile_zero_m, dims.size_m)
      zero_shape = (tile_zero_m, num_lanes)

    scratch_shapes += [
        pltpu.VMEM(zero_shape, cfgs.out_dtype),
        pltpu.SemaphoreType.DMA((1,)),
    ]
  else:
    scratch_shapes += [None, None]
  if cfgs.return_pre_act == "concat":
    # kernel_main_single_extra_out: the partial-row carry of [gate | up]; it zero-fills with zero_ref (same dtype).
    scratch_shapes += [pltpu.VMEM((dims.size_lhs_sublane, 2 * tiles.tile_n), cfgs.out_dtype), None]
  elif cfgs.return_pre_act:
    # Partial-row carries of the gate and up outputs (kernel_main_pre_act takes them last).
    scratch_shapes += [pltpu.VMEM(partial_out_shape, cfgs.out_dtype) for _ in range(2)]
  elif cfgs.return_dact_dot:
    # kernel_main_single_extra_out: the partial-row carry of the float32 dot, and a float32 zero buffer for its
    # zero-init DMAs.
    dot_zero_tile_m = max(
        dims.size_lhs_sublane,
        (min(256, dims.size_m) // dims.size_lhs_sublane) * dims.size_lhs_sublane,
    )
    dot_zero_scratch = pltpu.VMEM((dot_zero_tile_m, num_lanes), cfgs.acc_dtype) if cfgs.zero_init else None
    scratch_shapes += [pltpu.VMEM((dims.size_lhs_sublane, num_lanes), cfgs.acc_dtype), dot_zero_scratch]

  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  if cfgs.out_is_3d:
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n // num_lanes, num_lanes), cfgs.out_dtype)
  else:
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
  lhs_in = LhsRef(value=lhs, scale=lhs_scale)
  rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)
  partial_sum_spec = None
  ps_arg = partial_sum
  if partial_sum is not None:
    partial_sum_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  elif cfgs.fuse_dact is not None:
    # The fuse_dact inputs ride in the partial_sum slot (DactInputRef); the per-row scale is broadcast to a lane row
    # per row so that its blocks are DMA-able like the output's.
    row_scale_arg = None
    row_scale_spec = None
    if dact_row_scale is not None:
      row_scale_arg = jnp.broadcast_to(
          dact_row_scale.astype(cfgs.acc_dtype).reshape(dims.size_m, -1),
          (dims.size_m, num_lanes),
      )
      row_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    ps_arg = DactInputRef(gate_up=dact_gate_up, row_scale=row_scale_arg)
    partial_sum_spec = DactInputRef(
        gate_up=pl.BlockSpec(memory_space=pltpu.HBM),
        row_scale=row_scale_spec,
    )

  input_output_aliases = {}
  if partial_sum is not None:
    flat_args_preceding = (group_sizes, group_offset, lhs_in, rhs_weights)
    leaves = jax.tree_util.tree_leaves(flat_args_preceding)
    partial_sum_idx = sum(1 for x in leaves if x is not None)
    input_output_aliases = {partial_sum_idx: 0}

  out_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  if cfgs.return_pre_act == "concat":
    kernel_fn = kernel_main_single_extra_out
    concat_aligned_n = align_to(dims.size_n, num_lanes)
    pallas_out_shape = (out_init, jax.ShapeDtypeStruct((dims.size_m, concat_aligned_n), cfgs.out_dtype))
    pallas_out_specs = (out_spec, out_spec)
  elif cfgs.return_pre_act:
    kernel_fn = kernel_main_pre_act
    pallas_out_shape = (out_init,) * 3
    pallas_out_specs = (out_spec,) * 3
  elif cfgs.return_dact_dot:
    kernel_fn = kernel_main_single_extra_out
    pallas_out_shape = (out_init, jax.ShapeDtypeStruct((dims.size_m, num_lanes), cfgs.acc_dtype))
    pallas_out_specs = (out_spec, out_spec)
  else:
    kernel_fn = kernel_main
    pallas_out_shape = out_init
    pallas_out_specs = out_spec

  out = pl.pallas_call(
      functools.partial(kernel_fn, cfgs=cfgs),
      out_shape=pallas_out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              LhsRef(
                  value=pl.BlockSpec(memory_space=pltpu.HBM),
                  scale=lhs_scale_spec,
              ),
              WeightsRef(
                  weight=pl.BlockSpec(memory_space=pltpu.HBM),
                  scale=rhs_scale_spec,
                  bias=rhs_bias_spec,
              ),
              partial_sum_spec,
          ],
          out_specs=pallas_out_specs,
          scratch_shapes=scratch_shapes,  # pyrefly: ignore[bad-argument-type]
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=get_scope_name(cfgs),
      cost_estimate=get_cost_estimate(cfgs),
      metadata=get_metadata(cfgs),  # pyrefly: ignore[bad-argument-type]
      input_output_aliases=input_output_aliases,
  )(group_sizes, group_offset, lhs_in, rhs_weights, ps_arg)
  if cfgs.return_pre_act == "concat":
    return out[0][:, : cfgs.out_size_n], out[1][:, : dims.size_n]
  if cfgs.return_pre_act:
    return tuple(o[:, : cfgs.out_size_n] for o in out)
  if cfgs.return_dact_dot:
    return out[0][:, : cfgs.out_size_n], out[1][:, 0]
  if cfgs.out_is_3d:
    return out
  return out[:, : cfgs.out_size_n]
