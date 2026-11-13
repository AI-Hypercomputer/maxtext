# Copyright 2023–2026 Google LLC
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

"""Pure grouped matrix multiplication (GMM) kernel backends.

Each backend encapsulates only kernel execution mechanics (tiling, layout,
VMA/axis annotations, and low-level kernel arguments) and intentionally knows
nothing about quantization rules or configuration policies.
"""

# pylint: disable=too-many-positional-arguments

import abc
import dataclasses
import random
from typing import ClassVar

import jax
from jax.experimental import xla_metadata
import jax.numpy as jnp
import qwix
import qwix.pallas as qpl

set_xla_metadata = xla_metadata.set_xla_metadata

DLHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [2]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)

DRHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


def scale_like(scale: jnp.ndarray, x: jnp.ndarray, dtype) -> jnp.ndarray:
  """Per-tensor scale -> scalar; per-row scale [m, 1, ...] -> [m, 1, ..., 1] broadcastable to x."""
  if scale.size == 1:
    return scale.reshape(()).astype(dtype)
  if scale.ndim != x.ndim and scale.size == scale.shape[0]:
    # Per-row scale whose rank may differ from x (3D [m, 1, 1] vs 2D [m, k] or vice versa).
    return scale.reshape(scale.shape[0], *([1] * (x.ndim - 1))).astype(dtype)
  return scale.astype(dtype)


def _import_tokamax():
  # Imported lazily so environments without tokamax (or without xprof) do not fail at import time.
  import tokamax  # pylint: disable=import-outside-toplevel

  return tokamax


def _import_megablox_backend():
  from maxtext.kernels.megablox import backend as megablox_backend  # pylint: disable=import-outside-toplevel

  return megablox_backend


def _import_gmm_v2():
  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
  # pylint: enable=import-outside-toplevel

  return gmm_v2


def _import_tgmm_v2():
  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
  # pylint: enable=import-outside-toplevel

  return tgmm_v2


def _to_ragged_dot_array(x: jnp.ndarray | qpl.QArray, dtype: jax.typing.DTypeLike) -> jax.Array:
  """Converts a QArray or quantized array into an XLA ragged_dot-compatible float array."""
  if isinstance(x, qpl.QArray):
    qarr = (
        qpl.QArray(x.qvalue, x.scale.reshape((1,) * x.qvalue.ndim), x.zero_point, x.qtype)
        if x.scale.size == 1 and x.scale.ndim != x.qvalue.ndim
        else x
    )
    arr = qwix.dequantize(qarr)
  else:
    arr = x
  if jax.dtypes.itemsize_bits(arr.dtype) < 16:
    target_dtype = jnp.float32 if jnp.dtype(dtype).itemsize < 2 else dtype
    return arr.astype(target_dtype)
  return arr


def _prepare_rhs_scale_for_v2(rhs: qpl.QArray) -> jnp.ndarray:
  """Formats and broadcasts rhs scale to (size_group, num_quant_blocks, 1, size_n) for GMM v2."""
  g, _, n = rhs.qvalue.shape
  scale = rhs.scale

  if scale.ndim == 2:  # Per-channel quantization
    rhs_scale = jnp.expand_dims(scale, axis=(1, 2))
  elif scale.ndim == 3:  # Block-wise quantization
    rhs_scale = jnp.expand_dims(scale, axis=2)
  else:  # Per-tensor quantization, (1, 1, 1, 1)
    rhs_scale = scale

  num_quant_blocks = rhs_scale.shape[1] if rhs_scale.ndim > 1 else 1
  return jnp.broadcast_to(rhs_scale, (g, num_quant_blocks, 1, n))


def _prepare_bwd_scale_for_v2(drhs_dout: qpl.QArray) -> jnp.ndarray:
  """Formats and broadcasts drhs_dout scale to (1, 1, size_n) for V2 TGMM kernel."""
  scale = drhs_dout.scale
  size_n = drhs_dout.qvalue.size // drhs_dout.qvalue.shape[0]
  if drhs_dout.qvalue.ndim == 3 or scale.size == 1:
    # 3D grad [m, n // 128, 128]: scale is (1, 1, 1) per-tensor or (1, n // 128, 128) per-channel.
    return jnp.broadcast_to(scale.reshape(1, 1, -1), (1, 1, size_n))
  # per channel: (1, n) -> (1, 1, n)
  # per tensor: (1, 1) -> (1, 1, 1)
  rhs_scale = jnp.expand_dims(scale, axis=1)
  if rhs_scale.shape[2] == 1:
    rhs_scale = jnp.broadcast_to(rhs_scale, (1, 1, size_n))
  return rhs_scale


@dataclasses.dataclass(frozen=True)
class GmmBackend(abc.ABC):
  """Abstract base class for grouped matrix multiplication kernel backends.

  Subclasses are immutable, hashable dataclasses so they can be passed cleanly
  through JAX `nondiff_argnums` and `jit` static arguments.
  """

  name: ClassVar[str]
  requires_custom_vjp: ClassVar[bool] = False
  supports_qag: ClassVar[bool] = False
  supports_3d_lhs: ClassVar[bool] = False
  supports_partial_sum: ClassVar[bool] = False
  supports_tc_unsort: ClassVar[bool] = False
  supports_flat_fsdp_shard: ClassVar[bool] = False
  supports_bias: ClassVar[bool] = False
  supports_native_lhs_quant: ClassVar[bool] = False
  dlhs_needs_repeated_rhs_scale: ClassVar[bool] = False
  supports_native_qwix_interception: ClassVar[bool] = False
  accumulate_existing_out_in_bwd: ClassVar[bool] = False

  def call(
      self,
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike = jnp.float32,
      tiling: tuple = (128, 128, 128),
      group_offset: jnp.ndarray | None = None,
      bias: jnp.ndarray | None = None,
      partial_sum: jnp.ndarray | None = None,
      use_full_k_tiling: bool = False,
      use_rpa_group_sizes: bool = False,
  ) -> jax.Array:
    """Executes the backend directly using native JAX autodiff when supported."""
    del lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset, bias, partial_sum, use_full_k_tiling
    del use_rpa_group_sizes
    if self.requires_custom_vjp:
      raise NotImplementedError(f"Backend {self.name!r} requires custom VJP execution.")
    raise NotImplementedError(f"Backend {self.name!r} does not implement call().")

  @abc.abstractmethod
  def fwd(
      self,
      lhs: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      existing_out: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> jax.Array:
    """Executes forward grouped matmul OUT = LHS @ RHS inside custom VJP."""

  @abc.abstractmethod
  def dlhs(
      self,
      dlhs_dout: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> tuple[jax.Array, bool]:
    """Executes DLHS = DLHS_dout @ RHS^T, returning (dlhs, lhs_scale_applied_in_kernel)."""

  @abc.abstractmethod
  def drhs(
      self,
      lhs: jnp.ndarray,
      drhs_dout: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      num_actual_groups: int | None = None,
      interpret: bool = False,
      rhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
  ) -> jax.Array:
    """Executes DRHS = LHS^T @ DRHS_dout inside custom VJP."""


@dataclasses.dataclass(frozen=True)
class JaxRaggedDotBackend(GmmBackend):
  """XLA / JAX `jax.lax.ragged_dot` backend (works on CPU, GPU, and TPU)."""

  name: ClassVar[str] = "jax_ragged_dot"
  requires_custom_vjp: ClassVar[bool] = False
  supports_native_qwix_interception: ClassVar[bool] = True
  accumulate_existing_out_in_bwd: ClassVar[bool] = True

  is_tpu: bool = False

  def call(
      self,
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike = jnp.float32,
      tiling: tuple = (128, 128, 128),
      group_offset: jnp.ndarray | None = None,
      bias: jnp.ndarray | None = None,
      partial_sum: jnp.ndarray | None = None,
      use_full_k_tiling: bool = False,
      use_rpa_group_sizes: bool = False,
  ) -> jax.Array:
    del group_offset, bias, partial_sum, use_rpa_group_sizes
    m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]
    tm = tiling[0] if tiling[0] is not None else 128
    tk = tiling[1] if tiling[1] is not None else 128
    tn = tiling[2] if tiling[2] is not None else 128
    clamped_tiling = (min(tm, m), min(tk, k), min(tn, n))
    if use_full_k_tiling:
      # Use full contraction for Qwix interception to allow quantization fusion
      # (max reduce over contracting dimension).
      clamped_tiling = (clamped_tiling[0], k, clamped_tiling[2])

    # TPU needs random mosaic_fusion_group; GPU/CPU needs deterministic ID for autotuner sync.
    mosaic_group_id = f"{random.randint(0, 1000000000)}" if self.is_tpu else "0"
    with set_xla_metadata(
        ragged_dot_tiling=",".join(str(t) for t in clamped_tiling),
        mosaic_fusion_group=mosaic_group_id,
    ):
      return jax.lax.ragged_dot(
          lhs=lhs,
          rhs=rhs,
          group_sizes=group_sizes,
          preferred_element_type=preferred_element_type,
      )

  def fwd(
      self,
      lhs: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      existing_out: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> jax.Array:
    del tiling, interpret, lhs_vma_axes, use_manual_quantization, partial_sum, lhs_scale, out_is_3d
    lhs_arr = _to_ragged_dot_array(lhs, preferred_element_type)
    rhs_arr = _to_ragged_dot_array(rhs, preferred_element_type)
    if lhs_arr.dtype != rhs_arr.dtype:
      common_dtype = jnp.promote_types(lhs_arr.dtype, rhs_arr.dtype)
      lhs_arr = lhs_arr.astype(common_dtype)
      rhs_arr = rhs_arr.astype(common_dtype)
    out = jax.lax.ragged_dot(
        lhs=lhs_arr,
        rhs=rhs_arr,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
    if existing_out is not None:
      out = existing_out.astype(out.dtype) + out
    return out

  def dlhs(
      self,
      dlhs_dout: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> tuple[jax.Array, bool]:
    del tiling, interpret, lhs_vma_axes, use_manual_quantization, partial_sum, lhs_scale, out_is_3d
    dout_arr = _to_ragged_dot_array(dlhs_dout, preferred_element_type)
    rhs_arr = _to_ragged_dot_array(rhs, preferred_element_type)
    if dout_arr.dtype != rhs_arr.dtype:
      common_dtype = jnp.promote_types(dout_arr.dtype, rhs_arr.dtype)
      dout_arr = dout_arr.astype(common_dtype)
      rhs_arr = rhs_arr.astype(common_dtype)
    dlhs_val = jax.lax.ragged_dot_general(
        lhs=dout_arr,
        rhs=rhs_arr,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=DLHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
    return dlhs_val, False

  def drhs(
      self,
      lhs: jnp.ndarray,
      drhs_dout: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      num_actual_groups: int | None = None,
      interpret: bool = False,
      rhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
  ) -> jax.Array:
    del tiling, interpret, rhs_vma_axes, use_manual_quantization, partial_sum
    lhs_arr = _to_ragged_dot_array(lhs, preferred_element_type)
    dout_arr = _to_ragged_dot_array(drhs_dout, preferred_element_type)
    if lhs_arr.dtype != dout_arr.dtype:
      common_dtype = jnp.promote_types(lhs_arr.dtype, dout_arr.dtype)
      lhs_arr = lhs_arr.astype(common_dtype)
      dout_arr = dout_arr.astype(common_dtype)
    drhs_val = jax.lax.ragged_dot_general(
        lhs=lhs_arr,
        rhs=dout_arr,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=DRHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=preferred_element_type,
    )
    if num_actual_groups is not None and num_actual_groups < drhs_val.shape[0]:
      start = 0 if group_offset is None else group_offset
      drhs_val = jax.lax.dynamic_slice_in_dim(drhs_val, start, num_actual_groups, axis=0)
    return drhs_val


@dataclasses.dataclass(frozen=True)
class MegabloxBackend(GmmBackend):
  """Pure MaxText Pallas Mosaic TPU Megablox grouped matmul backend."""

  name: ClassVar[str] = "megablox"
  requires_custom_vjp: ClassVar[bool] = True
  accumulate_existing_out_in_bwd: ClassVar[bool] = True

  def fwd(
      self,
      lhs: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      existing_out: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> jax.Array:
    del use_manual_quantization, partial_sum, lhs_scale, out_is_3d
    megablox_backend = _import_megablox_backend()
    out = megablox_backend.gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling[:3],
        group_offset,
        existing_out,
        interpret=interpret,
    )
    for axis in lhs_vma_axes:
      out = jax.lax.pcast(out, axis_name=axis, to="varying")
    return out

  def dlhs(
      self,
      dlhs_dout: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> tuple[jax.Array, bool]:
    del use_manual_quantization, partial_sum, lhs_scale, out_is_3d
    megablox_backend = _import_megablox_backend()
    dlhs_val = megablox_backend.gmm(
        dlhs_dout,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling[3:6],
        group_offset,
        transpose_rhs=True,
        interpret=interpret,
        varying_axes=lhs_vma_axes,
    )
    return dlhs_val, False

  def drhs(
      self,
      lhs: jnp.ndarray,
      drhs_dout: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      num_actual_groups: int | None = None,
      interpret: bool = False,
      rhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
  ) -> jax.Array:
    del use_manual_quantization, partial_sum
    megablox_backend = _import_megablox_backend()
    return megablox_backend.tgmm(
        lhs.swapaxes(0, 1),
        drhs_dout,
        group_sizes,
        preferred_element_type,
        tiling[-3:],
        group_offset,
        num_actual_groups,
        interpret=interpret,
        varying_axes=rhs_vma_axes,
    )


@dataclasses.dataclass(frozen=True)
class TokamaxV1Backend(GmmBackend):
  """Tokamax `ragged_dot` v1 backend (supports both native autodiff and quantized custom VJP)."""

  name: ClassVar[str] = "tokamax_v1"
  requires_custom_vjp: ClassVar[bool] = False
  supports_qag: ClassVar[bool] = True

  def call(
      self,
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike = jnp.float32,
      tiling: tuple = (128, 128, 128),
      group_offset: jnp.ndarray | None = None,
      bias: jnp.ndarray | None = None,
      partial_sum: jnp.ndarray | None = None,
      use_full_k_tiling: bool = False,
      use_rpa_group_sizes: bool = False,
  ) -> jax.Array:
    del tiling, group_offset, bias, partial_sum, use_full_k_tiling
    tokamax = _import_tokamax()
    tokamax_group_sizes = group_sizes if use_rpa_group_sizes else tokamax.RaggedDotGroupSizes(group_sizes, lhs.shape[0])
    return tokamax.ragged_dot(
        lhs=lhs,
        rhs=rhs,
        group_sizes=tokamax_group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        implementation="mosaic",
        # `group_offset` is not yet supported
        group_offset=None,
    )

  def fwd(
      self,
      lhs: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      existing_out: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> jax.Array:
    del tiling, group_offset, existing_out, interpret, lhs_vma_axes, partial_sum, lhs_scale, out_is_3d
    tokamax = _import_tokamax()
    out_kwargs = {}
    if use_manual_quantization:
      # Used in batchsplit: manual_axis_type for gmm with shard_map check_vma=True.
      out_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(varying=frozenset(["data", "fsdp", "expert"]))
    return tokamax.ragged_dot(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        group_offset=None,
        implementation="mosaic",
        **out_kwargs,
    )

  def dlhs(
      self,
      dlhs_dout: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> tuple[jax.Array, bool]:
    del tiling, group_offset, interpret, lhs_vma_axes, partial_sum, lhs_scale, out_is_3d
    tokamax = _import_tokamax()
    dlhs_kwargs = {}
    if use_manual_quantization:
      dlhs_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(varying=frozenset(["data", "fsdp", "expert"]))
    dlhs_val = tokamax.ragged_dot_general(
        lhs=dlhs_dout,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=DLHS_RAGGED_DOT_DIM_NUMS,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        group_offset=None,
        implementation="mosaic",
        **dlhs_kwargs,
    )
    return dlhs_val, False

  def drhs(
      self,
      lhs: jnp.ndarray,
      drhs_dout: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      num_actual_groups: int | None = None,
      interpret: bool = False,
      rhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
  ) -> jax.Array:
    del tiling, group_offset, num_actual_groups, interpret, rhs_vma_axes, partial_sum
    tokamax = _import_tokamax()
    drhs_kwargs = {}
    if use_manual_quantization:
      # Per-device result is a partial sum over data/fsdp; declare varying then cast to unreduced.
      drhs_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(varying=frozenset(["data", "fsdp", "expert"]))
    drhs_val = tokamax.ragged_dot_general(
        lhs=lhs,
        rhs=drhs_dout,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=DRHS_RAGGED_DOT_DIM_NUMS,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        group_offset=None,
        implementation="mosaic",
        **drhs_kwargs,
    )
    if use_manual_quantization:
      drhs_val = jax.lax.pcast(drhs_val, ("data", "fsdp"), to="unreduced")
    return drhs_val


@dataclasses.dataclass(frozen=True)
class TokamaxV2Backend(GmmBackend):
  """Pallas Mosaic TPU v2 GMM / TGMM backend."""

  name: ClassVar[str] = "tokamax_v2"
  requires_custom_vjp: ClassVar[bool] = True
  supports_qag: ClassVar[bool] = True
  supports_3d_lhs: ClassVar[bool] = True
  supports_partial_sum: ClassVar[bool] = True
  supports_tc_unsort: ClassVar[bool] = True
  supports_flat_fsdp_shard: ClassVar[bool] = True
  supports_native_lhs_quant: ClassVar[bool] = True
  dlhs_needs_repeated_rhs_scale: ClassVar[bool] = True

  heuristic_tiling: bool = False
  dlhs_transpose_rhs: bool = False
  fuse_dlhs_scale: bool = False
  fuse_3d_dlhs_ps: bool = False

  def fwd(
      self,
      lhs: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      existing_out: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> jax.Array:
    del existing_out, interpret, lhs_vma_axes, use_manual_quantization
    rhs_operand = rhs
    rhs_scale = None
    if isinstance(rhs, qpl.QArray):
      rhs_operand = rhs.qvalue
      rhs_scale = _prepare_rhs_scale_for_v2(rhs)

    if isinstance(lhs, qpl.QArray):
      if partial_sum is not None:
        raise NotImplementedError(
            "partial_sum is not supported with pre-quantized lhs QArray because"
            " scaling output by lhs.scale would erroneously rescale partial_sum."
        )
      lhs_operand = lhs.qvalue
      effective_lhs_scale = None
      maybe_quantize_lhs = False
    else:
      lhs_operand = lhs
      effective_lhs_scale = lhs_scale
      maybe_quantize_lhs = True

    gmm_v2 = _import_gmm_v2()
    if self.heuristic_tiling:
      fwd_tiling = gmm_v2.calculate_tiling
    else:
      fwd_tiling = gmm_v2.TileSizes(tile_m=tiling[0], tile_k=tiling[1], tile_n=tiling[2])

    out = gmm_v2.gmm_v2(
        lhs=lhs_operand,
        rhs=rhs_operand,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        tile_info=fwd_tiling,
        preferred_element_type=preferred_element_type,
        partial_sum=partial_sum,
        group_offset=group_offset,
        lhs_scale=effective_lhs_scale,
        maybe_quantize_lhs=maybe_quantize_lhs,
        out_is_3d=out_is_3d,
    )
    if isinstance(lhs, qpl.QArray):
      out = out * scale_like(lhs.scale, out, out.dtype)
    return out

  def dlhs(
      self,
      dlhs_dout: jnp.ndarray | qpl.QArray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      interpret: bool = False,
      lhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
      lhs_scale: jax.Array | None = None,
      out_is_3d: bool = False,
  ) -> tuple[jax.Array, bool]:
    del interpret, lhs_vma_axes, use_manual_quantization
    gmm_v2 = _import_gmm_v2()
    is_qarray_dout = isinstance(dlhs_dout, qpl.QArray)
    dlhs_lhs = dlhs_dout.qvalue if is_qarray_dout else dlhs_dout

    if self.heuristic_tiling:
      dlhs_tiling = gmm_v2.calculate_tiling
    else:
      dlhs_tiling = gmm_v2.TileSizes(tile_m=tiling[3], tile_k=tiling[4], tile_n=tiling[5])

    # Sub-byte rhs always takes rhs^T, since gmm_v2 rejects transpose_rhs for sub-byte dtypes.
    kernel_transpose_rhs = self.dlhs_transpose_rhs and jax.dtypes.itemsize_bits(rhs.dtype) >= 8
    dlhs_rhs = rhs if kernel_transpose_rhs else rhs.swapaxes(1, 2)

    scales = [s for s in (dlhs_dout.scale if is_qarray_dout else None, lhs_scale) if s is not None]
    can_fuse_scale = (self.fuse_dlhs_scale or partial_sum is not None) and all(s.size == 1 for s in scales)
    kernel_lhs_scale = None
    kernel_partial_sum = None
    lhs_scale_applied = False
    if can_fuse_scale:
      for scale in scales:
        scale = scale.astype(jnp.float32).reshape(1, 1)
        kernel_lhs_scale = scale if kernel_lhs_scale is None else kernel_lhs_scale * scale
      lhs_scale_applied = lhs_scale is not None
      kernel_partial_sum = partial_sum

    dlhs_val = gmm_v2.gmm_v2(
        lhs=dlhs_lhs,
        rhs=dlhs_rhs,
        group_sizes=group_sizes,
        rhs_scale=None,
        partial_sum=kernel_partial_sum,
        tile_info=dlhs_tiling,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        lhs_scale=kernel_lhs_scale,
        transpose_rhs=kernel_transpose_rhs,
        maybe_quantize_lhs=not is_qarray_dout and kernel_lhs_scale is None,
        zero_initialize=not (self.fuse_3d_dlhs_ps and out_is_3d and kernel_partial_sum is not None),
        out_is_3d=out_is_3d,
    )

    if not can_fuse_scale:
      if is_qarray_dout:
        dlhs_val = dlhs_val * scale_like(dlhs_dout.scale, dlhs_val, dlhs_val.dtype)
      if partial_sum is not None:
        if lhs_scale is not None:
          dlhs_val = dlhs_val * scale_like(lhs_scale, dlhs_val, dlhs_val.dtype)
          lhs_scale_applied = True
        dlhs_val = dlhs_val + partial_sum

    return dlhs_val, lhs_scale_applied

  def drhs(
      self,
      lhs: jnp.ndarray,
      drhs_dout: jnp.ndarray | qpl.QArray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike,
      tiling: tuple,
      group_offset: jnp.ndarray | None = None,
      num_actual_groups: int | None = None,
      interpret: bool = False,
      rhs_vma_axes: tuple = (),
      use_manual_quantization: bool = False,
      partial_sum: jnp.ndarray | None = None,
  ) -> jax.Array:
    del interpret, rhs_vma_axes, use_manual_quantization
    gmm_v2 = _import_gmm_v2()
    tgmm_v2 = _import_tgmm_v2()
    drhs_rhs = drhs_dout.qvalue if isinstance(drhs_dout, qpl.QArray) else drhs_dout
    rhs_scale = _prepare_bwd_scale_for_v2(drhs_dout) if isinstance(drhs_dout, qpl.QArray) else None

    if self.heuristic_tiling:
      drhs_tiling = tgmm_v2.calculate_tgmm_tiling
    else:
      drhs_tiling = gmm_v2.TileSizes(tile_m=tiling[6], tile_k=tiling[7], tile_n=tiling[8])

    return tgmm_v2.tgmm_v2(
        lhs=lhs,
        rhs=drhs_rhs,
        group_sizes=group_sizes,
        num_actual_groups=num_actual_groups,
        rhs_scale=rhs_scale,
        partial_sum=partial_sum,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        tile_info=drhs_tiling,
    )


@dataclasses.dataclass(frozen=True)
class TeGroupedGemmBackend(GmmBackend):
  """GPU TransformerEngine `grouped_dense` backend (native custom VJP & bias)."""

  name: ClassVar[str] = "te_grouped_gemm"
  requires_custom_vjp: ClassVar[bool] = False
  supports_bias: ClassVar[bool] = True
  supports_partial_sum: ClassVar[bool] = True

  def call(
      self,
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      group_sizes: jnp.ndarray,
      *,
      preferred_element_type: jax.typing.DTypeLike = jnp.float32,
      tiling: tuple = (128, 128, 128),
      group_offset: jnp.ndarray | None = None,
      bias: jnp.ndarray | None = None,
      partial_sum: jnp.ndarray | None = None,
      use_full_k_tiling: bool = False,
      use_rpa_group_sizes: bool = False,
  ) -> jax.Array:
    del tiling, use_full_k_tiling, use_rpa_group_sizes
    from transformer_engine.jax.dense import grouped_dense  # pylint: disable=import-outside-toplevel

    num_local_experts = rhs.shape[0]
    start = 0 if group_offset is None else group_offset
    te_group_sizes = jax.lax.dynamic_slice_in_dim(group_sizes, start, num_local_experts)
    row_start = None
    if not (isinstance(start, int) and start == 0):
      row_start = jnp.sum(jnp.where(jnp.arange(group_sizes.shape[0]) < start, group_sizes, 0))
      lhs = jnp.roll(lhs, -row_start, axis=0)
    out_dtype = jnp.dtype(preferred_element_type)
    compute_dtype = min(
        (jnp.dtype(rhs.dtype), jnp.dtype(lhs.dtype), out_dtype),
        key=lambda dt: dt.itemsize,
    )
    lhs = jax.lax.convert_element_type(lhs, compute_dtype)
    rhs = jax.lax.convert_element_type(rhs, compute_dtype)
    if bias is not None:
      output = grouped_dense(
          lhs,
          rhs,
          te_group_sizes.astype(jnp.int32),
          contracting_dims=((1,), (1,)),
          bias=jax.lax.convert_element_type(bias, compute_dtype),
          preferred_element_type=compute_dtype,
      )
    else:
      output = grouped_dense(
          lhs,
          rhs,
          te_group_sizes.astype(jnp.int32),
          contracting_dims=((1,), (1,)),
          preferred_element_type=compute_dtype,
      )
    if row_start is not None:
      output = jnp.roll(output, row_start, axis=0)
    output = jax.lax.convert_element_type(output, out_dtype)
    if partial_sum is not None:
      output = output + partial_sum
    return output

  def fwd(self, *args, **kwargs):
    raise NotImplementedError("TeGroupedGemmBackend uses direct native custom VJP via call().")

  def dlhs(self, *args, **kwargs):
    raise NotImplementedError("TeGroupedGemmBackend uses direct native custom VJP via call().")

  def drhs(self, *args, **kwargs):
    raise NotImplementedError("TeGroupedGemmBackend uses direct native custom VJP via call().")


def select_gmm_backend(
    *,
    use_te_grouped_gemm: bool = False,
    use_tokamax_gmm: bool = False,
    use_gmm_v2: bool = False,
    use_gmm_v2_heuristic_tiling: bool = False,
    megablox: bool = False,
    hardware: str = "tpu",
    is_tpu: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    fuse_dlhs_scale: bool = False,
    fuse_3d_dlhs_ps: bool = False,
) -> GmmBackend:
  """Selects the grouped-matmul kernel backend solely from hardware & backend config."""
  if use_te_grouped_gemm:
    if use_tokamax_gmm:
      raise ValueError("use_te_grouped_gemm=True is incompatible with use_tokamax_gmm=True.")
    if megablox:
      raise ValueError("use_te_grouped_gemm=True is incompatible with megablox=True.")
    return TeGroupedGemmBackend()
  if hardware == "cpu":
    return JaxRaggedDotBackend(is_tpu=is_tpu)
  if use_gmm_v2 and not use_tokamax_gmm:
    raise ValueError("use_gmm_v2=True requires use_tokamax_gmm=True.")
  if use_tokamax_gmm:
    if use_gmm_v2:
      return TokamaxV2Backend(
          heuristic_tiling=use_gmm_v2_heuristic_tiling,
          dlhs_transpose_rhs=use_dlhs_transpose_rhs,
          fuse_dlhs_scale=fuse_dlhs_scale,
          fuse_3d_dlhs_ps=fuse_3d_dlhs_ps,
      )
    return TokamaxV1Backend()
  if megablox:
    return MegabloxBackend()
  return JaxRaggedDotBackend(is_tpu=is_tpu)
