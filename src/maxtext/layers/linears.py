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

"""Linear Layers."""

import dataclasses
import functools
import math
import operator
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from jax import lax
from jax.experimental.compute_on import compute_on
from jax.sharding import NamedSharding, Mesh, PartitionSpec
from jax.ad_checkpoint import checkpoint_name

from flax import nnx

from maxtext.common.common_types import DecoderBlockType, ShardMode, DType, Array, Config, Shape, is_fp8_dtype
from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.layers import nnx_wrappers, quantizations
from maxtext.layers import normalizations
from maxtext.layers.initializers import NdInitializer, nd_dense_init, default_bias_init, variable_to_logically_partitioned, Initializer
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import maybe_shard_with_logical
from maxtext.utils.sharding import maybe_shard_with_name
from maxtext.utils.sharding import get_physical_spec_without_axes
from maxtext.utils.sharding import FSDP_MESH_AXES
from maxtext.utils.sharding import truncate_out_sharding
from maxtext.utils.sharding import logical_to_mesh_sharding


# ---------------------------------------------------------------------------------------------------------------------
# FSDP DenseGeneral matmul in shard_map with an explicit weight-gradient reduce-scatter
# (config: dense_fsdp_shard_map_dot and dense_wgrad_rs_*; see configs/types.py).
#
# Under GSPMD the per-layer weight gradient of an FSDP-sharded kernel that is used by batch-sharded activations is
# partial over every data-parallel device (fsdp x expert for ep-as-dp), while the kernel is sharded over fsdp only.
# XLA then emits a full all-reduce over all devices followed by a dynamic-slice (instead of a reduce-scatter), and
# the all-reduce is exposed at the end of each layer's backward. Doing the matmul inside shard_map with an explicit
# all_gather of the kernel makes the transpose a psum_scatter over the FSDP axes plus a small psum of the 1/FSDP
# shard over the remaining data axes. The backward reduce-scatter is additionally
#   * issued on a [num_fsdp_shards, rows_per_shard, ...] view so the scatter dimension is the major, tile-aligned
#     dimension (XLA's TPU reduce-scatter decomposer rewrites reduce-scatters whose scatter dimension is not major,
#     or whose shards are not tile-aligned, back into all-reduce + dynamic-slice), and
#   * optionally pinned to one SparseCore (dense_wgrad_rs_sparse_core_id) so it does not queue behind the FSDP
#     weight all-gather prefetch on the other SparseCore.
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class DenseWgradReduceScatterConfig:
  """Settings for the shard_map FSDP dot; set once per model by `configure_dense_wgrad_reduce_scatter`."""

  enabled: bool = False
  mesh: Mesh | None = None
  # With qwix fp8_full and a fixed weight calibration ('fixed,-a,a') the kernel shard is quantized with the same
  # fixed scale before the all-gather, so the gather moves fp8 bytes as in the GSPMD path.
  fp8_fixed_absmax: float | None = None
  # Kernels with more elements keep the GSPMD path (<= 0: no limit); excludes the vocab head by default.
  max_kernel_elems: int = 0
  # SparseCore id the backward reduce-scatter is pinned to (compute_on); < 0 leaves the placement to XLA.
  sparse_core_id: int = -1
  # Reduce-scatter a [num_shards, rows_per_shard, ...] view of the cotangent so the scatter dimension is major.
  flatten_scatter_dim: bool = True


_DENSE_WGRAD_RS = DenseWgradReduceScatterConfig()


def configure_dense_wgrad_reduce_scatter(config: Config, mesh: Mesh | None) -> None:
  """Configures the FSDP shard_map dot (weight-gradient reduce-scatter) from `config` for all DenseGenerals."""
  global _DENSE_WGRAD_RS  # pylint: disable=global-statement
  fp8_absmax = None
  calibration = str(getattr(config, "weight_quantization_calibration_method", "") or "")
  if (
      getattr(config, "use_qwix_quantization", False)
      and getattr(config, "quantization", None) == "fp8_full"
      and calibration.startswith("fixed")
  ):
    vals = [abs(float(v)) for v in calibration.split(",")[1:]]
    if len(vals) == 1 or (len(vals) == 2 and vals[0] == vals[1]):
      fp8_absmax = vals[-1]
  _DENSE_WGRAD_RS = DenseWgradReduceScatterConfig(
      enabled=bool(getattr(config, "dense_fsdp_shard_map_dot", False)),
      mesh=mesh,
      fp8_fixed_absmax=fp8_absmax,
      max_kernel_elems=int(getattr(config, "dense_fsdp_shard_map_max_kernel_elems", 0)),
      sparse_core_id=int(getattr(config, "dense_wgrad_rs_sparse_core_id", -1)),
      flatten_scatter_dim=bool(getattr(config, "dense_wgrad_rs_flatten_scatter_dim", True)),
  )


def _flat_mesh_axes(spec) -> list[str]:
  out = []
  for dim in spec:
    if dim is None:
      continue
    out.extend(dim if isinstance(dim, (tuple, list)) else (dim,))
  return out


def _wgrad_reduce_scatter(g, axes, dim, num_shards, sparse_core_id, flatten):
  """psum_scatter of the gathered-kernel cotangent `g` over `axes` on `dim` (tiled), i.e. the all_gather transpose.

  With `flatten` the scatter dimension is moved to the front and split into [num_shards, rows_per_shard], so the
  scattered dimension is the major dimension (and tile-aligned) and XLA keeps the collective a reduce-scatter.
  With `sparse_core_id >= 0` the collective is pinned to that SparseCore.
  """
  axis_name = axes if len(axes) > 1 else axes[0]
  scatter_dimension = 0 if flatten else dim

  def _rs(z):
    return jax.lax.psum_scatter(z, axis_name, scatter_dimension=scatter_dimension, tiled=True)

  if sparse_core_id >= 0:
    # compute_on traces its body like a jit, so the collective's static arguments are closed over above.
    _rs = compute_on(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [sparse_core_id]}},
    )(_rs)

  if not flatten:
    return _rs(g)
  # [.., S*r, ..] -> [S, r, ..]: the scattered dimension becomes a leading dimension of its own, so each
  # device's shard [1, r, ..] is a whole number of (8,128) tiles (a 2-D [S, -1] view would scatter inside a
  # tile, which XLA also decomposes into all-reduce + dynamic-slice).
  g = jnp.moveaxis(g, dim, 0)
  shard_shape = (g.shape[0] // num_shards,) + g.shape[1:]
  out = _rs(g.reshape((num_shards,) + shard_shape))
  return jnp.moveaxis(out.reshape(shard_shape), 0, dim)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6))
def _fsdp_all_gather(w, axes, dim, fp8_absmax, num_shards, sparse_core_id, flatten):
  """Tiled all_gather of the kernel shard `w` over `axes` on `dim`; the backward is `_wgrad_reduce_scatter`.

  With `fp8_absmax` the shard is quantized to fp8 (fixed per-tensor scale) before the gather and dequantized after
  it; with a power-of-two scale the dequantized values are exactly the ones qwix re-quantizes with the same fixed
  calibration, so the fp8 matmul operands are unchanged (the backward is a straight-through psum_scatter).
  """
  del num_shards, sparse_core_id, flatten
  axis_name = axes if len(axes) > 1 else axes[0]
  if fp8_absmax is None:
    return jax.lax.all_gather(w, axis_name, axis=dim, tiled=True)
  fmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
  scale = jnp.asarray(fp8_absmax / fmax, w.dtype)
  q = jnp.clip(w / scale, -fmax, fmax).astype(jnp.float8_e4m3fn)
  q = jax.lax.all_gather(q, axis_name, axis=dim, tiled=True)
  return q.astype(w.dtype) * scale


def _fsdp_all_gather_fwd(w, axes, dim, fp8_absmax, num_shards, sparse_core_id, flatten):
  return _fsdp_all_gather(w, axes, dim, fp8_absmax, num_shards, sparse_core_id, flatten), None


def _fsdp_all_gather_bwd(axes, dim, fp8_absmax, num_shards, sparse_core_id, flatten, _, g):
  del fp8_absmax
  return (_wgrad_reduce_scatter(g, axes, dim, num_shards, sparse_core_id, flatten),)


_fsdp_all_gather.defvjp(_fsdp_all_gather_fwd, _fsdp_all_gather_bwd)


def _fsdp_shard_map_dot(inputs, kernel, kernel_axes, norm_axis, matmul_precision, settings):
  """`inputs @ kernel` inside shard_map with an explicit FSDP all-gather of the kernel.

  Returns None when the layout is not supported (the caller then uses the GSPMD path): the kernel must be sharded on
  exactly one dimension, the activations only on their leading (batch/sequence) dimensions, and the kernel's mesh
  axes must be a subset of the activation's.
  """
  mesh = settings.mesh
  k_spec = tuple(logical_to_mesh_sharding(PartitionSpec(*kernel_axes), mesh).spec)
  k_spec = k_spec + (None,) * (kernel.ndim - len(k_spec))
  sharded = [(i, d) for i, d in enumerate(k_spec) if d is not None]
  if len(sharded) != 1:
    return None
  gdim, gaxes = sharded[0]
  gaxes = tuple(gaxes) if isinstance(gaxes, (tuple, list)) else (gaxes,)
  num_shards = math.prod(mesh.shape[a] for a in gaxes)
  if kernel.shape[gdim] % num_shards:
    return None
  n_batch = inputs.ndim - len(norm_axis)
  if n_batch < 1 or tuple(norm_axis) != tuple(range(n_batch, inputs.ndim)):
    return None
  logical_x = ("activation_batch", "activation_norm_length")[: min(n_batch, 2)]
  logical_x = logical_x + (None,) * (inputs.ndim - len(logical_x))
  x_spec = tuple(logical_to_mesh_sharding(PartitionSpec(*logical_x), mesh).spec)
  x_spec = x_spec + (None,) * (inputs.ndim - len(x_spec))
  if any(d is not None for d in x_spec[n_batch:]):
    return None
  x_axes = set(_flat_mesh_axes(x_spec))
  k_axes = set(gaxes)
  if not k_axes or not k_axes <= x_axes:
    return None
  for size, d in zip(inputs.shape, x_spec):
    if size % math.prod(mesh.shape[a] for a in _flat_mesh_axes((d,))):
      return None
  # Data axes the kernel is replicated over: marking the kernel varying over them before the gather makes the
  # transpose a psum_scatter over `gaxes` first and then a psum of the small 1/FSDP shard over `vary`.
  vary = tuple(a for a in mesh.axis_names if a in x_axes and a not in k_axes)
  out_spec = PartitionSpec(*(x_spec[:n_batch] + (None,) * (kernel.ndim - len(norm_axis))))
  contract = (tuple(range(n_batch, inputs.ndim)), tuple(range(len(norm_axis))))
  precision = lax.Precision(matmul_precision)
  fp8_absmax = settings.fp8_fixed_absmax
  sparse_core_id, flatten = settings.sparse_core_id, settings.flatten_scatter_dim

  def _body(x, w):
    if vary:
      w = jax.lax.pcast(w, axis_name=vary, to="varying")
    w = _fsdp_all_gather(w, gaxes, gdim, fp8_absmax, num_shards, sparse_core_id, flatten)
    return lax.dot_general(x, w, (contract, ((), ())), precision=precision)

  return jax.shard_map(_body, mesh=mesh, in_specs=(PartitionSpec(*x_spec), PartitionSpec(*k_spec)), out_specs=out_spec)(
      inputs, kernel
  )


def _convert_to_activation_function(fn_or_string: str | Callable[..., Any]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == "linear":
    return lambda x: x
  elif fn_or_string == "sqrtsoftplus":
    # Custom activation function used by DeepSeek V4 Top-K MoE router
    return lambda x: jnp.sqrt(jax.nn.softplus(x))
  elif isinstance(fn_or_string, str):
    return getattr(jax.nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def normalize_axes(axes: Iterable[int], ndim: int) -> tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


# Re-export dequantize_weight and WeightQuantConfig from quantizations for backward compatibility.
dequantize_weight = quantizations.dequantize_weight
WeightQuantConfig = quantizations.WeightQuantConfig


def _compute_dot_general(
    inputs,
    kernel,
    kernel_axes,
    axis,
    contract_ind,
    matmul_precision,
    quant,
    kernel_scale: Array | None = None,
    compute_dtype: DType | None = None,
):
  """Computes a dot_general operation that may be quantized."""
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant:
    dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
    dot_general = dot_general_cls()
    return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)

  if kernel_scale is not None or is_fp8_dtype(getattr(kernel, "dtype", None)):
    if compute_dtype is None:
      compute_dtype = inputs.dtype
    kernel = dequantize_weight(kernel, kernel_scale, compute_dtype=compute_dtype)

  return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)


def _compute_dot_general_nnx(
    inputs,
    kernel,
    axis,
    contract_ind,
    matmul_precision,
    quant_dot_general: nnx_wrappers.ToNNX | None,
    initializing: bool,
    out_sharding: NamedSharding | None = None,
    kernel_scale: Array | None = None,
    compute_dtype: DType | None = None,
    native_fp8_compute: bool = False,
    scale_block_size: int | tuple[int, ...] | None = None,
    act_calibration_method: str = "absmax",
):
  """Computes a dot_general operation that may be quantized."""
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant_dot_general is not None:
    if initializing:
      quant_dot_general.lazy_init(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
    return quant_dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None, mutable=["aqt"])

  if compute_dtype is None:
    compute_dtype = inputs.dtype

  if out_sharding is not None:
    out_ndim = (inputs.ndim - len(axis)) + (kernel.ndim - len(contract_ind))
    out_sharding = truncate_out_sharding(out_sharding, out_ndim)

  if (
      native_fp8_compute
      and is_fp8_dtype(kernel.dtype)
      and kernel_scale is not None
      and len(axis) == 1
      and len(contract_ind) == 1
  ):
    # Native FP8 supports single contracted axis; multi-axis contractions fall back to dequantize.
    return quantizations.native_fp8_dot_general(
        inputs,
        kernel,
        kernel_scale,
        scale_block_size,
        axis,
        contract_ind,
        compute_dtype=compute_dtype,
        precision=matmul_precision,
        out_sharding=out_sharding,
        act_calibration_method=act_calibration_method,
    )

  if kernel_scale is not None or is_fp8_dtype(getattr(kernel, "dtype", None)):
    kernel = dequantize_weight(kernel, kernel_scale, compute_dtype=compute_dtype)

  return dot_general(
      inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision, out_sharding=out_sharding
  )


class DenseGeneral(nnx.Module):
  """A linear transformation with flexible axes."""

  def __init__(
      self,
      in_features_shape: Iterable[int] | int,
      out_features_shape: Iterable[int] | int,
      axis: Iterable[int] | int = -1,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: tuple[None | str, ...] = (),
      quant: None | Quant = None,
      use_bias: bool = False,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
      parameter_memory_host_offload: bool = False,
      mesh: Mesh | None = None,
      use_two_stage_all_gather: bool = False,
      debug_sharding: bool = False,
      has_scale: bool | None = None,
      kernel_scale_init: Initializer | None = None,
      scale_shape: Shape | None = None,
      scale_axes: tuple[None | str, ...] | None = None,
      scale_dtype: DType = jnp.float32,
      block_size: int | tuple[int, ...] | None = None,
      weight_quant: quantizations.WeightQuantConfig | None = None,
      *,  # Following arguments are keyword-only
      rngs: nnx.Rngs = None,
  ):
    """Initializes the DenseGeneral module.

    Args:
      in_features_shape: tuple with numbers of input features for axes specified in
        'axis'.
      out_features_shape: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      weight_dtype: the dtype of the weights (default: float32).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      kernel_axes: logical axes for partitioning the kernel.
      quant: quantization config, defaults to None implying no quantization.
      use_bias: whether to add bias in linear transformation.
      shard_mode: auto or explicit shard mode.
      matmul_precision: Precision for matrix multiplication.
      parameter_memory_host_offload: Determines whether to offload params to host
      mesh: Mesh of devices and physical axes, needed for two-stage all-gather.
      use_two_stage_all_gather: when the kernel is sharded on both the fsdp and
        fsdp_transpose axes, gather the two axes with two separate all-gather
        calls (separated by an optimization barrier) to avoid the relayout
        transpose XLA emits for a single combined 2-axis all-gather.
      debug_sharding: when True, log the logical/physical sharding of the
        two-stage all-gather constraints to the sharding dump files.
      has_scale: whether to initialize a separate scale parameter (kernel_scale).
      kernel_scale_init: initializer function for kernel_scale.
      scale_shape: explicit shape for kernel_scale.
      scale_axes: logical axes for partitioning kernel_scale.
      scale_dtype: dtype of kernel_scale (default: float32).
      block_size: block size for block-wise quantization scales.
      weight_quant: optional WeightQuantConfig decoupling weight quantization settings.
      rngs: RNG state for initialization in nnx.
    """
    if weight_quant is not None:
      weight_dtype = weight_quant.weight_dtype
      scale_dtype = weight_quant.scale_dtype
      if block_size is None:
        block_size = weight_quant.block_size

    self.in_features_shape = canonicalize_tuple(in_features_shape)
    self.out_features_shape = canonicalize_tuple(out_features_shape)
    self.axis = canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.quant = quant
    self.use_bias = use_bias
    self.shard_mode = shard_mode
    self.matmul_precision = matmul_precision
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.mesh = mesh
    self.use_two_stage_all_gather = use_two_stage_all_gather
    self.debug_sharding = debug_sharding
    self.has_scale = has_scale
    self.scale_dtype = scale_dtype
    self.block_size = block_size
    self.weight_quant = weight_quant

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    if not quantizations.in_serve_mode(self.quant):
      init_dtype = jnp.float32 if is_fp8_dtype(self.weight_dtype) else self.weight_dtype
      kernel_val = self.kernel_init(
          rngs.params(),
          kernel_shape,
          init_dtype,
          kernel_in_axis,
          kernel_out_axis,
      ).astype(self.weight_dtype)

      self.kernel = nnx.Param(
          kernel_val,
          sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      bias_val = default_bias_init(rngs.params(), bias_shape, self.weight_dtype)
      self.bias = nnx.Param(
          bias_val,
          sharding=bias_axes,
      )
    else:
      self.bias = None

    should_have_scale = is_fp8_dtype(self.weight_dtype) if has_scale is None else has_scale
    if should_have_scale and not quantizations.in_serve_mode(self.quant):
      # Phase 1: Resolve scale shape based on quantization granularity
      # - Explicit scale_shape: user or caller override.
      # - Block scaling (e.g. block_size=128): scales are partitioned into a grid
      #   of size (K // 128, N // 128). Dimensions smaller than block_size (e.g. head_dim < 128)
      #   are preserved as-is.
      # - Per-tensor scaling: a single scalar float32 scale with empty shape ().
      if scale_shape is not None:
        resolved_scale_shape = canonicalize_tuple(scale_shape)
      elif block_size is not None:
        if isinstance(block_size, int):
          block_sizes = (block_size,) * len(kernel_shape)
        elif len(block_size) == len(kernel_shape):
          block_sizes = tuple(block_size)
        else:
          block_sizes = (block_size[0],) * len(kernel_shape)
        resolved_scale_shape = tuple(d if d < b else d // b for d, b in zip(kernel_shape, block_sizes))
      else:
        resolved_scale_shape = ()

      # Phase 2: Resolve scale sharding axes to match the weight tensor's mesh partitioning
      # - Scalar scale (): cannot be sharded across mesh devices, so sharding is empty ().
      # - Block scale grid (matching kernel rank): inherits kernel_axes for any dimension
      #   spanning multiple blocks (> 1), ensuring the scale grid partitions synchronously
      #   with the weight matrix under FSDP, TP, or expert parallelism.
      # - Unpartitioned dimension (dimension size == 1): does not require sharding (None).
      if scale_axes is not None:
        resolved_scale_axes = scale_axes
      elif len(resolved_scale_shape) == 0:
        resolved_scale_axes = ()
      elif len(resolved_scale_shape) == len(kernel_shape):
        padded_kernel_axes = self.kernel_axes + (None,) * (len(kernel_shape) - len(self.kernel_axes))
        resolved_scale_axes = tuple(
            ax if s_dim > 1 else None for ax, s_dim in zip(padded_kernel_axes, resolved_scale_shape)
        )
      else:
        resolved_scale_axes = tuple(None for _ in resolved_scale_shape)

      actual_scale_init = kernel_scale_init if kernel_scale_init is not None else jax.nn.initializers.ones
      self.scale_axes = resolved_scale_axes
      self.kernel_scale = nnx.Param(
          actual_scale_init(
              rngs.params(),
              resolved_scale_shape,
              self.scale_dtype,
          ),
          sharding=resolved_scale_axes,
      )
    else:
      self.scale_axes = None
      self.kernel_scale = None

    if quant and not isinstance(quant, quantizations.ServeFp8WeightQuantization):
      # Native FP8 compute runs in _compute_dot_general_nnx; no Linen quantizer needed.
      dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
      dot_general_linen = dot_general_cls()
      quant_dot_general = nnx_wrappers.ToNNX(dot_general_linen, rngs=rngs)
      self._quant_dot_general_name = f"{type(dot_general_linen).__name__}_0"
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      block_size = getattr(quant, "get_block_size", lambda: 1)()  # needed for TE MXFP8
      dummy_inputs = jnp.zeros((block_size, *self.in_features_shape), dtype=self.dtype)
      self(dummy_inputs, _initializing=True)
      # Backends that never draw at apply time leave dead RNG state in the model.
      if not quant.needs_apply_rngs:
        quant_dot_general.release_rngs()
    else:
      self._quant_dot_general_name = None

  @property
  def quant_dot_general(self) -> nnx_wrappers.ToNNX | None:
    if self._quant_dot_general_name is None:
      return None
    return getattr(self, self._quant_dot_general_name)

  def _maybe_two_stage_all_gather(self, kernel):
    """Gather a 2D-FSDP-sharded MLP kernel with two single-axis all-gathers.

    When the kernel is sharded on both the `fsdp` and `fsdp_transpose` mesh axes,
    a single combined 2-axis all-gather forces XLA to materialize an interleave
    transpose to fix the layout. Splitting into two single-axis gathers separated
    by an `optimization_barrier` makes each stage produce a contiguous layout, so
    no transpose is emitted. Mirrors `moe_fsdp_use_two_stage_all_gather`.
    """
    if (
        not self.use_two_stage_all_gather
        or self.mesh is None
        or self.mesh.shape.get("fsdp", 1) <= 1
        or self.mesh.shape.get("fsdp_transpose", 1) <= 1
    ):
      return kernel

    # kernel_axes is a plain tuple of logical names; wrap it so the logical-to-physical
    # lookup treats it as a single spec rather than a pytree of strings.
    full_logical = PartitionSpec(*self.kernel_axes)
    # Stage 1 gathers fsdp_transpose, stage 2 gathers the remaining fsdp.
    stage1 = get_physical_spec_without_axes(full_logical, self.mesh, ("fsdp_transpose",))
    stage2 = get_physical_spec_without_axes(full_logical, self.mesh, FSDP_MESH_AXES)
    if stage1.spec == stage2.spec:
      # Not sharded on both FSDP axes, so a single all-gather is already optimal.
      return kernel

    shard = functools.partial(maybe_shard_with_name, shard_mode=self.shard_mode, debug_sharding=self.debug_sharding)
    kernel = shard(kernel, stage1)
    kernel = jax.lax.optimization_barrier(kernel)
    kernel = shard(kernel, stage2)
    return kernel

  def __call__(
      self,
      inputs: Array,
      _initializing: bool = False,
      out_sharding: NamedSharding | None = None,
      slice_bounds: tuple[int, int] | None = None,
  ) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.
      _initializing: Whether the module is initializing.
      out_sharding: Optional sharding for the output.
      slice_bounds: Optional tuple (begin, end) to slice the kernel and bias on
        the last (output-feature) axis before contraction. Unquantized only.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = normalize_axes(self.axis, inputs.ndim)

    for i, ax in enumerate(norm_axis):
      if inputs.shape[ax] != self.in_features_shape[i]:
        raise ValueError(
            f"Input dimension {inputs.shape[ax]} at axis {ax} "
            f"does not match expected input feature size {self.in_features_shape[i]}"
        )

    if quantizations.in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
      kernel_scale = None
    else:
      if hasattr(self.kernel, "get_value"):
        kernel = self.kernel.get_value()
      elif isinstance(self.kernel, (dict, nnx.State)) and "value" in self.kernel:
        kernel = self.kernel["value"]
      else:
        kernel = getattr(self.kernel, "value", self.kernel)
      if hasattr(kernel, "value"):
        kernel = kernel.value
      # Move logit_dense kernel to device if parameter offloading is enabled
      if self.parameter_memory_host_offload:
        max_logging.log("linear.py: Moving parameter logits_dense kernel to device")
        kernel = jax.device_put(kernel, max_utils.device_space())
      if self.kernel_scale is not None:
        kernel_scale = self.kernel_scale[...]
        if self.parameter_memory_host_offload:
          kernel_scale = jax.device_put(kernel_scale, max_utils.device_space())
      else:
        kernel_scale = None

      # Cast non-quantized weights to the compute dtype here, before slicing, the
      # two-stage all-gather and the dot_general dispatch. This matches the
      # unquantized baseline: AQT asserts that both operands share a dtype, and
      # the all-gather should move compute-precision (not weight-precision) bytes.
      # FP8 weights deliberately stay quantized until the fused dequantization in
      # _compute_dot_general_nnx, so that the all-gather moves 8-bit values.
      is_fp8_weight = kernel_scale is not None or is_fp8_dtype(getattr(kernel, "dtype", None))
      if not is_fp8_weight:
        kernel = jnp.asarray(kernel, self.dtype)

    if slice_bounds is not None:
      if self.quant is not None:
        raise ValueError("sliced contraction is only supported when quant is None")
      if is_fp8_dtype(getattr(kernel, "dtype", None)) or kernel_scale is not None:
        kernel = dequantize_weight(kernel, kernel_scale, compute_dtype=self.dtype)
        kernel_scale = None
      begin, end = slice_bounds
      if not 0 <= begin < end <= kernel.shape[-1]:
        raise ValueError(f"slice_bounds {slice_bounds} must be valid and within [0, {kernel.shape[-1]}]")
      kernel = kernel[..., begin:end]

    if (
        _DENSE_WGRAD_RS.enabled
        and not _initializing
        and slice_bounds is None
        and self.quant is None
        and kernel_scale is None
        and self.shard_mode == ShardMode.AUTO
        and self.kernel_axes
        and not self.parameter_memory_host_offload
        and isinstance(kernel, jax.Array)
        and not is_fp8_dtype(kernel.dtype)
        and len(self.kernel_axes) == kernel.ndim
        and (self.mesh is not None or _DENSE_WGRAD_RS.mesh is not None)
        and (_DENSE_WGRAD_RS.max_kernel_elems <= 0 or kernel.size <= _DENSE_WGRAD_RS.max_kernel_elems)
    ):
      settings = _DENSE_WGRAD_RS
      if self.mesh is not None and self.mesh is not settings.mesh:
        settings = dataclasses.replace(settings, mesh=self.mesh)
      output = _fsdp_shard_map_dot(inputs, kernel, self.kernel_axes, norm_axis, self.matmul_precision, settings)
      if output is not None:
        if self.bias is not None:
          output += jnp.asarray(self.bias[...], self.dtype)
        return output

    kernel = self._maybe_two_stage_all_gather(kernel)

    # out_sharding should be None for auto mesh axis
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    contract_ind = tuple(range(0, len(self.axis)))
    native_fp8_compute = isinstance(self.quant, quantizations.ServeFp8WeightQuantization)
    output = _compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general if slice_bounds is None else None,
        _initializing,
        out_sharding,
        kernel_scale=kernel_scale,
        compute_dtype=self.dtype,
        native_fp8_compute=native_fp8_compute,
        scale_block_size=self.block_size,
        act_calibration_method=self.quant.act_calibration_method if native_fp8_compute else "absmax",
    )

    if self.bias is not None:
      bias = jnp.asarray(self.bias[...], self.dtype)
      if slice_bounds is not None:
        begin, end = slice_bounds
        bias = bias[..., begin:end]
      output += bias
    return output


def dense_general(
    *,
    inputs_shape: tuple[int, ...] | None = None,
    in_features_shape: tuple[int, ...] | int | None = None,
    out_features_shape: Iterable[int] | int,
    axis: Iterable[int] | int = -1,
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: tuple[None | str, ...] = (),
    quant: None | Quant = None,
    use_bias: bool = False,
    shard_mode: ShardMode = ShardMode.AUTO,
    matmul_precision: str = "default",
    parameter_memory_host_offload: bool = False,
    has_scale: bool | None = None,
    kernel_scale_init: Initializer | None = None,
    scale_shape: Shape | None = None,
    scale_axes: tuple[None | str, ...] | None = None,
    scale_dtype: DType = jnp.float32,
    block_size: int | tuple[int, ...] | None = None,
    weight_quant: quantizations.WeightQuantConfig | None = None,
    name: None | str = None,
):
  """Creates a DenseGeneral Linen module using nnx.bridge.to_linen.

  Args:
    inputs_shape: tuple with the shape of the inputs
    in_features_shape: tuple with numbers of input features for axes specified in
      'axis'.
    out_features_shape: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    kernel_axes: logical axes for partitioning the kernel.
    quant: quantization config, defaults to None implying no quantization.
    use_bias: whether to add bias in linear transformation.
    shard_mode: indicating the shard mode
    matmul_precision: Precision for matrix multiplication.
    parameter_memory_host_offload: Determines whether to offload params to host
    has_scale: whether to initialize a separate scale parameter (kernel_scale).
    kernel_scale_init: initializer function for kernel_scale.
    scale_shape: explicit shape for kernel_scale.
    scale_axes: logical axes for partitioning kernel_scale.
    scale_dtype: dtype of kernel_scale (default: float32).
    block_size: block size for block-wise quantization scales.
    weight_quant: optional WeightQuantConfig decoupling weight quantization settings.
    name: name passed to the ToLinen Module
  """
  if not (inputs_shape is not None) ^ (in_features_shape is not None):
    raise ValueError("Exactly one of inputs_shape or in_features must be specified.")

  if inputs_shape is not None:
    axis = canonicalize_tuple(axis)
    in_features_shape = tuple(inputs_shape[ax] for ax in normalize_axes(axis, len(inputs_shape)))
  else:
    assert in_features_shape is not None
  module = nnx_wrappers.to_linen(
      DenseGeneral,
      in_features_shape=in_features_shape,
      out_features_shape=out_features_shape,
      axis=axis,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      quant=quant,
      use_bias=use_bias,
      shard_mode=shard_mode,
      matmul_precision=matmul_precision,
      parameter_memory_host_offload=parameter_memory_host_offload,
      has_scale=has_scale,
      kernel_scale_init=kernel_scale_init,
      scale_shape=scale_shape,
      scale_axes=scale_axes,
      scale_dtype=scale_dtype,
      block_size=block_size,
      weight_quant=weight_quant,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


class Dropout(nnx.Dropout):
  """Forked nnx.Dropout that is easier to use with bridge"""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      rate: float,
      *,
      broadcast_dims: Sequence[int] = (),
      deterministic: bool = False,
      rng_collection: str = "dropout",
      rngs: nnx.Rngs | None = None,
  ):
    self.rate = rate
    self.broadcast_dims = broadcast_dims
    self.deterministic = deterministic
    self.rng_collection = rng_collection

    if not isinstance(rngs, nnx.Rngs):
      raise TypeError(f"rngs must be a Rngs, RngStream or None, but got {type(rngs)}.")

    # fork() advances the caller's streams, so fork even at rate 0: skipping it would
    # shift every later draw and change parameter initialization.
    forked = rngs.fork() if hasattr(type(rngs), "fork") else rngs

    # nnx.Dropout returns its input before touching self.rngs at rate 0, so keeping the
    # fork would only add dead RNG state to the model.
    self.rngs = forked if rate > 0.0 else nnx.data(None)


class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      in_features: int,
      intermediate_dim: int = 2048,
      activations: Sequence[str | Callable[..., Any]] = ("relu",),
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      intermediate_dropout_rate: float = 0.1,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      use_bias: bool = False,
      use_pre_norm: bool = False,
      quant: None | Quant = None,
      model_mode: None | str = None,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    """A MlpBlock module.

    Args:
      config: Config object containing model parameters.
      mesh: Mesh object of device and physical axes information
      in_features: Number of input features.
      intermediate_dim: Shared dimension of hidden layers.
      activations: Type of activations for each layer.  Each element is either
        'linear', a string function name in flax.linen, or a function.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: computation data type for the dense layer.
      weight_dtype: weight data type for the dense layer.
      use_bias: whether to add bias in all feedforward layers.
      use_pre_norm: whether to add pre layer norm in mlp layers.
      quant: Optional quantization config, no quantization if None.
      out_sharding: Named sharding of outputs
    """
    self.config = config
    self.mesh = mesh
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.activations = activations
    self.kernel_init = kernel_init
    self.intermediate_dropout_rate = intermediate_dropout_rate
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.use_bias = use_bias
    self.use_pre_norm = use_pre_norm
    self.quant = quant
    self.model_mode = model_mode

    if self.use_pre_norm:
      self.mlp_layer_norm = self.get_norm_layer(num_features=in_features)(
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=config.normalization_layer_epsilon,
          rngs=rngs,
      )
    else:
      self.mlp_layer_norm = None

    if self.model_mode == MODEL_MODE_PREFILL:
      self.intermediate_logical = ("activation_batch", "prefill_activation_length", "activation_mlp")
    else:
      self.intermediate_logical = ("activation_batch", "activation_length", "activation_mlp")

    weight_quant = quantizations.get_weight_quant_config(config, "mlp")
    block_size = weight_quant.block_size if weight_quant is not None else getattr(config, "weight_block_size", None)

    if config.fused_mlp:
      self.wi = DenseGeneral(
          in_features_shape=in_features,
          out_features_shape=(len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          quant=self.quant,
          use_bias=self.use_bias,
          shard_mode=self.config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          mesh=self.mesh,
          use_two_stage_all_gather=self.config.dense_fsdp_use_two_stage_all_gather,
          debug_sharding=self.config.debug_sharding,
          block_size=block_size,
          weight_quant=weight_quant,
          rngs=rngs,
      )
    else:
      for idx in range(len(self.activations)):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        module = DenseGeneral(
            in_features_shape=in_features,
            out_features_shape=self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            quant=self.quant,
            use_bias=self.use_bias,
            shard_mode=self.config.shard_mode,
            matmul_precision=self.config.matmul_precision,
            mesh=self.mesh,
            use_two_stage_all_gather=self.config.dense_fsdp_use_two_stage_all_gather,
            debug_sharding=self.config.debug_sharding,
            block_size=block_size,
            weight_quant=weight_quant,
            rngs=rngs,
        )
        setattr(self, dense_name, module)
    self.dropout = Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,), rngs=rngs)
    self.wo = DenseGeneral(
        in_features_shape=self.intermediate_dim,
        out_features_shape=in_features,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        quant=self.quant,
        use_bias=self.use_bias,
        shard_mode=self.config.shard_mode,
        matmul_precision=self.config.matmul_precision,
        mesh=self.mesh,
        use_two_stage_all_gather=self.config.dense_fsdp_use_two_stage_all_gather,
        debug_sharding=self.config.debug_sharding,
        block_size=block_size,
        weight_quant=weight_quant,
        rngs=rngs,
    )

    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=config.shard_mode,
        debug_sharding=config.debug_sharding,
    )

  def get_norm_layer(self, num_features: int):
    """get normalization layer."""
    if self.config.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.GEMMA,
        DecoderBlockType.GEMMA2,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.QWEN3,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.LLAMA4,
        DecoderBlockType.OLMO3,
        DecoderBlockType.ENVY,
    ):
      return functools.partial(normalizations.RMSNorm, num_features=num_features)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      from maxtext.models import gpt3  # pylint: disable=import-outside-toplevel

      return functools.partial(
          gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=self.use_bias
      )
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def __call__(
      self,
      inputs,
      decode: bool = False,
      deterministic: bool = False,
      intermediate_sharding: NamedSharding | None = None,
      out_sharding: NamedSharding | None = None,
  ):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.mlp_layer_norm is not None:
      inputs = self.mlp_layer_norm(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = self.wi(inputs, out_sharding=intermediate_sharding)

      # Enforce fused activations don't shard on num_activations axis
      fused_intermediate_logical = self.intermediate_logical[:2] + (None,) + self.intermediate_logical[2:]
      x = self._maybe_shard_with_logical(x, fused_intermediate_logical)

      x = checkpoint_name(x, "mlpwi")
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        module = getattr(self, dense_name)
        x = module(inputs, out_sharding=intermediate_sharding)
        x = checkpoint_name(x, "mlp" + dense_name)
        if cfg.activations_in_float32:
          x = x.astype(jnp.float32)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = self.dropout(x, deterministic=deterministic)  # Broadcast along length.
    x = self._maybe_shard_with_logical(x, self.intermediate_logical)
    output = self.wo(x, out_sharding=out_sharding)

    output = checkpoint_name(output, "mlpwo")
    return output


def mlp_block(
    *,
    config: Config,
    mesh: Mesh,
    in_features: int,
    intermediate_dim: int = 2048,
    activations: Sequence[str | Callable[..., Any]] = ("relu",),
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    intermediate_dropout_rate: float = 0.1,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    use_bias: bool = False,
    use_pre_norm: bool = False,
    quant: None | Quant = None,
    model_mode: None | str = None,
    name: None | str = None,
):
  """Creates a MlpBlock Linen module using nnx.bridge.to_linen."""
  module = nnx_wrappers.to_linen(
      MlpBlock,
      config=config,
      mesh=mesh,
      in_features=in_features,
      intermediate_dim=intermediate_dim,
      activations=activations,
      kernel_init=kernel_init,
      intermediate_dropout_rate=intermediate_dropout_rate,
      dtype=dtype,
      weight_dtype=weight_dtype,
      use_bias=use_bias,
      use_pre_norm=use_pre_norm,
      quant=quant,
      model_mode=model_mode,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


class DeepSeekV4GroupedLinear(nnx.Module):
  """Block-diagonal grouped linear used by the grouped output projection in DeepSeek-V4.

  The core attention's stacked output is `num_attention_heads * head_dim`-dim,
  which is extremely large. A direct projection would dominate the per-token cost.
  This module splits the heads into `g` groups, projecting each independently
  to a smaller intermediate dimension, which are later mixed.
  """

  def __init__(
      self,
      in_features_per_group: int,
      out_features: int,
      n_groups: int,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: tuple[None | str, ...] = ("groups", "embed", "mlp"),
      matmul_precision: str = "default",
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes the DeepSeekV4GroupedLinear module.

    Args:
      in_features_per_group: The size of the input dimension for each group.
      out_features: The total output dimension across all groups. Must be divisible by n_groups.
      n_groups: The number of independent groups to split the projection into.
      weight_dtype: the dtype of the weights (default: float32).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      kernel_axes: logical axes for partitioning the kernel.
      matmul_precision: Precision for matrix multiplication.
      parameter_memory_host_offload: Determines whether to offload params to host
      rngs: RNG state for initialization in nnx.
    """
    if out_features % n_groups != 0:
      raise ValueError(f"out_features ({out_features}) must be divisible by n_groups ({n_groups})")

    self.in_features_per_group = in_features_per_group
    self.out_features = out_features
    self.n_groups = n_groups
    self.out_features_per_group = out_features // n_groups

    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.matmul_precision = matmul_precision
    self.parameter_memory_host_offload = parameter_memory_host_offload

    # Kernel shape splits the projection up into a batched representation
    kernel_shape = (self.n_groups, self.in_features_per_group, self.out_features_per_group)

    # NdInitializer takes tuple positions to calculate fan_in / fan_out.
    # Axis 1 represents the inner contracting dimension (fan_in).
    # Axis 2 represents the output features dimension (fan_out).
    kernel_in_axis = (1,)
    kernel_out_axis = (2,)

    self.kernel = nnx.Param(
        self.kernel_init(
            rngs.params(),
            kernel_shape,
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
        ),
        sharding=self.kernel_axes,
    )

  def __call__(self, inputs: Array) -> Array:
    """Applies a batched grouped linear transformation to the inputs.

    Args:
      inputs: The nd-array to be transformed. Expected shape is `[..., n_groups, in_features_per_group]`.

    Returns:
      The transformed input of shape `[..., n_groups, out_features_per_group]`.
      When later flattened across the last two dims, this results in `out_features`.
    """
    inputs = jnp.asarray(inputs, self.dtype)

    kernel = self.kernel[...]
    if self.parameter_memory_host_offload:
      max_logging.log("linear.py: Moving parameter grouped_linear kernel to device")
      kernel = jax.device_put(kernel, max_utils.device_space())
    kernel = jnp.asarray(kernel, self.dtype)

    # Perform a batched matrix multiplication using einsum with explicit precision.
    # We use jnp.einsum instead of explicitly flattening and using lax.dot_general
    # to make the group-wise broadcast highly readable and natively batched.
    #
    # Notation breakdown:
    #   ... : Any leading batch/sequence dimensions (e.g., [Batch, SeqLen]).
    #   g   : The n_groups dimension.
    #   i   : The in_features_per_group dimension (the contracting dimension).
    #   o   : The out_features_per_group dimension.
    #
    # Input shape:  [..., g, i]
    # Kernel shape: [g, i, o]
    # Output shape: [..., g, o]
    output = jnp.einsum("...gi,gio->...go", inputs, kernel, precision=lax.Precision(self.matmul_precision))

    return output


def deepseek_v4_grouped_linear(
    *,
    in_features_per_group: int,
    out_features: int,
    n_groups: int,
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: tuple[None | str, ...] = ("groups", "embed", "mlp"),
    matmul_precision: str = "default",
    parameter_memory_host_offload: bool = False,
    name: None | str = None,
):
  """Creates a DeepSeekV4GroupedLinear Linen module using nnx.bridge.to_linen."""
  module = nnx_wrappers.to_linen(
      DeepSeekV4GroupedLinear,
      in_features_per_group=in_features_per_group,
      out_features=out_features,
      n_groups=n_groups,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      matmul_precision=matmul_precision,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module
