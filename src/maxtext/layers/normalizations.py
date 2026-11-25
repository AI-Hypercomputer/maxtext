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

"""Normalization Layers."""

import functools
from typing import Any

from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from maxtext.common.common_types import Array, DecoderBlockType, DType, ShardMode, is_fp8_dtype
from maxtext.layers.initializers import Initializer
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import sharding as maxtext_sharding
from maxtext.utils.sharding import truncate_out_sharding


def _flatten_active_mesh_axes(entries: tuple[Any, ...], mesh: Any) -> tuple[str, ...]:
  out: list[str] = []
  for entry in entries:
    if entry is None:
      continue
    raw = (entry,) if isinstance(entry, str) else tuple(entry)
    for ax in raw:
      if ax is not None and mesh.shape.get(ax, 1) > 1 and ax not in out:
        out.append(ax)
  return tuple(out)


def _deferred_norm_unavailable_reason(config: Any, mesh: Any) -> str | None:
  """Why `defer_norm_grad_allreduce` cannot apply to `config`, or None when it can."""
  if mesh is None:
    return "no mesh"
  if not getattr(config, "scan_layers", False):
    return "scan_layers=False"
  if getattr(config, "prefetch_fsdp_weights", False):
    return "prefetch_fsdp_weights=True (the prefetch path has no post-scan all-reduce)"
  if (
      bool(getattr(config, "using_pipeline_parallelism", False))
      or int(getattr(config, "ici_pipeline_parallelism", 1) or 1) > 1
      or int(getattr(config, "dcn_pipeline_parallelism", 1) or 1) > 1
  ):
    return "pipeline parallelism"
  if getattr(config, "decoder_block", None) not in (
      DecoderBlockType.QWEN3,
      DecoderBlockType.QWEN3_MOE,
      "qwen3",
      "qwen3_moe",
  ):
    return f"decoder_block={getattr(config, 'decoder_block', None)!r} (supported decoder blocks: qwen3, qwen3_moe)"
  if getattr(config, "shard_mode", ShardMode.AUTO) != ShardMode.AUTO:
    return "shard_mode != auto"
  return None


@functools.lru_cache(maxsize=None)
def _warn_deferred_norm_unavailable(reason: str) -> None:
  """Logs once per reason that the deferral is requested but inactive."""
  max_logging.warning(
      f"defer_norm_grad_allreduce=True has no effect: {reason}; the norm scale gradients use the standard"
      " per-layer all-reduces."
  )


def get_deferred_norm_psum_axes(config: Any, mesh: Any) -> tuple[str, ...]:
  """Returns batch/FSDP mesh axes for deferred post-scan RMSNorm scale all-reduces, or `()` when disabled.

  When `defer_norm_grad_allreduce` is set but the deferral cannot apply, a warning is logged once per
  reason and `()` is returned so callers fall back to the standard path.
  """
  if not getattr(config, "defer_norm_grad_allreduce", False):
    return ()
  reason = _deferred_norm_unavailable_reason(config, mesh)
  if reason is None:
    rules = maxtext_sharding.get_logical_axis_rules() or getattr(config, "logical_axis_rules", None)
    norm_axes = maxtext_sharding.logical_to_mesh_axes(("norm",), mesh, rules=rules)
    act_axes = maxtext_sharding.logical_to_mesh_axes(
        ("activation_batch", "activation_norm_length", "activation_embed"), mesh, rules=rules
    )
    if (norm_axes and norm_axes[0] is not None) or (len(act_axes) >= 3 and act_axes[2] is not None):
      reason = "the norm scale or the embedding activations are sharded"
    else:
      axes = _flatten_active_mesh_axes(tuple(act_axes[:2]), mesh)
      if axes:
        return axes
      reason = "no batch or sequence mesh axis spans more than one device"
  _warn_deferred_norm_unavailable(reason)
  return ()


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def post_scan_psum_grad(w: jnp.ndarray, mesh: Any, psum_axes: tuple[str, ...]) -> jnp.ndarray:
  """Identity in forward; sums unreduced per-shard norm scale gradients across `psum_axes` once after `lax.scan`.

  Second half of the deferred norm-grad all-reduce; it must be paired with the first half. A scale that reaches
  `_scale_mul_deferred_grad` (`RMSNorm(defer_scale_grad_allreduce=True)`) or the fused QKV path with
  `defer_norm_grad_psum=True` (`Attention.defer_qk_norm_grad_allreduce`) gets a gradient that is summed over the
  local batch/seq shard only, although it is typed as replicated. The caller must wrap the stacked `[layers, D]`
  scale with this function outside the layer scan, over the same `psum_axes` (`get_deferred_norm_psum_axes`), and
  must not reshard the scale gradient in between; otherwise the scale gradients stay per-shard and the replicas
  diverge. `NNXDecoder._apply_layers_sequentially` is the only code that sets the two flags, and it wraps every
  deferred scale (raising if one cannot be found).
  """
  return w


def _post_scan_psum_grad_fwd(w: jnp.ndarray, mesh: Any, psum_axes: tuple[str, ...]):
  return w, ()


def _post_scan_psum_grad_bwd(mesh: Any, psum_axes: tuple[str, ...], _, dw: jnp.ndarray):
  """All-reduces the accumulated scale gradient across `psum_axes` after the decoder scan."""
  if not psum_axes:
    return (dw,)
  rep_spec = jax.sharding.PartitionSpec(*((None,) * dw.ndim))
  ax = psum_axes if len(psum_axes) > 1 else psum_axes[0]

  @functools.partial(jax.shard_map, mesh=mesh, in_specs=rep_spec, out_specs=rep_spec, check_vma=False)
  def _sum_across_shards(dw_local):
    return lax.psum(dw_local.astype(jnp.float32), ax).astype(dw_local.dtype)

  return (_sum_across_shards(dw),)


post_scan_psum_grad.defvjp(_post_scan_psum_grad_fwd, _post_scan_psum_grad_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _scale_mul_deferred_grad(
    y: jnp.ndarray,
    scale: jnp.ndarray,
    mesh: Any,
    y_pspec: jax.sharding.PartitionSpec,
    out_dtype: Any,
) -> jnp.ndarray:
  """Multiplies normalized `y` by `scale` while keeping `dscale` shard-local in backward.

  The backward returns `dscale` summed over the local batch/seq shard only, through a replicated out_spec with
  `check_vma=False`: the value is not replicated until `post_scan_psum_grad` sums it across the batch/seq mesh
  axes, so `scale` must be wrapped with `post_scan_psum_grad` (see its docstring for the pairing invariant).
  """
  scale_pspec = jax.sharding.PartitionSpec(None)

  @functools.partial(jax.shard_map, mesh=mesh, in_specs=(y_pspec, scale_pspec), out_specs=y_pspec, check_vma=False)
  def _fwd_shmap(y_l, s_l):
    return jnp.asarray(y_l * jnp.asarray(s_l, out_dtype), out_dtype)

  return _fwd_shmap(y, scale)


def _scale_mul_deferred_grad_fwd(
    y: jnp.ndarray,
    scale: jnp.ndarray,
    mesh: Any,
    y_pspec: jax.sharding.PartitionSpec,
    out_dtype: Any,
):
  return _scale_mul_deferred_grad(y, scale, mesh, y_pspec, out_dtype), (y, scale)


def _scale_mul_deferred_grad_bwd(
    mesh: Any,
    y_pspec: jax.sharding.PartitionSpec,
    out_dtype: Any,
    res,
    dout: jnp.ndarray,
):
  """Computes local `(dy, dscale)` without per-layer cross-shard `psum` on `dscale`."""
  y, scale = res
  scale_pspec = jax.sharding.PartitionSpec(None)

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(y_pspec, scale_pspec, y_pspec),
      # `ds_l` is per-shard, not replicated: the replicated spec with `check_vma=False` only defers its cross-shard
      # sum to `post_scan_psum_grad`, which the caller has to apply.
      out_specs=(y_pspec, scale_pspec),
      check_vma=False,
  )
  def _bwd_shmap(y_l, s_l, dout_l):
    dy_l = jnp.asarray(dout_l * jnp.asarray(s_l, out_dtype), y_l.dtype)
    ds_l = jnp.sum(dout_l.astype(jnp.float32) * y_l.astype(jnp.float32), axis=(0, 1)).astype(s_l.dtype)
    return dy_l, ds_l

  return _bwd_shmap(y, scale, dout)


_scale_mul_deferred_grad.defvjp(_scale_mul_deferred_grad_fwd, _scale_mul_deferred_grad_bwd)


def align_scale_with_normalized_axis(scale: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Reshards a 1-D per-feature scale to match the sharding of the axis it scales.

  A scale is stored with whatever layout its declared axes give it, which is
  picked for the usual case of scaling the embedding axis (`norm` -> `tensor`).
  A per-head QK norm instead normalizes over `head_dim`, which is unsharded
  while the neighbouring `heads` axis is the one on `tensor`; Gemma 4's router
  scale is declared `embed` (-> `fsdp`) but multiplies an activation whose last
  axis is `activation_embed` (-> `tensor`). Multiplying either pair directly
  under `ShardMode.EXPLICIT` puts one mesh axis on two axes of the result, or
  two different axes on the same one, which JAX rejects -- so align the scale
  with the activation instead.

  This is a no-op whenever the two already agree, which includes every
  auto-sharding-equivalent layout for the ordinary layer norms.
  """
  # Read through `.partitions`: a scale carrying a reduced/unreduced tag (the deferred
  # data-parallel all-reduce under gradient accumulation) rejects direct indexing. The tags
  # are carried over to the new spec so the scale keeps its place on the deferred path.
  activation_axis = jax.typeof(y).sharding.spec.partitions[-1]
  scale_spec = jax.typeof(scale).sharding.spec
  if scale_spec.partitions[-1] == activation_axis:
    return scale
  return jax.sharding.reshard(
      scale,
      jax.sharding.PartitionSpec(activation_axis, unreduced=scale_spec.unreduced, reduced=scale_spec.reduced),
  )


class RMSNorm(nnx.Module):
  """RMS normalization.

  `defer_scale_grad_allreduce=True` leaves the scale gradient summed over the local batch/seq shard only (see
  `_scale_mul_deferred_grad`); the caller must wrap the scale with `post_scan_psum_grad`.
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = jax.nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      scale_offset: float = 0.0,
      with_scale: bool = True,
      mesh: Any = None,
      defer_scale_grad_allreduce: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = dtype if is_fp8_dtype(weight_dtype) else weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale_offset = scale_offset
    self.with_scale = with_scale
    self.mesh = mesh
    self.defer_scale_grad_allreduce = defer_scale_grad_allreduce
    if self.with_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), (num_features,), self.weight_dtype),
          out_sharding=kernel_axes,
      )
    else:
      self.scale = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)

    # out_sharding must be None in auto shard mode
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    if out_sharding is not None:
      out_sharding = truncate_out_sharding(out_sharding, y.ndim)

    if not self.with_scale:
      if out_sharding is not None:
        y = jax.lax.with_sharding_constraint(y, out_sharding)
      return y

    scale = self.scale.get_value()
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, max_utils.device_space())

    if self.defer_scale_grad_allreduce:
      if (
          self.mesh is None
          or self.shard_mode != ShardMode.AUTO
          or x.ndim != 3
          or self.scale_offset != 0.0
          or out_sharding is not None
      ):
        raise ValueError(
            "defer_scale_grad_allreduce requires 3D inputs, ShardMode.AUTO, scale_offset==0.0, and out_sharding is None"
        )
      rules = maxtext_sharding.get_logical_axis_rules()
      norm_axes = maxtext_sharding.logical_to_mesh_axes(("norm",), self.mesh, rules=rules) if rules else ()
      act_axes = (
          maxtext_sharding.logical_to_mesh_axes(
              ("activation_batch", "activation_norm_length", "activation_embed"), self.mesh, rules=rules
          )
          if rules
          else ()
      )
      parts = tuple(act_axes) + (None,) * max(0, 3 - len(act_axes))
      y_pspec = jax.sharding.PartitionSpec(*parts[:3])
      if (
          not rules
          or (norm_axes and norm_axes[0] is not None)
          or parts[2] is not None
          or maxtext_sharding.adjust_pspec_for_indivisible_shapes(y_pspec, y.shape, self.mesh) != y_pspec
      ):
        raise ValueError(
            "defer_scale_grad_allreduce requires active logical_axis_rules with replicated norm/embed and divisible"
            f" batch/seq dims; got y.shape={y.shape}, y_pspec={y_pspec}."
        )
      return _scale_mul_deferred_grad(y, scale, self.mesh, y_pspec, self.dtype)

    scale_fp32 = jnp.asarray(scale, jnp.float32)
    effective_scale = scale_fp32
    if self.scale_offset != 0.0:
      # The offset constant must carry the scale's `reduced` axes (e.g. a pipeline stage weight under
      # pipeline_unreduced_weight_grads): JAX rejects mixing a reduced operand with a replicated one.
      offset = jnp.asarray(self.scale_offset, jnp.float32)
      scale_reduced = jax.typeof(scale_fp32).sharding.spec.reduced
      if scale_reduced:
        offset = jax.sharding.reshard(offset, jax.sharding.PartitionSpec(reduced=scale_reduced))
      effective_scale = scale_fp32 + offset
    if self.shard_mode == ShardMode.EXPLICIT:
      effective_scale = align_scale_with_normalized_axis(effective_scale, y)

    if self.scale_offset != 0.0:
      normed_fp32 = x * lax.rsqrt(mean2 + self.epsilon)
      y = jnp.einsum("...k,k->...k", normed_fp32, effective_scale, out_sharding=out_sharding)
      return jnp.asarray(y, self.dtype)

    effective_scale = jnp.asarray(effective_scale, self.dtype)
    return jnp.einsum("...k,k->...k", y, effective_scale, out_sharding=out_sharding)


class GlobalRMSNorm(RMSNorm):
  """
  Applies RMSNorm over the last two dimensions (Heads * HeadDim).
  Used for Olmo3 which normalizes across all heads combined.
  """

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    # x shape: [..., Heads, HeadDim]
    input_shape = x.shape

    # Flatten the last two dimensions: [..., Heads * HeadDim]
    # We use -2 and -1 to ensure we capture the last two dims regardless of rank
    flattened_shape = input_shape[:-2] + (input_shape[-2] * input_shape[-1],)
    x_flat = x.reshape(flattened_shape)

    # Apply standard RMSNorm (which normalizes over the last axis)
    y_flat = super().__call__(x_flat, out_sharding)

    # Reshape back to [..., Heads, HeadDim]
    return y_flat.reshape(input_shape)


def Qwen3NextRMSNorm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: DType = None,
    weight_dtype: DType = None,
    shard_mode=None,
    kernel_axes=None,
    parameter_memory_host_offload=None,
    *,
    rngs: nnx.Rngs,
):
  """Used for input and post attention layernorms in Qwen3NextDecoderLayer.

  This normalization layer is specific to Qwen3-Next. Key characteristics:
  1. The learnable scale parameter `scale` is initialized to ZEROS.
  2. The scale is applied as `(1.0 + self.scale)`, making the initial scale effectively 1.0.
     This matches the PyTorch implementation of Qwen3NextRMSNorm.
  """
  return nnx.data(
      RMSNorm(
          num_features=num_features,
          epsilon=epsilon,
          dtype=dtype,
          weight_dtype=weight_dtype,
          shard_mode=shard_mode if shard_mode is not None else ShardMode.AUTO,
          kernel_axes=kernel_axes if kernel_axes is not None else (),
          scale_init=jax.nn.initializers.zeros,
          scale_offset=1.0,
          parameter_memory_host_offload=bool(parameter_memory_host_offload),
          rngs=rngs,
      )
  )


class Qwen3NextRMSNormGated(nnx.Module):
  """
  This applies RMS Normalization and then a gated activation function (SiLU).
  This is used within the Qwen3NextGatedDeltaNet.

  The normalization is performed by an internal `RMSNorm` instance (`self.rms_norm`),
  which has its own learnable `scale` parameter, initialized to ONES.

  Attributes:
    num_features: The number of features in the input.
    eps: A small epsilon value to prevent division by zero in RMSNorm.
    dtype: The datatype of the computation.
    weight_dtype: The datatype of the internal RMSNorm scale.
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float,
      dtype: DType,
      weight_dtype: DType,
      shard_mode: ShardMode = ShardMode.AUTO,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rms_norm = nnx.data(
        RMSNorm(
            num_features=num_features,
            epsilon=self.epsilon,
            dtype=jnp.float32,
            weight_dtype=weight_dtype,
            shard_mode=shard_mode,
            scale_init=nnx.initializers.ones,
            rngs=rngs,
        )
    )

  def __call__(
      self,
      hidden_states: Array,
      gate: Array,
      out_sharding: NamedSharding | None = None,
  ) -> Array:
    """Applies RMSNorm and then a SiLU gate.

    Args:
      hidden_states: The input array to be normalized (o). Shape: (..., F)
      gate: The gating array for the activation (z). Shape: (..., F) where F is
        num_features.
      out_sharding: Optional layout for the normalized states, honoured only
        under `ShardMode.EXPLICIT`.

    Returns:
      The normalized and gated output array. Shape: (..., F)
    """
    normalized_states = self.rms_norm(hidden_states, out_sharding=out_sharding)

    # Gated Activation using SiLU (Sigmoid-weighted Linear Unit)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))

    return gated_states.astype(self.dtype)


def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  """L2 normalization function. Normalizes a vector to have a length of 1.

  Args:
    x: Input array.
    dim: The axis or axes along which to normalize. Defaults to the last axis.
    eps: Small epsilon to prevent division by zero.

  Returns:
    L2 normalized array with the same shape as x.
  """

  inv_norm = jax.lax.rsqrt((x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype))
  return x * inv_norm
