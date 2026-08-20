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
"""Shared zero-cost abstractions, block math, and tiling for mHC-lite Pallas kernels."""

import dataclasses
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_BLOCK_SIZE = 128
DEFAULT_BWD_BLOCK_SIZE = 32
DEFAULT_POST_BWD_BLOCK_SIZE = 32
DEFAULT_POST_BWD_FEATURE_BLOCK_SIZE = 1024
DEFAULT_VMEM_LIMIT_BYTES = 128 * 1024 * 1024
PARALLEL_DIMENSION_SEMANTICS = (pltpu.PARALLEL,)
SEQUENTIAL_DIMENSION_SEMANTICS = (pltpu.ARBITRARY,)
SEQUENTIAL_2D_DIMENSION_SEMANTICS = (pltpu.ARBITRARY, pltpu.ARBITRARY)
# Kernel-level context tuple: `(x, h_post, residual)`.
type KernelContext = tuple[jax.Array, jax.Array, jax.Array]


class UnsupportedInputError(ValueError):
  """Known Mosaic shape, dtype, or tiling restriction."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MHCContext:
  """Opaque token-local context passed from `pre` to `post`."""

  x: jax.Array
  h_post: jax.Array
  residual: jax.Array
  implementation: str = dataclasses.field(metadata={"static": True})


@dataclasses.dataclass(frozen=True)
class MhcKernelConfig:
  """Compiler and tiling configuration for mHC Pallas kernels."""

  block_size: int = DEFAULT_BLOCK_SIZE
  bwd_block_size: int = DEFAULT_BWD_BLOCK_SIZE
  bwd_feature_block_size: int = DEFAULT_POST_BWD_FEATURE_BLOCK_SIZE
  vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES
  rms_epsilon: float = 1e-5
  pre_mapping_epsilon: float = 1e-6
  interpret: bool = False


@dataclasses.dataclass(frozen=True)
class MhcDims:
  """Static dimension and cost descriptor for mHC-lite."""

  tokens: int
  streams: int
  embedding: int
  num_permutations: int = 24

  @property
  def flattened_size(self) -> int:
    return self.streams * self.embedding

  @property
  def pre_slice(self) -> slice:
    return slice(0, self.streams)

  @property
  def post_slice(self) -> slice:
    return slice(self.streams, 2 * self.streams)

  @property
  def res_slice(self) -> slice:
    return slice(2 * self.streams, 2 * self.streams + self.num_permutations)

  @property
  def phi_cols(self) -> int:
    return 2 * self.streams + self.num_permutations

  def coeff_fwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the forward coefficient kernel.

    Mathematical Derivations:
      - FLOPs:
        1. Fused Projection GEMM: `flattened_x (T, k*d) @ phi (k*d, 2*k+P)`
           [einsum: `tf,fc->tc` where `f = k*d`, `c = 2*k + P`]
           = `2 * T * (k*d) * (2*k + P)` FLOPs (2 FLOPs per multiply-accumulate).
        2. Permutation GEMM: `softmax_weights (T, P) @ permutations (P, k*k)`
           [einsum: `tp,pij->tij` where `i, j` are streams `k, k`]
           = `2 * T * P * k*k` FLOPs.
        Total FLOPs = `2 * tokens * flattened_size * phi_cols + 2 * tokens * num_permutations * streams^2`.
      - Transcendentals:
        Sigmoid gating for pre/post gates and softmax over permutation logits
        = `T * (k + k + P) = tokens * (2 * streams + num_permutations) = tokens * phi_cols`.
      - Bytes Accessed:
        Read input `x` (bfloat16: `T * k * d * 2` bytes) + read `phi` (float32: `(k*d) * phi_cols * 4` bytes)
        = `tokens * streams * embedding * 2 + flattened_size * phi_cols * 4` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(
        2 * self.tokens * self.flattened_size * self.phi_cols
        + 2 * self.tokens * self.num_permutations * self.streams * self.streams
    )
    transcendentals = int(self.tokens * (2 * self.streams + self.num_permutations))
    bytes_accessed = int(self.tokens * self.streams * self.embedding * 2 + self.flattened_size * self.phi_cols * 4)
    return pl.CostEstimate(flops=flops, transcendentals=transcendentals, bytes_accessed=bytes_accessed)

  def pre_apply_fwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the forward pre-apply gating kernel.

    Mathematical Derivations:
      - FLOPs:
        Stream reduction `sum_s (h_pre[t, s] * x[t, s, d])` [einsum: `ts,tsd->td`] across `k` streams
        for each of the `T * d` output elements (vector-matrix product `(1, k) @ (k, d) -> (1, d)` per token).
        = `2 * T * k * d = 2 * tokens * streams * embedding` FLOPs.
      - Transcendentals:
        0 (linear scaling and summation).
      - Bytes Accessed:
        Read input `x` (bfloat16: `T * k * d * 2` bytes) + write `layer_input` (bfloat16: `T * d * 2` bytes)
        = `tokens * streams * embedding * 2 + tokens * embedding * 2` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(2 * self.tokens * self.streams * self.embedding)
    bytes_accessed = int(self.tokens * self.streams * self.embedding * 2 + self.tokens * self.embedding * 2)
    return pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed)

  def post_apply_fwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the forward post-apply mixing kernel.

    Mathematical Derivations:
      - FLOPs:
        1. Residual mixing contraction `sum_s_in (residual[t, s_in, s_out] * x[t, s_in, d])`
           [einsum: `tkj,tkd->tjd`] (matrix product `(k, k) @ (k, d) -> (k, d)` per token):
           `2 * k` FLOPs per output element over `T * k * d` elements = `2 * T * k^2 * d` FLOPs.
        2. Post-gating and accumulation `h_post[t, s] * layer_output[t, d] + residual_mix[t, s, d]`
           [einsum: `ts,td->tsd`]:
           `2 * T * k * d` FLOPs (1 multiply + 1 add per element).
        Total FLOPs = `2 * tokens * streams^2 * embedding + 2 * tokens * streams * embedding`.
      - Transcendentals:
        0.
      - Bytes Accessed:
        Read `x` (bfloat16: `T * k * d * 2` bytes) + write output (bfloat16: `T * k * d * 2` bytes)
        + read `layer_output` and gating/residual context (`T * d * 4` bytes)
        = `2 * tokens * streams * embedding * 2 + tokens * embedding * 4` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(
        2 * self.tokens * self.streams * self.streams * self.embedding + 2 * self.tokens * self.streams * self.embedding
    )
    bytes_accessed = int(2 * self.tokens * self.streams * self.embedding * 2 + self.tokens * self.embedding * 4)
    return pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed)

  def pre_apply_bwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the backward pre-apply kernel.

    Mathematical Derivations:
      - FLOPs:
        1. Activation cotangent `d_x = h_pre * d_layer_input` [einsum: `ts,td->tsd`] (multiply)
           + accumulation with `d_x_acc` (add) = `2 * T * k * d` FLOPs.
        2. Gate cotangent `d_h_pre = sum_d (x * d_layer_input)` [einsum: `tsd,td->ts`] (multiply + reduce-add)
           = `2 * T * k * d` FLOPs.
        Total FLOPs = `4 * tokens * streams * embedding`.
      - Transcendentals:
        0.
      - Bytes Accessed:
        Read `x` (`T * k * d * 2`) + read `d_x_acc` (`T * k * d * 2`) + write `d_x_acc_out` (`T * k * d * 2`)
        + read `d_layer_input` (`T * d * 2`)
        = `3 * tokens * streams * embedding * 2 + tokens * embedding * 2` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(4 * self.tokens * self.streams * self.embedding)
    bytes_accessed = int(3 * self.tokens * self.streams * self.embedding * 2 + self.tokens * self.embedding * 2)
    return pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed)

  def coeff_bwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the backward coefficient kernel.

    Mathematical Derivations:
      - FLOPs:
        In-kernel forward recomputation plus VJP of `mhc_coeffs` (evaluating matrix multiplications
        for activation cotangents `d_x` [einsum: `tc,fc->tf`], parameter gradients `d_phi` [einsum: `tf,tc->fc`],
        and permutation gradients [einsum: `tij,pij->tp`]), scaling forward GEMMs by 2x:
        Total FLOPs = `2 * (2 * tokens * flattened_size * phi_cols + 2 * tokens * num_permutations * streams^2)`.
      - Transcendentals:
        Gating and softmax evaluations across tokens
        = `T * (k + k + P) = tokens * (2 * streams + num_permutations)`.
      - Bytes Accessed:
        Read `x` (`T * k * d * 2`) + read `d_x_acc` (`T * k * d * 2`) + write `d_x` (`T * k * d * 2`)
        + read `phi` and write `d_phi` (`2 * (k*d) * phi_cols * 4`)
        = `3 * tokens * streams * embedding * 2 + 2 * flattened_size * phi_cols * 4` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(
        2
        * (
            2 * self.tokens * self.flattened_size * self.phi_cols
            + 2 * self.tokens * self.num_permutations * self.streams * self.streams
        )
    )
    transcendentals = int(self.tokens * (2 * self.streams + self.num_permutations))
    bytes_accessed = int(
        3 * self.tokens * self.streams * self.embedding * 2 + 2 * self.flattened_size * self.phi_cols * 4
    )
    return pl.CostEstimate(flops=flops, transcendentals=transcendentals, bytes_accessed=bytes_accessed)

  def post_apply_bwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the backward post-apply kernel.

    Mathematical Derivations:
      - FLOPs:
        Backward VJP of post-apply evaluates transposed residual contraction `d_x = residual^T @ d_output`
        [einsum: `tkj,tjd->tkd`] (`2 * T * k^2 * d`), gate reduction `d_layer_output = sum_s (h_post * d_output)`
        [einsum: `ts,tsd->td`] (`2 * T * k * d`), and feature reductions `d_h_post` [einsum: `td,tsd->ts`]
        (`2 * T * k * d`) and `d_residual` [einsum: `tjd,tkd->tjk`] (`2 * T * k^2 * d`):
        Total FLOPs = `4 * tokens * streams^2 * embedding + 4 * tokens * streams * embedding`
        = `2 * (2 * tokens * streams^2 * embedding + 2 * tokens * streams * embedding)`.
      - Transcendentals:
        0.
      - Bytes Accessed:
        Read `x` (`T * k * d * 2`) + read `d_output` (`T * k * d * 2`) + write `d_x` (`T * k * d * 2`)
        + read/write `layer_output` and `d_layer_output` (`2 * tokens * embedding * 4`)
        = `3 * tokens * streams * embedding * 2 + 2 * tokens * embedding * 4` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(
        2
        * (
            2 * self.tokens * self.streams * self.streams * self.embedding
            + 2 * self.tokens * self.streams * self.embedding
        )
    )
    bytes_accessed = int(3 * self.tokens * self.streams * self.embedding * 2 + 2 * self.tokens * self.embedding * 4)
    return pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcWeights:
  """Structured layer weights container for mHC-lite."""

  norm_scale: jax.Array
  pre_alpha: jax.Array
  pre_bias: jax.Array
  pre_scale: jax.Array
  post_alpha: jax.Array
  post_bias: jax.Array
  post_scale: jax.Array
  res_alpha: jax.Array
  res_bias: jax.Array
  res_scale: jax.Array

  def fold_norm_scale(self) -> jax.Array:
    return fold_norm_scale(self.norm_scale, self.pre_alpha, self.post_alpha, self.res_alpha)

  def to_coeff_params(self) -> "MhcCoeffParams":
    return MhcCoeffParams(
        phi=self.fold_norm_scale(),
        pre_scale=self.pre_scale,
        pre_bias=self.pre_bias,
        post_scale=self.post_scale,
        post_bias=self.post_bias,
        res_scale=self.res_scale,
        res_bias=self.res_bias,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcCoeffParams:
  """Parameters required by the coefficient kernel."""

  phi: jax.Array
  pre_scale: jax.Array
  pre_bias: jax.Array
  post_scale: jax.Array
  post_bias: jax.Array
  res_scale: jax.Array
  res_bias: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcCoeffOutputs:
  """Outputs generated by the coefficient kernel."""

  h_pre: jax.Array
  h_post: jax.Array
  residual: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcCoeffGradients:
  """Backward gradients matching MhcCoeffParams."""

  phi: jax.Array
  pre_scale: jax.Array
  pre_bias: jax.Array
  post_scale: jax.Array
  post_bias: jax.Array
  res_scale: jax.Array
  res_bias: jax.Array


def whole(shape: tuple[int, ...]) -> pl.BlockSpec:
  """Returns a full-array BlockSpec for values that stay VMEM-resident."""
  return pl.BlockSpec(shape, lambda _: tuple(0 for _ in shape))


def token_block_spec(shape: tuple[int, ...], block_size: int) -> pl.BlockSpec:
  """Returns a BlockSpec tiling the leading token dimension."""
  block_shape = (block_size,) + shape[1:]
  return pl.BlockSpec(block_shape, lambda i: (i,) + tuple(0 for _ in shape[1:]))


def feature_tiled_block_spec(
    shape: tuple[int, ...],
    block_size: int,
    feature_block_size: int,
    tiled_feature: bool = True,
) -> pl.BlockSpec:
  """Returns a 2D BlockSpec over (token_idx, feature_idx) grid."""
  if tiled_feature:
    block_shape = (block_size,) + shape[1:-1] + (feature_block_size,)
    return pl.BlockSpec(
        block_shape,
        lambda token, feature: (token,) + tuple(0 for _ in shape[1:-1]) + (feature,),
    )
  block_shape = (block_size,) + shape[1:]
  return pl.BlockSpec(
      block_shape,
      lambda token, feature: (token,) + tuple(0 for _ in shape[1:]),
  )


def fold_norm_scale(norm_scale, pre_alpha, post_alpha, res_alpha) -> jax.Array:
  """Folds the RMSNorm channel scale into the three projections."""
  alpha = jnp.concatenate((pre_alpha, post_alpha, res_alpha), axis=-1)
  return norm_scale.astype(jnp.float32)[:, None] * alpha.astype(jnp.float32)


def fused_project_and_norm(flattened_x: jax.Array, phi: jax.Array, rms_epsilon: float) -> jax.Array:
  """Computes linear projection fused with RMSNorm on the fly."""
  projected = jnp.dot(flattened_x, phi.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
  flattened_f32 = flattened_x.astype(jnp.float32)
  mean_square = jnp.mean(flattened_f32 * flattened_f32, axis=-1, keepdims=True)
  return projected * jax.lax.rsqrt(mean_square + rms_epsilon)


def compute_sigmoid_gate(
    logits: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    *,
    multiplier: float = 1.0,
    epsilon: float = 0.0,
) -> jax.Array:
  """Computes scaled/biased sigmoid gating."""
  gate = jax.nn.sigmoid(scale.astype(jnp.float32) * logits + bias.astype(jnp.float32))
  if multiplier != 1.0:
    gate = multiplier * gate
  if epsilon != 0.0:
    gate = gate + epsilon
  return gate


def compute_residual_permutations(
    logits: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    permutations: jax.Array,
    dims: MhcDims,
) -> jax.Array:
  """Computes softmax permutation weighting and contraction."""
  weights = jax.nn.softmax(
      scale.astype(jnp.float32) * logits + bias.astype(jnp.float32),
      axis=-1,
  )
  return jnp.dot(
      weights,
      permutations.reshape(dims.num_permutations, dims.streams * dims.streams).astype(jnp.float32),
  ).reshape(dims.tokens, dims.streams, dims.streams)


def mhc_coeffs(
    x: jax.Array,
    coeff_params: MhcCoeffParams,
    permutations: jax.Array,
    *,
    rms_epsilon: float,
    pre_mapping_epsilon: float,
) -> MhcCoeffOutputs:
  """Computes all mHC-lite coefficients without materializing normalized x."""
  tokens, streams, embedding = x.shape
  dims = MhcDims(tokens=tokens, streams=streams, embedding=embedding, num_permutations=permutations.shape[0])
  flattened = x.reshape(tokens, dims.flattened_size)
  projected = fused_project_and_norm(flattened, coeff_params.phi, rms_epsilon)

  h_pre = compute_sigmoid_gate(
      projected[:, dims.pre_slice],
      coeff_params.pre_scale,
      coeff_params.pre_bias,
      multiplier=1.0,
      epsilon=pre_mapping_epsilon,
  )
  h_post = compute_sigmoid_gate(
      projected[:, dims.post_slice],
      coeff_params.post_scale,
      coeff_params.post_bias,
      multiplier=2.0,
      epsilon=0.0,
  )
  residual = compute_residual_permutations(
      projected[:, dims.res_slice],
      coeff_params.res_scale,
      coeff_params.res_bias,
      permutations,
      dims,
  )
  return MhcCoeffOutputs(h_pre=h_pre, h_post=h_post, residual=residual)


def pre_apply(x: jax.Array, h_pre: jax.Array) -> jax.Array:
  """Collapses the stream dimension before the wrapped model branch."""
  h_pre_f32 = h_pre.astype(jnp.float32)
  return jnp.sum(h_pre_f32[:, :, None] * x.astype(jnp.float32), axis=1).astype(jnp.bfloat16)


def post_apply(
    x: jax.Array,
    layer_output: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
) -> jax.Array:
  """Broadcasts the branch output and applies the residual stream mixing."""
  residual_mix = jnp.einsum(
      "tkj,tkd->tjd",
      residual.astype(jnp.bfloat16),
      x,
      preferred_element_type=jnp.float32,
  )
  post_mix = h_post.astype(jnp.float32)[:, :, None] * layer_output.astype(jnp.float32)[:, None, :]
  return (residual_mix + post_mix).astype(jnp.bfloat16)


def post_apply_bwd_pointwise(
    d_output: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes non-reduced pointwise gradients for post application."""
  d_output_f32 = d_output.astype(jnp.float32)
  d_x = jnp.einsum(
      "tkj,tjd->tkd",
      residual.astype(jnp.bfloat16),
      d_output_f32,
      preferred_element_type=jnp.float32,
  )
  d_layer_output = jnp.sum(h_post[:, :, None] * d_output_f32, axis=1)
  return d_x, d_layer_output


def post_apply_bwd_reductions(
    d_output: jax.Array,
    layer_output: jax.Array,
    x: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes feature-reduced gradients for post gating and residuals."""
  d_output_f32 = d_output.astype(jnp.float32)
  d_h_post = jnp.sum(layer_output[:, None, :] * d_output_f32, axis=-1)
  d_residual = jnp.einsum(
      "tjd,tkd->tjk",
      d_output_f32,
      x,
      preferred_element_type=jnp.float32,
  ).transpose(0, 2, 1)
  return d_h_post, d_residual


def validate_token_block_size(tokens: int, block_size: int, *, name: str) -> None:
  """Validates a token-axis Pallas block size."""
  if block_size < 8 or block_size % 8:
    raise UnsupportedInputError(f"{name} must be a positive multiple of 8; got {block_size}.")
  if tokens % block_size:
    raise UnsupportedInputError(f"The per-device token count ({tokens}) must be divisible by {name} ({block_size}).")


def validate_feature_block_size(embedding: int, block_size: int) -> None:
  """Validates the feature tile used by the post-application backward."""
  if block_size < 128 or block_size % 128:
    raise UnsupportedInputError(f"bwd_feature_block_size must be a positive multiple of 128; got {block_size}.")
  if embedding % block_size:
    raise UnsupportedInputError(
        f"The embedding dimension ({embedding}) must be divisible by bwd_feature_block_size ({block_size})."
    )


def validate_inputs(
    x: jax.Array,
    block_size: int,
    permutations_shape: tuple[int, ...] | None = None,
    *,
    block_size_name: str = "block_size",
) -> None:
  """Validates the shape, dtype, and forward token block constraints."""
  if x.dtype != jnp.bfloat16:
    raise UnsupportedInputError(f"The mHC Pallas kernel requires bfloat16 activations; got {x.dtype}.")
  if x.ndim != 4:
    raise UnsupportedInputError(f"Expected x to have shape (batch, sequence, streams, embedding); got {x.shape}.")
  batch, sequence, streams, embedding = x.shape
  if streams != 4 or (permutations_shape is not None and permutations_shape != (24, 4, 4)):
    raise UnsupportedInputError(
        "The optimized mHC Pallas kernel currently supports mHC-lite with"
        f" expansion rate 4 only; got x.shape={x.shape} and permutations.shape={permutations_shape}."
    )
  if embedding % 128:
    raise UnsupportedInputError(f"The embedding dimension must be divisible by 128; got {embedding}.")
  validate_token_block_size(batch * sequence, block_size, name=block_size_name)
