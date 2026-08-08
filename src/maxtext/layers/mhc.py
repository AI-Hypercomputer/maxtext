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

"""DeepSeek Manifold-Constrained Hyper Connections (mHC) Layer."""

import itertools
import math
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config
from maxtext.common.common_types import HyperConnectionType
from maxtext.layers.initializers import default_bias_init, default_scalar_init, nd_dense_init, variable_to_logically_partitioned
from maxtext.layers import nnx_wrappers
from maxtext.layers.normalizations import RMSNorm


def get_permutation_matrices(k: int) -> Array:
  """Generates all permutation matrices of size k.

  Reference: mHC-lite: https://openreview.net/pdf?id=5IJX6kvOif
  Shape: (k!, k, k)
  """
  perms = list(itertools.permutations(range(k)))
  perms_array = jnp.array(perms)
  return jnp.eye(k)[perms_array]


def get_functions(expansion_rate: int):
  """Creates functions to broadcast a single feature stream into multiple

  parallel paths (expand) and aggregate them back (reduce).
  """

  def expand(x: Array):
    # (batch, length, dim) -> (batch, length, streams, dim)
    return jnp.repeat(jnp.expand_dims(x, axis=2), expansion_rate, axis=2).astype(x.dtype)

  def reduce(x: Array):
    # (batch, length, streams, dim) -> (batch, length, dim)
    return jnp.sum(x, axis=2, dtype=x.dtype)

  return expand, reduce


def sinkhorn(t, iters=20):
  """Computes the Sinkhorn normalization of a matrix (rows and columns sum to 1)."""
  # Use float32 precision for numerical stability during normalization
  initial_dtype = t.dtype
  t = t.astype(jnp.float32)
  eps = 1e-6

  t = jax.nn.softmax(t, axis=-1) + eps
  t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)

  for _ in range(iters - 1):
    t = t / (jnp.sum(t, axis=-1, keepdims=True) + eps)
    t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)

  return t.astype(initial_dtype)


class ManifoldConstrainedHyperConnections(nnx.Module):
  """Implements Manifold-Constrained Hyper-Connections (mHC).

  Reference: https://arxiv.org/pdf/2512.24880

  Args:
      config: Configuration object containing hyperparameters.
      dim: The feature dimensionality.
      mesh: The hardware mesh for sharding.
      rngs: Random number generation in NNX.
  """

  def __init__(
      self,
      config: Config,
      dim: int,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.sinkhorn_iterations = config.sinkhorn_iterations
    self.k = config.mhc_expansion_rate
    self.dim = dim
    self.rngs = rngs
    self.mesh = mesh
    self.dtype = self.config.dtype
    self.weight_dtype = self.config.weight_dtype
    self.matmul_precision = jax.lax.Precision(self.config.matmul_precision)

    # Norm layer
    self.mhc_norm = RMSNorm(
        num_features=self.k * self.dim,
        dtype=self.config.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    # Scalars
    self.res_alpha_scale = nnx.Param(
        default_scalar_init(self.rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.pre_alpha_scale = nnx.Param(
        default_scalar_init(self.rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.post_alpha_scale = nnx.Param(
        default_scalar_init(self.rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )

    if self.config.enable_mhc_lite:
      num_perms = math.factorial(self.k)
      res_out_dim = num_perms
      res_beta_shape = (num_perms,)
      res_beta_sharding = (None,)
      self.permutation_matrices = get_permutation_matrices(self.k)
    else:
      res_out_dim = self.k * self.k
      res_beta_shape = (self.k, self.k)
      res_beta_sharding = (None, None)

    # Weight matrices
    scale_init = nd_dense_init(1.0, "fan_in", "normal")
    in_axis = 0
    out_axis = 1
    weight_sharding_axis_name = ("activation_embed", None)
    self.res_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k * self.dim, res_out_dim),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        out_sharding=weight_sharding_axis_name,
    )
    self.pre_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k * self.dim, self.k),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        out_sharding=weight_sharding_axis_name,
    )
    self.post_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k * self.dim, self.k),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        out_sharding=weight_sharding_axis_name,
    )

    # Biases
    self.res_beta = nnx.Param(
        default_bias_init(self.rngs.params(), res_beta_shape, self.weight_dtype),
        out_sharding=res_beta_sharding,
    )
    self.pre_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.post_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        out_sharding=(None,),
    )

  def res_mapping(self, h_res: Array):
    """Helper function for residual mapping after matmul."""
    # In MaxText, we match weight precision to activations before Matmul
    res_beta = jnp.asarray(self.res_beta[...], self.dtype)
    res_alpha_scale = jnp.asarray(self.res_alpha_scale[...], self.dtype)

    if self.config.enable_mhc_lite:
      intermediate = res_alpha_scale * h_res + res_beta[None, None, :]
      # Use float32 for numerical stability during softmax
      weights = jax.nn.softmax(intermediate.astype(jnp.float32), axis=-1).astype(self.dtype)
      # Sum the permutation matrices with the weights
      permutation_matrices = self.permutation_matrices.astype(self.dtype)
      output = jnp.einsum(
          "bsn,nkm -> bskm",
          weights,
          permutation_matrices,
          precision=self.matmul_precision,
      )
      return output
    else:
      b, s, _ = h_res.shape
      h_res = jnp.reshape(h_res, (b, s, self.k, self.k))
      intermediate = res_alpha_scale * h_res + res_beta[None, None, :, :]
      output = sinkhorn(intermediate, self.sinkhorn_iterations)
      return output

  def mapping(self, h: Array, alpha_scale: Array, beta: Array, scale: float, eps: float = 0.0):
    """Helper function for both pre and post mappings after matmul."""
    # In MaxText, we match weight precision to activations before Matmul
    beta = jnp.asarray(beta, self.dtype)
    alpha_scale = jnp.asarray(alpha_scale, self.dtype)
    intermediate = alpha_scale * h + beta[None, None, :]
    output = scale * jax.nn.sigmoid(intermediate) + eps
    return output

  def __call__(
      self,
      norm_fn: Callable,
      branch_fn: Callable,
      x: Array,
      mhc_type: HyperConnectionType,
      **kwargs,
  ) -> Array:
    """Applying manifold-constrained hyper connection based on callable function.

    Args:
        norm_fn: The pre-normalization function to be applied.
        branch_fn: The function to be wrapped by the hyper-connection.
        x: Input tensor of shape `(batch..., dim)`.
        mhc_type: The variant of the connection to apply.
        **kwargs: Additional context passed to the branch function.

    Returns:
        The processed tensor, maintaining the shape of `x`.
    """
    # x shape: [batch, seq, expansion_rate, emb]
    b, s, k, d = x.shape

    with jax.named_scope("mhc_norm"):
      # 1. Flatten the tensor, and RMS normalization
      norm_x = self.mhc_norm(jnp.reshape(x, (b, s, k * d)))

    # Fused Projections
    pre_alpha = jnp.asarray(self.pre_alpha[...], self.dtype)
    post_alpha = jnp.asarray(self.post_alpha[...], self.dtype)
    res_alpha = jnp.asarray(self.res_alpha[...], self.dtype)

    alpha_concat = jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1)

    # MatMul on normalized input
    h_concat = jnp.einsum("bsm,mn -> bsn", norm_x, alpha_concat, precision=self.matmul_precision)

    h_pre = h_concat[..., : self.k]
    h_post = h_concat[..., self.k : 2 * self.k]
    h_res = h_concat[..., 2 * self.k :]

    # 2. Pre mapping
    pre_mapping = self.mapping(
        h_pre,
        self.pre_alpha_scale[...],
        self.pre_beta[...],
        1.0,
        eps=1e-6,
    )
    # Moving away from einsum seems to allow XLA to perform better fusions
    # https://github.com/AI-Hypercomputer/maxtext/pull/4664#discussion_r3677899970
    # bskd, bsk -> bsd
    layer_input = jnp.sum(x * jnp.expand_dims(pre_mapping, axis=3), axis=2)

    # 3. Pre-norm
    layer_input = norm_fn(layer_input)

    # 4. Attention or MLP
    metadata = {}
    if mhc_type == HyperConnectionType.ATTENTION:
      layer_out, _ = branch_fn(inputs_q=layer_input, inputs_kv=layer_input, **kwargs)
    elif mhc_type == HyperConnectionType.MLP_DENSE:
      layer_out = branch_fn(inputs=layer_input, **kwargs)
    elif mhc_type == HyperConnectionType.MLP_MOE:
      layer_out, load_balance_loss, moe_bias_updates = branch_fn(inputs=layer_input, **kwargs)
      metadata["load_balance_loss"] = load_balance_loss
      metadata["moe_bias_updates"] = moe_bias_updates
    else:
      raise ValueError(f"Unsupported type: {mhc_type}")

    # 5. Post mapping
    post_mapping = self.mapping(
        h_post,
        self.post_alpha_scale[...],
        self.post_beta[...],
        2.0,
    )
    # Moving away from einsum seems to allow XLA to perform better fusions
    # bsd,bsk -> bskd
    post_out = jnp.expand_dims(layer_out, axis=2) * jnp.expand_dims(post_mapping, axis=3)

    # 6. Residual mapping, res_out shape as [batch, seq, expansion_rate, emb]
    res_mapping = self.res_mapping(h_res)

    # Moving away from einsum seems to allow XLA to perform better fusions
    # bskd,bskm -> bsmd
    res_out = jnp.sum(jnp.expand_dims(x, axis=3) * jnp.expand_dims(res_mapping, axis=4), axis=2)
    return res_out + post_out, metadata


class DeepSeek4HyperHead(nnx.Module):
  """Implements DeepSeek4 Hyper Head.

  Args:
      config: Configuration object containing hyperparameters.
      dim: The feature dimensionality.
      mesh: The hardware mesh for sharding.
      rngs: Random number generation in NNX.
  """

  def __init__(
      self,
      config: Config,
      dim: int,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.dim = dim
    self.rngs = rngs
    self.mesh = mesh
    self.dtype = self.config.dtype
    self.weight_dtype = self.config.weight_dtype
    self.matmul_precision = jax.lax.Precision(self.config.matmul_precision)
    self.hc_mult = self.config.mhc_expansion_rate

    # Norm layer
    self.input_norm = RMSNorm(
        num_features=self.hc_mult * self.dim,
        dtype=self.config.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    # Weight matrices
    scale_init = nd_dense_init(1.0, "fan_in", "normal")
    in_axis = 0
    out_axis = 1
    weight_sharding_axis_name = ("activation_embed", None)

    self.hc_fn = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.hc_mult * self.dim, self.hc_mult),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        out_sharding=weight_sharding_axis_name,
    )

    # Scalars
    self.hc_base = nnx.Param(
        default_scalar_init(self.rngs.params(), (self.hc_mult,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.hc_scale = nnx.Param(
        default_scalar_init(self.rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )

  def __call__(self, x: Array) -> Array:
    """Applying manifold-constrained hyper connection based on callable function.

    Args:
        x: Input tensor of shape `(batch..., expansion_rate, dim)`.

    Returns:
        The processed tensor, maintaining the shape of `x`.
    """
    b, s, k, d = x.shape

    with jax.named_scope("mhc_norm"):
      # 1. Flatten the tensor, and RMS normalization
      norm_x = self.input_norm(jnp.reshape(x, (b, s, k * d)))

    # Fused Projections
    hc_fn = jnp.asarray(self.hc_fn[...], self.dtype)
    hc_base = jnp.asarray(self.hc_base[...], self.dtype)
    hc_scale = jnp.asarray(self.hc_scale[...], self.dtype)

    # MatMul on normalized input
    h = jnp.einsum("bsm,mn -> bsn", norm_x, hc_fn, precision=self.matmul_precision)

    # 2. Pre mapping
    intermediate = hc_scale * h + hc_base[None, None, :]
    mapping = jax.nn.sigmoid(intermediate)

    # Moving away from einsum seems to allow XLA to perform better fusions
    # bskd, bsk -> bsd
    layer_input = jnp.sum(x * jnp.expand_dims(mapping, axis=3), axis=2)

    return layer_input
