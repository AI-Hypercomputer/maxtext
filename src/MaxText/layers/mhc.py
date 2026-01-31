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

import jax
from jax.sharding import Mesh

import functools
import jax.numpy as jnp
from flax import nnx
from typing import Callable
from MaxText.common_types import Config
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.initializers import nd_dense_init, default_bias_init
from MaxText.sharding import maybe_shard_with_logical
from MaxText.common_types import HyperConnectionType


def get_functions(expansion_rate: int):
  """
  Creates functions to broadcast a single feature stream into multiple
  parallel paths (expand) and aggregate them back (reduce).
  """

  def expand(x: jnp.ndarray):
    # (batch, length, dim) -> (streams, batch, length, dim)
    return jnp.repeat(jnp.expand_dims(x, axis=0), expansion_rate, axis=0)

  def reduce(x: jnp.ndarray):
    # (streams, batch, length, dim) -> (batch, length, dim)
    return jnp.sum(x, axis=0)

  return expand, reduce


def sinkhorn(t, iters=20):
  """
  Computes the Sinkhorn normalization of a matrix (rows and columns sum to 1).
  """
  # Use float32 precision for numerical stability during normalization
  initial_dtype = t.dtype
  t = t.astype(jnp.float32)

  # Initial softmax along the rows (dim -2)
  # Makes values to be positive and sum up to 1 across columns
  t = jax.nn.softmax(t, axis=-2)

  def body_fun(i, val):
    # L1 Normalization: val / sum(val) with clipping of denominator
    # Normalize rows (axis -1)
    val = val / jnp.clip(jnp.sum(val, axis=-1, keepdims=True), min=1e-12)
    # Normalize columns (axis -2)
    val = val / jnp.clip(jnp.sum(val, axis=-2, keepdims=True), min=1e-12)
    return val

  # Use lax.fori_loop for an efficient, JIT-friendly loop
  t = jax.lax.fori_loop(0, iters, body_fun, t)
  return t.astype(initial_dtype)


# TODO(ranran): Add sharding constraints
class ManifoldConstrainedHyperConnections(nnx.Module):
  """Manifold-Constrained Hyper Connection (mHC)."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
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
    self.weight_dtype = self.config.weight_dtype

    # Norm layer
    self.mhc_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    # Scalers
    self.mhc_res_alpha_scale = self.config.mhc_res_alpha_scale
    self.mhc_pre_alpha_scale = self.config.mhc_pre_alpha_scale
    self.mhc_post_alpha_scale = self.config.mhc_post_alpha_scale

    # Weight matrices
    scale_init = nd_dense_init(1.0, "fan_in", "normal")
    batch_size, sequence_length = self.config.per_device_batch_size, self.config.max_target_length
    in_axis = (0, 1, 2, 3)
    out_axis = 4
    weight_sharding_axis_name = (None, "activation_batch", "activation_norm_length", "activation_embed", None)
    self.res_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k, batch_size, sequence_length, self.dim, self.k * self.k),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        sharding=weight_sharding_axis_name,
    )
    self.pre_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k, batch_size, sequence_length, self.dim, self.k),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        sharding=weight_sharding_axis_name,
    )
    self.post_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k, batch_size, sequence_length, self.dim, self.k),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        sharding=weight_sharding_axis_name,
    )

    # Biases
    self.res_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k, self.k), self.weight_dtype),
        sharding=(None, None),
    )
    self.pre_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        sharding=(None, None),
    )
    self.post_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        sharding=(None, None),
    )

    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )

  def res_mapping(self, x: jnp.ndarray):
    """Helper function for residule mapping."""
    # Apply projection: (k, b, s, d) @ (k, b, s, d, k*k) -> (k*k)
    h_res = jnp.einsum("kbsd,kbsdm -> m", x, self.res_alpha[...])
    h_res = jnp.reshape(h_res, (self.k, self.k))
    intermediate = self.mhc_res_alpha_scale * h_res + self.res_beta[...]
    output = sinkhorn(intermediate, self.sinkhorn_iterations)
    return output

  def mapping(self, x: jnp.ndarray, alpha_scale: jnp.ndarray, alpha: jnp.ndarray, beta: jnp.ndarray, scale: int):
    """Helper function for both pre and post mappings."""
    # Apply projection: (k, b, s, d) @ (k, b, s, d, k) -> (k)
    h = jnp.einsum("kbsd,kbsdm -> m", x, alpha)
    intermediate = alpha_scale * h + beta
    output = scale * jax.nn.sigmoid(intermediate)
    return output

  def __call__(
      self,
      branch_fn: Callable,
      x: jnp.ndarray,
      mhc_type: HyperConnectionType,
      **kwargs,
  ) -> jnp.ndarray:
    # x shape: (k, batch, seq, emb)
    # 1. RMS normalization
    x = self.mhc_norm(x)

    # 2. Pre mapping
    pre_mapping = self.mapping(x, self.mhc_pre_alpha_scale, self.pre_alpha[...], self.pre_beta[...], 1.0)
    layer_input = jnp.einsum("kbsd,k -> bsd", x, pre_mapping)

    # 3. Attention or MLP
    if mhc_type == HyperConnectionType.ATTENTION:
      layer_out, _ = branch_fn(inputs_q=layer_input, inputs_kv=layer_input, **kwargs)
    elif mhc_type == HyperConnectionType.MLP_DENSE:
      layer_out = branch_fn(inputs=layer_input, **kwargs)
    elif mhc_type == HyperConnectionType.MLP_MOE:
      layer_out, _, _ = branch_fn(inputs=layer_input, **kwargs)
    else:
      raise ValueError(f"Unsupported type: {mhc_type}")

    # 4. Post mapping
    post_mapping = self.mapping(x, self.mhc_post_alpha_scale, self.post_alpha[...], self.post_beta[...], 2.0)
    post_out = jnp.einsum("bsd,k -> kbsd", layer_out, post_mapping)

    # 5. Residual mapping, res_out shape as [expansion_rate, batch, seq, emb]
    res_mapping = self.res_mapping(x)
    res_out = jnp.einsum("kbsd,km -> mbsd", x, res_mapping)
    return res_out + post_out
