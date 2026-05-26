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

from typing import Callable
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config, DecoderBlockType
from maxtext.common.common_types import HyperConnectionType
from maxtext.layers.initializers import default_bias_init, default_scalar_init, nd_dense_init
from maxtext.layers.normalizations import DeepSeekV4UnweightedRMSNorm, RMSNorm


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


def sinkhorn(t, iters=20, eps=1e-12):
  """Computes the Sinkhorn normalization of a matrix (rows and columns sum to 1)."""
  # Use float32 precision for numerical stability during alternating L1 row/column normalizations.
  # val: [B, S, H, H]
  initial_dtype = t.dtype
  t = t.astype(jnp.float32)

  # Column normalization first (sum along axis -2) matching Xie et al. Equation 8 initialization.
  t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)

  def body_fun(i, val):
    # L1 Normalization: val / (sum(val) + eps) matching the exact denominator addition.
    # Normalize rows (axis -1)
    val = val / (jnp.sum(val, axis=-1, keepdims=True) + eps)
    # Normalize columns (axis -2)
    val = val / (jnp.sum(val, axis=-2, keepdims=True) + eps)
    return val

  # Use lax.fori_loop for an efficient, JIT-friendly loop over exactly iters - 1 steps.
  t = jax.lax.fori_loop(0, iters - 1, body_fun, t)
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
    if getattr(self.config, "decoder_block", None) == DecoderBlockType.DEEPSEEK_V4:
      self.mhc_norm = DeepSeekV4UnweightedRMSNorm(
          eps=self.config.normalization_layer_epsilon,
          dtype=self.config.dtype,
      )
    else:
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

    # Weight matrices
    scale_init = nd_dense_init(1.0, "fan_in", "normal")
    in_axis = 0
    out_axis = 1
    weight_sharding_axis_name = ("activation_embed", None)
    self.res_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k * self.dim, self.k * self.k),
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
        default_bias_init(self.rngs.params(), (self.k, self.k), self.weight_dtype),
        out_sharding=(None, None),
    )
    self.pre_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.post_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        out_sharding=(None,),
    )

  def res_mapping(self, x: Array):
    """Helper function for residual mapping."""
    # In MaxText, we match weight precision to activations before Matmul.
    # x: [B, S, H * D] representing sequence token features.
    # res_alpha: [H * D, H * H]
    # res_beta: [H, H]
    res_alpha = jnp.asarray(self.res_alpha[...], self.dtype)
    res_beta = jnp.asarray(self.res_beta[...], self.dtype)
    res_alpha_scale = jnp.asarray(self.res_alpha_scale[...], self.dtype)
    h_res = jnp.einsum("bsm,mn -> bsn", x, res_alpha, precision=self.matmul_precision)
    b, s, _ = h_res.shape
    h_res = jnp.reshape(h_res, (b, s, self.k, self.k))
    intermediate = res_alpha_scale * h_res + res_beta[None, None, :, :]
    # Apply softmax pre-normalization along the trailing axis matching the exact initialization.
    # intermediate: [B, S, H, H]
    intermediate = jax.nn.softmax(intermediate, axis=-1) + self.config.hc_eps
    output = sinkhorn(intermediate, self.sinkhorn_iterations, eps=self.config.hc_eps)
    return output

  def mapping(self, x: Array, alpha_scale: Array, alpha: Array, beta: Array, scale: float, eps: float = 0.0):
    """Helper function for both pre and post mappings."""
    # In MaxText, we match weight precision to activations before Matmul
    alpha = jnp.asarray(alpha, self.dtype)
    beta = jnp.asarray(beta, self.dtype)
    alpha_scale = jnp.asarray(alpha_scale, self.dtype)
    # Apply projection: (b, s, e*d) @ (e*d, e) -> (b, s, e)
    h = jnp.einsum("bsm,me -> bse", x, alpha, precision=self.matmul_precision)
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

    # 1. Flatten the tensor, and RMS normalization
    norm_x = self.mhc_norm(jnp.reshape(x, (b, s, k * d)))

    # 2. Pre mapping
    pre_mapping = self.mapping(
        norm_x,
        self.pre_alpha_scale[...],
        self.pre_alpha[...],
        self.pre_beta[...],
        1.0,
        self.config.hc_eps,
    )
    layer_input = jnp.einsum(
        "bsed,bse -> bsd", x.astype(jnp.float32), pre_mapping.astype(jnp.float32), precision=self.matmul_precision
    )
    layer_input = layer_input.astype(self.dtype)

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

    # 5. Post mapping (multiplied by 2.0 matching post_scale)
    post_mapping = self.mapping(
        norm_x,
        self.post_alpha_scale[...],
        self.post_alpha[...],
        self.post_beta[...],
        2.0,
        0.0,
    )
    post_out = jnp.einsum(
        "bsd,bse -> bsed",
        layer_out.astype(jnp.float32),
        post_mapping.astype(jnp.float32),
        precision=self.matmul_precision,
    )

    # 6. Residual mapping, res_out shape as [batch, seq, expansion_rate, emb]
    res_mapping = self.res_mapping(norm_x)
    # Transposed residual mixing (bsme @ bsmd -> bsed) matching Xie et al. Equation 8
    # representing the projection index: comb.T @ residual stream values.
    # res_mapping: [B, S, H_src, H_dest]
    # x: [B, S, H_src, D]
    # res_out: [B, S, H_dest, D]
    res_out = jnp.einsum(
        "bsme,bsmd -> bsed", res_mapping.astype(jnp.float32), x.astype(jnp.float32), precision=self.matmul_precision
    )
    output = res_out + post_out
    return output.astype(self.dtype), metadata
