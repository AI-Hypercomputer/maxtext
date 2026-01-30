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

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable
from MaxText.common_types import Config
from MaxText.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from MaxText.layers.initializers import nd_dense_init, default_bias_init


# TODO: unit tests
def sinkhorn_log(logits, iterations=20, scaling=0.05):
  """
  Computes the Sinkhorn normalization in log-space to produce a
  doubly stochastic matrix (rows and columns sum to 1).
  """
  n = logits.shape[-1]
  Z = logits / scaling
  # Target marginals in log-space. log(1) = 0, so we target row/col sums of 1.
  log_marginal = jnp.zeros((n,))

  def body_fn(i, state):
    u, v = state
    # Update u (row) & v (column)
    u = log_marginal - jax.scipy.special.logsumexp(Z + jnp.expand_dims(v, axis=-2), axis=-1)
    v = log_marginal - jax.scipy.special.logsumexp(Z + jnp.expand_dims(u, axis=-1), axis=-2)
    return u, v

  u_init = jnp.zeros(logits.shape[:-1])
  v_init = jnp.zeros(logits.shape[:-1])
  # Execute the fixed iteration loop
  u, v = jax.lax.fori_loop(0, iterations, body_fn, (u_init, v_init))

  # Convert back from log-space to linear space.
  return jnp.exp(Z + jnp.expand_dims(u, axis=-1) + jnp.expand_dims(v, axis=-2))


# TODO: update sharding
# TODO: unit tests
class ManifoldConstrainedHyperConnections(nnx.Module):
  """Manifold-Constrained Hyper Connection (mHC)."""

  def __init__(
      self,
      config: Config,
      rngs: nnx.Rngs,
  ):
    super().__init__()

    self.config = config
    self.sinkhorn_iterations = config.sinkhorn_iterations
    self.sinkhorn_scaling = config.sinkhorn_scaling
    self.k = config.mhc_expansion_rate
    self.mhc_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # Scaler
    self.mhc_res_alpha_scale = self.config.mhc_res_alpha_scale
    self.mhc_pre_alpha_scale = self.config.mhc_pre_alpha_scale
    self.mhc_post_alpha_scale = self.config.mhc_post_alpha_scale

    # Weight matrices
    scale_init = nd_dense_init(1.0, "fan_in", "normal")
    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, self.model_mode)
    self.res_alpha = nnx.Param(
        scale_init(
            rngs.params(),
            (self.k, batch_size, sequence_length, -1, self.k * self.k),
            self.weight_dtype,
        ),
        sharding=(None, "activation_batch", "activation_norm_length", "activation_embed", None),
        in_axis=0,
        out_axis=1,
    )
    self.pre_alpha = nnx.Param(
        scale_init(
            rngs.params(),
            (self.k, batch_size, sequence_length, -1, self.k,),
            self.weight_dtype,
        ),
        sharding=("activation_batch", "activation_norm_length", "activation_embed", None),
    )
    self.post_alpha = nnx.Param(
        scale_init(
            rngs.params(),
            (self.k, batch_size, sequence_length, -1, self.k,),
            self.weight_dtype,
        ),
        sharding=("activation_batch", "activation_norm_length", "activation_embed", None),
    )

    # Bias
    self.res_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        sharding=(None, None),
    )
    self.pre_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        sharding=(None, None),
    )
    self.post_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k, self.k), self.weight_dtype),
        sharding=(None, None),
    )

  def res_mapping(self, x: jnp.ndarray):
    h_res = jnp.einsum("nbsd,nbsdm -> m", x, self.res_alpha)
    h_res = jnp.reshape(h_res, (self.k, self.k))
    intermediate = self.mhc_res_alpha_scale * h_res + self.res_beta
    output = sinkhorn_log(intermediate, self.sinkhorn_iterations, self.sinkhorn_scaling)
    return output

  def mapping(self, x: jnp.ndarray, alpha_scale: jnp.ndarray, alpha: jnp.ndarray, beta: jnp.ndarray, scale: int):
    h = jnp.einsum("nbsd,nbsdm -> m", x, alpha)
    intermediate = alpha_scale * h + beta
    output = scale * jax.nn.sigmoid(intermediate)
    return output

  def __call__(
      self,
      branch_fn: Callable,
      x: jnp.ndarray,
      *args,
      **kwargs,
  ) -> jnp.ndarray:
    # 1. RMS norm, x shape as [expansion_rate, batch, seq, emb]
    x = self.mhc_norm(x)
    # 2. Pre mapping, shape as [1, expansion_rate]
    pre_mapping = self.mapping(x, self.mhc_pre_alpha_scale, self.pre_alpha, self.pre_beta, 1.0)
    layer_input = jnp.einsum("nbsd,n -> bsd", x, pre_mapping)
    # 3. Attention or MLP
    layer_out = branch_fn(layer_input, *args, **kwargs)
    # 4. Post mapping, shape as [1, expansion_rate]
    post_mapping = self.mapping(x, self.mhc_post_alpha_scale, self.post_alpha, self.post_beta, 2.0)
    post_out = jnp.einsum("bsd,n -> nbsd", layer_out, post_mapping)
    # 5. Residual mapping, res_out shape as [expansion_rate, batch, seq, emb]
    res_mapping = self.res_mapping(x)
    res_out = jnp.einsum("nbsd,nm -> mbsd", x, res_mapping)
    return res_out, post_out


def get_expand_functions(expansion_rate: int):
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
