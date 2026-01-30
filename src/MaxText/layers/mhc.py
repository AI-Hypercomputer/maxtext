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
from typing import Callable, Optional, Any, Union, Dict
from functools import partial


def default(v, d):
  def exists(v):
    return v is not None

  def divisible_by(num, den):
    return (num % den) == 0

  return v if exists(v) else d


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


# TODO: parameter shape & values
# TODO: update sharding
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
    self.mhc_expansion_rate = config.mhc_expansion_rate

    # Parameters
    # Width Connection (Permutation matrix logits)
    h_res_init = jnp.full((self.mhc_expansion_rate, self.mhc_expansion_rate), -8.0)
    # Manually fill diagonal for JAX array
    indices = jnp.arange(self.mhc_expansion_rate)
    h_res_init = h_res_init.at[indices, indices].set(0.0)
    self.H_res_logits = nnx.Param(h_res_init)

    # Pre-branch selection logits
    h_pre_init = jnp.full((1, self.mhc_expansion_rate), -8.0)
    # TODO: how to handle layer index here
    init_idx = default(layer_index, jax.random.randint(rngs.params(), (), 0, self.mhc_expansion_rate))
    self.H_pre_logits = nnx.Param(h_pre_init.at[:, init_idx].set(0.0))

    # 3. Post-branch selection logits
    self.H_post_logits = nnx.Param(jnp.zeros((1, self.mhc_expansion_rate)))

  def width_connection(self, inputs: jnp.ndarray):
    """Processes residual streams and prepares branch input."""
    # 1. Compute Width Weights (Sinkhorn)
    h_res = sinkhorn_log(self.H_res_logits.value, self.sinkhorn_iterations, self.sinkhorn_scaling)
    # Apply permutation across streams: (s t, b l s d -> b l t d)
    residuals_out = jnp.einsum("st,blsd->bltd", h_res, inputs)

    # 2. Compute Pre-branch Selection (Softmax)
    h_pre = jax.nn.softmax(self.H_pre_logits.value, axis=-1)
    # Weighted sum of streams for branch input
    branch_input = jnp.einsum("vs,blsd->blvd", h_pre, inputs)

    # Single view handling, (batch, length, dim)
    branch_input = jnp.squeeze(branch_input, axis=2)

    # 3. Post-branch Weights
    h_post = jax.nn.softmax(self.H_post_logits.value, axis=-1)
    return branch_input, residuals_out, h_post

  def depth_connection(self, branch_output: jnp.ndarray, residuals: jnp.ndarray, beta: jnp.ndarray):
    """Adds branch results back into the multiple residual streams."""
    # branch_output: (batch, length, dim)
    # beta: (views, streams) -> (1, streams)
    if beta.ndim == 2:
      beta = beta[0]  # Take first view
    return jnp.einsum("bld,s->blsd", branch_output, beta)

  def __call__(
      self,
      branch_fn: Callable,
      inputs: jnp.ndarray,
      config: Any,
      mesh: jax.sharding.Mesh,
      logical_axes: tuple,
      *args,
      **kwargs,
  ) -> jnp.ndarray:
    # 1. Width Logic
    branch_input, processed_residuals, beta = self.width_connection(inputs)
    # 2. Execute Branch (Attention or MLP)
    branch_out = branch_fn(branch_input, *args, **kwargs)
    # 3. Depth Logic
    return self.depth_connection(branch_out, processed_residuals, beta)


def get_expand_functions(expansion_rate: int):
  """
  Creates functions to broadcast a single feature stream into multiple
  parallel paths (expand) and aggregate them back (reduce).
  """

  def expand(x: jnp.ndarray):
    # (batch, length, dim) -> (batch, length, streams, dim)
    return jnp.repeat(jnp.expand_dims(x, axis=2), expansion_rate, axis=2)

  def reduce(x: jnp.ndarray):
    # (batch, length, streams, dim) -> (batch, length, dim)
    return jnp.sum(x, axis=2)

  return expand, reduce
