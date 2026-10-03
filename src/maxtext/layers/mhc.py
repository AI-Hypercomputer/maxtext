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

import functools
import itertools
import math
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from maxtext.common.common_types import Array, Config
from maxtext.common.common_types import HyperConnectionType
from maxtext.kernels.mhc import api as mhc_kernel
from maxtext.layers.initializers import default_bias_init, default_scalar_init, nd_dense_init
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils.sharding import get_logical_axis_rules, logical_to_mesh_axes


@functools.lru_cache(maxsize=None)
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

    if getattr(self.config, "use_mhc_pallas_kernel", False) and not self.config.enable_mhc_lite:
      raise ValueError("use_mhc_pallas_kernel=True requires enable_mhc_lite=True.")

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

  def _get_mhc_weights(self) -> mhc_kernel.MhcWeights:
    """Collects layer parameters into a structured MhcWeights PyTree."""
    return mhc_kernel.MhcWeights(
        norm_scale=jnp.asarray(self.mhc_norm.scale[...], self.dtype),
        pre_alpha=jnp.asarray(self.pre_alpha[...], self.dtype),
        pre_bias=jnp.asarray(self.pre_beta[...], self.dtype),
        pre_scale=jnp.asarray(self.pre_alpha_scale[...], self.dtype),
        post_alpha=jnp.asarray(self.post_alpha[...], self.dtype),
        post_bias=jnp.asarray(self.post_beta[...], self.dtype),
        post_scale=jnp.asarray(self.post_alpha_scale[...], self.dtype),
        res_alpha=jnp.asarray(self.res_alpha[...], self.dtype),
        res_bias=jnp.asarray(self.res_beta[...], self.dtype),
        res_scale=jnp.asarray(self.res_alpha_scale[...], self.dtype),
    )

  def _kernel_token_axes(self):
    """Returns the mesh axes of the batch and length dims, or None if unsharded.

    Every mHC op is token-local (the RMS norm, projections and gates all reduce
    over `streams * embedding` of a single token), so the kernels can run
    independently on each batch/length shard. The embedding dim must stay whole.
    """
    if self.mesh is None:
      return None
    rules = get_logical_axis_rules() or self.config.logical_axis_rules
    token_axes = tuple(logical_to_mesh_axes(("activation_batch", "activation_length"), mesh=self.mesh, rules=rules))
    if all(axis in (None, ()) for axis in token_axes):
      return None
    return token_axes

  def _sharded_kernel_call(self, kernel_fn, args, in_ranks, out_ranks):
    """Runs a mHC Pallas kernel under a token-sharded `shard_map`.

    A bare `pallas_call` lowers to a Mosaic custom call that GSPMD cannot
    partition, so on a multi-device mesh it fails with "Mosaic kernels cannot be
    automatically partitioned". `shard_map` makes the partitioning explicit.

    Args:
      kernel_fn: Function taking `args` and returning a tuple of arrays.
      args: Positional arguments for `kernel_fn`.
      in_ranks: Per argument, the rank of a token-major activation
        (`[batch, length, ...]`), or None for a replicated pytree (weights,
        permutations).
      out_ranks: Rank of each output, all token-major.

    Returns:
      The tuple returned by `kernel_fn`.
    """
    token_axes = self._kernel_token_axes()
    if token_axes is None:
      return kernel_fn(*args)

    def spec(rank):
      if rank is None:
        return P()
      return P(*token_axes, *([None] * (rank - 2)))

    # check_vma=False is required, not defensive: `pallas_call` builds its
    # `out_shape` from a plain `jax.ShapeDtypeStruct`, which carries no
    # `manual_axis_type`, and check_vma=True rejects that outright. With
    # check_vma=False, the transpose of `shard_map` psums the cotangents of
    # replicated inputs, so weight gradients are reduced across token shards.
    sharded_fn = jax.shard_map(
        kernel_fn,
        mesh=self.mesh,
        in_specs=tuple(spec(rank) for rank in in_ranks),
        out_specs=tuple(spec(rank) for rank in out_ranks),
        check_vma=False,
    )
    return sharded_fn(*args)

  def _kernel_pre(self, x, weights, kernel_config):
    """Token-sharded `mhc_kernel.pre`."""

    def pre_fn(x, weights):
      # Built inside the body, not passed through `shard_map`: the kernel's
      # custom_vjp takes `permutations` as a non-differentiable argument, which
      # must be a constant rather than a tracer.
      permutations = jnp.asarray(get_permutation_matrices(self.k), self.dtype)
      layer_input, context = mhc_kernel.pre(x, weights, permutations, config=kernel_config)
      return layer_input, context.x, context.h_post, context.residual

    layer_input, context_x, h_post, residual = self._sharded_kernel_call(
        pre_fn,
        (x, weights),
        in_ranks=(4, None),
        out_ranks=(3, 4, 3, 4),
    )
    context = mhc_kernel.MhcContext(x=context_x, h_post=h_post, residual=residual, implementation="mosaic")
    return layer_input, context

  def _kernel_post(self, layer_out, context, kernel_config):
    """Token-sharded `mhc_kernel.post`."""
    implementation = context.implementation

    def post_fn(layer_out, context_x, h_post, residual):
      local_context = mhc_kernel.MhcContext(x=context_x, h_post=h_post, residual=residual, implementation=implementation)
      return (mhc_kernel.post(layer_out, local_context, config=kernel_config),)

    (output,) = self._sharded_kernel_call(
        post_fn,
        (layer_out, context.x, context.h_post, context.residual),
        in_ranks=(3, 4, 3, 4),
        out_ranks=(4,),
    )
    return output

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
      permutation_matrices = get_permutation_matrices(self.k).astype(self.dtype)
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

    h_post = None
    h_res = None
    context = None
    use_kernel = self.config.enable_mhc_lite and getattr(self.config, "use_mhc_pallas_kernel", False)
    if use_kernel:
      fwd_block_size = getattr(self.config, "mhc_pallas_kernel_fwd_block_size", 256)
      bwd_block_size = getattr(self.config, "mhc_pallas_kernel_bwd_block_size", 128)
      bwd_feature_block_size = getattr(self.config, "mhc_pallas_kernel_bwd_feature_block_size", 1024)
      kernel_config = mhc_kernel.MhcKernelConfig(
          block_size=fwd_block_size,
          bwd_block_size=bwd_block_size,
          bwd_feature_block_size=bwd_feature_block_size,
          rms_epsilon=self.config.normalization_layer_epsilon,
      )
      weights = self._get_mhc_weights()
      layer_input, context = self._kernel_pre(x, weights, kernel_config)
    else:
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
      # Shared with the Pallas kernel so both paths gate identically. The
      # helper computes in float32; cast back to keep the GEMM in self.dtype.
      pre_mapping = mhc_kernel.compute_sigmoid_gate(
          h_pre,
          self.pre_alpha_scale[...],
          self.pre_beta[...],
          multiplier=1.0,
          epsilon=1e-6,
      ).astype(self.dtype)
      # bskd, bsk -> bsd (fused contracted GEMM)
      layer_input = jnp.einsum(
          "bsk,bskd->bsd",
          pre_mapping,
          x,
          precision=self.matmul_precision,
      )

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

    if use_kernel:
      fwd_block_size = getattr(self.config, "mhc_pallas_kernel_fwd_block_size", 256)
      bwd_block_size = getattr(self.config, "mhc_pallas_kernel_bwd_block_size", 128)
      bwd_feature_block_size = getattr(self.config, "mhc_pallas_kernel_bwd_feature_block_size", 1024)
      kernel_config = mhc_kernel.MhcKernelConfig(
          block_size=fwd_block_size,
          bwd_block_size=bwd_block_size,
          bwd_feature_block_size=bwd_feature_block_size,
      )
      output = self._kernel_post(
          layer_out,
          context,
          kernel_config,
      )
      return output, metadata

    # 5. Post mapping
    post_mapping = mhc_kernel.compute_sigmoid_gate(
        h_post,
        self.post_alpha_scale[...],
        self.post_beta[...],
        multiplier=2.0,
        epsilon=0.0,
    ).astype(self.dtype)
    # Moving away from einsum seems to allow XLA to perform better fusions
    # bsd,bsk -> bskd
    post_out = jnp.expand_dims(layer_out, axis=2) * jnp.expand_dims(post_mapping, axis=3)

    # 6. Residual mapping, res_out shape as [batch, seq, expansion_rate, emb]
    res_mapping = self.res_mapping(h_res)

    # bskm,bskd -> bsmd (fused contracted GEMM)
    res_out = jnp.einsum(
        "bskm,bskd->bsmd",
        res_mapping,
        x,
        precision=self.matmul_precision,
    )
    return res_out + post_out, metadata


class DeepSeek4HyperHead(nnx.Module):
  """Final HC-stream collapse; used by DeepSeek V4 before the shared RMSNorm."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.mhc_expansion_rate = config.mhc_expansion_rate
    self.emb_dim = config.emb_dim
    self.eps = 1e-6

    # Weight matrices
    weight_init = nd_dense_init(1.0, "fan_in", "normal")
    self.hc_fn = nnx.Param(
        weight_init(
            rngs.params(),
            (self.mhc_expansion_rate * self.emb_dim, self.mhc_expansion_rate),
            self.weight_dtype,
            in_axis=0,
            out_axis=1,
        ),
        out_sharding=("activation_embed", None),
    )
    self.hc_base = nnx.Param(
        default_bias_init(rngs.params(), (self.mhc_expansion_rate,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.hc_scale = nnx.Param(
        default_scalar_init(rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )

  def __call__(self, x: Array) -> Array:
    # x shape: [batch, length, k, d]
    b, s, k, d = x.shape
    assert k == self.mhc_expansion_rate
    assert d == self.emb_dim

    flat = jnp.reshape(x, (b, s, k * d))
    flat_f32 = flat.astype(jnp.float32)
    variance = jnp.mean(jnp.square(flat_f32), axis=-1, keepdims=True)
    flat_norm = flat_f32 * jax.lax.rsqrt(variance + self.eps)

    hc_fn = jnp.asarray(self.hc_fn[...], jnp.float32)
    hc_base = jnp.asarray(self.hc_base[...], jnp.float32)
    hc_scale = jnp.asarray(self.hc_scale[...], jnp.float32)

    mixes = jnp.einsum("bsm,mk->bsk", flat_norm, hc_fn, precision=jax.lax.Precision(self.config.matmul_precision))
    pre = jax.nn.sigmoid(mixes * hc_scale[None, None, :] + hc_base[None, None, :]) + self.eps

    x_f32 = x.astype(jnp.float32)
    out = jnp.sum(pre[:, :, :, None] * x_f32, axis=2)
    return out.astype(self.dtype)
