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

"""Shared operations and configuration for the mHC-lite Pallas kernels."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


DEFAULT_BLOCK_SIZE = 64
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024
PARALLEL_DIMENSION_SEMANTICS = ("parallel",)
SEQUENTIAL_DIMENSION_SEMANTICS = ("arbitrary",)
MHCContext = tuple[jax.Array, jax.Array, jax.Array]


def whole(shape):
  """Returns a full-array BlockSpec for values that stay VMEM-resident."""
  return pl.BlockSpec(shape, lambda _: tuple(0 for _ in shape))


def fold_norm_scale(norm_scale, pre_alpha, post_alpha, res_alpha):
  """Folds the RMSNorm channel scale into the three projections."""
  alpha = jnp.concatenate((pre_alpha, post_alpha, res_alpha), axis=-1)
  return norm_scale.astype(jnp.float32)[:, None] * alpha.astype(jnp.float32)


def mhc_coeffs(
    x,
    phi,
    pre_scale,
    pre_bias,
    post_scale,
    post_bias,
    res_scale,
    res_bias,
    permutations,
    *,
    rms_epsilon,
    pre_mapping_epsilon,
):
  """Computes all mHC-lite coefficients without materializing normalized x."""
  tokens, streams, embedding = x.shape
  flattened = x.reshape(tokens, streams * embedding)
  projected = jnp.dot(flattened, phi.astype(jnp.bfloat16), preferred_element_type=jnp.float32)

  flattened_f32 = flattened.astype(jnp.float32)
  mean_square = jnp.mean(flattened_f32 * flattened_f32, axis=-1, keepdims=True)
  projected = projected * jax.lax.rsqrt(mean_square + rms_epsilon)

  pre_logits = projected[:, :streams]
  post_logits = projected[:, streams : 2 * streams]
  res_logits = projected[:, 2 * streams :]
  h_pre = jax.nn.sigmoid(pre_scale.astype(jnp.float32) * pre_logits + pre_bias.astype(jnp.float32)) + pre_mapping_epsilon
  h_post = 2.0 * jax.nn.sigmoid(post_scale.astype(jnp.float32) * post_logits + post_bias.astype(jnp.float32))
  weights = jax.nn.softmax(
      res_scale.astype(jnp.float32) * res_logits + res_bias.astype(jnp.float32),
      axis=-1,
  )
  permutation_count = permutations.shape[0]
  residual = jnp.dot(
      weights,
      permutations.reshape(permutation_count, streams * streams).astype(jnp.float32),
  ).reshape(tokens, streams, streams)
  return h_pre, h_post, residual


def pre_apply(x, h_pre):
  """Collapses the stream dimension before the wrapped model branch."""
  h_pre_f32 = h_pre.astype(jnp.float32)
  return sum(
      h_pre_f32[:, stream : stream + 1] * x[:, stream, :].astype(jnp.float32) for stream in range(x.shape[1])
  ).astype(jnp.bfloat16)


def post_apply(x, layer_output, h_post, residual):
  """Broadcasts the branch output and applies the residual stream mixing."""
  residual_mix = jnp.einsum(
      "tkj,tkd->tjd",
      residual.astype(jnp.bfloat16),
      x,
      preferred_element_type=jnp.float32,
  )
  post_mix = h_post.astype(jnp.float32)[:, :, None] * layer_output.astype(jnp.float32)[:, None, :]
  return (residual_mix + post_mix).astype(jnp.bfloat16)


def validate_inputs(x, block_size, permutations_shape=None):
  """Validates the shape and dtype constraints of the tuned kernel."""
  if x.dtype != jnp.bfloat16:
    raise ValueError(f"The mHC Pallas kernel requires bfloat16 activations; got {x.dtype}.")
  if x.ndim != 4:
    raise ValueError(f"Expected x to have shape (batch, sequence, streams, embedding); got {x.shape}.")
  batch, sequence, streams, embedding = x.shape
  if streams != 4 or (permutations_shape is not None and permutations_shape != (24, 4, 4)):
    raise ValueError(
        "The optimized mHC Pallas kernel currently supports mHC-lite with expansion rate 4 only; "
        f"got x.shape={x.shape} and permutations.shape={permutations_shape}."
    )
  if embedding % 128:
    raise ValueError(f"The embedding dimension must be divisible by 128; got {embedding}.")
  tokens = batch * sequence
  if block_size < 8 or block_size % 8:
    raise ValueError(f"block_size must be a positive multiple of 8; got {block_size}.")
  if tokens % block_size:
    raise ValueError(f"The per-device token count ({tokens}) must be divisible by block_size ({block_size}).")
