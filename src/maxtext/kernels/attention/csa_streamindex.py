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

"""Pallas TPU kernel for DeepSeek-V4 CSA StreamIndex score computation."""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def csa_streamindex_score_kernel(
    q_ref,        # [block_q, num_heads, head_dim]
    k_ref,        # [block_w, head_dim]
    w_ref,        # [block_q, num_heads]
    out_ref,      # [block_q, block_w]
    *,
    softmax_scale: float,
):
  """Pallas TPU kernel for fused indexer score calculation.

  Computes scores directly in VMEM without materializing intermediate
  [b, h, s, w] tensors in HBM.
  """
  q = q_ref[...]
  k = k_ref[...]
  w = w_ref[...]

  # QK dot product: [block_q, num_heads, head_dim] x [block_w, head_dim] -> [block_q, num_heads, block_w]
  scores = jnp.einsum(
      "shd,wd->shw",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )
  scores = jnp.maximum(scores, 0.0) * softmax_scale
  scores = scores * w[:, :, None].astype(jnp.float32)
  out = jnp.sum(scores, axis=1)
  out_ref[...] = out.astype(out_ref.dtype)


def csa_streamindex_score(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    *,
    softmax_scale: float,
    block_q: int = 128,
    block_w: int = 512,
    interpret: bool = False,
) -> jax.Array:
  """Computes CSA StreamIndex scores using a fused Pallas TPU kernel.

  Args:
    q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim].
    compressed: Compressed KV tensor of shape [batch_size, compressed_len, head_dim].
    weights: Indexer weights tensor of shape [batch_size, seq_len, num_heads].
    softmax_scale: Scaling factor applied post-ReLU (typically head_dim**-0.5).
    block_q: Query sequence block size (default 128).
    block_w: Compressed window block size (default 512).
    interpret: If True, executes via JAX interpreter on CPU.

  Returns:
    Index scores tensor of shape [batch_size, seq_len, compressed_len] in float32.
  """
  batch_size, seq_len, num_heads, head_dim = q.shape
  _, compressed_len, comp_head_dim = compressed.shape
  assert comp_head_dim == head_dim, f"{comp_head_dim=} != {head_dim=}"
  assert weights.shape == (batch_size, seq_len, num_heads), f"{weights.shape=} != {(batch_size, seq_len, num_heads)=}"

  q = jax.lax.stop_gradient(q)
  compressed = jax.lax.stop_gradient(compressed)
  weights = jax.lax.stop_gradient(weights)

  padded_s = ((seq_len + block_q - 1) // block_q) * block_q
  padded_w = ((compressed_len + block_w - 1) // block_w) * block_w

  if padded_s > seq_len:
    pad_s = padded_s - seq_len
    q = jnp.pad(q, ((0, 0), (0, pad_s), (0, 0), (0, 0)))
    weights = jnp.pad(weights, ((0, 0), (0, pad_s), (0, 0)))
  if padded_w > compressed_len:
    pad_w = padded_w - compressed_len
    compressed = jnp.pad(compressed, ((0, 0), (0, pad_w), (0, 0)))

  grid = (batch_size, padded_s // block_q, padded_w // block_w)

  in_specs = [
      pl.BlockSpec((None, block_q, num_heads, head_dim), lambda b, i, j: (b, i, 0, 0)),
      pl.BlockSpec((None, block_w, head_dim), lambda b, i, j: (b, j, 0)),
      pl.BlockSpec((None, block_q, num_heads), lambda b, i, j: (b, i, 0)),
  ]
  out_specs = pl.BlockSpec((None, block_q, block_w), lambda b, i, j: (b, i, j))

  out = pl.pallas_call(
      functools.partial(
          csa_streamindex_score_kernel,
          softmax_scale=softmax_scale,
      ),
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
      out_shape=jax.ShapeDtypeStruct((batch_size, padded_s, padded_w), jnp.float32),
      interpret=interpret,
  )(q, compressed, weights)

  return jax.lax.stop_gradient(out[:, :seq_len, :compressed_len])


def reference_csa_streamindex_score(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    *,
    softmax_scale: float,
) -> jax.Array:
  """Reference score computation matching the pure JAX einsum path.

  Args:
    q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim].
    compressed: Compressed KV tensor of shape [batch_size, compressed_len, head_dim].
    weights: Indexer weights tensor of shape [batch_size, seq_len, num_heads].
    softmax_scale: Scaling factor applied post-ReLU.

  Returns:
    Index scores tensor of shape [batch_size, seq_len, compressed_len] in float32.
  """
  b, s, h, d = q.shape
  _, w, _ = compressed.shape
  q_trans = jnp.transpose(q, (0, 2, 1, 3)).astype(jnp.float32)
  compressed_kv = jnp.expand_dims(compressed, axis=1)
  compressed_kv = jnp.broadcast_to(compressed_kv, (b, h, w, d)).astype(jnp.float32)
  scores = jnp.einsum("bhsd,bhwd->bhsw", q_trans, compressed_kv)
  scores = jax.nn.relu(scores) * softmax_scale
  return jnp.einsum("bhsw,bsh->bsw", scores, weights.astype(jnp.float32))
