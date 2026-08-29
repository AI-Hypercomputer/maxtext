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
    head_chunk: int = 32,
):
  """Pallas TPU kernel for fused indexer score calculation.

  Accumulates head scores chunked directly into a [block_q, block_w] accumulator
  in VMEM without allocating large [block_q, num_heads, block_w] intermediate buffers.
  """
  q = q_ref[...]
  k = k_ref[...]
  w = w_ref[...]

  block_q, num_heads, head_dim = q.shape
  block_w, _ = k.shape

  acc = jnp.zeros((block_q, block_w), dtype=jnp.float32)

  for h_start in range(0, num_heads, head_chunk):
    h_end = min(h_start + head_chunk, num_heads)
    q_c = q[:, h_start:h_end, :]
    w_c = w[:, h_start:h_end].astype(jnp.float32)

    # [block_q, h_c, head_dim] x [block_w, head_dim] -> [block_q, h_c, block_w]
    scores_c = jnp.einsum(
        "shd,wd->shw",
        q_c,
        k,
        preferred_element_type=jnp.float32,
    )
    scores_c = jnp.maximum(scores_c, 0.0)
    chunk_acc = jnp.sum(scores_c * w_c[:, :, None], axis=1)
    acc = acc + chunk_acc

  out_ref[...] = (acc * softmax_scale).astype(out_ref.dtype)


def _csa_streamindex_score_pallas_fwd(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    *,
    softmax_scale: float,
    block_q: int = 128,
    block_w: int = 1024,
    head_chunk: int = 32,
    interpret: bool = False,
) -> jax.Array:
  """Forward implementation using fused Pallas TPU kernel."""
  batch_size, seq_len, num_heads, head_dim = q.shape
  _, compressed_len, comp_head_dim = compressed.shape
  assert comp_head_dim == head_dim, f"{comp_head_dim=} != {head_dim=}"
  assert weights.shape == (batch_size, seq_len, num_heads), f"{weights.shape=} != {(batch_size, seq_len, num_heads)=}"

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
          head_chunk=head_chunk,
      ),
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      out_shape=jax.ShapeDtypeStruct((batch_size, padded_s, padded_w), jnp.float32),
      interpret=interpret,
  )(q, compressed, weights)

  return out[:, :seq_len, :compressed_len]


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def csa_streamindex_score(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    softmax_scale: float,
    block_q: int = 128,
    block_w: int = 1024,
    head_chunk: int = 32,
    interpret: bool = False,
) -> jax.Array:
  """Computes CSA StreamIndex scores using a fused Pallas TPU kernel.

  Differentiable via jax.custom_vjp: executes fused Pallas kernel in forward pass,
  and evaluates reference autograd in backward pass.

  Args:
    q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim].
    compressed: Compressed KV tensor of shape [batch_size, compressed_len, head_dim].
    weights: Indexer weights tensor of shape [batch_size, seq_len, num_heads].
    softmax_scale: Scaling factor applied post-ReLU (typically head_dim**-0.5).
    block_q: Query sequence block size (default 128).
    block_w: Compressed window block size (default 1024).
    head_chunk: Number of heads processed per accumulation step in VMEM (default 32).
    interpret: If True, executes via JAX interpreter on CPU.

  Returns:
    Index scores tensor of shape [batch_size, seq_len, compressed_len] in float32.
  """
  return _csa_streamindex_score_pallas_fwd(
      q,
      compressed,
      weights,
      softmax_scale=softmax_scale,
      block_q=block_q,
      block_w=block_w,
      head_chunk=head_chunk,
      interpret=interpret,
  )


def _csa_streamindex_score_fwd(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    softmax_scale: float,
    block_q: int = 128,
    block_w: int = 1024,
    head_chunk: int = 32,
    interpret: bool = False,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  out = _csa_streamindex_score_pallas_fwd(
      q,
      compressed,
      weights,
      softmax_scale=softmax_scale,
      block_q=block_q,
      block_w=block_w,
      head_chunk=head_chunk,
      interpret=interpret,
  )
  return out, (q, compressed, weights)


def _csa_streamindex_score_bwd(
    softmax_scale: float,
    block_q: int,
    block_w: int,
    head_chunk: int,
    interpret: bool,
    res: tuple[jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  del block_q, block_w, head_chunk, interpret
  q, compressed, weights = res
  _, vjp_fn = jax.vjp(
      functools.partial(reference_csa_streamindex_score, softmax_scale=softmax_scale),
      q,
      compressed,
      weights,
  )
  dq, dk, dw = vjp_fn(g)
  return dq, dk, dw


csa_streamindex_score.defvjp(_csa_streamindex_score_fwd, _csa_streamindex_score_bwd)


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
