# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Pallas TPU kernel for DeepSeek-V4 CSA StreamIndex Score Computation.

Computes:
  index_scores = sum_h(ReLU(q_h @ comp^T) * softmax_scale * w_h)
with optional in-VMEM causal future masking.

The kernel fuses dot-product, ReLU activation, softmax scaling, head-weight
contraction, and causal future masking into on-chip TPU VMEM registers, avoiding
the materialization of the intermediate [B, H, S, W] 4D tensor in HBM.
"""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def csa_streamindex_score_head_major_kernel(
    q_ref,        # [num_heads, block_q, head_dim]
    k_ref,        # [block_w, head_dim]
    w_ref,        # [block_q, num_heads]
    out_ref,      # [block_q, block_w]
    *,
    softmax_scale: float,
    compress_rate: int = 0,
):
  """Fused Pallas TPU kernel for head-major [num_heads, block_q, head_dim] input with 2D MXU matmul."""
  num_heads, block_q, head_dim = q_ref.shape
  block_w, _ = k_ref.shape

  # Reshape Q to 2D: (num_heads * block_q, head_dim) for native 2D systolic array MXU contraction
  q_2d = q_ref[...].reshape(num_heads * block_q, head_dim)
  k_2d = k_ref[...]

  # 2D MXU matmul: (num_heads * block_q, head_dim) @ (block_w, head_dim)^T -> (num_heads * block_q, block_w)
  s_2d = jnp.einsum("nd,md->nm", q_2d, k_2d, preferred_element_type=jnp.float32)

  # Reshape to (num_heads, block_q, block_w) and apply ReLU
  s = s_2d.reshape(num_heads, block_q, block_w)
  s = jnp.maximum(s, 0.0)

  # Multiply by weights and sum across heads in VMEM
  w = w_ref[...].astype(jnp.float32).transpose(1, 0)[:, :, None]
  s_weighted = jnp.sum(s * w, axis=0) * softmax_scale

  # In-VMEM causal future masking
  if compress_rate > 0:
    i = pl.program_id(1)
    j = pl.program_id(2)
    q_indices = i * block_q + jnp.arange(block_q, dtype=jnp.int32)[:, None]
    k_indices = (j * block_w + jnp.arange(block_w, dtype=jnp.int32)[None, :]) * compress_rate
    future_mask = (k_indices + compress_rate) > (q_indices + 1)
    s_weighted = jnp.where(future_mask, -1e9, s_weighted)

  out_ref[...] = s_weighted.astype(out_ref.dtype)


def _csa_streamindex_score_head_major_pallas_fwd(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    *,
    softmax_scale: float,
    compress_rate: int = 0,
    block_q: int | None = None,
    block_w: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """Forward implementation using fused Pallas TPU kernel for head-major [B, H, S, D] q."""
  batch_size, num_heads, seq_len, head_dim = q.shape
  _, compressed_len, comp_head_dim = compressed.shape
  assert comp_head_dim == head_dim, f"{comp_head_dim=} != {head_dim=}"
  assert weights.shape == (batch_size, seq_len, num_heads), f"{weights.shape=} != {(batch_size, seq_len, num_heads)=}"

  if block_q is None:
    block_q = 128 if num_heads >= 32 else 256
  if block_w is None:
    block_w = 1024 if num_heads >= 32 else 2048

  padded_s = ((seq_len + block_q - 1) // block_q) * block_q
  padded_w = ((compressed_len + block_w - 1) // block_w) * block_w

  if padded_s > seq_len:
    pad_s = padded_s - seq_len
    q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_s), (0, 0)))
    weights = jnp.pad(weights, ((0, 0), (0, pad_s), (0, 0)))
  if padded_w > compressed_len:
    pad_w = padded_w - compressed_len
    compressed = jnp.pad(compressed, ((0, 0), (0, pad_w), (0, 0)))

  grid = (batch_size, padded_s // block_q, padded_w // block_w)

  in_specs = [
      pl.BlockSpec((None, num_heads, block_q, head_dim), lambda b, i, j: (b, 0, i, 0)),
      pl.BlockSpec((None, block_w, head_dim), lambda b, i, j: (b, j, 0)),
      pl.BlockSpec((None, block_q, num_heads), lambda b, i, j: (b, i, 0)),
  ]
  out_specs = pl.BlockSpec((None, block_q, block_w), lambda b, i, j: (b, i, j))

  out = pl.pallas_call(
      functools.partial(
          csa_streamindex_score_head_major_kernel,
          softmax_scale=softmax_scale,
          compress_rate=compress_rate,
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
def csa_streamindex_score_head_major(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    softmax_scale: float,
    compress_rate: int = 0,
    block_q: int | None = None,
    block_w: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """Computes CSA StreamIndex scores using head-major [B, H, S, D] q layout with 2D MXU matmul."""
  return _csa_streamindex_score_head_major_pallas_fwd(
      q,
      compressed,
      weights,
      softmax_scale=softmax_scale,
      compress_rate=compress_rate,
      block_q=block_q,
      block_w=block_w,
      interpret=interpret,
  )


def _csa_streamindex_score_head_major_fwd(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    softmax_scale: float,
    compress_rate: int = 0,
    block_q: int | None = None,
    block_w: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  out = _csa_streamindex_score_head_major_pallas_fwd(
      q,
      compressed,
      weights,
      softmax_scale=softmax_scale,
      compress_rate=compress_rate,
      block_q=block_q,
      block_w=block_w,
      interpret=interpret,
  )
  return out, (q, compressed, weights)


def _csa_streamindex_score_head_major_bwd(
    softmax_scale: float,
    compress_rate: int,
    block_q: int | None,
    block_w: int | None,
    interpret: bool,
    res: tuple[jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  del block_q, block_w, interpret
  q, compressed, weights = res
  _, vjp_fn = jax.vjp(
      functools.partial(
          reference_csa_streamindex_score_head_major,
          softmax_scale=softmax_scale,
          compress_rate=compress_rate,
      ),
      q,
      compressed,
      weights,
  )
  dq, dk, dw = vjp_fn(g)
  return dq, dk, dw


csa_streamindex_score_head_major.defvjp(
    _csa_streamindex_score_head_major_fwd, _csa_streamindex_score_head_major_bwd
)


def reference_csa_streamindex_score_head_major(
    q: jax.Array,
    compressed: jax.Array,
    weights: jax.Array,
    *,
    softmax_scale: float,
    compress_rate: int = 0,
) -> jax.Array:
  """Reference score computation matching the pure JAX einsum path for head-major q."""
  scores = jnp.einsum("bhsd,bwd->bhsw", q.astype(jnp.float32), compressed.astype(jnp.float32))
  scores = jax.nn.relu(scores) * softmax_scale
  index_scores = jnp.einsum("bhsw,bsh->bsw", scores, weights.astype(jnp.float32))
  if compress_rate > 0:
    seq_len = q.shape[2]
    compressed_len = compressed.shape[1]
    position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    usable_len = compressed_len * compress_rate
    block_positions = position_ids[:, :usable_len:compress_rate]
    future_mask = (block_positions[:, None, :] + compress_rate) > (position_ids[:, :, None] + 1)
    index_scores = jnp.where(future_mask, -1e9, index_scores)
  return index_scores


# Public aliases for standard naming conventions
csa_streamindex_score = csa_streamindex_score_head_major
reference_csa_streamindex_score = reference_csa_streamindex_score_head_major
