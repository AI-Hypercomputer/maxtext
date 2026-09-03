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

r"""Cosmos 3 Core Attention Block for MaxText.

This module implements Attention of the Cosmos 3 architecture in MaxText:
- Sequence packing logic: Boundary metadata (offsets) and stream mapping to
  distinguish understanding (UND) and generation (GEN) streams without padding.
- Dual-attention kernels:
  - Kernel 1 (Causal Understanding): Self-attention for understanding/textual
    tokens using sample-specific lower-triangular causal masking with strict
    offset fences to prohibit cross-sample interference.
  - Kernel 2 (Full Generative Attention): Attention for generation queries
    attending to all preceding context (textual + visual) within the same
    sample (full cross-attention to prompts and bidirectional self-attention
    for generated content) with cross-sample fences.
- Linear projections: Independent Q, K, V, O projections for understanding and
  generation pathways with configurable QK normalization and 3D M-RoPE.
- Re-interleaving: Sliced outputs are merged back into the global sequence
order,
  yielding a transformed packed attention tensor of shape (N_total, D_model).

    ====================================================================================================
                                      COSMOS 3 DUAL-ATTENTION OPERATOR ARCHITECTURE
    ====================================================================================================

    Dimension Legend:
      T : Total sequence length (N_total = N_und + N_gen)
      U : Number of Understanding / Text prompt tokens (N_und)
      G : Number of Generation / Video tokens (N_gen)
      M : Model hidden dimension (D_model)
      H : Number of Query attention heads (num_heads)
      K : Number of Key / Value heads (num_kv_heads, where H % K == 0)
      D : Head dimension (head_dim)

    ----------------------------------------------------------------------------------------------------
    1. Stream Unpacking, Linear Projections, Normalization & 3D M-RoPE:
    ----------------------------------------------------------------------------------------------------
      [INPUTS to Block (1)]:
        • Tokens  : [T, M]   (Interleaved packed token sequence)
        • Cos/Sin : [T, D]   (Global 3D M-RoPE position embedding tables)
        • Indices : [U], [G] (packed_und_token_indexes, packed_gen_token_indexes)

                                              Tokens [T, M]
                                                    |
                                    unpack_streams(indices_und, indices_gen)
                                         /                     \
                             und_tokens [U, M]              gen_tokens [G, M]
                                    |                              |
                             W_q, W_k, W_v                  W_q_gen, W_k_gen, W_v_gen
                          [M, H*D] / [M, K*D]             [M, H*D] / [M, K*D]
                                    |                              |
                             +------+------+                +------+------+
                             |      |      |                |      |      |
                           Q_und  K_und  V_und            Q_gen  K_gen  V_gen
                          [U,H,D] [U,K,D] [U,K,D]        [G,H,D] [G,K,D] [G,K,D]
                             |      |      |                |      |      |
                          RMSNorm RMSNorm  |             RMSNorm RMSNorm  |
                             |      |      | (bypasses      |      |      | (bypasses
                             v      v      |  RoPE)         v      v      |  RoPE)
                         ┌──────────────┐  |            ┌──────────────┐  |
      cos/sin[und_idx] ──┤  3D M-RoPE   │  |  cos/sin ──┤  3D M-RoPE   │  |
         [U, D]          │ (UND Stream) │  | [gen_idx]  │ (GEN Stream) │  |
                         └──────┬───────┘  |   [G, D]   └──────┬───────┘  |
                                │          |                   │          |
                                v          |                   v          |
                        (Rotated Q & K)    |           (Rotated Q & K)    |
                        Q_und     K_und    |           Q_gen     K_gen    |
                       [U,H,D]   [U,K,D]   |          [G,H,D]   [G,K,D]   |
                          |         |      |             |         |      |
      [OUTPUTS of (1)]:   |         |      |             |         |      |
                          |         |      |             |   +-----+------+
                          |         |      |             |   |     |
                          |         +------+------+      |   |     |
                          |                |      |      |   |     |
                          |                |      v      |   v     v
                          |                |   ┌──────────────────────────────┐
                          |                |   │  BOUNDARY: SCATTER/ASSEMBLE  │
                          |                |   │  k_all[und_idx] = k_und      │
                          |                |   │  k_all[gen_idx] = k_gen      │
                          |                |   │  v_all[und_idx] = v_und      │
                          |                |   │  v_all[gen_idx] = v_gen      │
                          |                |   └──────────────┬───────────────┘
                          |                |                  |
                          |                |             K_all, V_all
                          |                |              [T, K, D]
                          |                |                  |
                          v                v                  v
    ----------------------------------------------------------------------------------------------------
    2. Dual-Kernel Attention Computation (Einsum & Masking):
    ----------------------------------------------------------------------------------------------------
      [INPUTS to Block (2)]:
        • Kernel 1 (Causal Understanding) receives 3 tensors:
            1. Q_und : [U, H, D]
            2. K_und : [U, K, D] (repeated to [U, H, D] if H != K)
            3. V_und : [U, K, D] (repeated to [U, H, D] if H != K)
        • Kernel 2 (Full Generative) receives 3 tensors:
            1. Q_gen : [G, H, D]
            2. K_all : [T, K, D] (repeated to [T, H, D] if H != K)
            3. V_all : [T, K, D] (repeated to [T, H, D] if H != K)

        [ Kernel 1: Causal Understanding ]               [ Kernel 2: Full Generative ]
          In: Q_und [U,H,D], K_und [U,H,D]                 In: Q_gen [G,H,D], K_all [T,H,D]
                         |                                                |
               einsum('uhd,vhd->huv')                           einsum('ghd,thd->hgt')
                         |                                                |
               Scores_und [H, U, U]                             Scores_gen [H, G, T]
                         |                                                |
               + Sample Causal Mask                             + Full Context Mask
                 where sample(u) == sample(v)                     where sample(g) == sample(t)
                 and u >= v (strict causal)                       (attend to all prompt & video)
                         |                                                |
               Softmax along key dim                            Softmax along key dim
                         |                                                |
               Attn_und [H, U, U]                               Attn_gen [H, G, T]
                         |                                                |
               einsum('huv,vhd->uhd')                           einsum('hgt,thd->ghd')
                with In: V_und [U, H, D]                         with In: V_all [T, H, D]
                         |                                                |
               Context_und [U, H, D]                            Context_gen [G, H, D]
                         |                                                |
               einsum('uhd,hdm->um')                            einsum('ghd,hdm->gm')
                via W_o [H*D, M]                                 via W_o_gen [H*D, M]
                         |                                                |
      [OUTPUTS of (2)]:  v                                                v
                  und_out [U, M]                                   gen_out [G, M]
                          \                                               /
                           \                                             /
                       reinterleave_streams(indices_und, indices_gen)
                                                 |
      [FINAL OUTPUT]:                            v
                                           Output [T, M]

    ----------------------------------------------------------------------------------------------------
    3. Mathematical Operator Formulations:
    ----------------------------------------------------------------------------------------------------

    A. Linear Projections & RMSNorm:
       Q_und = RMSNorm( einsum('um, mhd -> uhd', tokens_und, W_q) )
       K_und = RMSNorm( einsum('um, mkd -> ukd', tokens_und, W_k) )
       V_und = einsum('um, mkd -> ukd', tokens_und, W_v)       # Unrotated (bypasses RoPE)

       Q_gen = RMSNorm( einsum('gm, mhd -> ghd', tokens_gen, W_q_gen) )
       K_gen = RMSNorm( einsum('gm, mkd -> gkd', tokens_gen, W_k_gen) )
       V_gen = einsum('gm, mkd -> gkd', tokens_gen, W_v_gen)   # Unrotated (bypasses RoPE)

    B. 3D M-RoPE Rotary Embeddings (Applied to Q and K only):
       Q_und = (Q_und * cos[indices_und]) + (rotate_half(Q_und) * sin[indices_und])
       K_und = (K_und * cos[indices_und]) + (rotate_half(K_und) * sin[indices_und])
       Q_gen = (Q_gen * cos[indices_gen]) + (rotate_half(Q_gen) * sin[indices_gen])
       K_gen = (K_gen * cos[indices_gen]) + (rotate_half(K_gen) * sin[indices_gen])

    C. Boundary Interleaved Context Assembly:
       K_all = zeros([T, K, D]).at[indices_und].set(K_und).at[indices_gen].set(K_gen)
       V_all = zeros([T, K, D]).at[indices_und].set(V_und).at[indices_gen].set(V_gen)

    D. Grouped-Query Head Expansion (if H != K):
       K_und = repeat(K_und, 'u k d -> u (k r) d')    where r = H // K  --> [U, H, D]
       V_und = repeat(V_und, 'u k d -> u (k r) d')    where r = H // K  --> [U, H, D]
       K_all = repeat(K_all, 't k d -> t (k r) d')    where r = H // K  --> [T, H, D]
       V_all = repeat(V_all, 't k d -> t (k r) d')    where r = H // K  --> [T, H, D]

    E. Kernel 1 (Causal Understanding Attention):
       S_und_{h, u, v} = (1 / sqrt(D)) * einsum('uhd, vhd -> huv', Q_und, K_und)
       M_und_{u, v}    = (sample_id(u) == sample_id(v)) & (u >= v)
       A_und_{h, u, v} = softmax_{v}( where(M_und, S_und, -1e9) )
       O_und_{u, m}    = einsum('uhd, hdm -> um', einsum('huv, vhd -> uhd', A_und, V_und), W_o)

    F. Kernel 2 (Full Generative Attention):
       S_gen_{h, g, t} = (1 / sqrt(D)) * einsum('ghd, thd -> hgt', Q_gen, K_all)
       M_gen_{g, t}    = (sample_id(g) == sample_id(t))
       A_gen_{h, g, t} = softmax_{t}( where(M_gen, S_gen, -1e9) )
       O_gen_{g, m}    = einsum('ghd, hdm -> gm', einsum('hgt, thd -> ghd', A_gen, V_all), W_o_gen)

    G. Sequence Re-interleaving:
       Output[indices_und, :] = O_und    --> shape [U, M]
       Output[indices_gen, :] = O_gen    --> shape [G, M]
       Final Tensor: Output [T, M]
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal, overload, Sequence

from flax import nnx
import jax
from jax import sharding
import jax.numpy as jnp
from maxtext.kernels.attention import jax_flash_attention
from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask_info
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import normalizations
import numpy as np


# 1. Packing Metadata and Stream Helpers
@dataclasses.dataclass(frozen=True)
class CosmosPackingMetadata:
  """Boundary metadata and stream indexing for packed multimodal sequences.

  Attributes:
    causal_q_offsets: Cumulative offsets for understanding (UND) queries, shape
      [num_samples + 1]. Defines sample boundaries in the UND stream.
    full_q_offsets: Cumulative offsets for generation (GEN) queries, shape
      [num_samples + 1]. Defines sample boundaries in the GEN stream.
    sample_kv_offsets: Cumulative offsets for total tokens per sample in the
      interleaved sequence, shape [num_samples + 1]. Defines sample boundaries
      for the combined KV context.
    packed_und_token_indexes: Global token indices belonging to the UND stream,
      shape [num_und_tokens].
    packed_gen_token_indexes: Global token indices belonging to the GEN stream,
      shape [num_gen_tokens].
    max_causal_len: Maximum number of UND tokens across all samples.
    max_full_len: Maximum number of GEN tokens across all samples.
    max_sample_len: Maximum number of total tokens across all samples.
    num_samples: Number of batch instances B packed into the sequence.
    total_tokens: Total sequence length N_total = N_und + N_gen.
    num_und_tokens: Total number of UND tokens across all samples.
    num_gen_tokens: Total number of GEN tokens across all samples.
  """

  causal_q_offsets: jax.Array
  full_q_offsets: jax.Array
  sample_kv_offsets: jax.Array
  packed_und_token_indexes: jax.Array
  packed_gen_token_indexes: jax.Array
  max_causal_len: int
  max_full_len: int
  max_sample_len: int
  num_samples: int
  total_tokens: int
  num_und_tokens: int
  num_gen_tokens: int


def build_cosmos_packing_metadata(
    sample_und_lens: Sequence[int],
    sample_gen_lens: Sequence[int],
    packed_und_token_indexes: jax.Array | None = None,
    packed_gen_token_indexes: jax.Array | None = None,
) -> CosmosPackingMetadata:
  """Constructs CosmosPackingMetadata from per-sample token counts.

  If index arrays are not provided, assumes default contiguous layout per
  sample:
  each sample b has its UND tokens followed by its GEN tokens.

  Args:
    sample_und_lens: Number of UND tokens for each sample in the batch.
    sample_gen_lens: Number of GEN tokens for each sample in the batch.
    packed_und_token_indexes: Optional explicit global indices for UND tokens.
    packed_gen_token_indexes: Optional explicit global indices for GEN tokens.

  Returns:
    A fully populated CosmosPackingMetadata dataclass.
  """
  num_samples = len(sample_und_lens)
  if len(sample_gen_lens) != num_samples:
    raise ValueError(
        f"Length of sample_und_lens ({num_samples}) does not match " f"sample_gen_lens ({len(sample_gen_lens)})"
    )

  und_lens = list(sample_und_lens)
  gen_lens = list(sample_gen_lens)
  sample_lens = [u + g for u, g in zip(und_lens, gen_lens)]

  causal_q_offsets_list = [0]
  full_q_offsets_list = [0]
  sample_kv_offsets_list = [0]

  for u, g, s in zip(und_lens, gen_lens, sample_lens):
    causal_q_offsets_list.append(causal_q_offsets_list[-1] + u)
    full_q_offsets_list.append(full_q_offsets_list[-1] + g)
    sample_kv_offsets_list.append(sample_kv_offsets_list[-1] + s)

  total_tokens = sample_kv_offsets_list[-1]
  num_und_tokens = causal_q_offsets_list[-1]
  num_gen_tokens = full_q_offsets_list[-1]

  if packed_und_token_indexes is None or packed_gen_token_indexes is None:
    und_idx_list = []
    gen_idx_list = []
    current_offset = 0
    for u, g in zip(und_lens, gen_lens):
      und_idx_list.extend(range(current_offset, current_offset + u))
      gen_idx_list.extend(range(current_offset + u, current_offset + u + g))
      current_offset += u + g
    packed_und_token_indexes = jnp.array(und_idx_list, dtype=jnp.int32)
    packed_gen_token_indexes = jnp.array(gen_idx_list, dtype=jnp.int32)
  else:
    packed_und_token_indexes = jnp.asarray(packed_und_token_indexes, dtype=jnp.int32)
    packed_gen_token_indexes = jnp.asarray(packed_gen_token_indexes, dtype=jnp.int32)

  max_causal = max(und_lens) if und_lens else 0
  max_full = max(gen_lens) if gen_lens else 0
  max_sample = max(sample_lens) if sample_lens else 0

  return CosmosPackingMetadata(
      causal_q_offsets=jnp.array(causal_q_offsets_list, dtype=jnp.int32),
      full_q_offsets=jnp.array(full_q_offsets_list, dtype=jnp.int32),
      sample_kv_offsets=jnp.array(sample_kv_offsets_list, dtype=jnp.int32),
      packed_und_token_indexes=packed_und_token_indexes,
      packed_gen_token_indexes=packed_gen_token_indexes,
      max_causal_len=max_causal,
      max_full_len=max_full,
      max_sample_len=max_sample,
      num_samples=num_samples,
      total_tokens=total_tokens,
      num_und_tokens=num_und_tokens,
      num_gen_tokens=num_gen_tokens,
  )


def unpack_streams(
    packed_tokens: jax.Array,
    packed_und_token_indexes: jax.Array,
    packed_gen_token_indexes: jax.Array,
    use_ragged_ops: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Unpacks a single 1D packed sequence into understanding and generation streams.

  Args:
    packed_tokens: Tensor of shape (N_total, D_model) containing all packed
      tokens.
    packed_und_token_indexes: 1D array of indices for UND tokens.
    packed_gen_token_indexes: 1D array of indices for GEN tokens.
    use_ragged_ops: Whether to use TPU SparseCore ragged gather kernels if
      available. Default is False (standard JAX indexing).

  Returns:
    und_tokens: Tensor of shape (N_und, D_model).
    gen_tokens: Tensor of shape (N_gen, D_model).
  """
  del use_ragged_ops
  und_tokens = packed_tokens[packed_und_token_indexes]
  gen_tokens = packed_tokens[packed_gen_token_indexes]
  return und_tokens, gen_tokens


def reinterleave_streams(
    und_tokens: jax.Array,
    gen_tokens: jax.Array,
    packed_und_token_indexes: jax.Array,
    packed_gen_token_indexes: jax.Array,
    total_tokens: int,
    use_ragged_ops: bool = False,
) -> jax.Array:
  """Re-interleaves sliced understanding and generation outputs into the global packed order.

  Args:
    und_tokens: Tensor of shape (N_und, D_model) from the understanding stream.
    gen_tokens: Tensor of shape (N_gen, D_model) from the generation stream.
    packed_und_token_indexes: 1D array of indices where UND tokens belong in the
      packed sequence.
    packed_gen_token_indexes: 1D array of indices where GEN tokens belong in the
      packed sequence.
    total_tokens: Total sequence length N_total.
    use_ragged_ops: Whether to use TPU SparseCore ragged scatter kernels if
      available. Default is False (standard JAX dynamic tensor scatter).

  Returns:
    A single merged tensor of shape (N_total, D_model) matching the original
    global layout.
  """
  del use_ragged_ops
  trailing_shape = und_tokens.shape[1:]
  output_tokens = jnp.zeros((total_tokens, *trailing_shape), dtype=und_tokens.dtype)
  if und_tokens.shape[0] > 0:
    output_tokens = output_tokens.at[packed_und_token_indexes].set(und_tokens)
  if gen_tokens.shape[0] > 0:
    output_tokens = output_tokens.at[packed_gen_token_indexes].set(gen_tokens)
  return output_tokens


# 2. Attention Mask Construction
def build_causal_understanding_mask(
    causal_q_offsets: jax.Array,
    num_und_tokens: int,
) -> jax.Array:
  """Builds sample-specific lower-triangular causal attention mask for Kernel 1.

  Cross-sample interference is strictly prohibited via offset fences:
  tokens can only attend to earlier tokens within the same sample.

  Args:
    causal_q_offsets: Array of shape [num_samples + 1] giving cumulative UND
      token counts.
    num_und_tokens: Total number of UND tokens N_und.

  Returns:
    A boolean array of shape [num_und_tokens, num_und_tokens], where True
    indicates
    valid attention and False indicates masked-out positions.
  """
  if num_und_tokens == 0:
    return jnp.zeros((0, 0), dtype=jnp.bool_)

  indices = jnp.arange(num_und_tokens)
  # Assign sample ID to each token: sample b is
  # where causal_q_offsets[b] <= idx < causal_q_offsets[b+1]
  # causal_q_offsets[1:] has shape [num_samples]
  sample_ids = jnp.sum(indices[:, None] >= causal_q_offsets[None, 1:], axis=-1)

  same_sample = sample_ids[:, None] == sample_ids[None, :]
  is_causal = indices[:, None] >= indices[None, :]
  return same_sample & is_causal


def build_full_generative_mask(
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    num_gen_tokens: int,
    total_tokens: int,
) -> jax.Array:
  """Builds bidirectional full generative attention mask for Kernel 2.

  Enables generation queries to attend to all preceding context (both
  textual/UND
  and visual/GEN tokens) within the same sample. Facilitates full
  cross-attention
  to prompts and bidirectional self-attention for generated content, while
  strictly
  prohibiting cross-sample interference.

  Args:
    full_q_offsets: Array of shape [num_samples + 1] giving cumulative GEN query
      counts.
    sample_kv_offsets: Array of shape [num_samples + 1] giving cumulative total
      token counts.
    num_gen_tokens: Total number of GEN query tokens N_gen.
    total_tokens: Total sequence length N_total (combined UND + GEN).

  Returns:
    A boolean array of shape [num_gen_tokens, total_tokens], where True
    indicates
    valid attention and False indicates masked-out positions.
  """
  if num_gen_tokens == 0 or total_tokens == 0:
    return jnp.zeros((num_gen_tokens, total_tokens), dtype=jnp.bool_)

  q_indices = jnp.arange(num_gen_tokens)
  kv_indices = jnp.arange(total_tokens)

  q_sample_ids = jnp.sum(q_indices[:, None] >= full_q_offsets[None, 1:], axis=-1)
  kv_sample_ids = jnp.sum(kv_indices[:, None] >= sample_kv_offsets[None, 1:], axis=-1)

  return q_sample_ids[:, None] == kv_sample_ids[None, :]


def build_causal_understanding_splash_mask(
    causal_q_offsets: jax.Array,
    num_und_tokens: int,
):
  """Builds a Pallas Splash Attention Mask for Kernel 1 (Causal Understanding).

  Converts Cosmos 3 sample-fence causal mask into a Splash Attention Mask
  representation for memory-efficient FlashAttention on TPUs at scale
  (32K-128K).

  Args:
    causal_q_offsets: Cumulative UND token counts per sample, shape [num_samples
      + 1].
    num_und_tokens: Total number of UND tokens N_und.

  Returns:
    A splash_attention_mask.Mask object wrapping the sample-fenced causal mask.
  """
  dense_mask = build_causal_understanding_mask(causal_q_offsets, num_und_tokens)
  return splash_attention_mask.NumpyMask(np.asarray(dense_mask, dtype=np.bool_))


def build_full_generative_splash_mask(
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    num_gen_tokens: int,
    total_tokens: int,
):
  """Builds a Pallas Splash Attention Mask for Kernel 2 (Full Generative).

  Converts Cosmos 3 sample-fence bidirectional mask into a Splash Attention Mask
  representation for memory-efficient FlashAttention on TPUs at scale
  (32K-128K).

  Args:
    full_q_offsets: Cumulative GEN query counts per sample, shape [num_samples +
      1].
    sample_kv_offsets: Cumulative total token counts per sample, shape
      [num_samples + 1].
    num_gen_tokens: Total number of GEN query tokens N_gen.
    total_tokens: Total sequence length N_total (combined UND + GEN).

  Returns:
    A splash_attention_mask.Mask object wrapping the sample-fenced generative
    mask.
  """
  dense_mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, num_gen_tokens, total_tokens)
  return splash_attention_mask.NumpyMask(np.asarray(dense_mask, dtype=np.bool_))


def compile_cosmos_splash_mask(
    mask_array: jax.Array | np.ndarray,
    block_q: int = 128,
    block_kv: int = 128,
):
  """Compiles a dense boolean mask into Pallas Splash Attention MaskInfo.

  Decomposes the attention mask into active blocks, partial blocks, and prefetch
  indices for execution on TPU matrix multiply units with Pallas.

  Args:
    mask_array: 2D boolean array of shape [q_len, kv_len].
    block_q: Query block tile size (default 128 for TPU).
    block_kv: Key/Value block tile size (default 128 for TPU).

  Returns:
    splash_attention_mask_info.MaskInfo with active block coordinates and
    runtime structures.
  """
  mask_np = np.asarray(mask_array, dtype=np.bool_)
  q_len, kv_len = mask_np.shape
  if q_len % block_q != 0 or kv_len % block_kv != 0:
    raise ValueError(
        f"Sequence lengths ({q_len}, {kv_len}) must be multiples of block sizes"
        f" ({block_q}, {block_kv}) for Splash Attention compilation."
    )
  return splash_attention_mask_info.process_dynamic_mask(mask_np, (block_q, block_kv))


# 3. Rotary Position Embedding Helpers (3D M-RoPE)
def rotate_half(x: jax.Array) -> jax.Array:
  """Rotates half the hidden dimensions of the input tensor.

  Transforms [x1, x2] into [-x2, x1] along the last axis.

  Args:
    x: Input activation tensor whose last dimension will be rotated.

  Returns:
    Tensor of the same shape and dtype with the last dimension rotated.
  """
  half_dim = x.shape[-1] // 2
  x1 = x[..., :half_dim]
  x2 = x[..., half_dim:]
  return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    unsqueeze_dim: int | None = 1,
) -> jax.Array:
  """Applies Rotary Position Embedding (RoPE / 3D M-RoPE) to an activation tensor.

  Args:
    x: Input tensor of shape [N, num_heads, head_dim] or similar.
    cos: Cosine frequencies of shape [N, head_dim].
    sin: Sine frequencies of shape [N, head_dim].
    unsqueeze_dim: Dimension along which to expand cos/sin for broadcasting
      across heads. Default is 1 (e.g. [N, 1, head_dim]).

  Returns:
    Rotated tensor of the same shape and dtype as x.
  """
  if unsqueeze_dim is not None:
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)

  x_f32 = x.astype(jnp.float32)
  cos_f32 = cos.astype(jnp.float32)
  sin_f32 = sin.astype(jnp.float32)

  rotated = (x_f32 * cos_f32) + (rotate_half(x_f32) * sin_f32)
  return rotated.astype(x.dtype)


def compute_3d_mrope_cos_sin(
    position_ids_3d: jax.Array,
    head_dim: int,
    mrope_section: tuple[int, int, int] = (24, 20, 20),
    rope_theta: float = 1000000.0,
) -> tuple[jax.Array, jax.Array]:
  """Computes interleaved 3D M-RoPE cosine and sine tensors from 3D coordinates.

  Args:
    position_ids_3d: Global 3D coordinates (temporal, height, width) of shape
      [N, 3].
    head_dim: Attention head dimension D_head.
    mrope_section: Frequencies allocation for (temporal, height, width)
      dimensions. Sum of section dimensions must equal head_dim // 2.
    rope_theta: Base frequency timescale.

  Returns:
    cos: Array of shape [N, head_dim].
    sin: Array of shape [N, head_dim].
  """
  if sum(mrope_section) != head_dim // 2:
    raise ValueError(f"mrope_section {mrope_section} sum must equal head_dim // 2" f" ({head_dim // 2})")

  freq_dim = head_dim // 2
  inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, freq_dim, 1, dtype=jnp.float32) * 2 / head_dim))

  # Start with temporal positions (dim 0)
  pos_temporal = position_ids_3d[:, 0, None].astype(jnp.float32)
  freqs = pos_temporal * inv_freq[None, :]  # [N, freq_dim]

  # Interleaved replacement for spatial dimensions:
  # offset 1 with stride 3 for height (dim 1)
  # offset 2 with stride 3 for width (dim 2)
  pos_height = position_ids_3d[:, 1, None].astype(jnp.float32)
  pos_width = position_ids_3d[:, 2, None].astype(jnp.float32)

  h_limit = mrope_section[1] * 3
  w_limit = mrope_section[2] * 3

  h_indices = jnp.arange(1, min(h_limit, freq_dim), 3)
  w_indices = jnp.arange(2, min(w_limit, freq_dim), 3)

  freqs = freqs.at[:, h_indices].set(pos_height * inv_freq[h_indices])
  freqs = freqs.at[:, w_indices].set(pos_width * inv_freq[w_indices])

  # Concatenate frequencies to cover full head_dim
  emb = jnp.concatenate([freqs, freqs], axis=-1)  # [N, head_dim]
  return jnp.cos(emb), jnp.sin(emb)


# 4. Dual-Attention Functional Kernels
def causal_understanding_attention(
    q_und: jax.Array,
    k_und: jax.Array,
    v_und: jax.Array,
    causal_q_offsets: jax.Array,
    scale: float | None = None,
    mask_value: float = -1e9,
    attention_kernel: str = "dot_product",
) -> jax.Array:
  """Kernel 1: Causal Understanding Attention with cross-sample fences.

  Args:
    q_und: Query tensor of shape [N_und, num_heads, head_dim].
    k_und: Key tensor of shape [N_und, num_kv_heads, head_dim].
    v_und: Value tensor of shape [N_und, num_kv_heads, head_dim].
    causal_q_offsets: Sample boundaries in UND stream, shape [num_samples + 1].
    scale: Softmax scaling factor. Defaults to 1 / sqrt(head_dim).
    mask_value: Additive logit mask value for invalid positions.
    attention_kernel: Backend kernel ('dot_product', 'flash', 'splash').

  Returns:
    Context tensor of shape [N_und, num_heads, head_dim].
  """
  n_und, num_heads, head_dim = q_und.shape
  if n_und == 0:
    return jnp.zeros_like(q_und)

  num_kv_heads = k_und.shape[1]
  if num_heads != num_kv_heads:
    # Expand KV heads to match query heads (GQA)
    repeat_factor = num_heads // num_kv_heads
    k_und = jnp.repeat(k_und, repeat_factor, axis=1)
    v_und = jnp.repeat(v_und, repeat_factor, axis=1)

  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  if attention_kernel == "flash":
    block_q = 128
    block_kv = 128
    if n_und % block_q != 0:
      raise ValueError(
          "FlashAttention requires sequence length divisible by block_q"
          f" ({block_q}), but got N_und={n_und}. Use"
          " attention_kernel='dot_product' for unaligned lengths."
      )
    mask = build_causal_understanding_mask(causal_q_offsets, n_und)
    num_heads = q_und.shape[1]
    num_kv_heads = k_und.shape[1]
    if num_heads != num_kv_heads:
      repeat_factor = num_heads // num_kv_heads
      k_und = jnp.repeat(k_und, repeat_factor, axis=1)
      v_und = jnp.repeat(v_und, repeat_factor, axis=1)
    q_scaled = q_und * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)[jnp.newaxis, ...]  # [1, num_heads, n_und, head_dim]
    k_reshaped = k_und.transpose(1, 0, 2)[jnp.newaxis, ...]  # [1, num_heads, n_und, head_dim]
    v_reshaped = v_und.transpose(1, 0, 2)[jnp.newaxis, ...]  # [1, num_heads, n_und, head_dim]
    flash_out = jax_flash_attention.flash_attention_block_masked(
        q=q_reshaped,
        k=k_reshaped,
        v=v_reshaped,
        segment_ids=None,
        block_q=block_q,
        block_kv=block_kv,
        mask=mask,
        mask_value=mask_value,
    )
    # flash_out: [1, num_heads, n_und, head_dim] -> [n_und, num_heads, head_dim]
    out = jnp.squeeze(flash_out, axis=0).transpose(1, 0, 2)
    return out.astype(q_und.dtype)

  elif attention_kernel == "splash":
    block_q = 128
    block_kv = 128
    if n_und % block_q != 0:
      raise ValueError(
          "Splash Attention requires sequence length divisible by block size"
          f" ({block_q}), but got N_und={n_und}. Use"
          " attention_kernel='dot_product' for unaligned lengths."
      )
    splash_mask = build_causal_understanding_splash_mask(causal_q_offsets, n_und)
    sa_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        block_q=block_q,
        block_kv=block_kv,
    )
    splash_kernel_fn = splash_attention_kernel.make_splash_mha_single_device(
        mask=splash_mask,
        config=sa_config,
        downcast_smem_data=False,
        partial_mask_blocks_dtype=np.int32,
    )
    q_scaled = q_und * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)  # [num_heads, n_und, head_dim]
    k_reshaped = k_und.transpose(1, 0, 2)  # [num_heads, n_und, head_dim]
    v_reshaped = v_und.transpose(1, 0, 2)  # [num_heads, n_und, head_dim]
    splash_out = splash_kernel_fn(q_reshaped, k_reshaped, v_reshaped)  # [num_heads, n_und, head_dim]
    out = splash_out.transpose(1, 0, 2)  # [n_und, num_heads, head_dim]
    return out.astype(q_und.dtype)

  elif attention_kernel == "dot_product":
    # Compute attention scores: [num_heads, N_und, N_und]
    scores = jnp.einsum("qhd,khd->hqk", q_und, k_und) * scale

    # Apply sample-specific causal mask
    mask = build_causal_understanding_mask(causal_q_offsets, n_und)
    scores = jnp.where(mask[None, :, :], scores, mask_value)

    # Softmax along key axis in float32 for numerical stability
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q_und.dtype)

    # Compute weighted values: [N_und, num_heads, head_dim]
    out = jnp.einsum("hqk,khd->qhd", attn_weights, v_und)
    return out
  else:
    raise ValueError(
        f"Unsupported attention_kernel '{attention_kernel}'. " "Expected one of ['dot_product', 'flash', 'splash']."
    )


def full_generative_attention(
    q_gen: jax.Array,
    k_all: jax.Array,
    v_all: jax.Array,
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    scale: float | None = None,
    mask_value: float = -1e9,
    attention_kernel: str = "dot_product",
) -> jax.Array:
  """Kernel 2: Full Generative Attention with cross-sample fences.

  Enables generation queries to attend to all preceding context (textual +
  visual)
  within the same sample. Facilitates full cross-attention to prompts and
  bidirectional self-attention for generated content.

  Supports multiple backend implementations:
  - 'dot_product': Standard scaled dot-product attention with exact masking.
  - 'flash': MaxText JAX FlashAttention without materializing O(S^2) attention
  matrix.
  - 'splash': Pallas Splash Attention for scaling to 32K-128K on TPUs.

  Args:
    q_gen: Query tensor of shape [N_gen, num_heads, head_dim].
    k_all: All keys in the packed sequence, shape [N_total, num_kv_heads,
      head_dim].
    v_all: All values in the packed sequence, shape [N_total, num_kv_heads,
      head_dim].
    full_q_offsets: Sample boundaries in GEN query stream, shape [num_samples +
      1].
    sample_kv_offsets: Sample boundaries in combined KV stream, shape
      [num_samples + 1].
    scale: Softmax scaling factor. Defaults to 1 / sqrt(head_dim).
    mask_value: Additive logit mask value for invalid positions.
    attention_kernel: Backend kernel ('dot_product', 'flash', 'splash').

  Returns:
    Context tensor of shape [N_gen, num_heads, head_dim].
  """
  n_gen, num_heads, head_dim = q_gen.shape
  total_tokens = k_all.shape[0]
  if n_gen == 0 or total_tokens == 0:
    return jnp.zeros_like(q_gen)

  num_kv_heads = k_all.shape[1]
  if num_heads != num_kv_heads:
    # Expand KV heads to match query heads (GQA)
    repeat_factor = num_heads // num_kv_heads
    k_all = jnp.repeat(k_all, repeat_factor, axis=1)
    v_all = jnp.repeat(v_all, repeat_factor, axis=1)

  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  if attention_kernel == "flash":
    block_q = 128
    block_kv = 128
    if n_gen % block_q != 0 or total_tokens % block_kv != 0:
      raise ValueError(
          "FlashAttention requires sequence lengths divisible by block sizes"
          f" (block_q={block_q}, block_kv={block_kv}), but got N_gen={n_gen},"
          f" N_total={total_tokens}. Use attention_kernel='dot_product' for"
          " unaligned lengths."
      )
    mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    # k_all and v_all are already repeated to num_heads above
    q_scaled = q_gen * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)[jnp.newaxis, ...]  # [1, num_heads, n_gen, head_dim]
    k_reshaped = k_all.transpose(1, 0, 2)[jnp.newaxis, ...]  # [1, num_heads, total_tokens, head_dim]
    v_reshaped = v_all.transpose(1, 0, 2)[jnp.newaxis, ...]  # [1, num_heads, total_tokens, head_dim]
    flash_out = jax_flash_attention.flash_attention_block_masked(
        q=q_reshaped,
        k=k_reshaped,
        v=v_reshaped,
        segment_ids=None,
        block_q=block_q,
        block_kv=block_kv,
        mask=mask,
        mask_value=mask_value,
    )
    # flash_out: [1, num_heads, n_gen, head_dim] -> [n_gen, num_heads, head_dim]
    out = jnp.squeeze(flash_out, axis=0).transpose(1, 0, 2)
    return out.astype(q_gen.dtype)

  elif attention_kernel == "splash":
    block_q = 128
    block_kv = 128
    if n_gen % block_q != 0 or total_tokens % block_kv != 0:
      raise ValueError(
          "Splash Attention requires sequence lengths divisible by block sizes"
          f" (block_q={block_q}, block_kv={block_kv}), but got N_gen={n_gen},"
          f" N_total={total_tokens}. Use attention_kernel='dot_product' for"
          " unaligned lengths."
      )
    splash_mask = build_full_generative_splash_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    sa_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        block_q=block_q,
        block_kv=block_kv,
    )
    splash_kernel_fn = splash_attention_kernel.make_splash_mha_single_device(
        mask=splash_mask,
        config=sa_config,
        downcast_smem_data=False,
        partial_mask_blocks_dtype=np.int32,
    )
    q_scaled = q_gen * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)  # [num_heads, n_gen, head_dim]
    k_reshaped = k_all.transpose(1, 0, 2)  # [num_heads, total_tokens, head_dim]
    v_reshaped = v_all.transpose(1, 0, 2)  # [num_heads, total_tokens, head_dim]
    splash_out = splash_kernel_fn(q_reshaped, k_reshaped, v_reshaped)  # [num_heads, n_gen, head_dim]
    out = splash_out.transpose(1, 0, 2)  # [n_gen, num_heads, head_dim]
    return out.astype(q_gen.dtype)

  elif attention_kernel == "dot_product":
    # Compute attention scores: [num_heads, N_gen, N_total]
    scores = jnp.einsum("qhd,khd->hqk", q_gen, k_all) * scale

    # Apply full generative sample-fence mask
    mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    scores = jnp.where(mask[None, :, :], scores, mask_value)

    # Softmax along key axis in float32 for stability
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q_gen.dtype)

    # Compute weighted values: [N_gen, num_heads, head_dim]
    out = jnp.einsum("hqk,khd->qhd", attn_weights, v_all)
    return out
  else:
    raise ValueError(
        f"Unsupported attention_kernel '{attention_kernel}'. " "Expected one of ['dot_product', 'flash', 'splash']."
    )


# 5. Cosmos Dual-Attention Module
class CosmosDualAttention(nnx.Module):
  """Dual-pathway packed attention for Cosmos 3 architectures.

  Implements understanding and generation pathways with independent linear
  projections (Q, K, V, O), configurable QK normalization, 3D M-RoPE, and
  sample-isolated dual attention kernels.
  """

  def __init__(
      self,
      dim: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      *,
      qk_norm_for_text: bool = True,
      qk_norm_for_diffusion: bool = True,
      use_und_k_norm_for_gen: bool = False,
      use_bias: bool = False,
      attention_kernel: str = "dot_product",
      use_ragged_ops: bool = False,
      mesh: sharding.Mesh | None = None,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_init: initializers.NdInitializer | None = None,
      rngs: nnx.Rngs,
  ):
    """Initializes CosmosDualAttention module.

    Args:
      dim: Model dimension D_model.
      num_heads: Number of attention query heads.
      num_kv_heads: Number of key/value heads.
      head_dim: Dimension per attention head D_head.
      qk_norm_for_text: Whether to apply RMSNorm to Q and K for understanding
        stream.
      qk_norm_for_diffusion: Whether to apply RMSNorm to Q and K for generation
        stream.
      use_und_k_norm_for_gen: Whether to normalize UND keys specifically when
        seen by generation queries in Kernel 2 (needed when
        qk_norm_for_diffusion=True and qk_norm_for_text=False).
      use_bias: Whether linear projections include additive bias.
      attention_kernel: Backend kernel ('dot_product', 'flash', 'splash').
      use_ragged_ops: Whether to use TPU SparseCore ragged gather/scatter
        kernels if available.
      mesh: JAX device mesh for sharding.
      dtype: Computation data type.
      weight_dtype: Weight parameter data type.
      kernel_init: Kernel initializer factory.
      rngs: Flax NNX random number generators.
    """
    if attention_kernel not in ("dot_product", "flash", "splash"):
      raise ValueError(
          f"Unsupported attention_kernel '{attention_kernel}'. "
          "Supported kernels are ['dot_product', 'flash', 'splash']."
      )
    self.attention_kernel = attention_kernel
    self.use_ragged_ops = use_ragged_ops
    self.dim = dim
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.qk_norm_for_text = qk_norm_for_text
    self.qk_norm_for_diffusion = qk_norm_for_diffusion
    self.use_und_k_norm_for_gen = use_und_k_norm_for_gen
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.scaling = 1.0 / math.sqrt(head_dim)

    if kernel_init is None:
      kernel_init = initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")

    # Understanding Pathway Projections
    self.q_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=num_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        mesh=mesh,
        rngs=rngs,
    )
    self.k_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=num_kv_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        mesh=mesh,
        rngs=rngs,
    )
    self.v_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=num_kv_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        mesh=mesh,
        rngs=rngs,
    )
    self.o_proj = linears.DenseGeneral(
        in_features_shape=num_heads * head_dim,
        out_features_shape=dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        mesh=mesh,
        rngs=rngs,
    )

    # Understanding QK Normalization
    if qk_norm_for_text:
      self.q_norm = normalizations.RMSNorm(head_dim, dtype=dtype, weight_dtype=weight_dtype, rngs=rngs)
      self.k_norm = normalizations.RMSNorm(head_dim, dtype=dtype, weight_dtype=weight_dtype, rngs=rngs)
    else:
      self.q_norm = None
      self.k_norm = None

    # Generation Pathway Projections
    self.q_proj_gen = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=num_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        mesh=mesh,
        rngs=rngs,
    )
    self.k_proj_gen = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=num_kv_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        mesh=mesh,
        rngs=rngs,
    )
    self.v_proj_gen = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=num_kv_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        mesh=mesh,
        rngs=rngs,
    )
    self.o_proj_gen = linears.DenseGeneral(
        in_features_shape=num_heads * head_dim,
        out_features_shape=dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        mesh=mesh,
        rngs=rngs,
    )

    # Generation QK Normalization
    if qk_norm_for_diffusion:
      self.q_norm_gen = normalizations.RMSNorm(head_dim, dtype=dtype, weight_dtype=weight_dtype, rngs=rngs)
      self.k_norm_gen = normalizations.RMSNorm(head_dim, dtype=dtype, weight_dtype=weight_dtype, rngs=rngs)
    else:
      self.q_norm_gen = None
      self.k_norm_gen = None

    # Cross-Attention UND K Normalization
    if use_und_k_norm_for_gen and qk_norm_for_diffusion and not qk_norm_for_text:
      self.k_norm_und_for_gen = normalizations.RMSNorm(head_dim, dtype=dtype, weight_dtype=weight_dtype, rngs=rngs)
    else:
      self.k_norm_und_for_gen = None

  @overload
  def __call__(
      self,
      tokens: jax.Array,
      metadata: CosmosPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: Literal[True] = True,
  ) -> jax.Array:
    ...

  @overload
  def __call__(
      self,
      tokens: tuple[jax.Array, jax.Array],
      metadata: CosmosPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    ...

  @overload
  def __call__(
      self,
      tokens: jax.Array,
      metadata: CosmosPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: Literal[False],
  ) -> tuple[jax.Array, jax.Array]:
    ...

  def __call__(
      self,
      tokens: jax.Array | tuple[jax.Array, jax.Array],
      metadata: CosmosPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: bool = True,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Executes dual-attention on understanding and generation pathways.

    Args:
      tokens: Either a packed 2D tensor of shape (N_total, D_model) or a tuple
        of (und_tokens, gen_tokens) where und_tokens is of shape (N_und,
        D_model) and gen_tokens is of shape (N_gen, D_model).
      metadata: Boundary metadata and stream indices.
      cos: Optional 3D M-RoPE cosine frequencies of shape (N_total, head_dim).
      sin: Optional 3D M-RoPE sine frequencies of shape (N_total, head_dim).
      reinterleave: If True (default) and tokens was a packed 2D tensor,
        re-interleaves the attention outputs back into the global sequence
        order, returning a packed tensor of shape (N_total, D_model). If False
        or if tokens was passed as a tuple, returns (und_out, gen_out).

    Returns:
      attention_out: Transformed packed tensor of shape (N_total, D_model) if
        reinterleave is True and tokens is a packed 2D tensor; otherwise returns
        the tuple (und_out, gen_out).
    """
    if isinstance(tokens, tuple):
      und_tokens, gen_tokens = tokens
      should_reinterleave = False
    else:
      und_tokens, gen_tokens = unpack_streams(
          tokens,
          metadata.packed_und_token_indexes,
          metadata.packed_gen_token_indexes,
          use_ragged_ops=self.use_ragged_ops,
      )
      should_reinterleave = reinterleave

    n_und = und_tokens.shape[0]
    n_gen = gen_tokens.shape[0]
    total_tokens = metadata.total_tokens

    # 1. Linear Projections
    if n_und > 0:
      q_und = self.q_proj(und_tokens).reshape(n_und, self.num_heads, self.head_dim)
      k_und = self.k_proj(und_tokens).reshape(n_und, self.num_kv_heads, self.head_dim)
      v_und = self.v_proj(und_tokens).reshape(n_und, self.num_kv_heads, self.head_dim)
    else:
      q_und = jnp.zeros((0, self.num_heads, self.head_dim), dtype=self.dtype)
      k_und = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)
      v_und = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)

    if n_gen > 0:
      q_gen = self.q_proj_gen(gen_tokens).reshape(n_gen, self.num_heads, self.head_dim)
      k_gen = self.k_proj_gen(gen_tokens).reshape(n_gen, self.num_kv_heads, self.head_dim)
      v_gen = self.v_proj_gen(gen_tokens).reshape(n_gen, self.num_kv_heads, self.head_dim)
    else:
      q_gen = jnp.zeros((0, self.num_heads, self.head_dim), dtype=self.dtype)
      k_gen = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)
      v_gen = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)

    # 2. QK Normalization
    if self.q_norm is not None and n_und > 0:
      q_und = self.q_norm(q_und)
    if self.k_norm is not None and n_und > 0:
      k_und = self.k_norm(k_und)

    if self.q_norm_gen is not None and n_gen > 0:
      q_gen = self.q_norm_gen(q_gen)
    if self.k_norm_gen is not None and n_gen > 0:
      k_gen = self.k_norm_gen(k_gen)

    # 3. Apply 3D M-RoPE
    k_und_pre_rope = k_und
    if cos is not None and sin is not None:
      if n_und > 0:
        cos_und = cos[metadata.packed_und_token_indexes]
        sin_und = sin[metadata.packed_und_token_indexes]
        q_und = apply_rotary_pos_emb(q_und, cos_und, sin_und, unsqueeze_dim=1)
        k_und = apply_rotary_pos_emb(k_und, cos_und, sin_und, unsqueeze_dim=1)
      if n_gen > 0:
        cos_gen = cos[metadata.packed_gen_token_indexes]
        sin_gen = sin[metadata.packed_gen_token_indexes]
        q_gen = apply_rotary_pos_emb(q_gen, cos_gen, sin_gen, unsqueeze_dim=1)
        k_gen = apply_rotary_pos_emb(k_gen, cos_gen, sin_gen, unsqueeze_dim=1)

    # 4. Cross-Attention UND K Normalization (if configured)
    if self.k_norm_und_for_gen is not None and n_und > 0:
      # Normalize raw und K specifically for the gen pathway cross-attention
      # (pre-RoPE).
      k_und_norm_for_gen = self.k_norm_und_for_gen(k_und_pre_rope)
      if cos is not None and sin is not None:
        k_und_for_gen = apply_rotary_pos_emb(k_und_norm_for_gen, cos_und, sin_und, unsqueeze_dim=1)
      else:
        k_und_for_gen = k_und_norm_for_gen
    else:
      k_und_for_gen = k_und

    # 5. Assemble Interleaved K_all and V_all for Kernel 2
    k_all = jnp.zeros((total_tokens, self.num_kv_heads, self.head_dim), dtype=self.dtype)
    v_all = jnp.zeros((total_tokens, self.num_kv_heads, self.head_dim), dtype=self.dtype)
    if n_und > 0:
      k_all = k_all.at[metadata.packed_und_token_indexes].set(k_und_for_gen)
      v_all = v_all.at[metadata.packed_und_token_indexes].set(v_und)
    if n_gen > 0:
      k_all = k_all.at[metadata.packed_gen_token_indexes].set(k_gen)
      v_all = v_all.at[metadata.packed_gen_token_indexes].set(v_gen)

    # 6. Kernel 1: Causal Understanding Attention
    if n_und > 0:
      und_attn_context = causal_understanding_attention(
          q_und=q_und,
          k_und=k_und,
          v_und=v_und,
          causal_q_offsets=metadata.causal_q_offsets,
          scale=self.scaling,
          attention_kernel=self.attention_kernel,
      )
      und_attn_flat = und_attn_context.reshape(n_und, self.num_heads * self.head_dim)
      und_out = self.o_proj(und_attn_flat)
    else:
      und_out = jnp.zeros((0, self.dim), dtype=self.dtype)

    # 7. Kernel 2: Full Generative Attention
    if n_gen > 0:
      gen_attn_context = full_generative_attention(
          q_gen=q_gen,
          k_all=k_all,
          v_all=v_all,
          full_q_offsets=metadata.full_q_offsets,
          sample_kv_offsets=metadata.sample_kv_offsets,
          scale=self.scaling,
          attention_kernel=self.attention_kernel,
      )
      gen_attn_flat = gen_attn_context.reshape(n_gen, self.num_heads * self.head_dim)
      gen_out = self.o_proj_gen(gen_attn_flat)
    else:
      gen_out = jnp.zeros((0, self.dim), dtype=self.dtype)

    if should_reinterleave:
      return reinterleave_streams(
          und_tokens=und_out,
          gen_tokens=gen_out,
          packed_und_token_indexes=metadata.packed_und_token_indexes,
          packed_gen_token_indexes=metadata.packed_gen_token_indexes,
          total_tokens=metadata.total_tokens,
          use_ragged_ops=self.use_ragged_ops,
      )

    return und_out, gen_out


# Convenient alias
CosmosAttention = CosmosDualAttention
