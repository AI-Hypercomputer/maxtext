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

"""Memory reference and BlockSpec builders for reverse-scan GDN backward pass."""

from typing import Tuple

from jax.experimental import pallas as pl


def make_bwd_block_specs(
    num_chunks: int,
    chunk_size: int,
    dim_size: int,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    padded_num_v_heads: int | None = None,
) -> Tuple[list[pl.BlockSpec], list[pl.BlockSpec], int, int]:
  """Constructs reverse-scan Pallas emit_pipeline in_specs and out_specs for GDN backward.

  Reverse Time Traversal:
    Chunk c = 0 corresponds to physical chunk (num_chunks - 1) via reverse-chunk
    index mapping rc(c) = num_chunks - 1 - c.

  Memory Layout & Block Specifications:
    [in_specs]
      0: qkv_conv     [B, C, chunk_size, dim_size]              <- Forward conv activations
      1: b            [B, C, chunk_size, padded_num_v_heads]    <- Forward decay gates
      2: a            [B, C, chunk_size, padded_num_v_heads]    <- Forward gating / alpha
      3: do           [B, C, chunk_size, num_v_heads, v_head]   <- Incoming output cotangent
      4: chunk_states [B, C, num_v_heads, kq_head, v_head]      <- Cached forward state S_c
      5: t_inv        [B, C, num_v_heads, chunk_size, chunk_sz] <- Cached triangular inverse T_c^{-1}
      6: a_log        [B, 1, padded_num_v_heads]                <- Log decay parameter
      7: dt_bias      [B, 1, padded_num_v_heads]                <- Timestep bias
      8: reset        [B, C, 1, 128]                            <- Document boundary mask

    [out_specs]
      0: dy_conv      [B, C, chunk_size, dim_size]              -> Output conv cotangent
      1: db           [B, C, chunk_size, padded_num_v_heads]    -> Gradient for decay gate b
      2: da           [B, C, chunk_size, padded_num_v_heads]    -> Gradient for gating a
      3: dal          [B, C, 1, padded_num_v_heads]             -> Chunk gradient for a_log
      4: ddt          [B, C, 1, padded_num_v_heads]             -> Chunk gradient for dt_bias

  Args:
    num_chunks: Total number of sequence chunks.
    chunk_size: Token sequence chunk length.
    dim_size: Total feature dimension of convolved QKV activations.
    num_v_heads: Number of value heads.
    kq_head_dim: Feature dimension per key/query head.
    v_head_dim: Feature dimension per value head.
    padded_num_v_heads: Value head count aligned upward to 128 (hardware tile).

  Returns:
    A tuple of (in_specs, out_specs, num_in, num_out).
  """
  if padded_num_v_heads is None:
    padded_num_v_heads = ((num_v_heads + 127) // 128) * 128

  def rc(c):
    return num_chunks - 1 - c

  in_specs = [
      # 0: qkv_conv [B, C, chunk_size, dim_size]
      pl.BlockSpec(
          (None, None, chunk_size, dim_size),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      # 1: b [B, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      # 2: a [B, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      # 3: do [B, C, chunk_size, num_v_heads, v_head_dim]
      pl.BlockSpec(
          (None, None, chunk_size, num_v_heads, v_head_dim),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      # 4: chunk_states [B, C, num_v_heads, kq_head_dim, v_head_dim]
      pl.BlockSpec(
          (None, None, num_v_heads, kq_head_dim, v_head_dim),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      # 5: t_inv [B, C, num_v_heads, chunk_size, chunk_size]
      pl.BlockSpec(
          (None, None, num_v_heads, chunk_size, chunk_size),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      # 6: a_log [B, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, 1, padded_num_v_heads),
          lambda b, c: (b, 0, 0),
      ),
      # 7: dt_bias [B, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, 1, padded_num_v_heads),
          lambda b, c: (b, 0, 0),
      ),
      # 8: reset [B, C, 1, 128]
      pl.BlockSpec(
          (None, None, 1, 128),
          lambda b, c: (b, rc(c), 0, 0),
      ),
  ]
  out_specs = [
      # 0: dy_conv [B, C, chunk_size, dim_size]
      pl.BlockSpec(
          (None, None, chunk_size, dim_size),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      # 1: db [B, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      # 2: da [B, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      # 3: dal [B, C, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      # 4: ddt [B, C, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
  ]
  return in_specs, out_specs, len(in_specs), len(out_specs)
