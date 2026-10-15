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
    qk_tile_size: int,
    v_tile_size: int,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    k_block_offset: int,
    v_block_offset: int,
    padded_num_v_heads: int | None = None,
    has_dht: bool = False,
    has_dh0: bool = False,
) -> Tuple[list[pl.BlockSpec], list[pl.BlockSpec], int, int]:
  """Constructs reverse-scan Pallas emit_pipeline in_specs and out_specs for GDN backward.

  Head group g of q/k/v and dq/dk/dv is a column block of a token-major array (the k and v blocks of qkv_conv start
  at block k_block_offset / v_block_offset), and do / chunk_states / t_inv are indexed by group, so the kernel reads
  and writes them in place instead of through group-major copies.
  """
  if padded_num_v_heads is None:
    padded_num_v_heads = ((num_v_heads + 127) // 128) * 128

  def rc(c):
    return num_chunks - 1 - c

  in_specs = [
      # 0: q [B, C, chunk_size, dim_size], columns of group g
      pl.BlockSpec(
          (None, None, chunk_size, qk_tile_size),
          lambda b, g, c: (b, rc(c), 0, g),
      ),
      # 1: k [B, C, chunk_size, dim_size], columns of group g
      pl.BlockSpec(
          (None, None, chunk_size, qk_tile_size),
          lambda b, g, c: (b, rc(c), 0, k_block_offset + g),
      ),
      # 2: v [B, C, chunk_size, dim_size or v_size], columns of group g
      pl.BlockSpec(
          (None, None, chunk_size, v_tile_size),
          lambda b, g, c: (b, rc(c), 0, v_block_offset + g),
      ),
      # 3: b [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 4: a [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 5: do [B, C, chunk_size, G, num_v_heads, v_head_dim]
      pl.BlockSpec(
          (None, None, chunk_size, None, num_v_heads, v_head_dim),
          lambda b, g, c: (b, rc(c), 0, g, 0, 0),
      ),
      # 6: chunk_states [B, C, G, num_v_heads, kq_head_dim, v_head_dim]
      pl.BlockSpec(
          (None, None, None, num_v_heads, kq_head_dim, v_head_dim),
          lambda b, g, c: (b, rc(c), g, 0, 0, 0),
      ),
      # 7: t_inv [B, C, G, num_v_heads, chunk_size, chunk_size]
      pl.BlockSpec(
          (None, None, None, num_v_heads, chunk_size, chunk_size),
          lambda b, g, c: (b, rc(c), g, 0, 0, 0),
      ),
      # 8: a_log [B, G, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, 0, 0),
      ),
      # 9: dt_bias [B, G, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, 0, 0),
      ),
      # 10: reset [B, G, C, 1, 128]
      pl.BlockSpec(
          (None, None, None, 1, 128),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
  ]
  if has_dht:
    in_specs.append(
        # 11: dht [B, G, 1, num_v_heads, kq_head_dim, v_head_dim]
        pl.BlockSpec(
            (None, None, 1, num_v_heads, kq_head_dim, v_head_dim),
            lambda b, g, c: (b, g, 0, 0, 0, 0),
        )
    )

  out_specs = [
      # 0: dq [B, C, chunk_size, q_size], columns of group g
      pl.BlockSpec(
          (None, None, chunk_size, qk_tile_size),
          lambda b, g, c: (b, rc(c), 0, g),
      ),
      # 1: dk [B, C, chunk_size, k_size], columns of group g
      pl.BlockSpec(
          (None, None, chunk_size, qk_tile_size),
          lambda b, g, c: (b, rc(c), 0, g),
      ),
      # 2: dv [B, C, chunk_size, v_size], columns of group g
      pl.BlockSpec(
          (None, None, chunk_size, v_tile_size),
          lambda b, g, c: (b, rc(c), 0, g),
      ),
      # 3: db [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 4: da [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 5: dal [B, G, C, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 6: ddt [B, G, C, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
  ]
  if has_dh0:
    out_specs.append(
        # 7: dh0 [B, G, 1, num_v_heads, kq_head_dim, v_head_dim]
        pl.BlockSpec(
            (None, None, 1, num_v_heads, kq_head_dim, v_head_dim),
            lambda b, g, c: (b, g, 0, 0, 0, 0),
        )
    )
  return in_specs, out_specs, len(in_specs), len(out_specs)
