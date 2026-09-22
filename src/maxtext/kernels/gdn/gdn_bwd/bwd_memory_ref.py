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
    has_dht: bool = False,
    has_dh0: bool = False,
) -> Tuple[list[pl.BlockSpec], list[pl.BlockSpec], int, int]:
  """Constructs reverse-scan Pallas emit_pipeline in_specs and out_specs for GDN backward."""
  if padded_num_v_heads is None:
    padded_num_v_heads = ((num_v_heads + 127) // 128) * 128

  def rc(c):
    return num_chunks - 1 - c

  in_specs = [
      # 0: qkv_conv [B, G, C, chunk_size, dim_size]
      pl.BlockSpec(
          (None, None, None, chunk_size, dim_size),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 1: b [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 2: a [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 3: do [B, G, C, chunk_size, num_v_heads, v_head_dim]
      pl.BlockSpec(
          (None, None, None, chunk_size, num_v_heads, v_head_dim),
          lambda b, g, c: (b, g, rc(c), 0, 0, 0),
      ),
      # 4: chunk_states [B, G, C, num_v_heads, kq_head_dim, v_head_dim]
      pl.BlockSpec(
          (None, None, None, num_v_heads, kq_head_dim, v_head_dim),
          lambda b, g, c: (b, g, rc(c), 0, 0, 0),
      ),
      # 5: t_inv [B, G, C, num_v_heads, chunk_size, chunk_size]
      pl.BlockSpec(
          (None, None, None, num_v_heads, chunk_size, chunk_size),
          lambda b, g, c: (b, g, rc(c), 0, 0, 0),
      ),
      # 6: a_log [B, G, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, 0, 0),
      ),
      # 7: dt_bias [B, G, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, 0, 0),
      ),
      # 8: reset [B, G, C, 1, 128]
      pl.BlockSpec(
          (None, None, None, 1, 128),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
  ]
  if has_dht:
    in_specs.append(
        # 9: dht [B, G, 1, num_v_heads, kq_head_dim, v_head_dim]
        pl.BlockSpec(
            (None, None, 1, num_v_heads, kq_head_dim, v_head_dim),
            lambda b, g, c: (b, g, 0, 0, 0, 0),
        )
    )

  out_specs = [
      # 0: dy_conv [B, G, C, chunk_size, dim_size]
      pl.BlockSpec(
          (None, None, None, chunk_size, dim_size),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 1: db [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 2: da [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 3: dal [B, G, C, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # 4: ddt [B, G, C, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
  ]
  if has_dh0:
    out_specs.append(
        # 5: dh0 [B, G, 1, num_v_heads, kq_head_dim, v_head_dim]
        pl.BlockSpec(
            (None, None, 1, num_v_heads, kq_head_dim, v_head_dim),
            lambda b, g, c: (b, g, 0, 0, 0, 0),
        )
    )
  return in_specs, out_specs, len(in_specs), len(out_specs)
