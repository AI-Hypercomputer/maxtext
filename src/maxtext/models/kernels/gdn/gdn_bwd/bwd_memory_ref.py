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

from typing import Any, Tuple

from jax.experimental import pallas as pl


def make_bwd_block_specs(
    batch_size: int,
    num_chunks: int,
    chunk_size: int,
    dim_size: int,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    padded_num_v_heads: int | None = None,
    g: Any = None,
    kernel_size: int = 4,
    pad_len: int = 8,
) -> Tuple[list[pl.BlockSpec], list[pl.BlockSpec], int, int]:
  """Constructs reverse-scan Pallas emit_pipeline in_specs and out_specs for GDN backward."""
  del batch_size, kernel_size, pad_len, g
  if padded_num_v_heads is None:
    padded_num_v_heads = ((num_v_heads + 127) // 128) * 128

  def rc(c):
    return num_chunks - 1 - c

  in_specs = [
      pl.BlockSpec(
          (None, None, chunk_size, dim_size),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, num_v_heads, v_head_dim),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      pl.BlockSpec(
          (None, None, num_v_heads, kq_head_dim, v_head_dim),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      pl.BlockSpec(
          (None, None, num_v_heads, chunk_size, chunk_size),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      pl.BlockSpec(
          (None, 1, padded_num_v_heads),
          lambda b, c: (b, 0, 0),
      ),
      pl.BlockSpec(
          (None, 1, padded_num_v_heads),
          lambda b, c: (b, 0, 0),
      ),
      pl.BlockSpec(
          (None, None, 1, 128),
          lambda b, c: (b, rc(c), 0, 0),
      ),
  ]
  out_specs = [
      pl.BlockSpec(
          (None, None, chunk_size, dim_size),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
  ]
  return in_specs, out_specs, len(in_specs), len(out_specs)
