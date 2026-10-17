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
    has_t_inv: bool = True,
) -> Tuple[list[pl.BlockSpec], list[pl.BlockSpec], int, int]:
  """Constructs reverse-scan Pallas emit_pipeline in_specs and out_specs for GDN backward.

  With `has_t_inv=False` (t_inv recomputed in-kernel) the t_inv input spec is omitted.
  """
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
  ]
  if has_t_inv:
    in_specs.append(
        # 5: t_inv [B, G, C, num_v_heads, chunk_size, chunk_size]
        pl.BlockSpec(
            (None, None, None, num_v_heads, chunk_size, chunk_size),
            lambda b, g, c: (b, g, rc(c), 0, 0, 0),
        )
    )
  in_specs += [
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


def make_bwd_block_specs_natural(
    num_chunks: int,
    chunk_size: int,
    q_size: int,
    k_size: int,
    tile_q_size: int,
    tile_k_size: int,
    tile_v_size: int,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    padded_num_v_heads: int | None = None,
    has_dht: bool = False,
    has_dh0: bool = False,
    has_t_inv: bool = True,
) -> Tuple[list[pl.BlockSpec], list[pl.BlockSpec], int, int]:
  """Reverse-scan emit_pipeline specs that read and write the operands in their natural HBM layouts.

  Same pipeline as `make_bwd_block_specs`, except that the head group `g` is selected by the index maps instead of
  by XLA relayouts in front of and behind the pallas_call:

    in:  q, k, v windows of qkv_conv [B, S, dim]; b, a [B, G, C, cs, Hp]; do [B, S, Hv, dv];
         chunk_states [B, C, Hv, dk, dv]; t_inv [B, C, Hv, cs, cs] (omitted with has_t_inv=False);
         a_log, dt_bias [B, G, 1, Hp]; reset [B, G, C, 1, 128]; (dht [B, 1, Hv, dk, dv])
    out: dq, dk, dv windows of dy_conv [B, S, dim]; db, da [B, G, C, cs, Hp]; dal, ddt [B, G, C, 1, Hp];
         (dh0 [B, 1, Hv, dk, dv])

  The caller passes the same qkv_conv HBM ref for the three q / k / v inputs and the same dy_conv HBM ref for the
  three dq / dk / dv outputs. Every other block lands in VMEM with the shape `make_bwd_block_specs` gives it
  ([cs, Hv_tile, dv] for do, [Hv_tile, dk, dv] for chunk_states, [Hv_tile, cs, cs] for t_inv, [1, Hv_tile, dk, dv] for
  dht / dh0), so the kernel body only changes for q / k / v and dq / dk / dv. `num_v_heads` is the per-group head tile.
  The k / v channel offsets must be whole blocks: q_size % tile_k_size == 0 and (q_size + k_size) % tile_v_size == 0.
  """
  if padded_num_v_heads is None:
    padded_num_v_heads = ((num_v_heads + 127) // 128) * 128
  if q_size % tile_k_size != 0 or (q_size + k_size) % tile_v_size != 0:
    raise ValueError(
        f"natural-layout GDN backward specs need whole-block channel offsets, got q_size={q_size}, k_size={k_size},"
        f" tile_k_size={tile_k_size}, tile_v_size={tile_v_size}."
    )
  k_block_off = q_size // tile_k_size
  v_block_off = (q_size + k_size) // tile_v_size

  def rc(c):
    return num_chunks - 1 - c

  in_specs = [
      # q, k, v windows of qkv_conv [B, S, dim]
      pl.BlockSpec((None, chunk_size, tile_q_size), lambda b, g, c: (b, rc(c), g)),
      pl.BlockSpec((None, chunk_size, tile_k_size), lambda b, g, c: (b, rc(c), k_block_off + g)),
      pl.BlockSpec((None, chunk_size, tile_v_size), lambda b, g, c: (b, rc(c), v_block_off + g)),
      # b [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # a [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # do [B, S, Hv, v_head_dim], the layout the sequence-CP dh scan also reads in place
      pl.BlockSpec(
          (None, chunk_size, num_v_heads, v_head_dim),
          lambda b, g, c: (b, rc(c), g, 0),
      ),
      # chunk_states [B, C, Hv, kq_head_dim, v_head_dim]
      pl.BlockSpec(
          (None, None, num_v_heads, kq_head_dim, v_head_dim),
          lambda b, g, c: (b, rc(c), g, 0, 0),
      ),
  ]
  if has_t_inv:
    in_specs.append(
        # t_inv [B, C, Hv, chunk_size, chunk_size]
        pl.BlockSpec(
            (None, None, num_v_heads, chunk_size, chunk_size),
            lambda b, g, c: (b, rc(c), g, 0, 0),
        )
    )
  in_specs += [
      # a_log [B, G, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, 0, 0),
      ),
      # dt_bias [B, G, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, 0, 0),
      ),
      # reset [B, G, C, 1, 128]
      pl.BlockSpec(
          (None, None, None, 1, 128),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
  ]
  if has_dht:
    in_specs.append(
        # dht [B, 1, Hv, kq_head_dim, v_head_dim]
        pl.BlockSpec(
            (None, 1, num_v_heads, kq_head_dim, v_head_dim),
            lambda b, g, c: (b, 0, g, 0, 0),
        )
    )

  out_specs = [
      # dq, dk, dv windows of dy_conv [B, S, dim]
      pl.BlockSpec((None, chunk_size, tile_q_size), lambda b, g, c: (b, rc(c), g)),
      pl.BlockSpec((None, chunk_size, tile_k_size), lambda b, g, c: (b, rc(c), k_block_off + g)),
      pl.BlockSpec((None, chunk_size, tile_v_size), lambda b, g, c: (b, rc(c), v_block_off + g)),
      # db [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # da [B, G, C, chunk_size, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, chunk_size, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # dal [B, G, C, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
      # ddt [B, G, C, 1, padded_num_v_heads]
      pl.BlockSpec(
          (None, None, None, 1, padded_num_v_heads),
          lambda b, g, c: (b, g, rc(c), 0, 0),
      ),
  ]
  if has_dh0:
    out_specs.append(
        # dh0 [B, 1, Hv, kq_head_dim, v_head_dim]
        pl.BlockSpec(
            (None, 1, num_v_heads, kq_head_dim, v_head_dim),
            lambda b, g, c: (b, 0, g, 0, 0),
        )
    )
  return in_specs, out_specs, len(in_specs), len(out_specs)
