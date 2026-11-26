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

"""YaRN rotary embeddings, bit-matching MaxText's `YarnRotaryEmbedding`.

Reference: MaxText's `embeddings.YarnRotaryEmbedding`.
The table is a complex64 `exp(i*theta)` array, and the output layout is always
`[real..., imag...]` even though the input layout is interleaved.
"""

import functools
import math

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.deepseek import layers
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import sharding_mode as mode


def _find_correction_dim(num_rotations, dim, base, max_position_embeddings):
  """`YarnRotaryEmbedding._find_correction_dim` -- pure python float64."""
  return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings, truncate):
  """`YarnRotaryEmbedding._find_correction_range`."""
  low = _find_correction_dim(beta_fast, dim, base, max_position_embeddings)
  high = _find_correction_dim(beta_slow, dim, base, max_position_embeddings)
  if truncate:
    low = math.floor(low)
    high = math.ceil(high)
  low = max(low, 0)
  high = min(high, dim - 1)  # dim - 1, not half_dim - 1
  return low, high


def _linear_ramp_factor(min_val, max_val, dim):
  """`YarnRotaryEmbedding._linear_ramp_factor`."""
  if min_val == max_val:
    max_val += 0.001
  linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_val) / (max_val - min_val)
  return jnp.clip(linear_func, 0, 1)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7))
def yarn_freqs_cis(
    embedding_dims,
    rope_theta,
    rope_factor,
    beta_fast,
    beta_slow,
    original_max_position_embeddings,
    max_position_embeddings,
    truncate,
):
  """Complex64 `[max_position_embeddings, embedding_dims // 2]` rotation table."""
  half_dim = embedding_dims // 2
  freqs = 1.0 / (rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / embedding_dims))
  low, high = _find_correction_range(
      beta_fast, beta_slow, embedding_dims, rope_theta, original_max_position_embeddings, truncate
  )
  smooth = 1 - _linear_ramp_factor(low, high, half_dim)
  freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth
  t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
  freqs = jnp.outer(t, freqs)
  return jnp.exp(1j * freqs)


def freqs_cis_for(config: DeepSeekConfig, max_len: int | None = None):
  """Builds the table.  `max_len` trims the (huge) 163840-row default."""
  return yarn_freqs_cis(
      config.qk_rope_head_dim,
      float(config.rope_max_timescale),
      float(config.rope_factor),
      float(config.beta_fast),
      float(config.beta_slow),
      int(config.original_max_position_embeddings),
      int(max_len or config.max_position_embeddings),
      bool(config.rope_truncate),
  )


def apply_yarn(inputs, positions, freqs_cis, config: DeepSeekConfig):
  """`YarnRotaryEmbedding.__call__`.

  Args:
    inputs: `[b, s, n, qk_rope_head_dim]`, bf16.
    positions: `[b, s]` integer positions.
    freqs_cis: complex64 table from `freqs_cis_for`.
    config: model config.

  Returns:
    `[b, s, n, qk_rope_head_dim]` in `config.dtype`.

  The table is replicated but the positions are batch-sharded, and a gather
  with sharded indices has no inferable result type under explicit, so the
  rows are typed like the activation (`mode.take`).  Unlike the embedding
  lookup this needs no remat: the table is not differentiated, so nothing
  inside the gather is needed by the backward -- at most its typed output is.

  The rotation is linear in `inputs`: its VJP keeps the rows it multiplies
  by.  Under lean residuals (`layers.lean_residuals`) they enter the product
  as `[b, s, 1, half]`, which the multiply broadcasts over the heads itself,
  so the VJP keeps them so -- not the `[b, s, n, half]` complex64 broadcast
  (128 MiB a layer at DeepSeek-V3's 128 heads and 4096 tokens a sequence).
  The products are the same either way.
  """
  b, s, n, h = inputs.shape
  half = h // 2
  ids = positions.astype(jnp.int32)
  freqs = mode.take(freqs_cis, ids, out_spec=mode.act_spec(ids.ndim + 1))  # [b, s, half] complex64
  freqs = freqs[:, :, jnp.newaxis, :]

  if config.rope_interleave:
    reshaped = inputs.reshape(b, s, n, half, 2)
    first_half, second_half = reshaped[..., 0], reshaped[..., 1]
  else:
    first_half, second_half = jnp.split(inputs, 2, axis=-1)

  # bf16 against a weak-typed python 1j promotes to complex64.
  inputs_complex = first_half + 1j * second_half
  if not layers.lean():
    freqs = jnp.broadcast_to(freqs, inputs_complex.shape)
  rotated = inputs_complex * freqs
  output = jnp.concatenate([jnp.real(rotated), jnp.imag(rotated)], axis=-1)

  if config.rope_attention_scaling:
    scaling = 1.0 if config.rope_factor <= 1.0 else (0.1 * math.log(config.rope_factor) + 1.0)
    output = output * scaling
  return output.astype(config.dtype)
