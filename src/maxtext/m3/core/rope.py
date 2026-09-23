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

"""Rotary positional embedding (RoPE).

This is one "layer" m3 writes itself, because NNX has no RoPE primitive. It
is a pure function rather than a module: it holds no state, allocates no
parameters, and so has nothing to place on a mesh.

Only the base ("default") variant lives here. Models needing a specialized
variant -- YaRN, LLaMA 3.1 extended context, partial-rotary, mRoPE -- implement
it in their own model directory rather than growing a parameter on this
function.

Convention
----------
This is the half-split ("rotate half") formulation used by Hugging Face and by
the legacy MaxText `RotaryEmbedding`, *not* the interleaved (GPT-J) pairing.
The first and second halves of `head_dim` form the rotation pairs, so dimension
`i` is paired with dimension `i + head_dim // 2`.

Numerics
--------
Angles and their sinusoids are computed in float32 regardless of the input
dtype, then cast down to the input dtype before the rotation itself. This
mirrors legacy MaxText exactly, which keeps bf16 training runs bit-comparable
across the migration. Computing the rotation in float32 and casting the result
instead would be marginally more accurate, but would silently diverge from the
numbers the legacy stack produces.
"""

import jax
import jax.numpy as jnp


def apply_rope(
    x: jax.Array,
    positions: jax.Array,
    *,
    max_timescale: float = 10_000.0,
    min_timescale: float = 1.0,
) -> jax.Array:
  """Applies rotary positional embedding to a projected query or key tensor.

  Args:
    x: Projected queries or keys, shape `(batch, length, heads, head_dim)`.
      `head_dim` must be even. Applied after the q/k projection and before
      attention.
    positions: Integer token positions, shape `(batch, length)`. Supplying
      explicit positions (rather than assuming `arange(length)`) is what makes
      packed sequences work: each packed segment restarts at zero.
    max_timescale: Longest rotation period, commonly called `rope_theta`.
      10_000 is the original RoPE value; long-context models raise it (Qwen3
      uses 1_000_000).
    min_timescale: Shortest rotation period. Effectively always 1.

  Returns:
    An array of the same shape and dtype as `x`, rotated by position.

  Raises:
    ValueError: If `x` is not rank 4, if `head_dim` is odd, or if `positions`
      does not match the leading `(batch, length)` dimensions of `x`.
  """
  if x.ndim != 4:
    raise ValueError(f"apply_rope expects x of shape (batch, length, heads, head_dim); got shape {x.shape}.")

  head_dim = x.shape[-1]
  if head_dim % 2:
    raise ValueError(f"apply_rope requires an even head_dim to form rotation pairs; got {head_dim}.")

  if positions.shape != x.shape[:2]:
    raise ValueError(
        f"apply_rope expects positions of shape (batch, length) = {x.shape[:2]}; got shape {positions.shape}."
    )

  half_dim = head_dim // 2

  # Geometric series of rotation periods, one per dimension pair. The integer
  # `arange` is deliberate: it matches the legacy `RotaryEmbedding.timescale`
  # expression, which is what keeps the two bit-exact.
  fraction = 2 * jnp.arange(half_dim) / head_dim
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction

  # (batch, length) -> (batch, length, 1, half_dim), broadcasting over heads.
  angles = positions[:, :, jnp.newaxis, jnp.newaxis].astype(jnp.float32) / timescale
  sin = jnp.sin(angles).astype(x.dtype)
  cos = jnp.cos(angles).astype(x.dtype)

  # Rotate each (x1[i], x2[i]) pair by its angle, pairing dimension i with
  # dimension i + half_dim. Equivalent to the legacy
  # `x * cos + rotate_half(x) * sin`, without materializing the doubled
  # sin/cos tables.
  x1 = x[..., :half_dim]
  x2 = x[..., half_dim:]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
