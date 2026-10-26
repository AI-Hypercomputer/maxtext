#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""DeepSeek-V4.1 activation quantize -> dequantize round trips (reference `inference/kernel.py`).

The reference quantizes the input of every FP8 / FP4 GEMM to FP8 E4M3 with one power-of-two (UE8M0) scale per 32
values, and round-trips a few activations in place: the window KV (FP8), the compressed KV (FP4, E4M3 scale per
16), the indexer queries / keys (FP4, UE8M0 scale per 32) and the attention output before `wo_a` (FP8). Gradients
pass straight through the rounding.
"""

import jax
import jax.numpy as jnp

FP8_MAX = 448.0
FP4_MAX = 6.0


def enabled(config) -> bool:
  return bool(getattr(config, "ds41_activation_quant", False))


def _straight_through(x, y):
  return x + jax.lax.stop_gradient(y - x)


def _pow2_ceil(v):
  """2 ** ceil(log2(v)) for positive normal float32 v."""
  m, e = jnp.frexp(v)
  e = jnp.where(m > 0.5, e, e - 1)
  return jnp.ldexp(jnp.ones_like(v), e)


def _round_e2m1(x):
  """Round to the FP4 E2M1 grid, round-to-nearest-even (inputs already clipped to +-6)."""
  a = jnp.abs(x)
  q = jnp.zeros_like(a)
  for edge, value, inclusive in (
      (0.25, 0.5, False),
      (0.75, 1.0, True),
      (1.25, 1.5, False),
      (1.75, 2.0, True),
      (2.5, 3.0, False),
      (3.5, 4.0, True),
      (5.0, 6.0, False),
  ):
    q = jnp.where((a >= edge) if inclusive else (a > edge), value, q)
  return jnp.where(x < 0, -q, q)


def fp8_qdq(x, block_size: int = 32):
  """FP8 E4M3 with a power-of-two scale per `block_size` values of the last axis (amax floor 1e-4)."""
  g = x.astype(jnp.float32).reshape(x.shape[:-1] + (-1, block_size))
  amax = jnp.maximum(jnp.max(jnp.abs(g), axis=-1, keepdims=True), 1e-4)
  s = _pow2_ceil(amax * jnp.float32(1.0 / FP8_MAX))
  y = jnp.clip(g / s, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn).astype(jnp.float32) * s
  return _straight_through(x, y.reshape(x.shape).astype(x.dtype))


def fp4_qdq(x, block_size: int = 32, e4m3_scale: bool = False):
  """FP4 E2M1 with a power-of-two scale (indexer) or an E4M3 scale (compressed KV) per `block_size` values."""
  g = x.astype(jnp.float32).reshape(x.shape[:-1] + (-1, block_size))
  amax = jnp.max(jnp.abs(g), axis=-1, keepdims=True)
  if e4m3_scale:
    s = (jnp.maximum(amax, 6 * 2.0**-9) / FP4_MAX).astype(jnp.float8_e4m3fn).astype(jnp.float32)
  else:
    s = _pow2_ceil(jnp.maximum(amax, 6 * 2.0**-126) * jnp.float32(1.0 / FP4_MAX))
  y = _round_e2m1(jnp.clip(g / s, -FP4_MAX, FP4_MAX)) * s
  return _straight_through(x, y.reshape(x.shape).astype(x.dtype))
