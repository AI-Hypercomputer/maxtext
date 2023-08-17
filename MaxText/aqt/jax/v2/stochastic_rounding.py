# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Efficient stochastic rounding implementation.
"""

import jax
import jax.numpy as jnp


def random_centered_uniform(
    shape: tuple[int, ...], key: jax.random.KeyArray) -> jnp.ndarray:
  """Generates uniform number in [-0.5, 0.5]."""
  nbits = 16

  # Generate random bits.
  from jax._src import prng  # pylint: disable=g-import-not-at-top
  assert not jax.config.jax_enable_custom_prng
  key = prng.random_wrap(key, impl=jax.random.default_prng_impl())
  bits = prng.random_bits(key, bit_width=nbits, shape=shape)
  assert bits.shape == shape, (bits.shape, bits.shape)
  assert bits.dtype == {8: jnp.uint8, 16: jnp.uint16}[nbits], bits.dtype

  # Align bits with the mantissa of f32.
  nmant = jnp.finfo(jnp.float32).nmant
  r_bitpattern = jnp.uint32(bits) << (nmant - nbits)
  r_bitpattern = r_bitpattern | jnp.float32(1).view(jnp.uint32)
  assert r_bitpattern.dtype == jnp.uint32

  # Gen random floats and shift
  rand_floats = jax.lax.bitcast_convert_type(r_bitpattern, jnp.float32)
  shift = 2 ** (-1 - nbits)
  centered = rand_floats - (1.5 - shift)

  return centered

