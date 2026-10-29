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

"""JAX runtime decoder for MXFP4 (OCP microscaling FP4) weights.

Storage format (identical to the `compressed-tensors` `mxfp4-pack-quantized` release,
see `maxtext.checkpoint_conversion.utils.mxfp4` for the NumPy converter codec):

  * `packed`: uint8, two FP4 E2M1 codes per byte along the quantized axis. The LOW nibble
    is the even element, the HIGH nibble the odd one.
  * `scale`:  uint8 E8M0 exponent per group of 32 consecutive elements along the same
    axis; the multiplier is `2 ** (scale - 127)`.

`dequantize_mxfp4` is the in-graph inverse and is bit-exact with the NumPy codec
(every E2M1 value times a power of two is exactly representable in fp32 and bf16).
It is intended to run on *already gathered* weights (e.g. the top-k routed experts of a
MoE layer) so the dense tensor only ever exists for the slice being multiplied.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Format constants, shared with the NumPy converter codec
# (`maxtext.checkpoint_conversion.utils.mxfp4` imports them from here).
MXFP4_GROUP_SIZE = 32
E8M0_BIAS = 127
# OCP FP4 E2M1 signed values indexed by the 4-bit code (bit 3 is the sign).
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _e8m0_to_float(scale: jax.Array, dtype: jnp.dtype) -> jax.Array:
  """Decodes uint8 E8M0 exponents to `2**(scale-127)` in `dtype` by writing the exponent bits.

  `dtype` is float32 or bfloat16, which share the 8-bit exponent, so the result is exact for
  codes 1..254. Code 0 maps to +0 instead of the denormal 2**-127 (XLA flushes it anyway).
  Code 255 is NaN in the OCP MX spec but decodes to +inf here, as in the NumPy codec
  (`np.exp2(128)`); real weights never use it.
  """
  if dtype == jnp.bfloat16:
    bits = scale.astype(jnp.uint16) << 7
    return jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
  bits = scale.astype(jnp.uint32) << 23
  return jax.lax.bitcast_convert_type(bits, jnp.float32)


def dequantize_mxfp4(
    packed: jax.Array,
    scale: jax.Array,
    axis: int = -1,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
  """Decodes an MXFP4 (`packed`, `scale`) pair quantized along `axis`.

  Args:
    packed: uint8 `[..., n/2, ...]` with two E2M1 codes per byte along `axis`.
    scale: uint8 `[..., n/32, ...]` E8M0 exponents along `axis`; all other dims must
      equal `packed`'s.
    axis: the quantized (packed) axis. Use -2 for MaxText-orientation kernels
      `[..., in, out]` whose contraction axis `in` was quantized.
    dtype: output dtype.

  Returns:
    Dense `[..., n, ...]` array in `dtype`.
  """
  if packed.dtype != jnp.uint8:
    raise TypeError(f"packed must be uint8, got {packed.dtype}")
  if scale.dtype != jnp.uint8:
    raise TypeError(f"scale must be uint8, got {scale.dtype}")
  if packed.ndim != scale.ndim:
    raise ValueError(f"packed {packed.shape} and scale {scale.shape} have different ranks")
  axis = axis % packed.ndim
  other_p = packed.shape[:axis] + packed.shape[axis + 1 :]
  other_s = scale.shape[:axis] + scale.shape[axis + 1 :]
  if other_p != other_s:
    raise ValueError(f"packed {packed.shape} and scale {scale.shape} disagree outside axis {axis}")
  n = packed.shape[axis] * 2
  if scale.shape[axis] == 0 or n != scale.shape[axis] * MXFP4_GROUP_SIZE:
    raise ValueError(
        f"unpacked length {n} along axis {axis} is not {MXFP4_GROUP_SIZE} x {scale.shape[axis]} scale groups"
    )

  # Work with the quantized axis last, then move it back.
  packed_l = jnp.moveaxis(packed, axis, -1)
  scale_l = jnp.moveaxis(scale, axis, -1)
  # Every E2M1 value times a power of two is exact in bf16 (same exponent range as fp32), so
  # bf16 outputs are decoded in bf16 directly (half the peak memory of an fp32 decode);
  # any other dtype is decoded in fp32 and cast.
  compute_dtype = jnp.bfloat16 if jnp.dtype(dtype) == jnp.bfloat16 else jnp.float32
  lut = jnp.asarray(E2M1_VALUES, dtype=compute_dtype)
  low = lut[(packed_l & 0x0F).astype(jnp.int32)]
  high = lut[(packed_l >> 4).astype(jnp.int32)]
  values = jnp.stack([low, high], axis=-1)  # [..., n/2, 2]
  groups = values.reshape(*packed_l.shape[:-1], scale_l.shape[-1], MXFP4_GROUP_SIZE)
  out = groups * _e8m0_to_float(scale_l, compute_dtype)[..., None]
  out = out.reshape(*packed_l.shape[:-1], n).astype(dtype)
  return jnp.moveaxis(out, -1, axis)
