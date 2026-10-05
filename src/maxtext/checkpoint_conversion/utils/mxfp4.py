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

"""NumPy codec for `compressed-tensors` MXFP4 (`mxfp4-pack-quantized`) weights.

Hugging Face checkpoints produced by `llm-compressor` with
`quantization_config.format == "mxfp4-pack-quantized"` (e.g. Kimi-K3's routed
experts) replace each quantized `nn.Linear` weight `X.weight` of shape `[out, in]`
with two tensors:

  * `X.weight_packed`: uint8 `[out, in // 2]`. Each byte holds two OCP FP4 E2M1
    codes; the LOW nibble is the even input column, the HIGH nibble the odd one.
  * `X.weight_scale`:  uint8 `[out, in // 32]`. One E8M0 exponent per group of 32
    consecutive input columns; the multiplier is `2 ** (scale - 127)`.

MaxText has no sub-byte weight format at runtime, so the checkpoint converter
dequantizes these pairs to a dense floating-point `X.weight` on the fly (the same
strategy used for gpt-oss, DeepSeek-V4 and Kimi-K2). This module owns that codec.
The decode matches the reference implementations in `transformers`
(`integrations/mxfp4.py`), `compressed_tensors` and the JAX/NumPy codecs used by
other Google TPU stacks for the same release.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

MXFP4_PACKED_SUFFIX = ".weight_packed"
MXFP4_SCALE_SUFFIX = ".weight_scale"
MXFP4_WEIGHT_SUFFIX = ".weight"
MXFP4_CODES_PER_BYTE = 2
MXFP4_GROUP_SIZE = 32
E8M0_BIAS = 127

# OCP FP4 E2M1 magnitude for codes 0..7 (bit 3 is the sign).
_E2M1_MAGNITUDE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Full 16-entry signed lookup table indexed by the 4-bit code.
E2M1_LUT = np.concatenate([_E2M1_MAGNITUDE, -_E2M1_MAGNITUDE]).astype(np.float32)
E2M1_MAX = 6.0


def unpack_e2m1(packed: np.ndarray) -> np.ndarray:
  """Unpacks uint8 `[..., n]` into float32 E2M1 values `[..., 2n]` (low nibble first)."""
  packed = np.asarray(packed)
  if packed.dtype != np.uint8:
    raise TypeError(f"weight_packed must be uint8, got {packed.dtype}")
  low = packed & 0x0F
  high = packed >> 4
  codes = np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)
  return E2M1_LUT[codes]


def e8m0_to_float32(scale: np.ndarray) -> np.ndarray:
  """Decodes uint8 E8M0 exponents into float32 power-of-two multipliers."""
  scale = np.asarray(scale)
  if scale.dtype != np.uint8:
    raise TypeError(f"weight_scale must be uint8, got {scale.dtype}")
  return np.exp2(scale.astype(np.float32) - E8M0_BIAS)


def dequantize_mxfp4_packed(packed: np.ndarray, scale: np.ndarray, dtype: Any = np.float32) -> np.ndarray:
  """Decodes an `mxfp4-pack-quantized` (`weight_packed`, `weight_scale`) pair.

  Args:
    packed: uint8 `[..., out, in // 2]`.
    scale: uint8 `[..., out, in // 32]` E8M0 exponents.
    dtype: output dtype (e.g. `np.float32`, `ml_dtypes.bfloat16`).

  Returns:
    `[..., out, in]` dense weights in `dtype`.
  """
  values = unpack_e2m1(packed)
  scale = np.asarray(scale)
  if values.shape[:-1] != scale.shape[:-1]:
    raise ValueError(f"packed {packed.shape} and scale {scale.shape} disagree on leading dims")
  if values.shape[-1] % scale.shape[-1] != 0:
    raise ValueError(f"unpacked width {values.shape[-1]} is not a multiple of scale groups {scale.shape[-1]}")
  group_size = values.shape[-1] // scale.shape[-1]
  if group_size != MXFP4_GROUP_SIZE:
    raise ValueError(f"expected group size {MXFP4_GROUP_SIZE}, inferred {group_size}")
  factor = np.repeat(e8m0_to_float32(scale), group_size, axis=-1)
  return (values * factor).astype(dtype)


def mxfp4_pair_keys(weight_key: str) -> Optional[tuple[str, str]]:
  """For `X.weight` returns `(X.weight_packed, X.weight_scale)`; else None."""
  if not weight_key.endswith(MXFP4_WEIGHT_SUFFIX):
    return None
  base = weight_key[: -len(MXFP4_WEIGHT_SUFFIX)]
  return base + MXFP4_PACKED_SUFFIX, base + MXFP4_SCALE_SUFFIX


def resolve_mxfp4_pair(weight_key: str, has_key: Callable[[str], bool]) -> Optional[tuple[str, str]]:
  """Returns the packed/scale key pair backing `weight_key` if the container has it."""
  pair = mxfp4_pair_keys(weight_key)
  if pair is None:
    return None
  packed_key, scale_key = pair
  if has_key(packed_key) and has_key(scale_key):
    return packed_key, scale_key
  if has_key(packed_key) != has_key(scale_key):
    raise KeyError(f"MXFP4 pair for {weight_key!r} is incomplete: need both {packed_key!r} and {scale_key!r}")
  return None


def is_mxfp4_sidecar_key(key: str) -> bool:
  """True for `*.weight_packed` / `*.weight_scale` keys (consumed via their `.weight`)."""
  return key.endswith(MXFP4_PACKED_SUFFIX) or key.endswith(MXFP4_SCALE_SUFFIX)


# -----------------------------------------------------------------------------
# Reference quantizer. Not used by the converter; it exists so tests can build
# realistic packed checkpoints and check round-trip error bounds.
# -----------------------------------------------------------------------------
def _round_to_e2m1(x: np.ndarray) -> np.ndarray:
  """Rounds float32 values (already divided by the group scale) to the nearest E2M1 value.

  Uses round-half-to-even on the magnitude grid, matching the OCP MX reference
  behaviour; magnitudes above 6.0 saturate.
  """
  mag = np.abs(x)
  # Grid is non-uniform: {0, .5, 1, 1.5, 2, 3, 4, 6}. Round in the index domain by
  # comparing against midpoints, with ties going to the even code.
  mids = (_E2M1_MAGNITUDE[:-1] + _E2M1_MAGNITUDE[1:]) / 2.0  # 7 midpoints
  idx = np.searchsorted(mids, mag, side="right")  # values strictly above a midpoint round up
  on_tie = np.isin(mag, mids)
  # searchsorted(side="right") already rounded ties up; move odd results back down.
  idx = np.where(on_tie & (idx % 2 == 1), idx - 1, idx)
  idx = np.clip(idx, 0, 7)
  return np.sign(x) * _E2M1_MAGNITUDE[idx]


def quantize_mxfp4_reference(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Quantizes float `[..., out, in]` into (`weight_packed`, `weight_scale`).

  Per group of 32 input columns the shared E8M0 exponent is the smallest power of
  two `2**e` such that `amax / 2**e <= 6.0` (i.e. `e = ceil(log2(amax / 6))`), so
  no element saturates and the round-trip error is bounded elementwise by
  `2**e` (half the widest E2M1 grid gap, 4 -> 6). Returns
  `(weight_packed uint8 [..., out, in//2], weight_scale uint8 [..., out, in//32])`.
  """
  w = np.asarray(weight, dtype=np.float32)
  out_dim, in_dim = w.shape[-2], w.shape[-1]
  if in_dim % MXFP4_GROUP_SIZE != 0:
    raise ValueError(f"in_dim {in_dim} must be a multiple of {MXFP4_GROUP_SIZE}")
  groups = w.reshape(*w.shape[:-1], in_dim // MXFP4_GROUP_SIZE, MXFP4_GROUP_SIZE)
  amax = np.max(np.abs(groups), axis=-1)
  safe_amax = np.maximum(amax, np.finfo(np.float32).tiny)
  exp = np.where(amax > 0, np.ceil(np.log2(safe_amax / E2M1_MAX)), -E8M0_BIAS)
  exp = np.clip(exp, -E8M0_BIAS, 255 - E8M0_BIAS)
  scale_u8 = (exp + E8M0_BIAS).astype(np.uint8)
  factor = np.exp2(exp.astype(np.float32))[..., None]
  q = _round_to_e2m1(groups / factor).reshape(*w.shape[:-1], in_dim)

  codes = np.searchsorted(_E2M1_MAGNITUDE, np.abs(q)).astype(np.uint8)
  codes = codes | np.where(np.signbit(q) & (q != 0), 0x08, 0).astype(np.uint8)
  packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
  assert packed.shape[-2:] == (out_dim, in_dim // MXFP4_CODES_PER_BYTE)
  return packed, scale_u8
