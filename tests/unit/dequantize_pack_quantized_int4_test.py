# Copyright 2023–2026 Google LLC
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

"""Tests for dequantize_pack_quantized_int4 (kimi-k2-thinking / kimi-k2.5 path).

Validated against the canonical compressed_tensors packer/unpacker:
  - unpack matches `unpack_from_int32` bit-for-bit
  - bf16 dequant matches the (int * bf16_scale).bf16 reference exactly
  - signed boundary values (-8, +7) decode correctly
  - zero-magnitude groups dequant to exactly 0
  - end-to-end kimi-k2-thinking expert tile shape (in=7168, out=2048, group=32)

Not run in GitHub runners (depends on torch + compressed_tensors).
"""

import unittest
import pytest
import torch

from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
    pack_to_int32,
    unpack_from_int32,
)

from maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_ckpt import (
    INT4_GROUP_SIZE,
    dequantize_pack_quantized_int4,
)


_NUM_BITS = 4
_QMAX = 2 ** (_NUM_BITS - 1) - 1  # 7 for symmetric int4
_QMIN = -(_QMAX + 1)  # -8


def _quantize_per_group(w_fp32: torch.Tensor, group_size: int = INT4_GROUP_SIZE):
  """Symmetric per-group int4 quantization → (int8 values in [-8, 7], fp32 scale)."""
  out_features, in_features = w_fp32.shape
  groups = w_fp32.reshape(out_features, in_features // group_size, group_size)
  absmax = groups.abs().amax(dim=-1, keepdim=True)
  scale = (absmax / _QMAX).clamp(min=1e-12)
  w_int = torch.round(groups / scale).clamp(_QMIN, _QMAX).to(torch.int8)
  return w_int.reshape(out_features, in_features), scale.squeeze(-1)


@pytest.mark.cpu_only
class DequantizePackQuantizedInt4Test(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)

  def test_unpack_matches_canonical_bit_exact(self):
    """Helper's unpack (with scale=1) recovers signed int values matching the
    reference unpacker exactly. Pins down nibble endianness and the +8 offset
    used by compressed_tensors' pack_to_int32."""
    out_features, in_features = 64, 128
    w_fp = torch.randn(out_features, in_features) * 0.1
    w_int, _ = _quantize_per_group(w_fp)
    packed = pack_to_int32(w_int, num_bits=_NUM_BITS)
    self.assertEqual(packed.dtype, torch.int32)
    self.assertEqual(tuple(packed.shape), (out_features, in_features // 8))

    canonical = unpack_from_int32(packed, num_bits=_NUM_BITS, shape=torch.Size([out_features, in_features]))
    self.assertTrue(torch.equal(canonical.to(torch.int8), w_int))

    ones = torch.ones(out_features, in_features // INT4_GROUP_SIZE, dtype=torch.float32)
    ours = dequantize_pack_quantized_int4(packed, ones, [out_features, in_features])
    self.assertTrue(torch.equal(ours.to(torch.int8), w_int))

  def test_dequant_matches_bf16_reference(self):
    """Full dequant matches (int * bf16_scale).bf16 bit-for-bit."""
    out_features, in_features = 64, 128
    w_fp = torch.randn(out_features, in_features) * 0.1
    w_int, scale_fp32 = _quantize_per_group(w_fp)
    packed = pack_to_int32(w_int, num_bits=_NUM_BITS)

    scale_bf16 = scale_fp32.to(torch.bfloat16)
    # Round-trip the scale through bf16 on both sides for a fair comparison.
    ref = (
        (
            w_int.reshape(out_features, in_features // INT4_GROUP_SIZE, INT4_GROUP_SIZE).to(torch.float32)
            * scale_bf16.to(torch.float32).unsqueeze(-1)
        )
        .reshape(out_features, in_features)
        .to(torch.bfloat16)
    )

    ours = dequantize_pack_quantized_int4(packed, scale_bf16, [out_features, in_features])
    self.assertEqual(ours.dtype, torch.bfloat16)
    self.assertTrue(torch.equal(ours, ref))

  def test_signed_boundary_values_decode(self):
    """Most-negative (-8) and most-positive (+7) int4 values decode correctly."""
    out_features, in_features = 16, 64
    scale = torch.ones(out_features, in_features // INT4_GROUP_SIZE, dtype=torch.bfloat16)

    minus_eight = torch.full((out_features, in_features), -8, dtype=torch.int8)
    packed = pack_to_int32(minus_eight, num_bits=_NUM_BITS)
    out = dequantize_pack_quantized_int4(packed, scale, [out_features, in_features])
    self.assertTrue(torch.all(out.to(torch.float32) == -8.0))

    plus_seven = torch.full((out_features, in_features), 7, dtype=torch.int8)
    packed = pack_to_int32(plus_seven, num_bits=_NUM_BITS)
    out = dequantize_pack_quantized_int4(packed, scale, [out_features, in_features])
    self.assertTrue(torch.all(out.to(torch.float32) == 7.0))

  def test_zero_group_dequants_to_zero(self):
    """Groups whose magnitudes round to zero ints must dequant to exactly 0."""
    out_features, in_features = 8, 64
    zero_int = torch.zeros(out_features, in_features, dtype=torch.int8)
    packed = pack_to_int32(zero_int, num_bits=_NUM_BITS)
    scale = torch.full((out_features, in_features // INT4_GROUP_SIZE), 0.123, dtype=torch.bfloat16)
    out = dequantize_pack_quantized_int4(packed, scale, [out_features, in_features])
    self.assertTrue(torch.all(out == 0))

  def test_in_features_not_divisible_raises(self):
    out_features, in_features = 4, 48  # 48 % 32 != 0
    int_vals = torch.zeros(out_features, in_features, dtype=torch.int8)
    packed = pack_to_int32(int_vals, num_bits=_NUM_BITS)
    scale = torch.ones(out_features, 2, dtype=torch.bfloat16)
    with self.assertRaises(ValueError):
      dequantize_pack_quantized_int4(packed, scale, [out_features, in_features])

  def test_kimi_k2_thinking_expert_tile_shape(self):
    """Realistic tile size: a single expert's gate_proj is [moe_intermediate_size,
    hidden_size] = [2048, 7168] for kimi-k2 / kimi-k2-thinking / kimi-k2.5."""
    out_features, in_features = 2048, 7168
    w_fp = torch.randn(out_features, in_features) * 0.05
    w_int, scale_fp32 = _quantize_per_group(w_fp)
    packed = pack_to_int32(w_int, num_bits=_NUM_BITS)

    scale_bf16 = scale_fp32.to(torch.bfloat16)
    out = dequantize_pack_quantized_int4(packed, scale_bf16, [out_features, in_features])
    self.assertEqual(tuple(out.shape), (out_features, in_features))
    self.assertEqual(out.dtype, torch.bfloat16)

    ref = (
        (
            w_int.reshape(out_features, in_features // INT4_GROUP_SIZE, INT4_GROUP_SIZE).to(torch.float32)
            * scale_bf16.to(torch.float32).unsqueeze(-1)
        )
        .reshape(out_features, in_features)
        .to(torch.bfloat16)
    )
    self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
  unittest.main()
