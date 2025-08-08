"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Tests for dequantize_mxfp4.py (not run in GitHub runners). """

import unittest
import torch
from MaxText.scratch_code import dequantize_mxfp4


FP4_VALUES = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


class DequantizationTest(unittest.TestCase):

  def test_bf16_value_range(self):
    blocks = torch.randint(0, 255, (10 * 1024, 16), dtype=torch.uint8)
    scales = 127 * torch.ones((blocks.shape[0],), dtype=torch.uint8)
    dequant = dequantize_mxfp4.dequantize_mxfp4(blocks, scales, torch.bfloat16)
    assert torch.all(torch.unique(dequant) == torch.unique(torch.tensor(FP4_VALUES)))

  def test_fp16_value_range(self):
    blocks = torch.randint(0, 255, (10 * 1024, 16), dtype=torch.uint8)
    scales = 127 * torch.ones((blocks.shape[0],), dtype=torch.uint8)
    dequant = dequantize_mxfp4.dequantize_mxfp4(blocks, scales, torch.float16)
    assert torch.all(torch.unique(dequant) == torch.unique(torch.tensor(FP4_VALUES)))

  def test_fp32_value_range(self):
    blocks = torch.randint(0, 255, (10 * 1024, 16), dtype=torch.uint8)
    scales = 127 * torch.ones((blocks.shape[0],), dtype=torch.uint8)
    dequant = dequantize_mxfp4.dequantize_mxfp4(blocks, scales, torch.float32)
    assert torch.all(torch.unique(dequant) == torch.unique(torch.tensor(FP4_VALUES)))


if __name__ == "__main__":
  unittest.main()
