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

"""Tests for MaxText vLLM hybrid-cache layout helpers."""

from types import SimpleNamespace
import unittest

import pytest
import torch

from maxtext.integration.vllm._hybrid_cache import build_qwen_gdn_cache_layout


pytestmark = [pytest.mark.post_training]


class QwenGdnCacheLayoutTest(unittest.TestCase):
  """Verify the mixed-precision recurrent-cache contract."""

  @pytest.mark.cpu_only
  def test_recurrent_state_is_float32_and_page_size_uses_each_dtype(self):
    cfg = SimpleNamespace(
        gdn_num_value_heads=32,
        gdn_num_key_heads=16,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_kernel_dim=4,
    )

    shapes, dtypes, page_size_bytes = build_qwen_gdn_cache_layout(cfg, torch)

    self.assertEqual(shapes, ((3, 8192), (32, 128, 128)))
    self.assertEqual(dtypes, (torch.bfloat16, torch.float32))
    self.assertEqual(page_size_bytes, 2_146_304)


if __name__ == "__main__":
  unittest.main()
