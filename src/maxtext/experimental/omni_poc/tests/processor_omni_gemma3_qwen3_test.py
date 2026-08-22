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

"""Unit tests for Omni multimodal processor."""

from types import SimpleNamespace
import unittest
import numpy as np

from maxtext.experimental.omni_poc.utils import processor_omni_gemma3_qwen3 as omni_processor


class TestProcessorOmniGemma3Qwen3(unittest.TestCase):
  """Concise test suite for processor_omni_gemma3_qwen3."""

  def test_constants_and_offsets(self):
    self.assertEqual(omni_processor.IMAGE_PAD_ID, 151655)
    self.assertEqual(omni_processor.get_image_offsets_omni(None), 255)
    self.assertEqual(
        omni_processor.get_image_offsets_omni(SimpleNamespace(pixel_values=np.zeros((3, 1, 1, 3)))),
        255 * 3,
    )

  def test_add_extra_tokens_placeholder_expansion(self):
    pad_id = omni_processor.IMAGE_PAD_ID
    # Expand single placeholder with custom count
    tokens = [10, pad_id, 20]
    out = omni_processor.add_extra_tokens_for_omni(tokens, num_tokens_per_image=3)
    np.testing.assert_array_equal(out, [10, pad_id, pad_id, pad_id, 20])

    # 2D array and dtype preservation
    tokens_i64 = np.array([[pad_id], [99]], dtype=np.int64)
    out_i64 = omni_processor.add_extra_tokens_for_omni(tokens_i64, num_tokens_per_image=2)
    self.assertEqual(out_i64.dtype, np.int64)
    np.testing.assert_array_equal(out_i64, [pad_id, pad_id, 99])

  def test_add_extra_tokens_prepend_and_text_only(self):
    # Prepend vision block when pixel values exist
    proc_out = SimpleNamespace(pixel_values=np.zeros((1, 1, 1, 3)))
    out = omni_processor.add_extra_tokens_for_omni([1, 2], processor_output=proc_out, num_tokens_per_image=2)
    expected = [
        omni_processor.VISION_START_ID,
        omni_processor.IMAGE_PAD_ID,
        omni_processor.IMAGE_PAD_ID,
        omni_processor.VISION_END_ID,
        1,
        2,
    ]
    np.testing.assert_array_equal(out, expected)

    # Text-only returns original
    np.testing.assert_array_equal(omni_processor.add_extra_tokens_for_omni([1, 2]), [1, 2])


if __name__ == "__main__":
  unittest.main()
