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

from maxtext.experimental.omni_poc.utils import processor_maxtext_omni as omni_processor


class TestProcessorMaxtextOmni(unittest.TestCase):
  """Concise test suite for processor_maxtext_omni."""

  def test_constants_and_offsets(self):
    self.assertEqual(omni_processor.DECODER_SPECIAL_TOKENS["qwen3"]["image_pad"], 151655)
    self.assertEqual(omni_processor.VISION_TOKENS_PER_IMAGE["gemma3"], 256)
    self.assertEqual(omni_processor.get_image_offsets_omni("gemma3", "qwen3", None), 255)
    self.assertEqual(
        omni_processor.get_image_offsets_omni("gemma3", "qwen3", SimpleNamespace(pixel_values=np.zeros((3, 1, 1, 3)))),
        255 * 3,
    )

  def test_unsupported_combination_raises_error(self):
    with self.assertRaises(ValueError):
      omni_processor.get_image_offsets_omni("llama4", "qwen3", None)
    with self.assertRaises(ValueError):
      omni_processor.add_extra_tokens_for_omni([1, 2], "llama4", "qwen3")
    with self.assertRaises(ValueError):
      omni_processor.get_bidirectional_mask_vision_omni("llama4", "qwen3", np.array([1, 2]))

  def test_add_extra_tokens_placeholder_expansion(self):
    pad_id = omni_processor.DECODER_SPECIAL_TOKENS["qwen3"]["image_pad"]
    # Expand single placeholder with custom count
    tokens = [10, pad_id, 20]
    out = omni_processor.add_extra_tokens_for_omni(tokens, "gemma3", "qwen3", num_tokens_per_image=3)
    np.testing.assert_array_equal(out, [10, pad_id, pad_id, pad_id, 20])

    # 2D array and dtype preservation
    tokens_i64 = np.array([[pad_id], [99]], dtype=np.int64)
    out_i64 = omni_processor.add_extra_tokens_for_omni(tokens_i64, "gemma3", "qwen3", num_tokens_per_image=2)
    self.assertEqual(out_i64.dtype, np.int64)
    np.testing.assert_array_equal(out_i64, [pad_id, pad_id, 99])

  def test_add_extra_tokens_prepend_and_text_only(self):
    # Prepend vision block when pixel values exist
    proc_out = SimpleNamespace(pixel_values=np.zeros((1, 1, 1, 3)))
    out = omni_processor.add_extra_tokens_for_omni(
        [1, 2], "gemma3", "qwen3", processor_output=proc_out, num_tokens_per_image=2
    )
    tokens_cfg = omni_processor.DECODER_SPECIAL_TOKENS["qwen3"]
    expected = [
        tokens_cfg["vision_start"],
        tokens_cfg["image_pad"],
        tokens_cfg["image_pad"],
        tokens_cfg["vision_end"],
        1,
        2,
    ]
    np.testing.assert_array_equal(out, expected)

    # Text-only returns original
    np.testing.assert_array_equal(omni_processor.add_extra_tokens_for_omni([1, 2], "gemma3", "qwen3"), [1, 2])


if __name__ == "__main__":
  unittest.main()
