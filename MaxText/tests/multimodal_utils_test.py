"""
Copyright 2024 Google LLC

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

""" Tests for the common MaxText utilities """
import unittest
import numpy as np

from MaxText import multimodal_utils


class TestTextImageFusionGemma3(unittest.TestCase):
  """Test inserting place_holder tokens for image"""

  def setUp(self):
    super().setUp()
    self.BEGIN_IMAGE_TOKEN = 255999
    self.mm_tokens = [self.BEGIN_IMAGE_TOKEN, -2, -2]

  def test_add_single_image(self):
    tokens = np.asarray([1, 2, 3, self.BEGIN_IMAGE_TOKEN, 5, 6])
    num_images = 1
    new_tokens = multimodal_utils.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray([1, 2, 3] + self.mm_tokens + [5, 6])
    np.testing.assert_array_equal(new_tokens, expected)

  def test_add_two_images(self):
    tokens = np.asarray([1, self.BEGIN_IMAGE_TOKEN, 3, 4, self.BEGIN_IMAGE_TOKEN, 6])
    num_images = 2
    new_tokens = multimodal_utils.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray([1] + self.mm_tokens + [3, 4] + self.mm_tokens + [6])
    np.testing.assert_array_equal(new_tokens, expected)

  def test_add_images_in_batch(self):
    tokens = np.asarray(
        [[1, 2, 3, self.BEGIN_IMAGE_TOKEN, 5, 6], [1, self.BEGIN_IMAGE_TOKEN, 3, 4, self.BEGIN_IMAGE_TOKEN, 6]]
    )
    num_images = 2
    new_tokens = multimodal_utils.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray(
        [
            [1, 2, 3] + self.mm_tokens + [5, 6] + [0] * (len(self.mm_tokens) - 1),
            [1] + self.mm_tokens + [3, 4] + self.mm_tokens + [6],
        ]
    )
    np.testing.assert_array_equal(new_tokens, expected)


if __name__ == "__main__":
  unittest.main()
