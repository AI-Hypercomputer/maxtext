# Copyright 2023–2025 Google LLC
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

  def test_add_zero_image(self):
    tokens = np.asarray([1, 2, 3, 4, 5, 6])
    num_images = 0
    new_tokens = multimodal_utils.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    np.testing.assert_array_equal(new_tokens, tokens)
  
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


class TestLlama4ImageProcessing(unittest.TestCase):
  """Test Llama4 image processing"""

  def setUp(self):
    super().setUp()
    self.LLAMA4_TILES_NUM = 16
    self.LLAMA4_TILE_SIZE = 336
    self.NUM_IMAGE_CHANNELS = 3

  def test_get_best_resolution(self):
    image_1 = np.ones((224, 300, self.NUM_IMAGE_CHANNELS))
    image_2 = np.ones((536, 640, self.NUM_IMAGE_CHANNELS))

    possible_resolutions = multimodal_utils.find_supported_resolutions(
        max_num_chunks=self.LLAMA4_TILES_NUM, patch_size=self.LLAMA4_TILE_SIZE
    )
    best_resolution_1 = multimodal_utils.get_best_resolution(
        img_height=image_1.shape[0],
        image_width=image_1.shape[1],
        possible_resolutions=possible_resolutions,
        resize_to_max_canvas=False,
    )
    best_resolution_2 = multimodal_utils.get_best_resolution(
        img_height=image_2.shape[0],
        image_width=image_2.shape[1],
        possible_resolutions=possible_resolutions,
        resize_to_max_canvas=False,
    )
    self.assertEqual(best_resolution_1, (336, 336))
    self.assertEqual(best_resolution_2, (672, 672))

  def test_pad_to_best_fit_jax(self):
    image = np.zeros((536, 640, self.NUM_IMAGE_CHANNELS))
    best_resolution = (672, 672)
    padded_image = multimodal_utils.pad_to_best_fit_jax(image, best_resolution)
    self.assertEqual(padded_image.shape, (672, 672, self.NUM_IMAGE_CHANNELS))
    self.assertTrue(np.all(padded_image == 0))

  def test_split_to_tiles(self):
    image = np.ones((672, 672, self.NUM_IMAGE_CHANNELS))
    best_resolution = (672, 672)
    ratio_h, ratio_w = (
        best_resolution[0] // self.LLAMA4_TILE_SIZE,
        best_resolution[1] // self.LLAMA4_TILE_SIZE,
    )
    image_tiles = multimodal_utils.split_to_tiles(image, ratio_h, ratio_w)
    self.assertEqual(
        image_tiles.shape, (ratio_h * ratio_w, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE)
    )


class TestLlama4PostProcessing(unittest.TestCase):
  """Test Llama4 post-processing"""

  LLAMA4_FAKE_IMAGE_TOKEN = 200090  # <|image|>
  LLAMA4_BEGIN_IMAGE_TOKEN = 200080  # <|image_start|>
  LLAMA4_END_IMAGE_TOKEN = 200081  # <|image_end|>
  LLAMA4_PATCH_TOKEN = 200092  # <|patch|>
  LLAMA4_TILE_X_SEPARATOR_TOKEN = 200084  # <|tile_x_separator|>
  LLAMA4_TILE_Y_SEPARATOR_TOKEN = 200085  # <|tile_y_separator|>

  def setUp(self):
    super().setUp()
    self.NUM_IMAGE_CHANNELS = 3
    self.LLAMA4_TILE_SIZE = 336
    self.model_name = "llama4-17b-16e"

  def test_image_tokens_for_single_image(self):
    this_aspect_ratio = np.array([2, 2])
    num_patches_per_chunk = 4
    image_tokens = multimodal_utils.get_tokens_for_this_image(this_aspect_ratio, num_patches_per_chunk)
    expected_tokens = [
        self.LLAMA4_BEGIN_IMAGE_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_TILE_X_SEPARATOR_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_TILE_Y_SEPARATOR_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_TILE_X_SEPARATOR_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_TILE_Y_SEPARATOR_TOKEN,
        self.LLAMA4_FAKE_IMAGE_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_END_IMAGE_TOKEN,
    ]
    self.assertEqual(image_tokens, expected_tokens)

  def test_post_process_image_tokens(self):
    dummy_pixel_values = np.ones(
        (5, multimodal_utils.NUM_IMAGE_CHANNELS, multimodal_utils.LLAMA4_TILE_SIZE, multimodal_utils.LLAMA4_TILE_SIZE)
    )
    dummy_aspect_ratios = np.array([[2, 2]])
    dummy_tokens = np.array([1, 2, self.LLAMA4_FAKE_IMAGE_TOKEN, 4, 5])
    processor_output = multimodal_utils.PreprocessorOutput(
        pixel_values=dummy_pixel_values,
        aspect_ratios=dummy_aspect_ratios,
    )

    image_offsets = multimodal_utils.get_image_offsets(model_name=self.model_name, processor_output=processor_output)
    post_processed_tokens = multimodal_utils.add_extra_tokens_for_images_llama4(dummy_tokens, processor_output)
    self.assertEqual(post_processed_tokens.shape[0], dummy_tokens.shape[0] + image_offsets)


if __name__ == "__main__":
  unittest.main()
