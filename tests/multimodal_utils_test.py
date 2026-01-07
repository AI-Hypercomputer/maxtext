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
        max_num_tiles=self.LLAMA4_TILES_NUM, tile_size=self.LLAMA4_TILE_SIZE
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

  def test_pad_to_max_tiles(self):
    image = np.ones((5, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE))
    padded_image, image_mask = multimodal_utils.pad_to_max_tiles(image, self.LLAMA4_TILES_NUM)
    self.assertEqual(
        padded_image.shape, (self.LLAMA4_TILES_NUM, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE)
    )
    self.assertEqual(image_mask.shape, (self.LLAMA4_TILES_NUM,))
    self.assertEqual(np.sum(image_mask), 5)
    self.assertEqual(np.sum(padded_image[5:]), 0)


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

  def test_merge_mm_embeddings(self):
    # Setup Dummy Data
    batch_size = 1
    seq_len = 64
    d = 4
    num_images = 2
    num_tiles = 4
    num_toks_per_image = 8

    # text_embeddings: (B, S, D) -> (1, 64, 4)
    text_embeddings = np.arange(batch_size * seq_len * d, dtype=np.float32).reshape(batch_size, seq_len, d)

    # vision_embeddings: (B * N, T, K, D) -> (2, 4, 8, 4)
    vision_embeddings = (
        np.arange(batch_size * num_images * num_tiles * num_toks_per_image * d, dtype=np.float32).reshape(
            batch_size * num_images, num_tiles, num_toks_per_image, d
        )
        + 1000
    )

    # mask: (B, S) -> (1, 64)
    # Total of 8 + 16 = 24 token slots available for images.
    mask = np.zeros((batch_size, seq_len), dtype=np.int32)
    mask[:, 2:10] = 1  # 8 slots for the first image's valid tiles
    mask[:, 20:36] = 1  # 16 slots for the second image's valid tiles

    # image_masks: (B * N, T) -> (2, 4)
    # Specifies which tiles are valid.
    image_masks = np.zeros((batch_size * num_images, num_tiles), dtype=np.int32)
    # Image 0 has 1 valid tile -> 1 * 8 = 8 valid tokens.
    image_masks[0, 0] = 1
    # Image 1 has 2 valid tiles -> 2 * 8 = 16 valid tokens.
    image_masks[1, 0] = 1
    image_masks[1, 1] = 1
    # Total valid tokens = 8 + 16 = 24. This matches the mask slots.

    # Case 1: Use the image_mask to filter for valid tiles.
    merged = multimodal_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, image_masks)

    # Case 2: No image_mask, so all vision tokens are used in order.
    merged_null = multimodal_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, None)

    # The results should be different since one is masked and one is not.
    self.assertFalse(np.array_equal(merged, merged_null))

    # The code gathers all valid tiles first and then inserts them sequentially.
    # Valid tiles are: vision_embeddings[0, 0], vision_embeddings[1, 0], vision_embeddings[1, 1]

    # The first 8 slots (2:10) should be filled by the first valid tile's tokens.
    first_valid_tile = vision_embeddings[0, 0, :, :]
    np.testing.assert_array_equal(merged[0, 2:10], first_valid_tile)

    # The next 8 slots (20:28) get the second valid tile's tokens.
    second_valid_tile = vision_embeddings[1, 0, :, :]
    np.testing.assert_array_equal(merged[0, 20:28], second_valid_tile)

    # The final 8 slots (28:36) get the third valid tile's tokens.
    third_valid_tile = vision_embeddings[1, 1, :, :]
    np.testing.assert_array_equal(merged[0, 28:36], third_valid_tile)

    # When no mask is provided all vision tiles are inserted sequentially in their natural flattened order.
    np.testing.assert_array_equal(merged_null[0, 2:10], vision_embeddings[0, 0, :, :])
    np.testing.assert_array_equal(merged_null[0, 20:28], vision_embeddings[0, 1, :, :])
    np.testing.assert_array_equal(merged_null[0, 28:36], vision_embeddings[0, 2, :, :])

    # Verify that parts of the text sequence that were NOT masked remain untouched.
    np.testing.assert_array_equal(merged[0, 10:20], text_embeddings[0, 10:20])
    np.testing.assert_array_equal(merged[0, 36:], text_embeddings[0, 36:])

    # The first position should always be preserved.
    np.testing.assert_array_equal(merged[0, 0], text_embeddings[0, 0])
    np.testing.assert_array_equal(merged_null[0, 0], text_embeddings[0, 0])


if __name__ == "__main__":
  unittest.main()
