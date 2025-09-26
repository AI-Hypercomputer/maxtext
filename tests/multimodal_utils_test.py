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
    self.assertEqual(padded_image.shape, (self.LLAMA4_TILES_NUM, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE))
    self.assertEqual(image_mask.shape, (self.LLAMA4_TILES_NUM,))
    self.assertEqual(np.sum(image_mask), 5)
    self.assertTrue(np.all(padded_image[5:] == 0))


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
    # Setup: batch_size=1, seq_len=8, d=4
    batch_size = 1
    seq_len = 8
    d = 4
    num_images = 2
    num_toks_per_image = 2

    # text_embeddings: [batch_size, seq_len, d]
    text_embeddings = np.arange(batch_size * seq_len * d).reshape(batch_size, seq_len, d)

    # vision_embeddings: [batch_size, num_images, num_toks_per_image, d]
    vision_embeddings = np.arange(batch_size * num_images * num_toks_per_image * d).reshape(batch_size, num_images, num_toks_per_image, d) + 1000

    # mask: [batch_size, seq_len] - place vision embeddings at positions 2, 5, 6
    mask = np.zeros((batch_size, seq_len), dtype=np.int32)
    mask[:, 2] = 1  # first image
    mask[:, 5:7] = 1  # second image

    # image_masks: [batch_size, num_images, num_toks_per_image] - only some valid (only first token of each image)
    image_masks = np.zeros((batch_size, num_images, num_toks_per_image), dtype=np.int32)
    image_masks[0, 0, 0] = 1  # Only the first token of the first image is valid
    image_masks[0, 1, 0] = 1  # Both tokens of the second image are valid
    image_masks[0, 1, 1] = 1 

    # Call the function with image_masks (should only insert valid vision embeddings)
    merged = multimodal_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, image_masks)

    # Image mask is null - all vision embeddings are valid
    merged_null = multimodal_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, None)

    # The results should be different
    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(merged, merged_null)

    # For merged: only valid vision embeddings should be inserted, others remain as text_embeddings
    # The valid vision embeddings (after mask sorting) are vision_embeddings[0,0,0] and vision_embeddings[0,1,0]
    self.assertTrue(np.all(merged[0,2] == vision_embeddings[0,0,0]))
    self.assertTrue(np.all(merged[0,5] == vision_embeddings[0,1,0]))
    self.assertTrue(np.all(merged[0,6] == vision_embeddings[0,1,1]))
    
    # The other masked positions should not match the original vision_embeddings (since they are not valid)
    self.assertFalse(np.all(merged[0,3] == vision_embeddings[0,0,1]))

    # For merged_null all vision embeddings should be inserted in order (last embedding will be trucated)
    np.testing.assert_array_equal(merged_null[0,2], vision_embeddings[0,0,0])
    np.testing.assert_array_equal(merged_null[0,5], vision_embeddings[0,0,1])
    np.testing.assert_array_equal(merged_null[0,6], vision_embeddings[0,1,0])
    
    # The first position should always be preserved
    np.testing.assert_array_equal(merged[0,0], text_embeddings[0,0])
    np.testing.assert_array_equal(merged_null[0,0], text_embeddings[0,0])


if __name__ == "__main__":
  unittest.main()
