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

    # Case 1: Use the token_masks to filter for valid tiles.
    merged = multimodal_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, token_masks=image_masks)

    # Case 2: No token_masks, so all vision tokens are used in order.
    merged_null = multimodal_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, token_masks=None)

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


class TestQwen3OmniTokenExpansion(unittest.TestCase):
  """Test token expansion for Qwen3-Omni multimodal inputs"""

  def test_image_expansion_single(self):
    """Test single image token expansion"""
    # Input: [text, image_token, text]
    tokens = np.array([1, 2, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, 3, 4])

    # Image grid: 1 frame, 4x4 spatial (after merge: 2x2 = 4 tokens)
    image_grid_thw = np.array([[1, 4, 4]])

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
        use_audio_in_video=False,
    )

    # Expected: [1, 2, img, img, img, img, 3, 4]
    expected = np.array([1, 2, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN,
                        multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, 3, 4])
    np.testing.assert_array_equal(result, expected)

  def test_video_expansion_no_audio(self):
    """Test video token expansion without audio-in-video"""
    # Input: [text, video_token, text]
    tokens = np.array([1, multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN, 2])

    # Video grid: 2 frames, 4x4 spatial (total: 2*4*4/4 = 8 tokens)
    video_grid_thw = np.array([[2, 4, 4]])

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=2,
        use_audio_in_video=False,
    )

    # Expected: [1, vid, vid, vid, vid, vid, vid, vid, vid, 2]
    expected_length = 1 + 8 + 1
    self.assertEqual(len(result), expected_length)
    self.assertEqual(result[0], 1)
    self.assertEqual(result[-1], 2)
    # All middle tokens should be VIDEO_TOKEN
    self.assertTrue(np.all(result[1:9] == multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN))

  def test_audio_expansion_standalone(self):
    """Test standalone audio token expansion"""
    tokens = np.array([1, multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN, 2])
    audio_lengths = np.array([5])

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        audio_lengths=audio_lengths,
        use_audio_in_video=False,
    )

    # Expected: [1, aud, aud, aud, aud, aud, 2]
    expected = np.array([1, multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN, multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN,
                        multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN, multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN,
                        multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN, 2])
    np.testing.assert_array_equal(result, expected)

  def test_audio_in_video_interleaving(self):
    """Test audio-in-video mode with temporal interleaving"""
    # Input: [text, <vision_start>, <video_pad>, <vision_end>, text]
    tokens = np.array([1, multimodal_utils.QWEN3_OMNI_VISION_START_TOKEN, multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN,
                      multimodal_utils.QWEN3_OMNI_VISION_END_TOKEN, 2])

    # Small video: 2 frames, 2x2 spatial (after merge 1x1 = 2 tokens total)
    video_grid_thw = np.array([[2, 2, 2]])
    audio_lengths = np.array([3])  # 3 audio tokens
    second_per_grids = np.array([1.0])  # 1 second per grid

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        video_grid_thw=video_grid_thw,
        audio_lengths=audio_lengths,
        spatial_merge_size=2,
        use_audio_in_video=True,
        second_per_grids=second_per_grids,
        position_id_per_seconds=25,
    )

    # Check structure: [1, <vision_start>, <audio_start>, <interleaved>, <audio_end>, <vision_end>, 2]
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], multimodal_utils.QWEN3_OMNI_VISION_START_TOKEN)
    self.assertEqual(result[2], multimodal_utils.QWEN3_OMNI_AUDIO_START_TOKEN)
    self.assertEqual(result[-2], multimodal_utils.QWEN3_OMNI_VISION_END_TOKEN)
    self.assertEqual(result[-1], 2)

    # Check interleaved section contains both video and audio tokens
    interleaved = result[3:-2]
    has_video = np.any(interleaved == multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN)
    has_audio = np.any(interleaved == multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN)
    self.assertTrue(has_video, "Interleaved section should contain video tokens")
    self.assertTrue(has_audio, "Interleaved section should contain audio tokens")

  def test_audio_in_video_multiple(self):
    """Test audio-in-video mode with multiple videos"""
    # Input: [text, <vision_start>, <video_pad>, <vision_end>, text, <vision_start>, <video_pad>, <vision_end>, text]
    tokens = np.array([
        1,
        multimodal_utils.QWEN3_OMNI_VISION_START_TOKEN,
        multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN,
        multimodal_utils.QWEN3_OMNI_VISION_END_TOKEN,
        2,
        multimodal_utils.QWEN3_OMNI_VISION_START_TOKEN,
        multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN,
        multimodal_utils.QWEN3_OMNI_VISION_END_TOKEN,
        3
    ])

    # Two videos: 2 frames each, 2x2 spatial (2 tokens each after merge)
    video_grid_thw = np.array([[2, 2, 2], [2, 2, 2]])
    audio_lengths = np.array([3, 4])  # Different audio lengths
    second_per_grids = np.array([1.0, 1.5])  # Different temporal scales

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        video_grid_thw=video_grid_thw,
        audio_lengths=audio_lengths,
        spatial_merge_size=2,
        use_audio_in_video=True,
        second_per_grids=second_per_grids,
        position_id_per_seconds=25,
    )

    # Find the two vision_start tokens
    vision_starts = np.where(result == multimodal_utils.QWEN3_OMNI_VISION_START_TOKEN)[0]
    self.assertEqual(len(vision_starts), 2, "Should have 2 vision_start tokens")

    # Find the two vision_end tokens
    vision_ends = np.where(result == multimodal_utils.QWEN3_OMNI_VISION_END_TOKEN)[0]
    self.assertEqual(len(vision_ends), 2, "Should have 2 vision_end tokens")

    # Check first video section
    first_section = result[vision_starts[0]:vision_ends[0]+1]
    self.assertEqual(first_section[0], multimodal_utils.QWEN3_OMNI_VISION_START_TOKEN)
    self.assertEqual(first_section[1], multimodal_utils.QWEN3_OMNI_AUDIO_START_TOKEN)
    self.assertEqual(first_section[-1], multimodal_utils.QWEN3_OMNI_VISION_END_TOKEN)
    has_video_1 = np.any(first_section == multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN)
    has_audio_1 = np.any(first_section == multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN)
    self.assertTrue(has_video_1, "First video section should contain video tokens")
    self.assertTrue(has_audio_1, "First video section should contain audio tokens")

    # Check second video section
    second_section = result[vision_starts[1]:vision_ends[1]+1]
    self.assertEqual(second_section[0], multimodal_utils.QWEN3_OMNI_VISION_START_TOKEN)
    self.assertEqual(second_section[1], multimodal_utils.QWEN3_OMNI_AUDIO_START_TOKEN)
    self.assertEqual(second_section[-1], multimodal_utils.QWEN3_OMNI_VISION_END_TOKEN)
    has_video_2 = np.any(second_section == multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN)
    has_audio_2 = np.any(second_section == multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN)
    self.assertTrue(has_video_2, "Second video section should contain video tokens")
    self.assertTrue(has_audio_2, "Second video section should contain audio tokens")

  def test_batch_processing_2d(self):
    """Test that 2D batched input is processed correctly"""
    # Batch of 2 sequences
    tokens = np.array([
        [1, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, 2, 0],  # First sequence (padded)
        [3, 4, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, 5],  # Second sequence
    ])

    # Both images same size: 1 frame, 4x4 spatial (4 tokens after merge)
    image_grid_thw = np.array([[1, 4, 4], [1, 4, 4]])

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
        use_audio_in_video=False,
    )

    # Result should be 2D
    self.assertEqual(result.ndim, 2)
    self.assertEqual(result.shape[0], 2)

    # Check first sequence: [1, img, img, img, img, 2, 0, 0]
    self.assertEqual(result[0, 0], 1)
    self.assertTrue(np.all(result[0, 1:5] == multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN))

    # Check second sequence: [3, 4, img, img, img, img, 5, 0]
    self.assertEqual(result[1, 0], 3)
    self.assertEqual(result[1, 1], 4)
    self.assertTrue(np.all(result[1, 2:6] == multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN))

  def test_1d_input_returns_1d(self):
    """Test that 1D input returns 1D output"""
    tokens = np.array([1, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, 2])
    image_grid_thw = np.array([[1, 4, 4]])

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
        use_audio_in_video=False,
    )

    # Result should be 1D
    self.assertEqual(result.ndim, 1)

  def test_no_multimodal_tokens(self):
    """Test that text-only input passes through unchanged"""
    tokens = np.array([1, 2, 3, 4, 5])

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        use_audio_in_video=False,
    )

    np.testing.assert_array_equal(result, tokens)

  def test_multiple_images(self):
    """Test expansion of multiple image tokens"""
    tokens = np.array([1, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, 2, multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN, 3])

    # Two images with same dimensions
    image_grid_thw = np.array([[1, 4, 4], [1, 4, 4]])

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
        use_audio_in_video=False,
    )

    # Expected: [1, img*4, 2, img*4, 3]
    expected_length = 1 + 4 + 1 + 4 + 1
    self.assertEqual(len(result), expected_length)
    self.assertEqual(result[0], 1)
    self.assertEqual(result[5], 2)
    self.assertEqual(result[-1], 3)

  def test_videos_with_different_lengths(self):
    """Test multiple videos with different dimensions (different lengths)"""
    # Input: [text, video1, text, video2, text]
    tokens = np.array([1, multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN, 2,
                      multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN, 3])

    # First video: 2 frames, 4x4 spatial -> 2*4*4/4 = 8 tokens
    # Second video: 4 frames, 2x2 spatial -> 4*2*2/4 = 4 tokens (different!)
    video_grid_thw = np.array([
        [2, 4, 4],  # First video
        [4, 2, 2],  # Second video (different dimensions)
    ])

    result = multimodal_utils.add_extra_tokens_for_qwen3_omni(
        tokens=tokens,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=2,
        use_audio_in_video=False,
    )

    # Expected: [1, vid*8, 2, vid*4, 3]
    expected_length = 1 + 8 + 1 + 4 + 1
    self.assertEqual(len(result), expected_length,
                     f"Expected length {expected_length}, got {len(result)}")

    # Verify structure
    self.assertEqual(result[0], 1)
    self.assertEqual(result[9], 2)  # After 1 + 8 video tokens
    self.assertEqual(result[-1], 3)

    # Check first video has 8 tokens
    first_video_tokens = result[1:9]
    self.assertTrue(np.all(first_video_tokens == multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN))

    # Check second video has 4 tokens
    second_video_tokens = result[10:14]
    self.assertTrue(np.all(second_video_tokens == multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN))


if __name__ == "__main__":
  unittest.main()
