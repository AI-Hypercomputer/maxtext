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

"""Tests for the common MaxText utilities"""
import os
import types
import unittest
import numpy as np

from maxtext.common.common_types import DecoderBlockType, VisionEncoderBlockType
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.multimodal import processor_gemma3
from maxtext.multimodal import processor_gemma4
from maxtext.multimodal import processor_llama4


class TestTextImageFusionGemma3(unittest.TestCase):
  """Test inserting place_holder tokens for image"""

  def setUp(self):
    super().setUp()
    self.BEGIN_IMAGE_TOKEN = 255999
    self.mm_tokens = [self.BEGIN_IMAGE_TOKEN, -2, -2]

  def test_add_zero_image(self):
    tokens = np.asarray([1, 2, 3, 4, 5, 6])
    num_images = 0
    new_tokens = processor_gemma3.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    np.testing.assert_array_equal(new_tokens, tokens)

  def test_add_single_image(self):
    tokens = np.asarray([1, 2, 3, self.BEGIN_IMAGE_TOKEN, 5, 6])
    num_images = 1
    new_tokens = processor_gemma3.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray([1, 2, 3] + self.mm_tokens + [5, 6])
    np.testing.assert_array_equal(new_tokens, expected)

  def test_add_two_images(self):
    tokens = np.asarray([1, self.BEGIN_IMAGE_TOKEN, 3, 4, self.BEGIN_IMAGE_TOKEN, 6])
    num_images = 2
    new_tokens = processor_gemma3.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray([1] + self.mm_tokens + [3, 4] + self.mm_tokens + [6])
    np.testing.assert_array_equal(new_tokens, expected)

  def test_add_images_in_batch(self):
    tokens = np.asarray(
        [[1, 2, 3, self.BEGIN_IMAGE_TOKEN, 5, 6], [1, self.BEGIN_IMAGE_TOKEN, 3, 4, self.BEGIN_IMAGE_TOKEN, 6]]
    )
    num_images = 2
    new_tokens = processor_gemma3.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray(
        [
            [1, 2, 3] + self.mm_tokens + [5, 6] + [0] * (len(self.mm_tokens) - 1),
            [1] + self.mm_tokens + [3, 4] + self.mm_tokens + [6],
        ]
    )
    np.testing.assert_array_equal(new_tokens, expected)


class TestTextImageFusionGemma4(unittest.TestCase):
  """Test inserting place_holder tokens for image for Gemma 4"""

  def setUp(self):
    super().setUp()
    # Gemma 4 specific token IDs
    self.BOI_TOKEN = 255999
    self.EOI_TOKEN = 258882
    self.PLACEHOLDER_TOKEN = 258880

    # Mock sequence for testing the raw insert_sequence utility
    self.mock_sequence = [self.BOI_TOKEN, self.PLACEHOLDER_TOKEN, self.EOI_TOKEN]

  def test_insert_sequence_zero_image(self):
    tokens = np.asarray([1, 2, 3, 4, 5, 6])
    num_images = 0
    new_tokens = processor_gemma4.insert_sequence(
        at=self.PLACEHOLDER_TOKEN, sequence=self.mock_sequence, tokens=tokens, max_num_images=num_images
    )
    np.testing.assert_array_equal(new_tokens, tokens)

  def test_insert_sequence_single_image(self):
    tokens = np.asarray([1, 2, 3, self.PLACEHOLDER_TOKEN, 5, 6])
    num_images = 1
    new_tokens = processor_gemma4.insert_sequence(
        at=self.PLACEHOLDER_TOKEN, sequence=self.mock_sequence, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray([1, 2, 3] + self.mock_sequence + [5, 6])
    np.testing.assert_array_equal(new_tokens, expected)

  def test_insert_sequence_two_images(self):
    tokens = np.asarray([1, self.PLACEHOLDER_TOKEN, 3, 4, self.PLACEHOLDER_TOKEN, 6])
    num_images = 2
    new_tokens = processor_gemma4.insert_sequence(
        at=self.PLACEHOLDER_TOKEN, sequence=self.mock_sequence, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray([1] + self.mock_sequence + [3, 4] + self.mock_sequence + [6])
    np.testing.assert_array_equal(new_tokens, expected)

  def test_insert_sequence_in_batch(self):
    tokens = np.asarray(
        [[1, 2, 3, self.PLACEHOLDER_TOKEN, 5, 6], [1, self.PLACEHOLDER_TOKEN, 3, 4, self.PLACEHOLDER_TOKEN, 6]]
    )
    num_images = 2
    new_tokens = processor_gemma4.insert_sequence(
        at=self.PLACEHOLDER_TOKEN, sequence=self.mock_sequence, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray(
        [
            [1, 2, 3] + self.mock_sequence + [5, 6] + [0] * (len(self.mock_sequence) - 1),
            [1] + self.mock_sequence + [3, 4] + self.mock_sequence + [6],
        ]
    )
    np.testing.assert_array_equal(new_tokens, expected)

  def test_add_extra_tokens_for_images_gemma4_full_sequence(self):
    """Verifies that the correct number of pooled soft tokens are inserted."""
    tokens = np.asarray([1, 2, self.PLACEHOLDER_TOKEN, 4])

    # Use the actual Gemma4 function
    new_tokens = processor_gemma4.add_extra_tokens_for_images_gemma4(tokens=tokens, max_num_images=1)

    # Calculate expected soft tokens based on 672x960 image size, 16 patch size, and 3x3 pooling
    num_patches = (672 // 16) * (960 // 16)  # 2520
    num_soft_tokens = num_patches // (3**2)  # 280

    expected_sequence = [self.BOI_TOKEN] + [self.PLACEHOLDER_TOKEN] * num_soft_tokens + [self.EOI_TOKEN]
    expected_tokens = np.asarray([1, 2] + expected_sequence + [4])

    # The length should be 3 (original non-image tokens) + 282 (inserted sequence) = 285
    self.assertEqual(len(new_tokens), 285)
    np.testing.assert_array_equal(new_tokens, expected_tokens)


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

    possible_resolutions = processor_llama4.find_supported_resolutions(
        max_num_tiles=self.LLAMA4_TILES_NUM, tile_size=self.LLAMA4_TILE_SIZE
    )
    best_resolution_1 = processor_llama4.get_best_resolution(
        img_height=image_1.shape[0],
        image_width=image_1.shape[1],
        possible_resolutions=possible_resolutions,
        resize_to_max_canvas=False,
    )
    best_resolution_2 = processor_llama4.get_best_resolution(
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
    padded_image = processor_llama4.pad_to_best_fit_jax(image, best_resolution)
    self.assertEqual(padded_image.shape, (672, 672, self.NUM_IMAGE_CHANNELS))
    self.assertTrue(np.all(padded_image == 0))

  def test_split_to_tiles(self):
    image = np.ones((672, 672, self.NUM_IMAGE_CHANNELS))
    best_resolution = (672, 672)
    ratio_h, ratio_w = (
        best_resolution[0] // self.LLAMA4_TILE_SIZE,
        best_resolution[1] // self.LLAMA4_TILE_SIZE,
    )
    image_tiles = processor_llama4.split_to_tiles(image, ratio_h, ratio_w)
    self.assertEqual(
        image_tiles.shape, (ratio_h * ratio_w, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE)
    )

  def test_pad_to_max_tiles(self):
    image = np.ones((5, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE))
    padded_image, image_mask = processor_llama4.pad_to_max_tiles(image, self.LLAMA4_TILES_NUM)
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
    image_tokens = processor_llama4.get_tokens_for_this_image(this_aspect_ratio, num_patches_per_chunk)
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
        (5, mm_utils.NUM_IMAGE_CHANNELS, processor_llama4.LLAMA4_TILE_SIZE, processor_llama4.LLAMA4_TILE_SIZE)
    )
    dummy_aspect_ratios = np.array([[2, 2]])
    dummy_tokens = np.array([1, 2, self.LLAMA4_FAKE_IMAGE_TOKEN, 4, 5])
    processor_output = processor_llama4.Llama4PreprocessorOutput(
        pixel_values=dummy_pixel_values,
        aspect_ratios=dummy_aspect_ratios,
    )
    base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    config = pyconfig.initialize(
        ["", base_config_path],
        model_name="llama4-17b-16e",
        skip_jax_distributed_system=True,
    )
    image_offsets = mm_processor.get_image_offsets(config=config, processor_output=processor_output)
    post_processed_tokens = processor_llama4.add_extra_tokens_for_images_llama4(dummy_tokens, processor_output)
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
    merged = mm_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, image_masks)

    # Case 2: No image_mask, so all vision tokens are used in order.
    merged_null = mm_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, None)

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

  def test_merge_mm_embeddings_with_token_level_video_mask(self):
    """A projected-token mask packs only valid padded video embeddings."""
    text_embeddings = np.arange(1 * 8 * 2, dtype=np.float32).reshape(1, 8, 2)
    video_embeddings = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2) + 100.0
    placeholder_mask = np.zeros((1, 8), dtype=np.int32)
    placeholder_mask[:, 2:5] = 1
    video_token_mask = np.asarray([[1, 1, 1, 0, 0, 0]], dtype=np.int32)

    merged = mm_utils.merge_mm_embeddings(
        text_embeddings,
        video_embeddings,
        placeholder_mask,
        video_token_mask,
    )

    np.testing.assert_array_equal(merged[0, 2:5], video_embeddings[0, :3])
    np.testing.assert_array_equal(merged[0, 5:], text_embeddings[0, 5:])


class TestMultimodalProcessorRouting(unittest.TestCase):
  """Tests for multimodal processor prompt/response formatting and training preprocessing."""

  def test_reformat_response_multimodal_vs_text_only(self):
    # Multimodal Qwen3 model should append <|im_end|>
    self.assertEqual(mm_processor.reformat_response("Hello world", "qwen3-vl-2b"), "Hello world<|im_end|>")
    # Text-only Qwen3 model should return response unchanged
    self.assertEqual(mm_processor.reformat_response("Hello world", "qwen3-4b"), "Hello world")
    # Multimodal Gemma 3 model should append <end_of_turn>
    self.assertEqual(mm_processor.reformat_response("Hello world", "gemma3-4b"), "Hello world<end_of_turn>")
    # Multimodal Llama 4 model should append <|eot|>
    self.assertEqual(mm_processor.reformat_response("Hello world", "llama4-17b-16e"), "Hello world<|eot|>")

  def test_input_pipeline_reformat_response_string_and_list(self):
    # Test list response column (e.g. ChartQA where label = ['186600'])
    ex_list = {"label": ["186600"]}
    res_list = input_pipeline_utils.reformat_response(ex_list, "label", "maxtext-omni-gemma3-qwen3")
    self.assertEqual(res_list["label"], "186600<|im_end|>")

    # Test string response column (e.g. ChartNet where summary = 'This is a chart summary.')
    ex_str = {"summary": "This is a chart summary."}
    res_str = input_pipeline_utils.reformat_response(ex_str, "summary", "maxtext-omni-gemma3-qwen3")
    self.assertEqual(res_str["summary"], "This is a chart summary.<|im_end|>")

    # Test empty list and tuple raise ValueError
    with self.assertRaises(ValueError):
      input_pipeline_utils.reformat_response({"label": []}, "label", "maxtext-omni-gemma3-qwen3")
    # Test empty string raises ValueError
    with self.assertRaises(ValueError):
      input_pipeline_utils.reformat_response({"summary": ""}, "summary", "maxtext-omni-gemma3-qwen3")
    # Test None raises ValueError
    with self.assertRaises(ValueError):
      input_pipeline_utils.reformat_response({"label": None}, "label", "maxtext-omni-gemma3-qwen3")

  def test_reformat_prompt_text_only_fallback(self):
    # Text-only models should return prompt unchanged
    self.assertEqual(
        mm_processor.reformat_prompt("Tell me a story", "<img>", "qwen3-4b", num_images=0), "Tell me a story"
    )
    self.assertEqual(
        mm_processor.reformat_prompt("Tell me a story", "<img>", "llama2-7b", num_images=0), "Tell me a story"
    )

  def test_get_vision_and_decoder_block_routing(self):
    # pylint: disable=protected-access
    # Multimodal Qwen3-VL
    self.assertEqual(mm_processor._get_vision_block("qwen3-vl-2b"), "qwen3_vl")
    self.assertEqual(mm_processor._get_decoder_block("qwen3-vl-2b"), "qwen3")

    # Multimodal Qwen3.5
    self.assertEqual(mm_processor._get_vision_block("qwen3.5-35b-a3b"), "qwen3_5")
    self.assertEqual(mm_processor._get_decoder_block("qwen3.5-35b-a3b"), "qwen3_5")

    # Small Gemma 4 (tests gemma4_small decoder block)
    self.assertEqual(mm_processor._get_vision_block("gemma4-e4b"), "gemma4")
    self.assertEqual(mm_processor._get_decoder_block("gemma4-e4b"), "gemma4_small")

    # Text-only Qwen3-4B should have empty vision block and fallback to model_name
    self.assertIsNone(mm_processor._get_vision_block("qwen3-4b"))
    self.assertEqual(mm_processor._get_decoder_block("qwen3-4b"), "qwen3-4b")

    # Text-only Llama 2 should have empty vision block and fallback to model_name
    self.assertIsNone(mm_processor._get_vision_block("llama2-7b"))
    self.assertEqual(mm_processor._get_decoder_block("llama2-7b"), "llama2-7b")

  def test_get_vision_and_decoder_block_routing_from_config(self):
    # pylint: disable=protected-access
    # Test with pyconfig Config objects
    base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    config_qwen3_vl = pyconfig.initialize(
        ["", base_config_path],
        model_name="qwen3-vl-2b",
        scan_layers=False,
        skip_jax_distributed_system=True,
    )
    self.assertEqual(mm_processor._get_vision_block(config_qwen3_vl), "qwen3_vl")
    self.assertEqual(mm_processor._get_decoder_block(config_qwen3_vl), "qwen3")

    config_gemma3 = pyconfig.initialize(
        ["", base_config_path],
        model_name="gemma3-4b",
        skip_jax_distributed_system=True,
    )
    self.assertEqual(mm_processor._get_vision_block(config_gemma3), "gemma3")
    self.assertEqual(mm_processor._get_decoder_block(config_gemma3), "gemma3")

    config_text_only = pyconfig.initialize(
        ["", base_config_path],
        model_name="qwen3-4b",
        skip_jax_distributed_system=True,
    )
    self.assertIsNone(mm_processor._get_vision_block(config_text_only))
    self.assertEqual(mm_processor._get_decoder_block(config_text_only), "qwen3")

    # Test with config-like objects covering DecoderBlockType and VisionEncoderBlockType
    mock_config = types.SimpleNamespace(
        decoder_block=DecoderBlockType.GEMMA4_SMALL,
        vision_encoder_block=VisionEncoderBlockType.GEMMA4,
    )
    self.assertEqual(mm_processor._get_vision_block(mock_config), "gemma4")
    self.assertEqual(mm_processor._get_decoder_block(mock_config), "gemma4_small")

    # Test config with DEFAULT decoder_block and NONE vision_encoder_block
    mock_default = types.SimpleNamespace(
        decoder_block=DecoderBlockType.DEFAULT,
        vision_encoder_block=VisionEncoderBlockType.NONE,
        model_name="custom_model",
    )
    self.assertIsNone(mm_processor._get_vision_block(mock_default))
    self.assertEqual(mm_processor._get_decoder_block(mock_default), "custom_model")

  def test_get_dummy_image_shape_for_init(self):
    # Multimodal models should return non-empty dummy shape
    self.assertGreater(len(mm_processor.get_dummy_image_shape_for_init("gemma3-4b")), 0)
    self.assertGreater(len(mm_processor.get_dummy_image_shape_for_init("gemma4-26b")), 0)
    self.assertGreater(len(mm_processor.get_dummy_image_shape_for_init("llama4-17b-16e")), 0)
    self.assertGreater(len(mm_processor.get_dummy_image_shape_for_init("qwen3-vl-2b")), 0)

    # Text-only models should return empty tuple ()
    self.assertEqual(mm_processor.get_dummy_image_shape_for_init("qwen3-4b"), ())
    self.assertEqual(mm_processor.get_dummy_image_shape_for_init("llama2-7b"), ())

  def test_preprocess_image_for_training(self):
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")

    # Multimodal model should return non-empty output
    config_gemma3 = pyconfig.initialize(
        ["", base_config_path],
        model_name="gemma3-4b",
        skip_jax_distributed_system=True,
    )
    output = mm_processor.preprocess_image_for_training([dummy_image], config_gemma3)
    self.assertIsNotNone(output)

    # Text-only model should raise ValueError
    config_text_only = pyconfig.initialize(
        ["", base_config_path],
        model_name="qwen3-4b",
        skip_jax_distributed_system=True,
    )
    with self.assertRaises(ValueError):
      mm_processor.preprocess_image_for_training([dummy_image], config_text_only)

  def test_omni_gemma3_qwen3_processor_routing(self):
    # pylint: disable=protected-access,import-outside-toplevel
    self.assertEqual(mm_processor._get_vision_block("maxtext-omni-gemma3-qwen3"), "gemma3")
    self.assertEqual(mm_processor._get_decoder_block("maxtext-omni-gemma3-qwen3"), "qwen3")

    omni_config = types.SimpleNamespace(
        vision_encoder_block=VisionEncoderBlockType.GEMMA3,
        decoder_block=DecoderBlockType.QWEN3,
        model_name="maxtext-omni-gemma3-qwen3",
    )
    self.assertEqual(mm_processor.get_image_offsets(omni_config, None), 255)

    from maxtext.experimental.omni_poc.utils import processor_maxtext_omni

    qwen_image_pad = processor_maxtext_omni.DECODER_SPECIAL_TOKENS["qwen3"]["image_pad"]
    tokens = [1, qwen_image_pad, 2]
    fused = mm_processor.prepare_text_for_image_fusion(tokens, config=omni_config)
    self.assertEqual(len(fused), 1 + 256 + 1)

    mask = mm_processor.get_bidirectional_mask_vision(omni_config, np.array([[10, qwen_image_pad]]))
    np.testing.assert_array_equal(mask, [[False, True]])

    unsupported_omni = types.SimpleNamespace(
        vision_encoder_block=VisionEncoderBlockType.LLAMA4,
        decoder_block=DecoderBlockType.QWEN3,
        model_name="maxtext-omni-unsupported",
    )
    with self.assertRaises(ValueError):
      mm_processor.get_image_offsets(unsupported_omni, None)


if __name__ == "__main__":
  unittest.main()
