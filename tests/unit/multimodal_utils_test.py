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
from flax import nnx
import jax
from jax.sharding import Mesh
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import DecoderBlockType, VisionEncoderBlockType
from maxtext.configs import pyconfig
from maxtext.models.gemma4_vision import Gemma4VisionEncoderLayer
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.multimodal import processor_gemma3
from maxtext.multimodal import processor_gemma4
from maxtext.multimodal import processor_llama4
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


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

  def test_merge_image_columns_skips_missing_images(self):
    example = {"first": "a", "second": None, "third": ["b", None]}
    merged = input_pipeline_utils.merge_image_columns(example, ["first", "second", "third"], 0)
    self.assertEqual(merged["images"], ["a", "b"])
    self.assertEqual(input_pipeline_utils.merge_image_columns({"first": None}, ["first"], 2)["images"], [])

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

    # Ming has an audio decoder but no vision encoder.
    self.assertIsNone(mm_processor._get_vision_block("ming-3.0-audio-tiny"))
    self.assertEqual(mm_processor._get_decoder_block("ming-3.0-audio-tiny"), "ming")
    self.assertEqual(mm_processor.get_dummy_image_shape_for_init("ming-3.0-audio-tiny"), ())
    ming_prompt = mm_processor.reformat_prompt("transcribe", "<img>", "ming-3.0-audio-tiny", num_images=0, num_audios=1)
    self.assertLess(ming_prompt.index("<|audio_pad|>"), ming_prompt.index("transcribe"))
    self.assertEqual(mm_processor.reformat_response("done", "ming-3.0-audio-tiny"), "done<|role_end|>")

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

    from maxtext.experimental.omni_pipeline.utils import processor_maxtext_omni

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


class TestGemma4NativeResolution(unittest.TestCase):
  """Gemma-4 native-resolution preprocessing."""

  def _native_config(self, vision_output_length=280):
    return types.SimpleNamespace(
        model_name="gemma4-e2b",
        use_multimodal=True,
        vision_encoder_block=VisionEncoderBlockType.GEMMA4,
        use_native_resolution_for_vit=True,
        vision_output_length=vision_output_length,
    )

  def _native(self, *images):
    return processor_gemma4.preprocess_mm_data_gemma4(list(images), self._native_config())

  def _pad(self, output, max_length):
    pad = input_pipeline_utils.PadOrTrimToMaxLength(max_length, 0, config=self._native_config())
    return pad._pad_image_and_mask(output)  # pylint: disable=protected-access

  def _pyconfig(self, model_name="gemma4-e2b", **kwargs):
    return pyconfig.initialize(
        ["", get_test_config_path()], model_name=model_name, scan_layers=False, skip_jax_distributed_system=True, **kwargs
    )

  def test_size_matches_hf_grids(self):
    # Source (height, width) -> HF Gemma4ImageProcessor size, transformers 5.12.1.
    cases = {
        (672, 960): (672, 960),
        (536, 640): (720, 864),
        (536, 536): (768, 768),
        (160, 640): (384, 1584),
        (341, 512): (624, 960),
        (341, 191): (1056, 576),
        (48, 64): (672, 912),
        (360, 640): (576, 1056),
        (10, 100000): (48, 13440),
        (100000, 10): (13440, 48),
    }
    for source, expected in cases.items():
      size = processor_gemma4.get_aspect_ratio_preserving_size_gemma4(*source, max_soft_tokens=280)
      self.assertEqual(size, expected, msg=f"source {source}")

  def test_matches_hf_pil_processor(self):
    try:
      from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil  # pylint: disable=import-outside-toplevel
    except ImportError:
      self.skipTest("transformers without Gemma-4")
    hf = Gemma4ImageProcessorPil(patch_size=16, max_soft_tokens=280, pooling_kernel_size=3)
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, size=shape, dtype=np.uint8) for shape in ((160, 640, 3), (341, 191, 3))]
    for image in images:
      out = self._native(image)
      expected = hf([image], return_tensors="np")
      np.testing.assert_allclose(out.pixel_values, expected["pixel_values"], atol=1e-6)
      np.testing.assert_array_equal(out.image_position_ids, expected["image_position_ids"])
      self.assertEqual(out.num_soft_tokens_per_image.tolist(), expected["num_soft_tokens_per_image"].tolist())

  def test_patches_positions_and_padding(self):
    # 160x640 resizes to 384x1584: a 24x99 patch grid, 264 soft tokens.
    out = self._native(np.zeros((160, 640, 3), dtype=np.uint8))
    self.assertEqual(out.pixel_values.shape, (1, 2520, 768))
    self.assertEqual(out.num_soft_tokens_per_image.tolist(), [264])
    positions = out.image_position_ids[0]
    self.assertEqual(positions[1].tolist(), [1, 0])
    self.assertEqual(positions[99].tolist(), [0, 1])
    self.assertEqual(positions[2375].tolist(), [98, 23])
    np.testing.assert_array_equal(positions[2376:], mm_utils.IMAGE_POSITION_PAD_VALUE)

  def test_resize_keeps_the_image_layout(self):
    image = np.zeros((160, 640, 3), dtype=np.uint8)
    image[:, 320:] = 255
    out = self._native(image)
    # Mean brightness per patch on the 24x99 grid: left half dark, right half bright.
    brightness = out.pixel_values[0, :2376].mean(axis=-1).reshape(24, 99)
    self.assertTrue((brightness[:, :48] < 0.05).all())
    self.assertTrue((brightness[:, 51:] > 0.95).all())

  def test_patch_layout_is_row_major_pixels(self):
    # 672x960 is already on the grid, so patches equal the raw pixels.
    image = np.random.default_rng(1).integers(0, 255, size=(672, 960, 3), dtype=np.uint8)
    out = self._native(image)
    expected_second_patch = (image[0:16, 16:32] / 255.0).astype(np.float32).reshape(-1)
    np.testing.assert_array_equal(out.pixel_values[0, 1], expected_second_patch)

  def test_budget_comes_from_vision_output_length(self):
    out = processor_gemma4.preprocess_mm_data_gemma4(np.zeros((536, 536, 3), dtype=np.uint8), self._native_config(560))
    self.assertEqual(out.pixel_values.shape, (1, 560 * 9, 768))
    self.assertEqual(out.num_soft_tokens_per_image.tolist(), [529])  # 1104x1104: a 23x23 pooled grid

  def test_tokens_expand_per_image(self):
    placeholder = processor_gemma4.GEMMA4_TOKEN_PLACEHOLDER
    begin, end = processor_gemma4.GEMMA4_BEGIN_IMAGE_TOKEN, processor_gemma4.GEMMA4_END_IMAGE_TOKEN
    tokens = np.array([1, placeholder, 2, placeholder, 3, 0, 0], dtype=np.int32)
    out = processor_gemma4.add_extra_tokens_for_images_gemma4(tokens, num_soft_tokens_per_image=np.array([2, 3]))
    expected = [1, begin, placeholder, placeholder, end, 2, begin, placeholder, placeholder, placeholder, end, 3, 0, 0]
    self.assertEqual(out.tolist(), expected)
    self.assertEqual(out.dtype, np.int32)

  def test_placeholder_count_mismatch_raises(self):
    tokens = np.array([1, processor_gemma4.GEMMA4_TOKEN_PLACEHOLDER, 2], dtype=np.int32)
    with self.assertRaises(ValueError):
      processor_gemma4.add_extra_tokens_for_images_gemma4(tokens, num_soft_tokens_per_image=np.array([2, 3]))

  def test_text_fusion_uses_each_image_token_count(self):
    placeholder = processor_gemma4.GEMMA4_TOKEN_PLACEHOLDER
    output = self._native(np.zeros((536, 536, 3), dtype=np.uint8))
    tokens = mm_processor.prepare_text_for_image_fusion(
        np.array([1, placeholder, 2]), config=self._native_config(), processor_output=output
    )
    self.assertEqual(int((tokens == placeholder).sum()), 256)

  def test_offsets_per_image_and_largest_image(self):
    output = processor_gemma4.Gemma4PreprocessorOutput(num_soft_tokens_per_image=np.array([256, 264]))
    self.assertEqual(mm_processor.get_image_offsets(self._native_config(), output), (256 + 1) + (264 + 1))
    self.assertEqual(mm_processor.get_image_offsets(self._native_config(), None), 281)
    self.assertEqual(mm_processor.get_image_offsets(self._native_config(560), None), 561)

  def test_two_images_are_padded_per_image(self):
    out = self._pad(self._native(np.zeros((536, 536, 3), np.uint8), np.zeros((160, 640, 3), np.uint8)), 1024)
    self.assertEqual(out.pixel_values.shape, (3, 2520, 768))  # (1024 - 1) // 281 slots
    self.assertEqual(out.image_position_ids.shape, (3, 2520, 2))
    for slot, num_patches in enumerate((2304, 2376)):
      self.assertTrue((out.image_position_ids[slot, :num_patches] >= 0).all(), msg=f"slot {slot}")
      np.testing.assert_array_equal(out.image_position_ids[slot, num_patches:], mm_utils.IMAGE_POSITION_PAD_VALUE)
    np.testing.assert_array_equal(out.image_position_ids[2], mm_utils.IMAGE_POSITION_PAD_VALUE)

  def test_two_images_reach_their_own_placeholders(self):
    # 26B's encoder has no clip bounds, which stay NaN until a checkpoint is loaded.
    config = self._pyconfig(
        model_name="gemma4-26b",
        use_multimodal=True,
        use_native_resolution_for_vit=True,
        override_model_config=True,
        vision_output_length=12,
        hidden_size_for_vit=32,
        intermediate_size_for_vit=64,
        num_hidden_layers_for_vit=1,
        num_attention_heads_for_vit=2,
        attention="dot_product",
        dtype="float32",
        dtype_mm="float32",
        weight_dtype="float32",
        matmul_precision="highest",
    )
    encoder = Gemma4VisionEncoderLayer(config, Mesh(np.array(jax.devices()[:1]), ("data",)), rngs=nnx.Rngs(0))

    def encode(output):
      return encoder(output.pixel_values[None], deterministic=True, image_position_ids=output.image_position_ids[None])

    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (100, width, 3), dtype=np.uint8) for width in (100, 400)]
    output = processor_gemma4.preprocess_mm_data_gemma4(images, config)
    counts = output.num_soft_tokens_per_image.tolist()
    self.assertEqual(counts, [9, 6])
    placeholder = processor_gemma4.GEMMA4_TOKEN_PLACEHOLDER
    tokens = mm_processor.prepare_text_for_image_fusion(np.array([1, placeholder, 2, placeholder, 3]), config, output)
    mask = mm_processor.get_bidirectional_mask_vision(config, tokens[None])
    self.assertEqual(mask[0, : np.flatnonzero(tokens == 2)[0]].sum(), counts[0])

    # (40 - 1) // 13 == 3 slots, so the third one stays empty.
    pad = input_pipeline_utils.PadOrTrimToMaxLength(40, 0, config=config)
    embeddings, image_masks = encode(pad._pad_image_and_mask(output))  # pylint: disable=protected-access
    self.assertEqual(embeddings.shape[1], 3)
    text = jnp.full((1, len(tokens), embeddings.shape[-1]), 7.0)
    merged = np.asarray(mm_utils.merge_mm_embeddings(text, embeddings, mask, token_masks=image_masks))[0]

    alone = [
        np.asarray(encode(processor_gemma4.preprocess_mm_data_gemma4([image], config))[0])[0, 0, :count]
        for image, count in zip(images, counts)
    ]
    np.testing.assert_allclose(merged[mask[0]], np.concatenate(alone), rtol=1e-4, atol=1e-5, equal_nan=False)
    np.testing.assert_array_equal(merged[~mask[0]], 7.0)

  def test_batch_shape_does_not_depend_on_aspect_ratio(self):
    # Per-sample counts would give 1099 // 257 == 4 slots (square) vs 1099 // 281 == 3 (full).
    square = self._native(np.zeros((536, 536, 3), dtype=np.uint8))
    full = self._native(np.zeros((672, 960, 3), dtype=np.uint8))
    self.assertEqual(self._pad(square, 1100).pixel_values.shape, self._pad(full, 1100).pixel_values.shape)

  def test_training_preprocess_follows_the_flag(self):
    image = np.zeros((536, 536, 3), dtype=np.uint8)
    native = mm_processor.preprocess_image_for_training(image, self._native_config())
    self.assertEqual(native.num_soft_tokens_per_image.tolist(), [256])
    fixed_config = self._native_config()
    fixed_config.use_native_resolution_for_vit = False
    fixed = mm_processor.preprocess_image_for_training(image, fixed_config)
    self.assertIsNone(fixed.image_position_ids)
    self.assertEqual(fixed.pixel_values.shape, (1, 1, 672, 960, 3))

  def test_row_without_image_gets_only_empty_slots(self):
    for missing in (None, []):
      output = input_pipeline_utils.pre_process_image_sft({"images": missing}, "images", self._native_config())["images"]
      self.assertEqual(output.pixel_values.shape, (0, 2520, 768))
      tokens = mm_processor.prepare_text_for_image_fusion(np.array([1, 2, 3]), self._native_config(), output)
      self.assertEqual(tokens.tolist(), [1, 2, 3])
      padded = self._pad(output, 1024)
      self.assertEqual(padded.pixel_values.shape, (3, 2520, 768))
      np.testing.assert_array_equal(padded.image_position_ids, mm_utils.IMAGE_POSITION_PAD_VALUE)

  def test_row_without_image_needs_native_resolution(self):
    config = self._native_config()
    config.use_native_resolution_for_vit = False
    with self.assertRaisesRegex(ValueError, "use_native_resolution_for_vit"):
      input_pipeline_utils.pre_process_image_sft({"images": None}, "images", config)

  def test_native_resolution_is_on_for_e2b(self):
    self.assertTrue(self._pyconfig().use_native_resolution_for_vit)

  def test_rejected_for_other_vision_encoders(self):
    with self.assertRaisesRegex(ValueError, "use_native_resolution_for_vit requires"):
      self._pyconfig("gemma3-4b", use_multimodal=True, use_native_resolution_for_vit=True)

  def test_shaped_batch_matches_the_pipeline(self):
    config = self._pyconfig(use_multimodal=True, use_clipped_linears_for_vit=True, max_target_length=1024)
    batch = maxtext_utils.get_shaped_batch(config)
    padded = self._pad(self._native(np.zeros((536, 536, 3), dtype=np.uint8)), config.max_target_length)
    for key, value in (("images", padded.pixel_values), ("image_position_ids", padded.image_position_ids)):
      self.assertEqual(batch[key].shape[1:], value.shape, msg=key)
      self.assertEqual(batch[key].dtype, value.dtype, msg=key)


if __name__ == "__main__":
  unittest.main()
