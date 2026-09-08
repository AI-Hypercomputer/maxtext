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

"""Multimodal processor router for stitched / omni architectures in MaxText.

Dispatches preprocessing, offset calculations, token expansions, and mask generations
based on the specified vision encoder and text decoder combination.
"""

import numpy as np
from maxtext.multimodal.processor_gemma3 import GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE
from maxtext.multimodal.processor_qwen3_omni import QWEN_SPECIAL_TOKEN_CONFIGS
from maxtext.utils import max_logging

# Vision encoder image token counts
VISION_TOKENS_PER_IMAGE = {
    "gemma3": GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE,  # 256
}

# Text decoder token IDs for vision placeholders
DECODER_SPECIAL_TOKENS = {
    "qwen3": {
        "vision_start": QWEN_SPECIAL_TOKEN_CONFIGS["qwen3-omni-30b-a3b"]["vision_start"],  # 151652
        "vision_end": QWEN_SPECIAL_TOKEN_CONFIGS["qwen3-omni-30b-a3b"]["vision_end"],  # 151653
        "image_pad": QWEN_SPECIAL_TOKEN_CONFIGS["qwen3-omni-30b-a3b"]["image_pad"],  # 151655
    },
}


def get_image_offsets_omni(vision_block, decoder_block, processor_output=None):
  """Calculate the increase in total token count after inserting visual token sequences."""
  if vision_block not in VISION_TOKENS_PER_IMAGE or decoder_block not in DECODER_SPECIAL_TOKENS:
    raise ValueError(f"Stitched model not supported for vision='{vision_block}', decoder='{decoder_block}'.")

  num_tokens_per_image = VISION_TOKENS_PER_IMAGE[vision_block]
  has_images = processor_output is not None and processor_output.pixel_values is not None
  num_images = processor_output.pixel_values.shape[0] if has_images else 1
  # +num_tokens_per_image for image_pad, -1 for original placeholder
  return (num_tokens_per_image - 1) * num_images


def add_extra_tokens_for_omni(
    tokens,
    vision_block,
    decoder_block,
    processor_output=None,
    num_tokens_per_image=None,
):
  """Expands <|image_pad|> placeholders or prepends vision tokens if missing.

  - If <|image_pad|> is present in `tokens`, expands each placeholder into `num_tokens_per_image` copies.
  - If <|image_pad|> is absent but `processor_output` contains image data,
    automatically prepends the vision sequence: [<|vision_start|>, <|image_pad|>*N, <|vision_end|>].
  - Otherwise, returns the original tokens unchanged.
  """
  if vision_block not in VISION_TOKENS_PER_IMAGE or decoder_block not in DECODER_SPECIAL_TOKENS:
    raise ValueError(f"Stitched model not supported for vision='{vision_block}', decoder='{decoder_block}'.")

  if num_tokens_per_image is None:
    num_tokens_per_image = VISION_TOKENS_PER_IMAGE[vision_block]

  tokens_config = DECODER_SPECIAL_TOKENS[decoder_block]
  image_pad_id = tokens_config["image_pad"]
  vision_start_id = tokens_config["vision_start"]
  vision_end_id = tokens_config["vision_end"]

  dtype = tokens.dtype if isinstance(tokens, np.ndarray) else np.int32
  token_list = np.asarray(tokens).flatten().tolist()

  # Case 1: Prompt contains <|image_pad|> placeholders, expand them
  if image_pad_id in token_list:
    expanded_tokens = []
    for token in token_list:
      if token == image_pad_id:
        expanded_tokens.extend([image_pad_id] * num_tokens_per_image)
      else:
        expanded_tokens.append(token)
    return np.array(expanded_tokens, dtype=dtype)

  # Case 2: No placeholder in text, but image data is present, prepend vision block
  if processor_output is not None and processor_output.pixel_values is not None:
    max_logging.warning(
        "No visual placeholder (<|image_pad|>) found in prompt tokens, but image data is present. "
        "Ensure prompts are formatted via reformat_prompt to avoid token offset discrepancies."
    )
    num_images = processor_output.pixel_values.shape[0]
    vision_block_seq = []
    for _ in range(num_images):
      vision_block_seq.extend([vision_start_id] + [image_pad_id] * num_tokens_per_image + [vision_end_id])
    return np.array(vision_block_seq + token_list, dtype=dtype)

  # Case 3: Text-only sequence, return original tokens
  return np.array(token_list, dtype=dtype)


def get_bidirectional_mask_vision_omni(vision_block, decoder_block, decoder_input_tokens):
  """Generates bidirectional attention mask for vision tokens in stitched models."""
  if vision_block not in VISION_TOKENS_PER_IMAGE or decoder_block not in DECODER_SPECIAL_TOKENS:
    raise ValueError(f"Stitched model not supported for vision='{vision_block}', decoder='{decoder_block}'.")

  image_pad_id = DECODER_SPECIAL_TOKENS[decoder_block]["image_pad"]
  return decoder_input_tokens == image_pad_id
