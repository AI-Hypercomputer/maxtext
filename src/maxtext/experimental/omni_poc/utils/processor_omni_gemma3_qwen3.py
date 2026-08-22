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

"""Multimodal processor for Omni (Gemma 3 Vision Encoder + Qwen 3 LLM Decoder).

Handles Qwen 3 text-only tokenizer limitations by explicitly mapping and expanding
visual token sequences (<|vision_start|>, <|image_pad|>*256, <|vision_end|>),
where 256 is the default number of placeholder tokens per image for Gemma 3.
"""

import numpy as np
from maxtext.multimodal.processor_gemma3 import GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE
from maxtext.multimodal.processor_qwen3_omni import QWEN_SPECIAL_TOKEN_CONFIGS

# Load visual token IDs from Qwen 3 config
_QWEN_TOKENS = QWEN_SPECIAL_TOKEN_CONFIGS["qwen3-omni-30b-a3b"]
VISION_START_ID = _QWEN_TOKENS["vision_start"]  # 151652 (<|vision_start|>)
VISION_END_ID = _QWEN_TOKENS["vision_end"]  # 151653 (<|vision_end|>)
IMAGE_PAD_ID = _QWEN_TOKENS["image_pad"]  # 151655 (<|image_pad|>)
QWEN_IMAGE_TAG = "<|vision_start|><|image_pad|><|vision_end|>"

# Load vision token count per image from Gemma 3's config
DEFAULT_NUM_TOKENS_PER_IMAGE = GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE  # 256


def get_image_offsets_omni(processor_output=None):
  """Calculate the increase in total token count after inserting visual token sequences."""
  has_images = processor_output is not None and processor_output.pixel_values is not None
  num_images = processor_output.pixel_values.shape[0] if has_images else 1
  # +256 for <|image_pad|>, -1 for original placeholder (<|vision_start|> and <|vision_end|> already in prompt)
  return (DEFAULT_NUM_TOKENS_PER_IMAGE - 1) * num_images


def add_extra_tokens_for_omni(
    tokens,
    config=None,
    processor_output=None,
    num_tokens_per_image=DEFAULT_NUM_TOKENS_PER_IMAGE,
):
  """Expands <|image_pad|> placeholders or prepends vision tokens if missing.

  - If <|image_pad|> is present in `tokens`, expands each placeholder into `num_tokens_per_image` copies.
  - If <|image_pad|> is absent but `processor_output` contains image data,
    automatically prepends the vision sequence: [<|vision_start|>, <|image_pad|>*N, <|vision_end|>].
  - Otherwise, returns the original tokens unchanged.
  """
  dtype = tokens.dtype if isinstance(tokens, np.ndarray) else np.int32
  token_list = np.asarray(tokens).flatten().tolist()

  # Case 1: Prompt contains <|image_pad|> placeholders, expand them
  if IMAGE_PAD_ID in token_list:
    expanded_tokens = []
    for token in token_list:
      if token == IMAGE_PAD_ID:
        expanded_tokens.extend([IMAGE_PAD_ID] * num_tokens_per_image)
      else:
        expanded_tokens.append(token)
    return np.array(expanded_tokens, dtype=dtype)

  # Case 2: No placeholder in text, but image data is present, prepend vision block
  if processor_output is not None and processor_output.pixel_values is not None:
    num_images = processor_output.pixel_values.shape[0]
    vision_block = []
    for _ in range(num_images):
      vision_block.extend([VISION_START_ID] + [IMAGE_PAD_ID] * num_tokens_per_image + [VISION_END_ID])
    return np.array(vision_block + token_list, dtype=dtype)

  # Case 3: Text-only sequence, return original tokens
  return np.array(token_list, dtype=dtype)
