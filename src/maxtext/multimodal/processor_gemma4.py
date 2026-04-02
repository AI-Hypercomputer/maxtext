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


"""Gemma4-specific utilities for multimodal features."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from maxtext.multimodal import utils as mm_utils

# Constants for Gemma4-specific processing
# Using rectangular dimensions to yield exactly 2520 patches (280 tokens)
GEMMA4_IMAGE_HEIGHT = 672
GEMMA4_IMAGE_WIDTH = 960
GEMMA4_PATCH_SIZE = 16
GEMMA4_POOLING_KERNEL = 3

GEMMA4_IMAGE_PLACEHOLDER_IN_PROMPT = "<|image|>"
GEMMA4_BEGIN_IMAGE_TOKEN = 255999
GEMMA4_END_IMAGE_TOKEN = 258882
GEMMA4_NEW_LINE_TOKEN = 108
GEMMA4_TOKEN_PLACEHOLDER = 258880
# 2 extra tokens to pad around image: <begin_image>, <end_of_image>
GEMMA4_NUM_EXTRA_TOKENS_PER_MEDIA = 2


@dataclass
class Gemma4PreprocessorOutput(mm_utils.PreprocessorOutput):
  """The output of Gemma4 image preprocessor."""

  num_images: int = 0
  pixel_values: None | np.ndarray = None
  pixel_mask: None | np.ndarray = None
  positions_xy: None | np.ndarray = None


def preprocess_mm_data_gemma4(images):
  """Preprocesses multimodal data for Gemma4 models."""
  # PIL resize expects (width, height)
  target_size = (GEMMA4_IMAGE_WIDTH, GEMMA4_IMAGE_HEIGHT)

  images_in, images_out = [], []
  if isinstance(images, np.ndarray):
    images_in.append(images)
  else:
    images_in.extend(images)

  for img in images_in:
    pil_img = Image.fromarray(img)
    resized_pil_img = pil_img.resize(target_size, resample=Image.Resampling.BICUBIC)

    # Gemma 4 expects inputs strictly in the [0, 1] range.
    img = np.asarray(resized_pil_img, dtype=np.float32) / 255.0
    images_out.append(img)

  stacked_images = np.stack(images_out, axis=0).astype(np.float32)

  return Gemma4PreprocessorOutput(
      num_images=len(images_in),
      pixel_values=stacked_images[:, np.newaxis, ...],
  )


def get_image_offsets_gemma4(processor_output: mm_utils.PreprocessorOutput | None):
  """Gets the increase in total token count after inserting image token placeholders."""
  has_images = processor_output is not None and processor_output.pixel_values is not None
  num_images = processor_output.pixel_values.shape[0] if has_images else 1

  # Calculate soft tokens taking 3x3 pooling into account
  num_patches = (GEMMA4_IMAGE_HEIGHT // GEMMA4_PATCH_SIZE) * (GEMMA4_IMAGE_WIDTH // GEMMA4_PATCH_SIZE)
  num_soft_tokens = num_patches // (GEMMA4_POOLING_KERNEL**2)

  return (num_soft_tokens + GEMMA4_NUM_EXTRA_TOKENS_PER_MEDIA - 1) * num_images


def reformat_prompt_gemma4(prompt, image_placeholder, num_images):
  """Reformats prompt for Gemma4 models by inserting image placeholders."""
  if image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, GEMMA4_IMAGE_PLACEHOLDER_IN_PROMPT)
  image_placeholder_count = prompt.count(GEMMA4_IMAGE_PLACEHOLDER_IN_PROMPT)
  if image_placeholder_count < num_images:
    prompt = GEMMA4_IMAGE_PLACEHOLDER_IN_PROMPT * (num_images - image_placeholder_count) + prompt
  formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
  return formatted_prompt


def _get_new_text_positions(
    *,
    offset_on: np.ndarray,
    offset_by: int,
) -> np.ndarray:
  offset = np.cumsum(offset_on, axis=-1) * offset_by
  new_positions = np.arange(offset_on.shape[-1]) + offset
  new_positions -= offset_by * offset_on
  return new_positions


def insert_sequence(
    tokens: np.ndarray,
    *,
    at: int,
    sequence: list[int],
    max_num_images: int,
) -> np.ndarray:
  """Inserts a sequence of tokens into the given tokens array at the specified token position."""
  original_dim = tokens.ndim
  if original_dim == 1:
    tokens = tokens[None, :]

  batch_size, length = tokens.shape
  mm_tokens_to_insert = np.array(sequence)

  offset_by = len(mm_tokens_to_insert) - 1
  length_with_mm = length + max_num_images * offset_by

  # Create a boolean mask where the image trigger token `at` is present.
  mm_start = tokens == at

  new_tokens = np.zeros((batch_size, length_with_mm), dtype=np.int64)
  new_text_pos = _get_new_text_positions(offset_on=mm_start, offset_by=offset_by)

  np.put_along_axis(new_tokens, new_text_pos, tokens, axis=1)

  batch_indices_to_zero, _ = np.where(mm_start)
  new_pos_to_zero = new_text_pos[mm_start]
  if batch_indices_to_zero.size > 0:
    new_tokens[batch_indices_to_zero, new_pos_to_zero] = 0

  batch_indices, seq_indices = np.nonzero(mm_start)

  if batch_indices.size > 0:
    intra_batch_img_idx = np.cumsum(mm_start, axis=1)[mm_start] - 1
    final_img_start_pos = seq_indices + intra_batch_img_idx * offset_by

    indices_to_insert = final_img_start_pos[:, None] + np.arange(len(mm_tokens_to_insert))
    new_tokens[batch_indices[:, None], indices_to_insert] = mm_tokens_to_insert

  if original_dim == 1:
    new_tokens = np.squeeze(new_tokens)
  return new_tokens


def add_extra_tokens_for_images_gemma4(
    tokens: np.ndarray | list,
    *,
    max_num_images: int = 1,
):
  """Replaces image placeholder tokens with the full sequence of Gemma 4 image tokens."""
  # New tokens which will be inserted for each image (accounting for pooling)
  num_patches = (GEMMA4_IMAGE_HEIGHT // GEMMA4_PATCH_SIZE) * (GEMMA4_IMAGE_WIDTH // GEMMA4_PATCH_SIZE)
  num_soft_tokens = num_patches // (GEMMA4_POOLING_KERNEL**2)

  mm_tokens = [
      GEMMA4_BEGIN_IMAGE_TOKEN,
      *[GEMMA4_TOKEN_PLACEHOLDER] * num_soft_tokens,
      GEMMA4_END_IMAGE_TOKEN,
  ]
  if not isinstance(tokens, np.ndarray):
    tokens = np.asarray(tokens)
  return insert_sequence(
      at=GEMMA4_TOKEN_PLACEHOLDER,
      sequence=mm_tokens,
      tokens=tokens,
      max_num_images=max_num_images,
  )


def get_dummy_image_shape_for_init_gemma4(batch_size=1, num_image_per_sequence=1):
  """Returns the shape of the dummy image for Gemma4 model's initialization."""
  image_shape = (
      batch_size,
      num_image_per_sequence,
      GEMMA4_IMAGE_HEIGHT,
      GEMMA4_IMAGE_WIDTH,
      mm_utils.NUM_IMAGE_CHANNELS,
  )
  return image_shape
