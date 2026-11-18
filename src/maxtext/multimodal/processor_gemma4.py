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
import math

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
  num_soft_tokens_per_image: None | np.ndarray = None


def _resize_and_rescale(img: np.ndarray, height: int, width: int) -> np.ndarray:
  """Bicubic resize to (height, width), scaled to the [0, 1] range Gemma 4 expects."""
  # PIL resize expects (width, height)
  resized_pil_img = Image.fromarray(img).resize((width, height), resample=Image.Resampling.BICUBIC)
  return np.asarray(resized_pil_img, dtype=np.float32) / 255.0


def preprocess_mm_data_gemma4(images, config=None):
  """Preprocesses multimodal data for Gemma4 models."""
  if config is not None and config.use_native_resolution_for_vit:
    return preprocess_mm_data_gemma4_native(images, max_soft_tokens=config.vision_output_length)

  images_in, images_out = [], []
  if isinstance(images, np.ndarray):
    images_in.append(images)
  else:
    images_in.extend(images)
  if not images_in:
    raise ValueError("A row without an image needs use_native_resolution_for_vit=True.")

  for img in images_in:
    images_out.append(_resize_and_rescale(img, GEMMA4_IMAGE_HEIGHT, GEMMA4_IMAGE_WIDTH))

  stacked_images = np.stack(images_out, axis=0).astype(np.float32)

  return Gemma4PreprocessorOutput(
      num_images=len(images_in),
      pixel_values=stacked_images[:, np.newaxis, ...],
  )


def get_aspect_ratio_preserving_size_gemma4(height: int, width: int, max_soft_tokens: int) -> tuple[int, int]:
  """Port of HF `get_aspect_ratio_preserving_size` with the Gemma-4 patch and pooling sizes; returns (height, width)."""
  max_patches = max_soft_tokens * GEMMA4_POOLING_KERNEL**2
  side_mult = GEMMA4_PATCH_SIZE * GEMMA4_POOLING_KERNEL
  factor = math.sqrt(max_patches * GEMMA4_PATCH_SIZE**2 / (height * width))
  target_height = int(math.floor(factor * height / side_mult)) * side_mult
  target_width = int(math.floor(factor * width / side_mult)) * side_mult

  if target_height == 0 and target_width == 0:
    raise ValueError(f"Cannot resize a {height}x{width} image to a non-empty Gemma-4 patch grid.")
  # An extreme aspect ratio can round one side down to zero: keep one row or column of pooled tokens.
  max_side_length = max_soft_tokens * side_mult
  if target_height == 0:
    target_height = side_mult
    target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
  elif target_width == 0:
    target_width = side_mult
    target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
  return target_height, target_width


def preprocess_mm_data_gemma4_native(images, max_soft_tokens: int):
  """Preprocesses images like the HF Gemma-4 processor: aspect ratio kept, patches padded to the token budget.

  Returns:
    A Gemma4PreprocessorOutput with pixel_values [N, L, P*P*C], image_position_ids [N, L, 2] holding (x, y), and
    num_soft_tokens_per_image [N], where L = max_soft_tokens * GEMMA4_POOLING_KERNEL**2.
  """
  images_in = [images] if isinstance(images, np.ndarray) else list(images)
  num_images = len(images_in)
  patch = GEMMA4_PATCH_SIZE
  max_patches = max_soft_tokens * GEMMA4_POOLING_KERNEL**2

  pixel_values = np.zeros((num_images, max_patches, patch * patch * mm_utils.NUM_IMAGE_CHANNELS), dtype=np.float32)
  image_position_ids = np.full((num_images, max_patches, 2), mm_utils.IMAGE_POSITION_PAD_VALUE, dtype=np.int32)
  num_soft_tokens_per_image = np.zeros(num_images, dtype=np.int32)

  for i, img in enumerate(images_in):
    height, width = get_aspect_ratio_preserving_size_gemma4(img.shape[0], img.shape[1], max_soft_tokens)
    pixels = _resize_and_rescale(img, height, width)

    grid_h, grid_w = height // patch, width // patch
    num_patches = grid_h * grid_w
    pixel_values[i, :num_patches] = (
        pixels.reshape(grid_h, patch, grid_w, patch, -1).transpose(0, 2, 1, 3, 4).reshape(num_patches, -1)
    )
    xs, ys = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    image_position_ids[i, :num_patches] = np.stack([xs, ys], axis=-1).reshape(num_patches, 2)
    num_soft_tokens_per_image[i] = num_patches // GEMMA4_POOLING_KERNEL**2

  return Gemma4PreprocessorOutput(
      num_images=num_images,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      num_soft_tokens_per_image=num_soft_tokens_per_image,
  )


def get_image_offsets_gemma4(processor_output: mm_utils.PreprocessorOutput | None, config=None):
  """Gets the increase in total token count after inserting image token placeholders."""
  num_soft_tokens_per_image = getattr(processor_output, "num_soft_tokens_per_image", None)
  if num_soft_tokens_per_image is not None:
    return int(np.sum(np.asarray(num_soft_tokens_per_image) + GEMMA4_NUM_EXTRA_TOKENS_PER_MEDIA - 1))
  if processor_output is None and config is not None and config.use_native_resolution_for_vit:
    # One image at the full token budget.
    return config.vision_output_length + GEMMA4_NUM_EXTRA_TOKENS_PER_MEDIA - 1

  has_images = processor_output is not None and processor_output.pixel_values is not None
  num_images = processor_output.pixel_values.shape[0] if has_images else 1  # pyrefly: ignore[missing-attribute]

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
  formatted_prompt = f"<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n"
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


def _expand_image_tokens_native(tokens: np.ndarray, num_soft_tokens_per_image: np.ndarray) -> np.ndarray:
  """Replaces each placeholder in a 1D token array with its image's own soft token run."""
  if tokens.ndim != 1:
    raise ValueError(f"Native-resolution token expansion expects 1D tokens, got shape {tokens.shape}.")
  starts = np.flatnonzero(tokens == GEMMA4_TOKEN_PLACEHOLDER)
  if starts.size != len(num_soft_tokens_per_image):
    raise ValueError(
        f"Found {starts.size} image placeholder(s) in the prompt for {len(num_soft_tokens_per_image)} image(s)."
    )
  pieces = np.split(tokens, starts)
  out = [pieces[0]]
  for piece, num_soft_tokens in zip(pieces[1:], num_soft_tokens_per_image):
    image_tokens = [GEMMA4_BEGIN_IMAGE_TOKEN, *[GEMMA4_TOKEN_PLACEHOLDER] * int(num_soft_tokens), GEMMA4_END_IMAGE_TOKEN]
    out += [image_tokens, piece[1:]]
  return np.concatenate(out).astype(tokens.dtype)


def add_extra_tokens_for_images_gemma4(
    tokens: np.ndarray | list,
    *,
    max_num_images: int = 1,
    num_soft_tokens_per_image: np.ndarray | None = None,
):
  """Replaces image placeholder tokens with the full sequence of Gemma 4 image tokens."""
  if num_soft_tokens_per_image is not None:
    return _expand_image_tokens_native(np.asarray(tokens), num_soft_tokens_per_image)

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


def get_dummy_patch_shape_for_init_gemma4(config, batch_size=1):
  """Shape [B, N, L, P*P*C] of native-resolution images, as the input pipeline pads them; None when off."""
  if not config.use_native_resolution_for_vit:
    return None
  image_offset = get_image_offsets_gemma4(None, config)
  num_images = mm_utils.get_max_num_images(config.max_target_length, image_offset, config.max_num_images_per_example)
  max_patches = config.vision_output_length * GEMMA4_POOLING_KERNEL**2
  return (batch_size, num_images, max_patches, GEMMA4_PATCH_SIZE**2 * mm_utils.NUM_IMAGE_CHANNELS)
