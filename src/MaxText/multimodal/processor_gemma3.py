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

"""Gemma3-specific utilities for multimodal features. """

from dataclasses import dataclass

import numpy as np
from PIL import Image

from MaxText.multimodal import utils as mm_utils

# Constants for Gemma3-specific processing
GEMMA_DEFAULT_IMAGE_SIZE = 896
GEMMA_IMAGE_MEAN = (127.5,) * 3
GEMMA_IMAGE_STD = (127.5,) * 3
GEMMA_IMAGE_PLACEHOLDER_IN_PROMPT = "<start_of_image>"
GEMMA_BEGIN_IMAGE_TOKEN = 255999
GEMMA_END_IMAGE_TOKEN = 256000
GEMMA_NEW_LINE_TOKEN = 108
GEMMA_TOKEN_PLACEHOLDER = 262144
# The number of GEMMA_TOKEN_PLACEHOLDER tokens per image in Gemma3
GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE = 256
# +4 means 4 extra tokens to pad around image: \n\n, <start_of_image>, <end_of_image>, \n\n
# One MEDIA means one image or multiple images in one video, but now we only support one image
GEMMA_NUM_TOKENS_PER_MEDIA = GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE + 4


@dataclass
class Gemma3PreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of Gemma3 image preprocessor.

  Attributes:
    Inherited from `mm_utils.PreprocessorOutput`.
  """

  # Image attributes.
  num_images: int = 0
  pixel_values: None | np.ndarray = None
  pixel_mask: None | np.ndarray = None


def preprocess_mm_data_gemma3(images):
  """Preprocesses multimodal data for Gemma3 models."""
  # Performs a bi-linear resize (with anti-aliasing) and normalizes the image.
  target_size = (GEMMA_DEFAULT_IMAGE_SIZE, GEMMA_DEFAULT_IMAGE_SIZE)

  images_in, images_out = [], []
  if isinstance(images, np.ndarray):
    images_in.append(images)
  else:
    images_in.extend(images)

  for img in images_in:
    pil_img = Image.fromarray(img)
    resample_method = Image.Resampling.BILINEAR

    # Use a higher quality downsampling filter to approximate antialias=True
    if pil_img.size[0] > target_size[0] or pil_img.size[1] > target_size[1]:
      resample_method = Image.Resampling.LANCZOS

    resized_pil_img = pil_img.resize(target_size, resample=resample_method)
    img = np.asarray(resized_pil_img, dtype=np.float32)
    img = mm_utils.normalize_images(img, mean=GEMMA_IMAGE_MEAN, std=GEMMA_IMAGE_STD)
    img = np.clip(img, -1, 1)
    images_out.append(img)

  processor_output = Gemma3PreprocessorOutput(
      num_images=len(images),
      pixel_values=np.stack(images_out, axis=0).astype(np.float32),  # (N, H, W, C)
  )
  processor_output.num_images = len(images)
  return processor_output


def get_image_offsets_gemma3(processor_output: mm_utils.PreprocessorOutput | None):
  """Get the increase in total token count after inserting image token placeholders"""
  has_images = processor_output is not None and processor_output.pixel_values is not None
  num_images = processor_output.pixel_values.shape[0] if has_images else 1
  return (
      GEMMA_NUM_TOKENS_PER_MEDIA - 1
  ) * num_images  # -1 because <start_of_image> is already present in the input tokens.


def reformat_prompt_gemma3(prompt, image_placeholder, num_images):
  """Reformat prompt for Gemma3 models by inserting image placeholders."""
  if image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, GEMMA_IMAGE_PLACEHOLDER_IN_PROMPT)
  image_placeholder_count = prompt.count(GEMMA_IMAGE_PLACEHOLDER_IN_PROMPT)
  if image_placeholder_count < num_images:
    prompt = GEMMA_IMAGE_PLACEHOLDER_IN_PROMPT * (num_images - image_placeholder_count) + prompt
  formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
  return formatted_prompt


def _get_new_text_positions(
    *,
    offset_on: np.ndarray,
    offset_by: int,
) -> np.ndarray:
  """Create the positions of the new tokens.

  Input: `[x, x, x, offset_on, x, x, offset_on, x]`
  Output: `[0, 1, 2, 3, 4+Offset, 5+Offset, 6+Offset, 7+Offset^2]`

  Args:
    offset_on: The token to offset on.
    offset_by: The number of tokens to offset by.

  Returns:
    The new positions of the tokens.
  """
  offset = np.cumsum(offset_on, axis=-1) * offset_by
  new_positions = np.arange(offset_on.shape[-1]) + offset
  # Do not shift the `<start_of_image>` token, it will be overwritten by the MM
  # tokens.
  new_positions -= offset_by * offset_on
  return new_positions


def insert_sequence(
    tokens: np.ndarray,
    *,
    at: int,
    sequence: list[int],
    max_num_images: int,
) -> np.ndarray:
  """
  Inserts a sequence of tokens at all occurrences of a specific token `at`.
  This function is fully vectorized and operates on a batch of token sequences.

  Args:
      tokens: A 1D or 2D array of input tokens.
      at: The token ID to find and replace with the sequence.
      sequence: The list of new token IDs to insert.
      max_num_images: The maximum number of times `at` can appear.

  Returns:
      The modified token array with the sequences inserted.
  """
  # Ensure input is a 2D array (batch)
  original_dim = tokens.ndim
  if original_dim == 1:
    tokens = tokens[None, :]

  batch_size, length = tokens.shape
  mm_tokens_to_insert = np.array(sequence)

  # Net number of tokens added for each image placeholder.
  # It's -1 because the original '<begin_image>' token is replaced.
  offset_by = len(mm_tokens_to_insert) - 1
  length_with_mm = length + max_num_images * offset_by

  # Create a boolean mask where the image trigger token `at` is present.
  mm_start = tokens == at

  # 1. Create a new buffer for the final merged tokens.
  # This buffer will hold the text tokens in their new, shifted positions.
  new_tokens = np.zeros((batch_size, length_with_mm), dtype=np.int64)

  # Calculate the new, shifted positions for all original text tokens.
  new_text_pos = _get_new_text_positions(offset_on=mm_start, offset_by=offset_by)

  # Place the original tokens into their new positions.
  # `np.put_along_axis` is the NumPy equivalent of the JAX scatter operation.
  np.put_along_axis(new_tokens, new_text_pos, tokens, axis=1)

  # Zero out the placeholder for the `<begin_image>` token at its new position, which we will
  # overwrite with the full image sequence next.
  # We find where `mm_start` is True and use the corresponding new positions
  # to index `new_tokens` and set those locations to 0.
  batch_indices_to_zero, _ = np.where(mm_start)
  new_pos_to_zero = new_text_pos[mm_start]
  if batch_indices_to_zero.size > 0:
    new_tokens[batch_indices_to_zero, new_pos_to_zero] = 0

  # 2. Now, insert the actual image token sequences.
  # Find the row and column indices of all image trigger tokens.
  batch_indices, seq_indices = np.nonzero(mm_start)

  if batch_indices.size > 0:
    # Calculate the index of each image within its sequence (0th, 1st, etc.).
    intra_batch_img_idx = np.cumsum(mm_start, axis=1)[mm_start] - 1

    # Calculate the final start position for each new image sequence,
    # accounting for shifts from previous images in the same row.
    final_img_start_pos = seq_indices + intra_batch_img_idx * offset_by

    # Create the full index grid for placing all new tokens.
    # This uses broadcasting to add the start position of each image sequence
    # to a range of offsets [0, 1, ..., N] for the tokens within the sequence.
    indices_to_insert = final_img_start_pos[:, None] + np.arange(len(mm_tokens_to_insert))

    # Use the calculated indices to place the new tokens.
    # We use `batch_indices` to specify the row and `indices_to_insert` for columns.
    new_tokens[batch_indices[:, None], indices_to_insert] = mm_tokens_to_insert

  if original_dim == 1:
    new_tokens = np.squeeze(new_tokens)
  return new_tokens


def add_extra_tokens_for_images_gemma3(
    tokens: np.ndarray | list,
    *,
    max_num_images: int = 1,
):  # -> Int['B L+(max_num_images * (num_tokens_per_image + 3))']:
  r"""Add the extra image tokens to the text tokens.

  If the model has images, we expand each `<start_of_image>` token by the image
  placeholder tokens.

  Example:

  ```python
  input = [..., x, <start_of_image>, y, ...]
  output = [
      ..., x, \n\n, <start_of_image>, SOFT_TOKEN_PLACEHOLDER,
      SOFT_TOKEN_PLACEHOLDER, ..., SOFT_TOKEN_PLACEHOLDER,
      SOFT_TOKEN_PLACEHOLDER, <end_of_image>, \n\n, y, ...
  ]
  ```

  The `\n\n` tokens are added to match how the model was trained.

  Args:
    tokens: The text tokens.
    max_num_images: The maximum number of images in the batch.
    num_tokens_per_image: The number of soft tokens per image.

  Returns:
    The text tokens with the extra image tokens.
  """

  # New tokens which will be inserted for each image.
  mm_tokens = [
      GEMMA_NEW_LINE_TOKEN,
      GEMMA_BEGIN_IMAGE_TOKEN,
      *[GEMMA_TOKEN_PLACEHOLDER] * GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE,
      GEMMA_END_IMAGE_TOKEN,
      GEMMA_NEW_LINE_TOKEN,
  ]
  if not isinstance(tokens, np.ndarray):
    tokens = np.asarray(tokens)
  return insert_sequence(
      at=GEMMA_BEGIN_IMAGE_TOKEN,
      sequence=mm_tokens,
      tokens=tokens,
      max_num_images=max_num_images,
  )


def get_dummy_image_shape_for_init_gemma3(batch_size=1, num_image_per_sequence=1):
  """Return the shape of the dummy image for Gemma3 model's initialization."""
  image_shape = (
      batch_size,
      num_image_per_sequence,
      GEMMA_DEFAULT_IMAGE_SIZE,
      GEMMA_DEFAULT_IMAGE_SIZE,
      mm_utils.NUM_IMAGE_CHANNELS,
  )
  return image_shape
