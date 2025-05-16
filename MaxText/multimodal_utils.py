"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Utils needed by multimodal pipelines for image processing."""

from typing import List
import os

import numpy as np

from PIL import Image

import jax
import jax.numpy as jnp

NUM_IMAGE_CHANNELS = 3

# Constants for Gemma3-specific processing
GEMMA_DEFAULT_IMAGE_SIZE = 896
GEMMA_IMAGE_MEAN = (127.5,) * 3
GEMMA_IMAGE_STD = (127.5,) * 3
GEMMA_BEGIN_IMAGE_TOKEN = 255999
GEMMA_END_IMAGE_TOKEN = 262144
GEMMA_NEW_LINE_TOKEN = 108
GEMMA_TOKEN_PLACEHOLDER = -2
# The number of GEMMA_TOKEN_PLACEHOLDER tokens per image in Gemma3
GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE = 256
# +4 means 4 extra tokens to pad around image: \n\n, <start_of_image>, <end_of_image>, \n\n
# One MEDIA means one image or multiple images in one video, but now we only support one image
GEMMA_NUM_TOKENS_PER_MEDIA = GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE + 4


def load_image_from_path(image_path):
  """Loads an image from a given file path and returns a jnp.array."""
  if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found at path {image_path}. Please specify a valid image path")
  try:
    image = Image.open(image_path).convert("RGB")
    image.load()  # Load image data to catch errors early
    return jnp.asarray(np.array(image))
  except (IOError, OSError) as e:
    raise IOError(f"Error loading image from {image_path}")


def _normalize_images(images, mean, std):
  """Normalize the image to zero mean and unit variance.
  Change the image mean and std based on parameters mean and std.
  Args:
    images: The images to normalize.
    mean: tuple[float, float, float].
    std: tuple[float, float, float].
  Returns:
    The normalized images.
  """
  images -= jnp.asarray(mean)
  images /= jnp.asarray(std)
  return images


def pre_process_gemma3_image(image):
  """Performs a bi-linear resize (with anti-aliasing) and normalizes the image."""
  image_shape = (GEMMA_DEFAULT_IMAGE_SIZE, GEMMA_DEFAULT_IMAGE_SIZE, NUM_IMAGE_CHANNELS)
  image = jax.image.resize(
      image,
      shape=image_shape,
      method="bilinear",
      antialias=True,
  )
  image = _normalize_images(image, mean=GEMMA_IMAGE_MEAN, std=GEMMA_IMAGE_STD)
  image = jnp.clip(image, -1, 1)
  return image


def pre_process_image(image, model_name):
  """Pre-process image according to different model's requirements.
  Args:
    image: The jnp.array image [H, W, C] to pre-process.
    model_name: The config.model_name that specifies the image preprocess ways.
  Returns:
    The pre-processed image in jnp.array [H, W, C].
  """
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    return pre_process_gemma3_image(image)
  else:
    raise ValueError(f"Model {model_name} does not support multimodal inference.")


def reformat_prompt(prompt, model_name):
  """Reformat prompt for different models."""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return formatted_prompt
  else:
    return prompt


def get_image_offsets(model_name):
  """Get the increase in total token count after inserting image token placeholders"""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    return GEMMA_NUM_TOKENS_PER_MEDIA - 1  # -1 because <start_of_image> is already present in the input tokens.
  else:
    return 0


def prepare_text_for_image_fusion(texts, model_name):
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    return add_extra_tokens_for_images_gemma3(texts)
  else:
    raise ValueError(f"Model {model_name} does not support multimodal inference.")


def add_extra_tokens_for_images_gemma3(
    tokens: np.ndarray | jnp.ndarray,
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

  return insert_sequence(
      at=GEMMA_BEGIN_IMAGE_TOKEN,
      sequence=mm_tokens,
      tokens=tokens,
      max_num_images=max_num_images,
  )


def insert_sequence(
    tokens: np.ndarray | jnp.ndarray,
    *,
    at: int,
    sequence: List[int],
    max_num_images: int,
) -> np.ndarray | jnp.ndarray:
  """Insert a sequence of tokens at a given position."""
  tokens_dim = len(tokens.shape)
  if tokens_dim == 1:
    tokens = tokens[None, :]
  _, length = tokens.shape

  mm_tokens = jnp.array(sequence, dtype=jnp.int32)

  # `-1` because `<start_of_image>` is already present in the input tokens.
  offset_by = len(mm_tokens) - 1

  # Maximum length, if all images are present.
  length_with_mm = length + max_num_images * offset_by

  mm_start = tokens == at

  # Get the text tokens correctly placed at their final position.
  # The `<start_of_image>` are removed and expanded to leave space for the MM
  # tokens.
  # tokens = [..., x, <start_of_image>, y, ...]
  # new_text_tokens = [..., x, 0, 0, 0, ..., 0, 0, 0, y, ...]
  new_text_tokens = _get_new_text_tokens(
      mm_start=mm_start,
      text_tokens=tokens,
      offset_by=offset_by,
      length_with_mm=length_with_mm,
  )

  # Get the mm tokens placeholders, correctly placed at their final position.
  # new_mm_tokens = [
  #     ..., 0, 0, \n\n, <start_of_image>, ..., <end_of_image>, \n\n, 0, 0, ...
  # ]
  new_mm_tokens = _get_new_mm_tokens(
      mm_start=mm_start,
      mm_tokens_to_insert=mm_tokens,
      max_num_images=max_num_images,
      offset_by=offset_by,
      length_with_mm=length_with_mm,
  )

  # Merge the text and MM tokens.
  new_tokens = new_text_tokens + new_mm_tokens
  if tokens_dim < len(new_tokens.shape):
    new_tokens = jnp.squeeze(new_tokens)
  return new_tokens


def _get_new_text_tokens(
    *,
    mm_start: np.ndarray | jnp.ndarray,
    text_tokens: np.ndarray | jnp.ndarray,
    offset_by: int,
    length_with_mm: int,
) -> np.ndarray | jnp.ndarray:
  """Get new text tokens."""
  # Jax vmap does not support positional arguments, so need the
  # _get_new_text_tokens_inner indirection.
  return jax.vmap(_get_new_text_tokens_inner, in_axes=(0, 0, None, None))(mm_start, text_tokens, offset_by, length_with_mm)


def _get_new_text_tokens_inner(
    mm_start: np.ndarray | jnp.ndarray,
    text_tokens: np.ndarray | jnp.ndarray,
    offset_by: int,
    length_with_mm: int,
) -> np.ndarray | jnp.ndarray:
  """`_get_new_text_tokens_positions` without batch dimension."""

  # Empty buffer in which text and MM tokens will be inserted.
  tokens_with_mm = jnp.zeros((length_with_mm,), dtype=jnp.int32)

  # Shift the original tokens, so that the new soft tokens can be inserted.
  new_text_tokens_pos = _get_new_text_tokens_positions(
      offset_on=mm_start,
      offset_by=offset_by,
  )

  tokens_with_mm = tokens_with_mm.at[new_text_tokens_pos].set(text_tokens)

  # Remove the `<start_of_image>` tokens (will be added afterwards when
  # merging with `_get_new_mm_tokens`).
  first_mm_pos = tokens_with_mm[0]
  new_start_mm_pos = new_text_tokens_pos * mm_start
  tokens_with_mm = tokens_with_mm.at[new_start_mm_pos].set(0)
  tokens_with_mm = tokens_with_mm.at[0].set(first_mm_pos)

  return tokens_with_mm


def _get_new_text_tokens_positions(
    *,
    offset_on: np.ndarray | jnp.ndarray,
    offset_by: int,
) -> np.ndarray | jnp.ndarray:
  """Create the positions of the new tokens.

  Input: `[x, x, x, offset_on, x, x, offset_on, x]`
  Output: `[0, 1, 2, 3, 4+Offset, 5+Offset, 6+Offset, 7+Offset^2]`

  Args:
    offset_on: The token to offset on.
    offset_by: The number of tokens to offset by.

  Returns:
    The new positions of the tokens.
  """
  offset = jnp.cumsum(offset_on, axis=-1) * offset_by
  new_positions = jnp.arange(offset_on.shape[-1]) + offset
  # Do not shift the `<start_of_image>` token, it will be overwritten by the MM
  # tokens.
  new_positions -= offset_by * offset_on
  return new_positions


def _get_new_mm_tokens(
    *,
    mm_start: np.ndarray | jnp.ndarray,
    mm_tokens_to_insert: np.ndarray | jnp.ndarray,
    max_num_images: int,
    offset_by: int,
    length_with_mm: int,
) -> np.ndarray | jnp.ndarray:
  """batch dimension inclusive new mm_tokens"""
  # Jax vmap does not support positional argiments, so need the
  # _get_new_mm_tokens_inner indirection.
  return jax.vmap(_get_new_mm_tokens_inner, in_axes=(0, None, None, None, None))(
      mm_start, mm_tokens_to_insert, max_num_images, offset_by, length_with_mm
  )


def _get_new_mm_tokens_inner(
    mm_start: np.ndarray | jnp.ndarray,
    mm_tokens_to_insert: np.ndarray | jnp.ndarray,
    max_num_images: int,
    offset_by: int,
    length_with_mm: int,
) -> np.ndarray | jnp.ndarray:
  """`_get_new_mm_tokens` without batch dimension."""
  # Empty buffer row, which will be merged with the final tokens.
  row = jnp.zeros((length_with_mm,), dtype=jnp.int32)

  ones = jnp.ones((len(mm_tokens_to_insert),), dtype=jnp.int32)

  (offset,) = jnp.nonzero(mm_start, size=max_num_images)

  # Because not all elements in the batch do have the same number of images
  # we need to mask out the `offset == 0` values.
  # This means that `<start_of_images>` can never be the first token, but this
  # should never happen in practice as sequences should start by `<bos>`
  mask = offset != 0
  mask = jnp.einsum("...x,y->xy", mask, ones)

  # After the mask is created, offset each individual images
  offset += jnp.arange(len(offset)) * offset_by

  new_positions = jnp.einsum("x,y->xy", offset, ones)
  new_positions += jnp.arange(len(mm_tokens_to_insert))

  new_positions = new_positions * mask

  # Because not all elements in the batch do have the same number of images
  # we need to mask out the `offset == 0` values.
  # This means that `<start_of_images>` can never be the first token, but this
  # should never happen in practice as sequences should start by `<bos>`
  row = row.at[new_positions].set(mm_tokens_to_insert)
  row = row.at[0].set(0)
  return row


def merge_mm_embeddings(
    text_embeddings: np.ndarray | jnp.ndarray,
    vision_embeddings: np.ndarray | jnp.ndarray,
    mask,
) -> np.ndarray | jnp.ndarray:
  """Merge the text and MM tokens.

  Args:
    tokens: The text tokens.
    mm_tokens: The MM tokens.

  Returns:
    The merged tokens.
  """
  return jax.vmap(_merge_mm_embeddings_inner, in_axes=(0, 0, 0))(text_embeddings, vision_embeddings, mask)


def _merge_mm_embeddings_inner(text_embeddings, vision_embeddings, mask):
  """`merge_embeddings` without batch dimension."""

  # Rearrange the vision embeddings from [num_images, num_toks_per_image, d] to [num_images * num_toks_per_image, d]
  num_images, num_toks_per_image, d = vision_embeddings.shape
  vision_embeddings = jnp.reshape(vision_embeddings, (num_images * num_toks_per_image, d))

  # len(vision_embeddings) == max_num_images * num_tokens_per_image
  target_pos = jnp.nonzero(mask, size=len(vision_embeddings))

  # Save and restore the first position overwritten if there's no MM tokens.
  first_pos = text_embeddings[0]

  merged = text_embeddings.at[target_pos, :].set(vision_embeddings)

  merged = merged.at[0].set(first_pos)

  return merged
