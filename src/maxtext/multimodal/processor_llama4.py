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

"""Llama4-specific utilities for multimodal features. """

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby

import numpy as np
from PIL import Image

from MaxText.multimodal import utils as mm_utils

# Constants for Llama4-specific processing
LLAMA4_TILE_SIZE = 336
LLAMA4_TILES_NUM = 16
# Max number of tiles to pad to for Llama4 (should be >= LLAMA4_TILES_NUM + 1)
LLAMA4_TILES_PAD_TO = 20
LLAMA4_PIXEL_VALUE_RESCALE_FACTOR = 1.0 / 255.0
LLAMA4_IMAGE_MEAN = (0.5,) * 3
LLAMA4_IMAGE_STD = (0.5,) * 3
LLAMA4_PATCH_SIZE = 14
LLAMA4_IMAGE_PLACEHOLDER_IN_PROMPT = "<|image|>"
LLAMA4_FAKE_IMAGE_TOKEN = 200090  # <|image|>
LLAMA4_BEGIN_IMAGE_TOKEN = 200080  # <|image_start|>
LLAMA4_END_IMAGE_TOKEN = 200081  # <|image_end|>
LLAMA4_PATCH_TOKEN = 200092  # <|patch|>
LLAMA4_TILE_X_SEPARATOR_TOKEN = 200084  # <|tile_x_separator|>
LLAMA4_TILE_Y_SEPARATOR_TOKEN = 200085  # <|tile_y_separator|>
LLAMA4_PIXEL_SHUFFLE_RATIO = 0.5  # TODO(hengtaoguo): We should reuse config.pixel_shuffle_ratio_for_vit


@dataclass
class Llama4PreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of Llama4 image preprocessor.

  Attributes:
    Inherited from `mm_utils.PreprocessorOutput`.
  """

  # Image attributes.
  num_images: int = 0
  pixel_values: None | np.ndarray = None
  pixel_mask: None | np.ndarray = None
  aspect_ratios: None | np.ndarray = None


def get_factors(dividend: int):
  """
  Calculate all factors of a given number, i.e. a divisor that leaves
  no remainder. For example, if dividend=12, it will return {1, 2, 3, 4, 6, 12}.
  Args:
      dividend (int): The number to find factors for.
  Returns:
      set: A set containing all factors of the number.
  """
  factors_set = set()

  for i in range(1, int(dividend**0.5) + 1):
    if dividend % i == 0:
      factors_set.add(i)
      factors_set.add(dividend // i)
  return factors_set


def find_supported_resolutions(
    max_num_tiles: int = LLAMA4_TILES_NUM, tile_size: int = LLAMA4_TILE_SIZE
) -> list[tuple[int, int]]:
  """Find all possible resolutions for the image based on the number of chunks."""
  asp_dict = defaultdict(list)
  for num_tiles in range(max_num_tiles, 0, -1):
    _factors = sorted(get_factors(num_tiles))
    _asp_ratios = [(factor, num_tiles // factor) for factor in _factors]
    for height, width in _asp_ratios:
      ratio_float = height / width
      asp_dict[ratio_float].append((height, width))

  # Get the resolutions multiplied by the tile_size
  possible_resolutions = []
  for _, value in asp_dict.items():
    for height, depth in value:
      possible_resolutions.append((height * tile_size, depth * tile_size))

  return possible_resolutions


def get_best_resolution(
    img_height: int, image_width: int, possible_resolutions: list[tuple[int, int]], resize_to_max_canvas: bool = False
) -> tuple[int, int]:
  """
  Get the best resolution for the image based on the possible resolutions.
  Args:
      img_height (int): The height of the image.
      image_width (int): The width of the image.
      possible_resolutions (list): A list of possible resolutions.
      resize_to_max_canvas (bool): Whether to resize to max canvas or not.
  Returns:
      tuple: The best resolution for the image.
  """
  if resize_to_max_canvas:
    return max(possible_resolutions, key=lambda x: x[0] * x[1])
  else:
    # Find the resolution closest to the original image dimensions (minimizing padding/cropping)
    return min(possible_resolutions, key=lambda x: abs(x[0] - img_height) + abs(x[1] - image_width))


def pad_to_best_fit_jax(
    images: np.ndarray,
    target_size: tuple[int, int],
    background_color: int | tuple[int, ...] = 0,
) -> np.ndarray:
  """
  Pads and/or crops an image or batch of images to a target size using JAX.
  If the image is larger than the target size, it's cropped from the top-left.
  If smaller, it's padded on the right and bottom.

  Args:
      images (np.ndarray):
          The images to process. Expected shape (..., H, W, C).
      target_size (tuple[int, int]):
          The target (height, width).
      background_color (int | tuple[int, ...] | None):
          The color to use for padding.
          If int, it's used for the first channel and subsequent channels are padded with 0.
          If tuple, its length must match the number of channels in the image.
          Defaults to 0.

  Returns:
      np.ndarray: The processed images of shape (..., target_height, target_width, C).
  """
  original_shape = images.shape
  num_dims = len(original_shape)

  if num_dims < 3:
    raise ValueError("Images tensor must have at least 3 dimensions (..., H, W, C)")

  img_height, img_width, num_channels = original_shape[-3], original_shape[-2], original_shape[-1]
  target_height, target_width = target_size

  # Prepare background_color_array: shape (C,)
  if isinstance(background_color, int):
    # Mimics the PyTorch version's behavior: [val, 0, 0, ...]
    bg_list = [background_color] + [0] * (num_channels - 1)
    background_color_array = np.array(bg_list, dtype=images.dtype)
  elif isinstance(background_color, (tuple, list)):
    if len(background_color) != num_channels:
      raise ValueError(
          f"background_color tuple/list length {len(background_color)} " f"must match number of channels {num_channels}"
      )
    background_color_array = np.array(background_color, dtype=images.dtype)
  else:
    raise TypeError("background_color must be int or tuple/list of ints")

  # Create the full target canvas filled with background colors
  batch_dims = original_shape[:-3]
  target_canvas_shape = batch_dims + (target_height, target_width, num_channels)

  # Reshape background_color_array for broadcasting
  # e.g., for (H,W,C) -> (1,1,C); for (B,H,W,C) -> (1,1,1,C)
  broadcastable_bg_shape = tuple([1] * len(batch_dims)) + (1, 1, num_channels)
  background_fill = np.reshape(background_color_array, broadcastable_bg_shape)

  padded_output = np.ones(target_canvas_shape, dtype=images.dtype) * background_fill

  # Determine the region of the original image to copy
  h_to_copy = min(img_height, target_height)
  w_to_copy = min(img_width, target_width)

  # Create slices for selecting the part of the original image
  src_slicer_dims = []
  for _ in batch_dims:
    src_slicer_dims.append(slice(None))  # Ellipsis for batch dimensions
  src_slicer_dims.extend([slice(0, h_to_copy), slice(0, w_to_copy), slice(None)])

  image_data_to_place = images[tuple(src_slicer_dims)]

  # Create slices for placing the image data onto the canvas
  dest_slicer_dims = []
  for _ in batch_dims:
    dest_slicer_dims.append(slice(None))  # Ellipsis for batch dimensions
  dest_slicer_dims.extend([slice(0, h_to_copy), slice(0, w_to_copy), slice(None)])

  padded_output[tuple(dest_slicer_dims)] = image_data_to_place

  return padded_output


def pad_to_max_tiles(images: np.ndarray, max_num_tiles: int = LLAMA4_TILES_PAD_TO) -> tuple[np.ndarray, np.ndarray]:
  """
  Pads the image tiles to the maximum number of tiles using JAX.

  Args:
      images: The input image tiles with shape (num_tiles, C, H, W).
      max_num_tiles: The maximum number of tiles to pad to.

  Returns:
      The padded image tiles with shape (max_num_tiles, C, H, W).
      The mask indicating valid tiles with shape (max_num_tiles,).
  """
  num_tiles, num_channels, height, width = images.shape
  if num_tiles > max_num_tiles:
    raise ValueError(f"Number of tiles {num_tiles} exceeds max_num_tiles {max_num_tiles}")

  # Create a new array filled with zeros for padding
  # Note: no normalization is required for padding since there is no attention across tiles
  padded_tiles = np.zeros((max_num_tiles, num_channels, height, width), dtype=images.dtype)

  # Copy the original tiles into the new array
  padded_tiles[:num_tiles] = images

  # Create a mask indicating valid tiles in encoder input
  mask = np.zeros((max_num_tiles,), dtype=np.int32)
  mask[:num_tiles] = 1

  return padded_tiles, mask


def split_to_tiles(images: np.ndarray, num_tiles_height: int, num_tiles_width: int) -> np.ndarray:
  """
  Splits an image tensor into tiles using JAX.

  Args:
      images: The input image tensor with shape (batch_size, num_channels, height, width).
      num_tiles_height: The number of tiles along the height dimension.
      num_tiles_width: The number of tiles along the width dimension.

  Returns:
      The tiled image tensor with shape:
      (batch_size * num_tiles_height * num_tiles_width, num_channels, height // num_tiles_height, width // num_tiles_width).
  """
  images = np.transpose(images, (2, 0, 1))  # Change to (num_channels, height, width)
  num_channels, height, width = images.shape

  # Ensure the image dimensions are divisible by the number of tiles
  if height % num_tiles_height != 0 or width % num_tiles_width != 0:
    raise ValueError("Image dimensions must be divisible by the number of tiles.")

  # Reshape to introduce tile dimensions
  reshaped = np.reshape(
      images,
      (
          num_channels,
          num_tiles_height,
          height // num_tiles_height,
          num_tiles_width,
          width // num_tiles_width,
      ),
  )

  # Permute dimensions to group tiles together
  permuted = np.transpose(reshaped, (1, 3, 0, 2, 4))

  # Reshape to combine batch and tile dimensions
  tiled_images = np.reshape(
      permuted,
      (
          num_tiles_height * num_tiles_width,
          num_channels,
          height // num_tiles_height,
          width // num_tiles_width,
      ),
  )

  return tiled_images


def preprocess_mm_data_llama4(images):
  """
  Pre-process image for Llama4 model. Find best resolution and split into tiles with an additional global tile.
  Original implementation from image_processing_llama4.py: http://shortn/_VXLgQ1lmkz
  Args:
    images: The np.array image [H, W, C] or images [N, H, W, C] to pre-process.
  Returns:
    Llama4PreprocessorOutput. The pre-processed image in np.array [N, NUM_TILES, C, TILE_SIZE, TILE_SIZE].
  Example:
    image of (536, 640, 3), its best_resolution = (672, 672), image split into 4 tiles of (336, 336)
    Additional global tile of (336, 336) is added, and the final output image_tiles is (1, 5, 3, 336, 336).
  """
  images_in = []
  if isinstance(images, np.ndarray):
    images_in.append(images)
  else:
    images_in.extend(images)

  images_out, masks_out, aspect_ratios_out = [], [], []
  possible_resolutions = find_supported_resolutions(max_num_tiles=LLAMA4_TILES_NUM, tile_size=LLAMA4_TILE_SIZE)

  for img in images_in:
    # Find the best resolution canvas for the image
    best_resolution = get_best_resolution(
        img_height=img.shape[0],
        image_width=img.shape[1],
        possible_resolutions=possible_resolutions,
        resize_to_max_canvas=False,
    )

    # Pad the image to the best resolution and normalize it
    image_padded = pad_to_best_fit_jax(img, best_resolution)
    image_normalized = mm_utils.normalize_images(
        images=image_padded * LLAMA4_PIXEL_VALUE_RESCALE_FACTOR,
        mean=LLAMA4_IMAGE_MEAN,
        std=LLAMA4_IMAGE_STD,
    )

    # Split the image into tiles
    ratio_h, ratio_w = (
        best_resolution[0] // LLAMA4_TILE_SIZE,
        best_resolution[1] // LLAMA4_TILE_SIZE,
    )
    image_tiles = split_to_tiles(image_normalized, ratio_h, ratio_w)

    # If more than one tile, add a global tile by resizing the image to the tile size
    if ratio_h * ratio_w > 1:
      pil_img = Image.fromarray(img)
      resample_method = Image.Resampling.BILINEAR
      # Use a higher quality downsampling filter to approximate antialias=True
      if pil_img.size[0] > LLAMA4_TILE_SIZE or pil_img.size[1] > LLAMA4_TILE_SIZE:
        resample_method = Image.Resampling.LANCZOS
      global_tiles_pil = pil_img.resize((LLAMA4_TILE_SIZE, LLAMA4_TILE_SIZE), resample=resample_method)
      global_tiles = np.array(global_tiles_pil)
      global_tiles = mm_utils.normalize_images(
          global_tiles * LLAMA4_PIXEL_VALUE_RESCALE_FACTOR, mean=LLAMA4_IMAGE_MEAN, std=LLAMA4_IMAGE_STD
      )
      global_tiles = np.transpose(global_tiles, (2, 0, 1))
      global_tiles = np.expand_dims(global_tiles, axis=0)
      image_tiles = np.concatenate((image_tiles, global_tiles), axis=0)

    # Pad the tiles to the maximum number of tiles
    image_tiles, image_mask = pad_to_max_tiles(image_tiles, max_num_tiles=LLAMA4_TILES_PAD_TO)

    images_out.append(image_tiles)
    masks_out.append(image_mask)
    aspect_ratios_out.append([ratio_h, ratio_w])

  image_tiles = np.stack(images_out, axis=0).astype(np.float32)  # (N, NUM_TILES, C, TILE_SIZE, TILE_SIZE)
  image_mask = np.stack(masks_out, axis=0).astype(np.int32)  # (N, NUM_TILES)
  aspect_ratios_array = np.array(aspect_ratios_out, dtype=np.int32)  # (N, 2)

  processor_output = Llama4PreprocessorOutput(
      pixel_values=image_tiles,
      pixel_mask=image_mask,
      aspect_ratios=aspect_ratios_array,
      num_images=len(images),
  )
  return processor_output


def get_num_tokens_for_this_image(this_aspect_ratio, num_patches_per_chunk):
  """This function computes the length of the token sequence that would be generated by
  `get_tokens_for_this_image`, without explicit loops.

  Args:
    aspect_ratio: A tuple (ratio_h, ratio_w) representing the number of tiles
                  along height and width.
    num_patches_per_chunk: The number of patch tokens per image tile.

  Returns:
    The total number of tokens for the image representation.
  """
  ratio_h, ratio_w = this_aspect_ratio

  # Basic tokens: <|image_start|>, <|image|> (global image placeholder), <|image_end|>
  # Plus global patch tokens associated with the <|image|> placeholder.
  num_img_tokens = 3 + num_patches_per_chunk

  if ratio_h * ratio_w > 1:
    # Additional tokens for local tiles if the image is split into more than one tile:
    # - Patch tokens for each local tile: ratio_h * ratio_w * num_patches_per_chunk
    # - Separator tokens (TILE_X_SEPARATOR_TOKEN and TILE_Y_SEPARATOR_TOKEN):
    #   TILE_X_SEPARATOR_TOKEN count: ratio_h * (ratio_w - 1)
    #   TILE_Y_SEPARATOR_TOKEN count: ratio_h
    #   Total separator tokens: ratio_h * ratio_w
    num_img_tokens += ratio_h * ratio_w * (num_patches_per_chunk + 1)

  return int(num_img_tokens)


def get_image_offsets_llama4(processor_output: mm_utils.PreprocessorOutput | None):
  """Get the increase in total token count after inserting image token placeholders"""
  assert processor_output is not None, "Processor output must be provided for Llama4 image fusion."
  assert processor_output.aspect_ratios is not None, "Aspect ratio must be provided for Llama4 image fusion."
  image_height, image_width = LLAMA4_TILE_SIZE, LLAMA4_TILE_SIZE
  downsample_ratio = int(round(1.0 / (LLAMA4_PIXEL_SHUFFLE_RATIO**2)))
  num_patches_per_chunk = int(
      (image_height // LLAMA4_PATCH_SIZE) * (image_width // LLAMA4_PATCH_SIZE) // downsample_ratio
  )
  num_images = processor_output.aspect_ratios.shape[0]
  image_tokens_count = 0
  for image_index in range(num_images):
    image_tokens_count += get_num_tokens_for_this_image(
        processor_output.aspect_ratios[image_index], num_patches_per_chunk
    )
  images_offsets = image_tokens_count - num_images
  return images_offsets  # -num_images because replacing every <|image|> tokens.


def reformat_prompt_llama4(prompt, image_placeholder, num_images):
  """Reformat prompt for Llama4 model."""
  if image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, LLAMA4_IMAGE_PLACEHOLDER_IN_PROMPT)
  image_placeholder_count = prompt.count(LLAMA4_IMAGE_PLACEHOLDER_IN_PROMPT)
  if image_placeholder_count < num_images:
    prompt = LLAMA4_IMAGE_PLACEHOLDER_IN_PROMPT * (num_images - image_placeholder_count) + prompt
  formatted_prompt = (
      f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
      f"{prompt}<|eot|><|header_start|>assistant<|header_end|>\n\n"
  )
  return formatted_prompt


def get_tokens_for_this_image(this_aspect_ratio, num_patches_per_chunk):
  """Constructs the token sequence for a single image in Llama4.
  This function generates a list of special tokens that represent an image,
  including its tiled structure (if applicable) and a global representation.
  The sequence includes:
  - A beginning-of-image token.
  - Patch tokens for each local tile, interspersed with tile separators
    if the image is divided into multiple tiles (ratio_h * ratio_w > 1).
  - A fake image token placeholder for the global image representation.
  - Patch tokens associated with the global image representation.
  - An end-of-image token.

  Args:
    this_aspect_ratio: A tuple (ratio_h, ratio_w) representing the number
                       of tiles along the height and width dimensions for
                       the current image.
    num_patches_per_chunk: The number of patch tokens to use for each
                           image tile (both local and global).

  Returns:
    A list of integer token IDs representing the image.

  Example:
    If `this_aspect_ratio` is [2, 2] and `num_patches_per_chunk` is 4,
    the output will be:
    [
      LLAMA4_BEGIN_IMAGE_TOKEN,
      LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN,
      LLAMA4_TILE_X_SEPARATOR_TOKEN,
      LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN,
      LLAMA4_TILE_Y_SEPARATOR_TOKEN,
      LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN,
      LLAMA4_TILE_X_SEPARATOR_TOKEN,
      LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN,
      LLAMA4_TILE_Y_SEPARATOR_TOKEN,
      LLAMA4_FAKE_IMAGE_TOKEN,
      LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN, LLAMA4_PATCH_TOKEN,
      LLAMA4_END_IMAGE_TOKEN
    ], total 27 tokens.
  """

  img_tokens = [LLAMA4_BEGIN_IMAGE_TOKEN]
  ratio_h, ratio_w = this_aspect_ratio
  if ratio_h * ratio_w > 1:
    for _ in range(ratio_h):
      for xx in range(ratio_w):
        img_tokens += [LLAMA4_PATCH_TOKEN] * num_patches_per_chunk
        if xx < ratio_w - 1:
          img_tokens += [LLAMA4_TILE_X_SEPARATOR_TOKEN]

      img_tokens += [LLAMA4_TILE_Y_SEPARATOR_TOKEN]

  img_tokens += [LLAMA4_FAKE_IMAGE_TOKEN]
  img_tokens += [LLAMA4_PATCH_TOKEN] * num_patches_per_chunk
  img_tokens += [LLAMA4_END_IMAGE_TOKEN]

  return img_tokens


def add_extra_tokens_for_images_llama4(tokens, processor_output: mm_utils.PreprocessorOutput):
  """Add the extra image tokens to the text tokens for Llama4."""
  if not isinstance(tokens, list):
    tokens = tokens.tolist()

  grouped = groupby(tokens, lambda x: x == 200090)

  sublists = []
  for is_splitter, group in grouped:
    if not is_splitter:  # If the group does NOT consist of the split_value
      sublists.append(list(group))

  aspect_ratio = processor_output.aspect_ratios
  assert aspect_ratio is not None, "Aspect ratio must be provided for Llama4 image fusion."

  new_tokens = []

  image_height, image_width = LLAMA4_TILE_SIZE, LLAMA4_TILE_SIZE
  downsample_ratio = int(round(1.0 / (LLAMA4_PIXEL_SHUFFLE_RATIO**2)))
  num_patches_per_chunk = int(
      (image_height // LLAMA4_PATCH_SIZE) * (image_width // LLAMA4_PATCH_SIZE) // downsample_ratio
  )

  image_index = 0
  for local_image_index, split_part in enumerate(sublists):
    new_tokens += split_part  # Add the sublist
    if local_image_index < aspect_ratio.shape[0]:
      new_tokens += get_tokens_for_this_image(aspect_ratio[image_index], num_patches_per_chunk)
      image_index += 1
  new_tokens_np = np.array(new_tokens, dtype=np.int32)
  return new_tokens_np


def get_dummy_image_shape_for_init_llama4(batch_size=1, num_image_per_sequence=1):
  """Return the shape of the dummy image for Llama4 model's initialization."""
  image_shape = (
      batch_size * num_image_per_sequence,
      LLAMA4_TILES_PAD_TO,
      mm_utils.NUM_IMAGE_CHANNELS,
      LLAMA4_TILE_SIZE,
      LLAMA4_TILE_SIZE,
  )
  return image_shape
