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

"""Utils needed by multimodal pipelines for image processing."""

from dataclasses import dataclass
from collections import defaultdict
from itertools import groupby
import os

import numpy as np

from PIL import Image

import jax
import jax.numpy as jnp

NUM_IMAGE_CHANNELS = 3

# TODO(hengtaoguo): Move following constants to a separate file
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

# Qwen3OmniMoe-specific processing
QWEN3_OMNI_VISION_START_TOKEN = 151652  # <|vision_start|>
QWEN3_OMNI_IMAGE_TOKEN = 151655  # <|image_pad|>
QWEN3_OMNI_VIDEO_TOKEN = 151656  # <|video_pad|>
QWEN3_OMNI_AUDIO_START_TOKEN = 151669  # <|audio_start|>
QWEN3_OMNI_AUDIO_TOKEN = 151675  # <|audio_pad|>
QWEN3_TEMPORAL_PATCH_SIZE = 2
QWEN3_OMNI_IMAGE_SIZE = 768


@dataclass
class PreprocessorOutput:
  """Holds the output of an image preprocessor.

  Attributes:
    pixel_values: A JAX array containing the processed image pixel data.
                  The shape and format depend on the specific model and
                  preprocessing steps (e.g., [H, W, C] for Gemma3 or
                  [NUM_TILES, C, TILE_SIZE, TILE_SIZE] for Llama4).
    pixel_mask: An optional JAX array of shape (NUM_TILES,) indicating valid
                tiles in the image.
    aspect_ratios: An optional JAX array of shape (batch_size, 2) representing
                   the aspect ratio [ratio_h, ratio_w] of the processed image(s).
                   This is particularly relevant for models like Llama4 that handle
                   images by tiling.
  """

  pixel_values: None | np.ndarray = None
  pixel_mask: None | np.ndarray = None
  aspect_ratios: None | np.ndarray = None


def convert_to_RGB(image):
  if image.mode != "RGB":
    image = image.convert("RGB")
  return image


def load_image_from_path(image_path):
  """Loads an image from a given file path and returns a np.array."""
  if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found at path {image_path}. Please specify a valid image path")
  try:
    image = Image.open(image_path).convert("RGB")
    image.load()  # Load image data to catch errors early
    return np.asarray(image)

  except (IOError, OSError) as e:
    raise IOError(f"Error loading image from {image_path}") from e


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
  images -= np.asarray(mean)
  images /= np.asarray(std)
  return images


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


def pre_process_gemma3_image(image: np.ndarray | list[np.ndarray]) -> PreprocessorOutput:
  """Performs a bi-linear resize (with anti-aliasing) and normalizes the image."""
  target_size = (GEMMA_DEFAULT_IMAGE_SIZE, GEMMA_DEFAULT_IMAGE_SIZE)

  images_in, images_out = [], []
  if isinstance(image, np.ndarray):
    images_in.append(image)
  else:
    images_in.extend(image)

  for img in images_in:
    pil_img = Image.fromarray(img)
    resample_method = Image.Resampling.BILINEAR

    # Use a higher quality downsampling filter to approximate antialias=True
    if pil_img.size[0] > target_size[0] or pil_img.size[1] > target_size[1]:
      resample_method = Image.Resampling.LANCZOS

    resized_pil_img = pil_img.resize(target_size, resample=resample_method)
    img = np.asarray(resized_pil_img, dtype=np.float32)
    img = _normalize_images(img, mean=GEMMA_IMAGE_MEAN, std=GEMMA_IMAGE_STD)
    img = np.clip(img, -1, 1)
    images_out.append(img)

  processor_output = PreprocessorOutput(
      pixel_values=np.stack(images_out, axis=0).astype(np.float32),  # (N, H, W, C)
  )
  return processor_output


def pre_process_llama4_image(image: np.ndarray | list[np.ndarray]) -> PreprocessorOutput:
  """
  Pre-process image for Llama4 model. Find best resolution and split into tiles with an additional global tile.
  Original implementation from image_processing_llama4.py: http://shortn/_VXLgQ1lmkz
  Args:
    image: The np.array image [H, W, C] or images [N, H, W, C] to pre-process.
  Returns:
    The pre-processed image in np.array [N, NUM_TILES, C, TILE_SIZE, TILE_SIZE].
  Example:
    image of (536, 640, 3), its best_resolution = (672, 672), image split into 4 tiles of (336, 336)
    Additional global tile of (336, 336) is added, and the final output image_tiles is (1, 5, 3, 336, 336).
  """
  images_in = []
  if isinstance(image, np.ndarray):
    images_in.append(image)
  else:
    images_in.extend(image)

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
    image_normalized = _normalize_images(
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
      global_tiles = _normalize_images(
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

  processor_output = PreprocessorOutput(
      pixel_values=image_tiles,
      pixel_mask=image_mask,
      aspect_ratios=aspect_ratios_array,
  )
  return processor_output


def pre_process_image(image, model_name):
  """Pre-process image according to different model's requirements.
  Args:
    image: The np.array image [H, W, C] or images [N, H, W, C] to pre-process.
    model_name: The config.model_name that specifies the image preprocess ways.
  Returns:
    The PreprocessorOutput instance containing image in np.array [H, W, C] or [N, H, W, C].
  """
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    return pre_process_gemma3_image(image)
  elif model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    return pre_process_llama4_image(image)
  else:
    raise ValueError(f"Model {model_name} does not support multimodal inference.")


def reformat_prompt(prompt, image_placeholder, model_name, num_images):
  """Reformat prompt for different models."""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    if image_placeholder in prompt:
      prompt = prompt.replace(image_placeholder, GEMMA_IMAGE_PLACEHOLDER_IN_PROMPT)
    image_placeholder_count = prompt.count(GEMMA_IMAGE_PLACEHOLDER_IN_PROMPT)
    if image_placeholder_count < num_images:
      prompt = GEMMA_IMAGE_PLACEHOLDER_IN_PROMPT * (num_images - image_placeholder_count) + prompt
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return formatted_prompt
  elif model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
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
  else:
    return prompt


def reformat_response(response, model_name):
  """Reformat response for different models."""
  if model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    formatted_response = f"{response}<|eot|>"
    return formatted_response
  elif model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    formatted_response = f"{response}<end_of_turn>"
    return formatted_response
  else:
    return response


def get_image_offsets(model_name, processor_output: PreprocessorOutput | None):
  """Get the increase in total token count after inserting image token placeholders"""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    has_images = processor_output is not None and processor_output.pixel_values is not None
    num_images = processor_output.pixel_values.shape[0] if has_images else 1
    return (
        GEMMA_NUM_TOKENS_PER_MEDIA - 1
    ) * num_images  # -1 because <start_of_image> is already present in the input tokens.
  if model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
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
  else:
    return 0


def get_dummy_image_shape_for_init(
    model_name, batch_size=1, num_image_per_sequence=1, num_tiles_per_image=LLAMA4_TILES_PAD_TO
):
  """Return the shape of the dummy image for specific model's initialization."""
  image_shape = ()
  if model_name.startswith("gemma3"):
    image_shape = (
        batch_size,
        num_image_per_sequence,
        GEMMA_DEFAULT_IMAGE_SIZE,
        GEMMA_DEFAULT_IMAGE_SIZE,
        NUM_IMAGE_CHANNELS,
    )
  elif model_name.startswith("llama4"):
    image_shape = (
        batch_size * num_image_per_sequence,
        num_tiles_per_image,
        NUM_IMAGE_CHANNELS,
        LLAMA4_TILE_SIZE,
        LLAMA4_TILE_SIZE,
    )
  elif model_name.startswith("qwen3-omni-30b-a3b"):
    image_shape = (
        batch_size,
        NUM_IMAGE_CHANNELS,
        QWEN3_TEMPORAL_PATCH_SIZE,
        QWEN3_OMNI_IMAGE_SIZE,  # image_size_for_vit (height)
        QWEN3_OMNI_IMAGE_SIZE,  # video_num_frames
    )
  return image_shape


def get_dummy_audio_shape_for_init(model_name, config, batch_size=1):
  """Return the shape of the dummy audio for specific model's initialization.

  Args:
    model_name: Name of the model
    config: Model configuration containing audio parameters
    batch_size: Batch size for the dummy audio

  Returns:
    Tuple representing audio shape: (batch, num_mel_bins, audio_length)
    Returns None if audio is not configured
  """
  audio_shape = None
  if model_name.startswith("qwen3-omni"):
    # Audio shape: (batch, num_mel_bins, audio_length)
    # audio_length must be divisible by n_window * 2
    chunk_size = config.n_window_for_audio * 2
    audio_length = chunk_size * 4  # 4 chunks
    audio_shape = (batch_size, config.num_mel_bins_for_audio, audio_length)

  return audio_shape


def prepare_text_for_image_fusion(texts, model_name, processor_output=None):
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    num_images = processor_output.pixel_values.shape[0] if processor_output else 1
    return add_extra_tokens_for_images_gemma3(texts, max_num_images=num_images)
  if model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    return add_extra_tokens_for_images_llama4(texts, processor_output)
  else:
    raise ValueError(f"Model {model_name} does not support multimodal inference.")


def add_extra_tokens_for_images_llama4(tokens, processor_output: PreprocessorOutput):
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


def merge_mm_embeddings(
    text_embeddings: np.ndarray | jnp.ndarray,
    multimodal_embeddings: np.ndarray | jnp.ndarray,
    mask,
    token_masks: np.ndarray | jnp.ndarray | None = None,
) -> np.ndarray | jnp.ndarray:
  """Merges text and multimodal (vision/audio) embeddings based on a mask.

  This function handles two primary formats for multimodal embeddings:
  1. Tiled Format (e.g., Llama4 vision): Embeddings are provided as a batch of
     images and their tiles, with shape (B * N, T, K, D). These are flattened
     into a single sequence of tokens per batch item.
  2. Simple Format (e.g., Gemma3 vision, Qwen3-Omni audio): Embeddings are provided as
     (B, N, K, D) and are flattened into a sequence of tokens.

  Args:
    text_embeddings: (B, S, D) array of text embeddings.
    multimodal_embeddings: Multimodal embeddings (vision/audio) in one of two formats:
      - (B * N, T, K, D) for tiled inputs.
      - (B, N, K, D) for simple inputs.
      (B=batch_size, S=seq_len, D=embedding_dim, N=num_images/audio_chunks,
       T=num_tiles, K=toks_per_image/audio_chunk)
    mask: (B, S) boolean or integer array where non-zero positions
      indicate where multimodal embeddings should be placed.
    token_masks: (Optional) A mask for the multimodal tokens.
      - (B * N, T) for tiled inputs, indicating valid tiles.
      - If None, all multimodal embeddings are assumed to be valid.

  Returns:
    A (B, S, D) array of merged embeddings.
  """
  # Input Validation and Shape Unpacking
  batch_size, _, d_model = text_embeddings.shape
  # The number of tokens per image/tile/audio_chunk is the second to last dimension.
  num_toks_per_token = multimodal_embeddings.shape[-2]

  if d_model != multimodal_embeddings.shape[-1]:
    raise ValueError(
        "Embedding dimension mismatch between text and multimodal embeddings:"
        f" {d_model} vs {multimodal_embeddings.shape[-1]}"
    )

  # Reshape Multimodal Embeddings to a unified (B, S_mm, D) format
  # This single reshape robustly handles both documented cases:
  # Case 1: (B * N, T, K, D) -> (B, N*T*K, D)
  # Case 2: (B, N, K, D) -> (B, N*K, D)
  flat_multimodal_embeddings = multimodal_embeddings.reshape(batch_size, -1, d_model)

  # Process Optional Token Masks
  flat_token_masks_processed = None
  if token_masks is not None:
    # Handle the tiled case where token_masks batch dimension is (B * N)
    if token_masks.shape[0] != batch_size:
      if token_masks.shape[0] % batch_size != 0:
        raise ValueError(
            "Batch dimension of token_masks must be a multiple of the text"
            f" batch size. Got {token_masks.shape[0]} and {batch_size}."
        )
      # Reshape from (B * N, T) to (B, N * T)
      flat_tile_masks = token_masks.reshape(batch_size, -1)
    else:
      # This handles cases where token_masks is already (B, ...)
      flat_tile_masks = token_masks.reshape(batch_size, -1)

    # Expand the tile-level mask to a token-level mask to match the embeddings.
    # A mask of shape (B, N*T) becomes (B, N*T*K) by repeating each element K times.
    flat_token_masks_processed = jnp.repeat(flat_tile_masks, repeats=num_toks_per_token, axis=1)

  # Vmap the inner merge function over the batch dimension
  return jax.vmap(
      _merge_mm_embeddings_inner,  # Assumes this function is defined elsewhere
      in_axes=(0, 0, 0, None if flat_token_masks_processed is None else 0),
  )(text_embeddings, flat_multimodal_embeddings, mask, flat_token_masks_processed)


def _merge_mm_embeddings_inner(
    text_embeddings: jnp.ndarray,
    multimodal_embeddings: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    token_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """`merge_mm_embeddings` without batch dimension."""

  if token_mask is not None:
    # This logic packs valid multimodal tokens to the front of the array.
    # It correctly handles cases where some multimodal tokens are just padding.
    sort_indices = jnp.argsort(-token_mask)  # Sorts descending, putting 1s first
    multimodal_embeddings = multimodal_embeddings[sort_indices]

  # Find positions in the text sequence to place the multimodal embeddings.
  # The `size` argument ensures a fixed shape for JIT compilation.
  target_pos = jnp.nonzero(mask, size=multimodal_embeddings.shape[0])
  target_pos = target_pos[0]  # jnp.nonzero returns a tuple of arrays

  # Save the embedding at the first position.
  first_pos_embedding = text_embeddings[0]

  # Perform the insertion.
  merged = text_embeddings.at[target_pos, :].set(multimodal_embeddings)

  # Restore the first position's embedding, in case it was overwritten.
  merged = merged.at[0].set(first_pos_embedding)

  return merged


# ==============================================================================
# Qwen3-Omni Multimodal Position ID Utilities
# ==============================================================================
def _get_feat_extract_output_lengths(input_lengths: jax.Array) -> jax.Array:
  """Computes the output length of the audio encoder convolutional layers.

  The audio encoder processes audio in chunks of 100 samples, applying
  multiple convolutional layers with stride 2. Each full 100-sample chunk
  produces 13 output tokens, and remaining samples are processed separately.

  Args:
    input_lengths: Input audio sequence lengths. Shape: (batch,) or scalar.

  Returns:
    Output sequence lengths after audio encoding. Same shape as input.
  """
  input_lengths_leave = input_lengths % 100
  feat_lengths = (input_lengths_leave - 1) // 2 + 1
  output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
  return output_lengths


def get_llm_pos_ids_for_vision(
    start_idx: int | jax.Array,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: jax.Array,
    grid_hs: jax.Array,
    grid_ws: jax.Array,
) -> jax.Array:
  """Computes 3D position IDs (temporal, height, width) for vision tokens.

  Creates position embeddings for a grid of vision tokens representing an
  image or video. For each temporal frame, generates a spatial grid of
  (height, width) positions.

  Args:
    start_idx: Starting position ID value to add as offset.
    vision_idx: Index of the current image/video being processed.
    spatial_merge_size: Number of patches merged spatially (e.g., 2 means 2x2 patches → 1 token).
    t_index: Temporal position for each frame. Shape: (num_frames,).
    grid_hs: Height dimensions for all images/videos. Shape: (num_images,).
    grid_ws: Width dimensions for all images/videos. Shape: (num_images,).

  Returns:
    3D position IDs with shape (3, num_vision_tokens) where:
      - dim 0: temporal positions
      - dim 1: height positions
      - dim 2: width positions

  Example:
    If spatial_merge_size=2, grid_h=4, grid_w=4, num_frames=2:
      - After merge: 2x2 grid per frame
      - Total tokens: 2 frames x 2 x 2 = 8 tokens
      - Output shape: (3, 8)
      - t_index: [0, 0, 0, 0, 50, 50, 50, 50]
      - h_index: [0, 0, 1, 1, 0, 0, 1, 1]
      - w_index: [0, 1, 0, 1, 0, 1, 0, 1]
  """
  # Calculate effective spatial dimensions after merging patches
  llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
  llm_grid_w = grid_ws[vision_idx] // spatial_merge_size

  # Create height indices: [0, 0, ..., 0 (W times), 1, 1, ..., 1 (W times), ...]
  # Shape: (num_frames, llm_grid_h, 1) -> expand -> (num_frames, llm_grid_h, llm_grid_w) -> flatten
  h_index = jnp.arange(llm_grid_h).reshape(1, -1, 1).repeat(len(t_index), axis=0).repeat(llm_grid_w, axis=2).flatten()

  # Create width indices: [0, 1, 2, ..., W-1, 0, 1, 2, ..., W-1, ...]
  # Shape: (num_frames, 1, llm_grid_w) -> expand -> (num_frames, llm_grid_h, llm_grid_w) -> flatten
  w_index = jnp.arange(llm_grid_w).reshape(1, 1, -1).repeat(len(t_index), axis=0).repeat(llm_grid_h, axis=1).flatten()

  # Create temporal indices: [t0, t0, ..., t0 (HxW times), t1, t1, ..., t1 (HxW times), ...]
  # Shape: (num_frames, 1) -> expand -> (num_frames, llm_grid_h * llm_grid_w) -> flatten
  t_index_expanded = t_index.reshape(-1, 1).repeat(llm_grid_h * llm_grid_w, axis=1).flatten()

  # Stack all three dimensions and add starting offset
  _llm_pos_ids = jnp.stack([t_index_expanded, h_index, w_index])
  llm_pos_ids = _llm_pos_ids + start_idx

  return llm_pos_ids


def get_chunked_index(token_indices: jax.Array, tokens_per_chunk: int, remove_index: int) -> list[tuple[int, int]]:
  """Splits token index list into chunks based on token value ranges.

  Given a list of monotonically increasing token indices, returns a list of
  (start, end) index tuples representing slices where token values fall within
  successive ranges of `tokens_per_chunk`.

  Args:
    token_indices: Monotonically increasing array of token index values. Shape: (seq_len,).
    tokens_per_chunk: Chunk size threshold (e.g., 100 means first chunk has values < 100).
    remove_index: Offset to subtract from token_indices before chunking.

  Returns:
    List of (start_idx, end_idx) tuples, each representing a chunk.

  Example:
    token_indices = [5, 10, 52, 105, 150, 250]
    tokens_per_chunk = 100
    remove_index = 0

    Result: [(0, 3), (3, 5), (5, 6)]
      - Chunk 0: indices 0-3 (values 5, 10, 52 are < 100)
      - Chunk 1: indices 3-5 (values 105, 150 are >= 100 and < 200)
      - Chunk 2: indices 5-6 (value 250 is >= 200)
  """
  chunks = []
  i = 0
  start_idx = 0
  current_chunk = 1

  while i < len(token_indices):
    if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
      chunks.append((start_idx, i))
      start_idx = i
      current_chunk += 1
    i += 1

  # Append final chunk
  chunks.append((start_idx, len(token_indices)))

  return chunks


def get_rope_index(
    input_ids: jax.Array,
    image_grid_thw: jax.Array | None = None,
    video_grid_thw: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    use_audio_in_video: bool = False,
    audio_lengths: jax.Array | None = None,
    second_per_grids: jax.Array | None = None,
    spatial_merge_size: int = 2,
    position_id_per_seconds: int = 25,
) -> tuple[jax.Array, jax.Array]:
  """Calculate 3D RoPE position indices for multimodal sequences.

  This function computes position IDs that encode both sequential (text) and
  spatial-temporal (vision/audio) structure for Qwen3-Omni multimodal inputs.

  For pure text sequences:
    - All 3 dimensions receive the same sequential positions: [0, 1, 2, 3, 4]

  For multimodal sequences with vision:
    - Vision tokens get 3D positions (temporal, height, width)
    - Text tokens continue sequentially from max(vision_pos) + 1
    - Example with video (3 temporal patches, 2x2 spatial):
        Vision temporal: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
        Vision height:   [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        Vision width:    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        Text positions:  [101, 102, 103, 104, 105]

  Args:
    input_ids: Input token IDs. Shape: (batch, seq_len).
    image_grid_thw: Image dimensions (temporal, height, width). Shape: (num_images, 3).
    video_grid_thw: Video dimensions (temporal, height, width). Shape: (num_videos, 3).
    attention_mask: Padding mask (1 = real token, 0 = padding). Shape: (batch, seq_len).
    use_audio_in_video: If True, audio tokens are interleaved with video tokens.
    audio_lengths: Audio sequence lengths. Shape: (num_audios,).
    second_per_grids: Time interval per temporal grid (for videos). Shape: (num_videos,).
    spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
    position_id_per_seconds: Temporal granularity (tokens per second, typically 25).

  Returns:
    A tuple of:
      - position_ids: 3D position IDs. Shape: (3, batch, seq_len).
      - mrope_position_deltas: Position offset for each sequence. Shape: (batch, 1).

  Raises:
    ValueError: If multimodal tokens are present but grid info is missing.
  """
  batch_size, seq_len = input_ids.shape

  # Handle text-only case (no multimodal content)
  if image_grid_thw is None and video_grid_thw is None:
    if attention_mask is None:
      attention_mask = jnp.ones_like(input_ids)

    # Create sequential 1D positions
    position_ids = jnp.cumsum(attention_mask.astype(jnp.float32), axis=-1) - 1
    position_ids = jnp.where(attention_mask == 0, 1.0, position_ids)

    # Expand to 3D (same value in all dimensions for text-only)
    position_ids = jnp.broadcast_to(position_ids[jnp.newaxis, :, :], (3, batch_size, seq_len))

    # Calculate deltas for each sequence
    max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True).transpose(1, 0, 2)  # (batch, 1, 1)
    mrope_position_deltas = max_position_ids.squeeze(-1) + 1 - jnp.sum(attention_mask, axis=-1, keepdims=True)

    return position_ids, mrope_position_deltas

  # Multimodal case: process each sequence in batch
  if attention_mask is None:
    attention_mask = jnp.ones_like(input_ids)

  attention_mask_bool = attention_mask == 1
  position_ids = jnp.zeros((3, batch_size, seq_len), dtype=jnp.float32)
  mrope_position_deltas = []

  # Process each sequence in the batch
  for i in range(batch_size):
    # Get valid tokens (non-padding)
    valid_input_ids = input_ids[i][attention_mask_bool[i]]

    # Count multimodal elements in this sequence
    vision_start_indices = jnp.where(valid_input_ids == QWEN3_OMNI_VISION_START_TOKEN)[0]
    vision_tokens = valid_input_ids[vision_start_indices + 1] if len(vision_start_indices) > 0 else jnp.array([])

    audio_nums = jnp.sum(valid_input_ids == QWEN3_OMNI_AUDIO_START_TOKEN).item()
    image_nums = jnp.sum(vision_tokens == QWEN3_OMNI_IMAGE_TOKEN).item() if len(vision_tokens) > 0 else 0
    video_nums = (
        (
            jnp.sum(vision_tokens == QWEN3_OMNI_AUDIO_START_TOKEN).item()
            if use_audio_in_video
            else jnp.sum(vision_tokens == QWEN3_OMNI_VIDEO_TOKEN).item()
        )
        if len(vision_tokens) > 0
        else 0
    )

    input_tokens = valid_input_ids.tolist()
    llm_pos_ids_list = []
    st = 0
    remain_images = image_nums
    remain_videos = video_nums
    remain_audios = audio_nums
    image_idx = 0
    video_idx = 0
    audio_idx = 0

    multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums

    # Process each multimodal element
    for _ in range(multimodal_nums):
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0

      # Find next vision or audio start token
      if (QWEN3_OMNI_IMAGE_TOKEN in input_tokens or QWEN3_OMNI_VIDEO_TOKEN in input_tokens) and (
          remain_videos > 0 or remain_images > 0
      ):
        try:
          ed_vision_start = input_tokens.index(QWEN3_OMNI_VISION_START_TOKEN, st)
        except ValueError:
          ed_vision_start = len(input_tokens) + 1
      else:
        ed_vision_start = len(input_tokens) + 1

      if QWEN3_OMNI_AUDIO_TOKEN in input_tokens and remain_audios > 0:
        try:
          ed_audio_start = input_tokens.index(QWEN3_OMNI_AUDIO_START_TOKEN, st)
        except ValueError:
          ed_audio_start = len(input_tokens) + 1
      else:
        ed_audio_start = len(input_tokens) + 1

      min_ed = min(ed_vision_start, ed_audio_start)

      # Add text tokens before multimodal element
      text_len = min_ed - st
      if text_len > 0:
        text_pos = jnp.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
        llm_pos_ids_list.append(text_pos)
        st_idx += text_len

      # Determine BOS/EOS token lengths
      if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
        bos_len, eos_len = 2, 2  # Audio in video
      else:
        bos_len, eos_len = 1, 1

      # Add BOS token(s)
      bos_pos = jnp.arange(bos_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(bos_pos)
      st_idx += bos_len

      # Process modality-specific content
      # Audio Only
      if min_ed == ed_audio_start:
        audio_len = _get_feat_extract_output_lengths(audio_lengths[audio_idx]).item()
        audio_pos = jnp.arange(audio_len).reshape(1, -1).repeat(3, axis=0) + st_idx
        llm_pos_ids_list.append(audio_pos)

        st += int(text_len + bos_len + audio_len + eos_len)
        audio_idx += 1
        remain_audios -= 1

      # Image Only
      elif min_ed == ed_vision_start and input_tokens[ed_vision_start + 1] == QWEN3_OMNI_IMAGE_TOKEN:
        grid_t = image_grid_thw[image_idx, 0].item()
        grid_hs = image_grid_thw[:, 1]
        grid_ws = image_grid_thw[:, 2]
        t_index = jnp.arange(grid_t, dtype=jnp.float32) * 1 * position_id_per_seconds

        image_pos = get_llm_pos_ids_for_vision(st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws)
        llm_pos_ids_list.append(image_pos)

        image_len = int(jnp.prod(image_grid_thw[image_idx]).item() // (spatial_merge_size**2))
        st += int(text_len + bos_len + image_len + eos_len)
        image_idx += 1
        remain_images -= 1

      # Video Only
      elif min_ed == ed_vision_start and input_tokens[ed_vision_start + 1] == QWEN3_OMNI_VIDEO_TOKEN:
        grid_t = video_grid_thw[video_idx, 0].item()
        grid_hs = video_grid_thw[:, 1]
        grid_ws = video_grid_thw[:, 2]
        t_index = jnp.arange(grid_t, dtype=jnp.float32) * second_per_grids[video_idx].item() * position_id_per_seconds

        video_pos = get_llm_pos_ids_for_vision(st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws)
        llm_pos_ids_list.append(video_pos)

        video_len = int(jnp.prod(video_grid_thw[video_idx]).item() // (spatial_merge_size**2))
        st += int(text_len + bos_len + video_len + eos_len)
        video_idx += 1
        remain_videos -= 1

      # Audio in Video (interleaved)
      elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
        audio_len = _get_feat_extract_output_lengths(audio_lengths[audio_idx]).item()
        audio_llm_pos_ids = jnp.arange(audio_len).reshape(1, -1).repeat(3, axis=0) + st_idx

        grid_t = video_grid_thw[video_idx, 0].item()
        grid_hs = video_grid_thw[:, 1]
        grid_ws = video_grid_thw[:, 2]
        t_index = jnp.arange(grid_t, dtype=jnp.float32) * second_per_grids[video_idx].item() * position_id_per_seconds

        video_llm_pos_ids = get_llm_pos_ids_for_vision(st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws)

        # Interleave audio and video based on temporal ordering
        video_data_index = 0
        audio_data_index = 0
        while video_data_index < video_llm_pos_ids.shape[1] and audio_data_index < audio_llm_pos_ids.shape[1]:
          if video_llm_pos_ids[0, video_data_index] <= audio_llm_pos_ids[0, audio_data_index]:
            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
            video_data_index += 1
          else:
            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
            audio_data_index += 1

        # Append remaining tokens
        if video_data_index < video_llm_pos_ids.shape[1]:
          llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index:])
        if audio_data_index < audio_llm_pos_ids.shape[1]:
          llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index:])

        video_len = int(jnp.prod(video_grid_thw[video_idx]).item() // (spatial_merge_size**2))
        st += int(text_len + bos_len + audio_len + video_len + eos_len)

        audio_idx += 1
        video_idx += 1
        remain_videos -= 1
        remain_audios -= 1

      # Add EOS token(s)
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
      eos_pos = jnp.arange(eos_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(eos_pos)

    # Add any remaining text tokens
    if st < len(input_tokens):
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
      text_len = len(input_tokens) - st
      remaining_text_pos = jnp.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(remaining_text_pos)

    # Concatenate all position IDs for this sequence
    llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1)

    # Place into position_ids tensor at valid positions
    valid_positions = jnp.where(attention_mask_bool[i])[0]
    position_ids = position_ids.at[:, i, valid_positions].set(llm_positions)

    # Calculate delta for this sequence
    mrope_position_deltas.append(llm_positions.max().item() + 1 - len(valid_input_ids))

  mrope_position_deltas = jnp.array(mrope_position_deltas).reshape(batch_size, 1)

  return position_ids, mrope_position_deltas
