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

"""DeepSeek-V4.1 aspect-preserving dynamic-resolution image preprocessor."""

import base64
from dataclasses import dataclass
import io
import math
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageOps

from maxtext.multimodal import utils as mm_utils

DEEPSEEK41_IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
DEEPSEEK41_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK41_EOS_TOKEN = "<｜end▁of▁sentence｜>"
DEEPSEEK41_USER_TOKEN = "<｜User｜>"
DEEPSEEK41_ASSISTANT_TOKEN = "<｜Assistant｜>"
DEEPSEEK41_THINK_END_TOKEN = "</think>"

DEEPSEEK41_DEFAULT_PATCH_SIZE = 14
DEEPSEEK41_DEFAULT_DOWNSAMPLE_RATIO = 3
DEEPSEEK41_DEFAULT_MIN_PIXELS = 544 * 544
DEEPSEEK41_DEFAULT_MAX_N_TOKEN = 1024
DEEPSEEK41_NUM_CHANNELS = 3
DEEPSEEK41_PAD_COLOR = (127, 127, 127)


@dataclass
class DeepSeek41PreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds DeepSeek-V4.1 padded ViT patches, validity masks, and per-image `[n_vit_h, n_vit_w]` grids."""

  image_grids: None | np.ndarray = None


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
  """Returns total LLM token span length `1 + n_llm_h * (n_llm_w + 1) + 1` for a valid grid."""
  if n_llm_h <= 0 or n_llm_w <= 0:
    return 0
  return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int) -> tuple[int, int]:
  """Returns `(n_llm_h, n_llm_w)` aligned token grid dimensions for a pixel size."""
  return (
      math.ceil((best_height // patch_size) / downsample_ratio),
      math.ceil((best_width // patch_size) / downsample_ratio),
  )


def solve_resize_ratio(
    height: float,
    width: float,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int]:
  """Finds the largest aspect-preserving pixel size whose token span fits in `max_n_token`."""
  r = height / width
  max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
  max_h_float = max_w_float * r
  cell = patch_size * downsample_ratio
  if max_w_float < 1.0:
    return ((max_n_token - 2) // 2) * cell, cell
  if max_h_float < 1.0:
    return cell, (max_n_token - 3) * cell
  beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
  return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(
    height: float,
    width: float,
    best_height: int,
    best_width: int,
    *,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
  """Shrinks `(best_height, best_width)` if needed so `num_image_tokens <= max_n_token`."""
  n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
  if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
    best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
      raise ValueError(
          f"Planned grid {(n_llm_h, n_llm_w)} exceeds max_n_token={max_n_token} " f"for image size {(height, width)}."
      )
  return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(
    width: int,
    height: int,
    *,
    patch_size: int = DEEPSEEK41_DEFAULT_PATCH_SIZE,
    downsample_ratio: int = DEEPSEEK41_DEFAULT_DOWNSAMPLE_RATIO,
    min_pixels: int = DEEPSEEK41_DEFAULT_MIN_PIXELS,
    max_n_token: int = DEEPSEEK41_DEFAULT_MAX_N_TOKEN,
    max_wh_ratio: float | None = None,
) -> tuple[int, int, int, int]:
  """Plans `(n_llm_h, n_llm_w, best_height, best_width)` matching upstream `image_processor.py`."""
  if width <= 0 or height <= 0:
    raise ValueError(f"Image width and height must be positive, got {(width, height)}.")
  if patch_size <= 0 or downsample_ratio <= 0:
    raise ValueError(f"patch_size and downsample_ratio must be positive, got {(patch_size, downsample_ratio)}.")
  if max_n_token < 4:
    raise ValueError(f"ds41_vision_max_n_token must be >= 4, got {max_n_token}.")
  adj_w: float = float(width)
  adj_h: float = float(height)
  if max_wh_ratio is not None and adj_w > adj_h * max_wh_ratio:
    adj_w = adj_h * max_wh_ratio
  if 0 < adj_w * adj_h < min_pixels:
    ratio = (min_pixels / (adj_w * adj_h)) ** 0.5
    adj_w = int(adj_w * ratio)
    adj_h = int(adj_h * ratio)
  best_width = math.ceil(adj_w / patch_size) * patch_size
  best_height = math.ceil(adj_h / patch_size) * patch_size
  return safe_resize(
      adj_h,
      adj_w,
      best_height,
      best_width,
      patch_size=patch_size,
      downsample_ratio=downsample_ratio,
      max_n_token=max_n_token,
  )


def max_patches_for_config(config: Any = None) -> int:
  """Returns static worst-case ViT patch count `P_max = r^2 * (max_n_token - 3)`."""
  downsample_ratio = DEEPSEEK41_DEFAULT_DOWNSAMPLE_RATIO if config is None else int(config.ds41_vision_downsample_ratio)
  max_n_token = DEEPSEEK41_DEFAULT_MAX_N_TOKEN if config is None else int(config.ds41_vision_max_n_token)
  if downsample_ratio <= 0 or max_n_token < 4:
    raise ValueError(
        f"Invalid DeepSeek-V4.1 vision bounds: downsample_ratio={downsample_ratio}, max_n_token={max_n_token}."
    )
  return (downsample_ratio**2) * (max_n_token - 3)


def _to_rgb_pil_image(image: Any) -> Image.Image:
  """Converts a PIL Image, uint8/float RGB array, bytes, path, or dict record into an RGB PIL Image."""
  if isinstance(image, Image.Image):
    return image.convert("RGB") if image.mode != "RGB" else image
  if isinstance(image, np.ndarray):
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
      raise ValueError(f"Expected HWC image array, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
      arr_u8 = np.clip(np.rint(image * 255.0 if image.max(initial=0.0) <= 1.0 else image), 0, 255).astype(np.uint8)
    else:
      arr_u8 = image.astype(np.uint8, copy=False)
    pil_img = Image.fromarray(arr_u8)
    return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
  if isinstance(image, dict):
    if image.get("bytes") is not None:
      with Image.open(io.BytesIO(image["bytes"])) as src:
        return src.convert("RGB")
    data = image.get("data")
    if isinstance(data, bytes):
      with Image.open(io.BytesIO(data)) as src:
        return src.convert("RGB")
    if isinstance(data, str):
      with Image.open(io.BytesIO(base64.b64decode(data))) as src:
        return src.convert("RGB")
    if isinstance(image.get("path"), str) and image["path"]:
      with Image.open(image["path"]) as src:
        return src.convert("RGB")
    if isinstance(image.get("url"), str) and image["url"]:
      url = image["url"]
      if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" not in header:
          raise ValueError(f"Unsupported data URL encoding: {header}")
        with Image.open(io.BytesIO(base64.b64decode(payload))) as src:
          return src.convert("RGB")
      with Image.open(url) as src:
        return src.convert("RGB")
  pil_or_obj = mm_utils.convert_to_RGB(image)
  if isinstance(pil_or_obj, Image.Image):
    return pil_or_obj
  raise TypeError(f"Unsupported image input type for DeepSeek-V4.1 preprocessing: {type(image)}")


def extract_image_patches_deepseek41(
    image: Any,
    *,
    patch_size: int = DEEPSEEK41_DEFAULT_PATCH_SIZE,
    downsample_ratio: int = DEEPSEEK41_DEFAULT_DOWNSAMPLE_RATIO,
    min_pixels: int = DEEPSEEK41_DEFAULT_MIN_PIXELS,
    max_n_token: int = DEEPSEEK41_DEFAULT_MAX_N_TOKEN,
    max_wh_ratio: float | None = None,
) -> tuple[np.ndarray, int, int, int, int]:
  """Resizes/pads one image and extracts `[n_vit_h * n_vit_w, 3 * p * p]` float32 patches."""
  pil_img = _to_rgb_pil_image(image)
  n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
      pil_img.width,
      pil_img.height,
      patch_size=patch_size,
      downsample_ratio=downsample_ratio,
      min_pixels=min_pixels,
      max_n_token=max_n_token,
      max_wh_ratio=max_wh_ratio,
  )
  n_vit_h = best_height // patch_size
  n_vit_w = best_width // patch_size
  if max_wh_ratio is not None and pil_img.width >= max_wh_ratio * pil_img.height:
    resized = pil_img.resize((best_width, best_height))
  else:
    resized = ImageOps.pad(pil_img, (best_width, best_height), color=DEEPSEEK41_PAD_COLOR)
  x = np.transpose(np.asarray(resized, dtype=np.float32), (2, 0, 1)) / np.float32(255.0)
  x = (x - np.float32(0.5)) / np.float32(0.5)
  patches = (
      x.reshape(DEEPSEEK41_NUM_CHANNELS, n_vit_h, patch_size, n_vit_w, patch_size)
      .transpose(1, 3, 0, 2, 4)
      .reshape(n_vit_h * n_vit_w, DEEPSEEK41_NUM_CHANNELS * patch_size * patch_size)
  )
  return np.ascontiguousarray(patches, dtype=np.float32), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def preprocess_mm_data_deepseek41(
    images: Any,
    config: Any = None,
) -> DeepSeek41PreprocessorOutput:
  """Preprocesses one or more images into padded `(N_max, P_max, 3*p*p)` patches, masks, and grids."""
  patch_size = DEEPSEEK41_DEFAULT_PATCH_SIZE if config is None else int(config.patch_size_for_vit)
  num_channels = DEEPSEEK41_NUM_CHANNELS if config is None else int(config.num_channels_for_vit)
  if num_channels != DEEPSEEK41_NUM_CHANNELS:
    raise ValueError(f"DeepSeek-V4.1 RGB preprocessing requires num_channels_for_vit=3, got {num_channels}.")
  downsample_ratio = DEEPSEEK41_DEFAULT_DOWNSAMPLE_RATIO if config is None else int(config.ds41_vision_downsample_ratio)
  min_pixels = DEEPSEEK41_DEFAULT_MIN_PIXELS if config is None else int(config.ds41_vision_min_pixels)
  max_n_token = DEEPSEEK41_DEFAULT_MAX_N_TOKEN if config is None else int(config.ds41_vision_max_n_token)
  max_wh_ratio = None if config is None else config.ds41_vision_max_wh_ratio
  max_images_cfg = -1 if config is None else int(config.max_num_images_per_example)

  if images is None:
    images_in: list[Any] = []
  elif isinstance(images, (list, tuple)):
    images_in = list(images)
  else:
    images_in = [images]

  num_valid = len(images_in)
  if max_images_cfg > 0:
    if num_valid > max_images_cfg:
      raise ValueError(f"Number of images ({num_valid}) exceeds max_num_images_per_example ({max_images_cfg}).")
    num_slots = max_images_cfg
  else:
    num_slots = max(1, num_valid)

  p_max = (downsample_ratio**2) * (max_n_token - 3)
  patch_dim = num_channels * (patch_size**2)
  pixel_values = np.zeros((num_slots, p_max, patch_dim), dtype=np.float32)
  pixel_mask = np.zeros((num_slots,), dtype=np.int32)
  image_grids = np.zeros((num_slots, 2), dtype=np.int32)

  for idx, img in enumerate(images_in):
    patches, n_vit_h, n_vit_w, _, _ = extract_image_patches_deepseek41(
        img,
        patch_size=patch_size,
        downsample_ratio=downsample_ratio,
        min_pixels=min_pixels,
        max_n_token=max_n_token,
        max_wh_ratio=max_wh_ratio,
    )
    if patches.shape[0] > p_max:
      raise ValueError(f"Extracted {patches.shape[0]} patches for grid {(n_vit_h, n_vit_w)}, exceeding P_max={p_max}.")
    pixel_values[idx, : patches.shape[0], :] = patches
    pixel_mask[idx] = 1
    image_grids[idx, 0] = n_vit_h
    image_grids[idx, 1] = n_vit_w

  return DeepSeek41PreprocessorOutput(
      pixel_values=pixel_values,
      pixel_mask=pixel_mask,
      aspect_ratios=image_grids[:num_valid].copy(),
      num_images=num_valid,
      image_grids=image_grids,
  )


def get_image_offsets_deepseek41(
    config: Any,
    processor_output: mm_utils.PreprocessorOutput | None,
) -> int:
  """Returns net token count increase after replacing single-token placeholders with image spans."""
  r = DEEPSEEK41_DEFAULT_DOWNSAMPLE_RATIO if config is None else int(config.ds41_vision_downsample_ratio)
  image_grids = None if processor_output is None else getattr(processor_output, "image_grids", None)
  if image_grids is not None:
    total_offset = 0
    for row in np.asarray(image_grids, dtype=np.int32).reshape(-1, 2):
      n_vit_h, n_vit_w = int(row[0]), int(row[1])
      if n_vit_h > 0 and n_vit_w > 0:
        n_llm_h = (n_vit_h + r - 1) // r
        n_llm_w = (n_vit_w + r - 1) // r
        total_offset += num_image_tokens(n_llm_h, n_llm_w) - 1
    return total_offset

  if config is None:
    n_vit_h, n_vit_w = 39, 39
  else:
    n_vit_h, n_vit_w = int(config.ds41_vision_grid[0]), int(config.ds41_vision_grid[1])
  n_llm_h = (n_vit_h + r - 1) // r
  n_llm_w = (n_vit_w + r - 1) // r
  num_images = 1 if processor_output is None else max(0, int(processor_output.num_images))
  return (num_image_tokens(n_llm_h, n_llm_w) - 1) * num_images


def reformat_prompt_deepseek41(prompt: str, image_placeholder: str, num_images: int) -> str:
  """Formats a user prompt in DeepSeek-V4.1 chat format with image placeholders."""
  target_placeholder = DEEPSEEK41_IMAGE_PLACEHOLDER if image_placeholder in ("", "<|image|>") else image_placeholder
  if image_placeholder and image_placeholder != target_placeholder and image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, target_placeholder)
  placeholder_count = prompt.count(target_placeholder)
  if placeholder_count < num_images:
    prompt = target_placeholder * (num_images - placeholder_count) + prompt
  return (
      f"{DEEPSEEK41_BOS_TOKEN}{DEEPSEEK41_USER_TOKEN}{prompt}" f"{DEEPSEEK41_ASSISTANT_TOKEN}{DEEPSEEK41_THINK_END_TOKEN}"
  )


def reformat_response_deepseek41(response: str) -> str:
  """Appends DeepSeek-V4.1 EOS token to the assistant response."""
  return f"{response}{DEEPSEEK41_EOS_TOKEN}"


def add_extra_tokens_for_images_deepseek41(
    tokens: np.ndarray | list[int],
    config: Any,
    processor_output: mm_utils.PreprocessorOutput | None,
) -> np.ndarray:
  """Expands each `ds41_image_token_id` placeholder into `1 + n_llm_h * (n_llm_w + 1) + 1` tokens."""
  if config is None:
    raise ValueError("add_extra_tokens_for_images_deepseek41 requires config.")
  image_token_id = int(config.ds41_image_token_id)
  r = int(config.ds41_vision_downsample_ratio)
  if image_token_id <= 0 or image_token_id >= int(config.vocab_size):
    raise ValueError(
        f"DeepSeek-V4.1 requires 0 < ds41_image_token_id < vocab_size ({config.vocab_size}), " f"got {image_token_id}."
    )

  is_2d = isinstance(tokens, np.ndarray) and tokens.ndim == 2
  if is_2d:
    if tokens.shape[0] != 1:
      raise ValueError(f"Expected 1D tokens or 2D batch-1 tokens, got shape {tokens.shape}.")
    token_arr = np.asarray(tokens[0], dtype=np.int64)
  else:
    token_arr = np.asarray(tokens, dtype=np.int64)
    if token_arr.ndim != 1:
      raise ValueError(f"Expected 1D token sequence, got shape {token_arr.shape}.")

  valid_grids: list[tuple[int, int]] = []
  if processor_output is not None:
    raw_grids = getattr(processor_output, "image_grids", None)
    raw_mask = processor_output.pixel_mask
    if raw_grids is not None:
      grids_np = np.asarray(raw_grids, dtype=np.int32).reshape(-1, 2)
      mask_np = None if raw_mask is None else np.asarray(raw_mask).reshape(-1)
      for idx, (nh, nw) in enumerate(grids_np):
        if mask_np is not None and idx < mask_np.shape[0] and mask_np[idx] == 0:
          continue
        if int(nh) > 0 and int(nw) > 0:
          valid_grids.append((int(nh), int(nw)))
    elif processor_output.num_images > 0:
      nh, nw = int(config.ds41_vision_grid[0]), int(config.ds41_vision_grid[1])
      valid_grids = [(nh, nw)] * int(processor_output.num_images)

  placeholder_positions = np.flatnonzero(token_arr == image_token_id)
  if len(placeholder_positions) != len(valid_grids):
    raise ValueError(
        f"Found {len(placeholder_positions)} ds41_image_token_id ({image_token_id}) placeholders in prompt "
        f"tokens, but processor_output contains {len(valid_grids)} valid images."
    )

  if not valid_grids:
    out = token_arr.copy()
    return out[np.newaxis, :] if is_2d else out

  parts: list[np.ndarray] = []
  cursor = 0
  for pos, (n_vit_h, n_vit_w) in zip(placeholder_positions.tolist(), valid_grids):
    if pos > cursor:
      parts.append(token_arr[cursor:pos])
    n_llm_h = (n_vit_h + r - 1) // r
    n_llm_w = (n_vit_w + r - 1) // r
    span_len = num_image_tokens(n_llm_h, n_llm_w)
    parts.append(np.full((span_len,), image_token_id, dtype=token_arr.dtype))
    cursor = pos + 1
  if cursor < token_arr.shape[0]:
    parts.append(token_arr[cursor:])

  expanded = np.concatenate(parts, axis=0)
  return expanded[np.newaxis, :] if is_2d else expanded


def get_dummy_image_shape_for_init_deepseek41(
    batch_size: int = 1,
    num_image_per_sequence: int = 1,
    config: Any = None,
) -> tuple[int, int, int, int]:
  """Returns `[batch_size, num_images, num_patches, 3*p*p]` dummy patch shape for initialization."""
  if config is None:
    raise ValueError(
        "get_dummy_image_shape_for_init_deepseek41 requires config to read ds41_vision_grid and patch_size_for_vit."
    )
  use_dynamic = getattr(config, "dataset_type", "synthetic") in ("grain", "hf") or bool(
      getattr(config, "ds41_dynamic_vision_grid", False)
  )
  if use_dynamic:
    num_patches = max_patches_for_config(config)
  else:
    n_vit_h, n_vit_w = config.ds41_vision_grid
    num_patches = int(n_vit_h) * int(n_vit_w)
  patch_dim = int(config.num_channels_for_vit) * (int(config.patch_size_for_vit) ** 2)
  num_images = (
      max(1, int(config.max_num_images_per_example)) if num_image_per_sequence == 1 else int(num_image_per_sequence)
  )
  return (int(batch_size), num_images, num_patches, patch_dim)
