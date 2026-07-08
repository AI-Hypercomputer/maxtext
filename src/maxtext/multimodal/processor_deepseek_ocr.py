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

"""DeepSeek-OCR-2-specific utilities for multimodal features."""

from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps

from maxtext.multimodal import utils as mm_utils

# Constants for DeepSeek-OCR-2
DEEPSEEK_OCR_IMAGE_TOKEN_ID = 128815
DEEPSEEK_OCR_BASE_SIZE = 1024
DEEPSEEK_OCR_IMAGE_SIZE = 768
DEEPSEEK_OCR_MAX_CROPS = 6
DEEPSEEK_OCR_IMAGE_PLACEHOLDER_IN_PROMPT = "<image>"

DEEPSEEK_OCR_GLOBAL_TOKENS = 256
DEEPSEEK_OCR_CROP_TOKENS = 144
DEEPSEEK_OCR_SEPARATOR_TOKENS = 1
DEEPSEEK_OCR_NUM_TOKENS_PER_IMAGE = (
    DEEPSEEK_OCR_CROP_TOKENS * DEEPSEEK_OCR_MAX_CROPS + DEEPSEEK_OCR_GLOBAL_TOKENS + DEEPSEEK_OCR_SEPARATOR_TOKENS
)


@dataclass
class DeepseekOCR2PreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of DeepSeek-OCR-2 image preprocessor."""

  pixel_values: None | np.ndarray = None
  pixel_mask: None | np.ndarray = None
  aspect_ratios: None | np.ndarray = None
  num_images: int = 0


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
  best_ratio_diff = float("inf")
  best_ratio = (1, 1)
  area = width * height
  for ratio in target_ratios:
    target_aspect_ratio = ratio[0] / ratio[1]
    ratio_diff = abs(aspect_ratio - target_aspect_ratio)
    if ratio_diff < best_ratio_diff:
      best_ratio_diff = ratio_diff
      best_ratio = ratio
    elif ratio_diff == best_ratio_diff:
      if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
        best_ratio = ratio
  return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
  orig_width, orig_height = image.size
  aspect_ratio = orig_width / orig_height

  target_ratios = set(
      (i, j)
      for n in range(min_num, max_num + 1)
      for i in range(1, n + 1)
      for j in range(1, n + 1)
      if i * j <= max_num and i * j >= min_num
  )
  target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

  target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

  target_width = image_size * target_aspect_ratio[0]
  target_height = image_size * target_aspect_ratio[1]
  blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

  resized_img = image.resize((target_width, target_height))
  processed_images = []
  for i in range(blocks):
    box = (
        (i % (target_width // image_size)) * image_size,
        (i // (target_width // image_size)) * image_size,
        ((i % (target_width // image_size)) + 1) * image_size,
        ((i // (target_width // image_size)) + 1) * image_size,
    )
    split_img = resized_img.crop(box)
    processed_images.append(split_img)
  assert len(processed_images) == blocks
  if use_thumbnail and len(processed_images) != 1:
    thumbnail_img = image.resize((image_size, image_size))
    processed_images.append(thumbnail_img)
  return processed_images, target_aspect_ratio


def preprocess_mm_data_deepseek_ocr(
    images, base_size=DEEPSEEK_OCR_BASE_SIZE, image_size=DEEPSEEK_OCR_IMAGE_SIZE, crop_mode=True
):
  """Preprocesses images for DeepSeek-OCR-2."""
  images_in = []
  if isinstance(images, np.ndarray):
    images_in.append(images)
  elif isinstance(images, list):
    images_in.extend(images)
  else:
    images_in.append(images)

  out_pixel_values = []
  out_pixel_mask = []
  out_aspect_ratios = []

  for img in images_in:
    if isinstance(img, np.ndarray):
      # If it is HWC, convert to PIL
      if len(img.shape) == 3:
        pil_img = Image.fromarray(img)
      else:
        raise ValueError(f"Unsupported numpy array shape: {img.shape}")
    elif isinstance(img, Image.Image):
      pil_img = img
    else:
      # Try to load if it is a path (though usually loaded before)
      pil_img = mm_utils.load_image_from_path(img)
      if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(pil_img)
    pil_img = pil_img.convert("RGB")

    # Global view
    global_view = ImageOps.pad(pil_img, (base_size, base_size), color=(127, 127, 127))
    global_tensor = np.array(global_view, dtype=np.float32) / 255.0
    global_tensor = (global_tensor - 0.5) / 0.5

    crops = []
    crop_ratio = [1, 1]
    if crop_mode:
      if pil_img.size[0] <= 768 and pil_img.size[1] <= 768:
        crop_ratio = [1, 1]
      else:
        crops_raw, crop_ratio = dynamic_preprocess(
            pil_img, min_num=2, max_num=DEEPSEEK_OCR_MAX_CROPS, image_size=image_size
        )
        for crop in crops_raw:
          crop_tensor = np.array(crop, dtype=np.float32) / 255.0
          crop_tensor = (crop_tensor - 0.5) / 0.5
          padded_crop = np.zeros((base_size, base_size, 3), dtype=np.float32)
          padded_crop[:image_size, :image_size, :] = crop_tensor
          crops.append(padded_crop)

    num_crops = len(crops)
    while len(crops) < DEEPSEEK_OCR_MAX_CROPS:
      crops.append(np.zeros((base_size, base_size, 3), dtype=np.float32))

    img_pixel_values = np.stack([global_tensor] + crops, axis=0)

    img_pixel_mask = np.zeros((DEEPSEEK_OCR_NUM_TOKENS_PER_IMAGE,), dtype=np.bool_)
    for i in range(num_crops):
      start = i * DEEPSEEK_OCR_CROP_TOKENS
      img_pixel_mask[start : start + DEEPSEEK_OCR_CROP_TOKENS] = True
    global_start = DEEPSEEK_OCR_CROP_TOKENS * DEEPSEEK_OCR_MAX_CROPS
    img_pixel_mask[global_start : global_start + DEEPSEEK_OCR_GLOBAL_TOKENS + DEEPSEEK_OCR_SEPARATOR_TOKENS] = True

    out_pixel_values.append(img_pixel_values)
    out_pixel_mask.append(img_pixel_mask)
    out_aspect_ratios.append(crop_ratio)

  return DeepseekOCR2PreprocessorOutput(
      pixel_values=np.stack(out_pixel_values, axis=0),
      pixel_mask=np.stack(out_pixel_mask, axis=0),
      aspect_ratios=np.array(out_aspect_ratios, dtype=np.int32),
      num_images=len(images_in),
  )


def get_image_offsets_deepseek_ocr(processor_output: mm_utils.PreprocessorOutput | None):
  """Get the increase in total token count after inserting image token placeholders."""
  has_images = processor_output is not None and processor_output.pixel_values is not None
  if not has_images:
    return DEEPSEEK_OCR_NUM_TOKENS_PER_IMAGE - 1
  if processor_output.pixel_mask is None:
    return (DEEPSEEK_OCR_NUM_TOKENS_PER_IMAGE - 1) * processor_output.pixel_values.shape[0]
  return int(np.sum(processor_output.pixel_mask, dtype=np.int32) - processor_output.pixel_values.shape[0])


def reformat_prompt_deepseek_ocr(prompt, image_placeholder, num_images):
  """Reformat prompt for DeepSeek-OCR-2 (plain SFT, no wrapping)."""
  prompt = prompt.replace("\\n", "\n")
  if image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, DEEPSEEK_OCR_IMAGE_PLACEHOLDER_IN_PROMPT)
  image_placeholder_count = prompt.count(DEEPSEEK_OCR_IMAGE_PLACEHOLDER_IN_PROMPT)
  if image_placeholder_count < num_images:
    prompt = DEEPSEEK_OCR_IMAGE_PLACEHOLDER_IN_PROMPT * (num_images - image_placeholder_count) + prompt
  # The user verification script uses: "<image>\n<|grounding|>Convert the document to markdown. "
  # We don't need to add chat templates if we want to match the verification.
  return prompt


def add_extra_tokens_for_images_deepseek_ocr(tokens, processor_output: mm_utils.PreprocessorOutput | None = None):
  """Inserts image placeholder tokens into the token list."""
  if processor_output is not None and processor_output.pixel_mask is not None:
    token_counts = np.sum(processor_output.pixel_mask, axis=-1, dtype=np.int32).tolist()
  else:
    num_images = processor_output.num_images if processor_output is not None else 1
    token_counts = [DEEPSEEK_OCR_NUM_TOKENS_PER_IMAGE] * num_images
  return insert_variable_sequences(tokens, at=DEEPSEEK_OCR_IMAGE_TOKEN_ID, sequence_lengths=token_counts)


def get_dummy_image_shape_for_init_deepseek_ocr(batch_size=1, num_image_per_sequence=1):
  """Return the shape of the dummy image for initialization."""
  return (
      batch_size,
      num_image_per_sequence,
      1 + DEEPSEEK_OCR_MAX_CROPS,
      DEEPSEEK_OCR_BASE_SIZE,
      DEEPSEEK_OCR_BASE_SIZE,
      3,
  )


# Helper for insertion (copied from processor_gemma3.py to avoid import issues if it moves)
def _get_new_text_positions(offset_on: np.ndarray, offset_by: int) -> np.ndarray:
  offset = np.cumsum(offset_on, axis=-1) * offset_by
  new_positions = np.arange(offset_on.shape[-1]) + offset
  new_positions -= offset_by * offset_on
  return new_positions


def insert_sequence(tokens: np.ndarray, at: int, sequence: list[int], max_num_images: int) -> np.ndarray:
  """Inserts a sequence of tokens at all occurrences of a specific token `at`."""
  is_1d = len(tokens.shape) == 1
  if is_1d:
    tokens = np.expand_dims(tokens, axis=0)

  batch_size, seq_len = tokens.shape
  sequence_len = len(sequence)
  offset_by = sequence_len - 1

  # Find where the placeholder tokens are
  offset_on = tokens == at

  # Calculate new positions for all tokens
  new_positions = _get_new_text_positions(offset_on=offset_on, offset_by=offset_by)

  # Allocate new token array
  new_seq_len = seq_len + offset_by * max_num_images
  new_tokens = np.zeros((batch_size, new_seq_len), dtype=tokens.dtype)

  # Place old tokens in their new positions
  # We use advanced indexing to scatter
  batch_indices = np.arange(batch_size)[:, None]
  new_positions_clamped = np.clip(new_positions, 0, new_seq_len - 1)
  new_tokens[batch_indices, new_positions_clamped] = tokens

  # Fill in the inserted sequences
  # We find the new positions of the `at` token
  for b in range(batch_size):
    at_indices = np.where(offset_on[b])[0]
    for idx in at_indices:
      start_pos = new_positions[b, idx]
      new_tokens[b, start_pos : start_pos + sequence_len] = sequence

  if is_1d:
    new_tokens = np.squeeze(new_tokens, axis=0)

  return new_tokens


def insert_variable_sequences(tokens: np.ndarray, at: int, sequence_lengths: list[int]) -> np.ndarray:
  """Replaces each image token with the matching number of DeepSeek visual placeholders."""
  is_1d = len(tokens.shape) == 1
  if is_1d:
    tokens = np.expand_dims(tokens, axis=0)

  rows = []
  for row in tokens:
    image_index = 0
    pieces = []
    for token in row:
      if token == at:
        sequence_len = sequence_lengths[min(image_index, len(sequence_lengths) - 1)]
        pieces.extend([at] * sequence_len)
        image_index += 1
      else:
        pieces.append(token)
    rows.append(np.array(pieces, dtype=tokens.dtype))

  max_len = max(row.shape[0] for row in rows)
  new_tokens = np.zeros((len(rows), max_len), dtype=tokens.dtype)
  for i, row in enumerate(rows):
    new_tokens[i, : row.shape[0]] = row

  if is_1d:
    new_tokens = np.squeeze(new_tokens, axis=0)

  return new_tokens
