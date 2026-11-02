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

"""OpenVLA-specific utilities for multimodal (image + instruction -> action tokens) features.

Mirrors `PrismaticProcessor` / `OpenVLAForActionPrediction.predict_action` of
https://huggingface.co/openvla/openvla-7b:
  * Image: `resize-naive` bicubic resize to 224x224, then two normalizations (DINOv2 and SigLIP)
    channel-stacked into a single [224, 224, 6] array.
  * Text: "In: What action should the robot take to {instruction}?\\nOut:" tokenized with BOS; the
    256 projected patch embeddings are inserted right after BOS, and the empty token "" (29871) is
    appended after "Out:" to match the training-time inputs.
  * Output: 7 action tokens; token id `t` maps to bin `32000 - t - 1` of 256 uniform bins in [-1, 1].
"""

from dataclasses import dataclass
import json
import os

import numpy as np
from PIL import Image

from maxtext.multimodal import utils as mm_utils

OPENVLA_IMAGE_SIZE = 224
OPENVLA_NUM_CHANNELS = 6  # [DINOv2 RGB | SigLIP RGB]
# `tvf_normalize_params` from the HF `preprocessor_config.json` (these override the timm defaults in
# `PrismaticImageProcessor.__init__`, so they are the values HF actually uses, incl. bf16 rounding).
OPENVLA_DINO_MEAN = (0.484375, 0.455078125, 0.40625)
OPENVLA_DINO_STD = (0.228515625, 0.2236328125, 0.224609375)
OPENVLA_SIGLIP_MEAN = (0.5, 0.5, 0.5)
OPENVLA_SIGLIP_STD = (0.5, 0.5, 0.5)

OPENVLA_NUM_IMAGE_TOKENS = 256  # (224 / 14) ** 2 patches per image
OPENVLA_PAD_TOKEN = 32000  # tokenizer `<PAD>`; also used by the inference tokenizer for padding
# The LLM vocab is padded to 32064; ids 32001..32063 are never produced by the tokenizer, so 32001 is a
# safe soft-token placeholder that cannot collide with text or padding.
OPENVLA_IMAGE_PLACEHOLDER = 32001
OPENVLA_EMPTY_TOKEN = 29871  # "" token appended after "Out:" by `predict_action`
OPENVLA_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


@dataclass
class OpenVLAPreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of the OpenVLA image preprocessor."""

  num_images: int = 0
  pixel_values: None | np.ndarray = None  # [N, 224, 224, 6]


def _to_pil(img) -> Image.Image:
  if isinstance(img, Image.Image):
    return img.convert("RGB")
  return Image.fromarray(np.asarray(img)).convert("RGB")


def preprocess_mm_data_openvla(images):
  """Resize (bicubic, 224x224) and normalize images for the fused DINOv2 + SigLIP backbone."""
  images_in = [images] if isinstance(images, (np.ndarray, Image.Image)) else list(images)
  dino_mean, dino_std = np.asarray(OPENVLA_DINO_MEAN, np.float32), np.asarray(OPENVLA_DINO_STD, np.float32)
  sig_mean, sig_std = np.asarray(OPENVLA_SIGLIP_MEAN, np.float32), np.asarray(OPENVLA_SIGLIP_STD, np.float32)
  images_out = []
  for img in images_in:
    # torchvision `resize` on a PIL image (antialias=True) == PIL bicubic resize.
    resized = _to_pil(img).resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
    x = np.asarray(resized, dtype=np.float32) / np.float32(255.0)  # `to_tensor`
    dino = (x - dino_mean) / dino_std
    siglip = (x - sig_mean) / sig_std
    images_out.append(np.concatenate([dino, siglip], axis=-1))
  return OpenVLAPreprocessorOutput(
      num_images=len(images_in),
      pixel_values=np.stack(images_out, axis=0).astype(np.float32),  # (N, H, W, 6)
  )


def get_image_offsets_openvla(processor_output: mm_utils.PreprocessorOutput | None):
  """Extra tokens added to the text: 256 soft tokens per image, plus the trailing empty token."""
  has_images = processor_output is not None and processor_output.pixel_values is not None
  num_images = processor_output.pixel_values.shape[0] if has_images else 1  # pyrefly: ignore[missing-attribute]
  return OPENVLA_NUM_IMAGE_TOKENS * num_images + 1


def reformat_prompt_openvla(prompt, image_placeholder=None, num_images=1):
  """Wraps a bare instruction into the OpenVLA prompt template.

  The prompt is right-stripped so that its tokenization never ends with the empty token; the empty
  token is always appended by `add_extra_tokens_for_images_openvla` (as `predict_action` does).
  """
  del num_images
  if image_placeholder:
    prompt = prompt.replace(image_placeholder, "")
  prompt = prompt.strip()
  if not prompt.startswith("In:"):
    prompt = OPENVLA_PROMPT_TEMPLATE.format(instruction=prompt)
  return prompt.rstrip()


def add_extra_tokens_for_images_openvla(tokens, max_num_images: int = 1, pad_id: int = OPENVLA_PAD_TOKEN):
  """[BOS, t1..tn, PAD..] -> [BOS, IMG*256*N, t1..tn, 29871, PAD..] (keeps trailing padding)."""
  tokens = np.asarray(tokens)
  squeeze = tokens.ndim == 1
  if squeeze:
    tokens = tokens[None, :]
  num_mm = OPENVLA_NUM_IMAGE_TOKENS * max_num_images
  out = np.full((tokens.shape[0], tokens.shape[1] + num_mm + 1), pad_id, dtype=tokens.dtype)
  for b, row in enumerate(tokens):
    non_pad = np.nonzero(row != pad_id)[0]
    true_len = int(non_pad[-1]) + 1 if non_pad.size else 0
    new_row = np.concatenate(
        [
            row[:1],
            np.full((num_mm,), OPENVLA_IMAGE_PLACEHOLDER, dtype=row.dtype),
            row[1:true_len],
            np.asarray([OPENVLA_EMPTY_TOKEN], dtype=row.dtype),
        ]
    )
    out[b, : new_row.shape[0]] = new_row
  return out[0] if squeeze else out


def action_tokens_to_actions(token_ids, norm_stats: dict, n_action_bins: int = 256, vocab_size: int = 32000):
  """Decodes action token ids into (unnormalized) continuous actions, as in `predict_action`."""
  bins = np.linspace(-1, 1, n_action_bins)
  bin_centers = (bins[:-1] + bins[1:]) / 2.0
  discretized = vocab_size - np.asarray(token_ids)
  discretized = np.clip(discretized - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
  normalized = bin_centers[discretized]
  stats = norm_stats["action"]
  mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
  high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
  return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


OPENVLA_ACTION_DIM_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def load_openvla_norm_stats(path: str) -> dict:
  """Loads `norm_stats` from an HF OpenVLA `config.json` (or a model dir containing it, or a bare stats JSON)."""
  if os.path.isdir(path):
    path = os.path.join(path, "config.json")
  with open(path, "rt", encoding="utf-8") as f:
    data = json.load(f)
  return data.get("norm_stats", data)


def decode_actions_openvla(token_ids, config):
  """Generated token ids -> (action_token_ids, normalized actions, unnormalized actions, unnorm_key).

  Uses the first `action_dim` generated tokens, as `OpenVLAForActionPrediction.predict_action` does.
  """
  stats_path = config.action_norm_stats_path or config.tokenizer_path
  norm_stats = load_openvla_norm_stats(stats_path)
  key = config.action_unnorm_key
  if key not in norm_stats:
    raise ValueError(f"action_unnorm_key={key!r} not in norm_stats; choose one of {sorted(norm_stats)}")
  stats = norm_stats[key]
  action_dim = len(stats["action"]["q01"])
  ids = np.asarray(list(token_ids)[:action_dim])
  if ids.shape[0] < action_dim:
    raise ValueError(f"Need {action_dim} generated tokens to decode an action, got {ids.shape[0]}.")
  bins = np.linspace(-1, 1, 256)
  centers = (bins[:-1] + bins[1:]) / 2.0
  normalized = centers[np.clip(OPENVLA_PAD_TOKEN - ids - 1, 0, centers.shape[0] - 1)]
  return ids, normalized, action_tokens_to_actions(ids, stats), key


def format_actions_openvla(token_ids, config) -> str:
  """Human-readable action summary for `decode.py`."""
  ids, normalized, actions, key = decode_actions_openvla(token_ids, config)
  names = OPENVLA_ACTION_DIM_NAMES if len(actions) == len(OPENVLA_ACTION_DIM_NAMES) else range(len(actions))
  rows = "\n".join(
      f"  {str(n):>8}: token={int(t):>5}  normalized={v:+.4f}  action={a:+.6f}"
      for n, t, v, a in zip(names, ids, normalized, actions)
  )
  return f"OpenVLA action (unnorm_key={key}):\n{rows}\n  action = {np.array2string(actions, precision=6, separator=', ')}"


def get_dummy_image_shape_for_init_openvla(batch_size=1, num_image_per_sequence=1):
  """Return the shape of the dummy image for OpenVLA model's initialization."""
  return (batch_size, num_image_per_sequence, OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, OPENVLA_NUM_CHANNELS)
