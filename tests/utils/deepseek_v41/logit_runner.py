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

"""DeepSeek-V4.1 forward-pass logit parity runner, scope metrics, and threshold gates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
import pickle
import sys
import traceback
from typing import Any

import absl
import jax
import jax.numpy as jnp
import jsonlines
import numpy as np

from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR
from maxtext.configs import pyconfig
from maxtext.utils import max_logging, maxtext_utils
from maxtext.utils.globals import MAXTEXT_TEST_ASSETS_ROOT
from tests.utils import deepseek_v41_frozen_topk
from tests.utils import forward_pass_logit_checker as fplc

absl.logging.set_verbosity(absl.logging.INFO)

EXIT_DETERMINISTIC_GATE_FAIL = fplc.EXIT_DETERMINISTIC_GATE_FAIL
DeterministicGateError = fplc.DeterministicGateError
DETERMINISTIC_CODE_ERRORS = fplc.DETERMINISTIC_CODE_ERRORS
HAS_TORCH = fplc.HAS_TORCH
torch = fplc.torch
F = fplc.F
convert_jax_weight_to_torch = fplc.convert_jax_weight_to_torch

SHORT_TEXT_PROMPT_MAX_TOKENS = 16
MM_SHORT_PROMPT_MAX_TOKENS = 512
_REQUIRED_TEXT_TOP10_MATCHES = 10
DEFERRED_LONG_TEXT_STATUS = "not gated"
_DEFAULT_MULTIMODAL_SHORT_IDS = frozenset({"L1", "L2", "L3", "mm_short_prompts"})
_DEFAULT_MULTIMODAL_LONG_IDS = frozenset({"L4", "mm_long_prompts"})


def str2bool(v: Any) -> bool:
  """Converts a string or boolean CLI value to `bool`."""
  return fplc.str2bool(v)


def resolve_sample_name(golden_data_point: Mapping[str, Any], fallback_label: str) -> str:
  """Resolves sample name from `name` or `sample_id` (falling back to `fallback_label`)."""
  for key in ("name", "sample_id"):
    if key in golden_data_point and golden_data_point[key] is not None:
      raw_val = golden_data_point[key]
      val_str = str(np.asarray(raw_val).item()) if isinstance(raw_val, np.ndarray) else str(raw_val)
      if val_str:
        return val_str
  return fallback_label


def has_ds41_mm_images(golden_data_point: Mapping[str, Any]) -> bool:
  """Returns True when `golden_data_point` carries DeepSeek-V4.1 `patches` or `encoder_images`."""
  return any(k in golden_data_point and golden_data_point[k] is not None for k in ("patches", "encoder_images"))


def resolve_ds41_image_token_id(config: Any) -> int | None:
  """Returns configured `ds41_image_token_id` when present on `config`, else None."""
  if "get_keys" in dir(config):
    if "ds41_image_token_id" not in config.get_keys():
      return None
  elif "ds41_image_token_id" not in dir(config):
    return None
  raw_val = config.ds41_image_token_id
  return int(raw_val) if raw_val is not None else None


def validate_sample_token_types(
    golden_data_point: Mapping[str, Any],
    *,
    sample_name: str,
    sample_tokens: Any,
    seq_len: int,
    start_index: int,
    token_size: int,
    is_mm_sample: bool,
    ds41_image_token_id: int | None,
) -> np.ndarray | None:
  """Validates release `token_types` (`{-1, 0, 1, 2, 3}`) and image-token alignment."""
  has_token_types = "token_types" in golden_data_point and golden_data_point["token_types"] is not None
  tt_full = None
  if has_token_types:
    raw_tt = np.asarray(golden_data_point["token_types"])
    if np.issubdtype(raw_tt.dtype, np.bool_) or not np.issubdtype(raw_tt.dtype, np.integer):
      raise ValueError(
          f"Sample {sample_name!r} token_types must have integer dtype with values in "
          f"{{-1, 0, 1, 2, 3}}, got dtype={raw_tt.dtype}"
      )
    raw_tt_arr = raw_tt.reshape(-1)
    if raw_tt_arr.shape[0] != seq_len:
      raise ValueError(f"Sample {sample_name!r} token_types length {raw_tt_arr.shape[0]} != tokens length {seq_len}")
    if not np.all((raw_tt_arr >= -1) & (raw_tt_arr <= 3)):
      raise ValueError(
          f"Sample {sample_name!r} token_types must contain only {{-1, 0, 1, 2, 3}}, "
          f"got {np.unique(raw_tt_arr).tolist()}"
      )
    tt_full = raw_tt_arr.astype(np.int32)
    if not is_mm_sample and np.any(tt_full >= 0):
      raise ValueError(
          f"Text-only sample {sample_name!r} has no images but declares image-span entries "
          f"(token_types >= 0): {np.unique(tt_full[tt_full >= 0]).tolist()}"
      )
  if is_mm_sample:
    if tt_full is None:
      raise ValueError(
          f"Multimodal sample {sample_name!r} has patches/encoder_images but is missing non-None token_types"
      )
    mm_mask = tt_full >= 0
    text_mask = tt_full == -1
    if not np.any(mm_mask):
      raise ValueError(f"Multimodal sample {sample_name!r} has images but no image-span entries (token_types >= 0)")
    if not np.any(text_mask) or not np.any(tt_full[start_index:token_size] == -1):
      raise ValueError(f"Multimodal sample {sample_name!r} has no evaluated text positions (token_types == -1)")
    if ds41_image_token_id is not None:
      tokens_1d = np.asarray(sample_tokens).reshape(-1)[:seq_len]
      expected_mm_mask = tokens_1d == int(ds41_image_token_id)
      if not np.array_equal(mm_mask, expected_mm_mask):
        mismatch_idx = np.flatnonzero(mm_mask != expected_mm_mask).tolist()
        raise ValueError(
            f"Multimodal sample {sample_name!r} token_types >= 0 mask disagrees with "
            f"tokens == ds41_image_token_id ({ds41_image_token_id}) at {len(mismatch_idx)} "
            f"positions (first 8: {mismatch_idx[:8]})"
        )
  return tt_full


def prepare_ds41_mm_batch_inputs(
    raw_patches: np.ndarray,
    global_batch_size: int,
    max_images_per_sample: int = 4,
    *,
    image_masks: np.ndarray | None = None,
    num_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Pads `[N_img, P, 588]` patches to `[B, max_images, P, 588]` (`float32`) + `[B, max_images]` (`int32`) masks."""
  patches_f32 = np.asarray(raw_patches, dtype=np.float32)
  if patches_f32.ndim != 3:
    raise ValueError(f"Expected 3D patches [N_img, P, 588], got {patches_f32.shape}")
  n_img, num_patches, patch_dim = patches_f32.shape
  if n_img < 1 or n_img > max_images_per_sample:
    raise ValueError(
        f"Sample has invalid n_img={n_img}; expected 1 <= n_img <= max_images_per_sample={max_images_per_sample}"
    )

  padded = np.zeros((max_images_per_sample, num_patches, patch_dim), dtype=np.float32)
  padded[:n_img] = patches_f32
  mask_1d = np.zeros((max_images_per_sample,), dtype=np.int32)

  if image_masks is not None:
    raw_mask_arr = np.asarray(image_masks).reshape(-1)
    if raw_mask_arr.shape[0] < 1 or raw_mask_arr.shape[0] > max_images_per_sample:
      raise ValueError(
          f"image_masks length {raw_mask_arr.shape[0]} must satisfy "
          f"1 <= len <= max_images_per_sample={max_images_per_sample}"
      )
    if not np.all((raw_mask_arr == 0) | (raw_mask_arr == 1)):
      raise ValueError(f"image_masks must be strictly binary in {{0, 1}}, got {raw_mask_arr.tolist()}")
    mask_1d[: raw_mask_arr.shape[0]] = raw_mask_arr.astype(np.int32)
    if n_img < max_images_per_sample and np.any(mask_1d[n_img:] != 0):
      raise ValueError(f"image_masks {mask_1d.tolist()} has active entries in zero-padded image slots >= n_img={n_img}")
    mask_sum = int(np.sum(mask_1d))
    if mask_sum < 1:
      raise ValueError(f"image_masks {mask_1d.tolist()} must activate at least 1 image slot (sum={mask_sum})")
    if num_images is not None:
      num_valid = int(num_images)
      if num_valid < 1 or num_valid > n_img:
        raise ValueError(f"Invalid num_images={num_valid} for n_img={n_img}")
      if mask_sum != num_valid:
        raise ValueError(
            f"Sum of image validity mask {mask_sum} ({mask_1d.tolist()}) does not match num_images={num_valid}"
        )
  elif num_images is not None:
    num_valid = int(num_images)
    if num_valid < 1 or num_valid > max_images_per_sample or num_valid > n_img:
      raise ValueError(f"Invalid num_images={num_valid} for n_img={n_img}, max_images_per_sample={max_images_per_sample}")
    mask_1d[:num_valid] = 1
  else:
    mask_1d[:n_img] = 1

  encoder_images = np.broadcast_to(
      padded[None, ...], (global_batch_size, max_images_per_sample, num_patches, patch_dim)
  ).copy()
  encoder_image_masks = np.broadcast_to(mask_1d[None, ...], (global_batch_size, max_images_per_sample)).copy()
  return encoder_images, encoder_image_masks


def prepare_ds41_mm_inputs(
    golden_data_point: Mapping[str, Any],
    global_batch_size: int,
    max_images_per_sample: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None]:
  """Extracts DS4.1 multimodal image patches and masks from `golden_data_point` and pads to `[B, max_images, ...]`."""
  if not has_ds41_mm_images(golden_data_point):
    return None, None
  raw_source = (
      golden_data_point["patches"]
      if ("patches" in golden_data_point and golden_data_point["patches"] is not None)
      else golden_data_point["encoder_images"]
  )
  explicit_mask_raw = None
  for mask_key in ("encoder_image_masks", "image_masks", "image_mask"):
    if mask_key in golden_data_point and golden_data_point[mask_key] is not None:
      explicit_mask_raw = np.asarray(golden_data_point[mask_key]).reshape(-1)
      break
  num_images_val = (
      int(np.asarray(golden_data_point["num_images"]).item())
      if ("num_images" in golden_data_point and golden_data_point["num_images"] is not None)
      else None
  )
  return prepare_ds41_mm_batch_inputs(
      np.asarray(raw_source, dtype=np.float32),
      global_batch_size,
      max_images_per_sample=max_images_per_sample,
      image_masks=explicit_mask_raw,
      num_images=num_images_val,
  )


def get_data_with_masks(golden_data_point, config, max_images_per_sample: int = 4):
  """Returns 8-tuple `(ids, decoder_segment_ids, decoder_positions, logits, seq_len, pixel_values, masks, audio_values)`."""
  model_prefix = config.model_name.split("-")[0]
  max_logging.log(f"config.global_batch_size_to_train_on={config.global_batch_size_to_train_on}")

  if getattr(config, "use_audio", False) and golden_data_point.get("audio_values") is not None:
    audio_values = np.asarray(golden_data_point["audio_values"], dtype=np.float32)
    audio_values = np.concatenate([audio_values] * config.global_batch_size_to_train_on, axis=0)
    max_logging.log(f"audio_values.shape = {audio_values.shape}")
  else:
    audio_values = None

  encoder_image_masks = None
  if has_ds41_mm_images(golden_data_point):
    if not config.use_multimodal:
      raise ValueError("Multimodal golden sample with 'patches'/'encoder_images' requires config.use_multimodal=True")
    pixel_values, encoder_image_masks = prepare_ds41_mm_inputs(
        golden_data_point,
        int(config.global_batch_size_to_train_on),
        max_images_per_sample=max_images_per_sample,
    )
  elif config.use_multimodal and "pixel_values" in golden_data_point:
    _, _, _, _, _, pixel_values, _ = fplc.get_data(golden_data_point, config)
  elif config.use_multimodal and audio_values is None and not model_prefix.startswith("deepseek4"):
    assert "pixel_values" in golden_data_point, "no image found in golden data while use_multimodal=True"
    pixel_values = None
  else:
    pixel_values = None

  original_ids = np.asarray(golden_data_point["tokens"], dtype=np.int32)
  seq_len = len(original_ids)
  if seq_len > config.max_target_length:
    raise ValueError(
        f"Golden data sequence length ({seq_len}) is greater than max_target_length ({config.max_target_length})"
    )

  s = (config.global_batch_size_to_train_on, config.max_target_length)
  padded_ids = np.pad(original_ids, (0, config.max_target_length - seq_len), "constant", constant_values=0)
  ids = np.stack([padded_ids for _ in range(config.global_batch_size_to_train_on)])
  logits = np.asarray(golden_data_point["logits"], dtype=np.float32)

  prompt = golden_data_point.get("formatted_prompt") or golden_data_point.get("prompt")
  if not prompt:
    prompt = resolve_sample_name(golden_data_point, f"sample_{seq_len}")
  max_logging.log(f' prompt="{prompt}" raw ids (first 20)={original_ids[:20]}, logits.shape = {logits.shape}')

  decoder_segment_ids = np.zeros(s, dtype=np.int32)
  if "segment_ids" in golden_data_point and golden_data_point["segment_ids"] is not None:
    raw_seg = np.asarray(golden_data_point["segment_ids"])
    if np.issubdtype(raw_seg.dtype, np.bool_) or not np.issubdtype(raw_seg.dtype, np.integer):
      raise ValueError(f"segment_ids must have integer dtype, got {raw_seg.dtype}")
    seg_1d = raw_seg.reshape(-1).astype(np.int32)
    if seg_1d.shape[0] != seq_len:
      raise ValueError(f"segment_ids length {seg_1d.shape[0]} != tokens length {seq_len}")
    decoder_segment_ids[:, :seq_len] = seg_1d[None, :]
  else:
    decoder_segment_ids[:, :seq_len] = DECODING_ACTIVE_SEQUENCE_INDICATOR

  if model_prefix in ["qwen3"] and config.use_mrope:
    from maxtext.multimodal import processor_qwen3_omni  # pylint: disable=import-outside-toplevel

    image_grid_thw = np.atleast_2d(golden_data_point["image_grid_thw"]) if "image_grid_thw" in golden_data_point else None
    video_grid_thw = np.atleast_2d(golden_data_point["video_grid_thw"]) if "video_grid_thw" in golden_data_point else None
    attention_mask = np.zeros(s, dtype=np.int32)
    attention_mask[:, :seq_len] = 1
    position_ids, _ = processor_qwen3_omni.get_rope_index(
        input_ids=ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=config.spatial_merge_size_for_vit,
        position_id_per_seconds=config.position_id_per_seconds,
        config=config,
    )
    decoder_positions = position_ids.astype(np.int32)
  else:
    decoder_positions = np.stack(
        [np.arange(config.max_target_length, dtype=np.int32) for _ in range(config.global_batch_size_to_train_on)]
    )
  return (
      ids,
      decoder_segment_ids,
      decoder_positions,
      logits,
      seq_len,
      pixel_values,
      encoder_image_masks,
      audio_values,
  )


def _validate_tie_flip_logits_in_record(out: dict[str, Any], sample_label: str) -> None:
  """Validates `tie_flip_logits` shape and finiteness, squeezing leading batch dim 1 when matching `logits`."""
  if "tie_flip_logits" not in out or out["tie_flip_logits"] is None:
    return
  raw_logits = np.asarray(out["logits"], dtype=np.float32)
  raw_tie_flip = np.asarray(out["tie_flip_logits"], dtype=np.float32)
  if raw_tie_flip.shape != raw_logits.shape:
    raise ValueError(
        f"Sample {sample_label!r}: tie_flip_logits shape {raw_tie_flip.shape} != logits shape {raw_logits.shape}"
    )
  if not np.all(np.isfinite(raw_tie_flip)):
    raise ValueError(f"Sample {sample_label!r}: tie_flip_logits contains non-finite (NaN/Inf) entries")
  if raw_logits.ndim == 3 and raw_logits.shape[0] == 1:
    out["logits"] = raw_logits[0]
    out["tie_flip_logits"] = raw_tie_flip[0]
  else:
    out["tie_flip_logits"] = raw_tie_flip


def normalize_golden_record(
    rec: Mapping[str, Any],
    fallback_label: str,
    *,
    require_frozen_topk: bool = False,
) -> dict[str, Any]:
  """Normalizes `name`/`sample_id`, validates optional tie-flip logits, and validates frozen-topk fields."""
  out: dict[str, Any] = {**rec}
  resolved = resolve_sample_name(out, fallback_label)
  if "name" not in out or out["name"] is None:
    out["name"] = resolved
  if "sample_id" not in out or out["sample_id"] is None:
    out["sample_id"] = resolved
  _validate_tie_flip_logits_in_record(out, resolved)
  tok_len = len(np.asarray(out["tokens"]).reshape(-1))
  out.update(
      deepseek_v41_frozen_topk.extract_and_validate_frozen_topk_fields(
          out,
          sample_name=resolved,
          expected_seq_len=tok_len,
          is_multimodal=has_ds41_mm_images(out),
          require_frozen_topk=require_frozen_topk,
      )
  )
  return out


def load_single_npz_golden(npz_path: Path, *, require_frozen_topk: bool = False) -> dict[str, Any]:
  """Loads a single `.npz` golden file (text-only or multimodal `L1..L4` / `M1..M8`) into a dict."""
  eff_req_ftk = bool(require_frozen_topk or deepseek_v41_frozen_topk.is_mm_v3_golden_path(npz_path))
  with np.load(npz_path, allow_pickle=False) as loaded:
    raw_logits = np.asarray(loaded["logits"], dtype=np.float32)
    squeeze_b1 = raw_logits.ndim == 3 and raw_logits.shape[0] == 1
    logits = raw_logits[0] if squeeze_b1 else raw_logits
    has_mm = ("patches" in loaded) or ("encoder_images" in loaded)
    sample_id = str(np.asarray(loaded["sample_id"]).item()) if "sample_id" in loaded else npz_path.stem
    prompt_str = str(np.asarray(loaded["prompt"]).item()) if "prompt" in loaded else sample_id
    diag_val = (
        bool(np.asarray(loaded["cos_sim_diagnostic_only"]).item())
        if "cos_sim_diagnostic_only" in loaded
        else (not has_mm and sample_id == "sample_6_verbatim_repeat_stress")
    )
    rec: dict[str, Any] = {
        "name": sample_id,
        "sample_id": sample_id,
        "prompt": prompt_str,
        "tokens": np.asarray(loaded["tokens"], dtype=np.int32).tolist(),
        "logits": logits,
        "cos_sim_diagnostic_only": diag_val,
    }
    if "tie_flip_logits" in loaded and loaded["tie_flip_logits"] is not None:
      raw_tie_flip = np.asarray(loaded["tie_flip_logits"], dtype=np.float32)
      if raw_tie_flip.shape != raw_logits.shape:
        raise ValueError(
            f"{npz_path.name}: tie_flip_logits shape {raw_tie_flip.shape} != logits shape {raw_logits.shape}"
        )
      if not np.all(np.isfinite(raw_tie_flip)):
        raise ValueError(f"{npz_path.name}: tie_flip_logits contains non-finite (NaN/Inf) entries")
      rec["tie_flip_logits"] = raw_tie_flip[0] if squeeze_b1 else raw_tie_flip

    for f32_key in ("patches", "encoder_images", "audio_values"):
      if f32_key in loaded:
        rec[f32_key] = np.asarray(loaded[f32_key], dtype=np.float32)
    for raw_key in ("image_masks", "encoder_image_masks", "image_mask", "token_types"):
      if raw_key in loaded:
        rec[raw_key] = np.asarray(loaded[raw_key])
    for i32_key in ("grid_hw", "grids", "segment_ids"):
      if i32_key in loaded:
        rec[i32_key] = np.asarray(loaded[i32_key], dtype=np.int32)
    if "num_images" in loaded:
      rec["num_images"] = int(np.asarray(loaded["num_images"]).item())

    raw_ftk: dict[str, Any] = {
        k: loaded[k]
        for k in deepseek_v41_frozen_topk.FROZEN_TOPK_REQUIRED_ROUTING_KEYS
        if k in loaded and loaded[k] is not None
    }
    rec.update(
        deepseek_v41_frozen_topk.extract_and_validate_frozen_topk_fields(
            raw_ftk,
            sample_name=sample_id,
            expected_seq_len=len(rec["tokens"]),
            is_multimodal=has_ds41_mm_images(rec),
            require_frozen_topk=eff_req_ftk,
        )
    )
    return rec


def load_golden_entries(path_str: str, *, require_frozen_topk: bool = False) -> list[dict[str, Any]]:
  """Loads golden entries from `.jsonl`, `.pickle`/`.pkl`, `.npz`, or a directory of `.npz` files."""
  p = Path(path_str)
  eff_req_ftk = bool(require_frozen_topk or deepseek_v41_frozen_topk.is_mm_v3_golden_path(path_str))
  if p.is_dir():
    npz_files = sorted(p.glob("*.npz"))
    if not npz_files:
      raise ValueError(f"No .npz files found in directory {path_str}")
    return [load_single_npz_golden(f, require_frozen_topk=eff_req_ftk) for f in npz_files]
  if p.suffix == ".jsonl":
    max_logging.log("loading hf goldens from jsonl file")
    with jsonlines.open(p, "r") as f:
      return [
          normalize_golden_record(item, f"sample_{idx}", require_frozen_topk=eff_req_ftk) for idx, item in enumerate(f)
      ]
  if p.suffix in (".pickle", ".pkl"):
    max_logging.log("loading hf goldens from pickle file")
    with open(p, "rb") as f:
      loaded_list = pickle.load(f)
    return [
        normalize_golden_record(item, f"sample_{idx}", require_frozen_topk=eff_req_ftk)
        for idx, item in enumerate(loaded_list)
    ]
  if p.suffix == ".npz":
    max_logging.log("loading hf golden from npz file")
    return [load_single_npz_golden(p, require_frozen_topk=eff_req_ftk)]
  raise ValueError("golden_logits_path must end with .jsonl, .pickle, .pkl, .npz, or be a directory of .npz files")


def extract_tie_flip_slice(
    golden_data_point: Mapping[str, Any],
    golden_logits: np.ndarray,
    start_index: int,
    token_size: int,
    min_vocab_size: int,
) -> np.ndarray | None:
  """Extracts, validates, and slices optional `tie_flip_logits` from a golden sample record."""
  if "tie_flip_logits" not in golden_data_point or golden_data_point["tie_flip_logits"] is None:
    return None
  tie_arr = np.asarray(golden_data_point["tie_flip_logits"], dtype=np.float32)
  if tie_arr.ndim == 3 and tie_arr.shape[0] == 1 and golden_logits.ndim == 2:
    tie_arr = tie_arr[0]
  if tuple(tie_arr.shape) != tuple(golden_logits.shape):
    raise ValueError(
        f"tie_flip_logits shape {tuple(tie_arr.shape)} must match golden_logits shape {tuple(golden_logits.shape)}"
    )
  if not bool(np.all(np.isfinite(tie_arr))):
    raise ValueError("tie_flip_logits contains non-finite values (NaN or Inf)")
  return tie_arr[start_index:token_size, :min_vocab_size]


def compute_scope_metrics(
    train_logits_2d: np.ndarray,
    golden_logits_2d: np.ndarray,
    clip_logits_epsilon: float | None = None,
    tie_flip_logits_2d: np.ndarray | None = None,
) -> dict[str, Any]:
  """Computes `max_kl`, `mean_kl`, `p95_kl`, `p99_kl`, `top1_agreement`, `last_token_top10_matches`, and cos/L_inf."""
  g_np = np.asarray(golden_logits_2d, dtype=np.float32)
  t_np = np.asarray(train_logits_2d, dtype=np.float32)
  seq_len = int(t_np.shape[0])
  vocab_size = int(t_np.shape[1])

  abs_diff = np.abs(t_np - g_np)
  l_inf = float(np.max(abs_diff))
  mean_abs = float(np.mean(abs_diff))

  dot = np.sum(t_np * g_np, axis=-1, dtype=np.float64)
  norm_t = np.linalg.norm(t_np, axis=-1).astype(np.float64)
  norm_g = np.linalg.norm(g_np, axis=-1).astype(np.float64)
  cos_per_tok = dot / (norm_t * norm_g + 1e-8)
  mean_cos = float(np.mean(cos_per_tok))
  min_cos = float(np.min(cos_per_tok))

  kl_primary = fplc.per_token_kl_2d(g_np, t_np, clip_logits_epsilon=clip_logits_epsilon)
  kl_primary_legacy_fp32 = fplc.per_token_kl_2d_legacy_fp32(g_np, t_np, clip_logits_epsilon=clip_logits_epsilon)
  raw_primary_max_kl = float(np.max(kl_primary))
  raw_primary_max_kl_legacy_fp32 = float(np.max(kl_primary_legacy_fp32))
  raw_primary_argmax_kl_pos = int(np.argmax(kl_primary))
  mean_kl = float(np.mean(kl_primary))
  mean_kl_legacy_fp32 = float(np.mean(kl_primary_legacy_fp32))
  p95_kl = float(np.percentile(kl_primary, 95))
  p95_kl_legacy_fp32 = float(np.percentile(kl_primary_legacy_fp32, 95))
  p99_kl = float(np.percentile(kl_primary, 99))
  p99_kl_legacy_fp32 = float(np.percentile(kl_primary_legacy_fp32, 99))

  if tie_flip_logits_2d is not None:
    tf_np = np.asarray(tie_flip_logits_2d, dtype=np.float32)
    if tf_np.shape != g_np.shape:
      raise ValueError(f"tie_flip_logits shape {tf_np.shape} does not match golden_logits shape {g_np.shape}")
    if not np.all(np.isfinite(tf_np)):
      raise ValueError("tie_flip_logits contains non-finite (NaN/Inf) entries")
    kl_tie_flip = fplc.per_token_kl_2d(tf_np, t_np, clip_logits_epsilon=clip_logits_epsilon)
    kl_tie_flip_legacy_fp32 = fplc.per_token_kl_2d_legacy_fp32(tf_np, t_np, clip_logits_epsilon=clip_logits_epsilon)
    kl_envelope = np.minimum(kl_primary, kl_tie_flip)
    kl_envelope_legacy_fp32 = np.minimum(kl_primary_legacy_fp32, kl_tie_flip_legacy_fp32)
    max_kl = float(np.max(kl_envelope))
    max_kl_legacy_fp32 = float(np.max(kl_envelope_legacy_fp32))
    argmax_kl_pos = int(np.argmax(kl_envelope))
    tie_flip_envelope_used = True
  else:
    max_kl = raw_primary_max_kl
    max_kl_legacy_fp32 = raw_primary_max_kl_legacy_fp32
    argmax_kl_pos = raw_primary_argmax_kl_pos
    tie_flip_envelope_used = False

  top1_agreement = float(np.sum(np.argmax(t_np, axis=-1) == np.argmax(g_np, axis=-1)) / seq_len)

  k_top = min(10, vocab_size)
  topk_t_idx = np.argpartition(t_np[-1], -k_top)[-k_top:]
  topk_g_idx = np.argpartition(g_np[-1], -k_top)[-k_top:]
  last_token_top10_matches = len(set(topk_t_idx.tolist()) & set(topk_g_idx.tolist()))

  return {
      "seq_len": seq_len,
      "max_kl": max_kl,
      "max_kl_legacy_fp32": max_kl_legacy_fp32,
      "raw_primary_max_kl": raw_primary_max_kl,
      "raw_primary_max_kl_legacy_fp32": raw_primary_max_kl_legacy_fp32,
      "raw_primary_argmax_kl_pos": raw_primary_argmax_kl_pos,
      "argmax_kl_pos": argmax_kl_pos,
      "tie_flip_envelope_used": tie_flip_envelope_used,
      "mean_kl": mean_kl,
      "mean_kl_legacy_fp32": mean_kl_legacy_fp32,
      "p95_kl": p95_kl,
      "p95_kl_legacy_fp32": p95_kl_legacy_fp32,
      "p99_kl": p99_kl,
      "p99_kl_legacy_fp32": p99_kl_legacy_fp32,
      "top1_agreement": top1_agreement,
      "last_token_top10_matches": int(last_token_top10_matches),
      "last_token_top_k": int(k_top),
      "mean_cos_sim": mean_cos,
      "min_cos_sim": min_cos,
      "l_inf": l_inf,
      "mean_abs_diff": mean_abs,
  }


def validate_gate_spec_entry(
    raw_spec: Any,
    *,
    section_path: str,
    expected_id: str | None = None,
) -> dict[str, Any]:
  """Validates a gate dictionary from `thresholds.json` (`max_kl > 0` finite and `last_token_top10_matches == 10`)."""
  if not isinstance(raw_spec, Mapping):
    raise ValueError(f"{section_path} must be a JSON object, got {type(raw_spec).__name__}")
  if "id" in raw_spec:
    raw_id = raw_spec["id"]
    if not isinstance(raw_id, str) or not raw_id:
      raise ValueError(f"{section_path}.id must be a non-empty string, got {raw_id!r}")
    if expected_id is not None and raw_id != expected_id:
      raise ValueError(f"{section_path}.id must be {expected_id!r}, got {raw_id!r}")

  if "max_kl" not in raw_spec:
    raise ValueError(f"{section_path} missing required 'max_kl'")
  raw_max_kl = raw_spec["max_kl"]
  if isinstance(raw_max_kl, bool) or not isinstance(raw_max_kl, (int, float)):
    raise ValueError(f"{section_path}.max_kl must be a finite positive number, got {raw_max_kl!r}")
  max_kl_val = float(raw_max_kl)
  if not math.isfinite(max_kl_val) or max_kl_val <= 0.0:
    raise ValueError(f"{section_path}.max_kl must be finite and > 0, got {raw_max_kl!r}")

  if "last_token_top10_matches" not in raw_spec:
    raise ValueError(f"{section_path} missing required 'last_token_top10_matches'")
  raw_top10 = raw_spec["last_token_top10_matches"]
  if isinstance(raw_top10, bool) or not isinstance(raw_top10, (int, np.integer)):
    raise ValueError(f"{section_path}.last_token_top10_matches must be integer 10, got {raw_top10!r}")
  if int(raw_top10) != _REQUIRED_TEXT_TOP10_MATCHES:
    raise ValueError(
        f"{section_path}.last_token_top10_matches must equal {_REQUIRED_TEXT_TOP10_MATCHES}, got {int(raw_top10)}"
    )

  for opt_kl_key in ("mean_kl", "p95_kl", "p99_kl"):
    if opt_kl_key in raw_spec:
      val = raw_spec[opt_kl_key]
      if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) <= 0.0:
        raise ValueError(f"{section_path}.{opt_kl_key} must be finite and > 0, got {val!r}")

  for opt_ratio_key in ("top1_agreement", "mean_cos_sim"):
    if opt_ratio_key in raw_spec:
      val = raw_spec[opt_ratio_key]
      if (
          isinstance(val, bool)
          or not isinstance(val, (int, float))
          or not math.isfinite(float(val))
          or not (0.0 < float(val) <= 1.0)
      ):
        raise ValueError(f"{section_path}.{opt_ratio_key} must be finite and in (0, 1], got {val!r}")

  return dict(raw_spec)


def validate_forward_logit_checker_thresholds(thresholds_cfg: Mapping[str, Any]) -> dict[str, Any]:
  """Validates `thresholds['forward_logit_checker']` (`short_text_prompts` and `long_text_prompts`)."""
  if not isinstance(thresholds_cfg, Mapping):
    raise ValueError(f"thresholds config must be a JSON object, got {type(thresholds_cfg).__name__}")
  if "forward_logit_checker" not in thresholds_cfg:
    raise ValueError("thresholds.json missing required 'forward_logit_checker' section for text-only logit gates")
  flc = thresholds_cfg["forward_logit_checker"]
  if not isinstance(flc, Mapping):
    raise ValueError(f"thresholds.json 'forward_logit_checker' must be a JSON object, got {type(flc).__name__}")
  raw_short_max = flc["short_text_max_seq_len"] if "short_text_max_seq_len" in flc else SHORT_TEXT_PROMPT_MAX_TOKENS
  if isinstance(raw_short_max, bool) or not isinstance(raw_short_max, (int, np.integer)) or int(raw_short_max) < 1:
    raise ValueError(f"forward_logit_checker.short_text_max_seq_len must be a positive integer, got {raw_short_max!r}")
  if "short_text_prompts" not in flc:
    raise ValueError("thresholds.json 'forward_logit_checker' missing required 'short_text_prompts'")
  if "long_text_prompts" not in flc:
    raise ValueError("thresholds.json 'forward_logit_checker' missing required 'long_text_prompts'")

  short_spec = validate_gate_spec_entry(
      flc["short_text_prompts"],
      section_path="forward_logit_checker.short_text_prompts",
      expected_id="short_text_prompts",
  )
  long_spec = validate_gate_spec_entry(
      flc["long_text_prompts"],
      section_path="forward_logit_checker.long_text_prompts",
      expected_id="long_text_prompts",
  )
  return {
      "short_text_max_seq_len": int(raw_short_max),
      "short_text_prompts": short_spec,
      "long_text_prompts": long_spec,
  }


def select_forward_threshold_spec(
    thresholds_cfg: Mapping[str, Any],
    *,
    sample_name: str,
    seq_len: int,
    is_mm_sample: bool,
) -> tuple[str, dict[str, Any]]:
  """Selects `short_text_prompts`, `long_text_prompts`, `mm_short_prompts`, or `mm_long_prompts` spec."""
  if not isinstance(thresholds_cfg, Mapping):
    raise ValueError(f"thresholds config must be a JSON object, got {type(thresholds_cfg).__name__}")
  if is_mm_sample:
    gate_id = (
        "mm_long_prompts"
        if sample_name in _DEFAULT_MULTIMODAL_LONG_IDS
        or (sample_name not in _DEFAULT_MULTIMODAL_SHORT_IDS and int(seq_len) > MM_SHORT_PROMPT_MAX_TOKENS)
        else "mm_short_prompts"
    )
    flc = thresholds_cfg.get("forward_logit_checker")
    if isinstance(flc, Mapping):
      mm_cases = flc.get("multimodal_cases")
      if isinstance(mm_cases, Mapping):
        if sample_name in mm_cases:
          return gate_id, validate_gate_spec_entry(
              mm_cases[sample_name],
              section_path=f"forward_logit_checker.multimodal_cases.{sample_name}",
              expected_id=gate_id,
          )
        if gate_id in mm_cases:
          return gate_id, validate_gate_spec_entry(
              mm_cases[gate_id],
              section_path=f"forward_logit_checker.multimodal_cases.{gate_id}",
              expected_id=gate_id,
          )
    if gate_id in thresholds_cfg:
      return gate_id, validate_gate_spec_entry(thresholds_cfg[gate_id], section_path=gate_id, expected_id=gate_id)
    raise ValueError(f"thresholds.json missing multimodal gate specification for {gate_id!r} (sample {sample_name!r})")

  flc_specs = validate_forward_logit_checker_thresholds(thresholds_cfg)
  if int(seq_len) <= int(flc_specs["short_text_max_seq_len"]):
    return "short_text_prompts", dict(flc_specs["short_text_prompts"])
  return "long_text_prompts", dict(flc_specs["long_text_prompts"])


def evaluate_thresholds(
    metrics: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    *,
    check_last_token_top10: bool = True,
) -> dict[str, bool]:
  """Evaluates a scope's metrics against a threshold spec from `thresholds.json`."""
  out: dict[str, bool] = {}
  for kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl"):
    if kl_key in gate_spec:
      out[f"{kl_key}_pass"] = bool(float(metrics[kl_key]) <= float(gate_spec[kl_key]))
  for ratio_key in ("top1_agreement", "mean_cos_sim"):
    if ratio_key in gate_spec:
      out[f"{ratio_key}_pass"] = bool(float(metrics[ratio_key]) >= float(gate_spec[ratio_key]))
  if check_last_token_top10 and "last_token_top10_matches" in gate_spec:
    req_matches = int(gate_spec["last_token_top10_matches"])
    out["last_token_top10_matches_pass"] = bool(int(metrics["last_token_top10_matches"]) >= req_matches)
  return out


def format_gate_failures(
    scope_label: str,
    metrics: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    gates: Mapping[str, bool],
    *,
    max_kl_index: int | None = None,
) -> list[str]:
  """Formats human-readable measured-vs-required failure strings for any failed threshold checks."""
  reasons: list[str] = []
  for kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl"):
    pass_key = f"{kl_key}_pass"
    if pass_key in gates and not gates[pass_key]:
      idx_suffix = f" at index {max_kl_index}" if kl_key == "max_kl" and max_kl_index is not None else ""
      reasons.append(f"{scope_label}.{kl_key}={float(metrics[kl_key]):.8g}{idx_suffix} > {float(gate_spec[kl_key]):.8g}")
  for ratio_key in ("top1_agreement", "mean_cos_sim"):
    pass_key = f"{ratio_key}_pass"
    if pass_key in gates and not gates[pass_key]:
      reasons.append(f"{scope_label}.{ratio_key}={float(metrics[ratio_key]):.6f} < {float(gate_spec[ratio_key]):.6f}")
  if "last_token_top10_matches_pass" in gates and not gates["last_token_top10_matches_pass"]:
    reasons.append(
        f"{scope_label}.last_token_top10_matches={int(metrics['last_token_top10_matches'])} "
        f"< {int(gate_spec['last_token_top10_matches'])}"
    )
  return reasons


@dataclasses.dataclass
class FrozenTopkTraceResult:
  """Container for natural-routing trace and router/indexer agreement diagnostics."""

  natural_raw_logits_slice: np.ndarray
  router_agreement: dict[str, Any]
  unforced_router_agreement_summary: dict[str, Any]


def check_frozen_topk_prereqs(
    golden_data_point: Mapping[str, Any],
    *,
    sample_name: str,
    is_mm_sample: bool,
    state: Any,
    require_frozen_topk_gate: bool,
) -> bool:
  """Validates whether frozen-topk forced routing applies for `golden_data_point` and checks missing arrays."""
  any_extra = deepseek_v41_frozen_topk.any_frozen_topk_extra(golden_data_point)
  use_frozen_topk = bool(
      is_mm_sample and state is None and deepseek_v41_frozen_topk.has_frozen_topk_routing_fields(golden_data_point)
  )
  if is_mm_sample and (require_frozen_topk_gate or any_extra) and not use_frozen_topk:
    missing_fields = [
        k
        for k in deepseek_v41_frozen_topk.FROZEN_TOPK_REQUIRED_ROUTING_KEYS
        if k not in golden_data_point or golden_data_point[k] is None
    ]
    raise ValueError(
        f"Multimodal sample {sample_name!r}: FrozenTopK gate requires all 7 routing arrays "
        f"{deepseek_v41_frozen_topk.FROZEN_TOPK_REQUIRED_ROUTING_KEYS}; missing {missing_fields}."
    )
  if use_frozen_topk:
    missing_idx_fields = [
        k
        for k in ("ref_indexer_layer_ids", "ref_indexer_topk", "ref_indexer_margins", "ref_indexer_boundary_ulp")
        if k not in golden_data_point or golden_data_point[k] is None
    ]
    if missing_idx_fields:
      raise ValueError(
          f"Multimodal sample {sample_name!r}: FrozenTopK gate requires all 4 indexer routing fields; "
          f"missing {missing_idx_fields}"
      )
  return use_frozen_topk


def compute_router_agreement_pair(
    *,
    golden_data_point: Mapping[str, Any],
    sample_name: str,
    seq_len: int,
    tt_full: np.ndarray | None,
    nat_res: Any,
    forced_captured: Any,
    router_ulp_multiplier: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Computes unforced and forced-run router/indexer agreement diagnostics from natural and forced captures."""
  if router_ulp_multiplier > 0 and (
      forced_captured is None
      or not hasattr(forced_captured, "tpu_moe_topk")
      or forced_captured.tpu_moe_topk is None
      or not hasattr(forced_captured, "tpu_indexer_topk")
      or forced_captured.tpu_indexer_topk is None
  ):
    raise ValueError(
        f"Multimodal sample {sample_name!r}: FrozenTopK router-agreement diagnostic requires "
        "captured pre-override tpu_moe_topk and tpu_indexer_topk arrays when "
        f"router_agreement_ulp_multiplier={router_ulp_multiplier} > 0."
    )
  ra_kwargs = {
      "sample_name": sample_name,
      "seq_len": seq_len,
      "token_types": tt_full,
      "ref_topk_experts": golden_data_point["ref_topk_experts"],
      "ref_gate_margins": golden_data_point["ref_gate_margins"],
      "ref_gate_boundary_ulp": golden_data_point["ref_gate_boundary_ulp"],
      "ref_indexer_layer_ids": golden_data_point["ref_indexer_layer_ids"],
      "ref_indexer_topk": golden_data_point["ref_indexer_topk"],
      "ref_indexer_margins": golden_data_point["ref_indexer_margins"],
      "ref_indexer_boundary_ulp": golden_data_point["ref_indexer_boundary_ulp"],
      "ulp_multiplier": router_ulp_multiplier,
      "raise_on_failure": False,
  }
  unforced_ra = deepseek_v41_frozen_topk.check_router_agreement(
      tpu_moe_topk=nat_res.tpu_moe_topk,
      tpu_indexer_topk=nat_res.tpu_indexer_topk,
      **ra_kwargs,
  )
  unforced_summary = {
      "passed": bool(unforced_ra["passed"]),
      "diagnostic_only": True,
      "ulp_multiplier": float(unforced_ra["ulp_multiplier"]),
      "high_margin_moe_failure_count": int(unforced_ra["high_margin_moe_failure_count"]),
      "low_margin_moe_flip_count": int(unforced_ra["low_margin_moe_flip_count"]),
      "high_margin_indexer_failure_count": int(unforced_ra["high_margin_indexer_failure_count"]),
      "low_margin_indexer_flip_count": int(unforced_ra["low_margin_indexer_flip_count"]),
      "moe_router": dict(unforced_ra["moe_router"]),
      "sparse_attention_indexer": dict(unforced_ra["sparse_attention_indexer"]),
  }
  router_agreement = deepseek_v41_frozen_topk.check_router_agreement(
      tpu_moe_topk=forced_captured.tpu_moe_topk,
      tpu_indexer_topk=forced_captured.tpu_indexer_topk,
      **ra_kwargs,
  )
  router_agreement["diagnostic_only"] = True
  router_log_line = f"DS41_MM_ROUTER_AGREEMENT: {json.dumps({'sample': sample_name, **router_agreement})}"
  max_logging.log(router_log_line)
  if jax.process_index() == 0:
    print(router_log_line, flush=True)
  return router_agreement, unforced_summary


def log_frozen_topk_diagnostic(
    *,
    sample_name: str,
    nat_text_metrics: Mapping[str, Any],
    forced_text_metrics: Mapping[str, Any],
    unforced_router_agreement_summary: Mapping[str, Any],
    router_agreement: Mapping[str, Any],
) -> None:
  """Logs diagnostic comparisons between natural routing and forced-topk forward passes."""
  if not bool(router_agreement["passed"]):
    diag_reasons = "; ".join(deepseek_v41_frozen_topk.format_router_agreement_failure_reasons(router_agreement))
    router_diag_line = f"[DIAGNOSTIC] sample={sample_name} E2E router_agreement: {diag_reasons}"
    max_logging.log(router_diag_line)
    if jax.process_index() == 0:
      print(router_diag_line, flush=True)
  diag_line = (
      f"DS41_MM_FROZEN_TOPK_DIAGNOSTIC: sample={sample_name} "
      f"unforced_text_max_kl={float(nat_text_metrics['max_kl']):.8g} -> "
      f"forced_text_max_kl={float(forced_text_metrics['max_kl']):.8g}, "
      f"unforced_text_mean_kl={float(nat_text_metrics['mean_kl']):.8g} -> "
      f"forced_text_mean_kl={float(forced_text_metrics['mean_kl']):.8g} "
      f"unforced_router_agreement_summary={json.dumps(unforced_router_agreement_summary)}"
  )
  max_logging.log(diag_line)
  if jax.process_index() == 0:
    print(diag_line, flush=True)


def _write_summary_json_file(summary_json_path: str, summary: dict) -> None:
  """Writes summary dictionary to disk if `summary_json_path` is specified and `process_index == 0`."""
  if jax.process_index() == 0 and summary_json_path:
    s_dir = os.path.dirname(summary_json_path)
    if s_dir:
      os.makedirs(s_dir, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
      json.dump(summary, f, indent=2)


def _allgather_train_logits(raw_logits: Any, seq_len: int, max_target_length: int, vocab_size: int) -> np.ndarray:
  """Allgathers batch row 0 logits onto host numpy without materializing unnecessary batch copies."""
  sliced = raw_logits[:1]
  gathered = jax.experimental.multihost_utils.process_allgather(sliced, tiled=True)
  if gathered.ndim == 4:
    gathered = jnp.reshape(gathered, (-1, max_target_length, vocab_size))
  return np.asarray(gathered[:, :seq_len, :])


# pylint: disable=too-many-locals,too-many-branches,too-many-statements,redefined-outer-name
def _run_golden_comparison(config, test_args, model, mesh, tokenizer) -> dict[str, Any]:
  """Executes golden-file forward parity checks, threshold evaluation, and summary persistence."""
  run_ds41_mm_audit = bool(getattr(test_args, "run_ds41_mm_audit", False))
  hf_ckpt_for_audit = str(getattr(test_args, "hf_checkpoint_dir_for_audit", ""))
  export_orbax_dir = str(getattr(test_args, "export_orbax_dir", ""))
  mm_golden_logits_path = str(getattr(test_args, "mm_golden_logits_path", ""))
  thresholds_json_path = str(getattr(test_args, "thresholds_json", ""))
  max_images_per_sample = int(getattr(test_args, "max_images_per_sample", 4))
  summary_json_path = str(getattr(test_args, "summary_json", ""))
  enable_long_text_gate = bool(getattr(test_args, "enable_long_text_gate", False))
  require_frozen_topk_gate = bool(
      getattr(test_args, "require_frozen_topk_gate", False)
      or deepseek_v41_frozen_topk.is_mm_v3_golden_path(test_args.golden_logits_path)
      or deepseek_v41_frozen_topk.is_mm_v3_golden_path(mm_golden_logits_path)
  )
  router_ulp_multiplier = float(
      getattr(
          test_args,
          "router_agreement_ulp_multiplier",
          deepseek_v41_frozen_topk.DEFAULT_ROUTER_AGREEMENT_ULP_MULTIPLIER,
      )
  )

  thresholds_cfg: dict[str, Any] | None = None
  if thresholds_json_path:
    with open(thresholds_json_path, "r", encoding="utf-8") as f:
      loaded_thresholds = json.load(f)
    if not isinstance(loaded_thresholds, Mapping):
      raise ValueError(f"--thresholds_json must contain a JSON object, got {type(loaded_thresholds).__name__}")
    thresholds_cfg = dict(loaded_thresholds)

  audit_result = None
  if run_ds41_mm_audit or export_orbax_dir:
    from tests.utils import deepseek_v41_mm_audit  # pylint: disable=import-outside-toplevel

    audit_dir = hf_ckpt_for_audit if hf_ckpt_for_audit else str(config.load_parameters_path)
    try:
      audit_result = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
          model=model,
          config=config,
          hf_checkpoint_dir=audit_dir,
          mesh=mesh,
          verify_all_text_leaves=True,
          export_orbax_dir=export_orbax_dir if export_orbax_dir else None,
      )
    except RuntimeError as exc:
      if "HF-to-MaxText audit failed" not in str(exc):
        raise
      audit_result = {"passed": False, "error": str(exc)}
    max_logging.log(f"DS41_MM_AUDIT_RESULT: {json.dumps(audit_result)}")
    if not audit_result["passed"]:
      fail_summary = {
          "passed": False,
          "all_passed": False,
          "failure_reason": f"HF-to-MaxText pre-forward audit failed: {audit_result}",
          "audit_result": audit_result,
          "num_samples": 0,
          "num_passed": 0,
          "samples": [],
      }
      _write_summary_json_file(summary_json_path, fail_summary)
      raise DeterministicGateError(f"HF-to-MaxText pre-forward audit failed: {audit_result}")

  golden_data: list[dict[str, Any]] = []
  if test_args.golden_logits_path == "" and mm_golden_logits_path == "":
    default_path = os.path.join(MAXTEXT_TEST_ASSETS_ROOT, "golden_logits", f"golden_data_{config.model_name}.jsonl")
    golden_data.extend(load_golden_entries(default_path, require_frozen_topk=require_frozen_topk_gate))
  else:
    if test_args.golden_logits_path != "":
      golden_data.extend(load_golden_entries(test_args.golden_logits_path, require_frozen_topk=require_frozen_topk_gate))
    if mm_golden_logits_path != "":
      golden_data.extend(load_golden_entries(mm_golden_logits_path, require_frozen_topk=require_frozen_topk_gate))

  max_logging.log(f"loaded {len(golden_data)} golden data points")
  if thresholds_cfg is not None:
    if not golden_data:
      raise ValueError("--thresholds_json specified, but 0 golden samples were loaded.")
    if any(not has_ds41_mm_images(dp) for dp in golden_data):
      validate_forward_logit_checker_thresholds(thresholds_cfg)

  eval_step_nnx = None
  nnx_state = None
  all_data_to_save = []
  sample_reports = []
  failed_samples = []

  for golden_data_index, golden_data_point in enumerate(golden_data):
    sample_name = resolve_sample_name(golden_data_point, f"sample_{golden_data_index}")
    max_logging.log(f"\n--- Comparing forward pass for golden data index: {golden_data_index} ({sample_name}) ---")
    ids, decoder_segment_ids, decoder_positions, golden_logits, seq_len, images, image_masks, audios = (
        get_data_with_masks(golden_data_point, config, max_images_per_sample=max_images_per_sample)
    )
    token_size = int(test_args.token_size) if test_args.token_size else seq_len
    start_index = 1 if test_args.skip_first_token else 0

    tie_flip_full = None
    if "tie_flip_logits" in golden_data_point and golden_data_point["tie_flip_logits"] is not None:
      tie_flip_full = np.asarray(golden_data_point["tie_flip_logits"], dtype=np.float32)
      if tie_flip_full.shape != golden_logits.shape:
        raise ValueError(
            f"Sample {sample_name!r} tie_flip_logits shape {tie_flip_full.shape} "
            f"!= golden_logits shape {golden_logits.shape}"
        )
      if not np.all(np.isfinite(tie_flip_full)):
        raise ValueError(f"Sample {sample_name!r} tie_flip_logits contains non-finite (NaN/Inf) entries")

    is_mm_sample = has_ds41_mm_images(golden_data_point)
    tt_full = validate_sample_token_types(
        golden_data_point,
        sample_name=sample_name,
        sample_tokens=ids[0, :seq_len],
        seq_len=seq_len,
        start_index=start_index,
        token_size=token_size,
        is_mm_sample=is_mm_sample,
        ds41_image_token_id=resolve_ds41_image_token_id(config),
    )

    gate_id = ""
    gate_spec: dict[str, Any] | None = None
    if thresholds_cfg is not None:
      gate_id, gate_spec = select_forward_threshold_spec(
          thresholds_cfg,
          sample_name=sample_name,
          seq_len=seq_len,
          is_mm_sample=is_mm_sample,
      )
      if gate_id == "long_text_prompts" and not enable_long_text_gate:
        defer_line = (
            f"[long_text_prompts] sample={sample_name} (index={golden_data_index}, true_length={int(seq_len)}): "
            f"{DEFERRED_LONG_TEXT_STATUS}"
        )
        max_logging.log(defer_line)
        if jax.process_index() == 0:
          print(defer_line, flush=True)
        sample_reports.append(
            {
                "index": golden_data_index,
                "name": sample_name,
                "is_multimodal": False,
                "gate": "long_text_prompts",
                "gate_id": "long_text_prompts",
                "tier": "long_text_prompts",
                "true_length": int(seq_len),
                "status": DEFERRED_LONG_TEXT_STATUS,
            }
        )
        continue

    max_logging.log("maxtext forward pass")
    use_frozen_topk = check_frozen_topk_prereqs(
        golden_data_point,
        sample_name=sample_name,
        is_mm_sample=is_mm_sample,
        state=None,
        require_frozen_topk_gate=require_frozen_topk_gate,
    )
    natural_raw_logits_slice_np: np.ndarray | None = None
    router_agreement: dict[str, Any] | None = None
    unforced_router_agreement_summary: dict[str, Any] | None = None

    if use_frozen_topk:
      nat_res = deepseek_v41_frozen_topk.run_natural_routing_forward_with_trace_nnx(
          model,
          config,
          mesh,
          sample_name=sample_name,
          ids=ids,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          images=images,
          image_masks=image_masks,
      )
      nat_full = _allgather_train_logits(nat_res.logits, seq_len, config.max_target_length, config.vocab_size)
      nat_min_vocab = min(int(nat_full.shape[-1]), int(golden_logits.shape[-1]))
      natural_raw_logits_slice_np = nat_full[0, start_index:token_size, :nat_min_vocab]

      raw_train_logits, forced_captured = deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx(
          model,
          config,
          mesh,
          golden_data_point["ref_topk_experts"],
          sample_name=sample_name,
          ids=ids,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          images=images,
          image_masks=image_masks,
          seq_len=seq_len,
          raw_indexer_topk=golden_data_point["ref_indexer_topk"],
          raw_indexer_layer_ids=golden_data_point["ref_indexer_layer_ids"],
          return_captured=True,
      )
      router_agreement, unforced_router_agreement_summary = compute_router_agreement_pair(
          golden_data_point=golden_data_point,
          sample_name=sample_name,
          seq_len=seq_len,
          tt_full=tt_full,
          nat_res=nat_res,
          forced_captured=forced_captured,
          router_ulp_multiplier=router_ulp_multiplier,
      )
    else:
      if eval_step_nnx is None:
        eval_step_nnx, nnx_state = fplc.make_eval_step_nnx(model, mesh, getattr(config, "logical_axis_rules", None))
      raw_train_logits = eval_step_nnx(
          nnx_state,
          ids,
          decoder_positions,
          decoder_segment_ids,
          images,
          image_masks,
          audios,
      )

    full_train_logits = _allgather_train_logits(raw_train_logits, seq_len, config.max_target_length, config.vocab_size)
    if full_train_logits.shape[-1] != golden_logits.shape[-1]:
      max_logging.log(
          f"Vocab size mismatch: train logits vocab size {full_train_logits.shape[-1]}, "
          f"golden logits vocab size {golden_logits.shape[-1]}. "
          "Comparing up to the smaller vocab size."
      )
    min_vocab_size = min(full_train_logits.shape[-1], golden_logits.shape[-1])
    train_logits_slice = full_train_logits[0, start_index:token_size, :min_vocab_size]
    golden_logits_slice = golden_logits[start_index:token_size, :min_vocab_size]
    tie_flip_logits_slice = tie_flip_full[start_index:token_size, :min_vocab_size] if tie_flip_full is not None else None

    if train_logits_slice.shape[0] > 2:
      max_logging.log(f"\n[logits: token {start_index + 2}]")
      max_logging.log(f"{golden_logits_slice[2]=}")
      max_logging.log(f"{train_logits_slice[2]=}")

    abs_diff = np.abs(train_logits_slice - golden_logits_slice)
    safe_golden = np.where(golden_logits_slice == 0, 1e-8, golden_logits_slice)
    rel_diff = abs_diff / np.abs(safe_golden)
    max_abs_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    max_rel_diff_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
    max_logging.log(
        "\n[numerical difference]\n"
        f"Max absolute difference: {abs_diff[max_abs_diff_idx]:.2e} at index {max_abs_diff_idx}\n"
        f"  (Train: {train_logits_slice[max_abs_diff_idx]:.2e}, Golden: {golden_logits_slice[max_abs_diff_idx]:.2e})\n"
        f"Max relative difference: {rel_diff[max_rel_diff_idx]:.2e} at index {max_rel_diff_idx}\n"
        f"  (Train: {train_logits_slice[max_rel_diff_idx]:.2e}, Golden: {golden_logits_slice[max_rel_diff_idx]:.2e})"
    )

    if convert_jax_weight_to_torch is not None and torch is not None:
      mt_last = torch.tensor(np.asarray(train_logits_slice[-1:, :], dtype=np.float32))
      hf_last = torch.tensor(np.asarray(golden_logits_slice[-1:, :], dtype=np.float32))
    else:
      mt_last = np.asarray(train_logits_slice[-1:, :])
      hf_last = np.asarray(golden_logits_slice[-1:, :])
    tokens_maxtext = fplc.get_top_k_tokens_scores(mt_last, tokenizer, k=10, description="MaxText model")
    tokens_hf = fplc.get_top_k_tokens_scores(hf_last, tokenizer, k=10, description="HF model")
    fplc.compare_top_tokens(converted_tokens=tokens_maxtext, golden_tokens=tokens_hf)

    kl_div = fplc.per_token_kl_2d(
        golden_logits_slice, train_logits_slice, clip_logits_epsilon=test_args.clip_logits_epsilon
    )
    if tie_flip_logits_slice is not None:
      kl_div_tf = fplc.per_token_kl_2d(
          tie_flip_logits_slice, train_logits_slice, clip_logits_epsilon=test_args.clip_logits_epsilon
      )
      kl_div = np.minimum(kl_div, kl_div_tf)

    if getattr(config, "use_multimodal", False) and "qwen3" in config.model_name.lower():
      from maxtext.multimodal.processor_qwen3_omni import QwenTokens  # pylint: disable=import-outside-toplevel

      qwen_tokens = QwenTokens(config)
      ignore_ids = [
          qwen_tokens.vision_start,
          qwen_tokens.vision_end,
          qwen_tokens.image_pad,
          qwen_tokens.video_pad,
      ]
      slice_ids = ids[0, start_index:token_size]
      qwen_mask = np.ones_like(slice_ids, dtype=bool)
      for ignore_id in ignore_ids:
        qwen_mask &= slice_ids != ignore_id
      kl_div = np.where(qwen_mask, kl_div, 0.0)

    if is_mm_sample and tt_full is not None:
      seg_slice_for_kl = np.asarray(decoder_segment_ids[0, start_index:token_size], dtype=np.int32)
      mm_text_only_mask = (seg_slice_for_kl > 0) & (tt_full[start_index:token_size] == -1)
      kl_div = np.where(mm_text_only_mask, kl_div, 0.0)

    max_kl_div_val = float(np.max(kl_div))
    max_kl_div_idx = int(np.argmax(kl_div))
    per_token_kl = "" if config.use_multimodal else f"KL divergence = {kl_div}, "
    max_logging.log(
        f"\n[KL divergence]\n"
        f"{per_token_kl} max KL divergence = {max_kl_div_val} at index {max_kl_div_idx}, "
        f"the corresponding token id is {ids[0, max_kl_div_idx + start_index]}"
    )

    if jax.process_index() == 0 and test_args.output_logits_path:
      all_data_to_save.append(
          {
              "prompt": golden_data_point.get("prompt", sample_name),
              "tokens": ids[0, :seq_len].tolist(),
              "logits": full_train_logits[0].tolist(),
          }
      )

    seg_slice = np.asarray(decoder_segment_ids[0, start_index:token_size], dtype=np.int32)
    valid_mask = seg_slice > 0
    if not bool(np.any(valid_mask)):
      raise ValueError(f"Sample {sample_name!r} has zero non-padding positions in slice [{start_index}:{token_size}]")

    text_pos_metrics: dict[str, Any] | None = None
    image_pos_metrics: dict[str, Any] | None = None
    text_mask: np.ndarray | None = None
    image_mask_pos: np.ndarray | None = None
    tie_flip_text: np.ndarray | None = None

    if is_mm_sample:
      assert tt_full is not None
      tt_slice = tt_full[start_index:token_size]
      text_mask = valid_mask & (tt_slice == -1)
      image_mask_pos = valid_mask & (tt_slice != -1)
      if not bool(np.any(text_mask)):
        raise ValueError(f"Multimodal sample {sample_name!r} has zero valid text_positions (token_types == -1)")
      if not bool(np.any(image_mask_pos)):
        raise ValueError(f"Multimodal sample {sample_name!r} has zero valid image_positions (token_types != -1)")
      image_token_count = int(np.sum(image_mask_pos))
      text_token_count = int(np.sum(text_mask))
      tie_flip_text = np.asarray(tie_flip_logits_slice)[text_mask] if tie_flip_logits_slice is not None else None
      all_pos_metrics = compute_scope_metrics(
          train_logits_slice[valid_mask],
          golden_logits_slice[valid_mask],
          clip_logits_epsilon=test_args.clip_logits_epsilon,
          tie_flip_logits_2d=None,
      )
      text_pos_metrics = compute_scope_metrics(
          train_logits_slice[text_mask],
          golden_logits_slice[text_mask],
          clip_logits_epsilon=test_args.clip_logits_epsilon,
          tie_flip_logits_2d=tie_flip_text,
      )
      image_pos_metrics = compute_scope_metrics(
          train_logits_slice[image_mask_pos],
          golden_logits_slice[image_mask_pos],
          clip_logits_epsilon=test_args.clip_logits_epsilon,
          tie_flip_logits_2d=None,
      )
      text_seq_indices = np.flatnonzero(text_mask) + start_index
      text_max_kl_index: int | None = int(text_seq_indices[int(text_pos_metrics["argmax_kl_pos"])])
      max_kl_index = text_max_kl_index
    else:
      text_max_kl_index = None
      tie_flip_valid = np.asarray(tie_flip_logits_slice)[valid_mask] if tie_flip_logits_slice is not None else None
      all_pos_metrics = compute_scope_metrics(
          train_logits_slice[valid_mask],
          golden_logits_slice[valid_mask],
          clip_logits_epsilon=test_args.clip_logits_epsilon,
          tie_flip_logits_2d=tie_flip_valid,
      )
      if tt_full is not None:
        tt_slice = tt_full[start_index:token_size]
        text_mask_non_mm = valid_mask & (tt_slice == -1)
        image_token_count = int(np.sum(valid_mask & (tt_slice != -1)))
        text_token_count = int(np.sum(text_mask_non_mm))
        if bool(np.any(text_mask_non_mm)):
          tie_flip_non_mm = (
              np.asarray(tie_flip_logits_slice)[text_mask_non_mm] if tie_flip_logits_slice is not None else None
          )
          text_pos_metrics = compute_scope_metrics(
              train_logits_slice[text_mask_non_mm],
              golden_logits_slice[text_mask_non_mm],
              clip_logits_epsilon=test_args.clip_logits_epsilon,
              tie_flip_logits_2d=tie_flip_non_mm,
          )
          text_non_mm_indices = np.flatnonzero(text_mask_non_mm) + start_index
          text_max_kl_index = int(text_non_mm_indices[int(text_pos_metrics["argmax_kl_pos"])])
      else:
        image_token_count = 0
        text_token_count = int(np.sum(valid_mask))
      valid_seq_indices = np.flatnonzero(valid_mask) + start_index
      max_kl_index = int(valid_seq_indices[int(all_pos_metrics["argmax_kl_pos"])])

    sample_passed = True
    gates_all: dict[str, bool] = {}
    gates_text: dict[str, bool] = {}
    failure_reasons: list[str] = []

    if gate_spec is not None:
      gates_all = evaluate_thresholds(all_pos_metrics, gate_spec, check_last_token_top10=True)
      if is_mm_sample:
        assert text_pos_metrics is not None and text_max_kl_index is not None
        gates_text = evaluate_thresholds(text_pos_metrics, gate_spec, check_last_token_top10=True)
        failure_reasons.extend(
            format_gate_failures(
                "text_positions",
                text_pos_metrics,
                gate_spec,
                gates_text,
                max_kl_index=text_max_kl_index,
            )
        )
        sample_passed = bool(all(gates_text.values()))
      else:
        failure_reasons.extend(
            format_gate_failures(
                "all_positions",
                all_pos_metrics,
                gate_spec,
                gates_all,
                max_kl_index=max_kl_index,
            )
        )
        if text_pos_metrics is not None:
          gates_text = evaluate_thresholds(text_pos_metrics, gate_spec, check_last_token_top10=False)
          failure_reasons.extend(
              format_gate_failures(
                  "text_positions",
                  text_pos_metrics,
                  gate_spec,
                  gates_text,
                  max_kl_index=text_max_kl_index,
              )
          )
          sample_passed = bool(all(gates_all.values()) and all(gates_text.values()))
        else:
          sample_passed = bool(all(gates_all.values()))

    if test_args.atol is not None:
      rtol_val = float(test_args.rtol)
      atol_val = float(test_args.atol)
      atol_mask = text_mask if (is_mm_sample and text_mask is not None) else valid_mask
      atol_train = train_logits_slice[atol_mask]
      atol_golden = golden_logits_slice[atol_mask]
      if not bool(np.allclose(atol_train, atol_golden, rtol=rtol_val, atol=atol_val, equal_nan=False)):
        sample_passed = False
        atol_max_diff = float(np.max(np.abs(atol_train - atol_golden)))
        failure_reasons.append(f"allclose(max_abs_diff={atol_max_diff:.6g}) failed rtol={rtol_val}, atol={atol_val}")

    if test_args.max_kl_div is not None:
      if is_mm_sample:
        assert text_pos_metrics is not None
        mm_max_kl = float(text_pos_metrics["max_kl"])
        if mm_max_kl >= float(test_args.max_kl_div):
          sample_passed = False
          failure_reasons.append(f"text_positions.max_kl={mm_max_kl:.8g} >= {float(test_args.max_kl_div):.8g}")
      elif not bool(np.all(kl_div < test_args.max_kl_div)):
        sample_passed = False
        failure_reasons.append(f"max_kl_div={max_kl_div_val:.8g} >= {float(test_args.max_kl_div):.8g}")

    frozen_topk_diagnostic: dict[str, Any] | None = None
    natural_routing_metrics: dict[str, Any] | None = None
    if is_mm_sample:
      assert text_pos_metrics is not None and image_pos_metrics is not None
      assert text_mask is not None and image_mask_pos is not None
      if use_frozen_topk:
        assert natural_raw_logits_slice_np is not None and router_agreement is not None
        assert unforced_router_agreement_summary is not None
        nat_text_metrics = compute_scope_metrics(
            natural_raw_logits_slice_np[text_mask],
            golden_logits_slice[text_mask],
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            tie_flip_logits_2d=tie_flip_text,
        )
        nat_image_metrics = compute_scope_metrics(
            natural_raw_logits_slice_np[image_mask_pos],
            golden_logits_slice[image_mask_pos],
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            tie_flip_logits_2d=None,
        )
        nat_all_metrics = compute_scope_metrics(
            natural_raw_logits_slice_np[valid_mask],
            golden_logits_slice[valid_mask],
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            tie_flip_logits_2d=None,
        )
        natural_routing_metrics = {
            "text_positions_metrics": nat_text_metrics,
            "image_positions_metrics": nat_image_metrics,
            "all_positions_metrics": nat_all_metrics,
        }
        log_frozen_topk_diagnostic(
            sample_name=sample_name,
            nat_text_metrics=nat_text_metrics,
            forced_text_metrics=text_pos_metrics,
            unforced_router_agreement_summary=unforced_router_agreement_summary,
            router_agreement=router_agreement,
        )
        frozen_topk_diagnostic = {
            "ran": True,
            "mode": "frozen_topk_primary_gate",
            "text_positions_metrics": text_pos_metrics,
            "image_positions_metrics": image_pos_metrics,
            "all_positions_metrics": all_pos_metrics,
            "natural_text_positions_metrics": nat_text_metrics,
            "natural_image_positions_metrics": nat_image_metrics,
            "natural_all_positions_metrics": nat_all_metrics,
            "router_agreement": router_agreement,
            "unforced_router_agreement_summary": unforced_router_agreement_summary,
        }
      elif not sample_passed and golden_data_point.get("ref_topk_experts") is not None:
        raw_diag_logits = deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx(
            model,
            config,
            mesh,
            golden_data_point["ref_topk_experts"],
            sample_name=sample_name,
            ids=ids,
            decoder_positions=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            images=images,
            image_masks=image_masks,
            seq_len=seq_len,
            raw_indexer_topk=golden_data_point.get("ref_indexer_topk"),
            raw_indexer_layer_ids=golden_data_point.get("ref_indexer_layer_ids"),
        )
        diag_full = _allgather_train_logits(raw_diag_logits, seq_len, config.max_target_length, config.vocab_size)
        diag_slice = diag_full[0, start_index:token_size, :min_vocab_size]
        diag_text_metrics = compute_scope_metrics(
            diag_slice[text_mask],
            golden_logits_slice[text_mask],
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            tie_flip_logits_2d=tie_flip_text,
        )
        diag_image_metrics = compute_scope_metrics(
            diag_slice[image_mask_pos],
            golden_logits_slice[image_mask_pos],
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            tie_flip_logits_2d=None,
        )
        diag_all_metrics = compute_scope_metrics(
            diag_slice[valid_mask],
            golden_logits_slice[valid_mask],
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            tie_flip_logits_2d=None,
        )
        frozen_topk_diagnostic = {
            "ran": True,
            "text_positions_metrics": diag_text_metrics,
            "image_positions_metrics": diag_image_metrics,
            "all_positions_metrics": diag_all_metrics,
        }
        diag_line = (
            f"DS41_MM_FROZEN_TOPK_DIAGNOSTIC: sample={sample_name} "
            f"unforced_text_max_kl={float(text_pos_metrics['max_kl']):.8g} -> "
            f"forced_text_max_kl={float(diag_text_metrics['max_kl']):.8g}, "
            f"unforced_text_mean_kl={float(text_pos_metrics['mean_kl']):.8g} -> "
            f"forced_text_mean_kl={float(diag_text_metrics['mean_kl']):.8g}"
        )
        max_logging.log(diag_line)
        if jax.process_index() == 0:
          print(diag_line, flush=True)
      else:
        frozen_topk_diagnostic = {
            "ran": False,
            "reason": "text_positions_passed" if sample_passed else "missing_ref_topk_experts",
        }

      scope_log_payload = {
          "sample": sample_name,
          "passed": sample_passed,
          "frozen_topk_gate_active": use_frozen_topk,
          "text_positions_metrics": text_pos_metrics,
          "image_positions_metrics": image_pos_metrics,
          "all_positions_metrics": all_pos_metrics,
          "natural_routing_metrics": natural_routing_metrics,
          "router_agreement_passed": router_agreement["passed"] if router_agreement is not None else None,
      }
      scope_log_line = f"DS41_MM_FORWARD_SCOPE_METRICS: {json.dumps(scope_log_payload)}"
      max_logging.log(scope_log_line)
      if jax.process_index() == 0:
        print(scope_log_line, flush=True)

    max_kl_limit = (
        float(gate_spec["max_kl"])
        if gate_spec is not None and "max_kl" in gate_spec
        else (float(test_args.max_kl_div) if test_args.max_kl_div is not None else None)
    )
    mean_kl_limit = float(gate_spec["mean_kl"]) if gate_spec is not None and "mean_kl" in gate_spec else None
    required_top10 = (
        int(gate_spec["last_token_top10_matches"])
        if gate_spec is not None and "last_token_top10_matches" in gate_spec
        else None
    )
    gated_metrics = text_pos_metrics if is_mm_sample else all_pos_metrics
    assert gated_metrics is not None
    gate_label = gate_id if gate_id else "legacy"
    if sample_passed:
      max_logging.log(
          f"[{gate_label}] PASS sample={sample_name} true_length={int(seq_len)} "
          f"max_kl={float(gated_metrics['max_kl']):.8g} (limit={max_kl_limit}) at index {max_kl_index}, "
          f"last_token_top10={int(gated_metrics['last_token_top10_matches'])}/{required_top10}"
      )
    else:
      fail_detail = "; ".join(failure_reasons)
      logging.error("[%s] FAIL sample=%s true_length=%d: %s", gate_label, sample_name, int(seq_len), fail_detail)
      max_logging.log(f"[{gate_label}] FAIL sample={sample_name} true_length={int(seq_len)}: {fail_detail}")

    rec = {
        "index": golden_data_index,
        "name": sample_name,
        "is_multimodal": is_mm_sample,
        "gate": gate_id,
        "gate_id": gate_id,
        "tier": gate_id,
        "status": "passed" if sample_passed else "failed",
        "true_length": int(seq_len),
        "max_kl": float(gated_metrics["max_kl"]),
        "max_kl_legacy_fp32": float(gated_metrics["max_kl_legacy_fp32"]),
        "max_kl_limit": max_kl_limit,
        "max_kl_index": max_kl_index,
        "mean_kl": float(gated_metrics["mean_kl"]),
        "mean_kl_legacy_fp32": float(gated_metrics["mean_kl_legacy_fp32"]),
        "mean_kl_limit": mean_kl_limit,
        "p95_kl": float(gated_metrics["p95_kl"]),
        "p95_kl_legacy_fp32": float(gated_metrics["p95_kl_legacy_fp32"]),
        "p99_kl": float(gated_metrics["p99_kl"]),
        "p99_kl_legacy_fp32": float(gated_metrics["p99_kl_legacy_fp32"]),
        "last_token_top10_matches": int(gated_metrics["last_token_top10_matches"]),
        "required_top10_matches": required_top10,
        "image_token_count": image_token_count,
        "text_token_count": text_token_count,
        "all_positions_metrics": all_pos_metrics,
        "all_positions_gates": gates_all,
        "text_positions_metrics": text_pos_metrics,
        "text_positions_gates": gates_text,
        "image_positions_metrics": image_pos_metrics,
        "frozen_topk_gate_active": use_frozen_topk,
        "natural_routing_metrics": natural_routing_metrics,
        "router_agreement": router_agreement,
        "unforced_router_agreement_summary": unforced_router_agreement_summary,
        "frozen_topk_diagnostic": frozen_topk_diagnostic,
        "failure_reasons": failure_reasons,
        "passed": sample_passed,
    }
    sample_reports.append(rec)
    if not sample_passed:
      failed_samples.append(rec)

  gated_sample_reports = [r for r in sample_reports if "passed" in r]
  deferred_sample_reports = [r for r in sample_reports if "passed" not in r]
  text_reports = [r for r in sample_reports if not r["is_multimodal"]]
  gated_text_reports = [r for r in text_reports if "passed" in r]
  deferred_text_reports = [r for r in text_reports if "passed" not in r]
  mm_reports = [r for r in sample_reports if r["is_multimodal"]]
  all_text_passed = all(r["passed"] for r in gated_text_reports)
  all_mm_passed = all(r["passed"] for r in mm_reports)
  all_samples_passed = bool(all_text_passed and all_mm_passed)

  long_text_reports = [r for r in text_reports if r["gate_id"] == "long_text_prompts"]
  if not enable_long_text_gate:
    long_text_gate_status = DEFERRED_LONG_TEXT_STATUS
    long_text_sample_3_status = DEFERRED_LONG_TEXT_STATUS
  else:
    long_text_gate_status = (
        ("passed" if all(r["passed"] for r in long_text_reports) else "failed") if long_text_reports else "not_applicable"
    )
    sample_3_matches = [r for r in long_text_reports if r["name"] == "sample_3"]
    long_text_sample_3_status = (
        ("passed" if sample_3_matches[0]["passed"] else "failed") if sample_3_matches else long_text_gate_status
    )

  text_samples_summary = [
      (
          {
              "index": r["index"],
              "name": r["name"],
              "gate": r["gate"],
              "gate_id": r["gate_id"],
              "tier": r["tier"],
              "true_length": r["true_length"],
              "status": r["status"],
          }
          if "passed" not in r
          else {
              "index": r["index"],
              "name": r["name"],
              "gate": r["gate"],
              "gate_id": r["gate_id"],
              "true_length": r["true_length"],
              "status": r["status"],
              "max_kl": r["max_kl"],
              "max_kl_legacy_fp32": r["max_kl_legacy_fp32"],
              "max_kl_limit": r["max_kl_limit"],
              "max_kl_index": r["max_kl_index"],
              "mean_kl": r["mean_kl"],
              "mean_kl_legacy_fp32": r["mean_kl_legacy_fp32"],
              "mean_kl_limit": r["mean_kl_limit"],
              "p95_kl": r["p95_kl"],
              "p95_kl_legacy_fp32": r["p95_kl_legacy_fp32"],
              "p99_kl": r["p99_kl"],
              "p99_kl_legacy_fp32": r["p99_kl_legacy_fp32"],
              "last_token_top10_matches": r["last_token_top10_matches"],
              "required_top10_matches": r["required_top10_matches"],
              "failure_reasons": r["failure_reasons"],
              "passed": r["passed"],
          }
      )
      for r in text_reports
  ]
  text_gate_summary = {
      "enable_long_text_gate": enable_long_text_gate,
      "text_gates_passed": bool(all_text_passed),
      "long_text_gate_status": long_text_gate_status,
      "long_text_sample_3_status": long_text_sample_3_status,
      "text_samples": text_samples_summary,
  }
  text_gate_line = f"DS41_TEXT_GATE_SUMMARY: {json.dumps(text_gate_summary)}"
  max_logging.log(text_gate_line)
  if jax.process_index() == 0:
    print(text_gate_line, flush=True)

  summary = {
      "passed": all_samples_passed,
      "all_passed": all_samples_passed,
      "enable_long_text_gate": enable_long_text_gate,
      "long_text_gate_status": long_text_gate_status,
      "long_text_sample_3_status": long_text_sample_3_status,
      "text_gates_passed": bool(all_text_passed),
      "mm_gates_passed": bool(all_mm_passed),
      "audit_result": audit_result,
      "num_samples": len(sample_reports),
      "num_gated_samples": len(gated_sample_reports),
      "num_deferred_samples": len(deferred_sample_reports),
      "num_passed": sum(1 for r in gated_sample_reports if r["passed"]),
      "max_kl": max((float(r["max_kl"]) for r in gated_sample_reports), default=0.0),
      "max_kl_legacy_fp32": max((float(r["max_kl_legacy_fp32"]) for r in gated_sample_reports), default=0.0),
      "mean_kl": (float(np.mean([float(r["mean_kl"]) for r in gated_sample_reports])) if gated_sample_reports else 0.0),
      "mean_kl_legacy_fp32": (
          float(np.mean([float(r["mean_kl_legacy_fp32"]) for r in gated_sample_reports])) if gated_sample_reports else 0.0
      ),
      "p95_kl": max((float(r["p95_kl"]) for r in gated_sample_reports), default=0.0),
      "p95_kl_legacy_fp32": max((float(r["p95_kl_legacy_fp32"]) for r in gated_sample_reports), default=0.0),
      "p99_kl": max((float(r["p99_kl"]) for r in gated_sample_reports), default=0.0),
      "p99_kl_legacy_fp32": max((float(r["p99_kl_legacy_fp32"]) for r in gated_sample_reports), default=0.0),
      "max_kl_div": max((float(r["max_kl"]) for r in gated_sample_reports), default=0.0),
      "max_kl_div_legacy_fp32": max((float(r["max_kl_legacy_fp32"]) for r in gated_sample_reports), default=0.0),
      "mean_kl_div": (
          float(np.mean([float(r["mean_kl"]) for r in gated_sample_reports])) if gated_sample_reports else 0.0
      ),
      "mean_kl_div_legacy_fp32": (
          float(np.mean([float(r["mean_kl_legacy_fp32"]) for r in gated_sample_reports])) if gated_sample_reports else 0.0
      ),
      "text_only_summary": {
          "num_samples": len(text_reports),
          "num_gated_samples": len(gated_text_reports),
          "num_deferred_samples": len(deferred_text_reports),
          "num_passed": sum(1 for r in gated_text_reports if r["passed"]),
          "all_passed": bool(all_text_passed),
          "long_text_gate_status": long_text_gate_status,
          "long_text_sample_3_status": long_text_sample_3_status,
          "sample_names": [r["name"] for r in text_reports],
          "gated_sample_names": [r["name"] for r in gated_text_reports],
          "deferred_sample_names": [r["name"] for r in deferred_text_reports],
      },
      "multimodal_summary": {
          "num_samples": len(mm_reports),
          "num_passed": sum(1 for r in mm_reports if r["passed"]),
          "all_passed": bool(all_mm_passed),
          "total_image_tokens": sum(int(r["image_token_count"]) for r in mm_reports),
          "sample_names": [r["name"] for r in mm_reports],
      },
      "text_samples": text_samples_summary,
      "samples": sample_reports,
  }
  max_logging.log(f"DS41_FORWARD_CHECKER_SUMMARY: {json.dumps(summary)}")
  _write_summary_json_file(summary_json_path, summary)

  if jax.process_index() == 0 and test_args.output_logits_path and all_data_to_save:
    out_logits_dir = os.path.dirname(test_args.output_logits_path)
    if out_logits_dir:
      os.makedirs(out_logits_dir, exist_ok=True)
    with jsonlines.open(test_args.output_logits_path, "a") as f:
      f.write(all_data_to_save)
    max_logging.log(f"Saved logits to {test_args.output_logits_path}")
    if test_args.gcs_output_logits_path:
      bucket_name = test_args.gcs_output_logits_path.split("/")[2]
      destination_blob_name = "/".join(
          test_args.gcs_output_logits_path.split("/")[3:] + test_args.output_logits_path.split("/")[-1:]
      )
      fplc.upload_blob(bucket_name, test_args.output_logits_path, destination_blob_name)
      max_logging.log(f"Uploaded logits to {test_args.gcs_output_logits_path}")

  if not summary["passed"]:
    failure_lines = [
        f"{r['name']}[{r['gate_id'] if r['gate_id'] else 'legacy'}]: {'; '.join(r['failure_reasons'])}"
        for r in failed_samples
    ]
    err_msg = f"{len(failed_samples)}/{len(gated_sample_reports)} forward logit samples failed gates: " + " | ".join(
        failure_lines
    )
    logging.error("%s", err_msg)
    max_logging.log(f"ERROR: {err_msg}")
    raise AssertionError(err_msg)
  return summary


def main(config, test_args, *, preloaded_model: Any = None):  # pylint: disable=redefined-outer-name
  """Verifies DeepSeek-V4.1 forward logits against pre-loaded goldens, thresholds, and optional HF audit."""
  if not test_args.run_hf_model:
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    tokenizer = fplc.load_tokenizer(
        config,
        hf_model_path=getattr(test_args, "hf_model_path", ""),
        trust_remote_code=getattr(test_args, "trust_remote_code", True),
        run_hf_model=False,
    )
    model = fplc.init_maxtext_model(config, mesh, preloaded_model=preloaded_model)
    return _run_golden_comparison(config, test_args, model, mesh, tokenizer)

  if str(getattr(test_args, "thresholds_json", "")):
    raise ValueError("--thresholds_json is not supported with --run_hf_model; use pre-loaded golden logits.")
  return fplc.main(config, test_args, preloaded_model=preloaded_model)


def run_checker(config, test_args, *, preloaded_model: Any = None):  # pylint: disable=redefined-outer-name
  """CLI execution wrapper that translates deterministic gate and code errors to exit code 42."""
  try:
    main(config, test_args, preloaded_model=preloaded_model)
  except (DeterministicGateError, AssertionError) as e:
    err_str = str(e)
    if (
        isinstance(e, DeterministicGateError)
        or "forward logit samples failed gates" in err_str
        or "KL divergence values" in err_str
    ):
      max_logging.log(f"Deterministic gate failure: {e}; exiting with code 42")
      sys.exit(EXIT_DETERMINISTIC_GATE_FAIL)
    raise
  except DETERMINISTIC_CODE_ERRORS as exc:
    det_err_msg = f"[DETERMINISTIC_CODE_ERROR] Unhandled {type(exc).__name__}: {exc}"
    max_logging.log(det_err_msg)
    print(det_err_msg, file=sys.stderr)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(EXIT_DETERMINISTIC_GATE_FAIL)


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the CLI `ArgumentParser` for `tests.utils.deepseek_v41.logit_runner`."""
  arg_parser = fplc.build_arg_parser()
  arg_parser.add_argument("--mm_golden_logits_path", type=str, required=False, default="")
  arg_parser.add_argument("--thresholds_json", type=str, required=False, default="")
  arg_parser.add_argument("--enable_long_text_gate", type=str2bool, required=False, default=False)
  arg_parser.add_argument("--require_frozen_topk_gate", type=str2bool, required=False, default=False)
  arg_parser.add_argument(
      "--router_agreement_ulp_multiplier",
      type=float,
      required=False,
      default=deepseek_v41_frozen_topk.DEFAULT_ROUTER_AGREEMENT_ULP_MULTIPLIER,
  )
  arg_parser.add_argument("--max_images_per_sample", type=int, required=False, default=4)
  arg_parser.add_argument("--run_ds41_mm_audit", type=str2bool, required=False, default=False)
  arg_parser.add_argument("--hf_checkpoint_dir_for_audit", type=str, required=False, default="")
  arg_parser.add_argument("--export_orbax_dir", type=str, required=False, default="")
  arg_parser.add_argument("--summary_json", type=str, required=False, default="")
  return arg_parser


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  parser = build_arg_parser()
  test_args, remaining_args = parser.parse_known_args()
  model_args = [sys.argv[0]] + remaining_args

  cfg = pyconfig.initialize_pydantic(model_args)
  assert (
      test_args.atol is not None or test_args.max_kl_div is not None or bool(test_args.thresholds_json)
  ), "At least one of --atol, --max_kl_div, or --thresholds_json must be specified to define the test criteria."

  if cfg.use_multimodal:
    assert not test_args.run_hf_model, (
        "Multimodal does not support running hf model on-the-fly, please generate hf golden logits "
        "using tests/assets/logits_generation/generate_hf_golden_logits.py"
    )
  run_checker(cfg, test_args)
