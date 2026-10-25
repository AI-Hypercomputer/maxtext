# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper to extract mini-slice safetensors keys from index weight map."""

from collections.abc import Mapping
import json
import os
import shutil
from maxtext.m3.models.glm5_3_flash import glm5_config
import safetensors.torch
import torch

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def _normalize_key(key: str) -> str:
  prefix = "model.language_model."
  if key.startswith(prefix):
    return key[len(prefix) :]
  if key == "lm_head.weight" or key.startswith("lm_head."):
    return key
  raise ValueError(f"Unknown HF key prefix: {key!r}")


def extract_mini_slice_keys(
    weight_map: Mapping[str, str],
    num_hidden_layers: int,
    *,
    include_mtp_layer: int | None,
) -> dict[str, str]:
  """Filters index weight_map for mini-slice layers, embed_tokens, norm, and lm_head.

  Note: When include_mtp_layer is provided, it must match
  glm5_config.MTP_LAYER_INDEX (45). MTP layer weights in HuggingFace
  checkpoints retain their absolute layer index `layers.45` and are
  not re-indexed.
  """
  if num_hidden_layers <= 0:
    raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}.")
  if include_mtp_layer is not None and include_mtp_layer != glm5_config.MTP_LAYER_INDEX:
    raise ValueError(f"include_mtp_layer must be {glm5_config.MTP_LAYER_INDEX}, got" f" {include_mtp_layer}.")
  target_layers = set(range(num_hidden_layers))
  if include_mtp_layer is not None:
    target_layers.add(include_mtp_layer)
  counts = {l: 0 for l in target_layers}
  selected: dict[str, str] = {}
  for k, shard in weight_map.items():
    if k.startswith("model.visual.") or k.startswith("visual."):
      continue
    ck = _normalize_key(k)
    if any(ck.startswith(p) for p in ("embed_tokens.", "norm.", "lm_head.")):
      selected[k] = shard
    elif ck.startswith("layers."):
      l = int(ck.split(".")[1])
      if l in counts:
        selected[k] = shard
        counts[l] += 1
    else:
      raise ValueError(f"Unrecognized HF key in weight_map: {k!r}")

  for req in ("embed_tokens.", "norm.", "lm_head."):
    if not any(_normalize_key(k).startswith(req) for k in selected):
      raise ValueError(f"No {req[:-1]} keys found in weight_map.")
  for l, c in counts.items():
    if c == 0:
      raise ValueError(f"Required layer {l} has zero matching keys in weight_map.")
  return selected


def write_mini_slice_checkpoint(
    src_hf_dir: str,
    dst_hf_dir: str,
    *,
    num_hidden_layers: int,
    include_mtp_layer: int | None,
) -> dict[str, str]:
  """Extracts mini-slice layers from multi-shard HF directory and writes mini-slice checkpoint."""
  index_path = os.path.join(src_hf_dir, "model.safetensors.index.json")
  if not os.path.exists(index_path):
    raise FileNotFoundError(f"Missing index file in {src_hf_dir}: {index_path}")
  with open(index_path, "r", encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]

  sel = extract_mini_slice_keys(weight_map, num_hidden_layers, include_mtp_layer=include_mtp_layer)
  os.makedirs(dst_hf_dir, exist_ok=True)
  tensors: dict[str, torch.Tensor] = {}
  for s in sorted(set(sel.values())):
    with safetensors.safe_open(os.path.join(src_hf_dir, s), framework="pt") as f:
      for k in f.keys():
        if k in sel:
          tensors[k] = f.get_tensor(k)

  if missing := set(sel) - set(tensors):
    raise ValueError(f"Selected keys missing from safetensors shards: {missing}")

  safetensors.torch.save_file(tensors, os.path.join(dst_hf_dir, "model.safetensors"))
  sliced_map = {k: "model.safetensors" for k in sorted(tensors)}
  meta = {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())}
  with open(
      os.path.join(dst_hf_dir, "model.safetensors.index.json"),
      "w",
      encoding="utf-8",
  ) as f:
    json.dump({"metadata": meta, "weight_map": sliced_map}, f, indent=2)

  for fname in TOKENIZER_FILES:
    src_file = os.path.join(src_hf_dir, fname)
    if os.path.exists(src_file):
      shutil.copy2(src_file, os.path.join(dst_hf_dir, fname))

  if os.path.exists(cfg_p := os.path.join(src_hf_dir, "config.json")):
    with open(cfg_p, "r", encoding="utf-8") as f:
      cfg_data = json.load(f)
    num_nextn = 1 if include_mtp_layer is not None else 0
    cfg_data["num_hidden_layers"] = num_hidden_layers
    cfg_data["num_nextn_predict_layers"] = num_nextn
    if "text_config" in cfg_data and isinstance(cfg_data["text_config"], dict):
      cfg_data["text_config"]["num_hidden_layers"] = num_hidden_layers
      cfg_data["text_config"]["num_nextn_predict_layers"] = num_nextn
      for attr in ("layer_types", "mlp_layer_types", "indexer_types"):
        if attr in cfg_data["text_config"] and isinstance(cfg_data["text_config"][attr], list):
          cfg_data["text_config"][attr] = cfg_data["text_config"][attr][:num_hidden_layers]
      for attr in ("kda_layers", "full_attn_layers"):
        if attr in cfg_data["text_config"] and isinstance(cfg_data["text_config"][attr], list):
          cfg_data["text_config"][attr] = [x for x in cfg_data["text_config"][attr] if x < num_hidden_layers]
      if "linear_attn_config" in cfg_data["text_config"] and isinstance(
          cfg_data["text_config"]["linear_attn_config"], dict
      ):
        lac = cfg_data["text_config"]["linear_attn_config"]
        if "kda_layers" in lac and isinstance(lac["kda_layers"], list):
          lac["kda_layers"] = [idx for idx in lac["kda_layers"] if idx < num_hidden_layers]
        if "full_attn_layers" in lac and isinstance(lac["full_attn_layers"], list):
          lac["full_attn_layers"] = [idx for idx in lac["full_attn_layers"] if idx < num_hidden_layers]
    for attr in ("layer_types", "mlp_layer_types", "indexer_types"):
      if attr in cfg_data and isinstance(cfg_data[attr], list):
        cfg_data[attr] = cfg_data[attr][:num_hidden_layers]
    for attr in ("kda_layers", "full_attn_layers"):
      if attr in cfg_data and isinstance(cfg_data[attr], list):
        cfg_data[attr] = [x for x in cfg_data[attr] if x < num_hidden_layers]
    if "linear_attn_config" in cfg_data and isinstance(cfg_data["linear_attn_config"], dict):
      lac = cfg_data["linear_attn_config"]
      if "kda_layers" in lac and isinstance(lac["kda_layers"], list):
        lac["kda_layers"] = [idx for idx in lac["kda_layers"] if idx < num_hidden_layers]
      if "full_attn_layers" in lac and isinstance(lac["full_attn_layers"], list):
        lac["full_attn_layers"] = [idx for idx in lac["full_attn_layers"] if idx < num_hidden_layers]
    with open(os.path.join(dst_hf_dir, "config.json"), "w", encoding="utf-8") as f:
      json.dump(cfg_data, f, indent=2)

  return sliced_map
