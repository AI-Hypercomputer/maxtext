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

r"""Apply a `slice_model` trim plan directly to a Hugging Face checkpoint.

Reads `slice_report.json` or `model_overrides.json` (or `key=value` overrides),
patches `config.json`, and streams only the kept/sliced tensors from the source
`.safetensors` shards into `<output-dir>/` without loading full shards into RAM.

Example:

  python -m maxtext.checkpoint_conversion.slicing.slice_hf_checkpoint \
      --slice-report=/tmp/qwen35_small/slice_report.json \
      --hf-model-path=Qwen/Qwen3.5-35B-A3B \
      --output-dir=/tmp/qwen35_small_hf
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
from typing import Any, Mapping, Sequence

from absl import app
from huggingface_hub import HfFileSystem, hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file
import torch

from maxtext.utils import max_logging
from maxtext.utils.globals import HF_IDS

# Deliberately independent of `slice_model` / `slice_utils`: HF slicing needs only
# torch + safetensors, not the JAX / MaxText train stack.
GiB = 1024**3

_SUPPORTED_OVERRIDE_KEYS = frozenset(
    {
        "base_num_decoder_layers",
        "compress_ratios",
        "num_experts",
        "base_emb_dim",
        "base_mlp_dim",
        "base_moe_mlp_dim",
        "base_num_query_heads",
        "base_num_kv_heads",
    }
)

# Matches text decoder layer keys (`layers.0.*`, `model.layers.0.*`,
# `model.language_model.layers.0.*`, `thinker.model.layers.0.*`), while
# excluding vision/audio encoder layers (`vision_tower...layers.0.*`, `visual.blocks.0.*`).
_DECODER_LAYER_RE = re.compile(r"^(?:model\.)?(?:language_model\.|thinker\.model\.)?layers\.(\d+)\.(.+)$")
_EXPERT_INDEX_RE = re.compile(r"\.(?:mlp|ffn|block_sparse_moe)\.experts\.(\d+)\.(.+)$")
_ROUTER_RE = re.compile(
    r"\.(?:(?:mlp|ffn|block_sparse_moe)\.(?:gate|router)\.(?:weight|bias|e_score_correction_bias)"
    r"|router\.(?:proj\.weight|per_expert_scale))$"
)
_EXPERT_FUSED_RE = re.compile(r"\.experts\.(gate_up_proj|down_proj)(?:_bias|_scale|_blocks)?$")
_PER_EXPERT_PROJ_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.(weight|bias)$")
_SHARED_EXPERT_PROJ_RE = re.compile(r"\.shared_experts?\.(gate_proj|up_proj|down_proj)\.(weight|bias)$")
_DENSE_MLP_PROJ_RE = re.compile(r"\.(?:mlp|ffn)\.(gate_proj|up_proj|down_proj|wi_0|wi_1|wo)\.(weight|bias)$")
_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale")

_SAFETENSORS_DTYPE_MAP = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "BOOL": (torch.bool, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
}

_AUX_FILE_PATTERNS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "chat_template.json",
)


# ----------------------------------------------------------------------------- types + config spec


@dataclasses.dataclass(frozen=True)
class HFSliceSpec:
  """Original vs. sliced dimensions of a Hugging Face text decoder."""

  orig_layers: int
  new_layers: int
  orig_experts: int
  new_experts: int
  orig_hidden: int
  new_hidden: int
  orig_intermediate: int | None
  new_intermediate: int | None
  orig_moe_intermediate: int | None
  new_moe_intermediate: int | None
  orig_q_heads: int
  new_q_heads: int
  orig_kv_heads: int
  new_kv_heads: int
  head_dim: int
  compress_ratios: tuple[int, ...] | None = None
  keep_mtp: bool = False

  @property
  def orig_q_dim(self) -> int:
    return self.orig_q_heads * self.head_dim

  @property
  def new_q_dim(self) -> int:
    return self.new_q_heads * self.head_dim

  @property
  def orig_kv_dim(self) -> int:
    return self.orig_kv_heads * self.head_dim

  @property
  def new_kv_dim(self) -> int:
    return self.new_kv_heads * self.head_dim

  @property
  def depth_sliced(self) -> bool:
    return self.new_layers != self.orig_layers

  @property
  def expert_sliced(self) -> bool:
    return self.orig_experts > 0 and self.new_experts != self.orig_experts

  @property
  def width_sliced(self) -> bool:
    return (
        self.new_hidden != self.orig_hidden
        or self.new_intermediate != self.orig_intermediate
        or self.new_moe_intermediate != self.orig_moe_intermediate
        or self.new_q_heads != self.orig_q_heads
        or self.new_kv_heads != self.orig_kv_heads
    )


def _get_text_config(hf_config: dict[str, Any]) -> dict[str, Any]:
  """Return the nested `text_config` dictionary if present, else `hf_config` itself."""
  if isinstance(hf_config.get("thinker_config"), dict) and isinstance(
      hf_config["thinker_config"].get("text_config"), dict
  ):
    return hf_config["thinker_config"]["text_config"]
  if isinstance(hf_config.get("text_config"), dict):
    return hf_config["text_config"]
  return hf_config


def _is_gemma2(hf_config: Mapping[str, Any], model_name: str | None) -> bool:
  if model_name and model_name.lower().startswith("gemma2"):
    return True
  tcfg = _get_text_config(dict(hf_config))
  return str(tcfg.get("model_type", hf_config.get("model_type", ""))).lower() == "gemma2"


def build_hf_slice_spec(
    hf_config: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    model_name: str | None = None,
    keep_mtp: bool = False,
) -> HFSliceSpec:
  """Translate MaxText `overrides` + HF `config.json` into an `HFSliceSpec`."""
  unknown = sorted(set(overrides) - _SUPPORTED_OVERRIDE_KEYS)
  if unknown:
    raise ValueError(
        f"Unsupported override keys for HF slicing: {unknown}; supported: {sorted(_SUPPORTED_OVERRIDE_KEYS)}."
    )

  tcfg = _get_text_config(dict(hf_config))
  orig_layers = int(tcfg["num_hidden_layers"])
  orig_experts = int(tcfg.get("num_experts") or tcfg.get("num_local_experts") or tcfg.get("n_routed_experts") or 0)
  orig_hidden = int(tcfg["hidden_size"])
  orig_intermediate = int(tcfg["intermediate_size"]) if tcfg.get("intermediate_size") is not None else None
  orig_moe_intermediate = int(tcfg["moe_intermediate_size"]) if tcfg.get("moe_intermediate_size") is not None else None
  orig_q_heads = int(tcfg["num_attention_heads"])
  orig_kv_heads = int(tcfg.get("num_key_value_heads", orig_q_heads))
  head_dim = int(tcfg.get("head_dim") or (orig_hidden // orig_q_heads))

  layer_factor = 2 if _is_gemma2(hf_config, model_name) else 1
  new_layers = (
      int(overrides["base_num_decoder_layers"]) * layer_factor if "base_num_decoder_layers" in overrides else orig_layers
  )
  new_experts = int(overrides["num_experts"]) if "num_experts" in overrides else orig_experts
  new_hidden = int(overrides["base_emb_dim"]) if "base_emb_dim" in overrides else orig_hidden
  new_q_heads = int(overrides["base_num_query_heads"]) if "base_num_query_heads" in overrides else orig_q_heads
  new_kv_heads = int(overrides["base_num_kv_heads"]) if "base_num_kv_heads" in overrides else orig_kv_heads

  new_intermediate = orig_intermediate
  new_moe_intermediate = orig_moe_intermediate
  if "base_moe_mlp_dim" in overrides:
    moe_dim = int(overrides["base_moe_mlp_dim"])
    if orig_moe_intermediate is not None:
      new_moe_intermediate = moe_dim
    elif orig_experts > 1 and orig_intermediate is not None:
      new_intermediate = moe_dim
  if "base_mlp_dim" in overrides:
    mlp_dim = int(overrides["base_mlp_dim"])
    # For MoE models where HF stores expert dim in `moe_intermediate_size` and
    # MaxText sets `base_mlp_dim == base_moe_mlp_dim`, only overwrite `intermediate_size`
    # when dense layers exist or `moe_intermediate_size` is absent.
    if orig_moe_intermediate is None or "base_moe_mlp_dim" not in overrides or orig_intermediate == orig_moe_intermediate:
      if orig_intermediate is not None:
        new_intermediate = mlp_dim
    elif orig_intermediate is not None and new_moe_intermediate is not None and orig_moe_intermediate > 0:
      # Scale dense `intermediate_size` by the same width ratio when distinct.
      new_intermediate = int(round(orig_intermediate * (new_moe_intermediate / orig_moe_intermediate)))

  if not 1 <= new_layers <= orig_layers:
    raise ValueError(f"Invalid target HF num_hidden_layers={new_layers} for original {orig_layers}.")
  if orig_experts > 0 and not 1 <= new_experts <= orig_experts:
    raise ValueError(f"Invalid target num_experts={new_experts} for original {orig_experts}.")
  if not 1 <= new_hidden <= orig_hidden:
    raise ValueError(f"Invalid target hidden_size={new_hidden} for original {orig_hidden}.")
  if not 1 <= new_q_heads <= orig_q_heads or not 1 <= new_kv_heads <= orig_kv_heads:
    raise ValueError(
        f"Invalid target heads (q={new_q_heads}, kv={new_kv_heads}) for original (q={orig_q_heads}, kv={orig_kv_heads})."
    )

  compress_ratios = None
  if "compress_ratios" in overrides and overrides["compress_ratios"] is not None:
    compress_ratios = tuple(int(x) for x in overrides["compress_ratios"])

  return HFSliceSpec(
      orig_layers=orig_layers,
      new_layers=new_layers,
      orig_experts=orig_experts,
      new_experts=new_experts,
      orig_hidden=orig_hidden,
      new_hidden=new_hidden,
      orig_intermediate=orig_intermediate,
      new_intermediate=new_intermediate,
      orig_moe_intermediate=orig_moe_intermediate,
      new_moe_intermediate=new_moe_intermediate,
      orig_q_heads=orig_q_heads,
      new_q_heads=new_q_heads,
      orig_kv_heads=orig_kv_heads,
      new_kv_heads=new_kv_heads,
      head_dim=head_dim,
      compress_ratios=compress_ratios,
      keep_mtp=keep_mtp,
  )


def slice_hf_config(hf_config: Mapping[str, Any], spec: HFSliceSpec) -> dict[str, Any]:
  """Return a copy of `hf_config` patched to match `spec`."""
  out = copy.deepcopy(dict(hf_config))
  targets = [_get_text_config(out)]
  if targets[0] is not out:
    targets.append(out)

  for cfg in targets:
    if "num_hidden_layers" in cfg:
      cfg["num_hidden_layers"] = spec.new_layers
    if isinstance(cfg.get("layer_types"), list):
      cfg["layer_types"] = cfg["layer_types"][: spec.new_layers]
    if spec.compress_ratios is not None and "compress_ratios" in cfg:
      cfg["compress_ratios"] = list(spec.compress_ratios)
    elif isinstance(cfg.get("compress_ratios"), list):
      cfg["compress_ratios"] = cfg["compress_ratios"][: spec.new_layers]
    if isinstance(cfg.get("mlp_only_layers"), list):
      cfg["mlp_only_layers"] = [int(i) for i in cfg["mlp_only_layers"] if int(i) < spec.new_layers]
    if isinstance(cfg.get("per_layer_config"), list):
      cfg["per_layer_config"] = cfg["per_layer_config"][: spec.new_layers]

    if not spec.keep_mtp:
      if "mtp_num_hidden_layers" in cfg:
        cfg["mtp_num_hidden_layers"] = 0
      if "num_nextn_predict_layers" in cfg:
        cfg["num_nextn_predict_layers"] = 0

    if spec.orig_experts > 0:
      for expert_key in ("num_experts", "num_local_experts", "n_routed_experts"):
        if expert_key in cfg and cfg[expert_key] is not None:
          cfg[expert_key] = spec.new_experts

    if "hidden_size" in cfg:
      cfg["hidden_size"] = spec.new_hidden
    if "num_attention_heads" in cfg:
      cfg["num_attention_heads"] = spec.new_q_heads
    if "num_key_value_heads" in cfg:
      cfg["num_key_value_heads"] = spec.new_kv_heads
    if spec.new_intermediate is not None and "intermediate_size" in cfg and cfg["intermediate_size"] is not None:
      cfg["intermediate_size"] = spec.new_intermediate
    if (
        spec.new_moe_intermediate is not None
        and "moe_intermediate_size" in cfg
        and cfg["moe_intermediate_size"] is not None
    ):
      cfg["moe_intermediate_size"] = spec.new_moe_intermediate

  if isinstance(out.get("vision_config"), dict) and "out_hidden_size" in out["vision_config"]:
    if out["vision_config"]["out_hidden_size"] == spec.orig_hidden:
      out["vision_config"]["out_hidden_size"] = spec.new_hidden
  return out


# ----------------------------------------------------------------------------- tensor slice planning


def should_keep_key(key: str, spec: HFSliceSpec) -> bool:
  """Whether `key` belongs to a kept layer and kept expert (pure string check)."""
  if not spec.keep_mtp and (key.startswith("mtp.") or ".mtp." in key):
    return False
  layer_match = _DECODER_LAYER_RE.match(key)
  if layer_match and int(layer_match.group(1)) >= spec.new_layers:
    return False
  expert_match = _EXPERT_INDEX_RE.search(key)
  if expert_match and spec.orig_experts > 0 and int(expert_match.group(1)) >= spec.new_experts:
    return False
  return True


def _strip_scale_suffix(key: str) -> tuple[str, bool]:
  for suffix in _SCALE_SUFFIXES:
    if key.endswith(suffix):
      return key[: -len(suffix)] + ".weight", True
  return key, False


def _is_multimodal_encoder_key(key: str) -> bool:
  return any(
      marker in key for marker in ("visual.", "vision_tower.", "audio_tower.", "multi_modal_projector.", "embed_vision.")
  )


def _axis_bounds(
    base_key: str,
    shape: Sequence[int],
    spec: HFSliceSpec,
) -> tuple[tuple[int, int] | None, ...]:
  """Return `(orig_dim, new_dim)` (or `None` if unsliced) for each axis of `base_key`."""
  ndim = len(shape)

  # 1. Vision / audio encoder tensors: only the final projector into decoder hidden_size changes with width.
  if _is_multimodal_encoder_key(base_key):
    if base_key.endswith(
        (
            "visual.merger.linear_fc2.weight",
            "visual.merger.linear_fc2.bias",
            "multi_modal_projector.linear_2.weight",
            "multi_modal_projector.linear_2.bias",
            "embed_vision.embedding_projection.weight",
        )
    ):
      return ((spec.orig_hidden, spec.new_hidden),) + (None,) * (ndim - 1)
    return (None,) * ndim

  # 2. Fused 3D expert tensors (`experts.gate_up_proj`, `experts.down_proj`, ...).
  fused_match = _EXPERT_FUSED_RE.search(base_key)
  if fused_match:
    proj_kind = fused_match.group(1)
    if ndim == 2:
      # Bias `[num_experts, dim]`
      moe_dim = spec.orig_moe_intermediate or spec.orig_intermediate
      new_moe_dim = spec.new_moe_intermediate or spec.new_intermediate
      if proj_kind == "gate_up_proj" and moe_dim is not None and new_moe_dim is not None:
        if moe_dim != new_moe_dim:
          raise ValueError(f"Width slicing of fused {base_key!r} is not supported.")
        return ((spec.orig_experts, spec.new_experts), None)
      return ((spec.orig_experts, spec.new_experts), (spec.orig_hidden, spec.new_hidden))
    if ndim == 3:
      moe_dim = spec.orig_moe_intermediate or spec.orig_intermediate
      new_moe_dim = spec.new_moe_intermediate or spec.new_intermediate
      if proj_kind == "gate_up_proj":
        if moe_dim != new_moe_dim:
          raise ValueError(f"Width slicing of fused 3D {base_key!r} is not supported.")
        if shape[2] == spec.orig_hidden:
          return ((spec.orig_experts, spec.new_experts), None, (spec.orig_hidden, spec.new_hidden))
        return ((spec.orig_experts, spec.new_experts), (spec.orig_hidden, spec.new_hidden), None)
      # `down_proj`: either `[E, H, M_moe]` (Qwen3.5/Gemma4) or `[E, M_moe, H]` (GPT-OSS).
      if moe_dim is not None and new_moe_dim is not None:
        if shape[1] == spec.orig_hidden and shape[2] == moe_dim:
          return (
              (spec.orig_experts, spec.new_experts),
              (spec.orig_hidden, spec.new_hidden),
              (moe_dim, new_moe_dim),
          )
        if shape[1] == moe_dim and shape[2] == spec.orig_hidden:
          return (
              (spec.orig_experts, spec.new_experts),
              (moe_dim, new_moe_dim),
              (spec.orig_hidden, spec.new_hidden),
          )
      return ((spec.orig_experts, spec.new_experts), None, None)

  # 3. Router / gate tensors (`mlp.gate.weight`, `mlp.gate.e_score_correction_bias`, `router.proj.weight`, ...).
  if "shared_expert" not in base_key and _ROUTER_RE.search(base_key):
    if ndim == 1:
      return ((spec.orig_experts, spec.new_experts),)
    if ndim == 2:
      return ((spec.orig_experts, spec.new_experts), (spec.orig_hidden, spec.new_hidden))

  # Fast path when width is unchanged: no other tensor changes shape.
  if not spec.width_sliced:
    return (None,) * ndim

  # 4. Token embeddings and LM head (`[vocab_size, hidden_size]`).
  if base_key.endswith(("embed_tokens.weight", "lm_head.weight")):
    return (None, (spec.orig_hidden, spec.new_hidden))

  # 5. Shared expert gate (`[1, hidden_size]`).
  if base_key.endswith("shared_expert_gate.weight"):
    return (None, (spec.orig_hidden, spec.new_hidden))

  # 6. Self-attention projections & QK norms.
  if ".self_attn." in base_key:
    if base_key.endswith("q_proj.weight"):
      if shape[0] == 2 * spec.orig_q_dim:
        return ((2 * spec.orig_q_dim, 2 * spec.new_q_dim), (spec.orig_hidden, spec.new_hidden))
      return ((spec.orig_q_dim, spec.new_q_dim), (spec.orig_hidden, spec.new_hidden))
    if base_key.endswith("q_proj.bias"):
      return ((spec.orig_q_dim, spec.new_q_dim),)
    if base_key.endswith(("k_proj.weight", "v_proj.weight")):
      return ((spec.orig_kv_dim, spec.new_kv_dim), (spec.orig_hidden, spec.new_hidden))
    if base_key.endswith(("k_proj.bias", "v_proj.bias")):
      return ((spec.orig_kv_dim, spec.new_kv_dim),)
    if base_key.endswith("o_proj.weight"):
      return ((spec.orig_hidden, spec.new_hidden), (spec.orig_q_dim, spec.new_q_dim))
    if base_key.endswith("o_proj.bias"):
      return ((spec.orig_hidden, spec.new_hidden),)
    if base_key.endswith(("q_norm.weight", "q_norm.bias")):
      if shape[0] == spec.head_dim:
        return (None,)
      return ((spec.orig_q_dim, spec.new_q_dim),)
    if base_key.endswith(("k_norm.weight", "k_norm.bias", "v_norm.weight", "v_norm.bias")):
      if shape[0] == spec.head_dim:
        return (None,)
      return ((spec.orig_kv_dim, spec.new_kv_dim),)

  # 7. Linear attention (GDN) projections.
  if ".linear_attn." in base_key:
    if base_key.endswith(("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")):
      return (None, (spec.orig_hidden, spec.new_hidden))
    if base_key.endswith("out_proj.weight"):
      return ((spec.orig_hidden, spec.new_hidden), None)
    if base_key.endswith(("conv1d.weight", "A_log", "dt_bias", "norm.weight")):
      return (None,) * ndim

  # 8. Per-expert MoE projections (`experts.{e}.{gate_proj,up_proj,down_proj,w1,w2,w3}`).
  expert_proj_match = _PER_EXPERT_PROJ_RE.search(base_key)
  if expert_proj_match:
    proj, kind = expert_proj_match.group(1), expert_proj_match.group(2)
    moe_orig = spec.orig_moe_intermediate if spec.orig_moe_intermediate is not None else spec.orig_intermediate
    moe_new = spec.new_moe_intermediate if spec.new_moe_intermediate is not None else spec.new_intermediate
    if moe_orig is None or moe_new is None:
      raise ValueError(f"Missing MoE intermediate size in spec for {base_key!r}.")
    if proj in ("gate_proj", "up_proj", "w1", "w3"):
      return ((moe_orig, moe_new),) if kind == "bias" else ((moe_orig, moe_new), (spec.orig_hidden, spec.new_hidden))
    return (
        ((spec.orig_hidden, spec.new_hidden),)
        if kind == "bias"
        else (
            (spec.orig_hidden, spec.new_hidden),
            (moe_orig, moe_new),
        )
    )

  # 9. Shared expert projections (`shared_expert(s).{gate_proj,up_proj,down_proj}`).
  shared_match = _SHARED_EXPERT_PROJ_RE.search(base_key)
  if shared_match:
    proj, kind = shared_match.group(1), shared_match.group(2)
    if proj in ("gate_proj", "up_proj"):
      return (None,) if kind == "bias" else (None, (spec.orig_hidden, spec.new_hidden))
    return ((spec.orig_hidden, spec.new_hidden),) if kind == "bias" else ((spec.orig_hidden, spec.new_hidden), None)

  # 10. Dense MLP projections (`mlp.{gate_proj,up_proj,down_proj,wi_0,wi_1,wo}`).
  dense_match = _DENSE_MLP_PROJ_RE.search(base_key)
  if dense_match:
    proj, kind = dense_match.group(1), dense_match.group(2)
    if spec.orig_intermediate is None or spec.new_intermediate is None:
      raise ValueError(f"Missing dense intermediate_size in spec for {base_key!r}.")
    if proj in ("gate_proj", "up_proj", "wi_0", "wi_1"):
      return (
          ((spec.orig_intermediate, spec.new_intermediate),)
          if kind == "bias"
          else (
              (spec.orig_intermediate, spec.new_intermediate),
              (spec.orig_hidden, spec.new_hidden),
          )
      )
    return (
        ((spec.orig_hidden, spec.new_hidden),)
        if kind == "bias"
        else (
            (spec.orig_hidden, spec.new_hidden),
            (spec.orig_intermediate, spec.new_intermediate),
        )
    )

  # 11. 1D Norms & scalars.
  if ndim == 1:
    if shape[0] == spec.orig_hidden and ("norm" in base_key or "ln" in base_key):
      return ((spec.orig_hidden, spec.new_hidden),)
    if shape[0] == 1:
      return (None,)

  raise ValueError(f"Unrecognized tensor key {base_key!r} (shape {tuple(shape)}) for width slicing.")


def plan_tensor_slice(
    key: str,
    shape: Sequence[int],
    spec: HFSliceSpec,
) -> tuple[slice, ...] | None:
  """Return per-axis slices for `key`, or `None` if `key` should be dropped."""
  if not should_keep_key(key, spec):
    return None

  base_key, is_scale = _strip_scale_suffix(key)
  bounds = _axis_bounds(base_key, shape, spec)
  if len(bounds) != len(shape):
    raise ValueError(f"Dimension mismatch for {key!r}: shape {tuple(shape)} vs bounds {bounds}.")

  slices: list[slice] = []
  for axis, (dim, bound) in enumerate(zip(shape, bounds)):
    if bound is None:
      slices.append(slice(0, dim))
      continue
    orig_dim, new_dim = bound
    if not is_scale:
      if dim != orig_dim:
        raise ValueError(
            f"Unexpected shape for {key!r} on axis {axis}: expected {orig_dim} from config, "
            f"got {dim} (full shape {tuple(shape)})."
        )
      slices.append(slice(0, new_dim))
    else:
      if dim == 1 or new_dim == orig_dim:
        slices.append(slice(0, dim))
      elif dim == orig_dim:
        slices.append(slice(0, new_dim))
      elif orig_dim % dim == 0:
        block_size = orig_dim // dim
        if new_dim % block_size != 0:
          raise ValueError(
              f"Scale tensor {key!r} axis {axis} block_size={block_size} does not divide target dim {new_dim}."
          )
        slices.append(slice(0, new_dim // block_size))
      else:
        raise ValueError(f"Scale tensor {key!r} axis {axis} dim {dim} does not divide weight orig_dim {orig_dim}.")
  return tuple(slices)


def is_full_slice(slices: Sequence[slice], shape: Sequence[int]) -> bool:
  return all(s.start in (0, None) and s.stop == d and s.step in (1, None) for s, d in zip(slices, shape))


# ----------------------------------------------------------------------------- safetensors streaming I/O


def _read_safetensors_header_from_stream(stream: Any) -> tuple[int, dict[str, Any]]:
  """Read the 8-byte length prefix and JSON header from a binary stream."""
  raw_len = stream.read(8)
  if len(raw_len) != 8:
    raise ValueError("Truncated safetensors file: could not read 8-byte header size.")
  header_len = struct.unpack("<Q", raw_len)[0]
  header = json.loads(stream.read(header_len))
  header.pop("__metadata__", None)
  return 8 + header_len, header


def _load_sliced_tensor_from_stream(
    stream: Any,
    data_base_offset: int,
    info: Mapping[str, Any],
    slices: Sequence[slice],
) -> torch.Tensor:
  """Read only the needed byte prefix of a tensor from `stream` and apply `slices`."""
  dtype_str = info["dtype"]
  if dtype_str not in _SAFETENSORS_DTYPE_MAP:
    raise ValueError(f"Unsupported safetensors dtype {dtype_str!r}.")
  torch_dtype, _ = _SAFETENSORS_DTYPE_MAP[dtype_str]
  shape = tuple(int(d) for d in info["shape"])
  begin, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
  total_bytes = end - begin

  if total_bytes == 0 or math.prod(shape) == 0:
    return torch.empty(shape, dtype=torch_dtype)[slices].contiguous()

  # Optimization: only read up to `slices[0].stop` along axis 0!
  read_shape = list(shape)
  read_bytes = total_bytes
  if shape and slices and slices[0].stop is not None and slices[0].stop < shape[0]:
    row_bytes = total_bytes // shape[0]
    read_shape[0] = int(slices[0].stop)
    read_bytes = read_shape[0] * row_bytes

  stream.seek(data_base_offset + begin)
  buf = bytearray(stream.read(read_bytes))
  if len(buf) != read_bytes:
    raise IOError(f"Short read from safetensors stream: expected {read_bytes} bytes, got {len(buf)}.")
  tensor = torch.frombuffer(buf, dtype=torch_dtype).reshape(read_shape)
  remaining_slices = (slice(0, read_shape[0]), *slices[1:]) if slices else ()
  if remaining_slices and not is_full_slice(remaining_slices, read_shape):
    tensor = tensor[remaining_slices].contiguous()
  return tensor


def _load_local_sliced_tensor(handle: Any, key: str, slices: Sequence[slice], shape: Sequence[int]) -> torch.Tensor:
  if is_full_slice(slices, shape):
    return handle.get_tensor(key)
  return handle.get_slice(key)[slices].contiguous()


class ShardStreamWriter:
  """Streams tensors into temporary shard files and renames them once total shard count is known."""

  def __init__(self, output_dir: Path, max_shard_bytes: int):
    self.output_dir = output_dir
    self.max_shard_bytes = max(1, int(max_shard_bytes))
    self._buffer: dict[str, torch.Tensor] = {}
    self._buffer_bytes = 0
    self._temp_shards: list[tuple[Path, list[str]]] = []
    self.total_bytes = 0
    self.total_tensors = 0

  def add(self, key: str, tensor: torch.Tensor) -> None:
    tensor_bytes = tensor.numel() * tensor.element_size()
    if self._buffer and self._buffer_bytes + tensor_bytes > self.max_shard_bytes:
      self._flush_buffer()
    self._buffer[key] = tensor
    self._buffer_bytes += tensor_bytes
    self.total_bytes += tensor_bytes
    self.total_tensors += 1

  def _flush_buffer(self) -> None:
    if not self._buffer:
      return
    idx = len(self._temp_shards) + 1
    tmp_path = self.output_dir / f".tmp-model-{idx:05d}.safetensors"
    save_safetensors_file(self._buffer, str(tmp_path), metadata={"format": "pt"})
    self._temp_shards.append((tmp_path, list(self._buffer.keys())))
    self._buffer.clear()
    self._buffer_bytes = 0

  def finalize(self) -> dict[str, str]:
    """Flush remaining tensors, rename shards to `model-XXXXX-of-YYYYY.safetensors`, and write index.json."""
    self._flush_buffer()
    num_shards = len(self._temp_shards)
    weight_map: dict[str, str] = {}
    for idx, (tmp_path, keys) in enumerate(self._temp_shards, start=1):
      shard_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
      final_path = self.output_dir / shard_name
      tmp_path.replace(final_path)
      for k in keys:
        weight_map[k] = shard_name

    index_payload = {
        "metadata": {"total_size": self.total_bytes},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (self.output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return weight_map


# ----------------------------------------------------------------------------- checkpoint slicing


def _should_copy_aux_file(filename: str) -> bool:
  name = os.path.basename(filename)
  if name in ("config.json", "model.safetensors", "model.safetensors.index.json"):
    return False
  if name.endswith((".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5", ".ot")):
    return False
  if name in _AUX_FILE_PATTERNS:
    return True
  return name.endswith((".py", ".jinja")) or name.startswith("tokenizer")


def slice_hf_checkpoint(
    hf_model_path: str,
    output_dir: str | os.PathLike[str],
    overrides: Mapping[str, Any],
    *,
    model_name: str | None = None,
    max_shard_bytes: int = 5 * GiB,
    keep_mtp: bool = False,
    token: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
  """Apply `overrides` to `hf_model_path` and write the sliced HF checkpoint to `output_dir`."""
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  is_local = os.path.isdir(hf_model_path)

  # 1. Load config.json and build slice spec.
  if is_local:
    config_path = os.path.join(hf_model_path, "config.json")
  else:
    config_path = hf_hub_download(repo_id=hf_model_path, filename="config.json", token=token, revision=revision)
  raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))

  spec = build_hf_slice_spec(raw_config, overrides, model_name=model_name, keep_mtp=keep_mtp)
  sliced_config = slice_hf_config(raw_config, spec)
  (out_path / "config.json").write_text(json.dumps(sliced_config, indent=2) + "\n", encoding="utf-8")

  # 2. Discover source safetensors shards and weight_map.
  if is_local:
    repo_files = sorted(os.listdir(hf_model_path))
  else:
    repo_files = sorted(list_repo_files(hf_model_path, token=token, revision=revision))

  if "model.safetensors.index.json" in repo_files:
    if is_local:
      index_path = os.path.join(hf_model_path, "model.safetensors.index.json")
    else:
      index_path = hf_hub_download(
          repo_id=hf_model_path, filename="model.safetensors.index.json", token=token, revision=revision
      )
    source_weight_map: dict[str, str] = json.loads(Path(index_path).read_text(encoding="utf-8"))["weight_map"]
  elif "model.safetensors" in repo_files:
    source_weight_map = {}
  else:
    raise FileNotFoundError(f"No model.safetensors.index.json or model.safetensors found in {hf_model_path!r}.")

  # 3. Group kept keys by shard so shards with 0 kept keys are skipped completely.
  all_source_shards: set[str] = set(source_weight_map.values()) if source_weight_map else {"model.safetensors"}
  shard_to_keys: dict[str, list[str]] = {}
  dropped_keys_count = 0

  if source_weight_map:
    for key, shard_name in source_weight_map.items():
      if should_keep_key(key, spec):
        shard_to_keys.setdefault(shard_name, []).append(key)
      else:
        dropped_keys_count += 1
  else:
    shard_to_keys["model.safetensors"] = []

  skipped_shards = sorted(all_source_shards - set(shard_to_keys))
  max_logging.log(
      f"HF slice plan: layers {spec.orig_layers}->{spec.new_layers}, "
      f"experts {spec.orig_experts}->{spec.new_experts}, "
      f"hidden {spec.orig_hidden}->{spec.new_hidden}; "
      f"reading {len(shard_to_keys)}/{len(all_source_shards)} source shards "
      f"({len(skipped_shards)} shards skipped, {dropped_keys_count} keys dropped by index)."
  )

  # 4. Stream kept/sliced tensors shard-by-shard.
  writer = ShardStreamWriter(out_path, max_shard_bytes=max_shard_bytes)
  sliced_tensors_count = 0
  fs = None if is_local else HfFileSystem(token=token)

  for shard_idx, shard_name in enumerate(sorted(shard_to_keys), start=1):
    requested_keys = shard_to_keys[shard_name]
    max_logging.log(f"[{shard_idx}/{len(shard_to_keys)}] Streaming {shard_name} ({len(requested_keys) or 'all'} keys)...")
    if is_local:
      local_shard = os.path.join(hf_model_path, shard_name)
      with safe_open(local_shard, framework="pt", device="cpu") as handle:
        keys_in_shard = requested_keys if requested_keys else list(handle.keys())
        for key in keys_in_shard:
          slice_obj = handle.get_slice(key)
          shape = tuple(int(d) for d in slice_obj.get_shape())
          slices = plan_tensor_slice(key, shape, spec)
          if slices is None:
            dropped_keys_count += 1
            continue
          if not is_full_slice(slices, shape):
            sliced_tensors_count += 1
          tensor = _load_local_sliced_tensor(handle, key, slices, shape)
          writer.add(key, tensor)
    else:
      assert fs is not None
      remote_path = f"{hf_model_path}/{shard_name}"
      with fs.open(remote_path, "rb", revision=revision) as stream:
        data_base, header = _read_safetensors_header_from_stream(stream)
        keys_in_shard = requested_keys if requested_keys else sorted(header.keys())
        # Read keys in ascending byte-offset order for sequential HTTP range reads.
        keys_in_shard = sorted(keys_in_shard, key=lambda k, hdr=header: hdr[k]["data_offsets"][0])
        for key in keys_in_shard:
          info = header[key]
          shape = tuple(int(d) for d in info["shape"])
          slices = plan_tensor_slice(key, shape, spec)
          if slices is None:
            dropped_keys_count += 1
            continue
          if not is_full_slice(slices, shape):
            sliced_tensors_count += 1
          tensor = _load_sliced_tensor_from_stream(stream, data_base, info, slices)
          writer.add(key, tensor)

  output_weight_map = writer.finalize()

  # 5. Copy / download auxiliary tokenizer & processor files.
  copied_aux_files: list[str] = []
  for fname in repo_files:
    if "/" in fname or not _should_copy_aux_file(fname):
      continue
    if is_local:
      src_file = os.path.join(hf_model_path, fname)
      if os.path.isfile(src_file):
        shutil.copy2(src_file, out_path / fname)
        copied_aux_files.append(fname)
    else:
      downloaded = hf_hub_download(repo_id=hf_model_path, filename=fname, token=token, revision=revision)
      shutil.copy2(downloaded, out_path / fname)
      copied_aux_files.append(fname)

  report = {
      "status": "completed",
      "source_hf_model_path": str(hf_model_path),
      "output_dir": str(out_path),
      "model_name": model_name,
      "overrides": dict(overrides),
      "spec": dataclasses.asdict(spec),
      "stats": {
          "source_shards_total": len(all_source_shards),
          "source_shards_read": len(shard_to_keys),
          "source_shards_skipped": len(skipped_shards),
          "output_shards": len(set(output_weight_map.values())),
          "kept_tensors": writer.total_tensors,
          "sliced_tensors": sliced_tensors_count,
          "dropped_tensors": dropped_keys_count,
          "output_total_bytes": writer.total_bytes,
          "output_total_gib": round(writer.total_bytes / GiB, 3),
      },
      "auxiliary_files": copied_aux_files,
  }
  (out_path / "hf_slice_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  max_logging.log(
      f"Sliced HF checkpoint written to {out_path} "
      f"({writer.total_tensors} tensors, {report['stats']['output_total_gib']} GiB across "
      f"{report['stats']['output_shards']} shard(s))."
  )
  return report


# ----------------------------------------------------------------------------- CLI


def load_overrides(
    slice_report_path: str | None,
    overrides_path: str | None,
    kv_args: Sequence[str],
) -> tuple[dict[str, Any], str | None]:
  """Load overrides (and optional `model_name`) from `--slice-report`, `--overrides`, and/or `k=v` CLI args."""
  overrides: dict[str, Any] = {}
  model_name: str | None = None

  if slice_report_path:
    report = json.loads(Path(slice_report_path).read_text(encoding="utf-8"))
    model_name = report.get("model_name")
    selected = report.get("selected")
    if not isinstance(selected, dict) or "overrides" not in selected:
      raise ValueError(f"Slice report {slice_report_path!r} does not contain a selected model with `overrides`.")
    overrides.update(selected["overrides"])

  if overrides_path:
    overrides.update(json.loads(Path(overrides_path).read_text(encoding="utf-8")))

  for item in kv_args:
    key, value = item.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key == "compress_ratios":
      overrides[key] = json.loads(value) if value.startswith("[") else [int(x) for x in value.split(",") if x.strip()]
    else:
      overrides[key] = int(value)

  return overrides, model_name


def parse_flags(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
  """Parse CLI flags and optional `key=value` override arguments."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--slice-report", "--slice_report", default=None, help="Path to slice_report.json.")
  parser.add_argument("--overrides", default=None, help="Path to model_overrides.json.")
  parser.add_argument("--hf-model-path", "--hf_model_path", default=None, help="Local HF checkpoint dir or HF repo ID.")
  parser.add_argument(
      "--model-name", "--model_name", default=None, help="MaxText model_name (used to resolve HF repo ID)."
  )
  parser.add_argument(
      "--output-dir", "--output_dir", required=True, help="Output directory for the sliced HF checkpoint."
  )
  parser.add_argument(
      "--max-shard-size-gib",
      "--max_shard_size_gib",
      type=float,
      default=5.0,
      help="Maximum output safetensors shard size in GiB (default: 5, like save_pretrained).",
  )
  parser.add_argument("--keep-mtp", "--keep_mtp", action="store_true", help="Keep Multi-Token Prediction weights.")
  parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face Hub access token.")
  parser.add_argument("--revision", default=None, help="Hugging Face Hub revision.")
  flags, rest = parser.parse_known_args(list(argv[1:]))

  bad = [a for a in rest if "=" not in a or a.startswith("-")]
  if bad:
    parser.error(f"Unrecognized arguments {bad}; inline overrides must be key=value.")
  if not flags.slice_report and not flags.overrides and not rest:
    parser.error("Provide at least one of --slice-report, --overrides, or inline key=value overrides.")
  return flags, rest


def main(argv: Sequence[str], flags: argparse.Namespace, kv_args: Sequence[str]) -> None:
  del argv  # All flags are parsed by argparse in `run()`.
  overrides, report_model_name = load_overrides(flags.slice_report, flags.overrides, kv_args)
  model_name = flags.model_name or report_model_name
  hf_model_path = flags.hf_model_path
  if not hf_model_path:
    if model_name and model_name in HF_IDS:
      hf_model_path = HF_IDS[model_name]
    else:
      raise ValueError("Specify --hf-model-path or a --model-name present in maxtext.utils.globals.HF_IDS.")

  slice_hf_checkpoint(
      hf_model_path=hf_model_path,
      output_dir=flags.output_dir,
      overrides=overrides,
      model_name=model_name,
      max_shard_bytes=int(flags.max_shard_size_gib * GiB),
      keep_mtp=flags.keep_mtp,
      token=flags.token,
      revision=flags.revision,
  )


def run() -> None:
  # Parse our flags first; absl only sees the program name (as in `slice_model`).
  flags, kv_args = parse_flags(sys.argv)
  app.run(functools.partial(main, flags=flags, kv_args=kv_args), argv=sys.argv[:1])


if __name__ == "__main__":
  run()
