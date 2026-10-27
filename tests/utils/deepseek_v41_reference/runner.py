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

"""Builds / loads the plain-PyTorch DeepSeek-V4.1 reference from a HuggingFace-layout checkpoint directory.

The checkpoint directory holds `config.json` (HF layout, model settings under `text_config`), `*.safetensors`
using the upstream `inference/` parameter names (`embed.weight`, `layers.{i}.attn.wq_a.weight`, ...) and the
tokenizer files. Upstream's own `inference/config.json` uses `ModelArgs` field names; `model_args_from_hf_config`
maps the HF layout onto those.
"""

import glob
import json
import os
from typing import Any

import torch

from . import model as ref_model

# HF `text_config` key -> reference `ModelArgs` field. Keys missing here are named identically on both sides.
_HF_TO_ARGS = {
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_nextn_predict_layers": "n_mtp_layers",
    "num_attention_heads": "n_heads",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "routed_scaling_factor": "route_scale",
    "qk_rope_head_dim": "rope_head_dim",
    "rms_norm_eps": "norm_eps",
    "sliding_window": "window_size",
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "candidate_source_layer_id": "candidate_source_layer",
    "engram_pad_token_id": "engram_pad_id",
    "dspark_num_experts_per_tok": "dspark_n_activated_experts",
}

# Prefixes / suffixes of checkpoint tensors the text-only reference does not hold.
_SKIPPED_PREFIXES = ("vision.", "aligner.", "image_", "mtp.")
_SKIPPED_SUFFIXES = (".scale", ".bias_vl")


def text_config(hf_config: dict) -> dict:
  return hf_config.get("text_config", hf_config)


def model_args_from_hf_config(hf_config: dict, **overrides) -> ref_model.ModelArgs:
  """Maps an HF-layout `config.json` dict onto the reference `ModelArgs` (text model only, no DSpark)."""
  cfg = text_config(hf_config)
  fields = set(ref_model.ModelArgs.__dataclass_fields__)
  kwargs: dict[str, Any] = {}
  for key, value in cfg.items():
    name = _HF_TO_ARGS.get(key, key)
    if name in fields:
      kwargs[name] = tuple(value) if isinstance(value, list) else value
  rope_scaling = cfg.get("rope_scaling") or {}
  if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn":
    kwargs["original_seq_len"] = rope_scaling["original_max_position_embeddings"]
    kwargs["rope_factor"] = rope_scaling["factor"]
    kwargs["beta_fast"] = rope_scaling.get("beta_fast", 32)
    kwargs["beta_slow"] = rope_scaling.get("beta_slow", 1)
  # Text only: no vision tower and no DSpark draft layers.
  kwargs["vision_n_layers"] = 0
  kwargs["dspark_block_size"] = 0
  kwargs["n_mtp_layers"] = 0
  kwargs["dspark_target_layer_ids"] = ()
  kwargs.update(overrides)
  return ref_model.ModelArgs(**kwargs)


def load_hf_config(ckpt_dir: str) -> dict:
  with open(os.path.join(ckpt_dir, "config.json"), encoding="utf-8") as f:
    return json.load(f)


def read_safetensors(ckpt_dir: str) -> dict[str, torch.Tensor]:
  from safetensors.torch import load_file  # pylint: disable=import-outside-toplevel

  state = {}
  for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
    state.update(load_file(path))
  return state


def is_reference_key(name: str) -> bool:
  return not name.startswith(_SKIPPED_PREFIXES) and not name.endswith(_SKIPPED_SUFFIXES)


def build_model(args: ref_model.ModelArgs, tokenizer=None) -> ref_model.Transformer:
  """Builds the reference with torch's default dtype matching `args.dtype` (RMSNorm etc. use the default)."""
  prev = torch.get_default_dtype()
  # With fp32_compute the (large) Linear / engram weights stay in args.dtype, while the small parameters and the
  # activation caches follow torch's default dtype.
  torch.set_default_dtype(torch.float32 if args.dtype == "fp32" or args.fp32_compute else torch.bfloat16)
  try:
    model = ref_model.Transformer(args, tokenizer)
  finally:
    torch.set_default_dtype(prev)
  return model.eval()


_FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def dequantize_into(out: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, rows_per_chunk: int = 1 << 20):
  """Writes a block-scaled FP8 (E4M3) or packed FP4 (E2M1, int8 storage) weight into `out`, chunked over rows.

  The block size follows from the shapes of `weight` and `scale` (one E8M0 scale per block): 32 x 32 for FP8
  linears, 1 x 32 for FP4 experts and the FP8 engram table rows.
  """
  rows = weight.shape[0]
  cols = out.shape[1]
  block_rows = -(-rows // scale.shape[0])
  block_cols = -(-cols // scale.shape[1])
  rows_per_chunk = max(block_rows, rows_per_chunk // block_rows * block_rows)
  for start in range(0, rows, rows_per_chunk):
    stop = min(rows, start + rows_per_chunk)
    w = weight[start:stop]
    if w.dtype == torch.int8:
      packed = w.view(torch.uint8).long()
      w = torch.stack([_FP4_E2M1_VALUES[packed & 0xF], _FP4_E2M1_VALUES[packed >> 4]], dim=-1).flatten(1)
    else:
      w = w.float()
    s = scale[start // block_rows : -(-stop // block_rows)].float()
    s = s.repeat_interleave(block_rows, 0)[: stop - start].repeat_interleave(block_cols, 1)[:, :cols]
    out[start:stop].copy_(w * s)


def load_checkpoint_into(model: torch.nn.Module, ckpt_dir: str):
  """Copies an (optionally block-quantized) checkpoint into `model`, one tensor at a time, dequantizing weights
  that come with a `.scale` sibling. Never holds more than one checkpoint tensor (plus a row chunk) extra."""
  from safetensors import safe_open  # pylint: disable=import-outside-toplevel

  params = dict(model.named_parameters())
  loaded = set()
  for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
    with safe_open(path, framework="pt", device="cpu") as f:
      keys = set(f.keys())
      for name in sorted(keys):
        if not is_reference_key(name):
          continue
        if name not in params:
          raise ValueError(f"unexpected checkpoint tensor {name}")
        scale_name = name[: -len("weight")] + "scale" if name.endswith(".weight") else None
        with torch.no_grad():
          if scale_name is not None and scale_name in keys:
            dequantize_into(params[name].data, f.get_tensor(name), f.get_tensor(scale_name))
          else:
            params[name].data.copy_(f.get_tensor(name))
        loaded.add(name)
  missing = sorted(set(params) - loaded)
  if missing:
    raise ValueError(f"reference checkpoint is missing {len(missing)} tensors, e.g. {missing[:20]}")


def load_reference_model(
    ckpt_dir: str, dtype: str = "fp32", max_seq_len: int = 2048, tokenizer=None, fp32_compute=False, **overrides
):
  """Loads the text-only reference from an HF-layout checkpoint directory (quantized or not).

  Args:
    ckpt_dir: local checkpoint directory (config.json, *.safetensors, tokenizer files).
    dtype: "fp32" or "bf16" storage dtype of the reference weights (also the compute dtype unless fp32_compute).
    max_seq_len: sizes the reference's rope tables and caches.
    tokenizer: optional HF tokenizer (needed by the engram hashing); loaded from `ckpt_dir` when omitted.
    fp32_compute: run the matmuls in float32 on top of `dtype` weights.
    **overrides: further `ModelArgs` overrides (e.g. index_topk).

  Returns:
    (model, args)
  """
  hf_config = load_hf_config(ckpt_dir)
  args = model_args_from_hf_config(
      hf_config, dtype=dtype, max_seq_len=max_seq_len, max_batch_size=1, fp32_compute=fp32_compute, **overrides
  )
  if tokenizer is None and args.engram_layer_ids:
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
  model = build_model(args, tokenizer)
  load_checkpoint_into(model, ckpt_dir)
  return model, args
