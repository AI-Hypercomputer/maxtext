# Copyright 2026 Google LLC
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

"""MaxText <-> Hugging Face parameter mappings, hooks, and shapes for m3 Gemma 3 text models.

Covers text-only `Gemma3ForCausalLM` checkpoints (`model.*`) and the text tower of multimodal
`Gemma3ForConditionalGeneration` checkpoints (`model.language_model.*`), for both the stacked-cycle
(`scan_layers=True`) and flat (`scan_layers=False`) m3 layouts. RMSNorm offsets and embedding tables
keep raw HF values, so loading and exporting are exact transposes/reshapes with no arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

import jax
import jax.numpy as jnp
from maxtext.m3.models.gemma3 import modeling_gemma3
import numpy as np

_TEXT_LAYER_PARAMS = (
    ("pre_self_attention_norm-weight", "input_layernorm.weight"),
    ("post_self_attention_norm-weight", "post_attention_layernorm.weight"),
    ("self_attention-query_norm-weight", "self_attn.q_norm.weight"),
    ("self_attention-key_norm-weight", "self_attn.k_norm.weight"),
    ("pre_ffw_norm-weight", "pre_feedforward_layernorm.weight"),
    ("post_ffw_norm-weight", "post_feedforward_layernorm.weight"),
    ("self_attention-query-kernel", "self_attn.q_proj.weight"),
    ("self_attention-key-kernel", "self_attn.k_proj.weight"),
    ("self_attention-value-kernel", "self_attn.v_proj.weight"),
    ("self_attention-out-kernel", "self_attn.o_proj.weight"),
    ("mlp-wi_0-kernel", "mlp.gate_proj.weight"),
    ("mlp-wi_1-kernel", "mlp.up_proj.weight"),
    ("mlp-wo-kernel", "mlp.down_proj.weight"),
)
_ATTN_KERNELS = ("self_attention-query-kernel", "self_attention-key-kernel", "self_attention-value-kernel")
_MLP_KERNELS = ("mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel")


def _hf_config_dict(config: Any) -> dict[str, Any]:
  """Returns the HF config as a plain dict (`dict`, `transformers.PretrainedConfig`, or `None`)."""
  if config is None:
    return {}
  if isinstance(config, Mapping):
    return dict(config)
  try:
    return dict(config.to_dict())
  except AttributeError as exc:
    raise TypeError(f"Unsupported Gemma 3 HF config type {type(config).__name__}.") from exc


def _resolve_text_config(config: Any, maxtext_config: Any) -> tuple[modeling_gemma3.Gemma3Config, bool]:
  """Returns `(text config, has_multimodal_wrapper)` for the MaxText-side parameter tree.

  The MaxText config is authoritative for the model tree when given; otherwise the HF config is
  parsed. Raises instead of falling back to preset defaults when neither describes a model.
  """
  raw = _hf_config_dict(config)
  has_mm_wrapper = "text_config" in raw and isinstance(raw["text_config"], Mapping)
  if maxtext_config is not None:
    return modeling_gemma3.Gemma3Config.from_config(maxtext_config), has_mm_wrapper
  if not raw:
    raise ValueError("Gemma 3 mapping needs a MaxText config or a non-empty HF config.")
  return modeling_gemma3.Gemma3Config.from_hf_config_json(raw), has_mm_wrapper


def _layer_prefixes(cfg: modeling_gemma3.Gemma3Config, scan_layers: bool) -> list[tuple[str, int | list[int]]]:
  """Returns `(maxtext_prefix, hf_layer_index_or_indices)` for every decoder sublayer."""
  if not scan_layers:
    return [(f"params-decoder-layers_{i}-", i) for i in range(cfg.num_hidden_layers)]
  pattern = cfg.sliding_window_pattern
  num_rem = cfg.num_hidden_layers % pattern
  num_stacked = cfg.num_hidden_layers - num_rem
  prefixes: list[tuple[str, int | list[int]]] = []
  if num_stacked > 0:
    prefixes.extend((f"params-decoder-layers-layers_{b}-", list(range(b, num_stacked, pattern))) for b in range(pattern))
  prefixes.extend((f"params-decoder-layers_{num_stacked + r}-", num_stacked + r) for r in range(num_rem))
  return prefixes


# pylint: disable=invalid-name
def GEMMA3_M3_MAXTEXT_TO_HF_PARAM_MAPPING(
    config: Any,
    maxtext_config: Any = None,
    scan_layers: bool = False,
) -> dict[str, Any]:
  """Returns `{maxtext_key: hf_key or [hf_key per stacked cycle]}` for m3 Gemma 3 text models."""
  cfg, has_mm_wrapper = _resolve_text_config(config, maxtext_config)
  hf_root = "model.language_model" if has_mm_wrapper else "model"
  mapping: dict[str, Any] = {
      "params-token_embedder-weight": f"{hf_root}.embed_tokens.weight",
      "params-decoder-decoder_norm-weight": f"{hf_root}.norm.weight",
  }
  if not cfg.tie_word_embeddings:
    mapping["params-decoder-logits_dense-kernel"] = "lm_head.weight"
  for prefix, hf_idx in _layer_prefixes(cfg, scan_layers):
    for mx_name, hf_name in _TEXT_LAYER_PARAMS:
      if isinstance(hf_idx, list):
        mapping[prefix + mx_name] = [f"{hf_root}.layers.{i}.{hf_name}" for i in hf_idx]
      else:
        mapping[prefix + mx_name] = f"{hf_root}.layers.{hf_idx}.{hf_name}"
  return mapping


def GEMMA3_M3_MAXTEXT_TO_HF_PARAM_HOOK_FN(
    config: Any,
    maxtext_config: Any = None,
    scan_layers: bool = False,
    saving_to_hf: bool = False,
) -> dict[str, Any]:
  """Returns per-parameter conversion hooks (transposes, reshapes, vocab pad/slice).

  Norm weights need no hook: m3 `Gemma3RMSNorm` stores HF's zero-centered offset and m3
  `Gemma3TextScaledWordEmbedding` stores the unscaled table, unlike legacy MaxText Gemma 3.
  """
  cfg, has_mm_wrapper = _resolve_text_config(config, maxtext_config)
  if saving_to_hf and has_mm_wrapper:
    raise NotImplementedError(
        "Exporting an m3 Gemma 3 text model into a multimodal Gemma3ForConditionalGeneration checkpoint "
        "would drop the vision tower; export with a text-only Gemma3TextConfig instead."
    )

  def pad_or_slice_vocab(x: Any, target_shape: Any) -> Any:
    source_vocab = x.shape[0]
    target_vocab, target_hidden = target_shape
    if source_vocab > target_vocab:
      warnings.warn(f"Truncating Gemma 3 vocab from {source_vocab} to {target_vocab}.")
      return x[:target_vocab, :]
    if source_vocab < target_vocab:
      warnings.warn(f"Zero-padding Gemma 3 vocab from {source_vocab} to {target_vocab}.")
      pad_shape = (target_vocab - source_vocab, target_hidden)
      if isinstance(x, jax.Array):
        return jnp.concatenate([x, jnp.zeros(pad_shape, dtype=x.dtype)], axis=0)
      return np.concatenate([x, np.zeros(pad_shape, dtype=x.dtype)], axis=0)
    return x

  def lm_head(x: Any, target_shape: Any) -> Any:
    # HF `lm_head.weight` is `(vocab, hidden)`; MaxText `logits_dense.kernel` is `(hidden, vocab)`.
    if saving_to_hf:
      return pad_or_slice_vocab(x.T, target_shape)
    return pad_or_slice_vocab(x, (target_shape[1], target_shape[0])).T

  def attn_in_kernel(x: Any, target_shape: Any) -> Any:
    # HF `(heads * head_dim, hidden)` <-> MaxText `(hidden, heads, head_dim)`.
    if saving_to_hf:
      return x.reshape(x.shape[0], -1).T.reshape(target_shape)
    return x.T.reshape(target_shape)

  def attn_out_kernel(x: Any, target_shape: Any) -> Any:
    # HF `(hidden, heads * head_dim)` <-> MaxText `(heads, head_dim, hidden)`.
    if saving_to_hf:
      return x.reshape(-1, x.shape[-1]).T.reshape(target_shape)
    return x.T.reshape(target_shape)

  def transpose_2d(x: Any, target_shape: Any) -> Any:
    return x.T.reshape(target_shape)

  hooks: dict[str, Any] = {"params-token_embedder-weight": pad_or_slice_vocab}
  if not cfg.tie_word_embeddings:
    hooks["params-decoder-logits_dense-kernel"] = lm_head
  for prefix, _ in _layer_prefixes(cfg, scan_layers):
    for name in _ATTN_KERNELS:
      hooks[prefix + name] = attn_in_kernel
    hooks[prefix + "self_attention-out-kernel"] = attn_out_kernel
    for name in _MLP_KERNELS:
      hooks[prefix + name] = transpose_2d
  return hooks


def _sanitize_overridden_layer_types(config: Any) -> Any:
  """Drops stale `layer_types` when `num_hidden_layers` was overridden on an HF Gemma 3 config."""
  raw = _hf_config_dict(config)
  if not raw:
    return config
  if "text_config" in raw and isinstance(raw["text_config"], Mapping):
    text_raw = dict(raw["text_config"])
    if (
        "layer_types" in text_raw
        and isinstance(text_raw["layer_types"], list)
        and "num_hidden_layers" in text_raw
        and isinstance(text_raw["num_hidden_layers"], int)
        and len(text_raw["layer_types"]) != text_raw["num_hidden_layers"]
    ):
      text_raw.pop("layer_types")
      return {**raw, "text_config": text_raw}
    return raw
  if (
      "layer_types" in raw
      and isinstance(raw["layer_types"], list)
      and "num_hidden_layers" in raw
      and isinstance(raw["num_hidden_layers"], int)
      and len(raw["layer_types"]) != raw["num_hidden_layers"]
  ):
    sanitized = dict(raw)
    sanitized.pop("layer_types")
    return sanitized
  return raw


def GEMMA3_M3_HF_WEIGHTS_TO_SHAPE(config: Any) -> dict[str, list[int]]:
  """Returns `{hf_key: shape}` for the text weights of an HF Gemma 3 checkpoint."""
  cfg, has_mm_wrapper = _resolve_text_config(_sanitize_overridden_layer_types(config), None)
  hf_root = "model.language_model" if has_mm_wrapper else "model"
  h, inter, d = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
  q_dim, kv_dim = cfg.num_attention_heads * d, cfg.num_key_value_heads * d
  shapes: dict[str, list[int]] = {
      f"{hf_root}.embed_tokens.weight": [cfg.vocab_size, h],
      f"{hf_root}.norm.weight": [h],
  }
  if not cfg.tie_word_embeddings:
    shapes["lm_head.weight"] = [cfg.vocab_size, h]
  per_layer = {
      "input_layernorm.weight": [h],
      "post_attention_layernorm.weight": [h],
      "pre_feedforward_layernorm.weight": [h],
      "post_feedforward_layernorm.weight": [h],
      "self_attn.q_norm.weight": [d],
      "self_attn.k_norm.weight": [d],
      "self_attn.q_proj.weight": [q_dim, h],
      "self_attn.k_proj.weight": [kv_dim, h],
      "self_attn.v_proj.weight": [kv_dim, h],
      "self_attn.o_proj.weight": [h, q_dim],
      "mlp.gate_proj.weight": [inter, h],
      "mlp.up_proj.weight": [inter, h],
      "mlp.down_proj.weight": [h, inter],
  }
  for i in range(cfg.num_hidden_layers):
    for name, shape in per_layer.items():
      shapes[f"{hf_root}.layers.{i}.{name}"] = list(shape)
  return shapes


# pylint: enable=invalid-name
