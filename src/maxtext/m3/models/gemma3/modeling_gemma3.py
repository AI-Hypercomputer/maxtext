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

"""Gemma 3 text configuration, layers, scanned 5:1 decoder, and CausalLM for MaxText m3.

Implements the Gemma 3 1B / 4B / 12B / 27B text architecture in pure Flax NNX
matching Hugging Face `transformers.models.gemma3` numerical boundaries:
- Zero-centered `(1.0 + weight)` RMSNorm accumulated and scaled in `float32`.
- Scaled token embeddings (`* sqrt(hidden_size)` in activation dtype) storing
  raw unscaled weights for lossless 1:1 Hugging Face round-trip conversion.
- Host-`float64` precomputed dual RoPE (`10,000` unscaled on local sliding
  layers vs `1,000,000` with optional linear scaling on global layers).
- Multi-query (1B `num_key_value_heads=1`) and grouped-query (4B/12B/27B)
  self-attention with QK-norm before RoPE, `query_pre_attn_scalar**-0.5`
  scaling, exclusive sliding-window masking (`0 <= q - k < sliding_window`),
  packed `decoder_segment_ids` + padding isolation, and a `float32` softmax.
- GeGLU (`gelu_pytorch_tanh`) MLP and 4-RMSNorm pre/post residual decoder layer.
- Heterogeneous 6-layer cycle scanning (`5` local SWA + `1` global) via shared
  `maxtext.m3.infra.scan.scan_run` (`ScannedLayers`) plus trailing `layers_{idx}`
  layers (`num_hidden_layers % 6`), per-sublayer `jax.checkpoint(..., prevent_cse=True)`
  rematerialization, training FLOPs accounting, and `MaxTextTrainingEngine` hooks.
"""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
import dataclasses
import functools
import json
import os
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.m3.configs.config import BaseConfig
from maxtext.m3.core.sharding import (
    DefaultSharding,
    Sharding,
    TensorType,
    check_mesh,
    param_metadata,
)
from maxtext.m3.infra import scan as m3_scan
import numpy as np


_GEMMA3_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "gemma3-1b": {
        "vocab_size": 262_144,
        "hidden_size": 1152,
        "intermediate_size": 6912,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "query_pre_attn_scalar": 256.0,
        "sliding_window": 512,
        "rope_linear_scaling_factor": 1.0,
    },
    "gemma3-4b": {
        "vocab_size": 262_144,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_hidden_layers": 34,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "query_pre_attn_scalar": 256.0,
        "sliding_window": 1024,
        "rope_linear_scaling_factor": 8.0,
    },
    "gemma3-12b": {
        "vocab_size": 262_144,
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "query_pre_attn_scalar": 256.0,
        "sliding_window": 1024,
        "rope_linear_scaling_factor": 8.0,
    },
    "gemma3-27b": {
        "vocab_size": 262_144,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "num_hidden_layers": 62,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "query_pre_attn_scalar": 168.0,
        "sliding_window": 1024,
        "rope_linear_scaling_factor": 8.0,
    },
}


_GEMMA3_LOSS_BATCH_KEYS = (
    "inputs",
    "targets",
    "targets_segmentation",
    "inputs_segmentation",
    "inputs_position",
)

_GEMMA3_UNSUPPORTED_MODEL_MEDIA_KWARGS = (
    "encoder_images",
    "encoder_image_masks",
    "encoder_videos",
    "encoder_video_masks",
    "encoder_video_grid_thw",
)

_GEMMA3_UNSUPPORTED_ENGINE_MEDIA_KEYS = (
    "images",
    "image_masks",
    "videos",
    "video_masks",
    "video_grid_thw",
)


@dataclasses.dataclass(frozen=True)
class _FallbackWeightedMetric:
  """Fallback WeightedMetric container used when Tunix post-training is absent."""

  unreduced_sum: jax.Array
  denominator: jax.Array


def _resolve_weighted_metric_cls() -> Any:
  """Resolves canonical `WeightedMetric` class (falling back cleanly in pre-train installs)."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  try:
    from maxtext.training_engine import abstract_engine

    return abstract_engine.WeightedMetric
  except ImportError:
    return _FallbackWeightedMetric


# Defaults of HF `transformers` `Gemma3TextConfig.__init__` (v4.57 and v5). Published Gemma 3 `config.json`
# text configs omit keys equal to these, so a loader must fall back to these values, not a model preset.
_HF_GEMMA3_TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 262_208,
    "hidden_size": 2304,
    "intermediate_size": 9216,
    "num_hidden_layers": 26,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "rope_theta": 1_000_000.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "query_pre_attn_scalar": 256,
    "sliding_window": 4096,
    "final_logit_softcapping": None,
    "attn_logit_softcapping": None,
    "rope_local_base_freq": 10_000.0,
    "sliding_window_pattern": 6,
}


def resolve_jnp_dtype(dt: Any) -> jnp.dtype:
  """Resolves a string, NumPy dtype, or JAX dtype into a `jnp.dtype`."""
  return jnp.dtype(dt)


def resolve_precision(precision_name: Any) -> jax.lax.Precision:
  """Maps a config precision value or enum to `jax.lax.Precision`."""
  if isinstance(precision_name, jax.lax.Precision):
    return precision_name
  raw = precision_name.value if hasattr(precision_name, "value") else precision_name
  norm = str(raw).lower()
  match norm:
    case "highest" | "float32":
      return jax.lax.Precision.HIGHEST
    case "high":
      return jax.lax.Precision.HIGH
    case "default" | "bfloat16" | "":
      return jax.lax.Precision.DEFAULT
    case _:
      raise ValueError(f"Unsupported matmul_precision {precision_name!r}.")


def _normalize_remat_policy(raw_policy: Any) -> str:
  """Normalizes remat policy values into `'none'`, `'full'`, or `'minimal'`."""
  if raw_policy is None or raw_policy is False:
    return "none"
  if raw_policy is True:
    return "full"
  val = raw_policy.value if hasattr(raw_policy, "value") else raw_policy
  norm = str(val).lower()
  if norm in ("none", ""):
    return "none"
  if norm in ("full", "minimal"):
    return norm
  raise ValueError(f"Unsupported remat_policy {raw_policy!r}; expected 'none', 'full', or 'minimal'.")


def _hidden_activation_from_mlp_activations(mlp_activations: Any) -> str:
  """Maps MaxText `mlp_activations` to the Gemma 3 GeGLU activation; only `["gelu", "linear"]` is valid."""
  acts = [str(a) for a in mlp_activations]
  if acts != ["gelu", "linear"]:
    raise ValueError(f"Gemma 3 uses mlp_activations ['gelu', 'linear'] (GeGLU), got {acts}.")
  return "gelu_pytorch_tanh"


def _validate_rope_type(mapping: Mapping[str, Any]) -> str:
  """Validates optional `rope_type` / `type` entries in HF RoPE config dicts."""
  for key in ("rope_type", "type"):
    if key in mapping and mapping[key] is not None:
      rt = str(mapping[key]).lower()
      if rt not in ("linear", "default"):
        raise ValueError(f"Unsupported Gemma 3 {key}={mapping[key]!r}; expected 'linear' or 'default'.")
      return rt
  return "linear"


@dataclasses.dataclass(frozen=True)
class Gemma3Config:
  """Typed configuration for Gemma 3 text models (`gemma3-1b` defaults)."""

  vocab_size: int = 262_144
  hidden_size: int = 1152
  intermediate_size: int = 6912
  num_hidden_layers: int = 26
  num_attention_heads: int = 4
  num_key_value_heads: int = 1
  head_dim: int = 256
  query_pre_attn_scalar: float = 256.0
  sliding_window: int = 512
  sliding_window_pattern: int = 6
  rms_norm_eps: float = 1e-6
  rope_theta: float = 1_000_000.0
  rope_local_base_freq: float = 10_000.0
  rope_linear_scaling_factor: float = 1.0
  attn_logit_softcapping: float | None = None
  final_logit_softcapping: float | None = None
  hidden_activation: str = "gelu_pytorch_tanh"
  attention_bias: bool = False
  attention_dropout: float = 0.0
  initializer_range: float = 0.02
  tie_word_embeddings: bool = True
  dtype: str = "bfloat16"
  weight_dtype: str = "float32"
  scan_layers: bool = True
  remat_policy: str = "none"
  matmul_precision: str = "default"
  stack_cycles: bool = True
  param_scan_axis: int = 0

  def __post_init__(self) -> None:
    if self.vocab_size <= 0 or self.hidden_size <= 0 or self.intermediate_size <= 0:
      raise ValueError("vocab_size, hidden_size, and intermediate_size must be positive.")
    if self.num_hidden_layers <= 0:
      raise ValueError(f"num_hidden_layers ({self.num_hidden_layers}) must be positive.")
    if self.head_dim <= 0 or self.head_dim % 2 != 0:
      raise ValueError(f"head_dim ({self.head_dim}) must be positive and even for RoPE.")
    if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
      raise ValueError("num_attention_heads and num_key_value_heads must be positive.")
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
          f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
          f"num_key_value_heads ({self.num_key_value_heads})."
      )
    if self.query_pre_attn_scalar <= 0.0:
      raise ValueError(f"query_pre_attn_scalar ({self.query_pre_attn_scalar}) must be positive.")
    if self.sliding_window <= 0:
      raise ValueError(f"sliding_window ({self.sliding_window}) must be positive.")
    if self.sliding_window_pattern < 1:
      raise ValueError(f"sliding_window_pattern ({self.sliding_window_pattern}) must be >= 1.")
    if self.rope_theta <= 0.0 or self.rope_local_base_freq <= 0.0 or self.rope_linear_scaling_factor <= 0.0:
      raise ValueError("RoPE base frequencies and scaling factor must be positive.")
    if self.attn_logit_softcapping is not None:
      raise ValueError(
          "Gemma 3 does not support attn_logit_softcapping (HF Gemma3Attention.forward "
          "omits softcap and all Gemma 3 checkpoints set attn_logit_softcapping=None)."
      )
    if self.final_logit_softcapping is not None and self.final_logit_softcapping <= 0.0:
      raise ValueError("final_logit_softcapping must be positive when set.")
    if self.hidden_activation not in ("gelu_pytorch_tanh", "gelu"):
      raise ValueError(
          f"Unsupported hidden_activation {self.hidden_activation!r}; expected 'gelu_pytorch_tanh' or 'gelu'."
      )
    if self.attention_bias:
      raise ValueError("Gemma 3 text decoder does not use attention_bias=True.")
    if self.attention_dropout != 0.0:
      raise ValueError(f"attention_dropout ({self.attention_dropout}) is not implemented; Gemma 3 uses 0.0.")
    if self.initializer_range <= 0.0:
      raise ValueError(f"initializer_range ({self.initializer_range}) must be positive.")
    if self.remat_policy not in ("none", "full", "minimal"):
      raise ValueError(f"Unsupported remat_policy {self.remat_policy!r}; expected 'none', 'full', or 'minimal'.")
    resolve_jnp_dtype(self.dtype)
    resolve_jnp_dtype(self.weight_dtype)
    resolve_precision(self.matmul_precision)

  def is_sliding_layer(self, layer_idx: int) -> bool:
    """Returns True for local sliding-window layers, False on every `pattern`-th global layer."""
    return bool((layer_idx + 1) % self.sliding_window_pattern)

  @classmethod
  def preset(cls, model_name: str, **overrides: Any) -> Gemma3Config:
    """Returns a canonical Gemma 3 text config preset (`1b`, `4b`, `12b`, or `27b`)."""
    if model_name not in _GEMMA3_MODEL_PRESETS:
      raise ValueError(f"Unknown Gemma 3 preset {model_name!r}; expected one of {sorted(_GEMMA3_MODEL_PRESETS)}.")
    params = dict(_GEMMA3_MODEL_PRESETS[model_name])
    params.update(overrides)
    return cls(**params)

  @classmethod
  def from_hf_config_json(
      cls,
      path_or_dict: str | os.PathLike[str] | Mapping[str, Any],
      **overrides: Any,
  ) -> Gemma3Config:
    """Loads `Gemma3Config` from an HF `config.json` path, directory, or dictionary.

    Unwraps multimodal `{"text_config": {...}}` wrappers (ignoring `{"text_config": null}`), supports HF
    `transformers` v4 (`rope_scaling`) and v5 (`rope_parameters`) schemas, and fills omitted architecture keys
    with HF `Gemma3TextConfig` defaults. `layer_types` must match the 5:1 local/global pattern.
    """
    if isinstance(path_or_dict, Mapping):
      raw = dict(path_or_dict)
    else:
      cfg_path = os.fspath(path_or_dict)
      if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, "config.json")
      with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "text_config" in raw and isinstance(raw["text_config"], Mapping):
      text_raw = dict(raw["text_config"])
      if (
          "tie_word_embeddings" not in text_raw
          and "tie_word_embeddings" in raw
          and raw["tie_word_embeddings"] is not None
      ):
        text_raw["tie_word_embeddings"] = raw["tie_word_embeddings"]
      raw = text_raw

    if "use_bidirectional_attention" in raw and raw["use_bidirectional_attention"]:
      raise ValueError("Gemma 3 m3 implements causal attention only; use_bidirectional_attention=True is unsupported.")
    explicit = {k: v for k, v in raw.items() if v is not None or k.endswith("softcapping")}
    if "sliding_window_pattern" not in explicit and "_sliding_window_pattern" in explicit:
      explicit["sliding_window_pattern"] = explicit["_sliding_window_pattern"]
    raw = {**_HF_GEMMA3_TEXT_DEFAULTS, **explicit}

    kwargs: dict[str, Any] = {}
    direct_keys = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "query_pre_attn_scalar",
        "sliding_window",
        "sliding_window_pattern",
        "rms_norm_eps",
        "rope_theta",
        "rope_local_base_freq",
        "rope_linear_scaling_factor",
        "attn_logit_softcapping",
        "final_logit_softcapping",
        "hidden_activation",
        "attention_bias",
        "attention_dropout",
        "initializer_range",
        "tie_word_embeddings",
        "dtype",
        "weight_dtype",
        "scan_layers",
        "matmul_precision",
        "stack_cycles",
        "param_scan_axis",
    )
    for key in direct_keys:
      if key in raw and raw[key] is not None:
        kwargs[key] = raw[key]
    for nullable_key in ("attn_logit_softcapping", "final_logit_softcapping"):
      if nullable_key in raw and raw[nullable_key] is None:
        kwargs[nullable_key] = None

    if "rope_scaling" in raw and isinstance(raw["rope_scaling"], Mapping):
      rs = raw["rope_scaling"]
      rt = _validate_rope_type(rs)
      if rt == "default":
        kwargs["rope_linear_scaling_factor"] = 1.0
      elif "factor" in rs and rs["factor"] is not None:
        kwargs["rope_linear_scaling_factor"] = float(rs["factor"])

    if "rope_parameters" in raw and isinstance(raw["rope_parameters"], Mapping):
      rp = raw["rope_parameters"]
      if "full_attention" in rp and isinstance(rp["full_attention"], Mapping):
        full_rp = rp["full_attention"]
        full_rt = _validate_rope_type(full_rp)
        if "rope_theta" in full_rp and full_rp["rope_theta"] is not None:
          kwargs["rope_theta"] = float(full_rp["rope_theta"])
        if full_rt == "default":
          kwargs["rope_linear_scaling_factor"] = 1.0
        elif "factor" in full_rp and full_rp["factor"] is not None:
          kwargs["rope_linear_scaling_factor"] = float(full_rp["factor"])
      if "sliding_attention" in rp and isinstance(rp["sliding_attention"], Mapping):
        slide_rp = rp["sliding_attention"]
        for key in ("rope_type", "type"):
          if key in slide_rp and slide_rp[key] is not None and str(slide_rp[key]).lower() != "default":
            raise ValueError(f"Gemma 3 sliding_attention requires {key}='default', got {slide_rp[key]!r}.")
        if "factor" in slide_rp and slide_rp["factor"] is not None and float(slide_rp["factor"]) != 1.0:
          raise ValueError(
              f"Gemma 3 sliding_attention does not scale RoPE (factor must be 1.0, got {slide_rp['factor']!r})."
          )
        if "rope_theta" in slide_rp and slide_rp["rope_theta"] is not None:
          kwargs["rope_local_base_freq"] = float(slide_rp["rope_theta"])

    if "remat_policy" in raw:
      kwargs["remat_policy"] = _normalize_remat_policy(raw["remat_policy"])

    if "dtype" in kwargs:
      kwargs["dtype"] = resolve_jnp_dtype(kwargs["dtype"]).name
    if "weight_dtype" in kwargs:
      kwargs["weight_dtype"] = resolve_jnp_dtype(kwargs["weight_dtype"]).name

    kwargs.update(overrides)
    cfg = cls(**kwargs)
    if "layer_types" in raw and raw["layer_types"] is not None:
      expected = [
          "sliding_attention" if cfg.is_sliding_layer(i) else "full_attention" for i in range(cfg.num_hidden_layers)
      ]
      if list(raw["layer_types"]) != expected:
        raise ValueError(
            f"HF layer_types {list(raw['layer_types'])} do not match the Gemma 3 local/global pattern "
            f"(sliding_window_pattern={cfg.sliding_window_pattern}, num_hidden_layers={cfg.num_hidden_layers})."
        )
    return cfg

  @classmethod
  def from_config(cls, config: Any, **overrides: Any) -> Gemma3Config:
    """Builds `Gemma3Config` from `Gemma3Config`, dict, m3 `BaseConfig`, or `MaxTextConfig`."""
    if isinstance(config, cls):
      return dataclasses.replace(config, **overrides) if overrides else config
    if isinstance(config, Mapping):
      return cls.from_hf_config_json(config, **overrides)
    if isinstance(config, BaseConfig):
      raw = config.get_keys()
      for drop_key in ("dropout_rate", "dropout", "attention_dropout"):
        for src_map in (raw, overrides):
          if drop_key in src_map and src_map[drop_key] is not None and float(src_map[drop_key]) != 0.0:
            raise ValueError(f"{drop_key} ({src_map[drop_key]}) is not implemented for m3 Gemma 3; use 0.0.")

      mapped: dict[str, Any] = {}
      has_preset = False
      if "model_name" in raw and raw["model_name"] is not None and str(raw["model_name"]).strip():
        model_name = str(raw["model_name"])
        if model_name in _GEMMA3_MODEL_PRESETS:
          mapped.update(_GEMMA3_MODEL_PRESETS[model_name])
          has_preset = True
        elif "query_pre_attn_scalar" not in raw and "query_pre_attn_scalar" not in overrides:
          raise ValueError(
              f"Gemma 3 m3 needs a known model_name for query_pre_attn_scalar, got {model_name!r}; "
              f"expected one of {sorted(_GEMMA3_MODEL_PRESETS)}."
          )

      # Base keys precede effective/scaled keys (`dim`/`emb_dim`, `mlp_dim`, `num_*`),
      # which in turn precede explicit HF field names so overrides never invert precedence.
      key_pairs = (
          ("vocab_size", "vocab_size"),
          ("base_emb_dim", "hidden_size"),
          ("dim", "hidden_size"),
          ("emb_dim", "hidden_size"),
          ("hidden_size", "hidden_size"),
          ("base_mlp_dim", "intermediate_size"),
          ("mlp_dim", "intermediate_size"),
          ("intermediate_size", "intermediate_size"),
          ("base_num_decoder_layers", "num_hidden_layers"),
          ("num_layers", "num_hidden_layers"),
          ("num_decoder_layers", "num_hidden_layers"),
          ("num_hidden_layers", "num_hidden_layers"),
          ("base_num_query_heads", "num_attention_heads"),
          ("num_heads", "num_attention_heads"),
          ("num_query_heads", "num_attention_heads"),
          ("num_attention_heads", "num_attention_heads"),
          ("base_num_kv_heads", "num_key_value_heads"),
          ("num_kv_heads", "num_key_value_heads"),
          ("num_key_value_heads", "num_key_value_heads"),
          ("head_dim", "head_dim"),
          ("query_pre_attn_scalar", "query_pre_attn_scalar"),
          ("sliding_window_size", "sliding_window"),
          ("sliding_window", "sliding_window"),
          ("sliding_window_pattern", "sliding_window_pattern"),
          ("normalization_layer_epsilon", "rms_norm_eps"),
          ("rms_norm_eps", "rms_norm_eps"),
          ("rope_max_timescale", "rope_theta"),
          ("rope_theta", "rope_theta"),
          ("local_rope_max_timescale", "rope_local_base_freq"),
          ("rope_local_base_freq", "rope_local_base_freq"),
          ("rope_linear_scaling_factor", "rope_linear_scaling_factor"),
          ("logits_via_embedding", "tie_word_embeddings"),
          ("tie_word_embeddings", "tie_word_embeddings"),
          ("initializer_range", "initializer_range"),
          ("activations_dtype", "dtype"),
          ("dtype", "dtype"),
          ("weights_dtype", "weight_dtype"),
          ("weight_dtype", "weight_dtype"),
          ("scan_layers", "scan_layers"),
          ("stack_cycles", "stack_cycles"),
          ("param_scan_axis", "param_scan_axis"),
          ("remat_policy", "remat_policy"),
          ("precision", "matmul_precision"),
          ("matmul_precision", "matmul_precision"),
      )
      for src_key, dst_key in key_pairs:
        if src_key in raw and raw[src_key] is not None:
          mapped[dst_key] = raw[src_key]
      for src_cap, dst_cap in (
          ("attn_logits_soft_cap", "attn_logit_softcapping"),
          ("final_logits_soft_cap", "final_logit_softcapping"),
      ):
        if src_cap in raw and raw[src_cap] is not None:
          cap_val = float(raw[src_cap])
          if cap_val < 0.0:
            raise ValueError(f"{src_cap} ({cap_val}) must be non-negative.")
          mapped[dst_cap] = cap_val if cap_val > 0.0 else None
      if "mlp_activations" in raw and raw["mlp_activations"] is not None:
        mapped["hidden_activation"] = _hidden_activation_from_mlp_activations(raw["mlp_activations"])
      if "use_multimodal" in raw and bool(raw["use_multimodal"]):
        raise ValueError(
            "Gemma 3 m3 currently implements text-only Gemma3ForCausalLM; use_multimodal=True is unsupported."
        )
      if "stack_cycles" not in mapped and "scan_layers" in mapped:
        mapped["stack_cycles"] = bool(mapped["scan_layers"])
      mapped.update(overrides)
      if not has_preset:
        required_without_preset = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "query_pre_attn_scalar",
            "sliding_window",
            "rope_local_base_freq",
        )
        missing = [k for k in required_without_preset if k not in mapped]
        if missing:
          raise ValueError(
              f"BaseConfig without a known Gemma 3 model_name must explicitly specify {missing} "
              "(e.g. query_pre_attn_scalar, sliding_window_size, local_rope_max_timescale)."
          )
      if "remat_policy" in mapped:
        mapped["remat_policy"] = _normalize_remat_policy(mapped["remat_policy"])
      if "dtype" in mapped:
        mapped["dtype"] = resolve_jnp_dtype(mapped["dtype"]).name
      if "weight_dtype" in mapped:
        mapped["weight_dtype"] = resolve_jnp_dtype(mapped["weight_dtype"]).name
      if "matmul_precision" in mapped:
        mapped["matmul_precision"] = resolve_precision(mapped["matmul_precision"]).name.lower()
      return cls(**mapped)

    if not config.use_m3_model:
      raise ValueError("Gemma 3 m3 requires use_m3_model=True.")
    has_use_multimodal = (
        "use_multimodal" in config.get_keys() if "get_keys" in dir(config) else "use_multimodal" in dir(config)
    )
    if has_use_multimodal and bool(config.use_multimodal):
      raise ValueError(
          f"Multimodal is not yet supported for {config.model_name} with use_m3_model=True; "
          "Gemma 3 m3 currently implements text-only Gemma3ForCausalLM (set use_m3_model=False for multimodal)."
      )

    model_name = str(config.model_name)
    if model_name not in _GEMMA3_MODEL_PRESETS:
      raise ValueError(
          f"Gemma 3 m3 needs a known model_name for query_pre_attn_scalar, got {model_name!r}; "
          f"expected one of {sorted(_GEMMA3_MODEL_PRESETS)}."
      )
    q_scalar = float(_GEMMA3_MODEL_PRESETS[model_name]["query_pre_attn_scalar"])
    emb_dim = int(config.emb_dim)
    num_q_heads = int(config.num_query_heads)
    head_dim = int(config.head_dim)
    dropout_rate = float(config.dropout_rate)
    if dropout_rate != 0.0:
      raise ValueError(f"dropout_rate ({dropout_rate}) is not implemented for m3 Gemma 3; use 0.0.")
    hidden_activation = _hidden_activation_from_mlp_activations(config.mlp_activations)

    sw_size = int(config.sliding_window_size)
    if sw_size <= 0:
      raise ValueError(f"Gemma 3 requires positive sliding_window_size in config, got {sw_size}.")
    local_rope = float(config.local_rope_max_timescale)
    if local_rope <= 0.0:
      raise ValueError(f"Gemma 3 requires positive local_rope_max_timescale in config, got {local_rope}.")

    raw_attn_cap = config.attn_logits_soft_cap
    if raw_attn_cap is not None and float(raw_attn_cap) < 0.0:
      raise ValueError(f"attn_logits_soft_cap ({raw_attn_cap}) must be non-negative.")
    attn_cap = float(raw_attn_cap) if raw_attn_cap else None

    raw_final_cap = config.final_logits_soft_cap
    if raw_final_cap is not None and float(raw_final_cap) < 0.0:
      raise ValueError(f"final_logits_soft_cap ({raw_final_cap}) must be non-negative.")
    final_cap = float(raw_final_cap) if raw_final_cap else None
    scan_flag = bool(config.scan_layers)

    kwargs: dict[str, Any] = {
        "vocab_size": int(config.vocab_size),
        "hidden_size": emb_dim,
        "intermediate_size": int(config.mlp_dim),
        "num_hidden_layers": int(config.num_decoder_layers),
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": int(config.num_kv_heads),
        "head_dim": head_dim,
        "query_pre_attn_scalar": q_scalar,
        "sliding_window": sw_size,
        "rms_norm_eps": float(config.normalization_layer_epsilon),
        "rope_theta": float(config.rope_max_timescale),
        "rope_local_base_freq": local_rope,
        "rope_linear_scaling_factor": float(config.rope_linear_scaling_factor),
        "attn_logit_softcapping": attn_cap,
        "final_logit_softcapping": final_cap,
        "tie_word_embeddings": bool(config.logits_via_embedding),
        "hidden_activation": hidden_activation,
        "dtype": resolve_jnp_dtype(config.dtype).name,
        "weight_dtype": resolve_jnp_dtype(config.weight_dtype).name,
        "scan_layers": scan_flag,
        "remat_policy": _normalize_remat_policy(config.remat_policy),
        "matmul_precision": resolve_precision(config.matmul_precision).name.lower(),
        "stack_cycles": scan_flag,
        "param_scan_axis": int(config.param_scan_axis),
    }
    kwargs.update(overrides)
    return cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class Gemma3Sharding(DefaultSharding):
  """4-axis `('data', 'fsdp', 'tensor', 'expert')` logical sharding for Gemma 3.

  Handles both multi-query (`gemma3-1b`, `num_key_value_heads=1`) and grouped-query
  (`gemma3-4b/12b/27b`) variants by replicating `kv_heads` across `tensor` when
  `num_key_value_heads` is not divisible by `tensor` while keeping `key`/`value`
  input feature axes sharded across `('fsdp', 'expert')`.
  """

  replicate_kv_heads: bool = False
  vocab_tp: bool = True

  @classmethod
  def from_mesh(
      cls,
      mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None,
      config: Gemma3Config,
  ) -> Gemma3Sharding:
    """Constructs a validated `Gemma3Sharding` for `mesh` and `config`."""
    active = mesh
    if active is None:
      amb = jax.sharding.get_abstract_mesh()
      if not amb.empty:
        active = amb
    if active is None or active.empty:
      return cls(
          replicate_kv_heads=config.num_key_value_heads == 1,
          vocab_tp=True,
      )

    check_mesh(active)
    if any(t == jax.sharding.AxisType.Explicit for t in active.axis_types) and (
        isinstance(active, jax.sharding.Mesh)
        or any(t == jax.sharding.AxisType.Explicit for t in jax.sharding.get_abstract_mesh().axis_types)
    ):
      raise NotImplementedError("Gemma 3 m3 supports AxisType.Auto meshes only; ShardMode.EXPLICIT is not implemented.")
    fsdp_size = int(active.shape["fsdp"])
    tp_size = int(active.shape["tensor"])
    ep_size = int(active.shape["expert"])

    if tp_size > 1:
      if config.num_attention_heads % tp_size != 0:
        raise ValueError(f"num_attention_heads ({config.num_attention_heads}) must be divisible by tensor={tp_size}.")
      if config.intermediate_size % tp_size != 0:
        raise ValueError(f"intermediate_size ({config.intermediate_size}) must be divisible by tensor={tp_size}.")
    if config.hidden_size % (fsdp_size * ep_size) != 0:
      raise ValueError(f"hidden_size ({config.hidden_size}) must be divisible by fsdp*expert={fsdp_size * ep_size}.")

    rep_kv = (config.num_key_value_heads == 1) or (tp_size > 1 and config.num_key_value_heads % tp_size != 0)
    vocab_tp = tp_size <= 1 or (config.vocab_size % tp_size == 0)
    return cls(replicate_kv_heads=rep_kv, vocab_tp=vocab_tp)

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> Any:
    if axis == "kv_heads" and self.replicate_kv_heads:
      return None
    if axis == "vocab" and not self.vocab_tp:
      return None
    return super().map_axis(axis, tensor_name, tensor_type)


def _resolve_optional_sharding(
    sharding: Sharding | None,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None,
    config: Gemma3Config,
) -> Sharding | None:
  """Resolves `sharding` from `mesh` or the ambient mesh, returning `None` when no mesh is active."""
  if sharding is not None:
    return sharding
  active = mesh if (mesh is not None and not mesh.empty) else jax.sharding.get_abstract_mesh()
  if active is None or active.empty:
    return None
  return Gemma3Sharding.from_mesh(active, config)


def _param_meta(sharding: Sharding | None, name: str, axes: tuple[str, ...]) -> dict[str, Any]:
  """Returns `param_metadata` when `sharding` is active, else `{}` for mesh-free construction."""
  return param_metadata(sharding, name, axes) if sharding is not None else {}


def _norm_param_meta(sharding: Sharding | None, name: str) -> dict[str, Any]:
  """Returns replicated norm parameter metadata marked `preserve_unsharded` across scan stacking."""
  if sharding is None:
    return {}
  return {**param_metadata(sharding, name, ("norm",)), "preserve_unsharded": True}


def _constrain(sharding: Sharding | None, x: jax.Array, name: str, axes: tuple[str, ...]) -> jax.Array:
  """Applies `sharding.constrain` when a sharding strategy is attached, else returns `x` unchanged."""
  return sharding.constrain(x, name, axes) if sharding is not None else x


class Gemma3RMSNorm(nnx.Module):
  """Zero-centered `(1.0 + weight)` RMSNorm matching HF `Gemma3RMSNorm` in `float32`.

  The parameter is named `weight` and holds the raw HF offset. Legacy MaxText Gemma 3 checkpoints
  store `1 + weight` under `scale`; the different name makes restoring one fail instead of
  silently applying the offset twice.
  """

  def __init__(
      self,
      num_features: int,
      *,
      epsilon: float,
      param_dtype: Any,
      sharding: Sharding | None,
      name: str,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.weight = nnx.Param(
        jnp.zeros((num_features,), dtype=resolve_jnp_dtype(param_dtype)),
        **_norm_param_meta(sharding, name),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    input_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + self.epsilon)
    scaled = normed * (1.0 + self.weight.get_value().astype(jnp.float32))
    return scaled.astype(input_dtype)


class Gemma3TextScaledWordEmbedding(nnx.Module):
  """Token embedding scaled by `sqrt(features)` in activation dtype (`Gemma3TextScaledWordEmbedding`).

  The table is named `weight` and holds raw HF values. Legacy MaxText Gemma 3 checkpoints store
  pre-scaled tables under `embedding`; the different name makes restoring one fail.
  """

  def __init__(
      self,
      num_embeddings: int,
      features: int,
      *,
      dtype: Any,
      param_dtype: Any,
      initializer_range: float,
      sharding: Sharding | None,
      rngs: nnx.Rngs,
  ):
    self.num_embeddings = num_embeddings
    self.features = features
    self.dtype = resolve_jnp_dtype(dtype)
    self.param_dtype = resolve_jnp_dtype(param_dtype)
    self.sharding = sharding

    init_fn = nnx.initializers.normal(stddev=initializer_range)
    self.weight = nnx.Param(
        init_fn(rngs.params(), (num_embeddings, features), self.param_dtype),
        **_param_meta(self.sharding, "token_embedder", ("vocab", "embed")),
    )

  def __call__(self, inputs: jax.Array) -> jax.Array:
    gathered = jnp.take(self.weight.get_value(), inputs, axis=0).astype(self.dtype)
    embed_scale = jnp.asarray(self.features**0.5, dtype=self.dtype)
    scaled = gathered * embed_scale
    if scaled.ndim == 3:
      return _constrain(self.sharding, scaled, "embed_out", ("batch", "length", "embed"))
    return scaled

  def attend(
      self,
      hidden_states: jax.Array,
      *,
      precision: jax.lax.Precision,
  ) -> jax.Array:
    """Projects hidden states against the raw unscaled embedding table for tied `lm_head`."""
    weight = self.weight.get_value().astype(hidden_states.dtype)
    return jnp.einsum("...e,ve->...v", hidden_states, weight, precision=precision)


def compute_gemma3_inv_freq(
    head_dim: int,
    rope_theta: float,
    rope_linear_scaling_factor: float,
) -> jax.Array:
  """Computes Gemma 3 rotary inverse frequencies in host `np.float64` before casting to `float32`."""
  if head_dim <= 0 or head_dim % 2 != 0:
    raise ValueError(f"head_dim ({head_dim}) must be positive and even.")
  if rope_theta <= 0.0 or rope_linear_scaling_factor <= 0.0:
    raise ValueError("rope_theta and rope_linear_scaling_factor must be positive.")
  # Precompute in host numpy.float64 so rotary phases are immune to device exp2/log2 drift.
  exponents = np.arange(0, head_dim, 2, dtype=np.float64) / np.float64(head_dim)
  inv_freq_f64 = (1.0 / (np.float64(rope_theta) ** exponents)) / np.float64(rope_linear_scaling_factor)
  return jnp.asarray(inv_freq_f64, dtype=jnp.float32)


def apply_gemma3_rope(
    x: jax.Array,
    positions: jax.Array,
    inv_freq: jax.Array,
) -> jax.Array:
  """Applies half-split rotary position embeddings matching HF `Gemma3RotaryEmbedding`."""
  input_dtype = x.dtype
  pos_f32 = positions.astype(jnp.float32)
  if pos_f32.ndim == 1:
    pos_f32 = pos_f32[None, :]
  freqs = pos_f32[..., None] * inv_freq.astype(jnp.float32)[None, None, :]
  emb = jnp.concatenate([freqs, freqs], axis=-1)
  cos = jnp.cos(emb).astype(input_dtype)[..., None, :]
  sin = jnp.sin(emb).astype(input_dtype)[..., None, :]

  half_dim = x.shape[-1] // 2
  x1 = x[..., :half_dim]
  x2 = x[..., half_dim:]
  rotated = jnp.concatenate([-x2, x1], axis=-1)
  return x * cos + rotated * sin


def build_gemma3_attention_mask(
    seq_len: int,
    *,
    decoder_segment_ids: jax.Array | None,
    sliding_window: int | None,
) -> jax.Array:
  """Builds the causal / exclusive sliding-window + packed `decoder_segment_ids` boolean mask.

  Causality and the window use token indices (`0 <= q_idx - kv_idx < sliding_window`), as HF does; token
  positions only drive RoPE. Segment ID `0` marks padding, which attends to and is attended by nothing.

  Returns:
    Boolean mask of shape `(batch or 1, seq_len, seq_len)`.
  """
  idx = jnp.arange(seq_len, dtype=jnp.int32)
  delta = idx[:, None] - idx[None, :]
  mask = delta >= 0
  if sliding_window is not None:
    mask = mask & (delta < sliding_window)
  mask = mask[None, :, :]
  if decoder_segment_ids is not None:
    seg = decoder_segment_ids
    same_seg = seg[:, :, None] == seg[:, None, :]
    non_pad = (seg[:, :, None] != 0) & (seg[:, None, :] != 0)
    mask = mask & same_seg & non_pad
  return mask


class Gemma3MLP(nnx.Module):
  """Gemma 3 GeGLU feed-forward network (`wi_0`, `wi_1`, `wo`)."""

  def __init__(
      self,
      config: Gemma3Config,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    self.hidden_activation = config.hidden_activation
    self.dtype = resolve_jnp_dtype(config.dtype)
    self.param_dtype = resolve_jnp_dtype(config.weight_dtype)
    self.precision = resolve_precision(config.matmul_precision)
    self.sharding = _resolve_optional_sharding(sharding, None, config)
    s = self.sharding
    init_fn = nnx.initializers.normal(stddev=config.initializer_range)

    kw: dict[str, Any] = {
        "use_bias": False,
        "dtype": self.dtype,
        "param_dtype": self.param_dtype,
        "precision": self.precision,
        "kernel_init": init_fn,
        "rngs": rngs,
    }
    self.wi_0 = nnx.Linear(
        in_features=config.hidden_size,
        out_features=config.intermediate_size,
        kernel_metadata=_param_meta(s, "wi_0", ("embed", "mlp")),
        **kw,
    )
    self.wi_1 = nnx.Linear(
        in_features=config.hidden_size,
        out_features=config.intermediate_size,
        kernel_metadata=_param_meta(s, "wi_1", ("embed", "mlp")),
        **kw,
    )
    self.wo = nnx.Linear(
        in_features=config.intermediate_size,
        out_features=config.hidden_size,
        kernel_metadata=_param_meta(s, "wo", ("mlp", "embed")),
        **kw,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    approx = self.hidden_activation == "gelu_pytorch_tanh"
    gate = jax.nn.gelu(self.wi_0(x), approximate=approx)
    up = self.wi_1(x)
    hidden = (gate * up).astype(self.dtype)
    hidden = _constrain(self.sharding, hidden, "mlp_hidden", ("batch", "length", "mlp"))
    return self.wo(hidden)


class Gemma3Attention(nnx.Module):
  """Gemma 3 MQA/GQA self-attention with QK-norm, dual RoPE, and local/global masking."""

  def __init__(
      self,
      config: Gemma3Config,
      *,
      is_sliding: bool,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    self.is_sliding = is_sliding
    self.num_query_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.head_dim = config.head_dim
    self.scaling = float(config.query_pre_attn_scalar) ** -0.5
    self.sliding_window = config.sliding_window if is_sliding else None
    self.dtype = resolve_jnp_dtype(config.dtype)
    self.param_dtype = resolve_jnp_dtype(config.weight_dtype)
    self.precision = resolve_precision(config.matmul_precision)
    self.sharding = _resolve_optional_sharding(sharding, None, config)
    s = self.sharding
    init_fn = nnx.initializers.normal(stddev=config.initializer_range)

    if is_sliding:
      self.rope_theta = config.rope_local_base_freq
      self.rope_linear_scaling_factor = 1.0
    else:
      self.rope_theta = config.rope_theta
      self.rope_linear_scaling_factor = config.rope_linear_scaling_factor

    kw: dict[str, Any] = {
        "bias_shape": None,
        "dtype": self.dtype,
        "param_dtype": self.param_dtype,
        "precision": self.precision,
        "kernel_init": init_fn,
        "rngs": rngs,
    }
    self.query = nnx.Einsum(
        "...e,ehd->...hd",
        (config.hidden_size, self.num_query_heads, self.head_dim),
        kernel_metadata=_param_meta(s, "query", ("embed", "heads", "head_dim")),
        **kw,
    )
    self.key = nnx.Einsum(
        "...e,ehd->...hd",
        (config.hidden_size, self.num_kv_heads, self.head_dim),
        kernel_metadata=_param_meta(s, "key", ("embed", "kv_heads", "head_dim")),
        **kw,
    )
    self.value = nnx.Einsum(
        "...e,ehd->...hd",
        (config.hidden_size, self.num_kv_heads, self.head_dim),
        kernel_metadata=_param_meta(s, "value", ("embed", "kv_heads", "head_dim")),
        **kw,
    )
    self.out = nnx.Einsum(
        "...hd,hde->...e",
        (self.num_query_heads, self.head_dim, config.hidden_size),
        kernel_metadata=_param_meta(s, "out", ("heads", "head_dim", "embed")),
        **kw,
    )
    self.query_norm = Gemma3RMSNorm(
        self.head_dim,
        epsilon=config.rms_norm_eps,
        param_dtype=self.param_dtype,
        sharding=s,
        name="query_norm",
    )
    self.key_norm = Gemma3RMSNorm(
        self.head_dim,
        epsilon=config.rms_norm_eps,
        param_dtype=self.param_dtype,
        sharding=s,
        name="key_norm",
    )

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
  ) -> jax.Array:
    s = self.sharding
    batch_size, seq_len, _ = inputs.shape
    if decoder_positions is None:
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
    else:
      positions = decoder_positions

    q = _constrain(s, self.query(inputs), "query", ("batch", "length", "heads", "head_dim"))
    k = _constrain(s, self.key(inputs), "key", ("batch", "length", "kv_heads", "head_dim"))
    v = _constrain(s, self.value(inputs), "value", ("batch", "length", "kv_heads", "head_dim"))

    q = self.query_norm(q)
    k = self.key_norm(k)

    inv_freq = compute_gemma3_inv_freq(
        self.head_dim,
        self.rope_theta,
        self.rope_linear_scaling_factor,
    )
    q = apply_gemma3_rope(q, positions, inv_freq)
    k = apply_gemma3_rope(k, positions, inv_freq)

    groups = self.num_query_heads // self.num_kv_heads
    q_grouped = jnp.reshape(q, (batch_size, seq_len, self.num_kv_heads, groups, self.head_dim))
    logits = jnp.einsum("btkgd,bskd->bkgts", q_grouped, k, precision=self.precision)
    logits = logits * jnp.asarray(self.scaling, dtype=logits.dtype)

    mask = build_gemma3_attention_mask(
        seq_len,
        decoder_segment_ids=decoder_segment_ids,
        sliding_window=self.sliding_window,
    )
    mask_5d = mask[:, None, None, :, :]

    # Softmax in float32 regardless of activation dtype, as HF `softmax(..., dtype=torch.float32)`.
    masked_logits = jnp.where(mask_5d, logits.astype(jnp.float32), jnp.float32(-1e30))
    probs = jax.nn.softmax(masked_logits, axis=-1).astype(v.dtype)

    any_valid = jnp.any(mask_5d, axis=-1, keepdims=True)
    probs = jnp.where(any_valid, probs, jnp.zeros((), dtype=probs.dtype))

    context_g = jnp.einsum("bkgts,bskd->btkgd", probs, v, precision=self.precision)
    context = jnp.reshape(context_g, (batch_size, seq_len, self.num_query_heads, self.head_dim))
    context = _constrain(s, context, "attn_context", ("batch", "length", "heads", "head_dim"))
    return self.out(context)


class Gemma3DecoderLayer(nnx.Module):
  """Gemma 3 4-RMSNorm sandwich transformer decoder layer."""

  def __init__(
      self,
      config: Gemma3Config,
      *,
      layer_idx: int,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    sliding = config.is_sliding_layer(layer_idx)
    self.is_sliding = sliding
    self.dtype = resolve_jnp_dtype(config.dtype)
    self.param_dtype = resolve_jnp_dtype(config.weight_dtype)
    self.sharding = _resolve_optional_sharding(sharding, None, config)
    s = self.sharding

    self.pre_self_attention_norm = Gemma3RMSNorm(
        config.hidden_size,
        epsilon=config.rms_norm_eps,
        param_dtype=self.param_dtype,
        sharding=s,
        name="pre_self_attention_norm",
    )
    self.self_attention = Gemma3Attention(
        config,
        is_sliding=sliding,
        rngs=rngs,
        sharding=s,
    )
    self.post_self_attention_norm = Gemma3RMSNorm(
        config.hidden_size,
        epsilon=config.rms_norm_eps,
        param_dtype=self.param_dtype,
        sharding=s,
        name="post_self_attention_norm",
    )
    self.pre_ffw_norm = Gemma3RMSNorm(
        config.hidden_size,
        epsilon=config.rms_norm_eps,
        param_dtype=self.param_dtype,
        sharding=s,
        name="pre_ffw_norm",
    )
    self.mlp = Gemma3MLP(
        config,
        rngs=rngs,
        sharding=s,
    )
    self.post_ffw_norm = Gemma3RMSNorm(
        config.hidden_size,
        epsilon=config.rms_norm_eps,
        param_dtype=self.param_dtype,
        sharding=s,
        name="post_ffw_norm",
    )

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
  ) -> jax.Array:
    s = self.sharding
    x = _constrain(s, inputs, "layer_in", ("batch", "length", "embed"))
    attn_in = self.pre_self_attention_norm(x)
    attn_out = self.self_attention(
        attn_in,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    attn_out = self.post_self_attention_norm(attn_out)
    x = _constrain(s, x + attn_out, "post_attn_residual", ("batch", "length", "embed"))

    ffw_in = self.pre_ffw_norm(x)
    ffw_out = self.mlp(ffw_in)
    ffw_out = self.post_ffw_norm(ffw_out)
    return _constrain(s, x + ffw_out, "layer_out", ("batch", "length", "embed"))


def _remat_policy_fn(remat_policy: str) -> Any:
  """Maps `Gemma3Config.remat_policy` to a `jax.checkpoint` policy (`None` saves nothing)."""
  match remat_policy:
    case "full":
      return None
    case "minimal":
      return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    case _:
      raise ValueError(f"No checkpoint policy for remat_policy {remat_policy!r}.")


def _execute_layer_maybe_remat(
    sublayer: Gemma3DecoderLayer,
    x: jax.Array,
    *,
    decoder_positions: jax.Array | None,
    decoder_segment_ids: jax.Array | None,
    remat_policy: str,
) -> jax.Array:
  """Executes `sublayer` directly or under `jax.checkpoint(..., prevent_cse=True)`."""
  if remat_policy == "none":
    return sublayer(
        x,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
  graphdef, state = nnx.split(sublayer)

  # `prevent_cse=True`: sublayers are unrolled inside one scan step (or flat loop), where CSE could fold recompute.
  @functools.partial(jax.checkpoint, prevent_cse=True, policy=_remat_policy_fn(remat_policy))
  def _remat_sublayer(
      st: Any,
      h: jax.Array,
      pos: jax.Array | None,
      seg: jax.Array | None,
  ) -> jax.Array:
    merged = nnx.merge(graphdef, st)
    return merged(
        h,
        decoder_positions=pos,
        decoder_segment_ids=seg,
    )

  return _remat_sublayer(state, x, decoder_positions, decoder_segment_ids)


def _cycle_slice(state: Any, cycle_idx: int, param_scan_axis: int) -> Any:
  """Returns cycle `cycle_idx` of stacked `state`, dropping the stacked axis from `out_sharding` metadata."""

  def take(variable: nnx.Variable) -> nnx.Variable:
    rank = variable.ndim - 1
    axis = min(param_scan_axis, rank) if isinstance(variable, nnx.Param) else 0
    value = jnp.take(variable[...], cycle_idx, axis=axis)
    metadata = variable.get_metadata()
    spec = metadata["out_sharding"] if "out_sharding" in metadata else None
    if isinstance(spec, jax.sharding.PartitionSpec):
      entries = list(spec)
      del entries[axis]
      return variable.replace(value, out_sharding=jax.sharding.PartitionSpec(*entries))
    return variable.replace(value)

  return jax.tree.map(take, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


class Gemma3CycleBlock(nnx.Module):
  """One repeating Gemma 3 period (`sliding_window_pattern` layers: local SWA + global)."""

  def __init__(
      self,
      config: Gemma3Config,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    self.cycle_len = config.sliding_window_pattern
    self.remat_policy = config.remat_policy
    for block_idx in range(self.cycle_len):
      setattr(
          self,
          f"layers_{block_idx}",
          Gemma3DecoderLayer(
              config,
              layer_idx=block_idx,
              rngs=rngs,
              sharding=sharding,
          ),
      )

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
  ) -> jax.Array:
    x = inputs
    for block_idx in range(self.cycle_len):
      sublayer: Gemma3DecoderLayer = getattr(self, f"layers_{block_idx}")
      x = _execute_layer_maybe_remat(
          sublayer,
          x,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          remat_policy=self.remat_policy,
      )
    return x


class Gemma3Decoder(nnx.Module):
  """Gemma 3 transformer decoder supporting scanned 6-layer cycles and flat unscanned layout."""

  def __init__(
      self,
      config: Gemma3Config,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    self.config = config
    self.dtype = resolve_jnp_dtype(config.dtype)
    self.param_dtype = resolve_jnp_dtype(config.weight_dtype)
    self.precision = resolve_precision(config.matmul_precision)
    self.sharding = _resolve_optional_sharding(sharding, None, config)
    s = self.sharding

    self.cycle_len = config.sliding_window_pattern
    self.num_hidden_layers = config.num_hidden_layers
    self.stack_cycles = config.stack_cycles
    self.scan_layers = config.scan_layers
    self.param_scan_axis = config.param_scan_axis
    self.remat_policy = config.remat_policy

    if self.stack_cycles:
      self.num_cycles = config.num_hidden_layers // self.cycle_len
      self.num_remainder = config.num_hidden_layers % self.cycle_len
      if self.num_cycles > 0:
        cycles = [Gemma3CycleBlock(config, rngs=rngs, sharding=s) for _ in range(self.num_cycles)]
        self.layers = m3_scan.scan_run(cycles, param_scan_axis=config.param_scan_axis)
      start_rem = self.num_cycles * self.cycle_len
      # Attach trailing layers directly as `decoder.layers_{idx}` so `m3/infra/quantization.py`
      # (`r"decoder/layers(_\d+)?(/|$)"`) quantizes every projection alongside `self.layers`.
      for rem_idx in range(self.num_remainder):
        layer_idx = start_rem + rem_idx
        setattr(
            self,
            f"layers_{layer_idx}",
            Gemma3DecoderLayer(
                config,
                layer_idx=layer_idx,
                rngs=rngs,
                sharding=s,
            ),
        )
    else:
      self.num_cycles = 0
      self.num_remainder = 0
      for layer_idx in range(config.num_hidden_layers):
        setattr(
            self,
            f"layers_{layer_idx}",
            Gemma3DecoderLayer(
                config,
                layer_idx=layer_idx,
                rngs=rngs,
                sharding=s,
            ),
        )

    self.decoder_norm = Gemma3RMSNorm(
        config.hidden_size,
        epsilon=config.rms_norm_eps,
        param_dtype=self.param_dtype,
        sharding=s,
        name="decoder_norm",
    )
    if not config.tie_word_embeddings:
      init_fn = nnx.initializers.normal(stddev=config.initializer_range)
      self.logits_dense = nnx.Linear(
          in_features=config.hidden_size,
          out_features=config.vocab_size,
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          precision=self.precision,
          kernel_init=init_fn,
          kernel_metadata=_param_meta(s, "logits_dense", ("embed", "vocab")),
          rngs=rngs,
      )

  def _run_unscanned_stacked_cycles(
      self,
      y: jax.Array,
      *,
      decoder_positions: jax.Array | None,
      decoder_segment_ids: jax.Array | None,
  ) -> jax.Array:
    """Runs the stacked cycles as an unrolled Python loop (`scan_layers=False`, same state paths as scanned)."""
    graphdef, params, rest = nnx.split(self.layers, nnx.Param, ...)
    for cycle_idx in range(self.num_cycles):
      cycle = nnx.merge(
          graphdef,
          _cycle_slice(params, cycle_idx, self.param_scan_axis),
          _cycle_slice(rest, cycle_idx, 0),
      )
      y = Gemma3CycleBlock.__call__(
          cycle,
          y,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
      )
    return y

  def apply_output_head(
      self,
      shared_embedding: Gemma3TextScaledWordEmbedding,
      y: jax.Array,
  ) -> jax.Array:
    """Applies final `decoder_norm`, tied/untied vocabulary projection, and optional softcap."""
    s = self.sharding
    y = self.decoder_norm(y)
    if y.ndim == 2:
      y = _constrain(s, y, "pre_logits", ("batch", "embed"))
    else:
      y = _constrain(s, y, "pre_logits", ("batch", "length", "embed"))

    if self.config.tie_word_embeddings:
      logits = shared_embedding.attend(y, precision=self.precision)
    else:
      logits = self.logits_dense(y)

    # Logits leave the model in float32 (HF upcasts before the loss); the final softcap runs in float32.
    logits = logits.astype(jnp.float32)
    if self.config.final_logit_softcapping is not None:
      cap = jnp.float32(self.config.final_logit_softcapping)
      logits = jnp.tanh(logits / cap) * cap

    if logits.ndim == 2:
      return _constrain(s, logits, "logits", ("batch", "vocab"))
    return _constrain(s, logits, "logits", ("batch", "length", "vocab"))

  def __call__(
      self,
      shared_embedding: Gemma3TextScaledWordEmbedding,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      *,
      decoder_segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    y = shared_embedding(decoder_input_tokens)
    if self.stack_cycles:
      if self.num_cycles > 0:
        if self.scan_layers:
          y = self.layers(
              y,
              decoder_positions=decoder_positions,
              decoder_segment_ids=decoder_segment_ids,
          )
        else:
          y = self._run_unscanned_stacked_cycles(
              y,
              decoder_positions=decoder_positions,
              decoder_segment_ids=decoder_segment_ids,
          )
      start_rem = self.num_cycles * self.cycle_len
      for rem_idx in range(self.num_remainder):
        sublayer: Gemma3DecoderLayer = getattr(self, f"layers_{start_rem + rem_idx}")
        y = _execute_layer_maybe_remat(
            sublayer,
            y,
            decoder_positions=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            remat_policy=self.remat_policy,
        )
    else:
      for layer_idx in range(self.num_hidden_layers):
        sublayer: Gemma3DecoderLayer = getattr(self, f"layers_{layer_idx}")
        y = _execute_layer_maybe_remat(
            sublayer,
            y,
            decoder_positions=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            remat_policy=self.remat_policy,
        )

    logits = self.apply_output_head(shared_embedding, y)
    return logits, y


class Gemma3ForCausalLM(nnx.Module):
  """Top-level Gemma 3 autoregressive language model in MaxText m3 format."""

  def __init__(
      self,
      config: Any,
      mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
      **kwargs: Any,
  ):
    del kwargs
    if rngs is None:
      raise ValueError("Gemma3ForCausalLM requires a non-None nnx.Rngs instance.")
    g3_cfg = Gemma3Config.from_config(config)
    self.config = config
    self.gemma3_config = g3_cfg
    self.mesh = mesh

    ambient = jax.sharding.get_abstract_mesh()
    if (
        mesh is not None
        and not mesh.empty
        and not ambient.empty
        and (ambient.axis_sizes, ambient.axis_names) != (mesh.abstract_mesh.axis_sizes, mesh.abstract_mesh.axis_names)
    ):
      raise ValueError(f"Gemma3ForCausalLM was passed mesh {mesh} while a different mesh is in scope: {ambient}.")

    with contextlib.ExitStack() as stack:
      if mesh is not None and not mesh.empty and ambient.empty:
        if isinstance(mesh, jax.sharding.Mesh):
          try:
            stack.enter_context(jax.set_mesh(mesh))
          except ValueError as exc:
            if "set_mesh" not in str(exc):
              raise
            stack.enter_context(jax.sharding.use_abstract_mesh(mesh.abstract_mesh))
        else:
          abstract = mesh.abstract_mesh
          if all(t == jax.sharding.AxisType.Explicit for t in abstract.axis_types):
            abstract = abstract.update(axis_types=(jax.sharding.AxisType.Auto,) * len(abstract.axis_names))
          stack.enter_context(jax.sharding.use_abstract_mesh(abstract))

      self.sharding = _resolve_optional_sharding(sharding, mesh, g3_cfg)
      self.token_embedder = Gemma3TextScaledWordEmbedding(
          num_embeddings=g3_cfg.vocab_size,
          features=g3_cfg.hidden_size,
          dtype=g3_cfg.dtype,
          param_dtype=g3_cfg.weight_dtype,
          initializer_range=g3_cfg.initializer_range,
          sharding=self.sharding,
          rngs=rngs,
      )
      self.decoder = Gemma3Decoder(
          g3_cfg,
          rngs=rngs,
          sharding=self.sharding,
      )

    # Clear `eager_sharding` once parameters are placed so `nnx.Optimizer` rebuilds need no mesh scope.
    for _, node in nnx.iter_graph(self):
      if isinstance(node, nnx.Variable):
        meta = node.get_metadata()
        if (
            "eager_sharding" in meta
            and meta["eager_sharding"]
            and "out_sharding" in meta
            and meta["out_sharding"] is not None
        ):
          node.set_metadata(eager_sharding=False)

  def logits_from_hidden_states_for_vocab_tiling(
      self,
      hidden_states: jax.Array,
      deterministic: bool = True,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> jax.Array:
    """Projects hidden states to logits for vocab-tiled loss (`utils/vocabulary_tiling.py` call contract)."""
    del deterministic  # Gemma 3 has no dropout.
    if model_mode != MODEL_MODE_TRAIN:
      raise NotImplementedError(f"Gemma 3 m3 supports model_mode={MODEL_MODE_TRAIN!r} only, got {model_mode!r}.")
    return self.decoder.apply_output_head(self.token_embedder, hidden_states)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      *,
      cache: Any = None,
      enable_dropout: bool = False,
      true_length: int | None = None,
      **kwargs: Any,
  ) -> jax.Array:
    del enable_dropout, true_length  # No dropout (rates are rejected by the config) and no decode trimming.
    if cache is not None:
      raise NotImplementedError("Gemma 3 m3 does not implement KV-cache decoding.")
    if "model_mode" in kwargs and kwargs["model_mode"] != MODEL_MODE_TRAIN:
      raise NotImplementedError(
          f"Gemma 3 m3 supports model_mode={MODEL_MODE_TRAIN!r} only, got {kwargs['model_mode']!r}."
      )
    for media_key in _GEMMA3_UNSUPPORTED_MODEL_MEDIA_KWARGS:
      if media_key in kwargs and kwargs[media_key] is not None:
        raise NotImplementedError(f"Gemma 3 m3 implements text-only Gemma3ForCausalLM and does not accept {media_key!r}.")
    logits, _ = self.decoder(
        self.token_embedder,
        decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    return logits


def create_gemma3_model(
    config: Any,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    *,
    rngs: nnx.Rngs | None = None,
    **kwargs: Any,
) -> Gemma3ForCausalLM:
  """Factory function registered in `m3.models.MODEL_REGISTRY` for Gemma 3."""
  if rngs is None:
    rngs = nnx.Rngs(0)
  return Gemma3ForCausalLM(config, mesh, rngs=rngs, **kwargs)


def calculate_gemma3_tflops_per_device(
    config: Any,
    log: bool = True,
) -> tuple[float, float, float]:
  """Calculates per-device training TFLOPs for Gemma 3 (`(total, weight, attention)`).

  Global layers count the `S^2 / 2` causal triangle; local sliding-window layers count the
  `S * W - W^2 / 2` causal pairs inside the window. Reads the MaxText training fields
  `per_device_batch_size`, `max_target_length`, `gradient_accumulation_steps`, and `use_dpo`.
  """
  g3_cfg = Gemma3Config.from_config(config)
  b_dim = float(config.per_device_batch_size)
  s_dim = int(config.max_target_length)
  grad_accum = int(config.gradient_accumulation_steps)
  use_dpo = bool(config.use_dpo)

  batch_len = b_dim * s_dim
  h_dim = g3_cfg.hidden_size
  i_dim = g3_cfg.intermediate_size
  v_dim = g3_cfg.vocab_size
  num_layers = g3_cfg.num_hidden_layers
  num_q_heads = g3_cfg.num_attention_heads
  num_kv_heads = g3_cfg.num_key_value_heads
  head_dim = g3_cfg.head_dim

  unembedding_flops = 2.0 * batch_len * h_dim * v_dim
  q_dim = num_q_heads * head_dim
  kv_dim = num_kv_heads * head_dim
  attn_proj_flops_per_layer = 2.0 * batch_len * h_dim * (2 * q_dim + 2 * kv_dim)
  mlp_flops_per_layer = 6.0 * batch_len * h_dim * i_dim

  num_global_layers = sum(0 if g3_cfg.is_sliding_layer(i) else 1 for i in range(num_layers))
  num_local_layers = num_layers - num_global_layers
  window = min(g3_cfg.sliding_window, s_dim)

  global_attn_flops_per_layer = 2.0 * b_dim * (s_dim**2) * num_q_heads * head_dim
  local_attn_flops_per_layer = 4.0 * b_dim * (s_dim * window - 0.5 * (window**2)) * num_q_heads * head_dim

  total_weight_flops_fwd = unembedding_flops + num_layers * (attn_proj_flops_per_layer + mlp_flops_per_layer)
  total_attn_flops_fwd = num_global_layers * global_attn_flops_per_layer + num_local_layers * local_attn_flops_per_layer

  learnable_weight_tflops = (3.0 * total_weight_flops_fwd / 1e12) * grad_accum
  attention_tflops = (3.0 * total_attn_flops_fwd / 1e12) * grad_accum
  if use_dpo:
    ref_weight_tflops = learnable_weight_tflops / 3.0
    attention_tflops = attention_tflops + (attention_tflops / 3.0)
  else:
    ref_weight_tflops = 0.0

  total_tflops = learnable_weight_tflops + attention_tflops + ref_weight_tflops
  if log:
    pct_w = 100.0 * learnable_weight_tflops / max(total_tflops, 1e-12)
    pct_a = 100.0 * attention_tflops / max(total_tflops, 1e-12)
    print(
        f"Gemma3 per train step:\nTotal TFLOPs: {total_tflops:.4f}\n"
        f"split as {pct_w:.2f}% learnable weight flops and {pct_a:.2f}% attention flops"
    )
  return total_tflops, learnable_weight_tflops, attention_tflops


def compute_gemma3_loss(
    model: Gemma3ForCausalLM,
    *,
    tokens: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array | None = None,
    decoder_segment_ids: jax.Array | None = None,
    decoder_positions: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes masked `float32` cross-entropy loss and auxiliary token statistics for Gemma 3."""
  logits = model(
      tokens,
      decoder_positions=decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
  )
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  token_nll = -jnp.take_along_axis(log_probs, decoder_target_tokens[..., None], axis=-1).squeeze(-1)
  if decoder_target_mask is not None:
    weights = (decoder_target_mask != 0).astype(jnp.float32)
  else:
    weights = jnp.ones_like(token_nll, dtype=jnp.float32)
  total_weights = jnp.sum(weights)
  mean_loss = jnp.sum(token_nll * weights) / jnp.maximum(total_weights, 1.0)
  aux = {
      "total_loss": mean_loss,
      "total_weights": total_weights,
  }
  return mean_loss, aux


def gemma3_engine_loss_fn(
    model: Gemma3ForCausalLM,
    config: Any,
    data: Mapping[str, Any],
    dropout_rng: Any = None,
    params: Any = None,
    *,
    is_train: bool,
) -> tuple[Any, dict[str, jax.Array]]:
  """`MaxTextTrainingEngine` loss function adapter for Gemma 3."""
  del config, dropout_rng, params, is_train
  for media_key in _GEMMA3_UNSUPPORTED_ENGINE_MEDIA_KEYS:
    if media_key in data and data[media_key] is not None:
      raise NotImplementedError(
          f"Gemma 3 m3 implements text-only Gemma3ForCausalLM and does not accept batch key {media_key!r}."
      )
  metric_cls = _resolve_weighted_metric_cls()
  filtered = {k: data[k] for k in _GEMMA3_LOSS_BATCH_KEYS if k in data}
  target_mask = filtered["targets_segmentation"] if "targets_segmentation" in filtered else None
  seg_ids = filtered["inputs_segmentation"] if "inputs_segmentation" in filtered else None
  pos_ids = filtered["inputs_position"] if "inputs_position" in filtered else None
  mean_loss, aux = compute_gemma3_loss(
      model,
      tokens=filtered["inputs"],
      decoder_target_tokens=filtered["targets"],
      decoder_target_mask=target_mask,
      decoder_segment_ids=seg_ids,
      decoder_positions=pos_ids,
  )
  return (
      metric_cls(
          unreduced_sum=(mean_loss * aux["total_weights"]).astype(jnp.float32),
          denominator=aux["total_weights"],
      ),
      aux,
  )


def configure_gemma3_engine(engine: Any) -> Any:
  """Registers `gemma3_engine_loss_fn` on `MaxTextTrainingEngine` (`MODEL_ENGINE_CONFIG_REGISTRY`)."""
  engine.with_loss_fn(gemma3_engine_loss_fn, has_aux=True)
  engine.with_gen_model_input_fn(None)
  return engine
