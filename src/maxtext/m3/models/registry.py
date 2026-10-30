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

"""Python-first Model Garden registry, typed `ModelConfig` base class, and `DEFAULT_M3_MODELS`."""

from collections.abc import Callable, Mapping
import json
import os
from typing import Any, Literal

from flax import nnx
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field

from maxtext.configs.types import DType

# Stage 3 default-on governance set: models added here only after passing all Stage 3 parity
# and throughput gates. Initially empty so `use_m3_model="auto"` leaves non-migrated runs untouched.
DEFAULT_M3_MODELS: set[str] = set()


def resolve_use_m3_model(use_m3_model: bool | Literal["auto"], model_name: str) -> bool:
  """Resolves `use_m3_model` (`True`, `False`, or `'auto'`) into a concrete boolean."""
  if isinstance(use_m3_model, bool):
    return use_m3_model
  if isinstance(use_m3_model, str):
    lowered = use_m3_model.strip().lower()
    if lowered == "auto":
      return model_name in DEFAULT_M3_MODELS
    if lowered in ("true", "1", "yes"):
      return True
    if lowered in ("false", "0", "no"):
      return False
  raise ValueError(f"`use_m3_model` must be a boolean or 'auto'; got {use_m3_model!r}.")


ModelConstructor = Callable[..., nnx.Module]


class ModelConfig(BaseModel):
  """Frozen, strictly validated base architecture configuration and preset resolver for m3 Model Garden families.

  Uses HuggingFace-aligned canonical field names (`hidden_size`, `intermediate_size`,
  `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`) while exposing read-only
  aliases (`emb_dim`, `mlp_dim`, `num_decoder_layers`, `num_query_heads`, `num_kv_heads`,
  `dtype`, `weight_dtype`) so m3 model constructors run identically with `ModelConfig`
  or `pyconfig` objects.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)

  model_name: str = Field("qwen3-0.6b", description="Canonical preset identifier.")
  vocab_size: int = Field(151936, gt=0, description="Vocabulary size.")
  hidden_size: int = Field(1024, gt=0, description="Model hidden / embedding dimension.")
  intermediate_size: int = Field(3072, gt=0, description="Feed-forward intermediate dimension.")
  num_hidden_layers: int = Field(28, gt=0, description="Number of transformer decoder layers.")
  num_attention_heads: int = Field(16, gt=0, description="Number of attention query heads.")
  num_key_value_heads: int = Field(8, gt=0, description="Number of grouped-query KV heads.")
  head_dim: int = Field(128, gt=0, description="Per-head attention dimension.")
  rms_norm_eps: float = Field(1e-6, gt=0.0, description="RMSNorm epsilon.")
  rope_theta: float = Field(1_000_000.0, gt=0.0, description="RoPE base period / max timescale.")
  tie_word_embeddings: bool = Field(True, description="Whether input and output embeddings are tied.")
  use_qk_norm: bool = Field(True, description="Whether to apply RMSNorm to Q and K before RoPE.")
  param_dtype: DType = Field(DType.FLOAT32, description="Master parameter storage dtype.")
  compute_dtype: DType = Field(DType.BFLOAT16, description="Forward/backward activation compute dtype.")
  float32_logits: bool = Field(True, description="Cast attention logits to float32 before softmax.")
  logits_dot_in_fp32: bool = Field(False, description="Compute output vocabulary projection dot in float32.")
  cast_logits_to_fp32: bool = Field(True, description="Cast output logits to float32.")
  normalize_embedding_logits: bool = Field(False, description="Normalize tied embedding logits by sqrt(hidden_size).")
  final_logits_soft_cap: float | None = Field(None, description="Optional tanh soft-cap on output logits.")

  @property
  def emb_dim(self) -> int:
    return self.hidden_size

  @property
  def mlp_dim(self) -> int:
    return self.intermediate_size

  @property
  def num_decoder_layers(self) -> int:
    return self.num_hidden_layers

  @property
  def num_query_heads(self) -> int:
    return self.num_attention_heads

  @property
  def num_kv_heads(self) -> int:
    return self.num_key_value_heads

  @property
  def normalization_layer_epsilon(self) -> float:
    return self.rms_norm_eps

  @property
  def rope_max_timescale(self) -> float:
    return self.rope_theta

  @property
  def logits_via_embedding(self) -> bool:
    return self.tie_word_embeddings

  @property
  def dtype(self) -> jnp.dtype:
    return jnp.dtype(self.compute_dtype.value)

  @property
  def weight_dtype(self) -> jnp.dtype:
    return jnp.dtype(self.param_dtype.value)

  @classmethod
  def from_hf_config(
      cls,
      hf_config_or_path: Mapping[str, Any] | str | os.PathLike[str],
      **overrides: Any,
  ) -> "ModelConfig":
    """Constructs a validated `ModelConfig` from a HuggingFace `config.json` path or mapping."""
    if isinstance(hf_config_or_path, (str, os.PathLike)):
      path = os.fspath(hf_config_or_path)
      if os.path.isdir(path):
        path = os.path.join(path, "config.json")
      with open(path, "r", encoding="utf-8") as f:
        raw_hf: dict[str, Any] = json.load(f)
    else:
      raw_hf = dict(hf_config_or_path)

    known_fields = set(cls.model_fields.keys())
    mapped: dict[str, Any] = {}
    for k, v in raw_hf.items():
      if k in known_fields and v is not None:
        mapped[k] = v

    if "head_dim" not in mapped and "hidden_size" in mapped and "num_attention_heads" in mapped:
      mapped["head_dim"] = int(mapped["hidden_size"]) // int(mapped["num_attention_heads"])
    if "num_key_value_heads" not in mapped and "num_attention_heads" in mapped:
      mapped["num_key_value_heads"] = mapped["num_attention_heads"]

    # Explicit overrides are applied last and strictly checked by `extra="forbid"`.
    mapped.update(overrides)
    return cls(**mapped)

  @classmethod
  def from_preset(cls, preset_name: str, **overrides: Any) -> "ModelConfig":
    """Returns a frozen `ModelConfig` for `preset_name`, applying validated `overrides`."""
    _ensure_builtin_presets_registered()
    if preset_name not in MODEL_GARDEN_REGISTRY:
      available = ", ".join(sorted(MODEL_GARDEN_REGISTRY))
      raise KeyError(f"Unknown m3 model preset {preset_name!r}. Available presets: {available}.")
    base_cfg = MODEL_GARDEN_REGISTRY[preset_name][0]()
    if cls is not ModelConfig and not isinstance(base_cfg, cls):
      raise TypeError(f"Preset {preset_name!r} resolves to {type(base_cfg).__name__}, which is not a {cls.__name__}.")
    if not overrides:
      return base_cfg
    merged = base_cfg.model_dump()
    merged.update(overrides)
    return type(base_cfg)(**merged)

  @classmethod
  def get_constructor(cls, preset_name: str) -> ModelConstructor:
    """Returns the model constructor registered for `preset_name`."""
    del cls
    _ensure_builtin_presets_registered()
    if preset_name not in MODEL_GARDEN_REGISTRY:
      available = ", ".join(sorted(MODEL_GARDEN_REGISTRY))
      raise KeyError(f"Unknown m3 model preset {preset_name!r}. Available presets: {available}.")
    return MODEL_GARDEN_REGISTRY[preset_name][1]


PresetBuilder = Callable[[], ModelConfig]

MODEL_GARDEN_REGISTRY: dict[str, tuple[PresetBuilder, ModelConstructor]] = {}


def register_model_preset(
    name: str,
    config_fn: PresetBuilder,
    model_cls_or_fn: ModelConstructor,
) -> None:
  """Registers a named Model Garden preset `(config_fn, model_constructor)`."""
  MODEL_GARDEN_REGISTRY[name] = (config_fn, model_cls_or_fn)


_BUILTINS_REGISTERED = False


def _ensure_builtin_presets_registered() -> None:
  """Lazily populates `MODEL_GARDEN_REGISTRY` with built-in model families."""
  global _BUILTINS_REGISTERED  # pylint: disable=global-statement
  if _BUILTINS_REGISTERED:
    return
  _BUILTINS_REGISTERED = True
  # Import built-in preset definitions to trigger registration without circular imports.
  from maxtext.m3.models.qwen3 import qwen3_config as _qwen3_cfg  # pylint: disable=import-outside-toplevel,unused-import
