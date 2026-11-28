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

"""Model Garden config for the Qwen3 dense model family."""

from typing import Any

from maxtext.m3.models.model_config import ModelConfig, coerce_dtype, extract_config_attrs
from pydantic import Field, model_validator

_QWEN3_ATTR_MAP: dict[str, tuple[str, ...]] = {
    "model_name": ("model_name",),
    "vocab_size": ("vocab_size",),
    "hidden_size": ("hidden_size", "emb_dim", "base_emb_dim"),
    "intermediate_size": ("intermediate_size", "mlp_dim", "base_mlp_dim"),
    "num_hidden_layers": ("num_hidden_layers", "num_decoder_layers", "base_num_decoder_layers"),
    "num_attention_heads": ("num_attention_heads", "num_query_heads", "base_num_query_heads"),
    "num_key_value_heads": ("num_key_value_heads", "num_kv_heads", "base_num_kv_heads"),
    "head_dim": ("head_dim",),
    "rms_norm_eps": ("rms_norm_eps", "normalization_layer_epsilon"),
    "rope_theta": ("rope_theta", "rope_max_timescale"),
    "tie_word_embeddings": ("tie_word_embeddings", "logits_via_embedding"),
    "use_qk_norm": ("use_qk_norm",),
    "float32_logits": ("float32_logits",),
    "logits_dot_in_fp32": ("logits_dot_in_fp32",),
    "cast_logits_to_fp32": ("cast_logits_to_fp32",),
    "normalize_embedding_logits": ("normalize_embedding_logits",),
    "final_logits_soft_cap": ("final_logits_soft_cap",),
    "param_dtype": ("param_dtype", "weight_dtype"),
    "compute_dtype": ("compute_dtype", "dtype"),
}


def extract_qwen3_kwargs(config: Any) -> dict[str, Any]:
  """Extracts `Qwen3Config` constructor kwargs from a legacy `MaxTextConfig` or attribute namespace."""
  kwargs = extract_config_attrs(config, _QWEN3_ATTR_MAP)
  if "param_dtype" in kwargs:
    kwargs["param_dtype"] = coerce_dtype(kwargs["param_dtype"])
  if "compute_dtype" in kwargs:
    kwargs["compute_dtype"] = coerce_dtype(kwargs["compute_dtype"])
  if kwargs.get("final_logits_soft_cap") == 0.0:
    kwargs["final_logits_soft_cap"] = None
  return kwargs


class Qwen3Config(ModelConfig):
  """Frozen architecture configuration for the Qwen3 dense model family."""

  model_type: str = Field("qwen3", description="HuggingFace model_type identifier.")
  model_name: str = Field("qwen3-0.6b", description="Canonical Qwen3 model identifier.")
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
  float32_logits: bool = Field(True, description="Cast attention logits to float32 before softmax.")
  logits_dot_in_fp32: bool = Field(False, description="Compute output vocabulary projection dot in float32.")
  cast_logits_to_fp32: bool = Field(True, description="Cast output logits to float32.")
  normalize_embedding_logits: bool = Field(False, description="Normalize tied embedding logits by sqrt(hidden_size).")
  final_logits_soft_cap: float | None = Field(None, gt=0.0, description="Optional tanh soft-cap on output logits.")

  @model_validator(mode="after")
  def _validate_attention_geometry(self) -> "Qwen3Config":
    """Validates attention head divisibility and even RoPE `head_dim`."""
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
          f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
          f"num_key_value_heads ({self.num_key_value_heads})."
      )
    if self.head_dim % 2 != 0:
      raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE.")
    return self

  def to_pyconfig_dict(self) -> dict[str, Any]:
    """Returns `MaxTextConfig` field defaults matching `qwen3-0.6b.yml`."""
    return {
        "base_emb_dim": self.hidden_size,
        "base_mlp_dim": self.intermediate_size,
        "base_num_decoder_layers": self.num_hidden_layers,
        "base_num_query_heads": self.num_attention_heads,
        "base_num_kv_heads": self.num_key_value_heads,
        "head_dim": self.head_dim,
        "mlp_activations": ["silu", "linear"],
        "vocab_size": self.vocab_size,
        "decoder_block": "qwen3",
        "normalization_layer_epsilon": self.rms_norm_eps,
        "rope_max_timescale": int(self.rope_theta) if self.rope_theta.is_integer() else self.rope_theta,
        "use_qk_norm": self.use_qk_norm,
        "logits_via_embedding": self.tie_word_embeddings,
        "normalize_embedding_logits": self.normalize_embedding_logits,
        "enable_dropout": False,
        "tokenizer_type": "huggingface",
    }

  @classmethod
  def from_config(cls, config: Any) -> "Qwen3Config":
    """Returns a `Qwen3Config` from either a `Qwen3Config` or a legacy `MaxTextConfig` / namespace."""
    if isinstance(config, cls):
      return config
    return cls(**extract_qwen3_kwargs(config))

  @classmethod
  def qwen3_0_6b(cls, **overrides: Any) -> "Qwen3Config":
    """Returns the Qwen3-0.6B architecture configuration."""
    return cls(**overrides)
