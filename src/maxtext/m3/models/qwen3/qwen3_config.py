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

"""Typed Python Model Garden config for the Qwen3 dense model family."""

from typing import Any

import jax.numpy as jnp
from pydantic import Field, model_validator

from maxtext.m3.models.model_config import ModelConfig


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
  final_logits_soft_cap: float | None = Field(None, description="Optional tanh soft-cap on output logits.")

  @model_validator(mode="after")
  def _validate_attention_geometry(self) -> "Qwen3Config":
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
          f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
          f"num_key_value_heads ({self.num_key_value_heads})."
      )
    if self.head_dim % 2 != 0:
      raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE.")
    return self

  # Read-only alias properties matching `modeling_qwen3.py` field names.
  @property
  def emb_dim(self) -> int:
    return self.hidden_size

  @property
  def num_decoder_layers(self) -> int:
    return self.num_hidden_layers

  @property
  def dtype(self) -> jnp.dtype:
    return jnp.dtype(self.compute_dtype.value)

  @property
  def weight_dtype(self) -> jnp.dtype:
    return jnp.dtype(self.param_dtype.value)

  @property
  def mlp_dim(self) -> int:
    return self.intermediate_size

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

  def to_pyconfig_dict(self) -> dict[str, Any]:
    """Returns `MaxTextConfig` field defaults matching the Qwen3 dense model schema."""
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
        "rope_max_timescale": self.rope_theta,
        "use_qk_norm": self.use_qk_norm,
        "logits_via_embedding": self.tie_word_embeddings,
        "normalize_embedding_logits": self.normalize_embedding_logits,
        "enable_dropout": False,
    }

  @classmethod
  def qwen3_0_6b(cls, **overrides: Any) -> "Qwen3Config":
    """Returns the Qwen3-0.6B architecture configuration."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3-0.6b",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
    }
    defaults.update(overrides)
    return cls(**defaults)
