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

from typing import Any, ClassVar

from maxtext.m3.models.model_config import ModelConfig
from pydantic import Field, model_validator


class Qwen3Config(ModelConfig):
  """Frozen architecture configuration for the Qwen3 dense model family."""

  model_type: str = Field("qwen3", description="HuggingFace model_type identifier.")
  vocab_size: int = Field(151936, gt=0, description="Vocabulary size.")
  hidden_size: int = Field(..., gt=0, description="Model hidden / embedding dimension.")
  intermediate_size: int = Field(..., gt=0, description="Feed-forward intermediate dimension.")
  num_hidden_layers: int = Field(..., gt=0, description="Number of transformer decoder layers.")
  num_attention_heads: int = Field(..., gt=0, description="Number of attention query heads.")
  num_key_value_heads: int = Field(8, gt=0, description="Number of grouped-query KV heads.")
  head_dim: int = Field(128, gt=0, description="Per-head attention dimension.")
  rms_norm_eps: float = Field(1e-6, gt=0.0, description="RMSNorm epsilon.")
  rope_theta: float = Field(1_000_000.0, gt=0.0, description="RoPE base period / max timescale.")
  tie_word_embeddings: bool = Field(..., description="Whether input and output embeddings are tied.")
  use_qk_norm: bool = Field(True, description="Whether to apply RMSNorm to Q and K before RoPE.")
  # Runtime numerics selection (no effect on parameters or checkpoints), not architecture.
  # TODO(m3): move to a TrainingConfig kernel/numerics section once one exists.
  float32_logits: bool = Field(False, description="Cast attention logits to float32 before softmax.")
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

  def mesh_divisibility_constraints(self) -> dict[str, dict[str, int]]:
    """Returns dims that must divide evenly by the `tensor` mesh axis sizes."""
    return {
        "tensor": {"num_attention_heads": self.num_attention_heads, "num_key_value_heads": self.num_key_value_heads},
    }

  PYCONFIG_CONSTANTS: ClassVar[dict[str, Any]] = {
      "mlp_activations": ["silu", "linear"],
      "decoder_block": "qwen3",
      "enable_dropout": False,
      "tokenizer_type": "huggingface",
  }
  PYCONFIG_KEYS: ClassVar[dict[str, str]] = {
      "base_emb_dim": "hidden_size",
      "base_mlp_dim": "intermediate_size",
      "base_num_decoder_layers": "num_hidden_layers",
      "base_num_query_heads": "num_attention_heads",
      "base_num_kv_heads": "num_key_value_heads",
      "head_dim": "head_dim",
      "vocab_size": "vocab_size",
      "normalization_layer_epsilon": "rms_norm_eps",
      "rope_max_timescale": "rope_theta",
      "use_qk_norm": "use_qk_norm",
      "logits_via_embedding": "tie_word_embeddings",
      "normalize_embedding_logits": "normalize_embedding_logits",
      "float32_logits": "float32_logits",
      "logits_dot_in_fp32": "logits_dot_in_fp32",
      "cast_logits_to_fp32": "cast_logits_to_fp32",
      "final_logits_soft_cap": "final_logits_soft_cap",
  }

  @classmethod
  def qwen3_0_6b(cls) -> "Qwen3Config":
    """Returns the `qwen3-0.6b` architecture configuration."""
    return cls(
        model_name="qwen3-0.6b",
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        tie_word_embeddings=True,
    )
