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

"""Typed Python Model Garden presets for Qwen3 and registered m3 model families."""

from typing import Any

from pydantic import Field

from maxtext.configs.types import DType
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.registry import ModelConfig, register_model_preset


class Qwen3Config(ModelConfig):
  """Frozen architecture configuration and Python presets for the Qwen3 dense family."""

  model_name: str = Field("qwen3-0.6b", description="Qwen3 preset identifier.")
  vocab_size: int = 151936
  hidden_size: int = 1024
  intermediate_size: int = 3072
  num_hidden_layers: int = 28
  num_attention_heads: int = 16
  num_key_value_heads: int = 8
  head_dim: int = 128
  rms_norm_eps: float = 1e-6
  rope_theta: float = 1_000_000.0
  tie_word_embeddings: bool = True
  use_qk_norm: bool = True

  @classmethod
  def qwen3_0_6b(cls, **overrides: Any) -> "Qwen3Config":
    """Returns the Qwen3-0.6B architecture preset."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3-0.6b",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def qwen3_1_7b(cls, **overrides: Any) -> "Qwen3Config":
    """Returns the Qwen3-1.7B architecture preset."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3-1.7b",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def qwen3_4b(cls, **overrides: Any) -> "Qwen3Config":
    """Returns the Qwen3-4B architecture preset."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3-4b",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def qwen3_8b(cls, **overrides: Any) -> "Qwen3Config":
    """Returns the Qwen3-8B architecture preset."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3-8b",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def qwen3_14b(cls, **overrides: Any) -> "Qwen3Config":
    """Returns the Qwen3-14B architecture preset."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3-14b",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def qwen3_32b(cls, **overrides: Any) -> "Qwen3Config":
    """Returns the Qwen3-32B architecture preset."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3-32b",
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def mini_config(cls, **overrides: Any) -> "Qwen3Config":
    """2-layer CPU-friendly miniature preset for fast hermetic unit tests."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3-mini",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "tie_word_embeddings": True,
        "param_dtype": DType.FLOAT32,
        "compute_dtype": DType.FLOAT32,
    }
    defaults.update(overrides)
    return cls(**defaults)


class DeepSeekV3Config(ModelConfig):
  """Typed MoE + MLA architecture configuration preset for DeepSeek-V3 / R1."""

  model_name: str = "deepseek3-671b"
  vocab_size: int = 129280
  hidden_size: int = 7168
  intermediate_size: int = 18432
  num_hidden_layers: int = 61
  num_attention_heads: int = 128
  num_key_value_heads: int = 128
  head_dim: int = 128
  rms_norm_eps: float = 1e-6
  rope_theta: float = 10_000.0
  tie_word_embeddings: bool = False
  use_qk_norm: bool = False
  n_routed_experts: int = Field(256, gt=0, description="Total number of routed MoE experts.")
  n_shared_experts: int = Field(1, ge=0, description="Number of shared experts.")
  num_experts_per_tok: int = Field(8, gt=0, description="Top-k experts selected per token.")
  moe_intermediate_size: int = Field(2048, gt=0, description="Per-expert intermediate hidden dimension.")
  first_k_dense_replace: int = Field(3, ge=0, description="Number of leading dense MLP layers.")
  q_lora_rank: int = Field(1536, ge=0, description="MLA query low-rank projection dimension.")
  kv_lora_rank: int = Field(512, gt=0, description="MLA KV latent compression dimension.")
  qk_nope_head_dim: int = Field(128, gt=0, description="Non-positional QK head dimension.")
  qk_rope_head_dim: int = Field(64, gt=0, description="Rotary QK head dimension.")
  v_head_dim: int = Field(128, gt=0, description="Value head dimension.")
  routed_scaling_factor: float = Field(2.5, gt=0.0, description="Router weight scaling factor.")
  router_bias_update_rate: float = Field(1e-3, ge=0.0, description="Auxiliary-loss-free router bias step rate.")

  @classmethod
  def deepseek3_671b(cls, **overrides: Any) -> "DeepSeekV3Config":
    return cls(**overrides)

  @classmethod
  def mini_config(cls, **overrides: Any) -> "DeepSeekV3Config":
    """2-layer CPU-friendly miniature MoE + MLA preset for schema and routing tests."""
    defaults: dict[str, Any] = {
        "model_name": "deepseek3-mini",
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "first_k_dense_replace": 1,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "param_dtype": DType.FLOAT32,
        "compute_dtype": DType.FLOAT32,
    }
    defaults.update(overrides)
    return cls(**defaults)


def _unimplemented_deepseek_constructor(*args: Any, **kwargs: Any) -> Any:
  del args, kwargs
  raise NotImplementedError("DeepSeek-V3 m3 NNX model constructor is scheduled in Workstream 4 Cohort 2.")


def _register_qwen3_presets() -> None:
  register_model_preset("qwen3-0.6b", Qwen3Config.qwen3_0_6b, create_qwen3_model)
  register_model_preset("qwen3-1.7b", Qwen3Config.qwen3_1_7b, create_qwen3_model)
  register_model_preset("qwen3-4b", Qwen3Config.qwen3_4b, create_qwen3_model)
  register_model_preset("qwen3-8b", Qwen3Config.qwen3_8b, create_qwen3_model)
  register_model_preset("qwen3-14b", Qwen3Config.qwen3_14b, create_qwen3_model)
  register_model_preset("qwen3-32b", Qwen3Config.qwen3_32b, create_qwen3_model)
  register_model_preset("qwen3-mini", Qwen3Config.mini_config, create_qwen3_model)
  register_model_preset("deepseek3-671b", DeepSeekV3Config.deepseek3_671b, _unimplemented_deepseek_constructor)
  register_model_preset("deepseek3-mini", DeepSeekV3Config.mini_config, _unimplemented_deepseek_constructor)


_register_qwen3_presets()
