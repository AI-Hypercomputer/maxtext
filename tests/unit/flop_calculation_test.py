# Copyright 2023–2025 Google LLC
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

""" Tests for verifying FLOPs calculation in maxtext_utils.py"""

from typing import Any
import unittest
import pytest
from unittest.mock import MagicMock
from absl.testing import parameterized

from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils.maxtext_utils import calculate_tflops_training_per_device
from tests.utils.test_helpers import get_test_config_path


@pytest.mark.cpu_only
class FlopCalculation(parameterized.TestCase):
  """Tests for verifying FLOP calculation in MaxText"""

  def _get_model_config_args(
      self, model_name: str, max_target_length: int | None = None, per_device_batch_size: int | None = None
  ):
    """Returns the config args for a given model name, target length and batch size."""
    config_args = [None, get_test_config_path(f"models/{model_name}.yml"), "run_name=test"]
    if max_target_length is not None:
      config_args.append(f"max_target_length={max_target_length}")
    if per_device_batch_size is not None:
      config_args.append(f"per_device_batch_size={per_device_batch_size}")
    config_args.append("skip_jax_distributed_system=True")
    return config_args

  def _initialize_model_config(
      self,
      model_name: str,
      max_target_length: int | None = None,
      per_device_batch_size: int | None = None,
      **overrides: Any,
  ):
    """Initializes the model config."""
    config_args = self._get_model_config_args(
        model_name, max_target_length=max_target_length, per_device_batch_size=per_device_batch_size
    )
    return pyconfig.initialize(config_args, enable_checkpointing=False, **overrides)

  def assertFlopsAlmostEqual(self, flops1, flops2, rel_tol=5e-2):
    """Assert that two FLOPs values are almost equal, within 5% relative tolerance."""
    self.assertTrue(
        abs(flops1 - flops2) / max(abs(flops1), abs(flops2)) <= rel_tol,
        f"FLOPs values are not equal: {flops1} != {flops2} (rel_tol={rel_tol:.2e})",
    )

  def compute_regular_attention_flops_per_device(self, kwargs: dict) -> float:
    """
    Computes the attention TFLOPs per device for a Llama-style and Mixtral-style model.
    """
    # Configuration parameters from kwargs
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    N = kwargs["base_num_decoder_layers"]

    # Model dimensions
    D_head = kwargs["head_dim"]
    H_q = kwargs["base_num_query_heads"]

    # Attention flops
    # The factor of 2 is 1 for QK^T and 1 for SV.
    # 3 for forward plus backward pass
    # This accounts for causal masking.
    attention_flops = 2 * 3 * N * B * (S**2) * H_q * D_head

    return attention_flops / 1e12  # return tflops

  # ========== Unit Tests for Direct FLOP Calculation Functions ==========
  def compute_deepseek_attention_flops_per_device(self, kwargs: dict) -> float:
    """
    Computes the total training TFLOPs per device for a DeepSeek-style model.
    """
    # Configuration parameters
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    N = kwargs["base_num_decoder_layers"]

    # Attention FLOPs (MLA) - per layer
    qk_nope_hd = kwargs["qk_nope_head_dim"]
    qk_rope_hd = kwargs["qk_rope_head_dim"]
    v_hd = kwargs["v_head_dim"]
    H_q = kwargs["base_num_query_heads"]

    # Attention flops
    # 3 for forward plus backward pass
    # This accounts for causal masking.
    attention_flops = 3 * N * B * (S**2) * H_q * (qk_nope_hd + qk_rope_hd + v_hd)

    return attention_flops / 1e12  # Convert to TFLOPs (10^12)

  def compute_gpt_attention_flops_per_device(self, kwargs: dict) -> float:
    """
    Computes the total training TFLOPs per device for a GPT-style model.
    """
    # Configuration parameters from kwargs
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    W = kwargs["sliding_window_size"]
    N = kwargs["base_num_decoder_layers"]

    # Model dimensions
    D_head = kwargs["head_dim"]
    H_q = kwargs["base_num_query_heads"]

    # Attention flops (mixed with global and local_sliding attentions)
    # The factor of 2 is 1 for QK^T and 1 for SV.
    # 3 for forward plus backward pass
    # This accounts for causal masking.
    attention_flops = 2 * 3 * B * (N / 2 * S**2 + N / 2 * W**2) * H_q * D_head

    return attention_flops / 1e12  # return tflops

  def compute_qwen3_next_attention_flops_per_device(self, kwargs: dict) -> float:
    """
    Computes the total training TFLOPs per device for a Qwen3-Next model.
    Only counts the attention mechanism operations (non-weights).
    """
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    N = kwargs["base_num_decoder_layers"]
    cycle_interval = kwargs["inhomogeneous_layer_cycle_interval"]

    # Layer counts
    num_full_layers = N // cycle_interval
    num_linear_layers = N - num_full_layers

    # 1. Full Attention FLOPs (Causal)
    D_head = kwargs["head_dim"]
    H_q = kwargs["base_num_query_heads"]
    # 2 for QK^T and SV, 3 for fwd+bwd.
    # Note: maxtext_utils divides by 2 for causal masking.
    # Formula: 2 * 3 * B * S^2 * H * D
    full_attn_flops = 2 * 3 * num_full_layers * B * (S**2) * H_q * D_head

    # 2. Linear Attention (Gated Delta Net) FLOPs
    H_v = kwargs["gdn_num_value_heads"]
    D_k = kwargs["gdn_key_head_dim"]
    D_v = kwargs["gdn_value_head_dim"]
    C = kwargs["gdn_chunk_size"]

    # Formulas from maxtext_utils.calculate_gated_delta_net_flops_per_device
    flops_intra = 2 * B * S * H_v * C * (2 * D_k + D_v) + (B * H_v * S * C**2)
    flops_inter = (2 * B * S * H_v * C * (D_k + D_v)) + (6 * B * S * H_v * D_k * D_v)

    # 3 for fwd+bwd
    linear_attn_flops = 3 * num_linear_layers * (flops_intra + flops_inter)

    return (full_attn_flops + linear_attn_flops) / 1e12

  def test_qwen3_next_flops(self):
    """Test Qwen3-Next Flops calculation"""
    cfg = self._initialize_model_config(
        "qwen3-next-80b-a3b",
        max_target_length=4096,
        per_device_batch_size=1,
    )
    kwargs = cfg.get_keys()

    # 1. Calculate Attention TFLOPs
    attention_tflops = self.compute_qwen3_next_attention_flops_per_device(kwargs)

    # 2. Calculate Learnable Weight Active Params
    # Config Shortcuts
    emb_dim = kwargs["base_emb_dim"]
    vocab = kwargs["vocab_size"]
    N = kwargs["base_num_decoder_layers"]

    # MoE Active Params (per layer)
    # FFN uses SwiGLU (3 matrices), Qwen3-Next has 1 shared + N routed experts
    # Params = Gate + Shared + Routed
    # Gate: emb_dim * num_experts
    # Expert: 3 * emb_dim * moe_mlp_dim
    moe_mlp_dim = kwargs["base_moe_mlp_dim"]
    num_experts = kwargs["num_experts"]
    num_routed = kwargs["num_experts_per_tok"]

    params_moe_layer = (
        (emb_dim * num_experts) + (3 * emb_dim * moe_mlp_dim * 1) + (3 * emb_dim * moe_mlp_dim * num_routed)
    )

    # Full Attention Params (per full layer)
    Hq = kwargs["base_num_query_heads"]
    Hkv = kwargs["base_num_kv_heads"]
    Hd = kwargs["head_dim"]
    # Q, K, V, Out projections
    params_full_attn = (emb_dim * (Hq + 2 * Hkv) * Hd) + (Hq * Hd * emb_dim)

    # GDN Linear Attention Params (per linear layer)
    Hk_g = kwargs["gdn_num_key_heads"]
    Hv_g = kwargs["gdn_num_value_heads"]
    Dk_g = kwargs["gdn_key_head_dim"]
    Dv_g = kwargs["gdn_value_head_dim"]
    K_conv = kwargs["gdn_conv_kernel_dim"]

    K_dim = Hk_g * Dk_g
    V_dim = Hv_g * Dv_g

    # Projections: qkvz (in->2K+2V), ba (in->2Hv), out (V->in)
    params_gdn_proj = (emb_dim * (2 * K_dim + 2 * V_dim)) + (emb_dim * 2 * Hv_g) + (V_dim * emb_dim)
    # Conv: depthwise on 2K+V
    params_gdn_conv = (2 * K_dim + V_dim) * K_conv

    params_gdn_layer = params_gdn_proj + params_gdn_conv

    # Total Active Params
    # 12 Full Layers, 36 Linear Layers
    num_full = N // kwargs["inhomogeneous_layer_cycle_interval"]
    num_linear = N - num_full

    total_active_params = (
        (vocab * emb_dim)
        + (num_full * (params_full_attn + params_moe_layer))
        + (num_linear * (params_gdn_layer + params_moe_layer))
    )

    # Weight TFLOPs = 6 * B * S * P
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    weight_tflops = 6 * B * S * total_active_params / 1e12

    golden_tflops = weight_tflops + attention_tflops

    # Run Calculation
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)

    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  def test_llama2_7b_flops(self):
    """Test Llama2 7b Flops calculation with default parameters"""
    cfg = self._initialize_model_config(
        "llama2-7b",
        max_target_length=2048,
        per_device_batch_size=12,
    )
    kwargs = cfg.get_keys()
    B = cfg.per_device_batch_size
    S = cfg.max_target_length
    attention_flops = self.compute_regular_attention_flops_per_device(kwargs)
    # Llama2-7b has ~6.74B parameters
    # https://adithyask.medium.com/from-7b-to-8b-parameters-understanding-weight-matrix-changes-in-llama-transformer-models-31ea7ed5fd88
    golden_param_size = 6.74e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  def test_llama3_8b_flops(self):
    """Test Llama3 8b Flops calculation with default parameters"""
    cfg = self._initialize_model_config(
        "llama3-8b",
        max_target_length=2048,
        per_device_batch_size=4,
    )
    kwargs = cfg.get_keys()
    B = cfg.per_device_batch_size
    S = cfg.max_target_length
    attention_flops = self.compute_regular_attention_flops_per_device(kwargs)
    # LLaMA3-8b has ~8.03B parameters
    # https://adithyask.medium.com/from-7b-to-8b-parameters-understanding-weight-matrix-changes-in-llama-transformer-models-31ea7ed5fd88
    # Note: The commonly cited 8.03B parameter count for Llama 3 8B corresponds to a version with UNTIED embeddings.
    # Here we consider TIED embedding table, which reduces param count to 7.50B
    golden_param_size = 7.50e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  def test_mixtral_8x7b_flops(self):
    """Test Mixtral 8x7b Flops calculation"""
    cfg = self._initialize_model_config(
        "mixtral-8x7b",
        max_target_length=8192,
        per_device_batch_size=4,
    )
    kwargs = cfg.get_keys()
    B = cfg.per_device_batch_size
    S = cfg.max_target_length
    attention_flops = self.compute_regular_attention_flops_per_device(kwargs)
    # mixtral-8x7b has ~12.9B active parameters
    # https://mistral.ai/news/mixtral-of-experts
    golden_param_size = 12.9e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  def test_deepseek2_16b_flops(self):
    """Test DeepSeek2-16b FLops calculation"""
    cfg = self._initialize_model_config(
        "deepseek2-16b",
        max_target_length=8192,
        per_device_batch_size=4,
    )
    kwargs = cfg.get_keys()
    B = cfg.per_device_batch_size
    S = cfg.max_target_length
    attention_flops = self.compute_deepseek_attention_flops_per_device(kwargs)
    # deepseek2-16b has ~2.4B active parameters
    # https://arxiv.org/pdf/2405.04434
    golden_param_size = 2.4e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  def test_gpt_oss_20b_flops(self):
    """Test GPT OSS 20B Flops calculation"""
    cfg = self._initialize_model_config(
        "gpt-oss-20b",
        max_target_length=8192,
        per_device_batch_size=4,
    )
    kwargs = cfg.get_keys()
    B = cfg.per_device_batch_size
    S = cfg.max_target_length
    attention_flops = self.compute_gpt_attention_flops_per_device(kwargs)
    # gpt-oss-20b has ~3.6B active parameters
    # https://openai.com/index/introducing-gpt-oss/
    golden_param_size = 3.6e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  def test_deepseek32_671b_flops(self):
    """Test DeepSeek3.2-671b FLops calculation"""
    cfg = self._initialize_model_config(
        "deepseek3.2-671b",
        max_target_length=4096,
        per_device_batch_size=4,
        attention="dot_product",
    )
    kwargs = cfg.get_keys()
    B = cfg.per_device_batch_size
    S = cfg.max_target_length
    attention_flops = self.compute_deepseek_attention_flops_per_device(kwargs)
    # deepseek3-671b has ~37B active parameters
    # https://arxiv.org/pdf/2412.19437
    golden_param_size = 37e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  def test_custom_engram_flops(self):
    """Test model with Engram Flops calculation"""
    cfg = self._initialize_model_config(
        "deepseek2-16b",
        max_target_length=8192,
        per_device_batch_size=4,
    )
    kwargs = cfg.get_keys()
    B = cfg.per_device_batch_size
    S = cfg.max_target_length
    G = cfg.mhc_expansion_rate
    D = cfg.base_emb_dim
    K = cfg.engram_kernel_size
    H = cfg.engram_num_heads
    H_D = cfg.engram_head_dim
    L = len(cfg.engram_layers)
    N = cfg.engram_max_ngram_size

    attention_flops = self.compute_deepseek_attention_flops_per_device(kwargs)
    # deepseek2-16b has ~2.4B active parameters
    # https://arxiv.org/pdf/2405.04434
    golden_param_size = 2.4e9

    # calculate Engram active params
    engram_dim = H * H_D * (N - 1)
    key_params = engram_dim * (G * D)
    value_params = engram_dim * D
    conv_params = K * (G * D)
    engram_active_params = L * (key_params + value_params + conv_params)
    golden_tflops = 6 * B * S * (golden_param_size + engram_active_params) / 1e12 + attention_flops

    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  def test_calculate_gemma2_tflops_training_per_device(self):
    """Test calculate_gemma2_tflops_training_per_device."""
    config = MagicMock()
    config.per_device_batch_size = 2
    config.max_target_length = 8192
    config.sliding_window_size = 4096
    config.num_query_heads = 8
    config.head_dim = 128
    config.num_decoder_layers = 10
    config.share_kv_projections = False
    config.global_head_dim = None
    config.global_num_kv_heads = None

    total_ffn_flops = 100
    qkv_flops = 200
    projection_flops = 150
    embedding_flops = 50

    attention_tflops, learnable_weight_tflops = maxtext_utils.calculate_gemma2_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
    )

    B = config.per_device_batch_size
    T = config.max_target_length
    W = config.sliding_window_size
    H = config.num_query_heads
    D = config.head_dim

    expected_global = 2 * B * T * T * H * D
    expected_local = 4 * B * (T * W - 0.5 * W * W) * H * D
    expected_causal = expected_global + expected_local

    expected_attention_tflops = expected_causal * config.num_decoder_layers * 3 / 10**12

    self.assertAlmostEqual(attention_tflops, expected_attention_tflops, places=5)

    expected_learnable = (
        total_ffn_flops + qkv_flops + projection_flops
    ) * config.num_decoder_layers * 2 + embedding_flops
    expected_learnable_tflops = expected_learnable * 3 / 10**12
    self.assertAlmostEqual(learnable_weight_tflops, expected_learnable_tflops, places=5)

  def test_calculate_mixed_attention_model_tflops_training_per_device(self):
    """Test calculate_mixed_attention_model_tflops_training_per_device."""
    config = MagicMock()
    config.per_device_batch_size = 2
    config.max_target_length = 8192
    config.sliding_window_size = 4096
    config.num_query_heads = 8
    config.head_dim = 128
    config.num_decoder_layers = 10
    config.share_kv_projections = False
    config.global_head_dim = None
    config.global_num_kv_heads = None

    config.num_kv_heads = 4
    config.emb_dim = 512

    total_ffn_flops = 100
    embedding_flops = 50
    attention_pattern_length = 5

    qkv_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * (config.num_query_heads + 2 * config.num_kv_heads)
        * config.head_dim
    )
    projection_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * config.num_query_heads
        * config.head_dim
    )

    attention_tflops, learnable_weight_tflops = maxtext_utils.calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops, attention_pattern_length
    )

    B = config.per_device_batch_size
    T = config.max_target_length
    W = config.sliding_window_size
    H = config.num_query_heads
    D = config.head_dim

    num_global = 10 // 5
    num_local = 10 - num_global

    expected_global_per_layer = 2 * B * T * T * H * D
    expected_local_per_layer = 4 * B * (T * W - 0.5 * W * W) * H * D
    expected_causal = num_global * expected_global_per_layer + num_local * expected_local_per_layer

    expected_attention_tflops = expected_causal * 3 / 10**12

    self.assertAlmostEqual(attention_tflops, expected_attention_tflops, places=5)

    expected_learnable = (
        total_ffn_flops * config.num_decoder_layers
        + (qkv_flops + projection_flops) * config.num_decoder_layers
        + embedding_flops
    )
    expected_learnable_tflops = expected_learnable * 3 / 10**12
    self.assertAlmostEqual(learnable_weight_tflops, expected_learnable_tflops, places=5)

  def test_calculate_gemma4_tflops_training_per_device(self):
    """Test calculate_gemma4_tflops_training_per_device."""
    config = MagicMock()
    config.per_device_batch_size = 2
    config.max_target_length = 8192
    config.sliding_window_size = 4096
    config.num_query_heads = 16
    config.num_kv_heads = 8
    config.head_dim = 256
    config.num_decoder_layers = 12
    config.share_kv_projections = True
    config.global_head_dim = 512
    config.global_num_kv_heads = 2
    config.emb_dim = 2816

    total_ffn_flops = 0
    embedding_flops = 0

    attention_tflops, learnable_weight_tflops = maxtext_utils.calculate_gemma4_tflops_training_per_device(
        config, total_ffn_flops, embedding_flops, attention_pattern_length=6
    )

    B = config.per_device_batch_size
    T = config.max_target_length
    W = config.sliding_window_size
    H = config.num_query_heads
    D = config.head_dim
    GD = config.global_head_dim

    num_global = 12 // 6
    num_local = 12 - num_global

    expected_global_per_layer = 2 * B * T * T * H * GD
    expected_local_per_layer = 4 * B * (T * W - 0.5 * W * W) * H * D
    expected_causal = num_global * expected_global_per_layer + num_local * expected_local_per_layer

    expected_attention_tflops = expected_causal * 3 / 10**12

    self.assertAlmostEqual(attention_tflops, expected_attention_tflops, places=5)

    # Weights checking
    # local: share_kv_projections = False -> kv_multiplier = 2
    # qkv + proj
    expected_local_weights = (2 * B * T * config.emb_dim * (H + 2 * config.num_kv_heads) * D) + (
        2 * B * T * config.emb_dim * H * D
    )
    expected_global_weights = (2 * B * T * config.emb_dim * (H + 1 * config.global_num_kv_heads) * GD) + (
        2 * B * T * config.emb_dim * H * GD
    )

    expected_learnable = num_local * expected_local_weights + num_global * expected_global_weights
    expected_learnable_tflops = expected_learnable * 3 / 10**12

    self.assertAlmostEqual(learnable_weight_tflops, expected_learnable_tflops, places=5)

  def test_calculate_llama4_attention_tflops(self):
    """Test calculate_llama4_attention_tflops."""
    config = MagicMock()
    config.num_decoder_layers = 16
    config.max_target_length = 4096
    config.chunk_attn_window_size = 1024
    config.nope_layer_interval = 4
    config.per_device_batch_size = 2
    config.num_query_heads = 16
    config.head_dim = 128

    attention_tflops = maxtext_utils.calculate_llama4_attention_tflops(config)

    # Manual calculation
    num_global_layers = 16 // 4  # 4
    num_chunked_layers = 16 - 4  # 12
    global_flops = 4 * 2 * config.max_target_length**2 * config.num_query_heads * config.head_dim

    num_chunks = 4096 // 1024  # 4
    chunked_complexity = num_chunks * config.chunk_attn_window_size**2
    chunked_flops = 4 * 2 * chunked_complexity * config.num_query_heads * config.head_dim

    noncausal = (num_global_layers * global_flops) + (num_chunked_layers * chunked_flops)
    expected_attention_tflops = (noncausal / 2) * 3 / 10**12

    self.assertAlmostEqual(attention_tflops, expected_attention_tflops, places=5)

  def test_calculate_gemma4_tflops_training_per_device_shared_kv(self):
    """Test calculate_gemma4_tflops_training_per_device_shared_kv."""
    config = MagicMock()
    config.per_device_batch_size = 2
    config.max_target_length = 8192
    config.sliding_window_size = 1024
    config.num_query_heads = 32
    config.num_kv_heads = 8
    config.head_dim = 128
    config.num_decoder_layers = 12
    config.share_kv_projections = True
    config.global_head_dim = 128
    config.global_num_kv_heads = 8
    config.emb_dim = 4096

    total_ffn_flops_all_layers = 123456789
    embedding_flops = 333333333

    attention_tflops, learnable_weight_tflops = maxtext_utils.calculate_gemma4_tflops_training_per_device(
        config, total_ffn_flops_all_layers, embedding_flops, attention_pattern_length=6
    )

    B = config.per_device_batch_size
    T = config.max_target_length
    W = min(config.sliding_window_size, config.max_target_length)
    H = config.num_query_heads
    D = config.head_dim
    GD = config.global_head_dim
    GKH = config.global_num_kv_heads

    num_global = 12 // 6
    num_local = 12 - num_global

    expected_global_per_layer = 2 * B * T * T * H * GD
    expected_local_per_layer = 4 * B * (T * W - 0.5 * W * W) * H * D
    expected_causal = num_global * expected_global_per_layer + num_local * expected_local_per_layer

    expected_attention_tflops = expected_causal * 3 / 10**12

    self.assertAlmostEqual(attention_tflops, expected_attention_tflops, places=5)

    kv_multiplier = 1 if config.share_kv_projections else 2
    expected_global_qkv_flops_per_layer = 2 * B * T * config.emb_dim * (H + kv_multiplier * GKH) * GD
    expected_global_projection_flops_per_layer = 2 * B * T * config.emb_dim * H * GD

    expected_local_qkv_flops_per_layer = 2 * B * T * config.emb_dim * (H + 2 * config.num_kv_heads) * D
    expected_local_projection_flops_per_layer = 2 * B * T * config.emb_dim * H * D

    expected_learnable = (
        total_ffn_flops_all_layers
        + (expected_local_qkv_flops_per_layer + expected_local_projection_flops_per_layer) * num_local
        + (expected_global_qkv_flops_per_layer + expected_global_projection_flops_per_layer) * num_global
        + embedding_flops
    )
    expected_learnable_tflops = expected_learnable * 3 / 10**12

    self.assertAlmostEqual(learnable_weight_tflops, expected_learnable_tflops, places=5)

  def test_calculate_routed_and_shared_ffn_tflops_per_device(self):
    """Test calculate_routed_and_shared_ffn_tflops_per_device."""
    config = MagicMock()
    config.decoder_block = maxtext_utils.DecoderBlockType.DEEPSEEK
    config.per_device_batch_size = 1
    config.max_target_length = 2048
    config.emb_dim = 1024
    config.first_num_dense_layers = 2
    config.num_decoder_layers = 8
    config.num_experts = 4
    config.mlp_dim = 2048
    config.moe_mlp_dim = 1024
    config.shared_experts = 1
    config.num_experts_per_tok = 2
    config.mlp_activations = ["silu", "linear"]

    ffn_tflops = maxtext_utils.calculate_routed_and_shared_ffn_tflops_per_device(config)

    B = config.per_device_batch_size
    T = config.max_target_length
    E = config.emb_dim
    N = config.num_experts

    gate_flops = 2 * B * T * E * N

    # dense ffn mamtul (silu: 2 * mlp_dim)
    dense_ffn1 = 2 * B * T * E * (2 * config.mlp_dim)
    dense_ffn2 = 2 * B * T * config.mlp_dim * E
    dense_flops_per_layer = dense_ffn1 + dense_ffn2

    # moe ffn mamtul
    moe_ffn1 = 2 * B * T * E * (2 * config.moe_mlp_dim)
    moe_ffn2 = 2 * B * T * config.moe_mlp_dim * E
    moe_flops_per_expert = moe_ffn1 + moe_ffn2

    shared_flops = moe_flops_per_expert * config.shared_experts
    routed_flops = moe_flops_per_expert * config.num_experts_per_tok

    # layers
    dense_layers = config.first_num_dense_layers
    moe_layers = config.num_decoder_layers - config.first_num_dense_layers

    expected_total = (dense_flops_per_layer * dense_layers) + ((gate_flops + shared_flops + routed_flops) * moe_layers)

    self.assertAlmostEqual(ffn_tflops, expected_total, places=5)

  # ========== Parameterized Tests for Multiple Standard Models ==========

  def _verify_flops(self, model_name, max_target_length=1):
    """
    Verifies that for a given sequence length, the total compute matches exactly what we
    expect from manual parameter extraction using the `6 * active_params * tokens` estimation rule,
    plus the expected attention flops.
    """
    config_args = [
        None,
        get_test_config_path(f"models/{model_name}.yml"),
        "run_name=test",
        f"max_target_length={max_target_length}",
        "per_device_batch_size=1",
        "skip_jax_distributed_system=True",
    ]
    config = pyconfig.initialize(config_args, enable_checkpointing=False)
    tflops, _, attention_tflops = calculate_tflops_training_per_device(config)

    # 1. Determine layer counts (dense vs MoE)
    num_dense, num_moe = maxtext_utils.get_dense_moe_layers(config)

    # 2. Calculate FFN (Feed-Forward Network) parameters
    dense_ffn_params = (config.emb_dim * config.mlp_dim * 2 + config.mlp_dim * config.emb_dim) * num_dense
    moe_ffn_params = (
        (config.emb_dim * config.num_experts)  # gate (router module)
        + (
            (config.emb_dim * config.moe_mlp_dim * 2 + config.moe_mlp_dim * config.emb_dim) * config.shared_experts
        )  # shared experts
        + (
            (config.emb_dim * config.moe_mlp_dim * 2 + config.moe_mlp_dim * config.emb_dim) * config.num_experts_per_tok
        )  # routed experts
    ) * num_moe
    total_ffn_params = dense_ffn_params + moe_ffn_params

    # 3. Calculate embedding parameters
    embedding_params = config.vocab_size * config.emb_dim
    # If not sharing weights, there is a separate unembedding layer
    if getattr(config, "logits_via_embedding", False) is False:
      embedding_params += config.vocab_size * config.emb_dim

    # 4. Resolve attention pattern lengths based on architecture (local sliding vs global causal)
    attention_pattern_length = getattr(config, "attention_pattern_length", config.num_decoder_layers)
    if not attention_pattern_length:
      attention_pattern_length = config.num_decoder_layers

    if getattr(config, "decoder_block", None) == maxtext_utils.DecoderBlockType.GPT_OSS:
      attention_pattern_length = 2
    elif getattr(config, "decoder_block", None) in (
        maxtext_utils.DecoderBlockType.GEMMA4,
        maxtext_utils.DecoderBlockType.GEMMA3,
    ):
      attention_pattern_length = 6

    num_global_layers = config.num_decoder_layers // attention_pattern_length
    num_local_layers = config.num_decoder_layers - num_global_layers

    # 5. Calculate QKV and Projection parameters based on attention type
    if getattr(config, "attention_type", "") == "mla":
      # Multi-Head Latent Attention (MLA) used in DeepSeek models
      qk_head_dim_sum = config.qk_nope_head_dim + config.qk_rope_head_dim
      if config.q_lora_rank == 0:
        q_params = config.emb_dim * config.num_query_heads * qk_head_dim_sum
      else:
        q_params = config.emb_dim * config.q_lora_rank + config.q_lora_rank * config.num_query_heads * qk_head_dim_sum

      kv_params = config.emb_dim * (
          config.kv_lora_rank + config.qk_rope_head_dim
      ) + config.kv_lora_rank * config.num_query_heads * (config.qk_nope_head_dim + config.v_head_dim)
      proj_params = config.emb_dim * config.num_query_heads * config.v_head_dim
      total_qkv_proj_params = (q_params + kv_params + proj_params) * config.num_decoder_layers
    elif getattr(config, "decoder_block", None) == maxtext_utils.DecoderBlockType.QWEN3_NEXT:
      # Interleaved Full Attention and Gated Delta Net (Linear Attention)
      cycle_interval = config.inhomogeneous_layer_cycle_interval
      num_full_attn_layers = config.num_decoder_layers // cycle_interval
      num_linear_attn_layers = config.num_decoder_layers - num_full_attn_layers

      local_kv_multiplier = 1 if getattr(config, "share_kv_projections", False) else 2
      qkv_params = config.emb_dim * (config.num_query_heads + local_kv_multiplier * config.num_kv_heads) * config.head_dim
      proj_params = config.emb_dim * config.num_query_heads * config.head_dim

      H_k = config.gdn_num_key_heads
      H_v = config.gdn_num_value_heads
      D_k = config.gdn_key_head_dim
      D_v = config.gdn_value_head_dim
      K_conv = config.gdn_conv_kernel_dim
      K_dim = H_k * D_k
      V_dim = H_v * D_v

      # in_proj_qkvz + in_proj_ba + out_proj
      gdn_proj_params = config.emb_dim * (2 * K_dim + 2 * V_dim + 2 * H_v) + config.emb_dim * V_dim
      gdn_conv_params = K_conv * (2 * K_dim + V_dim)

      total_qkv_proj_params = (qkv_params + proj_params) * num_full_attn_layers + (
          gdn_proj_params + gdn_conv_params
      ) * num_linear_attn_layers
    else:
      # Standard Attention (MHA / GQA / MQA) with local window variations
      global_head_dim = getattr(config, "global_head_dim", config.head_dim) or config.head_dim
      global_num_kv_heads = getattr(config, "global_num_kv_heads", config.num_kv_heads) or config.num_kv_heads

      # Local window layer parameters
      # Local layers NEVER share KV projections in Gemma 4
      local_kv_multiplier = 2
      local_qkv_params = (
          config.emb_dim
          * (config.num_query_heads + local_kv_multiplier * config.num_kv_heads)
          * config.head_dim
          * num_local_layers
      )
      local_proj_params = config.emb_dim * config.num_query_heads * config.head_dim * num_local_layers

      # Global full attention layer parameters
      global_kv_multiplier = 1 if getattr(config, "share_kv_projections", False) else 2
      global_qkv_params = (
          config.emb_dim
          * (config.num_query_heads + global_kv_multiplier * global_num_kv_heads)
          * global_head_dim
          * num_global_layers
      )
      global_proj_params = config.emb_dim * config.num_query_heads * global_head_dim * num_global_layers

      total_qkv_proj_params = local_qkv_params + local_proj_params + global_qkv_params + global_proj_params

    active_params = total_ffn_params + total_qkv_proj_params + embedding_params

    expected_flops_from_params = 6 * active_params * config.per_device_batch_size * config.max_target_length / 10**12
    # If not sharing weights, active_params counts both embedding and unembedding matrices.
    # However, embedding lookup is a gather operation and does not use dense math (FLOPs).
    # We must subtract its FLOP equivalent from the expected result so it matches the physical math.
    if getattr(config, "logits_via_embedding", False) is False:
      expected_flops_from_params -= (
          6 * (config.vocab_size * config.emb_dim) * config.per_device_batch_size * config.max_target_length / 10**12
      )

    expected_total_flops = expected_flops_from_params + attention_tflops

    print(
        f"\nActive params for {model_name} (seq_len={max_target_length}): {active_params}, "
        f"Expected TFLOPs: {expected_total_flops} (Computed TFLOPs: {tflops})"
    )

    # 5% margin for approximations and any edge cases
    self.assertAlmostEqual(tflops, expected_total_flops, delta=max(expected_total_flops * 0.05, 0.001))

  def _verify_short_sequence_flops(self, model_name):
    """Verifies short sequence flops."""
    self._verify_flops(model_name, max_target_length=1)

  def _verify_long_sequence_flops(self, model_name):
    """Verifies long sequence flops."""
    self._verify_flops(model_name, max_target_length=8192)

  @parameterized.parameters(
      ("llama3-8b",),
      ("llama4-17b-16e",),
      ("gemma3-4b",),
      ("gemma3-12b",),
      ("gemma3-27b",),
      ("gemma4-26b",),
      ("gemma4-31b",),
      ("gpt-oss-20b",),
      ("gpt-oss-120b",),
      ("qwen3-8b",),
      ("qwen3-next-80b-a3b",),
      ("deepseek3-671b",),
  )
  def test_short_sequence_flops(self, model_name):
    """
    Validates that the computed TFLOPs match the `6 * active_params * tokens` estimation
    for various standard models when attention FLOPs are isolated (e.g. sequence length = 1)
    """
    self._verify_short_sequence_flops(model_name)

  @parameterized.parameters(
      ("llama3-8b",),
      ("llama4-17b-16e",),
      ("gemma3-4b",),
      ("gemma3-12b",),
      ("gemma3-27b",),
      ("gemma4-26b",),
      ("gemma4-31b",),
      ("gpt-oss-20b",),
      ("gpt-oss-120b",),
      ("qwen3-8b",),
      ("qwen3-next-80b-a3b",),
      ("deepseek3-671b",),
  )
  def test_long_sequence_flops(self, model_name):
    """
    Validates that the computed TFLOPs match the `6 * active_params * tokens` estimation
    plus expected attention flops for various standard models with a long sequence length.
    """
    self._verify_long_sequence_flops(model_name)


if __name__ == "__main__":
  unittest.main()
