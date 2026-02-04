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

import unittest
import pytest

from MaxText import pyconfig
from maxtext.utils.maxtext_utils import calculate_tflops_training_per_device
from tests.utils.test_helpers import get_test_config_path


class FlopCalculation(unittest.TestCase):
  """Tests for verifying FLOP calculation in MaxText"""

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

  @pytest.mark.cpu_only
  def test_qwen3_next_flops(self):
    """Test Qwen3-Next Flops calculation"""
    kwargs = {
        "model_name": "qwen3-next-80b-a3b",
        "override_model_config": True,
        "per_device_batch_size": 1,
        "max_target_length": 4096,
        "decoder_block": "qwen3_next",
        "gradient_accumulation_steps": 1,
        "skip_jax_distributed_system": True,
        # Core Architectural Parameters
        "base_emb_dim": 2048,
        "base_num_decoder_layers": 48,
        "base_num_query_heads": 16,
        "base_num_kv_heads": 2,
        "head_dim": 256,
        "vocab_size": 151936,
        # MoE Parameters
        "base_mlp_dim": 512,  # Note: maxtext_utils uses moe_mlp_dim for calculations
        "base_moe_mlp_dim": 512,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "mlp_activations": ["silu", "linear"],
        # Qwen3-Next Specific Parameters
        "inhomogeneous_layer_cycle_interval": 4,
        "gdn_conv_kernel_dim": 4,
        "gdn_key_head_dim": 128,
        "gdn_value_head_dim": 128,
        "gdn_num_key_heads": 16,
        "gdn_num_value_heads": 32,
        "gdn_chunk_size": 64,
    }

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
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        **kwargs,
    )
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)

    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  @pytest.mark.cpu_only
  def test_llama2_7b_flops(self):
    """Test Llama2 7b Flops calculation with default parameters"""
    kwargs = {
        # Model bases
        "model_name": "llama2-7b",
        "override_model_config": True,
        # Core workload parameters
        "per_device_batch_size": 12,
        "max_target_length": 2048,
        # Model dimensions
        "base_emb_dim": 4096,
        "base_mlp_dim": 11008,
        "base_num_query_heads": 32,
        "base_num_kv_heads": 32,
        "base_num_decoder_layers": 32,
        "head_dim": 128,
        "vocab_size": 32_000,
        "mlp_activations": ["silu", "linear"],
        "skip_jax_distributed_system": True,
    }
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    attention_flops = self.compute_regular_attention_flops_per_device(kwargs)
    # Llama2-7b has ~6.74B parameters
    # https://adithyask.medium.com/from-7b-to-8b-parameters-understanding-weight-matrix-changes-in-llama-transformer-models-31ea7ed5fd88
    golden_param_size = 6.74e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        **kwargs,
    )
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  @pytest.mark.cpu_only
  def test_llama3_8b_flops(self):
    """Test Llama3 8b Flops calculation with default parameters"""
    kwargs = {
        # Model bases
        "model_name": "llama3-8b",
        "override_model_config": True,
        # Core workload parameters
        "per_device_batch_size": 4,
        "max_target_length": 2048,
        "gradient_accumulation_steps": 1,
        # Model dimensions
        "base_emb_dim": 4096,
        "base_mlp_dim": 14336,
        "base_num_query_heads": 32,
        "base_num_kv_heads": 8,
        "base_num_decoder_layers": 32,
        "head_dim": 128,
        "vocab_size": 128256,
        "mlp_activations": ["silu", "linear"],
        "skip_jax_distributed_system": True,
    }
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    attention_flops = self.compute_regular_attention_flops_per_device(kwargs)
    # LLaMA3-8b has ~8.03B parameters
    # https://adithyask.medium.com/from-7b-to-8b-parameters-understanding-weight-matrix-changes-in-llama-transformer-models-31ea7ed5fd88
    # Note: The commonly cited 8.03B parameter count for Llama 3 8B corresponds to a version with UNTIED embeddings.
    # Here we consider TIED embedding table, which reduces param count to 7.50B
    golden_param_size = 7.50e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        **kwargs,
    )
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  @pytest.mark.cpu_only
  def test_mixtral_8x7b_flops(self):
    """Test Mixtral 8x7b Flops calculation"""
    kwargs = {
        # Model bases
        "model_name": "mixtral-8x7b",
        "override_model_config": True,
        # Core workload parameters
        "per_device_batch_size": 4,
        "max_target_length": 8192,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "gradient_accumulation_steps": 1,
        # model dimensions
        "base_emb_dim": 4096,
        "base_mlp_dim": 14336,
        "base_num_query_heads": 32,
        "base_num_kv_heads": 8,
        "head_dim": 128,
        "base_num_decoder_layers": 32,
        "vocab_size": 32000,
        "mlp_activations": ["silu", "linear"],
        "skip_jax_distributed_system": True,
    }
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    attention_flops = self.compute_regular_attention_flops_per_device(kwargs)
    # mixtral-8x7b has ~12.9B active parameters
    # https://mistral.ai/news/mixtral-of-experts
    golden_param_size = 12.9e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        **kwargs,
    )
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  @pytest.mark.cpu_only
  def test_deepseek2_16b_flops(self):
    """Test DeepSeek2-16b FLops calculation"""
    kwargs = {
        # Model bases
        "model_name": "deepseek2-16b",
        "override_model_config": True,
        # Core workload parameters
        "per_device_batch_size": 4,
        "max_target_length": 8192,
        "num_experts": 64,
        "num_experts_per_tok": 6,
        "shared_experts": 2,
        # Model dimensions
        "base_emb_dim": 2048,
        "base_num_query_heads": 16,
        "base_num_kv_heads": 16,
        "base_mlp_dim": 10944,
        "base_moe_mlp_dim": 1408,
        "base_num_decoder_layers": 27,
        "first_num_dense_layers": 1,
        "mlp_activations": ["silu", "linear"],
        "vocab_size": 102400,
        # MLA
        "q_lora_rank": 0,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "skip_jax_distributed_system": True,
    }
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    attention_flops = self.compute_deepseek_attention_flops_per_device(kwargs)
    # deepseek2-16b has ~2.4B active parameters
    # https://arxiv.org/pdf/2405.04434
    golden_param_size = 2.4e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        **kwargs,
    )
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  @pytest.mark.cpu_only
  def test_gpt_oss_20b_flops(self):
    """Test GPT OSS 20B Flops calculation"""
    kwargs = {
        # Model bases
        "model_name": "gpt-oss-20b",
        "override_model_config": True,
        # Core workload parameters
        "per_device_batch_size": 4,
        "max_target_length": 8192,
        "sliding_window_size": 128,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "gradient_accumulation_steps": 1,
        # model dimensions
        "base_emb_dim": 2880,
        "base_mlp_dim": 2880,
        "base_num_query_heads": 64,
        "base_num_kv_heads": 8,
        "head_dim": 64,
        "base_num_decoder_layers": 24,
        "vocab_size": 201088,
        "mlp_activations": ["silu", "linear"],
        "skip_jax_distributed_system": True,
    }
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    attention_flops = self.compute_gpt_attention_flops_per_device(kwargs)
    # gpt-oss-20b has ~3.6B active parameters
    # https://openai.com/index/introducing-gpt-oss/
    golden_param_size = 3.6e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        **kwargs,
    )
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)

  @pytest.mark.cpu_only
  def test_deepseek32_671b_flops(self):
    """Test DeepSeek3.2-671b FLops calculation"""
    kwargs = {
        # Model bases
        "model_name": "deepseek3.2-671b",
        "override_model_config": True,
        # Core workload parameters
        "per_device_batch_size": 4,
        "max_target_length": 4096,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "shared_experts": 1,
        # Model dimensions
        "base_emb_dim": 7168,
        "base_num_query_heads": 128,
        "base_num_kv_heads": 128,
        "base_mlp_dim": 18432,
        "base_moe_mlp_dim": 2048,
        "base_num_decoder_layers": 61,
        "first_num_dense_layers": 3,
        "mlp_activations": ["silu", "linear"],
        "vocab_size": 129280,
        # MLA
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "skip_jax_distributed_system": True,
        # Indexer for DeepSeek Sparse Attention
        "use_sparse_indexer": True,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        # TODO(ranran): remove after flash attention is supported
        "attention": "dot_product",
    }
    B = kwargs["per_device_batch_size"]
    S = kwargs["max_target_length"]
    attention_flops = self.compute_deepseek_attention_flops_per_device(kwargs)
    # deepseek3-671b has ~37B active parameters
    # https://arxiv.org/pdf/2412.19437
    golden_param_size = 37e9
    golden_tflops = 6 * B * S * golden_param_size / 1e12 + attention_flops
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        **kwargs,
    )
    calculated_tflops, _, _ = calculate_tflops_training_per_device(cfg)
    self.assertFlopsAlmostEqual(calculated_tflops, golden_tflops)
