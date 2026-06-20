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

"""Tests for Qwen2Moe layers comparing MaxText implementations against PyTorch references."""

print("Test imports started...")
import sys
from unittest.mock import MagicMock
sys.modules['triton'] = MagicMock()
sys.modules['triton.knobs'] = MagicMock()
sys.modules['triton.runtime'] = MagicMock()
sys.modules['triton.runtime.autotuner'] = MagicMock()
sys.modules['triton.backends'] = MagicMock()
sys.modules['triton.backends.compiler'] = MagicMock()
sys.modules['triton.compiler'] = MagicMock()
sys.modules['triton.compiler.compiler'] = MagicMock()

import os
import unittest
import numpy as np
import torch
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from maxtext.models.qwen2 import Qwen2MoeSparseMoeBlock, Qwen2MoeDecoderLayer

from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeSparseMoeBlock as TorchQwen2MoeSparseMoeBlock,
    Qwen2MoeDecoderLayer as TorchQwen2MoeDecoderLayer,
    Qwen2MoeRotaryEmbedding,
)

from tests.utils.multimodal_test_utils import (
    assert_all_close_jax_torch,
    copy_linear_weights,
    copy_rmsnorm_weights,
    create_random_jax_torch,
)

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
torch.set_grad_enabled(False)

# Define mini dimensions for fast CPU testing
vocab_size = 1000
hidden_size = 128
intermediate_size = 256
shared_expert_intermediate_size = 256
moe_intermediate_size = 64
num_hidden_layers = 1
num_attention_heads = 8
num_key_value_heads = 8
num_experts = 8
num_experts_per_tok = 2

# Initialize JAX config using Qwen1.5-MoE-A2.7B as base and overriding dimensions
base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
jax_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="qwen1.5-moe-a2.7b",
    base_emb_dim=hidden_size,
    head_dim=hidden_size // num_attention_heads,
    base_num_query_heads=num_attention_heads,
    base_num_kv_heads=num_key_value_heads,
    base_mlp_dim=shared_expert_intermediate_size,
    base_moe_mlp_dim=moe_intermediate_size,
    num_experts=num_experts,
    num_experts_per_tok=num_experts_per_tok,
    base_num_decoder_layers=num_hidden_layers,
    attention="dot_product",
    attention_type="global",
    matmul_precision="highest",
    dropout_rate=0.0,
    dtype="float32",
    weight_dtype="float32",
    float32_logits=True,
    float32_qk_product=True,
    megablox=False,
    override_model_config=True,
)

# PyTorch Qwen2Moe Config matching our mini dimensions
torch_config = Qwen2MoeConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    head_dim=hidden_size // num_attention_heads,
    intermediate_size=shared_expert_intermediate_size,
    shared_expert_intermediate_size=shared_expert_intermediate_size,
    moe_intermediate_size=moe_intermediate_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
    num_experts=num_experts,
    num_experts_per_tok=num_experts_per_tok,
    norm_topk_prob=False,
    rms_norm_eps=1e-06,
    rope_theta=1000000.0,
    hidden_act="silu",
    attention_bias=True,
    qkv_bias=True,
)


def init_torch_weights(module, std=0.02):
  for m in module.modules():
    if isinstance(m, torch.nn.Linear):
      torch.nn.init.normal_(m.weight, std=std)
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)
    elif type(m).__name__ == "Qwen2MoeRMSNorm":
      if hasattr(m, 'weight') and m.weight is not None:
         torch.nn.init.ones_(m.weight)
    elif type(m).__name__ == "Qwen2MoeExperts":
      torch.nn.init.normal_(m.gate_up_proj, std=std)
      torch.nn.init.normal_(m.down_proj, std=std)
    elif type(m).__name__ == "Qwen2MoeTopKRouter":
      torch.nn.init.normal_(m.weight, std=std)


def copy_qwen2_attention_weights(torch_attn, maxtext_attn):
  """Copy attention weights from PyTorch to MaxText Attention module."""
  num_heads = maxtext_attn.num_query_heads
  head_dim = maxtext_attn.head_dim
  hidden_size = num_heads * head_dim
  output_dim = hidden_size

  q_weight = torch_attn.q_proj.weight.detach().cpu().numpy()
  k_weight = torch_attn.k_proj.weight.detach().cpu().numpy()
  v_weight = torch_attn.v_proj.weight.detach().cpu().numpy()
  q_bias = torch_attn.q_proj.bias.detach().cpu().numpy()
  k_bias = torch_attn.k_proj.bias.detach().cpu().numpy()
  v_bias = torch_attn.v_proj.bias.detach().cpu().numpy()

  maxtext_attn.query.kernel.value = jnp.array(q_weight.T.reshape(hidden_size, num_heads, head_dim))
  maxtext_attn.query.bias.value = jnp.array(q_bias.reshape(num_heads, head_dim))

  maxtext_attn.key.kernel.value = jnp.array(k_weight.T.reshape(hidden_size, num_heads, head_dim))
  maxtext_attn.key.bias.value = jnp.array(k_bias.reshape(num_heads, head_dim))

  maxtext_attn.value.kernel.value = jnp.array(v_weight.T.reshape(hidden_size, num_heads, head_dim))
  maxtext_attn.value.bias.value = jnp.array(v_bias.reshape(num_heads, head_dim))

  out_weight = torch_attn.o_proj.weight.detach().cpu().numpy()
  maxtext_attn.out.kernel.value = jnp.array(out_weight.T.reshape(num_heads, head_dim, output_dim))


def copy_qwen2_moe_block_weights(torch_moe, jax_moe):
  """Copy weights from Torch Qwen2MoeSparseMoeBlock to JAX Qwen2MoeSparseMoeBlock."""
  # 1. Router (gate) weights: PyTorch (num_experts, hidden_size) -> JAX (hidden_size, num_experts)
  pt_gate_weight = torch_moe.gate.weight.detach().cpu().numpy()
  jax_moe.routed_experts.gate.kernel.value = jnp.array(pt_gate_weight.T)

  # 2. Routed experts weights:
  # PyTorch gate_up_proj: (num_experts, 2 * moe_intermediate_size, hidden_size)
  # PyTorch down_proj: (num_experts, hidden_size, moe_intermediate_size)
  # JAX wi_0, wi_1: (num_experts, hidden_size, moe_intermediate_size)
  # JAX wo: (num_experts, moe_intermediate_size, hidden_size)
  pt_gate_up_proj = torch_moe.experts.gate_up_proj.detach().cpu().numpy()
  pt_down_proj = torch_moe.experts.down_proj.detach().cpu().numpy()

  pt_gate_proj, pt_up_proj = np.split(pt_gate_up_proj, 2, axis=1)

  # Transpose and copy
  jax_moe.routed_experts.wi_0.value = jnp.array(np.transpose(pt_gate_proj, (0, 2, 1)))
  jax_moe.routed_experts.wi_1.value = jnp.array(np.transpose(pt_up_proj, (0, 2, 1)))
  jax_moe.routed_experts.wo.value = jnp.array(np.transpose(pt_down_proj, (0, 2, 1)))

  # 3. Shared expert weights:
  # PyTorch gate_proj, up_proj: Linear(hidden_size, shared_expert_intermediate_size)
  # PyTorch down_proj: Linear(shared_expert_intermediate_size, hidden_size)
  # JAX MlpBlock wi_0, wi_1: (hidden_size, shared_expert_intermediate_size)
  # JAX MlpBlock wo: (shared_expert_intermediate_size, hidden_size)
  copy_linear_weights(torch_moe.shared_expert.gate_proj, jax_moe.shared_expert.wi_0)
  copy_linear_weights(torch_moe.shared_expert.up_proj, jax_moe.shared_expert.wi_1)
  copy_linear_weights(torch_moe.shared_expert.down_proj, jax_moe.shared_expert.wo)

  # 4. Shared expert gate:
  # PyTorch shared_expert_gate: Linear(hidden_size, 1)
  # JAX shared_expert_gate: DenseGeneral(hidden_size, 1)
  copy_linear_weights(torch_moe.shared_expert_gate, jax_moe.shared_expert_gate)


class TestQwen2MoeLayers(unittest.TestCase):
  """Equivalence tests comparing JAX Qwen2Moe layers with PyTorch references."""

  def setUp(self):
    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

  def test_sparse_moe_block_equivalence(self):
    """Test Qwen2MoeSparseMoeBlock parity."""
    torch_moe = TorchQwen2MoeSparseMoeBlock(torch_config)
    init_torch_weights(torch_moe)
    torch_moe.eval()

    jax_moe = Qwen2MoeSparseMoeBlock(
        config=jax_config,
        mesh=self.mesh,
        quant=None,
        rngs=nnx.Rngs(42),
    )

    copy_qwen2_moe_block_weights(torch_moe, jax_moe)

    # Input: (batch=2, seq=4, hidden=128)
    batch_size, seq_len = 2, 4
    flat_data, _ = create_random_jax_torch(batch_size * seq_len * hidden_size)
    jax_input = flat_data.reshape(batch_size, seq_len, hidden_size)
    torch_input = torch.from_numpy(np.array(jax_input))

    torch_output = torch_moe(torch_input)

    # JAX forward
    jax_output, _ = jax_moe(jax_input, deterministic=True)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-5,
        atol=1e-5,
        error_msg="Qwen2MoeSparseMoeBlock output mismatch",
    )

  def test_decoder_layer_equivalence(self):
    """Test Qwen2MoeDecoderLayer parity."""
    # We need a layer index for TorchQwen2MoeDecoderLayer
    torch_layer = TorchQwen2MoeDecoderLayer(torch_config, layer_idx=0)
    init_torch_weights(torch_layer)
    torch_layer.eval()

    jax_layer = Qwen2MoeDecoderLayer(
        config=jax_config,
        mesh=self.mesh,
        model_mode="prefill",
        quant=None,
        rngs=nnx.Rngs(42),
    )

    # Copy weights
    copy_rmsnorm_weights(torch_layer.input_layernorm, jax_layer.pre_self_attention_layer_norm)
    copy_rmsnorm_weights(torch_layer.post_attention_layernorm, jax_layer.post_self_attention_layer_norm)
    copy_qwen2_attention_weights(torch_layer.self_attn, jax_layer.self_attention)
    copy_qwen2_moe_block_weights(torch_layer.mlp, jax_layer.moe_block)

    # Input: (batch=2, seq=4, hidden=128)
    batch_size, seq_len = 2, 4
    flat_data, _ = create_random_jax_torch(batch_size * seq_len * hidden_size)
    jax_input = flat_data.reshape(batch_size, seq_len, hidden_size)
    torch_input = torch.from_numpy(np.array(jax_input))

    # Position embeddings (for RoPE)
    # PyTorch expects position_embeddings: tuple[cos, sin] of shape (1, seq_len, head_dim)
    # MaxText handles RoPE internally.
    head_dim = hidden_size // num_attention_heads
    # Generate mock positional embeddings for PyTorch
    position_ids = torch.arange(seq_len).unsqueeze(0)  # (1, seq)
    rotary_emb = Qwen2MoeRotaryEmbedding(torch_config)
    cos, sin = rotary_emb(torch_input, position_ids=position_ids)
    # In newer transformers versions, rotary_emb takes (x, position_ids) and returns (cos, sin)

    # Construct causal mask for Torch
    mask_torch = torch.tril(torch.ones(seq_len, seq_len)).to(torch_input.device)
    torch_attn_mask = torch.zeros(seq_len, seq_len).to(torch_input.device)
    torch_attn_mask = torch_attn_mask.masked_fill(mask_torch == 0, float("-inf"))
    torch_attn_mask = torch_attn_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    torch_attn_mask = torch_attn_mask.to(torch_input.dtype)

    # PyTorch forward
    torch_output = torch_layer(
        torch_input,
        position_embeddings=(cos, sin),
        attention_mask=torch_attn_mask,
    )

    # JAX forward
    # Generate positions for JAX
    positions = jnp.tile(jnp.arange(seq_len), (batch_size, 1))
    jax_output, _ = jax_layer(
        jax_input,
        decoder_segment_ids=None,
        decoder_positions=positions,
        deterministic=True,
        model_mode="prefill",
    )

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-5,
        atol=1e-5,
        error_msg="Qwen2MoeDecoderLayer output mismatch",
    )


if __name__ == "__main__":
  unittest.main()
