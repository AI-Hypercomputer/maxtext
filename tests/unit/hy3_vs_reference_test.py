# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests validating Hy3 (Tencent Hunyuan V3) MaxText components against PyTorch references."""

import os
from types import SimpleNamespace
from typing import Optional, Tuple
import unittest

from flax import nnx
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maxtext.common.common_types import (
    Config,
    DecoderBlockType,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.layers import attentions, linears, moe, normalizations
from maxtext.models import hy3
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT


# ==============================================================================
# PyTorch Reference Implementations for Hy3
# ==============================================================================


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  """Converts PyTorch tensor to JAX array."""
  return jnp.asarray(pt_tensor.detach().cpu().numpy())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotates half the hidden dimensions of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Applies Rotary Position Embedding (RoPE) to query and key tensors."""
  cos = cos.unsqueeze(1)
  sin = sin.unsqueeze(1)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


class Hy3RMSNorm_PT(nn.Module):
  """PyTorch reference RMSNorm."""

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + self.eps)
    return self.weight * hidden_states


class Hy3RotaryEmbedding_PT(nn.Module):
  """PyTorch reference Rotary Embedding."""

  def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
    super().__init__()
    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # position_ids: (batch, seq_len)
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Hy3Attention_PT(nn.Module):
  """PyTorch reference implementation for Hy3 Attention (GQA + QK-Norm)."""

  def __init__(
      self,
      hidden_size: int,
      num_attention_heads: int,
      num_key_value_heads: int,
      head_dim: int,
      rms_norm_eps: float = 1e-6,
      rope_theta: float = 10000.0,
      max_position_embeddings: int = 4096,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_attention_heads
    self.num_kv_heads = num_key_value_heads
    self.head_dim = head_dim
    self.num_key_value_groups = self.num_heads // self.num_kv_heads
    self.scaling = head_dim**-0.5

    self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    self.q_norm = Hy3RMSNorm_PT(self.head_dim, eps=rms_norm_eps)
    self.k_norm = Hy3RMSNorm_PT(self.head_dim, eps=rms_norm_eps)
    self.rotary_emb = Hy3RotaryEmbedding_PT(self.head_dim, max_position_embeddings, rope_theta)

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_ids: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.shape

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    # QK-Norm per head
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    # Apply RoPE
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Repeat KV for GQA
    if self.num_key_value_groups > 1:
      key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
      value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
      attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
    return self.o_proj(attn_output)


class Hy3MLP_PT(nn.Module):
  """PyTorch reference implementation for Hy3 Dense SwiGLU MLP."""

  def __init__(self, hidden_size: int, intermediate_size: int):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Hy3MoE_PT(nn.Module):
  """PyTorch reference implementation for Hy3 MoE with shared expert and sigmoid+bias routing."""

  def __init__(
      self,
      hidden_size: int,
      moe_intermediate_size: int,
      shared_intermediate_size: int,
      num_experts: int,
      top_k: int,
      routed_scaling_factor: float = 1.0,
  ):
    super().__init__()
    self.num_experts = num_experts
    self.top_k = top_k
    self.routed_scaling_factor = routed_scaling_factor

    # Router: gate with bias
    self.gate = nn.Linear(hidden_size, num_experts, bias=True)

    # Routed Experts
    self.experts = nn.ModuleList([Hy3MLP_PT(hidden_size, moe_intermediate_size) for _ in range(num_experts)])

    # Shared Expert
    self.shared_expert = Hy3MLP_PT(hidden_size, shared_intermediate_size)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # hidden_states: (bsz, seq_len, hidden_size)
    orig_shape = hidden_states.shape
    x_flat = hidden_states.view(-1, orig_shape[-1])

    # Shared expert output
    shared_out = self.shared_expert(x_flat)

    # Routing logits with bias
    router_logits = self.gate(x_flat)
    scores = torch.sigmoid(router_logits)

    # Top-K selection
    topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)

    # Renormalize weights
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * self.routed_scaling_factor

    # Expert computation
    routed_out = torch.zeros_like(x_flat)
    for i in range(self.top_k):
      idx = topk_indices[:, i]
      weight = topk_weights[:, i : i + 1]
      for exp_id in range(self.num_experts):
        mask = idx == exp_id
        if mask.any():
          exp_input = x_flat[mask]
          exp_out = self.experts[exp_id](exp_input)
          routed_out[mask] += exp_out * weight[mask]

    total_out = (routed_out + shared_out).view(orig_shape)
    return total_out


class Hy3DenseLayer_PT(nn.Module):
  """PyTorch reference implementation for Hy3 Dense Decoder Layer."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rms_norm_eps: float = 1e-6,
      rope_theta: float = 10000.0,
  ):
    super().__init__()
    self.input_layernorm = Hy3RMSNorm_PT(hidden_size, eps=rms_norm_eps)
    self.self_attn = Hy3Attention_PT(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps, rope_theta=rope_theta)
    self.post_attention_layernorm = Hy3RMSNorm_PT(hidden_size, eps=rms_norm_eps)
    self.mlp = Hy3MLP_PT(hidden_size, intermediate_size)

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_ids: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    # Self Attention with pre-norm & residual
    normed_attn_in = self.input_layernorm(hidden_states)
    attn_out = self.self_attn(normed_attn_in, position_ids, attention_mask)
    hidden_states = hidden_states + attn_out

    # MLP with post-attention norm & residual
    normed_mlp_in = self.post_attention_layernorm(hidden_states)
    mlp_out = self.mlp(normed_mlp_in)
    hidden_states = hidden_states + mlp_out

    return hidden_states


class Hy3MoELayer_PT(nn.Module):
  """PyTorch reference implementation for Hy3 MoE Decoder Layer."""

  def __init__(
      self,
      hidden_size: int,
      moe_intermediate_size: int,
      shared_intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      num_experts: int,
      top_k: int,
      rms_norm_eps: float = 1e-6,
      rope_theta: float = 10000.0,
  ):
    super().__init__()
    self.input_layernorm = Hy3RMSNorm_PT(hidden_size, eps=rms_norm_eps)
    self.self_attn = Hy3Attention_PT(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps, rope_theta=rope_theta)
    self.post_attention_layernorm = Hy3RMSNorm_PT(hidden_size, eps=rms_norm_eps)
    self.moe_block = Hy3MoE_PT(
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_ids: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    # Self Attention with pre-norm & residual
    normed_attn_in = self.input_layernorm(hidden_states)
    attn_out = self.self_attn(normed_attn_in, position_ids, attention_mask)
    hidden_states = hidden_states + attn_out

    # MoE block with post-attention norm & residual
    normed_moe_in = self.post_attention_layernorm(hidden_states)
    moe_out = self.moe_block(normed_moe_in)
    hidden_states = hidden_states + moe_out

    return hidden_states


# ==============================================================================
# Test Suite
# ==============================================================================


class Hy3VsReferenceTest(unittest.TestCase):
  """Unit tests comparing MaxText Hy3 components against PyTorch reference."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    self.batch_size = 2
    self.seq_len = 8
    self.hidden_size = 128
    self.head_dim = 32
    self.num_heads = 4
    self.num_kv_heads = 2
    self.mlp_dim = 256
    self.moe_mlp_dim = 128
    self.num_experts = 4
    self.num_experts_per_tok = 2
    self.rms_norm_eps = 1e-6

    # Base test configuration for MaxText (must be initialized first before JAX operations)
    base_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "maxtext", "configs", "base.yml")
    )
    self.jax_config = pyconfig.initialize(
        ["", base_config_path],
        model_name="hy3-tiny",
        override_model_config=True,
        base_emb_dim=self.hidden_size,
        head_dim=self.head_dim,
        base_num_query_heads=self.num_heads,
        base_num_kv_heads=self.num_kv_heads,
        base_mlp_dim=self.mlp_dim,
        base_moe_mlp_dim=self.moe_mlp_dim,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        shared_experts=1,
        routed_scaling_factor=1.0,
        routed_score_func="sigmoid",
        routed_bias=True,
        use_qk_norm=True,
        attention="dot_product",
        matmul_precision="highest",
        max_target_length=self.seq_len,
        dtype="float32",
        weight_dtype="float32",
        float32_logits=True,
        float32_qk_product=True,
        dropout_rate=0.0,
    )

    # JAX mesh
    devices_array = maxtext_utils.create_device_mesh(self.jax_config)
    self.mesh = Mesh(devices_array, self.jax_config.mesh_axes)

    # Initial inputs
    self.pt_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
    self.pt_positions = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1)
    self.jax_input = to_jax(self.pt_input)
    self.jax_positions = to_jax(self.pt_positions)
    self.jax_segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

  def test_hy3_dense_mlp_vs_reference(self):
    """Validates Hy3 Dense MLP (SwiGLU) against PyTorch reference."""
    pt_mlp = Hy3MLP_PT(self.hidden_size, self.mlp_dim)
    pt_out = pt_mlp(self.pt_input)

    # MaxText MlpBlock
    rngs = nnx.Rngs(0)
    jax_mlp = linears.MlpBlock(
        in_features=self.hidden_size,
        intermediate_dim=self.mlp_dim,
        activations=["silu", "linear"],
        intermediate_dropout_rate=0.0,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        config=self.jax_config,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=rngs,
    )

    # Copy weights
    # MaxText wi_0 is gate, wi_1 is up, wo is down
    jax_mlp.wi_0.kernel.value = to_jax(pt_mlp.gate_proj.weight.t())
    jax_mlp.wi_1.kernel.value = to_jax(pt_mlp.up_proj.weight.t())
    jax_mlp.wo.kernel.value = to_jax(pt_mlp.down_proj.weight.t())

    jax_out = jax_mlp(self.jax_input, deterministic=True)

    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-3, atol=1e-3)

  def test_hy3_dense_layer_vs_reference(self):
    """Validates full Hy3DenseLayer forward pass against PyTorch reference."""
    pt_layer = Hy3DenseLayer_PT(
        hidden_size=self.hidden_size,
        intermediate_size=self.mlp_dim,
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        rms_norm_eps=self.rms_norm_eps,
        rope_theta=float(self.jax_config.rope_max_timescale),
    )

    # Create causal mask for PyTorch
    mask = torch.triu(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.float32).masked_fill(mask, -1e9)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    pt_out = pt_layer(self.pt_input, self.pt_positions, attention_mask=causal_mask)

    # MaxText Hy3DenseLayer
    rngs = nnx.Rngs(0)
    jax_layer = hy3.Hy3DenseLayer(
        config=self.jax_config,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=rngs,
        layer_idx=0,
    )

    # Copy weights: Norms
    jax_layer.pre_self_attention_layer_norm.scale.value = to_jax(pt_layer.input_layernorm.weight)
    jax_layer.post_self_attention_layer_norm.scale.value = to_jax(pt_layer.post_attention_layernorm.weight)

    # Copy weights: Attention QKV & Out
    jax_layer.self_attention.query.kernel.value = to_jax(
        pt_layer.self_attn.q_proj.weight.t().view(self.hidden_size, self.num_heads, self.head_dim)
    )
    jax_layer.self_attention.key.kernel.value = to_jax(
        pt_layer.self_attn.k_proj.weight.t().view(self.hidden_size, self.num_kv_heads, self.head_dim)
    )
    jax_layer.self_attention.value.kernel.value = to_jax(
        pt_layer.self_attn.v_proj.weight.t().view(self.hidden_size, self.num_kv_heads, self.head_dim)
    )
    jax_layer.self_attention.out.kernel.value = to_jax(
        pt_layer.self_attn.o_proj.weight.t().view(self.num_heads, self.head_dim, self.hidden_size)
    )
    jax_layer.self_attention.query_norm.scale.value = to_jax(pt_layer.self_attn.q_norm.weight)
    jax_layer.self_attention.key_norm.scale.value = to_jax(pt_layer.self_attn.k_norm.weight)

    # Copy weights: MLP
    jax_layer.mlp.wi_0.kernel.value = to_jax(pt_layer.mlp.gate_proj.weight.t())
    jax_layer.mlp.wi_1.kernel.value = to_jax(pt_layer.mlp.up_proj.weight.t())
    jax_layer.mlp.wo.kernel.value = to_jax(pt_layer.mlp.down_proj.weight.t())

    jax_out = jax_layer(
        inputs=self.jax_input,
        decoder_segment_ids=self.jax_segment_ids,
        decoder_positions=self.jax_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    if isinstance(jax_out, tuple):
      jax_out = jax_out[0]

    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  unittest.main()
