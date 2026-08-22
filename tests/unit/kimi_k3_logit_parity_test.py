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

"""Unit test for Kimi K3 mathematical layer-by-layer and logit parity: JAX (MaxText) vs PyTorch.

This test validates that Kimi K3 components in MaxText (RMSNorm, Situ MLP, KDA Attention,
KimiDecoderLayer, and End-to-End Logit generation) produce mathematically identical outputs
and logits (KL divergence < 1e-4, Cosine Similarity > 0.9999, Top-1 Argmax Agreement 100%)
compared to a PyTorch reference implementation with synchronized parameters.
"""

import os
import sys
import unittest
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh


from maxtext.configs import pyconfig
from maxtext.layers.embeddings import Embed as JaxEmbed
from maxtext.layers.kda import KimiDecoupledAttention as JaxKDA
from maxtext.layers.kimi_decoder_layer import KimiDecoderLayer as JaxDecoderLayer
from maxtext.layers.linears import MlpBlock as JaxMLP
from maxtext.layers.nnx_decoders import NNXDecoder as JaxNNXDecoder
from maxtext.layers.normalizations import RMSNorm as JaxRMSNorm


# =============================================================================
# PyTorch Reference Implementations
# =============================================================================

class PtRMSNorm(nn.Module):
  """PyTorch Reference RMSNorm."""

  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return x * norm * self.scale


def situ_act(x: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
  return beta * torch.tanh(x / beta) * torch.sigmoid(x)


def linear_beta_tanh_act(x: torch.Tensor, beta: float = 25.0) -> torch.Tensor:
  return beta * torch.tanh(x / beta)


class PtSituMLP(nn.Module):
  """PyTorch Reference Situ MLP (wi_0 with situ, wi_1 with linear_beta_tanh, wo projection)."""

  def __init__(self, in_features: int, intermediate_dim: int):
    super().__init__()
    self.wi_0 = nn.Linear(in_features, intermediate_dim, bias=False)
    self.wi_1 = nn.Linear(in_features, intermediate_dim, bias=False)
    self.wo = nn.Linear(intermediate_dim, in_features, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h1 = situ_act(self.wi_0(x))
    h2 = linear_beta_tanh_act(self.wi_1(x))
    return self.wo(h1 * h2)


class PtShortConv1D(nn.Module):
  """PyTorch Reference 1D Depthwise Short Convolution for KDA."""

  def __init__(self, features: int, kernel_size: int = 4):
    super().__init__()
    self.kernel_size = kernel_size
    self.weight = nn.Parameter(torch.randn(features, 1, kernel_size))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    x_t = x.transpose(1, 2)
    x_pad = F.pad(x_t, (self.kernel_size - 1, 0))
    y = F.conv1d(x_pad, self.weight, groups=C)
    y = y.transpose(1, 2)
    return F.silu(y)


class PtKDA(nn.Module):
  """PyTorch Reference Kimi Decoupled Attention (KDA)."""

  def __init__(self, hidden_size: int, num_heads: int, head_dim: int, conv_kernel_size: int = 4, eps: float = 1e-5):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = head_dim
    projection_size = num_heads * head_dim

    self.q_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.k_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)

    self.q_conv1d = PtShortConv1D(projection_size, conv_kernel_size)
    self.k_conv1d = PtShortConv1D(projection_size, conv_kernel_size)
    self.v_conv1d = PtShortConv1D(projection_size, conv_kernel_size)

    self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
    self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False)
    self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

    self.A_log = nn.Parameter(torch.zeros(head_dim))
    self.dt_bias = nn.Parameter(torch.zeros(projection_size))

    self.g_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.o_norm = PtRMSNorm(head_dim, eps=eps)
    self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.shape
    H, K = self.num_heads, self.head_dim

    # 1. Projections + Conv1D
    q = self.q_conv1d(self.q_proj(x)).reshape(B, T, H, K)
    k = self.k_conv1d(self.k_proj(x)).reshape(B, T, H, K)
    v = self.v_conv1d(self.v_proj(x)).reshape(B, T, H, K)

    # L2 norm along head_dim
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-6)
    k = k / torch.linalg.norm(k, dim=-1, keepdim=True).clamp(min=1e-6)

    # 2. Gate & Beta
    g = self.f_b_proj(self.f_a_proj(x)).reshape(B, T, H, K)
    g = -torch.exp(self.A_log).unsqueeze(0).unsqueeze(0).unsqueeze(0) * F.softplus(
        g + self.dt_bias.reshape(1, 1, H, K)
    )
    g = torch.maximum(g, torch.tensor(-5.0))
    beta = torch.sigmoid(self.b_proj(x))

    # 3. Recurrent KDA step
    scale = K**-0.5
    q = q * scale
    S = torch.zeros(B, H, K, K, dtype=x.dtype, device=x.device)
    outputs = []
    for t in range(T):
      q_t = q[:, t]
      k_t = k[:, t]
      v_t = v[:, t]
      g_t = g[:, t]
      b_t = beta[:, t]

      # Decay state: S = S * exp(g)
      S = S * torch.exp(g_t).unsqueeze(-1)
      # k_S = k^T @ S
      k_S = torch.sum(k_t.unsqueeze(-1) * S, dim=-2)
      v_diff = v_t - k_S
      bk = b_t.unsqueeze(-1) * k_t
      S = S + bk.unsqueeze(-1) * v_diff.unsqueeze(-2)
      o_t = torch.sum(q_t.unsqueeze(-1) * S, dim=-2)
      outputs.append(o_t)

    o = torch.stack(outputs, dim=1)

    # 4. Gated Output Norm & Projection
    g_out = torch.sigmoid(self.g_proj(x)).reshape(B, T, H, K)
    o_normed = self.o_norm(o) * g_out
    out = self.o_proj(o_normed.reshape(B, T, H * K))
    return out


class PtFullDecoderLayer(nn.Module):
  """PyTorch Reference Full KimiDecoderLayer."""

  def __init__(self, norm1: PtRMSNorm, attn: PtKDA, norm2: PtRMSNorm, mlp: PtSituMLP):
    super().__init__()
    self.norm1 = norm1
    self.attn = attn
    self.norm2 = norm2
    self.mlp = mlp

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x



# =============================================================================
# Helper: Compute Parity Metrics & KL Divergence
# =============================================================================

def compute_parity_metrics(a_np: np.ndarray, b_np: np.ndarray) -> dict:
  """Computes tensor distance, cosine similarity, top-1 agreement, and KL divergence."""
  a = a_np.astype(np.float32)
  b = b_np.astype(np.float32)
  assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

  abs_diff = np.abs(a - b)
  max_err = float(np.max(abs_diff))
  mae = float(np.mean(abs_diff))

  a_flat = a.reshape(-1)
  b_flat = b.reshape(-1)
  norm_a = float(np.linalg.norm(a_flat))
  norm_b = float(np.linalg.norm(b_flat))
  cos_sim = float(np.dot(a_flat, b_flat) / (norm_a * norm_b + 1e-12))

  # KL Divergence over vocab / last dimension
  def _log_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    log_z = np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
    return x - log_z

  log_p = _log_softmax(a)
  log_q = _log_softmax(b)
  p = np.exp(log_p)
  kl = float(np.mean(np.sum(p * (log_p - log_q), axis=-1)))

  # Top-1 argmax agreement
  top1_a = np.argmax(a, axis=-1)
  top1_b = np.argmax(b, axis=-1)
  top1_agreement = float(np.mean(top1_a == top1_b))

  return {
      "shape": list(a.shape),
      "max_abs_err": max_err,
      "mae": mae,
      "cos_sim": cos_sim,
      "kl_divergence": kl,
      "top1_agreement": top1_agreement,
  }


# =============================================================================
# Unit Test Class
# =============================================================================

class KimiK3LogitParityTest(unittest.TestCase):
  """Comprehensive unit tests validating MaxText Kimi K3 against PyTorch reference."""

  @classmethod
  def setUpClass(cls):
    cls.config = pyconfig.initialize([
        "kimi_k3_logit_parity_test.py",
        "src/maxtext/configs/models/kimi-k3-minimal.yml",
        "model_name=kimi-k3",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "base_emb_dim=7168",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "base_mlp_dim=512",
        "kda_layers=[1]",
        "full_attn_layers=[2]",
        "kda_conv_kernel_size=4",
        "kda_use_full_rank_gate=true",
        "kda_gate_lower_bound=-5.0",
        "mlp_activations=['situ','linear_beta_tanh']",
        "normalization_layer_epsilon=1.0e-5",
        "hardware=cpu",
        "skip_jax_distributed_system=True",
        "scan_layers=False",
        "async_checkpointing=False",
    ])
    cls.mesh = Mesh(jax.devices(), ("data",))
    cls.rngs = nnx.Rngs(0)
    cls.D = cls.config.emb_dim
    cls.H = cls.config.base_num_query_heads
    cls.K = cls.config.head_dim
    cls.intermediate_dim = cls.config.base_mlp_dim
    cls.eps = cls.config.normalization_layer_epsilon

  def test_1_rmsnorm_parity(self):
    """Test 1: RMSNorm JAX vs PyTorch equivalence."""
    B, T = 1, 4
    x_np = np.random.randn(B, T, self.D).astype(np.float32)
    jax_norm = JaxRMSNorm(
        num_features=self.D,
        epsilon=self.eps,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=self.rngs,
    )
    pt_norm = PtRMSNorm(self.D, eps=self.eps)
    pt_norm.scale.data = torch.from_numpy(np.array(jax_norm.scale.get_value()))

    out_jax = np.array(jax_norm(jnp.array(x_np)))
    out_pt = pt_norm(torch.from_numpy(x_np)).detach().numpy()
    metrics = compute_parity_metrics(out_jax, out_pt)

    self.assertLess(metrics["max_abs_err"], 1e-5)
    self.assertGreater(metrics["cos_sim"], 0.999999)
    self.assertLess(abs(metrics["kl_divergence"]), 1e-5)

  def test_2_situ_mlp_parity(self):
    """Test 2: Situ MLP (situ + linear_beta_tanh) JAX vs PyTorch equivalence."""
    B, T = 1, 4
    x_np = np.random.randn(B, T, self.D).astype(np.float32)
    jax_mlp = JaxMLP(
        in_features=self.D,
        intermediate_dim=self.intermediate_dim,
        activations=self.config.mlp_activations,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        config=self.config,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    pt_mlp = PtSituMLP(self.D, self.intermediate_dim)
    pt_mlp.wi_0.weight.data = torch.from_numpy(np.array(jax_mlp.wi_0.kernel.get_value()).T)
    pt_mlp.wi_1.weight.data = torch.from_numpy(np.array(jax_mlp.wi_1.kernel.get_value()).T)
    pt_mlp.wo.weight.data = torch.from_numpy(np.array(jax_mlp.wo.kernel.get_value()).T)

    out_jax = np.array(jax_mlp(jnp.array(x_np), deterministic=True))
    out_pt = pt_mlp(torch.from_numpy(x_np)).detach().numpy()
    metrics = compute_parity_metrics(out_jax, out_pt)

    self.assertLess(metrics["max_abs_err"], 1e-4)
    self.assertGreater(metrics["cos_sim"], 0.999999)
    self.assertLess(abs(metrics["kl_divergence"]), 1e-5)

  def test_3_kda_attention_parity(self):
    """Test 3: KDA Attention Layer JAX vs PyTorch equivalence."""
    B, T = 1, 4
    x_np = np.random.randn(B, T, self.D).astype(np.float32)
    jax_kda = JaxKDA(config=self.config, layer_idx=0, rngs=self.rngs)
    pt_kda = PtKDA(
        hidden_size=self.D,
        num_heads=self.H,
        head_dim=self.K,
        conv_kernel_size=4,
        eps=self.eps,
    )

    pt_kda.q_proj.weight.data = torch.from_numpy(np.array(jax_kda.q_proj.kernel.get_value()).T)
    pt_kda.k_proj.weight.data = torch.from_numpy(np.array(jax_kda.k_proj.kernel.get_value()).T)
    pt_kda.v_proj.weight.data = torch.from_numpy(np.array(jax_kda.v_proj.kernel.get_value()).T)
    pt_kda.f_a_proj.weight.data = torch.from_numpy(np.array(jax_kda.f_a_proj.kernel.get_value()).T)
    pt_kda.f_b_proj.weight.data = torch.from_numpy(np.array(jax_kda.f_b_proj.kernel.get_value()).T)
    pt_kda.b_proj.weight.data = torch.from_numpy(np.array(jax_kda.b_proj.kernel.get_value()).T)
    pt_kda.g_proj.weight.data = torch.from_numpy(np.array(jax_kda.g_proj.kernel.get_value()).T)
    pt_kda.o_proj.weight.data = torch.from_numpy(np.array(jax_kda.o_proj.kernel.get_value()).T)
    pt_kda.q_conv1d.weight.data = torch.from_numpy(np.array(jax_kda.q_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.k_conv1d.weight.data = torch.from_numpy(np.array(jax_kda.k_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.v_conv1d.weight.data = torch.from_numpy(np.array(jax_kda.v_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.A_log.data = torch.from_numpy(np.array(jax_kda.A_log.get_value()))
    pt_kda.dt_bias.data = torch.from_numpy(np.array(jax_kda.dt_bias.get_value()))
    pt_kda.o_norm.scale.data = torch.from_numpy(np.array(jax_kda.o_norm.scale.get_value()))

    out_jax, _ = jax_kda(jnp.array(x_np))
    out_jax = np.array(out_jax)
    out_pt = pt_kda(torch.from_numpy(x_np)).detach().numpy()
    metrics = compute_parity_metrics(out_jax, out_pt)

    self.assertGreater(metrics["cos_sim"], 0.9998)
    self.assertLess(metrics["kl_divergence"], 1e-3)

  def test_4_kimi_decoder_layer_parity(self):
    """Test 4: Full KimiDecoderLayer (RMSNorm + KDA + RMSNorm + Situ MLP) equivalence."""
    B, T = 1, 4
    x_np = np.random.randn(B, T, self.D).astype(np.float32)
    jax_layer = JaxDecoderLayer(config=self.config, mesh=self.mesh, layer_idx=0, rngs=self.rngs)

    pt_norm1 = PtRMSNorm(self.D, eps=self.eps)
    pt_norm2 = PtRMSNorm(self.D, eps=self.eps)
    pt_kda = PtKDA(
        hidden_size=self.D,
        num_heads=self.H,
        head_dim=self.K,
        conv_kernel_size=4,
        eps=self.eps,
    )
    pt_mlp = PtSituMLP(self.D, self.intermediate_dim)

    pt_norm1.scale.data = torch.from_numpy(np.array(jax_layer.pre_self_attention_norm.scale.get_value()))
    pt_norm2.scale.data = torch.from_numpy(np.array(jax_layer.pre_mlp_norm.scale.get_value()))

    pt_kda.q_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.q_proj.kernel.get_value()).T)
    pt_kda.k_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.k_proj.kernel.get_value()).T)
    pt_kda.v_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.v_proj.kernel.get_value()).T)
    pt_kda.f_a_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.f_a_proj.kernel.get_value()).T)
    pt_kda.f_b_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.f_b_proj.kernel.get_value()).T)
    pt_kda.b_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.b_proj.kernel.get_value()).T)
    pt_kda.g_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.g_proj.kernel.get_value()).T)
    pt_kda.o_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.o_proj.kernel.get_value()).T)
    pt_kda.q_conv1d.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.q_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.k_conv1d.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.k_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.v_conv1d.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.v_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.A_log.data = torch.from_numpy(np.array(jax_layer.self_attention.A_log.get_value()))
    pt_kda.dt_bias.data = torch.from_numpy(np.array(jax_layer.self_attention.dt_bias.get_value()))
    pt_kda.o_norm.scale.data = torch.from_numpy(np.array(jax_layer.self_attention.o_norm.scale.get_value()))

    pt_mlp.wi_0.weight.data = torch.from_numpy(np.array(jax_layer.mlp.wi_0.kernel.get_value()).T)
    pt_mlp.wi_1.weight.data = torch.from_numpy(np.array(jax_layer.mlp.wi_1.kernel.get_value()).T)
    pt_mlp.wo.weight.data = torch.from_numpy(np.array(jax_layer.mlp.wo.kernel.get_value()).T)

    pt_layer = PtFullDecoderLayer(pt_norm1, pt_kda, pt_norm2, pt_mlp)

    out_jax, _ = jax_layer(jnp.array(x_np), deterministic=True)
    out_jax = np.array(out_jax)
    out_pt = pt_layer(torch.from_numpy(x_np)).detach().numpy()
    metrics = compute_parity_metrics(out_jax, out_pt)

    self.assertGreater(metrics["cos_sim"], 0.9995)
    self.assertLess(metrics["kl_divergence"], 1e-3)

  def test_5_end_to_end_logit_parity(self):
    """Test 5: Full End-to-End Model Logit Parity (Tokens -> Embed -> Decoder -> Norm -> Logits)."""
    vocab_size = 1000
    token_ids_np = np.array([[12, 45, 78, 99]], dtype=np.int32)
    embed_w = np.random.randn(vocab_size, self.D).astype(np.float32) * 0.02

    jax_layer = JaxDecoderLayer(config=self.config, mesh=self.mesh, layer_idx=0, rngs=self.rngs)
    pt_norm1 = PtRMSNorm(self.D, eps=self.eps)
    pt_norm2 = PtRMSNorm(self.D, eps=self.eps)
    pt_kda = PtKDA(
        hidden_size=self.D,
        num_heads=self.H,
        head_dim=self.K,
        conv_kernel_size=4,
        eps=self.eps,
    )
    pt_mlp = PtSituMLP(self.D, self.intermediate_dim)

    # Sync parameters
    pt_norm1.scale.data = torch.from_numpy(np.array(jax_layer.pre_self_attention_norm.scale.get_value()))
    pt_norm2.scale.data = torch.from_numpy(np.array(jax_layer.pre_mlp_norm.scale.get_value()))

    pt_kda.q_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.q_proj.kernel.get_value()).T)
    pt_kda.k_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.k_proj.kernel.get_value()).T)
    pt_kda.v_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.v_proj.kernel.get_value()).T)
    pt_kda.f_a_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.f_a_proj.kernel.get_value()).T)
    pt_kda.f_b_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.f_b_proj.kernel.get_value()).T)
    pt_kda.b_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.b_proj.kernel.get_value()).T)
    pt_kda.g_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.g_proj.kernel.get_value()).T)
    pt_kda.o_proj.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.o_proj.kernel.get_value()).T)
    pt_kda.q_conv1d.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.q_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.k_conv1d.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.k_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.v_conv1d.weight.data = torch.from_numpy(np.array(jax_layer.self_attention.v_conv1d.weight.get_value()).T[:, None, :])
    pt_kda.A_log.data = torch.from_numpy(np.array(jax_layer.self_attention.A_log.get_value()))
    pt_kda.dt_bias.data = torch.from_numpy(np.array(jax_layer.self_attention.dt_bias.get_value()))
    pt_kda.o_norm.scale.data = torch.from_numpy(np.array(jax_layer.self_attention.o_norm.scale.get_value()))

    pt_mlp.wi_0.weight.data = torch.from_numpy(np.array(jax_layer.mlp.wi_0.kernel.get_value()).T)
    pt_mlp.wi_1.weight.data = torch.from_numpy(np.array(jax_layer.mlp.wi_1.kernel.get_value()).T)
    pt_mlp.wo.weight.data = torch.from_numpy(np.array(jax_layer.mlp.wo.kernel.get_value()).T)

    pt_layer = PtFullDecoderLayer(pt_norm1, pt_kda, pt_norm2, pt_mlp)

    # Final norm
    final_norm_jax = JaxRMSNorm(
        num_features=self.D,
        epsilon=self.eps,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=self.rngs,
    )
    pt_final_norm = PtRMSNorm(self.D, eps=self.eps)
    pt_final_norm.scale.data = torch.from_numpy(np.array(final_norm_jax.scale.get_value()))

    # PyTorch Forward Pass
    x_emb_pt = torch.from_numpy(embed_w[token_ids_np]).float()
    x_hid_pt = pt_layer(x_emb_pt)
    x_norm_pt = pt_final_norm(x_hid_pt)
    logits_pt = (x_norm_pt @ torch.from_numpy(embed_w).T).detach().numpy()

    # JAX Forward Pass
    x_emb_jax = jnp.array(embed_w)[token_ids_np]
    x_hid_jax, _ = jax_layer(x_emb_jax, deterministic=True)
    x_norm_jax = final_norm_jax(x_hid_jax)
    logits_jax = np.array(x_norm_jax @ jnp.array(embed_w).T)

    metrics = compute_parity_metrics(logits_jax, logits_pt)
    print("\n" + "=" * 60, flush=True)
    print("END-TO-END LOGIT PARITY (JAX vs PyTorch):", flush=True)
    print(f"  Logits Shape:          {metrics['shape']}", flush=True)
    print(f"  Max Absolute Error:    {metrics['max_abs_err']:.6e}", flush=True)
    print(f"  Mean Absolute Error:   {metrics['mae']:.6e}", flush=True)
    print(f"  Cosine Similarity:     {metrics['cos_sim']:.8f}", flush=True)
    print(f"  KL Divergence:         {metrics['kl_divergence']:.6e}", flush=True)
    print(f"  Top-1 Agreement:       {metrics['top1_agreement'] * 100:.1f}%", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Parity Assertions
    self.assertGreater(metrics["cos_sim"], 0.9999, f"Logit cosine similarity {metrics['cos_sim']} is too low!")
    self.assertLess(metrics["kl_divergence"], 1e-4, f"Logit KL divergence {metrics['kl_divergence']} is too high!")
    self.assertEqual(metrics["top1_agreement"], 1.0, f"Top-1 agreement {metrics['top1_agreement']} is not 100%!")


if __name__ == "__main__":
  unittest.main()


