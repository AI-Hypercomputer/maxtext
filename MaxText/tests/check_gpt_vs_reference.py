"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional
import os.path
import torch
from torch import nn
import torch.nn.functional as F
import jax
import unittest
import jax.numpy as jnp
from jax.sharding import Mesh
from MaxText.globals import PKG_DIR
from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText.layers import attentions, moe, embeddings
import numpy as np
from MaxText.layers.initializers import nd_dense_init
from types import SimpleNamespace
import math


"""  
Tests for Attention & MLP in GPT OSS.

GPT OSS PyTorch implementation at:
https://github.com/huggingface/transformers/blob/31ab7168ff7e07f61c90134e5238c4d97606aa70/src/transformers/models/gpt_oss/modular_gpt_oss.py
"""


# Reference implementation
class GptOssExperts(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.intermediate_size = config.intermediate_size
    self.num_experts = config.num_local_experts
    self.hidden_size = config.hidden_size
    self.expert_dim = self.intermediate_size
    self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
    self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
    self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
    self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
    self.alpha = 1.702
    self.limit = config.limit

  def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
    num_experts = routing_weights.shape[1]
    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

    gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    glu = gate * torch.sigmoid(gate * self.alpha)
    next_states = torch.bmm(((up + 1) * glu), self.down_proj)
    next_states = next_states + self.down_proj_bias[..., None, :]
    next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
    next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    next_states = next_states.sum(dim=0)
    return next_states.reshape(batch_size, seq_len, self.hidden_size)


# Reference implementation
class GptOssTopKRouter(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.top_k = config.num_experts_per_tok
    self.num_experts = config.num_local_experts
    self.hidden_dim = config.hidden_size
    self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
    self.bias = nn.Parameter(torch.empty(self.num_experts))

  def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    return router_scores, router_indices


# Reference implementation
class GptOssMLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.router = GptOssTopKRouter(config)
    self.experts = GptOssExperts(config)

  def forward(self, hidden_states):
    router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
    routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
    return routed_out, router_scores


# Standard implementation of repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """
  Repeats the key and value heads to match the number of query heads in GQA.
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Reference implementation
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
  key_states = repeat_kv(key, module.num_key_value_groups)
  value_states = repeat_kv(value, module.num_key_value_groups)
  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
  if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

  if hasattr(module, "sinks") and module.sinks is not None:
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
  else:
    probs = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
    scores = probs

  attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
  attn_output = torch.matmul(attn_weights, value_states)
  attn_output = attn_output.transpose(1, 2).contiguous()
  return attn_output, attn_weights


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().numpy())


class Config:

  hidden_size = 16
  intermediate_size = 16
  num_local_experts = 8
  num_experts_per_tok = 2
  limit = 7.0
  num_attention_heads = 8
  num_key_value_heads = 4
  head_dim = 8
  attention_dropout = 0.0


class GptOssMLPTest(unittest.TestCase):

  def test_mlp_block(self):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    config = Config()
    # Set print options to show full tensors and arrays
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=np.inf)
    predefined_weights = {
        "router.weight": torch.randn(config.num_local_experts, config.hidden_size),
        "router.bias": torch.randn(config.num_local_experts),
        "experts.gate_up_proj": torch.randn(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size),
        "experts.gate_up_proj_bias": torch.randn(config.num_local_experts, 2 * config.intermediate_size),
        "experts.down_proj": torch.randn(config.num_local_experts, config.intermediate_size, config.hidden_size),
        "experts.down_proj_bias": torch.randn(config.num_local_experts, config.hidden_size),
    }

    batch_size = 4
    seq_len = 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # reference model
    model = GptOssMLP(config)
    model.load_state_dict(predefined_weights)
    model.eval()
    with torch.no_grad():
      expected_output, _ = model(hidden_states)

    # MaxText model
    cfg = pyconfig.initialize(
        [None, os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="gpt_oss_mlp_test",
        enable_checkpointing=False,
        model_name="default",
        dtype="float32",
        weight_dtype="float32",
        megablox=False,
        sparse_matmul=True,
        per_device_batch_size=1,
        max_target_length=seq_len,
        max_prefill_predict_length=seq_len,
        base_emb_dim=config.hidden_size,
        base_mlp_dim=config.intermediate_size,
        base_moe_mlp_dim=config.intermediate_size,
        mlp_activations=["sigmoid", "linear"],
        mlp_activations_limit=config.limit,
        routed_bias=True,
        mlp_bias=True,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        decoder_block="gpt_oss",
        attention="dot_product",
    )
    jax_hidden_states = to_jax(hidden_states)
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    jax_model = moe.get_routed_moe(
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.base_mlp_dim,
        dtype=cfg.dtype,
    )

    moe_variables = {
        "moe_variables": {
            "gate": {
                "kernel": to_jax(predefined_weights["router.weight"].transpose(0, 1)),
                "bias": to_jax(predefined_weights["router.bias"]),
            },
            "wi_0": to_jax(predefined_weights["experts.gate_up_proj"][..., ::2]),
            "wi_0_bias": to_jax(predefined_weights["experts.gate_up_proj_bias"][..., ::2]),
            "wi_1": to_jax(predefined_weights["experts.gate_up_proj"][..., 1::2]),
            "wi_1_bias": to_jax(predefined_weights["experts.gate_up_proj_bias"][..., 1::2]),
            "wo": to_jax(predefined_weights["experts.down_proj"]),
            "wo_bias": to_jax(predefined_weights["experts.down_proj_bias"]),
        },
    }
    actual_output, _ = jax.jit(jax_model.apply)(moe_variables, jax_hidden_states)
    # Add normalization to let logits at same scale
    normalized_expected = to_jax(expected_output) / jnp.linalg.norm(to_jax(expected_output))
    normalized_actual = actual_output / jnp.linalg.norm(actual_output)
    mse = jnp.mean((normalized_expected - normalized_actual) ** 2)
    self.assertLess(mse, 1e-3, f"mlp mismatch, MSE: {mse}")
    np.testing.assert_allclose(normalized_expected, normalized_actual, rtol=1e-3, atol=1e-2)


class GptOssAttentionTest(unittest.TestCase):
  """Test for the MaxText GPT OSS implementation."""

  def setUp(self):
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    self.config = Config()
    self.batch_size = 4
    self.seq_len = 128

    self.mock_module_with_sinks = SimpleNamespace(
        num_key_value_groups=self.config.num_attention_heads // self.config.num_key_value_heads,
        sinks=torch.randn(self.config.num_attention_heads),
        training=False,
    )

    self.query = torch.randn(self.batch_size, self.config.num_attention_heads, self.seq_len, self.config.head_dim)
    self.key = torch.randn(self.batch_size, self.config.num_key_value_heads, self.seq_len, self.config.head_dim)
    self.value = torch.randn(self.batch_size, self.config.num_key_value_heads, self.seq_len, self.config.head_dim)
    self.attention_mask = None
    self.scaling = 1.0 / (self.config.head_dim**0.5)

    # JAX tensors
    self.jax_query = to_jax(self.query)
    self.jax_key = to_jax(self.key)
    self.jax_value = to_jax(self.value)
    self.jax_sinks = to_jax(self.mock_module_with_sinks.sinks)
    self.jax_query_t = jnp.transpose(self.jax_query, (0, 2, 1, 3))
    self.jax_key_t = jnp.transpose(self.jax_key, (0, 2, 1, 3))
    self.jax_value_t = jnp.transpose(self.jax_value, (0, 2, 1, 3))

  def test_dot_product_attention_with_sinks(self):
    expected_attn_output, _ = eager_attention_forward(
        module=self.mock_module_with_sinks,
        query=self.query,
        key=self.key,
        value=self.value,
        attention_mask=self.attention_mask,
        scaling=self.scaling,
        dropout=0.0,
    )

    cfg_dot = pyconfig.initialize(
        [None, os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="gpt_oss_attention_test_dot",
        enable_checkpointing=False,
        model_name="default",
        dtype="float32",
        per_device_batch_size=self.batch_size,
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        base_num_query_heads=self.config.num_attention_heads,
        base_num_kv_heads=self.config.num_key_value_heads,
        head_dim=self.config.head_dim,
        attention="dot_product",
        attention_bias=False,
        attention_sink=True,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg_dot)
    mesh = Mesh(devices_array, cfg_dot.mesh_axes)

    attention_op_dot = attentions.AttentionOp(
        config=cfg_dot,
        mesh=mesh,
        attention_kernel="dot_product",
        max_target_length=self.seq_len,
        num_query_heads=self.config.num_attention_heads,
        num_kv_heads=self.config.num_key_value_heads,
        dtype=jnp.float32,
        attention_type=attentions.AttentionType.FULL,
    )

    @jax.jit
    def run_dot_product_attention(q, k, v):
      unnormalized_output, _, sum_val = attention_op_dot.apply_attention_dot(
          query=q,
          key=k,
          value=v,
          decoder_segment_ids=jnp.ones((self.batch_size, self.seq_len)),
          model_mode="train",
          sinks=self.jax_sinks,
          qk_product_einsum=jnp.einsum,
          wv_product_einsum=jnp.einsum,
      )
      return unnormalized_output / sum_val

    scaled_jax_query_t = self.jax_query_t * self.scaling
    actual_attn_output_dot = run_dot_product_attention(scaled_jax_query_t, self.jax_key_t, self.jax_value_t)

    mse_dot = jnp.mean((to_jax(expected_attn_output) - actual_attn_output_dot) ** 2)
    self.assertLess(mse_dot, 1e-3, f"dot-product attention mismatch, MSE: {mse_dot}")
    np.testing.assert_allclose(to_jax(expected_attn_output), actual_attn_output_dot, rtol=1e-3, atol=1e-2)

  def test_flash_attention_with_sinks(self):
    expected_attn_output, _ = eager_attention_forward(
        module=self.mock_module_with_sinks,
        query=self.query,
        key=self.key,
        value=self.value,
        attention_mask=self.attention_mask,
        scaling=self.scaling,
        dropout=0.0,
    )

    cfg_flash = pyconfig.initialize(
        [None, os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="gpt_oss_attention_test_flash",
        enable_checkpointing=False,
        model_name="default",
        dtype="float32",
        per_device_batch_size=self.batch_size,
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        base_num_query_heads=self.config.num_attention_heads,
        base_num_kv_heads=self.config.num_key_value_heads,
        head_dim=self.config.head_dim,
        attention="flash",
        attention_bias=False,
        attention_sink=True,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg_flash)
    mesh = Mesh(devices_array, cfg_flash.mesh_axes)

    attention_op_flash = attentions.AttentionOp(
        config=cfg_flash,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=self.seq_len,
        num_query_heads=self.config.num_attention_heads,
        num_kv_heads=self.config.num_key_value_heads,
        dtype=jnp.float32,
        attention_type=attentions.AttentionType.FULL,
    )

    @jax.jit
    def run_flash_attention(q, k, v, sinks_logits):
      output = attention_op_flash.tpu_flash_attention(
          query=q,
          key=k,
          value=v,
          decoder_segment_ids=None,
          sinks=sinks_logits,
      )
      return output

    scaled_jax_query_flash_t = self.jax_query_t * self.scaling
    actual_attn_output_flash = run_flash_attention(
        scaled_jax_query_flash_t, self.jax_key_t, self.jax_value_t, self.jax_sinks
    )
    mse_flash = jnp.mean((to_jax(expected_attn_output) - actual_attn_output_flash) ** 2)
    self.assertLess(mse_flash, 1e-3, f"flash attention mismatch, MSE: {mse_flash}")
    np.testing.assert_allclose(to_jax(expected_attn_output), actual_attn_output_flash, rtol=1e-3, atol=1e-2)


# https://github.com/openai/gpt-oss/blob/152fc0ce3b500752214c8d59440ef3a909e1e556/gpt_oss/torch/model.py#L50-L150
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
  cos = cos.unsqueeze(-2).to(x.dtype)
  sin = sin.unsqueeze(-2).to(x.dtype)
  x1, x2 = torch.chunk(x, 2, dim=-1)
  o1 = x1 * cos - x2 * sin
  o2 = x2 * cos + x1 * sin
  return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):

  def __init__(
      self,
      head_dim: int,
      base: int,
      dtype: torch.dtype,
      initial_context_length: int = 4096,
      scaling_factor: float = 1.0,
      ntk_alpha: float = 1.0,
      ntk_beta: float = 32.0,
      device: torch.device | None = None,
  ) -> None:
    super().__init__()
    self.head_dim = head_dim
    self.base = base
    self.dtype = dtype
    self.initial_context_length = initial_context_length
    self.scaling_factor = scaling_factor
    self.ntk_alpha = ntk_alpha
    self.ntk_beta = ntk_beta
    self.device = device

  def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
    """See YaRN paper: https://arxiv.org/abs/2309.00071"""
    freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device) / self.head_dim)
    if self.scaling_factor > 1.0:
      concentration = 0.1 * math.log(self.scaling_factor) + 1.0  # YaRN concentration

      d_half = self.head_dim / 2
      # NTK by parts
      low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
      high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
      assert 0 < low < high < d_half - 1

      interpolation = 1.0 / (self.scaling_factor * freq)
      extrapolation = 1.0 / freq

      ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
      mask = 1 - ramp.clamp(0, 1)

      inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
      concentration = 1.0
      inv_freq = 1.0 / freq

    return concentration, inv_freq

  def _compute_cos_sin(self, num_tokens: int):
    concentration, inv_freq = self._compute_concentration_and_inv_freq()
    t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos() * concentration
    sin = freqs.sin() * concentration
    return cos, sin

  def forward(
      self,
      query: torch.Tensor,
      key: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = query.shape[0]
    cos, sin = self._compute_cos_sin(num_tokens)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_dim)
    query = _apply_rotary_emb(query, cos, sin)
    query = query.reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_dim)
    key = _apply_rotary_emb(key, cos, sin)
    key = key.reshape(key_shape)
    return query, key


class GptOssYarnTest(unittest.TestCase):
  """Test for the GptOss Yarn RoPE implementation."""

  def setUp(self):
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    self.dtype = "float32"
    # attention
    self.num_attention_heads = 8
    self.num_key_value_heads = 4
    self.head_dim = 8
    # rope
    self.rope_max_timescale = 150_000
    self.max_position_embeddings = 131072
    self.original_max_position_embeddings = 4096
    self.rope_factor = 32
    self.beta_fast = 32
    self.beta_slow = 1
    # data
    self.batch_size = 1
    self.seq_len = 512
    # Torch tensors
    self.query = torch.randn(self.batch_size, self.num_attention_heads, self.seq_len, self.head_dim)
    self.key = torch.randn(self.batch_size, self.num_key_value_heads, self.seq_len, self.head_dim)
    self.positions = torch.arange(self.seq_len).unsqueeze(0).repeat(self.batch_size, 1)
    # JAX tensors
    self.jax_query = to_jax(self.query)
    self.jax_key = to_jax(self.key)
    self.jax_positions = to_jax(self.positions)

  def test_yarn_rotary_embedding(self):

    # Create and initialize the torch Yarn RoPE
    torch_embedding = RotaryEmbedding(
        self.head_dim,
        base=self.rope_max_timescale,
        dtype=self.dtype,
        initial_context_length=self.original_max_position_embeddings,
        scaling_factor=self.rope_factor,
        ntk_alpha=self.beta_slow,
        ntk_beta=self.beta_fast,
    )
    # Apply RoPE using the PyTorch implementation
    # q_rope_pt, k_rope_pt = torch_embedding(self.query, self.key)

    # CRITICAL: Reshape tensors for the PyTorch model's expected input format.
    # From (B, S, N, H) -> (S, B * N, H)
    q_pt_reshaped = self.query.permute(1, 0, 2, 3).reshape(self.seq_len, -1, self.head_dim)
    k_pt_reshaped = self.key.permute(1, 0, 2, 3).reshape(self.seq_len, -1, self.head_dim)

    # Apply RoPE using the PyTorch implementation
    q_rope_pt_reshaped, k_rope_pt_reshaped = torch_embedding(q_pt_reshaped, k_pt_reshaped)

    # CRITICAL: Reshape the output back to the original format for comparison.
    # From (S, B * N, H) -> (B, S, N, H)
    q_rope_pt = q_rope_pt_reshaped.view(self.seq_len, self.batch_size, self.num_attention_heads, self.head_dim).permute(
        1, 0, 2, 3
    )
    k_rope_pt = k_rope_pt_reshaped.view(self.seq_len, self.batch_size, self.num_key_value_heads, self.head_dim).permute(
        1, 0, 2, 3
    )

    # Create and initialize the JAX Yarn RoPE
    model_jax = embeddings.YarnRotaryEmbedding(
        max_position_embeddings=self.max_position_embeddings,
        original_max_position_embeddings=self.original_max_position_embeddings,
        beta_fast=self.beta_fast,
        beta_slow=self.beta_slow,
        rope_theta=self.rope_max_timescale,
        rope_factor=self.rope_factor,
        embedding_dims=self.head_dim,
        fprop_dtype=self.dtype,
        interleave=False,
        truncate=True,
        attention_scaling=True,
        # rngs=self.rngs,
    )
    # # Apply the JAX RoPE
    q_rope_jax = model_jax(self.jax_query, self.jax_positions)
    k_rope_jax = model_jax(self.jax_key, self.jax_positions)

    # Compare outputs from the PyTorch and JAX implementations
    np.testing.assert_allclose(to_jax(q_rope_pt), q_rope_jax, rtol=1e-3, atol=0.05)
    np.testing.assert_allclose(to_jax(k_rope_pt), k_rope_jax, rtol=1e-3, atol=0.05)


if __name__ == "__main__":
  unittest.main()
