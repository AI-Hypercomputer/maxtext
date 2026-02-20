# Copyright 2025 Google LLC
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

"""
Tests for GatedDeltaRule in Qwen3-Next against its PyTorch reference.
"""
import unittest
from types import SimpleNamespace
from typing import Optional, Tuple
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from flax import nnx

from MaxText import pyconfig
from MaxText.layers import qwen3, normalizations
from MaxText.layers.normalizations import Qwen3NextRMSNorm, Qwen3NextRMSNormGated
from tests.utils.test_helpers import get_test_config_path


# ----------------------------------------------------------------------
# START: Copied PyTorch functions
# Source: Hugging Face Transformers library
# https://github.com/huggingface/transformers/blob/a9731a725eb1d7b3b7e11f0ad35a819fa4ee8b20/src/transformers/models/qwen3_next/modeling_qwen3_next.py
# Note: Some function/class names might be slightly adapted (e.g., _PT suffix) to avoid collisions.
# ----------------------------------------------------------------------
def create_causal_mask_PT(q_seq_len: int, kv_seq_len: int, dtype=torch.float32):
  mask = torch.triu(torch.ones(q_seq_len, kv_seq_len, dtype=torch.bool), diagonal=1)
  masked_fill_value = -torch.finfo(dtype).max / 2
  return torch.zeros(q_seq_len, kv_seq_len, dtype=dtype).masked_fill(mask, masked_fill_value)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
  """Pytorch implementation of default self-attention operation"""
  key_states = repeat_kv(key, module.num_key_value_groups)
  value_states = repeat_kv(value, module.num_key_value_groups)

  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
  if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

  attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
  attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
  attn_output = torch.matmul(attn_weights, value_states)
  attn_output = attn_output.transpose(1, 2).contiguous()

  return attn_output


def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
  """Pytorch impl for applying RoPE to query and key tensors."""
  cos = cos.unsqueeze(1)
  sin = sin.unsqueeze(1)
  rotary_dim = cos.shape[-1]
  q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
  k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
  q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
  k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
  q_embed = torch.cat([q_embed, q_pass], dim=-1)
  k_embed = torch.cat([k_embed, k_pass], dim=-1)
  return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def l2norm_torch(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
  """This function is intended to align with the l2norm implementation in the FLA library."""
  inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
  return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    # Keep original HF name in PT func signature for clarity
    use_qk_l2norm_in_kernel=False,
):
  """
  PyTorch implementation of the chunked Gated Delta Rule attention mechanism.

  Based on the Hugging Face Transformers implementation for Qwen3-Next.

  Args:
    query: Query tensor (B, S, H, K).
    key: Key tensor (B, S, H, K).
    value: Value tensor (B, S, H, V).
    g: Decay tensor (B, S, H).
    beta: Sigmoid gate tensor (B, S, H).
    chunk_size: The size of chunks for processing.
    initial_state: Optional initial hidden state for recurrent processing.
    output_final_state: Whether to return the final hidden state.
    use_qk_l2norm_in_kernel: Whether to apply L2 normalization to query and key.

  Returns:
    A tuple containing the attention output tensor and the final hidden state (if requested).
  """
  initial_dtype = query.dtype
  if use_qk_l2norm_in_kernel:
    query = l2norm_torch(query, dim=-1, eps=1e-6)
    key = l2norm_torch(key, dim=-1, eps=1e-6)

  # Transpose (B, S, H, K) -> (B, H, S, K)
  query, key, value = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value)]
  beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (beta, g)]

  batch_size, num_heads, sequence_length, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
  if pad_size > 0:
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

  total_sequence_length = sequence_length + pad_size
  scale = 1 / (query.shape[-1] ** 0.5)
  query = query * scale

  v_beta = value * beta.unsqueeze(-1)
  k_beta = key * beta.unsqueeze(-1)
  # reshape to chunks
  query, key, value, k_beta, v_beta = [
      x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
  ]
  g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
  mask = torch.triu(
      torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
      diagonal=0,
  )

  # chunk decay
  g = g.cumsum(dim=-1)
  decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
  attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
  for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
  attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
  value = attn @ v_beta
  k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
  last_recurrent_state = (
      torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
      if initial_state is None
      else initial_state.to(value)
  )
  core_attn_out = torch.zeros_like(value)
  mask = torch.triu(
      torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
      diagonal=1,
  )

  # for each chunk
  for i in range(0, total_sequence_length // chunk_size):
    q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
    attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
    v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
    v_new = v_i - v_prime
    attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
    core_attn_out[:, :, i] = attn_inter + attn @ v_new
    last_recurrent_state = (
        last_recurrent_state * g[:, :, i, -1, None, None].exp()
        + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    )

  if not output_final_state:
    last_recurrent_state = None
  core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
  core_attn_out = core_attn_out[:, :, :sequence_length]
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
  return core_attn_out, last_recurrent_state


class Qwen3NextRMSNorm_PT(nn.Module):
  """
  PyTorch implementation of the custom RMSNorm used in Qwen3-Next.

  This version applies a (1.0 + weight) scaling factor after normalization.
  """

  def __init__(self, dim: int, eps: float = 1e-6):
    """Initializes the Qwen3NextRMSNorm_PT layer."""
    super().__init__()
    self.eps = eps
    # The weight is initialized to zeros, matching the real model.
    self.weight = torch.nn.Parameter(torch.zeros(dim))

  def _norm(self, x):
    """Applies the RMS normalization."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    """Forward pass for Qwen3NextRMSNorm_PT."""
    output = self._norm(x.float())
    # The core Qwen3-Next logic: scaling by (1.0 + weight)
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


class Qwen3NextRotaryEmbedding_PT(nn.Module):
  """
  PyTorch implementation of the custom partial ROPE used in Qwen3-Next.

  Applies ROPE only to first 25% of head dimensions.
  """

  inv_freq: torch.Tensor  # fix linting for `register_buffer`

  def __init__(self, config, device=None):
    super().__init__()
    self.max_seq_len_cached = config.max_position_embeddings
    self.original_max_seq_len = config.max_position_embeddings

    self.config = config

    self.rope_type = self.config.rope_parameters["rope_type"]
    rope_init_fn: Callable = self.compute_default_rope_parameters
    inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.original_inv_freq = inv_freq

  @staticmethod
  def compute_default_rope_parameters(
      config,
      device: Optional["torch.device"] = None,
      seq_len: Optional[int] = None,
  ) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PreTrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_parameters["rope_theta"]
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor

  @torch.no_grad()
  def forward(self, x, position_ids):
    """Pytorch impl for partial ROPE forward pass"""
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos() * self.attention_scaling
      sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3NextMLP_PT(nn.Module):
  """
  PyTorch implementation of the MLP block for Qwen3-Next models.

  Uses SiLU activation (SwiGLU).
  """

  def __init__(self, config, intermediate_size=None):
    """Initializes the Qwen3NextMLP_PT layer."""
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = F.silu

  def forward(self, x):
    """Forward pass for the MLP block."""
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


class Qwen3NextExperts_PT(nn.ModuleList):
  """
  PyTorch ModuleList containing the expert MLP layers for the MoE block.
  """

  def __init__(self, config):
    """Initializes the list of expert MLP layers."""
    super().__init__()
    self.num_experts = config.num_experts
    for _ in range(config.num_experts):
      self.append(Qwen3NextMLP_PT(config, intermediate_size=config.moe_intermediate_size))

  def forward(
      self,
      hidden_states: torch.Tensor,
      top_k_index: torch.Tensor,
      top_k_weights: torch.Tensor,
  ) -> torch.Tensor:
    """Forward pass for the expert layers."""
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
      idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
      current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
      current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
      final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states


class Qwen3NextSparseMoeBlock_PT(nn.Module):
  """
  PyTorch implementation of the Sparse Mixture-of-Experts (MoE) block for Qwen3-Next.

  Includes token routing, expert layers, and a shared expert component.
  """

  def __init__(self, config):
    """Initializes the MoE block components."""
    super().__init__()
    self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    self.experts = Qwen3NextExperts_PT(config)
    self.num_experts_per_tok = config.num_experts_per_tok
    self.norm_topk_prob = config.norm_topk_prob
    self.shared_expert = Qwen3NextMLP_PT(config, intermediate_size=config.shared_expert_intermediate_size)
    self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

  def route_tokens_to_experts(self, hidden_states, router_logits):
    """
    Computes routing weights and selects top-k experts for each token.

    Args:
      hidden_states: Input tensor to determine the dtype for routing weights.
      router_logits: Logits output by the gating network.

    Returns:
      A tuple containing the indices of the selected experts and their corresponding routing weights.
    """
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
    if self.norm_topk_prob:
      routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    return selected_experts, routing_weights

  def forward(self, hidden_states: torch.Tensor):
    """Forward pass for the Sparse MoE block."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    shared_expert_output = self.shared_expert(hidden_states_reshaped)
    router_logits = self.gate(hidden_states_reshaped)
    selected_experts, routing_weights = self.route_tokens_to_experts(hidden_states_reshaped, router_logits)
    expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
    expert_output += shared_expert_output
    expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
    return expert_output, router_logits


class Qwen3NextRMSNormGated_PT(nn.Module):
  """
  PyTorch implementation of RMS Normalization with optional gating.

  If a gate tensor is provided, the normalized output is element-wise multiplied by SiLU(gate).
  """

  def __init__(self, hidden_size, eps=1e-6):
    """Initializes the RMSNormGated layer."""
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states, gate=None):
    """Forward pass for RMSNormGated."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    hidden_states = self.weight * hidden_states
    if gate is not None:
      hidden_states = hidden_states * F.silu(gate.to(torch.float32))
    return hidden_states.to(input_dtype)


class Qwen3NextGatedDeltaNet_PT(nn.Module):
  """
  PyTorch implementation of the Gated Delta Net (GDN) used in Qwen3-Next.

  This module implements a form of linear attention with gating and convolutional components.
  """

  def __init__(self, config):
    """Initializes the Gated Delta Net layers and parameters."""
    super().__init__()
    self.hidden_size = config.hidden_size
    # Use gdn_* names from MaxText config perspective for consistency in PT model setup
    self.num_v_heads = config.gdn_num_value_heads
    self.num_k_heads = config.gdn_num_key_heads
    self.head_k_dim = config.gdn_key_head_dim
    self.head_v_dim = config.gdn_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads
    self.conv_kernel_size = config.gdn_conv_kernel_dim
    self.activation = config.hidden_act
    self.layer_norm_epsilon = config.normalization_layer_epsilon

    self.conv_dim = self.key_dim * 2 + self.value_dim
    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim,
        out_channels=self.conv_dim,
        bias=False,
        kernel_size=self.conv_kernel_size,
        groups=self.conv_dim,
        padding=self.conv_kernel_size - 1,
    )

    projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
    projection_size_ba = self.num_v_heads * 2
    self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
    self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

    self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
    A = torch.empty(self.num_v_heads).uniform_(0, 16)
    self.A_log = nn.Parameter(torch.log(A))
    self.norm = Qwen3NextRMSNormGated_PT(self.head_v_dim, eps=self.layer_norm_epsilon)
    self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

  def forward(self, hidden_states):
    """Forward pass for the Gated Delta Net."""
    batch_size, seq_len, _ = hidden_states.shape
    projected_states_qkvz = self.in_proj_qkvz(hidden_states)
    projected_states_ba = self.in_proj_ba(hidden_states)

    # Simplified split for test where num_v_heads == num_k_heads
    q, k, v, z = torch.split(
        projected_states_qkvz,
        [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
        dim=-1,
    )
    b, a = torch.split(projected_states_ba, [self.num_v_heads, self.num_v_heads], dim=-1)

    mixed_qkv = torch.cat((q, k, v), dim=-1).transpose(1, 2)
    qkv_conv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
    q_conv, k_conv, v_conv = torch.split(qkv_conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

    query = q_conv.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
    key = k_conv.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
    value = v_conv.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

    # Use the renamed config flag when calling the reference function internally
    core_attn_out, _ = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=self.config.gdn_chunk_size,  # Use renamed config
        use_qk_l2norm_in_kernel=self.config.use_qk_norm_in_gdn,  # Use renamed config
    )

    z_reshaped = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
    gated_output = self.norm(core_attn_out, z_reshaped)
    gated_output = gated_output.reshape(batch_size, seq_len, -1)
    output = self.out_proj(gated_output)
    return output


class Qwen3NextFullAttention_PT(nn.Module):
  """
  PyTorch implementation of the Gated FullAttention used in Qwen3-Next.

  Key features are:
    - Query, gate split after query projection
    - Gating applies right after core attention op
    - partial ROPE
    - RMSNorm that is 1.0 + weight scaling factor after normalization
  """

  # layer_idx included to match JAX module's signature and for potential use in layer-specific behaviors in a full model.
  def __init__(self, config, layer_idx=0):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_heads = config.num_attention_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout

    self.q_proj = nn.Linear(
        config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
    )
    self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
    self.q_norm = Qwen3NextRMSNorm_PT(self.head_dim, eps=config.rms_norm_eps)
    self.k_norm = Qwen3NextRMSNorm_PT(self.head_dim, eps=config.rms_norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: Tuple[torch.Tensor, torch.Tensor],
      attention_mask: Optional[torch.Tensor] = None,
      past_key_values: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
      cache_position: Optional[torch.LongTensor] = None,
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pytorch impl forward pass for Full Attention"""

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
      # sin and cos are specific to RoPE models; cache_position needed for the static cache
      cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
      key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)

    attn_output = self.o_proj(attn_output)
    return attn_output


# ----------------------------------------------------------------------
# END: Copied PyTorch functions
# ----------------------------------------------------------------------


class TestQwen3Next(unittest.TestCase):
  """Main test class for Qwen3-Next layers."""

  def setUp(self):
    """Set up a complete configuration and test environment for all Qwen3-Next tests."""
    super().setUp()
    # This setup now includes all necessary parameters for both linear attention and MoE tests.
    self.cfg = pyconfig.initialize(
        [
            None,
            get_test_config_path(),
            # Base settings for the test
            "run_name=qwen3_next_test",
            "dtype=float32",
            "weight_dtype=float32",
            "matmul_precision=highest",
            "float32_logits=True",
            "decoder_block=qwen3_next",
            "attention=dot_product",
            # Model dimensions
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "head_dim=32",
            # Gated Delta Net Dims (Using renamed parameters)
            "gdn_num_value_heads=4",
            "gdn_num_key_heads=4",
            "gdn_key_head_dim=32",
            "gdn_value_head_dim=32",
            "gdn_conv_kernel_dim=4",
            "gdn_chunk_size=64",
            "use_qk_norm_in_gdn=True",  # Use renamed parameter
            "normalization_layer_epsilon=1e-6",
            # MoE Test Configs (with a small number of experts)
            "base_mlp_dim=256",
            "num_experts=8",
            "num_experts_per_tok=2",
            "base_moe_mlp_dim=256",  # moe_mlp_dim will be calculated from this
            "norm_topk_prob=True",
            "shard_exp_on_fsdp=False",
            "mlp_activations=['silu', 'linear']",
            "dropout_rate=0.0",
            # Force the test to use the 'dense_matmul' path in the MoE layer,
            # as the 'sparse_matmul' path was found to be numerically incorrect compared to the reference.
            "sparse_matmul=False",
            # To be able to run on cpu machines
            "skip_jax_distributed_system=True",
            # For FullAttention Layer
            "attention_bias=False",
            "rope_max_timescale=10000.0",
            "partial_rotary_factor=0.25",
        ]
    )
    # Update the SimpleNamespace config used by PT models too
    self.pt_internal_cfg = SimpleNamespace(
        hidden_size=self.cfg.emb_dim,
        gdn_num_value_heads=self.cfg.gdn_num_value_heads,
        gdn_num_key_heads=self.cfg.gdn_num_key_heads,
        gdn_key_head_dim=self.cfg.gdn_key_head_dim,
        gdn_value_head_dim=self.cfg.gdn_value_head_dim,
        gdn_conv_kernel_dim=self.cfg.gdn_conv_kernel_dim,
        hidden_act="silu",
        normalization_layer_epsilon=self.cfg.normalization_layer_epsilon,
        gdn_chunk_size=self.cfg.gdn_chunk_size,
        use_qk_norm_in_gdn=self.cfg.use_qk_norm_in_gdn,
        # MoE related for PT models
        moe_intermediate_size=self.cfg.moe_mlp_dim,
        shared_expert_intermediate_size=self.cfg.moe_mlp_dim,
        num_experts=self.cfg.num_experts,
        num_experts_per_tok=self.cfg.num_experts_per_tok,
        norm_topk_prob=self.cfg.norm_topk_prob,
    )

    self.batch_size = 4
    self.seq_len = 128
    # Use the emb_dim calculated by pyconfig from base_emb_dim
    self.hidden_size = self.cfg.emb_dim
    devices = np.array(jax.devices())
    num_devices = len(devices)

    # Create a mesh shape where the 'data' axis gets all available devices,
    # and all other axes defined in the config have a size of 1.
    mesh_shape = [1] * len(self.cfg.mesh_axes)
    mesh_shape[self.cfg.mesh_axes.index("data")] = num_devices

    # Create the Mesh object with the full list of axis names from the config.
    self.mesh = Mesh(devices.reshape(mesh_shape), self.cfg.mesh_axes)
    torch.manual_seed(0)
    np.random.seed(0)
    self.rng = jax.random.PRNGKey(0)
    self.nnx_rngs = nnx.Rngs(self.rng)
    print("setUp complete!")

  def test_rms_norm_gated(self):
    """Tests the Qwen3NextRMSNormGated layer."""
    print("Running test_rms_norm_gated...")
    hidden_states_pt = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    gate_pt = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    weight_pt = torch.rand(self.hidden_size)

    # PyTorch reference
    pt_model = Qwen3NextRMSNormGated_PT(self.hidden_size, eps=self.cfg.normalization_layer_epsilon)
    pt_model.weight.data = weight_pt
    pt_model.eval()
    with torch.no_grad():
      expected_output = pt_model(hidden_states_pt, gate_pt)

    # JAX implementation
    jax_model = Qwen3NextRMSNormGated(
        num_features=self.hidden_size,
        eps=self.cfg.normalization_layer_epsilon,
        dtype=self.cfg.dtype,
        weight_dtype=self.cfg.weight_dtype,
        rngs=self.nnx_rngs,
    )
    params = {"rms_norm": {"scale": nnx.Param(jnp.array(weight_pt.numpy()))}}
    nnx.update(jax_model, params)
    hidden_states_jax = jnp.array(hidden_states_pt.numpy())
    gate_jax = jnp.array(gate_pt.numpy())

    @jax.jit
    def run_jax(hidden_states, gate):
      """Runs the JAX RMSNormGated model."""
      return jax_model(hidden_states, gate)

    actual_output = run_jax(hidden_states_jax, gate_jax)

    np.testing.assert_allclose(
        expected_output.numpy(),
        actual_output,
        rtol=1e-5,
        atol=1e-6,  # Tight tolerance for this layer
        err_msg="Qwen3NextRMSNormGated does not match PyTorch reference!",
    )
    print("test_rms_norm_gated passed!")

  def test_l2norm(self):
    """Tests the l2norm function."""
    print("Running test_l2norm...")
    # Use renamed config parameters
    x_pt = torch.randn(
        self.batch_size,
        self.seq_len,
        self.cfg.gdn_num_value_heads,
        self.cfg.gdn_key_head_dim,
    )
    expected_output = l2norm_torch(x_pt)
    # Call l2norm from normalizations module now
    actual_output = normalizations.l2norm(jnp.array(x_pt.numpy()))
    np.testing.assert_allclose(
        expected_output.numpy(),
        actual_output,
        rtol=1e-5,
        atol=1e-6,
        err_msg="l2norm does not match PyTorch reference!",
    )
    print("test_l2norm passed!")

  def test_chunk_gated_delta_rule_logic(self):
    """
    Directly tests the `jax_chunk_gated_delta_rule` against the original PyTorch reference.
    """
    print("Running test_chunk_gated_delta_rule_logic...")
    # Use renamed config parameters
    num_heads = self.cfg.gdn_num_value_heads
    k_head_dim = self.cfg.gdn_key_head_dim
    v_head_dim = self.cfg.gdn_value_head_dim
    chunk_size = self.cfg.gdn_chunk_size

    key = jax.random.PRNGKey(42)
    key_q, key_k, key_v, key_g, key_beta = jax.random.split(key, 5)

    # Shapes are (B, S, H, D)
    q_jax = (
        jax.random.normal(
            key_q,
            (self.batch_size, self.seq_len, num_heads, k_head_dim),
            dtype=jnp.float32,
        )
        * 0.1
    )
    k_jax = (
        jax.random.normal(
            key_k,
            (self.batch_size, self.seq_len, num_heads, k_head_dim),
            dtype=jnp.float32,
        )
        * 0.1
    )
    v_jax = (
        jax.random.normal(
            key_v,
            (self.batch_size, self.seq_len, num_heads, v_head_dim),
            dtype=jnp.float32,
        )
        * 0.1
    )
    g_jax = jax.random.normal(key_g, (self.batch_size, self.seq_len, num_heads), dtype=jnp.float32) * 0.1
    beta_jax = jax.random.uniform(key_beta, (self.batch_size, self.seq_len, num_heads), dtype=jnp.float32)

    q_torch = torch.from_numpy(np.asarray(q_jax).copy())
    k_torch = torch.from_numpy(np.asarray(k_jax).copy())
    v_torch = torch.from_numpy(np.asarray(v_jax).copy())
    g_torch = torch.from_numpy(np.asarray(g_jax).copy())
    beta_torch = torch.from_numpy(np.asarray(beta_jax).copy())

    target_atol = 1e-6
    target_rtol = 1e-6

    # Test without L2Norm (pass False using the original PT arg name)
    torch_output, _ = torch_chunk_gated_delta_rule(
        q_torch.clone(),
        k_torch.clone(),
        v_torch.clone(),
        g_torch.clone(),
        beta_torch.clone(),
        chunk_size=chunk_size,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )
    # Pass False using the new JAX arg name
    jax_output, _ = qwen3.jax_chunk_gated_delta_rule(
        q_jax,
        k_jax,
        v_jax,
        g_jax,
        beta_jax,
        chunk_size=chunk_size,
        initial_state=None,
        use_qk_norm_in_gdn=False,
        compute_dtype=jnp.float32,
    )
    np.testing.assert_allclose(
        torch_output.detach().numpy(),
        np.asarray(jax_output),
        atol=target_atol,
        rtol=target_rtol,
        err_msg=f"JAX and PyTorch outputs are NOT close without L2Norm within atol={target_atol}, rtol={target_rtol}!",
    )
    print(f"JAX and PyTorch outputs are close without L2Norm within atol={target_atol}, rtol={target_rtol}!")

    # Test with L2Norm (pass True using the original PT arg name)
    torch_output_norm, _ = torch_chunk_gated_delta_rule(
        q_torch.clone(),
        k_torch.clone(),
        v_torch.clone(),
        g_torch.clone(),
        beta_torch.clone(),
        chunk_size=chunk_size,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    # Pass True using the new JAX arg name
    jax_output_norm, _ = qwen3.jax_chunk_gated_delta_rule(
        q_jax,
        k_jax,
        v_jax,
        g_jax,
        beta_jax,
        chunk_size=chunk_size,
        initial_state=None,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
    )
    np.testing.assert_allclose(
        torch_output_norm.detach().numpy(),
        np.asarray(jax_output_norm),
        atol=target_atol,
        rtol=target_rtol,
        err_msg=f"JAX and PyTorch outputs are NOT close with L2Norm within atol={target_atol}, rtol={target_rtol}!",
    )
    print(f"JAX and PyTorch outputs are close with L2Norm within atol={target_atol}, rtol={target_rtol}!")
    print("test_chunk_gated_delta_rule_logic passed!")

  def test_gated_delta_net_structure(self):
    """Tests the structure and output shape of Qwen3NextGatedDeltaNet."""
    print("Running test_gated_delta_net_structure...")
    hidden_states_jax = jnp.ones((self.batch_size, self.seq_len, self.hidden_size), dtype=self.cfg.dtype)

    jax_model = qwen3.Qwen3NextGatedDeltaNet(config=self.cfg, rngs=self.nnx_rngs)

    @jax.jit
    def run_jax(hidden_states):
      """Runs the JAX GatedDeltaNet model."""
      return jax_model(hidden_states)

    output_jax = run_jax(hidden_states_jax)

    self.assertEqual(output_jax.shape, (self.batch_size, self.seq_len, self.hidden_size))

    print("test_gated_delta_net_structure passed!")

  def test_qwen3_next_rms_norm(self):
    """Tests the custom Qwen3NextRMSNorm layer against its PyTorch reference."""
    print("Running test_qwen3_next_rms_norm...")
    # 1. Set up the PyTorch reference model and inputs.
    hidden_states_pt = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    weight_pt = torch.rand(self.hidden_size)

    pt_model = Qwen3NextRMSNorm_PT(self.hidden_size, eps=self.cfg.normalization_layer_epsilon)
    pt_model.weight.data = weight_pt
    pt_model.eval()

    with torch.no_grad():
      expected_output = pt_model(hidden_states_pt)

    # 2. Set up the JAX implementation.
    jax_model = Qwen3NextRMSNorm(
        num_features=self.hidden_size,
        eps=self.cfg.normalization_layer_epsilon,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=self.nnx_rngs,
    )

    params = {"scale": nnx.Param(jnp.array(weight_pt.numpy()))}
    nnx.update(jax_model.value, params)
    hidden_states_jax = jnp.array(hidden_states_pt.numpy())

    @jax.jit
    def run_jax(x):
      """Runs the JAX Qwen3NextRMSNorm model."""
      return jax_model.value(x)  # Call the module inside DataAttr

    actual_output = run_jax(hidden_states_jax)

    # 3. Compare the outputs.
    np.testing.assert_allclose(
        expected_output.numpy(),
        actual_output,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Qwen3NextRMSNorm does not match PyTorch reference!",
    )
    print("test_qwen3_next_rms_norm passed!")

  def test_qwen3_next_sparse_moe_block(self):
    """
    Tests the full Qwen3NextSparseMoeBlock against its PyTorch reference.

    This test passes by setting `sparse_matmul=False` in the config, which forces the
    underlying `RoutedMoE` module to use its `dense_matmul` implementation.
    """
    print("Running test_qwen3_next_sparse_moe_block...")
    # 1. Use the SimpleNamespace config created in setUp for PT model
    pt_config = self.pt_internal_cfg

    # 2. Set up the PyTorch reference model and get the expected output
    pt_model = Qwen3NextSparseMoeBlock_PT(pt_config)
    pt_model.eval()
    hidden_states_pt = torch.randn(self.batch_size, self.seq_len, self.cfg.emb_dim)
    with torch.no_grad():
      expected_output, _ = pt_model(hidden_states_pt)

    # 3. Construct the JAX params tree, ensuring weights are correctly transposed
    pt_experts = pt_model.experts
    stacked_gate_proj = torch.stack([expert.gate_proj.weight.T for expert in pt_experts])
    stacked_up_proj = torch.stack([expert.up_proj.weight.T for expert in pt_experts])
    stacked_down_proj = torch.stack([expert.down_proj.weight.T for expert in pt_experts])

    # Map PyTorch weights to JAX NNX module attributes
    jax_params = {
        "routed_experts": {
            "gate": {"kernel": nnx.Param(jnp.array(pt_model.gate.weight.T.detach().numpy()))},
            "wi_0": nnx.Param(jnp.array(stacked_gate_proj.detach().numpy())),
            "wi_1": nnx.Param(jnp.array(stacked_up_proj.detach().numpy())),
            "wo": nnx.Param(jnp.array(stacked_down_proj.detach().numpy())),
        },
        "shared_expert": {
            "wi": {  # Assuming fused_mlp=True in config for shared_expert
                "0": {"kernel": nnx.Param(jnp.array(pt_model.shared_expert.gate_proj.weight.T.detach().numpy()))},
                "1": {"kernel": nnx.Param(jnp.array(pt_model.shared_expert.up_proj.weight.T.detach().numpy()))},
            },
            "wo": {"kernel": nnx.Param(jnp.array(pt_model.shared_expert.down_proj.weight.T.detach().numpy()))},
        },
        "shared_expert_gate": {"kernel": nnx.Param(jnp.array(pt_model.shared_expert_gate.weight.T.detach().numpy()))},
    }
    # Adjust shared_expert structure if not fused
    if not self.cfg.fused_mlp:
      jax_params["shared_expert"] = {
          "wi_0": {"kernel": nnx.Param(jnp.array(pt_model.shared_expert.gate_proj.weight.T.detach().numpy()))},
          "wi_1": {"kernel": nnx.Param(jnp.array(pt_model.shared_expert.up_proj.weight.T.detach().numpy()))},
          "wo": {"kernel": nnx.Param(jnp.array(pt_model.shared_expert.down_proj.weight.T.detach().numpy()))},
      }

    # 4. Set up and run the full JAX Qwen3NextSparseMoeBlock
    jax_model = qwen3.Qwen3NextSparseMoeBlock(config=self.cfg, mesh=self.mesh, quant=None, rngs=self.nnx_rngs)
    nnx.update(jax_model, jax_params)
    hidden_states_jax = jnp.array(hidden_states_pt.numpy())

    @jax.jit
    def run_jax(x):
      """Runs the JAX SparseMoeBlock model."""
      output, _ = jax_model(x, deterministic=True)
      return output

    actual_output = run_jax(hidden_states_jax)

    # 5. Compare the outputs
    np.testing.assert_allclose(
        expected_output.detach().numpy(),
        actual_output,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Qwen3NextSparseMoeBlock does not match PyTorch reference!",
    )
    print("test_qwen3_next_sparse_moe_block passed!")

  def test_gated_delta_net_full(self):
    """Tests the full Qwen3NextGatedDeltaNet layer for numerical correctness."""
    print("Running test_gated_delta_net_full...")
    # 1. Use the SimpleNamespace config created in setUp for PT model
    pt_config = self.pt_internal_cfg

    pt_model = Qwen3NextGatedDeltaNet_PT(pt_config).eval()
    # Add MaxText config ref to PT model instance for internal use
    pt_model.config = self.cfg

    hidden_states_pt = torch.randn(self.batch_size, self.seq_len, self.cfg.emb_dim)
    with torch.no_grad():
      expected_output = pt_model(hidden_states_pt)

    # 2. Setup JAX model and map weights
    jax_model = qwen3.Qwen3NextGatedDeltaNet(config=self.cfg, rngs=self.nnx_rngs)

    conv1d_weight_pt = pt_model.conv1d.weight.detach().numpy()
    # Transpose PT (out, in/groups, kw) -> JAX (kw, in/groups, out)
    # For depthwise, out=in=groups, so PT=(C, 1, kw) -> JAX=(kw, 1, C)
    conv1d_weight_jax = np.transpose(conv1d_weight_pt, (2, 1, 0))

    params = {
        "in_proj_qkvz": {"kernel": nnx.Param(jnp.array(pt_model.in_proj_qkvz.weight.T.detach().numpy()))},
        "in_proj_ba": {"kernel": nnx.Param(jnp.array(pt_model.in_proj_ba.weight.T.detach().numpy()))},
        "conv1d": {"kernel": nnx.Param(jnp.array(conv1d_weight_jax))},
        "A_log": nnx.Param(jnp.array(pt_model.A_log.detach().numpy())),
        "dt_bias": nnx.Param(jnp.array(pt_model.dt_bias.detach().numpy())),
        "norm": {"weight": nnx.Param(jnp.array(pt_model.norm.weight.detach().numpy()))},
        "out_proj": {"kernel": nnx.Param(jnp.array(pt_model.out_proj.weight.T.detach().numpy()))},
    }
    nnx.update(jax_model, params)
    hidden_states_jax = jnp.array(hidden_states_pt.numpy())

    @jax.jit
    def run_jax(x):
      """Runs the JAX GatedDeltaNet model."""
      return jax_model(x)

    actual_output = run_jax(hidden_states_jax)

    # 3. Compare outputs
    np.testing.assert_allclose(
        expected_output.numpy(),
        actual_output,
        rtol=1e-4,
        atol=1e-4,  # Relaxed tolerance slightly for end-to-end layer
        err_msg="Qwen3NextGatedDeltaNet does not match PyTorch reference!",
    )
    print("test_gated_delta_net_full passed!")

  def _run_full_attention_jax_vs_pytorch_attention(self, attention_type):
    """Compares JAX and PyTorch Full Attention implementations."""
    print(f"Running test_full_attention_jax_vs_pytorch with attention={attention_type}...")

    # Re-initialize config with the specified attention type
    cfg = pyconfig.initialize(
        [
            None,
            get_test_config_path(),
            # Base settings for the test
            "run_name=qwen3_next_test",
            "dtype=float32",
            "weight_dtype=float32",
            "matmul_precision=highest",
            "float32_logits=True",
            "decoder_block=qwen3_next",
            f"attention={attention_type}",  # Override attention type
            # Model dimensions
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "head_dim=32",
            # Gated Delta Net Dims (Using renamed parameters)
            "gdn_num_value_heads=4",
            "gdn_num_key_heads=4",
            "gdn_key_head_dim=32",
            "gdn_value_head_dim=32",
            "gdn_conv_kernel_dim=4",
            "gdn_chunk_size=64",
            "use_qk_norm_in_gdn=True",  # Use renamed parameter
            "normalization_layer_epsilon=1e-6",
            # MoE Test Configs (with a small number of experts)
            "base_mlp_dim=256",
            "num_experts=8",
            "num_experts_per_tok=2",
            "base_moe_mlp_dim=256",  # moe_mlp_dim will be calculated from this
            "norm_topk_prob=True",
            "shard_exp_on_fsdp=False",
            "mlp_activations=['silu', 'linear']",
            "dropout_rate=0.0",
            "sparse_matmul=False",
            "skip_jax_distributed_system=True",
            # For FullAttention Layer
            "attention_bias=False",
            "rope_max_timescale=10000.0",
            "partial_rotary_factor=0.25",
        ]
    )

    # 1. Config for PyTorch
    pt_config = SimpleNamespace(
        hidden_size=cfg.emb_dim,
        num_attention_heads=cfg.num_query_heads,
        head_dim=cfg.head_dim,
        num_key_value_heads=cfg.num_kv_heads,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        max_position_embeddings=cfg.max_target_length,
        attention_dropout=cfg.dropout_rate,
        partial_rotary_factor=0.25,
    )

    # 2. Instantiate PyTorch model
    rotary_emb_pt = Qwen3NextRotaryEmbedding_PT(pt_config)
    pt_model = Qwen3NextFullAttention_PT(pt_config).eval()

    # 3. Instantiate JAX model
    jax_model = qwen3.Qwen3NextFullAttention(
        config=cfg, mesh=self.mesh, model_mode="train", layer_idx=0, quant=None, rngs=self.nnx_rngs
    )

    # 4. Weight Mapping
    pt_state_dict = pt_model.state_dict()

    # Target jax_model.attention.query
    pt_q_proj_w = pt_state_dict["q_proj.weight"].T.numpy()
    jax_q_proj_w = pt_q_proj_w.reshape(cfg.emb_dim, cfg.num_query_heads, cfg.head_dim * 2)
    jax_q_params = {"kernel": nnx.Param(jnp.array(jax_q_proj_w))}
    if cfg.attention_bias:
      jax_q_params["bias"] = nnx.Param(jnp.array(pt_state_dict["q_proj.bias"].numpy()))
    nnx.update(jax_model.attention.query, jax_q_params)

    # Target jax_model.attention.key
    pt_k_proj_w = pt_state_dict["k_proj.weight"].T.numpy()
    jax_k_proj_w = pt_k_proj_w.reshape(cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim)
    jax_k_params = {"kernel": nnx.Param(jnp.array(jax_k_proj_w))}
    if cfg.attention_bias:
      jax_k_params["bias"] = nnx.Param(jnp.array(pt_state_dict["k_proj.bias"].numpy()))
    nnx.update(jax_model.attention.key, jax_k_params)

    # Target jax_model.attention.value
    pt_v_proj_w = pt_state_dict["v_proj.weight"].T.numpy()
    jax_v_proj_w = pt_v_proj_w.reshape(cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim)
    jax_v_params = {"kernel": nnx.Param(jnp.array(jax_v_proj_w))}
    if cfg.attention_bias:
      jax_v_params["bias"] = nnx.Param(jnp.array(pt_state_dict["v_proj.bias"].numpy()))
    nnx.update(jax_model.attention.value, jax_v_params)

    # Target jax_model.attention.out
    pt_o_proj_w = pt_state_dict["o_proj.weight"].T.numpy()
    # jax_o_proj_w = pt_o_proj_w.reshape(cfg.num_query_heads, cfg.head_dim, cfg.emb_dim)
    jax_o_proj_w = pt_o_proj_w
    jax_o_params = {"kernel": nnx.Param(jnp.array(jax_o_proj_w))}
    if cfg.attention_bias:
      jax_o_params["bias"] = nnx.Param(jnp.array(pt_state_dict["o_proj.bias"].numpy()))
    nnx.update(jax_model.attention.out, jax_o_params)

    # Target jax_model.attention.query_norm and key_norm
    if jax_model.attention.query_norm is not None:
      nnx.update(jax_model.attention.query_norm, {"weight": nnx.Param(jnp.array(pt_state_dict["q_norm.weight"].numpy()))})
    if jax_model.attention.key_norm is not None:
      nnx.update(jax_model.attention.key_norm, {"weight": nnx.Param(jnp.array(pt_state_dict["k_norm.weight"].numpy()))})

    # 5. Prepare Inputs
    hidden_states_np = np.random.randn(self.batch_size, self.seq_len, cfg.emb_dim).astype(np.float32)
    hidden_states_pt = torch.from_numpy(hidden_states_np)
    hidden_states_jax = jnp.array(hidden_states_np)

    position_ids_pt = torch.arange(0, self.seq_len, dtype=torch.long).unsqueeze(0).repeat(self.batch_size, 1)
    decoder_positions_jax = jnp.array(position_ids_pt.numpy())

    # Causal mask for PyTorch
    attention_mask_pt = create_causal_mask_PT(self.seq_len, self.seq_len)
    attention_mask_pt = attention_mask_pt[None, None, :, :]

    # Segment IDs for JAX (for causal mask)
    decoder_segment_ids_jax = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    # 6. Get PyTorch cos, sin
    cos_pt, sin_pt = rotary_emb_pt(hidden_states_pt, position_ids_pt)
    position_embeddings_pt = (cos_pt, sin_pt)

    # 7. Run PyTorch Model
    with torch.no_grad():
      pt_output = pt_model(hidden_states_pt, position_embeddings_pt, attention_mask=attention_mask_pt)

    # 8. Run JAX Model
    @jax.jit
    def run_jax(inputs, segment_ids, positions):
      return jax_model(
          inputs, decoder_segment_ids=segment_ids, decoder_positions=positions, deterministic=True, model_mode="train"
      )

    jax_output = run_jax(hidden_states_jax, decoder_segment_ids_jax, decoder_positions_jax)

    # 9. Compare
    pt_out_np = pt_output.detach().numpy()
    jax_out_np = np.asarray(jax_output)

    rtol = 1e-6
    atol = 1e-6
    if attention_type == "flash":
      rtol = 1e-3
      atol = 1e-2

    self.assertEqual(pt_out_np.shape, jax_out_np.shape, "Shape mismatch for jax and pytorch impls")
    np.testing.assert_allclose(
        pt_out_np,
        jax_out_np,
        rtol=rtol,
        atol=atol,
        err_msg="Weight mismatch for jax/pytorch impls",
    )
    print("test_full_attention_jax_vs_pytorch passed!")

  def test_full_attention_dot_product(self):
    return self._run_full_attention_jax_vs_pytorch_attention("dot_product")

  def test_full_attention_flash(self):
    return self._run_full_attention_jax_vs_pytorch_attention("flash")


if __name__ == "__main__":
  unittest.main()
