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

"""Tests for the Gemma3 Rotary Positional Embedding (RoPE) implementation.

This module validates the correctness of MaxText's JAX-based RoPE layer
against a reference PyTorch implementation from the Hugging Face `transformers`
library. It specifically tests the linear scaling variant of RoPE used in
Gemma3.
"""

import unittest

import torch
from torch import nn

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import numpy as np

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from MaxText.layers import embeddings


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  """Converts a PyTorch tensor to a JAX array.

  Args:
    pt_tensor: The PyTorch tensor to convert.

  Returns:
    The equivalent JAX array.
  """
  return jnp.asarray(pt_tensor.detach().numpy())


### original Pytorch Reference implementation
class Gemma3RotaryEmbedding(nn.Module):
  """PyTorch reference implementation of Gemma3's Rotary Positional Embedding.

  This class is a reference implementation taken from the Hugging Face
  transformers library to validate the correctness of the MaxText RoPE
  implementation.
  """

  inv_freq: torch.Tensor  # fix linting for `register_buffer`

  def __init__(self, config, device=None):
    """Initializes the rotary embedding module.

    Args:
      config: A configuration object containing RoPE parameters like
        `rope_scaling`, `max_position_embeddings`, etc.
      device: The device to place tensors on.
    """
    super().__init__()
    # BC: "rope_type" was originally "type"
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
      self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
    else:
      self.rope_type = "default"
    self.max_seq_len_cached = config.max_position_embeddings
    self.original_max_seq_len = config.max_position_embeddings

    self.config = config
    self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

    self.mesh = Mesh(jax.devices(), "data")

    inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.original_inv_freq = self.inv_freq

  @torch.no_grad()
  def forward(self, x, position_ids):
    """Computes the cosine and sine components for rotary embeddings.

    Args:
      x: The input tensor, used to determine the device and dtype.
      position_ids: The 1D tensor of token positions.

    Returns:
      A tuple of (cos, sin) tensors for the rotary embedding.
    """
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos() * self.attention_scaling
      sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
  """Applies Rotary Position Embedding to the query and key tensors.

  Args:
      q (`torch.Tensor`): The query tensor.
      k (`torch.Tensor`): The key tensor.
      cos (`torch.Tensor`): The cosine part of the rotary embedding.
      sin (`torch.Tensor`): The sine part of the rotary embedding.
      position_ids (`torch.Tensor`, *optional*):
          Deprecated and unused.
      unsqueeze_dim (`int`, *optional*, defaults to 1):
          The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
          sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
          that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
          k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
          cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
          the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
  Returns:
      `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
  """
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


class Gemma3RotaryEmbeddingTest(unittest.TestCase):
  """Test for Gemma3 RoPE implementation with linear scaling."""

  def test_rope_compare_pytorch_and_jax(self):
    """Validates the MaxText RoPE implementation against the PyTorch reference."""
    # Config parameters
    batch_size = 4
    seq_len = 128
    num_heads = 8
    head_dim = 64
    # embedding_dims = num_heads * head_dim
    min_timescale = 1
    max_timescale = 1000000  # 10000

    # Create random input tensors
    q_pt = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_pt = torch.randn(batch_size, num_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    # PyTorch reference implementation
    class DummyConfig:
      """A dummy config class to hold RoPE parameters for the reference implementation."""

      def __init__(self, rope_theta, head_dim, max_position_embeddings):
        """Initializes the dummy configuration.

        Args:
          rope_theta: The base for the rotary frequency.
          head_dim: The dimension of each attention head.
          max_position_embeddings: The maximum sequence length.
        """
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = {"factor": 8.0, "rope_type": "linear"}

    config = DummyConfig(rope_theta=max_timescale, head_dim=head_dim, max_position_embeddings=seq_len)

    pt_rope = Gemma3RotaryEmbedding(config)
    cos_pt, sin_pt = pt_rope(q_pt, position_ids)
    q_rope_pt, k_rope_pt = apply_rotary_pos_emb(q_pt, k_pt, cos_pt, sin_pt, position_ids)

    # JAX implementation
    jax_rope = embeddings.RotaryEmbedding(
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      mesh=self.mesh,
      embedding_dims=head_dim,
      cast_as_fprop_dtype=False,
      fprop_dtype=jnp.float32,
      rope_linear_scaling_factor=8.0,
    )

    # JAX expects [B, S, N, H]
    q_jax = to_jax(q_pt.permute(0, 2, 1, 3))
    k_jax = to_jax(k_pt.permute(0, 2, 1, 3))
    position_jax = to_jax(position_ids)

    # Apply JAX rotary embedding
    q_rope_jax = jax_rope(q_jax, position=position_jax)
    k_rope_jax = jax_rope(k_jax, position=position_jax)

    # Compare outputs
    np.testing.assert_allclose(to_jax(q_rope_pt.permute(0, 2, 1, 3)), q_rope_jax, rtol=1e-3, atol=0.05)
    np.testing.assert_allclose(to_jax(k_rope_pt.permute(0, 2, 1, 3)), k_rope_jax, rtol=1e-3, atol=0.05)


if __name__ == "__main__":
  unittest.main()
