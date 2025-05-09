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

""" Tests for Llama4 Vision RoPE """
import math
import torch
from torch import nn
import jax
import unittest
import jax.numpy as jnp
from MaxText.layers import embeddings
import numpy as np


"""  
Llama4 Vision RoPE 
Details https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
"""


### original Pytorch Reference implementation
def reshape_for_broadcast(freqs_ci: torch.Tensor, query: torch.Tensor):
  """Reshape the frequency tensor for broadcasting."""
  ndim = query.ndim
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query.shape)]
  return freqs_ci.view(*shape)


def vision_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_ci: torch.Tensor,
):
  """Apply the rotary embedding to the query and key tensors."""
  query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
  key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
  freqs_ci = reshape_for_broadcast(freqs_ci=freqs_ci, query=query_)  # freqs_ci[:,:,None,:]
  freqs_ci = freqs_ci.to(query_.device)
  query_out = torch.view_as_real(query_ * freqs_ci).flatten(3)
  key_out = torch.view_as_real(key_ * freqs_ci).flatten(3)
  return query_out.type_as(query), key_out.type_as(key)  # but this drops to 8e-3


class Llama4VisionRotaryEmbedding(nn.Module):
  """Llama4 Vision RoPE implementation."""

  def __init__(self, image_size, patch_size, hidden_size, num_attention_heads, rope_theta):
    super().__init__()
    idx = image_size // patch_size
    img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
    img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
    img_idx[-1, -1] = -2  # ID_CLS_TOKEN
    frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
    frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
    freq_dim = hidden_size // num_attention_heads // 2
    rope_freq = 1.0 / (rope_theta ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim))
    freqs_x = ((frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
    freqs_y = ((frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
    freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
    freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
    freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
    self.freqs_ci = freq_cis  # idx**2, idx**2, idx * 2

  def forward(self):
    return self.freqs_ci


### original Pytorch Reference implementation


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().numpy())


class Llama4VisionRotaryEmbeddingTest(unittest.TestCase):
  """Test for the Llama4 Vision RoPE implementation."""

  def test_rope_multiple_seq(self):
    image_size = 336
    patch_size = 14
    hidden_size = 1408
    num_attention_heads = 16
    rope_theta = 10000
    seq_len = (image_size // patch_size) ** 2 + 1

    for batch_size in [10, 100, 1000]:
      with self.subTest(batch_size=batch_size):
        freqs_ci = Llama4VisionRotaryEmbedding(image_size, patch_size, hidden_size, num_attention_heads, rope_theta)
        freqs_ci = freqs_ci.forward()
        # Create random queries and keys
        q = torch.randn(batch_size, seq_len, num_attention_heads, hidden_size // num_attention_heads)
        k = torch.randn(batch_size, seq_len, num_attention_heads, hidden_size // num_attention_heads)

        q_rope_pt, k_rope_pt = vision_apply_rotary_emb(q, k, freqs_ci=freqs_ci)

        # # Create and initialize the JAX Llama4 Vision RoPE
        model_jax = embeddings.LlamaVisionRotaryEmbedding(
            image_size, patch_size, hidden_size, num_attention_heads, rope_theta
        )
        params = model_jax.init(jax.random.PRNGKey(0), to_jax(k))

        # Apply the JAX RoPE
        q_rope_jax = model_jax.apply(params, to_jax(q))
        k_rope_jax = model_jax.apply(params, to_jax(k))

        # Compare outputs from the PyTorch and JAX implementations
        np.testing.assert_allclose(to_jax(q_rope_pt), q_rope_jax, rtol=1e-3, atol=0.05)
        np.testing.assert_allclose(to_jax(k_rope_pt), k_rope_jax, rtol=1e-3, atol=0.05)


if __name__ == "__main__":
  unittest.main()
