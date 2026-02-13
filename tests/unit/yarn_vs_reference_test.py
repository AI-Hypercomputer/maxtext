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

""" Tests for DeepSeek """
import math
import unittest

import numpy as np

import torch

import jax
import jax.numpy as jnp

from MaxText.layers import embeddings


"""  
DeepSeek v3 PyTorch implementation of yarn rotary positional embedding.
Details https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L294
"""


def precompute_freqs_cis(dim, seqlen, beta_fast=32, beta_slow=1, base=10000.0, factor=40) -> torch.Tensor:
  """
  Precomputes frequency-based complex exponential values for rotary positional embeddings.

  Args:
      args (ModelArgs): Model arguments containing positional embedding parameters.

  Returns:
      torch.Tensor: Precomputed complex exponential values for positional embeddings.
  """
  original_seq_len = 4096

  def find_correction_dim(num_rotations, dim, base, max_seq_len):
    """
    Computes the correction dimension for a given number of rotations in the rotary positional embedding.

    Args:
        num_rotations (float): Number of rotations to compute the correction for.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_seq_len (int): Maximum sequence length.

    Returns:
        float: The correction dimension based on the input parameters.
    """
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

  def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    """
    Computes the range of correction dimensions for rotary positional embeddings.

    Args:
        low_rot (float): Lower bound for the number of rotations.
        high_rot (float): Upper bound for the number of rotations.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_seq_len (int): Maximum sequence length.

    Returns:
        tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
    """
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)

  def linear_ramp_factor(min_ramp, max_ramp, dim_ramp):
    """
    Computes a linear ramp function used to smooth values between a minimum and maximum range.

    Args:
        min_ramp (float): Minimum value for the ramp function.
        max_ramp (float): Maximum value for the ramp function.
        dim_ramp (int): Dimensionality of the ramp tensor.

    Returns:
        torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
            clamped to the range [0, 1].
    """
    if min_ramp == max_ramp:
      max_ramp += 0.001
    linear_func = (torch.arange(dim_ramp, dtype=torch.float32) - min_ramp) / (max_ramp - min_ramp)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

  freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
  if seqlen > original_seq_len:
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth

  t = torch.arange(seqlen)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
  """
  Applies rotary positional embeddings to the input tensor.

  Args:
      x (torch.Tensor): Input tensor with positional embeddings to be applied.
      freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

  Returns:
      torch.Tensor: Tensor with rotary embeddings applied.
  """
  dtype = x.dtype
  x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
  freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
  y = torch.view_as_real(x * freqs_cis).flatten(3)
  return y.to(dtype)


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().numpy())


class YarnRoPETest(unittest.TestCase):
  """Test for the Yarn RoPE implementation."""

  def test_rope_multiple_seq(self):
    hidden_dim = 32
    batch_size = 16
    num_heads = 8

    # Try multiple sequence lengths
    for seqlen in [128, 5000, 10000]:
      with self.subTest(seqlen=seqlen):
        # Precompute the frequencies for RoPE
        freqs_cis = precompute_freqs_cis(hidden_dim, seqlen=seqlen)

        # Create random queries and keys
        q = torch.randn(batch_size, seqlen, num_heads, hidden_dim)
        k = torch.randn(batch_size, seqlen, num_heads, hidden_dim)

        # Generate position ids
        position_ids = torch.arange(seqlen).unsqueeze(0).repeat(batch_size, 1)

        # Apply RoPE using the PyTorch implementation
        q_rope_pt = apply_rotary_emb(q, freqs_cis)
        k_rope_pt = apply_rotary_emb(k, freqs_cis)

        # Create and initialize the JAX Yarn RoPE
        model_jax = embeddings.YarnRotaryEmbedding(hidden_dim, seqlen)
        params = model_jax.init(jax.random.PRNGKey(0), to_jax(k))

        # Apply the JAX RoPE
        q_rope_jax = model_jax.apply(params, to_jax(q), to_jax(position_ids))
        # Infer positions from k.
        k_rope_jax = model_jax.apply(params, to_jax(k))

        # Compare outputs from the PyTorch and JAX implementations
        np.testing.assert_allclose(to_jax(q_rope_pt), q_rope_jax, rtol=1e-3, atol=0.05)
        np.testing.assert_allclose(to_jax(k_rope_pt), k_rope_jax, rtol=1e-3, atol=0.05)


if __name__ == "__main__":
  unittest.main()
