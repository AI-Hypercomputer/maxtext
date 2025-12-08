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

""" Tests for Llama. """

import unittest

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from MaxText.layers import embeddings


"""
An example reference jax_llama RoPE implementation from https://github.com/Sea-Snell/
Users should feel free to change and optimize the RoPE implementation in MaxText defined in layers.py
as long as it passes our tests. But they shouldn't change the "reference" implementation in
llama_test.py which is only to be used for comparison purpose.
"""


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
  """Calculate the frequencies."""
  freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
  t = np.arange(end)  # type: ignore
  freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
  sin, cos = np.sin(freqs), np.cos(freqs)
  freqs_cis = np.complex64(cos + 1j * sin)
  return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Apply the computed Rotary Positional Embedding."""
  reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
  reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

  xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
  xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

  # add head dim
  freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

  xq_out = xq_ * freqs_cis
  xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

  xk_out = xk_ * freqs_cis
  xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

  return xq_out.astype(dtype), xk_out.astype(dtype)


def permute_to_match_maxtext_rope(arr):
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


class RoPETest(unittest.TestCase):
  """Test for the RoPE implementation."""

  def setUp(self):
    self.mesh = Mesh(jax.devices(), "data")

  def test_rope(self):
    dim_per_head = 128
    seq_len = 8

    # Run the two implementations on some random query and key
    x_q = np.random.normal(1, 0.5, (1, seq_len, 4, dim_per_head))
    x_k = np.random.normal(1, 0.5, (1, seq_len, 4, dim_per_head))

    # Calculate RoPE embeddings from Sea-Snell implementation
    freqs_cis = precompute_freqs_cis(dim_per_head, seq_len * 2)
    freqs_cis = jnp.take(freqs_cis, jnp.arange(seq_len, dtype=np.int32)[None, :], axis=0)

    llama_output = apply_rotary_emb(jnp.asarray(x_q), jnp.asarray(x_k), freqs_cis)

    position = jnp.arange(seq_len, dtype=jnp.float32)[jnp.newaxis, :]
    rope = embeddings.RotaryEmbedding(min_timescale=1, max_timescale=10_000, embedding_dims=dim_per_head, mesh=self.mesh)
    query_proj = rope(permute_to_match_maxtext_rope(x_q), position)
    key_proj = rope(permute_to_match_maxtext_rope(x_k), position)

    # Compare results
    self.assertTrue(jnp.allclose(permute_to_match_maxtext_rope(llama_output[0]), query_proj, rtol=1e-01, atol=1e-04))
    self.assertTrue(jnp.allclose(permute_to_match_maxtext_rope(llama_output[1]), key_proj, rtol=1e-01, atol=1e-04))

  def test_scaling_rope(self):
    dim_per_head = 128
    seq_len = 8

    # Run the two implementations on some random query and key
    x_q = np.random.normal(1, 0.5, (1, seq_len, 4, dim_per_head))
    position = jnp.arange(seq_len, dtype=jnp.float32)[jnp.newaxis, :]

    # Calculate RoPE embeddings and then scale
    rope = embeddings.RotaryEmbedding(min_timescale=1, max_timescale=10_000, embedding_dims=dim_per_head, mesh=self.mesh)
    query_proj_1 = rope(x_q, position=position)

    query_proj_1 = query_proj_1 * (dim_per_head**-0.5)

    # scale first and then apply RoPE
    query_proj_2 = x_q * (dim_per_head**-0.5)
    query_proj_2 = rope(query_proj_2, position=position)

    self.assertTrue(jax.numpy.allclose(query_proj_2, query_proj_1, rtol=1e-01, atol=1e-04, equal_nan=False))

  def test_llama_rope_with_scaling(self):
    dim_per_head = 128
    seq_len = 8
    x_q = np.random.normal(1, 0.5, (1, seq_len, 4, dim_per_head))
    position = jnp.arange(seq_len, dtype=jnp.float32)[jnp.newaxis, :]

    # Test LLaMARotaryEmbedding with scaling
    llama_rope_scaled = embeddings.LLaMARotaryEmbedding(
        min_timescale=1, max_timescale=10000, embedding_dims=dim_per_head, use_scale=True, mesh=self.mesh
    )
    query_proj_scaled = llama_rope_scaled(x_q, position)

    # Test LLaMARotaryEmbedding without scaling
    llama_rope_no_scale = embeddings.LLaMARotaryEmbedding(
        min_timescale=1, max_timescale=10000, embedding_dims=dim_per_head, use_scale=False, mesh=self.mesh
    )
    query_proj_no_scale = llama_rope_no_scale(x_q, position)

    # Check that the outputs are different
    self.assertFalse(jnp.allclose(query_proj_scaled, query_proj_no_scale, rtol=1e-03, atol=1e-02))

  def test_llama_rope_single_rotation(self):
    dim_per_head = 128
    seq_len = 8
    min_timescale = 1
    max_timescale = 10000
    x_q = np.random.normal(1, 0.5, (1, seq_len, 4, dim_per_head))
    position = jnp.arange(seq_len, dtype=jnp.float32)[jnp.newaxis, :]

    # Use LLaMARotaryEmbedding
    llama_rope = embeddings.LLaMARotaryEmbedding(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        embedding_dims=dim_per_head,
        use_scale=False,
        mesh=self.mesh,
    )
    query_proj = llama_rope(x_q, position)

    # Manual computation of the single rotation
    inputs_shifted_left = jnp.concatenate([x_q[..., 1:], x_q[..., :1]], axis=-1)
    inputs_shifted_right = jnp.concatenate([x_q[..., -1:], x_q[..., :-1]], axis=-1)
    inputs_shifted = jax.lax.select(
        jnp.tile(jnp.mod(jnp.arange(dim_per_head, dtype=jnp.int32), 2), x_q.shape[:-1] + (1,)),
        inputs_shifted_right,
        inputs_shifted_left,
    )

    position = position[:, :, jnp.newaxis, jnp.newaxis]
    half_embedding_dim = dim_per_head // 2
    fraction = 2 * jnp.arange(half_embedding_dim) / dim_per_head
    fraction = jnp.repeat(fraction, 2)
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = position / timescale

    sin_theta = jnp.sin(sinusoid_inp)
    cos_theta = jnp.cos(sinusoid_inp)
    sign = jnp.tile(jnp.array([-1, 1]), half_embedding_dim)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

    expected_proj = x_q * cos_theta + inputs_shifted * sin_theta * sign

    self.assertTrue(jnp.allclose(query_proj, expected_proj, rtol=1e-03, atol=1e-02))


if __name__ == "__main__":
  unittest.main()
