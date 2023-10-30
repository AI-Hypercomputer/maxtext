"""
 Copyright 2023 Google LLC

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

""" Tests for Llama """
import jax
import unittest
import jax.numpy as jnp
from typing import Tuple
import layers
import numpy as np

jax.config.update('jax_platform_name', 'cpu')

  # A few jax_llama implementation from https://github.com/Sea-Snell/
def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
  """Calculate the frequencies"""
  freqs = 1.0 / (
      theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim)
  )
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """ Apply the computed Rotary Postional Embedding"""
  reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
  reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

  xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
  xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

  # add head dim
  freqs_cis = jnp.reshape(
      freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:])
  )

  xq_out = xq_ * freqs_cis
  xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(
      *xq_out.shape[:-1], -1
  )

  xk_out = xk_ * freqs_cis
  xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(
      *xk_out.shape[:-1], -1
  )

  return xq_out.astype(dtype), xk_out.astype(dtype)

class LlamaRoPETest(unittest.TestCase):
  """Test for the RoPE implementation """
  def test_llama_rope(self):
    dim_per_head = 128
    seq_len = 8

    # Run the two implementations on some random query and key
    x_q = np.random.normal(1, 0.5, (1, seq_len, 4, dim_per_head))
    x_k = np.random.normal(1, 0.5, (1, seq_len, 4, dim_per_head))

    # Calculate RoPE embeddings from Sea-Snell implementation
    freqs_cis = precompute_freqs_cis(dim_per_head, seq_len * 2)
    freqs_cis = jnp.take(
        freqs_cis, jnp.arange(seq_len, dtype=np.int32)[None, :], axis=0
    )

    llama_output = apply_rotary_emb(
    jnp.asarray(x_q), jnp.asarray(x_k), freqs_cis
    )

    # Calculate RoPE embeddings from MaxText implementation
    query_proj = layers.LLaMARotaryEmbedding(embedding_dims = dim_per_head)(x_q)
    key_proj = layers.LLaMARotaryEmbedding(embedding_dims = dim_per_head)(x_k)

    # Compare results
    self.assertTrue(jax.numpy.allclose(llama_output[0], query_proj, rtol=1e-05, atol=1e-05, equal_nan=False))
    self.assertTrue(jax.numpy.allclose(llama_output[1], key_proj, rtol=1e-05, atol=1e-05, equal_nan=False))

if __name__ == '__main__':
  unittest.main()

