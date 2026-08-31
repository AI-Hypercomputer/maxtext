# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for DeepSeek-V4 CSA StreamIndex Pallas TPU score kernel."""

import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs.pyconfig import initialize
from maxtext.kernels.attention import csa_streamindex
from maxtext.layers.attention_compressed import DeepseekV4Indexer
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from tests.utils.test_helpers import get_test_config_path


class TestCsaStreamIndexScoreKernel(unittest.TestCase):
  """Unit tests for Pallas CSA StreamIndex score kernel."""

  def setUp(self):
    self.key = jax.random.PRNGKey(42)

  def test_head_major_parity_exact_multiple(self):
    """Verifies numerical parity for head-major input matching reference einsum."""
    key1, key2, key3 = jax.random.split(self.key, 3)
    b, h, s, w, d = 2, 64, 256, 128, 128
    q = jax.random.normal(key1, (b, h, s, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    softmax_scale = d**-0.5

    expected = csa_streamindex.reference_csa_streamindex_score_head_major(
        q, compressed, weights, softmax_scale=softmax_scale
    )
    actual = csa_streamindex.csa_streamindex_score_head_major(
        q,
        compressed,
        weights,
        softmax_scale=softmax_scale,
        block_q=128,
        block_w=128,
        interpret=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)

  def test_head_major_parity_non_multiples(self):
    """Verifies padding handling when seq_len and compressed_len are not multiples of block sizes."""
    key1, key2, key3 = jax.random.split(self.key, 3)
    b, h, s, w, d = 2, 32, 150, 70, 64
    q = jax.random.normal(key1, (b, h, s, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    softmax_scale = d**-0.5

    expected = csa_streamindex.reference_csa_streamindex_score_head_major(
        q, compressed, weights, softmax_scale=softmax_scale
    )
    actual = csa_streamindex.csa_streamindex_score_head_major(
        q,
        compressed,
        weights,
        softmax_scale=softmax_scale,
        block_q=128,
        block_w=128,
        interpret=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

  def test_head_major_causal_parity(self):
    """Verifies numerical parity with in-VMEM causal masking."""
    key1, key2, key3 = jax.random.split(self.key, 3)
    b, h, s, w, d = 1, 4, 256, 64, 64
    q = jax.random.normal(key1, (b, h, s, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    softmax_scale = d**-0.5
    compress_rate = 4

    expected = csa_streamindex.reference_csa_streamindex_score_head_major(
        q, compressed, weights, softmax_scale=softmax_scale, compress_rate=compress_rate
    )
    actual = csa_streamindex.csa_streamindex_score_head_major(
        q,
        compressed,
        weights,
        softmax_scale=softmax_scale,
        compress_rate=compress_rate,
        block_q=128,
        block_w=128,
        interpret=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)

  def test_head_major_gradient_parity(self):
    """Verifies that head-major custom_vjp gradients match reference autograd."""
    b, h, s, w, d = 1, 4, 128, 128, 32
    key1, key2, key3 = jax.random.split(self.key, 3)
    q = jax.random.normal(key1, (b, h, s, d), dtype=jnp.bfloat16)
    comp = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    scale = 32.0**-0.5

    def loss_kernel(q, comp, weights):
      return jnp.sum(
          csa_streamindex.csa_streamindex_score_head_major(
              q, comp, weights, softmax_scale=scale, block_q=128, block_w=128, interpret=True
          )
      )

    def loss_ref(q, comp, weights):
      return jnp.sum(
          csa_streamindex.reference_csa_streamindex_score_head_major(
              q, comp, weights, softmax_scale=scale
          )
      )

    g_q_k, g_c_k, g_w_k = jax.grad(loss_kernel, argnums=(0, 1, 2))(q, comp, weights)
    g_q_r, g_c_r, g_w_r = jax.grad(loss_ref, argnums=(0, 1, 2))(q, comp, weights)

    np.testing.assert_allclose(g_q_k, g_q_r, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(g_c_k, g_c_r, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(g_w_k, g_w_r, rtol=1e-3, atol=1e-3)

  def test_tpu_compile_smoke_production_tiles(self):
    """Compiles and executes with interpret=False on TPU hardware (DeepSeek-V4 production shapes)."""
    if jax.default_backend() != "tpu":
      self.skipTest("TPU hardware required for Mosaic compilation smoke test.")
    b, h, s, d = 1, 64, 4096, 128
    w = s // 4
    scale = d**-0.5
    key1, key2, key3 = jax.random.split(self.key, 3)
    q = jax.random.normal(key1, (b, h, s, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)

    fn = jax.jit(
        lambda q, k, w: csa_streamindex.csa_streamindex_score_head_major(
            q, k, w, softmax_scale=scale, block_q=128, block_w=512, interpret=False
        )
    )
    out = fn(q, compressed, weights).block_until_ready()
    self.assertEqual(out.shape, (b, s, w))


class TestDeepseekv4IndexerIntegration(unittest.TestCase):
  """Integration tests for Deepseekv4Indexer with CSA StreamIndex kernel dispatch."""

  def setUp(self):
    self.mesh = Mesh(jax.devices(), ("data",))
    self.rotary = DeepSeekV4RotaryEmbedding(
        head_dim=64,
        partial_rotary_factor=16.0 / 64.0,
        mesh=self.mesh,
    )

  def _get_config(self, use_csa_streamindex_kernel: bool = False):
    with mock.patch("maxtext.utils.max_utils.maybe_initialize_jax_distributed_system"):
      return initialize(
          [
              None,
              get_test_config_path(),
              "model_name=deepseek4-284b",
              "attention=dot_product",
              "qk_rope_head_dim=16",
              "v_head_dim=16",
              "qk_nope_head_dim=16",
              "indexer_n_heads=16",
              "indexer_head_dim=64",
              "indexer_topk=32",
              "override_model_config=True",
              f"use_csa_streamindex_kernel={use_csa_streamindex_kernel}",
          ]
      )

  def test_indexer_kernel_vs_einsum_output_parity(self):
    """Verifies that Deepseekv4Indexer outputs match whether kernel or einsum is used."""
    config_einsum = self._get_config(use_csa_streamindex_kernel=False)
    config_kernel = self._get_config(use_csa_streamindex_kernel=True)
    b, s, emb_dim, q_lora = 1, 128, config_einsum.emb_dim, config_einsum.q_lora_rank

    indexer_einsum = DeepseekV4Indexer(
        config=config_einsum,
        compress_ratio=4,
        rotary_embedding=self.rotary,
        rngs=nnx.Rngs(0),
    )
    indexer_kernel = DeepseekV4Indexer(
        config=config_kernel,
        compress_ratio=4,
        rotary_embedding=self.rotary,
        rngs=nnx.Rngs(0),
    )

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    hidden = jax.random.normal(key1, (b, s, emb_dim), dtype=jnp.bfloat16)
    q_latent = jax.random.normal(key2, (b, s, q_lora), dtype=jnp.bfloat16)
    pos = jnp.arange(s, dtype=jnp.int32)[None, :]

    out_einsum = indexer_einsum(hidden, q_latent, pos)

    real_kernel_fn = csa_streamindex.csa_streamindex_score_head_major

    def interpret_kernel_fn(*args, **kwargs):
      kwargs["interpret"] = True
      return real_kernel_fn(*args, **kwargs)

    with mock.patch.object(csa_streamindex, "csa_streamindex_score_head_major", side_effect=interpret_kernel_fn):
      out_kernel = indexer_kernel(hidden, q_latent, pos)

    np.testing.assert_array_equal(out_kernel, out_einsum)

  def test_indexer_einsum_when_disabled(self):
    """Verifies that when use_csa_streamindex_kernel=False, einsum path is used."""
    config_einsum = self._get_config(use_csa_streamindex_kernel=False)
    b, s, emb_dim, q_lora = 1, 128, config_einsum.emb_dim, config_einsum.q_lora_rank

    indexer = DeepseekV4Indexer(
        config=config_einsum,
        compress_ratio=4,
        rotary_embedding=self.rotary,
        rngs=nnx.Rngs(0),
    )

    hidden = jnp.ones((b, s, emb_dim), dtype=jnp.bfloat16)
    q_latent = jnp.ones((b, s, q_lora), dtype=jnp.bfloat16)
    pos = jnp.arange(s, dtype=jnp.int32)[None, :]

    with mock.patch.object(csa_streamindex, "csa_streamindex_score_head_major") as mock_kernel:
      out = indexer(hidden, q_latent, pos)
      mock_kernel.assert_not_called()
      self.assertEqual(out.shape, (b, s, min(32, s // 4)))

  def test_jaxpr_verification(self):
    """Verifies that jaxpr contains pallas_call when enabled and dot_general when disabled."""
    q = jnp.zeros((1, 4, 256, 64), dtype=jnp.bfloat16)
    compressed = jnp.zeros((1, 128, 64), dtype=jnp.bfloat16)
    weights = jnp.zeros((1, 256, 4), dtype=jnp.float32)
    scale = 64.0**-0.5

    def compute_scores(q, compressed, weights, use_kernel):
      if use_kernel:
        return csa_streamindex.csa_streamindex_score_head_major(
            q, compressed, weights, softmax_scale=scale, block_q=128, block_w=128
        )
      else:
        return csa_streamindex.reference_csa_streamindex_score_head_major(
            q, compressed, weights, softmax_scale=scale
        )

    jaxpr_kernel = jax.make_jaxpr(compute_scores, static_argnums=(3,))(q, compressed, weights, True)
    self.assertIn("pallas_call", str(jaxpr_kernel))

    jaxpr_einsum = jax.make_jaxpr(compute_scores, static_argnums=(3,))(q, compressed, weights, False)
    self.assertNotIn("pallas_call", str(jaxpr_einsum))
    self.assertIn("dot_general", str(jaxpr_einsum))


if __name__ == "__main__":
  unittest.main()
