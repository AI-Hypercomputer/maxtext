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

  def test_kernel_vs_einsum_parity_exact_multiple(self):
    """Verifies numerical parity when shapes are exact multiples of block sizes."""
    key1, key2, key3 = jax.random.split(self.key, 3)
    b, s, w, h, d = 2, 256, 128, 64, 128
    q = jax.random.normal(key1, (b, s, h, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    softmax_scale = d**-0.5

    expected = csa_streamindex.reference_csa_streamindex_score(
        q, compressed, weights, softmax_scale=softmax_scale
    )
    actual = csa_streamindex.csa_streamindex_score(
        q,
        compressed,
        weights,
        softmax_scale=softmax_scale,
        block_q=128,
        block_w=128,
        interpret=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)

  def test_kernel_vs_einsum_parity_non_multiples(self):
    """Verifies padding handling when seq_len and compressed_len are not multiples of block_q/block_w."""
    key1, key2, key3 = jax.random.split(self.key, 3)
    b, s, w, h, d = 2, 150, 70, 32, 64
    q = jax.random.normal(key1, (b, s, h, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    softmax_scale = d**-0.5

    expected = csa_streamindex.reference_csa_streamindex_score(
        q, compressed, weights, softmax_scale=softmax_scale
    )
    actual = csa_streamindex.csa_streamindex_score(
        q,
        compressed,
        weights,
        softmax_scale=softmax_scale,
        block_q=128,
        block_w=128,
        interpret=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

  def test_kernel_small_compressed_window(self):
    """Verifies behavior when compressed_len < block_w."""
    key1, key2, key3 = jax.random.split(self.key, 3)
    b, s, w, h, d = 1, 128, 32, 16, 64
    q = jax.random.normal(key1, (b, s, h, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    softmax_scale = d**-0.5

    expected = csa_streamindex.reference_csa_streamindex_score(
        q, compressed, weights, softmax_scale=softmax_scale
    )
    actual = csa_streamindex.csa_streamindex_score(
        q,
        compressed,
        weights,
        softmax_scale=softmax_scale,
        block_q=128,
        block_w=128,
        interpret=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

  def test_tpu_compile_smoke_production_tiles(self):
    """Compiles and executes with interpret=False on TPU hardware (DeepSeek-V4 production shapes)."""
    if jax.default_backend() != "tpu":
      self.skipTest("TPU hardware required for Mosaic compilation smoke test.")
    b, s, h, d = 1, 4096, 64, 128
    w = s // 4
    scale = d**-0.5
    key1, key2, key3 = jax.random.split(self.key, 3)
    q = jax.random.normal(key1, (b, s, h, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)

    fn = jax.jit(
        lambda q, k, w: csa_streamindex.csa_streamindex_score(
            q, k, w, softmax_scale=scale, block_q=128, block_w=512, interpret=False
        )
    )
    out = fn(q, compressed, weights).block_until_ready()
    self.assertEqual(out.shape, (b, s, w))

  def test_vjp_backward_parity(self):
    """Verifies that backward gradients of custom VJP match reference autograd."""
    key1, key2, key3, key4 = jax.random.split(self.key, 4)
    b, s, w, h, d = 2, 256, 128, 32, 64
    scale = d**-0.5
    q = jax.random.normal(key1, (b, s, h, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    cotangent = jax.random.normal(key4, (b, s, w), dtype=jnp.float32)

    def loss_kernel(q, k, w):
      out = csa_streamindex.csa_streamindex_score(
          q, k, w, softmax_scale=scale, block_q=128, block_w=128, interpret=True
      )
      return jnp.sum(out * cotangent)

    def loss_ref(q, k, w):
      out = csa_streamindex.reference_csa_streamindex_score(
          q, k, w, softmax_scale=scale
      )
      return jnp.sum(out * cotangent)

    _, (dq_k, dk_k, dw_k) = jax.value_and_grad(loss_kernel, argnums=(0, 1, 2))(q, compressed, weights)
    _, (dq_r, dk_r, dw_r) = jax.value_and_grad(loss_ref, argnums=(0, 1, 2))(q, compressed, weights)

    np.testing.assert_allclose(dw_k, dw_r, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dq_k.astype(jnp.float32), dq_r.astype(jnp.float32), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dk_k.astype(jnp.float32), dk_r.astype(jnp.float32), rtol=1e-3, atol=1e-3)

  def test_tpu_backward_smoke(self):
    """Verifies that backward pass compiles and runs on TPU hardware."""
    if jax.default_backend() != "tpu":
      self.skipTest("TPU hardware required for backward smoke test.")
    b, s, h, d = 1, 1024, 64, 128
    w = s // 4
    scale = d**-0.5
    key1, key2, key3, key4 = jax.random.split(self.key, 4)
    q = jax.random.normal(key1, (b, s, h, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(key2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(key3, (b, s, h), dtype=jnp.float32)
    cotangent = jax.random.normal(key4, (b, s, w), dtype=jnp.float32)

    @jax.jit
    def grad_fn(q, k, w):
      def loss(q, k, w):
        out = csa_streamindex.csa_streamindex_score(
            q, k, w, softmax_scale=scale, block_q=128, block_w=512, interpret=False
        )
        return jnp.sum(out * cotangent)
      return jax.grad(loss, argnums=(0, 1, 2))(q, k, w)

    dq, dk, dw = grad_fn(q, compressed, weights)
    self.assertEqual(dq.shape, q.shape)
    self.assertEqual(dk.shape, compressed.shape)
    self.assertEqual(dw.shape, weights.shape)


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

    # 1. Forward with use_csa_streamindex_kernel=False
    out_einsum = indexer_einsum(hidden, q_latent, pos)

    # 2. Forward with use_csa_streamindex_kernel=True (intercept to set interpret=True on CPU)
    real_kernel_fn = csa_streamindex.csa_streamindex_score

    def interpret_kernel_fn(*args, **kwargs):
      kwargs["interpret"] = True
      return real_kernel_fn(*args, **kwargs)

    with mock.patch.object(csa_streamindex, "csa_streamindex_score", side_effect=interpret_kernel_fn):
      out_kernel = indexer_kernel(hidden, q_latent, pos)

    np.testing.assert_array_equal(out_kernel, out_einsum)

  def test_ar_decode_fallback(self):
    """Verifies that when seq_len < 128 (e.g. seq_len=64 with windows formed), einsum path is used."""
    config_kernel = self._get_config(use_csa_streamindex_kernel=True)
    b, s, emb_dim, q_lora = 1, 64, config_kernel.emb_dim, config_kernel.q_lora_rank

    indexer = DeepseekV4Indexer(
        config=config_kernel,
        compress_ratio=4,
        rotary_embedding=self.rotary,
        rngs=nnx.Rngs(0),
    )

    hidden = jnp.ones((b, s, emb_dim), dtype=jnp.bfloat16)
    q_latent = jnp.ones((b, s, q_lora), dtype=jnp.bfloat16)
    pos = jnp.arange(s, dtype=jnp.int32)[None, :]

    with mock.patch.object(csa_streamindex, "csa_streamindex_score") as mock_kernel:
      out = indexer(hidden, q_latent, pos)
      mock_kernel.assert_not_called()
      self.assertEqual(out.shape, (b, s, min(32, s // 4)))

  def test_jaxpr_verification(self):
    """Verifies that jaxpr contains pallas_call when enabled and dot_general when disabled."""
    q = jnp.zeros((1, 256, 16, 64), dtype=jnp.bfloat16)
    compressed = jnp.zeros((1, 128, 64), dtype=jnp.bfloat16)
    weights = jnp.zeros((1, 256, 16), dtype=jnp.float32)
    scale = 64.0**-0.5

    def compute_scores(q, compressed, weights, use_kernel):
      if use_kernel:
        return csa_streamindex.csa_streamindex_score(
            q, compressed, weights, softmax_scale=scale, block_q=128, block_w=128
        )
      else:
        return csa_streamindex.reference_csa_streamindex_score(
            q, compressed, weights, softmax_scale=scale
        )

    # Kernel enabled trace
    jaxpr_kernel = jax.make_jaxpr(compute_scores, static_argnums=(3,))(q, compressed, weights, True)
    jaxpr_kernel_str = str(jaxpr_kernel)
    self.assertIn("pallas_call", jaxpr_kernel_str)

    # Kernel disabled trace
    jaxpr_einsum = jax.make_jaxpr(compute_scores, static_argnums=(3,))(q, compressed, weights, False)
    jaxpr_einsum_str = str(jaxpr_einsum)
    self.assertNotIn("pallas_call", jaxpr_einsum_str)
    self.assertIn("dot_general", jaxpr_einsum_str)


if __name__ == "__main__":
  unittest.main()
