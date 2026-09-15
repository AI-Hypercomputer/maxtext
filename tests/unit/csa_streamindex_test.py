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
import pytest

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs.pyconfig import initialize
from maxtext.kernels.attention import csa_streamindex
from maxtext.layers.attention_compressed import DeepseekV4Indexer
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from tests.utils.test_helpers import get_test_config_path


class TestCsaStreamIndexScoreKernel(unittest.TestCase):
  """Unit tests for Pallas CSA StreamIndex score kernel."""

  def setUp(self):
    self.key = jax.random.PRNGKey(42)

  def _inputs(self, b, h, s, w, d, key=None):
    k1, k2, k3 = jax.random.split(key if key is not None else self.key, 3)
    q = jax.random.normal(k1, (b, h, s, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(k2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(k3, (b, s, h), dtype=jnp.float32)
    return q, compressed, weights

  def test_parity_exact_multiple(self):
    """Parity with the reference on exact block multiples.

    The reference is unmasked, so this also pins that the kernel does no
    in-kernel masking of its own.
    """
    b, h, s, w, d = 2, 64, 256, 128, 128
    q, compressed, weights = self._inputs(b, h, s, w, d)
    softmax_scale = d**-0.5

    expected = csa_streamindex.csa_indexer_scores_jax(q, compressed, weights, softmax_scale=softmax_scale)
    actual = csa_streamindex.csa_streamindex_score(q, compressed, weights, softmax_scale, 128, 128, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)
    self.assertTrue(bool(jnp.all(jnp.isfinite(actual))), "no entry may be a masked sentinel")

  def test_parity_non_multiples(self):
    """Verifies padding handling when seq_len and compressed_len are not multiples of block sizes."""
    b, h, s, w, d = 2, 32, 150, 70, 64
    q, compressed, weights = self._inputs(b, h, s, w, d)
    softmax_scale = d**-0.5

    expected = csa_streamindex.csa_indexer_scores_jax(q, compressed, weights, softmax_scale=softmax_scale)
    actual = csa_streamindex.csa_streamindex_score(q, compressed, weights, softmax_scale, 128, 128, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

  def test_parity_seq_len_smaller_than_block_q(self):
    """Verifies correctness when seq_len < block_q and compressed_len < block_w.

    The grid must still produce exactly one block per axis, with the padded
    region sliced back off.
    """
    b, h, d = 2, 16, 64
    softmax_scale = d**-0.5

    for s, w in ((1, 1), (7, 3), (64, 16), (127, 63)):
      with self.subTest(seq_len=s, compressed_len=w):
        q, compressed, weights = self._inputs(b, h, s, w, d)
        expected = csa_streamindex.csa_indexer_scores_jax(q, compressed, weights, softmax_scale=softmax_scale)
        actual = csa_streamindex.csa_streamindex_score(q, compressed, weights, softmax_scale, 128, 128, True)
        self.assertEqual(actual.shape, (b, s, w))
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

  def test_gradient_parity(self):
    """Verifies that custom_vjp gradients match reference autograd.

    Inputs are float32 deliberately. Gradients are cast back to the input
    dtype, so with bfloat16 inputs this would compare bf16 arrays whose
    spacing (~4e-3 relative) is coarser than any tolerance worth asserting,
    and it would measure rounding rather than the VJP.
    """
    b, h, s, w, d = 1, 4, 128, 128, 32
    k1, k2, k3 = jax.random.split(self.key, 3)
    q = jax.random.normal(k1, (b, h, s, d), dtype=jnp.float32)
    comp = jax.random.normal(k2, (b, w, d), dtype=jnp.float32)
    weights = jax.random.normal(k3, (b, s, h), dtype=jnp.float32)
    scale = float(d) ** -0.5

    def loss_kernel(q, comp, weights):
      return jnp.sum(csa_streamindex.csa_streamindex_score(q, comp, weights, scale, 128, 128, True))

    def loss_ref(q, comp, weights):
      return jnp.sum(csa_streamindex.csa_indexer_scores_jax(q, comp, weights, softmax_scale=scale))

    g_q_k, g_c_k, g_w_k = jax.grad(loss_kernel, argnums=(0, 1, 2))(q, comp, weights)
    g_q_r, g_c_r, g_w_r = jax.grad(loss_ref, argnums=(0, 1, 2))(q, comp, weights)

    np.testing.assert_allclose(g_q_k, g_q_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_c_k, g_c_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_w_k, g_w_r, rtol=1e-5, atol=1e-5)

  def test_head_chunked_backward_matches_unchunked(self):
    """Head-chunked backward must equal whole-head autodiff for every chunk size.

    The chunk/un-chunk reshapes permute head and batch axes; an error there
    would scramble per-head gradients while keeping shapes valid.
    """
    b, h, s, w, d = 2, 8, 64, 32, 16
    scale = float(d) ** -0.5
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(3), 4)
    q = jax.random.normal(k1, (b, h, s, d), dtype=jnp.float32)
    comp = jax.random.normal(k2, (b, w, d), dtype=jnp.float32)
    weights = jax.random.normal(k3, (b, s, h), dtype=jnp.float32)
    g = jax.random.normal(k4, (b, s, w), dtype=jnp.float32)

    _, vjp_ref = jax.vjp(
        lambda a, c, e: csa_streamindex.csa_indexer_scores_jax(a, c, e, softmax_scale=scale),
        q,
        comp,
        weights,
    )
    dq_ref, dc_ref, dw_ref = vjp_ref(g)

    # Includes 3, a non-divisor of 8, which must fall back to whole-head.
    for chunk in (1, 2, 3, 4, 8):
      with self.subTest(head_chunk_size=chunk):
        # pylint: disable-next=protected-access
        dq, dc, dw = csa_streamindex._csa_streamindex_score_bwd(scale, None, None, True, chunk, (q, comp, weights), g)
        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dc.shape, comp.shape)
        self.assertEqual(dw.shape, weights.shape)
        np.testing.assert_allclose(dq, dq_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(dc, dc_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(dw, dw_ref, rtol=1e-4, atol=1e-4)

  def test_defaults_produce_correct_results(self):
    """Passing block_q/block_w=None must resolve to the defaults and stay correct."""
    b, h, d = 1, 32, 64
    s, w = 256, 128
    scale = d**-0.5
    q, compressed, weights = self._inputs(b, h, s, w, d)

    expected = csa_streamindex.csa_indexer_scores_jax(q, compressed, weights, softmax_scale=scale)
    actual = csa_streamindex.csa_streamindex_score(q, compressed, weights, scale, None, None, True)
    self.assertEqual(actual.shape, (b, s, w))
    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)


@pytest.mark.tpu_only
class TestCsaStreamIndexOnTpu(unittest.TestCase):
  """Real-hardware tests (`interpret=False`), exercising the Mosaic lowering.

  `interpret=True` runs the kernel body in Python and never invokes Mosaic, so
  it cannot validate `dimension_semantics`, tile-size VMEM limits, or the actual
  `tpu_custom_call` lowering. These tests do.
  """

  def setUp(self):
    self.key = jax.random.PRNGKey(11)

  def _inputs(self, b, h, s, w, d):
    k1, k2, k3 = jax.random.split(self.key, 3)
    q = jax.random.normal(k1, (b, h, s, d), dtype=jnp.bfloat16)
    compressed = jax.random.normal(k2, (b, w, d), dtype=jnp.bfloat16)
    weights = jax.random.normal(k3, (b, s, h), dtype=jnp.float32)
    return q, compressed, weights

  def test_parallel_dimension_semantics_accepted(self):
    """All-`parallel` dimension_semantics must lower cleanly.

    Each (b, i, j) program writes a disjoint output block with no accumulation,
    so `W` was changed from `arbitrary` to `parallel`. Mosaic rejects invalid
    parallel claims, so a successful compile plus correct values is the check.
    """
    b, h, s, w, d = 2, 16, 256, 512, 64
    scale = d**-0.5
    q, compressed, weights = self._inputs(b, h, s, w, d)

    expected = csa_streamindex.csa_indexer_scores_jax(q, compressed, weights, softmax_scale=scale)
    actual = jax.jit(lambda a, c, e: csa_streamindex.csa_streamindex_score(a, c, e, scale, 128, 256, False))(
        q, compressed, weights
    ).block_until_ready()
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)

  def test_shard_map_multi_device_parity(self):
    """Batch-sharded `shard_map` must match the reference and emit no all-gather."""
    n = jax.device_count()
    if n < 2:
      self.skipTest("Requires >= 2 TPU devices.")

    mesh = Mesh(np.array(jax.devices()), ("data",))
    b, h, s, w, d = n, 16, 256, 128, 64
    scale = float(d) ** -0.5
    q, compressed, weights = self._inputs(b, h, s, w, d)

    pspec_q = jax.sharding.PartitionSpec("data", None, None, None)
    pspec_3d = jax.sharding.PartitionSpec("data", None, None)
    q_sh = jax.sharding.NamedSharding(mesh, pspec_q)
    c_sh = jax.sharding.NamedSharding(mesh, pspec_3d)
    o_sh = jax.sharding.NamedSharding(mesh, pspec_3d)

    q = jax.device_put(q, q_sh)
    compressed = jax.device_put(compressed, c_sh)
    weights = jax.device_put(weights, c_sh)

    def sharded(a, c, e):
      @jax.shard_map(
          mesh=mesh,
          in_specs=(pspec_q, pspec_3d, pspec_3d),
          out_specs=pspec_3d,
          check_vma=False,
      )
      def inner(la, lc, le):
        return csa_streamindex.csa_streamindex_score(la, lc, le, scale, 128, 128, False)

      return inner(a, c, e)

    jit_sharded = jax.jit(sharded, in_shardings=(q_sh, c_sh, c_sh), out_shardings=o_sh)
    hlo = jit_sharded.lower(q, compressed, weights).compile().as_text()
    self.assertEqual(hlo.count("all-gather"), 0, "shard_map must not introduce all-gather")

    actual = jit_sharded(q, compressed, weights).block_until_ready()
    expected = csa_streamindex.csa_indexer_scores_jax(q, compressed, weights, softmax_scale=scale)
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


class TestDeepseekv4IndexerIntegration(unittest.TestCase):
  """Integration tests for Deepseekv4Indexer with CSA StreamIndex kernel dispatch."""

  def setUp(self):
    # Single-device mesh: these tests exercise dispatch routing with batch=1,
    # which cannot be split across a multi-device batch axis. Multi-device
    # sharding behavior is covered by TestCsaStreamIndexSharding.
    self.mesh = Mesh(jax.devices()[:1], ("data",))
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

  def _build_indexer(self, config):
    return DeepseekV4Indexer(
        config=config,
        compress_ratio=4,
        rotary_embedding=self.rotary,
        rngs=nnx.Rngs(0),
        mesh=self.mesh,
    )

  def test_indexer_kernel_vs_einsum_output_parity(self):
    """Verifies that Deepseekv4Indexer outputs match whether kernel or einsum is used."""
    config_einsum = self._get_config(use_csa_streamindex_kernel=False)
    config_kernel = self._get_config(use_csa_streamindex_kernel=True)
    b, s, emb_dim, q_lora = 1, 128, config_einsum.emb_dim, config_einsum.q_lora_rank

    indexer_einsum = self._build_indexer(config_einsum)
    indexer_kernel = self._build_indexer(config_kernel)

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    hidden = jax.random.normal(key1, (b, s, emb_dim), dtype=jnp.bfloat16)
    q_latent = jax.random.normal(key2, (b, s, q_lora), dtype=jnp.bfloat16)
    pos = jnp.arange(s, dtype=jnp.int32)[None, :]

    idx_einsum, scores_einsum = indexer_einsum(hidden, q_latent, pos, return_scores=True)

    real_kernel_fn = csa_streamindex.csa_streamindex_score

    def interpret_kernel_fn(q, compressed, weights, softmax_scale, *_args):
      return real_kernel_fn(q, compressed, weights, softmax_scale, None, None, True)

    with mock.patch.object(csa_streamindex, "csa_streamindex_score", side_effect=interpret_kernel_fn):
      idx_kernel, scores_kernel = indexer_kernel(hidden, q_latent, pos, return_scores=True)

    # Indices must match exactly -- they decide which blocks attention reads.
    # Scores only need to agree numerically; Mosaic and the einsum accumulate
    # in a different order, which shows up a few ULP apart in float32.
    np.testing.assert_array_equal(idx_kernel, idx_einsum)
    np.testing.assert_allclose(scores_kernel, scores_einsum, rtol=1e-4, atol=1e-5)

  def test_kernel_dispatch_conditions(self):
    """The kernel runs only when the flag is on and the model is training."""
    cases = (
        ("flag off, train", False, MODEL_MODE_TRAIN, False),
        ("flag on, prefill", True, MODEL_MODE_PREFILL, False),
        ("flag on, train", True, MODEL_MODE_TRAIN, True),
    )
    real_kernel_fn = csa_streamindex.csa_streamindex_score

    def interpret_kernel_fn(q, compressed, weights, softmax_scale, *_args):
      return real_kernel_fn(q, compressed, weights, softmax_scale, None, None, True)

    for name, flag, mode, expect_kernel in cases:
      with self.subTest(name):
        config = self._get_config(use_csa_streamindex_kernel=flag)
        b, s = 1, 128
        indexer = self._build_indexer(config)
        hidden = jnp.ones((b, s, config.emb_dim), dtype=jnp.bfloat16)
        q_latent = jnp.ones((b, s, config.q_lora_rank), dtype=jnp.bfloat16)
        pos = jnp.arange(s, dtype=jnp.int32)[None, :]

        with mock.patch.object(csa_streamindex, "csa_streamindex_score", side_effect=interpret_kernel_fn) as mock_kernel:
          indices, _ = indexer(hidden, q_latent, pos, model_mode=mode)

        self.assertEqual(mock_kernel.called, expect_kernel)
        self.assertEqual(indices.shape, (b, s, min(32, s // 4)))


if __name__ == "__main__":
  unittest.main()
