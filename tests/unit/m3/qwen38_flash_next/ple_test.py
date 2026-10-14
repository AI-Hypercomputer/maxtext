# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for SplitMix64 n-gram hashing and Layer-2 PLE module."""

from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
from maxtext.m3.models.qwen38_flash_next import ple  # pylint: disable=no-name-in-module
import numpy as np
import pytest

torch: Any
configuration_qwen4_exp: Any
modeling_qwen4_exp: Any

try:
  import torch  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import configuration_qwen4_exp  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import modeling_qwen4_exp  # pylint: disable=g-import-not-at-top

  _HAS_QWEN4_EXP = True
except ImportError:
  torch = None
  configuration_qwen4_exp = None
  modeling_qwen4_exp = None
  _HAS_QWEN4_EXP = False


@pytest.mark.cpu_only
class Qwen38PLETest(absltest.TestCase):

  def test_splitmix64_known_vectors(self):
    multipliers, moduli, table_offsets, padded_vocab_size = ple.compute_ple_buffers(
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        seed=1234,
    )
    expected_multipliers = [23703573157769, 20109073645365, 8052911324071]
    expected_moduli = [
        int(x)
        for x in (
            "20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,"
            "20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171"
        ).split(",")
    ]
    expected_offsets = [
        int(x)
        for x in (
            "0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,"
            "160000374,180000455,200000548,220000655,240000802,260000955,280001114,"
            "300001275"
        ).split(",")
    ]
    self.assertEqual(multipliers, expected_multipliers)
    self.assertEqual(moduli, expected_moduli)
    self.assertEqual(table_offsets, expected_offsets)
    self.assertEqual(padded_vocab_size, 320_001_536)
    self.assertEqual(2560 // 16, 160)
    self.assertTrue(ple._is_prime(20000003))  # pylint: disable=protected-access
    self.assertFalse(ple._is_prime(20000001))  # pylint: disable=protected-access

  def test_ple_parameter_tree_contract(self):
    mod = ple.Qwen38PLELayer(
        eos_token_id=3,
        emb_dim=32,
        vocab_size=1000,
        ple_embed_dim=64,
        heads_per_ngram=2,
        ngram_vocab_size_base=100,
        make_ngram_vocab_size_divisible_by=16,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )
    self.assertIsNotNone(mod.ple_embedding.ngram_embedding)
    self.assertEqual(mod.ple_embedding.ngram_embedding.embedding.value.shape, (432, 16))
    self.assertEqual(mod.key_proj.kernel.value.shape, (64, 128))
    self.assertEqual(mod.value_proj.kernel.value.shape, (64, 32))
    self.assertEqual(mod.norm_key.scale.value.shape, (128,))
    self.assertEqual(mod.conv1d.kernel.value.shape, (4, 1, 128))
    np.testing.assert_array_equal(mod.conv1d.kernel.value, 0.0)

    # Ensure static integer buffers are NOT in nnx.Param state
    params = nnx.state(mod, nnx.Param)
    self.assertIn("embedding", params["ple_embedding"]["ngram_embedding"])
    self.assertNotIn("layer_multipliers", params["ple_embedding"])
    self.assertNotIn("head_vocab_sizes", params["ple_embedding"])

  def test_r2_uint32_invariants_raise_on_overflow(self):
    with self.assertRaises(ValueError):
      ple.Qwen38NGramEmbedding(eos_token_id=3, ngram_vocab_size_base=(1 << 31) + 1, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      ple.Qwen38NGramEmbedding(eos_token_id=3, ngram_vocab_size_base=(1 << 28) + 1, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      ple.Qwen38NGramEmbedding(eos_token_id=1 << 24, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      ple.Qwen38NGramEmbedding(eos_token_id=3, vocab_size=(1 << 24) + 1, rngs=nnx.Rngs(0))

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_ple_weight_copy_parity(self):
    self.assertIsNotNone(torch)
    self.assertIsNotNone(modeling_qwen4_exp)
    self.assertIsNotNone(configuration_qwen4_exp)
    cfg_args = {
        "vocab_size": 1024,
        "ple_embed_dim": 32,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 1000,
        "make_ngram_vocab_size_divisible_by": 4,
        "eos_token_id": 3,
    }
    hf_cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(hidden_size=16, hc_count=4, **cfg_args)
    hf_ple = modeling_qwen4_exp.Qwen4ExpTextPLELayer(hf_cfg, layer_idx=1, ple_layer_index=0)
    jax_ple = ple.Qwen38PLELayer(emb_dim=16, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(0), **cfg_args)

    # 1. Assert buffer arrays match 100% identically
    emb_j, emb_h = jax_ple.ple_embedding, hf_ple.ple_embedding
    self.assertEqual(list(emb_j.layer_multipliers), emb_h.layer_multipliers.tolist())
    self.assertEqual(list(emb_j.head_vocab_sizes), emb_h.ngram_heads_vocab_sizes.tolist())
    self.assertEqual(list(emb_j.head_offsets), emb_h.ngram_heads_offsets.tolist())
    self.assertEqual(emb_j.padded_vocab_size, emb_h.ngram_embedding.num_embeddings)

    # 2. Copy weights from PyTorch to JAX
    for j_w, p_w, tr in [
        (emb_j.ngram_embedding.embedding, emb_h.ngram_embedding.weight, None),
        (jax_ple.conv1d.kernel, hf_ple.conv1d.weight, (2, 1, 0)),
        (jax_ple.norm_key.scale, hf_ple.norm_key.weight, None),
        (jax_ple.norm_query.scale, hf_ple.norm_query.weight, None),
        (jax_ple.norm_conv.scale, hf_ple.norm_conv.weight, None),
        (jax_ple.key_proj.kernel, hf_ple.key_proj.weight, (1, 0)),
        (jax_ple.value_proj.kernel, hf_ple.value_proj.weight, (1, 0)),
    ]:
      arr = p_w.detach().cpu().numpy()
      j_w.value = jnp.array(arr.transpose(tr) if tr else arr)

    # 3. Test on batch with eos_token_id=3 in the middle of sequences
    input_ids = np.array([[10, 20, 3, 40, 50, 60, 3, 70], [5, 3, 15, 25, 35, 45, 55, 65]], dtype=np.int64)
    rng = np.random.default_rng(42)
    hidden_states = rng.standard_normal((2, 8, 4 * 16)).astype(np.float32)

    pt_emb = hf_ple.ple_embedding(torch.from_numpy(input_ids), past_key_values=None).detach().cpu().numpy()
    jax_emb = np.array(jax_ple.ple_embedding(jnp.array(input_ids)))
    np.testing.assert_allclose(jax_emb, pt_emb, atol=1e-5, rtol=1e-5)

    pt_out = hf_ple(torch.from_numpy(hidden_states), torch.from_numpy(input_ids), None).detach().cpu().numpy()
    jax_out = np.array(jax_ple(jnp.array(hidden_states), jnp.array(input_ids)))
    np.testing.assert_allclose(jax_out, pt_out, atol=1e-5, rtol=1e-5)

  def test_rank_handling_and_conv_mask(self):
    b, s, d, hc = 2, 4, 16, 4
    mod = ple.Qwen38PLELayer(
        eos_token_id=3,
        emb_dim=d,
        vocab_size=1000,
        ple_embed_dim=32,
        heads_per_ngram=2,
        ngram_vocab_size_base=100,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    mod.conv1d.kernel.value = jnp.ones_like(mod.conv1d.kernel.value)
    input_ids = jnp.zeros((b, s), dtype=jnp.int32)
    h_4d = jnp.ones((b, s, hc, d))
    h_3d = jnp.ones((b, s, hc * d))

    self.assertEqual(mod(h_4d, input_ids).shape, (b, s, hc, d))
    self.assertEqual(mod(h_3d, input_ids).shape, (b, s, hc * d))

    mask = jnp.array([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    out_masked = mod(h_3d, input_ids, conv_mask=mask)
    np.testing.assert_allclose(out_masked[0, 0], 0.0, atol=1e-6)
    self.assertFalse(np.allclose(out_masked[1, 0], 0.0, atol=1e-3))


if __name__ == "__main__":
  absltest.main()
