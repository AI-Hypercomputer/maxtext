# Copyright 2023–2026 Google LLC
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

"""Tests for zero_gen_logits_checker."""

import os
from absl.testing import absltest
from absl.testing import parameterized
from tests.utils import zero_gen_logits_checker as checker
import numpy as np
import pytest


@pytest.mark.cpu_only
class ZeroGenLogitsCheckerTest(parameterized.TestCase):
  """Concise test suite for zero_gen_logits_checker core metrics and I/O."""

  def test_routing_and_indexer_block_parity(self):
    routing_jax = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    res_pass = checker.check_routing_parity(routing_jax, routing_jax.copy())
    self.assertTrue(res_pass["passed"])
    self.assertEqual(res_pass["num_mismatches"], 0)
    routing_hf = routing_jax.copy()
    routing_hf[1, 0, 1] = 99
    res_fail = checker.check_routing_parity(routing_jax, routing_hf)
    self.assertFalse(res_fail["passed"])
    self.assertEqual(res_fail["num_mismatches"], 1)

    jax_blocks = np.array([[[10, 20, 30], [40, 50, 60]]])
    hf_permuted = np.array([[[30, 10, 20], [60, 50, 40]]])
    self.assertTrue(checker.check_indexer_block_parity(jax_blocks, hf_permuted)["passed"])
    hf_mismatch = hf_permuted.copy()
    hf_mismatch[0, 1, 0] = 999
    self.assertFalse(checker.check_indexer_block_parity(jax_blocks, hf_mismatch)["passed"])
    with self.assertRaises(ValueError):
      checker.check_indexer_block_parity(np.array([1, 2]), np.array([1, 2]))

  def test_logits_parity_and_dynamic_threshold(self):
    self.assertEqual(checker.get_dynamic_threshold(0.01, 512), 0.01)
    self.assertEqual(checker.get_dynamic_threshold(0.01, 2048), 0.02)
    self.assertEqual(checker.get_dynamic_threshold(0.01, 8192), 0.04)

    np.random.seed(0)
    logits = np.random.randn(2, 512, 32)
    self.assertTrue(checker.check_logits_parity(logits, logits.copy(), seq_len=512)["passed"])

    corrupt = logits.copy()
    corrupt[:, 200:, :] += 5.0
    mask = np.zeros((2, 512), dtype=bool)
    mask[:, :100] = True
    self.assertTrue(checker.check_logits_parity(logits, corrupt, mask=mask, seq_len=512)["passed"])

    logits_a = np.ones((1, 512, 10), dtype=np.float64)
    logits_b = logits_a.copy()
    logits_b[0, 0, :] = -np.ones(10, dtype=np.float64)
    res_min_fail = checker.check_logits_parity(logits_a, logits_b, seq_len=512)
    self.assertFalse(res_min_fail["passed"])
    self.assertLess(res_min_fail["cosine_sim_min"], res_min_fail["cos_tol"])

    with self.assertRaises(ValueError):
      checker.check_logits_parity(np.zeros((2, 10)), np.zeros((2, 12)))

  def test_masked_ce_loss_and_tinymmlu_delta(self):
    tokens = np.array([[1, 3, 5, 7, 2, 4]], dtype=np.int64)
    logits = np.zeros((1, 6, 10), dtype=np.float64)
    for i in range(5):
      logits[0, i, tokens[0, i + 1]] = 20.0
    self.assertLess(checker.compute_masked_ce_loss(logits, tokens, shift=True), 1e-4)
    self.assertGreater(checker.compute_masked_ce_loss(logits, tokens, shift=False), 10.0)

    tokens_ignore = np.array([[1, 2, -100]], dtype=np.int64)
    logits_inf = np.zeros((1, 3, 10), dtype=np.float64)
    logits_inf[0, 2, :] = np.inf
    mask_ignore = np.array([[True, True, False]], dtype=bool)
    loss = checker.compute_masked_ce_loss(logits_inf, tokens_ignore, mask_ignore, shift=False)
    self.assertTrue(np.isfinite(loss))

    with self.assertRaises(ValueError):
      checker.compute_masked_ce_loss(np.zeros((1, 2, 10)), np.array([[1, 99]]), shift=False)

    logits_jax = np.random.randn(2, 16, 10)
    targets = np.random.randint(0, 10, size=(2, 16))
    res_pass = checker.check_tinymmlu_loss_delta(logits_jax, logits_jax + 1e-6, targets)
    self.assertTrue(res_pass["passed"])
    self.assertLessEqual(res_pass["loss_delta"], 1e-4)
    corrupt_logits = logits_jax.copy()
    corrupt_logits[..., 0] += 0.5
    res_fail = checker.check_tinymmlu_loss_delta(logits_jax, corrupt_logits, targets)
    self.assertFalse(res_fail["passed"])

  def test_golden_npz_roundtrip(self):
    temp_dir = self.create_tempdir().full_path
    npz_path = os.path.join(temp_dir, "test_golden.npz")
    tokens, logits = np.array([[101, 102, 103]]), np.random.randn(1, 3, 50)
    routing = np.array([[[0, 1], [1, 2], [2, 3]]])
    blocks = np.array([[[1, 2], [2, 3], [3, 4]]])
    mask = np.array([[True, True, False]])
    mean_var = {
        "post_attn_proj": np.array([0.12, 0.45]),
        "post_mlp_act": np.array([0.33, 0.78]),
    }
    checker.save_golden_npz(
        npz_path,
        tokens,
        logits,
        routing,
        blocks,
        mask,
        mean_var,
        512,
        "float32",
    )
    loaded = checker.load_golden_npz(npz_path)
    for key, expected in (
        ("tokens", tokens),
        ("logits", logits),
        ("routing_indices", routing),
        ("indexer_selected_blocks", blocks),
        ("target_mask", mask),
    ):
      np.testing.assert_array_equal(loaded[key], expected)
    self.assertIn("mean_var", loaded)
    for name, arr in mean_var.items():
      np.testing.assert_array_equal(loaded["mean_var"][name], arr)
    self.assertEqual(
        (
            loaded.get("meta", {}).get("seq_len"),
            loaded.get("meta", {}).get("dtype"),
        ),
        (512, "float32"),
    )


if __name__ == "__main__":
  absltest.main()
