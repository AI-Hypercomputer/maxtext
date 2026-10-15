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
  """Concise test suite for zero_gen_logits_checker core metrics, I/O, and runners."""

  def test_routing_and_indexer_block_parity(self):
    routing_jax = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    self.assertTrue(checker.check_routing_parity(routing_jax, routing_jax.copy())["passed"])
    routing_hf = routing_jax.copy()
    routing_hf[1, 0, 1] = 99
    self.assertEqual(checker.check_routing_parity(routing_jax, routing_hf)["num_mismatches"], 1)
    jax_blocks, hf_perm = np.array([[[10, 20, 30], [40, 50, 60]]]), np.array([[[30, 10, 20], [60, 50, 40]]])
    self.assertTrue(checker.check_indexer_block_parity(jax_blocks, hf_perm)["passed"])
    hf_perm[0, 1, 0] = 999
    self.assertFalse(checker.check_indexer_block_parity(jax_blocks, hf_perm)["passed"])
    with self.assertRaises(ValueError):
      checker.check_indexer_block_parity(np.array([1, 2]), np.array([1, 2]))

  def test_logits_parity_and_dynamic_threshold(self):
    for sl, exp in ((512, 0.01), (2048, 0.02), (8192, 0.04)):
      self.assertEqual(checker.get_dynamic_threshold(0.01, sl), exp)
    np.random.seed(0)
    logits = np.random.randn(2, 512, 32)
    self.assertTrue(checker.check_logits_parity(logits, logits.copy(), seq_len=512)["passed"])
    corrupt, mask = logits.copy(), np.zeros((2, 512), dtype=bool)
    corrupt[:, 200:, :] += 5.0
    mask[:, :100] = True
    self.assertTrue(checker.check_logits_parity(logits, corrupt, mask=mask, seq_len=512)["passed"])
    logits_a, logits_b = np.ones((1, 512, 10), dtype=np.float64), np.ones((1, 512, 10), dtype=np.float64)
    logits_b[0, 0, :] = -1.0
    self.assertFalse(checker.check_logits_parity(logits_a, logits_b, seq_len=512)["passed"])
    with self.assertRaises(ValueError):
      checker.check_logits_parity(np.zeros((2, 10)), np.zeros((2, 12)))

  def test_masked_ce_loss_and_tinymmlu_delta(self):
    tokens, logits = np.array([[1, 3, 5, 7, 2, 4]], dtype=np.int64), np.zeros((1, 6, 10), dtype=np.float64)
    for i in range(5):
      logits[0, i, tokens[0, i + 1]] = 20.0
    self.assertLess(checker.compute_masked_ce_loss(logits, tokens, shift=True), 1e-4)
    self.assertGreater(checker.compute_masked_ce_loss(logits, tokens, shift=False), 10.0)
    logits_inf = np.zeros((1, 3, 10), dtype=np.float64)
    logits_inf[0, 2, :] = np.inf
    loss = checker.compute_masked_ce_loss(logits_inf, np.array([[1, 2, -100]]), np.array([[True, True, False]]), False)
    self.assertTrue(np.isfinite(loss))
    with self.assertRaises(ValueError):
      checker.compute_masked_ce_loss(np.zeros((1, 2, 10)), np.array([[1, 99]]), shift=False)
    logits_jax, targets = np.random.randn(2, 16, 10), np.random.randint(0, 10, size=(2, 16))
    self.assertTrue(checker.check_tinymmlu_loss_delta(logits_jax, logits_jax + 1e-6, targets)["passed"])
    corrupt_logits = logits_jax.copy()
    corrupt_logits[..., 0] += 0.5
    self.assertFalse(checker.check_tinymmlu_loss_delta(logits_jax, corrupt_logits, targets)["passed"])

  def test_golden_npz_roundtrip_and_prefill_runners(self):
    npz_path = os.path.join(self.create_tempdir().full_path, "test_golden.npz")
    tokens, logits = np.array([[1, 2, 3, 4]], dtype=np.int64), np.ones((1, 4, 10), dtype=np.float32)
    for i in range(3):
      logits[0, i, tokens[0, i + 1]] = 5.0
    routing = np.array([[[[0, 1], [1, 2], [2, 3], [3, 4]]]], dtype=np.int32)
    blocks = np.array([[[[1, 2], [2, 3], [3, 4], [4, 5]]]], dtype=np.int32)
    mask, mean_var = np.array([[True, True, True, False]]), {"post_attn_proj": np.array([0.12, 0.45])}
    checker.save_golden_npz(npz_path, tokens, logits, routing, blocks, mask, mean_var, 512, "float32")
    loaded = checker.load_golden_npz(npz_path)
    for k, exp in (("tokens", tokens), ("logits", logits), ("routing_indices", routing), ("target_mask", mask)):
      np.testing.assert_array_equal(loaded[k], exp)
    self.assertEqual((loaded["meta"]["seq_len"], loaded["meta"]["dtype"]), (512, "float32"))

    def fake_model(_, return_aux=False):
      return (logits, [{"router_indices": routing[0], "selected_block_indices": blocks[0]}]) if return_aux else logits

    self.assertTrue(checker.run_prefill_logits_check(fake_model, npz_path, tokens)["passed"])
    self.assertTrue(checker.run_tinymmlu_check(fake_model, npz_path, tokens)["passed"])
    for mode, m in (("prefill_logits", fake_model), ("tinymmlu", fake_model), ("holdout_ce", lambda _: logits * 2.0)):
      self.assertEqual(checker.main([f"--mode={mode}", f"--golden_tensor_path={npz_path}"], model=m), 0)
    with self.assertRaises(ValueError):
      checker.main(["--mode=invalid", f"--golden_tensor_path={npz_path}"], model=fake_model)


if __name__ == "__main__":
  absltest.main()
