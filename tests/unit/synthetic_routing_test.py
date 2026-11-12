# Copyright 2023–2026 Google LLC
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

"""CPU tests for the synthetic MoE routing generator."""

import unittest

import numpy as np

from maxtext.utils import synthetic_routing


class SyntheticRoutingTest(unittest.TestCase):

  def _lognormal(self, seed=0, step=0, **kw):
    return synthetic_routing.lognormal_routing(
        batch=2,
        seq=64,
        num_layers=3,
        num_experts=32,
        top_k=4,
        sigmas=[0.5, 0.8, 1.0],
        seed=seed,
        step=step,
        threads=1,
        **kw,
    )

  def test_lognormal_shape_dtype_and_range(self):
    out = self._lognormal()
    self.assertEqual(out.shape, (2, 64, 3, 4))
    self.assertEqual(out.dtype, np.int32)
    self.assertTrue(np.all(out >= 0) and np.all(out < 32))

  def test_lognormal_topk_distinct_per_token(self):
    out = self._lognormal(row_taus="0.5", row_len=16)
    srt = np.sort(out, axis=-1)
    self.assertTrue(np.all(srt[..., 1:] != srt[..., :-1]))

  def test_lognormal_is_deterministic(self):
    np.testing.assert_array_equal(self._lognormal(seed=3, step=1), self._lognormal(seed=3, step=1))
    np.testing.assert_array_equal(
        self._lognormal(seed=3, step=1, row_taus="0.5", row_len=16),
        self._lognormal(seed=3, step=1, row_taus="0.5", row_len=16),
    )

  def test_lognormal_depends_on_seed_and_step(self):
    self.assertFalse(np.array_equal(self._lognormal(seed=0), self._lognormal(seed=1)))
    self.assertFalse(np.array_equal(self._lognormal(step=0), self._lognormal(step=1)))

  def test_lognormal_threads_match_serial(self):
    serial = self._lognormal(seed=5)
    threaded = synthetic_routing.lognormal_routing(2, 64, 3, 32, 4, [0.5, 0.8, 1.0], 5, 0, threads=3)
    np.testing.assert_array_equal(serial, threaded)

  def test_balanced_is_exactly_balanced(self):
    out = synthetic_routing.balanced_routing(batch=2, seq=64, num_layers=2, num_experts=32, top_k=4, step=1)
    self.assertEqual(out.shape, (2, 64, 2, 4))
    for layer in range(2):
      counts = np.bincount(out[:, :, layer, :].ravel(), minlength=32)
      self.assertTrue(np.all(counts == counts[0]))
    srt = np.sort(out, axis=-1)
    self.assertTrue(np.all(srt[..., 1:] != srt[..., :-1]))


if __name__ == "__main__":
  unittest.main()
