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

"""Tests for inference_utils.py."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np

from MaxText import inference_utils


class LogitsProcessorTest(unittest.TestCase):
  """Test suite for LogitsProcessors in inference_utils."""

  def setUp(self):
    self.logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=jnp.float32)
    self.neg_inf = inference_utils.NEG_INF

  def test_temperature_warper(self):
    """Tests TemperatureLogitsWarper."""
    temp = 2.0
    warper = inference_utils.TemperatureLogitsWarper(temperature=temp)
    processed = warper(self.logits)
    expected = self.logits / temp
    np.testing.assert_allclose(processed, expected, rtol=1e-5)

  def test_top_k_warper(self):
    """Tests TopKLogitsWarper."""
    top_k = 2
    warper = inference_utils.TopKLogitsWarper(top_k=top_k, filter_value=self.neg_inf)
    processed = warper(self.logits)
    # Top 2 are 4.0 and 5.0. Others should be masked.
    expected = jnp.array([[self.neg_inf, self.neg_inf, self.neg_inf, 4.0, 5.0]], dtype=jnp.float32)
    np.testing.assert_allclose(processed, expected, rtol=1e-5)

  def test_top_p_warper(self):
    """Tests TopPLogitsWarper."""
    probs = jnp.array([0.4, 0.3, 0.2, 0.1])
    logits = jnp.log(probs)[None, :]

    # Top P = 0.5.
    warper = inference_utils.TopPLogitsWarper(top_p=0.5, filter_value=self.neg_inf)
    processed = warper(logits)

    expected_mask = jnp.array([1, 1, 0, 0], dtype=bool)
    self.assertTrue(jnp.all(jnp.where(expected_mask, processed > self.neg_inf, processed == self.neg_inf)))

  def test_min_p_warper(self):
    """Tests MinPLogitsWarper."""
    logits = jnp.log(jnp.array([[0.5, 0.4, 0.08, 0.02]]))
    warper = inference_utils.MinPLogitsWarper(min_p=0.1, filter_value=self.neg_inf)
    processed = warper(logits)

    expected_mask = jnp.array([[True, True, True, False]])
    self.assertTrue(jnp.all(jnp.where(expected_mask, processed > self.neg_inf, processed == self.neg_inf)))

  def test_processor_list(self):
    """Tests LogitsProcessorList chaining."""
    logits = jnp.array([[2.0, 4.0, 6.0, 8.0]])
    chain = inference_utils.LogitsProcessorList(
        [
            inference_utils.TemperatureLogitsWarper(temperature=2.0),
            inference_utils.TopKLogitsWarper(top_k=2, filter_value=self.neg_inf),
        ]
    )
    processed = chain(logits)
    expected = jnp.array([[self.neg_inf, self.neg_inf, 3.0, 4.0]])
    np.testing.assert_allclose(processed, expected, rtol=1e-5)


class SamplingTest(unittest.TestCase):
  """Tests for the sampling function."""

  def test_greedy_sampling(self):
    logits = jnp.array([[0.1, 0.9, 0.0]])
    rng = jax.random.PRNGKey(0)
    token = inference_utils.sampling(logits, rng, algorithm="greedy")
    self.assertEqual(token.item(), 1)

  def test_composite_sampling_construction(self):
    """Tests that sampling function applies processors correctly."""
    logits = jnp.array([[10.0, 20.0, 30.0]])
    rng = jax.random.PRNGKey(0)
    token = inference_utils.sampling(
        logits,
        rng,
        algorithm="composite",
        topk=1,
        nucleus_topp=1.0,
        temperature=1.0,
    )
    self.assertEqual(token.item(), 2)


if __name__ == "__main__":
  unittest.main()
