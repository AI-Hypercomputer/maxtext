# Copyright 2025 Google LLC
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


"""Tests for mmlu_eval_perplexity.py"""
import unittest

from absl.testing import absltest
from benchmarks.mmlu import mmlu_eval_perplexity


class MmluEvalPerplexityTest(unittest.TestCase):

  def test_construct_prompt(self):
    subject = "abstract_algebra"
    question = "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
    choices = ["0", "4", "2", "6"]

    prompt = mmlu_eval_perplexity.construct_prompt(subject, question, choices)

    expected_prompt = "The following are multiple choice questions (with answers) about abstract algebra.\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:"
    self.assertEqual(prompt, expected_prompt)

  def test_get_letter_indices(self):

    class MockTokenizer:

      def encode(self, text, *args, **kwargs):
        mapping = {"A": [100], "B": [101], "C": [102], "D": [103]}
        if text in mapping:
          return mapping[text]
        return [0]

    tokenizer = MockTokenizer()
    indices = mmlu_eval_perplexity.get_letter_indices(tokenizer)
    self.assertEqual(indices, [100, 101, 102, 103])

  def test_eval_step(self):
    import jax.numpy as jnp
    import numpy as np

    class MockModel:
      def __call__(self, decoder_input_tokens, decoder_positions, decoder_segment_ids, enable_dropout):
        batch_size = decoder_input_tokens.shape[0]
        seq_len = decoder_input_tokens.shape[1]
        vocab_size = 4

        # logit at (b, s, v) = s
        logits = jnp.ones((batch_size, seq_len, vocab_size))
        for s in range(seq_len):
          logits = logits.at[:, s, :].set(s)
        return logits

    class MockState:
      params = None

    mock_model = MockModel()
    mock_state = MockState()

    data = {
        "inputs": jnp.array([[1, 2, 3, 0], [1, 2, 0, 0]]),
        "inputs_position": jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        "inputs_segmentation": jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]]),
    }

    last_logits = mmlu_eval_perplexity.eval_step(mock_model, None, mock_state, data, None)

    # Batch 0: prompt_lens = 3, last_index = 2 -> value = 2
    # Batch 1: prompt_lens = 2, last_index = 1 -> value = 1
    self.assertEqual(last_logits.shape, (2, 4))
    np.testing.assert_array_equal(last_logits, np.array([[2, 2, 2, 2], [1, 1, 1, 1]]))
if __name__ == '__main__':
  absltest.main()