# Copyright 2023-2026 Google LLC
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

"""OLMo-core's in-loop perplexity evals: padded one-document rows, the CE denominator, the EMO eval pool."""

import os
import tempfile
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.input_pipeline import olmo_ppl_eval
from maxtext.models import olmoe3

EOS, PAD = 100257, 100277


class OlmoPplEvalRowsTest(unittest.TestCase):
  """Documents are EOS-delimited (EOS included), truncated, one per row; padding carries no weight."""

  def test_split_documents_keeps_eos_and_truncates(self):
    tokens = np.array([5, 6, EOS, 7, EOS, 8, 9, 10, 11, EOS, 12], dtype=np.uint32)  # trailing 12 has no EOS
    docs = olmo_ppl_eval.split_documents(tokens, EOS, max_len=3)
    self.assertEqual([d.tolist() for d in docs], [[5, 6, EOS], [7, EOS], [8, 9, 10]])

  def test_rows_shift_targets_and_mask_padding(self):
    docs = [np.array([5, 6, EOS], dtype=np.uint32), np.array([7, EOS], dtype=np.uint32)]
    rows = olmo_ppl_eval.build_rows(docs, seq_len=4, pad_token_id=PAD)
    np.testing.assert_array_equal(rows["inputs"], [[5, 6, EOS, PAD], [7, EOS, PAD, PAD]])
    np.testing.assert_array_equal(rows["targets"], [[6, EOS, PAD, PAD], [EOS, PAD, PAD, PAD]])
    # Position t predicts token t + 1: L - 1 targets per document.
    np.testing.assert_array_equal(rows["targets_segmentation"], [[1, 1, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_array_equal(rows["inputs_segmentation"], [[1, 1, 1, 0], [1, 1, 0, 0]])

  def test_batches_hold_one_set_and_count_document_tokens(self):
    """Each batch is one set; the last is padded with weightless rows; the denominator is sum(L), as OLMo-core."""
    with tempfile.TemporaryDirectory() as d:
      for name, tokens in (("a", [1, 2, EOS, 3, EOS, 4, 5, 6, EOS]), ("b", [7, EOS])):
        os.makedirs(os.path.join(d, name, "val"))
        np.array(tokens, dtype=np.uint32).tofile(os.path.join(d, name, "val", "part-0-00000.npy"))
      cfg = types.SimpleNamespace(
          olmo_ppl_eval_interval=1000,
          max_target_length=4,
          global_batch_size_to_load_eval=2,
          olmo_ppl_eval_eos_token_id=EOS,
          olmo_ppl_eval_pad_token_id=PAD,
          olmo_ppl_eval_sets=["a", "b"],
          olmo_ppl_eval_dir=d,
      )
      ev = olmo_ppl_eval.OlmoPplEval(cfg)
    (label_a, batches_a, denom_a), (label_b, batches_b, denom_b) = ev.sets
    self.assertEqual((label_a, label_b), ("a-validation", "b-validation"))
    self.assertEqual((len(batches_a), denom_a), (2, 3 + 2 + 4))  # 3 docs in batches of 2
    self.assertEqual((len(batches_b), denom_b), (1, 2))
    np.testing.assert_array_equal(batches_a[1]["targets_segmentation"][1], [0, 0, 0, 0])  # fill row
    self.assertTrue(ev.due(999) and not ev.due(1000) and ev.due(1999))


class EmoEvalPoolsTest(unittest.TestCase):
  """Under ``emo_eval_pools`` the router keeps the eval pool instead of sampling a pool size per document."""

  def _mask(self, key):
    cfg = types.SimpleNamespace(
        emo_min_document_expert_pool=2,
        emo_max_document_expert_pool=8,
        emo_eval_document_expert_pool=8,
        model_call_mode="train",
        moe_lean_routing=False,
    )
    router = types.SimpleNamespace(config=cfg)
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 8))
    segments = jnp.ones((2, 16), jnp.int32)
    return logits, olmoe3.OLMoE3LatentRoutedMoE._emo_mask_logits(  # pylint: disable=protected-access
        router, logits, segments, key
    )

  def test_training_samples_pools(self):
    logits, masked = self._mask(jax.random.PRNGKey(1))
    self.assertTrue(bool(jnp.any(jnp.isinf(masked))), "a sampled pool below 8 should mask some experts")
    del logits

  def test_eval_context_uses_the_eval_pool(self):
    with olmoe3.emo_eval_pools():
      logits, masked = self._mask(jax.random.PRNGKey(1))
    np.testing.assert_array_equal(np.asarray(masked), np.asarray(logits))  # eval pool 8 = every expert


if __name__ == "__main__":
  unittest.main()
