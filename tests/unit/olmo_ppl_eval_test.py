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

  def _eval(self, loaded, scored):
    """An OlmoPplEval over sets a (three documents) and b (one), loading `loaded` rows and scoring `scored`."""
    with tempfile.TemporaryDirectory() as d:
      for name, tokens in (("a", [1, 2, EOS, 3, EOS, 4, 5, 6, EOS]), ("b", [7, EOS])):
        os.makedirs(os.path.join(d, name, "val"))
        np.array(tokens, dtype=np.uint32).tofile(os.path.join(d, name, "val", "part-0-00000.npy"))
      cfg = types.SimpleNamespace(
          olmo_ppl_eval_interval=1000,
          max_target_length=4,
          global_batch_size_to_load_eval=loaded,
          micro_batch_size_to_eval_on=scored,
          olmo_ppl_eval_eos_token_id=EOS,
          olmo_ppl_eval_pad_token_id=PAD,
          olmo_ppl_eval_sets=["a", "b"],
          olmo_ppl_eval_dir=d,
      )
      return olmo_ppl_eval.OlmoPplEval(cfg)

  def test_batches_hold_one_set_and_count_document_tokens(self):
    """Each batch is one set; the last is padded with weightless rows; the denominator is sum(L), as OLMo-core."""
    ev = self._eval(loaded=2, scored=2)
    label_a, batches_a, denom_a = ev.sets[0]
    label_b, batches_b, denom_b = ev.sets[1]
    self.assertEqual((label_a, label_b), ("a-validation", "b-validation"))
    self.assertEqual((len(batches_a), denom_a), (2, 3 + 2 + 4))  # 3 docs in batches of 2
    self.assertEqual((len(batches_b), denom_b), (1, 2))
    np.testing.assert_array_equal(batches_a[1]["targets_segmentation"][1], [0, 0, 0, 0])  # fill row
    self.assertTrue(ev.due(999) and not ev.due(1000) and ev.due(1999))

  def test_documents_only_in_scored_rows(self):
    """With fewer rows scored than loaded, documents fill only the scored rows, so the denominator counts only
    documents the eval step sees; the arrays keep the loaded size."""
    ev = self._eval(loaded=4, scored=2)
    _, batches, denom = ev.sets[0]
    self.assertEqual((len(batches), denom), (2, 3 + 2 + 4))  # 3 docs, 2 per batch
    for rows in batches:
      self.assertEqual(rows["inputs"].shape, (4, 4))
      np.testing.assert_array_equal(rows["targets_segmentation"][2:], np.zeros((2, 4)))  # unscored rows
    scored_targets = sum(int(rows["targets_segmentation"][:2].sum()) for rows in batches)
    self.assertEqual(scored_targets, (3 - 1) + (2 - 1) + (4 - 1))


class EmoEvalPoolsTest(unittest.TestCase):
  """Under ``emo_eval_pools`` the router keeps the eval pool instead of sampling a pool size per document."""

  def _mask(self, key):
    """(logits, EMo-masked logits) for a router with pools in [2, 8] of 8 experts, sampled with `key`."""
    cfg = types.SimpleNamespace(
        emo_document_expert_pool_choices=[],
        emo_document_expert_pool_probs=[],
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


class PplEvalGraphdefTest(unittest.TestCase):
  """The perplexity program's MoE layers don't drop tokens, whatever the training ragged_buffer_factor."""

  def _moe_modules(self, rbf, eval_rbf=-1.0):
    """The perplexity graphdef's RoutedMoE modules for training / eval ragged_buffer_factor `rbf` / `eval_rbf`.

    The model is built dropless (a capped buffer needs expert parallelism, which one CPU device lacks); the builder
    only reads the two buffer factors, so they come from a stand-in config.
    """
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.layers import moe, quantizations  # pylint: disable=import-outside-toplevel
    from maxtext.trainers.pre_train import train  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils, model_creation_utils  # pylint: disable=import-outside-toplevel
    from maxtext.utils.globals import MAXTEXT_PKG_DIR  # pylint: disable=import-outside-toplevel

    cfg = pyconfig.initialize(
        [
            "",
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "model_name=olmoe3-30m",
            "run_name=ppl_graphdef",
            "enable_checkpointing=False",
            "scan_layers=False",
            "per_device_batch_size=1",
            "max_target_length=64",
            "override_model_config=True",
            "vocab_size=512",
        ]
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = model_creation_utils.get_transformer_model(
        cfg, mesh, quantizations.configure_quantization(cfg), rngs=nnx.Rngs(0)
    )
    graphdef, state = nnx.split(model)
    factors = types.SimpleNamespace(ragged_buffer_factor=rbf, eval_ragged_buffer_factor=eval_rbf)
    ppl = train.build_ppl_eval_graphdef(factors, graphdef, state)
    modules = [m for _, m in nnx.iter_graph(nnx.merge(ppl, state)) if isinstance(m, moe.RoutedMoE)]
    self.assertTrue(modules)
    self.assertTrue(all(type(m).__name__ == "OLMoE3LatentRoutedMoE" for m in modules))  # the subclass is matched
    return ppl is graphdef, modules

  def test_capped_training_buffer_runs_dropless(self):
    same, modules = self._moe_modules(1.0625)
    self.assertFalse(same)
    self.assertTrue(all(m.force_dropless for m in modules))

  def test_eval_buffer_factor_is_used_when_set(self):
    _, modules = self._moe_modules(1.0625, eval_rbf=2.0)
    self.assertTrue(all(m.get_ragged_buffer_factor() == 2.0 and not m.force_dropless for m in modules))

  def test_dropless_training_keeps_its_graphdef(self):
    same, _ = self._moe_modules(-1.0)
    self.assertTrue(same)


if __name__ == "__main__":
  unittest.main()
