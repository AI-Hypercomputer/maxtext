# Copyright 2026 Google LLC
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

"""CPU tests for distillation evaluation restarts and training iterator position."""

import itertools
import shutil
import tempfile
import unittest
from unittest import mock

import pytest

pytest.importorskip("tunix")
pytestmark = [pytest.mark.post_training]

import jax
import numpy as np

from maxtext.configs import pyconfig
from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator
from maxtext.trainers.post_train.distillation import distillation_utils, train_distill
from tests.utils.test_helpers import get_test_config_path


class ResettableIterator:
  """Models input wrappers that restart only through an explicit reset()."""

  def __init__(self, batches):
    self.batches = batches
    self.iterator = iter(batches)
    self.reset_count = 0

  def __iter__(self):
    return self

  def __next__(self):
    return next(self.iterator)

  def reset(self):
    self.iterator = iter(self.batches)
    self.reset_count += 1


class CountingMultiHostIterator(MultiHostDataLoadIterator):
  """Counts batches actually yielded by the real sharding iterator."""

  def __init__(self, batches, mesh):
    super().__init__(batches, mesh, generate_padding_batch=False)
    self.read_count = 0

  def __next__(self):
    batch = super().__next__()
    self.read_count += 1
    return batch


class DistillationIteratorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.batches = []
    for token in (1, 2, 3):
      self.batches.append(
          {
              "inputs": np.full((jax.device_count(), 4), token, dtype=np.int32),
              "targets": np.full((jax.device_count(), 4), token + 1, dtype=np.int32),
              "inputs_position": np.tile(np.arange(4, dtype=np.int32), (jax.device_count(), 1)),
              "inputs_segmentation": np.ones((jax.device_count(), 4), dtype=np.int32),
              "targets_segmentation": np.ones((jax.device_count(), 4), dtype=np.int32),
          }
      )

  def test_reiterable_evaluation_restarts(self):
    evaluation = distillation_utils.MaxTextToTunixIterable(self.batches)
    for _ in range(3):
      self.assertEqual([int(batch.input_tokens[0, 0]) for batch in evaluation], [1, 2, 3])

  def test_explicit_reset_evaluation_restarts(self):
    raw = ResettableIterator(self.batches)
    evaluation = distillation_utils.MaxTextToTunixIterable(raw)
    for _ in range(3):
      self.assertEqual([int(batch.input_tokens[0, 0]) for batch in evaluation], [1, 2, 3])
    self.assertEqual(raw.reset_count, 3)

  def test_partial_evaluation_restarts_only_at_pass_boundary(self):
    raw = ResettableIterator(self.batches)
    evaluation = distillation_utils.MaxTextToTunixIterable(raw)
    current_pass = iter(evaluation)
    self.assertEqual(int(next(current_pass).input_tokens[0, 0]), 1)
    self.assertIs(iter(current_pass), current_pass)
    self.assertEqual(int(next(current_pass).input_tokens[0, 0]), 2)
    self.assertEqual(raw.reset_count, 1)

    self.assertEqual([int(batch.input_tokens[0, 0]) for batch in evaluation], [1, 2, 3])
    self.assertEqual(raw.reset_count, 2)

  def test_training_adapter_preserves_restored_position(self):
    raw = ResettableIterator(self.batches)
    next(raw)  # Model an iterator restored after its first batch.
    training = distillation_utils.MaxTextToTunixIterator(raw)
    self.assertIs(iter(training), training)
    limited_training = itertools.islice(training, 2)
    self.assertEqual([int(batch.input_tokens[0, 0]) for batch in limited_training], [2, 3])
    self.assertEqual(list(training), [])
    self.assertEqual(raw.reset_count, 0)

  def test_empty_evaluation_stays_empty(self):
    for raw in ([], ResettableIterator([])):
      with self.subTest(iterator_type=type(raw).__name__):
        evaluation = distillation_utils.MaxTextToTunixIterable(raw)
        self.assertEqual(list(evaluation), [])
        self.assertEqual(list(evaluation), [])

  def test_evaluation_preserves_batch_fields(self):
    batch = dict(self.batches[0])
    batch["top_k_logits"] = np.ones((jax.device_count(), 4, 2), dtype=np.float32)
    batch["top_k_indices"] = np.zeros((jax.device_count(), 4, 2), dtype=np.int32)
    evaluation = distillation_utils.MaxTextToTunixIterable([batch])
    expected = next(distillation_utils.MaxTextToTunixIterator(iter([batch])))
    for _ in range(2):
      actual = next(iter(evaluation))
      for expected_leaf, actual_leaf in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(expected_leaf, actual_leaf)

  def test_train_distill_repeats_evaluation(self):
    """Run real tiny teacher/student models through the production launcher on CPU."""
    test_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, test_dir)
    args = [
        "",
        get_test_config_path(),
        f"base_output_directory={test_dir}",
        "run_name=distill_eval_test",
        "dataset_type=synthetic",
        "vocab_size=32",
        "base_emb_dim=8",
        "base_num_query_heads=1",
        "base_num_kv_heads=1",
        "base_mlp_dim=16",
        "base_num_decoder_layers=1",
        "head_dim=8",
        "per_device_batch_size=1",
        "max_target_length=4",
        "steps=3",
        "enable_checkpointing=False",
        "save_checkpoint_on_completion=False",
        "async_checkpointing=False",
        "log_period=1",
        "eval_interval=1",
        "use_sft=False",
        "distill_beta=0.0",
        "dataset_path=/",
        "skip_jax_distributed_system=True",
        "attention=dot_product",
    ]
    student_config = pyconfig.initialize(args)
    teacher_config = pyconfig.initialize(args)
    raw_iterators = []
    evaluations = []
    original_run_eval = train_distill.MaxTextDistillationTrainer._run_eval  # pylint: disable=protected-access

    def create_data(config, mesh):
      del config
      for _ in range(2):
        raw_iterators.append(CountingMultiHostIterator(self.batches, mesh))
      return tuple(raw_iterators)

    def run_eval(trainer, *eval_args, **eval_kwargs):
      if not evaluations:
        self.addCleanup(trainer.close)
      before = raw_iterators[1].read_count
      original_run_eval(trainer, *eval_args, **eval_kwargs)
      evaluations.append((trainer.train_steps, raw_iterators[1].read_count - before))
      loss = trainer.metrics_logger.get_metric(trainer.metrics_prefix, "loss", "eval")
      self.assertTrue(np.isfinite(float(loss)))

    # Supply local tokenized data; keep model creation, training and evaluation real.
    with (
        mock.patch.object(train_distill.tokenizer, "build_tokenizer", return_value=mock.Mock(pad_id=0)),
        mock.patch.object(train_distill.input_pipeline_interface, "create_data_iterator", side_effect=create_data),
        mock.patch.object(train_distill.MaxTextDistillationTrainer, "_run_eval", run_eval),
    ):
      train_distill.train_distill(student_config, teacher_config)

    self.assertEqual(evaluations, [(0, 3), (1, 3), (2, 3), (3, 3)])
    self.assertEqual(raw_iterators[0].read_count, 3)


if __name__ == "__main__":
  unittest.main()
