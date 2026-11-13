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

"""Tests for training and data loading hooks for DPO"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.external_training, pytest.mark.post_training]

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
import numpy as np
import optax
import os
import shutil
import tempfile
from tunix.sft import metrics_logger
from tunix.sft.dpo import dpo_trainer

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.trainers.post_train.dpo import hooks as dpo_hooks
from maxtext.utils import maxtext_utils
from maxtext.utils import exceptions


class ToyPolicy(nnx.Module):
  """Small token lookup model for exercising Tunix's real DPO/ORPO losses."""

  def __init__(self):
    self.logits = nnx.Param(jnp.zeros((8, 8)))

  def __call__(self, tokens, *args, **kwargs):
    return self.logits[tokens], None


class DPOHooksTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.test_dir)
    self.config = pyconfig.initialize_pydantic(
        ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "dpo.yml")],
        per_device_batch_size=1,
        run_name="test",
        base_output_directory=self.test_dir,
        tensorboard_dir=self.test_dir,
        skip_jax_distributed_system=True,
        eval_steps=-1,
    )
    self.mesh = Mesh(maxtext_utils.create_device_mesh(self.config), self.config.mesh_axes)

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_dpo_data_hooks_load_next_train_batch(self, mock_create_data_iterator):
    expected_batch = {"inputs": np.zeros([jax.device_count(), self.config.max_target_length], dtype=int)}
    mock_data_iterator = MagicMock()
    mock_data_iterator.__next__.return_value = expected_batch
    mock_create_data_iterator.return_value = mock_data_iterator, None

    data_hooks = dpo_hooks.DPODataHooks(self.config, self.mesh, goodput_recorder=None)
    data_hooks.load_next_train_batch(train_ctx=None)

    self.assertIsNotNone(data_hooks.train_batch)
    self.assertEqual(data_hooks.train_batch["inputs"].shape, expected_batch["inputs"].shape)

  def test_dpo_training_hooks_get_total_weights(self):
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(self.config)
    training_hooks = dpo_hooks.DPOTrainingHooks(self.config, self.mesh, learning_rate_schedule, goodput_recorder=None)
    batch = {"chosen_mask": np.array([[1, 1, 0], [1, 0, 0]]), "rejected_mask": np.array([[1, 0, 0], [1, 1, 0]])}
    total_weights = training_hooks.get_total_weights(batch)
    self.assertEqual(total_weights, 6)

  def _create_eval_trainer(self, algorithm):
    """Build a real Tunix trainer with in-memory data and captured MaxText metrics."""
    model = ToyPolicy()
    trainer = dpo_trainer.DPOTrainer(
        model=model,
        ref_model=nnx.clone(model) if algorithm == "dpo" else None,
        optimizer=optax.sgd(0.01),
        training_config=dpo_trainer.DPOTrainingConfig(
            algorithm=algorithm,
            max_steps=1,
            eval_every_n_steps=1,
            data_sharding_axis=("fsdp",),
            metrics_logging_options=metrics_logger.MetricsLoggerOptions(log_dir=self.test_dir),
        ),
    )
    self.addCleanup(trainer.close)
    # Keep the objectives, aggregation and hooks real; capture only the logging sink.
    with patch("maxtext.trainers.post_train.hooks.MetricLogger"):
      training_hooks = dpo_hooks.DPOTrainingHooks(
          SimpleNamespace(target_eval_loss=-1), self.mesh, lambda _: 0.01, goodput_recorder=None
      )
    with patch("maxtext.trainers.post_train.hooks.create_data_iterator", return_value=(iter(()), iter(()))):
      data_hooks = dpo_hooks.DPODataHooks(self.config, self.mesh, goodput_recorder=None)
    trainer.with_training_hooks(training_hooks)
    trainer.with_data_hooks(data_hooks)
    return trainer

  def _eval_batch(self, batch_size=1, reverse_preferences=False):
    chosen, rejected = ([5, 6], [3, 4]) if reverse_preferences else ([3, 4], [5, 6])
    return {
        "prompt_ids": jnp.tile(jnp.array([[1, 2]]), (batch_size, 1)),
        "prompt_mask": jnp.ones((batch_size, 2), dtype=jnp.int32),
        "chosen_ids": jnp.tile(jnp.array([chosen]), (batch_size, 1)),
        "chosen_mask": jnp.ones((batch_size, 2), dtype=jnp.int32),
        "rejected_ids": jnp.tile(jnp.array([rejected]), (batch_size, 1)),
        "rejected_mask": jnp.ones((batch_size, 2), dtype=jnp.int32),
    }

  def _run_eval(self, trainer, batches):
    trainer.data_hooks.eval_data_iterator = iter(batches)
    with jax.set_mesh(self.mesh):
      # Exercise the same Tunix evaluation loop and objective used by train_dpo.
      trainer._run_eval([], lambda example: trainer._eval_step(trainer.model, example))  # pylint: disable=protected-access
    return trainer.training_hooks.metric_logger.write_metrics.call_args.args[0]["scalar"]

  def test_eval_loss_is_independent_of_batch_count(self):
    for algorithm in ("dpo", "orpo"):
      trainer = self._create_eval_trainer(algorithm)
      expected_loss = self._run_eval(trainer, [self._eval_batch()])["eval/avg_loss"]
      for batch_count in (1, 2, 4):
        with self.subTest(algorithm=algorithm, batch_count=batch_count):
          metrics = self._run_eval(trainer, [self._eval_batch()] * batch_count)
          self.assertAlmostEqual(float(metrics["eval/avg_loss"]), float(expected_loss), places=6)
          self.assertAlmostEqual(
              float(metrics["eval/avg_loss"]),
              float(trainer.metrics_logger.get_metric(trainer.metrics_prefix, "loss", "eval")),
              places=6,
          )
          self.assertEqual(float(metrics["eval/total_weights"]), 4 * batch_count)
          self.assertFalse(trainer.training_hooks.eval_metadata)

  def test_eval_loss_weights_uneven_batches_by_example_count(self):
    for algorithm in ("dpo", "orpo"):
      with self.subTest(algorithm=algorithm):
        trainer = self._create_eval_trainer(algorithm)
        # Nonuniform policy logits make reversed preferences have different losses.
        trainer.model.logits[...] = jnp.broadcast_to(jnp.arange(8, dtype=jnp.float32) / 4, (8, 8))
        first = self._eval_batch()
        second = self._eval_batch(batch_size=2, reverse_preferences=True)
        first_loss = float(self._run_eval(trainer, [first])["eval/avg_loss"])
        second_loss = float(self._run_eval(trainer, [second])["eval/avg_loss"])
        self.assertNotAlmostEqual(first_loss, second_loss)

        metrics = self._run_eval(trainer, [first, second])

        self.assertAlmostEqual(float(metrics["eval/avg_loss"]), (first_loss + 2 * second_loss) / 3, places=6)
        self.assertEqual(float(metrics["eval/total_weights"]), 12)

  def test_early_stopping_uses_preaveraged_eval_loss(self):
    for algorithm in ("dpo", "orpo"):
      with self.subTest(algorithm=algorithm):
        trainer = self._create_eval_trainer(algorithm)
        batch = self._eval_batch()
        expected_loss = float(self._run_eval(trainer, [batch])["eval/avg_loss"])
        trainer.training_hooks.config.target_eval_loss = expected_loss * 0.75

        # Dividing the mean by two would incorrectly stop this evaluation.
        self._run_eval(trainer, [batch, batch])

        for threshold in (expected_loss, expected_loss + 0.1):
          trainer.training_hooks.config.target_eval_loss = threshold
          with self.assertRaises(exceptions.StopTraining):
            self._run_eval(trainer, [batch, batch])
          self.assertFalse(trainer.training_hooks.eval_metadata)


if __name__ == "__main__":
  unittest.main()
