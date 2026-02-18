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


"""Unit tests for the Distillation Trainer."""

import shutil
import tempfile
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl.testing import absltest

# Import the module under test
from maxtext.trainers.post_train.distillation import train_distill
from maxtext.trainers.post_train.distillation import distillation_utils
from MaxText import pyconfig


# pylint: disable=protected-access
class TrainDistillTest(unittest.TestCase):

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)

  def test_maxtext_to_tunix_iterator(self):
    """Verifies the adapter correctly converts dictionary batches to dataclasses."""

    # 1. Create a dummy iterator that simulates MaxText data loader
    dummy_batch = {
        "inputs": np.array([[10, 11]]),
        "inputs_position": np.array([[0, 1]]),
        "inputs_segmentation": np.array([[1, 1]]),
        "targets": np.array([[11, 12]]),
    }

    dummy_iter = iter([dummy_batch])

    # 2. Initialize Adapter
    adapter = distillation_utils.MaxTextToTunixIterator(dummy_iter)

    # 3. Fetch Batch
    tunix_input = next(adapter)

    # 4. Verify Fields
    self.assertIsInstance(tunix_input, distillation_utils.MaxTextTrainingInput)
    np.testing.assert_array_equal(tunix_input.input_tokens, dummy_batch["inputs"])
    np.testing.assert_array_equal(tunix_input.positions, dummy_batch["inputs_position"])
    np.testing.assert_array_equal(tunix_input.decoder_segment_ids, dummy_batch["inputs_segmentation"])
    np.testing.assert_array_equal(tunix_input.targets, dummy_batch["targets"])

    # Verify constructed mask (segmentation != 0)
    expected_mask = dummy_batch["inputs_segmentation"] != 0
    np.testing.assert_array_equal(tunix_input.input_mask, expected_mask)

  def test_maxtext_to_tunix_iterator_packed_fallback(self):
    """Verifies fallback behavior when segmentation is missing."""
    dummy_batch = {
        "inputs": np.array([[10, 11]]),
        "inputs_position": np.array([[0, 1]]),
        "targets": np.array([[11, 12]]),
    }
    dummy_iter = iter([dummy_batch])
    adapter = distillation_utils.MaxTextToTunixIterator(dummy_iter)
    tunix_input = next(adapter)

    self.assertIsNone(tunix_input.decoder_segment_ids)
    self.assertTrue(np.all(tunix_input.input_mask))

  def test_prepare_inputs_logic(self):
    """Verifies the filtering and teacher call logic in the custom trainer."""
    # 1. Initialize Trainer (bypass init)
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)

    # Setup mocks
    trainer._mode = "train"
    trainer.strategy = mock.Mock()
    trainer.teacher_model = mock.Mock()
    trainer.model = mock.Mock()
    trainer.gen_model_input_fn = lambda x: {"inputs": {"some_key": "some_val"}}

    # 2. Setup Input
    # pylint: disable=unexpected-keyword-arg
    input_data = distillation_utils.MaxTextTrainingInput(
        input_tokens=jnp.array([[1]]),
        input_mask=jnp.array([[True]]),
        positions=jnp.array([[0]]),
        targets=jnp.array([[1]]),
    )

    # 3. Mock Strategy Output
    fake_teacher_logits = jnp.zeros((1, 1, 10))
    trainer.strategy.get_teacher_outputs.return_value = fake_teacher_logits

    # 4. Run
    result = trainer._prepare_inputs(input_data)

    # 5. Verify
    trainer.strategy.get_teacher_outputs.assert_called_once()
    self.assertIsNotNone(result.teacher_output)
    self.assertEqual(result.teacher_output.shape, (1, 1, 10))

  def test_optimizer_factory(self):
    """Verifies the optimizer factory injects hyperparams and handles configs."""
    # Mock config
    config = mock.Mock(spec=pyconfig.HyperParameters)
    config.learning_rate = 1e-3
    config.opt_type = "adamw"
    config.adam_b1 = 0.9
    config.adam_b2 = 0.99
    config.adam_eps = 1e-8
    config.adam_eps_root = 0.0
    config.adam_weight_decay = 0.0
    config.mu_dtype = "float32"
    config.gradient_clipping_threshold = 1.0
    config.warmup_steps_fraction = 0.1
    config.learning_rate_final_fraction = 0.1

    # 1. Test Valid Creation
    opt = train_distill.get_distillation_optimizer(config, max_train_steps=100)

    # Initialize to check state structure
    params = {"a": jnp.array([0.0])}
    state = opt.init(params)

    # Verify InjectHyperparamsState is the top-level state (required for Tunix logging)
    # Note: When injecting a schedule (callable), optax returns InjectStatefulHyperparamsState
    self.assertTrue(
        isinstance(state, (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState)),
        f"State is {type(state)}, expected InjectHyperparamsState or InjectStatefulHyperparamsState",
    )
    self.assertIn("learning_rate", state.hyperparams)

    # 2. Test Muon Rejection
    config.opt_type = "muon"
    with self.assertRaisesRegex(ValueError, "Muon optimizer is not currently supported"):
      train_distill.get_distillation_optimizer(config, max_train_steps=100)

  def test_monitored_strategy(self):
    """Verifies the strategy calculates metrics and returns the correct tuple."""
    strategy = distillation_utils.MonitoredLogitStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        labels_fn=lambda t: t,
        temperature=1.0,
        alpha=0.5,
        cosine_weight=1.0,
    )

    # Dummy inputs (batch=1, seq=2, vocab=4)
    # Note: Shapes must align for broadcasting
    student_logits = jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]) * 10
    teacher_logits = jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]) * 10

    # Labels must be One-Hot Encoded to match logits shape (1, 2, 4)
    labels_indices = jnp.array([[0, 1]])
    labels = jax.nn.one_hot(labels_indices, 4)

    # Run calculation
    _, metrics = strategy.compute_loss(student_logits, teacher_logits, labels)

    # Verify structure
    self.assertIsInstance(metrics, dict)

    # Check keys required for TensorBoard
    expected_keys = ["distill/soft_loss", "distill/hard_loss", "distill/kl_div", "distill/teacher_loss"]
    for key in expected_keys:
      self.assertIn(key, metrics)

    # Since inputs match perfectly, KL should be near 0
    self.assertLess(metrics["distill/kl_div"], 1e-5)

  def test_strategy_compute_eval_loss(self):
    """Covers MonitoredLogitStrategy.compute_eval_loss."""
    strategy = distillation_utils.MonitoredLogitStrategy(
        student_forward_fn=mock.Mock(), teacher_forward_fn=mock.Mock(), labels_fn=mock.Mock(), temperature=1.0, alpha=0.5
    )
    logits = jnp.array([[[10.0, 0.0]]])
    labels = jnp.array([[[1.0, 0.0]]])

    loss, aux = strategy.compute_eval_loss(logits, labels)
    self.assertTrue(isinstance(loss, jax.Array))
    self.assertEqual(aux, {})

  def test_setup_pipeline_grain_enabled(self):
    """Covers _setup_and_restore_input_pipeline when Grain IS detected."""
    mock_trainer = mock.Mock()
    mock_trainer.checkpoint_manager = mock.Mock()
    # Mock restore returning None (no checkpoint yet)
    mock_trainer.checkpoint_manager.restore_iterator.return_value = None

    mock_iter = mock.Mock()
    mock_iter.save = mock.Mock()  # Has save method

    config = mock.Mock()
    config.dataset_type = "grain"

    # Use real options to avoid Orbax validation errors caused by Mocks
    train_config = mock.Mock()
    train_config.checkpoint_root_directory = self.test_dir
    train_config.checkpointing_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)

    # Run function
    result = train_distill._setup_and_restore_input_pipeline(mock_trainer, mock_iter, config, train_config)

    # Verify manager was swapped
    self.assertIsInstance(mock_trainer.checkpoint_manager, distillation_utils.MaxTextCheckpointManager)
    self.assertEqual(result, mock_iter)

  def test_setup_pipeline_restored(self):
    """Covers _setup_and_restore_input_pipeline when restore succeeds."""
    mock_trainer = mock.Mock()

    # Mock successful restore
    restored_iter = mock.Mock()
    mock_manager = mock.Mock()
    mock_manager.restore_iterator.return_value = restored_iter

    # We need to mock the constructor of MaxTextCheckpointManager to return our mock
    with mock.patch(
        "maxtext.trainers.post_train.distillation.distillation_utils.MaxTextCheckpointManager", return_value=mock_manager
    ):

      mock_iter = mock.Mock()
      mock_iter.save = mock.Mock()
      config = mock.Mock()
      config.dataset_type = "grain"

      # Use real options
      train_config = mock.Mock()
      train_config.checkpoint_root_directory = self.test_dir
      train_config.checkpointing_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)

      result = train_distill._setup_and_restore_input_pipeline(mock_trainer, mock_iter, config, train_config)

      # Verify it returned the restored iterator, NOT the raw one
      self.assertEqual(result, restored_iter)

  def test_setup_pipeline_disabled(self):
    """Covers _setup_and_restore_input_pipeline when checkpoiting is disabled."""
    mock_trainer = mock.Mock()
    mock_iter = object()  # No save method

    config = mock.Mock()
    config.dataset_type = "tfds"  # Not grain

    # Use real options
    train_config = mock.Mock()
    train_config.checkpoint_root_directory = self.test_dir
    train_config.checkpointing_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)

    result = train_distill._setup_and_restore_input_pipeline(mock_trainer, mock_iter, config, train_config)

    # Should still swap manager (to MaxTextCheckpointManager) but with None iterator
    self.assertIsInstance(mock_trainer.checkpoint_manager, distillation_utils.MaxTextCheckpointManager)
    # Result should be original iterator
    self.assertEqual(result, mock_iter)

  def test_post_process_train_step(self):
    """Verifies metrics are moved from aux dict to the trainer buffer."""
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)

    # Setup MetricsBuffer mock
    mock_buffer = mock.Mock()
    mock_buffer.additional_metrics = {}
    trainer._buffered_train_metrics = mock_buffer

    # Simulate auxiliary output from strategy
    aux_metrics = {
        "distill/kl_div": jnp.array(0.5),
        "distill/soft_loss": jnp.array(1.2),
        "distill/cosine_loss": jnp.array(0.01),
    }

    # Run Hook
    trainer._post_process_train_step(aux_metrics)

    # Verify buffer updated
    self.assertIn("distill/kl_div", mock_buffer.additional_metrics)
    self.assertIn("distill/soft_loss", mock_buffer.additional_metrics)
    self.assertIn("distill/cosine_loss", mock_buffer.additional_metrics)

    # Verify value appended to list
    values_list = mock_buffer.additional_metrics["distill/kl_div"][0]
    self.assertEqual(values_list[0], 0.5)


if __name__ == "__main__":
  absltest.main()
