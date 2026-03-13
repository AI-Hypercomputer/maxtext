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

import pytest

pytest.importorskip("tunix")
pytestmark = [pytest.mark.tpu_only]

import shutil
import tempfile
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl.testing import absltest

# Import the module under test
from maxtext.trainers.post_train.distillation import train_distill
from maxtext.trainers.post_train.distillation import distillation_utils
from maxtext.configs import pyconfig


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

  def test_maxtext_to_tunix_iterator_sft(self):
    """Verifies SFT-related fields are handled correctly."""
    # 1. Create a dummy batch with SFT fields
    dummy_batch_sft = {
        "inputs": np.array([[10, 11]]),
        "inputs_position": np.array([[0, 1]]),
        "targets": np.array([[11, 12]]),
        "targets_position": np.array([[100, 101]]),  # Custom position
        "targets_segmentation": np.array([[0, 1]]),  # Custom segmentation (mask)
    }
    dummy_iter_sft = iter([dummy_batch_sft])

    # 2. Initialize Adapter and get output
    adapter_sft = distillation_utils.MaxTextToTunixIterator(dummy_iter_sft)
    tunix_input_sft = next(adapter_sft)

    # 3. Verify SFT fields are passed through
    self.assertIsInstance(tunix_input_sft, distillation_utils.MaxTextTrainingInput)
    np.testing.assert_array_equal(tunix_input_sft.targets_position, dummy_batch_sft["targets_position"])
    np.testing.assert_array_equal(tunix_input_sft.targets_segmentation, dummy_batch_sft["targets_segmentation"])

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
    """Verifies the filtering in the custom trainer."""
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

    # 4. Run
    _ = trainer._prepare_inputs(input_data)

    # 5. Verify
    trainer.strategy.get_teacher_outputs.assert_not_called()

  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.jax.tree.map")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.nnx.value_and_grad")
  def test_train_step_skips_teacher_forward_when_output_present(self, mock_value_and_grad, mock_tree_map):
    """Verifies teacher forward is skipped when model_output is already in the batch."""
    # 1. Initialize Trainer
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)
    trainer.strategy = mock.Mock()

    # 2. Setup Batch WITH teacher_output
    mock_batch = {
        "input_tokens": mock.Mock(),
        "positions": mock.Mock(),
        "attention_mask": mock.Mock(),
        "decoder_segment_ids": mock.Mock(),
        "targets": mock.Mock(),
        "teacher_output": mock.Mock(),  # Present!
    }
    trainer.gen_model_input_fn = mock.Mock(return_value=mock_batch)

    # 3. Setup Models & Inputs
    teacher_model, student_model = mock.Mock(), mock.Mock()
    model_bundle = train_distill.ModelBundle(teacher_model=teacher_model, student_model=student_model)
    optimizer, inputs = mock.Mock(), mock.Mock()

    # 4. Configure mocked nnx.value_and_grad
    mock_loss, mock_aux, mock_grads = mock.Mock(), mock.Mock(), mock.Mock()
    mock_grad_fn = mock.Mock(return_value=((mock_loss, mock_aux), mock_grads))
    mock_value_and_grad.return_value = mock_grad_fn

    # 5. Execute outer function & trigger inner loss_wrapper
    trainer._train_step(model_bundle, optimizer, inputs)
    loss_wrapper = mock_value_and_grad.call_args[0][0]
    loss_wrapper(student_model, teacher_model, mock_batch)

    # 6. Assertions
    trainer.strategy.teacher_forward_fn.assert_not_called()
    trainer.strategy.student_forward_fn.assert_called_once_with(
        model=student_model,
        input_tokens=mock_batch["input_tokens"],
        positions=mock_batch["positions"],
        attention_mask=mock_batch["attention_mask"],
        decoder_segment_ids=mock_batch["decoder_segment_ids"],
        decoder_target_tokens=mock_batch.get("targets", None),
        decoder_target_mask=mock_batch.get("targets_segmentation", None),
        cache=None,
    )

  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.jax.tree.map")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.nnx.value_and_grad")
  def test_train_step_calls_teacher_forward_when_output_missing(self, mock_value_and_grad, mock_tree_map):
    """Verifies teacher forward is called when model_output is missing from the batch."""
    # 1. Initialize Trainer
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)
    trainer.strategy = mock.Mock()

    # 2. Setup Batch WITHOUT teacher_output
    mock_batch = {
        "input_tokens": mock.Mock(),
        "positions": mock.Mock(),
        "attention_mask": mock.Mock(),
        "decoder_segment_ids": mock.Mock(),
        "targets": mock.Mock(),
        # teacher_output is purposely missing here
    }
    trainer.gen_model_input_fn = mock.Mock(return_value=mock_batch)

    # 3. Setup Models & Inputs
    teacher_model, student_model = mock.Mock(), mock.Mock()
    model_bundle = train_distill.ModelBundle(teacher_model=teacher_model, student_model=student_model)
    optimizer, inputs = mock.Mock(), mock.Mock()

    # 4. Configure mocked nnx.value_and_grad
    mock_loss, mock_aux, mock_grads = mock.Mock(), mock.Mock(), mock.Mock()
    mock_grad_fn = mock.Mock(return_value=((mock_loss, mock_aux), mock_grads))
    mock_value_and_grad.return_value = mock_grad_fn

    # 5. Execute outer function & trigger inner loss_wrapper
    loss, aux = trainer._train_step(model_bundle, optimizer, inputs)
    loss_wrapper = mock_value_and_grad.call_args[0][0]
    loss_wrapper(student_model, teacher_model, mock_batch)

    # 6. Assertions
    trainer.strategy.teacher_forward_fn.assert_called_once_with(
        model=teacher_model,
        input_tokens=mock_batch["input_tokens"],
        positions=mock_batch["positions"],
        attention_mask=mock_batch["attention_mask"],
        decoder_segment_ids=mock_batch["decoder_segment_ids"],
        decoder_target_tokens=mock_batch.get("targets", None),
        cache=None,
        decoder_target_mask=None,
    )

    trainer.strategy.student_forward_fn.assert_called_once_with(
        model=student_model,
        input_tokens=mock_batch["input_tokens"],
        positions=mock_batch["positions"],
        attention_mask=mock_batch["attention_mask"],
        decoder_segment_ids=mock_batch["decoder_segment_ids"],
        decoder_target_tokens=mock_batch.get("targets", None),
        cache=None,
        decoder_target_mask=None,
    )

    # Verify loss computation and optimizer update
    trainer.strategy.labels_fn.assert_called_once_with(mock_batch["targets"], targets_segmentation=None)
    trainer.strategy.compute_loss.assert_called_once()
    optimizer.update.assert_called_once_with(student_model, mock_grads)

    # Verify the final returns match what grad_fn produced
    self.assertEqual(loss, mock_loss)
    self.assertEqual(aux, mock_aux)

  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.jax.tree.map")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.nnx.value_and_grad")
  def test_train_step_passes_targets_segmentation(self, mock_value_and_grad, mock_tree_map):
    """Verifies strategy callbacks receive decoder_target_tokens and decoder_target_mask."""
    # 1. Initialize Trainer
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)
    trainer.strategy = mock.Mock()

    # 2. Setup Batch WITH targets_segmentation
    mock_targets_segmentation = jnp.array([[1, 1, 0]])
    mock_batch = {
        "input_tokens": mock.Mock(),
        "positions": mock.Mock(),
        "attention_mask": mock.Mock(),
        "decoder_segment_ids": mock.Mock(),
        "targets": mock.Mock(),
        "targets_segmentation": mock_targets_segmentation,
    }
    trainer.gen_model_input_fn = mock.Mock(return_value=mock_batch)

    # 3. Setup Models & Inputs
    teacher_model, student_model = mock.Mock(), mock.Mock()
    model_bundle = train_distill.ModelBundle(teacher_model=teacher_model, student_model=student_model)
    optimizer, inputs = mock.Mock(), mock.Mock()

    # 4. Configure mocked nnx.value_and_grad
    mock_grad_fn = mock.Mock(return_value=((mock.Mock(), mock.Mock()), mock.Mock()))
    mock_value_and_grad.return_value = mock_grad_fn

    # 5. Execute outer function & trigger inner loss_wrapper
    trainer._train_step(model_bundle, optimizer, inputs)
    loss_wrapper = mock_value_and_grad.call_args[0][0]
    loss_wrapper(student_model, teacher_model, mock_batch)

    # 6. Assertions
    trainer.strategy.labels_fn.assert_called_once_with(
        mock_batch["targets"], targets_segmentation=mock_targets_segmentation
    )
    trainer.strategy.student_forward_fn.assert_called_once_with(
        model=student_model,
        input_tokens=mock_batch["input_tokens"],
        positions=mock_batch["positions"],
        attention_mask=mock_batch["attention_mask"],
        decoder_segment_ids=mock_batch["decoder_segment_ids"],
        decoder_target_tokens=mock_batch["targets"],
        decoder_target_mask=mock_targets_segmentation,
        cache=None,
    )
    trainer.strategy.teacher_forward_fn.assert_called_once_with(
        model=teacher_model,
        input_tokens=mock_batch["input_tokens"],
        positions=mock_batch["positions"],
        attention_mask=mock_batch["attention_mask"],
        decoder_segment_ids=mock_batch["decoder_segment_ids"],
        decoder_target_tokens=mock_batch["targets"],
        decoder_target_mask=mock_targets_segmentation,
        cache=None,
    )

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
    self._test_monitored_strategy(False)

  def test_monitored_strategy_sft(self):
    self._test_monitored_strategy(True)

  def _test_monitored_strategy(self, sft_mode: bool):
    """Verifies the strategy calculates metrics and returns the correct tuple."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        labels_fn=lambda t: t,
        temperature=1.0,
        alpha=0.5,
        beta_feature=1.0,
        layer_indices=None,
        sft_mode=sft_mode,
    )

    # Dummy inputs (batch=1, seq=2, vocab=4)
    # Note: Shapes must align for broadcasting
    student_output = distillation_utils.DistillationForwardOutput(
        logits=jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]) * 10,
        out_projection_activations=jnp.ones((32, 1, 1, 8)),
    )
    teacher_output = distillation_utils.DistillationForwardOutput(
        logits=jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]) * 10,
        out_projection_activations=jnp.ones((32, 1, 1, 8)),
    )

    # Labels must be One-Hot Encoded to match logits shape (1, 2, 4)
    labels_indices = jnp.array([[0, 1]])
    labels = jax.nn.one_hot(labels_indices, 4)

    # Run calculation
    _, metrics = strategy.compute_loss(student_output, teacher_output, labels)

    # Verify structure
    self.assertIsInstance(metrics, dict)

    # Check keys required for TensorBoard
    expected_keys = [
        "distill/soft_loss",
        "distill/hard_loss",
        "distill/kl_div",
        "distill/teacher_loss",
        "distill/out_proj_feature_loss",
        "distill/total_loss",
    ]
    for key in expected_keys:
      self.assertIn(key, metrics)

    # Since inputs match perfectly, KL, feature loss should be near 0
    self.assertLess(metrics["distill/kl_div"], 1e-5)
    self.assertLess(metrics["distill/out_proj_feature_loss"], 1e-5)

  def test_strategy_compute_eval_loss(self):
    self._verify_strategy_compute_eval_loss(sft_mode=False)

  def _verify_strategy_compute_eval_loss(self, sft_mode):
    """Covers MonitoredLogitStrategy.compute_eval_loss."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=mock.Mock(),
        teacher_forward_fn=mock.Mock(),
        labels_fn=mock.Mock(),
        temperature=1.0,
        alpha=0.5,
        sft_mode=sft_mode,
    )
    # Case where feature loss is enabled
    logits = distillation_utils.DistillationForwardOutput(
        logits=jnp.array([[[10.0, 0.0]]]), out_projection_activations=np.ones((32, 1, 1, 8))
    )
    labels = jnp.array([[[1.0, 0.0]]])

    loss, aux = strategy.compute_eval_loss(logits, labels)
    self.assertTrue(isinstance(loss, jax.Array))
    self.assertEqual(aux, {})

    # Case where feature loss is disabled.
    logits = distillation_utils.DistillationForwardOutput(
        logits=jnp.array([[[10.0, 0.0]]]), out_projection_activations=None
    )
    labels = jnp.array([[[1.0, 0.0]]])

    loss, aux = strategy.compute_eval_loss(logits, labels)
    self.assertTrue(isinstance(loss, jax.Array))
    self.assertEqual(aux, {})

  def test_strategy_compute_eval_loss_sft(self):
    self._verify_strategy_compute_eval_loss(sft_mode=True)

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

  def test_eval_step_calls_student_forward(self):
    """Verifies eval step correctly calls the student forward function and computes loss."""
    # 1. Initialize Trainer (bypass init)
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)
    trainer.strategy = mock.Mock()

    # 2. Setup Input Mocks
    raw_inputs = mock.Mock()
    mock_batch = {
        "input_tokens": mock.Mock(),
        "positions": mock.Mock(),
        "attention_mask": mock.Mock(),
        "decoder_segment_ids": mock.Mock(),
        "targets": mock.Mock(),
    }
    trainer.gen_model_input_fn = mock.Mock(return_value=mock_batch)

    # 3. Setup Model Mocks
    model_bundle = mock.Mock()
    student_model = mock.Mock()
    model_bundle.student_model = student_model

    # Setup return values for the strategy functions to track the data flow
    mock_student_output = mock.Mock()
    trainer.strategy.student_forward_fn.return_value = mock_student_output

    mock_labels = mock.Mock()
    trainer.strategy.labels_fn.return_value = mock_labels

    mock_loss = mock.Mock()
    trainer.strategy.compute_eval_loss.return_value = mock_loss

    # 4. Execute the evaluation step
    actual_loss = trainer._eval_step(model_bundle, raw_inputs)

    # 5. --- ASSERTIONS ---

    # Verify input generation was called with the raw inputs
    trainer.gen_model_input_fn.assert_called_once_with(raw_inputs)

    # Verify student forward was called exactly once with the right kwargs
    trainer.strategy.student_forward_fn.assert_called_once_with(
        model=student_model,
        input_tokens=mock_batch["input_tokens"],
        positions=mock_batch["positions"],
        attention_mask=mock_batch["attention_mask"],
        decoder_segment_ids=mock_batch["decoder_segment_ids"],
        cache=None,
    )

    # Verify that the teacher forward function was NEVER called
    # (Assuming teacher_forward_fn exists on the strategy)
    if hasattr(trainer.strategy, "teacher_forward_fn"):
      trainer.strategy.teacher_forward_fn.assert_not_called()

    # Verify loss computation pipeline
    trainer.strategy.labels_fn.assert_called_once_with(mock_batch["targets"])
    trainer.strategy.compute_eval_loss.assert_called_once_with(mock_student_output, mock_labels)

    # Verify it returns the correct loss
    self.assertEqual(actual_loss, mock_loss)

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
    aux_metrics = {"distill/kl_div": jnp.array(0.5), "distill/soft_loss": jnp.array(1.2)}

    # Run Hook
    trainer._post_process_train_step(aux_metrics)

    # Verify buffer updated
    self.assertIn("distill/kl_div", mock_buffer.additional_metrics)
    self.assertIn("distill/soft_loss", mock_buffer.additional_metrics)

    # Verify value appended to list
    values_list = mock_buffer.additional_metrics["distill/kl_div"][0]
    self.assertEqual(values_list[0], 0.5)

  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.distillation_utils.OfflineArrayRecordIterator")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.MaxTextDistillationTrainer")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.input_pipeline_interface.create_data_iterator")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.get_maxtext_model")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.tokenizer.build_tokenizer")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.maxtext_utils.create_device_mesh")
  @mock.patch("maxtext.configs.pyconfig.initialize")
  def test_main_offline_mode_skips_teacher_loading(
      self,
      mock_pyconfig_init,
      mock_create_mesh,
      mock_build_tokenizer,
      mock_get_model,
      mock_create_iterator,
      mock_trainer_cls,
      mock_offline_iter_cls,
  ):
    """Verifies offline mode (offline_data_dir is set) skips teacher model loading."""
    # 1. Configs
    mock_global = mock.Mock()
    mock_global.student_overrides = {}
    mock_global.teacher_overrides = {}  # No checkpoint needed
    mock_global.offline_data_dir = "gs://bucket/data"  # Triggers offline mode

    mock_student_cfg = mock.Mock()
    mock_student_cfg.vocab_size = 32000
    mock_student_cfg.mesh_axes = ("data",)
    mock_student_cfg.dataset_type = "grain"

    # Add dummy numbers for optimizer math
    mock_student_cfg.learning_rate = 1e-4
    mock_student_cfg.warmup_steps_fraction = 0.1
    mock_student_cfg.learning_rate_final_fraction = 0.1
    mock_student_cfg.steps = 100
    mock_student_cfg.checkpoint_period = 10
    mock_student_cfg.gradient_clipping_threshold = 0.0
    mock_student_cfg.eval_interval = -1

    # Add dummy numbers for strategy math/logic
    mock_student_cfg.distill_temperature = 1.0
    mock_student_cfg.distill_alpha = 0.5
    mock_student_cfg.distill_beta = 0.0
    mock_student_cfg.distill_layer_indices = None
    mock_student_cfg.use_sft = False
    mock_student_cfg.enable_dropout = False

    # Add dummy variables for Checkpointer and Logger
    mock_student_cfg.max_num_checkpoints_to_keep = 1
    mock_student_cfg.async_checkpointing = False
    mock_student_cfg.profiler = "none"
    mock_student_cfg.tensorboard_dir = ""
    mock_student_cfg.checkpoint_dir = ""
    mock_student_cfg.log_period = 10
    mock_student_cfg.save_checkpoint_on_completion = False
    mock_student_cfg.logical_axis_rules = []

    mock_teacher_cfg = mock.Mock()
    mock_teacher_cfg.vocab_size = 32000
    mock_pyconfig_init.side_effect = [mock_global, mock_student_cfg, mock_teacher_cfg]

    # 2. Model Loading
    mock_student_model = mock.Mock()
    mock_get_model.return_value = mock_student_model

    # 3. Tokenizer & Data Iterator
    mock_build_tokenizer.return_value = mock.Mock(pad_id=0)
    mock_create_iterator.return_value = (None, None)

    train_distill.main(["train_distill.py", "config.yml"])

    # 4. Assertions
    # checking to ensure get_maxtext_model is only called once for student and not for teacher
    mock_get_model.assert_called_once_with(mock_student_cfg, mock.ANY)

    trainer_init_kwargs = mock_trainer_cls.call_args.kwargs
    model_bundle = trainer_init_kwargs["model"]
    # check that student model is set but teacher model is None since offline mode should skip loading teacher
    self.assertIs(model_bundle.student_model, mock_student_model)
    self.assertIsNone(model_bundle.teacher_model)

  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.MaxTextDistillationTrainer")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.input_pipeline_interface.create_data_iterator")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.get_maxtext_model")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.tokenizer.build_tokenizer")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.maxtext_utils.create_device_mesh")
  @mock.patch("maxtext.configs.pyconfig.initialize")
  def test_main_online_mode_loads_teacher(
      self,
      mock_pyconfig_init,
      mock_create_mesh,
      mock_build_tokenizer,
      mock_get_model,
      mock_create_iterator,
      mock_trainer_cls,
  ):
    """Verifies online mode (offline_data_dir is None) loads both student and teacher models."""
    mock_global = mock.Mock()
    mock_global.student_overrides = {}
    mock_global.teacher_overrides = {"load_parameters_path": "gs://ckpt"}
    mock_global.offline_data_dir = None  # Triggers online mode

    mock_student_cfg = mock.Mock()
    mock_student_cfg.vocab_size = 32000
    mock_student_cfg.mesh_axes = ("data",)
    mock_student_cfg.dataset_type = "grain"

    # Add dummy numbers for optimizer math
    mock_student_cfg.learning_rate = 1e-4
    mock_student_cfg.warmup_steps_fraction = 0.1
    mock_student_cfg.learning_rate_final_fraction = 0.1
    mock_student_cfg.steps = 100
    mock_student_cfg.checkpoint_period = 10
    mock_student_cfg.gradient_clipping_threshold = 0.0
    mock_student_cfg.eval_interval = -1

    # Add dummy numbers for strategy math/logic
    mock_student_cfg.distill_temperature = 1.0
    mock_student_cfg.distill_alpha = 0.5
    mock_student_cfg.distill_beta = 0.0
    mock_student_cfg.distill_layer_indices = None
    mock_student_cfg.use_sft = False
    mock_student_cfg.enable_dropout = False

    # Add dummy variables for Checkpointer and Logger
    mock_student_cfg.max_num_checkpoints_to_keep = 1
    mock_student_cfg.async_checkpointing = False
    mock_student_cfg.profiler = "none"
    mock_student_cfg.tensorboard_dir = ""
    mock_student_cfg.checkpoint_dir = ""
    mock_student_cfg.log_period = 10
    mock_student_cfg.save_checkpoint_on_completion = False
    mock_student_cfg.logical_axis_rules = []

    mock_teacher_cfg = mock.Mock()
    mock_teacher_cfg.vocab_size = 32000
    mock_pyconfig_init.side_effect = [mock_global, mock_student_cfg, mock_teacher_cfg]

    mock_student_model = mock.Mock()
    mock_teacher_model = mock.Mock()
    mock_get_model.side_effect = [mock_student_model, mock_teacher_model]

    mock_build_tokenizer.return_value = mock.Mock(pad_id=0)
    mock_create_iterator.return_value = (mock.Mock(), mock.Mock())

    train_distill.main(["train_distill.py", "config.yml"])

    # checking to ensure get_maxtext_model is called for both student and teacher since online mode should load both
    self.assertEqual(mock_get_model.call_count, 2)
    mock_get_model.assert_any_call(mock_student_cfg, mock.ANY)
    mock_get_model.assert_any_call(mock_teacher_cfg, mock.ANY)

    trainer_init_kwargs = mock_trainer_cls.call_args.kwargs
    model_bundle = trainer_init_kwargs["model"]
    # check that both student and teacher models are set since online mode should load both
    self.assertIs(model_bundle.student_model, mock_student_model)
    self.assertIs(model_bundle.teacher_model, mock_teacher_model)
  def test_gradient_accumulation_requires_k_passes_for_update(self):
    """Verifies that weights only update after k distinct forward passes."""

    # 1. Setup a minimal NNX model
    class DummyModel(nnx.Module):

      def __init__(self):
        self.linear = nnx.Linear(in_features=2, out_features=2, rngs=nnx.Rngs(0))

      def __call__(self, x):
        return self.linear(x)

    student = DummyModel()
    teacher = DummyModel()
    model_bundle = train_distill.ModelBundle(teacher_model=teacher, student_model=student)

    # Snapshot the initial weights
    initial_weights = student.linear.kernel.value.copy()

    # 2. Setup Optimizer with MultiSteps (Accumulate over 2 passes)
    base_optimizer = optax.sgd(learning_rate=0.1)
    accumulating_optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=2)
    nnx_opt = nnx.Optimizer(student, accumulating_optimizer, wrt=nnx.Param)

    # 3. Initialize Trainer and Mocks
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)
    trainer.strategy = mock.Mock()

    dummy_batch = {
        "input_tokens": jnp.ones((1, 2)),
        "positions": None,
        "targets": None,
        "teacher_output": jnp.array([1.0, 1.0]),
    }
    trainer.gen_model_input_fn = mock.Mock(return_value=dummy_batch)
    trainer.strategy.labels_fn.return_value = None

    # 4. Mock the forward pass to COUNT how many times it executes
    # We wrap the actual dummy model execution in a mock to track it.
    mock_student_forward = mock.Mock(side_effect=lambda model, **kwargs: model(dummy_batch["input_tokens"]))
    trainer.strategy.student_forward_fn = mock_student_forward

    trainer.strategy.compute_loss.side_effect = lambda s_out, t_out, labels: (jnp.sum(s_out), {"aux": 1.0})

    # --- EXECUTE PASS 1 ---
    trainer._train_step(model_bundle, nnx_opt, dummy_batch)

    # ASSERTIONS AFTER PASS 1:
    # Verify exactly ONE forward pass happened
    self.assertEqual(mock_student_forward.call_count, 1)

    # Verify weights are completely UNCHANGED
    np.testing.assert_allclose(
        student.linear.kernel.value, initial_weights, err_msg="Weights should not update on the first pass."
    )

    # --- EXECUTE PASS 2 ---
    trainer._train_step(model_bundle, nnx_opt, dummy_batch)

    # ASSERTIONS AFTER PASS 2:
    # Verify exactly TWO forward passes have now happened
    self.assertEqual(mock_student_forward.call_count, 2)

    # Verify weights HAVE changed
    with self.assertRaises(AssertionError, msg="Weights should have updated on the second pass."):
      np.testing.assert_allclose(student.linear.kernel.value, initial_weights)


if __name__ == "__main__":
  absltest.main()
