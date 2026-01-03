# Copyright 2023–2025 Google LLC
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

import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

# Import the module under test
from MaxText.distillation import train_distill


# pylint: disable=protected-access
class TrainDistillTest(unittest.TestCase):

  def test_compute_debug_metrics_logic(self):
    """Verifies that the JIT-compiled metric calculation is mathematically correct."""
    batch_size, seq_len, vocab_size = 2, 4, 10

    # 1. Setup Dummy Data
    key = jax.random.PRNGKey(0)
    k1, _ = jax.random.split(key)

    student_logits = jax.random.normal(k1, (batch_size, seq_len, vocab_size))
    # Make teacher logits identical to check soft loss behavior
    teacher_logits = student_logits

    # Create targets (indices)
    targets = jnp.array([[1, 2, 0, 0], [3, 4, 5, 0]])  # 0 is padding

    # 2. Test Alpha = 0.0 (Only Hard Loss)
    # Mask: [[1, 1, 0, 0], [1, 1, 1, 0]] -> 5 valid tokens
    metrics_hard = train_distill._compute_debug_metrics(
        student_logits, teacher_logits, targets, temperature=1.0, alpha=0.0
    )

    self.assertTrue(jnp.isfinite(metrics_hard["hard_loss"]))
    # Soft loss is calculated but ignored in total_proxy when alpha=0
    self.assertTrue(jnp.isfinite(metrics_hard["soft_loss"]))
    # Correct assertion: Total Proxy should equal Hard Loss
    self.assertAlmostEqual(metrics_hard["total_proxy"], metrics_hard["hard_loss"], delta=1e-5)

    # 3. Test Alpha = 1.0 (Only Soft Loss)
    # Since student == teacher, soft loss should be minimized (entropy of target)
    metrics_soft = train_distill._compute_debug_metrics(
        student_logits, teacher_logits, targets, temperature=1.0, alpha=1.0
    )

    self.assertTrue(jnp.isfinite(metrics_soft["soft_loss"]))
    self.assertAlmostEqual(metrics_soft["total_proxy"], metrics_soft["soft_loss"], delta=1e-5)

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
    adapter = train_distill.MaxTextToTunixIterator(dummy_iter)

    # 3. Fetch Batch
    tunix_input = next(adapter)

    # 4. Verify Fields
    self.assertIsInstance(tunix_input, train_distill.MaxTextTrainingInput)
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
        # 'inputs_segmentation' is missing
    }
    dummy_iter = iter([dummy_batch])
    adapter = train_distill.MaxTextToTunixIterator(dummy_iter)

    tunix_input = next(adapter)

    # Should default to all True mask and None segment ids
    self.assertIsNone(tunix_input.decoder_segment_ids)
    self.assertTrue(np.all(tunix_input.input_mask))

  def test_prepare_inputs_logic(self):
    """Verifies the filtering logic in the custom trainer."""

    # 1. Initialize Trainer without calling parent __init__
    # We use __new__ to bypass the complex parent initialization logic
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)

    # Manually setup attributes expected by the method
    trainer.log_period = 10
    trainer.debug_mode = False
    trainer._log_step_counter = 0
    trainer._mode = "train"

    # Mock the strategy and model
    trainer.strategy = mock.Mock()
    trainer.teacher_model = mock.Mock()
    trainer.model = mock.Mock()  # Student

    # 2. Setup Input Data
    # pylint: disable=unexpected-keyword-arg
    input_data = train_distill.MaxTextTrainingInput(
        input_tokens=jnp.array([[1]]),
        input_mask=jnp.array([[True]]),
        positions=jnp.array([[0]]),
        targets=jnp.array([[1]]),
    )

    # 3. Setup Helper Mocks
    # Mock gen_model_input_fn to simulate standard Tunix behavior
    trainer.gen_model_input_fn = lambda x: {"inputs": {"some_key": "some_val"}}

    # Mock the strategy output
    fake_teacher_logits = jnp.zeros((1, 1, 10))
    trainer.strategy.get_teacher_outputs.return_value = fake_teacher_logits

    # 4. Run Method
    # We are testing the Trainer's _prepare_inputs method we overrode
    result = trainer._prepare_inputs(input_data)

    # 5. Assertions
    # Teacher should be called
    trainer.strategy.get_teacher_outputs.assert_called_once()

    # Result should have teacher_output populated
    self.assertIsNotNone(result.teacher_output)
    self.assertEqual(result.teacher_output.shape, (1, 1, 10))

    # Verify pass-through fields
    self.assertIs(result.targets, input_data.targets)


if __name__ == "__main__":
  absltest.main()
