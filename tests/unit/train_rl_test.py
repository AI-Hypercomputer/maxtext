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

"""Unit tests for train_rl.py."""

import unittest
from unittest import mock
import pytest
from types import SimpleNamespace
import jax


# Same as in rl_utils_test.py.
train_rl = pytest.importorskip(
    "maxtext.trainers.post_train.rl.train_rl",
    reason="Tunix is not installed on the GPU image",
)


def _get_mock_devices(num_devices):
  mock_devices = [mock.MagicMock() for _ in range(num_devices)]
  for i, d in enumerate(mock_devices):
    d.id = i
  return mock_devices


class TrainRLTest(unittest.TestCase):
  """Tests for train_rl.py."""

  @pytest.mark.cpu_only
  def test_setup_configs_and_devices_pathways_split(self):
    """Test setup_configs_and_devices with multiple VMs and Pathways."""
    mock_devices = _get_mock_devices(8)

    mock_config = SimpleNamespace(
        num_trainer_slices=-1,
        num_samplers_slices=-1,
        chips_per_vm=4,
        use_pathways=True,
        trainer_devices_fraction=0.5,
        sampler_devices_fraction=0.5,
    )

    # Following the pattern in distillation_checkpointing_test.py for mocking jax objects
    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch("maxtext.trainers.post_train.rl.train_rl.pyconfig.initialize_pydantic", return_value=mock_config),
    ):
      trainer_config, sampler_config, trainer_devices, sampler_devices = train_rl.setup_configs_and_devices(
          ["dummy", "dummy"]
      )

      self.assertEqual(trainer_config, mock_config)
      self.assertEqual(sampler_config, mock_config)
      self.assertEqual(len(trainer_devices), 4)
      self.assertEqual(len(sampler_devices), 4)
      self.assertEqual(trainer_devices, mock_devices[:4])
      self.assertEqual(sampler_devices, mock_devices[4:])

  @pytest.mark.cpu_only
  def test_setup_configs_and_devices_pathways_fractional_split(self):
    """Test setup_configs_and_devices with multiple VMs and custom fractions."""
    mock_devices = _get_mock_devices(8)

    mock_config = SimpleNamespace(
        num_trainer_slices=-1,
        num_samplers_slices=-1,
        chips_per_vm=4,
        use_pathways=True,
        trainer_devices_fraction=0.25,
        sampler_devices_fraction=0.75,
    )

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch("maxtext.trainers.post_train.rl.train_rl.pyconfig.initialize_pydantic", return_value=mock_config),
    ):
      _, _, trainer_devices, sampler_devices = train_rl.setup_configs_and_devices(["dummy", "dummy"])

      self.assertEqual(len(trainer_devices), 2)
      self.assertEqual(len(sampler_devices), 6)
      self.assertEqual(trainer_devices, mock_devices[:2])
      self.assertEqual(sampler_devices, mock_devices[2:])

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_no_dp(self):
    """Test case 1: sampler_config.rollout_data_parallelism=-1 -> verify result is calculated."""
    # num_sampler_devices=16, tp=2, ep=4 -> dp should be 16 // (2 * 4) = 2
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=4,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 4,
    }
    self.assertEqual(train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 16), expected_result)

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_auto_tp(self):
    """Test case 2: dp=2, tp=-1, num_sampler_devices=4."""
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 1,
    }
    self.assertEqual(train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4), expected_result)

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_fixed_tp_dp(self):
    """Test case 3: dp=2, tp=2, num_sampler_devices=4."""
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 1,
    }
    self.assertEqual(train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4), expected_result)

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_auto_ep(self):
    """Test case 4: ep=-1 -> verify result is calculated."""
    # num_sampler_devices=8, tp=2, dp=2 -> ep should be 8 // (2 * 2) = 2
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=-1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
    }
    self.assertEqual(train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8), expected_result)

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_errors(self):
    """Test various error cases for get_rollout_kwargs_for_parallelism."""
    # More than one -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, "At most one of .* can be -1"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4)

    # num_devices % (tp * ep) != 0 when dp == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=3,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4)

    # num_devices % (tp * dp) != 0 when ep == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=3,
        rollout_expert_parallelism=-1,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)

    # num_devices % (dp * ep) != 0 when tp == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=3,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=2,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)

    # tp * dp * ep != num_sampler_devices when all are positive
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, r"!= len\(sampler_devices\)"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)


if __name__ == "__main__":
  unittest.main()
