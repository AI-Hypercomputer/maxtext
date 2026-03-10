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


if __name__ == "__main__":
  unittest.main()
