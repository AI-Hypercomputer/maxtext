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

"""Unit tests for train_dpo.py."""

import unittest
from types import SimpleNamespace
import pytest

from maxtext.trainers.post_train.dpo import train_dpo

pytestmark = [pytest.mark.post_training]


class TrainDPOTest(unittest.TestCase):
  """Tests for train_dpo.py."""

  def test_validate_config_valid(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
        num_vocab_tiling=1,
    )
    # Should not raise any exception
    train_dpo.validate_config(config)

  def test_validate_config_invalid_offload(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=True,
        num_vocab_tiling=1,
    )
    with self.assertRaisesRegex(ValueError, "optimizer_memory_host_offload=True is not supported"):
      train_dpo.validate_config(config)

  def test_validate_config_invalid_vocab_tiling(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
        num_vocab_tiling=2,
    )
    with self.assertRaisesRegex(ValueError, "Vocab Tiling is not supported with DPO"):
      train_dpo.validate_config(config)


class TrainDPOTunixConfigTest(unittest.TestCase):
  """The Tunix config decides who checkpoints and whether the optimizer gets wrapped."""

  def _mt_config(self, grad_accum=1):
    return SimpleNamespace(
        checkpoint_period=5,
        async_checkpointing=False,
        tensorboard_dir="/tmp/tb",
        profiler="",
        eval_interval=1,
        steps=10,
        checkpoint_dir="/tmp/ckpt",
        data_sharding=["data"],
        gradient_accumulation_steps=grad_accum,
        max_target_length=128,
        dpo=SimpleNamespace(
            algo="dpo",
            orpo_lambda=1.0,
            dpo_beta=0.1,
            dpo_label_smoothing=0.0,
            max_prompt_length=32,
        ),
    )

  @pytest.mark.cpu_only
  def test_tunix_checkpointing_is_disabled(self):
    """post_train.checkpointing owns checkpointing, so Tunix's own manager must stay off."""
    self.assertIsNone(train_dpo.get_tunix_config(self._mt_config()).checkpoint_root_directory)

  @pytest.mark.cpu_only
  def test_single_step_accumulation_is_not_passed_through(self):
    """Tunix wraps the optimizer in MultiSteps whenever this is set, changing the state shape."""
    self.assertIsNone(train_dpo.get_tunix_config(self._mt_config(grad_accum=1)).gradient_accumulation_steps)

  @pytest.mark.cpu_only
  def test_real_accumulation_is_passed_through(self):
    self.assertEqual(train_dpo.get_tunix_config(self._mt_config(grad_accum=4)).gradient_accumulation_steps, 4)


if __name__ == "__main__":
  unittest.main()
