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

  @pytest.mark.cpu_only
  def test_validate_config_valid(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
        num_vocab_tiling=1,
    )
    # Should not raise any exception
    train_dpo.validate_config(config)

  @pytest.mark.cpu_only
  def test_validate_config_invalid_offload(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=True,
        num_vocab_tiling=1,
    )
    with self.assertRaisesRegex(ValueError, "optimizer_memory_host_offload=True is not supported"):
      train_dpo.validate_config(config)

  @pytest.mark.cpu_only
  def test_validate_config_invalid_vocab_tiling(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
        num_vocab_tiling=2,
    )
    with self.assertRaisesRegex(ValueError, "Vocab Tiling is not supported with DPO"):
      train_dpo.validate_config(config)


if __name__ == "__main__":
  unittest.main()
