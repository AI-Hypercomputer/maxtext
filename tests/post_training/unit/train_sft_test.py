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

"""Unit tests for train_sft.py."""

import unittest
from unittest import mock
from types import SimpleNamespace
import pytest

from maxtext.trainers.post_train.sft import train_sft

pytestmark = [pytest.mark.post_training]


class TrainSFTTest(unittest.TestCase):
  """Tests for train_sft.py."""

  @pytest.mark.cpu_only
  def test_validate_config_valid(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
    )
    # Should not raise any exception
    train_sft.validate_config(config)

  @pytest.mark.cpu_only
  def test_validate_config_invalid_offload(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=True,
    )
    with self.assertRaisesRegex(ValueError, "optimizer_memory_host_offload=True is not supported"):
      train_sft.validate_config(config)

  @pytest.mark.cpu_only
  def test_train_model_caching_moe(self):
    """Test that NNX graph caching is disabled for MoE models (num_experts > 1)."""
    mt_config = SimpleNamespace(
        logical_axis_rules=[],
        num_experts=8,
    )
    trainer = mock.MagicMock()
    trainer.data_hooks.train_data_iterator = "train_iter"
    trainer.data_hooks.eval_data_iterator = "eval_iter"
    mesh = mock.MagicMock()

    with mock.patch("jax.set_mesh"):
      train_sft.train_model(mt_config, trainer, mesh)

    trainer.train.assert_called_once_with(
        "train_iter",
        "eval_iter",
        cache_nnx_graph=False,
    )

  @pytest.mark.cpu_only
  def test_train_model_caching_dense(self):
    """Test that NNX graph caching is enabled for dense models (num_experts <= 1)."""
    mt_config = SimpleNamespace(
        logical_axis_rules=[],
        num_experts=1,
    )
    trainer = mock.MagicMock()
    trainer.data_hooks.train_data_iterator = "train_iter"
    trainer.data_hooks.eval_data_iterator = "eval_iter"
    mesh = mock.MagicMock()

    with mock.patch("jax.set_mesh"):
      train_sft.train_model(mt_config, trainer, mesh)

    trainer.train.assert_called_once_with(
        "train_iter",
        "eval_iter",
        cache_nnx_graph=True,
    )


if __name__ == "__main__":
  unittest.main()
