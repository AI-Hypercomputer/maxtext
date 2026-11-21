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

import dataclasses
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from jax.sharding import Mesh
import pytest

from maxtext.configs import pyconfig
from maxtext.trainers.post_train.dpo import train_dpo
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

pytestmark = [pytest.mark.post_training]


class TrainDPOTest(unittest.TestCase):
  """Tests for train_dpo.py."""

  def _valid_config(self, **overrides):
    defaults = {
        "optimizer_memory_host_offload": False,
        "num_vocab_tiling": 1,
        "context_parallel_load_balance": True,
        "ici_context_parallelism": 1,
        "dcn_context_parallelism": 1,
        "context_sharding": "context",
    }
    return SimpleNamespace(**{**defaults, **overrides})

  def test_validate_config_valid(self):
    config = self._valid_config()
    # Should not raise any exception
    train_dpo.validate_config(config)

  def test_validate_config_invalid_offload(self):
    config = self._valid_config(optimizer_memory_host_offload=True)
    with self.assertRaisesRegex(ValueError, "optimizer_memory_host_offload=True is not supported"):
      train_dpo.validate_config(config)

  def test_validate_config_invalid_vocab_tiling(self):
    config = self._valid_config(num_vocab_tiling=2)
    with self.assertRaisesRegex(ValueError, "Vocab Tiling is not supported with DPO"):
      train_dpo.validate_config(config)

  def test_validate_config_rejects_load_balanced_cp(self):
    config = self._valid_config(ici_context_parallelism=2, context_parallel_load_balance=True)
    with self.assertRaisesRegex(ValueError, "context_parallel_load_balance=False"):
      train_dpo.validate_config(config)

    dcn_config = self._valid_config(dcn_context_parallelism=2, context_parallel_load_balance=True)
    with self.assertRaisesRegex(ValueError, "context_parallel_load_balance=False"):
      train_dpo.validate_config(dcn_config)

  def test_validate_config_allows_cp_with_load_balance_disabled(self):
    config = self._valid_config(ici_context_parallelism=2, context_parallel_load_balance=False)
    train_dpo.validate_config(config)

  def test_setup_trainer_state_rejects_auto_cp_mesh_with_load_balance_enabled(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
        num_vocab_tiling=1,
        context_parallel_load_balance=True,
        ici_context_parallelism=-1,
        dcn_context_parallelism=1,
        context_sharding="context",
        tokenizer_path="tok",
        tokenizer_type="huggingface",
        hf_access_token="",
        checkpoint_period=10,
        async_checkpointing=False,
        tensorboard_dir="",
        profiler="",
        eval_interval=-1,
        steps=10,
        gradient_accumulation_steps=1,
        checkpoint_dir="",
        dpo=SimpleNamespace(
            algo="dpo",
            orpo_lambda=0.1,
            dpo_beta=0.1,
            dpo_label_smoothing=0.0,
            max_prompt_length=4,
        ),
        max_target_length=8,
    )
    with (
        mock.patch.object(
            train_dpo.maxtext_utils,
            "get_mesh_from_config",
            return_value=SimpleNamespace(shape={"context": 2}),
        ),
        mock.patch.object(train_dpo.tokenizer, "build_tokenizer") as mock_build_tokenizer,
        mock.patch.object(train_dpo.model_creation_utils, "from_pretrained") as mock_from_pretrained,
    ):
      with self.assertRaisesRegex(ValueError, "context_parallel_load_balance=False"):
        train_dpo.setup_trainer_state(config)
      mock_build_tokenizer.assert_not_called()
      mock_from_pretrained.assert_not_called()

  def test_get_tunix_config_uses_input_data_sharding_spec(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      mt_config = pyconfig.initialize_pydantic(
          ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "dpo.yml")],
          per_device_batch_size=1,
          run_name="test",
          base_output_directory=tmp_dir,
          tensorboard_dir=tmp_dir,
          skip_jax_distributed_system=True,
      )
      mesh = Mesh(maxtext_utils.create_device_mesh(mt_config), mt_config.mesh_axes)

      with_mesh = train_dpo.get_tunix_config(mt_config, mesh=mesh)
      without_mesh = train_dpo.get_tunix_config(mt_config)

    self.assertEqual(with_mesh.data_sharding_axis, tuple(sharding.get_input_data_sharding(mt_config, mesh).spec))
    tunix_default = next(
        f.default for f in dataclasses.fields(train_dpo.DPOTrainingConfig) if f.name == "data_sharding_axis"
    )
    self.assertEqual(without_mesh.data_sharding_axis, tunix_default)


if __name__ == "__main__":
  unittest.main()
