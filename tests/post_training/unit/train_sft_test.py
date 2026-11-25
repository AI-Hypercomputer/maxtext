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

import inspect
import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from absl.testing import absltest
import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
from tunix.sft import sharding_utils as tunix_sharding_utils

from maxtext.common.data_loader import DataLoader
from maxtext.configs import pyconfig
from maxtext.trainers.post_train.sft import train_sft
from maxtext.utils import sharding
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

pytestmark = [pytest.mark.post_training]


class TrainSFTTest(unittest.TestCase):
  """Tests for train_sft.py."""

  # pylint: disable=protected-access

  def test_validate_config_valid(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
    )
    # Should not raise any exception
    train_sft.validate_config(config)

  def test_validate_config_invalid_offload(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=True,
    )
    with self.assertRaisesRegex(ValueError, "optimizer_memory_host_offload=True is not supported"):
      train_sft.validate_config(config)

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

  def test_maxtext_peft_trainer_train_step_signature(self):
    """Test that MaxTextPeftTrainer train_step accepts Tunix args including is_update_step."""
    mock_model = mock.MagicMock()

    with mock.patch("flax.nnx.pop"), mock.patch("flax.nnx.split", return_value=(mock.MagicMock(), {}, {})):
      trainer = mock.MagicMock()
      trainer.loss_fn = mock.MagicMock()
      trainer._has_aux = False
      trainer.gen_model_input_fn = lambda x: x
      trainer._lora_enabled = False
      trainer.model = mock_model

      train_step_fn = train_sft.MaxTextPeftTrainer.create_train_step_fn(trainer)

      # Should accept positional and keyword args from Tunix PeftTrainer
      sig = inspect.signature(train_step_fn)
      params = list(sig.parameters.keys())
      self.assertEqual(params, ["model", "optimizer", "grad_accumulator", "inputs", "is_update_step"])

  def test_use_maxtext_loss_function_eval_uses_is_train_false(self):
    """Test that train loss runs with is_train=True and eval loss runs with is_train=False."""

    class FakeTrainer:
      """Mimics Tunix PeftTrainer.with_loss_fn, which sets both loss_fn and eval_loss_fn."""

      def with_loss_fn(self, loss_fn, has_aux=False):
        self.loss_fn = loss_fn
        self.eval_loss_fn = loss_fn
        self.has_aux = has_aux
        return self

    batch = {
        "inputs": "inputs",
        "inputs_position": "inputs_position",
        "inputs_segmentation": "inputs_segmentation",
        "targets": "targets",
        "targets_position": "targets_position",
        "targets_segmentation": "targets_segmentation",
    }
    mt_config = SimpleNamespace()
    model = mock.MagicMock()

    with mock.patch.object(train_sft, "loss_fn", return_value=(0.0, {})) as mock_loss_fn:
      trainer = train_sft.use_maxtext_loss_function(FakeTrainer(), mt_config)
      self.assertTrue(trainer.has_aux)
      self.assertIsNot(trainer.loss_fn, trainer.eval_loss_fn)

      trainer.loss_fn(model, **batch)
      _, train_kwargs = mock_loss_fn.call_args
      self.assertTrue(train_kwargs["is_train"])

      trainer.eval_loss_fn(model, **batch)
      eval_args, eval_kwargs = mock_loss_fn.call_args
      self.assertFalse(eval_kwargs["is_train"])
      self.assertIs(eval_args[0], model)
      self.assertIs(eval_args[1], mt_config)
      self.assertEqual(eval_args[2], batch)

  def test_train_model_disables_cache_for_diffusion_internal_metrics(self):
    mt_config = SimpleNamespace(
        logical_axis_rules=[],
        num_experts=1,
        training_objective="block_diffusion",
        record_internal_nn_metrics=True,
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

  def test_get_tunix_config_uses_input_data_sharding_spec(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      mt_config = pyconfig.initialize(
          ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")],
          per_device_batch_size=1,
          run_name="test",
          base_output_directory=tmp_dir,
          tensorboard_dir=tmp_dir,
          skip_jax_distributed_system=True,
      )
      mesh = Mesh(np.array(jax.devices()[:1]), ("context",))
      tunix_cfg = train_sft.get_tunix_config(mt_config, mesh=mesh)
      self.assertEqual(tunix_cfg.data_sharding_axis, tuple(sharding.get_input_data_sharding(mt_config, mesh).spec))

      if jax.device_count() >= 4:
        cp_config = pyconfig.initialize(
            ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")],
            per_device_batch_size=1,
            ici_fsdp_parallelism=2,
            ici_context_parallelism=2,
            run_name="test",
            base_output_directory=tmp_dir,
            tensorboard_dir=tmp_dir,
            skip_jax_distributed_system=True,
        )
        cp_mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "context"))
        cp_tunix_cfg = train_sft.get_tunix_config(cp_config, mesh=cp_mesh)
        self.assertEqual(cp_tunix_cfg.data_sharding_axis, ("fsdp", "context"))

  def test_tunix_shard_input_preserves_dataloader_2d_sharding_on_single_host_cp(self):
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 host devices to build a (fsdp=2, context=2) mesh.")
    with tempfile.TemporaryDirectory() as tmp_dir:
      cp_config = pyconfig.initialize(
          ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")],
          per_device_batch_size=1,
          ici_fsdp_parallelism=2,
          ici_context_parallelism=2,
          run_name="test",
          base_output_directory=tmp_dir,
          tensorboard_dir=tmp_dir,
          skip_jax_distributed_system=True,
      )
      cp_mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "context"))
      tunix_cfg = train_sft.get_tunix_config(cp_config, mesh=cp_mesh)
      raw_batch = {"inputs": np.arange(2 * 8, dtype=np.int32).reshape(2, 8)}
      loader = DataLoader(cp_config, cp_mesh, iter([raw_batch]), goodput_recorder=None)
      loaded_batch = loader.load_next_batch()
      self.assertEqual(loaded_batch["inputs"].sharding.spec, P("fsdp", "context"))

      with jax.set_mesh(cp_mesh):
        sharded = tunix_sharding_utils.shard_input(loaded_batch, tunix_cfg.data_sharding_axis)
      self.assertIs(sharded, loaded_batch)
      self.assertEqual(sharded["inputs"].sharding.spec, P("fsdp", "context"))


if __name__ == "__main__":
  absltest.main()
