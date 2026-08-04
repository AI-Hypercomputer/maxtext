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
        eval_interval=-1,
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
  def test_validate_config_accepts_weighted_diffusion_accumulation(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
        training_objective="block_diffusion",
        gradient_accumulation_steps=2,
        eval_interval=-1,
    )

    train_sft.validate_config(config)

  @pytest.mark.cpu_only
  def test_validate_config_accepts_weighted_diffusion_evaluation(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
        training_objective="block_diffusion",
        gradient_accumulation_steps=1,
        eval_interval=10,
    )

    train_sft.validate_config(config)

  @pytest.mark.cpu_only
  def test_setup_validates_before_initializing_model(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=True,
    )

    with (
        mock.patch.object(train_sft, "get_tunix_config") as get_tunix_config,
        mock.patch.object(train_sft.model_creation_utils, "from_pretrained") as from_pretrained,
        self.assertRaisesRegex(ValueError, "optimizer_memory_host_offload=True"),
    ):
      train_sft.setup_trainer_state(config)

    get_tunix_config.assert_not_called()
    from_pretrained.assert_not_called()

  @pytest.mark.cpu_only
  def test_maxtext_loss_wrapper_forwards_diffusion_masks(self):
    trainer = mock.MagicMock()
    trainer.with_loss_fn.return_value = trainer
    config = SimpleNamespace()
    train_sft.use_maxtext_loss_function(trainer, config)
    wrapped_loss = trainer.with_loss_fn.call_args.args[0]
    arrays = {
        name: mock.sentinel.__getattr__(name)
        for name in (
            "inputs",
            "inputs_position",
            "inputs_segmentation",
            "targets",
            "targets_position",
            "targets_segmentation",
            "completion_mask",
            "corruption_mask",
            "targets_loss_mask",
        )
    }

    with mock.patch.object(train_sft, "loss_fn", return_value=(1.0, {})) as loss_fn_mock:
      wrapped_loss(mock.sentinel.model, **arrays)

    forwarded = loss_fn_mock.call_args.args[2]
    self.assertEqual(forwarded, arrays)
    trainer.with_loss_fn.assert_called_once_with(wrapped_loss, has_aux=True)

  @pytest.mark.cpu_only
  def test_diffusion_objective_uses_tunix_adapter(self):
    trainer = mock.sentinel.trainer
    config = SimpleNamespace(training_objective="block_diffusion")
    with (
        mock.patch.object(train_sft.maxtext_diffusion_sft, "create_batch_adapter", return_value="adapter"),
        mock.patch.object(
            train_sft.maxtext_diffusion_sft,
            "create_target_aligned_logits_fn",
            return_value="logits_fn",
        ),
        mock.patch.object(
            train_sft.tunix_diffusion_sft,
            "configure_diffusion_sft",
            return_value=trainer,
        ) as configure,
    ):
      result = train_sft.configure_training_objective(trainer, config)

    self.assertIs(result, trainer)
    configure.assert_called_once_with(trainer, "adapter", "logits_fn")

  @pytest.mark.cpu_only
  def test_ar_objective_retains_maxtext_trainer_and_loss(self):
    config = SimpleNamespace(training_objective="causal_lm")
    with (
        mock.patch.object(train_sft, "MaxTextPeftTrainer", return_value=mock.sentinel.trainer) as trainer_type,
        mock.patch.object(train_sft, "use_maxtext_loss_function", return_value=mock.sentinel.configured) as configure,
    ):
      trainer = train_sft._create_trainer("model", "optimizer", "config", config)
      result = train_sft.configure_training_objective(trainer, config)

    self.assertIs(trainer, mock.sentinel.trainer)
    self.assertIs(result, mock.sentinel.configured)
    trainer_type.assert_called_once_with("model", "optimizer", "config")
    configure.assert_called_once_with(mock.sentinel.trainer, config)

  @pytest.mark.cpu_only
  def test_diffusion_objective_uses_current_tunix_trainer(self):
    config = SimpleNamespace(training_objective="block_diffusion")
    with mock.patch.object(train_sft.peft_trainer, "PeftTrainer", return_value=mock.sentinel.trainer) as trainer_type:
      trainer = train_sft._create_trainer("model", "optimizer", "config", config)

    self.assertIs(trainer, mock.sentinel.trainer)
    trainer_type.assert_called_once_with("model", "optimizer", "config")

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
