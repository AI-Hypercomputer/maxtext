# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
import numpy as np
import optax
import orbax.checkpoint as ocp


class DummyNNXModel(nnx.Module):

  def __init__(self):
    self.weights = nnx.Param(jnp.array([1.0, 2.0]))


@dataclasses.dataclass(kw_only=True)
class DummyPayload(abstract_engine.TrainerPayload):
  token_ids: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))
  token_mask: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))


class MaxTextTrainingEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    dummy_model = DummyNNXModel()
    dummy_opt = nnx.Optimizer(dummy_model, optax.sgd(0.01), wrt=nnx.Param)
    patcher = mock.patch.object(
        maxtext_engine.train_utils,
        "create_training_optimizer",
        return_value=(lambda step: jnp.array(0.001), dummy_opt),
    )
    self.addCleanup(patcher.stop)
    patcher.start()

    from_pretrained_patcher = mock.patch.object(
        maxtext_engine.model_creation_utils,
        "from_pretrained",
        return_value=dummy_model,
    )
    self.addCleanup(from_pretrained_patcher.stop)
    self.mock_from_pretrained = from_pretrained_patcher.start()
    self.mock_config = self.setup_config()

  def setup_config(self, enable_checkpointing: bool = False):
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.init_weights_seed = 42
    mock_config.model_name = "llama3.1-8b"
    mock_config.learning_rate_schedule = mock.MagicMock(return_value=0.001)
    mock_config.tensorboard_dir = "/tmp/tb_dir"
    mock_config.run_name = "test_run"
    mock_config.enable_tensorboard = False
    if enable_checkpointing:
      mock_config.checkpoint_directory = "/tmp/test_out/checkpoints"
      mock_config.checkpoint_period = 500
      mock_config.max_num_checkpoints_to_keep = 10
      mock_config.async_checkpointing = True
    return mock_config

  def test_raises_type_error_for_non_pyconfig(self):
    invalid_config = abstract_engine.TrainingConfig()
    with self.assertRaises(TypeError):
      maxtext_engine.MaxTextTrainingEngine(invalid_config)  # pytype: disable=wrong-arg-types

  def test_raises_value_error_for_missing_model_name(self):
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.model_name = None
    with self.assertRaises(ValueError):
      maxtext_engine.MaxTextTrainingEngine(mock_config)

  @mock.patch.object(maxtext_engine.checkpointing, "CheckpointManager")
  @mock.patch.object(
      maxtext_engine.gradient_accumulation,
      "gradient_accumulation_loss_and_grad",
  )
  def test_max_text_trainer_instantiation_with_pyconfig(
      self, mock_ga, unused_mock_ckpt_mgr
  ):
    mock_ga.return_value = (
        jnp.array(0.5),
        {},
        {"weights": jnp.array([0.1, 0.2])},
    )

    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    self.assertIsInstance(t, abstract_engine.AbstractTrainingEngine)
    self.mock_from_pretrained.assert_called_once()
    self.assertEqual(t.train_step, 0)
    payload = DummyPayload(
        token_ids=jnp.ones((2, 2)),
        token_mask=jnp.ones((2, 2)),
    )
    t.compile(payload)
    self.assertTrue(t._compiled)
    t.with_loss_fn(lambda *args, **kwargs: (jnp.array(0.5), {}))
    self.assertFalse(t._compiled)
    t.fwd_bwd(payload)
    self.assertEqual(t._micro_step_count, 1)
    t.update()
    self.assertEqual(t._micro_step_count, 0)
    self.assertIsNone(t._accumulated_grads)
    metrics = t.get_metrics()
    self.assertIsInstance(metrics, list)

  @mock.patch("orbax.checkpoint.CheckpointManager")
  def test_max_text_trainer_checkpoint_manager_init(self, mock_create_mgr):
    mock_config = self.setup_config(enable_checkpointing=True)

    _ = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_create_mgr.assert_called_once_with(
        directory=mock_config.checkpoint_directory,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=mock_config.checkpoint_period,
            max_to_keep=mock_config.max_num_checkpoints_to_keep,
            enable_async_checkpointing=mock_config.async_checkpointing,
        ),
    )

  def test_save_checkpoint_with_model_and_optimizer(self):
    mock_config = self.setup_config(enable_checkpointing=True)

    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = mock.MagicMock()
    mock_orbax_mgr.latest_step.return_value = None
    mock_orbax_mgr.save.return_value = True
    t._checkpoint_manager._checkpoint_manager = mock_orbax_mgr

    t.train_step = 10
    metadata_to_save = {"worker_id": "worker_0", "run_id": "run_123"}
    t.save_checkpoint(metadata=metadata_to_save)

    mock_orbax_mgr.save.assert_called_once()

  def test_save_checkpoint_skips_if_already_saved(self):
    mock_config = self.setup_config(enable_checkpointing=True)

    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = mock.MagicMock()
    mock_orbax_mgr.latest_step.return_value = 10
    t._checkpoint_manager._checkpoint_manager = mock_orbax_mgr
    t.train_step = 10

    t.save_checkpoint(metadata={"key": "val"})
    mock_orbax_mgr.save.assert_not_called()

  def test_restore_checkpoint_no_checkpoint_returns_defaults(self):
    mock_config = self.setup_config(enable_checkpointing=True)

    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = mock.MagicMock()
    mock_orbax_mgr.latest_step.return_value = None
    t._checkpoint_manager._checkpoint_manager = mock_orbax_mgr

    restored_metadata = t.restore_checkpoint()
    self.assertEqual(restored_metadata, {})

  def test_restore_checkpoint_restores_ckpt_metadata(self):
    mock_config = self.setup_config(enable_checkpointing=True)

    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = mock.MagicMock()
    mock_orbax_mgr.latest_step.return_value = 10

    dummy_metadata = mock.MagicMock()
    mock_orbax_mgr.metadata.return_value = dummy_metadata
    mock_orbax_mgr.restore.return_value = {
        "model_params": {"weights": jnp.array([1.0, 2.0])}
    }

    t._checkpoint_manager._checkpoint_manager = mock_orbax_mgr

    restored_metadata = t.restore_checkpoint(step=10)
    self.assertEqual(t.train_step, 10)
    self.assertEqual(restored_metadata, dummy_metadata)
    mock_orbax_mgr.restore.assert_called_once()

  def test_record_metrics(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    # Record WeightedMetric
    t.record_metrics(
        name="loss",
        metric=abstract_engine.WeightedMetric(
            unreduced_sum=jnp.array(20.0), denominator=jnp.array(4.0)
        ),
    )
    t.record_metrics(
        name="loss",
        metric=abstract_engine.WeightedMetric(
            unreduced_sum=jnp.array(30.0), denominator=jnp.array(6.0)
        ),
    )

    # Record scalar
    t.record_metrics(
        name="lr",
        metric=0.002,
        aggregation_fn=lambda x: np.round(np.asarray(x), 4),
    )

    metrics_buffer = t.get_metrics(clear_cache=True)
    self.assertLen(metrics_buffer, 1)
    self.assertIn("loss", metrics_buffer[0].weighted_metrics)
    np.testing.assert_array_equal(
        metrics_buffer[0].weighted_metrics["loss"].unreduced_sum,
        jnp.array([20.0, 30.0]),
    )
    np.testing.assert_array_equal(
        metrics_buffer[0].weighted_metrics["loss"].denominator,
        jnp.array([4.0, 6.0]),
    )
    self.assertIn("lr", metrics_buffer[0].scalar_metrics)
    np.testing.assert_array_equal(
        metrics_buffer[0].scalar_metrics["lr"], jnp.array([0.002])
    )
    self.assertIn("lr", metrics_buffer[0].aggregation_fns)
    self.assertEqual(
        metrics_buffer[0].aggregation_fns["lr"](jnp.array([0.002])), 0.002
    )

  def test_get_metrics(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t._metrics_logger._metrics_buffer = [
        abstract_engine.MetricsBuffer(
            id=1,
            weighted_metrics={
                "loss": abstract_engine.WeightedMetric(
                    unreduced_sum=jnp.array(50.0), denominator=jnp.array(10.0)
                )
            },
            scalar_metrics={"lr": jnp.array(0.002)},
            aggregation_fns={"lr": lambda x: np.round(np.asarray(x), 4)},
        )
    ]
    metrics = t.get_metrics(clear_cache=True)
    self.assertIn("loss", metrics[0].weighted_metrics)
    np.testing.assert_array_equal(
        metrics[0].weighted_metrics["loss"].unreduced_sum, jnp.array([50.0])
    )
    np.testing.assert_array_equal(
        metrics[0].weighted_metrics["loss"].denominator, jnp.array([10.0])
    )
    self.assertIn("lr", metrics[0].scalar_metrics)
    np.testing.assert_array_equal(
        metrics[0].scalar_metrics["lr"], jnp.array([0.002])
    )
    self.assertIn("lr", metrics[0].aggregation_fns)
    self.assertEqual(
        metrics[0].aggregation_fns["lr"](jnp.array([0.002])), 0.002
    )
    self.assertEmpty(t._metrics_logger._metrics_buffer)

  @mock.patch.object(
      maxtext_engine.gradient_accumulation,
      "gradient_accumulation_loss_and_grad",
  )
  def test_trainer_for_one_global_step(self, mock_ga):
    mock_ga.return_value = (
        jnp.array(0.5),
        {},
        {"weights": jnp.array([0.1, 0.1])},
    )
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    self.assertEqual(t.train_step, 0)
    for mini_b in range(3):
      for _ in range(2):
        t.fwd_bwd(DummyPayload())
      t.update()
      self.assertEqual(t.train_step, mini_b + 1)
      t.save_checkpoint(metadata={"batch": mini_b})
    metrics = t.get_metrics(clear_cache=True)
    self.assertLen(metrics, 3)
    for metric in metrics:
      self.assertIn("lr", metric.scalar_metrics)


if __name__ == "__main__":
  absltest.main()
