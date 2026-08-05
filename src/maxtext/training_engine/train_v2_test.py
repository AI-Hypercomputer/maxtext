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

"""Unit tests for MaxText pre-training loop driver (train_v2.py)."""

from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.training_engine import train_v2


class DummyPayload(abstract_engine.TrainerPayload):
  token_ids: jax.Array
  token_mask: jax.Array


class TrainV2Test(absltest.TestCase):

  def setup_config(self):
    """Creates a mock HyperParameters config for testing."""
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.init_weights_seed = 42
    mock_config.model_name = "llama3.1-8b"
    mock_config.steps = 2
    mock_config.eval_steps = 1
    mock_config.eval_interval = 1
    mock_config.eval_start_step = 0
    mock_config.checkpoint_period = 1
    mock_config.gradient_accumulation_steps = 1
    mock_config.elastic_enabled = False
    mock_config.packing = False
    mock_config.context_sharding = "data"
    mock_config.context_parallel_strategy = ""
    mock_config.context_parallel_load_balance = False
    mock_config.save_checkpoint_on_completion = True
    mock_config.shardy = False
    mock_config.remove_size_one_mesh_axis_from_type = True
    mock_config.dataset_path = ""
    mock_config.use_vertex_tensorboard = False
    mock_config.use_te_comm_gemm_overlap = False
    mock_config.logical_axis_rules = ()
    mock_config.logical_axis_rules_for_eval = ()
    mock_config.dcn_bandwidth_limit = None
    mock_config.dump_hlo = False
    mock_config.dump_step = -1
    mock_config.enable_tensorboard = False
    mock_config.tensorboard_dir = "/tmp/tb"
    mock_config.run_name = "test_run"
    mock_config.enable_profiler = False
    mock_config.profiler = ""
    mock_config.upload_all_profiler_results = False
    mock_config.profile_cleanly = False
    mock_config.profile_periodically_period = 0
    mock_config.skip_first_n_steps_for_profiler = 0
    mock_config.profiler_steps = 0
    mock_config.managed_mldiagnostics = False
    mock_config.profile_power_events = False
    mock_config.enable_goodput_recording = False
    mock_config.gcs_metrics = False
    mock_config.report_heartbeat_metric_for_gcp_monitoring = False
    mock_config.report_performance_metric_for_gcp_monitoring = False
    mock_config.enable_wandb = False
    mock_config.wandb_project_name = ""
    mock_config.wandb_run_name = ""
    mock_config.metrics_file = ""
    mock_config.max_target_length = 128
    mock_config.per_device_batch_size = 2
    return mock_config

  def get_real_mesh(self):
    devices = np.array(jax.devices()[:1])
    return Mesh(devices, ("data",))

  @mock.patch.object(train_v2, "create_data_iterator")
  @mock.patch.object(train_v2, "create_rampup_manager")
  @mock.patch.object(train_v2, "create_dataloader")
  @mock.patch.object(train_v2, "maybe_record_goodput")
  def test_setup_dataloaders(
      self,
      unused_mock_goodput,
      mock_create_dataloader,
      mock_create_rampup,
      mock_create_iterator,
  ):
    config = self.setup_config()
    mesh = self.get_real_mesh()

    mock_create_iterator.return_value = ("train_iter", "eval_iter")
    mock_create_rampup.return_value = "rampup_mgr"
    mock_create_dataloader.return_value = "train_loader"

    mock_ckpt_mgr = mock.MagicMock()
    mock_raw_ckpt_mgr = mock.MagicMock()
    mock_ckpt_mgr._checkpoint_manager = mock_raw_ckpt_mgr  # pylint: disable=protected-access

    train_loader, eval_iter, rampup_mgr = train_v2.setup_dataloaders(
        config=config,
        mesh=mesh,
        goodput_recorder=mock.MagicMock(),
        checkpoint_manager=mock_ckpt_mgr,
    )
    self.assertEqual(train_loader, "train_loader")
    self.assertEqual(eval_iter, "eval_iter")
    self.assertEqual(rampup_mgr, "rampup_mgr")
    mock_create_rampup.assert_called_once_with(config, mock_raw_ckpt_mgr)

  @mock.patch.object(train_v2, "create_data_iterator")
  @mock.patch.object(train_v2, "create_rampup_manager")
  @mock.patch.object(train_v2, "maybe_record_goodput")
  def test_setup_dataloaders_raises_on_invalid_packing_synthetic(self, unused_goodput, unused_rampup, unused_iterator):
    config = self.setup_config()
    config.packing = True
    config.dataset_type = "synthetic"
    config.context_sharding = "data"
    mesh = mock.MagicMock()
    mesh.shape = {"data": 2}

    with self.assertRaises(ValueError):
      train_v2.setup_dataloaders(config=config, mesh=mesh)

  @mock.patch.object(train_v2, "maybe_record_goodput")
  def test_load_next_batch(self, mock_goodput):
    mock_loader = mock.MagicMock()
    mock_loader.load_next_batch.return_value = {"inputs": jnp.array([1, 2])}
    mock_rampup = mock.MagicMock()
    batch = train_v2.load_next_batch(
        data_loader=mock_loader,
        rampup_manager=mock_rampup,
        goodput_recorder=None,
    )
    self.assertIn("inputs", batch)
    mock_loader.load_next_batch.assert_called_once_with(rampup_manager=mock_rampup)
    mock_goodput.assert_called_once_with(None, train_v2.GoodputEvent.DATA_LOADING)

  @mock.patch.object(train_v2.sharding, "get_input_data_sharding")
  def test_run_evaluation(self, mock_get_sharding):
    config = self.setup_config()
    config.eval_steps = 2
    mock_get_sharding.return_value = None
    mock_engine = mock.MagicMock(spec=maxtext_engine.MaxTextTrainingEngine)
    mock_engine.get_metrics.return_value = abstract_engine.MetricsBuffer(id=1, mode="eval")

    mock_iter = mock.MagicMock()
    mock_iter.__iter__.return_value = iter(
        [
            {"inputs": jnp.ones((1, 4))},
            {"inputs": jnp.ones((1, 4))},
        ]
    )

    history = train_v2.run_evaluation(
        engine=mock_engine,
        config=config,
        mesh=self.get_real_mesh(),
        eval_data_iterator=mock_iter,
        step=5,
    )
    self.assertLen(history, 2)
    self.assertEqual(mock_engine.eval_step.call_count, 2)
    mock_iter.reset.assert_called_once()

    # Confirm run_eval alias
    self.assertIs(train_v2.run_eval, train_v2.run_evaluation)

  @mock.patch.object(train_v2, "run_evaluation")
  @mock.patch.object(train_v2, "load_next_batch")
  def test_training_loop_iteration(self, mock_load_batch, mock_run_eval):
    config = self.setup_config()
    mock_load_batch.return_value = {"inputs": jnp.ones((1, 4))}

    mock_engine = mock.MagicMock(spec=maxtext_engine.MaxTextTrainingEngine)
    mock_engine.train_step = 1

    train_v2.training_loop_iteration(
        engine=mock_engine,
        config=config,
        mesh=self.get_real_mesh(),
        data_loader=mock.MagicMock(),
        rampup_manager=None,
        eval_data_iterator=None,
        goodput_recorder=mock.MagicMock(),
        step=1,
        start_step=0,
    )

    # fwd_bwd and update should be called once per iteration
    mock_engine.fwd_bwd.assert_called_once()
    mock_engine.update.assert_called_once()
    mock_engine.save_checkpoint.assert_called_once()
    mock_run_eval.assert_called_once()

  @mock.patch.object(train_v2, "run_evaluation")
  @mock.patch.object(train_v2, "load_next_batch")
  def test_training_loop_iteration_triggers_checkpoint_and_eval(self, mock_load_batch, mock_run_eval):
    config = self.setup_config()
    config.checkpoint_period = 5
    config.eval_interval = 5

    mock_load_batch.return_value = {"inputs": jnp.ones((1, 4))}
    mock_engine = mock.MagicMock(spec=maxtext_engine.MaxTextTrainingEngine)
    mock_engine.train_step = 5
    mock_engine.model = mock.MagicMock()

    train_v2.training_loop_iteration(
        engine=mock_engine,
        config=config,
        mesh=self.get_real_mesh(),
        data_loader=mock.MagicMock(),
        step=4,
        start_step=0,
    )

    mock_engine.save_checkpoint.assert_called_once_with(metadata={"step": 5, "source": "train_v2"}, step=5)
    mock_run_eval.assert_called_once()

  @mock.patch.object(train_v2, "setup_dataloaders")
  @mock.patch.object(train_v2.maxtext_utils, "get_mesh_from_config")
  @mock.patch.object(train_v2.sharding, "get_input_data_sharding")
  @mock.patch.object(train_v2.maxtext_utils, "get_shaped_batch")
  @mock.patch.object(train_v2, "training_loop_iteration")
  def test_train_loop(
      self,
      mock_iteration,
      unused_mock_shaped_batch,
      unused_mock_sharding,
      unused_mock_get_mesh,
      mock_setup_dataloaders,
  ):
    config = self.setup_config()
    config.steps = 2
    config.save_checkpoint_on_completion = True

    mock_engine = mock.MagicMock(spec=maxtext_engine.MaxTextTrainingEngine)
    mock_engine.train_step = 0
    mock_engine.model = mock.MagicMock()
    mock_engine.model.params = {"weights": jnp.ones((2, 2))}

    def fake_iteration(*_args, **_kwargs):
      mock_engine.train_step += 1

    mock_iteration.side_effect = fake_iteration
    mock_setup_dataloaders.return_value = (
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    )

    result = train_v2.train_loop(config=config, engine=mock_engine)

    self.assertIs(result, mock_engine)
    self.assertEqual(mock_iteration.call_count, 2)
    mock_engine.restore_checkpoint.assert_called_once()
    mock_engine.compile.assert_called_once()
    mock_engine.close.assert_called_once()

  def test_get_train_func(self):
    config = self.setup_config()
    config.elastic_enabled = False
    goodput_recorder = mock.MagicMock()

    func = train_v2.get_train_func(config, goodput_recorder, argv=[])
    self.assertTrue(callable(func))

  @mock.patch.object(train_v2, "initialize")
  @mock.patch.object(train_v2, "get_train_func")
  @mock.patch.object(train_v2, "record_goodput")
  @mock.patch.object(train_v2, "maybe_monitor_goodput")
  def test_main(
      self,
      unused_mock_monitor,
      unused_goodput,
      mock_get_func,
      mock_init,
  ):
    config = self.setup_config()
    goodput_recorder = mock.MagicMock()
    mock_init.return_value = (config, goodput_recorder)
    mock_train_func = mock.MagicMock()
    mock_get_func.return_value = mock_train_func

    train_v2.main([])
    mock_train_func.assert_called_once()


if __name__ == "__main__":
  absltest.main()
