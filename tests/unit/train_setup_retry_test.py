# Copyright 2023–2026 Google LLC
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

"""Unit tests for the elastic setup retry loop in train_loop."""

import threading
import unittest
from unittest import mock

import jax
import numpy as np

from maxtext.trainers.pre_train import train
from maxtext.utils import elastic_utils
from maxtext.utils import train_utils


class _StopAfterInit(Exception):
  """Stops train_loop right after initialization."""


def _fake_setup_results(mesh, model):
  """Returns a setup_train_loop result tuple tagged with `model`."""
  return (
      jax.random.PRNGKey(0),  # init_rng
      None,  # checkpoint_manager
      None,  # state_mesh_shardings
      model,
      mesh,
      None,  # learning_rate_schedule
      None,  # data_iterator
      None,  # data_loader
      None,  # rampup_manager
      None,  # eval_data_iterator
      None,  # state
  )


class TrainLoopSetupRetryTest(unittest.TestCase):
  """Tests that an abandoned setup attempt cannot leak into a newer attempt."""

  def test_abandoned_attempt_results_are_ignored(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    config = mock.MagicMock()
    config.elastic_enabled = True
    config.elastic_new_slice_check_period = 1
    config.num_target_devices = mesh.devices.size

    manager = mock.MagicMock()
    manager.active_slice_indices = frozenset({0, 1, 2})
    manager.available_inactive_slices = frozenset()

    attempt2_started = threading.Event()
    attempt1_done = threading.Event()
    cancel_events = []

    def fake_setup_train_loop(*_, cancel_event=None, **__):
      cancel_events.append(cancel_event)
      if len(cancel_events) == 1:
        # A new slice shows up while the first attempt is still restoring.
        manager.available_inactive_slices = frozenset({3})
        attempt2_started.wait(timeout=30)
        # The abandoned attempt finishes late, as in b/553546847.
        attempt1_done.set()
        return _fake_setup_results(mesh, "stale")
      attempt2_started.set()
      attempt1_done.wait(timeout=30)
      return _fake_setup_results(mesh, "current")

    def fake_get_active_slice_indices(*_, **__):
      manager.available_inactive_slices = frozenset()
      return frozenset({0, 1, 2, 3})

    used_models = []

    def fake_get_first_step(model, _):
      used_models.append(model)
      raise _StopAfterInit()

    with (
        mock.patch.object(elastic_utils, "elastic_manager", manager),
        mock.patch.object(elastic_utils, "ensure_elastic_manager_initialized"),
        mock.patch.object(elastic_utils, "record_slice_state"),
        mock.patch.object(elastic_utils, "mutate_config_for_topology"),
        mock.patch.object(elastic_utils, "live_devices", return_value=jax.devices()[:1]),
        mock.patch.object(elastic_utils, "elastic_enabled", return_value=True),
        mock.patch.object(elastic_utils, "elastic_snapshot", return_value=True),
        mock.patch.object(train.elastic, "get_active_slice_indices", side_effect=fake_get_active_slice_indices),
        mock.patch.object(train.time, "sleep"),
        mock.patch.object(train_utils, "setup_train_loop", side_effect=fake_setup_train_loop),
        mock.patch.object(train_utils, "maybe_apply_dcn_throttling"),
        mock.patch.object(train, "get_first_step", side_effect=fake_get_first_step),
    ):
      with self.assertRaises(_StopAfterInit):
        train.train_loop(config, recorder=None)

    self.assertEqual(used_models, ["current"])
    self.assertEqual(len(cancel_events), 2)
    self.assertTrue(cancel_events[0].is_set())
    self.assertFalse(cancel_events[1].is_set())


class SetupTrainLoopCancelTest(unittest.TestCase):
  """Tests that a cancelled setup_train_loop stops before loading data and state."""

  def test_cancelled_setup_skips_data_and_restore(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    config = mock.MagicMock()
    config.pure_nnx = False
    config.init_weights_seed = 0
    config.context_sharding = "context"
    cancel_event = threading.Event()
    cancel_event.set()

    with (
        mock.patch.object(train_utils.maxtext_utils, "get_mesh_from_config", return_value=mesh),
        mock.patch.object(train_utils.model_creation_utils, "from_config"),
        mock.patch.object(train_utils, "create_training_optimizer", return_value=(None, None)),
        mock.patch.object(train_utils, "create_checkpoint_manager", return_value=None),
        mock.patch("maxtext.input_pipeline.input_pipeline_interface.create_data_iterator") as create_data_iterator,
        mock.patch.object(train_utils.maxtext_utils, "setup_training_state") as setup_training_state,
    ):
      with self.assertRaises(train_utils.SetupCancelledError):
        train_utils.setup_train_loop(config, recorder=None, cancel_event=cancel_event)

    create_data_iterator.assert_not_called()
    setup_training_state.assert_not_called()


if __name__ == "__main__":
  unittest.main()
