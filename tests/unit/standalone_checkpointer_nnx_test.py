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

"""Unit tests for add_entropy_to_checkpoint on NNX state."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from maxtext.common import train_state_nnx
from maxtext.utils.standalone_checkpointer import add_entropy_to_checkpoint, checkpoint_loop


class _TinyModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.lin = nnx.Linear(4, 4, rngs=rngs)


def _expected_cos_sin(params_state):
  mu = jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), params_state)
  nu = jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), params_state)
  return mu, nu


class AddEntropyNNXTest(unittest.TestCase):

  def test_overwrites_adam_mu_and_nu(self):
    model = _TinyModel(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(model, optimizer)

    params_before = nnx.state(model, nnx.Param)
    expected_mu, expected_nu = _expected_cos_sin(params_before)

    new_state = add_entropy_to_checkpoint(state)

    self.assertIs(new_state, state)  # mutated in place
    actual_mu = new_state.optimizer.opt_state[0].mu
    actual_nu = new_state.optimizer.opt_state[0].nu

    expected_mu_leaves = jax.tree_util.tree_leaves(expected_mu)
    expected_nu_leaves = jax.tree_util.tree_leaves(expected_nu)
    actual_mu_leaves = jax.tree_util.tree_leaves(actual_mu)
    actual_nu_leaves = jax.tree_util.tree_leaves(actual_nu)
    self.assertEqual(len(expected_mu_leaves), len(actual_mu_leaves))
    for e, a in zip(expected_mu_leaves, actual_mu_leaves):
      self.assertTrue(jnp.allclose(e, a))
    for e, a in zip(expected_nu_leaves, actual_nu_leaves):
      self.assertTrue(jnp.allclose(e, a))

  def test_does_not_mutate_model_params(self):
    model = _TinyModel(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(model, optimizer)

    params_before = jax.tree_util.tree_map(jnp.array, nnx.state(model, nnx.Param).to_pure_dict())
    add_entropy_to_checkpoint(state)
    params_after = nnx.state(model, nnx.Param).to_pure_dict()

    for path, before in jax.tree_util.tree_leaves_with_path(params_before):
      after = params_after
      for key in path:
        after = after[key.key]
      self.assertTrue(jnp.array_equal(before, after))

  def test_works_on_split_nnx_state(self):
    """`setup_training_state` returns a flat `nnx.State`, not a `TrainStateNNX`."""
    model = _TinyModel(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    train_state = train_state_nnx.TrainStateNNX(model, optimizer)

    _, split_state = nnx.split(train_state)

    new_state = add_entropy_to_checkpoint(split_state)
    self.assertIs(new_state, split_state)

    # mu should now be cos(1000 * params); params for a freshly initialized
    # nnx.Linear bias is 0 so cos(0) = 1.
    mu_leaves = jax.tree_util.tree_leaves(new_state.optimizer.opt_state[0].mu)
    nu_leaves = jax.tree_util.tree_leaves(new_state.optimizer.opt_state[0].nu)
    self.assertTrue(any(jnp.allclose(leaf, 1.0) for leaf in mu_leaves))  # cos(0)=1
    self.assertTrue(any(jnp.allclose(leaf, 0.0) for leaf in nu_leaves))  # sin(0)=0


class CheckpointLoopTest(unittest.TestCase):
  """Unit tests for checkpoint_loop in standalone_checkpointer."""

  def _create_mock_config(self, start_from_checkpoint=False):
    """Creates a mock configuration for checkpoint_loop tests."""
    config = mock.MagicMock()
    config.init_weights_seed = 0
    config.pure_nnx = False
    config.steps = 2
    config.standalone_checkpointer_per_step_interval = 0.5
    config.standalone_checkpointer_drop_page_cache_before_restore = True
    config.standalone_checkpointer_enable_restore_in_loop = True
    config.standalone_checkpointer_start_from_checkpoint = start_from_checkpoint
    config.logical_axis_rules = []
    return config

  def test_checkpoint_loop_default_save_and_restore(self):
    """Tests standard checkpoint saving and restoring inside loop."""
    config = self._create_mock_config(start_from_checkpoint=False)
    mock_state = mock.MagicMock()
    mock_ckpt_mgr = mock.MagicMock()
    mock_ckpt_mgr.load_checkpointables.return_value = {"items": mock_state}

    with (
        mock.patch("maxtext.utils.standalone_checkpointer.from_config"),
        mock.patch("maxtext.utils.train_utils.create_training_optimizer", return_value=(None, mock.MagicMock())),
        mock.patch("maxtext.utils.train_utils.create_checkpoint_manager", return_value=mock_ckpt_mgr),
        mock.patch(
            "maxtext.utils.maxtext_utils.setup_training_state",
            return_value=(mock_state, None, None, None, None),
        ) as mock_setup,
        mock.patch("maxtext.utils.standalone_checkpointer.add_entropy_to_checkpoint", return_value=mock_state),
        mock.patch("maxtext.utils.standalone_checkpointer.get_first_step", return_value=1),
        mock.patch("maxtext.common.checkpointing.save_checkpoint", return_value=True),
        mock.patch("maxtext.common.checkpointing.wait_until_finished"),
        mock.patch("maxtext.utils.standalone_checkpointer.jax.block_until_ready"),
        mock.patch("maxtext.utils.standalone_checkpointer.jax.experimental.multihost_utils.sync_global_devices"),
        mock.patch("maxtext.utils.standalone_checkpointer.time.sleep") as mock_sleep,
        mock.patch("maxtext.utils.standalone_checkpointer.os.system") as mock_system,
    ):
      returned_state = checkpoint_loop(config)

    self.assertIs(returned_state, mock_state)
    mock_setup.assert_called_once()
    mock_sleep.assert_called_once()
    mock_system.assert_called_once_with("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    mock_ckpt_mgr.load_checkpointables.assert_called_once_with(1, {"items": mock_state})

  def test_checkpoint_loop_start_from_checkpoint(self):
    """Tests initializing state from an existing checkpoint."""
    config = self._create_mock_config(start_from_checkpoint=True)
    mock_state = mock.MagicMock()
    mock_ckpt_mgr = mock.MagicMock()
    mock_ckpt_mgr.load_checkpointables.return_value = {"items": mock_state}

    with (
        mock.patch("maxtext.utils.standalone_checkpointer.from_config"),
        mock.patch("maxtext.utils.train_utils.create_training_optimizer", return_value=(None, mock.MagicMock())),
        mock.patch("maxtext.utils.train_utils.create_checkpoint_manager", return_value=mock_ckpt_mgr),
        mock.patch("maxtext.utils.maxtext_utils.get_abstract_state", return_value=(mock.MagicMock(), None, None)),
        mock.patch(
            "maxtext.common.checkpointing.load_state_if_possible",
            return_value=({"items": mock_state}, None),
        ) as mock_load,
        mock.patch("maxtext.utils.maxtext_utils.setup_training_state") as mock_setup,
        mock.patch("maxtext.utils.standalone_checkpointer.add_entropy_to_checkpoint", return_value=mock_state),
        mock.patch("maxtext.utils.standalone_checkpointer.get_first_step", return_value=1),
        mock.patch("maxtext.common.checkpointing.save_checkpoint", return_value=True),
        mock.patch("maxtext.common.checkpointing.wait_until_finished"),
        mock.patch("maxtext.utils.standalone_checkpointer.jax.block_until_ready"),
        mock.patch("maxtext.utils.standalone_checkpointer.jax.experimental.multihost_utils.sync_global_devices"),
        mock.patch("maxtext.utils.standalone_checkpointer.time.sleep"),
        mock.patch("maxtext.utils.standalone_checkpointer.os.system"),
    ):
      returned_state = checkpoint_loop(config)

    self.assertIs(returned_state, mock_state)
    mock_load.assert_called_once()
    mock_setup.assert_not_called()

  def test_checkpoint_loop_start_from_checkpoint_fallback(self):
    """Tests falling back to setup_training_state when checkpoint is not found."""
    config = self._create_mock_config(start_from_checkpoint=True)
    mock_state = mock.MagicMock()
    mock_ckpt_mgr = mock.MagicMock()
    mock_ckpt_mgr.load_checkpointables.return_value = {"items": mock_state}

    with (
        mock.patch("maxtext.utils.standalone_checkpointer.from_config"),
        mock.patch("maxtext.utils.train_utils.create_training_optimizer", return_value=(None, mock.MagicMock())),
        mock.patch("maxtext.utils.train_utils.create_checkpoint_manager", return_value=mock_ckpt_mgr),
        mock.patch("maxtext.utils.maxtext_utils.get_abstract_state", return_value=(mock.MagicMock(), None, None)),
        mock.patch("maxtext.common.checkpointing.load_state_if_possible", return_value=(None, None)),
        mock.patch(
            "maxtext.utils.maxtext_utils.setup_training_state",
            return_value=(mock_state, None, None, None, None),
        ) as mock_setup,
        mock.patch("maxtext.utils.standalone_checkpointer.add_entropy_to_checkpoint", return_value=mock_state),
        mock.patch("maxtext.utils.standalone_checkpointer.get_first_step", return_value=1),
        mock.patch("maxtext.common.checkpointing.save_checkpoint", return_value=True),
        mock.patch("maxtext.common.checkpointing.wait_until_finished"),
        mock.patch("maxtext.utils.standalone_checkpointer.jax.block_until_ready"),
        mock.patch("maxtext.utils.standalone_checkpointer.jax.experimental.multihost_utils.sync_global_devices"),
        mock.patch("maxtext.utils.standalone_checkpointer.time.sleep"),
        mock.patch("maxtext.utils.standalone_checkpointer.os.system"),
    ):
      returned_state = checkpoint_loop(config)

    self.assertIs(returned_state, mock_state)
    mock_setup.assert_called_once()

  def test_checkpoint_loop_pure_nnx(self):
    """Tests checkpoint loop with pure_nnx enabled."""
    config = self._create_mock_config(start_from_checkpoint=False)
    config.pure_nnx = True
    mock_state = mock.MagicMock()
    mock_state.to_pure_dict.return_value = {"params": {}}
    mock_ckpt_mgr = mock.MagicMock()
    mock_ckpt_mgr.load_checkpointables.return_value = {"items": mock_state}

    with (
        mock.patch("maxtext.utils.maxtext_utils.get_mesh_from_config"),
        mock.patch("maxtext.utils.maxtext_utils_nnx.create_nnx_rngs"),
        mock.patch("maxtext.utils.standalone_checkpointer.from_config"),
        mock.patch("maxtext.utils.train_utils.create_training_optimizer", return_value=(None, mock.MagicMock())),
        mock.patch(
            "maxtext.utils.model_creation_utils.create_nnx_abstract_model",
            return_value=(mock.MagicMock(), None),
        ),
        mock.patch("maxtext.utils.train_utils.create_checkpoint_manager", return_value=mock_ckpt_mgr),
        mock.patch(
            "maxtext.utils.maxtext_utils.setup_training_state",
            return_value=(mock_state, None, None, None, None),
        ),
        mock.patch("maxtext.utils.standalone_checkpointer.add_entropy_to_checkpoint", return_value=mock_state),
        mock.patch("maxtext.utils.standalone_checkpointer.get_first_step", return_value=1),
        mock.patch(
            "maxtext.common.train_state_nnx.to_linen_checkpoint_dict", return_value={"params": {}}
        ) as mock_to_linen,
        mock.patch("maxtext.common.checkpointing.save_checkpoint", return_value=True),
        mock.patch("maxtext.common.checkpointing.wait_until_finished"),
        mock.patch("maxtext.utils.standalone_checkpointer.jax.block_until_ready"),
        mock.patch("maxtext.utils.standalone_checkpointer.jax.experimental.multihost_utils.sync_global_devices"),
        mock.patch("maxtext.utils.standalone_checkpointer.time.sleep"),
        mock.patch("maxtext.utils.standalone_checkpointer.os.system"),
    ):
      returned_state = checkpoint_loop(config)

    self.assertIs(returned_state, mock_state)
    mock_to_linen.assert_called_once()


if __name__ == "__main__":
  unittest.main()
