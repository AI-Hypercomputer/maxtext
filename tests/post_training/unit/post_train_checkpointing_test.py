# Copyright 2023-2026 Google LLC
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

"""Unit tests for post-training checkpointing in MaxText's on-disk layout."""

import os
import tempfile
import unittest
from unittest import mock

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest
from tunix.sft import checkpoint_manager as tunix_checkpoint_manager

from maxtext.trainers.post_train import checkpointing as post_train_checkpointing

pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]


class _Model(nnx.Module):
  """Tiny stand-in for a MaxText Transformer."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)


class _Adapter(nnx.Module):
  """Stand-in for TunixMaxTextAdapter: holds the model as its only child."""

  def __init__(self, base):
    self.base = base


def _build(wrapped):
  model = _Model(nnx.Rngs(0))
  outer = _Adapter(model) if wrapped else model
  optimizer = nnx.Optimizer(outer, optax.adamw(1e-3), wrt=nnx.Param)
  return outer, optimizer


def _train_a_step(model, optimizer):
  """Moves the weights and fills opt_state, so a restore has something to prove."""
  target = model.base if isinstance(model, _Adapter) else model
  grads = jax.tree.map(lambda p: jnp.full_like(p, 0.1), nnx.state(model, nnx.Param))
  optimizer.update(model, grads)
  return jnp.asarray(target.linear.kernel[...])


class PostTrainCheckpointLayoutTest(unittest.TestCase):
  """The on-disk layout has to be MaxText's, so pre-training can read what post-training wrote."""

  def _save(self, directory, wrapped):
    model, optimizer = _build(wrapped)
    trained = _train_a_step(model, optimizer)
    manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
        root_directory=directory, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
    )
    self.assertTrue(manager.save(1, model, optimizer, force=True))
    manager.close()
    return trained

  def _on_disk_keys(self, directory):
    metadata = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).metadata(epath.Path(directory) / "1" / "items")
    tree = getattr(metadata.item_metadata, "tree", metadata.item_metadata)
    return list(ocp.tree.to_flat_dict(tree, sep="/"))

  def test_saves_maxtext_layout_not_the_tunix_one(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      self._save(d, wrapped=False)
      self.assertEqual(sorted(os.listdir(os.path.join(d, "1"))), ["_CHECKPOINT_METADATA", "items"])
      keys = self._on_disk_keys(d)

    self.assertTrue(any(k.startswith("params/params/linear/") for k in keys), keys)
    self.assertTrue(any(k.startswith("opt_state/") for k in keys), keys)
    self.assertIn("step", keys)
    # rngs are NNX-only, so they belong in nnx_aux rather than the Linen collections.
    self.assertTrue(any(k.startswith("nnx_aux/") for k in keys), keys)

  def test_adapter_level_is_stripped_from_weights_and_optimizer(self):
    """DPO and RL train through the adapter; its `base` level must not reach the checkpoint.

    Pre-training builds its params and opt_state from the bare model, so a stray `base` level
    puts every weight at a path it will not look for.
    """
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      self._save(d, wrapped=True)
      keys = self._on_disk_keys(d)

    self.assertTrue(keys)
    self.assertEqual([k for k in keys if "base" in k.split("/")], [])

  def test_restores_weights_and_optimizer_it_saved(self):
    for wrapped in (False, True):
      with self.subTest(wrapped=wrapped):
        with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
          trained = self._save(d, wrapped=wrapped)

          model, optimizer = _build(wrapped)
          target = model.base if wrapped else model
          self.assertFalse(jnp.array_equal(trained, target.linear.kernel[...]), "fresh model should differ")

          manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
              root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
          )
          step, _ = manager.maybe_restore(model, optimizer)
          manager.close()

        self.assertEqual(step, 1)
        self.assertTrue(jnp.array_equal(trained, target.linear.kernel[...]), "weights were not restored")
        # A resume that dropped opt_state would silently restart the optimizer's moments.
        opt_leaves = jax.tree.leaves(nnx.to_pure_dict(nnx.state(optimizer, nnx.optimizer.OptState)))
        self.assertTrue(any(jnp.any(jnp.asarray(leaf) != 0) for leaf in opt_leaves), "opt_state came back empty")

  def test_no_checkpoint_yet_reports_step_zero(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      step, metadata = manager.maybe_restore(model, optimizer)
      manager.close()

    self.assertEqual(step, 0)
    self.assertEqual(metadata, {})

  def test_custom_metadata_survives_the_round_trip(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      manager.save(1, model, optimizer, force=True, custom_metadata={"run": "abc"})
      manager.wait_until_finished()
      _, metadata = manager.maybe_restore(*_build(wrapped=False))
      manager.close()

    self.assertEqual(metadata.get("run"), "abc")

  def test_weights_only_when_there_is_no_optimizer(self):
    """Some callers checkpoint the model alone; opt_state must simply be absent."""
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, _ = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      self.assertTrue(manager.save(1, model, optimizer=None, force=True))
      manager.close()
      keys = self._on_disk_keys(d)

    self.assertTrue(any(k.startswith("params/params/linear/") for k in keys), keys)
    self.assertEqual([k for k in keys if k.startswith("opt_state")], [])


class PostTrainCheckpointSaveDecisionTest(unittest.TestCase):
  """Saving has to honour the same enable/interval rules the Tunix manager applied."""

  def test_disabled_when_there_is_no_root_directory(self):
    model, optimizer = _build(wrapped=False)
    manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(root_directory=None)

    self.assertFalse(manager.save(1, model, optimizer, force=True))
    self.assertEqual(manager.maybe_restore(model, optimizer), (0, {}))

  def test_declines_when_the_policy_says_not_to_save(self):
    """`force` bypasses the policy; without it the manager's decision is honoured."""
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      with mock.patch.object(manager._checkpoint_manager, "should_save", return_value=False):  # pylint: disable=protected-access
        declined = manager.save(3, model, optimizer)
      forced = manager.save(3, model, optimizer, force=True)
      manager.close()

      self.assertFalse(declined)
      self.assertTrue(forced)
      self.assertEqual(sorted(x for x in os.listdir(d) if x.isdigit()), ["3"])


class PostTrainCheckpointLegacyLayoutTest(unittest.TestCase):
  """Checkpoints written before the layout change are still in Tunix's, and must still restore."""

  def test_falls_back_to_the_tunix_layout(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      trained = _train_a_step(model, optimizer)

      legacy = tunix_checkpoint_manager.CheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      legacy.save(1, model, optimizer, force=True)
      legacy.close()

      fresh_model, fresh_optimizer = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      step, _ = manager.maybe_restore(fresh_model, fresh_optimizer)
      manager.close()

    self.assertEqual(step, 1)
    self.assertTrue(jnp.array_equal(trained, fresh_model.linear.kernel[...]))


class PostTrainCheckpointSubclassHookTest(unittest.TestCase):
  """Distillation checkpoints a sub-module and an extra item, through these hooks."""

  class _Bundle(nnx.Module):

    def __init__(self, student):
      self.student_model = student

  class _Manager(post_train_checkpointing.MaxTextLayoutCheckpointManager):
    """Stand-in for the distillation manager: checkpoints a sub-module plus an extra item."""

    def __init__(self, root_directory, options):
      super().__init__(
          root_directory=root_directory,
          options=options,
          extra_item_handlers={"note": ocp.JsonCheckpointHandler()},
      )

    def model_to_checkpoint(self, model):
      return getattr(model, "student_model", model)

    def _extra_save_args(self, step):
      del step
      return {"note": ocp.args.JsonSave({"hello": "world"})}

  def test_hooks_pick_the_submodule_and_add_the_extra_item(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      student = _Model(nnx.Rngs(0))
      bundle = self._Bundle(student)
      optimizer = nnx.Optimizer(student, optax.adamw(1e-3), wrt=nnx.Param)
      manager = self._Manager(d, ocp.CheckpointManagerOptions(save_interval_steps=1))
      self.assertTrue(manager.save(1, bundle, optimizer, force=True))
      manager.close()

      self.assertIn("note", os.listdir(os.path.join(d, "1")))
      metadata = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).metadata(epath.Path(d) / "1" / "items")
      tree = getattr(metadata.item_metadata, "tree", metadata.item_metadata)
      keys = list(ocp.tree.to_flat_dict(tree, sep="/"))

    # The student's weights, not the bundle's wrapper level.
    self.assertTrue(any(k.startswith("params/params/linear/") for k in keys), keys)
    self.assertEqual([k for k in keys if "student_model" in k.split("/")], [])


class InstallTest(unittest.TestCase):
  """`install` swaps in the MaxText-layout manager and restores what it finds."""

  class _FakeConfig:

    def __init__(self):
      self.checkpointing_options = ocp.CheckpointManagerOptions(save_interval_steps=1)

    def get_with_default(self, key, default):
      del key
      return default

  class _FakeTrainer:

    def __init__(self, model, optimizer, checkpoint_manager):
      self.model = model
      self.optimizer = optimizer
      self.checkpoint_manager = checkpoint_manager
      self.config = InstallTest._FakeConfig()
      self._train_steps = 0
      self._iter_steps = 0

  def test_replaces_the_manager_and_restores_the_step(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      trained = _train_a_step(model, optimizer)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      manager.save(4, model, optimizer, force=True)
      manager.close()

      fresh_model, fresh_optimizer = _build(wrapped=False)
      trainer = self._FakeTrainer(fresh_model, fresh_optimizer, checkpoint_manager=None)
      post_train_checkpointing.install(trainer, d)
      trainer.checkpoint_manager.close()

    self.assertIsInstance(trainer.checkpoint_manager, post_train_checkpointing.MaxTextLayoutCheckpointManager)
    self.assertEqual(trainer._train_steps, 4)  # pylint: disable=protected-access
    self.assertEqual(trainer._iter_steps, 4)  # pylint: disable=protected-access
    self.assertTrue(jnp.array_equal(trained, fresh_model.linear.kernel[...]))

  def test_closes_the_manager_it_replaces(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      replaced = tunix_checkpoint_manager.CheckpointManager(root_directory=None)
      closed = []
      replaced.close = lambda: closed.append(True)

      trainer = self._FakeTrainer(model, optimizer, checkpoint_manager=replaced)
      post_train_checkpointing.install(trainer, d)
      trainer.checkpoint_manager.close()

    self.assertEqual(closed, [True])


class UnwrapModelTest(unittest.TestCase):
  """The adapter is matched on its child module, not on its class."""

  def test_unwraps_a_wrapper(self):
    model = _Model(nnx.Rngs(0))
    self.assertIs(post_train_checkpointing.unwrap_model(_Adapter(model)), model)

  def test_leaves_a_bare_model_alone(self):
    model = _Model(nnx.Rngs(0))
    self.assertIs(post_train_checkpointing.unwrap_model(model), model)

  def test_ignores_a_base_attribute_that_is_not_a_module(self):
    model = _Model(nnx.Rngs(0))
    model.base = "not a module"
    self.assertIs(post_train_checkpointing.unwrap_model(model), model)


if __name__ == "__main__":
  unittest.main()
