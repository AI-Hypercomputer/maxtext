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

import contextlib
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import orbax.checkpoint.checkpoint_manager
import pytest
from tunix.sft import checkpoint_manager as tunix_checkpoint_manager

from maxtext.common import checkpointing
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


class _ScannedModel(nnx.Module):
  """scan_layers=True: every decoder layer stacked under one `layers` key."""

  def __init__(self, rngs: nnx.Rngs, num_layers=3):
    self.layers = nnx.Param(jnp.zeros((num_layers, 2, 3)))
    self.decoder_norm = nnx.Param(jnp.ones((3,)))
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)


class _UnscannedModel(nnx.Module):
  """scan_layers=False: one key per decoder layer, each without the stacking axis."""

  def __init__(self, rngs: nnx.Rngs, num_layers=3):
    for i in range(num_layers):
      setattr(self, f"layers_{i}", nnx.Linear(2, 3, rngs=rngs))
    self.decoder_norm = nnx.Param(jnp.ones((3,)))
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)


def _build(wrapped):
  model = _Model(nnx.Rngs(0))
  outer = _Adapter(model) if wrapped else model
  optimizer = nnx.Optimizer(outer, optax.adamw(1e-3), wrt=nnx.Param)
  return outer, optimizer


def _on_disk_keys(directory, step=1):
  """Returns the checkpoint's leaf paths, slash-separated.

  Args:
    directory: Checkpoint root directory.
    step: Step to read.

  Returns:
    A list of leaf paths.
  """
  metadata = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).metadata(epath.Path(directory) / str(step) / "items")
  tree = getattr(metadata.item_metadata, "tree", metadata.item_metadata)
  return list(ocp.tree.to_flat_dict(tree, sep="/"))


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

  def test_saves_maxtext_layout_not_the_tunix_one(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      self._save(d, wrapped=False)
      self.assertEqual(sorted(os.listdir(os.path.join(d, "1"))), ["_CHECKPOINT_METADATA", "items"])
      keys = _on_disk_keys(d)

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
      keys = _on_disk_keys(d)

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
        opt_leaves = jax.tree.leaves(nnx.state(optimizer, nnx.optimizer.OptState))
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

  def test_config_metadata_is_stamped_like_pre_training_does(self):
    """`scan_layers` and the LoRA settings have to ride along for the loaders to read them back."""
    config = SimpleNamespace(scan_layers=False, lora=SimpleNamespace(lora_rank=8, model_dump=lambda: {"lora_rank": 8}))
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1), config=config
      )
      manager.save(1, model, optimizer, force=True)
      manager.wait_until_finished()
      _, metadata = manager.maybe_restore(*_build(wrapped=False))
      manager.close()

    self.assertIs(metadata.get("scan_layers"), False)
    self.assertEqual(metadata.get("lora"), {"lora_rank": 8})

  def test_a_caller_key_wins_over_the_config_derived_one(self):
    config = SimpleNamespace(scan_layers=True, lora=None)
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1), config=config
      )
      manager.save(1, model, optimizer, force=True, custom_metadata={"scan_layers": False, "run": "abc"})
      manager.wait_until_finished()
      _, metadata = manager.maybe_restore(*_build(wrapped=False))
      manager.close()

    self.assertIs(metadata.get("scan_layers"), False)
    self.assertEqual(metadata.get("run"), "abc")

  def test_no_config_still_saves(self):
    """The config is optional; a manager built without one just writes no sidecar metadata."""
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      self.assertTrue(manager.save(1, model, optimizer, force=True))
      manager.wait_until_finished()
      _, metadata = manager.maybe_restore(*_build(wrapped=False))
      manager.close()

    self.assertEqual(metadata, {})

  def test_weights_only_when_there_is_no_optimizer(self):
    """Some callers checkpoint the model alone; opt_state must simply be absent."""
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, _ = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      self.assertTrue(manager.save(1, model, optimizer=None, force=True))
      manager.close()
      keys = _on_disk_keys(d)

    self.assertTrue(any(k.startswith("params/params/linear/") for k in keys), keys)
    self.assertEqual([k for k in keys if k.startswith("opt_state")], [])


class PostTrainCheckpointScanLayoutTest(unittest.TestCase):
  """Both scan settings have to survive the layout conversion.

  Scanned stacks the decoder layers under one `layers` key; unscanned splits them into
  `layers_0 … layers_N`. The RL scripts ship both and vLLM requires unscanned.
  """

  def _round_trip(self, build):
    """Saves a trained model and restores it into a fresh one.

    Args:
      build: Callable taking rngs and returning the model to checkpoint.

    Returns:
      A tuple of the on-disk leaf paths, the restored step, and the params before and after.
    """
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model = build(nnx.Rngs(0))
      optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
      grads = jax.tree.map(lambda p: jnp.full_like(p, 0.1), nnx.state(model, nnx.Param))
      optimizer.update(model, grads)
      trained = jax.tree.leaves(nnx.state(model, nnx.Param))

      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1)
      )
      self.assertTrue(manager.save(1, model, optimizer, force=True))
      manager.wait_until_finished()
      keys = _on_disk_keys(d)

      restored_model = build(nnx.Rngs(0))
      restored_optimizer = nnx.Optimizer(restored_model, optax.adamw(1e-3), wrt=nnx.Param)
      step, _ = manager.maybe_restore(restored_model, restored_optimizer)
      manager.close()

      restored = jax.tree.leaves(nnx.state(restored_model, nnx.Param))
    return keys, step, trained, restored

  def test_scanned_layers_round_trip(self):
    keys, step, trained, restored = self._round_trip(_ScannedModel)
    self.assertEqual(step, 1)
    self.assertIn("params/params/layers", keys)
    self.assertEqual([k for k in keys if k.startswith("params/params/layers_")], [])
    for want, got in zip(trained, restored):
      self.assertTrue(jnp.array_equal(want, got))

  def test_unscanned_layers_round_trip(self):
    keys, step, trained, restored = self._round_trip(_UnscannedModel)
    self.assertEqual(step, 1)
    layer_keys = {k.split("/")[2] for k in keys if k.startswith("params/params/layers_")}
    self.assertEqual(layer_keys, {"layers_0", "layers_1", "layers_2"})
    for want, got in zip(trained, restored):
      self.assertTrue(jnp.array_equal(want, got))

  def test_the_two_layouts_are_actually_different_on_disk(self):
    """Guards the test itself: if both models wrote the same keys, neither case would prove much."""
    scanned, _, _, _ = self._round_trip(_ScannedModel)
    unscanned, _, _, _ = self._round_trip(_UnscannedModel)
    self.assertNotEqual(sorted(scanned), sorted(unscanned))


class PostTrainCheckpointMetadataReaderTest(unittest.TestCase):
  """The metadata has to come back through the function the loaders actually call."""

  def test_load_checkpoint_metadata_reads_what_the_manager_wrote(self):
    config = SimpleNamespace(
        scan_layers=False,
        lora=SimpleNamespace(lora_rank=8, model_dump=lambda: {"lora_rank": 8, "lora_alpha": 16.0}),
    )
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      model, optimizer = _build(wrapped=False)
      manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
          root_directory=d, options=ocp.CheckpointManagerOptions(save_interval_steps=1), config=config
      )
      manager.save(1, model, optimizer, force=True)
      manager.wait_until_finished()
      manager.close()

      # `verify_and_sync_scan_layers` and `sync_lora_metadata` both read a checkpoint this way.
      metadata = checkpointing.load_checkpoint_metadata(os.path.join(d, "1", "items"))

    self.assertIs(metadata.get("scan_layers"), False)
    self.assertEqual(metadata.get("lora"), {"lora_rank": 8, "lora_alpha": 16.0})


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


class PostTrainCheckpointBaseManagerTest(unittest.TestCase):
  """The base class builds a manager over Tunix's item names before we replace it."""

  def test_closes_the_base_class_manager_it_replaces(self):
    created = []
    real_cls = ocp.CheckpointManager

    class _Tracking(real_cls):
      """Records close calls so a replaced manager cannot be left open."""

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_calls = 0
        created.append(self)

      def close(self):
        self.close_calls += 1
        super().close()

    patches = [mock.patch.object(ocp, "CheckpointManager", _Tracking)]
    if hasattr(ocp, "checkpoint_manager") and hasattr(ocp.checkpoint_manager, "CheckpointManager"):
      patches.append(mock.patch.object(ocp.checkpoint_manager, "CheckpointManager", _Tracking))
    if hasattr(tunix_checkpoint_manager, "ocp") and hasattr(tunix_checkpoint_manager.ocp, "CheckpointManager"):
      patches.append(mock.patch.object(tunix_checkpoint_manager.ocp, "CheckpointManager", _Tracking))

    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      with contextlib.ExitStack() as stack:
        for p in patches:
          stack.enter_context(p)
        manager = post_train_checkpointing.MaxTextLayoutCheckpointManager(
            root_directory=d,
            options=ocp.CheckpointManagerOptions(save_interval_steps=1),
        )
      self.assertEqual(len(created), 2, "expected the base class's manager and its replacement")
      base, live = created[0], created[1]
      self.assertEqual(base.close_calls, 1, "the base class's manager was left open")
      self.assertEqual(live.close_calls, 0, "the live manager should still be open")
      manager.close()
      self.assertEqual(live.close_calls, 1)


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

  def test_forwards_the_run_config_to_the_manager(self):
    """The manager reads the config for the metadata it stamps, so `install` has to pass it on."""
    config = SimpleNamespace(scan_layers=False, lora=None)
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      trainer = self._FakeTrainer(*_build(wrapped=False), checkpoint_manager=None)
      post_train_checkpointing.install(trainer, d, config)
      trainer.checkpoint_manager.save(1, trainer.model, trainer.optimizer, force=True)
      trainer.checkpoint_manager.wait_until_finished()
      _, metadata = trainer.checkpoint_manager.maybe_restore(*_build(wrapped=False))
      trainer.checkpoint_manager.close()

    self.assertIs(metadata.get("scan_layers"), False)

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
