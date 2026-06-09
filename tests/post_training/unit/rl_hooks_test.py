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

"""Tests for post-train RL training hooks."""

import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

pytestmark = [pytest.mark.cpu_only, pytest.mark.post_training]

from maxtext.trainers.post_train.rl import hooks as rl_hooks
from maxtext.trainers.post_train.rl import utils_rl


def _make_trainer_config(**overrides):
  """Build a SimpleNamespace with the trainer-config attributes hooks reads."""
  defaults = {
      "num_test_batches": 5,
      "eval_interval": 10,
      "num_eval_passes": 1,
      "eval_corr_lst": False,
      "eval_make_lst": False,
  }
  defaults.update(overrides)
  return SimpleNamespace(**defaults)


def _make_rl_cluster(global_steps=0):
  cluster = SimpleNamespace()
  cluster.global_steps = global_steps
  cluster.actor_trainer = SimpleNamespace(training_hooks=None)
  return cluster


class RLTrainingHooksTest(unittest.TestCase):
  """Verify `RLTrainingHooks.on_train_step_end` step-gating + evaluate dispatch."""

  def setUp(self):
    super().setUp()
    eval_patcher = mock.patch.object(rl_hooks, "evaluate")
    self.mock_evaluate = eval_patcher.start()
    self.addCleanup(eval_patcher.stop)
    # evaluate(...) returns ((corr, total, acc, partial_acc, fmt_acc, mean_reward), _).
    self.mock_evaluate.return_value = ((1, 2, 50.0, 50.0, 100.0, 0.42), None)

  def _build_hook(self, eval_interval=10, global_steps=0):
    cluster = _make_rl_cluster(global_steps=global_steps)
    cfg = _make_trainer_config(eval_interval=eval_interval)
    return rl_hooks.RLTrainingHooks(cluster, cfg, test_dataset=None, eval_interval=eval_interval)

  def test_fires_on_matching_step(self):
    hook = self._build_hook(eval_interval=10, global_steps=10)
    hook.on_train_step_end(trainer=None, step=10, loss=None)
    self.mock_evaluate.assert_called_once()

  def test_skips_when_step_not_multiple_of_interval(self):
    hook = self._build_hook(eval_interval=10, global_steps=7)
    hook.on_train_step_end(trainer=None, step=7, loss=None)
    self.mock_evaluate.assert_not_called()

  def test_skips_when_step_is_zero(self):
    hook = self._build_hook(eval_interval=10, global_steps=0)
    hook.on_train_step_end(trainer=None, step=0, loss=None)
    self.mock_evaluate.assert_not_called()

  def test_dedupes_repeat_calls_on_same_step(self):
    hook = self._build_hook(eval_interval=10, global_steps=10)
    hook.on_train_step_end(trainer=None, step=10, loss=None)
    hook.on_train_step_end(trainer=None, step=10, loss=None)
    self.assertEqual(self.mock_evaluate.call_count, 1)

  def test_swallows_evaluate_exception(self):
    """A failing evaluate shouldn't propagate and break the training step."""
    self.mock_evaluate.side_effect = RuntimeError("boom")
    hook = self._build_hook(eval_interval=10, global_steps=10)
    hook.on_train_step_end(trainer=None, step=10, loss=None)  # must not raise

  def test_falls_back_to_step_arg_when_global_steps_unreadable(self):
    """When rl_cluster.global_steps raises, use the `step` arg instead."""

    class _ClusterWithBadGlobalSteps:
      """Stand-in rl_cluster whose `global_steps` property always raises."""

      def __init__(self):
        self.actor_trainer = SimpleNamespace(training_hooks=None)

      @property
      def global_steps(self):
        raise RuntimeError("not ready")

    bad_cluster = _ClusterWithBadGlobalSteps()
    cfg = _make_trainer_config(eval_interval=10)
    hook = rl_hooks.RLTrainingHooks(bad_cluster, cfg, test_dataset=None, eval_interval=10)
    hook.on_train_step_end(trainer=None, step=10, loss=None)
    self.mock_evaluate.assert_called_once()

  def test_reward_fns_plumbed_to_evaluate(self):
    """`reward_fns` passed to the hook is forwarded to evaluate(...)."""
    cluster = _make_rl_cluster(global_steps=10)
    cfg = _make_trainer_config(eval_interval=10)
    sentinel_reward_fns = [lambda **kw: [1.0]]
    hook = rl_hooks.RLTrainingHooks(cluster, cfg, test_dataset=None, eval_interval=10, reward_fns=sentinel_reward_fns)
    hook.on_train_step_end(trainer=None, step=10, loss=None)
    self.assertIs(self.mock_evaluate.call_args.kwargs["reward_fns"], sentinel_reward_fns)

  def test_reward_fns_default_none(self):
    """When `reward_fns` is omitted, evaluate(...) gets reward_fns=None."""
    hook = self._build_hook(eval_interval=10, global_steps=10)
    hook.on_train_step_end(trainer=None, step=10, loss=None)
    self.assertIsNone(self.mock_evaluate.call_args.kwargs["reward_fns"])


class InstallTrainingHooksTest(unittest.TestCase):
  """Verify `utils_rl.install_training_hooks` gating + attach behavior."""

  def test_noop_when_num_test_batches_nonpositive(self):
    cluster = _make_rl_cluster()
    cfg = _make_trainer_config(num_test_batches=0, eval_interval=10)
    utils_rl.install_training_hooks(cluster, cfg, test_dataset=None)
    self.assertIsNone(cluster.actor_trainer.training_hooks)

  def test_noop_when_eval_interval_nonpositive(self):
    cluster = _make_rl_cluster()
    cfg = _make_trainer_config(num_test_batches=5, eval_interval=0)
    utils_rl.install_training_hooks(cluster, cfg, test_dataset=None)
    self.assertIsNone(cluster.actor_trainer.training_hooks)

  def test_noop_when_eval_interval_attr_missing(self):
    cluster = _make_rl_cluster()
    cfg = SimpleNamespace(num_test_batches=5)  # no eval_interval attribute
    utils_rl.install_training_hooks(cluster, cfg, test_dataset=None)
    self.assertIsNone(cluster.actor_trainer.training_hooks)

  def test_attaches_hook_on_happy_path(self):
    cluster = _make_rl_cluster()
    cfg = _make_trainer_config(num_test_batches=5, eval_interval=10)
    utils_rl.install_training_hooks(cluster, cfg, test_dataset="dummy")
    self.assertIsInstance(cluster.actor_trainer.training_hooks, rl_hooks.RLTrainingHooks)

  def test_does_not_overwrite_existing_training_hooks(self):
    cluster = _make_rl_cluster()
    sentinel = object()
    cluster.actor_trainer.training_hooks = sentinel
    cfg = _make_trainer_config(num_test_batches=5, eval_interval=10)
    utils_rl.install_training_hooks(cluster, cfg, test_dataset=None)
    self.assertIs(cluster.actor_trainer.training_hooks, sentinel)

  def test_swallows_importerror_when_hooks_module_missing(self):
    """If `from .hooks import RLTrainingHooks` fails, install soft-skips.

    Setting `sys.modules[name] = None` makes Python's import system raise
    ImportError on the next import attempt for that name (documented behavior).
    """
    cluster = _make_rl_cluster()
    cfg = _make_trainer_config(num_test_batches=5, eval_interval=10)
    with mock.patch.dict("sys.modules", {"maxtext.trainers.post_train.rl.hooks": None}):
      utils_rl.install_training_hooks(cluster, cfg, test_dataset=None)
    self.assertIsNone(cluster.actor_trainer.training_hooks)


if __name__ == "__main__":
  unittest.main()
