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

"""With `skip_checkpoint_save_if_in_progress`, a save requested while the previous one is still being written is skipped.

Real Orbax on local disk. A save is kept in progress the way it is in production -- its background
half (storage writes and finalization) has not finished -- by handing Orbax one commit future that
completes only when the test says so. By default the next request waits for it and then saves; with
the option on it is skipped, so training is never blocked by a slow save and at most one save is in
flight. A save that has already failed is no longer in progress, so the skip never hides it from
`abandon_failed_checkpoint_saves`. The option is an Orbax save-decision policy and needs a single JAX
process; the last tests cover the policy on its own and that requirement.
"""

# pylint: disable=protected-access

import contextlib
import os
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx
import jax
from maxtext.training_engine import checkpointing
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import v1 as ocp_v1
from tests.unit.training_engine_checkpoint_roundtrip_test import _build, _leaves, _train_step
from tests.unit.training_engine_checkpoint_timeout_test import _StuckFuture, _Trained, _timeout_config

_policies = ocp_v1.training.save_decision_policies


def _skip_config(skip: bool, abandon: bool = False, timeout_secs: int = 60):
  config = _timeout_config(abandon=abandon, timeout_secs=timeout_secs)
  config.skip_checkpoint_save_if_in_progress = skip
  return config


def _skip_messages(logs):
  return [r.getMessage() for r in logs.records if r.getMessage().startswith("Skipping the checkpoint save")]


class _Call(threading.Thread):
  """Runs `fn` on its own thread, so a test can tell a call that returned from one that is still waiting."""

  def __init__(self, fn):
    super().__init__(daemon=True)
    self._fn = fn
    self.result = None
    self.error = None
    self.start()

  def run(self):
    try:
      self.result = self._fn()
    except Exception as e:  # pylint: disable=broad-except
      self.error = e

  def returned(self, within: float) -> bool:
    self.join(timeout=within)
    return not self.is_alive()


class TrainingEngineCheckpointSkipInProgressTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.ckpt_dir, ignore_errors=True)
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("ENABLE_PATHWAYS_PERSISTENCE", None)

  @contextlib.contextmanager
  def _stuck_commit(self):
    """Appends one commit future that completes only on `finish()` to the first item saved inside the context."""
    stuck = _StuckFuture()
    self.addCleanup(stuck.finish)  # Never leave Orbax's background thread waiting past the test.
    original = ocp.PyTreeCheckpointHandler.async_save
    armed = [True]

    async def async_save(handler, directory, *args, **kwargs):
      futures = list(await original(handler, directory, *args, **kwargs) or [])
      if armed[0]:
        armed[0] = False
        futures.append(stuck)
      return futures

    with mock.patch.object(ocp.PyTreeCheckpointHandler, "async_save", async_save):
      yield stuck

  def _wait_until_not_in_progress(self, manager, within: float = 30.0):
    """Waits for Orbax's background half of the in-flight save to end, successfully or not."""
    deadline = time.monotonic() + within
    while manager._checkpoint_manager.is_saving_in_progress():
      self.assertLess(time.monotonic(), deadline, "the in-flight save did not end")
      time.sleep(0.05)

  def _steps_on_disk(self):
    reader = checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=False))
    self.addCleanup(reader.close)
    return sorted(reader._checkpoint_manager.all_steps())

  def _assert_restores(self, trained, step):
    """A fresh manager restores `trained`'s weights from the checkpoint at `step`."""
    restorer = checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=False))
    self.addCleanup(restorer.close)
    fresh_model, fresh_opt = _build(seed=123)
    state = checkpointing.CheckpointState(model=fresh_model, optimizer=fresh_opt)
    restored, _, _ = restorer.restore_checkpoint(state, step=step)
    self.assertEqual(restored, step)
    want = _leaves(nnx.state(trained.model))
    got = _leaves(nnx.state(fresh_model))
    self.assertEqual(got.keys(), want.keys())
    for name, value in want.items():
      np.testing.assert_array_equal(got[name], value, err_msg=name)

  def test_a_save_requested_while_the_previous_one_is_still_writing_is_skipped(self):
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=True))
    trained = _Trained(1)
    with self._stuck_commit() as stuck:
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))
    _train_step(trained.model, trained.optimizer)

    # Step 1's write is still running, so step 2's request returns at once, declined, without
    # waiting for it; nothing changes about the save in flight.
    with self.assertLogs("absl", level="WARNING") as logs:
      call = _Call(lambda: manager.save_checkpoint(step=2, checkpoint_state=trained.state()))
      self.assertTrue(call.returned(within=10.0), "the request blocked on the save in progress")
    self.assertIsNone(call.error)
    self.assertFalse(call.result)
    (message,) = _skip_messages(logs)
    self.assertIn("Skipping the checkpoint save at step 2: the save at step 1 is still being written", message)
    self.assertEqual(manager._in_flight_step, 1)

    # Once step 1's background half has ended, the next request saves as usual.
    stuck.finish()
    self._wait_until_not_in_progress(manager)
    _train_step(trained.model, trained.optimizer)
    self.assertTrue(manager.save_checkpoint(step=3, checkpoint_state=trained.state()))
    manager.wait_until_finished()
    manager.close()
    self.assertEqual(self._steps_on_disk(), [1, 3])
    self._assert_restores(trained, step=3)

  def test_by_default_a_save_requested_while_the_previous_one_is_still_writing_waits_for_it(self):
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=False))
    trained = _Trained(1)
    with self._stuck_commit() as stuck:
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))
    _train_step(trained.model, trained.optimizer)

    call = _Call(lambda: manager.save_checkpoint(step=2, checkpoint_state=trained.state()))
    self.assertFalse(call.returned(within=1.0))  # Waiting on step 1's write, as before this option.
    stuck.finish()
    self.assertTrue(call.returned(within=30.0))
    self.assertIsNone(call.error)
    self.assertTrue(call.result)
    manager.wait_until_finished()
    manager.close()
    self.assertEqual(self._steps_on_disk(), [1, 2])

  def test_a_forced_save_still_waits_for_the_previous_one(self):
    # The engine forces the final save on close and the one after resuming mid-step; those must
    # reach disk, so they wait like before.
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=True))
    trained = _Trained(5)
    with self._stuck_commit() as stuck:
      self.assertTrue(manager.save_checkpoint(step=5, checkpoint_state=trained.state()))
    _train_step(trained.model, trained.optimizer)

    call = _Call(lambda: manager.save_checkpoint(step=6, checkpoint_state=trained.state(), force=True))
    self.assertFalse(call.returned(within=1.0))
    stuck.finish()
    self.assertTrue(call.returned(within=30.0))
    self.assertIsNone(call.error)
    self.assertTrue(call.result)
    manager.close()
    self.assertEqual(self._steps_on_disk(), [5, 6])
    self._assert_restores(trained, step=6)

  def test_a_timed_out_save_is_not_hidden_by_the_skip_and_is_abandoned_when_asked(self):
    # Orbax clears the in-progress flag when the background half ends, including by failing at the
    # deadline. The next request therefore drains the failed save instead of skipping, and
    # `abandon_failed_checkpoint_saves=true` drops it so step 2 is written.
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=True, abandon=True, timeout_secs=1))
    trained = _Trained(1)
    with self._stuck_commit():
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))
    self._wait_until_not_in_progress(manager)
    _train_step(trained.model, trained.optimizer)

    with self.assertLogs("absl", level="WARNING") as logs:
      self.assertTrue(manager.save_checkpoint(step=2, checkpoint_state=trained.state()))
    self.assertEqual(_skip_messages(logs), [])
    (message,) = [m for m in (r.getMessage() for r in logs.records) if m.startswith("Abandoning the checkpoint save")]
    self.assertIn("Abandoning the checkpoint save at step 1 (before saving step 2)", message)
    manager.wait_until_finished()
    manager.close()
    self.assertEqual(self._steps_on_disk(), [2])

  def test_a_timed_out_save_is_not_hidden_by_the_skip_and_is_raised_by_default(self):
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=True, abandon=False, timeout_secs=1))
    trained = _Trained(1)
    with self._stuck_commit():
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))
    self._wait_until_not_in_progress(manager)
    _train_step(trained.model, trained.optimizer)

    with self.assertLogs("absl", level="WARNING") as logs, self.assertRaises(TimeoutError):
      manager.save_checkpoint(step=2, checkpoint_state=trained.state())
    self.assertEqual(_skip_messages(logs), [])
    manager.close()

  def test_a_step_the_interval_policy_declines_is_declined_not_skipped(self):
    # The skip only applies to a step Orbax would save; a declined step is declined as before,
    # without the warning that names a step that will not be restorable.
    config = _skip_config(skip=True)
    config.checkpoint_period = 2
    manager = checkpointing.CheckpointManager(self.ckpt_dir, config)
    trained = _Trained(2)
    with self._stuck_commit() as stuck:
      self.assertTrue(manager.save_checkpoint(step=2, checkpoint_state=trained.state()))
    _train_step(trained.model, trained.optimizer)

    with mock.patch.object(checkpointing.logging, "warning", wraps=checkpointing.logging.warning) as warning:
      self.assertFalse(manager.save_checkpoint(step=3, checkpoint_state=trained.state()))
    self.assertEqual([c for c in warning.call_args_list if "Skipping the checkpoint save" in c.args[0]], [])
    stuck.finish()
    manager.wait_until_finished()
    manager.close()
    self.assertEqual(self._steps_on_disk(), [2])

  def test_synchronous_checkpointing_never_skips(self):
    # A synchronous save has finished by the time `save` returns, so nothing is ever in progress.
    config = _skip_config(skip=True)
    config.async_checkpointing = False
    manager = checkpointing.CheckpointManager(self.ckpt_dir, config)
    trained = _Trained(1)
    self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))
    _train_step(trained.model, trained.optimizer)
    self.assertTrue(manager.save_checkpoint(step=2, checkpoint_state=trained.state()))
    manager.close()
    self.assertEqual(self._steps_on_disk(), [1, 2])

  def test_the_policy_is_installed_only_when_asked(self):
    # With the option on, Orbax decides through the skip policy, which restates the default policy
    # for `checkpoint_period` (interval, preemption, first save). Off, Orbax's own default stays.
    config = _skip_config(skip=True)
    config.checkpoint_period = 3
    manager = checkpointing.CheckpointManager(self.ckpt_dir, config)
    self.addCleanup(manager.close)
    self.assertIsInstance(manager._skip_policy, checkpointing._SkipWhileSaveInProgressPolicy)
    self.assertIs(manager._checkpoint_manager._save_decision_policy, manager._skip_policy)
    default_config = _skip_config(skip=False)
    default_config.checkpoint_period = 3
    default = checkpointing.CheckpointManager(self.ckpt_dir, default_config)
    self.addCleanup(default.close)
    self.assertIsNone(default._skip_policy)
    self.assertIsInstance(default._checkpoint_manager._save_decision_policy, _policies.AnySavePolicy)
    # The restatement decides exactly as Orbax's default does (interval, first save, preemption).
    default_policy = default._checkpoint_manager._save_decision_policy
    self.assertEqual(manager._skip_policy._inner.policies[0].interval, 3)
    options = ocp.options.MultiprocessingOptions()
    for previous in ([], [SimpleNamespace(step=0)]):
      for reached_preemption in (False, True):
        context = _policies.DecisionContext(
            is_saving_in_progress=False, reached_preemption=reached_preemption, multiprocessing_options=options
        )
        for step in range(13):
          info = SimpleNamespace(step=step)
          self.assertEqual(
              manager._skip_policy._inner.should_save(info, previous, context=context),
              default_policy.should_save(info, previous, context=context),
              msg=f"step={step} previous={previous} reached_preemption={reached_preemption}",
          )

  def test_the_policy_declines_a_step_only_while_a_save_is_in_progress(self):
    step = SimpleNamespace(step=7)
    options = ocp.options.MultiprocessingOptions()

    def context(in_progress: bool):
      return _policies.DecisionContext(
          is_saving_in_progress=in_progress, reached_preemption=False, multiprocessing_options=options
      )

    policy = checkpointing._SkipWhileSaveInProgressPolicy(_policies.FixedIntervalPolicy(1))
    self.assertFalse(policy.should_save(step, [], context=context(True)))
    self.assertEqual(policy.take_declined_step(), 7)
    self.assertIsNone(policy.take_declined_step())  # Cleared once read.
    self.assertTrue(policy.should_save(step, [], context=context(False)))
    self.assertIsNone(policy.take_declined_step())

    # A step the interval policy declines is declined as before, not recorded as skipped.
    policy = checkpointing._SkipWhileSaveInProgressPolicy(_policies.FixedIntervalPolicy(2))
    self.assertFalse(policy.should_save(step, [], context=context(True)))
    self.assertIsNone(policy.take_declined_step())

  def test_the_option_needs_a_single_process(self):
    # Orbax tracks the in-progress flag per process, so with several processes the decision could
    # differ between them; the option is refused at construction, before Orbax's manager is built.
    with mock.patch.object(jax, "process_count", return_value=2):
      with mock.patch.object(checkpointing.ocp, "CheckpointManager") as orbax_manager:
        with self.assertRaisesRegex(ValueError, "skip_checkpoint_save_if_in_progress needs a single JAX process"):
          checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=True))
        orbax_manager.assert_not_called()
        # Off, several processes are fine.
        manager = checkpointing.CheckpointManager(self.ckpt_dir, _skip_config(skip=False))
        self.assertIsNone(manager._skip_policy)
        orbax_manager.assert_called_once()


if __name__ == "__main__":
  unittest.main()
