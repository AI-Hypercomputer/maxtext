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

"""A timed-out async save in `training_engine.checkpointing` is raised against its own step, or abandoned.

Real Orbax on local disk. The timeout is produced the way it happens in production -- the
background half of the save outlives `async_checkpointing_timeout_secs` -- by handing Orbax one
commit future that never completes on its own, under a 1-second deadline. By default the failure
is raised from the next checkpoint call; with `abandon_failed_checkpoint_saves=true` the save is
dropped and the run continues.
"""

# pylint: disable=protected-access

import contextlib
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from flax import nnx
from maxtext.training_engine import checkpointing
import numpy as np
import orbax.checkpoint as ocp
from tests.unit.training_engine_checkpoint_roundtrip_test import _build, _config, _leaves, _train_step


class _StuckFuture:
  """A commit future whose write runs until `finish()` is called, like a storage write the controller cannot stop."""

  name = "stuck_commit"

  def __init__(self):
    self._done = threading.Event()

  def result(self, timeout=None):
    if not self._done.wait(timeout):
      raise TimeoutError(f"{self.name} did not complete within {timeout} seconds.")

  def finish(self):
    self._done.set()


def _timeout_config(abandon: bool, timeout_secs: int = 1):
  config = _config()
  config.async_checkpointing_timeout_secs = timeout_secs
  config.abandon_failed_checkpoint_saves = abandon
  return config


def _abandon_messages(logs):
  return [r.getMessage() for r in logs.records if r.getMessage().startswith("Abandoning the checkpoint save")]


def _call(fn):
  """Runs `fn` and returns the exception it raised, or None; for asserting on another thread's outcome."""
  try:
    fn()
  except Exception as e:  # pylint: disable=broad-except
    return e
  return None


class _Trained:
  """A model/optimizer pair after `steps` updates, and the CheckpointState that saves it."""

  def __init__(self, steps):
    self.model, self.optimizer = _build(seed=0)
    for _ in range(steps):
      _train_step(self.model, self.optimizer)

  def state(self):
    return checkpointing.CheckpointState(model=self.model, optimizer=self.optimizer)


class TrainingEngineCheckpointTimeoutTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.ckpt_dir, ignore_errors=True)
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("ENABLE_PATHWAYS_PERSISTENCE", None)

  @contextlib.contextmanager
  def _stuck_commit(self):
    """Appends one never-completing commit future to the first item saved inside the context."""
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

  def _assert_restores(self, trained, step):
    """A fresh manager sees `step` as the latest checkpoint and restores `trained`'s weights from it."""
    restorer = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True))
    self.addCleanup(restorer.close)
    self.assertEqual(restorer.get_latest_step(), step)
    self.assertEqual(sorted(restorer._checkpoint_manager.all_steps()), [step])
    fresh_model, fresh_opt = _build(seed=123)
    restored, _, _ = restorer.restore_checkpoint(checkpointing.CheckpointState(model=fresh_model, optimizer=fresh_opt))
    self.assertEqual(restored, step)
    want = _leaves(nnx.state(trained.model))
    got = _leaves(nnx.state(fresh_model))
    self.assertEqual(got.keys(), want.keys())
    for name, value in want.items():
      np.testing.assert_array_equal(got[name], value, err_msg=name)

  def test_timed_out_save_is_abandoned_and_the_next_save_proceeds(self):
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True))
    trained = _Trained(1)
    with self._stuck_commit():
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))
    _train_step(trained.model, trained.optimizer)

    # The next save waits for the previous one, which fails at the 1s deadline. That failure is
    # absorbed here (not raised) and step 2 is saved.
    with self.assertLogs("absl", level="ERROR") as logs:
      self.assertTrue(manager.save_checkpoint(step=2, checkpoint_state=trained.state()))
    (message,) = _abandon_messages(logs)  # Orbax logs the background failure at ERROR too.
    self.assertIn("Abandoning the checkpoint save at step 1 (before saving step 2)", message)
    self.assertIn("TimeoutError", message)
    manager.wait_until_finished()
    manager.close()

    # Step 1 never became a checkpoint; step 2 is the latest and restores the live weights.
    self._assert_restores(trained, step=2)

  def test_the_abandoned_step_can_be_saved_again(self):
    # The engine's close() force-saves the current step. If that is the step just abandoned, the
    # save proceeds like any other and commits even though the abandoned attempt's write is still
    # running: nothing waits on that write any more.
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True))
    trained = _Trained(5)
    with self._stuck_commit():
      self.assertTrue(manager.save_checkpoint(step=5, checkpoint_state=trained.state()))

    with self.assertLogs("absl", level="ERROR") as logs:
      self.assertTrue(manager.save_checkpoint(step=5, checkpoint_state=trained.state(), force=True))
    (message,) = _abandon_messages(logs)
    self.assertIn("Abandoning the checkpoint save at step 5 (before saving step 5)", message)
    manager.close()  # The stuck write is still running here; close() does not wait for it.
    self._assert_restores(trained, step=5)

  def test_wait_until_finished_and_close_absorb_the_failure(self):
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True))
    trained = _Trained(1)
    with self._stuck_commit():
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))

    with self.assertLogs("absl", level="ERROR") as logs:
      manager.wait_until_finished()  # The public wait: returns instead of raising.
    (message,) = _abandon_messages(logs)
    self.assertIn("Abandoning the checkpoint save at step 1 (explicit wait)", message)
    self.assertIsNone(manager.get_latest_step())
    # The failure was consumed above, so nothing surfaces again on close.
    manager.close()

  def test_close_absorbs_a_timed_out_final_save(self):
    # The engine's close() does a forced final save and then closes the manager, which is where
    # the production run waited on the timed-out save.
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True))
    trained = _Trained(1)
    with self._stuck_commit():
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state(), force=True))
    with self.assertLogs("absl", level="ERROR") as logs:
      manager.close()
    (message,) = _abandon_messages(logs)
    self.assertIn("Abandoning the checkpoint save at step 1 (before close)", message)
    self.assertIsNone(checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True)).get_latest_step())

  def test_the_abandoned_failure_surfacing_on_another_thread_is_dropped(self):
    # Orbax's finalize thread re-raises its stored failure once per thread that joins it. The trainer
    # worker serves RPCs from a thread pool, so the thread that closes the engine need not be the one
    # that abandoned the save; it sees the same exception object, which is dropped at WARNING.
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True))
    trained = _Trained(1)
    with self._stuck_commit():
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))
    with self.assertLogs("absl", level="ERROR") as logs:
      manager.wait_until_finished()
    (message,) = _abandon_messages(logs)
    self.assertIn("step 1 (explicit wait)", message)

    failures = []
    with self.assertLogs("absl", level="WARNING") as logs:
      other = threading.Thread(target=lambda: failures.append(_call(manager.close)))
      other.start()
      other.join(timeout=30)
    self.assertFalse(other.is_alive())
    self.assertEqual(failures, [None])  # close() returned normally on the other thread.
    (message,) = [r.getMessage() for r in logs.records if "surfaced again" in r.getMessage()]
    self.assertIn("The already abandoned checkpoint save surfaced again on this thread (before close)", message)
    self.assertIn("TimeoutError", message)

  def test_a_wait_failure_with_no_save_in_flight_is_raised_even_when_abandoning(self):
    # Nothing was handed to Orbax, so a failure out of the wait cannot be a background save failure
    # this wrapper can attribute to a step; it is not the flag's business and must propagate.
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True))
    self.addCleanup(manager.close)
    with mock.patch.object(manager._checkpoint_manager, "wait_until_finished", side_effect=RuntimeError("barrier")):
      with self.assertLogs("absl", level="ERROR") as logs, self.assertRaises(RuntimeError):
        manager.wait_until_finished()
    (message,) = [r.getMessage() for r in logs.records if r.getMessage().startswith("Waiting on the checkpoint manager")]
    self.assertIn("failed with no save in flight (explicit wait): RuntimeError: barrier", message)

  def test_a_step_the_interval_policy_declines_does_not_wait_for_the_in_flight_save(self):
    # Orbax waits for the previous save only after `should_save`; the wrapper keeps that order, so
    # with checkpoint_period=2 the call for step 3 returns at once while step 2 is still writing.
    config = _timeout_config(abandon=True, timeout_secs=60)
    config.checkpoint_period = 2
    manager = checkpointing.CheckpointManager(self.ckpt_dir, config)
    trained = _Trained(2)
    with self._stuck_commit() as stuck:
      self.assertTrue(manager.save_checkpoint(step=2, checkpoint_state=trained.state()))
    _train_step(trained.model, trained.optimizer)

    start = time.perf_counter()
    self.assertFalse(manager.save_checkpoint(step=3, checkpoint_state=trained.state()))
    self.assertLess(time.perf_counter() - start, 10.0)
    self.assertEqual(manager._in_flight_step, 2)  # Untouched: nothing waited on it.

    stuck.finish()  # Let step 2's write complete so the save commits.
    manager.wait_until_finished()
    manager.close()
    self.assertEqual(checkpointing.CheckpointManager(self.ckpt_dir, config).get_latest_step(), 2)

  def test_failure_is_logged_against_its_step_and_raised_by_default(self):
    # `abandon_failed_checkpoint_saves=false` (the default): the failure of step 1's save is raised
    # from step 2's call, as before this change, but now with a log line that names step 1.
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=False))
    trained = _Trained(1)
    with self._stuck_commit():
      self.assertTrue(manager.save_checkpoint(step=1, checkpoint_state=trained.state()))
    _train_step(trained.model, trained.optimizer)
    with self.assertLogs("absl", level="ERROR") as logs, self.assertRaises(TimeoutError):
      manager.save_checkpoint(step=2, checkpoint_state=trained.state())
    (message,) = [m for m in (r.getMessage() for r in logs.records) if m.startswith("The background half")]
    self.assertIn("checkpoint save at step 1 failed with TimeoutError", message)
    self.assertIn("raising it from here (before saving step 2)", message)
    # Orbax raises a failure once; a later close finds nothing to raise.
    manager.close()

  def test_timeout_is_passed_to_orbax(self):
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True, timeout_secs=77))
    self.addCleanup(manager.close)
    self.assertEqual(manager._checkpoint_manager._checkpointer._async_manager._timeout_secs, 77)

  def test_a_save_that_finishes_in_time_is_unaffected(self):
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True, timeout_secs=60))
    trained = _Trained(2)
    self.assertTrue(manager.save_checkpoint(step=2, checkpoint_state=trained.state()))
    manager.wait_until_finished()
    manager.close()
    self.assertEqual(checkpointing.CheckpointManager(self.ckpt_dir, _timeout_config(abandon=True)).get_latest_step(), 2)

  def test_waits_are_no_ops_when_checkpointing_is_disabled(self):
    manager = checkpointing.CheckpointManager("", _timeout_config(abandon=True))  # No directory: no Orbax manager.
    self.assertIsNone(manager._checkpoint_manager)
    self.assertTrue(manager._drain_in_flight_save("explicit wait"))
    manager.wait_until_finished()
    manager.close()


if __name__ == "__main__":
  unittest.main()
