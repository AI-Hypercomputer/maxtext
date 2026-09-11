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

"""Tests for the training engine's micro-step profiler."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from maxtext.configs import pyconfig
from maxtext.training_engine import micro_step_profiler
from tests.utils.test_helpers import get_test_config_path


def _config(**overrides):
  """Builds a config with profiling enabled and a GCS output directory."""
  base = {
      "enable_checkpointing": False,
      "run_name": "micro_step_profiler_test",
      "base_output_directory": "gs://test-bucket/out",
      "profiler": "xplane",
  }
  base.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **base)


class _WithoutOutputDir:
  """A real config with `tensorboard_dir` forced to an unset value.

  `pyconfig.initialize` backfills `base_output_directory` and `run_name`, so it cannot produce
  an unset `tensorboard_dir` however it is called; overriding the one key is the only way to
  reach the case.
  """

  def __init__(self, config, tensorboard_dir):
    self._config = config
    self.tensorboard_dir = tensorboard_dir

  def __getattr__(self, name):
    return getattr(self._config, name)


def _drive(prof, micro_steps):
  """Runs `prof` over `micro_steps` micro-steps and returns the ones it captured."""
  captured = []
  for n in range(micro_steps):
    prof.maybe_activate(n)
    if prof.is_active:
      captured.append(n)
    prof.maybe_deactivate(n)
  return captured


class MicroStepProfilerWindowTest(unittest.TestCase):
  """Window placement, which is pure arithmetic over the cumulative micro-step counter."""

  def setUp(self):
    super().setUp()
    self.start_trace = patch("jax.profiler.start_trace").start()
    self.stop_trace = patch("jax.profiler.stop_trace").start()
    self.addCleanup(patch.stopall)

  def test_window_covers_exactly_the_requested_micro_steps(self):
    prof = micro_step_profiler.MicroStepProfiler(_config(skip_first_n_steps_for_profiler=3, profiler_steps=4))

    self.assertEqual(_drive(prof, 12), [3, 4, 5, 6])
    self.start_trace.assert_called_once()
    self.stop_trace.assert_called_once()

  def test_single_micro_step_window(self):
    prof = micro_step_profiler.MicroStepProfiler(_config(skip_first_n_steps_for_profiler=2, profiler_steps=1))

    self.assertEqual(_drive(prof, 6), [2])

  def test_window_at_micro_step_zero(self):
    prof = micro_step_profiler.MicroStepProfiler(_config(skip_first_n_steps_for_profiler=0, profiler_steps=2))

    self.assertEqual(_drive(prof, 6), [0, 1])

  def test_output_path_names_the_first_micro_step_of_the_window(self):
    config = _config(skip_first_n_steps_for_profiler=5, profiler_steps=2)
    prof = micro_step_profiler.MicroStepProfiler(config)

    _drive(prof, 8)

    path = self.start_trace.call_args[0][0]
    self.assertEqual(path, f"{config.tensorboard_dir.rstrip('/')}/micro_step_5")

  def test_periodic_windows_repeat(self):
    prof = micro_step_profiler.MicroStepProfiler(
        _config(skip_first_n_steps_for_profiler=2, profiler_steps=2, profile_periodically_period=5)
    )

    # Windows at 2-3, 7-8, 12-13.
    self.assertEqual(_drive(prof, 15), [2, 3, 7, 8, 12, 13])
    self.assertEqual(self.start_trace.call_count, 3)
    self.assertEqual(self.stop_trace.call_count, 3)

  def test_periodic_window_does_not_open_before_the_first_one(self):
    """Python's modulo is non-negative, so an unguarded congruence fires early."""
    prof = micro_step_profiler.MicroStepProfiler(
        _config(skip_first_n_steps_for_profiler=10, profiler_steps=2, profile_periodically_period=5)
    )

    self.assertEqual(_drive(prof, 10), [])
    self.start_trace.assert_not_called()

  def test_overlapping_period_falls_back_to_a_single_window(self):
    prof = micro_step_profiler.MicroStepProfiler(
        _config(skip_first_n_steps_for_profiler=1, profiler_steps=5, profile_periodically_period=3)
    )

    self.assertEqual(_drive(prof, 20), [1, 2, 3, 4, 5])
    self.assertEqual(self.start_trace.call_count, 1)


class MicroStepProfilerDisabledTest(unittest.TestCase):
  """Cases that must disable profiling rather than raise: a bad window must not kill a run."""

  def setUp(self):
    super().setUp()
    self.start_trace = patch("jax.profiler.start_trace").start()
    patch("jax.profiler.stop_trace").start()
    self.addCleanup(patch.stopall)

  def test_profiling_off_by_default(self):
    prof = micro_step_profiler.MicroStepProfiler(_config(profiler=""))

    self.assertEqual(_drive(prof, 10), [])
    self.start_trace.assert_not_called()

  def test_nsys_is_not_supported(self):
    prof = micro_step_profiler.MicroStepProfiler(_config(profiler="nsys"))

    self.assertEqual(_drive(prof, 10), [])
    self.start_trace.assert_not_called()

  def test_empty_window_disables_profiling(self):
    prof = micro_step_profiler.MicroStepProfiler(_config(skip_first_n_steps_for_profiler=3, profiler_steps=0))

    self.assertEqual(_drive(prof, 10), [])
    self.start_trace.assert_not_called()

  def test_managed_mldiagnostics_warns_but_still_profiles(self):
    """The engine does not route through the collector, so an operator must be told where to look."""
    with self.assertLogs(level="WARNING") as logs:
      prof = micro_step_profiler.MicroStepProfiler(
          _config(skip_first_n_steps_for_profiler=0, profiler_steps=2, managed_mldiagnostics=True)
      )

    self.assertTrue(any("managed_mldiagnostics" in line for line in logs.output))
    self.assertEqual(_drive(prof, 4), [0, 1])

  def test_unset_output_directory_disables_profiling(self):
    """`pyconfig` always backfills `tensorboard_dir`, so this only guards a config built by hand.

    Worth guarding anyway: `MaxTextConfig` leaves it None when constructed directly without a
    `run_name`/`base_output_directory`, and the trace path would then be built out of the unset
    value -- `None/micro_step_0`, or `/micro_step_0` at the filesystem root for an empty string.
    """
    for unset in (None, ""):
      with self.subTest(tensorboard_dir=unset):
        self.start_trace.reset_mock()
        config = _WithoutOutputDir(_config(skip_first_n_steps_for_profiler=0, profiler_steps=2), unset)

        with self.assertLogs(level="WARNING") as logs:
          prof = micro_step_profiler.MicroStepProfiler(config)

        self.assertTrue(any("tensorboard_dir" in line for line in logs.output))
        self.assertEqual(_drive(prof, 4), [])
        self.start_trace.assert_not_called()

  def test_a_backend_that_refuses_to_start_does_not_raise(self):
    self.start_trace.side_effect = RuntimeError("another trace is already running")
    prof = micro_step_profiler.MicroStepProfiler(_config(skip_first_n_steps_for_profiler=1, profiler_steps=2))

    self.assertEqual(_drive(prof, 6), [])
    self.assertFalse(prof.is_active)


class MicroStepProfilerCleanlyTest(unittest.TestCase):
  """The device drains that make an asynchronously dispatched window mean what it says."""

  def setUp(self):
    super().setUp()
    self.start_trace = patch("jax.profiler.start_trace").start()
    self.stop_trace = patch("jax.profiler.stop_trace").start()
    self.block = patch("jax.block_until_ready").start()
    self.addCleanup(patch.stopall)

  def test_drains_before_start_and_blocks_before_stop(self):
    prof = micro_step_profiler.MicroStepProfiler(
        _config(skip_first_n_steps_for_profiler=1, profiler_steps=2, profile_cleanly=True)
    )
    order = []
    drain = MagicMock(side_effect=lambda: order.append("drain"))
    self.start_trace.side_effect = lambda *a, **k: order.append("start")
    self.stop_trace.side_effect = lambda *a, **k: order.append("stop")
    self.block.side_effect = lambda *a, **k: order.append("block")
    accumulator = object()

    for n in range(4):
      prof.maybe_activate(n, drain=drain)
      prof.maybe_deactivate(n, block_on=accumulator)

    self.assertEqual(order, ["drain", "start", "block", "stop"])
    self.block.assert_called_once_with(accumulator)

  def test_profile_cleanly_false_skips_both(self):
    prof = micro_step_profiler.MicroStepProfiler(
        _config(skip_first_n_steps_for_profiler=1, profiler_steps=2, profile_cleanly=False)
    )
    drain = MagicMock()

    for n in range(4):
      prof.maybe_activate(n, drain=drain)
      prof.maybe_deactivate(n, block_on=object())

    drain.assert_not_called()
    self.block.assert_not_called()
    self.start_trace.assert_called_once()


class MicroStepProfilerHostsTest(unittest.TestCase):
  """`upload_all_profiler_results` maps onto Pathways' `max_num_hosts`."""

  def setUp(self):
    super().setUp()
    self.start_trace = patch("jax.profiler.start_trace").start()
    patch("jax.profiler.stop_trace").start()
    self.addCleanup(patch.stopall)

  def test_one_host_by_default(self):
    prof = micro_step_profiler.MicroStepProfiler(
        _config(skip_first_n_steps_for_profiler=0, profiler_steps=1, upload_all_profiler_results=False)
    )

    _drive(prof, 2)

    self.assertNotIn("max_num_hosts", self.start_trace.call_args[1])

  def test_all_hosts_derived_from_the_device_list(self):
    devices = [MagicMock(process_index=i // 4) for i in range(12)]
    with patch("jax.devices", return_value=devices):
      prof = micro_step_profiler.MicroStepProfiler(
          _config(skip_first_n_steps_for_profiler=0, profiler_steps=1, upload_all_profiler_results=True)
      )
      _drive(prof, 2)

    self.assertEqual(self.start_trace.call_args[1]["max_num_hosts"], 3)

  def test_backend_without_max_num_hosts_retries_without_it(self):
    devices = [MagicMock(process_index=i) for i in range(2)]
    self.start_trace.side_effect = [TypeError("unexpected keyword argument 'max_num_hosts'"), None]
    with patch("jax.devices", return_value=devices):
      prof = micro_step_profiler.MicroStepProfiler(
          _config(skip_first_n_steps_for_profiler=0, profiler_steps=1, upload_all_profiler_results=True)
      )
      _drive(prof, 2)

    self.assertEqual(self.start_trace.call_count, 2)
    self.assertNotIn("max_num_hosts", self.start_trace.call_args[1])


class MicroStepProfilerCloseTest(unittest.TestCase):
  """A run ending mid-window must still write its profile."""

  def setUp(self):
    super().setUp()
    patch("jax.profiler.start_trace").start()
    self.stop_trace = patch("jax.profiler.stop_trace").start()
    self.addCleanup(patch.stopall)

  def test_close_stops_an_open_trace(self):
    prof = micro_step_profiler.MicroStepProfiler(_config(skip_first_n_steps_for_profiler=0, profiler_steps=100))

    prof.maybe_activate(0)
    self.assertTrue(prof.is_active)
    prof.close()

    self.stop_trace.assert_called_once()
    self.assertFalse(prof.is_active)

  def test_close_summary_names_no_closing_micro_step(self):
    """The window never reached its end, so the summary must not report one."""
    prof = micro_step_profiler.MicroStepProfiler(_config(skip_first_n_steps_for_profiler=7, profiler_steps=100))

    prof.maybe_activate(7)
    with self.assertLogs(level="INFO") as logs:
      prof.close()

    summary = next(line for line in logs.output if "Profile written" in line)
    self.assertIn("micro-steps 7 onwards", summary)
    self.assertNotIn("None", summary)

  def test_close_is_a_noop_when_no_trace_is_open(self):
    prof = micro_step_profiler.MicroStepProfiler(_config(skip_first_n_steps_for_profiler=5, profiler_steps=2))

    prof.close()

    self.stop_trace.assert_not_called()


if __name__ == "__main__":
  unittest.main()
