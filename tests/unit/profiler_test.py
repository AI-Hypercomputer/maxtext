# Copyright 2023–2025 Google LLC
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

"""Profiler tests."""
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.common import profiler
from tests.utils.test_helpers import get_test_config_path


class ProfilerTest(unittest.TestCase):
  """Test for profiler."""

  def setUp(self):
    super().setUp()
    # Mock jax.devices() to be deterministic and avoid runtime initialization errors
    self.mock_devices = [MagicMock(slice_index=0) for _ in range(1)]
    self.jax_patcher = patch("jax.devices", return_value=self.mock_devices)
    self.jax_patcher.start()
    self.jax_process_index_patcher = patch("jax.process_index", return_value=0)
    self.jax_process_index_patcher.start()

  def tearDown(self):
    self.jax_patcher.stop()
    self.jax_process_index_patcher.stop()
    super().tearDown()

  @pytest.mark.tpu_only
  def test_profiler_options_populated_from_config(self):
    """Verifies that Profiler initializes jax.profiler.ProfileOptions from config."""
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        run_name="test_profiler_options",
        base_output_directory="/tmp",
        profiler="xplane",
        xprof_tpu_power_trace_level=1,
        xprof_e2e_enable_fw_throttle_event=True,
        xprof_e2e_enable_fw_power_level_event=True,
        xprof_e2e_enable_fw_thermal_event=True,
    )

    with patch("jax.profiler.ProfileOptions") as mock_options_cls:
      # Setup mock return value
      mock_options_instance = MagicMock()
      mock_options_cls.return_value = mock_options_instance

      prof = profiler.Profiler(config)

      # Check if ProfileOptions was instantiated
      mock_options_cls.assert_called_once()

      # Verify advanced_configuration was populated
      expected_advanced_config = {
          "tpu_power_trace_level": types.XProfTPUPowerTraceMode.POWER_TRACE_NORMAL,
          "e2e_enable_fw_throttle_event": True,
          "e2e_enable_fw_power_level_event": True,
          "e2e_enable_fw_thermal_event": True,
      }
      self.assertEqual(prof.profiling_options.advanced_configuration, expected_advanced_config)

  @pytest.mark.tpu_only
  @patch("jax.profiler.start_trace")
  def test_profiler_activate_passes_options(self, mock_start_trace):
    """Verifies that activate() passes the profiling_options to jax.profiler.start_trace."""
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        run_name="test_profiler_options",
        base_output_directory="/tmp",
        profiler="xplane",
        xprof_tpu_power_trace_level=2,
    )

    # We need to mock ProfileOptions as well to check identity or value
    with patch("jax.profiler.ProfileOptions"):
      prof = profiler.Profiler(config)
      prof.activate()

      # Verify start_trace was called with profiler_options
      mock_start_trace.assert_called_once()
      _, kwargs = mock_start_trace.call_args
      self.assertIn("profiler_options", kwargs)
      self.assertEqual(kwargs["profiler_options"], prof.profiling_options)
      self.assertEqual(
          prof.profiling_options.advanced_configuration["tpu_power_trace_level"],
          types.XProfTPUPowerTraceMode.POWER_TRACE_SPI,
      )

  # These periodic proilfer tests can run on any platform (cpu, gpu or tpu)
  @pytest.mark.tpu_only
  def test_periodic_profiler_third_period_starts(self):
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        run_name="test_periodic_profiler_starts_after_regular_profile",
        profiler="xplane",
        skip_first_n_steps_for_profiler=7,
        profiler_steps=4,
        profile_periodically_period=5,
    )
    prof = profiler.Profiler(config, offset_step=2)

    step = 24  # 3 * 5 + 7 + 2: 3 periods of 5 after skipping initial 7 skip + 2 offset.
    assert prof.should_activate_periodic_profile(step)

  @pytest.mark.tpu_only
  def test_periodic_profiler_not_start_middle_period(self):
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        run_name="test_periodic_profiler_starts_after_regular_profile",
        profiler="xplane",
        skip_first_n_steps_for_profiler=7,
        profiler_steps=4,
        profile_periodically_period=5,
    )
    prof = profiler.Profiler(config, offset_step=2)

    step = 25  # This corresponds to the middle of period 3 which started at step 24.
    assert not prof.should_activate_periodic_profile(step)

  @pytest.mark.tpu_only
  def test_periodic_profiler_third_period_ends(self):
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        run_name="test_periodic_profiler_starts_after_regular_profile",
        profiler="xplane",
        skip_first_n_steps_for_profiler=7,
        profiler_steps=4,
        profile_periodically_period=5,
    )
    prof = profiler.Profiler(config, offset_step=2)

    step = 27  # 3 * 5 + 4 + 7 + 2: 3 periods of 5, profile takes 4 steps + skipping initial 7 skip + 2 offset
    assert prof.should_deactivate_periodic_profile(step)

  @pytest.mark.tpu_only
  def test_periodic_profiler_third_period_middle_not_end(self):
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        run_name="test_periodic_profiler_starts_after_regular_profile",
        profiler="xplane",
        skip_first_n_steps_for_profiler=7,
        profiler_steps=4,
        profile_periodically_period=5,
    )
    prof = profiler.Profiler(config, offset_step=2)

    step = 28  # Corresponds to 1 after the third period ended.
    assert not prof.should_deactivate_periodic_profile(step)


if __name__ == "__main__":
  unittest.main()
