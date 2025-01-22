"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Profiler tests."""
import sys
import unittest
import pytest

import profiler
import pyconfig


class ProfilerTest(unittest.TestCase):
  """Test for profiler."""

  # These periodic proilfer tests can run on any platform (cpu, gpu or tpu)
  @pytest.mark.tpu_only
  def test_periodic_profiler_third_period_starts(self):
    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        enable_checkpointing=False,
        run_name="test_periodic_profiler_starts_after_regular_profile",
        profiler="xplane",
        skip_first_n_steps_for_profiler=7,
        profiler_steps=4,
        profile_periodically_period=5,
    )
    config = pyconfig.config
    prof = profiler.Profiler(config, offset_step=2)

    step = 24  # 3 * 5 + 7 + 2: 3 periods of 5 after skipping initial 7 skip + 2 offset.
    assert prof.should_activate_periodic_profile(step)

  @pytest.mark.tpu_only
  def test_periodic_profiler_not_start_middle_period(self):
    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        enable_checkpointing=False,
        run_name="test_periodic_profiler_starts_after_regular_profile",
        profiler="xplane",
        skip_first_n_steps_for_profiler=7,
        profiler_steps=4,
        profile_periodically_period=5,
    )
    config = pyconfig.config
    prof = profiler.Profiler(config, offset_step=2)

    step = 25  # This corresponds to the middle of period 3 which started at step 24.
    assert not prof.should_activate_periodic_profile(step)

  @pytest.mark.tpu_only
  def test_periodic_profiler_third_period_ends(self):
    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        enable_checkpointing=False,
        run_name="test_periodic_profiler_starts_after_regular_profile",
        profiler="xplane",
        skip_first_n_steps_for_profiler=7,
        profiler_steps=4,
        profile_periodically_period=5,
    )
    config = pyconfig.config
    prof = profiler.Profiler(config, offset_step=2)

    step = 27  # 3 * 5 + 4 + 7 + 2: 3 periods of 5, profile takes 4 steps + skipping initial 7 skip + 2 offset
    assert prof.should_deactivate_periodic_profile(step)

  @pytest.mark.tpu_only
  def test_periodic_profiler_third_period_middle_not_end(self):
    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        enable_checkpointing=False,
        run_name="test_periodic_profiler_starts_after_regular_profile",
        profiler="xplane",
        skip_first_n_steps_for_profiler=7,
        profiler_steps=4,
        profile_periodically_period=5,
    )
    config = pyconfig.config
    prof = profiler.Profiler(config, offset_step=2)

    step = 28  # Corresponds to 1 after the third period ended.
    assert not prof.should_deactivate_periodic_profile(step)


if __name__ == "__main__":
  unittest.main()
