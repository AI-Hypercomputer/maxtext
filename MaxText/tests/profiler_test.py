# SPDX-License-Identifier: Apache-2.0

"""Profiler tests."""
import sys
import unittest
import pytest
import os.path

from MaxText import profiler
from MaxText import pyconfig
from MaxText.globals import PKG_DIR


class ProfilerTest(unittest.TestCase):
  """Test for profiler."""

  # These periodic proilfer tests can run on any platform (cpu, gpu or tpu)
  @pytest.mark.tpu_only
  def test_periodic_profiler_third_period_starts(self):
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
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
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
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
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
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
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
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
