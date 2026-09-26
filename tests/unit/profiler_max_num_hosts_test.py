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

"""Tests for the xplane profiler with `profiler_max_num_hosts` (multi-host traces on Pathways).

These live outside profiler_test.py because pytest.ini ignores that file, so the default test run would not collect
them. Config validation of `profiler_max_num_hosts` is tested in pyconfig_test.py.
"""

import sys
import tempfile
import unittest
from unittest import mock

import jax

from maxtext.common import profiler
from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path

# pylint: disable=missing-function-docstring


def _pathways_start_trace(
    log_dir, create_perfetto_link=False, create_perfetto_trace=False, profiler_options=None, max_num_hosts=1
):
  """Signature of the jax.profiler.start_trace that pathwaysutils.initialize() installs on Pathways."""
  del log_dir, create_perfetto_link, create_perfetto_trace, profiler_options, max_num_hosts


def _config(**overrides):
  with mock.patch("pathwaysutils.is_pathways_backend_used", return_value=True):
    return pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        run_name="test_profiler_max_num_hosts",
        base_output_directory=tempfile.gettempdir(),
        profiler="xplane",
        **overrides,
    )


class ProfilerMaxNumHostsTest(unittest.TestCase):
  """Autospecs stand in for start_trace, so an argument the installed function does not take raises TypeError."""

  @mock.patch("jax.profiler.stop_trace", autospec=True)
  def test_traces_configured_number_of_hosts_with_pathways_start_trace(self, stop_trace):
    with mock.patch("jax.profiler.start_trace", mock.create_autospec(_pathways_start_trace)) as start_trace:
      prof = profiler.Profiler(_config(profiler_max_num_hosts=4))
      prof.activate()
      prof.deactivate()
    start_trace.assert_called_once_with(prof.output_path, profiler_options=prof.profiling_options, max_num_hosts=4)
    stop_trace.assert_called_once_with()

  @mock.patch("jax.profiler.start_trace", mock.create_autospec(jax.profiler.start_trace))
  def test_multiple_hosts_without_pathways_start_trace_fail_at_construction(self):
    # As in an entry point that does not call pathwaysutils.initialize(): fail here rather than at activate().
    with self.assertRaisesRegex(ValueError, r"requires pathwaysutils.initialize\(\)"):
      profiler.Profiler(_config(profiler_max_num_hosts=4))

  @mock.patch("jax.profiler.stop_trace", autospec=True)
  @mock.patch("jax.profiler.start_trace", autospec=True)
  def test_default_host_count_keeps_jax_start_trace_call(self, start_trace, stop_trace):
    prof = profiler.Profiler(_config())
    prof.activate()
    prof.deactivate()
    start_trace.assert_called_once_with(prof.output_path, profiler_options=prof.profiling_options)
    stop_trace.assert_called_once_with()


if __name__ == "__main__":
  unittest.main()
