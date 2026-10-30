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

"""CPU-only tests of bench_driver.py's end-of-step helpers (no Tunix, no JAX)."""

import json
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench_driver as d  # pylint: disable=g-import-not-at-top,wrong-import-position


class _Worker:

  def __init__(self, timing=None, timing_error=None):
    self.updates = 0
    self._timing = timing
    self._timing_error = timing_error

  def update(self):
    self.updates += 1


class _TimingWorker(_Worker):

  def get_step_timing(self):
    if self._timing_error is not None:
      raise self._timing_error
    return self._timing


class FinishStepTest(unittest.TestCase):

  def test_fused_update_is_not_repeated(self):
    w = _Worker()
    resp = types.SimpleNamespace(metadata={"updated": True, "update_ready": True})
    self.assertFalse(d.finish_step(w, resp, 0))
    self.assertEqual(w.updates, 0)

  def test_older_worker_gets_explicit_update(self):
    for resp in (None, types.SimpleNamespace(metadata={}), types.SimpleNamespace(metadata=None)):
      w = _Worker()
      self.assertTrue(d.finish_step(w, resp, 0))
      self.assertEqual(w.updates, 1)


class LogStepTimingTest(unittest.TestCase):

  def test_logs_compact_json(self):
    timing = {"v": 1, "mb": [{"i": 0, "s": 1.5}]}
    with self.assertLogs(level="INFO") as cm:
      d.log_step_timing(_TimingWorker(timing=timing), 3)
    line = cm.output[-1]
    self.assertIn("TRAINER_STEP_TIMING step=3 rtt_s=0.000 timing=", line)
    self.assertEqual(json.loads(line.split("timing=", 1)[1]), timing)

  def test_worker_without_timing_logs_nothing(self):
    with self.assertNoLogs(level="INFO"):
      d.log_step_timing(_Worker(), 0)

  def test_timing_error_is_logged_not_raised(self):
    with self.assertLogs(level="INFO") as cm:
      d.log_step_timing(_TimingWorker(timing_error=RuntimeError("timer off")), 1)
    self.assertIn("TRAINER_STEP_TIMING step=1 unavailable", cm.output[-1])


if __name__ == "__main__":
  unittest.main()
