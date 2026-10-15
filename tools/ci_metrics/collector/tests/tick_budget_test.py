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

"""Offline tests for the tick's API budget guard.

GITHUB_TOKEN's 1,000 requests an hour are shared by every workflow in the repository, and one
tick of 40 runs spends about 500. So the tick checks the budget before each run it reads,
before each in-flight re-read and before the merge checks, and stops cleanly when fewer than
`RATE_LIMIT_FLOOR` requests are left. It reads the budget two ways, from `/rate_limit` and from
the `X-RateLimit-Remaining` header of the latest answer, and trusts the lower one. A fake client
answers both readings and fails the test on anything else, so nothing here reaches GitHub.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import github
from collector import store
from collector import tick

# These tests drive the tick's private steps one at a time.
# pylint: disable=protected-access

# A run that is still going is never "already stored", so the tick would read it.
RUNNING = {"id": 101, "status": "in_progress", "run_attempt": 1, "created_at": "2026-10-15T00:00:00Z"}

# Pull requests waiting for the merge checks: enough for the budget to be checked twice inside the loop.
WAITING_PULL_REQUESTS = 60


class FakeClient:
  """Answers the budget question. Anything else would be a real request, so it fails the test."""

  def __init__(self, remaining: int | Exception, hint: int | None = None) -> None:
    self.remaining = remaining
    self.remaining_hint = hint
    self.budget_reads = 0

  def rate_limit(self) -> dict[str, int]:
    """Returns the scripted budget, or raises the scripted error."""
    self.budget_reads += 1
    if isinstance(self.remaining, Exception):
      raise self.remaining
    # The real client overwrites its header count with this reading, so the guard has to read the hint first.
    self.remaining_hint = self.remaining
    return {"limit": 1000, "remaining": self.remaining, "reset": 0}

  def __getattr__(self, name: str) -> Any:
    raise AssertionError(f"the tick asked the fake client for {name!r}; these tests must not reach GitHub")


class TickBudgetTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp(prefix="ci-metrics-budget-")
    self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
    patch = mock.patch.object(tick.Tick, "_collect_run")
    self.collect_run = patch.start()
    self.addCleanup(patch.stop)

  def make_tick(
      self, remaining: int | Exception, pending: tuple[int, ...] = (), hint: int | None = None
  ) -> tuple[tick.Tick, FakeClient]:
    """Returns a tick over an empty store and a one-day window, and its fake client."""
    row_store = store.Store(self.tmp_dir)
    state = row_store.load_state()
    for run_id in pending:
      state.mark_pending(run_id, 1, created_at="2026-10-15T00:00:00Z")
    client = FakeClient(remaining, hint=hint)
    options = tick.Options(out=Path(self.tmp_dir), backfill_days=1)
    return tick.Tick(options, client, row_store, state, tick.TickReport()), client

  def test_floor_is_300_requests(self):
    self.assertEqual(tick.RATE_LIMIT_FLOOR, 300)

  def test_low_budget_stops_before_a_new_run(self):
    subject, _ = self.make_tick(remaining=tick.RATE_LIMIT_FLOOR - 1)
    self.assertFalse(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_not_called()
    self.assertIn("API budget ran low", subject.report.stopped_because)
    # A budget stop is not lost data: the tick still exits 0 and the workflow still saves.
    self.assertEqual(subject.report.problems, [])

  def test_budget_at_the_floor_carries_on(self):
    subject, _ = self.make_tick(remaining=tick.RATE_LIMIT_FLOOR)
    self.assertTrue(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_called_once()
    self.assertIsNone(subject.report.stopped_because)

  def test_unreadable_budget_carries_on(self):
    subject, _ = self.make_tick(remaining=github.GitHubError("the rate-limit endpoint is down", status=503))
    printed = io.StringIO()
    with contextlib.redirect_stderr(printed):
      self.assertTrue(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_called_once()
    self.assertIn("could not be read", printed.getvalue())

  def test_lower_header_count_stops(self):
    subject, _ = self.make_tick(remaining=1000, hint=tick.RATE_LIMIT_FLOOR - 1)
    self.assertFalse(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_not_called()
    self.assertIn(f"{tick.RATE_LIMIT_FLOOR - 1} of", subject.report.stopped_because)

  def test_higher_header_count_does_not_hide_a_low_budget(self):
    subject, _ = self.make_tick(remaining=tick.RATE_LIMIT_FLOOR - 1, hint=1000)
    self.assertFalse(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_not_called()

  def test_header_count_decides_when_the_endpoint_is_down(self):
    subject, _ = self.make_tick(
        remaining=github.GitHubError("the rate-limit endpoint is down", status=503), hint=tick.RATE_LIMIT_FLOOR - 1
    )
    self.assertFalse(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_not_called()
    self.assertEqual(subject.report.problems, [])

  def test_already_stored_run_reads_no_budget(self):
    subject, client = self.make_tick(remaining=0)
    with mock.patch.object(tick.Tick, "_already_stored", return_value=True):
      self.assertTrue(subject._collect_one(dict(RUNNING)))
    self.assertEqual(client.budget_reads, 0)
    self.collect_run.assert_not_called()

  def test_low_budget_stops_in_flight_rereads(self):
    subject, _ = self.make_tick(remaining=tick.RATE_LIMIT_FLOOR - 1, pending=(202,))
    with mock.patch.object(tick.runs, "get_run") as get_run:
      subject._collect_pending()
    get_run.assert_not_called()
    self.assertIn("API budget ran low", subject.report.stopped_because)

  def test_low_budget_skips_merge_checks(self):
    subject, _ = self.make_tick(remaining=tick.RATE_LIMIT_FLOOR - 1)
    with mock.patch.object(tick.Tick, "_list_slice", return_value=[]):
      with mock.patch.object(tick.Tick, "_refresh_merges") as merges:
        subject.collect()
    merges.assert_not_called()

  def test_enough_budget_checks_merges(self):
    subject, _ = self.make_tick(remaining=tick.RATE_LIMIT_FLOOR)
    with mock.patch.object(tick.Tick, "_list_slice", return_value=[]):
      with mock.patch.object(tick.Tick, "_refresh_merges") as merges:
        subject.collect()
    merges.assert_called_once()

  def run_merge_checks(self, remaining: int) -> tuple[tick.Tick, FakeClient, mock.Mock]:
    """Runs the merge checks over waiting pull requests that are all still open; returns the pull request reads."""
    subject, client = self.make_tick(remaining=remaining)
    waiting = {number: 1000 + number for number in range(1, WAITING_PULL_REQUESTS + 1)}
    with mock.patch.object(tick.Tick, "_open_pull_requests", return_value=waiting):
      with mock.patch.object(tick.runs, "get_pull_request", return_value={"state": "open"}) as get_pull_request:
        subject._refresh_merges()
    return subject, client, get_pull_request

  def test_low_budget_stops_merge_checks_part_way(self):
    subject, _, get_pull_request = self.run_merge_checks(remaining=tick.RATE_LIMIT_FLOOR - 1)
    self.assertEqual(get_pull_request.call_count, tick.MERGE_BUDGET_CHECK_EVERY)
    self.assertIn("API budget ran low", subject.report.stopped_because)
    # A budget stop is not lost data: the tick still exits 0 and the workflow still saves.
    self.assertEqual(subject.report.problems, [])

  def test_enough_budget_finishes_merge_checks(self):
    subject, client, get_pull_request = self.run_merge_checks(remaining=tick.RATE_LIMIT_FLOOR)
    self.assertEqual(get_pull_request.call_count, WAITING_PULL_REQUESTS)
    self.assertIsNone(subject.report.stopped_because)
    # The budget is read after every 25th pull request (here twice), not before each one.
    self.assertEqual(client.budget_reads, 2)


if __name__ == "__main__":
  unittest.main()
