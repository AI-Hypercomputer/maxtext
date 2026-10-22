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

"""Offline tests for the run row a tick appends when it learns that a pull request merged.

A run read while its pull request was open is stored with no merge time. A later tick that learns of the merge
re-reads the run with `get_run` and appends a corrected row. A single run's payload carries no superseded mark
(only a listing adds one), so the corrected row keeps the mark the run was stored with: a superseded run stays
out of the views after its pull request merges.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import rows
from collector import store
from collector import tick
from collector import views

# These tests drive the tick's private steps one at a time.
# pylint: disable=protected-access

NOW = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc)
MONTH = "2026-11"
NUMBER = 6201
RUN_ID = 42900000001
CREATED = "2026-11-08T01:00:00Z"


def run_payload() -> dict[str, Any]:
  """The run as `get_run` answers it: no superseded mark."""
  return {
      "id": RUN_ID,
      "run_attempt": 1,
      "name": "CI Pipeline",
      "path": ".github/workflows/ci_pipeline.yml",
      "event": "pull_request",
      "status": "completed",
      "conclusion": "cancelled",
      "created_at": CREATED,
      "head_sha": "aaa",
  }


def pull(merged: bool) -> dict[str, Any]:
  """The pull request, open or merged."""
  return {
      "number": NUMBER,
      "state": "closed" if merged else "open",
      "title": f"PR {NUMBER}",
      "merged_at": "2026-11-08T02:00:00Z" if merged else None,
      "head": {"sha": "fff"},
      "base": {"ref": "main"},
      "user": {"login": "copybara-service[bot]"},
  }


class StoreMergeTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    tmp_dir = tempfile.mkdtemp(prefix="ci-metrics-merge-")
    self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
    self.out = Path(tmp_dir)
    self.row_store = store.Store(tmp_dir)

  def merge(self, superseded: bool | None) -> dict[str, Any]:
    """Stores the run as `_collect_run` does while its pull request is open, learns the merge, reads the row back."""
    listed = dict(run_payload(), superseded=superseded)
    self.row_store.append(
        rows.KIND_RUN, [rows.run_row(listed, pull(merged=False), collected_at="2026-11-08T01:30:00Z")], month=MONTH
    )
    state = self.row_store.load_state()
    state.mark_collected(RUN_ID, 1, CREATED)
    subject = tick.Tick(tick.Options(out=self.out), mock.Mock(), self.row_store, state, tick.TickReport())
    subject.now = NOW
    subject.collected_at = tick.iso_utc(NOW)

    waiting = subject._open_pull_requests()
    self.assertEqual(waiting, {NUMBER: RUN_ID})
    with mock.patch.object(tick.runs, "get_run", return_value=run_payload()):
      self.assertTrue(subject._store_merge(waiting[NUMBER], pull(merged=True)))
    stored = [payload for payload in self.row_store.read(rows.KIND_RUN, [MONTH]) if payload["run_id"] == RUN_ID]
    self.assertEqual(len(stored), 1)
    self.assertEqual(stored[0]["pr_merged_at"], "2026-11-08T02:00:00Z")
    return stored[0]

  def test_a_superseded_run_stays_superseded_after_its_merge_is_learned(self):
    self.assertIs(self.merge(superseded=True)["superseded"], True)
    data = views.load_month(self.row_store, MONTH)
    self.assertNotIn(RUN_ID, data.kept_run_ids)
    self.assertNotIn(NUMBER, data.merged_pr_runs)

  def test_a_run_that_was_not_superseded_stays_in_the_views(self):
    self.assertIs(self.merge(superseded=False)["superseded"], False)
    data = views.load_month(self.row_store, MONTH)
    self.assertEqual(data.merged_pr_runs.get(NUMBER), RUN_ID)

  def test_a_run_stored_with_no_mark_gets_none(self):
    self.assertIsNone(self.merge(superseded=None)["superseded"])

  def test_rebuild_views_flag_rewrites_untouched_months_in_the_window(self):
    october = dict(run_payload(), id=42000000001, created_at="2026-10-15T01:00:00Z")
    self.row_store.append(rows.KIND_RUN, [rows.run_row(october, pull(merged=True))], month="2026-10")
    self.row_store.append(rows.KIND_RUN, [rows.run_row(run_payload(), pull(merged=True))], month=MONTH)
    state = self.row_store.load_state()
    views.build_views(self.row_store, self.out, NOW.date(), months=["2026-10", MONTH], uncollected=0)

    report = tick.TickReport()
    report.months_touched.add(MONTH)
    tick.rebuild_views(tick.Options(out=self.out), self.row_store, state, report, NOW)
    self.assertEqual(report.months_rebuilt, [MONTH])

    parsed = tick.build_options(tick.parse_args(["--out", str(self.out), "--rebuild-views"]))
    self.assertTrue(parsed.rebuild_views)
    report_all = tick.TickReport()
    report_all.months_touched.add(MONTH)
    tick.rebuild_views(parsed, self.row_store, state, report_all, NOW)
    self.assertEqual(report_all.months_rebuilt, ["2026-10", MONTH])


if __name__ == "__main__":
  unittest.main()
