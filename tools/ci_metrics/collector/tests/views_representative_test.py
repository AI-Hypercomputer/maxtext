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

"""Offline tests for which run represents a merged pull request in the views.

A merged pull request is drawn from one run: the newest run on its final head, unless that run ran no tests and
an earlier run on the final head did (and ended in success or failure). A Copybara pull request gets one more
push when the change is submitted inside Google, and GitHub marks it merged seconds later, so the run on that push
tests after the merge. When that run is the one left, such a pull request is drawn from its newest run on an
earlier head that ran tests and ended in success or failure, and keeps the run on the submit push only when there
is none. Each month is built from its own rows, so the choice never reaches across a month boundary.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import rows
from collector import views

# These tests read the module's private helpers.
# pylint: disable=protected-access

MONTH = "2026-11"
COPYBARA = "copybara-service[bot]"
PERSON = "someone"
TPU_LABEL = "linux-x86-ct6e-180-4tpu"
HOSTED_LABEL = "ubuntu-latest"
MERGED = "2026-11-08T01:51:53Z"


class FakeStore:
  """Serves fixed rows for any month, like `store.Store.read_rows`."""

  def __init__(self, records: list[rows.Row]):
    self.records = records

  def months(self, kind: str) -> list[str]:
    del kind
    return [MONTH]

  def read_rows(self, kind: str, months: list[str] | None = None) -> list[rows.Row]:
    del months
    return [record for record in self.records if rows.row_kind(record) == kind]


class MonthStore:
  """Serves each month only its own rows, like the store's month files."""

  def __init__(self, by_month: dict[str, list[rows.Row]]):
    self.by_month = by_month

  def months(self, kind: str) -> list[str]:
    del kind
    return sorted(self.by_month)

  def read_rows(self, kind: str, months: list[str] | None = None) -> list[rows.Row]:
    wanted = sorted(self.by_month) if months is None else months
    return [record for month in wanted for record in self.by_month.get(month, []) if rows.row_kind(record) == kind]


def run_row(
    run_id: int,
    *,
    created: str,
    head: str,
    pr_head: str,
    author: str,
    conclusion: str = "success",
    number: int = 6199,
    merged: str = MERGED,
) -> rows.RunRow:
  """One attempt of a pull request run whose pull request merged."""
  return rows.RunRow(
      run_id=run_id,
      attempt=1,
      event="pull_request",
      status="completed",
      conclusion=conclusion,
      created_at=created,
      head_sha=head,
      superseded=False,
      pr_number=number,
      pr_state="closed",
      pr_merged_at=merged,
      pr_head_sha=pr_head,
      pr_user=author,
  )


def job_row(run_id: int, job_id: int, label: str, *, started: bool = True) -> rows.JobRow:
  """One job; `started=False` makes a job that never held a runner."""
  return rows.JobRow(
      job_id=job_id,
      run_id=run_id,
      attempt=1,
      name="TPU Pretrain Tests (tpu-unit) / Execute Tests (1) / tpu-unit" if label == TPU_LABEL else "Gate",
      status="completed",
      conclusion="success" if started else "cancelled",
      created_at="2026-11-08T01:00:00Z",
      started_at="2026-11-08T01:05:00Z" if started else None,
      completed_at="2026-11-08T01:25:00Z",
      labels=[label],
      steps=[
          {
              "name": "Run Tests",
              "number": 1,
              "status": "completed",
              "conclusion": "success",
              "started_at": "2026-11-08T01:06:00Z",
              "completed_at": "2026-11-08T01:24:00Z",
          }
      ]
      if started
      else [],
  )


def representative(records: list[rows.Row], number: int = 6199) -> int | None:
  return views.load_month(FakeStore(records), MONTH).merged_pr_runs.get(number)


class RepresentativeRunTest(unittest.TestCase):

  def test_copybara_pull_request_is_drawn_from_its_last_run_before_submit(self):
    records = [
        run_row(1, created="2026-11-08T01:13:35Z", head="aaa", pr_head="fff", author=COPYBARA),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=COPYBARA, conclusion="failure"),
        job_row(2, 21, TPU_LABEL),
    ]
    self.assertEqual(representative(records), 1)

  def test_the_newest_run_before_submit_wins(self):
    records = [
        run_row(1, created="2026-11-07T10:00:00Z", head="aaa", pr_head="fff", author=COPYBARA),
        job_row(1, 11, TPU_LABEL),
        run_row(3, created="2026-11-08T01:13:35Z", head="bbb", pr_head="fff", author=COPYBARA, conclusion="failure"),
        job_row(3, 31, TPU_LABEL),
        run_row(2, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(2, 21, TPU_LABEL),
    ]
    self.assertEqual(representative(records), 3)

  def test_a_run_before_submit_without_tests_is_not_used(self):
    records = [
        run_row(1, created="2026-11-08T01:13:35Z", head="aaa", pr_head="fff", author=COPYBARA),
        job_row(1, 11, HOSTED_LABEL),
        job_row(1, 12, TPU_LABEL, started=False),
        run_row(2, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(2, 21, TPU_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_a_superseded_run_before_submit_is_not_used(self):
    cancelled = run_row(
        1, created="2026-11-08T01:13:35Z", head="aaa", pr_head="fff", author=COPYBARA, conclusion="cancelled"
    )
    cancelled.superseded = True
    records = [
        cancelled,
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(2, 21, TPU_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_a_pull_request_opened_on_github_keeps_the_run_on_its_final_head(self):
    records = [
        run_row(1, created="2026-11-08T01:13:35Z", head="aaa", pr_head="fff", author=PERSON),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=PERSON),
        job_row(2, 21, TPU_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_any_pull_request_prefers_a_final_head_run_that_ran_tests(self):
    # Like #6173: the newest run on the final head lists no job at all; an earlier run on the same head tested it.
    records = [
        run_row(1, created="2026-11-07T20:00:00Z", head="fff", pr_head="fff", author=PERSON),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:40:00Z", head="fff", pr_head="fff", author=PERSON, conclusion="failure"),
    ]
    self.assertEqual(representative(records), 1)

  def test_the_newest_final_head_run_that_ran_tests_wins(self):
    records = [
        run_row(1, created="2026-11-07T18:00:00Z", head="fff", pr_head="fff", author=PERSON),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-07T20:00:00Z", head="fff", pr_head="fff", author=PERSON, conclusion="failure"),
        job_row(2, 21, TPU_LABEL),
        run_row(3, created="2026-11-08T01:40:00Z", head="fff", pr_head="fff", author=PERSON),
        job_row(3, 31, HOSTED_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_a_final_head_run_without_a_result_is_not_used(self):
    records = [
        run_row(1, created="2026-11-07T20:00:00Z", head="fff", pr_head="fff", author=PERSON, conclusion="cancelled"),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:40:00Z", head="fff", pr_head="fff", author=PERSON),
        job_row(2, 21, HOSTED_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_a_run_on_an_earlier_head_does_not_replace_a_final_head_run_of_a_github_pull_request(self):
    records = [
        run_row(1, created="2026-11-07T20:00:00Z", head="aaa", pr_head="fff", author=PERSON),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:40:00Z", head="fff", pr_head="fff", author=PERSON),
        job_row(2, 21, HOSTED_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_a_copybara_run_before_submit_that_was_cancelled_is_not_used(self):
    # Cancelled with no superseded mark, as merge-corrected rows were stored before the tick kept the mark.
    records = [
        run_row(1, created="2026-11-08T01:13:35Z", head="aaa", pr_head="fff", author=COPYBARA, conclusion="cancelled"),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(2, 21, TPU_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_a_copybara_run_on_the_final_head_created_before_the_merge_stays(self):
    # The final head was pushed long before the merge, so its run gated the merge: not a submit push run.
    records = [
        run_row(1, created="2026-11-07T10:00:00Z", head="aaa", pr_head="fff", author=COPYBARA),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T00:30:00Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(2, 21, TPU_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_a_copybara_run_created_a_minute_before_the_merge_counts_as_the_submit_push_run(self):
    records = [
        run_row(1, created="2026-11-08T01:13:35Z", head="aaa", pr_head="fff", author=COPYBARA),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:50:53Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(2, 21, TPU_LABEL),
    ]
    self.assertEqual(representative(records), 1)
    records[2] = run_row(2, created="2026-11-08T01:50:52Z", head="fff", pr_head="fff", author=COPYBARA)
    self.assertEqual(representative(records), 2)

  def test_a_tested_final_head_run_before_the_merge_beats_the_copybara_rule(self):
    # Step 1 replaces the submit push run (no tests) with the earlier run on the same head that tested it; that
    # run was created before the merge, so the Copybara rule leaves it.
    records = [
        run_row(1, created="2026-11-07T10:00:00Z", head="aaa", pr_head="fff", author=COPYBARA),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T00:30:00Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(2, 21, TPU_LABEL),
        run_row(3, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(3, 31, HOSTED_LABEL),
    ]
    self.assertEqual(representative(records), 2)

  def test_a_copybara_submit_push_run_without_tests_gives_way_to_the_run_before_it(self):
    records = [
        run_row(1, created="2026-11-08T01:13:35Z", head="aaa", pr_head="fff", author=COPYBARA, conclusion="failure"),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=COPYBARA),
    ]
    self.assertEqual(representative(records), 1)

  def test_a_copybara_pull_request_split_across_months_keeps_the_submit_push_run(self):
    # The runs before the submit push were created in October, the submit push run in November. Each month is
    # built on its own, so November holds only the submit push run and keeps it; October draws the pull request
    # from the run it holds, and the page shows the newest of the two.
    merged = "2026-11-01T00:10:03Z"
    october = [
        run_row(1, created="2026-10-31T23:40:00Z", head="aaa", pr_head="fff", author=COPYBARA, merged=merged),
        job_row(1, 11, TPU_LABEL),
    ]
    november = [
        run_row(2, created="2026-11-01T00:10:01Z", head="fff", pr_head="fff", author=COPYBARA, merged=merged),
        job_row(2, 21, HOSTED_LABEL),
    ]
    split = MonthStore({"2026-10": october, "2026-11": november})
    self.assertEqual(views.load_month(split, "2026-11").merged_pr_runs.get(6199), 2)
    self.assertEqual(views.load_month(split, "2026-10").merged_pr_runs.get(6199), 1)
    # The same runs in one month: the run before the submit push wins.
    self.assertEqual(representative(october + november), 1)

  def test_the_views_and_the_pull_request_file_follow_the_new_choice(self):
    records = [
        run_row(1, created="2026-11-08T01:13:35Z", head="aaa", pr_head="fff", author=COPYBARA),
        job_row(1, 11, TPU_LABEL),
        run_row(2, created="2026-11-08T01:51:52Z", head="fff", pr_head="fff", author=COPYBARA),
        job_row(2, 21, TPU_LABEL),
    ]
    data = views.load_month(FakeStore(records), MONTH)
    self.assertTrue(data.is_representative(1))
    self.assertFalse(data.is_representative(2))
    runs_view = views.build_runs_view(data)
    self.assertEqual(views.from_columnar(runs_view["tables"]["runs"])[0]["run_id"], 1)

    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "6199.json"
      # A file written before this change, for the newer run on the submit push.
      path.write_text(
          json.dumps({"run_id": 2, "created_at": "2026-11-08T01:51:52Z", "updated_at": "2026-11-09T00:00:00Z"}),
          encoding="utf-8",
      )
      candidate: dict[str, Any] = views.build_pr_view(6199, data)
      self.assertEqual(candidate["run_id"], 1)
      # The month holds both runs and chose run 1, so the stored newer run does not block it ...
      self.assertIsNone(views._kept_pr_file(path, candidate, {1, 2}))
      # ... but a newer run from another month still wins.
      self.assertEqual(views._kept_pr_file(path, candidate, {1}), "2026-11-09T00:00:00Z")


if __name__ == "__main__":
  unittest.main()
