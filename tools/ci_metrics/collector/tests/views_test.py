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

"""Offline unit tests for `collector.views`.

Everything here runs from the saved fixtures in `tests/fixtures/` and writes into a temporary
directory that is deleted again. The base test case replaces `socket.socket`, so a test that
reached for GitHub fails instead of hanging, and `views.py` itself never fetches anything.

How the store is assembled
--------------------------
`views.py` reads a row store through a structural `RowStore` protocol, so these tests hand it
a list-backed stub holding rows built by `rows.py` from real payloads. The store is arranged
as two months, because the whole month-open / month-closed contract cannot be tested with one:

  2026-08  merged pull request #5042, on the two-attempt run 32772626658. Both attempts'
           jobs, three rescues and seven failures that stayed failures, a tpu-unit suite that
           lost worker 2 in attempt 1, a cpu-unit suite that published no file at all because
           all four of its workers died at "Install the maxtext wheel", and worker 1's 102
           test rows. Plus run 32785979907, a pull request that never merged.
  2026-09  the scheduled run 33468578834 with its 54 jobs, a whole tpu-unit suite, the
           tpu-pathways-unit suite that publishes no JUnit file at all, and 102 test rows that
           must NOT reach a view. Plus run 33462758754, a merged pull request marked
           superseded, which must reach nothing at all.

Two pairings are the test's rather than GitHub's, and nothing else is:

  * The merged pull request payload (#5042) is attached to run 32772626658, because no saved
    run is both a merged pull request and has a saved job list. Every job, step, conclusion,
    timestamp and JUnit count under it is a real measurement of a real run; only "this run
    belongs to that pull request" is the test's, and no builder compares the two.
  * The tpu-unit JUnit files are the ones saved from run 33468578834. They are filed under
    their own flavor and their own worker numbers.

The numbers asserted below are measured, not invented. The ones quoted most often:

  * tpu-unit on the scheduled run: 1626 s of wall clock (27m06s) against 2519.7 s of JUnit
    time, because the suite runs on two workers at once. 203 collected, 6 skipped, 197
    executed.
  * attempt 1 of run 32772626658: queued 2 s, setup 1169 s, tests 3737 s, tail 17812 s,
    total 22720 s, 35196 s of machine time over 42 jobs.
  * attempt 2 of the same run: 2776 s of machine time over 14 jobs, 28 of them carried over.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/views_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/views_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any
from unittest import mock

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import derive
from collector import junit
from collector import rows
from collector import views

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The two months the fixture store holds, and the runs in them.
AUGUST = "2026-08"
SEPTEMBER = "2026-09"

# Two attempts, three rescues, seven failures that stayed failures. Carries the merged pull
# request in the fixture store.
MERGED_RUN_ID = 32772626658
MERGED_PR = 5042

# One attempt, 54 jobs, scheduled on main. The run every layer-1 fixture came from.
SCHEDULED_RUN_ID = 33468578834

# A pull request that never merged, and a merged one that a newer push superseded.
UNMERGED_RUN_ID = 32785979907
SUPERSEDED_RUN_ID = 33462758754
SUPERSEDED_PR = 5070

# Every row in the fixture store is stamped with this, so a rebuild is deterministic and the
# only thing that ever moves an `updated_at` is a correction the test appends on purpose.
COLLECTED_AT = "2026-09-01T06:00:00Z"

# tpu-unit measured on the scheduled run: the wall clock of its two workers' "Run Tests"
# steps, and the JUnit seconds those two workers piled up between them.
TPU_UNIT_DURATION_SECONDS = 1626.0
TPU_UNIT_MACHINE_SECONDS = 2966.0
TPU_UNIT_COLLECTED = 203
TPU_UNIT_SKIPPED = 6
TPU_UNIT_EXECUTED = 197
TPU_UNIT_JUNIT_SECONDS = 2519.747

# tpu-pathways-unit on the same run: two jobs held a machine for 1057 s of wall clock, and
# neither published a test file, because that workflow runs pytest without --junitxml.
PATHWAYS_DURATION_SECONDS = 1057.0
PATHWAYS_WORKERS_NAMED = 0
PATHWAYS_WORKERS_RAN = 2

# Attempt 1 of the merged pull request's run, from `derive.phase_split`.
ATTEMPT_1_PHASES = {
    "queued_seconds": 2.0,
    "setup_seconds": 1169.0,
    "tests_seconds": 3737.0,
    "tail_seconds": 17812.0,
    "total_seconds": 22720.0,
    "wall_seconds": 22718.0,
}
ATTEMPT_1_MACHINE_SECONDS = 35196.0
ATTEMPT_2_MACHINE_SECONDS = 2776.0

# The tpu-unit suite in attempt 1 published worker 1 only: worker 2 failed at "Run Tests".
PARTIAL_WORKER = 2
PARTIAL_COLLECTED = 102
PARTIAL_SKIPPED = 4
PARTIAL_EXECUTED = 98

# The exact column lists, written out again here so a rename or a reorder in `views.py`
# breaks a test instead of quietly changing what the dashboard is handed.
EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "RUNS_COLUMNS": (
        "pr",
        "title",
        "author",
        "merged_at",
        "head_sha",
        "base_ref",
        "html_url",
        "run_id",
        "run_number",
        "event",
        "status",
        "conclusion",
        "created_at",
        "run_started_at",
        "attempts",
        "attempts_stored",
        "first_created_at",
        "last_completed_at",
        "queued_seconds",
        "setup_seconds",
        "tests_seconds",
        "tail_seconds",
        "total_seconds",
        "wall_seconds",
        "machine_seconds",
        "rerun_machine_seconds",
        "jobs_counted",
        "jobs_with_tests",
        "overlapping_runs",
    ),
    "RUN_JOBS_COLUMNS": (
        "run_id",
        "pr",
        "attempt",
        "job_id",
        "name",
        "lane",
        "flavor",
        "worker",
        "conclusion",
        "runner_label",
        "queued_seconds",
        "setup_seconds",
        "run_seconds",
    ),
    "SUITES_COLUMNS": (
        "run_id",
        "pr",
        "is_representative",
        "event",
        "created_at",
        "merged_at",
        "attempt",
        "suite_id",
        "flavor",
        "nested_in",
        "workers_named",
        "workers_ran",
        "duration_seconds",
        "machine_seconds",
        "collected",
        "skipped",
        "executed",
        "failed",
        "errored",
        "junit_seconds",
        "reason",
        "is_partial",
        "missing_workers",
        "published_workers",
    ),
    "RESCUES_COLUMNS": (
        "run_id",
        "pr",
        "event",
        "created_at",
        "job_name",
        "lane",
        "flavor",
        "worker",
        "rescued",
        "failed_attempt",
        "failed_job_id",
        "failed_conclusion",
        "failed_started_at",
        "failed_completed_at",
        "rescued_attempt",
        "rescued_job_id",
        "final_attempt",
        "final_conclusion",
        "wasted_seconds",
        "html_url",
    ),
    "RESCUE_TESTS_COLUMNS": (
        "run_id",
        "pr",
        "job_name",
        "failed_attempt",
        "suite_id",
        "flavor",
        "worker",
        "classname",
        "name",
        "status",
        "duration",
        "failure_message",
    ),
    "QUEUE_COLUMNS": (
        "run_id",
        "pr",
        "is_representative",
        "event",
        "created_at",
        "pool",
        "lane",
        "jobs_counted",
        "longest_wait_seconds",
        "median_wait_seconds",
        "probe_run_id",
        "probe_created_at",
        "probe_wait_seconds",
    ),
    "WORKFLOWS_COLUMNS": (
        "day",
        "workflow_path",
        "workflow_name",
        "runs",
        "median_wall_seconds",
        "machine_seconds",
        "machine_seconds_tpu",
        "machine_seconds_gpu",
        "machine_seconds_cpu",
        "machine_seconds_build",
        "machine_seconds_hosted",
        "machine_seconds_unknown",
    ),
    "PR_ATTEMPTS_COLUMNS": (
        "attempt",
        "event",
        "status",
        "conclusion",
        "created_at",
        "run_started_at",
        "first_created_at",
        "first_started_at",
        "last_completed_at",
        "queued_seconds",
        "setup_seconds",
        "tests_seconds",
        "tail_seconds",
        "total_seconds",
        "wall_seconds",
        "machine_seconds",
        "jobs_counted",
        "jobs_with_tests",
    ),
    "PR_JOBS_COLUMNS": (
        "attempt",
        "job_id",
        "name",
        "lane",
        "flavor",
        "worker",
        "status",
        "conclusion",
        "runner_label",
        "runner_group_name",
        "created_at",
        "started_at",
        "completed_at",
        "queued_seconds",
        "setup_seconds",
        "run_seconds",
        "carried_over",
    ),
    "PR_STEPS_COLUMNS": (
        "attempt",
        "job_id",
        "number",
        "name",
        "status",
        "conclusion",
        "started_at",
        "completed_at",
    ),
    "PR_SUITES_COLUMNS": (
        "attempt",
        "suite_id",
        "flavor",
        "nested_in",
        "workers_named",
        "workers_ran",
        "duration_seconds",
        "machine_seconds",
        "collected",
        "skipped",
        "executed",
        "failed",
        "errored",
        "junit_seconds",
        "reason",
        "is_partial",
        "missing_workers",
        "published_workers",
    ),
    "PR_TESTS_COLUMNS": (
        "attempt",
        "suite_id",
        "flavor",
        "worker",
        "classname",
        "name",
        "status",
        "duration",
        "failure_message",
        "suite_partial",
    ),
    "PR_ERRORS_COLUMNS": (
        "attempt",
        "job_id",
        "job_name",
        "lane",
        "conclusion",
        "failed_step",
        "failed_step_number",
        "failed_step_started_at",
        "failed_step_completed_at",
        "html_url",
    ),
}

# What each view group has to carry for the dashboard to stop reading its hard-coded
# constants. The key names the mock constant the group replaces; the value lists the columns
# without which that constant cannot be rebuilt in the browser.
DASHBOARD_NEEDS: dict[str, dict[str, tuple[str, ...]]] = {
    "runs.runs": {
        "COMMITS": ("pr", "title", "author", "head_sha", "conclusion", "attempts"),
        "TIMES": ("merged_at", "created_at", "run_started_at", "first_created_at", "last_completed_at"),
        "RUN_IDS": ("run_id", "run_number", "html_url"),
        "TRIGGERS": ("event",),
        "phase split": ("queued_seconds", "setup_seconds", "tests_seconds", "tail_seconds", "total_seconds"),
    },
    "runs.jobs": {
        "JOBS": ("run_id", "pr", "job_id", "name", "flavor", "worker", "conclusion"),
        "MACHINES": ("lane", "runner_label"),
        "per-job wait, setup and run": ("queued_seconds", "setup_seconds", "run_seconds"),
    },
    "suites.suites": {
        "TH_CATEGORIES.data": ("suite_id", "flavor", "executed", "duration_seconds", "reason", "is_partial"),
        "TH_COMMITS": ("run_id", "pr", "created_at", "merged_at", "event"),
        "THI_WORKERS": ("workers_named", "workers_ran"),
    },
    "flaky.rescues": {
        "RESCUES": ("job_name", "lane", "flavor", "worker", "rescued", "failed_attempt", "wasted_seconds"),
    },
    "flaky.rescue_tests": {
        "FLAKY_TESTS": ("job_name", "classname", "name", "status", "failure_message"),
    },
    "queue.queue": {
        "the pool series": ("pool", "lane", "longest_wait_seconds", "median_wait_seconds"),
        "PROBE_Q": ("probe_run_id", "probe_created_at", "probe_wait_seconds"),
    },
    "workflows.workflows": {
        "the Workflow durations card": ("day", "workflow_path", "workflow_name", "runs", "median_wall_seconds"),
        "machine minutes per lane": (
            "machine_seconds_tpu",
            "machine_seconds_gpu",
            "machine_seconds_cpu",
            "machine_seconds_build",
        ),
    },
}

# The same, for the file one click fetches.
PR_VIEW_NEEDS: dict[str, dict[str, tuple[str, ...]]] = {
    "attempts": {"ATTEMPT_INFO": ("attempt", "conclusion", "queued_seconds", "setup_seconds", "tests_seconds")},
    "jobs": {"the job table": ("attempt", "job_id", "name", "lane", "queued_seconds", "run_seconds", "carried_over")},
    "steps": {"STEPS": ("attempt", "job_id", "number", "name", "started_at", "completed_at")},
    "suites": {"TEST_COUNTS": ("suite_id", "collected", "skipped", "executed", "reason", "is_partial")},
    "tests": {"TESTS": ("suite_id", "worker", "classname", "name", "status", "duration", "failure_message")},
    "errors": {"the error rows": ("attempt", "job_name", "failed_step", "html_url")},
}


def read_fixture(name: str) -> bytes:
  """Returns the raw bytes of one saved fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The file contents.
  """
  return (FIXTURES / name).read_bytes()


def load_json(name: str) -> Any:
  """Loads one saved JSON fixture, fresh each call so no test can mutate another's copy.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The parsed payload.
  """
  return json.loads(read_fixture(name))


def load_jobs(name: str) -> list[dict[str, Any]]:
  """Loads the job list out of a saved jobs-endpoint fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The jobs, in the order GitHub listed them.
  """
  payload = load_json(name)
  return payload["jobs"] if isinstance(payload, dict) else payload


def named(jobs: list[dict[str, Any]], name: str) -> dict[str, Any]:
  """Returns the one job of a list that carries a given name.

  Args:
    jobs: The jobs of one attempt.
    name: The full job name.

  Returns:
    The job payload.

  Raises:
    KeyError: No job of that name is in the list.
  """
  for job in jobs:
    if job["name"] == name:
      return job
  raise KeyError(name)


def parse_suite(name: str) -> junit.SuiteResult:
  """Parses one saved JUnit file.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The parsed suite result.
  """
  return junit.parse_junit_xml(read_fixture(name), file_name=name)


class FakeState:
  """The part of state.json `views.py` reads: how many attempts are still in flight."""

  def __init__(self, pending_count: int = 0) -> None:
    """Stores the count.

    Args:
      pending_count: How many run attempts have been seen but not collected.
    """
    self._pending_count = pending_count

  @property
  def pending_count(self) -> int:
    """Returns how many run attempts have been seen but not collected."""
    return self._pending_count


class FakeStore:
  """A list-backed `views.RowStore`, filed by month exactly as `store.Store` files by file.

  The real store puts every row of a run under the run's `created_at` month; this keeps the
  same arrangement without importing the writer, which is the point of `views.py` declaring
  its store structurally.
  """

  def __init__(self, pending_count: int = 0) -> None:
    """Builds an empty store.

    Args:
      pending_count: What `load_state().pending_count` will answer.
    """
    self.entries: list[tuple[str, str, rows.Row]] = []
    self.reads: list[tuple[str, tuple[str, ...] | None]] = []
    self._pending_count = pending_count

  def append(self, month: str, row: rows.Row) -> rows.Row:
    """Files one row under a month, at the end, the way an append-only file grows.

    Args:
      month: "YYYY-MM".
      row: The row.

    Returns:
      The row, so a caller can keep a reference to it.
    """
    self.entries.append((month, rows.row_kind(row), row))
    return row

  def extend(self, month: str, records: list[rows.Row]) -> None:
    """Files a list of rows under a month.

    Args:
      month: "YYYY-MM".
      records: The rows.
    """
    for record in records:
      self.append(month, record)

  def months(self, kind: str) -> list[str]:
    """Returns the months this kind has rows for, ascending.

    Args:
      kind: One of the `rows.KIND_*` values.

    Returns:
      The month keys.
    """
    return sorted({month for month, row_kind, _ in self.entries if row_kind == kind})

  def read_rows(self, kind: str, months: list[str] | None = None) -> list[rows.Row]:
    """Returns one kind's rows for the given months, in the order they were appended.

    Args:
      kind: One of the `rows.KIND_*` values.
      months: The months to read, or None for all of them.

    Returns:
      The rows.
    """
    self.reads.append((kind, tuple(months) if months is not None else None))
    wanted = set(months) if months is not None else None
    return [row for month, row_kind, row in self.entries if row_kind == kind and (wanted is None or month in wanted)]

  def load_state(self) -> FakeState:
    """Returns the store's state.

    Returns:
      A `FakeState` carrying the pending count this store was built with.
    """
    return FakeState(self._pending_count)


def build_fixture_store(pending_count: int = 2) -> FakeStore:
  """Assembles the two-month store every view test is built on.

  See the module docstring for what is in it and which two pairings are the test's.

  Args:
    pending_count: What `load_state().pending_count` will answer.

  Returns:
    The store, ready to hand to `views.build_views`.
  """
  store = FakeStore(pending_count=pending_count)
  merged_pr = load_json("fork-pr-5042-pulls-by-head.json")[0]
  run = load_json(f"rerun-{MERGED_RUN_ID}-run.json")
  attempt_jobs = {n: load_jobs(f"rerun-{MERGED_RUN_ID}-attempt{n}-jobs.json") for n in (1, 2)}

  # One run payload per attempt, which is what `runs.list_attempts` hands back.
  per_attempt = {n: dict(run, run_attempt=n) for n in (1, 2)}
  for attempt in (1, 2):
    store.append(AUGUST, rows.run_row(per_attempt[attempt], merged_pr, collected_at=COLLECTED_AT))
    for job in attempt_jobs[attempt]:
      store.append(AUGUST, rows.job_row(per_attempt[attempt], job, collected_at=COLLECTED_AT))

  # Parsed once and shared: nothing in `views.py` or `rows.py` mutates a `SuiteResult`, and
  # `junit.read_run_tests`, the one function that does, is never called here.
  worker_1 = parse_suite("tpu-unit-1.xml")
  worker_2 = parse_suite("tpu-unit-2.xml")
  partial = junit.SuiteEntry(
      suite_id="tpu-unit",
      result=worker_1,
      per_worker={1: worker_1},
      missing_workers={PARTIAL_WORKER: junit.REASON_NO_FILE},
  )
  no_file = junit.SuiteEntry(suite_id="cpu-unit", result=None, reason=junit.REASON_NO_FILE)
  store.append(AUGUST, rows.suite_row(per_attempt[1], partial, collected_at=COLLECTED_AT))
  store.append(AUGUST, rows.suite_row(per_attempt[1], no_file, collected_at=COLLECTED_AT))
  store.extend(
      AUGUST,
      rows.test_rows(per_attempt[1], "tpu-unit", 1, worker_1, suite_partial=True, collected_at=COLLECTED_AT),
  )

  whole = junit.SuiteEntry(
      suite_id="tpu-unit",
      result=junit.merge_suite_results([worker_1, worker_2]),
      per_worker={1: worker_1, 2: worker_2},
  )
  store.append(AUGUST, rows.suite_row(per_attempt[2], whole, collected_at=COLLECTED_AT))
  store.extend(AUGUST, rows.rescue_rows(per_attempt[2], attempt_jobs, collected_at=COLLECTED_AT))
  store.extend(AUGUST, rows.failed_never_rescued_rows(per_attempt[2], attempt_jobs, collected_at=COLLECTED_AT))

  # A pull request that never merged: its rows are stored, and no chart may draw it.
  store.append(
      AUGUST, rows.run_row(load_json(f"cancelled-job-{UNMERGED_RUN_ID}-run.json"), None, collected_at=COLLECTED_AT)
  )

  # September: the scheduled run, whole and real.
  scheduled = load_json("run.json")
  store.append(SEPTEMBER, rows.run_row(scheduled, None, collected_at=COLLECTED_AT))
  for job in load_jobs("jobs.json"):
    store.append(SEPTEMBER, rows.job_row(scheduled, job, collected_at=COLLECTED_AT))
  scheduled_tpu = junit.SuiteEntry(
      suite_id="tpu-unit",
      result=junit.merge_suite_results([worker_1, worker_2]),
      per_worker={1: worker_1, 2: worker_2},
  )
  pathways = junit.SuiteEntry(suite_id="tpu-pathways-unit", result=None, reason=junit.REASON_NO_FILE)
  store.append(SEPTEMBER, rows.suite_row(scheduled, scheduled_tpu, collected_at=COLLECTED_AT))
  store.append(SEPTEMBER, rows.suite_row(scheduled, pathways, collected_at=COLLECTED_AT))
  store.extend(SEPTEMBER, rows.test_rows(scheduled, "tpu-unit", 1, worker_1, collected_at=COLLECTED_AT))

  # A merged pull request whose run a newer push superseded. `superseded` is the flag
  # `runs.mark_superseded` adds to its copy of the payload.
  superseded = dict(load_json(f"superseded-cancelled-run-{SUPERSEDED_RUN_ID}.json"), superseded=True)
  store.append(
      SEPTEMBER,
      rows.run_row(superseded, load_json("merged-pr-5070-pulls-by-head.json")[0], collected_at=COLLECTED_AT),
  )
  return store


class OfflineTestCase(unittest.TestCase):
  """Base class that turns any attempt to open a socket into a test failure."""

  def setUp(self) -> None:
    """Blocks the network for the duration of one test."""
    super().setUp()

    def deny(*args: object, **kwargs: object) -> None:
      """Refuses every socket the test might try to open.

      Args:
        *args: Ignored.
        **kwargs: Ignored.

      Raises:
        AssertionError: Always.
      """
      del args, kwargs
      raise AssertionError("these tests must run offline, but something tried to open a socket")

    for attribute in ("socket", "create_connection"):
      patch = mock.patch.object(socket, attribute, deny)
      patch.start()
      self.addCleanup(patch.stop)

  def temp_dir(self) -> Path:
    """Returns a temporary directory that is deleted when the test ends.

    Returns:
      The directory path.
    """
    holder = Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, holder, ignore_errors=True)
    return holder


class ColumnarFormatTest(OfflineTestCase):
  """Covers `to_columnar` and `from_columnar`: the format the browser rehydrates."""

  COLUMNS = ("pr", "merged_at", "queued_seconds", "missing_workers", "is_partial")

  def records(self) -> list[dict[str, Any]]:
    """Returns records covering every JSON type a view row can hold, None included.

    Returns:
      Four records: a full one, one whose numbers are all missing, one with an empty list,
      and one carrying a unicode string.
    """
    return [
        {
            "pr": 5042,
            "merged_at": "2026-08-28T22:21:48Z",
            "queued_seconds": 2.0,
            "missing_workers": [{"worker": 2, "reason": "no_file_published"}],
            "is_partial": True,
        },
        {"pr": 5070, "merged_at": None, "queued_seconds": None, "missing_workers": None, "is_partial": False},
        {
            "pr": 4980,
            "merged_at": "2026-08-24T20:13:09Z",
            "queued_seconds": 0.0,
            "missing_workers": [],
            "is_partial": False,
        },
        {"pr": 1, "merged_at": "合併", "queued_seconds": 1e-06, "missing_workers": [], "is_partial": None},
    ]

  def test_round_trip_is_exact(self) -> None:
    """`from_columnar(to_columnar(x)) == x`, field for field, in order."""
    records = self.records()
    table = views.to_columnar(records, self.COLUMNS)

    self.assertEqual(table["columns"], list(self.COLUMNS))
    self.assertEqual(views.from_columnar(table), records)
    self.assertEqual([list(record) for record in views.from_columnar(table)], [list(self.COLUMNS)] * len(records))

  def test_round_trip_survives_a_real_json_trip(self) -> None:
    """The browser reads the file, not the object, so the trip has to go through JSON."""
    records = self.records()
    payload = json.dumps(views.to_columnar(records, self.COLUMNS), separators=(",", ":"), ensure_ascii=False)

    self.assertEqual(views.from_columnar(json.loads(payload)), records)

  def test_none_stays_none(self) -> None:
    """A missing number is null on the way out and None on the way back, never 0 or ""."""
    table = views.to_columnar(self.records(), self.COLUMNS)
    payload = json.dumps(table)

    self.assertIn("null", payload)
    self.assertIsNone(table["rows"][1][1])
    self.assertIsNone(table["rows"][1][2])
    back = views.from_columnar(json.loads(payload))
    self.assertIsNone(back[1]["queued_seconds"])
    self.assertIsNone(back[1]["missing_workers"])
    self.assertNotEqual(back[1]["queued_seconds"], 0)

  def test_an_empty_list_of_records_is_still_a_table(self) -> None:
    """A month with nothing in it writes an empty table, not a missing one."""
    table = views.to_columnar([], self.COLUMNS)

    self.assertEqual(table, {"columns": list(self.COLUMNS), "rows": []})
    self.assertEqual(views.from_columnar(table), [])
    self.assertEqual(views.table_row_count(table), 0)

  def test_an_empty_list_field_is_not_confused_with_a_missing_one(self) -> None:
    """`[]` means the suite lost no worker; `None` means nobody has looked."""
    back = views.from_columnar(views.to_columnar(self.records(), self.COLUMNS))

    self.assertEqual(back[2]["missing_workers"], [])
    self.assertIsNone(back[1]["missing_workers"])

  def test_columnar_is_smaller_than_the_object_form(self) -> None:
    """The reason the format exists: names once, not once per row."""
    record = self.records()[0]
    many = [dict(record, pr=5000 + index) for index in range(600)]

    columnar = json.dumps(views.to_columnar(many, self.COLUMNS), separators=(",", ":"))
    objects = json.dumps(many, separators=(",", ":"))

    self.assertLess(len(columnar), len(objects))
    self.assertLess(len(columnar) / len(objects), 0.7)

  def test_to_columnar_refuses_a_record_that_does_not_match(self) -> None:
    """A dropped field would reach the dashboard as a gap, so it is refused instead."""
    with self.assertRaises(views.ViewError) as missing:
      views.to_columnar([{"pr": 1}], self.COLUMNS)
    self.assertIn("missing", str(missing.exception))

    extra = dict(self.records()[0], surprise=1)
    with self.assertRaises(views.ViewError) as unexpected:
      views.to_columnar([extra], self.COLUMNS)
    self.assertIn("unexpected", str(unexpected.exception))
    self.assertIn("surprise", str(unexpected.exception))

  def test_to_columnar_refuses_a_repeated_column(self) -> None:
    """Two columns of the same name would make the round trip lossy."""
    with self.assertRaises(views.ViewError):
      views.to_columnar([], ("pr", "pr"))

  def test_from_columnar_refuses_what_is_not_a_table(self) -> None:
    """A half-written or renamed file fails loudly rather than rehydrating nonsense."""
    with self.assertRaises(views.ViewError):
      views.from_columnar({"rows": []})
    with self.assertRaises(views.ViewError):
      views.from_columnar({"columns": "pr", "rows": []})
    with self.assertRaises(views.ViewError):
      views.from_columnar({"columns": ["pr", "title"], "rows": [[1]]})


class ColumnListTest(OfflineTestCase):
  """Pins every column list, so a rename breaks a test instead of the dashboard."""

  def test_every_column_list_is_exactly_as_written(self) -> None:
    """The lists are the contract between this module and the browser."""
    for name, expected in EXPECTED_COLUMNS.items():
      with self.subTest(columns=name):
        self.assertEqual(getattr(views, name), expected)

  def test_no_column_list_repeats_a_name(self) -> None:
    """`to_columnar` refuses a repeated column, so a list that had one could never be used."""
    for name, expected in EXPECTED_COLUMNS.items():
      with self.subTest(columns=name):
        self.assertEqual(len(set(expected)), len(expected))

  def test_the_groups_are_the_five_the_dashboard_fetches(self) -> None:
    """Five month-split groups, in the order meta.json lists them."""
    self.assertEqual(views.VIEW_GROUPS, ("runs", "suites", "flaky", "queue", "workflows"))
    self.assertEqual(views.VIEW_WINDOW_DAYS, 90)
    self.assertEqual(views.SCHEMA_VERSION, 1)


class MonthWindowTest(OfflineTestCase):
  """Covers which months a tick rebuilds and which it leaves alone."""

  def test_month_key_reads_a_run_in_utc(self) -> None:
    """A run is filed under its `created_at`, in UTC, never under the local day."""
    run = rows.run_row(load_json("run.json"))

    self.assertEqual(views.month_of_run(run), SEPTEMBER)
    self.assertEqual(views.month_key("2026-08-31T23:59:59Z"), AUGUST)
    self.assertEqual(views.month_key(date(2026, 9, 1)), SEPTEMBER)
    self.assertIsNone(views.month_key("not a timestamp"))

  def test_only_the_current_month_is_open_in_mid_month(self) -> None:
    """On the 15th, last month cannot gain a row, so its file is never rewritten."""
    self.assertEqual(views.months_to_rebuild([AUGUST, SEPTEMBER], date(2026, 9, 15)), [SEPTEMBER])

  def test_the_month_before_stays_open_for_the_first_days(self) -> None:
    """A tick just after midnight on the 1st still collects last month's runs."""
    self.assertEqual(views.months_to_rebuild([AUGUST, SEPTEMBER], date(2026, 9, 1)), [AUGUST, SEPTEMBER])
    self.assertEqual(views.months_to_rebuild([AUGUST, SEPTEMBER], date(2026, 9, 3)), [SEPTEMBER])

  def test_the_window_covers_ninety_days(self) -> None:
    """Four months at most, which is what meta.json advertises."""
    self.assertEqual(views.months_in_window(date(2026, 9, 15)), ["2026-06", "2026-07", "2026-08", "2026-09"])


class ViewsTestCase(OfflineTestCase):
  """Base for the tests that build the fixture store's views into a temporary directory."""

  TODAY = date(2026, 9, 1)
  GENERATED_AT = "2026-09-01T07:00:00Z"

  def setUp(self) -> None:
    """Builds the fixture store and runs one tick, with both months open."""
    super().setUp()
    self.store = build_fixture_store()
    self.out = self.temp_dir()
    self.summary = self.build(self.TODAY, self.GENERATED_AT)

  def build(self, today: date, generated_at: str, **kwargs: Any) -> dict[str, Any]:
    """Runs one tick over the fixture store.

    Args:
      today: The day the tick is running.
      generated_at: The build time meta.json will carry.
      **kwargs: Passed through to `views.build_views`.

    Returns:
      The tick's summary.
    """
    return views.build_views(self.store, self.out, today, generated_at=generated_at, **kwargs)

  def view(self, group: str, month: str) -> dict[str, Any]:
    """Reads one month file back off disk.

    Args:
      group: The view group.
      month: "YYYY-MM".

    Returns:
      The parsed file.
    """
    return json.loads(views.view_path(self.out, group, month).read_text(encoding="utf-8"))

  def table(self, group: str, month: str, name: str) -> list[dict[str, Any]]:
    """Reads one table of one month file back as records.

    Args:
      group: The view group.
      month: "YYYY-MM".
      name: The table inside the file.

    Returns:
      The rehydrated records.
    """
    return views.from_columnar(self.view(group, month)["tables"][name])

  def meta(self) -> dict[str, Any]:
    """Reads meta.json back off disk.

    Returns:
      The parsed file.
    """
    return json.loads(views.meta_path(self.out).read_text(encoding="utf-8"))

  def pr_view(self, number: int) -> dict[str, Any]:
    """Reads one pull request file back off disk.

    Args:
      number: The merged pull request.

    Returns:
      The parsed file.
    """
    return json.loads(views.pr_view_path(self.out, number).read_text(encoding="utf-8"))

  def merged_jobs(self, attempt: int) -> list[dict[str, Any]]:
    """Returns the saved jobs of one attempt of the merged pull request's run.

    Args:
      attempt: 1 or 2.

    Returns:
      The job payloads, as GitHub listed them.
    """
    return load_jobs(f"rerun-{MERGED_RUN_ID}-attempt{attempt}-jobs.json")


class RunsViewTest(ViewsTestCase):
  """Covers the runs view: one row per merged pull request, plus its first attempt's jobs."""

  def test_the_month_holds_the_merged_pull_request_only(self) -> None:
    """A run that did not merge, and one that was superseded, are not drawn anywhere."""
    august = self.table("runs", AUGUST, "runs")

    self.assertEqual([record["pr"] for record in august], [MERGED_PR])
    self.assertEqual(august[0]["run_id"], MERGED_RUN_ID)
    self.assertEqual(august[0]["title"], "Fix dropped routing-weight gradient in ring_ragged_unsort")
    self.assertEqual(august[0]["merged_at"], "2026-08-28T22:21:48Z")
    self.assertEqual(august[0]["event"], "pull_request")
    self.assertEqual(self.table("runs", SEPTEMBER, "runs"), [])

  def test_the_run_row_carries_both_attempt_counts(self) -> None:
    """Trend charts read attempt 1; the re-run's cost is a separate number, not hidden."""
    row = self.table("runs", AUGUST, "runs")[0]

    self.assertEqual(row["attempts"], 2)
    self.assertEqual(row["attempts_stored"], 2)
    self.assertEqual(row["machine_seconds"], ATTEMPT_1_MACHINE_SECONDS)
    self.assertEqual(row["rerun_machine_seconds"], ATTEMPT_2_MACHINE_SECONDS)

  def test_the_job_table_leaves_out_the_jobs_no_chart_reads(self) -> None:
    """Hosted gates and jobs that asked for no runner stay in the pull request file."""
    jobs = self.table("runs", AUGUST, "jobs")
    lanes = {record["lane"] for record in jobs}

    self.assertEqual(len(jobs), 21)
    self.assertNotIn(derive.LANE_HOSTED, lanes)
    self.assertNotIn(derive.LANE_NO_RUNNER, lanes)
    self.assertEqual({record["attempt"] for record in jobs}, {1})
    self.assertEqual({record["pr"] for record in jobs}, {MERGED_PR})

  def test_a_test_job_is_named_by_flavor_and_worker(self) -> None:
    """A job named "Execute Tests (1) / tpu-unit" reaches the view as flavor tpu-unit, worker 1."""
    jobs = self.table("runs", AUGUST, "jobs")
    worker_1 = [job for job in jobs if job["name"].endswith("Execute Tests (1) / tpu-unit")][0]

    self.assertEqual(worker_1["flavor"], "tpu-unit")
    self.assertEqual(worker_1["worker"], 1)
    self.assertEqual(worker_1["lane"], derive.LANE_TPU)
    self.assertEqual(worker_1["runner_label"], "linux-x86-ct6e-180-4tpu")

  def test_a_job_that_is_not_a_test_job_carries_no_flavor(self) -> None:
    """`derive.flavor_of` answers "Build Wheel" for the wheel job; the view must not."""
    jobs = self.table("runs", AUGUST, "jobs")
    wheel = [job for job in jobs if job["name"].endswith("Build Wheel")][0]

    self.assertIsNone(wheel["flavor"])
    self.assertIsNone(wheel["worker"])
    self.assertEqual(wheel["lane"], derive.LANE_BUILD)

  def test_overlapping_runs_counts_the_other_run_in_the_month(self) -> None:
    """The count the dashboard prints as "N pipeline runs overlapped it"."""
    self.assertEqual(self.table("runs", AUGUST, "runs")[0]["overlapping_runs"], 1)


class SuitesViewTest(ViewsTestCase):
  """Covers the suites view: the T, D and W the test-suite health chart is drawn from."""

  def suite(self, month: str, run_id: int, attempt: int, suite_id: str) -> dict[str, Any]:
    """Returns one suite row of one attempt.

    Args:
      month: "YYYY-MM".
      run_id: The run.
      attempt: The attempt.
      suite_id: The suite.

    Returns:
      The record.
    """
    found = [
        record
        for record in self.table("suites", month, "suites")
        if record["run_id"] == run_id and record["attempt"] == attempt and record["suite_id"] == suite_id
    ]
    self.assertEqual(len(found), 1, f"expected one {suite_id} row for run {run_id} attempt {attempt}")
    return found[0]

  def test_a_suite_with_no_test_file_is_null_with_its_reason(self) -> None:
    """The Pathways jobs publish nothing, so every count is null and the reason says why."""
    row = self.suite(SEPTEMBER, SCHEDULED_RUN_ID, 1, "tpu-pathways-unit")

    for field in ("collected", "skipped", "executed", "failed", "errored", "junit_seconds"):
      with self.subTest(field=field):
        self.assertIsNone(row[field], f"{field} must be null, never 0")
    self.assertEqual(row["reason"], junit.REASON_NO_FILE)
    self.assertFalse(row["is_partial"])
    self.assertEqual(row["published_workers"], [])

  def test_a_suite_with_no_test_file_still_carries_the_time_it_took(self) -> None:
    """Two Pathways jobs really held machines for 1057 s; only the counts are unknown."""
    row = self.suite(SEPTEMBER, SCHEDULED_RUN_ID, 1, "tpu-pathways-unit")

    self.assertEqual(row["duration_seconds"], PATHWAYS_DURATION_SECONDS)
    self.assertEqual(row["workers_named"], PATHWAYS_WORKERS_NAMED)
    self.assertEqual(row["workers_ran"], PATHWAYS_WORKERS_RAN)

  def test_a_whole_suite_carries_the_counts_junit_reported(self) -> None:
    """203 collected, 6 skipped, 197 executed - the two workers of the scheduled run."""
    row = self.suite(SEPTEMBER, SCHEDULED_RUN_ID, 1, "tpu-unit")

    self.assertEqual(row["collected"], TPU_UNIT_COLLECTED)
    self.assertEqual(row["skipped"], TPU_UNIT_SKIPPED)
    self.assertEqual(row["executed"], TPU_UNIT_EXECUTED)
    self.assertAlmostEqual(row["junit_seconds"], TPU_UNIT_JUNIT_SECONDS, places=3)
    self.assertEqual(row["published_workers"], [1, 2])
    self.assertIsNone(row["reason"])
    self.assertFalse(row["is_partial"])

  def test_a_partial_suite_carries_its_marker_and_the_worker_that_is_missing(self) -> None:
    """Worker 2 died at "Run Tests", so the total is incomplete, not lower."""
    row = self.suite(AUGUST, MERGED_RUN_ID, 1, "tpu-unit")

    self.assertTrue(row["is_partial"])
    self.assertEqual(row["missing_workers"], [{"worker": PARTIAL_WORKER, "reason": junit.REASON_NO_FILE}])
    self.assertEqual(row["published_workers"], [1])
    self.assertEqual(row["collected"], PARTIAL_COLLECTED)
    self.assertEqual(row["skipped"], PARTIAL_SKIPPED)
    self.assertEqual(row["executed"], PARTIAL_EXECUTED)
    self.assertEqual(row["workers_named"], 2)

  def test_a_suite_that_failed_before_pytest_is_null_but_not_partial(self) -> None:
    """All four cpu-unit workers died installing the wheel: no file, so no partial flag."""
    row = self.suite(AUGUST, MERGED_RUN_ID, 1, "cpu-unit")

    self.assertIsNone(row["collected"])
    self.assertIsNone(row["executed"])
    self.assertEqual(row["reason"], junit.REASON_NO_FILE)
    self.assertFalse(row["is_partial"])
    self.assertEqual(row["workers_named"], 4)
    self.assertEqual(row["workers_ran"], 4)

  def test_a_scheduled_run_keeps_its_event_and_has_no_pull_request(self) -> None:
    """The browser draws the scheduled series on its own, never mixed with merges."""
    row = self.suite(SEPTEMBER, SCHEDULED_RUN_ID, 1, "tpu-unit")

    self.assertEqual(row["event"], "schedule")
    self.assertIsNone(row["pr"])
    self.assertIsNone(row["merged_at"])
    self.assertEqual(row["created_at"], "2026-09-01T04:06:01Z")

  def test_every_attempt_of_a_run_reaches_the_view(self) -> None:
    """Attempt 2 published both workers, so the same suite has two rows with two states."""
    attempts = sorted(
        record["attempt"] for record in self.table("suites", AUGUST, "suites") if record["suite_id"] == "tpu-unit"
    )

    self.assertEqual(attempts, [1, 2])
    self.assertFalse(self.suite(AUGUST, MERGED_RUN_ID, 2, "tpu-unit")["is_partial"])


class FlakyViewTest(ViewsTestCase):
  """Covers the flaky view: rescues, failures that were never rescued, and their tests."""

  def rescues(self) -> list[dict[str, Any]]:
    """Returns August's rescue rows.

    Returns:
      The records.
    """
    return self.table("flaky", AUGUST, "rescues")

  def test_both_halves_of_the_card_are_carried(self) -> None:
    """Three jobs failed and passed on the re-run; seven ended the run in failure."""
    rescued = [record for record in self.rescues() if record["rescued"]]
    failed = [record for record in self.rescues() if not record["rescued"]]

    self.assertEqual(len(rescued), 3)
    self.assertEqual(len(failed), 7)
    self.assertEqual({record["pr"] for record in self.rescues()}, {MERGED_PR})

  def test_a_rescue_names_the_flavor_worker_and_lane(self) -> None:
    """The card groups by suite and worker, so the job name has to be parsed here."""
    name = "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit"
    row = [record for record in self.rescues() if record["job_name"] == name][0]

    self.assertTrue(row["rescued"])
    self.assertEqual(row["flavor"], "cpu-unit")
    self.assertEqual(row["worker"], 1)
    self.assertEqual(row["lane"], derive.LANE_CPU)
    self.assertEqual(row["failed_attempt"], 1)
    self.assertEqual(row["rescued_attempt"], 2)
    self.assertEqual(row["final_conclusion"], "success")

  def test_wasted_seconds_is_the_failed_jobs_own_run_time(self) -> None:
    """Read by `derive.run_seconds` from the stored job row, not by subtracting strings."""
    name = "CPU Posttrain Tests (cpu-post-training-unit) / Execute Tests (3) / cpu-post-training-unit"
    row = [record for record in self.rescues() if record["job_name"] == name][0]
    job = named(self.merged_jobs(1), name)

    self.assertEqual(row["wasted_seconds"], derive.run_seconds(job))
    self.assertEqual(row["wasted_seconds"], 146.0)

  def test_a_job_that_is_not_a_test_job_has_no_flavor(self) -> None:
    """The gate job "All Required Tests Passed" runs no suite, so naming a flavor would be a guess."""
    row = [record for record in self.rescues() if record["job_name"] == "All Required Tests Passed"][0]

    self.assertFalse(row["rescued"])
    self.assertIsNone(row["flavor"])
    self.assertIsNone(row["worker"])
    self.assertEqual(row["lane"], derive.LANE_HOSTED)

  def test_no_test_is_named_when_no_test_file_was_published(self) -> None:
    """Every failure in this run died before pytest, so no test can be named.

    The three rescued jobs are cpu-unit and cpu-post-training-unit workers that failed at
    "Install the maxtext wheel", and the cpu-unit suite row says `no_file_published`. A
    Pathways job publishes no file at all and cannot even be tied to a suite. Naming a test
    for any of them would be an invention, so the table is empty.
    """
    self.assertEqual(self.table("flaky", AUGUST, "rescue_tests"), [])
    self.assertIsNone(self.table("suites", AUGUST, "suites")[0]["collected"])

  def test_a_month_without_a_rescue_still_writes_the_tables(self) -> None:
    """An empty table is a fact; a missing one would be a blank card."""
    view = self.view("flaky", SEPTEMBER)

    self.assertEqual(view["tables"]["rescues"], {"columns": list(views.RESCUES_COLUMNS), "rows": []})
    self.assertEqual(view["tables"]["rescue_tests"]["columns"], list(views.RESCUE_TESTS_COLUMNS))


class QueueViewTest(ViewsTestCase):
  """Covers the queue view: how long each runner pool made each run wait."""

  def test_one_row_per_pool_the_run_queued_against(self) -> None:
    """The pool is the runs-on label, because that is what a job queues against."""
    pools = [record["pool"] for record in self.table("queue", AUGUST, "queue")]

    self.assertEqual(
        pools,
        [
            "linux-x86-a2-48-a100-4gpu",
            "linux-x86-ct6e-180-4tpu",
            "linux-x86-n2-16-buildkit",
            "linux-x86-n2-32",
            "ubuntu-latest",
        ],
    )

  def test_the_wait_numbers_come_from_the_jobs_of_that_pool(self) -> None:
    """Longest and median over `derive.queue_seconds` of the first attempt's jobs."""
    row = [record for record in self.table("queue", AUGUST, "queue") if record["pool"] == "linux-x86-n2-32"][0]
    waits = [
        derive.queue_seconds(job)
        for job in self.merged_jobs(1)
        if views.pool_label(job) == "linux-x86-n2-32" and derive.queue_seconds(job) is not None
    ]

    self.assertEqual(row["jobs_counted"], len(waits))
    self.assertEqual(row["longest_wait_seconds"], max(waits))
    self.assertEqual(row["longest_wait_seconds"], 1740.0)

  def test_there_is_no_probe_when_no_scheduled_run_is_near(self) -> None:
    """The nearest scheduled run is in another month, so the honest answer is null."""
    row = [record for record in self.table("queue", AUGUST, "queue") if record["pool"] == "linux-x86-n2-32"][0]

    self.assertIsNone(row["probe_run_id"])
    self.assertIsNone(row["probe_created_at"])
    self.assertIsNone(row["probe_wait_seconds"])

  def test_a_scheduled_run_is_a_pool_row_of_its_own(self) -> None:
    """The probe series is the same measurement, taken on a timer."""
    september = self.table("queue", SEPTEMBER, "queue")

    self.assertEqual(len(september), 7)
    self.assertEqual({record["event"] for record in september}, {"schedule"})
    self.assertEqual({record["pr"] for record in september}, {None})


class WorkflowsViewTest(ViewsTestCase):
  """Covers the workflows view: one row per workflow per day."""

  def test_a_day_counts_every_run_that_was_not_superseded(self) -> None:
    """Even the pull request that never merged: it really did hold runners that day."""
    august = self.table("workflows", AUGUST, "workflows")

    self.assertEqual(len(august), 1)
    self.assertEqual(august[0]["day"], "2026-08-24")
    self.assertEqual(august[0]["workflow_path"], ".github/workflows/ci_pipeline.yml")
    self.assertEqual(august[0]["workflow_name"], "MaxText Package Tests")
    self.assertEqual(august[0]["runs"], 2)

  def test_the_machine_minutes_are_split_by_lane(self) -> None:
    """The lanes have to add up to the total, or the card's stack lies."""
    row = self.table("workflows", SEPTEMBER, "workflows")[0]
    lanes = [
        row["machine_seconds_tpu"],
        row["machine_seconds_gpu"],
        row["machine_seconds_cpu"],
        row["machine_seconds_build"],
        row["machine_seconds_hosted"],
        row["machine_seconds_unknown"],
    ]

    self.assertAlmostEqual(sum(lanes), row["machine_seconds"], places=6)
    self.assertEqual(row["machine_seconds"], derive.machine_seconds(load_jobs("jobs.json")))

  def test_the_clock_time_is_the_median_of_the_first_attempts(self) -> None:
    """A re-run must not make a workflow look slower than it was."""
    row = self.table("workflows", SEPTEMBER, "workflows")[0]

    self.assertEqual(row["median_wall_seconds"], derive.run_wall_seconds(load_jobs("jobs.json")))
    self.assertEqual(row["median_wall_seconds"], 2470.0)


class PullRequestViewTest(ViewsTestCase):
  """Covers pr/<n>.json: everything one click needs, in one file."""

  def test_the_file_names_the_pull_request_and_the_run(self) -> None:
    """The header the commit modal and the Single PR page both read."""
    view = self.pr_view(MERGED_PR)

    self.assertEqual(view["schema"], views.SCHEMA_VERSION)
    self.assertEqual(view["group"], "pr")
    self.assertEqual(view["pr"], MERGED_PR)
    self.assertEqual(view["run_id"], MERGED_RUN_ID)
    self.assertEqual(view["author"], "guowei-dev")
    self.assertEqual(view["attempts"], 2)
    self.assertEqual(view["attempts_stored"], 2)
    self.assertEqual(view["updated_at"], COLLECTED_AT)

  def test_every_attempt_job_step_suite_test_and_error_is_there(self) -> None:
    """The counts are what the store holds for that run, nothing sampled away."""
    tables = self.pr_view(MERGED_PR)["tables"]
    counts = {name: views.table_row_count(table) for name, table in tables.items()}

    self.assertEqual(
        counts,
        {"attempts": 2, "jobs": 84, "steps": 710, "suites": 3, "tests": 102, "errors": 15},
    )

  def test_a_carried_over_job_is_marked_as_one(self) -> None:
    """28 of attempt 2's 42 jobs were not re-run; their numbers belong to attempt 1."""
    jobs = views.from_columnar(self.pr_view(MERGED_PR)["tables"]["jobs"])
    second = [job for job in jobs if job["attempt"] == 2]

    self.assertEqual(len(second), 42)
    self.assertEqual(sum(1 for job in second if job["carried_over"]), 28)

  def test_an_error_row_quotes_the_step_that_failed(self) -> None:
    """Nothing is summarised: the step's own name, and its own two timestamps."""
    errors = views.from_columnar(self.pr_view(MERGED_PR)["tables"]["errors"])
    name = "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit"
    row = [record for record in errors if record["job_name"] == name and record["attempt"] == 1][0]

    self.assertEqual(row["conclusion"], "failure")
    self.assertEqual(row["failed_step"], "Run Tests")
    self.assertEqual(row["failed_step_number"], 7)
    self.assertEqual(row["failed_step_started_at"], "2026-08-24T20:33:08Z")
    self.assertTrue(row["html_url"].endswith("/job/97581988113"))

  def test_a_step_travels_as_two_timestamps(self) -> None:
    """`derive.py` has no seconds-of-a-named-step, and this module does not grow one."""
    steps = views.from_columnar(self.pr_view(MERGED_PR)["tables"]["steps"])

    self.assertEqual(views.PR_STEPS_COLUMNS[-2:], ("started_at", "completed_at"))
    self.assertTrue(all("seconds" not in column for column in views.PR_STEPS_COLUMNS))
    self.assertEqual(steps[0]["name"], "Set up job")
    self.assertEqual(steps[0]["number"], 1)

  def test_the_test_rows_carry_the_partial_flag_of_their_suite(self) -> None:
    """A reader must not add worker 1's 102 tests up as if the suite were whole."""
    tests = views.from_columnar(self.pr_view(MERGED_PR)["tables"]["tests"])

    self.assertEqual(len(tests), PARTIAL_COLLECTED)
    self.assertEqual({record["suite_partial"] for record in tests}, {True})
    self.assertEqual({record["worker"] for record in tests}, {1})

  def test_no_file_is_written_for_a_run_that_is_not_a_merged_pull_request(self) -> None:
    """The scheduled run and the superseded one have no page to open."""
    self.assertFalse(views.pr_view_path(self.out, SUPERSEDED_PR).exists())
    self.assertEqual(sorted(path.name for path in (self.out / "views" / "pr").iterdir()), [f"{MERGED_PR}.json"])

  def test_building_an_unknown_pull_request_is_an_error(self) -> None:
    """A caller that asks for a pull request the month does not hold is a bug, not a gap."""
    month = views.load_month(self.store, AUGUST)

    with self.assertRaises(views.ViewError):
      views.build_pr_view(9999, month)


class DerivedNumbersTest(ViewsTestCase):
  """Every duration in a view is `derive.py`'s answer, not a second implementation."""

  def test_the_run_rows_phases_are_the_phase_split_of_attempt_one(self) -> None:
    """Six numbers, all from `derive.phase_split` over the same job payloads."""
    row = self.table("runs", AUGUST, "runs")[0]
    split = derive.phase_split(self.merged_jobs(1))

    self.assertEqual(row["queued_seconds"], split.queued_seconds)
    self.assertEqual(row["setup_seconds"], split.setup_seconds)
    self.assertEqual(row["tests_seconds"], split.tests_seconds)
    self.assertEqual(row["tail_seconds"], split.tail_seconds)
    self.assertEqual(row["total_seconds"], split.total_seconds)
    self.assertEqual(row["wall_seconds"], split.wall_seconds)
    self.assertEqual(row["first_created_at"], split.first_created_at)
    self.assertEqual(row["last_completed_at"], split.last_completed_at)
    self.assertEqual(row["jobs_counted"], split.jobs_counted)
    self.assertEqual(row["jobs_with_tests"], split.jobs_with_tests)
    self.assertEqual({key: row[key] for key in ATTEMPT_1_PHASES}, ATTEMPT_1_PHASES)

  def test_the_run_rows_machine_time_is_derives_machine_seconds(self) -> None:
    """Runner time, over the jobs that held a runner."""
    row = self.table("runs", AUGUST, "runs")[0]

    self.assertEqual(row["machine_seconds"], derive.machine_seconds(self.merged_jobs(1)))
    self.assertEqual(row["rerun_machine_seconds"], derive.machine_seconds(self.merged_jobs(2)))

  def test_a_job_rows_three_numbers_are_derives(self) -> None:
    """Wait, setup and run, per job, straight from `derive.py`."""
    name = "TPU Pretrain Tests (tpu-unit) / Execute Tests (1) / tpu-unit"
    row = [record for record in self.table("runs", AUGUST, "jobs") if record["name"] == name][0]
    job = named(self.merged_jobs(1), name)

    self.assertEqual(row["queued_seconds"], derive.queue_seconds(job))
    self.assertEqual(row["setup_seconds"], derive.setup_seconds(job))
    self.assertEqual(row["run_seconds"], derive.run_seconds(job))
    self.assertEqual((row["queued_seconds"], row["setup_seconds"], row["run_seconds"]), (14.0, 64.0, 1204.0))

  def test_a_suites_duration_is_the_run_tests_window_of_its_own_jobs(self) -> None:
    """Not a sum of run times and not a sum of JUnit seconds: 1626 s against 2519.7 s."""
    row = [record for record in self.table("suites", SEPTEMBER, "suites") if record["suite_id"] == "tpu-unit"][0]
    flavor_jobs = derive.jobs_for_flavor(load_jobs("jobs.json"), "tpu-unit")

    self.assertEqual(row["duration_seconds"], derive.suite_duration_seconds(flavor_jobs))
    self.assertEqual(row["duration_seconds"], TPU_UNIT_DURATION_SECONDS)
    self.assertEqual(row["machine_seconds"], derive.machine_seconds(flavor_jobs))
    self.assertEqual(row["machine_seconds"], TPU_UNIT_MACHINE_SECONDS)
    self.assertLess(row["duration_seconds"], row["junit_seconds"])

  def test_a_suites_worker_counts_are_derives_two_answers(self) -> None:
    """How many were named, and how many held a machine. Zero named is a naming fact."""
    jobs = load_jobs("jobs.json")
    for suite_id, named_count in (("tpu-unit", 2), ("tpu-pathways-unit", 0)):
      with self.subTest(suite=suite_id):
        row = [record for record in self.table("suites", SEPTEMBER, "suites") if record["suite_id"] == suite_id][0]
        flavor_jobs = derive.jobs_for_flavor(jobs, suite_id)

        self.assertEqual(row["workers_named"], derive.worker_count(flavor_jobs, suite_id))
        self.assertEqual(row["workers_named"], named_count)
        self.assertEqual(row["workers_ran"], sum(1 for job in flavor_jobs if derive.held_a_runner(job)))

  def test_the_pull_request_files_attempts_are_the_same_phase_splits(self) -> None:
    """One row per attempt, each one `derive.phase_split` of that attempt's jobs."""
    attempts = views.from_columnar(self.pr_view(MERGED_PR)["tables"]["attempts"])

    for record in attempts:
      with self.subTest(attempt=record["attempt"]):
        split = derive.phase_split(self.merged_jobs(record["attempt"]))
        self.assertEqual(record["total_seconds"], split.total_seconds)
        self.assertEqual(record["tests_seconds"], split.tests_seconds)
        self.assertEqual(record["machine_seconds"], derive.machine_seconds(self.merged_jobs(record["attempt"])))


class PublishedRunsTest(ViewsTestCase):
  """Covers which runs reach a view at all."""

  def test_a_superseded_run_reaches_nothing(self) -> None:
    """It was cancelled by a newer push; its numbers describe abandoned work."""
    for group, table in (("runs", "runs"), ("suites", "suites"), ("queue", "queue")):
      with self.subTest(group=group):
        run_ids = {record["run_id"] for record in self.table(group, SEPTEMBER, table)}
        self.assertNotIn(SUPERSEDED_RUN_ID, run_ids)

    self.assertEqual(len(self.table("workflows", SEPTEMBER, "workflows")), 1)
    self.assertEqual(self.table("workflows", SEPTEMBER, "workflows")[0]["runs"], 1)

  def test_a_pull_request_that_never_merged_is_not_charted_but_is_counted(self) -> None:
    """No chart draws it, and the workflow's machine minutes still include it."""
    august_runs = {record["run_id"] for record in self.table("runs", AUGUST, "runs")}
    august_queue = {record["run_id"] for record in self.table("queue", AUGUST, "queue")}

    self.assertNotIn(UNMERGED_RUN_ID, august_runs)
    self.assertNotIn(UNMERGED_RUN_ID, august_queue)
    self.assertEqual(self.table("workflows", AUGUST, "workflows")[0]["runs"], 2)

  def test_a_scheduled_runs_test_rows_are_not_published(self) -> None:
    """A daily snapshot is every test in the repository and no view ever draws it."""
    stored = self.store.read_rows(rows.KIND_TEST, [SEPTEMBER])
    month = views.load_month(self.store, SEPTEMBER)

    self.assertEqual(len(stored), PARTIAL_COLLECTED)
    self.assertEqual(month.attempts[(SCHEDULED_RUN_ID, 1)].tests, [])
    self.assertEqual(views.load_month(self.store, AUGUST).attempts[(MERGED_RUN_ID, 1)].tests[0].suite_id, "tpu-unit")

  def test_a_correction_wins_over_the_row_it_corrects(self) -> None:
    """Append-only means the last row per key is the one the view draws."""
    scheduled = load_json("run.json")
    fixed = junit.SuiteEntry(
        suite_id="tpu-unit",
        result=parse_suite("tpu-unit-1.xml"),
        per_worker={1: parse_suite("tpu-unit-1.xml")},
        missing_workers={2: junit.REASON_ARTIFACT_EXPIRED},
    )
    self.store.append(SEPTEMBER, rows.suite_row(scheduled, fixed, collected_at="2026-09-02T00:00:00Z"))
    self.build(date(2026, 9, 2), "2026-09-02T07:00:00Z")

    row = [record for record in self.table("suites", SEPTEMBER, "suites") if record["suite_id"] == "tpu-unit"][0]
    self.assertTrue(row["is_partial"])
    self.assertEqual(row["executed"], PARTIAL_EXECUTED)
    self.assertEqual(row["missing_workers"], [{"worker": 2, "reason": junit.REASON_ARTIFACT_EXPIRED}])


class MetaTest(ViewsTestCase):
  """Covers meta.json: the first file the browser reads, and the only one with a build time."""

  def test_it_carries_every_documented_field(self) -> None:
    """Schema, build time, window, groups, totals, the pending count and the file list."""
    meta = self.meta()

    self.assertEqual(
        sorted(meta),
        ["generated_at", "groups", "pull_requests", "schema", "totals", "uncollected_runs", "window_days"],
    )
    self.assertEqual(meta["schema"], views.SCHEMA_VERSION)
    self.assertEqual(meta["generated_at"], self.GENERATED_AT)
    self.assertEqual(meta["window_days"], views.VIEW_WINDOW_DAYS)
    self.assertEqual(sorted(meta["groups"]), sorted(views.VIEW_GROUPS))
    self.assertEqual(sorted(meta["totals"]), sorted(views.VIEW_GROUPS))
    self.assertEqual(meta["uncollected_runs"], 2)

  def test_each_group_lists_its_months_and_its_row_counts(self) -> None:
    """What exists, and how much is in it, before anything else is fetched."""
    groups = self.meta()["groups"]

    self.assertEqual(groups["runs"]["months"], [AUGUST, SEPTEMBER])
    self.assertEqual(groups["runs"]["rows"][AUGUST], {"runs": 1, "jobs": 21})
    self.assertEqual(groups["suites"]["rows"][SEPTEMBER], {"suites": 2})
    self.assertEqual(groups["flaky"]["rows"][AUGUST], {"rescues": 10, "rescue_tests": 0})
    self.assertEqual(self.meta()["totals"]["runs"], 22)

  def test_the_counts_match_the_files_on_disk(self) -> None:
    """meta.json is what the browser trusts, so it has to agree with every file."""
    meta = self.meta()

    for group, entry in meta["groups"].items():
      for month in entry["months"]:
        with self.subTest(group=group, month=month):
          tables = self.view(group, month)["tables"]
          counted = {name: views.table_row_count(table) for name, table in tables.items()}
          self.assertEqual(entry["rows"][month], counted)

  def test_every_pull_request_is_listed_with_its_file_and_its_timestamp(self) -> None:
    """The browser refetches one pull request file only when its `updated_at` moved."""
    entry = self.meta()["pull_requests"]

    self.assertEqual(sorted(entry), [str(MERGED_PR)])
    self.assertEqual(entry[str(MERGED_PR)], {"file": f"pr/{MERGED_PR}.json", "updated_at": COLLECTED_AT})
    self.assertTrue((self.out / "views" / entry[str(MERGED_PR)]["file"]).exists())

  def test_the_pending_count_comes_from_the_store_unless_it_is_given(self) -> None:
    """A number here is normal; a number that keeps growing is not, so it is never hidden."""
    self.assertEqual(self.build(self.TODAY, "2026-09-01T08:00:00Z", uncollected=17)["meta"]["uncollected_runs"], 17)
    self.assertEqual(self.build(self.TODAY, "2026-09-01T09:00:00Z")["meta"]["uncollected_runs"], 2)

  def test_no_month_file_carries_a_build_time(self) -> None:
    """A timestamp in a view would change every file on every tick."""
    for group in views.VIEW_GROUPS:
      for month in (AUGUST, SEPTEMBER):
        with self.subTest(group=group, month=month):
          view = self.view(group, month)
          self.assertEqual(sorted(view), ["group", "month", "schema", "tables"])
          self.assertNotIn("generated_at", json.dumps(view))


class WriteCycleTest(ViewsTestCase):
  """Covers what a tick writes, what it leaves alone, and what it never reads."""

  def august_files(self) -> list[Path]:
    """Returns August's five month files.

    Returns:
      The paths, sorted by name.
    """
    return sorted((self.out / "views").glob(f"*-{AUGUST}.json"))

  def snapshot(self, paths: list[Path]) -> dict[str, tuple[bytes, int]]:
    """Records the bytes and the modification time of every path.

    Args:
      paths: The files to record.

    Returns:
      File name -> (bytes, modification time in nanoseconds).
    """
    return {path.name: (path.read_bytes(), os.stat(path).st_mtime_ns) for path in paths}

  def append_correction(self, collected_at: str) -> None:
    """Appends a corrected August job row that would change the runs view if it were read.

    Args:
      collected_at: The correction's write timestamp, later than the row it corrects.
    """
    run = dict(load_json(f"rerun-{MERGED_RUN_ID}-run.json"), run_attempt=1)
    job = dict(self.merged_jobs(1)[0], completed_at="2026-08-24T23:59:59Z")
    self.store.append(AUGUST, rows.job_row(run, job, collected_at=collected_at))

  def test_the_first_tick_writes_every_open_month_and_meta(self) -> None:
    """Five groups times two months, plus meta.json, plus one pull request file."""
    expected = sorted(
        [f"views/{group}-{month}.json" for group in views.VIEW_GROUPS for month in (AUGUST, SEPTEMBER)]
        + ["views/meta.json"]
    )

    self.assertEqual(self.summary["months_rebuilt"], [AUGUST, SEPTEMBER])
    self.assertEqual(self.summary["months_skipped"], [])
    self.assertEqual(self.summary["written"], expected)
    self.assertEqual(self.summary["unchanged"], [])
    self.assertEqual(self.summary["pr_written"], [f"views/pr/{MERGED_PR}.json"])

  def test_a_tick_that_changes_nothing_rewrites_nothing(self) -> None:
    """Same rows and the same build time means the same bytes, so git sees no commit."""
    before = self.snapshot(self.august_files() + [views.meta_path(self.out)])
    again = self.build(self.TODAY, self.GENERATED_AT)

    self.assertEqual(again["written"], [])
    self.assertEqual(again["pr_written"], [])
    self.assertEqual(again["pr_unchanged"], [f"views/pr/{MERGED_PR}.json"])
    self.assertEqual(len(again["unchanged"]), 11)
    self.assertEqual(self.snapshot(self.august_files() + [views.meta_path(self.out)]), before)

  def test_only_meta_changes_when_only_the_build_time_moves(self) -> None:
    """`generated_at` lives in meta.json alone, so nothing else can be dirtied by a tick."""
    again = self.build(self.TODAY, "2026-09-01T11:00:00Z")

    self.assertEqual(again["written"], ["views/meta.json"])
    self.assertEqual(len(again["unchanged"]), 10)

  def test_a_closed_month_is_not_touched_even_when_its_rows_changed(self) -> None:
    """The heart of the format: a closed month's file is not read and not rewritten."""
    before = self.snapshot(self.august_files())
    pr_path = views.pr_view_path(self.out, MERGED_PR)
    pr_before = self.snapshot([pr_path])
    self.append_correction("2026-09-20T00:00:00Z")

    later = self.build(date(2026, 9, 15), "2026-09-15T07:00:00Z")

    self.assertEqual(later["months_rebuilt"], [SEPTEMBER])
    self.assertEqual(later["months_skipped"], [AUGUST])
    self.assertEqual(self.snapshot(self.august_files()), before)
    self.assertEqual(self.snapshot([pr_path]), pr_before)
    for path in later["written"] + later["unchanged"] + later["pr_written"] + later["pr_unchanged"]:
      with self.subTest(path=path):
        self.assertNotIn(AUGUST, path)

  def test_a_closed_months_counts_still_reach_meta(self) -> None:
    """Read back off the finished file, which is cheaper than re-reading a month of rows."""
    before = self.meta()["groups"]
    self.build(date(2026, 9, 15), "2026-09-15T07:00:00Z")
    after = self.meta()["groups"]

    for group in views.VIEW_GROUPS:
      with self.subTest(group=group):
        self.assertEqual(after[group]["months"], [AUGUST, SEPTEMBER])
        self.assertEqual(after[group]["rows"][AUGUST], before[group]["rows"][AUGUST])

  def test_a_closed_month_is_not_even_read_out_of_the_store(self) -> None:
    """Skipping a month has to save the reading, not just the writing."""
    self.store.reads.clear()
    self.build(date(2026, 9, 15), "2026-09-15T07:00:00Z")

    self.assertTrue(self.store.reads)
    for kind, months in self.store.reads:
      with self.subTest(kind=kind, months=months):
        self.assertNotIn(AUGUST, months or ())

  def test_a_pull_requests_timestamp_moves_only_when_its_file_is_rewritten(self) -> None:
    """`updated_at` is the newest `collected_at` behind the file, not the build time."""
    key = str(MERGED_PR)
    self.assertEqual(self.meta()["pull_requests"][key]["updated_at"], COLLECTED_AT)

    # A tick that re-derives an unchanged pull request leaves the file and the stamp alone.
    self.build(self.TODAY, "2026-09-01T12:00:00Z")
    self.assertEqual(self.meta()["pull_requests"][key]["updated_at"], COLLECTED_AT)

    # A correction, with the month open: the file is rewritten and the stamp moves with it.
    self.append_correction("2026-09-20T00:00:00Z")
    reopened = self.build(self.TODAY, "2026-09-01T13:00:00Z")
    self.assertEqual(reopened["pr_written"], [f"views/pr/{MERGED_PR}.json"])
    self.assertEqual(self.meta()["pull_requests"][key]["updated_at"], "2026-09-20T00:00:00Z")

    # Another correction, with the month closed: nothing is rewritten and the stamp stays.
    self.append_correction("2026-09-21T00:00:00Z")
    closed = self.build(date(2026, 9, 15), "2026-09-15T07:00:00Z")
    self.assertEqual(closed["pr_written"], [])
    self.assertEqual(closed["pr_unchanged"], [])
    self.assertEqual(self.meta()["pull_requests"][key]["updated_at"], "2026-09-20T00:00:00Z")

  def test_a_caller_can_name_the_months_to_rebuild(self) -> None:
    """A tick that knows which months it appended to does not have to guess."""
    named_months = self.build(date(2026, 9, 15), "2026-09-15T08:00:00Z", months=[AUGUST, "2026-01"])

    self.assertEqual(named_months["months_rebuilt"], [AUGUST])
    self.assertEqual(named_months["months_skipped"], [SEPTEMBER])

  def test_the_summary_says_what_the_store_holds(self) -> None:
    """The fields a tick logs, so a run that published nothing can be seen to have run."""
    self.assertEqual(self.summary["months_available"], [AUGUST, SEPTEMBER])
    self.assertEqual(self.summary["window_days"], views.VIEW_WINDOW_DAYS)
    self.assertEqual(self.summary["generated_at"], self.GENERATED_AT)
    self.assertEqual(self.summary["out_dir"], str(self.out))
    self.assertEqual(self.summary["uncollected_runs"], 2)


class ViewFileShapeTest(ViewsTestCase):
  """Covers the envelope every written file shares, and its column lists on disk."""

  def test_every_month_file_names_its_group_and_month(self) -> None:
    """The browser checks the schema before it draws anything."""
    for group in views.VIEW_GROUPS:
      for month in (AUGUST, SEPTEMBER):
        with self.subTest(group=group, month=month):
          view = self.view(group, month)
          self.assertEqual(view["schema"], views.SCHEMA_VERSION)
          self.assertEqual(view["group"], group)
          self.assertEqual(view["month"], month)

  def test_the_columns_on_disk_are_the_column_lists(self) -> None:
    """A builder that wrote a different order would break the browser's rehydration."""
    expected = {
        "runs": {"runs": views.RUNS_COLUMNS, "jobs": views.RUN_JOBS_COLUMNS},
        "suites": {"suites": views.SUITES_COLUMNS},
        "flaky": {"rescues": views.RESCUES_COLUMNS, "rescue_tests": views.RESCUE_TESTS_COLUMNS},
        "queue": {"queue": views.QUEUE_COLUMNS},
        "workflows": {"workflows": views.WORKFLOWS_COLUMNS},
    }

    for group, tables in expected.items():
      view = self.view(group, AUGUST)
      self.assertEqual(sorted(view["tables"]), sorted(tables))
      for name, columns in tables.items():
        with self.subTest(group=group, table=name):
          self.assertEqual(view["tables"][name]["columns"], list(columns))

  def test_the_pull_request_files_columns_are_the_column_lists(self) -> None:
    """Six tables, each pinned the same way."""
    expected = {
        "attempts": views.PR_ATTEMPTS_COLUMNS,
        "jobs": views.PR_JOBS_COLUMNS,
        "steps": views.PR_STEPS_COLUMNS,
        "suites": views.PR_SUITES_COLUMNS,
        "tests": views.PR_TESTS_COLUMNS,
        "errors": views.PR_ERRORS_COLUMNS,
    }
    tables = self.pr_view(MERGED_PR)["tables"]

    self.assertEqual(sorted(tables), sorted(expected))
    for name, columns in expected.items():
      with self.subTest(table=name):
        self.assertEqual(tables[name]["columns"], list(columns))

  def test_every_table_the_dashboard_reads_carries_what_it_needs(self) -> None:
    """Named against the hard-coded constant each group replaces."""
    for path, needs in DASHBOARD_NEEDS.items():
      group, table = path.split(".")
      columns = self.view(group, AUGUST)["tables"][table]["columns"]
      for constant, wanted in needs.items():
        for column in wanted:
          with self.subTest(group=group, table=table, replaces=constant, column=column):
            self.assertIn(column, columns)

  def test_the_pull_request_file_carries_what_one_click_needs(self) -> None:
    """ATTEMPT_INFO, TESTS, TEST_COUNTS, STEPS and the error rows, in one fetch."""
    tables = self.pr_view(MERGED_PR)["tables"]

    for table, needs in PR_VIEW_NEEDS.items():
      for constant, wanted in needs.items():
        for column in wanted:
          with self.subTest(table=table, replaces=constant, column=column):
            self.assertIn(column, tables[table]["columns"])

  def test_every_table_on_disk_round_trips(self) -> None:
    """The browser's two lines have to work on the real files, not only on made-up ones."""
    files = list((self.out / "views").glob("*.json")) + list((self.out / "views" / "pr").glob("*.json"))
    checked = 0

    for path in files:
      payload = json.loads(path.read_text(encoding="utf-8"))
      for name, table in (payload.get("tables") or {}).items():
        with self.subTest(file=path.name, table=name):
          records = views.from_columnar(table)
          self.assertEqual(views.to_columnar(records, table["columns"]), table)
          checked += 1

    # Seven tables a month across the five groups, twice, and the six of the one pull request.
    self.assertEqual(checked, 7 * 2 + 6)

  def test_a_month_ahead_of_the_clock_is_rebuilt_and_advertised(self) -> None:
    """A store whose clock ran ahead gets its file written AND named in meta.json.

    Writing the file without advertising it is the same as not writing it: the browser only
    fetches the months meta.json lists. `months_to_rebuild` reopens a month later than today
    on purpose, so `build_meta` has to carry it too.
    """
    october = "2026-10"
    scheduled = dict(load_json("run.json"), id=33468578999, created_at="2026-10-02T04:06:01Z")
    self.store.append(october, rows.run_row(scheduled, None, collected_at=COLLECTED_AT))

    ahead = self.build(date(2026, 9, 15), "2026-09-15T09:00:00Z")

    self.assertIn(october, ahead["months_rebuilt"])
    self.assertTrue(views.view_path(self.out, "runs", october).exists())
    self.assertIn(october, self.meta()["groups"]["runs"]["months"])
    self.assertEqual(self.meta()["groups"]["runs"]["months"][-1], october, "the newest month sorts last")


class ColumnarSizeTest(ViewsTestCase):
  """The columnar form has to be smaller on the real tables, not only in principle."""

  def test_every_real_table_is_smaller_columnar(self) -> None:
    """Six hundred rows of each real shape, serialised both ways."""
    tables = {
        "runs": (self.view("runs", AUGUST)["tables"]["runs"], views.RUNS_COLUMNS),
        "jobs": (self.view("runs", AUGUST)["tables"]["jobs"], views.RUN_JOBS_COLUMNS),
        "suites": (self.view("suites", AUGUST)["tables"]["suites"], views.SUITES_COLUMNS),
        "rescues": (self.view("flaky", AUGUST)["tables"]["rescues"], views.RESCUES_COLUMNS),
        "queue": (self.view("queue", AUGUST)["tables"]["queue"], views.QUEUE_COLUMNS),
    }

    for name, (table, columns) in tables.items():
      with self.subTest(table=name):
        records = views.from_columnar(table)
        many = [dict(records[index % len(records)]) for index in range(600)]
        columnar = json.dumps(views.to_columnar(many, columns), separators=(",", ":"))
        objects = json.dumps(many, separators=(",", ":"))

        self.assertLess(len(columnar), len(objects))
        self.assertLess(len(columnar) / len(objects), 0.7)


class TwoPushesTestCase(OfflineTestCase):
  """Base for the two tests about a pull request that pushed twice.

  A push whose run finished before the next push arrived is NOT superseded - supersession
  only marks a run GitHub cancelled - so the store legitimately holds two completed runs for
  one merged pull request. Only one of them may be drawn as that pull request.
  """

  AUGUST_RUN = 800000010
  SEPTEMBER_RUN = 900000010
  PR_NUMBER = 6001

  def two_runs(self, months: tuple[str, str]) -> FakeStore:
    """Builds a store holding two completed runs of one merged pull request.

    Args:
      months: The month each run is filed under, older run first.

    Returns:
      The store.
    """
    store = FakeStore()
    pull = dict(load_json("fork-pr-5042-pulls-by-head.json")[0], number=self.PR_NUMBER)
    pull["head"] = dict(pull["head"], sha="b" * 40)
    base = load_json(f"rerun-{MERGED_RUN_ID}-run.json")
    jobs = load_jobs(f"rerun-{MERGED_RUN_ID}-attempt2-jobs.json")
    pushes = (
        (months[0], self.AUGUST_RUN, "2026-08-31T20:00:00Z", "a" * 40, "failure"),
        (months[1], self.SEPTEMBER_RUN, "2026-09-01T08:00:00Z", "b" * 40, "success"),
    )
    for month, run_id, created, head_sha, conclusion in pushes:
      payload = dict(
          base,
          id=run_id,
          run_attempt=1,
          created_at=created,
          run_started_at=created,
          updated_at=created,
          head_sha=head_sha,
          conclusion=conclusion,
      )
      store.append(month, rows.run_row(payload, pull, collected_at=COLLECTED_AT))
      for index, job in enumerate(jobs):
        # A job row carries its run id, and `rows.job_row` refuses one that disagrees with the
        # run it is passed, so the saved jobs are re-pointed at the run that is standing in.
        moved = dict(job, run_id=run_id, run_attempt=1, id=run_id * 100 + index)
        store.append(month, rows.job_row(payload, moved, collected_at=COLLECTED_AT))
      store.append(
          month,
          rows.suite_row(
              payload,
              junit.SuiteEntry(suite_id="tpu-unit", result=parse_suite("tpu-unit-1.xml"), per_worker={1: None}),
              collected_at=COLLECTED_AT,
          ),
      )
    return store


class PullRequestFileOwnerTest(TwoPushesTestCase):
  """`pr/<n>.json` must describe the newest run, whichever month a tick happens to rebuild."""

  def owner_of(self, out: Path) -> int:
    """Returns the run id the pull request file on disk names.

    Args:
      out: The output directory.

    Returns:
      The run id.
    """
    return json.loads(views.pr_view_path(out, self.PR_NUMBER).read_text(encoding="utf-8"))["run_id"]

  def test_the_newest_run_owns_the_file(self) -> None:
    """Both months rebuilt: the September push is the one the file describes."""
    out = self.temp_dir()

    views.build_views(self.two_runs((AUGUST, SEPTEMBER)), out, date(2026, 9, 1), generated_at=COLLECTED_AT)

    self.assertEqual(self.owner_of(out), self.SEPTEMBER_RUN)

  def test_rebuilding_the_older_month_alone_does_not_overwrite_it(self) -> None:
    """The bug this test exists for: backfill touching August must not undo September."""
    store = self.two_runs((AUGUST, SEPTEMBER))
    out = self.temp_dir()
    views.build_views(store, out, date(2026, 9, 1), generated_at=COLLECTED_AT)

    summary = views.build_views(store, out, date(2026, 9, 1), months=[AUGUST], generated_at=COLLECTED_AT)

    self.assertEqual(summary["months_rebuilt"], [AUGUST])
    self.assertEqual(self.owner_of(out), self.SEPTEMBER_RUN, "the abandoned push must not take the file")
    self.assertIn(f"views/pr/{self.PR_NUMBER}.json", summary["pr_unchanged"])
    self.assertEqual(summary["pr_written"], [])

  def test_the_file_still_stands_once_the_newer_month_has_closed(self) -> None:
    """Two months on, `months_to_rebuild` never picks September again; nothing may repair it."""
    store = self.two_runs((AUGUST, SEPTEMBER))
    out = self.temp_dir()
    views.build_views(store, out, date(2026, 9, 1), generated_at=COLLECTED_AT)

    views.build_views(store, out, date(2026, 11, 20), months=[AUGUST], generated_at=COLLECTED_AT)

    self.assertEqual(self.owner_of(out), self.SEPTEMBER_RUN)

  def test_its_updated_at_is_carried_rather_than_lost(self) -> None:
    """meta.json still has to name a timestamp for a file it did not rewrite."""
    store = self.two_runs((AUGUST, SEPTEMBER))
    out = self.temp_dir()
    views.build_views(store, out, date(2026, 9, 1), generated_at=COLLECTED_AT)

    summary = views.build_views(store, out, date(2026, 9, 1), months=[AUGUST], generated_at=COLLECTED_AT)

    entry = summary["meta"]["pull_requests"][str(self.PR_NUMBER)]
    self.assertEqual(entry["updated_at"], COLLECTED_AT)
    self.assertEqual(entry["file"], f"pr/{self.PR_NUMBER}.json")


class RepresentativeRunTest(TwoPushesTestCase):
  """A pull request is one point on the axis, even when two of its runs are published."""

  def setUp(self) -> None:
    """Builds a month holding both pushes of one pull request."""
    super().setUp()
    self.out = self.temp_dir()
    self.store = self.two_runs((SEPTEMBER, SEPTEMBER))
    views.build_views(self.store, self.out, date(2026, 9, 1), generated_at=COLLECTED_AT)

  def records(self, group: str, table: str) -> list[dict[str, Any]]:
    """Reads one table of the September file back as records.

    Args:
      group: The view group.
      table: The table name inside it.

    Returns:
      The rehydrated records.
    """
    payload = json.loads(views.view_path(self.out, group, SEPTEMBER).read_text(encoding="utf-8"))
    return views.from_columnar(payload["tables"][table])

  def test_the_runs_view_has_one_row_and_it_is_the_newest_push(self) -> None:
    """The runs view is the one place `pr` is unique, and it stays that way."""
    runs_rows = [row for row in self.records("runs", "runs") if row["pr"] == self.PR_NUMBER]

    self.assertEqual([row["run_id"] for row in runs_rows], [self.SEPTEMBER_RUN])

  def test_the_suites_and_queue_views_mark_which_run_that_was(self) -> None:
    """Both rows are published; exactly one carries the flag."""
    for group, table in (("suites", "suites"), ("queue", "queue")):
      with self.subTest(group=group):
        mine = [row for row in self.records(group, table) if row["pr"] == self.PR_NUMBER]

        self.assertEqual(len({row["run_id"] for row in mine}), 2, "both pushes really ran")
        flagged = {row["run_id"] for row in mine if row["is_representative"]}
        self.assertEqual(flagged, {self.SEPTEMBER_RUN})

  def test_a_scheduled_run_represents_itself(self) -> None:
    """The flag must not read as "not worth drawing" for a run with no pull request."""
    store = build_fixture_store()
    out = self.temp_dir()
    views.build_views(store, out, date(2026, 9, 1), generated_at=COLLECTED_AT)
    payload = json.loads(views.view_path(out, "suites", SEPTEMBER).read_text(encoding="utf-8"))

    scheduled = [row for row in views.from_columnar(payload["tables"]["suites"]) if row["pr"] is None]

    self.assertTrue(scheduled)
    self.assertTrue(all(row["is_representative"] for row in scheduled))


class OverlappingRunsTest(OfflineTestCase):
  """The overlap count has to be right for a run whose window is a single instant."""

  def store_with(self, windows: dict[int, tuple[str, str]]) -> FakeStore:
    """Builds a month of merged pull request runs with the given windows and no jobs.

    With no job rows the window falls back to `created_at`..`updated_at`, which is how a run
    whose jobs were never stored reaches the counter.

    Args:
      windows: run id -> (created_at, updated_at).

    Returns:
      The store.
    """
    store = FakeStore()
    base = load_json(f"rerun-{MERGED_RUN_ID}-run.json")
    for index, (run_id, (created, updated)) in enumerate(sorted(windows.items())):
      pull = dict(load_json("fork-pr-5042-pulls-by-head.json")[0], number=7000 + index)
      pull["head"] = dict(pull["head"], sha=f"{index:040d}")
      payload = dict(base, id=run_id, run_attempt=1, created_at=created, updated_at=updated, head_sha=f"{index:040d}")
      store.append(SEPTEMBER, rows.run_row(payload, pull, collected_at=COLLECTED_AT))
    return store

  def counts(self, windows: dict[int, tuple[str, str]]) -> dict[int, int]:
    """Builds the runs view and returns each run's overlap count.

    Args:
      windows: run id -> (created_at, updated_at).

    Returns:
      run id -> overlapping_runs.
    """
    out = self.temp_dir()
    views.build_views(self.store_with(windows), out, date(2026, 9, 1), generated_at=COLLECTED_AT)
    payload = json.loads(views.view_path(out, "runs", SEPTEMBER).read_text(encoding="utf-8"))
    return {row["run_id"]: row["overlapping_runs"] for row in views.from_columnar(payload["tables"]["runs"])}

  def test_two_runs_that_share_time_each_count_the_other(self) -> None:
    """The ordinary case, unchanged."""
    counts = self.counts(
        {
            901: ("2026-09-01T00:00:00Z", "2026-09-01T01:00:00Z"),
            902: ("2026-09-01T00:30:00Z", "2026-09-01T01:30:00Z"),
        }
    )

    self.assertEqual(counts, {901: 1, 902: 1})

  def test_a_run_with_no_window_of_its_own_still_counts_what_it_sat_inside(self) -> None:
    """A run created and finished in the same second used to report zero."""
    counts = self.counts(
        {
            901: ("2026-09-01T00:00:00Z", "2026-09-01T01:00:00Z"),
            902: ("2026-09-01T00:30:00Z", "2026-09-01T00:30:00Z"),
        }
    )

    self.assertEqual(counts, {901: 1, 902: 1})

  def test_a_run_that_shares_no_time_with_another_counts_none(self) -> None:
    """The counter still has to be able to say zero."""
    counts = self.counts(
        {
            901: ("2026-09-01T00:00:00Z", "2026-09-01T01:00:00Z"),
            902: ("2026-09-01T03:00:00Z", "2026-09-01T04:00:00Z"),
        }
    )

    self.assertEqual(counts, {901: 0, 902: 0})

  def test_the_count_matches_a_brute_force_reading_of_the_same_rule(self) -> None:
    """Ten windows, degenerate ones included, checked against the definition itself."""
    windows = {
        900 + index: (f"2026-09-01T0{index}:00:00Z", f"2026-09-01T0{min(index + span, 9)}:00:00Z")
        for index, span in enumerate([3, 0, 1, 0, 5, 2, 0, 4, 1, 0])
    }
    expected = {
        run_id: sum(
            1
            for other, (other_start, other_end) in windows.items()
            if other != run_id and other_start <= windows[run_id][1] and other_end >= windows[run_id][0]
        )
        for run_id in windows
    }

    self.assertEqual(self.counts(windows), expected)


class UnmeasurableSuiteTest(OfflineTestCase):
  """A suite whose flavor has no stored jobs is unmeasured, which is not the same as zero."""

  def test_a_suite_with_no_jobs_of_its_flavor_reports_null_not_zero(self) -> None:
    """`derive.machine_seconds([])` answers 0.0; publishing that would draw a free suite."""
    store = build_fixture_store()
    scheduled = load_json("run.json")
    orphan = junit.SuiteEntry(suite_id="gpu-post-training-unit", result=parse_suite("tpu-unit-1.xml"))
    store.append(SEPTEMBER, rows.suite_row(scheduled, orphan, collected_at=COLLECTED_AT))
    out = self.temp_dir()

    views.build_views(store, out, date(2026, 9, 1), generated_at=COLLECTED_AT)

    payload = json.loads(views.view_path(out, "suites", SEPTEMBER).read_text(encoding="utf-8"))
    found = [row for row in views.from_columnar(payload["tables"]["suites"]) if row["suite_id"] == orphan.suite_id]
    self.assertEqual(len(found), 1)
    row = found[0]
    for column in ("workers_named", "workers_ran", "duration_seconds", "machine_seconds"):
      with self.subTest(column=column):
        self.assertIsNone(row[column])
    self.assertIsNotNone(row["executed"], "its JUnit counts are real and stay")


if __name__ == "__main__":
  unittest.main(verbosity=2)
