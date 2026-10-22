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

"""Offline tests for the options a tick with no GitHub token needs.

With no token GitHub allows 60 requests an hour, answers 401 to every artifact download, and
asks a rate-limited caller to wait up to an hour. So such a tick runs with `--rate-floor` (the
default floor of 300 would stop it before its first run), `--skip-tests` (no artifact is asked
for; each suite is stored with no counts and a reason), `--max-wait` (a long wait fails at once)
and `--merge-checks list` (merges come from pages of closed pull requests, not one request per
waiting pull request). Fake clients answer what each test needs and fail on anything else, so
nothing here reaches GitHub.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import requests

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import derive
from collector import github
from collector import junit
from collector import rows
from collector import store
from collector import tick
from collector import views

# These tests drive the tick's private steps one at a time.
# pylint: disable=protected-access

FIXTURES = Path(__file__).resolve().parent / "fixtures"

RUNNING = {"id": 101, "status": "in_progress", "run_attempt": 1, "created_at": "2026-10-15T00:00:00Z"}

NOW = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)


def load_fixture(name: str) -> Any:
  """Reads one of the real payloads captured on 2026-09-01."""
  return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def iso(moment: datetime) -> str:
  """Formats a moment the way GitHub does."""
  return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def closed_pull(number: int, updated: datetime, merged: bool) -> dict[str, Any]:
  """Builds one entry of the closed pull requests listing, with the fields a run row copies."""
  return {
      "number": number,
      "state": "closed",
      "title": f"PR {number}",
      "updated_at": iso(updated),
      "closed_at": iso(updated),
      "merged_at": iso(updated) if merged else None,
      "created_at": iso(updated - timedelta(days=1)),
      "merge_commit_sha": f"sha{number}",
      "head": {"sha": f"head{number}", "label": f"someone:branch{number}"},
      "base": {"ref": "main"},
      "user": {"login": "someone"},
      "html_url": f"https://github.com/AI-Hypercomputer/maxtext/pull/{number}",
      "draft": False,
  }


def full_page(first_number: int, newest: datetime, step: timedelta = timedelta(minutes=10)) -> list[dict[str, Any]]:
  """Builds a full page (100 entries) of closed, unmerged pull requests, newest update first."""
  return [closed_pull(first_number + index, newest - index * step, merged=False) for index in range(github.PER_PAGE)]


class BudgetClient:
  """Answers the budget question from a script of readings. Anything else fails the test."""

  def __init__(self, readings: list[int] | int | Exception) -> None:
    self.readings = readings
    self.remaining_hint: int | None = None
    self.budget_reads = 0

  def rate_limit(self) -> dict[str, int]:
    """Returns the next scripted reading (the last one repeats), or raises the scripted error."""
    self.budget_reads += 1
    if isinstance(self.readings, Exception):
      raise self.readings
    if isinstance(self.readings, int):
      remaining = self.readings
    else:
      remaining = self.readings[min(self.budget_reads, len(self.readings)) - 1]
    self.remaining_hint = remaining
    return {"limit": 60, "remaining": remaining, "reset": 0}

  def __getattr__(self, name: str) -> Any:
    raise AssertionError(f"the tick asked the fake client for {name!r}; these tests must not reach GitHub")


class ListingClient(BudgetClient):
  """Also answers the closed pull requests listing, one scripted page per call."""

  def __init__(self, pages: list[list[dict[str, Any]]] | Exception, readings: list[int] | int = 60) -> None:
    super().__init__(readings)
    self.pages = pages
    self.listing_calls: list[tuple[str, str, int | None, dict[str, Any]]] = []

  def paginate(self, path: str, key: str, *, max_items: int | None = None, **params: Any) -> list[Any]:
    """Returns the page the call asks for, or raises the scripted error."""
    self.listing_calls.append((path, key, max_items, dict(params)))
    if isinstance(self.pages, Exception):
      raise self.pages
    page = int(params["page"])
    if page > len(self.pages):
      return []
    entries = self.pages[page - 1]
    if isinstance(entries, Exception):
      raise entries
    return [dict(entry) if isinstance(entry, dict) else entry for entry in entries]


class TickTestCase(unittest.TestCase):
  """Gives each test an empty store in a temporary directory."""

  def setUp(self) -> None:
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp(prefix="ci-metrics-no-token-")
    self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

  def make_tick(self, client: Any, **option_values: Any) -> tick.Tick:
    """Returns a tick over the empty store, with the given options."""
    row_store = store.Store(self.tmp_dir)
    state = row_store.load_state()
    options = tick.Options(out=Path(self.tmp_dir), backfill_days=1, **option_values)
    subject = tick.Tick(options, client, row_store, state, tick.TickReport())
    subject.now = NOW
    return subject


class RateFloorTest(TickTestCase):

  def setUp(self) -> None:
    super().setUp()
    patch = mock.patch.object(tick.Tick, "_collect_run")
    self.collect_run = patch.start()
    self.addCleanup(patch.stop)

  def test_default_floor_is_unchanged(self):
    self.assertEqual(tick.Options(out=Path(self.tmp_dir)).rate_floor, tick.RATE_LIMIT_FLOOR)
    self.assertEqual(tick.RATE_LIMIT_FLOOR, 300)

  def test_default_floor_stops_a_tick_with_no_token_before_its_first_run(self):
    # This is why the option exists: 60 requests is below the floor of 300.
    subject = self.make_tick(BudgetClient(60))
    self.assertFalse(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_not_called()

  def test_low_floor_reads_a_run(self):
    subject = self.make_tick(BudgetClient(15), rate_floor=15)
    self.assertTrue(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_called_once()
    self.assertIsNone(subject.report.stopped_because)

  def test_low_floor_still_stops_below_it(self):
    subject = self.make_tick(BudgetClient(14), rate_floor=15)
    self.assertFalse(subject._collect_one(dict(RUNNING)))
    self.collect_run.assert_not_called()
    self.assertIn("14 of 60 requests left, floor is 15", subject.report.stopped_because)
    # A budget stop is not lost data.
    self.assertEqual(subject.report.problems, [])

  def test_the_report_keeps_the_first_and_the_last_budget_reading(self):
    # With no token the budget belongs to an IP address, so these readings are the record of it.
    subject = self.make_tick(BudgetClient([60, 41, 16]), rate_floor=15)
    self.assertIsNone(subject.report.budget_first)
    for _ in range(3):
      self.assertTrue(subject._budget_allows())
    self.assertEqual(subject.report.budget_first, "60 of 60")
    self.assertEqual(subject.report.budget_last, "16 of 60")

  def test_a_reading_from_the_header_alone_is_kept_too(self):
    client = BudgetClient(github.GitHubError("the rate-limit endpoint is down", status=503))
    client.remaining_hint = 20
    subject = self.make_tick(client, rate_floor=15)
    self.assertTrue(subject._budget_allows())
    self.assertEqual(subject.report.budget_first, "20 of ?")

  def test_low_floor_checks_the_budget_before_every_merge_check(self):
    # With the default interval of 25, a floor of 15 could be overshot by 24 requests.
    subject = self.make_tick(BudgetClient(14), rate_floor=15)
    waiting = {number: 1000 + number for number in range(1, 61)}
    with mock.patch.object(tick.Tick, "_open_pull_requests", return_value=waiting):
      with mock.patch.object(tick.runs, "get_pull_request", return_value={"state": "open"}) as get_pull_request:
        subject._refresh_merges()
    get_pull_request.assert_called_once()
    self.assertIn("API budget ran low", subject.report.stopped_because)

  def test_default_floor_keeps_the_merge_check_interval(self):
    subject = self.make_tick(BudgetClient(1000))
    waiting = {number: 1000 + number for number in range(1, 61)}
    with mock.patch.object(tick.Tick, "_open_pull_requests", return_value=waiting):
      with mock.patch.object(tick.runs, "get_pull_request", return_value={"state": "open"}) as get_pull_request:
        subject._refresh_merges()
    self.assertEqual(get_pull_request.call_count, 60)
    self.assertEqual(subject.client.budget_reads, 2)


class SkipTestsTest(TickTestCase):

  def setUp(self) -> None:
    super().setUp()
    self.run = load_fixture("run.json")
    self.jobs = load_fixture("jobs.json")["jobs"]
    self.created = datetime(2026, 9, 1, 4, 6, 1, tzinfo=timezone.utc)
    patch = mock.patch.object(tick.junit, "read_run_tests", side_effect=AssertionError("an artifact was asked for"))
    self.read_run_tests = patch.start()
    self.addCleanup(patch.stop)

  def make_skipping_tick(self, hours_after_run: float = 1.0) -> tick.Tick:
    """Returns a skip-tests tick whose clock is a while after the fixture run was created."""
    subject = self.make_tick(BudgetClient(60), skip_tests=True, rate_floor=15)
    subject.now = self.created + timedelta(hours=hours_after_run)
    return subject

  def test_unread_run_tests_lists_every_suite_with_the_reason(self):
    found = junit.unread_run_tests(7, ["cpu-unit", "tpu-unit"])
    self.assertEqual(sorted(found.suites), ["cpu-unit", "decoupled", "tpu-unit"])
    for entry in found.suites.values():
      self.assertIsNone(entry.result)
      self.assertEqual(entry.reason, junit.REASON_NOT_READ)
      self.assertEqual(entry.per_worker, {})
      self.assertFalse(entry.is_partial)
    self.assertEqual(found.suites["decoupled"].nested_in, "cpu-unit")
    self.assertEqual(found.artifacts, [])

  def test_reason_code_is_stable(self):
    self.assertEqual(junit.REASON_NOT_READ, "not_read_no_token")

  def test_skipping_never_asks_for_artifacts(self):
    subject = self.make_skipping_tick()
    found = subject._harvest_tests(self.run, self.run["id"], self.jobs)
    self.read_run_tests.assert_not_called()
    self.assertEqual(set(found.suites), set(junit._suites_asked_about(derive.test_flavors(self.jobs))))
    self.assertEqual(subject.report.runs_tests_skipped, 1)
    self.assertEqual(subject.report.runs_tests_harvested, 0)

  def test_skipping_keeps_the_existing_reasons_to_read_nothing(self):
    subject = self.make_skipping_tick(hours_after_run=tick.ARTIFACT_LIFETIME_HOURS + 1)
    self.assertIsNone(subject._harvest_tests(self.run, self.run["id"], self.jobs))
    self.assertEqual(subject.report.runs_tests_expired, 1)

    superseded = dict(self.run, superseded=True)
    subject = self.make_skipping_tick()
    self.assertIsNone(subject._harvest_tests(superseded, self.run["id"], self.jobs))
    self.assertEqual(subject.report.runs_tests_superseded, 1)

    other_workflow = dict(self.run, path=".github/workflows/tpu_docker_images_pipeline.yml")
    self.assertIsNone(subject._harvest_tests(other_workflow, self.run["id"], self.jobs))
    self.assertEqual(subject.report.runs_tests_skipped, 0)
    self.read_run_tests.assert_not_called()

  def test_without_the_option_artifacts_are_still_read(self):
    subject = self.make_tick(BudgetClient(1000))
    subject.now = self.created + timedelta(hours=1)
    self.read_run_tests.side_effect = None
    self.read_run_tests.return_value = junit.RunTests(run_id=self.run["id"])
    subject._harvest_tests(self.run, self.run["id"], self.jobs)
    self.read_run_tests.assert_called_once()
    self.assertEqual(subject.report.runs_tests_skipped, 0)

  def test_a_collected_run_stores_suites_with_no_counts_and_no_tests(self):
    subject = self.make_skipping_tick()
    with mock.patch.object(tick.runs, "resolve_pull_request", return_value=None):
      with mock.patch.object(tick.runs, "get_jobs", return_value=self.jobs):
        subject._collect_run(self.run, self.run["id"])
    self.read_run_tests.assert_not_called()

    suites = subject.store.read(rows.KIND_SUITE, ["2026-09"])
    expected = set(junit._suites_asked_about(derive.test_flavors(self.jobs)))
    self.assertEqual({row["suite_id"] for row in suites}, expected)
    for row in suites:
      self.assertEqual(row["reason"], junit.REASON_NOT_READ)
      for count in ("collected", "skipped", "executed", "failed", "errored", "junit_seconds"):
        self.assertIsNone(row[count], f"{row['suite_id']}.{count} must be None, never zero")
      self.assertFalse(row["is_partial"])
    self.assertEqual(subject.store.read(rows.KIND_TEST, ["2026-09"]), [])
    self.assertEqual(subject.report.suites_written, len(expected))
    self.assertEqual(subject.report.tests_written, 0)
    self.assertEqual(subject.report.full_snapshots, 0)
    self.assertEqual(subject.report.suite_reasons, {junit.REASON_NOT_READ: len(expected)})
    self.assertEqual(subject.report.problems, [])
    self.assertEqual(subject.report.runs_collected, 1)

  def test_the_suites_view_keeps_each_suites_wall_clock_time(self):
    # The reason the empty suite rows are written at all: the view builds a suite's record from
    # its row, and takes the duration and the workers from the jobs.
    subject = self.make_skipping_tick()
    with mock.patch.object(tick.runs, "resolve_pull_request", return_value=None):
      with mock.patch.object(tick.runs, "get_jobs", return_value=self.jobs):
        subject._collect_run(self.run, self.run["id"])
    views.build_views(subject.store, Path(self.tmp_dir), date(2026, 9, 1), months=["2026-09"], uncollected=0)

    view = json.loads((Path(self.tmp_dir) / "views" / "suites-2026-09.json").read_text(encoding="utf-8"))
    records = views.from_columnar(view["tables"]["suites"])
    by_suite = {record["suite_id"]: record for record in records}
    self.assertIn("tpu-unit", by_suite)
    tpu_unit = by_suite["tpu-unit"]
    # Measured on these exact jobs: tpu-unit's wall clock is 1626 s (see the collector README).
    self.assertEqual(tpu_unit["duration_seconds"], 1626)
    self.assertIsNotNone(tpu_unit["workers_named"])
    self.assertIsNone(tpu_unit["collected"])
    self.assertEqual(tpu_unit["reason"], junit.REASON_NOT_READ)
    # The nested pass has no duration of its own, exactly as when artifacts are read.
    self.assertIsNone(by_suite["decoupled"]["duration_seconds"])


class DeferRunningTest(TickTestCase):
  """A run still queued or in progress costs no request with --defer-running."""

  QUEUED = {"id": 301, "status": "queued", "run_attempt": 1, "created_at": "2026-10-20T11:00:00Z"}

  def setUp(self) -> None:
    super().setUp()
    patch = mock.patch.object(tick.Tick, "_collect_run")
    self.collect_run = patch.start()
    self.addCleanup(patch.stop)

  def test_a_queued_run_stays_in_flight_without_a_request(self):
    client = BudgetClient(60)
    subject = self.make_tick(client, rate_floor=15, defer_running=True)
    self.assertTrue(subject._collect_one(dict(self.QUEUED)))
    self.collect_run.assert_not_called()
    self.assertEqual(client.budget_reads, 0, "not even the budget is read")
    self.assertIn((301, 1), subject.state.pending)
    self.assertEqual(subject.state.pending[(301, 1)].status, "queued")
    self.assertEqual(subject.report.runs_deferred, 1)
    self.assertEqual(subject.report.attempts_pending, 1)
    self.assertIn(301, subject._read_this_tick)

  def test_an_in_progress_re_run_is_kept_in_flight_under_its_newest_attempt(self):
    subject = self.make_tick(BudgetClient(60), rate_floor=15, defer_running=True)
    rerun = dict(self.QUEUED, status="in_progress", run_attempt=2)
    self.assertTrue(subject._collect_one(rerun))
    self.collect_run.assert_not_called()
    self.assertIn((301, 2), subject.state.pending)

  def test_without_the_option_a_queued_run_is_read(self):
    subject = self.make_tick(BudgetClient(60), rate_floor=15)
    self.assertTrue(subject._collect_one(dict(self.QUEUED)))
    self.collect_run.assert_called_once()
    self.assertEqual(subject.report.runs_deferred, 0)

  def test_a_completed_run_is_read(self):
    subject = self.make_tick(BudgetClient(60), rate_floor=15, defer_running=True)
    self.assertTrue(subject._collect_one(dict(self.QUEUED, status="completed")))
    self.collect_run.assert_called_once()
    self.assertEqual(subject.report.runs_deferred, 0)

  def test_an_attempt_waiting_too_long_is_read_so_it_is_stored_as_it_stands(self):
    subject = self.make_tick(BudgetClient(60), rate_floor=15, defer_running=True)
    subject._expired.add((301, 1))
    self.assertTrue(subject._collect_one(dict(self.QUEUED)))
    self.collect_run.assert_called_once()
    self.assertEqual(subject.report.runs_deferred, 0)

  def test_the_in_flight_pass_does_not_read_a_deferred_run_again(self):
    subject = self.make_tick(BudgetClient(60), rate_floor=15, defer_running=True)
    subject._pending_at_start = [301]
    self.assertTrue(subject._collect_one(dict(self.QUEUED)))
    with mock.patch.object(tick.runs, "get_run", side_effect=AssertionError("re-read a deferred run")):
      subject._collect_pending()
    self.collect_run.assert_not_called()

  def test_the_option_is_off_by_default_and_parsed(self):
    self.assertFalse(tick.Options(out=Path(self.tmp_dir)).defer_running)
    options = tick.build_options(tick.parse_args(["--out", self.tmp_dir, "--defer-running"]))
    self.assertTrue(options.defer_running)

  def test_the_report_counts_the_deferred_runs_only_with_the_option(self):
    on = tick.build_options(tick.parse_args(["--out", self.tmp_dir, "--defer-running"]))
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
      tick.print_report(tick.TickReport(runs_deferred=4), on)
    self.assertIn("4, kept in flight without a request (--defer-running)", printed.getvalue())
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
      tick.print_report(tick.TickReport(runs_deferred=4), tick.build_options(tick.parse_args(["--out", self.tmp_dir])))
    self.assertNotIn("Runs left running", printed.getvalue())


class MaxWaitTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    env_patch = mock.patch.dict(os.environ, {}, clear=False)
    env_patch.start()
    self.addCleanup(env_patch.stop)
    os.environ.pop("GITHUB_TOKEN", None)
    warned = mock.patch.object(github, "_TOKEN_WARNING_EMITTED", True)
    warned.start()
    self.addCleanup(warned.stop)
    self.sleeps: list[float] = []
    patch = mock.patch.object(github, "_sleep", side_effect=self.sleeps.append)
    patch.start()
    self.addCleanup(patch.stop)

  def make_client(
      self, responses: list[requests.Response], max_wait: float | None
  ) -> tuple[github.GitHubClient, mock.Mock]:
    """Returns a client with no token over a session that answers the scripted responses in order."""
    session = mock.Mock(spec=requests.Session)
    session.headers = {}
    session.request.side_effect = responses
    return github.GitHubClient("o", "r", token=None, session=session, max_wait=max_wait), session

  @staticmethod
  def response(status: int, headers: dict[str, str] | None = None, body: bytes = b"{}") -> requests.Response:
    """Builds a real requests.Response whose body is already read."""
    answer = requests.Response()
    answer.status_code = status
    answer.headers.update(headers or {})
    answer._content = body
    answer._content_consumed = True
    answer.raw = io.BytesIO(body)
    answer.url = "https://api.github.com/repos/o/r/pulls/1"
    return answer

  def test_default_trims_a_long_wait_to_one_hour(self):
    client, _ = self.make_client([], max_wait=None)
    self.assertEqual(client._affordable_delay(1800.0, 0.0), 1800.0)
    self.assertEqual(client._affordable_delay(5000.0, 0.0), github.MAX_TOTAL_RETRY_SLEEP_SECONDS)
    self.assertIsNone(client._affordable_delay(2.0, github.MAX_TOTAL_RETRY_SLEEP_SECONDS))

  def test_cap_allows_short_backoffs(self):
    client, _ = self.make_client([], max_wait=30)
    self.assertEqual(client._affordable_delay(2.0, 0.0), 2.0)
    self.assertEqual(client._affordable_delay(10.0, 15.0), 10.0)

  def test_cap_refuses_a_wait_that_does_not_fit(self):
    client, _ = self.make_client([], max_wait=30)
    self.assertIsNone(client._affordable_delay(1800.0, 0.0))
    self.assertIsNone(client._affordable_delay(20.0, 15.0))

  def test_negative_cap_is_refused(self):
    with self.assertRaises(ValueError):
      self.make_client([], max_wait=-1)

  def test_rate_limited_request_fails_at_once_instead_of_waiting_for_the_reset(self):
    reset = str(int(time.time()) + 1800)
    limited = self.response(403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset}, b'{"message": "rate limit"}')
    client, session = self.make_client([limited], max_wait=30)
    with self.assertRaises(github.GitHubError) as caught:
      client.get_json("pulls/1")
    self.assertEqual(caught.exception.status, 403)
    self.assertEqual(session.request.call_count, 1)
    self.assertEqual(self.sleeps, [])

  def test_server_errors_are_still_retried_under_the_cap(self):
    client, session = self.make_client([self.response(502), self.response(200, body=b'{"ok": true}')], max_wait=30)
    self.assertEqual(client.get_json("pulls/1"), {"ok": True})
    self.assertEqual(session.request.call_count, 2)
    self.assertEqual(self.sleeps, [github.BACKOFF_BASE_SECONDS])


class CommandLineTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp(prefix="ci-metrics-cli-")
    self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

  def options(self, *extra: str) -> tick.Options:
    """Parses a command line with the given extra arguments."""
    return tick.build_options(tick.parse_args(["--out", self.tmp_dir, *extra]))

  def test_defaults_keep_todays_behaviour(self):
    options = self.options()
    self.assertEqual(options.rate_floor, tick.RATE_LIMIT_FLOOR)
    self.assertFalse(options.skip_tests)
    self.assertIsNone(options.max_wait)
    self.assertEqual(options.merge_checks, tick.MERGE_CHECKS_EACH)

  def test_no_token_options_are_parsed(self):
    options = self.options("--rate-floor", "15", "--skip-tests", "--max-wait", "30", "--merge-checks", "list")
    self.assertEqual(options.rate_floor, 15)
    self.assertTrue(options.skip_tests)
    self.assertEqual(options.max_wait, 30.0)
    self.assertEqual(options.merge_checks, tick.MERGE_CHECKS_LIST)

  def test_impossible_values_are_usage_errors(self):
    for extra in (
        ("--rate-floor", "-1"),
        ("--max-wait", "-1"),
        ("--max-wait", "3601"),
        ("--max-wait", "nan"),
        ("--max-wait", "inf"),
    ):
      with self.subTest(extra=extra):
        with self.assertRaises(tick.UsageError):
          self.options(*extra)

  def test_unknown_merge_mode_is_refused_by_the_parser(self):
    with contextlib.redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit):
        self.options("--merge-checks", "sometimes")

  def test_main_hands_the_cap_to_the_client(self):
    with mock.patch.object(tick.github, "GitHubClient") as client_class:
      with mock.patch.object(tick.Tick, "collect"), mock.patch.object(tick, "prune_old_files"):
        with mock.patch.object(tick, "rebuild_views"), contextlib.redirect_stdout(io.StringIO()):
          code = tick.main(["--out", self.tmp_dir, "--max-wait", "30", "--dry-run"])
    self.assertEqual(code, tick.EXIT_OK)
    self.assertEqual(client_class.call_args.kwargs["max_wait"], 30.0)

  def test_report_names_the_skipped_runs_and_the_pages_read(self):
    report = tick.TickReport(runs_tests_skipped=3, merge_pages_read=2)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
      tick.print_report(report, self.options("--skip-tests", "--merge-checks", "list"))
    self.assertIn("3 not read (--skip-tests)", printed.getvalue())
    self.assertIn("2 page(s) of closed pull requests read", printed.getvalue())

  def test_report_is_unchanged_without_the_options(self):
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
      tick.print_report(tick.TickReport(), self.options())
    self.assertNotIn("--skip-tests", printed.getvalue())
    self.assertNotIn("page(s) of closed pull requests", printed.getvalue())
    self.assertNotIn("API budget", printed.getvalue())

  def test_report_prints_the_budget_readings(self):
    report = tick.TickReport(budget_first="60 of 60", budget_last="13 of 60")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
      tick.print_report(report, self.options())
    self.assertIn("API budget                 60 of 60 left at the first check, 13 of 60 at the last", printed.getvalue())


class MergeListTest(TickTestCase):

  def setUp(self) -> None:
    super().setUp()
    patch = mock.patch.object(tick.Tick, "_store_merge", return_value=True)
    self.store_merge = patch.start()
    self.addCleanup(patch.stop)

  def run_checks(self, client: ListingClient, waiting: dict[int, int]) -> tick.Tick:
    """Runs the list-mode merge checks over the given waiting pull requests."""
    subject = self.make_tick(client, rate_floor=15, merge_checks=tick.MERGE_CHECKS_LIST)
    with mock.patch.object(tick.Tick, "_open_pull_requests", return_value=waiting):
      subject._refresh_merges_from_list()
    return subject

  def test_collect_uses_the_list_mode_when_asked(self):
    subject = self.make_tick(BudgetClient(60), rate_floor=15, merge_checks=tick.MERGE_CHECKS_LIST)
    with mock.patch.object(tick.Tick, "_list_slice", return_value=[]):
      with mock.patch.object(tick.Tick, "_refresh_merges") as each, mock.patch.object(
          tick.Tick, "_refresh_merges_from_list"
      ) as listed:
        subject.collect()
    listed.assert_called_once()
    each.assert_not_called()

  def test_collect_keeps_the_each_mode_by_default(self):
    subject = self.make_tick(BudgetClient(1000))
    with mock.patch.object(tick.Tick, "_list_slice", return_value=[]):
      with mock.patch.object(tick.Tick, "_refresh_merges") as each, mock.patch.object(
          tick.Tick, "_refresh_merges_from_list"
      ) as listed:
        subject.collect()
    each.assert_called_once()
    listed.assert_not_called()

  def test_nothing_waiting_costs_no_request(self):
    client = ListingClient(AssertionError("listed with nothing waiting"))
    subject = self.run_checks(client, {})
    self.assertEqual(client.listing_calls, [])
    self.assertEqual(subject.report.prs_watched, 0)

  def test_asks_for_closed_pull_requests_newest_update_first(self):
    client = ListingClient([[closed_pull(5, NOW - timedelta(hours=1), merged=True)]])
    self.run_checks(client, {5: 905})
    self.assertEqual(
        client.listing_calls,
        [
            (
                "pulls",
                "pulls",
                github.PER_PAGE,
                {"state": "closed", "sort": "updated", "direction": "desc", "page": 1},
            )
        ],
    )

  def test_learns_merges_across_pages_and_stops_at_a_short_page(self):
    page_one = full_page(1000, NOW - timedelta(minutes=5))
    page_one[10] = closed_pull(101, NOW - timedelta(hours=2), merged=True)
    page_one[20] = closed_pull(102, NOW - timedelta(hours=3), merged=False)
    page_two = [closed_pull(103, NOW - timedelta(days=2), merged=True)]
    client = ListingClient([page_one, page_two, AssertionError("read past a short page")])
    subject = self.run_checks(client, {101: 9101, 102: 9102, 103: 9103, 104: 9104})

    self.assertEqual(len(client.listing_calls), 2)
    self.assertEqual([call.args[0] for call in self.store_merge.call_args_list], [9101, 9103])
    self.assertEqual([call.args[1]["number"] for call in self.store_merge.call_args_list], [101, 103])
    self.assertEqual(subject.report.merges_learned, 2)
    self.assertEqual(subject.report.merge_pages_read, 2)
    self.assertEqual(subject.report.prs_watched, 4)
    self.assertEqual(subject.report.problems, [])

  def test_stops_at_the_watch_window(self):
    # A full page whose oldest update is older than the watch window: nothing further back can be ours.
    old = NOW - timedelta(days=tick.MERGE_WATCH_DAYS, minutes=1)
    page_one = full_page(1000, old + timedelta(minutes=500))
    client = ListingClient([page_one, AssertionError("read past the watch window")])
    self.run_checks(client, {7: 907})
    self.assertEqual(len(client.listing_calls), 1)
    self.store_merge.assert_not_called()

  def test_stops_once_every_waiting_pull_request_is_found(self):
    page_one = full_page(1000, NOW - timedelta(minutes=5))
    page_one[0] = closed_pull(7, NOW - timedelta(minutes=1), merged=True)
    client = ListingClient([page_one, AssertionError("read after everything was found")])
    self.run_checks(client, {7: 907})
    self.assertEqual(len(client.listing_calls), 1)
    self.store_merge.assert_called_once()

  def test_reads_at_most_the_page_cap_and_says_so(self):
    pages = [
        full_page(1000 + 100 * index, NOW - timedelta(minutes=index), step=timedelta(seconds=1)) for index in range(10)
    ]
    client = ListingClient(pages)
    printed = io.StringIO()
    with contextlib.redirect_stderr(printed):
      self.run_checks(client, {7: 907})
    self.assertEqual(len(client.listing_calls), tick.MERGE_LIST_MAX_PAGES)
    self.assertIn("did not reach back", printed.getvalue())

  def test_low_budget_stops_between_pages(self):
    pages = [full_page(1000, NOW - timedelta(minutes=5)), full_page(2000, NOW - timedelta(hours=20))]
    client = ListingClient(pages, readings=10)
    subject = self.run_checks(client, {7: 907})
    self.assertEqual(len(client.listing_calls), 1)
    self.assertIn("API budget ran low", subject.report.stopped_because)
    self.store_merge.assert_not_called()

  def test_budget_is_checked_before_each_stored_merge(self):
    page = [closed_pull(number, NOW - timedelta(hours=number), merged=True) for number in (1, 2, 3)]
    # The first reading allows the first merge to be stored; the second stops the rest.
    client = ListingClient([page], readings=[60, 10])
    subject = self.run_checks(client, {1: 901, 2: 902, 3: 903})
    self.store_merge.assert_called_once()
    self.assertEqual(subject.report.merges_learned, 1)
    self.assertIn("API budget ran low", subject.report.stopped_because)

  def test_closed_unmerged_and_unknown_entries_are_ignored(self):
    page = [
        closed_pull(1, NOW - timedelta(hours=1), merged=False),
        closed_pull(999, NOW - timedelta(hours=2), merged=True),
        "not an object",
        {"number": "x", "merged_at": iso(NOW)},
    ]
    client = ListingClient([page])
    subject = self.run_checks(client, {1: 901})
    self.store_merge.assert_not_called()
    self.assertEqual(subject.report.merges_learned, 0)

  def test_a_listing_failure_is_reported_and_stores_nothing(self):
    client = ListingClient(github.GitHubError("GET pulls answered 502", status=502))
    with contextlib.redirect_stderr(io.StringIO()):
      subject = self.run_checks(client, {7: 907})
    self.store_merge.assert_not_called()
    self.assertEqual(len(subject.report.problems), 1)
    self.assertIn("closed pull requests could not be listed", subject.report.problems[0])

  def test_a_listed_pull_request_fills_the_run_row_like_a_full_one(self):
    # `_store_merge` hands the listing entry to rows.run_row, which must find every field it copies.
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    pull = closed_pull(5070, NOW, merged=True)
    row = rows.run_row(run, pull)
    self.assertEqual(row.pr_number, 5070)
    self.assertEqual(row.pr_merged_at, pull["merged_at"])
    self.assertEqual(row.pr_base_ref, "main")
    self.assertEqual(row.pr_user, "someone")
    self.assertEqual(row.pr_head_sha, "head5070")


if __name__ == "__main__":
  unittest.main()
