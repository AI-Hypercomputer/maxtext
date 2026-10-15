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

"""One collector tick: read what is new, store its rows, rebuild the views."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import requests

# The collector package's parent, so `python3 tools/ci_metrics/collector/tick.py --out DIR`
# works from anywhere.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import derive
from collector import github
from collector import junit
from collector import rows
from collector import runs
from collector import store as store_module
from collector import views

DEFAULT_REPO = "AI-Hypercomputer/maxtext"

# GitHub deletes test artifacts after about a day. A run older than this is not asked about
# its artifacts at all: the answer is already known, and the request would be spent to hear
# it. The gap is counted so the report can state it.
ARTIFACT_LIFETIME_HOURS = 24

# A re-run does not change a run's id or its creation time, so a watermark alone would never
# bring it back. Every tick therefore re-lists the last couple of days as well; a run already
# collected is skipped without a single extra request unless its attempt count has grown.
RECHECK_DAYS = 2

# A run is collected minutes after it starts, while its pull request is still open, so the
# merge time the runs view needs is not known yet. Runs of open pull requests are re-asked
# for this long, and then left alone.
MERGE_WATCH_DAYS = 14

# The tick stops cleanly while this many requests are still left. GITHUB_TOKEN inside Actions allows
# 1000 an hour per repository, shared by every workflow in it, so the others keep some.
RATE_LIMIT_FLOOR = 300

# The merge checks cost a request per waiting pull request, so they re-check the budget this often.
MERGE_BUDGET_CHECK_EVERY = 25

DEFAULT_BACKFILL_DAYS = 30

EXIT_OK = 0
EXIT_DATA_LOST = 1
EXIT_USAGE = 2


class TickError(RuntimeError):
  """The tick could not finish: a store could not be written, or a fetch could not be made."""


class UsageError(TickError):
  """The tick was asked for something impossible, so repeating it would fail the same way."""


def _warn(message: str) -> None:
  """Prints a warning on stderr, so stdout stays a clean report."""
  print(message, file=sys.stderr)


def utc_now() -> datetime:
  """Returns the current moment in UTC, with a timezone attached."""
  return datetime.now(timezone.utc)


def iso_utc(moment: datetime) -> str:
  """Formats a moment the way every stored timestamp is written."""
  return runs.as_utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_day(value: str, end_of_day: bool = False) -> datetime:
  """Turns a YYYY-MM-DD argument into a UTC moment."""
  try:
    day = datetime.strptime(value.strip(), "%Y-%m-%d")
  except ValueError as error:
    raise UsageError(f"expected a date as YYYY-MM-DD, got {value!r}") from error
  if end_of_day:
    day = day.replace(hour=23, minute=59, second=59)
  return day.replace(tzinfo=timezone.utc)


def split_repo(value: str) -> tuple[str, str]:
  """Splits an "owner/name" argument."""
  owner, _, name = value.partition("/")
  if not owner or not name or "/" in name:
    raise UsageError(f"--repo must be owner/name, got {value!r}")
  return owner, name


class CountingSession(requests.Session):
  """A requests session that counts what a tick spent, split by host."""

  def __init__(self) -> None:
    """Builds an empty session with both counters at zero."""
    super().__init__()
    self.api_requests = 0
    self.download_requests = 0

  def request(self, method: str | bytes, url: str | bytes, *args: Any, **kwargs: Any) -> requests.Response:
    """Counts one request and sends it."""
    if urlsplit(str(url)).netloc == github.API_HOST:
      self.api_requests += 1
    else:
      self.download_requests += 1
    return super().request(method, url, *args, **kwargs)


@dataclass(frozen=True)
class Options:
  """One tick's settings, all of them from the command line."""

  out: Path
  repo: str = DEFAULT_REPO
  since: datetime | None = None
  until: datetime | None = None
  backfill_days: int = DEFAULT_BACKFILL_DAYS
  dry_run: bool = False
  max_runs: int | None = None


@dataclass
class TickReport:
  """Everything the tick has to be able to say about itself when it is done."""

  mode: str = "tick"
  window_since: str | None = None
  window_until: str | None = None
  runs_seen: int = 0
  runs_collected: int = 0
  runs_already_stored: int = 0
  runs_failed: int = 0
  attempts_written: int = 0
  attempts_pending: int = 0
  attempts_incomplete: int = 0
  jobs_written: int = 0
  suites_written: int = 0
  tests_written: int = 0
  tests_seen: int = 0
  rescues_written: int = 0
  full_snapshots: int = 0
  runs_tests_harvested: int = 0
  runs_tests_expired: int = 0
  runs_tests_superseded: int = 0
  runs_tests_failed: int = 0
  prs_watched: int = 0
  merges_learned: int = 0
  views_written: int = 0
  views_unchanged: int = 0
  pr_files_written: int = 0
  data_files_pruned: int = 0
  view_files_pruned: int = 0
  months_touched: set[str] = field(default_factory=set)
  months_rebuilt: list[str] = field(default_factory=list)
  suite_reasons: dict[str, int] = field(default_factory=dict)
  suite_reason_examples: dict[str, list[str]] = field(default_factory=dict)
  partial_suites: list[str] = field(default_factory=list)
  api_requests: int = 0
  download_requests: int = 0
  stopped_because: str | None = None
  problems: list[str] = field(default_factory=list)

  def note_suite(self, run_id: int, suite_id: str, reason: str | None, is_partial: bool) -> None:
    """Records why a suite has no numbers, or that it only has some of them."""
    if reason:
      self.suite_reasons[reason] = self.suite_reasons.get(reason, 0) + 1
      examples = self.suite_reason_examples.setdefault(reason, [])
      if len(examples) < 5 and suite_id not in examples:
        examples.append(suite_id)
    if is_partial:
      self.partial_suites.append(f"{suite_id} in run {run_id}")

  def note_problem(self, message: str) -> None:
    """Records something that lost data, which is what makes the exit code non-zero."""
    self.problems.append(message)
    _warn(f"WARNING: {message}")

  @property
  def rows_written(self) -> int:
    """How many rows of every kind this tick appended."""
    return self.attempts_written + self.jobs_written + self.suites_written + self.tests_written + self.rescues_written

  @property
  def changed(self) -> bool:
    """True when the tick learned something, rather than only restamping meta.json."""
    return bool(self.rows_written or self.pr_files_written or self.months_rebuilt)


class Tick:
  """Collects one window of runs into the store."""

  def __init__(
      self,
      options: Options,
      client: github.GitHubClient,
      row_store: store_module.Store,
      state: store_module.State,
      report: TickReport,
  ) -> None:
    """Builds a tick."""
    self.options = options
    self.client = client
    self.store = row_store
    self.state = state
    self.report = report
    self.now = utc_now()
    # One timestamp for every row this tick writes. "Readers take the last row per key"
    # depends on that ordering, and per-row stamps would put one run's rows seconds apart.
    self.collected_at = iso_utc(self.now)
    self._expired = {(entry.run_id, entry.attempt) for entry in state.expired_pending(self.now)}
    # Taken now, before anything is collected. `state.pending` is the same object the tick
    # mutates all tick long, so reading it later would hand back the attempts this very tick
    # has just put on the list - and re-read them, spend the requests again and count them
    # twice in the report.
    self._pending_at_start = sorted({run_id for run_id, _ in state.pending})
    self._read_this_tick: set[int] = set()
    self._known_names_cache: set[tuple[str, str, str]] | None = None
    self._snapshot_days_cache: dict[str, int] | None = None
    self._runs_processed = 0

  # ---------------------------------------------------------------- the window

  def window(self) -> tuple[datetime, datetime | None, bool]:
    """Decides which window to ask the API for."""
    until = self.options.until
    if self.options.since is not None:
      return self.options.since, until, self.state.watermark_created_at is None

    watermark = runs.parse_timestamp(self.state.watermark_created_at)
    if watermark is None:
      return self.now - timedelta(days=max(self.options.backfill_days, 1)), until, True
    # The watermark alone would never see a re-run: it keeps the run's id and creation time
    # and only adds an attempt. Re-listing the last couple of days costs one listing and
    # catches it; runs already collected are skipped without a single extra request.
    return min(watermark, self.now - timedelta(days=RECHECK_DAYS)), until, False

  def collect(self) -> None:
    """Walks the window, oldest week first, collects every run in it, then chases merges."""
    since, until, backfill = self.window()
    self.report.mode = "backfill" if backfill else "tick"
    self.report.window_since = iso_utc(since)
    self.report.window_until = iso_utc(until) if until is not None else None

    end = until if until is not None else self.now
    if end < since:
      raise UsageError(
          f"the window ends before it starts: since={iso_utc(since)} until={iso_utc(end)}. "
          "The store already holds runs newer than --until."
      )

    for index, (slice_since, slice_until) in enumerate(runs.split_window(since, end, days=runs.BACKFILL_WINDOW_DAYS)):
      if index and not self._budget_allows():
        return
      for run in self._list_slice(slice_since, slice_until):
        if not self._collect_one(run):
          return
    self._collect_pending()
    if self._budget_allows():
      self._refresh_merges()

  def _budget_allows(self) -> bool:
    """Checks the API budget before more requests are spent; False means stop cleanly."""
    # The headers of the latest answer can show fewer requests left than /rate_limit does (a
    # real run saw 267 used against 223), so the lower reading decides. Read the header count
    # first: rate_limit() overwrites it.
    seen = self.client.remaining_hint
    try:
      status = self.client.rate_limit()
    except github.GitHubError as error:
      if seen is None:
        _warn(f"WARNING: the rate-limit endpoint could not be read ({error}); carrying on.")
        return True
      remaining, limit = seen, "?"
    else:
      remaining = status["remaining"] if seen is None else min(seen, status["remaining"])
      limit = status["limit"]
    if remaining >= RATE_LIMIT_FLOOR:
      return True
    self.report.stopped_because = (
        f"the API budget ran low: {remaining} of {limit} requests left, "
        f"floor is {RATE_LIMIT_FLOOR}. The next tick resumes from the watermark."
    )
    return False

  def _list_slice(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
    """Lists one week of runs, oldest first, with the superseded ones marked."""
    try:
      listed = runs.list_runs(self.client, since, until)
    except (github.GitHubError, runs.RunsError) as error:
      self.report.note_problem(f"runs created {iso_utc(since)}..{iso_utc(until)} could not be listed: {error}")
      return []
    # Oldest first: the watermark moves only past runs this tick has handled, so a stop part-way
    # (--max-runs, a low budget) never skips one.
    return list(reversed(runs.mark_superseded(listed)))

  # ------------------------------------------------------------------ one run

  def _collect_one(self, listed: dict[str, Any]) -> bool:
    """Collects one run."""
    if self.options.max_runs is not None and self._runs_processed >= self.options.max_runs:
      self.report.stopped_because = f"--max-runs {self.options.max_runs} reached; the next tick continues."
      return False

    try:
      run_id = runs.run_id_of(listed)
    except runs.RunsError as error:
      self.report.note_problem(f"a listed run has no usable id ({error}); it was not collected.")
      return True

    self.report.runs_seen += 1
    if self._already_stored(listed, run_id):
      self._read_this_tick.add(run_id)
      self.state.note_run(run_id, listed.get("created_at"))
      self.report.runs_already_stored += 1
      return True

    # A run already stored costs no request, so only a run that will be read checks the budget.
    if not self._budget_allows():
      return False

    self._runs_processed += 1
    self._read_this_tick.add(run_id)
    try:
      self._collect_run(listed, run_id)
    except (github.GitHubError, runs.RunsError, rows.RowError, store_module.StoreError, junit.JUnitError) as error:
      self._failed(run_id, listed, error)
    return True

  def _failed(self, run_id: int, listed: dict[str, Any], error: Exception) -> None:
    """Records a run that could not be collected, and arranges for it to be tried again."""
    self.report.runs_failed += 1
    self.state.mark_pending(
        run_id,
        int(listed.get("run_attempt") or 1),
        created_at=listed.get("created_at"),
        status=str(listed.get("status") or "unread"),
    )
    self.report.note_problem(f"run {run_id} could not be collected ({error}); it will be tried again next tick.")

  def _already_stored(self, listed: dict[str, Any], run_id: int) -> bool:
    """True when every attempt of a finished run is already in the store."""
    if listed.get("status") != store_module.COMPLETED_STATUS:
      return False
    attempts = int(listed.get("run_attempt") or 1)
    return all(self.state.is_collected(run_id, number) for number in range(1, attempts + 1))

  def _attempt_payloads(self, listed: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns one run payload per attempt, oldest attempt first."""
    if listed.get("status") == store_module.COMPLETED_STATUS and int(listed.get("run_attempt") or 1) == 1:
      payloads = [listed]
    else:
      payloads = runs.list_attempts(self.client, listed)
    flag = listed.get(runs.SUPERSEDED_FIELD)
    marked: list[dict[str, Any]] = []
    for payload in payloads:
      copy = dict(payload)
      if flag is not None:
        copy[runs.SUPERSEDED_FIELD] = flag
      marked.append(copy)
    return marked

  def _collect_run(self, listed: dict[str, Any], run_id: int) -> None:
    """Reads one run whole - attempts, jobs, pull request, tests - and stores its rows."""
    month = store_module.month_for_run(listed)
    attempts = self._attempt_payloads(listed)
    pull = runs.resolve_pull_request(self.client, listed)

    attempts_jobs: dict[int, list[dict[str, Any]]] = {}
    to_write: list[dict[str, Any]] = []
    for payload in attempts:
      number = int(payload.get("run_attempt") or 1)
      if payload.get("status") != store_module.COMPLETED_STATUS:
        self._note_unfinished(payload, run_id, number, month, pull)
        continue
      attempts_jobs[number] = runs.get_jobs(self.client, run_id, number)
      if not self.state.has_attempt(run_id, number):
        to_write.append(payload)

    if not attempts_jobs or not to_write:
      return

    newest = max(attempts_jobs)
    newest_payload = attempts[-1]
    for payload in attempts:
      if int(payload.get("run_attempt") or 1) == newest:
        newest_payload = payload

    for payload in to_write:
      number = int(payload.get("run_attempt") or 1)
      jobs = attempts_jobs.get(number, [])
      self._append(month, rows.KIND_RUN, [rows.run_row(payload, pull, collected_at=self.collected_at)])
      self._append(month, rows.KIND_JOB, [rows.job_row(payload, job, collected_at=self.collected_at) for job in jobs])
      self.report.attempts_written += 1
      self.report.jobs_written += len(jobs)

    run_tests = self._harvest_tests(listed, run_id, attempts_jobs[newest])
    if run_tests is not None:
      self._append(month, rows.KIND_SUITE, self._suite_rows(newest_payload, run_tests))
      self._append(month, rows.KIND_TEST, self._kept_test_rows(newest_payload, run_tests))

    # Also harvest tests for earlier failed attempts so rescue_tests can name
    # the tests that failed.  The newest attempt's results are already stored
    # above; this pass adds the ones that broke on an earlier try.  Only runs
    # with more than one attempt (i.e. rescues) trigger the extra download.
    if len(attempts_jobs) > 1:
      for attempt_num in sorted(attempts_jobs):
        if attempt_num == newest:
          continue  # already harvested above
        attempt_payload = None
        for payload in to_write:
          if int(payload.get("run_attempt") or 1) == attempt_num:
            attempt_payload = payload
            break
        if attempt_payload is None:
          continue
        failed_tests = self._harvest_tests(listed, run_id, attempts_jobs[attempt_num])
        if failed_tests is not None:
          self._append(month, rows.KIND_TEST, self._kept_test_rows(attempt_payload, failed_tests))

    rescues = rows.rescue_rows(newest_payload, attempts_jobs, collected_at=self.collected_at)
    rescues += rows.failed_never_rescued_rows(newest_payload, attempts_jobs, collected_at=self.collected_at)
    self._append(month, rows.KIND_RESCUE, rescues)
    self.report.rescues_written += len(rescues)

    for payload in to_write:
      self.state.mark_collected(run_id, int(payload.get("run_attempt") or 1), payload.get("created_at"))
    self.report.runs_collected += 1
    self.report.months_touched.add(month)

  def _note_unfinished(
      self,
      payload: dict[str, Any],
      run_id: int,
      number: int,
      month: str,
      pull: dict[str, Any] | None,
  ) -> None:
    """Remembers an attempt that has not finished, or gives up on one that never will."""
    if (run_id, number) not in self._expired:
      if self.state.mark_pending(run_id, number, created_at=payload.get("created_at"), status=payload.get("status")):
        self.report.attempts_pending += 1
      return

    jobs = runs.get_jobs(self.client, run_id, number)
    self._append(month, rows.KIND_RUN, [rows.run_row(payload, pull, collected_at=self.collected_at)])
    self._append(month, rows.KIND_JOB, [rows.job_row(payload, job, collected_at=self.collected_at) for job in jobs])
    self.state.mark_incomplete(run_id, number, payload.get("created_at"))
    self.report.attempts_written += 1
    self.report.attempts_incomplete += 1
    self.report.jobs_written += len(jobs)
    self.report.months_touched.add(month)
    _warn(
        f"WARNING: run {run_id} attempt {number} has been {payload.get('status')} for over "
        f"{store_module.PENDING_MAX_AGE_HOURS}h; storing it as it stands and no longer waiting for it."
    )

  def _collect_pending(self) -> None:
    """Looks again at the attempts that were still running when a PAST tick saw them."""
    for run_id in self._pending_at_start:
      if self.options.max_runs is not None and self._runs_processed >= self.options.max_runs:
        return
      if run_id in self._read_this_tick:
        continue
      if not self._budget_allows():
        return
      try:
        listed = runs.get_run(self.client, int(run_id))
      except github.GitHubError as error:
        if error.status == 404:
          for identity in [key for key in list(self.state.pending) if key[0] == int(run_id)]:
            self.state.drop_pending(*identity)
          _warn(f"WARNING: run {run_id} was in flight but GitHub no longer serves it; forgetting it.")
          continue
        self.report.note_problem(f"run {run_id} was in flight and could not be re-read: {error}")
        continue
      self._runs_processed += 1
      self._read_this_tick.add(int(run_id))
      try:
        self._collect_run(listed, int(run_id))
      except (github.GitHubError, runs.RunsError, rows.RowError, store_module.StoreError, junit.JUnitError) as error:
        self._failed(int(run_id), listed, error)

  # -------------------------------------------------------------- merge chasing

  def _refresh_merges(self) -> None:
    """Asks the pull requests that were open when their run was read whether they merged."""
    waiting = self._open_pull_requests()
    self.report.prs_watched = len(waiting)
    for index, number in enumerate(sorted(waiting)):
      if index and index % MERGE_BUDGET_CHECK_EVERY == 0 and not self._budget_allows():
        return
      try:
        pull = runs.get_pull_request(self.client, number)
      except github.GitHubError as error:
        self.report.note_problem(f"pull request #{number} could not be re-read ({error}); still waiting on it.")
        continue
      if pull is None or pull.get("state") == "open" or not pull.get("merged_at"):
        # Still open, or closed without merging. Its runs stay in the store; no chart draws
        # them, and a closed one simply ages out of the window.
        continue
      if self._store_merge(waiting[number], pull):
        self.report.merges_learned += 1

  def _open_pull_requests(self) -> dict[int, int]:
    """Finds the stored runs whose pull request had not merged when they were read."""
    cutoff = iso_utc(self.now - timedelta(days=MERGE_WATCH_DAYS))
    months = sorted({store_module.month_key(cutoff), store_module.month_key(self.collected_at)})
    waiting: dict[int, int] = {}
    try:
      stored = self.store.read(rows.KIND_RUN, months)
    except store_module.StoreError as error:
      _warn(f"WARNING: the stored runs could not be read for their merge state ({error}).")
      return {}
    for payload in stored:
      number = payload.get("pr_number")
      if payload.get("event") != runs.EVENT_PULL_REQUEST or number is None or payload.get("pr_merged_at"):
        continue
      created = payload.get("created_at")
      if not created or str(created) < cutoff:
        continue
      run_id = int(payload.get("run_id") or 0)
      if run_id > waiting.get(int(number), 0):
        waiting[int(number)] = run_id
    return waiting

  def _store_merge(self, run_id: int, pull: dict[str, Any]) -> bool:
    """Appends a corrected run row for a run whose pull request has now merged."""
    if not run_id:
      return False
    try:
      listed = runs.get_run(self.client, run_id)
    except github.GitHubError as error:
      self.report.note_problem(f"run {run_id} could not be re-read for its merge time ({error}).")
      return False

    attempt = int(listed.get("run_attempt") or 1)
    if not self.state.has_attempt(run_id, attempt):
      # The run gained an attempt after it was collected. Collect that attempt properly
      # rather than writing a run row with no jobs beside it.
      try:
        self._collect_run(listed, run_id)
      except (github.GitHubError, runs.RunsError, rows.RowError, store_module.StoreError, junit.JUnitError) as error:
        self._failed(run_id, listed, error)
        return False
      return True

    month = store_module.month_for_run(listed)
    self._append(
        month,
        rows.KIND_RUN,
        [rows.run_row(listed, pull, collected_at=self.collected_at)],
        correction=True,
    )
    self.report.attempts_written += 1
    self.report.months_touched.add(month)
    return True

  # ------------------------------------------------------------------- tests

  def _harvest_tests(self, listed: dict[str, Any], run_id: int, jobs: Sequence[dict[str, Any]]) -> junit.RunTests | None:
    """Downloads and parses the run's JUnit artifacts, or says why it did not."""
    if listed.get("path") != runs.CI_PIPELINE_PATH:
      return None
    if listed.get(runs.SUPERSEDED_FIELD):
      self.report.runs_tests_superseded += 1
      return None

    created = runs.parse_timestamp(listed.get("created_at"))
    if created is not None and self.now - runs.as_utc(created) > timedelta(hours=ARTIFACT_LIFETIME_HOURS):
      self.report.runs_tests_expired += 1
      return None

    flavors = derive.test_flavors(jobs)
    if not flavors:
      return None

    try:
      found = junit.read_run_tests(self.client, run_id, flavors=flavors)
    except (junit.JUnitError, github.GitHubError) as error:
      self.report.runs_tests_failed += 1
      self.report.note_problem(f"run {run_id}: the test artifacts could not be read ({error}); its tests are lost.")
      return None
    self.report.runs_tests_harvested += 1
    return found

  def _suite_rows(self, run_payload: dict[str, Any], run_tests: junit.RunTests) -> list[rows.SuiteRow]:
    """Builds the per-flavor totals row for every suite of one attempt."""
    built: list[rows.SuiteRow] = []
    for suite_id, entry in sorted(run_tests.suites.items()):
      row = rows.suite_row(run_payload, entry, collected_at=self.collected_at)
      self.report.note_suite(row.run_id, suite_id, row.reason, row.is_partial)
      built.append(row)
    self.report.suites_written += len(built)
    return built

  def _is_full_snapshot(self, run_payload: dict[str, Any]) -> bool:
    """True when this run is the day's full per-test snapshot."""
    if run_payload.get("event") != runs.EVENT_SCHEDULE or run_payload.get("head_branch") != "main":
      return False
    day = str(run_payload.get("created_at") or "")[:10]
    if not day:
      return False
    run_id = int(run_payload.get("id") or 0)
    claimed = self._snapshot_days().get(day)
    if claimed is not None and claimed != run_id:
      return False
    self._snapshot_days()[day] = run_id
    return True

  def _snapshot_days(self) -> dict[str, int]:
    """Returns which day already has a full snapshot, and the run that gave it."""
    if self._snapshot_days_cache is None:
      claimed: dict[str, int] = {}
      month = store_module.month_key(self.collected_at)
      try:
        stored = self.store.read(rows.KIND_RUN, [month])
      except store_module.StoreError as error:
        # Not a `note_problem`, so the exit code stays 0: with no answer, no day looks
        # claimed and this run keeps its full test list. The failure over-keeps rows, it
        # never drops them.
        _warn(f"WARNING: the stored runs could not be read for the daily snapshot ({error}).")
        stored = []
      for payload in stored:
        if payload.get("event") != runs.EVENT_SCHEDULE or payload.get("head_branch") != "main":
          continue
        day = str(payload.get("created_at") or "")[:10]
        run_id = int(payload.get("run_id") or 0)
        if day and run_id and claimed.get(day, run_id) >= run_id:
          claimed[day] = run_id
      self._snapshot_days_cache = claimed
    return self._snapshot_days_cache

  def _known_names(self) -> set[tuple[str, str, str]]:
    """Returns the test names the store already holds, so a new test can be recognised."""
    if self._known_names_cache is None:
      found: set[tuple[str, str, str]] = set()
      months = sorted(
          {store_module.month_key(iso_utc(self.now - timedelta(days=45))), store_module.month_key(self.collected_at)}
      )
      for month in months:
        try:
          for row in self.store.read_month(month, [rows.KIND_TEST]):
            found.add((row.suite_id, row.classname, row.name))
        except (store_module.StoreError, rows.RowError) as error:
          # Not a `note_problem` either, and for the same reason: with no names to compare
          # against, every test counts as new and is kept. Over-keeping, never losing.
          _warn(f"WARNING: {month}'s test rows could not be scanned for names ({error}); every test will look new.")
      self._known_names_cache = found
    return self._known_names_cache

  def _kept_test_rows(self, run_payload: dict[str, Any], run_tests: junit.RunTests) -> list[rows.TestRow]:
    """Builds this run's test rows and keeps the two tiers the store is allowed to hold."""
    built: list[rows.TestRow] = []
    for suite_id, entry in sorted(run_tests.suites.items()):
      for worker, result in sorted(entry.per_worker.items()):
        built.extend(
            rows.test_rows(
                run_payload,
                entry.nested_in or suite_id,
                worker,
                result,
                suite_id=suite_id,
                nested_in=entry.nested_in,
                suite_partial=entry.is_partial,
                collected_at=self.collected_at,
            )
        )
    self.report.tests_seen += len(built)

    if self._is_full_snapshot(run_payload):
      self.report.full_snapshots += 1
      self.report.tests_written += len(built)
      return built

    known = self._known_names()
    by_suite: dict[str, list[rows.TestRow]] = {}
    kept: set[str] = set()
    for row in built:
      by_suite.setdefault(row.suite_id, []).append(row)
      if row.status in (junit.STATUS_FAILED, junit.STATUS_ERROR):
        kept.add(row.key())
      elif (row.suite_id, row.classname, row.name) not in known:
        kept.add(row.key())
    # Slowest per suite, not per flavor: the nested decoupled pass carries flavor "cpu-unit"
    # and would otherwise compete with cpu-unit's own tests for the same 25 places.
    for group in by_suite.values():
      for row in derive.slowest_tests(group, per_flavor=derive.DEFAULT_SLOWEST_PER_FLAVOR):
        kept.add(row.key())

    ordered = [row for row in built if row.key() in kept]
    self.report.tests_written += len(ordered)
    return ordered

  # ------------------------------------------------------------------ writing

  def _append(self, month: str, kind: str, records: Sequence[Any], correction: bool = False) -> None:
    """Appends one kind of row to its month's file, unless this is a dry run."""
    if not records or self.options.dry_run:
      return
    self.store.append(kind, list(records), month=month, correction=correction)


def months_missing_views(row_store: store_module.Store, today: date) -> set[str]:
  """Finds the months whose view files have gone, so the next tick writes them back."""
  window = set(views.months_in_window(today))
  missing: set[str] = set()
  for month in row_store.months():
    if month not in window:
      continue
    for group in views.VIEW_GROUPS:
      if not (row_store.views_dir / f"{group}-{month}.json").exists():
        missing.add(month)
        break
  return missing


def prune_old_files(options: Options, row_store: store_module.Store, report: TickReport, now: datetime) -> None:
  """Deletes the rows and views the dashboard can no longer show, so the store stops growing."""
  today = now.date()
  try:
    data_files = row_store.prune_months(views.months_in_window(today)[0], dry_run=options.dry_run)
    view_files = views.prune_views(options.out, today, dry_run=options.dry_run)
  except (store_module.StoreError, OSError) as error:
    raise TickError(f"old files could not be deleted: {error}") from error
  report.data_files_pruned = len(data_files)
  report.view_files_pruned = len(view_files)


def rebuild_views(
    options: Options, row_store: store_module.Store, state: store_module.State, report: TickReport, now: datetime
) -> None:
  """Rewrites the view files for every month that needs them, then meta.json."""
  today = now.date()
  wanted = sorted(report.months_touched | months_missing_views(row_store, today))
  try:
    if options.dry_run:
      with tempfile.TemporaryDirectory(prefix="ci-metrics-dry-run-") as throwaway:
        summary = views.build_views(
            row_store,
            Path(throwaway),
            today,
            months=wanted,
            uncollected=state.pending_count,
            generated_at=iso_utc(now),
        )
      report.months_rebuilt = list(summary.get("months_rebuilt", []))
      return
    summary = views.build_views(
        row_store,
        options.out,
        today,
        months=wanted,
        uncollected=state.pending_count,
        generated_at=iso_utc(now),
    )
  except (views.ViewError, store_module.StoreError, OSError) as error:
    raise TickError(f"the views could not be rebuilt: {error}") from error

  report.months_rebuilt = list(summary.get("months_rebuilt", []))
  report.views_written = len(summary.get("written", []))
  report.views_unchanged = len(summary.get("unchanged", []))
  report.pr_files_written = len(summary.get("pr_written", []))


def format_count(value: int) -> str:
  """Formats a count with thousands separators, the way the report prints numbers."""
  return f"{value:,}"


def print_report(report: TickReport, options: Options) -> None:
  """Prints the tick's report: what it saw, what it wrote, and what it could not read."""
  width = 26
  print()
  print(f"CI metrics tick - {options.repo}")
  print("-" * (18 + len(options.repo)))

  def line(label: str, value: object) -> None:
    print(f"  {label:<{width}} {value}")

  line("Mode", f"{report.mode} ({report.window_since} .. {report.window_until or 'now'})")
  line("Store", str(options.out) + ("  [dry run: nothing written]" if options.dry_run else ""))
  line(
      "Runs seen",
      f"{format_count(report.runs_seen)} "
      f"({format_count(report.runs_already_stored)} already stored, "
      f"{format_count(report.runs_collected)} collected, {format_count(report.runs_failed)} failed)",
  )
  line(
      "Attempts written",
      f"{format_count(report.attempts_written)} "
      f"({format_count(report.attempts_pending)} still running, "
      f"{format_count(report.attempts_incomplete)} stored unfinished)",
  )
  line("Jobs written", format_count(report.jobs_written))
  line("Suites written", format_count(report.suites_written))
  line(
      "Tests written",
      f"{format_count(report.tests_written)} of {format_count(report.tests_seen)} read"
      + (f", {report.full_snapshots} full daily snapshot(s)" if report.full_snapshots else ""),
  )
  line("Rescue events written", format_count(report.rescues_written))
  line(
      "Test artifacts",
      f"{format_count(report.runs_tests_harvested)} run(s) harvested, "
      f"{format_count(report.runs_tests_expired)} too old to have any, "
      f"{format_count(report.runs_tests_superseded)} superseded, "
      f"{format_count(report.runs_tests_failed)} unreadable",
  )
  line(
      "Pull requests waiting",
      f"{format_count(report.prs_watched)} still open when their run was read, "
      f"{format_count(report.merges_learned)} merged since the last tick",
  )

  if report.suite_reasons:
    for reason, count in sorted(report.suite_reasons.items(), key=lambda item: -item[1]):
      line("Suites with no file", f"{reason}: {count} ({', '.join(report.suite_reason_examples.get(reason, []))})")
  else:
    line("Suites with no file", "none")
  line("Partial suites", ", ".join(report.partial_suites) if report.partial_suites else "none")

  line("Months touched", ", ".join(sorted(report.months_touched)) or "none")
  line(
      "Views",
      f"{format_count(report.views_written)} file(s) written, "
      f"{format_count(report.views_unchanged)} unchanged, "
      f"{format_count(report.pr_files_written)} pull request file(s)",
  )
  line(
      "Pruned",
      f"{format_count(report.data_files_pruned)} data file(s), {format_count(report.view_files_pruned)} view file(s) "
      f"older than the {views.VIEW_WINDOW_DAYS}-day window" + (" [dry run: would delete]" if options.dry_run else ""),
  )
  line("API requests spent", f"{format_count(report.api_requests)} (+{format_count(report.download_requests)} downloads)")
  if report.stopped_because:
    line("Stopped early", report.stopped_because)
  for problem in report.problems:
    line("Problem", problem)
  print()


def summary_line(report: TickReport, options: Options) -> str:
  """Builds the one line a log can grep, which is always the last thing printed."""
  if report.problems:
    outcome = "lost data"
  elif not report.changed:
    outcome = "nothing new"
  else:
    outcome = "ok"
  parts = [
      f"ci-metrics {report.mode}: {outcome}",
      f"{report.runs_collected} run(s)",
      f"{report.attempts_written} attempt(s)",
      f"{report.jobs_written} job(s)",
      f"{report.tests_written} test(s)",
      f"{report.views_written + report.pr_files_written} view file(s)",
      f"{report.api_requests} API request(s)",
  ]
  pruned = report.data_files_pruned + report.view_files_pruned
  if pruned:
    parts.append(f"{pruned} old file(s) pruned")
  if report.stopped_because:
    parts.append("stopped early")
  if options.dry_run:
    parts.append("dry run")
  return " | ".join(parts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses the command line."""
  parser = argparse.ArgumentParser(
      prog="python -m collector.tick",
      description="Collect one window of pipeline runs into a store and rebuild its views.",
  )
  parser.add_argument("--out", required=True, help="Store directory. Required: there is no default store.")
  parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Repository to read, as owner/name (default {DEFAULT_REPO}).")
  parser.add_argument("--since", default=None, help="Collect runs created on or after this day (YYYY-MM-DD).")
  parser.add_argument("--until", default=None, help="Collect runs created on or before this day (YYYY-MM-DD).")
  parser.add_argument(
      "--backfill-days",
      type=int,
      default=DEFAULT_BACKFILL_DAYS,
      help=f"How far back the first tick of an empty store reaches (default {DEFAULT_BACKFILL_DAYS}).",
  )
  parser.add_argument("--dry-run", action="store_true", help="Fetch and compute everything, write nothing.")
  parser.add_argument(
      "--max-runs",
      type=int,
      default=None,
      help="Stop after this many runs and leave the rest for the next tick.",
  )
  return parser.parse_args(argv)


def build_options(args: argparse.Namespace) -> Options:
  """Turns the parsed arguments into the tick's settings."""
  # `Path("")` is `PosixPath(".")`, so an empty --out would reach the store as the working
  # directory and only be caught when that directory happens to hold a `.git`. The store's own
  # guard cannot see it, because by then the empty string has already become a path.
  if not str(args.out).strip():
    raise UsageError("--out needs a directory; there is no default.")
  since = parse_day(args.since) if args.since else None
  until = parse_day(args.until, end_of_day=True) if args.until else None
  if since is not None and until is not None and until < since:
    raise UsageError(f"--until {args.until} is before --since {args.since}")
  if args.backfill_days < 1:
    raise UsageError(f"--backfill-days must be at least 1, got {args.backfill_days}")
  if args.max_runs is not None and args.max_runs < 1:
    raise UsageError(f"--max-runs must be at least 1, got {args.max_runs}")
  return Options(
      out=Path(args.out).expanduser(),
      repo=args.repo,
      since=since,
      until=until,
      backfill_days=args.backfill_days,
      dry_run=bool(args.dry_run),
      max_runs=args.max_runs,
  )


def main(argv: Sequence[str] | None = None) -> int:
  """Runs one tick."""
  args = parse_args(argv)
  try:
    owner, name = split_repo(args.repo)
    options = build_options(args)
    row_store = store_module.Store(options.out)
  except (UsageError, store_module.StoreError) as error:
    print(f"error: {error}", file=sys.stderr)
    return EXIT_USAGE

  session = CountingSession()
  client = github.GitHubClient(owner, name, session=session)
  report = TickReport()
  try:
    if not options.dry_run:
      row_store.sweep_temp()
    state = row_store.load_state()
    if state.rebuilt:
      _warn("WARNING: state.json was missing or unreadable; the index was rebuilt from the stored rows.")
    tick = Tick(options, client, row_store, state, report)
    tick.collect()
    prune_old_files(options, row_store, report, tick.now)
    rebuild_views(options, row_store, state, report, tick.now)
    if not options.dry_run:
      row_store.save_state(state)
  except UsageError as error:
    # Asked for something impossible: re-running would fail the same way, so this is a
    # command line to fix, not a tick to repeat. The `finally` below still closes the client.
    print(f"error: {error}", file=sys.stderr)
    return EXIT_USAGE
  except (TickError, store_module.StoreError) as error:
    report.note_problem(str(error))
  except github.GitHubError as error:
    report.note_problem(f"the GitHub API could not be read: {error}")
  finally:
    report.api_requests = session.api_requests
    report.download_requests = session.download_requests
    client.close()

  print_report(report, options)
  print(summary_line(report, options))
  return EXIT_DATA_LOST if report.problems else EXIT_OK


if __name__ == "__main__":
  sys.exit(main())
