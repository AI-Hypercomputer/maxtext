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

"""Builds the JSON files the dashboard loads, out of the rows the store already holds."""

from __future__ import annotations

import bisect
import dataclasses
import json
import os
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from collector import derive
from collector import rows

# View schema version. It travels in every file. The browser refuses a major it does not know
# rather than drawing a field that moved.
SCHEMA_VERSION = 1

# Where the files live under the output directory given by --out. `data/` next to it belongs
# to the store; nothing here writes there.
VIEWS_DIRNAME = "views"
PR_DIRNAME = "pr"
META_FILENAME = "meta.json"

# The five month-split groups. meta.json lists them in this order and so does the summary.
GROUP_RUNS = "runs"
GROUP_SUITES = "suites"
GROUP_FLAKY = "flaky"
GROUP_QUEUE = "queue"
GROUP_WORKFLOWS = "workflows"
VIEW_GROUPS = (GROUP_RUNS, GROUP_SUITES, GROUP_FLAKY, GROUP_QUEUE, GROUP_WORKFLOWS)

# How far back the dashboard can look. The range picker is client side, so every view holds
# the whole window and a range change fetches nothing.
VIEW_WINDOW_DAYS = 90

# A month stays open for rebuilding for this many days after the calendar turns. A tick just
# after midnight on the 1st collects runs created in the month that has just ended, and those
# rows belong to that month's file. Two days covers the 24-hour pending rule with room to
# spare. A caller that knows exactly which months it appended to passes `months=` instead.
MONTH_OVERLAP_DAYS = 2

# How far a scheduled run may sit from a merged pull request's run and still be quoted as its
# probe. The schedule fires every four hours; beyond half a day the two ran under different
# conditions and the honest answer is no probe.
PROBE_MAX_GAP_SECONDS = 12 * 3600

# Event names, copied rather than imported: `runs.py` pulls in `github.py` and therefore
# `requests`, and this module must stay importable without them. They are the values GitHub
# puts in a run's `event` field.
EVENT_SCHEDULE = "schedule"
EVENT_PULL_REQUEST = "pull_request"

# Lanes that never appear in the per-job table of the runs view. Hosted jobs are the ten
# "Setup Parameters" jobs and the gates - seconds long, on shared runners, never charted -
# and "No runner" is a job that was skipped and asked for no machine at all. Both are kept in
# full inside pr/<n>.json, where the whole job list of one run is the point.
UNCHARTED_LANES = (derive.LANE_HOSTED, derive.LANE_NO_RUNNER)

# A job conclusion that means the job failed, for the error rows of a pull request view.
CONCLUSION_FAILURE = "failure"

# The two test outcomes that count as a failure. Copied from `junit.STATUS_FAILED` and
# `junit.STATUS_ERROR` rather than imported, for the same reason as the event names: this
# module keeps its imports to `derive` and `rows`.
TEST_STATUS_FAILED = "failed"
TEST_STATUS_ERROR = "error"

# Column lists. Each one is the contract for one table: `to_columnar` refuses a record whose
# keys are not exactly these, so a builder and its column list can never drift apart.

RUNS_COLUMNS = (
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
)

# The per-job table of the runs view carries what the aggregate charts add up, and nothing
# else. The three job timestamps and the runner group are deliberately absent: they are in
# pr/<n>.json, where one run's jobs are the whole point, and here they would be a third of the
# file for numbers no chart on the Main page reads.
RUN_JOBS_COLUMNS = (
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
)

SUITES_COLUMNS = (
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
)

RESCUES_COLUMNS = (
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
)

RESCUE_TESTS_COLUMNS = (
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
)

QUEUE_COLUMNS = (
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
)

WORKFLOWS_COLUMNS = (
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
)

PR_ATTEMPTS_COLUMNS = (
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
)

PR_JOBS_COLUMNS = (
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
)

PR_STEPS_COLUMNS = (
    "attempt",
    "job_id",
    "number",
    "name",
    "status",
    "conclusion",
    "started_at",
    "completed_at",
)

PR_SUITES_COLUMNS = (
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
)

PR_TESTS_COLUMNS = (
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
)

PR_ERRORS_COLUMNS = (
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
)


class ViewError(ValueError):
  """Raised when a view cannot be built or a view file cannot be read back."""


class StateLike(Protocol):
  """The part of state.json this module reads: how many attempts are still in flight."""

  @property
  def pending_count(self) -> int:
    """Returns how many run attempts have been seen but not collected."""


class RowStore(Protocol):
  """The part of the store this module reads, satisfied by `store.Store`."""

  def months(self, kind: str) -> Sequence[str]:
    """Returns the months one row kind has a file for, as "YYYY-MM", ascending."""

  def read_rows(self, kind: str, months: Sequence[str] | None = None) -> Sequence[rows.Row]:
    """Returns one kind's stored rows for the given months, corrections applied."""

  def load_state(self) -> StateLike:
    """Returns the store's state, for the count of attempts still in flight."""


# ----------------------------------------------------------------------------------------
# The columnar format
# ----------------------------------------------------------------------------------------


def to_columnar(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> dict[str, Any]:
  """Turns a list of records into one columnar table."""
  names = list(columns)
  if len(set(names)) != len(names):
    raise ViewError(f"columnar table has a repeated column: {sorted({n for n in names if names.count(n) > 1})}")
  expected = set(names)
  table_rows: list[list[Any]] = []
  for index, record in enumerate(records):
    keys = set(record)
    if keys != expected:
      missing = sorted(expected - keys)
      unexpected = sorted(keys - expected)
      raise ViewError(f"record {index} does not match the column list: missing {missing}, unexpected {unexpected}")
    table_rows.append([record[name] for name in names])
  return {"columns": names, "rows": table_rows}


def from_columnar(obj: Mapping[str, Any]) -> list[dict[str, Any]]:
  """Turns a columnar table back into a list of records. The exact inverse of `to_columnar`."""
  if not isinstance(obj, Mapping) or "columns" not in obj or "rows" not in obj:
    raise ViewError("not a columnar table: expected an object with 'columns' and 'rows'")
  names = obj["columns"]
  table_rows = obj["rows"]
  if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
    raise ViewError("columnar 'columns' must be a list of names")
  if not isinstance(table_rows, Sequence) or isinstance(table_rows, (str, bytes)):
    raise ViewError("columnar 'rows' must be a list of rows")
  records: list[dict[str, Any]] = []
  for index, values in enumerate(table_rows):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != len(names):
      raise ViewError(f"columnar row {index} has {len(values)} values for {len(names)} columns")
    records.append(dict(zip(names, values)))
  return records


def table_row_count(table: Mapping[str, Any]) -> int:
  """Returns how many rows a columnar table holds, without rehydrating it."""
  table_rows = table.get("rows")
  return len(table_rows) if isinstance(table_rows, list) else 0


# ----------------------------------------------------------------------------------------
# Months
# ----------------------------------------------------------------------------------------


def parse_timestamp(value: Any) -> datetime | None:
  """Reads an ISO-8601 timestamp the way `derive.py` does, so both agree on every row."""
  return derive.parse_timestamp(value)


def month_key(moment: Any) -> str | None:
  """Returns the "YYYY-MM" a moment belongs to, in UTC."""
  if isinstance(moment, datetime):
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).strftime("%Y-%m")
  if isinstance(moment, date):
    return moment.strftime("%Y-%m")
  parsed = parse_timestamp(moment)
  if parsed is None:
    return None
  return parsed.astimezone(timezone.utc).strftime("%Y-%m")


def month_of_run(run: rows.RunRow) -> str | None:
  """Returns the month a run's rows are filed under: its `created_at` in UTC."""
  return month_key(run.created_at)


def _month_start(month: str) -> date:
  """Returns the first day of a "YYYY-MM" month."""
  try:
    return datetime.strptime(month, "%Y-%m").date()
  except (TypeError, ValueError) as error:
    raise ViewError(f"not a month key: {month!r}") from error


def _shift_month(month: str, back: int) -> str:
  """Returns the month `back` months before `month`."""
  start = _month_start(month)
  year, index = start.year, start.month - back
  year += (index - 1) // 12
  index = (index - 1) % 12 + 1
  return f"{year:04d}-{index:02d}"


def months_in_window(today: date, window_days: int = VIEW_WINDOW_DAYS) -> list[str]:
  """Lists the months the dashboard's window touches, oldest first."""
  first = window_first_day(today, window_days)
  months = []
  cursor = date(first.year, first.month, 1)
  while cursor <= today:
    months.append(cursor.strftime("%Y-%m"))
    cursor = date(cursor.year + (cursor.month // 12), cursor.month % 12 + 1, 1)
  return months


def window_first_day(today: date, window_days: int = VIEW_WINDOW_DAYS) -> date:
  """Returns the first day the dashboard's window still shows."""
  return today - timedelta(days=max(window_days, 1) - 1)


def prune_views(out_dir: Path | str, today: date, dry_run: bool = False) -> list[Path]:
  """Deletes the view files the dashboard can no longer show."""
  views_root = Path(out_dir) / VIEWS_DIRNAME
  if not views_root.is_dir():
    return []
  keep_from = months_in_window(today)[0]
  first_day = window_first_day(today)
  doomed: list[Path] = []
  for group in VIEW_GROUPS:
    for path in views_root.glob(f"{group}-*.json"):
      month = path.stem[len(group) + 1 :]
      try:
        _month_start(month)
      except ViewError:
        continue
      if month < keep_from:
        doomed.append(path)
  meta = read_view(views_root / META_FILENAME)
  entries = meta.get("pull_requests") if isinstance(meta, Mapping) else None
  for key, entry in (entries or {}).items():
    try:
      number = int(key)
    except (TypeError, ValueError):
      continue
    updated = parse_timestamp(entry.get("updated_at")) if isinstance(entry, Mapping) else None
    if updated is None or updated.astimezone(timezone.utc).date() >= first_day:
      continue
    path = views_root / PR_DIRNAME / f"{number}.json"
    if path.exists():
      doomed.append(path)
  doomed.sort()
  if not dry_run:
    for path in doomed:
      path.unlink()
  return doomed


def months_to_rebuild(available: Iterable[str], today: date) -> list[str]:
  """Picks the months whose view files this tick should rewrite."""
  current = today.strftime("%Y-%m")
  open_months = {current}
  if today.day <= MONTH_OVERLAP_DAYS:
    open_months.add(_shift_month(current, 1))
  held = set(available)
  open_months |= {month for month in held if month > current}
  return sorted(open_months & held)


# ----------------------------------------------------------------------------------------
# Reading stored rows
# ----------------------------------------------------------------------------------------


def store_months(store: RowStore) -> list[str]:
  """Returns every month the store holds a file for, in any kind, oldest first."""
  found: set[str] = set()
  for kind in (rows.KIND_RUN, rows.KIND_JOB, rows.KIND_SUITE, rows.KIND_TEST, rows.KIND_RESCUE):
    found.update(store.months(kind))
  return sorted(found)


def latest_rows(records: Iterable[rows.Row]) -> list[rows.Row]:
  """Applies the reader rule: the last row per key wins."""
  best: dict[str, tuple[str, int, rows.Row]] = {}
  for position, record in enumerate(records):
    key = record.key()
    stamp = getattr(record, "collected_at", "") or ""
    current = best.get(key)
    if current is None or (stamp, position) >= (current[0], current[1]):
      best[key] = (stamp, position, record)
  return [entry[2] for entry in best.values()]


def job_payload(job: rows.JobRow) -> dict[str, Any]:
  """Turns a stored job row back into the shape `derive.py` reads."""
  payload = dataclasses.asdict(job)
  payload["id"] = job.job_id
  payload["run_attempt"] = job.attempt
  return payload


def pool_label(job: Mapping[str, Any]) -> str | None:
  """Returns the runs-on label that names a job's runner pool."""
  labels = job.get("labels")
  if not isinstance(labels, (list, tuple)):
    return None
  known = [
      label.strip().lower()
      for label in labels
      if isinstance(label, str) and label.strip().lower() in derive.RUNNER_LABEL_LANES
  ]
  if known:
    return known[0]
  other = [label.strip().lower() for label in labels if isinstance(label, str) and label.strip()]
  return other[0] if other else None


def _median(values: Sequence[float]) -> float | None:
  """Returns the median of the values, or None for an empty list."""
  return statistics.median(values) if values else None


# ----------------------------------------------------------------------------------------
# One month of rows, deduplicated and indexed
# ----------------------------------------------------------------------------------------


@dataclass
class RunAttempt:
  """One attempt of one run, with its jobs, its suites and the numbers `derive.py` gives."""

  run: rows.RunRow
  jobs: list[dict[str, Any]] = field(default_factory=list)
  suites: list[rows.SuiteRow] = field(default_factory=list)
  tests: list[rows.TestRow] = field(default_factory=list)

  @property
  def run_id(self) -> int:
    """The run this attempt belongs to."""
    return self.run.run_id

  @property
  def attempt(self) -> int:
    """The attempt number."""
    return self.run.attempt

  @property
  def split(self) -> derive.PhaseSplit:
    """The attempt's phase split, from `derive.phase_split` over its jobs."""
    return derive.phase_split(self.jobs)

  @property
  def machine_seconds(self) -> float:
    """Runner time the attempt consumed, from `derive.machine_seconds`."""
    return derive.machine_seconds(self.jobs)


@dataclass
class MonthData:
  """Every row of one month, deduplicated, indexed by run and attempt."""

  month: str
  attempts: dict[tuple[int, int], RunAttempt] = field(default_factory=dict)
  rescues: dict[int, list[rows.RescueRow]] = field(default_factory=dict)
  merged_pr_runs: dict[int, int] = field(default_factory=dict)
  kept_run_ids: set[int] = field(default_factory=set)
  latest_collected_at: dict[int, str] = field(default_factory=dict)

  def attempts_of(self, run_id: int) -> list[RunAttempt]:
    """Returns a run's attempts, lowest number first."""
    found = [entry for key, entry in self.attempts.items() if key[0] == run_id]
    return sorted(found, key=lambda entry: entry.attempt)

  def first_attempt(self, run_id: int) -> RunAttempt | None:
    """Returns the run's lowest stored attempt, the one every trend chart is drawn from."""
    found = self.attempts_of(run_id)
    return found[0] if found else None

  def last_attempt(self, run_id: int) -> RunAttempt | None:
    """Returns the run's highest stored attempt, the one that carries the final conclusion."""
    found = self.attempts_of(run_id)
    return found[-1] if found else None

  def pr_of(self, run_id: int) -> int | None:
    """Returns the merged pull request number a run belongs to, or None."""
    last = self.last_attempt(run_id)
    if last is None or not last.run.is_merged_pr:
      return None
    return last.run.pr_number

  def is_representative(self, run_id: int) -> bool:
    """True when this run is the one its pull request is drawn from."""
    number = self.pr_of(run_id)
    if number is None:
      return True
    return self.merged_pr_runs.get(number) == run_id


def _is_kept(run: rows.RunRow) -> bool:
  """True when a run belongs in the published views."""
  if run.superseded:
    return False
  if run.is_merged_pr:
    return True
  return run.event == EVENT_SCHEDULE


def load_month(store: RowStore, month: str) -> MonthData:
  """Reads one month out of the store and indexes it."""
  data = MonthData(month=month)
  for kind in (rows.KIND_RUN, rows.KIND_JOB, rows.KIND_SUITE, rows.KIND_RESCUE):
    for record in latest_rows(store.read_rows(kind, [month])):
      _place_row(data, record)

  for key in sorted(data.attempts):
    entry = data.attempts[key]
    entry.jobs.sort(key=lambda payload: (str(payload.get("name") or ""), payload.get("id") or 0))
    entry.suites.sort(key=lambda suite: suite.suite_id)

  for run_id in {key[0] for key in data.attempts}:
    last = data.last_attempt(run_id)
    if last is None:
      continue
    if _is_kept(last.run):
      data.kept_run_ids.add(run_id)
    if last.run.is_merged_pr and last.run.pr_number is not None and _is_kept(last.run):
      _claim_pull_request(data, last.run)

  wanted = set(data.merged_pr_runs.values())
  if wanted:
    for record in latest_rows(store.read_rows(rows.KIND_TEST, [month])):
      if record.run_id in wanted:
        _place_row(data, record)
    for key in sorted(data.attempts):
      data.attempts[key].tests.sort(key=lambda test: (test.suite_id, test.classname, test.name))
  return data


def _place_row(data: MonthData, record: rows.Row) -> None:
  """Files one deduplicated row under its run and attempt."""
  kind = rows.row_kind(record)
  stamp = getattr(record, "collected_at", "") or ""
  run_id = record.run_id
  if stamp > data.latest_collected_at.get(run_id, ""):
    data.latest_collected_at[run_id] = stamp
  if kind == rows.KIND_RUN:
    data.attempts[(record.run_id, record.attempt)] = RunAttempt(run=record)
    return
  if kind == rows.KIND_RESCUE:
    data.rescues.setdefault(record.run_id, []).append(record)
    return
  entry = data.attempts.get((record.run_id, record.attempt))
  if entry is None:
    return
  if kind == rows.KIND_JOB:
    entry.jobs.append(job_payload(record))
  elif kind == rows.KIND_SUITE:
    entry.suites.append(record)
  elif kind == rows.KIND_TEST:
    entry.tests.append(record)


def _claim_pull_request(data: MonthData, run: rows.RunRow) -> None:
  """Records the one run that represents a merged pull request."""
  number = run.pr_number
  if number is None:
    return
  held = data.merged_pr_runs.get(number)
  if held is None:
    data.merged_pr_runs[number] = run.run_id
    return
  incumbent = data.last_attempt(held)
  if incumbent is None:
    data.merged_pr_runs[number] = run.run_id
    return
  challenger_matches = run.pr_head_sha is not None and run.head_sha == run.pr_head_sha
  incumbent_matches = incumbent.run.pr_head_sha is not None and incumbent.run.head_sha == incumbent.run.pr_head_sha
  if challenger_matches and not incumbent_matches:
    data.merged_pr_runs[number] = run.run_id
    return
  if incumbent_matches and not challenger_matches:
    return
  if (run.created_at or "") > (incumbent.run.created_at or ""):
    data.merged_pr_runs[number] = run.run_id


# ----------------------------------------------------------------------------------------
# Shared per-job reads
# ----------------------------------------------------------------------------------------


def _job_flavor(job: Mapping[str, Any], flavors: set[str]) -> tuple[str | None, int | None]:
  """Returns the test flavor and worker number a job ran, or (None, None)."""
  # `derive.flavor_of` also answers "Build Wheel" and "Pre-commit Linters". A name counts as a flavor only if it
  # is in `flavors`, the allowlist from `derive.test_flavors`.
  parsed = derive.parse_execute_tests_name(job.get("name"))
  if parsed is not None and parsed[0] in flavors:
    return parsed[0], parsed[1]
  candidate = derive.flavor_of(job)
  if candidate is not None and candidate in flavors:
    return candidate, None
  return None, None


def _job_record(job: Mapping[str, Any], flavors: set[str]) -> dict[str, Any]:
  """Builds the per-job fields every view spells the same way."""
  flavor, worker = _job_flavor(job, flavors)
  return {
      "job_id": job.get("id"),
      "name": job.get("name"),
      "lane": derive.device_lane(job),
      "flavor": flavor,
      "worker": worker,
      "conclusion": job.get("conclusion"),
      "runner_label": pool_label(job),
      "runner_group_name": job.get("runner_group_name"),
      "created_at": job.get("created_at"),
      "started_at": job.get("started_at"),
      "completed_at": job.get("completed_at"),
      "queued_seconds": derive.queue_seconds(job),
      "setup_seconds": derive.setup_seconds(job),
      "run_seconds": derive.run_seconds(job),
  }


def _suite_record(entry: RunAttempt, suite: rows.SuiteRow, flavors: set[str]) -> dict[str, Any]:
  """Builds the per-suite fields shared by the suites view and a pull request view."""
  nested = suite.nested_in is not None
  measurable = not nested and suite.flavor in flavors
  flavor_jobs = derive.jobs_for_flavor(entry.jobs, suite.flavor) if measurable else []
  return {
      "suite_id": suite.suite_id,
      "flavor": suite.flavor,
      "nested_in": suite.nested_in,
      "workers_named": derive.worker_count(flavor_jobs, suite.flavor) if measurable else None,
      "workers_ran": sum(1 for job in flavor_jobs if derive.held_a_runner(job)) if measurable else None,
      "duration_seconds": derive.suite_duration_seconds(flavor_jobs) if measurable else None,
      "machine_seconds": derive.machine_seconds(flavor_jobs) if measurable else None,
      "collected": suite.collected,
      "skipped": suite.skipped,
      "executed": suite.executed,
      "failed": suite.failed,
      "errored": suite.errored,
      "junit_seconds": suite.junit_seconds,
      "reason": suite.reason,
      "is_partial": suite.is_partial,
      "missing_workers": list(suite.missing_workers),
      "published_workers": list(suite.published_workers),
  }


def _overlap_counts(data: MonthData) -> dict[int, int]:
  """Counts, for every published run, how many other runs were in flight at the same time."""
  windows: dict[int, tuple[datetime, datetime]] = {}
  for run_id in sorted(data.kept_run_ids | {key[0] for key in data.attempts}):
    last = data.last_attempt(run_id)
    first = data.first_attempt(run_id)
    if last is None or first is None:
      continue
    start = parse_timestamp(last.run.created_at) or parse_timestamp(first.split.first_created_at)
    end = parse_timestamp(last.split.last_completed_at) or parse_timestamp(last.run.updated_at)
    if start is None or end is None or end < start:
      continue
    windows[run_id] = (start, end)

  starts = sorted(window[0] for window in windows.values())
  ends = sorted(window[1] for window in windows.values())
  counts: dict[int, int] = {}
  for run_id, (start, end) in windows.items():
    began_by_my_end = bisect.bisect_right(starts, end)
    ended_before_my_start = bisect.bisect_left(ends, start)
    counts[run_id] = max(0, began_by_my_end - ended_before_my_start - 1)
  return counts


# ----------------------------------------------------------------------------------------
# The five month views
# ----------------------------------------------------------------------------------------


def build_runs_view(data: MonthData) -> dict[str, Any]:
  """Builds the runs view: one row per merged pull request, plus its jobs."""
  overlaps = _overlap_counts(data)
  run_records: list[dict[str, Any]] = []
  job_records: list[dict[str, Any]] = []

  for number in sorted(data.merged_pr_runs):
    run_id = data.merged_pr_runs[number]
    first = data.first_attempt(run_id)
    last = data.last_attempt(run_id)
    if first is None or last is None:
      continue
    attempts = data.attempts_of(run_id)
    split = first.split
    run_records.append(
        {
            "pr": number,
            "title": last.run.pr_title,
            "author": last.run.pr_user,
            "merged_at": last.run.pr_merged_at,
            "head_sha": last.run.head_sha,
            "base_ref": last.run.pr_base_ref,
            "html_url": last.run.html_url,
            "run_id": run_id,
            "run_number": last.run.run_number,
            "event": last.run.event,
            "status": last.run.status,
            "conclusion": last.run.conclusion,
            "created_at": last.run.created_at,
            "run_started_at": last.run.run_started_at,
            "attempts": max(entry.attempt for entry in attempts),
            "attempts_stored": len(attempts),
            "first_created_at": split.first_created_at,
            "last_completed_at": split.last_completed_at,
            "queued_seconds": split.queued_seconds,
            "setup_seconds": split.setup_seconds,
            "tests_seconds": split.tests_seconds,
            "tail_seconds": split.tail_seconds,
            "total_seconds": split.total_seconds,
            "wall_seconds": split.wall_seconds,
            "machine_seconds": first.machine_seconds,
            "rerun_machine_seconds": sum(entry.machine_seconds for entry in attempts if entry.attempt > first.attempt),
            "jobs_counted": split.jobs_counted,
            "jobs_with_tests": split.jobs_with_tests,
            "overlapping_runs": overlaps.get(run_id),
        }
    )

    flavors = set(derive.test_flavors(first.jobs))
    for job in first.jobs:
      shared = _job_record(job, flavors)
      if shared["lane"] in UNCHARTED_LANES:
        continue
      job_records.append(
          {
              "run_id": run_id,
              "pr": number,
              "attempt": first.attempt,
              "job_id": shared["job_id"],
              "name": shared["name"],
              "lane": shared["lane"],
              "flavor": shared["flavor"],
              "worker": shared["worker"],
              "conclusion": shared["conclusion"],
              "runner_label": shared["runner_label"],
              "queued_seconds": shared["queued_seconds"],
              "setup_seconds": shared["setup_seconds"],
              "run_seconds": shared["run_seconds"],
          }
      )

  return _view_file(
      GROUP_RUNS,
      data.month,
      {
          "runs": to_columnar(run_records, RUNS_COLUMNS),
          "jobs": to_columnar(job_records, RUN_JOBS_COLUMNS),
      },
  )


def build_suites_view(data: MonthData) -> dict[str, Any]:
  """Builds the suites view: one row per suite per attempt of every published run."""
  records: list[dict[str, Any]] = []
  for run_id in sorted(data.kept_run_ids):
    last = data.last_attempt(run_id)
    if last is None:
      continue
    pr_number = data.pr_of(run_id)
    for entry in data.attempts_of(run_id):
      flavors = set(derive.test_flavors(entry.jobs))
      for suite in entry.suites:
        records.append(
            {
                "run_id": run_id,
                "pr": pr_number,
                "is_representative": data.is_representative(run_id),
                "event": entry.run.event,
                "created_at": entry.run.created_at,
                "merged_at": entry.run.pr_merged_at,
                "attempt": entry.attempt,
                **_suite_record(entry, suite, flavors),
            }
        )
  return _view_file(GROUP_SUITES, data.month, {"suites": to_columnar(records, SUITES_COLUMNS)})


def build_flaky_view(data: MonthData) -> dict[str, Any]:
  """Builds the flaky view: every rescue event, and the tests that failed in each."""
  rescue_records: list[dict[str, Any]] = []
  test_records: list[dict[str, Any]] = []

  for run_id in sorted(data.kept_run_ids):
    events = data.rescues.get(run_id)
    if not events:
      continue
    pr_number = data.pr_of(run_id)
    last = data.last_attempt(run_id)
    if last is None:
      continue
    # Skip superseded runs: a PR that pushed twice has an earlier run where
    # jobs may have failed, but the developer fixed the code and pushed again.
    # Only the representative run (the one the PR merged with) counts.
    # Scheduled runs (no PR) are always kept.
    if pr_number is not None and not data.is_representative(run_id):
      continue
    for event in sorted(events, key=lambda row: (row.failed_attempt, row.job_name)):
      parsed = derive.parse_execute_tests_name(event.job_name)
      flavor = parsed[0] if parsed else None
      worker = parsed[1] if parsed else None
      failed_attempt = data.attempts.get((run_id, event.failed_attempt))
      failed_job = _find_job(failed_attempt, event.failed_job_id)
      rescue_records.append(
          {
              "run_id": run_id,
              "pr": pr_number,
              "event": event.event,
              "created_at": last.run.created_at,
              "job_name": event.job_name,
              "lane": derive.device_lane({"labels": list(event.labels)}),
              "flavor": flavor,
              "worker": worker,
              "rescued": event.rescued,
              "failed_attempt": event.failed_attempt,
              "failed_job_id": event.failed_job_id,
              "failed_conclusion": event.failed_conclusion,
              "failed_started_at": event.failed_started_at,
              "failed_completed_at": event.failed_completed_at,
              "rescued_attempt": event.rescued_attempt,
              "rescued_job_id": event.rescued_job_id,
              "final_attempt": event.final_attempt,
              "final_conclusion": event.final_conclusion,
              "wasted_seconds": derive.run_seconds(failed_job) if failed_job is not None else None,
              "html_url": event.html_url,
          }
      )
      # Primary: tests that failed on the failed attempt (stored by the
      # tick.py fix that harvests artifacts for every completed attempt).
      # Fallback: if no test rows exist on the failed attempt — artifacts
      # expired before the tick ran, or the job died before pytest — there
      # is nothing to fall back to.  The newest attempt's tests all passed
      # (that is what "rescued" means), so _failed_tests returns nothing
      # useful there either.
      tests = _failed_tests(failed_attempt, flavor, worker)
      for test in tests:
        test_records.append(
            {
                "run_id": run_id,
                "pr": pr_number,
                "job_name": event.job_name,
                "failed_attempt": event.failed_attempt,
                "suite_id": test.suite_id,
                "flavor": test.flavor,
                "worker": test.worker,
                "classname": test.classname,
                "name": test.name,
                "status": test.status,
                "duration": test.duration,
                "failure_message": test.failure_message,
            }
        )

  return _view_file(
      GROUP_FLAKY,
      data.month,
      {
          "rescues": to_columnar(rescue_records, RESCUES_COLUMNS),
          "rescue_tests": to_columnar(test_records, RESCUE_TESTS_COLUMNS),
      },
  )


def _find_job(entry: RunAttempt | None, job_id: int | None) -> dict[str, Any] | None:
  """Returns one job payload of an attempt by its id, or None when it was not stored."""
  if entry is None or job_id is None:
    return None
  for job in entry.jobs:
    if job.get("id") == job_id:
      return job
  return None


def _failed_tests(entry: RunAttempt | None, flavor: str | None, worker: int | None) -> list[rows.TestRow]:
  """Returns the tests that failed in one job of one attempt."""
  if entry is None or flavor is None:
    return []
  found = []
  for test in entry.tests:
    if flavor not in (test.flavor, test.suite_id):
      continue
    if worker is not None and test.worker is not None and test.worker != worker:
      continue
    if test.status in (TEST_STATUS_FAILED, TEST_STATUS_ERROR):
      found.append(test)
  return found


def build_queue_view(data: MonthData) -> dict[str, Any]:
  """Builds the queue view: how long each runner pool made each run wait."""
  measured: dict[int, dict[str, dict[str, Any]]] = {}
  for run_id in sorted(data.kept_run_ids):
    first = data.first_attempt(run_id)
    if first is None:
      continue
    pools: dict[str, list[float]] = {}
    lanes: dict[str, str] = {}
    for job in first.jobs:
      label = pool_label(job)
      if label is None:
        continue
      waited = derive.queue_seconds(job)
      lanes.setdefault(label, derive.device_lane(job))
      if waited is not None:
        pools.setdefault(label, []).append(waited)
    measured[run_id] = {
        label: {
            "lane": lanes.get(label, derive.LANE_UNKNOWN),
            "jobs_counted": len(waits),
            "longest_wait_seconds": max(waits) if waits else None,
            "median_wait_seconds": _median(waits),
        }
        for label, waits in pools.items()
    }

  probes = _probe_index(data, measured)
  records: list[dict[str, Any]] = []
  for run_id in sorted(measured):
    last = data.last_attempt(run_id)
    if last is None:
      continue
    pr_number = data.pr_of(run_id)
    created = parse_timestamp(last.run.created_at)
    for label in sorted(measured[run_id]):
      pool = measured[run_id][label]
      probe = _nearest_probe(probes.get(label, []), created) if last.run.event != EVENT_SCHEDULE else None
      records.append(
          {
              "run_id": run_id,
              "pr": pr_number,
              "is_representative": data.is_representative(run_id),
              "event": last.run.event,
              "created_at": last.run.created_at,
              "pool": label,
              "lane": pool["lane"],
              "jobs_counted": pool["jobs_counted"],
              "longest_wait_seconds": pool["longest_wait_seconds"],
              "median_wait_seconds": pool["median_wait_seconds"],
              "probe_run_id": probe[1] if probe else None,
              "probe_created_at": probe[2] if probe else None,
              "probe_wait_seconds": probe[3] if probe else None,
          }
      )
  return _view_file(GROUP_QUEUE, data.month, {"queue": to_columnar(records, QUEUE_COLUMNS)})


def _probe_index(
    data: MonthData, measured: Mapping[int, Mapping[str, Mapping[str, Any]]]
) -> dict[str, list[tuple[datetime, int, str, float]]]:
  """Indexes the scheduled runs' longest wait per pool, sorted by time."""
  index: dict[str, list[tuple[datetime, int, str, float]]] = {}
  for run_id, pools in measured.items():
    last = data.last_attempt(run_id)
    if last is None or last.run.event != EVENT_SCHEDULE:
      continue
    created = parse_timestamp(last.run.created_at)
    if created is None:
      continue
    for label, pool in pools.items():
      longest = pool["longest_wait_seconds"]
      if longest is None:
        continue
      index.setdefault(label, []).append((created, run_id, last.run.created_at, longest))
  for entries in index.values():
    entries.sort(key=lambda item: item[0])
  return index


def _nearest_probe(
    probes: Sequence[tuple[datetime, int, str, float]], created: datetime | None
) -> tuple[datetime, int, str, float] | None:
  """Finds the scheduled run closest in time to a merged pull request's run."""
  if not probes or created is None:
    return None
  moments = [probe[0] for probe in probes]
  position = bisect.bisect_left(moments, created)
  candidates = [probes[index] for index in (position - 1, position) if 0 <= index < len(probes)]
  if not candidates:
    return None
  nearest = min(candidates, key=lambda probe: abs((probe[0] - created).total_seconds()))
  if abs((nearest[0] - created).total_seconds()) > PROBE_MAX_GAP_SECONDS:
    return None
  return nearest


def build_workflows_view(data: MonthData) -> dict[str, Any]:
  """Builds the workflows view: one row per workflow per day."""
  days: dict[tuple[str, str], dict[str, Any]] = {}
  for run_id in sorted({key[0] for key in data.attempts}):
    first = data.first_attempt(run_id)
    last = data.last_attempt(run_id)
    if first is None or last is None or last.run.superseded:
      continue
    created = parse_timestamp(last.run.created_at)
    if created is None:
      continue
    day = created.astimezone(timezone.utc).strftime("%Y-%m-%d")
    path = last.run.workflow_path or ""
    bucket = days.setdefault(
        (day, path),
        {"workflow_name": last.run.workflow_name, "walls": [], "lanes": {}, "runs": 0},
    )
    bucket["runs"] += 1
    wall = derive.run_wall_seconds(first.jobs)
    if wall is not None:
      bucket["walls"].append(wall)
    for entry in data.attempts_of(run_id):
      for job in entry.jobs:
        seconds = derive.run_seconds(job)
        if seconds is None:
          continue
        bucket["lanes"][derive.device_lane(job)] = bucket["lanes"].get(derive.device_lane(job), 0.0) + seconds

  records = []
  for day, path in sorted(days):
    bucket = days[(day, path)]
    lanes = bucket["lanes"]
    known = (derive.LANE_TPU, derive.LANE_GPU, derive.LANE_CPU, derive.LANE_BUILD, derive.LANE_HOSTED)
    records.append(
        {
            "day": day,
            "workflow_path": path,
            "workflow_name": bucket["workflow_name"],
            "runs": bucket["runs"],
            "median_wall_seconds": _median(bucket["walls"]),
            "machine_seconds": sum(lanes.values()),
            "machine_seconds_tpu": lanes.get(derive.LANE_TPU, 0.0),
            "machine_seconds_gpu": lanes.get(derive.LANE_GPU, 0.0),
            "machine_seconds_cpu": lanes.get(derive.LANE_CPU, 0.0),
            "machine_seconds_build": lanes.get(derive.LANE_BUILD, 0.0),
            "machine_seconds_hosted": lanes.get(derive.LANE_HOSTED, 0.0),
            "machine_seconds_unknown": sum(seconds for lane, seconds in lanes.items() if lane not in known),
        }
    )
  return _view_file(GROUP_WORKFLOWS, data.month, {"workflows": to_columnar(records, WORKFLOWS_COLUMNS)})


# ----------------------------------------------------------------------------------------
# One merged pull request in full
# ----------------------------------------------------------------------------------------


def build_pr_view(pr_number: int, data: MonthData) -> dict[str, Any]:
  """Builds one merged pull request's file: every attempt, job, step, suite and test stored."""
  run_id = data.merged_pr_runs.get(pr_number)
  if run_id is None:
    raise ViewError(f"no stored run for merged pull request #{pr_number} in {data.month}")
  attempts = data.attempts_of(run_id)
  last = attempts[-1]

  attempt_records: list[dict[str, Any]] = []
  job_records: list[dict[str, Any]] = []
  step_records: list[dict[str, Any]] = []
  suite_records: list[dict[str, Any]] = []
  test_records: list[dict[str, Any]] = []
  error_records: list[dict[str, Any]] = []

  for entry in attempts:
    split = entry.split
    flavors = set(derive.test_flavors(entry.jobs))
    attempt_records.append(
        {
            "attempt": entry.attempt,
            "event": entry.run.event,
            "status": entry.run.status,
            "conclusion": entry.run.conclusion,
            "created_at": entry.run.created_at,
            "run_started_at": entry.run.run_started_at,
            "first_created_at": split.first_created_at,
            "first_started_at": split.first_started_at,
            "last_completed_at": split.last_completed_at,
            "queued_seconds": split.queued_seconds,
            "setup_seconds": split.setup_seconds,
            "tests_seconds": split.tests_seconds,
            "tail_seconds": split.tail_seconds,
            "total_seconds": split.total_seconds,
            "wall_seconds": split.wall_seconds,
            "machine_seconds": entry.machine_seconds,
            "jobs_counted": split.jobs_counted,
            "jobs_with_tests": split.jobs_with_tests,
        }
    )

    for job in entry.jobs:
      shared = _job_record(job, flavors)
      job_records.append(
          {
              "attempt": entry.attempt,
              "status": job.get("status"),
              "carried_over": derive.is_carried_over(job),
              **shared,
          }
      )
      for step in job.get("steps") or []:
        step_records.append(
            {
                "attempt": entry.attempt,
                "job_id": job.get("id"),
                "number": step.get("number"),
                "name": step.get("name"),
                "status": step.get("status"),
                "conclusion": step.get("conclusion"),
                "started_at": step.get("started_at"),
                "completed_at": step.get("completed_at"),
            }
        )
      if job.get("conclusion") == CONCLUSION_FAILURE:
        error_records.append(_error_record(entry.attempt, job, shared))

    for suite in entry.suites:
      suite_records.append({"attempt": entry.attempt, **_suite_record(entry, suite, flavors)})

    for test in entry.tests:
      test_records.append(
          {
              "attempt": entry.attempt,
              "suite_id": test.suite_id,
              "flavor": test.flavor,
              "worker": test.worker,
              "classname": test.classname,
              "name": test.name,
              "status": test.status,
              "duration": test.duration,
              "failure_message": test.failure_message,
              "suite_partial": test.suite_partial,
          }
      )

  view = {
      "schema": SCHEMA_VERSION,
      "group": PR_DIRNAME,
      "pr": pr_number,
      "run_id": run_id,
      "title": last.run.pr_title,
      "author": last.run.pr_user,
      "merged_at": last.run.pr_merged_at,
      "head_sha": last.run.head_sha,
      "base_ref": last.run.pr_base_ref,
      "html_url": last.run.html_url,
      "pr_html_url": last.run.pr_html_url,
      "event": last.run.event,
      "created_at": last.run.created_at,
      "conclusion": last.run.conclusion,
      "attempts": max(entry.attempt for entry in attempts),
      "attempts_stored": len(attempts),
      "updated_at": data.latest_collected_at.get(run_id),
      "tables": {
          "attempts": to_columnar(attempt_records, PR_ATTEMPTS_COLUMNS),
          "jobs": to_columnar(job_records, PR_JOBS_COLUMNS),
          "steps": to_columnar(step_records, PR_STEPS_COLUMNS),
          "suites": to_columnar(suite_records, PR_SUITES_COLUMNS),
          "tests": to_columnar(test_records, PR_TESTS_COLUMNS),
          "errors": to_columnar(error_records, PR_ERRORS_COLUMNS),
      },
  }
  return view


def _error_record(attempt: int, job: Mapping[str, Any], shared: Mapping[str, Any]) -> dict[str, Any]:
  """Builds one error row: the first step of a failed job that failed."""
  failed_step = None
  for step in job.get("steps") or []:
    if step.get("conclusion") == CONCLUSION_FAILURE:
      failed_step = step
      break
  return {
      "attempt": attempt,
      "job_id": shared["job_id"],
      "job_name": shared["name"],
      "lane": shared["lane"],
      "conclusion": job.get("conclusion"),
      "failed_step": failed_step.get("name") if failed_step else None,
      "failed_step_number": failed_step.get("number") if failed_step else None,
      "failed_step_started_at": failed_step.get("started_at") if failed_step else None,
      "failed_step_completed_at": failed_step.get("completed_at") if failed_step else None,
      "html_url": job.get("html_url"),
  }


# ----------------------------------------------------------------------------------------
# meta.json
# ----------------------------------------------------------------------------------------


def build_meta(
    *,
    generated_at: str,
    months: Mapping[str, Sequence[str]],
    counts: Mapping[str, Mapping[str, Mapping[str, int]]],
    pull_requests: Mapping[int, str | None],
    uncollected: int,
    window_days: int = VIEW_WINDOW_DAYS,
) -> dict[str, Any]:
  """Builds meta.json, the first file the browser reads and the only one with a build time."""
  groups: dict[str, Any] = {}
  totals: dict[str, int] = {}
  for group in VIEW_GROUPS:
    group_months = list(months.get(group, ()))
    group_counts = {month: dict(counts.get(group, {}).get(month, {})) for month in group_months}
    groups[group] = {"months": group_months, "rows": group_counts}
    totals[group] = sum(sum(tables.values()) for tables in group_counts.values())
  return {
      "schema": SCHEMA_VERSION,
      "generated_at": generated_at,
      "window_days": window_days,
      "groups": groups,
      "totals": totals,
      "uncollected_runs": uncollected,
      "pull_requests": {
          str(number): {"file": f"{PR_DIRNAME}/{number}.json", "updated_at": pull_requests[number]}
          for number in sorted(pull_requests)
      },
  }


# ----------------------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------------------


def _view_file(group: str, month: str, tables: Mapping[str, Any]) -> dict[str, Any]:
  """Wraps a group's tables in the envelope every month file shares."""
  return {"schema": SCHEMA_VERSION, "group": group, "month": month, "tables": dict(tables)}


def _dumps(obj: Any) -> str:
  """Serialises a view the one way this module ever serialises anything."""
  return json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"


def write_view_file(path: Path, obj: Any) -> str:
  """Writes one view file, and says whether it actually changed."""
  payload = _dumps(obj)
  try:
    if path.read_text(encoding="utf-8") == payload:
      return "unchanged"
  except (OSError, UnicodeDecodeError):
    pass
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp")
  temporary.write_text(payload, encoding="utf-8")
  os.replace(temporary, path)
  return "written"


def read_view(path: Path) -> dict[str, Any] | None:
  """Reads a view file back, or returns None when it is missing or unreadable."""
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    return None


def view_path(out_dir: Path, group: str, month: str) -> Path:
  """Returns the path of one group's month file."""
  return Path(out_dir) / VIEWS_DIRNAME / f"{group}-{month}.json"


def pr_view_path(out_dir: Path, pr_number: int) -> Path:
  """Returns the path of one merged pull request's file."""
  return Path(out_dir) / VIEWS_DIRNAME / PR_DIRNAME / f"{pr_number}.json"


def meta_path(out_dir: Path) -> Path:
  """Returns the path of meta.json."""
  return Path(out_dir) / VIEWS_DIRNAME / META_FILENAME


# ----------------------------------------------------------------------------------------
# The tick
# ----------------------------------------------------------------------------------------


def build_views(
    store: RowStore,
    out_dir: Path,
    today: date,
    *,
    months: Sequence[str] | None = None,
    uncollected: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
  """Builds every file the browser reads, and reports exactly what it touched."""
  out = Path(out_dir)
  views_root = out / VIEWS_DIRNAME
  views_root.mkdir(parents=True, exist_ok=True)
  (views_root / PR_DIRNAME).mkdir(parents=True, exist_ok=True)

  available = store_months(store)
  window = months_in_window(today)
  if months is None:
    targets = months_to_rebuild(available, today)
  else:
    targets = sorted(set(months) & set(available))

  stamp = generated_at or rows.utc_now_iso()
  counts: dict[str, dict[str, dict[str, int]]] = {group: {} for group in VIEW_GROUPS}
  written: list[str] = []
  unchanged: list[str] = []
  pr_written: list[str] = []
  pr_unchanged: list[str] = []
  pull_requests: dict[int, str | None] = _carried_pull_requests(views_root)

  for month in targets:
    data = load_month(store, month)
    built = {
        GROUP_RUNS: build_runs_view(data),
        GROUP_SUITES: build_suites_view(data),
        GROUP_FLAKY: build_flaky_view(data),
        GROUP_QUEUE: build_queue_view(data),
        GROUP_WORKFLOWS: build_workflows_view(data),
    }
    for group, view in built.items():
      path = view_path(out, group, month)
      outcome = write_view_file(path, view)
      (written if outcome == "written" else unchanged).append(_relative(out, path))
      counts[group][month] = {name: table_row_count(table) for name, table in view["tables"].items()}

    for number in sorted(data.merged_pr_runs):
      view = build_pr_view(number, data)
      path = pr_view_path(out, number)
      kept = _kept_pr_file(path, view)
      if kept is not None:
        pr_unchanged.append(_relative(out, path))
        pull_requests[number] = kept
        continue
      outcome = write_view_file(path, view)
      (pr_written if outcome == "written" else pr_unchanged).append(_relative(out, path))
      pull_requests[number] = view["updated_at"]

  # A month later than today's is rebuilt on purpose - `months_to_rebuild` says why - so it
  # has to be advertised too, or its files would be written and never fetched.
  advertised = window + sorted(month for month in targets if month > window[-1])
  months_by_group: dict[str, list[str]] = {}
  for group in VIEW_GROUPS:
    present: list[str] = []
    for month in advertised:
      if month in counts[group]:
        present.append(month)
        continue
      path = view_path(out, group, month)
      if not path.exists():
        continue
      present.append(month)
      stored = read_view(path)
      tables = stored.get("tables", {}) if isinstance(stored, Mapping) else {}
      counts[group][month] = {name: table_row_count(table) for name, table in tables.items()}
    months_by_group[group] = present

  pending = uncollected if uncollected is not None else store.load_state().pending_count
  meta = build_meta(
      generated_at=stamp,
      months=months_by_group,
      counts=counts,
      pull_requests=pull_requests,
      uncollected=pending,
      window_days=VIEW_WINDOW_DAYS,
  )
  meta_outcome = write_view_file(meta_path(out), meta)
  (written if meta_outcome == "written" else unchanged).append(_relative(out, meta_path(out)))

  return {
      "generated_at": stamp,
      "out_dir": str(out),
      "window_days": VIEW_WINDOW_DAYS,
      "months_available": available,
      "months_rebuilt": targets,
      "months_skipped": [month for month in available if month not in targets],
      "written": sorted(written),
      "unchanged": sorted(unchanged),
      "pr_written": sorted(pr_written),
      "pr_unchanged": sorted(pr_unchanged),
      "counts": counts,
      "uncollected_runs": pending,
      "meta": meta,
  }


def _kept_pr_file(path: Path, candidate: Mapping[str, Any]) -> str | None:
  """Says whether the pull request file already on disk describes a newer run than this one."""
  stored = read_view(path)
  if not isinstance(stored, Mapping):
    return None
  if stored.get("run_id") == candidate.get("run_id"):
    return None
  theirs = parse_timestamp(stored.get("created_at"))
  mine = parse_timestamp(candidate.get("created_at"))
  if theirs is None or mine is None or theirs <= mine:
    return None
  updated = stored.get("updated_at")
  return updated if isinstance(updated, str) else None


def _carried_pull_requests(views_root: Path) -> dict[int, str | None]:
  """Carries meta.json's pull request list forward across a tick."""
  meta = read_view(views_root / META_FILENAME)
  if not isinstance(meta, Mapping):
    return {}
  carried: dict[int, str | None] = {}
  for key, entry in (meta.get("pull_requests") or {}).items():
    try:
      number = int(key)
    except (TypeError, ValueError):
      continue
    if not (views_root / PR_DIRNAME / f"{number}.json").exists():
      continue
    carried[number] = entry.get("updated_at") if isinstance(entry, Mapping) else None
  return carried


def _relative(out: Path, path: Path) -> str:
  """Returns a path relative to the output directory, for the summary."""
  try:
    return str(path.relative_to(out))
  except ValueError:
    return str(path)
