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

"""Builds the JSON files the dashboard loads, out of the rows the store already holds.

This is layer 3. Layer 1 reads GitHub, layer 2 turns what it read into numbers and row
shapes, and this module turns stored rows into the six view groups plus one file per merged
pull request. It is the only module that writes anything, and it never fetches: everything
it needs is already on disk.

What it writes, all under `<out>/views/`:

  meta.json               what exists, when it was generated, and which pull request files
                          changed. The browser reads this first and nothing else is fetched
                          until it has.
  <group>-YYYY-MM.json    one file per view group per month: runs, suites, flaky, queue,
                          workflows.
  pr/<n>.json             one merged pull request in full - every attempt, job, step, suite
                          and test row the store holds for it.

Four rules shape the format.

  * **Columnar.** A table is `{"columns": [...], "rows": [[...], ...]}`, not a list of
    objects, because repeating twenty field names on eight thousand rows costs about three
    times the bytes. `to_columnar` and `from_columnar` are exact inverses, including None,
    and the browser rehydrates with the same two lines.
  * **Split by month, rebuilt by month.** A tick rewrites the open months only. A closed
    month's file is byte-identical from one tick to the next, so git sees no change and the
    browser's cache keeps working. `build_views` reports what it wrote, what it left alone
    because the bytes were already right, and what it skipped without reading.
  * **No timestamp inside a view file.** `generated_at` lives in meta.json alone. If a view
    carried its own build time, every file would change on every tick and the store would
    grow by a full copy six times a day.
  * **Missing is null.** A suite that published no test file reaches the view as null counts
    with its reason code, never as zero, and a suite where only some workers reported
    carries `is_partial` so it cannot be drawn as a drop. This module never fills a gap with
    a number.
  * **One row per pull request, in the runs view only.** A pull request can have more than
    one completed run - an earlier push that finished before the next push arrived is kept,
    not superseded - so the suites and queue tables can carry two rows with the same `pr`.
    The runs view and `pr/<n>.json` describe exactly one of them, and every other table
    marks it with `is_representative`. Join on `run_id`, or filter on that flag; joining on
    `pr` alone draws the abandoned push as a second point at the same place on the axis.

Where the numbers come from
---------------------------
Every duration in every view is `derive.py`'s answer, not a second implementation. The one
thing this module computes itself is arithmetic `derive.py` does not claim: the median of a
list (`statistics.median`), the count of overlapping run windows, and which scheduled probe
run sits nearest a merged pull request's run. Per-step seconds are NOT computed here at all -
steps travel as their two timestamps, because `derive.py` has no public "seconds of a named
step" and this module will not grow one.

It imports `derive` and `rows` and nothing else from the package: like `junit.py`, it stays
importable with no network module - and therefore no `requests` - present.

On the length of the builders
-----------------------------
`build_pr_view`, `build_views`, `build_runs_view` and `build_flaky_view` are longer than
anything in the modules they sit on. That is deliberate and it is measured: their branch and
statement counts are small - `build_pr_view` is seven branches - and the lines are the column
dictionaries themselves, one field per line because the formatter spreads them. Splitting a
literal into a helper per table would move the same text and cost the reader one hop per
table to see which field a column carries. If a future change adds control flow rather than
fields, split it then.
"""

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
  """Raised when a view cannot be built or a view file cannot be read back.

  The message always names the table and the field at fault, so a tick that dies on one odd
  record says which one it was.
  """


class StateLike(Protocol):
  """The part of state.json this module reads: how many attempts are still in flight."""

  @property
  def pending_count(self) -> int:
    """Returns how many run attempts have been seen but not collected."""


class RowStore(Protocol):
  """The part of the store this module reads, satisfied by `store.Store`.

  Declared structurally, the way `junit.py` declares its client, so `views.py` does not import
  the writer and a test can hand it a list-backed stub.

  `read_rows` is expected to have applied the correction rule already - of the lines sharing a
  key, the one with the greatest `collected_at` wins. `store.Store.read` does. `latest_rows`
  is applied again here anyway, because it is idempotent and a stub is not obliged to.
  """

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
  """Turns a list of records into one columnar table.

  The names are written once, in the `columns` list, and each record becomes a plain array in
  that order. Nothing is coerced: a None stays a JSON null, a float stays a float, and a list
  field stays a list.

  Args:
    records: The records, each a mapping whose keys are exactly `columns`.
    columns: The column names, in the order they should be written.

  Returns:
    `{"columns": [...], "rows": [[...], ...]}`.

  Raises:
    ViewError: `columns` repeats a name, or a record's keys are not exactly `columns`. Both
      are refused rather than patched, because a silently dropped field would reach the
      dashboard as a missing number and be drawn as a gap.
  """
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
  """Turns a columnar table back into a list of records. The exact inverse of `to_columnar`.

  The browser does the same two lines. Keeping it this simple is the point: there is no type
  coercion and no ordering trick to reproduce in JavaScript.

  Args:
    obj: A table as `to_columnar` writes it.

  Returns:
    One dict per row, keys in column order.

  Raises:
    ViewError: The object is not a table, or a row is not as long as the column list.
  """
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
  """Returns the "YYYY-MM" a moment belongs to, in UTC.

  Args:
    moment: A date, a datetime, or an ISO-8601 timestamp string.

  Returns:
    The month key, or None when the value cannot be read as a moment.
  """
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
  """Returns the first day of a "YYYY-MM" month.

  Raises:
    ViewError: The string is not a month key.
  """
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
  """Lists the months the dashboard's window touches, oldest first.

  Args:
    today: The day the tick is running.
    window_days: How far back the dashboard can look.

  Returns:
    Every "YYYY-MM" between the first day of the window and today, inclusive. A 90-day window
    touches four months at most.
  """
  first = today - timedelta(days=max(window_days, 1) - 1)
  months = []
  cursor = date(first.year, first.month, 1)
  while cursor <= today:
    months.append(cursor.strftime("%Y-%m"))
    cursor = date(cursor.year + (cursor.month // 12), cursor.month % 12 + 1, 1)
  return months


def months_to_rebuild(available: Iterable[str], today: date) -> list[str]:
  """Picks the months whose view files this tick should rewrite.

  A month is open while it is still gaining rows. That is today's month, always; the month
  before it for the first `MONTH_OVERLAP_DAYS` days, because a tick just after midnight on
  the 1st collects runs created in the month that has just ended and those rows are filed
  under it; and any month later than today's, which can only mean the store ran under a clock
  ahead of this one and its rows would otherwise never be published.

  Every other month is closed. Its file is left exactly as it is - not read, not rewritten -
  which is what keeps a tick's commit small and a closed month's URL cacheable forever.

  Args:
    available: The months the store holds.
    today: The day the tick is running.

  Returns:
    The months to rebuild, oldest first, restricted to months the store actually has.
  """
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
  """Returns every month the store holds a file for, in any kind, oldest first.

  A store keeps one file per kind per month, so a month with runs but no test rows still
  counts. The union is what the view builder walks.

  Args:
    store: The row store.

  Returns:
    The month keys, ascending.
  """
  found: set[str] = set()
  for kind in (rows.KIND_RUN, rows.KIND_JOB, rows.KIND_SUITE, rows.KIND_TEST, rows.KIND_RESCUE):
    found.update(store.months(kind))
  return sorted(found)


def latest_rows(records: Iterable[rows.Row]) -> list[rows.Row]:
  """Applies the reader rule: the last row per key wins.

  The store is append-only, so a correction is a second row carrying the same key and a later
  `collected_at`. Ties - two rows written inside the same second - are broken by position, so
  the later line in the later file wins, exactly as the browser's loader does it.

  Args:
    records: Stored rows, in the order they were written.

  Returns:
    One row per key, in the order each key was first seen.
  """
  best: dict[str, tuple[str, int, rows.Row]] = {}
  for position, record in enumerate(records):
    key = record.key()
    stamp = getattr(record, "collected_at", "") or ""
    current = best.get(key)
    if current is None or (stamp, position) >= (current[0], current[1]):
      best[key] = (stamp, position, record)
  return [entry[2] for entry in best.values()]


def job_payload(job: rows.JobRow) -> dict[str, Any]:
  """Turns a stored job row back into the shape `derive.py` reads.

  A `JobRow` carries every field the jobs endpoint returned, under the same names, with two
  exceptions: the job's id is stored as `job_id` and its attempt as `attempt`. This puts the
  API's own names back so the row can be handed straight to `derive.queue_seconds`,
  `derive.phase_split` and the rest without any of them learning about stored rows.

  Args:
    job: The stored job row.

  Returns:
    A plain dict, safe to mutate, holding both spellings.
  """
  payload = dataclasses.asdict(job)
  payload["id"] = job.job_id
  payload["run_attempt"] = job.attempt
  return payload


def pool_label(job: Mapping[str, Any]) -> str | None:
  """Returns the runs-on label that names a job's runner pool.

  The label a job asked for is the pool it queued in: `linux-x86-ct6e-180-4tpu`,
  `linux-x86-n2-32`, and so on. A known label wins over an unknown one, so the pool and the
  lane always agree with `derive.device_lane`.

  Args:
    job: The job payload.

  Returns:
    The label in lower case, or None when the job asked for no runner at all - a skipped job
    has an empty labels list and never queued anywhere.
  """
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
  """Returns the median of the values, or None for an empty list.

  A median is statistics, not job arithmetic, so it comes from the standard library rather
  than from `derive.py`.
  """
  return statistics.median(values) if values else None


# ----------------------------------------------------------------------------------------
# One month of rows, deduplicated and indexed
# ----------------------------------------------------------------------------------------


@dataclass
class RunAttempt:
  """One attempt of one run, with its jobs, its suites and the numbers `derive.py` gives.

  Built once per attempt and handed to every builder, so `phase_split` runs once even though
  four views quote it.
  """

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
  """Every row of one month, deduplicated, indexed by run and attempt.

  `runs` holds one entry per (run id, attempt). `merged_pr_runs` maps a merged pull request
  number to the one run that represents it. `kept_run_ids` is the set of runs the views draw:
  everything else stays in the store but is not published.
  """

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
    """True when this run is the one its pull request is drawn from.

    A pull request that pushed twice can leave two completed runs behind. Only one of them
    reaches the runs view and `pr/<n>.json`; the others are published in the per-run tables
    but must not be drawn as a second point for the same pull request.

    Args:
      run_id: The run.

    Returns:
      True for a scheduled run, which represents itself, and for the one run chosen for its
      pull request.
    """
    number = self.pr_of(run_id)
    if number is None:
      return True
    return self.merged_pr_runs.get(number) == run_id


def _is_kept(run: rows.RunRow) -> bool:
  """True when a run belongs in the published views.

  Two kinds are published: the run of a merged pull request, which is what every chart on the
  Main page is made of, and a scheduled run, which is the probe series and the daily
  full-history snapshot. A superseded run is never published - it was cancelled by a newer
  push and its numbers describe work that was abandoned - and neither is a pull request that
  did not merge. Their rows stay in the store; compaction trims them when the month closes.

  Args:
    run: The stored run row.

  Returns:
    True when the run should reach a view file.
  """
  if run.superseded:
    return False
  if run.is_merged_pr:
    return True
  return run.event == EVENT_SCHEDULE


def load_month(store: RowStore, month: str) -> MonthData:
  """Reads one month out of the store and indexes it.

  Two passes. The first reads runs, jobs, suites and rescues, which is everything needed to
  decide which runs are published and which pull requests merged. The second reads test rows,
  and keeps only the ones belonging to a merged pull request's run: a scheduled run's daily
  snapshot is every test in the repository and none of it is ever drawn, so carrying it into
  a view would cost megabytes for nothing.

  Args:
    store: The row store.
    month: "YYYY-MM".

  Returns:
    The month's rows, deduplicated by key and indexed by run and attempt.
  """
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
  """Files one deduplicated row under its run and attempt.

  A job, suite or test row whose run has no run row is dropped: the store writes a run row
  first, so a row with no run is a half-written tick, and publishing it would draw a run with
  no identity.

  Args:
    data: The month being loaded.
    record: The row.
  """
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
  """Records the one run that represents a merged pull request.

  A pull request can have several runs, one per push. The dashboard traces the final commit
  only, so the run whose head sha is the pull request's head sha wins; failing that, the
  newest run by creation time wins, which is the same run in every case measured.

  Args:
    data: The month being loaded.
    run: A run row of a merged pull request.
  """
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
  """Returns the test flavor and worker number a job ran, or (None, None).

  `derive.flavor_of` reads the last segment of any job name, so on a real run it also answers
  "Build Wheel" and "Pre-commit Linters". `flavors` is the allowlist from
  `derive.test_flavors`, which keeps only the names backed by a job that ran tests, and this
  checks against it before believing either reading.

  Args:
    job: The job payload.
    flavors: The flavors that really ran in this attempt.

  Returns:
    (flavor, worker). The worker is None for a flavor whose jobs are not named
    "Execute Tests (N)" - the Pathways flavors use a different name shape.
  """
  parsed = derive.parse_execute_tests_name(job.get("name"))
  if parsed is not None and parsed[0] in flavors:
    return parsed[0], parsed[1]
  candidate = derive.flavor_of(job)
  if candidate is not None and candidate in flavors:
    return candidate, None
  return None, None


def _job_record(job: Mapping[str, Any], flavors: set[str]) -> dict[str, Any]:
  """Builds the per-job fields every view spells the same way.

  Args:
    job: The job payload.
    flavors: The attempt's test flavors, for `_job_flavor`.

  Returns:
    A dict of the shared fields. Every duration comes from `derive.py` and is None when that
    job cannot answer - a carried-over job, or one that never held a runner.
  """
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
  """Builds the per-suite fields shared by the suites view and a pull request view.

  A nested suite - the decoupled pass, which runs inside cpu-unit worker 1 - gets null for
  duration, machine time and workers. Its `flavor` is cpu-unit, so measuring it against
  cpu-unit's jobs would report cpu-unit's half hour as the nested pass's twenty seconds. Its
  JUnit seconds are real and are carried; its wall clock is simply not measurable, and a null
  says so.

  A suite whose flavor has no stored jobs in this attempt gets the same treatment. There is
  nothing to measure it against, and `derive.machine_seconds([])` answers 0.0 - a real answer
  to a different question. Publishing that would draw a suite as having used no machine at
  all, so the four measured fields are null instead.

  Args:
    entry: The attempt the suite ran in.
    suite: The stored suite row.
    flavors: The attempt's test flavors.

  Returns:
    A dict of the shared fields.
  """
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
  """Counts, for every published run, how many other runs were in flight at the same time.

  A run's window is its own `created_at` - the moment GitHub accepted it - to the finish of
  its last job, which is `derive.phase_split(...).last_completed_at` of its highest stored
  attempt and falls back to the run's `updated_at` when no job row was stored. Two runs
  overlap when their windows intersect, whatever workflow, trigger or branch they belong to.
  That is the definition the dashboard prints as "N pipeline runs overlapped it"; it never
  says "concurrent". A re-run stretches the window, because the run really was open that
  whole time.

  The windows are closed at both ends, so two runs a second apart count as overlapping and a
  run whose window is a single instant - created and finished inside the same second, or
  stored with no jobs at all - still counts the runs it sat inside. Timestamps here have
  one-second resolution, so treating the boundary as open would report zero for a run that
  demonstrably shared the machines.

  Runs whose window falls outside this month are not counted, so a run in the first hour of a
  month undercounts. The alternative is reading a closed month on every tick.

  Args:
    data: The month's rows.

  Returns:
    run id -> how many other runs overlapped it. A run with no readable window is absent.
  """
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
  """Builds the runs view: one row per merged pull request, plus its jobs.

  This is what the Main page is drawn from. It replaces the mock's COMMITS, TIMES, RUN_IDS,
  TRIGGERS, JOBS and MACHINES in one file.

  The run row's phase minutes come from the FIRST stored attempt, because every trend chart
  compares like with like and a re-run only re-executes the jobs that failed. What the
  re-runs cost is not hidden: `rerun_machine_seconds` carries the runner time every later
  attempt spent. The job table is the first attempt too, and holds only jobs that asked for a
  real runner; the ten "Setup Parameters" jobs and the gates are seconds long on shared
  machines and are kept in the pull request view instead.

  Args:
    data: The month's rows.

  Returns:
    The view object, with a "runs" table and a "jobs" table.
  """
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
  """Builds the suites view: one row per suite per attempt of every published run.

  This is the test-suite health chart's history - the executed count T, the wall-clock
  duration D and the worker count W the dashboard reads together. It replaces the mock's
  TH_CATEGORIES.data, TH_COMMITS and THI_WORKERS.

  Scheduled runs are included and carry `event`, because the daily scheduled run is the only
  place a full test list is kept and the browser draws it as its own series, never mixed with
  pull request runs.

  Two numbers are deliberately separate. `workers_named` is how many "Execute Tests (N)" jobs
  the flavor was configured with, which is zero for the Pathways flavors because their jobs
  are named differently; `workers_ran` is how many of that flavor's jobs held a machine. A
  zero in the first with a two in the second is a naming fact, not a disappearance.

  A pull request that pushed twice can have two completed runs in the month. Both are
  published - the first push's suites really did run - so `is_representative` says which run
  is the one the runs view and `pr/<n>.json` describe. Drawing one point per pull request
  means filtering on it; drawing every run means ignoring it.

  Args:
    data: The month's rows.

  Returns:
    The view object, with a "suites" table.
  """
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
  """Builds the flaky view: every rescue event, and the tests that failed in each.

  A rescue is a job that failed on one attempt and passed on a later one. A job that failed
  and was never re-run is stored under the same shape with `rescued` false, and is carried
  here too, because the dashboard has to be able to say "#4940 failed and was never re-run -
  a failure, not a rescue" instead of quietly leaving it out of both counts.

  `wasted_seconds` is the failed job's own run time, read by `derive.run_seconds` from the
  stored job row of the attempt that failed. A rescue row does not carry the job's steps, so
  when that job row is missing the answer is null rather than a subtraction of two timestamps
  that would ignore a cancelled job's overrunning steps.

  Args:
    data: The month's rows.

  Returns:
    The view object, with a "rescues" table and a "rescue_tests" table.
  """
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
      for test in _failed_tests(failed_attempt, flavor, worker):
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
  """Returns the tests that failed in one job of one attempt.

  Args:
    entry: The attempt whose test rows to search, or None when it was not stored.
    flavor: The flavor the job ran. None means the job cannot be tied to a suite - a Pathways
      job publishes no test file at all - and the answer is an empty list, never a guess.
    worker: The worker number, when the job has one. A flavor's rows are split per worker, so
      naming the worker keeps a four-worker suite's other three out of this job's list.

  Returns:
    The failed and errored test rows, in stored order.
  """
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
  """Builds the queue view: how long each runner pool made each run wait.

  One row per pool per published run, from the first stored attempt. The pool is the runs-on
  label the jobs asked for - `linux-x86-ct6e-180-4tpu`, `linux-x86-n2-32` - because that is
  what a job queues against. Jobs that asked for no runner at all are not in any pool and are
  left out.

  A pull request with two completed runs in the month has a row for each; `is_representative`
  marks the one the runs view describes, exactly as it does in the suites view.

  Every pull request row also carries the nearest scheduled run's wait for the same pool.
  That is the probe: the same pipeline on a timer, so when a merge looks slow and the probe
  next to it was slow too, the queue was the infrastructure and not the change. "Nearest" is
  the smallest gap in creation time within `PROBE_MAX_GAP_SECONDS`; beyond that the two ran
  under different conditions and the probe columns are null.

  Args:
    data: The month's rows.

  Returns:
    The view object, with a "queue" table.
  """
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
  """Indexes the scheduled runs' longest wait per pool, sorted by time.

  Args:
    data: The month's rows.
    measured: Per run, per pool, the numbers `build_queue_view` already worked out.

  Returns:
    pool label -> [(created, run id, created_at string, longest wait)], oldest first.
  """
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
  """Finds the scheduled run closest in time to a merged pull request's run.

  Args:
    probes: One pool's scheduled runs, oldest first.
    created: When the pull request's run was created.

  Returns:
    The nearest probe, or None when there is none inside `PROBE_MAX_GAP_SECONDS`. A run in the
    first hours of a month can find no probe, because only this month is read.
  """
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
  """Builds the workflows view: one row per workflow per day.

  The clock time is the median of that day's runs, measured by `derive.run_wall_seconds` over
  each run's first stored attempt, so a re-run does not make a workflow look slower than it
  was. The machine minutes are the opposite question and are summed over every stored attempt,
  because a re-run really did hold runners. Superseded runs are in neither: they were
  cancelled by a newer push and their numbers describe abandoned work.

  Every workflow the collector stores appears here, not only the pipeline: the nightly image
  builds are on the same card.

  Args:
    data: The month's rows.

  Returns:
    The view object, with a "workflows" table.
  """
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
  """Builds one merged pull request's file: every attempt, job, step, suite and test stored.

  This is the file a click fetches. It replaces the mock's ATTEMPT_INFO, TESTS, TEST_COUNTS,
  STEPS and error rows, and it is shared by the commit modal and the Single PR page, so a
  click on the same pull request from either place costs one request.

  Steps travel as their two timestamps rather than as a duration. `derive.py` has no public
  "seconds of a named step" and this module will not write a second one; subtracting two
  ISO-8601 strings is one line in the browser.

  `updated_at` is the newest `collected_at` of the rows behind the file, not the moment it was
  built, so a tick that re-derives an unchanged pull request writes the same bytes and the
  browser's cached copy stays valid.

  Args:
    pr_number: The merged pull request.
    data: The month's rows.

  Returns:
    The view object.

  Raises:
    ViewError: The month holds no run for that pull request.
  """
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
  """Builds one error row: the first step of a failed job that failed.

  Nothing is summarised. The step's own name is quoted, and the failing tests of the same job
  are in the tests table with their JUnit messages verbatim.

  Args:
    attempt: The attempt the job failed in.
    job: The job payload.
    shared: The job's shared fields, already built by `_job_record`.

  Returns:
    The error record. Every step field is None for a job that failed with no steps at all -
    a job cancelled while queued has an empty steps list, and that is the whole story.
  """
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
  """Builds meta.json, the first file the browser reads and the only one with a build time.

  It answers four questions before anything else is fetched: how old is this data, which
  month files exist for each group, how many rows each holds, and which pull request files
  have changed since the browser last looked.

  Args:
    generated_at: When this tick ran, ISO-8601 UTC.
    months: group -> the month keys that have a file, ascending.
    counts: group -> month -> table -> row count.
    pull_requests: pull request number -> the newest `collected_at` behind its file.
    uncollected: How many runs the store has seen but not collected - attempts still running,
      and attempts it could not read. A number here is normal; a number that keeps growing is
      not, and the dashboard shows it rather than quietly drawing a shorter history.
    window_days: How far back the views reach.

  Returns:
    The meta object.
  """
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
  """Wraps a group's tables in the envelope every month file shares.

  There is deliberately no timestamp in here. A month file whose rows have not changed
  serialises to the same bytes on every tick, so git records nothing and the browser's cached
  copy stays valid. The build time lives in meta.json alone.
  """
  return {"schema": SCHEMA_VERSION, "group": group, "month": month, "tables": dict(tables)}


def _dumps(obj: Any) -> str:
  """Serialises a view the one way this module ever serialises anything.

  Compact separators, no ASCII escaping, key order as written, one trailing newline. Two ticks
  that build the same view produce the same bytes.
  """
  return json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"


def write_view_file(path: Path, obj: Any) -> str:
  """Writes one view file, and says whether it actually changed.

  The file is written to a temporary name in the same directory and moved into place, so a
  reader never sees half a file and a tick that dies leaves the previous version intact.

  Args:
    path: Where the file goes.
    obj: The view object.

  Returns:
    "written" when the bytes on disk changed, "unchanged" when they did not. An unchanged file
    is not rewritten at all: its modification time stays put and git sees nothing to commit.
  """
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
  """Reads a view file back, or returns None when it is missing or unreadable.

  Used for the months this tick did not rebuild: their row counts still have to reach
  meta.json, and reading a finished view file is much cheaper than re-reading their rows.
  """
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
  """Builds every file the browser reads, and reports exactly what it touched.

  Only the open months are rebuilt - today's, and the one before it for the first couple of
  days of a month. A closed month's file is never read and never rewritten. Its row counts
  still reach meta.json, taken from the finished file on disk, which costs one small read
  instead of a month of rows.

  Nothing is deleted here. Pruning a month that has fallen out of the window, and pruning the
  pull request files with it, belongs to the store's compaction step; a file that has already
  gone simply stops being advertised in meta.json.

  A pull request file is written by the month that holds its newest run. A month rebuilt on
  its own leaves a newer month's file alone rather than replacing it with an abandoned push.

  Args:
    store: The row store.
    out_dir: The output directory, the one `--out` names. Files go under `<out_dir>/views/`.
    today: The day this tick is running, used to pick the open months and the window.
    months: Rebuild exactly these months instead of the default. A tick that knows which
      months it appended to should pass them; anything not in the store is ignored.
    uncollected: How many run attempts have been seen but not collected. Read from
      `store.load_state().pending_count` when not given.
    generated_at: The build time for meta.json. Defaults to now, in the same format the rows
      use.

  Returns:
    A summary:
      generated_at, out_dir, window_days
      months_rebuilt / months_skipped - what was read and what was left alone
      written / unchanged - view files whose bytes changed, and whose did not
      pr_written / pr_unchanged - the same, for pull request files
      counts - group -> month -> table -> rows
      uncollected_runs, meta - the state meta.json was given
  """
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
  """Says whether the pull request file already on disk describes a newer run than this one.

  A pull request can have completed runs in two months - a push late on the 31st, another on
  the 1st. Each month claims the pull request for the only run it can see, so a tick that
  rebuilds the older month alone would otherwise overwrite `pr/<n>.json` with the abandoned
  push, and once the newer month has closed nothing would ever repair it.

  Args:
    path: The pull request file.
    candidate: The view this build produced for it.

  Returns:
    The stored file's `updated_at` when it must be kept, or None when it should be written.
  """
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
  """Carries meta.json's pull request list forward across a tick.

  A pull request whose month is closed keeps its file and its `updated_at` from the last time
  it was built. Re-reading six hundred files every tick to recover a timestamp that cannot
  have changed would cost more than the whole rest of the build. An entry whose file has gone
  is dropped, so a prune heals itself on the next tick.

  Args:
    views_root: The `views/` directory.

  Returns:
    pull request number -> its stored `updated_at`.
  """
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
