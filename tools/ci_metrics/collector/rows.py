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

"""The shapes of the rows the collector stores, and the builders that fill them in."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import quote

# Row schema version. Bump it when a field changes meaning, not when one is added with a
# default: `from_json` reads any row whose version is at or below this one.
ROW_VERSION = 1

# Row kinds. The kind travels inside the JSON so `from_json` knows what to rebuild, and the
# writer can use it to pick a file.
KIND_RUN = "run"
KIND_JOB = "job"
KIND_SUITE = "suite"
KIND_TEST = "test"
KIND_RESCUE = "rescue"

# Joins the parts of a key. Parts are percent-encoded first, so this byte cannot appear
# inside one.
KEY_SEPARATOR = "|"

# The only fields the jobs endpoint reports for a step. Pinning them keeps every stored job
# row the same shape, and keeps the JSON free of anything that would not round-trip.
STEP_FIELDS = ("name", "number", "status", "conclusion", "started_at", "completed_at")

# A job conclusion of "failure" followed by "success" is a rescue. Nothing else is: a
# cancelled job that succeeds next time was never a failure.
CONCLUSION_FAILURE = "failure"
CONCLUSION_SUCCESS = "success"


class RowError(ValueError):
  """Raised when a payload cannot be turned into a row, or a stored row cannot be read back."""


def utc_now_iso() -> str:
  """Returns the current UTC time in the format GitHub uses for its own timestamps."""
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestCaseLike(Protocol):
  """One `<testcase>` as `junit.TestRow` exposes it."""

  name: str
  classname: str
  duration: float
  status: str
  failure_message: str | None
  worker: int | None


class SuiteResultLike(Protocol):
  """One suite's counts as `junit.SuiteResult` exposes them."""

  collected: int
  skipped: int
  executed: int
  junit_seconds: float
  tests: list[Any]
  failed: int
  errored: int
  reported_tests: int | None
  suite_seconds: float | None
  files: tuple[str, ...]


class SuiteEntryLike(Protocol):
  """One suite's outcome as `junit.SuiteEntry` exposes it: a result, or the reason for none."""

  suite_id: str
  result: SuiteResultLike | None
  reason: str | None
  nested_in: str | None
  per_worker: dict[int, SuiteResultLike]
  missing_workers: dict[int, str]


def _key_part(value: object) -> str:
  """Encodes one part of a key so it cannot be confused with another part."""
  if value is None:
    return ""
  return quote(str(value), safe="")


def _row_key(kind: str, *parts: object) -> str:
  """Builds a row key from its kind and its parts."""
  return KEY_SEPARATOR.join([kind, *(_key_part(part) for part in parts)])


def _need_int(payload: dict[str, Any], key: str, what: str) -> int:
  """Reads a required integer field."""
  raw = payload.get(key)
  if raw is None:
    raise RowError(f"{what} has no {key!r} field; it holds {sorted(payload)[:8]}.")
  try:
    return int(raw)
  except (TypeError, ValueError) as exc:
    raise RowError(f"{what} has a non-numeric {key}={raw!r}.") from exc


def _opt_int(payload: dict[str, Any], key: str) -> int | None:
  """Reads an optional integer field, returning None when it is absent or unreadable."""
  raw = payload.get(key)
  if raw is None:
    return None
  try:
    return int(raw)
  except (TypeError, ValueError):
    return None


def _need_str(payload: dict[str, Any], key: str, what: str) -> str:
  """Reads a required string field."""
  raw = payload.get(key)
  if raw is None:
    raise RowError(f"{what} has no {key!r} field; it holds {sorted(payload)[:8]}.")
  return str(raw)


def _opt_str(payload: dict[str, Any], key: str) -> str | None:
  """Reads an optional string field, keeping None as None rather than turning it into "None"."""
  raw = payload.get(key)
  if raw is None:
    return None
  return str(raw)


def _opt_bool(payload: dict[str, Any], key: str) -> bool | None:
  """Reads an optional boolean field. Absent stays None, which is not the same as False."""
  raw = payload.get(key)
  if raw is None:
    return None
  return bool(raw)


def _opt_nested_str(payload: dict[str, Any], outer: str, inner: str) -> str | None:
  """Reads `payload[outer][inner]` as a string, returning None when either level is missing."""
  nested = payload.get(outer)
  if not isinstance(nested, dict):
    return None
  return _opt_str(nested, inner)


def _str_list(payload: dict[str, Any], key: str) -> list[str]:
  """Reads a list-of-strings field, e.g. a job's runs-on `labels`."""
  raw = payload.get(key)
  if not isinstance(raw, list):
    return []
  return [str(item) for item in raw]


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
  """Copies a job's steps, keeping only the fields the API reports for them."""
  raw = job.get("steps")
  if not isinstance(raw, list):
    return []

  steps: list[dict[str, Any]] = []
  for index, step in enumerate(raw):
    if not isinstance(step, dict):
      raise RowError(f"job {job.get('id')} step {index} is a {type(step).__name__}, expected an object.")
    steps.append(
        {
            "name": _opt_str(step, "name"),
            "number": _opt_int(step, "number"),
            "status": _opt_str(step, "status"),
            "conclusion": _opt_str(step, "conclusion"),
            "started_at": _opt_str(step, "started_at"),
            "completed_at": _opt_str(step, "completed_at"),
        }
    )
  return steps


@dataclass
class RunRow:
  """One attempt of one workflow run, plus the pull request it belongs to."""

  run_id: int
  attempt: int
  workflow_id: int | None = None
  workflow_name: str | None = None
  workflow_path: str | None = None
  run_number: int | None = None
  event: str | None = None
  status: str | None = None
  conclusion: str | None = None
  created_at: str | None = None
  run_started_at: str | None = None
  updated_at: str | None = None
  head_branch: str | None = None
  head_sha: str | None = None
  repository: str | None = None
  head_repository: str | None = None
  actor: str | None = None
  triggering_actor: str | None = None
  display_title: str | None = None
  html_url: str | None = None
  previous_attempt_url: str | None = None
  superseded: bool | None = None
  pr_number: int | None = None
  pr_state: str | None = None
  pr_title: str | None = None
  pr_merged_at: str | None = None
  pr_created_at: str | None = None
  pr_closed_at: str | None = None
  pr_head_sha: str | None = None
  pr_head_label: str | None = None
  pr_base_ref: str | None = None
  pr_merge_commit_sha: str | None = None
  pr_user: str | None = None
  pr_html_url: str | None = None
  pr_draft: bool | None = None
  v: int = ROW_VERSION
  collected_at: str = field(default_factory=utc_now_iso)

  def key(self) -> str:
    """Returns `run|<run_id>|<attempt>`."""
    return _row_key(KIND_RUN, self.run_id, self.attempt)

  @property
  def is_fork(self) -> bool | None:
    """True when the branch lives in another repository, None when either name is unknown."""
    if self.repository is None or self.head_repository is None:
      return None
    return self.repository != self.head_repository

  @property
  def is_merged_pr(self) -> bool:
    """True when this run belongs to a pull request that merged."""
    return self.pr_merged_at is not None


@dataclass
class JobRow:
  """One job of one attempt, with its steps."""

  job_id: int
  run_id: int
  attempt: int
  name: str
  status: str | None = None
  conclusion: str | None = None
  created_at: str | None = None
  started_at: str | None = None
  completed_at: str | None = None
  labels: list[str] = field(default_factory=list)
  runner_id: int | None = None
  runner_name: str | None = None
  runner_group_id: int | None = None
  runner_group_name: str | None = None
  workflow_name: str | None = None
  head_branch: str | None = None
  head_sha: str | None = None
  html_url: str | None = None
  steps: list[dict[str, Any]] = field(default_factory=list)
  v: int = ROW_VERSION
  collected_at: str = field(default_factory=utc_now_iso)

  def key(self) -> str:
    """Returns `job|<run_id>|<attempt>|<job_id>`."""
    return _row_key(KIND_JOB, self.run_id, self.attempt, self.job_id)


@dataclass
class SuiteRow:
  """One suite's totals in one attempt: the per-flavor numbers, or the reason there are none."""

  run_id: int
  attempt: int
  suite_id: str
  flavor: str
  nested_in: str | None = None
  collected: int | None = None
  skipped: int | None = None
  executed: int | None = None
  failed: int | None = None
  errored: int | None = None
  junit_seconds: float | None = None
  reported_tests: int | None = None
  suite_seconds: float | None = None
  reason: str | None = None
  is_partial: bool = False
  missing_workers: list[dict[str, Any]] = field(default_factory=list)
  published_workers: list[int] = field(default_factory=list)
  files: list[str] = field(default_factory=list)
  v: int = ROW_VERSION
  collected_at: str = field(default_factory=utc_now_iso)

  def key(self) -> str:
    """Returns `suite|<run_id>|<attempt>|<suite_id>`."""
    return _row_key(KIND_SUITE, self.run_id, self.attempt, self.suite_id)

  @property
  def has_result(self) -> bool:
    """True when the suite published something, whether or not every worker did."""
    return self.collected is not None


@dataclass
class TestRow:
  """One `<testcase>` of one suite in one attempt."""

  run_id: int
  attempt: int
  suite_id: str
  flavor: str
  worker: int | None
  classname: str
  name: str
  status: str
  duration: float
  failure_message: str | None = None
  nested_in: str | None = None
  suite_partial: bool = False
  v: int = ROW_VERSION
  collected_at: str = field(default_factory=utc_now_iso)

  def key(self) -> str:
    """Returns `test|<run_id>|<attempt>|<suite_id>|<worker>|<classname>|<name>`."""
    return _row_key(KIND_TEST, self.run_id, self.attempt, self.suite_id, self.worker, self.classname, self.name)


@dataclass
class RescueRow:
  """One job name that failed and then passed on a re-run, or failed and never recovered."""

  # A job is followed across attempts by its name: GitHub gives every job a new id in every attempt, even a
  # job it did not re-run. That is why the key holds `job_name` and not a job id.
  run_id: int
  job_name: str
  rescued: bool
  failed_attempt: int
  failed_job_id: int | None = None
  failed_conclusion: str | None = None
  failed_created_at: str | None = None
  failed_started_at: str | None = None
  failed_completed_at: str | None = None
  rescued_attempt: int | None = None
  rescued_job_id: int | None = None
  rescued_started_at: str | None = None
  rescued_completed_at: str | None = None
  final_attempt: int | None = None
  final_conclusion: str | None = None
  attempts_seen: list[int] = field(default_factory=list)
  labels: list[str] = field(default_factory=list)
  head_branch: str | None = None
  workflow_name: str | None = None
  event: str | None = None
  html_url: str | None = None
  v: int = ROW_VERSION
  collected_at: str = field(default_factory=utc_now_iso)

  def key(self) -> str:
    """Returns `rescue|<run_id>|<job_name>|<failed_attempt>`."""
    return _row_key(KIND_RESCUE, self.run_id, self.job_name, self.failed_attempt)

  @property
  def rerun_after_failure(self) -> bool:
    """True when a later attempt ran this name again, whatever the outcome was."""
    return self.final_attempt is not None and self.final_attempt > self.failed_attempt


def run_row(run: dict[str, Any], pr: dict[str, Any] | None = None, collected_at: str | None = None) -> RunRow:
  """Builds the row for one attempt of one workflow run."""
  row = RunRow(
      run_id=_need_int(run, "id", "run"),
      attempt=_need_int(run, "run_attempt", f"run {run.get('id')}"),
      workflow_id=_opt_int(run, "workflow_id"),
      workflow_name=_opt_str(run, "name"),
      workflow_path=_opt_str(run, "path"),
      run_number=_opt_int(run, "run_number"),
      event=_opt_str(run, "event"),
      status=_opt_str(run, "status"),
      conclusion=_opt_str(run, "conclusion"),
      created_at=_opt_str(run, "created_at"),
      run_started_at=_opt_str(run, "run_started_at"),
      updated_at=_opt_str(run, "updated_at"),
      head_branch=_opt_str(run, "head_branch"),
      head_sha=_opt_str(run, "head_sha"),
      repository=_opt_nested_str(run, "repository", "full_name"),
      head_repository=_opt_nested_str(run, "head_repository", "full_name"),
      actor=_opt_nested_str(run, "actor", "login"),
      triggering_actor=_opt_nested_str(run, "triggering_actor", "login"),
      display_title=_opt_str(run, "display_title"),
      html_url=_opt_str(run, "html_url"),
      previous_attempt_url=_opt_str(run, "previous_attempt_url"),
      superseded=_opt_bool(run, "superseded"),
  )
  if collected_at is not None:
    row.collected_at = collected_at
  if pr is not None:
    _fill_pull_request(row, pr)
  return row


def _fill_pull_request(row: RunRow, pr: dict[str, Any]) -> None:
  """Copies the linked pull request's fields onto a run row."""
  row.pr_number = _opt_int(pr, "number")
  row.pr_state = _opt_str(pr, "state")
  row.pr_title = _opt_str(pr, "title")
  row.pr_merged_at = _opt_str(pr, "merged_at")
  row.pr_created_at = _opt_str(pr, "created_at")
  row.pr_closed_at = _opt_str(pr, "closed_at")
  row.pr_head_sha = _opt_nested_str(pr, "head", "sha")
  row.pr_head_label = _opt_nested_str(pr, "head", "label")
  row.pr_base_ref = _opt_nested_str(pr, "base", "ref")
  row.pr_merge_commit_sha = _opt_str(pr, "merge_commit_sha")
  row.pr_user = _opt_nested_str(pr, "user", "login")
  row.pr_html_url = _opt_str(pr, "html_url")
  row.pr_draft = _opt_bool(pr, "draft")


def job_row(run: dict[str, Any], job: dict[str, Any], collected_at: str | None = None) -> JobRow:
  """Builds the row for one job of one attempt."""
  run_id = _opt_int(job, "run_id")
  if run_id is None:
    run_id = _need_int(run, "id", "run")
  else:
    run_ref = _opt_int(run, "id")
    if run_ref is not None and run_ref != run_id:
      raise RowError(f"job {job.get('id')} belongs to run {run_id}, but run {run_ref} was passed with it.")

  attempt = _opt_int(job, "run_attempt")
  if attempt is None:
    attempt = _need_int(run, "run_attempt", f"run {run_id}")

  row = JobRow(
      job_id=_need_int(job, "id", "job"),
      run_id=run_id,
      attempt=attempt,
      name=_need_str(job, "name", f"job {job.get('id')}"),
      status=_opt_str(job, "status"),
      conclusion=_opt_str(job, "conclusion"),
      created_at=_opt_str(job, "created_at"),
      started_at=_opt_str(job, "started_at"),
      completed_at=_opt_str(job, "completed_at"),
      labels=_str_list(job, "labels"),
      runner_id=_opt_int(job, "runner_id"),
      runner_name=_opt_str(job, "runner_name"),
      runner_group_id=_opt_int(job, "runner_group_id"),
      runner_group_name=_opt_str(job, "runner_group_name"),
      workflow_name=_opt_str(job, "workflow_name"),
      head_branch=_opt_str(job, "head_branch"),
      head_sha=_opt_str(job, "head_sha"),
      html_url=_opt_str(job, "html_url"),
      steps=_steps(job),
  )
  if collected_at is not None:
    row.collected_at = collected_at
  return row


def suite_row(run: dict[str, Any], entry: SuiteEntryLike, collected_at: str | None = None) -> SuiteRow:
  """Builds the totals row for one suite of one attempt."""
  run_id = _need_int(run, "id", "run")
  attempt = _need_int(run, "run_attempt", f"run {run_id}")
  result = entry.result
  missing = dict(entry.missing_workers or {})

  row = SuiteRow(
      run_id=run_id,
      attempt=attempt,
      suite_id=entry.suite_id,
      flavor=entry.nested_in or entry.suite_id,
      nested_in=entry.nested_in,
      reason=entry.reason,
      is_partial=result is not None and bool(missing),
      missing_workers=[{"worker": int(worker), "reason": str(reason)} for worker, reason in sorted(missing.items())],
      published_workers=sorted(int(worker) for worker in (entry.per_worker or {})),
  )
  if result is not None:
    row.collected = result.collected
    row.skipped = result.skipped
    row.executed = result.executed
    row.failed = result.failed
    row.errored = result.errored
    row.junit_seconds = result.junit_seconds
    row.reported_tests = result.reported_tests
    row.suite_seconds = result.suite_seconds
    row.files = list(result.files)
  if collected_at is not None:
    row.collected_at = collected_at
  return row


def test_rows(
    run: dict[str, Any],
    flavor: str,
    worker: int,
    suite_result: SuiteResultLike,
    suite_id: str | None = None,
    nested_in: str | None = None,
    suite_partial: bool = False,
    collected_at: str | None = None,
) -> list[TestRow]:
  """Builds one row per `<testcase>` of one suite."""
  run_id = _need_int(run, "id", "run")
  attempt = _need_int(run, "run_attempt", f"run {run_id}")
  written_at = collected_at if collected_at is not None else utc_now_iso()

  rows: list[TestRow] = []
  for case in suite_result.tests:
    case_worker = getattr(case, "worker", None)
    rows.append(
        TestRow(
            run_id=run_id,
            attempt=attempt,
            suite_id=suite_id or flavor,
            flavor=flavor,
            worker=case_worker if case_worker is not None else worker,
            classname=case.classname,
            name=case.name,
            status=case.status,
            duration=case.duration,
            failure_message=case.failure_message,
            nested_in=nested_in,
            suite_partial=suite_partial,
            collected_at=written_at,
        )
    )
  return rows


def _job_history(attempts_jobs: dict[int, list[dict[str, Any]]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
  """Groups every attempt's jobs by job name, in attempt order."""
  history: dict[str, list[tuple[int, dict[str, Any]]]] = {}
  for raw_attempt in sorted(attempts_jobs, key=_attempt_number):
    attempt = _attempt_number(raw_attempt)
    jobs = attempts_jobs[raw_attempt]
    if not isinstance(jobs, list):
      raise RowError(f"attempt {attempt} holds a {type(jobs).__name__}, expected a list of jobs.")
    seen: set[str] = set()
    for job in jobs:
      if not isinstance(job, dict):
        raise RowError(f"attempt {attempt} holds a {type(job).__name__} where a job object was expected.")
      name = _need_str(job, "name", f"job {job.get('id')} of attempt {attempt}")
      if name in seen:
        continue
      seen.add(name)
      history.setdefault(name, []).append((attempt, job))
  return history


def _attempt_number(value: object) -> int:
  """Reads an attempt number, accepting the string keys a JSON round trip leaves behind."""
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    raise RowError(f"attempt key {value!r} is not a number.")
  try:
    return int(value)
  except (TypeError, ValueError) as exc:
    raise RowError(f"attempt key {value!r} is not a number.") from exc


def _rescue_row(
    run: dict[str, Any],
    name: str,
    observations: list[tuple[int, dict[str, Any]]],
    failed: tuple[int, dict[str, Any]],
    passed: tuple[int, dict[str, Any]] | None,
    collected_at: str | None,
) -> RescueRow:
  """Builds one rescue row from a failure and, when there was one, the attempt that passed."""
  failed_attempt, failed_job = failed
  final_attempt, final_job = observations[-1]
  row = RescueRow(
      run_id=_need_int(run, "id", "run"),
      job_name=name,
      rescued=passed is not None,
      failed_attempt=failed_attempt,
      failed_job_id=_opt_int(failed_job, "id"),
      failed_conclusion=_opt_str(failed_job, "conclusion"),
      failed_created_at=_opt_str(failed_job, "created_at"),
      failed_started_at=_opt_str(failed_job, "started_at"),
      failed_completed_at=_opt_str(failed_job, "completed_at"),
      final_attempt=final_attempt,
      final_conclusion=_opt_str(final_job, "conclusion"),
      attempts_seen=[attempt for attempt, _ in observations],
      labels=_str_list(failed_job, "labels"),
      head_branch=_opt_str(run, "head_branch"),
      workflow_name=_opt_str(run, "name"),
      event=_opt_str(run, "event"),
      html_url=_opt_str(failed_job, "html_url"),
  )
  if passed is not None:
    passed_attempt, passed_job = passed
    row.rescued_attempt = passed_attempt
    row.rescued_job_id = _opt_int(passed_job, "id")
    row.rescued_started_at = _opt_str(passed_job, "started_at")
    row.rescued_completed_at = _opt_str(passed_job, "completed_at")
  if collected_at is not None:
    row.collected_at = collected_at
  return row


def rescue_rows(
    run: dict[str, Any],
    attempts_jobs: dict[int, list[dict[str, Any]]],
    collected_at: str | None = None,
) -> list[RescueRow]:
  """Finds the jobs that failed and then passed on a re-run."""
  rows: list[RescueRow] = []
  for name, observations in sorted(_job_history(attempts_jobs).items()):
    for earlier, later in _rescue_pairs(observations):
      rows.append(_rescue_row(run, name, observations, earlier, later, collected_at))
  return rows


def _rescue_pairs(
    observations: list[tuple[int, dict[str, Any]]],
) -> list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]]:
  """Pairs each failure with the next attempt that ran the job to a verdict and passed."""
  pairs: list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]] = []
  pending: tuple[int, dict[str, Any]] | None = None
  for observation in observations:
    conclusion = observation[1].get("conclusion")
    if conclusion == CONCLUSION_FAILURE:
      pending = observation
    elif conclusion == CONCLUSION_SUCCESS and pending is not None:
      pairs.append((pending, observation))
      pending = None
  return pairs


def failed_never_rescued_rows(
    run: dict[str, Any],
    attempts_jobs: dict[int, list[dict[str, Any]]],
    collected_at: str | None = None,
) -> list[RescueRow]:
  """Builds the rows for jobs that failed and never passed, the other half of the flaky card."""
  rows: list[RescueRow] = []
  for name, observations in sorted(_job_history(attempts_jobs).items()):
    if observations[-1][1].get("conclusion") != CONCLUSION_FAILURE:
      continue
    first_of_streak = len(observations) - 1
    while first_of_streak > 0 and observations[first_of_streak - 1][1].get("conclusion") == CONCLUSION_FAILURE:
      first_of_streak -= 1
    rows.append(_rescue_row(run, name, observations, observations[first_of_streak], None, collected_at))
  return rows


# Every stored row type, by the kind that travels inside its JSON.
_ROW_TYPES: dict[str, type] = {
    KIND_RUN: RunRow,
    KIND_JOB: JobRow,
    KIND_SUITE: SuiteRow,
    KIND_TEST: TestRow,
    KIND_RESCUE: RescueRow,
}
_KIND_BY_TYPE: dict[type, str] = {row_type: kind for kind, row_type in _ROW_TYPES.items()}

Row = RunRow | JobRow | SuiteRow | TestRow | RescueRow


def row_kind(row: Row) -> str:
  """Returns the kind string of a row, e.g. "job"."""
  kind = _KIND_BY_TYPE.get(type(row))
  if kind is None:
    raise RowError(f"{type(row).__name__} is not a stored row type.")
  return kind


def to_json(row: Row) -> dict[str, Any]:
  """Turns a row into the JSON object that is written to the store."""
  payload: dict[str, Any] = {"kind": row_kind(row)}
  payload.update(dataclasses.asdict(row))
  return payload


def from_json(payload: dict[str, Any]) -> Row:
  """Rebuilds a row from a stored JSON object."""
  kind = payload.get("kind")
  row_type = _ROW_TYPES.get(str(kind))
  if row_type is None:
    raise RowError(f"row kind {kind!r} is not one of {sorted(_ROW_TYPES)}.")

  version = payload.get("v")
  if not isinstance(version, int) or isinstance(version, bool):
    raise RowError(f"{kind} row has a non-integer v={version!r}.")
  if version > ROW_VERSION:
    raise RowError(f"{kind} row was written with schema version {version}; this collector understands {ROW_VERSION}.")

  names = {f.name for f in dataclasses.fields(row_type)}
  given = set(payload) - {"kind"}
  missing = sorted(names - given)
  unexpected = sorted(given - names)
  if missing or unexpected:
    raise RowError(f"{kind} row does not match schema version {ROW_VERSION}: missing {missing}, unexpected {unexpected}.")

  return row_type(**{name: payload[name] for name in names})
