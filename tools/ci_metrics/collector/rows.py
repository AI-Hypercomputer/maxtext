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

"""The shapes of the rows the collector stores, and the builders that fill them in.

This module is the boundary between what GitHub said and what the dashboard shows. It holds
one dataclass per stored record and one builder per dataclass, and it does nothing else:

  * **No network.** Nothing here fetches anything; every builder is handed a payload that
    `runs.py` or `junit.py` already read.
  * **No arithmetic.** Not one duration, rate or total is computed here. A row carries the
    raw `created_at`, `started_at` and `completed_at` strings; `derive.py` turns them into
    seconds. That split is deliberate: the stored history stays faithful to what the API
    said, so when a rule changes - and the suite-duration rule already changed once - the new
    numbers can be recomputed over rows collected years earlier instead of being lost.
  * **Missing is None, never zero.** A suite that published nothing stores None counts and a
    reason code. A zero would draw a "tests vanished" alarm for a run that simply uploaded
    no file.

Every row carries two housekeeping fields:

  * `v` - the row schema version, currently 1. `from_json` refuses a row written by a newer
    version rather than reading it wrong.
  * `collected_at` - when the collector wrote the row, ISO-8601 UTC to the second, the same
    format GitHub uses. The store is append-only: a correction is a second row with the same
    key and a later `collected_at`, and readers keep the last row per key.

Key formats
-----------

`key()` identifies a row across ticks, so the writer can recognise a row it already has.
Every part is percent-encoded before it is joined with "|", which is why a job name full of
slashes and brackets or a pytest parameter id full of punctuation can never run two parts
together or fake a separator:

  * run     `run|<run_id>|<attempt>`
  * job     `job|<run_id>|<attempt>|<job_id>`
  * suite   `suite|<run_id>|<attempt>|<suite_id>`
  * test    `test|<run_id>|<attempt>|<suite_id>|<worker>|<classname>|<name>`
  * rescue  `rescue|<run_id>|<job_name>`

The test key uses the suite id rather than the flavor. For every real flavor the two are the
same string, but the nested `decoupled` pass runs the same 50 tests a second time inside
cpu-unit worker 1, so keying it by flavor would make each decoupled row overwrite the
cpu-unit row of the same test.

Job identity is the job NAME, not the job id: GitHub mints a fresh id for every job in every
attempt, including the jobs it did not re-run. That is why the rescue key holds the name.
"""

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
  """Raised when a payload cannot be turned into a row, or a stored row cannot be read back.

  The message always names the field at fault, so a collector tick that dies on one odd
  payload says which one it was.
  """


def utc_now_iso() -> str:
  """Returns the current UTC time in the format GitHub uses for its own timestamps.

  Seconds resolution, "Z" suffix, no offset. Ticks are hours apart, so two rows of the same
  key never share a `collected_at`.

  Returns:
    An ISO-8601 UTC timestamp, e.g. "2026-09-01T15:04:05Z".
  """
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestCaseLike(Protocol):
  """One `<testcase>` as `junit.TestRow` exposes it.

  Declared structurally so `rows.py` does not import `junit.py`: the two modules each define
  a `TestRow`, and a test can hand these builders a stub.
  """

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
  """Encodes one part of a key so it cannot be confused with another part.

  Args:
    value: The part. None becomes an empty part, which is what an unknown worker number is.

  Returns:
    The value percent-encoded, leaving only unreserved ASCII characters as they were.
  """
  if value is None:
    return ""
  return quote(str(value), safe="")


def _row_key(kind: str, *parts: object) -> str:
  """Builds a row key from its kind and its parts.

  Args:
    kind: One of the KIND_* constants.
    *parts: The identifying values, in the order the key format documents.

  Returns:
    The key, e.g. "job|33468578834|1|99733940992".
  """
  return KEY_SEPARATOR.join([kind, *(_key_part(part) for part in parts)])


def _need_int(payload: dict[str, Any], key: str, what: str) -> int:
  """Reads a required integer field.

  Args:
    payload: The API object to read from.
    key: Field name.
    what: What the payload is, for the error message.

  Returns:
    The field as an int.

  Raises:
    RowError: The field is absent, None, or not a number.
  """
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
  """Reads a required string field.

  Args:
    payload: The API object to read from.
    key: Field name.
    what: What the payload is, for the error message.

  Returns:
    The field as a string.

  Raises:
    RowError: The field is absent or None.
  """
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
  """Reads `payload[outer][inner]` as a string, returning None when either level is missing.

  Args:
    payload: The API object to read from.
    outer: Name of the nested object, e.g. "head_repository".
    inner: Field inside it, e.g. "full_name".

  Returns:
    The value as a string, or None.
  """
  nested = payload.get(outer)
  if not isinstance(nested, dict):
    return None
  return _opt_str(nested, inner)


def _str_list(payload: dict[str, Any], key: str) -> list[str]:
  """Reads a list-of-strings field, e.g. a job's runs-on `labels`.

  Args:
    payload: The API object to read from.
    key: Field name.

  Returns:
    The values as strings. An absent field and an empty list both give an empty list: a
    skipped job really does carry `labels: []`, and neither shape says anything else.
  """
  raw = payload.get(key)
  if not isinstance(raw, list):
    return []
  return [str(item) for item in raw]


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
  """Copies a job's steps, keeping only the fields the API reports for them.

  The step timestamps are the only place the "Run Tests" span lives, and that span is the
  suite duration, so they have to be stored. They are copied verbatim, never differenced.

  An empty list is a real answer, not a gap: a cancelled job that never held a runner reports
  `steps: []`, which is how the collector tells "ran nothing" from "ran and failed".

  Args:
    job: One job object from the jobs endpoint.

  Returns:
    One flat dict per step, each with exactly the STEP_FIELDS keys.

  Raises:
    RowError: A step is not a JSON object.
  """
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
  """One attempt of one workflow run, plus the pull request it belongs to.

  Key: `run|<run_id>|<attempt>`. A re-run is a new attempt of the same run id, so the attempt
  is part of the key; without it attempt 2 would overwrite attempt 1.

  Attributes:
    run_id: The workflow run id.
    attempt: `run_attempt` of this row. Not stable over time - a run grows attempts - so the
      collector re-reads the run before deciding how many attempts to fetch.
    workflow_id: Numeric workflow id, the field the allowlist matches on.
    workflow_name: Display name of the workflow, e.g. "MaxText Package Tests".
    workflow_path: Path of the workflow file.
    run_number: The per-workflow run number GitHub shows in its UI.
    event: The trigger: "pull_request", "push", "schedule" or "workflow_dispatch". Every
      chart's x axis is merged pull requests, so this is what separates the series.
    status: "completed", "in_progress" or "queued". Only completed attempts are stored.
    conclusion: "success", "failure", "cancelled", "action_required", or None while running.
    created_at: When the run was created, ISO-8601 UTC.
    run_started_at: When the run started, ISO-8601 UTC.
    updated_at: When the run last changed, ISO-8601 UTC.
    head_branch: Branch the run was triggered on.
    head_sha: Commit the run tested.
    repository: `owner/name` of the repository the run belongs to.
    head_repository: `owner/name` of the repository the branch lives in. Different from
      `repository` for a fork pull request.
    actor: Login of the actor GitHub attributes the run to.
    triggering_actor: Login of whoever triggered this attempt.
    display_title: The run's title line, usually the pull request title.
    html_url: Link to the run in the GitHub UI.
    previous_attempt_url: API URL of the previous attempt, None on attempt 1.
    superseded: True when a newer run exists for the same workflow and branch and this one
      was cancelled; set by `runs.mark_superseded`, None when nothing has decided yet.
      Superseded runs are stored and tagged, and excluded from every statistic.
    pr_number: Pull request number, or None when no pull request could be linked.
    pr_state: "open" or "closed".
    pr_title: Pull request title.
    pr_merged_at: When the pull request merged, ISO-8601 UTC; None when it never did. This is
      the field that decides whether the run belongs on a chart axis.
    pr_created_at: When the pull request was opened.
    pr_closed_at: When it was closed, merged or not.
    pr_head_sha: Head commit of the pull request; matching it against `head_sha` is how a
      run found through the `?head=` fallback is confirmed to be this run's pull request.
    pr_head_label: The `owner:branch` label of the head, e.g. "guowei-dev:pr/my-branch".
    pr_base_ref: Branch the pull request targets.
    pr_merge_commit_sha: The merge commit, which is what lands on main.
    pr_user: Login of the pull request author.
    pr_html_url: Link to the pull request.
    pr_draft: Whether the pull request was a draft.
    v: Row schema version.
    collected_at: When this row was written, ISO-8601 UTC.
  """

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
  """One job of one attempt, with its steps.

  Key: `job|<run_id>|<attempt>|<job_id>`. The job id alone would do - GitHub never reuses one
  - but run and attempt are in front so a key sorts and reads usefully, and so a job can be
  found without an index.

  Timestamps are stored as GitHub gave them. Three shapes matter downstream and none of them
  is judged here:

    * `started_at` earlier than `created_at` - the job did not run in this attempt. GitHub
      carries jobs it did not re-run into the next attempt with a fresh id and that attempt's
      `created_at`, so the queue wait computes negative (-21,934 s in one real run).
    * `steps == []` with timestamps present - the job never reached a step, so it has no
      setup time and contributes nothing to a suite duration.
    * `created_at == started_at` with `steps == []` and `labels == []` - a skipped job. It
      never held a runner, and its zero seconds mean "did not run", not "was instant".

  Attributes:
    job_id: The job id. Fresh in every attempt, even for a job that did not re-run.
    run_id: The run this job belongs to.
    attempt: The attempt this job was listed under.
    name: The full job name, e.g.
      "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit". The worker number and
      the flavor are read out of this string; nothing else names them.
    status: "completed", "in_progress" or "queued".
    conclusion: "success", "failure", "cancelled", "skipped", or None while running.
    created_at: When the job was created, ISO-8601 UTC.
    started_at: When a runner picked it up, or None while it is still queued.
    completed_at: When it finished, or None while it is still running.
    labels: The runs-on labels, e.g. ["linux-x86-ct6e-180-4tpu"]. The device lane is read
      from these. An empty list means no runner was ever assigned.
    runner_id: Numeric id of the runner, None when none was assigned.
    runner_name: Runner pod name, None when none was assigned.
    runner_group_id: Numeric id of the runner group.
    runner_group_name: Runner group, e.g. "ml-east5-general-a". This is the real pool
      dimension for queue statistics.
    workflow_name: Display name of the workflow the job belongs to.
    head_branch: Branch the job ran on.
    head_sha: Commit the job ran against.
    html_url: Link to the job in the GitHub UI.
    steps: One dict per step with the STEP_FIELDS keys, in the order GitHub listed them.
    v: Row schema version.
    collected_at: When this row was written, ISO-8601 UTC.
  """

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
  """One suite's totals in one attempt: the per-flavor numbers, or the reason there are none.

  Key: `suite|<run_id>|<attempt>|<suite_id>`.

  Every count is None when the suite published nothing, and `reason` says why. A count of
  zero would read on the chart as a suite whose tests vanished.

  `is_partial` is the flag that keeps a hole from being read as a drop. When some workers of
  a flavor published and others did not, the totals are the surviving workers' totals only,
  and `missing_workers` names the rest. A partial total must never be drawn as a fall in the
  test count.

  This row holds the counts. The wall-clock duration D of the suite is NOT here: it comes
  from the "Run Tests" step timestamps of the suite's jobs, and `derive.py` computes it from
  the job rows. The JUnit seconds stored here are the sum of the per-case `time` attributes,
  which on the CPU flavors runs 1.55x the real wall clock because pytest runs with `-n auto`.

  Attributes:
    run_id: The run these totals belong to.
    attempt: The attempt they were read from.
    suite_id: A flavor name, or "decoupled" for the nested pass.
    flavor: The flavor whose jobs and workers this suite ran in. Equal to `suite_id` for a
      real flavor; for the nested pass it is the parent, "cpu-unit".
    nested_in: The flavor this suite runs inside, or None for a real flavor. A nested suite's
      tests are counted in the parent too, so the two totals must never be added together.
    collected: Number of `<testcase>` elements, or None. Never the `<testsuite tests>`
      attribute, which disagrees with the elements on real files.
    skipped: Cases with a `<skipped>` child, or None.
    executed: `collected - skipped`, computed by `junit.py` and copied here. This is the
      number the dashboard calls the test count.
    failed: Cases with a `<failure>` child, or None.
    errored: Cases with an `<error>` child, or None.
    junit_seconds: Sum of the per-case `time` attributes, or None. Not wall clock.
    reported_tests: The `<testsuite tests>` attribute, kept only to cross-check `collected`.
    suite_seconds: The `<testsuite time>` attribute. Also not wall clock.
    reason: Why there is no result: "no_file_published", "artifact_expired" or
      "upload_empty". None when there is a result.
    is_partial: True when there is a result but at least one worker is missing from it.
    missing_workers: One dict per missing worker, {"worker": int, "reason": str}. A list of
      dicts rather than a map because JSON turns integer keys into strings, which would stop
      a row from round-tripping.
    published_workers: Worker numbers that published a readable file. A cross-check only -
      the authoritative worker count W is the number of "Execute Tests (N)" jobs.
    files: Names of the XML files the totals were built from.
    v: Row schema version.
    collected_at: When this row was written, ISO-8601 UTC.
  """

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
  """One `<testcase>` of one suite in one attempt.

  Key: `test|<run_id>|<attempt>|<suite_id>|<worker>|<classname>|<name>`.

  The key holds the suite id, not the flavor. They are the same string for every real flavor,
  but the nested `decoupled` pass runs the same 50 tests a second time inside cpu-unit worker
  1; keying on the flavor would make each decoupled row overwrite the cpu-unit row of the
  same test and silently halve that run's test history.

  Attributes:
    run_id: The run this result came from.
    attempt: The attempt the artifact belonged to.
    suite_id: A flavor name, or "decoupled" for the nested pass.
    flavor: The flavor whose artifact carried the file. "cpu-unit" for the decoupled pass.
    worker: The parallel worker that ran the test, or None when the file did not say.
    classname: The dotted module and class, e.g. "tests.unit.attention_test.AttentionTest".
      Empty for a module-level collection skip.
    name: The test name, including any pytest parameter id.
    status: "passed", "skipped", "failed" or "error".
    duration: The `time` attribute of the `<testcase>` in seconds, verbatim. Not wall clock on
      the CPU flavors, which run pytest with `-n auto`. Named as `junit.TestRow` names it, so
      the same field name reads a parsed case and a stored row.
    failure_message: First line of the failure or error message, quoted verbatim; None for a
      test that passed or was skipped. Nothing summarises or rewrites it.
    nested_in: The flavor this suite runs inside, or None for a real flavor.
    suite_partial: True when the suite this row belongs to is missing some of its workers, so
      a reader knows the run's totals are a hole rather than a drop.
    v: Row schema version.
    collected_at: When this row was written, ISO-8601 UTC.
  """

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
  """One job name that failed and then passed on a re-run, or failed and never recovered.

  Key: `rescue|<run_id>|<job_name>|<failed_attempt>`. The identity is the NAME because job
  ids are minted fresh in every attempt - in one real run all 42 name pairs had different ids
  between attempts, including the 28 jobs that never re-ran.

  The failed attempt is part of the key because one name can carry two different facts in one
  run. A job that goes failure -> success -> failure was rescued at attempt 1 and then failed
  for good at attempt 3, and both have to be stored: without the attempt the rescue row and
  the never-rescued row landed on the same key with the same `collected_at`, so write order
  alone decided which fact survived. Correcting a row still works, because a correction keeps
  the same failed attempt: a first tick that sees only the failure writes
  `rescue|<id>|<name>|1` with `rescued` False, and the next tick, which sees the success,
  writes the same key with `rescued` True and a later `collected_at`.

  `rescue_rows` emits only the rows where `rescued` is True, which is what the flaky card
  counts. The shape still carries `rescued` so the other half of the story fits in the same
  record: the dashboard has a "failed, never re-run" cell, and a job that failed and stayed
  failed has to be distinguishable from one that recovered, not simply absent. Build those
  rows with `failed_never_rescued_rows`. Storing both in one shape means the flaky card reads
  a single stream and a row can be corrected from one state into the other if a run grows
  another attempt - which real runs do.

  Nothing here is a duration. Wasted minutes are the run time of the failed attempt's job,
  and `derive.py` computes them from `failed_started_at` and `failed_completed_at`.

  Attributes:
    run_id: The run both attempts belong to.
    job_name: The full job name, identical across attempts.
    rescued: True when a later attempt of this name concluded success right after the
      failure. False on a failure that never recovered.
    failed_attempt: The attempt that failed. For a rescue this is the failure of the pair.
      For an unrescued failure it is the start of the run of failures the job ended on, so a
      job that failed three times in a row reports attempt 1 and a job that was rescued and
      then failed again reports the later failure, not the rescued one.
    failed_job_id: Job id of the failed attempt.
    failed_conclusion: Conclusion of the failed attempt, always "failure".
    failed_created_at: `created_at` of the failed job.
    failed_started_at: `started_at` of the failed job.
    failed_completed_at: `completed_at` of the failed job.
    rescued_attempt: The attempt that passed, or None when nothing did.
    rescued_job_id: Job id of the passing attempt, or None.
    rescued_started_at: `started_at` of the passing job, or None.
    rescued_completed_at: `completed_at` of the passing job, or None.
    final_attempt: The last attempt this name appeared in.
    final_conclusion: Its conclusion in that attempt. "failure" on an unrescued failure.
    attempts_seen: Every attempt this name appeared in, ascending. A name can be absent from
      an attempt: one real run's "All Required Tests Passed" job exists only in attempts 2
      and 3.
    labels: runs-on labels of the failed job, so the device lane can be read from the row.
    head_branch: Branch the run was on.
    workflow_name: Display name of the workflow.
    event: The run's trigger.
    html_url: Link to the failed job in the GitHub UI.
    v: Row schema version.
    collected_at: When this row was written, ISO-8601 UTC.
  """

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
    """True when a later attempt ran this name again, whatever the outcome was.

    False on an unrescued failure means the job failed and was never re-run at all, which is
    the dashboard's "failed, never re-run" cell. True with `rescued` False means it was re-run
    and failed again.
    """
    return self.final_attempt is not None and self.final_attempt > self.failed_attempt


def run_row(run: dict[str, Any], pr: dict[str, Any] | None = None, collected_at: str | None = None) -> RunRow:
  """Builds the row for one attempt of one workflow run.

  Args:
    run: A run object from the runs endpoint or the single-run endpoint. When
      `runs.mark_superseded` has tagged it, its `superseded` flag is carried through.
    pr: The pull request the run belongs to, as `runs.link_pull_request` returns it, or None.
      `run["pull_requests"]` is empty for fork runs and for merged same-repo runs alike, so
      the linked object is passed in rather than dug out of the run.
    collected_at: Overrides the write timestamp. Tests pass a fixed value; the collector
      leaves it out.

  Returns:
    The row. Nothing is computed: every field is the API's own value.

  Raises:
    RowError: The run has no `id` or no `run_attempt`.
  """
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
  """Copies the linked pull request's fields onto a run row.

  Args:
    row: The row to fill in. Modified in place.
    pr: The pull request object.
  """
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
  """Builds the row for one job of one attempt.

  The attempt comes from the job's own `run_attempt`, not the run's: a run's `run_attempt`
  moves while the collector works - one real run read 2 from the runs list and 3 from the
  single-run endpoint half an hour later - whereas a job payload always says which attempt it
  was listed under.

  Args:
    run: The run the job belongs to, used for the run id and as a cross-check.
    job: One job object from the jobs endpoint.
    collected_at: Overrides the write timestamp.

  Returns:
    The row, with the raw timestamps and every step.

  Raises:
    RowError: The job has no `id`, neither the job nor the run says which run or attempt it
      belongs to, the job belongs to a different run than the one passed in, or a step is not
      a JSON object.
  """
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
  """Builds the totals row for one suite of one attempt.

  Takes a `junit.SuiteEntry` rather than a bare `SuiteResult` because the partial state - the
  workers that published nothing - lives on the entry, and a partial total that loses that
  flag reads on the chart as a drop in the test count.

  Args:
    run: The run the suite belongs to.
    entry: The suite's entry from `junit.read_run_tests`, result or reason.
    collected_at: Overrides the write timestamp.

  Returns:
    The row. Counts are None, never zero, when the suite published nothing.

  Raises:
    RowError: The run has no `id` or no `run_attempt`.
  """
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
  """Builds one row per `<testcase>` of one suite.

  Args:
    run: The run the results came from.
    flavor: The flavor whose artifact carried the file, e.g. "cpu-unit".
    worker: The parallel worker the artifact belonged to. A case that carries its own worker
      number keeps it, so a merged multi-worker result still produces distinct keys.
    suite_result: A `junit.SuiteResult`. Its `tests` are copied field for field; the `time`
      attribute is stored as it was and never turned into anything.
    suite_id: The suite these cases belong to. Defaults to `flavor`; pass "decoupled" for the
      nested pass, whose tests would otherwise overwrite cpu-unit's rows for the same tests.
    nested_in: The flavor a nested suite runs inside, e.g. "cpu-unit" for "decoupled".
    suite_partial: True when some workers of the suite published nothing, so the reader knows
      the run's totals are incomplete rather than lower.
    collected_at: Overrides the write timestamp on every row.

  Returns:
    One row per case, in the order the XML listed them. An empty list when the result holds
    no cases; a suite that published nothing has no rows at all, and its SuiteRow carries the
    reason.

  Raises:
    RowError: The run has no `id` or no `run_attempt`.
  """
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
  """Groups every attempt's jobs by job name, in attempt order.

  Job ids change between attempts even for jobs that did not re-run, so the name is the only
  stable identity. GitHub lists each name once per attempt; a repeated name keeps its first
  entry, and a name can be missing from an attempt entirely.

  Args:
    attempts_jobs: Attempt number -> the jobs listed for that attempt.

  Returns:
    Job name -> [(attempt, job), ...] ascending by attempt.

  Raises:
    RowError: An attempt number is not an integer, an attempt's value is not a list, or a job
      has no name.
  """
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
  """Reads an attempt number, accepting the string keys a JSON round trip leaves behind.

  Args:
    value: The attempt key.

  Returns:
    The attempt as an int.

  Raises:
    RowError: The key is not a number.
  """
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
  """Builds one rescue row from a failure and, when there was one, the attempt that passed.

  Args:
    run: The run both attempts belong to.
    name: The job name.
    observations: Every (attempt, job) this name appeared in, ascending.
    failed: The (attempt, job) that failed and that this row is keyed on.
    passed: The (attempt, job) that passed right after it, or None when nothing did.
    collected_at: Overrides the write timestamp.

  Returns:
    The row, with `rescued` True only when `passed` is given.
  """
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
  """Finds the jobs that failed and then passed on a re-run.

  A rescue is one job NAME concluding "failure" in one attempt and "success" in the next
  attempt that name appears in, inside the same run. Two rules keep that honest:

    * Names, not ids. GitHub mints a fresh job id in every attempt, including for the jobs it
      carried over without re-running them.
    * Only failure then success. A cancelled job that succeeds next time is not a rescue: one
      real run has a worker going cancelled -> success -> success and yields zero rescues.

  A job that failed and never recovered gets no row here, because it is not a rescue and the
  flaky card would over-count if it did. It is not lost either: `failed_never_rescued_rows`
  builds the same shape with `rescued` False for the dashboard's "failed, never re-run" cell.

  Args:
    run: The run all the attempts belong to.
    attempts_jobs: Attempt number -> the jobs listed for that attempt, as
      `runs.get_jobs(client, run_id, attempt)` returns them.
    collected_at: Overrides the write timestamp on every row.

  Returns:
    One row per failure-then-success pair, ordered by job name. A name that failed, recovered
    and failed again yields one row here and one from `failed_never_rescued_rows`; they carry
    different failed attempts, so they are two keys and neither overwrites the other.

  Raises:
    RowError: The run has no `id`, an attempt key is not a number, or a job has no name.
  """
  rows: list[RescueRow] = []
  for name, observations in sorted(_job_history(attempts_jobs).items()):
    for earlier, later in zip(observations, observations[1:]):
      if earlier[1].get("conclusion") == CONCLUSION_FAILURE and later[1].get("conclusion") == CONCLUSION_SUCCESS:
        rows.append(_rescue_row(run, name, observations, earlier, later, collected_at))
  return rows


def failed_never_rescued_rows(
    run: dict[str, Any],
    attempts_jobs: dict[int, list[dict[str, Any]]],
    collected_at: str | None = None,
) -> list[RescueRow]:
  """Builds the rows for jobs that failed and never passed, the other half of the flaky card.

  These are not rescues, so `rescue_rows` leaves them out. The dashboard still has to draw
  them: a red "failed, never re-run" cell is a different fact from a clean run, and a chart
  that shows only rescues would make the worst runs look empty. Both halves share one shape
  so the card reads one stream, and so a row can be corrected from one state to the other
  when a run grows another attempt - which real runs do.

  `RescueRow.rerun_after_failure` separates the two shapes of failure: False means the job
  failed and nothing ever ran it again; True means it was re-run and failed again.

  The failure a row points at is the start of the unbroken run of failures the job ended on,
  never an earlier failure that was rescued. For a job reading failure -> success -> failure
  that is attempt 3, so its row does not claim the seconds attempt 1 already spent as a
  rescue, and it does not collide with the rescue row's key. For a job reading failure ->
  failure -> failure it is attempt 1, which keeps `rerun_after_failure` True: the job was
  re-run twice and failed again, which is a different fact from never being re-run.

  Args:
    run: The run all the attempts belong to.
    attempts_jobs: Attempt number -> the jobs listed for that attempt.
    collected_at: Overrides the write timestamp on every row.

  Returns:
    One row per job name whose last appearance concluded "failure", ordered by name, with
    `rescued` False.

  Raises:
    RowError: The run has no `id`, an attempt key is not a number, or a job has no name.
  """
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
  """Returns the kind string of a row, e.g. "job".

  Args:
    row: Any row.

  Returns:
    One of the KIND_* constants.

  Raises:
    RowError: The object is not one of this module's row types.
  """
  kind = _KIND_BY_TYPE.get(type(row))
  if kind is None:
    raise RowError(f"{type(row).__name__} is not a stored row type.")
  return kind


def to_json(row: Row) -> dict[str, Any]:
  """Turns a row into the JSON object that is written to the store.

  The object is flat and holds only JSON's own types: strings, numbers, booleans, None,
  lists, and - for a job's steps and a suite's missing workers - small plain dicts. No
  tuples, and no dict with integer keys, because JSON has neither and a row that cannot come
  back the way it went in is a row that quietly changes when it is re-read.

  Args:
    row: The row to serialise.

  Returns:
    A new dict carrying every field of the row plus "kind", which is what lets `from_json`
    rebuild the right type.

  Raises:
    RowError: The object is not one of this module's row types.
  """
  payload: dict[str, Any] = {"kind": row_kind(row)}
  payload.update(dataclasses.asdict(row))
  return payload


def from_json(payload: dict[str, Any]) -> Row:
  """Rebuilds a row from a stored JSON object.

  `from_json(to_json(row)) == row` for every row, None values included: the payload carries
  every field, so nothing is filled in from a default and nothing is coerced.

  Args:
    payload: A JSON object written by `to_json`.

  Returns:
    The row, of the type its "kind" names.

  Raises:
    RowError: The payload has no known "kind", was written by a newer row schema version, or
      does not carry exactly the fields that type has. An unexpected field is an error rather
      than something to drop, because dropping it would lose data the store already holds.
  """
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
