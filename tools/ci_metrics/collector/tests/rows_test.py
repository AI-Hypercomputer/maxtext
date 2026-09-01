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

"""Offline unit tests for `collector.rows`.

Every test runs against the saved fixtures in `tests/fixtures/` and never touches the
network: the base test case replaces `socket.socket`, so a test that reached for GitHub
would fail instead of hanging. `rows.py` makes no calls of its own, so the only stub here is
the one `junit.read_run_tests` needs to hand a real, partly-missing suite to the builders.

The numbers are measured facts about real runs of the "MaxText Package Tests" pipeline, not
round numbers picked for the test:

  * run 33468578834 - one attempt, 54 jobs, the run every layer-1 fixture came from.
  * run 32772626658 - two attempts; three jobs failed and passed on the re-run, five failed
    again, and two more failed only in attempt 2.
  * run 33037584699 - two attempts; every attempt-1 failure was rescued.
  * run 32785979907 - three attempts and zero rescues, because its only recovery is
    cancelled -> success, and cancelled is not failure.
  * run 32999133815 - eight jobs that never held a runner, kept here to prove no timestamp
    is ever turned into a number by this module.

The four promises the module makes are checked head on:

  1. A key identifies a row and nothing else. Every part is percent-encoded, so a job name
     full of slashes and brackets cannot forge a separator, and a key changes when - and only
     when - one of its own parts changes.
  2. `from_json(to_json(row)) == row` for every row type, None fields included, through a
     real `json.dumps`/`json.loads` trip.
  3. A rescue is failure then success, by job NAME, inside one run. A job that failed and
     never recovered gets no rescue row; it gets the same shape with `rescued` False from
     `failed_never_rescued_rows`.
  4. Nothing here is arithmetic. Timestamps are stored exactly as GitHub wrote them, counts
     exactly as `junit.py` counted them, and a suite that lost some of its workers carries
     that partial flag through to every test row.

The tests are plain `unittest`, so they need nothing but the standard library. pytest
collects them too, because the file is named the way the repository's pytest.ini expects.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/rows_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/rows_test.py
"""

from __future__ import annotations

import dataclasses
import io
import json
import socket
import sys
import unittest
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import junit
from collector import rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The single-attempt run the layer-1 fixtures came from.
RUN_ID = 33468578834

# Two attempts, three rescues, five failures that stayed failures.
RERUN_ID = 32772626658

# Two attempts, four rescues, nothing left failing.
CLEAN_RERUN_ID = 33037584699

# Three attempts, zero rescues: its only recovery is cancelled -> success.
CANCELLED_RUN_ID = 32785979907

# Eight jobs that never held a runner.
QUEUED_RUN_ID = 32999133815

# The tpu-unit worker 1 job of run 33468578834, measured by hand: queue 109 s, setup 71 s.
# The row must carry these strings and none of those seconds.
TPU_UNIT_WORKER_1 = "TPU Pretrain Tests (tpu-unit) / Execute Tests (1) / tpu-unit"
TPU_UNIT_WORKER_1_ID = 99733940992
TPU_UNIT_WORKER_1_CREATED = "2026-09-01T04:08:43Z"
TPU_UNIT_WORKER_1_STARTED = "2026-09-01T04:10:32Z"
TPU_UNIT_WORKER_1_COMPLETED = "2026-09-01T04:30:49Z"

# The three rescues of run 32772626658: job name -> the failed job's id, the span GitHub
# recorded for it, and the id of the job that passed on attempt 2. The seconds are what
# `derive.py` will later call wasted minutes; no row stores them.
RESCUES_32772626658 = {
    "CPU Posttrain Tests (cpu-post-training-unit) / Execute Tests (3) / cpu-post-training-unit": {
        "failed_job_id": 97581996129,
        "failed_started_at": "2026-08-24T20:56:42Z",
        "failed_completed_at": "2026-08-24T20:59:08Z",
        "rescued_job_id": 97664839107,
        "seconds": 146,
    },
    "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit": {
        "failed_job_id": 97581999134,
        "failed_started_at": "2026-08-24T20:59:11Z",
        "failed_completed_at": "2026-08-24T21:00:55Z",
        "rescued_job_id": 97664839240,
        "seconds": 104,
    },
    "CPU Pretrain Tests (cpu-unit) / Execute Tests (3) / cpu-unit": {
        "failed_job_id": 97581998878,
        "failed_started_at": "2026-08-24T20:57:00Z",
        "failed_completed_at": "2026-08-24T20:58:55Z",
        "rescued_job_id": 97664839310,
        "seconds": 115,
    },
}

# The five names of run 32772626658 that failed in attempt 1 and failed again in attempt 2.
FAILED_AGAIN_32772626658 = (
    "All Required Tests Passed",
    "CPU Pretrain Tests (cpu-unit) / Execute Tests (2) / cpu-unit",
    "TPU Pathways Unit Tests (1) / tpu-pathways-unit",
    "TPU Pathways Unit Tests (2) / tpu-pathways-unit",
    "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit",
)

# Two more names of the same run end on a failure without ever failing in attempt 1: one was
# cancelled first, one succeeded first. Both belong in the failed-and-never-rescued stream.
FAILED_ONLY_IN_ATTEMPT_2 = (
    "TPU Pathways Integration Tests / tpu-pathways-integration",
    "Track Test Performance / Track Test Duration",
)

# The four rescues of run 33037584699 and the span of each failed job.
RESCUES_33037584699 = {
    "All Required Tests Passed": 3,
    "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit": 224,
    "GPU Tests (gpu-integration) / Execute Tests (1) / gpu-integration": 862,
    "TPU Pretrain Tests (tpu-integration) / Execute Tests (1) / tpu-integration": 1545,
}

# The six names of run 32785979907 that end on a failure. Its tpu-unit worker 2 goes
# cancelled -> success -> success and is in neither stream.
NEVER_RESCUED_32785979907 = (
    "All Required Tests Passed",
    "CPU Pretrain Tests (cpu-integration) / Execute Tests (1) / cpu-integration",
    "GPU Tests (gpu-integration) / Execute Tests (1) / gpu-integration",
    "TPU Pathways Integration Tests / tpu-pathways-integration",
    "TPU Pathways Unit Tests (1) / tpu-pathways-unit",
    "TPU Pathways Unit Tests (2) / tpu-pathways-unit",
)

# The format GitHub writes, and the only format `collected_at` may use.
ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Row field -> the run payload key it must be copied from, for the flat fields.
RUN_FIELD_SOURCE = {
    "run_id": "id",
    "attempt": "run_attempt",
    "workflow_id": "workflow_id",
    "workflow_name": "name",
    "workflow_path": "path",
    "run_number": "run_number",
    "event": "event",
    "status": "status",
    "conclusion": "conclusion",
    "created_at": "created_at",
    "run_started_at": "run_started_at",
    "updated_at": "updated_at",
    "head_branch": "head_branch",
    "head_sha": "head_sha",
    "display_title": "display_title",
    "html_url": "html_url",
    "previous_attempt_url": "previous_attempt_url",
}

# Row field -> the job payload key it must be copied from.
JOB_FIELD_SOURCE = {
    "job_id": "id",
    "run_id": "run_id",
    "attempt": "run_attempt",
    "name": "name",
    "status": "status",
    "conclusion": "conclusion",
    "created_at": "created_at",
    "started_at": "started_at",
    "completed_at": "completed_at",
    "labels": "labels",
    "runner_id": "runner_id",
    "runner_name": "runner_name",
    "runner_group_id": "runner_group_id",
    "runner_group_name": "runner_group_name",
    "workflow_name": "workflow_name",
    "head_branch": "head_branch",
    "head_sha": "head_sha",
    "html_url": "html_url",
}

# Words that would name a computed quantity. None of them may appear in a row's field names:
# every duration, wait and rate belongs to `derive.py`, over rows collected years earlier.
DERIVED_WORDS = ("wait", "queue", "wasted", "elapsed", "minute", "machine", "median", "baseline", "wall", "_rate")

# The same, plus the two words that are legitimate on the rows carrying JUnit's own numbers:
# `TestRow.duration` is the `<testcase time>` attribute and `SuiteRow.junit_seconds` is the
# sum `junit.py` made of them. Nowhere else may a row name a second.
TIME_WORDS = ("second", "duration")


def read_fixture(name: str) -> bytes:
  """Returns the raw bytes of one saved fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The file contents.
  """
  return (FIXTURES / name).read_bytes()


def load_json(name: str) -> Any:
  """Loads one saved JSON fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The parsed payload, fresh each call so no test can mutate another's copy.
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


def attempts_of(run_id: int, prefix: str, last_attempt: int) -> dict[int, list[dict[str, Any]]]:
  """Loads every saved attempt of one multi-attempt run.

  Args:
    run_id: The run id, which is part of every fixture name.
    prefix: The fixture name prefix, e.g. "rerun".
    last_attempt: The highest attempt saved.

  Returns:
    Attempt number -> that attempt's jobs, the shape `rescue_rows` takes.
  """
  return {n: load_jobs(f"{prefix}-{run_id}-attempt{n}-jobs.json") for n in range(1, last_attempt + 1)}


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


def iso(value: str) -> datetime:
  """Parses one of GitHub's timestamps as UTC.

  Args:
    value: An ISO-8601 timestamp ending in "Z".

  Returns:
    The moment, tagged UTC.
  """
  return datetime.strptime(value, ISO_FORMAT).replace(tzinfo=timezone.utc)


def span_seconds(start: str, end: str) -> float:
  """Returns the seconds between two of GitHub's timestamps.

  This is the arithmetic `rows.py` must never do; the tests do it to prove the row kept the
  timestamps that bracket a measured span.

  Args:
    start: The earlier timestamp.
    end: The later timestamp.

  Returns:
    The difference in seconds, negative when `end` precedes `start`.
  """
  return (iso(end) - iso(start)).total_seconds()


def numbers_in(payload: Any) -> list[float]:
  """Collects every number anywhere inside a JSON payload.

  Args:
    payload: A value from `rows.to_json`.

  Returns:
    Every int and float in it, booleans excluded.
  """
  if isinstance(payload, bool):
    return []
  if isinstance(payload, (int, float)):
    return [payload]
  if isinstance(payload, dict):
    return [number for value in payload.values() for number in numbers_in(value)]
  if isinstance(payload, list):
    return [number for item in payload for number in numbers_in(item)]
  return []


def json_types_ok(payload: Any) -> bool:
  """Says whether a payload holds only types JSON can carry back unchanged.

  A tuple or a dict with integer keys survives `json.dumps` but comes back as something
  else, which would break the round trip quietly.

  Args:
    payload: A value from `rows.to_json`.

  Returns:
    True when every value in it is a string, number, boolean, None, list or string-keyed
    dict.
  """
  if payload is None or isinstance(payload, (str, bool, int, float)):
    return True
  if isinstance(payload, list):
    return all(json_types_ok(item) for item in payload)
  if isinstance(payload, dict):
    return all(isinstance(key, str) and json_types_ok(value) for key, value in payload.items())
  return False


def make_zip(members: dict[str, bytes]) -> bytes:
  """Builds an artifact zip in memory.

  Args:
    members: Member name -> file bytes.

  Returns:
    The zip bytes, as `GitHubClient.get_bytes` would return them.
  """
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as archive:
    for member_name, payload in members.items():
      archive.writestr(member_name, payload)
  return buffer.getvalue()


def artifact_payload(name: str, artifact_id: int, expired: bool = False) -> dict[str, Any]:
  """Builds one entry of the artifacts endpoint.

  Args:
    name: The artifact name.
    artifact_id: The numeric artifact id, also used to build the download URL.
    expired: Whether GitHub has already deleted the payload.

  Returns:
    A payload shaped like the real endpoint's entries.
  """
  return {
      "id": artifact_id,
      "name": name,
      "expired": expired,
      "size_in_bytes": 17613,
      "archive_download_url": f"https://api.github.com/artifacts/{artifact_id}/zip",
      "created_at": "2026-09-01T04:15:40Z",
      "expires_at": "2026-09-02T04:15:39Z",
  }


class StubClient:
  """A stand-in for `github.GitHubClient` that serves saved payloads.

  Only `junit.read_run_tests` needs it; `rows.py` itself never takes a client.
  """

  def __init__(self, payloads: list[dict[str, Any]], blobs: dict[str, bytes] | None = None) -> None:
    """Stores what the stub will serve.

    Args:
      payloads: What `paginate` returns for the artifacts endpoint.
      blobs: Download URL -> zip bytes for `get_bytes`.
    """
    self.payloads = payloads
    self.blobs = blobs or {}

  def paginate(self, path: str, key: str, **params: Any) -> list:
    """Returns the saved artifact payloads.

    Args:
      path: The endpoint path the module asked for.
      key: The list key inside the response.
      **params: Query parameters.

    Returns:
      The saved payload list.
    """
    del path, key, params
    return self.payloads

  def get_bytes(self, url: str) -> bytes:
    """Returns the saved zip for one download URL.

    Args:
      url: The absolute download URL.

    Returns:
      The zip bytes.

    Raises:
      KeyError: The test did not stage a body for this URL.
    """
    return self.blobs[url]


@dataclass
class StubSuiteResult:
  """A `junit.SuiteResult` look-alike whose counts disagree with each other on purpose.

  The builders take these by structural type - the module says so - so a stub can prove that
  `rows.py` copies what it is handed instead of recomputing it. A real `SuiteResult` always
  satisfies `executed == collected - skipped`, which is exactly why it cannot show that.
  """

  collected: int
  skipped: int
  executed: int
  junit_seconds: float
  tests: list[Any] = field(default_factory=list)
  failed: int = 0
  errored: int = 0
  reported_tests: int | None = None
  suite_seconds: float | None = None
  files: tuple[str, ...] = ()


@dataclass
class StubSuiteEntry:
  """A `junit.SuiteEntry` look-alike, for the same reason."""

  suite_id: str
  result: Any = None
  reason: str | None = None
  nested_in: str | None = None
  per_worker: dict[int, Any] = field(default_factory=dict)
  missing_workers: dict[int, str] = field(default_factory=dict)


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

  def tpu_unit_suite_entry(self, missing: dict[int, str] | None = None) -> junit.SuiteEntry:
    """Builds a real `junit.SuiteEntry` for tpu-unit worker 1 of run 33468578834.

    Args:
      missing: Worker number -> reason for the workers that published nothing.

    Returns:
      The entry, with worker 1's parsed result in it.
    """
    result = junit.parse_junit_xml(read_fixture("tpu-unit-1.xml"), file_name="tpu-unit-1.xml")
    return junit.SuiteEntry(
        suite_id="tpu-unit",
        result=result,
        per_worker={1: result},
        missing_workers=dict(missing or {}),
    )

  def one_row_of_each_kind(self) -> dict[str, rows.Row]:
    """Builds one row of every stored type from the saved fixtures.

    Returns:
      Kind string -> a row of that kind, all from real payloads.
    """
    run = load_json("run.json")
    job = named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1)
    entry = self.tpu_unit_suite_entry()
    rerun = load_json(f"rerun-{RERUN_ID}-run.json")
    attempts = attempts_of(RERUN_ID, "rerun", 2)
    merged = load_json("merged-pr-5070-run-33406483779.json")
    pull_request = load_json("merged-pr-5070-pulls-by-head.json")[0]
    return {
        rows.KIND_RUN: rows.run_row(merged, pull_request),
        rows.KIND_JOB: rows.job_row(run, job),
        rows.KIND_SUITE: rows.suite_row(run, entry),
        rows.KIND_TEST: rows.test_rows(run, "tpu-unit", 1, entry.result)[0],
        rows.KIND_RESCUE: rows.rescue_rows(rerun, attempts)[0],
    }


class KeyFormatTest(OfflineTestCase):
  """Covers `key()` on every row type, built from real payloads."""

  def test_run_key_names_the_run_and_the_attempt(self) -> None:
    """`run|<run_id>|<attempt>`, so a re-run cannot overwrite the attempt before it."""
    self.assertEqual(rows.run_row(load_json("run.json")).key(), f"run|{RUN_ID}|1")
    self.assertEqual(
        rows.run_row(load_json(f"cancelled-job-{CANCELLED_RUN_ID}-run.json")).key(),
        f"run|{CANCELLED_RUN_ID}|3",
    )

  def test_job_key_names_run_attempt_and_job(self) -> None:
    """`job|<run_id>|<attempt>|<job_id>`, in that order so a key sorts usefully."""
    row = rows.job_row(load_json("run.json"), named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1))
    self.assertEqual(row.key(), f"job|{RUN_ID}|1|{TPU_UNIT_WORKER_1_ID}")

  def test_suite_key_names_the_suite_not_the_flavor(self) -> None:
    """`suite|<run_id>|<attempt>|<suite_id>`. The nested pass keys on "decoupled"."""
    run = load_json("run.json")
    self.assertEqual(rows.suite_row(run, self.tpu_unit_suite_entry()).key(), f"suite|{RUN_ID}|1|tpu-unit")

    nested = junit.SuiteEntry(suite_id="decoupled", reason=junit.REASON_NO_FILE, nested_in="cpu-unit")
    nested_row = rows.suite_row(run, nested)
    self.assertEqual(nested_row.key(), f"suite|{RUN_ID}|1|decoupled")
    self.assertEqual(nested_row.flavor, "cpu-unit")

  def test_test_key_holds_every_identifying_part(self) -> None:
    """`test|<run_id>|<attempt>|<suite_id>|<worker>|<classname>|<name>`.

    The first case of the real tpu-unit file is a module-level collection skip, so its
    classname is empty and the key carries an empty part rather than dropping one.
    """
    entry = self.tpu_unit_suite_entry()
    first = rows.test_rows(load_json("run.json"), "tpu-unit", 1, entry.result)[0]

    self.assertEqual(first.classname, "")
    self.assertEqual(first.name, "tests.unit.goodput_utils_test")
    self.assertEqual(first.key(), f"test|{RUN_ID}|1|tpu-unit|1||tests.unit.goodput_utils_test")

  def test_test_key_percent_encodes_a_pytest_parameter_id(self) -> None:
    """A real parameterised id is full of brackets and slashes and still makes one part."""
    wanted = "test_llama_configs[/__w/maxtext/maxtext/src/maxtext/configs/models/llama3.1-8b.yml]"
    result = junit.parse_junit_xml(read_fixture("cpu-unit-1.xml"), file_name="cpu-unit-1.xml")
    cases = [case for case in result.tests if case.name == wanted]
    self.assertEqual(len(cases), 1, "the saved cpu-unit file should hold that parameterised case once")

    row = rows.test_rows(load_json("run.json"), "cpu-unit", 1, StubSuiteResult(1, 0, 1, 0.0, cases))[0]
    self.assertEqual(
        row.key(),
        f"test|{RUN_ID}|1|cpu-unit|1|tests.unit.configs_test|"
        "test_llama_configs%5B%2F__w%2Fmaxtext%2Fmaxtext%2Fsrc%2Fmaxtext%2Fconfigs%2Fmodels%2Fllama3.1-8b.yml%5D",
    )
    self.assertEqual(row.key().count("|"), 6)

  def test_rescue_key_is_the_run_the_encoded_job_name_and_the_failed_attempt(self) -> None:
    """`rescue|<run_id>|<job_name>|<failed_attempt>`. The name is the identity, ids change.

    The failed attempt is in the key so that one name can carry two facts in one run: a
    rescue at attempt 1 and, if the run is re-run again, a failure at attempt 3.
    """
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    built = {row.job_name: row for row in rows.rescue_rows(run, attempts_of(RERUN_ID, "rerun", 2))}
    row = built["CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit"]

    self.assertEqual(
        row.key(),
        f"rescue|{RERUN_ID}|CPU%20Pretrain%20Tests%20%28cpu-unit%29%20%2F%20Execute%20Tests%20%281%29%20%2F%20cpu-unit|1",
    )
    self.assertEqual(row.key().count("|"), 3)

  def test_a_name_cannot_forge_a_separator(self) -> None:
    """A "|" inside a value is encoded, so two different rows cannot share a key."""
    left = rows.TestRow(
        run_id=1, attempt=1, suite_id="s", flavor="s", worker=1, classname="a", name="b|c", status="passed", duration=0.0
    )
    right = rows.TestRow(
        run_id=1, attempt=1, suite_id="s", flavor="s", worker=1, classname="a|b", name="c", status="passed", duration=0.0
    )

    self.assertEqual(left.key(), "test|1|1|s|1|a|b%7Cc")
    self.assertEqual(right.key(), "test|1|1|s|1|a%7Cb|c")
    self.assertNotEqual(left.key(), right.key())

  def test_an_unknown_worker_is_an_empty_part_not_a_missing_one(self) -> None:
    """A file that did not say which worker ran a test still gets a well-formed key."""
    unknown = rows.TestRow(
        run_id=1, attempt=1, suite_id="s", flavor="s", worker=None, classname="a", name="b", status="passed", duration=0.0
    )
    known = dataclasses.replace(unknown, worker=1)

    self.assertEqual(unknown.key(), "test|1|1|s||a|b")
    self.assertEqual(unknown.key().count("|"), 6)
    self.assertNotEqual(unknown.key(), known.key())


class KeyStabilityTest(OfflineTestCase):
  """A key must not move when nothing identifying moved, and must move when one part does."""

  def test_the_same_payload_builds_the_same_key_twice(self) -> None:
    """Two constructions of one payload agree, even when their write timestamps differ."""
    run = load_json("run.json")
    job = named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1)
    entry = self.tpu_unit_suite_entry()
    rerun = load_json(f"rerun-{RERUN_ID}-run.json")
    attempts = attempts_of(RERUN_ID, "rerun", 2)

    first = [
        rows.run_row(run, collected_at="2026-09-01T04:00:00Z"),
        rows.job_row(run, job, collected_at="2026-09-01T04:00:00Z"),
        rows.suite_row(run, entry, collected_at="2026-09-01T04:00:00Z"),
        rows.test_rows(run, "tpu-unit", 1, entry.result, collected_at="2026-09-01T04:00:00Z")[0],
        rows.rescue_rows(rerun, attempts, collected_at="2026-09-01T04:00:00Z")[0],
    ]
    second = [
        rows.run_row(run, collected_at="2026-09-01T08:00:00Z"),
        rows.job_row(run, job, collected_at="2026-09-01T08:00:00Z"),
        rows.suite_row(run, entry, collected_at="2026-09-01T08:00:00Z"),
        rows.test_rows(run, "tpu-unit", 1, entry.result, collected_at="2026-09-01T08:00:00Z")[0],
        rows.rescue_rows(rerun, attempts, collected_at="2026-09-01T08:00:00Z")[0],
    ]

    self.assertEqual([row.key() for row in first], [row.key() for row in second])
    for before, after in zip(first, second):
      with self.subTest(kind=rows.row_kind(before)):
        self.assertNotEqual(before.collected_at, after.collected_at)

  def test_the_attempt_changes_the_key_of_every_row_that_carries_one(self) -> None:
    """Without the attempt in the key, a re-run would overwrite the attempt before it."""
    for row in self.one_row_of_each_kind().values():
      if not hasattr(row, "attempt"):
        continue
      with self.subTest(kind=rows.row_kind(row)):
        self.assertNotEqual(row.key(), dataclasses.replace(row, attempt=row.attempt + 1).key())

  def test_the_worker_changes_the_test_key(self) -> None:
    """Two workers of one flavor run different tests; their rows must not collide."""
    entry = self.tpu_unit_suite_entry()
    row = rows.test_rows(load_json("run.json"), "tpu-unit", 1, entry.result)[0]

    self.assertNotEqual(row.key(), dataclasses.replace(row, worker=2).key())
    self.assertIn("|1|", row.key())

  def test_the_job_id_changes_the_job_key(self) -> None:
    """Every job of a run needs its own row, so the id is part of the key."""
    row = rows.job_row(load_json("run.json"), named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1))
    self.assertNotEqual(row.key(), dataclasses.replace(row, job_id=row.job_id + 1).key())

  def test_the_job_name_changes_the_rescue_key(self) -> None:
    """Two rescued jobs of one run are two rows."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    keys = {row.key() for row in rows.rescue_rows(run, attempts_of(RERUN_ID, "rerun", 2))}
    self.assertEqual(len(keys), len(RESCUES_32772626658))

  def test_the_nested_pass_keeps_its_own_test_keys(self) -> None:
    """The decoupled pass re-runs 28 of cpu-unit worker 1's own tests in the same run.

    Keying on the suite id rather than the flavor is what stops each of those 28 from
    overwriting the cpu-unit row of the same test and halving that run's history.
    """
    run = load_json("run.json")
    cpu_unit = junit.parse_junit_xml(read_fixture("cpu-unit-1.xml"), file_name="cpu-unit-1.xml")
    decoupled = junit.parse_junit_xml(read_fixture("decoupled-targeted.xml"), file_name="decoupled-targeted.xml")

    cpu_rows = rows.test_rows(run, "cpu-unit", 1, cpu_unit)
    nested_rows = rows.test_rows(run, "cpu-unit", 1, decoupled, suite_id="decoupled", nested_in="cpu-unit")
    shared = {(row.classname, row.name) for row in cpu_rows} & {(row.classname, row.name) for row in nested_rows}

    self.assertEqual(len(shared), 28)
    self.assertEqual({row.key() for row in cpu_rows} & {row.key() for row in nested_rows}, set())
    self.assertEqual(len({row.key() for row in cpu_rows}), len(cpu_rows))
    self.assertEqual(len({row.key() for row in nested_rows}), len(nested_rows))

  def test_every_job_of_a_real_run_gets_its_own_key(self) -> None:
    """All 54 jobs of run 33468578834 key apart, including the repeated worker names."""
    run = load_json("run.json")
    jobs = load_jobs("jobs.json")
    keys = {rows.job_row(run, job).key() for job in jobs}

    self.assertEqual(len(jobs), 54)
    self.assertEqual(len(keys), 54)


class JsonRoundTripTest(OfflineTestCase):
  """`from_json(to_json(row)) == row`, through a real JSON trip, for every row type."""

  def test_every_row_type_round_trips(self) -> None:
    """A row written to the store and read back is the same row, field for field."""
    for kind, row in self.one_row_of_each_kind().items():
      with self.subTest(kind=kind):
        payload = json.loads(json.dumps(rows.to_json(row)))
        self.assertEqual(rows.from_json(payload), row)
        self.assertIs(type(rows.from_json(payload)), type(row))

  def test_rows_full_of_none_round_trip(self) -> None:
    """None is a value, not a gap: a suite that published nothing must come back as None.

    A row rebuilt from defaults instead of from the payload would turn these into whatever
    the dataclass declares, so the round trip is what proves nothing is filled in.
    """
    run = load_json("run.json")
    cancelled_run = load_json(f"cancelled-job-{CANCELLED_RUN_ID}-run.json")
    empty_suite = rows.suite_row(run, junit.SuiteEntry(suite_id="gpu-unit", reason=junit.REASON_NO_FILE))
    source = named(load_jobs(f"cancelled-job-{CANCELLED_RUN_ID}-attempt1-jobs.json"), "Gate and Formalize Parameters")
    # Synthesised: no job with null timestamps could be captured, because GitHub fills both
    # fields on every completed job. Nulls exist only while a job is queued or running, and
    # the repository had no run in flight while the fixtures were taken.
    queued_job = dict(source, status="queued", conclusion=None, started_at=None, completed_at=None, steps=[])
    running_run = dict(run, status="in_progress", conclusion=None)
    never_rescued = rows.failed_never_rescued_rows(
        cancelled_run,
        attempts_of(CANCELLED_RUN_ID, "cancelled-job", 3),
    )[0]
    queued_row = rows.job_row(cancelled_run, queued_job)

    self.assertIsNone(queued_row.started_at)
    self.assertIsNone(queued_row.completed_at)
    self.assertEqual(queued_row.created_at, source["created_at"])
    for row in (empty_suite, queued_row, rows.run_row(running_run), never_rescued):
      with self.subTest(kind=rows.row_kind(row)):
        payload = json.loads(json.dumps(rows.to_json(row)))
        self.assertIn(None, payload.values())
        self.assertEqual(rows.from_json(payload), row)

  def test_the_payload_holds_only_types_json_can_carry_back(self) -> None:
    """No tuples and no integer keys, because neither survives a JSON trip unchanged."""
    for kind, row in self.one_row_of_each_kind().items():
      with self.subTest(kind=kind):
        payload = rows.to_json(row)
        self.assertTrue(json_types_ok(payload))
        self.assertEqual(json.loads(json.dumps(payload)), payload)

  def test_a_suites_missing_workers_survive_the_trip(self) -> None:
    """Missing workers are a list of dicts, because a map keyed by worker number would not.

    JSON turns an integer key into a string, so a `{2: "artifact_expired"}` map would come
    back as `{"2": ...}` and the row would no longer equal itself.
    """
    entry = self.tpu_unit_suite_entry(missing={2: junit.REASON_ARTIFACT_EXPIRED})
    row = rows.suite_row(load_json("run.json"), entry)
    payload = json.loads(json.dumps(rows.to_json(row)))

    self.assertEqual(row.missing_workers, [{"worker": 2, "reason": junit.REASON_ARTIFACT_EXPIRED}])
    self.assertEqual(rows.from_json(payload), row)
    self.assertEqual(rows.from_json(payload).missing_workers[0]["worker"], 2)

  def test_the_kind_travels_inside_the_row(self) -> None:
    """The kind is what lets the reader rebuild the right type, so it is in the payload."""
    for kind, row in self.one_row_of_each_kind().items():
      with self.subTest(kind=kind):
        self.assertEqual(rows.to_json(row)["kind"], kind)
        self.assertEqual(rows.row_kind(row), kind)

  def test_a_row_from_a_newer_schema_is_refused_not_guessed_at(self) -> None:
    """Reading a row written by a later collector wrong is worse than refusing it."""
    payload = rows.to_json(self.one_row_of_each_kind()[rows.KIND_SUITE])
    with self.assertRaises(rows.RowError) as caught:
      rows.from_json(dict(payload, v=rows.ROW_VERSION + 1))

    self.assertIn(str(rows.ROW_VERSION + 1), str(caught.exception))
    self.assertIn(str(rows.ROW_VERSION), str(caught.exception))

  def test_a_payload_that_does_not_match_the_schema_is_refused(self) -> None:
    """A missing field, an extra field, an unknown kind and a non-integer version all raise.

    An unexpected field is an error rather than something to drop, because dropping it would
    lose data the store already holds.
    """
    payload = rows.to_json(self.one_row_of_each_kind()[rows.KIND_JOB])
    cases = {
        "missing": {name: value for name, value in payload.items() if name != "runner_name"},
        "unexpected": dict(payload, queue_seconds=109),
        "unknown kind": dict(payload, kind="metric"),
        "version is a string": dict(payload, v="1"),
        "version is a bool": dict(payload, v=True),
    }
    for label, bad in cases.items():
      with self.subTest(case=label):
        with self.assertRaises(rows.RowError):
          rows.from_json(bad)

  def test_row_kind_refuses_something_that_is_not_a_row(self) -> None:
    """The writer picks a file by kind, so it must never be handed a stranger."""
    with self.assertRaises(rows.RowError) as caught:
      rows.row_kind(object())
    self.assertIn("not a stored row type", str(caught.exception))


class RescueRowsTest(OfflineTestCase):
  """Covers `rescue_rows`: failure then success, by job name, inside one run."""

  def test_the_three_rescues_of_the_two_attempt_run(self) -> None:
    """Run 32772626658 has exactly three, and each names both jobs of its pair."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    built = {row.job_name: row for row in rows.rescue_rows(run, attempts_of(RERUN_ID, "rerun", 2))}

    self.assertEqual(sorted(built), sorted(RESCUES_32772626658))
    for name, truth in RESCUES_32772626658.items():
      with self.subTest(job=name):
        row = built[name]
        self.assertTrue(row.rescued)
        self.assertEqual(row.failed_attempt, 1)
        self.assertEqual(row.rescued_attempt, 2)
        self.assertEqual(row.failed_job_id, truth["failed_job_id"])
        self.assertEqual(row.rescued_job_id, truth["rescued_job_id"])
        self.assertEqual(row.failed_conclusion, "failure")
        self.assertEqual(row.final_attempt, 2)
        self.assertEqual(row.final_conclusion, "success")
        self.assertEqual(row.attempts_seen, [1, 2])

  def test_the_pair_ids_differ_because_github_mints_a_new_id_each_attempt(self) -> None:
    """The failed job and the passing job are two different ids of one name."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    for row in rows.rescue_rows(run, attempts_of(RERUN_ID, "rerun", 2)):
      with self.subTest(job=row.job_name):
        self.assertNotEqual(row.failed_job_id, row.rescued_job_id)

  def test_the_stored_timestamps_bracket_the_measured_waste(self) -> None:
    """The row keeps the failed job's span; the seconds themselves are `derive.py`'s job."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    built = {row.job_name: row for row in rows.rescue_rows(run, attempts_of(RERUN_ID, "rerun", 2))}
    total = 0.0

    for name, truth in RESCUES_32772626658.items():
      with self.subTest(job=name):
        row = built[name]
        self.assertEqual(row.failed_started_at, truth["failed_started_at"])
        self.assertEqual(row.failed_completed_at, truth["failed_completed_at"])
        measured = span_seconds(row.failed_started_at, row.failed_completed_at)
        self.assertEqual(measured, truth["seconds"])
        self.assertNotIn(measured, numbers_in(rows.to_json(row)))
        total += measured

    self.assertEqual(total, 365)

  def test_a_job_that_failed_and_failed_again_gets_no_rescue_row(self) -> None:
    """Five names of the same run failed twice. None of them is a rescue."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    attempts = attempts_of(RERUN_ID, "rerun", 2)
    names = {row.job_name for row in rows.rescue_rows(run, attempts)}

    for name in FAILED_AGAIN_32772626658:
      with self.subTest(job=name):
        self.assertEqual(named(attempts[1], name)["conclusion"], "failure")
        self.assertEqual(named(attempts[2], name)["conclusion"], "failure")
        self.assertNotIn(name, names)

  def test_every_attempt_one_failure_was_rescued_in_the_clean_run(self) -> None:
    """Run 33037584699 rescued all four of its failures, across three device lanes."""
    run = load_json(f"rerun-{CLEAN_RERUN_ID}-run.json")
    built = {row.job_name: row for row in rows.rescue_rows(run, attempts_of(CLEAN_RERUN_ID, "rerun", 2))}

    self.assertEqual(sorted(built), sorted(RESCUES_33037584699))
    for name, seconds in RESCUES_33037584699.items():
      with self.subTest(job=name):
        row = built[name]
        self.assertTrue(row.rescued)
        self.assertEqual(span_seconds(row.failed_started_at, row.failed_completed_at), seconds)
    self.assertEqual(sum(RESCUES_33037584699.values()), 2634)

  def test_cancelled_then_success_is_not_a_rescue(self) -> None:
    """Run 32785979907 yields zero rescues over three attempts.

    Its tpu-unit worker 2 goes cancelled -> success -> success. A cancelled job was never a
    failure, so counting it would inflate the flaky card with runs nobody's code broke.
    """
    run = load_json(f"cancelled-job-{CANCELLED_RUN_ID}-run.json")
    attempts = attempts_of(CANCELLED_RUN_ID, "cancelled-job", 3)
    recovered = "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit"

    self.assertEqual(named(attempts[1], recovered)["conclusion"], "cancelled")
    self.assertEqual(named(attempts[2], recovered)["conclusion"], "success")
    self.assertEqual(rows.rescue_rows(run, attempts), [])

  def test_the_run_and_lane_fields_come_off_the_run_and_the_failed_job(self) -> None:
    """A rescue row carries enough context to be read without joining anything back."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    name = "CPU Pretrain Tests (cpu-unit) / Execute Tests (3) / cpu-unit"
    row = {r.job_name: r for r in rows.rescue_rows(run, attempts_of(RERUN_ID, "rerun", 2))}[name]

    self.assertEqual(row.run_id, RERUN_ID)
    self.assertEqual(row.labels, ["linux-x86-n2-32"])
    self.assertEqual(row.head_branch, run["head_branch"])
    self.assertEqual(row.workflow_name, run["name"])
    self.assertEqual(row.event, run["event"])
    self.assertEqual(row.html_url, named(attempts_of(RERUN_ID, "rerun", 2)[1], name)["html_url"])

  def test_attempt_keys_may_arrive_as_strings(self) -> None:
    """A JSON round trip of the attempts map leaves string keys; the answer must not change."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    attempts = attempts_of(RERUN_ID, "rerun", 2)
    as_strings = {str(number): jobs for number, jobs in attempts.items()}

    from_ints = rows.rescue_rows(run, attempts, collected_at="2026-09-01T12:00:00Z")
    from_strings = rows.rescue_rows(run, as_strings, collected_at="2026-09-01T12:00:00Z")

    self.assertEqual(from_ints, from_strings)
    self.assertEqual(from_strings[0].attempts_seen, [1, 2])


class FailedNeverRescuedRowsTest(OfflineTestCase):
  """Covers the other half of the flaky card: a failure that never recovered."""

  def test_the_never_rescued_stream_is_the_same_shape_with_rescued_false(self) -> None:
    """A failure that stayed a failure is a row, not an absence, and says so explicitly."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    built = {row.job_name: row for row in rows.failed_never_rescued_rows(run, attempts_of(RERUN_ID, "rerun", 2))}

    self.assertEqual(sorted(built), sorted(FAILED_AGAIN_32772626658 + FAILED_ONLY_IN_ATTEMPT_2))
    for name, row in built.items():
      with self.subTest(job=name):
        self.assertIsInstance(row, rows.RescueRow)
        self.assertFalse(row.rescued)
        self.assertIsNone(row.rescued_attempt)
        self.assertIsNone(row.rescued_job_id)
        self.assertIsNone(row.rescued_started_at)
        self.assertIsNone(row.rescued_completed_at)
        self.assertEqual(row.final_conclusion, "failure")
        self.assertEqual(row.failed_conclusion, "failure")

  def test_the_two_streams_do_not_overlap_on_this_run(self) -> None:
    """A name is either rescued or still failing; run 32772626658 splits 3 and 7."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    attempts = attempts_of(RERUN_ID, "rerun", 2)
    rescued = {row.job_name for row in rows.rescue_rows(run, attempts)}
    failing = {row.job_name for row in rows.failed_never_rescued_rows(run, attempts)}

    self.assertEqual(len(rescued), 3)
    self.assertEqual(len(failing), 7)
    self.assertEqual(rescued & failing, set())

  def test_rerun_after_failure_separates_the_two_shapes_of_failure(self) -> None:
    """False means nothing ever ran the job again - the dashboard's "never re-run" cell."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    built = {row.job_name: row for row in rows.failed_never_rescued_rows(run, attempts_of(RERUN_ID, "rerun", 2))}

    for name in FAILED_AGAIN_32772626658:
      with self.subTest(job=name, shape="re-run and failed again"):
        self.assertTrue(built[name].rerun_after_failure)
        self.assertEqual(built[name].failed_attempt, 1)
        self.assertEqual(built[name].final_attempt, 2)
    for name in FAILED_ONLY_IN_ATTEMPT_2:
      with self.subTest(job=name, shape="failed last, never re-run"):
        self.assertFalse(built[name].rerun_after_failure)
        self.assertEqual(built[name].failed_attempt, 2)
        self.assertEqual(built[name].final_attempt, 2)

  def test_a_name_absent_from_an_attempt_keeps_the_attempts_it_was_seen_in(self) -> None:
    """Run 32785979907's gate job exists only in attempts 2 and 3, and the row says so."""
    run = load_json(f"cancelled-job-{CANCELLED_RUN_ID}-run.json")
    attempts = attempts_of(CANCELLED_RUN_ID, "cancelled-job", 3)
    built = {row.job_name: row for row in rows.failed_never_rescued_rows(run, attempts)}

    self.assertEqual(sorted(built), sorted(NEVER_RESCUED_32785979907))
    gate = built["All Required Tests Passed"]
    self.assertEqual(gate.attempts_seen, [2, 3])
    self.assertEqual(gate.failed_attempt, 2)
    self.assertEqual(gate.final_attempt, 3)
    self.assertNotIn("All Required Tests Passed", [job["name"] for job in attempts[1]])
    self.assertEqual(built["TPU Pathways Unit Tests (1) / tpu-pathways-unit"].attempts_seen, [1, 2, 3])

  def test_a_clean_rerun_leaves_nothing_failing(self) -> None:
    """Run 33037584699 ended green, so the never-rescued stream is empty."""
    run = load_json(f"rerun-{CLEAN_RERUN_ID}-run.json")
    self.assertEqual(rows.failed_never_rescued_rows(run, attempts_of(CLEAN_RERUN_ID, "rerun", 2)), [])


class SuiteAndTestRowTest(OfflineTestCase):
  """Covers `suite_row` and `test_rows`, including the partial state junit.py works out."""

  def partial_run_tests(self) -> junit.RunTests:
    """Reads a run where cpu-unit worker 1 published and workers 3 and 4 expired.

    Returns:
      The `junit.RunTests` for that run, built through the real `read_run_tests`.
    """
    blobs = {
        "https://api.github.com/artifacts/1/zip": make_zip(
            {
                "test-results-cpu-unit-1.xml": read_fixture("cpu-unit-1.xml"),
                "test-results-decoupled-targeted.xml": read_fixture("decoupled-targeted.xml"),
            }
        )
    }
    payloads = [
        artifact_payload(f"test-results-cpu-unit-1-{RUN_ID}", 1),
        artifact_payload(f"test-results-cpu-unit-3-{RUN_ID}", 3, expired=True),
        artifact_payload(f"test-results-cpu-unit-4-{RUN_ID}", 4, expired=True),
    ]
    return junit.read_run_tests(StubClient(payloads, blobs), RUN_ID, flavors=["cpu-unit"])

  def test_a_suite_that_published_nothing_stores_none_and_a_reason(self) -> None:
    """Zero would draw a "tests vanished" alarm for a run that simply uploaded no file."""
    entry = junit.read_run_tests(StubClient([], {}), RUN_ID, flavors=["gpu-unit"]).suites["gpu-unit"]
    row = rows.suite_row(load_json("run.json"), entry)

    for name in ("collected", "skipped", "executed", "failed", "errored", "junit_seconds"):
      with self.subTest(field=name):
        self.assertIsNone(getattr(row, name))
    self.assertEqual(row.reason, junit.REASON_NO_FILE)
    self.assertFalse(row.has_result)
    self.assertFalse(row.is_partial)
    self.assertEqual(row.files, [])
    self.assertEqual(row.published_workers, [])

  def test_a_partial_suite_says_which_workers_are_missing(self) -> None:
    """A hole in the data must never read as a fall in the test count."""
    entry = self.partial_run_tests().suites["cpu-unit"]
    row = rows.suite_row(load_json("run.json"), entry)

    self.assertTrue(entry.is_partial)
    self.assertTrue(row.is_partial)
    self.assertTrue(row.has_result)
    self.assertIsNone(row.reason)
    self.assertEqual(row.published_workers, [1])
    self.assertEqual(
        row.missing_workers,
        [
            {"worker": 3, "reason": junit.REASON_ARTIFACT_EXPIRED},
            {"worker": 4, "reason": junit.REASON_ARTIFACT_EXPIRED},
        ],
    )
    self.assertEqual(row.collected, 737)
    self.assertEqual(row.executed, 720)

  def test_test_rows_carry_the_partial_state_through(self) -> None:
    """Every test row of a partial suite is flagged, so a reader never joins back to find out."""
    run_tests = self.partial_run_tests()
    entry = run_tests.suites["cpu-unit"]
    built = rows.test_rows(load_json("run.json"), "cpu-unit", 1, entry.result, suite_partial=entry.is_partial)

    self.assertEqual(len(built), 737)
    self.assertTrue(all(row.suite_partial for row in built))

  def test_a_complete_suite_is_not_flagged_partial(self) -> None:
    """The flag has to mean something, so a suite with every worker in it stays False."""
    entry = self.tpu_unit_suite_entry()
    row = rows.suite_row(load_json("run.json"), entry)
    built = rows.test_rows(load_json("run.json"), "tpu-unit", 1, entry.result, suite_partial=entry.is_partial)

    self.assertFalse(entry.is_partial)
    self.assertFalse(row.is_partial)
    self.assertFalse(any(test_row.suite_partial for test_row in built))

  def test_the_nested_pass_names_its_parent_flavor(self) -> None:
    """The decoupled suite keeps its own id and points at the flavor it ran inside."""
    entry = self.partial_run_tests().suites["decoupled"]
    row = rows.suite_row(load_json("run.json"), entry)
    built = rows.test_rows(load_json("run.json"), "cpu-unit", 1, entry.result, suite_id="decoupled", nested_in="cpu-unit")

    self.assertEqual(row.suite_id, "decoupled")
    self.assertEqual(row.flavor, "cpu-unit")
    self.assertEqual(row.nested_in, "cpu-unit")
    self.assertEqual(row.collected, 54)
    self.assertEqual(row.executed, 50)
    self.assertTrue(all(test_row.suite_id == "decoupled" for test_row in built))
    self.assertTrue(all(test_row.flavor == "cpu-unit" for test_row in built))
    self.assertTrue(all(test_row.nested_in == "cpu-unit" for test_row in built))

  def test_a_case_keeps_the_worker_its_own_file_named(self) -> None:
    """A merged multi-worker result still produces distinct keys, one per real worker."""
    result = junit.parse_junit_xml(read_fixture("tpu-unit-2.xml"), file_name="tpu-unit-2.xml")
    for case in result.tests:
      case.worker = 2

    built = rows.test_rows(load_json("run.json"), "tpu-unit", 1, result)
    self.assertTrue(all(row.worker == 2 for row in built))
    self.assertTrue(all("|2|" in row.key() for row in built))

  def test_a_case_without_a_worker_falls_back_to_the_artifacts_worker(self) -> None:
    """`parse_junit_xml` leaves `worker` None; the artifact name is what says which one."""
    result = junit.parse_junit_xml(read_fixture("tpu-unit-2.xml"), file_name="tpu-unit-2.xml")
    self.assertTrue(all(case.worker is None for case in result.tests))

    built = rows.test_rows(load_json("run.json"), "tpu-unit", 2, result)
    self.assertTrue(all(row.worker == 2 for row in built))

  def test_a_suite_with_no_cases_yields_no_test_rows(self) -> None:
    """Nothing published means no rows at all; the SuiteRow carries the reason instead."""
    empty = StubSuiteResult(collected=0, skipped=0, executed=0, junit_seconds=0.0, tests=[])
    self.assertEqual(rows.test_rows(load_json("run.json"), "gpu-unit", 1, empty), [])

  def test_the_failure_text_is_quoted_not_rewritten(self) -> None:
    """The one failing case in the whole capture keeps its message word for word."""
    result = junit.parse_junit_xml(
        read_fixture("tpu-post-training-integration-1.failed-run-33467756955.xml"),
        file_name="tpu-post-training-integration-1.failed-run-33467756955.xml",
    )
    built = rows.test_rows(load_json("run.json"), "tpu-post-training-integration", 1, result)
    failed = [row for row in built if row.status == junit.STATUS_FAILED]

    self.assertEqual(len(failed), 1)
    self.assertEqual(failed[0].name, "test_grpo_loss_drives_a_training_step")
    self.assertEqual(
        failed[0].failure_message,
        "AssertionError: nan not greater than 0.0 : no parameter moved, so no training happened",
    )
    self.assertTrue(all(row.failure_message is None for row in built if row.status != junit.STATUS_FAILED))


class HousekeepingFieldsTest(OfflineTestCase):
  """Every row carries a schema version and the moment it was written."""

  def test_every_row_carries_the_current_schema_version(self) -> None:
    """`v` travels with the row so a later reader can refuse what it cannot understand."""
    self.assertEqual(rows.ROW_VERSION, 1)
    for kind, row in self.one_row_of_each_kind().items():
      with self.subTest(kind=kind):
        self.assertEqual(row.v, rows.ROW_VERSION)
        self.assertEqual(rows.to_json(row)["v"], rows.ROW_VERSION)

  def test_every_row_carries_a_collected_at_in_githubs_own_format(self) -> None:
    """Seconds resolution, "Z" suffix, no offset, and it parses back as UTC."""
    for kind, row in self.one_row_of_each_kind().items():
      with self.subTest(kind=kind):
        self.assertTrue(row.collected_at.endswith("Z"))
        self.assertEqual(len(row.collected_at), 20)
        moment = iso(row.collected_at)
        self.assertEqual(moment.tzinfo, timezone.utc)
        self.assertEqual(moment.strftime(ISO_FORMAT), row.collected_at)
        self.assertEqual(datetime.fromisoformat(row.collected_at.replace("Z", "+00:00")), moment)

  def test_the_write_timestamp_is_now_in_utc(self) -> None:
    """`utc_now_iso` reads the clock in UTC, not in whatever zone the runner sits in."""
    frozen = datetime(2026, 9, 1, 4, 6, 1, tzinfo=timezone.utc)

    class FrozenClock:
      """A `datetime` stand-in whose `now` never moves."""

      @staticmethod
      def now(tz: timezone | None = None) -> datetime:
        """Returns the frozen moment.

        Args:
          tz: The zone asked for; the frozen moment is already UTC.

        Returns:
          The frozen moment.
        """
        return frozen.astimezone(tz) if tz else frozen

    with mock.patch.object(rows, "datetime", FrozenClock):
      self.assertEqual(rows.utc_now_iso(), "2026-09-01T04:06:01Z")
      self.assertEqual(rows.RunRow(run_id=1, attempt=1).collected_at, "2026-09-01T04:06:01Z")
      self.assertEqual(
          rows.job_row(load_json("run.json"), named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1)).collected_at,
          "2026-09-01T04:06:01Z",
      )

  def test_the_write_timestamp_can_be_pinned_by_the_caller(self) -> None:
    """One tick writes one timestamp, so a correction can be told apart by its later one."""
    run = load_json("run.json")
    job = named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1)
    entry = self.tpu_unit_suite_entry()
    stamp = "2026-09-01T08:00:00Z"
    built = [
        rows.run_row(run, collected_at=stamp),
        rows.job_row(run, job, collected_at=stamp),
        rows.suite_row(run, entry, collected_at=stamp),
        *rows.test_rows(run, "tpu-unit", 1, entry.result, collected_at=stamp),
        *rows.rescue_rows(load_json(f"rerun-{RERUN_ID}-run.json"), attempts_of(RERUN_ID, "rerun", 2), collected_at=stamp),
    ]

    self.assertEqual({row.collected_at for row in built}, {stamp})

  def test_a_later_row_of_the_same_key_is_how_a_correction_is_stored(self) -> None:
    """The store is append-only: same key, later `collected_at`, and readers keep the last."""
    run = load_json("run.json")
    first = rows.run_row(run, collected_at="2026-09-01T04:00:00Z")
    corrected = rows.run_row(dict(run, conclusion="failure"), collected_at="2026-09-01T08:00:00Z")

    self.assertEqual(first.key(), corrected.key())
    self.assertLess(iso(first.collected_at), iso(corrected.collected_at))
    self.assertNotEqual(first.conclusion, corrected.conclusion)


class RawApiValuesTest(OfflineTestCase):
  """Nothing in this module is arithmetic: every field is what GitHub or junit.py said."""

  def test_no_row_type_declares_a_computed_field(self) -> None:
    """A duration, a wait or a rate on a stored row would freeze today's rule into history.

    `TestRow.duration` is the `<testcase time>` attribute and `SuiteRow.junit_seconds` is the
    sum junit.py made of those attributes; every other second belongs to `derive.py`.
    """
    allowed_time_words = {rows.TestRow: ("duration",), rows.SuiteRow: ("junit_seconds", "suite_seconds")}
    for row_type in (rows.RunRow, rows.JobRow, rows.SuiteRow, rows.TestRow, rows.RescueRow):
      allowed = allowed_time_words.get(row_type, ())
      for row_field in dataclasses.fields(row_type):
        with self.subTest(row=row_type.__name__, field=row_field.name):
          for word in DERIVED_WORDS:
            self.assertNotIn(word, row_field.name)
          if row_field.name not in allowed:
            for word in TIME_WORDS:
              self.assertNotIn(word, row_field.name)

  def test_a_run_row_is_the_run_payloads_own_values(self) -> None:
    """Field for field, with the nested objects flattened and nothing else touched."""
    run = load_json("run.json")
    row = rows.run_row(run)

    for name, source in RUN_FIELD_SOURCE.items():
      with self.subTest(field=name):
        self.assertEqual(getattr(row, name), run[source])
    self.assertEqual(row.repository, run["repository"]["full_name"])
    self.assertEqual(row.head_repository, run["head_repository"]["full_name"])
    self.assertEqual(row.actor, run["actor"]["login"])
    self.assertEqual(row.triggering_actor, run["triggering_actor"]["login"])

  def test_a_job_row_is_the_job_payloads_own_values(self) -> None:
    """Including the timestamps that `derive.py` will later difference."""
    run = load_json("run.json")
    job = named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1)
    row = rows.job_row(run, job)

    for name, source in JOB_FIELD_SOURCE.items():
      with self.subTest(field=name):
        self.assertEqual(getattr(row, name), job[source])
    self.assertEqual(row.created_at, TPU_UNIT_WORKER_1_CREATED)
    self.assertEqual(row.started_at, TPU_UNIT_WORKER_1_STARTED)
    self.assertEqual(row.completed_at, TPU_UNIT_WORKER_1_COMPLETED)

  def test_the_measured_queue_and_setup_seconds_are_nowhere_in_the_row(self) -> None:
    """This worker waited 109 s and set up in 71 s. The row stores the timestamps, not those.

    The two spans are measured here from the stored strings, which is the whole point of
    keeping them: the rule can change and the history can be recomputed.
    """
    run = load_json("run.json")
    row = rows.job_row(run, named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1))
    run_tests = [step for step in row.steps if step["name"] == "Run Tests"][0]

    self.assertEqual(span_seconds(row.created_at, row.started_at), 109)
    self.assertEqual(span_seconds(row.started_at, run_tests["started_at"]), 71)
    for measured in (109, 71, 1217):
      with self.subTest(seconds=measured):
        self.assertNotIn(measured, numbers_in(rows.to_json(row)))

  def test_the_steps_are_copied_verbatim_and_pinned_to_the_reported_fields(self) -> None:
    """The "Run Tests" span is the suite duration, so the step timestamps have to survive."""
    run = load_json("run.json")
    job = named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1)
    row = rows.job_row(run, job)

    self.assertEqual(len(row.steps), len(job["steps"]))
    self.assertEqual([step["name"] for step in row.steps], [step["name"] for step in job["steps"]])
    for stored, source in zip(row.steps, job["steps"]):
      with self.subTest(step=stored["name"]):
        self.assertEqual(set(stored), set(rows.STEP_FIELDS))
        for name in rows.STEP_FIELDS:
          self.assertEqual(stored[name], source.get(name))

  def test_a_step_field_github_does_not_report_is_dropped(self) -> None:
    """Every stored job row keeps the same shape, so the JSON stays round-trippable."""
    run = load_json("run.json")
    job = {
        "id": 1,
        "run_id": RUN_ID,
        "run_attempt": 1,
        "name": "made up",
        "steps": [
            {
                "name": "Run Tests",
                "number": 7,
                "status": "completed",
                "conclusion": "success",
                "started_at": "2026-09-01T04:11:43Z",
                "completed_at": "2026-09-01T04:30:32Z",
                "surprise": "not a field the endpoint reports",
            }
        ],
    }
    self.assertEqual(set(rows.job_row(run, job).steps[0]), set(rows.STEP_FIELDS))

  def test_a_carried_over_job_keeps_its_impossible_order(self) -> None:
    """GitHub carries a job it did not re-run into the next attempt with that attempt's
    `created_at`, so `started_at` precedes it and the queue wait computes negative.

    Storing the strings and nothing else is what lets `derive.py` recognise that shape and
    drop the job from the attempt instead of charging it a negative wait.
    """
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    job = named(load_jobs(f"rerun-{RERUN_ID}-attempt2-jobs.json"), "Analyze Code Changes for Test Orchestration")
    row = rows.job_row(run, job)

    self.assertEqual(row.attempt, 2)
    self.assertEqual(row.created_at, "2026-08-25T02:37:18Z")
    self.assertEqual(row.started_at, "2026-08-24T20:13:12Z")
    self.assertLess(span_seconds(row.created_at, row.started_at), 0)
    self.assertTrue(all(number >= 0 for number in numbers_in(rows.to_json(row))))

  def test_a_job_that_never_held_a_runner_is_stored_as_it_was_reported(self) -> None:
    """Eight jobs of run 32999133815 were cancelled while queued: `steps` is empty, and
    `created_at` equals `started_at`. Neither is turned into a zero or a None here."""
    run = load_json(f"queued-then-cancelled-{QUEUED_RUN_ID}-run.json")
    jobs = load_jobs(f"queued-then-cancelled-{QUEUED_RUN_ID}-attempt1-jobs.json")
    never_started = [
        job
        for job in jobs
        if job["conclusion"] == "cancelled" and not job["steps"] and job["created_at"] == job["started_at"]
    ]

    self.assertEqual(len(never_started), 8)
    for job in never_started:
      with self.subTest(job=job["name"]):
        row = rows.job_row(run, job)
        self.assertEqual(row.steps, [])
        self.assertEqual(row.created_at, row.started_at)
        self.assertEqual(row.completed_at, job["completed_at"])
        self.assertEqual(row.conclusion, "cancelled")

  def test_a_skipped_job_keeps_its_empty_labels(self) -> None:
    """No labels means no runner and no device lane. An empty list is the answer, not a gap."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    job = named(load_jobs(f"rerun-{RERUN_ID}-attempt2-jobs.json"), "Jupyter Notebook Tests")
    row = rows.job_row(run, job)

    self.assertEqual(row.conclusion, "skipped")
    self.assertEqual(row.labels, [])
    self.assertEqual(row.steps, [])
    self.assertIsNone(row.runner_name)
    self.assertIsNone(row.runner_id)

  def test_the_counts_are_copied_from_junit_not_recomputed(self) -> None:
    """Handed counts that disagree with each other, the row stores them as they came.

    A real `SuiteResult` always satisfies executed == collected - skipped, so only a stub can
    show that this module does not do that subtraction itself.
    """
    entry = StubSuiteEntry(
        suite_id="cpu-unit",
        result=StubSuiteResult(
            collected=100,
            skipped=10,
            executed=42,
            junit_seconds=1055.921,
            failed=3,
            errored=1,
            reported_tests=870,
            suite_seconds=177.756,
            files=("cpu-unit-1.xml",),
        ),
        per_worker={1: None},
    )
    row = rows.suite_row(load_json("run.json"), entry)

    self.assertEqual(row.collected, 100)
    self.assertEqual(row.skipped, 10)
    self.assertEqual(row.executed, 42)
    self.assertEqual(row.failed, 3)
    self.assertEqual(row.errored, 1)
    self.assertEqual(row.junit_seconds, 1055.921)
    self.assertEqual(row.reported_tests, 870)
    self.assertEqual(row.suite_seconds, 177.756)
    self.assertEqual(row.files, ["cpu-unit-1.xml"])

  def test_a_test_rows_duration_is_the_files_own_time_attribute(self) -> None:
    """The JUnit seconds run 1.55x the wall clock on the CPU flavors; the row keeps them raw."""
    result = junit.parse_junit_xml(read_fixture("tpu-unit-1.xml"), file_name="tpu-unit-1.xml")
    built = rows.test_rows(load_json("run.json"), "tpu-unit", 1, result)

    self.assertEqual(len(built), len(result.tests))
    for row, case in zip(built, result.tests):
      with self.subTest(test=case.name):
        self.assertEqual(row.duration, case.duration)
        self.assertEqual(row.status, case.status)
        self.assertEqual(row.classname, case.classname)
        self.assertEqual(row.name, case.name)
    self.assertEqual(round(sum(row.duration for row in built), 3), 1055.519)


class BuilderErrorTest(OfflineTestCase):
  """A payload that cannot be turned into a row says which field was at fault."""

  def test_a_run_without_an_id_or_an_attempt_is_refused(self) -> None:
    """Both are key parts, so neither can be guessed at."""
    run = load_json("run.json")
    for missing in ("id", "run_attempt"):
      with self.subTest(field=missing):
        payload = {name: value for name, value in run.items() if name != missing}
        with self.assertRaises(rows.RowError) as caught:
          rows.run_row(payload)
        self.assertIn(missing, str(caught.exception))

  def test_a_non_numeric_id_is_refused(self) -> None:
    """An id that is not a number would key a row nothing could find again."""
    run = dict(load_json("run.json"), id="thirty three billion")
    with self.assertRaises(rows.RowError) as caught:
      rows.run_row(run)
    self.assertIn("id", str(caught.exception))

  def test_a_job_belonging_to_another_run_is_refused(self) -> None:
    """Pairing a job with the wrong run would file it under a run it never belonged to."""
    job = named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1)
    with self.assertRaises(rows.RowError) as caught:
      rows.job_row(load_json(f"rerun-{RERUN_ID}-run.json"), job)

    self.assertIn(str(RUN_ID), str(caught.exception))
    self.assertIn(str(RERUN_ID), str(caught.exception))

  def test_the_jobs_own_attempt_wins_over_the_runs(self) -> None:
    """A run's `run_attempt` moves while the collector works; a job's does not.

    Run 32785979907 reads `run_attempt` 3, but its attempt-1 jobs each say 1, and that is
    the attempt those rows belong to.
    """
    run = load_json(f"cancelled-job-{CANCELLED_RUN_ID}-run.json")
    self.assertEqual(run["run_attempt"], 3)

    for attempt, jobs in attempts_of(CANCELLED_RUN_ID, "cancelled-job", 3).items():
      with self.subTest(attempt=attempt):
        self.assertEqual({rows.job_row(run, job).attempt for job in jobs}, {attempt})

  def test_a_job_without_a_name_is_refused(self) -> None:
    """The name is the rescue key and the only place the worker and flavor are written."""
    run = load_json("run.json")
    job = {name: value for name, value in named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1).items() if name != "name"}
    with self.assertRaises(rows.RowError) as caught:
      rows.job_row(run, job)
    self.assertIn("name", str(caught.exception))

  def test_a_step_that_is_not_an_object_is_refused(self) -> None:
    """A shape the endpoint should never return is an error naming the job and the step."""
    run = load_json("run.json")
    job = dict(named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1), steps=["Run Tests"])
    with self.assertRaises(rows.RowError) as caught:
      rows.job_row(run, job)

    self.assertIn(str(TPU_UNIT_WORKER_1_ID), str(caught.exception))
    self.assertIn("expected an object", str(caught.exception))

  def test_an_attempts_map_that_is_not_a_map_of_lists_is_refused(self) -> None:
    """The rescue pass walks attempts in order, so it has to be handed attempts."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    jobs = load_jobs(f"rerun-{RERUN_ID}-attempt1-jobs.json")
    cases = {
        "attempt key is not a number": {"first": jobs},
        "attempt holds one job": {1: jobs[0]},
        "attempt holds a string": {1: ["a job"]},
    }
    for label, bad in cases.items():
      with self.subTest(case=label):
        with self.assertRaises(rows.RowError):
          rows.rescue_rows(run, bad)


class FailureAfterARescueTest(OfflineTestCase):
  """A job that fails, is rescued, then fails again writes two rows that do not collide.

  This was a reported defect and is now the fix's proof. Both builders emit a row for the
  name, but the key carries the failed attempt, so the rescue is `...|1` and the unrescued
  failure is `...|3`. Neither overwrites the other under the store's "keep the last row per
  key" rule, and the unrescued row points at the attempt-3 failure rather than re-claiming
  the seconds `derive.py` already charged as the rescue's waste.

  No fixture holds this shape - a third attempt would be needed - so it is synthesised from
  two real ones: the rescued cpu-unit worker of run 32772626658, given a third attempt in
  which it fails. Three-attempt runs are real (32785979907), and `run_attempt` is not stable,
  so a run growing an attempt after a rescue is the case the module's own docstring names.
  """

  def third_attempt_that_fails(self) -> dict[int, list[dict[str, Any]]]:
    """Builds the synthetic attempt map: the rescued job fails again in attempt 3.

    Returns:
      Attempt number -> jobs, with a one-job third attempt.
    """
    attempts = attempts_of(RERUN_ID, "rerun", 2)
    name = "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit"
    failed_again = dict(
        named(attempts[2], name),
        id=99000000001,
        run_attempt=3,
        conclusion="failure",
        created_at="2026-08-25T05:00:00Z",
        started_at="2026-08-25T05:00:10Z",
        completed_at="2026-08-25T05:04:00Z",
    )
    attempts[3] = [failed_again]
    return attempts

  def test_the_two_rows_have_different_keys_so_neither_is_lost(self) -> None:
    """The rescue keys on attempt 1 and the unrescued failure on attempt 3."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    attempts = self.third_attempt_that_fails()
    name = "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit"
    stamp = "2026-09-01T12:00:00Z"

    rescued = [row for row in rows.rescue_rows(run, attempts, collected_at=stamp) if row.job_name == name]
    failing = [row for row in rows.failed_never_rescued_rows(run, attempts, collected_at=stamp) if row.job_name == name]

    self.assertEqual(len(rescued), 1)
    self.assertEqual(len(failing), 1)
    self.assertNotEqual(rescued[0].key(), failing[0].key())
    self.assertTrue(rescued[0].key().endswith("|1"))
    self.assertTrue(failing[0].key().endswith("|3"))
    self.assertEqual(rescued[0].collected_at, failing[0].collected_at)
    self.assertTrue(rescued[0].rescued)
    self.assertFalse(failing[0].rescued)

  def test_the_never_rescued_row_points_at_the_failure_that_was_never_rescued(self) -> None:
    """It names attempt 3, so it does not re-claim the seconds the rescue already spent."""
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    attempts = self.third_attempt_that_fails()
    name = "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit"
    truth = RESCUES_32772626658[name]

    failing = [row for row in rows.failed_never_rescued_rows(run, attempts) if row.job_name == name][0]

    self.assertEqual(failing.failed_attempt, 3)
    self.assertEqual(failing.failed_job_id, 99000000001)
    self.assertEqual(failing.failed_started_at, "2026-08-25T05:00:10Z")
    self.assertEqual(span_seconds(failing.failed_started_at, failing.failed_completed_at), 230.0)
    self.assertEqual(failing.final_attempt, 3)
    self.assertFalse(failing.rerun_after_failure)
    self.assertNotEqual(failing.failed_job_id, truth["failed_job_id"])

  def test_a_job_that_failed_three_times_still_reports_the_first_failure(self) -> None:
    """Without a rescue in between, the streak starts at attempt 1 and the re-runs stay visible.

    This is what stops the trailing-streak rule from turning every repeated failure into a
    "failed, never re-run" cell: `rerun_after_failure` has to stay True here.
    """
    run = load_json(f"rerun-{RERUN_ID}-run.json")
    name = "CPU Pretrain Tests (cpu-unit) / Execute Tests (2) / cpu-unit"

    failing = [
        row for row in rows.failed_never_rescued_rows(run, attempts_of(RERUN_ID, "rerun", 2)) if row.job_name == name
    ]

    self.assertEqual(len(failing), 1)
    self.assertEqual(failing[0].failed_attempt, 1)
    self.assertEqual(failing[0].final_attempt, 2)
    self.assertTrue(failing[0].rerun_after_failure)


if __name__ == "__main__":
  unittest.main(verbosity=2)
