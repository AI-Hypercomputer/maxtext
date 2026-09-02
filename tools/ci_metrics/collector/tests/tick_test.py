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

"""Offline tests for `collector.tick`: a whole tick, run end to end, with no network.

Nothing here opens a socket. The base test case replaces `socket.socket`, so a test that
reached for GitHub would fail loudly instead of hanging or spending rate limit, and
`tick.main` is handed a `FakeClient` in place of the `github.GitHubClient` it would build
for itself. The fake answers from the saved fixtures in `tests/fixtures/` and records every
path, query parameter and download URL it was asked for, which is how the window tests
prove what the collector asked the API for rather than only what it stored.

Two moments are pinned so a store written twice is byte-identical: `tick.utc_now`, which
every window, artifact-age and view timestamp is measured from, and `rows.utc_now_iso`,
which stamps `state.json` and any row built without an explicit `collected_at`.

The scenario is two real runs, and every expected number below was measured from the store
they produce, not chosen to make a test pass:

  1. Run 33468578834 is the scheduled run on main of 2026-09-01T04:06:01Z: 54 jobs, and the
     day's first scheduled main run, so it keeps every test row rather than the two-tier
     selection. Its artifacts are hours old at the pinned `now`, so they are downloaded.
     Seven of its 28 artifacts are served here, the seven the repository has saved JUnit
     files for, so ten suites of the sixteen it reports have no file and are stored as null
     with a reason - which is the rule that a suite with no file is not a suite with no
     tests.
  2. Run 32772626658 is a pull-request run with two attempts, 42 jobs each: three jobs
     failed on attempt 1 and passed on the re-run, and seven ended in failure without ever
     passing, so it yields ten rescue rows, three of them rescues. It was created on
     2026-08-24, more than a day before the pinned `now`, so its artifacts are counted as
     too old to have rather than fetched.

Two payloads in this file are built by hand rather than saved, and both are marked where
they are defined: pull request #4980, because the repository saved the short entry the run
carries but not the full object the collector re-reads, and the workflows listing, because
no capture of `GET /actions/workflows` was kept. Everything else is a fixture.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/tick_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/tick_test.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from unittest import mock

import requests

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import github
from collector import rows
from collector import runs
from collector import store as store_module
from collector import tick
from collector import views

FIXTURES = Path(__file__).resolve().parent / "fixtures"

OWNER = "AI-Hypercomputer"
REPO = "maxtext"

# The moment every test measures from. Chosen so run 33468578834 (04:06 that morning) is
# inside the 24-hour artifact window and run 32772626658 (2026-08-24) is outside it.
NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW_ISO = "2026-09-01T12:00:00Z"

# The scheduled run on main, and its month.
SCHEDULED_RUN_ID = 33468578834
SCHEDULED_MONTH = "2026-09"
SCHEDULED_JOB_COUNT = 54

# The two-attempt pull-request run, its pull request, and its month.
RERUN_RUN_ID = 32772626658
RERUN_PULL_NUMBER = 4980
RERUN_MONTH = "2026-08"
RERUN_JOB_COUNT = 42
RERUN_ATTEMPTS = 2
# Three jobs failed on attempt 1 and passed on the re-run; seven ended in failure without
# ever passing. Ten rescue rows in all, of which three carry rescued=true.
RERUN_RESCUE_ROWS = 10
RERUN_RESCUED_ROWS = 3

# The numeric id of .github/workflows/ci_pipeline.yml, read from every saved run payload.
CI_PIPELINE_WORKFLOW_ID = 297948505
# The image pipelines' ids are NOT in any saved fixture. They only have to be distinct from
# each other and from ci_pipeline's, because the tests assert on what was asked of the
# ci_pipeline endpoint.
TPU_IMAGES_WORKFLOW_ID = 100000001
GPU_IMAGES_WORKFLOW_ID = 100000002

CI_PIPELINE_RUNS_PATH = f"actions/workflows/{CI_PIPELINE_WORKFLOW_ID}/runs"

# The seven artifacts of run 33468578834 the repository saved JUnit files for, and the files
# inside each. Worker 1 of cpu-unit carries the nested decoupled pass as a second file.
ARTIFACT_FILES: dict[str, dict[str, str]] = {
    "test-results-cpu-unit-1-33468578834": {
        "test-results-cpu-unit-1.xml": "cpu-unit-1.xml",
        "test-results-decoupled-targeted.xml": "decoupled-targeted.xml",
    },
    "test-results-cpu-unit-3-33468578834": {"test-results-cpu-unit-3.xml": "cpu-unit-3.xml"},
    "test-results-cpu-post-training-unit-4-33468578834": {
        "test-results-cpu-post-training-unit-4.xml": "cpu-post-training-unit-4.xml"
    },
    "test-results-gpu-integration-1-33468578834": {"test-results-gpu-integration-1.xml": "gpu-integration-1.xml"},
    "test-results-tpu-post-training-integration-1-33468578834": {
        "test-results-tpu-post-training-integration-1.xml": "tpu-post-training-integration-1.xml"
    },
    "test-results-tpu-unit-1-33468578834": {"test-results-tpu-unit-1.xml": "tpu-unit-1.xml"},
    "test-results-tpu-unit-2-33468578834": {"test-results-tpu-unit-2.xml": "tpu-unit-2.xml"},
}

# Rows the scenario writes, counted from the store it produces.
EXPECTED_RUN_ROWS = {RERUN_MONTH: 2, SCHEDULED_MONTH: 1}
EXPECTED_JOB_ROWS = {RERUN_MONTH: 84, SCHEDULED_MONTH: 54}
EXPECTED_SUITE_ROWS = {SCHEDULED_MONTH: 16}
EXPECTED_TEST_ROWS = {SCHEDULED_MONTH: 1850}
EXPECTED_RESCUE_ROWS = {RERUN_MONTH: 10}

# Per-suite totals of run 33468578834, as the seven served artifacts report them. cpu-unit is
# two workers merged (737 + 737 collected, 720 + 0 executed); decoupled is nested inside it
# and its 54 tests are already inside cpu-unit's totals.
EXPECTED_SUITE_TOTALS = {
    "cpu-post-training-unit": (84, 7, 77),
    "cpu-unit": (1474, 754, 720),
    "decoupled": (54, 4, 50),
    "gpu-integration": (26, 15, 11),
    "tpu-post-training-integration": (9, 7, 2),
    "tpu-unit": (203, 6, 197),
}

# The suites the run really executed that published no JUnit file in this scenario. The two
# Pathways flavors publish none on the real pipeline either; the rest are simply not served
# here, which is the same thing as far as the collector can tell.
EXPECTED_NO_FILE_SUITES = (
    "cpu-integration",
    "cpu-post-training-integration",
    "gpu-unit",
    "tpu-integration",
    "tpu-pathways-integration",
    "tpu-pathways-unit",
    "tpu-post-training-unit",
    "tpu7x-integration",
    "tpu7x-post-training-unit",
    "tpu7x-unit",
)

# Built by hand: the repository saved the SHORT pull request entry that run 32772626658
# carries, but not the full object `resolve_pull_request` re-reads for its merge time. The
# number, branch and head sha are the run's own; the rest is the field set `rows.run_row`
# reads off a pull request.
PULL_4980_OPEN: dict[str, Any] = {
    "number": RERUN_PULL_NUMBER,
    "state": "open",
    "title": "Test the CPU unit lane under a re-run",
    "merged_at": None,
    "created_at": "2026-08-24T19:52:00Z",
    "closed_at": None,
    "merge_commit_sha": None,
    "draft": False,
    "html_url": f"https://github.com/{OWNER}/{REPO}/pull/{RERUN_PULL_NUMBER}",
    "user": {"login": "a-contributor"},
    "head": {
        "ref": "test_969931330",
        "label": f"{OWNER}:test_969931330",
        "sha": "bd39a005fd7e6df4744b14f3817867644d63cee3",
    },
    "base": {"ref": "main"},
}
PULL_4980_MERGED: dict[str, Any] = dict(
    PULL_4980_OPEN,
    state="closed",
    merged_at="2026-08-25T04:11:07Z",
    closed_at="2026-08-25T04:11:07Z",
    merge_commit_sha="4b1a0f5b4e0f0f6f2f3a4b5c6d7e8f90a1b2c3d4",
)

# Built by hand: no capture of GET /actions/workflows was kept. Only the paths matter - they
# are what `runs.resolve_workflow_ids` matches on.
WORKFLOWS_LISTING: list[dict[str, Any]] = [
    {"id": CI_PIPELINE_WORKFLOW_ID, "name": "MaxText Package Tests", "path": runs.CI_PIPELINE_PATH},
    {"id": TPU_IMAGES_WORKFLOW_ID, "name": "TPU Docker Images Pipeline", "path": runs.TPU_IMAGES_PATH},
    {"id": GPU_IMAGES_WORKFLOW_ID, "name": "GPU Docker Images Pipeline", "path": runs.GPU_IMAGES_PATH},
]


def read_fixture_bytes(name: str) -> bytes:
  """Returns the raw bytes of one saved fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The file contents, exactly as they came off the API.
  """
  return (FIXTURES / name).read_bytes()


def load_fixture(name: str) -> Any:
  """Reads one saved fixture as JSON.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The decoded payload, freshly parsed on every call so no test can mutate another's data.
  """
  return json.loads(read_fixture_bytes(name))


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


def scheduled_run() -> dict[str, Any]:
  """Returns the saved payload of run 33468578834, the scheduled run on main."""
  return load_fixture("run.json")


def scheduled_jobs() -> list[dict[str, Any]]:
  """Returns the 54 saved job payloads of run 33468578834."""
  return load_fixture("jobs.json")["jobs"]


def rerun_run() -> dict[str, Any]:
  """Returns the saved payload of run 32772626658, on its second attempt."""
  return load_fixture("rerun-32772626658-run.json")


def rerun_attempt_one() -> dict[str, Any]:
  """Returns the attempt-1 payload of run 32772626658.

  Derived, not saved: GitHub answers `/attempts/1` with the run as it stood on that attempt,
  and the only field `tick` reads off it that differs is `run_attempt`.

  Returns:
    The run payload with `run_attempt` set to 1.
  """
  payload = rerun_run()
  payload["run_attempt"] = 1
  return payload


def rerun_jobs(attempt: int) -> list[dict[str, Any]]:
  """Returns the 42 saved job payloads of one attempt of run 32772626658.

  Args:
    attempt: 1 or 2.

  Returns:
    The job payloads of that attempt.
  """
  return load_fixture(f"rerun-32772626658-attempt{attempt}-jobs.json")["jobs"]


def served_artifacts(expired: Iterable[str] = ()) -> list[dict[str, Any]]:
  """Returns the artifact payloads of run 33468578834 this scenario serves.

  Args:
    expired: Artifact names to hand back with `expired` set, to model a payload GitHub has
      already deleted.

  Returns:
    The saved artifact payloads whose JUnit files the repository kept.
  """
  gone = set(expired)
  chosen: list[dict[str, Any]] = []
  for payload in load_fixture("artifacts.json")["artifacts"]:
    if payload["name"] not in ARTIFACT_FILES:
      continue
    entry = copy.deepcopy(payload)
    if entry["name"] in gone:
      entry["expired"] = True
    chosen.append(entry)
  return chosen


def artifact_blobs(artifacts: Sequence[dict[str, Any]]) -> dict[str, bytes]:
  """Builds the zip each artifact downloads to.

  Args:
    artifacts: The artifact payloads being served.

  Returns:
    Download URL -> zip bytes. An expired artifact gets no blob, because the collector must
    never ask for one.
  """
  blobs: dict[str, bytes] = {}
  for payload in artifacts:
    if payload.get("expired"):
      continue
    members = {member: read_fixture_bytes(name) for member, name in ARTIFACT_FILES[payload["name"]].items()}
    blobs[payload["archive_download_url"]] = make_zip(members)
  return blobs


def running_run() -> dict[str, Any]:
  """Returns a run that GitHub has not finished yet.

  Derived from run.json, because every saved run is completed: one that is still going is
  exactly what the pending list exists for.

  Returns:
    The run payload, `in_progress` and created an hour before the pinned `now`.
  """
  payload = scheduled_run()
  payload["id"] = 33470000001
  payload["status"] = "in_progress"
  payload["conclusion"] = None
  payload["created_at"] = "2026-09-01T11:00:00Z"
  payload["run_started_at"] = "2026-09-01T11:00:00Z"
  return payload


def running_jobs(run_id: int, count: int = 6) -> list[dict[str, Any]]:
  """Returns job payloads re-pointed at another run.

  `rows.job_row` refuses a job whose `run_id` disagrees with the run it is passed with, so a
  derived run needs derived jobs.

  Args:
    run_id: The run the jobs should belong to.
    count: How many of the saved jobs to take.

  Returns:
    The job payloads, with their run id rewritten.
  """
  jobs = scheduled_jobs()[:count]
  for job in jobs:
    job["run_id"] = run_id
  return jobs


def snapshot(root: Path) -> dict[str, bytes]:
  """Returns every file under a directory, by relative path.

  Args:
    root: The directory to read.

  Returns:
    Relative path -> file bytes. Empty when the directory does not exist.
  """
  if not root.is_dir():
    return {}
  return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def read_ndjson(path: Path) -> list[dict[str, Any]]:
  """Reads one NDJSON file back, one object per line.

  Args:
    path: The file.

  Returns:
    The decoded rows, in the order they were written.
  """
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FakeClient:
  """Offline stand-in for `github.GitHubClient` that serves the saved fixtures.

  It answers the eight endpoints one tick reaches for, records every call, and raises
  whatever a test staged for a path so a failing endpoint can be modelled exactly. The
  runs listing is filtered by the `created` parameter the collector really sends, so a test
  can prove the window narrowed rather than only that fewer runs came back.
  """

  def __init__(
      self,
      run_payloads: Sequence[dict[str, Any]] = (),
      *,
      jobs: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
      attempts: dict[tuple[int, int], dict[str, Any]] | None = None,
      artifacts: dict[int, list[dict[str, Any]]] | None = None,
      blobs: dict[str, bytes] | None = None,
      pull_objects: dict[int, dict[str, Any]] | None = None,
      branch_pulls: dict[str, list[dict[str, Any]]] | None = None,
      budget: Sequence[int] = (),
      fail: dict[str, Exception] | None = None,
  ) -> None:
    """Stages what the fake will serve.

    Args:
      run_payloads: The runs that exist. They are listed under their own `workflow_id`.
      jobs: (run id, attempt) -> job payloads.
      attempts: (run id, attempt) -> the run payload for that attempt.
      artifacts: run id -> artifact payloads.
      blobs: Download URL -> zip bytes.
      pull_objects: Pull request number -> full payload for `GET /pulls/{n}`.
      branch_pulls: "owner:branch" -> the pull requests that head lookup answers with.
      budget: The `remaining` values `rate_limit` hands back, in order. The last one repeats.
      fail: Path -> exception to raise instead of answering it.
    """
    self.run_payloads = {int(payload["id"]): payload for payload in run_payloads}
    self.jobs = dict(jobs or {})
    self.attempts = dict(attempts or {})
    self.artifacts = dict(artifacts or {})
    self.blobs = dict(blobs or {})
    self.pull_objects = dict(pull_objects or {})
    self.branch_pulls = dict(branch_pulls or {})
    self.budget = list(budget)
    self.fail = dict(fail or {})
    self.get_json_calls: list[tuple[str, dict[str, Any]]] = []
    self.paginate_calls: list[tuple[str, str, dict[str, Any]]] = []
    self.get_bytes_calls: list[str] = []
    self.rate_limit_calls = 0
    self.closed = False

  # -------------------------------------------------------------- the client API

  def get_json(self, path: str, **params: Any) -> dict[str, Any]:
    """Answers one object endpoint and records the call.

    Args:
      path: The repository-relative path asked for.
      **params: Query parameters.

    Returns:
      The staged payload, deep-copied so a caller cannot mutate the fixture.

    Raises:
      github.GitHubError: The path is staged to fail, or nothing is staged for it and the
        real API would answer 404.
      AssertionError: The tick called an endpoint this fake does not model.
    """
    self.get_json_calls.append((path, dict(params)))
    self._maybe_fail(path)
    parts = path.split("/")
    if len(parts) == 3 and parts[:2] == ["actions", "runs"]:
      return self._run_payload(int(parts[2]), path)
    if len(parts) == 5 and parts[:2] == ["actions", "runs"] and parts[3] == "attempts":
      staged = self.attempts.get((int(parts[2]), int(parts[4])))
      if staged is None:
        raise github.GitHubError(f"{path}: no such attempt", status=404, url=path)
      return copy.deepcopy(staged)
    if len(parts) == 2 and parts[0] == runs.PULLS_ENDPOINT:
      staged = self.pull_objects.get(int(parts[1]))
      if staged is None:
        raise github.GitHubError(f"{path}: no such pull request", status=404, url=path)
      return copy.deepcopy(staged)
    raise AssertionError(f"the tick called an unexpected object endpoint: {path!r}")

  def paginate(self, path: str, key: str, **params: Any) -> list:
    """Answers one listing endpoint and records the call.

    Args:
      path: The repository-relative path asked for.
      key: The list key inside the response, recorded so a test can check it.
      **params: Query parameters.

    Returns:
      The staged list, deep-copied.

    Raises:
      github.GitHubError: The path is staged to fail, or the attempt has no jobs endpoint.
      AssertionError: The tick called an endpoint this fake does not model.
    """
    self.paginate_calls.append((path, key, dict(params)))
    self._maybe_fail(path)
    if path == runs.WORKFLOWS_ENDPOINT:
      return copy.deepcopy(WORKFLOWS_LISTING)
    if path == runs.PULLS_ENDPOINT:
      return copy.deepcopy(self.branch_pulls.get(str(params.get("head") or ""), []))
    parts = path.split("/")
    if len(parts) == 4 and parts[:2] == ["actions", "workflows"] and parts[3] == "runs":
      return self._runs_listing(int(parts[2]), str(params.get("created") or ""))
    if len(parts) == 4 and parts[:2] == ["actions", "runs"] and parts[3] == "artifacts":
      return copy.deepcopy(self.artifacts.get(int(parts[2]), []))
    if len(parts) == 6 and parts[:2] == ["actions", "runs"] and parts[3] == "attempts" and parts[5] == "jobs":
      staged = self.jobs.get((int(parts[2]), int(parts[4])))
      if staged is None:
        raise github.GitHubError(f"{path}: no jobs for this attempt", status=404, url=path)
      return copy.deepcopy(staged)
    raise AssertionError(f"the tick called an unexpected listing endpoint: {path!r}")

  def get_bytes(self, url: str) -> bytes:
    """Answers one artifact download and records the call.

    Args:
      url: The absolute download URL.

    Returns:
      The staged zip bytes.

    Raises:
      AssertionError: Nothing is staged for that URL, which means the tick downloaded an
        artifact it should not have.
    """
    self.get_bytes_calls.append(url)
    if url not in self.blobs:
      raise AssertionError(f"the tick downloaded an artifact this fake does not serve: {url!r}")
    return self.blobs[url]

  def rate_limit(self) -> dict[str, int]:
    """Returns the next scripted rate-limit reading.

    Returns:
      The `{"limit", "remaining", "reset"}` shape the real client returns. With nothing
      scripted the budget is comfortable.
    """
    self.rate_limit_calls += 1
    remaining = self.budget.pop(0) if len(self.budget) > 1 else (self.budget[0] if self.budget else 900)
    return {"limit": 1000, "remaining": remaining, "reset": 0}

  def wait_for_rate_limit(self, need: int = 50) -> None:
    """Does nothing: a test never waits for a budget it wrote itself."""
    del need

  def close(self) -> None:
    """Records that the tick closed its client."""
    self.closed = True

  # ------------------------------------------------------------------- what it saw

  def created_windows(self, workflow_id: int = CI_PIPELINE_WORKFLOW_ID) -> list[str]:
    """Returns the `created` filters one workflow's runs endpoint was asked for, in order.

    Args:
      workflow_id: The workflow whose listing calls to report.

    Returns:
      One string per call, e.g. "2026-08-11T00:00:00Z..2026-08-18T00:00:00Z".
    """
    path = f"actions/workflows/{workflow_id}/runs"
    return [str(params.get("created") or "") for called, _, params in self.paginate_calls if called == path]

  def paths(self) -> list[str]:
    """Returns every path the tick asked for, listings and objects together, in order."""
    return [path for path, _, _ in self.paginate_calls] + [path for path, _ in self.get_json_calls]

  # ---------------------------------------------------------------------- internals

  def _maybe_fail(self, path: str) -> None:
    """Raises whatever a test staged for this path.

    Args:
      path: The path being answered.

    Raises:
      Exception: The staged failure.
    """
    staged = self.fail.get(path)
    if staged is not None:
      raise staged

  def _run_payload(self, run_id: int, path: str) -> dict[str, Any]:
    """Returns one run, or the 404 the API answers for a run that is gone.

    Args:
      run_id: The run asked for.
      path: The path, for the error message.

    Returns:
      The run payload.

    Raises:
      github.GitHubError: No such run.
    """
    payload = self.run_payloads.get(run_id)
    if payload is None:
      raise github.GitHubError(f"{path}: no such run", status=404, url=path)
    return copy.deepcopy(payload)

  def _runs_listing(self, workflow_id: int, created: str) -> list[dict[str, Any]]:
    """Filters the staged runs the way the API's `created` parameter does.

    Args:
      workflow_id: The workflow being listed.
      created: The `created` filter, either ">=T" or "T..T".

    Returns:
      The matching runs, newest first, as GitHub returns them.
    """
    since, _, until = created.partition("..")
    since = since.lstrip(">=")
    found = [
        copy.deepcopy(payload)
        for payload in self.run_payloads.values()
        if int(payload.get("workflow_id") or 0) == workflow_id
        and (not since or str(payload.get("created_at") or "") >= since)
        and (not until or str(payload.get("created_at") or "") <= until)
    ]
    found.sort(key=lambda payload: str(payload.get("created_at") or ""), reverse=True)
    return found


class TickTestCase(unittest.TestCase):
  """Base class: blocks the network, pins the clock, and runs ticks into a temp store."""

  def setUp(self) -> None:
    """Installs the offline seams and undoes them after each test."""
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

    self.warnings: list[str] = []
    for module in (tick, runs, github, store_module):
      warn_patch = mock.patch.object(module, "_warn", self.warnings.append)
      warn_patch.start()
      self.addCleanup(warn_patch.stop)

    sleep_patch = mock.patch.object(github, "_sleep", lambda seconds: None)
    sleep_patch.start()
    self.addCleanup(sleep_patch.stop)

    runs.clear_caches()
    self.addCleanup(runs.clear_caches)

    self.root = Path(tempfile.mkdtemp(prefix="ci-metrics-tick-test-"))
    self.addCleanup(shutil.rmtree, self.root, True)
    self.out = self.root / "store"

  # ------------------------------------------------------------------ running a tick

  def run_tick(
      self,
      client: FakeClient,
      argv: Sequence[str] | None = None,
      now: datetime = NOW,
      out: Path | None = None,
  ) -> tuple[int, str, str]:
    """Runs one whole tick against a fake client, offline, with the clock pinned.

    Args:
      client: The fake to serve it.
      argv: Arguments after `--out`, or None for none. Pass a full argument list of your own
        by including "--out" in it.
      now: The moment the tick runs at.
      out: The store directory, defaulting to this test's own.

    Returns:
      (exit code, what it printed on stdout, what it printed on stderr).
    """
    target = self.out if out is None else out
    arguments = list(argv or [])
    if "--out" not in arguments:
      arguments = ["--out", str(target), *arguments]
    return self.run_main(client, arguments, now=now)

  def run_main(self, client: FakeClient | None, argv: Sequence[str], now: datetime = NOW) -> tuple[int, str, str]:
    """Runs `tick.main` with the clock pinned and, when given, a fake client in place.

    Args:
      client: The fake to serve it, or None to let `main` build its real client (which the
        socket block then stops the moment it is used).
      argv: The full argument list.
      now: The moment the tick runs at.

    Returns:
      (exit code, stdout, stderr).
    """
    stamp = tick.iso_utc(now)
    out_stream, err_stream = io.StringIO(), io.StringIO()
    patches: list[Any] = [
        mock.patch.object(tick, "utc_now", lambda: now),
        mock.patch.object(rows, "utc_now_iso", lambda: stamp),
    ]
    if client is not None:
      patches.append(mock.patch.object(github, "GitHubClient", self._factory(client)))
    with contextlib.ExitStack() as stack:
      for patch in patches:
        stack.enter_context(patch)
      stack.enter_context(contextlib.redirect_stdout(out_stream))
      stack.enter_context(contextlib.redirect_stderr(err_stream))
      code = tick.main(list(argv))
    return code, out_stream.getvalue(), err_stream.getvalue()

  def _factory(self, client: FakeClient) -> Callable[..., FakeClient]:
    """Returns a `GitHubClient` replacement that hands back one fake and checks the repo.

    Args:
      client: The fake to hand back.

    Returns:
      A callable with `GitHubClient`'s constructor signature.
    """

    def build(owner: str, repo: str, token: str | None = None, session: Any = None) -> FakeClient:
      """Records the repository the tick asked for and returns the fake.

      Args:
        owner: Repository owner.
        repo: Repository name.
        token: Ignored.
        session: Ignored.

      Returns:
        The fake client.
      """
      del token, session
      self.assertEqual((owner, repo), (OWNER, REPO))
      return client

    return build

  # ------------------------------------------------------------------ the scenario

  def full_client(self, **overrides: Any) -> FakeClient:
    """Builds the two-run scenario the module docstring describes.

    Args:
      **overrides: Constructor arguments to replace.

    Returns:
      The fake client.
    """
    artifacts = overrides.pop("artifacts_for_scheduled", served_artifacts())
    staged: dict[str, Any] = {
        "run_payloads": [scheduled_run(), rerun_run()],
        "jobs": {
            (SCHEDULED_RUN_ID, 1): scheduled_jobs(),
            (RERUN_RUN_ID, 1): rerun_jobs(1),
            (RERUN_RUN_ID, 2): rerun_jobs(2),
        },
        "attempts": {(RERUN_RUN_ID, 1): rerun_attempt_one()},
        "artifacts": {SCHEDULED_RUN_ID: artifacts, RERUN_RUN_ID: []},
        "blobs": artifact_blobs(artifacts),
        "pull_objects": {RERUN_PULL_NUMBER: copy.deepcopy(PULL_4980_MERGED)},
    }
    staged.update(overrides)
    return FakeClient(**staged)

  # --------------------------------------------------------------------- assertions

  def data_path(self, kind: str, month: str) -> Path:
    """Returns the NDJSON file one kind's rows of one month live in.

    Args:
      kind: One of the `rows.KIND_*` constants.
      month: "YYYY-MM".

    Returns:
      The path inside this test's store.
    """
    return self.out / "data" / f"{kind}-{month}.ndjson"

  def stored_rows(self, kind: str, month: str) -> list[dict[str, Any]]:
    """Reads one kind's stored rows for one month.

    Args:
      kind: One of the `rows.KIND_*` constants.
      month: "YYYY-MM".

    Returns:
      The rows, in the order they were written. Empty when the file does not exist.
    """
    path = self.data_path(kind, month)
    return read_ndjson(path) if path.exists() else []

  def state(self) -> dict[str, Any]:
    """Returns `data/state.json` as it stands on disk."""
    return json.loads((self.out / "data" / "state.json").read_text(encoding="utf-8"))

  def meta(self) -> dict[str, Any]:
    """Returns `views/meta.json` as it stands on disk."""
    return json.loads((self.out / "views" / "meta.json").read_text(encoding="utf-8"))

  def summary(self, printed: str) -> str:
    """Returns the one line a log greps, which is always the last thing a tick prints.

    Args:
      printed: Everything the tick wrote to stdout.

    Returns:
      The summary line.
    """
    lines = [line for line in printed.splitlines() if line.strip()]
    self.assertTrue(lines, "the tick printed nothing")
    return lines[-1]

  def assert_no_problems(self, printed: str) -> None:
    """Asserts the tick reported no lost data.

    Args:
      printed: Everything the tick wrote to stdout.
    """
    self.assertNotIn("Problem", printed)
    self.assertNotIn("lost data", self.summary(printed))


class FullTickTest(TickTestCase):
  """One whole tick, from an empty directory to rows and views on disk."""

  def test_a_full_tick_writes_rows_and_views(self) -> None:
    """Two runs go in, and every file the storage rules name comes out."""
    client = self.full_client()
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assert_no_problems(printed)
    self.assertTrue(client.closed, "the tick did not close its client")

    for month, count in EXPECTED_RUN_ROWS.items():
      self.assertEqual(len(self.stored_rows(rows.KIND_RUN, month)), count, month)
    for month, count in EXPECTED_JOB_ROWS.items():
      self.assertEqual(len(self.stored_rows(rows.KIND_JOB, month)), count, month)
    for month, count in EXPECTED_SUITE_ROWS.items():
      self.assertEqual(len(self.stored_rows(rows.KIND_SUITE, month)), count, month)
    for month, count in EXPECTED_TEST_ROWS.items():
      self.assertEqual(len(self.stored_rows(rows.KIND_TEST, month)), count, month)
    for month, count in EXPECTED_RESCUE_ROWS.items():
      self.assertEqual(len(self.stored_rows(rows.KIND_RESCUE, month)), count, month)

    # A run's rows are filed by the run's creation month, never by the attempt's, so the
    # re-run that ran on 2026-08-25 is still in the 2026-08 file with its first attempt.
    self.assertEqual(self.stored_rows(rows.KIND_RUN, SCHEDULED_MONTH)[0]["run_id"], SCHEDULED_RUN_ID)
    self.assertEqual(
        sorted(row["attempt"] for row in self.stored_rows(rows.KIND_RUN, RERUN_MONTH)),
        [1, 2],
    )

    stored_state = self.state()
    self.assertEqual(
        stored_state["collected"],
        {str(RERUN_RUN_ID): [1, 2], str(SCHEDULED_RUN_ID): [1]},
    )
    self.assertEqual(stored_state["pending"], [])
    self.assertEqual(stored_state["watermark_run_id"], SCHEDULED_RUN_ID)

    for month in (RERUN_MONTH, SCHEDULED_MONTH):
      for group in views.VIEW_GROUPS:
        self.assertTrue((self.out / "views" / f"{group}-{month}.json").exists(), f"{group}-{month}")
    self.assertTrue((self.out / "views" / "meta.json").exists())
    # Only a MERGED pull request reaches the runs view and gets a file of its own.
    self.assertTrue((self.out / "views" / "pr" / f"{RERUN_PULL_NUMBER}.json").exists())
    self.assertEqual(
        self.meta()["groups"]["runs"]["rows"][RERUN_MONTH],
        {"runs": 1, "jobs": 21},
    )
    self.assertEqual(list(self.meta()["pull_requests"]), [str(RERUN_PULL_NUMBER)])

    self.assertIn("ci-metrics backfill: ok", self.summary(printed))
    self.assertIn("2 run(s)", self.summary(printed))

  def test_stored_rows_still_match_the_row_schema(self) -> None:
    """Everything written reads back through `rows.from_json`, which checks the field set."""
    self.run_tick(self.full_client(), ["--since", "2026-08-20"])
    store = store_module.Store(self.out)

    run_rows = store.read_rows(rows.KIND_RUN)
    self.assertEqual(len(run_rows), sum(EXPECTED_RUN_ROWS.values()))
    merged = [row for row in run_rows if row.is_merged_pr]
    self.assertEqual({row.pr_number for row in merged}, {RERUN_PULL_NUMBER})
    self.assertEqual({row.pr_merged_at for row in merged}, {PULL_4980_MERGED["merged_at"]})

    rescue_rows = store.read_rows(rows.KIND_RESCUE)
    self.assertEqual(len(rescue_rows), RERUN_RESCUE_ROWS)
    self.assertEqual(sum(1 for row in rescue_rows if row.rescued), RERUN_RESCUED_ROWS)

    # One timestamp for the whole tick, which is what "readers take the last row per key"
    # depends on.
    self.assertEqual({row.collected_at for row in run_rows}, {NOW_ISO})

  def test_a_suite_with_no_test_file_is_stored_as_null_and_a_reason(self) -> None:
    """Missing is null, never zero: a suite with no file must not be drawable as a drop."""
    self.run_tick(self.full_client(), ["--since", "2026-08-20"])
    stored = {row["suite_id"]: row for row in self.stored_rows(rows.KIND_SUITE, SCHEDULED_MONTH)}

    for suite_id, (collected, skipped, executed) in EXPECTED_SUITE_TOTALS.items():
      row = stored[suite_id]
      self.assertEqual((row["collected"], row["skipped"], row["executed"]), (collected, skipped, executed), suite_id)
      self.assertIsNone(row["reason"], suite_id)

    for suite_id in EXPECTED_NO_FILE_SUITES:
      row = stored[suite_id]
      self.assertIsNone(row["collected"], suite_id)
      self.assertIsNone(row["executed"], suite_id)
      self.assertIsNone(row["junit_seconds"], suite_id)
      self.assertEqual(row["reason"], "no_file_published", suite_id)

    # The nested decoupled pass is filed under cpu-unit and its tests are already inside
    # cpu-unit's totals, so the two must never be added together.
    self.assertEqual(stored["decoupled"]["flavor"], "cpu-unit")
    self.assertEqual(stored["decoupled"]["nested_in"], "cpu-unit")

  def test_a_half_published_suite_is_marked_partial(self) -> None:
    """One worker's artifact expiring leaves a partial total, and it says so."""
    artifacts = served_artifacts(expired=["test-results-cpu-unit-3-33468578834"])
    client = self.full_client(artifacts_for_scheduled=artifacts)
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    stored = {row["suite_id"]: row for row in self.stored_rows(rows.KIND_SUITE, SCHEDULED_MONTH)}
    cpu_unit = stored["cpu-unit"]
    self.assertTrue(cpu_unit["is_partial"])
    self.assertEqual(cpu_unit["missing_workers"], [{"worker": 3, "reason": "artifact_expired"}])
    self.assertEqual(cpu_unit["published_workers"], [1])
    # Worker 1 alone. Half of 1,474 is not a drop, which is the whole point of the flag.
    self.assertEqual(cpu_unit["collected"], 737)
    self.assertIn("Partial suites", printed)
    expired_url = next(payload["archive_download_url"] for payload in artifacts if payload["expired"])
    self.assertNotIn(expired_url, client.get_bytes_calls, "an expired artifact must never be downloaded")
    self.assertEqual(len(client.get_bytes_calls), len(artifacts) - 1)

  def test_the_days_first_scheduled_main_run_keeps_every_test_row(self) -> None:
    """The daily snapshot is what gives the per-test history a daily resolution."""
    _, printed, _ = self.run_tick(self.full_client(), ["--since", "2026-08-20"])
    self.assertIn("1 full daily snapshot(s)", printed)
    kept = len(self.stored_rows(rows.KIND_TEST, SCHEDULED_MONTH))
    self.assertEqual(kept, EXPECTED_TEST_ROWS[SCHEDULED_MONTH])
    self.assertIn(f"{kept:,} of {kept:,} read", printed)


class SecondTickTest(TickTestCase):
  """What a tick does when it has already been run."""

  def test_running_the_same_tick_twice_writes_nothing_the_second_time(self) -> None:
    """The second tick finds nothing new, changes no byte, and still exits zero."""
    first_code, _, _ = self.run_tick(self.full_client(), ["--since", "2026-08-20"])
    self.assertEqual(first_code, tick.EXIT_OK)
    before = snapshot(self.out)
    self.assertTrue(before)

    client = self.full_client()
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertEqual(snapshot(self.out), before, "the second tick rewrote the store")
    self.assertIn("nothing new", self.summary(printed))
    self.assertIn("0 run(s)", self.summary(printed))
    self.assertIn("2 (2 already stored, 0 collected, 0 failed)", printed)
    # A run already stored costs the listing that found it and not one request more.
    self.assertEqual([path for path in client.paths() if "attempts" in path], [])
    self.assertEqual(client.get_bytes_calls, [])

  def test_a_temporary_file_a_killed_tick_left_behind_is_swept(self) -> None:
    """An atomic write that never got to rename leaves a full copy of a month behind."""
    self.run_tick(self.full_client(), ["--since", "2026-08-20"])
    leftover = self.out / "data" / f"{store_module.TEMP_PREFIX}dead.ndjson"
    leftover.write_text("{}\n", encoding="utf-8")
    old = leftover.stat().st_mtime - (store_module.TEMP_MAX_AGE_HOURS + 1) * 3600
    os.utime(leftover, (old, old))

    code, _, _ = self.run_tick(self.full_client(), ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertFalse(leftover.exists())
    self.assertTrue(any("killed mid-write" in warning for warning in self.warnings), self.warnings)

  def test_a_deleted_view_file_is_written_back(self) -> None:
    """Views are disposable: the next tick notices one is gone and rebuilds its month."""
    self.run_tick(self.full_client(), ["--since", "2026-08-20"])
    victim = self.out / "views" / f"suites-{SCHEDULED_MONTH}.json"
    wanted = victim.read_bytes()
    victim.unlink()

    code, printed, _ = self.run_tick(self.full_client(), ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertEqual(victim.read_bytes(), wanted)
    self.assertNotIn("nothing new", self.summary(printed))


class WatermarkTest(TickTestCase):
  """The window a tick asks the API for."""

  def _store_the_rerun_only(self) -> None:
    """Runs a first tick whose only run is the 2026-08-24 pull-request run."""
    client = FakeClient(
        run_payloads=[rerun_run()],
        jobs={(RERUN_RUN_ID, 1): rerun_jobs(1), (RERUN_RUN_ID, 2): rerun_jobs(2)},
        attempts={(RERUN_RUN_ID, 1): rerun_attempt_one()},
        artifacts={RERUN_RUN_ID: []},
        pull_objects={RERUN_PULL_NUMBER: copy.deepcopy(PULL_4980_MERGED)},
    )
    code, _, _ = self.run_tick(client, ["--since", "2026-08-20"])
    self.assertEqual(code, tick.EXIT_OK)

  def test_the_watermark_advances_and_the_next_tick_asks_only_after_it(self) -> None:
    """A tick never re-sweeps: its window starts where the stored watermark left off."""
    self._store_the_rerun_only()
    stored_state = self.state()
    self.assertEqual(stored_state["watermark_run_id"], RERUN_RUN_ID)
    self.assertEqual(stored_state["watermark_created_at"], "2026-08-24T20:13:09Z")

    client = FakeClient(
        run_payloads=[rerun_run()],
        jobs={(RERUN_RUN_ID, 1): rerun_jobs(1), (RERUN_RUN_ID, 2): rerun_jobs(2)},
        attempts={(RERUN_RUN_ID, 1): rerun_attempt_one()},
        artifacts={RERUN_RUN_ID: []},
        pull_objects={RERUN_PULL_NUMBER: copy.deepcopy(PULL_4980_MERGED)},
    )
    code, printed, _ = self.run_tick(client)

    self.assertEqual(code, tick.EXIT_OK)
    windows = client.created_windows()
    self.assertTrue(windows, "the second tick listed nothing")
    # Nothing older than the watermark was asked for, in any slice.
    self.assertEqual(min(window.split("..")[0] for window in windows), "2026-08-24T20:13:09Z")
    self.assertEqual(windows[0], "2026-08-24T20:13:09Z..2026-08-31T20:13:09Z")
    self.assertEqual(windows[-1].split("..")[1], NOW_ISO)
    self.assertIn("Mode                       tick (2026-08-24T20:13:09Z", printed)
    self.assertIn("nothing new", self.summary(printed))
    self.assertEqual([path for path in client.paths() if "attempts" in path], [])

  def test_a_re_listed_window_still_covers_the_last_two_days(self) -> None:
    """The watermark alone would never see a re-run, so a tick re-lists the recent days."""
    client = self.full_client()
    self.run_tick(client, ["--since", "2026-08-20"])
    # The newest run is hours old, so the re-check window, not the watermark, is the start.
    second = self.full_client()
    self.run_tick(second)
    self.assertEqual(
        second.created_windows()[0],
        f"{tick.iso_utc(NOW - timedelta(days=tick.RECHECK_DAYS))}..{NOW_ISO}",
    )


class BackfillTest(TickTestCase):
  """The first tick of an empty store, which walks history a week at a time."""

  def test_backfill_walks_the_window_in_weekly_chunks(self) -> None:
    """A wider listing would silently lose runs past GitHub's 1000-result cap."""
    client = self.full_client()
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-11"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertIn("ci-metrics backfill:", self.summary(printed))
    self.assertEqual(
        client.created_windows(),
        [
            "2026-08-11T00:00:00Z..2026-08-18T00:00:00Z",
            "2026-08-18T00:00:00Z..2026-08-25T00:00:00Z",
            "2026-08-25T00:00:00Z..2026-09-01T00:00:00Z",
            f"2026-09-01T00:00:00Z..{NOW_ISO}",
        ],
    )
    # Every allowlisted workflow is listed through its own endpoint, never the repository's
    # whole runs listing.
    self.assertNotIn(runs.RUNS_ENDPOINT, client.paths())
    self.assertEqual(len(client.created_windows(TPU_IMAGES_WORKFLOW_ID)), 4)

  def test_backfill_stops_cleanly_when_the_rate_budget_runs_low(self) -> None:
    """What was collected is kept, the watermark reflects it, and the exit code stays zero."""
    client = self.full_client(budget=[900, 40])
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-11"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assert_no_problems(printed)
    self.assertIn("Stopped early", printed)
    self.assertIn(f"floor is {tick.RATE_LIMIT_FLOOR}", printed)
    self.assertIn("stopped early", self.summary(printed))

    # The budget was checked before the second and third weeks, and the third was never asked
    # for, so the run created on 2026-09-01 is still out there.
    self.assertEqual(
        client.created_windows(),
        [
            "2026-08-11T00:00:00Z..2026-08-18T00:00:00Z",
            "2026-08-18T00:00:00Z..2026-08-25T00:00:00Z",
        ],
    )
    self.assertEqual(client.rate_limit_calls, 2)
    stored_state = self.state()
    self.assertEqual(list(stored_state["collected"]), [str(RERUN_RUN_ID)])
    self.assertEqual(stored_state["watermark_created_at"], "2026-08-24T20:13:09Z")
    self.assertEqual(self.stored_rows(rows.KIND_RUN, SCHEDULED_MONTH), [])

  def test_max_runs_stops_the_tick_and_says_so(self) -> None:
    """`--max-runs` leaves the rest for the next tick rather than dropping it."""
    client = self.full_client()
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-11", "--max-runs", "1"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertIn("--max-runs 1 reached", printed)
    self.assertEqual(list(self.state()["collected"]), [str(RERUN_RUN_ID)])


class DryRunTest(TickTestCase):
  """`--dry-run` fetches and computes everything and writes nothing."""

  def test_dry_run_writes_nothing(self) -> None:
    """The output directory is still empty afterwards, down to the last file."""
    self.out.mkdir(parents=True)
    client = self.full_client()
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-20", "--dry-run"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertEqual(list(self.out.iterdir()), [], "a dry run wrote into the store")
    self.assertEqual(snapshot(self.out), {})
    # It really did the work: the rows were built and counted, and the views were rendered
    # into a throwaway directory that no longer exists.
    self.assertIn("[dry run: nothing written]", printed)
    self.assertIn("dry run", self.summary(printed))
    self.assertIn(f"{EXPECTED_TEST_ROWS[SCHEDULED_MONTH]:,} of", printed)
    self.assertTrue(client.get_bytes_calls, "a dry run should still fetch")

  def test_a_dry_run_leaves_the_next_real_tick_with_everything_to_do(self) -> None:
    """Nothing was indexed, so the tick that follows collects the same runs in full."""
    self.out.mkdir(parents=True)
    self.run_tick(self.full_client(), ["--since", "2026-08-20", "--dry-run"])
    code, printed, _ = self.run_tick(self.full_client(), ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertIn("2 (0 already stored, 2 collected, 0 failed)", printed)
    self.assertEqual(len(self.stored_rows(rows.KIND_TEST, SCHEDULED_MONTH)), EXPECTED_TEST_ROWS[SCHEDULED_MONTH])


class UsageTest(TickTestCase):
  """A command line that could never work is refused, and nothing is written."""

  def test_out_is_required(self) -> None:
    """There is no default store, because a store is a thing a person chooses."""
    self.out.mkdir(parents=True)
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
      with self.assertRaises(SystemExit) as caught:
        tick.main(["--since", "2026-08-20"])

    self.assertEqual(caught.exception.code, tick.EXIT_USAGE)
    self.assertIn("--out", err.getvalue())
    self.assertIn("required", err.getvalue())
    self.assertEqual(list(self.out.iterdir()), [])

  def test_an_empty_out_is_refused_rather_than_meaning_here(self) -> None:
    """`Path("")` is the working directory, so an empty --out has to be caught by name.

    The store's own guard cannot see this: by the time the path reaches it, the empty string
    has already become `PosixPath(".")`, and it is only refused when the working directory
    happens to hold a `.git`.
    """
    for empty in ("", "   "):
      with self.subTest(out=repr(empty)):
        code, _, err = self.run_main(None, ["--out", empty, "--since", "2026-08-20"])

        self.assertEqual(code, tick.EXIT_USAGE)
        self.assertIn("--out needs a directory", err)

  def test_a_git_checkout_is_refused_as_a_store(self) -> None:
    """Writing a data store into somebody's checkout by accident has to be impossible."""
    checkout = self.root / "checkout"
    (checkout / ".git").mkdir(parents=True)
    code, _, err = self.run_main(None, ["--out", str(checkout)])

    self.assertEqual(code, tick.EXIT_USAGE)
    self.assertIn("is a git checkout, not a data store", err)
    self.assertEqual([path.name for path in checkout.iterdir()], [".git"])

  def test_impossible_arguments_are_refused_before_anything_is_fetched(self) -> None:
    """Each of these would fail the same way if it were retried, so the exit code is 2."""
    cases = {
        "a window that ends before it starts": ["--since", "2026-09-01", "--until", "2026-08-01"],
        "a date that is not a date": ["--since", "last tuesday"],
        "a backfill of no days": ["--backfill-days", "0"],
        "a run cap below one": ["--max-runs", "0"],
        "a repository that is not owner/name": ["--repo", "maxtext"],
    }
    for what, argument in cases.items():
      with self.subTest(what):
        self.out.mkdir(parents=True, exist_ok=True)
        code, _, err = self.run_main(None, ["--out", str(self.out), *argument])
        self.assertEqual(code, tick.EXIT_USAGE, err)
        self.assertTrue(err.startswith("error: "), err)
        self.assertEqual(snapshot(self.out), {})


class ExitCodeTest(TickTestCase):
  """Zero when nothing was lost, one when something was."""

  def test_a_failed_fetch_exits_one_and_keeps_the_run_for_next_time(self) -> None:
    """A run that could not be read goes on the in-flight list, not into the store."""
    client = self.full_client(
        fail={
            f"actions/runs/{SCHEDULED_RUN_ID}/attempts/1/jobs": github.GitHubError(
                "500 Server Error", status=500, url="actions/runs"
            )
        }
    )
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-27"])

    self.assertEqual(code, tick.EXIT_DATA_LOST)
    self.assertIn("could not be collected", printed)
    self.assertIn("it will be tried again next tick", printed)
    self.assertIn("lost data", self.summary(printed))
    self.assertEqual(self.stored_rows(rows.KIND_RUN, SCHEDULED_MONTH), [])
    pending = self.state()["pending"]
    self.assertEqual([entry["run_id"] for entry in pending], [SCHEDULED_RUN_ID])

  def test_a_failed_listing_exits_one_and_collects_nothing(self) -> None:
    """A window that could not be listed is data at risk, not an empty window."""
    client = self.full_client(
        fail={CI_PIPELINE_RUNS_PATH: github.GitHubError("502 Bad Gateway", status=502, url="actions/workflows")}
    )
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-27"])

    self.assertEqual(code, tick.EXIT_DATA_LOST)
    self.assertIn("could not be listed", printed)
    self.assertEqual(self.state()["collected"], {})

  def test_nothing_new_is_still_a_success(self) -> None:
    """An empty window is a normal answer; the commit step decides what to push."""
    client = FakeClient(run_payloads=[])
    code, printed, _ = self.run_tick(client, ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertIn("nothing new", self.summary(printed))
    self.assertIn("0 run(s)", self.summary(printed))


class PendingAttemptTest(TickTestCase):
  """An attempt GitHub has not finished yet."""

  def setUp(self) -> None:
    """Stages a run that is still going, plus the jobs it has so far."""
    super().setUp()
    self.running = running_run()
    self.running_id = int(self.running["id"])
    self.running_jobs = running_jobs(self.running_id)

  def _client(self) -> FakeClient:
    """Returns a fake serving only the unfinished run."""
    return FakeClient(
        run_payloads=[self.running],
        jobs={(self.running_id, 1): copy.deepcopy(self.running_jobs)},
        artifacts={self.running_id: []},
    )

  def test_an_attempt_still_running_is_pending_not_written(self) -> None:
    """Only a completed attempt may be written, so this one is remembered instead."""
    code, printed, _ = self.run_tick(self._client(), ["--since", "2026-08-25"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertEqual(self.stored_rows(rows.KIND_RUN, SCHEDULED_MONTH), [])
    self.assertEqual(sorted(path.name for path in (self.out / "data").iterdir()), ["state.json"])

    stored_state = self.state()
    self.assertEqual(stored_state["collected"], {})
    self.assertEqual(stored_state["incomplete"], {})
    self.assertEqual(
        stored_state["pending"],
        [
            {
                "run_id": self.running_id,
                "attempt": 1,
                "first_seen_at": NOW_ISO,
                "created_at": "2026-09-01T11:00:00Z",
                "status": "in_progress",
            }
        ],
    )
    self.assertIn("still running", printed)
    self.assertEqual(self.meta()["uncollected_runs"], 1)

  def test_a_pending_attempt_older_than_a_day_is_written_once(self) -> None:
    """A stuck run cannot block the store forever, and is written with the status it has."""
    self.run_tick(self._client(), ["--since", "2026-08-25"])
    later = NOW + timedelta(hours=store_module.PENDING_MAX_AGE_HOURS + 6)

    code, printed, _ = self.run_tick(self._client(), now=later)

    self.assertEqual(code, tick.EXIT_OK)
    self.assertIn("1 stored unfinished", printed)
    written = self.stored_rows(rows.KIND_RUN, SCHEDULED_MONTH)
    self.assertEqual(len(written), 1)
    self.assertEqual(written[0]["run_id"], self.running_id)
    # There is no `incomplete` field on a run row - adding one would break `rows.from_json`
    # for every row already stored - so the row carries the status it was stopped at, and the
    # index is what records that it is not final.
    self.assertEqual(written[0]["status"], "in_progress")
    self.assertIsNone(written[0]["conclusion"])
    self.assertEqual(len(self.stored_rows(rows.KIND_JOB, SCHEDULED_MONTH)), len(self.running_jobs))

    stored_state = self.state()
    self.assertEqual(stored_state["incomplete"], {str(self.running_id): [1]})
    self.assertEqual(stored_state["pending"], [])
    self.assertTrue(any("no longer waiting for it" in warning for warning in self.warnings), self.warnings)

    # Once, and only once: a third tick finds the attempt already accounted for.
    before = snapshot(self.out)
    code, printed, _ = self.run_tick(self._client(), now=later)
    self.assertEqual(code, tick.EXIT_OK)
    self.assertEqual(snapshot(self.out), before)
    self.assertIn("nothing new", self.summary(printed))

  def test_an_attempt_pended_by_this_tick_is_not_re_read_by_it(self) -> None:
    """The in-flight list is what a PAST tick left, not what this one is still adding.

    `state.pending` is the same object the tick mutates all the way through, so reading it at
    the end used to hand back the attempt that was pended seconds earlier - which was then
    fetched a second time, counted twice in the report and charged twice to the API budget.
    """
    client = self._client()

    code, printed, _ = self.run_tick(client, ["--since", "2026-08-25"])

    self.assertEqual(code, tick.EXIT_OK)
    reads = [path for path, _ in client.get_json_calls if path == f"actions/runs/{self.running_id}"]
    self.assertEqual(len(reads), 1, f"the run was read {len(reads)} times: {client.get_json_calls}")
    self.assertIn("(1 still running", printed)

  def test_a_run_that_failed_this_tick_is_left_for_the_next_one(self) -> None:
    """A failing endpoint must be hit once per tick, not twice, and still exit 1.

    A run put on the in-flight list because it FAILED is on the same list as one that is
    still going, so the same defect made the tick retry it immediately - hitting an endpoint
    that had just failed a second time and printing the problem twice.
    """
    failing = f"actions/runs/{SCHEDULED_RUN_ID}/attempts/1/jobs"
    client = self.full_client(fail={failing: github.GitHubError("500 Server Error", status=500, url="actions/runs")})

    code, printed, _ = self.run_tick(client, ["--since", "2026-08-27"])

    self.assertEqual(code, tick.EXIT_DATA_LOST)
    hits = [path for path, _, _ in client.paginate_calls if path == failing]
    self.assertEqual(len(hits), 1, f"the failing endpoint was asked {len(hits)} times")
    self.assertEqual(printed.count(f"run {SCHEDULED_RUN_ID} could not be collected"), 1)
    self.assertEqual([entry["run_id"] for entry in self.state()["pending"]], [SCHEDULED_RUN_ID])


class RescueUpgradeTest(TickTestCase):
  """A job that failed on one tick and was re-run before the next one."""

  def test_a_re_run_seen_later_replaces_the_never_re_run_row(self) -> None:
    """End to end: the flaky view has to say "rescued", not "failed and never re-run".

    The first tick sees attempt 1 only. The second sees the re-run. Both write a rescue row
    with the same key - the key names the failure, not the outcome - so a store that skipped
    the second one would freeze the answer at the first, wrong one.
    """
    one_attempt = dict(rerun_run(), run_attempt=1)
    first = FakeClient(
        run_payloads=[one_attempt],
        jobs={(RERUN_RUN_ID, 1): rerun_jobs(1)},
        artifacts={RERUN_RUN_ID: []},
        pull_objects={RERUN_PULL_NUMBER: copy.deepcopy(PULL_4980_MERGED)},
    )
    code, _, _ = self.run_tick(first, ["--since", "2026-08-20"])
    self.assertEqual(code, tick.EXIT_OK)
    before = self.stored_rows(rows.KIND_RESCUE, RERUN_MONTH)
    self.assertTrue(before)
    self.assertTrue(all(not row["rescued"] for row in before), "nothing could have been rescued yet")

    second = self.full_client()
    code, printed, _ = self.run_tick(second, ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    lines = self.stored_rows(rows.KIND_RESCUE, RERUN_MONTH)
    self.assertGreater(len(lines), len(before), "the correction is a second line, not an edit")
    latest: dict[str, dict[str, Any]] = {}
    for row in lines:
      latest[row["job_name"]] = row
    rescued = [row for row in latest.values() if row["rescued"]]
    self.assertTrue(rescued, "the re-run has to be visible")
    for row in rescued:
      self.assertEqual(row["rescued_attempt"], 2)
    self.assert_no_problems(printed)

  def test_a_settled_rescue_is_not_rewritten_by_the_next_tick(self) -> None:
    """Content dedup, not "always write": a repeat still has to change no byte."""
    self.run_tick(self.full_client(), ["--since", "2026-08-20"])
    before = snapshot(self.out)

    code, printed, _ = self.run_tick(self.full_client(), ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertEqual(snapshot(self.out), before)
    self.assertIn("nothing new", self.summary(printed))


class MergeChasingTest(TickTestCase):
  """A run read while its pull request was still open."""

  def _client(self, pull: dict[str, Any]) -> FakeClient:
    """Returns a fake whose pull request #4980 is in the state given.

    Args:
      pull: The pull request payload to serve.

    Returns:
      The fake client.
    """
    return FakeClient(
        run_payloads=[rerun_run()],
        jobs={(RERUN_RUN_ID, 1): rerun_jobs(1), (RERUN_RUN_ID, 2): rerun_jobs(2)},
        attempts={(RERUN_RUN_ID, 1): rerun_attempt_one()},
        artifacts={RERUN_RUN_ID: []},
        pull_objects={RERUN_PULL_NUMBER: copy.deepcopy(pull)},
    )

  def test_a_merge_learned_later_is_stored_as_a_correction(self) -> None:
    """Without this the runs view would stay empty: a run is read before its PR merges."""
    code, printed, _ = self.run_tick(self._client(PULL_4980_OPEN), ["--since", "2026-08-20"])
    self.assertEqual(code, tick.EXIT_OK)
    self.assertIn("1 still open when their run was read", printed)
    self.assertFalse((self.out / "views" / "pr" / f"{RERUN_PULL_NUMBER}.json").exists())
    first_pass = self.stored_rows(rows.KIND_RUN, RERUN_MONTH)
    self.assertEqual(len(first_pass), RERUN_ATTEMPTS)
    self.assertEqual({row["pr_merged_at"] for row in first_pass}, {None})

    code, printed, _ = self.run_tick(self._client(PULL_4980_MERGED), ["--since", "2026-08-20"])

    self.assertEqual(code, tick.EXIT_OK)
    self.assertIn("1 merged since the last tick", printed)
    corrected = self.stored_rows(rows.KIND_RUN, RERUN_MONTH)
    # Append only: the original lines stay where they were and the correction is a new line.
    self.assertEqual(len(corrected), RERUN_ATTEMPTS + 1)
    self.assertEqual(corrected[-1]["pr_merged_at"], PULL_4980_MERGED["merged_at"])
    store = store_module.Store(self.out)
    winners = store.read(rows.KIND_RUN, [RERUN_MONTH])
    self.assertEqual(len(winners), RERUN_ATTEMPTS, "a correction must not add a key")
    self.assertTrue((self.out / "views" / "pr" / f"{RERUN_PULL_NUMBER}.json").exists())


class CountingSessionTest(TickTestCase):
  """The seam that tells an API request apart from an artifact download."""

  def test_only_api_requests_count_against_the_hourly_budget(self) -> None:
    """An artifact download redirects to a storage host and is free."""
    answer = requests.Response()
    answer.status_code = 200
    with mock.patch.object(requests.Session, "request", lambda *args, **kwargs: answer):
      session = tick.CountingSession()
      session.request("GET", f"{github.API_ROOT}/repos/{OWNER}/{REPO}/actions/runs")
      session.request("GET", f"{github.API_ROOT}/repos/{OWNER}/{REPO}/actions/artifacts/1/zip")
      session.request("GET", "https://productionresultssa0.blob.core.windows.net/actions-results/1?sig=x")

    self.assertEqual(session.api_requests, 2)
    self.assertEqual(session.download_requests, 1)


if __name__ == "__main__":
  unittest.main()
