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

"""Offline unit tests for `collector.runs`.

Nothing here touches the network. The base test case replaces `socket.socket`, so a test
that reached for GitHub would fail loudly instead of hanging or spending rate limit. Two
offline seams serve the payloads:

  * `StubClient` satisfies the `GitHubClientLike` protocol and hands back saved fixtures,
    recording every path and query parameter it was asked for. It is used wherever the
    question is what `runs.py` decides.
  * A real `github.GitHubClient` wired to a `FakeSession` of canned `requests.Response`
    objects is used wherever the question is how paging behaves, so `Link` headers, short
    pages and the `created=` parameter travel the same code the collector really runs.

Every expected number is a measured fact about the saved fixtures, not a round number
chosen to make a test pass:

  1. The two-page listing holds 3 + 3 runs of workflow 297948505 with no id in common; the
     branch-filtered listing holds 3 + 3 + 1, and that last short page is what stops paging.
  2. Branch `parity-checkpoint-lifecycle` has exactly two runs 4m33s apart - cancelled
     33462758754 then successful 33463047689 - so the older one is superseded and the newer
     one is not. Branch `darisoy-skip-mla-indexer-all-gather-cp-test` has two cancelled runs,
     and only the older of the two is superseded, because the newest run of a branch never is.
  3. `run.pull_requests` is empty on merged same-repo run 33406483779 (PR #5070) and on fork
     run 33157998260 (PR #5042), so the `GET /pulls?head={head owner}:{branch}` fallback is
     the ordinary path. Both are matched back to their run by head sha.
  4. `action_required` run 33465601432 answers its jobs endpoint with
     `{"total_count": 0, "jobs": []}`, which means "ran no jobs", not "the request failed".

The tests are plain `unittest`, so they need nothing but the standard library and
`requests`. pytest collects them too, because it understands `unittest.TestCase` and the
file is named the way the repository's pytest.ini expects.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/runs_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/runs_test.py
"""

from __future__ import annotations

import copy
import json
import socket
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import requests
from requests.structures import CaseInsensitiveDict

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import github
from collector import runs

FIXTURES = Path(__file__).resolve().parent / "fixtures"

OWNER = "AI-Hypercomputer"
REPO = "maxtext"
TOKEN = "ghp-test-token-never-real"
REPO_ROOT_URL = f"{github.API_ROOT}/repos/{OWNER}/{REPO}"

# The numeric id of .github/workflows/ci_pipeline.yml, read from every saved run payload.
CI_PIPELINE_WORKFLOW_ID = 297948505
# A workflow the dashboard does not read, used to prove the allowlist actually excludes one.
OTHER_WORKFLOW_ID = 12345678
OTHER_WORKFLOW_PATH = ".github/workflows/UnitTests.yml"

# runs-list-page1.json then runs-list-page2.json, in the order GitHub returned them.
PAGE_ONE_IDS = (33477529994, 33468578834, 33467756955)
PAGE_TWO_IDS = (33466797307, 33466364329, 33465790456)
# runs-list-short-final-page{1,2,3}.json: 3 + 3 + 1, the last page short enough to stop paging.
SHORT_FINAL_IDS = (
    (33204220590, 33003795267, 33003763519),
    (32995972111, 32981293288, 32582507668),
    (32579394193,),
)
# total_count of both listings, kept only to prove the fixtures were not re-cut by hand.
PAGED_TOTAL_COUNT = 6215
SHORT_FINAL_TOTAL_COUNT = 7

# Branch parity-checkpoint-lifecycle: a cancelled run and the run that replaced it.
SUPERSEDED_RUN_ID = 33462758754
SUCCESSOR_RUN_ID = 33463047689
# Branch darisoy-skip-mla-indexer-all-gather-cp-test: two cancelled runs, newest first.
NEWEST_CANCELLED_RUN_ID = 32999554339
OLDER_CANCELLED_RUN_ID = 32998181337

# Merged same-repo pull request #5070, whose run carries an EMPTY pull_requests array.
MERGED_RUN_ID = 33406483779
MERGED_PULL_NUMBER = 5070
MERGED_HEAD_SHA = "d3283134ce543485e622bd6f52f2de1c645472bd"
MERGED_HEAD_QUERY = "AI-Hypercomputer:aireen/script_grain_tfrecord"

# Fork pull request #5042. The head query uses the CONTRIBUTOR's login, not the base owner.
FORK_RUN_ID = 33157998260
FORK_PULL_NUMBER = 5042
FORK_HEAD_SHA = "ead5faf12ca7d10797fe5d9fa2d22eea8be9bf84"
FORK_HEAD_QUERY = "guowei-dev:pr/maxtext-ragged-router-grad"

# Run 33477529994 is one of the few whose pull_requests array GitHub did fill in.
EMBEDDED_RUN_ID = 33477529994
EMBEDDED_PULL_NUMBER = 5084
EMBEDDED_HEAD_SHA = "ebf592b5fde905684afb56702444e0a55b512488"

# The approval-gated run: completed, never executed, no jobs and no artifacts.
ACTION_REQUIRED_RUN_ID = 33465601432
# Its embedded pull request entry points at a DIFFERENT head sha than the run tested.
ACTION_REQUIRED_PULL_NUMBER = 5067

# The re-run whose attempt 2 carries 28 of its 42 jobs over from attempt 1.
RERUN_RUN_ID = 32772626658
RERUN_JOB_COUNT = 42
RERUN_CARRIED_OVER_JOBS = 28


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
    The decoded payload, deep-copied on every call so no test can mutate another's data.
  """
  return json.loads(read_fixture_bytes(name))


def load_runs(name: str) -> list[dict[str, Any]]:
  """Reads the `workflow_runs` array out of a saved runs listing.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The run payloads of that page, in the order GitHub returned them.
  """
  return load_fixture(name)["workflow_runs"]


def ids_of(payloads: list[dict[str, Any]]) -> list[int]:
  """Returns the ids of a list of run payloads, in order.

  Args:
    payloads: Run payloads.

  Returns:
    One id per payload.
  """
  return [payload["id"] for payload in payloads]


def flags_by_id(marked: list[dict[str, Any]]) -> dict[int, bool]:
  """Returns the superseded flag `mark_superseded` gave each run.

  Args:
    marked: What `mark_superseded` returned.

  Returns:
    Run id -> flag.
  """
  return {run["id"]: run[runs.SUPERSEDED_FIELD] for run in marked}


def utc(text: str) -> datetime:
  """Reads a UTC timestamp written the way the API writes it.

  Args:
    text: A timestamp such as "2026-09-01T00:00:00Z".

  Returns:
    The moment as a timezone-aware UTC datetime.
  """
  return datetime.fromisoformat(text.replace("Z", "+00:00"))


class CannedResponse(requests.Response):
  """A `requests.Response` whose status, body and headers are set directly.

  Subclassing keeps every reader the client uses - `.json()`, `.links`, the case-insensitive
  `.headers` - on its real implementation, so a Link header is parsed by requests itself.
  """

  def __init__(
      self,
      status: int = 200,
      body: bytes = b"",
      json_body: Any = None,
      headers: dict[str, str] | None = None,
  ) -> None:
    """Builds the response.

    Args:
      status: HTTP status code.
      body: Raw body bytes, which is how the saved fixtures are served.
      json_body: Object serialised as the JSON body, used when `body` is empty.
      headers: Response headers, for example a Link header pointing at the next page.
    """
    super().__init__()
    self.status_code = status
    self.encoding = "utf-8"
    self._content = body if body else json.dumps(json_body).encode("utf-8")
    self._content_consumed = True
    if headers:
      self.headers.update(headers)


class FakeSession:
  """Offline stand-in for `requests.Session` that answers from a scripted queue."""

  def __init__(self, responses: list[Any] | None = None) -> None:
    """Builds the fake.

    Args:
      responses: Answers to hand back in order. An exception instance is raised instead of
        being returned, which models a transport failure.
    """
    self.headers = CaseInsensitiveDict()
    self.queue: list[Any] = list(responses or [])
    self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
    self.closed = False

  def request(
      self,
      method: str,
      url: str,
      params: dict[str, Any] | None = None,
      headers: dict[str, Any] | None = None,
      auth: Any = None,
      allow_redirects: bool = True,
      timeout: Any = None,
  ) -> requests.Response:
    """Records the call and returns the next scripted answer.

    Args:
      method: HTTP method.
      url: Absolute URL requested.
      params: Query-string parameters, or None once paging follows a Link header.
      headers: Per-request header overrides, unused here.
      auth: Auth callable, unused here.
      allow_redirects: Whether requests may follow redirects itself, unused here.
      timeout: (connect, read) timeout, unused here.

    Returns:
      The next response in the queue.

    Raises:
      AssertionError: The client sent more requests than the test scripted.
    """
    del headers, auth, allow_redirects, timeout
    self.calls.append((method, url, dict(params) if params is not None else None))
    if not self.queue:
      raise AssertionError(f"the fake session ran out of scripted answers at {method} {url}")
    answer = self.queue.pop(0)
    if isinstance(answer, Exception):
      raise answer
    if not answer.url:
      answer.url = url
    return answer

  def close(self) -> None:
    """Marks the session closed so ownership can be asserted."""
    self.closed = True


class StubClient:
  """A stand-in for `github.GitHubClient` that serves saved payloads.

  It satisfies the `GitHubClientLike` protocol `runs.py` declares, and records what it was
  asked for, so a test can prove which endpoint was called with which query parameters.
  """

  def __init__(
      self,
      objects: dict[str, Any] | None = None,
      listings: dict[str, Any] | None = None,
  ) -> None:
    """Stores what the stub will serve.

    Args:
      objects: Path -> payload for `get_json`. An exception instance is raised instead.
      listings: Path -> list for `paginate`. An exception instance is raised instead.
    """
    self.objects = dict(objects or {})
    self.listings = dict(listings or {})
    self.get_json_calls: list[tuple[str, dict[str, Any]]] = []
    self.paginate_calls: list[tuple[str, str, dict[str, Any]]] = []

  def get_json(self, path: str, **params: Any) -> dict[str, Any]:
    """Returns the saved object for one path and records the call.

    Args:
      path: The repository-relative path asked for.
      **params: Query parameters.

    Returns:
      The staged payload.

    Raises:
      AssertionError: The test staged nothing for this path, which means the module called
        an endpoint the test did not expect.
      Exception: Whatever the test staged, to model a failing request.
    """
    self.get_json_calls.append((path, dict(params)))
    if path not in self.objects:
      raise AssertionError(f"no object is staged for {path!r}; the module called an unexpected endpoint")
    answer = self.objects[path]
    if isinstance(answer, Exception):
      raise answer
    return answer

  def paginate(self, path: str, key: str, **params: Any) -> list:
    """Returns the saved listing for one path and records the call.

    Args:
      path: The repository-relative path asked for.
      key: The list key inside the response, recorded so the test can check it.
      **params: Query parameters.

    Returns:
      A copy of the staged list.

    Raises:
      AssertionError: The test staged nothing for this path.
      Exception: Whatever the test staged, to model a failing request.
    """
    self.paginate_calls.append((path, key, dict(params)))
    if path not in self.listings:
      raise AssertionError(f"no listing is staged for {path!r}; the module called an unexpected endpoint")
    answer = self.listings[path]
    if isinstance(answer, Exception):
      raise answer
    return list(answer)


class OfflineTestCase(unittest.TestCase):
  """Base class that blocks the network, captures warnings and clears the module caches."""

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
    for module in (runs, github):
      warn_patch = mock.patch.object(module, "_warn", self.warnings.append)
      warn_patch.start()
      self.addCleanup(warn_patch.stop)

    sleep_patch = mock.patch.object(github, "_sleep", lambda seconds: None)
    sleep_patch.start()
    self.addCleanup(sleep_patch.stop)

    runs.clear_caches()
    self.addCleanup(runs.clear_caches)

  def make_client(self, responses: list[Any]) -> tuple[github.GitHubClient, FakeSession]:
    """Builds a real client whose network is a scripted queue of responses.

    Paging, the `created` parameter and the Link header are then exercised by the real
    implementation rather than by a stub that could agree with a bug.

    Args:
      responses: Answers the fake session hands back, in order.

    Returns:
      The client and the fake session behind it.
    """
    session = FakeSession(responses)
    client = github.GitHubClient(OWNER, REPO, token=TOKEN, session=session)
    return client, session

  def assert_no_warnings(self) -> None:
    """Asserts the module printed nothing on stderr during the test."""
    self.assertEqual(self.warnings, [])


def runs_page(name: str, next_url: str | None = None) -> CannedResponse:
  """Serves one saved runs listing page, optionally pointing at the next one.

  Args:
    name: Fixture file name.
    next_url: Absolute URL of the following page, or None for the last page.

  Returns:
    A response holding the fixture bytes and, when asked for, a Link header.
  """
  headers = {"Link": f'<{next_url}>; rel="next"'} if next_url else None
  return CannedResponse(body=read_fixture_bytes(name), headers=headers)


def workflow_runs_url(workflow_id: int, page: int) -> str:
  """Builds the URL of one page of a workflow's runs listing.

  Args:
    workflow_id: Numeric workflow id.
    page: Page number.

  Returns:
    The absolute URL, shaped the way GitHub's own Link header shapes it.
  """
  return f"{REPO_ROOT_URL}/actions/workflows/{workflow_id}/runs?per_page=100&page={page}"


class TimestampTest(OfflineTestCase):
  """Covers the timestamp readers every ordering rule in the module depends on."""

  def test_reads_the_api_timestamp_format(self) -> None:
    """A trailing Z is read as UTC, which is the only shape the API sends."""
    self.assertEqual(runs.parse_timestamp("2026-09-01T04:06:01Z"), utc("2026-09-01T04:06:01Z"))
    self.assertEqual(runs.parse_timestamp("2026-09-01T04:06:01+00:00"), utc("2026-09-01T04:06:01Z"))

  def test_reads_an_offset_back_into_utc(self) -> None:
    """A timestamp with an offset is converted, not merely relabelled."""
    self.assertEqual(runs.parse_timestamp("2026-09-01T06:06:01+02:00"), utc("2026-09-01T04:06:01Z"))

  def test_unreadable_timestamps_are_none_not_an_error(self) -> None:
    """Null, empty and malformed values give None so a caller can decide what to do."""
    for value in (None, "", "   ", "not a timestamp", 17, {"created_at": "x"}):
      with self.subTest(value=value):
        self.assertIsNone(runs.parse_timestamp(value))

  def test_naive_datetimes_are_read_as_utc(self) -> None:
    """A naive datetime is read as UTC rather than as the collector machine's local time."""
    self.assertEqual(runs.as_utc(datetime(2026, 9, 1, 4, 6, 1)), utc("2026-09-01T04:06:01Z"))

  def test_run_id_of_rejects_a_payload_with_no_id(self) -> None:
    """A run with no usable id cannot be fetched, so it raises instead of guessing."""
    self.assertEqual(runs.run_id_of({"id": 33468578834}), 33468578834)
    with self.assertRaises(runs.RunsError):
      runs.run_id_of({"name": "MaxText Package Tests"})


class WindowTest(OfflineTestCase):
  """Covers the query window: the `created` filter, the slicer and the boundary guard."""

  def test_created_filter_builds_an_open_window(self) -> None:
    """With no end the filter is the API's >= form."""
    self.assertEqual(runs.created_filter(utc("2026-08-25T00:00:00Z")), ">=2026-08-25T00:00:00Z")

  def test_created_filter_builds_a_closed_window(self) -> None:
    """With an end the filter is the API's range form, both ends inclusive."""
    self.assertEqual(
        runs.created_filter(utc("2026-08-25T00:00:00Z"), utc("2026-09-01T00:00:00Z")),
        "2026-08-25T00:00:00Z..2026-09-01T00:00:00Z",
    )

  def test_created_filter_refuses_a_backwards_window(self) -> None:
    """An end before the start would ask for nothing at all, so it is an error."""
    with self.assertRaises(runs.RunsError):
      runs.created_filter(utc("2026-09-01T00:00:00Z"), utc("2026-08-25T00:00:00Z"))

  def test_split_window_covers_the_whole_window_in_slices(self) -> None:
    """A 30-day backfill is cut into 7-day slices, oldest first, with no gap between them."""
    slices = runs.split_window(utc("2026-08-02T00:00:00Z"), utc("2026-09-01T00:00:00Z"))

    self.assertEqual(len(slices), 5)
    self.assertEqual(slices[0][0], utc("2026-08-02T00:00:00Z"))
    self.assertEqual(slices[-1][1], utc("2026-09-01T00:00:00Z"))
    for earlier, later in zip(slices, slices[1:]):
      self.assertEqual(earlier[1], later[0])

  def test_split_window_returns_one_slice_for_an_empty_window(self) -> None:
    """Asking for a single instant still gives a slice, so a caller never loops over nothing."""
    moment = utc("2026-09-01T00:00:00Z")
    self.assertEqual(runs.split_window(moment, moment), [(moment, moment)])

  def test_split_window_rejects_bad_arguments(self) -> None:
    """A backwards window or a non-positive slice width is an error, not a silent empty list."""
    with self.assertRaises(runs.RunsError):
      runs.split_window(utc("2026-09-01T00:00:00Z"), utc("2026-08-25T00:00:00Z"))
    with self.assertRaises(runs.RunsError):
      runs.split_window(utc("2026-08-25T00:00:00Z"), utc("2026-09-01T00:00:00Z"), days=0)

  def test_filter_runs_to_window_includes_both_ends(self) -> None:
    """The window is inclusive, so a run created on the boundary second is kept."""
    page = load_runs("runs-list-page1.json")
    kept = runs.filter_runs_to_window(page, utc("2026-09-01T03:52:04Z"), utc("2026-09-01T04:06:01Z"))

    self.assertEqual(ids_of(kept), [33468578834, 33467756955])

  def test_filter_runs_to_window_keeps_a_run_with_an_unreadable_timestamp(self) -> None:
    """Dropping a run over an unreadable field would lose real history, so it is kept and reported."""
    broken = dict(load_runs("runs-list-page1.json")[0], created_at=None)

    kept = runs.filter_runs_to_window([broken], utc("2026-09-01T00:00:00Z"))

    self.assertEqual(ids_of(kept), [PAGE_ONE_IDS[0]])
    self.assertEqual(len(self.warnings), 1)
    self.assertIn("no readable created_at", self.warnings[0])


class OrderingTest(OfflineTestCase):
  """Covers the two pure list rules: newest-first ordering and de-duplication."""

  def test_sort_runs_newest_first(self) -> None:
    """Merging the two saved pages and re-sorting gives one history, newest first."""
    merged = load_runs("runs-list-page2.json") + load_runs("runs-list-page1.json")

    ordered = runs.sort_runs_newest_first(merged)

    self.assertEqual(ids_of(ordered), list(PAGE_ONE_IDS + PAGE_TWO_IDS))

  def test_sort_runs_newest_first_does_not_modify_the_input(self) -> None:
    """Ordering returns a new list; the caller's own list is untouched."""
    page = load_runs("runs-list-page1.json")
    before = ids_of(page)

    runs.sort_runs_newest_first(reversed(page))

    self.assertEqual(ids_of(page), before)

  def test_dedupe_keeps_the_copy_that_saw_the_most_attempts(self) -> None:
    """A run returned twice by two windows is kept once, with the later read winning."""
    early = dict(load_runs("runs-list-page1.json")[1], run_attempt=1)
    late = dict(early, run_attempt=3)
    other = load_runs("runs-list-page1.json")[0]

    deduped = runs.dedupe_runs([early, other, late])

    self.assertEqual(ids_of(deduped), [early["id"], other["id"]])
    self.assertEqual(deduped[0]["run_attempt"], 3)

  def test_runs_with_no_readable_id_are_kept_apart_and_reported(self) -> None:
    """Two payloads with no id are not evidence of the same run, so neither is dropped.

    They used to share the key 0 and one was discarded without a word - silent data loss in a
    module whose habit everywhere else is to warn and keep.
    """
    page = load_runs("runs-list-page1.json")
    first = dict(page[0])
    second = dict(page[1])
    first.pop("id")
    second.pop("id")

    deduped = runs.dedupe_runs([first, second])

    self.assertEqual(len(deduped), 2)
    self.assertEqual(len(self.warnings), 2)
    self.assertIn("no readable id", self.warnings[0])


class AllowlistTest(OfflineTestCase):
  """Covers the rule that only the dashboard's own workflows are ever collected."""

  def test_a_workflow_id_that_is_not_a_number_raises_runs_error(self) -> None:
    """Every bad input in this module raises RunsError, so a tick catches one exception type.

    Passing a path where an id belongs used to escape as a bare ValueError from `int()`.
    """
    page = load_runs("runs-list-page1.json")

    with self.assertRaises(runs.RunsError):
      runs.filter_runs_to_workflows(page, workflow_ids=[runs.CI_PIPELINE_PATH])

  def foreign_run(self) -> dict[str, Any]:
    """Builds a run of a workflow the dashboard does not read.

    The repository really does run other workflows; this one is synthesised from a saved
    ci_pipeline run so that only its workflow identity differs.

    Returns:
      A run payload of workflow `OTHER_WORKFLOW_PATH`.
    """
    return dict(
        load_runs("runs-list-page1.json")[0],
        id=33477529995,
        workflow_id=OTHER_WORKFLOW_ID,
        path=OTHER_WORKFLOW_PATH,
        name="Unit Tests",
    )

  def test_filter_by_id_excludes_another_workflows_run(self) -> None:
    """A run whose workflow id is not wanted is dropped, whatever else it looks like."""
    page = load_runs("runs-list-page1.json") + [self.foreign_run()]

    kept = runs.filter_runs_to_workflows(page, workflow_ids=[CI_PIPELINE_WORKFLOW_ID])

    self.assertEqual(ids_of(kept), list(PAGE_ONE_IDS))

  def test_filter_by_path_excludes_another_workflows_run(self) -> None:
    """The path test keeps the collector working when the id lookup failed."""
    page = load_runs("runs-list-page1.json") + [self.foreign_run()]

    kept = runs.filter_runs_to_workflows(page, paths=runs.WORKFLOW_ALLOWLIST)

    self.assertEqual(ids_of(kept), list(PAGE_ONE_IDS))

  def test_filter_matches_on_either_id_or_path(self) -> None:
    """A run is kept when either identity matches, so a renamed file still collects."""
    moved = dict(load_runs("runs-list-page1.json")[0], path=".github/workflows/ci_pipeline_v2.yml")
    by_id = runs.filter_runs_to_workflows([moved], workflow_ids=[CI_PIPELINE_WORKFLOW_ID])
    self.assertEqual(ids_of(by_id), [PAGE_ONE_IDS[0]])

    renumbered = dict(load_runs("runs-list-page1.json")[0], workflow_id=OTHER_WORKFLOW_ID)
    by_path = runs.filter_runs_to_workflows([renumbered], paths=[runs.CI_PIPELINE_PATH])
    self.assertEqual(ids_of(by_path), [PAGE_ONE_IDS[0]])

  def test_filter_with_nothing_to_match_against_raises(self) -> None:
    """Returning every run would let unrelated workflows into the dashboard, so it raises."""
    with self.assertRaises(runs.RunsError):
      runs.filter_runs_to_workflows(load_runs("runs-list-page1.json"))

  def test_list_runs_sweep_path_excludes_another_workflows_run(self) -> None:
    """When no id resolves, the whole-repository sweep is filtered by path before returning."""
    workflows_page = CannedResponse(
        json_body={
            "total_count": 1,
            "workflows": [{"id": OTHER_WORKFLOW_ID, "name": "Unit Tests", "path": OTHER_WORKFLOW_PATH}],
        }
    )
    sweep_page = CannedResponse(
        json_body={"total_count": 4, "workflow_runs": load_runs("runs-list-page1.json") + [self.foreign_run()]}
    )
    client, session = self.make_client([workflows_page, sweep_page])

    collected = runs.list_runs(client, utc("2026-09-01T00:00:00Z"))

    self.assertEqual(ids_of(collected), list(PAGE_ONE_IDS))
    self.assertEqual(session.calls[1][1], f"{REPO_ROOT_URL}/actions/runs")
    self.assertTrue(any("no workflow id could be resolved" in line for line in self.warnings))

  def test_list_runs_asks_only_the_allowlisted_workflow_endpoints(self) -> None:
    """With ids resolved, each workflow is listed through its own endpoint and no other."""
    pages = [CannedResponse(json_body={"total_count": 0, "workflow_runs": []}) for _ in range(2)]
    client, session = self.make_client(pages)

    runs.list_runs(client, utc("2026-09-01T00:00:00Z"), workflow_ids=[CI_PIPELINE_WORKFLOW_ID, OTHER_WORKFLOW_ID])

    listed = [call[1] for call in session.calls]
    self.assertEqual(
        listed,
        [
            f"{REPO_ROOT_URL}/actions/workflows/{OTHER_WORKFLOW_ID}/runs",
            f"{REPO_ROOT_URL}/actions/workflows/{CI_PIPELINE_WORKFLOW_ID}/runs",
        ],
    )


class ResolveWorkflowIdsTest(OfflineTestCase):
  """Covers path-first workflow resolution and its display-name fallback."""

  def test_resolves_by_path_and_caches_the_answer(self) -> None:
    """The answer cannot change inside one tick, so the endpoint is read once per client."""
    listing = [
        {"id": CI_PIPELINE_WORKFLOW_ID, "name": "MaxText Package Tests", "path": runs.CI_PIPELINE_PATH},
        {"id": OTHER_WORKFLOW_ID, "name": "Unit Tests", "path": OTHER_WORKFLOW_PATH},
    ]
    client = StubClient(listings={runs.WORKFLOWS_ENDPOINT: listing})

    first = runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH])
    second = runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH])

    self.assertEqual(first, {runs.CI_PIPELINE_PATH: CI_PIPELINE_WORKFLOW_ID})
    self.assertEqual(second, first)
    self.assertEqual(len(client.paginate_calls), 1)
    self.assert_no_warnings()

  def test_falls_back_to_the_display_name_and_says_so(self) -> None:
    """A moved file is still found by name, but the guess is always reported."""
    listing = [{"id": CI_PIPELINE_WORKFLOW_ID, "name": "MaxText Package Tests", "path": ".github/workflows/moved.yml"}]
    client = StubClient(listings={runs.WORKFLOWS_ENDPOINT: listing})

    resolved = runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH])

    self.assertEqual(resolved, {runs.CI_PIPELINE_PATH: CI_PIPELINE_WORKFLOW_ID})
    self.assertTrue(any("falling back to the display name" in line for line in self.warnings))

  def test_unresolvable_workflow_is_left_out_with_a_warning(self) -> None:
    """A workflow that is gone is reported and skipped rather than faked."""
    client = StubClient(listings={runs.WORKFLOWS_ENDPOINT: []})

    resolved = runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH])

    self.assertEqual(resolved, {})
    self.assertTrue(any("was not found by path or by display name" in line for line in self.warnings))

  def test_rejects_a_listing_that_is_not_objects(self) -> None:
    """A payload of the wrong shape is an error, because guessing ids would collect nonsense."""
    client = StubClient(listings={runs.WORKFLOWS_ENDPOINT: ["ci_pipeline.yml"]})

    with self.assertRaises(runs.RunsError):
      runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH])


class ListRunsPagingTest(OfflineTestCase):
  """Covers paging, concatenation and the `created` filter, through the real client."""

  def test_follows_the_link_header_and_concatenates_both_pages(self) -> None:
    """Two saved pages come back as one list of six runs, newest first, ids in order."""
    responses = [
        runs_page("runs-list-page1.json", next_url=workflow_runs_url(CI_PIPELINE_WORKFLOW_ID, 2)),
        runs_page("runs-list-page2.json"),
    ]
    client, session = self.make_client(responses)

    collected = runs.list_runs(client, utc("2026-09-01T00:00:00Z"), workflow_ids=[CI_PIPELINE_WORKFLOW_ID])

    self.assertEqual(ids_of(collected), list(PAGE_ONE_IDS + PAGE_TWO_IDS))
    self.assertEqual(len(session.calls), 2)
    self.assert_no_warnings()

  def test_passes_the_created_filter_through_to_the_api(self) -> None:
    """The window is asked for, not filtered afterwards, so a tick stays cheap."""
    responses = [
        runs_page("runs-list-page1.json", next_url=workflow_runs_url(CI_PIPELINE_WORKFLOW_ID, 2)),
        runs_page("runs-list-page2.json"),
    ]
    client, session = self.make_client(responses)

    runs.list_runs(
        client,
        utc("2026-09-01T00:00:00Z"),
        utc("2026-09-01T23:59:59Z"),
        workflow_ids=[CI_PIPELINE_WORKFLOW_ID],
    )

    first_params = session.calls[0][2]
    self.assertEqual(first_params["created"], "2026-09-01T00:00:00Z..2026-09-01T23:59:59Z")
    self.assertEqual(first_params["per_page"], 100)
    # The Link header already carries page and per_page, so the second hop sends no params.
    self.assertIsNone(session.calls[1][2])

  def test_open_ended_window_uses_the_greater_than_form(self) -> None:
    """A tick with no end date asks for everything created since the newest stored run."""
    client, session = self.make_client([runs_page("runs-list-page1.json")])

    runs.list_runs(client, utc("2026-09-01T00:00:00Z"), workflow_ids=[CI_PIPELINE_WORKFLOW_ID])

    self.assertEqual(session.calls[0][2]["created"], ">=2026-09-01T00:00:00Z")

  def test_stops_on_a_short_final_page(self) -> None:
    """The 3 + 3 + 1 listing stops after the one-run page and returns all seven runs."""
    responses = [
        runs_page("runs-list-short-final-page1.json", next_url=workflow_runs_url(CI_PIPELINE_WORKFLOW_ID, 2)),
        runs_page("runs-list-short-final-page2.json", next_url=workflow_runs_url(CI_PIPELINE_WORKFLOW_ID, 3)),
        runs_page("runs-list-short-final-page3.json"),
    ]
    client, session = self.make_client(responses)

    collected = runs.list_runs(
        client,
        utc("2026-08-22T00:00:00Z"),
        utc("2026-08-29T00:00:00Z"),
        workflow_ids=[CI_PIPELINE_WORKFLOW_ID],
    )

    expected = [run_id for page in SHORT_FINAL_IDS for run_id in page]
    self.assertEqual(ids_of(collected), expected)
    self.assertEqual(len(collected), SHORT_FINAL_TOTAL_COUNT)
    self.assertEqual(len(session.calls), 3)

  def test_stops_when_a_page_carries_no_next_link(self) -> None:
    """Without a Link header a short page ends the listing; the second page is never asked for."""
    client, session = self.make_client([runs_page("runs-list-page1.json")])

    collected = runs.list_runs(client, utc("2026-09-01T00:00:00Z"), workflow_ids=[CI_PIPELINE_WORKFLOW_ID])

    self.assertEqual(ids_of(collected), list(PAGE_ONE_IDS))
    self.assertEqual(len(session.calls), 1)

  def test_the_saved_pages_do_not_overlap(self) -> None:
    """Guards the fixtures themselves: two pages that shared a run would hide a paging bug."""
    page_one = load_fixture("runs-list-page1.json")
    page_two = load_fixture("runs-list-page2.json")

    self.assertEqual(page_one["total_count"], PAGED_TOTAL_COUNT)
    self.assertEqual(page_two["total_count"], PAGED_TOTAL_COUNT)
    self.assertEqual(set(ids_of(page_one["workflow_runs"])) & set(ids_of(page_two["workflow_runs"])), set())

  def test_drops_a_run_the_api_returned_outside_the_window(self) -> None:
    """The window guard trims what the API included at the boundary."""
    responses = [
        runs_page("runs-list-short-final-page1.json", next_url=workflow_runs_url(CI_PIPELINE_WORKFLOW_ID, 2)),
        runs_page("runs-list-short-final-page2.json"),
    ]
    client, _ = self.make_client(responses)

    collected = runs.list_runs(
        client,
        utc("2026-08-26T00:00:00Z"),
        utc("2026-08-26T23:59:59Z"),
        workflow_ids=[CI_PIPELINE_WORKFLOW_ID],
    )

    self.assertEqual(ids_of(collected), [33003795267, 33003763519, 32995972111, 32981293288])

  def test_a_listing_of_something_other_than_objects_raises(self) -> None:
    """A payload of the wrong shape is named as an error rather than half-collected."""
    page = CannedResponse(json_body={"total_count": 1, "workflow_runs": ["33468578834"]})
    client, _ = self.make_client([page])

    with self.assertRaises(runs.RunsError):
      runs.list_runs(client, utc("2026-09-01T00:00:00Z"), workflow_ids=[CI_PIPELINE_WORKFLOW_ID])

  def test_warns_when_a_listing_comes_back_at_the_result_cap(self) -> None:
    """At 1000 results the older runs of the window were silently dropped by GitHub."""
    template = load_runs("runs-list-page1.json")[0]
    padded = [dict(template, id=39000000000 + index) for index in range(runs.RUNS_API_RESULT_CAP)]
    page = CannedResponse(json_body={"total_count": runs.RUNS_API_RESULT_CAP, "workflow_runs": padded})
    # A full page is followed by another request; GitHub answers the run out with an empty page.
    empty = CannedResponse(json_body={"total_count": runs.RUNS_API_RESULT_CAP, "workflow_runs": []})
    client, _ = self.make_client([page, empty])

    runs.list_runs(client, utc("2026-09-01T00:00:00Z"), workflow_ids=[CI_PIPELINE_WORKFLOW_ID])

    self.assertTrue(any("at GitHub's 1000-result cap" in line for line in self.warnings))


class MarkSupersededTest(OfflineTestCase):
  """Covers the cancelled-and-replaced rule, which decides what leaves every statistic."""

  def test_marks_the_cancelled_run_a_newer_run_replaced(self) -> None:
    """On branch parity-checkpoint-lifecycle the older cancelled run is superseded."""
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")

    marked = runs.mark_superseded(branch_runs)

    self.assertEqual(flags_by_id(marked), {SUCCESSOR_RUN_ID: False, SUPERSEDED_RUN_ID: True})
    self.assert_no_warnings()

  def test_leaves_the_newer_run_alone_whatever_its_conclusion(self) -> None:
    """The newest run of a branch is never superseded, even when it was cancelled too."""
    branch_runs = load_runs("cancelled-not-superseded-branch-runs.json")

    marked = runs.mark_superseded(branch_runs)

    self.assertEqual(flags_by_id(marked), {NEWEST_CANCELLED_RUN_ID: False, OLDER_CANCELLED_RUN_ID: True})

  def test_a_lone_cancelled_run_is_not_marked(self) -> None:
    """Somebody cancelling a run by hand is not the same thing as a push replacing it."""
    lone = load_fixture("superseded-cancelled-run-33462758754.json")

    marked = runs.mark_superseded([lone])

    self.assertEqual(flags_by_id(marked), {SUPERSEDED_RUN_ID: False})

  def test_a_successor_that_is_not_yet_in_the_list_leaves_the_run_unmarked(self) -> None:
    """The decision is made against the runs in hand; a later tick marks the rest."""
    cancelled = load_fixture("superseded-cancelled-run-33462758754.json")
    unrelated = load_fixture("merged-pr-5070-run-33406483779.json")

    marked = runs.mark_superseded([cancelled, unrelated])

    self.assertFalse(flags_by_id(marked)[SUPERSEDED_RUN_ID])

  def test_the_answer_does_not_depend_on_list_order(self) -> None:
    """Age comes from created_at, so shuffling the input cannot change who is superseded."""
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")

    newest_first = flags_by_id(runs.mark_superseded(branch_runs))
    oldest_first = flags_by_id(runs.mark_superseded(list(reversed(branch_runs))))

    self.assertEqual(newest_first, oldest_first)

  def test_ordering_follows_created_at_not_the_run_id(self) -> None:
    """Ids usually rise with time, but created_at is what the rule reads.

    The pair is synthesised from the real branch fixture with the ids swapped, because no
    saved run has an id that disagrees with its own creation order.
    """
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")
    successor, cancelled = branch_runs[0], branch_runs[1]
    self.assertGreater(runs.run_created_at(successor), runs.run_created_at(cancelled))

    swapped_cancelled = dict(cancelled, id=99999999999)
    swapped_successor = dict(successor, id=1)

    marked = runs.mark_superseded([swapped_successor, swapped_cancelled])

    self.assertEqual(flags_by_id(marked), {1: False, 99999999999: True})

  def test_runs_of_different_pull_requests_never_supersede_each_other(self) -> None:
    """ci_pipeline.yml groups a pull request run by its NUMBER, so two pull requests never mix.

    The newer run is re-homed onto another branch and another pull request, which is the only
    coherent way to write this case: a pull request's number and its head branch move together.
    """
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")
    other = copy.deepcopy(branch_runs[0])
    other["head_branch"] = "some-other-branch"
    other["pull_requests"] = [dict(other["pull_requests"][0], number=9999)]

    marked = runs.mark_superseded([other, branch_runs[1]])

    self.assertFalse(flags_by_id(marked)[SUPERSEDED_RUN_ID])

  def test_the_same_branch_name_in_two_forks_does_not_supersede(self) -> None:
    """Two forks can both push "main"; those runs never cancel each other.

    Both sides are synthesised: the saved branch listings hold same-repository runs only, and
    a fork run carries no pull request number, so both fall back to the branch key - which is
    where the head repository has to be part of the identity.
    """
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")
    forked = copy.deepcopy(branch_runs[0])
    forked["head_repository"]["full_name"] = "guowei-dev/maxtext"
    forked["pull_requests"] = []
    cancelled = dict(branch_runs[1], pull_requests=[])

    marked = runs.mark_superseded([forked, cancelled])

    self.assertFalse(flags_by_id(marked)[SUPERSEDED_RUN_ID])

  def test_a_cancelled_push_or_dispatch_run_is_never_superseded(self) -> None:
    """Only pull_request and schedule runs share a group; everything else is a group of one.

    ci_pipeline.yml's concurrency expression falls through to `github.run_id` for any other
    trigger, so nothing can cancel such a run. Grouping those by branch made the 4-hourly
    scheduled run look like the successor of every manual run on main, and a manual run
    somebody cancelled by hand then vanished from every statistic.
    """
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")
    for event in ("push", "workflow_dispatch", "workflow_call"):
      with self.subTest(event=event):
        cancelled = dict(branch_runs[1], event=event, head_branch="main", pull_requests=[])
        newer = dict(branch_runs[0], event="schedule", head_branch="main", pull_requests=[])

        marked = runs.mark_superseded([newer, cancelled])

        self.assertFalse(flags_by_id(marked)[SUPERSEDED_RUN_ID])

  def test_scheduled_runs_all_share_one_group(self) -> None:
    """The schedule group is per workflow, so an older cancelled scheduled run is superseded."""
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")
    cancelled = dict(branch_runs[1], event="schedule", head_branch="main", pull_requests=[])
    newer = dict(branch_runs[0], event="schedule", head_branch="main", pull_requests=[])

    marked = runs.mark_superseded([newer, cancelled])

    self.assertTrue(flags_by_id(marked)[SUPERSEDED_RUN_ID])

  def test_a_merged_pull_request_run_borrows_its_number_from_a_sibling(self) -> None:
    """`pull_requests` empties once a pull request merges, and the two runs must stay together.

    Run 33462758754 is cancelled and run 33463047689 replaced it; if the cancelled one loses
    its number first, keying on the number alone would put the pair in two different groups and
    the cancelled run would never be marked.
    """
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")
    emptied = dict(branch_runs[1], pull_requests=[])

    marked = runs.mark_superseded([branch_runs[0], emptied])

    self.assertTrue(flags_by_id(marked)[SUPERSEDED_RUN_ID])

  def test_returns_copies_in_the_input_order(self) -> None:
    """The caller's payloads are not modified, and the list order they passed is preserved."""
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")

    marked = runs.mark_superseded(branch_runs)

    self.assertEqual(ids_of(marked), [SUCCESSOR_RUN_ID, SUPERSEDED_RUN_ID])
    for original in branch_runs:
      self.assertNotIn(runs.SUPERSEDED_FIELD, original)

  def test_a_run_with_no_creation_time_is_reported_and_left_unmarked(self) -> None:
    """Without a creation time the run cannot be placed, and guessing would invent history."""
    branch_runs = load_runs("superseded-branch-runs-parity-checkpoint-lifecycle.json")
    broken = dict(branch_runs[1], created_at=None)

    marked = runs.mark_superseded([branch_runs[0], broken])

    self.assertEqual(flags_by_id(marked), {SUCCESSOR_RUN_ID: False, SUPERSEDED_RUN_ID: False})
    self.assertTrue(any("left unmarked by mark_superseded" in line for line in self.warnings))


class LinkPullRequestTest(OfflineTestCase):
  """Covers how a run is joined to its pull request, which is every chart's x axis."""

  def test_uses_the_embedded_entry_when_github_filled_it_in(self) -> None:
    """A populated pull_requests array is the answer, and costs no extra request."""
    run = load_runs("runs-list-page1.json")[0]
    self.assertEqual(run["head_sha"], EMBEDDED_HEAD_SHA)
    client = StubClient()

    linked = runs.link_pull_request(client, run)

    self.assertIsNotNone(linked)
    self.assertEqual(linked["number"], EMBEDDED_PULL_NUMBER)
    self.assertEqual(client.paginate_calls, [])
    self.assert_no_warnings()

  def test_several_embedded_entries_with_no_sha_match_are_refused(self) -> None:
    """`embedded_pull_request` must decline exactly where `match_pull_request` declines.

    It used to return the first entry, so the embedded path invented a link that the branch
    path would have refused, and which of the two ran decided the answer.
    """
    run = load_runs("runs-list-page1.json")[0]
    two = [{"number": 4000, "head": {"sha": "0" * 40}}, {"number": 4001, "head": {"sha": "1" * 40}}]
    run = dict(run, pull_requests=two)

    self.assertIsNone(runs.embedded_pull_request(run))
    self.assertIsNone(runs.match_pull_request(two, run))

  def test_the_embedded_entry_is_matched_by_head_sha(self) -> None:
    """With several entries the one that tested this commit wins."""
    run = load_runs("runs-list-page1.json")[0]
    stale = {"number": 4000, "head": {"sha": "0" * 40}}
    run = dict(run, pull_requests=[stale] + run["pull_requests"])

    self.assertEqual(runs.embedded_pull_request(run)["number"], EMBEDDED_PULL_NUMBER)

  def test_the_only_embedded_entry_is_used_even_when_the_sha_moved_on(self) -> None:
    """Run 33465601432 points at a pull request whose branch has since been pushed to.

    The run still belongs to that pull request, so the single entry is returned rather than
    nothing at all.
    """
    run = load_fixture("action-required-run-33465601432.json")
    embedded = runs.embedded_pull_request(run)

    self.assertIsNotNone(embedded)
    self.assertEqual(embedded["number"], ACTION_REQUIRED_PULL_NUMBER)
    self.assertNotEqual(embedded["head"]["sha"], run["head_sha"])

  def test_a_scheduled_run_is_never_linked_by_its_branch(self) -> None:
    """A scheduled run on main is not a pull request run, and `main` has been a head branch.

    Found by demo.py against the live API: run 33468578834 is the 4-hourly scheduled run on
    main, and asking for `head=AI-Hypercomputer:main` returns pull request #771 from 2024. It
    is the branch's only pull request, so the "only candidate" rule linked every scheduled run
    to it. The branch lookup now runs only for a `pull_request` run.
    """
    run = dict(load_fixture("merged-pr-5070-run-33406483779.json"), event="schedule", head_branch="main")
    client = StubClient(listings={runs.PULLS_ENDPOINT: load_fixture("merged-pr-5070-pulls-by-head.json")})

    linked = runs.link_pull_request(client, run)

    self.assertIsNone(linked)
    self.assertEqual(client.paginate_calls, [])
    self.assertEqual(self.warnings, [])

  def test_a_push_run_is_never_linked_by_its_branch_either(self) -> None:
    """Same rule for a push to main: the merge commit is not the pull request's head branch."""
    run = dict(load_fixture("merged-pr-5070-run-33406483779.json"), event="push", head_branch="main")
    client = StubClient(listings={runs.PULLS_ENDPOINT: load_fixture("merged-pr-5070-pulls-by-head.json")})

    self.assertIsNone(runs.link_pull_request(client, run))
    self.assertEqual(client.paginate_calls, [])

  def test_falls_back_to_the_branch_query_for_a_merged_same_repo_run(self) -> None:
    """Merged run 33406483779 carries an empty array, so the pulls endpoint answers instead."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    self.assertEqual(run["pull_requests"], [])
    client = StubClient(listings={runs.PULLS_ENDPOINT: load_fixture("merged-pr-5070-pulls-by-head.json")})

    linked = runs.link_pull_request(client, run)

    self.assertIsNotNone(linked)
    self.assertEqual(linked["number"], MERGED_PULL_NUMBER)
    self.assertEqual(linked["head"]["sha"], run["head_sha"])
    self.assertEqual(linked["merged_at"], "2026-09-01T02:59:51Z")
    self.assertEqual(
        client.paginate_calls,
        [(runs.PULLS_ENDPOINT, "pulls", {"head": MERGED_HEAD_QUERY, "state": "all"})],
    )
    self.assert_no_warnings()

  def test_the_branch_query_uses_the_fork_owner(self) -> None:
    """For fork run 33157998260 the head query names the contributor, not the base owner."""
    run = load_fixture("fork-pr-5042-run-33157998260.json")
    client = StubClient(listings={runs.PULLS_ENDPOINT: load_fixture("fork-pr-5042-pulls-by-head.json")})

    linked = runs.link_pull_request(client, run)

    self.assertEqual(runs.head_owner(run), "guowei-dev")
    self.assertIsNotNone(linked)
    self.assertEqual(linked["number"], FORK_PULL_NUMBER)
    self.assertEqual(linked["head"]["sha"], FORK_HEAD_SHA)
    self.assertEqual(client.paginate_calls[0][2], {"head": FORK_HEAD_QUERY, "state": "all"})

  def test_the_branch_lookup_is_asked_once_per_branch(self) -> None:
    """Two runs of one branch share the answer, which halves the requests on a busy branch."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    second = dict(run, id=run["id"] + 1)
    client = StubClient(listings={runs.PULLS_ENDPOINT: load_fixture("merged-pr-5070-pulls-by-head.json")})

    runs.link_pull_request(client, run)
    runs.link_pull_request(client, second)

    self.assertEqual(len(client.paginate_calls), 1)

  def test_returns_none_when_the_branch_has_no_pull_request(self) -> None:
    """A push to a branch with no pull request is not an error; it simply has no link."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    client = StubClient(listings={runs.PULLS_ENDPOINT: []})

    self.assertIsNone(runs.link_pull_request(client, run))

  def test_returns_none_when_the_run_names_no_branch(self) -> None:
    """Without a branch there is nothing to query, so it says so and answers None."""
    run = dict(load_fixture("merged-pr-5070-run-33406483779.json"), head_branch="")
    client = StubClient()

    self.assertIsNone(runs.link_pull_request(client, run))
    self.assertTrue(any("names no head branch" in line for line in self.warnings))

  def test_a_failed_lookup_gives_none_rather_than_ending_the_tick(self) -> None:
    """One unlinkable run must not stop the hundreds of runs collected beside it."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    failure = github.GitHubError("GET pulls answered 500", status=500)
    client = StubClient(listings={runs.PULLS_ENDPOINT: failure})

    self.assertIsNone(runs.link_pull_request(client, run))
    self.assertTrue(any("could not be linked to a pull request" in line for line in self.warnings))

  def test_a_listing_of_something_other_than_objects_raises(self) -> None:
    """A malformed pulls payload is named rather than half-read."""
    client = StubClient(listings={runs.PULLS_ENDPOINT: ["#5070"]})

    with self.assertRaises(runs.RunsError):
      runs.find_pull_requests_for_branch(client, OWNER, "aireen/script_grain_tfrecord")


class MatchPullRequestTest(OfflineTestCase):
  """Covers the pure head-sha join, including the case where guessing is refused."""

  def test_matches_the_pull_request_that_tested_this_commit(self) -> None:
    """The head sha is the join, so the saved pulls answer maps back to its own run."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    pulls = load_fixture("merged-pr-5070-pulls-by-head.json")

    matched = runs.match_pull_request(pulls, run)

    self.assertIsNotNone(matched)
    self.assertEqual(matched["number"], MERGED_PULL_NUMBER)
    self.assertEqual(run["head_sha"], MERGED_HEAD_SHA)

  def test_falls_back_to_the_branchs_only_pull_request(self) -> None:
    """A run of a commit that was later replaced still belongs to that pull request."""
    run = dict(load_fixture("merged-pr-5070-run-33406483779.json"), head_sha="0" * 40)
    pulls = load_fixture("merged-pr-5070-pulls-by-head.json")

    matched = runs.match_pull_request(pulls, run)

    self.assertIsNotNone(matched)
    self.assertEqual(matched["number"], MERGED_PULL_NUMBER)

  def test_refuses_to_guess_between_two_reuses_of_one_branch(self) -> None:
    """When a branch served several pull requests and no sha matches, the link is None."""
    run = dict(load_fixture("merged-pr-5070-run-33406483779.json"), head_sha="0" * 40)
    pulls = load_fixture("merged-pr-5070-pulls-by-head.json")
    older = dict(copy.deepcopy(pulls[0]), number=4000)

    self.assertIsNone(runs.match_pull_request(pulls + [older], run))

  def test_prefers_the_highest_numbered_match(self) -> None:
    """A branch reused across closed pull requests resolves to the newest one that matches."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    pulls = load_fixture("merged-pr-5070-pulls-by-head.json")
    older = dict(copy.deepcopy(pulls[0]), number=4000)

    matched = runs.match_pull_request([older] + pulls, run)

    self.assertEqual(matched["number"], MERGED_PULL_NUMBER)

  def test_no_candidates_gives_none(self) -> None:
    """An empty answer is None, not an exception."""
    self.assertIsNone(runs.match_pull_request([], load_fixture("merged-pr-5070-run-33406483779.json")))


class ResolvePullRequestTest(OfflineTestCase):
  """Covers turning whatever was linked into the full payload the dashboard needs."""

  def test_the_branch_lookup_already_answers_with_the_full_payload(self) -> None:
    """The pulls endpoint carries merged_at, so no second request is made."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    client = StubClient(listings={runs.PULLS_ENDPOINT: load_fixture("merged-pr-5070-pulls-by-head.json")})

    resolved = runs.resolve_pull_request(client, run)

    self.assertEqual(resolved["merged_at"], "2026-09-01T02:59:51Z")
    self.assertEqual(client.get_json_calls, [])

  def test_the_short_embedded_entry_is_upgraded_to_the_full_payload(self) -> None:
    """The embedded entry has no merged_at, and the x axis is merged pull requests.

    The full payload is synthesised, because no saved fixture holds pull request #5084.
    """
    run = load_runs("runs-list-page1.json")[0]
    full = {"number": EMBEDDED_PULL_NUMBER, "state": "closed", "merged_at": "2026-09-01T07:02:11Z"}
    client = StubClient(objects={f"{runs.PULLS_ENDPOINT}/{EMBEDDED_PULL_NUMBER}": full})

    resolved = runs.resolve_pull_request(client, run)

    self.assertEqual(resolved, full)
    self.assertEqual(client.get_json_calls, [(f"{runs.PULLS_ENDPOINT}/{EMBEDDED_PULL_NUMBER}", {})])

  def test_keeps_the_short_entry_when_the_extra_read_fails(self) -> None:
    """Half an answer beats none: the run keeps its pull request number."""
    run = load_runs("runs-list-page1.json")[0]
    failure = github.GitHubError("GET pulls/5084 answered 500", status=500)
    client = StubClient(objects={f"{runs.PULLS_ENDPOINT}/{EMBEDDED_PULL_NUMBER}": failure})

    resolved = runs.resolve_pull_request(client, run)

    self.assertEqual(resolved["number"], EMBEDDED_PULL_NUMBER)
    self.assertTrue(any("keeping the short entry" in line for line in self.warnings))

  def test_a_missing_pull_request_is_reported_and_the_short_entry_kept(self) -> None:
    """A 404 from the pulls endpoint is an answer, not a failure."""
    run = load_runs("runs-list-page1.json")[0]
    missing = github.GitHubError("GET pulls/5084 answered 404", status=404)
    client = StubClient(objects={f"{runs.PULLS_ENDPOINT}/{EMBEDDED_PULL_NUMBER}": missing})

    resolved = runs.resolve_pull_request(client, run)

    self.assertEqual(resolved["number"], EMBEDDED_PULL_NUMBER)
    self.assertTrue(any("was not found" in line for line in self.warnings))

  def test_an_unlinkable_run_stays_none(self) -> None:
    """Nothing to upgrade means nothing is fetched."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    client = StubClient(listings={runs.PULLS_ENDPOINT: []})

    self.assertIsNone(runs.resolve_pull_request(client, run))
    self.assertEqual(client.get_json_calls, [])


class GetJobsTest(OfflineTestCase):
  """Covers the jobs endpoint, including the two shapes that mean "this attempt ran nothing"."""

  def test_an_action_required_run_returns_an_empty_list(self) -> None:
    """Run 33465601432 was never approved, so its jobs endpoint answers with an empty array."""
    page = CannedResponse(body=read_fixture_bytes("action-required-run-33465601432-jobs.json"))
    client, session = self.make_client([page])

    jobs = runs.get_jobs(client, ACTION_REQUIRED_RUN_ID, 1)

    self.assertEqual(jobs, [])
    self.assertEqual(len(session.calls), 1)
    self.assertEqual(
        session.calls[0][1],
        f"{REPO_ROOT_URL}/actions/runs/{ACTION_REQUIRED_RUN_ID}/attempts/1/jobs",
    )
    self.assert_no_warnings()

  def test_the_action_required_run_itself_never_executed(self) -> None:
    """Guards the fixture: the run is completed, approval-gated, and on its first attempt."""
    run = load_fixture("action-required-run-33465601432.json")

    self.assertEqual(run["status"], "completed")
    self.assertEqual(run["conclusion"], "action_required")
    self.assertEqual(run["run_attempt"], 1)
    self.assertEqual(load_fixture("action-required-run-33465601432-jobs.json")["total_count"], 0)

  def test_a_pruned_attempt_is_read_as_no_jobs(self) -> None:
    """A 404 means GitHub no longer serves that attempt, which is a fact about the run."""
    missing = github.GitHubError("GET attempts/1/jobs answered 404", status=404)
    client = StubClient(listings={f"actions/runs/{ACTION_REQUIRED_RUN_ID}/attempts/1/jobs": missing})

    self.assertEqual(runs.get_jobs(client, ACTION_REQUIRED_RUN_ID, 1), [])
    self.assertTrue(any("reading it as no jobs" in line for line in self.warnings))

  def test_any_other_failure_is_raised(self) -> None:
    """A 500 is a broken tick, not an empty run, so it is not swallowed."""
    failure = github.GitHubError("GET attempts/1/jobs answered 500", status=500)
    client = StubClient(listings={f"actions/runs/{ACTION_REQUIRED_RUN_ID}/attempts/1/jobs": failure})

    with self.assertRaises(github.GitHubError):
      runs.get_jobs(client, ACTION_REQUIRED_RUN_ID, 1)

  def test_an_attempt_number_below_one_raises(self) -> None:
    """Attempts count from 1; asking for 0 would silently fetch the wrong URL."""
    client = StubClient()

    with self.assertRaises(runs.RunsError):
      runs.get_jobs(client, ACTION_REQUIRED_RUN_ID, 0)

  def test_carried_over_jobs_are_returned_untouched(self) -> None:
    """Attempt 2 of run 32772626658 holds 42 jobs, 28 of them carried over from attempt 1.

    Those carried-over jobs have a started_at that precedes their created_at, which makes a
    naive queue wait negative. `get_jobs` must not hide them: deciding what a job's numbers
    mean belongs to derive.py.
    """
    payload = load_fixture("rerun-32772626658-attempt2-jobs.json")
    client = StubClient(listings={f"actions/runs/{RERUN_RUN_ID}/attempts/2/jobs": payload["jobs"]})

    jobs = runs.get_jobs(client, RERUN_RUN_ID, 2)

    self.assertEqual(len(jobs), RERUN_JOB_COUNT)
    carried = [
        job
        for job in jobs
        if runs.parse_timestamp(job.get("started_at"))
        and runs.parse_timestamp(job.get("created_at"))
        and runs.parse_timestamp(job["started_at"]) < runs.parse_timestamp(job["created_at"])
    ]
    self.assertEqual(len(carried), RERUN_CARRIED_OVER_JOBS)

  def test_a_listing_of_something_other_than_objects_raises(self) -> None:
    """A malformed jobs payload is named rather than half-read."""
    client = StubClient(listings={f"actions/runs/{RERUN_RUN_ID}/attempts/1/jobs": ["a job"]})

    with self.assertRaises(runs.RunsError):
      runs.get_jobs(client, RERUN_RUN_ID, 1)


class ListAttemptsTest(OfflineTestCase):
  """Covers re-reading the run, because `run_attempt` grows between two reads."""

  def test_rereads_the_run_and_fetches_every_earlier_attempt(self) -> None:
    """Run 32785979907 read attempt 2 from a listing and 3 from the run endpoint 26 minutes later.

    The stale copy is what a tick holds; the fresh read is what decides how many attempts to
    fetch, and the fresh payload is itself the newest attempt.
    """
    fresh = load_fixture("cancelled-job-32785979907-run.json")
    self.assertEqual(fresh["run_attempt"], 3)
    stale = dict(fresh, run_attempt=2)
    run_id = fresh["id"]
    client = StubClient(
        objects={
            f"actions/runs/{run_id}": fresh,
            f"actions/runs/{run_id}/attempts/1": dict(fresh, run_attempt=1),
            f"actions/runs/{run_id}/attempts/2": dict(fresh, run_attempt=2),
        }
    )

    attempts = runs.list_attempts(client, stale)

    self.assertEqual([attempt["run_attempt"] for attempt in attempts], [1, 2, 3])
    self.assertEqual(client.get_json_calls[0][0], f"actions/runs/{run_id}")
    self.assert_no_warnings()

  def test_a_single_attempt_run_costs_one_request(self) -> None:
    """A run that never re-ran is its own only attempt."""
    run = load_fixture("merged-pr-5070-run-33406483779.json")
    client = StubClient(objects={f"actions/runs/{MERGED_RUN_ID}": run})

    attempts = runs.list_attempts(client, run)

    self.assertEqual(len(attempts), 1)
    self.assertEqual(attempts[0]["id"], MERGED_RUN_ID)
    self.assertEqual(len(client.get_json_calls), 1)

  def test_an_attempt_github_no_longer_serves_is_skipped(self) -> None:
    """A pruned attempt shortens the list and is reported, rather than ending the tick."""
    fresh = load_fixture("rerun-32772626658-run.json")
    run_id = fresh["id"]
    client = StubClient(
        objects={
            f"actions/runs/{run_id}": fresh,
            f"actions/runs/{run_id}/attempts/1": github.GitHubError("gone", status=404),
        }
    )

    attempts = runs.list_attempts(client, fresh)

    self.assertEqual(len(attempts), 1)
    self.assertEqual(attempts[0]["run_attempt"], 2)
    self.assertTrue(any("no longer served by GitHub" in line for line in self.warnings))

  def test_a_failure_other_than_a_missing_attempt_is_raised(self) -> None:
    """A 500 on an attempt is a broken tick, so it is not mistaken for a pruned attempt."""
    fresh = load_fixture("rerun-32772626658-run.json")
    run_id = fresh["id"]
    client = StubClient(
        objects={
            f"actions/runs/{run_id}": fresh,
            f"actions/runs/{run_id}/attempts/1": github.GitHubError("boom", status=500),
        }
    )

    with self.assertRaises(github.GitHubError):
      runs.list_attempts(client, fresh)


class InFlightRunTest(OfflineTestCase):
  """Covers the shape no fixture could capture: a run that has not finished yet.

  No in-flight run existed in the repository while the fixtures were being collected, so the
  payload here is synthesised from a saved one by setting `status` and `conclusion` to what
  GitHub sends while a run is still going. Only `status` is checked, which is the field the
  storage rule reads.
  """

  def running_run(self) -> dict[str, Any]:
    """Builds a run payload as it looks while the run is still going.

    Returns:
      A copy of a saved run with `status` in_progress and no conclusion yet.
    """
    return dict(load_fixture("merged-pr-5070-run-33406483779.json"), status="in_progress", conclusion=None)

  def test_an_in_flight_run_is_listed_like_any_other(self) -> None:
    """Discovery does not judge a run; the caller checks `status` before storing it."""
    running = self.running_run()
    page = CannedResponse(json_body={"total_count": 1, "workflow_runs": [running]})
    client, _ = self.make_client([page])

    collected = runs.list_runs(client, utc("2026-08-31T00:00:00Z"), workflow_ids=[CI_PIPELINE_WORKFLOW_ID])

    self.assertEqual(ids_of(collected), [MERGED_RUN_ID])
    self.assertEqual(collected[0]["status"], "in_progress")
    self.assertIsNone(collected[0]["conclusion"])

  def test_an_in_flight_run_is_never_superseded(self) -> None:
    """Supersession needs the conclusion `cancelled`; a running run has no conclusion at all."""
    running = self.running_run()
    newer = dict(running, id=running["id"] + 1, created_at="2026-08-31T16:00:00Z")

    marked = runs.mark_superseded([running, newer])

    self.assertEqual(flags_by_id(marked), {running["id"]: False, newer["id"]: False})


class CacheTest(OfflineTestCase):
  """Covers `clear_caches`, the seam that stops one tick's answers leaking into the next."""

  def test_clearing_the_caches_makes_both_lookups_ask_again(self) -> None:
    """Workflow ids and branch pull requests are both per-tick answers, not per-process ones."""
    listing = [{"id": CI_PIPELINE_WORKFLOW_ID, "name": "MaxText Package Tests", "path": runs.CI_PIPELINE_PATH}]
    client = StubClient(
        listings={
            runs.WORKFLOWS_ENDPOINT: listing,
            runs.PULLS_ENDPOINT: load_fixture("merged-pr-5070-pulls-by-head.json"),
        }
    )
    run = load_fixture("merged-pr-5070-run-33406483779.json")

    runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH])
    runs.link_pull_request(client, run)
    runs.clear_caches()
    runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH])
    runs.link_pull_request(client, run)

    paths = [call[0] for call in client.paginate_calls]
    self.assertEqual(paths.count(runs.WORKFLOWS_ENDPOINT), 2)
    self.assertEqual(paths.count(runs.PULLS_ENDPOINT), 2)

  def test_refresh_reads_the_workflows_endpoint_again(self) -> None:
    """`refresh=True` is the documented way to re-read without clearing every cache."""
    listing = [{"id": CI_PIPELINE_WORKFLOW_ID, "name": "MaxText Package Tests", "path": runs.CI_PIPELINE_PATH}]
    client = StubClient(listings={runs.WORKFLOWS_ENDPOINT: listing})

    runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH])
    runs.resolve_workflow_ids(client, paths=[runs.CI_PIPELINE_PATH], refresh=True)

    self.assertEqual(len(client.paginate_calls), 2)


class BackfillShapeTest(OfflineTestCase):
  """Covers the backfill walk: slice a wide window, list each slice, join the answers."""

  def test_a_thirty_day_backfill_lists_each_slice_and_dedupes_the_seams(self) -> None:
    """Both ends of a slice are inclusive, so a run on a seam arrives twice and is kept once."""
    since = utc("2026-08-02T00:00:00Z")
    until = utc("2026-09-01T00:00:00Z")
    slices = runs.split_window(since, until)
    seam_run = dict(load_runs("runs-list-page1.json")[0], id=33400000000, created_at=slices[0][1].isoformat())

    collected: list[dict[str, Any]] = []
    for start, end in slices:
      page = CannedResponse(json_body={"total_count": 1, "workflow_runs": [dict(seam_run)]})
      client, session = self.make_client([page])
      collected.extend(runs.list_runs(client, start, end, workflow_ids=[CI_PIPELINE_WORKFLOW_ID]))
      self.assertIn("created", session.calls[0][2])

    # The seam run is inside slice 1 and slice 2, so it was listed twice before de-duplication.
    self.assertEqual(len(collected), 2)
    self.assertEqual(len(runs.dedupe_runs(collected)), 1)

  def test_slices_never_exceed_the_backfill_width(self) -> None:
    """Each slice stays inside seven days, which is what keeps a listing under the cap."""
    slices = runs.split_window(utc("2026-06-03T00:00:00Z"), utc("2026-09-01T00:00:00Z"))

    for start, end in slices:
      self.assertLessEqual(end - start, timedelta(days=runs.BACKFILL_WINDOW_DAYS))


if __name__ == "__main__":
  unittest.main(verbosity=2)
