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

"""Offline unit tests for the collector's read-only GitHub client.

Nothing here touches the network. Every test drives `GitHubClient` through a fake session
that answers from a scripted queue of responses, and the two seams that would make a test
slow or noisy -- `_sleep` and `_warn` -- are patched out. The suite therefore runs with no
GitHub token, no connection and no waiting.

The responses are real `requests.Response` subclasses rather than stubs, so `.json()`,
`.text`, `.links` and case-insensitive header lookup behave exactly as they do against
api.github.com.

The tests are plain `unittest`, so they need nothing but the standard library and
`requests`. pytest collects them too, because it understands `unittest.TestCase` and the
file is named the way the repository's pytest.ini expects.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/github_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/github_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import requests
from requests import sessions as requests_sessions
from requests.structures import CaseInsensitiveDict

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import github

OWNER = "AI-Hypercomputer"
REPO = "maxtext"
TOKEN = "ghp-test-token-never-real"
REPO_ROOT_URL = f"{github.API_ROOT}/repos/{OWNER}/{REPO}"


def set_token_warning_emitted(value: bool) -> None:
  """Sets the module's once-per-process no-token warning flag.

  The flag is a documented test seam: without resetting it, whichever test happens to run
  first would be the only one that can observe the warning.

  Args:
    value: True to mark the warning as already printed, False to arm it again.
  """
  github._TOKEN_WARNING_EMITTED = value  # pylint: disable=protected-access


def token_warning_emitted() -> bool:
  """Reads the module's once-per-process no-token warning flag.

  Returns:
    True when the warning has already been printed in this process.
  """
  return github._TOKEN_WARNING_EMITTED  # pylint: disable=protected-access


def merge_headers(session_headers: Any, request_headers: dict[str, Any] | None) -> CaseInsensitiveDict:
  """Merges per-request headers over session headers the way `requests` does.

  `requests` drops any header whose per-request value is None. That is the idiom github.py
  uses to strip Authorization before following an artifact redirect, so the tests have to
  model it to see what the redirect target would really receive.

  Args:
    session_headers: Headers held on the session.
    request_headers: Headers passed for this one request, or None.

  Returns:
    The headers that would actually go out on the wire.
  """
  merged = CaseInsensitiveDict(dict(session_headers))
  for name, value in (request_headers or {}).items():
    if value is None:
      merged.pop(name, None)
    else:
      merged[name] = value
  return merged


class CannedResponse(requests.Response):
  """A `requests.Response` whose status, body and headers are set directly.

  Subclassing keeps every reader the client uses -- `.json()`, `.text`, `.links`, the
  case-insensitive `.headers` -- on its real implementation.
  """

  def __init__(
      self,
      status: int = 200,
      json_body: Any = None,
      body: bytes = b"",
      headers: dict[str, str] | None = None,
      url: str | None = None,
  ) -> None:
    """Builds the response.

    Args:
      status: HTTP status code.
      json_body: Object serialised as the JSON body. Ignored when None.
      body: Raw body bytes, used when json_body is None.
      headers: Response headers, for example a Link header or rate-limit headers.
      url: URL to report as `.url`. The fake session fills it in when omitted.
    """
    super().__init__()
    self.status_code = status
    self.encoding = "utf-8"
    self._content = json.dumps(json_body).encode("utf-8") if json_body is not None else body
    self._content_consumed = True
    if headers:
      self.headers.update(headers)
    if url is not None:
      self.url = url


class RecordedCall:
  """One request the fake session was asked to send."""

  def __init__(
      self,
      method: str,
      url: str,
      params: dict[str, Any] | None,
      headers: dict[str, Any] | None,
      auth: Any,
      allow_redirects: bool,
      timeout: Any,
      session_headers: Any,
  ) -> None:
    """Records the arguments of one call.

    Args:
      method: HTTP method.
      url: Absolute URL requested.
      params: Query-string parameters, or None.
      headers: Per-request header overrides, or None.
      auth: Auth callable passed for this request, or None.
      allow_redirects: Whether requests was allowed to follow redirects itself.
      timeout: (connect, read) timeout tuple.
      session_headers: Session headers at the moment of the call.
    """
    self.method = method
    self.url = url
    self.params = dict(params) if params is not None else None
    self.headers = headers
    self.auth = auth
    self.allow_redirects = allow_redirects
    self.timeout = timeout
    self.sent_headers = merge_headers(session_headers, headers)


class FakeSession:
  """Offline stand-in for `requests.Session` that answers from a scripted queue."""

  def __init__(self, responses: list[Any] | None = None) -> None:
    """Builds the fake.

    Args:
      responses: Answers to hand back in order. Each item is either a response or an
        exception instance, which is raised instead of being returned.
    """
    self.headers = CaseInsensitiveDict()
    self.calls: list[RecordedCall] = []
    self.queue: list[Any] = list(responses or [])
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
      params: Query-string parameters, or None.
      headers: Per-request header overrides, or None.
      auth: Auth callable, or None.
      allow_redirects: Whether requests may follow redirects itself.
      timeout: (connect, read) timeout tuple.

    Returns:
      The next response in the queue.

    Raises:
      AssertionError: When the client sends more requests than the test scripted.
      Exception: Whatever exception instance the test queued, to model a transport failure.
    """
    self.calls.append(RecordedCall(method, url, params, headers, auth, allow_redirects, timeout, self.headers))
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

  @property
  def call_count(self) -> int:
    """Returns how many requests the client sent.

    Returns:
      The number of recorded calls.
    """
    return len(self.calls)


class GitHubClientTestCase(unittest.TestCase):
  """Base class that patches out sleeping, warnings and any ambient GITHUB_TOKEN."""

  def setUp(self) -> None:
    """Installs the offline seams and restores them after each test."""
    super().setUp()

    env_patch = mock.patch.dict(os.environ, {}, clear=False)
    env_patch.start()
    self.addCleanup(env_patch.stop)
    os.environ.pop("GITHUB_TOKEN", None)

    self.sleeps: list[float] = []
    sleep_patch = mock.patch.object(github, "_sleep", self.sleeps.append)
    sleep_patch.start()
    self.addCleanup(sleep_patch.stop)

    self.warnings: list[str] = []
    warn_patch = mock.patch.object(github, "_warn", self.warnings.append)
    warn_patch.start()
    self.addCleanup(warn_patch.stop)

    previously_emitted = token_warning_emitted()
    self.addCleanup(set_token_warning_emitted, previously_emitted)
    set_token_warning_emitted(True)

  def make_client(
      self,
      responses: list[Any] | None = None,
      token: str | None = TOKEN,
  ) -> tuple[github.GitHubClient, FakeSession]:
    """Builds a client wired to a fake session preloaded with scripted answers.

    Args:
      responses: Answers the fake session hands back, in order.
      token: Token to build the client with. None exercises the unauthenticated path.

    Returns:
      The client and the fake session behind it.
    """
    session = FakeSession(responses)
    client = github.GitHubClient(OWNER, REPO, token=token, session=session)
    return client, session


class ConstructionTest(GitHubClientTestCase):
  """Covers authentication, the once-per-process warning and session ownership."""

  def test_token_argument_sets_the_authorization_header(self) -> None:
    """An explicit token becomes a Bearer header on the session."""
    client, session = self.make_client(token=TOKEN)

    self.assertEqual(client.token, TOKEN)
    self.assertEqual(session.headers["Authorization"], f"Bearer {TOKEN}")
    self.assertEqual(self.warnings, [])

  def test_token_falls_back_to_the_environment_variable(self) -> None:
    """With no token argument the client reads GITHUB_TOKEN."""
    os.environ["GITHUB_TOKEN"] = "env-token"

    client, session = self.make_client(token=None)

    self.assertEqual(client.token, "env-token")
    self.assertEqual(session.headers["Authorization"], "Bearer env-token")

  def test_missing_token_warns_once_and_still_constructs(self) -> None:
    """Without a token the client is usable, unauthenticated, and warns exactly once."""
    set_token_warning_emitted(False)

    first, first_session = self.make_client(token=None)
    second, second_session = self.make_client(token=None)

    self.assertIsNone(first.token)
    self.assertIsNone(second.token)
    self.assertNotIn("Authorization", first_session.headers)
    self.assertNotIn("Authorization", second_session.headers)
    self.assertEqual(len(self.warnings), 1)
    self.assertIn("no GitHub token", self.warnings[0])
    self.assertTrue(token_warning_emitted())

  def test_default_headers_are_set_on_the_session(self) -> None:
    """Accept, API version and user agent are pinned on every request."""
    _, session = self.make_client()

    self.assertEqual(session.headers["Accept"], github.ACCEPT_HEADER)
    self.assertEqual(session.headers["X-GitHub-Api-Version"], github.API_VERSION_HEADER)
    self.assertEqual(session.headers["User-Agent"], github.USER_AGENT)

  def test_close_leaves_a_borrowed_session_open(self) -> None:
    """A session handed in by the caller stays the caller's to close."""
    client, session = self.make_client()

    client.close()

    self.assertFalse(session.closed)


class CredentialScopeTest(GitHubClientTestCase):
  """Covers the rule that only api.github.com is ever sent the token."""

  OTHER_HOST_URL = "https://attacker.example.com/artifact.zip"

  def test_the_token_is_not_sent_to_a_host_that_is_not_the_api(self) -> None:
    """A URL on any other host is requested anonymously, even on the very first hop.

    `get_bytes` is handed whatever URL the caller passes. If that URL ever pointed somewhere
    other than api.github.com -- a pre-signed storage link in `archive_download_url`, or a
    caller that built the URL itself -- sending the token would hand it to a third party.
    """
    client, session = self.make_client([CannedResponse(200, body=b"zip")], token=TOKEN)

    client.get_bytes(self.OTHER_HOST_URL)

    self.assertEqual(session.call_count, 1)
    hop = session.calls[0]
    self.assertNotIn("Authorization", hop.sent_headers)
    self.assertNotIn(TOKEN, repr(dict(hop.sent_headers)))
    # The session keeps its credential for the next api.github.com call.
    self.assertEqual(session.headers["Authorization"], f"Bearer {TOKEN}")

  def test_the_token_is_still_sent_to_the_api_host(self) -> None:
    """The rule is by host, so ordinary API calls are unaffected."""
    client, session = self.make_client([CannedResponse(200, json_body={"id": 1})], token=TOKEN)

    client.get_json("actions/runs/1")

    self.assertEqual(session.calls[0].sent_headers["Authorization"], f"Bearer {TOKEN}")

  def test_a_link_header_pointing_off_the_api_host_is_followed_anonymously(self) -> None:
    """Pagination follows Link headers, so the same host rule has to cover them."""
    elsewhere = "https://example.invalid/page2"
    page_1 = CannedResponse(200, json_body={"jobs": [{"id": 1}]}, headers={"Link": f'<{elsewhere}>; rel="next"'})
    page_2 = CannedResponse(200, json_body={"jobs": []})
    client, session = self.make_client([page_1, page_2], token=TOKEN)

    client.paginate("actions/runs/1/jobs", "jobs", per_page=1)

    self.assertEqual(session.calls[1].url, elsewhere)
    self.assertNotIn("Authorization", session.calls[1].sent_headers)

  def test_a_client_without_a_token_does_not_inherit_one_from_a_shared_session(self) -> None:
    """A tokenless client on a borrowed session must send nothing, not someone else's token.

    The constructor invites session sharing. Without clearing the header, `client.token` would
    be None, the no-token warning would print, and the requests would still go out
    authenticated as whoever built the first client.
    """
    session = FakeSession([CannedResponse(200, json_body={"id": 1})])
    first = github.GitHubClient(OWNER, REPO, token=TOKEN, session=session)
    self.assertEqual(session.headers["Authorization"], f"Bearer {TOKEN}")

    set_token_warning_emitted(False)
    second = github.GitHubClient(OWNER, REPO, token=None, session=session)

    self.assertIsNone(second.token)
    self.assertIsNotNone(first.token)
    self.assertNotIn("Authorization", session.headers)

    second.get_json("actions/runs/1")
    self.assertNotIn("Authorization", session.calls[0].sent_headers)
    self.assertEqual(self.warnings, [github.NO_TOKEN_WARNING])

  def test_a_signed_download_url_is_not_written_into_a_warning_or_an_error(self) -> None:
    """Storage URLs carry a signature in the query string; logs must not repeat it."""
    signed = "https://storage.example.com/artifacts/1?sig=SECRET-SIGNATURE"
    redirect = CannedResponse(302, headers={"Location": signed})
    failures = [CannedResponse(500, body=b"boom") for _ in range(github.MAX_ATTEMPTS)]
    client, _ = self.make_client([redirect, *failures])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(f"{REPO_ROOT_URL}/actions/artifacts/1/zip")

    self.assertNotIn("SECRET-SIGNATURE", str(caught.exception))
    self.assertNotIn("SECRET-SIGNATURE", str(caught.exception.url))
    self.assertTrue(self.warnings)
    self.assertNotIn("SECRET-SIGNATURE", " ".join(self.warnings))
    self.assertIn("storage.example.com/artifacts/1", str(caught.exception))


class GetJsonTest(GitHubClientTestCase):
  """Covers URL building, query parameters and body validation for single objects."""

  def test_builds_the_repository_url_from_a_relative_path(self) -> None:
    """A repository-relative path is hung under /repos/{owner}/{repo}."""
    client, session = self.make_client([CannedResponse(200, json_body={"id": 33468578834})])

    payload = client.get_json("actions/runs/33468578834")

    self.assertEqual(payload, {"id": 33468578834})
    self.assertEqual(session.call_count, 1)
    self.assertEqual(session.calls[0].method, "GET")
    self.assertEqual(session.calls[0].url, f"{REPO_ROOT_URL}/actions/runs/33468578834")

  def test_strips_a_leading_slash_from_the_path(self) -> None:
    """A leading slash never doubles up in the built URL."""
    client, session = self.make_client([CannedResponse(200, json_body={})])

    client.get_json("/actions/runs/1")

    self.assertEqual(session.calls[0].url, f"{REPO_ROOT_URL}/actions/runs/1")

  def test_passes_an_absolute_url_through_unchanged(self) -> None:
    """An absolute https URL is requested as given, not re-prefixed."""
    absolute = "https://api.github.com/repos/other/other/actions/runs/9"
    client, session = self.make_client([CannedResponse(200, json_body={})])

    client.get_json(absolute)

    self.assertEqual(session.calls[0].url, absolute)

  def test_passes_params_through(self) -> None:
    """Keyword arguments reach the wire as the query string."""
    client, session = self.make_client([CannedResponse(200, json_body={"ok": True})])

    client.get_json("actions/runs", per_page=5, status="completed")

    call = session.calls[0]
    self.assertEqual(call.params, {"per_page": 5, "status": "completed"})
    self.assertTrue(call.allow_redirects)
    self.assertEqual(call.timeout, github.API_TIMEOUT_SECONDS)

  def test_sends_no_params_when_none_are_given(self) -> None:
    """An empty parameter set is sent as None, not as an empty dict."""
    client, session = self.make_client([CannedResponse(200, json_body={})])

    client.get_json("actions/runs/1")

    self.assertIsNone(session.calls[0].params)

  def test_rejects_a_json_array(self) -> None:
    """A list body is a caller mistake and points at paginate()."""
    client, _ = self.make_client([CannedResponse(200, json_body=[1, 2, 3])])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_json("actions/runs/1")

    self.assertIn("paginate()", str(caught.exception))

  def test_rejects_a_body_that_is_not_json(self) -> None:
    """An HTML error page becomes a GitHubError, not a ValueError."""
    client, _ = self.make_client([CannedResponse(200, body=b"<html>maintenance</html>")])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_json("actions/runs/1")

    self.assertIn("did not return JSON", str(caught.exception))


class PaginateTest(GitHubClientTestCase):
  """Covers the three ways paging can stop and the order items come back in."""

  def test_stops_on_a_short_page(self) -> None:
    """A page holding fewer than per_page items is the last page."""
    page = CannedResponse(200, json_body={"jobs": [{"id": 1}]})
    client, session = self.make_client([page])

    items = client.paginate("actions/runs/1/jobs", "jobs", per_page=2)

    self.assertEqual(items, [{"id": 1}])
    self.assertEqual(session.call_count, 1)

  def test_stops_on_an_empty_page(self) -> None:
    """A full page followed by an empty one stops without keeping the empty page."""
    first = CannedResponse(
        200,
        json_body={"jobs": [{"id": 1}, {"id": 2}]},
        headers={"Link": f'<{REPO_ROOT_URL}/actions/runs/1/jobs?page=2>; rel="next"'},
    )
    second = CannedResponse(200, json_body={"jobs": []})
    client, session = self.make_client([first, second])

    items = client.paginate("actions/runs/1/jobs", "jobs", per_page=2)

    self.assertEqual(items, [{"id": 1}, {"id": 2}])
    self.assertEqual(session.call_count, 2)

  def test_concatenates_pages_followed_through_the_link_header(self) -> None:
    """Items from every page come back flattened, in the order GitHub returned them."""
    base = f"{REPO_ROOT_URL}/actions/runs/1/jobs"
    first = CannedResponse(
        200,
        json_body={"jobs": [{"id": 1}, {"id": 2}]},
        headers={"Link": f'<{base}?page=2>; rel="next", <{base}?page=3>; rel="last"'},
    )
    second = CannedResponse(
        200,
        json_body={"jobs": [{"id": 3}, {"id": 4}]},
        headers={"Link": f'<{base}?page=3>; rel="next"'},
    )
    third = CannedResponse(200, json_body={"jobs": [{"id": 5}]})
    client, session = self.make_client([first, second, third])

    items = client.paginate("actions/runs/1/jobs", "jobs", per_page=2)

    self.assertEqual([item["id"] for item in items], [1, 2, 3, 4, 5])
    self.assertEqual(session.call_count, 3)
    self.assertEqual(session.calls[1].url, f"{base}?page=2")
    self.assertEqual(session.calls[2].url, f"{base}?page=3")
    # The next link already carries page and per_page; sending them again would duplicate them.
    self.assertIsNone(session.calls[1].params)
    self.assertIsNone(session.calls[2].params)

  def test_walks_the_page_parameter_when_there_is_no_link_header(self) -> None:
    """Without a Link header paging falls back to incrementing ?page."""
    first = CannedResponse(200, json_body={"artifacts": [{"id": 1}, {"id": 2}]})
    second = CannedResponse(200, json_body={"artifacts": [{"id": 3}]})
    client, session = self.make_client([first, second])

    items = client.paginate("actions/runs/1/artifacts", "artifacts", per_page=2)

    self.assertEqual([item["id"] for item in items], [1, 2, 3])
    self.assertEqual(session.call_count, 2)
    self.assertEqual(session.calls[0].params, {"per_page": 2})
    self.assertEqual(session.calls[1].params, {"per_page": 2, "page": 2})
    self.assertEqual(session.calls[0].url, session.calls[1].url)

  def test_defaults_per_page_to_one_hundred(self) -> None:
    """The collector always asks for the biggest page GitHub allows."""
    client, session = self.make_client([CannedResponse(200, json_body={"workflow_runs": []})])

    client.paginate("actions/runs", "workflow_runs", status="completed")

    self.assertEqual(session.calls[0].params, {"status": "completed", "per_page": github.PER_PAGE})

  def test_accepts_a_bare_json_array(self) -> None:
    """Endpoints that answer with a plain array need no key."""
    client, _ = self.make_client([CannedResponse(200, json_body=[{"sha": "abc"}])])

    items = client.paginate("commits", "commits", per_page=100)

    self.assertEqual(items, [{"sha": "abc"}])

  def test_raises_when_the_key_is_missing(self) -> None:
    """A body without the expected list field is reported, not silently treated as empty."""
    client, _ = self.make_client([CannedResponse(200, json_body={"total_count": 0})])

    with self.assertRaises(github.GitHubError) as caught:
      client.paginate("actions/runs/1/artifacts", "artifacts")

    self.assertIn("artifacts", str(caught.exception))

  def test_raises_when_the_key_is_not_a_list(self) -> None:
    """A scalar where a list belongs is reported with the type it found."""
    client, _ = self.make_client([CannedResponse(200, json_body={"jobs": 7})])

    with self.assertRaises(github.GitHubError) as caught:
      client.paginate("actions/runs/1/jobs", "jobs")

    self.assertIn("not a list", str(caught.exception))


class RetryTest(GitHubClientTestCase):
  """Covers which failures are worth retrying, how long they wait, and when they give up."""

  def test_max_attempts_is_three(self) -> None:
    """The retry budget the rest of this class is written against."""
    self.assertEqual(github.MAX_ATTEMPTS, 3)

  def test_a_server_error_then_a_success_returns_the_body(self) -> None:
    """One 500 is retried and the second answer is the one that counts."""
    client, session = self.make_client(
        [CannedResponse(500, body=b"upstream boom"), CannedResponse(200, json_body={"id": 7})]
    )

    payload = client.get_json("actions/runs/7")

    self.assertEqual(payload, {"id": 7})
    self.assertEqual(session.call_count, 2)
    self.assertEqual(self.sleeps, [github._backoff_seconds(1)])  # pylint: disable=protected-access
    self.assertEqual(len(self.warnings), 1)

  def test_three_server_errors_raise_github_error(self) -> None:
    """The retry budget is spent and the failure is reported, not swallowed."""
    queue = [CannedResponse(500, body=b"boom") for _ in range(github.MAX_ATTEMPTS)]
    queue.append(CannedResponse(200, json_body={"id": 7}))
    client, session = self.make_client(queue)

    with self.assertRaises(github.GitHubError) as caught:
      client.get_json("actions/runs/7")

    self.assertEqual(session.call_count, github.MAX_ATTEMPTS)
    self.assertEqual(caught.exception.status, 500)
    self.assertEqual(len(self.sleeps), github.MAX_ATTEMPTS - 1)
    self.assertEqual(self.sleeps, [2.0, 4.0])

  def test_a_not_found_is_not_retried(self) -> None:
    """404 means the thing is gone; asking twice cannot change that."""
    client, session = self.make_client([CannedResponse(404, json_body={"message": "Not Found"}), CannedResponse(200)])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_json("actions/runs/1")

    self.assertEqual(session.call_count, 1)
    self.assertEqual(caught.exception.status, 404)
    self.assertEqual(self.sleeps, [])

  def test_a_plain_forbidden_is_not_retried(self) -> None:
    """A permission 403 keeps meaning the same thing, so it fails at once."""
    forbidden = CannedResponse(403, json_body={"message": "Resource not accessible by integration"})
    client, session = self.make_client([forbidden, CannedResponse(200, json_body={})])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_json("actions/runs/1")

    self.assertEqual(session.call_count, 1)
    self.assertEqual(caught.exception.status, 403)
    self.assertEqual(self.sleeps, [])

  def test_a_rate_limited_forbidden_is_retried(self) -> None:
    """A 403 with an exhausted budget waits for the reset and tries again."""
    reset = int(time.time()) + 30
    limited = CannedResponse(
        403,
        json_body={"message": "API rate limit exceeded"},
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset)},
    )
    client, session = self.make_client([limited, CannedResponse(200, json_body={"id": 1})])

    payload = client.get_json("actions/runs/1")

    self.assertEqual(payload, {"id": 1})
    self.assertEqual(session.call_count, 2)
    self.assertEqual(len(self.sleeps), 1)
    self.assertGreater(self.sleeps[0], 25.0)
    self.assertLess(self.sleeps[0], 40.0)

  def test_a_forbidden_with_retry_after_is_retried_for_that_long(self) -> None:
    """Retry-After is obeyed as given when it is inside the backoff cap."""
    limited = CannedResponse(403, body=b"slow down", headers={"Retry-After": "7"})
    client, session = self.make_client([limited, CannedResponse(200, json_body={})])

    client.get_json("actions/runs/1")

    self.assertEqual(session.call_count, 2)
    self.assertEqual(self.sleeps, [7.0])

  def test_retry_after_is_capped(self) -> None:
    """An absurd Retry-After never parks the collector for longer than the cap."""
    limited = CannedResponse(429, body=b"slow down", headers={"Retry-After": "99999"})
    client, _ = self.make_client([limited, CannedResponse(200, json_body={})])

    client.get_json("actions/runs/1")

    self.assertEqual(self.sleeps, [github.MAX_BACKOFF_SLEEP_SECONDS])

  def test_a_forbidden_whose_body_mentions_a_rate_limit_is_retried(self) -> None:
    """The secondary rate limit arrives as a 403 with no headers at all."""
    limited = CannedResponse(403, json_body={"message": "You have exceeded a secondary rate limit"})
    client, session = self.make_client([limited, CannedResponse(200, json_body={"id": 2})])

    payload = client.get_json("actions/runs/2")

    self.assertEqual(payload, {"id": 2})
    self.assertEqual(session.call_count, 2)
    self.assertEqual(self.sleeps, [2.0])

  def test_too_many_requests_is_retried(self) -> None:
    """429 is always worth another attempt."""
    client, session = self.make_client([CannedResponse(429, body=b"too many"), CannedResponse(200, json_body={})])

    client.get_json("actions/runs/1")

    self.assertEqual(session.call_count, 2)
    self.assertEqual(self.sleeps, [2.0])

  def test_a_transport_error_is_retried(self) -> None:
    """A dropped connection is retried like a server error."""
    client, session = self.make_client(
        [requests.ConnectionError("connection reset"), CannedResponse(200, json_body={"id": 3})]
    )

    payload = client.get_json("actions/runs/3")

    self.assertEqual(payload, {"id": 3})
    self.assertEqual(session.call_count, 2)
    self.assertEqual(self.sleeps, [2.0])

  def test_repeated_transport_errors_raise_github_error(self) -> None:
    """Transport failures end as a GitHubError with no status, never as a requests error."""
    client, session = self.make_client([requests.ConnectionError("reset") for _ in range(github.MAX_ATTEMPTS)])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_json("actions/runs/1")

    self.assertEqual(session.call_count, github.MAX_ATTEMPTS)
    self.assertIsNone(caught.exception.status)

  def test_one_request_never_sleeps_longer_than_the_total_budget(self) -> None:
    """Two rate-limit waits in a row cannot park the collector past its own tick.

    Each attempt may be told to wait for a reset up to an hour away. Without a budget across
    attempts one call could sleep for two hours, which outlives the four-hourly collector run
    and the Actions job timeout.
    """
    reset = int(time.time()) + int(github.MAX_RATE_LIMIT_SLEEP_SECONDS)
    limited = [
        CannedResponse(
            403,
            json_body={"message": "API rate limit exceeded"},
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset)},
        )
        for _ in range(github.MAX_ATTEMPTS)
    ]
    client, session = self.make_client(limited)

    with self.assertRaises(github.GitHubError) as caught:
      client.get_json("actions/runs/1")

    self.assertEqual(caught.exception.status, 403)
    self.assertLessEqual(sum(self.sleeps), github.MAX_TOTAL_RETRY_SLEEP_SECONDS)
    # The first wait spends the whole budget, so there is no second one.
    self.assertEqual(len(self.sleeps), 1)
    self.assertEqual(session.call_count, 2)


class GetBytesTest(GitHubClientTestCase):
  """Covers the artifact download path, including the credential guard on redirects."""

  ARTIFACT_URL = f"{REPO_ROOT_URL}/actions/artifacts/4242/zip"
  STORAGE_URL = "https://productionresultssa0.blob.core.windows.net/actions-results/4242?sig=abc"
  ZIP_BYTES = b"PK\x03\x04 pretend this is a junit zip"

  def test_follows_the_redirect_and_returns_the_body(self) -> None:
    """The 302 hop to storage is followed by hand and its body is returned."""
    redirect = CannedResponse(302, headers={"Location": self.STORAGE_URL})
    client, session = self.make_client([redirect, CannedResponse(200, body=self.ZIP_BYTES)])

    data = client.get_bytes(self.ARTIFACT_URL)

    self.assertEqual(data, self.ZIP_BYTES)
    self.assertEqual(session.call_count, 2)
    self.assertEqual(session.calls[0].url, self.ARTIFACT_URL)
    self.assertEqual(session.calls[1].url, self.STORAGE_URL)
    for call in session.calls:
      self.assertFalse(call.allow_redirects)
      self.assertEqual(call.timeout, github.DOWNLOAD_TIMEOUT_SECONDS)

  def test_does_not_forward_the_authorization_header_to_the_redirect_target(self) -> None:
    """Credential-leak guard: the storage host must never see the GitHub token.

    GitHub answers an artifact download with a redirect to a signed storage URL. That URL
    needs no credential of ours, and the host is not GitHub, so sending the token there
    would hand it to a third party.
    """
    redirect = CannedResponse(302, headers={"Location": self.STORAGE_URL})
    client, session = self.make_client([redirect, CannedResponse(200, body=self.ZIP_BYTES)], token=TOKEN)

    client.get_bytes(self.ARTIFACT_URL)

    self.assertEqual(session.call_count, 2)
    api_hop, storage_hop = session.calls[0], session.calls[1]
    # The token belongs on api.github.com and must still be sent there.
    self.assertEqual(api_hop.sent_headers["Authorization"], f"Bearer {TOKEN}")
    # It must be gone from what the storage host would receive.
    self.assertNotIn("Authorization", storage_hop.sent_headers)
    self.assertNotIn(TOKEN, repr(dict(storage_hop.sent_headers)))
    self.assertNotIn(TOKEN, repr(storage_hop.params))
    # The None sentinel is what makes requests drop the session header on that one request.
    self.assertIsNotNone(storage_hop.headers)
    self.assertIn("Authorization", storage_hop.headers)
    self.assertIsNone(storage_hop.headers["Authorization"])
    # A truthy auth callable stops requests from substituting .netrc credentials for the new host.
    self.assertIsNotNone(storage_hop.auth)
    self.assertTrue(callable(storage_hop.auth))
    # The session itself keeps its credential for the next api.github.com call.
    self.assertEqual(session.headers["Authorization"], f"Bearer {TOKEN}")

  def test_the_credential_guard_works_against_the_installed_requests(self) -> None:
    """Proves the two mechanisms the guard rests on, in the installed requests version.

    github.py removes the token by passing a None header value, and blocks a .netrc lookup
    for the new host by passing a truthy auth callable. Both are requests behaviours rather
    than ours, so they are pinned here: if a future requests release changed either one, the
    token (or a local .netrc credential) would start reaching the storage host unnoticed.
    """
    session = requests.Session()
    session.trust_env = True
    session.headers["Authorization"] = f"Bearer {TOKEN}"
    self.addCleanup(session.close)
    no_auth = github._no_auth  # pylint: disable=protected-access

    with mock.patch.object(requests_sessions, "get_netrc_auth", return_value=("netrc-user", "netrc-password")):
      unguarded = session.prepare_request(requests.Request("GET", self.STORAGE_URL))
      guarded = session.prepare_request(
          requests.Request("GET", self.STORAGE_URL, headers={"Authorization": None}, auth=no_auth)
      )

    # Without the guard the storage host receives a credential -- here .netrc's, which even
    # replaces our Bearer token.
    self.assertTrue(unguarded.headers["Authorization"].startswith("Basic "))
    # With the guard it receives none at all.
    self.assertNotIn("Authorization", guarded.headers)

  def test_resolves_a_relative_location_against_the_response_url(self) -> None:
    """A relative Location header is joined onto the URL that answered."""
    redirect = CannedResponse(302, headers={"Location": "/redirected/zip"})
    client, session = self.make_client([redirect, CannedResponse(200, body=self.ZIP_BYTES)])

    client.get_bytes(self.ARTIFACT_URL)

    self.assertEqual(session.calls[1].url, f"{github.API_ROOT}/redirected/zip")

  def test_rejects_a_relative_url(self) -> None:
    """get_bytes takes absolute URLs only, and never sends a request for a bad one."""
    client, session = self.make_client([])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes("actions/artifacts/1/zip")

    self.assertIn("absolute", str(caught.exception))
    self.assertEqual(session.call_count, 0)

  def test_raises_when_a_redirect_has_no_location(self) -> None:
    """A redirect with nowhere to go is an error, not an empty download."""
    client, _ = self.make_client([CannedResponse(302)])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(self.ARTIFACT_URL)

    self.assertIn("no Location header", str(caught.exception))

  def test_raises_after_too_many_redirects(self) -> None:
    """A redirect loop is cut off instead of running forever."""
    loop = [CannedResponse(302, headers={"Location": self.STORAGE_URL}) for _ in range(github.MAX_REDIRECTS + 2)]
    client, session = self.make_client(loop)

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(self.ARTIFACT_URL)

    self.assertIn("still redirecting", str(caught.exception))
    self.assertEqual(session.call_count, github.MAX_REDIRECTS + 1)

  def test_raises_when_the_final_status_is_not_two_hundred(self) -> None:
    """A 204 is below 400 but is not a downloaded artifact."""
    client, _ = self.make_client([CannedResponse(204)])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(self.ARTIFACT_URL)

    self.assertIn("expected 200", str(caught.exception))
    self.assertEqual(caught.exception.status, 204)


class RateLimitTest(GitHubClientTestCase):
  """Covers reading the budget and waiting for it, without ever really sleeping."""

  def rate_limit_body(self, remaining: int, reset: int, limit: int = 5000) -> dict[str, Any]:
    """Builds a /rate_limit response body.

    Args:
      remaining: Requests left in the window.
      reset: Unix timestamp the window resets at.
      limit: Size of the window.

    Returns:
      The body as GitHub shapes it.
    """
    core = {"limit": limit, "remaining": remaining, "reset": reset}
    return {"resources": {"core": core}, "rate": dict(core)}

  def test_rate_limit_reads_resources_core(self) -> None:
    """The core resource is the budget the collector spends."""
    reset = int(time.time()) + 600
    client, session = self.make_client([CannedResponse(200, json_body=self.rate_limit_body(4321, reset))])

    status = client.rate_limit()

    self.assertEqual(status, {"limit": 5000, "remaining": 4321, "reset": reset})
    self.assertEqual(session.calls[0].url, f"{github.API_ROOT}/rate_limit")

  def test_rate_limit_falls_back_to_the_rate_field(self) -> None:
    """Older shaped answers carry the same numbers under `rate`."""
    reset = int(time.time()) + 600
    body = {"rate": {"limit": 60, "remaining": 12, "reset": reset}}
    client, _ = self.make_client([CannedResponse(200, json_body=body)])

    status = client.rate_limit()

    self.assertEqual(status, {"limit": 60, "remaining": 12, "reset": reset})

  def test_rate_limit_raises_when_the_fields_are_missing(self) -> None:
    """A body with neither shape is reported rather than guessed at."""
    client, _ = self.make_client([CannedResponse(200, json_body={"message": "nope"})])

    with self.assertRaises(github.GitHubError) as caught:
      client.rate_limit()

    self.assertIn("neither resources.core nor rate", str(caught.exception))

  def test_wait_returns_without_a_request_when_the_budget_is_known_good(self) -> None:
    """Rate-limit headers from the last call spare us the /rate_limit round trip."""
    answer = CannedResponse(200, json_body={"id": 1}, headers={"X-RateLimit-Remaining": "500"})
    client, session = self.make_client([answer])

    client.get_json("actions/runs/1")
    client.wait_for_rate_limit(need=50)

    self.assertEqual(session.call_count, 1)
    self.assertEqual(self.sleeps, [])

  def test_wait_sleeps_when_the_budget_is_short(self) -> None:
    """A short budget waits for the window to reset, and the wait is a patched seam."""
    reset = int(time.time()) + 60
    first = CannedResponse(200, json_body={"id": 1}, headers={"X-RateLimit-Remaining": "2"})
    client, session = self.make_client([first, CannedResponse(200, json_body=self.rate_limit_body(2, reset))])

    client.get_json("actions/runs/1")
    started = time.monotonic()
    client.wait_for_rate_limit(need=50)
    elapsed = time.monotonic() - started

    self.assertEqual(session.call_count, 2)
    self.assertEqual(session.calls[1].url, f"{github.API_ROOT}/rate_limit")
    self.assertEqual(len(self.sleeps), 1)
    self.assertGreater(self.sleeps[0], 55.0)
    self.assertLess(self.sleeps[0], 70.0)
    self.assertLess(elapsed, 1.0, "wait_for_rate_limit must not really sleep in tests")
    self.assertTrue(any("Rate limit low" in line for line in self.warnings))

  def test_wait_does_not_sleep_when_the_fresh_reading_is_fine(self) -> None:
    """A stale low hint is corrected by the fresh read and costs no wait."""
    reset = int(time.time()) + 60
    first = CannedResponse(200, json_body={"id": 1}, headers={"X-RateLimit-Remaining": "2"})
    client, session = self.make_client([first, CannedResponse(200, json_body=self.rate_limit_body(4000, reset))])

    client.get_json("actions/runs/1")
    client.wait_for_rate_limit(need=50)

    self.assertEqual(session.call_count, 2)
    self.assertEqual(self.sleeps, [])

  def test_wait_is_capped_at_one_hour(self) -> None:
    """A reset far in the future never parks the collector for longer than the cap."""
    reset = int(time.time()) + 10 * 3600
    client, _ = self.make_client([CannedResponse(200, json_body=self.rate_limit_body(0, reset))])

    client.wait_for_rate_limit(need=50)

    self.assertEqual(self.sleeps, [github.MAX_RATE_LIMIT_SLEEP_SECONDS])


if __name__ == "__main__":
  unittest.main()
