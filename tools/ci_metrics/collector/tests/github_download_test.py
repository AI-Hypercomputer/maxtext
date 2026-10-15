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

"""Offline tests for the capped download in `GitHubClient.get_bytes`, and two smaller guards.

An artifact download is the one place where the collector reads a body that someone else
controls: a zip from any pull request, including a fork. The size that GitHub lists for it is
only a note, and a server can leave out the Content-Length header or write a wrong one. So
`get_bytes(url, max_bytes=N)` reads the body a piece at a time, counts the bytes that really
arrive, and stops with a `GitHubError` as soon as there are more than N. It also refuses at once
when Content-Length already says the body is too big. These tests check that, with a fake
session that streams a body in scripted pieces and never opens a socket.

They also check that streaming did not loosen the rules that were already there: the token goes
only to api.github.com and never to the storage host, a body that breaks off half way is
downloaded again, a server error is still retried, signed URLs stay out of messages, and every
response is closed.

Two smaller guards live here too. `paginate(..., max_items=N)` stops listing after the page that
reaches N items, so a run with thousands of artifacts is not listed to the end; without it,
paging works as it always did. And the text of a connection error, which can quote the whole
request path with its signed query, loses the query before it goes into a message.

Needs `requests`, like the client itself. Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/github_download_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/github_download_test.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from collections.abc import Iterable, Iterator
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

TOKEN = "ghp-test-token-never-real"
ARTIFACT_URL = f"{github.API_ROOT}/repos/example/example/actions/artifacts/4242/zip"
STORAGE_URL = "https://storage.example.net/actions-results/4242?sig=SECRET-SIGNATURE"
STORAGE_URL_WITHOUT_QUERY = "https://storage.example.net/actions-results/4242"

# The limit used by most tests, and the size of one piece of a streamed body.
LIMIT = 1000
PIECE = 100


def pieces_of(total: int, size: int = PIECE) -> list[bytes]:
  """Cuts a body of `total` bytes into pieces, the way a streamed download hands it out.

  Args:
    total: The number of bytes in the whole body.
    size: The size of each piece. The last one holds what is left.

  Returns:
    The pieces. Their lengths add up to `total`.
  """
  pieces = [b"x" * size for _ in range(total // size)]
  if total % size:
    pieces.append(b"x" * (total % size))
  return pieces


class StreamedResponse(requests.Response):
  """A response whose body arrives in scripted pieces, the way a streamed download does.

  The pieces are a finite list on purpose: a client that forgot to stop would then read to the
  end and fail the test, instead of hanging it.

  Attributes:
    pieces_served: How many pieces have been handed out.
    bytes_served: How many bytes those pieces held.
    closed: True once `close` has been called.
  """

  def __init__(
      self,
      status: int = 200,
      pieces: Iterable[bytes | Exception] = (),
      headers: dict[str, str] | None = None,
  ) -> None:
    """Builds the response.

    Args:
      status: HTTP status code.
      pieces: The body, piece by piece. An exception in the list is raised when the client
        gets to it, to model a connection that breaks off half way.
      headers: Response headers, for example Content-Length or Location.
    """
    super().__init__()
    self.status_code = status
    self.encoding = "utf-8"
    self._pieces = list(pieces)
    self.pieces_served = 0
    self.bytes_served = 0
    self.closed = False
    if headers:
      self.headers.update(headers)

  def iter_content(self, chunk_size: int | None = 1, decode_unicode: bool = False) -> Iterator[bytes]:
    """Hands out the scripted pieces one at a time and counts them.

    Args:
      chunk_size: Ignored; the pieces are already cut.
      decode_unicode: Ignored.

    Yields:
      The next piece of the body.

    Raises:
      Exception: Whatever exception was scripted in place of a piece.
    """
    del chunk_size, decode_unicode
    for piece in self._pieces:
      if isinstance(piece, Exception):
        raise piece
      self.pieces_served += 1
      self.bytes_served += len(piece)
      yield piece

  def close(self) -> None:
    """Remembers that the client let go of the connection."""
    self.closed = True


class JsonPage(requests.Response):
  """A 200 response that holds one page of a list endpoint, with a Link header when more follow."""

  def __init__(self, items: list[Any], next_page: int | None = None) -> None:
    """Builds the page.

    Args:
      items: The artifacts on this page.
      next_page: The number of the next page, or None for the last one.
    """
    super().__init__()
    self.status_code = 200
    self.encoding = "utf-8"
    self._content = json.dumps({"artifacts": items}).encode("utf-8")
    self._content_consumed = True
    if next_page is not None:
      next_url = f"{github.API_ROOT}/repos/example/example/actions/runs/1/artifacts?per_page=100&page={next_page}"
      self.headers["Link"] = f'<{next_url}>; rel="next"'


class FakeSession:
  """Offline stand-in for `requests.Session` that answers from a scripted queue."""

  def __init__(self, answers: list[Any]) -> None:
    """Builds the fake.

    Args:
      answers: Responses to hand back in order. An exception instance in the list is raised
        instead of being returned.
    """
    self.headers = CaseInsensitiveDict()
    self.queue = list(answers)
    self.calls: list[dict[str, Any]] = []

  def request(
      self,
      method: str,
      url: str,
      params: dict[str, Any] | None = None,
      headers: dict[str, Any] | None = None,
      auth: Any = None,
      allow_redirects: bool = True,
      timeout: Any = None,
      stream: bool = False,
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
      stream: Whether the body is to be read lazily.

    Returns:
      The next response in the queue.

    Raises:
      AssertionError: When the client sends more requests than the test scripted.
      Exception: Whatever exception instance the test queued.
    """
    self.calls.append(
        {
            "method": method,
            "url": url,
            "params": params,
            "headers": headers,
            "auth": auth,
            "allow_redirects": allow_redirects,
            "timeout": timeout,
            "stream": stream,
            "session_headers": dict(self.headers),
        }
    )
    if not self.queue:
      raise AssertionError(f"the fake session ran out of scripted answers at {method} {url}")
    answer = self.queue.pop(0)
    if isinstance(answer, Exception):
      raise answer
    if not answer.url:
      answer.url = url
    return answer


def redirect() -> StreamedResponse:
  """Builds the 302 that GitHub answers an artifact download with.

  Returns:
    A redirect to the storage URL, whose query carries a signature.
  """
  return StreamedResponse(302, headers={"Location": STORAGE_URL})


def broken_body() -> StreamedResponse:
  """Builds a 200 whose body breaks off after two pieces.

  Returns:
    The response. Reading its third piece raises `ChunkedEncodingError`.
  """
  return StreamedResponse(
      200, [b"a" * PIECE, b"a" * PIECE, requests.exceptions.ChunkedEncodingError("Connection broken")]
  )


class ClientTestCase(unittest.TestCase):
  """Base class: a client on a fake session, with no sleeping, no stderr noise and no ambient token."""

  def setUp(self) -> None:
    """Keeps the tests offline and instant: no sleeping, no warnings on stderr, no ambient token."""
    super().setUp()
    env_patch = mock.patch.dict(os.environ, {}, clear=False)
    env_patch.start()
    self.addCleanup(env_patch.stop)
    os.environ.pop("GITHUB_TOKEN", None)

    self.sleeps: list[float] = []
    self.warnings: list[str] = []
    for name, record in (("_sleep", self.sleeps.append), ("_warn", self.warnings.append)):
      patch = mock.patch.object(github, name, record)
      patch.start()
      self.addCleanup(patch.stop)

  def make_client(self, answers: list[Any]) -> tuple[github.GitHubClient, FakeSession]:
    """Builds a client with a token, on a fake session that answers from `answers`.

    Args:
      answers: The scripted responses, in order.

    Returns:
      The client and its session.
    """
    session = FakeSession(answers)
    return github.GitHubClient("example", "example", token=TOKEN, session=session), session


class CappedDownloadTest(ClientTestCase):
  """`get_bytes(url, max_bytes=N)` counts the bytes that arrive and stops at N."""

  def assert_stopped_early(self, body: StreamedResponse) -> None:
    """Checks that the client stopped reading soon after the limit, and let go of the response.

    Args:
      body: The response that streamed too much.
    """
    self.assertLessEqual(body.bytes_served, LIMIT + PIECE, "the client kept reading long after the limit")
    self.assertLess(body.pieces_served, len(pieces_of(10_000)), "the client read the whole body")
    self.assertTrue(body.closed, "the response was not closed")

  def test_a_body_with_no_content_length_is_stopped_when_it_passes_the_limit(self) -> None:
    body = StreamedResponse(200, pieces_of(10_000))
    client, _ = self.make_client([redirect(), body])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assert_stopped_early(body)
    message = str(caught.exception)
    self.assertIn("1,000 bytes", message)
    self.assertNotIn("SECRET-SIGNATURE", message, "a signed URL must never reach a message")
    self.assertEqual(caught.exception.url, STORAGE_URL_WITHOUT_QUERY)

  def test_a_body_is_stopped_even_when_content_length_lies(self) -> None:
    body = StreamedResponse(200, pieces_of(10_000), headers={"Content-Length": "50"})
    client, _ = self.make_client([redirect(), body])

    with self.assertRaises(github.GitHubError):
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assert_stopped_early(body)

  def test_a_content_length_over_the_limit_is_refused_before_anything_is_read(self) -> None:
    body = StreamedResponse(200, pieces_of(10_000), headers={"Content-Length": "5000"})
    client, _ = self.make_client([redirect(), body])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assertEqual(body.pieces_served, 0, "the body must not be read when the header already says it is too big")
    self.assertTrue(body.closed)
    self.assertIn("5,000 bytes", str(caught.exception))
    self.assertNotIn("SECRET-SIGNATURE", str(caught.exception))

  def test_a_body_exactly_at_the_limit_is_returned(self) -> None:
    body = StreamedResponse(200, pieces_of(LIMIT), headers={"Content-Length": str(LIMIT)})
    client, _ = self.make_client([redirect(), body])

    data = client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assertEqual(len(data), LIMIT)
    self.assertTrue(body.closed)

  def test_a_body_one_byte_over_the_limit_is_refused(self) -> None:
    body = StreamedResponse(200, pieces_of(LIMIT + 1))
    client, _ = self.make_client([redirect(), body])

    with self.assertRaises(github.GitHubError):
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assertTrue(body.closed)

  def test_without_a_limit_the_whole_body_is_returned_as_before(self) -> None:
    body = StreamedResponse(200, pieces_of(10_000), headers={"Content-Length": "10000"})
    client, _ = self.make_client([redirect(), body])

    data = client.get_bytes(ARTIFACT_URL)

    self.assertEqual(len(data), 10_000)

  def test_both_hops_are_streamed_and_the_token_stays_on_api_github_com(self) -> None:
    """Streaming must not loosen the credential rules the README promises."""
    first, last = redirect(), StreamedResponse(200, [b"zip bytes"])
    client, session = self.make_client([first, last])

    data = client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assertEqual(data, b"zip bytes")
    api_hop, storage_hop = session.calls[0], session.calls[1]
    self.assertEqual(api_hop["url"], ARTIFACT_URL)
    self.assertEqual(storage_hop["url"], STORAGE_URL)
    for call in session.calls:
      self.assertIs(
          call["stream"], True, "a download must be streamed, or the body is read whole before it can be counted"
      )
      self.assertFalse(call["allow_redirects"])
      self.assertEqual(call["timeout"], github.DOWNLOAD_TIMEOUT_SECONDS)
    # The token still goes to api.github.com ...
    self.assertEqual(api_hop["session_headers"]["Authorization"], f"Bearer {TOKEN}")
    self.assertIsNone(api_hop["headers"])
    # ... and is stripped from the storage hop, with .netrc lookups switched off too.
    self.assertEqual(storage_hop["headers"], {"Authorization": None})
    self.assertTrue(callable(storage_hop["auth"]))
    # Both responses were let go.
    self.assertTrue(first.closed)
    self.assertTrue(last.closed)

  def test_a_body_that_breaks_off_is_downloaded_again(self) -> None:
    """The body is read after `_request` returns, so the retry has to happen around the download."""
    broken, good = broken_body(), StreamedResponse(200, [b"b" * PIECE, b"b" * PIECE])
    client, session = self.make_client([redirect(), broken, redirect(), good])

    data = client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assertEqual(data, b"b" * 2 * PIECE)
    self.assertEqual(session.calls[0]["url"], ARTIFACT_URL)
    self.assertEqual(session.calls[2]["url"], ARTIFACT_URL, "the second try starts again from the API URL")
    self.assertEqual(self.sleeps, [github.BACKOFF_BASE_SECONDS])
    self.assertTrue(broken.closed)

  def test_it_gives_up_after_the_last_attempt(self) -> None:
    answers = []
    for _ in range(github.MAX_ATTEMPTS):
      answers += [redirect(), broken_body()]
    client, session = self.make_client(answers)

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assertIn(f"{github.MAX_ATTEMPTS} attempt(s)", str(caught.exception))
    self.assertNotIn("SECRET-SIGNATURE", str(caught.exception))
    self.assertEqual(len(session.calls), 2 * github.MAX_ATTEMPTS)
    self.assertEqual(self.sleeps, [github.BACKOFF_BASE_SECONDS, 2 * github.BACKOFF_BASE_SECONDS])

  def test_a_server_error_on_the_storage_hop_is_still_retried(self) -> None:
    failed, good = StreamedResponse(503), StreamedResponse(200, [b"zip bytes"])
    client, session = self.make_client([redirect(), failed, good])

    data = client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assertEqual(data, b"zip bytes")
    self.assertEqual(len(session.calls), 3)
    self.assertEqual(self.sleeps, [github.BACKOFF_BASE_SECONDS])
    self.assertTrue(failed.closed, "a response that is about to be retried must be closed")
    self.assertTrue(good.closed)

  def test_a_final_status_that_is_not_two_hundred_is_still_an_error(self) -> None:
    odd = StreamedResponse(204)
    client, _ = self.make_client([odd])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assertIn("expected 200", str(caught.exception))
    self.assertEqual(caught.exception.status, 204)
    self.assertTrue(odd.closed)


class CappedListingTest(ClientTestCase):
  """`paginate(..., max_items=N)` stops after the page that reaches N items."""

  PATH = "actions/runs/1/artifacts"

  def pages(self, count: int) -> list[JsonPage]:
    """Builds full pages of 100 items, each pointing at the next except the last.

    Args:
      count: How many pages.

    Returns:
      The pages, in order.
    """
    return [
        JsonPage([{"id": page * 100 + n} for n in range(100)], next_page=page + 2 if page + 1 < count else None)
        for page in range(count)
    ]

  def test_it_stops_after_the_page_that_reaches_the_limit(self) -> None:
    client, session = self.make_client(self.pages(5))

    items = client.paginate(self.PATH, "artifacts", per_page=100, max_items=201)

    self.assertEqual(len(items), 300, "whole pages are returned, so a little more than the limit")
    self.assertEqual(len(session.calls), 3, "pages 4 and 5 must never be asked for")

  def test_a_first_page_that_already_reaches_the_limit_is_the_only_one_asked_for(self) -> None:
    client, session = self.make_client(self.pages(5))

    items = client.paginate(self.PATH, "artifacts", per_page=100, max_items=100)

    self.assertEqual(len(items), 100)
    self.assertEqual(len(session.calls), 1)

  def test_fewer_items_than_the_limit_are_all_returned(self) -> None:
    client, session = self.make_client([JsonPage([{"id": n} for n in range(7)])])

    items = client.paginate(self.PATH, "artifacts", per_page=100, max_items=201)

    self.assertEqual(len(items), 7)
    self.assertEqual(len(session.calls), 1)

  def test_without_a_limit_every_page_is_read_as_before(self) -> None:
    client, session = self.make_client(self.pages(5))

    items = client.paginate(self.PATH, "artifacts", per_page=100)

    self.assertEqual(len(items), 500)
    self.assertEqual(len(session.calls), 5)

  def test_the_limit_is_not_sent_to_github_as_a_query_parameter(self) -> None:
    client, session = self.make_client(self.pages(2))

    client.paginate(self.PATH, "artifacts", per_page=100, max_items=100)

    self.assertEqual(session.calls[0]["params"], {"per_page": 100})


# What urllib3 puts into a connection error: the whole request path, and on the storage host that
# path carries the signature.
LEAKY_ERROR = (
    "HTTPSConnectionPool(host='storage.example.net', port=443): Max retries exceeded with url: "
    "/actions-results/4242?sig=SECRET-SIGNATURE&se=2026-10-14 (Caused by NewConnectionError("
    "'<urllib3.connection.HTTPSConnection object at 0x1>: Failed to establish a new connection'))"
)
QUOTED_LEAKY_ERROR = "could not reach 'https://storage.example.net/actions-results/4242?sig=SECRET-SIGNATURE'"
# A query can hold characters that would end a cut made only up to the next quote or bracket. Here a
# quote, a closing parenthesis and two other brackets all come before the signature.
TRICKY_ERROR = (
    "HTTPSConnectionPool(host='storage.example.net', port=443): Max retries exceeded with url: "
    "/actions-results/4242?se=2026-10-14&sp=r'x)y]z>w&sig=SECRET-SIGNATURE (Caused by NewConnectionError("
    "'<urllib3.connection.HTTPSConnection object at 0x1>: Failed to establish a new connection'))"
)


class SignedUrlTest(ClientTestCase):
  """The text of an error never carries a signed query, in a message or in a warning."""

  def assert_clean(self, error: github.GitHubError, url: str) -> None:
    """Checks the error and every warning for the signature, and that the rest of the text stayed.

    Args:
      error: The `GitHubError` the download ended with.
      url: The URL the error should carry, without a query.
    """
    texts = [str(error), *self.warnings]
    self.assertTrue(self.warnings, "the failures before the last one should have been warned about")
    for text in texts:
      self.assertNotIn("SECRET-SIGNATURE", text, "a signed query must never reach a message")
      self.assertNotIn("sig=", text, "not even the name of the parameter is left behind")
    self.assertIn("/actions-results/4242", str(error), "only the query is cut, the path stays")
    self.assertEqual(error.url, url)

  def test_a_connection_error_on_the_storage_hop(self) -> None:
    errors = [requests.exceptions.ConnectionError(LEAKY_ERROR) for _ in range(github.MAX_ATTEMPTS)]
    client, _ = self.make_client([redirect(), *errors])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assert_clean(caught.exception, STORAGE_URL_WITHOUT_QUERY)
    self.assertIn("Max retries exceeded", str(caught.exception))

  def test_a_query_with_a_quote_or_a_bracket_in_it_is_cut_whole(self) -> None:
    """The cut runs to the next space, so a quote or a bracket inside the query does not leave the signature."""
    errors = [requests.exceptions.ConnectionError(TRICKY_ERROR) for _ in range(github.MAX_ATTEMPTS)]
    client, _ = self.make_client([redirect(), *errors])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assert_clean(caught.exception, STORAGE_URL_WITHOUT_QUERY)
    self.assertIn("Max retries exceeded", str(caught.exception))
    self.assertIn("(Caused by NewConnectionError", str(caught.exception), "the text after the query stays")

  def test_a_url_in_quotes_loses_its_query_too(self) -> None:
    errors = [requests.exceptions.ConnectionError(QUOTED_LEAKY_ERROR) for _ in range(github.MAX_ATTEMPTS)]
    client, _ = self.make_client([redirect(), *errors])

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assert_clean(caught.exception, STORAGE_URL_WITHOUT_QUERY)

  def test_an_error_while_the_body_is_read(self) -> None:
    answers = []
    for _ in range(github.MAX_ATTEMPTS):
      answers += [redirect(), StreamedResponse(200, [b"a" * PIECE, requests.exceptions.ConnectionError(LEAKY_ERROR)])]
    client, _ = self.make_client(answers)

    with self.assertRaises(github.GitHubError) as caught:
      client.get_bytes(ARTIFACT_URL, max_bytes=LIMIT)

    self.assert_clean(caught.exception, ARTIFACT_URL)


if __name__ == "__main__":
  unittest.main(verbosity=2)
