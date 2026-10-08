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

"""Read-only GitHub REST API client for the CI metrics collector.

Every value the dashboard shows comes from the GitHub API, JUnit XML and arithmetic, so
this module is the only place that talks to the network. It wraps `requests` with the
three things the collector needs and nothing else:

  * authentication from the GITHUB_TOKEN environment variable,
  * pagination that stops correctly on a short page, an empty page or a missing next link,
  * retries with exponential backoff on server errors and rate-limit rejections.

All requests are GET. The client never writes to GitHub and never touches the gh-pages
branch.

The Authorization header is attached by host, not by call site: every request whose host is
not api.github.com is sent with no credential at all, whatever URL the caller passed. That
covers the artifact download, which GitHub answers with a redirect to a signed storage URL;
that hop is followed by hand so requests cannot re-apply a .netrc credential for the new
host either. See `GitHubClient.get_bytes` and `GitHubClient._request`.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any
from urllib.parse import urlsplit

import requests

API_ROOT = "https://api.github.com"
# Only this host is ever sent the token. Everything else is requested anonymously.
API_HOST = urlsplit(API_ROOT).netloc
ACCEPT_HEADER = "application/vnd.github+json"
API_VERSION_HEADER = "2022-11-28"
USER_AGENT = "maxtext-ci-metrics-collector"

PER_PAGE = 100
MAX_PAGES = 1000
MAX_ATTEMPTS = 3
MAX_REDIRECTS = 5
REDIRECT_STATUSES = (301, 302, 303, 307, 308)

BACKOFF_BASE_SECONDS = 2.0
MAX_BACKOFF_SLEEP_SECONDS = 120.0
MAX_RATE_LIMIT_SLEEP_SECONDS = 3600.0
# Total time one request may spend sleeping across all its retries. Without this a rate
# limited call could wait out two full reset windows and outlive the collector's own tick.
MAX_TOTAL_RETRY_SLEEP_SECONDS = 3600.0
RATE_LIMIT_BUFFER_SECONDS = 5.0

API_TIMEOUT_SECONDS = (10.0, 30.0)
DOWNLOAD_TIMEOUT_SECONDS = (10.0, 120.0)

NO_TOKEN_WARNING = (
    "WARNING: no GitHub token found (pass token= or set GITHUB_TOKEN). Unauthenticated "
    "requests are capped at 60 per hour, which is not enough to collect a single run."
)

# Set the first time a client is built without a token, so the warning is printed once per
# process. Tests may reset it to False.
_TOKEN_WARNING_EMITTED = False


class GitHubError(RuntimeError):
  """Raised when a GitHub request cannot be completed or its answer cannot be used.

  Attributes:
    status: HTTP status code of the failing response, or None for transport failures.
    url: URL that was being requested, when it is known.
  """

  def __init__(self, message: str, status: int | None = None, url: str | None = None) -> None:
    """Builds the error.

    Args:
      message: Human-readable description of what failed.
      status: HTTP status code of the failing response, if there was one.
      url: URL that was being requested, if it is known.
    """
    super().__init__(message)
    self.status = status
    self.url = url


def _sleep(seconds: float) -> None:
  """Sleeps for the given number of seconds (a seam tests can patch out).

  Args:
    seconds: How long to sleep. Values at or below zero return immediately.
  """
  if seconds > 0:
    time.sleep(seconds)


def _warn(message: str) -> None:
  """Prints a warning to stderr so it never lands in piped collector output.

  Args:
    message: The line to print.
  """
  print(message, file=sys.stderr, flush=True)


def _no_auth(request: Any) -> Any:
  """Auth callable that adds nothing, used to stop requests from applying .netrc credentials.

  `requests.Session.prepare_request` looks up .netrc whenever the request carries no auth
  and `trust_env` is on. Passing a truthy callable suppresses that lookup without adding a
  credential of our own.

  Args:
    request: The prepared request, returned unchanged.

  Returns:
    The same request object.
  """
  return request


def _is_api_url(url: str) -> bool:
  """Returns True when a URL points at api.github.com, the only host that may see the token.

  Args:
    url: Absolute URL to check.

  Returns:
    True when the host is exactly api.github.com.
  """
  return urlsplit(url).netloc.lower() == API_HOST


def _safe_url(url: str) -> str:
  """Returns a URL without its query string, for messages that may be logged.

  Artifact downloads redirect to a storage URL whose query carries a signed access token.
  Printing it would leak a credential into collector logs, so the query is cut off.

  Args:
    url: The URL to quote.

  Returns:
    The URL up to the first "?".
  """
  return url.split("?", 1)[0]


def _backoff_seconds(attempt: int) -> float:
  """Returns the exponential backoff delay for a failed attempt.

  Args:
    attempt: 1-based attempt number that just failed.

  Returns:
    Seconds to wait before the next attempt, capped at MAX_BACKOFF_SLEEP_SECONDS.
  """
  return min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SLEEP_SECONDS)


def _seconds_until(reset_epoch: int) -> float:
  """Returns how long to wait for a rate-limit window to reset.

  Args:
    reset_epoch: Unix timestamp the limit resets at, as GitHub reports it.

  Returns:
    Seconds to wait, never negative and never more than MAX_RATE_LIMIT_SLEEP_SECONDS. A
    small buffer is added because the client and GitHub clocks are not the same clock.
  """
  remaining = reset_epoch - time.time() + RATE_LIMIT_BUFFER_SECONDS
  return min(max(remaining, 0.0), MAX_RATE_LIMIT_SLEEP_SECONDS)


def _header_int(response: requests.Response, name: str) -> int | None:
  """Reads one response header as an integer.

  Args:
    response: Response to read the header from.
    name: Header name, matched case-insensitively.

  Returns:
    The integer value, or None when the header is absent or not a number.
  """
  raw = response.headers.get(name)
  if raw is None:
    return None
  try:
    return int(raw.strip())
  except (AttributeError, ValueError):
    return None


def _body_excerpt(response: requests.Response, limit: int = 200) -> str:
  """Returns a short, single-line excerpt of a response body for error messages.

  Args:
    response: Response whose body should be quoted.
    limit: Maximum number of characters to keep.

  Returns:
    The trimmed body text, or an empty string when the body cannot be read.
  """
  try:
    text = " ".join(response.text.split())
  except (UnicodeDecodeError, ValueError):
    return ""
  if len(text) > limit:
    return text[:limit] + "..."
  return text


class GitHubClient:
  """Read-only client for one repository's GitHub REST API.

  The client keeps a `requests.Session` so connections are reused, remembers the newest
  rate-limit headers it saw, and turns every unrecoverable failure into a GitHubError.
  """

  def __init__(
      self,
      owner: str,
      repo: str,
      token: str | None = None,
      session: requests.Session | None = None,
  ) -> None:
    """Builds a client for `owner/repo`.

    Args:
      owner: Repository owner, for example "AI-Hypercomputer".
      repo: Repository name, for example "maxtext".
      token: GitHub token. Falls back to the GITHUB_TOKEN environment variable. Without a
        token the API allows only 60 requests per hour, so a warning is printed once.
      session: Session to send requests through. A new one is created when omitted; pass
        your own to share connections or to stub the network in tests.
    """
    global _TOKEN_WARNING_EMITTED

    self.owner = owner
    self.repo = repo
    self.token = token or os.environ.get("GITHUB_TOKEN") or None
    self.session = session if session is not None else requests.Session()
    self._owns_session = session is None
    self._remaining_hint: int | None = None
    self._reset_hint: int | None = None

    self.session.headers.update(
        {
            "Accept": ACCEPT_HEADER,
            "X-GitHub-Api-Version": API_VERSION_HEADER,
            "User-Agent": USER_AGENT,
        }
    )
    if self.token:
      self.session.headers["Authorization"] = f"Bearer {self.token}"
    else:
      # A borrowed session may already carry another client's token. A client built without
      # one must not send it, or `client.token is None` would be a lie about the wire.
      self.session.headers.pop("Authorization", None)
      if not _TOKEN_WARNING_EMITTED:
        _TOKEN_WARNING_EMITTED = True
        _warn(NO_TOKEN_WARNING)

  def close(self) -> None:
    """Closes the session, but only when this client created it."""
    if self._owns_session:
      self.session.close()

  def get_json(self, path: str, **params: Any) -> dict[str, Any]:
    """Fetches one JSON object from a repository-relative API path.

    Args:
      path: Path under /repos/{owner}/{repo}, for example "actions/runs/33468578834". An
        absolute https URL is also accepted and used as given.
      **params: Query-string parameters.

    Returns:
      The decoded JSON object.

    Raises:
      GitHubError: On a failed request, a body that is not JSON, or a body that is a JSON
        array rather than an object (use `paginate` for list endpoints).
    """
    response = self._request("GET", self._url(path), params=params or None)
    payload = self._decode_json(response)
    if not isinstance(payload, dict):
      raise GitHubError(
          f"GET {response.url} returned a JSON {type(payload).__name__}, not an object; "
          "use paginate() for endpoints that answer with a list.",
          status=response.status_code,
          url=response.url,
      )
    return payload

  def paginate(self, path: str, key: str, **params: Any) -> list[Any]:
    """Fetches every page of a list endpoint and returns the items flattened.

    Paging follows the Link header when GitHub sends one. Without it, paging stops on a
    short page (fewer than per_page items) or on an empty page, whichever comes first.

    Args:
      path: Path under /repos/{owner}/{repo}, for example "actions/runs/1/artifacts".
      key: Field of the response object holding the list, for example "jobs", "artifacts"
        or "workflow_runs". Ignored when the endpoint answers with a bare JSON array.
      **params: Query-string parameters. `per_page` defaults to 100.

    Returns:
      Every item from every page, in the order GitHub returned them.

    Raises:
      GitHubError: On a failed request, a body that is not JSON, a body that has no `key`
        field, or a page count beyond MAX_PAGES (a paging loop that never ends).
    """
    query: dict[str, Any] | None = dict(params)
    query.setdefault("per_page", PER_PAGE)
    per_page = int(query["per_page"])

    url = self._url(path)
    items: list[Any] = []
    for _ in range(MAX_PAGES):
      response = self._request("GET", url, params=query)
      payload = self._decode_json(response)
      if isinstance(payload, list):
        page_items = payload
      elif isinstance(payload, dict):
        page_items = payload.get(key)
        if page_items is None:
          raise GitHubError(
              f"GET {response.url} has no '{key}' field; it holds {sorted(payload)[:8]}.",
              status=response.status_code,
              url=response.url,
          )
        if not isinstance(page_items, list):
          raise GitHubError(
              f"GET {response.url} field '{key}' is a {type(page_items).__name__}, not a list.",
              status=response.status_code,
              url=response.url,
          )
      else:
        raise GitHubError(
            f"GET {response.url} returned a JSON {type(payload).__name__}, not an object or list.",
            status=response.status_code,
            url=response.url,
        )

      if not page_items:
        return items
      items.extend(page_items)

      next_url = response.links.get("next", {}).get("url")
      if next_url:
        # The link already carries page and per_page; sending them again would duplicate them.
        url = next_url
        query = None
        continue
      if len(page_items) < per_page:
        return items
      if query is None:
        # GitHub stopped sending a next link but the page was full: nothing left to follow.
        return items
      query = dict(query)
      query["page"] = int(query.get("page", 1)) + 1

    raise GitHubError(f"Stopped paging {self._url(path)} after {MAX_PAGES} pages; the next links loop.")

  def get_bytes(self, url: str) -> bytes:
    """Downloads an absolute URL and returns its bytes, following GitHub's redirect.

    Artifact downloads answer 302 with a Location on a storage host. That hop is followed
    by hand, with the Authorization header removed and .netrc lookups suppressed, so the
    token never leaves api.github.com. Letting requests follow the redirect would strip the
    header too, but it would also re-apply any .netrc credentials for the new host.

    Storage URLs carry a signed access token in their query string, so every message this
    method raises quotes the URL without its query.

    Args:
      url: Absolute http(s) URL, for example an artifact's archive_download_url.

    Returns:
      The response body as bytes (for artifacts, the zip file).

    Raises:
      GitHubError: On a non-absolute URL, a failed request, a redirect without a Location,
        more than MAX_REDIRECTS hops, or a final status that is not 200.
    """
    if not url.lower().startswith(("http://", "https://")):
      raise GitHubError(f"get_bytes needs an absolute http(s) URL, got {url!r}.", url=url)
    safe = _safe_url(url)

    response = self._request("GET", url, allow_redirects=False, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    hops = 0
    while response.status_code in REDIRECT_STATUSES:
      if hops >= MAX_REDIRECTS:
        raise GitHubError(f"GET {safe} still redirecting after {MAX_REDIRECTS} hops.", url=safe)
      location = response.headers.get("Location")
      if not location:
        raise GitHubError(
            f"GET {_safe_url(response.url)} answered {response.status_code} with no Location header.",
            status=response.status_code,
            url=_safe_url(response.url),
        )
      target = requests.compat.urljoin(response.url, location)
      response = self._request(
          "GET",
          target,
          allow_redirects=False,
          timeout=DOWNLOAD_TIMEOUT_SECONDS,
          send_auth=False,
      )
      hops += 1

    if response.status_code != 200:
      raise GitHubError(
          f"GET {_safe_url(response.url)} answered {response.status_code}, expected 200.",
          status=response.status_code,
          url=_safe_url(response.url),
      )
    return response.content

  def rate_limit(self) -> dict[str, int]:
    """Reads the core rate-limit budget. This endpoint does not spend budget itself.

    Returns:
      A dict with "limit", "remaining" and "reset" (reset is a Unix timestamp).

    Raises:
      GitHubError: On a failed request or an answer without the expected fields.
    """
    response = self._request("GET", f"{API_ROOT}/rate_limit")
    payload = self._decode_json(response)
    if not isinstance(payload, dict):
      raise GitHubError(f"GET {response.url} returned a JSON {type(payload).__name__}, not an object.")

    core = payload.get("resources", {}).get("core") if isinstance(payload.get("resources"), dict) else None
    if not isinstance(core, dict):
      core = payload.get("rate")
    if not isinstance(core, dict):
      raise GitHubError(f"GET {response.url} has neither resources.core nor rate.", url=response.url)

    try:
      status = {
          "limit": int(core["limit"]),
          "remaining": int(core["remaining"]),
          "reset": int(core["reset"]),
      }
    except (KeyError, TypeError, ValueError) as error:
      raise GitHubError(f"GET {response.url} rate-limit fields are unusable: {error}", url=response.url) from error

    self._remaining_hint = status["remaining"]
    self._reset_hint = status["reset"]
    return status

  def wait_for_rate_limit(self, need: int = 50) -> None:
    """Waits until at least `need` requests are left in the current rate-limit window.

    Safe to call before every batch: when the newest rate-limit headers already show enough
    budget the call costs nothing, and when the budget is short it sleeps at most one hour.

    Args:
      need: How many requests the next batch is expected to spend.

    Raises:
      GitHubError: When the rate-limit endpoint cannot be read.
    """
    if self._remaining_hint is not None and self._remaining_hint >= need:
      return

    status = self.rate_limit()
    if status["remaining"] >= need:
      return

    delay = _seconds_until(status["reset"])
    _warn(
        f"Rate limit low: {status['remaining']} of {status['limit']} requests left and {need} needed. "
        f"Waiting {delay:.0f}s for the window to reset."
    )
    _sleep(delay)
    # Force a fresh read next time: the hint below is from before the reset.
    self._remaining_hint = None

  def _url(self, path: str) -> str:
    """Turns a repository-relative path into an absolute API URL.

    Args:
      path: Repository-relative path, or an absolute http(s) URL which is returned as is.

    Returns:
      The absolute URL to request.
    """
    if path.lower().startswith(("http://", "https://")):
      return path
    return f"{API_ROOT}/repos/{self.owner}/{self.repo}/{path.lstrip('/')}"

  def _request(
      self,
      method: str,
      url: str,
      params: dict[str, Any] | None = None,
      allow_redirects: bool = True,
      timeout: tuple[float, float] = API_TIMEOUT_SECONDS,
      send_auth: bool | None = None,
  ) -> requests.Response:
    """Sends one request, retrying transport errors, 5xx and rate-limit rejections.

    Args:
      method: HTTP method, always "GET" in this collector.
      url: Absolute URL to request.
      params: Query-string parameters, or None.
      allow_redirects: Whether requests may follow redirects itself. `get_bytes` sets this
        to False so it can drop the token before the next hop.
      timeout: (connect, read) timeout in seconds.
      send_auth: Whether to attach the token. None decides by host, which is the safe
        default: only api.github.com is authenticated. False forces the header off and also
        suppresses .netrc lookups, so no credential of any kind reaches the host.

    Returns:
      The response, whose status is below 400.

    Raises:
      GitHubError: When the request still fails after MAX_ATTEMPTS, when its retries would
        sleep longer than MAX_TOTAL_RETRY_SLEEP_SECONDS, or when it fails with a status that
        retrying cannot fix (404, 401, a plain 403).
    """
    if send_auth is None:
      send_auth = _is_api_url(url)
    safe = _safe_url(url)
    slept = 0.0
    headers: dict[str, Any] | None = None
    auth: Any = None
    if not send_auth:
      # A None header value makes requests drop the session header instead of sending it.
      headers = {"Authorization": None}
      auth = _no_auth

    for attempt in range(1, MAX_ATTEMPTS + 1):
      try:
        response = self.session.request(
            method,
            url,
            params=params,
            headers=headers,
            auth=auth,
            allow_redirects=allow_redirects,
            timeout=timeout,
        )
      except requests.RequestException as error:
        delay = self._affordable_delay(_backoff_seconds(attempt), slept)
        if attempt == MAX_ATTEMPTS or delay is None:
          raise GitHubError(f"{method} {safe} failed after {attempt} attempt(s): {error}", url=safe) from error
        _warn(f"{method} {safe} failed ({error}); retrying in {delay:.0f}s.")
        _sleep(delay)
        slept += delay
        continue

      if send_auth:
        self._note_rate_headers(response)
      if response.status_code < 400:
        return response

      wanted = self._retry_delay(response, attempt)
      delay = None if wanted is None else self._affordable_delay(wanted, slept)
      if delay is None or attempt == MAX_ATTEMPTS:
        raise GitHubError(
            f"{method} {safe} answered {response.status_code} after {attempt} attempt(s): {_body_excerpt(response)}",
            status=response.status_code,
            url=safe,
        )
      _warn(f"{method} {safe} answered {response.status_code}; retrying in {delay:.0f}s.")
      _sleep(delay)
      slept += delay

    raise GitHubError(f"{method} {safe} failed after {MAX_ATTEMPTS} attempts.", url=safe)

  @staticmethod
  def _affordable_delay(wanted: float, already_slept: float) -> float | None:
    """Trims a retry delay to what is left of this request's sleeping budget.

    Args:
      wanted: The delay the retry rule asks for, in seconds.
      already_slept: How long this request has slept across earlier attempts.

    Returns:
      The delay to use, or None when the budget is spent and the caller should give up
      instead of waiting.
    """
    left = MAX_TOTAL_RETRY_SLEEP_SECONDS - already_slept
    if left <= 0:
      return None
    return min(wanted, left)

  def _retry_delay(self, response: requests.Response, attempt: int) -> float | None:
    """Decides whether a failed response is worth retrying, and how long to wait first.

    Args:
      response: The failing response.
      attempt: 1-based attempt number that just failed.

    Returns:
      Seconds to wait before retrying, or None when retrying cannot help. 404 is never
      retried; 5xx always is; 403 only when it carries a rate-limit signal, because a plain
      403 means the token lacks permission and will keep meaning that.
    """
    status = response.status_code
    if status == 404:
      return None
    if 500 <= status < 600:
      return _backoff_seconds(attempt)
    if status not in (403, 429):
      return None

    retry_after = _header_int(response, "Retry-After")
    if retry_after is not None:
      return min(float(retry_after), MAX_BACKOFF_SLEEP_SECONDS)

    remaining = _header_int(response, "X-RateLimit-Remaining")
    reset = _header_int(response, "X-RateLimit-Reset")
    if remaining == 0 and reset is not None:
      return _seconds_until(reset)
    if status == 429:
      return _backoff_seconds(attempt)
    if "rate limit" in _body_excerpt(response).lower():
      # Secondary rate limit: GitHub sometimes answers 403 with no Retry-After header.
      return _backoff_seconds(attempt)
    return None

  def _note_rate_headers(self, response: requests.Response) -> None:
    """Remembers the rate-limit headers of a response so wait_for_rate_limit can skip a call.

    Args:
      response: Any response from api.github.com. Responses without the headers are ignored.
    """
    remaining = _header_int(response, "X-RateLimit-Remaining")
    if remaining is not None:
      self._remaining_hint = remaining
    reset = _header_int(response, "X-RateLimit-Reset")
    if reset is not None:
      self._reset_hint = reset

  def _decode_json(self, response: requests.Response) -> Any:
    """Decodes a response body as JSON.

    Args:
      response: Response to decode.

    Returns:
      The decoded object, list or scalar.

    Raises:
      GitHubError: When the body is not valid JSON.
    """
    try:
      return response.json()
    except ValueError as error:
      raise GitHubError(
          f"GET {response.url} did not return JSON: {error}. Body starts with: {_body_excerpt(response, 120)}",
          status=response.status_code,
          url=response.url,
      ) from error
