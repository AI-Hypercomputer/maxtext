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

"""Read-only GitHub REST API client for the CI metrics collector."""

from __future__ import annotations

import os
import re
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
# How much of a download body is read at a time, so a body that is too big is noticed early.
DOWNLOAD_CHUNK_BYTES = 64 * 1024

NO_TOKEN_WARNING = (
    "WARNING: no GitHub token found (pass token= or set GITHUB_TOKEN). Unauthenticated "
    "requests are capped at 60 per hour, which is not enough to collect a single run."
)

# Set the first time a client is built without a token, so the warning is printed once per
# process. Tests may reset it to False.
_TOKEN_WARNING_EMITTED = False


class GitHubError(RuntimeError):
  """Raised when a GitHub request cannot be completed or its answer cannot be used."""

  def __init__(self, message: str, status: int | None = None, url: str | None = None) -> None:
    """Builds the error."""
    super().__init__(message)
    self.status = status
    self.url = url


def _sleep(seconds: float) -> None:
  """Sleeps for the given number of seconds (a seam tests can patch out)."""
  if seconds > 0:
    time.sleep(seconds)


def _warn(message: str) -> None:
  """Prints a warning to stderr so it never lands in piped collector output."""
  print(message, file=sys.stderr, flush=True)


def _no_auth(request: Any) -> Any:
  """Auth callable that adds nothing, used to stop requests from applying .netrc credentials."""
  return request


def _is_api_url(url: str) -> bool:
  """Returns True when a URL points at api.github.com, the only host that may see the token."""
  return urlsplit(url).netloc.lower() == API_HOST


def _safe_url(url: str) -> str:
  """Returns a URL without its query string, for messages that may be logged."""
  return url.split("?", 1)[0]


def _without_queries(text: str) -> str:
  """Cuts the query string off every URL or path inside a piece of text."""
  return re.sub(r"\?\S*", "", text)


def _backoff_seconds(attempt: int) -> float:
  """Returns the exponential backoff delay for a failed attempt."""
  return min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SLEEP_SECONDS)


def _seconds_until(reset_epoch: int) -> float:
  """Returns how long to wait for a rate-limit window to reset."""
  remaining = reset_epoch - time.time() + RATE_LIMIT_BUFFER_SECONDS
  return min(max(remaining, 0.0), MAX_RATE_LIMIT_SLEEP_SECONDS)


def _header_int(response: requests.Response, name: str) -> int | None:
  """Reads one response header as an integer."""
  raw = response.headers.get(name)
  if raw is None:
    return None
  try:
    return int(raw.strip())
  except (AttributeError, ValueError):
    return None


def _body_excerpt(response: requests.Response, limit: int = 200) -> str:
  """Returns a short, single-line excerpt of a response body for error messages."""
  try:
    text = " ".join(response.text.split())
  except (UnicodeDecodeError, ValueError):
    return ""
  if len(text) > limit:
    return text[:limit] + "..."
  return text


def _read_limited(response: requests.Response, max_bytes: int | None) -> bytes:
  """Reads a streamed body, counting the bytes that really arrive."""
  safe = _safe_url(response.url)
  if max_bytes is not None:
    announced = _header_int(response, "Content-Length")
    if announced is not None and announced > max_bytes:
      raise GitHubError(
          f"GET {safe} announces {announced:,} bytes and the limit is {max_bytes:,} bytes, so it was not read.",
          status=response.status_code,
          url=safe,
      )
  pieces: list[bytes] = []
  received = 0
  for piece in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
    received += len(piece)
    if max_bytes is not None and received > max_bytes:
      raise GitHubError(
          f"GET {safe} sent more than {max_bytes:,} bytes, so it was stopped.",
          status=response.status_code,
          url=safe,
      )
    pieces.append(piece)
  return b"".join(pieces)


class GitHubClient:
  """Read-only client for one repository's GitHub REST API."""

  def __init__(
      self,
      owner: str,
      repo: str,
      token: str | None = None,
      session: requests.Session | None = None,
      max_wait: float | None = None,
  ) -> None:
    """Builds a client for `owner/repo`."""
    global _TOKEN_WARNING_EMITTED

    # `max_wait` caps the seconds one request may sleep across its retries. A wait longer than
    # what is left of it is not started at all, so the request fails at once instead of sleeping
    # part of the way to a reset it cannot reach. None keeps the one-hour default.
    if max_wait is not None and max_wait < 0:
      raise ValueError(f"max_wait must be 0 or more seconds, got {max_wait}")
    self.owner = owner
    self.repo = repo
    self.token = token or os.environ.get("GITHUB_TOKEN") or None
    self.session = session if session is not None else requests.Session()
    self.max_wait = None if max_wait is None else float(max_wait)
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
    """Fetches one JSON object from a repository-relative API path."""
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

  def paginate(self, path: str, key: str, *, max_items: int | None = None, **params: Any) -> list[Any]:
    """Fetches every page of a list endpoint and returns the items flattened."""
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
      if max_items is not None and len(items) >= max_items:
        return items

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

  def get_bytes(self, url: str, *, max_bytes: int | None = None) -> bytes:
    """Downloads an absolute URL and returns its bytes, following GitHub's redirect."""
    if not url.lower().startswith(("http://", "https://")):
      raise GitHubError(f"get_bytes needs an absolute http(s) URL, got {url!r}.", url=url)
    safe = _safe_url(url)

    slept = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
      try:
        return self._download_once(url, safe, max_bytes)
      except requests.RequestException as error:
        # `_request` retries the requests. The body is read after they return, so a connection
        # that breaks off while it is being read is retried here, from the first request.
        delay = self._affordable_delay(_backoff_seconds(attempt), slept)
        if attempt == MAX_ATTEMPTS or delay is None:
          raise GitHubError(
              f"GET {safe} broke off while its body was being read, after {attempt} attempt(s): "
              f"{_without_queries(str(error))}",
              url=safe,
          ) from error
        _warn(
            f"GET {safe} broke off while its body was being read ({_without_queries(str(error))}); "
            f"retrying in {delay:.0f}s."
        )
        _sleep(delay)
        slept += delay
    raise GitHubError(f"GET {safe} failed after {MAX_ATTEMPTS} attempts.", url=safe)

  def _download_once(self, url: str, safe: str, max_bytes: int | None) -> bytes:
    """Follows the redirects of one download by hand, then reads the final body."""
    response = self._request("GET", url, allow_redirects=False, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
    try:
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
        response.close()
        response = self._request(
            "GET",
            target,
            allow_redirects=False,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            send_auth=False,
            stream=True,
        )
        hops += 1

      if response.status_code != 200:
        raise GitHubError(
            f"GET {_safe_url(response.url)} answered {response.status_code}, expected 200.",
            status=response.status_code,
            url=_safe_url(response.url),
        )
      return _read_limited(response, max_bytes)
    finally:
      response.close()

  def rate_limit(self) -> dict[str, int]:
    """Reads the core rate-limit budget. This endpoint does not spend budget itself."""
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

  @property
  def remaining_hint(self) -> int | None:
    """X-RateLimit-Remaining from the latest API answer, or None before the first one."""
    return self._remaining_hint

  def wait_for_rate_limit(self, need: int = 50) -> None:
    """Waits until at least `need` requests are left in the current rate-limit window."""
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
    """Turns a repository-relative path into an absolute API URL."""
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
      stream: bool = False,
  ) -> requests.Response:
    """Sends one request, retrying transport errors, 5xx and rate-limit rejections."""
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
    streaming: dict[str, bool] = {"stream": True} if stream else {}

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
            **streaming,
        )
      except requests.RequestException as error:
        delay = self._affordable_delay(_backoff_seconds(attempt), slept)
        if attempt == MAX_ATTEMPTS or delay is None:
          raise GitHubError(
              f"{method} {safe} failed after {attempt} attempt(s): {_without_queries(str(error))}", url=safe
          ) from error
        _warn(f"{method} {safe} failed ({_without_queries(str(error))}); retrying in {delay:.0f}s.")
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
        error = GitHubError(
            f"{method} {safe} answered {response.status_code} after {attempt} attempt(s): {_body_excerpt(response)}",
            status=response.status_code,
            url=safe,
        )
        response.close()
        raise error
      # A streamed response holds its connection until it is closed, and this one is not used again.
      response.close()
      _warn(f"{method} {safe} answered {response.status_code}; retrying in {delay:.0f}s.")
      _sleep(delay)
      slept += delay

    raise GitHubError(f"{method} {safe} failed after {MAX_ATTEMPTS} attempts.", url=safe)

  def _affordable_delay(self, wanted: float, already_slept: float) -> float | None:
    """Trims a retry delay to what is left of this request's sleeping budget."""
    if self.max_wait is not None:
      # A hard cap: a wait that does not fit in full is not started. Sleeping part of the way to a
      # rate-limit reset only burns the time, because the retry would be refused all the same.
      if wanted > self.max_wait - already_slept:
        return None
      return wanted
    left = MAX_TOTAL_RETRY_SLEEP_SECONDS - already_slept
    if left <= 0:
      return None
    return min(wanted, left)

  def _retry_delay(self, response: requests.Response, attempt: int) -> float | None:
    """Decides whether a failed response is worth retrying, and how long to wait first."""
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
    """Remembers the rate-limit headers of a response so wait_for_rate_limit can skip a call."""
    remaining = _header_int(response, "X-RateLimit-Remaining")
    if remaining is not None:
      self._remaining_hint = remaining
    reset = _header_int(response, "X-RateLimit-Reset")
    if reset is not None:
      self._reset_hint = reset

  def _decode_json(self, response: requests.Response) -> Any:
    """Decodes a response body as JSON."""
    try:
      return response.json()
    except ValueError as error:
      raise GitHubError(
          f"GET {response.url} did not return JSON: {error}. Body starts with: {_body_excerpt(response, 120)}",
          status=response.status_code,
          url=response.url,
      ) from error
