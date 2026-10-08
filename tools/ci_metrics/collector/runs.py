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

"""Finds the workflow runs, attempts and jobs the CI metrics dashboard is built from.

This is the discovery half of the reading layer. It answers four questions and nothing
else: which runs happened in a window, how many attempts each run has, which jobs each
attempt ran, and which pull request a run belongs to. It derives no metric, stores nothing,
and writes nothing; `derive.py` does the arithmetic and `rows.py` shapes the records.

Every network call goes through `github.GitHubClient`, so authentication, pagination,
retries and rate limiting are not repeated here. Every rule that can be decided without the
network is a separate pure function - `mark_superseded`, `match_pull_request`,
`filter_runs_to_workflows`, `sort_runs_newest_first`, `created_filter`, `split_window` - so
the tests can prove them offline against saved payloads.

Five facts about this repository shape the module, all measured, none of them guesses:

  1. **Workflows are identified by id, resolved from their path.** A display name can be
     edited in the YAML at any time, and two workflows may share one. Ids are resolved once
     per client from `actions/workflows` by matching `path`, and cached. Matching by display
     name happens only when the path lookup finds nothing, and it prints a warning when it
     does.
  2. **The runs listing is capped at 1000 results.** ci_pipeline alone produced 654 runs in
     seven days, so a 30-day query would silently lose everything past the cap. `list_runs`
     warns when a listing comes back at the cap; the backfill is expected to walk the window
     one week at a time with `split_window`.
  3. **`run.pull_requests` is empty far more often than the fork case suggests.** It was
     empty on a merged same-repo run (PR #5070) and populated on three open ones, so the
     `GET /pulls?head={head_owner}:{branch}&state=all` fallback is the ordinary path, not an
     edge case. The head query uses the head repository's owner, which for a fork is the
     contributor, and the answer is matched to the run by head sha.
  4. **`run_attempt` is not stable.** One run read attempt 2 from the runs list and attempt 3
     from the single-run endpoint 26 minutes later. `list_attempts` therefore re-reads the run
     before deciding how many attempts to fetch, and tolerates a count that has grown.
  5. **`action_required` runs never executed.** Their jobs endpoint answers
     `{"total_count": 0, "jobs": []}`, and a run whose attempt has been pruned answers 404.
     `get_jobs` returns an empty list for both instead of raising, because a run with no jobs
     is a fact about that run, not a failure of the collector.

Everything read here is public GitHub REST data fetched with GET. Nothing in this module
calls an AI service: every value is an API field or arithmetic on one.
"""

from __future__ import annotations

import sys
import weakref
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from collector import github

# The workflows the dashboard reads, by path. Paths are stable identifiers inside the
# repository; the display names below are only ever used as a last-resort fallback.
CI_PIPELINE_PATH = ".github/workflows/ci_pipeline.yml"
TPU_IMAGES_PATH = ".github/workflows/tpu_docker_images_pipeline.yml"
GPU_IMAGES_PATH = ".github/workflows/gpu_docker_images_pipeline.yml"

WORKFLOW_ALLOWLIST = (CI_PIPELINE_PATH, TPU_IMAGES_PATH, GPU_IMAGES_PATH)

# Path -> the `name:` in that YAML file, used only when the path lookup fails.
WORKFLOW_DISPLAY_NAMES = {
    CI_PIPELINE_PATH: "MaxText Package Tests",
    TPU_IMAGES_PATH: "TPU Docker Images Pipeline",
    GPU_IMAGES_PATH: "GPU Docker Images Pipeline",
}

RUNS_ENDPOINT = "actions/runs"
WORKFLOWS_ENDPOINT = "actions/workflows"
PULLS_ENDPOINT = "pulls"

# GitHub stops serving a run listing after 1000 items, whatever the window asks for.
RUNS_API_RESULT_CAP = 1000
# How wide a slice of history one listing may safely cover, given the cap above.
BACKFILL_WINDOW_DAYS = 7

# The field `mark_superseded` adds to every run it returns.
SUPERSEDED_FIELD = "superseded"

# The conclusion that makes a run a candidate for supersession.
CONCLUSION_CANCELLED = "cancelled"

# The two triggers whose runs share a concurrency group, straight from ci_pipeline.yml:
#   pull_request     -> "{workflow}-pr-{number}"   one group per pull request
#   schedule         -> "{workflow}-schedule"      one group for every scheduled run
#   everything else  -> "{run_id}"                 a group of one, so it cancels nothing
# A push, a workflow_dispatch or a workflow_call run can therefore never be superseded, and
# saying otherwise drops real runs out of every statistic.
EVENT_PULL_REQUEST = "pull_request"
EVENT_SCHEDULE = "schedule"
SUPERSEDING_EVENTS = (EVENT_PULL_REQUEST, EVENT_SCHEDULE)

# Client -> {tuple of paths: {path: workflow id}}. Weak keys, so a client that goes out of
# scope takes its cache with it and one test's stub can never answer another test's call.
_WORKFLOW_ID_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# Client -> {(head owner, branch): the pull requests of that branch}.
_BRANCH_PULLS_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class RunsError(RuntimeError):
  """Raised when a payload from the runs, jobs or pulls endpoints cannot be used.

  The message always names the endpoint or the run at fault, so a collector tick that fails
  can say which answer broke it. Failures of the request itself stay `github.GitHubError`.
  """


class GitHubClientLike(Protocol):
  """The part of `github.GitHubClient` this module uses.

  Declared structurally so the module states its own needs and tests can pass a stub that
  serves saved payloads. Rate limiting, retries and redirects belong to the client, so
  nothing here waits, retries or builds a URL beyond the repository-relative paths above.
  """

  def get_json(self, path: str, **params: Any) -> dict[str, Any]:
    """Fetches one JSON object from a repository-relative API path."""

  def paginate(self, path: str, key: str, **params: Any) -> list:
    """Follows every page of a list endpoint and returns the flattened list."""


def _warn(message: str) -> None:
  """Prints a warning to stderr so it never lands in piped collector output.

  Args:
    message: The line to print.
  """
  print(message, file=sys.stderr, flush=True)


def _as_dict(value: Any) -> dict[str, Any]:
  """Returns a payload field as a dict, or an empty dict when it is missing or another type.

  Args:
    value: A field of an API payload, which GitHub may send as null.

  Returns:
    The dict, or an empty one. Callers can then chain `.get` without guarding every hop.
  """
  return value if isinstance(value, dict) else {}


def parse_timestamp(value: str | None) -> datetime | None:
  """Reads an ISO-8601 timestamp from the API into a timezone-aware UTC datetime.

  Args:
    value: A timestamp such as "2026-09-01T04:06:01Z". None, an empty string, and anything
      that is not a readable timestamp all give None.

  Returns:
    The moment as a UTC datetime, or None when the field could not be read. A timestamp
    without a zone is read as UTC, which is what the API always sends.
  """
  if not isinstance(value, str):
    return None
  text = value.strip()
  if not text:
    return None
  if text.endswith(("Z", "z")):
    text = text[:-1] + "+00:00"
  try:
    parsed = datetime.fromisoformat(text)
  except ValueError:
    return None
  if parsed.tzinfo is None:
    return parsed.replace(tzinfo=timezone.utc)
  return parsed.astimezone(timezone.utc)


def as_utc(moment: datetime) -> datetime:
  """Returns a datetime in UTC, reading a naive one as UTC rather than as local time.

  Args:
    moment: Any datetime.

  Returns:
    The same moment with a UTC timezone attached.
  """
  if moment.tzinfo is None:
    return moment.replace(tzinfo=timezone.utc)
  return moment.astimezone(timezone.utc)


def run_id_of(run: dict[str, Any]) -> int:
  """Returns a run's numeric id.

  Args:
    run: A workflow run payload.

  Returns:
    The run id.

  Raises:
    RunsError: The payload carries no usable id, so nothing can be fetched for it.
  """
  try:
    return int(run["id"])
  except (KeyError, TypeError, ValueError) as exc:
    raise RunsError(f"workflow run payload has no usable 'id' field: {run.get('id')!r}") from exc


def run_created_at(run: dict[str, Any]) -> datetime | None:
  """Returns the moment a run was created, or None when the field cannot be read.

  Args:
    run: A workflow run payload.

  Returns:
    The creation time as a UTC datetime, or None.
  """
  return parse_timestamp(run.get("created_at"))


def _run_id_or_zero(run: dict[str, Any]) -> int:
  """Returns a run's id for ordering, or 0 when it has none."""
  return _int_or_zero(run.get("id"))


def _order_key(run: dict[str, Any]) -> tuple[datetime, int]:
  """Returns the (created_at, id) key runs are ordered by, oldest first.

  Run ids rise over time, so the id settles the order when two runs share a creation second.
  A run with no readable created_at sorts oldest; callers that must not guess check
  `run_created_at` themselves first.
  """
  created = run_created_at(run) or datetime.min.replace(tzinfo=timezone.utc)
  return created, _run_id_or_zero(run)


def sort_runs_newest_first(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
  """Orders runs newest first by creation time, settling ties with the run id.

  Pure. Several workflows are listed separately by the API, so a merged list has to be
  re-ordered before it can be read as one history.

  Args:
    runs: The runs to order.

  Returns:
    A new list, newest first. The input is not modified.
  """
  return sorted(runs, key=_order_key, reverse=True)


def dedupe_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
  """Drops repeated runs, keeping the copy that saw the most attempts.

  Pure. Paging a listing while CI keeps creating runs can shift a run across a page
  boundary and return it twice, and two windows that share a boundary second return the run
  on the boundary twice.

  Args:
    runs: The runs to deduplicate.

  Returns:
    A new list in input order with one entry per run id. When the same run appears more than
    once, the copy with the highest `run_attempt` wins, because that is the later read. A run
    with no readable id is kept as it is and reported, because two of those are not evidence
    of the same run and dropping one would be silent data loss.
  """
  position: dict[int, int] = {}
  kept: list[dict[str, Any]] = []
  for run in runs:
    run_id = _run_id_or_zero(run)
    if not run_id:
      _warn("WARNING: a run payload has no readable id; dedupe_runs keeps it as a separate run.")
      kept.append(run)
      continue
    seen = position.get(run_id)
    if seen is None:
      position[run_id] = len(kept)
      kept.append(run)
      continue
    if _attempt_of(run) > _attempt_of(kept[seen]):
      kept[seen] = run
  return kept


def _attempt_of(run: dict[str, Any]) -> int:
  """Returns a run's attempt number, defaulting to 1 when the field is missing."""
  return _int_or_zero(run.get("run_attempt")) or 1


def filter_runs_to_workflows(
    runs: Iterable[dict[str, Any]],
    workflow_ids: Iterable[int] | None = None,
    paths: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
  """Keeps the runs that belong to the wanted workflows.

  Pure. A run is kept when its `workflow_id` is one of the wanted ids, or when its `path` is
  one of the wanted paths. The path test is what keeps the collector working on a repository
  where the id lookup failed: a run payload carries the workflow path itself.

  Args:
    runs: The runs to filter.
    workflow_ids: Numeric workflow ids to keep. None or empty means "do not match by id".
    paths: Workflow paths to keep, for example ".github/workflows/ci_pipeline.yml".

  Returns:
    A new list in input order holding only the matching runs.

  Raises:
    RunsError: Neither ids nor paths were given, so there is nothing to filter with - silently
      returning every run would let unrelated workflows into the dashboard - or a workflow id
      is not a number. Every other bad input in this module raises RunsError, so a tick that
      catches it must not have to catch a bare ValueError as well.
  """
  try:
    wanted_ids = {int(i) for i in workflow_ids} if workflow_ids else set()
  except (TypeError, ValueError) as exc:
    raise RunsError(f"workflow_ids must be numbers, got {list(workflow_ids or [])!r}") from exc
  wanted_paths = {str(p) for p in paths} if paths else set()
  if not wanted_ids and not wanted_paths:
    raise RunsError("filter_runs_to_workflows needs workflow ids or workflow paths to match against")

  kept: list[dict[str, Any]] = []
  for run in runs:
    try:
      workflow_id = int(run.get("workflow_id"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
      workflow_id = 0
    if workflow_id and workflow_id in wanted_ids:
      kept.append(run)
      continue
    if wanted_paths and str(run.get("path") or "") in wanted_paths:
      kept.append(run)
  return kept


def filter_runs_to_window(
    runs: Iterable[dict[str, Any]],
    since: datetime,
    until: datetime | None = None,
) -> list[dict[str, Any]]:
  """Keeps the runs created inside a window, both ends included.

  Pure. This is a guard, not the main filter: the window is asked for with the API's
  `created` parameter, and this only removes what the API included at the boundary.

  Args:
    runs: The runs to filter.
    since: Oldest creation time to keep. A naive datetime is read as UTC.
    until: Newest creation time to keep, or None for "up to now".

  Returns:
    A new list in input order. A run whose `created_at` cannot be read is kept and reported
    on stderr, because dropping a run over an unreadable field would lose real history.
  """
  since_utc = as_utc(since)
  until_utc = as_utc(until) if until is not None else None
  kept: list[dict[str, Any]] = []
  for run in runs:
    created = run_created_at(run)
    if created is None:
      _warn(
          f"WARNING: run {run.get('id')} has no readable created_at ({run.get('created_at')!r}); "
          "keeping it in the window rather than dropping it."
      )
      kept.append(run)
      continue
    if created < since_utc:
      continue
    if until_utc is not None and created > until_utc:
      continue
    kept.append(run)
  return kept


def created_filter(since: datetime, until: datetime | None = None) -> str:
  """Builds the value of the API's `created` query parameter.

  Pure. Asking the API for the window is what keeps a tick cheap; fetching everything and
  filtering afterwards would spend the rate-limit budget on runs nobody wants.

  Args:
    since: Oldest creation time to ask for, inclusive. A naive datetime is read as UTC.
    until: Newest creation time to ask for, inclusive, or None for an open-ended window.

  Returns:
    ">=2026-08-25T00:00:00Z" for an open window, or
    "2026-08-25T00:00:00Z..2026-09-01T00:00:00Z" for a closed one.

  Raises:
    RunsError: `until` is before `since`, which would ask for an empty window.
  """
  since_utc = as_utc(since)
  if until is None:
    return f">={_api_datetime(since_utc)}"
  until_utc = as_utc(until)
  if until_utc < since_utc:
    raise RunsError(
        f"created_filter needs until >= since, got since={since_utc.isoformat()} until={until_utc.isoformat()}"
    )
  return f"{_api_datetime(since_utc)}..{_api_datetime(until_utc)}"


def _api_datetime(moment: datetime) -> str:
  """Formats a UTC datetime the way GitHub's search-date syntax expects it."""
  return as_utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_window(
    since: datetime,
    until: datetime,
    days: int = BACKFILL_WINDOW_DAYS,
) -> list[tuple[datetime, datetime]]:
  """Cuts a window into slices narrow enough that no listing hits the 1000-result cap.

  Pure. ci_pipeline produced 654 runs in seven days, so a 30-day listing would lose
  everything past the cap without saying so. The backfill walks the history one slice at a
  time instead.

  Args:
    since: Start of the window, inclusive. A naive datetime is read as UTC.
    until: End of the window, inclusive.
    days: Width of one slice in days.

  Returns:
    Consecutive (start, end) pairs covering the window, oldest first. Both ends of each pair
    are inclusive, so a run created exactly on a seam is returned by both listings;
    `dedupe_runs` removes it.

  Raises:
    RunsError: `until` is before `since`, or `days` is not positive.
  """
  since_utc = as_utc(since)
  until_utc = as_utc(until)
  if until_utc < since_utc:
    raise RunsError(f"split_window needs until >= since, got since={since_utc.isoformat()} until={until_utc.isoformat()}")
  if days <= 0:
    raise RunsError(f"split_window needs a positive number of days, got {days}")

  width = timedelta(days=days)
  slices: list[tuple[datetime, datetime]] = []
  start = since_utc
  while start < until_utc:
    end = min(start + width, until_utc)
    slices.append((start, end))
    start = end
  if not slices:
    slices.append((since_utc, until_utc))
  return slices


def _branch_key(run: dict[str, Any]) -> tuple[str, str]:
  """Returns the (head repository, head branch) a run was started from.

  Two forks can push a branch of the same name - "main" and "patch-1" are not rare - and
  those runs never cancel each other, so the repository is part of the identity.
  """
  head_repo = _as_dict(run.get("head_repository")).get("full_name")
  return str(head_repo or ""), str(run.get("head_branch") or "")


def _embedded_pull_numbers(run: dict[str, Any]) -> set[int]:
  """Returns the pull request numbers GitHub attached to a run payload.

  Usually empty: the array is filled only while the pull request is open, and it is empty on
  fork runs and on runs of a pull request that has since been merged.
  """
  entries = run.get("pull_requests")
  if not isinstance(entries, list):
    return set()
  numbers = {_int_or_zero(entry.get("number")) for entry in entries if isinstance(entry, dict)}
  return {number for number in numbers if number}


def _supersede_key(run: dict[str, Any], numbers_by_branch: Mapping[tuple[str, str], int]) -> tuple | None:
  """Returns the concurrency group a run competes in, or None when it competes with nobody.

  This mirrors the `concurrency.group` expression in ci_pipeline.yml rather than guessing
  from the branch, because the two answers differ. GitHub groups a pull request run by the
  pull request NUMBER, every scheduled run of a workflow into one group, and everything else
  into a group of its own run id - so a cancelled push or workflow_dispatch run was never
  superseded by anything, however many newer runs share its branch. Grouping those by branch
  marked the 4-hourly schedule as the successor of every manual run on main.

  Args:
    run: The run to place.
    numbers_by_branch: (head repository, head branch) -> the one pull request number seen on
      that branch, from `_pull_numbers_by_branch`. A merged pull request's run carries an
      empty `pull_requests` array, so the number is borrowed from a sibling run of the same
      branch that still carries it; without that the two runs of one pull request would fall
      into different groups and neither would supersede the other.

  Returns:
    A hashable group key, or None for a run that cannot be superseded at all.
  """
  event = str(run.get("event") or "").lower()
  workflow_id = _int_or_zero(run.get("workflow_id"))
  if event == EVENT_SCHEDULE:
    return (workflow_id, EVENT_SCHEDULE)
  if event != EVENT_PULL_REQUEST:
    return None
  branch = _branch_key(run)
  numbers = _embedded_pull_numbers(run)
  number = numbers.pop() if len(numbers) == 1 else numbers_by_branch.get(branch, 0)
  if number:
    return (workflow_id, EVENT_PULL_REQUEST, number)
  return (workflow_id, "pull_request_branch", branch)


def _pull_numbers_by_branch(runs: Iterable[dict[str, Any]]) -> dict[tuple[str, str], int]:
  """Indexes the pull request number each branch's runs name, where they agree on one.

  Args:
    runs: The runs being judged.

  Returns:
    (head repository, head branch) -> the number, for every branch whose runs name exactly
    one. A branch whose runs name none, or several, is left out and falls back to the branch
    key, because borrowing one of several numbers would invent the link.
  """
  seen: dict[tuple[str, str], set[int]] = {}
  for run in runs:
    if str(run.get("event") or "").lower() != EVENT_PULL_REQUEST:
      continue
    seen.setdefault(_branch_key(run), set()).update(_embedded_pull_numbers(run))
  return {branch: numbers.pop() for branch, numbers in seen.items() if len(numbers) == 1}


def _int_or_zero(value: Any) -> int:
  """Returns a payload field as an int, or 0 when it is missing or not a number."""
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    return 0
  try:
    return int(value)
  except (TypeError, ValueError):
    return 0


def mark_superseded(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Flags the cancelled runs that a newer run of the same concurrency group replaced.

  Pure - no network, so the rule can be tested offline. A run is superseded when it concluded
  "cancelled" and a newer run exists in the same concurrency group. That is what a push to an
  open pull request does: the group cancels the run in flight and starts another one.
  Superseded runs are still stored, but they are excluded from every statistic, so they have
  to be told apart from a run somebody cancelled by hand.

  The groups are the ones ci_pipeline.yml declares, not "the same branch": one per pull
  request, one for all scheduled runs, and a group of one for every other trigger. A
  cancelled push or workflow_dispatch run is therefore never marked, because nothing could
  have cancelled it. See `_supersede_key`.

  Three limits worth knowing:

    * The decision is made against the runs in this list. A cancelled run whose successor
      was created after the window ends is not marked, and gets its flag on a later tick
      once the successor is in the list too.
    * The newest run of a group is never marked, whatever its conclusion.
    * A pull request run whose payload names no number, and whose branch has no sibling run
      that names one, is grouped by its branch instead. That is the same answer in every
      case seen so far, and it is only ever wrong for two pull requests sharing one head
      branch.

  Args:
    runs: The runs to judge, in any order.

  Returns:
    A new list, in the same order, of shallow copies each carrying a `superseded` bool. The
    input dicts are not modified.
  """
  ordered = list(runs)
  numbers_by_branch = _pull_numbers_by_branch(ordered)
  groups: dict[tuple, list[int]] = {}
  for index, run in enumerate(ordered):
    if run_created_at(run) is None:
      # Without a creation time the run cannot be placed against its siblings, and guessing
      # would either invent a supersession or hide one.
      _warn(f"WARNING: run {run.get('id')} has no readable created_at; it is left unmarked by mark_superseded.")
      continue
    key = _supersede_key(run, numbers_by_branch)
    if key is None:
      continue
    groups.setdefault(key, []).append(index)

  flags = [False] * len(ordered)
  for indexes in groups.values():
    by_age = sorted(indexes, key=lambda i: _order_key(ordered[i]))
    for index in by_age[:-1]:
      conclusion = str(ordered[index].get("conclusion") or "").lower()
      if conclusion == CONCLUSION_CANCELLED:
        flags[index] = True

  return [dict(run, **{SUPERSEDED_FIELD: flag}) for run, flag in zip(ordered, flags)]


def _client_cache(store: weakref.WeakKeyDictionary, client: GitHubClientLike) -> dict | None:
  """Returns this client's slot in a weak-keyed cache, or None when it cannot be cached.

  Args:
    store: The weak-keyed cache to look in.
    client: The client the entry belongs to.

  Returns:
    The client's own dict, or None when the client cannot be weakly referenced. A caller
    that gets None simply does the work again; caching is an optimisation, never a rule.
  """
  try:
    cache = store.get(client)
    if cache is None:
      cache = {}
      store[client] = cache
    return cache
  except TypeError:
    return None


def clear_caches() -> None:
  """Forgets the resolved workflow ids and branch pull-request lookups.

  Both caches live for as long as the client that filled them, which is one collector tick.
  Tests that reuse a stub client across cases call this between them.
  """
  _WORKFLOW_ID_CACHE.clear()
  _BRANCH_PULLS_CACHE.clear()


def resolve_workflow_ids(
    client: GitHubClientLike,
    paths: Iterable[str] | None = None,
    refresh: bool = False,
) -> dict[str, int]:
  """Resolves workflow paths to the numeric ids the runs endpoints take.

  Resolved once per client and cached, because the answer cannot change inside one tick.
  The match is on `path`, which is the workflow's identity in the repository. A display name
  is only tried when the path is not in the answer at all, and that fallback prints a
  warning: names are editable, and two workflows may share one.

  Args:
    client: The GitHub client.
    paths: Workflow paths to resolve. Defaults to `WORKFLOW_ALLOWLIST`.
    refresh: Re-read the workflows endpoint even when the answer is already cached.

  Returns:
    Path -> workflow id, holding only the paths that could be resolved. A path that matched
    neither by path nor by name is left out and reported on stderr.

  Raises:
    RunsError: The workflows endpoint returned something that is not a list of objects.
    github.GitHubError: The request itself failed.
  """
  wanted = tuple(paths) if paths is not None else WORKFLOW_ALLOWLIST
  cache = _client_cache(_WORKFLOW_ID_CACHE, client)
  if cache is not None and not refresh:
    cached = cache.get(wanted)
    if cached is not None:
      return dict(cached)

  payloads = client.paginate(WORKFLOWS_ENDPOINT, "workflows")
  by_path: dict[str, int] = {}
  by_name: dict[str, int] = {}
  ambiguous_names: set[str] = set()
  for payload in payloads:
    if not isinstance(payload, dict):
      raise RunsError(f"{WORKFLOWS_ENDPOINT}: returned a {type(payload).__name__}, expected an object")
    workflow_id = _int_or_zero(payload.get("id"))
    if not workflow_id:
      continue
    path = str(payload.get("path") or "")
    if path:
      by_path.setdefault(path, workflow_id)
    name = str(payload.get("name") or "")
    if name:
      if name in by_name and by_name[name] != workflow_id:
        ambiguous_names.add(name)
      by_name.setdefault(name, workflow_id)

  resolved: dict[str, int] = {}
  for path in wanted:
    if path in by_path:
      resolved[path] = by_path[path]
      continue
    display_name = WORKFLOW_DISPLAY_NAMES.get(path)
    if display_name and display_name in by_name and display_name not in ambiguous_names:
      _warn(
          f"WARNING: no workflow has the path {path}; falling back to the display name "
          f"{display_name!r} (id {by_name[display_name]}). Names can be edited in the YAML, "
          "so check whether the workflow was renamed or moved."
      )
      resolved[path] = by_name[display_name]
      continue
    _warn(f"WARNING: workflow {path} was not found by path or by display name; its runs will not be collected.")

  if cache is not None:
    cache[wanted] = dict(resolved)
  return resolved


def _list_runs_page(client: GitHubClientLike, path: str, created: str, what: str) -> list[dict[str, Any]]:
  """Fetches every page of one runs listing and checks the shape of what came back.

  Args:
    client: The GitHub client.
    path: The listing endpoint, repository-relative.
    created: The value of the `created` query parameter.
    what: What is being listed, for the cap warning.

  Returns:
    The run payloads, in the order GitHub returned them.

  Raises:
    RunsError: The listing held something that is not an object.
    github.GitHubError: The request itself failed.
  """
  payloads = client.paginate(path, "workflow_runs", created=created)
  runs: list[dict[str, Any]] = []
  for payload in payloads:
    if not isinstance(payload, dict):
      raise RunsError(f"{path}: the runs listing held a {type(payload).__name__}, expected an object")
    runs.append(payload)
  if len(runs) >= RUNS_API_RESULT_CAP:
    _warn(
        f"WARNING: the listing for {what} came back with {len(runs)} runs, at GitHub's "
        f"{RUNS_API_RESULT_CAP}-result cap. Older runs in this window were not returned; "
        "split the window with split_window() and list each slice."
    )
  return runs


def list_runs(
    client: GitHubClientLike,
    since: datetime,
    until: datetime | None = None,
    workflow_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
  """Lists the allowlisted workflows' runs created inside a window, newest first.

  The window is asked for with the API's `created` parameter, and each workflow is listed
  through its own endpoint, so the answer never includes a workflow the dashboard does not
  read. Only when no id could be resolved does this sweep the repository's whole runs
  listing and filter it by workflow path, and it says so when it does.

  Watch the cap: GitHub serves at most 1000 runs per listing, and ci_pipeline alone makes
  about 650 a week. Ask for a week at a time - `split_window` cuts a wider window up - and a
  warning is printed if a listing still comes back full.

  Args:
    client: The GitHub client.
    since: Oldest creation time to collect, inclusive. On a normal tick this is the newest
      run already stored, so nothing is fetched twice. A naive datetime is read as UTC.
    until: Newest creation time to collect, inclusive, or None for "up to now".
    workflow_ids: Numeric workflow ids to list. Defaults to the ids of `WORKFLOW_ALLOWLIST`,
      resolved through `resolve_workflow_ids`. When given, it is used as it stands.

  Returns:
    The run payloads, newest first, one entry per run id.

  Raises:
    RunsError: A listing held something that is not an object, or `until` is before `since`.
    github.GitHubError: A request failed after the client's own retries.
  """
  since_utc = as_utc(since)
  until_utc = as_utc(until) if until is not None else None
  created = created_filter(since_utc, until_utc)

  if workflow_ids:
    ids = sorted({int(workflow_id) for workflow_id in workflow_ids})
  else:
    ids = sorted(set(resolve_workflow_ids(client).values()))

  collected: list[dict[str, Any]] = []
  if ids:
    for workflow_id in ids:
      collected.extend(
          _list_runs_page(client, f"{WORKFLOWS_ENDPOINT}/{workflow_id}/runs", created, f"workflow {workflow_id}")
      )
  else:
    _warn(
        "WARNING: no workflow id could be resolved, so every workflow's runs are being listed "
        "and filtered by path instead. This costs far more requests than the normal path."
    )
    swept = _list_runs_page(client, RUNS_ENDPOINT, created, "the whole repository")
    collected.extend(filter_runs_to_workflows(swept, paths=WORKFLOW_ALLOWLIST))

  return sort_runs_newest_first(dedupe_runs(filter_runs_to_window(collected, since_utc, until_utc)))


def get_run(client: GitHubClientLike, run_id: int) -> dict[str, Any]:
  """Re-reads one workflow run.

  Args:
    client: The GitHub client.
    run_id: The run id.

  Returns:
    The run payload as it stands now, which may show more attempts than the copy from a
    listing did.

  Raises:
    github.GitHubError: The request failed, including 404 for a run that no longer exists.
  """
  return client.get_json(f"{RUNS_ENDPOINT}/{int(run_id)}")


def list_attempts(client: GitHubClientLike, run: dict[str, Any]) -> list[dict[str, Any]]:
  """Reads every attempt of one run, oldest attempt first.

  The run is re-read first, because `run_attempt` is not stable: one run read 2 from a
  listing and 3 from the single-run endpoint 26 minutes later. The freshly read payload is
  the newest attempt, so only the earlier attempts are fetched one by one.

  Callers must still check each attempt's `status`: an attempt that is still running is
  returned like any other, and only `completed` attempts may be written to the store.

  Args:
    client: The GitHub client.
    run: The run to expand, as returned by `list_runs`. Only its `id` is used.

  Returns:
    One run payload per attempt, attempt 1 first. An attempt GitHub no longer serves is
    left out and reported on stderr, so the list can be shorter than `run_attempt`.

  Raises:
    RunsError: The run payload has no usable id.
    github.GitHubError: A request failed for a reason other than a missing attempt.
  """
  run_id = run_id_of(run)
  latest = get_run(client, run_id)
  total = max(_attempt_of(latest), 1)

  attempts: list[dict[str, Any]] = []
  for number in range(1, total):
    try:
      attempts.append(client.get_json(f"{RUNS_ENDPOINT}/{run_id}/attempts/{number}"))
    except github.GitHubError as error:
      if error.status != 404:
        raise
      _warn(f"WARNING: run {run_id} attempt {number} is no longer served by GitHub ({error}); skipping it.")
  attempts.append(latest)
  return attempts


def get_jobs(client: GitHubClientLike, run_id: int, attempt: int) -> list[dict[str, Any]]:
  """Reads the jobs of one attempt of one run.

  A run that never executed - conclusion `action_required`, waiting on a maintainer's
  approval - answers this endpoint with an empty list, and an attempt GitHub has pruned
  answers 404. Both mean "this attempt ran no jobs", which is a fact about the run, so both
  give an empty list rather than an error.

  Args:
    client: The GitHub client.
    run_id: The run id.
    attempt: The attempt number, counting from 1.

  Returns:
    Every job payload of that attempt, in the order GitHub returned them. Jobs are returned
    exactly as the API sent them: the carried-over jobs of a re-run, whose timestamps come
    from the earlier attempt, are not filtered out here. `derive.py` decides what a job's
    numbers mean.

  Raises:
    RunsError: `attempt` is below 1, or the listing held something that is not an object.
    github.GitHubError: The request failed for a reason other than a missing attempt.
  """
  if int(attempt) < 1:
    raise RunsError(f"get_jobs needs an attempt number of 1 or more, got {attempt}")

  path = f"{RUNS_ENDPOINT}/{int(run_id)}/attempts/{int(attempt)}/jobs"
  try:
    payloads = client.paginate(path, "jobs")
  except github.GitHubError as error:
    if error.status != 404:
      raise
    _warn(f"WARNING: run {run_id} attempt {attempt} has no jobs endpoint ({error}); reading it as no jobs.")
    return []

  jobs: list[dict[str, Any]] = []
  for payload in payloads:
    if not isinstance(payload, dict):
      raise RunsError(f"{path}: the jobs listing held a {type(payload).__name__}, expected an object")
    jobs.append(payload)
  return jobs


def head_owner(run: dict[str, Any]) -> str | None:
  """Returns the owner of the repository the run's branch lives in.

  For a fork pull request that is the contributor, not the base repository's owner, and the
  `head` query of the pulls endpoint needs exactly that owner.

  Args:
    run: A workflow run payload.

  Returns:
    The login, or None when the payload names no repository.
  """
  owner = _as_dict(_as_dict(run.get("head_repository")).get("owner")).get("login")
  if owner:
    return str(owner)
  owner = _as_dict(_as_dict(run.get("repository")).get("owner")).get("login")
  return str(owner) if owner else None


def embedded_pull_request(run: dict[str, Any]) -> dict[str, Any] | None:
  """Returns the pull request GitHub attached to the run payload, if it attached one.

  Pure. The embedded entry is a short form - number, head, base and urls - with no `state`,
  `merged_at` or `title`. `resolve_pull_request` turns it into the full object when those
  fields are needed.

  Args:
    run: A workflow run payload.

  Returns:
    The entry whose head sha matches the run; the only entry when none matches, because a run
    of a commit that was later replaced still belongs to that pull request; and None when
    `pull_requests` is empty or holds several entries none of which matches. The last rule is
    `match_pull_request`'s rule: guessing between several pull requests would invent a link,
    and it should not depend on which of the two paths found them.

    The array is empty on most runs, including merged same-repo ones, which is why the branch
    lookup below exists.
  """
  entries = run.get("pull_requests")
  if not isinstance(entries, list):
    return None
  candidates = [entry for entry in entries if isinstance(entry, dict)]
  if not candidates:
    return None
  head_sha = str(run.get("head_sha") or "")
  if head_sha:
    matches = [entry for entry in candidates if str(_as_dict(entry.get("head")).get("sha") or "") == head_sha]
    if matches:
      return max(matches, key=_pull_number)
  if len(candidates) == 1:
    return candidates[0]
  return None


def _pull_number(pull: dict[str, Any]) -> int:
  """Returns a pull request's number, or 0 when the payload has none."""
  return _int_or_zero(pull.get("number"))


def match_pull_request(pulls: Iterable[dict[str, Any]], run: dict[str, Any]) -> dict[str, Any] | None:
  """Picks the pull request a run belongs to out of the pull requests of its branch.

  Pure. The head sha is the reliable join: a run records the commit it tested, and a pull
  request records the commit at the tip of its branch. They agree when the run tested the
  final commit, which is the run every chart's x axis is drawn from.

  Args:
    pulls: The pull requests of the run's branch, as the pulls endpoint returned them.
    run: The workflow run to match.

  Returns:
    The pull request whose head sha equals the run's head sha - the highest numbered one,
    should a branch have been reused across several closed pull requests. When no sha
    matches, the branch's only pull request, because a run of a commit that was later
    replaced still belongs to that pull request. None when several pull requests share the
    branch and none matches the sha, because guessing between them would invent a link.
  """
  candidates = [pull for pull in pulls if isinstance(pull, dict)]
  if not candidates:
    return None
  head_sha = str(run.get("head_sha") or "")
  if head_sha:
    matches = [pull for pull in candidates if str(_as_dict(pull.get("head")).get("sha") or "") == head_sha]
    if matches:
      return max(matches, key=_pull_number)
  if len(candidates) == 1:
    return candidates[0]
  return None


def find_pull_requests_for_branch(
    client: GitHubClientLike,
    owner: str,
    branch: str,
    state: str = "all",
) -> list[dict[str, Any]]:
  """Lists the pull requests opened from one branch, newest state included.

  Args:
    client: The GitHub client.
    owner: Owner of the repository the branch lives in - the contributor for a fork.
    branch: The branch name, without the owner prefix.
    state: "all", "open" or "closed". "all" is what the collector wants: the pull requests
      the dashboard cares about are merged, and merged counts as closed.

  Returns:
    The pull request payloads, full objects with `state`, `merged_at` and `title`.

  Raises:
    RunsError: The listing held something that is not an object.
    github.GitHubError: The request failed.
  """
  # This endpoint answers with a bare JSON array, so `paginate` ignores the key it is given.
  payloads = client.paginate(PULLS_ENDPOINT, "pulls", head=f"{owner}:{branch}", state=state)
  pulls: list[dict[str, Any]] = []
  for payload in payloads:
    if not isinstance(payload, dict):
      raise RunsError(f"{PULLS_ENDPOINT}: the listing for {owner}:{branch} held a {type(payload).__name__}")
    pulls.append(payload)
  return pulls


def _branch_pulls(client: GitHubClientLike, owner: str, branch: str) -> list[dict[str, Any]]:
  """Returns a branch's pull requests, asking GitHub once per branch per client."""
  cache = _client_cache(_BRANCH_PULLS_CACHE, client)
  key = (owner, branch)
  if cache is not None and key in cache:
    return cache[key]
  pulls = find_pull_requests_for_branch(client, owner, branch)
  if cache is not None:
    cache[key] = pulls
  return pulls


def link_pull_request(client: GitHubClientLike, run: dict[str, Any]) -> dict[str, Any] | None:
  """Finds the pull request a run belongs to.

  `run.pull_requests` is used when GitHub filled it in. It usually has not: it was empty on
  a merged same-repo run as well as on every fork run, so the branch lookup is the ordinary
  path. That lookup asks for the pull requests of `{head owner}:{branch}` and matches the
  answer to the run by head sha.

  The lookup only runs for a run whose trigger is `pull_request`, because for any other
  trigger the branch is not a pull request's head branch and the answer is a wrong link, not
  a missing one. The 4-hourly scheduled run of this repository is on `main`, and `main` has
  been the head branch of a pull request once, in 2024 - so every scheduled run was being
  linked to pull request #771.

  This never raises. A run that cannot be linked - not a pull request run, no branch, no
  matching pull request, or a lookup that failed - gets None and, where the failure is
  unexpected, a line on stderr, so one unlinkable run cannot end a tick that has hundreds of
  others to collect.

  Args:
    client: The GitHub client.
    run: The workflow run to link.

  Returns:
    The pull request, or None when there is none to link to. Mind the shape: the embedded
    entry is the short form and the looked-up one is the full object. Use
    `resolve_pull_request` when the caller needs `merged_at` either way.
  """
  embedded = embedded_pull_request(run)
  if embedded is not None:
    return embedded

  if str(run.get("event") or "").lower() != EVENT_PULL_REQUEST:
    return None

  branch = str(run.get("head_branch") or "")
  owner = head_owner(run)
  if not branch or not owner:
    _warn(f"WARNING: run {run.get('id')} names no head branch or head repository; it cannot be linked.")
    return None

  try:
    pulls = _branch_pulls(client, owner, branch)
  except github.GitHubError as error:
    _warn(f"WARNING: run {run.get('id')} could not be linked to a pull request ({error}); it has none this tick.")
    return None
  return match_pull_request(pulls, run)


def get_pull_request(client: GitHubClientLike, number: int) -> dict[str, Any] | None:
  """Reads one pull request by number.

  Args:
    client: The GitHub client.
    number: The pull request number.

  Returns:
    The full pull request payload, or None when there is no such pull request.

  Raises:
    github.GitHubError: The request failed for a reason other than a missing pull request.
  """
  try:
    return client.get_json(f"{PULLS_ENDPOINT}/{int(number)}")
  except github.GitHubError as error:
    if error.status != 404:
      raise
    _warn(f"WARNING: pull request #{number} was not found ({error}).")
    return None


def resolve_pull_request(client: GitHubClientLike, run: dict[str, Any]) -> dict[str, Any] | None:
  """Finds a run's pull request and always answers with the full payload.

  `link_pull_request` returns whatever it found, which is the short embedded entry when
  GitHub attached one. The dashboard's x axis is merged pull requests, and only the full
  payload carries `merged_at`, so this fetches it when the short form is what turned up.

  Args:
    client: The GitHub client.
    run: The workflow run to link.

  Returns:
    The full pull request payload, or None when the run has no pull request. If the extra
    read fails, the short entry is returned with a warning rather than nothing at all.
  """
  linked = link_pull_request(client, run)
  if linked is None:
    return None
  if "merged_at" in linked:
    return linked

  number = _pull_number(linked)
  if number <= 0:
    _warn(f"WARNING: run {run.get('id')} is linked to a pull request with no number; keeping the short entry.")
    return linked
  try:
    full = get_pull_request(client, number)
  except github.GitHubError as error:
    _warn(f"WARNING: pull request #{number} could not be read ({error}); keeping the short entry from the run.")
    return linked
  return full if full is not None else linked
