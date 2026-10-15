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

"""Finds the workflow runs, attempts and jobs the CI metrics dashboard is built from."""

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
  """Raised when a payload from the runs, jobs or pulls endpoints cannot be used."""


class GitHubClientLike(Protocol):
  """The part of `github.GitHubClient` this module uses."""

  def get_json(self, path: str, **params: Any) -> dict[str, Any]:
    """Fetches one JSON object from a repository-relative API path."""

  def paginate(self, path: str, key: str, **params: Any) -> list:
    """Follows every page of a list endpoint and returns the flattened list."""


def _warn(message: str) -> None:
  """Prints a warning to stderr so it never lands in piped collector output."""
  print(message, file=sys.stderr, flush=True)


def _as_dict(value: Any) -> dict[str, Any]:
  """Returns a payload field as a dict, or an empty dict when it is missing or another type."""
  return value if isinstance(value, dict) else {}


def parse_timestamp(value: str | None) -> datetime | None:
  """Reads an ISO-8601 timestamp from the API into a timezone-aware UTC datetime."""
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
  """Returns a datetime in UTC, reading a naive one as UTC rather than as local time."""
  if moment.tzinfo is None:
    return moment.replace(tzinfo=timezone.utc)
  return moment.astimezone(timezone.utc)


def run_id_of(run: dict[str, Any]) -> int:
  """Returns a run's numeric id."""
  try:
    return int(run["id"])
  except (KeyError, TypeError, ValueError) as exc:
    raise RunsError(f"workflow run payload has no usable 'id' field: {run.get('id')!r}") from exc


def run_created_at(run: dict[str, Any]) -> datetime | None:
  """Returns the moment a run was created, or None when the field cannot be read."""
  return parse_timestamp(run.get("created_at"))


def _run_id_or_zero(run: dict[str, Any]) -> int:
  """Returns a run's id for ordering, or 0 when it has none."""
  return _int_or_zero(run.get("id"))


def _order_key(run: dict[str, Any]) -> tuple[datetime, int]:
  """Returns the (created_at, id) key runs are ordered by, oldest first."""
  created = run_created_at(run) or datetime.min.replace(tzinfo=timezone.utc)
  return created, _run_id_or_zero(run)


def sort_runs_newest_first(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
  """Orders runs newest first by creation time, settling ties with the run id."""
  return sorted(runs, key=_order_key, reverse=True)


def dedupe_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
  """Drops repeated runs, keeping the copy that saw the most attempts."""
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
  """Keeps the runs that belong to the wanted workflows."""
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
  """Keeps the runs created inside a window, both ends included."""
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
  """Builds the value of the API's `created` query parameter."""
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
  """Cuts a window into slices narrow enough that no listing hits the 1000-result cap."""
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
  """Returns the (head repository, head branch) a run was started from."""
  head_repo = _as_dict(run.get("head_repository")).get("full_name")
  return str(head_repo or ""), str(run.get("head_branch") or "")


def _embedded_pull_numbers(run: dict[str, Any]) -> set[int]:
  """Returns the pull request numbers GitHub attached to a run payload."""
  entries = run.get("pull_requests")
  if not isinstance(entries, list):
    return set()
  numbers = {_int_or_zero(entry.get("number")) for entry in entries if isinstance(entry, dict)}
  return {number for number in numbers if number}


def _supersede_key(run: dict[str, Any], numbers_by_branch: Mapping[tuple[str, str], int]) -> tuple | None:
  """Returns the concurrency group a run competes in, or None when it competes with nobody."""
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
  """Indexes the pull request number each branch's runs name, where they agree on one."""
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
  """Flags the cancelled runs that a newer run of the same concurrency group replaced."""
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
  """Returns this client's slot in a weak-keyed cache, or None when it cannot be cached."""
  try:
    cache = store.get(client)
    if cache is None:
      cache = {}
      store[client] = cache
    return cache
  except TypeError:
    return None


def clear_caches() -> None:
  """Forgets the resolved workflow ids and branch pull-request lookups."""
  _WORKFLOW_ID_CACHE.clear()
  _BRANCH_PULLS_CACHE.clear()


def resolve_workflow_ids(
    client: GitHubClientLike,
    paths: Iterable[str] | None = None,
    refresh: bool = False,
) -> dict[str, int]:
  """Resolves workflow paths to the numeric ids the runs endpoints take."""
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
  """Fetches every page of one runs listing and checks the shape of what came back."""
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
  """Lists the allowlisted workflows' runs created inside a window, newest first."""
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
  """Re-reads one workflow run."""
  return client.get_json(f"{RUNS_ENDPOINT}/{int(run_id)}")


def list_attempts(client: GitHubClientLike, run: dict[str, Any]) -> list[dict[str, Any]]:
  """Reads every attempt of one run, oldest attempt first."""
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
  """Reads the jobs of one attempt of one run."""
  # A run that never executed (conclusion `action_required`) lists no jobs, and GitHub answers 404 for an
  # attempt it has pruned. Both mean the run has no jobs, not that the collector failed, so both read as [].
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
  """Returns the owner of the repository the run's branch lives in."""
  owner = _as_dict(_as_dict(run.get("head_repository")).get("owner")).get("login")
  if owner:
    return str(owner)
  owner = _as_dict(_as_dict(run.get("repository")).get("owner")).get("login")
  return str(owner) if owner else None


def embedded_pull_request(run: dict[str, Any]) -> dict[str, Any] | None:
  """Returns the pull request GitHub attached to the run payload, if it attached one."""
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
  """Picks the pull request a run belongs to out of the pull requests of its branch."""
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
  """Lists the pull requests opened from one branch, newest state included."""
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
  """Finds the pull request a run belongs to."""
  # GitHub leaves `pull_requests` empty on a fork run and on a run of a pull request that has merged, so the
  # branch lookup below is the ordinary path here, not an edge case.
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
  """Reads one pull request by number."""
  try:
    return client.get_json(f"{PULLS_ENDPOINT}/{int(number)}")
  except github.GitHubError as error:
    if error.status != 404:
      raise
    _warn(f"WARNING: pull request #{number} was not found ({error}).")
    return None


def resolve_pull_request(client: GitHubClientLike, run: dict[str, Any]) -> dict[str, Any] | None:
  """Finds a run's pull request and always answers with the full payload."""
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
