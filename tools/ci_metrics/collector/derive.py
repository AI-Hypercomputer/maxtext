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

"""Turns GitHub job objects into the numbers the CI Pulse dashboard draws."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# One job object exactly as GET /actions/runs/{id}/attempts/{n}/jobs returns it.
Job = Mapping[str, Any]

# The step whose start and finish bound the time a job spent running pytest. Present on
# every test job, including the Pathways ones, which run it without --junitxml.
RUN_TESTS_STEP = "Run Tests"

# A step with this conclusion never executed, whatever timestamps it carries.
SKIPPED_CONCLUSION = "skipped"

# Job conclusions that make a rescue pair.
FAILURE_CONCLUSION = "failure"
SUCCESS_CONCLUSION = "success"

# Device lanes. Build and Hosted are not devices, but every job belongs to exactly one of
# these buckets, so the dashboard can account for all of a run's machine time.
LANE_TPU = "TPU"
LANE_GPU = "GPU"
LANE_CPU = "CPU"
LANE_BUILD = "Build"
LANE_HOSTED = "Hosted"
# The job carried no runs-on label at all: a skipped job never asked for a runner.
LANE_NO_RUNNER = "No runner"
# A runs-on label this module has never been told about. Upstream adds hardware without
# announcing it, so an unknown label is reported as unknown rather than guessed into a lane.
LANE_UNKNOWN = "Unknown"

# Complete label inventory, counted over 196 runs and about 9,800 jobs on 2026-09-01.
RUNNER_LABEL_LANES = {
    "linux-x86-ct6e-180-4tpu": LANE_TPU,
    "linux-x86-ct6e-180-8tpu": LANE_TPU,
    "linux-x86-tpu7x-224-4tpu": LANE_TPU,
    "linux-x86-a2-48-a100-4gpu": LANE_GPU,
    "linux-x86-n2-32": LANE_CPU,
    "linux-x86-n2-16-buildkit": LANE_BUILD,
    "ubuntu-latest": LANE_HOSTED,
}

# "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit". The worker number is the
# one in "Execute Tests (N)", never the one in the caller's name: the Pathways jobs put a
# number in the caller's parentheses and the "Setup Parameters" jobs would otherwise be
# counted as workers of their flavor.
EXECUTE_TESTS_NAME = re.compile(r"^(?P<caller>.+?) / Execute Tests \((?P<worker>\d+)\) / (?P<flavor>[^/]+)$")

# How many of a flavor's slowest tests the collector keeps for every run.
DEFAULT_SLOWEST_PER_FLAVOR = 25


@dataclass(frozen=True)
class PhaseSplit:
  """Where a run's clock time went, as four spans that tile one interval without gaps."""

  queued_seconds: float | None
  setup_seconds: float | None
  tests_seconds: float | None
  tail_seconds: float | None
  total_seconds: float | None
  wall_seconds: float | None
  first_created_at: str | None
  first_started_at: str | None
  tests_started_at: str | None
  tests_completed_at: str | None
  last_completed_at: str | None
  jobs_counted: int
  jobs_with_tests: int
  jobs_ignored: int

  @property
  def parts_sum_to_total(self) -> bool:
    """True when all four spans are known, none is negative, and they add up to `total_seconds`."""
    parts = (self.queued_seconds, self.setup_seconds, self.tests_seconds, self.tail_seconds)
    if self.total_seconds is None or any(part is None for part in parts):
      return False
    if any(part < 0 for part in parts if part is not None):
      return False
    return abs(sum(part for part in parts if part is not None) - self.total_seconds) < 0.001


@dataclass(frozen=True)
class Rescue:
  """One job that failed on an attempt and passed when it was re-run."""

  job_name: str
  failed_attempt: int
  passed_attempt: int
  failed_job_id: int | None
  passed_job_id: int | None
  wasted_seconds: float | None
  flavor: str | None
  worker: int | None
  lane: str
  failed_at: str | None


def parse_timestamp(value: Any) -> datetime | None:
  """Reads one GitHub timestamp. The only place this module parses a date."""
  if not isinstance(value, str):
    return None
  text = value.strip()
  if not text:
    return None
  if text.endswith(("Z", "z")):
    text = text[:-1] + "+00:00"
  try:
    moment = datetime.fromisoformat(text)
  except ValueError:
    return None
  if moment.tzinfo is None:
    return moment.replace(tzinfo=timezone.utc)
  return moment.astimezone(timezone.utc)


def _format_timestamp(moment: datetime | None) -> str | None:
  """Formats a datetime back to the ISO-8601 UTC shape GitHub uses, to the second."""
  if moment is None:
    return None
  return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seconds_between(start: Any, end: Any) -> float | None:
  """Subtracts two GitHub timestamps."""
  first = parse_timestamp(start)
  second = parse_timestamp(end)
  if first is None or second is None:
    return None
  return (second - first).total_seconds()


def _gap(start: datetime | None, end: datetime | None) -> float | None:
  """Returns end - start in seconds, or None when either boundary moment is unknown."""
  if start is None or end is None:
    return None
  return (end - start).total_seconds()


def _steps(job: Job) -> list[Mapping[str, Any]]:
  """Returns a job's steps, or an empty list when it has none."""
  steps = job.get("steps")
  if not isinstance(steps, list):
    return []
  return [step for step in steps if isinstance(step, Mapping)]


def _job_end(job: Job) -> datetime | None:
  """When a job really stopped holding its runner."""
  end = parse_timestamp(job.get("completed_at"))
  if end is None:
    return None
  step_ends = [parse_timestamp(step.get("completed_at")) for step in _steps(job)]
  latest = [moment for moment in step_ends if moment is not None]
  return max([end, *latest])


def is_carried_over(job: Job) -> bool:
  """True when a job's timestamps come from an earlier attempt than the one listing it."""
  created = parse_timestamp(job.get("created_at"))
  started = parse_timestamp(job.get("started_at"))
  if created is None or started is None:
    return False
  return started < created


def held_a_runner(job: Job) -> bool:
  """True when a machine really ran this job in the attempt that lists it."""
  created = parse_timestamp(job.get("created_at"))
  started = parse_timestamp(job.get("started_at"))
  completed = parse_timestamp(job.get("completed_at"))
  if started is None or completed is None:
    return False
  if created is not None and started < created:
    return False
  if created is not None and started == created and not _steps(job):
    return False
  return True


def queue_seconds(job: Job) -> float | None:
  """How long a job waited for a runner: `started_at - created_at`."""
  if not held_a_runner(job):
    return None
  return _seconds_between(job.get("created_at"), job.get("started_at"))


def run_seconds(job: Job) -> float | None:
  """How long a job occupied a runner: `started_at` to the end of its last step."""
  if not held_a_runner(job):
    return None
  started = parse_timestamp(job.get("started_at"))
  end = _job_end(job)
  if started is None or end is None:
    return None
  return (end - started).total_seconds()


def _find_step(job: Job, step_name: str) -> Mapping[str, Any] | None:
  """Finds the first step of a job that ran under the given name."""
  for step in _steps(job):
    if step.get("name") != step_name:
      continue
    if step.get("conclusion") == SKIPPED_CONCLUSION:
      continue
    return step
  return None


def step_span(job: Job, step_name: str) -> tuple[str, str] | None:
  """Returns when one step of a job started and finished, as the raw timestamps."""
  step = _find_step(job, step_name)
  if step is None:
    return None
  started = step.get("started_at")
  completed = step.get("completed_at")
  if parse_timestamp(started) is None or parse_timestamp(completed) is None:
    return None
  return str(started), str(completed)


def _step_window(job: Job, step_name: str) -> tuple[datetime, datetime] | None:
  """Same as `step_span`, already parsed, for the callers that do date arithmetic."""
  span = step_span(job, step_name)
  if span is None:
    return None
  start = parse_timestamp(span[0])
  end = parse_timestamp(span[1])
  if start is None or end is None:
    return None
  return start, end


def setup_seconds(job: Job) -> float | None:
  """How long a job spent getting ready before pytest started."""
  if not held_a_runner(job):
    return None
  started = parse_timestamp(job.get("started_at"))
  window = _step_window(job, RUN_TESTS_STEP)
  if started is None or window is None:
    return None
  return (window[0] - started).total_seconds()


def parse_execute_tests_name(name: Any) -> tuple[str, int] | None:
  """Reads the flavor and worker number out of a test worker's job name."""
  if not isinstance(name, str):
    return None
  match = EXECUTE_TESTS_NAME.match(name.strip())
  if match is None:
    return None
  return match.group("flavor"), int(match.group("worker"))


def flavor_of(job: Job) -> str | None:
  """Returns the last segment of a job's name: a test flavor, or a name like "Build Wheel" for jobs that run no tests."""
  name = job.get("name")
  if not isinstance(name, str) or " / " not in name:
    return None
  tail = name.rsplit(" / ", 1)[-1].strip()
  if not tail or tail == "Setup Parameters":
    return None
  return tail


def jobs_for_flavor(jobs: Iterable[Job], flavor: str) -> list[Job]:
  """Picks out the jobs named for one flavor, including never-started and carried-over ones."""
  return [job for job in jobs if flavor_of(job) == flavor]


def test_flavors(jobs: Iterable[Job]) -> list[str]:
  """Lists the test flavors that really ran in one attempt, in alphabetical order."""
  flavors: set[str] = set()
  for job in jobs:
    flavor = flavor_of(job)
    if flavor is None:
      continue
    if parse_execute_tests_name(job.get("name")) is not None or _find_step(job, RUN_TESTS_STEP) is not None:
      flavors.add(flavor)
  return sorted(flavors)


def worker_count(jobs: Iterable[Job], flavor: str) -> int:
  """Counts a flavor's parallel workers W in one run."""
  workers = set()
  for job in jobs:
    parsed = parse_execute_tests_name(job.get("name"))
    if parsed is not None and parsed[0] == flavor:
      workers.add(parsed[1])
  return len(workers)


def device_lane(job: Job) -> str:
  """Maps a job's runs-on labels to the lane that paid for it."""
  labels = job.get("labels")
  if not isinstance(labels, (list, tuple)) or not labels:
    return LANE_NO_RUNNER
  for label in labels:
    if not isinstance(label, str):
      continue
    lane = RUNNER_LABEL_LANES.get(label.strip().lower())
    if lane is not None:
      return lane
  return LANE_UNKNOWN


def suite_duration_seconds(jobs: Iterable[Job]) -> float | None:
  """The wall-clock duration D of one suite: earliest test start to latest test finish."""
  starts: list[datetime] = []
  ends: list[datetime] = []
  for job in jobs:
    if not held_a_runner(job):
      continue
    window = _step_window(job, RUN_TESTS_STEP)
    if window is None:
      continue
    starts.append(window[0])
    ends.append(window[1])
  if not starts:
    return None
  return (max(ends) - min(starts)).total_seconds()


def run_wall_seconds(jobs: Iterable[Job]) -> float | None:
  """A run's clock time: first runner acquired to last job completed."""
  attempt_jobs = list(jobs)
  starts = [parse_timestamp(job.get("started_at")) for job in attempt_jobs if held_a_runner(job)]
  finishes = [_job_end(job) for job in attempt_jobs if not is_carried_over(job)]
  first = [moment for moment in starts if moment is not None]
  last = [moment for moment in finishes if moment is not None]
  if not first or not last:
    return None
  return (max(last) - min(first)).total_seconds()


def machine_seconds(jobs: Iterable[Job]) -> float:
  """Runner time a set of jobs consumed: the sum of their run times."""
  total = 0.0
  for job in jobs:
    seconds = run_seconds(job)
    if seconds is not None:
      total += seconds
  return total


def phase_split(jobs: Iterable[Job]) -> PhaseSplit:
  """Splits a run's clock time into queued, setup, tests and tail."""
  attempt_jobs = list(jobs)
  this_attempt = [job for job in attempt_jobs if not is_carried_over(job)]
  ignored = len(attempt_jobs) - len(this_attempt)

  created = [parse_timestamp(job.get("created_at")) for job in this_attempt]
  completed = [_job_end(job) for job in this_attempt]
  started = [parse_timestamp(job.get("started_at")) for job in this_attempt if held_a_runner(job)]

  created_moments = [moment for moment in created if moment is not None]
  completed_moments = [moment for moment in completed if moment is not None]
  started_moments = [moment for moment in started if moment is not None]

  test_starts: list[datetime] = []
  test_ends: list[datetime] = []
  for job in this_attempt:
    if not held_a_runner(job):
      continue
    window = _step_window(job, RUN_TESTS_STEP)
    if window is None:
      continue
    test_starts.append(window[0])
    test_ends.append(window[1])

  first_created = min(created_moments) if created_moments else None
  first_started = min(started_moments) if started_moments else None
  tests_started = min(test_starts) if test_starts else None
  tests_completed = max(test_ends) if test_ends else None
  last_completed = max(completed_moments) if completed_moments else None

  return PhaseSplit(
      queued_seconds=_gap(first_created, first_started),
      setup_seconds=_gap(first_started, tests_started),
      tests_seconds=_gap(tests_started, tests_completed),
      tail_seconds=_gap(tests_completed, last_completed),
      total_seconds=_gap(first_created, last_completed),
      wall_seconds=_gap(first_started, last_completed),
      first_created_at=_format_timestamp(first_created),
      first_started_at=_format_timestamp(first_started),
      tests_started_at=_format_timestamp(tests_started),
      tests_completed_at=_format_timestamp(tests_completed),
      last_completed_at=_format_timestamp(last_completed),
      jobs_counted=len(completed_moments),
      jobs_with_tests=len(test_starts),
      jobs_ignored=ignored,
  )


def _attempt_number(value: Any) -> int:
  """Reads an attempt key, accepting the string keys a JSON round trip leaves behind."""
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    raise ValueError(f"attempt key {value!r} is not a number")
  try:
    return int(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"attempt key {value!r} is not a number") from exc


def _job_history(attempts_jobs: Mapping[Any, Sequence[Job]]) -> dict[str, list[tuple[int, Job]]]:
  """Groups every attempt's jobs by name, in attempt order."""
  history: dict[str, list[tuple[int, Job]]] = {}
  for raw_attempt in sorted(attempts_jobs, key=_attempt_number):
    attempt = _attempt_number(raw_attempt)
    seen: set[str] = set()
    for job in attempts_jobs[raw_attempt]:
      name = job.get("name")
      if not isinstance(name, str) or name in seen:
        continue
      seen.add(name)
      history.setdefault(name, []).append((attempt, job))
  return history


def _int_or_none(value: Any) -> int | None:
  """Returns a value as an int, or None when it is missing or not a number."""
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    return None
  try:
    return int(value)
  except (TypeError, ValueError):
    return None


def find_rescues(attempts_jobs: Mapping[Any, Sequence[Job]]) -> list[Rescue]:
  """Finds the jobs that failed on one attempt and passed on a later one."""
  rescues: list[Rescue] = []
  for name, observations in _job_history(attempts_jobs).items():
    for (earlier, failed), (later, passed) in _rescue_pairs(observations):
      parsed = parse_execute_tests_name(name)
      rescues.append(
          Rescue(
              job_name=name,
              failed_attempt=earlier,
              passed_attempt=later,
              failed_job_id=_int_or_none(failed.get("id")),
              passed_job_id=_int_or_none(passed.get("id")),
              wasted_seconds=run_seconds(failed),
              flavor=parsed[0] if parsed else flavor_of(failed),
              worker=parsed[1] if parsed else None,
              lane=device_lane(failed),
              failed_at=_format_timestamp(_job_end(failed)),
          )
      )
  rescues.sort(key=lambda rescue: (rescue.failed_attempt, rescue.job_name))
  return rescues


def _rescue_pairs(
    observations: Sequence[tuple[int, Job]],
) -> list[tuple[tuple[int, Job], tuple[int, Job]]]:
  """Pairs each failure with the next attempt that ran the job to a verdict and passed."""
  pairs: list[tuple[tuple[int, Job], tuple[int, Job]]] = []
  pending: tuple[int, Job] | None = None
  for observation in observations:
    conclusion = observation[1].get("conclusion")
    if conclusion == FAILURE_CONCLUSION:
      pending = observation
    elif conclusion == SUCCESS_CONCLUSION and pending is not None:
      pairs.append((pending, observation))
      pending = None
  return pairs


def _row_field(row: Any, field_name: str) -> Any:
  """Reads one field of a test row, whether the row is an object or a mapping."""
  if isinstance(row, Mapping):
    return row.get(field_name)
  return getattr(row, field_name, None)


def _row_duration(row: Any) -> float | None:
  """Returns a test row's duration in seconds, or None when it is missing or unreadable."""
  value = _row_field(row, "duration")
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _slowest_sort_key(row: Any) -> tuple[int, float, str, str]:
  """Orders test rows slowest first, breaking ties by name so the order is reproducible."""
  duration = _row_duration(row)
  name = str(_row_field(row, "name") or "")
  classname = str(_row_field(row, "classname") or "")
  if duration is None:
    return (1, 0.0, name, classname)
  return (0, -duration, name, classname)


def slowest_tests(rows: Iterable[Any], per_flavor: int = DEFAULT_SLOWEST_PER_FLAVOR) -> list[Any]:
  """Keeps the slowest tests of each flavor, so a run's history stays small but useful."""
  if per_flavor < 1:
    raise ValueError(f"per_flavor must be at least 1, got {per_flavor}")

  by_flavor: dict[str, list[Any]] = {}
  for row in rows:
    flavor = _row_field(row, "flavor")
    by_flavor.setdefault("" if flavor is None else str(flavor), []).append(row)

  kept: list[Any] = []
  for flavor in sorted(by_flavor):
    ranked = sorted(by_flavor[flavor], key=_slowest_sort_key)
    kept.extend(ranked[:per_flavor])
  return kept
