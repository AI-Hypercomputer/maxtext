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

"""Turns GitHub job objects into the numbers the CI Pulse dashboard draws.

Every function here is pure: it takes job dictionaries exactly as the jobs endpoint returns
them, or test rows, and gives back numbers. No network, no files, no clock, no global state.
Nothing is summarised by a model; every value is a timestamp subtraction, a count or a sum.

The rule this module exists to get right
----------------------------------------
`suite_duration_seconds` is WALL CLOCK across a suite's parallel workers: the earliest
"Run Tests" step start to the latest "Run Tests" step finish. It is not the sum of the
workers' job run times, and it is not the sum of the JUnit `time` attributes. The CPU
flavors run pytest with `-n auto`, so JUnit seconds pile up across processes inside one
worker as well as across workers. On run 33468578834 the tpu-unit JUnit files add up to
2519.7 s while the suite really took 1626 s -- 1.55x too much.

Timestamps that do not mean what they look like
-----------------------------------------------
Three job shapes in the real data compute to a plausible-looking number that is false, so
every helper checks for them first and answers None instead:

  * Carried-over jobs. When a run is re-run, GitHub lists all 42 jobs in the new attempt,
    but only the ones it actually re-executed have new timestamps. The rest carry attempt-1
    timestamps under an attempt-2 created_at, so `started_at` precedes `created_at` and the
    queue wait computes negative (-23,056 s in one measured case). Those jobs did not run in
    this attempt and are excluded from this attempt's totals. Including them in a suite
    duration is catastrophic: tpu-unit in attempt 2 of run 32772626658 reads 23,271 s
    naively against a true 1358 s.
  * Jobs that never held a runner. A skipped job, and a job cancelled while still queued,
    both have `created_at == started_at` and an empty `steps` list. Eight such jobs in run
    32999133815 each compute to 13,843 s of machine time on a machine they never had.
  * Steps that did not execute. A cancelled job can still list "Run Tests" with a
    `skipped` conclusion and a zero-length span, which would drag a suite's window to the
    wrong minute. `step_span` ignores steps whose conclusion is "skipped".
  * A job clock that stops before its own steps do. GitHub stamps a cancelled job's
    `completed_at` at the moment the cancellation was issued, but the steps already running
    keep going and keep the runner. Two jobs in run 32999133815 finish their last step 28 s
    and 33 s after their recorded `completed_at`. Taking that field literally made a run's
    tail negative and let one worker's setup plus its suite duration exceed its own run
    time. `_job_end` is the fix: a job held its runner until the later of the two clocks.

None means "this input cannot answer the question". It is never rounded to zero, because a
zero is a value the dashboard would draw as a real drop.
"""

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
  """Where a run's clock time went, as four spans that tile one interval without gaps.

  The spans are wall clock over the whole set of jobs handed in, not per-job sums. Jobs run
  in parallel, so while one job is running tests another may still be waiting for a runner;
  that overlap is why these numbers cannot be added up per job and why `machine_seconds` is
  a different quantity entirely.

  The four spans are cut at five moments and therefore always tile the interval exactly:

      first_created_at -> first_started_at -> tests_started_at
                       -> tests_completed_at -> last_completed_at
           queued              setup                tests            tail

  So `queued + setup + tests + tail == total_seconds` and `setup + tests + tail ==
  wall_seconds`, whenever all four are known. `parts_sum_to_total` checks it.

  The tail is real and is not hidden inside the other three: after the last worker finishes
  pytest a run still uploads results, stops containers and runs its gate jobs. Anything
  ignored is counted in `jobs_ignored` rather than silently dropped.

  When a boundary cannot be measured the spans that need it are None and the rest still hold
  their own meaning. A run whose jobs never reached "Run Tests" has queued measured, setup
  and tests None, and `parts_sum_to_total` False.

  Attributes:
    queued_seconds: First job created to first runner acquired. How long the run waited
      before any machine was working on it.
    setup_seconds: First runner acquired to the first "Run Tests" step start. Image pull,
      checkout and wheel install of whichever job got going first.
    tests_seconds: First "Run Tests" start to last "Run Tests" finish, across every job in
      the input. For one flavor's jobs this is that suite's duration D.
    tail_seconds: Last "Run Tests" finish to the last job completing: result uploads,
      container teardown and the gate jobs.
    total_seconds: First job created to last job completed.
    wall_seconds: First runner acquired to last job completed. The run's wall clock.
    first_created_at: Boundary timestamps, ISO-8601 UTC to the second, or None.
    first_started_at: See first_created_at.
    tests_started_at: See first_created_at.
    tests_completed_at: See first_created_at.
    last_completed_at: See first_created_at. Taken as the later of the last job's
      `completed_at` and its last step's finish, because a cancelled job's steps outlive the
      moment GitHub stamped it complete.
    jobs_counted: Jobs of this attempt with usable timestamps, which set the outer bounds.
    jobs_with_tests: How many of those contributed a "Run Tests" window.
    jobs_ignored: Jobs left out because they carry an earlier attempt's timestamps.
  """

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
    """True when all four spans are known, none is negative, and they add up to `total_seconds`.

    False means at least one boundary was unmeasurable, so the parts describe less than the
    whole interval.

    The "none is negative" half of the test is not decoration. Four spans cut at shared
    moments always add up to the whole, even when one of them runs backwards, so the sum
    alone accepted a split with a -29 s tail. A span that a chart could draw as a bar has to
    be non-negative before this says the split tiles the interval.
    """
    parts = (self.queued_seconds, self.setup_seconds, self.tests_seconds, self.tail_seconds)
    if self.total_seconds is None or any(part is None for part in parts):
      return False
    if any(part < 0 for part in parts if part is not None):
      return False
    return abs(sum(part for part in parts if part is not None) - self.total_seconds) < 0.001


@dataclass(frozen=True)
class Rescue:
  """One job that failed on an attempt and passed when it was re-run.

  Matching is by job name across attempts, because job ids are never reused: in run
  32772626658 all 42 name pairs have different ids between attempt 1 and attempt 2, even
  for the 28 jobs that were not re-executed at all.

  Attributes:
    job_name: The full job name, which is what identifies the job across attempts.
    failed_attempt: Attempt number the job concluded "failure" in.
    passed_attempt: The next attempt present, in which it concluded "success".
    failed_job_id: Job id of the failed attempt, or None when the payload had none.
    passed_job_id: Job id of the passing attempt, or None when the payload had none.
    wasted_seconds: Run time of the failed job: the machine time the re-run threw away.
      None when the failed job never held a runner, which is measurable only as "unknown".
      Nothing in the API separates the seconds a failed job spent on setup from the seconds
      it spent on tests, so this is the whole job, which is what the re-run repeats.
    flavor: Test flavor from the job name, or None for a job that is not a test worker.
    worker: 1-based worker number from "Execute Tests (N)", or None as above.
    lane: Device lane of the failed job, from `device_lane`.
    failed_at: When the failed job stopped holding its runner, ISO-8601 UTC, or None.
  """

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
  """Reads one GitHub timestamp. The only place this module parses a date.

  Args:
    value: An ISO-8601 string such as "2026-09-01T04:08:43Z", or None, or anything else.

  Returns:
    A timezone-aware datetime in UTC, or None when the value is missing, empty, not a
    string or not a readable date. A cancelled job legitimately has no started_at, so None
    is an ordinary answer here and never an error.
  """
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
  """Formats a datetime back to the ISO-8601 UTC shape GitHub uses, to the second.

  Args:
    moment: The moment to format, or None.

  Returns:
    A string like "2026-09-01T04:08:43Z", or None. Sub-second precision is dropped; GitHub's
    job timestamps carry none.
  """
  if moment is None:
    return None
  return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seconds_between(start: Any, end: Any) -> float | None:
  """Subtracts two GitHub timestamps.

  Args:
    start: The earlier timestamp, in any shape `parse_timestamp` accepts.
    end: The later timestamp.

  Returns:
    end - start in seconds, or None when either side is unreadable. A negative result is
    returned as it is: the callers that can produce one check for it themselves and turn it
    into None with a reason, rather than having this helper hide it.
  """
  first = parse_timestamp(start)
  second = parse_timestamp(end)
  if first is None or second is None:
    return None
  return (second - first).total_seconds()


def _gap(start: datetime | None, end: datetime | None) -> float | None:
  """Returns end - start in seconds, or None when either boundary moment is unknown.

  Args:
    start: The earlier moment, or None.
    end: The later moment, or None.

  Returns:
    The span in seconds, or None when either side is missing.
  """
  if start is None or end is None:
    return None
  return (end - start).total_seconds()


def _steps(job: Job) -> list[Mapping[str, Any]]:
  """Returns a job's steps, or an empty list when it has none.

  Args:
    job: The job object.

  Returns:
    The steps list. Empty for a skipped job, for a job cancelled before it started, and for
    a cancelled worker that held a runner but published no steps.
  """
  steps = job.get("steps")
  if not isinstance(steps, list):
    return []
  return [step for step in steps if isinstance(step, Mapping)]


def _job_end(job: Job) -> datetime | None:
  """When a job really stopped holding its runner.

  Normally that is `completed_at`. On a cancelled job it is not: GitHub stamps
  `completed_at` at the moment the cancellation was issued, while the steps already running
  carry on. In run 32999133815 the job "CPU Pretrain Tests (cpu-integration) / Execute Tests
  (1) / cpu-integration" reads `completed_at` 22:30:50 and finishes its last step at
  22:31:23, and its "Run Tests" step alone ran 29 s past the stamp.

  Reading `completed_at` literally produced two wrong numbers: a run's tail went negative,
  because the test window ended after the run did, and one worker's setup plus its suite
  duration came to more seconds than its own run time. Both are impossible, and both go away
  when a job's end covers the steps it ran.

  Args:
    job: The job object.

  Returns:
    The later of `completed_at` and the last readable step finish, or None when the job has
    no readable `completed_at`. A job with no steps ends at `completed_at`, which is every
    normal job: only 2 of the roughly 9,800 jobs measured have a step outliving the stamp.
  """
  end = parse_timestamp(job.get("completed_at"))
  if end is None:
    return None
  step_ends = [parse_timestamp(step.get("completed_at")) for step in _steps(job)]
  latest = [moment for moment in step_ends if moment is not None]
  return max([end, *latest])


def is_carried_over(job: Job) -> bool:
  """True when a job's timestamps come from an earlier attempt than the one listing it.

  GitHub lists every job of a run in every attempt. The ones it did not re-execute keep the
  earlier attempt's `started_at` and `completed_at` but get the new attempt's `created_at`,
  so the start precedes the creation. Such a job did not run in this attempt and must be
  left out of this attempt's totals.

  Args:
    job: The job object.

  Returns:
    True when `started_at` is before `created_at`. False when either timestamp is missing,
    because then there is nothing to compare.
  """
  created = parse_timestamp(job.get("created_at"))
  started = parse_timestamp(job.get("started_at"))
  if created is None or started is None:
    return False
  return started < created


def held_a_runner(job: Job) -> bool:
  """True when a machine really ran this job in the attempt that lists it.

  This is the gate in front of every per-job number. It rejects three shapes:

    * a job still queued or in progress, which has no start or no finish yet;
    * a carried-over job, whose timestamps belong to an earlier attempt;
    * a job that never got a machine - `created_at == started_at` with no steps - which is
      how a skipped job and a job cancelled while queued both look. One of those in run
      32999133815 would otherwise report 13,843 s of machine time.

  A cancelled job that did hold a runner passes: job 97618961480 waited one second, got a
  machine and was cancelled with an empty steps list, so its queue wait is a real 1 s even
  though its setup and its share of the suite duration are unknowable.

  Args:
    job: The job object.

  Returns:
    True when the job's own timestamps describe work done on a machine in this attempt.
  """
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
  """How long a job waited for a runner: `started_at - created_at`.

  Args:
    job: The job object.

  Returns:
    The wait in seconds, or None when the job never held a runner in this attempt. Never a
    negative number and never a zero standing in for "did not run".
  """
  if not held_a_runner(job):
    return None
  return _seconds_between(job.get("created_at"), job.get("started_at"))


def run_seconds(job: Job) -> float | None:
  """How long a job occupied a runner: `started_at` to the end of its last step.

  The end comes from `_job_end`, not straight from `completed_at`, so a cancelled job whose
  steps outlived its stamp is not reported as shorter than the work it did. For every job
  that was not cancelled mid-step the two are the same field.

  Args:
    job: The job object.

  Returns:
    The run time in seconds, or None when the job never held a runner in this attempt.
    Returning None for those is what keeps `machine_seconds` honest: a carried-over job's
    span is real but belongs to the earlier attempt, and a never-started job's span is the
    time until the run was cancelled, on no machine at all.
  """
  if not held_a_runner(job):
    return None
  started = parse_timestamp(job.get("started_at"))
  end = _job_end(job)
  if started is None or end is None:
    return None
  return (end - started).total_seconds()


def _find_step(job: Job, step_name: str) -> Mapping[str, Any] | None:
  """Finds the first step of a job that ran under the given name.

  Steps whose conclusion is "skipped" are passed over: a cancelled job can list "Run Tests"
  as skipped with a zero-length span at the moment the cancellation landed, and treating
  that as a test window moves a suite's duration to the wrong minute.

  Args:
    job: The job object.
    step_name: Exact step name, for example "Run Tests".

  Returns:
    The step mapping, or None when the job has no such step or its only one was skipped.
  """
  for step in _steps(job):
    if step.get("name") != step_name:
      continue
    if step.get("conclusion") == SKIPPED_CONCLUSION:
      continue
    return step
  return None


def step_span(job: Job, step_name: str) -> tuple[str, str] | None:
  """Returns when one step of a job started and finished, as the raw timestamps.

  Args:
    job: The job object.
    step_name: Exact step name, for example "Run Tests" or "Initialize containers".

  Returns:
    (started_at, completed_at) exactly as the payload spells them, or None when the job has
    no such step, the step was skipped, or either timestamp is missing or unreadable.
  """
  step = _find_step(job, step_name)
  if step is None:
    return None
  started = step.get("started_at")
  completed = step.get("completed_at")
  if parse_timestamp(started) is None or parse_timestamp(completed) is None:
    return None
  return str(started), str(completed)


def _step_window(job: Job, step_name: str) -> tuple[datetime, datetime] | None:
  """Same as `step_span`, already parsed, for the callers that do date arithmetic.

  Args:
    job: The job object.
    step_name: Exact step name.

  Returns:
    (start, end) as UTC datetimes, or None on the same conditions as `step_span`.
  """
  span = step_span(job, step_name)
  if span is None:
    return None
  start = parse_timestamp(span[0])
  end = parse_timestamp(span[1])
  if start is None or end is None:
    return None
  return start, end


def setup_seconds(job: Job) -> float | None:
  """How long a job spent getting ready before pytest started.

  Measured from the job's `started_at` to the start of its "Run Tests" step, so it covers
  the container image pull, the checkout, the wheel install and the test assets copy. GPU
  jobs read two to five times every other lane here because the CUDA image pull happens
  inside "Initialize containers".

  Args:
    job: The job object.

  Returns:
    The setup time in seconds, or None when the job never held a runner or never reached
    "Run Tests" - a cancelled worker with an empty steps list has no measurable setup.
  """
  if not held_a_runner(job):
    return None
  started = parse_timestamp(job.get("started_at"))
  window = _step_window(job, RUN_TESTS_STEP)
  if started is None or window is None:
    return None
  return (window[0] - started).total_seconds()


def parse_execute_tests_name(name: Any) -> tuple[str, int] | None:
  """Reads the flavor and worker number out of a test worker's job name.

  Args:
    name: A job name, for example
      "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit".

  Returns:
    (flavor, worker) such as ("tpu-unit", 2), or None for any other job. "Setup Parameters"
    jobs, gate jobs and the Pathways jobs all return None, which is what keeps them out of
    the worker count.
  """
  if not isinstance(name, str):
    return None
  match = EXECUTE_TESTS_NAME.match(name.strip())
  if match is None:
    return None
  return match.group("flavor"), int(match.group("worker"))


def flavor_of(job: Job) -> str | None:
  """Returns the test flavor a job belongs to, from the last segment of its name.

  The last "/" segment of a test job's name is its flavor, both for the matrix workers
  ("... / Execute Tests (2) / tpu-unit") and for the Pathways jobs, which have no Execute
  Tests segment ("TPU Pathways Unit Tests (2) / tpu-pathways-unit").

  Args:
    job: The job object.

  Returns:
    The last "/" segment of the job name, or None for a job whose name has no "/" - the gate
    jobs such as "All Required Tests Passed" - and for the per-flavor "Setup Parameters"
    jobs, whose last segment is "Setup Parameters" rather than a flavor.

    A returned string is a name, not a promise that tests ran under it. The reusable jobs
    that build the wheel, lint and run notebooks are named the same way, so on run
    33468578834 this reads 12 names such as "Build Wheel" and "Pre-commit Linters"
    alongside the 15 real flavors. Use `test_flavors` to get the flavors that actually ran
    tests; every one of those extra names has no worker and no test window, so anything
    gated on a worker count or a duration drops them anyway.
  """
  name = job.get("name")
  if not isinstance(name, str) or " / " not in name:
    return None
  tail = name.rsplit(" / ", 1)[-1].strip()
  if not tail or tail == "Setup Parameters":
    return None
  return tail


def jobs_for_flavor(jobs: Iterable[Job], flavor: str) -> list[Job]:
  """Picks out the jobs that ran one flavor's tests.

  Use this before `suite_duration_seconds`, which takes one suite's jobs and has no way to
  filter them itself.

  Args:
    jobs: The jobs of one run attempt.
    flavor: Exact flavor name, for example "cpu-unit". Matching is exact, so "tpu-unit"
      never picks up "tpu7x-unit".

  Returns:
    The matching jobs in the order they were given. Carried-over and never-started jobs are
    kept: they are part of the flavor and the functions that consume them know to skip them.
  """
  return [job for job in jobs if flavor_of(job) == flavor]


def test_flavors(jobs: Iterable[Job]) -> list[str]:
  """Lists the test flavors that really ran in one attempt, in alphabetical order.

  `flavor_of` reads the last segment of any job name, so mapping it over a whole run also
  picks up the reusable jobs that build the wheel, lint the tree and execute notebooks. This
  keeps only the names backed by a job that ran tests: one named "Execute Tests (N)", or one
  carrying a "Run Tests" step, which is how the Pathways flavors qualify without an Execute
  Tests segment.

  Args:
    jobs: The jobs of one run attempt.

  Returns:
    The flavor names, sorted, with no repeats. Empty for a run whose jobs never reached a
    test step - an action_required run lists no jobs at all.
  """
  flavors: set[str] = set()
  for job in jobs:
    flavor = flavor_of(job)
    if flavor is None:
      continue
    if parse_execute_tests_name(job.get("name")) is not None or _find_step(job, RUN_TESTS_STEP) is not None:
      flavors.add(flavor)
  return sorted(flavors)


def worker_count(jobs: Iterable[Job], flavor: str) -> int:
  """Counts a flavor's parallel workers W in one run.

  W is the number of distinct N in that flavor's "Execute Tests (N)" job names. It is read
  per run and never hardcoded, because upstream changes the matrix size without renaming
  anything. Two jobs with the same N - the same worker listed again in a later attempt -
  count once.

  Args:
    jobs: The jobs of one run attempt.
    flavor: Exact flavor name.

  Returns:
    How many workers the flavor was configured with, counting the ones that never got a
    machine. Zero means no job of that flavor is named "Execute Tests (N)": either the
    flavor did not run at all, or it is a Pathways flavor, whose jobs use a different name
    shape. A zero here is a statement about job names, not about machines, so it must not be
    drawn as "the workers disappeared".
  """
  workers = set()
  for job in jobs:
    parsed = parse_execute_tests_name(job.get("name"))
    if parsed is not None and parsed[0] == flavor:
      workers.add(parsed[1])
  return len(workers)


def device_lane(job: Job) -> str:
  """Maps a job's runs-on labels to the lane that paid for it.

  Args:
    job: The job object.

  Returns:
    One of LANE_TPU, LANE_GPU, LANE_CPU, LANE_BUILD, LANE_HOSTED, LANE_NO_RUNNER for a job
    with no labels at all - a skipped job never asked for a runner - or LANE_UNKNOWN for a
    label that is not in `RUNNER_LABEL_LANES`. An unfamiliar label is never guessed into a
    lane and never raises: upstream added the tpu7x runners without notice, and the honest
    answer to new hardware is to say it is unknown until the map is updated.
  """
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
  """The wall-clock duration D of one suite: earliest test start to latest test finish.

  Hand the jobs of ONE flavor in, from `jobs_for_flavor`. D spans from the first worker's
  "Run Tests" step start to the last worker's "Run Tests" step finish, so a suite whose four
  workers overlap takes as long as the slowest of them, not as long as all four added up.

  D is NOT the sum of the workers' run times and NOT the sum of the JUnit durations. On run
  33468578834 the two tpu-unit JUnit files add up to 2519.7 s while D is 1626 s, because
  pytest runs with `-n auto` inside each worker as well as across workers.

  Workers that did not run in this attempt contribute nothing: a carried-over worker would
  otherwise stretch tpu-unit in attempt 2 of run 32772626658 from 1358 s to 23,271 s.

  Args:
    jobs: One flavor's jobs, from one run attempt.

  Returns:
    D in seconds, or None when no worker of the suite reached "Run Tests" in this attempt.
    None is the answer for a flavor whose workers were all cancelled while queued; a zero
    would be read as a suite that ran instantly.

    A number here says nothing about how many workers produced it. A suite where only some
    workers reported is a partial measurement: compare `worker_count` with the workers that
    contributed, and see `junit.SuiteEntry.is_partial` for the test-count side of the same
    problem.
  """
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
  """A run's clock time: first runner acquired to last job completed.

  The end is taken over every job of the attempt, including the skipped bookkeeping jobs,
  because the run was not over until they concluded. The start is taken only over jobs that
  held a runner, because a skipped job's `started_at` marks the moment it was skipped, not a
  machine starting work.

  Args:
    jobs: The jobs of one run attempt.

  Returns:
    The wall clock in seconds, or None when no job of this attempt held a runner. Jobs
    carrying an earlier attempt's timestamps are excluded from both ends.
  """
  attempt_jobs = list(jobs)
  starts = [parse_timestamp(job.get("started_at")) for job in attempt_jobs if held_a_runner(job)]
  finishes = [_job_end(job) for job in attempt_jobs if not is_carried_over(job)]
  first = [moment for moment in starts if moment is not None]
  last = [moment for moment in finishes if moment is not None]
  if not first or not last:
    return None
  return (max(last) - min(first)).total_seconds()


def machine_seconds(jobs: Iterable[Job]) -> float:
  """Runner time a set of jobs consumed: the sum of their run times.

  Parallel jobs each hold their own runner, so this adds up and is normally much larger than
  the run's wall clock: run 33468578834 took 2470 s of clock time and 23,335 s of machine
  time across 54 jobs.

  Args:
    jobs: Any set of jobs - a whole attempt, one lane, or one flavor.

  Returns:
    The total in seconds. Jobs that never held a runner contribute nothing rather than their
    fake span. 0.0 means no job in the set held a runner, which is a real answer about
    machine time, not a missing one; an empty input gives 0.0 for the same reason.
  """
  total = 0.0
  for job in jobs:
    seconds = run_seconds(job)
    if seconds is not None:
      total += seconds
  return total


def phase_split(jobs: Iterable[Job]) -> PhaseSplit:
  """Splits a run's clock time into queued, setup, tests and tail.

  These are wall-clock spans over the whole set of jobs, cut at five shared boundary
  moments, so they tile the interval exactly and never overlap each other. They are NOT
  per-job phases added up: while one job is running tests another may still be queuing, and
  the same second belongs to only one span here. `machine_seconds` is the quantity that adds
  up per job.

  The tail exists because the run does not end when pytest does; results are uploaded,
  containers are stopped and the gate jobs run. It is reported rather than folded into the
  other three, so that queued + setup + tests + tail == total_seconds whenever all four are
  known. See `PhaseSplit` for the full identity and `parts_sum_to_total` to check it.

  The last moment is `_job_end`, the later of a job's `completed_at` and its last step's
  finish. Taking `completed_at` literally made the tail negative on a cancelled run, because
  a cancelled job's steps keep running for a few seconds after GitHub stamps it complete.

  Args:
    jobs: The jobs of one run attempt. A filtered subset works too - one device lane, or one
      flavor - and then every span describes that subset.

  Returns:
    The split. Spans whose boundary could not be measured are None; the others keep their
    meaning. Jobs carrying an earlier attempt's timestamps are excluded and counted in
    `jobs_ignored`.
  """
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
  """Reads an attempt key, accepting the string keys a JSON round trip leaves behind.

  Args:
    value: The key of an entry in an attempt -> jobs mapping.

  Returns:
    The attempt as an int.

  Raises:
    ValueError: The key is not a number. Dropping it silently is what this replaces: an
      attempts map that had been through JSON reported zero rescues instead of three.
  """
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    raise ValueError(f"attempt key {value!r} is not a number")
  try:
    return int(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"attempt key {value!r} is not a number") from exc


def _job_history(attempts_jobs: Mapping[Any, Sequence[Job]]) -> dict[str, list[tuple[int, Job]]]:
  """Groups every attempt's jobs by name, in attempt order.

  The name is the only identity stable across attempts: GitHub mints a fresh job id in every
  attempt, including for the jobs it carried over without re-running them.

  Args:
    attempts_jobs: Attempt number -> that attempt's jobs, for one run.

  Returns:
    Job name -> [(attempt, job), ...] ascending by attempt. A name missing from an attempt
    simply has no entry for it; names are unique within an attempt in the real data, and a
    repeat keeps its first occurrence so the result does not depend on list order.

  Raises:
    ValueError: An attempt key is not a number.
  """
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
  """Finds the jobs that failed on one attempt and passed on the next attempt they appeared in.

  A rescue is a failure-then-success pair for the SAME job name inside one run. Job ids
  cannot be used for the match: GitHub issues new ids in every attempt, even for the jobs it
  did not re-execute.

  The pairing walks each NAME's own history, not the attempt numbers, because a name can be
  missing from an attempt entirely - attempt 1 of run 32785979907 lists 38 jobs where
  attempts 2 and 3 list 42. Pairing by attempt number would lose a rescue whenever the job
  skipped a middle attempt. `rows.rescue_rows` walks the same history, so the two modules
  cannot disagree about the same run.

  Two shapes that look similar are deliberately not rescues, and stay distinguishable:

    * a job that failed and was never re-run - it has no success to pair with, so it is
      absent from this list while its failure is still in the job rows;
    * a job that was cancelled and later succeeded. Run 32785979907 has exactly that
      (cancelled, success, success across three attempts) and yields zero rescues, because
      cancelled is not failure.

  Args:
    attempts_jobs: Attempt number -> that attempt's jobs, for one run. String keys are
      accepted, because a map that has been through JSON has them.

  Returns:
    One Rescue per pair, sorted by failed attempt then job name so the same input always
    gives the same list. Empty when nothing was rescued.

  Raises:
    ValueError: An attempt key is not a number. It used to be dropped, which reported an
      attempts map that had been through JSON as having no rescues at all.
  """
  rescues: list[Rescue] = []
  for name, observations in _job_history(attempts_jobs).items():
    for (earlier, failed), (later, passed) in zip(observations, observations[1:]):
      if failed.get("conclusion") != FAILURE_CONCLUSION or passed.get("conclusion") != SUCCESS_CONCLUSION:
        continue
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


def _row_field(row: Any, field_name: str) -> Any:
  """Reads one field of a test row, whether the row is an object or a mapping.

  Args:
    row: A `rows.TestRow`, or any object or mapping carrying the same field names.
    field_name: The field to read.

  Returns:
    The value, or None when the row does not carry that field.
  """
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
  """Orders test rows slowest first, breaking ties by name so the order is reproducible.

  Args:
    row: The test row.

  Returns:
    A sort key. Rows with no readable duration sort after every timed row rather than being
    dropped or treated as instant.
  """
  duration = _row_duration(row)
  name = str(_row_field(row, "name") or "")
  classname = str(_row_field(row, "classname") or "")
  if duration is None:
    return (1, 0.0, name, classname)
  return (0, -duration, name, classname)


def slowest_tests(rows: Iterable[Any], per_flavor: int = DEFAULT_SLOWEST_PER_FLAVOR) -> list[Any]:
  """Keeps the slowest tests of each flavor, so a run's history stays small but useful.

  The order is total: rows are ranked by duration descending and ties are broken by test
  name and then class name, so the same input always produces the same output.

  Args:
    rows: Test rows carrying `flavor`, `name`, `classname` and `duration` - `rows.TestRow`
      objects, or mappings with those keys.
    per_flavor: How many to keep per flavor.

  Returns:
    The kept rows, grouped by flavor in alphabetical order and slowest first inside each
    flavor. Skipped tests carry roughly zero seconds, so they sink to the bottom on their
    own and are not filtered out here.

  Raises:
    ValueError: `per_flavor` is below 1, which would silently return nothing.
  """
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
