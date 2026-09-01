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

"""Prints every number the CI Pulse dashboard would show for one workflow run.

This is the end-to-end proof that the four collector modules fit together: `github` fetches,
`runs` finds the run, its attempts and its pull request, `derive` turns the job objects into
numbers, `junit` reads the test artifacts, and `rows` shows what would be stored. Nothing is
written anywhere and nothing is sent: every request is a read-only GET.

Run it with a run id:

    GITHUB_TOKEN=$(gh auth token) python3 tools/ci_metrics/collector/demo.py 33468578834

Add `--attempt N` to read one attempt instead of the latest, `--no-tests` to skip the
artifact downloads, which are the slow part, and `--repo owner/name` to point somewhere else.

What to read in the output:

  * Per suite: W workers and D, the wall clock from the first worker's "Run Tests" start to
    the last one's finish. D is never the sum of the JUnit seconds, which the report prints
    next to it precisely so the gap is visible - on run 33468578834 tpu-unit reads 27m06s of
    wall clock against 42 minutes of JUnit time, because pytest runs with `-n auto`.
  * Per job: the queue wait and the setup time, and a dash wherever the number cannot be
    measured. A dash is never a zero: a job that never held a runner has no wait to report.
  * The phase split: four wall-clock spans that tile the run, not per-job sums.
  * Rescues: a job that failed on one attempt and passed on the next, with the machine time
    the failure threw away.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

# The collector package's parent, so `python3 tools/ci_metrics/collector/demo.py <run id>`
# works from anywhere without PYTHONPATH, and the import stays inside tools/ci_metrics.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import derive
from collector import github
from collector import junit
from collector import rows
from collector import runs

DEFAULT_REPO = "AI-Hypercomputer/maxtext"

# Column width of the label in the "key: value" lines, so the values line up.
LABEL_WIDTH = 24


def format_seconds(seconds: float | None) -> str:
  """Formats a duration the way the dashboard reads it, or a dash when there is none.

  Args:
    seconds: The duration, or None when it could not be measured.

  Returns:
    "-" for None, "42s" under a minute, "27m06s" under an hour, otherwise "1h07m".
  """
  if seconds is None:
    return "-"
  total = int(round(seconds))
  sign = "-" if total < 0 else ""
  total = abs(total)
  if total < 60:
    return f"{sign}{total}s"
  if total < 3600:
    return f"{sign}{total // 60}m{total % 60:02d}s"
  return f"{sign}{total // 3600}h{(total % 3600) // 60:02d}m"


def format_count(value: int | None) -> str:
  """Formats a count, or a dash when it was never measured."""
  return "-" if value is None else f"{value:,}"


def line(label: str, value: object) -> None:
  """Prints one aligned "label: value" line."""
  print(f"  {label:<{LABEL_WIDTH}} {value}")


def heading(title: str) -> None:
  """Prints a section heading."""
  print()
  print(title)
  print("-" * len(title))


def split_repo(value: str) -> tuple[str, str]:
  """Splits an "owner/name" argument.

  Args:
    value: The repository, for example "AI-Hypercomputer/maxtext".

  Returns:
    (owner, name).

  Raises:
    ValueError: The value is not two non-empty parts around one "/".
  """
  owner, _, name = value.partition("/")
  if not owner or not name or "/" in name:
    raise ValueError(f"--repo must be owner/name, got {value!r}")
  return owner, name


def print_run_header(run: dict[str, Any], pull: dict[str, Any] | None) -> None:
  """Prints what the run is: workflow, trigger, conclusion and the pull request it belongs to.

  Args:
    run: The run payload.
    pull: The pull request the run belongs to, or None when it has none.
  """
  heading(f"Run {run.get('id')} - {run.get('name')}")
  line("Trigger", run.get("event"))
  line("Status / conclusion", f"{run.get('status')} / {run.get('conclusion')}")
  line("Attempts recorded", run.get("run_attempt"))
  line("Branch", run.get("head_branch"))
  line("Head commit", str(run.get("head_sha") or "")[:12])
  line("Created", run.get("created_at"))
  if pull is None:
    line("Pull request", "none linked")
  else:
    merged = pull.get("merged_at")
    state = "merged " + str(merged) if merged else str(pull.get("state"))
    line("Pull request", f"#{pull.get('number')} ({state})")
  line("Superseded", run.get(runs.SUPERSEDED_FIELD, "not judged"))


def print_phase_split(jobs: Sequence[dict[str, Any]]) -> None:
  """Prints the four wall-clock spans a run's time is divided into.

  Args:
    jobs: The jobs of the attempt being reported.
  """
  split = derive.phase_split(jobs)
  heading("Run phases (wall clock, not per-job sums)")
  line("Queued", format_seconds(split.queued_seconds))
  line("Setup", format_seconds(split.setup_seconds))
  line("Tests", format_seconds(split.tests_seconds))
  line("Tail", format_seconds(split.tail_seconds))
  line("Total", format_seconds(split.total_seconds))
  line("Wall clock", format_seconds(split.wall_seconds))
  line("Machine time", format_seconds(derive.machine_seconds(jobs)))
  line("Parts tile the total", split.parts_sum_to_total)
  line("Jobs counted", f"{split.jobs_counted} ({split.jobs_with_tests} ran tests)")
  line("Jobs from an earlier try", split.jobs_ignored)


def suite_test_line(entry: junit.SuiteEntry | None) -> str:
  """Describes one suite's test counts, or why there are none.

  Args:
    entry: The suite's JUnit entry, or None when the artifacts were not read.

  Returns:
    A one-line summary such as "197 executed of 203 (6 skipped), 2519.7s of JUnit time".
  """
  if entry is None:
    return "not read"
  if entry.result is None:
    return f"no test results ({entry.reason})"
  result = entry.result
  text = (
      f"{format_count(result.executed)} executed of {format_count(result.collected)} "
      f"({format_count(result.skipped)} skipped, {format_count(result.failed)} failed), "
      f"{result.junit_seconds:.1f}s of JUnit time"
  )
  if entry.is_partial:
    text += f" - PARTIAL, workers {sorted(entry.missing_workers)} missing"
  return text


def print_suites(jobs: Sequence[dict[str, Any]], tests: junit.RunTests | None) -> None:
  """Prints workers, wall-clock duration and test counts for every flavor that ran.

  Args:
    jobs: The jobs of the attempt being reported.
    tests: The run's JUnit results, or None when they were not read.
  """
  heading("Suites")
  flavors = derive.test_flavors(jobs)
  if not flavors:
    print("  no test job ran in this attempt")
    return
  for flavor in flavors:
    flavor_jobs = derive.jobs_for_flavor(jobs, flavor)
    duration = derive.suite_duration_seconds(flavor_jobs)
    workers = derive.worker_count(jobs, flavor)
    reported = sum(1 for job in flavor_jobs if derive.held_a_runner(job))
    print(f"  {flavor}")
    line("  workers (W)", f"{workers} named, {reported} held a runner")
    line("  duration (D)", format_seconds(duration))
    line("  machine time", format_seconds(derive.machine_seconds(flavor_jobs)))
    line("  tests", suite_test_line(tests.suites.get(flavor) if tests else None))
  if tests is not None:
    for suite_id, entry in sorted(tests.suites.items()):
      if entry.nested_in and entry.result is not None:
        print(f"  {suite_id} (nested inside {entry.nested_in}; its tests are already counted there)")
        line("  tests", suite_test_line(entry))


def print_jobs(jobs: Sequence[dict[str, Any]]) -> None:
  """Prints one line per job: lane, queue wait, setup, run time.

  Args:
    jobs: The jobs of the attempt being reported.
  """
  heading("Jobs")
  print(f"  {'lane':<9} {'queue':>8} {'setup':>8} {'run':>8}  job")
  for job in sorted(jobs, key=lambda item: str(item.get("name"))):
    note = ""
    if derive.is_carried_over(job):
      note = "  (carried over from an earlier attempt)"
    elif not derive.held_a_runner(job):
      note = "  (never held a runner)"
    print(
        f"  {derive.device_lane(job):<9} "
        f"{format_seconds(derive.queue_seconds(job)):>8} "
        f"{format_seconds(derive.setup_seconds(job)):>8} "
        f"{format_seconds(derive.run_seconds(job)):>8}  "
        f"{job.get('name')}{note}"
    )


def print_rescues(run: dict[str, Any], attempts_jobs: dict[int, list[dict[str, Any]]]) -> None:
  """Prints the jobs that failed on one attempt and passed on the next, and those that did not.

  Args:
    run: The run all the attempts belong to.
    attempts_jobs: Attempt number -> that attempt's jobs.
  """
  heading("Rescues and unrescued failures")
  if len(attempts_jobs) < 2:
    print("  one attempt only, so nothing could have been rescued")
  found = derive.find_rescues(attempts_jobs)
  if not found:
    print("  no job failed and then passed on a re-run")
  for rescue in found:
    print(f"  RESCUED  attempt {rescue.failed_attempt} -> {rescue.passed_attempt}  {rescue.job_name}")
    line("  wasted", f"{format_seconds(rescue.wasted_seconds)} on {rescue.lane}")
  unrescued = rows.failed_never_rescued_rows(run, attempts_jobs)
  if not unrescued:
    print("  no job ended this run in failure")
  for row in unrescued:
    state = "re-run and failed again" if row.rerun_after_failure else "never re-run"
    print(f"  FAILED   attempt {row.failed_attempt}, {state}  {row.job_name}")


def print_stored_rows(run: dict[str, Any], jobs: Sequence[dict[str, Any]], attempt: int) -> None:
  """Prints what one tick would store for this attempt, so the row shapes are visible.

  Args:
    run: The run payload.
    jobs: The jobs of the attempt being reported.
    attempt: The attempt number being reported.
  """
  heading("Stored rows (keys only)")
  built = rows.run_row(run)
  line("Run row", built.key())
  if jobs:
    line("First job row", rows.job_row(run, jobs[0]).key())
  line("Attempt reported", attempt)


def read_tests(client: github.GitHubClient, run_id: int, flavors: Sequence[str]) -> junit.RunTests | None:
  """Reads the run's JUnit artifacts, or reports why it could not.

  Args:
    client: The GitHub client.
    run_id: The workflow run id.
    flavors: The flavors that really ran, from `derive.test_flavors`. Asking about exactly
      those is what turns "not read" into a reason code for the Pathways flavors, which run
      pytest without --junitxml and so publish nothing to find.

  Returns:
    The parsed results, or None when the artifacts could not be read at all. Artifacts live
    about a day, so an older run legitimately has none, and that is a fact rather than an
    error.
  """
  try:
    return junit.read_run_tests(client, run_id, flavors=flavors)
  except junit.JUnitError as error:
    print(f"  test artifacts could not be read: {error}", file=sys.stderr)
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses the command line.

  Args:
    argv: Arguments to parse, or None to read sys.argv.

  Returns:
    The parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description="Print every number the CI Pulse dashboard would show for one workflow run.",
  )
  parser.add_argument("run_id", type=int, help="Workflow run id, for example 33468578834.")
  parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Repository as owner/name (default {DEFAULT_REPO}).")
  parser.add_argument("--attempt", type=int, default=None, help="Report this attempt instead of the latest.")
  parser.add_argument("--no-tests", action="store_true", help="Skip the JUnit artifact downloads.")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """Fetches one run and prints its dashboard numbers.

  Args:
    argv: Command-line arguments, or None to read sys.argv.

  Returns:
    0 on success, 2 when the run could not be read.
  """
  args = parse_args(argv)
  try:
    owner, name = split_repo(args.repo)
  except ValueError as error:
    print(f"error: {error}", file=sys.stderr)
    return 2

  client = github.GitHubClient(owner, name)
  try:
    try:
      run = runs.get_run(client, args.run_id)
    except github.GitHubError as error:
      print(f"error: run {args.run_id} could not be read: {error}", file=sys.stderr)
      return 2

    pull = runs.resolve_pull_request(client, run)
    print_run_header(run, pull)

    attempts = runs.list_attempts(client, run)
    attempts_jobs: dict[int, list[dict[str, Any]]] = {}
    for payload in attempts:
      number = int(payload.get("run_attempt") or 1)
      attempts_jobs[number] = runs.get_jobs(client, args.run_id, number)

    reported = args.attempt if args.attempt is not None else max(attempts_jobs, default=1)
    if reported not in attempts_jobs:
      print(f"error: attempt {reported} is not one of {sorted(attempts_jobs)}", file=sys.stderr)
      return 2
    jobs = attempts_jobs[reported]
    line("Attempts fetched", ", ".join(str(number) for number in sorted(attempts_jobs)))

    tests = None if args.no_tests else read_tests(client, args.run_id, derive.test_flavors(jobs))

    print_phase_split(jobs)
    print_suites(jobs, tests)
    print_jobs(jobs)
    print_rescues(run, attempts_jobs)
    print_stored_rows(run, jobs, reported)
    return 0
  finally:
    client.close()


if __name__ == "__main__":
  sys.exit(main())
