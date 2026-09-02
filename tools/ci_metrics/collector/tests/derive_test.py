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

"""Offline unit tests for `collector.derive`.

Everything here runs against the saved job payloads in `tests/fixtures/` and never touches
the network: the base test case replaces `socket.socket`, so a test that reached for GitHub
would fail instead of hanging or spending rate limit. `derive` has no client and no I/O of
its own, so the fixtures are simply read from disk and handed in as dictionaries.

Every expected number is a measured fact about a real run, not a round number chosen to make
a test pass. The runs used are:

  * 33468578834 - one clean attempt, 54 jobs, 13 test flavors. The source of the worked
    example in the module docstring.
  * 32772626658 and 33037584699 - two runs that were re-run, used for the rescue pairs and
    for the carried-over-job trap.
  * 32785979907 - three attempts, a cancelled worker with an empty steps list, and zero
    rescues even though six job names failed.
  * 32999133815 - eight jobs cancelled before they ever held a runner, each of which computes
    to 13,843 s of machine time on a machine it never had.
  * 33465601432 - an `action_required` run whose jobs list is literally empty.

The headline test is `SuiteDurationTest.test_tpu_unit_duration_is_wall_clock_not_a_sum`: the
tpu-unit suite of run 33468578834 took 1626 s of wall clock, while its two JUnit files add up
to 2519.747 s and its two jobs add up to 2966 s of runner time. All three numbers are checked
in one test so the difference between them cannot quietly disappear.

Two cases have no real data and are synthesised, which is called out at the point of use:

  * a job with a null `started_at` or `completed_at`. GitHub fills both on every completed
    job, so this shape only exists while a job is still queued or in progress, and no such
    job was in flight when the fixtures were captured.
  * test rows with tied durations, for the `slowest_tests` ordering test.

The tests are plain `unittest`, so they need nothing but the standard library. pytest collects
them too, because it understands `unittest.TestCase` and the file is named the way the
repository's pytest.ini expects.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/derive_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/derive_test.py
"""

from __future__ import annotations

import copy
import json
import socket
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock
from xml.etree import ElementTree

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import derive
from collector import rows as row_module

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The clean single-attempt run every flavor number below was measured from.
CLEAN_RUN_JOBS = "jobs.json"

# Measured per flavor in run 33468578834: parallel workers W, the suite's wall-clock duration
# D in seconds, and the runner seconds its jobs consumed. D is never the machine total.
FLAVOR_TRUTH = {
    "tpu-unit": {"workers": 2, "duration": 1626.0, "machine": 2966.0},
    "cpu-unit": {"workers": 4, "duration": 1753.0, "machine": 2431.0},
    "cpu-post-training-unit": {"workers": 4, "duration": 466.0, "machine": 1135.0},
    "gpu-unit": {"workers": 1, "duration": 171.0, "machine": 570.0},
    "gpu-integration": {"workers": 1, "duration": 464.0, "machine": 856.0},
    "tpu-integration": {"workers": 1, "duration": 1674.0, "machine": 1746.0},
    "cpu-integration": {"workers": 1, "duration": 1155.0, "machine": 1244.0},
    "tpu-post-training-unit": {"workers": 1, "duration": 145.0, "machine": 263.0},
    "tpu-post-training-integration": {"workers": 1, "duration": 150.0, "machine": 266.0},
    "cpu-post-training-integration": {"workers": 1, "duration": 206.0, "machine": 349.0},
    "tpu7x-unit": {"workers": 1, "duration": 2224.0, "machine": 2301.0},
    "tpu7x-integration": {"workers": 1, "duration": 1479.0, "machine": 1548.0},
    "tpu7x-post-training-unit": {"workers": 1, "duration": 218.0, "machine": 338.0},
}

# Measured per worker in run 33468578834: job id, queue wait, setup time and run time, all in
# seconds. Keyed by (flavor, worker number), because the jobs endpoint does not list workers
# in worker order - tpu-unit worker 2 comes first in the payload.
WORKER_TRUTH = {
    ("tpu-unit", 1): {"job_id": 99733940992, "queue": 109.0, "setup": 71.0, "run": 1217.0},
    ("tpu-unit", 2): {"job_id": 99733940989, "queue": 74.0, "setup": 193.0, "run": 1749.0},
    ("cpu-unit", 1): {"job_id": 99733937900, "queue": 88.0, "setup": 74.0, "run": 334.0},
    ("cpu-unit", 2): {"job_id": 99733937893, "queue": 62.0, "setup": 81.0, "run": 260.0},
    ("cpu-unit", 3): {"job_id": 99733938036, "queue": 319.0, "setup": 80.0, "run": 197.0},
    ("cpu-unit", 4): {"job_id": 99733937943, "queue": 278.0, "setup": 83.0, "run": 1640.0},
    ("cpu-post-training-unit", 1): {"job_id": 99733935093, "queue": 330.0, "setup": 110.0, "run": 283.0},
    ("cpu-post-training-unit", 2): {"job_id": 99733935092, "queue": 55.0, "setup": 107.0, "run": 283.0},
    ("cpu-post-training-unit", 3): {"job_id": 99733935040, "queue": 1.0, "setup": 126.0, "run": 305.0},
    ("cpu-post-training-unit", 4): {"job_id": 99733935084, "queue": 51.0, "setup": 106.0, "run": 264.0},
    ("gpu-unit", 1): {"job_id": None, "queue": 1.0, "setup": 374.0, "run": 570.0},
    ("gpu-integration", 1): {"job_id": None, "queue": 1.0, "setup": 367.0, "run": 856.0},
    ("tpu-integration", 1): {"job_id": None, "queue": 289.0, "setup": 55.0, "run": 1746.0},
    ("cpu-integration", 1): {"job_id": None, "queue": 1.0, "setup": 68.0, "run": 1244.0},
}

# The two JUnit files tpu-unit uploaded on run 33468578834, and the seconds their `<testcase>`
# elements add up to. The sum is 1.55x the suite's real duration because pytest runs with
# `-n auto` inside each worker as well as across the two workers.
TPU_UNIT_JUNIT_FILES = ("tpu-unit-1.xml", "tpu-unit-2.xml")
TPU_UNIT_JUNIT_SECONDS = 2519.747

# Whole-run figures for run 33468578834, measured over all 54 jobs.
CLEAN_RUN_JOB_COUNT = 54
CLEAN_RUN_WALL_SECONDS = 2470.0
CLEAN_RUN_MACHINE_SECONDS = 23335.0

# Rescues of run 32772626658, attempt 1 -> attempt 2: job name -> wasted seconds.
RESCUES_32772626658 = {
    "CPU Posttrain Tests (cpu-post-training-unit) / Execute Tests (3) / cpu-post-training-unit": 146.0,
    "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit": 104.0,
    "CPU Pretrain Tests (cpu-unit) / Execute Tests (3) / cpu-unit": 115.0,
}

# Job names of run 32772626658 that failed in attempt 1 and failed again in attempt 2. They
# are failures, not rescues, and must never appear in `find_rescues` output.
FAILED_AGAIN_32772626658 = (
    "All Required Tests Passed",
    "CPU Pretrain Tests (cpu-unit) / Execute Tests (2) / cpu-unit",
    "TPU Pathways Unit Tests (1) / tpu-pathways-unit",
    "TPU Pathways Unit Tests (2) / tpu-pathways-unit",
    "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit",
)

# Rescues of run 33037584699, attempt 1 -> attempt 2. Every attempt-1 failure was rescued, and
# they span three device lanes plus one gate job on the hosted runners.
RESCUES_33037584699 = {
    "All Required Tests Passed": 3.0,
    "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit": 224.0,
    "GPU Tests (gpu-integration) / Execute Tests (1) / gpu-integration": 862.0,
    "TPU Pretrain Tests (tpu-integration) / Execute Tests (1) / tpu-integration": 1545.0,
}

# Suite durations of run 32772626658 attempt 2, counting only the workers that really
# re-executed. Every other flavor of that attempt has no re-executed worker at all, so its
# duration is None rather than a number.
ATTEMPT_2_DURATIONS_32772626658 = {
    "cpu-unit": 234.0,
    "cpu-post-training-unit": 140.0,
    "tpu-unit": 1358.0,
}

# What the same three suites read if carried-over workers are not filtered out. Kept in the
# test so the size of the trap stays visible.
ATTEMPT_2_NAIVE_DURATIONS_32772626658 = {
    "cpu-unit": 19594.0,
    "cpu-post-training-unit": 20872.0,
    "tpu-unit": 23271.0,
}


def read_jobs(fixture_name: str) -> list[dict[str, Any]]:
  """Loads one saved jobs payload.

  Args:
    fixture_name: File name inside `tests/fixtures/`, e.g. "jobs.json".

  Returns:
    The `jobs` list, parsed fresh so no test can mutate another test's input.
  """
  payload = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
  return list(payload["jobs"])


def junit_seconds(fixture_name: str) -> float:
  """Adds up the `time` attributes of one JUnit file's `<testcase>` elements.

  This is deliberately computed here rather than imported from `collector.junit`, so the
  headline test proves the JUnit total straight from the saved XML.

  Args:
    fixture_name: File name inside `tests/fixtures/`, e.g. "tpu-unit-1.xml".

  Returns:
    The summed seconds.
  """
  root = ElementTree.parse(FIXTURES / fixture_name).getroot()
  return sum(float(case.get("time") or 0.0) for case in root.iter("testcase"))


def jobs_by_worker(jobs: list[dict[str, Any]], flavor: str) -> dict[int, dict[str, Any]]:
  """Indexes one flavor's matrix workers by their "Execute Tests (N)" number.

  Args:
    jobs: The jobs of one run attempt.
    flavor: Exact flavor name.

  Returns:
    Worker number -> job. Jobs of the flavor that are not matrix workers are left out.
  """
  by_worker: dict[int, dict[str, Any]] = {}
  for job in derive.jobs_for_flavor(jobs, flavor):
    parsed = derive.parse_execute_tests_name(job.get("name"))
    if parsed is not None:
      by_worker[parsed[1]] = job
  return by_worker


def job_named(jobs: list[dict[str, Any]], name: str) -> dict[str, Any]:
  """Finds one job by its exact name.

  Args:
    jobs: The jobs to search.
    name: The exact job name.

  Returns:
    The job.

  Raises:
    KeyError: No job of that name is in the list, which means the fixture changed.
  """
  for job in jobs:
    if job.get("name") == name:
      return job
  raise KeyError(f"no job named {name!r} in this fixture")


def test_row(flavor: str, name: str, classname: str, duration: float | None) -> dict[str, Any]:
  """Builds one mapping-shaped test row for the `slowest_tests` tests.

  Args:
    flavor: The flavor the row belongs to.
    name: The test name.
    classname: The dotted module and class.
    duration: Seconds, or None for a row whose duration could not be read.

  Returns:
    A mapping carrying the four fields `slowest_tests` reads.
  """
  return {"flavor": flavor, "name": name, "classname": classname, "duration": duration}


class OfflineTestCase(unittest.TestCase):
  """Base class that turns any attempt to open a socket into a test failure."""

  def setUp(self) -> None:
    """Blocks the network for the duration of one test."""
    super().setUp()

    def deny(*args: object, **kwargs: object) -> None:
      """Refuses every socket the test might try to open.

      Args:
        *args: Ignored.
        **kwargs: Ignored.

      Raises:
        AssertionError: Always.
      """
      raise AssertionError("these tests must run offline, but something tried to open a socket")

    for attribute in ("socket", "create_connection"):
      patch = mock.patch.object(socket, attribute, deny)
      patch.start()
      self.addCleanup(patch.stop)

  def assert_seconds(self, actual: float | None, expected: float) -> None:
    """Asserts a derived number of seconds, allowing for float noise only.

    Args:
      actual: What the module returned.
      expected: The measured figure.
    """
    self.assertIsNotNone(actual)
    assert actual is not None
    self.assertAlmostEqual(actual, expected, places=3)


class SuiteDurationTest(OfflineTestCase):
  """Covers `suite_duration_seconds`, the rule the whole module exists to get right."""

  def setUp(self) -> None:
    """Loads the clean run's jobs."""
    super().setUp()
    self.jobs = read_jobs(CLEAN_RUN_JOBS)

  def test_tpu_unit_duration_is_wall_clock_not_a_sum(self) -> None:
    """tpu-unit took 1626 s of clock time, not the 2519.7 s of JUnit nor the 2966 s of runners.

    This is the headline: three plausible-looking totals for one suite, and only the first is
    the suite's duration. The other two are asserted in the same test so that a change which
    quietly swapped one for another could not pass.
    """
    suite_jobs = derive.jobs_for_flavor(self.jobs, "tpu-unit")
    duration = derive.suite_duration_seconds(suite_jobs)

    self.assert_seconds(duration, 1626.0)

    junit_total = sum(junit_seconds(name) for name in TPU_UNIT_JUNIT_FILES)
    self.assertAlmostEqual(junit_total, TPU_UNIT_JUNIT_SECONDS, places=3)
    self.assertNotAlmostEqual(duration, junit_total, places=3)
    self.assertAlmostEqual(junit_total / 1626.0, 1.55, places=2)

    job_total = sum(derive.run_seconds(job) or 0.0 for job in suite_jobs)
    self.assertAlmostEqual(job_total, 2966.0, places=3)
    self.assertNotAlmostEqual(duration, job_total, places=3)
    self.assertAlmostEqual(derive.machine_seconds(suite_jobs), job_total, places=3)

  def test_duration_is_the_window_the_two_workers_share(self) -> None:
    """D runs from the earliest "Run Tests" start to the latest finish, across both workers."""
    by_worker = jobs_by_worker(self.jobs, "tpu-unit")
    self.assertEqual(
        derive.step_span(by_worker[1], derive.RUN_TESTS_STEP), ("2026-09-01T04:11:43Z", "2026-09-01T04:30:32Z")
    )
    self.assertEqual(
        derive.step_span(by_worker[2], derive.RUN_TESTS_STEP), ("2026-09-01T04:13:10Z", "2026-09-01T04:38:49Z")
    )

    # 04:11:43 (worker 1 starts) -> 04:38:49 (worker 2 finishes) = 1626 s.
    self.assert_seconds(derive.suite_duration_seconds(derive.jobs_for_flavor(self.jobs, "tpu-unit")), 1626.0)

  def test_every_flavor_of_the_clean_run(self) -> None:
    """Worker count, duration and machine seconds for all 13 flavors of run 33468578834."""
    for flavor, truth in FLAVOR_TRUTH.items():
      with self.subTest(flavor=flavor):
        suite_jobs = derive.jobs_for_flavor(self.jobs, flavor)
        self.assertEqual(derive.worker_count(self.jobs, flavor), truth["workers"])
        self.assert_seconds(derive.suite_duration_seconds(suite_jobs), truth["duration"])
        self.assertAlmostEqual(derive.machine_seconds(suite_jobs), truth["machine"], places=3)

  def test_duration_can_outlast_the_slowest_single_worker(self) -> None:
    """cpu-unit ran 1753 s although no one worker ran longer than 1640 s: the workers stagger."""
    suite_jobs = derive.jobs_for_flavor(self.jobs, "cpu-unit")
    self.assert_seconds(derive.suite_duration_seconds(suite_jobs), 1753.0)
    longest_worker = max(derive.run_seconds(job) or 0.0 for job in suite_jobs)
    self.assertAlmostEqual(longest_worker, 1640.0, places=3)
    self.assertGreater(1753.0, longest_worker)

  def test_matching_is_exact_so_tpu_unit_never_picks_up_tpu7x_unit(self) -> None:
    """The flavor match is exact, so "tpu-unit" never picks up "tpu7x-unit"."""
    self.assertEqual(len(derive.jobs_for_flavor(self.jobs, "tpu-unit")), 2)
    self.assertEqual(len(derive.jobs_for_flavor(self.jobs, "tpu7x-unit")), 1)
    self.assert_seconds(derive.suite_duration_seconds(derive.jobs_for_flavor(self.jobs, "tpu7x-unit")), 2224.0)

  def test_no_jobs_is_none_and_never_zero(self) -> None:
    """A flavor nothing ran for answers None, because a zero would be drawn as an instant suite."""
    self.assertIsNone(derive.suite_duration_seconds([]))
    self.assertIsNone(derive.suite_duration_seconds(derive.jobs_for_flavor(self.jobs, "no-such-flavor")))


class QueueAndSetupTest(OfflineTestCase):
  """Covers `queue_seconds` and `setup_seconds` against the measured per-worker figures."""

  def setUp(self) -> None:
    """Loads the clean run's jobs."""
    super().setUp()
    self.jobs = read_jobs(CLEAN_RUN_JOBS)

  def test_every_measured_worker(self) -> None:
    """Queue wait, setup time and run time per worker, for every flavor that was measured."""
    for (flavor, worker), truth in WORKER_TRUTH.items():
      with self.subTest(flavor=flavor, worker=worker):
        job = jobs_by_worker(self.jobs, flavor)[worker]
        if truth["job_id"] is not None:
          self.assertEqual(job["id"], truth["job_id"])
        self.assert_seconds(derive.queue_seconds(job), truth["queue"])
        self.assert_seconds(derive.setup_seconds(job), truth["setup"])
        self.assert_seconds(derive.run_seconds(job), truth["run"])

  def test_gpu_setup_dwarfs_every_other_lane(self) -> None:
    """The CUDA image pull sits inside "Initialize containers", so GPU setup is 4x-5x the rest."""
    gpu_setup = derive.setup_seconds(jobs_by_worker(self.jobs, "gpu-unit")[1])
    cpu_setup = derive.setup_seconds(jobs_by_worker(self.jobs, "cpu-unit")[1])
    self.assert_seconds(gpu_setup, 374.0)
    self.assert_seconds(cpu_setup, 74.0)
    assert gpu_setup is not None and cpu_setup is not None
    self.assertGreater(gpu_setup / cpu_setup, 4.0)

  def test_setup_plus_tests_never_exceeds_the_job_run_time(self) -> None:
    """A worker's setup and its test window both fit inside the runner time it consumed."""
    for flavor in ("tpu-unit", "cpu-unit", "cpu-post-training-unit"):
      for worker, job in jobs_by_worker(self.jobs, flavor).items():
        with self.subTest(flavor=flavor, worker=worker):
          setup = derive.setup_seconds(job)
          run = derive.run_seconds(job)
          span = derive.step_span(job, derive.RUN_TESTS_STEP)
          assert setup is not None and run is not None and span is not None
          tests = derive.parse_timestamp(span[1]) - derive.parse_timestamp(span[0])
          self.assertLessEqual(setup + tests.total_seconds(), run)


class MissingTimestampTest(OfflineTestCase):
  """Covers the job shapes whose timestamps do not mean what they look like."""

  def setUp(self) -> None:
    """Loads the fixtures that hold the degenerate shapes."""
    super().setUp()
    self.cancelled_jobs = read_jobs("cancelled-job-32785979907-attempt1-jobs.json")
    self.never_started_jobs = read_jobs("queued-then-cancelled-32999133815-attempt1-jobs.json")

  def queued_job(self) -> dict[str, Any]:
    """Builds the one shape no fixture could capture: a job still waiting for a runner.

    GitHub fills `started_at` and `completed_at` on every job whose status is "completed",
    including cancelled and skipped ones, so a null timestamp only exists while a job is
    queued or in progress. About 9,800 real jobs were scanned without finding one, so this
    case is synthesised from a real job object.

    Returns:
      A copy of a real tpu-unit worker with its start and finish removed.
    """
    job = copy.deepcopy(job_named(self.cancelled_jobs, "TPU Pretrain Tests (tpu-unit) / Execute Tests (1) / tpu-unit"))
    job["status"] = "queued"
    job["conclusion"] = None
    job["started_at"] = None
    job["completed_at"] = None
    job["steps"] = []
    return job

  def in_progress_job(self) -> dict[str, Any]:
    """Builds the other synthesised shape: a job that started but has not finished.

    Returns:
      A copy of a real tpu-unit worker with only its finish removed.
    """
    job = copy.deepcopy(job_named(self.cancelled_jobs, "TPU Pretrain Tests (tpu-unit) / Execute Tests (1) / tpu-unit"))
    job["status"] = "in_progress"
    job["conclusion"] = None
    job["completed_at"] = None
    return job

  def test_a_queued_job_answers_none_everywhere_and_raises_nothing(self) -> None:
    """Every helper returns None for a job with no start and no finish. Synthesised shape."""
    job = self.queued_job()
    self.assertFalse(derive.held_a_runner(job))
    self.assertFalse(derive.is_carried_over(job))
    self.assertIsNone(derive.queue_seconds(job))
    self.assertIsNone(derive.run_seconds(job))
    self.assertIsNone(derive.setup_seconds(job))
    self.assertIsNone(derive.step_span(job, derive.RUN_TESTS_STEP))
    self.assertIsNone(derive.suite_duration_seconds([job]))
    self.assertIsNone(derive.run_wall_seconds([job]))
    self.assertEqual(derive.machine_seconds([job]), 0.0)
    self.assertEqual(derive.device_lane(job), derive.LANE_TPU)

  def test_an_in_progress_job_answers_none_everywhere(self) -> None:
    """A job with a start but no finish is unmeasurable too, and still does not raise."""
    job = self.in_progress_job()
    self.assertFalse(derive.held_a_runner(job))
    self.assertIsNone(derive.queue_seconds(job))
    self.assertIsNone(derive.run_seconds(job))
    self.assertIsNone(derive.setup_seconds(job))
    self.assertIsNone(derive.suite_duration_seconds([job]))

  def test_a_cancelled_worker_with_no_steps_keeps_its_real_queue_wait(self) -> None:
    """Job 97618961480 held a runner for 1 s of wait, so its queue is real and its setup is not.

    It was cancelled with an empty steps list, so there is no "Run Tests" to measure setup or
    a test window from, and its 9435 s of run time is the time until the run was cancelled.
    """
    job = job_named(self.cancelled_jobs, "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit")
    self.assertEqual(job["conclusion"], "cancelled")
    self.assertEqual(job["steps"], [])
    self.assertTrue(derive.held_a_runner(job))
    self.assert_seconds(derive.queue_seconds(job), 1.0)
    self.assertIsNone(derive.setup_seconds(job))
    self.assertIsNone(derive.step_span(job, derive.RUN_TESTS_STEP))
    self.assertIsNone(derive.suite_duration_seconds([job]))

  def test_a_partial_suite_reports_the_workers_that_did_report(self) -> None:
    """tpu-unit was configured with 2 workers and only 1 reached "Run Tests": D is that one's."""
    suite_jobs = derive.jobs_for_flavor(self.cancelled_jobs, "tpu-unit")
    self.assertEqual(derive.worker_count(self.cancelled_jobs, "tpu-unit"), 2)
    self.assert_seconds(derive.suite_duration_seconds(suite_jobs), 1083.0)
    reporting = [w for w, job in jobs_by_worker(self.cancelled_jobs, "tpu-unit").items() if derive.setup_seconds(job)]
    self.assertEqual(reporting, [1])

  def test_a_job_that_never_held_a_runner_contributes_nothing(self) -> None:
    """Eight cancelled jobs of run 32999133815 compute to 13,843 s each on machines they never had."""
    job = job_named(self.never_started_jobs, "CPU Pretrain Tests (cpu-unit) / Execute Tests (2) / cpu-unit")
    self.assertEqual(job["created_at"], job["started_at"])
    self.assertEqual(job["steps"], [])
    self.assertFalse(derive.held_a_runner(job))
    self.assertIsNone(derive.queue_seconds(job))
    self.assertIsNone(derive.run_seconds(job))
    # The raw subtraction the module refuses to report.
    naive = derive.parse_timestamp(job["completed_at"]) - derive.parse_timestamp(job["started_at"])
    self.assertEqual(naive.total_seconds(), 13843.0)

  def test_a_flavor_whose_workers_all_died_queued_has_no_duration(self) -> None:
    """cpu-post-training-unit had 0 of 4 workers report, so its duration is None, not zero."""
    self.assertEqual(derive.worker_count(self.never_started_jobs, "cpu-post-training-unit"), 4)
    suite_jobs = derive.jobs_for_flavor(self.never_started_jobs, "cpu-post-training-unit")
    self.assertEqual(len(suite_jobs), 4)
    self.assertIsNone(derive.suite_duration_seconds(suite_jobs))
    self.assertEqual(derive.machine_seconds(suite_jobs), 0.0)
    integration = derive.jobs_for_flavor(self.never_started_jobs, "cpu-post-training-integration")
    self.assertIsNone(derive.suite_duration_seconds(integration))

  def test_the_two_enormous_queue_waits_of_that_attempt_survive(self) -> None:
    """The jobs that did get a runner kept their real waits: 12,822 s and 13,681 s."""
    cpu_integration = jobs_by_worker(self.never_started_jobs, "cpu-integration")[1]
    cpu_unit_worker_4 = jobs_by_worker(self.never_started_jobs, "cpu-unit")[4]
    self.assert_seconds(derive.queue_seconds(cpu_integration), 12822.0)
    self.assert_seconds(derive.queue_seconds(cpu_unit_worker_4), 13681.0)
    self.assert_seconds(derive.suite_duration_seconds(derive.jobs_for_flavor(self.never_started_jobs, "cpu-unit")), 113.0)

  def test_a_skipped_job_is_did_not_run_and_not_a_zero(self) -> None:
    """A skipped job has no labels, no steps and equal timestamps, so it holds no lane and no time."""
    job = job_named(read_jobs(CLEAN_RUN_JOBS), "Investigate failed build")
    self.assertEqual(job["conclusion"], "skipped")
    self.assertEqual(job["labels"], [])
    self.assertFalse(derive.held_a_runner(job))
    self.assertIsNone(derive.queue_seconds(job))
    self.assertIsNone(derive.run_seconds(job))
    self.assertEqual(derive.device_lane(job), derive.LANE_NO_RUNNER)

  def test_parse_timestamp_reads_what_github_sends_and_refuses_the_rest(self) -> None:
    """The one date parser: "Z" suffix in, UTC out; anything unreadable is None, not an error."""
    moment = derive.parse_timestamp("2026-09-01T04:08:43Z")
    assert moment is not None
    self.assertEqual(moment.isoformat(), "2026-09-01T04:08:43+00:00")
    for bad in (None, "", "   ", "not a date", 17, ["2026-09-01T04:08:43Z"]):
      with self.subTest(value=bad):
        self.assertIsNone(derive.parse_timestamp(bad))


class CarriedOverJobTest(OfflineTestCase):
  """Covers the re-run trap: 28 of 42 jobs in an attempt carry the previous attempt's clock."""

  def setUp(self) -> None:
    """Loads both attempts of run 32772626658."""
    super().setUp()
    self.attempt_1 = read_jobs("rerun-32772626658-attempt1-jobs.json")
    self.attempt_2 = read_jobs("rerun-32772626658-attempt2-jobs.json")

  def test_twenty_eight_of_forty_two_jobs_are_carried_over(self) -> None:
    """Attempt 2 lists all 42 jobs but really re-executed only 14 of them."""
    self.assertEqual(len(self.attempt_2), 42)
    carried = [job for job in self.attempt_2 if derive.is_carried_over(job)]
    self.assertEqual(len(carried), 28)
    for job in carried:
      with self.subTest(job=job["name"]):
        self.assertIsNone(derive.queue_seconds(job))
        self.assertIsNone(derive.run_seconds(job))
        self.assertFalse(derive.held_a_runner(job))

  def test_job_ids_are_never_reused_across_attempts(self) -> None:
    """Even the 28 jobs that did not re-run got fresh ids, which is why names are the identity."""
    ids_1 = {job["name"]: job["id"] for job in self.attempt_1}
    ids_2 = {job["name"]: job["id"] for job in self.attempt_2}
    self.assertEqual(set(ids_1), set(ids_2))
    shared = [name for name in ids_1 if ids_1[name] == ids_2[name]]
    self.assertEqual(shared, [])

  def test_suite_duration_uses_only_the_workers_that_re_executed(self) -> None:
    """Filtering carried-over workers turns 23,271 s of nonsense into tpu-unit's real 1358 s."""
    for flavor, expected in ATTEMPT_2_DURATIONS_32772626658.items():
      with self.subTest(flavor=flavor):
        suite_jobs = derive.jobs_for_flavor(self.attempt_2, flavor)
        self.assert_seconds(derive.suite_duration_seconds(suite_jobs), expected)

        naive_starts = []
        naive_ends = []
        for job in suite_jobs:
          span = derive.step_span(job, derive.RUN_TESTS_STEP)
          if span is not None:
            naive_starts.append(derive.parse_timestamp(span[0]))
            naive_ends.append(derive.parse_timestamp(span[1]))
        naive = (max(naive_ends) - min(naive_starts)).total_seconds()
        self.assertAlmostEqual(naive, ATTEMPT_2_NAIVE_DURATIONS_32772626658[flavor], places=3)

  def test_a_flavor_with_no_re_executed_worker_has_no_duration_for_that_attempt(self) -> None:
    """gpu-unit was listed in attempt 2 but never re-ran, so its attempt-2 duration is None."""
    for flavor in ("gpu-unit", "gpu-integration", "tpu-integration", "cpu-integration"):
      with self.subTest(flavor=flavor):
        suite_jobs = derive.jobs_for_flavor(self.attempt_2, flavor)
        self.assertTrue(suite_jobs)
        self.assertIsNone(derive.suite_duration_seconds(suite_jobs))


class WorkerCountTest(OfflineTestCase):
  """Covers `worker_count` and the name parsing it rests on."""

  def setUp(self) -> None:
    """Loads the clean run and both attempts of a re-run."""
    super().setUp()
    self.jobs = read_jobs(CLEAN_RUN_JOBS)
    self.attempt_1 = read_jobs("rerun-32772626658-attempt1-jobs.json")
    self.attempt_2 = read_jobs("rerun-32772626658-attempt2-jobs.json")

  def test_a_repeated_worker_number_counts_once(self) -> None:
    """Both attempts of cpu-unit together list 8 jobs numbered 1-4, and W is still 4."""
    both_attempts = self.attempt_1 + self.attempt_2
    self.assertEqual(len(derive.jobs_for_flavor(both_attempts, "cpu-unit")), 8)
    self.assertEqual(derive.worker_count(both_attempts, "cpu-unit"), 4)
    self.assertEqual(derive.worker_count(self.attempt_1, "cpu-unit"), 4)

  def test_zero_workers_and_no_jobs_are_different_answers(self) -> None:
    """A Pathways flavor has jobs but no "Execute Tests (N)" names; an absent flavor has neither.

    `worker_count` is a statement about job names, so both read 0. The caller tells them apart
    by asking `jobs_for_flavor` whether the flavor ran at all.
    """
    self.assertEqual(derive.worker_count(self.jobs, "tpu-pathways-unit"), 0)
    self.assertEqual(len(derive.jobs_for_flavor(self.jobs, "tpu-pathways-unit")), 2)

    self.assertEqual(derive.worker_count(self.jobs, "no-such-flavor"), 0)
    self.assertEqual(derive.jobs_for_flavor(self.jobs, "no-such-flavor"), [])

  def test_setup_parameters_jobs_are_never_counted_as_workers(self) -> None:
    """Thirteen "Setup Parameters" jobs run in the clean run and none of them is a worker."""
    setup_jobs = [job for job in self.jobs if str(job["name"]).endswith(" / Setup Parameters")]
    self.assertEqual(len(setup_jobs), 13)
    for job in setup_jobs:
      with self.subTest(job=job["name"]):
        self.assertIsNone(derive.parse_execute_tests_name(job["name"]))
        self.assertIsNone(derive.flavor_of(job))

  def test_parse_execute_tests_name_reads_the_flavor_and_the_worker(self) -> None:
    """The worker number comes from "Execute Tests (N)", never from the caller's parentheses."""
    self.assertEqual(
        derive.parse_execute_tests_name("TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit"),
        ("tpu-unit", 2),
    )
    self.assertEqual(
        derive.parse_execute_tests_name(
            "CPU Posttrain Tests (cpu-post-training-unit) / Execute Tests (4) / cpu-post-training-unit"
        ),
        ("cpu-post-training-unit", 4),
    )
    for other in (
        "TPU Pathways Unit Tests (2) / tpu-pathways-unit",
        "All Required Tests Passed",
        "CPU Pretrain Tests (cpu-unit) / Setup Parameters",
        None,
        42,
    ):
      with self.subTest(name=other):
        self.assertIsNone(derive.parse_execute_tests_name(other))

  def test_flavor_of_reads_the_last_segment_of_a_test_job_name(self) -> None:
    """Matrix workers and Pathways jobs both name their flavor last; gate jobs have no flavor."""
    self.assertEqual(
        derive.flavor_of({"name": "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit"}), "tpu-unit"
    )
    self.assertEqual(derive.flavor_of({"name": "TPU Pathways Unit Tests (2) / tpu-pathways-unit"}), "tpu-pathways-unit")
    self.assertIsNone(derive.flavor_of({"name": "All Required Tests Passed"}))
    self.assertIsNone(derive.flavor_of({"name": "Gate and Formalize Parameters"}))
    self.assertIsNone(derive.flavor_of({}))

  def test_infrastructure_jobs_with_a_slash_are_read_as_flavors(self) -> None:
    """Known defect, pinned so a fix is noticed: `flavor_of` invents flavors for infra jobs.

    The docstring promises None for "the gate and infrastructure jobs", but the rule it uses is
    the last "/" segment, and twelve infrastructure jobs of run 33468578834 have one:
    "Documentation Build Check / Build Sphinx Docs" reads as a flavor called "Build Sphinx
    Docs". They are harmless today - each yields W = 0 and D = None, so anything that gates on
    a worker count drops them - but a caller that discovers a run's flavors by mapping
    `flavor_of` over the jobs gets 12 phantom suites alongside the 15 real ones.
    """
    phantom = {
        "Documentation Build Check / Build Sphinx Docs": "Build Sphinx Docs",
        "Code Quality Check / Pre-commit Linters": "Pre-commit Linters",
        "Build MaxText Package / Build Wheel": "Build Wheel",
        "Track Test Performance / Track Test Duration": "Track Test Duration",
        "Jupyter Notebook Tests / Execute lora_llama3_demo.ipynb": "Execute lora_llama3_demo.ipynb",
    }
    for job_name, read_as in phantom.items():
      with self.subTest(job=job_name):
        job = job_named(self.jobs, job_name)
        self.assertEqual(derive.flavor_of(job), read_as)
        self.assertEqual(derive.worker_count(self.jobs, read_as), 0)
        self.assertIsNone(derive.suite_duration_seconds(derive.jobs_for_flavor(self.jobs, read_as)))

    discovered = {derive.flavor_of(job) for job in self.jobs} - {None}
    real = set(FLAVOR_TRUTH) | {"tpu-pathways-unit", "tpu-pathways-integration"}
    self.assertEqual(len(discovered & real), 15)
    self.assertEqual(len(discovered - real), 12)


class DeviceLaneTest(OfflineTestCase):
  """Covers `device_lane` over the complete real label inventory."""

  def test_every_label_the_repository_uses(self) -> None:
    """All eight real runs-on shapes, including the three the first lane map missed."""
    expected = {
        "linux-x86-ct6e-180-4tpu": derive.LANE_TPU,
        "linux-x86-ct6e-180-8tpu": derive.LANE_TPU,
        "linux-x86-tpu7x-224-4tpu": derive.LANE_TPU,
        "linux-x86-a2-48-a100-4gpu": derive.LANE_GPU,
        "linux-x86-n2-32": derive.LANE_CPU,
        "linux-x86-n2-16-buildkit": derive.LANE_BUILD,
        "ubuntu-latest": derive.LANE_HOSTED,
    }
    for label, lane in expected.items():
      with self.subTest(label=label):
        self.assertEqual(derive.device_lane({"labels": [label]}), lane)
    self.assertEqual(derive.device_lane({"labels": []}), derive.LANE_NO_RUNNER)

  def test_an_unseen_label_is_unknown_and_never_guessed_or_raised(self) -> None:
    """New hardware appears without notice, so it reports Unknown rather than falling into a lane."""
    self.assertEqual(derive.device_lane({"labels": ["linux-x86-something-new-2026"]}), derive.LANE_UNKNOWN)
    self.assertEqual(derive.device_lane({"labels": ["self-hosted", "linux-x86-n2-32"]}), derive.LANE_CPU)
    self.assertEqual(derive.device_lane({"labels": ["LINUX-X86-N2-32"]}), derive.LANE_CPU)
    self.assertEqual(derive.device_lane({}), derive.LANE_NO_RUNNER)
    self.assertEqual(derive.device_lane({"labels": None}), derive.LANE_NO_RUNNER)

  def test_the_clean_run_accounts_for_every_job(self) -> None:
    """All 54 jobs of run 33468578834 land in a lane, and no job reads Unknown."""
    jobs = read_jobs(CLEAN_RUN_JOBS)
    lanes: dict[str, int] = {}
    for job in jobs:
      lane = derive.device_lane(job)
      lanes[lane] = lanes.get(lane, 0) + 1
    self.assertEqual(sum(lanes.values()), CLEAN_RUN_JOB_COUNT)
    self.assertNotIn(derive.LANE_UNKNOWN, lanes)
    self.assertEqual(
        lanes,
        {
            derive.LANE_HOSTED: 22,
            derive.LANE_TPU: 18,
            derive.LANE_CPU: 10,
            derive.LANE_GPU: 2,
            derive.LANE_BUILD: 1,
            derive.LANE_NO_RUNNER: 1,
        },
    )


class RescueTest(OfflineTestCase):
  """Covers `find_rescues` on all three multi-attempt runs."""

  def attempts(self, *fixture_names: str) -> dict[int, list[dict[str, Any]]]:
    """Loads consecutive attempt payloads and numbers them from 1.

    Args:
      *fixture_names: Attempt fixtures in attempt order.

    Returns:
      Attempt number -> that attempt's jobs.
    """
    return {number: read_jobs(name) for number, name in enumerate(fixture_names, start=1)}

  def test_three_rescues_in_run_32772626658(self) -> None:
    """Exactly three CPU workers failed and passed on the re-run, wasting 365 s between them."""
    rescues = derive.find_rescues(
        self.attempts("rerun-32772626658-attempt1-jobs.json", "rerun-32772626658-attempt2-jobs.json")
    )
    self.assertEqual({rescue.job_name: rescue.wasted_seconds for rescue in rescues}, RESCUES_32772626658)
    self.assertAlmostEqual(sum(rescue.wasted_seconds or 0.0 for rescue in rescues), 365.0, places=3)
    for rescue in rescues:
      with self.subTest(job=rescue.job_name):
        self.assertEqual(rescue.failed_attempt, 1)
        self.assertEqual(rescue.passed_attempt, 2)
        self.assertEqual(rescue.lane, derive.LANE_CPU)
        self.assertIsNotNone(rescue.flavor)
        self.assertIsNotNone(rescue.worker)
        self.assertNotEqual(rescue.failed_job_id, rescue.passed_job_id)

  def test_a_job_that_failed_again_is_not_a_rescue(self) -> None:
    """Five job names failed in attempt 1 and failed once more in attempt 2, so none is rescued."""
    rescues = derive.find_rescues(
        self.attempts("rerun-32772626658-attempt1-jobs.json", "rerun-32772626658-attempt2-jobs.json")
    )
    names = {rescue.job_name for rescue in rescues}
    for failed_again in FAILED_AGAIN_32772626658:
      with self.subTest(job=failed_again):
        self.assertNotIn(failed_again, names)

  def test_a_job_that_failed_with_no_later_attempt_is_not_a_rescue(self) -> None:
    """Attempt 1 alone yields nothing: a failure needs a following attempt before it can pair."""
    only_attempt_1 = self.attempts("rerun-32772626658-attempt1-jobs.json")
    self.assertEqual(derive.find_rescues(only_attempt_1), [])
    failures = [job for job in only_attempt_1[1] if job.get("conclusion") == "failure"]
    self.assertEqual(len(failures), 8)

  def test_four_rescues_across_three_lanes_in_run_33037584699(self) -> None:
    """Every attempt-1 failure of the clean re-run recovered, wasting 2634 s in total."""
    rescues = derive.find_rescues(
        self.attempts("rerun-33037584699-attempt1-jobs.json", "rerun-33037584699-attempt2-jobs.json")
    )
    self.assertEqual({rescue.job_name: rescue.wasted_seconds for rescue in rescues}, RESCUES_33037584699)
    self.assertAlmostEqual(sum(rescue.wasted_seconds or 0.0 for rescue in rescues), 2634.0, places=3)
    self.assertEqual(
        {rescue.lane for rescue in rescues},
        {derive.LANE_CPU, derive.LANE_GPU, derive.LANE_TPU, derive.LANE_HOSTED},
    )
    gate = next(rescue for rescue in rescues if rescue.job_name == "All Required Tests Passed")
    self.assertIsNone(gate.flavor)
    self.assertIsNone(gate.worker)

  def test_cancelled_then_success_is_not_a_rescue(self) -> None:
    """Run 32785979907 yields zero rescues over three attempts, because cancelled is not failure."""
    attempts = self.attempts(
        "cancelled-job-32785979907-attempt1-jobs.json",
        "cancelled-job-32785979907-attempt2-jobs.json",
        "cancelled-job-32785979907-attempt3-jobs.json",
    )
    self.assertEqual(derive.find_rescues(attempts), [])
    worker = job_named(attempts[1], "TPU Pretrain Tests (tpu-unit) / Execute Tests (2) / tpu-unit")
    self.assertEqual(worker["conclusion"], "cancelled")
    self.assertEqual(job_named(attempts[2], worker["name"])["conclusion"], "success")

  def test_the_output_is_ordered_and_repeatable(self) -> None:
    """Rescues sort by failed attempt then job name, whatever order the jobs arrived in."""
    attempts = self.attempts("rerun-33037584699-attempt1-jobs.json", "rerun-33037584699-attempt2-jobs.json")
    first = derive.find_rescues(attempts)
    shuffled = {number: list(reversed(jobs)) for number, jobs in attempts.items()}
    second = derive.find_rescues(shuffled)
    self.assertEqual([rescue.job_name for rescue in first], [rescue.job_name for rescue in second])
    self.assertEqual([rescue.job_name for rescue in first], sorted(rescue.job_name for rescue in first))

  def test_no_attempts_is_an_empty_list(self) -> None:
    """Nothing to compare gives nothing, rather than an error."""
    self.assertEqual(derive.find_rescues({}), [])
    self.assertEqual(derive.find_rescues({1: []}), [])


class RescueRuleAgreesWithRowsTest(OfflineTestCase):
  """`derive.find_rescues` and `rows.rescue_rows` must never disagree about one run.

  They used to. `find_rescues` paired consecutive ATTEMPT NUMBERS and looked the name up in
  each, so a name missing from a middle attempt lost its rescue, while `rows.rescue_rows`
  paired consecutive APPEARANCES of the name and kept it. Names do go missing: attempt 1 of
  run 32785979907 lists 38 jobs where attempts 2 and 3 list 42.
  """

  def attempts(self, *fixture_names: str) -> dict[int, list[dict[str, Any]]]:
    """Loads consecutive attempt payloads and numbers them from 1."""
    return {number: read_jobs(name) for number, name in enumerate(fixture_names, start=1)}

  def test_a_name_absent_from_a_middle_attempt_still_pairs_up(self) -> None:
    """failure, absent, success is one rescue, from attempt 1 to attempt 3."""
    attempts = self.attempts("rerun-32772626658-attempt1-jobs.json", "rerun-32772626658-attempt2-jobs.json")
    name = "CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit"
    attempts[3] = attempts[2]
    attempts[2] = [job for job in attempts[2] if job.get("name") != name]

    found = {rescue.job_name: rescue for rescue in derive.find_rescues(attempts)}

    self.assertIn(name, found)
    self.assertEqual(found[name].failed_attempt, 1)
    self.assertEqual(found[name].passed_attempt, 3)

  def test_both_modules_find_the_same_pairs_on_every_multi_attempt_fixture(self) -> None:
    """The same run, read by both modules, gives the same (name, failed, passed) triples."""
    cases = [
        ("rerun-32772626658-attempt1-jobs.json", "rerun-32772626658-attempt2-jobs.json"),
        ("rerun-33037584699-attempt1-jobs.json", "rerun-33037584699-attempt2-jobs.json"),
        (
            "cancelled-job-32785979907-attempt1-jobs.json",
            "cancelled-job-32785979907-attempt2-jobs.json",
            "cancelled-job-32785979907-attempt3-jobs.json",
        ),
    ]
    for fixtures in cases:
      with self.subTest(run=fixtures[0]):
        attempts = self.attempts(*fixtures)
        run = {"id": 1}
        from_derive = {(r.job_name, r.failed_attempt, r.passed_attempt) for r in derive.find_rescues(attempts)}
        from_rows = {(r.job_name, r.failed_attempt, r.rescued_attempt) for r in row_module.rescue_rows(run, attempts)}
        self.assertEqual(from_derive, from_rows)

  def test_string_attempt_keys_are_read_the_same_as_integers(self) -> None:
    """An attempts map that has been through JSON has string keys, and used to report nothing."""
    attempts = self.attempts("rerun-32772626658-attempt1-jobs.json", "rerun-32772626658-attempt2-jobs.json")
    as_strings = {str(number): jobs for number, jobs in attempts.items()}

    from_ints = derive.find_rescues(attempts)
    from_strings = derive.find_rescues(as_strings)

    self.assertEqual(len(from_ints), 3)
    self.assertEqual(
        [(r.job_name, r.failed_attempt, r.passed_attempt) for r in from_ints],
        [(r.job_name, r.failed_attempt, r.passed_attempt) for r in from_strings],
    )

  def test_an_attempt_key_that_is_not_a_number_is_refused(self) -> None:
    """Dropping it silently is what turned three rescues into none."""
    with self.assertRaises(ValueError):
      derive.find_rescues({"first": read_jobs("rerun-32772626658-attempt1-jobs.json")})


class TestFlavorsTest(OfflineTestCase):
  """Covers `test_flavors`, which separates real suites from the jobs that merely have a "/".

  `flavor_of` reads the last name segment of any job, so mapping it over a run also yields
  "Build Wheel", "Pre-commit Linters" and seven notebook jobs. Anything that builds its
  flavor set from the jobs list has to filter them out, and this is where that lives.
  """

  def test_the_clean_run_has_fifteen_test_flavors_and_no_infrastructure_names(self) -> None:
    """Run 33468578834 runs 13 Execute Tests flavors plus the two Pathways ones."""
    jobs = read_jobs(CLEAN_RUN_JOBS)

    flavors = derive.test_flavors(jobs)

    self.assertEqual(len(flavors), 15)
    self.assertEqual(flavors, sorted(flavors))
    for flavor in FLAVOR_TRUTH:
      self.assertIn(flavor, flavors)
    self.assertIn("tpu-pathways-unit", flavors)
    self.assertIn("tpu-pathways-integration", flavors)
    for phantom in ("Build Wheel", "Build Sphinx Docs", "Pre-commit Linters", "Track Test Duration"):
      self.assertIn(phantom, {derive.flavor_of(job) for job in jobs})
      self.assertNotIn(phantom, flavors)

  def test_a_run_with_no_jobs_has_no_flavors(self) -> None:
    """The action_required run never executed, so there is nothing to name."""
    self.assertEqual(derive.test_flavors(read_jobs("action-required-run-33465601432-jobs.json")), [])

  def test_a_flavor_whose_workers_never_started_is_still_named(self) -> None:
    """The eight cancelled workers of run 32999133815 are named "Execute Tests (N)" all the same.

    The flavor ran as far as being scheduled, so it belongs in the list; its duration is None,
    which is the honest answer, and that is what the dashboard draws as "did not run".
    """
    jobs = read_jobs("queued-then-cancelled-32999133815-attempt1-jobs.json")

    flavors = derive.test_flavors(jobs)

    self.assertIn("cpu-post-training-unit", flavors)
    self.assertIsNone(derive.suite_duration_seconds(derive.jobs_for_flavor(jobs, "cpu-post-training-unit")))
    self.assertEqual(derive.worker_count(jobs, "cpu-post-training-unit"), 4)


class PhaseSplitTest(OfflineTestCase):
  """Covers `phase_split` and the summing identity its docstring promises."""

  def setUp(self) -> None:
    """Loads the clean run's jobs."""
    super().setUp()
    self.jobs = read_jobs(CLEAN_RUN_JOBS)

  def test_the_four_spans_tile_the_whole_interval(self) -> None:
    """queued + setup + tests + tail == total, and setup + tests + tail == wall, as documented."""
    split = derive.phase_split(self.jobs)
    self.assertTrue(split.parts_sum_to_total)
    self.assert_seconds(split.queued_seconds, 2.0)
    self.assert_seconds(split.setup_seconds, 214.0)
    self.assert_seconds(split.tests_seconds, 2231.0)
    self.assert_seconds(split.tail_seconds, 25.0)
    self.assert_seconds(split.total_seconds, 2472.0)
    self.assert_seconds(split.wall_seconds, CLEAN_RUN_WALL_SECONDS)

    parts = (split.queued_seconds, split.setup_seconds, split.tests_seconds, split.tail_seconds)
    assert all(part is not None for part in parts)
    self.assertAlmostEqual(sum(parts), split.total_seconds, places=3)
    self.assertAlmostEqual(sum(parts[1:]), split.wall_seconds, places=3)

  def test_the_boundary_moments_are_reported_with_the_spans(self) -> None:
    """Every span is cut at a named moment, so a reader can check the arithmetic by hand."""
    split = derive.phase_split(self.jobs)
    self.assertEqual(split.first_created_at, "2026-09-01T04:06:01Z")
    self.assertEqual(split.first_started_at, "2026-09-01T04:06:03Z")
    self.assertEqual(split.tests_started_at, "2026-09-01T04:09:37Z")
    self.assertEqual(split.tests_completed_at, "2026-09-01T04:46:48Z")
    self.assertEqual(split.last_completed_at, "2026-09-01T04:47:13Z")
    self.assertEqual(split.jobs_counted, CLEAN_RUN_JOB_COUNT)
    self.assertEqual(split.jobs_with_tests, 23)
    self.assertEqual(split.jobs_ignored, 0)

  def test_a_single_flavor_split_agrees_with_its_suite_duration(self) -> None:
    """Handed one flavor's jobs, `tests_seconds` is that suite's D and the identity still holds."""
    for flavor in ("tpu-unit", "cpu-unit"):
      with self.subTest(flavor=flavor):
        suite_jobs = derive.jobs_for_flavor(self.jobs, flavor)
        split = derive.phase_split(suite_jobs)
        self.assertTrue(split.parts_sum_to_total)
        self.assertEqual(split.tests_seconds, derive.suite_duration_seconds(suite_jobs))
        self.assert_seconds(split.tests_seconds, FLAVOR_TRUTH[flavor]["duration"])

  def test_carried_over_jobs_are_counted_out_loud(self) -> None:
    """Attempt 2 of run 32772626658 ignores 28 jobs and says so instead of dropping them quietly."""
    split = derive.phase_split(read_jobs("rerun-32772626658-attempt2-jobs.json"))
    self.assertEqual(split.jobs_ignored, 28)
    self.assertEqual(split.jobs_counted, 14)
    self.assertEqual(split.jobs_with_tests, 5)
    self.assert_seconds(split.tests_seconds, 1358.0)
    self.assertTrue(split.parts_sum_to_total)

  def test_an_unmeasurable_run_reports_none_rather_than_zero(self) -> None:
    """An empty jobs list, and a run whose jobs never started, both give None spans."""
    for jobs in ([], read_jobs("action-required-run-33465601432-jobs.json")):
      with self.subTest(job_count=len(jobs)):
        split = derive.phase_split(jobs)
        self.assertFalse(split.parts_sum_to_total)
        self.assertIsNone(split.queued_seconds)
        self.assertIsNone(split.tests_seconds)
        self.assertIsNone(split.total_seconds)
        self.assertEqual(split.jobs_counted, 0)

  def test_no_span_is_ever_negative(self) -> None:
    """A cancelled job's steps outlive its own `completed_at`, and the tail must survive that.

    This was a reported defect. GitHub stamps a cancelled job complete at the moment the
    cancellation is issued, while the steps already running keep going. In run 32999133815 the
    job "CPU Pretrain Tests (cpu-integration) / Execute Tests (1) / cpu-integration" reads
    `completed_at` 22:30:50 with its "Run Tests" step ending at 22:31:19, so a tail measured
    from `completed_at` alone came out as -29 s - a duration a stacked bar would draw as a
    negative segment. Reading a job's end as the later of the two clocks fixes it.
    """
    for name in (CLEAN_RUN_JOBS, "queued-then-cancelled-32999133815-attempt1-jobs.json"):
      with self.subTest(fixture=name):
        split = derive.phase_split(read_jobs(name))
        for span in (split.queued_seconds, split.setup_seconds, split.tests_seconds, split.tail_seconds):
          assert span is not None
          self.assertGreaterEqual(span, 0.0)

  def test_the_cancelled_run_split_is_pinned_to_the_measured_values(self) -> None:
    """Run 32999133815 attempt 1, measured with the job end covering its own steps.

    The last job clock reads 22:30:50 and the last step finishes at 22:31:23, so the run's end
    is 22:31:23 and the tail is the 4 s between the last test finishing and that.
    """
    split = derive.phase_split(read_jobs("queued-then-cancelled-32999133815-attempt1-jobs.json"))
    self.assert_seconds(split.queued_seconds, 34.0)
    self.assert_seconds(split.setup_seconds, 2067.0)
    self.assert_seconds(split.tests_seconds, 12582.0)
    self.assert_seconds(split.tail_seconds, 4.0)
    self.assert_seconds(split.total_seconds, 14687.0)
    self.assertTrue(split.parts_sum_to_total)
    self.assertEqual(split.tests_completed_at, "2026-08-26T22:31:19Z")
    self.assertEqual(split.last_completed_at, "2026-08-26T22:31:23Z")

  def test_parts_sum_to_total_refuses_a_negative_span(self) -> None:
    """The sum alone accepted a -29 s tail, because four spans cut at shared moments always add up.

    So the identity has to test the signs as well, or it is not a guard at all.
    """
    drifting = derive.PhaseSplit(
        queued_seconds=34.0,
        setup_seconds=2067.0,
        tests_seconds=12582.0,
        tail_seconds=-29.0,
        total_seconds=14654.0,
        wall_seconds=14620.0,
        first_created_at=None,
        first_started_at=None,
        tests_started_at=None,
        tests_completed_at=None,
        last_completed_at=None,
        jobs_counted=1,
        jobs_with_tests=1,
        jobs_ignored=0,
    )

    self.assertAlmostEqual(34.0 + 2067.0 + 12582.0 - 29.0, drifting.total_seconds, places=3)
    self.assertFalse(drifting.parts_sum_to_total)


class OneWorkerSuiteFitsInsideItsJobTest(OfflineTestCase):
  """setup + D can never exceed the run time of the single worker that produced both.

  This is the second symptom of the job-clock-versus-step-clock defect and it had no guard at
  all. On run 32999133815 attempt 1, cpu-integration read setup 76 s and D 976 s against a run
  time of 1023 s - 29 s more machine time than the machine ever held.
  """

  def test_every_single_worker_suite_in_every_fixture(self) -> None:
    """Sweeps every jobs fixture, so a future clock change cannot reintroduce the overlap."""
    checked = 0
    for name in sorted(path.name for path in FIXTURES.iterdir()):
      if "jobs" not in name or not name.endswith(".json"):
        continue
      jobs = read_jobs(name)
      for flavor in derive.test_flavors(jobs):
        workers = [job for job in derive.jobs_for_flavor(jobs, flavor) if derive.held_a_runner(job)]
        if len(workers) != 1:
          continue
        setup = derive.setup_seconds(workers[0])
        duration = derive.suite_duration_seconds(workers)
        run_time = derive.run_seconds(workers[0])
        if setup is None or duration is None or run_time is None:
          continue
        with self.subTest(fixture=name, flavor=flavor):
          self.assertLessEqual(setup + duration, run_time)
        checked += 1
    self.assertGreater(checked, 20)

  def test_the_case_that_used_to_overflow(self) -> None:
    """cpu-integration of run 32999133815: the worker's end now covers the steps it ran."""
    jobs = read_jobs("queued-then-cancelled-32999133815-attempt1-jobs.json")
    workers = [job for job in derive.jobs_for_flavor(jobs, "cpu-integration") if derive.held_a_runner(job)]

    self.assertEqual(len(workers), 1)
    self.assert_seconds(derive.setup_seconds(workers[0]), 76.0)
    self.assert_seconds(derive.suite_duration_seconds(workers), 976.0)
    self.assert_seconds(derive.run_seconds(workers[0]), 1056.0)
    self.assertEqual(workers[0]["completed_at"], "2026-08-26T22:30:50Z")


class MachineAndWallTest(OfflineTestCase):
  """Covers `machine_seconds` and `run_wall_seconds`, the two whole-run totals."""

  def setUp(self) -> None:
    """Loads the clean run's jobs."""
    super().setUp()
    self.jobs = read_jobs(CLEAN_RUN_JOBS)

  def test_the_clean_run_totals(self) -> None:
    """54 jobs, 2470 s of clock time and 23,335 s of runner time: parallel work adds up, clock does not."""
    self.assertEqual(len(self.jobs), CLEAN_RUN_JOB_COUNT)
    self.assert_seconds(derive.run_wall_seconds(self.jobs), CLEAN_RUN_WALL_SECONDS)
    self.assertAlmostEqual(derive.machine_seconds(self.jobs), CLEAN_RUN_MACHINE_SECONDS, places=3)
    self.assertGreater(CLEAN_RUN_MACHINE_SECONDS, CLEAN_RUN_WALL_SECONDS * 9)

  def test_machine_seconds_of_an_empty_or_idle_set_is_a_real_zero(self) -> None:
    """No job held a runner is a true answer about machine time, so it is 0.0 and not None."""
    self.assertEqual(derive.machine_seconds([]), 0.0)
    self.assertEqual(derive.machine_seconds(read_jobs("action-required-run-33465601432-jobs.json")), 0.0)
    self.assertIsNone(derive.run_wall_seconds([]))
    self.assertIsNone(derive.run_wall_seconds(read_jobs("action-required-run-33465601432-jobs.json")))

  def test_machine_seconds_ignores_time_on_machines_that_were_never_held(self) -> None:
    """The eight never-started jobs of run 32999133815 would add 110,738 s of imaginary runner time."""
    jobs = read_jobs("queued-then-cancelled-32999133815-attempt1-jobs.json")
    honest = derive.machine_seconds(jobs)
    naive = 0.0
    for job in jobs:
      started = derive.parse_timestamp(job.get("started_at"))
      completed = derive.parse_timestamp(job.get("completed_at"))
      if started is not None and completed is not None:
        naive += (completed - started).total_seconds()
    self.assertGreater(naive - honest, 110000.0)


class SlowestTestsTest(OfflineTestCase):
  """Covers `slowest_tests`, which decides what test history a run keeps."""

  def test_ties_are_broken_by_name_then_classname_so_the_order_is_stable(self) -> None:
    """Synthesised rows: four tests share 1.0 s, and the order does not depend on input order.

    No fixture holds tied durations, so the rows are built here. The point of the test is the
    total order: ranked by duration descending, then test name, then class name.
    """
    rows = [
        test_row("cpu-unit", "b_test", "tests.unit.z_test.ZTest", 1.0),
        test_row("cpu-unit", "a_test", "tests.unit.y_test.YTest", 1.0),
        test_row("cpu-unit", "a_test", "tests.unit.x_test.XTest", 1.0),
        test_row("cpu-unit", "c_test", "tests.unit.w_test.WTest", 2.0),
        test_row("cpu-unit", "d_test", "tests.unit.v_test.VTest", None),
    ]
    expected = [
        ("c_test", "tests.unit.w_test.WTest"),
        ("a_test", "tests.unit.x_test.XTest"),
        ("a_test", "tests.unit.y_test.YTest"),
        ("b_test", "tests.unit.z_test.ZTest"),
        ("d_test", "tests.unit.v_test.VTest"),
    ]
    for order in (rows, list(reversed(rows)), rows[2:] + rows[:2]):
      with self.subTest(order=[row["name"] for row in order]):
        kept = derive.slowest_tests(order, per_flavor=10)
        self.assertEqual([(row["name"], row["classname"]) for row in kept], expected)

  def test_rows_with_no_duration_sink_below_every_timed_row(self) -> None:
    """An unreadable duration is not treated as instant and is not dropped either."""
    rows = [
        test_row("cpu-unit", "untimed", "tests.unit.a_test.ATest", None),
        test_row("cpu-unit", "timed", "tests.unit.b_test.BTest", 0.001),
        test_row("cpu-unit", "unparsable", "tests.unit.c_test.CTest", "not a number"),
    ]
    kept = derive.slowest_tests(rows, per_flavor=10)
    # Both untimed rows fall to the bottom and are then ordered by test name.
    self.assertEqual([row["name"] for row in kept], ["timed", "unparsable", "untimed"])

  def test_each_flavor_is_capped_separately_and_flavors_come_out_sorted(self) -> None:
    """The cap is per flavor, and the flavors are emitted in alphabetical order."""
    rows = [test_row("tpu-unit", f"t{index}", "tests.unit.t_test.TTest", float(index)) for index in range(5)]
    rows += [test_row("cpu-unit", f"c{index}", "tests.unit.c_test.CTest", float(index)) for index in range(5)]
    kept = derive.slowest_tests(rows, per_flavor=2)
    self.assertEqual(
        [(row["flavor"], row["name"]) for row in kept],
        [
            ("cpu-unit", "c4"),
            ("cpu-unit", "c3"),
            ("tpu-unit", "t4"),
            ("tpu-unit", "t3"),
        ],
    )

  def test_it_reads_row_objects_as_well_as_mappings(self) -> None:
    """`rows.TestRow` objects and plain dictionaries rank identically."""
    made = [
        row_module.TestRow(
            run_id=33468578834,
            attempt=1,
            suite_id="tpu-unit",
            flavor="tpu-unit",
            worker=1,
            classname="tests.unit.attention_test.AttentionTest",
            name=f"test_{index}",
            status="passed",
            duration=float(index),
        )
        for index in range(3)
    ]
    kept = derive.slowest_tests(made, per_flavor=2)
    self.assertEqual([row.name for row in kept], ["test_2", "test_1"])

  def test_the_default_cap_is_twenty_five_per_flavor(self) -> None:
    """The documented default is what the collector stores per flavor per run."""
    self.assertEqual(derive.DEFAULT_SLOWEST_PER_FLAVOR, 25)
    rows = [test_row("cpu-unit", f"t{index:03d}", "tests.unit.c_test.CTest", float(index)) for index in range(60)]
    self.assertEqual(len(derive.slowest_tests(rows)), 25)

  def test_a_cap_below_one_is_refused_rather_than_returning_nothing(self) -> None:
    """`per_flavor=0` would silently throw the history away, so it raises instead."""
    rows = [test_row("cpu-unit", "t", "tests.unit.c_test.CTest", 1.0)]
    for bad in (0, -1):
      with self.subTest(per_flavor=bad):
        with self.assertRaises(ValueError):
          derive.slowest_tests(rows, per_flavor=bad)

  def test_no_rows_is_an_empty_list(self) -> None:
    """Nothing in, nothing out, and no error."""
    self.assertEqual(derive.slowest_tests([]), [])


if __name__ == "__main__":
  unittest.main(verbosity=2)
