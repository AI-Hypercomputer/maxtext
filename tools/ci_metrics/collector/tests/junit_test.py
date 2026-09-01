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

"""Offline unit tests for `collector.junit`.

Every test in this file runs against the saved fixtures in `tests/fixtures/` and never
touches the network: the base test case replaces `socket.socket`, so a test that reached for
GitHub would fail instead of hanging or hitting the rate limit. The GitHub calls
(`list_test_artifacts`, `read_run_tests`) are exercised through a stub client that serves the
saved payloads and zips built in memory.

The expected numbers are measured facts about run 33468578834 of the "MaxText Package Tests"
pipeline (and one failing job of run 33467756955), not round numbers picked for the test. The
three rules the module promises are checked head on:

  1. A suite with nothing to show is None plus a reason code, never a count of zero, and a
     flavor that lost only some of its workers says so instead of passing a partial total off
     as a complete one.
  2. The test count is the number of `<testcase>` elements minus the skipped ones; the
     `<testsuite tests>` attribute disagrees on two of the seven fixtures and is only ever
     used as a cross-check.
  3. The `decoupled` pass is its own suite, nested inside `cpu-unit` worker 1, and its tests
     are never added into the `cpu-unit` totals.

The tests are plain `unittest`, so they need nothing but the standard library. pytest
collects them too, because it understands `unittest.TestCase` and the file is named the way
the repository's pytest.ini expects.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/junit_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/junit_test.py
"""

from __future__ import annotations

import io
import json
import socket
import sys
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock
from xml.etree import ElementTree

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import junit

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The run every fixture except the failed-run file was captured from.
RUN_ID = 33468578834
FAILED_RUN_ID = 33467756955

# Measured counts of each JUnit fixture: collected `<testcase>` elements, the skipped ones,
# executed = collected - skipped, the failure and error children, the summed per-case `time`
# attributes, and the two `<testsuite>` attributes kept only for cross-checking.
FIXTURE_TRUTH = {
    "cpu-unit-1.xml": {
        "collected": 737,
        "skipped": 17,
        "executed": 720,
        "failed": 0,
        "errored": 0,
        "junit_seconds": 1055.921,
        "reported_tests": 870,
        "suite_seconds": 177.756,
    },
    "cpu-unit-3.xml": {
        "collected": 737,
        "skipped": 737,
        "executed": 0,
        "failed": 0,
        "errored": 0,
        "junit_seconds": 2.371,
        "reported_tests": 737,
        "suite_seconds": 84.677,
    },
    "decoupled-targeted.xml": {
        "collected": 54,
        "skipped": 4,
        "executed": 50,
        "failed": 0,
        "errored": 0,
        "junit_seconds": 21.097,
        "reported_tests": 54,
        "suite_seconds": 39.469,
    },
    "cpu-post-training-unit-4.xml": {
        "collected": 84,
        "skipped": 7,
        "executed": 77,
        "failed": 0,
        "errored": 0,
        "junit_seconds": 21.686,
        "reported_tests": 86,
        "suite_seconds": 125.783,
    },
    "gpu-integration-1.xml": {
        "collected": 26,
        "skipped": 15,
        "executed": 11,
        "failed": 0,
        "errored": 0,
        "junit_seconds": 349.0,
        "reported_tests": 26,
        "suite_seconds": 440.358,
    },
    "tpu-post-training-integration-1.xml": {
        "collected": 9,
        "skipped": 7,
        "executed": 2,
        "failed": 0,
        "errored": 0,
        "junit_seconds": 69.15,
        "reported_tests": 9,
        "suite_seconds": 132.255,
    },
    "tpu-post-training-integration-1.failed-run-33467756955.xml": {
        "collected": 9,
        "skipped": 7,
        "executed": 2,
        "failed": 1,
        "errored": 0,
        "junit_seconds": 69.996,
        "reported_tests": 9,
        "suite_seconds": 133.572,
    },
}

FIXTURE_NAMES = tuple(FIXTURE_TRUTH)

# The one failing test in the whole capture, quoted from the failed run's XML.
FAILED_FIXTURE = "tpu-post-training-integration-1.failed-run-33467756955.xml"
FAILED_CLASSNAME = "tests.post_training.integration.maxtext_engine_grpo_loss_test.MaxTextEngineGrpoLossTest"
FAILED_NAME = "test_grpo_loss_drives_a_training_step"
FAILED_MESSAGE = "AssertionError: nan not greater than 0.0 : no parameter moved, so no training happened"

# Parallel workers per flavor in run 33468578834, counted from the 20 "Execute Tests (N)"
# jobs. The artifact names must agree with this.
WORKERS_PER_FLAVOR = {
    "cpu-integration": 1,
    "cpu-post-training-integration": 1,
    "cpu-post-training-unit": 4,
    "cpu-unit": 4,
    "gpu-integration": 1,
    "gpu-unit": 1,
    "tpu-integration": 1,
    "tpu-post-training-integration": 1,
    "tpu-post-training-unit": 1,
    "tpu-unit": 2,
    "tpu7x-integration": 1,
    "tpu7x-post-training-unit": 1,
    "tpu7x-unit": 1,
}


def read_fixture(name: str) -> bytes:
  """Returns the raw bytes of one saved fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The file contents.
  """
  return (FIXTURES / name).read_bytes()


def parse_fixture(name: str) -> junit.SuiteResult:
  """Parses one saved JUnit fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The suite result, parsed fresh so that tests never share a mutable row.
  """
  return junit.parse_junit_xml(read_fixture(name), file_name=name)


def make_zip(members: dict[str, bytes]) -> bytes:
  """Builds an artifact zip in memory.

  Args:
    members: Member name -> file bytes.

  Returns:
    The zip bytes, as `GitHubClient.get_bytes` would return them.
  """
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as archive:
    for member_name, payload in members.items():
      archive.writestr(member_name, payload)
  return buffer.getvalue()


def artifact_payload(name: str, artifact_id: int, expired: bool = False) -> dict:
  """Builds one entry of the artifacts endpoint.

  Args:
    name: The artifact name.
    artifact_id: The numeric artifact id, also used to build the download URL.
    expired: Whether GitHub has already deleted the payload.

  Returns:
    A payload shaped like the real endpoint's entries.
  """
  return {
      "id": artifact_id,
      "name": name,
      "expired": expired,
      "size_in_bytes": 17613,
      "archive_download_url": f"https://api.github.com/artifacts/{artifact_id}/zip",
      "created_at": "2026-09-01T04:15:40Z",
      "expires_at": "2026-09-02T04:15:39Z",
  }


def download_url(artifact_id: int) -> str:
  """Returns the download URL `artifact_payload` builds for one artifact id.

  Args:
    artifact_id: The numeric artifact id.

  Returns:
    The absolute URL the stub client serves that artifact's zip from.
  """
  return f"https://api.github.com/artifacts/{artifact_id}/zip"


class StubClient:
  """A stand-in for `github.GitHubClient` that serves saved payloads.

  It satisfies the `GitHubClientLike` protocol and records what it was asked for, so a test
  can prove which endpoint the module called and which artifacts it downloaded.
  """

  def __init__(self, payloads: list[dict], blobs: dict[str, bytes] | None = None) -> None:
    """Stores what the stub will serve.

    Args:
      payloads: What `paginate` returns for the artifacts endpoint.
      blobs: Download URL -> zip bytes for `get_bytes`.
    """
    self.payloads = payloads
    self.blobs = blobs or {}
    self.paginate_calls: list[tuple[str, str, dict]] = []
    self.downloads: list[str] = []

  def paginate(self, path: str, key: str, **params: Any) -> list:
    """Returns the saved artifact payloads and records the call.

    Args:
      path: The endpoint path the module asked for.
      key: The list key inside the response.
      **params: Query parameters.

    Returns:
      The saved payload list.
    """
    self.paginate_calls.append((path, key, dict(params)))
    return self.payloads

  def get_bytes(self, url: str) -> bytes:
    """Returns the saved zip for one download URL and records the call.

    Args:
      url: The absolute download URL.

    Returns:
      The zip bytes.

    Raises:
      KeyError: The test did not stage a body for this URL.
    """
    self.downloads.append(url)
    return self.blobs[url]


class FailingClient(StubClient):
  """A stub whose download always fails, to model a transport error."""

  def get_bytes(self, url: str) -> bytes:
    """Raises the kind of error a transport layer would raise.

    Args:
      url: The download URL.

    Returns:
      Never returns.

    Raises:
      OSError: Always.
    """
    raise OSError("connection reset by peer")


def cpu_unit_worker_1_zip() -> bytes:
  """Rebuilds artifact 9785903147, which held the cpu-unit worker 1 and decoupled files.

  Returns:
    The zip bytes.
  """
  return make_zip(
      {
          "test-results-cpu-unit-1.xml": read_fixture("cpu-unit-1.xml"),
          "test-results-decoupled-targeted.xml": read_fixture("decoupled-targeted.xml"),
      }
  )


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

  def assert_every_row_is_skipped(self, result: junit.SuiteResult) -> None:
    """Asserts every row of a result is a skip carrying no failure text.

    Args:
      result: The parsed suite.
    """
    for row in result.tests:
      self.assertEqual(row.status, junit.STATUS_SKIPPED)
      self.assertIsNone(row.failure_message)


class ArtifactNameTest(OfflineTestCase):
  """Covers `parse_artifact_name`, which decides what is one of our uploads."""

  def test_reads_a_flavor_that_contains_hyphens(self) -> None:
    """A four-word flavor is read from the right, so the hyphens inside it survive."""
    self.assertEqual(
        junit.parse_artifact_name("test-results-cpu-post-training-unit-3-33468578834"),
        ("cpu-post-training-unit", 3),
    )
    self.assertEqual(
        junit.parse_artifact_name("test-results-tpu-post-training-integration-1-33468578834"),
        ("tpu-post-training-integration", 1),
    )

  def test_reads_a_single_word_flavor(self) -> None:
    """A flavor with no hyphen at all still parses, worker and run id included."""
    self.assertEqual(junit.parse_artifact_name("test-results-decoupled-1-33468578834"), ("decoupled", 1))
    self.assertEqual(junit.parse_artifact_name("test-results-gpu-unit-1-33468578834"), ("gpu-unit", 1))

  def test_ignores_uploads_that_are_not_ours(self) -> None:
    """Anything without the `test-results-` prefix is dropped.

    That is how the wheel and the notebook outputs are filtered out of the same response.
    """
    for name in (
        "maxtext-wheel",
        "notebook-outputs-sft_llama3_demo_tpu.ipynb-v6e-8",
        "notebook-outputs-lora_llama3_demo.ipynb-v6e-8",
        "coverage-report-cpu-unit-1-33468578834",
        "",
    ):
      with self.subTest(name=name):
        self.assertIsNone(junit.parse_artifact_name(name))

  def test_rejects_malformed_names(self) -> None:
    """A name that starts like ours but does not end in worker and run id is not guessed at."""
    for name in (
        "test-results-cpu-unit-33468578834",  # worker number missing
        "test-results-cpu-unit-x-33468578834",  # worker is not a number
        "test-results-cpu-unit-1-run33468578834",  # run id is not a number
        "test-results-cpu-unit-1-33468578834-extra",  # an extra field on the end
        "test-results-1-2",  # nothing left for the flavor
        "test-results--1-33468578834",  # empty flavor
        "test-results-",  # prefix only
        "TEST-RESULTS-cpu-unit-1-33468578834",  # wrong case, so not our prefix
    ):
      with self.subTest(name=name):
        self.assertIsNone(junit.parse_artifact_name(name))

  def test_agrees_with_the_saved_run(self) -> None:
    """Exactly the 20 test-result artifacts of run 33468578834 parse, with the right workers.

    The worker numbers must match the 20 "Execute Tests (N)" jobs of the same run.
    """
    payload = json.loads(read_fixture("artifacts.json"))
    parsed = {a["name"]: junit.parse_artifact_name(a["name"]) for a in payload["artifacts"]}

    kept = {name: value for name, value in parsed.items() if value is not None}
    self.assertEqual(len(payload["artifacts"]), 28)
    self.assertEqual(len(kept), 20)
    self.assertEqual(
        sorted(name for name, value in parsed.items() if value is None),
        [
            "maxtext-wheel",
            "notebook-outputs-dpo_qwen3_demo_no_eval.ipynb-v6e-8",
            "notebook-outputs-lora_llama3_demo.ipynb-v6e-8",
            "notebook-outputs-native_lora_demo.ipynb-v6e-8",
            "notebook-outputs-rl_llama3_demo.ipynb-v6e-8",
            "notebook-outputs-sft_llama3_demo_tpu.ipynb-v6e-8",
            "notebook-outputs-sft_multimodal_gemma3_demo.ipynb-v6e-8",
            "notebook-outputs-sft_qwen3_demo.ipynb-v6e-8",
        ],
    )

    workers: dict[str, list[int]] = {}
    for flavor, worker in kept.values():
      workers.setdefault(flavor, []).append(worker)
    self.assertEqual({flavor: len(numbers) for flavor, numbers in workers.items()}, WORKERS_PER_FLAVOR)
    self.assertEqual(sorted(workers["cpu-unit"]), [1, 2, 3, 4])
    self.assertEqual(sorted(workers["tpu-unit"]), [1, 2])


class ParseJUnitXmlTest(OfflineTestCase):
  """Covers `parse_junit_xml` against the seven saved files."""

  def test_counts_match_the_saved_files(self) -> None:
    """Every count comes out exactly as measured on the real file."""
    for name in FIXTURE_NAMES:
      with self.subTest(fixture=name):
        truth = FIXTURE_TRUTH[name]
        result = parse_fixture(name)

        self.assertEqual(result.collected, truth["collected"])
        self.assertEqual(result.skipped, truth["skipped"])
        self.assertEqual(result.executed, truth["executed"])
        self.assertEqual(result.failed, truth["failed"])
        self.assertEqual(result.errored, truth["errored"])
        self.assertEqual(round(result.junit_seconds, 3), truth["junit_seconds"])
        self.assertEqual(result.reported_tests, truth["reported_tests"])
        self.assertEqual(result.suite_seconds, truth["suite_seconds"])
        self.assertEqual(result.files, (name,))
        self.assertEqual(len(result.tests), result.collected)

  def test_executed_is_collected_minus_skipped(self) -> None:
    """The rule holds on all seven files, including the one where nothing ran."""
    for name in FIXTURE_NAMES:
      with self.subTest(fixture=name):
        result = parse_fixture(name)
        self.assertEqual(result.executed, result.collected - result.skipped)
        self.assertEqual(sum(1 for row in result.tests if row.status == junit.STATUS_SKIPPED), result.skipped)

  def test_each_file_is_one_pytest_suite_from_one_runner_pod(self) -> None:
    """Each fixture carries the runner pod name and the ISO-8601 start time pytest wrote."""
    for name in FIXTURE_NAMES:
      with self.subTest(fixture=name):
        result = parse_fixture(name)
        self.assertIsNotNone(result.hostname)
        self.assertTrue(result.hostname.startswith("linux-x86-"))
        self.assertIsNotNone(result.timestamp)
        self.assertTrue(result.timestamp.startswith("2026-09-01T"))
        self.assertTrue(result.timestamp.endswith("+00:00"))

  def test_the_testsuite_tests_attribute_is_only_a_cross_check(self) -> None:
    """Two of the seven files disagree with their own `<testsuite tests>` attribute.

    That is why the counted elements are the test count and the attribute is kept beside them.
    """
    parsed = {name: parse_fixture(name) for name in FIXTURE_NAMES}
    liars = {name: result for name, result in parsed.items() if not result.count_matches_attribute}

    self.assertEqual(sorted(liars), ["cpu-post-training-unit-4.xml", "cpu-unit-1.xml"])
    self.assertEqual(liars["cpu-unit-1.xml"].reported_tests, 870)
    self.assertEqual(liars["cpu-unit-1.xml"].collected, 737)
    self.assertEqual(liars["cpu-post-training-unit-4.xml"].reported_tests, 86)
    self.assertEqual(liars["cpu-post-training-unit-4.xml"].collected, 84)

  def test_summed_case_times_are_not_the_suite_time(self) -> None:
    """The per-case seconds and the `<testsuite time>` are two different numbers.

    Neither may be read as the suite's wall-clock duration.
    """
    for name in FIXTURE_NAMES:
      with self.subTest(fixture=name):
        result = parse_fixture(name)
        self.assertIsNotNone(result.suite_seconds)
        self.assertNotEqual(round(result.junit_seconds, 3), result.suite_seconds)

  def test_rows_keep_the_classnames_pytest_wrote(self) -> None:
    """The busiest file carries 144 distinct classnames, one of them empty.

    The empty one is a module-level collection skip, and the rows stay in document order.
    """
    result = parse_fixture("cpu-unit-1.xml")
    classnames = [row.classname for row in result.tests]

    self.assertEqual(len(set(classnames)), 144)
    self.assertEqual(sum(1 for value in classnames if value == ""), 1)
    self.assertNotEqual(result.tests[0].name, "")


class AllSkippedTest(OfflineTestCase):
  """Zero executed tests is a real result, not a missing file."""

  def test_the_file_reports_zero_executed_but_still_has_a_result(self) -> None:
    """cpu-unit worker 3 collected 737 tests and ran none of them.

    That is a real result with executed 0, so the parser must return counts rather than
    nothing.
    """
    result = parse_fixture("cpu-unit-3.xml")

    self.assertIsNotNone(result)
    self.assertEqual(result.collected, 737)
    self.assertEqual(result.skipped, 737)
    self.assertEqual(result.executed, 0)
    self.assertEqual(result.failed, 0)
    self.assert_every_row_is_skipped(result)
    self.assertEqual(round(result.junit_seconds, 3), 2.371)

  def test_an_all_skipped_suite_is_not_reported_as_a_missing_file(self) -> None:
    """Through the run reader, an all-skipped upload has a result and no reason code.

    A flavor that published nothing has a reason code and no result. The two must not look
    the same to the dashboard.
    """
    payloads = [artifact_payload("test-results-cpu-unit-3-33468578834", 3)]
    blobs = {download_url(3): make_zip({"test-results-cpu-unit-3.xml": read_fixture("cpu-unit-3.xml")})}
    run = junit.read_run_tests(StubClient(payloads, blobs), RUN_ID, flavors=["cpu-unit", "gpu-unit"])

    ran_nothing = run.result_for("cpu-unit")
    self.assertIsNotNone(ran_nothing)
    self.assertEqual(ran_nothing.executed, 0)
    self.assertEqual(ran_nothing.collected, 737)
    self.assertIsNone(run.reason_for("cpu-unit"))

    self.assertIsNone(run.result_for("gpu-unit"))
    self.assertEqual(run.reason_for("gpu-unit"), junit.REASON_NO_FILE)


class DecoupledTest(OfflineTestCase):
  """The nested pass is its own suite and is never added into its parent."""

  def test_the_file_is_recognised_as_its_own_nested_suite(self) -> None:
    """The file name decides the suite, and it names cpu-unit as the flavor it runs inside."""
    self.assertEqual(
        junit.suite_id_for_file("test-results-decoupled-targeted.xml", "cpu-unit"), ("decoupled", "cpu-unit")
    )
    self.assertEqual(
        junit.suite_id_for_file("some/dir/test-results-decoupled-targeted.xml", "cpu-unit"), ("decoupled", "cpu-unit")
    )
    self.assertEqual(junit.suite_id_for_file("test-results-cpu-unit-1.xml", "cpu-unit"), ("cpu-unit", None))
    self.assertEqual(junit.NESTED_SUITES["decoupled"], "cpu-unit")

  def test_it_is_parsed_apart_from_the_cpu_unit_file_in_the_same_artifact(self) -> None:
    """Artifact 9785903147 holds two XML files.

    They must come back as two suites with their own counts, never merged into one.
    """
    suites = junit.parse_artifact_zip(cpu_unit_worker_1_zip(), "cpu-unit", "test-results-cpu-unit-1-33468578834")

    self.assertEqual(sorted(suites), ["cpu-unit", "decoupled"])
    self.assertEqual(suites["cpu-unit"].collected, 737)
    self.assertEqual(suites["cpu-unit"].executed, 720)
    self.assertEqual(suites["cpu-unit"].files, ("test-results-cpu-unit-1.xml",))
    self.assertEqual(suites["decoupled"].collected, 54)
    self.assertEqual(suites["decoupled"].executed, 50)
    self.assertEqual(suites["decoupled"].files, ("test-results-decoupled-targeted.xml",))

  def test_its_tests_are_never_added_into_the_cpu_unit_totals(self) -> None:
    """The same 50 tests also run in cpu-unit's normal pass.

    Adding the two suites would double count them, so cpu-unit's totals must be the cpu-unit
    files alone.
    """
    payloads = [
        artifact_payload("test-results-cpu-unit-1-33468578834", 1),
        artifact_payload("test-results-cpu-unit-3-33468578834", 3),
    ]
    blobs = {
        download_url(1): cpu_unit_worker_1_zip(),
        download_url(3): make_zip({"test-results-cpu-unit-3.xml": read_fixture("cpu-unit-3.xml")}),
    }
    run = junit.read_run_tests(StubClient(payloads, blobs), RUN_ID, flavors=["cpu-unit"])

    cpu_unit = run.result_for("cpu-unit")
    decoupled = run.result_for("decoupled")
    self.assertIsNotNone(cpu_unit)
    self.assertIsNotNone(decoupled)

    # Workers 1 and 3 of cpu-unit, and nothing else.
    self.assertEqual(cpu_unit.collected, 737 + 737)
    self.assertEqual(cpu_unit.executed, 720 + 0)
    self.assertNotEqual(cpu_unit.collected, 737 + 737 + decoupled.collected)
    self.assertNotEqual(cpu_unit.executed, 720 + decoupled.executed)
    self.assertEqual(sorted(cpu_unit.files), ["test-results-cpu-unit-1.xml", "test-results-cpu-unit-3.xml"])

    self.assertEqual(decoupled.collected, 54)
    self.assertEqual(decoupled.executed, 50)
    self.assertEqual(run.suites["decoupled"].nested_in, "cpu-unit")
    self.assertIsNone(run.suites["cpu-unit"].nested_in)

  def test_it_is_attributed_to_the_worker_that_ran_it(self) -> None:
    """The decoupled step only ever succeeds on cpu-unit worker 1.

    So every one of its rows is tagged worker 1 and the suite reports one publishing worker.
    """
    payloads = [artifact_payload("test-results-cpu-unit-1-33468578834", 1)]
    client = StubClient(payloads, {download_url(1): cpu_unit_worker_1_zip()})
    run = junit.read_run_tests(client, RUN_ID, flavors=["cpu-unit"])

    entry = run.suites["decoupled"]
    self.assertEqual(sorted(entry.per_worker), [1])
    self.assertEqual(entry.published_worker_count, 1)
    self.assertEqual({row.worker for row in entry.result.tests}, {1})


class FailureTextTest(OfflineTestCase):
  """Failure text is quoted verbatim, first line only."""

  def test_the_message_is_quoted_verbatim_from_the_file(self) -> None:
    """The one failing test of run 33467756955 keeps pytest's own wording.

    Character for character, with nothing added, shortened or reworded.
    """
    result = parse_fixture(FAILED_FIXTURE)
    failures = [row for row in result.tests if row.status == junit.STATUS_FAILED]

    self.assertEqual(len(failures), 1)
    row = failures[0]
    self.assertEqual(row.name, FAILED_NAME)
    self.assertEqual(row.classname, FAILED_CLASSNAME)
    self.assertEqual(row.duration, 32.902)
    self.assertEqual(row.failure_message, FAILED_MESSAGE)

    # The same string as the raw XML attribute, so nothing rewrote it on the way through.
    root = ElementTree.fromstring(read_fixture(FAILED_FIXTURE))
    raw = root.find(".//testcase[@name='" + FAILED_NAME + "']/failure")
    self.assertIsNotNone(raw)
    self.assertEqual(raw.get("message"), row.failure_message)
    self.assertIsNone(raw.get("type"))
    # The element body is the 2,086 character traceback; it is not what gets stored.
    self.assertIsNotNone(raw.text)
    self.assertTrue(raw.text.startswith("self = <tests.post_training.integration"))
    self.assertFalse(row.failure_message.startswith("self = "))

  def test_only_the_first_line_of_a_multi_line_message_is_kept(self) -> None:
    """A pytest failure message can run to many lines.

    Only the first survives, stripped of surrounding whitespace but otherwise untouched.
    """
    data = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="1" skipped="0" tests="1" time="1.0">'
        '<testcase classname="t.A" name="test_one" time="1.0">'
        '<failure message="AssertionError: first line, kept as is&#10;second line dropped">body</failure>'
        "</testcase></testsuite></testsuites>"
    ).encode()

    row = junit.parse_junit_xml(data, "multiline.xml").tests[0]
    self.assertEqual(row.status, junit.STATUS_FAILED)
    self.assertEqual(row.failure_message, "AssertionError: first line, kept as is")

  def test_a_failure_without_a_message_attribute_falls_back_to_its_text(self) -> None:
    """Some writers put the message in the element body.

    The first non-empty line of that body is used, still verbatim.
    """
    data = (
        '<testsuite name="pytest" tests="1" time="1.0"><testcase classname="t.A" name="test_one" time="1.0">'
        "<failure>\n\n   E   AssertionError: two arrays differ\nmore traceback\n</failure>"
        "</testcase></testsuite>"
    ).encode()

    row = junit.parse_junit_xml(data, "textonly.xml").tests[0]
    self.assertEqual(row.failure_message, "E   AssertionError: two arrays differ")

  def test_an_error_child_is_labelled_error_and_keeps_its_first_line(self) -> None:
    """A crashed test is an error, not a failure, and its message is treated the same way."""
    data = (
        '<testsuite name="pytest" tests="1" time="1.0"><testcase classname="t.A" name="test_one" time="1.0">'
        '<error type="RuntimeError" message="RuntimeError: worker crashed&#10;traceback">tb</error>'
        "</testcase></testsuite>"
    ).encode()

    result = junit.parse_junit_xml(data, "errored.xml")
    self.assertEqual(result.errored, 1)
    self.assertEqual(result.failed, 0)
    self.assertEqual(result.tests[0].status, junit.STATUS_ERROR)
    self.assertEqual(result.tests[0].failure_message, "RuntimeError: worker crashed")

  def test_passed_and_skipped_rows_carry_no_failure_text(self) -> None:
    """A row with nothing to quote holds None.

    Never an empty string, which would print as a blank failure line.
    """
    result = parse_fixture("gpu-integration-1.xml")

    for row in result.tests:
      self.assertIsNone(row.failure_message)
    self.assertEqual(sum(1 for row in result.tests if row.status == junit.STATUS_PASSED), 11)
    self.assertEqual(sum(1 for row in result.tests if row.status == junit.STATUS_SKIPPED), 15)

  def test_the_green_run_of_the_same_flavor_has_no_failure_text_at_all(self) -> None:
    """The passing capture of the same flavor and worker is the control case.

    Same nine tests, no failure anywhere.
    """
    passing = parse_fixture("tpu-post-training-integration-1.xml")

    self.assertEqual(passing.failed, 0)
    self.assertEqual(passing.errored, 0)
    for row in passing.tests:
      self.assertIsNone(row.failure_message)


class BrokenInputTest(OfflineTestCase):
  """Unreadable input raises a JUnitError that names the file or the artifact."""

  def test_truncated_xml_raises_a_junit_error_naming_the_file(self) -> None:
    """Half a saved file is not valid XML.

    The caller gets a JUnitError that names the file, never a bare ParseError from the XML
    library.
    """
    half = read_fixture("cpu-unit-1.xml")[: 169721 // 2]

    with self.assertRaises(junit.JUnitError) as caught:
      junit.parse_junit_xml(half, "test-results-cpu-unit-1.xml")

    self.assertIn("test-results-cpu-unit-1.xml", str(caught.exception))
    self.assertIn("not valid XML", str(caught.exception))
    self.assertNotIsInstance(caught.exception, ElementTree.ParseError)
    self.assertIsInstance(caught.exception.__cause__, ElementTree.ParseError)

  def test_unreadable_files_raise_a_junit_error_naming_the_file(self) -> None:
    """Every unreadable shape is reported as a JUnitError that says which file broke."""
    cases = (
        (b"", "not valid XML"),
        (b"<testsuites><testsuite tests='1'>", "not valid XML"),
        (b"<html><body>502 Bad Gateway</body></html>", "root element is <html>"),
        (b"<testsuite tests='lots'/>", "non-integer tests='lots'"),
        (b"<testsuite tests='1'><testcase name='n' time='fast'/></testsuite>", "non-numeric time='fast'"),
    )
    for data, expected in cases:
      with self.subTest(expected=expected):
        with self.assertRaises(junit.JUnitError) as caught:
          junit.parse_junit_xml(data, "broken-upload.xml")

        self.assertIn("broken-upload.xml", str(caught.exception))
        self.assertIn(expected, str(caught.exception))

  def test_a_zip_that_is_not_a_zip_raises_a_junit_error_naming_the_artifact(self) -> None:
    """An HTML error page served instead of the artifact names the artifact, not the file."""
    with self.assertRaises(junit.JUnitError) as caught:
      junit.parse_artifact_zip(b"<html>404</html>", "cpu-unit", "test-results-cpu-unit-1-33468578834")

    self.assertIn("test-results-cpu-unit-1-33468578834", str(caught.exception))
    self.assertIn("not a readable zip", str(caught.exception))

  def test_a_download_failure_is_reported_as_a_junit_error_naming_the_artifact(self) -> None:
    """Whatever the client raises, the caller sees a JUnitError that names the upload."""
    payloads = [artifact_payload("test-results-cpu-unit-1-33468578834", 1)]

    with self.assertRaises(junit.JUnitError) as caught:
      junit.read_run_tests(FailingClient(payloads), RUN_ID, flavors=["cpu-unit"])

    self.assertIn("test-results-cpu-unit-1-33468578834", str(caught.exception))
    self.assertIn("download failed", str(caught.exception))


class MissingResultTest(OfflineTestCase):
  """A suite with nothing to show is None plus a reason code, never a zero."""

  def test_a_flavor_that_published_nothing_is_none_with_a_reason(self) -> None:
    """Three TPU Pathways jobs of run 33468578834 finished green without publishing.

    Their suites must read as None plus `no_file_published`, because a zero would draw a
    tests-vanished alarm for a run that simply published nothing.
    """
    payloads = [artifact_payload("test-results-cpu-unit-1-33468578834", 1)]
    run = junit.read_run_tests(StubClient(payloads, {download_url(1): cpu_unit_worker_1_zip()}), RUN_ID)

    for suite_id in ("tpu-unit", "tpu-integration", "gpu-unit", "gpu-integration"):
      with self.subTest(suite=suite_id):
        entry = run.suites[suite_id]
        self.assertIsNone(entry.result)
        self.assertEqual(entry.reason, junit.REASON_NO_FILE)
        self.assertEqual(entry.per_worker, {})
        self.assertIsNone(run.result_for(suite_id))

  def test_an_expired_artifact_is_none_with_the_expired_reason(self) -> None:
    """Artifacts live about a day.

    Once GitHub has deleted the payload the suite is None with `artifact_expired`, and the
    module does not try to download it.
    """
    payloads = [artifact_payload("test-results-gpu-unit-1-33468578834", 5, expired=True)]
    client = StubClient(payloads)
    run = junit.read_run_tests(client, RUN_ID, flavors=["gpu-unit"])

    self.assertIsNone(run.result_for("gpu-unit"))
    self.assertEqual(run.reason_for("gpu-unit"), junit.REASON_ARTIFACT_EXPIRED)
    self.assertEqual(client.downloads, [])
    self.assertTrue(run.artifacts[0].expired)

  def test_an_upload_without_any_xml_is_none_with_the_empty_reason(self) -> None:
    """An artifact that exists but holds no XML file is `upload_empty`.

    That is a different story from never publishing and from expiring.
    """
    payloads = [artifact_payload("test-results-gpu-unit-1-33468578834", 5)]
    blobs = {download_url(5): make_zip({"pytest.log": b"no xml here"})}
    run = junit.read_run_tests(StubClient(payloads, blobs), RUN_ID, flavors=["gpu-unit"])

    self.assertIsNone(run.result_for("gpu-unit"))
    self.assertEqual(run.reason_for("gpu-unit"), junit.REASON_UPLOAD_EMPTY)

  def test_every_suite_has_either_a_result_or_a_reason_and_never_a_bare_zero(self) -> None:
    """Across a mixed run, no suite is ever left with a count of zero and no explanation."""
    payloads = [
        artifact_payload("test-results-cpu-unit-1-33468578834", 1),
        artifact_payload("test-results-gpu-integration-1-33468578834", 4),
        artifact_payload("test-results-cpu-integration-1-33468578834", 5, expired=True),
        artifact_payload("test-results-cpu-post-training-unit-4-33468578834", 6),
        artifact_payload("maxtext-wheel", 7),
    ]
    blobs = {
        download_url(1): cpu_unit_worker_1_zip(),
        download_url(4): make_zip({"test-results-gpu-integration-1.xml": read_fixture("gpu-integration-1.xml")}),
        download_url(6): make_zip({"build.log": b"empty upload"}),
    }
    run = junit.read_run_tests(StubClient(payloads, blobs), RUN_ID)

    reasons = {junit.REASON_NO_FILE, junit.REASON_UPLOAD_EMPTY, junit.REASON_ARTIFACT_EXPIRED}
    for suite_id, entry in run.suites.items():
      with self.subTest(suite=suite_id):
        if entry.result is None:
          self.assertIn(entry.reason, reasons)
        else:
          self.assertIsNone(entry.reason)
          self.assertGreater(entry.result.collected, 0)

    self.assertEqual(run.reason_for("cpu-integration"), junit.REASON_ARTIFACT_EXPIRED)
    self.assertEqual(run.reason_for("cpu-post-training-unit"), junit.REASON_UPLOAD_EMPTY)
    self.assertEqual(run.reason_for("tpu-post-training-unit"), junit.REASON_NO_FILE)
    self.assertEqual(run.result_for("gpu-integration").executed, 11)
    self.assertEqual(len(run.artifacts), 4)  # the wheel is not one of ours

  def test_an_unknown_suite_reads_as_missing_rather_than_zero(self) -> None:
    """Asking about a suite the run never heard of returns None and the no-file reason."""
    run = junit.read_run_tests(StubClient([]), RUN_ID, flavors=["gpu-unit"])

    self.assertIsNone(run.result_for("pathways-unit"))
    self.assertEqual(run.reason_for("pathways-unit"), junit.REASON_NO_FILE)


class PartialWorkerTest(OfflineTestCase):
  """A flavor that lost some of its workers must not pass a partial total off as complete.

  These are the two defects the first review found: the reason code used to be worked out per
  flavor rather than per worker, so a surviving worker's total was reported as if nothing were
  missing, and a flavor nobody asked about disappeared entirely once its artifact expired.
  """

  def cpu_unit_run(self, payloads: list[dict], blobs: dict[str, bytes]) -> junit.RunTests:
    """Reads one cpu-unit run through the stub client.

    Args:
      payloads: The artifact payloads the run has.
      blobs: Download URL -> zip bytes.

    Returns:
      The parsed run.
    """
    return junit.read_run_tests(StubClient(payloads, blobs), RUN_ID, flavors=["cpu-unit"])

  def test_a_surviving_worker_total_is_marked_partial_when_another_expired(self) -> None:
    """cpu-unit publishes four artifacts that expire up to 27 minutes apart.

    A tick landing inside that window reads some and not others. The merged number is then a
    partial one, and saying so is the whole point: a silent partial total is worse than a
    zero, because it looks plausible.
    """
    payloads = [
        artifact_payload("test-results-cpu-unit-1-33468578834", 1, expired=True),
        artifact_payload("test-results-cpu-unit-3-33468578834", 3),
    ]
    blobs = {download_url(3): make_zip({"test-results-cpu-unit-3.xml": read_fixture("cpu-unit-3.xml")})}
    entry = self.cpu_unit_run(payloads, blobs).suites["cpu-unit"]

    self.assertIsNotNone(entry.result)
    self.assertEqual(entry.result.collected, 737)  # worker 3 only; worker 1's 737 are gone
    self.assertIsNone(entry.reason)
    self.assertEqual(entry.missing_workers, {1: junit.REASON_ARTIFACT_EXPIRED})
    self.assertTrue(entry.is_partial)
    self.assertEqual(sorted(entry.per_worker), [3])

  def test_a_worker_whose_upload_held_no_xml_is_listed_too(self) -> None:
    """A shard that died before writing its XML is the other way a worker goes missing."""
    payloads = [
        artifact_payload("test-results-cpu-unit-1-33468578834", 1),
        artifact_payload("test-results-cpu-unit-2-33468578834", 2),
    ]
    blobs = {
        download_url(1): make_zip({"test-results-cpu-unit-1.xml": read_fixture("cpu-unit-1.xml")}),
        download_url(2): make_zip({"pytest.log": b"no xml here"}),
    }
    entry = self.cpu_unit_run(payloads, blobs).suites["cpu-unit"]

    self.assertEqual(entry.result.collected, 737)
    self.assertEqual(entry.missing_workers, {2: junit.REASON_UPLOAD_EMPTY})
    self.assertTrue(entry.is_partial)

  def test_a_complete_flavor_lists_no_missing_worker(self) -> None:
    """The control case: when every worker published, nothing is flagged."""
    payloads = [
        artifact_payload("test-results-cpu-unit-1-33468578834", 1),
        artifact_payload("test-results-cpu-unit-3-33468578834", 3),
    ]
    blobs = {
        download_url(1): make_zip({"test-results-cpu-unit-1.xml": read_fixture("cpu-unit-1.xml")}),
        download_url(3): make_zip({"test-results-cpu-unit-3.xml": read_fixture("cpu-unit-3.xml")}),
    }
    entry = self.cpu_unit_run(payloads, blobs).suites["cpu-unit"]

    self.assertEqual(entry.result.collected, 737 + 737)
    self.assertEqual(entry.missing_workers, {})
    self.assertFalse(entry.is_partial)

  def test_decoupled_does_not_borrow_another_workers_empty_upload(self) -> None:
    """Worker 1 publishes a normal cpu-unit file and no decoupled file.

    Worker 2's upload holds no XML at all. The decoupled pass simply did not run, so its
    reason must say that rather than repeat worker 2's failure.
    """
    payloads = [
        artifact_payload("test-results-cpu-unit-1-33468578834", 1),
        artifact_payload("test-results-cpu-unit-2-33468578834", 2),
    ]
    blobs = {
        download_url(1): make_zip({"test-results-cpu-unit-1.xml": read_fixture("cpu-unit-1.xml")}),
        download_url(2): make_zip({"pytest.log": b"no xml here"}),
    }
    run = self.cpu_unit_run(payloads, blobs)

    self.assertIsNotNone(run.result_for("cpu-unit"))
    self.assertIsNone(run.result_for("decoupled"))
    self.assertEqual(run.reason_for("decoupled"), junit.REASON_NO_FILE)

  def test_decoupled_does_not_borrow_another_workers_expiry_either(self) -> None:
    """Same shape, with worker 2's artifact expired instead of empty."""
    payloads = [
        artifact_payload("test-results-cpu-unit-1-33468578834", 1),
        artifact_payload("test-results-cpu-unit-2-33468578834", 2, expired=True),
    ]
    blobs = {download_url(1): make_zip({"test-results-cpu-unit-1.xml": read_fixture("cpu-unit-1.xml")})}
    run = self.cpu_unit_run(payloads, blobs)

    self.assertEqual(run.reason_for("decoupled"), junit.REASON_NO_FILE)

  def test_a_flavor_nobody_asked_about_still_reports_why_it_is_missing(self) -> None:
    """tpu7x flavors run only outside pull requests, so they are not in the default ask list.

    They still appear in the artifacts of a scheduled run, and once one expires the suite must
    say `artifact_expired`. Dropping the suite entirely, as it used to, hid a gap in exactly
    the scheduled history the dashboard's long-term trend is built from.
    """
    for expired, blobs, expected in (
        (True, {}, junit.REASON_ARTIFACT_EXPIRED),
        (False, {download_url(9): make_zip({"pytest.log": b"no xml here"})}, junit.REASON_UPLOAD_EMPTY),
    ):
      with self.subTest(reason=expected):
        payloads = [artifact_payload("test-results-tpu7x-unit-1-33468578834", 9, expired=expired)]
        run = junit.read_run_tests(StubClient(payloads, blobs), RUN_ID, flavors=["gpu-unit"])

        self.assertIn("tpu7x-unit", run.suites)
        self.assertIsNone(run.result_for("tpu7x-unit"))
        self.assertEqual(run.reason_for("tpu7x-unit"), expected)
        self.assertEqual(run.suites["tpu7x-unit"].missing_workers, {1: expected})

  def test_a_flavor_nobody_asked_about_still_reports_its_results(self) -> None:
    """The same flavor with a readable artifact is reported with its counts, as before."""
    payloads = [artifact_payload("test-results-tpu7x-unit-1-33468578834", 9)]
    blobs = {download_url(9): make_zip({"test-results-gpu-integration-1.xml": read_fixture("gpu-integration-1.xml")})}
    run = junit.read_run_tests(StubClient(payloads, blobs), RUN_ID, flavors=["gpu-unit"])

    self.assertEqual(run.result_for("tpu7x-unit").executed, 11)
    self.assertIsNone(run.reason_for("tpu7x-unit"))
    self.assertEqual(run.suites["tpu7x-unit"].missing_workers, {})


class ArtifactListingTest(OfflineTestCase):
  """Covers `list_test_artifacts` against the saved 28-artifact response."""

  def test_it_keeps_only_our_uploads(self) -> None:
    """The listing keeps the 20 test-result uploads and drops the wheel and the notebooks."""
    payload = json.loads(read_fixture("artifacts.json"))
    client = StubClient(payload["artifacts"])
    refs = junit.list_test_artifacts(client, RUN_ID)

    self.assertEqual(len(refs), 20)
    self.assertEqual(client.paginate_calls, [(f"actions/runs/{RUN_ID}/artifacts", "artifacts", {"per_page": 100})])
    self.assertEqual(client.downloads, [])
    for ref in refs:
      self.assertTrue(ref.name.startswith(junit.ARTIFACT_PREFIX))
    self.assertEqual({ref.flavor for ref in refs}, set(WORKERS_PER_FLAVOR))

  def test_refs_carry_the_fields_the_collector_stores(self) -> None:
    """One reference is checked field by field against the saved payload."""
    payload = json.loads(read_fixture("artifacts.json"))
    refs = junit.list_test_artifacts(StubClient(payload["artifacts"]), RUN_ID)
    worker_1 = next(ref for ref in refs if ref.name == "test-results-cpu-unit-1-33468578834")

    self.assertEqual(worker_1.artifact_id, 9785903147)
    self.assertEqual(worker_1.flavor, "cpu-unit")
    self.assertEqual(worker_1.worker, 1)
    self.assertEqual(worker_1.run_id, RUN_ID)
    self.assertFalse(worker_1.expired)
    self.assertEqual(worker_1.size_in_bytes, 17613)
    self.assertEqual(worker_1.created_at, "2026-09-01T04:15:40Z")
    self.assertEqual(worker_1.expires_at, "2026-09-02T04:15:39Z")
    self.assertTrue(worker_1.download_url.endswith("/actions/artifacts/9785903147/zip"))
    for ref in refs:
      self.assertFalse(ref.expired)

  def test_it_rejects_a_response_that_is_not_a_list_of_objects(self) -> None:
    """A payload shape the endpoint should never return is an error naming the run."""
    with self.assertRaises(junit.JUnitError) as caught:
      junit.list_test_artifacts(StubClient(["test-results-cpu-unit-1-33468578834"]), RUN_ID)

    self.assertIn(str(RUN_ID), str(caught.exception))
    self.assertIn("expected an object", str(caught.exception))


class MergeTest(OfflineTestCase):
  """Covers `merge_suite_results`, which adds the workers of one flavor together."""

  def test_merging_nothing_returns_none(self) -> None:
    """No parts means no result, not an empty result with zeros in it."""
    self.assertIsNone(junit.merge_suite_results([]))

  def test_merging_workers_adds_counts_and_drops_the_pod_name(self) -> None:
    """Two workers of the same flavor add up.

    The runner pod name is dropped because the merged result no longer belongs to one machine.
    """
    worker_1 = parse_fixture("cpu-unit-1.xml")
    worker_3 = parse_fixture("cpu-unit-3.xml")
    merged = junit.merge_suite_results([worker_1, worker_3])

    self.assertIsNotNone(merged)
    self.assertEqual(merged.collected, 737 + 737)
    self.assertEqual(merged.skipped, 17 + 737)
    self.assertEqual(merged.executed, 720)
    self.assertEqual(round(merged.junit_seconds, 3), round(1055.921 + 2.371, 3))
    self.assertEqual(merged.reported_tests, 870 + 737)
    self.assertIsNone(merged.hostname)
    self.assertIsNone(merged.timestamp)
    self.assertEqual(merged.files, ("cpu-unit-1.xml", "cpu-unit-3.xml"))
    self.assertEqual(len(merged.tests), 1474)
    self.assertFalse(merged.count_matches_attribute)


if __name__ == "__main__":
  unittest.main(verbosity=2)
