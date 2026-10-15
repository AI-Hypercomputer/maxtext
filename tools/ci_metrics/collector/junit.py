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

"""Reads the JUnit XML test results that the MaxText Package Tests pipeline uploads.

Every test job of `.github/workflows/ci_pipeline.yml` uploads one artifact per parallel
worker, named `test-results-<flavor>-<worker>-<run_id>`. The zip inside holds one or more
`test-results-*.xml` files written by pytest.

This module turns those artifacts into plain numbers:

  * `parse_artifact_name` and `parse_junit_xml` are pure - no network, no clock, no state -
    so they can be tested offline against saved fixtures.
  * `list_test_artifacts` and `read_run_tests` talk to GitHub through the client object
    defined in `github.py`; they never build URLs of their own beyond the artifact paths.

Three rules from the data catalog are enforced here and must not be relaxed:

  1. A missing result is None with a reason code, never a zero. A zero would make the
     dashboard draw a "tests vanished" alarm for a run that simply published nothing. When
     only some workers of a flavor are missing, the surviving total is a partial number, so
     the workers that published nothing are listed in `SuiteEntry.missing_workers` and the
     reader must decide whether that total can be trusted.
  2. The test count is the number of `<testcase>` elements minus the skipped ones. The
     `<testsuite tests="...">` attribute lies (870 against 737 real elements on one file),
     so it is stored for cross-checking only.
  3. The `decoupled` pass is its own suite id, nested inside `cpu-unit` worker 1. Its tests
     are also counted inside `cpu-unit`, so the two totals must never be added together.

Failure text is quoted verbatim, first line only. Nothing in this module summarises,
rewrites or scores anything; every value is an XML attribute or arithmetic on one.

The XML comes from this repository's CI runs, including runs for pull requests from forks,
so treat it as untrusted text. `xml.etree.ElementTree` does not load external entities, and
the zip is read in memory, never unpacked to disk. A truncated or malformed file raises
`JUnitError` naming the file, instead of escaping as a bare `ParseError`.

A zip can also be too big: a small zip that unpacks to gigabytes, or a huge upload. Either
could run the tick out of memory. So a zip over `MAX_ARTIFACT_BYTES` is not downloaded, and a
zip with an XML file over `MAX_XML_BYTES`, or XML files over `MAX_ZIP_XML_BYTES` together, is
not read. Each of these raises `JUnitError` before the zip is downloaded or unpacked. Each file
is then read with a cap at the size the zip declares for it.

The sizes that GitHub lists are only notes, and one run can have many artifacts. So the client
is also asked to count the bytes that really arrive and to stop at the limit. One run may not
download more than `MAX_RUN_ARTIFACT_BYTES` in all, and may not list more than
`MAX_RUN_ARTIFACTS` artifacts, because each download costs a request from an hourly budget that
the whole repository shares. Zips pack well, though, so the XML of all the zips of a run is
capped as well (`MAX_RUN_XML_BYTES`, by declared size, checked before the XML that would cross
it is read): the parsed results of a run all stay in memory. An XML file with a DOCTYPE is
refused, because the entities declared in one can blow a small file up past `MAX_XML_BYTES`.

A zip can also be damaged: corrupt data, a compression method that `zipfile` does not know, an
encrypted file, a bad file name. `zipfile`, `zlib` and `lzma` raise many kinds of error for
these. All of them become a `JUnitError` that names the file, because the caller keeps the run
when it gets a `JUnitError`, but any other error would crash the whole tick.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol
from xml.etree import ElementTree

# Artifact naming, e.g. test-results-cpu-post-training-unit-3-33468578834
ARTIFACT_PREFIX = "test-results-"

# The flavors ci_pipeline.yml can run on a pull request. tpu7x flavors only run when the
# trigger is not pull_request, so they are not in the default ask list; read_run_tests still
# reports them when their artifacts show up.
KNOWN_FLAVORS = (
    "tpu-unit",
    "tpu-integration",
    "tpu-post-training-unit",
    "tpu-post-training-integration",
    "gpu-unit",
    "gpu-integration",
    "cpu-unit",
    "cpu-integration",
    "cpu-post-training-unit",
    "cpu-post-training-integration",
)

# Suites that are an extra pytest pass inside another flavor's worker rather than a flavor of
# their own: suite id -> the flavor it runs inside.
NESTED_SUITES = {"decoupled": "cpu-unit"}

# Why a suite has no result. Stored next to a None result, never as a count of zero.
REASON_NO_FILE = "no_file_published"
REASON_UPLOAD_EMPTY = "upload_empty"
REASON_ARTIFACT_EXPIRED = "artifact_expired"

# Per-test outcome labels.
STATUS_PASSED = "passed"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_ERROR = "error"

# Size limits for test-result zips. These zips can come from any pull request, including forks,
# so a zip that is far too big must be refused before it can run the tick out of memory. Real
# files are far smaller: the biggest XML file seen so far is about 285 KB.
#
# The largest zip we download (compressed). A zip that GitHub lists as bigger is not downloaded at
# all, and the bytes that really arrive are counted too, because the listed size is only a note.
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
# The largest single XML file inside a zip, unpacked.
MAX_XML_BYTES = 20 * 1024 * 1024
# All the XML files of one zip together, unpacked.
MAX_ZIP_XML_BYTES = 100 * 1024 * 1024
# All the zips of one run together, compressed.
MAX_RUN_ARTIFACT_BYTES = 200 * 1024 * 1024
# All the XML files of all the zips of one run together, unpacked. A run can have many zips, and
# the parsed results of all of them stay in memory until the run is done, so this is what bounds
# the memory one run can use. Real runs need a few MB.
MAX_RUN_XML_BYTES = 100 * 1024 * 1024
# The most artifacts one run may list. Real runs have tens. Each download costs a request from an
# hourly budget that the whole repository shares, so a run with thousands of tiny artifacts could
# use it up.
MAX_RUN_ARTIFACTS = 200


class JUnitError(Exception):
  """Raised when an artifact or a JUnit XML file cannot be read, or is too big to be read.

  The message always names the file or artifact at fault, so a collector tick that fails can
  say which upload broke it.
  """


class GitHubClientLike(Protocol):
  """The part of `github.GitHubClient` this module uses.

  Declared structurally so that `junit.py` does not import `github.py`: the pure parsers stay
  importable on their own, and tests can pass a stub client. Rate limiting and retries belong
  to the client, so nothing here waits or retries by itself.
  """

  def paginate(self, path: str, key: str, *, max_items: int | None = None, **params: object) -> list:
    """Follows the pages of a list endpoint and returns the flattened list.

    With `max_items` set, the client may stop after the page that makes the list that long, so
    the list can be cut short. The caller must not take a list that long for the whole thing.
    """

  def get_bytes(self, url: str, *, max_bytes: int | None = None) -> bytes:
    """Downloads an absolute URL, following redirects, and returns the raw body.

    With `max_bytes` set, the client must count the bytes that really arrive and raise as soon
    as there are more than that.
    """


@dataclass
class ArtifactRef:
  """One `test-results-*` artifact of a workflow run.

  Attributes:
    name: The artifact name exactly as GitHub reports it.
    artifact_id: The numeric artifact id.
    flavor: The test flavor parsed out of the name, e.g. "cpu-post-training-unit".
    worker: The 1-based parallel worker number parsed out of the name.
    expired: True once GitHub has deleted the payload (retention is one day).
    size_in_bytes: Compressed size of the zip.
    download_url: Absolute URL of the zip, for `GitHubClient.get_bytes`.
    run_id: The run id parsed out of the name; None when the name carried none.
    created_at: ISO-8601 upload time, or None when the payload did not carry it.
    expires_at: ISO-8601 deletion time, or None when the payload did not carry it.
  """

  name: str
  artifact_id: int
  flavor: str
  worker: int
  expired: bool
  size_in_bytes: int
  download_url: str
  run_id: int | None = None
  created_at: str | None = None
  expires_at: str | None = None

  @classmethod
  def from_api(cls, payload: dict) -> ArtifactRef | None:
    """Builds a reference from one entry of the artifacts endpoint.

    Args:
      payload: One element of `GET /repos/{owner}/{repo}/actions/runs/{id}/artifacts`.

    Returns:
      The reference, or None when the name is not one of our test-result artifacts (the same
      response also carries `maxtext-wheel` and `notebook-outputs-*`).
    """
    parsed = _split_artifact_name(str(payload.get("name", "")))
    if parsed is None:
      return None
    flavor, worker, run_id = parsed
    return cls(
        name=str(payload.get("name", "")),
        artifact_id=int(payload.get("id", 0)),
        flavor=flavor,
        worker=worker,
        expired=bool(payload.get("expired", False)),
        size_in_bytes=int(payload.get("size_in_bytes", 0)),
        download_url=str(payload.get("archive_download_url", "")),
        run_id=run_id,
        created_at=payload.get("created_at"),
        expires_at=payload.get("expires_at"),
    )


@dataclass
class TestRow:
  """One `<testcase>` element.

  Attributes:
    name: The test method name, e.g. "test_grpo_loss_drives_a_training_step".
    classname: The dotted module and class. Empty for module-level collection skips.
    duration: The `time` attribute in seconds.
    status: One of "passed", "skipped", "failed", "error".
    failure_message: First line of the failure or error message, quoted verbatim and
      stripped; None for a test that passed or was skipped.
    worker: The parallel worker that ran the test, filled in by `read_run_tests`; None when
      the row came straight out of `parse_junit_xml`.
  """

  name: str
  classname: str
  duration: float
  status: str
  failure_message: str | None = None
  worker: int | None = None


@dataclass
class SuiteResult:
  """The counts of one suite in one run, from one XML file or several merged.

  Attributes:
    collected: Number of `<testcase>` elements. NOT the `<testsuite tests>` attribute.
    skipped: Cases carrying a `<skipped>` child.
    executed: collected - skipped. The number the dashboard calls the test count.
    junit_seconds: Sum of the per-case `time` attributes. For the CPU flavors pytest runs
      with `-n auto`, so this adds up across parallel processes and is NOT wall-clock time;
      wall-clock comes from the job step timestamps elsewhere.
    tests: One row per `<testcase>`, in document order.
    failed: Cases carrying a `<failure>` child.
    errored: Cases carrying an `<error>` child.
    reported_tests: The `<testsuite tests>` attribute, kept only to cross-check against
      `collected`; it disagrees on real files.
    suite_seconds: The `<testsuite time>` attribute. Also not wall-clock time.
    hostname: The `<testsuite hostname>` attribute (the runner pod), None when files with
      different hostnames were merged.
    timestamp: The `<testsuite timestamp>` attribute, None when merged files disagree.
    files: Names of the XML files this result was built from.
  """

  collected: int
  skipped: int
  executed: int
  junit_seconds: float
  tests: list[TestRow]
  failed: int = 0
  errored: int = 0
  reported_tests: int | None = None
  suite_seconds: float | None = None
  hostname: str | None = None
  timestamp: str | None = None
  files: tuple[str, ...] = ()

  @property
  def count_matches_attribute(self) -> bool:
    """True when `<testsuite tests>` agrees with the counted elements."""
    return self.reported_tests is not None and self.reported_tests == self.collected


@dataclass
class SuiteEntry:
  """One suite's outcome in one run: either a result, or the reason there is none.

  Attributes:
    suite_id: A flavor name, or "decoupled" for the nested pass.
    result: The merged result across every worker that published, or None.
    reason: One of the REASON_* codes when `result` is None, otherwise None.
    nested_in: The flavor this suite runs inside, for "decoupled"; None for a real flavor.
      A nested suite's tests are also counted in the parent, so never add the two.
    per_worker: Result per parallel worker number, for the workers that published a file.
    missing_workers: Worker number -> REASON_* for the workers whose payload could not be
      read. The workers are those of this suite's flavor, which for a nested suite means the
      parent's. Non-empty next to a result means the result is a partial total, missing those
      workers' tests. Treat that as a hole in the data, not as a drop in the test count.
  """

  suite_id: str
  result: SuiteResult | None = None
  reason: str | None = None
  nested_in: str | None = None
  per_worker: dict[int, SuiteResult] = field(default_factory=dict)
  missing_workers: dict[int, str] = field(default_factory=dict)

  @property
  def is_partial(self) -> bool:
    """True when there is a result but at least one worker of the flavor is missing from it."""
    return self.result is not None and bool(self.missing_workers)

  @property
  def published_worker_count(self) -> int:
    """How many workers published a file.

    This is a cross-check only. The authoritative worker count W is the number of
    "Execute Tests (N)" jobs in the run, read from the jobs endpoint, never from files.
    """
    return len(self.per_worker)


@dataclass
class RunTests:
  """Every suite of one workflow run.

  Attributes:
    run_id: The workflow run these results belong to.
    suites: Suite id -> entry. Holds an entry for every suite that was asked about, whether
      or not it published anything, plus any extra flavor found in the artifacts.
    artifacts: The test-result artifacts the run had, including expired ones.
  """

  run_id: int
  suites: dict[str, SuiteEntry] = field(default_factory=dict)
  artifacts: list[ArtifactRef] = field(default_factory=list)

  def result_for(self, suite_id: str) -> SuiteResult | None:
    """Returns the merged result of one suite, or None when it published nothing."""
    entry = self.suites.get(suite_id)
    return entry.result if entry else None

  def reason_for(self, suite_id: str) -> str | None:
    """Returns the reason code of one suite, or None when it has a result."""
    entry = self.suites.get(suite_id)
    return entry.reason if entry else REASON_NO_FILE


def _split_artifact_name(name: str) -> tuple[str, int, int] | None:
  """Splits a test-result artifact name into flavor, worker and run id.

  Flavor names contain hyphens, so the name is read from the right: the last field is the run
  id, the field before it is the worker number, and everything between the `test-results-`
  prefix and the worker is the flavor.

  Args:
    name: The artifact name.

  Returns:
    (flavor, worker, run_id), or None when the name is not one of ours.
  """
  if not name.startswith(ARTIFACT_PREFIX):
    return None
  stem = name[len(ARTIFACT_PREFIX) :]
  parts = stem.split("-")
  if len(parts) < 3:
    return None
  flavor_parts, worker_part, run_part = parts[:-2], parts[-2], parts[-1]
  if not flavor_parts or not worker_part.isdigit() or not run_part.isdigit():
    return None
  flavor = "-".join(flavor_parts)
  if not flavor:
    return None
  return flavor, int(worker_part), int(run_part)


def parse_artifact_name(name: str) -> tuple[str, int] | None:
  """Reads the flavor and worker number out of an artifact name.

  Pure: no network, no state.

  Args:
    name: The artifact name, e.g. "test-results-cpu-post-training-unit-3-33468578834".

  Returns:
    (flavor, worker), e.g. ("cpu-post-training-unit", 3). None when the name is not one of
    ours, which is how `maxtext-wheel` and `notebook-outputs-*` are filtered out.
  """
  parsed = _split_artifact_name(name)
  if parsed is None:
    return None
  flavor, worker, _ = parsed
  return flavor, worker


def suite_id_for_file(file_name: str, flavor: str) -> tuple[str, str | None]:
  """Decides which suite an XML file inside an artifact belongs to.

  `test-results-decoupled-targeted.xml` is written by the extra "Run Targeted Decoupled
  Tests" step into the same artifact as cpu-unit worker 1. It is a suite of its own.

  Args:
    file_name: The member name inside the artifact zip.
    flavor: The flavor the artifact belongs to.

  Returns:
    (suite_id, nested_in). `nested_in` names the flavor a nested suite runs inside, and is
    None for an ordinary flavor file.
  """
  base = file_name.rsplit("/", 1)[-1].lower()
  for nested_id, parent in NESTED_SUITES.items():
    if nested_id in base:
      return nested_id, parent
  return flavor, None


def _first_line(text: str | None) -> str | None:
  """Returns the first non-empty line of a message, stripped, or None."""
  if not text:
    return None
  for line in text.splitlines():
    stripped = line.strip()
    if stripped:
      return stripped
  return None


def _float_attr(element: ElementTree.Element, attr: str, file_name: str, default: float = 0.0) -> float:
  """Reads a numeric attribute, defaulting when absent and raising when unreadable."""
  raw = element.get(attr)
  if raw is None or raw.strip() == "":
    return default
  try:
    return float(raw)
  except ValueError as exc:
    raise JUnitError(f"{file_name}: <{element.tag}> has a non-numeric {attr}={raw!r}") from exc


def _int_attr(element: ElementTree.Element, attr: str, file_name: str) -> int | None:
  """Reads an integer attribute, returning None when absent and raising when unreadable."""
  raw = element.get(attr)
  if raw is None or raw.strip() == "":
    return None
  try:
    return int(raw)
  except ValueError as exc:
    raise JUnitError(f"{file_name}: <{element.tag}> has a non-integer {attr}={raw!r}") from exc


def _case_outcome(
    skipped: ElementTree.Element | None,
    failure: ElementTree.Element | None,
    error: ElementTree.Element | None,
) -> tuple[str, str | None]:
  """Turns the outcome children of one `<testcase>` into a label and a message.

  Real pytest output sets at most one of the three. When a file sets two, the worse outcome
  wins the label, while the counters in `parse_junit_xml` still count each child on its own,
  so `executed` stays `collected - skipped`.

  Args:
    skipped: The `<skipped>` child, or None.
    failure: The `<failure>` child, or None.
    error: The `<error>` child, or None.

  Returns:
    (status, failure_message). The message is the first line of the element's `message`
    attribute, or of its text when the attribute is absent, quoted verbatim and stripped.
    None for a passed or skipped case.
  """
  if error is not None:
    return STATUS_ERROR, _first_line(error.get("message") or error.text)
  if failure is not None:
    return STATUS_FAILED, _first_line(failure.get("message") or failure.text)
  if skipped is not None:
    return STATUS_SKIPPED, None
  return STATUS_PASSED, None


def _testsuite_elements(root: ElementTree.Element, file_name: str) -> list[ElementTree.Element]:
  """Returns the `<testsuite>` elements of a parsed file, whichever root pytest wrote."""
  if root.tag == "testsuite":
    return [root]
  if root.tag == "testsuites":
    return list(root.findall("testsuite"))
  raise JUnitError(f"{file_name}: root element is <{root.tag}>, expected <testsuites> or <testsuite>")


class _RefuseDoctype(ElementTree.TreeBuilder):
  """Builds the element tree as usual, but raises `JUnitError` when the file has a DOCTYPE.

  A JUnit file never has a DOCTYPE. A hostile one can: entities declared in it are expanded
  while the file is parsed, so a file far under `MAX_XML_BYTES` can fill the memory (expat
  stops only an expansion of more than 100 times, and only after 8 MB). Refusing the DOCTYPE
  stops that before a single entity is read.

  The parser calls `doctype` for the declaration itself, so this does not depend on how the
  file is encoded (in UTF-16 the bytes `<!DOCTYPE` are not there to be found), and text that
  only looks like a DOCTYPE, in a CDATA section or escaped, is left alone.
  """

  def __init__(self, file_name: str) -> None:
    """Builds the builder.

    Args:
      file_name: Name used in the error message.
    """
    super().__init__()
    self._file_name = file_name

  def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
    """Refuses the DOCTYPE, which the parser reports before it reads anything inside it.

    Args:
      name: The name of the root element, as the DOCTYPE gives it.
      pubid: The public identifier, or None.
      system: The system identifier, or None.

    Raises:
      JUnitError: Always.
    """
    del name, pubid, system
    raise JUnitError(
        f"{self._file_name}: has a DOCTYPE, which a JUnit file never has "
        "(the entities in one can blow a small file up into a huge one)"
    )


def parse_junit_xml(data: bytes, file_name: str = "<junit xml>") -> SuiteResult:
  """Turns one JUnit XML file into counts and per-test rows.

  Pure: no network, no state. Counts `<testcase>` elements and ignores the `<testsuite
  tests>` attribute, which disagrees with the elements on real files.

  Args:
    data: The raw bytes of the XML file.
    file_name: Name used in error messages and recorded in the result.

  Returns:
    The suite result. `executed` is `collected - skipped`.

  Raises:
    JUnitError: The file is truncated, malformed, has a DOCTYPE, has an unexpected root
      element, or carries an unreadable numeric attribute.
  """
  parser = ElementTree.XMLParser(target=_RefuseDoctype(file_name))
  try:
    parser.feed(data)
    root = parser.close()
  except ElementTree.ParseError as exc:
    raise JUnitError(f"{file_name}: not valid XML ({exc})") from exc

  suites = _testsuite_elements(root, file_name)
  rows: list[TestRow] = []
  collected = skipped = failed = errored = 0
  junit_seconds = 0.0
  reported_tests: int | None = None
  suite_seconds: float | None = None
  hostnames: set[str] = set()
  timestamps: set[str] = set()

  for suite in suites:
    attr_tests = _int_attr(suite, "tests", file_name)
    if attr_tests is not None:
      reported_tests = attr_tests if reported_tests is None else reported_tests + attr_tests
    if suite.get("time") is not None:
      suite_seconds = _float_attr(suite, "time", file_name) + (suite_seconds or 0.0)
    if suite.get("hostname"):
      hostnames.add(str(suite.get("hostname")))
    if suite.get("timestamp"):
      timestamps.add(str(suite.get("timestamp")))

    for case in suite.iter("testcase"):
      collected += 1
      duration = _float_attr(case, "time", file_name)
      junit_seconds += duration
      skipped_child = case.find("skipped")
      failure_child = case.find("failure")
      error_child = case.find("error")
      if skipped_child is not None:
        skipped += 1
      if failure_child is not None:
        failed += 1
      if error_child is not None:
        errored += 1
      status, message = _case_outcome(skipped_child, failure_child, error_child)
      rows.append(
          TestRow(
              name=case.get("name", ""),
              classname=case.get("classname", ""),
              duration=duration,
              status=status,
              failure_message=message,
          )
      )

  return SuiteResult(
      collected=collected,
      skipped=skipped,
      executed=collected - skipped,
      junit_seconds=junit_seconds,
      tests=rows,
      failed=failed,
      errored=errored,
      reported_tests=reported_tests,
      suite_seconds=suite_seconds,
      hostname=hostnames.pop() if len(hostnames) == 1 else None,
      timestamp=timestamps.pop() if len(timestamps) == 1 else None,
      files=(file_name,),
  )


def merge_suite_results(results: Iterable[SuiteResult]) -> SuiteResult | None:
  """Adds up the results of one suite across the workers that published a file.

  Pure. `hostname` and `timestamp` survive only when every part agrees, because a merged
  result spans several runner pods.

  Args:
    results: The per-file or per-worker results, in the order they should be added.

  Returns:
    The merged result, or None when there was nothing to merge.
  """
  parts = list(results)
  if not parts:
    return None
  if len(parts) == 1:
    return parts[0]

  rows: list[TestRow] = []
  collected = skipped = failed = errored = 0
  junit_seconds = 0.0
  reported = 0
  has_reported = False
  suite_seconds = 0.0
  has_suite_seconds = False
  files: list[str] = []
  hostnames = {p.hostname for p in parts}
  timestamps = {p.timestamp for p in parts}

  for part in parts:
    collected += part.collected
    skipped += part.skipped
    failed += part.failed
    errored += part.errored
    junit_seconds += part.junit_seconds
    rows.extend(part.tests)
    files.extend(part.files)
    if part.reported_tests is not None:
      reported += part.reported_tests
      has_reported = True
    if part.suite_seconds is not None:
      suite_seconds += part.suite_seconds
      has_suite_seconds = True

  return SuiteResult(
      collected=collected,
      skipped=skipped,
      executed=collected - skipped,
      junit_seconds=junit_seconds,
      tests=rows,
      failed=failed,
      errored=errored,
      reported_tests=reported if has_reported else None,
      suite_seconds=suite_seconds if has_suite_seconds else None,
      hostname=hostnames.pop() if len(hostnames) == 1 else None,
      timestamp=timestamps.pop() if len(timestamps) == 1 else None,
      files=tuple(files),
  )


def _check_xml_sizes(
    archive: zipfile.ZipFile, members: list[str], artifact_name: str, run_left: int | None = None
) -> int:
  """Refuses a zip whose XML files would unpack too big, before any of them is read.

  The sizes are the ones the zip declares. They can be trusted, because `_read_member` takes no
  more than the declared size from a file. A name that appears twice is read twice, and each
  read gets the last file of that name, so it is counted twice here too.

  Args:
    archive: The open zip.
    members: The names of the XML files about to be read, repeats included.
    artifact_name: Name used in error messages.
    run_left: How many bytes of XML the rest of the run may still unpack, or None when the zip
      is not read as part of a run.

  Returns:
    The total declared size of the XML files, which the caller takes off the run's budget.

  Raises:
    JUnitError: One XML file is over `MAX_XML_BYTES`, all of them together are over
      `MAX_ZIP_XML_BYTES`, or they are more than `run_left`.
  """
  total = 0
  for member in members:
    size = archive.getinfo(member).file_size
    if size > MAX_XML_BYTES:
      raise JUnitError(
          f"{artifact_name}: {member} unpacks to {size:,} bytes; the limit for one XML file is {MAX_XML_BYTES:,} bytes"
      )
    total += size
  if total > MAX_ZIP_XML_BYTES:
    raise JUnitError(
        f"{artifact_name}: all XML files together unpack to {total:,} bytes; the limit is {MAX_ZIP_XML_BYTES:,} bytes"
    )
  if run_left is not None and total > run_left:
    raise JUnitError(
        f"{artifact_name}: its XML files unpack to {total:,} bytes, but the limit for the XML of one run is "
        f"{MAX_RUN_XML_BYTES:,} bytes and only {run_left:,} bytes of it are left"
    )
  return total


def _error_label(error: Exception) -> str:
  """Names the type of an error for a message, e.g. `zlib.error` or `ValueError`.

  The class name alone would read as just "error" for the one that zlib raises.

  Args:
    error: The error.

  Returns:
    The class name, with its module in front unless it is a built-in one.
  """
  kind = type(error)
  return kind.__name__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__name__}"


def _read_member(archive: zipfile.ZipFile, member: str, artifact_name: str) -> bytes:
  """Reads one XML file out of the zip, taking no more bytes than the zip says it has.

  `ZipFile.read` never hands back more than the declared size either, but it asks the
  decompressor for up to 1 GB in one call and only then cuts the result down. A file that lies
  about its size could use that much memory for a moment. Asking for the declared size keeps
  the memory near the size that was checked against the limits.

  The read asks for one byte more than the declared size. That makes `zipfile` reach the end of
  the file and run its CRC check, even when the declared size is zero. A file that holds more
  than it declares fails that check, which is how it is found out.

  `zipfile`, `zlib`, `bz2` and `lzma` each raise their own kinds of error on damaged data, and
  `zipfile` raises `RuntimeError` for an encrypted file and `NotImplementedError` for a method
  it does not know. A list of them would always be one short, and an error that escapes
  crashes the tick. So any error from these two calls means the same thing: this file cannot
  be read.

  Args:
    archive: The open zip.
    member: The name of the file inside the zip.
    artifact_name: Name used in error messages.

  Returns:
    The bytes of the file.

  Raises:
    JUnitError: The file cannot be read.
  """
  declared = archive.getinfo(member).file_size
  try:
    with archive.open(member) as handle:
      return handle.read(declared + 1)
  except Exception as exc:
    raise JUnitError(f"{artifact_name}: cannot read {member} out of the zip ({_error_label(exc)}: {exc})") from exc


def _parse_zip(
    data: bytes, flavor: str, artifact_name: str, run_xml_left: int | None
) -> tuple[dict[str, SuiteResult], int]:
  """Does the work of `parse_artifact_zip`, and also says how much XML the zip declared.

  Args:
    data: The raw zip bytes.
    flavor: The flavor the artifact belongs to.
    artifact_name: Name used in error messages.
    run_xml_left: How many bytes of XML the rest of the run may still unpack, or None when the
      zip is not read as part of a run.

  Returns:
    (suite id -> result, the total declared size of the zip's XML files).

  Raises:
    JUnitError: See `parse_artifact_zip`; also when the XML is more than `run_xml_left`.
  """
  try:
    archive = zipfile.ZipFile(io.BytesIO(data))
  except Exception as exc:
    # More than BadZipFile can come out here: a damaged file name raises UnicodeDecodeError.
    raise JUnitError(f"{artifact_name}: not a readable zip ({_error_label(exc)}: {exc})") from exc

  by_suite: dict[str, list[SuiteResult]] = {}
  with archive:
    members = sorted(n for n in archive.namelist() if n.lower().endswith(".xml") and not n.endswith("/"))
    declared_bytes = _check_xml_sizes(archive, members, artifact_name, run_xml_left)
    for member in members:
      raw = _read_member(archive, member, artifact_name)
      suite_id, _ = suite_id_for_file(member, flavor)
      by_suite.setdefault(suite_id, []).append(parse_junit_xml(raw, file_name=member))

  merged: dict[str, SuiteResult] = {}
  for suite_id, parts in by_suite.items():
    result = merge_suite_results(parts)
    if result is not None:
      merged[suite_id] = result
  return merged, declared_bytes


def parse_artifact_zip(data: bytes, flavor: str, artifact_name: str = "<artifact>") -> dict[str, SuiteResult]:
  """Reads every JUnit file inside one artifact zip.

  Pure: takes the zip bytes, so tests can build one in memory. A zip may hold several XML
  files; files of the same suite are merged, and the decoupled pass is kept apart under its
  own suite id. The size limits `MAX_XML_BYTES` and `MAX_ZIP_XML_BYTES` are checked first, so
  a zip bomb is refused without being unpacked. Files that are not XML are never read, so
  they do not count. The limit for a whole run, `MAX_RUN_XML_BYTES`, belongs to
  `read_run_tests`, which is the only code that sees more than one zip.

  Args:
    data: The raw zip bytes.
    flavor: The flavor the artifact belongs to.
    artifact_name: Name used in error messages.

  Returns:
    Suite id -> result. Empty when the zip holds no XML file at all, which the caller reports
    as `upload_empty`.

  Raises:
    JUnitError: The zip cannot be opened, one of its XML files cannot be unpacked or is not
      valid JUnit, or the XML files are over a size limit. Whatever `zipfile`, `zlib` or `lzma`
      raise about damaged data, the caller gets a `JUnitError`.
  """
  return _parse_zip(data, flavor, artifact_name, None)[0]


def list_test_artifacts(client: GitHubClientLike, run_id: int) -> list[ArtifactRef]:
  """Lists the `test-results-*` artifacts of a workflow run.

  Args:
    client: The GitHub client.
    run_id: The workflow run id.

  Returns:
    One reference per test-result artifact, newest first as GitHub returns them. Artifacts
    that are not ours (`maxtext-wheel`, `notebook-outputs-*`) are dropped. Expired artifacts
    are kept, so the caller can tell "expired" apart from "never published".

  Raises:
    JUnitError: The artifacts endpoint returned something that is not a list of objects, or the
      run lists more than `MAX_RUN_ARTIFACTS` artifacts of any kind. The listing is stopped soon
      after the cap, so a list that long is cut short and must not be read as the whole run.
  """
  payloads = client.paginate(
      f"actions/runs/{run_id}/artifacts", "artifacts", per_page=100, max_items=MAX_RUN_ARTIFACTS + 1
  )
  if len(payloads) > MAX_RUN_ARTIFACTS:
    raise JUnitError(
        f"run {run_id}: it lists more than {MAX_RUN_ARTIFACTS} artifacts, so none were downloaded "
        "(each download costs a request from the hourly budget that the whole repository shares)"
    )
  refs: list[ArtifactRef] = []
  for payload in payloads:
    if not isinstance(payload, dict):
      raise JUnitError(f"run {run_id}: artifacts endpoint returned a {type(payload).__name__}, expected an object")
    ref = ArtifactRef.from_api(payload)
    if ref is not None:
      refs.append(ref)
  return refs


def _limit_text(max_bytes: int, run_bound: bool) -> str:
  """Words the limit of one download, and says so when it is what is left of the run's budget.

  Args:
    max_bytes: The limit of this download.
    run_bound: True when `max_bytes` is what is left of the run's budget, not the limit for one zip.

  Returns:
    The limit, ready to go into a message.
  """
  if run_bound:
    return f"{max_bytes:,} bytes, which is what is left of the run's {MAX_RUN_ARTIFACT_BYTES:,}-byte limit"
  return f"{max_bytes:,} bytes"


def _download_artifact(client: GitHubClientLike, ref: ArtifactRef, max_bytes: int, run_bound: bool = False) -> bytes:
  """Downloads the zip of one artifact, and refuses a zip that is bigger than `max_bytes`.

  The size that GitHub lists is checked first, so a zip that is listed as too big is not
  downloaded at all. But the listed size is only a note: it reads as 0 when the API leaves it
  out, and the download is not bound by it. So the client is also asked to count the bytes that
  really arrive and to stop at `max_bytes`, and the length of what comes back is checked once
  more here, for a client that does not keep to the limit.

  A `TypeError` or an `AttributeError` from the client is a bug in the code, not a bad upload, so
  it is not turned into a `JUnitError`: it is allowed to show. Any other error is.

  Args:
    client: The GitHub client.
    ref: The artifact to download.
    max_bytes: The most bytes to accept.
    run_bound: True when `max_bytes` is what is left of the run's download budget, so that the
      messages say so instead of calling it the limit for one zip.

  Returns:
    The bytes of the zip.

  Raises:
    JUnitError: The zip is over the limit, or the download failed.
  """
  limit = _limit_text(max_bytes, run_bound)
  if ref.size_in_bytes > max_bytes:
    raise JUnitError(
        f"{ref.name}: the zip is {ref.size_in_bytes:,} bytes and the limit is {limit}, so it was not downloaded"
    )
  try:
    data = client.get_bytes(ref.download_url, max_bytes=max_bytes)
  except JUnitError:
    raise
  except (TypeError, AttributeError):
    raise
  except Exception as exc:
    note = f"; the limit for this download was {limit}" if run_bound else ""
    raise JUnitError(f"{ref.name}: download failed ({type(exc).__name__}: {exc}){note}") from exc
  if len(data) > max_bytes:
    raise JUnitError(f"{ref.name}: the zip came back {len(data):,} bytes and the limit is {limit}")
  return data


def read_artifact_suites(client: GitHubClientLike, ref: ArtifactRef) -> dict[str, SuiteResult]:
  """Downloads one artifact and parses every JUnit file in it.

  A zip that GitHub lists as bigger than `MAX_ARTIFACT_BYTES` is refused before the download,
  and a download is stopped as soon as more than `MAX_ARTIFACT_BYTES` have really arrived.

  Args:
    client: The GitHub client.
    ref: The artifact to download.

  Returns:
    Suite id -> result. Empty when the upload held no XML file.

  Raises:
    JUnitError: The zip is over the size limit (listed, or counted as it arrives), the
      download failed, or the zip or one of its files is unreadable or over a size limit.
      Expired artifacts are the caller's business: check `ref.expired` before calling.
  """
  data = _download_artifact(client, ref, MAX_ARTIFACT_BYTES)
  return parse_artifact_zip(data, ref.flavor, artifact_name=ref.name)


def _suites_asked_about(flavors: Iterable[str] | None) -> list[str]:
  """Returns the suite ids to report on: the given flavors plus their nested suites."""
  asked = list(flavors) if flavors is not None else list(KNOWN_FLAVORS)
  for nested_id, parent in NESTED_SUITES.items():
    if parent in asked and nested_id not in asked:
      asked.append(nested_id)
  return asked


def _ensure_entry(suites: dict[str, SuiteEntry], suite_id: str) -> SuiteEntry:
  """Adds an entry for a suite, and for any suite nested inside it, if they are missing.

  A flavor is entered as soon as one of its artifacts is seen, before anything is read out
  of that artifact. That is what lets an expired or empty upload of a flavor nobody asked
  about - the tpu7x flavors, which only run outside pull requests - still be reported with
  the reason its payload could not be read, instead of vanishing from the run.

  Args:
    suites: The suite id -> entry map being built.
    suite_id: The suite to make sure exists.

  Returns:
    The entry for `suite_id`.
  """
  entry = suites.get(suite_id)
  if entry is None:
    entry = SuiteEntry(suite_id=suite_id, reason=REASON_NO_FILE, nested_in=NESTED_SUITES.get(suite_id))
    suites[suite_id] = entry
  for nested_id, parent in NESTED_SUITES.items():
    if parent == suite_id and nested_id not in suites:
      suites[nested_id] = SuiteEntry(suite_id=nested_id, reason=REASON_NO_FILE, nested_in=parent)
  return entry


def _finish_entry(entry: SuiteEntry, missing: dict[tuple[str, int], str], published: dict[str, set[int]]) -> None:
  """Fills in one suite's merged result, its missing workers and its reason code.

  The reason is worked out per worker, not per flavor. A flavor whose workers all failed to
  publish gets that failure as its reason; a flavor where at least one worker published gets
  a result plus the list of workers that are missing from it. A suite with no rows of its own
  in an artifact that did publish - the decoupled pass on a run where its step did not
  execute - is `no_file_published`, and never borrows another worker's failure.

  Args:
    entry: The entry to complete. Modified in place.
    missing: (flavor, worker) -> REASON_* for every payload that could not be read.
    published: Flavor -> the worker numbers whose artifact yielded at least one suite.
  """
  parent = entry.nested_in or entry.suite_id
  entry.missing_workers = {worker: reason for (flavor, worker), reason in sorted(missing.items()) if flavor == parent}

  if entry.per_worker:
    entry.result = merge_suite_results(entry.per_worker[worker] for worker in sorted(entry.per_worker))
    entry.reason = None
    return

  entry.result = None
  if published.get(parent):
    # Some worker of this flavor published a readable file that simply held nothing for this
    # suite, so nothing was lost in transit: the pass did not write a file.
    entry.reason = REASON_NO_FILE
    return

  reasons = set(entry.missing_workers.values())
  if reasons == {REASON_ARTIFACT_EXPIRED}:
    entry.reason = REASON_ARTIFACT_EXPIRED
  elif reasons == {REASON_UPLOAD_EMPTY}:
    entry.reason = REASON_UPLOAD_EMPTY
  elif REASON_ARTIFACT_EXPIRED in reasons:
    # Mixed causes across the workers. Expiry is the one no later run can undo, so it is the
    # one named here; `missing_workers` still carries the cause of each worker.
    entry.reason = REASON_ARTIFACT_EXPIRED
  else:
    entry.reason = REASON_NO_FILE


def read_run_tests(client: GitHubClientLike, run_id: int, flavors: Iterable[str] | None = None) -> RunTests:
  """Reads every test result of one workflow run.

  Downloads each `test-results-*` artifact, parses the JUnit files inside, and merges the
  workers of a flavor into one result. The nested `decoupled` pass is reported separately and
  is never added into `cpu-unit`, whose totals already include those tests.

  A suite with nothing to show gets a None result and a reason code, never a zero:

    * `no_file_published` - the run has no artifact for that flavor, or the artifacts it
      does have carry no file for this suite. Normal for the TPU Pathways jobs, which run
      pytest without `--junitxml`, for tpu7x flavors on a pull-request run, and for the
      decoupled pass on a run where its step did not execute.
    * `artifact_expired` - GitHub has deleted the payload; artifacts live about one day.
    * `upload_empty` - the artifact exists but holds no XML file.

  When only some workers of a flavor are missing, the entry keeps the surviving workers'
  merged result and lists the rest in `SuiteEntry.missing_workers`. The total is then a
  partial one and must not be read as a drop in the test count.

  Args:
    client: The GitHub client.
    run_id: The workflow run id.
    flavors: The flavors to report on. Defaults to `KNOWN_FLAVORS`. Nested suites of any
      listed flavor are added automatically.

  Returns:
    The run's results, with an entry for every flavor asked about plus every extra flavor
    found in the artifacts.

  Raises:
    JUnitError: The run lists more than `MAX_RUN_ARTIFACTS` artifacts, an artifact could not be
      downloaded or parsed, the run's artifacts together are over `MAX_RUN_ARTIFACT_BYTES`, as
      listed or as downloaded, or the XML in them is over `MAX_RUN_XML_BYTES` together. Rate
      limits, retries and transport errors are the client's business.
  """
  refs = list_test_artifacts(client, run_id)

  # One run can have many artifacts, each under the per-artifact limit. The listed sizes are only
  # a note, but a run that lists too much is not worth starting.
  listed_bytes = sum(ref.size_in_bytes for ref in refs if not ref.expired)
  if listed_bytes > MAX_RUN_ARTIFACT_BYTES:
    raise JUnitError(
        f"run {run_id}: its test-result artifacts add up to {listed_bytes:,} bytes and the limit for one run is "
        f"{MAX_RUN_ARTIFACT_BYTES:,} bytes, so none were downloaded"
    )
  # What the run may still download. Every download is counted by the bytes that really came back.
  left = MAX_RUN_ARTIFACT_BYTES
  # What the run may still unpack. The parsed results of every zip stay in memory until the run is
  # done, so the XML is counted too, by the size each zip declares, before any of it is read.
  xml_left = MAX_RUN_XML_BYTES

  suites: dict[str, SuiteEntry] = {}
  for suite_id in _suites_asked_about(flavors):
    _ensure_entry(suites, suite_id)

  missing: dict[tuple[str, int], str] = {}
  published: dict[str, set[int]] = {}

  for ref in sorted(refs, key=lambda r: (r.flavor, r.worker)):
    _ensure_entry(suites, ref.flavor)
    if ref.expired:
      missing[(ref.flavor, ref.worker)] = REASON_ARTIFACT_EXPIRED
      continue
    data = _download_artifact(client, ref, min(MAX_ARTIFACT_BYTES, left), run_bound=left < MAX_ARTIFACT_BYTES)
    left -= len(data)
    found, xml_bytes = _parse_zip(data, ref.flavor, ref.name, xml_left)
    xml_left -= xml_bytes
    if not found:
      missing[(ref.flavor, ref.worker)] = REASON_UPLOAD_EMPTY
      continue
    published.setdefault(ref.flavor, set()).add(ref.worker)
    for suite_id, result in found.items():
      for row in result.tests:
        row.worker = ref.worker
      entry = _ensure_entry(suites, suite_id)
      entry.per_worker[ref.worker] = merge_suite_results(
          [p for p in (entry.per_worker.get(ref.worker), result) if p is not None]
      )

  for entry in suites.values():
    _finish_entry(entry, missing, published)

  return RunTests(run_id=run_id, suites=suites, artifacts=refs)
