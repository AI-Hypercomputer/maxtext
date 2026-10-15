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

"""Offline tests for how `collector.junit` treats a hostile or broken test-result zip.

Test-result zips come from any pull request, including pull requests from forks, so a zip
cannot be trusted. Two kinds of trouble would hurt the tick:

  * A zip that is too big. A tiny zip can unpack to gigabytes of XML (a zip bomb), and an
    upload can simply be huge. Either one could run the tick out of memory.
  * A zip that cannot be unpacked. `zipfile`, `zlib` and `lzma` raise many kinds of error on
    damaged data. A `JUnitError` only costs one run its tests, but any other kind of error
    that escapes crashes the whole tick.

`junit.py` has three size limits, and each is checked here:

  1. `MAX_ARTIFACT_BYTES` is the largest zip it will download. It is checked on the size that
     GitHub reports, so a zip over the limit is never downloaded.
  2. `MAX_XML_BYTES` is the largest single XML file inside a zip, once unpacked.
  3. `MAX_ZIP_XML_BYTES` is all the XML files of one zip together, once unpacked.

Limits 2 and 3 are checked on the sizes that the zip declares, before any file is read. Each
file is then read with a cap at its declared size, so a file that lies about its size cannot
fill the memory. Python's `zipfile` also fails the CRC check on a file that holds more than it
declares.

The damaged zips are built in memory, one for each kind of failure that was seen to escape. Then
real zips are damaged at random (with a fixed seed) to look for kinds that nobody thought of.

More guards close ways around those limits:

  * The download is capped on the bytes that really arrive, not only on the size that GitHub
    lists, which can be missing (it then reads as 0) or wrong. `github_download_test.py` checks
    the streaming itself; the tests here check that `junit.py` asks for the cap.
  * An XML file with a DOCTYPE is refused. Entities declared in one can expand a small file to
    many times its size, past the per-file limit. JUnit files never have a DOCTYPE.
  * One run's artifacts are capped together, however many there are: the bytes downloaded
    (`MAX_RUN_ARTIFACT_BYTES`) and the XML they unpack to (`MAX_RUN_XML_BYTES`). Zips pack
    well, and the parsed results of a whole run stay in memory, so the second one is what
    bounds the memory.
  * A run may list at most `MAX_RUN_ARTIFACTS` artifacts. Each download costs a request from an
    hourly budget that the whole repository shares.
  * An error that is a bug in the code (`TypeError`, `AttributeError`) is not turned into "tests
    lost"; every other error from the client is.

Most tests set the three limits to small numbers, so no test has to build a big zip. A few use
the real limits and real sizes instead: a normal zip still parses, a 21 MB XML file squeezed
into a small zip is refused, an artifact that GitHub lists as 51 MB is not downloaded, and a file
that claims 1,000 bytes but unpacks to 64 MB is read without unpacking 64 MB. Nothing here uses
the network: the GitHub client is a small fake that records the downloads it is asked for.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/junit_limits_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/junit_limits_test.py
"""

from __future__ import annotations

import io
import random
import struct
import sys
import tracemalloc
import unittest
import warnings
import zipfile
import zlib
from pathlib import Path
from typing import Any
from unittest import mock

try:
  import lzma
except ImportError:  # A Python built without liblzma.
  lzma = None

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import junit

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The scheduled run of 2026-09-01, and its smallest real JUnit file: nine cases, seven skipped.
RUN_ID = 33468578834
SMALL_SUITE_FILE = "tpu-post-training-integration-1.xml"
FLAVOR = "tpu-post-training-integration"
ARTIFACT_NAME = f"test-results-{FLAVOR}-1-{RUN_ID}"
SMALL_SUITE_CASES = 9

# The smallest JUnit document the parser accepts. `xml_of_size` pads it with spaces, which are
# allowed after the root element, so a padded file still parses when nothing refuses it.
EMPTY_SUITE = b'<testsuites><testsuite name="pytest" tests="0"></testsuite></testsuites>'

MEGABYTE = 1024 * 1024

# The header in front of every file inside a zip: signature, version, flags, compression method,
# time, date, CRC, packed size, unpacked size, name length, extra-field length.
LOCAL_HEADER_FORMAT = "<4sHHHHHLLLHH"
LOCAL_HEADER_SIZE = struct.calcsize(LOCAL_HEADER_FORMAT)
# The directory at the end of a zip has one entry per file, 46 bytes before the name. The fields
# used below sit at these offsets inside an entry: flags 8, method 10, packed size 20, unpacked 24.
DIRECTORY_ENTRY_SIZE = 46


def xml_of_size(size: int) -> bytes:
  """Builds a valid JUnit file of an exact size.

  Args:
    size: The number of bytes wanted, at least the length of `EMPTY_SUITE`.

  Returns:
    An XML document that parses, with trailing spaces to make it `size` bytes long.

  Raises:
    ValueError: `size` is too small to hold a document.
  """
  if size < len(EMPTY_SUITE):
    raise ValueError(f"{size} bytes is too small for a JUnit document")
  return EMPTY_SUITE + b" " * (size - len(EMPTY_SUITE))


def build_zip(members: list[tuple[str, bytes]], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
  """Builds a zip in memory.

  Args:
    members: (name, content) pairs, written in this order. A name may repeat.
    compression: The zip compression method. Deflate is what GitHub uses.

  Returns:
    The zip bytes.
  """
  buffer = io.BytesIO()
  with warnings.catch_warnings():
    # zipfile warns when a name is written twice, which one test does on purpose.
    warnings.simplefilter("ignore", UserWarning)
    with zipfile.ZipFile(buffer, "w", compression) as archive:
      for name, content in members:
        archive.writestr(name, content)
  return buffer.getvalue()


def make_ref(size_in_bytes: int) -> junit.ArtifactRef:
  """Builds the reference to an artifact that GitHub says is `size_in_bytes` big.

  Args:
    size_in_bytes: The compressed size, as the artifacts endpoint reports it.

  Returns:
    The reference, not expired.
  """
  return junit.ArtifactRef(
      name=ARTIFACT_NAME,
      artifact_id=1,
      flavor=FLAVOR,
      worker=1,
      expired=False,
      size_in_bytes=size_in_bytes,
      download_url="https://api.github.com/repos/example/example/actions/artifacts/1/zip",
      run_id=RUN_ID,
  )


def small_suite_zip(compression: int = zipfile.ZIP_DEFLATED) -> bytearray:
  """Builds a zip that holds the one small real fixture, as bytes that a test can damage.

  Args:
    compression: The zip compression method.

  Returns:
    The zip bytes.
  """
  return bytearray(build_zip([(SMALL_SUITE_FILE, (FIXTURES / SMALL_SUITE_FILE).read_bytes())], compression))


def packed_data_offset(data: bytes) -> int:
  """Finds where the packed bytes of the first file start.

  Args:
    data: A zip.

  Returns:
    The offset just after the first file's header, name and extra field.
  """
  name_length, extra_length = struct.unpack_from(LOCAL_HEADER_FORMAT, data, 0)[-2:]
  return LOCAL_HEADER_SIZE + name_length + extra_length


def directory_offset(data: bytes) -> int:
  """Finds the directory entry of the last file in a zip.

  Args:
    data: A zip.

  Returns:
    The offset of the entry, which starts with the signature `PK 01 02`.
  """
  return data.rindex(b"PK\x01\x02")


def zip_with_corrupt_deflate() -> bytes:
  """Builds a zip whose packed bytes are not valid deflate.

  Returns:
    The zip. Block type 3 is reserved, so zlib refuses the very first byte.
  """
  data = small_suite_zip()
  data[packed_data_offset(data)] |= 0b111
  return bytes(data)


def zip_with_compression_method(method: int) -> bytes:
  """Builds a zip that says its file is packed with a method of the given number.

  Args:
    method: The method number to write into the file header and the directory.

  Returns:
    The zip.
  """
  data = small_suite_zip()
  struct.pack_into("<H", data, 8, method)
  struct.pack_into("<H", data, directory_offset(data) + 10, method)
  return bytes(data)


def zip_with_encrypted_flag() -> bytes:
  """Builds a zip whose directory says the file is encrypted.

  Returns:
    The zip. Bit 0 of the flags is the encryption bit.
  """
  data = small_suite_zip()
  struct.pack_into("<H", data, directory_offset(data) + 8, 0x0001)
  return bytes(data)


def zip_claiming_more_data_than_it_holds() -> bytes:
  """Builds a zip whose directory gives its file 10 MB more data than there is.

  Returns:
    The zip, with the file stored as it is.
  """
  data = small_suite_zip(zipfile.ZIP_STORED)
  at = directory_offset(data) + 20
  packed, unpacked = struct.unpack_from("<LL", data, at)
  struct.pack_into("<LL", data, at, packed + 10 * MEGABYTE, unpacked + 10 * MEGABYTE)
  return bytes(data)


def zip_with_corrupt_lzma() -> bytes:
  """Builds a zip whose packed bytes are not valid LZMA.

  Returns:
    The zip.
  """
  data = small_suite_zip(zipfile.ZIP_LZMA)
  start = packed_data_offset(data)
  for position in range(start + 4, start + 24):
    data[position] ^= 0xFF
  return bytes(data)


def zip_with_shifted_directory() -> bytes:
  """Builds a zip whose directory offset is 100 bytes too big.

  Returns:
    The zip. Every file then seems to start 100 bytes before the zip does.
  """
  data = small_suite_zip()
  at = data.rindex(b"PK\x05\x06") + 16
  (offset,) = struct.unpack_from("<L", data, at)
  struct.pack_into("<L", data, at, offset + 100)
  return bytes(data)


def zip_with_bad_name_encoding() -> bytes:
  """Builds a zip that says its file name is UTF-8 and then breaks the name.

  Returns:
    The zip.
  """
  data = small_suite_zip()
  struct.pack_into("<H", data, 6, 0x0800)
  entry = directory_offset(data)
  struct.pack_into("<H", data, entry + 8, 0x0800)
  data[LOCAL_HEADER_SIZE] = 0xFF
  data[entry + DIRECTORY_ENTRY_SIZE] = 0xFF
  return bytes(data)


def zip_that_lies_about_its_size(content: bytes, declared: int, compression: int, forge_crc: bool = False) -> bytes:
  """Builds a zip whose directory gives a wrong unpacked size for its one file.

  Args:
    content: What the file really holds.
    declared: The unpacked size that the directory claims.
    compression: The zip compression method.
    forge_crc: If true, the directory also gets the CRC of the first `declared` bytes, as a
      careful attacker would write it. If false, it keeps the CRC of the real content.

  Returns:
    The zip.
  """
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w", compression) as archive:
    archive.writestr("lie.xml", content)
    # The directory is written when the zip closes, from these fields.
    info = archive.infolist()[0]
    info.file_size = declared
    if forge_crc:
      info.CRC = zlib.crc32(content[:declared])
  return buffer.getvalue()


def zips_to_damage() -> dict[str, bytes]:
  """Builds real zips in each compression method that this Python can write.

  Returns:
    Label -> zip. Each zip has two XML files and one text file.
  """
  raw = (FIXTURES / SMALL_SUITE_FILE).read_bytes()
  members = [(SMALL_SUITE_FILE, raw), ("extra/second.xml", raw), ("notes.txt", b"hello")]
  zips = {"deflate": build_zip(members), "stored": build_zip(members, zipfile.ZIP_STORED)}
  for label, method in (("bzip2", zipfile.ZIP_BZIP2), ("lzma", zipfile.ZIP_LZMA)):
    try:
      zips[label] = build_zip(members, method)
    except RuntimeError:  # This Python was built without the module.
      continue
  return zips


def damage(original: bytes, rng: random.Random) -> bytes:
  """Copies a zip and changes a few bytes of the copy.

  Half of the changes land in the first 100 bytes (the first file's header) or in the last 200
  (the directory), because that is where one changed byte does the most harm.

  Args:
    original: The zip.
    rng: The source of randomness, seeded by the caller so a run can be repeated.

  Returns:
    The damaged copy.
  """
  data = bytearray(original)
  for _ in range(rng.choice([1, 1, 2, 3, 5, 8])):
    where = rng.random()
    if where < 0.25:
      position = rng.randrange(min(len(data), 100))
    elif where < 0.5:
      position = len(data) - 1 - rng.randrange(min(len(data), 200))
    else:
      position = rng.randrange(len(data))
    data[position] = rng.randrange(256)
  return bytes(data)


def artifact_payload(worker: int, size: int, expired: bool = False) -> dict[str, Any]:
  """Builds one entry of the artifacts endpoint, as GitHub lists a test-result artifact.

  Args:
    worker: The parallel worker number, which is part of the artifact name.
    size: The compressed size GitHub lists for it.
    expired: Whether GitHub has already deleted the payload.

  Returns:
    The entry.
  """
  return {
      "name": f"test-results-{FLAVOR}-{worker}-{RUN_ID}",
      "id": worker,
      "expired": expired,
      "size_in_bytes": size,
      "archive_download_url": f"https://api.github.com/repos/example/example/actions/artifacts/{worker}/zip",
  }


class FakeClient:
  """Stands in for `github.GitHubClient`, and writes down every download it is asked for."""

  def __init__(
      self,
      body: bytes = b"",
      payloads: list[dict[str, Any]] | None = None,
      enforce_limit: bool = False,
      error: Exception | None = None,
  ) -> None:
    """Builds a client that answers every download with the same bytes.

    Args:
      body: The bytes `get_bytes` returns.
      payloads: What the artifacts endpoint lists. None means that listing is never expected.
      enforce_limit: If true, act like `GitHubClient` and raise when the body is longer than the
        `max_bytes` it was given. If false, ignore the limit, as a client that predates it would.
      error: If set, `get_bytes` raises it, after writing the download down.
    """
    self.body = body
    self.payloads = payloads
    self.enforce_limit = enforce_limit
    self.error = error
    self.downloads: list[str] = []
    self.max_bytes_seen: list[int | None] = []
    self.max_items_seen: list[int | None] = []
    self.listed_counts: list[int] = []

  def paginate(self, path: str, key: str, *, max_items: int | None = None, **params: object) -> list:
    """Lists the payloads this client was built with, a page of 100 at a time like the real one.

    Args:
      path: Ignored.
      key: Ignored.
      max_items: Stop after the page that makes the list this long, as `GitHubClient.paginate` does.
      **params: Ignored.

    Returns:
      The payloads, or the first whole pages of them when `max_items` is set.

    Raises:
      AssertionError: The client was built with no payloads, so no listing was expected.
    """
    if self.payloads is None:
      raise AssertionError(f"read_artifact_suites listed {path!r}, which it has no reason to do")
    self.max_items_seen.append(max_items)
    listed: list[dict[str, Any]] = []
    for start in range(0, len(self.payloads), 100):
      listed.extend(self.payloads[start : start + 100])
      if max_items is not None and len(listed) >= max_items:
        break
    self.listed_counts.append(len(listed))
    return listed

  def get_bytes(self, url: str, *, max_bytes: int | None = None) -> bytes:
    """Records the download and hands back the canned bytes.

    Args:
      url: The URL asked for.
      max_bytes: The most bytes the caller will accept.

    Returns:
      The bytes this client was built with.

    Raises:
      RuntimeError: `enforce_limit` is on and the body is longer than `max_bytes`.
      Exception: Whatever `error` was set to.
    """
    self.downloads.append(url)
    self.max_bytes_seen.append(max_bytes)
    if self.error is not None:
      raise self.error
    if self.enforce_limit and max_bytes is not None and len(self.body) > max_bytes:
      raise RuntimeError(f"GET {url} sent more than {max_bytes:,} bytes")
    return self.body


class LimitTestCase(unittest.TestCase):
  """Base class: lets a test set the three limits for itself alone, and parse a zip."""

  def set_limits(self, **limits: int) -> None:
    """Replaces limits of `junit` until the test ends.

    The code reads the constants when it runs, so a changed value takes effect at once.

    Args:
      **limits: Constant name -> new value, e.g. `MAX_XML_BYTES=1000`.
    """
    for name, value in limits.items():
      patch = mock.patch.object(junit, name, value)
      patch.start()
      self.addCleanup(patch.stop)

  def parse(self, data: bytes) -> dict[str, junit.SuiteResult]:
    """Parses a zip the way `read_artifact_suites` does.

    Args:
      data: The zip bytes.

    Returns:
      Suite id -> result.
    """
    return junit.parse_artifact_zip(data, FLAVOR, artifact_name=ARTIFACT_NAME)


class ArtifactSizeTest(LimitTestCase):
  """`read_artifact_suites` refuses a big zip before it downloads anything."""

  def test_an_artifact_over_the_limit_is_not_downloaded(self) -> None:
    """The size comes from the listing, so the refusal costs no download at all."""
    self.set_limits(MAX_ARTIFACT_BYTES=1000)
    client = FakeClient(body=build_zip([("results.xml", xml_of_size(200))]))

    with self.assertRaises(junit.JUnitError) as caught:
      junit.read_artifact_suites(client, make_ref(1001))

    self.assertEqual(client.downloads, [], "a zip over the limit must not be downloaded")
    message = str(caught.exception)
    self.assertIn(ARTIFACT_NAME, message)
    self.assertIn("1,001 bytes", message)
    self.assertIn("1,000 bytes", message)
    self.assertNotIn("left of the run", message, "this is the limit for one zip, not what is left of a run")

  def test_an_artifact_at_the_limit_is_downloaded_and_read(self) -> None:
    """The limit is the largest size that is allowed, so a zip of exactly that size passes."""
    self.set_limits(MAX_ARTIFACT_BYTES=1000)
    client = FakeClient(body=build_zip([("results.xml", xml_of_size(200))]))
    ref = make_ref(1000)

    found = junit.read_artifact_suites(client, ref)

    self.assertEqual(client.downloads, [ref.download_url])
    self.assertEqual(list(found), [FLAVOR])

  def test_a_zip_with_a_file_that_is_too_big_is_refused_after_the_download(self) -> None:
    """A small listed size says nothing about what is inside, so the contents are checked too."""
    self.set_limits(MAX_ARTIFACT_BYTES=1000, MAX_XML_BYTES=500)
    client = FakeClient(body=build_zip([("results.xml", xml_of_size(501))]))

    with self.assertRaises(junit.JUnitError) as caught:
      junit.read_artifact_suites(client, make_ref(400))

    self.assertEqual(len(client.downloads), 1)
    self.assertIn("results.xml", str(caught.exception))

  def test_an_artifact_bigger_than_the_real_limit_is_not_downloaded(self) -> None:
    """With the real limit: a zip of 51 MB is refused. Nothing is built, only a size is claimed."""
    client = FakeClient()

    with self.assertRaises(junit.JUnitError):
      junit.read_artifact_suites(client, make_ref(51 * MEGABYTE))

    self.assertEqual(client.downloads, [])

  def test_the_download_is_given_the_limit_to_count_against(self) -> None:
    """The listed size is only a note, so the client is asked to count the bytes that really arrive."""
    self.set_limits(MAX_ARTIFACT_BYTES=1000)
    client = FakeClient(body=build_zip([("results.xml", xml_of_size(200))]))

    junit.read_artifact_suites(client, make_ref(100))

    self.assertEqual(client.max_bytes_seen, [1000])

  def test_a_listed_size_of_zero_does_not_skip_the_limit(self) -> None:
    """GitHub's size reads as 0 when the API leaves it out. The count of real bytes still stops the download."""
    self.set_limits(MAX_ARTIFACT_BYTES=1000)
    client = FakeClient(body=b"x" * 5000, enforce_limit=True)

    with self.assertRaises(junit.JUnitError) as caught:
      junit.read_artifact_suites(client, make_ref(0))

    self.assertIn(ARTIFACT_NAME, str(caught.exception))
    self.assertIn("1,000 bytes", str(caught.exception))

  def test_a_client_that_ignores_the_limit_is_caught_afterwards(self) -> None:
    """If a client hands back more than it was allowed, the zip is refused before it is opened."""
    self.set_limits(MAX_ARTIFACT_BYTES=1000)
    client = FakeClient(body=build_zip([("results.xml", xml_of_size(1500))], zipfile.ZIP_STORED))

    with mock.patch.object(zipfile, "ZipFile", side_effect=AssertionError("the zip was opened")):
      with self.assertRaises(junit.JUnitError) as caught:
        junit.read_artifact_suites(client, make_ref(100))

    self.assertIn(ARTIFACT_NAME, str(caught.exception))
    self.assertIn("1,000 bytes", str(caught.exception))


class RunCapTest(LimitTestCase):
  """`read_run_tests` caps what one run may download, however many artifacts it has."""

  def setUp(self) -> None:
    """Sets limits that leave room for two zips per run, but not for three."""
    super().setUp()
    self.body = build_zip([("results.xml", xml_of_size(300))], zipfile.ZIP_STORED)
    self.size = len(self.body)
    self.set_limits(MAX_ARTIFACT_BYTES=self.size + 50, MAX_RUN_ARTIFACT_BYTES=2 * self.size + 100)

  def read(self, client: FakeClient) -> junit.RunTests:
    """Reads the one flavor of the run.

    Args:
      client: The fake client.

    Returns:
      The run's tests.
    """
    return junit.read_run_tests(client, RUN_ID, flavors=[FLAVOR])

  def test_artifacts_listed_over_the_run_cap_are_not_downloaded(self) -> None:
    client = FakeClient(self.body, payloads=[artifact_payload(worker, self.size) for worker in (1, 2, 3)])

    with self.assertRaises(junit.JUnitError) as caught:
      self.read(client)

    self.assertEqual(client.downloads, [], "a run over the cap must not download anything")
    message = str(caught.exception)
    self.assertIn(str(RUN_ID), message)
    self.assertIn(f"{3 * self.size:,} bytes", message)
    self.assertIn(f"{2 * self.size + 100:,} bytes", message)

  def test_a_run_under_the_cap_is_read(self) -> None:
    client = FakeClient(self.body, payloads=[artifact_payload(worker, self.size) for worker in (1, 2)])

    run = self.read(client)

    self.assertEqual(len(client.downloads), 2)
    self.assertIsNotNone(run.result_for(FLAVOR))

  def test_a_run_exactly_at_the_cap_is_read(self) -> None:
    self.set_limits(MAX_RUN_ARTIFACT_BYTES=2 * self.size)
    client = FakeClient(self.body, payloads=[artifact_payload(worker, self.size) for worker in (1, 2)])

    run = self.read(client)

    self.assertEqual(len(client.downloads), 2)
    self.assertIsNotNone(run.result_for(FLAVOR))

  def test_expired_artifacts_do_not_count_toward_the_cap(self) -> None:
    payloads = [
        artifact_payload(1, self.size),
        artifact_payload(2, self.size),
        artifact_payload(3, self.size, expired=True),
    ]
    client = FakeClient(self.body, payloads=payloads)

    run = self.read(client)

    self.assertEqual(len(client.downloads), 2, "the expired artifact is never downloaded")
    self.assertIsNotNone(run.result_for(FLAVOR))

  def test_listed_sizes_that_are_too_small_do_not_hide_a_big_run(self) -> None:
    """Each download is limited to what is left of the run's budget, and a zip over that is refused."""
    client = FakeClient(self.body, payloads=[artifact_payload(worker, 10) for worker in (1, 2, 3)])

    with self.assertRaises(junit.JUnitError) as caught:
      self.read(client)

    self.assertEqual(client.max_bytes_seen, [self.size + 50, self.size + 50, 100])
    self.assertIn(f"-3-{RUN_ID}", str(caught.exception))
    self.assertIn("what is left of the run's", str(caught.exception), "the limit here is the run's budget")

  def test_a_client_that_enforces_the_limit_stops_the_run_the_same_way(self) -> None:
    client = FakeClient(self.body, payloads=[artifact_payload(worker, 10) for worker in (1, 2, 3)], enforce_limit=True)

    with self.assertRaises(junit.JUnitError) as caught:
      self.read(client)

    self.assertIn("100 bytes", str(caught.exception))
    self.assertIn("what is left of the run's", str(caught.exception), "the limit here is the run's budget")
    self.assertEqual(len(client.downloads), 3)

  def test_a_zip_over_its_own_limit_inside_a_run_is_not_blamed_on_the_run_budget(self) -> None:
    """When the limit in force is the one for a single zip, the message must not call it the run's."""
    client = FakeClient(self.body, payloads=[artifact_payload(1, self.size + 60)])

    with self.assertRaises(junit.JUnitError) as caught:
      self.read(client)

    message = str(caught.exception)
    self.assertEqual(client.downloads, [])
    self.assertIn(f"the limit is {self.size + 50:,} bytes", message)
    self.assertNotIn("left of the run", message)

  def test_a_listed_size_that_does_not_fit_what_is_left_says_so(self) -> None:
    """The limit in the message is what is left of the run's budget, not the limit for one zip."""
    payloads = [artifact_payload(1, 10), artifact_payload(2, 10), artifact_payload(3, self.size)]
    client = FakeClient(self.body, payloads=payloads)

    with self.assertRaises(junit.JUnitError) as caught:
      self.read(client)

    message = str(caught.exception)
    self.assertEqual(len(client.downloads), 2, "the third zip is refused before it is downloaded")
    self.assertIn("the limit is 100 bytes", message)
    self.assertIn(f"what is left of the run's {2 * self.size + 100:,}-byte limit", message)

  def test_each_run_gets_its_own_budget(self) -> None:
    client = FakeClient(self.body, payloads=[artifact_payload(worker, self.size) for worker in (1, 2)])

    self.read(client)
    self.read(client)

    self.assertEqual(len(client.downloads), 4)


class ArtifactCountTest(LimitTestCase):
  """`read_run_tests` refuses a run with too many artifacts, before it downloads any of them.

  Every download starts with a request to api.github.com, which comes out of an hourly budget that
  the whole repository shares, so a run with thousands of tiny artifacts could use it up. The
  listing is cut short soon after the cap, so a run over the cap must never be read from what
  was listed: it is refused.
  """

  def setUp(self) -> None:
    """Sets a cap of 5 artifacts per run."""
    super().setUp()
    self.body = build_zip([("results.xml", xml_of_size(300))], zipfile.ZIP_STORED)
    self.set_limits(MAX_RUN_ARTIFACTS=5)

  def client_with(self, count: int, others: int = 0) -> FakeClient:
    """Builds a client whose run lists test-result artifacts and, maybe, other artifacts.

    Args:
      count: How many test-result artifacts the run lists.
      others: How many artifacts that are not test results it lists as well.

    Returns:
      The client.
    """
    payloads = [artifact_payload(worker, len(self.body)) for worker in range(1, count + 1)]
    for number in range(others):
      payloads.append(
          {
              "name": f"notebook-outputs-{number}",
              "id": 1000 + number,
              "expired": False,
              "size_in_bytes": 10,
              "archive_download_url": "https://api.github.com/repos/example/example/actions/artifacts/9/zip",
          }
      )
    return FakeClient(self.body, payloads=payloads)

  def read(self, client: FakeClient) -> junit.RunTests:
    """Reads the one flavor of the run.

    Args:
      client: The fake client.

    Returns:
      The run's tests.
    """
    return junit.read_run_tests(client, RUN_ID, flavors=[FLAVOR])

  def test_one_artifact_over_the_cap_refuses_the_run_with_no_download(self) -> None:
    client = self.client_with(6)

    with self.assertRaises(junit.JUnitError) as caught:
      self.read(client)

    self.assertEqual(client.downloads, [], "a run over the cap must not download anything")
    self.assertIn(str(RUN_ID), str(caught.exception))
    self.assertIn("more than 5 artifacts", str(caught.exception))

  def test_a_run_exactly_at_the_cap_is_read(self) -> None:
    client = self.client_with(5)

    run = self.read(client)

    self.assertEqual(len(client.downloads), 5)
    self.assertIsNotNone(run.result_for(FLAVOR))

  def test_the_listing_is_told_to_stop_soon_after_the_cap(self) -> None:
    client = self.client_with(5)

    self.read(client)

    self.assertEqual(client.max_items_seen, [6], "one more than the cap is enough to know the run is over it")

  def test_a_run_with_a_thousand_artifacts_is_refused_after_one_page(self) -> None:
    client = self.client_with(1000)

    with self.assertRaises(junit.JUnitError):
      self.read(client)

    self.assertEqual(client.listed_counts, [100], "the listing must stop at the first page past the cap")
    self.assertEqual(client.downloads, [])

  def test_a_run_one_artifact_past_a_page_boundary_is_still_refused(self) -> None:
    """The listing comes in pages of 100. Asking for exactly the cap would stop at page one and miss this."""
    self.set_limits(MAX_RUN_ARTIFACTS=100)
    client = self.client_with(101)

    with self.assertRaises(junit.JUnitError):
      self.read(client)

    self.assertEqual(client.downloads, [], "a listing cut at page one must not be read as the whole run")
    self.assertEqual(client.listed_counts, [101], "page two had to be fetched to know the run was over the cap")

  def test_artifacts_that_are_not_test_results_count_too(self) -> None:
    """A cut-short listing must never be read as the whole run, so everything the run lists counts."""
    client = self.client_with(3, others=4)

    with self.assertRaises(junit.JUnitError):
      self.read(client)

    self.assertEqual(client.downloads, [])


class DownloadErrorTest(LimitTestCase):
  """What `read_artifact_suites` makes of an error that comes out of the client."""

  def test_a_coding_mistake_is_not_hidden_as_lost_tests(self) -> None:
    """A TypeError or AttributeError is a bug in the code, so it must show, not read as a bad upload."""
    for error in (TypeError("get_bytes() got an unexpected keyword argument"), AttributeError("no such attribute")):
      with self.subTest(error=type(error).__name__):
        with self.assertRaises(type(error)):
          junit.read_artifact_suites(FakeClient(error=error), make_ref(100))

  def test_every_other_error_becomes_a_junit_error(self) -> None:
    for error in (RuntimeError("boom"), OSError("disk"), ValueError("value"), ConnectionError("down")):
      with self.subTest(error=type(error).__name__):
        with self.assertRaises(junit.JUnitError) as caught:
          junit.read_artifact_suites(FakeClient(error=error), make_ref(100))

        self.assertIn(ARTIFACT_NAME, str(caught.exception))
        self.assertIn("download failed", str(caught.exception))
        self.assertIn(type(error).__name__, str(caught.exception))
        self.assertIs(caught.exception.__cause__, error)


class RunXmlCapTest(LimitTestCase):
  """`read_run_tests` also caps the XML that all the zips of one run unpack to, together.

  The parsed results of every zip of a run stay in memory until the run is done, so the size of
  the zips says little: dense XML packs 20 to 1 and keeps about 5 bytes of Python objects per
  byte of XML. The cap is on the declared, unpacked size, and it is checked before the XML that
  would cross it is read.
  """

  XML_BYTES = 5000

  def setUp(self) -> None:
    """Sets limits so that each zip is fine on its own and only three of them together are too much."""
    super().setUp()
    self.body = build_zip([("results.xml", xml_of_size(self.XML_BYTES))])
    self.assertLess(len(self.body), 500, "each zip is tiny")
    self.set_limits(
        MAX_ARTIFACT_BYTES=1000,
        MAX_XML_BYTES=10_000,
        MAX_ZIP_XML_BYTES=10_000,
        MAX_RUN_ARTIFACT_BYTES=100_000,
        MAX_RUN_XML_BYTES=12_000,
    )

  def client_with(self, workers: int) -> FakeClient:
    """Builds a client whose run has this many artifacts, all alike.

    Args:
      workers: How many artifacts the run lists.

    Returns:
      The client.
    """
    payloads = [artifact_payload(worker, len(self.body)) for worker in range(1, workers + 1)]
    return FakeClient(self.body, payloads=payloads)

  def read(self, client: FakeClient) -> junit.RunTests:
    """Reads the one flavor of the run.

    Args:
      client: The fake client.

    Returns:
      The run's tests.
    """
    return junit.read_run_tests(client, RUN_ID, flavors=[FLAVOR])

  def test_zips_that_are_fine_alone_but_too_much_together_are_refused(self) -> None:
    client = self.client_with(3)

    with mock.patch.object(junit, "parse_junit_xml", wraps=junit.parse_junit_xml) as parser:
      with self.assertRaises(junit.JUnitError) as caught:
        self.read(client)

    self.assertEqual(parser.call_count, 2, "the XML that would cross the cap must not be read")
    message = str(caught.exception)
    self.assertIn(f"-3-{RUN_ID}", message, "the message names the artifact that went over")
    self.assertIn("5,000 bytes", message)
    self.assertIn("2,000 bytes", message, "what was left of the run's budget")
    self.assertIn("12,000 bytes", message, "the limit")

  def test_a_run_exactly_at_the_xml_cap_is_read(self) -> None:
    self.set_limits(MAX_RUN_XML_BYTES=3 * self.XML_BYTES)
    client = self.client_with(3)

    run = self.read(client)

    self.assertEqual(len(client.downloads), 3)
    self.assertIsNotNone(run.result_for(FLAVOR))

  def test_a_zip_over_its_own_xml_limit_is_still_refused_for_that_reason(self) -> None:
    self.set_limits(MAX_ZIP_XML_BYTES=4000)

    with self.assertRaises(junit.JUnitError) as caught:
      self.read(self.client_with(1))

    self.assertIn("all XML files together", str(caught.exception))
    self.assertNotIn("of one run", str(caught.exception), "the zip's own limit comes first")

  def test_each_run_gets_its_own_xml_budget(self) -> None:
    client = self.client_with(2)

    self.read(client)
    self.read(client)

    self.assertEqual(len(client.downloads), 4)

  def test_the_run_cap_does_not_apply_to_a_zip_read_on_its_own(self) -> None:
    """`parse_artifact_zip` takes one zip; only `read_run_tests` knows about a run."""
    self.set_limits(MAX_RUN_XML_BYTES=1000)

    found = self.parse(self.body)

    self.assertEqual(list(found), [FLAVOR])


class ZipContentTest(LimitTestCase):
  """`parse_artifact_zip` refuses a zip whose XML files would unpack too big."""

  def assert_refused_without_reading(self, data: bytes) -> str:
    """Parses a zip that must be refused, and fails if any file is opened first.

    The point of declared sizes is that a bomb is never unpacked. So opening a file is made to
    fail loudly: a refusal that came only after a file was read would not protect anything.

    Args:
      data: The zip bytes.

    Returns:
      The text of the `JUnitError` that was raised.
    """
    with mock.patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("opened too early")) as opener:
      with self.assertRaises(junit.JUnitError) as caught:
        self.parse(data)
    # The code turns any error from opening a file into a JUnitError, this one included. So the
    # call itself is what is checked.
    opener.assert_not_called()
    return str(caught.exception)

  def test_a_normal_zip_still_parses(self) -> None:
    """With the real limits and a real file, nothing changes."""
    raw = (FIXTURES / SMALL_SUITE_FILE).read_bytes()

    found = self.parse(build_zip([(SMALL_SUITE_FILE, raw)]))

    self.assertEqual(list(found), [FLAVOR])
    self.assertEqual(found[FLAVOR].collected, SMALL_SUITE_CASES)
    self.assertEqual(found[FLAVOR].files, (SMALL_SUITE_FILE,))

  def test_one_xml_file_over_the_limit_is_refused(self) -> None:
    self.set_limits(MAX_XML_BYTES=1000, MAX_ZIP_XML_BYTES=10_000)
    data = build_zip([("small.xml", xml_of_size(300)), ("big.xml", xml_of_size(1001))])

    message = self.assert_refused_without_reading(data)

    self.assertIn(ARTIFACT_NAME, message)
    self.assertIn("big.xml", message)
    self.assertIn("1,001 bytes", message)
    self.assertIn("1,000 bytes", message)
    self.assertNotIn("small.xml", message)

  def test_one_xml_file_at_the_limit_is_read(self) -> None:
    self.set_limits(MAX_XML_BYTES=1000, MAX_ZIP_XML_BYTES=10_000)

    found = self.parse(build_zip([("exact.xml", xml_of_size(1000))]))

    self.assertEqual(list(found), [FLAVOR])

  def test_files_that_are_small_alone_but_too_big_together_are_refused(self) -> None:
    self.set_limits(MAX_XML_BYTES=1000, MAX_ZIP_XML_BYTES=1500)
    data = build_zip([(f"worker-{n}.xml", xml_of_size(600)) for n in range(3)])

    message = self.assert_refused_without_reading(data)

    self.assertIn(ARTIFACT_NAME, message)
    self.assertIn("all XML files", message)
    self.assertIn("1,800 bytes", message)
    self.assertIn("1,500 bytes", message)

  def test_files_that_add_up_to_exactly_the_limit_are_read(self) -> None:
    self.set_limits(MAX_XML_BYTES=1000, MAX_ZIP_XML_BYTES=1500)

    found = self.parse(build_zip([(f"worker-{n}.xml", xml_of_size(500)) for n in range(3)]))

    self.assertEqual(list(found), [FLAVOR])

  def test_a_name_that_repeats_is_counted_every_time_it_is_read(self) -> None:
    """Reading by name returns the last file of that name each time, so each repeat costs its size.

    The first twenty files are tiny and the last one is big. Adding up the declared sizes of
    all twenty-one files gives a small total, yet 21 reads of the big one are far over the limit.
    """
    self.set_limits(MAX_XML_BYTES=1000, MAX_ZIP_XML_BYTES=5000)
    tiny = [("results.xml", xml_of_size(len(EMPTY_SUITE))) for _ in range(20)]
    data = build_zip(tiny + [("results.xml", xml_of_size(900))])

    message = self.assert_refused_without_reading(data)

    self.assertIn("all XML files", message)

  def test_a_big_file_that_is_not_xml_is_ignored(self) -> None:
    """Only members whose name ends in .xml are read, so only they count."""
    self.set_limits(MAX_XML_BYTES=1000, MAX_ZIP_XML_BYTES=1500)
    data = build_zip(
        [
            ("results.xml", xml_of_size(300)),
            ("pytest.log", b"x" * 100_000),
            ("coverage.json", b"{}" + b" " * 100_000),
        ]
    )

    found = self.parse(data)

    self.assertEqual(list(found), [FLAVOR])
    self.assertEqual(found[FLAVOR].files, ("results.xml",))

  def test_an_upper_case_xml_extension_still_counts(self) -> None:
    """The extension is matched without regard to case, so a renamed file cannot slip by."""
    self.set_limits(MAX_XML_BYTES=1000, MAX_ZIP_XML_BYTES=10_000)

    message = self.assert_refused_without_reading(build_zip([("RESULTS.XML", xml_of_size(1001))]))

    self.assertIn("RESULTS.XML", message)

  def test_a_real_zip_bomb_is_refused_with_the_real_limits(self) -> None:
    """A 21 MB XML file squeezes into a few kilobytes. With no patch, it is still refused."""
    data = build_zip([("results.xml", xml_of_size(21 * MEGABYTE))])
    self.assertLess(len(data), 100_000, "the bomb should be a small zip")

    message = self.assert_refused_without_reading(data)

    self.assertIn("results.xml", message)


class LyingSizeTest(LimitTestCase):
  """A zip says in its directory how big each file is, and the directory can be wrong."""

  def test_data_longer_than_declared_fails_the_crc_check(self) -> None:
    """The file holds 2,000 bytes and says 100. The CRC of the first 100 bytes does not match.

    This is how a file that holds more than it declares is found out. The tick gets the usual
    `JUnitError`, and the message says which file and why.
    """
    data = zip_that_lies_about_its_size(xml_of_size(2000), declared=100, compression=zipfile.ZIP_STORED)

    with self.assertRaises(junit.JUnitError) as caught:
      self.parse(data)

    self.assertIsInstance(caught.exception.__cause__, zipfile.BadZipFile)
    self.assertIn("lie.xml", str(caught.exception))
    self.assertIn("cannot read", str(caught.exception))
    self.assertIn("CRC", str(caught.exception))

  def test_data_longer_than_declared_is_cut_at_the_declared_size(self) -> None:
    """Even with a forged CRC, the parser is handed the declared 100 bytes and no more.

    The first 100 bytes are a whole document and the rest is junk. A careful attacker writes the
    CRC of the first 100 bytes into the directory, so there is no error to catch. What matters
    is that the extra 1,900 bytes never reach the parser.
    """
    content = xml_of_size(100) + b"?" * 1900
    data = zip_that_lies_about_its_size(content, declared=100, compression=zipfile.ZIP_DEFLATED, forge_crc=True)

    with mock.patch.object(junit, "parse_junit_xml", wraps=junit.parse_junit_xml) as parser:
      found = self.parse(data)

    self.assertEqual(list(found), [FLAVOR])
    self.assertEqual(len(parser.call_args.args[0]), 100)

  def test_a_file_declared_empty_that_holds_data_fails_the_crc_check(self) -> None:
    """A read of zero bytes would never reach the end of the file, so the CRC check would not run.

    The read asks for one byte more than the declared size. That makes `zipfile` reach the end
    of the file, and fail the CRC check, even when the declared size is zero.
    """
    data = zip_that_lies_about_its_size(xml_of_size(200), declared=0, compression=zipfile.ZIP_STORED)

    with self.assertRaises(junit.JUnitError) as caught:
      self.parse(data)

    self.assertIsInstance(caught.exception.__cause__, zipfile.BadZipFile)
    self.assertIn("CRC", str(caught.exception))

  def test_data_far_longer_than_declared_does_not_fill_the_memory(self) -> None:
    """The file says 1,000 bytes and holds 64 MB. Reading it must not unpack the 64 MB.

    A plain `ZipFile.read` asks the decompressor for up to 1 GB in one call and only then cuts
    the result down to the declared size. This is the same lie at a size that is cheap to test:
    without the cap the peak was more than 64 MB, and with it the peak stays small.
    """
    data = zip_that_lies_about_its_size(b"\0" * (64 * MEGABYTE), declared=1000, compression=zipfile.ZIP_DEFLATED)
    self.assertLess(len(data), 200_000, "the lie should be a small zip")
    if not tracemalloc.is_tracing():
      tracemalloc.start()
      self.addCleanup(tracemalloc.stop)
    tracemalloc.reset_peak()

    with self.assertRaises(junit.JUnitError):
      self.parse(data)

    _, peak = tracemalloc.get_traced_memory()
    self.assertLess(peak, 8 * MEGABYTE, f"reading used {peak:,} bytes at its peak")


class UnreadableFileTest(LimitTestCase):
  """A file that cannot be unpacked is a `JUnitError`, whatever `zipfile` raises about it.

  Each kind below used to escape as itself. `tick` catches only `JUnitError`, so one bad upload
  from one pull request crashed the whole tick.
  """

  def assert_unreadable(self, data: bytes, cause: type[BaseException] | tuple[type[BaseException], ...]) -> str:
    """Parses a zip that cannot be unpacked, and checks what the tick would see.

    Args:
      data: The zip bytes.
      cause: The error type, or types, that `zipfile` is expected to have raised. A
        `JUnitError` keeps it as `__cause__`, so a test can tell that it checked the kind it meant to.

    Returns:
      The text of the `JUnitError` that was raised.
    """
    with self.assertRaises(junit.JUnitError) as caught:
      self.parse(data)
    self.assertIsInstance(caught.exception.__cause__, cause)
    self.assertIn(ARTIFACT_NAME, str(caught.exception))
    return str(caught.exception)

  def test_corrupt_deflate_data(self) -> None:
    """The packed bytes are not valid deflate, and zlib raises its own `zlib.error`."""
    message = self.assert_unreadable(zip_with_corrupt_deflate(), zlib.error)

    self.assertIn(SMALL_SUITE_FILE, message)
    self.assertIn("zlib.error", message, "the class name alone would read as just 'error'")

  def test_a_compression_method_zipfile_does_not_know(self) -> None:
    """Method 99 is not one that zipfile can unpack, and it raises `NotImplementedError`."""
    message = self.assert_unreadable(zip_with_compression_method(99), NotImplementedError)

    self.assertIn(SMALL_SUITE_FILE, message)

  def test_an_encrypted_file(self) -> None:
    """With no password, zipfile raises a plain `RuntimeError` for an encrypted file."""
    message = self.assert_unreadable(zip_with_encrypted_flag(), RuntimeError)

    self.assertIn("encrypted", message)
    self.assertIn(SMALL_SUITE_FILE, message)

  def test_a_file_cut_short(self) -> None:
    """The directory claims 10 MB more data than the zip holds, and the data runs out.

    Python 3.11 before the zip-bomb fix raises `EOFError`, which has no text at all. Newer
    versions notice that the file overlaps the directory and raise `BadZipFile`.
    """
    message = self.assert_unreadable(zip_claiming_more_data_than_it_holds(), (EOFError, zipfile.BadZipFile))

    self.assertIn(SMALL_SUITE_FILE, message)

  @unittest.skipIf(lzma is None, "this Python was built without lzma")
  def test_corrupt_lzma_data(self) -> None:
    """LZMA has an error type of its own, `lzma.LZMAError`."""
    message = self.assert_unreadable(zip_with_corrupt_lzma(), lzma.LZMAError)

    self.assertIn(SMALL_SUITE_FILE, message)

  def test_a_directory_that_points_before_the_start_of_the_zip(self) -> None:
    """Every file seems to start 100 bytes before the zip does, so zipfile seeks to a negative place.

    That raises a bare `ValueError`. A later Python may check for it and raise `BadZipFile`.
    """
    message = self.assert_unreadable(zip_with_shifted_directory(), (ValueError, zipfile.BadZipFile))

    self.assertIn(SMALL_SUITE_FILE, message)

  def test_a_file_name_that_is_not_valid_utf8(self) -> None:
    """This one fails while the zip is opened, before any file is looked at."""
    message = self.assert_unreadable(zip_with_bad_name_encoding(), UnicodeDecodeError)

    self.assertIn("not a readable zip", message)

  def test_a_damaged_zip_never_raises_anything_but_junit_error(self) -> None:
    """Real zips with a few bytes changed at random: the tick gets a JUnitError or a result.

    The same seed gives the same damage every time. Every kind listed above was first found
    this way, along with `ValueError` and `EOFError`, so this is what looks for the next kind.
    """
    rng = random.Random(20261013)
    for label, original in zips_to_damage().items():
      for attempt in range(300):
        try:
          self.parse(damage(original, rng))
        except junit.JUnitError:
          pass  # This is the way a bad zip is supposed to fail.
        except Exception as error:
          error.add_note(f"raised for damaged copy {attempt} of the {label} zip")
          raise


# A JUnit file never has a DOCTYPE, so each of these is refused. This one has an entity.
ENTITY_XML = (
    b'<?xml version="1.0" encoding="utf-8"?>\n'
    b'<!DOCTYPE testsuites [ <!ENTITY name "pytest"> ]>\n'
    b'<testsuites><testsuite name="&name;" tests="0"></testsuite></testsuites>\n'
)

# Text that only looks like a DOCTYPE: one inside a CDATA section, and one in escaped text.
DOCTYPE_TEXT_XML = (
    b'<testsuites><testsuite name="s" tests="2">'
    b'<testcase classname="a" name="one" time="0.1">'
    b'<system-out><![CDATA[<!DOCTYPE html><!ENTITY x "y">]]></system-out></testcase>'
    b'<testcase classname="a" name="two" time="0.1"><system-out>&lt;!DOCTYPE html&gt;</system-out></testcase>'
    b"</testsuite></testsuites>"
)


def fifty_times_bomb(references: int) -> bytes:
  """Builds a small XML file that expat expands about 50 times, and does not stop.

  expat refuses an entity bomb only when the text it made is more than 100 times the input. This
  file stays at about 50 times, so the parser lets it through: 150 KB of file becomes 10 MB of
  text in the tree, and a 20 MB file (the per-file limit) would become a gigabyte.

  Args:
    references: How many times the 200-character entity is used.

  Returns:
    The XML file.
  """
  value = "a" * 200
  return (
      '<?xml version="1.0"?>\n'
      f'<!DOCTYPE bomb [ <!ENTITY e "{value}"> ]>\n'
      '<testsuites><testsuite name="s" tests="0"><system-out>'
      + "&e;" * references
      + "</system-out></testsuite></testsuites>\n"
  ).encode()


def classic_entity_bomb(levels: int = 9) -> bytes:
  """Builds the "billion laughs" file: each entity repeats the one below it ten times.

  Args:
    levels: How many entities are stacked. Nine levels make 10 billion characters.

  Returns:
    The XML file.
  """
  lines = ['<!ENTITY e0 "aaaaaaaaaa">']
  for level in range(1, levels + 1):
    lines.append(f'<!ENTITY e{level} "' + f"&e{level - 1};" * 10 + '">')
  declarations = "\n".join(lines)
  return (
      f'<?xml version="1.0"?>\n<!DOCTYPE bomb [\n{declarations}\n]>\n'
      f'<testsuites><testsuite name="&e{levels};" tests="0"/></testsuites>\n'
  ).encode()


class XmlDeclarationTest(LimitTestCase):
  """An XML file with a DOCTYPE is refused before any entity in it is expanded."""

  def assert_refused(self, data: bytes) -> str:
    """Parses a file that must be refused as a DOCTYPE.

    Args:
      data: The XML file.

    Returns:
      The text of the `JUnitError` that was raised.
    """
    with self.assertRaises(junit.JUnitError) as caught:
      junit.parse_junit_xml(data, file_name="results.xml")
    message = str(caught.exception)
    self.assertIn("results.xml", message)
    self.assertIn("DOCTYPE", message)
    return message

  def test_a_file_with_an_entity_is_refused(self) -> None:
    self.assert_refused(ENTITY_XML)

  def test_a_file_with_only_a_doctype_is_refused(self) -> None:
    self.assert_refused(b"<!DOCTYPE testsuites>\n<testsuites><testsuite name='s' tests='0'/></testsuites>")

  def test_a_file_that_points_at_an_outside_dtd_is_refused(self) -> None:
    self.assert_refused(b'<!DOCTYPE testsuites SYSTEM "http://example.invalid/x.dtd"><testsuites/>')

  def test_a_utf16_file_with_a_doctype_is_refused(self) -> None:
    """In UTF-16 the bytes `<!DOCTYPE` are spread out with zeros, so a search of the bytes would miss it."""
    text = ENTITY_XML.decode("utf-8").replace(' encoding="utf-8"', "")
    data = text.encode("utf-16")
    self.assertNotIn(b"<!DOCTYPE", data)

    self.assert_refused(data)

  def test_the_classic_entity_bomb_is_refused_as_a_doctype(self) -> None:
    """expat would also stop this one, but only after it had expanded 8 MB. Here the DOCTYPE is the reason."""
    self.assert_refused(classic_entity_bomb())

  def test_a_small_file_that_expands_fifty_times_is_refused_before_it_is_expanded(self) -> None:
    """expat lets this file through, so the size limit alone would not stop it."""
    data = fifty_times_bomb(50_000)
    self.assertLess(len(data), 200_000, "the file is far under the 20 MB limit")
    if not tracemalloc.is_tracing():
      tracemalloc.start()
      self.addCleanup(tracemalloc.stop)
    tracemalloc.reset_peak()

    self.assert_refused(data)

    _, peak = tracemalloc.get_traced_memory()
    self.assertLess(peak, 2 * MEGABYTE, f"parsing used {peak:,} bytes at its peak")

  def test_doctype_text_inside_cdata_or_escaped_is_not_a_doctype(self) -> None:
    """Test output can quote HTML. Only a real DOCTYPE is refused, not text that looks like one."""
    result = junit.parse_junit_xml(DOCTYPE_TEXT_XML, file_name="results.xml")

    self.assertEqual(result.collected, 2)

  def test_a_doctype_inside_a_zip_names_the_file(self) -> None:
    data = build_zip([("worker-1/results.xml", ENTITY_XML)])

    with self.assertRaises(junit.JUnitError) as caught:
      self.parse(data)

    self.assertIn("worker-1/results.xml", str(caught.exception))
    self.assertIn("DOCTYPE", str(caught.exception))


if __name__ == "__main__":
  unittest.main(verbosity=2)
