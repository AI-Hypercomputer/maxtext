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

"""Offline unit tests for `collector.store`.

Everything here happens inside a fresh temporary directory that is deleted again when the
test ends, and nothing here opens a socket: the base test case replaces `socket.socket`, so a
test that reached for GitHub would fail instead of hanging. `store.py` makes no requests of
its own, so no stub client is needed at all.

The rows are built from the same saved fixtures the rest of the suite uses, so the store is
exercised with real payloads rather than invented ones:

  * run 33468578834 - a scheduled run of 2026-09-01, one attempt, 54 jobs. Month "2026-09".
  * run 33406483779 - pull request 5070, merged on 2026-09-01. Created 2026-08-31, so its
    rows live in the closed month "2026-08" and its test detail is worth keeping.
  * run 32772626658 - a pull request run of 2026-08-24 with two attempts and three rescues,
    whose pull request never merged. Same closed month, and the run whose test detail
    compaction is meant to throw away.

Six promises are checked head on, because the whole collector rests on them.

  1. **Running the collector twice is harmless.** A repeated tick writes zero rows and leaves
     every byte of every file where it was, whether it is the same process or a new one, and
     whether or not `state.json` survived.
  2. **Append-only means append-only.** A correction is a second line with the same key and a
     later `collected_at`. Both lines stay on disk; `read` hands back the newer one. When two
     lines share a `collected_at`, the later line wins, and it wins the same way every time.
  3. **`state.json` is an index of attempts, nothing finer.** No test row, no job id and no
     test name ever appears in it, and deleting it costs nothing that cannot be rebuilt by
     scanning the rows.
  4. **A month is a file.** Two runs created in two months land in two files, so a closed
     month stops changing.
  5. **Compaction is idempotent and never touches what it is not aimed at.** It drops
     superseded lines and the per-test detail of runs that did not merge; the second run
     removes nothing and does not even rewrite the file.
  6. **A write is atomic.** An interrupted write leaves the previous file exactly as it was -
     never truncated, never half a line - and leaves no temporary file behind.

Two defects were found while this file was written - a half-written attempt that could never
be completed, and an attempt that could sit in both indexes at once. Both are fixed in
`store.py` now, and `HalfWrittenAttemptTest` is what keeps them fixed.

Run from the repository root, either way:

  python3 tools/ci_metrics/collector/tests/store_test.py
  python3 -m pytest tools/ci_metrics/collector/tests/store_test.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

# The collector package's parent, so the import stays inside tools/ci_metrics and running
# these tests never byte-compiles anything outside it.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import junit
from collector import rows
from collector import store

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The scheduled run of 2026-09-01: one attempt, 54 jobs, created in the month the clock is
# currently in when these fixtures were captured.
RUN_ID = 33468578834
RUN_CREATED = "2026-09-01T04:06:01Z"
RUN_MONTH = "2026-09"

# Pull request 5070, merged. Created on 2026-08-31, so it files under the closed month.
MERGED_RUN_ID = 33406483779
MERGED_CREATED = "2026-08-31T15:06:03Z"

# A pull request run with two attempts and three rescues whose pull request never merged.
UNMERGED_RUN_ID = 32772626658
UNMERGED_CREATED = "2026-08-24T20:13:09Z"
UNMERGED_ATTEMPT = 2

# Both of the above were created in August, which will never be the open month again.
CLOSED_MONTH = "2026-08"

# The tpu-unit worker 1 job of run 33468578834.
TPU_UNIT_WORKER_1 = "TPU Pretrain Tests (tpu-unit) / Execute Tests (1) / tpu-unit"

# The small real JUnit file used wherever a test only needs "some test rows": nine cases, of
# which seven were skipped.
SMALL_SUITE_FILE = "tpu-post-training-integration-1.xml"
SMALL_SUITE_ID = "tpu-post-training-integration"
SMALL_SUITE_CASES = 9

# Two collection times four hours apart, so which line a reader picks is never a matter of
# how fast the test ran.
FIRST_TICK = "2026-09-01T04:20:00Z"
SECOND_TICK = "2026-09-01T08:20:00Z"


def read_fixture(name: str) -> bytes:
  """Returns the raw bytes of one saved fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The file contents.
  """
  return (FIXTURES / name).read_bytes()


def load_json(name: str) -> Any:
  """Loads one saved JSON fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The parsed payload, fresh each call so no test can mutate another's copy.
  """
  return json.loads(read_fixture(name))


def load_jobs(name: str) -> list[dict[str, Any]]:
  """Loads the job list out of a saved jobs-endpoint fixture.

  Args:
    name: File name inside `tests/fixtures/`.

  Returns:
    The jobs, in the order GitHub listed them.
  """
  payload = load_json(name)
  return payload["jobs"] if isinstance(payload, dict) else payload


def named(jobs: list[dict[str, Any]], name: str) -> dict[str, Any]:
  """Returns the one job of a list that carries a given name.

  Args:
    jobs: The jobs.
    name: The full job name.

  Returns:
    The job.

  Raises:
    AssertionError: The name is not in the list exactly once.
  """
  found = [job for job in jobs if job.get("name") == name]
  if len(found) != 1:
    raise AssertionError(f"{name!r} appears {len(found)} times in the saved jobs, not once")
  return found[0]


def attempts_of(run_id: int, prefix: str, last_attempt: int) -> dict[int, list[dict[str, Any]]]:
  """Loads every saved attempt of one multi-attempt run.

  Args:
    run_id: The run id, which is part of every fixture name.
    prefix: The fixture name prefix, e.g. "rerun".
    last_attempt: The highest attempt saved.

  Returns:
    Attempt number -> that attempt's jobs, the shape `rescue_rows` takes.
  """
  return {n: load_jobs(f"{prefix}-{run_id}-attempt{n}-jobs.json") for n in range(1, last_attempt + 1)}


def suite_entry(
    file_name: str = SMALL_SUITE_FILE,
    suite_id: str = SMALL_SUITE_ID,
    worker: int = 1,
    missing: dict[int, str] | None = None,
) -> junit.SuiteEntry:
  """Parses one saved JUnit file into the entry shape `rows.suite_row` takes.

  Args:
    file_name: The saved JUnit file.
    suite_id: The suite the file belongs to.
    worker: Which worker published it.
    missing: Worker number -> reason, for the workers that published nothing.

  Returns:
    The entry, with the parsed result in it.
  """
  result = junit.parse_junit_xml(read_fixture(file_name), file_name=file_name)
  return junit.SuiteEntry(
      suite_id=suite_id,
      result=result,
      per_worker={worker: result},
      missing_workers=dict(missing or {}),
  )


@contextlib.contextmanager
def captured_stderr() -> Iterator[io.StringIO]:
  """Collects what the store prints to stderr, so a warning can be asserted on.

  Yields:
    The buffer the warnings land in.
  """
  buffer = io.StringIO()
  with contextlib.redirect_stderr(buffer):
    yield buffer


class StoreTestCase(unittest.TestCase):
  """Base class: no network, and one throwaway output directory per test."""

  def setUp(self) -> None:
    """Blocks the network and opens an empty store in a temporary directory."""
    super().setUp()

    def deny(*args: object, **kwargs: object) -> None:
      """Refuses every socket the test might try to open.

      Args:
        *args: Ignored.
        **kwargs: Ignored.

      Raises:
        AssertionError: Always.
      """
      del args, kwargs
      raise AssertionError("these tests must run offline, but something tried to open a socket")

    for attribute in ("socket", "create_connection"):
      patch = mock.patch.object(socket, attribute, deny)
      patch.start()
      self.addCleanup(patch.stop)

    self.tmp_dir = tempfile.mkdtemp(prefix="ci-metrics-store-")
    self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
    self.store = store.Store(self.tmp_dir)

  def reopen(self) -> store.Store:
    """Opens the same directory again, the way the next tick's process would.

    Returns:
      A second `Store` with no caches, pointing at the same files.
    """
    return store.Store(self.tmp_dir)

  def ndjson_snapshot(self) -> dict[str, bytes]:
    """Returns every stored NDJSON file, byte for byte.

    Returns:
      File name -> contents. Empty when nothing has been written.
    """
    data_dir = self.store.data_dir
    if not data_dir.is_dir():
      return {}
    return {path.name: path.read_bytes() for path in sorted(data_dir.glob(f"*{store.NDJSON_SUFFIX}"))}

  def temp_leftovers(self) -> list[str]:
    """Returns the names of any half-written temporary files left in the store.

    Returns:
      The file names, sorted. Empty is what every finished operation should leave.
    """
    found: list[str] = []
    for directory in (self.store.data_dir, self.store.views_dir, self.store.pr_views_dir):
      if directory.is_dir():
        found.extend(path.name for path in directory.iterdir() if path.name.startswith(store.TEMP_PREFIX))
    return sorted(found)

  def lines_of(self, kind: str, month: str) -> list[dict[str, Any]]:
    """Reads one month's file as raw lines, corrections and all.

    Args:
      kind: One of the `rows.KIND_*` constants.
      month: "YYYY-MM".

    Returns:
      Every line, decoded, in the order it was written.
    """
    path = self.store.path_for(kind, month)
    if not path.exists():
      return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

  def write_a_tick(self, target: store.Store, collected_at: str) -> dict[str, int]:
    """Writes one whole tick's worth of rows and saves the index, exactly as a tick would.

    Two runs go in: the scheduled September run, with its jobs, one suite and that suite's
    tests, and the August pull request run with its attempt-2 jobs and its three rescues. So
    the store ends up holding every kind, across two months.

    Args:
      target: The store to write into.
      collected_at: The tick's collection timestamp, stamped on every row.

    Returns:
      Kind -> how many rows that call actually wrote.
    """
    run = load_json("run.json")
    rerun = load_json(f"rerun-{UNMERGED_RUN_ID}-run.json")
    attempts = attempts_of(UNMERGED_RUN_ID, "rerun", UNMERGED_ATTEMPT)
    entry = suite_entry()
    written = {
        rows.KIND_RUN: target.append(
            rows.KIND_RUN,
            [rows.run_row(run, collected_at=collected_at), rows.run_row(rerun, collected_at=collected_at)],
        ),
        # One call per run: a job row files under its RUN's month, and these two runs were
        # created in different months.
        rows.KIND_JOB: target.append(
            rows.KIND_JOB,
            [rows.job_row(run, job, collected_at=collected_at) for job in load_jobs("jobs.json")],
            month=store.month_for_run(run),
        )
        + target.append(
            rows.KIND_JOB,
            [rows.job_row(rerun, job, collected_at=collected_at) for job in attempts[UNMERGED_ATTEMPT]],
            month=store.month_for_run(rerun),
        ),
        rows.KIND_SUITE: target.append(
            rows.KIND_SUITE,
            [rows.suite_row(run, entry, collected_at=collected_at)],
            month=store.month_for_run(run),
        ),
        rows.KIND_TEST: target.append(
            rows.KIND_TEST,
            rows.test_rows(run, SMALL_SUITE_ID, 1, entry.result, collected_at=collected_at),
            month=store.month_for_run(run),
        ),
        rows.KIND_RESCUE: target.append(rows.KIND_RESCUE, rows.rescue_rows(rerun, attempts, collected_at=collected_at)),
    }
    state = target.load_state()
    state.mark_collected(RUN_ID, 1, RUN_CREATED)
    state.mark_collected(UNMERGED_RUN_ID, UNMERGED_ATTEMPT, UNMERGED_CREATED)
    target.save_state(state)
    return written


class OutputDirectoryTest(StoreTestCase):
  """The store never guesses where to write. There is no default and no fallback."""

  def test_a_store_cannot_be_opened_without_a_directory(self) -> None:
    """No argument is a `TypeError`; an empty one is a `StoreError` that says why."""
    with self.assertRaises(TypeError):
      store.Store()  # pylint: disable=no-value-for-parameter

    for empty in (None, "", "   ", "\t\n"):
      with self.subTest(empty=empty):
        with self.assertRaises(store.StoreError) as caught:
          store.Store(empty)
        self.assertIn("no default", str(caught.exception))

  def test_the_filesystem_root_and_the_home_directory_are_refused(self) -> None:
    """Two ways a missing `--out` used to turn into a store written over somebody's files."""
    with self.assertRaises(store.StoreError) as at_root:
      store.Store("/")
    self.assertIn("root", str(at_root.exception))

    with self.assertRaises(store.StoreError) as at_home:
      store.Store(Path.home())
    self.assertIn("home directory", str(at_home.exception))

  def test_a_git_checkout_is_refused_unless_it_is_already_a_store(self) -> None:
    """Pointing `--out` at a checkout has to be impossible; pointing it at a store is fine."""
    checkout = Path(tempfile.mkdtemp(prefix="ci-metrics-checkout-"))
    self.addCleanup(shutil.rmtree, checkout, ignore_errors=True)
    (checkout / ".git").mkdir()

    with self.assertRaises(store.StoreError) as caught:
      store.Store(checkout)
    self.assertIn("git checkout", str(caught.exception))

    (checkout / store.DATA_DIRNAME).mkdir()
    self.assertEqual(store.Store(checkout).out_dir, checkout.resolve())

  def test_opening_a_store_creates_nothing(self) -> None:
    """A reader that finds an empty store must not leave directories behind."""
    self.assertFalse(self.store.data_dir.exists())
    self.assertFalse(self.store.views_dir.exists())
    self.assertEqual(self.store.months(), [])
    self.assertEqual(self.store.read(rows.KIND_RUN), [])
    self.assertEqual(self.store.pending_run_ids(), [])
    self.assertFalse(self.store.data_dir.exists())

  def test_every_path_hangs_off_the_directory_it_was_given(self) -> None:
    """The dashboard's loader hard-codes these names, so they are pinned here."""
    root = self.store.out_dir
    self.assertEqual(self.store.data_dir, root / "data")
    self.assertEqual(self.store.views_dir, root / "views")
    self.assertEqual(self.store.pr_views_dir, root / "views" / "pr")
    self.assertEqual(self.store.state_path, root / "data" / "state.json")
    self.assertEqual(self.store.path_for(rows.KIND_TEST, "2026-09"), root / "data" / "test-2026-09.ndjson")

  def test_a_bad_kind_or_month_is_refused_before_anything_is_written(self) -> None:
    """`path_for` is the choke point, so both checks live there."""
    with self.assertRaises(store.StoreError):
      self.store.path_for("commit", "2026-09")
    for bad in ("2026-13", "26-09", "2026/09", "", "2026-9"):
      with self.subTest(month=bad):
        with self.assertRaises(store.StoreError):
          self.store.path_for(rows.KIND_RUN, bad)


class KeyMirrorTest(StoreTestCase):
  """`store.row_key` reads a stored line's fields; `rows.key()` builds from the object.

  They have to agree exactly, or a row would be stored under one key and looked up under
  another. `store.py` says so in its own comment; this is the check it points at.
  """

  def one_row_of_each_kind(self) -> dict[str, Any]:
    """Builds one row of every stored kind from the saved payloads.

    Returns:
      Kind -> a row of that kind.
    """
    run = load_json("run.json")
    rerun = load_json(f"rerun-{UNMERGED_RUN_ID}-run.json")
    entry = suite_entry()
    return {
        rows.KIND_RUN: rows.run_row(run),
        rows.KIND_JOB: rows.job_row(run, named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1)),
        rows.KIND_SUITE: rows.suite_row(run, entry),
        rows.KIND_TEST: rows.test_rows(run, SMALL_SUITE_ID, 1, entry.result)[0],
        rows.KIND_RESCUE: rows.rescue_rows(rerun, attempts_of(UNMERGED_RUN_ID, "rerun", UNMERGED_ATTEMPT))[0],
    }

  def test_the_store_keys_a_line_exactly_as_the_row_keys_itself(self) -> None:
    """Every kind, from a real payload, through a real `json.dumps` trip."""
    for kind, row in self.one_row_of_each_kind().items():
      with self.subTest(kind=kind):
        payload = json.loads(json.dumps(rows.to_json(row)))
        self.assertEqual(store.row_key(payload), row.key())

  def test_every_stored_kind_has_key_fields(self) -> None:
    """A kind the store files but cannot key would fail at write time, not at import."""
    self.assertEqual(sorted(store.KEY_FIELDS), sorted(store.ROW_KINDS))
    self.assertEqual(len(store.ROW_KINDS), 5)
    self.assertNotIn(rows.KIND_RESCUE, store.ATTEMPT_KINDS)

  def test_a_line_that_cannot_be_keyed_is_an_error_naming_the_field(self) -> None:
    """A row with no kind, an unknown kind, or no run id is refused, not guessed at."""
    with self.assertRaises(store.StoreError) as unknown:
      store.row_key({"kind": "commit", "run_id": 1})
    self.assertIn("commit", str(unknown.exception))

    with self.assertRaises(store.StoreError) as headless:
      store.row_key({"kind": rows.KIND_RUN, "attempt": 1})
    self.assertIn("run_id", str(headless.exception))


class RunningTwiceIsHarmlessTest(StoreTestCase):
  """The headline promise: a repeated tick writes nothing and changes no byte.

  The collector runs on a cron every four hours and is re-run by hand whenever someone wants
  fresher numbers. If a repeat could duplicate rows, every chart would double-count.
  """

  def test_a_repeated_tick_writes_zero_rows_and_changes_no_byte(self) -> None:
    """Same process, same rows, four hours later: nothing is written, nothing moves."""
    first = self.write_a_tick(self.store, FIRST_TICK)
    self.assertEqual(first[rows.KIND_RUN], 2)
    self.assertEqual(first[rows.KIND_JOB], 96, "54 jobs of the scheduled run and 42 of the re-run's second attempt")
    self.assertEqual(first[rows.KIND_SUITE], 1)
    self.assertEqual(first[rows.KIND_TEST], SMALL_SUITE_CASES)
    self.assertEqual(first[rows.KIND_RESCUE], 3)
    before = self.ndjson_snapshot()
    self.assertEqual(
        sorted(before),
        [
            "job-2026-08.ndjson",
            "job-2026-09.ndjson",
            "rescue-2026-08.ndjson",
            "run-2026-08.ndjson",
            "run-2026-09.ndjson",
            "suite-2026-09.ndjson",
            "test-2026-09.ndjson",
        ],
    )

    second = self.write_a_tick(self.store, SECOND_TICK)

    self.assertEqual(second, dict.fromkeys(store.ROW_KINDS, 0))
    self.assertEqual(self.ndjson_snapshot(), before)

  def test_the_next_process_repeating_the_tick_changes_no_byte(self) -> None:
    """A new `Store` has no caches, so this is the dedup working off the files and the index."""
    self.write_a_tick(self.store, FIRST_TICK)
    before = self.ndjson_snapshot()

    written = self.write_a_tick(self.reopen(), SECOND_TICK)

    self.assertEqual(written, dict.fromkeys(store.ROW_KINDS, 0))
    self.assertEqual(self.ndjson_snapshot(), before)

  def test_a_repeat_after_losing_the_index_still_changes_no_byte(self) -> None:
    """`state.json` is an index, not the truth: the rows themselves stop the second write."""
    self.write_a_tick(self.store, FIRST_TICK)
    before = self.ndjson_snapshot()
    self.store.state_path.unlink()

    written = self.write_a_tick(self.reopen(), SECOND_TICK)

    self.assertEqual(written, dict.fromkeys(store.ROW_KINDS, 0))
    self.assertEqual(self.ndjson_snapshot(), before)

  def test_ten_repeats_leave_one_row_per_key(self) -> None:
    """The count that matters downstream: one row per key however often the tick ran."""
    for _ in range(10):
      self.write_a_tick(self.reopen(), FIRST_TICK)

    self.assertEqual(len(self.store.read(rows.KIND_RUN)), 2)
    self.assertEqual(len(self.store.read(rows.KIND_JOB)), 96)
    self.assertEqual(len(self.store.read(rows.KIND_TEST)), SMALL_SUITE_CASES)
    self.assertEqual(len(self.store.read(rows.KIND_RESCUE)), 3)
    self.assertEqual(sum(len(self.lines_of(rows.KIND_TEST, month)) for month in self.store.months()), SMALL_SUITE_CASES)

  def test_one_batch_holding_the_same_row_twice_writes_it_once(self) -> None:
    """A caller that listed a run twice must not get two lines out of one call."""
    row = rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)

    written = self.store.append(rows.KIND_RUN, [row, row, row])

    self.assertEqual(written, 1)
    self.assertEqual(len(self.lines_of(rows.KIND_RUN, RUN_MONTH)), 1)

  def test_appending_nothing_writes_nothing_and_creates_no_file(self) -> None:
    """An empty tick - a docs-only day with no runs - must not leave an empty file behind."""
    self.assertEqual(self.store.append(rows.KIND_RUN, []), 0)
    self.assertFalse(self.store.data_dir.exists())


class CorrectionTest(StoreTestCase):
  """Append-only with corrections: a fix is a new line, never an edit of the old one.

  The artifact that was unreadable at 04:00 and readable at 08:00 is the case this exists
  for. Both lines stay on disk so the history of what the collector believed is auditable;
  `read` applies the rule so nothing downstream has to remember it.
  """

  def append_run(self, collected_at: str, title: str, correction: bool = False) -> int:
    """Appends the September run row with a given collection time and display title.

    Args:
      collected_at: The row's `collected_at`.
      title: A value that differs between the two lines, so the winner is identifiable.
      correction: Pass the correction flag through to `append`.

    Returns:
      How many rows were written.
    """
    payload = rows.to_json(rows.run_row(load_json("run.json"), collected_at=collected_at))
    payload["display_title"] = title
    return self.store.append(rows.KIND_RUN, [payload], correction=correction)

  def test_a_correction_wins_and_the_line_it_corrects_stays_on_disk(self) -> None:
    """The whole rule in one test: two lines stored, one row read back, the newer one."""
    self.assertEqual(self.append_run(FIRST_TICK, "first read"), 1)
    self.assertEqual(self.append_run(SECOND_TICK, "second read", correction=True), 1)

    on_disk = self.lines_of(rows.KIND_RUN, RUN_MONTH)
    self.assertEqual([line["display_title"] for line in on_disk], ["first read", "second read"])
    self.assertEqual([line["collected_at"] for line in on_disk], [FIRST_TICK, SECOND_TICK])

    read_back = self.store.read(rows.KIND_RUN)
    self.assertEqual(len(read_back), 1)
    self.assertEqual(read_back[0]["display_title"], "second read")
    self.assertEqual(read_back[0]["collected_at"], SECOND_TICK)

  def test_without_the_flag_a_second_read_of_the_same_row_is_skipped(self) -> None:
    """The dedup is what makes a repeated tick free; a correction has to ask for itself."""
    self.assertEqual(self.append_run(FIRST_TICK, "first read"), 1)

    self.assertEqual(self.append_run(SECOND_TICK, "second read"), 0)

    self.assertEqual(len(self.lines_of(rows.KIND_RUN, RUN_MONTH)), 1)
    self.assertEqual(self.store.read(rows.KIND_RUN)[0]["display_title"], "first read")

  def test_an_older_correction_never_beats_a_newer_line(self) -> None:
    """A tick that re-read an old artifact must not push a stale value back to the top."""
    self.append_run(SECOND_TICK, "newer read")
    self.append_run(FIRST_TICK, "older read", correction=True)

    self.assertEqual(len(self.lines_of(rows.KIND_RUN, RUN_MONTH)), 2)
    self.assertEqual(self.store.read(rows.KIND_RUN)[0]["display_title"], "newer read")

  def test_a_tie_on_collected_at_is_broken_by_the_later_line(self) -> None:
    """Two writes inside the same second: the one written second wins, every time."""
    self.append_run(FIRST_TICK, "written first")
    self.append_run(FIRST_TICK, "written second", correction=True)
    self.append_run(FIRST_TICK, "written third", correction=True)

    self.assertEqual(len(self.lines_of(rows.KIND_RUN, RUN_MONTH)), 3)
    for _ in range(3):
      self.store.refresh()
      self.assertEqual(self.store.read(rows.KIND_RUN)[0]["display_title"], "written third")

  def test_a_tie_across_months_is_broken_by_the_later_file(self) -> None:
    """Months are read oldest first, so a line in the later file is the later line."""
    payload = rows.to_json(rows.run_row(load_json("run.json"), collected_at=FIRST_TICK))
    august = dict(payload, display_title="filed under august")
    september = dict(payload, display_title="filed under september")
    self.store.append(rows.KIND_RUN, [august], month=CLOSED_MONTH, correction=True)
    self.store.append(rows.KIND_RUN, [september], month=RUN_MONTH, correction=True)

    read_back = self.store.read(rows.KIND_RUN)

    self.assertEqual(len(read_back), 1)
    self.assertEqual(read_back[0]["display_title"], "filed under september")

  def test_read_is_deterministic_across_processes(self) -> None:
    """Same files, same answer - the store never depends on dict order or on a cache."""
    self.append_run(FIRST_TICK, "first read")
    self.append_run(SECOND_TICK, "second read", correction=True)

    self.assertEqual(self.reopen().read(rows.KIND_RUN), self.reopen().read(rows.KIND_RUN))
    self.assertEqual(self.store.read(rows.KIND_RUN), self.reopen().read(rows.KIND_RUN))

  def test_read_month_hands_back_every_line_including_the_ones_that_lost(self) -> None:
    """`read` is the deduplicated view; `read_month` is the raw one, for an auditor."""
    self.append_run(FIRST_TICK, "first read")
    self.append_run(SECOND_TICK, "second read", correction=True)

    raw = list(self.store.read_month(RUN_MONTH, kinds=[rows.KIND_RUN]))

    self.assertEqual([row.collected_at for row in raw], [FIRST_TICK, SECOND_TICK])
    self.assertEqual(len(self.store.read(rows.KIND_RUN)), 1)

  def test_a_correction_of_a_test_row_keeps_its_key(self) -> None:
    """Corrections are not a run-row trick; a re-parsed test case corrects itself too."""
    run = load_json("run.json")
    entry = suite_entry()
    original = rows.test_rows(run, SMALL_SUITE_ID, 1, entry.result, collected_at=FIRST_TICK)[0]
    fixed = dataclasses.replace(original, duration=original.duration + 1.5, collected_at=SECOND_TICK)
    self.store.append(rows.KIND_TEST, [original], month=RUN_MONTH)

    self.store.append(rows.KIND_TEST, [fixed], month=RUN_MONTH, correction=True)

    self.assertEqual(original.key(), fixed.key())
    self.assertEqual(len(self.lines_of(rows.KIND_TEST, RUN_MONTH)), 2)
    read_back = self.store.read(rows.KIND_TEST)
    self.assertEqual(len(read_back), 1)
    self.assertEqual(read_back[0]["duration"], fixed.duration)


class MonthRoutingTest(StoreTestCase):
  """One file per month, chosen by the run's `created_at` in UTC, so a closed month stops."""

  def test_two_runs_from_two_months_land_in_two_files(self) -> None:
    """The September run and the August pull request run never share a file."""
    september = rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)
    august = rows.run_row(load_json(f"rerun-{UNMERGED_RUN_ID}-run.json"), collected_at=FIRST_TICK)

    written = self.store.append(rows.KIND_RUN, [september, august])

    self.assertEqual(written, 2)
    self.assertEqual(self.store.months(rows.KIND_RUN), [CLOSED_MONTH, RUN_MONTH])
    self.assertEqual([line["run_id"] for line in self.lines_of(rows.KIND_RUN, RUN_MONTH)], [RUN_ID])
    self.assertEqual([line["run_id"] for line in self.lines_of(rows.KIND_RUN, CLOSED_MONTH)], [UNMERGED_RUN_ID])

  def test_reading_across_months_returns_both_and_the_older_file_first(self) -> None:
    """`read` with no month argument is every month, oldest file first."""
    self.store.append(
        rows.KIND_RUN,
        [
            rows.run_row(load_json("run.json"), collected_at=FIRST_TICK),
            rows.run_row(load_json(f"rerun-{UNMERGED_RUN_ID}-run.json"), collected_at=FIRST_TICK),
        ],
    )

    self.assertEqual([line["run_id"] for line in self.store.read(rows.KIND_RUN)], [UNMERGED_RUN_ID, RUN_ID])
    self.assertEqual([line["run_id"] for line in self.store.read(rows.KIND_RUN, [RUN_MONTH])], [RUN_ID])
    self.assertEqual(self.store.read(rows.KIND_RUN, ["2026-07"]), [])

  def test_a_runs_rows_all_file_under_the_run_not_under_themselves(self) -> None:
    """A re-run started days later must not scatter one run across two months."""
    run = load_json("run.json")
    month = store.month_for_run(run)
    entry = suite_entry()

    self.store.append(rows.KIND_SUITE, [rows.suite_row(run, entry, collected_at=FIRST_TICK)], month=month)
    self.store.append(
        rows.KIND_TEST, rows.test_rows(run, SMALL_SUITE_ID, 1, entry.result, collected_at=FIRST_TICK), month=month
    )

    self.assertEqual(month, RUN_MONTH)
    self.assertEqual(self.store.months(rows.KIND_SUITE), [RUN_MONTH])
    self.assertEqual(self.store.months(rows.KIND_TEST), [RUN_MONTH])
    self.assertEqual(self.store.months(), [RUN_MONTH])

  def test_a_job_suite_or_test_row_without_a_month_is_refused_with_the_fix_in_the_message(self) -> None:
    """Guessing a month for these would silently split one run across two files.

    A suite and a test row carry no timestamp at all. A job row does carry one, but it is the
    JOB's creation: a re-run started in the next month would file its jobs there while its run
    row stayed behind. All three take the run's month from the caller.
    """
    run = load_json("run.json")
    entry = suite_entry()

    for kind, records in (
        (rows.KIND_JOB, [rows.job_row(run, named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1))]),
        (rows.KIND_SUITE, [rows.suite_row(run, entry)]),
        (rows.KIND_TEST, rows.test_rows(run, SMALL_SUITE_ID, 1, entry.result)),
    ):
      with self.subTest(kind=kind):
        with self.assertRaises(store.StoreError) as caught:
          self.store.append(kind, records)
        self.assertIn("month_for_run", str(caught.exception))

  def test_a_re_run_in_the_next_month_still_files_under_its_run(self) -> None:
    """The case the rule exists for: attempt 2 started after midnight on the 1st."""
    run = load_json("run.json")
    month = store.month_for_run(run)
    late = [dict(job, created_at="2026-10-01T09:00:00Z", run_attempt=2) for job in load_jobs("jobs.json")[:3]]

    self.store.append(rows.KIND_JOB, [rows.job_row(run, job, collected_at=FIRST_TICK) for job in late], month=month)

    self.assertEqual(self.store.months(rows.KIND_JOB), [RUN_MONTH])
    self.assertEqual(len(self.store.read(rows.KIND_JOB, [RUN_MONTH])), 3)

  def test_the_month_is_taken_in_utc(self) -> None:
    """A run created at 01:00 on the first, in +03:00, still belongs to the month before."""
    self.assertEqual(store.month_key("2026-09-01T01:00:00+03:00"), CLOSED_MONTH)
    self.assertEqual(store.month_key("2026-08-31T23:00:00Z"), CLOSED_MONTH)
    self.assertEqual(store.month_key(RUN_CREATED), RUN_MONTH)
    self.assertEqual(store.month_for_run(load_json("run.json")), RUN_MONTH)
    self.assertEqual(store.month_for_run({"created_at": MERGED_CREATED}), CLOSED_MONTH)

  def test_a_run_with_no_created_at_names_no_month(self) -> None:
    """Better a named error than a row filed under whatever month the collector ran in."""
    with self.assertRaises(store.StoreError) as caught:
      store.month_for_run({"id": 1})
    self.assertIn("created_at", str(caught.exception))

    with self.assertRaises(store.StoreError):
      store.month_key("not a timestamp")

  def test_one_kind_per_call(self) -> None:
    """A job row appended as a run row would be unreadable later; it is refused now."""
    run = load_json("run.json")
    job = rows.job_row(run, named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1))

    with self.assertRaises(store.StoreError) as caught:
      self.store.append(rows.KIND_RUN, [job])
    self.assertIn("one call writes one kind", str(caught.exception))

  def test_a_file_holding_the_wrong_kind_is_an_error_not_a_silent_skip(self) -> None:
    """If a file is ever hand-edited, reading it has to stop rather than lose rows."""
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    path = self.store.path_for(rows.KIND_RUN, RUN_MONTH)
    with path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps({"kind": rows.KIND_JOB, "run_id": RUN_ID, "attempt": 1, "job_id": 1}) + "\n")

    self.store.refresh()
    with self.assertRaises(store.StoreError) as caught:
      self.store.read(rows.KIND_RUN)
    self.assertIn("not 'run'", str(caught.exception))


class StateIndexTest(StoreTestCase):
  """`state.json` records run attempts and nothing finer, and can always be thrown away."""

  def test_the_index_holds_attempts_and_no_row_of_any_kind(self) -> None:
    """A test row in the index would reach a hundred megabytes and be rewritten six times a day."""
    self.write_a_tick(self.store, FIRST_TICK)
    text = self.store.state_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    self.assertEqual(
        sorted(payload),
        ["collected", "incomplete", "pending", "rebuilt", "updated_at", "v", "watermark_created_at", "watermark_run_id"],
    )
    self.assertEqual(payload["collected"], {str(UNMERGED_RUN_ID): [UNMERGED_ATTEMPT], str(RUN_ID): [1]})
    for kind in store.ROW_KINDS:
      with self.subTest(kind=kind):
        self.assertNotIn(f"{kind}{rows.KEY_SEPARATOR}", text)
    self.assertNotIn(SMALL_SUITE_ID, text)
    self.assertNotIn("tests.", text)
    self.assertNotIn(TPU_UNIT_WORKER_1, text)
    self.assertNotIn("job_id", text)

  def test_the_index_is_small_and_stays_small(self) -> None:
    """96 job rows and nine test rows go in; the index still lists two attempts."""
    self.write_a_tick(self.store, FIRST_TICK)

    state = self.reopen().load_state()

    self.assertEqual(state.attempt_count, 2)
    self.assertEqual(state.pending_count, 0)
    self.assertLess(self.store.state_path.stat().st_size, 400)

  def test_a_deleted_index_is_rebuilt_from_the_rows(self) -> None:
    """What the index says about stored attempts survives losing the file."""
    self.write_a_tick(self.store, FIRST_TICK)
    before = self.store.load_state().to_json()
    self.store.state_path.unlink()

    after = self.reopen().load_state().to_json()

    self.assertEqual(after["collected"], before["collected"])
    self.assertEqual(after["incomplete"], before["incomplete"])
    self.assertEqual(after["watermark_run_id"], before["watermark_run_id"])

  def test_a_rebuild_says_so_and_rewinds_its_watermark_by_a_day(self) -> None:
    """The two things a scan cannot know are stated, not guessed: re-ask, do not assume."""
    self.write_a_tick(self.store, FIRST_TICK)
    self.assertFalse(self.store.load_state().rebuilt)
    self.store.state_path.unlink()

    rebuilt = self.reopen().load_state()

    self.assertTrue(rebuilt.rebuilt)
    self.assertEqual(rebuilt.watermark_created_at, "2026-08-31T04:06:01Z")
    self.assertEqual(store.REBUILD_REWIND_HOURS, 24)

  def test_a_rebuild_cannot_recover_the_in_flight_list(self) -> None:
    """Nothing was written for a running attempt, so a scan has nothing to find."""
    self.write_a_tick(self.store, FIRST_TICK)
    state = self.store.load_state()
    state.mark_pending(33500000000, 1, "2026-09-01T09:00:00Z", status="in_progress")
    self.store.save_state(state)
    self.assertEqual(self.reopen().pending_run_ids(), [33500000000])
    self.store.state_path.unlink()

    rebuilt = self.reopen()

    self.assertEqual(rebuilt.load_state().pending, {})
    self.assertEqual(rebuilt.pending_run_ids(), [])
    self.assertTrue(rebuilt.load_state().rebuilt)

  def test_an_empty_store_rebuilds_to_an_empty_state_that_is_not_flagged(self) -> None:
    """A first tick has nothing to recover, so it must not be told to widen its window."""
    state = self.store.load_state()

    self.assertFalse(state.rebuilt)
    self.assertIsNone(state.watermark_run_id)
    self.assertIsNone(state.watermark_created_at)
    self.assertEqual(state.attempt_count, 0)

  def test_an_unreadable_index_is_rebuilt_and_said_out_loud(self) -> None:
    """A corrupt index must not stop a tick: the NDJSON is the truth, the index is a cache."""
    self.write_a_tick(self.store, FIRST_TICK)
    self.store.state_path.write_text("{ this is not json", encoding="utf-8")

    with captured_stderr() as warnings:
      state = self.reopen().load_state()

    self.assertEqual(sorted(state.collected), [UNMERGED_RUN_ID, RUN_ID])
    self.assertTrue(state.rebuilt)
    self.assertIn("rebuilding the index", warnings.getvalue())

  def test_an_index_that_is_not_an_object_is_rebuilt_too(self) -> None:
    """Valid JSON of the wrong shape is the same problem as invalid JSON."""
    self.write_a_tick(self.store, FIRST_TICK)
    self.store.state_path.write_text("[]", encoding="utf-8")

    with captured_stderr() as warnings:
      state = self.reopen().load_state()

    self.assertTrue(state.rebuilt)
    self.assertIn("not a state object", warnings.getvalue())

  def test_an_index_from_a_newer_collector_is_refused_rather_than_read_wrong(self) -> None:
    """Reading a future schema as if it were this one would corrupt what the tick believes."""
    self.write_a_tick(self.store, FIRST_TICK)
    payload = json.loads(self.store.state_path.read_text(encoding="utf-8"))
    payload["v"] = store.STATE_VERSION + 1
    self.store.state_path.write_text(json.dumps(payload), encoding="utf-8")

    with self.assertRaises(store.StoreError) as caught:
      self.reopen().load_state()

    self.assertIn("schema version", str(caught.exception))

  def test_an_attempt_still_running_is_rebuilt_as_incomplete(self) -> None:
    """The store's word for "written before GitHub finished with it" survives a rebuild."""
    running = load_json("run.json")
    running["status"] = "in_progress"
    running["conclusion"] = None
    self.store.append(rows.KIND_RUN, [rows.run_row(running, collected_at=FIRST_TICK)])

    state = self.reopen().load_state()

    self.assertEqual(state.incomplete, {RUN_ID: {1}})
    self.assertEqual(state.collected, {})
    self.assertTrue(state.has_attempt(RUN_ID, 1))
    self.assertFalse(state.is_collected(RUN_ID, 1))

  def test_an_attempt_in_flight_for_a_day_is_reported_as_expired(self) -> None:
    """One stuck run must not hold the store open for ever."""
    state = store.State()
    state.mark_pending(RUN_ID, 1, RUN_CREATED, status="in_progress", first_seen_at="2026-09-01T00:00:00Z")
    state.mark_pending(33500000000, 1, "2026-09-02T09:00:00Z", status="queued", first_seen_at="2026-09-02T09:00:00Z")

    expired = state.expired_pending(now=store.parse_timestamp("2026-09-02T10:00:00Z"))

    self.assertEqual(store.PENDING_MAX_AGE_HOURS, 24)
    self.assertEqual([entry.run_id for entry in expired], [RUN_ID])
    self.assertEqual(state.pending_count, 2)

  def test_a_pending_attempt_keeps_the_moment_it_was_first_seen(self) -> None:
    """Otherwise every tick resets the 24-hour clock and the entry never expires."""
    state = store.State()
    state.mark_pending(RUN_ID, 1, RUN_CREATED, status="queued", first_seen_at="2026-09-01T00:00:00Z")

    state.mark_pending(RUN_ID, 1, RUN_CREATED, status="in_progress", first_seen_at="2026-09-01T20:00:00Z")

    entry = state.pending[(RUN_ID, 1)]
    self.assertEqual(entry.first_seen_at, "2026-09-01T00:00:00Z")
    self.assertEqual(entry.status, "in_progress")

  def test_an_attempt_that_is_already_stored_is_not_put_back_in_flight(self) -> None:
    """It has an answer, so it is not waiting for one."""
    state = store.State()
    state.mark_collected(RUN_ID, 1, RUN_CREATED)

    self.assertFalse(state.mark_pending(RUN_ID, 1, RUN_CREATED))
    self.assertEqual(state.pending, {})

  def test_the_index_round_trips_through_the_file(self) -> None:
    """Sets become sorted lists and come back as sets, with the pending list intact."""
    state = store.State()
    state.mark_collected(RUN_ID, 1, RUN_CREATED)
    state.mark_incomplete(UNMERGED_RUN_ID, 1, UNMERGED_CREATED)
    state.mark_pending(33500000000, 2, "2026-09-01T09:00:00Z", status="queued")
    self.store.save_state(state)

    reloaded = self.reopen().load_state()

    self.assertEqual(reloaded.collected, {RUN_ID: {1}})
    self.assertEqual(reloaded.incomplete, {UNMERGED_RUN_ID: {1}})
    self.assertEqual(reloaded.pending[(33500000000, 2)].status, "queued")
    self.assertIsNotNone(reloaded.updated_at)
    self.assertFalse(reloaded.rebuilt)

  def test_forgetting_an_attempt_lets_its_rows_be_harvested_again(self) -> None:
    """The escape hatch for an attempt whose rows were wrong; the old rows stay put."""
    run = load_json("run.json")
    jobs = [rows.job_row(run, job, collected_at=FIRST_TICK) for job in load_jobs("jobs.json")]
    self.store.append(rows.KIND_JOB, jobs[:1], month=RUN_MONTH)
    state = self.store.load_state()
    state.mark_collected(RUN_ID, 1, RUN_CREATED)
    self.store.save_state(state)
    self.assertEqual(self.store.append(rows.KIND_JOB, jobs, month=RUN_MONTH), 0)

    self.assertTrue(state.forget_attempt(RUN_ID, 1))
    self.store.save_state(state)

    self.assertEqual(self.store.append(rows.KIND_JOB, jobs, month=RUN_MONTH), len(jobs) - 1)
    self.assertFalse(state.forget_attempt(RUN_ID, 1))


class CompactMonthTest(StoreTestCase):
  """Compaction of a closed month: drop what is superseded, keep what cannot be re-fetched.

  The store holds one merged pull request run and one that never merged, both created in
  August, both with the same nine-case suite. Compaction has to throw away the per-test
  detail of the run that never merged - its totals live on in the suite row - and leave the
  merged run's detail alone. Running it again must change nothing at all.
  """

  def setUp(self) -> None:
    """Fills the closed month with both runs, their suites, their tests and one correction."""
    super().setUp()
    self.merged_run = load_json(f"merged-pr-5070-run-{MERGED_RUN_ID}.json")
    self.pull_request = load_json("merged-pr-5070-pulls-by-head.json")[0]
    self.unmerged_run = load_json(f"rerun-{UNMERGED_RUN_ID}-run.json")
    entry = suite_entry()

    self.store.append(
        rows.KIND_RUN,
        [
            rows.run_row(self.merged_run, self.pull_request, collected_at=FIRST_TICK),
            rows.run_row(self.unmerged_run, collected_at=FIRST_TICK),
        ],
    )
    self.store.append(
        rows.KIND_SUITE,
        [
            rows.suite_row(self.merged_run, entry, collected_at=FIRST_TICK),
            rows.suite_row(self.unmerged_run, entry, collected_at=FIRST_TICK),
        ],
        month=CLOSED_MONTH,
    )
    self.merged_tests = rows.test_rows(self.merged_run, SMALL_SUITE_ID, 1, entry.result, collected_at=FIRST_TICK)
    self.unmerged_tests = rows.test_rows(self.unmerged_run, SMALL_SUITE_ID, 1, entry.result, collected_at=FIRST_TICK)
    self.store.append(rows.KIND_TEST, self.merged_tests, month=CLOSED_MONTH)
    self.store.append(rows.KIND_TEST, self.unmerged_tests, month=CLOSED_MONTH)
    self.correction = dataclasses.replace(
        self.merged_tests[0], duration=self.merged_tests[0].duration + 2.0, collected_at=SECOND_TICK
    )
    self.store.append(rows.KIND_TEST, [self.correction], month=CLOSED_MONTH, correction=True)

  def test_the_month_starts_with_every_line_that_was_ever_written(self) -> None:
    """The starting point the other tests measure against: nothing was dropped on the way in."""
    self.assertEqual(len(self.lines_of(rows.KIND_TEST, CLOSED_MONTH)), 2 * SMALL_SUITE_CASES + 1)
    self.assertEqual(len(self.store.read(rows.KIND_TEST)), 2 * SMALL_SUITE_CASES)
    self.assertEqual(len(self.lines_of(rows.KIND_SUITE, CLOSED_MONTH)), 2)

  def test_compaction_keeps_the_merged_runs_detail_and_drops_the_rest(self) -> None:
    """The rule that keeps the store inside its budget, and the one that must not overreach."""
    removed = self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID])

    self.assertEqual(removed, SMALL_SUITE_CASES + 1, "nine unmerged rows and one superseded line")
    survivors = self.lines_of(rows.KIND_TEST, CLOSED_MONTH)
    self.assertEqual(len(survivors), SMALL_SUITE_CASES)
    self.assertEqual({line["run_id"] for line in survivors}, {MERGED_RUN_ID})
    self.assertEqual(self.store.read(rows.KIND_TEST), survivors)

  def test_the_superseded_line_goes_and_the_correction_stays(self) -> None:
    """Of the lines sharing a key, only the last one survives - and it is the right one."""
    self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID])

    survivors = self.lines_of(rows.KIND_TEST, CLOSED_MONTH)
    keys = [store.row_key(line) for line in survivors]
    self.assertEqual(len(keys), len(set(keys)), "compaction leaves one line per key")
    corrected = [line for line in survivors if store.row_key(line) == self.correction.key()]
    self.assertEqual(len(corrected), 1)
    self.assertEqual(corrected[0]["duration"], self.correction.duration)
    self.assertEqual(corrected[0]["collected_at"], SECOND_TICK)

  def test_the_totals_of_the_dropped_run_survive_in_its_suite_row(self) -> None:
    """Nothing is actually lost: the per-flavor counts a chart draws are in the suite rows."""
    self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID])

    suites = {line["run_id"]: line for line in self.store.read(rows.KIND_SUITE)}
    self.assertIn(UNMERGED_RUN_ID, suites)
    dropped = suites[UNMERGED_RUN_ID]
    self.assertEqual(dropped["collected"], SMALL_SUITE_CASES)
    self.assertEqual(dropped["skipped"], 7)
    self.assertEqual(dropped["executed"], 2)
    self.assertEqual([line["run_id"] for line in self.store.read(rows.KIND_TEST)], [MERGED_RUN_ID] * SMALL_SUITE_CASES)

  def test_a_second_compaction_removes_nothing_and_does_not_rewrite_the_file(self) -> None:
    """A closed month has to stop changing, or every tick's commit re-touches it."""
    self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID])
    path = self.store.path_for(rows.KIND_TEST, CLOSED_MONTH)
    body = path.read_bytes()
    stamp = path.stat().st_mtime_ns

    again = self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID])

    self.assertEqual(again, 0)
    self.assertEqual(path.read_bytes(), body)
    self.assertEqual(path.stat().st_mtime_ns, stamp, "an unchanged month is not even rewritten")

  def test_compaction_is_idempotent_across_processes(self) -> None:
    """Two collectors, two caches, same answer."""
    self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID])
    body = self.store.path_for(rows.KIND_TEST, CLOSED_MONTH).read_bytes()

    self.assertEqual(self.reopen().compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID]), 0)
    self.assertEqual(self.store.path_for(rows.KIND_TEST, CLOSED_MONTH).read_bytes(), body)

  def test_a_scheduled_run_kept_at_full_resolution_is_kept_by_naming_it(self) -> None:
    """The keep list is wider than its name: a scheduled main run has no pull request."""
    removed = self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID, UNMERGED_RUN_ID])

    self.assertEqual(removed, 1, "only the superseded line")
    self.assertEqual(len(self.lines_of(rows.KIND_TEST, CLOSED_MONTH)), 2 * SMALL_SUITE_CASES)

  def test_an_empty_keep_list_is_refused_because_the_artifacts_are_gone(self) -> None:
    """A caller that failed to load its merged runs must not wipe a month of test detail."""
    with self.assertRaises(store.StoreError) as caught:
      self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [])

    self.assertIn("empty keep list", str(caught.exception))
    self.assertEqual(len(self.lines_of(rows.KIND_TEST, CLOSED_MONTH)), 2 * SMALL_SUITE_CASES + 1)

  def test_an_empty_keep_list_is_allowed_when_the_caller_insists(self) -> None:
    """The override exists, and it says what it does."""
    removed = self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [], allow_dropping_all=True)

    self.assertEqual(removed, 2 * SMALL_SUITE_CASES + 1)
    self.assertEqual(self.lines_of(rows.KIND_TEST, CLOSED_MONTH), [])

  def test_the_open_month_is_refused_because_a_tick_may_be_writing_to_it(self) -> None:
    """Compaction rewrites a file whole, so it must never race an append."""
    open_month = store.month_key(store.utc_now())

    with self.assertRaises(store.StoreError) as caught:
      self.store.compact_month(rows.KIND_TEST, open_month, [MERGED_RUN_ID])

    self.assertIn("has not closed yet", str(caught.exception))

  def test_the_open_month_can_be_compacted_when_the_caller_says_nothing_is_writing(self) -> None:
    """Same override shape as the keep list: allowed, but only on purpose."""
    open_month = store.month_key(store.utc_now())
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)], month=open_month)

    removed = self.store.compact_month(rows.KIND_RUN, open_month, [], allow_open_month=True)

    self.assertEqual(removed, 0)

  def test_other_kinds_ignore_the_keep_list_entirely(self) -> None:
    """Only test rows are dropped by run. A run row is never deleted by compaction."""
    removed = self.store.compact_month(rows.KIND_RUN, CLOSED_MONTH, [])

    self.assertEqual(removed, 0)
    self.assertEqual({line["run_id"] for line in self.store.read(rows.KIND_RUN)}, {MERGED_RUN_ID, UNMERGED_RUN_ID})

  def test_compacting_a_month_with_no_file_is_zero(self) -> None:
    """A month the store never wrote is not an error; it is nothing to do."""
    self.assertEqual(self.store.compact_month(rows.KIND_TEST, "2026-07", [MERGED_RUN_ID]), 0)
    self.assertFalse(self.store.path_for(rows.KIND_TEST, "2026-07").exists())

  def test_a_compacted_month_reads_back_the_same_rows_it_did_before(self) -> None:
    """Compaction changes the file, never the answer `read` gives for what it kept."""
    before = [line for line in self.store.read(rows.KIND_TEST) if line["run_id"] == MERGED_RUN_ID]

    self.store.compact_month(rows.KIND_TEST, CLOSED_MONTH, [MERGED_RUN_ID])

    self.assertEqual(self.reopen().read(rows.KIND_TEST), before)


class AtomicWriteTest(StoreTestCase):
  """A killed tick leaves either the old file or the new one, never half of either.

  The collector runs inside a GitHub Actions job that can be cancelled at any moment, and it
  writes into a checkout that is then committed. A truncated NDJSON line or a half-written
  `state.json` would be committed too, so the interruption is simulated here rather than
  hoped about: `os.replace` is the last step of every write, and it is made to fail.
  """

  @staticmethod
  def refuse_replace(*args: object, **kwargs: object) -> None:
    """Stands in for `os.replace` on a full disk.

    Args:
      *args: Ignored.
      **kwargs: Ignored.

    Raises:
      OSError: Always.
    """
    del args, kwargs
    raise OSError(28, "No space left on device")

  @staticmethod
  def kill_process(*args: object, **kwargs: object) -> None:
    """Stands in for the runner being pulled out from under the tick.

    Args:
      *args: Ignored.
      **kwargs: Ignored.

    Raises:
      KeyboardInterrupt: Always. It is a `BaseException`, not an `Exception`, so it also
        proves the cleanup path does not depend on catching `Exception`.
    """
    del args, kwargs
    raise KeyboardInterrupt("the runner was taken away")

  def test_an_interrupted_append_leaves_the_previous_file_untouched(self) -> None:
    """The first tick's rows are still there, whole, and the second tick's are simply absent."""
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    path = self.store.path_for(rows.KIND_RUN, RUN_MONTH)
    before = path.read_bytes()

    with mock.patch.object(os, "replace", self.refuse_replace):
      with self.assertRaises(store.StoreError) as caught:
        self.store.append(
            rows.KIND_RUN,
            [rows.run_row(load_json(f"rerun-{UNMERGED_RUN_ID}-run.json"), collected_at=SECOND_TICK)],
            month=RUN_MONTH,
        )

    self.assertIn("could not be appended to", str(caught.exception))
    self.assertEqual(path.read_bytes(), before)
    self.assertTrue(before.endswith(b"\n"))
    self.assertEqual(self.temp_leftovers(), [])

  def test_an_interrupted_append_never_leaves_half_a_line(self) -> None:
    """Every stored line is complete, so the next append cannot glue a row onto a fragment."""
    self.store.append(rows.KIND_TEST, self.small_test_rows(FIRST_TICK), month=RUN_MONTH)
    path = self.store.path_for(rows.KIND_TEST, RUN_MONTH)

    with mock.patch.object(os, "replace", self.refuse_replace):
      with self.assertRaises(store.StoreError):
        self.store.append(rows.KIND_TEST, self.small_test_rows(SECOND_TICK), month=RUN_MONTH, correction=True)

    body = path.read_bytes()
    self.assertEqual(len(body), store._complete_length(path))  # pylint: disable=protected-access
    self.assertEqual(len(self.lines_of(rows.KIND_TEST, RUN_MONTH)), SMALL_SUITE_CASES)
    self.assertEqual(len(self.reopen().read(rows.KIND_TEST)), SMALL_SUITE_CASES)

  def test_a_killed_append_cleans_up_after_itself(self) -> None:
    """`BaseException` is not `Exception`; the temporary file has to go either way."""
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    path = self.store.path_for(rows.KIND_RUN, RUN_MONTH)
    before = path.read_bytes()

    with mock.patch.object(os, "replace", self.kill_process):
      with self.assertRaises(KeyboardInterrupt):
        self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=SECOND_TICK)], correction=True)

    self.assertEqual(path.read_bytes(), before)
    self.assertEqual(self.temp_leftovers(), [])

  def test_an_interrupted_first_append_leaves_no_file_at_all(self) -> None:
    """A month with no rows and a month that was never written have to look the same."""
    with mock.patch.object(os, "replace", self.refuse_replace):
      with self.assertRaises(store.StoreError):
        self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])

    self.assertFalse(self.store.path_for(rows.KIND_RUN, RUN_MONTH).exists())
    self.assertEqual(self.store.months(), [])
    self.assertEqual(self.temp_leftovers(), [])

  def test_an_interrupted_index_write_leaves_the_previous_index(self) -> None:
    """A truncated `state.json` would be read as "nothing is stored" on the next tick."""
    self.write_a_tick(self.store, FIRST_TICK)
    before = self.store.state_path.read_bytes()
    state = self.store.load_state()
    state.mark_collected(33500000000, 1, "2026-09-01T09:00:00Z")

    with mock.patch.object(os, "replace", self.refuse_replace):
      with self.assertRaises(store.StoreError):
        self.store.save_state(state)

    self.assertEqual(self.store.state_path.read_bytes(), before)
    self.assertEqual(json.loads(before)["collected"], {str(UNMERGED_RUN_ID): [UNMERGED_ATTEMPT], str(RUN_ID): [1]})
    self.assertEqual(self.temp_leftovers(), [])

  def test_an_interrupted_view_write_leaves_the_view_the_browser_already_has(self) -> None:
    """Half a view JSON breaks the dashboard, not the collector, so it must be impossible."""
    self.store.write_view("runs-2026-09.json", {"columns": ["run_id"], "rows": [[RUN_ID]]})
    path = self.store.views_dir / "runs-2026-09.json"
    before = path.read_bytes()

    with mock.patch.object(os, "replace", self.refuse_replace):
      with self.assertRaises(store.StoreError):
        self.store.write_view("runs-2026-09.json", {"columns": ["run_id"], "rows": [[1]]})

    self.assertEqual(path.read_bytes(), before)
    self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["rows"], [[RUN_ID]])
    self.assertEqual(self.temp_leftovers(), [])

  def test_a_view_that_does_not_serialise_writes_nothing(self) -> None:
    """The error names the file, and no empty file is left where a view should be."""
    with self.assertRaises(store.StoreError) as caught:
      self.store.write_view("pr/5070.json", {"when": object()})

    self.assertIn("could not be serialised", str(caught.exception))
    self.assertFalse((self.store.pr_views_dir / "5070.json").exists())

  def small_test_rows(self, collected_at: str) -> list[Any]:
    """Builds the nine test rows of the small saved suite.

    Args:
      collected_at: The collection timestamp to stamp on them.

    Returns:
      The rows.
    """
    return rows.test_rows(load_json("run.json"), SMALL_SUITE_ID, 1, suite_entry().result, collected_at=collected_at)


class TornFileTest(StoreTestCase):
  """Something outside the collector - a full disk, a killed copy - can still truncate a file.

  The store cannot produce a torn line itself, but it must survive finding one: the rest of
  the file is good data that cannot be fetched again.
  """

  def truncate_the_last_line(self) -> None:
    """Appends half a JSON object with no newline, the way an interrupted copy would."""
    path = self.store.path_for(rows.KIND_RUN, RUN_MONTH)
    with path.open("ab") as handle:
      handle.write(b'{"kind":"run","run_id":33500000000')

  def test_a_torn_last_line_is_reported_and_skipped_not_fatal(self) -> None:
    """Reading has to keep working, and has to say what it ignored."""
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    self.truncate_the_last_line()

    with captured_stderr() as warnings:
      read_back = self.reopen().read(rows.KIND_RUN)

    self.assertEqual([line["run_id"] for line in read_back], [RUN_ID])
    self.assertIn("torn write", warnings.getvalue())

  def test_the_next_append_drops_the_fragment_instead_of_gluing_a_row_onto_it(self) -> None:
    """Two half rows on one line would be unreadable for ever; the fragment goes."""
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    self.truncate_the_last_line()
    fresh = self.reopen()

    with captured_stderr() as warnings:
      written = fresh.append(
          rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=SECOND_TICK)], correction=True
      )

    self.assertEqual(written, 1)
    self.assertIn("torn line", warnings.getvalue())
    lines = self.lines_of(rows.KIND_RUN, RUN_MONTH)
    self.assertEqual([line["run_id"] for line in lines], [RUN_ID, RUN_ID], "the fragment went, the two real rows stayed")
    self.assertEqual([line["collected_at"] for line in lines], [FIRST_TICK, SECOND_TICK])
    self.assertNotIn("33500000000", self.store.path_for(rows.KIND_RUN, RUN_MONTH).read_text(encoding="utf-8"))

  def test_a_broken_line_in_the_middle_of_a_file_is_an_error(self) -> None:
    """A fragment at the end is a torn write; one in the middle is corruption, and stops."""
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    path = self.store.path_for(rows.KIND_RUN, RUN_MONTH)
    path.write_text('{"kind":"run","run_id":1\n' + path.read_text(encoding="utf-8"), encoding="utf-8")

    with self.assertRaises(store.StoreError) as caught:
      self.reopen().read(rows.KIND_RUN)

    self.assertIn("is not JSON", str(caught.exception))

  def test_blank_lines_are_ignored(self) -> None:
    """A stray newline is not a row and is not an error."""
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    path = self.store.path_for(rows.KIND_RUN, RUN_MONTH)
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    self.assertEqual(len(self.reopen().read(rows.KIND_RUN)), 1)

  def test_a_file_that_is_not_named_for_a_month_is_ignored_with_a_warning(self) -> None:
    """Somebody's backup copy in `data/` must not be read as rows."""
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    (self.store.data_dir / f"run-backup{store.NDJSON_SUFFIX}").write_text("{}\n", encoding="utf-8")

    with captured_stderr() as warnings:
      months = self.reopen().months(rows.KIND_RUN)

    self.assertEqual(months, [RUN_MONTH])
    self.assertIn("was ignored", warnings.getvalue())


class MissingIsNullTest(StoreTestCase):
  """A gap goes in as null and comes out as null. A zero would be drawn as a real drop."""

  def test_a_suite_that_published_nothing_stores_nulls_not_zeros(self) -> None:
    """The Pathways flavors publish no JUnit file at all; their counts are unknown, not zero."""
    run = load_json("run.json")
    silent = junit.SuiteEntry(suite_id="tpu-pathways-unit", reason=junit.REASON_NO_FILE)
    self.store.append(rows.KIND_SUITE, [rows.suite_row(run, silent, collected_at=FIRST_TICK)], month=RUN_MONTH)

    stored = self.store.read(rows.KIND_SUITE)[0]

    for missing in ("collected", "skipped", "executed", "failed", "errored", "junit_seconds", "suite_seconds"):
      with self.subTest(field=missing):
        self.assertIsNone(stored[missing])
    self.assertEqual(stored["reason"], junit.REASON_NO_FILE)
    self.assertIn('"collected":null', self.store.path_for(rows.KIND_SUITE, RUN_MONTH).read_text(encoding="utf-8"))

  def test_a_partial_suite_carries_its_flag_and_the_workers_that_are_missing(self) -> None:
    """A total drawn from half the workers must never be drawable as a drop."""
    run = load_json("run.json")
    partial = suite_entry(missing={2: junit.REASON_NO_FILE})
    self.store.append(rows.KIND_SUITE, [rows.suite_row(run, partial, collected_at=FIRST_TICK)], month=RUN_MONTH)

    stored = self.store.read(rows.KIND_SUITE)[0]

    self.assertTrue(stored["is_partial"])
    self.assertEqual(stored["missing_workers"], [{"worker": 2, "reason": junit.REASON_NO_FILE}])
    self.assertEqual(stored["published_workers"], [1])

  def test_a_row_comes_back_field_for_field(self) -> None:
    """The store adds nothing, drops nothing and reorders nothing."""
    run = load_json("run.json")
    original = rows.job_row(run, named(load_jobs("jobs.json"), TPU_UNIT_WORKER_1), collected_at=FIRST_TICK)
    self.store.append(rows.KIND_JOB, [original], month=RUN_MONTH)

    stored = self.store.read(rows.KIND_JOB)[0]

    self.assertEqual(stored, rows.to_json(original))
    self.assertEqual(list(stored), list(rows.to_json(original)))
    self.assertEqual(self.store.read_rows(rows.KIND_JOB)[0], original)

  def test_a_test_row_with_no_worker_number_stores_the_gap(self) -> None:
    """An unknown worker is an empty key part, and stays null in the row."""
    run = load_json("run.json")
    rows_without_worker = rows.test_rows(run, SMALL_SUITE_ID, None, suite_entry().result, collected_at=FIRST_TICK)
    self.store.append(rows.KIND_TEST, rows_without_worker, month=RUN_MONTH)

    stored = self.store.read(rows.KIND_TEST)

    self.assertEqual(len(stored), SMALL_SUITE_CASES)
    self.assertTrue(all(line["worker"] is None for line in stored))
    self.assertTrue(all(store.row_key(line).count(rows.KEY_SEPARATOR) == 6 for line in stored))


class ViewWriteTest(StoreTestCase):
  """Views are what the browser fetches, so their names are checked before they are written."""

  def test_a_view_lands_under_views_and_a_pull_request_view_under_views_pr(self) -> None:
    """The dashboard's loader hard-codes both paths."""
    monthly = self.store.write_view("runs-2026-09.json", {"columns": ["run_id"], "rows": [[RUN_ID]]})
    per_pr = self.store.write_view("pr/5070.json", {"pr": 5070})

    self.assertEqual(monthly, self.store.views_dir / "runs-2026-09.json")
    self.assertEqual(per_pr, self.store.pr_views_dir / "5070.json")
    self.assertEqual(json.loads(per_pr.read_text(encoding="utf-8")), {"pr": 5070})
    self.assertTrue(per_pr.read_text(encoding="utf-8").endswith("\n"))

  def test_a_view_name_cannot_escape_the_views_directory(self) -> None:
    """An absolute path or a walk upwards would write outside the store."""
    for bad in ("/etc/passwd.json", "../data/run-2026-09.json", "pr/../../escape.json"):
      with self.subTest(name=bad):
        with self.assertRaises(store.StoreError) as caught:
          self.store.write_view(bad, {})
        self.assertIn("stay inside", str(caught.exception))

  def test_a_view_has_to_be_json(self) -> None:
    """The browser fetches `.json`; anything else is a caller mistake worth catching."""
    with self.assertRaises(store.StoreError) as caught:
      self.store.write_view("runs-2026-09.txt", {})
    self.assertIn(".json", str(caught.exception))

  def test_rewriting_a_view_replaces_it_whole(self) -> None:
    """Views are rebuilt every tick, so the old one is never appended to."""
    self.store.write_view("meta.json", {"generated_at": FIRST_TICK})

    self.store.write_view("meta.json", {"generated_at": SECOND_TICK})

    body = (self.store.views_dir / "meta.json").read_text(encoding="utf-8")
    self.assertEqual(json.loads(body), {"generated_at": SECOND_TICK})
    self.assertEqual(body.count("\n"), 1)


class RescueRowsCorrectThemselvesTest(StoreTestCase):
  """A rescue's key names the failure, not the outcome, so its row has to be re-writable.

  The tick that sees only attempt 1 stores "this job failed and was never re-run". Hours
  later somebody re-runs it, the next tick sees attempt 2, and the same key now means
  "rescued". If the store skipped that second row on its key, the flaky card would report
  every rescue as an unrescued failure for ever.
  """

  def rescue_of(self, attempts: dict[int, list[dict[str, Any]]], last: int) -> list[Any]:
    """Builds the rescue rows a tick would derive from the attempts it has read.

    Args:
      attempts: Attempt number -> that attempt's jobs.
      last: The highest attempt read so far.

    Returns:
      The rescue rows, rescued ones first, exactly as `tick.py` builds them.
    """
    seen = {number: jobs for number, jobs in attempts.items() if number <= last}
    run = load_json(f"rerun-{UNMERGED_RUN_ID}-run.json")
    stamp = FIRST_TICK if last == 1 else SECOND_TICK
    return rows.rescue_rows(run, seen, collected_at=stamp) + rows.failed_never_rescued_rows(run, seen, collected_at=stamp)

  def test_a_re_run_seen_on_a_later_tick_lands_as_a_correction(self) -> None:
    """The headline: the second answer reaches the store and is the one a reader gets."""
    attempts = attempts_of(UNMERGED_RUN_ID, "rerun", UNMERGED_ATTEMPT)
    before = self.rescue_of(attempts, 1)
    self.assertTrue(before, "attempt 1 has failures to record")
    self.assertEqual(self.store.append(rows.KIND_RESCUE, before), len(before))

    after = self.rescue_of(attempts, UNMERGED_ATTEMPT)
    written = self.store.append(rows.KIND_RESCUE, after)

    self.assertEqual(written, len(after), "every rescue changed its answer, so every row is written")
    rescued = {payload["job_name"]: payload for payload in self.store.read(rows.KIND_RESCUE)}
    upgraded = [payload for payload in rescued.values() if payload["rescued"]]
    self.assertTrue(upgraded, "the re-run has to be visible after the second tick")
    for payload in upgraded:
      self.assertEqual(payload["rescued_attempt"], UNMERGED_ATTEMPT)
      self.assertEqual(payload["final_conclusion"], "success")

  def test_the_first_answer_is_still_on_disk(self) -> None:
    """Append-only: the correction is a second line, not an edit."""
    attempts = attempts_of(UNMERGED_RUN_ID, "rerun", UNMERGED_ATTEMPT)
    first = self.rescue_of(attempts, 1)
    self.store.append(rows.KIND_RESCUE, first)
    self.store.append(rows.KIND_RESCUE, self.rescue_of(attempts, UNMERGED_ATTEMPT))

    lines = self.lines_of(rows.KIND_RESCUE, CLOSED_MONTH)

    self.assertGreater(len(lines), len(self.store.read(rows.KIND_RESCUE)))
    self.assertFalse(lines[0]["rescued"], "the first line is still the answer the first tick had")

  def test_an_unchanged_rescue_is_still_written_only_once(self) -> None:
    """Content dedup, not "always write": a repeat must stay a no-op."""
    attempts = attempts_of(UNMERGED_RUN_ID, "rerun", UNMERGED_ATTEMPT)
    settled = self.rescue_of(attempts, UNMERGED_ATTEMPT)
    self.store.append(rows.KIND_RESCUE, settled)
    before = self.ndjson_snapshot()

    repeated = rows.rescue_rows(
        load_json(f"rerun-{UNMERGED_RUN_ID}-run.json"), attempts, collected_at="2026-09-02T00:00:00Z"
    ) + rows.failed_never_rescued_rows(
        load_json(f"rerun-{UNMERGED_RUN_ID}-run.json"), attempts, collected_at="2026-09-02T00:00:00Z"
    )

    self.assertEqual(self.store.append(rows.KIND_RESCUE, repeated), 0)
    self.assertEqual(self.reopen().append(rows.KIND_RESCUE, repeated), 0, "and in the next process too")
    self.assertEqual(self.ndjson_snapshot(), before)


class IndexPruningTest(StoreTestCase):
  """`state.json` is committed six times a day, so it cannot grow without limit."""

  def test_saving_drops_the_oldest_run_ids_beyond_the_cap(self) -> None:
    """The newest ids are kept, because run ids rise with time."""
    state = self.store.load_state()
    for offset in range(store.MAX_INDEXED_RUNS + 50):
      state.mark_collected(33000000000 + offset, 1)

    self.store.save_state(state)

    reloaded = self.reopen().load_state()
    self.assertEqual(len(reloaded.collected), store.MAX_INDEXED_RUNS)
    self.assertIn(33000000000 + store.MAX_INDEXED_RUNS + 49, reloaded.collected)
    self.assertNotIn(33000000000, reloaded.collected)

  def test_an_index_under_the_cap_is_left_alone(self) -> None:
    """Nothing is dropped from a normal store, so the file is not churned."""
    state = self.store.load_state()
    state.mark_collected(RUN_ID, 1, RUN_CREATED)
    state.mark_incomplete(UNMERGED_RUN_ID, 1, UNMERGED_CREATED)

    self.assertEqual(state.prune(), 0)
    self.assertEqual(sorted(state.collected), [RUN_ID])
    self.assertEqual(sorted(state.incomplete), [UNMERGED_RUN_ID])

  def test_an_in_flight_attempt_is_never_pruned(self) -> None:
    """The one thing a rebuild cannot recover has to survive a prune."""
    state = self.store.load_state()
    state.mark_pending(1, 1, "2026-09-01T00:00:00Z", status="in_progress")
    for offset in range(store.MAX_INDEXED_RUNS + 5):
      state.mark_collected(33000000000 + offset, 1)
    state.mark_pending(2, 1, "2026-09-01T00:00:00Z", status="in_progress")

    state.prune()

    self.assertEqual(sorted(run_id for run_id, _ in state.pending), [1, 2])

  def test_a_pruned_run_can_still_be_stored_without_duplicating_it(self) -> None:
    """Dropping an id costs a file read, never a duplicated row."""
    run = load_json("run.json")
    self.store.append(rows.KIND_RUN, [rows.run_row(run, collected_at=FIRST_TICK)])
    state = self.store.load_state()
    state.mark_collected(RUN_ID, 1, RUN_CREATED)
    self.store.save_state(state)
    state.prune(keep=0)
    self.store.save_state(state)

    written = self.reopen().append(rows.KIND_RUN, [rows.run_row(run, collected_at=SECOND_TICK)])

    self.assertEqual(written, 0, "the file's own keys are the exact answer")


class TemporaryFileSweepTest(StoreTestCase):
  """A killed process leaves a full copy of a month behind; something has to clear it."""

  def leftover(self, name: str, age_hours: float) -> Path:
    """Puts a fake leftover temporary file in the data directory.

    Args:
      name: The file name, without the temporary prefix.
      age_hours: How long ago it was last written.

    Returns:
      The path it was written to.
    """
    self.store.append(rows.KIND_RUN, [rows.run_row(load_json("run.json"), collected_at=FIRST_TICK)])
    path = self.store.data_dir / f"{store.TEMP_PREFIX}{name}"
    path.write_text("{}\n", encoding="utf-8")
    old = path.stat().st_mtime - age_hours * 3600
    os.utime(path, (old, old))
    return path

  def test_an_old_leftover_is_removed_and_said_out_loud(self) -> None:
    """It is not a data loss, so it warns rather than failing the tick."""
    path = self.leftover("abcd.ndjson", store.TEMP_MAX_AGE_HOURS + 1)

    with captured_stderr() as warnings:
      swept = self.store.sweep_temp()

    self.assertEqual(swept, 1)
    self.assertFalse(path.exists())
    self.assertIn("killed mid-write", warnings.getvalue())
    self.assertEqual(self.temp_leftovers(), [])

  def test_a_temporary_file_that_might_still_be_in_use_is_left_alone(self) -> None:
    """Two collectors on one store must not delete each other's half-written file."""
    path = self.leftover("efgh.ndjson", 0)

    self.assertEqual(self.store.sweep_temp(), 0)
    self.assertTrue(path.exists())

  def test_sweeping_an_empty_store_is_not_an_error(self) -> None:
    """Nothing has been written yet, so there is no directory to look in."""
    self.assertEqual(self.store.sweep_temp(), 0)

  def test_the_stored_rows_are_untouched(self) -> None:
    """The sweep only ever removes files with the temporary prefix."""
    self.leftover("ijkl.ndjson", store.TEMP_MAX_AGE_HOURS + 1)
    stored = self.store.path_for(rows.KIND_RUN, RUN_MONTH)
    before = stored.read_bytes()

    with captured_stderr():
      self.store.sweep_temp()

    self.assertEqual(stored.read_bytes(), before)
    self.assertEqual(sorted(path.name for path in self.store.data_dir.iterdir()), [stored.name])


class HalfWrittenAttemptTest(StoreTestCase):
  """Recovering from a tick that died between two of an attempt's five kinds.

  Both cases here were found as defects while these tests were written, and both are now
  fixed in `store.py`:

  1. A tick that dies after writing its run row, before `state.json` is ever saved, used to
     lose the rest of that attempt for good. The next tick rebuilt the index by scanning run
     rows, the run row alone marked the attempt "collected", and `append` then skipped every
     job, suite and test row of it. The rebuild now asks for the jobs as well.

  2. `mark_collected` cleared the attempt from `incomplete`, but `mark_incomplete` did not
     clear it from `collected`, so an attempt could sit in both indexes at once.
  """

  def test_a_tick_that_died_after_the_run_row_can_still_store_the_rest(self) -> None:
    """The jobs of a half-written attempt are storable, not silently skipped."""
    run = load_json("run.json")
    first = self.reopen()
    first.append(rows.KIND_RUN, [rows.run_row(run, collected_at=FIRST_TICK)])
    self.assertFalse(self.store.state_path.exists(), "the tick died before it saved its index")

    job_rows = [rows.job_row(run, job, collected_at=SECOND_TICK) for job in load_jobs("jobs.json")]
    written = self.reopen().append(rows.KIND_JOB, job_rows, month=RUN_MONTH)

    self.assertEqual(written, len(job_rows), "the attempt's jobs were never stored, so they must be storable")

  def test_a_run_row_with_no_jobs_is_not_reported_as_collected(self) -> None:
    """The rebuild says what it can prove: a run row alone does not mean a whole attempt."""
    run = load_json("run.json")
    self.store.append(rows.KIND_RUN, [rows.run_row(run, collected_at=FIRST_TICK)])

    rebuilt = self.reopen().load_state()

    self.assertFalse(rebuilt.is_collected(RUN_ID, 1))
    self.assertFalse(rebuilt.has_attempt(RUN_ID, 1))
    self.assertEqual(rebuilt.watermark_run_id, RUN_ID, "the run was still seen, so the watermark moves")

  def test_an_attempt_with_its_jobs_is_reported_as_collected(self) -> None:
    """The other half of the same rule: a whole attempt is recognised without its index."""
    run = load_json("run.json")
    self.store.append(rows.KIND_RUN, [rows.run_row(run, collected_at=FIRST_TICK)])
    self.store.append(
        rows.KIND_JOB,
        [rows.job_row(run, job, collected_at=FIRST_TICK) for job in load_jobs("jobs.json")],
        month=RUN_MONTH,
    )

    rebuilt = self.reopen().load_state()

    self.assertTrue(rebuilt.is_collected(RUN_ID, 1))

  def test_an_attempt_is_only_ever_in_one_index(self) -> None:
    """`collected` and `incomplete` are exclusive in both directions."""
    state = store.State()
    state.mark_collected(RUN_ID, 1, RUN_CREATED)

    state.mark_incomplete(RUN_ID, 1, RUN_CREATED)

    self.assertFalse(state.is_collected(RUN_ID, 1), "an attempt cannot be both final and not final")
    self.assertTrue(state.is_incomplete(RUN_ID, 1))
    self.assertEqual(state.attempt_count, 1)

    state.mark_collected(RUN_ID, 1, RUN_CREATED)

    self.assertFalse(state.is_incomplete(RUN_ID, 1))
    self.assertEqual(state.attempt_count, 1)


if __name__ == "__main__":
  unittest.main(verbosity=2)
