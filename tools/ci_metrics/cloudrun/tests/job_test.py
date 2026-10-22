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

"""Offline tests for the Cloud Run job: settings, lock, download, upload order and the whole run.

A bucket held in memory stands in for GCS. It keeps generations the way GCS does, so the
generation preconditions the job relies on are exercised for real. Nothing here reaches GCS or
GitHub; the one test that starts the real collector gives it an impossible argument, which it
refuses before it opens a connection.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

# tools/ci_metrics, so `cloudrun` and `collector` import as packages.
_APP_DIR = str(Path(__file__).resolve().parents[2])
if _APP_DIR not in sys.path:
  sys.path.insert(0, _APP_DIR)

from cloudrun import email_report  # pylint: disable=wrong-import-position
from cloudrun import job  # pylint: disable=wrong-import-position
from collector import rows as collector_rows  # pylint: disable=wrong-import-position
from collector import tick as collector_tick  # pylint: disable=wrong-import-position

# pylint: disable=protected-access

T0 = datetime(2026, 10, 20, 12, 20, tzinfo=timezone.utc)


def md5_of(data: bytes) -> str:
  """Returns bytes' MD5 the way GCS reports it."""
  return base64.b64encode(hashlib.md5(data).digest()).decode("ascii")


class MemoryObjects:
  """An in-memory bucket with GCS generation semantics, recording every call."""

  def __init__(self, clock: Any = None) -> None:
    self.objects: dict[str, tuple[bytes, int, datetime]] = {}
    self.next_generation = 1000
    self.calls: list[tuple[str, str]] = []
    self.clock = clock or (lambda: T0)
    self.fail_upload: set[str] = set()

  def put(self, name: str, data: bytes, created: datetime | None = None) -> int:
    """Stores an object directly, as if something else wrote it."""
    self.next_generation += 1
    self.objects[name] = (data, self.next_generation, created or self.clock())
    return self.next_generation

  def _info(self, name: str) -> job.ObjectInfo:
    data, generation, created = self.objects[name]
    return job.ObjectInfo(name=name, md5=md5_of(data), generation=generation, size=len(data), created=created)

  def list(self, prefix: str) -> dict[str, job.ObjectInfo]:
    self.calls.append(("list", prefix))
    return {name: self._info(name) for name in sorted(self.objects) if name.startswith(prefix)}

  def download(self, name: str, generation: int, path: Path) -> None:
    self.calls.append(("download", name))
    data, current, _ = self.objects[name]
    if current != generation:
      raise RuntimeError(f"generation {generation} of {name} is gone")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

  def upload(self, path: Path, name: str, if_generation_match: int, content_type_value: str) -> None:
    self.calls.append(("upload", name))
    if name in self.fail_upload:
      raise job.SyncError(f"{name}: scripted failure")
    current = self.objects[name][1] if name in self.objects else 0
    if current != if_generation_match:
      raise job.SyncError(f"{name} changed in the bucket while this tick ran; nothing more is uploaded")
    assert content_type_value in ("application/json", "application/x-ndjson")
    self.put(name, path.read_bytes())

  def create_if_absent(self, name: str, data: bytes) -> int | None:
    self.calls.append(("create", name))
    if name in self.objects:
      return None
    return self.put(name, data)

  def info(self, name: str) -> job.ObjectInfo | None:
    return self._info(name) if name in self.objects else None

  def delete(self, name: str, if_generation_match: int) -> bool:
    self.calls.append(("delete", name))
    if name not in self.objects or self.objects[name][1] != if_generation_match:
      return False
    del self.objects[name]
    return True

  def names(self) -> list[str]:
    return sorted(self.objects)


class ConfigTest(unittest.TestCase):

  def test_bucket_is_required(self):
    with self.assertRaises(job.ConfigError):
      job.Config.from_env({})

  def test_defaults(self):
    config = job.Config.from_env({"CI_PULSE_BUCKET": "b"})
    self.assertEqual(config.prefix, "")
    self.assertEqual(config.repo, "AI-Hypercomputer/maxtext")
    self.assertEqual(config.tick_args, ())
    self.assertEqual(config.lock_stale_seconds, 1800)
    self.assertEqual(config.tick_timeout_seconds, 600)
    self.assertFalse(config.no_upload)
    self.assertEqual(config.lock_name, ".lock")

  def test_tick_args_are_split_like_a_shell(self):
    config = job.Config.from_env({"CI_PULSE_BUCKET": "b", "CI_PULSE_TICK_ARGS": "--skip-tests --rate-floor 15"})
    self.assertEqual(config.tick_args, ("--skip-tests", "--rate-floor", "15"))

  def test_the_job_owns_out_repo_and_dry_run(self):
    for args in ("--out /x", "--out=/x", "--repo a/b", "--dry-run"):
      with self.subTest(args=args):
        with self.assertRaises(job.ConfigError):
          job.Config.from_env({"CI_PULSE_BUCKET": "b", "CI_PULSE_TICK_ARGS": args})

  def test_prefix_must_be_a_relative_folder(self):
    self.assertEqual(job.Config.from_env({"CI_PULSE_BUCKET": "b", "CI_PULSE_PREFIX": "poc/"}).lock_name, "poc/.lock")
    for prefix in ("/abs/", "no-slash", "a/../b/"):
      with self.subTest(prefix=prefix):
        with self.assertRaises(job.ConfigError):
          job.Config.from_env({"CI_PULSE_BUCKET": "b", "CI_PULSE_PREFIX": prefix})

  def test_numbers_are_checked(self):
    for name in ("CI_PULSE_LOCK_STALE_SECONDS", "CI_PULSE_TICK_TIMEOUT_SECONDS"):
      for value in ("0", "-5", "ten"):
        with self.subTest(name=name, value=value):
          with self.assertRaises(job.ConfigError):
            job.Config.from_env({"CI_PULSE_BUCKET": "b", name: value})


class NamesTest(unittest.TestCase):

  def test_store_and_view_files_are_managed(self):
    for name in (
        "data/run-2026-10.ndjson",
        "data/job-2026-10-part2.ndjson",
        "data/state.json",
        "data/email_state.json",
        "views/meta.json",
        "views/runs-2026-10.json",
        "views/pr/5599.json",
    ):
      with self.subTest(name=name):
        self.assertTrue(job.is_managed(name))

  def test_everything_else_is_left_alone(self):
    for name in (
        ".lock",
        "README.md",
        "data/.tmp-abc.ndjson",
        "views/runs-2026-10.json.tmp",
        "views/pr/x.json",
        "data/../views/meta.json",
        "other/views/meta.json",
        "data/run-2026-10.ndjson.bak",
    ):
      with self.subTest(name=name):
        self.assertFalse(job.is_managed(name))

  def test_upload_order_puts_meta_last(self):
    names = [
        "views/meta.json",
        "views/runs-2026-10.json",
        "data/state.json",
        "data/run-2026-10.ndjson",
        "views/pr/1.json",
    ]
    ordered = sorted(names, key=lambda name: (job.upload_phase(name), name))
    self.assertEqual(
        ordered,
        ["data/run-2026-10.ndjson", "data/state.json", "views/pr/1.json", "views/runs-2026-10.json", "views/meta.json"],
    )


class LockTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.now = T0
    self.objects = MemoryObjects(clock=lambda: self.now)

  def lock(self, owner: str) -> job.Lock:
    return job.Lock(self.objects, ".lock", 1800, owner, clock=lambda: self.now)

  def test_first_tick_takes_the_lock_and_releases_it(self):
    first = self.lock("a")
    self.assertTrue(first.acquire())
    self.assertEqual(json.loads(self.objects.objects[".lock"][0])["owner"], "a")
    self.assertTrue(first.release())
    self.assertNotIn(".lock", self.objects.objects)

  def test_a_second_tick_is_kept_out(self):
    self.assertTrue(self.lock("a").acquire())
    self.now = T0 + timedelta(minutes=29)
    second = self.lock("b")
    self.assertFalse(second.acquire())
    self.assertIsNone(second.generation)
    self.assertFalse(second.release())

  def test_a_stale_lock_is_taken_over(self):
    self.assertTrue(self.lock("dead").acquire())
    self.now = T0 + timedelta(minutes=31)
    second = self.lock("b")
    self.assertTrue(second.acquire())
    self.assertIn("taken over", second.took_over)
    self.assertEqual(json.loads(self.objects.objects[".lock"][0])["owner"], "b")

  def test_release_never_removes_someone_elses_lock(self):
    first = self.lock("a")
    self.assertTrue(first.acquire())
    self.now = T0 + timedelta(minutes=31)
    self.assertTrue(self.lock("b").acquire())
    self.assertFalse(first.release())
    self.assertEqual(json.loads(self.objects.objects[".lock"][0])["owner"], "b")


class SyncTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.tmp = Path(tempfile.mkdtemp(prefix="ci-pulse-sync-"))
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
    self.store = self.tmp / "store"
    self.objects = MemoryObjects()

  def write(self, name: str, data: bytes) -> None:
    path = self.store / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

  def test_download_takes_managed_objects_only(self):
    self.objects.put("p/data/run-2026-10.ndjson", b"r\n")
    self.objects.put("p/views/meta.json", b"{}")
    self.objects.put("p/.lock", b"{}")
    self.objects.put("p/notes.txt", b"x")
    self.objects.put("other/data/run-2026-10.ndjson", b"no")
    remote, report = job.download_store(self.objects, "p/", self.store)
    self.assertEqual(sorted(remote), ["data/run-2026-10.ndjson", "views/meta.json"])
    self.assertEqual((self.store / "data/run-2026-10.ndjson").read_bytes(), b"r\n")
    self.assertFalse((self.store / "notes.txt").exists())
    self.assertEqual(report.files, 2)
    self.assertEqual(report.bytes, 4)

  def test_download_refuses_a_corrupt_copy(self):
    self.objects.put("data/run-2026-10.ndjson", b"good")
    real_download = self.objects.download

    def corrupt(name: str, generation: int, path: Path) -> None:
      real_download(name, generation, path)
      path.write_bytes(b"bad")

    self.objects.download = corrupt
    with self.assertRaises(job.SyncError):
      job.download_store(self.objects, "", self.store)

  def test_upload_sends_only_changes_in_order_and_deletes_last(self):
    self.objects.put("data/run-2026-09.ndjson", b"same")
    self.objects.put("data/run-2026-10.ndjson", b"old")
    self.objects.put("data/run-2026-06.ndjson", b"pruned by the tick")
    self.objects.put("views/meta.json", b"{}")
    self.objects.put("views/runs-2026-06.json", b"pruned too")
    remote, _ = job.download_store(self.objects, "", self.store)
    (self.store / "data/run-2026-06.ndjson").unlink()
    (self.store / "views/runs-2026-06.json").unlink()
    self.write("data/run-2026-10.ndjson", b"old\nnew\n")
    self.write("data/state.json", b"{}")
    self.write("views/runs-2026-10.json", b"{}")
    self.write("views/meta.json", b'{"generated_at": "now"}')
    self.write("data/.tmp-left.ndjson", b"temp")
    self.objects.calls.clear()

    report = job.upload_store(self.objects, "", self.store, remote)

    self.assertEqual(
        self.objects.calls,
        [
            ("upload", "data/run-2026-10.ndjson"),
            ("upload", "data/state.json"),
            ("upload", "views/runs-2026-10.json"),
            ("upload", "views/meta.json"),
            ("delete", "data/run-2026-06.ndjson"),
            ("delete", "views/runs-2026-06.json"),
        ],
    )
    self.assertEqual(report.files, 4)
    self.assertEqual(report.unchanged, 1)
    self.assertEqual(report.deleted, 2)
    self.assertEqual(report.skipped_local, ["data/.tmp-left.ndjson"])
    self.assertEqual(self.objects.objects["data/run-2026-10.ndjson"][0], b"old\nnew\n")
    self.assertNotIn("data/.tmp-left.ndjson", self.objects.objects)

  def test_upload_stops_when_the_bucket_changed_underneath(self):
    self.objects.put("data/run-2026-10.ndjson", b"old")
    remote, _ = job.download_store(self.objects, "", self.store)
    self.write("data/run-2026-10.ndjson", b"mine")
    self.objects.put("data/run-2026-10.ndjson", b"someone else")
    with self.assertRaises(job.SyncError):
      job.upload_store(self.objects, "", self.store, remote)
    self.assertEqual(self.objects.objects["data/run-2026-10.ndjson"][0], b"someone else")

  def test_a_new_file_is_only_created_if_still_absent(self):
    remote, _ = job.download_store(self.objects, "", self.store)
    self.write("views/pr/7.json", b"{}")
    self.objects.put("views/pr/7.json", b"raced")
    with self.assertRaises(job.SyncError):
      job.upload_store(self.objects, "", self.store, remote)

  def test_a_store_that_vanished_is_not_mirrored(self):
    for month in range(1, 31):
      self.objects.put(f"views/pr/{month}.json", b"{}")
    remote, _ = job.download_store(self.objects, "", self.store)
    shutil.rmtree(self.store)
    with self.assertRaises(job.SyncError) as caught:
      job.upload_store(self.objects, "", self.store, remote)
    self.assertIn("refusing", str(caught.exception))
    self.assertEqual(len(self.objects.objects), 30)

  def test_a_small_prune_is_allowed(self):
    for number in range(1, 31):
      self.objects.put(f"views/pr/{number}.json", b"{}")
    remote, _ = job.download_store(self.objects, "", self.store)
    for number in range(1, 11):
      (self.store / f"views/pr/{number}.json").unlink()
    report = job.upload_store(self.objects, "", self.store, remote)
    self.assertEqual(report.deleted, 10)
    self.assertEqual(len(self.objects.objects), 20)

  def test_the_prefix_is_kept_on_every_name(self):
    self.objects.put("poc/views/meta.json", b"{}")
    remote, _ = job.download_store(self.objects, "poc/", self.store)
    self.write("views/meta.json", b'{"a": 1}')
    job.upload_store(self.objects, "poc/", self.store, remote)
    self.assertEqual(self.objects.names(), ["poc/views/meta.json"])
    self.assertEqual(self.objects.objects["poc/views/meta.json"][0], b'{"a": 1}')


class RunCommandTest(unittest.TestCase):

  def test_output_is_echoed_and_the_summary_kept(self):
    script = "print('report line'); print('ci-metrics tick: ok | 3 run(s) | 9 API request(s)'); print('after')"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
      result = job.run_command([sys.executable, "-c", script], 30, {"PATH": "/usr/bin:/bin"})
    self.assertEqual(result.exit_code, 0)
    self.assertEqual(result.summary, "ci-metrics tick: ok | 3 run(s) | 9 API request(s)")
    self.assertEqual(result.lines, ["report line", "ci-metrics tick: ok | 3 run(s) | 9 API request(s)", "after"])
    self.assertFalse(result.timed_out)
    self.assertIn("report line", printed.getvalue())

  def test_exit_codes_pass_through(self):
    with contextlib.redirect_stdout(io.StringIO()):
      result = job.run_command([sys.executable, "-c", "raise SystemExit(1)"], 30, {})
    self.assertEqual(result.exit_code, 1)
    self.assertIsNone(result.summary)

  def test_a_tick_past_its_deadline_is_stopped(self):
    with contextlib.redirect_stdout(io.StringIO()):
      result = job.run_command([sys.executable, "-c", "import time; time.sleep(60)"], 1, {})
    self.assertTrue(result.timed_out)
    self.assertIsNone(result.exit_code)

  def test_the_real_collector_starts_and_refuses_an_impossible_argument(self):
    config = job.Config(bucket="b", tick_args=("--backfill-days", "0"))
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
      result = job.run_tick(config, Path(tmp) / "store", {"PATH": "/usr/bin:/bin"})
    self.assertEqual(result.exit_code, job.EXIT_USAGE)


class ReportValuesTest(unittest.TestCase):

  LINES = [
      "",
      "CI metrics tick - AI-Hypercomputer/maxtext",
      "  API requests spent         72 (+0 downloads)",
      "  API budget                 60 of 60 left at the first check, 13 of 60 at the last",
      "  Stopped early              the API budget ran low: 13 of 60 requests left, floor is 15.",
      "  Problem                    run 1 could not be collected (boom); it will be tried again next tick.",
      "  Problem                    run 2 could not be collected (bang); it will be tried again next tick.",
      "ci-metrics backfill: ok | 25 run(s) | stopped early",
  ]

  def test_reads_one_labelled_line(self):
    self.assertEqual(
        job.report_values(self.LINES, "API budget"), ["60 of 60 left at the first check, 13 of 60 at the last"]
    )
    self.assertEqual(
        job.report_values(self.LINES, "Stopped early"), ["the API budget ran low: 13 of 60 requests left, floor is 15."]
    )

  def test_reads_every_line_with_the_label(self):
    self.assertEqual(len(job.report_values(self.LINES, "Problem")), 2)

  def test_a_label_is_matched_whole(self):
    self.assertEqual(job.report_values(self.LINES, "API"), [])
    self.assertEqual(job.report_values(self.LINES, "Problems"), [])
    self.assertEqual(job.report_values([], "API budget"), [])

  def test_reads_what_the_real_tick_report_prints(self):
    # Ties the parser to collector.tick.print_report, so a change of its layout fails here.
    report = collector_tick.TickReport(
        budget_first="60 of 60",
        budget_last="13 of 60",
        stopped_because="the API budget ran low: 13 of 60 requests left, floor is 15.",
        problems=["run 1 could not be collected (boom)."],
    )
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
      collector_tick.print_report(report, collector_tick.Options(out=Path("/nonexistent")))
    lines = printed.getvalue().splitlines()
    self.assertEqual(job.report_values(lines, "API budget"), ["60 of 60 left at the first check, 13 of 60 at the last"])
    self.assertEqual(job.report_values(lines, "Stopped early"), [report.stopped_because])
    self.assertEqual(job.report_values(lines, "Problem"), report.problems)


class RunTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.tmp = Path(tempfile.mkdtemp(prefix="ci-pulse-run-"))
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
    self.objects = MemoryObjects(clock=lambda: datetime.now(timezone.utc))
    self.objects.put("data/run-2026-10.ndjson", b'{"k": 1}\n')
    self.objects.put("views/meta.json", b'{"generated_at": "before"}')
    self.config = job.Config(bucket="b", workdir=str(self.tmp))

  def fake_tick(self, code: int | None, timed_out: bool = False, write: bool = True, lines: list[str] | None = None):
    """A stand-in for the collector that edits the downloaded store the way a tick would."""

    def tick(config: job.Config, store_dir: Path, env: Any) -> job.TickResult:
      del config, env
      self.assertTrue((store_dir / "views/meta.json").is_file(), "the tick must see the downloaded store")
      if write:
        with (store_dir / "data/run-2026-10.ndjson").open("ab") as handle:
          handle.write(b'{"k": 2}\n')
        (store_dir / "views/meta.json").write_bytes(b'{"generated_at": "after"}')
      return job.TickResult(
          exit_code=code,
          summary="ci-metrics tick: ok | 1 run(s) | 29 API request(s)",
          timed_out=timed_out,
          lines=list(lines or []),
      )

    return tick

  def run_job(self, tick: Any, config: job.Config | None = None) -> tuple[int, dict[str, Any]]:
    """Runs the job with a fake tick; returns its exit code and its last log line."""
    printed = io.StringIO()
    with mock.patch.object(job, "run_tick", side_effect=tick), contextlib.redirect_stdout(printed):
      code = job.run(config or self.config, self.objects, {"CLOUD_RUN_EXECUTION": "exec-1"})
    last = json.loads(printed.getvalue().strip().splitlines()[-1])
    return code, last

  def test_the_summary_carries_the_ticks_own_budget_and_stop_reason(self):
    code, last = self.run_job(self.fake_tick(job.EXIT_OK, lines=ReportValuesTest.LINES))
    self.assertEqual(code, job.EXIT_OK)
    facts = last["ci_pulse"]
    self.assertEqual(facts["tick_budget"], "60 of 60 left at the first check, 13 of 60 at the last")
    self.assertIn("13 of 60 requests left", facts["tick_stopped_early"])
    self.assertEqual(len(facts["tick_problems"]), 2)

  def test_a_good_tick_is_saved_and_the_lock_released(self):
    code, last = self.run_job(self.fake_tick(job.EXIT_OK))
    self.assertEqual(code, job.EXIT_OK)
    self.assertEqual(self.objects.objects["views/meta.json"][0], b'{"generated_at": "after"}')
    self.assertEqual(self.objects.objects["data/run-2026-10.ndjson"][0], b'{"k": 1}\n{"k": 2}\n')
    self.assertNotIn(".lock", self.objects.objects)
    facts = last["ci_pulse"]
    self.assertEqual(last["severity"], "INFO")
    self.assertEqual(facts["execution"], "exec-1")
    self.assertEqual(facts["uploaded"]["files"], 2)
    self.assertIsNone(facts["tick_budget"])
    self.assertIsNone(facts["tick_stopped_early"])
    self.assertEqual(facts["tick_problems"], [])
    self.assertTrue(facts["lock_released"])
    self.assertFalse(facts["github_token"])
    self.assertEqual(list(self.tmp.iterdir()), [], "the work directory is removed")

  def test_a_tick_that_lost_data_is_still_saved_but_fails(self):
    code, last = self.run_job(self.fake_tick(job.EXIT_DATA_LOST))
    self.assertEqual(code, job.EXIT_DATA_LOST)
    self.assertEqual(self.objects.objects["views/meta.json"][0], b'{"generated_at": "after"}')
    self.assertEqual(last["severity"], "ERROR")

  def test_a_usage_error_saves_nothing(self):
    code, _ = self.run_job(self.fake_tick(job.EXIT_USAGE, write=False))
    self.assertEqual(code, job.EXIT_USAGE)
    self.assertEqual(self.objects.objects["views/meta.json"][0], b'{"generated_at": "before"}')
    self.assertNotIn(".lock", self.objects.objects)

  def test_a_crashed_tick_saves_nothing(self):
    code, _ = self.run_job(self.fake_tick(-9))
    self.assertEqual(code, job.EXIT_TICK_CRASHED)
    self.assertEqual(self.objects.objects["views/meta.json"][0], b'{"generated_at": "before"}')

  def test_an_unknown_exit_code_saves_nothing_and_passes_through(self):
    code, _ = self.run_job(self.fake_tick(120))
    self.assertEqual(code, 120)
    self.assertEqual(self.objects.objects["views/meta.json"][0], b'{"generated_at": "before"}')

  def test_a_stopped_tick_keeps_what_it_wrote_and_fails(self):
    code, last = self.run_job(self.fake_tick(None, timed_out=True))
    self.assertEqual(code, job.EXIT_TICK_TIMEOUT)
    self.assertEqual(self.objects.objects["views/meta.json"][0], b'{"generated_at": "after"}')
    self.assertTrue(last["ci_pulse"]["tick_timed_out"])

  def test_a_held_lock_means_nothing_is_touched(self):
    self.objects.put(".lock", b'{"owner": "other"}')
    before = dict(self.objects.objects)
    code, last = self.run_job(self.fake_tick(job.EXIT_OK))
    self.assertEqual(code, job.EXIT_LOCKED)
    self.assertEqual(self.objects.objects, before)
    self.assertEqual(last["severity"], "WARNING")
    self.assertNotIn(("list", ""), self.objects.calls)

  def test_a_failed_upload_fails_the_job_and_still_releases_the_lock(self):
    self.objects.fail_upload.add("views/meta.json")
    code, last = self.run_job(self.fake_tick(job.EXIT_OK))
    self.assertEqual(code, job.EXIT_SYNC_FAILED)
    self.assertIn("upload: SyncError", last["ci_pulse"]["error"])
    self.assertNotIn(".lock", self.objects.objects)
    # The rows went up before meta.json failed; meta still describes the old views.
    self.assertEqual(self.objects.objects["views/meta.json"][0], b'{"generated_at": "before"}')

  def test_a_failed_download_fails_the_job_before_the_tick(self):
    def broken(name: str, generation: int, path: Path) -> None:
      raise OSError("network down")

    self.objects.download = broken
    tick = mock.Mock()
    code, last = self.run_job(tick)
    self.assertEqual(code, job.EXIT_SYNC_FAILED)
    tick.assert_not_called()
    self.assertIn("download: OSError", last["ci_pulse"]["error"])
    self.assertNotIn(".lock", self.objects.objects)

  def test_no_upload_runs_the_tick_but_writes_nothing(self):
    config = job.Config(bucket="b", workdir=str(self.tmp), no_upload=True)
    code, last = self.run_job(self.fake_tick(job.EXIT_OK), config)
    self.assertEqual(code, job.EXIT_OK)
    self.assertEqual(self.objects.objects["views/meta.json"][0], b'{"generated_at": "before"}')
    self.assertIn("skipped", last["ci_pulse"]["uploaded"])

  def test_an_empty_bucket_starts_a_new_store(self):
    self.objects.objects.clear()

    def first_tick(config: job.Config, store_dir: Path, env: Any) -> job.TickResult:
      del config, env
      (store_dir / "data").mkdir(parents=True)
      (store_dir / "data/run-2026-10.ndjson").write_bytes(b"{}\n")
      (store_dir / "views").mkdir()
      (store_dir / "views/meta.json").write_bytes(b"{}")
      return job.TickResult(exit_code=0, summary="ci-metrics backfill: ok | 1 run(s)")

    code, _ = self.run_job(first_tick)
    self.assertEqual(code, job.EXIT_OK)
    self.assertEqual(self.objects.names(), ["data/run-2026-10.ndjson", "views/meta.json"])

  def test_a_bucket_that_cannot_be_reached_fails_cleanly(self):
    def unreachable(name: str, data: bytes) -> int | None:
      raise OSError("403 Forbidden")

    self.objects.create_if_absent = unreachable
    tick = mock.Mock()
    code, last = self.run_job(tick)
    self.assertEqual(code, job.EXIT_SYNC_FAILED)
    self.assertIn("lock: OSError", last["ci_pulse"]["error"])
    tick.assert_not_called()

  def test_a_lock_taken_away_during_the_tick_fails_the_job(self):
    good_tick = self.fake_tick(job.EXIT_OK)

    def tick_losing_the_lock(config: job.Config, store_dir: Path, env: Any) -> job.TickResult:
      del self.objects.objects[".lock"]
      self.objects.put(".lock", b'{"owner": "intruder"}')
      return good_tick(config, store_dir, env)

    code, last = self.run_job(tick_losing_the_lock)
    self.assertEqual(code, job.EXIT_SYNC_FAILED)
    self.assertFalse(last["ci_pulse"]["lock_released"])
    # Someone else's lock is never removed.
    self.assertEqual(json.loads(self.objects.objects[".lock"][0])["owner"], "intruder")


class MainTest(unittest.TestCase):

  def test_bad_settings_exit_2_without_opening_the_bucket(self):
    with mock.patch.dict(job.os.environ, {}, clear=True), mock.patch.object(job, "GcsObjectStore") as store_class:
      with contextlib.redirect_stdout(io.StringIO()) as printed:
        self.assertEqual(job.main([]), job.EXIT_USAGE)
    store_class.assert_not_called()
    self.assertIn("CI_PULSE_BUCKET", printed.getvalue())

  def test_good_settings_open_the_bucket_and_run(self):
    with mock.patch.dict(job.os.environ, {"CI_PULSE_BUCKET": "mesa-ci-pulse"}, clear=True):
      with mock.patch.object(job, "GcsObjectStore") as store_class, mock.patch.object(job, "run", return_value=0) as run:
        self.assertEqual(job.main([]), 0)
    store_class.assert_called_once_with("mesa-ci-pulse")
    run.assert_called_once()


class EmailReportTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.tmp = Path(tempfile.mkdtemp(prefix="ci-pulse-email-"))
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
    self.objects = MemoryObjects(clock=lambda: T0)
    self.email_cfg = email_report.EmailSettings(
        recipients=("a@example.com", "b@example.com"),
        integration="projects/p/locations/us-central1/integrations/mesa-ci-pulse-email",
        invoker_sa="invoker@p.iam.gserviceaccount.com",
        interval_seconds=27600,
    )

  def test_email_config_from_env(self):
    cfg = job.Config.from_env(
        {
            "CI_PULSE_BUCKET": "b",
            "CI_PULSE_EMAIL_RECIPIENTS": " a@example.com , b@example.com ",
            "CI_PULSE_EMAIL_INTEGRATION": "projects/p/locations/us-central1/integrations/x",
            "CI_PULSE_EMAIL_INVOKER_SA": "sa@p.iam.gserviceaccount.com",
            "CI_PULSE_EMAIL_INTERVAL_SECONDS": "28800",
            "CI_PULSE_FORCE_EMAIL": "1",
        }
    )
    self.assertTrue(cfg.email.enabled)
    self.assertEqual(cfg.email.recipients, ("a@example.com", "b@example.com"))
    self.assertEqual(cfg.email.interval_seconds, 28800)
    self.assertTrue(cfg.email.force)

  def test_should_send_digest_respects_8h_interval(self):
    now = datetime(2026, 11, 10, 9, 20, tzinfo=timezone.utc)
    self.assertTrue(email_report.should_send_digest(now, None, 27600))
    self.assertFalse(email_report.should_send_digest(now, "2026-11-10T05:20:00Z", 27600))
    self.assertTrue(email_report.should_send_digest(now, "2026-11-10T01:20:00Z", 27600))
    self.assertTrue(email_report.should_send_digest(now, "2026-11-10T05:20:00Z", 27600, force=True))

  def test_footer_names_no_fixed_send_times(self):
    # The digest goes out on the first run at least 7h40m after the last one, so its clock times drift
    # whenever the schedule changes; the footer must not promise any.
    meta = {"generated_at": "2026-11-10T09:20:00Z", "pull_requests": {}, "totals": {}}
    _, body = email_report.build_report_email(self.tmp, meta, meta, {"exit_code": 0}, fetcher=lambda url: None)
    self.assertIn("This digest goes out about every 8 hours; a failed collector run sends an alert right away.", body)
    self.assertNotRegex(body, r"\d\d:20, \d\d:20 and \d\d:20")

  def test_build_report_email_includes_changes_prs_and_insights(self):
    store = self.tmp / "store"
    (store / "views/pr").mkdir(parents=True)
    old_meta = {
        "generated_at": "2026-11-10T01:20:00Z",
        "pull_requests": {"100": {}},
        "totals": {"runs": 10, "suites": 5, "flaky": 1, "queue": 4, "workflows": 2},
    }
    new_meta = {
        "generated_at": "2026-11-10T09:20:00Z",
        "uncollected_runs": 0,
        "pull_requests": {"100": {}, "101": {}},
        "totals": {"runs": 14, "suites": 7, "flaky": 2, "queue": 6, "workflows": 3},
    }
    (store / "views/runs-2026-11.json").write_text(
        json.dumps(
            {
                "tables": {
                    "runs": {
                        "columns": ["run_id", "pr", "jobs_with_tests", "merged_at", "created_at"],
                        "rows": [[1, 101, 3, "2026-11-10T08:00:00Z", "2026-11-10T07:00:00Z"]],
                    },
                    "jobs": {
                        "columns": ["run_id", "pr", "flavor", "worker", "run_seconds", "conclusion"],
                        "rows": [
                            [1, 101, "tpu-unit", 0, 900, "success"],
                            [1, 101, "tpu-unit", 1, 300, "success"],
                            [1, 101, "tpu-unit", 2, 300, "success"],
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (store / "views/pr/101.json").write_text(
        json.dumps(
            {
                "tables": {
                    "tests": {
                        "columns": ["flavor", "worker", "classname", "name", "duration"],
                        "rows": [
                            ["tpu-unit", 0, "pkg.Mod", "test_slow", 200.0],
                            ["tpu-unit", 0, "pkg.Mod", "test_brand_new", 45.0],
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (store / "data").mkdir()
    run_row = collector_rows.RunRow(
        run_id=1,
        attempt=1,
        event="pull_request",
        created_at="2026-11-10T08:00:00Z",
        pr_number=101,
        pr_state="closed",
        pr_merged_at="2026-11-10T08:30:00Z",
    )
    (store / "data/run-2026-11.ndjson").write_text(json.dumps(collector_rows.to_json(run_row)) + "\n", encoding="utf-8")
    job_rows = [
        collector_rows.JobRow(
            job_id=10 + n,
            run_id=1,
            attempt=1,
            name=f"TPU7X Tests ({n})",
            created_at="2026-11-10T08:05:00Z",
            started_at="2026-11-10T08:15:00Z",
            completed_at="2026-11-10T08:27:00Z",
            labels=["linux-x86-tpu7x-224-4tpu"],
        )
        for n in range(5)
    ]
    (store / "data/job-2026-11.ndjson").write_text(
        "".join(json.dumps(collector_rows.to_json(row)) + "\n" for row in job_rows), encoding="utf-8"
    )
    (store / "views/flaky-2026-11.json").write_text(
        json.dumps(
            {
                "tables": {
                    "rescues": {
                        "columns": ["created_at", "job_name", "rescued", "wasted_seconds"],
                        "rows": [["2026-11-10T08:00:00Z", "tpu-unit (0)", True, 420]],
                    },
                    "rescue_tests": {"columns": [], "rows": []},
                }
            }
        ),
        encoding="utf-8",
    )
    facts = {
        "exit_code": 0,
        "seconds": 125.0,
        "execution": "exec-9",
        "bucket": "mesa-ci-pulse",
        "uploaded": {"files": 5, "unchanged": 100},
        "tick_summary": "ci-metrics tick: ok | 2 run(s)",
        "tick_budget": "4900 of 5000",
    }

    def fake_fetch(url: str) -> str | None:
      if url.endswith("per_test_baseline.json"):
        return json.dumps({"tpu-unit::pkg.Mod.test_slow": 60.0})
      return None

    subject, body = email_report.build_report_email(store, old_meta, new_meta, facts, fetcher=fake_fetch)
    self.assertIn("Dashboard updated", subject)
    self.assertIn("1 new merged PR", subject)
    self.assertIn("#101", body)
    self.assertIn("test_slow", body)
    self.assertIn("test_brand_new", body)
    self.assertIn("Worker Shard Imbalance", body)
    self.assertIn("Runner Queue Watchdog", body)
    self.assertIn("100% of 5 jobs waited 5+ min", body)
    self.assertIn('href="https://github.com/AI-Hypercomputer/maxtext/pull/101">#101</a>, merged PR)', body)
    self.assertIn("Flaky &amp; failed job retries", body)

  def test_run_sends_8h_digest_saves_email_state_and_skips_4h_tick(self):
    self.objects.put(
        "views/meta.json",
        json.dumps(
            {
                "generated_at": "2026-11-10T01:20:00Z",
                "pull_requests": {"1": {}},
                "totals": {"runs": 5},
            }
        ).encode("utf-8"),
    )
    cfg = job.Config(bucket="b", workdir=str(self.tmp), email=self.email_cfg)

    def tick_1(config: job.Config, store_dir: Path, env: Any) -> job.TickResult:
      del config, env
      (store_dir / "views/meta.json").write_text(
          json.dumps(
              {
                  "generated_at": "2026-11-10T01:22:00Z",
                  "pull_requests": {"1": {}, "2": {}},
                  "totals": {"runs": 8},
              }
          ),
          encoding="utf-8",
      )
      return job.TickResult(exit_code=0, summary="ci-metrics tick: ok")

    sent: list[tuple[str, str]] = []
    with (
        mock.patch.object(job, "run_tick", side_effect=tick_1),
        mock.patch.object(job, "send_email", side_effect=lambda s, subj, b: sent.append((subj, b)) or "exec-e1"),
        mock.patch.object(email_report, "_fetch_text", return_value=None),
        contextlib.redirect_stdout(io.StringIO()),
    ):
      code = job.run(cfg, self.objects, {"CLOUD_RUN_EXECUTION": "run-1"})
    self.assertEqual(code, 0)
    self.assertEqual(len(sent), 1)
    self.assertIn("1 new merged PR", sent[0][0])
    self.assertIn("data/email_state.json", self.objects.objects)

    # Next 4-hour tick: email_state.json is recent, so digest is skipped.
    with (
        mock.patch.object(job, "run_tick", side_effect=tick_1),
        mock.patch.object(job, "send_email", side_effect=lambda s, subj, b: sent.append((subj, b)) or "exec-e2"),
        mock.patch.object(email_report, "_fetch_text", return_value=None),
        contextlib.redirect_stdout(io.StringIO()) as out2,
    ):
      job.run(cfg, self.objects, {"CLOUD_RUN_EXECUTION": "run-2"})
    self.assertEqual(len(sent), 1)
    last2 = json.loads(out2.getvalue().strip().splitlines()[-1])
    self.assertEqual(last2["ci_pulse"]["email"]["status"], "skipped")

    # Failure tick: sends ALERT immediately even within the 8-hour window.
    def failing_tick(config: job.Config, store_dir: Path, env: Any) -> job.TickResult:
      del config, store_dir, env
      return job.TickResult(exit_code=job.EXIT_DATA_LOST, summary="ci-metrics tick: failed")

    with (
        mock.patch.object(job, "run_tick", side_effect=failing_tick),
        mock.patch.object(job, "send_email", side_effect=lambda s, subj, b: sent.append((subj, b)) or "exec-alert"),
        mock.patch.object(email_report, "_fetch_text", return_value=None),
        contextlib.redirect_stdout(io.StringIO()),
    ):
      code3 = job.run(cfg, self.objects, {"CLOUD_RUN_EXECUTION": "run-3"})
    self.assertEqual(code3, job.EXIT_DATA_LOST)
    self.assertEqual(len(sent), 2)
    self.assertIn("ALERT:", sent[1][0])


V6E = "linux-x86-ct6e-180-4tpu"
V6E_8 = "linux-x86-ct6e-180-8tpu"
TPU7X = "linux-x86-tpu7x-224-4tpu"
A100 = "linux-x86-a2-48-a100-4gpu"
N4_32 = "linux-x86-n4-32"


def iso(moment: datetime) -> str:
  """Formats a moment the way GitHub and the store write timestamps."""
  return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class QueueWatchdogTest(unittest.TestCase):
  """The email's Runner Queue Watchdog, computed from the raw job rows of every run."""

  GENERATED = datetime(2026, 11, 10, 9, 20, tzinfo=timezone.utc)

  def setUp(self) -> None:
    super().setUp()
    self.store = Path(tempfile.mkdtemp(prefix="ci-pulse-queue-"))
    self.addCleanup(shutil.rmtree, self.store, ignore_errors=True)
    self.last_id = 1000
    self.run_created: dict[int, datetime] = {}

  def _new_id(self) -> int:
    self.last_id += 1
    return self.last_id

  def _append(self, kind: str, month_of: datetime, row: Any) -> None:
    """Appends one row to the store file the collector would put it in (the run's month)."""
    path = self.store / "data" / f"{kind}-{month_of:%Y-%m}.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(collector_rows.to_json(row)) + "\n")

  def add_run(
      self,
      created: datetime,
      *,
      event: str | None = "pull_request",
      pr: int | None = None,
      merged: datetime | None = None,
      state: str = "open",
      superseded: bool = False,
      attempt: int = 1,
      run_id: int | None = None,
  ) -> int:
    """Stores one attempt of a workflow run and returns its run id."""
    run_id = run_id or self._new_id()
    row = collector_rows.RunRow(
        run_id=run_id,
        attempt=attempt,
        event=event,
        status="completed",
        conclusion="success",
        created_at=iso(created),
        superseded=superseded,
        pr_number=pr,
        pr_state=state if pr else None,
        pr_merged_at=iso(merged) if merged else None,
    )
    self._append("run", created, row)
    self.run_created[run_id] = created
    return run_id

  def add_job(
      self,
      run_id: int,
      label: str,
      ready: datetime,
      wait_minutes: float,
      *,
      attempt: int = 1,
      started: bool = True,
      with_steps: bool = True,
  ) -> None:
    """Stores one job that became ready at `ready` and waited `wait_minutes` for a runner."""
    start = ready + timedelta(seconds=round(wait_minutes * 60))  # whole seconds, like GitHub's timestamps
    end = start + timedelta(minutes=12)
    step = {
        "name": "Run tests",
        "status": "completed",
        "conclusion": "success",
        "started_at": iso(start),
        "completed_at": iso(end),
    }
    row = collector_rows.JobRow(
        job_id=self._new_id(),
        run_id=run_id,
        attempt=attempt,
        name="Execute Tests",
        status="completed",
        conclusion="success" if started else "cancelled",
        created_at=iso(ready),
        started_at=iso(start) if started else None,
        completed_at=iso(end) if started else None,
        labels=[label],
        steps=[step] if started and with_steps else [],
    )
    self._append("job", self.run_created[run_id], row)

  def watchdog(self, generated: datetime | None = None) -> list[str]:
    """Returns the watchdog group as [title, line, ...], or [] when the email has no such group."""
    meta = {"generated_at": iso(generated or self.GENERATED)}
    for title, lines in email_report.build_ci_insights(self.store, meta, fetcher=lambda url: None):
      if title.startswith("Runner Queue Watchdog"):
        return [title, *lines]
    return []

  def pool_line(self, lines: list[str], pool: str) -> str:
    found = [line for line in lines if f"`{pool}`" in line]
    self.assertEqual(len(found), 1, f"expected exactly one line for {pool} in {lines}")
    return found[0]

  def test_counts_every_kind_of_run(self):
    t = self.GENERATED - timedelta(hours=6)
    runs = [
        self.add_run(t, pr=6202),
        self.add_run(t, pr=6200, superseded=True),
        self.add_run(t, event="schedule"),
        self.add_run(t, pr=6199, merged=t, state="closed"),
        self.add_run(t, event="workflow_dispatch"),
    ]
    for run_id in runs:
      self.add_job(run_id, V6E, t, 1)
    self.add_run(t, pr=6199, merged=t, state="closed", attempt=2, run_id=runs[3])
    self.add_job(runs[3], V6E, t + timedelta(hours=1), 1, attempt=2)
    self.assertIn("of 6 jobs", self.pool_line(self.watchdog(), V6E))

  def test_share_of_jobs_that_waited_5_min_or_more(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, pr=6202)
    for wait in (0, 0, 0, 0, 0, 0, 0, 5, 6, 40):
      self.add_job(run_id, A100, t, wait)
    self.assertIn("30% of 10 jobs waited 5+ min", self.pool_line(self.watchdog(), A100))

  def test_p90_wait(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, pr=6202)
    for wait in range(1, 11):
      self.add_job(run_id, N4_32, t, wait)
    self.assertIn("p90 9.0m", self.pool_line(self.watchdog(), N4_32))

  def test_longest_wait_names_the_pr_and_its_state(self):
    t = self.GENERATED - timedelta(hours=6)
    cases = (
        (V6E, {"pr": 6202}, "(#6202, not merged)"),
        (A100, {"pr": 6199, "merged": t, "state": "closed"}, "(#6199, merged PR)"),
        (TPU7X, {"event": "schedule"}, "(a scheduled run)"),
        (N4_32, {"pr": 6100, "state": "closed"}, "(#6100, closed PR)"),
        (V6E_8, {"pr": None}, "(a PR run)"),
    )
    for pool, run_fields, _ in cases:
      self.add_job(self.add_run(t, **run_fields), pool, t, 40)
      other = self.add_run(t, pr=5000)
      for _ in range(4):
        self.add_job(other, pool, t, 1)
    lines = self.watchdog()
    for pool, _, expected in cases:
      self.assertIn(f"longest 40.0m {expected}", self.pool_line(lines, pool))

  def test_lists_every_test_pool(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, pr=6202)
    pools = (V6E, V6E_8, TPU7X, A100, N4_32)
    for pool in pools:
      for _ in range(5):
        self.add_job(run_id, pool, t, 1)
    lines = self.watchdog()
    for pool in pools:
      self.pool_line(lines, pool)

  def test_leaves_out_hosted_build_and_jobs_that_never_ran(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, pr=6202)
    for _ in range(5):
      self.add_job(run_id, V6E, t, 1)
      self.add_job(run_id, "ubuntu-latest", t, 30)
      self.add_job(run_id, "linux-x86-n4-16-buildkit", t, 30)
    self.add_job(run_id, V6E, t, 30, started=False)
    lines = self.watchdog()
    self.assertIn("): 0% of 5 jobs waited 5+ min", self.pool_line(lines, V6E))
    self.assertFalse([line for line in lines if "ubuntu-latest" in line or "n4-16-buildkit" in line])

  def test_leaves_out_pools_with_few_jobs(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, pr=6202)
    for _ in range(5):
      self.add_job(run_id, V6E, t, 1)
    for _ in range(4):
      self.add_job(run_id, TPU7X, t, 30)
    lines = self.watchdog()
    self.pool_line(lines, V6E)
    self.assertFalse([line for line in lines if f"`{TPU7X}`" in line])

  def test_no_job_rows_means_no_watchdog(self):
    self.assertEqual(self.watchdog(), [])

  def add_jobs(self, when: datetime, pool: str, total: int, long_ones: int) -> None:
    """Stores `total` jobs of one scheduled run at `when`; `long_ones` waited 10 min, the rest 1 min."""
    run_id = self.add_run(when, event="schedule")
    for n in range(total):
      self.add_job(run_id, pool, when, 10 if n < long_ones else 1)

  def test_reads_the_previous_month(self):
    t = datetime(2026, 10, 31, 20, 0, tzinfo=timezone.utc)
    self.add_jobs(t, V6E, total=5, long_ones=0)
    self.assertIn("of 5 jobs", self.pool_line(self.watchdog(datetime(2026, 11, 1, 5, 20, tzinfo=timezone.utc)), V6E))

  def test_compares_with_the_7_days_before(self):
    self.add_jobs(self.GENERATED - timedelta(days=4), A100, total=20, long_ones=2)
    self.add_jobs(self.GENERATED - timedelta(hours=6), A100, total=20, long_ones=10)
    line = self.pool_line(self.watchdog(), A100)
    self.assertIn("(7 days before: 10%)", line)
    self.assertTrue(line.startswith("WORSE: "), line)

  def test_a_small_rise_is_not_flagged(self):
    self.add_jobs(self.GENERATED - timedelta(days=4), A100, total=20, long_ones=2)
    self.add_jobs(self.GENERATED - timedelta(hours=6), A100, total=20, long_ones=3)
    line = self.pool_line(self.watchdog(), A100)
    self.assertIn("(7 days before: 10%)", line)
    self.assertFalse(line.startswith("WORSE"), line)

  def test_a_rise_of_exactly_10_points_is_flagged(self):
    self.add_jobs(self.GENERATED - timedelta(days=4), A100, total=20, long_ones=4)
    self.add_jobs(self.GENERATED - timedelta(hours=6), A100, total=20, long_ones=6)
    self.assertTrue(self.pool_line(self.watchdog(), A100).startswith("WORSE: "))

  def test_no_flag_without_enough_jobs(self):
    self.add_jobs(self.GENERATED - timedelta(days=4), A100, total=10, long_ones=0)
    self.add_jobs(self.GENERATED - timedelta(hours=6), A100, total=10, long_ones=10)
    self.assertFalse(self.pool_line(self.watchdog(), A100).startswith("WORSE"))

  def test_baseline_with_too_few_jobs(self):
    self.add_jobs(self.GENERATED - timedelta(days=4), A100, total=4, long_ones=0)
    self.add_jobs(self.GENERATED - timedelta(hours=6), A100, total=20, long_ones=10)
    self.assertIn("(7 days before: too few jobs)", self.pool_line(self.watchdog(), A100))

  def test_worse_pools_come_first(self):
    self.add_jobs(self.GENERATED - timedelta(days=4), V6E, total=20, long_ones=0)
    self.add_jobs(self.GENERATED - timedelta(hours=6), V6E, total=20, long_ones=6)
    self.add_jobs(self.GENERATED - timedelta(days=4), TPU7X, total=20, long_ones=16)
    self.add_jobs(self.GENERATED - timedelta(hours=6), TPU7X, total=20, long_ones=16)
    pool_lines = [line for line in self.watchdog()[1:] if "jobs waited" in line]
    self.assertTrue(pool_lines[0].startswith(f"WORSE: `{V6E}`"), pool_lines)
    self.assertTrue(pool_lines[1].startswith(f"`{TPU7X}`"), pool_lines)

  def write_watermark(self, newest_run: datetime) -> None:
    (self.store / "data").mkdir(parents=True, exist_ok=True)
    (self.store / "data/state.json").write_text(
        json.dumps({"v": 1, "watermark_created_at": iso(newest_run)}), encoding="utf-8"
    )

  def test_window_ends_at_the_newest_run_read(self):
    self.write_watermark(self.GENERATED - timedelta(hours=5))
    self.add_jobs(self.GENERATED - timedelta(hours=8), V6E, total=5, long_ones=0)
    self.add_jobs(self.GENERATED - timedelta(hours=2), V6E, total=1, long_ones=1)
    lines = self.watchdog()
    self.assertIn("48h to Nov 10, 04:20 UTC", lines[0])
    self.assertIn("): 0% of 5 jobs", self.pool_line(lines, V6E))
    self.assertTrue([line for line in lines if line.startswith("Newest CI run read: Nov 10, 04:20 UTC")], lines)

  def test_no_lag_note_when_the_data_is_current(self):
    self.write_watermark(self.GENERATED - timedelta(minutes=30))
    self.add_jobs(self.GENERATED - timedelta(hours=8), V6E, total=5, long_ones=0)
    lines = self.watchdog()
    self.assertIn("48h to Nov 10, 08:50 UTC", lines[0])
    self.assertFalse([line for line in lines if line.startswith("Newest CI run read")], lines)

  def test_explains_how_to_read_it(self):
    self.add_jobs(self.GENERATED - timedelta(hours=8), V6E, total=5, long_ones=0)
    lines = self.watchdog()
    self.assertTrue(lines[-2].startswith("Wait = time from a job being ready to a runner starting it."), lines)
    self.assertIn("Runs still in progress at the last update are counted next time", lines[-2])
    self.assertTrue(lines[-1].startswith("WORSE = the share of 5+ min waits is 10+ points above"), lines)
    self.assertIn("20+ jobs in both periods", lines[-1])

  def test_unreadable_rows_give_a_message_not_a_crash(self):
    (self.store / "data").mkdir(parents=True)
    (self.store / "data/job-2026-11.ndjson").write_text("this is not json\n", encoding="utf-8")
    lines = self.watchdog()
    self.assertTrue(lines and lines[0].startswith("Runner Queue Watchdog"), lines)
    found = [line for line in lines[1:] if line.startswith("Could not read the job rows: StoreError")]
    self.assertTrue(found, lines)
    self.assertNotIn(str(self.store), found[0])

  def test_rows_that_are_not_utf8_give_a_message_not_a_crash(self):
    (self.store / "data").mkdir(parents=True)
    (self.store / "data/job-2026-11.ndjson").write_bytes(b'{"kind": "job", "note": "\xff"}\n')
    lines = self.watchdog()
    self.assertTrue(
        [line for line in lines[1:] if line.startswith("Could not read the job rows: UnicodeDecodeError")], lines
    )

  def test_pr_state_comes_from_every_run_of_that_pr(self):
    t = self.GENERATED - timedelta(hours=6)
    older = self.add_run(t, pr=6300)
    newer = self.add_run(t + timedelta(hours=1), pr=6300, merged=t + timedelta(hours=2), state="closed")
    self.add_job(older, V6E, t, 40)
    for _ in range(4):
      self.add_job(newer, V6E, t + timedelta(hours=1), 1)
    self.assertIn("longest 40.0m (#6300, merged PR)", self.pool_line(self.watchdog(), V6E))

  def test_shows_pools_with_unlisted_runner_labels(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, pr=6202)
    for _ in range(5):
      self.add_job(run_id, "linux-x86-tpu8-new-4tpu", t, 1)
    line = self.pool_line(self.watchdog(), "linux-x86-tpu8-new-4tpu")
    self.assertIn("`linux-x86-tpu8-new-4tpu` (unlisted label): 0% of 5 jobs", line)

  def test_flag_follows_the_whole_percentages_shown(self):
    # 49 of 250 = 19.6% prints as 20%, and 26 of 250 = 10.4% prints as 10%: the reader sees +10 points.
    self.add_jobs(self.GENERATED - timedelta(days=4), A100, total=250, long_ones=26)
    self.add_jobs(self.GENERATED - timedelta(hours=6), A100, total=250, long_ones=49)
    line = self.pool_line(self.watchdog(), A100)
    self.assertIn("20% of 250 jobs", line)
    self.assertIn("(7 days before: 10%)", line)
    self.assertTrue(line.startswith("WORSE: "), line)

  def test_says_so_when_no_pool_has_enough_jobs(self):
    self.add_jobs(self.GENERATED - timedelta(hours=6), TPU7X, total=4, long_ones=4)
    lines = self.watchdog()
    self.assertTrue(lines and lines[0].startswith("Runner Queue Watchdog"), lines)
    self.assertIn("No test pool ran 5+ jobs in this window.", lines)

  def test_carried_over_jobs_are_not_counted_twice(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, event="schedule")
    for _ in range(5):
      self.add_job(run_id, V6E, t, 1)
    self.add_run(t, event="schedule", attempt=2, run_id=run_id)
    self.add_job(run_id, V6E, t + timedelta(hours=1), -30, attempt=2)  # started before it was created
    self.assertIn("of 5 jobs", self.pool_line(self.watchdog(), V6E))

  def test_a_job_cancelled_before_getting_a_runner_is_not_counted(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, pr=6202)
    for _ in range(5):
      self.add_job(run_id, V6E, t, 1)
    self.add_job(run_id, V6E, t, 0, with_steps=False)  # GitHub's shape: started == created, no steps
    self.assertIn("of 5 jobs", self.pool_line(self.watchdog(), V6E))

  def test_window_start_belongs_to_the_window_and_one_second_earlier_to_the_baseline(self):
    start = self.GENERATED - timedelta(hours=48)
    self.add_jobs(start, V6E, total=5, long_ones=0)
    self.add_jobs(start - timedelta(seconds=1), V6E, total=5, long_ones=5)
    line = self.pool_line(self.watchdog(), V6E)
    self.assertIn("): 0% of 5 jobs", line)
    self.assertIn("(7 days before: 100%)", line)

  def test_a_wait_just_under_5_minutes_is_not_long(self):
    t = self.GENERATED - timedelta(hours=6)
    run_id = self.add_run(t, pr=6202)
    for _ in range(5):
      self.add_job(run_id, A100, t, 299 / 60)
    self.assertIn("): 0% of 5 jobs waited 5+ min", self.pool_line(self.watchdog(), A100))

  def test_no_flag_when_either_period_has_19_jobs(self):
    for before_total, now_total in ((19, 20), (20, 19)):
      with self.subTest(before=before_total, now=now_total):
        shutil.rmtree(self.store / "data", ignore_errors=True)
        self.add_jobs(self.GENERATED - timedelta(days=4), A100, total=before_total, long_ones=0)
        self.add_jobs(self.GENERATED - timedelta(hours=6), A100, total=now_total, long_ones=now_total)
        self.assertFalse(self.pool_line(self.watchdog(), A100).startswith("WORSE"))

  def test_reads_december_for_a_january_email(self):
    t = datetime(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
    self.add_jobs(t, V6E, total=5, long_ones=0)
    self.assertIn("of 5 jobs", self.pool_line(self.watchdog(datetime(2027, 1, 1, 5, 20, tzinfo=timezone.utc)), V6E))

  def test_no_lag_note_at_exactly_one_hour(self):
    self.write_watermark(self.GENERATED - timedelta(hours=1))
    self.add_jobs(self.GENERATED - timedelta(hours=8), V6E, total=5, long_ones=0)
    self.assertFalse([line for line in self.watchdog() if line.startswith("Newest CI run read")])

  def test_a_run_without_an_event_is_an_unknown_run(self):
    t = self.GENERATED - timedelta(hours=6)
    self.add_job(self.add_run(t, event=None), V6E, t, 40)
    other = self.add_run(t, pr=5000)
    for _ in range(4):
      self.add_job(other, V6E, t, 1)
    self.assertIn("longest 40.0m (an unknown run)", self.pool_line(self.watchdog(), V6E))


if __name__ == "__main__":
  unittest.main()
