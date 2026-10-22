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

from cloudrun import job  # pylint: disable=wrong-import-position
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


if __name__ == "__main__":
  unittest.main()
