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

"""Live check of `GcsObjectStore` against a real bucket; skipped unless CI_PULSE_LIVE_BUCKET is set.

    CI_PULSE_LIVE_BUCKET=mesa-ci-pulse python3 cloudrun/tests/gcs_live_test.py

It works under its own prefix, `live-test-<pid>/`, and deletes what it wrote, so it never touches
the store. It needs google-cloud-storage and credentials that may write the bucket.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_APP_DIR = str(Path(__file__).resolve().parents[2])
if _APP_DIR not in sys.path:
  sys.path.insert(0, _APP_DIR)

from cloudrun import job  # pylint: disable=wrong-import-position

BUCKET = os.environ.get("CI_PULSE_LIVE_BUCKET", "")


@unittest.skipUnless(BUCKET, "set CI_PULSE_LIVE_BUCKET to run against a real bucket")
class GcsObjectStoreLiveTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.objects = job.GcsObjectStore(BUCKET)
    self.prefix = f"live-test-{os.getpid()}/"
    self.tmp = Path(tempfile.mkdtemp(prefix="ci-pulse-live-"))
    self.addCleanup(self.clean)

  def clean(self) -> None:
    for name, info in self.objects.list(self.prefix).items():
      self.objects.delete(name, info.generation)
    shutil.rmtree(self.tmp, ignore_errors=True)

  def test_the_lock_primitives(self):
    name = f"{self.prefix}.lock"
    generation = self.objects.create_if_absent(name, b'{"owner": "live-test"}')
    self.assertIsInstance(generation, int)
    self.assertIsNone(self.objects.create_if_absent(name, b'{"owner": "second"}'))
    info = self.objects.info(name)
    self.assertEqual(info.generation, generation)
    self.assertIsNotNone(info.created)
    self.assertFalse(self.objects.delete(name, generation + 1))
    self.assertTrue(self.objects.delete(name, generation))
    self.assertIsNone(self.objects.info(name))
    self.assertFalse(self.objects.delete(name, generation))

  def test_upload_download_and_preconditions(self):
    source = self.tmp / "meta.json"
    source.write_bytes(b'{"generated_at": "live"}')
    name = f"{self.prefix}views/meta.json"
    self.objects.upload(source, name, 0, "application/json")
    listed = self.objects.list(self.prefix)
    self.assertEqual(list(listed), [name])
    self.assertEqual(listed[name].md5, job.md5_base64(source))
    self.assertEqual(listed[name].size, source.stat().st_size)

    with self.assertRaises(job.SyncError):
      self.objects.upload(source, name, 0, "application/json")
    with self.assertRaises(job.SyncError):
      self.objects.upload(source, name, listed[name].generation + 1, "application/json")

    source.write_bytes(b'{"generated_at": "live again"}')
    self.objects.upload(source, name, listed[name].generation, "application/json")
    current = self.objects.info(name)
    self.assertNotEqual(current.generation, listed[name].generation)

    target = self.tmp / "copy" / "meta.json"
    self.objects.download(name, current.generation, target)
    self.assertEqual(target.read_bytes(), b'{"generated_at": "live again"}')

  def test_a_whole_sync_round_trip(self):
    store = self.tmp / "store"
    (store / "data").mkdir(parents=True)
    (store / "data" / "run-2026-10.ndjson").write_bytes(b'{"k": 1}\n')
    (store / "views").mkdir()
    (store / "views" / "meta.json").write_bytes(b"{}")
    report = job.upload_store(self.objects, self.prefix, store, {})
    self.assertEqual(report.files, 2)

    fresh = self.tmp / "fresh"
    remote, downloaded = job.download_store(self.objects, self.prefix, fresh)
    self.assertEqual(sorted(remote), ["data/run-2026-10.ndjson", "views/meta.json"])
    self.assertEqual(downloaded.files, 2)
    (fresh / "views" / "meta.json").unlink()
    report = job.upload_store(self.objects, self.prefix, fresh, remote)
    self.assertEqual((report.files, report.unchanged, report.deleted), (0, 1, 1))
    self.assertEqual(sorted(self.objects.list(self.prefix)), [f"{self.prefix}data/run-2026-10.ndjson"])


if __name__ == "__main__":
  unittest.main()
