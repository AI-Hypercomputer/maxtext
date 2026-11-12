# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests helpers of the checkpoint load benchmark harness, benchmark_ckpt_load_time.py."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

# pylint: disable=protected-access

_HARNESS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_ckpt_load_time.py")
_spec = importlib.util.spec_from_file_location("benchmark_ckpt_load_time", _HARNESS_PATH)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


class PersistWorkerUploadMaxTest(unittest.TestCase):
  """The worker writes the phase result JSON before the parent has parsed the upload times from the worker log."""

  RESULT = {"hf_to_gcs_s": 41.5, "safetensors_and_convert_s": 97.25}

  def setUp(self):
    super().setUp()
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    self.result_json = os.path.join(tmp_dir, "safetensors_dynamic_result.json")
    with open(self.result_json, "w", encoding="utf-8") as f:
      json.dump(self.RESULT, f, indent=2)

  def _read_text(self) -> str:
    with open(self.result_json, "r", encoding="utf-8") as f:
      return f.read()

  def test_writes_the_slowest_upload_into_the_result_json(self):
    harness._persist_worker_upload_max_s(self.result_json, [3.5, 9.25, 4.0])
    self.assertEqual(json.loads(self._read_text()), {**self.RESULT, "worker_upload_max_s": 9.25})

  def test_returns_the_result_it_persisted(self):
    data = harness._persist_worker_upload_max_s(self.result_json, [3.5, 9.25, 4.0])
    self.assertEqual(data, json.loads(self._read_text()))

  def test_leaves_the_result_json_untouched_without_uploads(self):
    before = self._read_text()
    data = harness._persist_worker_upload_max_s(self.result_json, [])
    self.assertEqual(data, self.RESULT)
    self.assertEqual(self._read_text(), before)


class OncePerProcessTest(unittest.TestCase):
  """Each worker runs SITE_CUSTOMIZE_CONTENT twice: as sitecustomize.py at interpreter start and again from main()."""

  TEST_SENTINEL = "_benchmark_ckpt_load_time_test_sentinel"

  def setUp(self):
    super().setUp()
    # The sentinels are attributes of `sys`: clear them for the test and restore them afterwards.
    for sentinel in (self.TEST_SENTINEL, harness._SITE_CUSTOMIZE_SENTINEL):
      saved = vars(sys).pop(sentinel, None)
      self.addCleanup(self._restore_sys_attr, sentinel, saved)

  @staticmethod
  def _restore_sys_attr(name: str, value) -> None:
    vars(sys).pop(name, None)
    if value is not None:
      setattr(sys, name, value)

  def test_second_run_does_not_patch_again(self):
    target = types.SimpleNamespace(load=lambda: "load")
    code = harness._once_per_process(
        "_orig_load = target.load\ntarget.load = lambda: 'patched ' + _orig_load()\n", self.TEST_SENTINEL
    )
    # Separate globals, like the sitecustomize module and then the harness module.
    exec(code, {"target": target})  # pylint: disable=exec-used
    exec(code, {"target": target})  # pylint: disable=exec-used
    self.assertEqual(target.load(), "patched load")

  def test_site_customize_content_does_nothing_once_applied(self):
    setattr(sys, harness._SITE_CUSTOMIZE_SENTINEL, True)
    namespace = {}
    exec(harness.SITE_CUSTOMIZE_CONTENT, namespace)  # pylint: disable=exec-used
    self.assertEqual(set(namespace) - {"__builtins__", "sys"}, set())


if __name__ == "__main__":
  unittest.main()
