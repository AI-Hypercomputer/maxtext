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

"""Tests TPU verification and NOT-TPU failure handling in benchmark_ckpt_load_time.py."""

import importlib.util
import json
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

# pylint: disable=protected-access

_HARNESS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_ckpt_load_time.py")
_spec = importlib.util.spec_from_file_location("benchmark_ckpt_load_time", _HARNESS_PATH)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


class CheckTpuBackendTest(unittest.TestCase):
  """Tests the pure check_tpu_backend decision helper."""

  def test_passes_when_tpu_and_device_count_matches(self):
    self.assertEqual(harness.check_tpu_backend("tpu", "tpu", 8, 8), (True, ""))
    self.assertEqual(harness.check_tpu_backend("tpu", "tpu", 256, 256), (True, ""))

  def test_fails_when_hardware_tpu_falls_back_to_cpu(self):
    ok, reason = harness.check_tpu_backend("tpu", "cpu", 1, 8)
    self.assertFalse(ok)
    self.assertIn("cpu", reason)

  def test_fails_when_tpu_device_count_differs(self):
    ok, reason = harness.check_tpu_backend("tpu", "tpu", 4, 8)
    self.assertFalse(ok)
    self.assertIn("8", reason)
    self.assertIn("4", reason)

  def test_allows_local_cpu_simulation_when_hardware_cpu(self):
    self.assertEqual(harness.check_tpu_backend("cpu", "cpu", 4, 8), (True, ""))
    self.assertEqual(harness.check_tpu_backend("auto", "cpu", 1, 8), (True, ""))


class WorkerTpuCheckTest(unittest.TestCase):
  """Tests worker TPU metadata capture and parent NOT-TPU FAILED record creation."""

  def _fake_jax(self, backend: str, devices_list: list, local_count: int):
    return types.SimpleNamespace(
        default_backend=lambda: backend,
        device_count=lambda: len(devices_list),
        local_device_count=lambda: local_count,
        devices=lambda: devices_list,
    )

  def test_records_tpu_metadata_on_success(self):
    args = types.SimpleNamespace(hardware="tpu", num_slices=1, devices_per_slice=8)
    devs = [types.SimpleNamespace(device_kind="TPU v6e") for _ in range(8)]
    info = harness._verify_and_measure_jax_backend(args, self._fake_jax("tpu", devs, 8))
    self.assertEqual(
        info,
        {
            "jax_backend_platform": "tpu",
            "tpu_device_kind": "TPU v6e",
            "total_tpu_devices_measured": 8,
            "tpu_devices_per_host_measured": 8,
        },
    )

  def test_fails_with_marker_before_loading_on_mismatch(self):
    args = types.SimpleNamespace(hardware="tpu", num_slices=1, devices_per_slice=8, slice_id=0)
    devs = [types.SimpleNamespace(device_kind="cpu")]
    with self.assertRaisesRegex(RuntimeError, "TPU check failed"):
      harness._verify_and_measure_jax_backend(args, self._fake_jax("cpu", devs, 1))

  def test_parent_writes_not_tpu_failed_record_with_no_timings(self):
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)

    recorded_cmds: list[list[str]] = []

    class _FakeProc:
      """Minimal Popen double emitting a [NOT-TPU] failure marker."""

      def __init__(self, cmd, *a, **kw):
        del a, kw
        recorded_cmds.append(cmd)
        self.pid = 12345
        self.returncode = 1
        self.stdout = iter(["[NOT-TPU] backend=cpu device_count=1 expected=8: Expected JAX backend 'tpu', got 'cpu'\n"])

      def __enter__(self):
        return self

      def __exit__(self, *exc):
        return False

      def poll(self):
        return 1

      def wait(self, timeout=None):
        del timeout
        return 1

    with mock.patch.object(harness.subprocess, "Popen", _FakeProc):
      rec = harness.invoke_worker_subprocess(
          phase="safetensors_dynamic",
          model="qwen3-0.6b",
          scan_mode="scanned",
          gcs_run_dir=os.path.join(tmp_dir, "gcs"),
          local_work_dir=tmp_dir,
          hardware="tpu",
          devices_per_slice=8,
          allow_step_failure=True,
      )

    self.assertEqual(rec["status"], "FAILED")
    self.assertEqual(rec["failure_category"], "NOT-TPU")
    self.assertEqual(rec["jax_backend_platform"], "cpu")
    self.assertEqual(rec["total_tpu_devices_measured"], 1)
    for timing_key in (
        "hf_to_gcs_s",
        "hf_to_local_s",
        "convert_to_orbax_s",
        "load_orbax_s",
        "safetensors_and_convert_s",
        "worker_upload_max_s",
    ):
      self.assertNotIn(timing_key, rec)

    result_path = os.path.join(tmp_dir, "logs", "qwen3-0.6b", "scanned", "safetensors_dynamic_result.json")
    with open(result_path, "r", encoding="utf-8") as f:
      persisted = json.load(f)
    self.assertEqual(persisted["failure_category"], "NOT-TPU")
    self.assertEqual(recorded_cmds[0][recorded_cmds[0].index("--devices-per-slice") + 1], "8")


if __name__ == "__main__":
  unittest.main()
