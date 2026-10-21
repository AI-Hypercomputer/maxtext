# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for tools/dev/calculate_workers.py."""

import json
import os
import subprocess
import sys
from tools.dev.calculate_workers import CPU_XDIST_PARALLELISM
from tools.dev.calculate_workers import DEFAULT_WORKERS
from tools.dev.calculate_workers import LIGHTWEIGHT_SUITE_MINUTES_THRESHOLD
from tools.dev.calculate_workers import MAX_CPU_WORKERS
from tools.dev.calculate_workers import MAX_GPU_WORKERS
from tools.dev.calculate_workers import MAX_TPU_WORKERS
from tools.dev.calculate_workers import TARGET_MINUTES_PER_WORKER
from tools.dev.calculate_workers import calculate_workers
from tools.dev.calculate_workers import get_max_workers_for_flavor


def test_constants():
  """Tests that module constants have expected defaults."""
  assert MAX_CPU_WORKERS == 4
  assert MAX_TPU_WORKERS == 3
  assert MAX_GPU_WORKERS == 3
  assert DEFAULT_WORKERS == 1
  assert LIGHTWEIGHT_SUITE_MINUTES_THRESHOLD == 16.0
  assert TARGET_MINUTES_PER_WORKER == 8.0
  assert CPU_XDIST_PARALLELISM == 16


def test_get_max_workers_for_flavor():
  """Tests device-specific maximum worker safety caps."""
  assert get_max_workers_for_flavor("cpu-unit") == 4
  assert get_max_workers_for_flavor("cpu-integration") == 4
  assert get_max_workers_for_flavor("cpu-post-training-unit") == 4
  assert get_max_workers_for_flavor("tpu-unit") == 3
  assert get_max_workers_for_flavor("tpu-integration") == 3
  assert get_max_workers_for_flavor("tpu7x-unit") == 3
  assert get_max_workers_for_flavor("gpu-unit") == 3
  assert get_max_workers_for_flavor("gpu-integration") == 3
  assert get_max_workers_for_flavor("unknown-flavor") == 1


def test_calculate_workers_no_baseline():
  """Tests fallback behavior when baseline data is not provided."""
  workers, groups = calculate_workers("cpu-unit", None)
  assert workers == 4
  assert groups == [1, 2, 3, 4]

  workers, groups = calculate_workers("tpu-unit", None)
  assert workers == 3
  assert groups == [1, 2, 3]

  workers, groups = calculate_workers("unknown-flavor", None)
  assert workers == 1
  assert groups == [1]


def test_calculate_workers_zero_matching_tests():
  """Tests behavior when baseline exists but has 0 matching tests for flavor."""
  baseline = {
      "tpu-unit::tests.unit.test_a": 10.0,
  }
  workers, groups = calculate_workers("cpu-unit", baseline)
  assert workers == 1
  assert groups == [1]


def test_calculate_workers_heavy_suite():
  """Tests worker allocation for heavy suites (>= 16 effective mins)."""
  baseline = {f"cpu-unit::test_{i}": 600.0 for i in range(30)}  # 300 CPU-mins -> 18.75 eff mins
  workers, groups = calculate_workers("cpu-unit", baseline)
  assert workers == 4
  assert groups == [1, 2, 3, 4]

  baseline_tpu = {f"tpu-unit::test_{i}": 60.0 for i in range(20)}  # 20 mins total
  workers, groups = calculate_workers("tpu-unit", baseline_tpu)
  assert workers == 3
  assert groups == [1, 2, 3]


def test_calculate_workers_lightweight_suite():
  """Tests worker right-sizing for lightweight suites and CPU xdist scaling."""
  # 3 GPU tests taking 120s each = 6.0 mins total -> 1 worker
  baseline = {f"gpu-integration::test_{i}": 120.0 for i in range(3)}
  assert calculate_workers("gpu-integration", baseline) == (1, [1])

  # 6 GPU tests taking 100s each = 10.0 mins total -> ceil(10.0 / 8.0) = 2 workers
  baseline_medium = {f"gpu-integration::test_{i}": 100.0 for i in range(6)}
  assert calculate_workers("gpu-integration", baseline_medium) == (2, [1, 2])

  # 20 CPU tests taking 60s each = 20.0 CPU-mins -> 1.25 eff mins under xdist -> 1 worker
  baseline_cpu = {f"cpu-post-training-unit::test_{i}": 60.0 for i in range(20)}
  assert calculate_workers("cpu-post-training-unit", baseline_cpu) == (1, [1])

  # Single slow CPU test (600s = 10.0m floor) bounds effective duration -> 2 workers
  baseline_cpu["cpu-post-training-unit::test_slow"] = 600.0
  assert calculate_workers("cpu-post-training-unit", baseline_cpu) == (2, [1, 2])


def test_calculate_workers_fewer_tests_than_max_workers():
  """Tests that worker count never exceeds the number of tests."""
  # 2 tests taking 1000s each = 16.67m max single test -> max_workers is 4, but only 2 tests!
  baseline = {
      "cpu-unit::test_heavy_1": 1000.0,
      "cpu-unit::test_heavy_2": 1000.0,
  }
  workers, groups = calculate_workers("cpu-unit", baseline)
  assert workers == 2
  assert groups == [1, 2]


def test_main_cli(tmp_path):
  """Tests the CLI entry point stdout output."""
  baseline_file = tmp_path / "baseline.json"
  baseline_data = {
      "tpu-unit::test_1": 600.0,
      "tpu-unit::test_2": 600.0,
  }
  baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

  cmd = [
      sys.executable,
      "tools/dev/calculate_workers.py",
      "--flavor",
      "tpu-unit",
      "--baseline",
      str(baseline_file),
  ]
  env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

  res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
  assert res.returncode == 0
  lines = res.stdout.strip().splitlines()
  assert "total_workers=2" in lines
  assert "worker_groups=[1, 2]" in lines


def test_filters_stale_and_scheduled_entries(tmp_path):
  """Excludes deleted, skipped, mismatched-marker, and (on PR runs) scheduled_only entries."""
  unit_dir = tmp_path / "tests" / "unit"
  unit_dir.mkdir(parents=True)
  (unit_dir / "s_test.py").write_text(
      "import pytest\n"
      "class G:\n"
      "  @pytest.mark.gpu_only\n  def test_ok(self): pass\n"
      "  @pytest.mark.gpu_only\n  @pytest.mark.skip\n  def test_skip(self): pass\n"
      "  @pytest.mark.gpu_only\n  @pytest.mark.scheduled_only\n  def test_sched(self): pass\n"
      "class C:\n  def test_cpu(self): pass\n"
      "class T:\n  @pytest.mark.tpu_only\n  def test_tpu(self): pass\n",
      encoding="utf-8",
  )
  baseline = {
      "gpu-unit::tests.unit.s_test.G.test_ok": 40.0,
      "gpu-unit::tests.unit.s_test.G.test_skip": 600.0,
      "gpu-unit::tests.unit.s_test.G.test_deleted": 600.0,
      "gpu-unit::tests.unit.s_test.G.test_sched": 600.0,
      "gpu-unit::tests.unit.s_test.C.test_cpu": 600.0,
      "gpu-unit::tests.unit.s_test.T.test_tpu": 600.0,
      "gpu-unit::tests.unit.missing_test.M.test_x": 600.0,
  }
  assert calculate_workers("gpu-unit", baseline, repo_root=str(tmp_path), is_scheduled_run=False) == (1, [1])
  assert calculate_workers("gpu-unit", baseline, repo_root=str(tmp_path), is_scheduled_run=True) == (2, [1, 2])


def test_split_classes_and_prefix_collisions(tmp_path):
  """Matches moved methods and @parameterized_class without matching deleted prefix-sharing names."""
  unit_dir = tmp_path / "tests" / "unit"
  unit_dir.mkdir(parents=True)
  (unit_dir / "m_test.py").write_text(
      "from parameterized import parameterized_class\nimport pytest\n"
      "class A:\n  @pytest.mark.tpu_only\n  def test_x(self): pass\n"
      "class B:\n  @pytest.mark.tpu_only\n  def test_x_moved(self): pass\n"
      "@parameterized_class([{'k': 1}])\nclass P:\n  @pytest.mark.tpu_only\n  def test_p(self): pass\n",
      encoding="utf-8",
  )
  baseline = {
      "tpu-unit::tests.unit.m_test.A.test_x": 400.0,
      "tpu-unit::tests.unit.m_test.A.test_x_moved": 400.0,
      "tpu-unit::tests.unit.m_test.A.test_x_deleted": 600.0,
      "tpu-unit::tests.unit.m_test.P_0.test_p": 400.0,
  }
  assert calculate_workers("tpu-unit", baseline, repo_root=str(tmp_path)) == (3, [1, 2, 3])
  deleted_only = {
      "tpu-unit::tests.unit.m_test.A.test_x_deleted": 600.0,
      "tpu-unit::tests.unit.m_test.A.test_x_deleted_2": 600.0,
  }
  assert calculate_workers("tpu-unit", deleted_only, repo_root=str(tmp_path)) == (1, [1])
