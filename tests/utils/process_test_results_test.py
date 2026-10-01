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

"""Unit tests for process_test_results.py."""

import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import pytest

spec = importlib.util.spec_from_file_location("process_test_results", "tests/utils/process_test_results.py")
process_test_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_test_results)

extract_job_name = process_test_results.extract_job_name
process_testcase = process_test_results.process_testcase
extract_shard_layout = process_test_results.extract_shard_layout
find_missing_shards = process_test_results.find_missing_shards


@pytest.mark.cpu_only
class ProcessTestResultsTest(unittest.TestCase):

  def test_extract_job_name(self):
    self.assertEqual(extract_job_name("test-results-gpu-unit-1.xml"), "gpu-unit")
    self.assertEqual(extract_job_name("test-results-tpu-unit-1.xml"), "tpu-unit")
    self.assertEqual(
        extract_job_name("test-results-cpu-torch-reference-1.xml"),
        "cpu-torch-reference",
    )
    self.assertEqual(
        extract_job_name("test-results-tpu7x-post-training-unit-2.xml"),
        "tpu7x-post-training-unit",
    )
    self.assertEqual(extract_job_name("test-results-cpu-1.xml"), "cpu")
    self.assertEqual(extract_job_name("random.xml"), "unknown")

  def test_extract_job_name_drops_shard_layout_suffix(self):
    self.assertEqual(extract_job_name("test-results-tpu-unit-1of3.xml"), "tpu-unit")
    self.assertEqual(
        extract_job_name("test-results-tpu7x-post-training-unit-2of2.xml"),
        "tpu7x-post-training-unit",
    )

  def test_extract_shard_layout(self):
    self.assertEqual(extract_shard_layout("test-results-tpu-unit-1of3.xml"), (1, 3))
    self.assertEqual(extract_shard_layout("results/test-results-cpu-post-training-unit-2of2.xml"), (2, 2))
    self.assertEqual(extract_shard_layout("test-results-tpu-unit-2.xml"), (2, None))
    self.assertIsNone(extract_shard_layout("test-results-decoupled-targeted.xml"))
    self.assertIsNone(extract_shard_layout("random.xml"))

  def test_process_testcase_flavor_isolation(self):
    """Verifies that different test flavors have isolated baselines and do not trigger false regressions."""
    baseline_data = {
        "cpu-unit::tests.unit.qk_clip_test.QKClipMLATest.test_mla_dot_product_integration": 0.94,
        "gpu-unit::tests.unit.qk_clip_test.QKClipMLATest.test_mla_dot_product_integration": 15.0,
    }
    new_baseline = {}

    testcase_xml = ET.Element(
        "testcase",
        {
            "name": "test_mla_dot_product_integration",
            "classname": "tests.unit.qk_clip_test.QKClipMLATest",
            "time": "15.99",
        },
    )

    # Process under gpu-unit flavor. 15.99s compared against 15.0s baseline for gpu-unit should NOT trigger regression.
    failed = process_testcase(
        testcase_xml,
        "test-results-gpu-unit-1.xml",
        "gpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertIs(failed, False)
    self.assertIn(
        "gpu-unit::tests.unit.qk_clip_test.QKClipMLATest.test_mla_dot_product_integration",
        new_baseline,
    )
    self.assertEqual(
        new_baseline["gpu-unit::tests.unit.qk_clip_test.QKClipMLATest.test_mla_dot_product_integration"],
        15.99,
    )

  def test_process_testcase_regression_enforced_for_non_excluded_module(self):
    """Verifies that a genuine regression in a non-excluded module returns True."""
    baseline_data = {
        "gpu-unit::tests.unit.slow_test.SlowTest.test_slow": 1.0,
    }
    new_baseline = {}

    testcase_xml = ET.Element(
        "testcase",
        {
            "name": "test_slow",
            "classname": "tests.unit.slow_test.SlowTest",
            "time": "20.0",
        },
    )

    failed = process_testcase(
        testcase_xml,
        "test-results-gpu-unit-1.xml",
        "gpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertIs(failed, True)

  def test_process_testcase_update_baseline_false_checks_regression_without_recording(self):
    """A flavor with a missing shard still gets its regression check, but its baseline entry stays untouched."""
    baseline_data = {
        "gpu-unit::tests.unit.slow_test.SlowTest.test_slow": 1.0,
    }
    new_baseline = {}

    testcase_xml = ET.Element(
        "testcase",
        {
            "name": "test_slow",
            "classname": "tests.unit.slow_test.SlowTest",
            "time": "20.0",
        },
    )

    failed = process_testcase(
        testcase_xml,
        "test-results-gpu-unit-1of2.xml",
        "gpu-unit",
        baseline_data,
        new_baseline,
        update_baseline=False,
    )
    self.assertIs(failed, True)
    self.assertEqual(new_baseline, {})

  def test_process_testcase_regression_warn_only_for_excluded_module(self):
    """Verifies that a regression in an excluded module (moe_test) returns False."""
    baseline_data = {
        "tpu-unit::tests.unit.moe_test.RoutedMoeTest.test_ragged_sort": 25.0,
    }
    new_baseline = {}

    testcase_xml = ET.Element(
        "testcase",
        {
            "name": "test_ragged_sort",
            "classname": "tests.unit.moe_test.RoutedMoeTest",
            "time": "130.0",
        },
    )

    failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertIs(failed, False)

  def test_process_testcase_regression_warn_only_for_attention_test(self):
    """Verifies that a regression in attention_test (excluded) returns False."""
    baseline_data = {
        "tpu-unit::tests.unit.attention_test.AttentionTest.test_ring_cp": 5.0,
    }
    new_baseline = {}

    testcase_xml = ET.Element(
        "testcase",
        {
            "name": "test_ring_cp",
            "classname": "tests.unit.attention_test.AttentionTest",
            "time": "40.0",
        },
    )

    failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertIs(failed, False)

  def test_process_testcase_regression_warn_only_for_function_style_test_in_excluded_module(self):
    """Verifies that a module-level test function in an excluded module is warn-only too.

    For a plain test function the junit classname is the module itself, with no class segment.
    """
    baseline_data = {
        "tpu-unit::tests.unit.moe_test.test_ragged_sort_grad": 25.0,
    }
    new_baseline = {}

    testcase_xml = ET.Element(
        "testcase",
        {
            "name": "test_ragged_sort_grad",
            "classname": "tests.unit.moe_test",
            "time": "130.0",
        },
    )

    failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertIs(failed, False)

  def test_process_testcase_regression_enforced_for_module_whose_name_merely_starts_like_an_excluded_one(self):
    """`tests.unit.moe_test_extra` is not `tests.unit.moe_test`; the match must compare whole dotted names."""
    baseline_data = {
        "tpu-unit::tests.unit.moe_test_extra.ExtraTest.test_slow": 1.0,
    }
    new_baseline = {}

    testcase_xml = ET.Element(
        "testcase",
        {
            "name": "test_slow",
            "classname": "tests.unit.moe_test_extra.ExtraTest",
            "time": "20.0",
        },
    )

    failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertIs(failed, True)

  def test_process_testcase_skipped_returns_false(self):
    """Verifies that skipped tests return False and do not update baseline."""
    testcase_xml = ET.Element(
        "testcase",
        {"name": "test_skip", "classname": "tests.unit.foo.Bar", "time": "0.0"},
    )
    ET.SubElement(testcase_xml, "skipped")
    new_baseline = {}

    failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        {},
        new_baseline,
    )
    self.assertIs(failed, False)
    self.assertEqual(new_baseline, {})

  def test_process_testcase_failed_returns_false(self):
    """Verifies that failed tests return False and do not update baseline."""
    testcase_xml = ET.Element(
        "testcase",
        {"name": "test_fail", "classname": "tests.unit.foo.Bar", "time": "5.0"},
    )
    ET.SubElement(testcase_xml, "failure")
    new_baseline = {}

    failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        {},
        new_baseline,
    )
    self.assertIs(failed, False)
    self.assertEqual(new_baseline, {})

  def _write_suite(self, path, cases):
    """Writes a junit XML file with one <testcase> per (name, classname, time) tuple."""
    suite = ET.Element("testsuite")
    for name, classname, time in cases:
      ET.SubElement(suite, "testcase", {"name": name, "classname": classname, "time": time})
    ET.ElementTree(suite).write(path)

  def _run_main(self, argv):
    """Runs main() with the given arguments and returns its exit code."""
    with mock.patch("sys.argv", ["process_test_results.py"] + argv):
      with self.assertRaises(SystemExit) as cm:
        process_test_results.main()
    return cm.exception.code

  def test_main_adding_new_test_in_module_does_not_trigger_module_regression(self):
    """Verifies that adding new tests to a module does not trigger a per-module regression alert."""
    with tempfile.TemporaryDirectory() as tmpdir:
      self._write_suite(
          os.path.join(tmpdir, "test-results-tpu-unit-1.xml"),
          [
              ("test_existing", "tests.unit.some_test.SomeTest", "20.0"),
              ("test_newly_added", "tests.unit.some_test.SomeTest", "30.0"),
          ],
      )
      baseline_path = os.path.join(tmpdir, "baseline.json")
      with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tpu-unit::tests.unit.some_test.SomeTest.test_existing": 20.0,
                "tpu-unit::MODULE::tests.unit.some_test": 20.0,
            },
            f,
        )
      save_baseline_path = os.path.join(tmpdir, "new_baseline.json")

      code = self._run_main([tmpdir, "--baseline", baseline_path, "--save-baseline", save_baseline_path])

      self.assertEqual(code, 0)
      with open(save_baseline_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
      self.assertNotIn("tpu-unit::MODULE::tests.unit.some_test", saved)

  def test_main_keeps_old_entries_and_skips_dashboard_for_a_flavor_with_a_missing_shard(self):
    """Only the flavor with the missing shard is held back; complete flavors are updated as usual."""
    with tempfile.TemporaryDirectory() as tmpdir:
      self._write_suite(
          os.path.join(tmpdir, "test-results-tpu-unit-1of2.xml"),
          [("test_a", "tests.unit.some_test.SomeTest", "25.0")],
      )
      self._write_suite(
          os.path.join(tmpdir, "test-results-gpu-unit-1of1.xml"),
          [("test_a", "tests.unit.some_test.SomeTest", "7.0")],
      )
      baseline_path = os.path.join(tmpdir, "baseline.json")
      with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tpu-unit::tests.unit.some_test.SomeTest.test_a": 20.0,
                "tpu-unit::tests.unit.some_test.SomeTest.test_b": 30.0,
                "gpu-unit::tests.unit.some_test.SomeTest.test_a": 5.0,
            },
            f,
        )
      save_baseline_path = os.path.join(tmpdir, "new_baseline.json")
      benchmark_path = os.path.join(tmpdir, "benchmark.json")

      code = self._run_main(
          [
              tmpdir,
              "--baseline",
              baseline_path,
              "--save-baseline",
              save_baseline_path,
              "--output-benchmark",
              benchmark_path,
          ]
      )

      self.assertEqual(code, 0)
      with open(save_baseline_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
      self.assertEqual(saved["tpu-unit::tests.unit.some_test.SomeTest.test_a"], 20.0)
      self.assertEqual(saved["tpu-unit::tests.unit.some_test.SomeTest.test_b"], 30.0)
      self.assertEqual(saved["gpu-unit::tests.unit.some_test.SomeTest.test_a"], 7.0)
      with open(benchmark_path, "r", encoding="utf-8") as f:
        names = [b["name"] for b in json.load(f)]
      self.assertIn("Total GPU-UNIT Tests Duration", names)
      self.assertNotIn("Total TPU-UNIT Tests Duration", names)

  def test_main_updates_entries_when_all_shards_are_present(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self._write_suite(
          os.path.join(tmpdir, "test-results-tpu-unit-1of2.xml"),
          [("test_a", "tests.unit.some_test.SomeTest", "20.0")],
      )
      self._write_suite(
          os.path.join(tmpdir, "test-results-tpu-unit-2of2.xml"),
          [("test_b", "tests.unit.some_test.SomeTest", "30.0")],
      )
      save_baseline_path = os.path.join(tmpdir, "new_baseline.json")
      benchmark_path = os.path.join(tmpdir, "benchmark.json")

      code = self._run_main([tmpdir, "--save-baseline", save_baseline_path, "--output-benchmark", benchmark_path])

      self.assertEqual(code, 0)
      with open(save_baseline_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
      self.assertEqual(saved["tpu-unit::tests.unit.some_test.SomeTest.test_b"], 30.0)
      with open(benchmark_path, "r", encoding="utf-8") as f:
        names = [b["name"] for b in json.load(f)]
      self.assertIn("Total TPU-UNIT Tests Duration", names)

  def test_cpu_excluded_from_macro_benchmarks(self):
    """Verifies that CPU suites are skipped when building macro-level benchmark entries."""
    total_times_by_job = {
        "gpu-unit": 10.0,
        "tpu-unit": 20.0,
        "cpu-unit": 30.0,
        "cpu-torch-reference": 40.0,
    }
    benchmarks = []
    for job, total_time in total_times_by_job.items():
      if "cpu" in job.lower():
        continue
      benchmarks.append({"name": f"Total {job.upper()} Tests Duration", "value": total_time})

    names = [b["name"] for b in benchmarks]
    self.assertIn("Total GPU-UNIT Tests Duration", names)
    self.assertIn("Total TPU-UNIT Tests Duration", names)
    self.assertNotIn("Total CPU-UNIT Tests Duration", names)
    self.assertNotIn("Total CPU-TORCH-REFERENCE Tests Duration", names)

  def test_find_missing_shards_reports_missing_group(self):
    files = [
        "test-results/test-results-gpu-unit-1of2.xml",
        "test-results/test-results-gpu-unit-2of2.xml",
        "test-results/test-results-tpu-unit-1of3.xml",
        "test-results/test-results-tpu-unit-2of3.xml",
    ]
    self.assertEqual(find_missing_shards(files), {"tpu-unit": [3]})

  def test_find_missing_shards_complete_or_unsharded(self):
    files = [
        "test-results-tpu-unit-1of2.xml",
        "test-results-tpu-unit-2of2.xml",
        "test-results-cpu-unit-1.xml",
        "test-results-decoupled-targeted.xml",
    ]
    self.assertEqual(find_missing_shards(files), {})


if __name__ == "__main__":
  unittest.main()
