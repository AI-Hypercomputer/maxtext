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
import unittest
import xml.etree.ElementTree as ET
import pytest

spec = importlib.util.spec_from_file_location("process_test_results", "tests/utils/process_test_results.py")
process_test_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_test_results)

extract_job_name = process_test_results.extract_job_name
process_testcase = process_test_results.process_testcase


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
    module_name, time_val, is_integration, failed = process_testcase(
        testcase_xml,
        "test-results-gpu-unit-1.xml",
        "gpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertEqual(module_name, "tests.unit.qk_clip_test")
    self.assertEqual(time_val, 15.99)
    self.assertFalse(is_integration)
    self.assertFalse(failed)
    self.assertIn(
        "gpu-unit::tests.unit.qk_clip_test.QKClipMLATest.test_mla_dot_product_integration",
        new_baseline,
    )
    self.assertEqual(
        new_baseline["gpu-unit::tests.unit.qk_clip_test.QKClipMLATest.test_mla_dot_product_integration"],
        15.99,
    )

  def test_process_testcase_regression_enforced_for_non_excluded_module(self):
    """Verifies that a genuine regression in a non-excluded module sets failed=True."""
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

    module_name, time_val, is_integration, failed = process_testcase(
        testcase_xml,
        "test-results-gpu-unit-1.xml",
        "gpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertEqual(module_name, "tests.unit.slow_test")
    self.assertEqual(time_val, 20.0)
    self.assertFalse(is_integration)
    self.assertTrue(failed)

  def test_process_testcase_regression_warn_only_for_excluded_module(self):
    """Verifies that a regression in an excluded module (moe_test) sets failed=False."""
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

    module_name, time_val, _, failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertEqual(module_name, "tests.unit.moe_test")
    self.assertEqual(time_val, 130.0)
    self.assertFalse(failed)

  def test_process_testcase_regression_warn_only_for_attention_test(self):
    """Verifies that a regression in attention_test (excluded) sets failed=False."""
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

    module_name, _, _, failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        baseline_data,
        new_baseline,
    )
    self.assertEqual(module_name, "tests.unit.attention_test")
    self.assertFalse(failed)

  def test_process_testcase_skipped_returns_none(self):
    """Verifies that skipped tests return None for module_name."""
    testcase_xml = ET.Element(
        "testcase",
        {"name": "test_skip", "classname": "tests.unit.foo.Bar", "time": "0.0"},
    )
    ET.SubElement(testcase_xml, "skipped")

    module_name, time_val, _, failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        {},
        {},
    )
    self.assertIsNone(module_name)
    self.assertEqual(time_val, 0.0)
    self.assertFalse(failed)

  def test_process_testcase_failed_returns_none(self):
    """Verifies that failed tests return None for module_name."""
    testcase_xml = ET.Element(
        "testcase",
        {"name": "test_fail", "classname": "tests.unit.foo.Bar", "time": "5.0"},
    )
    ET.SubElement(testcase_xml, "failure")

    module_name, _, _, failed = process_testcase(
        testcase_xml,
        "test-results-tpu-unit-1.xml",
        "tpu-unit",
        {},
        {},
    )
    self.assertIsNone(module_name)
    self.assertFalse(failed)

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


if __name__ == "__main__":
  unittest.main()
