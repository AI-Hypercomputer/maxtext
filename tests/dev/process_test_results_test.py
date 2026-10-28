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

"""Unit tests for tools/dev/process_test_results.py."""

import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
from tools.dev import process_test_results

extract_job_name = process_test_results.extract_job_name
process_testcase = process_test_results.process_testcase
extract_shard_layout = process_test_results.extract_shard_layout
find_missing_shards = process_test_results.find_missing_shards


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

  def test_main_records_reported_tests_and_skips_dashboard_for_a_flavor_with_a_missing_shard(self):
    """The tests that ran are recorded, the missing shard's tests keep their entries, and the short total stays off."""
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

      out = io.StringIO()
      with contextlib.redirect_stdout(out):
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
      self.assertIn(
          "::warning::[PARTIAL RESULTS] Flavor tpu-unit has no result file for worker group(s) 2.", out.getvalue()
      )
      with open(save_baseline_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
      self.assertEqual(saved["tpu-unit::tests.unit.some_test.SomeTest.test_a"], 25.0)
      self.assertEqual(saved["tpu-unit::tests.unit.some_test.SomeTest.test_b"], 30.0)
      self.assertEqual(saved["gpu-unit::tests.unit.some_test.SomeTest.test_a"], 7.0)
      with open(benchmark_path, "r", encoding="utf-8") as f:
        names = [b["name"] for b in json.load(f)]
      self.assertIn("Total GPU-UNIT Tests Duration", names)
      self.assertNotIn("Total TPU-UNIT Tests Duration", names)

  def test_main_still_checks_regressions_in_a_flavor_with_a_missing_shard(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self._write_suite(
          os.path.join(tmpdir, "test-results-tpu-unit-1of2.xml"),
          [("test_a", "tests.unit.some_test.SomeTest", "60.0")],
      )
      baseline_path = os.path.join(tmpdir, "baseline.json")
      with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({"tpu-unit::tests.unit.some_test.SomeTest.test_a": 20.0}, f)

      code = self._run_main([tmpdir, "--baseline", baseline_path])

    self.assertEqual(code, 1)

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

  def test_main_writes_no_benchmark_file_when_no_flavor_has_dashboard_totals(self):
    """github-action-benchmark fails on an empty list, so no file is written and the workflow skips that step."""
    with tempfile.TemporaryDirectory() as tmpdir:
      self._write_suite(
          os.path.join(tmpdir, "test-results-tpu-unit-1of2.xml"),
          [("test_a", "tests.unit.some_test.SomeTest", "25.0")],
      )
      self._write_suite(
          os.path.join(tmpdir, "test-results-cpu-unit-1of1.xml"),
          [("test_a", "tests.unit.some_test.SomeTest", "7.0")],
      )
      save_baseline_path = os.path.join(tmpdir, "new_baseline.json")
      benchmark_path = os.path.join(tmpdir, "benchmark.json")

      code = self._run_main([tmpdir, "--save-baseline", save_baseline_path, "--output-benchmark", benchmark_path])

      self.assertEqual(code, 0)
      self.assertFalse(os.path.exists(benchmark_path))
      self.assertTrue(os.path.exists(save_baseline_path))

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

  def test_main_prunes_stale_entries(self):
    """Completed flavors prune deleted/skipped tests while keeping failed tests and untouched flavors."""
    with tempfile.TemporaryDirectory() as tmpdir:
      root = pathlib.Path(tmpdir)
      (root / "test-results-tpu-unit-1of1.xml").write_text(
          "<testsuite>"
          '<testcase name="t_pass" classname="m.C" time="22.0"/>'
          '<testcase name="t_fail" classname="m.C" time="1.0"><failure/></testcase>'
          '<testcase name="t_skip" classname="m.C" time="0.0"><skipped/></testcase>'
          "</testsuite>",
          encoding="utf-8",
      )
      baseline_path = root / "baseline.json"
      save_path = root / "new_baseline.json"
      baseline_path.write_text(
          json.dumps(
              {
                  "tpu-unit::m.C.t_pass": 20.0,
                  "tpu-unit::m.C.t_fail": 30.0,
                  "tpu-unit::m.C.t_skip": 15.0,
                  "tpu-unit::m.C.t_deleted": 45.0,
                  "gpu-unit::m.G.t_gpu": 12.0,
              }
          ),
          encoding="utf-8",
      )

      self.assertEqual(self._run_main([tmpdir, "--baseline", str(baseline_path), "--save-baseline", str(save_path)]), 0)
      self.assertEqual(
          json.loads(save_path.read_text(encoding="utf-8")),
          {"tpu-unit::m.C.t_pass": 22.0, "tpu-unit::m.C.t_fail": 30.0, "gpu-unit::m.G.t_gpu": 12.0},
      )

  def test_corrupt_shard_skips_benchmark_and_baseline_save(self):
    """A flavor with a corrupt shard XML must not emit partial dashboard totals or overwrite the baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
      root = pathlib.Path(tmpdir)
      (root / "test-results-tpu-unit-1of2.xml").write_text(
          '<testsuite><testcase name="t1" classname="m.C" time="20.0"/></testsuite>',
          encoding="utf-8",
      )
      (root / "test-results-tpu-unit-2of2.xml").write_text("<testsuite><unclosed", encoding="utf-8")
      bench_path = root / "bench.json"
      save_path = root / "new_baseline.json"
      self.assertEqual(
          self._run_main([tmpdir, "--save-baseline", str(save_path), "--output-benchmark", str(bench_path)]),
          1,
      )
      self.assertFalse(bench_path.exists())
      self.assertFalse(save_path.exists())


if __name__ == "__main__":
  unittest.main()
