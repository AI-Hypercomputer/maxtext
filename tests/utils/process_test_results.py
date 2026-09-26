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

"""Enforces test limits and generates macro-level benchmark metrics in a single pass."""

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

# Relative regression threshold ratio (1.20 = 20% slower)
REL_REGRESSION_OTHER_RATIO = 1.20

# Absolute increase thresholds to filter out noise
ABS_INCREASE_UNIT_SEC = 15.0
ABS_INCREASE_INTEGRATION_SEC = 30.0

# TODO: Remove this exclusion list once pytest-split is fixed to avoid cold-compile noise on first test in shard.
# Modules excluded from per-test regression enforcement.
# These modules have heavy XLA compilation costs that are shared across tests
# within the same pytest-split shard. Because pytest-split randomly redistributes
# tests across shards on every run, the first test in each shard pays a cold-compile
# penalty (~30-130s) while subsequent tests reuse the cache (~4-25s). This creates
# massive false-positive duration swings (e.g. 5s -> 130s) depending on shard
# position, not code changes. Per-test regressions in these modules are logged as
# warnings for visibility but do not fail the CI run.
# See: https://github.com/AI-Hypercomputer/maxtext/pull/5308
EXCLUDED_MODULES_PER_TEST = {
    "tests.unit.moe_test",
    "tests.unit.attention_test",
}


def extract_job_name(xml_file):
  """Extracts job/flavor name from XML filename."""
  basename = os.path.basename(xml_file)
  parts = basename.replace(".xml", "").split("-")
  if len(parts) >= 4 and parts[0] == "test" and parts[1] == "results":
    return "-".join(parts[2:-1])
  elif len(parts) >= 3:
    return parts[2]
  else:
    return "unknown"


def process_testcase(testcase, xml_file, job_name, baseline_data, new_baseline_data):
  """Processes a single testcase and checks for limit violations or regressions.

  Returns:
    A tuple of (module_name, time_val, is_integration, failed) where failed is True
    only if the test regressed AND is not in the exclusion list.
  """
  # 1. Skip processing for skipped tests to avoid corrupting the baseline with ~0s durations
  if testcase.find("skipped") is not None:
    return None, 0.0, False, False

  # 2. Skip processing for failed/errored tests to prevent capturing abnormally short durations
  if testcase.find("failure") is not None or testcase.find("error") is not None:
    return None, 0.0, False, False

  time_val = float(testcase.get("time", 0.0))
  name = testcase.get("name", "unknown")
  classname = testcase.get("classname", "unknown")
  full_name = f"{classname}.{name}"
  module_name = classname.rsplit(".", 1)[0] if "." in classname else classname

  # Parse custom properties to extract markers
  markers = set()
  properties_elem = testcase.find("properties")
  if properties_elem is not None:
    for prop in properties_elem.findall("property"):
      if prop.get("name") == "marker":
        markers.add(prop.get("value"))

  is_integration = "integration_test" in markers
  is_cpu = "cpu" in os.path.basename(xml_file).lower() or "cpu" in job_name.lower()

  if is_integration:
    abs_noise_threshold = ABS_INCREASE_INTEGRATION_SEC
    rel_regression_ratio = REL_REGRESSION_OTHER_RATIO
    test_type = "Integration Test"
  else:
    abs_noise_threshold = ABS_INCREASE_UNIT_SEC
    rel_regression_ratio = REL_REGRESSION_OTHER_RATIO
    test_type = "Unit Test"

  baseline_key = f"{job_name}::{full_name}"
  new_baseline_data[baseline_key] = time_val

  # Skip regression checking for CPU tests due to shared CPU multi-tenancy noise
  skip_regression = is_cpu
  failed = False

  # Check relative regression if baseline exists
  if not skip_regression and baseline_key in baseline_data:
    base_time = baseline_data[baseline_key]
    if isinstance(base_time, (int, float)) and base_time > 0:
      ratio = time_val / base_time
      increase = time_val - base_time
      if ratio >= rel_regression_ratio and increase > abs_noise_threshold:
        is_excluded = module_name in EXCLUDED_MODULES_PER_TEST
        if is_excluded:
          print(
              f"::warning::[PER-TEST REGRESSION ALERT] {test_type} significantly degraded"
              + " (Warn-only: module excluded due to pytest-split sharding noise)."
          )
        else:
          print(f"::error::[REGRESSION ALERT] {test_type} significantly degraded!")
          failed = True
        print(f"  Test: {full_name}")
        print(f"  Flavor: {job_name}")
        print(f"  File: {os.path.basename(xml_file)}")
        print(f"  Previous Duration: {base_time:.2f}s")
        print(f"  New Duration: {time_val:.2f}s")
        print(f"  Increase: +{increase:.2f}s ({(ratio - 1) * 100:.1f}%)")
        print(f"  Thresholds: >{(rel_regression_ratio - 1) * 100:.0f}% AND >{abs_noise_threshold}s")
        print("-" * 50)

  return module_name, time_val, is_integration, failed


def main():
  """Parses XML files and processes test results."""
  parser = argparse.ArgumentParser(
      description="Enforce test limits, check regressions, and parse JUnit XML to Benchmark format."
  )
  parser.add_argument("xml_dir", help="Directory containing JUnit XML files")
  parser.add_argument("--baseline", type=str, help="Path to the baseline JSON file", default=None)
  parser.add_argument(
      "--save-baseline",
      type=str,
      help="Path to save the new baseline JSON file",
      default=None,
  )
  parser.add_argument(
      "--warn-only",
      action="store_true",
      help="Report regressions but do not return a non-zero exit code",
  )
  parser.add_argument(
      "--output-benchmark",
      type=str,
      help="Path to save the benchmark JSON file",
      default=None,
  )
  args = parser.parse_args()

  xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
  if not xml_files:
    print(f"No XML files found in {args.xml_dir}")
    sys.exit(0)

  baseline_data = {}
  if args.baseline and os.path.exists(args.baseline):
    try:
      with open(args.baseline, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error loading baseline {args.baseline}: {e}")

  # Initialize with existing baseline data, filtering out old legacy keys without flavor separator '::'
  new_baseline_data = {k: v for k, v in baseline_data.items() if "::" in k and isinstance(v, (int, float))}
  has_regression = False
  has_errors = False

  total_times_by_job = {}
  total_tests_by_job = {}
  total_times_by_module = {}

  for xml_file in xml_files:
    job_name = extract_job_name(xml_file)
    if job_name not in total_times_by_module:
      total_times_by_module[job_name] = {}

    try:
      tree = ET.parse(xml_file)
      root = tree.getroot()

      job_time = 0.0
      job_count = 0

      for testcase in root.iter("testcase"):
        job_count += 1
        time_val = float(testcase.get("time", 0.0))
        job_time += time_val

        # Per-test regression check (enforced for non-excluded modules, warn-only for excluded)
        res = process_testcase(testcase, xml_file, job_name, baseline_data, new_baseline_data)
        if res and res[0] is not None:
          module_name, t_val, is_integration, per_test_failed = res
          if per_test_failed:
            has_regression = True
          if module_name not in total_times_by_module[job_name]:
            total_times_by_module[job_name][module_name] = [0.0, is_integration]
          total_times_by_module[job_name][module_name][0] += t_val

      if job_name != "unknown" or job_count > 0:
        total_times_by_job[job_name] = total_times_by_job.get(job_name, 0.0) + job_time
        total_tests_by_job[job_name] = total_tests_by_job.get(job_name, 0) + job_count

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error parsing or processing {xml_file}: {e}")
      has_errors = True

  # Check per-module regression
  for job_name, modules in total_times_by_module.items():
    is_cpu = "cpu" in job_name.lower()
    if is_cpu:
      continue

    for module_name, (time_val, is_integration) in modules.items():
      baseline_key = f"{job_name}::MODULE::{module_name}"
      new_baseline_data[baseline_key] = time_val

      if baseline_key in baseline_data:
        base_time = baseline_data[baseline_key]
        if isinstance(base_time, (int, float)) and base_time > 0:
          ratio = time_val / base_time
          increase = time_val - base_time
          abs_noise_threshold = ABS_INCREASE_INTEGRATION_SEC if is_integration else ABS_INCREASE_UNIT_SEC

          if ratio >= REL_REGRESSION_OTHER_RATIO and increase > abs_noise_threshold:
            print(f"::error::[MODULE REGRESSION ALERT] Module {module_name} significantly degraded!")
            print(f"  Flavor: {job_name}")
            print(f"  Previous Duration: {base_time:.2f}s")
            print(f"  New Duration: {time_val:.2f}s")
            print(f"  Increase: +{increase:.2f}s ({(ratio - 1) * 100:.1f}%)")
            print(f"  Thresholds: >{(REL_REGRESSION_OTHER_RATIO - 1) * 100:.0f}% AND >{abs_noise_threshold}s")
            print("-" * 50)
            has_regression = True

  # Output macro-level benchmark JSON if requested
  if args.output_benchmark:
    benchmarks = []
    for job, total_time in sorted(total_times_by_job.items()):
      # Exclude CPU suites from macro-level dashboard tracking to avoid false alerts from CPU runner noise
      if "cpu" in job.lower():
        continue
      benchmarks.append(
          {
              "name": f"Total {job.upper()} Tests Duration",
              "unit": "sec",
              "value": total_time,
          }
      )
      benchmarks.append(
          {
              "name": f"Total {job.upper()} Tests Count",
              "unit": "count",
              "value": total_tests_by_job.get(job, 0),
          }
      )

    try:
      dirname = os.path.dirname(args.output_benchmark)
      if dirname:
        os.makedirs(dirname, exist_ok=True)
      with open(args.output_benchmark, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
      print(f"Saved benchmark results to {args.output_benchmark}")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error saving benchmark results: {e}")
      has_errors = True

  # Save baseline only if no critical execution/parsing errors occurred
  if args.save_baseline:
    if has_errors:
      print(
          "Warning: Critical errors occurred during parsing. "
          "Skipping saving of the new baseline to prevent corruption."
      )
    else:
      try:
        dirname = os.path.dirname(args.save_baseline)
        if dirname:
          os.makedirs(dirname, exist_ok=True)
        sorted_baseline = dict(sorted(new_baseline_data.items()))
        with open(args.save_baseline, "w", encoding="utf-8") as f:
          json.dump(sorted_baseline, f, indent=2)
        print(f"Saved new baseline to {args.save_baseline}")
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error saving baseline {args.save_baseline}: {e}")
        has_errors = True

  # Determine final exit status
  if has_errors:
    print("\nOne or more critical errors occurred during execution.")
    sys.exit(1)
  elif has_regression:
    print("\nOne or more tests or modules regressed significantly.")
    if args.warn_only:
      print("Non-blocking mode active: exiting with code 0.")
      sys.exit(0)
    else:
      sys.exit(1)
  else:
    print("All tests passed execution time and regression checks.")
    sys.exit(0)


if __name__ == "__main__":
  main()
