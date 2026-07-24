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


def process_testcase(testcase, xml_file, baseline_data, new_baseline_data):
  """Processes a single testcase and checks for limit violations or regressions."""
  failed = False

  # 1. Skip processing for skipped tests to avoid corrupting the baseline with ~0s durations
  if testcase.find("skipped") is not None:
    return False

  # 2. Skip processing for failed/errored tests to prevent capturing abnormally short durations
  if testcase.find("failure") is not None or testcase.find("error") is not None:
    return False

  time_val = float(testcase.get("time", 0.0))
  name = testcase.get("name", "unknown")
  classname = testcase.get("classname", "unknown")
  full_name = f"{classname}.{name}"

  # Parse custom properties to extract markers
  markers = set()
  properties_elem = testcase.find("properties")
  if properties_elem is not None:
    for prop in properties_elem.findall("property"):
      if prop.get("name") == "marker":
        markers.add(prop.get("value"))

  is_integration = "integration_test" in markers
  is_cpu = "cpu" in os.path.basename(xml_file).lower()

  if is_integration:
    abs_noise_threshold = ABS_INCREASE_INTEGRATION_SEC
    rel_regression_ratio = REL_REGRESSION_OTHER_RATIO
    test_type = "Integration Test"
  else:
    abs_noise_threshold = ABS_INCREASE_UNIT_SEC
    rel_regression_ratio = REL_REGRESSION_OTHER_RATIO
    test_type = "Unit Test"

  new_baseline_data[full_name] = time_val

  # Skip regression checking for CPU tests due to shared CPU multi-tenancy noise
  skip_regression = is_cpu

  # Check relative regression if baseline exists
  if not skip_regression and full_name in baseline_data:
    base_time = baseline_data[full_name]
    if base_time > 0:
      ratio = time_val / base_time
      increase = time_val - base_time
      if ratio >= rel_regression_ratio and increase > abs_noise_threshold:
        print(f"::error::[REGRESSION ALERT] {test_type} significantly degraded!")
        print(f"  Test: {full_name}")
        print(f"  File: {os.path.basename(xml_file)}")
        print(f"  Previous Duration: {base_time:.2f}s")
        print(f"  New Duration: {time_val:.2f}s")
        print(f"  Increase: +{increase:.2f}s ({(ratio - 1) * 100:.1f}%)")
        print(f"  Thresholds: >{(rel_regression_ratio - 1) * 100:.0f}% AND >{abs_noise_threshold}s")
        print("-" * 50)
        failed = True

  return failed


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

  xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))
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

  # Initialize with a copy of baseline_data to preserve history for tests not run or skipped
  new_baseline_data = dict(baseline_data)
  has_regression = False
  has_errors = False

  total_times_by_job = {}
  total_tests_by_job = {}

  for xml_file in xml_files:
    basename = os.path.basename(xml_file)
    parts = basename.replace(".xml", "").split("-")
    if len(parts) >= 4 and parts[0] == "test" and parts[1] == "results":
      job_name = "-".join(parts[2:-1])
    else:
      # Fallback to device type if naming is simple (e.g. test-results-cpu-1.xml)
      if len(parts) >= 3:
        job_name = parts[2]
      else:
        job_name = "unknown"

    try:
      tree = ET.parse(xml_file)
      root = tree.getroot()

      job_time = 0.0
      job_count = 0

      for testcase in root.iter("testcase"):
        job_count += 1
        time_val = float(testcase.get("time", 0.0))
        job_time += time_val

        # Micro-level regression check
        if process_testcase(testcase, xml_file, baseline_data, new_baseline_data):
          has_regression = True

      if job_name != "unknown" or job_count > 0:
        total_times_by_job[job_name] = total_times_by_job.get(job_name, 0.0) + job_time
        total_tests_by_job[job_name] = total_tests_by_job.get(job_name, 0) + job_count

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error parsing or processing {xml_file}: {e}")
      has_errors = True

  # Output macro-level benchmark JSON if requested
  if args.output_benchmark:
    benchmarks = []
    for job, total_time in total_times_by_job.items():
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
        with open(args.save_baseline, "w", encoding="utf-8") as f:
          json.dump(new_baseline_data, f, indent=2)
        print(f"Saved new baseline to {args.save_baseline}")
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error saving baseline {args.save_baseline}: {e}")
        has_errors = True

  # Determine final exit status
  if has_errors:
    print("\nOne or more critical errors occurred during execution.")
    sys.exit(1)
  elif has_regression:
    print("\nOne or more tests regressed significantly.")
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
