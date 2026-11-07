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
import re
import sys
import xml.etree.ElementTree as ET

# Relative regression threshold ratio (1.20 = 20% slower)
REL_REGRESSION_OTHER_RATIO = 1.20

# Absolute increase thresholds to filter out noise
ABS_INCREASE_UNIT_SEC = 15.0
ABS_INCREASE_INTEGRATION_SEC = 30.0

# Modules excluded from per-test regression enforcement: their per-test regressions are logged as warnings for
# visibility but do not fail the CI run.
# Before the class-level split (tests/utils/class_split.py), pytest-split dealt single tests, and its deal changed
# whenever the set of selected tests or their recorded durations changed (PR runs deselect scheduled_only tests, so
# a PR's shards never matched the nightly's). The first test in each shard paid a cold-compile penalty (~30-130s)
# while later tests reused the cache (~4-25s), which created false duration swings (e.g. 5s -> 130s) that depended
# on shard position, not code changes. See: https://github.com/AI-Hypercomputer/maxtext/pull/5308
# The class split keeps a class's tests together on TPU unit flavors, which ended those swings for
# megablox_gmm_3d_test, megablox_relayout_test, m3.qwen3_vl_comparison_test and tiling_test (no alert for them in a
# week of runs; their largest increase in 28 PR runs was about 2s), so they are enforced again. What stays:
# - tests.unit.moe_test: MoePinSparseCoreAllGathersTest is a one-test class that reuses a compile of another class
#   when both land on the same shard (~0.5s) and pays it otherwise (~95s), as the deal follows recorded durations.
# - tests.unit.attention_test: MLATest reuses compiles of AttentionTest and CompressedAttentionTest when they share a
#   shard, so tests such as test_tpu_dot_product_context_parallel_with_indexer_lb_cp4_smallk take ~38s or ~50s
#   depending on the deal, just under the 15s limit today.
# - tests.integration.pipeline_parallelism_test: integration flavors still use pytest-split.
# TODO: Drop moe_test and attention_test once their classes no longer depend on another class's compile (e.g. by
# keeping the classes of a module on one shard; tests/conftest.py clears the JAX caches after each module anyway).
EXCLUDED_MODULES_PER_TEST = {
    "tests.unit.moe_test",
    "tests.unit.attention_test",
    "tests.integration.pipeline_parallelism_test",
}

# Suite-level check of new-test cost, per flavor: enforced on PR runs; other runs only warn (--warn-only).
# A flavor fails when its passed tests, new ones included, take more than SUITE_RATIO_LIMIT times the baseline time of
# the same existing tests AND more than SUITE_FLOOR_SEC longer. Using the same existing tests on both sides keeps the
# difference in test selection between PR and scheduled runs out of the ratio. A normal PR run stays within a few
# percent of its baseline, so the limit only trips on real growth; the floor keeps small flavors from failing over
# a single quick test. Known limit: a renamed, moved or re-parametrized test has a new baseline key, so it counts as
# new and its old entry gives no credit; a large rename can trip the check on the renaming PR and on PRs that run
# before the next scheduled run records the new names. Track is not a required check, so such a failure can be ignored.
SUITE_RATIO_LIMIT = 1.20
SUITE_FLOOR_SEC = 300.0
SUITE_TOP_NEW_TESTS = 10


def extract_job_name(xml_file):
  """Extracts job/flavor name from XML filename."""
  basename = os.path.basename(xml_file)
  parts = basename.removesuffix(".xml").split("-")
  if len(parts) >= 4 and parts[0] == "test" and parts[1] == "results":
    return "-".join(parts[2:-1])
  elif len(parts) >= 3:
    return parts[2]
  else:
    return "unknown"


_SHARD_SUFFIX_RE = re.compile(r"^(\d+)(?:of(\d+))?$")


def extract_shard_layout(xml_file):
  """Extracts (worker_group, total_workers) from a shard result filename.

  'test-results-<flavor>-<G>of<N>.xml' gives (G, N) and 'test-results-<flavor>-<G>.xml' gives
  (G, None). Any other name gives None.
  """
  parts = os.path.basename(xml_file).removesuffix(".xml").split("-")
  if len(parts) < 4 or parts[0] != "test" or parts[1] != "results":
    return None
  match = _SHARD_SUFFIX_RE.match(parts[-1])
  if not match:
    return None
  group = int(match.group(1))
  total = int(match.group(2)) if match.group(2) else None
  return group, total


def find_missing_shards(xml_files):
  """Returns {job_name: [missing worker groups]} for jobs whose result files do not cover 1..N.

  A shard that was cancelled or timed out never uploads its XML, so the total N carried by the
  'GofN' suffix of its sibling files is larger than the number of groups seen. Files without a
  'GofN' suffix carry no expectation and are never reported.
  """
  expected_total = {}
  seen_groups = {}
  for xml_file in xml_files:
    layout = extract_shard_layout(xml_file)
    if layout is None:
      continue
    group, total = layout
    job_name = extract_job_name(xml_file)
    seen_groups.setdefault(job_name, set()).add(group)
    if total is not None:
      expected_total[job_name] = max(expected_total.get(job_name, 0), total)

  missing = {}
  for job_name, total in sorted(expected_total.items()):
    missing_groups = [g for g in range(1, total + 1) if g not in seen_groups[job_name]]
    if missing_groups:
      missing[job_name] = missing_groups
  return missing


def testcase_markers(testcase):
  """Returns the pytest markers that tests/conftest.py records as 'marker' properties of a JUnit testcase."""
  markers = set()
  properties_elem = testcase.find("properties")
  if properties_elem is not None:
    for prop in properties_elem.findall("property"):
      if prop.get("name") == "marker":
        markers.add(prop.get("value"))
  return markers


def process_testcase(testcase, xml_file, job_name, baseline_data, new_baseline_data, suite_stats=None):
  """Processes a passed testcase: updates `new_baseline_data` and `suite_stats`, then checks for a per-test regression.

  Returns True when the test regressed and its module is not in EXCLUDED_MODULES_PER_TEST.
  `suite_stats[job_name]` holds the flavor's totals for check_suite_growth(): "base" (baseline seconds of the tests
  that have a baseline entry), "now" (seconds of the same tests plus the new ones), "new_tests" ((seconds, name) of the
  tests with no entry) and "increases" ((seconds over baseline, name) of the tests with one).
  """
  # Skip skipped, failed, or errored tests to avoid capturing ~0s or truncated durations.
  if testcase.find("skipped") is not None or testcase.find("failure") is not None or testcase.find("error") is not None:
    return False

  time_val = float(testcase.get("time", 0.0))
  classname = testcase.get("classname", "unknown")
  full_name = f"{classname}.{testcase.get('name', 'unknown')}"
  markers = testcase_markers(testcase)

  baseline_key = f"{job_name}::{full_name}"
  base_time = baseline_data.get(baseline_key)
  new_baseline_data[baseline_key] = time_val

  if suite_stats is not None:
    stats = suite_stats.setdefault(job_name, {"base": 0.0, "now": 0.0, "new_tests": [], "increases": []})
    if base_time is not None:
      stats["base"] += base_time
      stats["now"] += time_val
      stats["increases"].append((time_val - base_time, full_name))
    elif "scheduled_only" not in markers:
      # A new scheduled_only test runs here only because the PR adds or changes it (newly_added), and later PR runs
      # skip it, so it is left out.
      stats["now"] += time_val
      stats["new_tests"].append((time_val, full_name))

  # Skip per-test regression checking for CPU tests due to shared CPU multi-tenancy noise, and for tests without a
  # baseline above 0 (the ratio needs one).
  is_cpu = "cpu" in os.path.basename(xml_file).lower() or "cpu" in job_name.lower()
  if is_cpu or base_time is None or base_time <= 0:
    return False

  is_integration = "integration_test" in markers
  abs_noise_threshold = ABS_INCREASE_INTEGRATION_SEC if is_integration else ABS_INCREASE_UNIT_SEC
  test_type = "Integration Test" if is_integration else "Unit Test"

  ratio = time_val / base_time
  increase = time_val - base_time
  failed = False
  if ratio >= REL_REGRESSION_OTHER_RATIO and increase > abs_noise_threshold:
    # classname is "<module>.<Class>" for a test method and "<module>" for a test function.
    is_excluded = any(classname == m or classname.startswith(m + ".") for m in EXCLUDED_MODULES_PER_TEST)
    if is_excluded:
      print(
          f"::warning::[PER-TEST REGRESSION ALERT] {test_type} significantly degraded"
          " (Warn-only: module excluded due to shard-position noise)."
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
    print(f"  Thresholds: >{(REL_REGRESSION_OTHER_RATIO - 1) * 100:.0f}% AND >{abs_noise_threshold}s")
    print("-" * 50)

  return failed


def check_suite_growth(suite_stats, skip_jobs):
  """Prints each checked flavor's suite growth and returns the flavors over both suite limits.

  CPU and decoupled flavors (runner noise), flavors in `skip_jobs` (a missing or unreadable shard result file) and
  flavors with no baseline time are not checked.
  """
  over_limit = []
  for job_name, stats in sorted(suite_stats.items()):
    if job_name == "unknown" or "cpu" in job_name.lower() or "decoupled" in job_name.lower() or job_name in skip_jobs:
      continue
    base, now, new_tests = stats["base"], stats["now"], stats["new_tests"]
    if base <= 0:
      continue
    ratio = now / base
    increase = now - base
    new_time = sum(t for t, _ in new_tests)
    print(
        f"[SUITE] Flavor {job_name}: {now:.0f}s vs {base:.0f}s baseline for the same existing tests "
        f"({ratio:.2f}x, {increase:+.0f}s); {len(new_tests)} new test(s) took {new_time:.0f}s."
    )
    if ratio > SUITE_RATIO_LIMIT and increase > SUITE_FLOOR_SEC:
      over_limit.append(job_name)
      print(f"::error::[SUITE REGRESSION] Flavor {job_name} takes {ratio:.2f}x its baseline time (+{increase:.0f}s).")
      print(f"  Thresholds: >{SUITE_RATIO_LIMIT:.2f}x AND >{SUITE_FLOOR_SEC:.0f}s")
      if new_tests:
        print("  Costliest new tests (mark long sweeps scheduled_only to keep them off PR runs):")
        for t, name in sorted(new_tests, reverse=True)[:SUITE_TOP_NEW_TESTS]:
          print(f"    {t:8.1f}s  {name}")
        print("  A renamed or moved test counts as new: if these were only renamed, this can be ignored (not required).")
      slower = [(d, name) for d, name in sorted(stats["increases"], reverse=True) if d > 0][:SUITE_TOP_NEW_TESTS]
      if slower:
        print("  Existing tests that got slower the most:")
        for d, name in slower:
          print(f"    {d:+8.1f}s  {name}")
      print("-" * 50)
  return over_limit


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
        loaded = json.load(f)
      # Keep the '<flavor>::<test>' entries that hold a duration (a number >= 0). Old keys without the flavor separator
      # '::' (or with '::MODULE::') and any other value are dropped here, once, for the checks and the saved baseline.
      baseline_data = {
          k: v
          for k, v in loaded.items()
          if "::" in k and "::MODULE::" not in k and isinstance(v, (int, float)) and v >= 0
      }
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error loading baseline {args.baseline}: {e}")

  # Initialize with existing baseline data, so flavors and tests that do not run in this invocation keep their entries.
  new_baseline_data = dict(baseline_data)
  has_regression = False
  has_errors = False

  total_times_by_job = {}
  total_tests_by_job = {}
  active_keys_by_job = {}
  corrupt_jobs = set()
  suite_stats = {}

  # A shard that was cancelled or timed out leaves no result file. Its tests keep their previous baseline
  # entries and the tests that ran are recorded as usual, but the flavor's totals would be short, so they
  # stay off the dashboard.
  missing_shards = find_missing_shards(xml_files)
  for job_name, groups in missing_shards.items():
    missing_groups = ", ".join(str(g) for g in groups)
    print(
        f"::warning::[PARTIAL RESULTS] Flavor {job_name} has no result file for worker group(s) {missing_groups}. "
        "Its totals stay off the dashboard; the tests of the missing group(s) keep their previous baseline entries."
    )

  for xml_file in xml_files:
    job_name = extract_job_name(xml_file)

    try:
      tree = ET.parse(xml_file)
      root = tree.getroot()

      job_time = 0.0
      job_count = 0

      for testcase in root.iter("testcase"):
        job_count += 1
        time_val = float(testcase.get("time", 0.0))
        job_time += time_val

        if testcase.find("skipped") is None:
          name = testcase.get("name", "unknown")
          classname = testcase.get("classname", "unknown")
          active_keys_by_job.setdefault(job_name, set()).add(f"{job_name}::{classname}.{name}")

        # Micro-level regression check (enforced for non-excluded modules, warn-only for excluded); also adds the
        # test to its flavor's suite totals.
        if process_testcase(testcase, xml_file, job_name, baseline_data, new_baseline_data, suite_stats):
          has_regression = True

      if job_name != "unknown" or job_count > 0:
        total_times_by_job[job_name] = total_times_by_job.get(job_name, 0.0) + job_time
        total_tests_by_job[job_name] = total_tests_by_job.get(job_name, 0) + job_count

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error parsing or processing {xml_file}: {e}")
      corrupt_jobs.add(job_name)
      has_errors = True

  # Suite-level check of new-test cost (like-for-like time per flavor, new tests included).
  has_suite_regression = bool(check_suite_growth(suite_stats, set(missing_shards) | corrupt_jobs))

  # Prune stale baseline entries (deleted, renamed, moved, or skipped tests) for flavors that ran all shards.
  # Flavors that did not run in this invocation or have missing/corrupt shards keep their previous baseline entries.
  for job_name in total_times_by_job:
    if job_name == "unknown" or job_name in missing_shards or job_name in corrupt_jobs:
      continue
    active_keys = active_keys_by_job.get(job_name, set())
    if not active_keys:
      print(f"::warning::[EMPTY RESULTS] Flavor {job_name} recorded 0 active tests; preserving its baseline entries.")
      continue
    prefix = f"{job_name}::"
    for key in [k for k in new_baseline_data if k.startswith(prefix) and k not in active_keys]:
      del new_baseline_data[key]

  # Output macro-level benchmark JSON if requested
  if args.output_benchmark:
    benchmarks = []
    for job, total_time in sorted(total_times_by_job.items()):
      # Exclude CPU/decoupled suites (runner noise) and flavors with missing/corrupt shards or 0 active tests
      if (
          "cpu" in job.lower()
          or "decoupled" in job.lower()
          or job in missing_shards
          or job in corrupt_jobs
          or not active_keys_by_job.get(job)
      ):
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

    if not benchmarks:
      # github-action-benchmark fails on an empty list. Without the file, track_performance.yml skips that step.
      print(f"No flavor has dashboard totals in this run; {args.output_benchmark} is not written.")
    else:
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
  elif has_regression or has_suite_regression:
    if has_regression:
      print("\nOne or more tests regressed significantly.")
    if has_suite_regression:
      print("\nOne or more flavors grew past the suite limits (see [SUITE REGRESSION] above).")
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
