"""Utility to convert test durations format for pytest-split."""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple


def get_repo_root() -> str:
  """Returns the absolute path to the repository root directory."""
  return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def convert_entry(
    key: str,
    duration: float,
    repo_root: Optional[str] = None,
) -> Optional[Tuple[str, str, float]]:
  """Converts a test entry from dot notation to pytest-split format.

  Args:
    key: Raw baseline key (e.g. '<flavor>::tests.unit.test_file.Class.method'
      or 'tests.unit.test_file.Class.method').
    duration: Test execution duration in seconds.
    repo_root: Path to repository root for resolving test files.

  Returns:
    Tuple of (flavor, pytest_node_id, duration) or None if not resolvable.
  """
  if repo_root is None:
    repo_root = get_repo_root()

  flavor = ""
  if "::" in key:
    flavor, key = key.split("::", 1)

  parts = key.split(".")
  if not parts or parts[0] != "tests":
    return None

  file_path = ""
  remaining_parts = []
  for i in range(1, len(parts)):
    candidate_rel = os.path.join(*parts[:i]) + ".py"
    candidate_abs = os.path.join(repo_root, candidate_rel)
    if os.path.isfile(candidate_abs):
      file_path = candidate_rel.replace(os.sep, "/")
      remaining_parts = parts[i:]
      break
  else:
    return None

  new_key = f"{file_path}::" + "::".join(remaining_parts)
  return flavor, new_key, duration


def convert_durations(
    raw_data: Dict[str, Any],
    target_flavor: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, float]:
  """Converts raw baseline entries to pytest-split format with flavor priority.

  Args:
    raw_data: Mapping of test key to duration.
    target_flavor: Optional target flavor to prioritize/filter.
    repo_root: Path to repository root.

  Returns:
    Dictionary of {pytest_node_id: duration}.
  """
  exact_matches: Dict[str, float] = {}
  fallback_matches: Dict[str, float] = {}

  for key, duration in raw_data.items():
    res = convert_entry(key, float(duration), repo_root=repo_root)
    if not res:
      continue
    flavor, node_id, dur = res
    if target_flavor and flavor == target_flavor:
      exact_matches[node_id] = dur
    elif not flavor:
      fallback_matches[node_id] = dur
    elif not target_flavor:
      exact_matches[node_id] = dur

  # Overlay exact matches on top of fallback matches
  fallback_matches.update(exact_matches)
  return fallback_matches


def main():
  parser = argparse.ArgumentParser(description="Convert test durations format for pytest-split.")
  parser.add_argument("input_file", help="Path to input per_test_baseline.json")
  parser.add_argument("output_file", help="Path to output .test_durations JSON")
  parser.add_argument(
      "--flavor",
      default=None,
      help="Target test flavor to filter/prioritize durations for.",
  )

  args = parser.parse_args()

  try:
    with open(args.input_file, "r", encoding="utf-8") as f:
      data = json.load(f)
  except FileNotFoundError:
    print(f"Error: Input file {args.input_file} not found.", file=sys.stderr)
    sys.exit(1)
  except json.JSONDecodeError as e:
    print(f"Error: Failed to decode JSON from {args.input_file}: {e}", file=sys.stderr)
    sys.exit(1)

  converted_data = convert_durations(data, target_flavor=args.flavor)

  with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2)

  print(f"Successfully converted {len(converted_data)} test durations to" f" {args.output_file}")


if __name__ == "__main__":
  main()
