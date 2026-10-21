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

"""Utility to convert test durations format for pytest-split."""

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

_PARAM_METHOD_DECORATORS = {"named_parameters", "parameters", "product", "expand"}
_PARAM_CLASS_DECORATORS = {"parameterized_class", "named_parameters", "parameters", "product"}

# Cache keyed by (file_abs, mtime_ns, size) -> {cls_name: ([(method_name, is_param)], [base_names], is_param_cls)}
_CLASS_INDEX_CACHE: dict[
    tuple[str, int, int],
    dict[str, tuple[list[tuple[str, bool]], list[str], bool]],
] = {}


def get_repo_root() -> str:
  """Returns the absolute path to the repository root directory."""
  return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def is_parameterized_method_decorator(decorator_list: list[ast.expr]) -> bool:
  """Returns True if `decorator_list` contains a method parameterization decorator."""
  for dec in decorator_list:
    node = dec.func if isinstance(dec, ast.Call) else dec
    name = node.attr if isinstance(node, ast.Attribute) else (node.id if isinstance(node, ast.Name) else "")
    if name in _PARAM_METHOD_DECORATORS:
      return True
  return False


def is_parameterized_class_decorator(decorator_list: list[ast.expr]) -> bool:
  """Returns True if `decorator_list` contains a class parameterization decorator."""
  for dec in decorator_list:
    node = dec.func if isinstance(dec, ast.Call) else dec
    name = node.attr if isinstance(node, ast.Attribute) else (node.id if isinstance(node, ast.Name) else "")
    if name in _PARAM_CLASS_DECORATORS:
      return True
  return False


def _get_class_index(
    file_abs: str,
) -> dict[str, tuple[list[tuple[str, bool]], list[str], bool]]:
  """Parses and caches top-level class method definitions for `file_abs`."""
  try:
    st = os.stat(file_abs)
  except OSError:
    return {}
  if st.st_size == 0:
    return {}

  cache_key = (file_abs, st.st_mtime_ns, st.st_size)
  if cache_key not in _CLASS_INDEX_CACHE:
    try:
      with open(file_abs, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_abs)
    except (OSError, SyntaxError):
      _CLASS_INDEX_CACHE[cache_key] = {}
      return {}

    index: dict[str, tuple[list[tuple[str, bool]], list[str], bool]] = {}
    for stmt in tree.body:
      if not isinstance(stmt, ast.ClassDef):
        continue
      methods = [
          (m.name, is_parameterized_method_decorator(m.decorator_list))
          for m in stmt.body
          if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
      ]
      bases = [b.id for b in stmt.bases if isinstance(b, ast.Name)]
      is_param_cls = is_parameterized_class_decorator(stmt.decorator_list)
      index[stmt.name] = (methods, bases, is_param_cls)
    _CLASS_INDEX_CACHE[cache_key] = index
  return _CLASS_INDEX_CACHE[cache_key]


def _resolve_class_name(file_abs: str, cls_name: str, method_name: str) -> str:
  """Resolves `cls_name` when a test method moved to another class in the same file."""
  classes = _get_class_index(file_abs)
  if not classes:
    return cls_name

  def match_score(cname: str, visited: set[str] | None = None) -> int:
    if visited is None:
      visited = set()
    if cname in visited or cname not in classes:
      return -1
    visited.add(cname)
    methods, bases, _ = classes[cname]
    best = -1
    for m_name, is_param in methods:
      if m_name == method_name:
        return 100000
      if is_param and method_name.startswith(m_name):
        best = max(best, len(m_name))
    for base in bases:
      best = max(best, match_score(base, visited))
    return best

  ast_cls = cls_name
  if ast_cls not in classes:
    for cname, (_, _, is_param_cls) in classes.items():
      if is_param_cls and cls_name.startswith(f"{cname}_"):
        ast_cls = cname
        break

  if match_score(ast_cls) >= 0:
    return cls_name

  # Check if the method moved to a single concrete class in the same file (e.g. class split).
  scored = [
      (cname, match_score(cname))
      for cname, (_, _, is_param_cls) in classes.items()
      if not cname.startswith("_") and not is_param_cls
  ]
  scored = [(cname, score) for cname, score in scored if score >= 0]
  if not scored:
    return cls_name

  max_score = max(score for _, score in scored)
  best_classes = [cname for cname, score in scored if score == max_score]
  if len(best_classes) == 1:
    return best_classes[0]
  return cls_name


def resolve_entry(
    key: str,
    duration: float,
    repo_root: Optional[str] = None,
) -> Optional[Tuple[str, str, float, bool]]:
  """Resolves a baseline key into (flavor, pytest_node_id, duration, is_remapped)."""
  if repo_root is None:
    repo_root = get_repo_root()

  flavor = ""
  if "::" in key:
    flavor, key = key.split("::", 1)

  # Extract pytest parametrization bracket (e.g. '[param1.ext-param2]') if present
  # so dots within parameter values are preserved and not split into '::'.
  params = ""
  if "[" in key and key.endswith("]"):
    bracket_idx = key.index("[")
    params = key[bracket_idx:]
    key = key[:bracket_idx]

  parts = key.split(".")
  if not parts or parts[0] != "tests":
    return None

  file_path = ""
  file_abs = ""
  remaining_parts = []
  for i in range(1, len(parts)):
    candidate_rel = os.path.join(*parts[:i]) + ".py"
    candidate_abs = os.path.join(repo_root, candidate_rel)
    if os.path.isfile(candidate_abs):
      file_path = candidate_rel.replace(os.sep, "/")
      file_abs = candidate_abs
      remaining_parts = parts[i:]
      break
  else:
    return None

  if not remaining_parts:
    return None

  remapped = False
  if len(remaining_parts) >= 2:
    cls_name = remaining_parts[0]
    method_name = ".".join(remaining_parts[1:])
    resolved_cls = _resolve_class_name(file_abs, cls_name, method_name)
    remapped = resolved_cls != cls_name
    node_parts = [resolved_cls, method_name]
  else:
    node_parts = remaining_parts

  new_key = f"{file_path}::" + "::".join(node_parts) + params
  return flavor, new_key, duration, remapped


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
  res = resolve_entry(key, duration, repo_root=repo_root)
  return res[:3] if res is not None else None


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
  best: Dict[str, Tuple[int, float]] = {}

  for key, duration in raw_data.items():
    res = resolve_entry(key, float(duration), repo_root=repo_root)
    if not res:
      continue
    flavor, node_id, dur, remapped = res
    if target_flavor:
      if flavor == target_flavor:
        prio = 3 if remapped else 4
      elif not flavor:
        prio = 1 if remapped else 2
      else:
        continue
    else:
      prio = 1 if remapped else 2
    if node_id not in best or prio >= best[node_id][0]:
      best[node_id] = (prio, dur)

  return {node_id: dur for node_id, (_, dur) in best.items()}


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
