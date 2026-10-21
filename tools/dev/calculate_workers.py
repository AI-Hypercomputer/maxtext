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

"""Calculates the optimal number of workers for a given test flavor."""

import argparse
import ast
import json
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

# pylint: disable=wrong-import-position
from tools.dev.convert_durations import get_repo_root
from tools.dev.convert_durations import is_parameterized_class_decorator
from tools.dev.convert_durations import is_parameterized_method_decorator
from tools.dev.convert_durations import resolve_entry
# pylint: enable=wrong-import-position


# Maximum worker limits by hardware flavor
MAX_CPU_WORKERS = 4
MAX_TPU_WORKERS = 3
MAX_GPU_WORKERS = 3
DEFAULT_WORKERS = 1

# Right-sizing parameters for fast test suites
LIGHTWEIGHT_SUITE_MINUTES_THRESHOLD = 16.0
TARGET_MINUTES_PER_WORKER = 8.0
CPU_XDIST_PARALLELISM = 16

HARDWARE_MARKERS = {"tpu_only", "gpu_only", "cpu_only"}


def get_max_workers_for_flavor(flavor: str) -> int:
  """Returns the maximum worker safety cap for a given test flavor."""
  if flavor.startswith("cpu-"):
    return MAX_CPU_WORKERS
  if flavor.startswith("tpu-") or flavor.startswith("tpu7x-"):
    return MAX_TPU_WORKERS
  if flavor.startswith("gpu-"):
    return MAX_GPU_WORKERS
  return DEFAULT_WORKERS


def _extract_decorator_markers(decorator_list: list[ast.expr]) -> set[str]:
  """Extracts pytest marker names from an AST decorator list."""
  marks: set[str] = set()
  for dec in decorator_list:
    node = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
      if isinstance(node.value.value, ast.Name) and node.value.value.id == "pytest" and node.value.attr == "mark":
        marks.add(node.attr)
  return marks


def _extract_pytestmark_assignments(stmts: list[ast.stmt]) -> set[str]:
  """Extracts pytest marker names from `pytestmark = ...` assignments in a statement block."""
  marks: set[str] = set()
  for stmt in stmts:
    if not isinstance(stmt, ast.Assign):
      continue
    if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets):
      continue
    for sub in ast.walk(stmt.value):
      if (
          isinstance(sub, ast.Attribute)
          and isinstance(sub.value, ast.Attribute)
          and isinstance(sub.value.value, ast.Name)
          and sub.value.value.id == "pytest"
          and sub.value.attr == "mark"
      ):
        marks.add(sub.attr)
  return marks


def _parse_file_ast(
    file_abs: str,
    ast_cache: dict[str, tuple[set[str], dict[str, ast.ClassDef], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]],
) -> tuple[set[str], dict[str, ast.ClassDef], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] | None:
  """Parses a Python test file into (module_markers, classes_by_name, functions_by_name)."""
  if file_abs not in ast_cache:
    try:
      with open(file_abs, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_abs)
    except (OSError, SyntaxError):
      return None
    mod_marks = _extract_pytestmark_assignments(tree.body)
    classes: dict[str, ast.ClassDef] = {}
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for stmt in tree.body:
      if isinstance(stmt, ast.ClassDef):
        classes[stmt.name] = stmt
      elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        funcs[stmt.name] = stmt
    ast_cache[file_abs] = (mod_marks, classes, funcs)
  return ast_cache[file_abs]


def _get_test_markers(
    file_path: str,
    parts: list[str],
    repo_root: str,
    ast_cache: dict[str, tuple[set[str], dict[str, ast.ClassDef], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]],
) -> set[str] | None:
  """Resolves the effective pytest markers for a test symbol, or None if the symbol no longer exists."""
  file_abs = os.path.join(repo_root, file_path)
  if not os.path.isfile(file_abs):
    return None
  parsed = _parse_file_ast(file_abs, ast_cache)
  if parsed is None:
    return None
  mod_marks, classes, funcs = parsed

  def get_cls_marks(cls_name: str, visited: set[str] | None = None) -> set[str]:
    if visited is None:
      visited = set()
    if cls_name in visited or cls_name not in classes:
      return set()
    visited.add(cls_name)
    stmt = classes[cls_name]
    marks = _extract_decorator_markers(stmt.decorator_list) | _extract_pytestmark_assignments(stmt.body)
    for base in stmt.bases:
      if isinstance(base, ast.Name):
        marks |= get_cls_marks(base.id, visited)
    return marks

  def find_method_marks(cls_name: str, func_name: str, visited: set[str] | None = None) -> tuple[int, set[str]] | None:
    if visited is None:
      visited = set()
    if cls_name in visited or cls_name not in classes:
      return None
    visited.add(cls_name)
    stmt = classes[cls_name]
    best_score = -1
    best_marks: set[str] | None = None
    for member in stmt.body:
      if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if member.name == func_name:
          return 100000, _extract_decorator_markers(member.decorator_list)
        if is_parameterized_method_decorator(member.decorator_list) and func_name.startswith(member.name):
          score = len(member.name)
          if score > best_score:
            best_score = score
            best_marks = _extract_decorator_markers(member.decorator_list)
    for base in stmt.bases:
      if isinstance(base, ast.Name):
        res = find_method_marks(base.id, func_name, visited)
        if res is not None and res[0] > best_score:
          best_score, best_marks = res
    if best_marks is None:
      return None
    return best_score, best_marks

  if len(parts) >= 2:
    cls_name = parts[0]
    func_name = ".".join(parts[1:])
    if cls_name not in classes:
      for cname, cstmt in classes.items():
        if is_parameterized_class_decorator(cstmt.decorator_list) and cls_name.startswith(f"{cname}_"):
          cls_name = cname
          break
      else:
        return None
    method_res = find_method_marks(cls_name, func_name)
    if method_res is None:
      return None
    return mod_marks | get_cls_marks(cls_name) | method_res[1]

  if len(parts) == 1:
    func_name = parts[0]
    if func_name in funcs:
      return mod_marks | _extract_decorator_markers(funcs[func_name].decorator_list)
    best_func_score = -1
    best_func_marks: set[str] | None = None
    for name, stmt in funcs.items():
      if is_parameterized_method_decorator(stmt.decorator_list) and func_name.startswith(name):
        if len(name) > best_func_score:
          best_func_score = len(name)
          best_func_marks = _extract_decorator_markers(stmt.decorator_list)
    if best_func_marks is not None:
      return mod_marks | best_func_marks
    return None

  return None


def _matches_flavor(flavor: str, file_path: str, markers: set[str], is_scheduled_run: bool = False) -> bool:
  """Returns whether a test with the given markers is selected by `flavor`."""
  if "skip" in markers or (flavor.startswith("tpu7x-") and "skip_on_tpu7x" in markers):
    return False
  if not is_scheduled_run and "scheduled_only" in markers:
    return False

  effective = set(markers)
  if effective.isdisjoint(HARDWARE_MARKERS):
    effective.add("cpu_only")

  if flavor.startswith(("tpu-", "tpu7x-")):
    if "cpu_only" in effective or "gpu_only" in effective:
      return False
  elif flavor.startswith("gpu-"):
    if "cpu_only" in effective or "tpu_only" in effective:
      return False
  elif flavor.startswith("cpu-"):
    if "cpu_only" not in effective:
      return False
  else:
    return True

  if "integration" in flavor:
    if "integration_test" not in effective:
      return False
  elif "unit" in flavor:
    if "integration_test" in effective:
      return False

  if "post-training" in flavor:
    if "post_training" not in effective:
      return False
    allowed_roots = (
        ("tests/post_training/integration/",) if "integration" in flavor else ("tests/post_training/unit/", "tests/unit/")
    )
    return file_path.startswith(allowed_roots)
  if "unit" in flavor or "integration" in flavor:
    return not ("post_training" in effective or file_path.startswith("tests/post_training/"))

  return True


def _resolve_active_baseline_entry(
    key: str,
    duration: float,
    flavor: str,
    *,
    repo_root: str,
    is_scheduled_run: bool,
    ast_cache: dict[str, tuple[set[str], dict[str, ast.ClassDef], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]],
) -> tuple[str, float, bool] | None:
  """Resolves an active baseline entry into (node_id, duration, is_remapped), or None if inactive."""
  test_path = key.split("::", 1)[1] if "::" in key else key
  if not test_path.startswith("tests."):
    return key, duration, False

  res = resolve_entry(key, duration, repo_root=repo_root)
  if res is None:
    return None
  _, node_id, dur, remapped = res
  segments = node_id.split("[", 1)[0].split("::")
  file_path, parts = segments[0], segments[1:]
  markers = _get_test_markers(file_path, parts, repo_root, ast_cache)
  if markers is None or not _matches_flavor(flavor, file_path, markers, is_scheduled_run=is_scheduled_run):
    return None
  return node_id, dur, remapped


def calculate_workers(
    flavor: str,
    baseline_data: dict[str, float] | None = None,
    repo_root: str | None = None,
    is_scheduled_run: bool = False,
) -> tuple[int, list[int]]:
  """Calculates total workers and worker groups based on baseline data.

  Args:
    flavor: The test flavor name (e.g. 'cpu-unit', 'tpu-unit').
    baseline_data: Optional dictionary mapping baseline keys to durations (sec).
    repo_root: Optional repository root path for validating test entries.
    is_scheduled_run: Whether scheduled_only tests are included in the run.

  Returns:
    A tuple of (total_workers, worker_groups).
  """
  max_workers = get_max_workers_for_flavor(flavor)

  if not baseline_data:
    return max_workers, list(range(1, max_workers + 1))

  if repo_root is None:
    repo_root = get_repo_root()

  prefix = f"{flavor}::"
  ast_cache: dict[str, tuple[set[str], dict[str, ast.ClassDef], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]] = {}
  deduped: dict[str, tuple[int, float]] = {}
  for k, dur in baseline_data.items():
    if not k.startswith(prefix) or not isinstance(dur, (int, float)):
      continue
    resolved = _resolve_active_baseline_entry(
        k,
        float(dur),
        flavor,
        repo_root=repo_root,
        is_scheduled_run=is_scheduled_run,
        ast_cache=ast_cache,
    )
    if resolved is None:
      continue
    node_id, active_dur, remapped = resolved
    prio = 1 if remapped else 2
    if node_id not in deduped or prio >= deduped[node_id][0]:
      deduped[node_id] = (prio, active_dur)
  matching = [dur for _, dur in deduped.values()]
  test_count = len(matching)
  total_seconds = sum(matching)
  total_minutes = total_seconds / 60.0

  if test_count == 0:
    return DEFAULT_WORKERS, [DEFAULT_WORKERS]

  # CPU flavors run under pytest-xdist (-n auto on 32-vCPU runners), so summed per-test
  # durations reflect CPU-minutes across workers rather than single-VM wall-minutes.
  if flavor.startswith("cpu-"):
    effective_minutes = max(max(matching) / 60.0, total_minutes / CPU_XDIST_PARALLELISM)
  else:
    effective_minutes = total_minutes

  # Never allocate more workers than available tests
  workers = min(max_workers, test_count)

  # For lightweight test suites, right-size worker count
  if effective_minutes < LIGHTWEIGHT_SUITE_MINUTES_THRESHOLD:
    workers = min(workers, max(1, math.ceil(effective_minutes / TARGET_MINUTES_PER_WORKER)))

  workers = max(1, workers)
  return workers, list(range(1, workers + 1))


def main() -> int:
  """CLI entry point for calculating worker parameters."""
  parser = argparse.ArgumentParser(description="Calculate total workers and worker groups for a test flavor.")
  parser.add_argument(
      "--flavor",
      type=str,
      required=True,
      help="Test flavor name (e.g. cpu-unit, tpu-unit)",
  )
  parser.add_argument(
      "--baseline",
      type=str,
      default=None,
      help="Path to per_test_baseline.json",
  )
  parser.add_argument(
      "--is-scheduled-run",
      action="store_true",
      help="Include @pytest.mark.scheduled_only tests when calculating worker counts",
  )

  args = parser.parse_args()

  baseline_data = None
  if args.baseline and os.path.isfile(args.baseline):
    try:
      with open(args.baseline, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"Warning: Failed to read baseline file '{args.baseline}': {e}",
          file=sys.stderr,
      )

  total_workers, worker_groups = calculate_workers(
      args.flavor,
      baseline_data,
      is_scheduled_run=args.is_scheduled_run,
  )
  worker_groups_json = json.dumps(worker_groups)

  print(f"total_workers={total_workers}")
  print(f"worker_groups={worker_groups_json}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
