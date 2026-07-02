# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detect the tests a PR adds or modifies, for scheduled_only pre-submit verification.

The pre-submit CI runs ``pytest -m "<marker> and (not scheduled_only or newly_added)"``.
This module supplies the ``newly_added`` set: the tests a pull request touched, so that a
newly added or modified ``scheduled_only`` test runs at least once before merge instead of
being silently skipped until the nightly scheduled pipeline.

Detection maps changed *line numbers* (from ``git diff --unified=0``) onto each test's
line span (from an ``ast`` parse of the new file). A test counts as changed only when a
changed line lands inside its own span. This avoids trusting git's hunk-header function
name, which points at the function *preceding* an insertion and would otherwise flag an
untouched test that merely sits above newly added code.

Only the Python standard library is used, since this runs on a bare CI runner where
MaxText is not necessarily importable.
"""

import ast
import os
import re
import subprocess

# Matches pytest.ini's ``python_files = *_test.py *_tests.py``.
_TEST_SUFFIXES = ("_test.py", "_tests.py")
# Matches pytest.ini's ``testpaths = tests``.
_TESTS_ROOT = "tests/"
# Captures the new-file start line and (optional) length from a unified-diff hunk header:
#   @@ -<old_start>[,<old_len>] +<new_start>[,<new_len>] @@
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _is_test_file(path):
  """Return True if ``path`` is a MaxText test file (under tests/ with a test suffix)."""
  normalized = path.strip().replace(os.sep, "/")
  return normalized.startswith(_TESTS_ROOT) and normalized.endswith(_TEST_SUFFIXES)


def parse_changed_line_map(diff_text):
  """Map each changed file to the set of new-file line numbers it touched.

  Args:
    diff_text: Raw ``git diff --unified=0`` output.

  Returns:
    A dict of ``{file_path: set_of_new_line_numbers}``. ``file_path`` is repo-root
    relative (the ``b/`` prefix is stripped). For a pure deletion (new length 0) the
    two lines bracketing the removal are recorded, so a test that had lines removed is
    still detected. Deleted files (``+++ /dev/null``) are omitted.
  """
  line_map = {}
  current_file = None
  for line in diff_text.splitlines():
    if line.startswith("+++"):
      path = line[3:].strip()
      if path.startswith("b/"):
        path = path[2:]
      current_file = None if path == "/dev/null" else path
      continue
    if line.startswith("@@"):
      if current_file is None:
        continue
      match = _HUNK_RE.match(line)
      if match is None:
        continue
      new_start = int(match.group(1))
      new_len = int(match.group(2)) if match.group(2) is not None else 1
      touched = line_map.setdefault(current_file, set())
      if new_len > 0:
        touched.update(range(new_start, new_start + new_len))
      else:
        touched.update({new_start, new_start + 1})
  return line_map


def _iter_test_defs(tree):
  """Yield ``(name, start_line, end_line)`` for every test function in an AST.

  Covers module-level test functions and methods declared directly inside a class,
  which is what pytest collects. The span starts at the first decorator (if any) so
  decorator-only edits are attributed to the test they decorate.
  """
  def_types = (ast.FunctionDef, ast.AsyncFunctionDef)
  for node in tree.body:
    if isinstance(node, def_types) and node.name.startswith("test_"):
      yield node.name, _span_start(node), node.end_lineno
    elif isinstance(node, ast.ClassDef):
      for sub in node.body:
        if isinstance(sub, def_types) and sub.name.startswith("test_"):
          yield sub.name, _span_start(sub), sub.end_lineno


def _span_start(node):
  """Return the first source line of a def, including any decorator lines above it."""
  start = node.lineno
  if node.decorator_list:
    start = min(start, min(dec.lineno for dec in node.decorator_list))
  return start


def touched_test_names(source, touched_lines):
  """Return the names of test functions in ``source`` whose span includes a changed line.

  Args:
    source: The new file's Python source.
    touched_lines: Set of changed new-file line numbers for that file.

  Returns:
    A set of test function names. Empty if ``touched_lines`` is empty or ``source`` does
    not parse (pytest collection surfaces a syntax error on its own, so raising here would
    only hide it).
  """
  if not touched_lines:
    return set()
  try:
    tree = ast.parse(source)
  except SyntaxError:
    return set()
  found = set()
  for name, start, end in _iter_test_defs(tree):
    if any(start <= line <= end for line in touched_lines):
      found.add(name)
  return found


def get_changed_tests(base_ref=None):
  """Return ``(file_path, test_name)`` for every test the PR added or modified.

  Args:
    base_ref: Base git ref to diff against. Defaults to the ``GITHUB_BASE_REF``
      environment variable, then ``"main"``.

  Returns:
    A set of ``(file_path, test_name)`` tuples, or an empty set when not in a git work
    tree or when the diff cannot be computed.
  """
  base = base_ref or os.environ.get("GITHUB_BASE_REF") or "main"
  try:
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        check=False,
    )
    if inside.returncode != 0:
      return set()
    if os.environ.get("GITHUB_ACTIONS") == "true":
      subprocess.run(
          ["git", "fetch", "origin", f"{base}:refs/remotes/origin/{base}"],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
          check=False,
      )
    try:
      diff_text = subprocess.check_output(
          ["git", "diff", "--unified=0", f"origin/{base}...HEAD"],
          text=True,
          stderr=subprocess.DEVNULL,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      diff_text = subprocess.check_output(
          ["git", "diff", "--unified=0", f"origin/{base}", "HEAD"],
          text=True,
          stderr=subprocess.DEVNULL,
      )
  except Exception:  # pylint: disable=broad-exception-caught
    return set()

  changed = set()
  for path, touched_lines in parse_changed_line_map(diff_text).items():
    if not _is_test_file(path):
      continue
    try:
      with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    except OSError:
      continue
    for name in touched_test_names(source, touched_lines):
      changed.add((path, name))
  return changed
