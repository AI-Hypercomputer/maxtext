# Copyright 2026 Google LLC
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

"""Detect the tests a PR adds or modifies, for scheduled_only and TPU7X pre-submit verification.

The pre-submit CI runs ``pytest -m "<marker> and (not scheduled_only or newly_added)"``.
This module supplies the ``newly_added`` set: the tests a pull request touched, so that a
newly added or modified ``scheduled_only`` test runs at least once before merge instead of
being silently skipped until the nightly scheduled pipeline. On TPU7X runners a pull
request runs ``pytest -m "<marker> and newly_added"`` instead, so only the touched tests
occupy that scarce hardware.

Detection maps changed *line numbers* (from ``git diff --unified=0``) onto each test's
line span (from an ``ast`` parse of the new file). A test counts as changed only when a
changed line lands inside its own span. This avoids trusting git's hunk-header function
name, which points at the function *preceding* an insertion and would otherwise flag an
untouched test that merely sits above newly added code.

The module doubles as a command-line tool for CI jobs that cannot import the ``tests``
package (``tests/__init__.py`` imports packages a bare runner does not have)::

    python3 tests/utils/newly_added_detection.py --base main

It prints one ``path::test_name`` line per changed test and exits 0, or exits 2 when no
diff against the base could be computed, so callers can tell "nothing changed" from
"detection is broken".

Only the Python standard library is used, since this runs on a bare CI runner where
MaxText is not necessarily importable.
"""

import argparse
import ast
import os
import re
import subprocess
import sys

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


def _build_diff_commands(base):
  """Return the ordered ``git diff`` argument lists to try, most-precise first.

  Both ranges are three-dot (merge-base) ranges, so only the branch's own commits
  are reported. Two-dot tip-vs-tip ranges are deliberately excluded: they over-report
  every commit the base gained past the fork point (a stale local ``main`` can inflate
  the changed set many-fold), which would drag unrelated ``scheduled_only`` tests into
  pre-submit. The remote range covers CI and local checkouts that have an ``origin``
  remote; the local range is the fallback for a developer without ``origin`` and is
  never reached in CI, where ``origin/<base>`` is always fetched first.
  """
  return [
      ["git", "diff", "--unified=0", f"origin/{base}...HEAD"],
      ["git", "diff", "--unified=0", f"{base}...HEAD"],
  ]


def _resolve_base(base_ref=None):
  """Return the base ref to diff against: ``base_ref``, else ``$GITHUB_BASE_REF``, else ``main``."""
  return base_ref or os.environ.get("GITHUB_BASE_REF") or "main"


def diff_against_base(base):
  """Return ``git diff --unified=0`` text against the merge-base with ``base``, or None.

  None means no diff could be computed: the working directory is not inside a git work
  tree, or neither ``origin/<base>`` nor ``<base>`` resolves to a ref that shares history
  with HEAD. Callers that gate CI on the result must treat None as "unknown", never as
  "nothing changed".
  """
  try:
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        check=False,
    )
    if inside.returncode != 0:
      return None
    if os.environ.get("GITHUB_ACTIONS") == "true":
      subprocess.run(
          ["git", "fetch", "origin", f"{base}:refs/remotes/origin/{base}"],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
          check=False,
      )
    for command in _build_diff_commands(base):
      try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
      except Exception:  # pylint: disable=broad-exception-caught
        continue
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return None


def changed_tests_from_diff(diff_text):
  """Return ``(file_path, test_name)`` for every test the diff added or modified.

  Args:
    diff_text: Raw ``git diff --unified=0`` output. Test sources are read from the
      current working directory, which must be the repository root.
  """
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


def get_changed_tests(base_ref=None):
  """Return ``(file_path, test_name)`` for every test the PR added or modified.

  Args:
    base_ref: Base git ref to diff against. Defaults to the ``GITHUB_BASE_REF``
      environment variable, then ``"main"``.

  Returns:
    A set of ``(file_path, test_name)`` tuples, or an empty set when not in a git work
    tree or when the diff cannot be computed.
  """
  diff_text = diff_against_base(_resolve_base(base_ref))
  if diff_text is None:
    return set()
  return changed_tests_from_diff(diff_text)


def main(argv=None):
  """Command-line entry point: print one ``path::test_name`` line per changed test.

  Returns 0 when a diff was computed (the output may be empty) and 2 when it was not, so a
  CI gate can tell "no changed tests" apart from "detection is broken".
  """
  parser = argparse.ArgumentParser(description="List the tests a branch added or modified relative to a base ref.")
  parser.add_argument(
      "--base",
      default=None,
      help="Base ref to diff against (default: $GITHUB_BASE_REF, then main).",
  )
  args = parser.parse_args(argv)
  base = _resolve_base(args.base)
  diff_text = diff_against_base(base)
  if diff_text is None:
    print(
        f"newly_added_detection: cannot diff against origin/{base} or {base} (cwd: {os.getcwd()})",
        file=sys.stderr,
    )
    return 2
  for path, name in sorted(changed_tests_from_diff(diff_text)):
    print(f"{path}::{name}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
