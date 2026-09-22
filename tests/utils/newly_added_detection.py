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

    python3 tests/utils/newly_added_detection.py --base main \
        [--require-marker tpu_only] [--exclude-marker skip_on_tpu7x]

It prints one ``path::test_name`` line per changed test and exits 0, or exits 2 when no
diff against the base could be computed, so callers can tell "nothing changed" from
"detection is broken". The marker flags filter by the ``pytest.mark.*`` decorators a test
carries (on the function, its class, or the module's ``pytestmark``). CI asks for
``tpu_only`` and not ``skip_on_tpu7x``: that is the set the scheduled TPU7X flavors
collect and do not skip, because ``tests/conftest.py`` marks every test without a
hardware marker ``cpu_only`` (excluded by the TPU flavors) and skips ``skip_on_tpu7x``
tests at runtime on TPU7X hardware.

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


def _marker_names(decorator_list):
  """Return the pytest marker names applied by a decorator list.

  Recognises ``@pytest.mark.<name>`` and ``@pytest.mark.<name>(...)``. Any other
  decorator (``@mock.patch`` and the like) is ignored.
  """
  names = set()
  for dec in decorator_list:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "pytest"
    ):
      names.add(target.attr)
  return names


def _module_markers(tree):
  """Return the marker names from a module-level ``pytestmark = ...`` assignment.

  Accepts a single marker or a list/tuple of markers, the two forms pytest documents.
  """
  names = set()
  for node in tree.body:
    if not isinstance(node, ast.Assign):
      continue
    if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
      continue
    value = node.value
    elements = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    names |= _marker_names(elements)
  return names


def _iter_test_defs(tree):
  """Yield ``(name, start_line, end_line, markers)`` for every test function in an AST.

  Covers module-level test functions and methods declared directly inside a class,
  which is what pytest collects. The span starts at the first decorator (if any) so
  decorator-only edits are attributed to the test they decorate. ``markers`` is the
  set of pytest marker names the test carries, from its own decorators, its class's
  decorators and the module's ``pytestmark``, mirroring how pytest inherits markers.
  """
  def_types = (ast.FunctionDef, ast.AsyncFunctionDef)
  module_marks = _module_markers(tree)
  for node in tree.body:
    if isinstance(node, def_types) and node.name.startswith("test_"):
      yield node.name, _span_start(node), node.end_lineno, module_marks | _marker_names(node.decorator_list)
    elif isinstance(node, ast.ClassDef):
      class_marks = module_marks | _marker_names(node.decorator_list)
      for sub in node.body:
        if isinstance(sub, def_types) and sub.name.startswith("test_"):
          yield sub.name, _span_start(sub), sub.end_lineno, class_marks | _marker_names(sub.decorator_list)


def _span_start(node):
  """Return the first source line of a def, including any decorator lines above it."""
  start = node.lineno
  if node.decorator_list:
    start = min(start, min(dec.lineno for dec in node.decorator_list))
  return start


def touched_tests(source, touched_lines):
  """Map each test in ``source`` whose span includes a changed line to its marker names.

  Args:
    source: The new file's Python source.
    touched_lines: Set of changed new-file line numbers for that file.

  Returns:
    ``{test_name: frozenset_of_marker_names}``. Empty if ``touched_lines`` is empty or
    ``source`` does not parse (pytest collection surfaces a syntax error on its own, so
    raising here would only hide it).
  """
  if not touched_lines:
    return {}
  try:
    tree = ast.parse(source)
  except SyntaxError:
    return {}
  found = {}
  for name, start, end, markers in _iter_test_defs(tree):
    if any(start <= line <= end for line in touched_lines):
      found[name] = frozenset(markers)
  return found


def touched_test_names(source, touched_lines):
  """Return the names of test functions in ``source`` whose span includes a changed line."""
  return set(touched_tests(source, touched_lines))


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


def changed_tests_from_diff(diff_text, require_marker=None, exclude_marker=None):
  """Return ``(file_path, test_name)`` for every test the diff added or modified.

  Args:
    diff_text: Raw ``git diff --unified=0`` output. Test sources are read from the
      current working directory, which must be the repository root.
    require_marker: Optional pytest marker name. When given, only tests carrying it (on
      the function, its class or the module's ``pytestmark``) are reported.
    exclude_marker: Optional pytest marker name. Tests carrying it are dropped.
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
    for name, markers in touched_tests(source, touched_lines).items():
      if require_marker is not None and require_marker not in markers:
        continue
      if exclude_marker is not None and exclude_marker in markers:
        continue
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
  parser.add_argument(
      "--require-marker",
      default=None,
      help="Only report tests carrying this pytest marker (e.g. tpu_only).",
  )
  parser.add_argument(
      "--exclude-marker",
      default=None,
      help="Drop tests carrying this pytest marker (e.g. skip_on_tpu7x).",
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
  for path, name in sorted(changed_tests_from_diff(diff_text, args.require_marker, args.exclude_marker)):
    print(f"{path}::{name}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
