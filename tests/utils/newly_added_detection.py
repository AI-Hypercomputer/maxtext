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
being silently skipped until the nightly scheduled pipeline. TPU7X runners are scarce, so a
pull request runs ``pytest -m "<flavor marker> and newly_added"`` there instead, and only on
the ``TPU7X_FLAVORS`` that run at least one touched test.

Detection maps changed *line numbers* (from ``git diff --unified=0``) onto an ``ast`` parse
of each test file. A test counts as changed when a changed line lands inside its own span,
in the decorators or header of its class, or in a ``pytestmark`` assignment of its class or
module. This avoids trusting git's hunk-header function name, which points at the function
*preceding* an insertion and would otherwise flag an untouched test that merely sits above
newly added code.

Markers are read statically from ``@pytest.mark.<name>`` decorators and ``pytestmark``
assignments. Tests that a class inherits, marker aliases and ``pytest.param(..., marks=...)``
are not seen, so such tests may not run on TPU7X before merge.

As a script, run from the repository root, it prints one ``path::test_name`` line per touched
test. ``--flavor NAME`` keeps the tests that the TPU7X flavor runs, and ``--eligible-flavors``
prints instead a compact JSON array of the flavors that run at least one, e.g.
``["tpu7x-unit"]``. It exits 2 when it cannot diff against the base.

Only the Python standard library is used, since this runs on a bare CI runner where
MaxText is not necessarily importable.
"""

import argparse
import ast
import configparser
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
from typing import NamedTuple

# Matches pytest.ini's ``python_files = *_test.py *_tests.py``.
_TEST_SUFFIXES = ("_test.py", "_tests.py")
# Matches pytest.ini's ``testpaths = tests``.
_TESTS_ROOT = "tests/"
# Captures the new-file start line and (optional) length from a unified-diff hunk header:
#   @@ -<old_start>[,<old_len>] +<new_start>[,<new_len>] @@
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
# tests/conftest.py marks a test that has none of these ``cpu_only``.
_HARDWARE_MARKERS = frozenset({"tpu_only", "gpu_only", "cpu_only"})


class Flavor(NamedTuple):
  """How a TPU7X flavor selects tests: its ``-m`` expression, the paths it passes to pytest (pytest.ini's
  ``testpaths`` when empty) and its own ``--ignore`` paths."""

  marker: str
  paths: tuple[str, ...] = ()
  ignores: tuple[str, ...] = ()


# The tpu7x-tests matrix of ci_pipeline.yml, in order, with the settings of run_tests_coordinator.yml.
TPU7X_FLAVORS = {
    "tpu7x-unit": Flavor(
        marker="not cpu_only and not gpu_only and not integration_test and not post_training",
        ignores=("tests/post_training",),
    ),
    "tpu7x-integration": Flavor(
        marker="not cpu_only and not gpu_only and integration_test and not post_training",
        ignores=("tests/post_training",),
    ),
    "tpu7x-post-training-unit": Flavor(
        marker="not cpu_only and not gpu_only and not integration_test and post_training",
        paths=("tests/post_training/unit", "tests/unit"),
    ),
}


def _marker_matches(expression, markers):
  """Return whether ``markers`` satisfy a ``pytest -m`` expression of the form ``[not] a and [not] b ...``.

  Raises:
    ValueError: The expression has another form (``or``, parentheses), which no TPU7X flavor uses.
  """
  for term in expression.split(" and "):
    name = term.removeprefix("not ")
    if not name.isidentifier():
      raise ValueError(f"unsupported marker expression: {expression!r}")
    if (name in markers) == term.startswith("not "):
      return False
  return True


def _read_pytest_ini(path="pytest.ini"):
  """Return pytest.ini's ``testpaths`` and the ``--ignore=`` paths of its ``addopts``."""
  parser = configparser.ConfigParser(interpolation=None)
  try:
    parser.read(path, encoding="utf-8")
    testpaths = tuple(parser.get("pytest", "testpaths", fallback="tests").split())
    addopts = shlex.split(parser.get("pytest", "addopts", fallback=""))
  except (configparser.Error, ValueError):
    return ("tests",), ()
  return testpaths, tuple(arg.removeprefix("--ignore=") for arg in addopts if arg.startswith("--ignore="))


def _is_under(path, roots):
  """Return whether ``path`` is one of ``roots`` or inside one of them."""
  return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def _is_eligible(flavor, ini, path, markers):
  """Return whether ``flavor`` collects and runs a test with the static ``markers`` in file ``path``.

  ``ini`` is what ``_read_pytest_ini`` returns.
  """
  testpaths, ini_ignores = ini
  if "skip" in markers or "skip_on_tpu7x" in markers:  # Skipped (on TPU7X) at run time.
    return False
  if not _is_under(path, flavor.paths or testpaths) or _is_under(path, ini_ignores + flavor.ignores):
    return False
  if markers.isdisjoint(_HARDWARE_MARKERS):
    markers = markers | {"cpu_only"}
  return _marker_matches(flavor.marker, markers)


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


def _marker_names(nodes):
  """Return the names of the ``pytest.mark.<name>`` markers, called or not, among decorators or list items."""
  names = set()
  for node in nodes:
    target = node.func if isinstance(node, ast.Call) else node
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "pytest"
    ):
      names.add(target.attr)
  return names


def _pytestmark(statements):
  """Return the marker names and the lines of the ``pytestmark = ...`` assignments among ``statements``."""
  markers, lines = set(), set()
  for node in statements:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
      values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
      markers |= _marker_names(values)
      lines.update(range(node.lineno, node.end_lineno + 1))
  return markers, lines


def _iter_test_defs(tree):
  """Yield ``(name, lines, markers)`` for every test function in an AST.

  Covers module-level test functions and methods declared directly inside a class,
  which is what pytest collects. ``lines`` are the lines whose edit changes the test: its
  span, which starts at the first decorator (if any) so decorator-only edits are attributed
  to the test they decorate, the decorators and header of its class, and the ``pytestmark``
  assignments of its class and module. ``markers`` are its static marker names.
  """
  module_markers, module_lines = _pytestmark(tree.body)
  for node in tree.body:
    if isinstance(node, _DEF_TYPES) and node.name.startswith("test"):
      yield node.name, module_lines | _span(node), frozenset(module_markers | _marker_names(node.decorator_list))
    elif isinstance(node, ast.ClassDef):
      class_markers, class_lines = _pytestmark(node.body)
      class_markers |= module_markers | _marker_names(node.decorator_list)
      # The class decorators and header: from the first decorator to the line before the first body statement.
      class_lines |= module_lines | set(range(_span_start(node), max(node.lineno, _span_start(node.body[0]) - 1) + 1))
      for sub in node.body:
        if isinstance(sub, _DEF_TYPES) and sub.name.startswith("test"):
          yield sub.name, class_lines | _span(sub), frozenset(class_markers | _marker_names(sub.decorator_list))


def _span_start(node):
  """Return the first source line of a def or class, including any decorator lines above it."""
  start = node.lineno
  if getattr(node, "decorator_list", None):
    start = min(start, min(dec.lineno for dec in node.decorator_list))
  return start


def _span(node):
  """Return the lines of a def, from its first decorator to its last line."""
  return set(range(_span_start(node), node.end_lineno + 1))


def touched_tests(source, touched_lines):
  """Map each test function in ``source`` that a changed line touches to its static marker names.

  Args:
    source: The new file's Python source.
    touched_lines: Set of changed new-file line numbers for that file.

  Returns:
    A dict of ``{test_name: frozenset_of_marker_names}``; tests of the same name in different classes share
    one entry. Empty if ``touched_lines`` is empty or ``source`` does not parse (pytest collection surfaces a
    syntax error on its own, so raising here would only hide it).
  """
  if not touched_lines:
    return {}
  try:
    tree = ast.parse(source)
  except SyntaxError:
    return {}
  found = {}
  for name, lines, markers in _iter_test_defs(tree):
    if not lines.isdisjoint(touched_lines):
      found[name] = found.get(name, frozenset()) | markers
  return found


def touched_test_names(source, touched_lines):
  """Return the names of the test functions in ``source`` that a changed line touches; see ``touched_tests``."""
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


def _diff_against_base(base):
  """Return the ``git diff --unified=0`` of HEAD against its merge-base with ``base``, or None if it fails."""
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
        return subprocess.check_output(command, encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL)
      except Exception:  # pylint: disable=broad-exception-caught
        continue
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  return None


def _touched_tests_in_diff(diff_text):
  """Return ``{(file_path, test_name): markers}`` for the tests ``diff_text`` touched, read from the cwd."""
  touched = {}
  for path, touched_lines in parse_changed_line_map(diff_text).items():
    if not _is_test_file(path):
      continue
    try:
      with open(path, "r", encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    except OSError:
      continue
    for name, markers in touched_tests(source, touched_lines).items():
      touched[(path, name)] = markers
  return touched


def get_changed_tests(base_ref=None):
  """Return ``(file_path, test_name)`` for every test the PR added or modified.

  Args:
    base_ref: Base git ref to diff against. Defaults to the ``GITHUB_BASE_REF``
      environment variable, then ``"main"``.

  Returns:
    A set of ``(file_path, test_name)`` tuples, or an empty set when not in a git work
    tree or when the diff cannot be computed.
  """
  diff_text = _diff_against_base(base_ref or os.environ.get("GITHUB_BASE_REF") or "main")
  return set() if diff_text is None else set(_touched_tests_in_diff(diff_text))


def is_changed_item(item, changed_tests):
  """Return whether a pytest item is one of ``changed_tests`` (``(file_path, test_name)`` pairs).

  The item's name is ``item.originalname`` (without pytest's ``[params]`` suffix), ``item.name`` or the
  name of the function it wraps. The last one is the def name of an absl ``parameterized`` test, which
  pytest collects under generated names such as ``test_x_bf16`` or ``test_x0``.
  """
  if not changed_tests:
    return False
  names = {getattr(item, "originalname", None), item.name}
  function = getattr(item, "function", None)
  if function is not None:
    names.add(getattr(inspect.unwrap(function), "__name__", None))
  item_file = item.nodeid.split("::", 1)[0]
  return any((item_file, name) in changed_tests for name in names if name)


def _tests_run_by(flavor, touched, ini):
  """Return the sorted ``path::test_name`` of the ``touched`` tests that a TPU7X flavor runs."""
  spec = TPU7X_FLAVORS[flavor]
  return sorted(f"{path}::{name}" for (path, name), markers in touched.items() if _is_eligible(spec, ini, path, markers))


def main(argv=None):
  """Command-line entry point; see the module docstring."""
  parser = argparse.ArgumentParser(description="List the tests that a branch added or modified relative to a base ref.")
  parser.add_argument("--base", help="Base ref to diff against (default: $GITHUB_BASE_REF, then main).")
  scope = parser.add_mutually_exclusive_group()
  scope.add_argument("--flavor", choices=TPU7X_FLAVORS, help="Only list the tests that this TPU7X flavor runs.")
  scope.add_argument(
      "--eligible-flavors",
      action="store_true",
      help="Print a JSON array of the TPU7X flavors that run a listed test, and those tests on stderr.",
  )
  args = parser.parse_args(argv)
  base = args.base or os.environ.get("GITHUB_BASE_REF") or "main"
  diff_text = _diff_against_base(base)
  if diff_text is None:
    print(f"newly_added_detection: cannot diff against origin/{base} or {base}", file=sys.stderr)
    return 2
  touched = _touched_tests_in_diff(diff_text)
  ini = _read_pytest_ini()
  if args.eligible_flavors:
    eligible = []
    for flavor in TPU7X_FLAVORS:
      tests = _tests_run_by(flavor, touched, ini)
      for test in tests:
        print(f"{flavor}: {test}", file=sys.stderr)
      if tests:
        eligible.append(flavor)
    print(json.dumps(eligible, separators=(",", ":")))
    return 0
  tests = _tests_run_by(args.flavor, touched, ini) if args.flavor else sorted(f"{p}::{n}" for p, n in touched)
  for test in tests:
    print(test)
  return 0


if __name__ == "__main__":
  sys.exit(main())
