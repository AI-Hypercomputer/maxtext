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
request runs ``pytest -m "<flavor marker> and newly_added"`` instead, and only on the TPU7X
flavors that collect at least one touched test, so only those tests occupy that scarce
hardware (unless the pull request is labeled ``run-tpu7x`` or ``scheduled-only``, which run
every flavor of ``TPU7X_PR_MATRIX``). ``tests/conftest.py`` adds the marker through
``is_changed_item``.

What counts as touched
----------------------
Detection maps the line numbers changed by ``git diff --unified=0`` (from the merge-base
with the base branch to ``HEAD``, limited to ``tests/``) onto an ``ast`` parse of each test
file. It never trusts git's hunk-header function name, which points at the function
*preceding* an insertion. A test is a ``def`` or ``async def`` whose name starts with
``test`` (pytest's default ``python_functions``), at module level or directly in a class
body, that pytest can collect: not in a module or class that sets ``__test__ = False``, and,
for a class with no base other than ``object``, only if the class is named ``Test*`` and
defines no ``__init__`` or ``__new__``. A class that another class in the same file
subclasses is always kept, because the subclass may collect the tests it inherits. A test T
named N is touched when any of these holds:

(a) An added or changed line falls in T's span: its first decorator through its last line.
(b) A deleted line falls in the span of a test also named N, in the same class (or also at
    module level), in the base version of the file. The base version is the blob that
    ``git diff --full-index`` names, read with ``git cat-file``, so renames are followed.
(c) An added, changed or deleted line falls in the decorators of T's class, its header
    (``class ...:`` through the line before its first body statement) or a class-body
    ``pytestmark`` or ``__test__`` assignment. Every test in that class is then touched.
(d) An added, changed or deleted line falls in a module-level ``pytestmark`` or ``__test__``
    assignment. Every test in the module is then touched.

A test that exists only in the base version (a deleted test) yields nothing, and its
untouched neighbours are not flagged. If the base version cannot be read or parsed, a pure
deletion falls back to the two new-file lines around the cut, which may flag a neighbour.

Edits to these never count as touched: helpers, fixtures, ``setUp``/``tearDown`` and any
other non-test method, other class attributes, module code other than ``pytestmark``
(imports, constants), ``conftest.py``, anything under ``src/``, and ``pytest.ini``. Nor does
renaming or moving a test file without changing its lines. A pull request that only makes
such changes runs no test on TPU7X unless labeled ``run-tpu7x`` or ``scheduled-only``.

Markers and TPU7X flavors
-------------------------
A test's static markers are the ``@pytest.mark.<name>`` and ``@pytest.mark.<name>(...)``
decorators on the function and on its class, plus the class-body and module-level
``pytestmark`` (a single marker, a list or a tuple). Markers pytest would see but this
parser does not: aliases (``from pytest import mark``, ``tpu = pytest.mark.tpu_only``),
markers inherited from a base class, ``pytest.param(..., marks=...)``, and markers added by
hooks other than the ``cpu_only`` auto-marking described below.

A test is eligible for a TPU7X flavor (``TPU7X_FLAVORS``, which mirrors the flavor maps of
``.github/workflows/run_tests_coordinator.yml``) when all of these hold:

1. The flavor collects its file: the file is under the flavor's paths (pytest.ini
   ``testpaths`` when it has none) and not under any ``--ignore`` of pytest.ini or of the
   flavor.
2. Its markers match the flavor's ``-m`` expression, after adding ``newly_added`` and, when
   it has no hardware marker, ``cpu_only`` (as ``tests/conftest.py`` does).
3. It is not marked ``skip_on_tpu7x`` (skipped at runtime on TPU7X) or ``skip``.

Command line
------------
The module doubles as a command-line tool for CI jobs that cannot import the ``tests``
package (``tests/__init__.py`` imports packages a bare runner does not have)::

    python3 tests/utils/newly_added_detection.py [--base REF] [--require-marker M] [--exclude-marker M]
        [--flavor NAME | --eligible-flavors] [--all-tests]

Run it from the repository root. By default it prints one ``path::test_name`` line per
touched test, sorted. ``--require-marker`` and ``--exclude-marker`` filter on static
markers. ``--flavor NAME`` keeps only the tests eligible for that TPU7X flavor.
``--eligible-flavors`` prints a single compact JSON array instead: the flavors of
``TPU7X_PR_MATRIX`` that have at least one eligible test, in matrix order (for example
``["tpu7x-unit"]`` or ``[]``), with one ``<flavor>: path::test_name`` line per eligible test
on stderr. ``--all-tests`` treats every test under ``tests/`` as touched and runs no git
command, for verification and debugging.

Exit status: 0 when a diff was computed (or with ``--all-tests``), even if nothing is
listed; 2 when no diff against the base could be computed (the message says "cannot
diff"), so callers can tell "nothing changed" from "detection is broken"; 64 for usage
errors.

Only the Python standard library is used, since this runs on a bare CI runner where
MaxText is not necessarily importable.
"""

import argparse
import ast
import configparser
import dataclasses
import functools
import inspect
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys

# Matches pytest.ini's ``python_files = *_test.py *_tests.py``.
_TEST_SUFFIXES = ("_test.py", "_tests.py")
# Matches pytest.ini's ``testpaths = tests``.
_TESTS_ROOT = "tests/"
# Captures both sides of a unified-diff hunk header; a missing length means 1:
#   @@ -<old_start>[,<old_len>] +<new_start>[,<new_len>] @@
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# Captures the full blob ids of a ``git diff --full-index`` line ``index <old>..<new>[ <mode>]``.
_INDEX_RE = re.compile(r"^index ([0-9a-f]+)\.\.([0-9a-f]+)")
# The diff options, chosen so that the output does not depend on the user's git config
# (color.ui, diff.external, textconv drivers, diff.noprefix, diff.mnemonicPrefix,
# diff.renames, diff.interHunkContext, core.quotePath).
_DIFF_COMMAND = (
    "git",
    "-c",
    "core.quotePath=false",
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--unified=0",
    "--inter-hunk-context=0",
    "-M",
    "--full-index",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)
# GIT_DIFF_OPTS overrides --unified on the command line, so it is dropped from git's environment.
_GIT_ENV_DROP = ("GIT_DIFF_OPTS",)
# Escapes git uses in C-quoted paths (core.quotePath=false still quotes '"', '\' and control characters).
_C_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}
# An identifier in a pytest ``-m`` expression, as pytest's own lexer defines it.
_MARKER_IDENT_RE = re.compile(r"(?:\w|[:+\-.\[\]\\/])+")
_MARKER_KEYWORDS = frozenset({"and", "or", "not"})
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
# tests/conftest.py marks a test without one of these ``cpu_only``.
_HARDWARE_MARKERS = frozenset({"tpu_only", "gpu_only", "cpu_only"})
_EXIT_CANNOT_DIFF = 2
# sysexits.h EX_USAGE: usage errors must never look like 2 ("cannot diff").
_EXIT_USAGE = 64


@dataclasses.dataclass(frozen=True)
class Flavor:
  """How a TPU7X flavor selects tests, as run_tests_coordinator.yml configures it.

  Attributes:
    marker: The ``pytest -m`` expression (the ``pytest_marker`` map).
    paths: The paths passed to pytest (the ``pytest_addopts`` map). Empty means pytest.ini ``testpaths``.
    ignores: The flavor's own ``--ignore`` paths (the ``pytest_extra_args`` map), on top of pytest.ini's.
  """

  marker: str
  paths: tuple[str, ...] = ()
  ignores: tuple[str, ...] = ()


# Mirrors the pytest_marker / pytest_addopts / pytest_extra_args maps in run_tests_coordinator.yml.
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
    "tpu7x-post-training-integration": Flavor(
        marker="not cpu_only and not gpu_only and integration_test and post_training",
        paths=("tests/post_training/integration",),
    ),
}
# The tpu7x-tests matrix of ci_pipeline.yml, in its order. tpu7x-post-training-integration is not in it.
TPU7X_PR_MATRIX = ("tpu7x-unit", "tpu7x-integration", "tpu7x-post-training-unit")


# --- pytest marker expressions -------------------------------------------------------------------------------------


def _tokenize_marker_expression(expression):
  """Split a ``pytest -m`` expression into parentheses, keywords and identifiers.

  Like pytest's own lexer, only spaces and tabs separate tokens: pytest rejects any other whitespace, such as a
  newline, and so does this function.
  """
  tokens = []
  pos = 0
  while pos < len(expression):
    char = expression[pos]
    if char in " \t":
      pos += 1
    elif char in "()":
      tokens.append(char)
      pos += 1
    else:
      match = _MARKER_IDENT_RE.match(expression, pos)
      if match is None:
        raise ValueError(f"invalid marker expression {expression!r}: unexpected character {char!r} at column {pos + 1}")
      tokens.append(match.group())
      pos = match.end()
  return tokens


class _MarkerExpressionParser:
  """Recursive-descent parser for pytest's ``-m`` grammar, producing a predicate over marker sets.

  The grammar, with the usual precedence (``not`` binds tightest, then ``and``, then ``or``)::

      expression: expr? EOF
      expr:       and_expr ('or' and_expr)*
      and_expr:   not_expr ('and' not_expr)*
      not_expr:   'not' not_expr | '(' expr ')' | ident
  """

  def __init__(self, expression):
    self._expression = expression
    self._tokens = _tokenize_marker_expression(expression)
    self._pos = 0

  def parse(self):
    """Return the predicate for the whole expression; an empty expression matches everything."""
    if not self._tokens:
      return lambda markers: True
    predicate = self._expr()
    if self._pos != len(self._tokens):
      self._fail(f"unexpected {self._tokens[self._pos]!r}")
    return predicate

  def _fail(self, reason):
    raise ValueError(f"invalid marker expression {self._expression!r}: {reason}")

  def _peek(self):
    return self._tokens[self._pos] if self._pos < len(self._tokens) else None

  def _accept(self, token):
    if self._peek() == token:
      self._pos += 1
      return True
    return False

  def _expr(self):
    terms = [self._and_expr()]
    while self._accept("or"):
      terms.append(self._and_expr())
    if len(terms) == 1:
      return terms[0]
    return lambda markers: any(term(markers) for term in terms)

  def _and_expr(self):
    terms = [self._not_expr()]
    while self._accept("and"):
      terms.append(self._not_expr())
    if len(terms) == 1:
      return terms[0]
    return lambda markers: all(term(markers) for term in terms)

  def _not_expr(self):
    """Parse ``'not' not_expr | '(' expr ')' | ident``."""
    if self._accept("not"):
      operand = self._not_expr()
      return lambda markers: not operand(markers)
    if self._accept("("):
      inner = self._expr()
      if not self._accept(")"):
        self._fail("expected ')'")
      return inner
    token = self._peek()
    if token is None:
      self._fail("unexpected end of expression")
    if token in _MARKER_KEYWORDS or token in ("(", ")"):
      self._fail(f"unexpected {token!r}")
    self._pos += 1
    return lambda markers: token in markers


@functools.lru_cache(maxsize=None)
def _compile_marker_expression(expression):
  return _MarkerExpressionParser(expression).parse()


def marker_expression_matches(expression, markers):
  """Return whether a test carrying ``markers`` is selected by ``pytest -m expression``.

  Implements pytest's ``-m`` grammar: marker names combined with ``and``, ``or``, ``not`` and
  parentheses. An empty expression selects every test, as ``pytest -m ""`` does.

  Args:
    expression: A ``pytest -m`` expression, e.g. ``"not cpu_only and (tpu_only or newly_added)"``.
    markers: The names of the markers the test carries.

  Raises:
    ValueError: ``expression`` is malformed.
  """
  return _compile_marker_expression(expression)(frozenset(markers))


# --- pytest.ini and flavor eligibility -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _PytestIni:
  """The parts of pytest.ini that decide which files pytest collects."""

  testpaths: tuple[str, ...] = ("tests",)
  ignores: tuple[str, ...] = ()


def _read_pytest_ini(path="pytest.ini"):
  """Read ``testpaths`` and the ``--ignore`` options of ``addopts`` from pytest.ini.

  A missing or unreadable file gives the defaults: testpaths ``("tests",)`` and no ignores.
  """
  parser = configparser.ConfigParser(interpolation=None, strict=False)
  try:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
      parser.read_file(handle)
  except (OSError, configparser.Error):
    return _PytestIni()
  if not parser.has_section("pytest"):
    return _PytestIni()
  testpaths = tuple(parser.get("pytest", "testpaths", fallback="").split()) or ("tests",)
  addopts = parser.get("pytest", "addopts", fallback="")
  try:
    args = shlex.split(addopts)
  except ValueError:
    args = addopts.split()
  ignores = []
  for index, arg in enumerate(args):
    if arg.startswith("--ignore="):
      ignores.append(arg[len("--ignore=") :])
    elif arg == "--ignore" and index + 1 < len(args):
      ignores.append(args[index + 1])
  return _PytestIni(testpaths=testpaths, ignores=tuple(ignores))


def _normalize_path(path):
  """Return ``path`` as a normalized, repo-root-relative POSIX path ("" for the root itself)."""
  normalized = posixpath.normpath(path.strip().replace(os.sep, "/"))
  return "" if normalized == "." else normalized


def _is_under(path, roots):
  """Return whether ``path`` is one of ``roots`` or inside one of them (whole path components only)."""
  path = _normalize_path(path)
  for root in roots:
    root = _normalize_path(root)
    if not root or path == root or path.startswith(root + "/"):
      return True
  return False


def _is_eligible(flavor, ini, path, markers):
  """Return whether a test with static ``markers`` in file ``path`` is collected and run by ``flavor`` on TPU7X."""
  if "skip_on_tpu7x" in markers or "skip" in markers:
    return False
  if not _is_under(path, flavor.paths or ini.testpaths) or _is_under(path, ini.ignores + flavor.ignores):
    return False
  effective = set(markers) | {"newly_added"}
  if effective.isdisjoint(_HARDWARE_MARKERS):
    effective.add("cpu_only")
  return marker_expression_matches(flavor.marker, effective)


# --- Test files: tests, markers and the lines that touch them ------------------------------------------------------


def _is_test_file(path):
  """Return True if ``path`` is a MaxText test file (under tests/ with a test suffix)."""
  normalized = path.strip().replace(os.sep, "/")
  return normalized.startswith(_TESTS_ROOT) and normalized.endswith(_TEST_SUFFIXES)


def _is_test_name(name):
  """Return whether pytest collects a function with this name (``python_functions = test``)."""
  return name.startswith("test")


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


def _span_start(node):
  """Return the first source line of a statement, including any decorator lines above it."""
  start = node.lineno
  for dec in getattr(node, "decorator_list", ()):
    start = min(start, dec.lineno)
  return start


def _lines(node):
  """Return the set of source lines a statement spans."""
  return set(range(node.lineno, node.end_lineno + 1))


def _assigns(node, name):
  """Return whether a statement is an assignment to the plain name ``name``."""
  if isinstance(node, ast.Assign):
    return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
  return isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name


def _pytestmark_markers(value):
  """Return the marker names in a ``pytestmark`` value: a single marker, or a list or tuple of them."""
  elements = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
  return _marker_names(elements)


def _collection_scope(statements):
  """Scan the statements of a module or class body for what decides how pytest collects its tests.

  Returns:
    ``(markers, lines, disabled)``: the marker names of its ``pytestmark`` assignments, the lines of its
    ``pytestmark`` and ``__test__`` assignments, and whether it sets ``__test__`` to a false constant (then pytest
    collects no test from it).
  """
  markers = set()
  lines = set()
  disabled = False
  for node in statements:
    if _assigns(node, "pytestmark"):
      if node.value is not None:
        markers |= _pytestmark_markers(node.value)
      lines |= _lines(node)
    elif _assigns(node, "__test__"):
      if isinstance(node.value, ast.Constant) and not node.value.value:
        disabled = True
      lines |= _lines(node)
  return markers, lines, disabled


def _base_name(base):
  """Return the name a class base refers to when it is a plain (or subscripted) name, else None."""
  if isinstance(base, ast.Subscript):
    base = base.value
  return base.id if isinstance(base, ast.Name) else None


def _uncollectable_classes(classes):
  """Return the names of the module-level classes whose test methods pytest never collects.

  A class with a base other than ``object`` may be a ``unittest.TestCase`` (the base may be imported), which pytest
  collects whatever its name; it is dropped only when it sets ``__test__ = False``. A class with no such base is
  collected only if it is named ``Test*`` and defines neither ``__init__`` nor ``__new__``. A class that another class
  in the file subclasses is always kept, since the subclass may collect the tests it inherits.
  """
  subclassed = {_base_name(base) for node in classes for base in node.bases}
  uncollectable = set()
  for node in classes:
    if node.name in subclassed:
      continue
    if _collection_scope(node.body)[2]:
      uncollectable.add(node.name)
    elif all(_base_name(base) == "object" for base in node.bases):
      constructors = {sub.name for sub in node.body if isinstance(sub, _DEF_TYPES)} & {"__init__", "__new__"}
      if not node.name.startswith("Test") or constructors:
        uncollectable.add(node.name)
  return uncollectable


@dataclasses.dataclass(frozen=True)
class _TestDef:
  """A test function in one version of a file.

  Attributes:
    name: The function name.
    cls: The name of the enclosing class, or None at module level.
    start: The first line of the span (the first decorator, if any).
    end: The last line of the span.
    markers: The static pytest marker names the test carries.
  """

  name: str
  cls: str | None
  start: int
  end: int
  markers: frozenset[str]

  def overlaps(self, lines):
    """Return whether any of ``lines`` falls in this test's span."""
    return any(self.start <= line <= self.end for line in lines)


@dataclasses.dataclass(frozen=True)
class _TestModule:
  """The tests of one version of a test file, and the lines whose edits touch groups of them.

  Attributes:
    tests: Every test pytest may collect, in source order.
    class_lines: ``(class_name, lines)`` per class: its decorators, its header, and its class-body ``pytestmark``
      and ``__test__`` assignments. An edit there touches every test in the class.
    module_lines: The lines of the module-level ``pytestmark`` and ``__test__`` assignments. An edit there touches
      every test in the module.
  """

  tests: tuple[_TestDef, ...]
  class_lines: tuple[tuple[str, frozenset[int]], ...]
  module_lines: frozenset[int]


def _parse_test_module(source):
  """Return the ``_TestModule`` of a test file's source, or None if ``source`` is None or does not parse.

  pytest collection surfaces a syntax error on its own, so a file that does not parse contributes nothing here.
  """
  if source is None:
    return None
  try:
    tree = ast.parse(source)
  except (SyntaxError, ValueError, RecursionError):  # ValueError: null bytes, before Python 3.12.
    return None
  module_markers, module_lines, module_disabled = _collection_scope(tree.body)
  if module_disabled:
    return _TestModule((), (), frozenset(module_lines))
  uncollectable = _uncollectable_classes([node for node in tree.body if isinstance(node, ast.ClassDef)])
  tests = []
  class_lines = []
  for node in tree.body:
    if isinstance(node, _DEF_TYPES) and _is_test_name(node.name):
      markers = frozenset(module_markers | _marker_names(node.decorator_list))
      tests.append(_TestDef(node.name, None, _span_start(node), node.end_lineno, markers))
    elif isinstance(node, ast.ClassDef):
      body_markers, lines, _ = _collection_scope(node.body)
      # Decorators and header: the first decorator through the line before the first body statement.
      lines |= set(range(_span_start(node), max(node.lineno, _span_start(node.body[0]) - 1) + 1))
      class_lines.append((node.name, frozenset(lines)))
      if node.name in uncollectable:
        continue
      class_markers = module_markers | _marker_names(node.decorator_list) | body_markers
      for sub in node.body:
        if isinstance(sub, _DEF_TYPES) and _is_test_name(sub.name):
          markers = frozenset(class_markers | _marker_names(sub.decorator_list))
          tests.append(_TestDef(sub.name, node.name, _span_start(sub), sub.end_lineno, markers))
  return _TestModule(tuple(tests), tuple(class_lines), frozenset(module_lines))


def _touched_defs(new, added, old=None, deleted=frozenset()):
  """Return the tests of ``new`` that rules (a)-(d) of the module docstring mark as touched.

  Args:
    new: The ``_TestModule`` of the new version of the file.
    added: Added or changed new-file line numbers.
    old: The ``_TestModule`` of the base version, or None when it is unavailable.
    deleted: Deleted or changed base-file line numbers. Ignored when ``old`` is None.
  """
  touched = set()
  if added:
    touched.update(test for test in new.tests if test.overlaps(added))  # (a)
    for cls, lines in new.class_lines:  # (c)
      if not lines.isdisjoint(added):
        touched.update(test for test in new.tests if test.cls == cls)
    if not new.module_lines.isdisjoint(added):  # (d)
      touched.update(new.tests)
  if old is not None and deleted:
    for old_test in old.tests:  # (b)
      if old_test.overlaps(deleted):
        touched.update(test for test in new.tests if (test.cls, test.name) == (old_test.cls, old_test.name))
    for cls, lines in old.class_lines:  # (c)
      if not lines.isdisjoint(deleted):
        touched.update(test for test in new.tests if test.cls == cls)
    if not old.module_lines.isdisjoint(deleted):  # (d)
      touched.update(new.tests)
  return touched


def touched_tests(source, touched_lines, base_source=None, deleted_lines=None):
  """Map each touched test in ``source`` to its static marker names.

  Args:
    source: The new file's Python source.
    touched_lines: Added or changed new-file line numbers for that file.
    base_source: Optional source of the base version of the file.
    deleted_lines: Optional deleted or changed base-file line numbers. Used only with ``base_source``.

  Returns:
    ``{test_name: frozenset_of_marker_names}`` for the tests touched under rules (a)-(d) of the module
    docstring (rules (b) and the deletion half of (c) and (d) need ``base_source``). Tests of the same name in
    different classes share one entry with the union of their markers. Empty if nothing changed or ``source``
    does not parse (pytest collection surfaces a syntax error on its own, so raising here would only hide it).
  """
  added = frozenset(touched_lines or ())
  deleted = frozenset(deleted_lines or ())
  if not added and not deleted:
    return {}
  new = _parse_test_module(source)
  if new is None:
    return {}
  old = _parse_test_module(base_source) if deleted else None
  found = {}
  for test in _touched_defs(new, added, old, deleted):
    found[test.name] = found.get(test.name, frozenset()) | test.markers
  return found


def touched_test_names(source, touched_lines, base_source=None, deleted_lines=None):
  """Return the names of the touched tests in ``source``; see ``touched_tests``."""
  return set(touched_tests(source, touched_lines, base_source, deleted_lines))


# --- git diff ------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _FileDiff:
  """The changed lines of one file in a ``git diff --unified=0``.

  Attributes:
    new_path: Repo-root-relative path in the new version, or None for a deleted file.
    old_blob: Blob id of the base version (``--full-index``), or None for an added file or when not shown.
    hunks: Number of hunks.
    added: Added or changed new-file line numbers.
    deleted: Deleted or changed base-file line numbers.
    boundary: For pure-deletion hunks, the two new-file lines around the cut. Used only when the base version
      cannot be read.
    saw_new_header: Whether the ``+++`` line was seen.
  """

  new_path: str | None = None
  old_blob: str | None = None
  hunks: int = 0
  added: set[int] = dataclasses.field(default_factory=set)
  deleted: set[int] = dataclasses.field(default_factory=set)
  boundary: set[int] = dataclasses.field(default_factory=set)
  saw_new_header: bool = False


def _unquote_c_path(path):
  """Undo git's C-style quoting of a path (``"tests/a\\"b_test.py"``); other paths are returned unchanged."""
  if len(path) < 2 or not path.startswith('"') or not path.endswith('"'):
    return path
  body = path[1:-1]
  raw = bytearray()
  index = 0
  while index < len(body):
    char = body[index]
    if char == "\\" and index + 1 < len(body):
      escaped = body[index + 1]
      octal = body[index + 1 : index + 4]
      if len(octal) == 3 and all(c in "01234567" for c in octal):
        raw.append(int(octal, 8) & 0xFF)
        index += 4
        continue
      if escaped in _C_ESCAPES:
        raw.append(_C_ESCAPES[escaped])
        index += 2
        continue
    raw.extend(char.encode("utf-8", errors="replace"))
    index += 1
  return raw.decode("utf-8", errors="replace")


def _diff_header_path(value, prefix):
  """Return the path of a ``---``/``+++`` header value without its prefix, or None for ``/dev/null``."""
  if value.endswith("\t"):  # git appends a tab to a name that contains a space.
    value = value[:-1]
  if value == "/dev/null":
    return None
  path = _unquote_c_path(value)
  return path[len(prefix) :] if path.startswith(prefix) else path


def _parse_diff(diff_text):
  """Parse ``git diff --unified=0`` output into ``{new_path: _FileDiff}`` for every file with at least one hunk.

  Hunk bodies are consumed by count, so a content line that starts with ``+++`` or ``---`` is never mistaken for a
  file header. Context lines (should a caller pass a diff with context) are left out of the changed lines.
  """
  files = []
  current = None
  old_left = new_left = 0  # Body lines still expected from the current hunk.
  old_line = new_line = 0  # Line numbers of the next body line on each side.
  for line in diff_text.split("\n"):
    if line.endswith("\r"):
      line = line[:-1]
    if old_left > 0 or new_left > 0:
      kind = line[:1]
      if kind == "+" and new_left > 0:
        new_line += 1
        new_left -= 1
        continue
      if kind == "-" and old_left > 0:
        old_line += 1
        old_left -= 1
        continue
      if kind == " " and old_left > 0 and new_left > 0:
        if current is not None:
          current.added.discard(new_line)
          current.deleted.discard(old_line)
        old_line += 1
        new_line += 1
        old_left -= 1
        new_left -= 1
        continue
      if kind == "\\":  # "\ No newline at end of file"
        continue
      old_left = new_left = 0  # A truncated hunk body: read this line as a header.
    if line.startswith("diff --git "):
      current = _FileDiff()
      files.append(current)
    elif line.startswith("index "):
      if current is None or current.hunks:
        current = _FileDiff()
        files.append(current)
      match = _INDEX_RE.match(line)
      if match is not None and match.group(1).strip("0"):
        current.old_blob = match.group(1)
    elif line.startswith("--- "):
      if current is None or current.hunks or current.saw_new_header:
        current = _FileDiff()
        files.append(current)
    elif line.startswith("+++ "):
      if current is None or current.hunks or current.saw_new_header:
        current = _FileDiff()
        files.append(current)
      current.new_path = _diff_header_path(line[4:], "b/")
      current.saw_new_header = True
    elif line.startswith("@@"):
      match = _HUNK_RE.match(line)
      if match is None:
        continue
      old_start = int(match.group(1))
      old_len = int(match.group(2)) if match.group(2) is not None else 1
      new_start = int(match.group(3))
      new_len = int(match.group(4)) if match.group(4) is not None else 1
      old_left, new_left, old_line, new_line = old_len, new_len, old_start, new_start
      if current is None:
        continue
      current.hunks += 1
      current.added.update(range(new_start, new_start + new_len))
      current.deleted.update(range(old_start, old_start + old_len))
      if new_len == 0:
        current.boundary.update({new_start, new_start + 1})
  by_path = {}
  for file_diff in files:
    if file_diff.new_path is None or not file_diff.hunks:
      continue
    merged = by_path.setdefault(file_diff.new_path, _FileDiff(new_path=file_diff.new_path))
    merged.old_blob = merged.old_blob or file_diff.old_blob
    merged.hunks += file_diff.hunks
    merged.added |= file_diff.added
    merged.deleted |= file_diff.deleted
    merged.boundary |= file_diff.boundary
  return by_path


def parse_changed_line_map(diff_text):
  """Map each changed file to the set of new-file line numbers it touched.

  Args:
    diff_text: Raw ``git diff --unified=0`` output.

  Returns:
    A dict of ``{file_path: set_of_new_line_numbers}``. ``file_path`` is repo-root
    relative (the ``b/`` prefix is stripped). For a pure deletion (new length 0) the
    two lines bracketing the removal are recorded: that is the fallback used when the base
    version of a file cannot be read (``changed_tests_from_diff`` otherwise attributes deleted
    lines on the base side). Deleted files (``+++ /dev/null``) are omitted.
  """
  return {path: file_diff.added | file_diff.boundary for path, file_diff in _parse_diff(diff_text).items()}


def _git_env():
  """Return the environment for git commands: the current one minus variables that change diff output."""
  return {key: value for key, value in os.environ.items() if key not in _GIT_ENV_DROP}


def _read_text(path):
  """Return a file's text (UTF-8, undecodable bytes replaced), or None if it cannot be read."""
  try:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
      return handle.read()
  except (OSError, ValueError):
    return None


def _read_blob(blob):
  """Return the text of a git blob (UTF-8, undecodable bytes replaced), or None if it cannot be read."""
  if not blob:
    return None
  try:
    result = subprocess.run(
        ["git", "cat-file", "blob", blob],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        env=_git_env(),
        check=False,
    )
  except OSError:
    return None
  if result.returncode != 0:
    return None
  return result.stdout.decode("utf-8-sig", errors="replace")


def _build_diff_commands(base):
  """Return the ordered ``git diff`` argument lists to try, most-precise first.

  Both ranges are three-dot (merge-base) ranges, so only the branch's own commits
  are reported. Two-dot tip-vs-tip ranges are deliberately excluded: they over-report
  every commit the base gained past the fork point (a stale local ``main`` can inflate
  the changed set many-fold), which would drag unrelated ``scheduled_only`` tests into
  pre-submit. The remote range covers CI and local checkouts that have an ``origin``
  remote; the local range is the fallback for a developer without ``origin`` and is
  never reached in CI, where ``origin/<base>`` is always fetched first.

  Each command is ``[*_DIFF_COMMAND, "<ref>...HEAD", "--", "tests/"]``: its options keep the
  output independent of the user's git config, and the ``tests/`` pathspec keeps files that
  can never hold a test out of the diff.
  """
  return [[*_DIFF_COMMAND, f"{ref}...HEAD", "--", _TESTS_ROOT] for ref in (f"origin/{base}", base)]


def _resolve_base(base_ref=None):
  """Return the base ref to diff against: ``base_ref``, else ``$GITHUB_BASE_REF``, else ``main``."""
  return base_ref or os.environ.get("GITHUB_BASE_REF") or "main"


def diff_against_base(base):
  """Return ``git diff --unified=0`` text against the merge-base with ``base``, or None.

  None means no diff could be computed: the working directory is not inside a git work
  tree, or neither ``origin/<base>`` nor ``<base>`` resolves to a ref that shares history
  with HEAD. Callers that gate CI on the result must treat None as "unknown", never as
  "nothing changed". The diff is limited to ``tests/`` and decoded as UTF-8 with
  undecodable bytes replaced, so file contents never make it fail.
  """
  if not base or base.startswith("-"):  # Never let a ref be read as a git option.
    return None
  env = _git_env()
  text = {"encoding": "utf-8", "errors": "replace"}
  try:
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        env=env,
        check=False,
        **text,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
      return None
    if os.environ.get("GITHUB_ACTIONS") == "true":
      subprocess.run(
          ["git", "fetch", "origin", f"{base}:refs/remotes/origin/{base}"],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
          stdin=subprocess.DEVNULL,
          env=env,
          check=False,
      )
    for command in _build_diff_commands(base):
      result = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL, env=env, check=False, **text)
      if result.returncode == 0:
        return result.stdout
  except OSError:  # git is not installed.
    return None
  return None


# --- Selecting tests -----------------------------------------------------------------------------------------------


def _touched_from_diff(diff_text):
  """Return ``(file_path, _TestDef)`` for every test the diff touched; sources are read relative to the cwd."""
  touched = []
  for path, file_diff in sorted(_parse_diff(diff_text).items()):
    if not _is_test_file(path):
      continue
    new = _parse_test_module(_read_text(path))
    if new is None:
      continue
    added, old = file_diff.added, None
    if file_diff.deleted:
      old = _parse_test_module(_read_blob(file_diff.old_blob))
      if old is None:
        added = added | file_diff.boundary  # Cannot attribute deletions on the base side: be conservative.
    touched.extend((path, test) for test in _touched_defs(new, added, old, file_diff.deleted))
  return touched


def _all_tests():
  """Return ``(file_path, _TestDef)`` for every test in every test file under ``tests/``, read from the cwd."""
  found = []
  for directory, subdirs, filenames in os.walk(_TESTS_ROOT.rstrip("/")):
    subdirs[:] = sorted(d for d in subdirs if not d.startswith(".") and d != "__pycache__")
    for filename in sorted(filenames):
      path = posixpath.join(directory.replace(os.sep, "/"), filename)
      if not _is_test_file(path):
        continue
      module = _parse_test_module(_read_text(path))
      if module is not None:
        found.extend((path, test) for test in module.tests)
  return found


def _select(touched, require_marker=None, exclude_marker=None, flavor=None, ini=None):
  """Return the ``(file_path, test_name)`` pairs of ``touched`` that pass the marker filters and ``flavor``.

  A pair is selected when at least one touched test of that name in that file passes every filter.
  """
  if flavor is not None and ini is None:
    ini = _read_pytest_ini()
  selected = set()
  for path, test in touched:
    if require_marker is not None and require_marker not in test.markers:
      continue
    if exclude_marker is not None and exclude_marker in test.markers:
      continue
    if flavor is not None and not _is_eligible(flavor, ini, path, test.markers):
      continue
    selected.add((path, test.name))
  return selected


def changed_tests_from_diff(diff_text, require_marker=None, exclude_marker=None, flavor=None):
  """Return ``(file_path, test_name)`` for every test the diff added or modified.

  Args:
    diff_text: Raw ``git diff --unified=0`` output. Test sources are read from the
      current working directory, which must be the repository root; base versions are
      read from git through the diff's ``index`` lines.
    require_marker: Optional pytest marker name. When given, only tests carrying it (on
      the function, its class or the module's ``pytestmark``) are reported.
    exclude_marker: Optional pytest marker name. Tests carrying it are dropped.
    flavor: Optional ``TPU7X_FLAVORS`` key. When given, only tests eligible for that flavor
      are reported (pytest.ini is read from the current working directory).

  Raises:
    ValueError: ``flavor`` is not a ``TPU7X_FLAVORS`` key.
  """
  spec = None
  if flavor is not None:
    if flavor not in TPU7X_FLAVORS:
      raise ValueError(f"unknown TPU7X flavor {flavor!r}; expected one of {sorted(TPU7X_FLAVORS)}")
    spec = TPU7X_FLAVORS[flavor]
  return _select(_touched_from_diff(diff_text), require_marker, exclude_marker, spec)


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


# --- pytest items (used by tests/conftest.py) ----------------------------------------------------------------------


def item_test_names(item):
  """Return every name a pytest item may be listed under by ``get_changed_tests``.

  That is ``item.originalname`` (the name without pytest's ``[params]`` suffix), ``item.name``,
  and the ``__name__`` of the item's function after ``inspect.unwrap``. The last one is the
  def name for absl ``parameterized`` tests, whose generated ``test_x_bf16`` or ``test_x0``
  methods wrap ``test_x`` with ``functools.wraps`` and are collected under the generated name.
  """
  names = set()
  for attribute in ("originalname", "name"):
    value = getattr(item, attribute, None)
    if isinstance(value, str) and value:
      names.add(value)
  try:
    function = item.function
  except AttributeError:
    function = None
  if function is not None:
    try:
      name = getattr(inspect.unwrap(function), "__name__", None)
    except ValueError:  # A cycle in the __wrapped__ chain.
      name = None
    if isinstance(name, str) and name:
      names.add(name)
  return names


def is_changed_item(item, changed_tests):
  """Return whether a pytest item is one of ``changed_tests`` (``(file_path, test_name)`` pairs).

  The file is the part of ``item.nodeid`` before the first ``::``; the name is any of ``item_test_names(item)``.
  """
  if not changed_tests:
    return False
  item_file = item.nodeid.split("::", 1)[0]
  return any((item_file, name) in changed_tests for name in item_test_names(item))


# --- Command line --------------------------------------------------------------------------------------------------


class _ArgumentParser(argparse.ArgumentParser):
  """An ``ArgumentParser`` whose usage errors exit 64, never 2 (which means "cannot diff")."""

  def error(self, message):
    self.print_usage(sys.stderr)
    self.exit(_EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _build_arg_parser():
  """Return the command-line parser; see the module docstring."""
  parser = _ArgumentParser(
      description="List the tests a branch added or modified relative to a base ref.",
      allow_abbrev=False,
  )
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
  scope = parser.add_mutually_exclusive_group()
  scope.add_argument(
      "--flavor",
      choices=sorted(TPU7X_FLAVORS),
      default=None,
      help="Only report tests that this TPU7X flavor collects and runs.",
  )
  scope.add_argument(
      "--eligible-flavors",
      action="store_true",
      help=(
          "Print one compact JSON array of the TPU7X PR-matrix flavors that have at least one eligible test, in"
          " matrix order, and one '<flavor>: path::test_name' line per eligible test on stderr."
      ),
  )
  parser.add_argument(
      "--all-tests",
      action="store_true",
      help="Treat every test under tests/ as touched, without running git (for verification and debugging).",
  )
  return parser


def main(argv=None):
  """Command-line entry point; see the module docstring.

  Returns 0 when a diff was computed (the output may be empty) or with ``--all-tests``, 2 when no diff could be
  computed, so a CI gate can tell "no changed tests" apart from "detection is broken", and 64 for usage errors.
  """
  try:
    args = _build_arg_parser().parse_args(argv)
  except SystemExit as exit_request:  # --help (0) or a usage error (64).
    return exit_request.code if isinstance(exit_request.code, int) else _EXIT_USAGE
  if args.all_tests:
    touched = _all_tests()
  else:
    base = _resolve_base(args.base)
    diff_text = diff_against_base(base)
    if diff_text is None:
      print(
          f"newly_added_detection: cannot diff against origin/{base} or {base} (cwd: {os.getcwd()})",
          file=sys.stderr,
      )
      return _EXIT_CANNOT_DIFF
    touched = _touched_from_diff(diff_text)
  if args.eligible_flavors:
    ini = _read_pytest_ini()
    eligible = []
    for name in TPU7X_PR_MATRIX:
      selected = _select(touched, args.require_marker, args.exclude_marker, TPU7X_FLAVORS[name], ini)
      if selected:
        eligible.append(name)
      for path, test_name in sorted(selected):
        print(f"{name}: {path}::{test_name}", file=sys.stderr)
    print(json.dumps(eligible, separators=(",", ":")))
    return 0
  flavor = TPU7X_FLAVORS[args.flavor] if args.flavor is not None else None
  for path, test_name in sorted(_select(touched, args.require_marker, args.exclude_marker, flavor)):
    print(f"{path}::{test_name}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
