# Copyright 2023–2026 Google LLC
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

"""Unit tests for maxtext/__init__.py export consistency and lazy loading."""

import ast
import pathlib
import unittest
import maxtext


def _extract_all_items(node: ast.AST) -> list[str]:
  """Extracts string elements from '__all__ = [...]' assignment."""
  if not isinstance(node, ast.Assign):
    return []
  for target in node.targets:
    if isinstance(target, ast.Name) and target.id == "__all__":
      if isinstance(node.value, (ast.List, ast.Tuple)):
        return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
  return []


def _extract_type_checking_imports(node: ast.AST) -> list[str]:
  """Extracts imported symbol names inside 'if TYPE_CHECKING:' block."""
  if not isinstance(node, ast.If):
    return []
  is_type_checking = (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING") or (
      isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
  )
  if not is_type_checking:
    return []

  names = []
  for stmt in node.body:
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
      for alias in stmt.names:
        names.append(alias.asname or alias.name)
  return names


def _extract_match_case_names(pattern: ast.pattern) -> list[str]:
  """Extracts string case names from match pattern."""
  if isinstance(pattern, ast.MatchValue):
    if isinstance(pattern.value, ast.Constant) and isinstance(pattern.value.value, str):
      return [pattern.value.value]
  elif isinstance(pattern, ast.MatchOr):
    names = []
    for sub_pat in pattern.patterns:
      names.extend(_extract_match_case_names(sub_pat))
    return names
  return []


def _extract_getattr_cases(node: ast.AST) -> list[str]:
  """Extracts handled attribute names from '__getattr__' match statements."""
  if not (isinstance(node, ast.FunctionDef) and node.name == "__getattr__"):
    return []
  cases = []
  for stmt in node.body:
    if isinstance(stmt, ast.Match):
      for case in stmt.cases:
        cases.extend(_extract_match_case_names(case.pattern))
  return cases


def _extract_init_ast_metadata(file_path: pathlib.Path):
  """Extracts __all__, TYPE_CHECKING imports, and __getattr__ match cases from an __init__.py file."""
  tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

  all_items: list[str] = []
  type_checking_names: list[str] = []
  getattr_cases: list[str] = []

  for node in tree.body:
    all_items.extend(_extract_all_items(node))
    type_checking_names.extend(_extract_type_checking_imports(node))
    getattr_cases.extend(_extract_getattr_cases(node))

  return {
      "all_items": all_items,
      "type_checking_names": type_checking_names,
      "getattr_cases": getattr_cases,
  }


class MaxtextInitConsistencyTest(unittest.TestCase):
  """Ensures __init__.py static declarations, TYPE_CHECKING block, and __getattr__ stay in sync."""

  def setUp(self):
    super().setUp()
    # Locate src/maxtext/__init__.py
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    self.init_path = root_dir / "src" / "maxtext" / "__init__.py"
    if not self.init_path.exists():
      # Fallback to module __file__ if available
      self.init_path = pathlib.Path(maxtext.__file__).resolve()

  def test_type_checking_block_matches_getattr(self):
    meta = _extract_init_ast_metadata(self.init_path)
    self.assertTrue(meta["type_checking_names"], "No imports found in 'if TYPE_CHECKING:' block.")
    self.assertTrue(meta["getattr_cases"], "No match cases found in '__getattr__'.")
    self.assertCountEqual(
        meta["type_checking_names"],
        meta["getattr_cases"],
        "The 'if TYPE_CHECKING:' block does not match the cases in '__getattr__'.",
    )

  def test_type_checking_block_matches_all(self):
    meta = _extract_init_ast_metadata(self.init_path)
    self.assertTrue(meta["all_items"], "No exports found in '__all__'.")
    expected_all = set(meta["type_checking_names"]) | {
        "__author__",
        "__description__",
        "__version__",
    }
    self.assertEqual(
        set(meta["all_items"]),
        expected_all,
        "The '__all__' export list does not match 'if TYPE_CHECKING:' + version metadata.",
    )

  def test_all_exports_accessible_at_runtime(self):
    for name in dir(maxtext):
      if name.startswith("_") and name not in ("__author__", "__description__", "__version__"):
        continue
      with self.subTest(name=name):
        self.assertTrue(hasattr(maxtext, name), f"Missing attribute '{name}' on maxtext")
        val = getattr(maxtext, name)
        self.assertIsNotNone(val, f"Attribute '{name}' resolved to None on maxtext")


if __name__ == "__main__":
  unittest.main()
