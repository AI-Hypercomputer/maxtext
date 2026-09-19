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

"""Scaffolding tests for the m3 package.

These guard the landing zone itself: the package is importable under its
expected name, the subpackage layout matches the design, and every subpackage
is a real Python package rather than an implicit namespace directory.
"""

import importlib
import os

from absl.testing import absltest
from absl.testing import parameterized

import maxtext.m3

_PACKAGE_DIR = os.path.dirname(maxtext.m3.__file__)

# The layout the design calls for. Kept explicit so that deleting or renaming a
# subpackage fails loudly, which a purely disk-derived check cannot catch.
_DESIGNED_SUBPACKAGES = ("configs", "core", "infra", "models", "train")


def _subpackages_on_disk():
  """Returns the directories under `m3/` that are real Python packages.

  Keying on `__init__.py` rather than on a naming convention keeps
  `__pycache__` out for the right reason, and leaves room for non-package data
  directories later.
  """
  return tuple(
      sorted(name for name in os.listdir(_PACKAGE_DIR) if os.path.isfile(os.path.join(_PACKAGE_DIR, name, "__init__.py")))
  )


class M3ScaffoldingTest(parameterized.TestCase):
  """Tests that the m3 landing zone is importable and correctly laid out."""

  def test_package_is_importable(self):
    """`import maxtext.m3` works and resolves to a package directory."""
    self.assertTrue(hasattr(maxtext.m3, "__path__"))

  def test_declared_subpackages_match_layout(self):
    """`__all__` lists exactly the subpackages present on disk."""
    self.assertEqual(sorted(maxtext.m3.__all__), list(_subpackages_on_disk()))

  def test_designed_subpackages_are_present(self):
    """Every subpackage the design calls for still exists."""
    missing = sorted(set(_DESIGNED_SUBPACKAGES) - set(_subpackages_on_disk()))
    self.assertEmpty(missing, f"missing m3 subpackages: {missing}")

  @parameterized.parameters(*_subpackages_on_disk())
  def test_subpackage_is_importable(self, name):
    """Each subpackage on disk imports and is documented."""
    module = importlib.import_module(f"maxtext.m3.{name}")
    self.assertIsNotNone(module.__doc__, f"maxtext.m3.{name} is missing a module docstring")

  def test_readme_is_present(self):
    """The architecture rules ship with the package."""
    self.assertTrue(os.path.isfile(os.path.join(_PACKAGE_DIR, "README.md")))


if __name__ == "__main__":
  absltest.main()
