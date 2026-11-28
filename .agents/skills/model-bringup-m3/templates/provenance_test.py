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

"""Provenance test template: pins the PyTorch reference, oracle files, and goldens.

Copy to `tests/unit/m3/<model>_provenance_test.py` and replace every `_UNFILLED`
constant. Uses `unittest` (collected by `pytest`) with method-level
`@unittest.skipUnless(HAS_TORCH, ...)` on reference-module import checks so
in-repo hash and manifest checks always run in torch-free CI.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import pathlib
from typing import Any
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

_UNFILLED = object()

# Reference modules (`{'transformers.models.<m>.modeling_<m>': '<sha256>'}`) or `{}`.
PINNED_REFERENCE_MODULES: Any = _UNFILLED
# In-repo oracle files relative to this test's directory (`{'testdata/<m>/oracle.py': '<sha256>'}`) or `{}`.
PINNED_ORACLE_FILES: Any = _UNFILLED
# Golden manifest relative path and its sha256 digest.
GOLDEN_MANIFEST: Any = _UNFILLED
GOLDEN_MANIFEST_SHA256: Any = _UNFILLED

_HERE = pathlib.Path(__file__).resolve().parent


def _require_filled(val: Any, what: str) -> Any:
  if val is _UNFILLED:
    raise NotImplementedError(f"Provenance template not filled in: {what}")
  return val


def _sha256(path: pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class ProvenanceTest(unittest.TestCase):
  """Verifies sha256 digests of reference sources, oracle files, and goldens."""

  def test_some_reference_is_pinned(self) -> None:
    mods = _require_filled(PINNED_REFERENCE_MODULES, "PINNED_REFERENCE_MODULES")
    oracles = _require_filled(PINNED_ORACLE_FILES, "PINNED_ORACLE_FILES")
    self.assertTrue(mods or oracles, "no reference or oracle file is pinned")

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch / transformers reference modules")
  def test_reference_module_sources_unchanged(self) -> None:
    mods = _require_filled(PINNED_REFERENCE_MODULES, "PINNED_REFERENCE_MODULES")
    for module_name, expected in mods.items():
      src_file = inspect.getsourcefile(importlib.import_module(module_name))
      self.assertIsNotNone(src_file, f"No source file found for {module_name}")
      source = pathlib.Path(str(src_file))
      self.assertEqual(_sha256(source), expected, f"{module_name} ({source}) changed")

  def test_in_test_oracle_files_unchanged(self) -> None:
    oracles = _require_filled(PINNED_ORACLE_FILES, "PINNED_ORACLE_FILES")
    for rel_path, expected in oracles.items():
      self.assertEqual(
          _sha256(_HERE / rel_path),
          expected,
          f"oracle file {rel_path} changed",
      )

  def test_golden_manifest_unchanged(self) -> None:
    manifest_rel = _require_filled(GOLDEN_MANIFEST, "GOLDEN_MANIFEST")
    expected = _require_filled(GOLDEN_MANIFEST_SHA256, "GOLDEN_MANIFEST_SHA256")
    self.assertEqual(_sha256(_HERE / manifest_rel), expected)

  def test_every_golden_matches_manifest(self) -> None:
    manifest_rel = _require_filled(GOLDEN_MANIFEST, "GOLDEN_MANIFEST")
    manifest = json.loads((_HERE / manifest_rel).read_text(encoding="utf-8"))
    goldens = manifest["goldens"]
    self.assertGreater(len(goldens), 0)
    for entry in goldens:
      self.assertEqual(
          _sha256(_HERE / entry["local_path"]),
          entry["sha256"],
          entry["local_path"],
      )

  def test_manifest_reference_matches_pins(self) -> None:
    manifest_rel = _require_filled(GOLDEN_MANIFEST, "GOLDEN_MANIFEST")
    mods = _require_filled(PINNED_REFERENCE_MODULES, "PINNED_REFERENCE_MODULES")
    oracles = _require_filled(PINNED_ORACLE_FILES, "PINNED_ORACLE_FILES")
    manifest = json.loads((_HERE / manifest_rel).read_text(encoding="utf-8"))
    self.assertEqual(manifest["reference"]["pinned_modules"], mods)
    self.assertEqual(manifest["reference"]["pinned_oracle_files"], oracles)


if __name__ == "__main__":
  unittest.main()
