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

"""Tests for the Shardy MPMD import shim (the frontend's import needs shardy-mpmd)."""

import importlib
import logging
import sys
import unittest
from unittest import mock

import jaxlib

from maxtext.experimental.mpmd.partir import sdy_mpmd_versions

_COMPAT = "maxtext.experimental.mpmd.partir.sdy_mpmd_compat"

try:
  from maxtext.experimental.mpmd.partir import sdy_mpmd_compat as compat

  _SKIP = ""
except ModuleNotFoundError as error:  # shardy-mpmd is an optional dependency
  if not (error.name or "").startswith("shardy"):
    raise
  _SKIP = f"needs shardy-mpmd: {error}"


class ShimTest(unittest.TestCase):

  @unittest.skipIf(bool(_SKIP), _SKIP)
  def test_imports_the_frontend_through_the_aliases(self):
    self.assertTrue(callable(compat.mpmd.jit))
    self.assertTrue(callable(compat.shardy_pipeline.build_merge_rules_from_predicate))
    self.assertIs(sys.modules["jax.jaxlib"], jaxlib)
    self.assertIn("mlir.ir", sys.modules)

  @unittest.skipIf(bool(_SKIP), _SKIP)
  def test_a_mismatched_frontend_warns(self):
    """pip recorded another Shardy commit than the installed jaxlib's: importing the shim warns, with the fix."""
    for commit, warns in (("0" * 40, True), (sdy_mpmd_versions.SHARDY_COMMITS.get(jaxlib.__version__), False)):
      with (
          mock.patch.dict(sys.modules),
          mock.patch.object(sdy_mpmd_versions, "installed_commit", return_value=commit),
          self.assertLogs("absl", level="WARNING") as logs,
      ):
        del sys.modules[_COMPAT]
        importlib.import_module(_COMPAT)
        logging.getLogger("absl").warning("end of the import")  # assertLogs needs one record
      mismatch = [line for line in logs.output if "shardy-mpmd was installed from Shardy 000000000000" in line]
      self.assertEqual(bool(mismatch), warns, logs.output)
      if warns:
        self.assertIn(sdy_mpmd_versions.install_command(jaxlib.__version__), mismatch[0])

  def test_a_missing_frontend_names_the_install_command(self):
    """Without shardy-mpmd, the MPMD paths stop with the pip command for the installed jaxlib."""
    with mock.patch.dict(sys.modules):
      for name in [name for name in sys.modules if name == "shardy" or name.startswith(("shardy.", _COMPAT))]:
        del sys.modules[name]
      sys.modules["shardy"] = None  # makes `import shardy...` raise ModuleNotFoundError
      with self.assertRaises(ModuleNotFoundError) as raised:
        importlib.import_module(_COMPAT)
    self.assertIn(sdy_mpmd_versions.install_command(jaxlib.__version__), str(raised.exception))
    self.assertTrue(raised.exception.name.startswith("shardy"), raised.exception.name)


if __name__ == "__main__":
  unittest.main()
