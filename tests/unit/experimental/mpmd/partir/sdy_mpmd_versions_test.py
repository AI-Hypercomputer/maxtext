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

"""Tests for the jaxlib/Shardy MPMD version table and its install hints (no Shardy needed)."""

import importlib.metadata
import json
import types
import unittest
from unittest import mock

from maxtext.experimental.mpmd.partir import sdy_mpmd_versions as versions

COMMIT = "38ab99f4d252c59d491899a694fe8cbadfeccec3"
# PEP 610 direct_url.json records of a git install, an archive install and a local directory install.
VCS = {"url": "https://github.com/openxla/shardy", "vcs_info": {"vcs": "git", "commit_id": COMMIT}}
ARCHIVE = {"url": f"https://github.com/openxla/shardy/archive/{COMMIT}.zip", "archive_info": {}}
LOCAL = {"url": "file:///src/shardy/integrations/python/jax/mpmd", "dir_info": {}}


class VersionsTest(unittest.TestCase):

  def test_install_command(self):
    self.assertEqual(
        versions.install_command("0.12.0.1"),
        f'pip install --no-deps "shardy-mpmd @ git+https://github.com/openxla/shardy@{COMMIT}'
        '#subdirectory=shardy/integrations/python/jax/mpmd"',
    )
    self.assertIn("no Shardy commit is known for jaxlib 9.9.9", versions.install_command("9.9.9"))
    self.assertEqual(set(versions.SHARDY_COMMITS), {"0.11.1", "0.12.0.1", "0.12.1"})

  def test_commit_from_direct_url(self):
    self.assertEqual(versions.commit_from_direct_url(json.dumps(VCS)), COMMIT)
    self.assertEqual(versions.commit_from_direct_url(json.dumps(ARCHIVE)), COMMIT)
    self.assertIsNone(versions.commit_from_direct_url(json.dumps(LOCAL)))
    self.assertIsNone(versions.commit_from_direct_url("not json"))

  def test_installed_commit(self):
    """The commit pip recorded for the shardy-mpmd distribution, or None when there is no record or no distribution."""

    def distribution(record):
      """A distribution whose direct_url.json is `record` (None: the file is missing)."""
      return types.SimpleNamespace(
          read_text=lambda name: json.dumps(record) if name == "direct_url.json" and record is not None else None
      )

    for record, want in ((VCS, COMMIT), (ARCHIVE, COMMIT), (LOCAL, None), (None, None)):
      with mock.patch.object(importlib.metadata, "distribution", return_value=distribution(record)) as found:
        self.assertEqual(versions.installed_commit(), want, record)
      found.assert_called_once_with("shardy-mpmd")
    # A Shardy checkout on PYTHONPATH: no distribution at all.
    missing = importlib.metadata.PackageNotFoundError("shardy-mpmd")
    with mock.patch.object(importlib.metadata, "distribution", side_effect=missing):
      self.assertIsNone(versions.installed_commit())

  def test_version_mismatch(self):
    self.assertIsNone(versions.version_mismatch("0.12.0.1", COMMIT))
    self.assertIsNone(versions.version_mismatch("0.12.0.1", None))  # a checkout on PYTHONPATH: cannot be checked
    mismatch = versions.version_mismatch("0.12.1", COMMIT)
    self.assertIn("jaxlib 0.12.1 was built with Shardy 57fffc23dfdf", mismatch)
    self.assertIn("--force-reinstall", mismatch)
    self.assertIn("no Shardy commit is known", versions.version_mismatch("9.9.9", None))


if __name__ == "__main__":
  unittest.main()
