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

"""Unit tests for the update-changelog skill's `gather_changes.py` helper (no git repository required)."""

import importlib.util
import pathlib
import unittest
from unittest import mock

SCRIPT_PATH = pathlib.Path(__file__).resolve().with_name("gather_changes.py")


def _load_script():
  spec = importlib.util.spec_from_file_location("gather_changes", SCRIPT_PATH)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _log_entry(gc, short_hash, full_hash, subject, body="", files="src/maxtext/foo.py"):
  """Builds one `run_git_log` entry for a single-parent commit."""
  fields = (short_hash, full_hash, "0" * 40, subject, body)
  return f"{gc.COMMIT_DELIMITER}{gc.FIELD_SEP.join(fields)}{gc.FILES_SEP}\n{files}\n\n"


class GatherChangesTest(unittest.TestCase):
  """Covers the generator defects behind PR #5887: heading spacing, revert pairs, and pre-commit-only noise."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.gc = _load_script()

  def test_merge_unreleased_notes_separates_every_heading_with_one_blank_line(self):
    existing = (
        "#### Changes\n\n##### Models\n\n- old model\n\n##### Pre-Training\n\n- old pretrain\n\n"
        "#### Bug Fixes\n\n- old fix\n"
    )
    new = "#### Changes\n\n##### Models\n\n- new model\n\n#### Bug Fixes\n\n- new fix\n"
    merged = self.gc.merge_unreleased_notes(existing, new)
    self.assertEqual(
        merged,
        "#### Changes\n\n##### Models\n\n- new model\n- old model\n\n##### Pre-Training\n\n- old pretrain\n\n"
        "#### Bug Fixes\n\n- new fix\n- old fix",
    )
    # mdformat rewrites a heading glued to the previous bullet list; make sure none is emitted.
    self.assertNotRegex(merged, r"[^\n]\n#{4,5} ")
    self.assertNotRegex(merged, r"#{4,5} [^\n]*\n- ")

  def test_cleanup_subject_patterns(self):
    excluded = [
        "Fix Pre-commit Linters failure on main",
        "Fix precommit hook config to run pytype",
        "Fix pylint warnings in types.py",
        "Fix typos in docs",
        'Revert "Add foo"',
    ]
    kept = [
        "Fix lint errors and add support for X",
        "Fix loss masking for packed sequences",
        "Add support for On-The-Fly Dynamic SafeTensors loading",
    ]
    for subject in excluded:
      self.assertTrue(self.gc.is_followup_or_cleanup_subject(subject), subject)
    for subject in kept:
      self.assertFalse(self.gc.is_followup_or_cleanup_subject(subject), subject)

  def test_revert_marker_regex(self):
    for message, expected in [
        ("Reverts b64f00a115d1e05954a6eafc663bf7779d316e90", "b64f00a115d1e05954a6eafc663bf7779d316e90"),
        (
            "PiperOrigin-RevId: 1\nThis reverts commit 0123456789abcdef0123456789abcdef01234567.",
            "0123456789abcdef0123456789abcdef01234567",
        ),
        ("Reverts changelist 908368682", None),
        ("Add revert support for abc1234", None),
    ]:
      match = self.gc.REVERT_MARKER_RE.search(message)
      self.assertEqual(match.group(1) if match else None, expected, message)

  def _parse(self, raw_log):
    # No git repository is needed: hashes resolve to themselves and out-of-window commits have no message.
    with mock.patch.object(self.gc, "resolve_commit", side_effect=lambda rev: rev if len(rev) == 40 else None):
      with mock.patch.object(self.gc, "get_commit_message", return_value=""):
        with mock.patch.object(self.gc, "extract_rtd_links", return_value=""):
          return self.gc.parse_commits(raw_log)

  def test_revert_pair_inside_window_is_dropped(self):
    gc = self.gc
    original, revert = "a" * 40, "b" * 40
    raw_log = (
        _log_entry(gc, "bbbbbbbbb", revert, "move splash attention out of experimental ops", f"Reverts {original}")
        + _log_entry(gc, "aaaaaaaaa", original, "move splash attention out of experimental ops")
        + _log_entry(gc, "ccccccccc", "c" * 40, "Add Gemma 4 26B GRPO tutorial")
    )
    self.assertEqual([c["ref_id"] for c in self._parse(raw_log)], ["ccccccccc"])

  def test_revert_of_revert_inside_window_keeps_original(self):
    gc = self.gc
    original, rollback, rollforward = "a" * 40, "b" * 40, "c" * 40
    subject = "Migrate SFT Checkpoint Manager to V1 Checkpointer"
    raw_log = (
        _log_entry(gc, "ccccccccc", rollforward, subject, f"Reverts {rollback}")
        + _log_entry(gc, "bbbbbbbbb", rollback, subject, f"Reverts {original}")
        + _log_entry(gc, "aaaaaaaaa", original, subject)
    )
    self.assertEqual([c["ref_id"] for c in self._parse(raw_log)], ["aaaaaaaaa"])

  def test_revert_of_older_commit_is_labelled(self):
    gc = self.gc
    raw_log = _log_entry(gc, "bbbbbbbbb", "b" * 40, "Add support for dynamic loading", f"Reverts {'f' * 40}")
    commits = self._parse(raw_log)
    self.assertEqual([c["summary"] for c in commits], ["Revert of: Add support for dynamic loading"])
    self.assertEqual(
        gc.generate_changelog_block(commits), "- Revert of: Add support for dynamic loading. [ref:bbbbbbbbb]"
    )


if __name__ == "__main__":
  unittest.main()
