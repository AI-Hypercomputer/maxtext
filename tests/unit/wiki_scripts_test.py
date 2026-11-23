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

"""Unit and live-repository validation tests for .agents/wiki/ scripts."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DIR = REPO_ROOT / ".agents/wiki"
SCRIPTS_DIR = WIKI_DIR / "scripts"


def _load_script_module(module_name: str, script_path: Path):
  """Loads a standalone script module without writing .pyc bytecode into .agents/."""
  prev_flag = sys.dont_write_bytecode
  sys.dont_write_bytecode = True
  try:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
  finally:
    sys.dont_write_bytecode = prev_flag


validate_wiki = _load_script_module("validate_wiki", SCRIPTS_DIR / "validate_wiki.py")
harvest_candidates = _load_script_module("harvest_candidates", SCRIPTS_DIR / "harvest_candidates.py")


def _sample_leaf_lines(
    category: str = "troubleshooting",
    status: str = "active",
    code_ref: str = "src/maxtext/configs/base.yml",
    cfg_key: str = "param_scan_axis",
    extra_lines: int = 4,
) -> list[str]:
  """Builds a valid synthetic wiki leaf page for positive and negative unit tests."""
  lines = [
      "---",
      "title: Sample Playbook",
      f"category: {category}",
      f"status: {status}",
      "phases: [phase4]",
      "applies_to: [linen, m3]",
      "tags: [hbm, moe, remat]",
      "code_refs:",
      f"  - {code_ref}",
      "config_keys:",
      f"  - {cfg_key}",
      "verified_on:",
      '  - "SampleModel (TPU v5p, PR #1)"',
      "last_verified: 2026-11-19",
      "---",
      "# Sample Playbook",
      "## Symptom / When to Use",
      "Sample symptom description.",
      "## Action Ladder",
      "Sample Stage 1 action.",
      "## Guardrails & Cleanup",
      "Sample guardrail.",
      "## Related",
      "- [Hub](../index.md)",
  ]
  lines.extend(f"Detail line {i}" for i in range(extra_lines))
  return lines


class WikiScriptsTest(unittest.TestCase):
  """Verifies live .agents/wiki/ integrity and validator/harvester invariants."""

  def test_live_repository_wiki_passes_validation(self):
    """Checks that the committed .agents/wiki/ tree passes all validator rules."""
    errors = validate_wiki.validate_wiki(REPO_ROOT, WIKI_DIR)
    self.assertEqual(errors, [], msg=f"Live .agents/wiki/ failed validation: {errors}")

  def test_validator_catches_grounding_links_and_lifecycle_errors(self):
    """Checks negative controls for stale code_refs/config_keys, broken links, and graduated status."""
    with tempfile.TemporaryDirectory() as tmp:
      wiki = Path(tmp)
      (wiki / "index.md").write_text(
          "# Hub\n## Topic Index\n| [troubleshooting/sample.md](troubleshooting/sample.md) |\n",
          encoding="utf-8",
      )
      (wiki / "PROTOCOL.md").write_text("# Protocol\n", encoding="utf-8")
      (wiki / "troubleshooting").mkdir()

      good = "\n".join(_sample_leaf_lines()) + "\n"
      (wiki / "troubleshooting/sample.md").write_text(good, encoding="utf-8")
      self.assertEqual(validate_wiki.validate_wiki(REPO_ROOT, wiki), [])

      # Inject stale code_ref, stale config_key, broken link, and leaked personal path.
      broken = "\n".join(
          _sample_leaf_lines(code_ref="src/maxtext/configs/missing_xyz.yml", cfg_key="nonexistent_flag_xyz")
      )
      broken += "\nSee [missing](does_not_exist.md) in /home/alice/work\n"
      (wiki / "troubleshooting/sample.md").write_text(broken, encoding="utf-8")
      errs = validate_wiki.validate_wiki(REPO_ROOT, wiki)
      self.assertTrue(any("stale code_refs" in e for e in errs), errs)
      self.assertTrue(any("stale config_keys" in e for e in errs), errs)
      self.assertTrue(any("broken relative link" in e for e in errs), errs)
      self.assertTrue(any("personal path" in e for e in errs), errs)

      # Graduated page must declare superseded_by and be removed from active Topic Index.
      grad_lines = _sample_leaf_lines(status="graduated")
      (wiki / "troubleshooting/sample.md").write_text("\n".join(grad_lines) + "\n", encoding="utf-8")
      grad_errs = validate_wiki.validate_wiki(REPO_ROOT, wiki)
      self.assertTrue(any("superseded_by" in e for e in grad_errs), grad_errs)
      self.assertTrue(any("removed from active index.md" in e for e in grad_errs), grad_errs)

  def test_validator_emits_compaction_advisory_above_threshold(self):
    """Checks that pages between 120 and 140 lines emit a non-blocking compaction advisory."""
    with tempfile.TemporaryDirectory() as tmp:
      wiki = Path(tmp)
      (wiki / "index.md").write_text(
          "# Hub\n## Topic Index\n| [troubleshooting/dense.md](troubleshooting/dense.md) |\n",
          encoding="utf-8",
      )
      (wiki / "PROTOCOL.md").write_text("# Protocol\n", encoding="utf-8")
      (wiki / "troubleshooting").mkdir()

      near_cap = "\n".join(_sample_leaf_lines(extra_lines=100)) + "\n"
      (wiki / "troubleshooting/dense.md").write_text(near_cap, encoding="utf-8")
      errors, warnings = validate_wiki.audit_wiki(REPO_ROOT, wiki, check_drift=False)
      self.assertEqual(errors, [])
      self.assertTrue(any("compact older bullets" in w for w in warnings), warnings)

  def test_harvester_parses_handover_and_matches_overlaps(self):
    """Checks handover Section 4 parsing, shared-infra classification, and tag overlap matching."""
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      handover = root / "demo_handover.md"
      handover.write_text(
          "# Demo Handover\n"
          "## 4. Bugs, Pitfalls & Wiki Distillation Log\n"
          "| Stage | Symptom | Root Cause & Fix | Verified Evidence | Wiki Disposition |\n"
          "| :--- | :--- | :--- | :--- | :--- |\n"
          "| P2 | eps typo | 1e-6 vs 1e-5 | unit test pass | local-only |\n"
          "| P4 | MoE OOM | Set param_scan_axis=0 + remat | 90 -> 52 GiB | pending |\n",
          encoding="utf-8",
      )
      items = harvest_candidates.extract_handover_candidates([handover], root)
      self.assertEqual(len(items), 1)
      self.assertIn("MoE OOM", items[0]["entry"])

      self.assertTrue(harvest_candidates.is_shared_infra_path("src/maxtext/configs/base.yml"))
      self.assertTrue(harvest_candidates.is_shared_infra_path("src/maxtext/kernels/ragged/ragged_gather_reduce.py"))
      self.assertFalse(harvest_candidates.is_shared_infra_path("src/maxtext/configs/models/demo.yml"))

      overlaps = harvest_candidates.match_overlapping_wiki_pages(
          WIKI_DIR,
          shared_files=[],
          config_keys=[],
          signal_tokens={"moe", "remat", "hbm"},
      )
      self.assertTrue(any(ov["page"] == "troubleshooting/hbm_verification.md" for ov in overlaps), overlaps)


if __name__ == "__main__":
  unittest.main()
