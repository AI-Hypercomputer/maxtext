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

"""Unit tests for maxtext.tools.model_lifecycle_audit."""

import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from maxtext.tools import model_lifecycle_audit


class ModelLifecycleAuditTest(unittest.TestCase):
  """Tests for the automated Model Lifecycle Audit tool."""

  def test_live_repo_audit_passes_rule1_and_finds_25_deprecated_configs(self):
    """Verifies that the current repository state satisfies Rule 1 (N-2 cap) and tracks all 25 Sunset configs."""
    report = model_lifecycle_audit.run_audit(
        reference_date=datetime.date(2026, 10, 22),
    )
    self.assertTrue(report.rule1_passed, f"Expected Rule 1 to pass, got: {report.rule1_generation_cap}")
    self.assertEqual(report.deprecated_model_count, 25)

    by_family = {r.family: r for r in report.rule1_generation_cap}
    for family in ("gemma", "qwen", "deepseek", "llama"):
      self.assertIn(family, by_family)
      self.assertLessEqual(len(by_family[family].active_generations), 2)
      self.assertEqual(by_family[family].violating_generations, [])
      self.assertTrue(by_family[family].passed)

    # Check that tiering.md was parsed and stale entries (e.g. llama2-70b, mixtral-8x7b) are surfaced
    self.assertGreater(len(report.tiering_entries), 0)
    self.assertIn("llama2-70b", report.deprecated_models_in_tiering)
    self.assertIn("mixtral-8x7b", report.deprecated_models_in_tiering)

  def test_rule1_flags_n_minus_2_generation_violation(self):
    """Verifies Rule 1 flags a family when more than 2 generations exist in configs/models/."""
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      models_dir = root / "src" / "maxtext" / "configs" / "models"
      deprecated_dir = models_dir / "deprecated"
      models_dir.mkdir(parents=True)
      deprecated_dir.mkdir(parents=True)

      # Put 3 generations of Gemma in active models/
      for name in ("gemma2-9b.yml", "gemma3-4b.yml", "gemma4-26b.yml"):
        (models_dir / name).write_text("decoder_block: gemma\n", encoding="utf-8")

      report = model_lifecycle_audit.run_audit(
          repo_root=root,
          reference_date=datetime.date(2026, 10, 22),
      )
      self.assertFalse(report.rule1_passed)
      gemma_rule = next(r for r in report.rule1_generation_cap if r.family == "gemma")
      self.assertFalse(gemma_rule.passed)
      self.assertEqual(gemma_rule.active_generations, ["gemma2", "gemma3", "gemma4"])
      self.assertEqual(gemma_rule.violating_generations, ["gemma2"])
      self.assertEqual(gemma_rule.violating_models, ["gemma2-9b"])

  def test_rule2_flags_family_depending_only_on_standalone_converter(self):
    """Verifies Rule 2 flags an active model family that has standalone scripts but no unified converter entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      models_dir = root / "src" / "maxtext" / "configs" / "models"
      standalone_dir = root / "src" / "maxtext" / "checkpoint_conversion" / "standalone_scripts"
      utils_dir = root / "src" / "maxtext" / "checkpoint_conversion" / "utils"
      models_dir.mkdir(parents=True)
      standalone_dir.mkdir(parents=True)
      utils_dir.mkdir(parents=True)

      (models_dir / "gemma3-4b.yml").write_text("decoder_block: gemma3\n", encoding="utf-8")
      (models_dir / "llama3.1-8b.yml").write_text("decoder_block: llama2\n", encoding="utf-8")
      (standalone_dir / "convert_gemma3_chkpt.py").write_text("# standalone\n", encoding="utf-8")
      (standalone_dir / "llama_or_mistral_ckpt.py").write_text("# standalone\n", encoding="utf-8")

      # Only register llama3.1-8b in unified PARAM_MAPPING, leaving gemma3-4b on standalone only
      (utils_dir / "param_mapping.py").write_text(
          'PARAM_MAPPING = {\n    "llama3.1-8b": {},\n}\nHOOK_FNS = {\n    "llama3.1-8b": {},\n}\n',
          encoding="utf-8",
      )

      report = model_lifecycle_audit.run_audit(
          repo_root=root,
          reference_date=datetime.date(2026, 10, 22),
      )
      self.assertFalse(report.rule2_passed)
      by_family = {r.family: r for r in report.rule2_standalone_converters}
      self.assertFalse(by_family["gemma"].passed)
      self.assertTrue(by_family["gemma"].depends_only_on_standalone)
      self.assertTrue(by_family["llama"].passed)
      self.assertFalse(by_family["llama"].depends_only_on_standalone)

  def test_rule3_computes_90_day_grace_period_and_expiry(self):
    """Verifies Rule 3 calculates days elapsed, days remaining, and expiry status accurately."""
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      deprecated_dir = root / "src" / "maxtext" / "configs" / "models" / "deprecated"
      deprecated_dir.mkdir(parents=True)

      (deprecated_dir / "llama2-7b.yml").write_text(
          "# deprecated_date: 2026-06-01\ndecoder_block: llama2\n",
          encoding="utf-8",
      )
      (deprecated_dir / "gemma2-2b.yml").write_text(
          "# deprecated_date: 2026-10-01\ndecoder_block: gemma2\n",
          encoding="utf-8",
      )

      report = model_lifecycle_audit.run_audit(
          repo_root=root,
          grace_period_days=90,
          reference_date=datetime.date(2026, 10, 22),
      )
      self.assertEqual(report.deprecated_model_count, 2)
      self.assertEqual(report.rule3_expired_count, 1)

      by_model = {r.model_name: r for r in report.rule3_deprecated_grace_period}
      self.assertTrue(by_model["llama2-7b"].expired)
      self.assertEqual(by_model["llama2-7b"].status, "EXPIRED_READY_FOR_REMOVAL")
      self.assertLessEqual(by_model["llama2-7b"].days_remaining, 0)

      self.assertFalse(by_model["gemma2-2b"].expired)
      self.assertEqual(by_model["gemma2-2b"].status, "IN_GRACE_PERIOD")
      self.assertEqual(by_model["gemma2-2b"].days_in_deprecated, 21)
      self.assertEqual(by_model["gemma2-2b"].days_remaining, 69)

  def test_cli_markdown_and_json_formats(self):
    """Verifies CLI --format markdown and --format json output."""
    buf_md = io.StringIO()
    with mock.patch("sys.stdout", buf_md):
      rc_md = model_lifecycle_audit.main(["--format", "markdown", "--reference-date", "2026-10-22"])
    self.assertEqual(rc_md, 0)
    md_out = buf_md.getvalue()
    self.assertIn("# MaxText Model Lifecycle Audit Report", md_out)
    self.assertIn("## Rule 1: `N-2` Generation Cap", md_out)
    self.assertIn("## Rule 2: Unified Checkpoint Converter Coverage", md_out)
    self.assertIn("## Rule 3: `deprecated/` 90-Day Grace Period Status", md_out)

    buf_json = io.StringIO()
    with mock.patch("sys.stdout", buf_json):
      rc_json = model_lifecycle_audit.main(["--format", "json", "--reference-date", "2026-10-22"])
    self.assertEqual(rc_json, 0)
    payload = json.loads(buf_json.getvalue())
    self.assertEqual(payload["summary"]["deprecated_model_count"], 25)
    self.assertTrue(payload["summary"]["rule1_passed"])


if __name__ == "__main__":
  unittest.main()
