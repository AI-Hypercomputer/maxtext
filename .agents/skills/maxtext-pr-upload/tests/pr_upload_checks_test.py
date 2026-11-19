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

"""Torch-free unit tests and negative controls for maxtext-pr-upload scripts and KB."""

import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
from typing import Any
import unittest
from unittest import mock

_SKILL_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _SKILL_DIR / "scripts"
_RESOURCES_DIR = _SKILL_DIR / "resources"

_EXPECTED_SCRIPTS = (
    "abstract_model_diff.py",
    "audit_pr_review_hygiene.py",
    "audit_repo_and_ci_contracts.py",
    "build_local_review_prompt.py",
    "caller_impact_tests.py",
    "check_doc_train_cmds.py",
    "ci_status.py",
    "mini_aot_fingerprint.py",
    "pilot.py",
    "pr_comments.py",
    "review_scope_gate.py",
    "sweep_param_mapping_contracts.py",
)


def _load_script(script_filename: str) -> Any:
  """Loads a script module from .agents/skills/maxtext-pr-upload/scripts/."""
  mod_name = f"_mpu_test_{script_filename[:-3]}"
  cached = sys.modules.get(mod_name)
  if cached is not None:
    return cached
  script_path = _SCRIPTS_DIR / script_filename
  spec = importlib.util.spec_from_file_location(mod_name, script_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load {script_path}")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[mod_name] = mod
  spec.loader.exec_module(mod)
  return mod


def _init_git_repo(root: Path) -> None:
  """Initializes a hermetic git repository for diff and scope tests."""
  for args in (
      ["init", "-q"],
      ["config", "user.email", "test@example.com"],
      ["config", "user.name", "MaxText Test"],
  ):
    subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_commit_all(root: Path, message: str) -> str:
  """Stages all files in `root`, creates a commit, and returns HEAD SHA."""
  subprocess.run(
      ["git", "-c", "safe.directory=*", "-C", str(root), "add", "-A"],
      check=True,
      capture_output=True,
      text=True,
  )
  subprocess.run(
      [
          "git",
          "-c",
          "safe.directory=*",
          "-C",
          str(root),
          "commit",
          "-q",
          "-m",
          message,
      ],
      check=True,
      capture_output=True,
      text=True,
  )
  proc = subprocess.run(
      ["git", "-c", "safe.directory=*", "-C", str(root), "rev-parse", "HEAD"],
      check=True,
      capture_output=True,
      text=True,
  )
  return proc.stdout.strip()


class _FakeHttpResponse:
  """Minimal context-manager mock for urllib.request.urlopen responses."""

  def __init__(self, payload: Any) -> None:
    self._raw = json.dumps(payload).encode("utf-8")

  def read(self) -> bytes:
    return self._raw

  def __enter__(self) -> "_FakeHttpResponse":
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
    del exc_type, exc, tb
    return False


class PrUploadChecksTest(unittest.TestCase):
  """Verifies all maxtext-pr-upload scripts, KB resources, and negative controls."""

  def test_all_python_files_have_apache_header_and_clean_cli_surface(
      self,
  ) -> None:
    """Checks Apache 2.0 headers and ensures stripped internal flags stay absent."""
    py_files = [
        *(_SCRIPTS_DIR / name for name in _EXPECTED_SCRIPTS),
        Path(__file__).resolve(),
    ]
    for path in py_files:
      self.assertTrue(path.is_file(), f"Missing expected file: {path}")
      text = path.read_text(encoding="utf-8")
      self.assertIn("Licensed under the Apache License, Version 2.0", text[:600])
      ast.parse(text, filename=str(path))

    expected_cli_flags = {
        "audit_repo_and_ci_contracts.py": {
            "--allowlist-ref-dir",
            "--base",
            "--fail-on-warn",
            "--files",
            "--repo-root",
            "--run-torch-free-import",
            "--shared-legacy",
        },
        "audit_pr_review_hygiene.py": {
            "--base",
            "--diff-file",
            "--files",
            "--repo-root",
        },
        "build_local_review_prompt.py": {
            "--base",
            "--description-file",
            "--diff-file",
            "--findings-out",
            "--head",
            "--kb-ledger",
            "--kb-repo",
            "--max-inline-kb",
            "--max-kb-do-not-flag",
            "--max-kb-must-catch",
            "--maxtext-root",
            "--no-kb-ledger",
            "--out",
            "--target",
        },
        "review_scope_gate.py": {
            "--allow-unreviewed",
            "--base",
            "--changed-files",
            "--findings-file",
            "--head",
            "--repo-root",
            "--report",
            "--strict-pre-push",
        },
    }
    for script_name, expected in expected_cli_flags.items():
      src = (_SCRIPTS_DIR / script_name).read_text(encoding="utf-8")
      found = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
      self.assertEqual(found, expected, f"Unexpected CLI flags in {script_name}")

  def test_learned_lessons_ledger_conforms_to_schema_and_counts(self) -> None:
    """Validates learned_lessons.jsonl against learned_lessons_schema.json."""
    prompt_mod = _load_script("build_local_review_prompt.py")
    schema_path = _RESOURCES_DIR / "learned_lessons_schema.json"
    ledger_path = _RESOURCES_DIR / "learned_lessons.jsonl"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required_keys = set(schema["required"])
    allowed_keys = set(schema["properties"])
    kind_enum = set(schema["properties"]["kind"]["enum"])
    status_enum = set(schema["properties"]["status"]["enum"])
    fp_re = re.compile(schema["properties"]["fingerprint"]["pattern"])
    ts_re = re.compile(schema["properties"]["timestamp"]["pattern"])

    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    self.assertEqual(len(entries), 10)
    must_catch = [e for e in entries if e["kind"] == "must_catch"]
    do_not_flag = [e for e in entries if e["kind"] == "do_not_flag"]
    self.assertEqual(len(must_catch), 8)
    self.assertEqual(len(do_not_flag), 2)

    seen_fps: set[str] = set()
    for entry in entries:
      self.assertTrue(required_keys.issubset(entry))
      self.assertTrue(set(entry).issubset(allowed_keys))
      self.assertIn(entry["kind"], kind_enum)
      self.assertIn(entry.get("status", "active"), status_enum)
      self.assertGreaterEqual(len(entry["rule"]), 10)
      expected_fp = prompt_mod._kb_fingerprint(  # pylint: disable=protected-access
          entry["repo"], entry["kind"], entry["rule"]
      )
      self.assertEqual(entry["fingerprint"], expected_fp)
      self.assertRegex(entry["fingerprint"], fp_re)
      self.assertRegex(entry["timestamp"], ts_re)
      self.assertNotIn(entry["fingerprint"], seen_fps)
      seen_fps.add(entry["fingerprint"])

  def test_audit_repo_and_ci_contracts_positive_and_negative_controls(
      self,
  ) -> None:
    """Tests schema-backed SILENT_CONFIG_FALLBACK, TORCH_*, TF_*, and --base."""
    mod = _load_script("audit_repo_and_ci_contracts.py")
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _init_git_repo(root)
      (root / "pytest.ini").write_text("[pytest]\npython_files = *_test.py *_tests.py\n", encoding="utf-8")
      wf_dir = root / ".github" / "workflows"
      wf_dir.mkdir(parents=True)
      (wf_dir / "run_tests_coordinator.yml").write_text(
          "install_torch_cpu: ${{ github.event_name == 'schedule' }}\n",
          encoding="utf-8",
      )
      cfg_dir = root / "src" / "maxtext" / "configs"
      cfg_dir.mkdir(parents=True)
      (cfg_dir / "types.py").write_text(
          "class MaxTextConfig:\n  declared_flag: bool = False\n",
          encoding="utf-8",
      )
      layer_dir = root / "src" / "maxtext" / "layers"
      layer_dir.mkdir(parents=True)
      clean_layer = layer_dir / "decoder.py"
      clean_layer.write_text(
          "class Decoder:\n" "  def __init__(self, config):\n" "    self.flag = config.declared_flag\n",
          encoding="utf-8",
      )
      unit_dir = root / "tests" / "unit"
      unit_dir.mkdir(parents=True)
      clean_test = unit_dir / "decoder_test.py"
      clean_test.write_text(
          "import unittest\n\n"
          "class DecoderTest(unittest.TestCase):\n"
          "  def test_ok(self):\n"
          "    self.assertEqual(1 + 1, 2)\n",
          encoding="utf-8",
      )
      base_sha = _git_commit_all(root, "Initial clean baseline")

      clean_rep = mod.audit_files(
          repo_root=str(root),
          files=[str(clean_layer), str(clean_test)],
          run_torch_free_import=True,
      )
      self.assertEqual(clean_rep["status"], "PASS")
      self.assertEqual(clean_rep["issue_count"], 0)

      # Negative control 1: undeclared getattr/hasattr + TF import + torch import
      clean_layer.write_text(
          "import tensorflow as tf\n\n"
          "class Decoder:\n"
          "  def __init__(self, config):\n"
          "    self.bad = getattr(config, 'typo_undeclared_flag', False)\n"
          "    self.probe = hasattr(config, 'other_missing_flag')\n",
          encoding="utf-8",
      )
      bad_test = unit_dir / "bad_torch_test.py"
      bad_test.write_text(
          "import pytest\n" "import torch\n" "pytest.importorskip('torch')\n",
          encoding="utf-8",
      )
      _git_commit_all(root, "Introduce contract violations")

      buf = io.StringIO()
      with mock.patch("sys.stdout", buf):
        rc_bad = mod.main(
            [
                "--repo-root",
                str(root),
                "--base",
                base_sha,
            ]
        )
      self.assertEqual(rc_bad, 1)
      bad_rep = json.loads(buf.getvalue())
      codes = {item["code"] for item in bad_rep["issues"]}
      self.assertIn("SILENT_CONFIG_FALLBACK", codes)
      self.assertIn("TF_TOP_LEVEL_IMPORT", codes)
      self.assertIn("TORCH_TOP_LEVEL_IMPORT", codes)
      self.assertIn("TORCH_MODULE_IMPORTORSKIP", codes)

      # Reverting violations restores exit 0
      clean_layer.write_text(
          "class Decoder:\n" "  def __init__(self, config):\n" "    self.flag = config.declared_flag\n",
          encoding="utf-8",
      )
      bad_test.unlink()
      _git_commit_all(root, "Revert violations")
      rc_fixed = mod.main(
          [
              "--repo-root",
              str(root),
              "--files",
              str(clean_layer),
              str(clean_test),
              "--run-torch-free-import",
          ]
      )
      self.assertEqual(rc_fixed, 0)

  def test_audit_pr_review_hygiene_positive_and_negative_controls(self) -> None:
    """Tests AST and shell hygiene rules alongside positive controls."""
    mod = _load_script("audit_pr_review_hygiene.py")
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      clean_py = root / "offload_helper.py"
      clean_py.write_text(
          "import jax\n\n"
          "def route_buffer(x):\n"
          "  if isinstance(x, jax.core.Tracer):\n"
          "    return x\n"
          "  return x\n",
          encoding="utf-8",
      )
      clean_sh = root / "bench_clean.sh"
      clean_sh.write_text(
          "#!/bin/bash\n" "export LIBTPU_INIT_ARGS='--xla_tpu_enable_ragged_dot_kernel=false' # intentional-baseline\n",
          encoding="utf-8",
      )
      self.assertEqual(
          mod.main(
              [
                  "--repo-root",
                  str(root),
                  "--files",
                  str(clean_py),
                  str(clean_sh),
              ]
          ),
          0,
      )

      # Negative controls across Python AST, test validity, and shell scripts
      bad_py = root / "bad_layer.py"
      bad_py.write_text(
          "import jax\n"
          "from orbax.checkpoint import _src as private_orbax\n\n"
          "def validate(x, hf_config_dict):\n"
          "  hf_config_dict['quantization_config'] = None\n"
          "  if not isinstance(x, jax.core.Tracer):\n"
          "    if float(x) < 0:\n"
          "      raise ValueError('bad')\n"
          "  return private_orbax\n",
          encoding="utf-8",
      )
      bad_test = root / "tests" / "unit" / "laundered_test.py"
      bad_test.parent.mkdir(parents=True)
      bad_test.write_text(
          "import jax\n"
          "import unittest\n\n"
          "class LaunderedTest(unittest.TestCase):\n"
          "  def test_silent_guard_and_lower_hlo(self, fn, x):\n"
          "    if jax.device_count() < 8:\n"
          "      return\n"
          "    hlo = jax.jit(fn).lower(x).as_text()\n"
          "    self.assertIn('all-gather', hlo)\n",
          encoding="utf-8",
      )
      bad_sh = root / "bench_bad.sh"
      bad_sh.write_text(
          "#!/bin/bash\n" "export LIBTPU_INIT_ARGS='--xla_tpu_enable_ragged_dot_kernel=false'\n",
          encoding="utf-8",
      )

      rep = mod.audit_hygiene(
          repo_root=str(root),
          files=[str(bad_py), str(bad_test), str(bad_sh)],
      )
      self.assertEqual(rep["status"], "FAIL")
      codes = {item["code"] for item in rep["issues"]}
      self.assertIn("PRIVATE_THIRD_PARTY_SRC_ACCESS", codes)
      self.assertIn("DICT_NONE_OVERWRITE_HAZARD", codes)
      self.assertIn("TRACER_TYPECHECK_BYPASSES_VALIDATION", codes)
      self.assertIn("SILENT_DEVICE_COUNT_GUARD", codes)
      self.assertIn("HLO_REGEX_ON_UNCOMPILED_LOWERING", codes)
      self.assertIn("BENCHMARK_CRIPPLED_XLA_FLAG", codes)

  def test_sweep_param_mapping_contracts_positive_and_negative_controls(
      self,
  ) -> None:
    """Verifies BF16, null-subconfig, to_dict, and FP8 streaming contract gates."""
    mod = _load_script("sweep_param_mapping_contracts.py")

    class _GoodHfConfig:
      """Mock HuggingFace PretrainedConfig."""

      def to_dict(self) -> dict[str, Any]:
        return {"num_hidden_layers": 2, "quantization_config": {"bits": 8}}

    def _good_mapping(hf_cfg: dict[str, Any], mt_cfg: Any, scan_layers: bool = False) -> dict[str, str]:
      del mt_cfg, scan_layers
      qcfg = hf_cfg.get("quantization_config") or {}
      _ = qcfg.get("bits", 16)
      return {"params-proj": "model.proj.weight"}

    def _good_ckpt_qcfg(path: str) -> dict[str, Any]:
      with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
      tc = raw.get("text_config")
      sub = tc if isinstance(tc, dict) else {}
      return raw.get("quantization_config") or sub.get("quantization_config") or {}

    def _good_get_hf(mt_cfg: Any, path: str = "") -> dict[str, str]:
      cfg = _GoodHfConfig().to_dict()
      qcfg = _good_ckpt_qcfg(path) if path else {}
      if qcfg:
        cfg["quantization_config"] = qcfg
      else:
        cfg.pop("quantization_config", None)
      return _good_mapping(cfg, mt_cfg, scan_layers=mt_cfg.scan_layers)

    clean_mods = {
        "param_mapping": types.SimpleNamespace(PARAM_MAPPING={"demo-model": _good_mapping}, HOOK_FNS={}),
        "hf_model_configs": types.SimpleNamespace(HF_MODEL_CONFIGS={"demo-model": _GoodHfConfig()}),
        "load_dynamic": types.SimpleNamespace(
            get_hf_config_and_mappings=_good_get_hf,
            _checkpoint_quantization_config=_good_ckpt_qcfg,
        ),
        "hf_streaming_load": None,
    }
    issues, swept = mod._sweep_param_mappings(clean_mods, None)  # pylint: disable=protected-access
    self.assertEqual(swept, 1)
    self.assertEqual(issues, [])

    # Negative control: setting quantization_config = None crashes .get(...).get(...)
    def _buggy_get_hf(mt_cfg: Any, path: str = "") -> dict[str, str]:
      del path
      cfg = {"num_hidden_layers": 2, "quantization_config": None}
      _ = cfg.get("quantization_config", {}).get("bits")  # type: ignore[union-attr]
      return _good_mapping(cfg, mt_cfg)

    def _buggy_ckpt_qcfg(path: str) -> dict[str, Any]:
      with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
      return raw.get("text_config", {}).get("quantization_config", {})

    bad_mods = {
        **clean_mods,
        "load_dynamic": types.SimpleNamespace(
            get_hf_config_and_mappings=_buggy_get_hf,
            _checkpoint_quantization_config=_buggy_ckpt_qcfg,
        ),
    }
    bad_issues, _ = mod._sweep_param_mappings(bad_mods, None)  # pylint: disable=protected-access
    bad_codes = {item["code"] for item in bad_issues}
    self.assertIn("NULL_SUBCONFIG_JSON_CRASH", bad_codes)
    self.assertIn("BF16_CONFIG_MAPPING_CRASH", bad_codes)

    # Negative control: FP8 streaming module omitting _check_float8_sources
    with tempfile.TemporaryDirectory() as tmp:
      fake_hsl_path = Path(tmp) / "fake_hf_streaming_load.py"
      fake_hsl_path.write_text(
          "# Handles float8 streaming loads without scale validation\n"
          "def load_hf_params_streaming(path, target_tree):\n"
          "  return target_tree\n",
          encoding="utf-8",
      )
      spec = importlib.util.spec_from_file_location("_fake_hsl_mod", fake_hsl_path)
      assert spec is not None and spec.loader is not None
      fake_hsl = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(fake_hsl)
      fp8_issues = mod._check_streaming_fp8_and_sharding_contracts(  # pylint: disable=protected-access
          {"hf_streaming_load": fake_hsl}
      )
      self.assertIn(
          "UNSCALED_FP8_TO_BF16_SILENT_CAST",
          {item["code"] for item in fp8_issues},
      )

  def test_build_local_review_prompt_kb_filtering_and_git_fallback(self) -> None:
    """Tests subsystem KB filtering on learned_lessons.jsonl and git fallback."""
    mod = _load_script("build_local_review_prompt.py")
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _init_git_repo(root)
      cmd_dir = root / ".gemini" / "commands"
      cmd_dir.mkdir(parents=True)
      (cmd_dir / "maxtext-review-criteria.md").write_text(
          "# MaxText Criteria\nCheck numerical parity.\n", encoding="utf-8"
      )
      (cmd_dir / "maxtext-review-performance.md").write_text(
          "# Performance Focus\nCheck remat and sharding.\n", encoding="utf-8"
      )
      dec_file = root / "src" / "maxtext" / "layers" / "nnx_decoders.py"
      dec_file.parent.mkdir(parents=True)
      dec_file.write_text("def block(x):\n  return x\n", encoding="utf-8")
      base_sha = _git_commit_all(root, "Base commit")

      dec_file.write_text("def block(x, remat_policy='full'):\n  return x\n", encoding="utf-8")
      head_sha = _git_commit_all(root, "Modify nnx_decoders")

      prompt_out = root / "prompt.md"
      findings_out = root / "findings.md"
      with mock.patch.dict(os.environ, {"MAXTEXT_REVIEW_KB_LEDGER": ""}, clear=False):
        rc = mod.main(
            [
                "--maxtext-root",
                str(root),
                "--base",
                base_sha,
                "--head",
                head_sha,
                "--target",
                "PR#9999",
                "--findings-out",
                str(findings_out),
                "--out",
                str(prompt_out),
            ]
        )
      self.assertEqual(rc, 0)
      rendered = prompt_out.read_text(encoding="utf-8")
      self.assertIn("## Team Review Knowledge Base", rendered)
      kb_sec = rendered.split("## Team Review Knowledge Base", 1)[1].split("-----", 1)[0]
      self.assertIn("remat_policy == 'none'", kb_sec)
      self.assertIn("isinstance(x, jax.core.Tracer)", kb_sec)
      self.assertNotIn("MAXTEXT_CONFIGS_DIR", kb_sec)

      moe_mc, moe_dnf = mod.select_kb_lessons(
          [str(_RESOURCES_DIR / "learned_lessons.jsonl")],
          ["src/maxtext/m3/core/moe.py", "tests/unit/moe_test.py"],
      )
      moe_rules = "\n".join(item["rule"] for item in moe_mc)
      self.assertIn("count > cap", moe_rules)
      self.assertIn("MAXTEXT_CONFIGS_DIR", moe_rules)
      self.assertIn("@unittest.skipUnless(HAS_TORCH)", moe_rules)
      self.assertTrue(any("Track Test Duration" in item["rule"] for item in moe_dnf))

      # --no-kb-ledger disables KB section
      prompt_no_kb = root / "prompt_no_kb.md"
      self.assertEqual(
          mod.main(
              [
                  "--maxtext-root",
                  str(root),
                  "--base",
                  base_sha,
                  "--head",
                  head_sha,
                  "--target",
                  "PR#9999",
                  "--no-kb-ledger",
                  "--findings-out",
                  str(findings_out),
                  "--out",
                  str(prompt_no_kb),
              ]
          ),
          0,
      )
      self.assertNotIn(
          "## Team Review Knowledge Base",
          prompt_no_kb.read_text(encoding="utf-8"),
      )

      # Removing working-tree .gemini/commands falls back to git HEAD:.gemini/commands/
      for p in cmd_dir.glob("*.md"):
        p.unlink()
      prompt_fb = root / "prompt_fallback.md"
      self.assertEqual(
          mod.main(
              [
                  "--maxtext-root",
                  str(root),
                  "--base",
                  base_sha,
                  "--head",
                  head_sha,
                  "--target",
                  "PR#9999",
                  "--findings-out",
                  str(findings_out),
                  "--out",
                  str(prompt_fb),
              ]
          ),
          0,
      )
      self.assertIn("# MaxText Criteria", prompt_fb.read_text(encoding="utf-8"))

  def test_review_scope_gate_strict_pre_push_and_negative_controls(
      self,
  ) -> None:
    """Verifies strict pre-push verdict, scope coverage, and YELLOW defect gates."""
    mod = _load_script("review_scope_gate.py")
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _init_git_repo(root)
      f1 = root / "src" / "a.py"
      f2 = root / "src" / "b.py"
      f1.parent.mkdir(parents=True)
      f1.write_text("x = 1\n", encoding="utf-8")
      f2.write_text("y = 1\n", encoding="utf-8")
      base_sha = _git_commit_all(root, "Base")

      f1.write_text("x = 2\n", encoding="utf-8")
      f2.write_text("y = 2\n", encoding="utf-8")
      head_sha = _git_commit_all(root, "Head")

      findings = root / "local_review.md"
      findings.write_text(
          f"LOCAL_REVIEW target=PR#42 head={head_sha}\n\n"
          "## Findings\n"
          "- severity=RED file=src/a.py line=1 status=fixed :: Fixed issue\n"
          '- severity=ORANGE file=src/b.py line=1 status=dismissed reason="Verified safe in caller" :: Checked\n'
          "- severity=YELLOW file=src/b.py line=1 status=open :: [nit] Minor naming preference\n",
          encoding="utf-8",
      )
      good_report = root / "gate_c.md"
      good_report.write_text(
          "# Review\n\n"
          "```review-scope\n"
          "target: PR#42\n"
          f"base: {base_sha}\n"
          f"head: {head_sha}\n"
          "verdict: APPROVED\n"
          "files:\n"
          "- src/a.py: verified x update and callers\n"
          "- src/b.py: verified y update and types\n"
          "```\n",
          encoding="utf-8",
      )
      self.assertEqual(
          mod.main(
              [
                  "--report",
                  str(good_report),
                  "--repo-root",
                  str(root),
                  "--base",
                  base_sha,
                  "--head",
                  head_sha,
                  "--findings-file",
                  str(findings),
                  "--strict-pre-push",
              ]
          ),
          0,
      )

      # Negative control A: APPROVED_WITH_FOLLOWUPS is banned
      bad_verdict = root / "bad_verdict.md"
      bad_verdict.write_text(
          good_report.read_text(encoding="utf-8").replace("verdict: APPROVED", "verdict: APPROVED_WITH_FOLLOWUPS"),
          encoding="utf-8",
      )
      res_v = mod.run_gate(
          bad_verdict.read_text(encoding="utf-8"),
          ["src/a.py", "src/b.py"],
          base_sha,
          head_sha,
          repo=str(root),
          strict_pre_push=True,
      )
      self.assertEqual(res_v["status"], "FAIL")
      self.assertIn("INVALID_VERDICT", {i["code"] for i in res_v["issues"]})

      # Negative control B: CHANGES_REQUESTED fails --strict-pre-push
      cr_report = good_report.read_text(encoding="utf-8").replace("verdict: APPROVED", "verdict: CHANGES_REQUESTED")
      res_cr = mod.run_gate(
          cr_report,
          ["src/a.py", "src/b.py"],
          base_sha,
          head_sha,
          repo=str(root),
          strict_pre_push=True,
      )
      self.assertIn("NOT_APPROVED_FOR_PUSH", {i["code"] for i in res_cr["issues"]})

      # Negative control C: open defect YELLOW in findings_file is rejected
      bad_findings = root / "bad_findings.md"
      bad_findings.write_text(
          f"LOCAL_REVIEW target=PR#42 head={head_sha}\n\n"
          "- severity=YELLOW file=src/a.py line=1 status=open :: Validator bug causes crash on empty input\n",
          encoding="utf-8",
      )
      self.assertEqual(
          mod.main(
              [
                  "--report",
                  str(good_report),
                  "--repo-root",
                  str(root),
                  "--base",
                  base_sha,
                  "--head",
                  head_sha,
                  "--findings-file",
                  str(bad_findings),
                  "--strict-pre-push",
              ]
          ),
          1,
      )

      # Negative control D: uncovered file or skip note fails
      skip_report = good_report.read_text(encoding="utf-8").replace("verified y update and types", "unchanged")
      res_skip = mod.run_gate(skip_report, ["src/a.py", "src/b.py"], base_sha, head_sha, repo=str(root))
      self.assertIn("UNCOVERED_FILE", {i["code"] for i in res_skip["issues"]})

  def test_ci_status_attempt_scoped_jobs_and_verdicts(self) -> None:
    """Verifies per_page=100, attempts/{attempt}/jobs?per_page=25, and verdicts."""
    mod = _load_script("ci_status.py")
    seen_urls: list[str] = []

    def _fake_urlopen(req: Any, timeout: int = 30) -> _FakeHttpResponse:
      del timeout
      url = req.full_url
      seen_urls.append(url)
      if url.endswith("/pulls/101"):
        return _FakeHttpResponse(
            {
                "head": {"sha": "abcdef1234567890"},
                "commits": 1,
                "labels": [{"name": "pull ready"}],
            }
        )
      if "/actions/runs?head_sha=abcdef1234567890" in url:
        return _FakeHttpResponse(
            {
                "workflow_runs": [
                    {
                        "id": 555,
                        "name": "Track Test Performance",
                        "event": "pull_request",
                        "run_number": 2,
                        "run_attempt": 2,
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ]
            }
        )
      if "/actions/runs/555/attempts/2/jobs?per_page=25&page=1" in url:
        return _FakeHttpResponse(
            {
                "jobs": [
                    {
                        "name": "Track Test Duration",
                        "conclusion": "failure",
                        "steps": [{"name": "Compare", "conclusion": "failure"}],
                    }
                ]
            }
        )
      if "/commits/abcdef1234567890/check-runs?per_page=100&page=1" in url:
        return _FakeHttpResponse(
            {
                "check_runs": [
                    {
                        "id": 1,
                        "name": "Require PR Checklist",
                        "status": "completed",
                        "conclusion": "success",
                    }
                ]
            }
        )
      raise AssertionError(f"Unexpected URL: {url}")

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
      info = mod.summarize_pr_ci(101, throttle_s=0.0, repo="org/repo")
    self.assertEqual(info["verdict"], "PULL_READY")
    self.assertEqual(info["blocking_failures"], [])
    self.assertEqual(
        info["non_blocking_failures"],
        ["Track Test Performance / Track Test Duration"],
    )
    self.assertTrue(
        any(
            "https://api.github.com/repos/org/repo/actions/runs/555/attempts/2/jobs?per_page=25&page=1" in u
            for u in seen_urls
        )
    )

    # Negative control: real Run Tests failure -> FAILING
    def _failing_urlopen(req: Any, timeout: int = 30) -> _FakeHttpResponse:
      del timeout
      url = req.full_url
      if url.endswith("/pulls/102"):
        return _FakeHttpResponse(
            {
                "head": {"sha": "9999999999999999"},
                "commits": 1,
                "labels": [],
            }
        )
      if "/actions/runs?head_sha=9999999999999999" in url:
        return _FakeHttpResponse(
            {
                "workflow_runs": [
                    {
                        "id": 777,
                        "name": "MaxText Package Tests",
                        "event": "pull_request",
                        "run_number": 1,
                        "run_attempt": 1,
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ]
            }
        )
      if "/actions/runs/777/attempts/1/jobs?per_page=25&page=1" in url:
        return _FakeHttpResponse(
            {
                "jobs": [
                    {
                        "name": "cpu-unit (1)",
                        "conclusion": "failure",
                        "steps": [
                            {"name": "Run Tests", "conclusion": "failure"},
                            {"name": "Upload Test Results XML", "conclusion": "success"},
                        ],
                    }
                ]
            }
        )
      if "/commits/9999999999999999/check-runs" in url:
        return _FakeHttpResponse({"check_runs": []})
      raise AssertionError(f"Unexpected URL: {url}")

    with mock.patch("urllib.request.urlopen", side_effect=_failing_urlopen):
      bad_info = mod.summarize_pr_ci(102, throttle_s=0.0, repo="org/repo")
    self.assertEqual(bad_info["verdict"], "FAILING")
    self.assertIn("[test_failure]", bad_info["blocking_failures"][0])

  def test_pr_comments_inline_and_top_level_unanswered_detection(self) -> None:
    """Verifies inline thread grouping and [TOP-LEVEL] issue_comments checks."""
    mod = _load_script("pr_comments.py")
    inline = [
        {
            "id": 10,
            "path": "src/a.py",
            "line": 12,
            "user": {"login": "reviewer1"},
            "created_at": "2026-11-18T10:00:00Z",
            "body": "Please check dtype here.",
        },
        {
            "id": 11,
            "in_reply_to_id": 10,
            "path": "src/a.py",
            "line": 12,
            "user": {"login": "pr-author"},
            "created_at": "2026-11-18T10:05:00Z",
            "body": "Fixed in latest commit.",
        },
        {
            "id": 20,
            "path": "src/b.py",
            "line": 5,
            "user": {"login": "reviewer2"},
            "created_at": "2026-11-18T11:00:00Z",
            "body": "What happens when batch=1?",
        },
    ]
    threads = mod.group_inline_threads(inline, "pr-author")
    by_root = {t["root_id"]: t for t in threads}
    self.assertFalse(by_root[10]["needs_reply"])
    self.assertTrue(by_root[20]["needs_reply"])

    issue_comments = [
        {
            "id": 1,
            "user": {"login": "github-actions[bot]", "type": "Bot"},
            "created_at": "2026-11-18T09:00:00Z",
            "body": "## Summary of Changes\nAutomated banner.",
        },
        {
            "id": 2,
            "user": {"login": "reviewer1", "type": "User"},
            "created_at": "2026-11-18T12:00:00Z",
            "body": "Seeing a +400ms step-time regression on 64 chips.",
        },
    ]
    top_open = mod.find_top_level_comments(issue_comments, "pr-author")
    self.assertEqual(len(top_open), 1)
    self.assertTrue(top_open[0]["needs_reply"])

    issue_comments.append(
        {
            "id": 3,
            "user": {"login": "pr-author", "type": "User"},
            "created_at": "2026-11-18T12:30:00Z",
            "body": "Moved mask outside checkpoint; step time restored.",
        }
    )
    top_answered = mod.find_top_level_comments(issue_comments, "pr-author")
    self.assertFalse(top_answered[0]["needs_reply"])

  def test_check_doc_train_cmds_positive_and_backslash_comment_negative(
      self,
  ) -> None:
    """Verifies Markdown command extraction and '#' continuation truncation gate."""
    mod = _load_script("check_doc_train_cmds.py")
    subs = mod.parse_substitutions(["RUN=demo_run", "OUT=/tmp/out"])
    good_block = (
        "python3 -m maxtext.train src/maxtext/configs/base.yml \\\n"
        "  run_name=${RUN} \\\n"
        "  base_output_directory=$OUT \\\n"
        "  override_model_config=true"
    )
    argv, truncated = mod.build_pyconfig_argv(good_block, subs)
    self.assertFalse(truncated)
    self.assertIn("run_name=demo_run", argv)
    self.assertIn("override_model_config=true", argv)

    # Negative control: '#' comment inside '\' continuation truncates args
    bad_block = (
        "python3 -m maxtext.train src/maxtext/configs/base.yml \\\n"
        "  # broken inline comment inside continuation\n"
        "  override_model_config=true"
    )
    _, bad_truncated = mod.build_pyconfig_argv(bad_block, subs)
    self.assertTrue(bad_truncated)

    with tempfile.TemporaryDirectory() as tmp:
      doc_path = Path(tmp) / "guide.md"
      doc_path.write_text(
          "## Start\n```bash\n" + bad_block + "\n```\n## End\n",
          encoding="utf-8",
      )
      fake_pyconfig = types.SimpleNamespace(
          initialize=lambda a: types.SimpleNamespace(model_name="default", override_model_config=False)
      )
      fake_configs = types.ModuleType("maxtext.configs")
      fake_configs.pyconfig = fake_pyconfig  # type: ignore[attr-defined]
      fake_maxtext = types.ModuleType("maxtext")
      fake_maxtext.configs = fake_configs  # type: ignore[attr-defined]
      with mock.patch.dict(
          sys.modules,
          {"maxtext": fake_maxtext, "maxtext.configs": fake_configs},
      ):
        with mock.patch.object(
            sys,
            "argv",
            ["check_doc_train_cmds.py", str(doc_path), "--start", "## Start", "--end", "## End"],
        ):
          self.assertEqual(mod.main(), 1)

  def test_tiers_a_b_c_and_pilot_positive_and_negative_controls(self) -> None:
    """Tests abstract_model_diff, mini_aot_fingerprint, caller_impact_tests, and pilot."""
    amd = _load_script("abstract_model_diff.py")
    maf = _load_script("mini_aot_fingerprint.py")
    cit = _load_script("caller_impact_tests.py")
    pilot = _load_script("pilot.py")

    with tempfile.TemporaryDirectory() as tmp:
      tdir = Path(tmp)
      # 1. Tier A compare positive & MUON_SCAN_AXIS_FOLDED negative control
      b_dir = tdir / "A_base"
      h_dir = tdir / "A_head"
      b_dir.mkdir()
      h_dir.mkdir()
      base_rec = {
          "model": "demo-8b",
          "variants": {
              "scan=True": {
                  "status": "ok",
                  "params": {
                      "w": {
                          "shape": [16, 4, 64],
                          "shard_shape": [8, 4, 32],
                          "dtype": "bfloat16",
                          "logical": {"param_scan_axis": 1},
                          "spec": "PartitionSpec('fsdp', None, 'tensor')",
                          "type": "Param",
                      }
                  },
                  "muon": {"['w']": "([0], [2])"},
              }
          },
      }
      (b_dir / "demo-8b.json").write_text(json.dumps(base_rec), encoding="utf-8")
      (h_dir / "demo-8b.json").write_text(json.dumps(base_rec), encoding="utf-8")
      self.assertEqual(amd.compare(str(b_dir), str(h_dir), ""), {})

      bad_rec = json.loads(json.dumps(base_rec))
      bad_rec["variants"]["scan=True"]["muon"]["['w']"] = "([0], [1, 2])"
      (h_dir / "demo-8b.json").write_text(json.dumps(bad_rec), encoding="utf-8")
      diff_a = amd.compare(str(b_dir), str(h_dir), "")
      self.assertTrue(any("MUON_SCAN_AXIS_FOLDED" in item for item in diff_a["demo-8b"]))

      # 2. Tier B fingerprint, batch ok==0 non-zero exit, and DOT_FP32_UPCAST
      hlo_sample = (
          "HloModule jit_train_step\n"
          "ENTRY main {\n"
          "  %p0 = bf16[2,256,128] parameter(0)\n"
          "  %p1 = bf16[128,128] parameter(1)\n"
          "  %ag = bf16[2,256,128] all-gather(%p0)\n"
          "  %d = bf16[2,256,128] dot(%ag, %p1)\n"
          '  %m = pred[2,256,256] custom-call(), custom_call_target="tpu_custom_call"\n'
          "}\n"
      )
      fp = maf.fingerprint(hlo_sample, "temp_size_in_bytes=10000")
      self.assertEqual(fp["collectives"]["all-gather bf16[2,256,128]"], 1)
      self.assertEqual(fp["pred_seq_seq"]["2,256,256"], 1)
      self.assertEqual(fp["memory"]["temp_size_in_bytes"], 10000)

      # Empty batch succeeds and writes batch_summary.json
      empty_batch_dir = tdir / "B_empty"
      with mock.patch.object(
          sys,
          "argv",
          ["mini_aot_fingerprint.py", "batch", "--worktree", str(tdir), "--models", "", "--out", str(empty_batch_dir)],
      ):
        self.assertEqual(maf.main(), 0)
      self.assertTrue((empty_batch_dir / "batch_summary.json").is_file())

      # Batch where all requested models fail exits 2
      fail_batch_dir = tdir / "B_fail"
      with mock.patch.object(
          maf,
          "compile_one",
          return_value={"model": "broken", "rc": 1, "error": "boom"},
      ):
        with mock.patch.object(
            sys,
            "argv",
            [
                "mini_aot_fingerprint.py",
                "batch",
                "--worktree",
                str(tdir),
                "--models",
                "broken",
                "--out",
                str(fail_batch_dir),
            ],
        ):
          self.assertEqual(maf.main(), 2)

      # Compare catches DOT_FP32_UPCAST
      bb_dir = tdir / "B_base" / "demo-8b"
      bh_dir = tdir / "B_head" / "demo-8b"
      bb_dir.mkdir(parents=True)
      bh_dir.mkdir(parents=True)
      (bb_dir / "fp.json").write_text(
          json.dumps({"model": "demo-8b", "layers": 2, "fingerprint": fp}),
          encoding="utf-8",
      )
      fp_bad = json.loads(json.dumps(fp))
      fp_bad["dots"]["dot out=f32 lhs=f32 rhs=f32"] = 2
      (bh_dir / "fp.json").write_text(
          json.dumps({"model": "demo-8b", "layers": 2, "fingerprint": fp_bad}),
          encoding="utf-8",
      )
      diff_b = maf.compare(str(tdir / "B_base"), str(tdir / "B_head"), "")
      self.assertTrue(any("DOT_FP32_UPCAST" in x for x in diff_b["demo-8b"]))

      # 3. Tier C caller_impact_tests select + head-vs-base regression detection
      wt_base = tdir / "wt_base"
      wt_head = tdir / "wt_head"
      for wt in (wt_base, wt_head):
        (wt / "src" / "maxtext" / "configs" / "models").mkdir(parents=True)
        (wt / "src" / "maxtext" / "models").mkdir(parents=True)
        (wt / "src" / "maxtext" / "__init__.py").write_text("", encoding="utf-8")
        (wt / "src" / "maxtext" / "models" / "__init__.py").write_text("", encoding="utf-8")
        (wt / "tests" / "unit").mkdir(parents=True)
        (wt / "pytest.ini").write_text("[pytest]\npython_files = *_test.py\n", encoding="utf-8")
        (wt / "src" / "maxtext" / "configs" / "models" / "mini.yml").write_text(
            "decoder_block: mini_blk\n", encoding="utf-8"
        )
      _init_git_repo(wt_head)
      (wt_base / "src" / "maxtext" / "models" / "mini_blk.py").write_text(
          "def compute_scale(x):\n  return x * 2\n", encoding="utf-8"
      )
      (wt_head / "src" / "maxtext" / "models" / "mini_blk.py").write_text(
          "def compute_scale(x):\n  return x * 2\n", encoding="utf-8"
      )
      for wt in (wt_base, wt_head):
        (wt / "tests" / "unit" / "mini_blk_test.py").write_text(
            "from maxtext.models import mini_blk\n\n" "def test_scale():\n" "  assert mini_blk.compute_scale(3) == 6\n",
            encoding="utf-8",
        )
      base_ref = _git_commit_all(wt_head, "Base worktree state")
      # Introduce a regression in wt_head
      (wt_head / "src" / "maxtext" / "models" / "mini_blk.py").write_text(
          "def compute_scale(x):\n  return x * 999\n", encoding="utf-8"
      )
      _git_commit_all(wt_head, "Break compute_scale")

      sel = cit.select(str(wt_base), str(wt_head), base_ref)
      self.assertIn("tests/unit/mini_blk_test.py", sel["tests"])
      self.assertIn("mini", sel["affected_models"])

      if importlib.util.find_spec("pytest") is not None:
        res_reg = cit.run(sel, str(wt_base), str(wt_head), str(tdir / "C_out"), jobs=1)
        self.assertEqual(len(res_reg["regressions"]), 1)

        (wt_head / "src" / "maxtext" / "models" / "mini_blk.py").write_text(
            "def compute_scale(x):\n  return x * 2\n", encoding="utf-8"
        )
        res_ok = cit.run(sel, str(wt_base), str(wt_head), str(tdir / "C_out_ok"), jobs=1)
        self.assertEqual(res_ok["regressions"], [])
      else:

        def _fake_pytest(wt: str, rel: str, xml: str, **kwargs: Any) -> dict[str, Any]:
          del xml, kwargs
          is_head = Path(wt) == wt_head
          return {
              "rc": 1 if is_head else 0,
              "seconds": 0,
              "failed": [f"{rel}::test_scale"] if is_head else [],
              "counts": {"tests": 1, "failures": int(is_head), "errors": 0, "skipped": 0},
              "tail": "",
          }

        with mock.patch.object(cit, "_pytest", side_effect=_fake_pytest):
          res_reg = cit.run(sel, str(wt_base), str(wt_head), str(tdir / "C_out"), jobs=1)
        self.assertEqual(res_reg["regressions"], ["tests/unit/mini_blk_test.py::test_scale"])

      # 4. pilot pick_tier_b variant selection & MAXTEXT_PILOT_PYTHON override
      picks = pilot.pick_tier_b(
          {
              "changed_files": [
                  "src/maxtext/layers/nnx_decoders.py",
                  "src/maxtext/m3/core/offload.py",
                  "src/maxtext/m3/core/moe.py",
              ],
              "affected_models": {"llama3-8b": "llama2"},
              "model_distance": {"llama3-8b": 1},
          },
          named=[],
          cap=4,
          all_names=["llama3-8b", "m3__qwen3-0.6b", "m3__qwen3-30b-a3b"],
      )
      self.assertIn("m3__qwen3-0.6b@remat_none", picks)
      self.assertIn("m3__qwen3-0.6b@offload_param", picks)
      self.assertIn("m3__qwen3-0.6b@moe_ragged_sort", picks)
      self.assertIn("m3__qwen3-30b-a3b@moe_dropless_ragged_sort", picks)

      with mock.patch.dict(os.environ, {"MAXTEXT_PILOT_PYTHON": "/custom/bin/python3"}):
        self.assertEqual(
            pilot._default_python(),  # pylint: disable=protected-access
            "/custom/bin/python3",
        )

  def test_review_scope_gate_matches_leading_dot_directories_and_files(
      self,
  ) -> None:
    """Verifies dot-prefixed paths (.agents/, .github/, ./...) match in scope gate."""
    mod = _load_script("review_scope_gate.py")
    changed = [
        ".agents/skills/maxtext-pr-upload/SKILL.md",
        ".agents/skills/maxtext-pr-upload/references/review_gates.md",
        ".github/workflows/ci.yml",
        ".pre-commit-config.yaml",
    ]
    report = (
        "# Review\n\n"
        "```review-scope\n"
        "target: PR#100\n"
        "base: 11111111\n"
        "head: 22222222\n"
        "verdict: APPROVED\n"
        "files:\n"
        "- .agents/skills/maxtext-pr-upload/SKILL.md: verified skill workflow\n"
        "- ./.agents/skills/maxtext-pr-upload/references/review_gates.md: verified gates\n"
        "- .github/workflows/ci.yml: verified workflow triggers\n"
        "- .pre-commit-config.yaml: verified linter hooks\n"
        "```\n"
    )
    res = mod.run_gate(
        report,
        changed,
        "11111111",
        "22222222",
        strict_pre_push=True,
    )
    self.assertEqual(res["status"], "PASS", res)
    self.assertEqual(res["listed_not_in_diff"], [])
    self.assertEqual(res["issues"], [])

  def test_pilot_main_propagates_nonzero_tier_exits_batch_rc_and_python_override(
      self,
  ) -> None:
    """Verifies pilot.main() fails closed on Tier A/B/C errors and forwards --python."""
    pilot = _load_script("pilot.py")
    maf = _load_script("mini_aot_fingerprint.py")

    # 1. @dot_product compile failure is flagged even when base lacks fingerprint
    dp_diff = maf._diff_pair(  # pylint: disable=protected-access
        "m3__qwen3-0.6b@dot_product",
        {"model": "m3__qwen3-0.6b@dot_product", "rc": 1, "error": "missing"},
        {"model": "m3__qwen3-0.6b@dot_product", "rc": 1, "error": "dot crash"},
    )
    self.assertTrue(any("OPT_IN_VARIANT_COMPILE_FAIL" in item for item in dp_diff))
    self.assertNotEqual(
        pilot._python_env_tag("/venv_a/bin/python"),  # pylint: disable=protected-access
        pilot._python_env_tag("/venv_b/bin/python"),  # pylint: disable=protected-access
    )

    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp) / "repo"
      out_dir = Path(tmp) / "pilot_out"
      (root / "src" / "maxtext" / "configs" / "models").mkdir(parents=True)
      (root / "src" / "maxtext" / "models").mkdir(parents=True)
      (root / "src" / "maxtext" / "__init__.py").write_text("", encoding="utf-8")
      (root / "src" / "maxtext" / "models" / "__init__.py").write_text("", encoding="utf-8")
      (root / "src" / "maxtext" / "configs" / "types.py").write_text("class MaxTextConfig:\n  pass\n", encoding="utf-8")
      (root / "src" / "maxtext" / "configs" / "models" / "mini.yml").write_text(
          "decoder_block: mini_blk\n", encoding="utf-8"
      )
      (root / "src" / "maxtext" / "models" / "mini_blk.py").write_text(
          "def scale(x):\n  return x * 2\n", encoding="utf-8"
      )
      (root / "tests" / "unit").mkdir(parents=True)
      (root / "pytest.ini").write_text("[pytest]\npython_files = *_test.py\n", encoding="utf-8")
      (root / "tests" / "unit" / "mini_blk_test.py").write_text(
          "from maxtext.models import mini_blk\n\n" "def test_scale():\n" "  assert mini_blk.scale(3) == 6\n",
          encoding="utf-8",
      )
      _init_git_repo(root)
      base_sha = _git_commit_all(root, "Base commit")

      # Introduce a real regression on head
      (root / "src" / "maxtext" / "models" / "mini_blk.py").write_text(
          "def scale(x):\n  return x * 99\n", encoding="utf-8"
      )
      head_sha = _git_commit_all(root, "Break scale")

      # Invalid git ref fails fast with exit 2
      with mock.patch.object(
          sys,
          "argv",
          [
              "pilot.py",
              "--repo-root",
              str(root),
              "--label",
              "ut_bad_ref",
              "--head",
              head_sha,
              "--base",
              "nonexistent_ref_xyz",
              "--tiers",
              "",
              "--out-dir",
              str(out_dir / "bad_ref"),
          ],
      ):
        self.assertEqual(pilot.main(), 2)

      # Simulate subprocess calls to verify --python forwarding & non-zero exit
      seen_cmds: list[list[str]] = []
      real_sh = pilot.sh

      def _spy_sh(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(list(cmd))
        cmd_str = " ".join(cmd)
        if "mini_aot_fingerprint.py" in cmd_str and " batch " in f" {cmd_str} ":
          out_idx = cmd.index("--out") + 1
          Path(cmd[out_idx]).mkdir(parents=True, exist_ok=True)
          (Path(cmd[out_idx]) / "batch_summary.json").write_text(
              json.dumps({"total": 1, "ok": 0, "failed": ["mini"]}),
              encoding="utf-8",
          )
          return subprocess.CompletedProcess(cmd, 2, "batch ok=0\n", "")
        if "mini_aot_fingerprint.py" in cmd_str and " compare " in f" {cmd_str} ":
          return subprocess.CompletedProcess(cmd, 0, "# Tier B: 0 models differ\n", "")
        if "caller_impact_tests.py" in cmd_str and " run " in f" {cmd_str} ":
          return subprocess.CompletedProcess(cmd, 1, "regression\n", "")
        return real_sh(cmd, **kw)

      try:
        with (
            mock.patch.dict(os.environ, {"MAXTEXT_PILOT_PYTHON": "/nonexistent/env_python"}),
            mock.patch.object(pilot, "sh", side_effect=_spy_sh),
            mock.patch.object(pilot, "_check_python_deps", return_value=None, create=True),
            mock.patch.object(
                sys,
                "argv",
                [
                    "pilot.py",
                    "--repo-root",
                    str(root),
                    "--python",
                    sys.executable,
                    "--label",
                    "ut_pilot_fail",
                    "--head",
                    head_sha,
                    "--base",
                    base_sha,
                    "--tiers",
                    "BC",
                    "--out-dir",
                    str(out_dir / "bc_fail"),
                ],
            ),
        ):
          rc_pilot = pilot.main()
        self.assertEqual(rc_pilot, 1)
        summary = json.loads((out_dir / "bc_fail" / "summary.json").read_text(encoding="utf-8"))
        self.assertNotEqual(summary["tierB_rc"], 0)
        self.assertEqual(summary["tierC_rc"], 1)
        sub_cmds = [
            c
            for c in seen_cmds
            if any(s in " ".join(c) for s in ("mini_aot_fingerprint.py", "caller_impact_tests.py"))
            and ("batch" in c or "run" in c)
        ]
        self.assertTrue(sub_cmds)
        for c in sub_cmds:
          self.assertEqual(c[0], sys.executable)
          self.assertIn("--python", c)
          self.assertEqual(c[c.index("--python") + 1], sys.executable)
      finally:
        for sfx in ("base", "head", "main"):
          wt_tmp = f"/tmp/rt_ut_pilot_fail_{sfx}"
          if os.path.isdir(wt_tmp):
            real_sh(
                [
                    "git",
                    "-c",
                    "safe.directory=*",
                    "-C",
                    str(root),
                    "worktree",
                    "remove",
                    "--force",
                    wt_tmp,
                ]
            )

  def test_tier_missing_dep_preflight_and_crash_cache_invalidation(
      self,
  ) -> None:
    """Verifies missing jax/pytest fails closed (rc=2) and crash JSONs are not cached."""
    amd = _load_script("abstract_model_diff.py")
    cit = _load_script("caller_impact_tests.py")
    pilot = _load_script("pilot.py")

    with tempfile.TemporaryDirectory() as tmp:
      tdir = Path(tmp)
      # Create a minimal fake Python wrapper that exits 1 on `import jax` or `import pytest`
      bad_py = tdir / "no_deps_python.sh"
      bad_py.write_text(
          "#!/bin/sh\n" "echo 'ModuleNotFoundError: No module named jax' >&2\n" "exit 1\n",
          encoding="utf-8",
      )
      bad_py.chmod(0o755)

      # 1. pilot.py exits 2 immediately when --python lacks jax/pytest
      with mock.patch.object(
          sys,
          "argv",
          [
              "pilot.py",
              "--label",
              "ut_nodeps",
              "--head",
              "HEAD",
              "--base",
              "HEAD",
              "--python",
              str(bad_py),
              "--tiers",
              "A",
              "--out-dir",
              str(tdir / "p_out"),
          ],
      ):
        self.assertEqual(pilot.main(), 2)

      # 2. abstract_model_diff.py rejects poisoned crash JSONs and exits 2 when 0 models succeed
      cache_dir = tdir / "rtA_poisoned"
      cache_dir.mkdir()
      poisoned = cache_dir / "demo.json"
      poisoned.write_text(
          json.dumps({"model": "demo", "crash": "ModuleNotFoundError: No module named 'jax'"}),
          encoding="utf-8",
      )
      self.assertFalse(amd._is_valid_cached_dump(str(poisoned)))  # pylint: disable=protected-access
      with mock.patch.object(
          sys,
          "argv",
          [
              "abstract_model_diff.py",
              "dump",
              "--worktree",
              str(tdir),
              "--out",
              str(cache_dir),
              "--models",
              "demo",
              "--python",
              str(bad_py),
          ],
      ):
        self.assertEqual(amd.main(), 2)

      # 3. caller_impact_tests.py exits 2 when tests are selected but 0 test cases execute
      sel_file = tdir / "sel.json"
      sel_file.write_text(
          json.dumps(
              {
                  "tests": {
                      "tests/unit/broken_env_test.py": {
                          "distance": 0,
                          "symbol_hit": "",
                          "ci_ignored": False,
                          "changed_in_pr": True,
                      }
                  }
              }
          ),
          encoding="utf-8",
      )
      for sub in ("wt_b", "wt_h"):
        (tdir / sub / "tests" / "unit").mkdir(parents=True)
        (tdir / sub / "tests" / "unit" / "broken_env_test.py").write_text(
            "def test_x():\n  assert True\n", encoding="utf-8"
        )

      def _broken_startup(wt: str, rel: str, xml: str, **kw: Any) -> dict[str, Any]:
        del wt, rel, xml, kw
        return {
            "rc": 2,
            "seconds": 0,
            "failed": [],
            "counts": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
            "tail": "ImportError while loading conftest",
        }

      # 3. caller_impact_tests.py exits 2 on missing pytest preflight AND when 0 test cases execute
      with mock.patch.object(
          sys,
          "argv",
          [
              "caller_impact_tests.py",
              "run",
              "--sel",
              str(sel_file),
              "--base-wt",
              str(tdir / "wt_b"),
              "--head-wt",
              str(tdir / "wt_h"),
              "--out",
              str(tdir / "c_nodeps"),
              "--python",
              str(bad_py),
          ],
      ):
        self.assertEqual(cit.main(), 2)

      with (
          mock.patch.object(cit, "_pytest", side_effect=_broken_startup),
          mock.patch.object(
              sys,
              "argv",
              [
                  "caller_impact_tests.py",
                  "run",
                  "--sel",
                  str(sel_file),
                  "--base-wt",
                  str(tdir / "wt_b"),
                  "--head-wt",
                  str(tdir / "wt_h"),
                  "--out",
                  str(tdir / "c_broken"),
                  "--python",
                  sys.executable,
              ],
          ),
      ):
        self.assertEqual(cit.main(), 2)
      broken_rep = json.loads((tdir / "c_broken" / "result.json").read_text(encoding="utf-8"))
      self.assertTrue(broken_rep["collection_or_startup_failed"])

  def test_audit_repo_contracts_allows_zero_python_base_diff(self) -> None:
    """Verifies YAML/Markdown-only --base diffs exit 0 with files_audited == 0."""
    mod = _load_script("audit_repo_and_ci_contracts.py")
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _init_git_repo(root)
      (root / "pytest.ini").write_text("[pytest]\npython_files = *_test.py\n", encoding="utf-8")
      yml_file = root / "src" / "maxtext" / "configs" / "models" / "demo.yml"
      yml_file.parent.mkdir(parents=True)
      yml_file.write_text("base_emb_dim: 128\n", encoding="utf-8")
      base_sha = _git_commit_all(root, "Initial YAML")

      yml_file.write_text("base_emb_dim: 256\n", encoding="utf-8")
      _git_commit_all(root, "Update YAML only")

      buf = io.StringIO()
      with mock.patch("sys.stdout", buf):
        rc = mod.main(
            [
                "--repo-root",
                str(root),
                "--base",
                base_sha,
                "--run-torch-free-import",
            ]
        )
      self.assertEqual(rc, 0)
      rep = json.loads(buf.getvalue())
      self.assertEqual(rep["status"], "PASS")
      self.assertEqual(rep["files_audited"], 0)
      self.assertEqual(rep["issue_count"], 0)

      # Calling without --files, paths, or --base still exits 2
      with self.assertRaises(SystemExit) as ctx:
        mod.main(["--repo-root", str(root)])
      self.assertEqual(ctx.exception.code, 2)

  def test_static_auditors_use_merge_base_and_reject_invalid_base_ref(
      self,
  ) -> None:
    """Verifies behind-main branches use merge-base and bad --base refs exit 2."""
    hyg = _load_script("audit_pr_review_hygiene.py")
    repo_aud = _load_script("audit_repo_and_ci_contracts.py")
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _init_git_repo(root)
      subprocess.run(
          ["git", "-c", "safe.directory=*", "-C", str(root), "checkout", "-b", "main"],
          check=True,
          capture_output=True,
          text=True,
      )
      (root / "pytest.ini").write_text("[pytest]\npython_files = *_test.py\n", encoding="utf-8")
      ref_test = root / "tests" / "unit" / "demo_vs_reference_test.py"
      ref_test.parent.mkdir(parents=True)
      ref_test.write_text(
          "import unittest\n\n"
          "def reference_fn(x):\n  return x + 1\n\n"
          "class DemoRefTest(unittest.TestCase):\n"
          "  def test_ref(self):\n    self.assertEqual(reference_fn(1), 2)\n",
          encoding="utf-8",
      )
      layer_py = root / "src" / "maxtext" / "layers" / "demo_layer.py"
      layer_py.parent.mkdir(parents=True)
      layer_py.write_text("def run_layer(x):\n  return x\n", encoding="utf-8")
      _git_commit_all(root, "M0 baseline")

      # Create feature branch at M0 touching only demo_layer.py
      subprocess.run(
          ["git", "-c", "safe.directory=*", "-C", str(root), "checkout", "-b", "feat"],
          check=True,
          capture_output=True,
          text=True,
      )
      layer_py.write_text("def run_layer(x):\n  return x * 2\n", encoding="utf-8")
      _git_commit_all(root, "Feature commit")

      # Advance main to M1 editing reference_fn (would trigger REFERENCE_ORACLE_EDITED under 2-dot diff)
      subprocess.run(
          ["git", "-c", "safe.directory=*", "-C", str(root), "checkout", "main"],
          check=True,
          capture_output=True,
          text=True,
      )
      ref_test.write_text(
          "import unittest\n\n"
          "def reference_fn(x):\n  return x + 999\n\n"
          "class DemoRefTest(unittest.TestCase):\n"
          "  def test_ref(self):\n    self.assertEqual(reference_fn(1), 1000)\n",
          encoding="utf-8",
      )
      _git_commit_all(root, "M1 upstream reference update")

      # Switch back to feat (behind main) and audit against --base main
      subprocess.run(
          ["git", "-c", "safe.directory=*", "-C", str(root), "checkout", "feat"],
          check=True,
          capture_output=True,
          text=True,
      )
      self.assertEqual(hyg.main(["--repo-root", str(root), "--base", "main"]), 0)
      self.assertEqual(repo_aud.main(["--repo-root", str(root), "--base", "main"]), 0)

      # Invalid --base ref must fail closed with exit code 2 in both scripts
      self.assertEqual(
          hyg.main(["--repo-root", str(root), "--base", "nonexistent_branch_xyz"]),
          2,
      )
      self.assertEqual(
          repo_aud.main(["--repo-root", str(root), "--base", "nonexistent_branch_xyz"]),
          2,
      )


if __name__ == "__main__":
  unittest.main()
