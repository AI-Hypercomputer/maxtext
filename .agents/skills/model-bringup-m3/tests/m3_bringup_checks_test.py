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

"""Unit tests for `.agents/skills/model-bringup-m3/` scripts and templates."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest

import jax.numpy as jnp
import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parents[2]
AUDIT_PATH = SKILL_ROOT / "scripts/audit_m3_contracts.py"
TEMPLATES_DIR = SKILL_ROOT / "templates"


def _load_module_from_path(name: str, path: pathlib.Path) -> Any:
  """Loads a Python module dynamically from a filesystem path."""
  if name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


audit_m3_contracts = _load_module_from_path("_test_audit_m3_contracts", AUDIT_PATH)


def _init_git_repo(repo_dir: str) -> bool:
  """Initializes a temporary git repository for diff-scoping tests."""
  git_bin = shutil.which("git")
  if not git_bin:
    return False
  env = dict(os.environ)
  env["GIT_AUTHOR_NAME"] = "Test"
  env["GIT_AUTHOR_EMAIL"] = "test@example.com"
  env["GIT_COMMITTER_NAME"] = "Test"
  env["GIT_COMMITTER_EMAIL"] = "test@example.com"
  subprocess.run([git_bin, "init", "-q", repo_dir], env=env, check=True)
  return True


def _run_git(repo_dir: str, *args: str) -> None:
  """Runs a git command inside `repo_dir` with deterministic author metadata."""
  git_bin = shutil.which("git")
  assert git_bin is not None
  env = dict(os.environ)
  env["GIT_AUTHOR_NAME"] = "Test"
  env["GIT_AUTHOR_EMAIL"] = "test@example.com"
  env["GIT_COMMITTER_NAME"] = "Test"
  env["GIT_COMMITTER_EMAIL"] = "test@example.com"
  subprocess.run([git_bin, "-C", repo_dir, *args], env=env, check=True)


def _git_commit_all(repo_dir: str, msg: str) -> None:
  """Stages and commits all files in a temporary git repository."""
  _run_git(repo_dir, "add", "-A")
  _run_git(repo_dir, "commit", "-q", "-m", msg)


class AuditM3ContractsTest(unittest.TestCase):
  """Tests all structural, registry, BUILD, and diff-scoping checks in `audit_m3_contracts.py`."""

  def test_dual_driver_wiring_and_registry_completeness(self) -> None:
    with tempfile.TemporaryDirectory() as root:
      loop_dir = pathlib.Path(root) / "src/maxtext/m3/train"
      mcu_dir = pathlib.Path(root) / "src/maxtext/utils"
      models_dir = pathlib.Path(root) / "src/maxtext/m3/models"
      loop_dir.mkdir(parents=True)
      mcu_dir.mkdir(parents=True)
      models_dir.mkdir(parents=True)

      (loop_dir / "loop.py").write_text(
          "from maxtext.m3.infra import scan\n" "from maxtext.m3.infra import offload\n",
          encoding="utf-8",
      )
      (mcu_dir / "model_creation_utils.py").write_text(
          "from maxtext.m3.infra import scan\n",
          encoding="utf-8",
      )
      (models_dir / "__init__.py").write_text(
          'MODEL_REGISTRY = {"alpha": object(), "beta": object()}\n' 'MODEL_FLOPS_REGISTRY = {"alpha": object()}\n',
          encoding="utf-8",
      )

      rep = audit_m3_contracts.run_audit(root, [], check_reg=True)
      self.assertEqual(rep["status"], "FAIL")
      codes = {item["code"] for item in rep["issues"]}
      self.assertEqual(
          codes,
          {"INFRA_NOT_IN_MODEL_CREATION_UTILS", "MISSING_FLOPS_REGISTRY_ENTRY"},
      )

      (mcu_dir / "model_creation_utils.py").write_text(
          "from maxtext.m3.infra import scan, offload\n",
          encoding="utf-8",
      )
      (models_dir / "__init__.py").write_text(
          'MODEL_REGISTRY = {"alpha": object(), "beta": object()}\n'
          'MODEL_FLOPS_REGISTRY = {"alpha": object(), "beta": object()}\n',
          encoding="utf-8",
      )
      rep_clean = audit_m3_contracts.run_audit(root, [], check_reg=True)
      self.assertEqual(rep_clean["status"], "PASS", rep_clean)

  def test_m3_python_invariants_caught_and_clean_module_passes(self) -> None:
    with tempfile.TemporaryDirectory() as root:
      infra_dir = pathlib.Path(root) / "src/maxtext/m3/infra"
      family_dir = pathlib.Path(root) / "src/maxtext/m3/models/synth"
      infra_dir.mkdir(parents=True)
      family_dir.mkdir(parents=True)

      bad_infra = infra_dir / "bad_transform.py"
      bad_infra.write_text(
          "from flax import linen as nn\n\n"
          "def apply_transform(mod, name):\n"
          '  if name == "qwen3-8b":\n'
          '    mod.sow("intermediates", "loss", 1.0)\n'
          '  meta = {"out_sharding": ("fsdp",)}\n'
          "  return nn, meta\n",
          encoding="utf-8",
      )
      bespoke_ckpt = family_dir / "synth_ckpt.py"
      bespoke_ckpt.write_text("X = 1\n", encoding="utf-8")

      rep = audit_m3_contracts.run_audit(root, [str(bad_infra), str(bespoke_ckpt)])
      self.assertEqual(rep["status"], "FAIL")
      codes = {item["code"] for item in rep["issues"]}
      self.assertEqual(
          codes,
          {
              "FORBIDDEN_LINEN_OR_LEGACY_DECODER_IMPORT",
              "MODEL_NAME_BRANCH_IN_SHARED_M3",
              "ENGINE_UNSAFE_SOW_IN_M3",
              "OUT_SHARDING_WITHOUT_EAGER_SHARDING",
              "BESPOKE_PER_MODEL_BOILERPLATE_FILE",
          },
      )

      clean_mod = family_dir / "modeling_synth.py"
      clean_mod.write_text(
          "from flax import nnx\n\n"
          "def make_meta(spec):\n"
          '  return {"out_sharding": spec, "eager_sharding": True, "mod": nnx}\n',
          encoding="utf-8",
      )
      rep_clean = audit_m3_contracts.run_audit(root, [str(clean_mod)])
      self.assertEqual(rep_clean["status"], "PASS", rep_clean)

  def test_check_blaze_build_default_false_and_opt_in(self) -> None:
    with tempfile.TemporaryDirectory() as root:
      tests_dir = pathlib.Path(root) / "tests"
      tests_dir.mkdir(parents=True)
      build_file = tests_dir / "BUILD"
      build_file.write_text(
          'py_test(name = "bad", srcs = ["unit/m3/foo_test.py"])\n',
          encoding="utf-8",
      )

      # Default check_blaze_build=False skips tests/BUILD on GitHub checkouts.
      self.assertEqual(
          audit_m3_contracts.main(["--maxtext-root", root, str(build_file)]),
          0,
      )
      # Opt-in --check-blaze-build flags tests/BUILD violation when present.
      self.assertEqual(
          audit_m3_contracts.main(
              [
                  "--maxtext-root",
                  root,
                  "--check-blaze-build",
                  str(build_file),
              ]
          ),
          1,
      )

  def test_git_base_diff_scoping_ignores_preexisting_debt(self) -> None:
    with tempfile.TemporaryDirectory() as root:
      if not _init_git_repo(root):
        return
      infra_dir = pathlib.Path(root) / "src/maxtext/m3/infra"
      infra_dir.mkdir(parents=True)
      mod_file = infra_dir / "legacy.py"
      mod_file.write_text(
          "def old_fn(mod):\n" '  mod.sow("intermediates", "x", 0.0)\n\n' "def clean_fn(x):\n" "  return x\n",
          encoding="utf-8",
      )
      _git_commit_all(root, "Initial")
      mod_file.write_text(
          "def old_fn(mod):\n" '  mod.sow("intermediates", "x", 0.0)\n\n' "def clean_fn(x):\n" "  return x + 1\n",
          encoding="utf-8",
      )
      self.assertEqual(audit_m3_contracts.main(["--maxtext-root", root, str(mod_file)]), 1)
      self.assertEqual(
          audit_m3_contracts.main(["--maxtext-root", root, "--base", "HEAD", str(mod_file)]),
          0,
      )

  def test_git_base_invalid_ref_fails_closed_and_uses_merge_base(self) -> None:
    with tempfile.TemporaryDirectory() as root:
      if not _init_git_repo(root):
        return
      _run_git(root, "checkout", "-B", "main")
      infra_dir = pathlib.Path(root) / "src/maxtext/m3/infra"
      family_dir = pathlib.Path(root) / "src/maxtext/m3/models/synth"
      infra_dir.mkdir(parents=True)
      family_dir.mkdir(parents=True)

      shared_py = infra_dir / "shared.py"
      shared_py.write_text(
          "def old_fn(mod):\n  mod.sow('intermediates', 'x', 0.0)\n",
          encoding="utf-8",
      )
      _git_commit_all(root, "Base commit on main")

      _run_git(root, "checkout", "-b", "feature")
      clean_py = family_dir / "modeling_clean.py"
      clean_py.write_text("from flax import nnx\n\nX = nnx\n", encoding="utf-8")
      _git_commit_all(root, "Add clean model on feature")

      _run_git(root, "checkout", "main")
      shared_py.write_text("def old_fn(mod):\n  return mod\n", encoding="utf-8")
      _git_commit_all(root, "Clean shared.py independently on main")
      _run_git(root, "checkout", "feature")

      # 3-dot merge-base(main, HEAD) must see 0 changed lines on shared.py.
      self.assertEqual(
          audit_m3_contracts.main(["--maxtext-root", root, "--base", "main", str(shared_py)]),
          0,
      )
      # Invalid --base ref must fail closed with exit code 2 (never silent PASS).
      self.assertEqual(
          audit_m3_contracts.main(["--maxtext-root", root, "--base", "nonexistent_ref_xyz", str(clean_py)]),
          2,
      )


class M3TemplatesValidityTest(unittest.TestCase):
  """Tests AST/JSON validity, torch-free imports, and fail-closed template behavior."""

  def test_all_templates_parse_and_import_torch_free(self) -> None:
    expected = {
        "conformance_spec.py",
        "design_handoff.md",
        "golden_manifest.json",
        "mini_twin.md",
        "parity_test.py",
        "provenance_test.py",
    }
    actual = {p.name for p in TEMPLATES_DIR.iterdir() if p.is_file()}
    self.assertEqual(actual, expected)

    for py_name in ("conformance_spec.py", "parity_test.py", "provenance_test.py"):
      src = (TEMPLATES_DIR / py_name).read_text(encoding="utf-8")
      self.assertIn("Licensed under the Apache License, Version 2.0", src)
      ast.parse(src, filename=py_name)

    manifest = json.loads((TEMPLATES_DIR / "golden_manifest.json").read_text(encoding="utf-8"))
    self.assertIn("reference", manifest)
    self.assertIn("mini_twin", manifest)
    self.assertIn("goldens", manifest)

    probe = """
import importlib.util
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
templates_dir = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
sys.modules["torch"] = None

from tests.unit.m3.conformance import spec as live_spec
before_count = len(live_spec._REGISTERED_SPECS)

for name in ("conformance_spec.py", "parity_test.py", "provenance_test.py"):
  p = templates_dir / name
  s = importlib.util.spec_from_file_location(f"_tmpl_{p.stem}", p)
  mod = importlib.util.module_from_spec(s)
  sys.modules[s.name] = mod
  s.loader.exec_module(mod)

assert len(live_spec._REGISTERED_SPECS) == before_count
"""
    env = dict(os.environ)
    res = subprocess.run(
        [sys.executable, "-", str(REPO_ROOT), str(TEMPLATES_DIR)],
        input=probe,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    self.assertEqual(res.returncode, 0, res.stderr)

  def test_conformance_spec_template_matches_live_spec_and_fails_closed(
      self,
  ) -> None:
    if str(REPO_ROOT) not in sys.path:
      sys.path.insert(0, str(REPO_ROOT))
    from tests.unit.m3.conformance import spec as live_spec  # pylint: disable=import-outside-toplevel

    tmpl = _load_module_from_path("_tmpl_conformance_spec", TEMPLATES_DIR / "conformance_spec.py")
    self.assertIs(tmpl.ConformanceSpec, live_spec.ConformanceSpec)
    self.assertIs(tmpl.ModelConformanceSpec, live_spec.ConformanceSpec)
    self.assertIs(
        tmpl.CheckId.PACKING_AND_PADDING_ISOLATION,
        live_spec.CheckId.PACKING_AND_PADDING_ISOLATION,
    )
    with self.assertRaises(NotImplementedError):
      tmpl.build_spec()

  def test_provenance_and_parity_templates_fail_closed_and_run_when_filled(
      self,
  ) -> None:
    prov = _load_module_from_path("_tmpl_provenance_test", TEMPLATES_DIR / "provenance_test.py")
    case = prov.ProvenanceTest("test_some_reference_is_pinned")
    with self.assertRaises(NotImplementedError):
      case.test_some_reference_is_pinned()

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = pathlib.Path(tmp)
      oracle_file = tmp_path / "oracle_toy.py"
      oracle_file.write_text("X = 42\n", encoding="utf-8")
      oracle_sha = hashlib.sha256(oracle_file.read_bytes()).hexdigest()

      golden_npz = tmp_path / "forward.npz"
      np.savez(str(golden_npz), out=np.ones((2, 2), dtype=np.float32))
      golden_sha = hashlib.sha256(golden_npz.read_bytes()).hexdigest()

      manifest_data = {
          "reference": {
              "description": "toy",
              "pinned_modules": {},
              "pinned_oracle_files": {"oracle_toy.py": oracle_sha},
              "deviations": [],
          },
          "mini_twin": {"manifest": "m.json", "command": "slice"},
          "goldens": [
              {
                  "local_path": "forward.npz",
                  "sha256": golden_sha,
                  "storage_uri": "local/forward.npz",
                  "command": "capture",
                  "seed": 0,
              }
          ],
      }
      manifest_file = tmp_path / "golden_manifest.json"
      manifest_file.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
      manifest_sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

      orig_vals = (
          prov._HERE,  # pylint: disable=protected-access
          prov.PINNED_REFERENCE_MODULES,
          prov.PINNED_ORACLE_FILES,
          prov.GOLDEN_MANIFEST,
          prov.GOLDEN_MANIFEST_SHA256,
      )
      try:
        prov._HERE = tmp_path  # pylint: disable=protected-access
        prov.PINNED_REFERENCE_MODULES = {}
        prov.PINNED_ORACLE_FILES = {"oracle_toy.py": oracle_sha}
        prov.GOLDEN_MANIFEST = "golden_manifest.json"
        prov.GOLDEN_MANIFEST_SHA256 = manifest_sha
        filled_case = prov.ProvenanceTest("test_some_reference_is_pinned")
        filled_case.test_some_reference_is_pinned()
        filled_case.test_in_test_oracle_files_unchanged()
        filled_case.test_golden_manifest_unchanged()
        filled_case.test_every_golden_matches_manifest()
        filled_case.test_manifest_reference_matches_pins()
      finally:
        (
            prov._HERE,  # pylint: disable=protected-access
            prov.PINNED_REFERENCE_MODULES,
            prov.PINNED_ORACLE_FILES,
            prov.GOLDEN_MANIFEST,
            prov.GOLDEN_MANIFEST_SHA256,
        ) = orig_vals

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch reference")
  def test_filled_parity_template_passes_with_torch(self) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    parity = _load_module_from_path("_tmpl_parity_test", TEMPLATES_DIR / "parity_test.py")
    tol = parity.layer_parity.Tolerance(atol=1e-6, rtol=1e-6, frob_rel=1e-6)
    fp64_tol = parity.layer_parity.Tolerance(atol=1e-11, rtol=1e-11, frob_rel=1e-11)
    orig = (
        parity.FWD_TOL,
        parity.GRAD_TOL,
        parity.FP64_TOL,
        parity.WEIGHT_MAP,
        parity.NEGCTL_LEAF,
        parity.make_reference,
        parity.make_candidate,
    )
    try:
      parity.FWD_TOL = tol
      parity.GRAD_TOL = tol
      parity.FP64_TOL = fp64_tol
      parity.WEIGHT_MAP = {"w": "w:T"}
      parity.NEGCTL_LEAF = "w"

      def _make_ref(dtype: Any) -> tuple[Any, dict[str, Any], Any]:
        w = torch.tensor([[1.5, -0.5, 0.25], [0.5, 2.0, -1.0]], dtype=dtype)
        x = torch.tensor([[1.0, 2.0, -0.5]], dtype=dtype)
        return (lambda p, inp: inp @ p["w"].T), {"w": w}, x

      def _make_cand(ref_params: dict[str, Any], ref_input: Any, dtype: jnp.dtype) -> tuple[Any, Any, Any]:
        w_j = jnp.asarray(ref_params["w"].detach().numpy().T, dtype=dtype)
        x_j = jnp.asarray(ref_input.detach().numpy(), dtype=dtype)
        return (lambda p, inp: inp @ p["w"]), {"w": w_j}, x_j

      parity.make_reference = _make_ref
      parity.make_candidate = _make_cand

      tc = parity.ComponentParityTest("test_fp32_forward_and_every_gradient_leaf_match_reference")
      tc.test_fp32_forward_and_every_gradient_leaf_match_reference()
      tc.test_negative_control_one_perturbed_leaf_breaks_parity()
      tc.test_fp64_frobenius_matches_reference()
    finally:
      (
          parity.FWD_TOL,
          parity.GRAD_TOL,
          parity.FP64_TOL,
          parity.WEIGHT_MAP,
          parity.NEGCTL_LEAF,
          parity.make_reference,
          parity.make_candidate,
      ) = orig


if __name__ == "__main__":
  unittest.main()
