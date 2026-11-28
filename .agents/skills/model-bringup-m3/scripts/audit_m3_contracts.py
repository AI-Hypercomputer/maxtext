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

"""Deterministic AST and wiring auditor for MaxText m3 modules and PRs.

Checks touching src/maxtext/m3/ and driver wiring for recurring m3-specific
defects that unit tests on m3/train/loop.py alone miss:
  1. INFRA_NOT_IN_MODEL_CREATION_UTILS: m3/infra transform wired into
     m3/train/loop.py but omitted from utils/model_creation_utils.py.
  2. MISSING_FLOPS_REGISTRY_ENTRY: model in MODEL_REGISTRY missing from
     MODEL_FLOPS_REGISTRY in m3/models/__init__.py (--check-registries).
  3. FORBIDDEN_LINEN_OR_LEGACY_DECODER_IMPORT: flax.linen or
     maxtext.layers.decoders imported inside src/maxtext/m3/.
  4. MODEL_NAME_BRANCH_IN_SHARED_M3: model-family branching inside shared
     src/maxtext/m3/{core,infra,configs,train}/ files.
  5. ENGINE_UNSAFE_SOW_IN_M3: .sow(...) calls inside src/maxtext/m3/ that
     mutate the NNX state treedef and break MaxTextTrainingEngine.
  6. OUT_SHARDING_WITHOUT_EAGER_SHARDING: raw {"out_sharding": ...} dict literal
     in src/maxtext/m3/ missing eager_sharding=True.
  7. OSS_M3_TEST_IN_BLAZE_TESTS_BUILD: open-source tests/unit/m3/*_test.py wired
     into tests/BUILD against repository policy (opt-in via --check-blaze-build).
  8. BESPOKE_PER_MODEL_BOILERPLATE_FILE: per-model *_ckpt.py, *_flops.py,
     *_train.py, or convert_*_ckpt.py added under src/maxtext/m3/models/ instead
     of wiring centralized PARAM_MAPPING / HOOK_FNS / HF_SHAPE,
     MODEL_FLOPS_REGISTRY, and m3/train/loop.py.
"""

import argparse
import ast
from collections.abc import Sequence
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

_MODEL_FAMILY_TOKENS = (
    "deepseek",
    "gemma",
    "glm",
    "granite",
    "kimi",
    "llama",
    "mimo",
    "mistral",
    "mixtral",
    "muse",
    "nemotron",
    "qwen",
    "wan",
)
_SHARED_M3_SUBDIRS = ("core", "infra", "configs", "train")
_MODEL_BRANCH_RE = re.compile(
    r"(?<![a-z0-9_])(" + "|".join(_MODEL_FAMILY_TOKENS) + r")(?![a-z])",
    re.IGNORECASE,
)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _rel_to_maxtext(path: str, maxtext_root: str) -> str:
  abs_p = os.path.abspath(path)
  abs_r = os.path.abspath(maxtext_root)
  if abs_p.startswith(abs_r + os.sep):
    return os.path.relpath(abs_p, abs_r).replace(os.sep, "/")
  return path.replace(os.sep, "/")


def _resolve_git_merge_base(maxtext_root: str, base_ref: str) -> tuple[str, str]:
  """Resolves `(git_toplevel, merge_base_sha)` for `--base`, failing closed on invalid refs."""
  git_bin = shutil.which("git")
  if not git_bin:
    raise ValueError("git binary not found in PATH for --base diff scoping.")
  abs_root = os.path.abspath(maxtext_root)
  top_proc = subprocess.run(
      [git_bin, "-c", "safe.directory=*", "-C", abs_root, "rev-parse", "--show-toplevel"],
      capture_output=True,
      text=True,
      check=False,
  )
  if top_proc.returncode != 0 or not top_proc.stdout.strip():
    err = top_proc.stderr.strip() or f"{abs_root!r} is not a git repository"
    raise ValueError(f"Failed to resolve git repository root for --base: {err}")
  toplevel = os.path.abspath(top_proc.stdout.strip())

  mb_proc = subprocess.run(
      [git_bin, "-c", "safe.directory=*", "-C", toplevel, "merge-base", base_ref, "HEAD"],
      capture_output=True,
      text=True,
      check=False,
  )
  if mb_proc.returncode != 0 or not mb_proc.stdout.strip():
    err = mb_proc.stderr.strip() or f"invalid git base ref {base_ref!r}"
    raise ValueError(f"Failed to resolve git merge-base for --base={base_ref!r}: {err}")
  return toplevel, mb_proc.stdout.strip()


def _changed_lines_by_file(
    maxtext_root: str,
    git_toplevel: str,
    merge_base: str,
    files: Sequence[str],
) -> dict[str, set[int]]:
  """Returns added/modified line numbers per maxtext-root-relative path vs `merge_base`."""
  git_bin = shutil.which("git")
  assert git_bin is not None
  existing = [os.path.abspath(f) for f in files if os.path.isfile(f)]
  if not existing:
    return {}
  top_rel = [os.path.relpath(f, git_toplevel) for f in existing]
  proc = subprocess.run(
      [
          git_bin,
          "-c",
          "safe.directory=*",
          "-C",
          git_toplevel,
          "diff",
          "-U0",
          "--no-color",
          merge_base,
          "--",
          *top_rel,
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    raise ValueError(f"git diff against merge-base {merge_base} failed: {proc.stderr.strip()}")
  changed: dict[str, set[int]] = {}
  cur_file = ""
  for line in proc.stdout.splitlines():
    if line.startswith("+++ b/"):
      abs_file = os.path.abspath(os.path.join(git_toplevel, line[6:].strip()))
      cur_file = _rel_to_maxtext(abs_file, maxtext_root)
      changed.setdefault(cur_file, set())
    elif line.startswith("@@ ") and cur_file:
      m = _HUNK_RE.match(line)
      if m:
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        if count > 0:
          changed[cur_file].update(range(start, start + count))
  return changed


def _git_show_file(maxtext_root: str, git_toplevel: str, git_ref: str, rel_path: str) -> str | None:
  """Reads `<git_ref>:<path>` via git, returning None if absent at `git_ref`."""
  git_bin = shutil.which("git")
  if not git_ref or not git_bin:
    return None
  abs_target = os.path.abspath(os.path.join(maxtext_root, rel_path))
  top_rel = os.path.relpath(abs_target, git_toplevel).replace(os.sep, "/")
  proc = subprocess.run(
      [
          git_bin,
          "-c",
          "safe.directory=*",
          "-C",
          git_toplevel,
          "show",
          f"{git_ref}:{top_rel}",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  return proc.stdout if proc.returncode == 0 else None


def _extract_dict_keys(tree: ast.Module, target_name: str) -> tuple[set[str], int]:
  """Extracts string literal keys assigned to a top-level dict `target_name`."""
  keys: set[str] = set()
  lineno = 1
  for node in tree.body:
    val: ast.expr | None = None
    if isinstance(node, ast.Assign):
      for tgt in node.targets:
        if isinstance(tgt, ast.Name) and tgt.id == target_name:
          val = node.value
          lineno = node.lineno
    elif isinstance(node, ast.AnnAssign):
      if isinstance(node.target, ast.Name) and node.target.id == target_name:
        val = node.value
        lineno = node.lineno
    if isinstance(val, ast.Dict):
      for k in val.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
          keys.add(k.value)
  return keys, lineno


def _extract_loop_infra_mods(loop_src: str, filename: str) -> dict[str, int]:
  """Extracts m3/infra module names imported by loop.py -> first line number."""
  tree = ast.parse(loop_src, filename=filename)
  infra_mods: dict[str, int] = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module:
      if node.module.endswith("m3.infra"):
        for alias in node.names:
          if alias.name != "__init__":
            infra_mods.setdefault(alias.name, node.lineno)
      elif ".m3.infra." in node.module or node.module.startswith("maxtext.m3.infra."):
        mod_name = node.module.rsplit(".", 1)[-1]
        infra_mods.setdefault(mod_name, node.lineno)
  return infra_mods


def _mcu_wires_infra_mod(mcu_src: str, mod_name: str) -> bool:
  pattern = re.compile(
      rf"\bm3\.infra\.{re.escape(mod_name)}\b|\bfrom\s+\S*m3\.infra\s+import\b.*\b{re.escape(mod_name)}\b"
  )
  return bool(pattern.search(mcu_src))


def check_dual_driver_wiring(
    maxtext_root: str,
    base_ref: str = "",
    *,
    git_toplevel: str = "",
) -> list[dict[str, Any]]:
  """Checks that m3/infra modules used in loop.py are also wired into model_creation_utils.py."""
  issues: list[dict[str, Any]] = []
  loop_rel = "src/maxtext/m3/train/loop.py"
  mcu_rel = "src/maxtext/utils/model_creation_utils.py"
  loop_path = os.path.join(maxtext_root, loop_rel)
  mcu_path = os.path.join(maxtext_root, mcu_rel)
  if not (os.path.isfile(loop_path) and os.path.isfile(mcu_path)):
    return issues

  with open(loop_path, "r", encoding="utf-8") as f:
    loop_src = f.read()
  with open(mcu_path, "r", encoding="utf-8") as f:
    mcu_src = f.read()

  infra_mods = _extract_loop_infra_mods(loop_src, loop_path)

  # Ignore pre-existing infra gaps already present at base_ref so unrelated
  # PRs do not inherit historical wiring debt.
  base_missing: set[str] = set()
  if base_ref:
    top_dir = git_toplevel or os.path.abspath(maxtext_root)
    b_loop = _git_show_file(maxtext_root, top_dir, base_ref, loop_rel)
    b_mcu = _git_show_file(maxtext_root, top_dir, base_ref, mcu_rel)
    if b_loop is not None and b_mcu is not None:
      try:
        b_mods = _extract_loop_infra_mods(b_loop, loop_rel)
        base_missing = {m for m in b_mods if not _mcu_wires_infra_mod(b_mcu, m)}
      except SyntaxError:
        pass

  for mod_name, lineno in sorted(infra_mods.items()):
    if mod_name in base_missing:
      continue
    if not _mcu_wires_infra_mod(mcu_src, mod_name):
      issues.append(
          {
              "code": "INFRA_NOT_IN_MODEL_CREATION_UTILS",
              "severity": "HIGH",
              "file": _rel_to_maxtext(loop_path, maxtext_root),
              "line": lineno,
              "message": (
                  f"m3/infra/{mod_name}.py is wired into m3/train/loop.py but"
                  " missing from src/maxtext/utils/model_creation_utils.py"
                  " (skipped in MaxTextTrainingEngine, train.py, and"
                  " train_compile.py)."
              ),
          }
      )
  return issues


def check_registries(
    maxtext_root: str,
    base_ref: str = "",
    *,
    git_toplevel: str = "",
) -> list[dict[str, Any]]:
  """Checks MODEL_REGISTRY entries against MODEL_FLOPS_REGISTRY in m3/models/__init__.py."""
  issues: list[dict[str, Any]] = []
  reg_rel = "src/maxtext/m3/models/__init__.py"
  reg_path = os.path.join(maxtext_root, reg_rel)
  if not os.path.isfile(reg_path):
    return issues
  with open(reg_path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=reg_path)
  models, _ = _extract_dict_keys(tree, "MODEL_REGISTRY")
  flops, flops_line = _extract_dict_keys(tree, "MODEL_FLOPS_REGISTRY")

  base_missing: set[str] = set()
  if base_ref:
    top_dir = git_toplevel or os.path.abspath(maxtext_root)
    b_src = _git_show_file(maxtext_root, top_dir, base_ref, reg_rel)
    if b_src is not None:
      try:
        base_tree = ast.parse(b_src)
        b_models, _ = _extract_dict_keys(base_tree, "MODEL_REGISTRY")
        b_flops, _ = _extract_dict_keys(base_tree, "MODEL_FLOPS_REGISTRY")
        base_missing = b_models - b_flops
      except SyntaxError:
        pass

  for m_name in sorted((models - flops) - base_missing):
    issues.append(
        {
            "code": "MISSING_FLOPS_REGISTRY_ENTRY",
            "severity": "MED",
            "file": _rel_to_maxtext(reg_path, maxtext_root),
            "line": flops_line,
            "message": (
                f"Model {m_name!r} is in MODEL_REGISTRY but missing from"
                " MODEL_FLOPS_REGISTRY; train.py and train_compile.py will raise"
                " ValueError during TFLOPs calculation."
            ),
        }
    )
  return issues


def scan_m3_python_file(
    path: str,
    maxtext_root: str,
    *,
    changed_lines: set[int] | None = None,
    existed_at_base: bool = False,
) -> list[dict[str, Any]]:
  """Scans a Python file under src/maxtext/m3/ for structural invariant violations."""
  issues: list[dict[str, Any]] = []
  rel = _rel_to_maxtext(path, maxtext_root)
  if "src/maxtext/m3/" not in rel:
    return issues
  after_m3 = rel.split("src/maxtext/m3/", 1)[1]
  top_sub = after_m3.split("/", 1)[0]
  in_shared_m3 = top_sub in _SHARED_M3_SUBDIRS

  with open(path, "r", encoding="utf-8") as f:
    src = f.read()
  tree = ast.parse(src, filename=path)

  def _in_diff(lineno: int) -> bool:
    return changed_lines is None or lineno in changed_lines

  bname = os.path.basename(rel)
  if (
      top_sub == "models"
      and (
          bname.endswith(("_ckpt.py", "_flops.py", "_train.py"))
          or (bname.startswith("convert_") and bname.endswith("_ckpt.py"))
      )
      and "allow-bespoke-m3-infra:" not in src
      and not existed_at_base
  ):
    issues.append(
        {
            "code": "BESPOKE_PER_MODEL_BOILERPLATE_FILE",
            "severity": "HIGH",
            "file": rel,
            "line": 1,
            "message": (
                f"Bespoke per-model file {bname!r} duplicates centralized MaxText/"
                "m3 infrastructure; wire standard safetensors models into"
                " PARAM_MAPPING / HOOK_FNS / HF_SHAPE (`to_maxtext` /"
                " `to_huggingface`), `MODEL_FLOPS_REGISTRY`, and `m3/train/loop.py`"
                " instead (or annotate `# allow-bespoke-m3-infra: <reason>`)."
            ),
        }
    )

  docstring_nodes: set[int] = set()
  for parent in ast.walk(tree):
    if isinstance(
        parent,
        (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
    ):
      if parent.body and isinstance(parent.body[0], ast.Expr) and isinstance(parent.body[0].value, ast.Constant):
        docstring_nodes.add(id(parent.body[0].value))

  for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)) and _in_diff(node.lineno):
      mods: list[str] = []
      if isinstance(node, ast.Import):
        mods = [a.name for a in node.names]
      elif node.module:
        mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
      for mod in mods:
        if mod == "flax.linen" or mod.startswith("flax.linen.") or "layers.decoders" in mod:
          issues.append(
              {
                  "code": "FORBIDDEN_LINEN_OR_LEGACY_DECODER_IMPORT",
                  "severity": "HIGH",
                  "file": rel,
                  "line": node.lineno,
                  "message": (
                      f"Forbidden legacy/Linen import {mod!r} inside m3 module; use" " pure flax.nnx primitives."
                  ),
              }
          )

    if isinstance(node, ast.Call) and _in_diff(node.lineno):
      if isinstance(node.func, ast.Attribute) and node.func.attr == "sow":
        issues.append(
            {
                "code": "ENGINE_UNSAFE_SOW_IN_M3",
                "severity": "HIGH",
                "file": rel,
                "line": node.lineno,
                "message": (
                    ".sow(...) mutates the NNX state treedef during forward"
                    " execution, violating MaxTextTrainingEngine stability; return"
                    " structured outputs / carry aux_loss instead."
                ),
            }
        )

    if isinstance(node, ast.Dict) and top_sub != "core" and _in_diff(node.lineno):
      key_strs = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
      if "out_sharding" in key_strs and "eager_sharding" not in key_strs:
        issues.append(
            {
                "code": "OUT_SHARDING_WITHOUT_EAGER_SHARDING",
                "severity": "HIGH",
                "file": rel,
                "line": node.lineno,
                "message": (
                    "Raw {'out_sharding': ...} dict omits 'eager_sharding': True;"
                    " importing legacy layers/nnx_wrappers.py disables Flax eager"
                    " sharding globally and silently replicates weights. Use"
                    " m3/core/sharding.py param_metadata."
                ),
            }
        )

    if (
        in_shared_m3
        and isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
        and _in_diff(node.lineno)
    ):
      if len(node.value) <= 64 and "\n" not in node.value:
        m = _MODEL_BRANCH_RE.search(node.value)
        if m:
          issues.append(
              {
                  "code": "MODEL_NAME_BRANCH_IN_SHARED_M3",
                  "severity": "HIGH",
                  "file": rel,
                  "line": node.lineno,
                  "message": (
                      f"Shared m3 module ({top_sub}/) contains model-family string"
                      f" {node.value!r} ({m.group(1)}); route via MODEL_REGISTRY or"
                      " typed config fields instead."
                  ),
              }
          )

  return issues


def check_tests_build_file(
    path: str,
    maxtext_root: str,
    *,
    changed_lines: set[int] | None = None,
) -> list[dict[str, Any]]:
  """Checks optional tests/BUILD file for open-source tests/unit/m3 targets."""
  rel = _rel_to_maxtext(path, maxtext_root)
  if rel != "tests/BUILD":
    return []
  issues: list[dict[str, Any]] = []
  with open(path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
      if changed_lines is not None and idx not in changed_lines:
        continue
      if "unit/m3/" in line and not line.lstrip().startswith("#"):
        issues.append(
            {
                "code": "OSS_M3_TEST_IN_BLAZE_TESTS_BUILD",
                "severity": "HIGH",
                "file": rel,
                "line": idx,
                "message": (
                    "tests/BUILD forbids adding Bazel py_test targets for"
                    " open-source GitHub CI unit tests under tests/unit/m3/."
                ),
            }
        )
  return issues


def run_audit(
    maxtext_root: str,
    files: Sequence[str],
    check_reg: bool = False,
    base_ref: str = "",
    check_blaze_build: bool = False,
) -> dict[str, Any]:
  """Runs all requested m3 contract checks and returns a JSON-serializable report."""
  git_toplevel = ""
  merge_base = ""
  if base_ref:
    git_toplevel, merge_base = _resolve_git_merge_base(maxtext_root, base_ref)

  issues: list[dict[str, Any]] = []
  issues.extend(check_dual_driver_wiring(maxtext_root, base_ref=merge_base, git_toplevel=git_toplevel))
  if check_reg:
    issues.extend(check_registries(maxtext_root, base_ref=merge_base, git_toplevel=git_toplevel))

  changed_map = _changed_lines_by_file(maxtext_root, git_toplevel, merge_base, files) if merge_base else None

  for fpath in files:
    if not os.path.isfile(fpath):
      continue
    rel = _rel_to_maxtext(fpath, maxtext_root)
    f_changed = changed_map.get(rel, set()) if changed_map is not None else None
    if fpath.endswith(".py"):
      existed = bool(merge_base and _git_show_file(maxtext_root, git_toplevel, merge_base, rel) is not None)
      issues.extend(
          scan_m3_python_file(
              fpath,
              maxtext_root,
              changed_lines=f_changed,
              existed_at_base=existed,
          )
      )
    elif (
        check_blaze_build
        and os.path.basename(fpath) == "BUILD"
        and os.path.isfile(os.path.join(maxtext_root, "tests", "BUILD"))
    ):
      issues.extend(check_tests_build_file(fpath, maxtext_root, changed_lines=f_changed))

  deduped: list[dict[str, Any]] = []
  seen: set[tuple[str, str, int, str]] = set()
  for item in issues:
    key = (item["code"], item["file"], int(item["line"]), item["message"])
    if key not in seen:
      seen.add(key)
      deduped.append(item)

  return {
      "status": "FAIL" if deduped else "PASS",
      "files_audited": len(files),
      "issue_count": len(deduped),
      "issues": deduped,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Audit MaxText m3 contracts.")
  parser.add_argument(
      "--maxtext-root",
      required=True,
      help="Path to the MaxText root directory (containing src/maxtext/).",
  )
  parser.add_argument(
      "--check-registries",
      action="store_true",
      help="Also audit MODEL_REGISTRY vs MODEL_FLOPS_REGISTRY completeness.",
  )
  parser.add_argument(
      "--base",
      default="",
      help="Optional git base ref to scope checks to newly added/modified lines.",
  )
  parser.add_argument(
      "--check-blaze-build",
      action="store_true",
      default=False,
      help="Optionally check tests/BUILD when present (auto-skipped when absent).",
  )
  parser.add_argument(
      "files",
      nargs="*",
      help="Touched Python or BUILD files to audit.",
  )
  args = parser.parse_args(argv)
  try:
    report = run_audit(
        args.maxtext_root,
        args.files,
        check_reg=args.check_registries,
        base_ref=args.base,
        check_blaze_build=args.check_blaze_build,
    )
  except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
    sys.stderr.write(f"Error: {exc}\n")
    return 2
  sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
