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

"""Audits repo coupling, torch-free CI safety, mock configs, and config schemas.

Catches cross-repo and CI failures before handoff:
- TORCH_TOP_LEVEL_IMPORT: unguarded top-level torch / safetensors.torch import
  in a CI-collected test file or a helper transitively imported at top level
- TF_TOP_LEVEL_IMPORT: unguarded top-level tensorflow / tensorflow_datasets /
  tfds_data_processing import in an exported production module or CI-reachable
  test/helper (breaks TF-free PR cpu-unit collection)
- TORCH_MODULE_IMPORTORSKIP: module-level pytest.importorskip('torch') skipping
  entire test files and causing 0% patch coverage in torch-free CI
- TORCH_FREE_IMPORT_FAILED: collected test fails to import when torch is blocked
- EAGER_VS_JIT_GRAD_COMPARE: comparing jax.jit(value_and_grad) against eager
  value_and_grad in the same test function
- MOCK_CONFIG_MISSING_ATTR: bare self.config.<attr> read in a shared class whose
  existing SimpleNamespace mock tests construct that class without <attr>
- SILENT_CONFIG_FALLBACK: getattr(config, ..., default) or hasattr(config, ...)
  masking missing or undeclared config schema fields
- TORCH_EAGER_CHECKPOINT_LOAD: eager torch.load / from_pretrained in converter

Exit code: 0 = PASS, 1 = HIGH-confidence issues found, 2 = usage or I/O error.
"""

import argparse
import ast
from collections.abc import Mapping, Sequence
import configparser
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

_EXEMPT_SUBCONFIG_ATTRS = frozenset(
    {
        "text_config",
        "vision_config",
        "audio_config",
        "llm_config",
        "language_config",
    }
)
_POLYMORPHIC_RUNTIME_CFG_ATTRS = frozenset(
    {
        "shard_mode",
        "custom_mesh",
        "model_name",
        "quantization",
        "use_multimodal",
        "use_m3_model",
        "allow_split_physical_axes",
        "optimize_mesh_for_tpu_v6e",
        "context_parallel_strategy",
        "context_parallel_load_balance",
    }
)
_POLYMORPHIC_RUNTIME_CFG_PREFIXES = (
    "ici_",
    "dcn_",
    "context_parallel_",
)
_SHARED_RUNTIME_PATH_TOKENS = (
    "/core/",
    "/infra/",
    "/train/",
    "/trainers/",
    "/training_engine/",
    "/utils/",
)
_BLOCKED_TORCH_MODULES = (
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "safetensors.torch",
    "torchvision",
    "torchaudio",
    "tensorflow_datasets",
)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _dotted_name(node: ast.AST) -> str:
  """Returns dotted attribute name or identifier."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    prefix = _dotted_name(node.value)
    return f"{prefix}.{node.attr}" if prefix else node.attr
  return ""


def _tail_name(node: ast.AST) -> str:
  """Returns rightmost attribute or name."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    return node.attr
  return ""


def _is_config_expr(node: ast.AST) -> bool:
  """Returns True if node looks like config / cfg / self.config."""
  dotted = _dotted_name(node)
  parts = dotted.split(".")
  return any(
      p in ("config", "cfg", "text_config", "hf_config") or p.endswith("_cfg") or p.endswith("_config") for p in parts
  )


def _is_test_path(path: str) -> bool:
  """Returns True if path is a Python test module or under tests/."""
  base = os.path.basename(path)
  norm = os.path.abspath(path)
  return base.endswith(("_test.py", "_tests.py")) or base.startswith("test_") or f"{os.sep}tests{os.sep}" in norm


def _parse_pytest_ini(repo_root: str) -> dict[str, Any]:
  """Parses pytest.ini in repo_root for python_files, norecursedirs, addopts."""
  ini_path = os.path.join(repo_root, "pytest.ini")
  python_files = ["*_test.py", "test_*.py", "*_tests.py"]
  norecursedirs = {
      ".git",
      "__pycache__",
      ".pytest_cache",
      "build",
      "dist",
      "third_party",
  }
  ignored_paths: set[str] = set()
  ignored_globs: list[str] = []

  if not os.path.isfile(ini_path):
    return {
        "present": False,
        "python_files": python_files,
        "norecursedirs": norecursedirs,
        "ignored_paths": ignored_paths,
        "ignored_globs": ignored_globs,
    }

  cp = configparser.ConfigParser(interpolation=None)
  try:
    cp.read(ini_path, encoding="utf-8")
  except configparser.Error:
    return {
        "present": True,
        "python_files": python_files,
        "norecursedirs": norecursedirs,
        "ignored_paths": ignored_paths,
        "ignored_globs": ignored_globs,
    }

  sec = "pytest" if cp.has_section("pytest") else ("tool:pytest" if cp.has_section("tool:pytest") else "")
  if sec:
    pf_raw = cp.get(sec, "python_files", fallback="").strip()
    if pf_raw:
      python_files = [tok.strip() for tok in pf_raw.split() if tok.strip()]
    nr_raw = cp.get(sec, "norecursedirs", fallback="").strip()
    if nr_raw:
      norecursedirs.update(tok.strip() for tok in nr_raw.split() if tok.strip())
    addopts = cp.get(sec, "addopts", fallback="")
    for token in addopts.split():
      token = token.strip()
      if token.startswith("--ignore="):
        rel = token.split("=", 1)[1].strip().strip("\"'")
        if rel:
          ignored_paths.add(os.path.abspath(os.path.join(repo_root, rel)))
      elif token.startswith("--ignore-glob="):
        pat = token.split("=", 1)[1].strip().strip("\"'")
        if pat:
          ignored_globs.append(pat)

  return {
      "present": True,
      "python_files": python_files,
      "norecursedirs": norecursedirs,
      "ignored_paths": ignored_paths,
      "ignored_globs": ignored_globs,
  }


def _parse_coordinator_workflow(repo_root: str) -> bool:
  """Returns True if run_tests_coordinator.yml runs PR cpu-unit torch-free."""
  wf_path = os.path.join(repo_root, ".github", "workflows", "run_tests_coordinator.yml")
  if not os.path.isfile(wf_path):
    return False
  try:
    with open(wf_path, "r", encoding="utf-8") as f:
      text = f.read()
  except OSError:
    return False
  return "install_torch_cpu" in text and "is_scheduled_run" in text


def _discover_provenance_ref_dirs(repo_root: str, extra_dirs: Sequence[str]) -> list[str]:
  """Finds directories containing provenance.json plus --allowlist-ref-dir."""
  dirs: set[str] = {
      os.path.abspath(os.path.join(repo_root, d)) if not os.path.isabs(d) else os.path.abspath(d) for d in extra_dirs if d
  }
  for root, subdirs, files in os.walk(repo_root):
    subdirs[:] = sorted(d for d in subdirs if d not in (".git", "__pycache__", ".pytest_cache"))
    if "provenance.json" in files:
      dirs.add(os.path.abspath(root))
  return sorted(dirs)


def _is_under_any_dir(path: str, dirs: Sequence[str]) -> bool:
  """Returns True if `path` is inside any directory in `dirs`."""
  abs_p = os.path.abspath(path)
  for d in dirs:
    abs_d = os.path.abspath(d)
    try:
      if os.path.commonpath([abs_p, abs_d]) == abs_d:
        return True
    except ValueError:
      continue
  return False


def _is_ignored_by_pytest(path: str, repo_root: str, pytest_cfg: Mapping[str, Any]) -> bool:
  """Returns True if `path` is excluded by pytest.ini ignore/norecursedirs."""
  abs_p = os.path.abspath(path)
  abs_root = os.path.abspath(repo_root)
  ignored_paths: set[str] = pytest_cfg.get("ignored_paths", set())
  if abs_p in ignored_paths or _is_under_any_dir(abs_p, tuple(ignored_paths)):
    return True
  try:
    rel = os.path.relpath(abs_p, abs_root).replace(os.sep, "/")
  except ValueError:
    rel = os.path.basename(abs_p)
  parts = [p for p in rel.split("/") if p and p != "."]
  norecursedirs: set[str] = pytest_cfg.get("norecursedirs", set())
  if any(part in norecursedirs for part in parts[:-1]):
    return True
  for pat in pytest_cfg.get("ignored_globs", ()):
    if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(abs_p), pat):
      return True
  return False


def _is_collected_test_file(
    path: str,
    repo_root: str,
    pytest_cfg: Mapping[str, Any],
    allowlisted_dirs: Sequence[str],
) -> bool:
  """Returns True if `path` is a pytest-collected test module."""
  if _is_under_any_dir(path, allowlisted_dirs):
    return False
  if _is_ignored_by_pytest(path, repo_root, pytest_cfg):
    return False
  base = os.path.basename(path)
  patterns: Sequence[str] = pytest_cfg.get("python_files", ("*_test.py", "test_*.py", "*_tests.py"))
  return any(fnmatch.fnmatch(base, pat) for pat in patterns)


def _discover_repo_test_files(
    repo_root: str,
    pytest_cfg: Mapping[str, Any],
    allowlisted_dirs: Sequence[str],
) -> list[str]:
  """Finds pytest-collected test files under repo_root/tests (if present)."""
  tests_root = os.path.join(repo_root, "tests")
  if not os.path.isdir(tests_root):
    return []
  out: list[str] = []
  norecursedirs: set[str] = pytest_cfg.get("norecursedirs", set())
  for root, subdirs, files in os.walk(tests_root):
    subdirs[:] = sorted(d for d in subdirs if d not in norecursedirs)
    for fn in sorted(files):
      if not fn.endswith(".py"):
        continue
      full = os.path.join(root, fn)
      if _is_collected_test_file(full, repo_root, pytest_cfg, allowlisted_dirs):
        out.append(os.path.abspath(full))
  return sorted(set(out))


def _is_torch_import_name(mod_name: str, imported_symbol: str = "") -> bool:
  """Returns True if module/symbol imports torch or safetensors.torch."""
  if mod_name == "torch" or mod_name.startswith("torch."):
    return True
  if mod_name == "safetensors.torch" or mod_name.startswith("safetensors.torch."):
    return True
  if mod_name == "safetensors" and imported_symbol == "torch":
    return True
  return False


def _is_tf_import_name(mod_name: str, imported_symbol: str = "") -> bool:
  """Returns True if module/symbol imports TF, TFDS, or tfds_data_processing."""
  if mod_name == "tensorflow" or (mod_name.startswith("tensorflow.") and not mod_name.startswith("tensorflow.compiler")):
    return True
  if mod_name == "tensorflow_datasets" or mod_name.startswith("tensorflow_datasets."):
    return True
  full = f"{mod_name}.{imported_symbol}".strip(".")
  return any(seg in ("tfds_data_processing", "tfds_data_processing_c4_mlperf") for seg in full.split("."))


def _default_repo_root() -> str:
  """Resolves MaxText repository root from __file__ ancestors or CWD."""
  starts = (
      os.path.dirname(os.path.abspath(__file__)),
      os.path.abspath(os.getcwd()),
  )
  for start in starts:
    cur = start
    while True:
      if os.path.isfile(os.path.join(cur, "src", "maxtext", "configs", "types.py")):
        return cur
      parent = os.path.dirname(cur)
      if parent == cur:
        break
      cur = parent
  return os.path.abspath(os.getcwd())


def _is_offline_or_tooling_script(path: str, repo_root: str) -> bool:
  """Returns True for offline dataset, benchmark, or developer-tooling scripts."""
  base = os.path.basename(path)
  if base.endswith(("_internal.py", "_offline.py")):
    return True
  try:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(repo_root)).replace(os.sep, "/")
  except ValueError:
    return False
  return rel.startswith(("tools/", "benchmarks/", "docs/", "tests/assets/", ".agents/"))


def _extract_unguarded_top_level_tf_imports(
    tree: ast.Module,
) -> tuple[list[tuple[str, int]], bool]:
  """Extracts unguarded module-level TF/TFDS imports and importorskip state."""
  tf_imports: list[tuple[str, int]] = []
  has_tf_importorskip = False
  for stmt in tree.body:
    for sub in ast.walk(stmt):
      if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        continue
      if isinstance(sub, ast.Call) and _tail_name(sub.func) == "importorskip":
        if sub.args and isinstance(sub.args[0], ast.Constant) and str(sub.args[0].value).startswith("tensorflow"):
          has_tf_importorskip = True
    if has_tf_importorskip:
      continue
    if isinstance(stmt, ast.Import):
      for alias in stmt.names:
        if _is_tf_import_name(alias.name):
          tf_imports.append((alias.name, stmt.lineno))
    elif isinstance(stmt, ast.ImportFrom):
      mod = stmt.module or ""
      for alias in stmt.names:
        if _is_tf_import_name(mod, alias.name):
          full = f"{mod}.{alias.name}" if mod else alias.name
          tf_imports.append((full, stmt.lineno))
  return tf_imports, has_tf_importorskip


def _extract_unguarded_top_level_imports(
    tree: ast.Module,
) -> tuple[list[tuple[str, int]], list[tuple[str, str | None, int, int]]]:
  """Extracts unguarded module-level torch imports and local module imports."""
  torch_imports: list[tuple[str, int]] = []
  local_imports: list[tuple[str, str | None, int, int]] = []

  for stmt in tree.body:
    if isinstance(stmt, ast.Import):
      for alias in stmt.names:
        if _is_torch_import_name(alias.name):
          torch_imports.append((alias.name, stmt.lineno))
        else:
          local_imports.append((alias.name, None, stmt.lineno, 0))
    elif isinstance(stmt, ast.ImportFrom):
      mod = stmt.module or ""
      lvl = int(stmt.level or 0)
      for alias in stmt.names:
        if _is_torch_import_name(mod, alias.name):
          full = f"{mod}.{alias.name}" if mod else alias.name
          torch_imports.append((full, stmt.lineno))
        else:
          local_imports.append((mod, alias.name, stmt.lineno, lvl))
  return torch_imports, local_imports


def _resolve_local_import(
    importer_path: str,
    mod_name: str,
    symbol: str | None,
    level: int,
    repo_root: str,
) -> list[str]:
  """Resolves an import statement to local .py files inside repo_root."""
  candidates: list[str] = []
  importer_dir = os.path.dirname(os.path.abspath(importer_path))
  roots: list[str] = []
  if level > 0:
    base_dir = importer_dir
    for _ in range(level - 1):
      base_dir = os.path.dirname(base_dir)
    roots.append(base_dir)
  else:
    roots.extend(
        [
            os.path.abspath(repo_root),
            os.path.join(os.path.abspath(repo_root), "src"),
            importer_dir,
        ]
    )

  mod_variants = [mod_name]
  if mod_name.startswith("maxtext."):
    mod_variants.append(mod_name[len("maxtext.") :])
  for mod_var in mod_variants:
    rel_mod = mod_var.replace(".", os.sep) if mod_var else ""
    for r in roots:
      if rel_mod and symbol:
        p_sub = os.path.join(r, rel_mod, f"{symbol}.py")
        if os.path.isfile(p_sub):
          candidates.append(os.path.abspath(p_sub))
      if rel_mod:
        p_mod = os.path.join(r, f"{rel_mod}.py")
        if os.path.isfile(p_mod):
          candidates.append(os.path.abspath(p_mod))
        p_init = os.path.join(r, rel_mod, "__init__.py")
        if os.path.isfile(p_init):
          candidates.append(os.path.abspath(p_init))
      elif symbol:
        p_sym = os.path.join(r, f"{symbol}.py")
        if os.path.isfile(p_sym):
          candidates.append(os.path.abspath(p_sym))
  return sorted(set(candidates))


def _build_ci_torch_reachability(
    collected_tests: Sequence[str],
    repo_root: str,
    allowlisted_dirs: Sequence[str],
) -> tuple[
    set[str],
    dict[str, list[tuple[str, int]]],
    dict[str, list[tuple[str, str, int]]],
]:
  """Traces unguarded top-level imports from collected tests.

  Args:
    collected_tests: Collected pytest test file paths.
    repo_root: Repository root directory.
    allowlisted_dirs: Vendored reference directories to exempt.

  Returns:
    reachable_files: set of abs paths reachable at top level from collected
      tests (including the collected tests themselves).
    direct_torch_by_file: abs_path -> list of (torch_mod, lineno)
    transitive_torch_by_test: test_abs_path -> list of (helper_abs_path,
      torch_mod, import_lineno_in_test)
  """
  reachable_files: set[str] = set()
  direct_torch_by_file: dict[str, list[tuple[str, int]]] = {}
  local_edges: dict[str, list[tuple[str, int]]] = {}

  queue: list[str] = [os.path.abspath(t) for t in collected_tests if not _is_under_any_dir(t, allowlisted_dirs)]
  for t in queue:
    reachable_files.add(t)

  idx = 0
  while idx < len(queue):
    cur = queue[idx]
    idx += 1
    try:
      with open(cur, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=cur)
    except (OSError, SyntaxError):
      continue
    t_imps, l_imps = _extract_unguarded_top_level_imports(tree)
    if t_imps:
      direct_torch_by_file[cur] = t_imps
    edges: list[tuple[str, int]] = []
    for mod_name, sym, lineno, lvl in l_imps:
      for target in _resolve_local_import(cur, mod_name, sym, lvl, repo_root):
        if _is_under_any_dir(target, allowlisted_dirs):
          continue
        edges.append((target, lineno))
        if target not in reachable_files:
          reachable_files.add(target)
          queue.append(target)
    local_edges[cur] = edges

  # For each collected test, check if any direct/transitive local import has
  # direct_torch_by_file entries.
  transitive_torch_by_test: dict[str, list[tuple[str, str, int]]] = {}
  for t in collected_tests:
    abs_t = os.path.abspath(t)
    for first_target, test_lineno in local_edges.get(abs_t, ()):
      visited: set[str] = {abs_t, first_target}
      sub_q = [first_target]
      while sub_q:
        node = sub_q.pop(0)
        if node in direct_torch_by_file:
          torch_mod, _ = direct_torch_by_file[node][0]
          transitive_torch_by_test.setdefault(abs_t, []).append((node, torch_mod, test_lineno))
          break
        for nxt, _ in local_edges.get(node, ()):
          if nxt not in visited:
            visited.add(nxt)
            sub_q.append(nxt)

  return reachable_files, direct_torch_by_file, transitive_torch_by_test


def _build_ci_tf_reachability(
    collected_tests: Sequence[str],
    repo_root: str,
    allowlisted_dirs: Sequence[str],
) -> tuple[
    dict[str, list[tuple[str, int]]],
    dict[str, list[tuple[str, str, int]]],
]:
  """Traces unguarded top-level TF/TFDS imports from collected tests."""
  direct_tf_by_file: dict[str, list[tuple[str, int]]] = {}
  tf_skipped_tests: set[str] = set()
  local_edges: dict[str, list[tuple[str, int]]] = {}
  visited_files: set[str] = set()

  queue: list[str] = [
      os.path.abspath(t)
      for t in collected_tests
      if not _is_under_any_dir(t, allowlisted_dirs) and not _is_offline_or_tooling_script(t, repo_root)
  ]
  visited_files.update(queue)

  idx = 0
  while idx < len(queue):
    cur = queue[idx]
    idx += 1
    try:
      with open(cur, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=cur)
    except (OSError, SyntaxError):
      continue
    tf_imps, has_tf_skip = _extract_unguarded_top_level_tf_imports(tree)
    if has_tf_skip:
      tf_skipped_tests.add(cur)
    if tf_imps:
      direct_tf_by_file[cur] = tf_imps
    _, l_imps = _extract_unguarded_top_level_imports(tree)
    edges: list[tuple[str, int]] = []
    for mod_name, sym, lineno, lvl in l_imps:
      if has_tf_skip:
        continue
      for target in _resolve_local_import(cur, mod_name, sym, lvl, repo_root):
        if _is_under_any_dir(target, allowlisted_dirs) or _is_offline_or_tooling_script(target, repo_root):
          continue
        edges.append((target, lineno))
        if target not in visited_files:
          visited_files.add(target)
          queue.append(target)
    local_edges[cur] = edges

  transitive_tf_by_test: dict[str, list[tuple[str, str, int]]] = {}
  for t in collected_tests:
    abs_t = os.path.abspath(t)
    if abs_t in tf_skipped_tests:
      continue
    for first_target, test_lineno in local_edges.get(abs_t, ()):
      visited: set[str] = {abs_t, first_target}
      sub_q = [first_target]
      while sub_q:
        node = sub_q.pop(0)
        if node in direct_tf_by_file:
          tf_mod, _ = direct_tf_by_file[node][0]
          transitive_tf_by_test.setdefault(abs_t, []).append((node, tf_mod, test_lineno))
          break
        for nxt, _ in local_edges.get(node, ()):
          if nxt not in visited:
            visited.add(nxt)
            sub_q.append(nxt)

  return direct_tf_by_file, transitive_tf_by_test


def _scan_test_or_helper_ast(  # pylint: disable=too-many-nested-blocks
    path: str,
    tree: ast.Module,
    *,
    is_ci_reachable: bool,
    is_collected_test: bool,
    direct_torch: Sequence[tuple[str, int]],
    transitive_torch: Sequence[tuple[str, str, int]],
    direct_tf: Sequence[tuple[str, int]] = (),
    transitive_tf: Sequence[tuple[str, str, int]] = (),
) -> list[dict[str, Any]]:
  """Checks top-level torch/TF imports, skips, jit-vs-eager grad, K==V dims."""
  issues: list[dict[str, Any]] = []

  if is_ci_reachable:
    for mod_name, lineno in direct_torch:
      issues.append(
          {
              "code": "TORCH_TOP_LEVEL_IMPORT",
              "severity": "HIGH",
              "confidence": "HIGH",
              "file": path,
              "line": lineno,
              "message": (
                  f"Unguarded top-level import `{mod_name}` in CI-reachable module"
                  " breaks pytest collection in torch-free PR CI; guard with"
                  " HAS_TORCH or import inside @skipif test methods."
              ),
          }
      )

    for mod_name, lineno in direct_tf:
      issues.append(
          {
              "code": "TF_TOP_LEVEL_IMPORT",
              "severity": "HIGH",
              "confidence": "HIGH",
              "file": path,
              "line": lineno,
              "message": (
                  f"Unguarded top-level import `{mod_name}` in CI-reachable module"
                  " breaks pytest collection in TF-free PR cpu-unit CI; guard with"
                  " try/except ImportError, lazy import, or"
                  " pytest.importorskip('tensorflow_datasets')."
              ),
          }
      )

    if is_collected_test:
      for helper_path, torch_mod, lineno in transitive_torch:
        issues.append(
            {
                "code": "TORCH_TOP_LEVEL_IMPORT",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": lineno,
                "message": (
                    f"Collected test imports `{os.path.basename(helper_path)}` at"
                    f" top level, which transitively imports `{torch_mod}`"
                    " unguarded in torch-free CI."
                ),
            }
        )

      for helper_path, tf_mod, lineno in transitive_tf:
        issues.append(
            {
                "code": "TF_TOP_LEVEL_IMPORT",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": lineno,
                "message": (
                    f"Collected test imports `{os.path.basename(helper_path)}` at"
                    f" top level, which transitively imports `{tf_mod}`"
                    " unguarded in TF-free PR cpu-unit CI."
                ),
            }
        )

    for stmt in tree.body:
      for sub in ast.walk(stmt):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
          continue
        if isinstance(sub, ast.Call) and _tail_name(sub.func) == "importorskip":
          if sub.args and isinstance(sub.args[0], ast.Constant) and str(sub.args[0].value).startswith("torch"):
            issues.append(
                {
                    "code": "TORCH_MODULE_IMPORTORSKIP",
                    "severity": "HIGH",
                    "confidence": "HIGH",
                    "file": path,
                    "line": sub.lineno,
                    "message": (
                        "Module-level pytest.importorskip('torch') skips 100% of"
                        " tests in torch-free CI and causes 0% codecov/patch"
                        " failure; keep float64 NumPy oracle tests unconditional."
                    ),
                }
            )

  if not is_collected_test:
    return issues

  # Per-function eager-vs-jit grad comparison.
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
      has_jit_grad = False
      has_eager_grad = False
      jit_grad_calls: set[int] = set()
      for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and _tail_name(sub.func) == "jit":
          for arg in sub.args:
            for inner in ast.walk(arg):
              if isinstance(inner, ast.Call) and _tail_name(inner.func) in (
                  "grad",
                  "value_and_grad",
              ):
                has_jit_grad = True
                jit_grad_calls.add(id(inner))
      for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and _tail_name(sub.func) in ("grad", "value_and_grad")
            and id(sub) not in jit_grad_calls
        ):
          has_eager_grad = True
      if has_jit_grad and has_eager_grad:
        issues.append(
            {
                "code": "EAGER_VS_JIT_GRAD_COMPARE",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": node.lineno,
                "message": (
                    f"{node.name} mixes jax.jit(grad/value_and_grad) with eager"
                    " grad/value_and_grad; XLA contraction fusion causes 4-device"
                    " false mismatches. Wrap both reference and sharded runs in"
                    " jax.jit."
                ),
            }
        )

  return issues


def _extract_prod_config_reads(  # pylint: disable=too-many-nested-blocks
    tree: ast.Module,
) -> dict[str, dict[str, int]]:
  """Maps ClassName -> {attr_name: first_lineno} of bare self.config.<attr>."""
  class_attrs: dict[str, dict[str, int]] = {}
  for node in ast.walk(tree):
    if not isinstance(node, ast.ClassDef):
      continue
    guarded_hasattr: set[str] = set()
    for sub in ast.walk(node):
      if (
          isinstance(sub, ast.Call)
          and _tail_name(sub.func) == "hasattr"
          and len(sub.args) >= 2
          and _is_config_expr(sub.args[0])
          and isinstance(sub.args[1], ast.Constant)
          and isinstance(sub.args[1].value, str)
      ):
        guarded_hasattr.add(sub.args[1].value)

    attrs: dict[str, int] = {}
    for sub in ast.walk(node):
      if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Attribute):
        if isinstance(sub.value.value, ast.Name) and sub.value.value.id == "self" and sub.value.attr in ("config", "cfg"):
          attr = sub.attr
          if attr not in guarded_hasattr and not attr.startswith("_"):
            if attr not in attrs or sub.lineno < attrs[attr]:
              attrs[attr] = int(sub.lineno)
    if attrs:
      class_attrs[node.name] = attrs
  return class_attrs


def _extract_base_class_attrs(prod_path: str, repo_root: str, base_ref: str) -> dict[str, set[str]]:
  """Extracts {ClassName: set(attrs)} from `<base_ref>:<rel_path>` via git."""
  git_bin = shutil.which("git")
  if not git_bin or not base_ref:
    return {}
  root_abs = os.path.abspath(repo_root)
  try:
    rel_path = os.path.relpath(os.path.abspath(prod_path), root_abs).replace(os.sep, "/")
  except ValueError:
    return {}
  proc = subprocess.run(
      [git_bin, "-c", "safe.directory=*", "-C", root_abs, "show", f"{base_ref}:{rel_path}"],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    return {}
  try:
    base_tree = ast.parse(proc.stdout, filename=rel_path)
  except SyntaxError:
    return {}
  raw = _extract_prod_config_reads(base_tree)
  return {cls_name: set(amap.keys()) for cls_name, amap in raw.items()}


def _extract_ns_keys_in_scope(  # pylint: disable=too-many-nested-blocks
    scope_node: ast.AST, helper_ns_map: Mapping[str, set[str]]
) -> tuple[bool, set[str]]:
  """Extracts SimpleNamespace attribute keys constructed or assigned in scope."""
  has_ns = False
  keys: set[str] = set()
  for sub in ast.walk(scope_node):
    if isinstance(sub, ast.Call):
      cname = _tail_name(sub.func)
      if cname == "SimpleNamespace":
        has_ns = True
        for kw in sub.keywords:
          if kw.arg is not None:
            keys.add(kw.arg)
          elif isinstance(kw.value, ast.Dict):
            for k in kw.value.keys:
              if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.add(k.value)
      elif cname in helper_ns_map:
        has_ns = True
        keys.update(helper_ns_map[cname])
      elif (
          cname == "setattr"
          and len(sub.args) >= 2
          and isinstance(sub.args[1], ast.Constant)
          and isinstance(sub.args[1].value, str)
      ):
        keys.add(sub.args[1].value)
    elif isinstance(sub, ast.Assign):
      for tgt in sub.targets:
        if isinstance(tgt, ast.Attribute):
          keys.add(tgt.attr)
  return has_ns, keys


def _find_mocked_classes_in_tests(test_files: Sequence[str]) -> set[str]:
  """Returns set of class names constructed inside SimpleNamespace mock tests."""
  mocked: set[str] = set()
  for tpath in test_files:
    try:
      with open(tpath, "r", encoding="utf-8") as f:
        tsrc = f.read()
      ttree = ast.parse(tsrc, filename=tpath)
    except (OSError, SyntaxError):
      continue
    if "SimpleNamespace" not in tsrc:
      continue
    for sub in ast.walk(ttree):
      if isinstance(sub, ast.Call):
        cname = _tail_name(sub.func)
        if cname and cname[0].isupper() and cname != "SimpleNamespace":
          mocked.add(cname)
  return mocked


def _check_mock_config_blast_radius(
    prod_path: str,
    class_attrs: Mapping[str, Mapping[str, int]],
    test_files: Sequence[str],
    *,
    base_class_attrs: Mapping[str, set[str]] | None = None,
    changed_lines: set[int] | None = None,
) -> list[dict[str, Any]]:
  """Scans collected test ASTs for SimpleNamespace mocks calling classes."""
  if not class_attrs or not test_files:
    return []

  # Filter to newly added attributes when --base diff scoping is active.
  filtered_attrs: dict[str, dict[str, int]] = {}
  for cls_name, amap in class_attrs.items():
    base_set = (base_class_attrs or {}).get(cls_name, set())
    kept: dict[str, int] = {}
    for attr, lineno in amap.items():
      if base_class_attrs is not None and attr in base_set:
        continue
      if changed_lines is not None and lineno not in changed_lines:
        continue
      kept[attr] = lineno
    if kept:
      filtered_attrs[cls_name] = kept

  if not filtered_attrs:
    return []

  affected_by_attr: dict[tuple[str, str, int], set[str]] = {}
  for tpath in test_files:
    try:
      with open(tpath, "r", encoding="utf-8") as f:
        tsrc = f.read()
    except OSError:
      continue
    if "SimpleNamespace" not in tsrc:
      continue
    try:
      ttree = ast.parse(tsrc, filename=tpath)
    except SyntaxError:
      continue

    # Map helper functions in this test file that construct SimpleNamespace.
    helper_ns_map: dict[str, set[str]] = {}
    for node in ast.walk(ttree):
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        has_ns, keys = _extract_ns_keys_in_scope(node, {})
        if has_ns and keys:
          helper_ns_map[node.name] = keys

    # Check each function/method scope (and module scope) that actually calls
    # `cls_name(...)` as a constructor.
    scopes: list[ast.AST] = [
        n for n in ast.walk(ttree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not scopes:
      scopes = [ttree]

    for scope in scopes:
      called_classes = {_tail_name(sub.func) for sub in ast.walk(scope) if isinstance(sub, ast.Call)}
      matched_classes = [c for c in filtered_attrs if c in called_classes]
      if not matched_classes:
        continue
      has_ns, ns_keys = _extract_ns_keys_in_scope(scope, helper_ns_map)
      if not has_ns or not ns_keys:
        continue
      for cls_name in matched_classes:
        for attr, lineno in filtered_attrs[cls_name].items():
          if attr not in ns_keys:
            affected_by_attr.setdefault((cls_name, attr, lineno), set()).add(tpath)

  issues: list[dict[str, Any]] = []
  for (cls_name, attr, lineno), tpaths in sorted(affected_by_attr.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
    sorted_tests = sorted(tpaths)
    sample_test = sorted_tests[0]
    extra = f" (+{len(sorted_tests) - 1} more)" if len(sorted_tests) > 1 else ""
    issues.append(
        {
            "code": "MOCK_CONFIG_MISSING_ATTR",
            "severity": "HIGH",
            "confidence": "HIGH",
            "file": prod_path,
            "line": lineno,
            "affected_tests": sorted_tests,
            "message": (
                f"{cls_name} reads self.config.{attr} on line {lineno}, which is"
                f" absent from SimpleNamespace mock test {sample_test}{extra}"
                f" (update the mock fixture or use getattr(self.config, {attr!r},"
                " sentinel))."
            ),
        }
    )
  return issues


def _collect_declared_config_fields(repo_root: str) -> frozenset[str]:  # pylint: disable=too-many-nested-blocks
  """Collects declared attributes from MaxText config schema classes."""
  candidates: list[str] = []
  types_py = os.path.join(repo_root, "src", "maxtext", "configs", "types.py")
  if os.path.isfile(types_py):
    candidates.append(types_py)
  m3_cfg_dir = os.path.join(repo_root, "src", "maxtext", "m3", "configs")
  if os.path.isdir(m3_cfg_dir):
    for fn in sorted(os.listdir(m3_cfg_dir)):
      if fn.endswith(".py"):
        candidates.append(os.path.join(m3_cfg_dir, fn))
  if not candidates:
    return frozenset()

  fields: set[str] = set()
  for path in candidates:  # pylint: disable=too-many-nested-blocks
    try:
      with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    except (OSError, SyntaxError):
      continue
    for node in ast.walk(tree):
      if not isinstance(node, ast.ClassDef):
        continue
      for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
          fields.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
          for target in stmt.targets:
            if isinstance(target, ast.Name):
              fields.add(target.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
          for dec in stmt.decorator_list:
            if _tail_name(dec) in ("property", "cached_property"):
              fields.add(stmt.name)
              break
  return frozenset(fields)


def _scan_prod_file_ast(  # pylint: disable=too-many-nested-blocks
    path: str,
    tree: ast.Module,
    *,
    shared_legacy: bool,
    mocked_classes: set[str],
    check_tf_imports: bool = True,
    declared_cfg_fields: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
  """Scans a production .py file for TF/torch hazards and config fallbacks."""
  issues: list[dict[str, Any]] = []

  if check_tf_imports:
    tf_imps, _ = _extract_unguarded_top_level_tf_imports(tree)
    for mod_name, lineno in tf_imps:
      issues.append(
          {
              "code": "TF_TOP_LEVEL_IMPORT",
              "severity": "HIGH",
              "confidence": "HIGH",
              "file": path,
              "line": lineno,
              "message": (
                  f"Unguarded top-level import `{mod_name}` in exported module"
                  " breaks import in TF-free PR cpu-unit CI; wrap in try/except"
                  " ImportError, lazy-import inside the TFDS branch, or use"
                  " etils.epath instead of tf.io.gfile."
              ),
          }
      )

  norm_path = os.path.abspath(path).replace(os.sep, "/")
  is_shared_runtime_path = (
      any(tok in norm_path for tok in _SHARED_RUNTIME_PATH_TOKENS) or os.path.basename(norm_path) == "__init__.py"
  )

  def _visit(node: ast.AST, cls_name: str = "") -> None:
    cur_cls = node.name if isinstance(node, ast.ClassDef) else cls_name
    if isinstance(node, ast.Call):
      tail = _tail_name(node.func)
      dotted = _dotted_name(node.func)
      if not shared_legacy:
        if declared_cfg_fields and tail == "hasattr" and len(node.args) >= 2 and _is_config_expr(node.args[0]):
          obj_d = _dotted_name(node.args[0])
          is_hf_or_subattr = any(tok in obj_d for tok in ("hf_config", "hf_cfg", "pretrained")) or (
              obj_d.count(".") >= 1 and not obj_d.startswith("self.")
          )
          arg1 = node.args[1]
          if not is_hf_or_subattr and isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
            attr_name = arg1.value
            if attr_name and attr_name not in _EXEMPT_SUBCONFIG_ATTRS and attr_name not in declared_cfg_fields:
              issues.append(
                  {
                      "code": "SILENT_CONFIG_FALLBACK",
                      "severity": "HIGH",
                      "confidence": "HIGH",
                      "file": path,
                      "line": node.lineno,
                      "message": (
                          f"`hasattr` probes undeclared config field {attr_name!r}"
                          f" in `{ast.unparse(node)}`; declare the field on the"
                          " config schema or remove the dead probe."
                      ),
                  }
              )
        elif tail == "getattr" and len(node.args) >= 3 and _is_config_expr(node.args[0]):
          obj_d = _dotted_name(node.args[0])
          arg1 = node.args[1]
          if isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
            is_literal_attr = True
            attr_name = arg1.value
          else:
            is_literal_attr = False
            attr_name = ""
          is_enum_value = attr_name == "value" and (obj_d.count(".") >= 1 and obj_d not in ("self.config", "self.cfg"))
          is_hf_or_subattr = any(tok in obj_d for tok in ("hf_config", "hf_cfg", "pretrained")) or (
              obj_d.count(".") >= 1 and not obj_d.startswith("self.")
          )
          if attr_name not in _EXEMPT_SUBCONFIG_ATTRS and not is_enum_value:
            if declared_cfg_fields and is_literal_attr and not is_hf_or_subattr and attr_name not in declared_cfg_fields:
              if attr_name not in _POLYMORPHIC_RUNTIME_CFG_ATTRS:
                issues.append(
                    {
                        "code": "SILENT_CONFIG_FALLBACK",
                        "severity": "HIGH",
                        "confidence": "HIGH",
                        "file": path,
                        "line": node.lineno,
                        "message": (
                            f"Silent config fallback `{ast.unparse(node)}` probes"
                            f" undeclared config field {attr_name!r}; declare the"
                            " field or read typed config attributes directly."
                        ),
                    }
                )
            elif not (
                declared_cfg_fields and is_literal_attr and attr_name in declared_cfg_fields and is_shared_runtime_path
            ):
              is_polymorphic_probe = (
                  attr_name in _POLYMORPHIC_RUNTIME_CFG_ATTRS
                  or attr_name.startswith(_POLYMORPHIC_RUNTIME_CFG_PREFIXES)
                  or (not is_literal_attr and is_shared_runtime_path)
              )
              if not is_polymorphic_probe:
                default_node = node.args[2]
                is_none_default = isinstance(default_node, ast.Constant) and default_node.value is None
                # Allow legacy numeric/boolean sentinel on `self.config` when
                # the enclosing class is mocked with SimpleNamespace in tests.
                is_self_cfg = obj_d in ("self.config", "self.cfg")
                if not (is_self_cfg and cur_cls in mocked_classes and not is_none_default):
                  issues.append(
                      {
                          "code": "SILENT_CONFIG_FALLBACK",
                          "severity": "HIGH",
                          "confidence": "HIGH",
                          "file": path,
                          "line": node.lineno,
                          "message": (
                              f"Silent config fallback `{ast.unparse(node)}` masks"
                              " missing or misspelled config fields; read typed"
                              " config attributes directly."
                          ),
                      }
                  )
        elif (
            tail == "get"
            and isinstance(node.func, ast.Attribute)
            and len(node.args) >= 2
            and _is_config_expr(node.func.value)
        ):
          key_name = ""
          if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            key_name = node.args[0].value
          if key_name not in _EXEMPT_SUBCONFIG_ATTRS:
            issues.append(
                {
                    "code": "SILENT_CONFIG_FALLBACK",
                    "severity": "HIGH",
                    "confidence": "HIGH",
                    "file": path,
                    "line": node.lineno,
                    "message": (
                        f"Silent config dict fallback `{ast.unparse(node)}`;" " require explicit keys and fail closed."
                    ),
                }
            )
      if dotted in ("torch.load", "AutoModelForCausalLM.from_pretrained"):
        issues.append(
            {
                "code": "TORCH_EAGER_CHECKPOINT_LOAD",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": node.lineno,
                "message": (
                    f"`{dotted}(...)` eagerly loads full weights into host RAM;"
                    " stream tensors per shard via safetensors.safe_open."
                ),
            }
        )
    for child in ast.iter_child_nodes(node):
      _visit(child, cur_cls)

  _visit(tree)
  return issues


def _find_python3() -> str | None:
  """Returns a Python 3 interpreter path if available."""
  if sys.executable:
    return sys.executable
  return shutil.which("python3")


def _import_in_process_without_torch(tpath: str, add_paths: Sequence[str]) -> str | None:
  """Fallback torch-blocked import check for embedded test runtimes."""
  saved_all_modules = dict(sys.modules)
  saved_path = list(sys.path)
  mod_name = "_audit_mod"
  try:
    for k in list(sys.modules):
      if k == "tests" or k.startswith("tests."):
        sys.modules.pop(k, None)
    for k in _BLOCKED_TORCH_MODULES:
      sys.modules[k] = None  # type: ignore[assignment]
    sys.path = list(add_paths) + saved_path
    spec = importlib.util.spec_from_file_location(mod_name, os.path.abspath(tpath))
    if spec is None or spec.loader is None:
      return f"Could not create module spec for {tpath}"
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so @dataclass lookup succeeds.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return None
  except (KeyboardInterrupt, SystemExit):
    raise
  except BaseException as exc:  # pylint: disable=broad-exception-caught
    if type(exc).__name__ in ("Skipped", "SkipTest"):
      return None
    return f"{type(exc).__name__}: {exc}"
  finally:
    sys.path = saved_path
    sys.modules.clear()
    sys.modules.update(saved_all_modules)


def _run_torch_free_import_check(test_files: Sequence[str], repo_root: str) -> list[dict[str, Any]]:
  """Imports test modules with torch blocked via subprocess or in-process."""
  issues: list[dict[str, Any]] = []
  py_bin = _find_python3()
  src_dir = os.path.join(repo_root, "src")
  add_paths = [repo_root] + ([src_dir] if os.path.isdir(src_dir) else [])

  if py_bin is None:
    for tpath in test_files:
      err = _import_in_process_without_torch(tpath, add_paths)
      if err is not None:
        issues.append(
            {
                "code": "TORCH_FREE_IMPORT_FAILED",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": tpath,
                "line": 1,
                "message": (f"Importing test module with torch blocked failed: {err}"),
            }
        )
    return issues

  driver = (
      "import importlib.util, sys\n"
      "for _k in ('torch', 'torch.nn', 'torch.nn.functional',"
      " 'safetensors.torch', 'torchvision', 'torchaudio',"
      " 'tensorflow_datasets'):\n"
      "  sys.modules[_k] = None\n"
      "p = sys.argv[1]\n"
      "spec = importlib.util.spec_from_file_location('_audit_mod', p)\n"
      "mod = importlib.util.module_from_spec(spec)\n"
      "sys.modules[spec.name] = mod\n"
      "try:\n"
      "  spec.loader.exec_module(mod)\n"
      "except BaseException as _e:\n"
      "  if type(_e).__name__ not in ('Skipped', 'SkipTest'):\n"
      "    raise\n"
  )
  env = dict(os.environ)
  env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
  existing_pp = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = os.pathsep.join(add_paths + ([existing_pp] if existing_pp else []))

  for tpath in test_files:
    proc = subprocess.run(
        [py_bin, "-", os.path.abspath(tpath)],
        input=driver,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
      err_tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
      issues.append(
          {
              "code": "TORCH_FREE_IMPORT_FAILED",
              "severity": "HIGH",
              "confidence": "HIGH",
              "file": tpath,
              "line": 1,
              "message": ("Importing test module with torch blocked failed: " + " | ".join(err_tail)),
          }
      )
  return issues


def _resolve_merge_base(repo_root: str, base_ref: str) -> str:
  """Resolves `git merge-base <base_ref> HEAD`, raising OSError on failure."""
  git_bin = shutil.which("git")
  if not git_bin:
    raise OSError("git executable not found in PATH")
  root_abs = os.path.abspath(repo_root)
  proc = subprocess.run(
      [
          git_bin,
          "-c",
          "safe.directory=*",
          "-C",
          root_abs,
          "merge-base",
          base_ref,
          "HEAD",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  mb = proc.stdout.strip()
  if proc.returncode != 0 or not mb:
    detail = proc.stderr.strip() or f"exit code {proc.returncode}"
    raise OSError(f"git merge-base {base_ref!r} HEAD failed in {root_abs}: {detail}")
  return mb


def _compute_changed_lines(repo_root: str, base_ref: str, files: Sequence[str]) -> dict[str, set[int]]:
  """Computes added/modified line numbers per file from `git diff -U0 <merge_base>`."""
  if not base_ref or not files:
    return {}
  mb = _resolve_merge_base(repo_root, base_ref)
  git_bin = shutil.which("git")
  if not git_bin:
    raise OSError("git executable not found in PATH")
  root_abs = os.path.abspath(repo_root)
  rel_files = [os.path.relpath(os.path.abspath(f), root_abs) for f in files if os.path.exists(f)]
  if not rel_files:
    return {}
  cmd = [
      git_bin,
      "-c",
      "safe.directory=*",
      "-C",
      root_abs,
      "diff",
      "-U0",
      mb,
      "--",
  ] + rel_files
  proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    detail = proc.stderr.strip() or f"exit code {proc.returncode}"
    raise OSError(f"git diff -U0 {mb} failed in {root_abs}: {detail}")

  changed: dict[str, set[int]] = {}
  cur_file: str | None = None
  for line in proc.stdout.splitlines():
    if line.startswith("+++ b/"):
      rel_p = line[6:].strip()
      cur_file = os.path.abspath(os.path.join(root_abs, rel_p))
      changed.setdefault(cur_file, set())
    elif line.startswith("@@ ") and cur_file is not None:
      match = _HUNK_RE.match(line)
      if match:
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) is not None else 1
        if count > 0:
          changed[cur_file].update(range(start, start + count))
  return changed


def _discover_diff_files(repo_root: str, base_ref: str) -> list[str]:
  """Returns modified/added .py files from `git diff --name-only <merge_base>`."""
  if not base_ref:
    return []
  mb = _resolve_merge_base(repo_root, base_ref)
  git_bin = shutil.which("git")
  if not git_bin:
    raise OSError("git executable not found in PATH")
  root_abs = os.path.abspath(repo_root)
  proc = subprocess.run(
      [
          git_bin,
          "-c",
          "safe.directory=*",
          "-C",
          root_abs,
          "diff",
          "--name-only",
          "--diff-filter=ACMR",
          mb,
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    detail = proc.stderr.strip() or f"exit code {proc.returncode}"
    raise OSError(f"git diff --name-only {mb} failed in {root_abs}: {detail}")
  out: list[str] = []
  for rel in proc.stdout.splitlines():
    rel = rel.strip()
    if rel.endswith(".py"):
      full = os.path.abspath(os.path.join(root_abs, rel))
      if os.path.isfile(full):
        out.append(full)
  return sorted(set(out))


def audit_files(
    *,
    repo_root: str,
    files: Sequence[str],
    base_ref: str = "",
    allowlist_ref_dirs: Sequence[str] = (),
    run_torch_free_import: bool = False,
    shared_legacy: bool = False,
    fail_on_warn: bool = False,
) -> dict[str, Any]:
  """Runs all repository, CI, mock-config, and config schema checks."""
  pytest_cfg = _parse_pytest_ini(repo_root)
  ci_torch_free = _parse_coordinator_workflow(repo_root)
  allowlisted_dirs = _discover_provenance_ref_dirs(repo_root, allowlist_ref_dirs)
  resolved_base = _resolve_merge_base(repo_root, base_ref) if base_ref else ""
  if not files:
    return {
        "status": "PASS",
        "files_audited": 0,
        "ci_pr_cpu_unit_torch_free": ci_torch_free,
        "allowlisted_ref_dirs": allowlisted_dirs,
        "issue_count": 0,
        "high_confidence_issues": 0,
        "issues": [],
    }
  repo_collected_tests = _discover_repo_test_files(repo_root, pytest_cfg, allowlisted_dirs)
  declared_cfg_fields = _collect_declared_config_fields(repo_root)

  # Identify any explicitly passed collected test files as well.
  explicit_collected_tests: list[str] = []
  for fpath in files:
    if fpath.endswith(".py") and _is_collected_test_file(fpath, repo_root, pytest_cfg, allowlisted_dirs):
      explicit_collected_tests.append(os.path.abspath(fpath))

  all_collected_tests = sorted(set(repo_collected_tests) | set(explicit_collected_tests))
  (
      ci_reachable_files,
      direct_torch_by_file,
      transitive_torch_by_test,
  ) = _build_ci_torch_reachability(all_collected_tests, repo_root, allowlisted_dirs)
  direct_tf_by_file, transitive_tf_by_test = _build_ci_tf_reachability(all_collected_tests, repo_root, allowlisted_dirs)
  mocked_classes = _find_mocked_classes_in_tests(all_collected_tests)
  changed_map = _compute_changed_lines(repo_root, resolved_base, files) if resolved_base else {}

  issues: list[dict[str, Any]] = []
  touched_collected_tests: list[str] = []

  for fpath in files:
    if not fpath.endswith(".py"):
      continue
    if not os.path.isfile(fpath):
      raise FileNotFoundError(f"File not found: {fpath}")
    abs_f = os.path.abspath(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
      tree = ast.parse(f.read(), filename=fpath)

    is_allowlisted = _is_under_any_dir(abs_f, allowlisted_dirs)
    is_internal_or_offline = _is_offline_or_tooling_script(abs_f, repo_root)
    rel_norm = os.path.relpath(abs_f, os.path.abspath(repo_root)).replace(os.sep, "/")
    is_agent_tool_script = rel_norm.startswith(".agents/") and not _is_test_path(fpath)
    is_collected_test = not is_allowlisted and _is_collected_test_file(abs_f, repo_root, pytest_cfg, allowlisted_dirs)
    is_ci_reachable = not is_allowlisted and (is_collected_test or abs_f in ci_reachable_files)

    if _is_test_path(fpath):
      if is_collected_test:
        touched_collected_tests.append(fpath)
      direct_t = direct_torch_by_file.get(abs_f, [])
      if not direct_t and is_ci_reachable:
        direct_t, _ = _extract_unguarded_top_level_imports(tree)
      trans_t = transitive_torch_by_test.get(abs_f, [])
      direct_tf = direct_tf_by_file.get(abs_f, []) if not is_internal_or_offline else []
      if not direct_tf and is_ci_reachable and not is_internal_or_offline:
        direct_tf, _ = _extract_unguarded_top_level_tf_imports(tree)
      trans_tf = transitive_tf_by_test.get(abs_f, []) if not is_internal_or_offline else []
      file_issues = _scan_test_or_helper_ast(
          fpath,
          tree,
          is_ci_reachable=is_ci_reachable,
          is_collected_test=is_collected_test,
          direct_torch=direct_t,
          transitive_torch=trans_t,
          direct_tf=direct_tf,
          transitive_tf=trans_tf,
      )
      if resolved_base and abs_f in changed_map:
        file_issues = [i for i in file_issues if int(i["line"]) in changed_map[abs_f]]
      issues.extend(file_issues)
    else:
      class_attrs = _extract_prod_config_reads(tree)
      base_attrs = _extract_base_class_attrs(fpath, repo_root, resolved_base) if resolved_base else None
      c_lines = changed_map.get(abs_f) if resolved_base else None
      issues.extend(
          _check_mock_config_blast_radius(
              fpath,
              class_attrs,
              all_collected_tests,
              base_class_attrs=base_attrs,
              changed_lines=c_lines,
          )
      )
      prod_issues = _scan_prod_file_ast(
          fpath,
          tree,
          shared_legacy=shared_legacy or is_agent_tool_script,
          mocked_classes=mocked_classes,
          check_tf_imports=not is_allowlisted and not is_internal_or_offline,
          declared_cfg_fields=declared_cfg_fields,
      )
      if resolved_base and abs_f in changed_map:
        prod_issues = [i for i in prod_issues if int(i["line"]) in changed_map[abs_f]]
      issues.extend(prod_issues)

  if run_torch_free_import and touched_collected_tests:
    issues.extend(_run_torch_free_import_check(touched_collected_tests, repo_root))

  # Deduplicate identical issue dicts while preserving deterministic order.
  deduped: list[dict[str, Any]] = []
  seen: set[tuple[str, str, int, str]] = set()
  for item in issues:
    item.setdefault("severity", "HIGH")
    item.setdefault("confidence", "HIGH")
    key = (item["code"], item["file"], int(item["line"]), item["message"])
    if key not in seen:
      seen.add(key)
      deduped.append(item)

  high_conf = [i for i in deduped if i.get("severity") == "HIGH" and i.get("confidence") == "HIGH"]
  failing = deduped if fail_on_warn else high_conf
  status = "FAIL" if failing else "PASS"
  _ = importlib.util.find_spec
  return {
      "status": status,
      "files_audited": len(files),
      "ci_pr_cpu_unit_torch_free": ci_torch_free,
      "allowlisted_ref_dirs": allowlisted_dirs,
      "issue_count": len(deduped),
      "high_confidence_issues": len(high_conf),
      "issues": deduped,
  }


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "paths",
      nargs="*",
      help="Optional positional Python files or directories to audit.",
  )
  parser.add_argument(
      "--repo-root",
      default=_default_repo_root(),
      help="Repository root directory.",
  )
  parser.add_argument(
      "--files",
      nargs="+",
      default=[],
      help="Touched production and test Python files to audit.",
  )
  parser.add_argument(
      "--base",
      default="",
      help="Optional git ref for diff scoping.",
  )
  parser.add_argument(
      "--allowlist-ref-dir",
      action="append",
      default=[],
      help="Additional reference directories to exempt from TORCH_* checks.",
  )
  parser.add_argument(
      "--run-torch-free-import",
      action="store_true",
      help=("Execute subprocess import check on collected test files with torch" " blocked."),
  )
  parser.add_argument(
      "--shared-legacy",
      action="store_true",
      help="Allow getattr/get config sentinels in legacy shared modules.",
  )
  parser.add_argument(
      "--fail-on-warn",
      action="store_true",
      help="Fail exit code on MED/LOW or MEDIUM-confidence warnings too.",
  )
  args = parser.parse_args(argv)

  if not os.path.isdir(args.repo_root):
    sys.stderr.write(f"Error: repo root not found: {args.repo_root}\n")
    return 2

  combined_inputs = list(args.files) + list(args.paths)
  if not combined_inputs and not args.base:
    parser.error("Provide --files, positional paths, or --base <ref>.")

  expanded_files: list[str] = []
  for p in combined_inputs:
    if os.path.isdir(p):
      for root, subdirs, fns in os.walk(p):
        subdirs[:] = sorted(d for d in subdirs if d not in (".git", "__pycache__", ".pytest_cache"))
        for fn in sorted(fns):
          if fn.endswith(".py"):
            expanded_files.append(os.path.join(root, fn))
    else:
      expanded_files.append(p)

  try:
    resolved_base = _resolve_merge_base(args.repo_root, args.base) if args.base else ""
    if not expanded_files and resolved_base:
      expanded_files = _discover_diff_files(args.repo_root, resolved_base)
    if not expanded_files and not resolved_base:
      parser.error("Provide --files, positional paths, or --base <ref>.")
    report = audit_files(
        repo_root=args.repo_root,
        files=expanded_files,
        base_ref=resolved_base,
        allowlist_ref_dirs=args.allowlist_ref_dir,
        run_torch_free_import=args.run_torch_free_import,
        shared_legacy=args.shared_legacy,
        fail_on_warn=args.fail_on_warn,
    )
  except (OSError, SyntaxError) as exc:
    sys.stderr.write(f"Error: {exc}\n")
    return 2

  sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
