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

"""Deterministic AST and shell hygiene auditor for MaxText PR diffs.

Catches high-signal code-review blind spots across Python and shell diffs:
- UNDEFINED_LOCAL_OR_DELETED_BINDING: undefined identifier reference in a
  modified function/module or a scope where a local binding was deleted
- PRIVATE_THIRD_PARTY_SRC_ACCESS: coupling to private `._src` internals of
  external libraries (for example checkpoint or serialization internals)
- DICT_NONE_OVERWRITE_HAZARD: setting `cfg[key] = None` instead of `.pop()`,
  breaking downstream `cfg.get(key, {}).get(...)` callers with `AttributeError`
- HLO_REGEX_ON_UNCOMPILED_LOWERING: test asserts compiled-HLO syntax (`f32[..]`,
  `all-gather`) against `.lower(...).as_text()` StableHLO, so it never matches
- SILENT_DEVICE_COUNT_GUARD: test body guarded by `if jax.device_count() >= N:`
  (or `< N: return`) that passes silently instead of calling `skipTest`
- REFERENCE_ORACLE_EDITED: diff edits a pre-existing reference oracle (non-test
  code in `*reference*` test files or vendored dirs with `provenance.json`)
- ADAPTER_FAKES_CAPABILITY: runtime or conformance adapter reassigns
  `__class__`, masks model inputs/outputs, or re-splits the batch
- BLANKET_EXCEPT_VALUE_ERROR_FALLBACK: non-reraise `except ValueError:` around
  mesh entry or model/module construction that masks real validation errors
- TRACER_TYPECHECK_BYPASSES_VALIDATION: `not isinstance(..., Tracer)` guard in
  production code that silently skips validation or fallback under `jax.jit`
- BENCHMARK_CRIPPLED_XLA_FLAG: shell script disables native TPU ragged-dot
  kernel (`--xla_tpu_enable_ragged_dot_kernel=false`) without an explicit
  `# intentional-baseline:` marker

Exit code: 0 = PASS, 1 = HIGH-confidence issues found, 2 = usage or I/O error.
"""

import argparse
import ast
import builtins
from collections.abc import Sequence
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins)) | frozenset(
    {
        "__all__",
        "__annotations__",
        "__builtins__",
        "__doc__",
        "__file__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "TYPE_CHECKING",
    }
)

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_CRIPPLED_BENCH_XLA_RE = re.compile(
    r"--xla_tpu_enable_ragged_dot_kernel\s*=\s*(?:false|0)\b",
    re.IGNORECASE,
)
_INTENTIONAL_BENCH_COMMENT_RE = re.compile(
    r"#\s*(?:intentional-baseline|negative-control)\b",
    re.IGNORECASE,
)


def _dotted_name(node: ast.AST) -> str:
  """Returns dotted attribute chain or identifier if static, else ''."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    prefix = _dotted_name(node.value)
    return f"{prefix}.{node.attr}" if prefix else node.attr
  return ""


def _tail_name(node: ast.AST) -> str:
  """Returns rightmost attribute or identifier."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    return node.attr
  return ""


def _is_test_path(path: str) -> bool:
  """Returns True if `path` is a test file or sits inside a `tests/` tree."""
  base = os.path.basename(path)
  norm = path.replace("\\", "/")
  return base.endswith(("_test.py", "_tests.py")) or base.startswith("test_") or "/tests/" in f"/{norm}"


def _can_evaluate_to_none_literal(node: ast.AST) -> bool:
  """Returns True if `node` is `None`, `x or None`, or ternary `None`."""
  if isinstance(node, ast.Constant) and node.value is None:
    return True
  if (
      isinstance(node, ast.BoolOp)
      and isinstance(node.op, ast.Or)
      and node.values
      and isinstance(node.values[-1], ast.Constant)
      and node.values[-1].value is None
  ):
    return True
  if isinstance(node, ast.IfExp) and (
      (isinstance(node.body, ast.Constant) and node.body.value is None)
      or (isinstance(node.orelse, ast.Constant) and node.orelse.value is None)
  ):
    return True
  return False


def _is_config_dict_expr(node: ast.AST) -> bool:
  """Returns True if `node` names a config or metadata dictionary."""
  dotted = _dotted_name(node).lower()
  if not dotted:
    return False
  parts = dotted.split(".")
  return any(
      p in ("config", "cfg", "hf_config", "hf_config_dict", "text_config")
      or p.endswith(("_config", "_cfg", "_config_dict", "_cfg_dict"))
      for p in parts
  )


class _DiffInfo:
  """Stores per-file changed/deleted line metadata parsed from a diff."""

  def __init__(self) -> None:
    self.added_lines: dict[str, set[int]] = {}
    self.deletion_anchors: dict[str, set[int]] = {}
    self.deleted_text: list[tuple[str, str]] = []
    self.added_text: list[tuple[str, int, str]] = []
    self.new_files: set[str] = set()

  def mark_added(self, abs_path: str, lineno: int, text: str) -> None:
    self.added_lines.setdefault(abs_path, set()).add(lineno)
    self.added_text.append((abs_path, lineno, text))

  def mark_deleted(self, abs_path: str, anchor_lineno: int, text: str) -> None:
    self.deletion_anchors.setdefault(abs_path, set()).add(anchor_lineno)
    self.deleted_text.append((abs_path, text))


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


def _resolve_diff_path(raw_path: str, repo_root: str) -> str:
  """Maps a unified git diff path onto `repo_root`."""
  cleaned = raw_path.split("\t", 1)[0].strip()
  if cleaned.startswith(("a/", "b/")):
    cleaned = cleaned[2:]
  return os.path.abspath(os.path.join(repo_root, cleaned))


def _parse_patch_text(patch_text: str, repo_root: str) -> _DiffInfo:
  """Parses unified git diff output."""
  info = _DiffInfo()
  cur_file: str | None = None
  new_ln = 0
  lhs_is_null = False
  for raw in patch_text.splitlines():
    if raw.startswith("--- "):
      lhs_is_null = raw[4:].strip() == "/dev/null"
      continue
    if raw.startswith("+++ "):
      rhs = raw[4:].strip()
      if rhs == "/dev/null":
        cur_file = None
      else:
        cur_file = _resolve_diff_path(rhs, repo_root)
        if lhs_is_null:
          info.new_files.add(cur_file)
      continue
    m_hunk = _HUNK_RE.match(raw)
    if m_hunk and cur_file is not None:
      new_ln = int(m_hunk.group(3))
      continue
    if cur_file is None:
      continue
    if raw.startswith("+") and not raw.startswith("+++"):
      info.mark_added(cur_file, new_ln, raw[1:])
      new_ln += 1
    elif raw.startswith("-") and not raw.startswith("---"):
      info.mark_deleted(cur_file, max(new_ln, 1), raw[1:])
    elif raw.startswith(" "):
      new_ln += 1
  return info


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


def _collect_git_diff(repo_root: str, base_ref: str) -> tuple[_DiffInfo, list[str]]:
  """Runs `git diff -U0 <merge_base>` and returns parsed diff + changed paths."""
  info = _DiffInfo()
  if not base_ref:
    return info, []
  mb = _resolve_merge_base(repo_root, base_ref)
  git_bin = shutil.which("git")
  if not git_bin:
    raise OSError("git executable not found in PATH")
  root_abs = os.path.abspath(repo_root)
  proc = subprocess.run(
      [git_bin, "-c", "safe.directory=*", "-C", root_abs, "diff", "-U0", mb],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    detail = proc.stderr.strip() or f"exit code {proc.returncode}"
    raise OSError(f"git diff -U0 {mb} failed in {root_abs}: {detail}")
  info = _parse_patch_text(proc.stdout, root_abs)
  proc_names = subprocess.run(
      [git_bin, "-c", "safe.directory=*", "-C", root_abs, "diff", "--name-only", mb],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc_names.returncode != 0:
    detail = proc_names.stderr.strip() or f"exit code {proc_names.returncode}"
    raise OSError(f"git diff --name-only {mb} failed in {root_abs}: {detail}")
  changed_files: list[str] = []
  for rel in proc_names.stdout.splitlines():
    rel = rel.strip()
    if not rel:
      continue
    full = os.path.abspath(os.path.join(root_abs, rel))
    if os.path.isfile(full):
      changed_files.append(full)
  return info, changed_files


def _collect_target_names(node: ast.AST, out: set[str]) -> None:
  """Collects assigned identifier names from an assignment target AST."""
  if isinstance(node, ast.Name):
    out.add(node.id)
  elif isinstance(node, (ast.Tuple, ast.List, ast.Starred)):
    for child in ast.iter_child_nodes(node):
      _collect_target_names(child, out)


def _collect_pattern_names(pat: ast.pattern, out: set[str]) -> None:
  """Collects bound names inside Python `match`/`case` patterns."""
  if isinstance(pat, ast.MatchAs):
    if pat.name:
      out.add(pat.name)
    if pat.pattern is not None:
      _collect_pattern_names(pat.pattern, out)
  elif isinstance(pat, ast.MatchStar):
    if pat.name:
      out.add(pat.name)
  elif isinstance(pat, ast.MatchMapping):
    if pat.rest:
      out.add(pat.rest)
    for sub in pat.patterns:
      _collect_pattern_names(sub, out)
  else:
    for child in ast.iter_child_nodes(pat):
      if isinstance(child, ast.pattern):
        _collect_pattern_names(child, out)


def _collect_scope_bindings(
    scope_node: ast.AST,
) -> tuple[set[str], set[str], bool]:
  """Collects local bindings, explicit global/nonlocal declarations, and star-imports."""
  bound: set[str] = set()
  explicit_ext: set[str] = set()
  has_star_import = False

  if isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
    args = scope_node.args
    for a in (
        list(args.posonlyargs)
        + list(args.args)
        + list(args.kwonlyargs)
        + ([args.vararg] if args.vararg else [])
        + ([args.kwarg] if args.kwarg else [])
    ):
      bound.add(a.arg)
    for tparam in getattr(scope_node, "type_params", ()):
      tname = getattr(tparam, "name", None)
      if isinstance(tname, str):
        bound.add(tname)
  elif isinstance(scope_node, ast.ClassDef):
    for tparam in getattr(scope_node, "type_params", ()):
      tname = getattr(tparam, "name", None)
      if isinstance(tname, str):
        bound.add(tname)
  elif isinstance(scope_node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
    for gen in scope_node.generators:
      _collect_target_names(gen.target, bound)

  def _walk_body(node: ast.AST, in_comp: bool = False) -> None:
    nonlocal has_star_import
    for child in ast.iter_child_nodes(node):
      if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if not in_comp:
          bound.add(child.name)
        continue
      if isinstance(child, ast.Lambda):
        continue
      if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        # PEP 572: NamedExpr (`:=`) inside a comprehension binds in the
        # enclosing non-comprehension scope.
        _walk_body(child, in_comp=True)
        continue
      if isinstance(child, ast.NamedExpr):
        _collect_target_names(child.target, bound)
        _walk_body(child, in_comp=in_comp)
        continue
      if in_comp:
        _walk_body(child, in_comp=True)
        continue
      if isinstance(child, ast.Global):
        explicit_ext.update(child.names)
      elif isinstance(child, ast.Nonlocal):
        explicit_ext.update(child.names)
      elif isinstance(child, ast.Import):
        for alias in child.names:
          bound.add(alias.asname or alias.name.split(".", 1)[0])
      elif isinstance(child, ast.ImportFrom):
        for alias in child.names:
          if alias.name == "*":
            has_star_import = True
          else:
            bound.add(alias.asname or alias.name)
      elif isinstance(child, ast.Assign):
        for tgt in child.targets:
          _collect_target_names(tgt, bound)
      elif isinstance(child, ast.AnnAssign):
        if isinstance(child.target, ast.Name):
          bound.add(child.target.id)
        else:
          _collect_target_names(child.target, bound)
      elif isinstance(child, ast.AugAssign):
        _collect_target_names(child.target, bound)
      elif isinstance(child, (ast.For, ast.AsyncFor)):
        _collect_target_names(child.target, bound)
      elif isinstance(child, (ast.With, ast.AsyncWith)):
        for item in child.items:
          if item.optional_vars is not None:
            _collect_target_names(item.optional_vars, bound)
      elif isinstance(child, ast.ExceptHandler):
        if child.name:
          bound.add(child.name)
      elif isinstance(child, ast.match_case):
        _collect_pattern_names(child.pattern, bound)
      else:
        type_alias_cls = getattr(ast, "TypeAlias", None)
        if type_alias_cls is not None and isinstance(child, type_alias_cls):
          t_name = getattr(child, "name", None)
          if isinstance(t_name, ast.Name):
            bound.add(t_name.id)
      _walk_body(child, in_comp=False)

  if isinstance(scope_node, ast.Module):
    for stmt in scope_node.body:
      _walk_body(stmt)
    # Walk top-level statements themselves too (`_walk_body` walks children).
    temp_wrapper = ast.Module(body=list(scope_node.body), type_ignores=[])
    bound.clear()
    explicit_ext.clear()
    has_star_import = False
    _walk_body(temp_wrapper)
  elif isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
    temp_wrapper = ast.Module(body=list(scope_node.body), type_ignores=[])
    _walk_body(temp_wrapper)
  return bound - explicit_ext, explicit_ext, has_star_import


def _check_undefined_names(
    path: str,
    tree: ast.Module,
    lines: Sequence[str],
    diff_info: _DiffInfo | None,
) -> list[dict[str, Any]]:
  """Checks for undefined `ast.Load` names in touched scopes or changed lines."""
  has_future_annotations = False
  for stmt in tree.body:
    if (
        isinstance(stmt, ast.ImportFrom)
        and stmt.module == "__future__"
        and any(a.name == "annotations" for a in stmt.names)
    ):
      has_future_annotations = True
      break

  abs_p = os.path.abspath(path)
  added = diff_info.added_lines.get(abs_p, set()) if diff_info else set()
  del_anchors = diff_info.deletion_anchors.get(abs_p, set()) if diff_info else set()
  has_diff_filter = diff_info is not None and bool(diff_info.added_lines or diff_info.deletion_anchors)
  if has_diff_filter and not added and not del_anchors:
    return []

  mod_bound, _, mod_star = _collect_scope_bindings(tree)
  if mod_star:
    return []

  issues: list[dict[str, Any]] = []

  def _scope_is_touched(node: ast.AST) -> bool:
    if not has_diff_filter:
      return True
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", len(lines))
    return any(start <= ln <= end for ln in added) or any(max(1, start - 1) <= ln <= end + 1 for ln in del_anchors)

  def _visit(
      node: ast.AST,
      enclosing_chain: tuple[frozenset[str], ...],
      cur_class_bound: frozenset[str] | None,
      scope_touched: bool,
      in_annotation: bool = False,
  ) -> None:
    if in_annotation and has_future_annotations:
      return

    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
      ident = node.id
      resolved = ident in _BUILTIN_NAMES or ident in mod_bound
      if not resolved and cur_class_bound is not None and ident in cur_class_bound:
        resolved = True
      if not resolved:
        for frame in enclosing_chain:
          if ident in frame:
            resolved = True
            break
      if not resolved and scope_touched:
        ln = node.lineno
        line_txt = lines[ln - 1] if 1 <= ln <= len(lines) else ""
        if "noqa" not in line_txt and "pylint: disable=undefined-variable" not in line_txt:
          issues.append(
              {
                  "code": "UNDEFINED_LOCAL_OR_DELETED_BINDING",
                  "severity": "HIGH",
                  "file": path,
                  "line": ln,
                  "message": (
                      f"Undefined identifier `{ident}` referenced at"
                      f" {os.path.basename(path)}:{ln} (verify a local binding or"
                      " import was not accidentally deleted)."
                  ),
              }
          )
      return

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      # Decorators, default values, and signature annotations evaluate in the
      # outer scope!
      for dec in node.decorator_list:
        _visit(dec, enclosing_chain, cur_class_bound, scope_touched, False)
      for dflt in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
        _visit(dflt, enclosing_chain, cur_class_bound, scope_touched, False)
      all_args = (
          list(node.args.posonlyargs)
          + list(node.args.args)
          + list(node.args.kwonlyargs)
          + ([node.args.vararg] if node.args.vararg else [])
          + ([node.args.kwarg] if node.args.kwarg else [])
      )
      for a in all_args:
        if a.annotation is not None:
          _visit(
              a.annotation,
              enclosing_chain,
              cur_class_bound,
              scope_touched,
              True,
          )
      if node.returns is not None:
        _visit(
            node.returns,
            enclosing_chain,
            cur_class_bound,
            scope_touched,
            True,
        )
      fn_bound, _, fn_star = _collect_scope_bindings(node)
      if fn_star:
        return
      fn_touched = _scope_is_touched(node)
      new_chain = enclosing_chain + (frozenset(fn_bound),)
      for stmt in node.body:
        _visit(stmt, new_chain, None, fn_touched, False)
      return

    if isinstance(node, ast.ClassDef):
      for dec in node.decorator_list:
        _visit(dec, enclosing_chain, cur_class_bound, scope_touched, False)
      for b in node.bases:
        _visit(b, enclosing_chain, cur_class_bound, scope_touched, False)
      for kw in node.keywords:
        _visit(kw.value, enclosing_chain, cur_class_bound, scope_touched, False)
      cls_bound, _, cls_star = _collect_scope_bindings(node)
      if cls_star:
        return
      cls_touched = _scope_is_touched(node)
      # Note: Python class scopes do NOT enclose method bodies, so
      # `enclosing_chain` stays unchanged while `cur_class_bound` applies only
      # to direct class-body statements.
      for stmt in node.body:
        _visit(
            stmt,
            enclosing_chain,
            frozenset(cls_bound),
            cls_touched,
            False,
        )
      return

    if isinstance(node, ast.Lambda):
      for dflt in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
        _visit(dflt, enclosing_chain, cur_class_bound, scope_touched, False)
      lam_bound, _, _ = _collect_scope_bindings(node)
      new_chain = enclosing_chain + (frozenset(lam_bound),)
      _visit(node.body, new_chain, None, scope_touched, in_annotation)
      return

    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
      # The first generator's `iter` evaluates in the outer scope!
      comp_bound: set[str] = set()
      for idx, gen in enumerate(node.generators):
        iter_chain = enclosing_chain if idx == 0 else enclosing_chain + (frozenset(comp_bound),)
        iter_cls = cur_class_bound if idx == 0 else None
        _visit(gen.iter, iter_chain, iter_cls, scope_touched, in_annotation)
        _collect_target_names(gen.target, comp_bound)
        inner_chain = enclosing_chain + (frozenset(comp_bound),)
        for cond in gen.ifs:
          _visit(cond, inner_chain, None, scope_touched, in_annotation)
      final_chain = enclosing_chain + (frozenset(comp_bound),)
      if isinstance(node, ast.DictComp):
        _visit(node.key, final_chain, None, scope_touched, in_annotation)
        _visit(node.value, final_chain, None, scope_touched, in_annotation)
      else:
        _visit(node.elt, final_chain, None, scope_touched, in_annotation)
      return

    if isinstance(node, ast.AnnAssign):
      if isinstance(node.target, (ast.Attribute, ast.Subscript)):
        _visit(
            node.target.value,
            enclosing_chain,
            cur_class_bound,
            scope_touched,
            False,
        )
      _visit(
          node.annotation,
          enclosing_chain,
          cur_class_bound,
          scope_touched,
          True,
      )
      if node.value is not None:
        _visit(
            node.value,
            enclosing_chain,
            cur_class_bound,
            scope_touched,
            False,
        )
      return

    for child in ast.iter_child_nodes(node):
      _visit(
          child,
          enclosing_chain,
          cur_class_bound,
          scope_touched,
          in_annotation,
      )

  mod_touched = _scope_is_touched(tree)
  for stmt in tree.body:
    _visit(stmt, (), None, mod_touched, False)
  return issues


def _line_is_in_diff(path: str, lineno: int, diff_info: _DiffInfo | None) -> bool:
  """Returns True if `lineno` was added/modified or no diff scoping is active."""
  if diff_info is None or not (diff_info.added_lines or diff_info.deletion_anchors):
    return True
  abs_p = os.path.abspath(path)
  return lineno in diff_info.added_lines.get(abs_p, set())


def _scan_python_ast_rules(
    path: str,
    tree: ast.Module,
    lines: Sequence[str],
    diff_info: _DiffInfo | None,
) -> list[dict[str, Any]]:
  """Scans a Python AST for private `_src` coupling and `cfg[k] = None`."""
  issues: list[dict[str, Any]] = []
  is_test = _is_test_path(path)

  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        if "._src" in alias.name and _line_is_in_diff(path, node.lineno, diff_info):
          line_txt = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
          if "allow-private-src" not in line_txt:
            issues.append(
                {
                    "code": "PRIVATE_THIRD_PARTY_SRC_ACCESS",
                    "severity": "HIGH",
                    "file": path,
                    "line": node.lineno,
                    "message": (
                        f"Import `{alias.name}` couples to private `._src` library"
                        " internals that break across upstream upgrades; use public"
                        " APIs instead."
                    ),
                }
            )
    elif isinstance(node, ast.ImportFrom) and node.module:
      mod_and_names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
      for mpath in mod_and_names:
        if "._src" in mpath and _line_is_in_diff(path, node.lineno, diff_info):
          line_txt = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
          if "allow-private-src" not in line_txt:
            issues.append(
                {
                    "code": "PRIVATE_THIRD_PARTY_SRC_ACCESS",
                    "severity": "HIGH",
                    "file": path,
                    "line": node.lineno,
                    "message": (
                        f"Import `{mpath}` couples to private `._src` library"
                        " internals that break across upstream upgrades; use public"
                        " APIs instead."
                    ),
                }
            )

    if isinstance(node, ast.Attribute) and node.attr == "_src":
      dotted = _dotted_name(node)
      if dotted and _line_is_in_diff(path, node.lineno, diff_info):
        line_txt = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if "allow-private-src" not in line_txt:
          issues.append(
              {
                  "code": "PRIVATE_THIRD_PARTY_SRC_ACCESS",
                  "severity": "HIGH",
                  "file": path,
                  "line": node.lineno,
                  "message": (
                      f"Attribute access `{dotted}...` couples to private `._src`"
                      " internals that break across upstream upgrades; use public"
                      " APIs instead."
                  ),
              }
          )

    if (
        not is_test
        and isinstance(node, ast.Assign)
        and _can_evaluate_to_none_literal(node.value)
        and _line_is_in_diff(path, node.lineno, diff_info)
    ):
      for tgt in node.targets:
        if isinstance(tgt, ast.Subscript) and _is_config_dict_expr(tgt.value):
          issues.append(
              {
                  "code": "DICT_NONE_OVERWRITE_HAZARD",
                  "severity": "HIGH",
                  "file": path,
                  "line": node.lineno,
                  "message": (
                      f"`{ast.unparse(tgt)} = None` leaves the key present with"
                      " value `None`, causing downstream `.get(key, {}).get(...)`"
                      " callers to fail with `AttributeError`; use"
                      f" `{ast.unparse(tgt.value)}.pop(..., None)` instead."
                  ),
              }
          )

  return issues


_HLO_ONLY_SYNTAX_RE = re.compile(
    r"\b(?:pred|[su](?:4|8|16|32|64)|f16|bf16|f32|f64|f8e\w*)\["
    r"|\b(?:all-gather|all-reduce|reduce-scatter|all-to-all"
    r"|collective-permute|custom-call|dynamic-update-slice|dynamic-slice)\b"
)
_TEXT_MATCH_CALLS: frozenset[str] = frozenset(
    {
        "search",
        "match",
        "fullmatch",
        "findall",
        "finditer",
        "count",
        "assertIn",
        "assertNotIn",
        "assertRegex",
        "assertNotRegex",
    }
)
_DEVICE_COUNT_CALLS: frozenset[str] = frozenset(
    {
        "device_count",
        "local_device_count",
    }
)
_DEVICE_LIST_CALLS: frozenset[str] = frozenset({"devices", "local_devices"})
_SKIP_CALLS: frozenset[str] = frozenset(
    {
        "skipTest",
        "skip",
        "SkipTest",
        "fail",
    }
)


def _issue(code: str, path: str, line: int, message: str) -> dict[str, Any]:
  return {
      "code": code,
      "severity": "HIGH",
      "file": path,
      "line": line,
      "message": message,
  }


def _call_chain_attrs(node: ast.AST) -> list[str]:
  """Returns method names along a call/attribute chain, outermost first."""
  attrs: list[str] = []
  cur: ast.AST | None = node
  while cur is not None:
    if isinstance(cur, ast.Call):
      cur = cur.func
    elif isinstance(cur, ast.Attribute):
      attrs.append(cur.attr)
      cur = cur.value
    elif isinstance(cur, ast.Subscript):
      cur = cur.value
    else:
      cur = None
  return attrs


def _is_lowered_text_expr(node: ast.AST, lowered_names: set[str]) -> bool:
  """True for `<f>.lower(...).as_text()` or `<lowered>.as_text()` w/o compile."""
  if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "as_text"):
    return False
  attrs = _call_chain_attrs(node.func.value)
  if "compile" in attrs:
    return False
  if "lower" in attrs:
    return True
  base = node.func.value
  return isinstance(base, ast.Name) and base.id in lowered_names


def _hlo_literals(nodes: Sequence[ast.AST]) -> list[str]:
  out: list[str] = []
  for n in nodes:
    for sub in ast.walk(n):
      if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and _HLO_ONLY_SYNTAX_RE.search(sub.value):
        out.append(sub.value)
  return out


def _refs_any(node: ast.AST, names: set[str]) -> bool:
  return any(isinstance(s, ast.Name) and s.id in names for s in ast.walk(node))


def _check_hlo_text_on_lowering(path: str, fn: ast.AST, diff_info: _DiffInfo | None) -> list[dict[str, Any]]:
  """Flags compiled-HLO literals matched against uncompiled StableHLO text."""
  lowered_objs: set[str] = set()
  text_vars: dict[str, int] = {}
  for sub in ast.walk(fn):
    if isinstance(sub, ast.Assign) and len(sub.targets) == 1:
      tgt = sub.targets[0]
      if not isinstance(tgt, ast.Name):
        continue
      val = sub.value
      if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) and val.func.attr == "lower":
        lowered_objs.add(tgt.id)
  for sub in ast.walk(fn):
    if isinstance(sub, ast.Assign) and len(sub.targets) == 1:
      tgt = sub.targets[0]
      if isinstance(tgt, ast.Name) and _is_lowered_text_expr(sub.value, lowered_objs):
        text_vars[tgt.id] = sub.lineno

  issues: list[dict[str, Any]] = []

  def _uses_lowered(n: ast.AST) -> int:
    """Returns the defining line of lowered text used by `n`, else 0."""
    for s in ast.walk(n):
      if isinstance(s, ast.Name) and s.id in text_vars:
        return text_vars[s.id]
      if _is_lowered_text_expr(s, lowered_objs):
        return getattr(s, "lineno", 0)
    return 0

  for sub in ast.walk(fn):
    probes: list[ast.expr] = []
    subject: list[ast.expr] = []
    in_ops = (ast.In, getattr(ast, "Not" + "In"))
    if isinstance(sub, ast.Compare) and any(isinstance(op, in_ops) for op in sub.ops):
      probes = [sub.left]
      subject = list(sub.comparators)
    elif isinstance(sub, ast.Call) and _tail_name(sub.func) in _TEXT_MATCH_CALLS:
      probes = list(sub.args) + [k.value for k in sub.keywords]
      if isinstance(sub.func, ast.Attribute):
        subject = [sub.func.value]
      subject += probes
    else:
      continue
    def_line = 0
    for n in subject:
      def_line = def_line or _uses_lowered(n)
    if not def_line:
      continue
    literals = _hlo_literals(probes)
    if not literals:
      continue
    if _line_is_in_diff(path, sub.lineno, diff_info) or _line_is_in_diff(path, def_line, diff_info):
      issues.append(
          _issue(
              "HLO_REGEX_ON_UNCOMPILED_LOWERING",
              path,
              sub.lineno,
              f"Matches compiled-HLO syntax {literals[0]!r} against"
              " `.lower(...).as_text()`, which is pre-partitioning StableHLO"
              " (`tensor<..xf32>`, `stablehlo.all_gather`); the check can never"
              " fire. Use `.lower(...).compile().as_text()` and add a positive"
              " control that matches.",
          )
      )
  return issues


def _device_count_side(node: ast.AST) -> bool:
  """True if `node` is `len(jax.devices())` or `jax.device_count()`."""
  if not isinstance(node, ast.Call):
    return False
  if _tail_name(node.func) in _DEVICE_COUNT_CALLS:
    return True
  return (
      isinstance(node.func, ast.Name)
      and node.func.id == "len"
      and len(node.args) == 1
      and isinstance(node.args[0], ast.Call)
      and _tail_name(node.args[0].func) in _DEVICE_LIST_CALLS
  )


def _enough_devices_polarity(test: ast.AST) -> bool | None:
  """Returns True if `test` is true with enough devices, False if true with too few."""
  if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
    return None
  left, op, right = test.left, test.ops[0], test.comparators[0]
  if _device_count_side(left):
    if isinstance(op, (ast.Gt, ast.GtE)):
      return True
    if isinstance(op, (ast.Lt, ast.LtE)):
      return False
  elif _device_count_side(right):
    if isinstance(op, (ast.Lt, ast.LtE)):
      return True
    if isinstance(op, (ast.Gt, ast.GtE)):
      return False
  return None


def _has_skip(stmts: Sequence[ast.stmt]) -> bool:
  for st in stmts:
    for s in ast.walk(st):
      if isinstance(s, ast.Raise):
        return True
      if isinstance(s, ast.Call) and _tail_name(s.func) in _SKIP_CALLS:
        return True
  return False


def _check_silent_device_guard(path: str, fn: ast.AST, diff_info: _DiffInfo | None) -> list[dict[str, Any]]:
  """Flags device-count branches that skip assertions without `skipTest`."""
  issues: list[dict[str, Any]] = []
  for sub in ast.walk(fn):
    if not isinstance(sub, ast.If):
      continue
    polarity = _enough_devices_polarity(sub.test)
    if polarity is None or not _line_is_in_diff(path, sub.lineno, diff_info):
      continue
    silent = False
    if polarity and not sub.orelse:
      silent = True
    elif not polarity and not _has_skip(sub.body):
      silent = all(isinstance(s, (ast.Return, ast.Pass)) for s in sub.body)
    if silent:
      issues.append(
          _issue(
              "SILENT_DEVICE_COUNT_GUARD",
              path,
              sub.lineno,
              f"`if {ast.unparse(sub.test)}:` silently skips the multi-device"
              " assertions on hosts with fewer devices and still reports PASS;"
              " use `if <count> < N: self.skipTest(...)` so the skip is"
              " visible.",
          )
      )
  return issues


def _is_test_case_class(node: ast.ClassDef) -> bool:
  return (
      node.name.startswith("Test")
      or node.name.endswith(("Test", "Tests"))
      or any("TestCase" in ast.unparse(b) for b in node.bases)
  )


def _has_provenance(path: str, repo_root: str) -> bool:
  """Returns True if `path` or an ancestor directory contains provenance.json."""
  root = os.path.abspath(repo_root)
  cur = os.path.dirname(os.path.abspath(path))
  while cur.startswith(root):
    if os.path.isfile(os.path.join(cur, "provenance.json")):
      return True
    parent = os.path.dirname(cur)
    if parent == cur:
      break
    cur = parent
  return False


def _check_reference_oracle_edits(
    path: str,
    tree: ast.Module,
    diff_info: _DiffInfo | None,
    repo_root: str,
) -> list[dict[str, Any]]:
  """Flags diff edits to pre-existing reference implementations."""
  if diff_info is None or not (diff_info.added_lines or diff_info.deletion_anchors):
    return []
  abs_p = os.path.abspath(path)
  if abs_p in diff_info.new_files:
    return []
  touched = diff_info.added_lines.get(abs_p, set()) | (diff_info.deletion_anchors.get(abs_p, set()))
  if not touched:
    return []
  base = os.path.basename(path).lower()
  vendored = _has_provenance(path, repo_root)
  whole_file = vendored or ("reference" in base and not _is_test_path(path))
  if not whole_file and not ("reference" in base and _is_test_path(path)):
    return []
  regions: list[tuple[str, int, int]] = []
  for node in tree.body:
    if isinstance(node, ast.ClassDef) and _is_test_case_class(node):
      continue
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (node.name.startswith("test")):
      continue
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
      end = getattr(node, "end_lineno", node.lineno) or node.lineno
      regions.append((node.name, node.lineno, end))
  hits: dict[str, list[int]] = {}
  for ln in sorted(touched):
    if whole_file:
      hits.setdefault(os.path.basename(path), []).append(ln)
      continue
    for name, lo, hi in regions:
      if lo <= ln <= hi:
        hits.setdefault(name, []).append(ln)
  issues: list[dict[str, Any]] = []
  for name, lns in hits.items():
    issues.append(
        _issue(
            "REFERENCE_ORACLE_EDITED",
            path,
            lns[0],
            f"Edits reference oracle `{name}` (lines {lns[:6]}). A reference"
            " must track its upstream source; changing it to match the"
            " implementation hides real mismatches. Cite the upstream commit"
            " for the edit or move the fix into the implementation.",
        )
    )
  return issues


def _is_adapter_path(path: str) -> bool:
  base = os.path.basename(path)
  return base == "conformance_spec.py" or "adapter" in base


def _check_adapter_capability_faking(  # pylint: disable=too-many-nested-blocks
    path: str, tree: ast.Module, diff_info: _DiffInfo | None
) -> list[dict[str, Any]]:
  """Flags `__class__` swaps and input/output faking in tests and adapters."""
  issues: list[dict[str, Any]] = []
  is_test_or_adapter = _is_adapter_path(path) or _is_test_path(path)
  for sub in ast.walk(tree):
    swapped = False
    if isinstance(sub, ast.Assign):
      swapped = any(isinstance(t, ast.Attribute) and t.attr == "__class__" for t in sub.targets)
    elif (
        isinstance(sub, ast.Call)
        and _tail_name(sub.func) in ("__setattr__", "setattr")
        and any(isinstance(a, ast.Constant) and a.value == "__class__" for a in sub.args)
    ):
      swapped = True
    ln = getattr(sub, "lineno", 0)
    if swapped and _line_is_in_diff(path, ln, diff_info):
      if is_test_or_adapter:
        issues.append(
            _issue(
                "ADAPTER_FAKES_CAPABILITY",
                path,
                ln,
                "Reassigns `__class__` on a live module, so checks exercise the"
                " substituted class instead of the shipped model code."
                " Implement the capability in the model or declare the check"
                " skipped.",
            )
        )
      else:
        issues.append(
            _issue(
                "ADAPTER_FAKES_CAPABILITY",
                path,
                ln,
                "Mutates `__class__` on a live instance at runtime; dynamic"
                " prototype swapping breaks static type analysis, pytree"
                " registration, and subclass checks. Use explicit composition,"
                " wrapper modules, or constructor dispatch instead.",
            )
        )
  if not _is_adapter_path(path):
    return issues

  for fn in ast.walk(tree):
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
      continue
    if not fn.args.args:
      continue
    model_name = fn.args.args[0].arg
    batch_names = {a.arg for a in fn.args.args[1:]}
    derived_inputs: set[str] = set(batch_names)
    model_outputs: set[str] = set()
    faked_inputs: dict[str, tuple[int, str]] = {}
    for st in ast.walk(fn):
      if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
        continue
      tgt = st.targets[0]
      if not isinstance(tgt, ast.Name):
        continue
      val = st.value
      calls_model = any(
          isinstance(c, ast.Call) and _dotted_name(c.func).split(".", 1)[0] == model_name for c in ast.walk(val)
      )
      if calls_model:
        model_outputs.add(tgt.id)
        continue
      if _refs_any(val, model_outputs):
        model_outputs.add(tgt.id)
      if _refs_any(val, derived_inputs):
        derived_inputs.add(tgt.id)
        if not isinstance(val, ast.Call):
          continue
        tail = _tail_name(val.func)
        if tail == "where":
          faked_inputs[tgt.id] = (st.lineno, "masks model inputs")
        elif tail == "concatenate" and any(
            k.arg == "axis" and isinstance(k.value, ast.Constant) and k.value.value == 0 for k in val.keywords
        ):
          faked_inputs[tgt.id] = (st.lineno, "re-splits the batch")
    for st in ast.walk(fn):
      if not isinstance(st, ast.Call):
        continue
      fn_root = _dotted_name(st.func).split(".", 1)[0]
      if fn_root == model_name:
        for arg in list(st.args) + [k.value for k in st.keywords]:
          for nm in ast.walk(arg):
            if isinstance(nm, ast.Name) and nm.id in faked_inputs:
              ln, what = faked_inputs[nm.id]
              if _line_is_in_diff(path, ln, diff_info):
                issues.append(
                    _issue(
                        "ADAPTER_FAKES_CAPABILITY",
                        path,
                        ln,
                        f"Adapter {what} before calling the model, so the"
                        " check tests the wrapper instead of the model's own"
                        " segment/padding handling. Pass the batch through"
                        " unchanged or declare the check skipped.",
                    )
                )
      elif (
          _tail_name(st.func) == "where"
          and len(st.args) >= 2
          and any(_refs_any(a, model_outputs) for a in st.args[1:])
          and _line_is_in_diff(path, st.lineno, diff_info)
      ):
        issues.append(
            _issue(
                "ADAPTER_FAKES_CAPABILITY",
                path,
                st.lineno,
                "Adapter masks model outputs after the forward pass, hiding"
                " leakage the check is meant to detect. Return the model"
                " output unchanged or declare the check skipped.",
            )
        )
  return issues


_MESH_OR_MODEL_INIT_CALLS: frozenset[str] = frozenset(
    {
        "set_mesh",
        "use_abstract_mesh",
        "enter_context",
        "enter_mesh_scope",
        "create_model",
        "maybe_apply_scan",
        "maybe_quantize_model",
        "apply_remat",
    }
)
_MODEL_CLASS_SUFFIXES: tuple[str, ...] = (
    "Model",
    "Decoder",
    "Layer",
    "Attention",
    "Block",
    "Experts",
)


def _body_calls_mesh_or_model_init(stmts: Sequence[ast.stmt]) -> bool:
  """Returns True if `stmts` invoke mesh setup or NNX/model construction."""
  for st in stmts:
    for sub in ast.walk(st):
      if not isinstance(sub, ast.Call):
        continue
      tail = _tail_name(sub.func)
      dotted = _dotted_name(sub.func)
      if tail in _MESH_OR_MODEL_INIT_CALLS:
        return True
      if dotted.startswith("nnx."):
        return True
      if tail and tail[0].isupper() and tail.endswith(_MODEL_CLASS_SUFFIXES):
        return True
  return False


def _catches_value_error(exc_type: ast.expr | None) -> bool:
  """Returns True if an `except` type catches `ValueError` or `Exception`."""
  if exc_type is None:
    return True
  elts = exc_type.elts if isinstance(exc_type, ast.Tuple) else [exc_type]
  return any(_tail_name(e) in ("ValueError", "Exception", "BaseException") for e in elts)


def _check_blanket_except_value_error(path: str, tree: ast.Module, diff_info: _DiffInfo | None) -> list[dict[str, Any]]:
  """Flags non-reraising `except ValueError` around mesh/model construction."""
  norm = path.replace("\\", "/")
  if _is_test_path(path) or "/.agents/" in f"/{norm}":
    return []
  issues: list[dict[str, Any]] = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.Try):
      continue
    if not _body_calls_mesh_or_model_init(node.body):
      continue
    for handler in node.handlers:
      if not _catches_value_error(handler.type):
        continue
      if any(isinstance(s, ast.Raise) for s in ast.walk(handler)):
        continue
      handler_lines = [handler.lineno] + [getattr(s, "lineno", 0) for s in handler.body]
      if any(_line_is_in_diff(path, ln, diff_info) for ln in handler_lines if ln):
        issues.append(
            _issue(
                "BLANKET_EXCEPT_VALUE_ERROR_FALLBACK",
                path,
                handler.lineno,
                "Catches `ValueError` around mesh entry or model/module"
                " construction without re-raising, silently masking"
                " configuration, sharding, or quantization validation errors."
                " Validate preconditions explicitly instead of falling back on"
                " `ValueError`.",
            )
        )
  return issues


def _isinstance_type_nodes(node: ast.expr) -> list[ast.expr]:
  """Flattens tuple or union type arguments passed to `isinstance`."""
  if isinstance(node, ast.Tuple):
    out: list[ast.expr] = []
    for e in node.elts:
      out.extend(_isinstance_type_nodes(e))
    return out
  if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
    return _isinstance_type_nodes(node.left) + _isinstance_type_nodes(node.right)
  return [node]


_HOST_SYNC_CALLS: frozenset[str] = frozenset(
    {
        "asarray",
        "array",
        "item",
        "tolist",
    }
)


def _is_tracer_isinstance_call(node: ast.AST) -> bool:
  return (
      isinstance(node, ast.Call)
      and _tail_name(node.func) == "isinstance"
      and len(node.args) == 2
      and any(_tail_name(t) == "Tracer" for t in _isinstance_type_nodes(node.args[1]))
  )


def _expr_has_host_materialization(node: ast.AST) -> bool:
  """Returns True if `node` calls a host-synchronizing conversion."""
  for sub in ast.walk(node):
    if not isinstance(sub, ast.Call):
      continue
    dotted = _dotted_name(sub.func)
    tail = _tail_name(sub.func)
    if dotted.startswith(("np.", "numpy.")):
      return True
    if tail in _HOST_SYNC_CALLS and not dotted.startswith("jnp."):
      return True
  return False


def _find_negated_tracer_checks(node: ast.AST) -> list[ast.Call]:
  out: list[ast.Call] = []
  for sub in ast.walk(node):
    if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not):
      for c in ast.walk(sub.operand):
        if isinstance(c, ast.Call) and _is_tracer_isinstance_call(c):
          out.append(c)
  return out


def _check_tracer_typecheck(path: str, tree: ast.Module, diff_info: _DiffInfo | None) -> list[dict[str, Any]]:
  """Flags `not isinstance(..., Tracer)` guards that bypass validation/fallback."""
  if _is_test_path(path):
    return []
  issues: list[dict[str, Any]] = []
  flagged_lines: set[int] = set()
  for stmt in ast.walk(tree):
    if isinstance(stmt, ast.If):
      neg_calls = _find_negated_tracer_checks(stmt.test)
      if not neg_calls:
        continue
      guards_raise_only = not stmt.orelse and any(isinstance(s, (ast.Raise, ast.Assert)) for s in ast.walk(stmt))
      if guards_raise_only or _expr_has_host_materialization(stmt.test):
        for c in neg_calls:
          if c.lineno not in flagged_lines and _line_is_in_diff(path, c.lineno, diff_info):
            flagged_lines.add(c.lineno)
            issues.append(
                _issue(
                    "TRACER_TYPECHECK_BYPASSES_VALIDATION",
                    path,
                    c.lineno,
                    "Checks `not isinstance(..., Tracer)` to guard host-only"
                    " value inspection or validation; under `jax.jit` and"
                    " `lax.scan`, array inputs are always `Tracer` instances,"
                    " silently bypassing validation or backend fallbacks.",
                )
            )
    elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Return)):
      val = getattr(stmt, "value", None)
      if val is None or not _expr_has_host_materialization(val):
        continue
      for c in _find_negated_tracer_checks(val):
        if c.lineno not in flagged_lines and _line_is_in_diff(path, c.lineno, diff_info):
          flagged_lines.add(c.lineno)
          issues.append(
              _issue(
                  "TRACER_TYPECHECK_BYPASSES_VALIDATION",
                  path,
                  c.lineno,
                  "Checks `not isinstance(..., Tracer)` to guard host-only"
                  " value inspection or validation; under `jax.jit` and"
                  " `lax.scan`, array inputs are always `Tracer` instances,"
                  " silently bypassing validation or backend fallbacks.",
              )
          )
  return issues


def _scan_test_validity_rules(
    path: str,
    tree: ast.Module,
    diff_info: _DiffInfo | None,
    repo_root: str,
) -> list[dict[str, Any]]:
  """Runs high-signal checks for vacuous tests and runtime/adapter hazards."""
  issues: list[dict[str, Any]] = []
  is_test = _is_test_path(path)
  if is_test:
    for fn in ast.walk(tree):
      if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
      issues.extend(_check_hlo_text_on_lowering(path, fn, diff_info))
      issues.extend(_check_silent_device_guard(path, fn, diff_info))
  issues.extend(_check_reference_oracle_edits(path, tree, diff_info, repo_root))
  issues.extend(_check_adapter_capability_faking(path, tree, diff_info))
  issues.extend(_check_blanket_except_value_error(path, tree, diff_info))
  issues.extend(_check_tracer_typecheck(path, tree, diff_info))
  return issues


def _scan_shell_file(fpath: str, diff_info: _DiffInfo | None) -> list[dict[str, Any]]:
  """Scans shell scripts for benchmark XLA flags that cripple native kernels."""
  issues: list[dict[str, Any]] = []
  abs_p = os.path.abspath(fpath)
  added_lines = diff_info.added_lines.get(abs_p, set()) if diff_info is not None else None
  with open(fpath, "r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
      if added_lines is not None and lineno not in added_lines:
        continue
      if line.strip().startswith("#"):
        continue
      if _CRIPPLED_BENCH_XLA_RE.search(line) and (not _INTENTIONAL_BENCH_COMMENT_RE.search(line)):
        issues.append(
            {
                "code": "BENCHMARK_CRIPPLED_XLA_FLAG",
                "severity": "HIGH",
                "file": fpath,
                "line": lineno,
                "message": (
                    "Shell script disables native TPU ragged-dot kernel"
                    " (`--xla_tpu_enable_ragged_dot_kernel=false`), invalidating"
                    " MoE/kernel speedup comparisons unless marked"
                    " `# intentional-baseline:` and disclosed."
                ),
            }
        )
  return issues


def audit_hygiene(
    repo_root: str,
    files: Sequence[str],
    base_ref: str = "",
    diff_file: str = "",
) -> dict[str, Any]:
  """Runs all PR review hygiene checks and returns a JSON report dict."""
  diff_info: _DiffInfo | None = None
  target_files = [os.path.abspath(f) for f in files if os.path.isfile(f)]

  if diff_file:
    with open(diff_file, "r", encoding="utf-8") as f:
      diff_info = _parse_patch_text(f.read(), repo_root)
    if not target_files:
      for p in sorted(set(diff_info.added_lines) | set(diff_info.deletion_anchors)):
        if os.path.isfile(p):
          target_files.append(p)
  elif base_ref:
    diff_info, git_files = _collect_git_diff(repo_root, base_ref)
    if not target_files:
      target_files = git_files

  issues: list[dict[str, Any]] = []
  audited_count = 0
  for fpath in target_files:
    if fpath.endswith(".py"):
      audited_count += 1
      with open(fpath, "r", encoding="utf-8") as f:
        src = f.read()
      lines = src.splitlines()
      tree = ast.parse(src, filename=fpath)
      issues.extend(_check_undefined_names(fpath, tree, lines, diff_info))
      issues.extend(_scan_python_ast_rules(fpath, tree, lines, diff_info))
      issues.extend(_scan_test_validity_rules(fpath, tree, diff_info, repo_root))
    elif fpath.endswith(".sh"):
      audited_count += 1
      issues.extend(_scan_shell_file(fpath, diff_info))

  deduped: list[dict[str, Any]] = []
  seen: set[tuple[str, str, int, str]] = set()
  for item in issues:
    key = (item["code"], item["file"], int(item["line"]), item["message"])
    if key not in seen:
      seen.add(key)
      deduped.append(item)

  return {
      "status": "FAIL" if deduped else "PASS",
      "files_audited": audited_count,
      "issue_count": len(deduped),
      "issues": deduped,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Audit MaxText PR diffs for review blind-spot hazards.")
  parser.add_argument(
      "--repo-root",
      default=_default_repo_root(),
      help="Root directory of the repository being audited.",
  )
  parser.add_argument(
      "--base",
      default="",
      help="Optional git base ref to scope findings to changed lines/scopes.",
  )
  parser.add_argument(
      "--diff-file",
      default="",
      help="Optional unified git diff file for line/deletion scoping.",
  )
  parser.add_argument(
      "--files",
      nargs="*",
      default=[],
      help="Explicit Python or shell files to audit.",
  )
  args = parser.parse_args(argv)
  try:
    report = audit_hygiene(
        repo_root=args.repo_root,
        files=args.files,
        base_ref=args.base,
        diff_file=args.diff_file,
    )
  except (OSError, SyntaxError) as exc:
    sys.stderr.write(f"Error: {exc}\n")
    return 2
  sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
