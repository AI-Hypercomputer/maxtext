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

r"""Diffs reference modeling code and configs between a model and its closest family model.

Phase 0 tool (run by the designer, output outside the client):

  python3 family_diff.py --old <files...> --new <files...> \
      [--old-prefix OldModel --new-prefix NewModel] \
      [--old-config a.json --new-config b.json] --out report.json --md report.md \
      [--reuse-claims claims.json]

Symbols are compared by docstring-free AST, so formatting and docs do not count
as changes. Hugging Face code prefixes class names per model; `--old-prefix` and
`--new-prefix` replace each prefix with a common placeholder in symbol names,
names and attributes before comparing, so an identical class under a new prefix
is UNCHANGED. Symbols are keyed by the normalised name; reports show both
original names. Exit codes: 0 ok, 3 a claimed-reused symbol (new-side name) is
neither UNCHANGED nor RENAMED.
If the model has no family model, do not run this tool; the reuse table says
"none" and every component is new.
"""

import argparse
import ast
import copy
import difflib
import json
import pathlib
import sys
from typing import Any

STATUS_ORDER = ("CHANGED", "NEW", "RENAMED", "REMOVED", "UNCHANGED")
REUSABLE = ("UNCHANGED", "RENAMED")
PREFIX_PLACEHOLDER = "PREFIX_"


class _PrefixNormalizer(ast.NodeTransformer):
  """Replaces a leading model prefix in definition names, names and attributes."""

  def __init__(self, prefix: str):
    self.prefix = prefix

  def norm(self, name: str) -> str:
    """Replaces leading model prefix with canonical placeholder."""
    return PREFIX_PLACEHOLDER + name[len(self.prefix) :] if name.startswith(self.prefix) else name

  def generic_visit(self, node: ast.AST) -> ast.AST:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
      node.name = self.norm(node.name)
    elif isinstance(node, ast.Name):
      node.id = self.norm(node.id)
    elif isinstance(node, ast.Attribute):
      node.attr = self.norm(node.attr)
    return super().generic_visit(node)


def _strip_docstrings(node: ast.AST) -> ast.AST:
  """Returns a deep copy of `node` with module/class/function docstrings removed."""
  node = copy.deepcopy(node)
  for n in ast.walk(node):
    body = getattr(n, "body", None)
    if (
        isinstance(body, list)
        and body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
      n.body = body[1:] or [ast.Pass()]
  return node


def _canonical_dump(node: ast.AST) -> str:
  """AST dump without docstrings, positions, or the symbol's own name."""
  node = _strip_docstrings(node)
  node.name = "_canonical"
  return ast.dump(node, include_attributes=False)


def extract_symbols(files: list[str], prefix: str | None) -> dict[str, dict[str, Any]]:
  """Returns top-level classes and functions keyed by (normalised) name; raises on duplicates."""
  symbols: dict[str, dict[str, Any]] = {}
  for path in files:
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"), filename=path)
    originals = {id(n): n.name for n in tree.body if hasattr(n, "name")}
    if prefix is not None:
      tree = _PrefixNormalizer(prefix).visit(tree)
    for node in tree.body:
      if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
      if node.name in symbols:
        raise ValueError(f"Symbol {node.name!r} defined in both {symbols[node.name]['file']} and {path}")
      methods = (
          {m.name: m for m in getattr(node, "body", []) if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
          if isinstance(node, ast.ClassDef)
          else {}
      )
      symbols[node.name] = {
          "kind": "class" if isinstance(node, ast.ClassDef) else "function",
          "original_name": originals[id(node)],
          "file": path,
          "line": node.lineno,
          "node": node,
          "methods": methods,
      }
  return symbols


def _entry(name: str, status: str, old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
  return {
      "name": name,
      "status": status,
      "kind": (new or old)["kind"],
      "old_name": old["original_name"] if old else None,
      "new_name": new["original_name"] if new else None,
      "old_location": f"{old['file']}:{old['line']}" if old else None,
      "new_location": f"{new['file']}:{new['line']}" if new else None,
  }


def diff_symbols(old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
  """Classifies every symbol as UNCHANGED, CHANGED, NEW, REMOVED or RENAMED."""
  results = {}
  for name in sorted(old.keys() & new.keys()):
    o, n = old[name], new[name]
    same = o["kind"] == n["kind"] and _canonical_dump(o["node"]) == _canonical_dump(n["node"])
    entry = _entry(name, "UNCHANGED" if same else "CHANGED", o, n)
    if not same:
      entry["changed_methods"] = sorted(
          m
          for m in o["methods"].keys() | n["methods"].keys()
          if m not in o["methods"]
          or m not in n["methods"]
          or _canonical_dump(o["methods"][m]) != _canonical_dump(n["methods"][m])
      )
      entry["diff"] = "\n".join(
          difflib.unified_diff(
              ast.unparse(_strip_docstrings(o["node"])).splitlines(),
              ast.unparse(_strip_docstrings(n["node"])).splitlines(),
              fromfile=f"old/{name}",
              tofile=f"new/{name}",
              lineterm="",
          )
      )
    results[name] = entry
  only_old = {k: v for k, v in old.items() if k not in new}
  only_new = {k: v for k, v in new.items() if k not in old}
  for old_name, o in sorted(only_old.items()):
    twin = next(
        (
            k
            for k, n in sorted(only_new.items())
            if n["kind"] == o["kind"] and _canonical_dump(n["node"]) == _canonical_dump(o["node"])
        ),
        None,
    )
    if twin is None:
      results[old_name] = _entry(old_name, "REMOVED", o, None)
    else:
      results[twin] = _entry(twin, "RENAMED", o, only_new.pop(twin))
  for name, n in only_new.items():
    results[name] = _entry(name, "NEW", None, n)
  return results


def diff_configs(old_path: str, new_path: str) -> dict[str, dict[str, Any]]:
  """Computes added, removed, and changed keys between two config JSON files."""
  old_cfg = json.loads(pathlib.Path(old_path).read_text(encoding="utf-8"))
  new_cfg = json.loads(pathlib.Path(new_path).read_text(encoding="utf-8"))
  return {
      "added": {k: new_cfg[k] for k in new_cfg.keys() - old_cfg.keys()},
      "removed": {k: old_cfg[k] for k in old_cfg.keys() - new_cfg.keys()},
      "changed": {
          k: {"old": old_cfg[k], "new": new_cfg[k]} for k in old_cfg.keys() & new_cfg.keys() if old_cfg[k] != new_cfg[k]
      },
  }


def to_markdown(symbols: dict[str, dict[str, Any]], config_diff: dict[str, Any] | str) -> str:
  """Renders symbol and config diffs as a Markdown report."""
  lines = [
      "# Family diff",
      "",
      "| Symbol | Kind | Status | Old | New | Changed methods |",
      "| :--- | :--- | :--- | :--- | :--- | :--- |",
  ]
  for s in sorted(symbols.values(), key=lambda s: (STATUS_ORDER.index(s["status"]), s["name"])):
    names = [n for n in (s["old_name"], s["new_name"]) if n is not None]
    name = " -> ".join(f"`{n}`" for n in dict.fromkeys(names))
    methods = ", ".join(s["changed_methods"]) if "changed_methods" in s else ""
    lines.append(
        f"| {name} | {s['kind']} | {s['status']} | {s['old_location'] or '-'} | {s['new_location'] or '-'} | {methods} |"
    )
  lines += ["", "## Config diff", ""]
  if isinstance(config_diff, str):
    lines.append(config_diff)
  else:
    lines += ["| Key | Change | Old | New |", "| :--- | :--- | :--- | :--- |"]
    lines += [f"| `{k}` | ADDED | - | `{json.dumps(v)}` |" for k, v in sorted(config_diff["added"].items())]
    lines += [f"| `{k}` | REMOVED | `{json.dumps(v)}` | - |" for k, v in sorted(config_diff["removed"].items())]
    lines += [
        f"| `{k}` | CHANGED | `{json.dumps(v['old'])}` | `{json.dumps(v['new'])}` |"
        for k, v in sorted(config_diff["changed"].items())
    ]
  return "\n".join(lines) + "\n"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--old", nargs="+", required=True, help="Family model's reference modeling files")
  parser.add_argument("--new", nargs="+", required=True, help="New model's reference modeling files")
  parser.add_argument("--old-prefix", help="Family model's class-name prefix (give both prefixes or neither)")
  parser.add_argument("--new-prefix", help="New model's class-name prefix (give both prefixes or neither)")
  parser.add_argument("--old-config", help="Family model config.json (give both configs or neither)")
  parser.add_argument("--new-config", help="New model config.json (give both configs or neither)")
  parser.add_argument("--out", required=True, help="Output report.json")
  parser.add_argument("--md", required=True, help="Output report.md")
  parser.add_argument("--reuse-claims", help="JSON file with the list of symbols the design claims are reused")
  args = parser.parse_args()
  if (args.old_config is None) != (args.new_config is None):
    parser.error("--old-config and --new-config must be given together")
  if (args.old_prefix is None) != (args.new_prefix is None):
    parser.error("--old-prefix and --new-prefix must be given together")

  symbols = diff_symbols(extract_symbols(args.old, args.old_prefix), extract_symbols(args.new, args.new_prefix))
  config_diff = (
      diff_configs(args.old_config, args.new_config)
      if args.old_config
      else "NOT_RUN: no configs given (stated explicitly in the reuse table)"
  )
  summary = {s: sum(1 for v in symbols.values() if v["status"] == s) for s in STATUS_ORDER}
  pathlib.Path(args.out).write_text(
      json.dumps({"summary": summary, "symbols": symbols, "config_diff": config_diff}, indent=2),
      encoding="utf-8",
  )
  pathlib.Path(args.md).write_text(to_markdown(symbols, config_diff), encoding="utf-8")
  print("FAMILY_DIFF " + " ".join(f"{k.lower()}={v}" for k, v in summary.items()))

  if args.reuse_claims:
    claims = json.loads(pathlib.Path(args.reuse_claims).read_text(encoding="utf-8"))
    by_new_name = {s["new_name"]: s for s in symbols.values() if s["new_name"] is not None}
    bad = [
        f"{c} ({by_new_name[c]['status'] if c in by_new_name else 'NOT_FOUND'})"
        for c in claims
        if c not in by_new_name or by_new_name[c]["status"] not in REUSABLE
    ]
    if bad:
      print("REUSE_CLAIMS_FAILED: " + ", ".join(bad), file=sys.stderr)
      return 3
  return 0


if __name__ == "__main__":
  sys.exit(main())
