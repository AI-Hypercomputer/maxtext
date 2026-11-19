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

"""Tier C: caller-impact test selection and head-then-base test runs.

  python caller_impact_tests.py select --base-wt BASE --head-wt HEAD --out sel.json
  python caller_impact_tests.py run --sel sel.json --base-wt BASE --head-wt HEAD \
      --out DIR [-j 24]

`select` builds a static import graph of src/ and tests/ in the head worktree,
finds changed modules and changed top-level symbols (base vs head source), and
selects test files that import a changed module within 2 hops or mention a
changed symbol through any import path, plus every `pytest.ini --ignore` test
that reaches a changed module (CI never runs those). It also lists the model
configs whose decoder implementation reaches a changed module, for Tier B.

`run` executes the selected test files on 4 simulated CPU devices at head, one
pytest process per file in parallel, then reruns each failing test id at base.
Only failures that are new at head are reported as regressions.
"""

import argparse
import ast
import collections
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

_PY_ROOTS = ("src", "tests")
# Modules that dispatch to every classic decoder; config schema changes alone
# (types.py, base.yml) only trigger the cheap Tier A sweep.
_DISPATCH_MODULES = (
    "maxtext.layers.decoders",
    "maxtext.layers.nnx_decoders",
    "maxtext.models.models",
)
_M3_SHARED_PREFIXES = (
    "maxtext.m3.core",
    "maxtext.m3.infra",
    "maxtext.m3.train",
)
_GENERIC_SYMBOLS = frozenset(
    {
        "__init__",
        "__call__",
        "setup",
        "main",
        "forward",
        "apply",
        "init",
        "call",
        "run",
        "get",
        "update",
        "build",
        "create",
        "test",
    }
)


def _module_name(root: str, path: str) -> str:
  rel = os.path.relpath(path, root)
  if rel.startswith("src" + os.sep):
    rel = rel[4:]
  rel = rel[:-3] if rel.endswith(".py") else rel
  if rel.endswith(os.sep + "__init__"):
    rel = rel[: -len("/__init__")]
  return rel.replace(os.sep, ".")


def _iter_py(root: str):
  for top in _PY_ROOTS:
    for dp, dns, fns in os.walk(os.path.join(root, top)):
      dns[:] = [d for d in dns if not d.startswith((".", "__pycache__"))]
      for fn in fns:
        if fn.endswith(".py"):
          yield os.path.join(dp, fn)


def build_graph(root: str) -> tuple[dict, dict]:
  """Returns (module -> path, module -> set of imported internal modules)."""
  mods: dict[str, str] = {}
  for p in _iter_py(root):
    mods[_module_name(root, p)] = p
  edges: dict[str, set[str]] = collections.defaultdict(set)
  for mod, p in mods.items():  # pylint: disable=too-many-nested-blocks
    try:
      with open(p, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
      continue
    pkg = mod if p.endswith("__init__.py") else mod.rsplit(".", 1)[0]
    for node in ast.walk(tree):
      targets: list[str] = []
      if isinstance(node, ast.Import):
        targets = [a.name for a in node.names]
      elif isinstance(node, ast.ImportFrom):
        base = node.module or ""
        if node.level:
          parts = pkg.split(".")
          up = parts[: len(parts) - node.level + 1]
          base = ".".join(up + ([base] if base else []))
        targets = [base] + [f"{base}.{a.name}" for a in node.names]
      for t in targets:
        while t:
          if t in mods:
            if t != mod:
              edges[mod].add(t)
            break
          t = t.rsplit(".", 1)[0] if "." in t else ""
  return mods, edges


def _default_python() -> str:
  """Resolves Python executable from $MAXTEXT_PILOT_PYTHON or sys.executable."""
  return os.environ.get("MAXTEXT_PILOT_PYTHON", "").strip() or sys.executable


def _git(wt: str, *args: str) -> str:
  return subprocess.run(
      ["git", "-c", "safe.directory=*", "-C", wt, *args],
      capture_output=True,
      text=True,
      check=True,
  ).stdout


def _top_level_sources(src: str) -> dict[str, str]:
  """Extracts source segments for top-level functions, classes, and methods."""
  try:
    tree = ast.parse(src)
  except SyntaxError:
    return {}
  out: dict[str, str] = {}
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      out[node.name] = ast.get_source_segment(src, node) or ""
      if isinstance(node, ast.ClassDef):
        for sub in node.body:
          if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[f"{node.name}.{sub.name}"] = ast.get_source_segment(src, sub) or ""
  return out


def _changed_symbols(base_wt: str, head_wt: str, rel: str) -> list[str]:
  def _read(wt: str) -> str:
    p = os.path.join(wt, rel)
    if not os.path.exists(p):
      return ""
    with open(p, encoding="utf-8") as f:
      return f.read()

  b, h = _top_level_sources(_read(base_wt)), _top_level_sources(_read(head_wt))
  return sorted(k for k in set(b) | set(h) if b.get(k) != h.get(k))


def _ignored_tests(wt: str) -> list[str]:
  p = os.path.join(wt, "pytest.ini")
  if not os.path.exists(p):
    return []
  with open(p, encoding="utf-8") as f:
    return re.findall(r"--ignore=(\S+)", f.read())


def _m3_registry(wt: str) -> list[str]:
  p = os.path.join(wt, "src/maxtext/m3/models/__init__.py")
  if not os.path.exists(p):
    return []
  with open(p, encoding="utf-8") as f:
    src = f.read()
  m = re.search(r"MODEL_REGISTRY[^=]*=\s*\{(.*?)\n\}", src, re.S)
  return re.findall(r'^\s*"([^"]+)"\s*:', m.group(1), re.M) if m else []


def _model_blocks(wt: str) -> dict[str, str]:
  """Returns model config name -> decoder_block."""
  d = os.path.join(wt, "src/maxtext/configs/models")
  out: dict[str, str] = {}
  for fn in sorted(os.listdir(d)):
    if fn.endswith(".yml"):
      with open(os.path.join(d, fn), encoding="utf-8") as f:
        m = re.search(r"^decoder_block:\s*\"?([\w.]+)", f.read(), re.M)
      out[fn[:-4]] = m.group(1) if m else ""
  return out


def select(base_wt: str, head_wt: str, base_ref: str) -> dict:
  """Computes changed modules/symbols, selected tests, and affected models."""
  changed = [ln for ln in _git(head_wt, "diff", "--name-only", f"{base_ref}..HEAD").splitlines() if ln.strip()]
  mods, edges = build_graph(head_wt)
  rev: dict[str, set[str]] = collections.defaultdict(set)
  for a, bs in edges.items():
    for b in bs:
      rev[b].add(a)
  path_to_mod = {os.path.relpath(p, head_wt): m for m, p in mods.items()}
  changed_mods = sorted(path_to_mod[c] for c in changed if c in path_to_mod)
  config_changed = [c for c in changed if c.endswith((".yml", ".yaml"))]
  symbols: dict[str, list[str]] = {}
  for c in changed:
    if c.endswith(".py") and c.startswith("src/"):
      symbols[c] = _changed_symbols(base_wt, head_wt, c)

  # BFS distances from changed modules over reverse edges.
  dist: dict[str, int] = {m: 0 for m in changed_mods}
  q = collections.deque(changed_mods)
  while q:
    cur = q.popleft()
    for nxt in rev.get(cur, ()):
      if nxt not in dist:
        dist[nxt] = dist[cur] + 1
        q.append(nxt)

  sym_names = {s.split(".")[-1] for v in symbols.values() for s in v} - _GENERIC_SYMBOLS
  sym_re = (
      re.compile(r"\b(" + "|".join(sorted(re.escape(s) for s in sym_names if len(s) > 3)) + r")\b") if sym_names else None
  )
  ignored = set(_ignored_tests(head_wt))
  tests: dict[str, dict] = {}
  for m, d in dist.items():
    p = os.path.relpath(mods[m], head_wt)
    if not p.startswith("tests/") or not p.endswith(("_test.py", "_tests.py")):
      continue
    with open(mods[m], encoding="utf-8") as f:
      text = f.read()
    hit = sym_re.search(text).group(1) if sym_re and sym_re.search(text) else ""
    is_ignored = any(p == i or p.startswith(i.rstrip("/") + "/") for i in ignored)
    if d <= 2 or hit or is_ignored or p in changed:
      tests[p] = {
          "distance": d,
          "symbol_hit": hit,
          "ci_ignored": is_ignored,
          "changed_in_pr": p in changed,
      }

  blocks = _model_blocks(head_wt)
  all_models = any(m in _DISPATCH_MODULES for m in changed_mods)
  reached = {m.rsplit(".", 1)[-1]: d for m, d in dist.items() if m.startswith("maxtext.models.")}
  affected: dict[str, str] = {}
  model_dist: dict[str, int] = {}
  for name, blk in blocks.items():
    ds = [d for r, d in reached.items() if r and (r == blk or r.startswith(blk) or blk.startswith(r))]
    if ds or all_models or any(name in c for c in config_changed):
      affected[name] = blk
      model_dist[name] = min(ds) if ds else 9
  # m3 models: shared m3 code reaches every registry model; a model dir
  # reaches the registry keys of that family.
  m3_keys = _m3_registry(head_wt)
  m3_shared = any(m.startswith(_M3_SHARED_PREFIXES) for m in changed_mods)
  for key in m3_keys:
    norm = re.sub(r"[^a-z0-9]", "", key)
    fams = {
        re.sub(r"[^a-z0-9]", "", m.split(".")[3])
        for m in dist
        if m.startswith("maxtext.m3.models.") and len(m.split(".")) > 4
    }
    hit = [f for f in fams if f and norm.startswith(f)]
    if m3_shared or hit:
      affected[f"m3__{key}"] = "m3:" + (max(hit, key=len) if hit else "shared")
      model_dist[f"m3__{key}"] = 0 if hit else 1
  return {
      "model_distance": model_dist,
      "base_ref": base_ref,
      "changed_files": changed,
      "changed_modules": changed_mods,
      "changed_symbols": symbols,
      "all_models_affected": all_models,
      "affected_models": affected,
      "tests": dict(sorted(tests.items())),
  }


def _pytest(
    wt: str,
    rel: str,
    xml: str,
    node_ids: list[str] | None = None,
    *,
    timeout: int = 600,
    py_exe: str | None = None,
) -> dict:
  env = dict(os.environ)
  env["PYTHONPATH"] = f"{os.path.join(wt, 'src')}{os.pathsep}{wt}"
  env["JAX_PLATFORMS"] = "cpu"
  env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
  env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "0")
  targets = node_ids or [rel]
  cmd = [
      py_exe or _default_python(),
      "-m",
      "pytest",
      "-q",
      "-p",
      "no:cacheprovider",
      "-o",
      "addopts=-rf --import-mode=importlib",
      "-m",
      "not tpu_only and not gpu_only and not tpu_backend",
      f"--junitxml={xml}",
      *targets,
  ]
  t0 = time.time()
  try:
    p = subprocess.run(
        cmd,
        cwd=wt,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    rc, tail = p.returncode, p.stdout[-1500:] + p.stderr[-800:]
  except subprocess.TimeoutExpired:
    rc, tail = -9, f"timeout {timeout}s"
  failed: list[str] = []
  counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
  if os.path.exists(xml):
    root = ET.parse(xml).getroot()
    for tc in root.iter("testcase"):
      counts["tests"] += 1
      bad = tc.find("failure") is not None or tc.find("error") is not None
      if tc.find("skipped") is not None:
        counts["skipped"] += 1
      if bad:
        cls = tc.get("classname", "").rsplit(".", 1)[-1]
        failed.append(
            f"{rel}::{cls}::{tc.get('name')}" if cls and not rel.endswith(cls + ".py") else f"{rel}::{tc.get('name')}"
        )
  return {
      "rc": rc,
      "seconds": round(time.time() - t0),
      "failed": failed,
      "counts": counts,
      "tail": tail if rc not in (0, 5) else "",
  }


def run(
    sel: dict,
    base_wt: str,
    head_wt: str,
    out: str,
    jobs: int,
    *,
    py_exe: str = "",
    fallback_ci_py: str = "",
) -> dict:
  """Runs selected tests at head, reruns failures at base, classifies them."""
  os.makedirs(out, exist_ok=True)
  files = list(sel["tests"])
  primary_py = py_exe or _default_python()

  def head_job(rel: str):
    xml = os.path.join(out, "head_" + rel.replace("/", "__") + ".xml")
    # Tests the PR itself adds or edits get a longer budget.
    budget = 1800 if sel["tests"][rel].get("changed_in_pr") else 600
    return rel, _pytest(head_wt, rel, xml, timeout=budget, py_exe=primary_py)

  t0 = time.time()
  head: dict[str, dict] = {}
  with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
    for rel, res in ex.map(head_job, files):
      head[rel] = res
      print(
          f"head {rel}: rc={res['rc']} {res['counts']} {res['seconds']}s",
          flush=True,
      )

  def base_job(rel: str):
    res = head[rel]
    xml = os.path.join(out, "base_" + rel.replace("/", "__") + ".xml")
    if not os.path.exists(os.path.join(base_wt, rel)):
      return rel, {"missing_at_base": True}
    ids = res["failed"] or None
    return rel, _pytest(base_wt, rel, xml, node_ids=ids, py_exe=primary_py)

  timeouts = [r for r, v in head.items() if v["rc"] == -9]
  rerun = [r for r, v in head.items() if r not in timeouts and (v["failed"] or v["rc"] not in (0, 5))]
  base: dict[str, dict] = {}
  with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
    for rel, res in ex.map(base_job, rerun):
      base[rel] = res
  regressions, preexisting, new_tests_failing = [], [], []
  venv_jax012_only: list[str] = []
  raw_ci_py = fallback_ci_py or os.environ.get("MAXTEXT_CI_PYTHON", "").strip()
  ci_py = os.path.expanduser(raw_ci_py) if raw_ci_py else ""
  for rel in rerun:
    b = base[rel]
    if b.get("missing_at_base"):
      fails = head[rel]["failed"] or [f"{rel} (rc={head[rel]['rc']})"]
      if head[rel]["failed"] and ci_py and os.path.exists(ci_py) and ci_py != primary_py:
        ci_xml = os.path.join(out, "ci_" + rel.replace("/", "__") + ".xml")
        ci_res = _pytest(head_wt, rel, ci_xml, node_ids=head[rel]["failed"], py_exe=ci_py)
        still_bad = set(ci_res["failed"])
        for t in head[rel]["failed"]:
          if t in still_bad or ci_res["rc"] not in (0, 5):
            new_tests_failing.append(t)
          else:
            venv_jax012_only.append(t)
      else:
        new_tests_failing += fails
      continue
    bfail = set(b["failed"])
    for t in head[rel]["failed"] or [f"{rel} (collection rc={head[rel]['rc']})"]:
      (preexisting if t in bfail or (b["rc"] not in (0, 5) and not b["failed"]) else regressions).append(t)
  total_head_cases = sum(v["counts"]["tests"] for v in head.values())
  collection_failed = bool(files) and total_head_cases == 0 and all(v["rc"] not in (0, 5) for v in head.values())
  return {
      "note": (
          "Failures in new test files are not proof of a bug: compare"
          " their first error line with preexisting failures (the local"
          " venv may differ from CI)."
      ),
      "files_run": len(files),
      "collection_or_startup_failed": collection_failed,
      "timeouts_inconclusive": timeouts,
      "wall_seconds": round(time.time() - t0),
      "regressions": regressions,
      "new_tests_failing": new_tests_failing,
      "venv_jax012_only_failures": venv_jax012_only,
      "preexisting_failures": preexisting,
      "head": head,
      "base": base,
  }


def _check_python_deps(py_exe: str, modules: tuple[str, ...]) -> str | None:
  """Verifies `py_exe` can import `modules`; returns error text or None."""
  if not modules:
    return None
  stmt = "import " + ", ".join(modules)
  try:
    proc = subprocess.run(
        [py_exe, "-c", stmt],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    return (
        f"Python interpreter {py_exe!r} failed preflight ({stmt!r}): {exc}. "
        "Pass --python /path/to/maxtext-venv/bin/python or set MAXTEXT_PILOT_PYTHON."
    )
  if proc.returncode != 0:
    detail = (proc.stderr or proc.stdout).strip()[-400:]
    return (
        f"Python interpreter {py_exe!r} cannot import {', '.join(modules)} "
        "(pass --python /path/to/maxtext-venv/bin/python or set "
        f"MAXTEXT_PILOT_PYTHON): {detail}"
    )
  return None


_REAL_PYTEST_HELPER = _pytest


def main() -> int:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  s = sub.add_parser("select")
  s.add_argument("--base-wt", required=True)
  s.add_argument("--head-wt", required=True)
  s.add_argument("--base-ref", required=True)
  s.add_argument("--out", required=True)
  r = sub.add_parser("run")
  r.add_argument("--sel", required=True)
  r.add_argument("--base-wt", required=True)
  r.add_argument("--head-wt", required=True)
  r.add_argument("--out", required=True)
  r.add_argument("-j", type=int, default=32)
  r.add_argument("--max-files", type=int, default=35)
  r.add_argument("--python", default=_default_python())
  r.add_argument(
      "--fallback-ci-python",
      default=os.environ.get("MAXTEXT_CI_PYTHON", "").strip(),
      help="Optional secondary Python venv for JAX version skew recheck.",
  )
  a = ap.parse_args()
  if a.cmd == "select":
    res = select(a.base_wt, a.head_wt, a.base_ref)
    with open(a.out, "w", encoding="utf-8") as f:
      json.dump(res, f, indent=1)
    t = res["tests"]
    print(
        f"changed modules {len(res['changed_modules'])}, tests {len(t)}"
        f" (ci-ignored {sum(v['ci_ignored'] for v in t.values())}),"
        f" affected models {len(res['affected_models'])}"
        f" (all={res['all_models_affected']})"
    )
    return 0
  with open(a.sel, encoding="utf-8") as f:
    sel = json.load(f)
  if sel.get("tests") and _pytest is _REAL_PYTEST_HELPER:
    dep_err = _check_python_deps(a.python, ("pytest",))
    if dep_err:
      sys.stderr.write(f"Error: {dep_err}\n")
      return 2
  if len(sel["tests"]) > a.max_files:
    ranked = sorted(
        sel["tests"].items(),
        key=lambda kv: (
            not kv[1]["changed_in_pr"],
            not kv[1]["ci_ignored"],
            kv[1]["distance"],
            not kv[1]["symbol_hit"],
        ),
    )
    sel["tests"] = dict(ranked[: a.max_files])
    print(f"capped to {a.max_files} files", flush=True)
  res = run(
      sel,
      a.base_wt,
      a.head_wt,
      a.out,
      a.j,
      py_exe=a.python,
      fallback_ci_py=a.fallback_ci_python,
  )
  with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
  print(
      json.dumps(
          {
              k: res[k]
              for k in (
                  "files_run",
                  "collection_or_startup_failed",
                  "wall_seconds",
                  "regressions",
                  "new_tests_failing",
                  "timeouts_inconclusive",
              )
          },
          indent=1,
      )[:4000]
  )
  print(f"preexisting failures: {len(res['preexisting_failures'])}")
  if res["collection_or_startup_failed"]:
    sys.stderr.write(
        "Error: 0 test cases executed across selected test files while pytest "
        "exited non-zero (collection or environment startup failure).\n"
    )
    return 2
  return 1 if res["regressions"] or res["new_tests_failing"] else 0


if __name__ == "__main__":
  sys.exit(main())
