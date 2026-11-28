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

"""Runs Tiers A-C for one change (head vs its parent) and writes a summary.

  python pilot.py --label NAME --head SHA --base REF [--tiers ABC] [--b-max 12]

Worktrees go to /tmp/rt_<label>_{base,head[,main]}. Tier A base dumps are cached
per interpreter/library tag and SHA in /tmp/rtA_cache_ci/<env_tag>/<sha>. On
stacked PRs where `base` differs from `merge-base(origin/main, head)`, Tiers A
and B also check `origin/main` so restoring `origin/main` is not misclassified
as a cross-model regression.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))


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


def _default_python() -> str:
  """Resolves Python executable from $MAXTEXT_PILOT_PYTHON or sys.executable."""
  return os.environ.get("MAXTEXT_PILOT_PYTHON", "").strip() or sys.executable


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


def _python_env_tag(py_exe: str) -> str:
  """Returns a 12-char cache tag salted by interpreter path and jax/flax versions."""
  probe = (
      "import os, sys\n"
      "parts = [os.path.realpath(sys.executable), sys.version.split()[0]]\n"
      "for m in ('jax', 'flax'):\n"
      "  try:\n"
      "    mod = __import__(m)\n"
      '    parts.append(f\'{m}={getattr(mod, "__version__", "?")}\')\n'
      "  except (ImportError, AttributeError):\n"
      "    parts.append(f'{m}=none')\n"
      "print('|'.join(parts))\n"
  )
  raw = os.path.abspath(py_exe)
  try:
    proc = subprocess.run(
        [py_exe, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
      raw = proc.stdout.strip()
  except (OSError, subprocess.TimeoutExpired):
    pass
  return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _load_sibling_module(mod_name: str) -> Any:
  """Loads a sibling script module from HERE without mutating sys.path."""
  path = os.path.join(HERE, f"{mod_name}.py")
  spec = importlib.util.spec_from_file_location(mod_name, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load sibling module {mod_name} from {path}")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


_REMAT_SWEEP_SUFFIXES = (
    "remat_none",
    "remat_none__unscanned",
    "remat_full__unscanned",
    "remat_minimal_offloaded",
    "remat_qkv_proj_offloaded",
    "remat_save_dot_with_context_except_mlp",
)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
  return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


def worktree(repo_root: str, sha: str, path: str) -> None:
  """Creates or reuses a detached temporary git worktree at `sha`."""
  if os.path.isdir(path):
    cur = sh(["git", "-c", "safe.directory=*", "-C", path, "rev-parse", "HEAD"]).stdout.strip()
    if cur == sha:
      return
    sh(
        [
            "git",
            "-c",
            "safe.directory=*",
            "-C",
            repo_root,
            "worktree",
            "remove",
            "--force",
            path,
        ]
    )
  r = sh(
      [
          "git",
          "-c",
          "safe.directory=*",
          "-C",
          repo_root,
          "worktree",
          "add",
          "-q",
          "--detach",
          path,
          sha,
      ]
  )
  if r.returncode:
    raise RuntimeError(r.stderr)


def named_models(sel: dict, all_names: list[str]) -> list[str]:
  """Models whose own config or model directory the change edits."""
  names: set[str] = set()
  for c in sel["changed_files"]:
    m = re.match(r"src/maxtext/configs/models/(.+)\.yml$", c)
    if m:
      names.add(m.group(1))
    m = re.match(r"src/maxtext/m3/models/(\w+)/", c)
    if m:
      fam = re.sub(r"[^a-z0-9]", "", m.group(1))
      names |= {n for n in all_names if n.startswith("m3__") and re.sub(r"[^a-z0-9]", "", n[4:]).startswith(fam)}
  return sorted(names)


_SMALL = r"tiny|test|mini|0\.6b|1\.7b|-2b|-4b|8b"


def _prefer_small(ms: list[str]) -> list[str]:
  small = [m for m in ms if re.search(_SMALL, m)]
  return sorted(small) or sorted(ms)


def pick_tier_b(sel: dict, named: list[str], cap: int, all_names: list[str]) -> list[str]:
  """Named models (small variants first), one per block, plus policy sweeps."""
  dist = sel.get("model_distance", {})
  by_block: dict[str, list[str]] = {}
  for m, blk in sel["affected_models"].items():
    by_block.setdefault(blk, []).append(m)
  picks: list[str] = []
  for n in named:
    blk = sel["affected_models"].get(n)
    fam = [m for m in by_block.get(blk, [n]) if m in named] if blk else [n]
    best = _prefer_small(fam)[0]
    if best not in picks:
      picks.append(best)
  ranked = sorted(
      by_block.items(),
      key=lambda kv: (min(dist.get(m, 9) for m in kv[1]), kv[0]),
  )
  for _, ms in ranked:
    if any(p in ms for p in picks):
      continue
    picks.append(_prefer_small(ms)[0])
  picks = picks[:cap]

  m3_small = _prefer_small([n for n in all_names if n.startswith("m3__")])
  m3_rep = next(
      (p for p in picks if p.startswith("m3__")),
      m3_small[0] if m3_small else "",
  )
  if any(c.endswith(("kernels.py", "attentions.py")) for c in sel["changed_files"]):
    dp_target = m3_rep or (picks[0] if picks else "")
    if dp_target and f"{dp_target}@dot_product" not in picks:
      picks.append(f"{dp_target}@dot_product")
  remat_target = m3_rep or (named[0] if named else "")
  if remat_target and any(c.endswith(("remat.py", "nnx_decoders.py")) for c in sel["changed_files"]):
    for sfx in _REMAT_SWEEP_SUFFIXES:
      cand = f"{remat_target}@{sfx}"
      if cand not in picks:
        picks.append(cand)
  if m3_rep and any(
      c.endswith(
          (
              "offload.py",
              "model_creation_utils.py",
              "gradient_accumulation.py",
          )
      )
      for c in sel["changed_files"]
  ):
    for sfx in ("offload_param", "offload_ga2"):
      cand = f"{m3_rep}@{sfx}"
      if cand not in picks:
        picks.append(cand)
  if m3_rep and any(
      c.endswith(
          (
              "moe.py",
              "ragged_sort.py",
              "ragged_gather.py",
              "ragged_gather_reduce_v2.py",
          )
      )
      for c in sel["changed_files"]
  ):
    for cand in (
        f"{m3_rep}@moe_ragged_sort",
        f"{m3_rep}@moe_dropless_ragged_sort",
    ):
      if cand not in picks:
        picks.append(cand)
    qwen_moe = "m3__qwen3-30b-a3b"
    if m3_rep != qwen_moe and qwen_moe in all_names and f"{qwen_moe}@moe_dropless_ragged_sort" not in picks:
      picks.append(f"{qwen_moe}@moe_dropless_ragged_sort")
  return picks


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--label", required=True)
  ap.add_argument("--head", required=True)
  ap.add_argument("--base", required=True)
  ap.add_argument("--tiers", default="ABC")
  ap.add_argument("--b-max", type=int, default=12)
  ap.add_argument("--c-max", type=int, default=35)
  ap.add_argument(
      "--repo-root",
      default=_default_repo_root(),
      help="Git repository/worktree root (default: auto-detected).",
  )
  ap.add_argument(
      "--python",
      default=_default_python(),
      help="Python interpreter for Tier A-C subprocesses.",
  )
  ap.add_argument(
      "--out-dir",
      default="",
      help="Optional output directory override (default: /tmp/rt_out/<label>).",
  )
  a = ap.parse_args()
  repo_root = os.path.abspath(a.repo_root)
  py_exe = a.python
  out = a.out_dir or f"/tmp/rt_out/{a.label}"
  os.makedirs(out, exist_ok=True)
  req_mods: list[str] = []
  if "A" in a.tiers or "B" in a.tiers:
    req_mods.append("jax")
  if "C" in a.tiers:
    req_mods.append("pytest")
  if req_mods:
    dep_err = _check_python_deps(py_exe, tuple(req_mods))
    if dep_err:
      sys.stderr.write(f"Error: {dep_err}\n")
      return 2
  git_pfx = ["git", "-c", "safe.directory=*", "-C", repo_root]
  r_head = sh([*git_pfx, "rev-parse", "--verify", f"{a.head}^{{commit}}"])
  if r_head.returncode != 0 or not r_head.stdout.strip():
    sys.stderr.write(f"Error: cannot resolve --head {a.head!r}: {r_head.stderr.strip()}\n")
    return 2
  head = r_head.stdout.strip()
  r_base = sh([*git_pfx, "merge-base", a.base, head])
  if r_base.returncode != 0 or not r_base.stdout.strip():
    sys.stderr.write(f"Error: cannot resolve merge-base({a.base!r}, {head[:12]}): {r_base.stderr.strip()}\n")
    return 2
  base = r_base.stdout.strip()
  r_main = sh([*git_pfx, "merge-base", "origin/main", head])
  main_mb = r_main.stdout.strip() if r_main.returncode == 0 else ""
  has_stacked_main = bool(main_mb and main_mb != base)
  bwt, hwt = f"/tmp/rt_{a.label}_base", f"/tmp/rt_{a.label}_head"
  mwt = f"/tmp/rt_{a.label}_main" if has_stacked_main else ""
  worktree(repo_root, base, bwt)
  worktree(repo_root, head, hwt)
  if mwt:
    worktree(repo_root, main_mb, mwt)
  timing: dict[str, int] = {}
  summary: dict = {"label": a.label, "head": head, "base": base}
  if has_stacked_main:
    summary["main_merge_base"] = main_mb

  t0 = time.time()
  sel_path = f"{out}/sel.json"
  r = sh(
      [
          py_exe,
          f"{HERE}/caller_impact_tests.py",
          "select",
          "--base-wt",
          bwt,
          "--head-wt",
          hwt,
          "--base-ref",
          base,
          "--out",
          sel_path,
      ]
  )
  print(r.stdout, r.stderr[-800:], flush=True)
  if r.returncode != 0 or not os.path.isfile(sel_path):
    sys.stderr.write(f"Error: caller_impact_tests.py select failed (rc={r.returncode}): {r.stderr[-800:]}\n")
    return 2
  with open(sel_path, encoding="utf-8") as f:
    sel = json.load(f)
  abstract_model_diff = _load_sibling_module("abstract_model_diff")
  all_names = abstract_model_diff._list_models(hwt)  # pylint: disable=protected-access
  named = named_models(sel, all_names)
  summary["named"] = named
  summary["affected_models"] = len(sel["affected_models"])
  summary["selected_tests"] = len(sel["tests"])
  timing["select"] = round(time.time() - t0)
  named_re = "^(" + "|".join(re.escape(n) for n in named) + ")$" if named else ""

  def tier_a():
    t0 = time.time()
    env_tag = _python_env_tag(py_exe)
    cache_root = f"/tmp/rtA_cache_ci/{env_tag}"
    bdir, hdir = f"{cache_root}/{base}", f"{cache_root}/{head}"
    mdir = f"{cache_root}/{main_mb}" if mwt else ""
    pairs = [(bwt, bdir), (hwt, hdir)] + ([(mwt, mdir)] if mwt else [])
    dump_fail_rc = 0
    for wt, d in pairs:
      r = sh(
          [
              py_exe,
              f"{HERE}/abstract_model_diff.py",
              "dump",
              "--worktree",
              wt,
              "--out",
              d,
              "-j",
              "48",
              "--python",
              py_exe,
          ]
      )
      if r.returncode != 0 and dump_fail_rc == 0:
        dump_fail_rc = r.returncode
      print(r.stdout[-300:], r.stderr[-500:], flush=True)
    cmp_cmd = [
        py_exe,
        f"{HERE}/abstract_model_diff.py",
        "compare",
        bdir,
        hdir,
        "--named",
        named_re,
        "--md",
        f"{out}/A.md",
    ]
    if mdir:
      cmp_cmd += ["--main-base", mdir]
    r = sh(cmp_cmd)
    rc = r.returncode or dump_fail_rc
    head_line = r.stdout.splitlines()[0] if r.stdout else r.stderr[-300:]
    if dump_fail_rc and r.returncode == 0:
      head_line = f"{head_line} | dump failed (rc={dump_fail_rc})"
    if any(
        c.endswith(("sharding.py", "m3/core/moe.py", "m3/models/__init__.py"))
        or re.search(r"m3/models/.+/modeling_[^/]+\.py$", c)
        for c in sel["changed_files"]
    ):
      rs = sh(
          [
              py_exe,
              f"{HERE}/abstract_model_diff.py",
              "sharding-sweep",
              "--worktree",
              hwt,
              "--md",
              f"{out}/A.md",
              "--changed-files",
              ",".join(sel["changed_files"]),
              "--python",
              py_exe,
          ]
      )
      if rs.returncode != 0:
        rc = rc or rs.returncode
        s_line = rs.stdout.splitlines()[0] if rs.stdout else "sharding-sweep failed"
        head_line = f"{head_line} | {s_line}"
    summary["tierA_rc"] = rc
    summary["tierA_head"] = head_line
    timing["A"] = round(time.time() - t0)

  def tier_b():
    t0 = time.time()
    models = pick_tier_b(sel, named, a.b_max, all_names)
    summary["tierB_models"] = models
    base_ok = False
    batch_fail_rc = 0
    for wt, d, ref in (
        (bwt, f"{out}/B_base", ""),
        (hwt, f"{out}/B_head", f"{out}/B_base"),
    ):
      cmd = [
          py_exe,
          f"{HERE}/mini_aot_fingerprint.py",
          "batch",
          "--worktree",
          wt,
          "--models",
          ",".join(models),
          "--out",
          d,
          "-j",
          "16",
          "--python",
          py_exe,
      ]
      use_ref = bool(ref) and base_ok
      r = sh(cmd + (["--layers-from", ref] if use_ref else []))
      if models and r.returncode != 0 and batch_fail_rc == 0:
        batch_fail_rc = r.returncode
      if not ref:
        base_ok = r.returncode == 0
        bs_path = f"{d}/batch_summary.json"
        if os.path.exists(bs_path):
          with open(bs_path, encoding="utf-8") as f:
            base_ok = base_ok and json.load(f).get("ok", 0) > 0
      print(r.stdout[-1500:], r.stderr[-500:], flush=True)
    r = sh(
        [
            py_exe,
            f"{HERE}/mini_aot_fingerprint.py",
            "compare",
            f"{out}/B_base",
            f"{out}/B_head",
            "--named",
            named_re,
        ]
    )
    if r.returncode != 0 and mwt:
      # Compile only the differing models against merge-base(origin/main, head)
      # so stacked PRs that restore origin/main are recognized cleanly.
      mini_aot_fingerprint = _load_sibling_module("mini_aot_fingerprint")
      diff_models = list(mini_aot_fingerprint.compare(f"{out}/B_base", f"{out}/B_head", named_re))
      if diff_models:
        sh(
            [
                py_exe,
                f"{HERE}/mini_aot_fingerprint.py",
                "batch",
                "--worktree",
                mwt,
                "--models",
                ",".join(diff_models),
                "--out",
                f"{out}/B_main",
                "-j",
                "16",
                "--python",
                py_exe,
                "--layers-from",
                f"{out}/B_head",
            ]
        )
        r = sh(
            [
                py_exe,
                f"{HERE}/mini_aot_fingerprint.py",
                "compare",
                f"{out}/B_base",
                f"{out}/B_head",
                "--named",
                named_re,
                "--main-base",
                f"{out}/B_main",
            ]
        )
    with open(f"{out}/B.md", "w", encoding="utf-8") as f:
      f.write(r.stdout)
    final_b_rc = r.returncode or (batch_fail_rc if models else 0)
    b_head_line = r.stdout.splitlines()[0] if r.stdout else ""
    if batch_fail_rc and models and r.returncode == 0:
      b_head_line = f"{b_head_line} | batch failed (rc={batch_fail_rc})".strip(" |")
    summary["tierB_rc"] = final_b_rc
    summary["tierB_head"] = b_head_line
    timing["B"] = round(time.time() - t0)

  def tier_c():
    t0 = time.time()
    r = sh(
        [
            py_exe,
            f"{HERE}/caller_impact_tests.py",
            "run",
            "--sel",
            sel_path,
            "--base-wt",
            bwt,
            "--head-wt",
            hwt,
            "--out",
            f"{out}/C",
            "-j",
            "32",
            "--max-files",
            str(a.c_max),
            "--python",
            py_exe,
        ]
    )
    with open(f"{out}/C.md", "w", encoding="utf-8") as f:
      f.write(r.stdout[-8000:])
    summary["tierC_rc"] = r.returncode
    timing["C"] = round(time.time() - t0)

  t_all = time.time()
  jobs = {"A": tier_a, "B": tier_b, "C": tier_c}
  with concurrent.futures.ThreadPoolExecutor(3) as ex:
    futs = [ex.submit(fn) for k, fn in jobs.items() if k in a.tiers]
    for fut in futs:
      fut.result()
  timing["wall"] = round(time.time() - t_all)

  summary["timing_s"] = timing
  with open(f"{out}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1)
  print(json.dumps(summary, indent=1))
  any_failed = any(summary.get(f"tier{k}_rc", 0) != 0 for k in a.tiers if k in jobs)
  return 1 if any_failed else 0


if __name__ == "__main__":
  sys.exit(main())
