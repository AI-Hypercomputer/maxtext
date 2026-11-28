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

"""Tier A: abstract cross-model diff and m3 sharding-axis sweep (no compile).

Builds every model config abstractly (`nnx.eval_shape`) with scan_layers on and
off, at a base and a head worktree, and reports per-model differences in the
parameter tree, dtypes, logical axes, resolved PartitionSpecs, Muon dimension
numbers, and build success. On stacked PRs, `--main-base` suppresses false
regressions when `head` restores `origin/main` behavior modified by an unmerged
parent PR. When `sharding.py` is modified, `sharding-sweep` checks every axis in
`MESH_AXIS_NAMES` and `shard_mode=explicit` across representative m3 models.

  python abstract_model_diff.py dump --worktree WT --out DIR [--models a,b] [-j
  32]
  python abstract_model_diff.py compare BASE_DIR HEAD_DIR [--named REGEX]
  [--main-base DIR] [--md OUT]
  python abstract_model_diff.py sharding-sweep --worktree WT [--md OUT]

Exit code of `compare` / `sharding-sweep`: 0 = pass, 1 = issues found, 2 =
error.
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import traceback
import types
from typing import Any

_HERE = os.path.abspath(__file__)
_COMMON_OVERRIDES = (
    "enable_single_controller=True",
    "override_model_config=true",
    "per_device_batch_size=1",
    "max_target_length=128",
    "dataset_type=synthetic",
    "run_name=abstract_diff",
    "base_output_directory=/tmp/abstract_diff_bo",
    "skip_jax_distributed_system=True",
)
_MESH_OVERRIDES = (
    "ici_fsdp_parallelism=2",
    "ici_tensor_parallelism=2",
    "ici_expert_parallelism=2",
)
_LOGICAL_PROBES = (
    ("batch", ("batch", "length", "embed")),
    ("length", ("batch", "length", "embed")),
    ("norm_length", ("batch", "norm_length", "embed")),
    ("heads", ("batch", "length", "heads", "head_dim")),
    ("kv_heads", ("batch", "kv_length", "kv_heads", "head_dim")),
    ("vocab", ("batch", "length", "vocab")),
    ("mlp", ("batch", "length", "mlp")),
)


def _is_valid_cached_dump(path: str) -> bool:
  """Returns True iff `path` is a valid dump JSON without subprocess crash markers."""
  if not os.path.isfile(path):
    return False
  try:
    with open(path, encoding="utf-8") as f:
      rec = json.load(f)
  except (OSError, json.JSONDecodeError):
    return False
  if not isinstance(rec, dict) or "crash" in rec:
    return False
  variants = rec.get("variants")
  return isinstance(variants, dict) and bool(variants)


def _has_ok_variant(path: str) -> bool:
  """Returns True iff `path` has at least one variant with `status == 'ok'`."""
  if not _is_valid_cached_dump(path):
    return False
  try:
    with open(path, encoding="utf-8") as f:
      rec = json.load(f)
  except (OSError, json.JSONDecodeError):
    return False
  variants = rec.get("variants", {})
  return any(isinstance(v, dict) and v.get("status") == "ok" for v in variants.values())


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


def _dump_many(models: list[str], outdir: str, extra: list[str]) -> None:
  for m in models:
    dst = os.path.join(outdir, f"{m}.json")
    if not _is_valid_cached_dump(dst):
      _dump_one(m, dst, extra)


def _extract_params(abs_model) -> dict:
  """Extracts parameter metadata and Muon dimension numbers from an abstract model."""
  from flax import nnx  # pylint: disable=import-outside-toplevel

  params = {}
  state = nnx.state(abs_model)
  flat = nnx.to_flat_state(state) if hasattr(nnx, "to_flat_state") else state.flat_state()
  for path, leaf in flat:
    val = leaf.get_value() if hasattr(leaf, "get_value") else leaf.value
    meta = leaf.get_metadata() if hasattr(leaf, "get_metadata") else {}
    names = {k: str(v) for k, v in sorted(meta.items()) if "sharding" in k or k in ("param_scan_axis",)}
    shd = getattr(val, "sharding", None)
    shape = tuple(getattr(val, "shape", ()))
    try:
      shard_shape = list(shd.shard_shape(shape)) if shd else list(shape)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
      shard_shape = list(shape)
    params["/".join(str(p) for p in path)] = {
        "type": type(leaf).__name__,
        "shape": list(shape),
        "dtype": str(getattr(val, "dtype", "")),
        "spec": str(getattr(shd, "spec", None)),
        "shard_shape": shard_shape,
        "logical": names,
    }
  return params


def _dump_one(model: str, out: str, extra: list[str]) -> None:
  """Runs inside the worktree's PYTHONPATH; writes one JSON for both scans."""
  result: dict = {"model": model, "variants": {}}
  import jax  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

  try:
    from maxtext.utils import muon_utils  # pylint: disable=import-outside-toplevel
  except ImportError:
    muon_utils = None
  base_yml = os.path.join(os.environ["PYTHONPATH"].split(":")[0], "maxtext/configs/base.yml")
  model_name, m3_flags = model, []
  if model.startswith("m3__"):
    model_name, m3_flags = model[len("m3__") :], ["use_m3_model=true"]
  for scan in (True, False):
    key = f"scan={scan}"
    t0 = time.time()
    var: dict = {}
    try:
      argv = [None, base_yml, f"model_name={model_name}", f"scan_layers={scan}"]
      argv += list(_COMMON_OVERRIDES) + m3_flags + extra
      attempts = (
          ("fsdp2_tensor2_expert2", list(_MESH_OVERRIDES)),
          ("default", []),
          ("default+dot_product", ["attention=dot_product"]),
      )
      first_err = None
      for label, ov in attempts:
        try:
          config = pyconfig.initialize(argv + ov)
          mesh = maxtext_utils.get_mesh_from_config(config)
          _, abs_model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh)
          var["mesh"] = label
          break
        except Exception as e:  # pylint: disable=broad-except
          first_err = first_err or e
      else:
        raise first_err
      var["params"] = _extract_params(abs_model)
      if muon_utils is not None and hasattr(muon_utils, "get_muon_weight_dimension_numbers"):
        try:
          mdn = muon_utils.get_muon_weight_dimension_numbers(abs_model, config)
          leaves = jax.tree_util.tree_flatten_with_path(
              mdn,
              is_leaf=lambda x: x is None or "DimensionNumbers" in type(x).__name__,
          )[0]
          var["muon"] = {jax.tree_util.keystr(p): repr(v) for p, v in leaves}
        except Exception as e:  # pylint: disable=broad-except
          var["muon_error"] = f"{type(e).__name__}: {str(e)[:300]}"
      var["status"] = "ok"
    except Exception as e:  # pylint: disable=broad-except
      var["status"] = "error"
      var["error"] = f"{type(e).__name__}: {str(e)[:500]}"
      var["trace"] = traceback.format_exc()[-1500:]
    var["seconds"] = round(time.time() - t0, 1)
    result["variants"][key] = var
  with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f)


def _list_models(wt: str) -> list[str]:
  """Classic model ymls plus m3 MODEL_REGISTRY keys (prefixed `m3__`)."""
  d = os.path.join(wt, "src/maxtext/configs/models")
  names = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".yml"))
  reg = os.path.join(wt, "src/maxtext/m3/models/__init__.py")
  if os.path.exists(reg):
    with open(reg, encoding="utf-8") as f:
      src = f.read()
    m = re.search(r"MODEL_REGISTRY[^=]*=\s*\{(.*?)\n\}", src, re.S)
    if m:
      names += [f"m3__{k}" for k in re.findall(r'^\s*"([^"]+)"\s*:', m.group(1), re.M)]
  return names


def _default_python() -> str:
  """Resolves Python executable from $MAXTEXT_PILOT_PYTHON or sys.executable."""
  return os.environ.get("MAXTEXT_PILOT_PYTHON", "").strip() or sys.executable


def _dump(
    wt: str,
    out: str,
    models: list[str],
    jobs: int,
    extra: list[str],
    *,
    py_exe: str = "",
):
  """Dumps abstract parameter metadata across `models` in parallel chunks."""
  os.makedirs(out, exist_ok=True)
  env = dict(os.environ)
  env["PYTHONPATH"] = os.path.join(os.path.abspath(wt), "src")
  env["JAX_PLATFORMS"] = "cpu"
  env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
  env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "0")
  env.pop("TPU_NAME", None)
  resolved_py = py_exe or _default_python()

  todo = [m for m in models if not _is_valid_cached_dump(os.path.join(out, f"{m}.json"))]
  chunks = [todo[i : i + 3] for i in range(0, len(todo), 3)]

  def run(chunk: list[str]) -> str:
    cmd = [
        resolved_py,
        _HERE,
        "dump-many",
        "--models",
        ",".join(chunk),
        "--outdir",
        out,
    ]
    cmd += ["--extra", *extra] if extra else []
    try:
      p = subprocess.run(
          cmd,
          env=env,
          cwd=wt,
          capture_output=True,
          text=True,
          timeout=1800,
          check=False,
      )
      err = p.stderr[-1500:]
    except subprocess.TimeoutExpired:
      err = "timeout 1800s"
    for m in chunk:
      dst = os.path.join(out, f"{m}.json")
      if not _is_valid_cached_dump(dst):
        with open(dst, "w", encoding="utf-8") as f:
          json.dump({"model": m, "crash": err}, f)
    return f"{chunk}: done"

  t0 = time.time()
  with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
    for msg in ex.map(run, chunks):
      print(msg, flush=True)
  print(f"dumped {len(models)} models in {time.time() - t0:.0f}s")


def _keystr_to_path(k: str) -> str:
  return "/".join(re.findall(r"\['?([^'\]]+)'?\]", k))


def muon_scan_folds(var: dict) -> set[str]:
  """Params whose Muon reduction/output axes include the layer scan axis."""
  out: set[str] = set()
  for k, v in var.get("muon", {}).items():
    prm = var.get("params", {}).get(_keystr_to_path(k))
    if not prm or "param_scan_axis" not in prm["logical"] or not prm["shape"]:
      continue
    nd = len(prm["shape"])
    ax = int(prm["logical"]["param_scan_axis"]) % nd
    if ax in {int(x) % nd for x in re.findall(r"-?\d+", v)}:
      out.add(f"{_keystr_to_path(k)} {prm['shape']} {v}")
  return out


def tensor_asymmetries(var: dict, min_elems: int = 1 << 20) -> set[str]:
  """Large sibling weights where some use the `tensor` mesh axis and some don't."""
  by_parent: dict[str, list[tuple[str, dict]]] = {}
  for path, prm in var.get("params", {}).items():
    n = 1
    for d in prm["shape"]:
      n *= d
    if prm["type"] != "Param" or n < min_elems:
      continue
    by_parent.setdefault(path.rsplit("/", 1)[0], []).append((path, prm))
  out: set[str] = set()
  for parent, items in by_parent.items():
    uses = {p: "'tensor'" in prm["spec"] for p, prm in items}
    if len(items) > 1 and any(uses.values()) and not all(uses.values()):
      missing = sorted(p.rsplit("/", 1)[1] for p, u in uses.items() if not u)
      present = sorted(p.rsplit("/", 1)[1] for p, u in uses.items() if u)
      out.add(f"{parent}: {missing} not tensor-sharded, {present} are")
  return out


def _load(d: str) -> dict:
  if not d or not os.path.isdir(d):
    return {}
  out = {}
  for f in sorted(os.listdir(d)):
    if f.endswith(".json"):
      with open(os.path.join(d, f), encoding="utf-8") as fh:
        out[f[:-5]] = json.load(fh)
  return out


def _variant_status(rec: dict, key: str) -> str:
  if "crash" in rec:
    return "crash"
  return rec.get("variants", {}).get(key, {}).get("status", "missing")


def _per_dev(ps: dict) -> int:
  tot = 0
  for x in ps.values():
    n = 1
    for d in x.get("shard_shape", x["shape"]):
      n *= d
    tot += n
  return tot


def _diff_records(b_rec: dict | None, h_rec: dict | None) -> list[str]:
  """Diffs one model record (`b_rec` vs `h_rec`), returning finding strings."""
  if b_rec is None and h_rec is None:
    return []
  if b_rec is None:
    assert h_rec is not None
    found = ["NEW_MODEL_CONFIG"]
    for key in ("scan=True", "scan=False"):
      hv = h_rec.get("variants", {}).get(key, {})
      if hv.get("status") != "ok" and "crash" not in h_rec:
        found.append(f"NEW_MODEL_BUILD_FAILS [{key}]: {hv.get('error', '')[:300]}")
      folds = sorted(muon_scan_folds(hv))
      if folds:
        found.append(f"MUON_SCAN_AXIS_FOLDED [{key}] {len(folds)} params, e.g." f" {folds[:2]}")
      asym = sorted(tensor_asymmetries(hv))
      if asym:
        found.append(f"TENSOR_AXIS_ASYMMETRY [{key}] {len(asym)}: {asym[:2]}")
    return found
  if h_rec is None:
    return ["MODEL_CONFIG_REMOVED"]

  found = []
  for key in ("scan=True", "scan=False"):
    bs, hs = _variant_status(b_rec, key), _variant_status(h_rec, key)
    if bs == "ok" and hs != "ok":
      err = h_rec.get("crash") or h_rec["variants"][key].get("error")
      found.append(f"BUILD_REGRESSION [{key}]: base builds, head fails: {err}")
      continue
    if bs != "ok" and hs == "ok":
      found.append(f"BUILD_FIXED [{key}]")
      continue
    if bs != "ok":
      continue
    bv, hv = b_rec["variants"][key], h_rec["variants"][key]
    bp, hp = bv["params"], hv["params"]
    added = sorted(set(hp) - set(bp))
    removed = sorted(set(bp) - set(hp))
    if added:
      found.append(f"PARAMS_ADDED [{key}] {len(added)}: {added[:4]}")
    if removed:
      found.append(f"PARAMS_REMOVED [{key}] {len(removed)}: {removed[:4]}")
    for p in sorted(set(bp) & set(hp)):
      for fld in ("shape", "dtype", "spec", "shard_shape", "logical", "type"):
        if bp[p][fld] != hp[p][fld]:
          found.append(f"{fld.upper()}_CHANGED [{key}] {p}: {bp[p][fld]} -> {hp[p][fld]}")
    bd, hd = _per_dev(bp), _per_dev(hp)
    if bd and hd > bd * 1.05:
      found.append(
          f"PER_DEVICE_PARAMS_UP [{key}] {bd:,} -> {hd:,} elements per device"
          f" on the 2x2x2 mesh (+{100 * (hd - bd) / bd:.0f}%): weights lost"
          " sharding (replicated on some mesh axis)"
      )
    new_folds = sorted(muon_scan_folds(hv) - muon_scan_folds(bv))
    if new_folds:
      found.append(
          f"MUON_SCAN_AXIS_FOLDED [{key}] {len(new_folds)} params, e.g."
          f" {new_folds[:2]}: Muon orthogonalizes across stacked layers"
      )
    new_asym = sorted(tensor_asymmetries(hv) - tensor_asymmetries(bv))
    if new_asym:
      found.append(
          f"TENSOR_AXIS_ASYMMETRY [{key}] {len(new_asym)}, e.g. {new_asym[:2]}:"
          " these weights are replicated across tensor ranks"
      )
    bm, hm = bv.get("muon", {}), hv.get("muon", {})
    for p in sorted(set(bm) | set(hm)):
      if bm.get(p) != hm.get(p):
        found.append(f"MUON_DIMS_CHANGED [{key}] {p}: {bm.get(p)} -> {hm.get(p)}")
    if ("muon_error" in hv) != ("muon_error" in bv):
      found.append(f"MUON_ERROR_CHANGED [{key}]: {hv.get('muon_error')}")
  if found:
    prio = ("BUILD_REGRESSION", "MUON_SCAN", "TENSOR_AXIS", "PER_DEVICE")
    found = [x for x in found if x.startswith(prio)] + [x for x in found if not x.startswith(prio)]
  return found


def compare(base_dir: str, head_dir: str, named: str, main_base_dir: str = "") -> dict:
  """Returns {model: [finding, ...]} for every model with a difference."""
  base, head = _load(base_dir), _load(head_dir)
  main_base = _load(main_base_dir) if main_base_dir else {}
  named_re = re.compile(named) if named else None
  report: dict = {}
  for m in sorted(set(base) | set(head)):
    found = _diff_records(base.get(m), head.get(m))
    if not found:
      continue
    is_named = bool(named_re and named_re.search(m))
    restores_main = bool(main_base and m in main_base and m in head and not _diff_records(main_base.get(m), head.get(m)))
    prefix: list[str] = []
    if is_named:
      prefix.append("(named by PR)")
    elif restores_main:
      prefix.append("(restores origin/main)")
    report[m] = prefix + found
  return report


def _select_sweep_reps(model_registry: dict[str, Any], changed_files: str = "") -> list[str]:
  """Selects representative m3 models plus any families in changed_files."""
  by_fam: dict[str, list[str]] = {}
  for k in sorted(model_registry):
    fam = k.split("-", 1)[0]
    by_fam.setdefault(fam, []).append(k)
  reps: list[str] = []
  for _, ks in sorted(by_fam.items()):
    small = [k for k in ks if re.search(r"tiny|micro|mini|0\.6b|1\.7b|2b|4b", k)]
    reps.append((small or ks)[0])
  reps = reps[:4]
  if not changed_files:
    return reps
  dir_slugs = []
  for c in changed_files.split(","):
    m_dir = re.search(r"m3/models/([^/]+)/", c.strip())
    if m_dir:
      dir_slugs.append(re.sub(r"[^a-z0-9]", "", m_dir.group(1).lower()))
  if not dir_slugs:
    return reps
  for fam, ks in sorted(by_fam.items()):
    fam_slug = re.sub(r"[^a-z0-9]", "", fam.lower())
    matched_ks = [
        k
        for k in ks
        if any(re.sub(r"[^a-z0-9]", "", k.lower()).startswith(ds) or ds.startswith(fam_slug) for ds in dir_slugs)
    ]
    if matched_ks:
      small = [k for k in matched_ks if re.search(r"tiny|micro|mini|0\.6b|1\.7b|2b|4b", k)]
      pick = (small or matched_ks)[0]
      if pick not in reps:
        reps.append(pick)
  return reps


def _probe_out_of_mesh_2d_fsdp(m3_sharding: Any) -> list[str]:
  """Checks sharding_from_config outside an active mesh preserves 2D-FSDP."""
  if not (
      hasattr(m3_sharding, "sharding_from_config")
      and hasattr(m3_sharding, "TensorType")
      and hasattr(m3_sharding.TensorType, "WEIGHT")
  ):
    return []
  try:
    cfg_2d = types.SimpleNamespace(
        compound_fsdp_ep=True,
        ici_fsdp_transpose_parallelism=2,
        dcn_fsdp_transpose_parallelism=1,
    )
    shd_2d = m3_sharding.sharding_from_config(cfg_2d)
    spec_2d = str(
        shd_2d(
            "experts.gate_up_proj",
            ("experts", "expert_embed", "expert_mlp"),
            m3_sharding.TensorType.WEIGHT,
        )
    )
    if "fsdp_transpose" not in spec_2d:
      return ["OUT_OF_MESH_2D_FSDP_DROPPED: sharding_from_config outside mesh" f" omitted fsdp_transpose (got {spec_2d})"]
  except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    return []
  return []


def _probe_explicit_forward(  # pylint: disable=too-many-positional-arguments
    mname: str,
    base_argv: list[str],
    ov: list[str],
    token_pspec: tuple[str | None, str | None],
    code: str,
    pyconfig: Any,
    maxtext_utils: Any,
    model_creation_utils: Any,
    m3_models: Any,
    jax: Any,
    jnp: Any,
    nnx: Any,
) -> list[str]:
  """Runs an abstract forward pass under shard_mode=explicit."""
  try:
    cfg_ex = pyconfig.initialize(base_argv + ["shard_mode=explicit", *ov])
    mesh_ex = maxtext_utils.get_mesh_from_config(cfg_ex)
    _, _ = model_creation_utils.create_nnx_abstract_model(cfg_ex, mesh=mesh_ex)
  except (
      AttributeError,
      KeyError,
      NotImplementedError,
      RuntimeError,
      TypeError,
      ValueError,
  ):
    return []
  try:
    from jax.sharding import NamedSharding  # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from jax.sharding import PartitionSpec as P  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    tok_shd = NamedSharding(mesh_ex, P(*token_pspec))

    def _run_ex(c=cfg_ex, m=mesh_ex, ts=tok_shd):
      mod = m3_models.create_model(c, m, rngs=nnx.Rngs(0))
      tok = jax.lax.with_sharding_constraint(jnp.zeros((4, 128), jnp.int32), ts)
      return mod(tok)

    with jax.set_mesh(mesh_ex):
      nnx.eval_shape(_run_ex)
  except Exception as e:  # pylint: disable=broad-except
    return [
        f"{code} [m3__{mname}]: create_model accepted shard_mode=explicit"
        f" ({', '.join(ov)}), but forward failed:"
        f" {type(e).__name__}: {str(e)[:240]}"
    ]
  return []


def _probe_tp_grad_parity(  # pylint: disable=too-many-positional-arguments
    mname: str,
    base_yml: str,
    pyconfig,
    maxtext_utils,
    m3_models,
    jax,
    jnp,
    nnx,
) -> list[str]:
  """Checks 1-step float32 CPU TP=1 vs TP=2 per-leaf gradient L2 norm parity."""
  cfg1 = cfg2 = mesh1 = mesh2 = None
  for nlayers in (2, 4):
    common = [
        None,
        base_yml,
        f"model_name={mname}",
        "use_m3_model=true",
        "scan_layers=False",
        f"base_num_decoder_layers={nlayers}",
        "attention=dot_product",
        "max_target_length=32",
        "enable_checkpointing=false",
        "dtype=float32",
        "weight_dtype=float32",
        *(x for x in _COMMON_OVERRIDES if not x.startswith("max_target_length=")),
    ]
    try:
      cfg1 = pyconfig.initialize(common + ["ici_fsdp_parallelism=8", "ici_tensor_parallelism=1"])
      cfg2 = pyconfig.initialize(common + ["ici_fsdp_parallelism=4", "ici_tensor_parallelism=2"])
      mesh1 = maxtext_utils.get_mesh_from_config(cfg1)
      mesh2 = maxtext_utils.get_mesh_from_config(cfg2)
      break
    except (
        AttributeError,
        KeyError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as e:
      low = str(e).lower()
      cfg1 = cfg2 = mesh1 = mesh2 = None
      if "multiple of" not in low and "divisible" not in low:
        return []
  if cfg1 is None or cfg2 is None or mesh1 is None or mesh2 is None:
    return []

  vocab_hi = max(2, min(128, int(getattr(cfg1, "vocab_size", 128) or 128)))
  toks = jax.random.randint(jax.random.PRNGKey(0), (8, 32), 1, vocab_hi)

  def _grads_on_mesh(cfg, mesh):
    with jax.set_mesh(mesh):
      mod = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(42))
      gdef, state = nnx.split(mod)
      flat = nnx.to_flat_state(state) if hasattr(nnx, "to_flat_state") else state.flat_state()
      new_leaves = []
      for i, (path, var) in enumerate(flat):
        val = var.get_value() if hasattr(var, "get_value") else var.value
        if jnp.issubdtype(val.dtype, jnp.floating):
          arr = jax.random.normal(jax.random.PRNGKey(1000 + i), val.shape, dtype=val.dtype) * 0.02
          shd = getattr(val, "sharding", None)
          if shd is not None:
            arr = jax.device_put(arr, shd)
          var = var.replace(arr) if hasattr(var, "replace") else var
        new_leaves.append((path, var))
      state = nnx.from_flat_state(new_leaves) if hasattr(nnx, "from_flat_state") else state
      mod = nnx.merge(gdef, state)

      @nnx.jit
      def _step(m, x):
        def _loss_fn(model):
          out = model(x)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.mean(jnp.square(logits.astype(jnp.float32)))

        return nnx.value_and_grad(_loss_fn)(m)

      _, grads = _step(mod, toks)
      g_flat = nnx.to_flat_state(grads) if hasattr(nnx, "to_flat_state") else grads.flat_state()
      g_norms: dict[str, float] = {}
      for path, gv in g_flat:
        arr = gv.get_value() if hasattr(gv, "get_value") else gv.value
        if arr is not None and jnp.issubdtype(arr.dtype, jnp.floating):
          g_norms["/".join(str(p) for p in path)] = float(jnp.linalg.norm(arr.astype(jnp.float32)))
      return g_norms

  try:
    g1 = _grads_on_mesh(cfg1, mesh1)
    g2 = _grads_on_mesh(cfg2, mesh2)
  except (
      AttributeError,
      KeyError,
      NotImplementedError,
      RuntimeError,
      TypeError,
      ValueError,
  ):
    return []

  mismatches: list[str] = []
  for leaf in sorted(set(g1) & set(g2)):
    n1, n2 = g1[leaf], g2[leaf]
    mx = max(n1, n2)
    if mx > 1e-6 and abs(n1 - n2) / mx > 0.02:
      ratio = n2 / max(n1, 1e-12)
      mismatches.append(
          f"TP_GRAD_NORM_SCALING_MISMATCH [m3__{mname}]: leaf {leaf}"
          f" grad norm TP=1 ({n1:.4g}) vs TP=2 ({n2:.4g}, ratio={ratio:.3f})"
      )
  return mismatches


def _sharding_sweep_inner(changed_files: str = "") -> list[str]:
  """Runs inside the worktree's PYTHONPATH; checks m3 mesh axes & explicit mode."""
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.m3 import models as m3_models  # pylint: disable=import-outside-toplevel
  from maxtext.m3.core import sharding as m3_sharding  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

  MODEL_REGISTRY = m3_models.MODEL_REGISTRY

  base_yml = os.path.join(os.environ["PYTHONPATH"].split(":")[0], "maxtext/configs/base.yml")
  reps = _select_sweep_reps(MODEL_REGISTRY, changed_files=changed_files)

  axis_names = tuple(getattr(m3_sharding, "MESH_AXIS_NAMES", ("data", "fsdp", "tensor", "expert")))
  issues: list[str] = _probe_out_of_mesh_2d_fsdp(m3_sharding)
  for mname in reps:  # pylint: disable=too-many-nested-blocks
    base_argv = [
        None,
        base_yml,
        f"model_name={mname}",
        "use_m3_model=true",
        "scan_layers=True",
        "base_num_decoder_layers=2",
        "attention=dot_product",
        *_COMMON_OVERRIDES,
    ]
    try:
      cfg8 = pyconfig.initialize(base_argv + ["ici_fsdp_parallelism=8"])
      mesh8 = maxtext_utils.get_mesh_from_config(cfg8)
      _, mod8 = model_creation_utils.create_nnx_abstract_model(cfg8, mesh=mesh8)
      base_elems = _per_dev(_extract_params(mod8))
    except (
        AttributeError,
        KeyError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
      continue

    for ax in axis_names:
      if ax in ("data", "fsdp"):
        continue
      ov = ["ici_fsdp_parallelism=4", f"ici_{ax}_parallelism=2"]
      try:
        cfg_ax = pyconfig.initialize(base_argv + ov)
        mesh_ax = maxtext_utils.get_mesh_from_config(cfg_ax)
        _, mod_ax = model_creation_utils.create_nnx_abstract_model(cfg_ax, mesh=mesh_ax)
      except (
          AttributeError,
          KeyError,
          NotImplementedError,
          RuntimeError,
          TypeError,
          ValueError,
      ):
        continue
      # If `stage` was silently clamped to 1 in mesh.shape, check if config
      # accepted `ici_stage_parallelism=2` without raising!
      if mesh_ax.shape.get(ax, 1) <= 1:
        issues.append(
            f"INERT_MESH_AXIS [m3__{mname} {ax}=2]: config accepted"
            f" ici_{ax}_parallelism=2 without raising, but mesh.shape[{ax!r}]"
            f" is {mesh_ax.shape.get(ax, 1)}"
        )
        continue
      ps_ax = _extract_params(mod_ax)
      param_uses_ax = any(f"'{ax}'" in p["spec"] for p in ps_ax.values())
      act_uses_ax = False
      shd_obj = getattr(mod_ax, "sharding", None)
      if shd_obj is not None and callable(shd_obj):
        for tname, laxes in _LOGICAL_PROBES:
          try:
            spec = str(shd_obj(tname, laxes, m3_sharding.TensorType.ACTIVATION))
            if f"'{ax}'" in spec:
              act_uses_ax = True
              break
          except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
      if not param_uses_ax and not act_uses_ax:
        issues.append(
            f"INERT_MESH_AXIS [m3__{mname} {ax}=2]: accepted without error, but"
            f" {ax!r} appears in zero parameter or activation PartitionSpecs"
        )
      elif ax not in ("context", "tensor_sequence") and not param_uses_ax:
        ax_elems = _per_dev(ps_ax)
        if base_elems and ax_elems >= base_elems * 1.8:
          issues.append(
              f"UNSHARDED_WEIGHT_AXIS [m3__{mname} {ax}=2]: per-device params"
              f" {base_elems:,} -> {ax_elems:,}"
              f" (+{100*(ax_elems-base_elems)/base_elems:.0f}%) when {ax}=2"
              f" replaces fsdp; weights ignore {ax!r}"
          )

    # Check shard_mode=explicit: if model init accepts it, 1-step forward trace
    # must not crash.
    issues.extend(
        _probe_explicit_forward(
            mname,
            base_argv,
            ["ici_fsdp_parallelism=4", "ici_tensor_parallelism=2"],
            ("fsdp", None),
            "EXPLICIT_SHARD_MODE_FORWARD_CRASH",
            pyconfig,
            maxtext_utils,
            model_creation_utils,
            m3_models,
            jax,
            jnp,
            nnx,
        )
    )
    if "context" in axis_names:
      comp_ov = ["ici_fsdp_parallelism=2", "ici_context_parallelism=2"]
      if "tensor_sequence" in axis_names:
        comp_ov.append("ici_tensor_sequence_parallelism=2")
      elif "tensor" in axis_names:
        comp_ov.append("ici_tensor_parallelism=2")
      issues.extend(
          _probe_explicit_forward(
              mname,
              base_argv,
              comp_ov,
              ("fsdp", "context"),
              "EXPLICIT_SHARD_MODE_COMPOUND_CRASH",
              pyconfig,
              maxtext_utils,
              model_creation_utils,
              m3_models,
              jax,
              jnp,
              nnx,
          )
      )
    if "tensor" in axis_names:
      issues.extend(
          _probe_tp_grad_parity(
              mname,
              base_yml,
              pyconfig,
              maxtext_utils,
              m3_models,
              jax,
              jnp,
              nnx,
          )
      )
  return issues


def run_sharding_sweep(wt: str, md_out: str = "", changed_files: str = "", py_exe: str = "") -> int:
  """Executes the isolated CPU sharding and TP gradient sweep subprocess."""
  resolved_py = py_exe or _default_python()
  dep_err = _check_python_deps(resolved_py, ("jax",))
  if dep_err:
    sys.stderr.write(f"Error: {dep_err}\n")
    return 2
  env = dict(os.environ)
  env["PYTHONPATH"] = os.path.join(os.path.abspath(wt), "src")
  env["JAX_PLATFORMS"] = "cpu"
  env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
  env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "0")
  env.pop("TPU_NAME", None)
  cmd = [resolved_py, _HERE, "sharding-sweep-inner"]
  if changed_files:
    cmd += ["--changed-files", changed_files]
  p = subprocess.run(
      cmd,
      env=env,
      cwd=wt,
      capture_output=True,
      text=True,
      timeout=600,
      check=False,
  )
  if p.returncode != 0:
    sys.stderr.write(p.stderr[-1500:])
    return 2
  issues = json.loads(p.stdout)
  lines = [f"# Tier A sharding.py sweep: {len(issues)} issues", ""]
  lines += [f"- {x}" for x in issues]
  text = "\n".join(lines) + "\n"
  if md_out:
    with open(md_out, "a", encoding="utf-8") as f:
      f.write("\n" + text)
  print(text)
  return 1 if issues else 0


def main() -> int:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  many = sub.add_parser("dump-many")
  many.add_argument("--models", required=True)
  many.add_argument("--outdir", required=True)
  many.add_argument("--extra", nargs="*", default=[])
  dp = sub.add_parser("dump")
  dp.add_argument("--worktree", required=True)
  dp.add_argument("--out", required=True)
  dp.add_argument("--models", default="")
  dp.add_argument("-j", type=int, default=32)
  dp.add_argument("--python", default=_default_python())
  dp.add_argument("--extra", nargs="*", default=[])
  cp = sub.add_parser("compare")
  cp.add_argument("base")
  cp.add_argument("head")
  cp.add_argument("--named", default="")
  cp.add_argument("--main-base", default="")
  cp.add_argument("--md", default="")
  ss = sub.add_parser("sharding-sweep")
  ss.add_argument("--worktree", required=True)
  ss.add_argument("--md", default="")
  ss.add_argument("--changed-files", default="")
  ss.add_argument("--python", default=_default_python())
  ssi = sub.add_parser("sharding-sweep-inner")
  ssi.add_argument("--changed-files", default="")
  a = ap.parse_args()
  if a.cmd == "dump-many":
    _dump_many(a.models.split(","), a.outdir, a.extra)
    return 0
  if a.cmd == "dump":
    models = [m for m in a.models.split(",") if m] if a.models else _list_models(a.worktree)
    if models:
      dep_err = _check_python_deps(a.python, ("jax",))
      if dep_err:
        sys.stderr.write(f"Error: {dep_err}\n")
        return 2
    _dump(a.worktree, a.out, models, a.j, a.extra, py_exe=a.python)
    if models and not any(_has_ok_variant(os.path.join(a.out, f"{m}.json")) for m in models):
      sys.stderr.write(f"Error: 0/{len(models)} models produced status='ok' in {a.out}\n")
      return 2
    return 0
  if a.cmd == "sharding-sweep-inner":
    print(json.dumps(_sharding_sweep_inner(a.changed_files)))
    return 0
  if a.cmd == "sharding-sweep":
    return run_sharding_sweep(a.worktree, a.md, a.changed_files, py_exe=a.python)
  rep = compare(a.base, a.head, a.named, a.main_base)
  exempt = {"(named by PR)", "(restores origin/main)"}
  unnamed = {m: v for m, v in rep.items() if v and v[0] not in exempt}
  bug_prefixes = (
      "BUILD_REGRESSION",
      "NEW_MODEL_BUILD_FAILS",
      "MUON_SCAN_AXIS_FOLDED",
      "TENSOR_AXIS_ASYMMETRY",
      "PER_DEVICE_PARAMS_UP",
  )
  structural_bugs = {
      m: [x for x in v if x.startswith(bug_prefixes)]
      for m, v in rep.items()
      if v and v[0] != "(restores origin/main)" and any(x.startswith(bug_prefixes) for x in v)
  }
  lines = [
      (
          f"# Tier A abstract diff: {len(rep)} models differ, "
          f"{len(unnamed)} not named by the PR, "
          f"{len(structural_bugs)} with structural bugs"
      ),
      "",
  ]
  for m, items in rep.items():
    lines.append(f"## {m}")
    lines += [f"- {x}" for x in items[:40]]
    if len(items) > 40:
      lines.append(f"- ... {len(items) - 40} more")
    lines.append("")
  text = "\n".join(lines)
  if a.md:
    with open(a.md, "w", encoding="utf-8") as f:
      f.write(text)
  print(text[:6000])
  return 1 if (unnamed or structural_bugs) else 0


if __name__ == "__main__":
  sys.exit(main())
