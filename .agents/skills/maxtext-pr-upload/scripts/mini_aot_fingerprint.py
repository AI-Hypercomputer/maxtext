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

"""Tier B: mini-model AOT compile fingerprint diff and remat/kernel policy sweep.

  python mini_aot_fingerprint.py compile --worktree WT --model M --out DIR
  python mini_aot_fingerprint.py batch --worktree WT --models a,b --out DIR [-j
  16]
  python mini_aot_fingerprint.py compare BASE_DIR HEAD_DIR [--named REGEX]
  [--main-base DIR]

`compile` AOT-compiles the train step for a few-layer version of the model on
a virtual 8-chip TPU v7x (`tpu7x-8`), dumps the optimized HLO, and writes a
structural fingerprint: collectives by op and shape, dot/convolution operand
dtypes, rank>=3 boolean tensors with two sequence-length dims, custom-call
targets, and compiled memory sizes. Supports `@dot_product` and `@remat_*`
variant suffixes. On stacked PRs, `--main-base` classifies changes that restore
`origin/main` as `(restores origin/main)` instead of regressions.
"""

import argparse
import collections
import concurrent.futures
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

_SEQ = 256
_COLL_RE = re.compile(
    r"=\s*(\(?[a-z0-9]+\[[^\]]*\][^=]*?)\s+"
    r"(all-gather|all-reduce|reduce-scatter|all-to-all|collective-permute|"
    r"ragged-all-to-all)(?:-start)?\("
)
_DOT_RE = re.compile(
    r"=\s*([a-z0-9]+)\[[^\]]*\][^\s]*\s+" r"(dot|convolution(?:-[\w\-]+)?)\(\s*(%[\w.\-]+)\s*,\s*(%[\w.\-]+)"
)
_SSA_RE = re.compile(r"(%[\w.\-]+)\s*=\s*([a-z0-9]+)\[")
_PRED_RE = re.compile(r"\bpred\[(\d+(?:,\d+){2,})\]")
_CC_RE = re.compile(r'custom_call_target="([^"]+)"')
_MEM_RE = re.compile(r"(\w+_size_in_bytes)=(\d+)")


def _load_yml_dict(wt: str, model: str) -> dict:
  """Loads a model YAML dictionary if present, else returns an empty dict."""
  p = os.path.join(wt, "src/maxtext/configs/models", f"{model}.yml")
  if not os.path.exists(p):
    return {}
  try:
    import yaml  # pylint: disable=import-outside-toplevel
  except ImportError:
    return {}
  try:
    with open(p, encoding="utf-8") as f:
      d = yaml.safe_load(f)
    return d if isinstance(d, dict) else {}
  except (OSError, ValueError, yaml.YAMLError):
    return {}


def _yml_val(wt: str, model: str, key: str) -> str:
  d = _load_yml_dict(wt, model)
  if key in d and d[key] is not None:
    return str(d[key])
  p = os.path.join(wt, "src/maxtext/configs/models", f"{model}.yml")
  if not os.path.exists(p):
    return ""
  with open(p, encoding="utf-8") as f:
    m = re.search(rf"^{key}:\s*([^\s#]+)", f.read(), re.M)
  return m.group(1).strip("\"'") if m else ""


def _has_fixed_layer_schedule(yml: dict) -> bool:
  native = yml.get("base_num_decoder_layers")
  if not isinstance(native, int) or native <= 0:
    return False
  return any(isinstance(v, list) and len(v) == native for v in yml.values())


def _default_layer_options(wt: str, model: str, m3: bool) -> tuple[int | None, ...]:
  if m3:
    return (2, 4, 6, 8)
  yml = _load_yml_dict(wt, model)
  if _has_fixed_layer_schedule(yml):
    return (None,)
  native = yml.get("base_num_decoder_layers")
  if isinstance(native, int) and 0 < native <= 24:
    return (2, 4, 6, 8, None)
  return (2, 4, 6, 8)


def _mini_overrides(
    wt: str,
    model: str,
    layers: int | None,
    extra: list[str] | None = None,
) -> list[str]:
  """Few-layer config that keeps both dense and MoE layer kinds."""
  yml = _load_yml_dict(wt, model)
  if layers is None:
    ov = ["scan_layers=True"]
    native = int(yml.get("base_num_decoder_layers") or 0)
    if native > 16:
      if int(yml.get("num_experts") or 0) > 16:
        ov.append("num_experts=8")
        if int(yml.get("num_experts_per_tok") or 0) > 8:
          ov.append("num_experts_per_tok=6")
      if int(yml.get("ds41_engram_vocab_size") or 0) > 32768:
        ov.append("ds41_engram_vocab_size=10000")
      if int(yml.get("indexer_topk") or 0) > 32:
        ov.append("indexer_topk=4")
      if int(yml.get("ds41_candidate_topk_blocks") or 0) > 32:
        ov.append("ds41_candidate_topk_blocks=8")
      if not any(x.startswith("remat_policy=") for x in extra or []):
        ov.append("remat_policy=full")
  else:
    ov = [f"base_num_decoder_layers={layers}", "scan_layers=True"]
    if _yml_val(wt, model, "first_num_dense_layers"):
      ov.append("first_num_dense_layers=1")
  experts = _yml_val(wt, model, "num_experts")
  if experts and experts.isdigit() and int(experts) > 1:
    ov += [
        "ici_fsdp_parallelism=2",
        "ici_tensor_parallelism=2",
        "ici_expert_parallelism=2",
    ]
  else:
    ov += ["ici_fsdp_parallelism=2", "ici_tensor_parallelism=4"]
  return ov


def fingerprint(hlo_text: str, log_text: str) -> dict:
  """Extracts collective, dot, mask, custom-call, and memory counts from HLO."""
  colls = collections.Counter()
  for m in _COLL_RE.finditer(hlo_text):
    shape = re.sub(r"\{[^}]*\}", "", m.group(1)).strip()
    colls[f"{m.group(2)} {shape}"] += 1
  var_dtype = dict(_SSA_RE.findall(hlo_text))
  dots = collections.Counter(
      f"{m.group(2)} out={m.group(1)} lhs={var_dtype.get(m.group(3), '?')}" f" rhs={var_dtype.get(m.group(4), '?')}"
      for m in _DOT_RE.finditer(hlo_text)
  )
  preds = collections.Counter()
  for m in _PRED_RE.finditer(hlo_text):
    dims = [int(x) for x in m.group(1).split(",")]
    if dims.count(_SEQ) >= 2:
      preds[m.group(1)] += 1
  ccs = collections.Counter(_CC_RE.findall(hlo_text))
  mem = {k: int(v) for k, v in _MEM_RE.findall(log_text)}
  return {
      "collectives": dict(colls),
      "dots": dict(dots),
      "pred_seq_seq": dict(preds),
      "custom_calls": dict(ccs),
      "memory": mem,
      "instructions": hlo_text.count("\n  %"),
  }


_OPT_IN_VARIANT_OVERRIDES: dict[str, tuple[list[str], bool]] = {
    "dot_product": (["attention=dot_product"], False),
    "offload_param": (
        [
            "parameter_memory_host_offload=true",
            "optimizer_memory_host_offload=false",
        ],
        False,
    ),
    "offload_ga2": (
        [
            "parameter_memory_host_offload=true",
            "optimizer_memory_host_offload=true",
            "param_scan_axis=0",
            "gradient_accumulation_steps=2",
        ],
        False,
    ),
    "moe_ragged_sort": (
        [
            "ici_fsdp_parallelism=2",
            "ici_expert_parallelism=4",
            "ici_tensor_parallelism=1",
            "use_ragged_sort=true",
        ],
        False,
    ),
    "moe_dropless_ragged_sort": (
        [
            "ici_fsdp_parallelism=1",
            "ici_expert_parallelism=8",
            "ici_tensor_parallelism=1",
            "use_ragged_sort=true",
            "moe_dropless_buffer_factor=1.3",
            "max_target_length=512",
        ],
        False,
    ),
}
_TOKENIZER_LLAMA2_ERRS = (
    "AutoTokenizer",
    "JSONDecodeError",
    "Can't load tokenizer",
    "Unrecognized model",
)


def _suffix_overrides(suffix: str) -> tuple[list[str], bool]:
  """Returns `(extra_cli_args, force_unscanned)` for `@suffix`."""
  if not suffix:
    return [], False
  if suffix in _OPT_IN_VARIANT_OVERRIDES:
    args, unscanned = _OPT_IN_VARIANT_OVERRIDES[suffix]
    return list(args), unscanned
  if suffix.startswith("remat_"):
    pol = suffix[len("remat_") :]
    unscanned = False
    if pol.endswith("__unscanned"):
      pol = pol[: -len("__unscanned")]
      unscanned = True
    return [f"remat_policy={pol}"], unscanned
  return [], False


def _is_tokenizer_llama2_error(log: str) -> bool:
  return "tokenizer.llama2" in log and any(marker in log for marker in _TOKENIZER_LLAMA2_ERRS)


def _default_python() -> str:
  """Resolves Python executable from $MAXTEXT_PILOT_PYTHON or sys.executable."""
  return os.environ.get("MAXTEXT_PILOT_PYTHON", "").strip() or sys.executable


def _run_train_compile(  # pylint: disable=too-many-positional-arguments
    wt: str,
    name: str,
    out: str,
    dump: str,
    ov: list[str],
    extra: list[str],
    py_exe: str = "",
) -> tuple[subprocess.CompletedProcess[str], str, list[str]]:
  """Runs train_compile once and returns (process, combined_log, hlo_files)."""
  cmd = [
      py_exe or _default_python(),
      "-m",
      "maxtext.trainers.pre_train.train_compile",
      os.path.join(wt, "src/maxtext/configs/base.yml"),
      f"model_name={name}",
      "dataset_type=synthetic",
      "override_model_config=true",
      "compile_topology=tpu7x-8",
      "compile_topology_num_slices=1",
      "per_device_batch_size=1",
      f"max_target_length={_SEQ}",
      f"base_output_directory={out}/bo",
      "run_name=fp",
      "enable_checkpointing=false",
      *ov,
      *extra,
      (
          f"compile_xla_flags=--xla_dump_to={dump}"
          " --xla_dump_hlo_as_text=True"
          " --xla_dump_hlo_module_re=jit_train_step"
      ),
  ]
  env = dict(os.environ)
  env.update(
      {
          "PYTHONPATH": os.path.join(wt, "src"),
          "TPU_SKIP_MDS_QUERY": "1",
          "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
      }
  )
  env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "0")
  p = subprocess.run(
      cmd,
      cwd=wt,
      env=env,
      capture_output=True,
      text=True,
      timeout=1800,
      check=False,
  )
  log = p.stdout + p.stderr
  files = sorted(glob.glob(os.path.join(dump, "*after_optimizations.txt")))
  return p, log, files


def compile_one(
    wt: str,
    model: str,
    out: str,
    extra: list[str],
    layer_options: tuple[int | None, ...] | None = None,
    *,
    py_exe: str = "",
) -> dict[str, Any]:
  """Compiles a mini/m3 model variant and returns its HLO fingerprint record."""
  os.makedirs(out, exist_ok=True)
  base_model, _, suffix = model.partition("@")
  sfx_extra, force_unscanned = _suffix_overrides(suffix)
  extra = list(extra) + sfx_extra
  m3 = base_model.startswith("m3__")
  name = base_model[len("m3__") :] if m3 else base_model
  yml = {} if m3 else _load_yml_dict(wt, name)
  native_layers = yml.get("base_num_decoder_layers")
  if layer_options is None:
    layer_options = _default_layer_options(wt, name, m3)
  res: dict[str, Any] = {"model": model}
  tok_fallback_used = False
  for layers in layer_options:
    dump = os.path.join(out, "dump")
    shutil.rmtree(dump, ignore_errors=True)
    scan_flag = "scan_layers=False" if force_unscanned else "scan_layers=True"
    if m3:
      ov = [
          "use_m3_model=true",
          f"base_num_decoder_layers={layers}",
          scan_flag,
          "ici_fsdp_parallelism=4",
          "ici_tensor_parallelism=2",
      ]
    else:
      ov = _mini_overrides(wt, name, layers, extra=extra)
    if force_unscanned and not m3:
      ov = [scan_flag if x.startswith("scan_layers=") else x for x in ov]
    effective_layers = layers if layers is not None else (native_layers if isinstance(native_layers, int) else "native")
    t0 = time.time()
    p, log, files = _run_train_compile(wt, name, out, dump, ov, extra, py_exe=py_exe)
    if (
        not files
        and not tok_fallback_used
        and _is_tokenizer_llama2_error(log)
        and not any(x.startswith("tokenizer_path=") for x in extra)
    ):
      tok_fallback_used = True
      qwen3_tok = ""
      for rel_tok in (
          "src/maxtext/assets/tokenizers/qwen3-tokenizer",
          "assets/tokenizers/qwen3-tokenizer",
          "src/maxtext/assets/qwen3-tokenizer",
          "assets/qwen3-tokenizer",
      ):
        cand = os.path.join(wt, rel_tok)
        if os.path.isdir(cand):
          qwen3_tok = cand
          break
      if qwen3_tok:
        tok_ovs = [f"tokenizer_path={qwen3_tok}"]
        for k, v in yml.items():
          if (
              k.endswith("_compressed_vocab_size")
              and isinstance(v, int)
              and v > 0
              and not any(x.startswith(f"{k}=") for x in extra)
          ):
            tok_ovs.append(f"{k}=0")
      else:
        tok_ovs = ["tokenizer_type=tiktoken"]
      extra = list(extra) + tok_ovs
      shutil.rmtree(dump, ignore_errors=True)
      p, log, files = _run_train_compile(wt, name, out, dump, ov, extra, py_exe=py_exe)
    res.update(
        {
            "layers": effective_layers,
            "rc": p.returncode,
            "seconds": round(time.time() - t0),
            "overrides": ov + extra,
        }
    )
    if files:
      with open(files[-1], encoding="utf-8") as f:
        res["fingerprint"] = fingerprint(f.read(), log)
      break
    res["error"] = log[-1200:]
    if m3:
      low = log.lower()
      if "multiple of" not in low and "divisible" not in low:
        break
  shutil.rmtree(os.path.join(out, "dump"), ignore_errors=True)
  with open(os.path.join(out, "fp.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
  return res


def _diff_pair(model: str, b: dict, h: dict) -> list[str]:
  """Computes structural HLO and memory differences between base and head records."""
  items: list[str] = []
  if "@" in model and "fingerprint" not in h:
    if "fingerprint" in b or any(tag in model for tag in ("@dot_product", "@offload_", "@moe_", "@remat_")):
      items.append("OPT_IN_VARIANT_COMPILE_FAIL: head fails:" f" {h.get('error', '')[-400:]}")
      return items
  if "fingerprint" in b and "fingerprint" not in h:
    items.append(f"COMPILE_REGRESSION: head fails: {h.get('error', '')[-400:]}")
  elif "fingerprint" in h and "fingerprint" in b:
    if b.get("layers") != h.get("layers"):
      items.append(f"LAYER_COUNT_DIFFERS {b.get('layers')} vs {h.get('layers')}")
    bf, hf = b["fingerprint"], h["fingerprint"]
    for cat in ("collectives", "dots", "pred_seq_seq", "custom_calls"):
      keys = set(bf[cat]) | set(hf[cat])
      for k in sorted(keys):
        if bf[cat].get(k, 0) != hf[cat].get(k, 0):
          items.append(f"{cat.upper()} {k}: {bf[cat].get(k, 0)} -> {hf[cat].get(k, 0)}")
    for k in ("temp_size_in_bytes", "argument_size_in_bytes"):
      bv, hv = bf["memory"].get(k), hf["memory"].get(k)
      if bv and hv and hv > bv * 1.05:
        items.append(f"MEMORY {k}: {bv:,} -> {hv:,} (+{100*(hv-bv)/bv:.0f}%)")
    for k, hv in sorted(hf["dots"].items()):
      bv = bf["dots"].get(k, 0)
      if hv > bv and ("out=f32" in k or "lhs=f32" in k or "rhs=f32" in k):
        items.insert(0, f"DOT_FP32_UPCAST {k}: {bv} -> {hv}")
    if model.endswith("@dot_product"):
      b_pred = sum(bf["pred_seq_seq"].values())
      h_pred = sum(hf["pred_seq_seq"].values())
      if h_pred > b_pred:
        items.insert(
            0,
            f"PRED_SEQ_SEQ_UP {b_pred} -> {h_pred}: dense [B,S,S] mask added",
        )
  return items


def compare(base_dir: str, head_dir: str, named: str, main_base_dir: str = "") -> dict:
  """Compares base vs head AOT fingerprint directories across all models."""
  named_re = re.compile(named) if named else None
  out: dict = {}
  head_records: dict[str, dict] = {}
  for hp in sorted(glob.glob(os.path.join(head_dir, "*", "fp.json"))):
    model = os.path.basename(os.path.dirname(hp))
    with open(hp, encoding="utf-8") as f:
      h = json.load(f)
    head_records[model] = h
    bp = os.path.join(base_dir, model, "fp.json")
    if not os.path.exists(bp):
      if "@" in model and "fingerprint" not in h:
        out[model] = ["OPT_IN_VARIANT_COMPILE_FAIL: head fails:" f" {h.get('error', '')[-400:]}"]
      else:
        out[model] = ["NO_BASE_RESULT"]
      continue
    with open(bp, encoding="utf-8") as f:
      b = json.load(f)
    items = _diff_pair(model, b, h)
    base_model = model.partition("@")[0]
    is_named = bool(named_re and (named_re.search(model) or named_re.search(base_model)))
    if not items and "fingerprint" not in b and "fingerprint" not in h and is_named:
      out[model] = [
          "(named by PR)",
          f"NAMED_MODEL_COMPILE_FAILED_BOTH: {h.get('error', '')[-320:]}",
      ]
      continue
    if items:
      restores_main = False
      if main_base_dir:
        mp = os.path.join(main_base_dir, model, "fp.json")
        if os.path.exists(mp):
          with open(mp, encoding="utf-8") as f:
            mb = json.load(f)
          restores_main = not _diff_pair(model, mb, h)
      if is_named:
        items.insert(0, "(named by PR)")
      elif restores_main:
        items.insert(0, "(restores origin/main)")
      out[model] = items

  # Cross-policy check on head: remat=full (unscanned) must cut temp memory vs
  # remat=none (unscanned).
  for model, h in head_records.items():
    if not model.endswith("@remat_full__unscanned") or "fingerprint" not in h:
      continue
    none_key = model.replace("@remat_full__unscanned", "@remat_none__unscanned")
    h_none = head_records.get(none_key, {})
    if "fingerprint" not in h_none:
      continue
    t_full = h["fingerprint"]["memory"].get("temp_size_in_bytes", 0)
    t_none = h_none["fingerprint"]["memory"].get("temp_size_in_bytes", 0)
    if t_full and t_none and t_full >= t_none * 0.98:
      out.setdefault(model, []).insert(
          0,
          f"REMAT_FULL_UNSCANNED_NO_SAVINGS: full temp={t_full:,} vs none"
          f" temp={t_none:,} (prevent_cse=False or missing checkpoint)",
      )
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("compile")
  c.add_argument("--worktree", required=True)
  c.add_argument("--model", required=True)
  c.add_argument("--out", required=True)
  c.add_argument("--python", default=_default_python())
  c.add_argument("--extra", nargs="*", default=[])
  bt = sub.add_parser("batch")
  bt.add_argument("--worktree", required=True)
  bt.add_argument("--models", required=True)
  bt.add_argument("--out", required=True)
  bt.add_argument("-j", type=int, default=16)
  bt.add_argument("--python", default=_default_python())
  bt.add_argument("--extra", nargs="*", default=[])
  bt.add_argument(
      "--layers-from",
      default="",
      help="Reuse the layer count each model compiled with here.",
  )
  cp = sub.add_parser("compare")
  cp.add_argument("base")
  cp.add_argument("head")
  cp.add_argument("--named", default="")
  cp.add_argument("--main-base", default="")
  a = ap.parse_args()
  if a.cmd == "compile":
    r = compile_one(a.worktree, a.model, a.out, a.extra, py_exe=a.python)
    print(json.dumps({k: r.get(k) for k in ("model", "layers", "rc", "seconds")}))
    return 0
  if a.cmd == "batch":
    os.makedirs(a.out, exist_ok=True)
    models = [m for m in a.models.split(",") if m]

    def job(m):
      dst = os.path.join(a.out, m)
      if os.path.exists(os.path.join(dst, "fp.json")):
        return f"{m}: cached"
      opts: tuple[int | None, ...] | None = None
      ref = os.path.join(a.layers_from, m, "fp.json") if a.layers_from else ""
      if ref and os.path.exists(ref):
        with open(ref, encoding="utf-8") as f:
          ref_obj = json.load(f)
        ref_layers = ref_obj.get("layers") if "fingerprint" in ref_obj else None
        if isinstance(ref_layers, int):
          bm = m.partition("@")[0]
          yml_bm = {} if bm.startswith("m3__") else _load_yml_dict(a.worktree, bm)
          if _has_fixed_layer_schedule(yml_bm) and ref_layers == yml_bm.get("base_num_decoder_layers"):
            opts = (None,)
          else:
            opts = (ref_layers,)
      r = compile_one(a.worktree, m, dst, a.extra, opts, py_exe=a.python)
      return f"{m}: layers={r.get('layers')} ok={'fingerprint' in r}" f" {r.get('seconds')}s"

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.j) as ex:
      for msg in ex.map(job, models):
        print(msg, flush=True)
    ok_count = 0
    failed_models: list[str] = []
    for m in models:
      fp_path = os.path.join(a.out, m, "fp.json")
      ok = False
      if os.path.exists(fp_path):
        try:
          with open(fp_path, encoding="utf-8") as f:
            ok = "fingerprint" in json.load(f)
        except (OSError, json.JSONDecodeError):
          ok = False
      if ok:
        ok_count += 1
      else:
        failed_models.append(m)
    with open(os.path.join(a.out, "batch_summary.json"), "w", encoding="utf-8") as f:
      json.dump(
          {"total": len(models), "ok": ok_count, "failed": failed_models},
          f,
          indent=1,
      )
    print(f"batch {len(models)} models ({ok_count} ok," f" {len(failed_models)} failed) in {time.time() - t0:.0f}s")
    return 2 if (models and ok_count == 0) else 0
  rep = compare(a.base, a.head, a.named, a.main_base)
  exempt = {"(named by PR)", "(restores origin/main)"}
  unnamed = [m for m, v in rep.items() if v and v[0] not in exempt]
  bug_prefixes = (
      "COMPILE_REGRESSION",
      "OPT_IN_VARIANT_COMPILE_FAIL",
      "NAMED_MODEL_COMPILE_FAILED_BOTH",
      "REMAT_FULL_UNSCANNED_NO_SAVINGS",
      "DOT_FP32_UPCAST",
      "PRED_SEQ_SEQ_UP",
      "MEMORY",
  )
  structural_bugs = [
      m for m, v in rep.items() if v and v[0] != "(restores origin/main)" and any(x.startswith(bug_prefixes) for x in v)
  ]
  print(
      f"# Tier B: {len(rep)} models differ, {len(unnamed)} not named by the PR,"
      f" {len(structural_bugs)} with structural bugs"
  )
  for m, items in rep.items():
    print(f"## {m}")
    for x in items[:30]:
      print(f"- {x}")
  return 1 if (unnamed or structural_bugs) else 0


if __name__ == "__main__":
  sys.exit(main())
