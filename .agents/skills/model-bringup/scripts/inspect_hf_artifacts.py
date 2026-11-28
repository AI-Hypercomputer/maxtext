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

"""Inspects unmodified Hugging Face config, safetensors headers, and source AST.

Reads config.json (including inference/config.json fallback), 8-byte safetensors
headers (without loading tensor bytes into memory), and reference Python files
recursively to extract a deterministic contract (including precision_contract
and derived_prod_dims) before porting a model to JAX.

Exit code: 0 = PASS, 2 = usage or I/O error.
"""

import argparse
import ast
import collections
from collections.abc import Mapping, Sequence
import glob
import json
import os
import re
import struct
import sys
from typing import Any

_NESTED_CONFIG_KEYS = (
    "text_config",
    "vision_config",
    "audio_config",
    "llm_config",
    "language_config",
)
_CORE_CONFIG_KEYS = (
    "model_type",
    "architectures",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_head_dim",
    "v_head_dim",
    "vocab_size",
    "rms_norm_eps",
    "layer_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
    "torch_dtype",
    "dtype",
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "num_experts_per_tok",
)
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_SUB_IDX_RE = re.compile(r"\.(\d+)(?=\.|$)")


def read_safetensors_header(path: str) -> dict[str, Any]:
  """Reads only the JSON header of a .safetensors shard without mmap/payload."""
  with open(path, "rb") as f:
    raw_len = f.read(8)
    if len(raw_len) != 8:
      raise ValueError(f"Truncated safetensors header in {path}")
    (header_len,) = struct.unpack("<Q", raw_len)
    if header_len <= 0 or header_len > 256 * 1024 * 1024:
      raise ValueError(f"Invalid header length {header_len} in {path}")
    header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
      raise ValueError(f"Incomplete safetensors header in {path}")
  parsed = json.loads(header_bytes.decode("utf-8"))
  if not isinstance(parsed, dict):
    raise ValueError(f"Expected JSON object header in {path}")
  parsed.pop("__metadata__", None)
  return parsed


def _compute_derived_prod_dims(
    core: Mapping[str, Any], text_cfg: Mapping[str, Any], raw: Mapping[str, Any]
) -> dict[str, int]:
  """Derives default production symbol dimensions from config.json fields."""
  if "chunk_size" in text_cfg:
    chunk_raw = text_cfg["chunk_size"]
  elif "chunk_size" in raw:
    chunk_raw = raw["chunk_size"]
  else:
    chunk_raw = 64
  if isinstance(chunk_raw, int) and not isinstance(chunk_raw, bool) and chunk_raw > 0:
    chunk_size = int(chunk_raw)
  else:
    chunk_size = 64
  seq_len = 4096
  dims: dict[str, int] = {
      "B": 1,
      "S": seq_len,
      "C": chunk_size,
      "N_chunks": (seq_len + chunk_size - 1) // chunk_size,
  }
  hidden = core.get("hidden_size")
  if isinstance(hidden, int) and not isinstance(hidden, bool) and hidden > 0:
    dims["hidden"] = int(hidden)
  heads = core.get("num_attention_heads")
  if isinstance(heads, int) and not isinstance(heads, bool) and heads > 0:
    dims["H"] = int(heads)
  head_dim = core.get("head_dim") or core.get("qk_head_dim") or core.get("v_head_dim")
  if isinstance(head_dim, int) and not isinstance(head_dim, bool) and head_dim > 0:
    dims["D"] = int(head_dim)
  elif "hidden" in dims and "H" in dims and dims["H"] > 0:
    dims["D"] = dims["hidden"] // dims["H"]
  experts = core.get("n_routed_experts") or core.get("num_experts") or core.get("num_local_experts")
  if isinstance(experts, int) and not isinstance(experts, bool) and experts > 0:
    dims["experts"] = int(experts)
  topk = core.get("num_experts_per_tok")
  if isinstance(topk, int) and not isinstance(topk, bool) and topk > 0:
    dims["topk"] = int(topk)
  return dims


def summarize_config(hf_dir: str) -> dict[str, Any]:
  """Extracts nested structure, core dimensions, and epsilon warnings."""
  cfg_path = os.path.join(hf_dir, "config.json")
  inf_cfg_path = os.path.join(hf_dir, "inference", "config.json")
  raw: dict[str, Any] = {}
  found_any = False

  if os.path.isfile(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
      loaded = json.load(f)
    if not isinstance(loaded, dict):
      raise ValueError("config.json root must be a JSON object")
    raw.update(loaded)
    found_any = True

  if os.path.isfile(inf_cfg_path):
    with open(inf_cfg_path, "r", encoding="utf-8") as f:
      inf_loaded = json.load(f)
    if isinstance(inf_loaded, dict):
      for k, v in inf_loaded.items():
        raw.setdefault(k, v)
      found_any = True

  if not found_any:
    return {"present": False, "warnings": ["Missing config.json"]}

  nested_found = [k for k in _NESTED_CONFIG_KEYS if isinstance(raw.get(k), dict)]
  text_cfg: Mapping[str, Any] = raw["text_config"] if isinstance(raw.get("text_config"), dict) else raw
  core: dict[str, Any] = {}
  for k in _CORE_CONFIG_KEYS:
    if k in text_cfg:
      core[k] = text_cfg[k]
    elif k in raw:
      core[k] = raw[k]

  extra_arch_keys = sorted(
      k
      for k in text_cfg.keys()
      if any(
          tok in k
          for tok in (
              "moe",
              "expert",
              "linear",
              "conv",
              "chunk",
              "lora",
              "latent",
              "rope",
              "nope",
              "layer",
              "norm",
              "act",
              "mtp",
          )
      )
      and k not in core
  )
  warnings: list[str] = []
  if nested_found:
    warnings.append(
        f"Nested sub-configs present ({', '.join(nested_found)}); unwrap both" " Mapping and PretrainedConfig inputs."
    )
  eps = core.get("rms_norm_eps")
  if eps is not None and float(eps) != 1e-6:
    warnings.append(
        f"rms_norm_eps={eps} differs from typical 1e-6 RMSNorm class default;"
        " verify every submodule norm constructor passes eps explicitly."
    )
  return {
      "present": True,
      "nested_subconfigs": nested_found,
      "core_fields": core,
      "extra_arch_fields": {k: text_cfg[k] for k in extra_arch_keys},
      "derived_prod_dims": _compute_derived_prod_dims(core, text_cfg, raw),
      "warnings": warnings,
  }


def _canonical_pattern(key: str) -> tuple[str, int | None]:
  """Replaces layers.<idx>. with {i} and expert/sub-indices with {j}."""
  match = _LAYER_RE.search(key)
  if not match:
    return _SUB_IDX_RE.sub(".{j}", key), None
  idx = int(match.group(1))
  start, end = match.span(1)
  with_layer = f"{key[:start]}{{i}}{key[end:]}"
  return _SUB_IDX_RE.sub(".{j}", with_layer), idx


def summarize_safetensors(hf_dir: str, core_fields: Mapping[str, Any]) -> dict[str, Any]:
  """Builds a compact pattern inventory from .safetensors headers or index."""
  shard_paths = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
  if not shard_paths:
    shard_paths = sorted(glob.glob(os.path.join(hf_dir, "inference", "*.safetensors")))
  index_path = os.path.join(hf_dir, "model.safetensors.index.json")

  raw_tensors: dict[str, dict[str, Any]] = {}
  dtypes_available = False
  if shard_paths:
    dtypes_available = True
    for sp in shard_paths:
      hdr = read_safetensors_header(sp)
      for name, meta in hdr.items():
        if isinstance(meta, dict):
          raw_tensors[name] = {
              "dtype": str(meta.get("dtype", "UNKNOWN")),
              "shape": list(meta.get("shape", [])),
          }
  elif os.path.isfile(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
      idx_data = json.load(f)
    weight_map = idx_data.get("weight_map", {})
    if isinstance(weight_map, dict):
      for name in sorted(weight_map.keys()):
        raw_tensors[name] = {"dtype": "INDEX_ONLY", "shape": []}

  if not raw_tensors:
    return {
        "present": False,
        "dtypes_available": False,
        "total_tensors": 0,
        "patterns": [],
    }

  dtype_counts: collections.Counter[str] = collections.Counter()
  grouped: dict[tuple[str, str, tuple[int, ...]], list[int]] = collections.defaultdict(list)
  non_layer_counts: collections.Counter[tuple[str, str, tuple[int, ...]]] = collections.Counter()

  for name, meta in sorted(raw_tensors.items()):
    dtype = meta["dtype"]
    shape = tuple(int(d) for d in meta["shape"])
    dtype_counts[dtype] += 1
    pat, layer_idx = _canonical_pattern(name)
    if layer_idx is None:
      non_layer_counts[(pat, dtype, shape)] += 1
    else:
      grouped[(pat, dtype, shape)].append(layer_idx)

  dominant_dtype = dtype_counts.most_common(1)[0][0] if dtype_counts else "UNKNOWN"
  precision_sensitive: list[dict[str, Any]] = []
  quantized_packed: list[dict[str, Any]] = []
  patterns_out: list[dict[str, Any]] = []

  for (pat, dtype, shape), cnt in sorted(non_layer_counts.items()):
    item: dict[str, Any] = {
        "pattern": pat,
        "dtype": dtype,
        "shape": list(shape),
        "count": cnt,
    }
    patterns_out.append(item)
    if dtypes_available and item["dtype"] in ("F32", "I32", "I64") and dominant_dtype not in ("F32", "UNKNOWN"):
      precision_sensitive.append(item)
    if dtypes_available and (
        item["dtype"] in ("U8", "I8", "F8_E4M3", "F8_E5M2") or "packed" in item["pattern"] or "scale" in item["pattern"]
    ):
      quantized_packed.append(item)

  for (pat, dtype, shape), indices in sorted(grouped.items()):
    sorted_idx = sorted(set(indices))
    diffs = sorted({sorted_idx[i + 1] - sorted_idx[i] for i in range(len(sorted_idx) - 1)})
    entry = {
        "pattern": pat,
        "dtype": dtype,
        "shape": list(shape),
        "count": len(sorted_idx),
        "layer_min": sorted_idx[0],
        "layer_max": sorted_idx[-1],
        "step_sizes": diffs,
    }
    if len(diffs) > 1:
      entry["irregular_schedule_layers"] = sorted_idx
    patterns_out.append(entry)
    if dtypes_available and dtype in ("F32", "I32", "I64") and dominant_dtype not in ("F32", "UNKNOWN"):
      precision_sensitive.append(entry)
    if dtypes_available and (dtype in ("U8", "I8", "F8_E4M3", "F8_E5M2") or "packed" in pat or "scale" in pat):
      quantized_packed.append(entry)

  known_dims = {int(v) for v in core_fields.values() if isinstance(v, int) and not isinstance(v, bool) and v > 0}
  shape_anomalies: list[dict[str, Any]] = []
  if known_dims:
    for item in patterns_out:
      shp = item.get("shape", [])
      if len(shp) == 1 and shp[0] not in known_dims:
        shape_anomalies.append(
            {
                "pattern": item["pattern"],
                "shape": shp,
                "note": (
                    "1D tensor length does not match any top-level integer config"
                    " dimension (check for head sub-dims or zero-padding)."
                ),
            }
        )

  return {
      "present": True,
      "dtypes_available": dtypes_available,
      "total_tensors": len(raw_tensors),
      "dtype_counts": dict(dtype_counts),
      "dominant_dtype": dominant_dtype,
      "precision_sensitive_tensors": precision_sensitive,
      "quantized_packed_tensors": quantized_packed,
      "shape_anomalies": shape_anomalies,
      "patterns": patterns_out,
  }


def _call_name(node: ast.Call) -> str:
  """Returns dotted or simple name for an AST Call node."""
  func = node.func
  if isinstance(func, ast.Name):
    return func.id
  if isinstance(func, ast.Attribute):
    return func.attr
  return ""


def _dotted_expr(node: ast.AST) -> str:
  """Returns dotted name for Name/Attribute chains."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    prefix = _dotted_expr(node.value)
    return f"{prefix}.{node.attr}" if prefix else node.attr
  return ""


def _discover_reference_py_files(hf_dir: str) -> list[str]:
  """Recursively discovers .py files under hf_dir, skipping .git/__pycache__."""
  found: list[str] = []
  for root, dirs, files in os.walk(hf_dir):
    dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache"))
    for fn in sorted(files):
      if fn.endswith(".py"):
        found.append(os.path.join(root, fn))
  return sorted(found)


def _classify_precision_site(
    site: dict[str, Any],
    router_sites: list[dict[str, Any]],
    norm_sites: list[dict[str, Any]],
    attn_sites: list[dict[str, Any]],
    mlp_hc_sites: list[dict[str, Any]],
) -> None:
  """Buckets a cast/dtype site into router, norm, attention, or MLP/HC."""
  ctx = (f"{site.get('class_name', '')}.{site.get('func_name', '')}" f" {site.get('snippet', '')}").lower()
  if any(tok in ctx for tok in ("gate", "router", "routed", "topk", "expert_score")):
    router_sites.append(site)
  elif any(tok in ctx for tok in ("rmsnorm", "layernorm", "norm", "rsqrt", "var")):
    norm_sites.append(site)
  elif any(tok in ctx for tok in ("softmax", "attn", "attention", "scores")):
    attn_sites.append(site)
  elif any(tok in ctx for tok in ("mlp", "hyper", "hc", "engram", "set_dtype", "mix", "silu")):
    mlp_hc_sites.append(site)


def scan_reference_sources(hf_dir: str) -> dict[str, Any]:
  """Scans reference .py files recursively for imports, norms, and precision."""
  py_files = _discover_reference_py_files(hf_dir)
  rel_py_files = [os.path.relpath(p, hf_dir).replace(os.sep, "/") for p in py_files]
  relative_imports: list[str] = []
  omitted_norm_eps: list[str] = []
  explicit_fp32_casts: list[str] = []
  explicit_fp64_casts: list[str] = []

  router_sites: list[dict[str, Any]] = []
  rope_sites: list[dict[str, Any]] = []
  norm_sites: list[dict[str, Any]] = []
  attn_sites: list[dict[str, Any]] = []
  mlp_hc_sites: list[dict[str, Any]] = []

  def _visit(
      node: ast.AST,
      cur_rel_path: str,
      class_name: str = "",
      func_name: str = "",
  ) -> None:
    cur_cls = class_name
    cur_fn = func_name
    if isinstance(node, ast.ClassDef):
      cur_cls = node.name
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      cur_fn = node.name
      fn_lower = f"{cur_cls}.{cur_fn}".lower()
      if any(tok in fn_lower for tok in ("freqs_cis", "inv_freq", "rotary", "rope", "yarn")):
        for sub in ast.walk(node):
          if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow):
            rope_sites.append(
                {
                    "file": cur_rel_path,
                    "line": sub.lineno,
                    "class_name": cur_cls,
                    "func_name": cur_fn,
                    "snippet": ast.unparse(sub),
                }
            )
            break

    if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
      mod = node.module or ""
      relative_imports.append(f"{cur_rel_path}:{node.lineno}: from .{mod} import ...")
    elif isinstance(node, (ast.With, ast.AsyncWith)):
      for item in node.items:
        ctx_str = ast.unparse(item.context_expr)
        if "set_dtype" in ctx_str or "autocast" in ctx_str:
          dtype = ""
          if "float64" in ctx_str or "double" in ctx_str:
            dtype = "float64"
          elif "float32" in ctx_str:
            dtype = "float32"
          if dtype:
            site = {
                "file": cur_rel_path,
                "line": node.lineno,
                "class_name": cur_cls,
                "func_name": cur_fn,
                "dtype": dtype,
                "snippet": f"with {ctx_str}:",
            }
            if dtype == "float32":
              explicit_fp32_casts.append(f"{cur_rel_path}:{node.lineno}: with {ctx_str}")
            else:
              explicit_fp64_casts.append(f"{cur_rel_path}:{node.lineno}: with {ctx_str}")
            _classify_precision_site(site, router_sites, norm_sites, attn_sites, mlp_hc_sites)
    elif isinstance(node, ast.Call):
      cname = _call_name(node)
      dotted = _dotted_expr(node.func)
      if "Norm" in cname:
        kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        if "eps" not in kw_names and "variance_epsilon" not in kw_names and len(node.args) < 2:
          omitted_norm_eps.append(f"{cur_rel_path}:{node.lineno}: {cname}(...) omits explicit eps=")

      detected_dtype = ""
      desc = ""
      if cname == "float" and isinstance(node.func, ast.Attribute):
        detected_dtype = "float32"
        desc = ".float()"
      elif cname == "double" and isinstance(node.func, ast.Attribute):
        detected_dtype = "float64"
        desc = ".double()"
      elif cname in ("to", "astype") and isinstance(node.func, ast.Attribute):
        unparsed = [ast.unparse(a) for a in node.args] + [ast.unparse(kw.value) for kw in node.keywords]
        if any("float64" in s or "double" in s for s in unparsed):
          detected_dtype = "float64"
          desc = f".{cname}(...float64...)"
        elif any("float32" in s for s in unparsed):
          detected_dtype = "float32"
          desc = f".{cname}(...float32...)"
      elif dotted in ("np.float64", "jnp.float64", "torch.float64"):
        detected_dtype = "float64"
        desc = f"{dotted}(...)"
      elif dotted in ("np.float32", "jnp.float32", "torch.float32"):
        detected_dtype = "float32"
        desc = f"{dotted}(...)"
      else:
        for kw in node.keywords:
          if kw.arg == "dtype":
            val_s = ast.unparse(kw.value)
            if "float64" in val_s:
              detected_dtype = "float64"
              desc = f"{cname}(..., dtype={val_s})"
            elif "float32" in val_s:
              detected_dtype = "float32"
              desc = f"{cname}(..., dtype={val_s})"

      if detected_dtype:
        if detected_dtype == "float32":
          explicit_fp32_casts.append(f"{cur_rel_path}:{node.lineno}: {desc}")
        else:
          explicit_fp64_casts.append(f"{cur_rel_path}:{node.lineno}: {desc}")
        site = {
            "file": cur_rel_path,
            "line": node.lineno,
            "class_name": cur_cls,
            "func_name": cur_fn,
            "dtype": detected_dtype,
            "snippet": ast.unparse(node),
        }
        _classify_precision_site(site, router_sites, norm_sites, attn_sites, mlp_hc_sites)

    for child in ast.iter_child_nodes(node):
      _visit(child, cur_rel_path, cur_cls, cur_fn)

  for py_path, rel_path in zip(py_files, rel_py_files):
    with open(py_path, "r", encoding="utf-8") as f:
      src = f.read()
    try:
      tree = ast.parse(src, filename=rel_path)
    except SyntaxError as exc:
      relative_imports.append(f"{rel_path}:{exc.lineno}: SyntaxError ({exc.msg})")
      continue

    _visit(tree, rel_path)

  precision_contract = {
      "router_gate_fp32": any(s["dtype"] == "float32" for s in router_sites),
      "router_gate_sites": router_sites,
      "rope_inv_freq_sites": rope_sites,
      "norm_fp32": any(s["dtype"] == "float32" for s in norm_sites),
      "norm_sites": norm_sites,
      "attention_softmax_fp32": any(s["dtype"] == "float32" for s in attn_sites),
      "attention_softmax_sites": attn_sites,
      "mlp_or_hc_fp32": any(s["dtype"] == "float32" for s in mlp_hc_sites),
      "mlp_or_hc_sites": mlp_hc_sites,
  }

  return {
      "python_files": rel_py_files,
      "relative_imports": relative_imports,
      "requires_synthetic_package_loader": bool(relative_imports),
      "omitted_norm_eps_calls": omitted_norm_eps,
      "explicit_fp32_casts": explicit_fp32_casts,
      "explicit_fp64_casts": explicit_fp64_casts,
      "precision_contract": precision_contract,
  }


def inspect_directory(hf_dir: str) -> dict[str, Any]:
  """Inspects hf_dir and returns the full structured artifact contract."""
  cfg_summary = summarize_config(hf_dir)
  tensor_summary = summarize_safetensors(hf_dir, cfg_summary.get("core_fields", {}))
  src_summary = scan_reference_sources(hf_dir)
  return {
      "status": "PASS",
      "hf_dir": os.path.abspath(hf_dir),
      "config_summary": cfg_summary,
      "tensor_inventory": tensor_summary,
      "source_hazards": src_summary,
      "precision_contract": src_summary["precision_contract"],
  }


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--hf-dir",
      required=True,
      help="Directory containing unmodified config.json, safetensors, and .py.",
  )
  parser.add_argument(
      "--out",
      default="",
      help="Optional path to write JSON output.",
  )
  args = parser.parse_args(argv)

  if not os.path.isdir(args.hf_dir):
    sys.stderr.write(f"Error: directory does not exist: {args.hf_dir}\n")
    return 2

  try:
    report = inspect_directory(args.hf_dir)
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    sys.stderr.write(f"Error inspecting {args.hf_dir}: {exc}\n")
    return 2

  cfg_summary = report["config_summary"]
  tensor_summary = report["tensor_inventory"]
  src_summary = report["source_hazards"]

  encoded = json.dumps(report, indent=2, sort_keys=True)
  if args.out:
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(encoded + "\n")
    brief = {
        "status": "PASS",
        "out_path": os.path.abspath(args.out),
        "nested_subconfigs": cfg_summary.get("nested_subconfigs", []),
        "derived_prod_dims": cfg_summary.get("derived_prod_dims", {}),
        "warnings": cfg_summary.get("warnings", []),
        "total_tensors": tensor_summary.get("total_tensors", 0),
        "pattern_count": len(tensor_summary.get("patterns", [])),
        "requires_synthetic_package_loader": src_summary.get("requires_synthetic_package_loader", False),
        "omitted_norm_eps_count": len(src_summary.get("omitted_norm_eps_calls", [])),
        "explicit_fp32_casts_count": len(src_summary.get("explicit_fp32_casts", [])),
        "router_gate_fp32": (report["precision_contract"].get("router_gate_fp32", False)),
    }
    sys.stdout.write(json.dumps(brief, indent=2, sort_keys=True) + "\n")
  else:
    sys.stdout.write(encoded + "\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
