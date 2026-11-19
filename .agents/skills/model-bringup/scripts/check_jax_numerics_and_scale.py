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

"""Checks JAX model files and YAML configs for numerical, layout, and scale bugs.

Detects defects that pass toy float32 unit tests while failing in production:
- POST_EXP_MASK_NAN / EXP_OVERFLOW_BEFORE_WHERE: masking applied after exp/expm1
- EXPERT_3D_INIT_FAN_IN: >=3D variance-scaling init without batch_axis
- UNINITIALIZED_EMPTY_BUFFER: jnp.empty / jnp.empty_like allocation
- REF_ROUTER_FP32_MISMATCH: MoE router config/layer missing float32_gate_logits
- ROUTER_BFLOAT16_LOGITS: router gate matmul/einsum computed in bfloat16
- ROPE_INV_FREQ_LOW_PRECISION: 1D RoPE inv_freq/ramp computed in float32 not
fp64
- RMSNORM_LOW_PRECISION_VARIANCE: norm variance accumulated without float32 cast
- MISSING_SEGMENT_MASK: attention/indexer/ngram layer ignoring segment_ids
- EXPLICIT_KV_HEAD_REPEAT: jnp.repeat/tile duplicating KV heads (MED warning)
- PROD_TENSOR_OOM: symbolic intermediate shape exceeding --max-tensor-gib

Exit code: 0 = PASS, 1 = HIGH-confidence issues found, 2 = usage or I/O error.
"""

import argparse
import ast
from collections.abc import Mapping, Sequence
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

_VARIANCE_SCALING_INITS = frozenset(
    {
        "lecun_normal",
        "lecun_uniform",
        "he_normal",
        "he_uniform",
        "glorot_normal",
        "glorot_uniform",
        "xavier_normal",
        "xavier_uniform",
        "variance_scaling",
    }
)
_EXP_FUNCS = frozenset({"exp", "expm1", "exp2"})
_MASK_SELECT_FUNCS = frozenset({"where", "select", "clip", "minimum"})
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_WAIVER_RE = re.compile(r"#\s*jmp-waiver(?:-file)?:\s*([A-Za-z0-9_,\s]+)")


def _dotted_name(node: ast.AST) -> str:
  """Returns dotted attribute name or bare identifier for an AST node."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    prefix = _dotted_name(node.value)
    return f"{prefix}.{node.attr}" if prefix else node.attr
  return ""


def _tail_name(node: ast.AST) -> str:
  """Returns the rightmost identifier of a Name or Attribute node."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    return node.attr
  return ""


def _is_guarded_exp_arg(arg: ast.AST, masked_vars: frozenset[str]) -> bool:
  """Returns True if an exp() argument is masked or clamped before exp()."""
  if isinstance(arg, ast.Name) and arg.id in masked_vars:
    return True
  if isinstance(arg, ast.Call) and _tail_name(arg.func) in _MASK_SELECT_FUNCS:
    return True
  return False


def _contains_exp_call(node: ast.AST, masked_vars: frozenset[str]) -> bool:
  """Returns True if subtree contains an unguarded exp/expm1/exp2 call."""
  for sub in ast.walk(node):
    if isinstance(sub, ast.Call) and _tail_name(sub.func) in _EXP_FUNCS:
      if sub.args and _is_guarded_exp_arg(sub.args[0], masked_vars):
        continue
      return True
  return False


def _check_init_call_3d(call_node: ast.Call) -> str | None:
  """Checks outer init_fn(rng, shape_tuple) calls for >=3D missing batch_axis."""
  if not isinstance(call_node.func, ast.Call):
    return None
  inner = call_node.func
  init_name = _tail_name(inner.func)
  if init_name not in _VARIANCE_SCALING_INITS:
    return None
  kw_names = {kw.arg for kw in inner.keywords if kw.arg is not None}
  if "batch_axis" in kw_names:
    return None
  if len(call_node.args) >= 2 and isinstance(call_node.args[1], (ast.Tuple, ast.List)):
    ndim = len(call_node.args[1].elts)
    if ndim >= 3:
      return (
          f"{init_name}() called on {ndim}D shape without batch_axis;"
          " axis 0 multiplies fan_in unless batch_axis=(0,) is passed."
      )
  return None


def _parse_waivers_from_lines(
    lines: Sequence[str],
) -> tuple[set[str], dict[int, set[str]]]:
  """Extracts file-level and per-line `# jmp-waiver:` rule IDs."""
  file_waivers: set[str] = set()
  line_waivers: dict[int, set[str]] = {}
  for idx, line in enumerate(lines, 1):
    match = _WAIVER_RE.search(line)
    if not match:
      continue
    codes = {tok.strip() for tok in match.group(1).split(",") if tok.strip()}
    if "jmp-waiver-file:" in line:
      file_waivers.update(codes)
    else:
      line_waivers.setdefault(idx, set()).update(codes)
      line_waivers.setdefault(idx + 1, set()).update(codes)
  return file_waivers, line_waivers


def _is_waived(
    code: str,
    lineno: int,
    file_waivers: set[str],
    line_waivers: Mapping[int, set[str]],
    global_waivers: set[str],
) -> bool:
  """Returns True if `code` is waived globally, for the file, or on `lineno`."""
  if code in global_waivers or code in file_waivers:
    return True
  return code in line_waivers.get(lineno, set())


def _check_router_class(
    cls_node: ast.ClassDef,
    path: str,
    hf_router_fp32: bool,
) -> list[dict[str, Any]]:
  """Checks router/gate classes for bfloat16 gate score computation."""
  issues: list[dict[str, Any]] = []
  cls_lower = cls_node.name.lower()
  if not any(tok in cls_lower for tok in ("router", "gate")):
    return issues

  for item in cls_node.body:
    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
      continue
    if item.name not in ("__call__", "forward", "route", "compute_gate"):
      continue
    body_src = ast.unparse(item)
    has_contraction = any(
        tok in body_src
        for tok in (
            "einsum",
            "matmul",
            "dot_general",
            "jnp.dot",
            "@",
            "F.linear",
        )
    )
    if not has_contraction:
      continue
    has_fp32 = any(
        tok in body_src
        for tok in (
            "float32",
            "float64",
            "promote_types",
            "HIGHEST",
            "highest",
            "float32_gate_logits",
        )
    )
    if not has_fp32:
      issues.append(
          {
              "code": "ROUTER_BFLOAT16_LOGITS",
              "severity": "HIGH",
              "confidence": "HIGH",
              "file": path,
              "line": item.lineno,
              "message": (
                  f"{cls_node.name}.{item.name} computes router gate logits"
                  " without float32 promotion or Precision.HIGHEST; bfloat16"
                  " router matmuls cause top-k expert selection flips."
              ),
          }
      )
      if hf_router_fp32:
        issues.append(
            {
                "code": "REF_ROUTER_FP32_MISMATCH",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": item.lineno,
                "message": (
                    "Reference Gate casts inputs/weights to float32, but"
                    f" {cls_node.name}.{item.name} omits float32 promotion."
                ),
            }
        )
  return issues


def _check_rope_inv_freq(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
    cls_name: str,
    path: str,
) -> list[dict[str, Any]]:
  """Checks 1D rotary inverse-frequency functions for float32 double-rounding."""
  issues: list[dict[str, Any]] = []
  fn_lower = f"{cls_name}.{fn_node.name}".lower()
  is_rope_fn = fn_node.name in (
      "inv_freq",
      "_compute_inv_freq",
      "precompute_freqs_cis",
  ) or (
      any(tok in fn_lower for tok in ("rope", "rotary", "yarn", "inv_freq"))
      and any(tok in fn_lower for tok in ("inv_freq", "freq", "timescale", "init", "setup", "call"))
  )
  if not is_rope_fn:
    return issues

  # Skip 2D spatial vision RoPE functions taking (n_h, n_w) or (grid_h, grid_w).
  arg_names = {a.arg for a in fn_node.args.args}
  if {"n_h", "n_w"}.issubset(arg_names) or {"grid_h", "grid_w"}.issubset(arg_names):
    return issues

  has_pow = any(isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow) for sub in ast.walk(fn_node))
  if not has_pow:
    return issues

  # Inspect statements before the return statement (or inside the power/ramp
  # computation) for float32 arange/casts or missing float64.
  body_src = ast.unparse(fn_node)
  has_f64 = "float64" in body_src

  # Check if jnp.arange is used or if float32 is used inside arange/pow/ramp.
  uses_jnp_arange = False
  uses_f32_in_arange_or_pow = False
  first_bad_line = fn_node.lineno
  for sub in ast.walk(fn_node):
    if isinstance(sub, ast.Call):
      dotted = _dotted_name(sub.func)
      if dotted in ("jnp.arange", "jax.numpy.arange"):
        uses_jnp_arange = True
        first_bad_line = sub.lineno
      elif dotted in ("np.arange", "jnp.arange", "jax.numpy.arange"):
        call_src = ast.unparse(sub)
        if "float32" in call_src or "bfloat16" in call_src:
          uses_f32_in_arange_or_pow = True
          first_bad_line = sub.lineno
    elif isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow):
      pow_src = ast.unparse(sub)
      if "float32" in pow_src or "bfloat16" in pow_src:
        uses_f32_in_arange_or_pow = True
        first_bad_line = sub.lineno

  if not has_f64 or uses_jnp_arange or uses_f32_in_arange_or_pow:
    issues.append(
        {
            "code": "ROPE_INV_FREQ_LOW_PRECISION",
            "severity": "HIGH",
            "confidence": "HIGH",
            "file": path,
            "line": first_bad_line,
            "message": (
                f"{cls_name + '.' if cls_name else ''}{fn_node.name} computes"
                " rotary inverse frequencies or YaRN ramp in float32 / device"
                " jnp.arange; precompute exponents, powers, and ramp in host"
                " np.float64 before casting once to jnp.float32."
            ),
        }
    )
  return issues


def _check_rmsnorm_precision(cls_node: ast.ClassDef, path: str) -> list[dict[str, Any]]:
  """Checks RMSNorm/LayerNorm classes for float32 variance accumulation."""
  issues: list[dict[str, Any]] = []
  if not any(tok in cls_node.name.lower() for tok in ("rmsnorm", "layernorm")):
    return issues
  for item in cls_node.body:
    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
      continue
    if item.name not in ("__call__", "forward"):
      continue
    body_src = ast.unparse(item)
    if any(tok in body_src for tok in ("mean(", "square(", "rsqrt(", "** 2")):
      if not any(tok in body_src for tok in ("float32", "promote_types", ".float()", ".double()")):
        issues.append(
            {
                "code": "RMSNORM_LOW_PRECISION_VARIANCE",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": item.lineno,
                "message": (
                    f"{cls_node.name}.{item.name} computes normalization variance"
                    " without upcasting activations to float32."
                ),
            }
        )
  return issues


def _check_missing_segment_mask(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    cls_name: str,
    path: str,
) -> list[dict[str, Any]]:
  """Checks attention, indexer, and n-gram layers for segment_ids masking."""
  issues: list[dict[str, Any]] = []
  target_cls = any(
      tok in cls_name.lower()
      for tok in (
          "attention",
          "indexer",
          "ngram",
          "engram",
          "decoderlayer",
          "block",
      )
  )
  target_fn = any(tok in node.name.lower() for tok in ("attention", "indexer", "ngram", "engram"))
  if not target_cls and not target_fn:
    return issues
  if cls_name and node.name not in ("__call__", "forward"):
    return issues

  arg_names = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
  seg_params = [a for a in arg_names if "segment_id" in a or a in ("seg_ids", "seg")]
  body_src = "\n".join(ast.unparse(stmt) for stmt in node.body)

  builds_pos_or_shift_mask = any(
      tok in body_src
      for tok in (
          "window_size",
          "compressed_visible_len",
          "for shift in range",
          "delta >= 0",
          "idx[:, None] >= idx[None, :]",
          "self.self_attention(",
          "self.engram(",
      )
  )
  if not builds_pos_or_shift_mask:
    return issues

  has_seg_in_body = any(re.search(rf"\b{re.escape(sp)}\b", body_src) for sp in seg_params)
  if not seg_params or not has_seg_in_body:
    issues.append(
        {
            "code": "MISSING_SEGMENT_MASK",
            "aliases": ["CAUSAL_MASK_IGNORES_SEGMENT_IDS"],
            "severity": "HIGH",
            "confidence": "HIGH",
            "file": path,
            "line": node.lineno,
            "message": (
                f"{cls_name + '.' if cls_name else ''}{node.name} builds a"
                " causal/window mask or n-gram shift without masking cross-segment"
                " and padding (decoder_segment_ids == 0) positions."
            ),
        }
    )
  return issues


def scan_yaml_file(
    path: str,
    *,
    hf_contract: Mapping[str, Any] | None = None,
    global_waivers: set[str] | None = None,
) -> list[dict[str, Any]]:
  """Scans a model .yml/.yaml file for MoE router precision mismatches."""
  if os.path.basename(path) == "base.yml":
    return []
  with open(path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

  file_waivers, line_waivers = _parse_waivers_from_lines(lines)
  g_waivers = global_waivers or set()

  kv_map: dict[str, tuple[str, int]] = {}
  for lineno, raw_line in enumerate(lines, 1):
    stripped = raw_line.split("#", 1)[0].strip()
    if not stripped or ":" not in stripped or stripped.startswith("-"):
      continue
    k, v = stripped.split(":", 1)
    key = k.strip()
    val = v.strip().strip("\"'")
    kv_map[key] = (val, lineno)

  hf_router_fp32 = False
  if hf_contract:
    pc = hf_contract.get("precision_contract", {})
    if isinstance(pc, Mapping):
      hf_router_fp32 = bool(pc.get("router_gate_fp32", False))

  routed_score_val, routed_score_line = kv_map.get("routed_score_func", ("", 0))
  num_experts_val, num_experts_line = kv_map.get("num_experts", kv_map.get("n_routed_experts", ("0", 0)))
  try:
    num_experts_int = int(num_experts_val)
  except ValueError:
    num_experts_int = 0

  is_moe_config = bool(routed_score_val) or num_experts_int > 1 or hf_router_fp32
  if not is_moe_config:
    return []

  issues: list[dict[str, Any]] = []
  if "float32_gate_logits" in kv_map:
    f32_val, f32_line = kv_map["float32_gate_logits"]
    if f32_val.lower() in ("false", "no", "0"):
      if not _is_waived(
          "REF_ROUTER_FP32_MISMATCH",
          f32_line,
          file_waivers,
          line_waivers,
          g_waivers,
      ):
        issues.append(
            {
                "code": "REF_ROUTER_FP32_MISMATCH",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": f32_line,
                "message": (
                    f"{path}:{f32_line} sets `float32_gate_logits: false` on an"
                    " MoE model config; reference Gate casts inputs/weights to"
                    " float32 (`float32_gate_logits: true`)."
                ),
            }
        )
  else:
    anchor_line = routed_score_line or num_experts_line or 1
    if not _is_waived(
        "REF_ROUTER_FP32_MISMATCH",
        anchor_line,
        file_waivers,
        line_waivers,
        g_waivers,
    ):
      issues.append(
          {
              "code": "REF_ROUTER_FP32_MISMATCH",
              "severity": "HIGH",
              "confidence": "HIGH",
              "file": path,
              "line": anchor_line,
              "message": (
                  f"{path}:{anchor_line} configures MoE routing without setting"
                  " `float32_gate_logits: true`; inherited `false` defaults compute"
                  " router logits in bfloat16."
              ),
          }
      )
  return issues


def _is_vendored_ref_file(path: str) -> bool:
  """Returns True if path lives inside a provenance.json-pinned reference dir."""
  cur = os.path.dirname(os.path.abspath(path))
  for _ in range(3):
    if os.path.isfile(os.path.join(cur, "provenance.json")):
      return True
    parent = os.path.dirname(cur)
    if parent == cur:
      break
    cur = parent
  return False


# pylint: disable=too-many-nested-blocks
def scan_python_file(
    path: str,
    *,
    shared_legacy: bool = False,
    hf_contract: Mapping[str, Any] | None = None,
    global_waivers: set[str] | None = None,
) -> list[dict[str, Any]]:
  """Scans a single Python file and returns detected hazard dictionaries."""
  del shared_legacy  # Kept for CLI backwards compatibility.
  if _is_vendored_ref_file(path):
    return []
  with open(path, "r", encoding="utf-8") as f:
    src = f.read()
  lines = src.splitlines()
  file_waivers, line_waivers = _parse_waivers_from_lines(lines)
  g_waivers = global_waivers or set()

  tree = ast.parse(src, filename=path)
  is_test_file = os.path.basename(path).endswith(("_test.py", "_tests.py")) or (
      f"{os.sep}tests{os.sep}" in os.path.abspath(path)
  )

  hf_router_fp32 = False
  if hf_contract:
    pc = hf_contract.get("precision_contract", {})
    if isinstance(pc, Mapping):
      hf_router_fp32 = bool(pc.get("router_gate_fp32", False))

  # Track local variable bindings `init_fn = nnx.initializers.lecun_normal()`
  # and pre-masked exp arguments `safe_x = jnp.where(mask, x, 0.0)`.
  unbatched_init_vars: dict[str, str] = {}
  masked_vars: set[str] = set()
  unmasked_exp_vars: dict[str, int] = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
      cname = _tail_name(node.value.func)
      if cname in _VARIANCE_SCALING_INITS:
        kw_names = {kw.arg for kw in node.value.keywords if kw.arg is not None}
        if "batch_axis" not in kw_names:
          for tgt in node.targets:
            if isinstance(tgt, ast.Name):
              unbatched_init_vars[tgt.id] = cname
      elif cname in _MASK_SELECT_FUNCS:
        for tgt in node.targets:
          if isinstance(tgt, ast.Name):
            masked_vars.add(tgt.id)
      elif cname in _EXP_FUNCS:
        if not (node.value.args and _is_guarded_exp_arg(node.value.args[0], frozenset(masked_vars))):
          for tgt in node.targets:
            if isinstance(tgt, ast.Name):
              unmasked_exp_vars[tgt.id] = node.lineno

  guarded = frozenset(masked_vars)
  raw_issues: list[dict[str, Any]] = []

  # Structural class and function checks (skip unit test files and tests/
  # utilities where negative-control regressions are intentionally constructed).
  if not is_test_file:
    for top_node in tree.body:
      if isinstance(top_node, ast.ClassDef):
        raw_issues.extend(_check_router_class(top_node, path, hf_router_fp32))
        raw_issues.extend(_check_rmsnorm_precision(top_node, path))
        for item in top_node.body:
          if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raw_issues.extend(_check_rope_inv_freq(item, top_node.name, path))
            raw_issues.extend(_check_missing_segment_mask(item, top_node.name, path))
      elif isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raw_issues.extend(_check_rope_inv_freq(top_node, "", path))
        raw_issues.extend(_check_missing_segment_mask(top_node, "", path))

  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      tail = _tail_name(node.func)
      dotted = _dotted_name(node.func)

      # 1. Post-exp masking hazard: jnp.where(mask, jnp.exp(x), 0.0).
      if not is_test_file and tail in ("where", "select") and len(node.args) >= 3:
        inline_exp = _contains_exp_call(node.args[1], guarded) or _contains_exp_call(node.args[2], guarded)
        var_exp = (isinstance(node.args[1], ast.Name) and node.args[1].id in unmasked_exp_vars) or (
            isinstance(node.args[2], ast.Name) and node.args[2].id in unmasked_exp_vars
        )
        if inline_exp or var_exp:
          raw_issues.append(
              {
                  "code": "POST_EXP_MASK_NAN",
                  "aliases": ["EXP_OVERFLOW_BEFORE_WHERE"],
                  "severity": "HIGH",
                  "confidence": "HIGH",
                  "file": path,
                  "line": node.lineno,
                  "message": (
                      f"{dotted} wraps an unmasked exp() call; masked overflow"
                      " produces 0.0 * inf = NaN in backward VJP. Mask the input"
                      " to exp() first."
                  ),
              }
          )

      # 2. >=3D variance scaling init without batch_axis.
      msg_3d = _check_init_call_3d(node)
      if msg_3d:
        raw_issues.append(
            {
                "code": "EXPERT_3D_INIT_FAN_IN",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": node.lineno,
                "message": msg_3d,
            }
        )
      elif (
          isinstance(node.func, ast.Name)
          and node.func.id in unbatched_init_vars
          and len(node.args) >= 2
          and isinstance(node.args[1], (ast.Tuple, ast.List))
          and len(node.args[1].elts) >= 3
      ):
        init_name = unbatched_init_vars[node.func.id]
        ndim = len(node.args[1].elts)
        raw_issues.append(
            {
                "code": "EXPERT_3D_INIT_FAN_IN",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": node.lineno,
                "message": (
                    f"{node.func.id} ({init_name}) called on {ndim}D shape without"
                    " batch_axis; pass batch_axis=(0,) for stacked experts."
                ),
            }
        )

      # 3. Uninitialized jnp.empty / jnp.empty_like buffers.
      if tail in ("empty", "empty_like") and dotted.startswith(("jnp.", "jax.numpy.")):
        raw_issues.append(
            {
                "code": "UNINITIALIZED_EMPTY_BUFFER",
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": path,
                "line": node.lineno,
                "message": (
                    f"{dotted} allocates uninitialized memory in JAX >= 0.11;" " use jnp.zeros / jnp.zeros_like."
                ),
            }
        )

      # 6. GateLogit(...) with float32_gate_logits dtype but unpromoted
      # matmul_precision.
      if tail == "GateLogit":
        kw_map = {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg is not None}
        dtype_s = kw_map.get("dtype", "")
        prec_s = kw_map.get("matmul_precision", "")
        if "float32_gate_logits" in dtype_s and (
            not prec_s or ("highest" not in prec_s.lower() and "float32_gate_logits" not in prec_s)
        ):
          raw_issues.append(
              {
                  "code": "REF_ROUTER_FP32_MISMATCH",
                  "severity": "HIGH",
                  "confidence": "HIGH",
                  "file": path,
                  "line": node.lineno,
                  "message": (
                      "GateLogit sets float32 dtype via float32_gate_logits"
                      " without promoting matmul_precision to 'highest'."
                  ),
              }
          )

      # 7. Explicit KV head repeat (medium-confidence warning).
      if tail in ("repeat", "tile") and dotted.startswith(("jnp.", "jax.numpy.")):
        call_s = ast.unparse(node)
        if any(tok in call_s for tok in ("kv_head", "num_key_value_heads", "n_rep", "num_Groups")):
          raw_issues.append(
              {
                  "code": "EXPLICIT_KV_HEAD_REPEAT",
                  "severity": "MED",
                  "confidence": "MEDIUM",
                  "file": path,
                  "line": node.lineno,
                  "message": (
                      f"{dotted} explicitly duplicates KV heads in memory; prefer"
                      " grouped einsum/dot_general indexing where supported."
                  ),
              }
          )

  issues: list[dict[str, Any]] = []
  for item in raw_issues:
    if not _is_waived(
        item["code"],
        int(item["line"]),
        file_waivers,
        line_waivers,
        g_waivers,
    ):
      issues.append(item)
  return issues


def _parse_prod_dims(spec: str) -> dict[str, int]:
  """Parses B=1,S=8192,H=64 into a symbol dictionary."""
  dims: dict[str, int] = {}
  if not spec.strip():
    return dims
  for item in spec.split(","):
    item = item.strip()
    if not item:
      continue
    if "=" not in item:
      raise ValueError(f"Expected KEY=INT in --prod-dims, got {item!r}")
    k, v = item.split("=", 1)
    dims[k.strip()] = int(v.strip())
  if "S" in dims and "C" in dims and dims["C"] > 0 and "N_chunks" not in dims:
    dims["N_chunks"] = (dims["S"] + dims["C"] - 1) // dims["C"]
  return dims


def _eval_dim_expr(node: ast.AST, dims: Mapping[str, int]) -> int:
  """Safely evaluates an integer arithmetic AST over production dimensions."""
  if isinstance(node, ast.Constant) and isinstance(node.value, int):
    return int(node.value)
  if isinstance(node, ast.Name):
    if node.id not in dims:
      raise ValueError(f"Unknown dimension symbol {node.id!r}")
    return dims[node.id]
  if isinstance(node, ast.BinOp):
    left = _eval_dim_expr(node.left, dims)
    right = _eval_dim_expr(node.right, dims)
    if isinstance(node.op, ast.Mult):
      return left * right
    if isinstance(node.op, ast.Add):
      return left + right
    if isinstance(node.op, ast.Sub):
      return left - right
    if isinstance(node.op, ast.FloorDiv):
      return left // right
    if isinstance(node.op, ast.Pow):
      return int(left**right)
  raise ValueError(f"Unsupported shape expression: {ast.unparse(node)}")


def evaluate_shape_expressions(
    exprs: Sequence[str],
    dims: Mapping[str, int],
    max_gib: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Evaluates NAME=EXPR[:DTYPE_BYTES] strings against max_gib."""
  estimates: list[dict[str, Any]] = []
  issues: list[dict[str, Any]] = []
  for raw in exprs:
    if "=" not in raw:
      raise ValueError(f"Expected NAME=EXPR[:BYTES] in --shape-expr, got {raw!r}")
    name, rhs = raw.split("=", 1)
    dtype_bytes = 4
    if ":" in rhs:
      rhs, b_str = rhs.rsplit(":", 1)
      dtype_bytes = int(b_str)
    elements = _eval_dim_expr(ast.parse(rhs.strip(), mode="eval").body, dims)
    gib = (elements * dtype_bytes) / float(1024**3)
    rec = {
        "name": name.strip(),
        "expr": rhs.strip(),
        "dtype_bytes": dtype_bytes,
        "elements": elements,
        "gib": round(gib, 4),
    }
    estimates.append(rec)
    if gib > max_gib:
      issues.append(
          {
              "code": "PROD_TENSOR_OOM",
              "severity": "HIGH",
              "confidence": "HIGH",
              "file": "<shape-expr>",
              "line": 0,
              "message": (
                  f"Intermediate tensor {name.strip()} ({rhs.strip()}) requires"
                  f" {gib:.2f} GiB > {max_gib:.2f} GiB at production dims"
                  " (move inside @jax.checkpoint lax.scan body)."
              ),
          }
      )
  return estimates, issues


def collect_py_files(paths: Sequence[str]) -> list[str]:
  """Expands files and directories into sorted .py file paths."""
  found: list[str] = []
  for p in paths:
    if os.path.isfile(p):
      if p.endswith(".py"):
        found.append(p)
    elif os.path.isdir(p):
      for root, dirs, files in os.walk(p):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache"))
        for fn in sorted(files):
          if fn.endswith(".py"):
            found.append(os.path.join(root, fn))
    else:
      raise FileNotFoundError(f"Path not found: {p}")
  return sorted(set(found))


def collect_yaml_files(paths: Sequence[str]) -> list[str]:
  """Expands files and directories into sorted .yml/.yaml file paths."""
  found: list[str] = []
  for p in paths:
    if os.path.isfile(p):
      if p.endswith((".yml", ".yaml")):
        found.append(p)
    elif os.path.isdir(p):
      for root, dirs, files in os.walk(p):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache"))
        for fn in sorted(files):
          if fn.endswith((".yml", ".yaml")):
            found.append(os.path.join(root, fn))
    else:
      raise FileNotFoundError(f"Path not found: {p}")
  return sorted(set(found))


def _load_hf_contract(hf_contract_path: str, hf_dir: str) -> dict[str, Any] | None:
  """Loads hf_contract JSON from path or runs inspect_hf_artifacts on hf_dir."""
  if hf_contract_path:
    with open(hf_contract_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    if isinstance(data, dict):
      return data
  if hf_dir:
    inspector_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspect_hf_artifacts.py")
    spec = importlib.util.spec_from_file_location("inspect_hf_artifacts", inspector_py)
    if spec is not None and spec.loader is not None:
      mod: Any = importlib.util.module_from_spec(spec)
      sys.modules[spec.name] = mod
      try:
        spec.loader.exec_module(mod)
        return dict(mod.inspect_directory(hf_dir))
      finally:
        sys.modules.pop(spec.name, None)
  return None


def _compute_changed_lines(
    repo_root_arg: str,
    fallback_dir: str,
    base_ref: str,
    files: Sequence[str],
) -> dict[str, set[int]]:
  """Computes added/modified line numbers per file vs `git merge-base <base_ref> HEAD`."""
  git_bin = shutil.which("git")
  if not git_bin:
    raise ValueError("git binary not found in PATH for --base diff scoping.")

  probe_dir = os.path.abspath(repo_root_arg or fallback_dir)
  if os.path.isfile(probe_dir):
    probe_dir = os.path.dirname(probe_dir)

  top_proc = subprocess.run(
      [git_bin, "-c", "safe.directory=*", "-C", probe_dir, "rev-parse", "--show-toplevel"],
      capture_output=True,
      text=True,
      check=False,
  )
  if top_proc.returncode != 0 or not top_proc.stdout.strip():
    err = top_proc.stderr.strip() or f"{probe_dir!r} is not a git repository"
    raise ValueError(f"Failed to resolve git repository root for --base: {err}")
  root_abs = os.path.abspath(top_proc.stdout.strip())

  mb_proc = subprocess.run(
      [git_bin, "-c", "safe.directory=*", "-C", root_abs, "merge-base", base_ref, "HEAD"],
      capture_output=True,
      text=True,
      check=False,
  )
  if mb_proc.returncode != 0 or not mb_proc.stdout.strip():
    err = mb_proc.stderr.strip() or f"invalid git base ref {base_ref!r}"
    raise ValueError(f"Failed to resolve git merge-base for --base={base_ref!r}: {err}")
  merge_base = mb_proc.stdout.strip()

  rel_files = [os.path.relpath(os.path.abspath(f), root_abs) for f in files if os.path.exists(f)]
  if not rel_files:
    return {}

  cmd = [git_bin, "-c", "safe.directory=*", "-C", root_abs, "diff", "-U0", "--no-color", merge_base, "--"] + rel_files
  proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    raise ValueError(f"git diff against merge-base {merge_base} failed: {proc.stderr.strip()}")

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


def _parse_global_waivers(raw: str) -> set[str]:
  """Parses comma-separated rule IDs or a JSON list/object of waivers."""
  if not raw.strip():
    return set()
  if os.path.isfile(raw):
    with open(raw, "r", encoding="utf-8") as f:
      parsed = json.load(f)
    if isinstance(parsed, list):
      return {str(x).strip() for x in parsed}
    if isinstance(parsed, dict):
      return {str(k).strip() for k in parsed.keys()}
  return {tok.strip() for tok in raw.split(",") if tok.strip()}


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "paths",
      nargs="*",
      help="Python (.py) or config (.yml/.yaml) files or directories to scan.",
  )
  parser.add_argument(
      "--config-yml",
      action="append",
      default=[],
      help="Additional .yml/.yaml model config files to scan.",
  )
  parser.add_argument(
      "--base",
      default="",
      help="Optional git ref to scope findings to modified lines in the diff.",
  )
  parser.add_argument(
      "--repo-root",
      default="",
      help="Repository root directory for --base git diff.",
  )
  parser.add_argument(
      "--all-lines",
      action="store_true",
      help="Report issues on all lines even when --base is set.",
  )
  parser.add_argument(
      "--hf-contract",
      default="",
      help="Path to JSON output from inspect_hf_artifacts.py.",
  )
  parser.add_argument(
      "--hf-dir",
      default="",
      help="Directory of unmodified HF reference artifacts.",
  )
  parser.add_argument(
      "--waivers",
      default="",
      help="Comma-separated rule IDs or JSON waiver file path.",
  )
  parser.add_argument(
      "--shared-legacy",
      action="store_true",
      help="Allow getattr/get config sentinels in legacy shared modules.",
  )
  parser.add_argument(
      "--prod-dims",
      default="",
      help="Comma-separated DIM=INT pairs (overrides --hf-contract dims).",
  )
  parser.add_argument(
      "--shape-expr",
      action="append",
      default=[],
      help="Symbolic tensor expression NAME=EXPR[:DTYPE_BYTES] to check.",
  )
  parser.add_argument(
      "--max-tensor-gib",
      type=float,
      default=4.0,
      help="Maximum allowed single per-layer intermediate size in GiB.",
  )
  parser.add_argument(
      "--fail-on-warn",
      action="store_true",
      help="Also fail exit code on MED/LOW or MEDIUM-confidence warnings.",
  )
  args = parser.parse_args(argv)

  all_input_paths = list(args.paths) + list(args.config_yml)
  if not all_input_paths and not args.shape_expr:
    parser.error("Provide at least one path, --config-yml, or --shape-expr.")

  try:
    hf_contract = _load_hf_contract(args.hf_contract, args.hf_dir)
    global_waivers = _parse_global_waivers(args.waivers)
    py_files = collect_py_files(all_input_paths)
    yml_files = collect_yaml_files(all_input_paths)

    dims: dict[str, int] = {}
    if hf_contract:
      cfg_sum = hf_contract.get("config_summary", {})
      if isinstance(cfg_sum, Mapping):
        derived = cfg_sum.get("derived_prod_dims", {})
        if isinstance(derived, Mapping):
          for k, v in derived.items():
            if isinstance(v, int):
              dims[str(k)] = int(v)
    dims.update(_parse_prod_dims(args.prod_dims))
    if "S" in dims and "C" in dims and dims["C"] > 0:
      dims["N_chunks"] = (dims["S"] + dims["C"] - 1) // dims["C"]

    all_issues: list[dict[str, Any]] = []
    for path in py_files:
      all_issues.extend(
          scan_python_file(
              path,
              shared_legacy=args.shared_legacy,
              hf_contract=hf_contract,
              global_waivers=global_waivers,
          )
      )
    for yml_path in yml_files:
      all_issues.extend(
          scan_yaml_file(
              yml_path,
              hf_contract=hf_contract,
              global_waivers=global_waivers,
          )
      )

    if args.base and not args.all_lines:
      fallback_dir = os.path.dirname(os.path.abspath(all_input_paths[0])) if all_input_paths else os.getcwd()
      changed_map = _compute_changed_lines(args.repo_root, fallback_dir, args.base, py_files + yml_files)
      all_issues = [
          iss
          for iss in all_issues
          if iss["file"] == "<shape-expr>" or int(iss["line"]) in changed_map.get(os.path.abspath(iss["file"]), set())
      ]

    shape_estimates, shape_issues = evaluate_shape_expressions(args.shape_expr, dims, args.max_tensor_gib)
    all_issues.extend(shape_issues)
  except (OSError, RuntimeError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
    sys.stderr.write(f"Error: {exc}\n")
    return 2

  for iss in all_issues:
    iss.setdefault("severity", "HIGH")
    iss.setdefault("confidence", "HIGH")

  high_conf_issues = [iss for iss in all_issues if iss.get("severity") == "HIGH" and iss.get("confidence") == "HIGH"]
  failing_issues = all_issues if args.fail_on_warn else high_conf_issues
  status = "FAIL" if failing_issues else "PASS"
  report = {
      "status": status,
      "files_scanned": len(py_files) + len(yml_files),
      "prod_dims": dims,
      "shape_estimates": shape_estimates,
      "issue_count": len(all_issues),
      "high_confidence_issues": len(high_conf_issues),
      "issues": all_issues,
  }
  sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return 0 if status == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
