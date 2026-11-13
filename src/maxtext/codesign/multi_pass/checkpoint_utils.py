# Copyright 2026 Google LLC
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

r"""Checkpoint materialization, error/SNR inspection, and metadata utilities for Multi-Pass Emulation.

Ensures that checkpoints trained or evaluated with Multi-Pass Emulation can be:
1. Materialized/baked into multi-pass reconstructed weights (`sum_k p_k`) on save,
   load, or RL rollout sync when `bake_weight_multipass_on_save=True` or
   `apply_weight_multipass_on_load=True`.
2. Inspected per-tensor and per-pass for reconstruction error (RMSE, relative L2
   error, max abs error, SNR in dB across passes $1 \dots K$).
3. Annotated with `multipass_emulation_metadata.json` so evaluation runners can
   automatically discover the checkpoint's multi-pass configuration on load.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

try:
  import numpy as np
except ImportError:  # pragma: no cover
  np = None  # type: ignore

try:
  from etils import epath
except ImportError:  # pragma: no cover
  import pathlib as epath  # type: ignore

from maxtext.codesign.multi_pass.multipass_config import (
    MultipassEmulationConfig,
    MultipassRuleConfig,
    OperandMultipassSpec,
)
from maxtext.codesign.multi_pass.multipass_kernel import (
    decompose_tensor_into_passes,
    quantize_to_dtype_spec,
    reconstruct_multipass_eager,
)

METADATA_FILENAME = "multipass_emulation_metadata.json"


def infer_weight_contracting_axes(param_path: str, tensor: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Infers `(contracting_axes, batch_axes)` for a MaxText weight parameter tensor."""
  ndim = getattr(tensor, "ndim", 0)
  if ndim <= 1:
    return ((0,), ())

  norm_path = param_path.replace(".", "/")
  leaf_name = norm_path.rsplit("/", 1)[-1]
  if leaf_name in ("value", "kernel"):
    parts = norm_path.split("/")
    leaf_name = parts[-2] if len(parts) >= 2 else leaf_name

  if (
      leaf_name in ("wi_0", "wi_1", "wo", "wi")
      and ndim == 3
      and ("routed_experts" in norm_path or "MoeBlock" in norm_path or "moe" in norm_path.lower())
  ):
    return ((1,), (0,))

  # Attention output projection in DenseGeneral has shape (num_heads, head_dim, embed_dim)
  if leaf_name in ("out", "o_proj") and ndim == 3:
    return ((0, 1), ())

  return ((0,), ())


def _match_weight_rule_for_param(
    param_path: str, multipass_config: MultipassEmulationConfig
) -> MultipassRuleConfig | None:
  """Finds the matching `MultipassRuleConfig` for a parameter path in a state tree."""
  if not multipass_config.multipass_enabled:
    return None
  norm = param_path.replace(".", "/")
  module_candidate = norm
  for suffix in ("/value", "/kernel"):
    if module_candidate.endswith(suffix):
      module_candidate = module_candidate[: -len(suffix)]

  leaf = norm.rsplit("/", 1)[-1]
  if leaf == "value" and "/" in norm:
    leaf = norm.rsplit("/", 2)[-2]
  if any(s in leaf for s in ("scale", "bias", "norm", "embedding", "per_expert_scale")):
    return None

  for op in ("dot_general", "gmm", "ragged_dot"):
    rule = multipass_config.match_rule(module_candidate, op)
    if rule is not None and rule.rhs and rule.rhs.enabled:
      return rule
    rule_full = multipass_config.match_rule(norm, op)
    if rule_full is not None and rule_full.rhs and rule_full.rhs.enabled:
      return rule_full
  return None


def apply_multipass_to_model_weights(
    model: Any,
    multipass_config: MultipassEmulationConfig | None,
) -> int:
  """Materializes multi-pass weight reconstruction (`sum_k p_k`) into an NNX model."""
  if multipass_config is None or not multipass_config.enabled:
    return 0

  modified_count = 0
  for feat in multipass_config.active_auxiliary_features:
    modified_count += feat.materialize_model_weights(model)

  if not multipass_config.multipass_enabled:
    return modified_count

  try:
    from flax import nnx  # pylint: disable=import-outside-toplevel
  except ImportError:
    return modified_count

  if not isinstance(model, nnx.Module):
    return modified_count

  for path, node in nnx.iter_graph(model):
    if isinstance(node, nnx.Param):
      path_str = "/".join(str(p) for p in path)
      val = node.value if hasattr(node, "value") else node[...]
      if val is None or getattr(val, "ndim", 0) < 2:
        continue
      rule = _match_weight_rule_for_param(path_str, multipass_config)
      if rule is None or rule.rhs is None or not rule.rhs.enabled:
        continue
      c_axes, b_axes = infer_weight_contracting_axes(path_str, val)
      mp_val = reconstruct_multipass_eager(val, rule.rhs, contracting_axes=c_axes, batch_axes=b_axes)
      if hasattr(node, "set_value"):
        node.set_value(mp_val)
      else:
        node.value = mp_val
      modified_count += 1

  return modified_count


def apply_multipass_to_pytree(
    params_pytree: Any,
    multipass_config: MultipassEmulationConfig | None,
) -> tuple[Any, int]:
  """Applies multi-pass weight reconstruction (`sum_k p_k`) to a PyTree of parameters."""
  if multipass_config is None or not multipass_config.enabled:
    return params_pytree, 0

  modified = 0
  for feat in multipass_config.active_auxiliary_features:
    params_pytree, n_aux = feat.materialize_pytree_weights(params_pytree)
    modified += n_aux

  if not multipass_config.multipass_enabled:
    return params_pytree, modified

  def _walk(node: Any, path_parts: tuple[str, ...]) -> Any:
    nonlocal modified
    if hasattr(node, "value") and hasattr(node, "replace") and not isinstance(node, Mapping):
      val = node.value
      if hasattr(val, "ndim") and val.ndim >= 2:
        path_str = "/".join(path_parts)
        rule = _match_weight_rule_for_param(path_str, multipass_config)
        if rule is not None and rule.rhs and rule.rhs.enabled:
          c_axes, b_axes = infer_weight_contracting_axes(path_str, val)
          mp_val = reconstruct_multipass_eager(val, rule.rhs, contracting_axes=c_axes, batch_axes=b_axes)
          modified += 1
          return node.replace(value=mp_val)
      return node

    if hasattr(node, "value") and not isinstance(node, Mapping) and hasattr(node.value, "ndim"):
      val = node.value
      if val.ndim >= 2:
        path_str = "/".join(path_parts)
        rule = _match_weight_rule_for_param(path_str, multipass_config)
        if rule is not None and rule.rhs and rule.rhs.enabled:
          c_axes, b_axes = infer_weight_contracting_axes(path_str, val)
          mp_val = reconstruct_multipass_eager(val, rule.rhs, contracting_axes=c_axes, batch_axes=b_axes)
          node.value = mp_val
          modified += 1
      return node

    if isinstance(node, Mapping):
      out = {}
      for k, v in node.items():
        out[k] = _walk(v, path_parts + (str(k),))
      if type(node) is not dict:  # pylint: disable=unidiomatic-typecheck
        try:
          return type(node)(out)
        except Exception:  # pylint: disable=broad-exception-caught
          return out
      return out

    if isinstance(node, (list, tuple)):
      items = [_walk(v, path_parts + (str(i),)) for i, v in enumerate(node)]
      return tuple(items) if isinstance(node, tuple) else items

    if hasattr(node, "ndim") and node.ndim >= 2:
      path_str = "/".join(path_parts)
      rule = _match_weight_rule_for_param(path_str, multipass_config)
      if rule is not None and rule.rhs and rule.rhs.enabled:
        c_axes, b_axes = infer_weight_contracting_axes(path_str, node)
        modified += 1
        return reconstruct_multipass_eager(node, rule.rhs, contracting_axes=c_axes, batch_axes=b_axes)
    return node

  new_tree = _walk(params_pytree, ())
  return new_tree, modified


def evaluate_multipass_error(
    tensor: Any,
    spec: OperandMultipassSpec,
    *,
    contracting_axes: tuple[int, ...] = (0,),
    batch_axes: tuple[int, ...] = (),
) -> dict[str, Any]:
  """Computes per-pass cumulative reconstruction error and SNR (dB) for `tensor` under `spec`."""
  arr = np.asarray(tensor, dtype=np.float32)
  total_elems = arr.size
  if total_elems == 0 or not spec.enabled:
    return {
        "num_passes": 0,
        "pass_dtypes": [],
        "relative_l2_error": 0.0,
        "max_abs_error": 0.0,
        "snr_db": float("inf"),
        "per_pass_relative_l2_error": [],
    }

  slices = decompose_tensor_into_passes(
      arr,
      spec,
      contracting_axes=contracting_axes,
      batch_axes=batch_axes,
  )
  r0 = np.asarray(quantize_to_dtype_spec(arr, spec.src_dtype), dtype=np.float32)
  norm_orig = float(np.linalg.norm(r0))
  norm_fp32 = float(np.linalg.norm(arr))
  cum = np.zeros_like(r0, dtype=np.float32)
  per_pass_rel_err: list[float] = []
  per_pass_fp32_rel_err: list[float] = []
  for sl in slices:
    cum = cum + np.asarray(sl.dequantized, dtype=np.float32)
    diff = r0 - cum
    rel_l2 = float(np.linalg.norm(diff) / norm_orig) if norm_orig > 0 else 0.0
    per_pass_rel_err.append(rel_l2)
    diff_fp32 = arr - cum
    rel_l2_fp32 = float(np.linalg.norm(diff_fp32) / norm_fp32) if norm_fp32 > 0 else 0.0
    per_pass_fp32_rel_err.append(rel_l2_fp32)

  final_diff = r0 - cum
  final_rel_l2 = per_pass_rel_err[-1] if per_pass_rel_err else 0.0
  max_abs_err = float(np.max(np.abs(final_diff)))
  snr_db = float(-20.0 * math.log10(max(final_rel_l2, 1e-16))) if final_rel_l2 > 0 else 320.0
  return {
      "num_passes": spec.num_passes,
      "src_dtype": spec.src_dtype.name,
      "pass_dtypes": spec.pass_dtype_names,
      "scale_mode": spec.scale_mode,
      "relative_l2_error": final_rel_l2,
      "fp32_relative_l2_error": (per_pass_fp32_rel_err[-1] if per_pass_fp32_rel_err else 0.0),
      "max_abs_error": max_abs_err,
      "snr_db": snr_db,
      "per_pass_relative_l2_error": per_pass_rel_err,
      "per_pass_fp32_relative_l2_error": per_pass_fp32_rel_err,
  }


def inspect_model_or_pytree_multipass(
    model_or_pytree: Any,
    multipass_config: MultipassEmulationConfig | None = None,
) -> dict[str, Any]:
  """Computes per-tensor and aggregate multi-pass reconstruction metrics for a model or PyTree."""
  tensors: list[tuple[str, Any]] = []

  try:
    from flax import nnx  # pylint: disable=import-outside-toplevel

    if isinstance(model_or_pytree, nnx.Module):
      for path, node in nnx.iter_graph(model_or_pytree):
        if isinstance(node, nnx.Param):
          val = node.value if hasattr(node, "value") else node[...]
          if val is not None and getattr(val, "ndim", 0) >= 2:
            tensors.append(("/".join(str(p) for p in path), val))
  except ImportError:
    pass

  if not tensors:

    def _collect(node: Any, path_parts: tuple[str, ...]):
      if hasattr(node, "value") and not isinstance(node, Mapping):
        val = node.value
        if hasattr(val, "ndim") and val.ndim >= 2:
          tensors.append(("/".join(path_parts), val))
        return
      if isinstance(node, Mapping):
        for k, v in node.items():
          _collect(v, path_parts + (str(k),))
        return
      if isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
          _collect(v, path_parts + (str(i),))
        return
      if hasattr(node, "ndim") and node.ndim >= 2:
        tensors.append(("/".join(path_parts), node))

    _collect(model_or_pytree, ())

  per_tensor_report: dict[str, Any] = {}
  total_params = 0
  targeted_params = 0
  rel_errors: list[float] = []

  for path_str, val in tensors:
    arr = np.asarray(val, dtype=np.float32)
    n_elems = int(arr.size)
    total_params += n_elems
    entry: dict[str, Any] = {
        "shape": list(arr.shape),
        "num_elements": n_elems,
    }
    if multipass_config is not None and multipass_config.enabled:
      rule = _match_weight_rule_for_param(path_str, multipass_config)
      if rule is not None and rule.rhs and rule.rhs.enabled:
        c_axes, b_axes = infer_weight_contracting_axes(path_str, arr)
        err_stats = evaluate_multipass_error(arr, rule.rhs, contracting_axes=c_axes, batch_axes=b_axes)
        entry.update(err_stats)
        entry["matched_rule"] = rule.name
        targeted_params += n_elems
        rel_errors.append(err_stats["relative_l2_error"])
    per_tensor_report[path_str] = entry

  mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 0.0
  return {
      "total_parameters": total_params,
      "targeted_parameters": targeted_params,
      "num_targeted_tensors": len(rel_errors),
      "mean_targeted_relative_l2_error": mean_rel_err,
      "tensors": per_tensor_report,
  }


def save_multipass_metadata(
    checkpoint_dir: str,
    multipass_config: MultipassEmulationConfig,
    extra_metadata: Mapping[str, Any] | None = None,
) -> str | None:
  """Saves `multipass_emulation_metadata.json` in a local or GCS checkpoint directory."""
  if not checkpoint_dir:
    return None
  try:
    dir_path = epath.Path(checkpoint_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    meta_path = dir_path / METADATA_FILENAME
    payload: dict[str, Any] = {
        "multipass_emulation": multipass_config.to_dict(),
    }
    if extra_metadata:
      payload["metadata"] = dict(extra_metadata)
    meta_path.write_text(json.dumps(payload, indent=2))
    return str(meta_path)
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def load_multipass_metadata(
    checkpoint_path: str,
) -> MultipassEmulationConfig | None:
  """Loads `MultipassEmulationConfig` from a local or GCS checkpoint directory or its parent dir."""
  if not checkpoint_path:
    return None
  try:
    base_path = epath.Path(checkpoint_path)
    candidates = []
    if base_path.is_dir():
      candidates.append(base_path / METADATA_FILENAME)
      parent = base_path.parent
      if str(parent) and str(parent) != str(base_path):
        candidates.append(parent / METADATA_FILENAME)
        grandparent = parent.parent
        if str(grandparent) and str(grandparent) != str(parent):
          candidates.append(grandparent / METADATA_FILENAME)
    elif base_path.is_file():
      candidates.append(base_path.parent / METADATA_FILENAME)

    for cand in candidates:
      if cand.is_file():
        try:
          data = json.loads(cand.read_text())
          return MultipassEmulationConfig.from_dict(data.get("multipass_emulation", data))
        except Exception:  # pylint: disable=broad-exception-caught
          continue
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  return None
