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

"""Offline PyTree and Orbax checkpoint multi-pass and fake-quantization utility using Qwix rules."""

from collections.abc import Sequence
import re
from typing import Any

from etils import epath
import jax
import jax.numpy as jnp
from maxtext.layers import quantizations
import orbax.checkpoint as ocp
import qwix


def _path_to_str(key_path: tuple[Any, ...]) -> str:
  parts = []
  for k in key_path:
    if hasattr(k, "key"):
      parts.append(str(k.key))
    elif hasattr(k, "name"):
      parts.append(str(k.name))
    else:
      parts.append(str(k))
  return "/".join(parts)


def _match_rule(module_path: str, rules: Sequence[qwix.QtRule]) -> qwix.QtRule | None:
  for rule in rules:
    if rule.module_path and not re.search(rule.module_path, module_path):
      continue
    if getattr(rule, "exclude_module_path", None) and re.search(rule.exclude_module_path, module_path):
      continue
    return rule
  return None


def _transform_weight_array(weight: jax.Array, rule: qwix.QtRule) -> jax.Array:
  """Applies a single Qwix QtRule's multipass decomposition or weight_qtype quantization to a weight array."""
  orig_dtype = weight.dtype
  out = weight
  add_cfg = rule.additional_qt_config or {}
  if (mp_mode := add_cfg.get("multipass_mode")) is not None:
    mp_cfg = qwix.resolve_multipass_config(mp_mode, tile_size=getattr(rule, "tile_size", None))
    if mp_cfg is None:
      raise ValueError(f"Unsupported multipass_mode for checkpoint transformation: {mp_mode!r}")
    passes = qwix.decompose_multipass_operand(out, mp_cfg.rhs, for_lhs=False)
    out = qwix.reconstruct_multipass_operand(passes)
  elif rule.weight_qtype is not None:
    how = qwix.HowToQuantize(
        qtype=rule.weight_qtype,
        calibration_method=rule.weight_calibration_method or "absmax",
    )
    out = qwix.dequantize(qwix.quantize(out, how))
  return out.astype(orig_dtype)


def apply_qwix_rules_to_pytree(
    params: Any,
    rules: Sequence[qwix.QtRule],
    *,
    param_scan_axis: int = 0,
    num_scanned_layers: int | None = None,
) -> Any:
  """Applies Qwix QtRules (multi-pass decomposition or fake-quantization) to a PyTree."""

  def _map_leaf(key_path: tuple[Any, ...], leaf: Any) -> Any:
    if not isinstance(leaf, jax.Array) or leaf.ndim < 2 or not jnp.issubdtype(leaf.dtype, jnp.floating):
      return leaf
    path_str = _path_to_str(key_path)
    if any(skip in path_str for skip in ("scale", "bias", "embedder", "norm")):
      return leaf
    rule = _match_rule(path_str, rules)
    if rule is None:
      return leaf

    target_indices = getattr(rule, "target_scan_indices", None)
    if (
        target_indices is not None
        and leaf.ndim >= 3
        and (num_scanned_layers is None or leaf.shape[param_scan_axis] == num_scanned_layers)
    ):
      slices = []
      for idx in range(leaf.shape[param_scan_axis]):
        w_slice = jnp.take(leaf, idx, axis=param_scan_axis)
        if idx in target_indices:
          w_slice = _transform_weight_array(w_slice, rule)
        slices.append(w_slice)
      return jnp.stack(slices, axis=param_scan_axis)
    return _transform_weight_array(leaf, rule)

  return jax.tree_util.tree_map_with_path(_map_leaf, params)


def quantize_orbax_checkpoint(
    input_path: str,
    output_path: str,
    rules: Sequence[qwix.QtRule],
    *,
    param_scan_axis: int = 0,
    num_scanned_layers: int | None = None,
) -> Any:
  """Restores an Orbax PyTree checkpoint, applies Qwix rules, and saves to output_path."""
  checkpointer = ocp.PyTreeCheckpointer()
  in_dir = epath.Path(input_path)
  out_dir = epath.Path(output_path)
  restored = checkpointer.restore(in_dir)
  transformed = apply_qwix_rules_to_pytree(
      restored,
      rules,
      param_scan_axis=param_scan_axis,
      num_scanned_layers=num_scanned_layers,
  )
  if out_dir.exists():
    out_dir.rmtree()
  checkpointer.save(out_dir, transformed)
  return transformed


def quantize_checkpoint_from_config(input_path: str, output_path: str, config: Any) -> Any:
  """Builds Qwix QtRules from a MaxText config and transforms an Orbax checkpoint."""
  rules = quantizations.get_quantization_rule(config)
  if not rules:
    raise ValueError("No Qwix quantization rules configured.")
  num_scanned = getattr(config, "num_decoder_layers", None) if getattr(config, "scan_layers", False) else None
  return quantize_orbax_checkpoint(
      input_path,
      output_path,
      rules,
      param_scan_axis=getattr(config, "param_scan_axis", 0),
      num_scanned_layers=num_scanned,
  )
