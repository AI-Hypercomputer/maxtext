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

"""Utilities for QK-Clip (Muon Clip)."""

import jax
import jax.numpy as jnp
from flax import nnx
from maxtext.utils import muon_utils


def _get_key_name(k):
  """Helper to unwrap JAX path keys."""
  if hasattr(k, "key"):
    return k.key
  if hasattr(k, "idx"):
    return k.idx
  return k


def calculate_max_logit_metric(intermediate_outputs):
  """Extracts and computes the global maximum logit from intermediate outputs.

  Recognizes two shapes: Linen sow stores `(array,)` so the leaf path ends in
  `max_logits, 0`; NNX `nnx.Intermediate(array)` stores the array directly so
  the leaf path ends in `max_logits`.

  Returns the global max scalar, or None if no logits were found.
  """
  all_max_logits = []

  def extract_logits(path, val):
    if not path:
      return
    last_key = _get_key_name(path[-1])
    parent_key = _get_key_name(path[-2]) if len(path) >= 2 else None
    if last_key == "max_logits" or parent_key == "max_logits":
      all_max_logits.append(val)

  jax.tree_util.tree_map_with_path(extract_logits, intermediate_outputs)

  if not all_max_logits:
    return None

  return jnp.max(jnp.stack([jnp.max(x) for x in all_max_logits]))


def _check_attention_type(config):
  if getattr(config, "attention_type", None) != "mla":
    raise ValueError(
        f"QK-Clip is only supported for MLA attention (attention_type='mla'). "
        f"Current configuration: {getattr(config, 'attention_type', 'None')}"
    )


def _max_logits_at(curr):
  """Read max_logits from a node in the intermediates tree.

  Returns the [batch, num_heads] array, or None if not present. Handles both
  the Linen sow shape (`{"max_logits": (array,)}`) and the NNX shape
  (`{"max_logits": array}` or `{"attention_op": {"max_logits": array}}`).
  """
  if not isinstance(curr, dict):
    return None
  ml = curr.get("max_logits")
  if ml is None and "attention_op" in curr and isinstance(curr["attention_op"], dict):
    ml = curr["attention_op"].get("max_logits")
  if ml is None:
    return None
  if isinstance(ml, (tuple, list)):
    return ml[0] if ml else None
  return ml


def _scale_from_max_logits(max_logits_batch, tau, *, preserve_scan_axis: bool = False):
  start_axis = 1 if preserve_scan_axis else 0
  axes = tuple(range(start_axis, max_logits_batch.ndim - 1))
  s_max = jnp.max(max_logits_batch, axis=axes) if axes else max_logits_batch
  return jnp.minimum(1.0, tau / (s_max + 1e-6))


def _resolve_scan_axis(param, max_logits_batch, config, param_var=None):
  """Returns the scan axis index on `param` if `max_logits_batch` is layer-stacked, else None."""
  if param.ndim < 4 or max_logits_batch.ndim < 3:
    return None
  if param_var is not None:
    scan_axes = muon_utils.get_scan_axes(param_var)
    if len(scan_axes) == 1 and param.shape[scan_axes[0]] == max_logits_batch.shape[0]:
      return scan_axes[0]
  psa = getattr(config, "param_scan_axis", None)
  if isinstance(psa, int) and not isinstance(psa, bool) and 0 <= psa < param.ndim - 2:
    if param.shape[psa] == max_logits_batch.shape[0]:
      return psa
  return None


def _clip_mla_weight(layer_name, param, scale, qk_nope, scan_axis=None):
  """Apply the per-head (and per-scanned-layer when scanned) scale to a wq_b or wkv_b kernel."""
  if scan_axis is not None and scale.ndim == 2:
    heads_axis = param.ndim - 2
    bcast_shape = [1] * param.ndim
    bcast_shape[scan_axis] = scale.shape[0]
    bcast_shape[heads_axis] = scale.shape[1]
    if scan_axis > heads_axis:
      scale = jnp.swapaxes(scale, 0, 1)
    scale_b = scale.reshape(bcast_shape)
  else:
    scale_b = jnp.expand_dims(scale, axis=-1)  # broadcasts over [..., rank, heads, dim]
  head = param[..., :qk_nope]
  tail = param[..., qk_nope:]
  head_new = head * jnp.sqrt(scale_b)
  if layer_name == "wq_b":
    tail_new = tail * scale_b
  else:  # wkv_b: tail is the V slice, untouched
    tail_new = tail
  return jnp.concatenate([head_new, tail_new], axis=-1)


def apply_qk_clip(state, intermediate_outputs, config):
  """Applies QK-Clip to MLA weights based on max_logits (Linen path).

  Returns a new TrainState with `wq_b`/`wkv_b` kernels rescaled per-head.
  """
  _check_attention_type(config)
  tau = float(config.qk_clip_threshold)

  def clip_mla_weights(path, param):
    if len(path) < 2:
      return param
    layer_name = _get_key_name(path[-2])
    if layer_name not in ("wq_b", "wkv_b"):
      return param

    curr = intermediate_outputs.get("intermediates", intermediate_outputs)
    for node in path[:-2]:
      key = _get_key_name(node)
      if isinstance(curr, dict) and key in curr:
        curr = curr[key]
      else:
        return param

    max_logits_batch = _max_logits_at(curr)
    if max_logits_batch is None:
      return param

    scan_axis = _resolve_scan_axis(param, max_logits_batch, config)
    scale = _scale_from_max_logits(max_logits_batch, tau, preserve_scan_axis=scan_axis is not None)
    return _clip_mla_weight(layer_name, param, scale, config.qk_nope_head_dim, scan_axis=scan_axis)

  new_params = jax.tree_util.tree_map_with_path(clip_mla_weights, state.params)
  return state.replace(params=new_params)


def apply_qk_clip_nnx(state, intermediate_outputs, config):
  """Applies QK-Clip to MLA weights on an NNX TrainStateNNX.

  `state.model` is mutated in place (NNX modules are mutable). Returns `state`
  so call sites can use the same `new_state = apply_qk_clip(...)` pattern as
  the Linen path.

  The intermediates tree mirrors the NNX module hierarchy, so `max_logits`
  sowed by `AttentionOp` lives at `...self_attention.attention_op.max_logits`.
  We accept either that shape or `...self_attention.max_logits` (matching the
  Linen-side fixtures and small-test setups).
  """
  _check_attention_type(config)
  tau = float(config.qk_clip_threshold)

  _, params_state, _ = nnx.split(state.model, nnx.Param, ...)

  def clip_mla_weights(path, param_var):
    if len(path) < 2:
      return param_var
    layer_name = _get_key_name(path[-2])
    if layer_name not in ("wq_b", "wkv_b"):
      return param_var

    curr = intermediate_outputs
    for node in path[:-2]:
      key = _get_key_name(node)
      if isinstance(curr, dict) and key in curr:
        curr = curr[key]
      else:
        return param_var

    max_logits_batch = _max_logits_at(curr)
    if max_logits_batch is None:
      return param_var

    param = param_var.get_value()
    scan_axis = _resolve_scan_axis(param, max_logits_batch, config, param_var=param_var)
    scale = _scale_from_max_logits(max_logits_batch, tau, preserve_scan_axis=scan_axis is not None)
    return param_var.replace(
        value=_clip_mla_weight(layer_name, param, scale, config.qk_nope_head_dim, scan_axis=scan_axis)
    )

  new_params_state = jax.tree_util.tree_map_with_path(
      clip_mla_weights, params_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )
  nnx.update(state.model, new_params_state)
  return state
