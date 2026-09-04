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

"""Unscans a trainer's scanned MaxText param state for Raiden weight sync.

The trainer runs with ``scan_layers=True`` (one array per param, with a
``layer`` axis) for training performance. The rollout side loads its MaxText
model via ``maxtext_vllm_adapter.MaxTextForCausalLM`` using MaxText's own
``configs/inference/vllm.yml``, which sets ``scan_layers=False`` (one array
per param *per layer*, named ``layers_0``, ``layers_1``, ...). Raiden's
weight-sync transport matches trainer and sampler tensors by name, so the
trainer's scanned tree must be unscanned into that same per-layer shape
before binding, or names/shapes never line up.

This differs from ``maxtext.integration.vllm.maxtext_vllm_rollout``'s
``unroll_gemma_scanned_weights``: that function targets Gemma 3/4's
attention-pattern-interleaved scan (an explicit ``layers.layers_N`` pytree
level per attention-pattern block, requiring a scan-length/pattern-length
computation). Qwen3 (and MaxText's default non-interleaved scan) has no such
sub-block nesting -- the layer axis is embedded directly in each param array
at axis 1, with a single flat ``layers`` container. Reusing the Gemma
function's pattern-block matching would silently no-op on Qwen3 trees (its
``_find_scanned_layer_idx`` requires a ``layers_<N>`` key immediately under
``layers``, which never appears here), so this module implements the
simpler, single-axis case directly instead of adapting that function.
"""

import re
from typing import Any

import jax
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict


def unscan_layers(
    state: Any,
    num_layers: int,
    layer_container: str = "layers",
    scan_axis: int = 1,
    cycle_interval: int = 1,
) -> Any:
  """Splits `state`'s scanned `layer_container` axis into per-layer entries.

  Args:
    state: An `nnx.State` (or any pytree exposing `to_pure_dict`/`to_dict`, or a plain nested dict) of MaxText params,
      scanned along `scan_axis` under a `layer_container` key (MaxText's default scan layout).
    num_layers: Number of layers the scanned axis must have. Used both to validate the input and to bound the unscan
      loop.
    layer_container: The pytree key holding the scanned per-layer params (MaxText's decoder body uses "layers").
    scan_axis: The axis along which layers are scanned (default 1).
    cycle_interval: `config.inhomogeneous_layer_cycle_interval`. When > 1 the
      trainer scans a *block* of this many heterogeneous layers (qwen3.5's
      GDN/attention cycle is 4), so the tree carries an extra `layer_<slot>`
      level under `layer_container` and the scan axis is
      `num_layers // cycle_interval` repeats rather than `num_layers`. Slot `j`
      of repeat `i` is physical layer `i * cycle_interval + j`, which is the
      flat `layers_<n>` name the unscanned rollout model uses.

  Returns:
    A nested dict with `layer_container` keys replaced by `f"{layer_container}_{i}"` for each layer `i`, each holding
    the corresponding `scan_axis` slice of the original array, wrapped in `nnx.Param` -- matching `nnx.state(...,
    nnx.Param)`'s leaf type, so downstream consumers (`raiden_synchronizer.flatten_weights`, which unwraps `.value`)
    see the same leaf shape whether or not this transform ran. Non-scanned entries (e.g. embeddings, final norm) pass
    through unchanged, also rewrapped.
  """
  if hasattr(state, "to_pure_dict"):
    pure = state.to_pure_dict()
  elif hasattr(state, "to_dict"):
    pure = state.to_dict()
  elif isinstance(state, dict):
    pure = state
  else:
    return state

  flat = flatten_dict(pure)
  new_flat = {}
  unscanned_count = 0

  # Drain `flat` as we go (pop, not iterate-then-keep) rather than holding
  # every original scanned array alive for the whole function: at 30B-A3B
  # scale (padded MoE weights are tens of GB each), keeping both the
  # original scanned tree and the ~num_layers-times-larger unscanned tree
  # alive simultaneously roughly doubles peak host memory during Raiden's
  # D2H staging -- confirmed as the direct cause of an OOMKill there.
  for key in list(flat.keys()):
    value = flat.pop(key)
    if layer_container not in key:
      new_flat[key] = value
      continue

    idx = key.index(layer_container)
    prefix = key[:idx]
    suffix = key[idx + 1 :]
    # A scanned inhomogeneous block nests one `layer_<slot>` level under
    # `layers`; the rollout has no such level, so drop it and fold the slot
    # into the physical layer number below.
    slot = None
    if cycle_interval > 1 and suffix:
      m = re.fullmatch(r"layer_(\d+)", suffix[0])
      if m:
        slot = int(m.group(1))
        suffix = suffix[1:]
    arr = getattr(value, "value", value)

    if arr is None or not hasattr(arr, "shape") or getattr(arr, "ndim", 0) <= scan_axis:
      # Not a per-layer leaf (shouldn't happen for real params under
      # `layers`, but don't silently drop anything unexpected). Keep the
      # original (possibly already-wrapped) value, matching pre-existing
      # behavior -- unlike the actually-scanned case below, there's no
      # multi-copy blowup here worth restructuring around.
      new_flat[key] = value
      continue
    del value

    # Each slot is scanned over the repeats of the cycle, not over all layers.
    expected = num_layers // cycle_interval if slot is not None else num_layers
    if arr.shape[scan_axis] != expected:
      raise ValueError(
          f"unscan_layers: {'.'.join(key)!r} has shape {arr.shape}, expected axis {scan_axis} to be"
          f" {expected} (num_layers={num_layers}, cycle_interval={cycle_interval})."
      )

    for i in range(expected):
      sliced = jax.lax.index_in_dim(arr, i, axis=scan_axis, keepdims=False)
      layer_no = i * cycle_interval + slot if slot is not None else i
      new_key = prefix + (f"{layer_container}_{layer_no}",) + suffix
      new_flat[new_key] = sliced
    del arr
    unscanned_count += 1

  if unscanned_count == 0:
    raise ValueError(
        f"unscan_layers: found no scanned '{layer_container}' entries to unscan "
        "-- state may already be unscanned, or layer_container is wrong."
    )

  nested = unflatten_dict(new_flat)
  return jax.tree_util.tree_map(nnx.Param, nested)
