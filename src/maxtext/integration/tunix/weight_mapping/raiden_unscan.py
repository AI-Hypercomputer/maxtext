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

import gc
from typing import Any, Iterator, Tuple, List

import jax
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict


def _unscan_one_key(
    key: Tuple[Any, ...],
    value: Any,
    num_layers: int,
    layer_container: str = "layers",
    scan_axis: int = 1,
) -> Tuple[List[Tuple[Tuple[Any, ...], Any]], bool]:
  """Unscans a single flattened pytree entry.

  Returns:
    (list_of_entries, is_scanned), where list_of_entries is a list of (new_key, value_slice) pairs.
  """
  if layer_container not in key:
    return [(key, value)], False

  idx = key.index(layer_container)
  prefix = key[:idx]
  suffix = key[idx + 1 :]
  arr = getattr(value, "value", value)

  if not hasattr(arr, "shape") or arr.ndim <= scan_axis:
    return [(key, value)], False

  if arr.shape[scan_axis] != num_layers:
    raise ValueError(
        f"unscan_layers: {'.'.join(str(k) for k in key)!r} has shape {arr.shape}, expected axis {scan_axis} to be"
        f" num_layers={num_layers}."
    )

  entries = []
  for i in range(num_layers):
    sliced = jax.lax.index_in_dim(arr, i, axis=scan_axis, keepdims=False)
    new_key = prefix + (f"{layer_container}_{i}",) + suffix
    entries.append((new_key, sliced))
  return entries, True


def unscan_layers_streaming(
    state: Any,
    num_layers: int,
    layer_container: str = "layers",
    scan_axis: int = 1,
    *,
    keys_per_piece: int = 1,
) -> Iterator[Any]:
  """Yields unscanned layer pieces incrementally for Raiden weight sync.

  Args:
    state: An `nnx.State` (or any pytree exposing `to_pure_dict`/`to_dict`, or a plain nested dict) of MaxText params,
      scanned along `scan_axis` under a `layer_container` key (MaxText's default scan layout).
    num_layers: Number of layers the scanned axis must have.
    layer_container: The pytree key holding the scanned per-layer params.
    scan_axis: The axis along which layers are scanned (default 1).
    keys_per_piece: Number of original flattened keys to batch per yielded piece (default 1).

  Yields:
    Nested dicts with `nnx.Param`-wrapped leaves, each containing `keys_per_piece` keys' worth of
    unscanned slices.
  """
  if hasattr(state, "to_pure_dict"):
    pure = state.to_pure_dict()
  elif hasattr(state, "to_dict"):
    pure = state.to_dict()
  elif isinstance(state, dict):
    pure = state
  else:
    yield state
    return

  flat = flatten_dict(pure)

  has_scanned = any(
      layer_container in key
      and hasattr(getattr(flat[key], "value", flat[key]), "shape")
      and getattr(getattr(flat[key], "value", flat[key]), "ndim", 0) > scan_axis
      for key in flat
  )
  if not has_scanned:
    raise ValueError(
        f"unscan_layers: found no scanned '{layer_container}' entries to unscan "
        "-- state may already be unscanned, or layer_container is wrong."
    )

  keys_per_piece = max(1, keys_per_piece)
  flat_keys = list(flat.keys())
  for i in range(0, len(flat_keys), keys_per_piece):
    chunk_keys = flat_keys[i : i + keys_per_piece]
    piece_flat = {}
    for key in chunk_keys:
      value = flat.pop(key)
      outputs, _ = _unscan_one_key(
          key, value, num_layers=num_layers, layer_container=layer_container, scan_axis=scan_axis
      )
      for new_key, sliced in outputs:
        piece_flat[new_key] = sliced
      del value, outputs

    nested = unflatten_dict(piece_flat)
    del piece_flat
    yield jax.tree_util.tree_map(
        lambda x: nnx.Param(x) if not isinstance(x, (nnx.Param, nnx.Variable)) else x,
        nested,
    )

  del flat
  gc.collect()


def unscan_layers(
    state: Any,
    num_layers: int,
    layer_container: str = "layers",
    scan_axis: int = 1,
) -> Any:
  """Splits `state`'s scanned `layer_container` axis into per-layer entries.

  Args:
    state: An `nnx.State` (or any pytree exposing `to_pure_dict`/`to_dict`, or a plain nested dict) of MaxText params,
      scanned along `scan_axis` under a `layer_container` key (MaxText's default scan layout).
    num_layers: Number of layers the scanned axis must have. Used both to validate the input and to bound the unscan
      loop.
    layer_container: The pytree key holding the scanned per-layer params (MaxText's decoder body uses "layers").
    scan_axis: The axis along which layers are scanned (default 1).

  Returns:
    A nested dict with `layer_container` keys replaced by `f"{layer_container}_{i}"` for each layer `i`, each holding
    the corresponding `scan_axis` slice of the original array, wrapped in `nnx.Param` -- matching `nnx.state(...,
    nnx.Param)`'s leaf type, so downstream consumers (`raiden_synchronizer.flatten_weights`, which unwraps `.value`)
    see the same leaf shape whether or not this transform ran. Non-scanned entries (e.g. embeddings, final norm) pass
    through unchanged, also rewrapped.
  """
  if not hasattr(state, "to_pure_dict") and not hasattr(state, "to_dict") and not isinstance(state, dict):
    return state

  new_flat = {}
  for piece in unscan_layers_streaming(
      state, num_layers=num_layers, layer_container=layer_container, scan_axis=scan_axis
  ):
    new_flat.update(flatten_dict(piece))
  gc.collect()
  nested = unflatten_dict(new_flat)
  del new_flat
  return nested
