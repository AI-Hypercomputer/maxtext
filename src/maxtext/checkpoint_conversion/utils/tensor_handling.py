# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor handling utility functions for checkpoint conversion."""

from functools import partial
from typing import Any, Callable, List
import jax
import jax.numpy as np


def apply_hook_fns(weight, target_shape, hook_fns):
  """Apply hook functions, essential for to_maxtext and to_huggingface"""
  # If hook is unsepecified, use identity
  if hook_fns is None:
    return weight
  if not isinstance(hook_fns, list):
    hook_fns = [hook_fns]
  # Apply a list of hooks, be careful of order
  for hook_fn in hook_fns:
    weight = hook_fn(weight, target_shape)
  return weight


def _binary_chunked_stack(tensors: List[np.ndarray], axis: int) -> np.ndarray:
  """Stacks JAX arrays along axis by binary division to limit memory usage from JAX compiler."""
  if not tensors:
    raise ValueError("Cannot stack empty list of tensors.")
  if len(tensors) == 1:
    return np.expand_dims(tensors[0], axis=axis)
  if len(tensors) == 2:
    return np.stack(tensors, axis=axis)

  mid = len(tensors) // 2
  left = _binary_chunked_stack(tensors[:mid], axis=axis)
  right = _binary_chunked_stack(tensors[mid:], axis=axis)
  return np.concatenate([left, right], axis=axis)


def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_leaf: Any,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along two axes (experts and layers) directly in place on device."""
  if hasattr(target_leaf, "sharding"):
    target_shape = target_leaf.shape
    target_sharding = target_leaf.sharding
    target_dtype = target_leaf.dtype
  else:
    target_shape = target_leaf
    target_sharding = None
    target_dtype = target_leaf.dtype if hasattr(target_leaf, "dtype") else np.float32

  mt_slice_shape = target_shape[2:]

  # Pre-derive the compatible sharding specs to avoid rank mismatches
  if target_sharding is not None and hasattr(target_sharding, "spec"):
    # Target shape is (experts, layers, ...) -> slice is from index 2 onwards
    spec_list = list(target_sharding.spec)[2:]
    slice_sharding = jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec(*spec_list))
    # Stacking layer shards
    layer_spec_list = list(target_sharding.spec)[1:]
    layer_sharding = jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec(*layer_spec_list))
  else:
    slice_sharding = target_sharding
    layer_sharding = target_sharding

  all_expert_tensors = []
  # Outer loop iterates through experts
  for layer_keys_for_expert in hf_source_keys:
    layer_tensors_for_expert = []
    # Inner loop iterates through layers for the current expert
    for hf_key_single in layer_keys_for_expert:
      hf_tensor_numpy = tensor_getter_fn(hf_key_single)
      processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)

      if target_sharding is not None:
        processed_hf_tensor = jax.device_put(processed_hf_tensor, slice_sharding)
      layer_tensors_for_expert.append(processed_hf_tensor)

    expert_tensor = _binary_chunked_stack(layer_tensors_for_expert, axis=0)
    if target_sharding is not None:
      expert_tensor = jax.device_put(expert_tensor, layer_sharding)
    all_expert_tensors.append(expert_tensor)

  stacked_array = _binary_chunked_stack(all_expert_tensors, axis=0).astype(target_dtype)
  if target_sharding is not None:
    stacked_array = jax.device_put(stacked_array, target_sharding)
  return stacked_array


def _build_single_axis_stacked_tensor(
    hf_source_keys: List[str],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_leaf: Any,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along a single axis directly in place on device."""
  if hasattr(target_leaf, "sharding"):
    target_shape = target_leaf.shape
    target_sharding = target_leaf.sharding
    target_dtype = target_leaf.dtype
  else:
    target_shape = target_leaf
    target_sharding = None
    target_dtype = target_leaf.dtype if hasattr(target_leaf, "dtype") else np.float32

  if config.scan_layers:
    # If it's a standard scanned layer, we use the configured param_scan_axis.
    axis_to_stack = config.param_scan_axis
  else:
    # Otherwise, if an unscanned MoE layer, and we stack along the expert axis (0).
    axis_to_stack = 0

  # The hook function needs the shape of an individual slice, not the full stacked tensor.
  # We calculate it by removing the stacking dimension from the final target shape.
  mt_slice_shape_list = list(target_shape)
  del mt_slice_shape_list[axis_to_stack]
  mt_slice_shape = tuple(mt_slice_shape_list)

  if target_sharding is not None and hasattr(target_sharding, "spec"):
    spec_list = list(target_sharding.spec)
    del spec_list[axis_to_stack]
    slice_sharding = jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec(*spec_list))
  else:
    slice_sharding = target_sharding

  tensors_to_stack = []
  for hf_key_single in hf_source_keys:
    hf_tensor_numpy = tensor_getter_fn(hf_key_single)
    processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)

    if target_sharding is not None:
      processed_hf_tensor = jax.device_put(processed_hf_tensor, slice_sharding)
    tensors_to_stack.append(processed_hf_tensor)

  stacked_array = _binary_chunked_stack(tensors_to_stack, axis=axis_to_stack).astype(target_dtype)
  if target_sharding is not None:
    stacked_array = jax.device_put(stacked_array, target_sharding)
  return stacked_array


def get_hf_loading_function(hf_source_keys_or_key, tensor_getter, hook_fn, mt_target_leaf, config):
  """Determine the loading function for HF keys."""
  if not isinstance(hf_source_keys_or_key, list):
    # Case 1: Single hf key (str)
    def _loader(getter, key, leaf, hook):
      if hasattr(leaf, "sharding"):
        array = apply_hook_fns(getter(key), leaf.shape, hook)
        return jax.device_put(array, device=leaf.sharding)
      else:
        return apply_hook_fns(getter(key), leaf, hook)

    return partial(
        _loader,
        tensor_getter,
        hf_source_keys_or_key,
        mt_target_leaf,
        hook_fn,
    )
  # Stacked mapping
  elif not isinstance(hf_source_keys_or_key[0], list):
    # Case 2 or 3: Single-Axis Stacked hf keys (un-nested list)
    return partial(
        _build_single_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_leaf,
        config,
    )
  else:
    # isinstance(hf_source_keys_or_key[0], list)
    # Case 4: Multi-Axis Stacked hf keys (nested list)
    return partial(
        _build_multi_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_leaf,
        config,
    )
