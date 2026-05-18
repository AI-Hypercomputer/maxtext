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
import numpy as np


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


def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_shape: tuple,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along two axes (experts and layers).

  This function handles the complex case for scanned MoE layers, producing a tensor
  with the shape (num_experts, num_layers, ...).

  Args:
      hf_source_keys: A nested (2D) list of Hugging Face parameter names.
                      Outer list iterates experts, inner list iterates layers.
      tensor_getter_fn: A callable that takes a HF key and returns the tensor (as numpy array).
      hook_fns: The hook function(s) to apply to each individual weight.
      target_shape: The final shape of the target MaxText tensor.
      config: The MaxText pyconfig object.

  Returns:
      The final, assembled NumPy array for the MaxText parameter.
  """
  all_expert_tensors = []
  # The hook function needs the shape of an individual slice, not the full stacked tensor.
  # For multi-axis stacking (experts, layers, ...), the slice shape is target_shape[2:]
  mt_slice_shape = target_shape[2:]

  # Outer loop iterates through experts
  for layer_keys_for_expert in hf_source_keys:
    layer_tensors_for_expert = []
    # Inner loop iterates through layers for the current expert
    for hf_key_single in layer_keys_for_expert:
      hf_tensor_numpy = tensor_getter_fn(hf_key_single)
      processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)
      layer_tensors_for_expert.append(processed_hf_tensor)
    all_expert_tensors.append(np.stack(layer_tensors_for_expert, axis=0))
  return np.stack(all_expert_tensors, axis=0)


def _build_single_axis_stacked_tensor(
    hf_source_keys: List[str],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_shape: tuple,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along a single axis.

  This function handles both standard scanned layers (e.g., attention) and
  unscanned MoE layers (which are stacked along the expert axis).

  Args:
      hf_source_keys: A 1D list of Hugging Face parameter names.
      tensor_getter_fn: A callable that takes a HF key and returns the tensor (as numpy array).
      hook_fns: The hook function(s) to apply to each individual weight.
      target_shape: The final shape of the target MaxText tensor.
      config: The MaxText pyconfig object.

  Returns:
      The final, assembled NumPy array for the MaxText parameter.
  """
  tensors_to_stack = []

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

  for hf_key_single in hf_source_keys:
    hf_tensor_numpy = tensor_getter_fn(hf_key_single)
    processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)
    tensors_to_stack.append(processed_hf_tensor)

  # Stack all processed tensors along the determined axis.
  return np.stack(tensors_to_stack, axis=axis_to_stack)


def _get_hf_loading_function(hf_source_keys_or_key, tensor_getter, hook_fn, mt_target_shape_or_shapes, config):
  """Determine the loading function for HF keys.
  HF keys can take four forms:
    Case 1: Unscanned (single string)
    Case 2: Scanned (list of strings)
    Case 3: Unscanned with expert stacking (list of strings)
    Case 4: Scanned with expert stacking (nested list of strings)
  """
  load_fn = None
  if not isinstance(hf_source_keys_or_key, list):
    # Case 1: Single hf key (str)
    def _loader(getter, key, shape, hook):
      return apply_hook_fns(getter(key), shape, hook)

    load_fn = partial(
        _loader,
        tensor_getter,
        hf_source_keys_or_key,
        mt_target_shape_or_shapes,
        hook_fn,
    )
  # Stacked mapping
  elif not isinstance(hf_source_keys_or_key[0], list):
    # Case 2 or 3: Single-Axis Stacked hf keys (un-nested list)
    load_fn = partial(
        _build_single_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_shape_or_shapes,
        config,
    )
  else:
    # isinstance(hf_source_keys_or_key[0], list)
    # Case 4: Multi-Axis Stacked hf keys (nested list)
    load_fn = partial(
        _build_multi_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_shape_or_shapes,
        config,
    )
  return load_fn
