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


def nesting_depth(hf_source_keys: Any) -> int:
  """Counts how many axes a (possibly nested) list of HF keys stacks over.

  Only ``list`` nesting counts: a ``tuple`` of HF keys is the composite-key
  convention (several HF tensors combined into one MaxText leaf by a hook), not
  a stacking axis.
  """
  depth = 0
  while isinstance(hf_source_keys, list):
    depth += 1
    hf_source_keys = hf_source_keys[0]
  return depth


def stacked_axes(mt_key: str, config, depth: int) -> tuple:
  """Returns where each of the ``depth`` stacked axes lands in the MaxText tensor.

  The outer-to-inner ordering of the returned axes matches the outer-to-inner
  nesting of the HF key list. Three layouts occur:

  * MoE expert stacking -- ``(experts, layers, ...)``: every stacked axis is
    leading, so the axes are ``0, 1, ...``.
  * A nested block scan (``...-local_layers-...``, used by gemma4 and
    qwen3-next): the block's local layers are an inner scan nested inside the
    outer block scan, so ``(blocks, local)`` sit at
    ``(param_scan_axis, param_scan_axis + 1)`` rather than at the leading axes.
  * A nested block scan whose weights are *also* expert-stacked (qwen3-next's
    routed experts): the expert axis still leads, giving
    ``(0, param_scan_axis, param_scan_axis + 1)``.
  """
  if isinstance(mt_key, str) and "-local_layers" in mt_key:
    param_scan_axis = config.param_scan_axis
    nested_axes = (param_scan_axis, param_scan_axis + 1)
    return nested_axes if depth == 2 else (0,) + nested_axes
  return tuple(range(depth))


def slice_shape(target_shape: tuple, axes: tuple) -> tuple:
  """Returns ``target_shape`` with the stacked ``axes`` removed.

  Hook functions operate on a single un-stacked slice, so they need this shape
  rather than the shape of the fully assembled tensor.
  """
  return tuple(dim for i, dim in enumerate(target_shape) if i not in set(axes))


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
    hf_source_keys: List[Any],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_leaf: Any,
    config,
    mt_key: str = "",
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along several axes, in place on device.

  ``hf_source_keys`` is nested one level per stacked axis, outermost first (see
  ``nesting_depth``), and ``stacked_axes`` decides where those axes land in the
  target: leading for MoE expert stacking, or at ``param_scan_axis`` and beyond
  for a nested block scan.
  """
  if hasattr(target_leaf, "sharding"):
    target_shape = target_leaf.shape
    target_sharding = target_leaf.sharding
    target_dtype = target_leaf.dtype
  else:
    target_shape = target_leaf
    target_sharding = None
    target_dtype = target_leaf.dtype if hasattr(target_leaf, "dtype") else np.float32

  depth = nesting_depth(hf_source_keys)
  axes = stacked_axes(mt_key, config, depth)
  mt_slice_shape = slice_shape(target_shape, axes)

  # Pre-derive the compatible sharding specs to avoid rank mismatches. The tensor
  # is assembled with its stacked axes leading and only moved into place at the
  # end, so level ``l`` carries the specs of the axes not yet stacked, followed by
  # the specs of the slice itself.
  target_spec = list(target_sharding.spec) if target_sharding is not None and hasattr(target_sharding, "spec") else None

  def sharding_at(level):
    if target_spec is None:
      return target_sharding
    stacked_spec = [target_spec[axis] for axis in axes[level:]]
    slice_spec = [spec for i, spec in enumerate(target_spec) if i not in set(axes)]
    return jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec(*stacked_spec, *slice_spec))

  def gather(keys, level):
    if level == depth:
      # A tuple of keys is a composite HF source that the hook fuses into one leaf.
      raw = tuple(tensor_getter_fn(k) for k in keys) if isinstance(keys, tuple) else tensor_getter_fn(keys)
      tensor = apply_hook_fns(raw, mt_slice_shape, hook_fns)
    else:
      tensor = _binary_chunked_stack([gather(sub, level + 1) for sub in keys], axis=0)
    if target_sharding is not None and level > 0:
      tensor = jax.device_put(tensor, sharding_at(level))
    return tensor

  stacked_array = gather(hf_source_keys, 0).astype(target_dtype)
  if axes != tuple(range(depth)):
    stacked_array = np.moveaxis(stacked_array, tuple(range(depth)), axes)
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


def get_hf_loading_function(hf_source_keys_or_key, tensor_getter, hook_fn, mt_target_leaf, config, mt_key=""):
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
        mt_key,
    )
