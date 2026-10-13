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
from jax.experimental import multihost_utils
import jax.numpy as np
import numpy

# An all-gather puts a copy of the whole array on every device, so larger arrays are gathered in slices of at most
# this many bytes along the leading axis.
_MAX_GATHER_BYTES = 1 << 30

# Variable collections besides `params` that param mappings name as a key prefix (DeepSeek MoE gate bias and
# hash-routing table).
VARIABLE_COLLECTIONS = ("MoEBiasVar", "Tid2EidVar")


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


def _spans_other_hosts(weight) -> bool:
  """Returns True if `weight`, or any tensor in a tuple of them, has shards on other hosts."""
  weights = weight if isinstance(weight, tuple) else (weight,)
  return any(isinstance(w, jax.Array) and not w.is_fully_addressable for w in weights)


def _gather_in_slices(weight, gather_fn, max_bytes=_MAX_GATHER_BYTES):
  """Gathers `weight` with `gather_fn`, in slices along the leading axis if it is larger than `max_bytes`.

  Each slice is gathered to every host and copied to host memory before the next one, so the copy on every
  device stays at most `max_bytes` (e.g. 8.6 GB qwen3.5-397b expert weights do not fit in one TPU's HBM).
  """
  if weight.ndim == 0 or weight.shape[0] < 2 or weight.nbytes <= max_bytes:
    return gather_fn(weight)
  rows = max(1, weight.shape[0] * max_bytes // weight.nbytes)
  return numpy.concatenate(
      [numpy.asarray(gather_fn(weight[i : i + rows])) for i in range(0, weight.shape[0], rows)], axis=0
  )


def _gather_to_host(weight):
  """Gathers `weight`, or each tensor in a tuple of them, into a full NumPy array on every host."""
  if isinstance(weight, tuple):
    return tuple(_gather_to_host(w) for w in weight)
  if isinstance(weight, jax.Array) and not weight.is_fully_addressable:
    return _gather_in_slices(weight, partial(multihost_utils.process_allgather, tiled=True))
  return weight


def _apply_hook_fns_any_host(weight, target_shape, hook_fns):
  """Applies hook functions to a weight that may be sharded across hosts.

  Many hooks use NumPy, which cannot read a `jax.Array` with shards on other hosts. If a hook fails on
  such a weight, the weight is gathered to every host and the hooks are applied again. Every host loads
  the same tensors in the same order, so all hosts take the fallback together, as the gather requires.
  """
  try:
    return apply_hook_fns(weight, target_shape, hook_fns)
  except RuntimeError:
    if not _spans_other_hosts(weight):
      raise
    return apply_hook_fns(_gather_to_host(weight), target_shape, hook_fns)


def _device_put(array, sharding):
  """Places `array` with `sharding`.

  A host array, such as a NumPy hook result after the gather fallback, is placed shard by shard with
  `jax.make_array_from_callback`, since every host holds all of it. `jax.device_put` would first all-gather it
  across hosts on device to check that all hosts hold the same value, which needs the array's size times the
  number of hosts in HBM.
  """
  if not isinstance(array, jax.Array) and not getattr(sharding, "is_fully_addressable", True):
    return jax.make_array_from_callback(array.shape, sharding, lambda index: array[index])
  return jax.device_put(array, sharding)


def _get_hf_tensor(tensor_getter_fn: Callable[[str], np.ndarray], hf_key: Any):
  """Returns the HF tensor for `hf_key`, or a tuple of tensors for a tuple of keys.

  A `None` key marks a leaf with no HF tensor (e.g. DeepSeek-V4 mHC norm scales): it returns `None`, and the
  hook builds the value, as in to_maxtext.
  """
  if hf_key is None:
    return None
  if isinstance(hf_key, tuple):
    return tuple(tensor_getter_fn(k) for k in hf_key)
  return tensor_getter_fn(hf_key)


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


def stacked_axes(mt_key: str | tuple[str, ...], config, depth: int) -> tuple:
  """Returns where each of the ``depth`` stacked axes lands in the MaxText tensor.

  The outer-to-inner ordering of the returned axes matches the outer-to-inner
  nesting of the HF key list. Three layouts occur:

  * MoE expert stacking -- ``(experts, layers, ...)``: every stacked axis is
    leading, so the axes are ``0, 1, ...``.
  * A nested block scan (``...-local_layers-...``, used by gemma4 and
    qwen3-next): the block's local layers are an inner scan nested inside the
    outer block scan, so ``(blocks, local)`` sit at
    ``(param_scan_axis, param_scan_axis + 1)`` rather than at the leading axes.
    A remainder block (``layers_remainder-local_layers-...``) is not inside the
    block scan, so its local layers stack like any other scanned layers.
  * A nested block scan whose weights are *also* expert-stacked (qwen3-next's
    routed experts): the expert axis still leads, giving
    ``(0, param_scan_axis, param_scan_axis + 1)``.
  """
  if isinstance(mt_key, tuple) and mt_key:
    mt_key = mt_key[0]
  if isinstance(mt_key, str) and "-local_layers" in mt_key and "layers_remainder-" not in mt_key:
    param_scan_axis = config.param_scan_axis
    nested_axes = (param_scan_axis, param_scan_axis + 1)
    if depth == 2:
      return nested_axes
    if param_scan_axis == 0:
      # Both scan axes would then sit in front of the expert axis, which no longer leads.
      raise ValueError(f"{mt_key}: expert-stacked weights in a nested block scan need param_scan_axis >= 1.")
    return (0,) + nested_axes
  return tuple(range(depth))


def slice_shape(target_shape: tuple, axes: tuple) -> tuple:
  """Returns ``target_shape`` with the stacked ``axes`` removed.

  Hook functions operate on a single un-stacked slice, so they need this shape
  rather than the shape of the fully assembled tensor.
  """
  return tuple(dim for i, dim in enumerate(target_shape) if i not in set(axes))


def _full_spec(sharding, ndim: int) -> list:
  """Returns the partition spec of `sharding` with one entry per axis of an `ndim`-rank array.

  A `PartitionSpec` may be shorter than the array rank; the missing trailing axes are replicated.
  """
  spec = list(sharding.spec)
  return spec + [None] * (ndim - len(spec))


def composite_target_leaf(leaves: List[Any]) -> Any:
  """Returns the target for a composite MaxText key: its leaves stacked along a new last axis.

  The hook for a composite MaxText key returns the leaves stacked along the last axis, so the stacked
  tensor is loaded as one target and then split. The new axis is replicated.
  """
  leaf = leaves[0]
  if not hasattr(leaf, "sharding"):
    return tuple(leaf) + (len(leaves),)
  sharding = leaf.sharding
  if hasattr(sharding, "spec"):
    spec = jax.sharding.PartitionSpec(*_full_spec(sharding, len(leaf.shape)), None)
    sharding = jax.sharding.NamedSharding(sharding.mesh, spec)
  return jax.ShapeDtypeStruct(shape=tuple(leaf.shape) + (len(leaves),), dtype=leaf.dtype, sharding=sharding)


def composite_hook_fn(hook_fns: Any, num_leaves: int) -> Callable:
  """Wraps the hook of a composite MaxText key to take the stacked shape from `composite_target_leaf`.

  Composite hooks expect one target shape per leaf, as `to_maxtext` passes them.
  """

  def hook(weight, target_shape):
    return apply_hook_fns(weight, [tuple(target_shape[:-1])] * num_leaves, hook_fns)

  return hook


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
    mt_key: str | tuple[str, ...] = "",
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
  target_spec = (
      _full_spec(target_sharding, len(target_shape))
      if target_sharding is not None and hasattr(target_sharding, "spec")
      else None
  )

  def sharding_at(level):
    if target_spec is None:
      return target_sharding
    stacked_spec = [target_spec[axis] for axis in axes[level:]]
    slice_spec = [spec for i, spec in enumerate(target_spec) if i not in set(axes)]
    return jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec(*stacked_spec, *slice_spec))

  def gather(keys, level):
    if level == depth:
      # A tuple of keys is a composite HF source that the hook fuses into one leaf.
      tensor = _apply_hook_fns_any_host(_get_hf_tensor(tensor_getter_fn, keys), mt_slice_shape, hook_fns)
    else:
      tensor = _binary_chunked_stack([gather(sub, level + 1) for sub in keys], axis=0)
    if target_sharding is not None and level > 0:
      tensor = _device_put(tensor, sharding_at(level))
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
    mt_key: Any = "",
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

  # Same rule as to_maxtext: with scan_layers, an MoE layer outside the scanned stack (e.g. the DeepSeek-V4
  # prefix layers `layers_0`..) and the MoE bias and tid2eid collections still stack experts along axis 0.
  key_str = mt_key[0] if isinstance(mt_key, tuple) else mt_key
  is_var_collection = any(collection in key_str for collection in VARIABLE_COLLECTIONS)
  is_unscanned_moe = "MoeBlock" in key_str and not any(
      s in key_str for s in ("scanned_blocks", "moe_layers-", "decoder-layers-")
  )
  if config.scan_layers and not (is_var_collection or is_unscanned_moe):
    # If the target tensor rank exceeds param_scan_axis (e.g., multidimensional weights or 2D block scales),
    # stack along param_scan_axis (typically axis 1 in scanned layers). For 1D tensors (e.g., per-layer scalar
    # quantization scales like Llama 3.1 FP8 `kernel_scale` with shape (num_layers,)), stack along axis 0
    # because axis 1 does not exist on 1D tensors and stacking scalars along axis 0 produces the 1D vector.
    axis_to_stack = config.param_scan_axis if len(target_shape) > config.param_scan_axis else 0
  else:
    # Otherwise, if an unscanned MoE layer, and we stack along the expert axis (0).
    axis_to_stack = 0

  # The hook function needs the shape of an individual slice, not the full stacked tensor.
  # We calculate it by removing the stacking dimension from the final target shape.
  mt_slice_shape_list = list(target_shape)
  del mt_slice_shape_list[axis_to_stack]
  mt_slice_shape = tuple(mt_slice_shape_list)

  if target_sharding is not None and hasattr(target_sharding, "spec"):
    spec_list = _full_spec(target_sharding, len(target_shape))
    del spec_list[axis_to_stack]
    slice_sharding = jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec(*spec_list))
  else:
    slice_sharding = target_sharding

  tensors_to_stack = []
  for hf_key_single in hf_source_keys:
    hf_tensor = _get_hf_tensor(tensor_getter_fn, hf_key_single)
    processed_hf_tensor = _apply_hook_fns_any_host(hf_tensor, mt_slice_shape, hook_fns)

    if target_sharding is not None:
      processed_hf_tensor = _device_put(processed_hf_tensor, slice_sharding)
    tensors_to_stack.append(processed_hf_tensor)

  stacked_array = _binary_chunked_stack(tensors_to_stack, axis=axis_to_stack).astype(target_dtype)
  if target_sharding is not None:
    stacked_array = jax.device_put(stacked_array, target_sharding)
  return stacked_array


def get_hf_loading_function(hf_source_keys_or_key, tensor_getter, hook_fn, mt_target_leaf, config, mt_key=""):
  """Returns a function that builds one MaxText weight from its HF tensors.

  The function takes no arguments. If `mt_target_leaf` has a sharding, the weight is returned in the
  leaf's dtype and sharding.

  Args:
    hf_source_keys_or_key: A single HF key, or a list of HF keys to stack, nested once per stacked
      axis.
    tensor_getter: A function that returns the HF tensor for a key.
    hook_fn: A hook function, or a list of them, applied to each HF tensor. None means no hook.
    mt_target_leaf: The target leaf, for example a `jax.ShapeDtypeStruct` with a sharding.
    config: The MaxText config.
    mt_key: The MaxText parameter key, used to place the axes of nested block scans.
  """
  if not isinstance(hf_source_keys_or_key, list):
    # Case 1: Single hf key (str), or a tuple of keys that the hook fuses into one leaf
    def _loader(getter, key, leaf, hook):
      if hasattr(leaf, "sharding"):
        array = _apply_hook_fns_any_host(_get_hf_tensor(getter, key), leaf.shape, hook)
        # Cast to the target dtype, as the stacked cases do. Otherwise a bf16 checkpoint stays bf16
        # even when weight_dtype is float32.
        return _device_put(array.astype(leaf.dtype), leaf.sharding)
      else:
        return _apply_hook_fns_any_host(_get_hf_tensor(getter, key), leaf, hook)

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
        mt_key,
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
