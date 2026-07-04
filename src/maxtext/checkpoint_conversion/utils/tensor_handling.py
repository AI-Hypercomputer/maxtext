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
import numpy as onp


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


def get_safe_local_array(array, shape, dtype):
  """Safely extracts a SingleDevice sharded tensor into a purely local CPU Numpy Array 
  (returning Zeros on nodes that do not own the payload) to enable math processing."""
  if array is None:
    return onp.zeros(shape, dtype=dtype), False

  is_np = not hasattr(array, "addressable_shards")
  host_id = jax.process_index()

  if not is_np and len(array.addressable_shards) > 0:
    is_owner = True
  elif not is_np and hasattr(array, "sharding") and hasattr(array.sharding, "device_set"):
    owner_device = list(array.sharding.device_set)[0]
    is_owner = (owner_device.process_index == host_id)
  else:
    is_owner = is_np

  if is_owner:
    return onp.asarray(array), is_owner
  return onp.zeros(shape, dtype=dtype), is_owner


def reshard_to_target(array, sharding, hook_fns=None, target_shape=None, is_owner=None):
  """Reshards a local CPU array cross-host explicitly to the target sharding."""
  if hasattr(array, "sharding") and isinstance(array.sharding, jax.sharding.NamedSharding):
    # For fully formed NamedSharding inputs, we process the hooks normally here.
    if hook_fns is not None and target_shape is not None:
      array = apply_hook_fns(array, target_shape, hook_fns)
    _reshard = jax.jit(lambda x: x, out_shardings=sharding)
    return _reshard(array)

  # Secure purely CPU memory
  if is_owner is None:
    local_arr, is_owner = get_safe_local_array(array, target_shape if target_shape is not None else array.shape, array.dtype if array is not None else np.float32)
  else:
    local_arr = onp.asarray(array)

  if hook_fns is not None and target_shape is not None:
    local_arr = apply_hook_fns(local_arr, target_shape, hook_fns)
    local_arr = onp.asarray(local_arr)
  
  import math
  bytes_per_slice = math.prod(local_arr.shape[1:]) * local_arr.itemsize if len(local_arr.shape) > 1 else local_arr.itemsize
  max_bytes_per_chunk = 128 * 1024 * 1024  # 128 MB max buffer per transmission

  if bytes_per_slice >= max_bytes_per_chunk:
    chunk_size = 1
  else:
    chunk_size = max(1, max_bytes_per_chunk // bytes_per_slice)
    
  total_len = local_arr.shape[0]

  # Broadcast over network sequentially in mathematically stable chunks 
  # to rigorously prevent Host OOM Resource Exhausted Limits on the FSDP Grid.
  if total_len <= chunk_size or total_len == 0:
    global_replicated = jax.experimental.multihost_utils.broadcast_one_to_all(
        local_arr, is_source=is_owner
    )
    cpu_replicated = onp.asarray(global_replicated)
    del global_replicated
  else:
    replicated_chunks = []
    for i in range(0, total_len, chunk_size):
      chunk_arr_cpu = local_arr[i : i + chunk_size]
      global_replicated_chunk = jax.experimental.multihost_utils.broadcast_one_to_all(
          chunk_arr_cpu, is_source=is_owner
      )
      replicated_chunks.append(onp.asarray(global_replicated_chunk))
      del global_replicated_chunk

    cpu_replicated = onp.concatenate(replicated_chunks, axis=0)
    del replicated_chunks

  del local_arr

  # jax.make_array_from_callback directly pulls the memory slices mathematically matched 
  # by the FSDP Sharding layout and instantiates them purely into device memory! 
  res = jax.make_array_from_callback(
      cpu_replicated.shape,
      sharding,
      lambda index: cpu_replicated[index]
  )
  return res


def unpack_getter_output(val):
  """Safely unpacks the (array, shape, dtype) tuple or defaults to None for mock arrays."""
  if isinstance(val, tuple) and len(val) == 3:
    return val
  return val, None, None
    
def unpack_getter_output(val):
  """Safely unpacks the (array, shape, dtype) tuple or defaults to None for mock arrays."""
  if isinstance(val, tuple) and len(val) == 3:
    return val
  return val, None, None


def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_leaf: Any,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along experts and layers using CPU-TPU hybrid phases."""
  if hasattr(target_leaf, "sharding"):
    target_shape = target_leaf.shape
    target_sharding = target_leaf.sharding
    target_dtype = target_leaf.dtype
  else:
    target_shape = target_leaf
    target_sharding = None
    target_dtype = target_leaf.dtype if hasattr(target_leaf, "dtype") else np.float32

  # Slice shape is (7168, 2048) or similar, from index 2 onwards
  mt_slice_shape = target_shape[2:]

  # pre-derive layer sharding spec (removes layer dimension at axis 1)
  if target_sharding is not None and hasattr(target_sharding, "spec"):
    spec_list = list(target_sharding.spec)
    del spec_list[1] # Delete layers axis
    layer_sharding = jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec(*spec_list))
  else:
    layer_sharding = target_sharding

  # Transpose key matrix from [256 experts][61 layers] to [61 layers][256 experts]
  # This allows us to CPU-stack all experts for a single layer together (takes only ~5.2 GB CPU RAM)
  keys_by_layer = list(zip(*hf_source_keys))

  expert_tensors_across_layers = []
  
  for j, expert_keys_for_layer in enumerate(keys_by_layer):
    expert_tensors_numpy = []
    
    # Establish ownership dynamically based on the first expert in this layer
    sample_tensor, sample_shape, sample_dtype = unpack_getter_output(tensor_getter_fn(expert_keys_for_layer[0]))
    _, is_owner = get_safe_local_array(sample_tensor, sample_shape if sample_shape is not None else mt_slice_shape, sample_dtype if sample_dtype is not None else target_dtype)

    # Fetch and format all 256 experts for this single layer on CPU
    for hf_key_single in expert_keys_for_layer:
      hf_tensor_raw, raw_shape, raw_dtype = unpack_getter_output(tensor_getter_fn(hf_key_single))
      local_arr, _ = get_safe_local_array(hf_tensor_raw, raw_shape if raw_shape is not None else mt_slice_shape, raw_dtype if raw_dtype is not None else target_dtype)
      processed = apply_hook_fns(local_arr, mt_slice_shape, hook_fns)
      expert_tensors_numpy.append(onp.asarray(processed))

    # Stack experts locally on Host CPU (Shape: 256, 7168, 2048)
    stacked_layer = onp.stack(expert_tensors_numpy, axis=0)

    # Reshard this layer to TPU. Peak HBM overhead per node: (5.2GB / 256) = 20 Megabytes!
    if layer_sharding is not None:
      sharded_layer = reshard_to_target(stacked_layer, layer_sharding, is_owner=is_owner)
    else:
      sharded_layer = jax.device_put(stacked_layer, jax.local_devices()[0])
      
    # Block CPU thread here to force the TPU sequencer queue to flush
    sharded_layer.block_until_ready()
    expert_tensors_across_layers.append(sharded_layer)

  # Stack the 61 FSDP-sharded layers on TPU (takes < 0.1s!)
  stacked_array = _binary_chunked_stack(expert_tensors_across_layers, axis=1).astype(target_dtype)
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

  axis_to_stack = config.param_scan_axis if config.scan_layers else 0

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
  
  # Reshard each layer slice sequentially. Since there are only 61 layers, this is extremely fast.
  for hf_key_single in hf_source_keys:
    hf_tensor_raw, raw_shape, raw_dtype = unpack_getter_output(tensor_getter_fn(hf_key_single))
    local_arr, is_owner = get_safe_local_array(hf_tensor_raw, raw_shape if raw_shape is not None else mt_slice_shape, raw_dtype if raw_dtype is not None else target_dtype)
    if slice_sharding is not None:
      processed = reshard_to_target(local_arr, slice_sharding, hook_fns=hook_fns, target_shape=mt_slice_shape, is_owner=is_owner)
    else:
      processed = apply_hook_fns(local_arr, mt_slice_shape, hook_fns)
      
    # Block CPU thread here to force the TPU sequencer queue to flush
    processed.block_until_ready()
    tensors_to_stack.append(processed)

  stacked_array = _binary_chunked_stack(tensors_to_stack, axis=axis_to_stack).astype(target_dtype)
  return stacked_array


def get_hf_loading_function(hf_source_keys_or_key, tensor_getter, hook_fn, mt_target_leaf, config):
  """Determine the loading function for HF keys."""
  if not isinstance(hf_source_keys_or_key, list):
    # Case 1: Single hf key (str)
    def _loader(getter, key, leaf, hook):
      if hasattr(leaf, "sharding"):
        t_jax, raw_shape, raw_dtype = unpack_getter_output(getter(key))
        local_arr, is_owner = get_safe_local_array(t_jax, raw_shape if raw_shape is not None else leaf.shape, raw_dtype if raw_dtype is not None else (leaf.dtype if hasattr(leaf, "dtype") else np.float32))
        res = reshard_to_target(local_arr, leaf.sharding, hook_fns=hook, target_shape=leaf.shape, is_owner=is_owner)
        res.block_until_ready()
        return res
      else:
        # Fallback Local Path
        t_jax, raw_shape, raw_dtype = unpack_getter_output(getter(key))
        local_arr, _ = get_safe_local_array(t_jax, raw_shape if raw_shape is not None else leaf.shape, raw_dtype if raw_dtype is not None else (leaf.dtype if hasattr(leaf, "dtype") else np.float32))
        return apply_hook_fns(local_arr, leaf.shape, hook)

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
