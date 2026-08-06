from collections import abc

import jax
from absl import logging
from typing import Mapping, Any, Callable, Dict, Tuple, Optional
from flax import traverse_util
import re
import functools
import numpy as np
import jax.numpy as jnp

def _delete_target_buffers(tgt_flat: Mapping[str, Any], src_flat: Mapping[str, Any]):
    """Physically deletes target buffers to free HBM before resharding."""
    deleted_count = 0
    preserved_count = 0
    
    src_buffers = set()
    for v in src_flat.values():
        if hasattr(v, "device_buffers"):
            try:
                for b in v.device_buffers():
                    src_buffers.add(b)
            except Exception:
                pass

    for key, tgt_val in tgt_flat.items():
        if hasattr(tgt_val, "value"):
            tgt_val = tgt_val.value
            
        if hasattr(tgt_val, "device_buffers"):
            try:
                is_aliased = False
                for b in tgt_val.device_buffers():
                    if b in src_buffers:
                        is_aliased = True
                        break
                if not is_aliased:
                    tgt_val.delete()
                    deleted_count += 1
                else:
                    preserved_count += 1
            except Exception:
                pass
                
    logging.info(
        "Deleted %d non-aliased target buffers (preserved %d aliased) to free HBM.", 
        deleted_count, preserved_count
    )

def _reshard_in_chunks(
    src_flat: Dict[tuple, Any],
    spec_flat: Dict[tuple, Any],
    reshard_fn: Callable,
    chunk_size: int,
    delete_dst_buffers: bool,
) -> Dict[tuple, Any]:
    """Batches resharding into chunks to prevent XLA contiguous memory fragmentation."""
    resharded_flat = {}
    keys = [k for k in src_flat.keys() if k in spec_flat]
    
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i : i + chunk_size]
        src_chunk = {k: src_flat[k] for k in chunk_keys}
        spec_chunk = {k: spec_flat[k] for k in chunk_keys}
        
        if delete_dst_buffers:
            _delete_target_buffers(spec_chunk, src_chunk)
            
        # Optional: unflatten back to dicts before passing to reshard_fn if your 
        # reshard_fn expects hierarchical dicts instead of flat tuple-keyed dicts.
        resharded_chunk = reshard_fn(source=src_chunk, target=spec_chunk)
        resharded_flat.update(resharded_chunk)
        
    return resharded_flat

def _fuse_moe_weights(
    src_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
    tgt_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
) -> Dict[Tuple[str, ...], jax.Array | np.ndarray]:
  """Stage 1: Bulk fuses MoE wi_0/wi_1 into wi in the source tree."""
  new_src_flat = dict(src_flat)

  sample_tgt_wi = None
  for tgt_k, tgt_v in tgt_flat.items():
    if tgt_k and tgt_k[-1] == 'wi':
      sample_tgt_wi = tgt_v
      break

  for src_key in list(new_src_flat.keys()):
    if not src_key or src_key[-1] != 'wi_0':
      continue
    wi_0_key = src_key
    wi_1_key = src_key[:-1] + ('wi_1',)
    wi_target_key = src_key[:-1] + ('wi',)
    if wi_1_key not in new_src_flat:
      continue

    wi_0 = new_src_flat.pop(wi_0_key)
    wi_1 = new_src_flat.pop(wi_1_key)

    axis = len(wi_0.shape) - 1
    matching_tgt = tgt_flat.get(wi_target_key, sample_tgt_wi)
    if matching_tgt is not None:
      tgt_axis = len(matching_tgt.shape) - 1
      n_shards = _get_n_shards(matching_tgt, tgt_axis)
      target_dim = matching_tgt.shape[tgt_axis]
    else:
      n_shards = 1
      target_dim = wi_0.shape[axis] + wi_1.shape[axis]

    fused_shape = list(wi_0.shape)
    fused_shape[axis] = target_dim
    fused_shape_tuple = tuple(fused_shape)

    logging.info(
        'Fusing MoE %s: wi_0=%s, wi_1=%s -> %s on axis %d',
        '.'.join(str(k) for k in wi_target_key),
        wi_0.shape, wi_1.shape, fused_shape_tuple, axis,
    )
    new_src_flat[wi_target_key] = _interleave_moe_weights(
        wi_0, wi_1, fused_shape_tuple, n_shards, axis=axis
    )
    del wi_0, wi_1

  return new_src_flat



class ShapeMismatchError(ValueError):
  """Raised when source and target shapes are incompatible."""

  pass

def _apply_dtype_cast(
    val: jax.Array | np.ndarray, tgt_dtype: jnp.dtype, src_key: str
) -> jax.Array | np.ndarray:
  if val.dtype != tgt_dtype:
    logging.log_first_n(
        logging.WARNING,
        'Type mismatch on %s: %s -> %s',
        1,
        src_key,
        val.dtype,
        tgt_dtype,
    )
    return val.astype(tgt_dtype)
  return val

def _shapes_are_repeatable(
    candidate_shape: tuple[int, ...],
    tgt_shape: tuple[int, ...],
) -> bool:
  """Returns True if candidate_shape can be repeated to match tgt_shape."""
  if len(candidate_shape) != len(tgt_shape):
    return False

  for s, t in zip(candidate_shape, tgt_shape):
    if s > t or t % s != 0:
      return False
  return True

def _unstack_scanned_param(
    src_val: jax.Array | np.ndarray,
    tgt_val: jax.Array | np.ndarray,
    key_path: str,
    scan_axis: Optional[int] = None,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Unstacks a scanned parameter by moving the scan axis to 0.

  This helper unstacks a scanned array at the specified scan_axis. When scan_axis
  is provided, it transposes that axis to position 0 and unstacks it. This is used
  when transferring weights from a scanned representation (e.g., MaxText) to an
  unrolled one (e.g., vLLM).

  Args:
    src_val: The source array (scanned) to slice from.
    tgt_val: The target array whose shape we want to match.
    key_path: The dot-separated path to the parameter for debugging.
    scan_axis: The axis containing the scanned dimension. If None, attempts to
      auto-detect it for backward compatibility.

  Returns:
      A tuple of unstacked arrays, or a tuple containing just the original src_val
      if unstacking fails or is unnecessary.
  """
  if not (hasattr(src_val, 'shape') and hasattr(tgt_val, 'shape')):
    return (src_val,)

  src_shape = src_val.shape
  tgt_shape = tgt_val.shape

  if src_shape == tgt_shape:
    return (src_val,)

  if len(src_shape) == len(tgt_shape) + 1:
    # If scan_axis not provided, try to detect it
    if scan_axis is None:
      for i in range(len(src_shape)):
        candidate = src_shape[:i] + src_shape[i + 1 :]
        if _shapes_are_repeatable(candidate, tgt_shape):
          scan_axis = i
          break
    
    if scan_axis is not None:
      # Transpose the scanned axis to the 0th position
      if scan_axis != 0:
        perm = (scan_axis,) + tuple(i for i in range(len(src_shape)) if i != scan_axis)
        if hasattr(src_val, 'transpose'):
          src_val = src_val.transpose(perm)
        elif isinstance(src_val, np.ndarray):
          src_val = np.transpose(src_val, perm)

      # Unstack along the 0th axis
      # Handling JAX version differences where unstack might be under jnp
      try:
        if hasattr(jax, 'unstack'):
          return jax.unstack(src_val)
        elif hasattr(jnp, 'unstack'):
          return jnp.unstack(src_val)
        else:
           # Fallback for older JAX versions
          return [src_val[i] for i in range(src_val.shape[0])]  # pyrefly: ignore[bad-return]
      except Exception as e:
        logging.debug(
            "Failed to unstack parameter '%s'. Error: %s. Using original.",
            key_path, e
        )
        return (src_val,)
    else:
      logging.warning(
          "Shape mismatch in scanned param '%s'. Src: %s, Tgt: %s. Cannot"
          ' determine scan axis.',
          key_path, src_shape, tgt_shape,
      )

  return (src_val,)

_MOE_MLP_WEIGHTS = frozenset({'wi', 'wi_0', 'wi_1'})

def _partition_size(
    partition: Optional[str | Tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> int:
  """Total mesh-axis size used to shard a single tensor axis."""
  if partition is None:
    return 1
  names = (partition,) if isinstance(partition, str) else tuple(partition)
  size = 1
  for n in names:
    size *= mesh.shape[n]
  return size

def _spec_at_axis(
    sharding: Optional[jax.sharding.Sharding],
    axis: int,
) -> Optional[str | Tuple[str, ...]]:
  """Returns the PartitionSpec entry at the given axis, or None if absent."""
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return None
  spec = sharding.spec
  return spec[axis] if axis < len(spec) else None

def _get_n_shards(arr: jax.Array | np.ndarray, axis: int) -> int:
  """Returns the number of shards for a given axis of an array."""
  sharding = getattr(arr, 'sharding', None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    return _partition_size(_spec_at_axis(sharding, axis), sharding.mesh)  # pyrefly: ignore[bad-argument-type]
  return 1

@functools.partial(jax.jit, static_argnames=('pad_specs',))
def _jit_zero_pad_axes(arr, pad_specs):
  """Shard-aware tail-pad on multiple axes within a single JIT.

  `pad_specs` is a tuple of `(axis, n_shards, per_shard_extra)`; entries
  with `per_shard_extra == 0` are no-ops. Composing every padded axis
  inside one trace lets XLA fuse them with the surrounding reshape ops
  rather than launching each as a separate eager primitive.
  """
  out = arr
  for axis, n_shards, per_shard_extra in pad_specs:
    if per_shard_extra <= 0:
      continue
    src_dim = out.shape[axis]
    src_chunk_size = src_dim // n_shards
    split_shape = list(out.shape)
    split_shape.insert(axis + 1, src_chunk_size)
    split_shape[axis] = n_shards
    arr_split = out.reshape(split_shape)
    pad_width = [(0, 0)] * arr_split.ndim
    pad_width[axis + 1] = (0, per_shard_extra)
    arr_padded = jnp.pad(arr_split, pad_width)
    final_shape = list(out.shape)
    final_shape[axis] = src_dim + per_shard_extra * n_shards
    out = arr_padded.reshape(final_shape)
  return out

@functools.partial(jax.jit, static_argnames=('repeats',))
def _jit_repeat_axes(arr, repeats):
  """Apply `jnp.repeat` on multiple axes within a single JIT.

  `repeats` is a tuple of `(axis, count)`. One trace covers every
  repeated axis so XLA can fuse them rather than dispatching per axis.
  """
  out = arr
  for axis, count in repeats:
    out = jnp.repeat(out, count, axis=axis)
  return out

def _align_per_axis(
    arr: jax.Array | np.ndarray,
    tgt_shape: Tuple[int, ...],
    tgt_sharding: Optional[jax.sharding.Sharding],
    key_path: str,
) -> jax.Array | np.ndarray:
  """Aligns `arr` to `tgt_shape` via either pure-repeat or pure-zero_pad.

  Each tensor needs exactly one transform mode in practice:
    * MoE linear weights (`wi`, `wi_0`, `wi_1`, `wo`) → `zero_pad` on
      every mismatched axis (TPU-lane / GMM_v2 alignment).
    * Anything else (attention QKV/O projections, biases, …) → `repeat`
      on every mismatched axis (e.g. KV-head expansion).
  KV-head expansion and MoE-dim padding never co-occur on the same
  tensor, so we classify by leaf key once and dispatch the whole
  transform to a single JIT compiled helper. That collapses what used
  to be N eager primitive launches into one fused XLA program — the hot
  path here is bulk alignment of scanned MoE weights, where eager
  dispatch was costing tens of seconds per tensor.
  """
  if not hasattr(arr, 'shape'):
    return arr
  if arr.shape == tgt_shape:
    return arr
  if len(arr.shape) != len(tgt_shape):
    raise ShapeMismatchError(
        f"Rank mismatch for {key_path}: src={arr.shape} vs tgt={tgt_shape}"
    )

  mismatches = []
  for axis, (s, t) in enumerate(zip(arr.shape, tgt_shape)):
    if s == t:
      continue
    if t < s:
      raise ShapeMismatchError(
          f"Cannot shrink axis {axis} for {key_path}: src={s} -> tgt={t}"
      )
    mismatches.append((axis, s, t))
  if not mismatches:
    return arr

  last_key = key_path.split('.')[-1]
  if last_key in _MOE_MLP_WEIGHTS:
    if isinstance(tgt_sharding, jax.sharding.NamedSharding):
      mesh = tgt_sharding.mesh
      pad_specs = []
      for axis, s, t in mismatches:
        n_shards = _partition_size(_spec_at_axis(tgt_sharding, axis), mesh)  # pyrefly: ignore[bad-argument-type]
        if t % n_shards != 0:
          raise ValueError(
              f"Target dimension {t} on axis {axis} for {key_path} is not "
              f"divisible by n_shards={n_shards}; the target shape itself "
              f"is misconfigured for the requested sharding."
          )
        if (t - s) % n_shards != 0 or s % n_shards != 0:
          raise ValueError(
              f"Cannot interleave pad axis {axis} for {key_path}: src_dim "
              f"({s}) or extra ({t - s}) is not cleanly divisible by "
              f"n_shards ({n_shards}). Ensure the source tensor is evenly "
              f"partitionable."
          )
        pad_specs.append((axis, n_shards, (t - s) // n_shards))
    else:
      pad_specs = [(axis, 1, t - s) for axis, s, t in mismatches]
    return _jit_zero_pad_axes(arr, tuple(pad_specs))

  repeats = []
  for axis, s, t in mismatches:
    if t % s != 0:
      raise ShapeMismatchError(
          f"Cannot align axis {axis} for {key_path}: src={s} -> tgt={t} "
          f"is not an integer multiple and the key is not a recognized "
          f"MoE pattern."
      )
    repeats.append((axis, t // s))
  return _jit_repeat_axes(arr, tuple(repeats))

def _interleave_moe_weights(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    tgt_shape: Tuple[int, ...],
    n_shards: int,
    axis: Optional[int] = None,
) -> jax.Array | np.ndarray:
  """Interleaves wi_0 and wi_1 per-shard into a single tensor."""
  if axis is None:
    axis = len(tgt_shape) - 1
    
  target_half_dim = tgt_shape[axis] // 2

  def _pad_and_chunk(arr):
    current_total_size = arr.shape[axis]
    chunk_size = current_total_size // n_shards
    target_chunk_size = target_half_dim // n_shards

    # Safely reshape to expose per-shard chunk without assuming the last axis
    new_shape = list(arr.shape)
    new_shape[axis] = n_shards
    new_shape.insert(axis + 1, chunk_size)
    arr_reshaped = arr.reshape(new_shape)

    pad_amount = target_chunk_size - chunk_size
    if pad_amount > 0:
      pad_widths = [(0, 0)] * arr_reshaped.ndim
      pad_widths[axis + 1] = (0, pad_amount)
      arr_reshaped = jnp.pad(arr_reshaped, pad_widths)
    return arr_reshaped

  p_wi_0 = _pad_and_chunk(wi_0)
  p_wi_1 = _pad_and_chunk(wi_1)

  # Interleave along the chunked dimension
  combined = jnp.concatenate([p_wi_0, p_wi_1], axis=axis + 1)
  return combined.reshape(tgt_shape)

def _align_to_model_shape(
    src_val: jax.Array | np.ndarray,
    tgt_val: jax.Array | np.ndarray,
    key_path: str,
) -> jax.Array | np.ndarray:
  """Aligns src_val to tgt_val's shape via per-axis repeat / zero-pad.

  Thin wrapper around `_align_per_axis` that pulls the target's sharding off
  the target leaf. The per-axis helper is what actually decides repeat vs
  zero-pad per axis and chooses between per-shard and global padding.
  """
  if not (hasattr(src_val, 'shape') and hasattr(tgt_val, 'shape')):
    return src_val
  if src_val.shape == tgt_val.shape:
    return src_val

  tgt_sharding = getattr(tgt_val, 'sharding', None)
  return _align_per_axis(src_val, tgt_val.shape, tgt_sharding, key_path)

def _bulk_align_and_unstack(
    arr: jax.Array | np.ndarray,
    scan_axis: int,
    per_layer_tgt_val: jax.Array | np.ndarray,
    key_path: str,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Applies per-axis alignment on a scanned tensor, then unstacks.

  Operates on the FULL scanned tensor (one bulk repeat / zero-pad per axis,
  performed under JIT so XLA can fuse the pad with surrounding ops), then
  emits `num_layers` per-layer slices. Replaces the prior pattern of unstack
  → align-each-layer, which created N intermediate allocations and N small
  ops.

  The per-layer target's logical axes map to the scanned tensor's axes by
  inserting the scan axis: `scan_padded_axis = a if a < scan_axis else a + 1`.

  Args:
    arr: The full scanned source tensor (shape includes `num_layers` at
      `scan_axis`).
    scan_axis: Axis at which `num_layers` lives in `arr`.
    per_layer_tgt_val: A target leaf — its `.shape`, `.sharding`, and dtype
      drive the alignment policy.
    key_path: Dot-separated source path for diagnostics.

  Returns:
    A tuple of `num_layers` per-layer arrays at the per-layer target shape.
  """
  per_layer_shape = per_layer_tgt_val.shape
  scanned_tgt_shape = (
      per_layer_shape[:scan_axis]
      + (arr.shape[scan_axis],)
      + per_layer_shape[scan_axis:]
  )
  scanned_tgt_sharding = _scanned_sharding_from_per_layer(
      getattr(per_layer_tgt_val, 'sharding', None), scan_axis
  )

  if arr.shape == scanned_tgt_shape:
    return tuple(jnp.unstack(arr, axis=scan_axis))

  aligned = _align_per_axis(
      arr, scanned_tgt_shape, scanned_tgt_sharding, key_path
  )
  return tuple(jnp.unstack(aligned, axis=scan_axis))

def _scanned_sharding_from_per_layer(
    per_layer_sharding: Optional[jax.sharding.Sharding],
    scan_axis: int,
) -> Optional[jax.sharding.NamedSharding]:
  """Builds a scanned `NamedSharding` from a per-layer one by inserting
  `PartitionSpec(None)` at `scan_axis`. Returns None if input isn't a
  NamedSharding."""
  if not isinstance(per_layer_sharding, jax.sharding.NamedSharding):
    return None
  spec = list(per_layer_sharding.spec)
  spec.insert(scan_axis, None)
  return jax.sharding.NamedSharding(
      per_layer_sharding.mesh,
      jax.sharding.PartitionSpec(*spec),
      memory_kind=per_layer_sharding.memory_kind,
  )

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _jit_fuse_single_slice_moe(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    slice_idx: int,
    scan_axis: int,
    n_shards: int,
    tgt_shape: Tuple[int, ...],
    tgt_padded_axis: int,
) -> jax.Array | np.ndarray:
  """Fuses a single slice of wi_0/wi_1 on-demand to minimize HBM footprint."""
  w0_slice = jnp.take(wi_0, slice_idx, axis=scan_axis)
  w1_slice = jnp.take(wi_1, slice_idx, axis=scan_axis)
  return _interleave_moe_weights(
      w0_slice, w1_slice, tgt_shape, n_shards, axis=tgt_padded_axis
  )

@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _jit_extract_single_layer_slice(
    arr: jax.Array | np.ndarray,
    slice_idx: int,
    scan_axis: int,
    tgt_shape: Tuple[int, ...],
    key_path: str,
) -> jax.Array | np.ndarray:
  """Extracts and aligns a single layer slice on-demand to minimize HBM footprint."""
  slice_val = jnp.take(arr, slice_idx, axis=scan_axis)
  return _align_per_axis(slice_val, tgt_shape, None, key_path)

# ==============================================================================
# Modular 4-Stage MaxText-to-MaxText Structural Synchronization Pipeline
# ==============================================================================

_LAYER_PATTERN = re.compile(r'^layers?_(\d+)$')


def _resolve_scanned_path(
    key_tuple: Tuple[Any, ...], src_flat: Mapping[Tuple[Any, ...], Any]
) -> Tuple[Optional[int], Optional[Tuple[Any, ...]], int, Optional[int]]:
  """Stage 2: Resolves an unrolled layer path ('layers_X' or 'layer_X' or ('layers', 'X')) to a scanned candidate in src_flat.

  Returns:
      (layer_idx, candidate_key, match_index, slice_idx) or (None, None, -1, None) if not
      scanned.
  """
  for i, part in enumerate(key_tuple):
    layer_idx = None
    if isinstance(part, str):
      m = _LAYER_PATTERN.match(part)
      if m:
        layer_idx = int(m.group(1))
      elif part.isdigit() and i > 0 and key_tuple[i - 1] in ('layers', 'layer'):
        layer_idx = int(part)
    elif isinstance(part, int) and i > 0 and key_tuple[i - 1] in ('layers', 'layer'):
      layer_idx = part

    if layer_idx is not None:
      prefix = list(key_tuple[:i])
      if i > 0 and key_tuple[i - 1] in ('layers', 'layer'):
        prefix = list(key_tuple[: i - 1])
      suffix = list(key_tuple[i + 1:])
      match_idx = (i - 1 if (i > 0 and key_tuple[i - 1] in ('layers', 'layer')) else i)

      # 1. Inhomogeneous cyclic check (e.g. cycle_len = 4 for Qwen3.5, where src_flat has layer_0, layer_1, etc.)
      for cycle_len in (4, 2, 8, 16):
        cyclic_idx = layer_idx % cycle_len
        slice_idx = layer_idx // cycle_len
        for cand in [
            tuple(prefix + ['layers', f'layer_{cyclic_idx}'] + suffix),
            tuple(prefix + ['layers', str(cyclic_idx)] + suffix),
            tuple(prefix + [f'layer_{cyclic_idx}'] + suffix),
        ]:
          if cand in src_flat:
            return layer_idx, cand, match_idx, slice_idx
          if suffix and suffix[-1] == 'wi':
            wi_0_cand = cand[:-1] + ('wi_0',)
            if wi_0_cand in src_flat:
              return layer_idx, None, match_idx, slice_idx

      # 2. Homogeneous scanned check (e.g. all layers scanned under 'layers')
      for cand in [
          tuple(prefix + ['layers'] + suffix),
          tuple(prefix + suffix),
          tuple(prefix + [f'layers_{layer_idx}'] + suffix),
          tuple(prefix + [f'layer_{layer_idx}'] + suffix),
      ]:
        if cand in src_flat:
          return layer_idx, cand, match_idx, layer_idx
        if suffix and suffix[-1] == 'wi':
          wi_0_cand = cand[:-1] + ('wi_0',)
          if wi_0_cand in src_flat:
            return layer_idx, None, match_idx, layer_idx

      return layer_idx, None, match_idx, None
  return None, None, -1, None


def _align_tensor_to_shape(val: Any, tgt_val: Any, path_str: str) -> Any:
  """Stage 4: Aligns shape and casts dtype of a tensor to match the target specification."""
  val = _apply_dtype_cast(val, tgt_val.dtype, path_str)
  return _align_to_model_shape(val, tgt_val, path_str)


class ScannedLayerUnroller:
  """Stage 3: Manages bulk-unstacking and caching of scanned layer parameters."""

  def __init__(self, src_flat: Mapping[Tuple[Any, ...], Any], scan_axis: int = 1):
    self.src_flat = src_flat
    self.scan_axis = scan_axis
    self._cache = {}

  def _infer_scan_axis(self, src_shape: Tuple[int, ...], tgt_shape: Tuple[int, ...]) -> int:
    """Infers which axis in src_shape is the scan dimension by comparing with tgt_shape."""
    if len(src_shape) == len(tgt_shape) + 1:
      for ax in range(len(src_shape)):
        if src_shape[:ax] + src_shape[ax + 1 :] == tgt_shape:
          return ax
      for ax in range(len(src_shape)):
        candidate = src_shape[:ax] + src_shape[ax + 1 :]
        if _shapes_are_repeatable(candidate, tgt_shape):
          return ax
      for ax in range(len(src_shape)):
        if src_shape[ax] in (4, 10, 18, 40, 48, 64, 72, 80) and len(src_shape) - 1 == len(tgt_shape):
          return ax
    return self.scan_axis

  def get_layer_slice(
      self,
      candidate_key: Tuple[Any, ...],
      layer_idx: int,
      tgt_val: Any,
      candidate_path: str,
  ) -> Any:
    src_val = self.src_flat[candidate_key]
    src_val = _apply_dtype_cast(src_val, tgt_val.dtype, candidate_path)
    if getattr(src_val, 'ndim', -1) == getattr(tgt_val, 'ndim', -2):
      return _align_to_model_shape(src_val, tgt_val, candidate_path)

    scan_axis = self._infer_scan_axis(src_val.shape, tgt_val.shape)
    return _jit_extract_single_layer_slice(
        src_val, layer_idx, scan_axis, tgt_val.shape, candidate_path
    )

  def get_moe_fused_slice(
      self,
      scanned_prefix: Tuple[Any, ...],
      layer_idx: int,
      tgt_val: Any,
  ) -> Optional[Any]:
    wi_0_key = scanned_prefix + ('wi_0',)
    wi_1_key = scanned_prefix + ('wi_1',)
    if wi_0_key not in self.src_flat or wi_1_key not in self.src_flat:
      return None

    wi_0_full = _apply_dtype_cast(
        self.src_flat[wi_0_key],
        tgt_val.dtype,
        '.'.join(str(k) for k in wi_0_key),
    )
    wi_1_full = _apply_dtype_cast(
        self.src_flat[wi_1_key],
        tgt_val.dtype,
        '.'.join(str(k) for k in wi_1_key),
    )

    if getattr(wi_0_full, 'ndim', -1) == getattr(tgt_val, 'ndim', -2):
      mismatched_axes = [
          i for i, (s, t) in enumerate(zip(wi_0_full.shape, tgt_val.shape)) if s != t
      ]
      tgt_axis = (
          mismatched_axes[-1] if mismatched_axes else len(tgt_val.shape) - 1
      )
      n_shards = _get_n_shards(tgt_val, tgt_axis)
      return _interleave_moe_weights(
          wi_0_full, wi_1_full, tgt_val.shape, n_shards, axis=tgt_axis
      )

    scan_axis = self._infer_scan_axis(wi_0_full.shape, tgt_val.shape)
    wi_0_single_shape = (
        wi_0_full.shape[: scan_axis]
        + wi_0_full.shape[scan_axis + 1 :]
    )
    mismatched_axes = [
        i
        for i, (s, t) in enumerate(zip(wi_0_single_shape, tgt_val.shape))
        if s != t
    ]
    tgt_axis = (
        mismatched_axes[-1] if mismatched_axes else len(tgt_val.shape) - 1
    )
    n_shards = _get_n_shards(tgt_val, tgt_axis)
    return _jit_fuse_single_slice_moe(
        wi_0_full,
        wi_1_full,
        layer_idx,
        scan_axis,
        n_shards,
        tgt_val.shape,
        tgt_axis,
    )


def intersect_trees(
    src: Mapping[str, Any],
    tgt_spec: Mapping[str, Any],
    scan_axis: int = 1,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
  """Optimized 4-Stage intersection pipeline for MaxText-to-MaxText synchronization.

  Stages:
    1. Pre-Fusion Pass (`_fuse_moe_weights`): Bulk fuses unscanned MoE wi_0/wi_1 -> wi.
    2. Path Canonicalization (`_resolve_scanned_path`): Maps target 'layers_X' to scanned 'layers'.
    3. Scanned Layer Unrolling & MoE Fusion (`ScannedLayerUnroller`): Bulk-unstacks and caches layers.
    4. Shape Alignment (`_align_tensor_to_shape`): Casts dtype & composes padding/repeating per axis.
  """
  # Fast path for non-dict inputs (leaves)
  if not isinstance(src, abc.Mapping) or not isinstance(tgt_spec, abc.Mapping):
    return src, tgt_spec

  src_flat = traverse_util.flatten_dict(src)
  tgt_flat = traverse_util.flatten_dict(tgt_spec)
  print("DEBUG intersect_trees src_flat sample:", list(src_flat.keys())[:10], flush=True)
  print("DEBUG intersect_trees tgt_flat sample:", list(tgt_flat.keys())[:10], flush=True)

  # Stage 1: Pre-Fusion Pass for Unscanned MoE Weights
  src_flat = _fuse_moe_weights(src_flat, tgt_flat)

  filtered_src_flat = {}
  filtered_tgt_flat = {}

  # Stage 3: Initialize ScannedLayerUnroller (manages bulk-unstack cache & MoE scanned fusion)
  unroller = ScannedLayerUnroller(src_flat, scan_axis=scan_axis)

  for key_tuple, tgt_val in tgt_flat.items():
    path_str = '.'.join(str(k) for k in key_tuple)
    # Skip transient runtime cache/buffer/rng objects (not trainable/transferred weights)
    if any(ignore_str in path_str for ignore_str in (".cache.", ".rngs.", "cache_ar_", "cached_prefill_", "cached_ar_", "rngs.")):
      continue

    # Try 1: Direct Match (Unscanned leaves or global weights)
    if key_tuple in src_flat:
      filtered_src_flat[key_tuple] = _align_tensor_to_shape(
          src_flat[key_tuple], tgt_val, path_str
      )
      filtered_tgt_flat[key_tuple] = tgt_val
      continue

    # Stage 2: Path Canonicalizer (Resolve scanned layer path 'layers_X' -> 'layers')
    layer_idx, cand_key, match_idx, slice_idx = _resolve_scanned_path(key_tuple, src_flat)
    if cand_key is not None and layer_idx is not None and slice_idx is not None:
      cand_path = '.'.join(str(k) for k in cand_key)
      sliced_val = unroller.get_layer_slice(
          cand_key, slice_idx, tgt_val, cand_path
      )
      filtered_src_flat[key_tuple] = _align_tensor_to_shape(
          sliced_val, tgt_val, path_str
      )
      filtered_tgt_flat[key_tuple] = tgt_val
      continue

    # Check Scanned MoE Fusion (e.g. layers/wi_0 + layers/wi_1 -> layers_X/.../wi)
    if (
        key_tuple
        and key_tuple[-1] == 'wi'
        and match_idx != -1
        and layer_idx is not None
        and slice_idx is not None
    ):
      cyclic_idx = layer_idx % 4
      for scanned_prefix in [
          key_tuple[:match_idx]
          + ('layers', f'layer_{cyclic_idx}')
          + key_tuple[match_idx + 1 : -1],
          key_tuple[:match_idx] + ('layers',) + key_tuple[match_idx + 1 : -1],
          key_tuple[:match_idx]
          + ('layers', f'layer_{layer_idx}')
          + key_tuple[match_idx + 1 : -1],
          key_tuple[:match_idx]
          + ('layers', str(cyclic_idx))
          + key_tuple[match_idx + 1 : -1],
          key_tuple[:match_idx]
          + ('layers', str(layer_idx))
          + key_tuple[match_idx + 1 : -1],
          key_tuple[:match_idx]
          + (f'layer_{cyclic_idx}',)
          + key_tuple[match_idx + 1 : -1],
          key_tuple[:match_idx]
          + (f'layer_{layer_idx}',)
          + key_tuple[match_idx + 1 : -1],
          key_tuple[:match_idx] + key_tuple[match_idx + 1 : -1],
      ]:
        sliced_val = unroller.get_moe_fused_slice(
            scanned_prefix, slice_idx, tgt_val
        )
        if sliced_val is not None:
          filtered_src_flat[key_tuple] = _align_tensor_to_shape(
              sliced_val, tgt_val, path_str
          )
          filtered_tgt_flat[key_tuple] = tgt_val
          break
      if key_tuple in filtered_src_flat:
        continue

  print("DEBUG intersect_trees matched keys count:", len(filtered_src_flat), flush=True)
  print("DEBUG intersect_trees matched keys sample:", list(filtered_src_flat.keys())[:10], flush=True)
  return (
      traverse_util.unflatten_dict(filtered_src_flat),
      traverse_util.unflatten_dict(filtered_tgt_flat),
  )
