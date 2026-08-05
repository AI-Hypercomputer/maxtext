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
    keys = list(spec_flat.keys())
    
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i : i + chunk_size]
        src_chunk = {k: src_flat[k] for k in chunk_keys if k in src_flat}
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
  """Fuses unscanned wi_0/wi_1 into wi for unscanned-fused targets.

  Only catches the case where source and target share the same prefix (e.g.
  src `('layers', 'wi_0')` + `('layers', 'wi_1')` → tgt `('layers', 'wi')`,
  or src `('layers_0', 'wi_0')` + `('layers_0', 'wi_1')` →
  tgt `('layers_0', 'wi')`). The scanned-source / unrolled-target case is
  handled in `intersect_trees` via `_jit_fuse_and_unstack_moe`.

  Args:
    src_flat: Flat dict of source key tuples to JAX arrays.
    tgt_flat: Flat dict of target key tuples to target leaves.

  Returns:
    A new flat dict with wi_0/wi_1 fused into wi at matching prefixes. Any
    remaining shape mismatch on non-fused axes is left for the per-target
    `_align_to_model_shape` call to handle (it composes repeat + zero-pad).
  """
  new_src_flat = dict(src_flat)
  for tgt_key in tgt_flat.keys():
    if not tgt_key or tgt_key[-1] != 'wi':
      continue
    wi_0_key = tgt_key[:-1] + ('wi_0',)
    wi_1_key = tgt_key[:-1] + ('wi_1',)
    if wi_0_key not in new_src_flat or wi_1_key not in new_src_flat:
      continue
    wi_0 = new_src_flat.pop(wi_0_key)
    wi_1 = new_src_flat.pop(wi_1_key)
    tgt_val = tgt_flat[tgt_key]
    # Pick the fused axis as the last axis where src and tgt differ. For the
    # canonical wi_0/wi_1 -> wi case this is the last axis (the mlp dim).
    mismatched_axes = [
        i for i, (s, t) in enumerate(zip(wi_0.shape, tgt_val.shape)) if s != t
    ]
    axis = mismatched_axes[-1] if mismatched_axes else len(tgt_val.shape) - 1
    n_shards = _get_n_shards(tgt_val, axis)
    logging.info(
        'Fusing MoE %s: wi_0=%s, wi_1=%s -> %s on axis %d',
        '.'.join(str(k) for k in tgt_key),
        wi_0.shape, wi_1.shape, tgt_val.shape, axis,
    )
    new_src_flat[tgt_key] = _interleave_moe_weights(
        wi_0, wi_1, tgt_val.shape, n_shards, axis=axis
    )
    del wi_0, wi_1  # Free memory immediately after fusion.
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

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def _jit_fuse_and_unstack_moe(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    scan_axis: int,
    num_layers: int,
    n_shards: int,
    tgt_shape: Tuple[int, ...],
    scan_padded_axis: int,
    tgt_padded_axis: int,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Fuses wi_0/wi_1 along the padded axis, then unstacks along scan_axis.

  By combining concat and unstack under jax.jit, XLA fuses both ops and
  avoids materializing the full concatenated intermediate on device.

  Args:
    wi_0: First MoE gate weight; per-layer layout matches `tgt_shape`
      (with the padded dim halved), with `num_layers` inserted at `scan_axis`.
    wi_1: Second MoE gate weight, same layout as `wi_0`.
    scan_axis: Axis at which `num_layers` is stacked in `wi_0` / `wi_1`.
    num_layers: Number of layers (== `wi_0.shape[scan_axis]`).
    n_shards: Mesh shards along the per-layer fused/padded axis.
    tgt_shape: Per-layer fused target shape (fused mlp dim on `tgt_padded_axis`).
    scan_padded_axis: Position of the padded axis in the scanned layout.
    tgt_padded_axis: Position of the padded axis in the per-layer layout.

  Returns:
    Tuple of `num_layers` per-layer arrays, each with shape `tgt_shape`.
  """
  del num_layers  # Only used to make this a static arg for JIT cache keying.
  fused_shape = list(wi_0.shape)
  fused_shape[scan_padded_axis] = tgt_shape[tgt_padded_axis]

  fused = _interleave_moe_weights(
      wi_0, wi_1, tuple(fused_shape), n_shards, axis=scan_padded_axis
  )
  return jnp.unstack(fused, axis=scan_axis)

# ==============================================================================
# Modular 4-Stage MaxText-to-MaxText Structural Synchronization Pipeline
# ==============================================================================

_LAYER_PATTERN = re.compile(r'^layers_(\d+)$')


def _resolve_scanned_path(
    key_tuple: Tuple[Any, ...], src_flat: Mapping[Tuple[Any, ...], Any]
) -> Tuple[Optional[int], Optional[Tuple[Any, ...]], int]:
  """Stage 2: Resolves an unrolled layer path ('layers_X') to a scanned candidate in src_flat.

  Returns:
      (layer_idx, candidate_key, match_index) or (None, None, -1) if not
      scanned.
  """
  for i, part in enumerate(key_tuple):
    if isinstance(part, str) and part.startswith('layers_'):
      m = _LAYER_PATTERN.match(part)
      if m:
        layer_idx = int(m.group(1))
        # Candidate A: Replace 'layers_X' with 'layers' (Standard MaxText)
        candidate_a = list(key_tuple)
        candidate_a[i] = 'layers'
        # Candidate B: Remove 'layers_X' (Implicit Container / GPT-OSS)
        candidate_b = list(key_tuple)
        candidate_b.pop(i)

        for cand in [tuple(candidate_a), tuple(candidate_b)]:
          if cand in src_flat:
            return layer_idx, cand, i
        return layer_idx, None, i
  return None, None, -1


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

  def get_layer_slice(
      self,
      candidate_key: Tuple[Any, ...],
      layer_idx: int,
      tgt_val: Any,
      candidate_path: str,
  ) -> Any:
    cache_key = (candidate_key, tgt_val.shape, 'aligned')
    if cache_key not in self._cache:
      src_val = self.src_flat[candidate_key]
      src_val = _apply_dtype_cast(src_val, tgt_val.dtype, candidate_path)
      scanned_per_layer_shape = (
          src_val.shape[: self.scan_axis] + src_val.shape[self.scan_axis + 1 :]
      )
      if scanned_per_layer_shape == tgt_val.shape:
        self._cache[cache_key] = _unstack_scanned_param(
            src_val, tgt_val, candidate_path, scan_axis=self.scan_axis
        )
      else:
        logging.info(
            'Bulk-aligning scanned %s: %s -> per-layer %s',
            candidate_path,
            src_val.shape,
            tgt_val.shape,
        )
        self._cache[cache_key] = _bulk_align_and_unstack(
            src_val, self.scan_axis, tgt_val, candidate_path
        )
    return self._cache[cache_key][layer_idx]

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

    fused_scanned_key = scanned_prefix + ('wi_fused',)
    if fused_scanned_key not in self._cache:
      scanned_prefix_path = '.'.join(str(k) for k in scanned_prefix)
      logging.info('Fusing scanned MoE weights for %s', scanned_prefix_path)
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
      num_layers = self.src_flat[wi_0_key].shape[self.scan_axis]

      wi_0_single_shape = (
          wi_0_full.shape[: self.scan_axis]
          + wi_0_full.shape[self.scan_axis + 1 :]
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

      scan_padded_axis = (
          tgt_axis if tgt_axis < self.scan_axis else tgt_axis + 1
      )
      self._cache[fused_scanned_key] = _jit_fuse_and_unstack_moe(
          wi_0_full,
          wi_1_full,
          self.scan_axis,
          num_layers,
          n_shards,
          tgt_val.shape,
          scan_padded_axis,
          tgt_axis,
      )
    return self._cache[fused_scanned_key][layer_idx]


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

  # Stage 1: Pre-Fusion Pass for Unscanned MoE Weights
  src_flat = _fuse_moe_weights(src_flat, tgt_flat)

  filtered_src_flat = {}
  filtered_tgt_flat = {}

  # Stage 3: Initialize ScannedLayerUnroller (manages bulk-unstack cache & MoE scanned fusion)
  unroller = ScannedLayerUnroller(src_flat, scan_axis=scan_axis)

  for key_tuple, tgt_val in tgt_flat.items():
    path_str = '.'.join(str(k) for k in key_tuple)

    # Try 1: Direct Match (Unscanned leaves or global weights)
    if key_tuple in src_flat:
      filtered_src_flat[key_tuple] = _align_tensor_to_shape(
          src_flat[key_tuple], tgt_val, path_str
      )
      filtered_tgt_flat[key_tuple] = tgt_val
      continue

    # Stage 2: Path Canonicalizer (Resolve scanned layer path 'layers_X' -> 'layers')
    layer_idx, cand_key, match_idx = _resolve_scanned_path(key_tuple, src_flat)
    if cand_key is not None and layer_idx is not None:
      cand_path = '.'.join(str(k) for k in cand_key)
      sliced_val = unroller.get_layer_slice(
          cand_key, layer_idx, tgt_val, cand_path
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
    ):
      scanned_prefix = (
          key_tuple[:match_idx] + ('layers',) + key_tuple[match_idx + 1 : -1]
      )
      sliced_val = unroller.get_moe_fused_slice(
          scanned_prefix, layer_idx, tgt_val
      )
      if sliced_val is not None:
        filtered_src_flat[key_tuple] = _align_tensor_to_shape(
            sliced_val, tgt_val, path_str
        )
        filtered_tgt_flat[key_tuple] = tgt_val
        continue

  return (
      traverse_util.unflatten_dict(filtered_src_flat),
      traverse_util.unflatten_dict(filtered_tgt_flat),
  )
