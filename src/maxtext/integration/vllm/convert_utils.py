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

"""Weight-conversion primitives for MaxText -> vLLM rollout weight sync.

Provenance: most of this module is vendored from `tunix/generate/utils.py`.
The goal is for MaxText to own its whole conversion pipeline rather than reach
into Tunix internals, so these copies are kept deliberately, including the
ones with no current caller; they are the substrate the remaining Tunix
entry points will be ported onto. Do not delete them as dead code.

Vendored from Tunix:
    ShapeMismatchError, _apply_dtype_cast, _shapes_are_repeatable,
    _partition_size, _spec_at_axis, _get_n_shards, _jit_zero_pad_axes,
    _jit_repeat_axes, _align_per_axis, _align_to_model_shape,
    _interleave_moe_weights, _fuse_moe_weights, _unstack_scanned_param,
    _bulk_align_and_unstack, _scanned_sharding_from_per_layer,
    _fuse_and_unstack_moe, _jit_unstack, _delete_target_buffers,
    _reshard_in_chunks

Deliberate divergences from the legacy transfer_state_directly():
    * `_interleave_moe_weights` is `jax.jit`-wrapped here.
    * `_bulk_align_and_unstack` splits via the jitted `_jit_unstack` rather than
      a bare `jnp.unstack`, which eagerly costs one slice dispatch per layer.

Still imported from Tunix at runtime, i.e. the remaining port surface:
    generate.utils.transfer_state_directly, transfer_state_with_mappings,
    build_flat_dict, _unroll_scanned_layers; rl.reshard.reshard_pytree.
    `train_rl.py` also monkeypatches Tunix's `_unstack_scanned_param` and
    `_bulk_align_and_unstack`; that patch goes away once the port lands.
"""

import jax
from absl import logging
from typing import Mapping, Any, Callable, Dict, Tuple, Optional
import functools
import numpy as np
import jax.numpy as jnp


def _delete_target_buffers(tgt_flat: Mapping[str, Any], src_flat: Mapping[str, Any]):
  """Physically deletes target buffers to free HBM before resharding."""
  deleted_count = 0
  preserved_count = 0

  src_buffers = set()
  for v in src_flat.values():
    arr = getattr(v, "value", v)
    if hasattr(arr, "device_buffers"):
      for b in arr.device_buffers():
        src_buffers.add(b)

  for tgt_val in tgt_flat.values():
    tgt_arr = getattr(tgt_val, "value", tgt_val)
    if hasattr(tgt_arr, "device_buffers"):
      is_aliased = any(b in src_buffers for b in tgt_arr.device_buffers())
      if not is_aliased:
        tgt_arr.delete()
        deleted_count += 1
      else:
        preserved_count += 1

  logging.info(
      "Deleted %d non-aliased target buffers (preserved %d aliased) to free HBM.", deleted_count, preserved_count
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
    if tgt_k and tgt_k[-1] == "wi":
      sample_tgt_wi = tgt_v
      break

  for src_key in list(new_src_flat.keys()):
    if not src_key or src_key[-1] != "wi_0":
      continue
    wi_0_key = src_key
    wi_1_key = src_key[:-1] + ("wi_1",)
    wi_target_key = src_key[:-1] + ("wi",)
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
        "Fusing MoE %s: wi_0=%s, wi_1=%s -> %s on axis %d",
        ".".join(str(k) for k in wi_target_key),
        wi_0.shape,
        wi_1.shape,
        fused_shape_tuple,
        axis,
    )
    new_src_flat[wi_target_key] = _interleave_moe_weights(wi_0, wi_1, fused_shape_tuple, n_shards, axis=axis)
    del wi_0, wi_1

  return new_src_flat


class ShapeMismatchError(ValueError):
  """Raised when source and target shapes are incompatible."""


def _apply_dtype_cast(val: jax.Array | np.ndarray, tgt_dtype: jnp.dtype, src_key: str) -> jax.Array | np.ndarray:
  """Casts val to target dtype if needed, logging a warning on type mismatch."""
  if val.dtype != tgt_dtype:
    logging.log_first_n(
        logging.WARNING,
        "Type mismatch on %s: %s -> %s",
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
  if not (hasattr(src_val, "shape") and hasattr(tgt_val, "shape")):
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
        if hasattr(src_val, "transpose"):
          src_val = src_val.transpose(perm)
        elif isinstance(src_val, np.ndarray):
          src_val = np.transpose(src_val, perm)

      # Unstack along the 0th axis
      if hasattr(jax, "unstack"):
        return tuple(jax.unstack(src_val))
      elif hasattr(jnp, "unstack"):
        return tuple(jnp.unstack(src_val))
      else:
        return tuple(src_val[i] for i in range(src_val.shape[0]))
    else:
      logging.warning(
          "Shape mismatch in scanned param '%s'. Src: %s, Tgt: %s. Cannot" " determine scan axis.",
          key_path,
          src_shape,
          tgt_shape,
      )

  return (src_val,)


# Leaf names whose axis mismatches must be closed by zero-padding rather than
# by repeating. `wo` belongs here for a semantic reason, not just by analogy
# with `wi`: the MoE intermediate dim is `wo`'s *contracting* (input) axis, so
# zero rows contribute nothing to the output, whereas repeating rows would
# double-count every padded lane. Repeat is not merely suboptimal for `wo`, it
# is numerically wrong.
_MOE_MLP_WEIGHTS = frozenset({"wi", "wi_0", "wi_1", "wo"})


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
  sharding = getattr(arr, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    return _partition_size(_spec_at_axis(sharding, axis), sharding.mesh)  # pyrefly: ignore[bad-argument-type]
  return 1


def _device_ids(x: Any) -> Tuple[int, ...]:
  """Sorted device ids backing an array's sharding, or () if unplaced."""
  arr = getattr(x, "value", x)
  sharding = getattr(arr, "sharding", None)
  if sharding is None:
    return ()
  mesh = getattr(sharding, "mesh", None)
  if mesh is not None and hasattr(mesh, "devices"):
    devices = np.asarray(mesh.devices).flatten().tolist()
  elif hasattr(sharding, "device_set"):
    devices = list(sharding.device_set)
  else:
    return ()
  return tuple(sorted(int(d.id) for d in devices if hasattr(d, "id")))


def _sharding_summary(x: Any) -> str:
  """One-line description of an array's placement, for weight-sync debugging.

  Reports the device-id span of the backing mesh so a cross-mesh operand is
  obvious at a glance. On a split Pathways cluster the trainer holds the low
  half of the id range and the rollout the high half (see
  `model_creation_utils.setup_configs_and_devices`), so any *source* operand
  reported outside the trainer span -- or any converter output landing outside
  it -- is a mesh leak.
  """
  arr = getattr(x, "value", x)
  shape = getattr(arr, "shape", None)
  sharding = getattr(arr, "sharding", None)
  if sharding is None:
    return f"<unsharded> shape={shape}"
  ids = _device_ids(arr)
  span = f"[{ids[0]}..{ids[-1]}]({len(ids)})" if ids else "[unknown]"
  spec = getattr(sharding, "spec", None)
  return f"{type(sharding).__name__} shape={shape} devices={span} spec={spec}"


@functools.partial(jax.jit, static_argnames=("pad_specs",))
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


@functools.partial(jax.jit, static_argnames=("axis",))
def _jit_unstack(arr, axis: int):
  """Splits `arr` along `axis` inside a single XLA program.

  Eager `jnp.unstack` lowers to one slice dispatch per element, so unrolling a
  48-layer scanned parameter costs 48 launches. Under jit the whole fan-out is
  one execution returning 48 results.
  """
  return tuple(jnp.unstack(arr, axis=axis))


@functools.partial(jax.jit, static_argnames=("repeats",))
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
  if not hasattr(arr, "shape"):
    return arr
  if arr.shape == tgt_shape:
    return arr
  if len(arr.shape) != len(tgt_shape):
    raise ShapeMismatchError(f"Rank mismatch for {key_path}: src={arr.shape} vs tgt={tgt_shape}")

  mismatches = []
  for axis, (s, t) in enumerate(zip(arr.shape, tgt_shape)):
    if s == t:
      continue
    if t < s:
      raise ShapeMismatchError(f"Cannot shrink axis {axis} for {key_path}: src={s} -> tgt={t}")
    mismatches.append((axis, s, t))
  if not mismatches:
    return arr

  last_key = key_path.split(".")[-1]
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


@functools.partial(jax.jit, static_argnames=("tgt_shape", "n_shards", "axis"))
def _interleave_moe_weights(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    tgt_shape: Tuple[int, ...],
    n_shards: int,
    axis: Optional[int] = None,
) -> jax.Array | np.ndarray:
  """Interleaves wi_0 and wi_1 per-shard into a single tensor.

  JIT-compiled: run eagerly this is 2 reshapes + 2 `jnp.pad` + 1 concatenate +
  1 reshape, and every intermediate becomes a materialized device buffer. Under
  one trace XLA fuses the pads into the concatenate's output write, so the only
  buffer allocated is the result. On a 48-layer MoE model called once per layer
  that is the difference between ~3x and ~1x the output size in live transient
  memory, plus ~5 fewer dispatches per call.

  `tgt_shape`, `n_shards` and `axis` are static, so the trace is keyed on them;
  identical layers share a single compilation.
  """
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


@functools.partial(
    jax.jit,
    static_argnames=(
        "scan_axis",
        "n_shards",
        "tgt_shape",
        "scan_fused_axis",
        "tgt_fused_axis",
    ),
)
def _fuse_and_unstack_moe(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    scan_axis: int,
    n_shards: int,
    tgt_shape: Tuple[int, ...],
    scan_fused_axis: int,
    tgt_fused_axis: int,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Fuses wi_0/wi_1 per unstacked layer to keep peak intermediate HBM allocation low.

  Args:
    wi_0: Scanned gate kernel, `num_blocks` at `scan_axis`.
    wi_1: Scanned up kernel, same layout as `wi_0`.
    scan_axis: Axis holding `num_blocks` in `wi_0` / `wi_1`.
    n_shards: Mesh shards along the per-layer fused axis.
    tgt_shape: *Per-layer* fused target shape.
    scan_fused_axis: Position of the fused axis in the scanned layout.
    tgt_fused_axis: Position of the fused axis in the per-layer layout.

  Returns:
    `num_blocks` per-layer arrays, each of shape `tgt_shape`.
  """
  unstacked_0 = jnp.unstack(wi_0, axis=scan_axis)
  unstacked_1 = jnp.unstack(wi_1, axis=scan_axis)
  return tuple(
      _interleave_moe_weights(w0, w1, tgt_shape, n_shards, axis=tgt_fused_axis)
      for w0, w1 in zip(unstacked_0, unstacked_1)
  )


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
  if not (hasattr(src_val, "shape") and hasattr(tgt_val, "shape")):
    return src_val
  if src_val.shape == tgt_val.shape:
    return src_val

  tgt_sharding = getattr(tgt_val, "sharding", None)
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
  scanned_tgt_shape = per_layer_shape[:scan_axis] + (arr.shape[scan_axis],) + per_layer_shape[scan_axis:]
  scanned_tgt_sharding = _scanned_sharding_from_per_layer(getattr(per_layer_tgt_val, "sharding", None), scan_axis)

  if arr.shape == scanned_tgt_shape:
    return _jit_unstack(arr, scan_axis)

  aligned = _align_per_axis(arr, scanned_tgt_shape, scanned_tgt_sharding, key_path)
  return _jit_unstack(aligned, scan_axis)


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
