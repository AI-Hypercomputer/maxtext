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

"""Streams a HuggingFace SafeTensors checkpoint straight into sharded MaxText weights.

The checkpoint is read a few decoder layers at a time: whole layers are packed
into Orbax read calls of about `READ_BYTES_PER_HOST` per host, starting with the
tensors outside the decoder layers (embeddings, final norm, lm_head). For each
read call:

1. Orbax reads only that call's HF tensors from storage into HBM, each split
   along dim 0 across the TPU chips. SafeTensors is row-major, so every piece is
   one unbroken byte range in the file.
2. The HF->MaxText hooks from `param_mapping.py` (transpose, reshape, RoPE
   permutation, ...) run on the TPU chips, exactly as in the offline conversion.
3. A jitted writer casts each result to the MaxText dtype, moves it into the
   MaxText sharding, and writes it in place into the preallocated MaxText weight
   (stacked along the layer and/or expert axes where the mapping says so).

Peak HBM is therefore about the final MaxText weights plus one call's HF
tensors, and peak host RAM about one call's HF tensors, instead of the whole HF
checkpoint.

Requires orbax-checkpoint>=0.12.3, whose SafeTensors loader reads only the
requested tensors and only the byte ranges each process's TPU chips need.
"""

import collections
import dataclasses
import functools
import importlib.metadata
import math
import re
import time
from typing import Any, Iterator

import flax.traverse_util
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import tensor_handling
from maxtext.utils import max_logging
import numpy as np
from orbax.checkpoint import v1 as ocp_v1

# HF tensors smaller than this get a full copy on every TPU chip: splitting them
# saves almost no memory and costs extra read requests.
MIN_BYTES_TO_SPLIT = 1 << 20

# Every Orbax read call has a fixed cost (it re-reads the header of every file),
# and one call keeps at most 2 GiB of reads in flight. Packing whole decoder
# layers into calls of about this many bytes per host keeps enough reads in
# flight and pays that cost rarely, while HBM and host RAM hold only one call's
# HF tensors at a time. On a v4-8 host this read Qwen3-8B (16 GB) in ~13 s; one
# layer per call took ~80 s, and one call for everything ~11 s.
READ_BYTES_PER_HOST = 4 << 30

# Size of each ranged read Orbax sends to storage. On a v4-8 host, 32 MiB was
# ~15% faster than Orbax's 128 MiB default; 16 MiB was slower than both.
READ_CHUNK_BYTES = 32 << 20

_MIN_ORBAX_VERSION = (0, 12, 3)
_LOAD_MESH_AXES = ("rows", "copies")
# "model.layers.12.self_attn.q_proj.weight" -> ("model.", "12").
_LAYER_KEY = re.compile(r"^(.*?\.)?layers\.(\d+)\.")
# Sorts before every decoder-layer group.
_NON_LAYER_GROUP = ("", -1)


def load_split_factor(shape, dtype, num_devices, min_bytes_to_split=MIN_BYTES_TO_SPLIT) -> int:
  """Returns how many pieces to split an HF tensor into along dim 0 while loading it.

  Only dim 0 is ever split: SafeTensors is row-major, so each piece is then one
  unbroken byte range in the file, whatever the number of dimensions. The factor
  is gcd(dim 0, num_devices), so the pieces are equal-sized and each piece goes
  to num_devices / factor TPU chips.

  Args:
    shape: Shape of the HF tensor.
    dtype: Dtype of the HF tensor, as stored in the file.
    num_devices: Number of TPU chips the tensor is loaded onto.
    min_bytes_to_split: Tensors smaller than this are not split.

  Returns:
    The number of pieces. 1 means every TPU chip gets a full copy, which is used
    for scalars, tensors under `min_bytes_to_split`, and a dim 0 that shares no
    factor with `num_devices`.
  """
  if not shape:
    return 1
  if math.prod(shape) * np.dtype(dtype).itemsize < min_bytes_to_split:
    return 1
  return math.gcd(shape[0], num_devices)


@functools.lru_cache(maxsize=None)
def _load_mesh(devices: tuple, factor: int) -> jax.sharding.Mesh:
  grid = np.array(devices, dtype=object).reshape(factor, len(devices) // factor)
  return jax.sharding.Mesh(grid, _LOAD_MESH_AXES)


def choose_load_sharding(shape, dtype, devices, min_bytes_to_split=MIN_BYTES_TO_SPLIT) -> jax.sharding.NamedSharding:
  """Returns the sharding an HF tensor is loaded with (see `load_split_factor`)."""
  devices = tuple(devices)
  factor = load_split_factor(shape, dtype, len(devices), min_bytes_to_split)
  spec = jax.sharding.PartitionSpec(_LOAD_MESH_AXES[0]) if factor > 1 else jax.sharding.PartitionSpec()
  return jax.sharding.NamedSharding(_load_mesh(devices, factor), spec)


@dataclasses.dataclass(eq=False)
class _Target:
  """A MaxText weight that one or more HF tensors are written into."""

  name: str  # Key in the flattened target tree.
  mt_key: str  # Key in the param mapping.
  shape: tuple[int, ...]
  dtype: Any
  sharding: jax.sharding.Sharding
  axes: tuple[int, ...]  # Stacked axes, outermost first; () if not stacked.
  hook_shape: tuple[int, ...]  # Shape the hooks must produce: `shape` without `axes`.
  hooks: Any


@dataclasses.dataclass(frozen=True)
class _Write:
  """Writes the hooked HF tensor(s) `hf_source` into `target` at `index` (one entry per stacked axis)."""

  target: _Target
  index: tuple[int, ...]
  hf_source: str | tuple[str, ...]  # One HF key, or a tuple the hook fuses into one tensor.

  @property
  def hf_keys(self) -> tuple[str, ...]:
    return self.hf_source if isinstance(self.hf_source, tuple) else (self.hf_source,)


@dataclasses.dataclass
class LoadPlan:
  """What to load and where to write it, grouped by decoder layer."""

  targets: list[_Target]
  groups: list[list[_Write]]
  unmatched_mt_keys: list[str]  # Mapping entries with no weight in the target tree.


def resolve_target_name(mt_key: str, flat_target: dict) -> str | None:
  """Finds the key in the flattened target tree that a param-mapping key refers to."""
  mt_name = mt_key.replace("params-", "").replace("-", ".")
  for candidate in (mt_name, f"params.{mt_name}", mt_key.replace("-", ".")):
    if candidate in flat_target:
      return candidate
  return None


def _stacked_axes(mt_key: str, shape: tuple, depth: int, config) -> tuple[int, ...]:
  """Where each nesting level of the HF key list lands in the MaxText weight.

  Matches `tensor_handling`: a flat list stacks along `param_scan_axis` for
  scanned layers (or axis 0 for rank-1 weights and unscanned MoE experts), and
  deeper nesting follows `tensor_handling.stacked_axes`.
  """
  if depth == 0:
    return ()
  if depth == 1:
    scan_axis = config.param_scan_axis
    return (scan_axis,) if config.scan_layers and len(shape) > scan_axis else (0,)
  return tuple(tensor_handling.stacked_axes(mt_key, config, depth))


def _enumerate_sources(hf_source: Any, target: _Target) -> Iterator[tuple[tuple[int, ...], Any]]:
  """Yields `(index, hf_source)` per HF entry, checking the nested lists exactly fill the stacked axes.

  The weights are preallocated, so a short list would otherwise leave part of a
  weight silently zero.
  """

  def walk(node, level, index):
    if level == len(target.axes):
      if not isinstance(node, (str, tuple)):
        raise ValueError(
            f"Param mapping for {target.mt_key} nests unevenly: expected an HF key at depth {level}, got {node!r}."
        )
      yield index, node
      return
    axis = target.axes[level]
    if not isinstance(node, list) or len(node) != target.shape[axis]:
      got = f"{len(node)} entries" if isinstance(node, list) else repr(node)
      raise ValueError(
          f"Param mapping for {target.mt_key} gives {got} at stacking level {level}, but MaxText weight"
          f" {target.name} has shape {target.shape} with {target.shape[axis]} along axis {axis}."
      )
    for i, child in enumerate(node):
      yield from walk(child, level + 1, index + (i,))

  yield from walk(hf_source, 0, ())


@functools.lru_cache(maxsize=None)
def _replicated_sharding(devices: tuple) -> jax.sharding.NamedSharding:
  mesh = jax.sharding.Mesh(np.array(devices, dtype=object), ("replica",))
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _devices_of(sharding: jax.sharding.Sharding) -> tuple:
  if isinstance(sharding, jax.sharding.NamedSharding):
    return tuple(sharding.mesh.devices.flat)
  return tuple(sorted(sharding.device_set, key=lambda d: d.id))


def _target_sharding(leaf: Any, name: str) -> jax.sharding.Sharding:
  sharding = getattr(leaf, "sharding", None)
  if sharding is None:
    max_logging.warning(f"{name} has no sharding in the target tree; keeping a full copy on every device.")
    sharding = _replicated_sharding(tuple(jax.devices()))
  return sharding


def _layer_group(hf_key: str) -> tuple[str, int]:
  match = _LAYER_KEY.match(hf_key)
  return (match.group(1) or "", int(match.group(2))) if match else _NON_LAYER_GROUP


def _call_name(call: list[_Write]) -> str:
  """Names the layers one read call covers, e.g. "non-layer tensors + model.layers.0-3"."""
  layers = collections.defaultdict(set)
  for write in call:
    prefix, layer = _layer_group(write.hf_keys[0])
    layers[prefix].add(layer)
  parts = ["non-layer tensors"] if _NON_LAYER_GROUP[1] in layers.get(_NON_LAYER_GROUP[0], ()) else []
  for prefix, numbers in layers.items():
    numbers = sorted(n for n in numbers if n >= 0)
    if numbers:
      parts.append(f"{prefix}layers.{numbers[0]}" + (f"-{numbers[-1]}" if len(numbers) > 1 else ""))
  return " + ".join(parts)


def _nbytes(meta: Any) -> int:
  """Size in bytes of a tensor given anything with `.shape` and `.dtype`."""
  return math.prod(meta.shape) * np.dtype(meta.dtype).itemsize


def _group_bytes(group: list[_Write], hf_metadata: dict) -> int:
  return sum(_nbytes(hf_metadata[key]) for key in {key for write in group for key in write.hf_keys})


def pack_groups(sizes: list[int], max_bytes: int | None) -> list[range]:
  """Packs consecutive groups into read calls of at most `max_bytes` each.

  Groups stay whole and in order; a group bigger than `max_bytes` gets a call of
  its own.

  Args:
    sizes: Bytes of HF tensors in each group, in load order.
    max_bytes: Most bytes per call; None reads everything in one call.

  Returns:
    The group indices of each read call.
  """
  calls, start, total = [], 0, 0
  for i, size in enumerate(sizes):
    if max_bytes is not None and i > start and total + size > max_bytes:
      calls.append(range(start, i))
      start, total = i, 0
    total += size
  if sizes:
    calls.append(range(start, len(sizes)))
  return calls


def build_plan(param_map: dict, hook_map: dict, target_tree: Any, config) -> LoadPlan:
  """Plans the load: one `_Target` per mapped MaxText weight and one `_Write` per HF source.

  Writes are grouped by the decoder layer in their HF key (`...layers.<i>....`),
  with every other HF tensor in one extra group, loaded first.

  Args:
    param_map: MaxText key -> HF key(s), as returned by `param_mapping.PARAM_MAPPING[model]`.
    hook_map: MaxText key -> hook function(s), as returned by `param_mapping.HOOK_FNS[model]`.
    target_tree: Abstract MaxText weights (e.g. `jax.ShapeDtypeStruct`s with shardings).
    config: The MaxText config (only `scan_layers` and `param_scan_axis` are read).

  Returns:
    The load plan.
  """
  flat_target = flax.traverse_util.flatten_dict(target_tree, sep=".")
  targets, writes, unmatched = [], [], []
  for mt_key, hf_source in param_map.items():
    name = resolve_target_name(mt_key, flat_target)
    if name is None:
      unmatched.append(mt_key)
      continue
    leaf = flat_target[name]
    shape = tuple(leaf.shape)
    axes = _stacked_axes(mt_key, shape, tensor_handling.nesting_depth(hf_source), config)
    target = _Target(
        name=name,
        mt_key=mt_key,
        shape=shape,
        dtype=np.dtype(leaf.dtype),
        sharding=_target_sharding(leaf, name),
        axes=axes,
        hook_shape=tensor_handling.slice_shape(shape, axes),
        hooks=hook_map.get(mt_key),
    )
    targets.append(target)
    writes.extend(_Write(target, index, source) for index, source in _enumerate_sources(hf_source, target))

  groups = collections.defaultdict(list)
  for write in writes:
    groups[_layer_group(write.hf_keys[0])].append(write)
  return LoadPlan(targets=targets, groups=[groups[k] for k in sorted(groups)], unmatched_mt_keys=unmatched)


@functools.lru_cache(maxsize=None)
def _alloc_fn(shape, dtype, sharding):
  return jax.jit(lambda: jnp.zeros(shape, dtype), out_shardings=sharding)


@functools.lru_cache(maxsize=None)
def _cast_fn(dtype, sharding):
  return jax.jit(lambda x: x.astype(dtype), out_shardings=sharding)


@functools.lru_cache(maxsize=None)
def _write_fn(dtype, sharding, axes, ndim):
  """Jitted `(stacked, x, *index) -> stacked` with x written in place at `index` along `axes`.

  `index` is traced, so every layer (and expert) reuses one compiled program.
  """

  def write(stacked, x, *index):
    update = jnp.expand_dims(x.astype(dtype), sorted(axes))
    start = [0] * ndim
    for axis, i in zip(axes, index):
      start[axis] = i
    return jax.lax.dynamic_update_slice(stacked, update, start)

  return jax.jit(write, donate_argnums=0, out_shardings=sharding)


def _check_orbax_version():
  try:
    version = tuple(int(part) for part in importlib.metadata.version("orbax-checkpoint").split(".")[:3])
  except (importlib.metadata.PackageNotFoundError, ValueError):
    return  # Unknown or pre-release version; assume it is recent enough.
  if version < _MIN_ORBAX_VERSION:
    raise RuntimeError(
        f"Streaming SafeTensors loading needs orbax-checkpoint>={'.'.join(map(str, _MIN_ORBAX_VERSION))}"
        f" (found {'.'.join(map(str, version))}). Older versions read every file in full on each call."
    )


def _check_sources(plan: LoadPlan, hf_metadata: dict):
  """Raises if a mapped HF tensor is missing from the checkpoint; logs HF tensors nothing reads."""
  needed = {}
  for group in plan.groups:
    for write in group:
      for key in write.hf_keys:
        needed.setdefault(key, write.target.mt_key)
  missing = [key for key in needed if key not in hf_metadata]
  if missing:
    examples = ", ".join(f"{key} (for {needed[key]})" for key in missing[:10])
    raise ValueError(f"{len(missing)} HF tensors in the param mapping are not in the checkpoint, e.g. {examples}.")
  unused = sorted(set(hf_metadata) - set(needed))
  if unused:
    max_logging.log(f"Not loading {len(unused)} HF tensors that no mapped MaxText weight uses, e.g. {unused[:5]}.")
  if plan.unmatched_mt_keys:
    max_logging.log(
        f"Skipping {len(plan.unmatched_mt_keys)} param-mapping entries with no weight in the target tree,"
        f" e.g. {plan.unmatched_mt_keys[:5]}."
    )


def _load_request(call: list[_Write], hf_metadata: dict, min_bytes_to_split: int) -> dict:
  """The flat abstract tree that makes Orbax read this call's HF tensors, split by rows."""
  request = {}
  for write in call:
    devices = _devices_of(write.target.sharding)
    for key in write.hf_keys:
      if key not in request:
        meta = hf_metadata[key]
        # Request the file's own dtype: Orbax would otherwise cast on the host.
        sharding = choose_load_sharding(meta.shape, meta.dtype, devices, min_bytes_to_split)
        request[key] = jax.ShapeDtypeStruct(meta.shape, meta.dtype, sharding=sharding)
  return request


def _apply_write(write: _Write, hf_arrays: dict, results: dict):
  """Hooks one HF source, then casts and writes it into its MaxText weight."""
  target = write.target
  if isinstance(write.hf_source, tuple):
    raw = tuple(hf_arrays[key] for key in write.hf_source)
  else:
    raw = hf_arrays[write.hf_source]
  # The hooks run op by op, as in the offline conversion. Tracing them into one
  # jitted program would let XLA skip intermediate roundings (e.g. a bf16 cast
  # inside a hook), so the result could differ from `to_maxtext` in the last bits.
  x = tensor_handling.apply_hook_fns(raw, target.hook_shape, target.hooks)
  if tuple(x.shape) != target.hook_shape:
    raise ValueError(
        f"Hooks for {target.mt_key} turned {write.hf_source} into shape {tuple(x.shape)}, but MaxText weight"
        f" {target.name} needs {target.hook_shape} per entry (full shape {target.shape})."
    )
  if isinstance(x, jax.Array) and x.sharding.device_set != target.sharding.device_set:
    # Only when the target lives on a different set of devices than we loaded onto.
    x = jax.device_put(x, _replicated_sharding(_devices_of(target.sharding)))
  if target.axes:
    write_fn = _write_fn(target.dtype, target.sharding, target.axes, len(target.shape))
    results[target.name] = write_fn(results[target.name], x, *write.index)
  else:
    results[target.name] = _cast_fn(target.dtype, target.sharding)(x)


def _peak_hbm_gb() -> float | None:
  stats = jax.local_devices()[0].memory_stats()
  return stats["peak_bytes_in_use"] / 1e9 if stats and "peak_bytes_in_use" in stats else None


def load_hf_params_streaming(
    path: str,
    target_tree: Any,
    param_map: dict,
    hook_map: dict,
    config,
    min_bytes_to_split: int = MIN_BYTES_TO_SPLIT,
    read_bytes_per_host: int | None = READ_BYTES_PER_HOST,
    read_chunk_bytes: int = READ_CHUNK_BYTES,
) -> dict:
  """Loads an HF SafeTensors checkpoint into MaxText weights, a few decoder layers at a time.

  Args:
    path: Directory (local or gs://) holding the `.safetensors` files.
    target_tree: Abstract MaxText weights: a nested dict of `jax.ShapeDtypeStruct`s carrying
      the target shardings. Linen's `params` collection or an NNX pure dict both work.
    param_map: MaxText key -> HF key(s), from `param_mapping.PARAM_MAPPING[model]`.
    hook_map: MaxText key -> hook function(s), from `param_mapping.HOOK_FNS[model]`.
    config: The MaxText config (`scan_layers` and `param_scan_axis` are read).
    min_bytes_to_split: HF tensors smaller than this are loaded as a full copy on every TPU chip.
    read_bytes_per_host: About how many bytes of HF tensors each host reads per Orbax call.
      Whole decoder layers are packed up to this; None reads the whole checkpoint in one call.
    read_chunk_bytes: Size of each ranged read Orbax sends to storage.

  Returns:
    `{"params": weights}`, the same structure `transform_hf_state_to_mt_state` returns.
    Weights no mapping covers are left as their abstract leaf, so the caller's
    weight-mismatch check reports them.
  """
  _check_orbax_version()
  t_start = time.time()
  plan = build_plan(param_map, hook_map, target_tree, config)
  context = ocp_v1.Context(
      checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS,
      safetensors_options=ocp_v1.options.SafetensorsOptions(read_chunk_bytes=read_chunk_bytes),
  )
  with context:
    hf_metadata = ocp_v1.metadata(path).metadata
    _check_sources(plan, hf_metadata)
    # Each host reads only the pieces its own TPU chips need, so a call of N bytes
    # reads about N / process_count on each host.
    max_bytes = None if read_bytes_per_host is None else read_bytes_per_host * jax.process_count()
    spans = pack_groups([_group_bytes(group, hf_metadata) for group in plan.groups], max_bytes)
    calls = [[write for i in span for write in plan.groups[i]] for span in spans]
    max_logging.log(
        f"Streaming {sum(len(g) for g in plan.groups)} HF sources into {len(plan.targets)} MaxText weights"
        f" in {len(calls)} read calls from {path}"
    )

    results = {t.name: _alloc_fn(t.shape, t.dtype, t.sharding)() for t in plan.targets if t.axes}
    total_bytes = 0
    for call_index, call in enumerate(calls):
      t_call = time.time()
      request = _load_request(call, hf_metadata, min_bytes_to_split)
      hf_arrays = ocp_v1.load(path, request)
      t_read = time.time()
      for write in call:
        _apply_write(write, hf_arrays, results)
      # Finish this call's writes before the next read, so at most one call's HF
      # tensors are ever held in HBM.
      jax.block_until_ready([results[write.target.name] for write in call])
      del hf_arrays
      call_bytes = sum(_nbytes(sds) for sds in request.values())
      total_bytes += call_bytes
      max_logging.log(
          f"[{call_index + 1}/{len(calls)}] {_call_name(call)}: {len(request)} HF tensors,"
          f" {call_bytes / 1e9:.3f} GB, read {t_read - t_call:.2f}s, convert {time.time() - t_read:.2f}s"
      )

  elapsed = time.time() - t_start
  peak = _peak_hbm_gb()
  max_logging.log(
      f"Streamed {total_bytes / 1e9:.2f} GB of HF weights in {elapsed:.1f}s"
      f" ({total_bytes / 1e9 / max(elapsed, 1e-9):.2f} GB/s)"
      + (f", peak HBM on {jax.local_devices()[0]}: {peak:.2f} GB" if peak is not None else "")
  )

  flat_restored = flax.traverse_util.flatten_dict(target_tree, sep=".")
  flat_restored.update(results)
  restored = flax.traverse_util.unflatten_dict(flat_restored, sep=".")
  # A Linen tree carries the `params` collection; return it wrapped exactly once.
  return {"params": restored.get("params", restored)}
