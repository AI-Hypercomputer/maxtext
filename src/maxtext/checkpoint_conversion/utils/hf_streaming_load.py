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
into read calls of about `READ_BYTES_PER_HOST` per host, starting with the
tensors outside the decoder layers (embeddings, final norm, lm_head). For each
read call:

1. `safetensors_reader` reads only that call's HF tensors from storage into HBM,
   each split along dim 0 across the TPU chips. SafeTensors is row-major, so
   every piece is one unbroken byte range in the file.
2. The HF->MaxText hooks from `param_mapping.py` (transpose, reshape, RoPE
   permutation, ...) run on the TPU chips, exactly as in the offline conversion.
3. A jitted writer casts each result to the MaxText dtype, moves it into the
   MaxText sharding, and writes it in place into the preallocated MaxText weight
   (stacked along the layer and/or expert axes where the mapping says so).

While one call's tensors are converted (steps 2-3), the next call's tensors are
read in a background thread. Peak HBM is therefore about the final MaxText
weights plus two calls' HF tensors, and peak host RAM about one call's HF
tensors, instead of the whole HF checkpoint.
"""

import collections
import concurrent.futures
import dataclasses
import functools
import math
import re
import time
from typing import Any, Iterator

import flax.traverse_util
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import safetensors_reader
from maxtext.checkpoint_conversion.utils import tensor_handling
from maxtext.utils import max_logging
import numpy as np

# HF tensors smaller than this get a full copy on every TPU chip: splitting them
# saves almost no memory.
MIN_BYTES_TO_SPLIT = 1 << 20

# Whole decoder layers are packed into read calls of about this many bytes per
# host. Bigger calls leave fewer idle read threads at the end of each call;
# smaller calls hold fewer HF tensors in HBM (two calls' worth with read-ahead).
READ_BYTES_PER_HOST = 4 << 30

READ_CHUNK_BYTES = safetensors_reader.READ_CHUNK_BYTES
READ_THREADS = safetensors_reader.READ_THREADS
MIN_PIECE_BYTES = safetensors_reader.MIN_PIECE_BYTES

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
  """Writes the hooked HF tensor(s) `hf_source` into `targets` at `index` (one entry per stacked axis).

  Usually there is one target. A tuple MaxText key in the param mapping (e.g. HF
  `gate_up_proj` -> MaxText `(wi_0, wi_1)`) gives several: the hooks then return
  them stacked along a new last axis, in key order, as in `to_maxtext`.
  """

  targets: tuple[_Target, ...]
  index: tuple[int, ...]
  hf_source: str | tuple[str, ...]  # One HF key, or a tuple the hook fuses into one tensor.

  @property
  def target(self) -> _Target:
    return self.targets[0]

  @property
  def hf_keys(self) -> tuple[str, ...]:
    return self.hf_source if isinstance(self.hf_source, tuple) else (self.hf_source,)


@dataclasses.dataclass
class LoadPlan:
  """What to load and where to write it, grouped by decoder layer."""

  targets: list[_Target]
  groups: list[list[_Write]]
  unmatched_mt_keys: list[str]  # Mapping entries with no weight in the target tree.
  # Weights whose mapping has no HF tensor (HF key None): the hook makes the whole value
  # from nothing (e.g. DeepSeek-V4's `mhc_norm` scales are all ones), as in `to_maxtext`.
  generated: list[_Target] = dataclasses.field(default_factory=list)


def resolve_target_name(mt_key: str, flat_target: dict) -> str | None:
  """Finds the key in the flattened target tree that a param-mapping key refers to."""
  mt_name = mt_key.replace("params-", "").replace("-", ".")
  candidates = [mt_name, f"params.{mt_name}", mt_key.replace("-", ".")]
  # An NNX tree holds weights of other Flax collections (e.g. `Tid2EidVar`) beside the rest,
  # without the collection name.
  candidates += [mt_key[len(c) :].replace("-", ".") for c in _LAYER_FIRST_COLLECTIONS if mt_key.startswith(c)]
  for candidate in candidates:
    if candidate in flat_target:
      return candidate
  return None


def _stacked_axes(mt_key: str, shape: tuple, depth: int, config) -> tuple[int, ...]:
  """Where each nesting level of the HF key list lands in the MaxText weight.

  Matches `tensor_handling`: a flat list stacks along `param_scan_axis` for
  scanned layers (or axis 0 for rank-1 weights and unscanned MoE experts), and
  deeper nesting follows `tensor_handling.stacked_axes`.

  Two flat-list cases stack along axis 0 even with scanned layers, as in
  `to_maxtext`: weights outside the `params` collection (`MoEBiasVar`,
  `Tid2EidVar`), which put the layer axis first, and the expert list of a single
  unscanned layer (`...-layers_<i>-...-MoeBlock...`, e.g. DeepSeek-V4's first
  layers), which has no layer axis at all.
  """
  if depth == 0:
    return ()
  if depth == 1:
    scan_axis = config.param_scan_axis
    if not config.scan_layers or len(shape) <= scan_axis or _stacks_on_axis_0(mt_key):
      return (0,)
    return (scan_axis,)
  return tuple(tensor_handling.stacked_axes(mt_key, config, depth))


# A single unscanned decoder layer, e.g. "params-decoder-layers_2-mlp-...".
_UNSCANNED_LAYER_KEY = re.compile(r"-layers_\d+-")
# Flax collections other than `params` whose stacked weights put the layer axis first.
_LAYER_FIRST_COLLECTIONS = ("MoEBiasVar-", "Tid2EidVar-")


def _stacks_on_axis_0(mt_key: str) -> bool:
  """Whether a flat HF key list for `mt_key` stacks along axis 0 with scanned layers (see `_stacked_axes`)."""
  if mt_key.startswith(_LAYER_FIRST_COLLECTIONS):
    return True
  return "MoeBlock" in mt_key and "scanned_blocks" not in mt_key and bool(_UNSCANNED_LAYER_KEY.search(mt_key))


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
  targets, writes, unmatched, generated = [], [], [], []
  for mt_key, hf_source in param_map.items():
    # A tuple MaxText key: the hooks give all its weights at once, stacked on a new last axis.
    mt_keys = mt_key if isinstance(mt_key, tuple) else (mt_key,)
    names = [resolve_target_name(key, flat_target) for key in mt_keys]
    if None in names:
      unmatched.extend(key for key, name in zip(mt_keys, names) if name is None)
      continue
    depth = tensor_handling.nesting_depth(hf_source)
    parts = []
    for key, name in zip(mt_keys, names):
      leaf = flat_target[name]
      shape = tuple(leaf.shape)
      axes = _stacked_axes(key, shape, depth, config)
      parts.append(
          _Target(
              name=name,
              mt_key=key,
              shape=shape,
              dtype=np.dtype(leaf.dtype),
              sharding=_target_sharding(leaf, name),
              axes=axes,
              hook_shape=tensor_handling.slice_shape(shape, axes),
              hooks=hook_map.get(mt_key),
          )
      )
    if len({(p.shape, p.axes) for p in parts}) > 1:
      raise ValueError(f"The MaxText weights of {mt_key} differ in shape: {[p.shape for p in parts]}.")
    if hf_source is None:
      if len(parts) > 1 or parts[0].hooks is None:
        raise ValueError(f"Param mapping gives no HF tensor for {mt_key}, so it needs exactly one weight and a hook.")
      generated.append(parts[0])
      continue
    targets.extend(parts)
    writes.extend(_Write(tuple(parts), index, source) for index, source in _enumerate_sources(hf_source, parts[0]))

  groups = collections.defaultdict(list)
  for write in writes:
    groups[_layer_group(write.hf_keys[0])].append(write)
  return LoadPlan(
      targets=targets, groups=[groups[k] for k in sorted(groups)], unmatched_mt_keys=unmatched, generated=generated
  )


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
  """The request that makes the reader read this call's HF tensors, split by rows."""
  request = {}
  for write in call:
    devices = _devices_of(write.target.sharding)
    for key in write.hf_keys:
      if key not in request:
        meta = hf_metadata[key]
        # Request the file's own dtype: the reader never casts.
        sharding = choose_load_sharding(meta.shape, meta.dtype, devices, min_bytes_to_split)
        request[key] = jax.ShapeDtypeStruct(meta.shape, meta.dtype, sharding=sharding)
  return request


def _apply_write(write: _Write, hf_arrays: dict, results: dict):
  """Hooks one HF source, then casts and writes it into its MaxText weight(s)."""
  first = write.target
  if isinstance(write.hf_source, tuple):
    raw = tuple(hf_arrays[key] for key in write.hf_source)
  else:
    raw = hf_arrays[write.hf_source]
  # A tuple MaxText key: the hooks stack its weights along a new last axis.
  hook_shape = first.hook_shape + ((len(write.targets),) if len(write.targets) > 1 else ())
  # The hooks run op by op, as in the offline conversion. Tracing them into one
  # jitted program would let XLA skip intermediate roundings (e.g. a bf16 cast
  # inside a hook), so the result could differ from `to_maxtext` in the last bits.
  try:
    x = tensor_handling.apply_hook_fns(raw, hook_shape, first.hooks)
  except RuntimeError as e:
    if jax.process_count() > 1 and "non-addressable" in str(e):
      raise RuntimeError(
          f"A hook for {first.mt_key} ({first.hooks}) copied an HF tensor to host numpy (e.g. np.concatenate)."
          " With several hosts each host holds only part of the tensor, so hooks must use jnp ops or array"
          " methods on jax.Arrays (see `param_mapping._array_module`)."
      ) from e
    raise
  if tuple(x.shape) != hook_shape:
    raise ValueError(
        f"Hooks for {first.mt_key} turned {write.hf_source} into shape {tuple(x.shape)}, but MaxText weight"
        f" {first.name} needs {hook_shape} per entry (full shape {first.shape})."
    )
  parts = [x[..., i] for i in range(len(write.targets))] if len(write.targets) > 1 else [x]
  for target, part in zip(write.targets, parts):
    if isinstance(part, jax.Array) and part.sharding.device_set != target.sharding.device_set:
      # Only when the target lives on a different set of devices than we loaded onto.
      part = jax.device_put(part, _replicated_sharding(_devices_of(target.sharding)))
    if target.axes:
      write_fn = _write_fn(target.dtype, target.sharding, target.axes, len(target.shape))
      results[target.name] = write_fn(results[target.name], part, *write.index)
    else:
      results[target.name] = _cast_fn(target.dtype, target.sharding)(part)


def _generate(target: _Target) -> jax.Array:
  """Makes a weight whose mapping has no HF tensor: the hook gets None and the full shape, as in `to_maxtext`."""
  x = tensor_handling.apply_hook_fns(None, target.shape, target.hooks)
  if tuple(x.shape) != target.shape:
    raise ValueError(
        f"Hooks for {target.mt_key} made shape {tuple(x.shape)} from no HF tensor, but {target.name} is {target.shape}."
    )
  return _cast_fn(target.dtype, target.sharding)(x)


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
    read_threads: int = READ_THREADS,
    min_piece_bytes: int = MIN_PIECE_BYTES,
    prefetch: bool = True,
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
    read_bytes_per_host: About how many bytes of HF tensors each host reads per read call.
      Whole decoder layers are packed up to this; None reads the whole checkpoint in one call.
    read_chunk_bytes: Size of each ranged read sent to storage.
    read_threads: How many ranged reads each host keeps in flight.
    min_piece_bytes: Groups of same-shape HF tensors whose per-chip pieces would be smaller
      than this are read whole and rearranged on the TPU chips (see `safetensors_reader`).
    prefetch: Read the next call while the current one is converted. Faster, but holds
      up to two calls of HF tensors in HBM instead of one.

  Returns:
    `target_tree` filled in, wrapped as `{"params": weights}` if it isn't already (an NNX pure
    dict); a Linen tree keeps any collections beside `params`.
    Weights no mapping covers are left as their abstract leaf, so the caller's
    weight-mismatch check reports them.
  """
  t_start = time.time()
  plan = build_plan(param_map, hook_map, target_tree, config)
  with (
      safetensors_reader.SafetensorsReader(
          path, num_threads=read_threads, chunk_bytes=read_chunk_bytes, min_piece_bytes=min_piece_bytes
      ) as reader,
      concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="hf_prefetch") as prefetcher,
  ):
    hf_metadata = reader.metadata
    _check_sources(plan, hf_metadata)
    # Each host reads only the pieces its own TPU chips need, so a call of N bytes
    # reads about N / process_count on each host.
    max_bytes = None if read_bytes_per_host is None else read_bytes_per_host * jax.process_count()
    spans = pack_groups([_group_bytes(group, hf_metadata) for group in plan.groups], max_bytes)
    calls = [[write for i in span for write in plan.groups[i]] for span in spans]
    requests = [_load_request(call, hf_metadata, min_bytes_to_split) for call in calls]
    max_logging.log(
        f"Streaming {sum(len(g) for g in plan.groups)} HF sources into {len(plan.targets)} MaxText weights"
        f" in {len(calls)} read calls from {path}"
        + (f"; {len(plan.generated)} more weights come from hooks alone" if plan.generated else "")
    )

    results = {t.name: _alloc_fn(t.shape, t.dtype, t.sharding)() for t in plan.targets if t.axes}
    for target in plan.generated:
      results[target.name] = _generate(target)
    total_bytes = 0
    # Only reads and their copies into HBM (`fetch`) run on the background thread. The
    # jitted rearranging (`unpack`) and writes stay on this thread in a fixed order, so
    # every host issues the same programs in the same order.
    future = prefetcher.submit(reader.fetch, requests[0]) if calls else None
    try:
      for call_index, call in enumerate(calls):
        t_wait = time.time()
        fetched = future.result()
        t_read = time.time()
        has_next = call_index + 1 < len(calls)
        if prefetch and has_next:
          future = prefetcher.submit(reader.fetch, requests[call_index + 1])
        hf_arrays = reader.unpack(fetched)
        del fetched
        for write in call:
          _apply_write(write, hf_arrays, results)
        # Finish this call's writes before dropping its HF tensors, so at most one
        # call (two with prefetch) of HF tensors is held in HBM.
        jax.block_until_ready([results[t.name] for write in call for t in write.targets])
        del hf_arrays
        t_done = time.time()
        if not prefetch and has_next:
          future = prefetcher.submit(reader.fetch, requests[call_index + 1])
        call_bytes = sum(_nbytes(sds) for sds in requests[call_index].values())
        total_bytes += call_bytes
        max_logging.log(
            f"[{call_index + 1}/{len(calls)}] {_call_name(call)}: {len(requests[call_index])} HF tensors,"
            f" {call_bytes / 1e9:.3f} GB, waited {t_read - t_wait:.2f}s for read, convert {t_done - t_read:.2f}s"
        )
    except BaseException:
      if future is not None:
        future.cancel()
      raise

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
  # A Linen tree carries the `params` collection, and maybe others beside it (e.g. DeepSeek-V4's
  # `MoEBiasVar` and `Tid2EidVar`): return it as is. An NNX pure dict has bare weights: wrap them once.
  return restored if "params" in restored else {"params": restored}
