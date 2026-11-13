# Copyright 2026 Google LLC
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

"""Streaming HuggingFace safetensors reader and writer for Flint (local or gs:// via `etils.epath`)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent import futures
import dataclasses
import json
import math
import os
import struct

from etils import epath
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import ml_dtypes
import numpy as np

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types

_INDEX_FILENAME = "model.safetensors.index.json"
_SINGLE_SHARD_FILENAME = "model.safetensors"
_METADATA_KEY = "__metadata__"
_MAX_HEADER_BYTES = 100 * 1024 * 1024
_LOAD_MAX_WORKERS = 16
_SLOT_CACHE_MAX_BYTES = 64 * 1024 * 1024
_HF_DTYPE_ITEMSIZE: dict[str, int] = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
}


@dataclasses.dataclass(frozen=True)
class _TensorEntry:
  """Byte offset, length, shape, and HF dtype for one safetensors tensor."""

  shard_path: str
  shape: tuple[int, ...]
  hf_dtype: str
  byte_offset: int
  byte_length: int


def _cast_np(arr: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
  """Casts `arr` to `target_dtype`, bridging `float16 -> bfloat16` via f32."""
  if arr.dtype == target_dtype:
    return arr
  if arr.dtype == np.dtype(np.float16) and target_dtype == np.dtype(ml_dtypes.bfloat16):
    return arr.astype(np.float32).astype(target_dtype)
  return arr.astype(target_dtype)


def _parse_shard_header(shard_path: str) -> dict[str, _TensorEntry]:
  """Parses and validates a `.safetensors` shard header without payload reads."""
  if not epath.Path(shard_path).exists():
    raise FileNotFoundError(f"Safetensors shard not found: {shard_path!r}.")
  file_size = epath.Path(shard_path).stat().length
  if file_size < 8:
    raise ValueError(f"Safetensors shard {shard_path!r} is truncated ({file_size} < 8" " bytes).")
  with epath.Path(shard_path).open("rb") as shard_file:
    len_bytes = shard_file.read(8)
    if len(len_bytes) != 8:
      raise ValueError(f"Failed to read 8-byte header size in {shard_path!r}.")
    (header_len,) = struct.unpack("<Q", len_bytes)
    if header_len == 0 or header_len > _MAX_HEADER_BYTES:
      raise ValueError(
          f"Invalid safetensors header length {header_len} in {shard_path!r}" f" (allowed 1..{_MAX_HEADER_BYTES})."
      )
    payload_base = 8 + header_len
    if payload_base > file_size:
      raise ValueError(f"Safetensors header overflow in {shard_path!r}: 8 + {header_len} >" f" file size {file_size}.")
    raw_header = shard_file.read(header_len)
    if len(raw_header) != header_len:
      raise ValueError(f"Truncated safetensors JSON header in {shard_path!r}.")

  try:
    header_map = json.loads(raw_header.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as err:
    raise ValueError(f"Malformed UTF-8 JSON header in {shard_path!r}: {err}") from err
  if not isinstance(header_map, dict):
    raise ValueError(f"Safetensors header in {shard_path!r} must be a JSON object.")

  entries: dict[str, _TensorEntry] = {}
  intervals: list[tuple[int, int, str]] = []
  for key, meta in header_map.items():
    if not isinstance(key, str):
      raise ValueError(f"Non-string tensor key {key!r} in {shard_path!r}.")
    if key == _METADATA_KEY:
      if not isinstance(meta, dict):
        raise ValueError(f"'{_METADATA_KEY}' in {shard_path!r} must be a JSON object.")
      continue
    if not isinstance(meta, dict):
      raise ValueError(f"Header entry for {key!r} in {shard_path!r} must be a JSON object.")
    hf_dtype = meta.get("dtype")
    if not isinstance(hf_dtype, str) or hf_dtype not in _HF_DTYPE_ITEMSIZE:
      raise ValueError(
          f"Unsupported safetensors dtype {hf_dtype!r} for {key!r} in"
          f" {shard_path!r}; expected one of {sorted(_HF_DTYPE_ITEMSIZE)}."
      )
    raw_shape = meta.get("shape")
    if not isinstance(raw_shape, list):
      raise ValueError(f"Invalid shape {raw_shape!r} for {key!r} in {shard_path!r}.")
    dims: list[int] = []
    for dim in raw_shape:
      if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
        raise ValueError(f"Invalid dimension {dim!r} in shape for {key!r} in {shard_path!r}.")
      dims.append(dim)
    shape = tuple(dims)

    offsets = meta.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or isinstance(offsets[0], bool)
        or isinstance(offsets[1], bool)
        or not isinstance(offsets[0], int)
        or not isinstance(offsets[1], int)
    ):
      raise ValueError(f"Invalid data_offsets {offsets!r} for {key!r} in {shard_path!r}.")
    start_off, end_off = offsets
    if start_off < 0 or end_off < start_off:
      raise ValueError(f"Negative or misordered data_offsets [{start_off}, {end_off}] for" f" {key!r} in {shard_path!r}.")
    expected_bytes = math.prod(shape) * _HF_DTYPE_ITEMSIZE[hf_dtype]
    byte_len = end_off - start_off
    if byte_len != expected_bytes:
      raise ValueError(
          f"Payload size {byte_len} disagrees with shape {shape} and dtype"
          f" {hf_dtype!r} ({expected_bytes} bytes) for {key!r} in"
          f" {shard_path!r}."
      )
    if payload_base + end_off > file_size:
      raise ValueError(
          f"Truncated byte payload for {key!r} in {shard_path!r}: end offset"
          f" {payload_base + end_off} exceeds file size {file_size}."
      )
    intervals.append((start_off, end_off, key))
    entries[key] = _TensorEntry(
        shard_path=shard_path,
        shape=shape,
        hf_dtype=hf_dtype,
        byte_offset=payload_base + start_off,
        byte_length=byte_len,
    )

  intervals.sort()
  prev_end = 0
  for start_off, end_off, key in intervals:
    if start_off < prev_end:
      raise ValueError(f"Overlapping data_offsets for {key!r} at {start_off} < {prev_end}" f" in {shard_path!r}.")
    prev_end = end_off
  return entries


class SafetensorsReader:
  """Streaming `etils.epath` reader for single or sharded `.safetensors`."""

  def __init__(self, path: str) -> None:
    """Indexes `.safetensors` headers from a file or checkpoint directory.

    Args:
      path: Directory containing `model.safetensors.index.json` or
        `model.safetensors`, or a direct `.safetensors` file path.

    Raises:
      FileNotFoundError: If `path` or any referenced shard file is missing.
      ValueError: If JSON headers, dtypes, offsets, or `weight_map` are invalid.
    """
    if not epath.Path(path).exists():
      raise FileNotFoundError(f"Safetensors path does not exist: {path!r}.")
    if epath.Path(path).is_dir():
      index_path = os.path.join(path, _INDEX_FILENAME)
      single_path = os.path.join(path, _SINGLE_SHARD_FILENAME)
      if epath.Path(index_path).exists():
        self._entries = self._load_indexed_dir(path, index_path)
      elif epath.Path(single_path).exists():
        self._entries = _parse_shard_header(single_path)
      else:
        raise FileNotFoundError(
            f"Directory {path!r} contains neither {_INDEX_FILENAME!r} nor" f" {_SINGLE_SHARD_FILENAME!r}."
        )
    else:
      self._entries = _parse_shard_header(path)
    self._keys = frozenset(self._entries)

  def _load_indexed_dir(self, dir_path: str, index_path: str) -> dict[str, _TensorEntry]:
    """Loads shard headers and verifies exact `weight_map` bijection."""
    with epath.Path(index_path).open("rb") as idx_file:
      raw_index = idx_file.read()
    try:
      index_doc = json.loads(raw_index.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
      raise ValueError(f"Malformed index JSON {index_path!r}: {err}") from err
    if not isinstance(index_doc, dict):
      raise ValueError(f"Index {index_path!r} must be a JSON object.")
    weight_map = index_doc.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
      raise ValueError(f"Index {index_path!r} must define a non-empty 'weight_map' dict.")

    shard_names_set: set[str] = set()
    for key, shard_name in weight_map.items():
      if not isinstance(key, str) or not isinstance(shard_name, str):
        raise ValueError(f"Invalid weight_map entry {key!r}: {shard_name!r} in" f" {index_path!r}.")
      norm_name = os.path.normpath(shard_name)
      if not shard_name or os.path.isabs(shard_name) or norm_name.startswith(".."):
        raise ValueError(f"Unsafe shard filename {shard_name!r} for {key!r} in" f" {index_path!r}.")
      shard_names_set.add(shard_name)

    combined: dict[str, _TensorEntry] = {}
    actual_shard_for_key: dict[str, str] = {}
    for shard_name in sorted(shard_names_set):
      shard_path = os.path.join(dir_path, shard_name)
      shard_entries = _parse_shard_header(shard_path)
      for key, entry in shard_entries.items():
        if key in combined:
          raise ValueError(
              f"Duplicate tensor key {key!r} in shards" f" {actual_shard_for_key[key]!r} and {shard_name!r}."
          )
        combined[key] = entry
        actual_shard_for_key[key] = shard_name

    for key, expected_shard in weight_map.items():
      actual_shard = actual_shard_for_key.get(key)
      if actual_shard != expected_shard:
        raise ValueError(
            f"weight_map disagrees with shard headers for {key!r}: mapped to"
            f" {expected_shard!r}, found in {actual_shard!r}."
        )
    if len(combined) != len(weight_map):
      unlisted = sorted(set(combined) - set(weight_map))[:10]
      raise ValueError(f"Shard headers contain keys missing from weight_map: {unlisted}.")
    return combined

  def keys(self) -> frozenset[str]:
    """Returns the set of tensor keys across all shards."""
    return self._keys

  def tensor_meta(self, key: str) -> tuple[tuple[int, ...], str]:
    """Returns `(shape, hf_dtype)` from cached JSON headers without I/O."""
    entry = self._entries.get(key)
    if entry is None:
      raise KeyError(f"Tensor key {key!r} not found in safetensors checkpoint.")
    return entry.shape, entry.hf_dtype

  def read_tensor(
      self,
      key: str,
      *,
      dtype: jax.typing.DTypeLike | None = None,
  ) -> np.ndarray:
    """Streams `key` from its shard via `epath.Path.open` seek + read.

    Args:
      key: HuggingFace tensor name.
      dtype: Optional output NumPy/JAX dtype; when `None`, preserves the exact
        checkpoint dtype (`ml_dtypes.bfloat16`, `np.float16`, or `np.float32`).

    Returns:
      Writable `np.ndarray` with the tensor's exact shape and requested dtype.

    Raises:
      KeyError: If `key` is not present in the checkpoint.
      ValueError: If the shard payload is truncated or has an unsupported dtype.
    """
    entry = self._entries.get(key)
    if entry is None:
      raise KeyError(f"Tensor key {key!r} not found in safetensors checkpoint.")
    with epath.Path(entry.shard_path).open("rb") as shard_file:
      shard_file.seek(entry.byte_offset)
      raw_bytes = shard_file.read(entry.byte_length)
    if len(raw_bytes) != entry.byte_length:
      raise ValueError(
          f"Truncated read for {key!r} in {entry.shard_path!r}: expected"
          f" {entry.byte_length} bytes, got {len(raw_bytes)}."
      )

    if entry.hf_dtype == "BF16":
      arr = np.frombuffer(raw_bytes, dtype=np.dtype("<u2")).view(ml_dtypes.bfloat16).reshape(entry.shape)
    elif entry.hf_dtype == "F16":
      arr = np.frombuffer(raw_bytes, dtype=np.dtype("<f2")).reshape(entry.shape)
    elif entry.hf_dtype == "F32":
      arr = np.frombuffer(raw_bytes, dtype=np.dtype("<f4")).reshape(entry.shape)
    else:
      raise ValueError(f"Unsupported safetensors dtype: {entry.hf_dtype!r}.")

    if dtype is not None:
      target_dt = np.dtype(dtype)
      if arr.dtype != target_dt:
        return _cast_np(arr, target_dt)
    return arr.copy()


def expected_hf_shapes(
    model: flint_config.FlintModelConfig,
) -> dict[str, tuple[int, ...]]:
  """Returns exact HuggingFace `RecursiveSI/flint` keys and `[out, in]` shapes."""
  v = model.vocab_size
  d = model.hidden_size
  q_dim = model.num_attention_heads * model.head_dim
  kv_dim = model.num_key_value_heads * model.head_dim
  head_dim = model.head_dim
  h_q = model.num_attention_heads
  inter = model.intermediate_size
  e_cnt = model.num_experts
  shared_inter = model.shared_expert_intermediate_size
  moe_inter = model.moe_intermediate_size

  shapes: dict[str, tuple[int, ...]] = {
      "model.embed_tokens.weight": (v, d),
  }
  for layer_idx in range(model.num_hidden_layers):
    pfx = f"model.layers.{layer_idx}"
    shapes[f"{pfx}.input_layernorm.weight"] = (d,)
    shapes[f"{pfx}.self_attn.q_proj.weight"] = (q_dim, d)
    shapes[f"{pfx}.self_attn.k_proj.weight"] = (kv_dim, d)
    shapes[f"{pfx}.self_attn.v_proj.weight"] = (kv_dim, d)
    shapes[f"{pfx}.self_attn.o_proj.weight"] = (d, q_dim)
    shapes[f"{pfx}.self_attn.q_norm.weight"] = (head_dim,)
    shapes[f"{pfx}.self_attn.k_norm.weight"] = (head_dim,)
    shapes[f"{pfx}.self_attn.softmax_offset"] = (h_q,)
    shapes[f"{pfx}.post_attention_layernorm.weight"] = (d,)
    if flint_config.is_moe_layer(layer_idx, model):
      shapes[f"{pfx}.mlp.gate.weight"] = (e_cnt, d)
      shapes[f"{pfx}.mlp.shared_expert.gate_proj.weight"] = (shared_inter, d)
      shapes[f"{pfx}.mlp.shared_expert.up_proj.weight"] = (shared_inter, d)
      shapes[f"{pfx}.mlp.shared_expert.down_proj.weight"] = (d, shared_inter)
      for exp_idx in range(e_cnt):
        epfx = f"{pfx}.mlp.experts.{exp_idx}"
        shapes[f"{epfx}.gate_proj.weight"] = (moe_inter, d)
        shapes[f"{epfx}.up_proj.weight"] = (moe_inter, d)
        shapes[f"{epfx}.down_proj.weight"] = (d, moe_inter)
    else:
      shapes[f"{pfx}.mlp.gate_proj.weight"] = (inter, d)
      shapes[f"{pfx}.mlp.up_proj.weight"] = (inter, d)
      shapes[f"{pfx}.mlp.down_proj.weight"] = (d, inter)

  shapes["model.norm.weight"] = (d,)
  shapes["lm_head.weight"] = (v, d)
  return shapes


def validate_hf_checkpoint(
    reader: SafetensorsReader,
    model: flint_config.FlintModelConfig,
) -> None:
  """Validates exact HF key bijection and header shapes before payload reads.

  Args:
    reader: Indexed `SafetensorsReader`.
    model: Target `FlintModelConfig`.

  Raises:
    ValueError: If any expected keys are missing, extra keys exist, or shapes
      differ from `expected_hf_shapes(model)`.
  """
  expected = expected_hf_shapes(model)
  expected_keys = frozenset(expected)
  actual_keys = reader.keys()

  missing = sorted(expected_keys - actual_keys)
  unexpected = sorted(actual_keys - expected_keys)
  mismatched: list[str] = []
  for key in sorted(expected_keys & actual_keys):
    actual_shape, _ = reader.tensor_meta(key)
    if actual_shape != expected[key]:
      mismatched.append(f"{key} (expected {expected[key]}, got {actual_shape})")

  problems: list[str] = []
  if missing:
    problems.append(f"missing {len(missing)} keys: {missing[:10]}")
  if unexpected:
    problems.append(f"unexpected {len(unexpected)} keys: {unexpected[:10]}")
  if mismatched:
    problems.append(f"shape mismatch on {len(mismatched)} keys: {mismatched[:10]}")
  if problems:
    raise ValueError("HuggingFace safetensors validation failed: " + "; ".join(problems) + ".")


def _addressable_dim_bounds(
    global_shape: tuple[int, ...],
    sharding: NamedSharding,
) -> tuple[tuple[int, int], ...] | None:
  """Computes per-dimension `[lo, hi)` union across addressable devices."""
  idx_map = sharding.addressable_devices_indices_map(global_shape)
  if not idx_map:
    return None
  bounds: list[tuple[int, int]] = []
  for dim_idx, dim_size in enumerate(global_shape):
    lo = dim_size
    hi = 0
    for dev_idx in idx_map.values():
      if dev_idx is None:
        raise RuntimeError("Unexpected None slice in addressable_devices map.")
      start, stop, step = dev_idx[dim_idx].indices(dim_size)
      if step != 1:
        raise ValueError(f"Expected unit slice step, got {dev_idx[dim_idx]!r}.")
      lo = min(lo, start)
      hi = max(hi, stop)
    bounds.append((lo, hi))
  return tuple(bounds)


def _addressable_needed_indices(
    global_shape: tuple[int, ...],
    dim_idx: int,
    sharding: NamedSharding,
) -> list[int]:
  """Returns sorted unique indices along `dim_idx` held on this host."""
  idx_map = sharding.addressable_devices_indices_map(global_shape)
  dim_size = global_shape[dim_idx]
  needed: set[int] = set()
  for dev_idx in idx_map.values():
    if dev_idx is None:
      raise RuntimeError("Unexpected None slice in addressable_devices map.")
    start, stop, step = dev_idx[dim_idx].indices(dim_size)
    if step != 1:
      raise ValueError(f"Expected unit slice step, got {dev_idx[dim_idx]!r}.")
    needed.update(range(start, stop))
  return sorted(needed)


def _make_leaf_from_host_bbox(
    global_shape: tuple[int, ...],
    sharding: NamedSharding,
    bounds: tuple[tuple[int, int], ...] | None,
    host_buf: np.ndarray | None,
    target_dtype: jnp.dtype,
) -> jax.Array:
  """Places a host-sliced leaf onto `sharding` via `make_array_from_callback`."""
  target_np_dtype = np.dtype(target_dtype)

  def _callback(index: tuple[slice, ...] | None) -> np.ndarray:
    if index is None or bounds is None or host_buf is None:
      raise RuntimeError("Callback invoked without addressable host slice.")
    rel_slices: list[slice] = []
    for dim_idx, (dim_slice, (lo, _)) in enumerate(zip(index, bounds, strict=True)):
      start, stop, step = dim_slice.indices(global_shape[dim_idx])
      if step != 1:
        raise ValueError(f"Expected unit slice step, got {dim_slice!r}.")
      rel_slices.append(slice(start - lo, stop - lo))
    return _cast_np(host_buf[tuple(rel_slices)], target_np_dtype)

  return jax.make_array_from_callback(
      global_shape,
      sharding,
      _callback,
      dtype=target_dtype,
  )


def _load_single_leaf(
    reader: SafetensorsReader,
    key: str,
    *,
    transpose: bool,
    target_dtype: jnp.dtype,
    sharding: NamedSharding,
) -> jax.Array:
  """Reads one HF tensor once on the host and shards it onto `sharding`."""
  hf_shape, _ = reader.tensor_meta(key)
  global_shape = (hf_shape[1], hf_shape[0]) if transpose else hf_shape
  bounds = _addressable_dim_bounds(global_shape, sharding)
  if bounds is None:
    return _make_leaf_from_host_bbox(global_shape, sharding, None, None, target_dtype)
  arr = reader.read_tensor(key)
  if transpose:
    arr = arr.T
  bbox_slices = tuple(slice(lo, hi) for lo, hi in bounds)
  host_buf = _cast_np(arr[bbox_slices], np.dtype(target_dtype))
  return _make_leaf_from_host_bbox(global_shape, sharding, bounds, host_buf, target_dtype)


def _load_stacked_leaf(
    reader: SafetensorsReader,
    keys: Sequence[str],
    *,
    transpose: bool,
    target_dtype: jnp.dtype,
    sharding: NamedSharding,
    executor: futures.ThreadPoolExecutor,
) -> jax.Array:
  """Reads `keys` concurrently, stacks along axis 0, and shards onto `sharding`."""
  hf_shape, _ = reader.tensor_meta(keys[0])
  inner_shape = (hf_shape[1], hf_shape[0]) if transpose else hf_shape
  global_shape = (len(keys), *inner_shape)
  bounds = _addressable_dim_bounds(global_shape, sharding)
  if bounds is None:
    return _make_leaf_from_host_bbox(global_shape, sharding, None, None, target_dtype)
  sb_lo, sb_hi = bounds[0]
  inner_slices = tuple(slice(lo, hi) for lo, hi in bounds[1:])
  target_np_dtype = np.dtype(target_dtype)

  def _read_one(key: str) -> np.ndarray:
    arr = reader.read_tensor(key)
    if transpose:
      arr = arr.T
    return _cast_np(arr[inner_slices], target_np_dtype)

  needed_keys = keys[sb_lo:sb_hi]
  if len(needed_keys) > 1:
    layers = list(executor.map(_read_one, needed_keys))
  else:
    layers = [_read_one(k) for k in needed_keys]
  host_buf = np.stack(layers, axis=0)
  return _make_leaf_from_host_bbox(global_shape, sharding, bounds, host_buf, target_dtype)


def _choose_expert_buf_dtype(hf_dtypes: set[str], target_dtype: jnp.dtype) -> np.dtype:
  """Uses 2-byte `bfloat16` buffer when widening BF16 -> FP32 to halve RAM."""
  if hf_dtypes == {"BF16"} and target_dtype == jnp.dtype(jnp.float32):
    return np.dtype(ml_dtypes.bfloat16)
  return np.dtype(target_dtype)


def _load_routed_gate_up(
    reader: SafetensorsReader,
    layer_indices: Sequence[int],
    *,
    stacked: bool,
    model: flint_config.FlintModelConfig,
    target_dtype: jnp.dtype,
    sharding: NamedSharding,
    executor: futures.ThreadPoolExecutor,
) -> jax.Array:
  """Reads only host-local experts concurrently and packs `gate_up_proj`."""
  e_cnt = model.num_experts
  d = model.hidden_size
  m2 = 2 * model.moe_intermediate_size
  global_shape = (len(layer_indices), e_cnt, d, m2) if stacked else (e_cnt, d, m2)
  bounds = _addressable_dim_bounds(global_shape, sharding)
  if bounds is None:
    return _make_leaf_from_host_bbox(global_shape, sharding, None, None, target_dtype)

  exp_axis = 1 if stacked else 0
  needed_exps = _addressable_needed_indices(global_shape, exp_axis, sharding)
  exp_to_slot = {exp_id: slot for slot, exp_id in enumerate(needed_exps)}
  is_contiguous = bool(needed_exps) and needed_exps[-1] - needed_exps[0] + 1 == len(needed_exps)

  if stacked:
    sb_lo, sb_hi = bounds[0]
    d_lo, d_hi = bounds[2]
    m2_lo, m2_hi = bounds[3]
  else:
    sb_lo, sb_hi = 0, 1
    d_lo, d_hi = bounds[1]
    m2_lo, m2_hi = bounds[2]

  tasks: list[tuple[int, int, int, int]] = []
  hf_dtypes: set[str] = set()
  for sb_rel, layer_idx in enumerate(layer_indices[sb_lo:sb_hi]):
    for slot, exp_idx in enumerate(needed_exps):
      epfx = f"model.layers.{layer_idx}.mlp.experts.{exp_idx}"
      hf_dtypes.add(reader.tensor_meta(f"{epfx}.gate_proj.weight")[1])
      hf_dtypes.add(reader.tensor_meta(f"{epfx}.up_proj.weight")[1])
      tasks.append((sb_rel, slot, layer_idx, exp_idx))

  buf_dtype = _choose_expert_buf_dtype(hf_dtypes, target_dtype)
  buf_shape = (
      (sb_hi - sb_lo, len(needed_exps), d_hi - d_lo, m2_hi - m2_lo)
      if stacked
      else (len(needed_exps), d_hi - d_lo, m2_hi - m2_lo)
  )
  host_buf = np.empty(buf_shape, dtype=buf_dtype)

  def _read_pair(
      task: tuple[int, int, int, int],
  ) -> tuple[int, int, np.ndarray]:
    sb_rel, slot, layer_idx, exp_idx = task
    epfx = f"model.layers.{layer_idx}.mlp.experts.{exp_idx}"
    gate_t = reader.read_tensor(f"{epfx}.gate_proj.weight").T[d_lo:d_hi, :]
    up_t = reader.read_tensor(f"{epfx}.up_proj.weight").T[d_lo:d_hi, :]
    packed = np.concatenate([gate_t, up_t], axis=-1)[:, m2_lo:m2_hi]
    return sb_rel, slot, _cast_np(packed, buf_dtype)

  for sb_rel, slot, packed_arr in executor.map(_read_pair, tasks):
    if stacked:
      host_buf[sb_rel, slot] = packed_arr
    else:
      host_buf[slot] = packed_arr

  target_np_dtype = np.dtype(target_dtype)

  def _callback(index: tuple[slice, ...] | None) -> np.ndarray:
    if index is None:
      raise RuntimeError("Callback invoked with None index.")
    e_start, e_stop, _ = index[exp_axis].indices(e_cnt)
    exp_sel: slice | list[int]
    if is_contiguous:
      exp_sel = slice(e_start - needed_exps[0], e_stop - needed_exps[0])
    else:
      exp_sel = [exp_to_slot[e_id] for e_id in range(e_start, e_stop)]
    if stacked:
      s0, s1, _ = index[0].indices(global_shape[0])
      d0, d1, _ = index[2].indices(d)
      m0, m1, _ = index[3].indices(m2)
      sub = host_buf[
          s0 - sb_lo : s1 - sb_lo,
          exp_sel,
          d0 - d_lo : d1 - d_lo,
          m0 - m2_lo : m1 - m2_lo,
      ]
    else:
      d0, d1, _ = index[1].indices(d)
      m0, m1, _ = index[2].indices(m2)
      sub = host_buf[exp_sel, d0 - d_lo : d1 - d_lo, m0 - m2_lo : m1 - m2_lo]
    return _cast_np(sub, target_np_dtype)

  return jax.make_array_from_callback(global_shape, sharding, _callback, dtype=target_dtype)


def _load_routed_down(
    reader: SafetensorsReader,
    layer_indices: Sequence[int],
    *,
    stacked: bool,
    model: flint_config.FlintModelConfig,
    target_dtype: jnp.dtype,
    sharding: NamedSharding,
    executor: futures.ThreadPoolExecutor,
) -> jax.Array:
  """Reads only host-local experts concurrently for `down_proj`."""
  e_cnt = model.num_experts
  m_dim = model.moe_intermediate_size
  d = model.hidden_size
  global_shape = (len(layer_indices), e_cnt, m_dim, d) if stacked else (e_cnt, m_dim, d)
  bounds = _addressable_dim_bounds(global_shape, sharding)
  if bounds is None:
    return _make_leaf_from_host_bbox(global_shape, sharding, None, None, target_dtype)

  exp_axis = 1 if stacked else 0
  needed_exps = _addressable_needed_indices(global_shape, exp_axis, sharding)
  exp_to_slot = {exp_id: slot for slot, exp_id in enumerate(needed_exps)}
  is_contiguous = bool(needed_exps) and needed_exps[-1] - needed_exps[0] + 1 == len(needed_exps)

  if stacked:
    sb_lo, sb_hi = bounds[0]
    m_lo, m_hi = bounds[2]
    d_lo, d_hi = bounds[3]
  else:
    sb_lo, sb_hi = 0, 1
    m_lo, m_hi = bounds[1]
    d_lo, d_hi = bounds[2]

  tasks: list[tuple[int, int, int, int]] = []
  hf_dtypes: set[str] = set()
  for sb_rel, layer_idx in enumerate(layer_indices[sb_lo:sb_hi]):
    for slot, exp_idx in enumerate(needed_exps):
      epfx = f"model.layers.{layer_idx}.mlp.experts.{exp_idx}"
      hf_dtypes.add(reader.tensor_meta(f"{epfx}.down_proj.weight")[1])
      tasks.append((sb_rel, slot, layer_idx, exp_idx))

  buf_dtype = _choose_expert_buf_dtype(hf_dtypes, target_dtype)
  buf_shape = (
      (sb_hi - sb_lo, len(needed_exps), m_hi - m_lo, d_hi - d_lo)
      if stacked
      else (len(needed_exps), m_hi - m_lo, d_hi - d_lo)
  )
  host_buf = np.empty(buf_shape, dtype=buf_dtype)

  def _read_down(
      task: tuple[int, int, int, int],
  ) -> tuple[int, int, np.ndarray]:
    sb_rel, slot, layer_idx, exp_idx = task
    epfx = f"model.layers.{layer_idx}.mlp.experts.{exp_idx}"
    down_t = reader.read_tensor(f"{epfx}.down_proj.weight").T[m_lo:m_hi, d_lo:d_hi]
    return sb_rel, slot, _cast_np(down_t, buf_dtype)

  for sb_rel, slot, down_arr in executor.map(_read_down, tasks):
    if stacked:
      host_buf[sb_rel, slot] = down_arr
    else:
      host_buf[slot] = down_arr

  target_np_dtype = np.dtype(target_dtype)

  def _callback(index: tuple[slice, ...] | None) -> np.ndarray:
    if index is None:
      raise RuntimeError("Callback invoked with None index.")
    e_start, e_stop, _ = index[exp_axis].indices(e_cnt)
    exp_sel: slice | list[int]
    if is_contiguous:
      exp_sel = slice(e_start - needed_exps[0], e_stop - needed_exps[0])
    else:
      exp_sel = [exp_to_slot[e_id] for e_id in range(e_start, e_stop)]
    if stacked:
      s0, s1, _ = index[0].indices(global_shape[0])
      m0, m1, _ = index[2].indices(m_dim)
      d0, d1, _ = index[3].indices(d)
      sub = host_buf[
          s0 - sb_lo : s1 - sb_lo,
          exp_sel,
          m0 - m_lo : m1 - m_lo,
          d0 - d_lo : d1 - d_lo,
      ]
    else:
      m0, m1, _ = index[1].indices(m_dim)
      d0, d1, _ = index[2].indices(d)
      sub = host_buf[exp_sel, m0 - m_lo : m1 - m_lo, d0 - d_lo : d1 - d_lo]
    return _cast_np(sub, target_np_dtype)

  return jax.make_array_from_callback(global_shape, sharding, _callback, dtype=target_dtype)


def _load_attn_unstacked(
    reader: SafetensorsReader,
    layer_idx: int,
    param_dt: jnp.dtype,
    shardings: flint_types.FlintAttnWeights[NamedSharding],
) -> flint_types.FlintAttnWeights[jax.Array]:
  """Loads unstacked `FlintAttnWeights` (`sinks` always `float32`)."""
  pfx = f"model.layers.{layer_idx}.self_attn"
  f32_dt = jnp.dtype(jnp.float32)
  return flint_types.FlintAttnWeights(
      q_proj=_load_single_leaf(
          reader,
          f"{pfx}.q_proj.weight",
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.q_proj,
      ),
      k_proj=_load_single_leaf(
          reader,
          f"{pfx}.k_proj.weight",
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.k_proj,
      ),
      v_proj=_load_single_leaf(
          reader,
          f"{pfx}.v_proj.weight",
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.v_proj,
      ),
      o_proj=_load_single_leaf(
          reader,
          f"{pfx}.o_proj.weight",
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.o_proj,
      ),
      q_norm=_load_single_leaf(
          reader,
          f"{pfx}.q_norm.weight",
          transpose=False,
          target_dtype=param_dt,
          sharding=shardings.q_norm,
      ),
      k_norm=_load_single_leaf(
          reader,
          f"{pfx}.k_norm.weight",
          transpose=False,
          target_dtype=param_dt,
          sharding=shardings.k_norm,
      ),
      sinks=_load_single_leaf(
          reader,
          f"{pfx}.softmax_offset",
          transpose=False,
          target_dtype=f32_dt,
          sharding=shardings.sinks,
      ),
  )


def _load_mlp_unstacked(
    reader: SafetensorsReader,
    prefix: str,
    param_dt: jnp.dtype,
    shardings: flint_types.FlintMLPWeights[NamedSharding],
) -> flint_types.FlintMLPWeights[jax.Array]:
  """Loads unstacked SwiGLU `FlintMLPWeights`."""
  return flint_types.FlintMLPWeights(
      gate_proj=_load_single_leaf(
          reader,
          f"{prefix}.gate_proj.weight",
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.gate_proj,
      ),
      up_proj=_load_single_leaf(
          reader,
          f"{prefix}.up_proj.weight",
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.up_proj,
      ),
      down_proj=_load_single_leaf(
          reader,
          f"{prefix}.down_proj.weight",
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.down_proj,
      ),
  )


def _load_sparse_unstacked(
    reader: SafetensorsReader,
    layer_idx: int,
    model: flint_config.FlintModelConfig,
    param_dt: jnp.dtype,
    shardings: flint_types.FlintSparseLayerWeights[NamedSharding],
    executor: futures.ThreadPoolExecutor,
) -> flint_types.FlintSparseLayerWeights[jax.Array]:
  """Loads one unstacked `FlintSparseLayerWeights`."""
  pfx = f"model.layers.{layer_idx}"
  return flint_types.FlintSparseLayerWeights(
      input_layernorm=_load_single_leaf(
          reader,
          f"{pfx}.input_layernorm.weight",
          transpose=False,
          target_dtype=param_dt,
          sharding=shardings.input_layernorm,
      ),
      self_attn=_load_attn_unstacked(reader, layer_idx, param_dt, shardings.self_attn),
      post_attention_layernorm=_load_single_leaf(
          reader,
          f"{pfx}.post_attention_layernorm.weight",
          transpose=False,
          target_dtype=param_dt,
          sharding=shardings.post_attention_layernorm,
      ),
      moe=flint_types.FlintMoEWeights(
          router=flint_types.FlintRouterWeights(
              gate=_load_single_leaf(
                  reader,
                  f"{pfx}.mlp.gate.weight",
                  transpose=True,
                  target_dtype=param_dt,
                  sharding=shardings.moe.router.gate,
              ),
          ),
          shared_expert=_load_mlp_unstacked(
              reader,
              f"{pfx}.mlp.shared_expert",
              param_dt,
              shardings.moe.shared_expert,
          ),
          routed_experts=flint_types.FlintRoutedExpertsWeights(
              gate_up_proj=_load_routed_gate_up(
                  reader,
                  (layer_idx,),
                  stacked=False,
                  model=model,
                  target_dtype=param_dt,
                  sharding=shardings.moe.routed_experts.gate_up_proj,
                  executor=executor,
              ),
              down_proj=_load_routed_down(
                  reader,
                  (layer_idx,),
                  stacked=False,
                  model=model,
                  target_dtype=param_dt,
                  sharding=shardings.moe.routed_experts.down_proj,
                  executor=executor,
              ),
          ),
      ),
  )


def _load_sparse_stacked(
    reader: SafetensorsReader,
    layer_indices: Sequence[int],
    model: flint_config.FlintModelConfig,
    param_dt: jnp.dtype,
    shardings: flint_types.FlintSparseLayerWeights[NamedSharding],
    executor: futures.ThreadPoolExecutor,
) -> flint_types.FlintSparseLayerWeights[jax.Array]:
  """Loads stacked `[N_sb, ...]` `FlintSparseLayerWeights` for one scan slot."""
  f32_dt = jnp.dtype(jnp.float32)

  def _keys(suffix: str) -> list[str]:
    return [f"model.layers.{idx}.{suffix}" for idx in layer_indices]

  attn = flint_types.FlintAttnWeights(
      q_proj=_load_stacked_leaf(
          reader,
          _keys("self_attn.q_proj.weight"),
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.self_attn.q_proj,
          executor=executor,
      ),
      k_proj=_load_stacked_leaf(
          reader,
          _keys("self_attn.k_proj.weight"),
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.self_attn.k_proj,
          executor=executor,
      ),
      v_proj=_load_stacked_leaf(
          reader,
          _keys("self_attn.v_proj.weight"),
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.self_attn.v_proj,
          executor=executor,
      ),
      o_proj=_load_stacked_leaf(
          reader,
          _keys("self_attn.o_proj.weight"),
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.self_attn.o_proj,
          executor=executor,
      ),
      q_norm=_load_stacked_leaf(
          reader,
          _keys("self_attn.q_norm.weight"),
          transpose=False,
          target_dtype=param_dt,
          sharding=shardings.self_attn.q_norm,
          executor=executor,
      ),
      k_norm=_load_stacked_leaf(
          reader,
          _keys("self_attn.k_norm.weight"),
          transpose=False,
          target_dtype=param_dt,
          sharding=shardings.self_attn.k_norm,
          executor=executor,
      ),
      sinks=_load_stacked_leaf(
          reader,
          _keys("self_attn.softmax_offset"),
          transpose=False,
          target_dtype=f32_dt,
          sharding=shardings.self_attn.sinks,
          executor=executor,
      ),
  )
  shared = flint_types.FlintMLPWeights(
      gate_proj=_load_stacked_leaf(
          reader,
          _keys("mlp.shared_expert.gate_proj.weight"),
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.moe.shared_expert.gate_proj,
          executor=executor,
      ),
      up_proj=_load_stacked_leaf(
          reader,
          _keys("mlp.shared_expert.up_proj.weight"),
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.moe.shared_expert.up_proj,
          executor=executor,
      ),
      down_proj=_load_stacked_leaf(
          reader,
          _keys("mlp.shared_expert.down_proj.weight"),
          transpose=True,
          target_dtype=param_dt,
          sharding=shardings.moe.shared_expert.down_proj,
          executor=executor,
      ),
  )
  return flint_types.FlintSparseLayerWeights(
      input_layernorm=_load_stacked_leaf(
          reader,
          _keys("input_layernorm.weight"),
          transpose=False,
          target_dtype=param_dt,
          sharding=shardings.input_layernorm,
          executor=executor,
      ),
      self_attn=attn,
      post_attention_layernorm=_load_stacked_leaf(
          reader,
          _keys("post_attention_layernorm.weight"),
          transpose=False,
          target_dtype=param_dt,
          sharding=shardings.post_attention_layernorm,
          executor=executor,
      ),
      moe=flint_types.FlintMoEWeights(
          router=flint_types.FlintRouterWeights(
              gate=_load_stacked_leaf(
                  reader,
                  _keys("mlp.gate.weight"),
                  transpose=True,
                  target_dtype=param_dt,
                  sharding=shardings.moe.router.gate,
                  executor=executor,
              ),
          ),
          shared_expert=shared,
          routed_experts=flint_types.FlintRoutedExpertsWeights(
              gate_up_proj=_load_routed_gate_up(
                  reader,
                  layer_indices,
                  stacked=True,
                  model=model,
                  target_dtype=param_dt,
                  sharding=shardings.moe.routed_experts.gate_up_proj,
                  executor=executor,
              ),
              down_proj=_load_routed_down(
                  reader,
                  layer_indices,
                  stacked=True,
                  model=model,
                  target_dtype=param_dt,
                  sharding=shardings.moe.routed_experts.down_proj,
                  executor=executor,
              ),
          ),
      ),
  )


def load_hf_checkpoint(
    ckpt_dir: str,
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
) -> flint_types.FlintParams[jax.Array]:
  """Streams a HuggingFace safetensors checkpoint directly onto `mesh`.

  Validates key bijection and shapes upfront, reads each host-needed tensor at
  most once with bounded host RAM, transposes 2D linears to `[in, out]`, packs
  routed `gate_up_proj`, stacks superblocks when `cfg.model.scan_layers=True`,
  and keeps attention `sinks` in `jnp.float32`.

  Args:
    ckpt_dir: Checkpoint directory or single `.safetensors` file path.
    cfg: Validated `FlintConfig`.
    mesh: Target 3D `('fsdp', 'cp', 'ep')` JAX `Mesh`.

  Returns:
    Sharded `FlintParams[jax.Array]` on `mesh`.
  """
  reader = SafetensorsReader(ckpt_dir)
  model = cfg.model
  validate_hf_checkpoint(reader, model)

  shardings = flint_sharding.named_param_shardings(mesh, cfg)
  param_dt = cfg.train.param_jnp_dtype

  with futures.ThreadPoolExecutor(max_workers=_LOAD_MAX_WORKERS) as executor:
    embed = _load_single_leaf(
        reader,
        "model.embed_tokens.weight",
        transpose=False,
        target_dtype=param_dt,
        sharding=shardings.embed,
    )

    prefix_dense: list[flint_types.FlintDenseLayerWeights[jax.Array]] = []
    for layer_idx in range(model.num_dense_prefix_layers):
      pfx = f"model.layers.{layer_idx}"
      sh_i = shardings.decoder.prefix_dense[layer_idx]
      prefix_dense.append(
          flint_types.FlintDenseLayerWeights(
              input_layernorm=_load_single_leaf(
                  reader,
                  f"{pfx}.input_layernorm.weight",
                  transpose=False,
                  target_dtype=param_dt,
                  sharding=sh_i.input_layernorm,
              ),
              self_attn=_load_attn_unstacked(reader, layer_idx, param_dt, sh_i.self_attn),
              post_attention_layernorm=_load_single_leaf(
                  reader,
                  f"{pfx}.post_attention_layernorm.weight",
                  transpose=False,
                  target_dtype=param_dt,
                  sharding=sh_i.post_attention_layernorm,
              ),
              mlp=_load_mlp_unstacked(reader, f"{pfx}.mlp", param_dt, sh_i.mlp),
          )
      )

    scanned_blocks: flint_types.FlintSuperblockWeights[jax.Array] | None
    unscanned_moe: tuple[flint_types.FlintSparseLayerWeights[jax.Array], ...]
    tail_moe: tuple[flint_types.FlintSparseLayerWeights[jax.Array], ...]
    if model.scan_layers:
      prefix, num_sb, tail_start = flint_config.flint_scan_layout(model)
      sb_sh = shardings.decoder.scanned_blocks
      if num_sb > 0 and sb_sh is not None:
        slot_layers = [[prefix + 4 * sb_idx + slot for sb_idx in range(num_sb)] for slot in range(4)]
        scanned_blocks = flint_types.FlintSuperblockWeights(
            sliding_0=_load_sparse_stacked(
                reader,
                slot_layers[0],
                model,
                param_dt,
                sb_sh.sliding_0,
                executor,
            ),
            sliding_1=_load_sparse_stacked(
                reader,
                slot_layers[1],
                model,
                param_dt,
                sb_sh.sliding_1,
                executor,
            ),
            sliding_2=_load_sparse_stacked(
                reader,
                slot_layers[2],
                model,
                param_dt,
                sb_sh.sliding_2,
                executor,
            ),
            global_3=_load_sparse_stacked(
                reader,
                slot_layers[3],
                model,
                param_dt,
                sb_sh.global_3,
                executor,
            ),
        )
      else:
        scanned_blocks = None
      unscanned_moe = ()
      tail_moe = tuple(
          _load_sparse_unstacked(
              reader,
              tail_start + j,
              model,
              param_dt,
              shardings.decoder.tail_moe[j],
              executor,
          )
          for j in range(3)
      )
    else:
      scanned_blocks = None
      num_moe = model.num_hidden_layers - model.num_dense_prefix_layers
      unscanned_moe = tuple(
          _load_sparse_unstacked(
              reader,
              model.num_dense_prefix_layers + j,
              model,
              param_dt,
              shardings.decoder.unscanned_moe[j],
              executor,
          )
          for j in range(num_moe)
      )
      tail_moe = ()

    final_norm = _load_single_leaf(
        reader,
        "model.norm.weight",
        transpose=False,
        target_dtype=param_dt,
        sharding=shardings.final_norm,
    )
    lm_head = _load_single_leaf(
        reader,
        "lm_head.weight",
        transpose=True,
        target_dtype=param_dt,
        sharding=shardings.lm_head,
    )

  return flint_types.FlintParams(
      embed=embed,
      decoder=flint_types.FlintDecoderWeights(
          prefix_dense=tuple(prefix_dense),
          scanned_blocks=scanned_blocks,
          unscanned_moe=unscanned_moe,
          tail_moe=tail_moe,
      ),
      final_norm=final_norm,
      lm_head=lm_head,
  )


def _gather_leaf_to_np(arr: jax.Array | np.ndarray) -> np.ndarray:
  """Gathers a single `jax.Array` leaf to a host `np.ndarray`."""
  if isinstance(arr, np.ndarray):
    return arr
  if arr.is_fully_addressable:
    return np.asarray(arr)
  sharding = arr.sharding
  if not isinstance(sharding, NamedSharding):
    raise ValueError(f"Non-addressable array requires NamedSharding, got {sharding!r}.")
  rep_sharding = NamedSharding(sharding.mesh, PartitionSpec())
  gathered = jax.jit(lambda x: x, out_shardings=rep_sharding)(arr)
  return np.asarray(gathered.addressable_data(0))


def _encode_tensor_bytes(arr: np.ndarray, hf_dtype: str) -> bytes:
  """Encodes `arr` into contiguous little-endian safetensors payload bytes."""
  if hf_dtype == "BF16":
    bf16 = np.ascontiguousarray(_cast_np(arr, np.dtype(ml_dtypes.bfloat16)))
    return bf16.view(np.uint16).astype(np.dtype("<u2"), copy=False).tobytes()
  if hf_dtype == "F16":
    f16 = np.ascontiguousarray(arr.astype(np.dtype("<f2")))
    return f16.tobytes()
  if hf_dtype == "F32":
    f32 = np.ascontiguousarray(_cast_np(arr, np.dtype(np.float32)).astype(np.dtype("<f4"), copy=False))
    return f32.tobytes()
  raise ValueError(f"Unsupported save hf_dtype {hf_dtype!r}; expected one of" f" {sorted(_HF_DTYPE_ITEMSIZE)}.")


def _yield_attn_hf(
    pfx: str,
    attn: flint_types.FlintAttnWeights[np.ndarray],
) -> Iterator[tuple[str, np.ndarray]]:
  """Yields HF attention `(key, [out, in])` pairs in canonical order."""
  yield f"{pfx}.self_attn.q_proj.weight", attn.q_proj.T
  yield f"{pfx}.self_attn.k_proj.weight", attn.k_proj.T
  yield f"{pfx}.self_attn.v_proj.weight", attn.v_proj.T
  yield f"{pfx}.self_attn.o_proj.weight", attn.o_proj.T
  yield f"{pfx}.self_attn.q_norm.weight", attn.q_norm
  yield f"{pfx}.self_attn.k_norm.weight", attn.k_norm
  yield f"{pfx}.self_attn.softmax_offset", attn.sinks


def _yield_mlp_hf(
    prefix: str,
    mlp: flint_types.FlintMLPWeights[np.ndarray],
) -> Iterator[tuple[str, np.ndarray]]:
  """Yields HF SwiGLU MLP `(key, [out, in])` pairs."""
  yield f"{prefix}.gate_proj.weight", mlp.gate_proj.T
  yield f"{prefix}.up_proj.weight", mlp.up_proj.T
  yield f"{prefix}.down_proj.weight", mlp.down_proj.T


def _iter_hf_tensors(
    params: flint_types.FlintParams[jax.Array],
    model: flint_config.FlintModelConfig,
) -> Iterator[tuple[str, np.ndarray]]:
  """Streams `(hf_key, np_array)` in the exact order of `expected_hf_shapes`."""
  yield "model.embed_tokens.weight", _gather_leaf_to_np(params.embed)

  for layer_idx in range(model.num_dense_prefix_layers):
    pfx = f"model.layers.{layer_idx}"
    dense_np = jax.tree.map(_gather_leaf_to_np, params.decoder.prefix_dense[layer_idx])
    yield f"{pfx}.input_layernorm.weight", dense_np.input_layernorm
    yield from _yield_attn_hf(pfx, dense_np.self_attn)
    yield (
        f"{pfx}.post_attention_layernorm.weight",
        dense_np.post_attention_layernorm,
    )
    yield from _yield_mlp_hf(f"{pfx}.mlp", dense_np.mlp)

  slot_cache: dict[tuple[int, int], np.ndarray] = {}

  def _extract_scanned_slot_layer(
      slot_idx: int,
      sb_idx: int,
      slot_weights: flint_types.FlintSparseLayerWeights[jax.Array],
  ) -> flint_types.FlintSparseLayerWeights[np.ndarray]:
    leaves, treedef = jax.tree.flatten(slot_weights)
    out_leaves: list[np.ndarray] = []
    for leaf_idx, leaf in enumerate(leaves):
      elem_bytes = math.prod(leaf.shape) * jnp.dtype(leaf.dtype).itemsize
      cache_key = (slot_idx, leaf_idx)
      if elem_bytes <= _SLOT_CACHE_MAX_BYTES:
        cached = slot_cache.get(cache_key)
        if cached is None:
          cached = _gather_leaf_to_np(leaf)
          slot_cache[cache_key] = cached
        out_leaves.append(cached[sb_idx])
      else:
        out_leaves.append(_gather_leaf_to_np(leaf[sb_idx]))
    return treedef.unflatten(out_leaves)

  moe_inter = model.moe_intermediate_size
  if model.scan_layers:
    prefix, _, tail_start = flint_config.flint_scan_layout(model)
  else:
    prefix = model.num_dense_prefix_layers
    tail_start = model.num_hidden_layers

  for layer_idx in range(model.num_dense_prefix_layers, model.num_hidden_layers):
    pfx = f"model.layers.{layer_idx}"
    if model.scan_layers:
      if layer_idx < tail_start:
        sb_idx, slot_idx = divmod(layer_idx - prefix, 4)
        sb = params.decoder.scanned_blocks
        if sb is None:
          raise ValueError("Expected non-None scanned_blocks for scan_layers.")
        slot_w = (sb.sliding_0, sb.sliding_1, sb.sliding_2, sb.global_3)[slot_idx]
        sparse_np = _extract_scanned_slot_layer(slot_idx, sb_idx, slot_w)
      else:
        sparse_np = jax.tree.map(
            _gather_leaf_to_np,
            params.decoder.tail_moe[layer_idx - tail_start],
        )
    else:
      sparse_np = jax.tree.map(
          _gather_leaf_to_np,
          params.decoder.unscanned_moe[layer_idx - model.num_dense_prefix_layers],
      )

    yield f"{pfx}.input_layernorm.weight", sparse_np.input_layernorm
    yield from _yield_attn_hf(pfx, sparse_np.self_attn)
    yield (
        f"{pfx}.post_attention_layernorm.weight",
        sparse_np.post_attention_layernorm,
    )
    yield f"{pfx}.mlp.gate.weight", sparse_np.moe.router.gate.T
    yield from _yield_mlp_hf(f"{pfx}.mlp.shared_expert", sparse_np.moe.shared_expert)
    gu_all = sparse_np.moe.routed_experts.gate_up_proj
    down_all = sparse_np.moe.routed_experts.down_proj
    for exp_idx in range(model.num_experts):
      epfx = f"{pfx}.mlp.experts.{exp_idx}"
      gu_e = gu_all[exp_idx]
      yield f"{epfx}.gate_proj.weight", gu_e[:, :moe_inter].T
      yield f"{epfx}.up_proj.weight", gu_e[:, moe_inter:].T
      yield f"{epfx}.down_proj.weight", down_all[exp_idx].T

  slot_cache.clear()
  yield "model.norm.weight", _gather_leaf_to_np(params.final_norm)
  yield "lm_head.weight", _gather_leaf_to_np(params.lm_head).T


def _build_shard_header_bytes(
    shard_specs: Sequence[tuple[str, tuple[int, ...], int]],
    hf_dtype: str,
) -> bytes:
  """Encodes an 8-byte-aligned UTF-8 safetensors JSON header."""
  header: dict[str, object] = {_METADATA_KEY: {"format": "pt"}}
  offset = 0
  for key, shape, nbytes in shard_specs:
    header[key] = {
        "dtype": hf_dtype,
        "shape": list(shape),
        "data_offsets": [offset, offset + nbytes],
    }
    offset += nbytes
  raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
  pad = (-len(raw)) % 8
  return raw + (b" " * pad)


def save_hf_checkpoint(
    params: flint_types.FlintParams[jax.Array],
    cfg: flint_config.FlintConfig,
    out_dir: str,
    *,
    hf_dtype: str = "BF16",
    max_shard_bytes: int = 5 * 1024**3,
) -> list[str]:
  """Exports `FlintParams` to single- or multi-shard HuggingFace safetensors.

  Args:
    params: Sharded or host `FlintParams` matching `cfg`.
    cfg: Validated `FlintConfig`.
    out_dir: Destination directory path (created if absent).
    hf_dtype: Output safetensors dtype (`'BF16'`, `'F16'`, or `'F32'`).
    max_shard_bytes: Maximum payload byte budget per `.safetensors` shard.

  Returns:
    Ordered list of written `.safetensors` shard file paths.

  Raises:
    ValueError: If `hf_dtype`, `max_shard_bytes`, or `params` shapes/structure
      are invalid.
    RuntimeError: If internal tensor streaming order or encoded payload byte
      length disagrees with the precomputed shard layout.
  """
  if hf_dtype not in _HF_DTYPE_ITEMSIZE:
    raise ValueError(f"Unsupported hf_dtype={hf_dtype!r}; expected one of" f" {sorted(_HF_DTYPE_ITEMSIZE)}.")
  if max_shard_bytes <= 0:
    raise ValueError(f"max_shard_bytes must be > 0, got {max_shard_bytes}.")

  expected_tree = flint_sharding.abstract_param_shapes(cfg)
  if jax.tree.structure(params) != jax.tree.structure(expected_tree):
    raise ValueError("params PyTree structure does not match cfg (check scan_layers).")
  for got_leaf, exp_sds in zip(jax.tree.leaves(params), jax.tree.leaves(expected_tree), strict=True):
    if tuple(got_leaf.shape) != exp_sds.shape:
      raise ValueError(f"Parameter leaf shape {tuple(got_leaf.shape)} does not match" f" expected {exp_sds.shape}.")

  itemsize = _HF_DTYPE_ITEMSIZE[hf_dtype]
  expected_shapes = expected_hf_shapes(cfg.model)
  shards: list[list[tuple[str, tuple[int, ...], int]]] = []
  cur_shard: list[tuple[str, tuple[int, ...], int]] = []
  cur_bytes = 0
  total_bytes = 0
  for key, shape in expected_shapes.items():
    nbytes = math.prod(shape) * itemsize
    total_bytes += nbytes
    if cur_shard and cur_bytes + nbytes > max_shard_bytes:
      shards.append(cur_shard)
      cur_shard = []
      cur_bytes = 0
    cur_shard.append((key, shape, nbytes))
    cur_bytes += nbytes
  if cur_shard:
    shards.append(cur_shard)

  num_shards = len(shards)
  if num_shards == 1:
    shard_filenames = [_SINGLE_SHARD_FILENAME]
  else:
    shard_filenames = [f"model-{idx:05d}-of-{num_shards:05d}.safetensors" for idx in range(1, num_shards + 1)]
  written_paths = [os.path.join(out_dir, name) for name in shard_filenames]

  is_writer = jax.process_index() == 0
  if is_writer:
    epath.Path(out_dir).mkdir(parents=True, exist_ok=True)

  tensor_iter = _iter_hf_tensors(params, cfg.model)
  weight_map: dict[str, str] = {}

  for shard_name, shard_path, shard_specs in zip(shard_filenames, written_paths, shards, strict=True):
    header_bytes = _build_shard_header_bytes(shard_specs, hf_dtype)
    if is_writer:
      with epath.Path(shard_path).open("wb") as out_file:
        out_file.write(struct.pack("<Q", len(header_bytes)))
        out_file.write(header_bytes)
        for exp_key, exp_shape, exp_nbytes in shard_specs:
          got_key, arr = next(tensor_iter)
          if got_key != exp_key or arr.shape != exp_shape:
            raise RuntimeError(
                f"Internal stream mismatch: expected ({exp_key!r}," f" {exp_shape}), got ({got_key!r}, {arr.shape})."
            )
          payload = _encode_tensor_bytes(arr, hf_dtype)
          if len(payload) != exp_nbytes:
            raise RuntimeError(f"Encoded payload length {len(payload)} != {exp_nbytes} for" f" {exp_key!r}.")
          out_file.write(payload)
          weight_map[exp_key] = shard_name
    else:
      for exp_key, exp_shape, _ in shard_specs:
        got_key, arr = next(tensor_iter)
        if got_key != exp_key or arr.shape != exp_shape:
          raise RuntimeError(
              f"Internal stream mismatch: expected ({exp_key!r}, {exp_shape})," f" got ({got_key!r}, {arr.shape})."
          )
        weight_map[exp_key] = shard_name

  for extra_key, _ in tensor_iter:
    raise RuntimeError(f"Unexpected trailing tensor yielded: {extra_key!r}.")

  if is_writer and num_shards > 1:
    index_payload = json.dumps(
        {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    index_path = os.path.join(out_dir, _INDEX_FILENAME)
    with epath.Path(index_path).open("wb") as idx_file:
      idx_file.write(index_payload)

  return written_paths
