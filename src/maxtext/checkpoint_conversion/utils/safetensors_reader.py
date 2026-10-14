# Copyright 2026 Google LLC
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

"""Reads tensors from SafeTensors files (local or gs://) straight into sharded `jax.Array`s.

This is a stand-in for Orbax's SafeTensors loader (`ocp_v1.load` with the
SAFETENSORS layout), which `hf_streaming_load` used before. On a v4-8 host
Orbax read a DeepSeek-V2-Lite file at ~1.1 GB/s, while plain parallel ranged
reads with the same GCS client reached ~2.8 GB/s. Orbax 0.12.x loses the
difference in three places, which this reader avoids:

1. Every `load` call re-reads every file's header (~0.8 s per call). Here the
   headers are read once, when the reader is created.
2. Every downloaded chunk is copied into its buffer on the one asyncio thread,
   which then can't start new reads. Here each read thread copies its own chunk.
3. No tensor is moved to the TPU chips until all of its file's reads are done.
   Here each tensor is moved as soon as its bytes are in, while later reads
   continue.

Two ways to read a tensor:

* By rows: each host reads only the rows its own TPU chips need. Neighbouring
  byte ranges are merged and fetched as `chunk_bytes` ranged reads.
* Whole, then rearranged ("stacked"): with many TPU chips, a tensor's rows per
  chip get tiny (a 16 MB DeepSeek-V4 expert over 256 chips is 64 KB per chip),
  and reading by rows turns into millions of tiny GCS requests. So a group of
  same-shape tensors whose pieces would be smaller than `min_piece_bytes` is
  read as whole tensors instead, spread over the TPU chips (each tensor on one
  chip, so each host reads whole tensors with large requests), and then
  rearranged into the requested sharding on the TPU chips, over ICI.

`read` = `fetch` + `unpack`. `fetch` does the GCS reads and copies onto this
host's own TPU chips, with no communication between hosts, so it is safe on a
background thread. `unpack` runs the jitted rearranging, which every host must
run in the same order, so it belongs on the main thread.

Once Orbax's loader fixes these, `hf_streaming_load` can switch back to it.
"""

import collections
import concurrent.futures
import dataclasses
import functools
import json
import math
import os
import time
from typing import Any, Callable

from etils import epath
import jax
import jax.numpy as jnp
from maxtext.utils import max_logging
import numpy as np

# Size of each ranged read sent to storage. On a v4-8 host, 32 MiB beat both
# 16 MiB and 128 MiB.
READ_CHUNK_BYTES = 32 << 20
# Read threads per host. On a v4-8 host, 32-64 threads reached ~2.3 GB/s from GCS
# with this reader; 128 was slower.
READ_THREADS = 64
# Groups of same-shape tensors whose per-chip pieces would be smaller than this are
# read whole and rearranged on the TPU chips (see the module docstring).
MIN_PIECE_BYTES = 4 << 20
# Byte ranges closer together than this are fetched as one range: reading a small
# gap is cheaper than another request.
_MERGE_GAP_BYTES = 1 << 20
# Most tensors split out of a stack by one program. A program per tensor costs a launch each
# (DeepSeek-V4 has ~34k tensors), while one program for thousands of tensors compiles slowly.
_SPLIT_CHUNK = 256
_SUFFIX = ".safetensors"
_DTYPES = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "I16": np.int16,
    "U16": np.uint16,
    "I32": np.int32,
    "U32": np.uint32,
    "I64": np.int64,
    "U64": np.uint64,
    "F16": np.float16,
    "BF16": jnp.bfloat16,
    "F32": np.float32,
    "F64": np.float64,
    "F8_E4M3": jnp.float8_e4m3fn,
    "F8_E5M2": jnp.float8_e5m2,
}


@dataclasses.dataclass(frozen=True)
class _Entry:
  """Where one tensor's bytes are."""

  path: str
  start: int  # Absolute byte offset of the tensor in the file.
  shape: tuple[int, ...]
  dtype: np.dtype


@dataclasses.dataclass(eq=False)
class _Piece:
  """A run of whole rows of one tensor (possibly all of it) that this host reads."""

  name: str
  start: int  # Absolute byte range of the rows in the file.
  end: int
  rows_shape: tuple[int, ...]  # Shape of those rows.
  sub_index: tuple[slice, ...] | None  # Applied to the rows when the sharding also splits other dims.
  devices: list[Any]
  block: "_Block | None" = None


@dataclasses.dataclass(eq=False)
class _Block:
  """One merged byte range of one file, fetched as one or more ranged reads into `buf`."""

  path: str
  start: int
  end: int
  pieces: list[_Piece]
  buf: np.ndarray | None = None
  reads_left: int = 0  # Ranged reads not yet done.
  units_left: int = 0  # Units with a piece here that are not yet built; `buf` is freed at 0.


@dataclasses.dataclass(eq=False)
class _Unit:
  """What is built once all its pieces are read: a tensor read by rows, or one TPU chip's part of a stack."""

  pieces: list[_Piece]
  build: Callable[[], None]
  blocks_left: int = 0


@dataclasses.dataclass(eq=False)
class _Stack:
  """Same-shape tensors read whole, `per_chip` consecutive tensors on each TPU chip, then rearranged."""

  names: list[str]  # In stack order; the stack is padded with zeros to a multiple of the chip count.
  shape: tuple[int, ...]  # Of one tensor.
  dtype: np.dtype
  sharding: jax.sharding.NamedSharding  # Requested sharding of each tensor.
  stacked_sharding: jax.sharding.NamedSharding  # How the stack is read: split along the stack axis.
  num_padded: int
  shards: dict = dataclasses.field(default_factory=dict)  # Device -> its part of the stack.
  array: jax.Array | None = None


@dataclasses.dataclass
class Fetched:
  """Tensors read onto this host's TPU chips by `SafetensorsReader.fetch`, before `unpack`."""

  names: list[str]  # Requested order.
  arrays: dict[str, jax.Array]  # Tensors read by rows: already in their requested sharding.
  stacks: list[_Stack]


def _bounds(index: tuple, shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
  bounds = []
  for s, dim in zip(index, shape):
    start, stop, step = s.indices(dim)
    if step != 1:
      raise ValueError(f"Strided shardings are not supported: {index}.")
    bounds.append((start, stop))
  return tuple(bounds)


def _merge(path: str, pieces: list[_Piece], max_gap: int) -> list[_Block]:
  """Merges one file's pieces into blocks of byte ranges at most `max_gap` apart."""
  blocks = []
  for piece in sorted(pieces, key=lambda p: p.start):
    if blocks and piece.start <= blocks[-1].end + max_gap:
      blocks[-1].end = max(blocks[-1].end, piece.end)
      blocks[-1].pieces.append(piece)
    else:
      blocks.append(_Block(path, piece.start, piece.end, [piece]))
    piece.block = blocks[-1]
  return blocks


def _read_header(path: str) -> tuple[dict, int, int]:
  """Returns a file's header, the offset where its data starts, and its size in bytes."""
  p = epath.Path(path)
  with p.open("rb") as f:
    header_len = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(header_len))
  return header, 8 + header_len, p.stat().length


@functools.lru_cache(maxsize=None)
def _reshard_fn(sharding: jax.sharding.Sharding):
  return jax.jit(lambda x: x, out_shardings=sharding)


@functools.lru_cache(maxsize=None)
def _split_fn(sharding: jax.sharding.NamedSharding, num: int):
  """One program that splits `num` tensors, from `start`, out of a stack already in `P(None, *spec)`."""

  def split(x, start):
    x = jax.lax.dynamic_slice_in_dim(x, start, num, axis=0)
    return tuple(x[i] for i in range(num))

  # `start` is traced, so every full chunk of a stack reuses one compiled program.
  return jax.jit(split, out_shardings=(sharding,) * num)


class SafetensorsReader:
  """Reads tensors from a SafeTensors checkpoint into sharded `jax.Array`s.

  Usage:
    with SafetensorsReader("gs://bucket/model") as reader:
      shapes = reader.metadata
      arrays = reader.read({"lm_head.weight": jax.ShapeDtypeStruct(shape, dtype, sharding=...)})
  """

  def __init__(
      self,
      path: str,
      num_threads: int = READ_THREADS,
      chunk_bytes: int = READ_CHUNK_BYTES,
      min_piece_bytes: int = MIN_PIECE_BYTES,
  ):
    """Finds the `.safetensors` files at `path` (a file or a directory) and reads their headers.

    Args:
      path: A `.safetensors` file, or a directory of them (local or gs://).
      num_threads: Read threads per host.
      chunk_bytes: Size of each ranged read sent to storage.
      min_piece_bytes: Groups of same-shape tensors whose per-chip pieces would be smaller than
        this are read whole and rearranged on the TPU chips. 0 always reads by rows.
    """
    if chunk_bytes <= 0:
      raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}.")
    self._chunk_bytes = chunk_bytes
    self._min_piece_bytes = min_piece_bytes
    self._pool = concurrent.futures.ThreadPoolExecutor(num_threads, thread_name_prefix="safetensors_read")
    root = epath.Path(path)
    files = sorted(str(f) for f in root.glob(f"*{_SUFFIX}")) if root.is_dir() else [str(root)]
    if not files:
      raise FileNotFoundError(f"No {_SUFFIX} files in {path}.")
    if jax.process_index() == 0:
      max_logging.log(
          f"SafetensorsReader host 0: {len(files)} files, {num_threads} read threads,"
          f" {chunk_bytes >> 20} MiB reads, {len(os.sched_getaffinity(0))} usable CPUs (of {os.cpu_count()})"
      )
    self._index: dict[str, _Entry] = {}
    for file, (header, data_start, length) in zip(files, self._pool.map(_read_header, files)):
      for name, info in header.items():
        if name == "__metadata__":
          continue
        if name in self._index:
          raise ValueError(f"Tensor {name} is in both {self._index[name].path} and {file}.")
        begin, end = info["data_offsets"]
        if data_start + end > length:
          raise ValueError(f"{file} is truncated: {name} ends at byte {data_start + end}, the file has {length}.")
        try:
          dtype = np.dtype(_DTYPES[info["dtype"]])
        except KeyError as e:
          raise ValueError(f"Unsupported SafeTensors dtype {info['dtype']} for {name} in {file}.") from e
        shape = tuple(info["shape"])
        if end - begin != math.prod(shape) * dtype.itemsize:
          raise ValueError(f"{name} in {file} has {end - begin} bytes, which doesn't match shape {shape} of {dtype}.")
        self._index[name] = _Entry(file, data_start + begin, shape, dtype)

  @property
  def metadata(self) -> dict[str, jax.ShapeDtypeStruct]:
    """Shape and dtype of every tensor in the checkpoint."""
    return {name: jax.ShapeDtypeStruct(e.shape, e.dtype) for name, e in self._index.items()}

  def close(self):
    self._pool.shutdown(wait=True, cancel_futures=True)

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()

  def _check(self, name: str, request: jax.ShapeDtypeStruct) -> _Entry:
    """Raises unless `name` is in the checkpoint with the requested shape and dtype, and has a sharding."""
    if name not in self._index:
      raise KeyError(f"Tensor {name} is not in the checkpoint.")
    entry = self._index[name]
    if tuple(request.shape) != entry.shape or np.dtype(request.dtype) != entry.dtype:
      raise ValueError(
          f"Requested {name} as {tuple(request.shape)} {np.dtype(request.dtype)}, but the checkpoint has"
          f" {entry.shape} {entry.dtype}."
      )
    if getattr(request, "sharding", None) is None:
      raise ValueError(f"Requested {name} without a sharding.")
    return entry

  def _whole(self, name: str, devices: list[Any]) -> _Piece:
    entry = self._index[name]
    nbytes = math.prod(entry.shape) * entry.dtype.itemsize
    return _Piece(name, entry.start, entry.start + nbytes, entry.shape, None, devices)

  def _row_pieces(self, name: str, sharding: jax.sharding.Sharding) -> list[_Piece]:
    """The row runs this host must read for one tensor, one per distinct piece."""
    entry = self._index[name]
    by_bounds = collections.defaultdict(list)
    for device, index in sharding.addressable_devices_indices_map(entry.shape).items():
      by_bounds[_bounds(index, entry.shape)].append(device)
    if not entry.shape:  # Scalar.
      return [self._whole(name, devices) for devices in by_bounds.values()]
    row_bytes = math.prod(entry.shape[1:]) * entry.dtype.itemsize
    pieces = []
    for bounds, devices in by_bounds.items():
      (r0, r1), rest = bounds[0], bounds[1:]
      split_rest = any(b != (0, dim) for b, dim in zip(rest, entry.shape[1:]))
      pieces.append(
          _Piece(
              name,
              entry.start + r0 * row_bytes,
              entry.start + r1 * row_bytes,
              (r1 - r0,) + entry.shape[1:],
              (slice(None),) + tuple(slice(a, b) for a, b in rest) if split_rest else None,
              devices,
          )
      )
    return pieces

  def _stackable(self, names: list[str], request: jax.ShapeDtypeStruct) -> bool:
    """Whether a group of same-shape tensors is read whole and rearranged instead of by rows."""
    sharding = request.sharding
    num_devices = len(sharding.device_set)
    if not isinstance(sharding, jax.sharding.NamedSharding) or num_devices == 1 or not request.shape:
      return False
    piece_bytes = math.prod(sharding.shard_shape(request.shape)) * np.dtype(request.dtype).itemsize
    # Padding the stack to a multiple of the chip count may at most double it.
    return 0 < piece_bytes < self._min_piece_bytes and 2 * len(names) >= num_devices

  def _plan_stack(self, names: list[str], request: jax.ShapeDtypeStruct, units: list[_Unit]) -> _Stack:
    """Spreads whole tensors over the TPU chips, `per_chip` consecutive ones (in file order) each."""
    sharding = request.sharding
    num_devices = len(sharding.device_set)
    names = sorted(names, key=lambda n: (self._index[n].path, self._index[n].start))
    num_padded = -len(names) % num_devices
    mesh = sharding.mesh
    stack = _Stack(
        names=names,
        shape=tuple(request.shape),
        dtype=np.dtype(request.dtype),
        sharding=sharding,
        stacked_sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(tuple(mesh.axis_names))),
        num_padded=num_padded,
    )
    stacked_shape = (len(names) + num_padded,) + stack.shape
    for device, index in stack.stacked_sharding.addressable_devices_indices_map(stacked_shape).items():
      first, last = _bounds(index[:1], stacked_shape[:1])[0]
      pieces = [self._whole(names[i], [device]) for i in range(first, min(last, len(names)))]
      units.append(_Unit(pieces, functools.partial(self._build_stack_part, stack, device, first, last, pieces)))
    return stack

  def _build_stack_part(self, stack: _Stack, device, first: int, last: int, pieces: list[_Piece]):
    part = np.zeros((last - first,) + stack.shape, stack.dtype)
    for i, piece in enumerate(pieces):
      part[i] = self._rows(piece)
    stack.shards[device] = jax.device_put(part, device)

  def _rows(self, piece: _Piece) -> np.ndarray:
    block = piece.block
    entry = self._index[piece.name]
    rows = block.buf[piece.start - block.start : piece.end - block.start].view(entry.dtype).reshape(piece.rows_shape)
    return rows if piece.sub_index is None else rows[piece.sub_index]

  def _build(self, name: str, request: jax.ShapeDtypeStruct, pieces: list[_Piece], arrays: dict):
    entry = self._index[name]
    shards = [jax.device_put(self._rows(piece), device) for piece in pieces for device in piece.devices]
    arrays[name] = jax.make_array_from_single_device_arrays(entry.shape, request.sharding, shards)

  def _read_range(self, block: _Block, offset: int, length: int) -> float:
    """Reads `length` bytes at `offset` into the block's buffer; returns the seconds it took."""
    t = time.perf_counter()
    with epath.Path(block.path).open("rb") as f:
      f.seek(offset)
      data = f.read(length)
    if len(data) != length:
      raise IOError(f"Short read from {block.path}: wanted {length} bytes at {offset}, got {len(data)}.")
    block.buf[offset - block.start : offset - block.start + length] = np.frombuffer(data, np.uint8)
    return time.perf_counter() - t

  def fetch(self, request: dict[str, jax.ShapeDtypeStruct]) -> Fetched:
    """Reads this host's part of the requested tensors onto its TPU chips; finish with `unpack`.

    No communication between hosts, so this is safe to run on a background thread.

    Args:
      request: Tensor name -> `jax.ShapeDtypeStruct` with the checkpoint's shape and dtype and the
        sharding to load it with.

    Returns:
      What `unpack` turns into the requested arrays.
    """
    t_start = time.perf_counter()
    groups = collections.defaultdict(list)
    for name, sds in request.items():
      self._check(name, sds)
      groups[(tuple(sds.shape), np.dtype(sds.dtype), sds.sharding)].append(name)

    arrays, stacks, units = {}, [], []
    for names in groups.values():
      sds = request[names[0]]
      if self._stackable(names, sds):
        stacks.append(self._plan_stack(names, sds, units))
        continue
      for name in names:
        pieces = self._row_pieces(name, request[name].sharding)
        if pieces:
          units.append(_Unit(pieces, functools.partial(self._build, name, request[name], pieces, arrays)))
        else:  # None of this host's TPU chips hold any of it.
          arrays[name] = jax.make_array_from_single_device_arrays(
              request[name].shape, request[name].sharding, [], dtype=request[name].dtype
          )

    by_file = collections.defaultdict(list)
    for unit in units:
      for piece in unit.pieces:
        by_file[self._index[piece.name].path].append(piece)
    blocks = [block for path, pieces in by_file.items() for block in _merge(path, pieces, _MERGE_GAP_BYTES)]

    # A unit is built once every block holding one of its pieces is fully read, and a
    # block's host buffer is dropped once every unit using it is built.
    units_of_block = collections.defaultdict(list)
    for unit in units:
      used = {id(p.block): p.block for p in unit.pieces}
      unit.blocks_left = len(used)
      for block in used.values():
        block.units_left += 1
        units_of_block[id(block)].append(unit)
    build_seconds = 0.0

    def build(unit):
      nonlocal build_seconds
      t = time.perf_counter()
      unit.build()
      build_seconds += time.perf_counter() - t
      for used in {id(p.block): p.block for p in unit.pieces}.values():
        used.units_left -= 1
        if not used.units_left:
          used.buf = None  # The TPU chips now hold copies.

    def finish(block):
      for unit in units_of_block[id(block)]:
        unit.blocks_left -= 1
        if not unit.blocks_left:
          build(unit)

    futures = {}
    for block in blocks:
      block.buf = np.empty(block.end - block.start, np.uint8)
      offsets = range(block.start, block.end, self._chunk_bytes)
      block.reads_left = len(offsets)
      for offset in offsets:
        length = min(self._chunk_bytes, block.end - offset)
        futures[self._pool.submit(self._read_range, block, offset, length)] = block
    read_seconds, t_last_read = 0.0, t_start
    try:
      for unit in units:
        if not unit.pieces:  # A TPU chip that holds only stack padding.
          build(unit)
      for block in blocks:
        if not block.reads_left:  # Zero-byte tensors.
          finish(block)
      for future in concurrent.futures.as_completed(futures):
        read_seconds += future.result()
        t_last_read = time.perf_counter()
        block = futures[future]
        block.reads_left -= 1
        if not block.reads_left:
          finish(block)
    except BaseException:
      for future in futures:
        future.cancel()
      raise

    for stack in stacks:
      stacked_shape = (len(stack.names) + stack.num_padded,) + stack.shape
      local = stack.stacked_sharding.addressable_devices_indices_map(stacked_shape)
      stack.array = jax.make_array_from_single_device_arrays(
          stacked_shape, stack.stacked_sharding, [stack.shards.pop(device) for device in local]
      )
    fetched_bytes = sum(b.end - b.start for b in blocks)
    wall = max(t_last_read - t_start, 1e-9)
    stacked = sum(len(s.names) for s in stacks)
    if jax.process_index() == 0:
      max_logging.log(
          f"SafetensorsReader host 0: {len(request)} tensors ({stacked} read whole in"
          f" {len(stacks)} stacks), {sum(len(u.pieces) for u in units)} pieces in {len(blocks)} byte ranges over"
          f" {len(by_file)} files, {len(futures)} ranged reads, {fetched_bytes / 1e9:.3f} GB in {wall:.2f}s"
          f" ({fetched_bytes / 1e9 / wall:.2f} GB/s); avg read {read_seconds / max(len(futures), 1):.3f}s,"
          f" {read_seconds / wall:.1f} reads in flight on average; into HBM {build_seconds:.2f}s;"
          f" total {time.perf_counter() - t_start:.2f}s"
      )
    return Fetched(names=list(request), arrays=arrays, stacks=stacks)

  def unpack(self, fetched: Fetched) -> dict[str, jax.Array]:
    """Rearranges stacked tensors into their requested shardings on the TPU chips.

    Every host must call this for the same `fetch` results in the same order.

    Returns:
      Tensor name -> `jax.Array`, in the order of the request.
    """
    arrays = dict(fetched.arrays)
    for stack in fetched.stacks:
      # Rearrange so every TPU chip holds its piece of every tensor (one all-to-all over ICI);
      # splitting the tensors out is then local to each chip.
      by_piece = jax.sharding.NamedSharding(stack.sharding.mesh, jax.sharding.PartitionSpec(None, *stack.sharding.spec))
      stacked, stack.array = _reshard_fn(by_piece)(stack.array), None
      for start in range(0, len(stack.names), _SPLIT_CHUNK):
        names = stack.names[start : start + _SPLIT_CHUNK]
        arrays.update(zip(names, _split_fn(stack.sharding, len(names))(stacked, start)))
      del stacked
    fetched.stacks = []
    return {name: arrays[name] for name in fetched.names}

  def read(self, request: dict[str, jax.ShapeDtypeStruct]) -> dict[str, jax.Array]:
    """Reads the requested tensors, each with the sharding it is requested with (`fetch` + `unpack`).

    Args:
      request: Tensor name -> `jax.ShapeDtypeStruct` with the checkpoint's shape and dtype and the
        sharding to load it with.

    Returns:
      Tensor name -> `jax.Array`, in the order of `request`.
    """
    return self.unpack(self.fetch(request))
