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

Each host reads only the rows its own TPU chips need. Neighbouring byte ranges
are merged and fetched as `chunk_bytes` ranged reads.

`fetch` does the GCS reads and copies onto this host's own TPU chips, with no
communication between hosts, so it is safe on a background thread.

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
# Byte ranges closer together than this are fetched as one range: reading a small
# gap is cheaper than another request.
_MERGE_GAP_BYTES = 1 << 20
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
  """A tensor, built once all its pieces are read."""

  pieces: list[_Piece]
  build: Callable[[], None]
  blocks_left: int = 0


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
  ):
    """Finds the `.safetensors` files at `path` (a file or a directory) and reads their headers.

    Args:
      path: A `.safetensors` file, or a directory of them (local or gs://).
      num_threads: Read threads per host.
      chunk_bytes: Size of each ranged read sent to storage.
    """
    if chunk_bytes <= 0:
      raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}.")
    self._chunk_bytes = chunk_bytes
    self._pool = concurrent.futures.ThreadPoolExecutor(num_threads, thread_name_prefix="safetensors_read")
    try:
      self._index = self._read_index(path, num_threads, chunk_bytes)
    except BaseException:
      # `__exit__` never runs if `__init__` raises, so shut the read threads down here.
      self._pool.shutdown(wait=False, cancel_futures=True)
      raise

  def _read_index(self, path: str, num_threads: int, chunk_bytes: int) -> dict[str, _Entry]:
    """Reads every file's header and indexes its tensors by name."""
    root = epath.Path(path)
    files = sorted(str(f) for f in root.glob(f"*{_SUFFIX}")) if root.is_dir() else [str(root)]
    if not files:
      raise FileNotFoundError(f"No {_SUFFIX} files in {path}.")
    if jax.process_index() == 0:
      max_logging.log(
          f"SafetensorsReader host 0: {len(files)} files, {num_threads} read threads,"
          f" {chunk_bytes >> 20} MiB reads, {len(os.sched_getaffinity(0))} usable CPUs (of {os.cpu_count()})"
      )
    index: dict[str, _Entry] = {}
    for file, (header, data_start, length) in zip(files, self._pool.map(_read_header, files)):
      for name, info in header.items():
        if name == "__metadata__":
          continue
        if name in index:
          raise ValueError(f"Tensor {name} is in both {index[name].path} and {file}.")
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
        index[name] = _Entry(file, data_start + begin, shape, dtype)
    return index

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

  def fetch(self, request: dict[str, jax.ShapeDtypeStruct]) -> dict[str, jax.Array]:
    """Reads this host's part of the requested tensors onto its TPU chips.

    No communication between hosts, so this is safe to run on a background thread.

    Args:
      request: Tensor name -> `jax.ShapeDtypeStruct` with the checkpoint's shape and dtype and the
        sharding to load it with.

    Returns:
      Tensor name -> `jax.Array`, in the order of `request`.
    """
    t_start = time.perf_counter()
    arrays, units = {}, []
    for name, sds in request.items():
      self._check(name, sds)
      pieces = self._row_pieces(name, sds.sharding)
      if pieces:
        units.append(_Unit(pieces, functools.partial(self._build, name, sds, pieces, arrays)))
      else:  # None of this host's TPU chips hold any of it.
        arrays[name] = jax.make_array_from_single_device_arrays(sds.shape, sds.sharding, [], dtype=sds.dtype)

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

    fetched_bytes = sum(b.end - b.start for b in blocks)
    wall = max(t_last_read - t_start, 1e-9)
    if jax.process_index() == 0:
      max_logging.log(
          f"SafetensorsReader host 0: {len(request)} tensors, {sum(len(u.pieces) for u in units)} pieces in"
          f" {len(blocks)} byte ranges over"
          f" {len(by_file)} files, {len(futures)} ranged reads, {fetched_bytes / 1e9:.3f} GB in {wall:.2f}s"
          f" ({fetched_bytes / 1e9 / wall:.2f} GB/s); avg read {read_seconds / max(len(futures), 1):.3f}s,"
          f" {read_seconds / wall:.1f} reads in flight on average; into HBM {build_seconds:.2f}s;"
          f" total {time.perf_counter() - t_start:.2f}s"
      )
    return {name: arrays[name] for name in request}

  def read(self, request: dict[str, jax.ShapeDtypeStruct]) -> dict[str, jax.Array]:
    """Reads the requested tensors, each with the sharding it is requested with.

    Args:
      request: Tensor name -> `jax.ShapeDtypeStruct` with the checkpoint's shape and dtype and the
        sharding to load it with.

    Returns:
      Tensor name -> `jax.Array`, in the order of `request`.
    """
    return self.fetch(request)
