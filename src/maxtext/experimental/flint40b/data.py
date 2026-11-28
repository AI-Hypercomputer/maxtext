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

"""Multi-context pre-tokenized C4 ArrayRecord and synthetic data pipelines."""

from __future__ import annotations

import bisect
from collections.abc import Callable
import concurrent.futures
import functools
import threading
import types

from etils import epath
from google.protobuf import message
import jax
import numpy as np

from maxtext.experimental.flint40b import flint_types
from maxtext.input_pipeline.protos import example_pb2
from array_record.python import array_record_module

TOKEN_FEATURE = "text"
_MAX_OPEN_WORKERS = 64
_MAX_IO_WORKERS = 32
_MAX_CACHED_EPOCHS = 4


def parse_row(record: bytes) -> np.ndarray:
  """Decodes one serialized `tf.train.Example` into its int32 1D token array.

  Args:
    record: Serialized `tf.train.Example` bytes containing `TOKEN_FEATURE`.

  Returns:
    1D `np.ndarray` of `int32` token IDs.

  Raises:
    ValueError: If `record` is malformed or `TOKEN_FEATURE` is missing or empty.
  """
  if not record:
    raise ValueError(f"Serialized tf.train.Example is empty; missing {TOKEN_FEATURE!r}.")
  try:
    example = example_pb2.Example.FromString(record)
  except message.DecodeError as err:
    raise ValueError(f"Failed to decode tf.train.Example: {err}") from err
  values = example.features.feature[TOKEN_FEATURE].int64_list.value
  if not values:
    raise ValueError(f"Feature {TOKEN_FEATURE!r} is missing or empty in tf.train.Example.")
  arr64 = np.asarray(values, dtype=np.int64)
  if int(arr64.min()) < 0 or int(arr64.max()) > np.iinfo(np.int32).max:
    raise ValueError("Out-of-range token IDs in tf.train.Example:" f" min={int(arr64.min())}, max={int(arr64.max())}.")
  return arr64.astype(np.int32)


def _resolve_array_record_paths(data_dir_or_glob: str) -> tuple[str, ...]:
  """Resolves a directory, file path, or final-component glob into sorted shard paths."""
  if not data_dir_or_glob:
    raise FileNotFoundError("data_dir_or_glob must be non-empty.")
  if "*" in data_dir_or_glob or "?" in data_dir_or_glob:
    pattern = epath.Path(data_dir_or_glob)
    paths = sorted(str(p) for p in pattern.parent.glob(pattern.name))
  elif epath.Path(data_dir_or_glob).is_dir():
    paths = sorted(str(p) for p in epath.Path(data_dir_or_glob).glob("*.array_record"))
  elif epath.Path(data_dir_or_glob).exists():
    paths = [data_dir_or_glob]
  else:
    paths = []
  if not paths:
    raise FileNotFoundError(f"No ArrayRecord files found for {data_dir_or_glob!r}.")
  return tuple(paths)


def _make_sequence_metadata(
    num_chunks: int,
    chunk_len: int,
    *,
    segment_per_chunk: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Builds 1D `(positions, segment_ids, loss_mask)` templates of `int32`.

  Args:
    num_chunks: Number of concatenated chunks per row.
    chunk_len: Tokens per chunk.
    segment_per_chunk: If `False`, the row is one full-length causal sequence
      (`positions=0..S-1`, `segment_ids=1`). If `True`, each chunk is its own
      segment (`positions` tiled per chunk, `segment_ids=1..num_chunks`).

  Returns:
    `(positions, segment_ids, loss_mask)`, each `int32[num_chunks * chunk_len]`,
    with `loss_mask=0` on the last token of every chunk.
  """
  seq_len = num_chunks * chunk_len
  if segment_per_chunk:
    positions_1d = np.tile(np.arange(chunk_len, dtype=np.int32), num_chunks)
    segment_ids_1d = np.repeat(np.arange(1, num_chunks + 1, dtype=np.int32), chunk_len)
  else:
    positions_1d = np.arange(seq_len, dtype=np.int32)
    segment_ids_1d = np.ones(seq_len, dtype=np.int32)
  loss_mask_2d = np.ones((num_chunks, chunk_len), dtype=np.int32)
  loss_mask_2d[:, -1] = 0
  loss_mask_1d = loss_mask_2d.reshape(seq_len)
  return positions_1d, segment_ids_1d, loss_mask_1d


def _shift_chunks_left(chunks: np.ndarray) -> np.ndarray:
  """Left-shifts `int32[B, k, chunk_len]` within each chunk with trailing 0."""
  return np.concatenate([chunks[:, :, 1:], np.zeros_like(chunks[:, :, :1])], axis=2)


_CallbackIndex = tuple[slice, ...] | None


def _normalize_index(index: _CallbackIndex) -> tuple[slice, slice]:
  if index is None:
    return slice(None), slice(None)
  return index[0], index[1]


def _make_sharded_flint_batch(
    *,
    global_batch_size: int,
    seq_len: int,
    num_chunks: int,
    chunk_len: int,
    positions_1d: np.ndarray,
    segment_ids_1d: np.ndarray,
    loss_mask_1d: np.ndarray,
    sharding: jax.sharding.Sharding,
    row_chunks_fn: Callable[[int], np.ndarray],
) -> flint_types.FlintBatch:
  """Constructs a host-local sharded `FlintBatch` via callbacks."""
  shape = (global_batch_size, seq_len)
  cached_row = functools.cache(row_chunks_fn)

  def _rows_chunks(b_slice: slice) -> np.ndarray:
    b_indices = range(*b_slice.indices(global_batch_size))
    if not b_indices:
      return np.empty((0, num_chunks, chunk_len), dtype=np.int32)
    return np.stack([cached_row(b) for b in b_indices], axis=0)

  def _tokens(index: _CallbackIndex) -> np.ndarray:
    b_slice, s_slice = _normalize_index(index)
    chunks = _rows_chunks(b_slice)
    return chunks.reshape(chunks.shape[0], seq_len)[:, s_slice].copy()

  def _targets(index: _CallbackIndex) -> np.ndarray:
    b_slice, s_slice = _normalize_index(index)
    chunks = _rows_chunks(b_slice)
    shifted = _shift_chunks_left(chunks).reshape(chunks.shape[0], seq_len)
    return shifted[:, s_slice].copy()

  def _make_template_cb(
      template_1d: np.ndarray,
  ) -> Callable[[_CallbackIndex], np.ndarray]:
    def _cb(index: _CallbackIndex) -> np.ndarray:
      b_slice, s_slice = _normalize_index(index)
      b_len = len(range(*b_slice.indices(global_batch_size)))
      sliced = template_1d[s_slice]
      return np.broadcast_to(sliced[None, :], (b_len, sliced.size)).copy()

    return _cb

  return flint_types.FlintBatch(
      tokens=jax.make_array_from_callback(shape, sharding, _tokens),
      targets=jax.make_array_from_callback(shape, sharding, _targets),
      positions=jax.make_array_from_callback(shape, sharding, _make_template_cb(positions_1d)),
      segment_ids=jax.make_array_from_callback(shape, sharding, _make_template_cb(segment_ids_1d)),
      loss_mask=jax.make_array_from_callback(shape, sharding, _make_template_cb(loss_mask_1d)),
  )


class ArrayRecordC4:
  """Random-access multi-context reader over pre-tokenized C4 ArrayRecords."""

  def __init__(
      self,
      data_dir_or_glob: str,
      seq_len: int,
      *,
      shuffle_seed: int | None = None,
  ):
    """Initializes the C4 ArrayRecord reader and validates record geometry.

    Every row concatenates `seq_len // record_len` consecutive records into one
    full-length causal sequence (`positions=0..S-1`, `segment_ids=1`).

    Args:
      data_dir_or_glob: Directory containing `*.array_record` shards, a single
        `.array_record` file path, or a glob with `*` or `?` in the final path
        component only.
      seq_len: Target sequence length; must be a positive multiple of the
        on-disk record length (`>= 2`).
      shuffle_seed: Non-negative seed for deterministic per-epoch row
        permutation, or `None` for sequential row order.

    Raises:
      FileNotFoundError: If no matching ArrayRecord files exist.
      ValueError: If `seq_len`, `shuffle_seed`, record length, or total record
        count is invalid.
    """
    if seq_len <= 0:
      raise ValueError(f"seq_len must be > 0, got {seq_len}.")
    if shuffle_seed is not None and shuffle_seed < 0:
      raise ValueError(f"shuffle_seed must be >= 0 when set, got {shuffle_seed}.")

    self.paths: tuple[str, ...] = _resolve_array_record_paths(data_dir_or_glob)
    self.seq_len: int = seq_len
    self.shuffle_seed: int | None = shuffle_seed
    self._closed: bool = False
    self._perm_cache: dict[int, np.ndarray] = {}
    self._perm_lock = threading.Lock()
    self._readers: list[array_record_module.ArrayRecordReader] = self._open_readers(self.paths)
    self._reader_locks: list[threading.Lock] = [threading.Lock() for _ in self._readers]
    self._io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_IO_WORKERS)

    try:
      counts = [int(r.num_records()) for r in self._readers]
      self._offsets: list[int] = np.cumsum([0, *counts]).tolist()
      self.total_records: int = int(self._offsets[-1])
      if self.total_records <= 0:
        raise ValueError(f"ArrayRecord files for {data_dir_or_glob!r} contain 0 records.")
      rec0 = parse_row(self._read_raw_record(0))
      self.record_len: int = int(rec0.shape[0])
      if self.record_len < 2:
        raise ValueError(
            "record_len must be >= 2 so each record has at least 1 valid" f" shifted target, got {self.record_len}."
        )
      if seq_len % self.record_len != 0:
        raise ValueError(
            f"seq_len={seq_len} must be a positive multiple of" f" record_len={self.record_len} in {data_dir_or_glob!r}."
        )
      self.chunks_per_row: int = seq_len // self.record_len
      self.num_rows: int = self.total_records // self.chunks_per_row
      if self.num_rows <= 0:
        raise ValueError(
            f"Need at least {self.chunks_per_row} records of"
            f" {self.record_len} tokens for seq_len={seq_len}, got"
            f" {self.total_records}."
        )
      self._positions_1d, self._segment_ids_1d, self._loss_mask_1d = _make_sequence_metadata(
          self.chunks_per_row, self.record_len, segment_per_chunk=False
      )
    except BaseException:
      self.close()
      raise

  def close(self) -> None:
    """Closes all open `ArrayRecordReader` shard handles and the I/O pool."""
    self._closed = True
    self._io_pool.shutdown(wait=True, cancel_futures=True)
    for lock, reader in zip(self._reader_locks, self._readers, strict=True):
      with lock:
        if reader.is_open():
          reader.close()

  def __enter__(self) -> ArrayRecordC4:
    """Enters context manager, returning `self`."""
    if self._closed:
      raise ValueError("Cannot enter a closed ArrayRecordC4 instance.")
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    """Closes all shard handles on context exit."""
    del exc_type, exc_val, exc_tb
    self.close()

  def read_row_chunks(self, global_row_idx: int) -> np.ndarray:
    """Reads `int32[chunks_per_row, record_len]` chunks for `global_row_idx`.

    Args:
      global_row_idx: Non-negative global logical row index (wraps across epochs
        and applies per-epoch permutation when `shuffle_seed` is set).

    Returns:
      `int32` array of shape `(chunks_per_row, record_len)`.

    Raises:
      ValueError: If the reader is closed, `global_row_idx < 0`, or any record
        fails decoding or has a length different from `self.record_len`.
    """
    physical_row = self._resolve_row_index(global_row_idx)
    base_rec = physical_row * self.chunks_per_row
    chunks = []
    for chunk_offset in range(self.chunks_per_row):
      rec_idx = base_rec + chunk_offset
      tokens = parse_row(self._read_raw_record(rec_idx))
      if tokens.shape != (self.record_len,):
        raise ValueError(f"Record {rec_idx} has shape {tokens.shape}, expected" f" ({self.record_len},).")
      chunks.append(tokens)
    return np.stack(chunks, axis=0)

  def make_batch_np(
      self,
      step: int,
      global_batch_size: int,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns full host NumPy `(tokens, targets, positions, segment_ids, loss_mask)`."""
    if step < 0:
      raise ValueError(f"step must be >= 0, got {step}.")
    if global_batch_size < 1:
      raise ValueError(f"global_batch_size must be >= 1, got {global_batch_size}.")
    base_row = step * global_batch_size
    chunks = np.stack(
        [self.read_row_chunks(base_row + b) for b in range(global_batch_size)],
        axis=0,
    )
    b_shape = (global_batch_size, self.seq_len)
    tokens = chunks.reshape(b_shape).copy()
    if int(tokens.min()) < 0:
      raise ValueError(f"Negative token ID {int(tokens.min())} in C4 batch at step={step}.")
    targets = _shift_chunks_left(chunks).reshape(b_shape).copy()
    positions = np.broadcast_to(self._positions_1d[None, :], b_shape).copy()
    segment_ids = np.broadcast_to(self._segment_ids_1d[None, :], b_shape).copy()
    loss_mask = np.broadcast_to(self._loss_mask_1d[None, :], b_shape).copy()
    return tokens, targets, positions, segment_ids, loss_mask

  def prefetch_host_rows(
      self,
      step: int,
      global_batch_size: int,
      sharding: jax.sharding.Sharding,
  ) -> dict[int, np.ndarray]:
    """Prefetches all host-local `int32[chunks_per_row, record_len]` rows in parallel."""
    if step < 0:
      raise ValueError(f"step must be >= 0, got {step}.")
    if global_batch_size < 1:
      raise ValueError(f"global_batch_size must be >= 1, got {global_batch_size}.")
    if self._closed:
      raise ValueError("ArrayRecordC4 reader is closed.")

    needed_rows_set: set[int] = set()
    shape = (global_batch_size, self.seq_len)
    for dev_index in sharding.addressable_devices_indices_map(shape).values():
      b_slice, _ = _normalize_index(dev_index)
      needed_rows_set.update(range(*b_slice.indices(global_batch_size)))
    needed_rows = sorted(needed_rows_set)
    if not needed_rows:
      return {}

    base_row = step * global_batch_size
    row_base_recs: dict[int, int] = {}
    by_file: dict[int, list[tuple[int, int]]] = {}
    seen_recs: set[int] = set()
    for b in needed_rows:
      phys_row = self._resolve_row_index(base_row + b)
      base_rec = phys_row * self.chunks_per_row
      row_base_recs[b] = base_rec
      for chunk_offset in range(self.chunks_per_row):
        rec_idx = base_rec + chunk_offset
        if rec_idx in seen_recs:
          continue
        seen_recs.add(rec_idx)
        file_idx = bisect.bisect_right(self._offsets, rec_idx) - 1
        local_idx = rec_idx - self._offsets[file_idx]
        by_file.setdefault(file_idx, []).append((rec_idx, local_idx))

    def _fetch_file(
        item: tuple[int, list[tuple[int, int]]],
    ) -> list[tuple[int, np.ndarray]]:
      file_idx, pairs = item
      with self._reader_locks[file_idx]:
        if self._closed:
          raise ValueError("ArrayRecordC4 reader is closed.")
        raw_records = self._readers[file_idx].read([li for _, li in pairs])
      parsed: list[tuple[int, np.ndarray]] = []
      for (rec_idx, _), raw in zip(pairs, raw_records, strict=True):
        tokens = parse_row(raw)
        if tokens.shape != (self.record_len,):
          raise ValueError(f"Record {rec_idx} has shape {tokens.shape}, expected" f" ({self.record_len},).")
        parsed.append((rec_idx, tokens))
      return parsed

    rec_tokens: dict[int, np.ndarray] = {}
    futs = [self._io_pool.submit(_fetch_file, item) for item in by_file.items()]
    for fut in futs:
      for rec_idx, tokens in fut.result():
        rec_tokens[rec_idx] = tokens

    return {
        b: np.stack(
            [rec_tokens[row_base_recs[b] + c] for c in range(self.chunks_per_row)],
            axis=0,
        )
        for b in needed_rows
    }

  def make_batch_from_prefetched(
      self,
      step: int,
      prefetched_rows: dict[int, np.ndarray],
      global_batch_size: int,
      sharding: jax.sharding.Sharding,
  ) -> flint_types.FlintBatch:
    """Materializes a host-local sharded `FlintBatch` from `prefetch_host_rows`."""
    if step < 0:
      raise ValueError(f"step must be >= 0, got {step}.")
    if global_batch_size < 1:
      raise ValueError(f"global_batch_size must be >= 1, got {global_batch_size}.")

    def _lookup_row(b: int) -> np.ndarray:
      if b not in prefetched_rows:
        raise ValueError(f"Missing prefetched host row b={b} at step={step} (have" f" {sorted(prefetched_rows)}).")
      return prefetched_rows[b]

    return _make_sharded_flint_batch(
        global_batch_size=global_batch_size,
        seq_len=self.seq_len,
        num_chunks=self.chunks_per_row,
        chunk_len=self.record_len,
        positions_1d=self._positions_1d,
        segment_ids_1d=self._segment_ids_1d,
        loss_mask_1d=self._loss_mask_1d,
        sharding=sharding,
        row_chunks_fn=_lookup_row,
    )

  def make_batch(
      self,
      step: int,
      global_batch_size: int,
      sharding: jax.sharding.Sharding,
  ) -> flint_types.FlintBatch:
    """Returns a host-local sharded `FlintBatch` for `step`."""
    prefetched = self.prefetch_host_rows(step, global_batch_size, sharding)
    return self.make_batch_from_prefetched(step, prefetched, global_batch_size, sharding)

  def _open_readers(self, paths: tuple[str, ...]) -> list[array_record_module.ArrayRecordReader]:
    """Opens shard readers concurrently, closing partial handles on failure."""
    opened: list[array_record_module.ArrayRecordReader] = []
    lock = threading.Lock()

    def _open_one(path: str) -> array_record_module.ArrayRecordReader:
      reader = array_record_module.ArrayRecordReader(path)
      with lock:
        opened.append(reader)
      return reader

    try:
      max_workers = min(_MAX_OPEN_WORKERS, len(paths))
      with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_open_one, paths))
    except BaseException:
      for reader in opened:
        if reader.is_open():
          reader.close()
      raise

  def _read_raw_record(self, rec_idx: int) -> bytes:
    if self._closed:
      raise ValueError("ArrayRecordC4 reader is closed.")
    file_idx = bisect.bisect_right(self._offsets, rec_idx) - 1
    local_idx = rec_idx - self._offsets[file_idx]
    with self._reader_locks[file_idx]:
      if self._closed:
        raise ValueError("ArrayRecordC4 reader is closed.")
      return self._readers[file_idx].read([local_idx])[0]

  def _get_epoch_permutation(self, epoch: int) -> np.ndarray:
    """Returns cached `int32[num_rows]` deterministic permutation for `epoch`."""
    assert self.shuffle_seed is not None
    with self._perm_lock:
      cached = self._perm_cache.get(epoch)
    if cached is not None:
      return cached
    perm = np.random.default_rng((self.shuffle_seed, epoch)).permutation(self.num_rows).astype(np.int32)
    with self._perm_lock:
      if epoch not in self._perm_cache:
        if len(self._perm_cache) >= _MAX_CACHED_EPOCHS:
          oldest_epoch = next(iter(self._perm_cache))
          del self._perm_cache[oldest_epoch]
        self._perm_cache[epoch] = perm
      return self._perm_cache[epoch]

  def _resolve_row_index(self, global_row_idx: int) -> int:
    if global_row_idx < 0:
      raise ValueError(f"global_row_idx must be >= 0, got {global_row_idx}.")
    row_in_epoch = global_row_idx % self.num_rows
    if self.shuffle_seed is None:
      return row_in_epoch
    epoch = global_row_idx // self.num_rows
    perm = self._get_epoch_permutation(epoch)
    return int(perm[row_in_epoch])


def make_synthetic_batch(
    step: int,
    global_batch_size: int,
    seq_len: int,
    vocab_size: int,
    sharding: jax.sharding.Sharding,
    *,
    seed: int = 0,
    pack_num_segments: int = 1,
) -> flint_types.FlintBatch:
  """Builds a deterministic mesh-invariant synthetic `FlintBatch`.

  Generates random `int32` tokens in `[0, vocab_size)` using per-row RNG seed
  `(seed, step, b)` so global batch contents are bit-for-bit invariant across
  any `(fsdp, cp, ep)` mesh sharding.

  Args:
    step: Non-negative step index.
    global_batch_size: Global batch size (`>= 1`).
    seq_len: Sequence length; must be divisible by `pack_num_segments` with each
      segment of length `>= 2`.
    vocab_size: Vocabulary upper bound (`>= 1`).
    sharding: Target 2D `jax.sharding.Sharding` for `[batch, seq]`.
    seed: Non-negative base random seed.
    pack_num_segments: Number of packed segments per sequence (`1` yields
      continuous `positions=0..S-1` and `segment_ids=1`; `> 1` yields tiled
      positions and `segment_ids=1..pack_num_segments`).

  Returns:
    Sharded `FlintBatch(tokens, targets, positions, segment_ids, loss_mask)`.

  Raises:
    ValueError: If any dimension, `step`, `seed`, or segment count is invalid.
  """
  if step < 0:
    raise ValueError(f"step must be >= 0, got {step}.")
  if global_batch_size < 1:
    raise ValueError(f"global_batch_size must be >= 1, got {global_batch_size}.")
  if vocab_size < 1:
    raise ValueError(f"vocab_size must be >= 1, got {vocab_size}.")
  if seed < 0:
    raise ValueError(f"seed must be >= 0, got {seed}.")
  if pack_num_segments < 1 or seq_len % pack_num_segments != 0:
    raise ValueError(f"seq_len={seq_len} must be divisible by" f" pack_num_segments={pack_num_segments} (>= 1).")
  seg_len = seq_len // pack_num_segments
  if seg_len < 2:
    raise ValueError(f"Segment length seq_len // pack_num_segments = {seg_len} must be >= 2.")
  positions_1d, segment_ids_1d, loss_mask_1d = _make_sequence_metadata(
      pack_num_segments, seg_len, segment_per_chunk=pack_num_segments > 1
  )

  def _synth_row_chunks(b: int) -> np.ndarray:
    rng = np.random.default_rng((seed, step, b))
    row = rng.integers(0, vocab_size, size=(seq_len,), dtype=np.int32)
    return row.reshape(pack_num_segments, seg_len)

  return _make_sharded_flint_batch(
      global_batch_size=global_batch_size,
      seq_len=seq_len,
      num_chunks=pack_num_segments,
      chunk_len=seg_len,
      positions_1d=positions_1d,
      segment_ids_1d=segment_ids_1d,
      loss_mask_1d=loss_mask_1d,
      sharding=sharding,
      row_chunks_fn=_synth_row_chunks,
  )
