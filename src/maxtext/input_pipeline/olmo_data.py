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

"""Shared utilities for OLMo-core-style numpy FSL datasets.

Dependency-free layer. AI2's mix files describe a virtual concatenation of
flat token-ID arrays; instances are non-overlapping ``sequence_length``-token
windows of that stream. This module builds the index that maps a global
instance index to (file, byte-offset), and ports OLMo-core's repeated-n-gram
filter (``olmo_core/data/utils.py::find_periodic_sequences``).
"""

from __future__ import annotations

import ast
import bisect
import dataclasses
import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Generator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


# Bumped whenever the on-disk index format or fingerprint inputs change.
INDEX_FORMAT_VERSION = "1"


@dataclass(frozen=True)
class OlmoNpyFileEntry:
  """One file in the mix: ``n_tokens // sequence_length`` instances starting
  at global index ``instance_offset``. Trailing tokens are dropped (matches
  OLMo-core)."""

  path: str
  label: str
  n_tokens: int
  n_instances: int
  instance_offset: int


@dataclass
class OlmoNpyIndex:
  """Index over the files in an OLMo data mix. Build via
  :func:`build_index`, persist via :meth:`save`, restore via
  :func:`load_index`. Mutating fields invalidates :attr:`fingerprint`."""

  format_version: str
  sequence_length: int
  dtype: str  # numpy dtype string, e.g. "uint32"
  tokenizer: str  # informational, e.g. "allenai/dolma3-tokenizer"
  files: Tuple[OlmoNpyFileEntry, ...]
  total_instances: int
  total_tokens: int
  fingerprint: str = ""

  # Lazily computed bisect helper: cumulative instance_offsets+sentinel for
  # binary search in ``global_to_local``. Not serialized.
  _instance_offset_starts: Optional[List[int]] = field(default=None, repr=False, compare=False)

  def __post_init__(self):
    starts = [f.instance_offset for f in self.files]
    starts.append(self.total_instances)  # sentinel for bisect
    object.__setattr__(self, "_instance_offset_starts", starts)

  def to_json_dict(self) -> dict:
    """Return a JSON-serializable view (drops cached lookup helpers)."""
    return {
        "format_version": self.format_version,
        "sequence_length": self.sequence_length,
        "dtype": self.dtype,
        "tokenizer": self.tokenizer,
        "total_instances": self.total_instances,
        "total_tokens": self.total_tokens,
        "fingerprint": self.fingerprint,
        "files": [dataclasses.asdict(f) for f in self.files],
    }

  def save(self, path: str) -> None:
    """Write the index as JSON to ``path`` (local filesystem)."""
    with open(path, "w", encoding="utf-8") as fh:
      json.dump(self.to_json_dict(), fh, indent=2)


def load_index(path: str) -> OlmoNpyIndex:
  """Load an index from JSON written by :meth:`OlmoNpyIndex.save`.

  Args:
    path: Local filesystem path to the JSON file.

  Returns:
    The materialized :class:`OlmoNpyIndex`.

  Raises:
    ValueError: If ``format_version`` doesn't match this code's expectation.
  """
  with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
  if data.get("format_version") != INDEX_FORMAT_VERSION:
    raise ValueError(
        f"Index format version mismatch: file has "
        f"{data.get('format_version')!r}, code expects {INDEX_FORMAT_VERSION!r}."
    )
  files = tuple(OlmoNpyFileEntry(**entry) for entry in data["files"])
  return OlmoNpyIndex(
      format_version=data["format_version"],
      sequence_length=data["sequence_length"],
      dtype=data["dtype"],
      tokenizer=data["tokenizer"],
      files=files,
      total_instances=data["total_instances"],
      total_tokens=data["total_tokens"],
      fingerprint=data["fingerprint"],
  )


def global_to_local(index: OlmoNpyIndex, instance_id: int) -> Tuple[int, int]:
  """Global instance index → ``(file_idx, token_offset)``.

  ``token_offset`` is in *tokens* (not bytes); the slice
  ``arr[token_offset : token_offset + sequence_length]`` is the instance.
  """
  if instance_id < 0 or instance_id >= index.total_instances:
    raise IndexError(f"instance_id {instance_id} out of range " f"[0, {index.total_instances})")
  starts = index._instance_offset_starts  # type: ignore[attr-defined]  # pylint: disable=protected-access
  file_idx = bisect.bisect_right(starts, instance_id) - 1
  local_instance = instance_id - index.files[file_idx].instance_offset
  token_offset = local_instance * index.sequence_length
  return file_idx, token_offset


def compute_fingerprint(
    sequence_length: int,
    dtype: str,
    tokenizer: str,
    files: Sequence[OlmoNpyFileEntry],
) -> str:
  """Stable hash over the fields a restart must preserve.

  If any of these change, the global instance ordering changes and resuming
  training from a checkpoint would silently produce different batches.
  """
  h = hashlib.sha256()
  h.update(INDEX_FORMAT_VERSION.encode("utf-8"))
  h.update(b"\x00")
  h.update(str(sequence_length).encode("utf-8"))
  h.update(b"\x00")
  h.update(dtype.encode("utf-8"))
  h.update(b"\x00")
  h.update(tokenizer.encode("utf-8"))
  h.update(b"\x00")
  for f in files:
    h.update(f.path.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(f.n_tokens).encode("utf-8"))
    h.update(b"\x00")
  return f"sha256:{h.hexdigest()}"


_NPY_MAGIC = b"\x93NUMPY"


def parse_npy_header(stream: BinaryIO) -> Tuple[str, Tuple[int, ...]]:
  """Parse a .npy v1/v2/v3 header. Returns ``(dtype_str, shape)``."""
  magic = stream.read(6)
  if magic != _NPY_MAGIC:
    raise ValueError(f"Not a .npy file (magic={magic!r})")
  major = stream.read(1)[0]
  stream.read(1)  # minor version byte — unused
  if major == 1:
    header_len = struct.unpack("<H", stream.read(2))[0]
  elif major in (2, 3):
    header_len = struct.unpack("<I", stream.read(4))[0]
  else:
    raise ValueError(f"Unsupported .npy major version {major}")
  header = stream.read(header_len).decode("latin1")
  # Header is a Python literal dict like
  # "{'descr': '<u4', 'fortran_order': False, 'shape': (123,), }".
  # ``ast.literal_eval`` parses it safely (no arbitrary-code execution).
  parsed = ast.literal_eval(header)
  descr = parsed["descr"]
  shape = tuple(parsed["shape"])
  dtype_str = str(np.dtype(descr))  # canonicalize, e.g. "<u4" -> "uint32"
  return dtype_str, shape


def read_npy_header_from_path(path: str) -> Tuple[str, Tuple[int, ...]]:
  """Convenience wrapper for :func:`parse_npy_header` on a local file."""
  with open(path, "rb") as fh:
    return parse_npy_header(fh)


def read_raw_metadata_from_path(path: str, dtype: str) -> Tuple[str, Tuple[int, ...]]:
  """Headerless raw binary: ``n_tokens = file_size // itemsize``.

  AI2's ``.npy``-extension files are actually raw uint32 dumps, no header;
  olmo-core reads them with ``np.memmap`` and a known dtype.
  """
  itemsize = np.dtype(dtype).itemsize
  size_bytes = os.path.getsize(path)
  if size_bytes % itemsize != 0:
    raise ValueError(
        f"File size {size_bytes} of {path} is not a multiple of itemsize "
        f"{itemsize} for dtype {dtype}; this is unexpected."
    )
  return dtype, (size_bytes // itemsize,)


def has_npy_magic(first_bytes: bytes) -> bool:
  """Quick check: does this look like a real .npy file?"""
  return len(first_bytes) >= 6 and first_bytes[:6] == _NPY_MAGIC


def _file_entry_from_header(
    path: str,
    label: str,
    dtype: str,
    shape: Tuple[int, ...],
    sequence_length: int,
    instance_offset: int,
) -> OlmoNpyFileEntry:
  """Build a file entry from a parsed .npy header (validates shape is 1-D)."""
  if len(shape) != 1:
    raise ValueError(f"Expected 1-D .npy array for {path}, got shape {shape}.")
  n_tokens = int(shape[0])
  n_instances = n_tokens // sequence_length
  return OlmoNpyFileEntry(
      path=path,
      label=label,
      n_tokens=n_tokens,
      n_instances=n_instances,
      instance_offset=instance_offset,
  )


def build_index(
    paths_and_labels: Sequence[Tuple[str, str]],
    sequence_length: int,
    *,
    tokenizer: str,
    header_reader=read_npy_header_from_path,
) -> OlmoNpyIndex:
  """Build an :class:`OlmoNpyIndex` from ``(path, label)`` entries.

  Order matters — global instance ordering is the concatenation in this
  order. ``header_reader`` is the seam tests use to avoid disk; production
  paths pass a GCS-aware reader.
  """
  if sequence_length <= 0:
    raise ValueError(f"sequence_length must be positive, got {sequence_length}")
  if not paths_and_labels:
    raise ValueError("paths_and_labels must be non-empty")

  entries: List[OlmoNpyFileEntry] = []
  observed_dtype: Optional[str] = None
  cum_offset = 0
  for path, label in paths_and_labels:
    dtype, shape = header_reader(path)
    if observed_dtype is None:
      observed_dtype = dtype
    elif dtype != observed_dtype:
      raise ValueError(f"Heterogeneous dtypes across mix files: {observed_dtype!r} " f"and {dtype!r} (at {path}).")
    entry = _file_entry_from_header(
        path=path,
        label=label,
        dtype=dtype,
        shape=shape,
        sequence_length=sequence_length,
        instance_offset=cum_offset,
    )
    entries.append(entry)
    cum_offset += entry.n_instances

  files = tuple(entries)
  total_instances = cum_offset
  total_tokens = sum(f.n_tokens for f in files)

  fingerprint = compute_fingerprint(
      sequence_length=sequence_length,
      dtype=observed_dtype or "",
      tokenizer=tokenizer,
      files=files,
  )

  return OlmoNpyIndex(
      format_version=INDEX_FORMAT_VERSION,
      sequence_length=sequence_length,
      dtype=observed_dtype or "",
      tokenizer=tokenizer,
      files=files,
      total_instances=total_instances,
      total_tokens=total_tokens,
      fingerprint=fingerprint,
  )


class RepetitionTuple(NamedTuple):
  """``arr[start:end]`` is a periodic span of length ``period``,
  ``times = (end - start) // period``."""

  start: int
  end: int
  period: int
  times: int


def _find_end_first_consecutive_true(arr: np.ndarray) -> int:
  """End offset (exclusive) of the leading run of True in ``arr``.

  Returns 0 if ``arr[0]`` is False, ``len(arr)`` if all True.
  """
  if not arr[0]:
    return 0
  prog = np.cumsum(arr)
  if prog[-1] == len(arr):
    return int(len(arr))
  # First index where the cumulative sum stops increasing == start of False run.
  true_locs = np.where(prog[:-1] == prog[1:])[0]
  return int(true_locs[0] + 1)


def _find_start_last_consecutive_true(arr: np.ndarray) -> int:
  """Start offset of the trailing run of True in ``arr``, or -1 if none."""
  reverse = _find_end_first_consecutive_true(arr[::-1])
  return len(arr) - reverse if reverse > 0 else -1


def _group_consecutive_values(arr: np.ndarray, stepsize: int = 1) -> List[np.ndarray]:
  """Split a 1-D array of ints into runs of consecutive values."""
  if len(arr) == 0:
    return []
  return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)


def find_periodic_sequences(
    arr: np.ndarray,
    max_period: int,
    min_period: int = 1,
    mask_value: int = -1,
) -> Generator[RepetitionTuple, None, None]:
  """Yield :class:`RepetitionTuple` for periodic spans of length ≥ 3 in ``arr``.

  ``mask_value`` is reshape padding and must not appear in ``arr``. Default
  -1 is the max uint32 value, above any realistic vocab; pass an
  out-of-vocab sentinel if your vocab hits that id.
  """
  if (arr == mask_value).sum() > 0:
    raise ValueError("`mask_value` is in the array")

  max_period = min(max_period, len(arr) // 3)

  for period in range(min_period, max_period + 1):
    pad = (period - (len(arr) % period)) % period
    padded_arr = np.pad(arr, (0, pad), constant_values=mask_value) if pad else arr
    shaped_arr = padded_arr.reshape(-1, period)

    is_equal_to_prev_row = shaped_arr == np.roll(shaped_arr, shift=1, axis=0)
    rows_with_period = np.where(is_equal_to_prev_row.all(axis=1))[0]
    if len(rows_with_period) == 0:
      continue

    for sequence in _group_consecutive_values(rows_with_period):
      start_row = int(sequence[0])
      end_row = int(sequence[-1])

      start_offset = _find_start_last_consecutive_true(is_equal_to_prev_row[start_row - 1])
      start_offset = period - start_offset if start_offset > 0 else 0

      end_offset = _find_end_first_consecutive_true(is_equal_to_prev_row[(end_row + 1) % shaped_arr.shape[0]])

      start_pos = (start_row - 1) * period - start_offset
      end_pos = ((end_row + 1) * period) + end_offset

      out = RepetitionTuple(
          start=start_pos,
          end=end_pos,
          period=period,
          times=(end_pos - start_pos) // period,
      )
      if out.times > 2:
        yield out


def is_clean_instance(
    input_ids: np.ndarray,
    *,
    repetition_max_period: int = 13,
    repetition_min_period: int = 1,
    repetition_max_count: int = 32,
    mask_value: int = -1,
) -> bool:
  """``False`` iff ``input_ids`` has any periodic span (period ∈
  [min, max]) that repeats ≥ ``repetition_max_count`` times. Defaults
  match OLMo-core's ``_validate_instance``."""
  for m in find_periodic_sequences(
      input_ids,
      max_period=repetition_max_period,
      min_period=repetition_min_period,
      mask_value=mask_value,
  ):
    if m.times >= repetition_max_count:
      return False
  return True
