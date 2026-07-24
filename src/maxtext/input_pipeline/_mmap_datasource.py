"""MMap Indexed Dataset reader for Megatron-LM format (.bin + .idx file pairs).

This module provides random access to pre-tokenized datasets stored in
Megatron-LM's memory-mapped format, integrated with Grain's data pipeline.
"""

import bisect
import dataclasses
import glob as _glob
import logging
import os
import struct

import numpy as np
import grain.python as grain

log = logging.getLogger(__name__)


# Megatron MMap format constants
MMAP_INDEX_MAGIC = b"MMIDIDX\x00\x00"
MMAP_INDEX_MAGIC_LEN = 9
MMAP_INDEX_VERSION = 1
# Fixed header size: magic(9) + version(8) + dtype_code(1) + num_seq(8) + num_doc(8) = 34
MMAP_INDEX_HEADER_SIZE = 34

# dtype code mapping (matches Megatron-Core's indexed_dataset.py)
DTYPE_CODES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}
DTYPE_CODES_INV = {v: k for k, v in DTYPE_CODES.items()}


@dataclasses.dataclass(frozen=True)
class MMapDatasetConfig:
  """Dataset-specific config for the Grain mmap and mmap_npy formats."""

  max_target_length: int
  eod_id: int
  mmap_split_sentences: bool
  blend_cache_dir: str = ""
  blend_index_dir: str = ""
  # mmap_npy-specific parameters
  num_samples: int | None = None
  seed: int = 1234
  split_ratio: str | None = None
  split_index: int = 0


class MMapIndexedDataset:
  """Low-level reader for Megatron-LM MMap indexed datasets.

  Reads the .idx index file to get sequence metadata (sizes, byte offsets),
  then uses os.pread for random access to the .bin data file.

  Args:
      path_prefix: Path prefix for the dataset files. The reader expects
          ``{path_prefix}.idx`` and ``{path_prefix}.bin`` to exist.
  """

  def __init__(self, path_prefix: str):
    self._path_prefix = path_prefix
    self._idx_path = path_prefix + ".idx"
    self._bin_path = path_prefix + ".bin"
    self._dtype = None
    self._sizes = None
    self._pointers = None
    self._doc_idx = None
    self._bin_fd = None
    self._idx_buffer_mmap = None
    self._idx_buffer = None
    self._read_index()
    self._open_bin()

  def _read_index(self):
    """Parse the .idx file header and metadata arrays."""
    if not os.path.exists(self._idx_path):
      raise FileNotFoundError(f"Index file not found: {self._idx_path}")
    if not os.path.exists(self._bin_path):
      raise FileNotFoundError(f"Binary file not found: {self._bin_path}")

    file_size = os.path.getsize(self._idx_path)
    if file_size < MMAP_INDEX_HEADER_SIZE:
      raise ValueError(
          f"Index file {self._idx_path} is too small ({file_size} bytes), "
          f"expected at least {MMAP_INDEX_HEADER_SIZE} bytes for header"
      )

    with open(self._idx_path, "rb") as f:
      magic = f.read(MMAP_INDEX_MAGIC_LEN)
      if magic != MMAP_INDEX_MAGIC:
        raise ValueError(f"Invalid magic bytes in {self._idx_path}: " f"expected {MMAP_INDEX_MAGIC!r}, got {magic!r}")

      version = struct.unpack("<Q", f.read(8))[0]
      if version != MMAP_INDEX_VERSION:
        raise ValueError(f"Unsupported MMap index version: {version} " f"(expected {MMAP_INDEX_VERSION})")

      (dtype_code,) = struct.unpack("<B", f.read(1))
      if dtype_code not in DTYPE_CODES:
        raise ValueError(f"Unknown dtype code: {dtype_code}")
      self._dtype = DTYPE_CODES[dtype_code]

      self._num_sequences = struct.unpack("<Q", f.read(8))[0]
      num_documents_raw = struct.unpack("<Q", f.read(8))[0]

      # Detect doc_idx convention.
      # Convention A (original Megatron-LM): num_documents = number of
      #   documents, doc_idx has (num_documents + 1) entries.
      # Convention B (some Megatron forks): num_documents = len(doc_idx),
      #   doc_idx has exactly num_documents entries.
      # We distinguish by checking which interpretation yields the
      # correct file size.
      body_a = self._num_sequences * 4 + self._num_sequences * 8 + (num_documents_raw + 1) * 8
      body_b = self._num_sequences * 4 + self._num_sequences * 8 + num_documents_raw * 8
      expected_a = MMAP_INDEX_HEADER_SIZE + body_a
      expected_b = MMAP_INDEX_HEADER_SIZE + body_b

      if file_size >= expected_a:
        # Convention A: num_documents is the document count
        self._num_documents = num_documents_raw
        doc_idx_entries = num_documents_raw + 1
        expected_total = expected_a
      elif file_size >= expected_b:
        # Convention B: num_documents is len(doc_idx)
        self._num_documents = num_documents_raw - 1
        doc_idx_entries = num_documents_raw
        expected_total = expected_b
      else:
        raise ValueError(
            f"Index file {self._idx_path} is truncated: "
            f"expected at least {expected_b} bytes "
            f"(num_sequences={self._num_sequences}, "
            f"num_documents_raw={num_documents_raw}), "
            f"got {file_size} bytes"
        )

      sizes_offset = f.tell()
      pointers_offset = sizes_offset + self._num_sequences * 4
      doc_idx_offset = pointers_offset + self._num_sequences * 8

      trailing_bytes = file_size - expected_total
      if trailing_bytes > 0:
        # Megatron-Core's multimodal format appends a uint8
        # sequence_modes array of length num_sequences after doc_idx.
        multimodal_extra = self._num_sequences * 1  # uint8
        if trailing_bytes == multimodal_extra:
          raise ValueError(
              f"Index file {self._idx_path} contains a multimodal "
              f"sequence_modes section ({trailing_bytes} trailing bytes "
              f"= {self._num_sequences} x uint8). Multimodal MMap "
              f"datasets are not supported by this reader."
          )
        raise ValueError(
            f"Index file {self._idx_path} has {trailing_bytes} unexpected "
            f"trailing bytes after the standard index sections "
            f"(expected exactly {expected_total} bytes for "
            f"num_sequences={self._num_sequences}, "
            f"num_documents={self._num_documents}). The file may be "
            f"corrupt or use an unsupported format extension."
        )

    self._idx_buffer_mmap = np.memmap(self._idx_path, mode="r", order="C")
    self._idx_buffer = memoryview(self._idx_buffer_mmap)
    self._sizes = np.frombuffer(self._idx_buffer, dtype=np.int32, count=self._num_sequences, offset=sizes_offset)
    self._pointers = np.frombuffer(self._idx_buffer, dtype=np.int64, count=self._num_sequences, offset=pointers_offset)
    self._doc_idx = np.frombuffer(self._idx_buffer, dtype=np.int64, count=doc_idx_entries, offset=doc_idx_offset)

    # O(1) boundary checks — always run.
    if self._num_documents > 0:
      if self._doc_idx[0] != 0:
        raise ValueError(f"Invalid doc_idx in {self._idx_path}: " f"first entry must be 0, got {self._doc_idx[0]}")
      if self._doc_idx[-1] != self._num_sequences:
        raise ValueError(
            f"Invalid doc_idx in {self._idx_path}: "
            f"last entry must equal num_sequences "
            f"({self._num_sequences}), got {self._doc_idx[-1]}"
        )

    # Extended validation behind MMAP_IDX_FULL_VALIDATION=1 for
    # debugging corrupt .idx files. Disabled by default because
    # the sampled mmap page faults cost ~23s on cold GCS Fuse.
    if os.environ.get("MMAP_IDX_FULL_VALIDATION", "").lower() in ("1", "true", "yes"):
      self._validate_idx_arrays()

  def _validate_idx_arrays(self):
    """Full O(N) validation of .idx arrays. Gated behind MMAP_IDX_FULL_VALIDATION=1."""
    if self._num_sequences > 0:
      neg_mask = self._sizes < 0
      if neg_mask.any():
        bad_pos = np.flatnonzero(neg_mask)[:5].tolist()
        raise ValueError(
            f"Negative sizes in {self._idx_path}: "
            f"sequence(s) {bad_pos} have negative sizes "
            f"{self._sizes[bad_pos].tolist()}"
        )

      neg_mask = self._pointers < 0
      if neg_mask.any():
        bad_pos = np.flatnonzero(neg_mask)[:5].tolist()
        raise ValueError(
            f"Negative pointers in {self._idx_path}: "
            f"sequence(s) {bad_pos} have negative byte offsets "
            f"{self._pointers[bad_pos].tolist()}"
        )

    if self._num_documents > 0 and len(self._doc_idx) > 1:
      a = self._doc_idx[:-1]
      b = self._doc_idx[1:]
      bad = b < a
      if bad.any():
        bad_pos = np.flatnonzero(bad)[:5].tolist()
        raise ValueError(f"Non-monotonic doc_idx in {self._idx_path}: " f"decreases at position(s) {bad_pos}")

    element_size = np.dtype(self._dtype).itemsize
    if element_size > 1 and self._num_sequences > 0:
      misaligned_mask = (self._pointers % element_size) != 0
      if misaligned_mask.any():
        bad_pos = np.flatnonzero(misaligned_mask)[:5].tolist()
        raise ValueError(
            f"Misaligned pointers in {self._idx_path}: "
            f"sequence(s) {bad_pos} have byte offsets "
            f"not aligned to dtype itemsize ({element_size})"
        )

    bin_size = os.path.getsize(self._bin_path)
    if self._num_sequences > 0:
      ends = self._pointers.astype(np.int64) + self._sizes.astype(np.int64) * element_size
      max_end = int(np.max(ends))
      if max_end > bin_size:
        raise ValueError(
            f"Binary file {self._bin_path} is too small ({bin_size} bytes) "
            f"for the indexed data (requirement: {max_end} bytes)"
        )

  def _open_bin(self):
    """Open the .bin data file for direct reads (no mmap).

    Using file I/O instead of np.memmap avoids SIGBUS crashes that occur
    when worker processes mmap large files on some shared filesystems or
    under memory pressure.  We keep only the file descriptor (fd) and use
    ``os.pread`` for thread-safe, positional reads without a shared cursor.
    """
    self._bin_fd = os.open(self._bin_path, os.O_RDONLY)
    self._element_size = np.dtype(self._dtype).itemsize

  @property
  def dtype(self):
    return self._dtype

  @property
  def sizes(self):
    return self._sizes

  @property
  def pointers(self):
    return self._pointers

  @property
  def doc_idx(self):
    return self._doc_idx

  def __len__(self):
    return self._num_sequences

  def _resolve_idx(self, idx):
    """Normalize a possibly-negative index and bounds-check it."""
    if idx < 0:
      idx += self._num_sequences
    if idx < 0 or idx >= self._num_sequences:
      raise IndexError(f"Index {idx} out of range for dataset with " f"{self._num_sequences} sequences")
    return idx

  def get(self, idx, offset=0, length=None):
    """Read a (sub-)sequence by index with optional offset and length.

    Args:
        idx: Sequence index (supports negative indexing).
        offset: Element offset within the sequence (default 0).
        length: Number of elements to read. None means read to end of
            sequence from ``offset``.

    Returns:
        A numpy array of the requested tokens.
    """
    idx = self._resolve_idx(idx)
    size = int(self._sizes[idx])
    if offset < 0 or offset > size:
      raise IndexError(f"Offset {offset} out of range for sequence {idx} " f"with size {size}")
    if length is None:
      length = size - offset
    if length < 0 or offset + length > size:
      raise IndexError(f"offset={offset}, length={length} exceeds sequence {idx} " f"size {size}")
    element_size = self._element_size
    byte_offset = int(self._pointers[idx]) + offset * element_size
    nbytes = length * element_size
    data = os.pread(self._bin_fd, nbytes, byte_offset)
    if len(data) != nbytes:
      raise IOError(
          f"Short read from {self._bin_path}: expected {nbytes} bytes "
          f"at offset {byte_offset}, got {len(data)} "
          f"(sequence {idx}, file may be truncated)"
      )
    return np.frombuffer(data, dtype=self._dtype)

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      start, stop, step = idx.indices(self._num_sequences)
      if step != 1:
        raise ValueError(f"MMapIndexedDataset only supports contiguous slices (step=1), " f"got step={step}")
      return [self.get(i) for i in range(start, stop)]
    return self.get(idx)

  def close(self):
    """Close the .bin file descriptor and idx mmap if open."""
    if self._bin_fd is not None:
      os.close(self._bin_fd)
      self._bin_fd = None
    self._sizes = None
    self._pointers = None
    self._doc_idx = None
    if self._idx_buffer is not None:
      try:
        self._idx_buffer.release()
      except (BufferError, ValueError):
        pass
      self._idx_buffer = None
    if self._idx_buffer_mmap is not None:
      del self._idx_buffer_mmap
      self._idx_buffer_mmap = None

  def __del__(self):
    try:
      self.close()
    except OSError:
      pass

  def __getstate__(self):
    """Support pickling for Grain multi-process workers."""
    return {"path_prefix": self._path_prefix}

  def __setstate__(self, state):
    """Restore from pickle."""
    self.__init__(state["path_prefix"])


class MMapIndexedDataSource(grain.RandomAccessDataSource):
  """Grain-compatible data source wrapping MMapIndexedDataset.

  Each element is returned as a dictionary with a single key containing the
  token IDs as a numpy array, compatible with the existing KeepFeatures ->
  Rekey pipeline path.

  Args:
      path_prefix: Path prefix for the .idx and .bin files.
      feature_name: Key name for the output dictionary. Defaults to "text".
      split_sentences: When True, the data was produced with
          ``--split-sentences`` so each sequence is a single sentence.
          Indexing switches to document-level: ``__len__`` returns the
          number of *documents* and ``__getitem__(d)`` concatenates all
          sentences belonging to document ``d``.  This ensures that
          Grain's shuffle operates on documents (matching Megatron's
          document-level shuffle) rather than individual sentences.
  """

  def __init__(self, path_prefix: str, feature_name: str = "text", split_sentences: bool = False):
    self._path_prefix = path_prefix
    self._feature_name = feature_name
    self._split_sentences = split_sentences
    self._dataset = MMapIndexedDataset(path_prefix)

  def check_eod_presence(self, eod_id: int, mode_label: str):
    """Warn if documents do not appear to end with eod_id.

    Reads only the last token of each checked sequence to minimise IO.
    """
    ds = self._dataset
    num_docs = ds._num_documents if self._split_sentences else ds._num_sequences  # pylint: disable=protected-access
    if num_docs == 0:
      return
    check_count = min(num_docs, 20)
    docs_with_eod = 0
    for i in range(check_count):
      if self._split_sentences:
        seq_start = int(ds.doc_idx[i])
        seq_end = int(ds.doc_idx[i + 1])
        if seq_end <= seq_start:
          continue
        last_seq = seq_end - 1
      else:
        last_seq = i
      seq_size = int(ds.sizes[last_seq])
      if seq_size <= 0:
        continue
      last_token = ds.get(last_seq, offset=seq_size - 1, length=1)
      if int(last_token[0]) == eod_id:
        docs_with_eod += 1
    if docs_with_eod == 0:
      log.warning(
          "None of the first %d documents end with eod_id=%d. "
          "%s does NOT insert EOD tokens — the dataset should "
          "be preprocessed with --append-eod for correct document boundary "
          "detection. Segment IDs and loss masking may be incorrect.",
          check_count,
          eod_id,
          mode_label,
      )

  def __len__(self):
    if self._split_sentences:
      return self._dataset._num_documents  # pylint: disable=protected-access
    return len(self._dataset)

  def __getitem__(self, idx):
    if self._split_sentences:
      # Document-level access: concatenate all sentences in document idx
      num_docs = self._dataset._num_documents
      if idx < 0:
        idx += num_docs
      if idx < 0 or idx >= num_docs:
        raise IndexError(f"Document index {idx} out of range for dataset with " f"{num_docs} documents")
      doc_idx = self._dataset.doc_idx
      seq_start = int(doc_idx[idx])
      seq_end = int(doc_idx[idx + 1])
      parts = [np.array(self._dataset.get(i)) for i in range(seq_start, seq_end)]
      if len(parts) == 0:
        tokens = np.array([], dtype=self._dataset.dtype)
      elif len(parts) == 1:
        tokens = parts[0]
      else:
        tokens = np.concatenate(parts)
    else:
      tokens = np.array(self._dataset.get(idx))
    return {self._feature_name: tokens}

  def doc_token_counts(self):
    """Return per-document token counts without materializing tokens."""
    if self._split_sentences:
      doc_idx = self._dataset.doc_idx
      sizes = self._dataset.sizes
      num_docs = self._dataset._num_documents  # pylint: disable=protected-access
      if num_docs == 0:
        return np.array([], dtype=np.int64)
      starts = doc_idx[:num_docs].astype(np.intp)
      counts = np.add.reduceat(sizes.astype(np.int64), starts)
      return counts
    else:
      return self._dataset.sizes.astype(np.int64)

  def __getstate__(self):
    return {
        "path_prefix": self._path_prefix,
        "feature_name": self._feature_name,
        "split_sentences": self._split_sentences,
    }

  def __setstate__(self, state):
    self.__init__(
        state["path_prefix"],
        state["feature_name"],
        state.get("split_sentences", False),
    )


class MultiShardMMapIndexedDataSource(grain.RandomAccessDataSource):
  """Grain-compatible data source that concatenates multiple MMap shards.

  When a dataset directory contains multiple .bin/.idx file pairs (e.g.,
  different crawl splits), this class presents them as a single logical
  dataset with unified random access indexing.

  Args:
      path_prefixes: List of path prefixes for the .idx and .bin files.
      feature_name: Key name for the output dictionary. Defaults to "text".
      split_sentences: When True, use document-level indexing within each
          shard. See ``MMapIndexedDataSource`` for details.
  """

  def __init__(self, path_prefixes: list[str], feature_name: str = "text", split_sentences: bool = False):
    self._path_prefixes = list(path_prefixes)
    self._feature_name = feature_name
    self._split_sentences = split_sentences
    self._sources = [MMapIndexedDataSource(p, feature_name, split_sentences) for p in self._path_prefixes]
    self._cumulative_sizes = []
    total = 0
    for s in self._sources:
      total += len(s)
      self._cumulative_sizes.append(total)

  def check_eod_presence(self, eod_id: int, mode_label: str):
    """Delegate EOD check to the first shard."""
    self._sources[0].check_eod_presence(eod_id, mode_label)

  def __len__(self):
    return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

  def __getitem__(self, idx):
    if idx < 0:
      idx += len(self)
    if idx < 0 or idx >= len(self):
      raise IndexError(f"Index {idx} out of range for dataset with {len(self)} elements")
    shard_idx = bisect.bisect_right(self._cumulative_sizes, idx)
    local_idx = idx if shard_idx == 0 else idx - self._cumulative_sizes[shard_idx - 1]
    return self._sources[shard_idx][local_idx]

  def doc_token_counts(self):
    """Return per-document token counts across all shards."""
    return np.concatenate([s.doc_token_counts() for s in self._sources])

  def __getstate__(self):
    return {
        "path_prefixes": self._path_prefixes,
        "feature_name": self._feature_name,
        "split_sentences": self._split_sentences,
    }

  def __setstate__(self, state):
    self.__init__(
        state["path_prefixes"],
        state["feature_name"],
        state.get("split_sentences", False),
    )


class MMapSampleIndexDataSource(grain.RandomAccessDataSource):
  """Expose fixed-length windows over a concatenated MMap document stream.

  ``mmap`` is the simple sequential format.  Unlike ``mmap_npy``, it does
  not reproduce Megatron's document/sample/shuffle index ordering; it only
  provides fixed-length windows for the existing Grain pipeline.  The input
  must already contain EOD tokens when document-boundary semantics matter.
  """

  def __init__(self, inner_source, seq_length: int, eod_id: int, drop_last: bool = True):
    if seq_length <= 0:
      raise ValueError(f"seq_length must be positive, got {seq_length}")
    self._inner_source = inner_source
    self._seq_length = seq_length
    self._eod_id = eod_id
    self._drop_last = drop_last
    self._cumulative_tokens = np.cumsum(inner_source.doc_token_counts(), dtype=np.int64)
    total_tokens = int(self._cumulative_tokens[-1]) if len(self._cumulative_tokens) else 0
    self._num_samples = total_tokens // seq_length if drop_last else (total_tokens + seq_length - 1) // seq_length
    inner_source.check_eod_presence(eod_id, "mmap mode")

  def __len__(self):
    return self._num_samples

  def __getitem__(self, idx):
    if idx < 0:
      idx += self._num_samples
    if idx < 0 or idx >= self._num_samples:
      raise IndexError(f"Sample index {idx} out of range for dataset with {self._num_samples} samples")

    result = np.full(self._seq_length, self._eod_id, dtype=np.int32)
    global_offset = idx * self._seq_length
    doc_idx = int(np.searchsorted(self._cumulative_tokens, global_offset, side="right"))
    output_offset = 0

    while output_offset < self._seq_length and doc_idx < len(self._cumulative_tokens):
      doc_start = int(self._cumulative_tokens[doc_idx - 1]) if doc_idx else 0
      offset_in_doc = global_offset - doc_start
      doc_tokens = self._inner_source[doc_idx]["text"]
      copy_length = min(len(doc_tokens) - offset_in_doc, self._seq_length - output_offset)
      if copy_length > 0:
        result[output_offset : output_offset + copy_length] = doc_tokens[offset_in_doc : offset_in_doc + copy_length]
        output_offset += copy_length
        global_offset += copy_length
      doc_idx += 1

    return {"text": result}

  def __getstate__(self):
    return {
        "inner_source": self._inner_source,
        "seq_length": self._seq_length,
        "eod_id": self._eod_id,
        "drop_last": self._drop_last,
    }

  def __setstate__(self, state):
    self.__init__(**state)


def _resolve_bin_prefixes(bin_paths):
  """Resolve one or more bin paths into sorted MMap dataset prefixes.

  Delegates to :func:`._mmap_index_utils.resolve_shard_prefixes` which
  uses ``.idx`` files for discovery.  The ``.idx`` file is always present
  alongside ``.bin`` in valid Megatron datasets.

  Args:
      bin_paths: A string (single path or directory) or list of strings.

  Returns:
      Sorted list of path prefixes (without extension).

  Raises:
      FileNotFoundError: If no shard files can be found.
  """
  from maxtext.input_pipeline._mmap_index_utils import resolve_shard_prefixes  # pylint: disable=import-outside-toplevel

  return resolve_shard_prefixes(bin_paths)


def _discover_npy_indices(npy_dir, expected_hash=None):
  """Discover the three Megatron .npy index files in a directory.

  Looks for files matching ``*-document_index.npy``, ``*-sample_index.npy``,
  and ``*-shuffle_index.npy``.  All three must share the same hash prefix
  (the portion before the first ``-`` suffix).

  Args:
      npy_dir: Path to the directory containing the ``.npy`` files.
      expected_hash: When provided, look only for files with this exact
          hash prefix.  This allows multiple index triplets to coexist
          in the same directory (e.g. from different epoch configurations).

  Returns:
      Tuple of ``(document_index_path, sample_index_path, shuffle_index_path)``.

  Raises:
      FileNotFoundError: If the directory doesn't exist or the required
          files are missing / don't share a common prefix.
  """
  if not os.path.isdir(npy_dir):
    raise FileNotFoundError(f"NPY directory does not exist: {npy_dir}")

  if expected_hash is not None:
    doc = os.path.join(npy_dir, f"{expected_hash}-document_index.npy")
    sample = os.path.join(npy_dir, f"{expected_hash}-sample_index.npy")
    shuffle = os.path.join(npy_dir, f"{expected_hash}-shuffle_index.npy")
    if all(os.path.isfile(f) for f in [doc, sample, shuffle]):
      return doc, sample, shuffle
    raise FileNotFoundError(f"No index files with hash {expected_hash} in {npy_dir}")

  doc_files = sorted(_glob.glob(os.path.join(npy_dir, "*-document_index.npy")))
  sample_files = sorted(_glob.glob(os.path.join(npy_dir, "*-sample_index.npy")))
  shuffle_files = sorted(_glob.glob(os.path.join(npy_dir, "*-shuffle_index.npy")))

  if not doc_files or not sample_files or not shuffle_files:
    raise FileNotFoundError(
        f"Could not find all three index files "
        f"(*-document_index.npy, *-sample_index.npy, *-shuffle_index.npy) "
        f"in directory: {npy_dir}"
    )

  # Extract hash prefixes and find a matching triplet
  def _prefix(path, suffix):
    base = os.path.basename(path)
    return base[: -len(suffix)]

  doc_prefixes = {_prefix(f, "-document_index.npy"): f for f in doc_files}
  sample_prefixes = {_prefix(f, "-sample_index.npy"): f for f in sample_files}
  shuffle_prefixes = {_prefix(f, "-shuffle_index.npy"): f for f in shuffle_files}

  common = set(doc_prefixes) & set(sample_prefixes) & set(shuffle_prefixes)
  if not common:
    raise FileNotFoundError(f"No matching hash prefix found among the three index file types " f"in directory: {npy_dir}")

  if len(common) > 1:
    raise ValueError(
        f"Ambiguous NPY directory: found {len(common)} index triplets with "
        f"prefixes {sorted(common)} in {npy_dir}. Remove stale index files "
        f"so that exactly one triplet remains, or pass expected_hash to "
        f"select a specific triplet."
    )

  prefix = sorted(common)[0]
  return (
      doc_prefixes[prefix],
      sample_prefixes[prefix],
      shuffle_prefixes[prefix],
  )


# ---------------------------------------------------------------------------
# MegatronNpyDataSource
# ---------------------------------------------------------------------------


class MegatronNpyDataSource(grain.RandomAccessDataSource):
  """Grain-compatible data source that uses pre-built Megatron .npy indices.

  Loads ``document_index.npy``, ``sample_index.npy``, and
  ``shuffle_index.npy`` (as produced by :func:`tools.data_processing.mmap_index_builder.convert`)
  together with one or more MMap ``.bin/.idx`` dataset shards to provide
  random access to pre-shuffled, fixed-length training samples.

  Args:
      npy_dir: Directory containing the three ``*-{document,sample,shuffle}_index.npy`` files.
      bin_paths: Path prefix, directory, or list thereof pointing to the
          Megatron ``.bin/.idx`` data files.
      eod_id: End-of-document token ID already present in data produced with
          Megatron preprocessing. The data source validates but never inserts it.
      seq_length: Maximum number of tokens per sample.  The output is
          truncated to ``seq_length + 1`` tokens (matching Megatron-LM's
          convention where the extra token is used for next-token prediction).
      split_sentences: Passed through to the underlying ``MMapIndexedDataSource``
          (default ``False``).
  """

  def __init__(
      self,
      npy_dir: str,
      bin_paths: str | list[str],
      eod_id: int,
      seq_length: int,
      split_sentences: bool = False,
      expected_hash: str | None = None,
      prebuilt_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
  ):
    self._npy_dir = npy_dir
    self._bin_paths = bin_paths
    self._eod_id = eod_id
    self._seq_length = seq_length
    self._split_sentences = split_sentences
    self._expected_hash = expected_hash
    self._prebuilt_indices = prebuilt_indices

    if prebuilt_indices is not None:
      self._document_index, self._sample_index, self._shuffle_index = prebuilt_indices
      index_label = "<in-memory>"
    else:
      doc_path, sample_path, shuffle_path = _discover_npy_indices(npy_dir, expected_hash=expected_hash)
      self._document_index = np.load(doc_path, allow_pickle=False, mmap_mode="r")
      self._sample_index = np.load(sample_path, allow_pickle=False, mmap_mode="r")
      self._shuffle_index = np.load(shuffle_path, allow_pickle=False, mmap_mode="r")
      index_label = npy_dir

    prefixes = _resolve_bin_prefixes(bin_paths)
    if len(prefixes) == 1:
      self._token_source = MMapIndexedDataSource(prefixes[0], split_sentences=split_sentences)
    else:
      self._token_source = MultiShardMMapIndexedDataSource(prefixes, split_sentences=split_sentences)

    self._check_split_sentences_consistency(split_sentences)

    self._validate_indices(index_label)
    self._token_source.check_eod_presence(self._eod_id, "mmap_npy mode")

  def _check_split_sentences_consistency(self, split_sentences: bool):
    """Detect split_sentences misconfiguration against the actual data.

    When the data was preprocessed with --split-sentences, num_sequences >
    num_documents in the .idx file.  If split_sentences=False is used at
    runtime, _token_source indexes by sequence instead of document, causing
    document_index IDs to map to wrong data silently.
    """
    if split_sentences:
      return  # Document-level indexing — always correct

    # Collect (num_documents, num_sequences) from each underlying shard
    if isinstance(self._token_source, MultiShardMMapIndexedDataSource):
      sources = self._token_source._sources  # pylint: disable=protected-access
    else:
      sources = [self._token_source]

    for src in sources:
      ds = src._dataset  # pylint: disable=protected-access
      if ds._num_documents != ds._num_sequences:  # pylint: disable=protected-access
        raise ValueError(
            f"split_sentences=False but data at '{ds._path_prefix}' has "  # pylint: disable=protected-access
            f"{ds._num_documents} documents != {ds._num_sequences} sequences, "  # pylint: disable=protected-access
            f"indicating it was preprocessed with --split-sentences. "
            f"Set mmap_split_sentences=true to match the data format."
        )

  def _validate_indices(self, index_label: str):
    """Validate precomputed index files and fail fast with Python errors.

    Default lightweight mode checks shape/dtype + head/tail elements only.
    Set MMAP_NPY_FULL_VALIDATION=1 for the original O(N) scan.
    """
    if self._sample_index.ndim != 2 or self._sample_index.shape[1] != 2:
      raise ValueError(f"Invalid sample_index shape {self._sample_index.shape} in {index_label}; expected (N, 2).")
    if self._sample_index.shape[0] < 2:
      raise ValueError(f"sample_index in {index_label} has too few rows ({self._sample_index.shape[0]}).")
    if self._document_index.ndim != 1:
      raise ValueError(f"Invalid document_index shape {self._document_index.shape} in {index_label}; expected 1-D.")
    if self._shuffle_index.ndim != 1:
      raise ValueError(f"Invalid shuffle_index shape {self._shuffle_index.shape} in {index_label}; expected 1-D.")
    if len(self._shuffle_index) == 0:
      raise ValueError(f"shuffle_index in {index_label} is empty.")

    _full_validation = os.environ.get("MMAP_NPY_FULL_VALIDATION", "").lower() in ("1", "true", "yes")
    sample_upper = int(self._sample_index.shape[0] - 1)
    max_doc_upper = int(self._document_index.shape[0] - 1)
    num_total_docs = len(self._token_source)

    if _full_validation:
      min_sample_id = int(np.min(self._shuffle_index))
      max_sample_id = int(np.max(self._shuffle_index))
      if min_sample_id < 0 or max_sample_id > sample_upper - 1:
        raise ValueError(
            f"shuffle_index in {index_label} references sample_id range [{min_sample_id}, {max_sample_id}], "
            f"but valid range is [0, {sample_upper - 1}]."
        )

      min_doc_pos = int(np.min(self._sample_index[:, 0]))
      max_doc_pos = int(np.max(self._sample_index[:, 0]))
      if min_doc_pos < 0 or max_doc_pos > max_doc_upper:
        raise ValueError(
            f"sample_index in {index_label} references doc_pos range [{min_doc_pos}, {max_doc_pos}], "
            f"but valid document_index range is [0, {max_doc_upper}]."
        )

      min_doc_id = int(np.min(self._document_index))
      max_doc_id = int(np.max(self._document_index))
      if min_doc_id < 0 or max_doc_id >= num_total_docs:
        raise ValueError(
            f"document_index in {index_label} references doc_id range [{min_doc_id}, {max_doc_id}], "
            f"but token source only has {num_total_docs} documents (valid range [0, {num_total_docs - 1}])."
        )

      min_token_offset = int(np.min(self._sample_index[:, 1]))
      if min_token_offset < 0:
        raise ValueError(f"sample_index in {index_label} contains negative token offset {min_token_offset}.")
      return

    for sid in (int(self._shuffle_index[0]), int(self._shuffle_index[-1])):
      if sid < 0 or sid > sample_upper - 1:
        raise ValueError(
            f"shuffle_index in {index_label} contains sample_id {sid} outside valid range [0, {sample_upper - 1}]."
        )

    for label, row in (("first", self._sample_index[0]), ("last", self._sample_index[-1])):
      doc_pos, tok_off = int(row[0]), int(row[1])
      if doc_pos < 0 or doc_pos > max_doc_upper:
        raise ValueError(
            f"sample_index[{label}] in {index_label} references doc_pos {doc_pos} outside valid range "
            f"[0, {max_doc_upper}]."
        )
      if tok_off < 0:
        raise ValueError(f"sample_index[{label}] in {index_label} has negative token offset {tok_off}.")

    for did in (int(self._document_index[0]), int(self._document_index[-1])):
      if did < 0 or did >= num_total_docs:
        raise ValueError(
            f"document_index in {index_label} references doc_id {did} outside valid range " f"[0, {num_total_docs - 1}]."
        )

  def __len__(self):
    return len(self._shuffle_index)

  def __getitem__(self, idx):
    if idx < 0:
      idx += len(self._shuffle_index)
    if idx < 0 or idx >= len(self._shuffle_index):
      raise IndexError(f"Index {idx} out of range for dataset with " f"{len(self._shuffle_index)} samples")
    sample_idx = int(self._shuffle_index[idx])
    doc_idx_beg = self._sample_index[sample_idx]  # shape (2,): [doc_pos, token_offset]
    doc_idx_end = self._sample_index[sample_idx + 1]
    tokens = self._build_sample(doc_idx_beg, doc_idx_end)
    # Hardcoded +1: Megatron-LM's GPTDatasetConfig.add_extra_token_to_sequence
    # defaults to True and no production pretrain entry point (pretrain_gpt.py,
    # pretrain_mamba.py, etc.) ever overrides it — the field is not exposed as
    # a command-line argument.  The only False usage is in unit tests.
    # With add_extra_token=1, build_sample_index produces samples spanning
    # seq_length+1 raw document tokens.  The pipeline's
    # MegatronSplitInputsTargets then splits: inputs = tokens[:-1],
    # targets = tokens[1:], giving valid prediction targets at every
    # position without padding.
    target_len = self._seq_length + 1
    if len(tokens) > target_len:
      tokens = tokens[:target_len]
    elif len(tokens) < target_len:
      tokens = np.pad(tokens, (0, target_len - len(tokens)), constant_values=self._eod_id)
    return {"text": tokens}

  def _build_sample(self, doc_idx_beg, doc_idx_end):
    """Construct a token sample from pre-built index boundaries.

    Replicates Megatron-LM's ``GPTDataset._query_document_sample_shuffle_indices``:
    iterates over document positions from ``doc_idx_beg`` to ``doc_idx_end``,
    slicing the appropriate token ranges and concatenating them directly.

    No explicit EOD insertion is performed — Megatron relies on EOD tokens
    being present in the raw data (via ``--append-eod`` during preprocessing)
    rather than inserting them at read time.

    Args:
        doc_idx_beg: Array of ``[doc_pos_beg, offset_beg]``.
        doc_idx_end: Array of ``[doc_pos_end, offset_end]``.

    Returns:
        1-D numpy array of int32 tokens.
    """
    doc_pos_beg = int(doc_idx_beg[0])
    offset_beg = int(doc_idx_beg[1])
    doc_pos_end = int(doc_idx_end[0])
    offset_end = int(doc_idx_end[1])

    parts = []
    for doc_pos in range(doc_pos_beg, doc_pos_end + 1):
      doc_id = int(self._document_index[doc_pos])
      doc_tokens = self._token_source[doc_id]["text"]

      # Determine the slice within this document
      start = offset_beg if doc_pos == doc_pos_beg else 0
      # +1 on the last document to include the extra token for next-token
      # prediction (add_extra_token_to_sequence is always 1 in production
      # Megatron pretrain configs — see comment in __getitem__ above).
      end = (offset_end + 1) if doc_pos == doc_pos_end else len(doc_tokens)

      parts.append(doc_tokens[start:end].astype(np.int32))

    return np.concatenate(parts) if parts else np.array([], dtype=np.int32)

  # -- Pickle support for Grain multi-process workers ----------------------

  def __getstate__(self):
    return {
        "npy_dir": self._npy_dir,
        "bin_paths": self._bin_paths,
        "eod_id": self._eod_id,
        "seq_length": self._seq_length,
        "split_sentences": self._split_sentences,
        "expected_hash": self._expected_hash,
        "prebuilt_indices": self._prebuilt_indices,
    }

  def __setstate__(self, state):
    self.__init__(**state)


# ---------------------------------------------------------------------------
# Dataset factory functions
# ---------------------------------------------------------------------------


def _parse_weighted_mixture(data_file_pattern, format_name):
  """Parse 'spec,weight;spec,weight;...' into ([specs], [normalized_weights]).

  Uses ``rfind(",")`` so the spec part itself may contain commas
  (e.g. mmap_npy specs with ``npy_dir|bin1:bin2``).

  Returns:
      Tuple of (specs, weights) with zero-weight entries removed and
      weights normalized to sum to 1.

  Raises:
      ValueError: On malformed entries, negative weights, or all-zero weights.
  """
  parts = data_file_pattern.split(";")
  parsed = []
  for i, part in enumerate(parts):
    last_comma = part.rfind(",")
    if last_comma == -1:
      raise ValueError(f"Malformed {format_name} mixture entry at position {i}: " f"expected 'spec,weight', got '{part}'")
    spec_str = part[:last_comma].strip()
    if not spec_str:
      raise ValueError(f"Empty spec in {format_name} mixture entry at position {i}")
    try:
      w = float(part[last_comma + 1 :])
    except ValueError as exc:
      raise ValueError(
          f"Invalid weight in {format_name} mixture entry at position {i}: "
          f"'{part[last_comma + 1:]}' is not a valid number"
      ) from exc
    if w < 0:
      raise ValueError(f"Negative weight ({w}) in {format_name} mixture entry at position {i}")
    parsed.append((spec_str, w))

  specs = [s for s, _ in parsed]
  weights = [w for _, w in parsed]
  total = sum(weights)
  if total <= 0:
    raise ValueError(f"Total weight of {format_name} mixture is {total}; must be positive")
  # Normalize with numpy float64, matching Megatron's normalize() in utils.py:
  #   w = numpy.array(weights, dtype=numpy.float64); w = (w / w.sum()).tolist()
  # Using Python float sum()/division would introduce a ~2e-16 discrepancy
  # that cascades into ~35K tie-breaking differences over 488M blend steps.
  w_arr = np.asarray(weights, dtype=np.float64)
  weights = (w_arr / np.sum(w_arr)).tolist()

  filtered = [(s, w) for s, w in zip(specs, weights) if w > 0]
  if not filtered:
    raise ValueError(f"All {format_name} mixture components have zero weight")
  return [s for s, _ in filtered], [w for _, w in filtered]


def _parse_mmap_npy_spec(spec):
  """Parse ``'npy_dir|bin_dir1:bin_dir2:...'`` into ``(npy_dir, [bin_paths])``."""
  parts = spec.strip().split("|")
  if len(parts) != 2:
    raise ValueError(f"mmap_npy spec must be 'npy_dir|bin_paths', got: {spec!r}")
  npy_dir = parts[0].strip()
  bin_paths = [p.strip() for p in parts[1].split(":")]
  return npy_dir, bin_paths


def create_mmap_source(path_prefix, split_sentences, seq_length, eod_id):
  """Create a Grain map dataset for a simple MMap path or shard directory."""
  prefixes = _resolve_bin_prefixes(path_prefix.strip())
  if len(prefixes) == 1:
    source = MMapIndexedDataSource(prefixes[0], split_sentences=split_sentences)
  else:
    source = MultiShardMMapIndexedDataSource(prefixes, split_sentences=split_sentences)
  if seq_length and eod_id is not None:
    source = MMapSampleIndexDataSource(source, seq_length=seq_length, eod_id=eod_id)
  return grain.MapDataset.source(source)


def get_mmap_dataset(
    data_file_pattern,
    split_sentences,
    seq_length,
    eod_id,
    shuffle,
    shuffle_seed,
    num_epoch,
    host_index,
    host_count,
    num_threads,
    prefetch_buffer_size,
    apply_transforms,
):
  """Build the simple ``mmap`` Grain pipeline, including weighted mixtures."""
  if ";" in data_file_pattern:
    prefixes, weights = _parse_weighted_mixture(data_file_pattern, "mmap")
    datasets = [create_mmap_source(prefix, split_sentences, seq_length, eod_id) for prefix in prefixes]
    iter_datasets = [
        apply_transforms(
            dataset,
            shuffle,
            shuffle_seed,
            num_epoch,
            host_index,
            host_count,
            num_threads,
            prefetch_buffer_size,
        )
        for dataset in datasets
    ]
    return grain.IterDataset.mix(iter_datasets, weights)

  dataset = create_mmap_source(data_file_pattern, split_sentences, seq_length, eod_id)
  return apply_transforms(
      dataset,
      shuffle,
      shuffle_seed,
      num_epoch,
      host_index,
      host_count,
      num_threads,
      prefetch_buffer_size,
  )


def _ensure_npy_indices(
    npy_dir, bin_paths, num_samples, seq_length, seed=1234, split=None, split_index=0, add_extra_token=1, num_epoch=1
):
  """Check for cached npy indices and build them if missing.

  Uses ``(num_epochs, separate_final_epoch)`` as the cache key — different
  ``num_samples`` values that map to the same epoch bucket share the same
  index files.

  In multi-host training every host calls this function independently.
  On a cache hit all hosts read from disk.  On a cache miss every host
  computes the (deterministic) indices in memory and only host 0 writes
  to the shared cache directory — no barrier or locking is needed.

  Returns:
      Tuple of ``(expected_hash, prebuilt_indices)`` where
      *prebuilt_indices* is ``(doc_index, sample_index, shuffle_index)``
      on a cache miss, or ``None`` on a cache hit (caller should read
      from disk).
  """
  from maxtext.input_pipeline import _mmap_index_utils  # pylint: disable=import-outside-toplevel

  all_prefixes = []
  for bp in bin_paths if isinstance(bin_paths, list) else [bin_paths]:
    all_prefixes.extend(_mmap_index_utils.resolve_shard_prefixes(bp))
  all_prefixes = sorted(all_prefixes)

  if split is not None:
    num_docs = _mmap_index_utils.get_num_documents(all_prefixes)
    start_doc, end_doc = _mmap_index_utils.parse_split_range(split, split_index, num_docs)
    tokens_per_epoch = _mmap_index_utils.get_split_tokens(all_prefixes, start_doc, end_doc)
  else:
    tokens_per_epoch = _mmap_index_utils.get_total_tokens(all_prefixes)
  if num_samples is None:
    samples_per_epoch = (tokens_per_epoch - add_extra_token) // seq_length
    num_samples = samples_per_epoch * max(1, num_epoch)
  num_epochs = _mmap_index_utils.compute_num_epochs(tokens_per_epoch, num_samples, seq_length, add_extra_token)
  separate = _mmap_index_utils.should_separate_last_epoch(
      num_epochs, tokens_per_epoch, num_samples, seq_length, add_extra_token
  )

  expected_hash = _mmap_index_utils.compute_index_hash(
      input_paths=all_prefixes,
      num_epochs=num_epochs,
      separate_final_epoch=separate,
      seed=seed,
      seq_length=seq_length,
      split=split,
      split_index=split_index,
      add_extra_token=add_extra_token,
  )

  # Check if index files already exist
  try:
    _discover_npy_indices(npy_dir, expected_hash=expected_hash)
    log.info("Cache hit: npy indices with hash %s found in %s", expected_hash, npy_dir)
    return expected_hash, None
  except FileNotFoundError:
    pass

  # Cache miss: build in memory (deterministic — every host gets the same result)
  log.info(
      "Cache miss: building npy indices in memory (num_epochs=%d, separate=%s)",
      num_epochs,
      separate,
  )
  input_list = bin_paths if isinstance(bin_paths, list) else [bin_paths]
  doc_index, sample_index, shuffle_index, _ = _mmap_index_utils.build_indices(
      input_paths=input_list,
      seq_length=seq_length,
      num_samples=num_samples,
      seed=seed,
      split=split,
      split_index=split_index,
      add_extra_token=add_extra_token,
  )

  # Only host 0 persists to disk for future cache hits (atomic write).
  is_primary = _mmap_index_utils.is_primary_process()
  if is_primary:
    log.info("Host 0: writing npy cache to %s", npy_dir)
    metadata = _mmap_index_utils.build_metadata(
        file_hash=expected_hash,
        input_paths=all_prefixes,
        seq_length=seq_length,
        num_samples=num_samples,
        num_epochs=num_epochs,
        seed=seed,
        split=split,
        split_index=split_index,
        add_extra_token=add_extra_token,
        num_docs=doc_index.shape[0],
        total_samples=sample_index.shape[0] - 1,
        source="runtime_auto_build",
    )
    _mmap_index_utils.save_indices_atomic(
        npy_dir, expected_hash, doc_index, sample_index, shuffle_index, metadata=metadata
    )

  return expected_hash, (doc_index, sample_index, shuffle_index)


def create_mmap_npy_source(
    spec, eod_id, seq_length, split_sentences, num_samples=None, seed=1234, split=None, split_index=0, num_epoch=1
):
  """Create a ``grain.MapDataset`` from an ``mmap_npy`` spec string.

  The *spec* format is ``'npy_dir|bin_path1:bin_path2:...'``.

  When *num_samples* is provided, index files are automatically built
  if not already cached in the npy directory.

  Args:
      split: Megatron-style split ratio string (e.g. ``'99,1'``).
          When set, only the document range for *split_index* is used.
      split_index: Which partition to use (0=train, 1=eval, 2=test).
      num_epoch: Number of epochs for auto-index building when
          *num_samples* is not provided (default 1).
  """
  npy_dir, bin_paths = _parse_mmap_npy_spec(spec)

  expected_hash, prebuilt_indices = _ensure_npy_indices(
      npy_dir,
      bin_paths,
      num_samples,
      seq_length,
      seed=seed,
      split=split,
      split_index=split_index,
      num_epoch=num_epoch,
  )

  source = MegatronNpyDataSource(
      npy_dir=npy_dir,
      bin_paths=bin_paths,
      eod_id=eod_id,
      seq_length=seq_length,
      split_sentences=split_sentences,
      expected_hash=expected_hash,
      prebuilt_indices=prebuilt_indices,
  )
  return grain.MapDataset.source(source)


def get_mmap_npy_dataset(
    data_file_pattern,
    split_sentences,
    seq_length,
    eod_id,
    num_epoch,
    host_index,
    host_count,
    num_threads,
    prefetch_buffer_size,
    blend_cache_dir,
    blend_index_dir,
    blend_split,
    apply_transforms,
    num_samples=None,
    seed=1234,
    blend_margin=0.5,
    split=None,
    split_index=0,
):
  """Build an mmap_npy dataset pipeline with optional Megatron blending.

  Single-spec patterns are wrapped via *apply_transforms*.  Multi-spec
  patterns (``';'``-separated) use :class:`MegatronBlendedDataSource` for
  blend-then-shard ordering that matches Megatron-LM.

  When *num_samples* is provided, index files are automatically built
  (or loaded from cache) for each sub-dataset.  For blending, per-dataset
  buffer sizes are computed using Megatron's formula:
  ``buffer_i = ceil(ceil(num_samples * w_i) * (1 + margin/100))``.
  The ``num_epoch`` parameter is ignored when ``num_samples`` is set —
  the epoch count is derived from ``num_samples`` instead.

  Args:
      apply_transforms: Callback that converts a MapDataset into this host's
        iterator, applying repeat, host sharding, and read options.
      num_samples: Total training samples.  When provided, triggers
          auto-rebuild of npy indices if cached files are missing.
      seed: Random seed for index construction (default 1234).
      blend_margin: Overprovisioning margin %% for blend buffers (default 0.5).
      split: Megatron-style split ratio string (e.g. ``'99,1'``).
      split_index: Which partition to use (0=train, 1=eval, 2=test).
  """
  from maxtext.input_pipeline import _mmap_index_utils  # pylint: disable=import-outside-toplevel
  from maxtext.input_pipeline._megatron_blending import MegatronBlendedDataSource  # pylint: disable=import-outside-toplevel  # avoid circular at module level

  # Epoch management is always encoded in the npy indices (either via
  # explicit num_samples or via num_epoch auto-computation), so repeat
  # should not multiply epochs again.
  effective_num_epoch = 1

  if ";" not in data_file_pattern:
    ds = create_mmap_npy_source(
        data_file_pattern,
        eod_id,
        seq_length,
        split_sentences,
        num_samples=num_samples,
        seed=seed,
        split=split,
        split_index=split_index,
        num_epoch=num_epoch,
    )
    return apply_transforms(
        ds,
        False,
        0,
        effective_num_epoch,
        host_index,
        host_count,
        num_threads,
        prefetch_buffer_size,
    )

  specs, weights = _parse_weighted_mixture(data_file_pattern, "mmap_npy")

  # Compute per-dataset buffer sizes for blending
  if num_samples is not None:
    buffer_per_ds = _mmap_index_utils.compute_blend_buffers(num_samples, weights, blend_margin)
  else:
    buffer_per_ds = [None] * len(specs)

  map_datasets = [
      create_mmap_npy_source(
          s,
          eod_id,
          seq_length,
          split_sentences,
          num_samples=buf,
          seed=seed,
          split=split,
          split_index=split_index,
          num_epoch=num_epoch,
      )
      for s, buf in zip(specs, buffer_per_ds)
  ]

  if len(map_datasets) == 1:
    return apply_transforms(
        map_datasets[0],
        False,
        0,
        effective_num_epoch,
        host_index,
        host_count,
        num_threads,
        prefetch_buffer_size,
    )

  # Blend-then-shard: blend on full datasets so that host sharding
  # splits the global blend output, and interleave across hosts
  # reconstructs the exact Megatron sequence.
  for i, ds in enumerate(map_datasets):
    map_datasets[i] = ds.repeat(effective_num_epoch)

  blended_source = MegatronBlendedDataSource(
      map_datasets=map_datasets,
      weights=weights,
      size=num_samples,
      cache_dir=blend_cache_dir,
      blend_index_dir=blend_index_dir,
      split=blend_split,
  )
  dataset = grain.MapDataset.source(blended_source)
  dataset = dataset[host_index::host_count]
  return dataset.to_iter_dataset(
      read_options=grain.ReadOptions(
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size,
      )
  )
