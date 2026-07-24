"""Core index-building algorithms for Megatron-LM .npy index files.

Functions in this module are used both at offline build time
(``tools/data_processing/mmap_index_builder.py``) and at training-time
auto-rebuild (``_mmap_datasource._ensure_npy_indices``).

All pure-computation helpers are deterministic given the same inputs
and random seed, matching Megatron-LM's ``gpt_dataset.py`` RNG flow.
"""

import functools
import glob as _glob
import hashlib
import json
import logging
import math
import os
import struct

import numpy as np

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

log = logging.getLogger(__name__)


def is_primary_process() -> bool:
  """Return True if this is JAX process 0 (or if JAX is unavailable)."""
  try:
    import jax  # pylint: disable=import-outside-toplevel

    return jax.process_index() == 0
  except (ImportError, RuntimeError):
    return True


def save_npy_atomic(path, array):
  """Write a single numpy array to *path* using atomic rename."""
  path_str = str(path)
  tmp_path = path_str + f".tmp.{os.getpid()}"
  with open(tmp_path, "wb") as writer:
    np.save(writer, array, allow_pickle=False)
  os.replace(tmp_path, path_str)


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------


def resolve_shard_prefixes(paths):
  """Find MMap dataset shard prefixes from paths, directories, or a mix.

  Accepts a single string or a list of strings.  For each path:
  - If ``path + ".idx"`` exists, treat it as a single prefix.
  - Else if the path is a directory, scan it for ``*.idx`` files and
    derive prefixes by stripping the ``.idx`` extension.
  - Otherwise raise ``FileNotFoundError``.

  Results are cached per unique input to avoid repeated GCS Fuse
  directory scans during multi-dataset initialization.

  Returns:
      Sorted list of path prefixes (without ``.idx``/``.bin`` extension).

  Raises:
      FileNotFoundError: If no ``.idx`` files can be found.
  """
  key = (paths,) if isinstance(paths, str) else tuple(paths)
  return list(_resolve_shard_prefixes_cached(key))


@functools.lru_cache(maxsize=64)
def _resolve_shard_prefixes_cached(paths_tuple):
  """Cached implementation of shard prefix resolution."""
  prefixes = []
  for p in paths_tuple:
    if os.path.isfile(p + ".idx"):
      prefixes.append(p)
    elif os.path.isdir(p):
      idx_files = sorted(_glob.glob(os.path.join(p, "*.idx")))
      if not idx_files:
        raise FileNotFoundError(f"No .idx files found in directory: {p}")
      prefixes.extend(f[:-4] for f in idx_files)  # strip .idx
    else:
      raise FileNotFoundError(f"No .idx files found at: {p}")

  if not prefixes:
    raise FileNotFoundError("No .idx files found from the provided paths")

  return sorted(prefixes)


# ---------------------------------------------------------------------------
# .idx header parsing (lightweight — no array reads)
# ---------------------------------------------------------------------------

_IDX_MAGIC = b"MMIDIDX\x00\x00"
# magic(9) + version(8) + dtype_code(1) + num_sequences(8) + num_documents(8)
_IDX_HEADER_SIZE = 34
_IDX_DTYPE_CODES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def _read_idx_header(prefix: str):
  """Read .idx header. Returns (num_sequences, num_documents, numpy dtype, doc_idx_entries).

  Detects convention A (num_documents_raw = doc count, doc_idx has raw+1
  entries) vs convention B (num_documents_raw = len(doc_idx), actual doc
  count = raw-1) by checking the file size.
  """
  idx_path = prefix + ".idx"
  file_size = os.path.getsize(idx_path)
  with open(idx_path, "rb") as f:
    magic = f.read(9)
    if magic != _IDX_MAGIC:
      raise ValueError(f"Invalid magic in {idx_path}")
    f.read(8)  # version
    dtype_code = struct.unpack("<B", f.read(1))[0]
    num_sequences = struct.unpack("<Q", f.read(8))[0]
    num_documents_raw = struct.unpack("<Q", f.read(8))[0]

  body_a = num_sequences * 4 + num_sequences * 8 + (num_documents_raw + 1) * 8
  body_b = num_sequences * 4 + num_sequences * 8 + num_documents_raw * 8

  if file_size >= _IDX_HEADER_SIZE + body_a:
    num_documents = num_documents_raw
    doc_idx_entries = num_documents_raw + 1
  elif file_size >= _IDX_HEADER_SIZE + body_b:
    num_documents = num_documents_raw - 1
    doc_idx_entries = num_documents_raw
  else:
    raise ValueError(f"Index file {idx_path} is truncated")

  return num_sequences, num_documents, _IDX_DTYPE_CODES[dtype_code], doc_idx_entries


# ---------------------------------------------------------------------------
# O(1) token counting (no array reads)
# ---------------------------------------------------------------------------


def get_total_tokens(path_prefixes: list[str]) -> int:
  """Total token count derived from .bin file sizes — O(1) per shard, no array reads."""
  total = 0
  for prefix in path_prefixes:
    _, _, dtype, _ = _read_idx_header(prefix)
    element_size = np.dtype(dtype).itemsize
    total += os.path.getsize(prefix + ".bin") // element_size
  return total


def get_split_tokens(path_prefixes: list[str], start_doc: int, end_doc: int) -> int:
  """Token count for a document range using pointer arithmetic — O(1) per shard.

  Reads only .idx headers + a few bytes via pread to compute the token
  sum for docs [start_doc, end_doc) across concatenated shards, instead
  of reading the full sizes/doc_idx arrays.
  """
  global_doc = 0
  total_tokens = 0

  for prefix in path_prefixes:
    num_seq, num_docs, dtype, _ = _read_idx_header(prefix)
    shard_end = global_doc + num_docs
    if start_doc >= shard_end or end_doc <= global_doc:
      global_doc = shard_end
      continue

    local_start = max(0, start_doc - global_doc)
    local_end = min(num_docs, end_doc - global_doc)
    element_size = np.dtype(dtype).itemsize

    if local_start == 0 and local_end == num_docs:
      total_tokens += os.path.getsize(prefix + ".bin") // element_size
    else:
      total_tokens += _pread_split_tokens(prefix, num_seq, num_docs, element_size, local_start, local_end)

    global_doc = shard_end

  return total_tokens


def _pread_split_tokens(prefix, num_seq, num_docs, element_size, local_start, local_end):
  """Read token count for a doc range within a single shard via pread."""
  idx_path = prefix + ".idx"
  pointers_offset = _IDX_HEADER_SIZE + num_seq * 4
  doc_idx_offset = pointers_offset + num_seq * 8

  fd = os.open(idx_path, os.O_RDONLY)
  try:
    buf = os.pread(fd, 8, doc_idx_offset + local_start * 8)
    seq_start = struct.unpack("<q", buf)[0]

    if local_end >= num_docs:
      seq_end = num_seq
    else:
      buf = os.pread(fd, 8, doc_idx_offset + local_end * 8)
      seq_end = struct.unpack("<q", buf)[0]

    if seq_start == seq_end:
      return 0

    buf = os.pread(fd, 8, pointers_offset + seq_start * 8)
    ptr_start = struct.unpack("<q", buf)[0]

    if seq_end >= num_seq:
      ptr_end = os.path.getsize(prefix + ".bin")
    else:
      buf = os.pread(fd, 8, pointers_offset + seq_end * 8)
      ptr_end = struct.unpack("<q", buf)[0]
  finally:
    os.close(fd)

  return (ptr_end - ptr_start) // element_size


def get_num_documents(path_prefixes: list[str]) -> int:
  """Total document count from .idx headers — O(1) per shard."""
  return sum(_read_idx_header(p)[1] for p in path_prefixes)


# ---------------------------------------------------------------------------
# Document sizes
# ---------------------------------------------------------------------------


def get_document_sizes(path_prefixes: list[str]) -> np.ndarray:
  """Extract per-document token counts from one or more shards.

  Per-shard results are cached individually so that different dataset
  specs sharing some of the same shards avoid redundant GCS Fuse I/O.
  """
  all_sizes = [_get_single_shard_doc_sizes(p) for p in path_prefixes]
  all_sizes = [s for s in all_sizes if len(s) > 0]
  if not all_sizes:
    return np.array([], dtype=np.int64)
  return np.concatenate(all_sizes)


@functools.lru_cache(maxsize=256)
def _get_single_shard_doc_sizes(prefix: str) -> np.ndarray:
  """Return per-document token counts for a single shard (cached)."""
  from maxtext.input_pipeline._mmap_datasource import MMapIndexedDataset  # pylint: disable=import-outside-toplevel

  ds = MMapIndexedDataset(prefix)
  try:
    doc_idx = ds.doc_idx
    sizes = ds.sizes
    num_docs = len(doc_idx) - 1
    if num_docs == 0:
      return np.array([], dtype=np.int64)
    starts = doc_idx[:num_docs].astype(np.intp)
    return np.add.reduceat(sizes.astype(np.int64), starts)
  finally:
    ds.close()


def parse_split_range(split_str, split_index, num_docs):
  """Parse a Megatron split-ratio string and return ``(start_doc, end_doc)``.

  Args:
      split_str: Comma-separated ratio string, e.g. ``'99,1'`` or ``'0.9,0.05,0.05'``.
      split_index: Which split to extract (0-based).
      num_docs: Total number of documents.

  Returns:
      ``(start_doc, end_doc)`` for the requested split.  Boundaries are
      computed with ``round()`` to match Megatron-LM.
  """
  ratios = [float(x) for x in split_str.split(",")]
  if split_index < 0 or split_index >= len(ratios):
    raise ValueError(f"split_index {split_index} out of range for {len(ratios)} splits")
  total = sum(ratios)
  ratios = [r / total for r in ratios]
  cumulative = [0.0]
  for r in ratios:
    cumulative.append(cumulative[-1] + r)
  start_doc = int(round(cumulative[split_index] * num_docs))
  end_doc = int(round(cumulative[split_index + 1] * num_docs))
  return start_doc, end_doc


def _normalize_weights(weights):
  """Normalize weights using numpy float64 arithmetic.

  Matches Megatron-LM's ``megatron.core.datasets.utils.normalize`` exactly::

      w = numpy.array(weights, dtype=numpy.float64)
      w = (w / numpy.sum(w)).tolist()

  Megatron applies this normalization **twice** during dataset construction:

  1. In ``BlendedMegatronDatasetBuilder`` before computing per-dataset buffer
     sizes (determines how many samples each sub-dataset must produce).
  2. In ``BlendedDataset.__init__`` on the already-normalized weights before
     building the blend index (determines the interleaved sampling order)
     and computing the cache hash.

  Due to floating-point arithmetic, the second pass can produce values that
  differ from the first by ~1 ULP.  Both passes must use numpy float64
  division (not Python float) to stay bit-identical with Megatron's cache
  keys and index arrays.
  """
  w = np.array(weights, dtype=np.float64)
  w_sum = np.sum(w)
  w = (w / w_sum).tolist()
  return w


def compute_blend_buffers(num_samples, weights, margin):
  """Compute per-dataset buffer sizes for Megatron-style blending.

  Uses the Megatron formula: ``buffer_i = ceil(ceil(num_samples * w_i) * (1 + margin/100))``.

  Args:
      num_samples: Total number of blended samples.
      weights: Normalized per-dataset weights (must already be normalized via
          :func:`_normalize_weights`; this function does **not** re-normalize).
      margin: Over-provisioning margin percentage.

  Returns:
      List of buffer sizes, one per dataset.
  """
  buffers = []
  for w in weights:
    target = math.ceil(num_samples * w)
    buffers.append(math.ceil(target * (1.0 + margin / 100.0)))
  return buffers


# ---------------------------------------------------------------------------
# Document index
# ---------------------------------------------------------------------------


def _build_doc_index_flat(num_docs, num_epochs, rng):
  doc_index = np.tile(np.arange(num_docs, dtype=np.int32), num_epochs)
  rng.shuffle(doc_index)
  return doc_index


def build_document_index(num_docs, num_epochs, seed, separate_last_epoch=False):
  """Build epoch-shuffled document ordering matching Megatron-LM."""
  rng = np.random.RandomState(seed)
  if not separate_last_epoch or num_epochs == 1:
    return _build_doc_index_flat(num_docs, num_epochs, rng)
  else:
    doc_idx_first = _build_doc_index_flat(num_docs, num_epochs - 1, rng)
    doc_idx_last = _build_doc_index_flat(num_docs, 1, rng)
    return np.concatenate((doc_idx_first, doc_idx_last))


# ---------------------------------------------------------------------------
# Sample index
# ---------------------------------------------------------------------------


def build_sample_index(doc_sizes, doc_index, seq_length, drop_last=True, add_extra_token=1):
  """Build sample boundary index matching Megatron's build_sample_idx.

  Returns array of shape (num_samples + 1, 2) where each row is
  (doc_index_position, token_offset_within_document).  Uses int32 by
  default and falls back to int64 when doc_index length or max doc size
  exceeds int32 range, matching Megatron's dynamic type selection.
  """
  # Match Megatron: use int32 unless doc_index length or max doc size
  # exceeds int32 range, then fall back to int64.
  sample_idx_max = max(len(doc_index), int(doc_sizes.max()) if len(doc_sizes) > 0 else 0)
  dtype = np.int32 if sample_idx_max <= np.iinfo(np.int32).max else np.int64

  total_tokens = int(np.sum(doc_sizes[doc_index]))

  if drop_last:
    num_samples = (total_tokens - add_extra_token) // seq_length
  else:
    num_samples = -(-(total_tokens - add_extra_token) // seq_length)

  if num_samples == 0:
    return np.zeros((1, 2), dtype=dtype)

  sample_idx = np.zeros((num_samples + 1, 2), dtype=dtype)
  doc_idx_index = 0
  doc_offset = 0

  for sample_idx_index in (
      tqdm(range(1, num_samples + 1), desc="Building sample index", unit="sample") if tqdm else range(1, num_samples + 1)
  ):
    remaining = seq_length + add_extra_token
    while remaining > 0:
      doc_id = doc_index[doc_idx_index]
      doc_length = int(doc_sizes[doc_id]) - doc_offset
      remaining -= doc_length
      if remaining <= 0:
        doc_offset += remaining + doc_length - add_extra_token
        remaining = 0
      else:
        if doc_idx_index == len(doc_index) - 1:
          doc_offset = int(doc_sizes[doc_index[doc_idx_index]]) - add_extra_token
          break
        doc_idx_index += 1
        doc_offset = 0
    sample_idx[sample_idx_index][0] = doc_idx_index
    sample_idx[sample_idx_index][1] = doc_offset

  return sample_idx


# ---------------------------------------------------------------------------
# Shuffle index
# ---------------------------------------------------------------------------


def _build_shuffle_index(num_samples, total_size, rng):
  """Build random permutation using an existing RandomState."""
  dtype = np.uint32
  if total_size >= np.iinfo(np.uint32).max - 1:
    dtype = np.int64
  shuffle_first = np.arange(0, num_samples, dtype=dtype)
  rng.shuffle(shuffle_first)
  if num_samples == total_size:
    return shuffle_first
  shuffle_last = np.arange(num_samples, total_size, dtype=dtype)
  rng.shuffle(shuffle_last)
  return np.concatenate((shuffle_first, shuffle_last))


def build_shuffle_index(num_samples, total_size, seed):
  """Build random permutation over samples matching Megatron-LM."""
  return _build_shuffle_index(num_samples, total_size, np.random.RandomState(seed))


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------


def compute_num_epochs(total_tokens, num_samples, seq_length, add_extra_token=1):
  tokens_needed = num_samples * seq_length + add_extra_token
  return max(1, math.ceil(tokens_needed / total_tokens))


def should_separate_last_epoch(
    num_epochs,
    tokens_per_epoch,
    num_samples,
    seq_length,
    add_extra_token=1,
    threshold=0.80,
):
  """Return True if the last epoch should be shuffled separately."""
  if num_epochs <= 1:
    return False
  num_samples_sans_final = ((num_epochs - 1) * tokens_per_epoch - add_extra_token) // seq_length
  num_from_final = num_samples - num_samples_sans_final
  num_per_epoch = (tokens_per_epoch - add_extra_token) // seq_length
  return num_from_final < int(threshold * num_per_epoch)


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------


def compute_index_hash(
    input_paths,
    num_epochs,
    separate_final_epoch,
    seed,
    seq_length,
    split=None,
    split_index=0,
    add_extra_token=1,
):
  """Compute the cache hash for a set of index parameters.

  This is the same hash used by :func:`convert` when saving files,
  exposed here so callers can check for cache hits before building.
  """
  desc = json.dumps(
      {
          "class": "MegatronNpyDataSource",
          "input_paths": sorted(input_paths) if not isinstance(input_paths[0], list) else input_paths,
          "num_epochs": num_epochs,
          "separate_final_epoch": separate_final_epoch,
          "seed": seed,
          "sequence_length": seq_length,
          "split": split,
          "split_index": split_index,
          "add_extra_token": add_extra_token,
      },
      indent=4,
  )
  return hashlib.md5(desc.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# convert / convert_blend  (orchestration)
# ---------------------------------------------------------------------------


def build_indices(
    input_paths,
    seq_length,
    num_samples=None,
    num_epochs=None,
    seed=1234,
    split=None,
    split_index=0,
    add_extra_token=1,
):
  """Compute Megatron-LM index arrays in memory without writing to disk.

  This is the pure-computation core shared by :func:`convert` (offline CLI)
  and :func:`_mmap_datasource._ensure_npy_indices` (runtime auto-rebuild).
  Because the computation is fully deterministic for a given seed, every
  host in a multi-host setup can call this independently and obtain
  identical results — no cross-host synchronisation is needed.

  Returns:
      Tuple of ``(doc_index, sample_index, shuffle_index, file_hash)``.
  """
  if num_samples is None and num_epochs is None:
    raise ValueError("Must specify either num_samples or num_epochs")
  if num_samples is not None and num_epochs is not None:
    raise ValueError("Cannot specify both num_samples and num_epochs")
  if add_extra_token != 1:
    log.warning(
        "add_extra_token=%d: MegatronNpyDataSource hardcodes add_extra_token=1 "
        "(matching Megatron production defaults). Index files built with a "
        "different value will produce incorrect sample boundaries at runtime.",
        add_extra_token,
    )

  all_prefixes = []
  for path in input_paths:
    all_prefixes.extend(resolve_shard_prefixes(path))
  # Sort globally to match _resolve_bin_prefixes() in _mmap_datasource.py,
  # which sorts across all directories.  Document IDs in the .npy index
  # files must use the same shard ordering as the runtime data source.
  all_prefixes = sorted(all_prefixes)
  log.info("Found %d shard(s)", len(all_prefixes))

  doc_sizes = get_document_sizes(all_prefixes)
  num_docs = len(doc_sizes)
  log.info("Total documents: %d, total tokens: %d", num_docs, int(doc_sizes.sum()))

  start_doc = 0
  if split is not None:
    start_doc, end_doc = parse_split_range(split, split_index, num_docs)
    num_docs = end_doc - start_doc
    log.info(
        "After split[%d]: %d documents (global %d-%d)",
        split_index,
        num_docs,
        start_doc,
        end_doc,
    )

  if num_docs == 0:
    raise ValueError("No documents found in input data" + (" after applying split" if split else ""))
  tokens_per_epoch = int(doc_sizes[start_doc : start_doc + num_docs].sum())
  if tokens_per_epoch == 0:
    raise ValueError("Total token count is 0 — all documents are empty")

  if num_epochs is not None:
    actual_epochs = num_epochs
    actual_samples = (actual_epochs * tokens_per_epoch - add_extra_token) // seq_length
  else:
    actual_epochs = compute_num_epochs(tokens_per_epoch, num_samples, seq_length, add_extra_token)
    actual_samples = num_samples

  separate = should_separate_last_epoch(actual_epochs, tokens_per_epoch, actual_samples, seq_length, add_extra_token)
  log.info(
      "Epochs: %d, samples: %d, separate_last_epoch: %s",
      actual_epochs,
      actual_samples,
      separate,
  )

  # Use a single RandomState that flows through doc_index -> shuffle_index,
  # matching Megatron-LM's gpt_dataset.py RNG state management exactly.
  rng = np.random.RandomState(seed)

  log.info(
      "Phase 1/3: Building document index (%d docs, %d epochs)...",
      num_docs,
      actual_epochs,
  )
  if not separate or actual_epochs == 1:
    doc_index = _build_doc_index_flat(num_docs, actual_epochs, rng)
  else:
    doc_idx_first = _build_doc_index_flat(num_docs, actual_epochs - 1, rng)
    doc_idx_last = _build_doc_index_flat(num_docs, 1, rng)
    doc_index = np.concatenate((doc_idx_first, doc_idx_last))

  # Offset local doc IDs [0..num_docs) to global IDs [start_doc..end_doc),
  # matching Megatron's `document_index[:] = documents` which stores global IDs.
  if start_doc > 0:
    doc_index += start_doc

  log.info("Phase 2/3: Building sample index (seq_length=%d)...", seq_length)
  # Pass the full doc_sizes array -- doc_index contains global IDs that index
  # into it, matching Megatron's build_sample_idx(sequence_lengths_full, ...).
  sample_index = build_sample_index(
      doc_sizes,
      doc_index,
      seq_length,
      drop_last=True,
      add_extra_token=add_extra_token,
  )
  total_samples = sample_index.shape[0] - 1

  log.info("Phase 3/3: Building shuffle index (%d samples)...", total_samples)
  if separate:
    num_samples_sans_final = ((actual_epochs - 1) * tokens_per_epoch - add_extra_token) // seq_length
    shuffle_index = _build_shuffle_index(num_samples_sans_final, total_samples, rng)
  else:
    shuffle_index = _build_shuffle_index(total_samples, total_samples, rng)

  # Hash based on (num_epochs, separate_final_epoch) rather than num_samples.
  # This means different num_samples values that map to the same epoch bucket
  # produce identical index files and share the same cache entry.
  file_hash = compute_index_hash(
      input_paths=all_prefixes,
      num_epochs=actual_epochs,
      separate_final_epoch=separate,
      seed=seed,
      seq_length=seq_length,
      split=split,
      split_index=split_index,
      add_extra_token=add_extra_token,
  )

  return doc_index, sample_index, shuffle_index, file_hash


def build_metadata(
    file_hash,
    input_paths,
    seq_length,
    num_samples,
    num_epochs,
    seed,
    split,
    split_index,
    add_extra_token,
    num_docs,
    total_samples,
    source,
):
  """Build a metadata dict for debugging npy cache files."""
  from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel

  return {
      "hash": file_hash,
      "source": source,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "input_paths": sorted(input_paths) if input_paths else [],
      "seq_length": seq_length,
      "num_samples_requested": num_samples,
      "num_epochs": num_epochs,
      "seed": seed,
      "split": split,
      "split_index": split_index,
      "add_extra_token": add_extra_token,
      "doc_index_len": num_docs,
      "total_samples": total_samples,
  }


def save_indices_atomic(output_dir, file_hash, doc_index, sample_index, shuffle_index, metadata=None):
  """Write index arrays to disk using atomic rename to prevent corruption."""
  os.makedirs(output_dir, exist_ok=True)
  paths = {}
  for name, data in [
      ("document_index", doc_index),
      ("sample_index", sample_index),
      ("shuffle_index", shuffle_index),
  ]:
    final_path = os.path.join(output_dir, f"{file_hash}-{name}.npy")
    save_npy_atomic(final_path, data)
    log.info("Saved %s: %s (shape=%s, dtype=%s)", name, final_path, data.shape, data.dtype)
    paths[name] = final_path

  if metadata is not None:
    meta_path = os.path.join(output_dir, f"{file_hash}-metadata.json")
    tmp_meta = os.path.join(output_dir, f"{file_hash}-metadata.tmp.{os.getpid()}.json")
    with open(tmp_meta, "w", encoding="utf-8") as f:
      json.dump(metadata, f, indent=2)
    os.replace(tmp_meta, meta_path)
    log.info("Saved metadata: %s", meta_path)
    paths["metadata"] = meta_path

  return paths


def convert(
    input_paths,
    output_dir,
    seq_length,
    num_samples=None,
    num_epochs=None,
    seed=1234,
    split=None,
    split_index=0,
    add_extra_token=1,
):
  """Precompute Megatron-LM index .npy files. Returns dict of output paths."""
  doc_index, sample_index, shuffle_index, file_hash = build_indices(
      input_paths=input_paths,
      seq_length=seq_length,
      num_samples=num_samples,
      num_epochs=num_epochs,
      seed=seed,
      split=split,
      split_index=split_index,
      add_extra_token=add_extra_token,
  )
  metadata = build_metadata(
      file_hash=file_hash,
      input_paths=input_paths,
      seq_length=seq_length,
      num_samples=num_samples,
      num_epochs=num_epochs,
      seed=seed,
      split=split,
      split_index=split_index,
      add_extra_token=add_extra_token,
      num_docs=doc_index.shape[0],
      total_samples=sample_index.shape[0] - 1,
      source="convert",
  )
  return save_indices_atomic(output_dir, file_hash, doc_index, sample_index, shuffle_index, metadata=metadata)


def convert_blend(
    dataset_specs,
    total_samples,
    seq_length,
    seed=1234,
    margin=0.5,
    max_workers=None,
    split=None,
    split_index=0,
    add_extra_token=1,
    blend_index_output_dir=None,
):
  """Precompute Megatron-LM index .npy files for a blended mixture.

  Follows Megatron's ``BlendedMegatronDatasetBuilder._get_size_per_split_per_dataset``
  formula to compute per-dataset buffer sizes, then builds all datasets
  concurrently using :func:`convert`.

  Args:
      dataset_specs: List of dicts, each with keys:
          - ``"input"``: list of input path(s) or directory(ies)
          - ``"weight"``: float blend weight
          - ``"output_dir"``: directory for this dataset's .npy output
      total_samples: Total training samples (= train_steps x global_batch_size).
      seq_length: Sequence length for sample construction.
      seed: Random seed (passed to each dataset's :func:`convert`).
      margin: Overprovisioning margin percentage (default 0.5, matching Megatron).
      max_workers: Maximum concurrent builds (default: number of datasets).
      split: Comma-separated split ratios (e.g. '0.9,0.05,0.05').
      split_index: Index into the split ratios.
      add_extra_token: Extra token for next-token prediction (default 1).
      blend_index_output_dir: If set, also write the fixed-name global blend
        dispatch pair (``dataset_index.npy`` and ``dataset_sample_index.npy``)
        accepted by ``MegatronBlendedDataSource(blend_index_dir=...)``.

  Returns:
      List of dicts, one per dataset, each containing:
          - ``"paths"``: dict of {name: path} from :func:`convert`
          - ``"weight"``: the dataset's normalized blend weight (numpy float64)
          - ``"buffer_samples"``: number of samples allocated to this dataset
  """
  from concurrent.futures import ThreadPoolExecutor, as_completed  # pylint: disable=import-outside-toplevel

  if not dataset_specs:
    raise ValueError("dataset_specs must be non-empty")

  raw_weights = np.asarray([s["weight"] for s in dataset_specs], dtype=np.float64)
  if np.any(raw_weights < 0):
    raise ValueError(f"Blend weights must be non-negative, got {raw_weights.tolist()}")
  if raw_weights.sum() <= 0:
    raise ValueError(f"Total weight must be positive, got {float(raw_weights.sum())}")
  # Runtime parsing removes zero-weight entries before constructing either
  # child sources or the global dispatcher. Do the same offline so a zero
  # entry never tries to build a meaningless zero-sample child cache.
  keep = raw_weights > 0
  dataset_specs = [spec for spec, keep_spec in zip(dataset_specs, keep) if keep_spec]
  weights = raw_weights[keep].tolist()

  # Normalize once, matching Megatron's BlendedMegatronDatasetBuilder.
  # Downstream _normalize_and_filter_weights does the second normalize
  # to match BlendedDataset.__init__.
  norm_weights = _normalize_weights(weights)

  buffer_per_ds = compute_blend_buffers(total_samples, norm_weights, margin)

  log.info(
      "Blend: %d datasets, total_samples=%d, margin=%.1f%%",
      len(dataset_specs),
      total_samples,
      margin,
  )
  for i, (spec, buf) in enumerate(zip(dataset_specs, buffer_per_ds)):
    log.info(
        "  Dataset %d: weight=%.4f, buffer_samples=%d, output=%s",
        i,
        norm_weights[i],
        buf,
        spec["output_dir"],
    )

  if max_workers is None:
    max_workers = len(dataset_specs)

  results = [None] * len(dataset_specs)

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {}
    for i, spec in enumerate(dataset_specs):
      fut = executor.submit(
          convert,
          input_paths=spec["input"],
          output_dir=spec["output_dir"],
          seq_length=seq_length,
          num_samples=buffer_per_ds[i],
          seed=seed,
          split=split,
          split_index=split_index,
          add_extra_token=add_extra_token,
      )
      futures[fut] = i

    for fut in as_completed(futures):
      idx = futures[fut]
      paths = fut.result()  # propagates exceptions
      results[idx] = {
          "paths": paths,
          "weight": norm_weights[idx],
          "buffer_samples": buffer_per_ds[idx],
      }
      log.info("Dataset %d done: %s", idx, list(paths.values()))

  if blend_index_output_dir:
    # Import locally: _megatron_blending imports this module for shared atomic
    # writes, while this offline-only path needs its dispatcher builder.
    from maxtext.input_pipeline._megatron_blending import build_and_save_blend_indices  # pylint: disable=import-outside-toplevel

    dataset_lengths = [
        int(np.load(result["paths"]["sample_index"], allow_pickle=False, mmap_mode="r").shape[0] - 1)
        for result in results
    ]
    blend_paths = build_and_save_blend_indices(
        output_dir=blend_index_output_dir,
        # Megatron normalizes mixture weights once while constructing child
        # datasets and once again in its blended dataset.  The helper performs
        # that second normalization, matching runtime construction exactly.
        weights=norm_weights,
        dataset_lengths=dataset_lengths,
        size=total_samples,
    )
    log.info("Blend dispatch indices written: %s", list(blend_paths.values()))

  return results
