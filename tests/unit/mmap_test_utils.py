"""Shared test utilities for MMap dataset tests."""

import struct

import numpy as np

from maxtext.input_pipeline._mmap_datasource import (
    DTYPE_CODES_INV,
    MMAP_INDEX_MAGIC,
    MMAP_INDEX_VERSION,
)


def create_mmap_test_data(path_prefix, sequences, dtype=np.int32, doc_boundaries=None):
  """Generate Megatron-LM format .idx + .bin test files.

  Args:
      path_prefix: Path prefix (without .idx/.bin extension).
      sequences: List of 1-D numpy arrays (token ID sequences).
      dtype: Numpy dtype for the token IDs.
      doc_boundaries: Optional list of document boundary indices into the
          sequences list. If None, all sequences belong to one document.

  Returns:
      path_prefix for convenience.
  """
  dtype = np.dtype(dtype)
  dtype_code = DTYPE_CODES_INV[dtype.type]

  num_sequences = len(sequences)
  sizes = np.array([len(s) for s in sequences], dtype=np.int32)

  # Compute byte pointers
  pointers = np.zeros(num_sequences, dtype=np.int64)
  offset = 0
  for i, seq in enumerate(sequences):
    pointers[i] = offset
    offset += len(seq) * dtype.itemsize

  # Document index — use Convention B (num_documents_field = len(doc_idx))
  # so that both MaxText's auto-detecting reader and Megatron-Core's
  # _IndexReader can parse the file correctly.
  if doc_boundaries is None:
    doc_idx = np.array([0, num_sequences], dtype=np.int64)
  else:
    doc_idx = np.array(doc_boundaries, dtype=np.int64)
  num_documents = len(doc_idx)  # Convention B: header stores len(doc_idx)

  # Write .idx
  with open(path_prefix + ".idx", "wb") as f:
    f.write(MMAP_INDEX_MAGIC)
    f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
    f.write(struct.pack("<B", dtype_code))
    f.write(struct.pack("<Q", num_sequences))
    f.write(struct.pack("<Q", num_documents))
    f.write(sizes.tobytes())
    f.write(pointers.tobytes())
    f.write(doc_idx.tobytes())

  # Write .bin
  with open(path_prefix + ".bin", "wb") as f:
    for seq in sequences:
      f.write(np.array(seq, dtype=dtype).tobytes())

  return path_prefix
