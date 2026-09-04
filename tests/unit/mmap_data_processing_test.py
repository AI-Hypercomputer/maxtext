# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mmap and mmap_npy data processing."""

# pylint: disable=redefined-outer-name

import os
import pickle
import struct
import tempfile
from types import SimpleNamespace
from unittest import mock, TestCase
from concurrent.futures import ThreadPoolExecutor  # pylint: disable=no-name-in-module

import numpy as np
import pytest

import grain.python as grain

from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline._mmap_datasource import (
    DTYPE_CODES,
    DTYPE_CODES_INV,
    MMAP_INDEX_HEADER_SIZE,
    MMAP_INDEX_MAGIC,
    MMAP_INDEX_VERSION,
    MMapDatasetConfig,
    MegatronNpyDataSource,
    MMapIndexedDataset,
    MMapIndexedDataSource,
    MMapSampleIndexDataSource,
    MultiShardMMapIndexedDataSource,
    create_mmap_npy_source,
    _discover_npy_indices,
    _ensure_npy_indices,
    _resolve_bin_prefixes,
    _parse_mmap_npy_spec,
    _parse_weighted_mixture,
)
from tests.unit.mmap_test_utils import create_mmap_test_data
from tools.data_processing.mmap_index_builder import convert

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
  with tempfile.TemporaryDirectory() as d:
    yield d


@pytest.fixture
def simple_dataset(tmp_dir):
  """3 sequences of varying length, int32."""
  seqs = [
      np.array([1, 2, 3], dtype=np.int32),
      np.array([4, 5, 6, 7], dtype=np.int32),
      np.array([8, 9], dtype=np.int32),
  ]
  prefix = os.path.join(tmp_dir, "simple")
  create_mmap_test_data(prefix, seqs)
  return prefix, seqs


# ===========================================================================
# Unit tests: MMapIndexedDataset
# ===========================================================================


class TestMMapIndexedDataset:
  """Tests for MMapIndexedDataset low-level read and validation logic."""

  def test_basic_read(self, simple_dataset):
    prefix, seqs = simple_dataset
    ds = MMapIndexedDataset(prefix)
    assert len(ds) == len(seqs)
    for i, expected in enumerate(seqs):
      np.testing.assert_array_equal(ds[i], expected)

  def test_sizes_and_pointers(self, simple_dataset):
    prefix, seqs = simple_dataset
    ds = MMapIndexedDataset(prefix)
    expected_sizes = np.array([len(s) for s in seqs], dtype=np.int32)
    np.testing.assert_array_equal(ds.sizes, expected_sizes)
    assert len(ds.pointers) == len(seqs)

  def test_document_boundaries(self, tmp_dir):
    seqs = [
        np.array([1, 2], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
        np.array([5, 6], dtype=np.int32),
    ]
    doc_boundaries = [0, 2, 3]  # doc0: seq0,seq1; doc1: seq2
    prefix = os.path.join(tmp_dir, "docs")
    create_mmap_test_data(prefix, seqs, doc_boundaries=doc_boundaries)
    ds = MMapIndexedDataset(prefix)
    np.testing.assert_array_equal(ds.doc_idx, np.array(doc_boundaries, dtype=np.int64))

  def test_single_sequence(self, tmp_dir):
    seqs = [np.array([42, 43, 44], dtype=np.int32)]
    prefix = os.path.join(tmp_dir, "single")
    create_mmap_test_data(prefix, seqs)
    ds = MMapIndexedDataset(prefix)
    assert len(ds) == 1
    np.testing.assert_array_equal(ds[0], seqs[0])

  def test_large_sequence(self, tmp_dir):
    seqs = [np.arange(10000, dtype=np.int32)]
    prefix = os.path.join(tmp_dir, "large")
    create_mmap_test_data(prefix, seqs)
    ds = MMapIndexedDataset(prefix)
    np.testing.assert_array_equal(ds[0], seqs[0])

  def test_variable_length_sequences(self, tmp_dir):
    seqs = [np.arange(i + 1, dtype=np.int32) for i in range(10)]
    prefix = os.path.join(tmp_dir, "varlen")
    create_mmap_test_data(prefix, seqs)
    ds = MMapIndexedDataset(prefix)
    for i, expected in enumerate(seqs):
      np.testing.assert_array_equal(ds[i], expected)

  def test_different_dtypes(self, tmp_dir):
    for dtype in [np.int16, np.uint16, np.int64, np.uint8]:
      seqs = [np.array([10, 20, 30], dtype=dtype)]
      prefix = os.path.join(tmp_dir, f"dtype_{np.dtype(dtype).name}")
      create_mmap_test_data(prefix, seqs, dtype=dtype)
      ds = MMapIndexedDataset(prefix)
      assert ds.dtype == np.dtype(dtype).type
      np.testing.assert_array_equal(ds[0], seqs[0])

  def test_index_out_of_range(self, simple_dataset):
    prefix, seqs = simple_dataset
    ds = MMapIndexedDataset(prefix)
    with pytest.raises(IndexError):
      _ = ds[len(seqs)]
    with pytest.raises(IndexError):
      _ = ds[-(len(seqs) + 1)]

  def test_negative_index(self, simple_dataset):
    prefix, seqs = simple_dataset
    ds = MMapIndexedDataset(prefix)
    np.testing.assert_array_equal(ds[-1], seqs[-1])

  # --- get() partial read ---

  def test_get_full(self, simple_dataset):
    prefix, seqs = simple_dataset
    ds = MMapIndexedDataset(prefix)
    np.testing.assert_array_equal(ds.get(1), seqs[1])

  def test_get_with_offset(self, simple_dataset):
    prefix, _ = simple_dataset
    ds = MMapIndexedDataset(prefix)
    # seq1 = [4, 5, 6, 7], offset=1 -> [5, 6, 7]
    np.testing.assert_array_equal(ds.get(1, offset=1), np.array([5, 6, 7], dtype=np.int32))

  def test_get_with_offset_and_length(self, simple_dataset):
    prefix, _ = simple_dataset
    ds = MMapIndexedDataset(prefix)
    # seq1 = [4, 5, 6, 7], offset=1, length=2 -> [5, 6]
    np.testing.assert_array_equal(ds.get(1, offset=1, length=2), np.array([5, 6], dtype=np.int32))

  def test_get_offset_out_of_range(self, simple_dataset):
    prefix, _ = simple_dataset
    ds = MMapIndexedDataset(prefix)
    with pytest.raises(IndexError, match="Offset"):
      ds.get(0, offset=100)

  def test_get_length_exceeds_size(self, simple_dataset):
    prefix, _ = simple_dataset
    ds = MMapIndexedDataset(prefix)
    with pytest.raises(IndexError, match="exceeds"):
      ds.get(0, offset=0, length=100)

  def test_get_zero_length(self, simple_dataset):
    prefix, _ = simple_dataset
    ds = MMapIndexedDataset(prefix)
    result = ds.get(0, offset=0, length=0)
    assert len(result) == 0

  # --- slice support ---

  def test_slice(self, simple_dataset):
    prefix, seqs = simple_dataset
    ds = MMapIndexedDataset(prefix)
    result = ds[0:2]
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], seqs[0])
    np.testing.assert_array_equal(result[1], seqs[1])

  def test_slice_with_step_raises(self, simple_dataset):
    """Slices with step != 1 should raise ValueError (Megatron semantics)."""
    prefix, _ = simple_dataset
    ds = MMapIndexedDataset(prefix)
    with pytest.raises(ValueError, match="step=1"):
      _ = ds[::2]

  # --- error handling: corrupted/truncated idx ---

  def test_invalid_magic(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "bad_magic")
    with open(prefix + ".idx", "wb") as f:
      f.write(b"BADMAGIC\x00")
      # Pad to header size
      f.write(b"\x00" * (MMAP_INDEX_HEADER_SIZE - 9))
    with open(prefix + ".bin", "wb") as f:
      f.write(b"")
    with pytest.raises(ValueError, match="Invalid magic"):
      MMapIndexedDataset(prefix)

  def test_bad_version(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "bad_ver")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", 999))  # bad version
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 0))  # num_seq
      f.write(struct.pack("<Q", 0))  # num_doc
      # doc_idx for 0 documents: just [0]
      f.write(np.array([0], dtype=np.int64).tobytes())
    with open(prefix + ".bin", "wb") as f:
      f.write(b"")
    with pytest.raises(ValueError, match="Unsupported MMap index version"):
      MMapIndexedDataset(prefix)

  def test_unknown_dtype_code(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "bad_dtype")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 99))  # unknown dtype
      f.write(struct.pack("<Q", 0))
      f.write(struct.pack("<Q", 0))
      f.write(np.array([0], dtype=np.int64).tobytes())
    with open(prefix + ".bin", "wb") as f:
      f.write(b"")
    with pytest.raises(ValueError, match="Unknown dtype code"):
      MMapIndexedDataset(prefix)

  def test_truncated_header(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "trunc_header")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)  # only 9 bytes, header needs 34
    with open(prefix + ".bin", "wb") as f:
      f.write(b"")
    with pytest.raises(ValueError, match="too small"):
      MMapIndexedDataset(prefix)

  def test_truncated_body(self, tmp_dir):
    """Header claims N sequences but body is truncated."""
    prefix = os.path.join(tmp_dir, "trunc_body")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 100))  # claims 100 sequences
      f.write(struct.pack("<Q", 1))  # 1 document
      # Don't write the sizes/pointers/doc_idx → truncated
    with open(prefix + ".bin", "wb") as f:
      f.write(b"")
    with pytest.raises(ValueError, match="truncated"):
      MMapIndexedDataset(prefix)

  def test_missing_file(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "nonexistent")
    with pytest.raises(FileNotFoundError):
      MMapIndexedDataset(prefix)

  def test_missing_bin_file(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "no_bin")
    # Create idx but no bin
    seqs = [np.array([1], dtype=np.int32)]
    create_mmap_test_data(prefix, seqs)
    os.remove(prefix + ".bin")
    with pytest.raises(FileNotFoundError, match="Binary file"):
      MMapIndexedDataset(prefix)

  # --- full validation (MMAP_IDX_FULL_VALIDATION=1) ---

  def test_bin_too_small(self, tmp_dir, monkeypatch):
    """Pointers reference data beyond the end of .bin file."""
    monkeypatch.setenv("MMAP_IDX_FULL_VALIDATION", "1")
    prefix = os.path.join(tmp_dir, "bin_small")
    seqs = [np.array([1, 2, 3], dtype=np.int32)]
    create_mmap_test_data(prefix, seqs)
    # Truncate .bin to 4 bytes (needs 12)
    with open(prefix + ".bin", "wb") as f:
      f.write(b"\x00" * 4)
    with pytest.raises(ValueError, match="too small"):
      MMapIndexedDataset(prefix)

  def test_negative_size_raises(self, tmp_dir, monkeypatch):
    """Negative sizes in the idx file should raise ValueError with full validation."""
    monkeypatch.setenv("MMAP_IDX_FULL_VALIDATION", "1")
    prefix = os.path.join(tmp_dir, "neg_size")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 1))  # 1 sequence
      f.write(struct.pack("<Q", 1))  # 1 document
      f.write(np.array([-5], dtype=np.int32).tobytes())  # negative size
      f.write(np.array([0], dtype=np.int64).tobytes())  # pointer
      f.write(np.array([0, 1], dtype=np.int64).tobytes())  # doc_idx
    with open(prefix + ".bin", "wb") as f:
      f.write(b"\x00" * 4)
    with pytest.raises(ValueError, match="Negative sizes"):
      MMapIndexedDataset(prefix)

  def test_negative_pointer_raises(self, tmp_dir, monkeypatch):
    """Negative pointers in the idx file should raise ValueError with full validation."""
    monkeypatch.setenv("MMAP_IDX_FULL_VALIDATION", "1")
    prefix = os.path.join(tmp_dir, "neg_ptr")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 1))  # 1 sequence
      f.write(struct.pack("<Q", 1))  # 1 document
      f.write(np.array([1], dtype=np.int32).tobytes())  # size
      f.write(np.array([-8], dtype=np.int64).tobytes())  # negative pointer
      f.write(np.array([0, 1], dtype=np.int64).tobytes())
    with open(prefix + ".bin", "wb") as f:
      f.write(b"\x00" * 4)
    with pytest.raises(ValueError, match="Negative pointers"):
      MMapIndexedDataset(prefix)

  def test_misaligned_pointer_raises(self, tmp_dir, monkeypatch):
    """Pointer not aligned to dtype itemsize should raise ValueError with full validation."""
    monkeypatch.setenv("MMAP_IDX_FULL_VALIDATION", "1")
    prefix = os.path.join(tmp_dir, "misaligned")
    # int32 has itemsize=4, so pointer=3 is misaligned
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 1))  # 1 sequence
      f.write(struct.pack("<Q", 1))  # 1 document
      f.write(np.array([1], dtype=np.int32).tobytes())  # size=1
      f.write(np.array([3], dtype=np.int64).tobytes())  # pointer=3 (misaligned for int32)
      f.write(np.array([0, 1], dtype=np.int64).tobytes())
    with open(prefix + ".bin", "wb") as f:
      f.write(b"\x00" * 8)  # enough bytes
    with pytest.raises(ValueError, match="Misaligned pointers"):
      MMapIndexedDataset(prefix)

  # --- doc_idx validation ---

  def test_non_monotonic_doc_idx_raises(self, tmp_dir, monkeypatch):
    """doc_idx that decreases should raise ValueError with full validation."""
    monkeypatch.setenv("MMAP_IDX_FULL_VALIDATION", "1")
    prefix = os.path.join(tmp_dir, "bad_docidx")
    seqs = [
        np.array([1, 2], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
        np.array([5, 6], dtype=np.int32),
    ]
    # 3 sequences, 3 documents, doc_idx = [0, 2, 1, 3] — non-monotonic at position 1
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 3))  # 3 sequences
      f.write(struct.pack("<Q", 3))  # 3 documents
      f.write(np.array([2, 2, 2], dtype=np.int32).tobytes())  # sizes
      f.write(np.array([0, 8, 16], dtype=np.int64).tobytes())  # pointers
      f.write(np.array([0, 2, 1, 3], dtype=np.int64).tobytes())  # non-monotonic: 0,2,1,3
    with open(prefix + ".bin", "wb") as f:
      for seq in seqs:
        f.write(seq.tobytes())
    with pytest.raises(ValueError, match="Non-monotonic doc_idx"):
      MMapIndexedDataset(prefix)

  def test_doc_idx_exceeds_num_sequences_raises(self, tmp_dir):
    """doc_idx with last entry > num_sequences should raise."""
    prefix = os.path.join(tmp_dir, "docidx_oob")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 1))  # 1 sequence
      f.write(struct.pack("<Q", 1))  # 1 document
      f.write(np.array([2], dtype=np.int32).tobytes())
      f.write(np.array([0], dtype=np.int64).tobytes())
      f.write(np.array([0, 99], dtype=np.int64).tobytes())  # 99 > 1
    with open(prefix + ".bin", "wb") as f:
      f.write(np.array([1, 2], dtype=np.int32).tobytes())
    with pytest.raises(ValueError, match="must equal num_sequences"):
      MMapIndexedDataset(prefix)

  def test_doc_idx_first_entry_nonzero_raises(self, tmp_dir):
    """doc_idx[0] != 0 should raise ValueError (unreachable sequences)."""
    prefix = os.path.join(tmp_dir, "docidx_nonzero_start")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 2))  # 2 sequences
      f.write(struct.pack("<Q", 1))  # 1 document
      f.write(np.array([2, 2], dtype=np.int32).tobytes())  # sizes
      f.write(np.array([0, 8], dtype=np.int64).tobytes())  # pointers
      f.write(np.array([1, 2], dtype=np.int64).tobytes())  # doc_idx starts at 1, not 0
    with open(prefix + ".bin", "wb") as f:
      f.write(np.array([1, 2, 3, 4], dtype=np.int32).tobytes())
    with pytest.raises(ValueError, match="first entry must be 0"):
      MMapIndexedDataset(prefix)

  def test_doc_idx_last_entry_less_than_num_sequences_raises(self, tmp_dir):
    """doc_idx[-1] < num_sequences should raise (unreachable sequences at tail)."""
    prefix = os.path.join(tmp_dir, "docidx_short")
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 4))  # int32
      f.write(struct.pack("<Q", 3))  # 3 sequences
      f.write(struct.pack("<Q", 1))  # 1 document
      f.write(np.array([2, 2, 2], dtype=np.int32).tobytes())
      f.write(np.array([0, 8, 16], dtype=np.int64).tobytes())
      f.write(np.array([0, 2], dtype=np.int64).tobytes())  # only covers 2 of 3
    with open(prefix + ".bin", "wb") as f:
      f.write(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32).tobytes())
    with pytest.raises(ValueError, match="must equal num_sequences"):
      MMapIndexedDataset(prefix)

  def test_trailing_garbage_bytes_raises(self, tmp_dir):
    """Index file with unexpected trailing bytes should fail-fast."""
    prefix = os.path.join(tmp_dir, "garbage")
    seqs = [np.array([1, 2], dtype=np.int32)]
    create_mmap_test_data(prefix, seqs)
    # Append arbitrary trailing bytes (not matching multimodal size)
    with open(prefix + ".idx", "ab") as f:
      f.write(b"\xff\xff\xff")
    with pytest.raises(ValueError, match="unexpected trailing bytes"):
      MMapIndexedDataset(prefix)

  def test_multimodal_sequence_modes_trailing_bytes_raise(self, tmp_dir):
    """Megatron-Core multimodal idx files have an extra sequence_modes block."""
    prefix = os.path.join(tmp_dir, "multimodal")
    seqs = [
        np.array([1, 2], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
    ]
    create_mmap_test_data(prefix, seqs)
    with open(prefix + ".idx", "ab") as f:
      f.write(np.array([0, 1], dtype=np.uint8).tobytes())

    with pytest.raises(ValueError, match="multimodal"):
      MMapIndexedDataset(prefix)

  def test_convention_a_doc_count_header_is_supported(self, tmp_dir):
    """Original Megatron convention stores document count, not len(doc_idx)."""
    prefix = os.path.join(tmp_dir, "convention_a")
    sequences = [
        np.array([1, 2], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int32),
    ]
    dtype_code = DTYPE_CODES_INV[np.int32]
    sizes = np.array([len(seq) for seq in sequences], dtype=np.int32)
    pointers = np.array([0, sequences[0].nbytes], dtype=np.int64)
    doc_idx = np.array([0, 1, 2], dtype=np.int64)
    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", dtype_code))
      f.write(struct.pack("<Q", len(sequences)))
      f.write(struct.pack("<Q", 2))  # Convention A: number of documents.
      f.write(sizes.tobytes())
      f.write(pointers.tobytes())
      f.write(doc_idx.tobytes())
    with open(prefix + ".bin", "wb") as f:
      for seq in sequences:
        f.write(seq.tobytes())

    ds = MMapIndexedDataset(prefix)
    assert ds._num_documents == 2  # pylint: disable=protected-access
    np.testing.assert_array_equal(ds.doc_idx, doc_idx)

  def test_short_read_raises_when_bin_is_truncated_after_index_load(self, tmp_dir):
    """Default validation is lightweight, so get() still detects short reads."""
    prefix = os.path.join(tmp_dir, "short_read")
    create_mmap_test_data(prefix, [np.array([1, 2, 3], dtype=np.int32)])
    with open(prefix + ".bin", "wb") as f:
      f.write(np.array([1], dtype=np.int32).tobytes())

    ds = MMapIndexedDataset(prefix)
    with pytest.raises(IOError, match="Short read"):
      ds.get(0)

  def test_close_is_idempotent_and_releases_arrays(self, simple_dataset):
    prefix, _ = simple_dataset
    ds = MMapIndexedDataset(prefix)

    ds.close()
    ds.close()

    assert ds.sizes is None
    assert ds.pointers is None
    assert ds.doc_idx is None

  # --- pickle ---

  def test_pickle_roundtrip(self, simple_dataset):
    prefix, seqs = simple_dataset
    ds = MMapIndexedDataset(prefix)
    ds2 = pickle.loads(pickle.dumps(ds))
    assert len(ds2) == len(seqs)
    for i, expected in enumerate(seqs):
      np.testing.assert_array_equal(ds2[i], expected)


# ===========================================================================
# Unit tests: MMapIndexedDataSource
# ===========================================================================


class TestMMapIndexedDataSource:
  """Tests for MMapIndexedDataSource Grain-compatible data source wrapper."""

  def test_getitem_returns_dict(self, simple_dataset):
    prefix, seqs = simple_dataset
    source = MMapIndexedDataSource(prefix)
    item = source[0]
    assert isinstance(item, dict)
    assert "text" in item
    np.testing.assert_array_equal(item["text"], seqs[0])

  def test_getitem_returns_copy(self, simple_dataset):
    """DataSource should return a copy (safe for Grain workers)."""
    prefix, _ = simple_dataset
    source = MMapIndexedDataSource(prefix)
    item = source[0]
    assert item["text"].flags.owndata

  def test_custom_feature_name(self, simple_dataset):
    prefix, _ = simple_dataset
    source = MMapIndexedDataSource(prefix, feature_name="tokens")
    item = source[0]
    assert "tokens" in item
    assert "text" not in item

  def test_len(self, simple_dataset):
    prefix, seqs = simple_dataset
    source = MMapIndexedDataSource(prefix)
    assert len(source) == len(seqs)

  def test_as_grain_map_dataset(self, simple_dataset):
    prefix, seqs = simple_dataset
    source = MMapIndexedDataSource(prefix)
    ds = grain.MapDataset.source(source)
    assert len(ds) == len(seqs)
    item = ds[0]
    assert "text" in item

  def test_pickle_roundtrip(self, simple_dataset):
    prefix, seqs = simple_dataset
    source = MMapIndexedDataSource(prefix)
    source2 = pickle.loads(pickle.dumps(source))
    assert len(source2) == len(seqs)
    np.testing.assert_array_equal(source2[0]["text"], seqs[0])

  def test_concurrent_reads(self, simple_dataset):
    prefix, seqs = simple_dataset
    source = MMapIndexedDataSource(prefix)

    def read_item(idx):
      return source[idx]["text"]

    with ThreadPoolExecutor(max_workers=4) as executor:
      results = list(executor.map(read_item, range(len(seqs))))
    for _, (result, expected) in enumerate(zip(results, seqs)):
      np.testing.assert_array_equal(result, expected)

  def test_check_eod_presence_does_not_warn_when_any_checked_doc_has_eod(self, tmp_dir):
    prefix = create_mmap_test_data(
        os.path.join(tmp_dir, "has_eod"),
        [
            np.array([], dtype=np.int32),
            np.array([10, 0], dtype=np.int32),
        ],
        doc_boundaries=[0, 0, 2],
    )
    source = MMapIndexedDataSource(prefix, split_sentences=True)

    with mock.patch("maxtext.input_pipeline._mmap_datasource.log.warning") as warning:
      source.check_eod_presence(eod_id=0, mode_label="unit")

    warning.assert_not_called()


# ===========================================================================
# Unit tests: MMapIndexedDataSource with split_sentences=True
# ===========================================================================


class TestMMapIndexedDataSourceSplitSentences:
  """Tests for document-level indexing when split_sentences=True."""

  @pytest.fixture
  def multi_doc_dataset(self, tmp_dir):
    """3 documents: doc0 has 2 sentences, doc1 has 1, doc2 has 3."""
    seqs = [
        np.array([10, 11], dtype=np.int32),  # doc0 sent0
        np.array([12, 13, 14], dtype=np.int32),  # doc0 sent1
        np.array([20, 21], dtype=np.int32),  # doc1 sent0
        np.array([30], dtype=np.int32),  # doc2 sent0
        np.array([31, 32], dtype=np.int32),  # doc2 sent1
        np.array([33, 34, 35], dtype=np.int32),  # doc2 sent2
    ]
    doc_boundaries = [0, 2, 3, 6]  # 3 documents
    prefix = os.path.join(tmp_dir, "split_sent")
    create_mmap_test_data(prefix, seqs, doc_boundaries=doc_boundaries)
    return prefix, seqs, doc_boundaries

  def test_len_returns_num_documents(self, multi_doc_dataset):
    prefix, *_ = multi_doc_dataset
    source = MMapIndexedDataSource(prefix, split_sentences=True)
    assert len(source) == 3  # 3 documents, not 6 sequences

  def test_getitem_concatenates_sentences(self, multi_doc_dataset):
    prefix, *_ = multi_doc_dataset
    source = MMapIndexedDataSource(prefix, split_sentences=True)
    # doc0 = concat(seqs[0], seqs[1]) = [10, 11, 12, 13, 14]
    np.testing.assert_array_equal(
        source[0]["text"],
        np.array([10, 11, 12, 13, 14], dtype=np.int32),
    )
    # doc1 = seqs[2] = [20, 21]
    np.testing.assert_array_equal(
        source[1]["text"],
        np.array([20, 21], dtype=np.int32),
    )
    # doc2 = concat(seqs[3], seqs[4], seqs[5]) = [30, 31, 32, 33, 34, 35]
    np.testing.assert_array_equal(
        source[2]["text"],
        np.array([30, 31, 32, 33, 34, 35], dtype=np.int32),
    )

  def test_single_sentence_doc(self, multi_doc_dataset):
    """Document with a single sentence returns that sentence directly."""
    prefix, _, _ = multi_doc_dataset
    source = MMapIndexedDataSource(prefix, split_sentences=True)
    # doc1 has 1 sentence
    result = source[1]["text"]
    np.testing.assert_array_equal(result, np.array([20, 21], dtype=np.int32))

  def test_split_sentences_false_returns_sequences(self, multi_doc_dataset):
    """With split_sentences=False, len/getitem use sequence-level indexing."""
    prefix, seqs, _ = multi_doc_dataset
    source = MMapIndexedDataSource(prefix, split_sentences=False)
    assert len(source) == 6
    np.testing.assert_array_equal(source[0]["text"], seqs[0])

  def test_pickle_roundtrip_split_sentences(self, multi_doc_dataset):
    prefix, _, _ = multi_doc_dataset
    source = MMapIndexedDataSource(prefix, split_sentences=True)
    source2 = pickle.loads(pickle.dumps(source))
    assert len(source2) == 3
    np.testing.assert_array_equal(
        source2[0]["text"],
        np.array([10, 11, 12, 13, 14], dtype=np.int32),
    )

  def test_grain_shuffle_operates_on_documents(self, multi_doc_dataset):
    """Grain shuffle with split_sentences=True shuffles documents, not sentences."""
    prefix, _, _ = multi_doc_dataset
    source = MMapIndexedDataSource(prefix, split_sentences=True)
    ds = grain.MapDataset.source(source)
    ds = ds.shuffle(seed=42)
    ds = ds.to_iter_dataset(read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1))
    items = list(ds)
    assert len(items) == 3
    # Each item should be a complete document (concatenated sentences)
    doc_lengths = sorted([len(item["text"]) for item in items])
    assert doc_lengths == [2, 5, 6]  # doc1=2, doc0=5, doc2=6

  def test_negative_index_document(self, multi_doc_dataset):
    """Negative index should resolve correctly in document mode."""
    prefix, _, _ = multi_doc_dataset
    source = MMapIndexedDataSource(prefix, split_sentences=True)
    # -1 should be last document (doc2)
    np.testing.assert_array_equal(
        source[-1]["text"],
        np.array([30, 31, 32, 33, 34, 35], dtype=np.int32),
    )
    # -3 should be first document (doc0)
    np.testing.assert_array_equal(
        source[-3]["text"],
        np.array([10, 11, 12, 13, 14], dtype=np.int32),
    )

  def test_document_index_out_of_range(self, multi_doc_dataset):
    """Out-of-range document index should raise IndexError."""
    prefix, _, _ = multi_doc_dataset
    source = MMapIndexedDataSource(prefix, split_sentences=True)
    with pytest.raises(IndexError, match="out of range"):
      _ = source[3]
    with pytest.raises(IndexError, match="out of range"):
      _ = source[-4]

  def test_empty_document(self, tmp_dir):
    """Empty document (adjacent equal doc_idx entries) returns empty array."""
    seqs = [
        np.array([10, 11], dtype=np.int32),  # doc0
        np.array([20, 21], dtype=np.int32),  # doc2 (doc1 is empty)
    ]
    # doc_boundaries: doc0=[0,1), doc1=[1,1) (empty), doc2=[1,2)
    doc_boundaries = [0, 1, 1, 2]
    prefix = os.path.join(tmp_dir, "empty_doc")
    create_mmap_test_data(prefix, seqs, doc_boundaries=doc_boundaries)
    source = MMapIndexedDataSource(prefix, split_sentences=True)
    assert len(source) == 3
    np.testing.assert_array_equal(source[0]["text"], np.array([10, 11], dtype=np.int32))
    # doc1 is empty
    result = source[1]["text"]
    assert len(result) == 0
    assert result.dtype == np.int32
    # doc2
    np.testing.assert_array_equal(source[2]["text"], np.array([20, 21], dtype=np.int32))

  def test_get_datasets_mmap_split_sentences(self, multi_doc_dataset):
    """get_datasets with mmap_split_sentences=True uses document-level indexing."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    prefix, _, _ = multi_doc_dataset
    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=True),
    )
    items = list(ds)
    assert len(items) == 3  # 3 documents, not 6 sequences


class TestMultiShardMMapIndexedDataSource:
  """Tests for concatenating multiple mmap shards behind one Grain source."""

  def test_len_getitem_negative_index_and_token_counts(self, tmp_dir):
    prefix_a = create_mmap_test_data(
        os.path.join(tmp_dir, "shard_a"),
        [
            np.array([1, 2], dtype=np.int32),
            np.array([3], dtype=np.int32),
        ],
        doc_boundaries=[0, 1, 2],
    )
    prefix_b = create_mmap_test_data(
        os.path.join(tmp_dir, "shard_b"),
        [np.array([10, 11, 12], dtype=np.int32)],
        doc_boundaries=[0, 1],
    )

    source = MultiShardMMapIndexedDataSource([prefix_a, prefix_b])
    assert len(source) == 3
    np.testing.assert_array_equal(source[0]["text"], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(source[2]["text"], np.array([10, 11, 12], dtype=np.int32))
    np.testing.assert_array_equal(source[-1]["text"], np.array([10, 11, 12], dtype=np.int32))
    np.testing.assert_array_equal(source.doc_token_counts(), np.array([2, 1, 3], dtype=np.int64))

  def test_out_of_range_raises(self, tmp_dir):
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "shard"), [np.array([1], dtype=np.int32)])
    source = MultiShardMMapIndexedDataSource([prefix])
    with pytest.raises(IndexError, match="out of range"):
      _ = source[1]
    with pytest.raises(IndexError, match="out of range"):
      _ = source[-2]

  def test_pickle_roundtrip(self, tmp_dir):
    prefix_a = create_mmap_test_data(os.path.join(tmp_dir, "a"), [np.array([1], dtype=np.int32)])
    prefix_b = create_mmap_test_data(os.path.join(tmp_dir, "b"), [np.array([2], dtype=np.int32)])
    restored = pickle.loads(pickle.dumps(MultiShardMMapIndexedDataSource([prefix_a, prefix_b])))
    assert len(restored) == 2
    np.testing.assert_array_equal(restored[1]["text"], np.array([2], dtype=np.int32))


class TestMMapSampleIndexDataSource:
  """Tests for fixed-length windowing over raw mmap documents."""

  def test_rejects_non_positive_sequence_length(self, simple_dataset):
    prefix, _ = simple_dataset
    inner = MMapIndexedDataSource(prefix)
    with pytest.raises(ValueError, match="seq_length must be positive"):
      MMapSampleIndexDataSource(inner, seq_length=0, eod_id=0)

  def test_drop_last_false_pads_tail_with_eod(self, tmp_dir):
    prefix = create_mmap_test_data(
        os.path.join(tmp_dir, "sample_windows"),
        [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ],
        doc_boundaries=[0, 1, 2],
    )
    inner = MMapIndexedDataSource(prefix, split_sentences=True)
    source = MMapSampleIndexDataSource(inner, seq_length=4, eod_id=99, drop_last=False)

    assert len(source) == 2
    np.testing.assert_array_equal(source[0]["text"], np.array([1, 2, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(source[1]["text"], np.array([5, 99, 99, 99], dtype=np.int32))
    np.testing.assert_array_equal(source[-1]["text"], source[1]["text"])
    with pytest.raises(IndexError, match="out of range"):
      _ = source[2]

  def test_pickle_roundtrip_preserves_windowing(self, tmp_dir):
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "pickle_sample"), [np.array([1, 2, 3, 4], dtype=np.int32)])
    source = MMapSampleIndexDataSource(MMapIndexedDataSource(prefix), seq_length=2, eod_id=0)
    restored = pickle.loads(pickle.dumps(source))
    assert len(restored) == 2
    np.testing.assert_array_equal(restored[1]["text"], np.array([3, 4], dtype=np.int32))


class TestMMapPatternParsing:
  """Unit coverage for mmap and mmap_npy pattern parsers."""

  def test_weighted_mixture_uses_last_comma_and_filters_zero_weights(self):
    specs, weights = _parse_weighted_mixture("npy|bin_a:bin_b,0.0;npy|bin,c,3.0", "mmap_npy")
    assert specs == ["npy|bin,c"]
    np.testing.assert_allclose(weights, [1.0])

  def test_weighted_mixture_rejects_empty_spec(self):
    with pytest.raises(ValueError, match="Empty spec"):
      _parse_weighted_mixture(",1.0", "mmap")

  def test_weighted_mixture_rejects_bad_weight_values(self):
    with pytest.raises(ValueError, match="Invalid weight"):
      _parse_weighted_mixture("path,not-a-number", "mmap")
    with pytest.raises(ValueError, match="Negative weight"):
      _parse_weighted_mixture("path,-1.0", "mmap")
    with pytest.raises(ValueError, match="Total weight"):
      _parse_weighted_mixture("a,0;b,0", "mmap")

  def test_mmap_npy_spec_trims_paths(self):
    npy_dir, bin_paths = _parse_mmap_npy_spec(" /tmp/npy | /tmp/a : /tmp/b ")
    assert npy_dir == "/tmp/npy"
    assert bin_paths == ["/tmp/a", "/tmp/b"]

  def test_mmap_npy_spec_requires_exactly_one_separator(self):
    with pytest.raises(ValueError, match="mmap_npy spec"):
      _parse_mmap_npy_spec("no-separator")
    with pytest.raises(ValueError, match="mmap_npy spec"):
      _parse_mmap_npy_spec("a|b|c")


# ===========================================================================
# Integration tests: MMap with Grain pipeline
# ===========================================================================


class TestMMapGrainPipeline:
  """Integration tests for MMap datasets within the Grain pipeline."""

  def test_get_datasets_mmap(self, simple_dataset):
    """get_datasets returns an iterable dataset for mmap type."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    prefix, seqs = simple_dataset
    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
    )
    items = []
    for item in ds:
      items.append(item)
      if len(items) >= len(seqs):
        break
    assert len(items) == len(seqs)

  def test_get_datasets_mmap_with_weights(self, tmp_dir):
    """Weighted mixture of two mmap datasets."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seqs1 = [np.array([1, 2, 3], dtype=np.int32)]
    seqs2 = [np.array([4, 5, 6], dtype=np.int32)]
    p1 = create_mmap_test_data(os.path.join(tmp_dir, "ds1"), seqs1)
    p2 = create_mmap_test_data(os.path.join(tmp_dir, "ds2"), seqs2)

    pattern = f"{p1},0.5;{p2},0.5"
    ds = get_datasets(
        data_file_pattern=pattern,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
    )
    items = []
    for item in ds:
      items.append(item)
      if len(items) >= 4:
        break
    assert len(items) > 0

  def test_shuffle_determinism(self, tmp_dir):
    """Same seed produces same order."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seqs = [np.array([i], dtype=np.int32) for i in range(20)]
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "det"), seqs)

    def get_order(seed):
      ds = get_datasets(
          data_file_pattern=prefix,
          data_file_type="mmap",
          shuffle=True,
          shuffle_seed=seed,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
      )
      items = []
      for item in ds:
        items.append(item["text"][0])
        if len(items) >= 20:
          break
      return items

    order1 = get_order(42)
    order2 = get_order(42)
    assert order1 == order2

  def test_multi_host_shard_no_overlap(self, tmp_dir):
    """Two hosts with shard 0/2 and 1/2 produce disjoint data."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seqs = [np.array([i], dtype=np.int32) for i in range(10)]
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "shard"), seqs)

    def get_items(host_index, host_count):
      ds = get_datasets(
          data_file_pattern=prefix,
          data_file_type="mmap",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=host_index,
          dataloading_host_count=host_count,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
      )
      return [item["text"][0] for item in ds]

    host0 = get_items(0, 2)
    host1 = get_items(1, 2)
    # No overlap
    assert set(host0).isdisjoint(set(host1))
    # Together cover all items
    assert sorted(host0 + host1) == list(range(10))

  def test_num_epoch_repeats(self, tmp_dir):
    """num_epoch=2 yields twice the data."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seqs = [np.array([i], dtype=np.int32) for i in range(5)]
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "epoch"), seqs)

    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=2,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
    )
    items = list(ds)
    assert len(items) == 10

  def test_mixture_missing_weight_raises(self, tmp_dir):
    """Malformed mixture pattern (missing weight) raises ValueError."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seqs = [np.array([1], dtype=np.int32)]
    p1 = create_mmap_test_data(os.path.join(tmp_dir, "m1"), seqs)

    with pytest.raises(ValueError, match="Malformed mmap mixture"):
      get_datasets(
          data_file_pattern=f"{p1};{p1}",  # missing weights
          data_file_type="mmap",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
      )

  def test_mixture_negative_weight_raises(self, tmp_dir):
    """Negative weight in mixture raises ValueError."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seqs = [np.array([1], dtype=np.int32)]
    p1 = create_mmap_test_data(os.path.join(tmp_dir, "m2"), seqs)
    p2 = create_mmap_test_data(os.path.join(tmp_dir, "m3"), seqs)

    with pytest.raises(ValueError, match="Negative weight"):
      get_datasets(
          data_file_pattern=f"{p1},0.5;{p2},-0.5",
          data_file_type="mmap",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
      )

  def test_mixture_zero_total_weight_raises(self, tmp_dir):
    """All-zero weights in mixture raises ValueError."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seqs = [np.array([1], dtype=np.int32)]
    p1 = create_mmap_test_data(os.path.join(tmp_dir, "m4"), seqs)
    p2 = create_mmap_test_data(os.path.join(tmp_dir, "m5"), seqs)

    with pytest.raises(ValueError, match="Total weight"):
      get_datasets(
          data_file_pattern=f"{p1},0.0;{p2},0.0",
          data_file_type="mmap",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
      )

  def test_mixture_invalid_weight_string_raises(self, tmp_dir):
    """Non-numeric weight string raises ValueError."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seqs = [np.array([1], dtype=np.int32)]
    p1 = create_mmap_test_data(os.path.join(tmp_dir, "m6"), seqs)
    p2 = create_mmap_test_data(os.path.join(tmp_dir, "m7"), seqs)

    with pytest.raises(ValueError, match="not a valid number"):
      get_datasets(
          data_file_pattern=f"{p1},abc;{p2},0.5",
          data_file_type="mmap",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(max_target_length=0, eod_id=0, mmap_split_sentences=False),
      )


# ===========================================================================
# Megatron compatibility tests
# ===========================================================================


class TestMegatronCompatibility:
  """Tests verifying wire-format compatibility with Megatron-Core."""

  def test_dtype_code_table_complete(self):
    """All Megatron dtype codes 1-8 are mapped."""
    for code in range(1, 9):
      assert code in DTYPE_CODES

  def test_dtype_code_inverse_roundtrip(self):
    """DTYPE_CODES and DTYPE_CODES_INV are consistent."""
    for code, dtype in DTYPE_CODES.items():
      assert DTYPE_CODES_INV[dtype] == code

  def test_raw_binary_float64_compatibility(self, tmp_dir):
    """Manually construct a float64 dataset with dtype code 6 and verify."""
    prefix = os.path.join(tmp_dir, "compat_f64")
    data = np.array([1.1, 2.2, 3.3], dtype=np.float64)

    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 6))  # float64
      f.write(struct.pack("<Q", 1))  # 1 sequence
      f.write(struct.pack("<Q", 1))  # 1 document
      f.write(np.array([3], dtype=np.int32).tobytes())  # sizes
      f.write(np.array([0], dtype=np.int64).tobytes())  # pointers
      f.write(np.array([0, 1], dtype=np.int64).tobytes())  # doc_idx

    with open(prefix + ".bin", "wb") as f:
      f.write(data.tobytes())

    ds = MMapIndexedDataset(prefix)
    assert ds.dtype == np.float64
    np.testing.assert_allclose(ds[0], data)

  def test_raw_binary_float32_compatibility(self, tmp_dir):
    """Manually construct a float32 dataset with dtype code 7 and verify."""
    prefix = os.path.join(tmp_dir, "compat_f32")
    data = np.array([4.0, 5.0], dtype=np.float32)

    with open(prefix + ".idx", "wb") as f:
      f.write(MMAP_INDEX_MAGIC)
      f.write(struct.pack("<Q", MMAP_INDEX_VERSION))
      f.write(struct.pack("<B", 7))  # float32
      f.write(struct.pack("<Q", 1))  # 1 sequence
      f.write(struct.pack("<Q", 1))  # 1 document
      f.write(np.array([2], dtype=np.int32).tobytes())
      f.write(np.array([0], dtype=np.int64).tobytes())
      f.write(np.array([0, 1], dtype=np.int64).tobytes())

    with open(prefix + ".bin", "wb") as f:
      f.write(data.tobytes())

    ds = MMapIndexedDataset(prefix)
    assert ds.dtype == np.float32
    np.testing.assert_allclose(ds[0], data)


# ===========================================================================
# Megatron sample_index / shuffle_index semantic alignment tests
# ===========================================================================


class TestMMapPipelineSemantics:
  """Tests verifying that Grain + MMap pipeline produces semantically
  equivalent results to Megatron's document_index -> sample_index ->
  shuffle_index construction.

  Megatron's approach (GPTDataset):
    1. Shuffle documents (document_index)
    2. Concatenate all tokens in document order, then slice into fixed-length
       samples (sample_index) -- each sample is exactly `seq_length + 1` tokens
    3. Shuffle samples (shuffle_index)

  Grain + ConcatThenSplit equivalent:
    1. MMapIndexedDataSource with split_sentences=True -> documents
    2. Grain shuffle -> shuffled documents
    3. ConcatThenSplitIterDataset -> fixed-length samples
  """

  def _build_documents(self, tmp_dir, rng_seed=0):
    """Create a dataset with 5 documents of varying lengths.

    Returns (path_prefix, doc_tokens_list) where doc_tokens_list[i]
    is the full token array for document i.
    """
    rng = np.random.RandomState(rng_seed)
    # 5 documents, each with 1-3 sentences
    docs = []
    seqs = []
    doc_boundaries = [0]
    for _ in range(5):
      num_sents = rng.randint(1, 4)
      doc_tokens = []
      for _ in range(num_sents):
        sent_len = rng.randint(5, 15)
        sent = rng.randint(1, 1000, size=sent_len).astype(np.int32)
        seqs.append(sent)
        doc_tokens.append(sent)
      docs.append(np.concatenate(doc_tokens))
      doc_boundaries.append(len(seqs))

    prefix = os.path.join(tmp_dir, "megatron_align")
    create_mmap_test_data(prefix, seqs, doc_boundaries=doc_boundaries)
    return prefix, docs

  def test_all_tokens_preserved_through_pipeline(self, tmp_dir):
    """All source tokens appear in the ConcatThenSplit output (no data loss)."""
    prefix, docs = self._build_documents(tmp_dir)

    source = MMapIndexedDataSource(prefix, split_sentences=True)
    ds = grain.MapDataset.source(source)
    ds = ds.shuffle(seed=42)
    ds = ds.to_iter_dataset(read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1))

    # Collect all tokens from the shuffled document stream
    all_tokens = []
    for item in ds:
      all_tokens.extend(item["text"].tolist())

    # All source tokens must be present
    expected_tokens = []
    for doc in docs:
      expected_tokens.extend(doc.tolist())

    assert sorted(all_tokens) == sorted(expected_tokens)

  def test_document_integrity_after_shuffle(self, tmp_dir):
    """Each shuffled element is a complete document (no partial documents)."""
    prefix, docs = self._build_documents(tmp_dir)

    source = MMapIndexedDataSource(prefix, split_sentences=True)
    ds = grain.MapDataset.source(source)
    ds = ds.shuffle(seed=123)
    ds = ds.to_iter_dataset(read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1))

    items = list(ds)
    assert len(items) == len(docs)

    # Each item must match exactly one document (order may differ)
    doc_set = {tuple(d.tolist()) for d in docs}
    item_set = {tuple(item["text"].tolist()) for item in items}
    assert doc_set == item_set

  def test_concat_then_split_fixed_length_samples(self, tmp_dir):
    """ConcatThenSplit on shuffled documents produces fixed-length samples,
    mirroring Megatron's sample_index construction."""
    prefix, _ = self._build_documents(tmp_dir)
    seq_length = 8  # fixed sample length

    source = MMapIndexedDataSource(prefix, split_sentences=True)
    ds = grain.MapDataset.source(source)
    ds = ds.shuffle(seed=7)

    # Rekey to match pipeline expectations
    from maxtext.input_pipeline import input_pipeline_utils  # pylint: disable=import-outside-toplevel

    ds = ds.map(input_pipeline_utils.KeepFeatures(feature_names=["text"], tokenize=False))
    rekey_dict = {"inputs": "text", "targets": "text"}
    ds = ds.map(input_pipeline_utils.Rekey(rekey_dict))

    ds = ds.to_iter_dataset(read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1))
    length_struct = {"inputs": seq_length, "targets": seq_length}
    ds = grain.experimental.ConcatThenSplitIterDataset(ds, length_struct=length_struct)

    samples = []
    for sample in ds:
      samples.append(sample)
      if len(samples) >= 20:
        break

    assert len(samples) > 0
    for sample in samples:
      assert sample["inputs"].shape == (seq_length,)
      assert sample["targets"].shape == (seq_length,)

  def test_shuffle_seed_changes_document_order(self, tmp_dir):
    """Different seeds produce different document orderings
    (confirms shuffle is effective, like Megatron's document_index)."""
    prefix, _ = self._build_documents(tmp_dir)

    def get_doc_order(seed):
      source = MMapIndexedDataSource(prefix, split_sentences=True)
      ds = grain.MapDataset.source(source)
      ds = ds.shuffle(seed=seed)
      ds = ds.to_iter_dataset(read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1))
      return [tuple(item["text"].tolist()) for item in ds]

    order_a = get_doc_order(1)
    order_b = get_doc_order(2)
    # Same documents
    assert sorted(order_a) == sorted(order_b)
    # But different order (with very high probability for 5 docs)
    assert order_a != order_b


# ===========================================================================
# End-to-end pipeline tests: pretrain_preprocessing_pipeline + mmap
# ===========================================================================


class _FakeTokenizer:
  """Minimal tokenizer stub for pipeline tests (avoids loading real models)."""

  pad_id = 0
  unk_id = 1
  eos_id = 3


class TestMMapPretrainPipeline:
  """End-to-end tests exercising pretrain_preprocessing_pipeline with mmap data,
  including split_sentences mode. These test the full path from
  get_datasets -> pretrain_preprocessing_pipeline -> batched output."""

  @pytest.fixture(autouse=True)
  def _mock_tokenizer(self, monkeypatch):
    """Patch build_tokenizer to avoid loading real sentencepiece models."""
    from maxtext.input_pipeline import data_processing_utils  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(
        data_processing_utils.tokenizer,
        "build_tokenizer",
        lambda *args, **kwargs: _FakeTokenizer(),
    )

  @staticmethod
  def _make_config(
      tmp_dir,
      prefix,
      split_sentences=False,
      packing=False,
      max_target_length=16,
      batch_size=2,
  ):
    """Build a minimal ml_collections.ConfigDict for pipeline testing."""
    import ml_collections  # pylint: disable=import-outside-toplevel

    config = ml_collections.ConfigDict()
    config.grain_file_type = "mmap"
    config.grain_train_files = prefix
    config.tokenizer_path = "unused"
    config.tokenizer_type = "sentencepiece"
    config.add_bos = False
    config.add_eos = False
    config.hf_access_token = ""
    config.dataset_type = "grain"
    config.tokenize_train_data = False
    config.train_data_columns = ["text"]
    config.max_target_length = max_target_length
    config.global_batch_size_to_load = batch_size
    config.expansion_factor_real_data = 1
    config.elastic_enabled = False
    config.packing = packing
    config.grain_packing_type = "concat_then_split"
    config.max_segments_per_seq = None
    config.grain_ram_budget_mb = 256
    config.mmap_split_sentences = split_sentences
    config.use_truncation = False
    config.mmap_eod_id = 0
    config.reset_attention_mask = False
    config.eod_mask_loss = False
    return config

  def test_no_packing_output_shape(self, tmp_dir):
    """Full pipeline without packing produces correctly shaped batches."""
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    max_len = 8
    batch_size = 2
    # 4 sequences, each long enough to pad/trim to max_len
    seqs = [np.arange(10, 10 + max_len + 5, dtype=np.int32) for _ in range(4)]
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "e2e_nopack"), seqs)
    config = self._make_config(tmp_dir, prefix, max_target_length=max_len, batch_size=batch_size)

    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=max_len, eod_id=0, mmap_split_sentences=False),
    )
    pipeline = pretrain_preprocessing_pipeline(
        ds,
        config,
        data_columns=["text"],
        tokenize=False,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )

    batch = next(iter(pipeline))
    assert "inputs" in batch
    assert "targets" in batch
    assert batch["inputs"].shape == (batch_size, max_len)
    assert batch["targets"].shape == (batch_size, max_len)

  def test_packing_concat_then_split_output(self, tmp_dir):
    """Full pipeline with concat_then_split packing produces fixed-length packed samples."""
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    max_len = 8
    batch_size = 2
    # Many short sequences to ensure concat_then_split has enough data
    seqs = [np.arange(i * 10, i * 10 + 6, dtype=np.int32) for i in range(20)]
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "e2e_pack"), seqs)
    config = self._make_config(
        tmp_dir,
        prefix,
        packing=True,
        max_target_length=max_len,
        batch_size=batch_size,
    )

    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=max_len, eod_id=0, mmap_split_sentences=False),
    )
    pipeline = pretrain_preprocessing_pipeline(
        ds,
        config,
        data_columns=["text"],
        tokenize=False,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )

    batch = next(iter(pipeline))
    assert batch["inputs"].shape == (batch_size, max_len)
    assert batch["targets"].shape == (batch_size, max_len)
    # concat_then_split produces segmentation keys
    assert "inputs_segmentation" in batch or "inputs_segment_ids" in batch

  def test_split_sentences_packing_end_to_end(self, tmp_dir):
    """Full pipeline with split_sentences=True + concat_then_split packing.
    Verifies that document-level shuffling feeds into packing correctly."""
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    max_len = 8
    batch_size = 2
    # 3 documents with multiple sentences each
    seqs = [
        np.array([10, 11, 12], dtype=np.int32),  # doc0 sent0
        np.array([13, 14], dtype=np.int32),  # doc0 sent1
        np.array([20, 21, 22, 23], dtype=np.int32),  # doc1 sent0
        np.array([30, 31], dtype=np.int32),  # doc2 sent0
        np.array([32, 33, 34], dtype=np.int32),  # doc2 sent1
        np.array([35, 36, 37, 38], dtype=np.int32),  # doc2 sent2
    ]
    doc_boundaries = [0, 2, 3, 6]
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "e2e_split"), seqs, doc_boundaries=doc_boundaries)
    config = self._make_config(
        tmp_dir,
        prefix,
        split_sentences=True,
        packing=True,
        max_target_length=max_len,
        batch_size=batch_size,
    )

    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=True,
        shuffle_seed=42,
        shuffle_buffer_size=0,
        num_epoch=2,  # repeat to get enough data for batching
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=max_len, eod_id=0, mmap_split_sentences=True),
    )
    pipeline = pretrain_preprocessing_pipeline(
        ds,
        config,
        data_columns=["text"],
        tokenize=False,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )

    batch = next(iter(pipeline))
    assert batch["inputs"].shape == (batch_size, max_len)
    assert batch["targets"].shape == (batch_size, max_len)

  def test_no_packing_with_mp_prefetch(self, tmp_dir):
    """Full pipeline with grain_worker_count=2 (mp_prefetch enabled).

    Verifies that the mmap path — which places mp_prefetch AFTER batch +
    ShiftData — produces correct output when multiprocessing is active.
    """
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    max_len = 8
    batch_size = 2
    # Many sequences to give mp_prefetch meaningful work
    seqs = [np.arange(10 + i * 20, 10 + i * 20 + max_len + 5, dtype=np.int32) for i in range(20)]
    prefix = create_mmap_test_data(os.path.join(tmp_dir, "e2e_mp"), seqs)
    config = self._make_config(tmp_dir, prefix, max_target_length=max_len, batch_size=batch_size)

    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=max_len, eod_id=0, mmap_split_sentences=False),
    )
    pipeline = pretrain_preprocessing_pipeline(
        ds,
        config,
        data_columns=["text"],
        tokenize=False,
        grain_worker_count=2,
        grain_per_worker_buffer_size=2,
    )

    batch = next(iter(pipeline))
    assert batch["inputs"].shape == (batch_size, max_len)
    assert batch["targets"].shape == (batch_size, max_len)

  def test_split_sentences_with_mp_prefetch(self, tmp_dir):
    """Full pipeline with split_sentences + grain_worker_count=2.

    Exercises the mmap path's mp_prefetch with document-level splitting.
    """
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    max_len = 16
    batch_size = 2
    # 8 documents, each single-sequence, long enough for splitting
    seqs = [np.arange(i * 100, i * 100 + 20, dtype=np.int32) for i in range(8)]
    doc_boundaries = list(range(len(seqs) + 1))
    prefix = create_mmap_test_data(
        os.path.join(tmp_dir, "e2e_split_mp"),
        seqs,
        doc_boundaries=doc_boundaries,
    )
    config = self._make_config(
        tmp_dir,
        prefix,
        split_sentences=True,
        packing=False,
        max_target_length=max_len,
        batch_size=batch_size,
    )

    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=max_len, eod_id=0, mmap_split_sentences=True),
    )
    pipeline = pretrain_preprocessing_pipeline(
        ds,
        config,
        data_columns=["text"],
        tokenize=False,
        grain_worker_count=2,
        grain_per_worker_buffer_size=2,
    )

    batch = next(iter(pipeline))
    assert batch["inputs"].shape == (batch_size, max_len)
    assert batch["targets"].shape == (batch_size, max_len)

  def test_split_sentences_no_packing_end_to_end(self, tmp_dir):
    """Full pipeline with split_sentences=True, no packing.
    Documents are padded/trimmed to max_target_length."""
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    max_len = 16
    batch_size = 2
    # 4 documents, each is a single sequence long enough
    seqs = [np.arange(i * 100, i * 100 + 20, dtype=np.int32) for i in range(4)]
    doc_boundaries = [0, 1, 2, 3, 4]
    prefix = create_mmap_test_data(
        os.path.join(tmp_dir, "e2e_split_nopack"),
        seqs,
        doc_boundaries=doc_boundaries,
    )
    config = self._make_config(
        tmp_dir,
        prefix,
        split_sentences=True,
        packing=False,
        max_target_length=max_len,
        batch_size=batch_size,
    )

    ds = get_datasets(
        data_file_pattern=prefix,
        data_file_type="mmap",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=max_len, eod_id=0, mmap_split_sentences=True),
    )
    pipeline = pretrain_preprocessing_pipeline(
        ds,
        config,
        data_columns=["text"],
        tokenize=False,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )

    batch = next(iter(pipeline))
    assert batch["inputs"].shape == (batch_size, max_len)
    assert batch["targets"].shape == (batch_size, max_len)


# ===========================================================================
# Tests: MMapSampleIndexDataSource does NOT insert EOD (mmap path)
# ===========================================================================


class TestMMapSampleIndexNoEodInsertion:
  """Verify that MMapSampleIndexDataSource reads raw tokens without
  inserting or removing EOD tokens.  EOD presence relies on preprocessing
  with --append-eod."""

  @pytest.fixture
  def tmp_dir(self, tmp_path):
    return str(tmp_path)

  def test_eod_from_data_preserved(self, tmp_dir):
    """Docs preprocessed with --append-eod: EOD appears in output from
    raw data (not inserted by dataloader), and no double EOD."""

    eod_id = 0
    # Documents already contain trailing EOD (--append-eod)
    prefix = create_mmap_test_data(
        os.path.join(tmp_dir, "data"),
        sequences=[
            np.array([10, 11, 12, eod_id], dtype=np.int32),
            np.array([20, 21, eod_id], dtype=np.int32),
            np.array([30, 31, 32, eod_id], dtype=np.int32),
        ],
        doc_boundaries=[0, 1, 2, 3],
    )

    inner = MMapIndexedDataSource(prefix, split_sentences=True)
    seq_length = 8
    ds = MMapSampleIndexDataSource(
        inner_source=inner,
        seq_length=seq_length,
        eod_id=eod_id,
    )

    # Collect all content tokens (non-padding) from all samples
    all_tokens = []
    for sample in ds:
      all_tokens.extend(sample["text"].tolist())

    # The raw concatenation is [10,11,12,0, 20,21,0, 30,31,32,0] = 11 tokens
    expected = [10, 11, 12, eod_id, 20, 21, eod_id, 30, 31, 32, eod_id]
    # With seq_length=8, drop_last=True: 11//8 = 1 sample of 8 tokens
    assert len(all_tokens) == seq_length
    assert all_tokens == expected[:seq_length]

    # No double EOD in non-padding region
    for j in range(len(all_tokens) - 1):
      if all_tokens[j] == eod_id and all_tokens[j + 1] == eod_id:
        remaining = all_tokens[j:]
        if not all(t == eod_id for t in remaining):
          raise AssertionError(f"Double EOD at positions {j},{j+1} (not trailing pad): {all_tokens}")

  def test_no_eod_insertion_when_docs_lack_eod(self, tmp_dir):
    """Docs without trailing EOD: verify NO eod is inserted by the
    dataloader, and a warning is emitted."""
    import logging  # pylint: disable=import-outside-toplevel

    eod_id = 0
    prefix = create_mmap_test_data(
        os.path.join(tmp_dir, "data"),
        sequences=[
            np.array([10, 11, 12], dtype=np.int32),
            np.array([20, 21], dtype=np.int32),
            np.array([30, 31, 32], dtype=np.int32),
        ],
        doc_boundaries=[0, 1, 2, 3],
    )

    inner = MMapIndexedDataSource(prefix, split_sentences=True)
    seq_length = 8

    # Capture warning log
    import maxtext.input_pipeline._mmap_datasource as mmap_mod  # pylint: disable=import-outside-toplevel

    with mock.patch.object(logging.getLogger(mmap_mod.__name__), "warning") as mock_warn:
      ds = MMapSampleIndexDataSource(
          inner_source=inner,
          seq_length=seq_length,
          eod_id=eod_id,
      )
      # Verify warning was emitted about missing EOD
      mock_warn.assert_called_once()
      assert "does NOT insert EOD" in mock_warn.call_args[0][0]

    # Collect all content tokens — raw data is [10,11,12, 20,21, 30,31,32] = 8 tokens
    # 8 // 8 = 1 sample
    all_tokens = []
    for sample in ds:
      all_tokens.extend(sample["text"].tolist())

    expected = [10, 11, 12, 20, 21, 30, 31, 32]
    assert all_tokens == expected
    # No EOD was inserted
    assert eod_id not in all_tokens

  def test_num_samples_exact_with_append_eod(self, tmp_dir):
    """With --append-eod docs, len(ds) == total_raw_tokens // seq_length
    (no inflation from +1 per doc)."""

    eod_id = 0
    sequences = [
        np.array([10, 11, 12, eod_id], dtype=np.int32),  # 4 tokens
        np.array([20, 21, eod_id], dtype=np.int32),  # 3 tokens
        np.array([30, 31, 32, eod_id], dtype=np.int32),  # 4 tokens
    ]
    # Total raw tokens: 4 + 3 + 4 = 11
    prefix = create_mmap_test_data(
        os.path.join(tmp_dir, "data"),
        sequences=sequences,
        doc_boundaries=[0, 1, 2, 3],
    )

    inner = MMapIndexedDataSource(prefix, split_sentences=True)
    seq_length = 4
    ds = MMapSampleIndexDataSource(
        inner_source=inner,
        seq_length=seq_length,
        eod_id=eod_id,
    )

    total_raw_tokens = sum(len(s) for s in sequences)
    assert total_raw_tokens == 11
    assert len(ds) == total_raw_tokens // seq_length  # 11 // 4 = 2

  def test_cross_boundary_no_token_overlap(self, tmp_dir):
    """Adjacent samples have no overlapping content tokens."""

    eod_id = 0
    prefix = create_mmap_test_data(
        os.path.join(tmp_dir, "data"),
        sequences=[
            np.array([10, 11, 12, eod_id], dtype=np.int32),
            np.array([20, 21, 22, 23, eod_id], dtype=np.int32),
            np.array([30, 31, eod_id], dtype=np.int32),
        ],
        doc_boundaries=[0, 1, 2, 3],
    )

    inner = MMapIndexedDataSource(prefix, split_sentences=True)
    seq_length = 4
    ds = MMapSampleIndexDataSource(
        inner_source=inner,
        seq_length=seq_length,
        eod_id=eod_id,
    )

    # Total raw tokens: 4 + 5 + 3 = 12, so 12 // 4 = 3 samples
    assert len(ds) == 3
    samples = [ds[i]["text"].tolist() for i in range(len(ds))]

    # Concatenation should reproduce raw stream exactly
    raw = [10, 11, 12, eod_id, 20, 21, 22, 23, eod_id, 30, 31, eod_id]
    flat = []
    for s in samples:
      flat.extend(s)
    assert flat == raw

    # Adjacent samples share no tokens
    for i in range(len(samples) - 1):
      assert samples[i] != samples[i + 1], f"Samples {i} and {i+1} are identical"


# ===========================================================================
# Grain integration and multiprocessing
# ===========================================================================


@pytest.mark.cpu_only
class GrainMmapNpyEvalConfigTest(TestCase):
  """Regression coverage for the bounded mmap_npy eval stream."""

  @staticmethod
  def _config(mmap_npy_split):
    return SimpleNamespace(
        global_batch_size_to_load=8,
        global_batch_size_to_load_eval=4,
        max_target_length=128,
        grain_file_type="mmap_npy",
        eval_steps=3,
        eval_interval=2,
        steps=10,
        data_shuffle_seed=42,
        mmap_npy_split=mmap_npy_split,
        mmap_eod_id=0,
        mmap_split_sentences=False,
        blend_cache_dir="",
        blend_index_dir="",
        grain_eval_files="unused",
        grain_shuffle_buffer_size=0,
        grain_worker_count_eval=0,
        grain_num_threads_eval=1,
        grain_prefetch_buffer_size_eval=1,
        grain_per_worker_buffer_size_eval=1,
        grain_data_source_max_workers=1,
        eval_data_columns=("text",),
        tokenize_eval_data=False,
        colocated_python_data_input=False,
        generate_padding_batch_eval=False,
        use_sft=False,
        use_multimodal=False,
    )

  @staticmethod
  def _capture_dataset_config(config):
    with (
        mock.patch.object(grain_data_processing, "_get_pipeline_fn", return_value=lambda **_kwargs: object()),
        mock.patch.object(grain_data_processing, "get_datasets", return_value=object()) as get_datasets,
        mock.patch.object(grain_data_processing.multihost_dataloading, "MultiHostDataLoadIterator"),
    ):
      grain_data_processing.make_grain_eval_iterator(config, SimpleNamespace(size=1), [0])
    return get_datasets.call_args.kwargs["dataset_config"]

  def test_mmap_npy_eval_pins_all_scheduled_eval_samples(self):
    dataset_config = self._capture_dataset_config(self._config("99,1"))
    # ceil(10 / 2) eval rounds * 3 eval steps * 4 samples per batch.
    self.assertEqual(dataset_config.num_samples, 60)
    self.assertEqual(dataset_config.split_index, 1)

  def test_mmap_npy_eval_without_split_uses_train_partition(self):
    dataset_config = self._capture_dataset_config(self._config(""))
    self.assertEqual(dataset_config.split_index, 0)

  def test_non_mmap_file_type_has_no_mmap_dataset_config(self):
    config = self._config("")
    config.grain_file_type = "arrayrecord"
    self.assertIsNone(grain_data_processing._build_dataset_config(config))  # pylint: disable=protected-access

  def test_mmap_dataset_config_copies_runtime_fields(self):
    config = self._config("80,20")
    config.blend_cache_dir = "/tmp/blend-cache"
    config.blend_index_dir = "/tmp/blend-index"

    dataset_config = grain_data_processing._build_dataset_config(  # pylint: disable=protected-access
        config,
        num_samples=123,
        seed=7,
        split_ratio="80,20",
        split_index=1,
    )

    self.assertEqual(dataset_config.max_target_length, 128)
    self.assertEqual(dataset_config.eod_id, 0)
    self.assertEqual(dataset_config.blend_cache_dir, "/tmp/blend-cache")
    self.assertEqual(dataset_config.blend_index_dir, "/tmp/blend-index")
    self.assertEqual(dataset_config.num_samples, 123)
    self.assertEqual(dataset_config.seed, 7)
    self.assertEqual(dataset_config.split_ratio, "80,20")
    self.assertEqual(dataset_config.split_index, 1)

  def test_mmap_rejects_elastic_iterator(self):
    with self.assertRaisesRegex(ValueError, "elastic_iterator"):
      grain_data_processing.get_datasets(
          data_file_pattern="unused",
          data_file_type="mmap",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          elastic=True,
          dataset_config=grain_data_processing.MMapDatasetConfig(
              max_target_length=8, eod_id=0, mmap_split_sentences=False
          ),
      )

  def test_mmap_npy_rejects_elastic_iterator(self):
    with self.assertRaisesRegex(ValueError, "elastic_iterator"):
      grain_data_processing.get_datasets(
          data_file_pattern="unused",
          data_file_type="mmap_npy",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          elastic=True,
          dataset_config=grain_data_processing.MMapDatasetConfig(
              max_target_length=8, eod_id=0, mmap_split_sentences=False
          ),
      )


class _RecordingDataset:
  """Minimal dataset double that records Grain transform order."""

  def __init__(self):
    self.ops = []

  def map(self, transform):
    self.ops.append(("map", type(transform).__name__))
    return self

  def batch(self, batch_size, batch_fn):
    self.ops.append(("batch", batch_size, batch_fn))
    return self

  def mp_prefetch(self, options):
    self.ops.append(("mp_prefetch", options))
    return self


@pytest.mark.cpu_only
class GrainMmapPretrainPipelineTest(TestCase):
  """Pure unit coverage for mmap/mmap_npy pretraining pipeline branches."""

  @staticmethod
  def _config(grain_file_type, eod_mask_loss=False):
    return SimpleNamespace(
        grain_file_type=grain_file_type,
        mmap_eod_id=99,
        reset_attention_mask=True,
        eod_mask_loss=eod_mask_loss,
        max_target_length=8,
        packing_max_segments_per_sample=25,
    )

  def test_mmap_pipeline_rekeys_generates_segments_shifts_then_prefetches(self):
    dataset = _RecordingDataset()
    with (
        mock.patch.object(grain_data_processing.data_processing_utils, "get_local_batch_size", return_value=2),
        mock.patch.object(grain_data_processing, "_make_mmap_multiprocessing_options", return_value="mp-options"),
        mock.patch.object(grain_data_processing.max_logging, "log") as log,
    ):
      result = grain_data_processing.pretrain_preprocessing_pipeline(
          dataset,
          self._config("mmap", eod_mask_loss=False),
          data_columns=("text",),
          tokenize=False,
          grain_worker_count=0,
          grain_per_worker_buffer_size=1,
      )

    self.assertIs(result, dataset)
    self.assertEqual([op[0] for op in dataset.ops], ["map", "map", "batch", "map", "mp_prefetch"])
    self.assertEqual([op[1] for op in dataset.ops if op[0] == "map"], ["Rekey", "GenerateDocSegmentIds", "ShiftData"])
    self.assertEqual(dataset.ops[-1], ("mp_prefetch", "mp-options"))
    log.assert_called_once()

  def test_mmap_npy_pipeline_splits_before_prefetch_and_skips_shift(self):
    dataset = _RecordingDataset()
    with (
        mock.patch.object(grain_data_processing.data_processing_utils, "get_local_batch_size", return_value=2),
        mock.patch.object(grain_data_processing, "_make_mmap_multiprocessing_options", return_value="mp-options"),
    ):
      result = grain_data_processing.pretrain_preprocessing_pipeline(
          dataset,
          self._config("mmap_npy"),
          data_columns=("text",),
          tokenize=False,
          grain_worker_count=0,
          grain_per_worker_buffer_size=1,
      )

    self.assertIs(result, dataset)
    self.assertEqual([op[0] for op in dataset.ops], ["map", "mp_prefetch", "batch"])
    self.assertEqual(dataset.ops[0][1], "MegatronSplitInputsTargets")

  def test_mmap_pretrain_requires_one_pretokenized_column(self):
    with self.assertRaisesRegex(AssertionError, "requires exactly one"):
      grain_data_processing.pretrain_preprocessing_pipeline(
          _RecordingDataset(),
          self._config("mmap"),
          data_columns=("inputs", "targets"),
          tokenize=False,
          grain_worker_count=0,
          grain_per_worker_buffer_size=1,
      )

  def test_mmap_multiprocessing_options_can_pick_performance_config(self):
    dataset = object()
    config = SimpleNamespace(grain_ram_budget_mb=256)
    performance_config = SimpleNamespace(multiprocessing_options="picked-options")

    with mock.patch.object(
        grain_data_processing.grain.experimental,
        "pick_performance_config",
        return_value=performance_config,
    ) as pick_performance_config:
      self.assertEqual(
          grain_data_processing._make_mmap_multiprocessing_options(  # pylint: disable=protected-access
              dataset, config, grain_worker_count=-1, grain_per_worker_buffer_size=4
          ),
          "picked-options",
      )

    pick_performance_config.assert_called_once_with(
        ds=dataset,
        ram_budget_mb=256,
        max_workers=None,
        max_buffer_size=None,
    )


# ===========================================================================
# mmap_npy datasource and pipeline behavior
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixture: create sample dataset with .bin/.idx and .npy index files
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset():
  """Create a temp directory with bin_dir and npy_dir containing test data.

  Creates 3 documents: [10,11,12], [20,21], [30,31,32,33]
  with doc_boundaries=[0,1,2,3], then uses convert() to generate
  the .npy index files with seq_length=4, num_epochs=2, seed=42.
  """
  with tempfile.TemporaryDirectory() as tmp_dir:
    bin_dir = os.path.join(tmp_dir, "bin_dir")
    npy_dir = os.path.join(tmp_dir, "npy_dir")
    os.makedirs(bin_dir)
    os.makedirs(npy_dir)

    prefix = os.path.join(bin_dir, "test_data")
    create_mmap_test_data(
        prefix,
        sequences=[
            np.array([10, 11, 12], dtype=np.int32),
            np.array([20, 21], dtype=np.int32),
            np.array([30, 31, 32, 33], dtype=np.int32),
        ],
        doc_boundaries=[0, 1, 2, 3],
    )

    seq_length = 4
    npy_paths = convert(
        input_paths=[prefix],
        output_dir=npy_dir,
        seq_length=seq_length,
        num_epochs=2,
        seed=42,
    )

    yield {
        "bin_dir": bin_dir,
        "npy_dir": npy_dir,
        "prefix": prefix,
        "npy_paths": npy_paths,
        "seq_length": seq_length,
        "tmp_dir": tmp_dir,
    }


# ===========================================================================
# Tests: _resolve_bin_prefixes
# ===========================================================================


class TestResolveBinPrefixes:
  """Tests for the _resolve_bin_prefixes helper function."""

  def test_single_prefix_string(self):
    """A string path to an existing .bin file returns that prefix."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      prefix = os.path.join(tmp_dir, "data")
      create_mmap_test_data(prefix, [np.array([1, 2, 3], dtype=np.int32)])
      result = _resolve_bin_prefixes(prefix)
      assert result == [prefix]

  def test_directory_scans_for_bins(self):
    """A directory path scans for *.bin and returns sorted prefixes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      for name in ["c_shard", "a_shard", "b_shard"]:
        p = os.path.join(tmp_dir, name)
        create_mmap_test_data(p, [np.array([1], dtype=np.int32)])
      result = _resolve_bin_prefixes(tmp_dir)
      expected = sorted([os.path.join(tmp_dir, n) for n in ["c_shard", "a_shard", "b_shard"]])
      assert result == expected

  def test_list_of_prefixes(self):
    """A list of path prefixes returns sorted prefixes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      p1 = os.path.join(tmp_dir, "z_data")
      p2 = os.path.join(tmp_dir, "a_data")
      create_mmap_test_data(p1, [np.array([1], dtype=np.int32)])
      create_mmap_test_data(p2, [np.array([2], dtype=np.int32)])
      result = _resolve_bin_prefixes([p1, p2])
      assert result == sorted([p1, p2])

  def test_list_of_directories(self):
    """A list containing directories scans each for .bin files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      dir1 = os.path.join(tmp_dir, "dir1")
      dir2 = os.path.join(tmp_dir, "dir2")
      os.makedirs(dir1)
      os.makedirs(dir2)
      p1 = os.path.join(dir1, "shard")
      p2 = os.path.join(dir2, "shard")
      create_mmap_test_data(p1, [np.array([1], dtype=np.int32)])
      create_mmap_test_data(p2, [np.array([2], dtype=np.int32)])
      result = _resolve_bin_prefixes([dir1, dir2])
      assert sorted(result) == sorted([p1, p2])

  def test_missing_bin_raises(self):
    """A path that doesn't point to a .bin file raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      with pytest.raises(FileNotFoundError):
        _resolve_bin_prefixes(os.path.join(tmp_dir, "nonexistent"))

  def test_empty_dir_raises(self):
    """An empty directory raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      with pytest.raises(FileNotFoundError):
        _resolve_bin_prefixes(tmp_dir)


# ===========================================================================
# Tests: _discover_npy_indices
# ===========================================================================


class TestDiscoverNpyIndices:
  """Tests for the _discover_npy_indices helper function."""

  def test_discovers_npy_files(self, sample_dataset):
    """Discovers the three .npy index files from a directory."""
    doc_path, sample_path, shuffle_path = _discover_npy_indices(sample_dataset["npy_dir"])
    assert doc_path.endswith("-document_index.npy")
    assert sample_path.endswith("-sample_index.npy")
    assert shuffle_path.endswith("-shuffle_index.npy")
    assert os.path.isfile(doc_path)
    assert os.path.isfile(sample_path)
    assert os.path.isfile(shuffle_path)

  def test_shared_hash_prefix(self, sample_dataset):
    """All three files share the same hash prefix."""
    doc_path, sample_path, shuffle_path = _discover_npy_indices(sample_dataset["npy_dir"])
    doc_prefix = os.path.basename(doc_path).replace("-document_index.npy", "")
    sample_prefix = os.path.basename(sample_path).replace("-sample_index.npy", "")
    shuffle_prefix = os.path.basename(shuffle_path).replace("-shuffle_index.npy", "")
    assert doc_prefix == sample_prefix == shuffle_prefix

  def test_expected_hash_selects_matching_triplet(self, sample_dataset):
    """An explicit hash selects only the matching index triplet."""
    doc_path, sample_path, shuffle_path = _discover_npy_indices(sample_dataset["npy_dir"])
    expected_hash = os.path.basename(doc_path).replace("-document_index.npy", "")

    selected = _discover_npy_indices(sample_dataset["npy_dir"], expected_hash=expected_hash)

    assert selected == (doc_path, sample_path, shuffle_path)

  def test_expected_hash_missing_raises(self, sample_dataset):
    """A stale expected_hash fails instead of silently choosing another triplet."""
    with pytest.raises(FileNotFoundError, match="No index files with hash"):
      _discover_npy_indices(sample_dataset["npy_dir"], expected_hash="missing")

  def test_missing_dir_raises(self):
    """A non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
      _discover_npy_indices("/nonexistent/directory/path")

  def test_missing_files_raises(self):
    """A directory without the expected .npy files raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      with pytest.raises(FileNotFoundError):
        _discover_npy_indices(tmp_dir)

  def test_incomplete_set_raises(self):
    """A directory with only some of the .npy files raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      # Create only one of the three files
      np.save(os.path.join(tmp_dir, "abc123-document_index.npy"), np.array([0]))
      with pytest.raises(FileNotFoundError):
        _discover_npy_indices(tmp_dir)

  def test_ambiguous_triplets_raises(self):
    """Multiple matching hash prefixes in one directory raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      for prefix in ["aaa111", "bbb222"]:
        for suffix in ["document_index", "sample_index", "shuffle_index"]:
          np.save(os.path.join(tmp_dir, f"{prefix}-{suffix}.npy"), np.array([0]))
      with pytest.raises(ValueError, match="Ambiguous NPY directory"):
        _discover_npy_indices(tmp_dir)


# ===========================================================================
# Tests: MegatronNpyDataSource init
# ===========================================================================


class TestMegatronNpyDataSourceInit:
  """Tests for MegatronNpyDataSource initialization."""

  def test_load_from_directory(self, sample_dataset):
    """Can load from a directory containing .bin files and a npy_dir."""
    ds = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=sample_dataset["bin_dir"],
        eod_id=0,
        seq_length=sample_dataset["seq_length"],
    )
    assert len(ds) > 0

  def test_load_from_explicit_prefix(self, sample_dataset):
    """Can load from an explicit .bin path prefix."""
    ds = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=sample_dataset["prefix"],
        eod_id=0,
        seq_length=sample_dataset["seq_length"],
    )
    assert len(ds) > 0

  def test_load_from_multi_dir(self, sample_dataset):
    """Can load from a list of paths."""
    ds = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=[sample_dataset["prefix"]],
        eod_id=0,
        seq_length=sample_dataset["seq_length"],
    )
    assert len(ds) > 0

  def test_missing_npy_dir_raises(self, sample_dataset):
    """Raises FileNotFoundError if npy_dir does not exist."""
    with pytest.raises(FileNotFoundError):
      MegatronNpyDataSource(
          npy_dir="/nonexistent/npy/dir",
          bin_paths=sample_dataset["prefix"],
          eod_id=0,
          seq_length=sample_dataset["seq_length"],
      )

  def test_missing_bin_dir_raises(self, sample_dataset):
    """Raises FileNotFoundError if bin_paths points to nonexistent data."""
    with pytest.raises(FileNotFoundError):
      MegatronNpyDataSource(
          npy_dir=sample_dataset["npy_dir"],
          bin_paths="/nonexistent/bin/prefix",
          eod_id=0,
          seq_length=sample_dataset["seq_length"],
      )

  def test_split_sentences_mismatch_raises(self):
    """Raises ValueError when data has split-sentences but config says False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      prefix = os.path.join(tmp_dir, "shard")
      # 2 documents, each with multiple sentences (sequences)
      # doc0: sentence0=[10,11], sentence1=[12,13,14]  → 5 tokens
      # doc1: sentence2=[20,21,22], sentence3=[23]     → 4 tokens
      # doc_boundaries: doc0 spans seq[0:2], doc1 spans seq[2:4]
      create_mmap_test_data(
          prefix,
          sequences=[
              np.array([10, 11], dtype=np.int32),
              np.array([12, 13, 14], dtype=np.int32),
              np.array([20, 21, 22], dtype=np.int32),
              np.array([23], dtype=np.int32),
          ],
          doc_boundaries=[0, 2, 4],
      )
      # Build npy indices (always document-granularity)
      npy_dir = os.path.join(tmp_dir, "npy")
      convert([prefix], npy_dir, seq_length=4, num_epochs=1, seed=42)

      # split_sentences=True should work fine
      ds = MegatronNpyDataSource(npy_dir=npy_dir, bin_paths=prefix, eod_id=0, seq_length=4, split_sentences=True)
      assert len(ds) > 0

      # split_sentences=False should detect mismatch (4 sequences != 2 documents)
      with pytest.raises(ValueError, match="split_sentences=False.*--split-sentences"):
        MegatronNpyDataSource(npy_dir=npy_dir, bin_paths=prefix, eod_id=0, seq_length=4, split_sentences=False)


class TestMegatronNpyIndexValidation:
  """Fast tests for lightweight and full index validation."""

  @staticmethod
  def _prefix(tmp_dir):
    return create_mmap_test_data(
        os.path.join(tmp_dir, "validation_data"),
        [
            np.array([10, 11, 0], dtype=np.int32),
            np.array([20, 21, 0], dtype=np.int32),
        ],
        doc_boundaries=[0, 1, 2],
    )

  @staticmethod
  def _source(prefix, document_index, sample_index, shuffle_index):
    return MegatronNpyDataSource(
        npy_dir=os.path.dirname(prefix),
        bin_paths=prefix,
        eod_id=0,
        seq_length=2,
        prebuilt_indices=(document_index, sample_index, shuffle_index),
    )

  def test_valid_prebuilt_indices_support_negative_index(self, tmp_dir):
    prefix = self._prefix(tmp_dir)
    source = self._source(
        prefix,
        np.array([0, 1], dtype=np.int32),
        np.array([[0, 0], [0, 2], [1, 2]], dtype=np.int32),
        np.array([0, 1], dtype=np.uint32),
    )

    assert len(source) == 2
    np.testing.assert_array_equal(source[-1]["text"], source[1]["text"])

  @pytest.mark.parametrize(
      "document_index,sample_index,shuffle_index,error",
      [
          (
              np.array([0, 1], dtype=np.int32),
              np.array([0, 1], dtype=np.int32),
              np.array([0], dtype=np.uint32),
              "Invalid sample_index shape",
          ),
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[0, 0]], dtype=np.int32),
              np.array([0], dtype=np.uint32),
              "too few rows",
          ),
          (
              np.array([[0, 1]], dtype=np.int32),
              np.array([[0, 0], [0, 1]], dtype=np.int32),
              np.array([0], dtype=np.uint32),
              "Invalid document_index shape",
          ),
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[0, 0], [0, 1]], dtype=np.int32),
              np.array([[0]], dtype=np.uint32),
              "Invalid shuffle_index shape",
          ),
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[0, 0], [0, 1]], dtype=np.int32),
              np.array([], dtype=np.uint32),
              "is empty",
          ),
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[0, 0], [0, 1]], dtype=np.int32),
              np.array([1], dtype=np.uint32),
              "outside valid range",
          ),
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[2, 0], [0, 1]], dtype=np.int32),
              np.array([0], dtype=np.uint32),
              "sample_index\\[first\\]",
          ),
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[0, 0], [0, -1]], dtype=np.int32),
              np.array([0], dtype=np.uint32),
              "negative token offset",
          ),
          (
              np.array([0, 2], dtype=np.int32),
              np.array([[0, 0], [0, 1]], dtype=np.int32),
              np.array([0], dtype=np.uint32),
              "document_index.*outside valid range",
          ),
      ],
  )
  def test_lightweight_validation_rejects_bad_prebuilt_indices(
      self, tmp_dir, document_index, sample_index, shuffle_index, error
  ):
    prefix = self._prefix(tmp_dir)
    with pytest.raises(ValueError, match=error):
      self._source(prefix, document_index, sample_index, shuffle_index)

  @pytest.mark.parametrize(
      "document_index,sample_index,shuffle_index,error",
      [
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int32),
              np.array([0, 2], dtype=np.uint32),
              "references sample_id range",
          ),
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[0, 0], [3, 0], [1, 1]], dtype=np.int32),
              np.array([0, 1], dtype=np.uint32),
              "references doc_pos range",
          ),
          (
              np.array([0, 3, 1], dtype=np.int32),
              np.array([[0, 0], [1, 0], [2, 1]], dtype=np.int32),
              np.array([0, 1], dtype=np.uint32),
              "references doc_id range",
          ),
          (
              np.array([0, 1], dtype=np.int32),
              np.array([[0, 0], [0, -1], [1, 1]], dtype=np.int32),
              np.array([0, 1], dtype=np.uint32),
              "contains negative token offset",
          ),
      ],
  )
  def test_full_validation_rejects_interior_bad_values(
      self, tmp_dir, monkeypatch, document_index, sample_index, shuffle_index, error
  ):
    monkeypatch.setenv("MMAP_NPY_FULL_VALIDATION", "1")
    prefix = self._prefix(tmp_dir)
    with pytest.raises(ValueError, match=error):
      self._source(prefix, document_index, sample_index, shuffle_index)


# ===========================================================================
# Tests: MegatronNpyDataSource __getitem__
# ===========================================================================


class TestMegatronNpyDataSourceGetitem:
  """Tests for MegatronNpyDataSource __getitem__ behavior."""

  def test_returns_dict_with_text(self, sample_dataset):
    """Each item is a dict with a 'text' key containing a numpy array."""
    ds = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=sample_dataset["prefix"],
        eod_id=0,
        seq_length=sample_dataset["seq_length"],
    )
    item = ds[0]
    assert isinstance(item, dict)
    assert "text" in item
    assert isinstance(item["text"], np.ndarray)

  def test_index_out_of_range_raises(self, sample_dataset):
    """Accessing an out-of-range index raises IndexError."""
    ds = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=sample_dataset["prefix"],
        eod_id=0,
        seq_length=sample_dataset["seq_length"],
    )
    with pytest.raises(IndexError):
      _ = ds[len(ds)]
    with pytest.raises(IndexError):
      _ = ds[len(ds) + 100]

  def test_no_extra_eod_at_doc_boundaries(self, sample_dataset):
    """No EOD tokens inserted at doc boundaries (matching Megatron behavior).

    Megatron's GPTDataset._query_document_sample_shuffle_indices does NOT
    insert EOD tokens between documents — it simply concatenates document
    token slices.  Our test data ([10,11,12], [20,21], [30,31,32,33]) has
    no eod_id=0 in the raw tokens, so the output should contain no zeros.
    """
    eod_id = 0
    ds = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=sample_dataset["prefix"],
        eod_id=eod_id,
        seq_length=sample_dataset["seq_length"],
    )
    all_tokens = []
    for sample in ds:
      all_tokens.append(sample["text"])

    combined = np.concatenate(all_tokens)
    # Raw data has no eod_id tokens; without EOD insertion, none should appear
    # (except possibly in padding at the tail of the last sample).
    non_pad = combined[combined != eod_id]
    assert len(non_pad) > 0, "All tokens are eod_id — unexpected"
    # All non-padding tokens come from the raw data (10-33 range)
    assert np.all(non_pad >= 10), (
        f"Unexpected token values below 10 (possible EOD insertion): " f"{non_pad[non_pad < 10].tolist()}"
    )


# ===========================================================================
# Tests: MegatronNpyDataSource pickle support
# ===========================================================================


class TestMegatronNpyDataSourcePickle:
  """Tests for MegatronNpyDataSource pickle serialization support."""

  def test_pickle_roundtrip(self, sample_dataset):
    """Pickling and unpickling produces an equivalent data source."""
    ds = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=sample_dataset["prefix"],
        eod_id=0,
        seq_length=sample_dataset["seq_length"],
    )
    # Get a sample before pickling
    original_item = ds[0]
    original_len = len(ds)

    # Pickle roundtrip
    data = pickle.dumps(ds)
    ds2 = pickle.loads(data)

    # Verify equivalence
    assert len(ds2) == original_len
    restored_item = ds2[0]
    np.testing.assert_array_equal(restored_item["text"], original_item["text"])

    # Verify all items match
    for i, original in enumerate(ds):
      np.testing.assert_array_equal(ds2[i]["text"], original["text"])


class TestMmapNpyIndexCache:
  """Exercise the runtime cache-miss path used before Grain workers start."""

  @staticmethod
  def _create_eod_dataset(tmp_dir):
    """Create a small indexed dataset with EOD-terminated documents."""
    prefix = os.path.join(tmp_dir, "data")
    create_mmap_test_data(
        prefix,
        [
            np.array([10, 11, 12, 0], dtype=np.int32),
            np.array([20, 21, 22, 23, 0], dtype=np.int32),
            np.array([30, 31, 32, 33, 34, 0], dtype=np.int32),
        ],
        doc_boundaries=[0, 1, 2, 3],
    )
    return prefix

  def test_cache_miss_prebuilt_indices_survive_pickle(self, tmp_path):
    """A worker can use cache-miss indices even before it observes the cache files."""
    tmp_dir = str(tmp_path)
    prefix = self._create_eod_dataset(tmp_dir)
    npy_dir = os.path.join(tmp_dir, "indices")
    with mock.patch("maxtext.input_pipeline._mmap_index_utils.is_primary_process", return_value=True):
      expected_hash, prebuilt = _ensure_npy_indices(
          npy_dir,
          [prefix],
          num_samples=4,
          seq_length=4,
          seed=42,
      )

    assert prebuilt is not None
    disk_source = MegatronNpyDataSource(
        npy_dir=npy_dir,
        bin_paths=prefix,
        eod_id=0,
        seq_length=4,
        expected_hash=expected_hash,
    )
    restored_memory_source = pickle.loads(
        pickle.dumps(
            MegatronNpyDataSource(
                npy_dir=npy_dir,
                bin_paths=prefix,
                eod_id=0,
                seq_length=4,
                expected_hash=expected_hash,
                prebuilt_indices=prebuilt,
            )
        )
    )
    assert len(restored_memory_source) == len(disk_source)
    for index, disk_sample in enumerate(disk_source):
      np.testing.assert_array_equal(restored_memory_source[index]["text"], disk_sample["text"])

  def test_cache_key_reuses_an_epoch_bucket(self, tmp_path):
    """Different requested sizes share an index triplet when epochs are unchanged."""
    tmp_dir = str(tmp_path)
    prefix = self._create_eod_dataset(tmp_dir)
    npy_dir = os.path.join(tmp_dir, "indices")
    with mock.patch("maxtext.input_pipeline._mmap_index_utils.is_primary_process", return_value=True):
      first_hash, first_prebuilt = _ensure_npy_indices(
          npy_dir,
          [prefix],
          num_samples=3,
          seq_length=4,
          seed=42,
      )
      same_bucket_hash, same_bucket_prebuilt = _ensure_npy_indices(
          npy_dir,
          [prefix],
          num_samples=2,
          seq_length=4,
          seed=42,
      )
      next_bucket_hash, next_bucket_prebuilt = _ensure_npy_indices(
          npy_dir,
          [prefix],
          num_samples=4,
          seq_length=4,
          seed=42,
      )

    assert first_prebuilt is not None
    assert first_hash == same_bucket_hash
    assert same_bucket_prebuilt is None
    assert next_bucket_hash != first_hash
    assert next_bucket_prebuilt is not None

  def test_non_primary_cache_miss_returns_in_memory_indices_without_writing(self, tmp_path):
    """Non-primary hosts build deterministic memory indices but skip cache writes."""
    tmp_dir = str(tmp_path)
    prefix = self._create_eod_dataset(tmp_dir)
    npy_dir = os.path.join(tmp_dir, "indices")

    with mock.patch("maxtext.input_pipeline._mmap_index_utils.is_primary_process", return_value=False):
      expected_hash, prebuilt = _ensure_npy_indices(
          npy_dir,
          [prefix],
          num_samples=4,
          seq_length=4,
          seed=42,
      )

    assert expected_hash
    assert prebuilt is not None
    assert not os.path.exists(npy_dir)

  def test_runtime_split_matches_offline_conversion(self, tmp_path):
    """Runtime split auto-build must consume the same document partition as ``convert``."""
    tmp_dir = str(tmp_path)
    prefix = os.path.join(tmp_dir, "data")
    sequences = []
    for document_id in range(20):
      tokens = np.arange(document_id * 16 + 1, document_id * 16 + 17, dtype=np.int32)
      tokens[-1] = 0
      sequences.append(tokens)
    create_mmap_test_data(prefix, sequences, doc_boundaries=list(range(21)))

    for split_index in (0, 1):
      offline_dir = os.path.join(tmp_dir, f"offline_{split_index}")
      convert(
          [prefix],
          offline_dir,
          seq_length=8,
          num_epochs=1,
          seed=42,
          split="0.9,0.1",
          split_index=split_index,
      )
      offline = MegatronNpyDataSource(
          npy_dir=offline_dir,
          bin_paths=prefix,
          eod_id=0,
          seq_length=8,
      )
      runtime = create_mmap_npy_source(
          f"{os.path.join(tmp_dir, f'runtime_{split_index}')}|{prefix}",
          eod_id=0,
          seq_length=8,
          split_sentences=False,
          seed=42,
          split="0.9,0.1",
          split_index=split_index,
      )

      assert len(runtime) == len(offline)
      for index, offline_sample in enumerate(offline):
        np.testing.assert_array_equal(runtime[index]["text"], offline_sample["text"])


# ===========================================================================
# Tests: mmap_npy integration with Grain pipeline
# ===========================================================================


class TestMmapNpyPipelineIntegration:
  """End-to-end integration with Grain pipeline."""

  def test_data_file_pattern_parsing(self, sample_dataset):
    """Verify the 'npy_dir|bin_dir' pattern creates a working dataset."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    pattern = f"{sample_dataset['npy_dir']}|{sample_dataset['bin_dir']}"
    dataset = get_datasets(
        data_file_pattern=pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(
            max_target_length=sample_dataset["seq_length"], eod_id=0, mmap_split_sentences=False
        ),
    )
    batch = next(iter(dataset))
    assert "text" in batch

  def test_single_spec_returns_all_samples(self, sample_dataset):
    """Single spec without mixture returns correct number of samples."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    pattern = f"{sample_dataset['npy_dir']}|{sample_dataset['bin_dir']}"
    # Use num_samples=4 and seed=42 to match the fixture's 2-epoch npy build,
    # so that _ensure_npy_indices finds the cached index files (cache hit).
    dataset = get_datasets(
        data_file_pattern=pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(
            max_target_length=sample_dataset["seq_length"],
            eod_id=0,
            mmap_split_sentences=False,
            num_samples=4,
            seed=42,
        ),
    )
    items = list(dataset)
    # MegatronNpyDataSource length equals the shuffle_index length
    source = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=sample_dataset["bin_dir"],
        eod_id=0,
        seq_length=sample_dataset["seq_length"],
    )
    assert len(items) == len(source)

  def test_num_samples_none_auto_builds_1_epoch(self, sample_dataset):
    """When num_samples=None, auto-build 1-epoch npy indices (Megatron alignment)."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    pattern = f"{sample_dataset['npy_dir']}|{sample_dataset['bin_dir']}"
    dataset = get_datasets(
        data_file_pattern=pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(
            max_target_length=sample_dataset["seq_length"], eod_id=0, mmap_split_sentences=False
        ),
    )
    items = list(dataset)
    # Total tokens = 3+2+4 = 9, seq_length = 4, add_extra_token = 1
    # 1-epoch samples = (9 - 1) // 4 = 2
    expected_1epoch_samples = (9 - 1) // sample_dataset["seq_length"]
    assert len(items) == expected_1epoch_samples

  def test_explicit_prefix_spec(self, sample_dataset):
    """Spec can use an explicit .bin prefix instead of a directory."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    pattern = f"{sample_dataset['npy_dir']}|{sample_dataset['prefix']}"
    dataset = get_datasets(
        data_file_pattern=pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(
            max_target_length=sample_dataset["seq_length"], eod_id=0, mmap_split_sentences=False
        ),
    )
    batch = next(iter(dataset))
    assert "text" in batch

  def test_malformed_spec_raises(self):
    """Spec missing the '|' separator raises ValueError."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    with pytest.raises(ValueError, match="mmap_npy spec must be"):
      get_datasets(
          data_file_pattern="no_pipe_here",
          data_file_type="mmap_npy",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(max_target_length=4, eod_id=0, mmap_split_sentences=False),
      )

  def test_mixture_with_weights(self, sample_dataset):
    """Semicolon-separated mixture spec with weights works."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    spec = f"{sample_dataset['npy_dir']}|{sample_dataset['bin_dir']}"
    # Same dataset twice with equal weights
    pattern = f"{spec},0.5;{spec},0.5"
    dataset = get_datasets(
        data_file_pattern=pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(
            max_target_length=sample_dataset["seq_length"], eod_id=0, mmap_split_sentences=False
        ),
    )
    batch = next(iter(dataset))
    assert "text" in batch

  def test_zero_weight_mixture_collapses_to_single_dataset(self, sample_dataset):
    """A zero-weight mmap_npy component is filtered before blending."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    spec = f"{sample_dataset['npy_dir']}|{sample_dataset['bin_dir']}"
    dataset = get_datasets(
        data_file_pattern=f"{spec},0.0;{spec},1.0",
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(
            max_target_length=sample_dataset["seq_length"],
            eod_id=0,
            mmap_split_sentences=False,
            num_samples=4,
            seed=42,
        ),
    )

    items = list(dataset)
    assert items
    assert all("text" in item for item in items)

  def test_mixture_malformed_entry_raises(self, sample_dataset):
    """Mixture entry without a weight raises ValueError."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    spec = f"{sample_dataset['npy_dir']}|{sample_dataset['bin_dir']}"
    # Construct a malformed mixture: missing weight
    pattern_bad = f"{spec};{spec}"
    with pytest.raises(ValueError, match="Malformed mmap_npy mixture"):
      get_datasets(
          data_file_pattern=pattern_bad,
          data_file_type="mmap_npy",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(
              max_target_length=sample_dataset["seq_length"], eod_id=0, mmap_split_sentences=False
          ),
      )

  def test_unsupported_file_type_error_includes_mmap_npy(self):
    """Error message for unsupported file types now mentions mmap_npy."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    with pytest.raises(ValueError, match="mmap_npy"):
      get_datasets(
          data_file_pattern="dummy",
          data_file_type="unsupported_type",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
      )


# ===========================================================================
# Tests: fixed-length output and end-to-end pipeline batch
# ===========================================================================


class TestMegatronNpyFixedLengthOutput:
  """Verify that all samples have exactly seq_length + 1 tokens and that
  the Grain pretrain pipeline can batch them without errors.

  Regression test for the bug where _build_sample inserted EOD tokens at
  document boundaries, producing variable-length arrays that caused
  ``np.stack`` to fail during batching.
  """

  def test_all_samples_uniform_length(self, sample_dataset):
    """Every sample is exactly seq_length tokens after truncation."""
    seq_length = sample_dataset["seq_length"]
    ds = MegatronNpyDataSource(
        npy_dir=sample_dataset["npy_dir"],
        bin_paths=sample_dataset["prefix"],
        eod_id=0,
        seq_length=seq_length,
    )
    expected_len = seq_length + 1
    for i, sample in enumerate(ds):
      tokens = sample["text"]
      assert len(tokens) == expected_len, f"Sample {i}: expected {expected_len} tokens, got {len(tokens)}"

  def test_pretrain_pipeline_batches_without_error(self, sample_dataset):
    """End-to-end: get_datasets -> pretrain_preprocessing_pipeline -> batch.

    Reproduces the production crash where variable-length samples caused
    ``ValueError: all input arrays must have the same shape`` at batch time.
    """
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    seq_length = sample_dataset["seq_length"]
    pattern = f"{sample_dataset['npy_dir']}|{sample_dataset['prefix']}"
    ds = get_datasets(
        data_file_pattern=pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(max_target_length=seq_length, eod_id=0, mmap_split_sentences=False),
    )
    cfg = SimpleNamespace(
        grain_file_type="mmap_npy",
        mmap_eod_id=0,
        tokenizer_path="",
        tokenizer_type="sentencepiece",
        add_bos=False,
        add_eos=False,
        hf_access_token="",
        dataset_type="grain",
        max_target_length=seq_length,
        use_truncation=False,
        global_batch_size_to_load=4,
        expansion_factor_real_data=1,
        elastic_enabled=False,
        packing=False,
        grain_packing_type="concat_then_split",
        max_segments_per_seq=None,
        reset_attention_mask=False,
        grain_ram_budget_mb=256,
        eod_mask_loss=False,
    )
    pipe = pretrain_preprocessing_pipeline(
        ds,
        cfg,
        data_columns=["text"],
        tokenize=False,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )
    batch = next(iter(pipe))

    # Verify batch has the expected keys and shapes
    expected_keys = {
        "inputs",
        "targets",
        "inputs_segmentation",
        "targets_segmentation",
        "inputs_position",
        "targets_position",
    }
    assert set(batch.keys()) == expected_keys
    batch_size = 4
    for key in expected_keys:
      assert batch[key].shape == (batch_size, seq_length), (
          f"{key}: expected shape ({batch_size}, {seq_length}), " f"got {batch[key].shape}"
      )

  def test_pretrain_pipeline_with_mp_prefetch(self):
    """End-to-end pipeline with grain_worker_count=2 (mp_prefetch enabled).

    Verifies that the mmap_npy path — which places mp_prefetch BEFORE batch
    — produces correct output shapes when multiprocessing is active.
    """
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      bin_dir = os.path.join(tmp_dir, "bin_dir")
      npy_dir = os.path.join(tmp_dir, "npy_dir")
      os.makedirs(bin_dir)
      os.makedirs(npy_dir)

      # Create enough data for mp_prefetch to be meaningful:
      # 10 documents, ~15 tokens each, 3 epochs -> many samples
      seqs = [np.arange(i * 20 + 1, i * 20 + 16, dtype=np.int32) for i in range(10)]
      prefix = create_mmap_test_data(
          os.path.join(bin_dir, "test_data"),
          seqs,
          doc_boundaries=list(range(len(seqs) + 1)),
      )
      seq_length = 8
      convert([prefix], npy_dir, seq_length=seq_length, num_epochs=3, seed=42)

      pattern = f"{npy_dir}|{prefix}"
      ds = get_datasets(
          data_file_pattern=pattern,
          data_file_type="mmap_npy",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=MMapDatasetConfig(max_target_length=seq_length, eod_id=0, mmap_split_sentences=False),
      )
      cfg = SimpleNamespace(
          grain_file_type="mmap_npy",
          mmap_eod_id=0,
          tokenizer_path="",
          tokenizer_type="sentencepiece",
          add_bos=False,
          add_eos=False,
          hf_access_token="",
          dataset_type="grain",
          max_target_length=seq_length,
          use_truncation=False,
          global_batch_size_to_load=4,
          expansion_factor_real_data=1,
          elastic_enabled=False,
          packing=False,
          grain_packing_type="concat_then_split",
          max_segments_per_seq=None,
          reset_attention_mask=False,
          grain_ram_budget_mb=256,
          eod_mask_loss=False,
      )
      pipe = pretrain_preprocessing_pipeline(
          ds,
          cfg,
          data_columns=["text"],
          tokenize=False,
          grain_worker_count=2,
          grain_per_worker_buffer_size=2,
      )
      batch = next(iter(pipe))

      expected_keys = {
          "inputs",
          "targets",
          "inputs_segmentation",
          "targets_segmentation",
          "inputs_position",
          "targets_position",
      }
      assert set(batch.keys()) == expected_keys
      batch_size = 4
      for key in expected_keys:
        assert batch[key].shape == (
            batch_size,
            seq_length,
        ), f"{key}: expected shape ({batch_size}, {seq_length}), got {batch[key].shape}"

  def test_mp_prefetch_preserves_sample_ordering(self):
    """grain_worker_count=2 produces identical token content as grain_worker_count=0.

    Guards against Grain library changes that might alter mp_prefetch
    ordering semantics.  Collects all batches from both configurations
    and compares inputs/targets element-by-element.
    """
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      bin_dir = os.path.join(tmp_dir, "bin_dir")
      npy_dir = os.path.join(tmp_dir, "npy_dir")
      os.makedirs(bin_dir)
      os.makedirs(npy_dir)

      seqs = [np.arange(i * 20 + 1, i * 20 + 16, dtype=np.int32) for i in range(10)]
      prefix = create_mmap_test_data(
          os.path.join(bin_dir, "test_data"),
          seqs,
          doc_boundaries=list(range(len(seqs) + 1)),
      )
      seq_length = 8
      convert([prefix], npy_dir, seq_length=seq_length, num_epochs=3, seed=42)

      pattern = f"{npy_dir}|{prefix}"
      cfg = SimpleNamespace(
          grain_file_type="mmap_npy",
          mmap_eod_id=0,
          tokenizer_path="",
          tokenizer_type="sentencepiece",
          add_bos=False,
          add_eos=False,
          hf_access_token="",
          dataset_type="grain",
          max_target_length=seq_length,
          use_truncation=False,
          global_batch_size_to_load=4,
          expansion_factor_real_data=1,
          elastic_enabled=False,
          packing=False,
          grain_packing_type="concat_then_split",
          max_segments_per_seq=None,
          reset_attention_mask=False,
          grain_ram_budget_mb=256,
          eod_mask_loss=False,
      )

      def _collect_all_batches(worker_count, buffer_size):
        ds = get_datasets(
            data_file_pattern=pattern,
            data_file_type="mmap_npy",
            shuffle=False,
            shuffle_seed=0,
            shuffle_buffer_size=0,
            num_epoch=1,
            dataloading_host_index=0,
            dataloading_host_count=1,
            grain_worker_count=0,
            grain_num_threads=1,
            grain_prefetch_buffer_size=1,
            grain_data_source_max_workers=1,
            dataset_config=MMapDatasetConfig(
                max_target_length=seq_length,
                eod_id=0,
                mmap_split_sentences=False,
            ),
        )
        pipe = pretrain_preprocessing_pipeline(
            ds,
            cfg,
            data_columns=["text"],
            tokenize=False,
            grain_worker_count=worker_count,
            grain_per_worker_buffer_size=buffer_size,
        )
        return list(pipe)

      batches_w0 = _collect_all_batches(worker_count=0, buffer_size=1)
      batches_w2 = _collect_all_batches(worker_count=2, buffer_size=2)

      assert len(batches_w0) == len(
          batches_w2
      ), f"Batch count mismatch: worker=0 got {len(batches_w0)}, worker=2 got {len(batches_w2)}"
      for b_idx, (b0, b2) in enumerate(zip(batches_w0, batches_w2)):
        for key in ("inputs", "targets"):
          np.testing.assert_array_equal(
              b0[key],
              b2[key],
              err_msg=f"Batch {b_idx}, key '{key}': worker_count=0 vs worker_count=2 mismatch",
          )


# ===========================================================================
# Tests: no extra EOD insertion
# ===========================================================================


class TestMegatronNpyNoExtraEod:
  """Verify that _build_sample does not insert extra EOD tokens.

  Megatron's GPTDataset concatenates document slices without inserting
  EOD tokens.  EODs only appear if the raw data already contains them
  (from ``--append-eod`` during preprocessing)."""

  def test_no_double_eod_when_docs_end_with_eod(self):
    """Documents whose tokens already end with eod_id should not get a
    second EOD inserted at the boundary."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      bin_dir = os.path.join(tmp_dir, "bin")
      npy_dir = os.path.join(tmp_dir, "npy")
      os.makedirs(bin_dir)
      os.makedirs(npy_dir)

      eod_id = 0
      # Documents already contain trailing EOD
      prefix = create_mmap_test_data(
          os.path.join(bin_dir, "data"),
          sequences=[
              np.array([10, 11, 12, eod_id], dtype=np.int32),
              np.array([20, 21, eod_id], dtype=np.int32),
              np.array([30, 31, 32, eod_id], dtype=np.int32),
          ],
          doc_boundaries=[0, 1, 2, 3],
      )

      seq_length = 8
      convert(
          input_paths=[prefix],
          output_dir=npy_dir,
          seq_length=seq_length,
          num_epochs=2,
          seed=42,
      )

      ds = MegatronNpyDataSource(
          npy_dir=npy_dir,
          bin_paths=prefix,
          eod_id=eod_id,
          seq_length=seq_length,
      )

      for i, sample in enumerate(ds):
        tokens = sample["text"]
        # There should be no consecutive eod_id pair
        for j in range(len(tokens) - 1):
          if tokens[j] == eod_id and tokens[j + 1] == eod_id:
            # Allow trailing eod padding at the end of the sample
            # (all remaining tokens are eod), but not mid-stream
            remaining = tokens[j:]
            if not np.all(remaining == eod_id):
              raise AssertionError(
                  f"Sample {i}: double EOD at positions {j},{j+1} " f"(not trailing pad): {tokens.tolist()}"
              )

  def test_no_double_eod_mixed_docs(self):
    """Mix of documents with and without trailing EOD."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      bin_dir = os.path.join(tmp_dir, "bin")
      npy_dir = os.path.join(tmp_dir, "npy")
      os.makedirs(bin_dir)
      os.makedirs(npy_dir)

      eod_id = 0
      prefix = create_mmap_test_data(
          os.path.join(bin_dir, "data"),
          sequences=[
              np.array([10, 11, eod_id], dtype=np.int32),  # has EOD
              np.array([20, 21, 22], dtype=np.int32),  # no EOD
              np.array([30, 31, eod_id], dtype=np.int32),  # has EOD
          ],
          doc_boundaries=[0, 1, 2, 3],
      )

      seq_length = 6
      convert(
          input_paths=[prefix],
          output_dir=npy_dir,
          seq_length=seq_length,
          num_epochs=2,
          seed=42,
      )

      ds = MegatronNpyDataSource(
          npy_dir=npy_dir,
          bin_paths=prefix,
          eod_id=eod_id,
          seq_length=seq_length,
      )

      for i, sample in enumerate(ds):
        tokens = sample["text"]
        for j in range(len(tokens) - 1):
          if tokens[j] == eod_id and tokens[j + 1] == eod_id:
            remaining = tokens[j:]
            if not np.all(remaining == eod_id):
              raise AssertionError(
                  f"Sample {i}: double EOD at positions {j},{j+1} " f"(not trailing pad): {tokens.tolist()}"
              )


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
