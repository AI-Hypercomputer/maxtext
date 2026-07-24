"""Tests for MMap indexed dataset support (Megatron-LM format)."""

# pylint: disable=redefined-outer-name

import os
import pickle
import struct
import tempfile
import unittest.mock
from concurrent.futures import ThreadPoolExecutor  # pylint: disable=no-name-in-module

import numpy as np
import pytest

import grain.python as grain

from maxtext.input_pipeline._mmap_datasource import (
    DTYPE_CODES,
    DTYPE_CODES_INV,
    MMAP_INDEX_HEADER_SIZE,
    MMAP_INDEX_MAGIC,
    MMAP_INDEX_VERSION,
    MMapDatasetConfig,
    MMapIndexedDataset,
    MMapIndexedDataSource,
)
from tests.unit.mmap_test_utils import create_mmap_test_data

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
    from maxtext.input_pipeline._mmap_datasource import MMapSampleIndexDataSource  # pylint: disable=import-outside-toplevel

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
    from maxtext.input_pipeline._mmap_datasource import MMapSampleIndexDataSource  # pylint: disable=import-outside-toplevel

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

    with unittest.mock.patch.object(logging.getLogger(mmap_mod.__name__), "warning") as mock_warn:
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
    from maxtext.input_pipeline._mmap_datasource import MMapSampleIndexDataSource  # pylint: disable=import-outside-toplevel

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
    from maxtext.input_pipeline._mmap_datasource import MMapSampleIndexDataSource  # pylint: disable=import-outside-toplevel

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


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
