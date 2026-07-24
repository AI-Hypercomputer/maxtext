"""Tests for the mmap-to-npy index prebuilder tool."""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from tools.data_processing.mmap_index_builder import (
    build_document_index,
    build_sample_index,
    build_shuffle_index,
    convert,
    convert_blend,
    discover_shards,
    get_document_sizes,
)
from maxtext.input_pipeline._megatron_blending import MegatronBlendedDataSource
from tests.unit.mmap_test_utils import create_mmap_test_data
from maxtext.input_pipeline._mmap_datasource import _discover_npy_indices

pytestmark = pytest.mark.cpu_only

# pylint: disable=redefined-outer-name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
  with tempfile.TemporaryDirectory() as d:
    yield d


# ===========================================================================
# Unit tests: discover_shards
# ===========================================================================


class TestDiscoverShards:
  """Tests for the discover_shards function."""

  def test_discovers_single_shard(self, tmp_dir):
    """A directory with a single .idx/.bin pair returns one prefix."""
    prefix = os.path.join(tmp_dir, "shard_00")
    create_mmap_test_data(prefix, [np.array([1, 2, 3], dtype=np.int32)])

    result = discover_shards(tmp_dir)
    assert result == [prefix]

  def test_discovers_multiple_shards_sorted(self, tmp_dir):
    """Multiple .idx/.bin pairs are returned sorted alphabetically."""
    names = ["c_shard", "a_shard", "b_shard"]
    prefixes = []
    for name in names:
      prefix = os.path.join(tmp_dir, name)
      create_mmap_test_data(prefix, [np.array([1], dtype=np.int32)])
      prefixes.append(prefix)

    result = discover_shards(tmp_dir)
    assert result == sorted(prefixes)

  def test_empty_dir_raises(self, tmp_dir):
    """An empty directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No .idx files found"):
      discover_shards(tmp_dir)

  def test_single_prefix_as_path(self, tmp_dir):
    """A path prefix (not a directory) that has a matching .idx file."""
    prefix = os.path.join(tmp_dir, "my_dataset")
    create_mmap_test_data(prefix, [np.array([10, 20], dtype=np.int32)])

    result = discover_shards(prefix)
    assert result == [prefix]


# ===========================================================================
# Unit tests: get_document_sizes
# ===========================================================================


class TestGetDocumentSizes:
  """Tests for the get_document_sizes function."""

  def test_single_shard_one_doc(self, tmp_dir):
    """3 sequences in 1 document -> sizes summed to 6."""
    prefix = os.path.join(tmp_dir, "shard")
    create_mmap_test_data(
        prefix,
        [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        ],
        doc_boundaries=None,  # single doc
    )
    result = get_document_sizes([prefix])
    np.testing.assert_array_equal(result, [6])

  def test_single_shard_multi_doc(self, tmp_dir):
    """5 sequences in 3 documents -> [5, 1, 5]."""
    prefix = os.path.join(tmp_dir, "shard")
    # doc 0: seq 0,1 (sizes 3,2 -> 5 tokens)
    # doc 1: seq 2 (size 1 -> 1 token)
    # doc 2: seq 3,4 (sizes 2,3 -> 5 tokens)
    create_mmap_test_data(
        prefix,
        [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
            np.array([6], dtype=np.int32),
            np.array([7, 8], dtype=np.int32),
            np.array([9, 10, 11], dtype=np.int32),
        ],
        doc_boundaries=[0, 2, 3, 5],
    )
    result = get_document_sizes([prefix])
    np.testing.assert_array_equal(result, [5, 1, 5])

  def test_multi_shard(self, tmp_dir):
    """2 shards -> concatenated [3, 2]."""
    prefix1 = os.path.join(tmp_dir, "shard_00")
    create_mmap_test_data(
        prefix1,
        [np.array([1, 2, 3], dtype=np.int32)],
        doc_boundaries=None,
    )
    prefix2 = os.path.join(tmp_dir, "shard_01")
    create_mmap_test_data(
        prefix2,
        [np.array([4, 5], dtype=np.int32)],
        doc_boundaries=None,
    )
    result = get_document_sizes([prefix1, prefix2])
    np.testing.assert_array_equal(result, [3, 2])


# ===========================================================================
# Unit tests: build_document_index
# ===========================================================================


class TestBuildDocumentIndex:
  """Tests for the build_document_index function."""

  def test_single_epoch_contains_all_docs(self):
    """Single epoch contains each document exactly once."""
    num_docs = 5
    doc_index = build_document_index(num_docs, num_epochs=1, seed=42)
    assert len(doc_index) == num_docs
    assert sorted(doc_index.tolist()) == list(range(num_docs))

  def test_multi_epoch_contains_all_docs_repeated(self):
    """Multiple epochs contain each document the correct number of times."""
    num_docs = 4
    num_epochs = 3
    doc_index = build_document_index(num_docs, num_epochs=num_epochs, seed=42)
    assert len(doc_index) == num_docs * num_epochs
    for d in range(num_docs):
      assert np.sum(doc_index == d) == num_epochs

  def test_separate_last_epoch(self):
    """With separate_last_epoch, last epoch is shuffled independently."""
    num_docs = 5
    num_epochs = 3
    doc_index = build_document_index(num_docs, num_epochs=num_epochs, seed=42, separate_last_epoch=True)
    assert len(doc_index) == num_docs * num_epochs
    # First (num_epochs-1) epochs
    first_part = doc_index[: num_docs * (num_epochs - 1)]
    last_part = doc_index[num_docs * (num_epochs - 1) :]
    for d in range(num_docs):
      assert np.sum(first_part == d) == num_epochs - 1
      assert np.sum(last_part == d) == 1

  def test_deterministic_with_same_seed(self):
    """Same seed produces identical results."""
    idx1 = build_document_index(10, num_epochs=2, seed=123)
    idx2 = build_document_index(10, num_epochs=2, seed=123)
    np.testing.assert_array_equal(idx1, idx2)

  def test_different_seed_different_result(self):
    """Different seeds produce different orderings."""
    idx1 = build_document_index(100, num_epochs=1, seed=1)
    idx2 = build_document_index(100, num_epochs=1, seed=2)
    # With 100 docs, extremely unlikely to be identical
    assert not np.array_equal(idx1, idx2)


# ===========================================================================
# Unit tests: build_sample_index
# ===========================================================================


class TestBuildSampleIndex:
  """Tests for the build_sample_index function."""

  def test_basic_single_doc(self):
    """10 tokens, seq_length=3 -> 3 samples with shape (4, 2)."""
    doc_sizes = np.array([10], dtype=np.int64)
    doc_index = np.array([0], dtype=np.int32)
    result = build_sample_index(doc_sizes, doc_index, seq_length=3)
    # (10 - 1) // 3 = 3 samples -> shape (4, 2)
    assert result.shape == (4, 2)

  def test_multi_doc_spanning(self):
    """Two docs [5, 5], seq_length=4 -> samples span across doc boundary."""
    doc_sizes = np.array([5, 5], dtype=np.int64)
    doc_index = np.array([0, 1], dtype=np.int32)
    sample_idx = build_sample_index(doc_sizes, doc_index, seq_length=4)
    # Total tokens = 10, (10-1)//4 = 2 samples
    assert sample_idx.shape[0] == 3  # 2 + 1
    # Verify offsets: sample 0 starts at (0, 0)
    np.testing.assert_array_equal(sample_idx[0], [0, 0])
    # Sample 1 starts at (0, 4) — consumed 4 tokens from doc 0, offset stays at 4
    np.testing.assert_array_equal(sample_idx[1], [0, 4])

  def test_exact_division(self):
    """5 tokens, seq_length=4 -> 1 sample."""
    doc_sizes = np.array([5], dtype=np.int64)
    doc_index = np.array([0], dtype=np.int32)
    result = build_sample_index(doc_sizes, doc_index, seq_length=4)
    # (5 - 1) // 4 = 1 sample
    assert result.shape[0] - 1 == 1

  def test_doc_offset_tracking(self):
    """20 tokens, seq_length=3 -> 6 samples, verify offsets."""
    doc_sizes = np.array([20], dtype=np.int64)
    doc_index = np.array([0], dtype=np.int32)
    result = build_sample_index(doc_sizes, doc_index, seq_length=3)
    # (20 - 1) // 3 = 6 samples
    assert result.shape[0] - 1 == 6
    # For a single doc, offsets should be [0, 3, 6, 9, 12, 15, 18]
    expected_offsets = [0, 3, 6, 9, 12, 15, 18]
    actual_offsets = result[:, 1].tolist()
    assert actual_offsets == expected_offsets

  def test_drop_last_true(self):
    """7 tokens, seq_length=3, drop_last=True -> 2 samples."""
    doc_sizes = np.array([7], dtype=np.int64)
    doc_index = np.array([0], dtype=np.int32)
    result = build_sample_index(doc_sizes, doc_index, seq_length=3, drop_last=True)
    # (7 - 1) // 3 = 2 samples
    assert result.shape[0] - 1 == 2

  def test_drop_last_false(self):
    """8 tokens, seq_length=3, drop_last=False -> 3 samples (vs 2 with drop_last=True)."""
    doc_sizes = np.array([8], dtype=np.int64)
    doc_index = np.array([0], dtype=np.int32)
    result = build_sample_index(doc_sizes, doc_index, seq_length=3, drop_last=False)
    # ceil((8 - 1) / 3) = 3 samples
    assert result.shape[0] - 1 == 3


# ===========================================================================
# Unit tests: build_shuffle_index
# ===========================================================================


class TestBuildShuffleIndex:
  """Tests for the build_shuffle_index function."""

  def test_basic(self):
    """10 samples, full shuffle, check it is a permutation."""
    result = build_shuffle_index(10, 10, seed=42)
    assert len(result) == 10
    assert sorted(result.tolist()) == list(range(10))

  def test_separate_last_epoch(self):
    """7 main + 3 extra, verify independent shuffles."""
    result = build_shuffle_index(7, 10, seed=42)
    assert len(result) == 10
    # First 7 are a permutation of [0..6]
    assert sorted(result[:7].tolist()) == list(range(7))
    # Last 3 are a permutation of [7..9]
    assert sorted(result[7:].tolist()) == list(range(7, 10))

  def test_deterministic(self):
    """Same seed produces identical results."""
    r1 = build_shuffle_index(10, 10, seed=99)
    r2 = build_shuffle_index(10, 10, seed=99)
    np.testing.assert_array_equal(r1, r2)

  def test_small_uses_uint32(self):
    """Small sizes use uint32 dtype."""
    result = build_shuffle_index(10, 10, seed=42)
    assert result.dtype == np.uint32


# ===========================================================================
# Unit tests: convert
# ===========================================================================


class TestConvert:
  """Tests for the convert function."""

  def test_generates_three_npy_files(self, tmp_dir):
    """convert() produces three .npy files."""
    prefix = os.path.join(tmp_dir, "shard")
    create_mmap_test_data(
        prefix,
        [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)],
    )
    out_dir = os.path.join(tmp_dir, "output")
    paths = convert([prefix], out_dir, seq_length=3, num_samples=2, seed=42)
    assert "document_index" in paths
    assert "sample_index" in paths
    assert "shuffle_index" in paths
    for key, p in paths.items():
      assert os.path.isfile(p)
      if key != "metadata":
        assert p.endswith(".npy")

  def test_output_shapes(self, tmp_dir):
    """Output arrays have expected shapes."""
    prefix = os.path.join(tmp_dir, "shard")
    create_mmap_test_data(
        prefix,
        [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)],
    )
    out_dir = os.path.join(tmp_dir, "output")
    paths = convert([prefix], out_dir, seq_length=3, num_samples=2, seed=42)
    doc_idx = np.load(paths["document_index"])
    sample_idx = np.load(paths["sample_index"])
    shuffle_idx = np.load(paths["shuffle_index"])
    assert doc_idx.ndim == 1
    assert sample_idx.ndim == 2
    assert sample_idx.shape[1] == 2
    assert shuffle_idx.ndim == 1

  def test_num_epochs_mode(self, tmp_dir):
    """num_epochs mode produces valid output."""
    prefix = os.path.join(tmp_dir, "shard")
    create_mmap_test_data(
        prefix,
        [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)],
    )
    out_dir = os.path.join(tmp_dir, "output")
    paths = convert([prefix], out_dir, seq_length=3, num_epochs=2, seed=42)
    doc_idx = np.load(paths["document_index"])
    # 2 epochs of 1 doc = 2 entries
    assert len(doc_idx) == 2

  def test_dir_input(self, tmp_dir):
    """Passing a directory discovers shards automatically."""
    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir)
    prefix = os.path.join(data_dir, "shard_00")
    create_mmap_test_data(
        prefix,
        [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)],
    )
    out_dir = os.path.join(tmp_dir, "output")
    paths = convert([data_dir], out_dir, seq_length=3, num_samples=2, seed=42)
    for p in paths.values():
      assert os.path.isfile(p)


# ===========================================================================
# End-to-end tests: blend offline output and runtime loading
# ===========================================================================


class TestBlendIndexOutput:
  """The blend command must emit the exact ``blend_index_dir`` protocol."""

  def _create_input(self, tmp_dir, name, offset):
    prefix = os.path.join(tmp_dir, name)
    create_mmap_test_data(
        prefix,
        [np.arange(offset + 32 * i, offset + 32 * (i + 1), dtype=np.int32) for i in range(8)],
        doc_boundaries=list(range(9)),
    )
    return prefix

  def test_convert_blend_writes_runtime_dispatch_pair(self, tmp_dir):
    """Offline blend output is consumable verbatim through ``blend_index_dir``."""
    prefix_a = self._create_input(tmp_dir, "source_a", 100)
    prefix_b = self._create_input(tmp_dir, "source_b", 1000)
    output_dir = os.path.join(tmp_dir, "blend_indices")
    total_samples = 12

    results = convert_blend(
        dataset_specs=[
            {"input": [prefix_a], "weight": 0.7, "output_dir": os.path.join(output_dir, "dataset_0")},
            {"input": [prefix_b], "weight": 0.3, "output_dir": os.path.join(output_dir, "dataset_1")},
        ],
        total_samples=total_samples,
        seq_length=8,
        seed=42,
        max_workers=1,
        blend_index_output_dir=output_dir,
    )
    expected_dataset_index = np.load(os.path.join(output_dir, "dataset_index.npy"))
    expected_sample_index = np.load(os.path.join(output_dir, "dataset_sample_index.npy"))
    assert expected_dataset_index.shape == (total_samples,)
    assert expected_sample_index.shape == (total_samples,)

    lengths = [np.load(result["paths"]["sample_index"]).shape[0] - 1 for result in results]
    source = MegatronBlendedDataSource(
        map_datasets=[list(range(length)) for length in lengths],
        weights=[result["weight"] for result in results],
        size=total_samples,
        dataset_lengths=lengths,
        blend_index_dir=output_dir,
    )
    np.testing.assert_array_equal(source._dataset_index, expected_dataset_index)  # pylint: disable=protected-access
    np.testing.assert_array_equal(source._dataset_sample_index, expected_sample_index)  # pylint: disable=protected-access

  def test_size_is_pinned_and_zero_weight_lengths_are_filtered(self):
    """A requested training size is exact, and zero weights retain aligned lengths."""
    pinned_source = MegatronBlendedDataSource(
        map_datasets=[list(range(20)), list(range(20)), list(range(20))],
        weights=[0.5, 0.3, 0.2],
        dataset_lengths=[20, 20, 20],
        # Megatron's per-dataset ceil calculation totals 8 here.  MaxText's
        # requested global training size intentionally remains 7.
        size=7,
    )
    assert len(pinned_source) == 7

    zero_filtered_source = MegatronBlendedDataSource(
        map_datasets=[["dropped"], list(range(20))],
        weights=[0.0, 1.0],
        dataset_lengths=[1, 20],
        size=7,
    )
    assert [zero_filtered_source[i] for i in range(len(zero_filtered_source))] == list(range(7))

  def test_convert_blend_skips_a_zero_weight_input(self, tmp_dir):
    """Offline building filters zero weight before it attempts child conversion."""
    active_prefix = self._create_input(tmp_dir, "active", 200)
    output_dir = os.path.join(tmp_dir, "blend_indices")
    results = convert_blend(
        dataset_specs=[
            {
                "input": [os.path.join(tmp_dir, "does_not_exist")],
                "weight": 0.0,
                "output_dir": os.path.join(output_dir, "zero"),
            },
            {"input": [active_prefix], "weight": 1.0, "output_dir": os.path.join(output_dir, "active")},
        ],
        total_samples=4,
        seq_length=8,
        seed=42,
        max_workers=1,
        blend_index_output_dir=output_dir,
    )
    assert len(results) == 1
    np.testing.assert_array_equal(np.load(os.path.join(output_dir, "dataset_index.npy")), np.zeros(4, dtype=np.int16))

  def test_invalid_prebuilt_pair_falls_back_to_a_valid_in_memory_pair(self, tmp_dir):
    """Corrupt prebuilt indices are recoverable, rather than a training-start failure."""
    np.save(os.path.join(tmp_dir, "dataset_index.npy"), np.array([0, 0, 0, 0], dtype=np.int16))
    # In-range but non-contiguous: the stronger cache validation must reject it.
    np.save(os.path.join(tmp_dir, "dataset_sample_index.npy"), np.array([0, 2, 1, 3], dtype=np.int64))

    source = MegatronBlendedDataSource(
        map_datasets=[list(range(10))],
        weights=[1.0],
        dataset_lengths=[10],
        size=4,
        blend_index_dir=tmp_dir,
    )
    np.testing.assert_array_equal(source._dataset_sample_index, np.arange(4))  # pylint: disable=protected-access

  def test_cache_write_failure_keeps_the_in_memory_blend_usable(self, tmp_dir):
    """A read-only or malformed cache location must not block training startup."""
    cache_path = os.path.join(tmp_dir, "not_a_directory")
    with open(cache_path, "w", encoding="utf-8") as writer:
      writer.write("cache path intentionally occupied by a file")

    source = MegatronBlendedDataSource(
        map_datasets=[list(range(10))],
        weights=[1.0],
        size=4,
        cache_dir=cache_path,
    )
    assert [source[i] for i in range(len(source))] == [0, 1, 2, 3]


# ===========================================================================
# Unit tests: CLI
# ===========================================================================


class TestCLI:
  """Tests for the mmap_index_builder CLI entrypoint."""

  def _script_path(self):
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tools",
        "data_processing",
        "mmap_index_builder.py",
    )

  def _env(self):
    env = os.environ.copy()
    src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env["PYTHONPATH"] = os.path.abspath(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    return env

  def test_cli_basic(self, tmp_dir):
    """subprocess run with valid args produces 3 .npy files."""
    prefix = os.path.join(tmp_dir, "shard")
    create_mmap_test_data(
        prefix,
        [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)],
    )
    out_dir = os.path.join(tmp_dir, "output")
    cwd = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    result = subprocess.run(
        [
            sys.executable,
            self._script_path(),
            "--input",
            prefix,
            "--output-dir",
            out_dir,
            "--seq-length",
            "3",
            "--num-samples",
            "2",
            "--seed",
            "42",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=self._env(),
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    npy_files = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
    assert len(npy_files) == 3

  def test_cli_missing_required_args(self):
    """No args returns nonzero exit code."""
    cwd = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    result = subprocess.run(
        [sys.executable, self._script_path()],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=self._env(),
        check=False,
    )
    assert result.returncode != 0


# ===========================================================================
# End-to-end integration tests
# ===========================================================================


class TestEndToEnd:
  """End-to-end integration tests for the mmap-to-npy pipeline."""

  def test_indices_are_valid_for_data_access(self, tmp_dir):
    """Verify generated indices can be used to reconstruct samples."""
    seqs = []
    doc_boundaries = [0]
    for d in range(10):
      tokens = np.arange(d * 20, (d + 1) * 20, dtype=np.int32)
      seqs.append(tokens)
      doc_boundaries.append(len(seqs))
    prefix = os.path.join(tmp_dir, "data")
    create_mmap_test_data(prefix, seqs, doc_boundaries=doc_boundaries)

    out_dir = os.path.join(tmp_dir, "output")
    os.makedirs(out_dir)
    result = convert(
        input_paths=[prefix],
        output_dir=out_dir,
        seq_length=8,
        num_epochs=2,
        seed=42,
    )

    doc_idx = np.load(result["document_index"])
    sample_idx = np.load(result["sample_index"])
    shuffle_idx = np.load(result["shuffle_index"])

    doc_sizes = get_document_sizes([prefix])
    num_samples = sample_idx.shape[0] - 1

    assert len(shuffle_idx) == num_samples

    # Verify each shuffled sample index is in range
    assert np.all(shuffle_idx < num_samples)
    assert np.all(shuffle_idx >= 0)

    # Verify sample_idx boundaries are valid
    for i in range(num_samples):
      doc_pos, offset = sample_idx[i]
      assert 0 <= doc_pos < len(doc_idx), f"sample {i}: doc_pos {doc_pos} out of range"
      doc_id = doc_idx[doc_pos]
      assert 0 <= doc_id < len(doc_sizes), f"sample {i}: doc_id {doc_id} out of range"
      assert 0 <= offset < doc_sizes[doc_id], f"sample {i}: offset {offset} >= doc_size {doc_sizes[doc_id]}"

  def test_multi_shard_integration(self, tmp_dir):
    """Multi-shard dataset produces valid indices."""
    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir)
    for i in range(3):
      seqs = [np.arange(i * 30 + j * 10, i * 30 + (j + 1) * 10, dtype=np.int32) for j in range(3)]
      create_mmap_test_data(
          os.path.join(data_dir, f"shard_{i}"),
          seqs,
          doc_boundaries=[0, 1, 2, 3],
      )

    out_dir = os.path.join(tmp_dir, "output")
    os.makedirs(out_dir)
    result = convert(
        input_paths=[data_dir],
        output_dir=out_dir,
        seq_length=4,
        num_samples=15,
        seed=99,
    )

    sample_idx = np.load(result["sample_index"])
    assert sample_idx.shape[0] > 1  # at least one sample


# ===========================================================================
# Unit tests: convert with --split
# ===========================================================================

_SPLIT_NUM_DOCS = 20
_SPLIT_RATIOS = "0.5,0.3,0.2"  # 10, 6, 4 docs
_SPLIT_SEED = 1234
_SPLIT_SEQ_LENGTH = 8
_SPLIT_NUM_EPOCHS = 2


def _create_split_dataset(tmp_dir, num_docs=_SPLIT_NUM_DOCS, eod_id=0, seed=42):
  """Create synthetic data: num_docs documents with --append-eod, varying lengths."""
  os.makedirs(tmp_dir, exist_ok=True)
  rng = np.random.RandomState(seed)
  seqs = []
  for d in range(num_docs):
    length = 20 + d * 5  # 20, 25, 30, ...
    tokens = rng.randint(1, 10000, size=length, dtype=np.int32)
    tokens[-1] = eod_id  # simulate --append-eod
    seqs.append(tokens)
  prefix = os.path.join(tmp_dir, "data")
  doc_boundaries = list(range(num_docs + 1))  # 1 seq per doc
  create_mmap_test_data(prefix, seqs, doc_boundaries=doc_boundaries)
  return prefix, seqs


def _split_doc_range(num_docs, split_str, split_index):
  """Compute (start_doc, end_doc) for a given split_index."""
  ratios = [float(x) for x in split_str.split(",")]
  total_ratio = sum(ratios)
  ratios = [r / total_ratio for r in ratios]
  cumulative = [0.0]
  for r in ratios:
    cumulative.append(cumulative[-1] + r)
  start_doc = int(cumulative[split_index] * num_docs)
  end_doc = int(cumulative[split_index + 1] * num_docs)
  return start_doc, end_doc


def _run_convert_and_load_doc_index(tmp_dir, prefix, split_index):
  """Run convert() with split and return the document_index array."""
  out_dir = os.path.join(tmp_dir, f"npy_split_{split_index}")
  convert(
      [prefix],
      out_dir,
      seq_length=_SPLIT_SEQ_LENGTH,
      num_epochs=_SPLIT_NUM_EPOCHS,
      seed=_SPLIT_SEED,
      split=_SPLIT_RATIOS,
      split_index=split_index,
  )
  doc_path, _, _ = _discover_npy_indices(out_dir)
  return np.load(doc_path)


class TestConvertSplitDocumentIndex:
  """Verify convert() with --split produces correct global document IDs."""

  def test_no_split_document_ids_unchanged(self, tmp_dir):
    """Without split, document IDs span the full range [0, num_docs)."""
    prefix, _ = _create_split_dataset(tmp_dir)
    out_dir = os.path.join(tmp_dir, "npy_nosplit")
    convert(
        [prefix],
        out_dir,
        seq_length=_SPLIT_SEQ_LENGTH,
        num_epochs=_SPLIT_NUM_EPOCHS,
        seed=_SPLIT_SEED,
    )
    doc_path, _, _ = _discover_npy_indices(out_dir)
    doc_index = np.load(doc_path)

    assert set(doc_index).issubset(range(0, _SPLIT_NUM_DOCS))
    # All docs should appear at least once over 2 epochs
    assert set(doc_index) == set(range(0, _SPLIT_NUM_DOCS))

  def test_split_zero_document_ids_are_global(self, tmp_dir):
    """split_index=0 (train, docs 0-9): IDs should be in [0, 10)."""
    prefix, _ = _create_split_dataset(tmp_dir)
    doc_index = _run_convert_and_load_doc_index(tmp_dir, prefix, split_index=0)

    start_doc, end_doc = _split_doc_range(_SPLIT_NUM_DOCS, _SPLIT_RATIOS, 0)
    assert start_doc == 0
    assert end_doc == 10
    # For split_index=0, local == global since offset is 0
    assert set(doc_index).issubset(range(start_doc, end_doc))

  def test_split_nonzero_document_ids_are_global(self, tmp_dir):
    """split_index=1 (eval, docs 10-15): IDs should be in [10, 16)."""
    prefix, _ = _create_split_dataset(tmp_dir)
    doc_index = _run_convert_and_load_doc_index(tmp_dir, prefix, split_index=1)

    start_doc, end_doc = _split_doc_range(_SPLIT_NUM_DOCS, _SPLIT_RATIOS, 1)
    assert start_doc == 10
    assert end_doc == 16
    assert set(doc_index).issubset(range(start_doc, end_doc)), (
        f"Expected doc IDs in [{start_doc}, {end_doc}), " f"got {sorted(set(doc_index))}"
    )

  def test_split_last_document_ids_are_global(self, tmp_dir):
    """split_index=2 (test, docs 16-19): IDs should be in [16, 20)."""
    prefix, _ = _create_split_dataset(tmp_dir)
    doc_index = _run_convert_and_load_doc_index(tmp_dir, prefix, split_index=2)

    start_doc, end_doc = _split_doc_range(_SPLIT_NUM_DOCS, _SPLIT_RATIOS, 2)
    assert start_doc == 16
    assert end_doc == 20
    assert set(doc_index).issubset(range(start_doc, end_doc)), (
        f"Expected doc IDs in [{start_doc}, {end_doc}), " f"got {sorted(set(doc_index))}"
    )
