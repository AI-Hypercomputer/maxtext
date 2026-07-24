"""Tests for MegatronNpyDataSource and its helper functions."""

import os
import pickle
import tempfile
from unittest import mock

import numpy as np
import pytest

from maxtext.input_pipeline._mmap_datasource import (
    MMapDatasetConfig,
    MegatronNpyDataSource,
    create_mmap_npy_source,
    _discover_npy_indices,
    _ensure_npy_indices,
    _resolve_bin_prefixes,
)
from tools.data_processing.mmap_index_builder import convert
from tests.unit.mmap_test_utils import create_mmap_test_data

pytestmark = pytest.mark.cpu_only

# pylint: disable=redefined-outer-name


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
    for index in range(len(disk_source)):
      np.testing.assert_array_equal(restored_memory_source[index]["text"], disk_source[index]["text"])

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
      for index in range(len(offline)):
        np.testing.assert_array_equal(runtime[index]["text"], offline[index]["text"])


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
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
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
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
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
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
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
