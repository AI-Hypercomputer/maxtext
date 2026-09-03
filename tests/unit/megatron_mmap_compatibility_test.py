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

"""Compatibility tests for Megatron mmap indices, blending, and tooling."""

import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

try:
  from megatron.core.datasets import helpers as megatron_helpers
  from megatron.core.datasets.gpt_dataset import (
      GPTDataset,
      GPTDatasetConfig,
      _build_document_index,
      _build_shuffle_index,
      _get_ltor_masks_and_position_ids,
  )
  from megatron.core.datasets.indexed_dataset import IndexedDataset
  from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer
  from megatron.core.datasets.utils import Split

  MEGATRON_CORE_AVAILABLE = True
except ImportError:
  MegatronLegacyTokenizer = object
  MEGATRON_CORE_AVAILABLE = False

requires_megatron_core = pytest.mark.skipif(
    not MEGATRON_CORE_AVAILABLE,
    reason="megatron.core is required for element-wise compatibility checks",
)


from maxtext.input_pipeline import _megatron_blending
from maxtext.input_pipeline import _mmap_index_utils
from tools.data_processing.mmap_index_builder import (
    build_document_index,
    build_indices,
    build_sample_index,
    build_shuffle_index,
    compute_index_hash,
    compute_num_epochs,
    convert,
    convert_blend,
    discover_shards,
    should_separate_last_epoch,
    get_document_sizes,
)
from maxtext.input_pipeline._megatron_blending import MegatronBlendedDataSource, build_blending_indices
from tests.unit.mmap_test_utils import create_mmap_test_data
from maxtext.input_pipeline._mmap_datasource import MMapDatasetConfig, MegatronNpyDataSource, _discover_npy_indices
from maxtext.input_pipeline._mmap_index_utils import parse_split_range

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


class TestMMapIndexUtilityHelpers:
  """Tests for canonical helpers in _mmap_index_utils."""

  def test_total_tokens_num_documents_and_split_tokens(self, tmp_dir):
    prefix_a = os.path.join(tmp_dir, "a")
    create_mmap_test_data(
        prefix_a,
        [
            np.array([1, 2], dtype=np.int16),
            np.array([], dtype=np.int16),
            np.array([3, 4, 5], dtype=np.int16),
        ],
        dtype=np.int16,
        doc_boundaries=[0, 1, 2, 3],
    )
    prefix_b = os.path.join(tmp_dir, "b")
    create_mmap_test_data(
        prefix_b,
        [np.array([6, 7, 8, 9], dtype=np.int32)],
        doc_boundaries=[0, 1],
    )

    assert _mmap_index_utils.get_total_tokens([prefix_a, prefix_b]) == 9
    assert _mmap_index_utils.get_num_documents([prefix_a, prefix_b]) == 4
    assert _mmap_index_utils.get_split_tokens([prefix_a, prefix_b], 1, 2) == 0
    assert _mmap_index_utils.get_split_tokens([prefix_a, prefix_b], 2, 4) == 7

  def test_resolve_shard_prefixes_rejects_empty_input(self):
    with pytest.raises(FileNotFoundError, match="No .idx files found"):
      _mmap_index_utils.resolve_shard_prefixes([])

  def test_primary_process_uses_jax_when_available_and_falls_back_without_it(self):
    fake_jax = SimpleNamespace(process_index=mock.Mock(return_value=1))
    with mock.patch.dict(sys.modules, {"jax": fake_jax}):
      assert _mmap_index_utils.is_primary_process() is False
    fake_jax.process_index.assert_called_once_with()

    with mock.patch.dict(sys.modules, {"jax": None}):
      assert _mmap_index_utils.is_primary_process() is True

  def test_parse_split_range_uses_round_and_validates_index(self):
    assert _mmap_index_utils.parse_split_range("0.5,0.25,0.25", 1, 10) == (5, 8)
    with pytest.raises(ValueError, match="out of range"):
      _mmap_index_utils.parse_split_range("99,1", 2, 10)

  def test_weight_helpers_match_megatron_buffer_formula(self):
    weights = _mmap_index_utils._normalize_weights([2.0, 6.0])  # pylint: disable=protected-access
    np.testing.assert_allclose(weights, [0.25, 0.75])
    assert _mmap_index_utils.compute_blend_buffers(10, weights, margin=0.5) == [4, 9]

  def test_build_metadata_sorts_inputs_and_preserves_request_fields(self):
    metadata = _mmap_index_utils.build_metadata(
        file_hash="abc",
        input_paths=["b", "a"],
        seq_length=8,
        num_samples=10,
        num_epochs=2,
        seed=7,
        split="99,1",
        split_index=1,
        add_extra_token=1,
        num_docs=4,
        total_samples=9,
        source="unit",
    )

    assert metadata["hash"] == "abc"
    assert metadata["input_paths"] == ["a", "b"]
    assert metadata["seq_length"] == 8
    assert metadata["num_samples_requested"] == 10
    assert metadata["source"] == "unit"
    assert "created_at" in metadata

  def test_build_sample_index_zero_samples_uses_int64_when_needed(self):
    doc_sizes = np.array([np.iinfo(np.int32).max + 1], dtype=np.int64)
    doc_index = np.array([0], dtype=np.int32)
    sample_index = build_sample_index(doc_sizes, doc_index, seq_length=np.iinfo(np.int32).max + 1)
    assert sample_index.shape == (1, 2)
    assert sample_index.dtype == np.int64

  def test_epoch_helpers_cover_edge_cases(self):
    assert compute_num_epochs(total_tokens=10, num_samples=0, seq_length=8) == 1
    assert should_separate_last_epoch(num_epochs=1, tokens_per_epoch=100, num_samples=10, seq_length=8) is False
    assert should_separate_last_epoch(num_epochs=2, tokens_per_epoch=100, num_samples=13, seq_length=8) is True

  def test_index_hash_sorts_flat_paths_but_preserves_nested_paths(self):
    flat_a = compute_index_hash(["b", "a"], 1, False, 7, 8)
    flat_b = compute_index_hash(["a", "b"], 1, False, 7, 8)
    nested_a = compute_index_hash([["b", "a"]], 1, False, 7, 8)
    nested_b = compute_index_hash([["a", "b"]], 1, False, 7, 8)
    assert flat_a == flat_b
    assert nested_a != nested_b

  def test_build_indices_validates_mode_arguments(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "data")
    create_mmap_test_data(prefix, [np.array([1, 2, 3, 4], dtype=np.int32)])

    with pytest.raises(ValueError, match="either num_samples or num_epochs"):
      build_indices([prefix], seq_length=2)
    with pytest.raises(ValueError, match="Cannot specify both"):
      build_indices([prefix], seq_length=2, num_samples=1, num_epochs=1)

  def test_build_indices_rejects_empty_split_and_empty_documents(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "data")
    create_mmap_test_data(
        prefix,
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        doc_boundaries=[0, 1, 2],
    )
    with pytest.raises(ValueError, match="No documents found.*after applying split"):
      build_indices([prefix], seq_length=2, num_epochs=1, split="1,0", split_index=1)

    empty_prefix = os.path.join(tmp_dir, "empty")
    create_mmap_test_data(empty_prefix, [np.array([], dtype=np.int32)], doc_boundaries=[0, 1])
    with pytest.raises(ValueError, match="Total token count is 0"):
      build_indices([empty_prefix], seq_length=2, num_epochs=1)

  def test_build_indices_warns_for_non_default_extra_token(self, tmp_dir, caplog):
    prefix = os.path.join(tmp_dir, "extra_token")
    create_mmap_test_data(prefix, [np.arange(12, dtype=np.int32)])

    build_indices([prefix], seq_length=4, num_epochs=1, add_extra_token=0)

    assert "add_extra_token=0" in caplog.text

  def test_save_indices_atomic_writes_metadata(self, tmp_dir):
    paths = _mmap_index_utils.save_indices_atomic(
        tmp_dir,
        "abc",
        np.array([0], dtype=np.int32),
        np.array([[0, 0], [0, 1]], dtype=np.int32),
        np.array([0], dtype=np.uint32),
        metadata={"source": "unit_test"},
    )

    assert set(paths) == {"document_index", "sample_index", "shuffle_index", "metadata"}
    assert os.path.isfile(paths["metadata"])


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


class TestMegatronBlendingHelpers:
  """Focused validation coverage for _megatron_blending."""

  def test_build_blending_indices_validates_inputs(self):
    dataset_index = np.zeros(2, dtype=np.int16)
    sample_index = np.zeros(2, dtype=np.int64)

    with pytest.raises(ValueError, match="non-negative"):
      _megatron_blending.build_blending_indices(dataset_index, sample_index, np.array([1.0]), 1, -1)
    with pytest.raises(ValueError, match="shape"):
      _megatron_blending.build_blending_indices(np.zeros(1, dtype=np.int16), sample_index, np.array([1.0]), 1, 2)
    with pytest.raises(ValueError, match="positive"):
      _megatron_blending.build_blending_indices(dataset_index, sample_index, np.array([], dtype=np.float64), 0, 2)
    with pytest.raises(ValueError, match="one positive"):
      _megatron_blending.build_blending_indices(dataset_index, sample_index, np.array([0.0]), 1, 2)
    with pytest.raises(ValueError, match="sum to 1"):
      _megatron_blending.build_blending_indices(dataset_index, sample_index, np.array([0.4, 0.4]), 2, 2)

  def test_normalize_datasets_filters_zero_and_validates_lengths(self):
    datasets, weights, lengths = _megatron_blending._normalize_datasets(  # pylint: disable=protected-access
        [["drop"], ["keep"]],
        [0.0, 2.0],
        [1, 3],
    )
    assert datasets == [["keep"]]
    np.testing.assert_allclose(weights, [1.0])
    assert lengths == [3]

    with pytest.raises(ValueError, match="At least one dataset"):
      _megatron_blending._normalize_datasets([], [])  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="dataset_lengths must match"):
      _megatron_blending._normalize_datasets([[1]], [1.0], [1, 2])  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="non-negative"):
      _megatron_blending._normalize_datasets([[1]], [-1.0])  # pylint: disable=protected-access

  def test_infer_size_and_datasource_index_errors(self):
    source = MegatronBlendedDataSource([list(range(2)), list(range(10))], weights=[0.5, 0.5])
    assert len(source) == 4
    assert source[-1] in range(10)

    with pytest.raises(ValueError, match="positive and match"):
      _megatron_blending._infer_size(np.array([1.0]), [0])  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="would be empty"):
      _megatron_blending._infer_size(np.array([0.1]), [0.05])  # pylint: disable=protected-access
    with pytest.raises(IndexError, match="out of range"):
      _ = source[len(source)]

  def test_validate_indices_rejects_corrupt_arrays(self):
    valid_dataset_index = np.array([0, 1, 0], dtype=np.int16)
    valid_sample_index = np.array([0, 0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="one-dimensional"):
      _megatron_blending._validate_indices(  # pylint: disable=protected-access
          valid_dataset_index.reshape(1, 3), valid_sample_index, [2, 1], 3
      )
    with pytest.raises(ValueError, match="unexpected size"):
      _megatron_blending._validate_indices(  # pylint: disable=protected-access
          valid_dataset_index, valid_sample_index, [2, 1], 2
      )
    with pytest.raises(ValueError, match="integer dtypes"):
      _megatron_blending._validate_indices(  # pylint: disable=protected-access
          valid_dataset_index.astype(np.float32), valid_sample_index, [2, 1], 3
      )
    with pytest.raises(ValueError, match="out of range"):
      _megatron_blending._validate_indices(  # pylint: disable=protected-access
          np.array([0, 2], dtype=np.int16), np.array([0, 0]), [1, 1], 2
      )
    with pytest.raises(ValueError, match="exceed dataset"):
      _megatron_blending._validate_indices(  # pylint: disable=protected-access
          np.array([0, 0], dtype=np.int16), np.array([0, 2]), [2], 2
      )
    with pytest.raises(ValueError, match="not contiguous"):
      _megatron_blending._validate_indices(  # pylint: disable=protected-access
          np.array([0, 0], dtype=np.int16), np.array([0, 2]), [3], 2
      )

  def test_find_index_pair_in_dir_variants(self, tmp_dir):
    with pytest.raises(FileNotFoundError, match="does not exist"):
      _megatron_blending._find_index_pair_in_dir(  # pylint: disable=protected-access
          os.path.join(tmp_dir, "missing"), "train"
      )

    fixed_dir = os.path.join(tmp_dir, "fixed")
    os.makedirs(fixed_dir)
    np.save(os.path.join(fixed_dir, "dataset_index.npy"), np.array([0], dtype=np.int16))
    np.save(os.path.join(fixed_dir, "dataset_sample_index.npy"), np.array([0], dtype=np.int64))
    fixed_pair = _megatron_blending._find_index_pair_in_dir(fixed_dir, "train")  # pylint: disable=protected-access
    assert [path.name for path in fixed_pair] == ["dataset_index.npy", "dataset_sample_index.npy"]

    split_dir = os.path.join(tmp_dir, "split")
    os.makedirs(split_dir)
    np.save(os.path.join(split_dir, "abc-BlendedDataset-train-dataset_index.npy"), np.array([0], dtype=np.int16))
    np.save(os.path.join(split_dir, "abc-BlendedDataset-train-dataset_sample_index.npy"), np.array([0], dtype=np.int64))
    split_pair = _megatron_blending._find_index_pair_in_dir(split_dir, "train")  # pylint: disable=protected-access
    assert split_pair[0].name.endswith("dataset_index.npy")

    np.save(os.path.join(split_dir, "def-BlendedDataset-train-dataset_index.npy"), np.array([0], dtype=np.int16))
    np.save(os.path.join(split_dir, "def-BlendedDataset-train-dataset_sample_index.npy"), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="Multiple blend index pairs"):
      _megatron_blending._find_index_pair_in_dir(split_dir, "train")  # pylint: disable=protected-access

  def test_build_and_save_blend_indices_validates_size(self, tmp_dir):
    with pytest.raises(ValueError, match="size must be positive"):
      _megatron_blending.build_and_save_blend_indices(tmp_dir, weights=[1.0], dataset_lengths=[3], size=0)

  def test_cache_paths_are_stable_and_split_specific(self, tmp_dir):
    weights = np.array([0.25, 0.75], dtype=np.float64)
    train_paths = _megatron_blending._cache_paths(  # pylint: disable=protected-access
        tmp_dir, weights, [4, 8], 6, "train"
    )
    eval_paths = _megatron_blending._cache_paths(tmp_dir, weights, [4, 8], 6, "eval")  # pylint: disable=protected-access

    assert train_paths == _megatron_blending._cache_paths(  # pylint: disable=protected-access
        tmp_dir, weights, [4, 8], 6, "train"
    )
    assert train_paths != eval_paths
    assert train_paths[0].name.endswith("dataset_index.npy")
    assert train_paths[1].name.endswith("dataset_sample_index.npy")

  def test_build_and_save_blend_indices_writes_fixed_runtime_pair(self, tmp_dir):
    paths = _megatron_blending.build_and_save_blend_indices(
        tmp_dir,
        weights=[0.0, 1.0],
        dataset_lengths=[1, 5],
        size=3,
    )

    assert paths["dataset_index"].name == "dataset_index.npy"
    assert paths["dataset_sample_index"].name == "dataset_sample_index.npy"
    np.testing.assert_array_equal(np.load(paths["dataset_index"]), np.zeros(3, dtype=np.int16))
    np.testing.assert_array_equal(np.load(paths["dataset_sample_index"]), np.arange(3, dtype=np.int64))

  def test_datasource_uses_valid_cache_hit(self, tmp_dir, monkeypatch):
    source = MegatronBlendedDataSource([list(range(10))], weights=[1.0], size=3, cache_dir=tmp_dir)
    assert [source[i] for i in range(len(source))] == [0, 1, 2]

    def fail_build(*_args, **_kwargs):
      raise AssertionError("cache hit should not rebuild")

    monkeypatch.setattr(_megatron_blending, "build_blending_indices", fail_build)
    cached = MegatronBlendedDataSource([list(range(10))], weights=[1.0], size=3, cache_dir=tmp_dir)
    assert [cached[i] for i in range(len(cached))] == [0, 1, 2]

  def test_datasource_skips_cache_write_on_non_primary_process(self, tmp_dir):
    with mock.patch.object(_mmap_index_utils, "is_primary_process", return_value=False):
      source = MegatronBlendedDataSource([list(range(10))], weights=[1.0], size=3, cache_dir=tmp_dir)

    assert [source[i] for i in range(len(source))] == [0, 1, 2]
    assert not any(name.endswith(".npy") for name in os.listdir(tmp_dir))


class TestConvertBlendValidation:
  """Additional convert_blend edge cases."""

  def test_convert_blend_rejects_bad_weights(self, tmp_dir):
    with pytest.raises(ValueError, match="non-empty"):
      convert_blend([], total_samples=1, seq_length=4)

    with pytest.raises(ValueError, match="non-negative"):
      convert_blend(
          [{"input": [tmp_dir], "weight": -1.0, "output_dir": os.path.join(tmp_dir, "out")}],
          total_samples=1,
          seq_length=4,
      )

    with pytest.raises(ValueError, match="positive"):
      convert_blend(
          [{"input": [tmp_dir], "weight": 0.0, "output_dir": os.path.join(tmp_dir, "out")}],
          total_samples=1,
          seq_length=4,
      )

  def test_convert_blend_default_workers_without_dispatch_output(self, tmp_dir):
    prefix = os.path.join(tmp_dir, "single")
    create_mmap_test_data(prefix, [np.arange(16, dtype=np.int32)])

    results = convert_blend(
        [{"input": [prefix], "weight": 1.0, "output_dir": os.path.join(tmp_dir, "out")}],
        total_samples=2,
        seq_length=4,
        seed=7,
    )

    assert len(results) == 1
    assert results[0]["buffer_samples"] == 3
    assert not os.path.exists(os.path.join(tmp_dir, "dataset_index.npy"))


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


# ===========================================================================
# Element-wise comparisons with megatron.core
# ===========================================================================


class _StubTokenizer(MegatronLegacyTokenizer):
  """The smallest tokenizer surface required by ``GPTDataset``."""

  def __init__(self, eod):
    super().__init__(None)
    self._eod = eod

  @property
  def vocab_size(self):
    return 50000

  @property
  def vocab(self):
    raise NotImplementedError

  @property
  def inv_vocab(self):
    raise NotImplementedError

  @property
  def eod(self):
    return self._eod

  def tokenize(self, text):
    raise NotImplementedError

  def detokenize(self, ids):
    raise NotImplementedError


def _create_eod_dataset(tmp_dir, num_docs=10, eod_id=0, seed=123):
  """Create a valid Megatron indexed dataset with pre-appended EOD tokens."""
  os.makedirs(tmp_dir, exist_ok=True)
  rng = np.random.RandomState(seed)
  sequences = []
  for doc_id in range(num_docs):
    tokens = rng.randint(1, 10000, size=20 + doc_id * 3, dtype=np.int32)
    tokens[-1] = eod_id
    sequences.append(tokens)
  prefix = os.path.join(tmp_dir, "data")
  create_mmap_test_data(prefix, sequences, doc_boundaries=list(range(num_docs + 1)))
  return prefix


def _megatron_dataset(prefix, seq_length, seed, eod_id):
  """Construct the reference Megatron GPTDataset used by parity tests."""
  indexed_dataset = IndexedDataset(prefix, multimodal=False, mmap=True)
  config = GPTDatasetConfig(
      random_seed=seed,
      sequence_length=seq_length,
      reset_position_ids=False,
      reset_attention_mask=False,
      eod_mask_loss=False,
      tokenizer=_StubTokenizer(eod_id),
  )
  return GPTDataset(
      indexed_dataset=indexed_dataset,
      dataset_path=prefix,
      indexed_indices=np.arange(indexed_dataset.document_indices.shape[0] - 1, dtype=np.int32),
      num_samples=None,
      index_split=Split.train,
      config=config,
  )


def _raw_megatron_tokens(dataset, sample_id):
  """Join GPTDataset's input and label views back into seq_length + 1 tokens."""
  sample = dataset[sample_id]
  tokens = sample["tokens"].numpy().astype(np.int32)
  labels = sample["labels"].numpy().astype(np.int32)
  return np.concatenate([tokens[:1], labels])


@pytest.mark.parametrize("num_docs,num_epochs,seed", [(10, 1, 42), (10, 3, 42), (50, 2, 1234)])
@pytest.mark.megatron_alignment
@requires_megatron_core
def test_document_index_matches_megatron(num_docs, num_epochs, seed):
  expected_rng = np.random.RandomState(seed)
  expected = _build_document_index(np.arange(num_docs, dtype=np.int32), num_epochs, expected_rng, False)
  actual = build_document_index(num_docs, num_epochs, seed)
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "sizes,seq_length,num_epochs,seed",
    [([100], 8, 1, 42), ([50, 50], 16, 2, 1234), ([20, 30, 15], 8, 2, 42)],
)
@pytest.mark.megatron_alignment
@requires_megatron_core
def test_sample_index_matches_megatron_cpp(sizes, seq_length, num_epochs, seed):
  sizes = np.asarray(sizes, dtype=np.int32)
  rng = np.random.RandomState(seed)
  document_index = _build_document_index(np.arange(len(sizes), dtype=np.int32), num_epochs, rng, False)
  expected = megatron_helpers.build_sample_idx(
      sizes,
      document_index,
      seq_length,
      num_epochs=num_epochs,
      tokens_per_epoch=int(sizes.sum()),
      drop_last_partial_sequence=True,
      add_extra_token_to_sequence=True,
  )
  actual = build_sample_index(sizes.astype(np.int64), document_index, seq_length, add_extra_token=1)
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.megatron_alignment
@requires_megatron_core
def test_persisted_shuffle_index_matches_megatron_rng_flow():
  """The production builder must consume one RNG exactly as Megatron does."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    prefix = _create_eod_dataset(tmp_dir)
    output_dir = os.path.join(tmp_dir, "indices")
    seq_length, seed, num_epochs = 8, 42, 2
    convert([prefix], output_dir, seq_length=seq_length, num_epochs=num_epochs, seed=seed)
    _, _, actual_path = _discover_npy_indices(output_dir)
    actual = np.load(actual_path)

    sizes = get_document_sizes([prefix]).astype(np.int32)
    rng = np.random.RandomState(seed)
    document_index = _build_document_index(np.arange(len(sizes), dtype=np.int32), num_epochs, rng, False)
    sample_index = megatron_helpers.build_sample_idx(
        sizes,
        document_index,
        seq_length,
        num_epochs=num_epochs,
        tokens_per_epoch=int(sizes.sum()),
        drop_last_partial_sequence=True,
        add_extra_token_to_sequence=True,
    )
    expected = _build_shuffle_index(sample_index.shape[0] - 1, sample_index.shape[0] - 1, rng)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.megatron_alignment
@requires_megatron_core
def test_all_mmap_npy_tokens_match_real_megatron_gpt_dataset():
  """Each mmap_npy sample must equal Megatron's actual GPTDataset output."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    eod_id, seq_length, seed = 0, 8, 42
    prefix = _create_eod_dataset(tmp_dir, eod_id=eod_id)
    output_dir = os.path.join(tmp_dir, "indices")
    convert([prefix], output_dir, seq_length=seq_length, num_epochs=1, seed=seed)

    actual = MegatronNpyDataSource(output_dir, prefix, eod_id=eod_id, seq_length=seq_length)
    expected = _megatron_dataset(prefix, seq_length, seed, eod_id)
    assert len(actual) == len(expected)
    for sample_id, actual_sample in enumerate(actual):
      np.testing.assert_array_equal(actual_sample["text"], _raw_megatron_tokens(expected, sample_id))


@pytest.mark.parametrize("weights,size", [([0.7, 0.3], 50), ([0.5, 0.3, 0.2], 80)])
@pytest.mark.megatron_alignment
@requires_megatron_core
def test_blend_indices_match_megatron_cpp(weights, size):
  """The blend dispatcher must use exactly Megatron's greedy schedule."""
  normalized_weights = np.asarray(weights, dtype=np.float64)
  actual_dataset_indices = np.zeros(size, dtype=np.int16)
  actual_sample_indices = np.zeros(size, dtype=np.int64)
  build_blending_indices(
      actual_dataset_indices,
      actual_sample_indices,
      normalized_weights,
      len(normalized_weights),
      size,
  )

  expected_dataset_indices = np.zeros(size, dtype=np.int16)
  expected_sample_indices = np.zeros(size, dtype=np.int64)
  megatron_helpers.build_blending_indices(
      expected_dataset_indices,
      expected_sample_indices,
      normalized_weights,
      len(normalized_weights),
      size,
      False,
  )
  np.testing.assert_array_equal(actual_dataset_indices, expected_dataset_indices)
  np.testing.assert_array_equal(actual_sample_indices, expected_sample_indices)


@pytest.mark.megatron_alignment
@requires_megatron_core
def test_blended_data_source_tokens_match_megatron_dispatch():
  """Blend dispatch selects the same underlying ``L + 1`` token sample as Megatron."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    eod_id, seq_length, seed, size = 0, 8, 42, 24
    prefix_a = _create_eod_dataset(os.path.join(tmp_dir, "a"), num_docs=10, eod_id=eod_id, seed=11)
    prefix_b = _create_eod_dataset(os.path.join(tmp_dir, "b"), num_docs=10, eod_id=eod_id, seed=29)
    output_a, output_b = os.path.join(tmp_dir, "indices_a"), os.path.join(tmp_dir, "indices_b")
    convert([prefix_a], output_a, seq_length=seq_length, num_epochs=1, seed=seed)
    convert([prefix_b], output_b, seq_length=seq_length, num_epochs=1, seed=seed)

    actual_sources = [
        MegatronNpyDataSource(output_a, prefix_a, eod_id=eod_id, seq_length=seq_length),
        MegatronNpyDataSource(output_b, prefix_b, eod_id=eod_id, seq_length=seq_length),
    ]
    actual = MegatronBlendedDataSource(actual_sources, weights=[0.7, 0.3], size=size)
    expected_sources = [
        _megatron_dataset(prefix_a, seq_length, seed, eod_id),
        _megatron_dataset(prefix_b, seq_length, seed, eod_id),
    ]

    expected_dataset_indices = np.zeros(size, dtype=np.int16)
    expected_sample_indices = np.zeros(size, dtype=np.int64)
    megatron_helpers.build_blending_indices(
        expected_dataset_indices,
        expected_sample_indices,
        np.asarray([0.7, 0.3], dtype=np.float64),
        2,
        size,
        False,
    )
    np.testing.assert_array_equal(actual._dataset_index, expected_dataset_indices)  # pylint: disable=protected-access
    np.testing.assert_array_equal(actual._dataset_sample_index, expected_sample_indices)  # pylint: disable=protected-access
    for sample_id in range(size):
      dataset_id = int(expected_dataset_indices[sample_id])
      expected = _raw_megatron_tokens(expected_sources[dataset_id], int(expected_sample_indices[sample_id]))
      np.testing.assert_array_equal(actual[sample_id]["text"], expected)


def _mmap_npy_host_samples(pattern, seq_length, host_index, host_count, seed):
  """Read raw mmap_npy samples through the production host-sharding path."""
  from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

  dataset = get_datasets(
      data_file_pattern=pattern,
      data_file_type="mmap_npy",
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
      dataset_config=MMapDatasetConfig(
          max_target_length=seq_length,
          eod_id=0,
          mmap_split_sentences=False,
          seed=seed,
      ),
  )
  return [sample["text"] for sample in dataset]


def _reassemble_host_strides(pattern, seq_length, seed, host_count):
  """Reconstruct the global sequence from the production per-host slices."""
  global_samples = _mmap_npy_host_samples(pattern, seq_length, host_index=0, host_count=1, seed=seed)
  reassembled = [None] * len(global_samples)
  for host_index in range(host_count):
    host_samples = _mmap_npy_host_samples(pattern, seq_length, host_index, host_count, seed)
    for local_index, sample in enumerate(host_samples):
      reassembled[host_index + local_index * host_count] = sample
  assert all(sample is not None for sample in reassembled)
  return global_samples, reassembled


@pytest.mark.megatron_alignment
@requires_megatron_core
def test_mmap_npy_host_shards_reassemble_the_global_order():
  """The single-source mmap_npy path must also preserve global stride order."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    eod_id, seq_length, seed, host_count = 0, 8, 42, 4
    prefix = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=eod_id, seed=11)
    output_dir = os.path.join(tmp_dir, "indices")
    convert([prefix], output_dir, seq_length=seq_length, num_epochs=1, seed=seed)

    global_samples, reassembled = _reassemble_host_strides(f"{output_dir}|{prefix}", seq_length, seed, host_count)
    for index, (expected, actual) in enumerate(zip(global_samples, reassembled)):
      np.testing.assert_array_equal(actual, expected, err_msg=f"Global mmap_npy sample {index}")


@pytest.mark.megatron_alignment
@requires_megatron_core
def test_blended_mmap_npy_host_shards_reassemble_the_global_order():
  """Host striding must partition, rather than independently rebuild, a blend."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    eod_id, seq_length, seed, host_count = 0, 8, 42, 4
    prefix_a = _create_eod_dataset(os.path.join(tmp_dir, "a"), num_docs=10, eod_id=eod_id, seed=11)
    prefix_b = _create_eod_dataset(os.path.join(tmp_dir, "b"), num_docs=10, eod_id=eod_id, seed=29)
    output_a, output_b = os.path.join(tmp_dir, "indices_a"), os.path.join(tmp_dir, "indices_b")
    convert([prefix_a], output_a, seq_length=seq_length, num_epochs=1, seed=seed)
    convert([prefix_b], output_b, seq_length=seq_length, num_epochs=1, seed=seed)
    pattern = f"{output_a}|{prefix_a},0.7;{output_b}|{prefix_b},0.3"

    global_samples, reassembled = _reassemble_host_strides(pattern, seq_length, seed, host_count)
    for index, (expected, actual) in enumerate(zip(global_samples, reassembled)):
      np.testing.assert_array_equal(actual, expected, err_msg=f"Global blend sample {index}")


@pytest.mark.parametrize(
    "reset_attention_mask,eod_mask_loss",
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.megatron_alignment
@requires_megatron_core
def test_mmap_npy_loss_mask_and_positions_match_megatron(reset_attention_mask, eod_mask_loss):
  """The mmap_npy transform must retain Megatron's EOD loss and position semantics."""
  torch = pytest.importorskip("torch")
  from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets  # pylint: disable=import-outside-toplevel

  eod_id = 0
  tokens = np.array([10, 20, eod_id, 30, 40, 50, eod_id, 60, 70], dtype=np.int32)
  inputs = tokens[:-1]
  _, megatron_loss_mask, megatron_positions = _get_ltor_masks_and_position_ids(
      torch.from_numpy(inputs.astype(np.int64)),
      eod_id,
      reset_position_ids=reset_attention_mask,
      reset_attention_mask=reset_attention_mask,
      eod_mask_loss=eod_mask_loss,
      create_attention_mask=False,
  )

  actual = MegatronSplitInputsTargets(
      eod_id=eod_id,
      reset_attention_mask=reset_attention_mask,
      eod_mask_loss=eod_mask_loss,
  ).map({"text": tokens})
  actual_loss_mask = (actual["targets_segmentation"] > 0).astype(np.float32)
  np.testing.assert_array_equal(actual_loss_mask, megatron_loss_mask.numpy())
  np.testing.assert_array_equal(actual["inputs_position"], megatron_positions.numpy().astype(np.int32))


@pytest.mark.parametrize("split_index", [0, 1])
@pytest.mark.megatron_alignment
@requires_megatron_core
def test_split_indices_match_megatron_on_a_rounding_boundary(split_index):
  """A ``99,1`` split must use Megatron's round-based document boundaries."""
  from megatron.core.datasets.blended_megatron_dataset_config import (  # pylint: disable=import-outside-toplevel
      convert_split_vector_to_split_matrix,
      parse_and_normalize_split,
  )

  with tempfile.TemporaryDirectory() as tmp_dir:
    eod_id, seq_length, seed, num_docs = 0, 8, 42, 101
    prefix = _create_eod_dataset(tmp_dir, num_docs=num_docs, eod_id=eod_id, seed=7)
    split = "99,1"
    output_dir = os.path.join(tmp_dir, f"split_{split_index}")
    convert(
        [prefix],
        output_dir,
        seq_length=seq_length,
        num_epochs=1,
        seed=seed,
        split=split,
        split_index=split_index,
    )
    actual_document, actual_sample, actual_shuffle = (np.load(path) for path in _discover_npy_indices(output_dir))

    split_bookend = convert_split_vector_to_split_matrix(parse_and_normalize_split(split))[split_index]
    expected_boundary = (
        int(round(split_bookend[0] * num_docs)),
        int(round(split_bookend[1] * num_docs)),
    )
    start_doc, end_doc = parse_split_range(split, split_index, num_docs)
    assert (start_doc, end_doc) == expected_boundary
    assert (start_doc, end_doc) == ((0, 100) if split_index == 0 else (100, 101))
    sizes = get_document_sizes([prefix]).astype(np.int32)
    rng = np.random.RandomState(seed)
    expected_document = _build_document_index(
        np.arange(start_doc, end_doc, dtype=np.int32),
        1,
        rng,
        False,
    )
    expected_sample = megatron_helpers.build_sample_idx(
        sizes,
        expected_document,
        seq_length,
        num_epochs=1,
        tokens_per_epoch=int(sizes[start_doc:end_doc].sum()),
        drop_last_partial_sequence=True,
        add_extra_token_to_sequence=True,
    )
    expected_shuffle = _build_shuffle_index(expected_sample.shape[0] - 1, expected_sample.shape[0] - 1, rng)
    np.testing.assert_array_equal(actual_document, expected_document)
    np.testing.assert_array_equal(actual_sample, expected_sample)
    np.testing.assert_array_equal(actual_shuffle, expected_shuffle)


@pytest.mark.megatron_alignment
@requires_megatron_core
def test_shuffle_index_helper_matches_megatron():
  """Keep a direct unit-level check independent of on-disk index creation."""
  seed, num_samples, total_size = 1234, 30, 50
  expected = _build_shuffle_index(num_samples, total_size, np.random.RandomState(seed))
  actual = build_shuffle_index(num_samples, total_size, seed)
  np.testing.assert_array_equal(actual, expected)
