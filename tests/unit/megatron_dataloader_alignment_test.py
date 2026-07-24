"""Element-wise compatibility checks against Megatron-Core's GPTDataset.

These tests deliberately use Megatron-Core as the oracle.  They cover the
three persistent mmap_npy indices, the actual next-token samples returned by
the data source, and the blend scheduler.  They are skipped when the optional
Megatron-Core test dependency is unavailable.
"""

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("megatron.core")

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

from maxtext.input_pipeline._megatron_blending import MegatronBlendedDataSource, build_blending_indices
from maxtext.input_pipeline._mmap_datasource import MMapDatasetConfig, MegatronNpyDataSource, _discover_npy_indices
from maxtext.input_pipeline._mmap_index_utils import (
    build_document_index,
    build_sample_index,
    build_shuffle_index,
    convert,
    get_document_sizes,
    parse_split_range,
)
from tests.unit.mmap_test_utils import create_mmap_test_data

pytestmark = [pytest.mark.cpu_only, pytest.mark.megatron_alignment]


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
def test_document_index_matches_megatron(num_docs, num_epochs, seed):
  expected_rng = np.random.RandomState(seed)
  expected = _build_document_index(np.arange(num_docs, dtype=np.int32), num_epochs, expected_rng, False)
  actual = build_document_index(num_docs, num_epochs, seed)
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "sizes,seq_length,num_epochs,seed",
    [([100], 8, 1, 42), ([50, 50], 16, 2, 1234), ([20, 30, 15], 8, 2, 42)],
)
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
    for sample_id in range(len(actual)):
      np.testing.assert_array_equal(actual[sample_id]["text"], _raw_megatron_tokens(expected, sample_id))


@pytest.mark.parametrize("weights,size", [([0.7, 0.3], 50), ([0.5, 0.3, 0.2], 80)])
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


def test_shuffle_index_helper_matches_megatron():
  """Keep a direct unit-level check independent of on-disk index creation."""
  seed, num_samples, total_size = 1234, 30, 50
  expected = _build_shuffle_index(num_samples, total_size, np.random.RandomState(seed))
  actual = build_shuffle_index(num_samples, total_size, seed)
  np.testing.assert_array_equal(actual, expected)
