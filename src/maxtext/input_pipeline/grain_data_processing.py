# Copyright 2023–2025 Google LLC
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

"""Input pipeline using Grain."""

from collections.abc import Sequence
from concurrent import futures
import dataclasses
import functools
import glob
import itertools
import json
import math
from pathlib import Path
import re
import sys
import zlib

import ml_collections

import jax
import numpy as np

import grain.python as grain
from grain.experimental import ElasticIterator

from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import grain_tokenizer
from maxtext.input_pipeline import dpo_utils
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline._mmap_datasource import MMapDatasetConfig, get_mmap_dataset, get_mmap_npy_dataset
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging


def construct_hf_dataset_path(hf_path: str, hf_train_files: str | None = None, split: str = "train") -> str:
  """
  Constructs a robust Hugging Face fsspec (hf://) path for dataset loading.
  """
  hf_path = hf_path.strip().strip("/")

  scheme_prefix = "hf://"
  if hf_path.startswith("hf://"):
    hf_path = hf_path[len("hf://") :]

  if not hf_path.startswith("datasets/"):
    hf_path = f"datasets/{hf_path}"

  match = re.match(r"^([^@]+)(@.+)?$", hf_path)
  repo_path = match.group(1) if match else hf_path
  revision_suffix = match.group(2) if match and match.group(2) else ""

  if hf_train_files:
    file_pattern = hf_train_files.lstrip("/")
  else:
    if split:
      file_pattern = f"**/*{split}*.parquet"
    else:
      file_pattern = "**/*.parquet"

  full_path = f"{scheme_prefix}{repo_path}{revision_suffix}/{file_pattern}"
  return full_path


def construct_tfds_tfrecord_path(dataset_path: str, dataset_name: str, split: str) -> str:
  """Constructs a glob for TFRecords in the standard TFDS prepared-data layout."""
  dataset_dir = dataset_name.strip().strip("/").replace(":", "/")
  path = f"{dataset_path.strip().rstrip('/')}/{dataset_dir}/*-{split}.tfrecord-*"
  max_logging.log(f"Automatically constructed Grain TFRecord path from TFDS configuration: {path}")
  return path


def find_data_files(data_file_pattern, hf_access_token=None):
  """Find data files matching the pattern."""
  if data_file_pattern.startswith("gs://"):
    data_files = gcs_utils.gcs_glob_pattern(data_file_pattern)
  elif data_file_pattern.startswith("hf://"):
    from huggingface_hub import HfFileSystem  # pylint: disable=import-outside-toplevel

    fs = HfFileSystem(token=hf_access_token)
    stripped_pattern = data_file_pattern[len("hf://") :]
    data_files = [f"hf://{f}" for f in fs.glob(stripped_pattern)]
  else:
    # Local files
    data_files = sorted(glob.glob(str(Path(data_file_pattern).expanduser().resolve())))
  if not data_files:
    if data_file_pattern.startswith("hf://") and "/**/*.parquet" in data_file_pattern:
      raise FileNotFoundError(
          f"No parquet files found for HuggingFace dataset pattern: {data_file_pattern}. "
          "Please make sure the dataset contains parquet files."
      )
    raise FileNotFoundError(f"No files found matching pattern: {data_file_pattern}")
  max_logging.log(f"Found {len(data_files)} files for train/eval with grain")
  return data_files


@dataclasses.dataclass(frozen=True)
class _FileInstruction:
  """Shard record range matching `ArrayRecordDataSource.FileInstruction`."""

  filename: str
  skip: int
  take: int
  examples_in_shard: int


def _parse_arrayrecord_patterns(data_file_pattern, mixture_config_path=None):
  """Parses ArrayRecord dataset names, file patterns, and mixture weights."""
  if mixture_config_path:
    with open(mixture_config_path, "r", encoding="utf-8") as f:
      mixture_config = json.load(f)
    names = list(mixture_config.keys())
    patterns = [component["path"] for component in mixture_config.values()]
    weights = [float(component["weight"]) for component in mixture_config.values()]
    return names, patterns, weights
  if data_file_pattern and ";" in data_file_pattern:
    entries = [entry.split(",") for entry in data_file_pattern.split(";")]
    if any(len(entry) != 2 for entry in entries):
      raise ValueError(f"Expected pattern and weight pairs separated by commas, got: {data_file_pattern}")
    patterns, raw_weights = zip(*entries, strict=True)
    return None, list(patterns), [float(weight) for weight in raw_weights]
  return None, [data_file_pattern] if data_file_pattern else [], None


_ARRAYRECORD_RECORD_COUNTS: dict[str, int] = {}


@functools.cache
def _find_arrayrecord_files(pattern: str, hf_access_token: str | None = None) -> tuple[str, ...]:
  """Lists ArrayRecord files matching `pattern` once per host process."""
  return tuple(sorted(find_data_files(pattern, hf_access_token=hf_access_token)))


def _count_arrayrecord_file(filename: str) -> int:
  """Counts records in a single ArrayRecord shard once per host process."""
  count = _ARRAYRECORD_RECORD_COUNTS.get(filename)
  if count is None:
    count = len(grain.ArrayRecordDataSource(filename))
    _ARRAYRECORD_RECORD_COUNTS[filename] = count
  return count


def _clear_arrayrecord_metadata_cache() -> None:
  """Clears process-local ArrayRecord file list and record count caches."""
  _find_arrayrecord_files.cache_clear()
  _ARRAYRECORD_RECORD_COUNTS.clear()


class _ColocatedArrayRecordDatasetLoader:
  """Resolves ArrayRecord `FileInstruction`s across colocated hosts."""

  def __init__(
      self,
      get_ds_fn,
      data_file_pattern: str | None,
      mixture_config_path: str | None = None,
      hf_access_token: str | None = None,
      max_workers: int = 1,
  ):
    self._get_ds_fn = get_ds_fn
    self._data_file_pattern = data_file_pattern
    self._mixture_config_path = mixture_config_path
    self._hf_access_token = hf_access_token
    self._max_workers = max_workers
    self._files_per_pattern: dict[str, tuple[str, ...]] | None = None
    self._unique_files: tuple[str, ...] = ()
    self._file_instructions: dict[str, tuple[_FileInstruction, ...]] | None = None

  def count_local_shards(self, shard_index: int, shard_count: int) -> tuple[int, int, int, list[int]]:
    """Lists pattern files locally and counts this host's shard slice."""
    _, patterns, _ = _parse_arrayrecord_patterns(self._data_file_pattern, self._mixture_config_path)
    unique_patterns = list(dict.fromkeys(patterns))
    if not unique_patterns:
      self._files_per_pattern = {}
      self._unique_files = ()
      return 0, 0, 0, []
    with futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
      self._files_per_pattern = dict(
          zip(
              unique_patterns,
              executor.map(
                  functools.partial(_find_arrayrecord_files, hf_access_token=self._hf_access_token),
                  unique_patterns,
              ),
          )
      )
      self._unique_files = tuple(sorted(set(itertools.chain.from_iterable(self._files_per_pattern.values()))))
      local_files = self._unique_files[shard_index::shard_count]
      local_counts = list(executor.map(_count_arrayrecord_file, local_files)) if local_files else []
    digest_payload = "\1".join(f"{pattern}\0" + "\0".join(files) for pattern, files in self._files_per_pattern.items())
    files_digest = zlib.crc32(digest_payload.encode("utf-8"))
    return len(self._unique_files), len(unique_patterns), files_digest, local_counts

  def set_shard_counts(self, counts: Sequence[int]) -> None:
    """Builds per-pattern `_FileInstruction`s from combined shard counts."""
    if self._files_per_pattern is None:
      raise RuntimeError("count_local_shards must be called before set_shard_counts.")
    if len(counts) != len(self._unique_files):
      raise ValueError(f"Expected {len(self._unique_files)} shard counts, got {len(counts)}.")
    if not self._files_per_pattern:
      self._file_instructions = None
      return
    _ARRAYRECORD_RECORD_COUNTS.update((filename, int(count)) for filename, count in zip(self._unique_files, counts))
    instruction_by_file = {
        filename: _FileInstruction(filename, 0, int(count), int(count))
        for filename, count in zip(self._unique_files, counts)
    }
    self._file_instructions = {
        pattern: tuple(instruction_by_file[filename] for filename in files)
        for pattern, files in self._files_per_pattern.items()
    }

  def __call__(self, **kwargs):
    """Invokes the wrapped dataset function with resolved file instructions."""
    return self._get_ds_fn(file_instructions=self._file_instructions, **kwargs)


def _apply_mapdataset_transforms(
    dataset,
    shuffle,
    shuffle_seed,
    num_epoch,
    dataloading_host_index,
    dataloading_host_count,
    grain_num_threads,
    grain_prefetch_buffer_size,
    elastic=False,
):
  """Apply standard shuffle, repeat, shard, and iter conversion transforms.

  When `elastic` is True, sharding and conversion to IterDataset are
  skipped so that the resulting MapDataset can be fed to `ElasticIterator`,
  which performs sharding and batching internally.
  """
  if shuffle:
    dataset = dataset.shuffle(seed=shuffle_seed)
  dataset = dataset.repeat(num_epoch)
  if elastic:
    return dataset
  dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
  dataset = dataset.to_iter_dataset(
      read_options=grain.ReadOptions(
          num_threads=grain_num_threads,
          prefetch_buffer_size=grain_prefetch_buffer_size,
      )
  )
  return dataset


def _build_dataset_config(config, num_samples=None, seed=1234, split_ratio=None, split_index=0, batch_size=None):
  """Build the configuration used by the Megatron-compatible data sources.

  `batch_size` is the replica's global batch per step; with several data replicas per process it sets the blend
  window (the step batch of all replicas) that keeps each replica's share of an mmap_npy blend mixed.
  """
  if config.grain_file_type not in ("mmap", "mmap_npy"):
    return None
  _, num_replicas = input_pipeline_utils.get_data_replica(config)
  return MMapDatasetConfig(
      max_target_length=config.max_target_length,
      eod_id=config.mmap_eod_id,
      mmap_split_sentences=config.mmap_split_sentences,
      blend_cache_dir=config.blend_cache_dir,
      blend_index_dir=config.blend_index_dir,
      num_samples=num_samples,
      seed=seed,
      split_ratio=split_ratio,
      split_index=split_index,
      blend_shard_window=batch_size * num_replicas if batch_size and num_replicas > 1 else None,
  )


def _make_mmap_multiprocessing_options(dataset, config, grain_worker_count, grain_per_worker_buffer_size):
  """Return Grain multiprocessing options without changing dataset ordering."""
  if grain_worker_count == -1:
    return grain.experimental.pick_performance_config(
        ds=dataset, ram_budget_mb=config.grain_ram_budget_mb, max_workers=None, max_buffer_size=None
    ).multiprocessing_options
  return grain.MultiprocessingOptions(num_workers=grain_worker_count, per_worker_buffer_size=grain_per_worker_buffer_size)


def get_datasets(
    data_file_pattern,
    data_file_type,
    shuffle,
    shuffle_seed,
    shuffle_buffer_size,
    num_epoch,
    dataloading_host_index,
    dataloading_host_count,
    grain_worker_count,
    grain_num_threads,
    grain_prefetch_buffer_size,
    grain_data_source_max_workers,
    mixture_config_path=None,
    elastic=False,
    hf_access_token=None,
    dataset_config=None,
    split="train",
    grain_index_storage_option=None,
    file_instructions=None,
):
  """Load a Grain dataset for the selected ``grain_file_type``."""
  if data_file_type == "arrayrecord":
    # Helper function to find files, create data source, and wrap in MapDataset
    def create_dataset_from_pattern(pattern):
      files = (
          file_instructions[pattern]
          if file_instructions is not None
          else find_data_files(pattern, hf_access_token=hf_access_token)
      )
      reader_options = (
          {"index_storage_option": grain_index_storage_option} if grain_index_storage_option is not None else None
      )
      source = grain.ArrayRecordDataSource(files, reader_options=reader_options)
      return grain.MapDataset.source(source)

    names, patterns, weights = _parse_arrayrecord_patterns(data_file_pattern, mixture_config_path)
    # Handle mixture config with named datasets, allows flexibility in recovering checkpoints
    if names is not None and weights is not None:
      executor = futures.ThreadPoolExecutor(max_workers=grain_data_source_max_workers)
      dataset_list = list(executor.map(create_dataset_from_pattern, patterns))
      executor.shutdown(wait=True)

      datasets_dict = dict(zip(names, dataset_list))

      for name, ds in datasets_dict.items():
        datasets_dict[name] = _apply_mapdataset_transforms(
            ds,
            shuffle,
            shuffle_seed,
            num_epoch,
            dataloading_host_index,
            dataloading_host_count,
            grain_num_threads,
            grain_prefetch_buffer_size,
        )

      # Normalize weights
      total_weight = sum(weights)
      weights_dict = {name: weight / total_weight for name, weight in zip(names, weights)}

      dataset = grain.IterDataset.mix(datasets_dict, weights_dict)
      return dataset
    elif weights is not None:
      weights = [round(weight / sum(weights), 4) for weight in weights]

      # Parallelize file finding (globbing), data source creation, and dataset wrapping
      # File finding and source creation are I/O-bound operations that release the GIL
      executor = futures.ThreadPoolExecutor(max_workers=grain_data_source_max_workers)
      dataset_list = list(executor.map(create_dataset_from_pattern, patterns))
      executor.shutdown(wait=True)

      # Apply shuffle, repeat, sharding, and conversion to IterDataset to each dataset before mixing
      for d, _ in enumerate(dataset_list):
        dataset_list[d] = _apply_mapdataset_transforms(
            dataset_list[d],
            shuffle,
            shuffle_seed,
            num_epoch,
            dataloading_host_index,
            dataloading_host_count,
            grain_num_threads,
            grain_prefetch_buffer_size,
        )
      # Use IterDataset.mix instead of MapDataset.mix in order to have per-mixture component checkpoints
      # for supporting changing the mixture after checkpointing
      dataset = grain.IterDataset.mix(dataset_list, weights)
      return dataset
    else:
      # Single pattern case - no need for parallelization
      dataset = create_dataset_from_pattern(patterns[0] if patterns else data_file_pattern)
      dataset = _apply_mapdataset_transforms(
          dataset,
          shuffle,
          shuffle_seed,
          num_epoch,
          dataloading_host_index,
          dataloading_host_count,
          grain_num_threads,
          grain_prefetch_buffer_size,
          elastic=elastic,
      )
      return dataset
  elif data_file_type in ("tfrecord", "parquet"):
    data_files = find_data_files(data_file_pattern, hf_access_token=hf_access_token)
    file_slice, files_per_host, row_shard = input_pipeline_utils.compute_file_sharding(
        len(data_files), dataloading_host_index, dataloading_host_count
    )
    if len(data_files) < dataloading_host_count:
      max_logging.warning(
          f"Data shard count ({len(data_files)}) < host count ({dataloading_host_count}). "
          f"Each shard will be read by at most {math.ceil(dataloading_host_count / len(data_files))} hosts. "
          f"Concurrent reading by multiple hosts may cause slow down."
      )
    if grain_worker_count > files_per_host:
      raise ValueError(
          f"grain_worker_count ({grain_worker_count}) exceeds the number of {data_file_type} files "
          f"per host ({files_per_host}). "
          f"Lower grain_worker_count to at most {files_per_host}."
      )
    dataset = grain.MapDataset.source(data_files)
    if shuffle:
      dataset = dataset.shuffle(seed=shuffle_seed)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset[file_slice]
    if data_file_type == "tfrecord":
      dataset = dataset.map(input_pipeline_utils.make_tfrecord_iter_dataset)  # pyrefly: ignore[missing-attribute]
    else:
      dataset = dataset.map(  # pyrefly: ignore[missing-attribute]
          functools.partial(
              input_pipeline_utils.make_parquet_iter_dataset,
              hf_access_token=hf_access_token,
          )
      )
    cycle_length = min(files_per_host, grain_num_threads)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=cycle_length)
    if row_shard is not None:
      dataset = input_pipeline_utils.IndexShardIterDataset(dataset, host_index=row_shard[0], host_count=row_shard[1])
    if shuffle:
      dataset = grain.experimental.WindowShuffleIterDataset(dataset, window_size=shuffle_buffer_size, seed=shuffle_seed)
    return dataset
  elif data_file_type == "mmap":
    if elastic:
      raise ValueError("grain_use_elastic_iterator is not supported for mmap datasets.")
    return get_mmap_dataset(
        data_file_pattern,
        dataset_config.mmap_split_sentences,
        dataset_config.max_target_length,
        dataset_config.eod_id,
        shuffle,
        shuffle_seed,
        num_epoch,
        dataloading_host_index,
        dataloading_host_count,
        grain_num_threads,
        grain_prefetch_buffer_size,
        apply_transforms=_apply_mapdataset_transforms,
    )
  elif data_file_type == "mmap_npy":
    if elastic:
      raise ValueError("grain_use_elastic_iterator is not supported for mmap_npy datasets.")
    return get_mmap_npy_dataset(
        data_file_pattern,
        dataset_config.mmap_split_sentences,
        dataset_config.max_target_length,
        dataset_config.eod_id,
        num_epoch,
        dataloading_host_index,
        dataloading_host_count,
        grain_num_threads,
        grain_prefetch_buffer_size,
        dataset_config.blend_cache_dir or None,
        dataset_config.blend_index_dir or None,
        split,
        apply_transforms=_apply_mapdataset_transforms,
        num_samples=dataset_config.num_samples,
        seed=dataset_config.seed,
        split=dataset_config.split_ratio,
        split_index=dataset_config.split_index,
        blend_shard_window=dataset_config.blend_shard_window,
    )
  else:
    raise ValueError(
        f"grain pipeline supports (arrayrecord, tfrecord, parquet, mmap, mmap_npy) as grain_file_type, "
        f"but got {data_file_type}"
    )


def _mmap_pretrain_pipeline(
    dataset,
    config,
    text_column,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Pretrain pipeline for Megatron-compatible mmap / mmap_npy pre-tokenized formats."""
  # Match Megatron's EOD, position, and loss semantics.
  eod_id = config.mmap_eod_id
  is_npy = config.grain_file_type == "mmap_npy"

  # Split or rekey
  if is_npy:
    # MegatronNpyDataSource returns seq_length+1 tokens; split into
    # inputs[:-1] / targets[1:] with EOD-aware segmentation.
    dataset = dataset.map(
        input_pipeline_utils.MegatronSplitInputsTargets(
            eod_id=eod_id,
            reset_attention_mask=config.reset_attention_mask,
            eod_mask_loss=config.eod_mask_loss,
            min_segment_length=input_pipeline_utils.megatron_min_segment_length(config),
        )
    )
  else:
    data_columns = ("inputs", "targets")
    rekey_dict = {col: text_column for col in data_columns}
    dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict))
    # Samples are already exactly max_target_length with EOD tokens between
    # documents.  Generate doc-boundary-aware segmentation, skip packing.
    dataset = dataset.map(
        input_pipeline_utils.GenerateDocSegmentIds(
            eod_id=eod_id,
            reset_attention_mask=config.reset_attention_mask,
            eod_mask_loss=config.eod_mask_loss,
            min_segment_length=input_pipeline_utils.megatron_min_segment_length(config),
        )
    )

  batch_size = data_processing_utils.get_local_batch_size(config)

  mp_options = _make_mmap_multiprocessing_options(dataset, config, grain_worker_count, grain_per_worker_buffer_size)

  # mmap_npy: mp_prefetch BEFORE batch to preserve sample ordering in
  # blend-then-shard.  Grain's mp_prefetch does per-worker sharding; placing
  # it before batch ensures each worker processes a contiguous chunk and the
  # round-robin interleaver reconstructs global order.
  if is_npy:
    dataset = dataset.mp_prefetch(mp_options)

  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=eod_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  # mmap: shift needed (Rekey produced identical inputs/targets);
  # mmap_npy: skip (MegatronSplitInputsTargets already split).
  if not is_npy:
    if not config.eod_mask_loss:
      max_logging.log(
          "WARNING: mmap mode with eod_mask_loss=False uses mmap_eod_id as both "
          "padding and EOD sentinel. ShiftData will zero targets_segmentation "
          "at all EOD positions, effectively masking EOD from loss regardless "
          "of eod_mask_loss. Use mmap_npy mode for correct eod_mask_loss=False behavior."
      )
    dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[eod_id], axis=1))
    dataset = dataset.mp_prefetch(mp_options)

  return dataset


def pretrain_preprocessing_pipeline(
    dataset,
    config,
    data_columns,
    tokenize,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Use grain pipeline to pre-process the dataset and return iterators for pretrain"""
  if config.grain_file_type in ("mmap", "mmap_npy"):
    assert len(data_columns) == 1, (
        f"grain_file_type={config.grain_file_type!r} requires exactly one pre-tokenized "
        f"text column, got {data_columns}"
    )
    return _mmap_pretrain_pipeline(dataset, config, data_columns[0], grain_worker_count, grain_per_worker_buffer_size)

  is_offline = getattr(config, "is_offline_distillation", False)

  columns_to_parse = list(data_columns)
  if is_offline:
    columns_to_parse.extend(["top_k_logits", "top_k_indices"])

  dataset = data_processing_utils.parse_and_keep_features(dataset, config, columns_to_parse, tokenize)

  assert len(data_columns) == 1
  text_column = data_columns[0]

  tokenizer_model, pad_id = data_processing_utils.get_tokenizer_and_pad_id(config)

  if tokenize:
    if config.use_truncation:
      dataset = dataset.map(grain_tokenizer.TokenizeAndTrim(text_column, config.max_target_length, tokenizer_model))
    else:
      dataset = dataset.apply(grain_tokenizer.TokenizeAndChunk(text_column, config.max_target_length, tokenizer_model))

  pipeline_columns = ["inputs", "targets"]
  rekey_dict = {col: text_column for col in pipeline_columns}

  dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict, keep_old_keys=is_offline))

  if is_offline:
    pipeline_columns.extend(["top_k_logits", "top_k_indices"])

  batch_size = data_processing_utils.get_local_batch_size(config)
  dataset = data_processing_utils.format_and_batch(
      dataset, config, batch_size, pad_id, tuple(pipeline_columns), tokenizer_model
  )

  dataset = data_processing_utils.apply_multiprocessing_and_prefetch(
      dataset, config, grain_worker_count, grain_per_worker_buffer_size
  )
  return dataset


def dpo_preprocessing_pipeline(
    dataset,
    config,
    data_columns,
    tokenize,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Use grain to pre-process the dataset and return iterators for dpo fine-tuning"""
  dataset = data_processing_utils.parse_and_keep_features(dataset, config, data_columns, tokenize)
  tokenizer_model, pad_id = data_processing_utils.get_tokenizer_and_pad_id(config)

  if tokenize:
    dataset = dataset.map(grain_tokenizer.TokenizeAndTrim(data_columns, config.max_target_length, tokenizer_model))

  # Renames arbitrary DPO columns and performs DPO-aware padding.
  dataset = dataset.map(
      dpo_utils.DPODataFormatting(
          pad_id=pad_id,
          max_target_length=config.max_target_length,
          data_column_names=data_columns,
          max_prompt_length=config.dpo.max_prompt_length,
      )
  )

  batch_size = data_processing_utils.get_local_batch_size(config)
  if config.grain_use_elastic_iterator:
    # ElasticIterator batches internally, so return the pre-batch dataset.
    pass
  else:
    batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
    dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  dataset = data_processing_utils.apply_multiprocessing_and_prefetch(
      dataset, config, grain_worker_count, grain_per_worker_buffer_size
  )
  return dataset


def _format_chat_template_grain(element, data_columns, tokenizer_model, model_name=None):
  """Grain-compatible mapping function to format raw columns into conversational messages."""
  # Convert raw columns to conversational messages
  if "messages" in data_columns:
    messages = element["messages"]
  elif set(data_columns) == {"prompt", "completion"}:
    messages = [
        {"role": "user", "content": element["prompt"]},
        {"role": "assistant", "content": element["completion"]},
    ]
  elif set(data_columns) == {"question", "answer"}:
    messages = [
        {"role": "user", "content": element["question"]},
        {"role": "assistant", "content": element["answer"]},
    ]
  else:
    # Fallback if it's already a single string
    messages = element[data_columns[0]]

  assert all(
      hasattr(m, "__contains__")
      and "role" in m
      and any(k in m for k in ("content", "tool_calls", "reasoning_content", "content_blocks"))
      for m in messages
  ), ("SFT requires a conversational format. Expected dicts with 'role' and content/tool_calls, " f"but got: {messages}")

  # Assign the standardized messages back to the primary column
  element[data_columns[0]] = messages

  return input_pipeline_utils.apply_chat_template(
      element, tokenizer_model=tokenizer_model, data_column_name=data_columns[0], model_name=model_name
  )


def _tokenize_sft_chunks(element, text_column_name, tokenizer_model):
  """Tokenize each chunk individually without truncating."""
  text_chunks = element[text_column_name]
  element[text_column_name] = [tokenizer_model.encode(chunk) for chunk in text_chunks]
  return element


def sft_preprocessing_pipeline(
    dataset,
    config,
    data_columns,
    tokenize,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Use grain pipeline to pre-process the dataset and return iterators for sft fine-tuning"""
  dataset = data_processing_utils.parse_and_keep_features(dataset, config, data_columns, tokenize)

  # Chunks are tokenized one by one and the chat template already emits BOS/EOS, so the tokenizer
  # must not add them to every chunk (same as the HF SFT path and vision SFT below).
  tokenizer_model, pad_id = data_processing_utils.get_tokenizer_and_pad_id(config, add_bos=False, add_eos=False)
  base_tokenizer_model = tokenizer_model

  tokenizer_model = getattr(tokenizer_model, "tokenizer", tokenizer_model)
  if input_pipeline_utils.fix_pad_token_if_needed(tokenizer_model):
    pad_id = tokenizer_model.pad_token_id
    if hasattr(base_tokenizer_model, "pad_id"):
      base_tokenizer_model.pad_id = pad_id

  data_processing_utils.validate_and_configure_sft_columns(
      data_columns, tokenizer_model, getattr(config, "chat_template", None)
  )

  dataset = dataset.map(
      functools.partial(
          _format_chat_template_grain,
          data_columns=data_columns,
          tokenizer_model=tokenizer_model,
          model_name=getattr(config, "model_name", None),
      )
  )

  if tokenize:
    dataset = dataset.map(
        functools.partial(
            _tokenize_sft_chunks,
            text_column_name=data_columns[0],
            tokenizer_model=tokenizer_model,
        )
    )

  dataset = dataset.map(
      input_pipeline_utils.SFTPromptMasking(
          text_column_name=data_columns[0],
          completion_only=config.sft_train_on_completion_only,
          max_target_length=config.max_target_length,
          unk_id=pad_id,
      )
  )
  data_columns = ("inputs", "targets")

  batch_size = data_processing_utils.get_local_batch_size(config)
  dataset = data_processing_utils.format_and_batch(
      dataset, config, batch_size, pad_id, data_columns, base_tokenizer_model
  )
  dataset = data_processing_utils.apply_multiprocessing_and_prefetch(
      dataset, config, grain_worker_count, grain_per_worker_buffer_size
  )
  return dataset


def vision_sft_preprocessing_pipeline(
    dataset,
    config,
    data_columns,
    image_column=None,
    tokenize: bool = True,
    grain_worker_count: int = 0,
    grain_per_worker_buffer_size: int = 1,
):
  """Use grain pipeline to pre-process dataset and return iterators for multimodal SFT fine-tuning."""
  if config.grain_use_elastic_iterator:
    raise ValueError(
        "ElasticIterator is not supported yet for multimodal SFT because post-batch "
        "transformations (like folding images) cannot be easily applied to the iterator."
    )
  assert len(data_columns) == 2, f"Need two data_columns for query and response, received {data_columns=}"
  text_columns = list(data_columns)

  if image_column is None:
    image_column = getattr(config, "train_image_column", "image")

  if isinstance(image_column, (list, tuple)):
    columns_to_parse = text_columns + list(image_column)
  else:
    columns_to_parse = text_columns + [image_column]

  dataset = data_processing_utils.parse_and_keep_features(dataset, config, columns_to_parse, tokenize=tokenize)

  # If multiple image columns are provided, merge them into a single 'images' column.
  if isinstance(image_column, (list, tuple)):
    dataset = dataset.map(
        functools.partial(
            input_pipeline_utils.merge_image_columns,
            image_columns=list(image_column),
            max_num_images_per_example=config.max_num_images_per_example,
        )
    )
    image_column = "images"
  elif image_column != "images":
    dataset = dataset.map(input_pipeline_utils.Rekey({"images": image_column}))

  dataset = dataset.map(
      functools.partial(
          input_pipeline_utils.reformat_prompt,
          column=text_columns[0],
          image_placeholder=config.image_placeholder,
          model_name=config.model_name,
      )
  )
  dataset = dataset.map(
      functools.partial(
          input_pipeline_utils.reformat_response,
          column=text_columns[1],
          model_name=config.model_name,
      )
  )

  dataset = dataset.map(
      functools.partial(
          input_pipeline_utils.pre_process_image_sft,
          image_column="images",
          config=config,
      )
  )

  # In multimodal SFT, prompt and response reformatting steps already handle special tokens (BOS/EOS).
  # Therefore, we explicitly disable add_bos and add_eos in the tokenizer to avoid duplicate tokens.
  tokenizer_model, pad_id = data_processing_utils.get_tokenizer_and_pad_id(config, add_bos=False, add_eos=False)

  if tokenize:
    dataset = dataset.map(grain_tokenizer.TokenizeAndTrim(text_columns, config.max_target_length, tokenizer_model))

  dataset = dataset.map(
      functools.partial(
          input_pipeline_utils.prepare_text_for_image_fusion,
          column_name=text_columns[0],
          config=config,
      )
  )

  dataset = dataset.map(
      input_pipeline_utils.SFTPromptMaskingVision(
          query_column=text_columns[0],
          response_column=text_columns[1],
          max_target_length=config.max_target_length,
          pad_id=pad_id,
      )
  )

  dataset = dataset.map(
      input_pipeline_utils.PadOrTrimToMaxLength(
          config.max_target_length,
          pad_id,
          config=config,
          max_num_images_per_example=config.max_num_images_per_example,
      )
  )

  dataset = dataset.map(input_pipeline_utils.ExtractImagesAndMasks())

  batch_size = data_processing_utils.get_local_batch_size(config)
  if config.use_tunix_gradient_accumulation:
    batch_size = batch_size // config.gradient_accumulation_steps

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.map(input_pipeline_utils.FoldImagesIntoBatch(model_name=config.model_name))
  if config.use_mrope:
    dataset = dataset.map(
        input_pipeline_utils.ComputeQwen3OmniPositions(
            data_column="inputs",
            spatial_merge_size=config.spatial_merge_size_for_vit,
            position_id_per_seconds=config.position_id_per_seconds,
            use_audio_in_video=getattr(config, "use_audio_in_video", False),
            config=config,
            keep_aux_fields=False,  # drop aux to match train shape
        )
    )

  if hasattr(dataset, "to_iter_dataset"):
    dataset = dataset.to_iter_dataset()

  dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[pad_id], axis=1))
  dataset = data_processing_utils.apply_multiprocessing_and_prefetch(
      dataset, config, grain_worker_count, grain_per_worker_buffer_size
  )
  return dataset


def _get_pipeline_fn(config):
  """Returns the appropriate preprocessing pipeline function based on config."""
  if config.use_sft and config.use_multimodal:
    return vision_sft_preprocessing_pipeline
  if config.use_dpo:
    return dpo_preprocessing_pipeline
  if config.use_sft:
    return sft_preprocessing_pipeline
  return pretrain_preprocessing_pipeline


class _PaddedBatchMapDataset(grain.MapDataset):
  """Pads each `logical_batch_size` step with zero-filled elements up to `physical_batch_size`."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(self, parent: grain.MapDataset, logical_batch_size: int, physical_batch_size: int):
    super().__init__(parent)
    self._logical_batch_size = logical_batch_size
    self._physical_batch_size = physical_batch_size
    self.offset = 0
    self._pad = None

  def __len__(self) -> int:
    parent_steps = max(0, (len(self._parent) - self.offset) // self._logical_batch_size)
    return min(parent_steps, sys.maxsize // self._physical_batch_size) * self._physical_batch_size

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    step, rem = divmod(index, self._physical_batch_size)
    if rem < self._logical_batch_size:
      return self._parent[self.offset + step * self._logical_batch_size + rem]
    if self._pad is None:
      self._pad = jax.tree.map(np.zeros_like, self._parent[0])
    return jax.tree.map(np.copy, self._pad)


class _PaddedElasticIterator(ElasticIterator):
  """`ElasticIterator` that pads `logical_batch_size` steps up to `physical_batch_size`."""

  def __init__(self, ds: grain.MapDataset, *, logical_batch_size: int, physical_batch_size: int, **kwargs):
    self._logical_batch_size = logical_batch_size
    self._padded_ds = _PaddedBatchMapDataset(ds, logical_batch_size, physical_batch_size)
    super().__init__(self._padded_ds, global_batch_size=physical_batch_size, **kwargs)

  def __iter__(self):  # pylint: disable=non-iterator-returned
    # OSS Grain 0.2.x returns `_default_iterator` from `__iter__`, which would bypass `get_state`/`set_state`.
    return self

  def close(self) -> None:
    for attr in ("base_iterator", "_default_iterator"):
      if (it := self.__dict__.pop(attr, None)) is not None:
        it.close()

  def get_state(self) -> dict[str, int]:
    steps = super().get_state()["global_next_index"] // self._global_batch_size
    return {"global_next_index": self._padded_ds.offset + steps * self._logical_batch_size}

  def set_state(self, state: dict[str, int]) -> None:
    steps, self._padded_ds.offset = divmod(state["global_next_index"], self._logical_batch_size)
    super().set_state({"global_next_index": steps * self._global_batch_size})


def _make_elastic_iterator(dataset, config, preprocessing_fn, shard_index=None, shard_count=None, mp_opts=None):
  """Applies `preprocessing_fn` and wraps the result in `ElasticIterator` (padding to physical batch if needed).

  ElasticIterator gives each shard `global_batch_size // shard_count` rows. The shards of `get_dataloading_shard`
  cover every data replica, so its global batch is that of all replicas.
  """
  ds = preprocessing_fn(dataset=dataset)
  physical_batch_size = input_pipeline_utils.get_all_replicas_batch_size(config, config.global_batch_size_to_load)
  logical_batch_size = (
      input_pipeline_utils.get_all_replicas_batch_size(config, config.logical_global_batch_size_to_load)
      if config.logical_global_batch_size_to_load
      else physical_batch_size
  )
  common_kwargs = {
      "shard_options": grain.ShardOptions(
          shard_index=shard_index if shard_index is not None else jax.process_index(),
          shard_count=shard_count if shard_count is not None else jax.process_count(),
      ),
      "read_options": grain.ReadOptions(
          num_threads=config.grain_num_threads,
          prefetch_buffer_size=config.grain_prefetch_buffer_size,
      ),
      "multiprocessing_options": mp_opts,
  }
  if logical_batch_size == physical_batch_size:
    return ElasticIterator(ds, global_batch_size=physical_batch_size, **common_kwargs)
  return _PaddedElasticIterator(
      ds, logical_batch_size=logical_batch_size, physical_batch_size=physical_batch_size, **common_kwargs
  )


def make_grain_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  assert (
      config.global_batch_size_to_load % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."

  pipeline_fn = _get_pipeline_fn(config)
  # The mmap_npy index is sharded among every data replica, so it holds the samples of all of them.
  mmap_npy_num_samples = (
      config.steps * input_pipeline_utils.get_all_replicas_batch_size(config, config.global_batch_size_to_load)
      if config.grain_file_type == "mmap_npy" and getattr(config, "steps", 0) > 0
      else None
  )
  dataset_config = _build_dataset_config(
      config,
      num_samples=mmap_npy_num_samples,
      seed=config.data_shuffle_seed,
      split_ratio=config.mmap_npy_split or None,
      split_index=0,
      batch_size=config.global_batch_size_to_load,
  )

  grain_train_files = config.grain_train_files
  if (
      not grain_train_files
      and not config.grain_train_mixture_config_path
      and config.grain_file_type == "parquet"
      and config.hf_path
  ):
    grain_train_files = construct_hf_dataset_path(config.hf_path, split="train")
  elif (
      not grain_train_files
      and not config.grain_train_mixture_config_path
      and config.grain_file_type == "tfrecord"
      and config.dataset_path
      and config.dataset_name
      and config.train_split
  ):
    grain_train_files = construct_tfds_tfrecord_path(config.dataset_path, config.dataset_name, config.train_split)

  get_ds_fn = functools.partial(
      get_datasets,
      grain_train_files,
      config.grain_file_type,
      shuffle=config.enable_data_shuffling,
      shuffle_seed=config.data_shuffle_seed,
      shuffle_buffer_size=config.grain_shuffle_buffer_size,
      num_epoch=config.num_epoch,
      grain_worker_count=config.grain_worker_count,
      grain_num_threads=config.grain_num_threads,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
      grain_index_storage_option=getattr(config, "grain_index_storage_option", None),
      mixture_config_path=config.grain_train_mixture_config_path,
      elastic=config.grain_use_elastic_iterator,
      hf_access_token=config.hf_access_token,
      dataset_config=dataset_config,
      split="train",
  )

  preprocessing_fn_kwargs = {
      "config": config,
      "data_columns": config.train_data_columns,
      "tokenize": config.tokenize_train_data,
      "grain_worker_count": config.grain_worker_count,
      "grain_per_worker_buffer_size": config.grain_per_worker_buffer_size,
  }
  if config.use_sft and config.use_multimodal:
    preprocessing_fn_kwargs["image_column"] = config.train_image_column

  preprocessing_fn = functools.partial(
      pipeline_fn,
      **preprocessing_fn_kwargs,
  )

  # In the case of using colocated python for data input, partial functions such as
  # get_ds_fn (data initialization) and preprocessing_fn (data transformation)
  # are passed to the RemoteIteratorWrapper, which will then be passed to RemoteIterator
  # that runs in the colocated python environment.
  # While in other cases, get_ds_fn and preprocessing_fn to produce a data iterator and
  # pass to MultiHostDataLoadIterator
  if config.colocated_python_data_input:
    count_shards = config.grain_file_type == "arrayrecord"
    if count_shards:
      get_ds_fn = _ColocatedArrayRecordDatasetLoader(
          get_ds_fn,
          grain_train_files,
          mixture_config_path=config.grain_train_mixture_config_path,
          hf_access_token=config.hf_access_token,
          max_workers=config.grain_data_source_max_workers,
      )
    if config.grain_use_elastic_iterator:
      preprocessing_fn = functools.partial(_make_elastic_iterator, config=config, preprocessing_fn=preprocessing_fn)

    global_shape = (config.global_batch_size_to_load, config.max_target_length)
    return multihost_dataloading.RemoteIteratorWrapper(
        get_ds_fn,
        preprocessing_fn,
        global_mesh,
        global_shape,
        checkpoint_path=config.checkpoint_dir,
        elastic=config.grain_use_elastic_iterator,
        count_shards=count_shards,
    )

  if 0 < config.expansion_factor_real_data < 1:
    num_dataloader_to_restore = int(1 / config.expansion_factor_real_data)
    train_dataloader_list = []
    for i in range(num_dataloader_to_restore):
      dataloading_host_index, dataloading_host_count = input_pipeline_utils.get_dataloading_shard(
          config, process_indices, shard_offset=i, shards_per_host=num_dataloader_to_restore
      )
      train_ds = get_ds_fn(
          dataloading_host_index=dataloading_host_index,
          dataloading_host_count=dataloading_host_count,
      )
      train_dataloader = preprocessing_fn(dataset=train_ds)
      train_dataloader_list.append(train_dataloader)
    return [
        multihost_dataloading.MultiHostDataLoadIterator(x, global_mesh, config.generate_padding_batch_train)
        for x in train_dataloader_list
    ]

  # Default non-colocated, expansion_factor_real_data >=1 path
  shard_index, shard_count = input_pipeline_utils.get_dataloading_shard(config, process_indices)
  train_ds = get_ds_fn(
      dataloading_host_index=shard_index,
      dataloading_host_count=shard_count,
  )
  if config.grain_use_elastic_iterator:
    mp_options = (
        grain.MultiprocessingOptions(
            num_workers=config.grain_worker_count,
            per_worker_buffer_size=config.grain_per_worker_buffer_size,
        )
        if config.grain_worker_count > 0
        else None
    )
    train_dataloader = _make_elastic_iterator(
        train_ds,
        config,
        preprocessing_fn,
        shard_index=shard_index,
        shard_count=shard_count,
        mp_opts=mp_options,
    )
  else:
    train_dataloader = preprocessing_fn(dataset=train_ds)

  return multihost_dataloading.MultiHostDataLoadIterator(
      train_dataloader,
      global_mesh,
      config.generate_padding_batch_train,
      expansion_loading_factor_for_grain=config.expansion_factor_real_data,
  )


def make_grain_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  assert (
      config.global_batch_size_to_load_eval % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."

  pipeline_fn = _get_pipeline_fn(config)
  mmap_npy_eval_num_samples = None
  if config.grain_file_type == "mmap_npy" and hasattr(config, "eval_steps") and config.eval_steps > 0:
    eval_interval = getattr(config, "eval_interval", 0)
    if eval_interval > 0 and hasattr(config, "steps") and config.steps > 0:
      eval_rounds = -(-config.steps // eval_interval)
    else:
      eval_rounds = 1
    mmap_npy_eval_num_samples = (
        eval_rounds
        * config.eval_steps
        * input_pipeline_utils.get_all_replicas_batch_size(config, config.global_batch_size_to_load_eval)
    )

  dataset_config = _build_dataset_config(
      config,
      num_samples=mmap_npy_eval_num_samples,
      seed=config.data_shuffle_seed,
      split_ratio=config.mmap_npy_split or None,
      split_index=1 if config.mmap_npy_split else 0,
      batch_size=config.global_batch_size_to_load_eval,
  )

  grain_eval_files = config.grain_eval_files
  if not grain_eval_files and config.grain_file_type == "parquet" and getattr(config, "hf_path", None):
    split = getattr(config, "hf_eval_split", None) or "validation"
    grain_eval_files = construct_hf_dataset_path(config.hf_path, split=split)
  elif (
      not grain_eval_files
      and config.grain_file_type == "tfrecord"
      and config.dataset_path
      and config.eval_dataset_name
      and config.eval_split
  ):
    grain_eval_files = construct_tfds_tfrecord_path(config.dataset_path, config.eval_dataset_name, config.eval_split)

  get_ds_fn = functools.partial(
      get_datasets,
      grain_eval_files,
      config.grain_file_type,
      shuffle=False,  # No shuffle for eval
      shuffle_seed=config.data_shuffle_seed,
      shuffle_buffer_size=config.grain_shuffle_buffer_size,
      num_epoch=1,
      grain_worker_count=config.grain_worker_count_eval,
      grain_num_threads=config.grain_num_threads_eval,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size_eval,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
      grain_index_storage_option=getattr(config, "grain_index_storage_option", None),
      hf_access_token=config.hf_access_token,
      dataset_config=dataset_config,
      split="eval",
  )

  eval_preprocessing_fn_kwargs = {
      "config": config,
      "data_columns": config.eval_data_columns,
      "tokenize": config.tokenize_eval_data,
      "grain_worker_count": config.grain_worker_count_eval,
      "grain_per_worker_buffer_size": config.grain_per_worker_buffer_size_eval,
  }
  if config.use_sft and config.use_multimodal:
    eval_preprocessing_fn_kwargs["image_column"] = config.eval_image_column

  preprocessing_fn = functools.partial(
      pipeline_fn,
      **eval_preprocessing_fn_kwargs,
  )

  if not config.colocated_python_data_input:
    dataloading_host_index, dataloading_host_count = input_pipeline_utils.get_dataloading_shard(config, process_indices)
    eval_ds = get_ds_fn(
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
    )
    eval_dataloader = preprocessing_fn(dataset=eval_ds)
    return multihost_dataloading.MultiHostDataLoadIterator(
        eval_dataloader, global_mesh, config.generate_padding_batch_eval
    )
  else:
    count_shards = config.grain_file_type == "arrayrecord"
    if count_shards:
      get_ds_fn = _ColocatedArrayRecordDatasetLoader(
          get_ds_fn,
          grain_eval_files,
          hf_access_token=config.hf_access_token,
          max_workers=config.grain_data_source_max_workers,
      )
    global_shape = (config.global_batch_size_to_load_eval, config.max_target_length)
    return multihost_dataloading.RemoteIteratorWrapper(
        get_ds_fn,
        preprocessing_fn,
        global_mesh,
        global_shape,
        checkpoint_path=config.checkpoint_dir,
        # The eval pipeline is not wrapped in an ElasticIterator.
        elastic=False,
        count_shards=count_shards,
    )
