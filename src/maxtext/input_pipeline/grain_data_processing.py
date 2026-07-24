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

import math
import glob
from pathlib import Path
import functools
import ml_collections
from concurrent import futures
import json

import jax

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


def find_data_files(data_file_pattern):
  """Find data files matching the pattern."""
  if data_file_pattern.startswith("gs://"):
    data_files = gcs_utils.gcs_glob_pattern(data_file_pattern)
  else:
    # Local files
    data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  if not data_files:
    raise FileNotFoundError(f"No files found matching pattern: {data_file_pattern}")
  max_logging.log(f"Found {len(data_files)} files for train/eval with grain")
  return data_files


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


def _build_dataset_config(config, num_samples=None, seed=1234, split_ratio=None, split_index=0):
  """Build the configuration used by the Megatron-compatible data sources."""
  if config.grain_file_type not in ("mmap", "mmap_npy"):
    return None
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
    dataset_config=None,
    split="train",
):
  """Load a Grain dataset for the selected ``grain_file_type``."""
  if data_file_type == "arrayrecord":
    # Helper function to find files, create data source, and wrap in MapDataset
    def create_dataset_from_pattern(pattern):
      files = find_data_files(pattern)
      source = grain.ArrayRecordDataSource(files)
      return grain.MapDataset.source(source)

    # Handle mixture config with named datasets, allows flexibility in recovering checkpoints
    if mixture_config_path:
      with open(mixture_config_path, "r", encoding="utf-8") as f:
        mixture_config = json.load(f)

      paths = [config["path"] for config in mixture_config.values()]
      weights = [float(config["weight"]) for config in mixture_config.values()]

      executor = futures.ThreadPoolExecutor(max_workers=grain_data_source_max_workers)
      dataset_list = list(executor.map(create_dataset_from_pattern, paths))
      executor.shutdown(wait=True)

      datasets_dict = dict(zip(mixture_config.keys(), dataset_list))

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
      weights_dict = {name: weight / total_weight for name, weight in zip(mixture_config.keys(), weights)}

      dataset = grain.IterDataset.mix(datasets_dict, weights_dict)
      return dataset
    elif ";" in data_file_pattern:
      data_file_patterns, weights = zip(*[pattern.split(",") for pattern in data_file_pattern.split(";")])
      assert len(data_file_patterns) == len(weights), "Number of data file patterns and weights must match"
      weights = [float(weight) for weight in weights]
      weights = [round(weight / sum(weights), 4) for weight in weights]

      # Parallelize file finding (globbing), data source creation, and dataset wrapping
      # File finding and source creation are I/O-bound operations that release the GIL
      executor = futures.ThreadPoolExecutor(max_workers=grain_data_source_max_workers)
      dataset_list = list(executor.map(create_dataset_from_pattern, data_file_patterns))
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
      dataset = create_dataset_from_pattern(data_file_pattern)
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
    data_files = find_data_files(data_file_pattern)
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
      dataset = dataset.map(grain.experimental.ParquetIterDataset)  # pyrefly: ignore[missing-attribute]
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


def _format_chat_template_grain(element, data_columns, tokenizer_model):
  """Grain-compatible mapping function to format raw columns into conversational messages."""
  # Convert raw columns to conversational messages
  if "messages" in data_columns:
    messages = element["messages"]
  elif set(data_columns) == {"prompt", "completion"}:
    messages = [{"role": "user", "content": element["prompt"]}, {"role": "assistant", "content": element["completion"]}]
  elif set(data_columns) == {"question", "answer"}:
    messages = [{"role": "user", "content": element["question"]}, {"role": "assistant", "content": element["answer"]}]
  else:
    # Fallback if it's already a single string
    messages = element[data_columns[0]]

  assert all(
      hasattr(m, "__contains__") and "role" in m and "content" in m for m in messages
  ), f"SFT requires a conversational format. Expected dicts with 'role' and 'content', but got: {messages}"

  # Assign the standardized messages back to the primary column
  element[data_columns[0]] = messages

  return input_pipeline_utils.apply_chat_template(
      element, tokenizer_model=tokenizer_model, data_column_name=data_columns[0]
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

  tokenizer_model, pad_id = data_processing_utils.get_tokenizer_and_pad_id(config)
  base_tokenizer_model = tokenizer_model

  tokenizer_model = getattr(tokenizer_model, "tokenizer", tokenizer_model)

  data_processing_utils.validate_and_configure_sft_columns(
      data_columns, tokenizer_model, getattr(config, "chat_template", None)
  )

  dataset = dataset.map(
      functools.partial(_format_chat_template_grain, data_columns=data_columns, tokenizer_model=tokenizer_model)
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


def _get_pipeline_fn(config):
  """Returns the appropriate preprocessing pipeline function based on config."""
  if config.use_dpo:
    return dpo_preprocessing_pipeline
  if config.use_sft:
    return sft_preprocessing_pipeline
  return pretrain_preprocessing_pipeline


def _make_elastic_iterator(dataset, config, preprocessing_fn, shard_index=None, shard_count=None, mp_opts=None):
  """Applies preprocessing_fn then wraps the result with ElasticIterator.

  When shard_index/shard_count are None, defaults to jax.process_index()/jax.process_count().
  """
  ds = preprocessing_fn(dataset=dataset)
  return ElasticIterator(
      ds,
      global_batch_size=config.global_batch_size_to_load,
      shard_options=grain.ShardOptions(
          shard_index=shard_index if shard_index is not None else jax.process_index(),
          shard_count=shard_count if shard_count is not None else jax.process_count(),
      ),
      read_options=grain.ReadOptions(
          num_threads=config.grain_num_threads,
          prefetch_buffer_size=config.grain_prefetch_buffer_size,
      ),
      multiprocessing_options=mp_opts,
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
  mmap_npy_num_samples = (
      config.steps * config.global_batch_size_to_load
      if config.grain_file_type == "mmap_npy" and getattr(config, "steps", 0) > 0
      else None
  )
  dataset_config = _build_dataset_config(
      config,
      num_samples=mmap_npy_num_samples,
      seed=config.data_shuffle_seed,
      split_ratio=config.mmap_npy_split or None,
      split_index=0,
  )

  get_ds_fn = functools.partial(
      get_datasets,
      config.grain_train_files,
      config.grain_file_type,
      shuffle=config.enable_data_shuffling,
      shuffle_seed=config.data_shuffle_seed,
      shuffle_buffer_size=config.grain_shuffle_buffer_size,
      num_epoch=config.num_epoch,
      grain_worker_count=config.grain_worker_count,
      grain_num_threads=config.grain_num_threads,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
      mixture_config_path=config.grain_train_mixture_config_path,
      elastic=config.grain_use_elastic_iterator,
      dataset_config=dataset_config,
      split="train",
  )

  preprocessing_fn = functools.partial(
      pipeline_fn,
      config=config,
      data_columns=config.train_data_columns,
      tokenize=config.tokenize_train_data,
      grain_worker_count=config.grain_worker_count,
      grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
  )

  # In the case of using colocated python for data input, partial functions such as
  # get_ds_fn (data initialization) and preprocessing_fn (data transformation)
  # are passed to the RemoteIteratorWrapper, which will then be passed to RemoteIterator
  # that runs in the colocated python environment.
  # While in other cases, get_ds_fn and preprocessing_fn to produce a data iterator and
  # pass to MultiHostDataLoadIterator
  if config.colocated_python_data_input:
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
    )

  if 0 < config.expansion_factor_real_data < 1:
    num_dataloader_to_restore = int(1 / config.expansion_factor_real_data)
    train_dataloader_list = []
    dataloading_host_count = len(process_indices) * num_dataloader_to_restore
    for i in range(num_dataloader_to_restore):
      dataloading_host_index = len(process_indices) * i + process_indices.index(jax.process_index())
      train_ds = get_ds_fn(dataloading_host_index=dataloading_host_index, dataloading_host_count=dataloading_host_count)
      train_dataloader = preprocessing_fn(dataset=train_ds)
      train_dataloader_list.append(train_dataloader)
    return [
        multihost_dataloading.MultiHostDataLoadIterator(x, global_mesh, config.generate_padding_batch_train)
        for x in train_dataloader_list
    ]

  # Default non-colocated, expansion_factor_real_data >=1 path
  shard_index = process_indices.index(jax.process_index())
  shard_count = len(process_indices)
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
        train_ds, config, preprocessing_fn, shard_index=shard_index, shard_count=shard_count, mp_opts=mp_options
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
    mmap_npy_eval_num_samples = eval_rounds * config.eval_steps * config.global_batch_size_to_load

  dataset_config = _build_dataset_config(
      config,
      num_samples=mmap_npy_eval_num_samples,
      seed=config.data_shuffle_seed,
      split_ratio=config.mmap_npy_split or None,
      split_index=1 if config.mmap_npy_split else 0,
  )

  get_ds_fn = functools.partial(
      get_datasets,
      config.grain_eval_files,
      config.grain_file_type,
      shuffle=False,  # No shuffle for eval
      shuffle_seed=config.data_shuffle_seed,
      shuffle_buffer_size=config.grain_shuffle_buffer_size,
      num_epoch=1,
      grain_worker_count=config.grain_worker_count_eval,
      grain_num_threads=config.grain_num_threads_eval,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size_eval,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
      dataset_config=dataset_config,
      split="eval",
  )

  preprocessing_fn = functools.partial(
      pipeline_fn,
      config=config,
      data_columns=config.eval_data_columns,
      tokenize=config.tokenize_eval_data,
      grain_worker_count=config.grain_worker_count_eval,
      grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
  )

  if not config.colocated_python_data_input:
    eval_ds = get_ds_fn(
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
    )
    eval_dataloader = preprocessing_fn(dataset=eval_ds)
    return multihost_dataloading.MultiHostDataLoadIterator(
        eval_dataloader, global_mesh, config.generate_padding_batch_eval
    )
  else:
    global_shape = (config.global_batch_size_to_load, config.max_target_length)
    return multihost_dataloading.RemoteIteratorWrapper(
        get_ds_fn,
        preprocessing_fn,
        global_mesh,
        global_shape,
        checkpoint_path=config.checkpoint_dir,
        elastic=config.grain_use_elastic_iterator,
    )
