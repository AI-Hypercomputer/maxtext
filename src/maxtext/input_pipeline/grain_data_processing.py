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
from maxtext.input_pipeline import multihost_dataloading
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
):
  """Load dataset from array_record files for using with grain"""
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
  elif data_file_type == "tfrecord":
    data_files = find_data_files(data_file_pattern)
    dataset = grain.MapDataset.source(data_files)
    if shuffle:
      dataset = dataset.shuffle(seed=shuffle_seed)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
    dataset = dataset.map(input_pipeline_utils.make_tfrecord_iter_dataset)
    files_per_host = max(len(data_files) // dataloading_host_count, 1)
    cycle_length = min(files_per_host, grain_num_threads)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=cycle_length)
    if shuffle:
      dataset = grain.experimental.WindowShuffleIterDataset(dataset, window_size=shuffle_buffer_size, seed=shuffle_seed)
    return dataset
  elif data_file_type == "parquet":
    data_files = find_data_files(data_file_pattern)
    dataset = grain.MapDataset.source(data_files)
    if shuffle:
      dataset = dataset.shuffle(seed=shuffle_seed)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
    assert grain_worker_count <= len(dataset), (
        f"grain worker count is currently {grain_worker_count}, exceeding the max allowable value {len(dataset)} "
        f"(file shard count of a data loading host) for your dataset. "
        f"Please lower grain_worker_count or increase file shard count."
    )
    dataset = dataset.map(grain.experimental.ParquetIterDataset)
    cycle_length = min(len(dataset) // num_epoch, grain_num_threads)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=cycle_length)
    if shuffle:
      dataset = grain.experimental.WindowShuffleIterDataset(dataset, window_size=shuffle_buffer_size, seed=shuffle_seed)
    return dataset
  else:
    raise ValueError(
        f"grain pipeline supports (arrayrecord, tfrecord, parquet) as grain_file_type, but got {data_file_type}"
    )


def pretrain_preprocessing_pipeline(
    dataset,
    config,
    data_columns,
    tokenize,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Use grain pipeline to pre-process the dataset and return iterators for pretrain.

  When `config.grain_use_elastic_iterator` is True, the pipeline stops before batching
  and multiprocessing (which `ElasticIterator` performs itself) and applies
  shift pre-batch on axis 0 rather than post-batch on axis 1.
  """
  dataset = data_processing_utils.parse_and_keep_features(dataset, config, data_columns, tokenize)

  assert len(data_columns) == 1
  text_column = data_columns[0]

  tokenizer_model, pad_id = data_processing_utils.get_tokenizer_and_pad_id(config)

  if tokenize:
    if config.use_truncation:
      dataset = dataset.map(grain_tokenizer.TokenizeAndTrim(text_column, config.max_target_length, tokenizer_model))
    else:
      dataset = dataset.apply(grain_tokenizer.TokenizeAndChunk(text_column, config.max_target_length, tokenizer_model))

  data_columns = ("inputs", "targets")
  rekey_dict = {col: text_column for col in data_columns}
  dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict))

  batch_size = data_processing_utils.get_local_batch_size(config)
  dataset = data_processing_utils.format_and_batch(dataset, config, batch_size, pad_id, data_columns, tokenizer_model)
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

  batch_size = config.global_batch_size_to_load // jax.process_count()
  # DPO scores full sequences, so no shift.
  dataset = data_processing_utils.format_and_batch(
      dataset, config, batch_size, pad_id, data_columns, tokenizer_model, shift=False
  )
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

  # Default non-colocated, non-expansion path
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
