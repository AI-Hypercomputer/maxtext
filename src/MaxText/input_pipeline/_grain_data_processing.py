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

from MaxText.utils import gcs_utils
from MaxText.input_pipeline import _input_pipeline_utils
from MaxText.input_pipeline import _grain_tokenizer
from MaxText import multihost_dataloading
from MaxText import max_logging
from MaxText import tokenizer


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


def get_datasets(
    data_file_pattern,
    data_file_type,
    shuffle,
    shuffle_seed,
    num_epoch,
    dataloading_host_index,
    dataloading_host_count,
    grain_worker_count,
    grain_num_threads,
    grain_prefetch_buffer_size,
    grain_data_source_max_workers,
    mixture_config_path=None,
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
        if shuffle:
          ds = ds.shuffle(seed=shuffle_seed)
        ds = ds.repeat(num_epoch)
        ds = ds[dataloading_host_index::dataloading_host_count]  # sharding
        ds = ds.to_iter_dataset()
        datasets_dict[name] = ds

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
        if shuffle:
          dataset_list[d] = dataset_list[d].shuffle(seed=shuffle_seed)
        dataset_list[d] = dataset_list[d].repeat(num_epoch)
        dataset_list[d] = dataset_list[d][dataloading_host_index::dataloading_host_count]  # sharding
        dataset_list[d] = dataset_list[d].to_iter_dataset(
            read_options=grain.ReadOptions(
                num_threads=grain_num_threads,
                prefetch_buffer_size=grain_prefetch_buffer_size,
            )
        )
      # Use IterDataset.mix instead of MapDataset.mix in order to have per-mixture component checkpoints
      # for supporting changing the mixture after checkpointing
      dataset = grain.IterDataset.mix(dataset_list, weights)
      return dataset
    else:
      # Single pattern case - no need for parallelization
      dataset = create_dataset_from_pattern(data_file_pattern)
      if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)
      dataset = dataset.repeat(num_epoch)
      dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
      dataset = dataset.to_iter_dataset(
          read_options=grain.ReadOptions(
              num_threads=grain_num_threads,
              prefetch_buffer_size=grain_prefetch_buffer_size,
          )
      )
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
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=len(dataset))
    dataset = grain.experimental.WindowShuffleIterDataset(dataset, window_size=100, seed=shuffle_seed)
    return dataset
  else:
    raise ValueError(f"grain pipeline supports (arrayrecord, parquet) as grain_file_type, but got {data_file_type}")


def pretrain_preprocessing_pipeline(
    dataset,
    config,
    data_columns,
    tokenize,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Use grain pipeline to pre-process the dataset and return iterators for pretrain"""
  if config.grain_file_type == "arrayrecord":
    dataset = dataset.map(_input_pipeline_utils.ParseFeatures(data_columns, tokenize))
    dataset = dataset.map(_input_pipeline_utils.NormalizeFeatures(data_columns, tokenize))

  assert len(data_columns) == 1
  text_column = data_columns[0]

  tokenizer_model = tokenizer.build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      config.add_bos,
      config.add_eos,
      config.hf_access_token,
      config.dataset_type,
  )
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1

  if tokenize:
    if config.use_truncation:
      dataset = dataset.map(_grain_tokenizer.TokenizeAndTrim(text_column, config.max_target_length, tokenizer_model))
    else:
      dataset = dataset.apply(_grain_tokenizer.TokenizeAndChunk(text_column, config.max_target_length, tokenizer_model))

  data_columns = ("inputs", "targets")
  rekey_dict = {col: text_column for col in data_columns}
  dataset = dataset.map(_input_pipeline_utils.Rekey(rekey_dict))

  # Pack and Batch examples.
  batch_size = config.global_batch_size_to_load // jax.process_count()
  if config.expansion_factor_real_data > 1:
    # global_batch_size_to_load has been expanded in pyconfig.py when expansion_factor_real_data > 1.
    # But when using Grain, we want to keep the batch_size consistent with that in the checkpoint.
    # We revert the batch_size expansion here, but load multiple batches per step in multihost_dataloading.py.
    batch_size = batch_size // config.expansion_factor_real_data
  if config.packing:
    length_struct = {col: config.max_target_length for col in data_columns}
    dataset = grain.experimental.FirstFitPackIterDataset(
        dataset, length_struct=length_struct, num_packing_bins=batch_size
    )
    rekey_dict = {
        "targets_segmentation": "targets_segment_ids",
        "inputs_segmentation": "inputs_segment_ids",
        "targets_position": "targets_positions",
        "inputs_position": "inputs_positions",
    }
    dataset = dataset.map(_input_pipeline_utils.Rekey(rekey_dict))
  else:
    dataset = dataset.map(_input_pipeline_utils.PadOrTrimToMaxLength(config.max_target_length, pad_id))
  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  # Shift inputs for teacher-forced training
  dataset = dataset.map(
      _input_pipeline_utils.ShiftData(
          ignored_ids=[pad_id],
          axis=1,
      )
  )
  dataset = dataset.mp_prefetch(
      grain.MultiprocessingOptions(
          num_workers=grain_worker_count,
          per_worker_buffer_size=grain_per_worker_buffer_size,
      )
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
  if config.grain_file_type == "arrayrecord":
    dataset = dataset.map(_input_pipeline_utils.ParseFeatures(data_columns, tokenize))
    dataset = dataset.map(_input_pipeline_utils.NormalizeFeatures(data_columns, tokenize))
  tokenizer_model = tokenizer.build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      config.add_bos,
      config.add_eos,
      config.hf_access_token,
      config.dataset_type,
  )
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1

  if tokenize:
    dataset = dataset.map(_grain_tokenizer.TokenizeAndTrim(data_columns, config.max_target_length, tokenizer_model))

  dataset = dataset.map(_input_pipeline_utils.PadOrTrimToMaxLength(config.max_target_length, pad_id))
  batch_size = config.global_batch_size_to_load // jax.process_count()
  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)
  dataset = dataset.mp_prefetch(
      grain.MultiprocessingOptions(
          num_workers=grain_worker_count,
          per_worker_buffer_size=grain_per_worker_buffer_size,
      )
  )
  return dataset


def make_grain_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  assert (
      config.global_batch_size_to_load % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."
  if not config.colocated_python_data_input and not 0 < config.expansion_factor_real_data < 1:
    train_ds = get_datasets(
        config.grain_train_files,
        config.grain_file_type,
        shuffle=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed,
        num_epoch=config.num_epoch,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        grain_worker_count=config.grain_worker_count,
        grain_num_threads=config.grain_num_threads,
        grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
        mixture_config_path=config.grain_train_mixture_config_path,
    )
    if config.use_dpo:
      train_dataloader = dpo_preprocessing_pipeline(
          train_ds,
          config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
      )
    else:
      train_dataloader = pretrain_preprocessing_pipeline(
          train_ds,
          config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
      )
    return multihost_dataloading.MultiHostDataLoadIterator(
        train_dataloader,
        global_mesh,
        config.generate_padding_batch_train,
        expansion_loading_factor_for_grain=config.expansion_factor_real_data,
    )
  else:
    get_ds_fn = functools.partial(
        get_datasets,
        config.grain_train_files,
        config.grain_file_type,
        shuffle=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed,
        num_epoch=config.num_epoch,
        grain_worker_count=config.grain_worker_count,
        grain_num_threads=config.grain_num_threads,
        grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
    )
    if config.use_dpo:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
      )
    else:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
      )
    if config.colocated_python_data_input:
      global_shape = (config.global_batch_size_to_load, config.max_target_length)
      return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, global_mesh, global_shape)
    else:
      # config.expansion_factor_real_data is between 0 and 1
      num_dataloader_to_restore = int(1 / config.expansion_factor_real_data)
      train_dataloader_list = []
      dataloading_host_count = len(process_indices) * num_dataloader_to_restore
      for i in range(num_dataloader_to_restore):
        dataloading_host_index = len(process_indices) * i + process_indices.index(jax.process_index())
        train_ds = get_ds_fn(dataloading_host_index=dataloading_host_index, dataloading_host_count=dataloading_host_count)
        train_dataloader = preprocessing_fn(train_ds)
        train_dataloader_list.append(train_dataloader)
      return [
          multihost_dataloading.MultiHostDataLoadIterator(x, global_mesh, config.generate_padding_batch_train)
          for x in train_dataloader_list
      ]


def make_grain_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  assert (
      config.global_batch_size_to_load_eval % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."
  if not config.colocated_python_data_input:
    eval_ds = get_datasets(
        config.grain_eval_files,
        config.grain_file_type,
        shuffle=False,
        shuffle_seed=config.data_shuffle_seed,
        num_epoch=1,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        grain_worker_count=config.grain_worker_count_eval,
        grain_num_threads=config.grain_num_threads_eval,
        grain_prefetch_buffer_size=config.grain_prefetch_buffer_size_eval,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
    )
    if config.use_dpo:
      eval_dataloader = dpo_preprocessing_pipeline(
          eval_ds,
          config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      )
    else:
      eval_dataloader = pretrain_preprocessing_pipeline(
          eval_ds,
          config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      )
    return multihost_dataloading.MultiHostDataLoadIterator(
        eval_dataloader, global_mesh, config.generate_padding_batch_eval
    )
  else:
    get_ds_fn = functools.partial(
        get_datasets,
        config.grain_eval_files,
        config.grain_file_type,
        shuffle=False,  # No shuffle for eval
        shuffle_seed=config.data_shuffle_seed,
        num_epoch=1,
        grain_worker_count=config.grain_worker_count_eval,
        grain_num_threads=config.grain_num_threads_eval,
        grain_prefetch_buffer_size=config.grain_prefetch_buffer_size_eval,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
    )
    if config.use_dpo:
      preprocessing_fn = functools.partial(
          dpo_preprocessing_pipeline,
          config=config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      )
    else:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      )
    global_shape = (config.global_batch_size_to_load, config.max_target_length)
    return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, global_mesh, global_shape)
