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

import jax

import grain.python as grain

from MaxText.input_pipeline import _input_pipeline_utils
from MaxText.input_pipeline import _grain_tokenizer
from MaxText import multihost_dataloading
from MaxText import max_logging
from MaxText import tokenizer


def find_data_files(data_file_pattern):
  data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
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
):
  """Load dataset from array_record files for using with grain"""
  if data_file_type == "arrayrecord":
    if ";" in data_file_pattern:
      data_file_patterns, weights = zip(*[pattern.split(":") for pattern in data_file_pattern.split(";")])
      assert len(data_file_patterns) == len(weights), "Number of data file patterns and weights must match"
      weights = [float(weight) for weight in weights]
      weights = [round(weight / sum(weights), 4) for weight in weights]
      dataset_list = [
          grain.MapDataset.source(grain.ArrayRecordDataSource(find_data_files(pattern))) for pattern in data_file_patterns
      ]
      dataset = grain.MapDataset.mix(dataset_list, weights)
    else:
      data_files = find_data_files(data_file_pattern)
      dataset = grain.MapDataset.source(grain.ArrayRecordDataSource(data_files))
    if shuffle:
      dataset = dataset.shuffle(seed=shuffle_seed)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
    dataset = dataset.to_iter_dataset()
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
  else:
    raise ValueError(f"grain pipeline supports (arrayrecord, parquet) as grain_file_type, but got {data_file_type}")

  return dataset


def pretrain_preprocessing_pipeline(dataset, config, data_columns, tokenize, grain_worker_count):
  """Use grain pipeline to pre-process the dataset and return iterators for pretrain"""
  if config.grain_file_type == "arrayrecord":
    dataset = dataset.map(_input_pipeline_utils.ParseFeatures(data_columns, tokenize))
    dataset = dataset.map(_input_pipeline_utils.NormalizeFeatures(data_columns, tokenize))

  assert len(data_columns) == 1
  rekey_dict = {"inputs": "text", "targets": "text"}
  dataset = dataset.map(_input_pipeline_utils.Rekey(rekey_dict))
  data_columns = ("inputs", "targets")

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
    dataset = dataset.map(
        _grain_tokenizer.TokenizeAndTrim(
            data_columns, config.max_target_length, config.add_bos, config.add_eos, tokenizer_model
        )
    )
  # Pack and Batch examples.
  batch_size = config.global_batch_size_to_load // jax.process_count()
  if config.packing:
    length_struct = {col: config.max_target_length for col in data_columns}
    dataset = grain.experimental.FirstFitPackIterDataset(dataset, length_struct=length_struct, num_packing_bins=batch_size)
    rekey_dict = {
        "targets_segmentation": "targets_segment_ids",
        "inputs_segmentation": "inputs_segment_ids",
        "targets_position": "targets_positions",
        "inputs_position": "inputs_positions",
    }
    dataset = dataset.map(_input_pipeline_utils.Rekey(rekey_dict))
  else:
    dataset = dataset.map(_input_pipeline_utils.PadToMaxLength(config.max_target_length, pad_id))
  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  # Shift inputs for teacher-forced training
  dataset = dataset.map(
      _input_pipeline_utils.ShiftData(
          ignored_ids=[pad_id],
          axis=1,
      )
  )
  dataset = dataset.mp_prefetch(grain.MultiprocessingOptions(num_workers=grain_worker_count))
  return dataset


def dpo_preprocessing_pipeline(dataset, config, data_columns, tokenize, grain_worker_count):
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
    dataset = dataset.map(
        _grain_tokenizer.TokenizeAndTrim(
            data_columns, config.max_target_length, config.add_bos, config.add_eos, tokenizer_model
        )
    )

  dataset = dataset.map(_input_pipeline_utils.PadToMaxLength(config.max_target_length, pad_id))
  batch_size = config.global_batch_size_to_load // jax.process_count()
  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)
  dataset = dataset.mp_prefetch(grain.MultiprocessingOptions(num_workers=grain_worker_count))
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
  if not config.colocated_python_data_input:
    train_ds = get_datasets(
        config.grain_train_files,
        config.grain_file_type,
        shuffle=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed,
        num_epoch=config.num_epoch,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        grain_worker_count=config.grain_worker_count,
    )
    if config.use_dpo:
      train_dataloader = dpo_preprocessing_pipeline(
          train_ds,
          config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
      )
    else:
      train_dataloader = pretrain_preprocessing_pipeline(
          train_ds,
          config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
      )
    return multihost_dataloading.MultiHostDataLoadIterator(
        train_dataloader, global_mesh, config.generate_padding_batch_train
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
    )
    if config.use_dpo:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
      )
    else:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
      )
    global_shape = (config.global_batch_size_to_load, config.max_target_length)
    return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, global_mesh, global_shape)


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
    )
    if config.use_dpo:
      eval_dataloader = dpo_preprocessing_pipeline(
          eval_ds,
          config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
      )
    else:
      eval_dataloader = pretrain_preprocessing_pipeline(
          eval_ds,
          config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
      )
    return multihost_dataloading.MultiHostDataLoadIterator(eval_dataloader, global_mesh, config.generate_padding_batch_eval)
  else:
    get_ds_fn = functools.partial(
        get_datasets,
        config.grain_eval_files,
        config.grain_file_type,
        shuffle=False,  # No shuffle for eval
        shuffle_seed=config.data_shuffle_seed,
        num_epoch=1,
        grain_worker_count=config.grain_worker_count_eval,
    )
    if config.use_dpo:
      preprocessing_fn = functools.partial(
          dpo_preprocessing_pipeline,
          config=config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
      )
    else:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
      )
    global_shape = (config.global_batch_size_to_load, config.max_target_length)
    return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, global_mesh, global_shape)
