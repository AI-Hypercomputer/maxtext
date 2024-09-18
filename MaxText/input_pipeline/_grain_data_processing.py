"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Input pipeline using Grain."""

import glob

import ml_collections
import jax
import grain.python as grain
from grain._src.python.dataset.transformations.concat_then_split import ConcatThenSplitConfig

from input_pipeline import _input_pipeline_utils
from input_pipeline import _grain_tokenizer

import multihost_dataloading


def get_datasets(data_file_pattern, generate_padding_example=False, data_column_name=None, max_target_length=None):
  """Load dataset from array_record files for using with grain"""
  data_files = glob.glob(data_file_pattern)
  if generate_padding_example:
    print("Using padded datasource")
    dataset = _input_pipeline_utils.ArrayRecordDataSourceWithPadding(data_files, data_column_name, max_target_length)
  else:
    dataset = grain.ArrayRecordDataSource(data_files)
  return dataset


def preprocessing_pipeline(
    dataset,
    tokenizer_path,
    global_batch_size: int,
    global_mesh,
    max_target_length: int,
    grain_worker_count: int,
    dataloading_host_index,
    dataloading_host_count,
    data_column,
    shuffle: bool = False,
    data_shuffle_seed=0,
    tokenize=True,
    add_bos=True,
    add_eos=True,
    num_epochs=1,
    packing=True,
    shift=True,
    drop_remainder=False,
):
  """Use grain to pre-process the dataset and return iterators"""
  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

  ds = grain.MapDataset.source(dataset)
  ds = ds.map(_input_pipeline_utils.ParseFeatures(data_column, tokenize))
  ds = ds.map(_input_pipeline_utils.NormalizeFeatures(data_column, tokenize))

  if tokenize:
    ds = ds.map(_grain_tokenizer.TokenizeAndTrim(["inputs", "targets"], max_target_length, tokenizer_path, add_bos, add_eos))
    
  if shuffle:
    ds = ds.shuffle(data_shuffle_seed)
    
  ds = ds[dataloading_host_index::dataloading_host_count]
  ds = ds.to_iter_dataset(
    grain.ReadOptions(num_threads=64)
  )
  # Pack and Batch examples.
  if packing:
    ds = grain.experimental.ConcatThenSplitIterDataset(
      ds,
      config = ConcatThenSplitConfig(
          sequence_lengths={"inputs": max_target_length, "targets": max_target_length},
      ),
    )
    ds = ds.map(_input_pipeline_utils.ReformatPacking())

  else:
    ds = ds.map(_input_pipeline_utils.PadToMaxLength(max_target_length))

  ds = ds.batch(global_batch_size // jax.process_count(), drop_remainder=drop_remainder)

  # Shift inputs for teacher-forced training
  if shift:
    ds = ds.map(_input_pipeline_utils.ShiftData(axis=1))

  if not drop_remainder:
    ds = ds.map(_input_pipeline_utils.PadBatch(global_batch_size // jax.process_count()))
  
  if grain_worker_count > 0:
    ds = ds.prefetch(
      grain.MultiprocessingOptions(num_workers=grain_worker_count)
    )
    
  dataloader = ds.__iter__()

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen

def make_grain_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  if jax.process_index() in process_indices:
    train_ds = get_datasets(config.grain_train_files)
    train_iter = preprocessing_pipeline(
      dataset=train_ds,
      tokenizer_path=config.tokenizer_path,
      global_batch_size=config.global_batch_size_to_load,
      global_mesh=global_mesh,
      max_target_length=config.max_target_length,
      grain_worker_count=config.grain_worker_count,
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
      data_column=config.train_data_column,
      shuffle=config.enable_data_shuffling,
      data_shuffle_seed=config.data_shuffle_seed,
      tokenize=config.tokenize_train_data,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
    )
  else:
    train_iter = None

  if config.eval_interval > 0:
    if not config.tokenize_eval_data and config.eval_steps:
      # generate_padding_example only supports pre-tokenized dataset for now
      eval_ds = get_datasets(config.grain_eval_files,
                             generate_padding_example=True,
                             data_column_name=config.eval_data_column,
                             max_target_length=config.max_target_length)
    else:
      eval_ds = get_datasets(config.grain_eval_files, generate_padding_example=False)
    
    if config.eval_per_device_batch_size > 0:
      eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
    else:
      eval_batch_size = config.global_batch_size_to_load
      
    eval_iter = preprocessing_pipeline(
      dataset=eval_ds,
      tokenizer_path=config.tokenizer_path,
      global_batch_size=eval_batch_size,
      global_mesh=global_mesh,
      max_target_length=config.max_target_length,
      grain_worker_count=config.grain_worker_count,
      dataloading_host_index=jax.process_index(),
      dataloading_host_count=jax.process_count(),
      data_column=config.eval_data_column,
      shuffle=False,
      data_shuffle_seed=config.data_shuffle_seed,
      tokenize=config.tokenize_eval_data,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
    )
  else:
    eval_iter = None
  return train_iter, eval_iter
