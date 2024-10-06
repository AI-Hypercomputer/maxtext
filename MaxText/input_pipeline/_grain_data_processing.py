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

from input_pipeline import _input_pipeline_utils
from input_pipeline import _grain_tokenizer

import multihost_dataloading


def get_datasets(data_file_pattern):
  """Load dataset from array_record files for using with grain"""
  data_files = glob.glob(data_file_pattern)
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

  operations = []
  operations.append(_input_pipeline_utils.ParseFeatures(data_column, tokenize))
  operations.append(_input_pipeline_utils.NormalizeFeatures(data_column, tokenize))

  if tokenize:
    operations.append(
        _grain_tokenizer.TokenizeAndTrim(["inputs", "targets"], max_target_length, tokenizer_path, add_bos, add_eos)
    )

  # Pack and Batch examples.
  if packing:
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=global_batch_size // jax.process_count(),
            length_struct={"inputs": max_target_length, "targets": max_target_length},
        )
    )
    operations.append(_input_pipeline_utils.ReformatPacking())
  else:
    operations.append(_input_pipeline_utils.PadToMaxLength(max_target_length))
    operations.append(grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder))

  # Shift inputs for teacher-forced training
  if shift:
    operations.append(_input_pipeline_utils.ShiftData(axis=1))

  index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=num_epochs,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=drop_remainder
      ),
      shuffle=shuffle,
      seed=data_shuffle_seed,
  )

  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=index_sampler,
      worker_count=grain_worker_count,
  )

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen


def make_grain_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
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

  if config.eval_interval > 0:
    eval_ds = get_datasets(config.grain_eval_files)
    eval_iter = preprocessing_pipeline(
        dataset=eval_ds,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=config.global_batch_size_to_load_eval,
        global_mesh=global_mesh,
        max_target_length=config.max_target_length,
        grain_worker_count=config.grain_worker_count,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
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
