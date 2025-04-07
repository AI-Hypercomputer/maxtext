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
from pathlib import Path

import ml_collections
import jax
import grain.python as grain

from input_pipeline import _input_pipeline_utils
from input_pipeline import _grain_tokenizer

import multihost_dataloading


def get_datasets(data_file_pattern):
  """Load dataset from array_record files for using with grain"""
  data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
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
    data_columns,
    shuffle: bool = False,
    data_shuffle_seed=0,
    tokenize=True,
    add_bos=True,
    add_eos=True,
    num_epochs=1,
    packing=True,
    shift=True,
    drop_remainder=False,
    use_dpo: bool = False,
):
  """Use grain to pre-process the dataset and return iterators"""
  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

  operations = []
  operations.append(_input_pipeline_utils.ParseFeatures(data_columns, tokenize))
  if not use_dpo:
    assert len(data_columns) == 1
    operations.append(_input_pipeline_utils.InputsTargetsFeatures(data_columns[0]))
    data_columns = ("inputs", "targets")
  operations.append(_input_pipeline_utils.NormalizeFeatures(data_columns, tokenize))

  if tokenize:
    operations.append(_grain_tokenizer.TokenizeAndTrim(data_columns, max_target_length, tokenizer_path, add_bos, add_eos))

  # Pack and Batch examples.
  if packing and not use_dpo:
    length_struct = {col: max_target_length for col in data_columns}
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=global_batch_size // jax.process_count(),
            length_struct=length_struct,
        )
    )
    operations.append(_input_pipeline_utils.ReformatPacking(data_columns))
  else:
    operations.append(_input_pipeline_utils.PadToMaxLength(max_target_length))
    operations.append(grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder))

  # Shift inputs for teacher-forced training
  if shift and not use_dpo:
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


def make_grain_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  assert config.global_batch_size_to_load % global_mesh.size == 0, "Batch size should be divisible number of global devices."
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
    return multihost_dataloading.MultiHostDataLoadIterator(train_dataloader, global_mesh, config)
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
  assert (
      config.global_batch_size_to_load_eval % global_mesh.size == 0
  ), "Batch size should be divisible number of global devices."
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
    return multihost_dataloading.MultiHostDataLoadIterator(eval_dataloader, global_mesh, config)
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
