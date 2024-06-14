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


def get_datasets(config: ml_collections.ConfigDict):
  """Load dataset from array_record files for using with grain"""
  train_files = glob.glob(config.grain_data_files)
  train_ds = grain.ArrayRecordDataSource(train_files)

  return train_ds, None


def preprocess_dataset(
    config: ml_collections.ConfigDict,
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    dataset,
    num_epochs=1,
    add_bos=True,
    add_eos=True,
    packing=True,
    shift=True,
    drop_remainder=True,
):
  """Use grain to pre-process the dataset and return iterators"""
  # Set global batch size.
  global_batch_size = config.global_batch_size_to_load
  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

  operations = []
  operations.append(_input_pipeline_utils.ParseFeatures())
  operations.append(_input_pipeline_utils.NormalizeFeatures())
  operations.append(_grain_tokenizer.TokenizeAndTrim(["inputs", "targets"], config.max_target_length, config.tokenizer_path, add_bos, add_eos))

  # Pack and Batch examples.
  if packing:
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=global_batch_size // jax.process_count(), length_struct={"inputs": config.max_target_length, "targets": config.max_target_length}
        )
    )
    operations.append(_input_pipeline_utils.ReformatPacking())
  else:
    operations.append(_input_pipeline_utils.PadToMaxLength(config.max_target_length))
    operations.append(grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder))

  # Shift inputs for teacher-forced training
  if shift:
    operations.append(_input_pipeline_utils.ShiftData(axis=1))

  index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=num_epochs,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=True
      ),
      shuffle=config.enable_data_shuffling,
      seed=config.data_shuffle_seed,
  )

  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=index_sampler,
      worker_count=config.grain_worker_count,
  )

  train_iter = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

  # Return multi-host jax.Array prep iterator
  return train_iter, None, None
