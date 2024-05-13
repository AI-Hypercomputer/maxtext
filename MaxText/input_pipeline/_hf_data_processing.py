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

"""Input pipeline using Huggingface datasets."""

from typing import Optional, Union

import ml_collections
import jax

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from functools import partial
from transformers import AutoTokenizer, LlamaTokenizer

import grain.python as grain
from input_pipeline import _hf_operations
from input_pipeline import _grain_operations
import multihost_dataloading


def get_datasets(
  config: ml_collections.ConfigDict
):
  """Load huggingface dataset"""
  train_ds = load_dataset(config.dataset_name,
                            data_dir=config.dataset_dir_hf,
                            data_files=config.dataset_path,
                            split="train",
                            streaming=True,
                            token=config.hf_access_token)
  return train_ds, None

def preprocess_dataset(config: ml_collections.ConfigDict,
                        dataloading_host_index,
                        dataloading_host_count,
                        global_mesh,
                        dataset,
                        add_bos = True,
                        add_eos = True,
                        ):
  """preprocess dataset"""
  # Set global batch size.
  batch_size = config.global_batch_size_to_load

  assert (
        batch_size % global_mesh.size == 0
  ), 'Batch size should be divisible number of global devices.'
  
  if config.enable_data_shuffling:
    dataset = dataset.shuffle(seed=config.data_shuffle_seed)

  tokenizer =  AutoTokenizer.from_pretrained(config.tokenizer_path,
                                            add_bos_token=add_bos,
                                            add_eos_token=add_eos,
                                            model_max_length=config.max_target_length)

  dataset = dataset.map(_hf_operations.tokenization, batched=True,
                        fn_kwargs={"tokenizer": tokenizer, "max_length": config.max_target_length-1})
  dataset = dataset.select_columns(["input_ids"])

  #dataset = split_dataset_by_node(dataset, world_size=jax.process_count(), rank=jax.process_index())

  dataset = _hf_operations.HFDataSource(dataset,
                                        dataloading_host_index,
                                        dataloading_host_count,
                                        config.num_threads,
                                        config.grain_worker_count,
                                        add_bos,
                                        add_eos)

  operations = []
  operations.append(_grain_operations.HFNormalizeFeatures())
  operations.append(grain.experimental.PackAndBatchOperation(
                          batch_size=batch_size // dataloading_host_count,
                          length_struct={'inputs':config.max_target_length,
                                        'targets':config.max_target_length}))
  operations.append(_grain_operations.ReformatPacking())

  # operations.append(_grain_operations.PadToMaxLength(config.max_target_length))
  # operations.append(grain.Batch(batch_size=batch_size // jax.process_count(), drop_remainder=True))

  operations.append(_grain_operations.ShiftData(axis=1))

  index_sampler = grain.IndexSampler(
    num_records=len(dataset),
    num_epochs = 1,
    shard_options=grain.ShardOptions(
      shard_index = 0, shard_count = 1, drop_remainder = True
    ),
    shuffle = False,
    seed = 0
  )

  dataloader = grain.DataLoader(
    data_source = dataset,
    operations = operations,
    sampler = index_sampler,
    worker_count = config.grain_worker_count,
    worker_buffer_size = 1,
    read_options = grain.ReadOptions(num_threads=config.num_threads, prefetch_buffer_size=128)
  )

  train_iter = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

  # Return multi-host jax.Array prep iterator
  return train_iter, None, None

