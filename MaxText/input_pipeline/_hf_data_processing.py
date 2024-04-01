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

from typing import Optional

import ml_collections
import jax

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

#from transformers import LlamaTokenizer
from transformers import AutoTokenizer
from input_pipeline import _hf_operations

import multihost_dataloading

def get_datasets(
  config: ml_collections.ConfigDict
):
  """Load huggingface dataset"""
  train_ds = load_dataset(config.dataset_name,
                            data_dir=config.dataset_dir,
                            data_files=config.dataset_files,
                            split="train",
                            streaming=True)

  if config.eval_dataset_name:
    eval_ds = load_dataset(config.eval_dataset_name,
                            data_dir=config.dataset_dir,
                            data_files=config.dataset_files,
                            split="train",
                            streaming=True)
  else:
    eval_ds = train_ds
  return train_ds, eval_ds

def preprocess_dataset(config: ml_collections.ConfigDict,
                        global_mesh,
                        train_ds, eval_ds,
                        tokenizer_path,
                        data_shuffle_seed = 0,
                        add_bos = True,
                        add_eos = True
                        ):
  """preprocess dataset"""
  # Set global batch size.
  global_batch_size_to_load = config.global_batch_size_to_load

#   if config.eval_per_device_batch_size > 0:
#     eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
#   else:
#     eval_batch_size = global_batch_size_to_load

  train_iter = preprocessing_pipeline(
      train_ds,
      tokenizer_path,
      add_bos,
      add_eos,
      global_batch_size_to_load,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=1,
      pack_examples=True,
      max_length=config.max_target_length,
      data_shuffle_seed=data_shuffle_seed,)

  eval_iter = preprocessing_pipeline(
      eval_ds,
      tokenizer_path,
      add_bos,
      add_eos,
      global_batch_size_to_load,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=1,
      pack_examples=True,
      max_length=config.max_target_length,
      data_shuffle_seed=data_shuffle_seed,)

  predict_iter = preprocessing_pipeline(
      eval_ds,
      tokenizer_path,
      add_bos,
      add_eos,
      global_batch_size_to_load,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=1,
      pack_examples=True,
      max_length=config.max_target_length,
      data_shuffle_seed=data_shuffle_seed,)

  return train_iter, eval_iter, predict_iter

def preprocessing_pipeline(
  dataset,
  tokenizer_path,
  add_bos: bool,
  add_eos: bool,
  batch_size: int,
  global_mesh,
  shuffle: bool,
  num_epochs: Optional[int] = 1,  # only support num_epoch=1 for now
  pack_examples: bool = True,
  max_length: int = 512,
  shift: bool = True,
  drop_remainder: bool = True,
  data_shuffle_seed = 0,
):
  """pipeline for preprocessing"""
  assert (
        batch_size % global_mesh.size == 0
  ), 'Batch size should be divisible number of global devices.'

  #dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())
  dataset = split_dataset_by_node(dataset, world_size=jax.process_count(), rank=jax.process_index())

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                            add_bos_token=add_bos,
                                            add_eos_token=add_eos)

  dataset = dataset.map(_hf_operations.tokenization, batched=True,
                   fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})

  dataset = dataset.map(_hf_operations.normalize_features, batched=True,
                        fn_kwargs={"key":"input_ids"})

  dataset = dataset.select_columns(['inputs', 'targets'])

  dataset = dataset.with_format("np")

  if shuffle:
    dataset = dataset.shuffle(seed=data_shuffle_seed)

#   if shift:
#     dataset = dataset.map(_hf_operations.shift_input, batched=True)

  if pack_examples:
    pack_op = _hf_operations.PackAndBatchOperation(
      batch_size=batch_size // jax.process_count(),
      length_struct={"inputs": max_length, "targets":max_length},
      shift_inputs=shift,
    )
    dataset = _hf_operations.TransformedDataset(pack_op, dataset)

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataset, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen
