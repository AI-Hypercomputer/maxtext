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

from input_pipeline import _hf_operations

import multihost_dataloading

def get_datasets(
  config: ml_collections.ConfigDict
):
  """Load huggingface dataset"""
  train_ds = load_dataset(config.dataset_name,
                            config.dataset_dir,
                            data_dir=config.dataset_dir,
                            data_files=config.dataset_path,
                            split="train",
                            streaming=True)
  return train_ds, None

def preprocess_dataset(config: ml_collections.ConfigDict,
                        global_mesh,
                        train_ds,
                        add_bos = True,
                        add_eos = True,
                        ):
  """preprocess dataset"""
  # Set global batch size.
  global_batch_size_to_load = config.global_batch_size_to_load

  train_iter = preprocessing_pipeline(
      dataset=train_ds,
      tokenizer_loader=config.tokenizer_loader,
      tokenizer_path=config.tokenizer_path,
      add_bos=add_bos,
      add_eos=add_eos,
      batch_size=global_batch_size_to_load,
      global_mesh=global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=1,
      pack_examples=True,
      max_length=config.max_target_length,
      data_shuffle_seed=config.data_shuffle_seed,
      access_token=config.hf_access_token,)

  return train_iter, None, None

def preprocessing_pipeline(
  dataset,
  tokenizer_loader,
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
  drop_remainder: bool = True,  # does not support drop_remainder
  data_shuffle_seed = 0,
  access_token: Union[str | None] = None
):
  """pipeline for preprocessing"""
  assert (
        batch_size % global_mesh.size == 0
  ), 'Batch size should be divisible number of global devices.'

  #dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())
  dataset = split_dataset_by_node(dataset, world_size=jax.process_count(), rank=jax.process_index())

  tokenizer = _hf_operations.load_tokenizer(tokenizer_loader,
                                            tokenizer_path,
                                            add_bos,
                                            add_eos,
                                            max_length,
                                            access_token)

  dataset = dataset.map(_hf_operations.tokenization, batched=True,
                   fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})

  dataset = dataset.map(_hf_operations.normalize_features, batched=True,
                        fn_kwargs={"key":"input_ids"})

  dataset = dataset.select_columns(['inputs', 'targets'])

  dataset = dataset.with_format("np")

  if shuffle:
    dataset = dataset.shuffle(seed=data_shuffle_seed)

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
