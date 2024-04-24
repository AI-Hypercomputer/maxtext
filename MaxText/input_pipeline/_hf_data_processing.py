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
from torchdata.datapipes.iter import Collator, IterableWrapper

from torchdata.dataloader2 import DataLoader2, InProcessReadingService

from input_pipeline import _hf_operations
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
  access_token: Union[str | None] = None,
  prefetch_buffer_size: int = 10,
):
  """pipeline for preprocessing"""
  assert (
        batch_size % global_mesh.size == 0
  ), 'Batch size should be divisible number of global devices.'

  dataset = split_dataset_by_node(dataset, world_size=jax.process_count(), rank=jax.process_index())

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                            add_bos_token=add_bos,
                                            add_eos_token=add_eos,
                                            model_max_length=max_length,
                                            token=access_token,
                                            legacy=False)

  dataset = dataset.map(_hf_operations.tokenization, batched=True,
                        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length-1})

  # dataset = dataset.map(_hf_operations.normalize_features, batched=True,
  #                       fn_kwargs={"key":"input_ids"})

  # dataset = dataset.select_columns(['inputs', 'targets'])
  dataset = dataset.select_columns(["input_ids"])
  dataset = dataset.rename_column("input_ids", "targets")

  # dataset = dataset.map(batched=True, batch_size=max_length)
  dataset = dataset.map(_hf_operations.shift)

  if shuffle:
    dataset = dataset.shuffle(seed=data_shuffle_seed)

  dataset = dataset.map(_hf_operations.group_batch, batched=True, batch_size=20)
  dataset = dataset.map(_hf_operations.pack_in_batch_hf, fn_kwargs={"max_len": max_length})
  dataset = dataset.map(_hf_operations.unbatch, batched=True)
  # dataset = dataset.map(_hf_operations.group_batch, batched=True,
  #                       batch_size=batch_size // jax.process_count())

  dataset = IterableWrapper(dataset)
  dataset = dataset.batch(batch_size // jax.process_count())
  dataset = dataset.map(_hf_operations.group_in_batch)
  # dataset.prefetch(prefetch_buffer_size)

  # if shift:
  #   dataset = dataset.map(_hf_operations.shift)

  # if pack_examples:
  #   dataset = dataset.batch(max_length)
  #   dataset = Collator(dataset, collate_fn=partial(_hf_operations.pack_in_batch, max_len=max_length))
  #   dataset = dataset.unbatch()
  #   dataset = dataset.batch(batch_size // jax.process_count())

  # dataset = dataset.map(_hf_operations.group_in_batch)

  # rs = InProcessReadingService(prefetch_cnt=batch_size // jax.process_count())
  # #rs = InProcessReadingService()
  # dataset = DataLoader2(dataset, reading_service=rs)

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataset, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen
