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

import ml_collections

import jax

import datasets

import transformers

import grain.python as grain

import numpy as np

from MaxText.input_pipeline import _input_pipeline_utils
from MaxText import multihost_dataloading


def preprocessing_pipeline(
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    dataset,
    data_column_names,
    tokenize,
    tokenizer_path,
    hf_access_token,
    global_batch_size,
    max_target_length,
    shuffle,
    data_shuffle_seed,
    add_bos=True,
    add_eos=True,
    packing=True,
    shift=True,
    num_threads=1,
    drop_remainder=False,
    generate_padding_example=False,
    use_dpo=None,
    use_sft=None,
    sft_train_on_completion_only=True,
    grain_worker_count=1,  # only support 0 or 1
):
  """pipeline for preprocessing HF dataset"""

  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

  if shuffle:
    dataset = dataset.shuffle(seed=data_shuffle_seed)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      tokenizer_path,
      add_bos_token=add_bos if not use_sft else False,
      add_eos_token=add_eos if not use_sft else False,
      legacy=False,
      token=hf_access_token,
  )

  if use_sft:
    dataset = dataset.select_columns(data_column_names)

    supported_columns = [["prompt", "completion"], ["messages"]]
    assert any(
        set(data_column_names) == set(supported) for supported in supported_columns
    ), f"Dataset column names mismatch. Expected columns to match one of {supported_columns}, but got {data_column_names}"
    assert _input_pipeline_utils.is_conversational(
        dataset.features, data_column_names
    ), "Dataset is not in conversational format."

    if len(data_column_names) > 1:
      combined_column_name = "messages"
      dataset_features = datasets.Features(
          {combined_column_name: [{"content": datasets.Value(dtype="string"), "role": datasets.Value(dtype="string")}]}
      )
      dataset = dataset.map(
          _input_pipeline_utils.combine_columns,
          fn_kwargs={"columns": data_column_names, "data_column": combined_column_name},
          remove_columns=data_column_names,
          features=dataset_features,
      )

    data_column_names = list(dataset.features.keys())
    dataset = dataset.map(
        _input_pipeline_utils.apply_chat_template,
        fn_kwargs={"tokenizer_model": tokenizer, "data_column_name": data_column_names[0]},
    )
  else:
    dataset = dataset.select_columns(data_column_names)

  if tokenizer.pad_token_id is not None:
    pad_id = tokenizer.pad_token_id
  elif tokenizer.unk_token_id is not None:
    pad_id = tokenizer.unk_token_id
  else:
    pad_id = -1

  if tokenize:
    dataset = dataset.map(
        _input_pipeline_utils.tokenization,
        batched=True,
        fn_kwargs={
            "hf_tokenizer": tokenizer,
            "truncation": not use_sft,
            "max_length": max_target_length,
            "column_names": data_column_names,
        },
    )

  dataset = _input_pipeline_utils.HFDataSource(
      dataset,
      dataloading_host_index,
      dataloading_host_count,
      num_threads,
      generate_padding_example,
      max_target_length,
      data_column_names,
  )
  operations = []
  if use_sft:
    operations.append(
        _input_pipeline_utils.SFTPromptMasking(
            text_column_name=data_column_names[0],
            completion_only=sft_train_on_completion_only,
            max_target_length=max_target_length,
            unk_id=pad_id,
        )
    )
    data_column_names = ("inputs", "targets")
  elif use_dpo:

    def lists2array(x):
      """Convert lists/tuples to array"""
      return jax.tree.map(np.asarray, x, is_leaf=lambda y: isinstance(y, (list, tuple)))

    operations.append(grain.MapOperation(lists2array))
  else:
    assert len(data_column_names) == 1
    operations.append(_input_pipeline_utils.HFNormalizeFeatures(data_column_names[0]))
    data_column_names = ("inputs", "targets")

  if packing and not use_dpo:
    length_struct = {col: max_target_length for col in data_column_names}
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=global_batch_size // jax.process_count(),
            length_struct=length_struct,
        )
    )
    operations.append(_input_pipeline_utils.ReformatPacking(data_column_names))
  else:
    operations.append(_input_pipeline_utils.PadToMaxLength(max_target_length, pad_id))
    operations.append(grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder))

  if shift and not use_dpo:
    operations.append(_input_pipeline_utils.ShiftData(ignored_ids=[pad_id, tokenizer.bos_token_id], axis=1))

  # Since HuggingFace IterableDataset does not support access through index
  # Indexes generated by dummy_index_sampler is not used.
  # dummy_index_sampler is used as an input place holder for grain.Dataloader
  dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=False,
      seed=0,
  )

  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=dummy_index_sampler,
      worker_count=grain_worker_count,  # only supports <=1 for now, more workers results in duplicated data
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=128),
  )

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen


def make_hf_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_train,
):
  """Load, preprocess dataset and return iterators"""
  train_ds = datasets.load_dataset(
      config.hf_path,
      data_dir=config.hf_data_dir,
      data_files=config.hf_train_files,
      split=config.train_split,
      streaming=True,
      token=config.hf_access_token,
  )
  train_iter = preprocessing_pipeline(
      dataloading_host_index=process_indices_train.index(jax.process_index()),
      dataloading_host_count=len(process_indices_train),
      global_mesh=global_mesh,
      dataset=train_ds,
      data_column_names=config.train_data_columns,
      tokenize=config.tokenize_train_data,
      tokenizer_path=config.tokenizer_path,
      hf_access_token=config.hf_access_token,
      global_batch_size=config.global_batch_size_to_load,
      max_target_length=config.max_target_length,
      shuffle=config.enable_data_shuffling,
      data_shuffle_seed=config.data_shuffle_seed,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
      packing=config.packing,
      generate_padding_example=False,
      use_dpo=config.use_dpo,
      use_sft=config.use_sft,
      sft_train_on_completion_only=config.sft_train_on_completion_only,
  )
  return train_iter


def make_hf_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_eval,
):
  """Make Hugging Face evaluation iterator. Load and preprocess eval dataset: and return iterator."""
  eval_ds = datasets.load_dataset(
      config.hf_path,
      data_dir=config.hf_data_dir,
      data_files=config.hf_eval_files,
      split=config.hf_eval_split,
      streaming=True,
      token=config.hf_access_token,
  )

  eval_generate_padding_example = config.eval_steps > 0
  eval_iter = preprocessing_pipeline(
      dataloading_host_index=process_indices_eval.index(jax.process_index()),
      dataloading_host_count=len(process_indices_eval),
      global_mesh=global_mesh,
      dataset=eval_ds,
      data_column_names=config.eval_data_columns,
      tokenize=config.tokenize_eval_data,
      tokenizer_path=config.tokenizer_path,
      hf_access_token=config.hf_access_token,
      global_batch_size=config.global_batch_size_to_load_eval,
      max_target_length=config.max_target_length,
      shuffle=False,
      data_shuffle_seed=config.data_shuffle_seed,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
      packing=config.packing,
      generate_padding_example=eval_generate_padding_example,
      use_dpo=config.use_dpo,
      use_sft=config.use_sft,
      sft_train_on_completion_only=config.sft_train_on_completion_only,
  )
  return eval_iter
