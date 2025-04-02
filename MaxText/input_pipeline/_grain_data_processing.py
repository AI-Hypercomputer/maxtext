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

import functools
import ml_collections
import jax
import transformers
import grain.python as grain
from sentencepiece import SentencePieceProcessor
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
    max_target_length: int,
    grain_worker_count: int,
    dataloading_host_index,
    dataloading_host_count,
    data_columns,
    tokenizer_type: str = "sentencepiece",
    hf_access_token=None,
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
  operations = []
  operations.append(_input_pipeline_utils.ParseFeatures(data_columns, tokenize))
  if not use_dpo:
    assert len(data_columns) == 1
    operations.append(_input_pipeline_utils.InputsTargetsFeatures(data_columns[0]))
    data_columns = ("inputs", "targets")
  operations.append(_input_pipeline_utils.NormalizeFeatures(data_columns, tokenize))

  if tokenizer_type == "sentencepiece":
    tokenizer_model = SentencePieceProcessor()
    tokenizer_model.Load(tokenizer_path)
    if tokenizer_model.pad_id() is not None:
      pad_id = tokenizer_model.pad_id()
    elif tokenizer_model.unk_id() is not None:
      pad_id = tokenizer_model.unk_id()
    else:
      pad_id = -1
  else:
    tokenizer_model = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        legacy=False,
        token=hf_access_token,
    )
    if tokenizer_model.pad_token_id is not None:
      pad_id = tokenizer_model.pad_token_id
    elif tokenizer_model.unk_token_id is not None:
      pad_id = tokenizer_model.unk_token_id
    else:
      pad_id = -1

  if tokenize:
    operations.append(_grain_tokenizer.TokenizeAndTrim(data_columns, max_target_length, add_bos, add_eos, tokenizer_model))

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
    operations.append(_input_pipeline_utils.PadToMaxLength(max_target_length, pad_id))
    operations.append(grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder))

  # Shift inputs for teacher-forced training
  if shift and not use_dpo:
    operations.append(
        _input_pipeline_utils.ShiftData(
            ignored_ids=[
                pad_id,
                tokenizer_model.bos_id()
                if isinstance(tokenizer_model, SentencePieceProcessor)
                else tokenizer_model.bos_token_id,
            ],
            axis=1,
        )
    )

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

  return dataloader


def make_grain_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  assert config.global_batch_size_to_load % global_mesh.size == 0, "Batch size should be divisible number of global devices."
  if not config.colocated_python_data_input:
    train_ds = get_datasets(config.grain_train_files)
    train_dataloader = preprocessing_pipeline(
        dataset=train_ds,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=config.global_batch_size_to_load,
        max_target_length=config.max_target_length,
        grain_worker_count=config.grain_worker_count,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        data_columns=config.train_data_columns,
        tokenizer_type=config.tokenizer_type,
        hf_access_token=config.hf_access_token,
        shuffle=config.enable_data_shuffling,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_train_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        use_dpo=config.use_dpo,
    )
    return multihost_dataloading.MultiHostDataLoadIterator(train_dataloader, global_mesh)
  else:
    get_ds_fn = functools.partial(get_datasets, config.grain_train_files)
    preprocessing_fn = functools.partial(
        preprocessing_pipeline,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=config.global_batch_size_to_load,
        max_target_length=config.max_target_length,
        grain_worker_count=config.grain_worker_count,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        data_columns=config.train_data_columns,
        tokenizer_type=config.tokenizer_type,
        hf_access_token=config.hf_access_token,
        shuffle=config.enable_data_shuffling,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_train_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        use_dpo=config.use_dpo,
    )
    return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, config, global_mesh)


def make_grain_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  assert (
      config.global_batch_size_to_load_eval % global_mesh.size == 0
  ), "Batch size should be divisible number of global devices."
  if config.colocated_python_data_input:
    eval_ds = get_datasets(config.grain_eval_files)
    eval_dataloader = preprocessing_pipeline(
        dataset=eval_ds,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=config.global_batch_size_to_load_eval,
        max_target_length=config.max_target_length,
        grain_worker_count=config.grain_worker_count,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        data_columns=config.eval_data_columns,
        tokenizer_type=config.tokenizer_type,
        hf_access_token=config.hf_access_token,
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_eval_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        use_dpo=config.use_dpo,
    )
    return multihost_dataloading.MultiHostDataLoadIterator(eval_dataloader, global_mesh)
  else:
    get_ds_fn = functools.partial(get_datasets, config.grain_eval_files)
    preprocessing_fn = functools.partial(
        preprocessing_pipeline,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=config.global_batch_size_to_load_eval,
        max_target_length=config.max_target_length,
        grain_worker_count=config.grain_worker_count,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        data_columns=config.eval_data_columns,
        tokenizer_type=config.tokenizer_type,
        hf_access_token=config.hf_access_token,
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_eval_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        use_dpo=config.use_dpo,
    )
    return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, config, global_mesh)
