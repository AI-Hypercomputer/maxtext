# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline for a LM1B dataset."""

import warnings
import functools

import ml_collections

import tensorflow as tf
import tensorflow_datasets as tfds

import jax

from MaxText import multihost_dataloading
from MaxText import tokenizer
from MaxText import sequence_packing
from MaxText.input_pipeline import _input_pipeline_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE

# reserve GPU memory for JAX only if tensorflow is built with GPU support
try:
  tf.config.experimental.set_visible_devices([], "GPU")
except tf.errors.NotFoundError:
  pass


def get_datasets(
    dataset_name,
    data_split,
    shuffle_files,
    shuffle_seed,
    dataloading_host_index,
    dataloading_host_count,
    dataset_path=None,
):
  """Load a TFDS dataset."""
  ds_builder = tfds.builder(dataset_name, data_dir=dataset_path)

  if shuffle_files:
    read_config = tfds.ReadConfig(shuffle_seed=shuffle_seed)
  else:
    read_config = tfds.ReadConfig()

  if ds_builder.info.splits[data_split].num_shards >= dataloading_host_count:
    read_config.input_context = tf.distribute.InputContext(
        input_pipeline_id=dataloading_host_index,
        num_input_pipelines=dataloading_host_count,
    )
    ds = ds_builder.as_dataset(split=data_split, read_config=read_config, shuffle_files=shuffle_files)
  else:
    warnings.warn(
        f"WARNING: Inefficient dataloading. Your {dataset_name} contains {ds_builder.info.splits[data_split].num_shards}"
        f"shards, smaller than {dataloading_host_count=}. This is known to lead to inefficient dataloading."
        "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md"
        "#multihost-dataloading-best-practice"
    )
    ds = ds_builder.as_dataset(split=data_split, read_config=read_config, shuffle_files=shuffle_files)
    ds = ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)

  return ds


def preprocessing_pipeline(
    dataset,
    tokenizer_path,
    tokenizer_type: str,
    global_batch_size: int,
    max_target_length: int,
    data_column_names,
    shuffle: bool = False,
    data_shuffle_seed=0,
    tokenize: bool = True,
    add_bos: bool = True,
    add_eos: bool = True,
    num_epochs: None | int = 1,
    pack_examples: bool = True,
    shuffle_buffer_size: int = 1024,
    shift: bool = True,
    drop_remainder: bool = True,
    prefetch_size=tf.data.experimental.AUTOTUNE,
    use_dpo: bool = False,
    hf_access_token: str = "",
):
  """pipeline for preprocessing TFDS dataset."""
  if not use_dpo:
    assert len(data_column_names) == 1
    dataset = dataset.map(
        lambda x: _input_pipeline_utils.normalize_features(x, data_column_names[0]), num_parallel_calls=AUTOTUNE
    )
  else:
    dataset = dataset.map(lambda x: {col: x[col] for col in data_column_names}, num_parallel_calls=AUTOTUNE)

  data_column_names = data_column_names if use_dpo else ("inputs", "targets")

  tokenizer_model = _input_pipeline_utils.get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1

  if tokenize:
    dataset = dataset.map(
        lambda x: tokenizer.TokenizeOp(tokenizer=tokenizer_model, features=x, data_keys=data_column_names),
        num_parallel_calls=AUTOTUNE,
    )

  if max_target_length > 0:
    # in pre-training we can take upto max_length+1 because there would be truncation by
    # 1 token for both inputs and targets
    extra_tokens = 1 if not use_dpo else 0
    dataset = dataset.map(
        lambda x: _input_pipeline_utils.truncate_to_max_allowable_length(x, max_target_length + extra_tokens),
        num_parallel_calls=AUTOTUNE,
    )

  # Shuffle and repeat.
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)

  dataset = dataset.repeat(num_epochs)

  # Shift inputs for teacher-forced training
  if shift and not use_dpo:
    dataset = dataset.map(
        _input_pipeline_utils.shift_data_by_truncation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )

  # Perform greedy sequence packing and batching
  if pack_examples and not use_dpo:
    dataset = sequence_packing.pack_dataset(dataset, max_target_length, pad_id)
    dataset = dataset.batch(global_batch_size // jax.process_count(), drop_remainder=drop_remainder)
  else:
    # simple (static-shape) padded batching
    dataset = dataset.padded_batch(
        global_batch_size // jax.process_count(),
        padded_shapes={k: max_target_length for k in data_column_names},
        padding_values={k: pad_id for k in data_column_names},
        drop_remainder=drop_remainder,
    )
    dataset = dataset.map(
        lambda x: _input_pipeline_utils.add_segmentation_and_position(x, data_column_names, padding_token=pad_id),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def make_tfds_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_train,
):
  """load dataset, preprocess and return iterators"""
  assert (
      config.global_batch_size_to_load % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."
  if not config.colocated_python_data_input:
    train_ds = get_datasets(
        dataset_name=config.dataset_name,
        data_split=config.train_split,
        shuffle_files=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed,
        dataloading_host_index=process_indices_train.index(jax.process_index()),
        dataloading_host_count=len(process_indices_train),
    )
    train_dataloader = preprocessing_pipeline(
        dataset=train_ds,
        tokenizer_path=config.tokenizer_path,
        tokenizer_type=config.tokenizer_type,
        global_batch_size=config.global_batch_size_to_load,
        max_target_length=config.max_target_length,
        data_column_names=config.train_data_columns,
        shuffle=config.enable_data_shuffling,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_train_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        num_epochs=config.num_epoch,
        pack_examples=config.packing,
        use_dpo=config.use_dpo,
        hf_access_token=config.hf_access_token,
    )
    return multihost_dataloading.MultiHostDataLoadIterator(
        train_dataloader, global_mesh, config.generate_padding_batch_train
    )
  else:
    get_ds_fn = functools.partial(
        get_datasets,
        dataset_name=config.dataset_name,
        dataset_path=config.dataset_path,
        data_split=config.train_split,
        shuffle_files=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed,
    )
    preprocessing_fn = functools.partial(
        preprocessing_pipeline,
        tokenizer_path=config.tokenizer_path,
        tokenizer_type=config.tokenizer_type,
        global_batch_size=config.global_batch_size_to_load,
        max_target_length=config.max_target_length,
        data_column_names=config.train_data_columns,
        shuffle=config.enable_data_shuffling,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_train_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        num_epochs=config.num_epoch,
        pack_examples=config.packing,
        use_dpo=config.use_dpo,
        hf_access_token=config.hf_access_token,
    )
    global_shape = (config.global_batch_size_to_load, config.max_target_length)
    return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, global_mesh, global_shape)


def make_tfds_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_eval,
):
  """load eval dataset, preprocess and return iterators"""
  assert (
      config.global_batch_size_to_load_eval % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."
  if not config.colocated_python_data_input:
    eval_ds = get_datasets(
        dataset_name=config.eval_dataset_name,
        data_split=config.eval_split,
        shuffle_files=False,
        shuffle_seed=config.data_shuffle_seed,
        dataloading_host_index=process_indices_eval.index(jax.process_index()),
        dataloading_host_count=len(process_indices_eval),
    )
    eval_dataloader = preprocessing_pipeline(
        dataset=eval_ds,
        tokenizer_path=config.tokenizer_path,
        tokenizer_type=config.tokenizer_type,
        global_batch_size=config.global_batch_size_to_load_eval,
        max_target_length=config.max_target_length,
        data_column_names=config.eval_data_columns,
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_eval_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        pack_examples=config.packing,
        use_dpo=config.use_dpo,
        hf_access_token=config.hf_access_token,
    )
    return multihost_dataloading.MultiHostDataLoadIterator(eval_dataloader, global_mesh, config.generate_padding_batch_eval)
  else:
    get_ds_fn = functools.partial(
        get_datasets,
        dataset_name=config.eval_dataset_name,
        data_split=config.eval_split,
        shuffle_files=False,
        shuffle_seed=config.data_shuffle_seed,
    )
    preprocessing_fn = functools.partial(
        preprocessing_pipeline,
        tokenizer_path=config.tokenizer_path,
        tokenizer_type=config.tokenizer_type,
        global_batch_size=config.global_batch_size_to_load_eval,
        max_target_length=config.max_target_length,
        data_column_names=config.eval_data_columns,
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_eval_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        pack_examples=config.packing,
        use_dpo=config.use_dpo,
        hf_access_token=config.hf_access_token,
    )
    return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, config, global_mesh)
