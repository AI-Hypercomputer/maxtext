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
from jax.sharding import Mesh

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


class TfdsPipelineBuilder:
  """Builder pattern implementation for TFDS data pipeline construction.

  This class wraps a tf.data.Dataset and provides a fluent API to apply
  various transformations like tokenization, batching, shuffling, etc.
  """

  def __init__(self, dataset: tf.data.Dataset, data_column_names: tuple[str, ...], use_dpo: bool = False):
    """Initializes the pipeline builder.

    Args:
      dataset: The source tf.data.Dataset.
      data_column_names: Names of the columns/features in the dataset.
      use_dpo: Whether Direct Preference Optimization (DPO) mode is enabled.
    """
    self._dataset = dataset
    self._data_column_names = data_column_names
    self._use_dpo = use_dpo

    # Apply initial normalization and column handling based on DPO setting.
    if not self._use_dpo:
      assert len(data_column_names) == 1
      self._dataset = self._dataset.map(
          lambda x: _input_pipeline_utils.normalize_features(x, data_column_names[0]),
          num_parallel_calls=AUTOTUNE,
      )
      # Standardizes to (inputs, targets) for non-DPO
      self._data_column_names = ("inputs", "targets")
    else:
      self._dataset = self._dataset.map(
          lambda x: {col: x[col] for col in data_column_names},
          num_parallel_calls=AUTOTUNE,
      )

  def with_tokenization(self, tokenizer_model: tokenizer.TokenizerType) -> "TfdsPipelineBuilder":
    """Applies tokenization to the dataset columns.

    Args:
      tokenizer_model: The tokenizer object to use for encoding.

    Returns:
      The builder instance (self).
    """
    self._dataset = self._dataset.map(
        lambda x: tokenizer.TokenizeOp(tokenizer=tokenizer_model, features=x, data_keys=self._data_column_names),
        num_parallel_calls=AUTOTUNE,
    )
    return self

  def with_truncation(self, max_target_length: int) -> "TfdsPipelineBuilder":
    """Applies truncation to the dataset fields to max allowable length.

    Args:
      max_target_length: The maximum length for the sequence.

    Returns:
      The builder instance (self).
    """
    if max_target_length > 0:
      # in pre-training we can take upto max_length+1 because there would be truncation by
      # 1 token for both inputs and targets
      extra_tokens = 1 if not self._use_dpo else 0
      self._dataset = self._dataset.map(
          lambda x: _input_pipeline_utils.truncate_to_max_allowable_length(x, max_target_length + extra_tokens),
          num_parallel_calls=AUTOTUNE,
      )
    return self

  def with_shuffling(self, shuffle_buffer_size: int, seed: int) -> "TfdsPipelineBuilder":
    """Applies shuffling to the dataset.

    Args:
      shuffle_buffer_size: Size of the shuffle buffer.
      seed: Random seed for shuffling.

    Returns:
      The builder instance (self).
    """
    self._dataset = self._dataset.shuffle(shuffle_buffer_size, seed=seed)
    return self

  def with_repeat(self, num_epochs: int | None) -> "TfdsPipelineBuilder":
    """Repeats the dataset.

    Args:
      num_epochs: Number of epochs to repeat. None means infinite.

    Returns:
      The builder instance (self).
    """
    self._dataset = self._dataset.repeat(num_epochs)
    return self

  def with_shift(self) -> "TfdsPipelineBuilder":
    """Applies data shifting for teacher-forced training.

    Only applies if not in DPO mode.

    Returns:
      The builder instance (self).
    """
    if not self._use_dpo:
      self._dataset = self._dataset.map(
          _input_pipeline_utils.shift_data_by_truncation,
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=True,
      )
    return self

  def with_batching(
      self,
      batch_size: int,
      pack_examples: bool,
      max_target_length: int,
      pad_id: int,
      drop_remainder: bool,
  ) -> "TfdsPipelineBuilder":
    """Applies batching to the dataset, optionally with packing.

    Args:
      batch_size: The batch size per process.
      pack_examples: Whether to use sequence packing.
      max_target_length: The target sequence length.
      pad_id: The padding text ID.
      drop_remainder: Whether to drop the last partial batch.

    Returns:
      The builder instance (self).
    """
    if pack_examples and not self._use_dpo:
      self._dataset = sequence_packing.pack_dataset(self._dataset, max_target_length, pad_id)
      self._dataset = self._dataset.batch(batch_size, drop_remainder=drop_remainder)
    else:
      # simple (static-shape) padded batching
      self._dataset = self._dataset.padded_batch(
          batch_size,
          padded_shapes={k: max_target_length for k in self._data_column_names},
          padding_values={k: pad_id for k in self._data_column_names},
          drop_remainder=drop_remainder,
      )
      self._dataset = self._dataset.map(
          lambda x: _input_pipeline_utils.add_segmentation_and_position(x, self._data_column_names, padding_token=pad_id),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=True,
      )
    return self

  def with_prefetch(self, prefetch_size: int) -> "TfdsPipelineBuilder":
    """Applies prefetching to the dataset.

    Args:
      prefetch_size: The number of elements to prefetch.

    Returns:
      The builder instance (self).
    """
    if prefetch_size:
      self._dataset = self._dataset.prefetch(prefetch_size)
    return self

  def build(self) -> tf.data.Dataset:
    """Returns the constructed dataset.

    Returns:
      The processed tf.data.Dataset.
    """
    return self._dataset


def get_datasets(
    dataset_name: str,
    data_split: str,
    shuffle_files: bool,
    shuffle_seed: int,
    dataloading_host_index: int,
    dataloading_host_count: int,
    dataset_path: str | None = None,
) -> tf.data.Dataset:
  """Loads and shards a TFDS dataset.

  Args:
    dataset_name: Name of the dataset in TFDS.
    data_split: Split of the dataset to load (e.g., 'train', 'validation').
    shuffle_files: Whether to shuffle file reading order.
    shuffle_seed: Seed for file shuffling.
    dataloading_host_index: Index of the current host among data loaders.
    dataloading_host_count: Total number of data loading hosts.
    dataset_path: Optional path to the dataset directory.

  Returns:
    A sharded tf.data.Dataset.
  """
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
    dataset: tf.data.Dataset,
    tokenizer_path: str,
    tokenizer_type: str,
    global_batch_size: int,
    max_target_length: int,
    data_column_names: tuple[str, ...],
    shuffle: bool = False,
    data_shuffle_seed: int = 0,
    tokenize: bool = True,
    add_bos: bool = True,
    add_eos: bool = True,
    num_epochs: None | int = 1,
    pack_examples: bool = True,
    shuffle_buffer_size: int = 1024,
    shift: bool = True,
    drop_remainder: bool = True,
    prefetch_size: int = tf.data.experimental.AUTOTUNE,
    use_dpo: bool = False,
    hf_access_token: str = "",
) -> tf.data.Dataset:
  """Applies the preprocessing pipeline to a TFDS dataset using a builder pattern.

  Args:
     dataset: The source tf.data.Dataset.
     tokenizer_path: Path to the tokenizer file.
     tokenizer_type: The type of tokenizer (e.g., 'sentencepiece', 'huggingface').
     global_batch_size: The global batch size across all devices.
     max_target_length: The maximum token length for sequences.
     data_column_names: Names of the data columns to process.
     shuffle: Whether to shuffle the dataset.
     data_shuffle_seed: Seed used for shuffling.
     tokenize: Whether to apply tokenization.
     add_bos: Whether to prepend a BOS token.
     add_eos: Whether to append an EOS token.
     num_epochs: Number of epochs to repeat the dataset.
     pack_examples: Whether to pack multiple examples into a single sequence (pre-training only).
     shuffle_buffer_size: Size of the buffer for shuffling.
     shift: Whether to shift inputs vs targets (for next-token prediction).
     drop_remainder: Whether to drop the last batch if it is incomplete.
     prefetch_size: Number of batches to prefetch.
     use_dpo: Whether to process data for Direct Preference Optimization (DPO).
     hf_access_token: Hugging Face access token for loading specific tokenizers.

  Returns:
     A processed tf.data.Dataset ready for training or evaluation.
  """
  builder = TfdsPipelineBuilder(dataset, data_column_names, use_dpo)

  tokenizer_model = _input_pipeline_utils.get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1

  if tokenize:
    builder.with_tokenization(tokenizer_model)

  builder.with_truncation(max_target_length)

  if shuffle:
    builder.with_shuffling(shuffle_buffer_size, data_shuffle_seed)

  builder.with_repeat(num_epochs)

  if shift:
    builder.with_shift()

  # Calculate batch size per process as TF input pipeline runs locally on each host
  global_per_process_batch_size = global_batch_size // jax.process_count()
  builder.with_batching(global_per_process_batch_size, pack_examples, max_target_length, pad_id, drop_remainder)

  builder.with_prefetch(prefetch_size)

  return builder.build()


def make_tfds_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh: Mesh,
    process_indices_train: list[int],
):
  """Loads the train dataset and creating the preprocessing iterator.

  Args:
    config: The configuration Dict.
    global_mesh: JAX Mesh for sharding (used to check batch size compatibility).
    process_indices_train: List of process indices participating in training loading.

  Returns:
     A MultiHostDataLoadIterator for training.
  """
  assert (
      config.global_batch_size_to_load % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."

  get_datasets_kwargs = {
      "dataset_name": config.dataset_name,
      "dataset_path": config.dataset_path,
      "data_split": config.train_split,
      "shuffle_files": config.enable_data_shuffling,
      "shuffle_seed": config.data_shuffle_seed,
  }
  if not config.colocated_python_data_input:
    train_ds = get_datasets(
        dataloading_host_index=process_indices_train.index(jax.process_index()),
        dataloading_host_count=len(process_indices_train),
        **get_datasets_kwargs,
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
        **get_datasets_kwargs,
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
    global_mesh: Mesh,
    process_indices_eval: list[int],
):
  """Loads the eval dataset and creating the preprocessing iterator.

  Args:
    config: The configuration Dict.
    global_mesh: JAX Mesh for sharding.
    process_indices_eval: List of process indices participating in eval loading.

  Returns:
     A MultiHostDataLoadIterator for evaluation.
  """
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
    return multihost_dataloading.MultiHostDataLoadIterator(
        eval_dataloader, global_mesh, config.generate_padding_batch_eval
    )
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
