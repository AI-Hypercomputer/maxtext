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

"""DEPRECATED: Input pipeline for c4 mlperf dataset."""

from collections.abc import Sequence
import functools

import numpy as np

import ml_collections

try:
  import tensorflow as tf
  import tensorflow_datasets as tfds
except ImportError as error:
  raise ImportError(
      "TensorFlow and tensorflow-datasets are required. Run `pip install tensorflow tensorflow-datasets`"
  ) from error


import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P

from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline.packing import sequence_packing
from maxtext.input_pipeline.input_pipeline_utils import get_tokenizer
from maxtext.input_pipeline.input_pipeline_utils import TokenizeOp
from maxtext.utils import max_logging
from maxtext.utils.sharding import remove_size_one_mesh_axis

AUTOTUNE = tf.data.experimental.AUTOTUNE


# data processing functions:
#   _shift_left_and_pad, rekey, reduce_concat_tokens and split_tokens_to_targets_length
# Adapted from:
#   https://github.com/google-research/text-to-text-transfer-transformer/blob/ba171b6/t5/data/preprocessors.py
# -----------------------------------------------------------------------------
def _shift_left_and_pad(tensor, pad_val):
  """Shift the input to the left with pad_val"""
  # Expand dims here so that the below code can work with 1-d tensors.
  v = tf.expand_dims(tensor, 0)
  # Make sure we keep tensor as ragged to allow for uneven concat.
  if isinstance(v, tf.Tensor):
    v = tf.RaggedTensor.from_tensor(v)

  # Append padding to the last item of every sequence.
  pad_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
  pad_tensor = tf.broadcast_to(pad_val, pad_shape)
  last_in_sequence = tf.concat([v[..., -1:, 1:], pad_tensor], axis=-1)
  # Concat back the newly modified final sequence item.
  v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)
  # Un-expand outer dimension.
  v = v[0]
  return v


def rekey(ds, key_map=None):
  """normalization with key mapping"""

  def _rekey(x, key_map=None):
    """Replace the feature keys according to the mapping in `key_map`.
    For example, if the dataset returns examples of the format:
    {'foo': 'something', 'bar': 'something else', 'zoo': 'others'}
    and key_map = {'boo': 'foo', 'spar': 'bar', 'zoo': None} then this function will return
    examples with the format
    {'boo': 'something', 'spar': 'something else'}
    If a mapping is to None, then the key will be dropped.
    Args:
      x: an example to process.
      key_map: dictionary mapping new keys to original keys
    Returns:
      A preprocessed example with the format listed above.
    """
    if key_map:
      return {new_key: x[old_key] for new_key, old_key in key_map.items() if old_key}
    return x

  return ds.map(functools.partial(_rekey, key_map=key_map), num_parallel_calls=AUTOTUNE)


def reduce_concat_tokens(
    dataset,
    feature_key="targets",
    batch_size=128,
):
  """Token-preprocessor to concatenate multiple unrelated documents.
  If we want to generate examples of exactly the right length,
  (to avoid wasting space on padding), then we use this function, followed by
  split_tokens.
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    feature_key: an string
    batch_size: an integer - how many documents to concatenate into one
  Returns:
    a dataset
  """
  dataset = dataset.map(lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})

  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    # strip padding
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


def split_tokens(
    dataset,
    max_tokens_per_segment=128,
    feature_key="targets",
):
  """Split examples into multiple examples each.
  The intended use case is to break up long examples for use in unsupervised
  transfer-learning.
  This function is generally preceded by select_random_chunk.
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    max_tokens_per_segment: an integer, the maximum number of tokens in each
      segment. Only the final segment may be shorter.
    feature_key: a string, the feature to split
  Returns:
    a dataset
  """

  def _split_tokens(x):
    """Split one token sequence into multiple multiple."""
    tokens = x[feature_key]
    n_tokens = tf.size(tokens)
    length = max_tokens_per_segment

    # Pad to a multiple of length, then use tf.reshape to split up the tokens
    # into num_segments segments each of the given length.
    num_segments = tf.cast(tf.math.ceil(tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)), tf.int32)
    padding = num_segments * length - tf.size(tokens)
    tokens = tf.pad(tokens, [[0, padding]])
    return tf.reshape(tokens, [-1, length])

  def _strip_padding(x):
    return {feature_key: tf.boolean_mask(x, tf.cast(x, tf.bool))}

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  dataset = dataset.map(_split_tokens, num_parallel_calls=AUTOTUNE)
  dataset = dataset.unbatch()
  return dataset.map(_strip_padding, num_parallel_calls=AUTOTUNE)


def split_tokens_to_targets_length(dataset, sequence_length):
  return split_tokens(dataset, max_tokens_per_segment=sequence_length)


def _pad_to_batch_size(
    ds: tf.data.Dataset,
    batch_size: int,
    num_examples: int = 0,
    num_batches: int = 0,
    dataloading_host_count: int = 0,
) -> tf.data.Dataset:
  """Pad unevenly distributed eval data in each shard with new entries to multiples of batch size.

  Args:
    ds: the unbatched eval dataset of the current host.
    batch_size: the number of real examples in each local batch.
    num_examples: the number of examples in `ds`, when it is already known. Avoids a
      full scan of the dataset.
    num_batches: the number of batches every data loading host has to produce. When
      positive it is used as is, which keeps the hosts in sync without communication.
    dataloading_host_count: the number of JAX processes loading real data. Defaults to
      `jax.process_count()`.

  Returns:
    The dataset padded with entries whose `targets_segmentation` is 0.
  """

  # local_num represents the total number of examples in eval dataset,
  if num_examples > 0:
    local_num = num_examples
  else:

    def _get_num_examples(ds: tf.data.Dataset) -> int:
      # Iterate one-by-one instead of len(list(...)) to reduce peak memory.
      num_examples = 0
      for _ in ds:
        num_examples += 1

      return num_examples

    local_num = _get_num_examples(ds)
  local_num_batches = (local_num + batch_size - 1) // batch_size
  if dataloading_host_count <= 0:
    dataloading_host_count = jax.process_count()
  if num_batches <= 0:
    if dataloading_host_count < jax.process_count():
      # multihost_utils.process_allgather is a collective over *all* processes, but
      # only the data loading hosts run this code, so calling it would hang. Every
      # loading host reads an equally sized shard, so use the local batch count.
      max_logging.log(
          f"Only {dataloading_host_count} of {jax.process_count()} processes load eval data, "
          "skipping the cross-host batch count allgather."
      )
      num_batches = local_num_batches
    else:
      # Find the max number of batches required across all Jax processes.
      num_batches_all = multihost_utils.process_allgather(jnp.array([local_num_batches]), tiled=False)
      num_batches = int(np.max(num_batches_all))

  pad_num = num_batches * batch_size - local_num
  assert pad_num >= 0
  max_logging.log(
      f"Eval data has {local_num} local entries, padding now with "
      f"{pad_num} extra entries to get {num_batches} batches."
  )

  # Repeat a random example to make the last batch full.
  def _add_pad(x):
    x["targets_segmentation"] *= 0
    return x

  pad_ds = ds.take(1).map(_add_pad).repeat(pad_num)
  return ds.concatenate(pad_ds)


def get_local_rows_loading_real_data(
    data_sharding, global_batch_size_to_load, global_batch_size_to_train_on, max_target_length, mesh
):
  """Get the rows of this host's batch that are not decimated away by the train/eval step.

  `MultiHostDataLoadIterator` splits the array loaded by a host into
  `len(mesh.local_devices)` equally sized row blocks and hands block `i` to
  `mesh.local_devices[i]`. When `global_batch_size_to_train_on` is smaller than
  `global_batch_size_to_load`, the train/eval step keeps only the first
  `global_batch_size_to_train_on` rows of the global batch (see the decimation in
  `loss_fn`), so only the local rows placed on a device that holds one of those global
  rows are actually used. This returns the sorted indices of those local rows, which is
  the same mechanism `input_pipeline_interface.get_process_loading_real_data` uses to
  select the loading hosts.
  """
  local_batch_size = global_batch_size_to_load // jax.process_count()
  num_local_devices = len(mesh.local_devices)
  if num_local_devices == 0 or local_batch_size % num_local_devices != 0:
    # The loader cannot split the local batch evenly over the local devices either,
    # so there is no row-to-device mapping to reason about.
    return list(range(local_batch_size))
  rows_per_device = local_batch_size // num_local_devices

  data_sharding_pspec = remove_size_one_mesh_axis(P(*data_sharding), mesh)
  sharding = jax.sharding.NamedSharding(mesh, data_sharding_pspec)
  devices_indices_map = sharding.devices_indices_map((global_batch_size_to_load, max_target_length))
  batch_cutoff = global_batch_size_to_train_on
  local_rows = []
  for device_position, device in enumerate(mesh.local_devices):
    indices = devices_indices_map[device]
    if not indices[0].stop or indices[0].stop <= batch_cutoff:
      local_rows.extend(range(device_position * rows_per_device, (device_position + 1) * rows_per_device))
  return local_rows


def _place_real_rows_in_local_batch(batch, local_batch_size: int, real_data_rows: Sequence[int]):
  """Scatter the real examples of a batch into the local rows that are evaluated.

  When `global_batch_size_to_eval_on` is smaller than `global_batch_size_to_load_eval`
  the eval step keeps only the first `global_batch_size_to_eval_on` rows of the global
  batch. Only some of the rows this host loads end up in that prefix, so the real
  examples are placed in those rows and the remaining rows are filled with dummy
  examples (all zeros, hence `targets_segmentation == 0`) which carry no loss weight.
  """
  rows = tf.constant(list(real_data_rows), dtype=tf.int32)
  output = {}
  for key, value in batch.items():
    # The final batch may hold fewer examples than `len(real_data_rows)`; the
    # remaining real rows then stay dummy rows.
    indices = rows[: tf.shape(value)[0], tf.newaxis]
    shape = tf.concat([[local_batch_size], tf.shape(value)[1:]], axis=0)
    scattered = tf.scatter_nd(indices, value, shape)
    output[key] = tf.ensure_shape(scattered, [local_batch_size] + value.shape.as_list()[1:])
  return output


def get_dataset(
    dataset_name: str,
    split: str,
    dataloading_host_index: int,
    dataloading_host_count: int,
    data_dir: str | None = None,
    enable_data_shuffling: bool = False,
    data_shuffle_seed: int = 0,
    shard_in_read: bool = False,
) -> tf.data.Dataset:
  """Load and return a dataset of examples."""
  if shard_in_read:
    # shard dataset in reading
    read_config = tfds.ReadConfig(
        shuffle_seed=data_shuffle_seed,
        input_context=tf.distribute.InputContext(
            input_pipeline_id=dataloading_host_index,
            num_input_pipelines=dataloading_host_count,
        ),
    )
    ds_builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset(split=split, read_config=read_config, shuffle_files=enable_data_shuffling)
  else:
    # shard dataset after reading
    read_config = tfds.ReadConfig(shuffle_seed=data_shuffle_seed)
    ds_builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = ds_builder.as_dataset(split=split, read_config=read_config, shuffle_files=enable_data_shuffling)
    ds = ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)
  return ds


def _files_hold_one_example_each(
    dataset_name: str, split: str, data_dir: str | None, dataloading_host_count: int
) -> bool:
  """Whether sharding `split` by file gives every host the same examples as sharding by example.

  With `shard_in_read=True` host i opens only files i, i + n, i + 2n, ... (n loading hosts),
  while `ds.shard` after reading makes host i read files 0..i to find its first example.
  When every file holds exactly one example the two select the same examples in the same
  order, so the eval batches are unchanged. Only the dataset metadata is read here.
  """
  file_instructions = tfds.builder(dataset_name, data_dir=data_dir).info.splits[split].file_instructions
  return len(file_instructions) >= dataloading_host_count and all(f.skip == 0 and f.take == 1 for f in file_instructions)


def format_fn(x, eos_id: int = 1, pad_id: int = 0):
  """Format function for c4_mlperf."""
  x["inputs"] = x["targets"]
  x["inputs_position"] = x["targets_position"]
  x["targets"] = _shift_left_and_pad(x["targets"], eos_id)
  x["inputs_segmentation"] = tf.where(
      tf.logical_and(x["targets"] != eos_id, x["targets"] != pad_id), x["targets_segmentation"], 0
  )
  x["targets_segmentation"] = x["inputs_segmentation"]
  return x


def chunk_token_stream(dataset, feature_key="targets", sequence_length=4096, eod_id: int | None = None):
  """Flattens a dataset of token sequences across document boundaries and chunks them.

  Appends the delimiter eod_id token to each document before unbatching so that document
  boundaries are clearly delineated in the continuous stream, preserving all valid token IDs
  (including 0) and guaranteeing 0% padding waste.
  """
  if eod_id is not None:
    eod_tensor = tf.constant([eod_id], dtype=tf.int32)
    ds = dataset.map(
        lambda x: tf.concat([tf.cast(x[feature_key], tf.int32), eod_tensor], axis=0),
        num_parallel_calls=AUTOTUNE,
    )
  else:
    ds = dataset.map(lambda x: tf.cast(x[feature_key], tf.int32), num_parallel_calls=AUTOTUNE)

  ds = ds.unbatch()
  ds = ds.batch(sequence_length, drop_remainder=True)
  return ds.map(lambda tokens: {feature_key: tokens}, num_parallel_calls=AUTOTUNE)


def format_continuous_stream_fn(x, max_target_length: int, eod_id: int = 1):
  """Format function for continuous token stream chunks.

  Sets monotonic position IDs 0..max_target_length-1, uniform 1s for segmentation
  to allow standard causal cross-document attention, shifts targets left by 1 with
  eod_id appended at the chunk boundary, and ensures 100% loss participation.
  """
  targets_raw = tf.cast(x["targets"], tf.int32)
  inputs = targets_raw
  targets = tf.concat([targets_raw[1:], [eod_id]], axis=0)

  inputs_position = tf.range(max_target_length, dtype=tf.int32)
  targets_position = inputs_position

  inputs_segmentation = tf.ones([max_target_length], dtype=tf.int32)
  targets_segmentation = tf.ones([max_target_length], dtype=tf.int32)

  return {
      "inputs": inputs,
      "targets": targets,
      "inputs_position": inputs_position,
      "targets_position": targets_position,
      "inputs_segmentation": inputs_segmentation,
      "targets_segmentation": targets_segmentation,
  }


def preprocess_train_dataset(
    train_ds: tf.data.Dataset,
    sp_tokenizer,
    train_global_batch_size_to_load: int,
    max_target_length: int,
    shuffle_buffer_size: int,
    data_shuffle_seed: int,
    is_tokenized_dataset: bool = False,
    use_stream_chunking: bool | None = None,
) -> tf.data.Dataset:
  """Preprocess the training dataset."""
  pad_id = sp_tokenizer.pad_id
  eod_id = sp_tokenizer.eos_id

  if use_stream_chunking is None:
    use_stream_chunking = is_tokenized_dataset

  if not is_tokenized_dataset:
    train_ds = train_ds.map(
        lambda x: TokenizeOp(tokenizer_model=sp_tokenizer, features=x, data_keys=("targets",)),
        num_parallel_calls=AUTOTUNE,
    )

  if use_stream_chunking:
    # Continuous stream chunking:
    # 1. Shuffle documents before chunking to maximize global token diversity across contiguous windows.
    # 2. Append eod_id to each document and flatten token streams into contiguous max_target_length chunks (0% padding).
    # 3. Shift targets left by 1 and append the eod_id token as the final target of each chunk.
    # 4. Set monotonic position IDs: [0, 1, ..., max_target_length - 1].
    # 5. Uniform 1s for segmentation to allow cross-document causal attention and 100% loss participation.
    train_ds = train_ds.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)
    train_ds = chunk_token_stream(train_ds, feature_key="targets", sequence_length=max_target_length, eod_id=eod_id)
    train_ds = train_ds.map(
        lambda x: format_continuous_stream_fn(x, max_target_length=max_target_length, eod_id=eod_id),
        num_parallel_calls=AUTOTUNE,
    )
  else:
    train_ds = reduce_concat_tokens(train_ds, feature_key="targets", batch_size=4096)
    train_ds = split_tokens_to_targets_length(train_ds, max_target_length)
    train_ds = train_ds.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)
    train_ds = sequence_packing.pack_dataset(train_ds, max_target_length, pad_id=pad_id)
    train_ds = train_ds.map(lambda x: format_fn(x, pad_id=pad_id), num_parallel_calls=AUTOTUNE)

  train_ds = train_ds.batch(train_global_batch_size_to_load // jax.process_count(), drop_remainder=True)
  train_ds = train_ds.prefetch(AUTOTUNE)
  return train_ds


def preprocess_eval_dataset(
    eval_ds: tf.data.Dataset,
    sp_tokenizer,
    eval_global_batch_size_to_load: int,
    max_target_length: int,
    num_examples: int = 0,
    is_tokenized_dataset: bool = True,
    use_stream_chunking: bool | None = None,
    real_data_rows: Sequence[int] | None = None,
    num_batches: int = 0,
    dataloading_host_count: int = 0,
) -> tf.data.Dataset:
  """Preprocess the evaluation dataset.

  Args:
    eval_ds: the raw eval dataset shard of the current host.
    sp_tokenizer: tokenizer, used for its pad and eos ids.
    eval_global_batch_size_to_load: cluster-wide number of rows loaded per eval step.
    max_target_length: sequence length.
    num_examples: number of examples in `eval_ds`, when already known.
    is_tokenized_dataset: whether `eval_ds` already holds token ids.
    use_stream_chunking: whether to chunk a continuous token stream, defaults to
      `is_tokenized_dataset`.
    real_data_rows: rows of the per-host batch that are not decimated away by the eval
      step. Defaults to every row. The other rows are filled with dummy examples.
    num_batches: number of batches every data loading host has to produce.
    dataloading_host_count: number of JAX processes loading real data.

  Returns:
    The batched eval dataset, with one batch of `eval_global_batch_size_to_load //
    jax.process_count()` rows per eval step.
  """
  pad_id = sp_tokenizer.pad_id
  eod_id = sp_tokenizer.eos_id

  if use_stream_chunking is None:
    use_stream_chunking = is_tokenized_dataset

  if not is_tokenized_dataset:
    eval_ds = eval_ds.map(
        lambda x: TokenizeOp(tokenizer_model=sp_tokenizer, features=x, data_keys=("targets",)),
        num_parallel_calls=AUTOTUNE,
    )

  if use_stream_chunking:
    # Continuous stream chunking:
    # 1. Append eod_id to each document and flatten token streams into contiguous max_target_length chunks (0% padding).
    # 2. Shift targets left by 1 and append the eod_id token as the final target of each chunk.
    # 3. Set monotonic position IDs: [0, 1, ..., max_target_length - 1].
    # 4. Uniform 1s for segmentation to allow cross-document causal attention and 100% loss participation.
    eval_ds = chunk_token_stream(eval_ds, feature_key="targets", sequence_length=max_target_length, eod_id=eod_id)
    eval_ds = eval_ds.map(
        lambda x: format_continuous_stream_fn(x, max_target_length=max_target_length, eod_id=eod_id),
        num_parallel_calls=AUTOTUNE,
    )
  else:
    # hardcode batch_sizes 24567 i.e. the exp size in split validation_24567exp
    #   to avoid padding tokens inserted in group text
    eval_ds = reduce_concat_tokens(eval_ds, feature_key="targets", batch_size=24567)
    eval_ds = split_tokens_to_targets_length(eval_ds, max_target_length)
    eval_ds = sequence_packing.pack_dataset(eval_ds, max_target_length, pad_id=pad_id)
    eval_ds = eval_ds.map(lambda x: format_fn(x, pad_id=pad_id), num_parallel_calls=AUTOTUNE)

  # Every host contributes the same number of rows to the global batch, but only the
  # rows in `real_data_rows` survive the decimation in the eval step, so only those are
  # filled with real examples.
  local_batch_size = eval_global_batch_size_to_load // jax.process_count()
  if real_data_rows is None:
    real_data_rows = range(local_batch_size)
  real_data_rows = list(real_data_rows)
  real_batch_size = len(real_data_rows)

  # ensure array split in an equal division for each device
  # pad zeros up to the same batch_size among all processes
  eval_ds = _pad_to_batch_size(eval_ds, real_batch_size, num_examples, num_batches, dataloading_host_count)
  eval_ds = eval_ds.batch(real_batch_size, drop_remainder=False)
  if real_data_rows != list(range(local_batch_size)):
    eval_ds = eval_ds.map(
        functools.partial(
            _place_real_rows_in_local_batch,
            local_batch_size=local_batch_size,
            real_data_rows=real_data_rows,
        ),
        num_parallel_calls=AUTOTUNE,
    )

  # The eval loop consumes at most `num_batches` batches per eval. Truncate before the
  # cache so that the first eval completes it: `cache()` only commits after a full pass
  # over its input, and a partially read cache is discarded when the iterator is reset,
  # which would make every eval read the eval files again.
  if num_batches > 0:
    eval_ds = eval_ds.take(num_batches)

  # We are running eval over exactly one epoch.
  # We explicitly cache the entire epoch (in memory) to ensure that it is the
  # same across different iterations.
  eval_ds = eval_ds.cache()
  eval_ds = eval_ds.prefetch(AUTOTUNE)

  return eval_ds


def make_c4_mlperf_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Make train iterator of customized C4 dataset for mlperf training."""
  train_col = config.train_data_columns[0]
  is_tokenized_dataset = not config.tokenize_train_data
  # Fall back to commonly used train2 train_split for processed dataset
  train_split = getattr(config, "train_split", "train2")
  train_data_dir = getattr(config, "dataset_path", None)

  train_ds = get_dataset(
      dataset_name=config.dataset_name,
      split=train_split,
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
      data_dir=train_data_dir,
      enable_data_shuffling=config.enable_data_shuffling,
      data_shuffle_seed=config.data_shuffle_seed,
  )

  train_ds = rekey(train_ds, {"inputs": None, "targets": train_col})
  sp_tokenizer = get_tokenizer(
      config.tokenizer_path, config.tokenizer_type, config.add_bos, config.add_eos, config.hf_access_token
  )
  use_stream_chunking = getattr(config, "use_stream_chunking", None)
  train_ds = preprocess_train_dataset(
      train_ds,
      sp_tokenizer=sp_tokenizer,
      train_global_batch_size_to_load=config.global_batch_size_to_load,
      max_target_length=config.max_target_length,
      shuffle_buffer_size=128,
      data_shuffle_seed=config.data_shuffle_seed,
      is_tokenized_dataset=is_tokenized_dataset,
      use_stream_chunking=use_stream_chunking,
  )
  train_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(train_ds, global_mesh)
  return train_multihost_gen


def make_c4_mlperf_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
    num_examples: int = 0,
):
  """Make eval iterator of customized C4 dataset for mlperf training."""
  eval_col = config.eval_data_columns[0]
  is_tokenized_dataset = not config.tokenize_eval_data
  eval_split = getattr(config, "eval_split", None)
  # Fall back to commonly used eval split for processed dataset
  if eval_split is None:
    if config.eval_dataset_name == "c4/en:3.0.4":
      eval_split = "validation_24567exp"
    elif config.eval_dataset_name in [
        "c4/en:3.0.1",
        "c4/en:3.0.5",
        "c4/en:3.0.8",
        "c4/en:3.0.9",
    ]:
      eval_split = "validation"
    else:
      raise ValueError(
          f"{config.eval_dataset_name=} should be one of "
          "('c4/en:3.0.1', 'c4/en:3.0.4', 'c4/en:3.0.5', "
          "'c4/en:3.0.8', 'c4/en:3.0.9')"
      )

  eval_data_dir = getattr(config, "dataset_path", None)
  eval_ds = get_dataset(
      dataset_name=config.eval_dataset_name,
      split=eval_split,
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
      data_dir=eval_data_dir,
      enable_data_shuffling=False,
      shard_in_read=_files_hold_one_example_each(
          config.eval_dataset_name, eval_split, eval_data_dir, len(process_indices)
      ),
  )
  eval_ds = rekey(eval_ds, {"inputs": None, "targets": eval_col})

  sp_tokenizer = get_tokenizer(
      config.tokenizer_path, config.tokenizer_type, config.add_bos, config.add_eos, config.hf_access_token
  )

  # Only the first `global_batch_size_to_eval_on` rows of the loaded global batch are
  # evaluated (see the decimation in `loss_fn`), so those are the only rows that need
  # real examples.
  global_batch_size_to_load_eval = config.global_batch_size_to_load_eval
  global_batch_size_to_eval_on = getattr(config, "global_batch_size_to_eval_on", 0) or global_batch_size_to_load_eval
  local_batch_size = global_batch_size_to_load_eval // jax.process_count()
  real_data_rows = list(range(local_batch_size))
  if global_batch_size_to_eval_on < global_batch_size_to_load_eval:
    rows = get_local_rows_loading_real_data(
        config.data_sharding,
        global_batch_size_to_load_eval,
        global_batch_size_to_eval_on,
        config.max_target_length,
        global_mesh,
    )
    if rows:
      real_data_rows = rows
    else:
      # This host was selected to load real data, so it should hold at least one
      # evaluated row. Fall back to filling every row rather than dropping the shard.
      max_logging.log(
          f"Host {jax.process_index()} loads real eval data but holds no evaluated row, "
          "filling the whole local batch with real examples."
      )
  max_logging.log(
      f"Eval data loading host {jax.process_index()} fills {len(real_data_rows)} of "
      f"{local_batch_size} local rows with real examples."
  )

  if num_examples <= 0:
    eval_steps = getattr(config, "eval_steps", -1)
    # Each eval step only evaluates `global_batch_size_to_eval_on` examples, the rest of
    # the loaded batch is decimated before the loss is computed.
    eval_batch_size = global_batch_size_to_eval_on
    # When eval_steps > 0, derive total cluster-wide eval examples to avoid
    # the slow linear scan of the entire dataset during startup
    # (_get_num_examples).
    if eval_steps > 0 and eval_batch_size > 0:
      num_examples = int(eval_steps * eval_batch_size)

  # Partition the total cluster-wide eval examples across all hosts.
  # _pad_to_batch_size expects the per-host local example count rather than
  # the cluster-wide total. Distributing examples evenly across hosts (with
  # remainder distributed to the first remainder hosts) ensures each host
  # calculates the correct local batch count and padding entries without
  # linear counting.
  local_num_examples = 0
  num_batches = 0
  if num_examples > 0:
    host_idx = process_indices.index(jax.process_index()) if jax.process_index() in process_indices else 0
    per_host, remainder = divmod(num_examples, len(process_indices))
    local_num_examples = per_host + (1 if host_idx < remainder else 0)
    # Every loading host has to produce the same number of batches. Derive it from the
    # largest local share instead of an allgather, which would hang when only a subset
    # of the processes loads real data.
    max_local_num_examples = per_host + (1 if remainder else 0)
    num_batches = (max_local_num_examples + len(real_data_rows) - 1) // len(real_data_rows)

  use_stream_chunking = getattr(config, "use_stream_chunking", None)
  eval_ds = preprocess_eval_dataset(
      eval_ds,
      sp_tokenizer=sp_tokenizer,
      eval_global_batch_size_to_load=global_batch_size_to_load_eval,
      max_target_length=config.max_target_length,
      num_examples=local_num_examples,
      is_tokenized_dataset=is_tokenized_dataset,
      use_stream_chunking=use_stream_chunking,
      real_data_rows=real_data_rows,
      num_batches=num_batches,
      dataloading_host_count=len(process_indices),
  )

  eval_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(eval_ds, global_mesh)

  # Return multi-host jax.Array prep iterator
  return eval_multihost_gen
