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
# pylint: disable=g-inconsistent-quotes

import functools
import types
import warnings

import jax
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline.packing import sequence_packing
from maxtext.multimodal import processor as mm_processor
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

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


def vision_sft_preprocessing_pipeline(
    dataset,
    config,
    global_batch_size: int,
    max_target_length: int,
    data_column_names,
    image_column: str,
    shuffle: bool = False,
    data_shuffle_seed=0,
    num_epochs: None | int = 1,
    drop_remainder: bool = True,
    prefetch_size=tf.data.experimental.AUTOTUNE,
):
  """Pipeline for preprocessing TFDS dataset for Multimodal SFT."""
  assert (
      len(data_column_names) == 2
  ), f'Need two data_column_names for SFT, received {data_column_names=}'

  # Load tokenizer to get pad_id
  tokenizer_model = input_pipeline_utils.get_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      add_bos=False,
      add_eos=False,
      hf_access_token=config.hf_access_token,
  )
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1

  # 1. Reformat prompt and response
  def _reformat_prompt_fn(prompt):
    prompt_str = prompt.numpy().decode('utf-8')
    formatted = mm_processor.reformat_prompt(
        prompt_str, config.image_placeholder, config.model_name, num_images=1
    )
    return formatted

  def _reformat_response_fn(response):
    response_str = response.numpy().decode('utf-8')
    formatted = mm_processor.reformat_response(response_str, config.model_name)
    return formatted

  def map_reformat(x):
    prompt_formatted = tf.py_function(
        _reformat_prompt_fn, [x[data_column_names[0]]], tf.string
    )
    response_formatted = tf.py_function(
        _reformat_response_fn, [x[data_column_names[1]]], tf.string
    )
    prompt_formatted.set_shape([])
    response_formatted.set_shape([])
    x[data_column_names[0]] = prompt_formatted
    x[data_column_names[1]] = response_formatted
    return x

  dataset = dataset.map(map_reformat, num_parallel_calls=tf.data.AUTOTUNE)

  # 2. Preprocess image
  def _preprocess_image_fn(image_tensor):
    image_np = image_tensor.numpy()
    output = mm_processor.preprocess_image_for_training(
        image_np, config.model_name
    )
    pixel_values = output.pixel_values
    pixel_grid_thw = getattr(
        output, 'pixel_grid_thw', np.zeros([0, 3], dtype=np.int32)
    )
    return pixel_values, pixel_grid_thw

  dummy_image_shape = mm_processor.get_dummy_image_shape_for_init(
      config.model_name, batch_size=1
  )

  def map_image(x):
    pixel_values, pixel_grid_thw = tf.py_function(
        _preprocess_image_fn, [x[image_column]], [tf.float32, tf.int32]
    )
    pixel_values.set_shape(dummy_image_shape)
    pixel_grid_thw.set_shape([None, 3])
    x['images'] = pixel_values
    x['image_grid_thw'] = pixel_grid_thw
    return x

  dataset = dataset.map(map_image, num_parallel_calls=tf.data.AUTOTUNE)

  # 3. Tokenization
  dataset = dataset.map(
      lambda x: input_pipeline_utils.TokenizeOp(
          tokenizer_model=tokenizer_model,
          features=x,
          data_keys=data_column_names,
      ),
      num_parallel_calls=tf.data.AUTOTUNE,
  )

  # 4. Prepare text for image fusion (expand <|image_pad|>)
  def _prepare_text_fn(tokens, grid_thw):
    tokens_np = tokens.numpy()
    grid_thw_np = grid_thw.numpy()
    processor_output = types.SimpleNamespace(
        pixel_grid_thw=grid_thw_np,
        num_images=grid_thw_np.shape[0] if grid_thw_np is not None else 0,
    )
    expanded_tokens = mm_processor.prepare_text_for_image_fusion(
        tokens_np, config, processor_output
    )
    return expanded_tokens

  def map_prepare_text(x):
    expanded = tf.py_function(
        _prepare_text_fn,
        [x[data_column_names[0]], x['image_grid_thw']],
        tf.int32,
    )
    expanded.set_shape([None])
    x[data_column_names[0]] = expanded
    return x

  dataset = dataset.map(map_prepare_text, num_parallel_calls=tf.data.AUTOTUNE)

  # 5. Prompt Masking and Concatenation
  def map_sft_masking(x):
    query = x[data_column_names[0]]
    response = x[data_column_names[1]]

    inputs = tf.concat([query, response], axis=0)

    query_len = tf.shape(query)[0]
    response_len = tf.shape(response)[0]
    pad_tokens = tf.fill([query_len], tf.cast(pad_id, tf.int32))
    targets = tf.concat([pad_tokens, response], axis=0)

    inputs_len = tf.shape(inputs)[0]
    segmentation = tf.ones([inputs_len], dtype=tf.int32)

    targets_segmentation = tf.concat(
        [
            tf.zeros([query_len], dtype=tf.int32),
            tf.ones([response_len], dtype=tf.int32),
        ],
        axis=0,
    )

    x['inputs'] = inputs
    x['targets'] = targets
    x['inputs_segmentation'] = segmentation
    x['targets_segmentation'] = targets_segmentation
    return x

  dataset = dataset.map(map_sft_masking, num_parallel_calls=tf.data.AUTOTUNE)

  # 6. Position ID computation
  def _compute_positions_fn(input_ids, image_grid_thw, attention_mask):
    input_ids_np = input_ids.numpy()[np.newaxis, :]
    image_grid_thw_np = image_grid_thw.numpy()
    attention_mask_np = attention_mask.numpy()[np.newaxis, :]

    position_ids = mm_processor.get_rope_index(
        config=config,
        input_ids=input_ids_np,
        image_grid_thw=image_grid_thw_np,
        attention_mask=attention_mask_np,
    )
    if config.use_mrope:
      return position_ids[:, 0, :]
    else:
      return position_ids[0, :]

  def map_positions(x):
    if config.use_mrope:
      out_shape = [3, None]
    else:
      out_shape = [None]

    positions = tf.py_function(
        _compute_positions_fn,
        [x['inputs'], x['image_grid_thw'], x['inputs_segmentation']],
        tf.int32,
    )
    positions.set_shape(out_shape)
    x['inputs_position'] = positions
    return x

  dataset = dataset.map(map_positions, num_parallel_calls=tf.data.AUTOTUNE)

  # Shuffle and repeat.
  if shuffle:
    dataset = dataset.shuffle(1024, seed=data_shuffle_seed)

  dataset = dataset.repeat(num_epochs)

  # 7. Padding to max_target_length
  def map_padding(x):
    max_len = max_target_length

    inputs_len = tf.shape(x['inputs'])[0]
    pad_amount = tf.maximum(max_len - inputs_len, 0)

    x['inputs'] = tf.pad(
        x['inputs'], [[0, pad_amount]], constant_values=pad_id
    )[:max_len]
    x['targets'] = tf.pad(
        x['targets'], [[0, pad_amount]], constant_values=pad_id
    )[:max_len]
    x['inputs_segmentation'] = tf.pad(
        x['inputs_segmentation'], [[0, pad_amount]], constant_values=0
    )[:max_len]
    x['targets_segmentation'] = tf.pad(
        x['targets_segmentation'], [[0, pad_amount]], constant_values=0
    )[:max_len]

    if config.use_mrope:
      x['inputs_position'] = tf.pad(
          x['inputs_position'], [[0, 0], [0, pad_amount]], constant_values=0
      )[:, :max_len]
    else:
      x['inputs_position'] = tf.pad(
          x['inputs_position'], [[0, pad_amount]], constant_values=0
      )[:max_len]

    return {
        'inputs': x['inputs'],
        'targets': x['targets'],
        'inputs_position': x['inputs_position'],
        'inputs_segmentation': x['inputs_segmentation'],
        'targets_segmentation': x['targets_segmentation'],
        'images': x['images'],
        'image_grid_thw': x['image_grid_thw'],
    }

  dataset = dataset.map(map_padding, num_parallel_calls=tf.data.AUTOTUNE)

  # 8. Batching
  dataset = dataset.batch(
      global_batch_size // jax.process_count(), drop_remainder=drop_remainder
  )

  # 9. Post-batching transformations (folding images if needed)
  trailing_dims = dummy_image_shape[1:]

  def map_post_batch(x):
    # Fold images: (B, N, ...) -> (B * N, ...)
    x['images'] = tf.reshape(x['images'], [-1] + list(trailing_dims))
    return x

  dataset = dataset.map(map_post_batch, num_parallel_calls=tf.data.AUTOTUNE)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


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
    hf_access_token: str = "",
):
  """pipeline for preprocessing TFDS dataset."""
  missing = [c for c in data_column_names if c not in dataset.element_spec]
  if missing:
    raise ValueError(
        f"Column {missing} not found in dataset. Available columns: {sorted(dataset.element_spec.keys())}. "
        "Please set train_data_columns or eval_data_columns accordingly."
    )

  for col in data_column_names:
    col_dtype = dataset.element_spec[col].dtype
    if tokenize and col_dtype != tf.string:
      raise ValueError(
          f"tokenize_data=True but column '{col}' has dtype {col_dtype} (expected tf.string). "
          "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
      )
    if not tokenize and col_dtype == tf.string:
      raise ValueError(
          f"tokenize_data=False but column '{col}' has dtype tf.string (expected integer). "
          "Set tokenize_train_data or tokenize_eval_data to True if your dataset needs tokenization."
      )

  assert len(data_column_names) == 1
  dataset = dataset.map(
      lambda x: input_pipeline_utils.normalize_features(x, data_column_names[0]), num_parallel_calls=AUTOTUNE
  )
  data_column_names = ("inputs", "targets")

  tokenizer_model = input_pipeline_utils.get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1

  if tokenize:
    dataset = dataset.map(
        lambda x: input_pipeline_utils.TokenizeOp(
            tokenizer_model=tokenizer_model, features=x, data_keys=data_column_names
        ),
        num_parallel_calls=AUTOTUNE,
    )

  if max_target_length > 0:
    # in pre-training we can take upto max_length+1 because there would be truncation by
    # 1 token for both inputs and targets
    extra_tokens = 1
    dataset = dataset.map(
        lambda x: input_pipeline_utils.truncate_to_max_allowable_length(x, max_target_length + extra_tokens),
        num_parallel_calls=AUTOTUNE,
    )

  # Shuffle and repeat.
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)

  dataset = dataset.repeat(num_epochs)

  # Shift inputs for teacher-forced training
  if shift:
    dataset = dataset.map(
        input_pipeline_utils.shift_data_by_truncation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )

  # Perform greedy sequence packing and batching
  if pack_examples:
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
        lambda x: input_pipeline_utils.add_segmentation_and_position(x, data_column_names, padding_token=pad_id),
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
    if config.use_multimodal and config.use_sft:
      train_dataloader = vision_sft_preprocessing_pipeline(
          dataset=train_ds,
          config=config,
          global_batch_size=config.global_batch_size_to_load,
          max_target_length=config.max_target_length,
          data_column_names=config.train_data_columns,
          image_column=config.train_image_column,
          shuffle=config.enable_data_shuffling,
          data_shuffle_seed=config.data_shuffle_seed,
          num_epochs=config.num_epoch,
      )
    else:
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
          hf_access_token=config.hf_access_token,
      )
    return multihost_dataloading.MultiHostDataLoadIterator(
        train_dataloader, global_mesh, config.generate_padding_batch_train
    )
  else:
    if config.use_multimodal and config.use_sft:
      raise NotImplementedError(
          'colocated_python_data_input is not supported for multimodal SFT yet.'
      )
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
        hf_access_token=config.hf_access_token,
    )
    global_shape = (config.global_batch_size_to_load, config.max_target_length)
    return multihost_dataloading.RemoteIteratorWrapper(
        get_ds_fn, preprocessing_fn, global_mesh, global_shape, checkpoint_path=config.checkpoint_dir
    )


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
        dataset_path=config.dataset_path,
        data_split=config.eval_split,
        shuffle_files=False,
        shuffle_seed=config.data_shuffle_seed,
        dataloading_host_index=process_indices_eval.index(jax.process_index()),
        dataloading_host_count=len(process_indices_eval),
    )
    if config.use_multimodal and config.use_sft:
      eval_dataloader = vision_sft_preprocessing_pipeline(
          dataset=eval_ds,
          config=config,
          global_batch_size=config.global_batch_size_to_load_eval,
          max_target_length=config.max_target_length,
          data_column_names=config.eval_data_columns,
          image_column=config.eval_image_column,
          shuffle=False,
          data_shuffle_seed=config.data_shuffle_seed,
          num_epochs=1,
      )
    else:
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
          hf_access_token=config.hf_access_token,
      )
    return multihost_dataloading.MultiHostDataLoadIterator(
        eval_dataloader, global_mesh, config.generate_padding_batch_eval
    )
  else:
    if config.use_multimodal and config.use_sft:
      raise NotImplementedError(
          'colocated_python_data_input is not supported for multimodal SFT yet.'
      )
    get_ds_fn = functools.partial(
        get_datasets,
        dataset_name=config.eval_dataset_name,
        dataset_path=config.dataset_path,
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
        hf_access_token=config.hf_access_token,
    )
    global_shape = (config.global_batch_size_to_load_eval, config.max_target_length)
    return multihost_dataloading.RemoteIteratorWrapper(
        get_ds_fn, preprocessing_fn, global_mesh, global_shape, checkpoint_path=config.checkpoint_dir
    )
