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

"""Input pipeline using Huggingface datasets."""

import ml_collections

import jax

import datasets

import transformers

import grain.python as grain

import numpy as np

from MaxText.input_pipeline import _input_pipeline_utils
from MaxText.input_pipeline import instruction_data_processing
from MaxText import multihost_dataloading


def vision_sft_preprocessing_pipeline(
    dataset,
    config,
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    text_columns,
    image_column,
    global_batch_size,
):
  """pipeline for multimodal SFT with HF dataset"""

  assert len(text_columns) == 2, f"Need two text_columns for query and response, received {text_columns=}"
  batch_size = global_batch_size // jax.process_count()
  if config.enable_data_shuffling:
    dataset = dataset.shuffle(seed=config.data_shuffle_seed)

  # If multiple image columns are provided, merge them into a single 'images' column.
  if isinstance(image_column, list):
    dataset = dataset.map(
        _input_pipeline_utils.merge_image_columns,
        fn_kwargs={
            "image_columns": image_column,
            "max_num_images_per_example": config.max_num_images_per_example,
        },
        remove_columns=image_column,  # Drop the original image columns
    )
    image_column = "images"

  dataset = dataset.select_columns(text_columns + [image_column])
  if image_column != "images":
    dataset = dataset.rename_column(image_column, "images")

  dataset = dataset.map(
      _input_pipeline_utils.reformat_prompt,
      fn_kwargs={
          "column": text_columns[0],
          "image_placeholder": config.image_placeholder,
          "video_placeholder": config.video_placeholder,
          "audio_placeholder": config.audio_placeholder,
          "model_name": config.model_name,
      },
  )
  dataset = dataset.map(
      _input_pipeline_utils.reformat_response,
      fn_kwargs={"column": text_columns[1], "model_name": config.model_name},
  )

  dataset = dataset.map(
      _input_pipeline_utils.pre_process_image_sft,
      fn_kwargs={"image_column": "images", "model_name": config.model_name, "config": config},
  )

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      add_bos_token=False,
      add_eos_token=False,
      legacy=False,
      token=config.hf_access_token,
  )
  if tokenizer.pad_token_id is not None:
    pad_id = tokenizer.pad_token_id
  elif tokenizer.unk_token_id is not None:
    pad_id = tokenizer.unk_token_id
  else:
    pad_id = -1

  dataset = dataset.map(
      _input_pipeline_utils.tokenization,
      batched=True,
      batch_size=global_batch_size,
      fn_kwargs={
          "hf_tokenizer": tokenizer,
          "truncation": False,
          "max_length": config.max_target_length,
          "column_names": text_columns,
      },
  )
  dataset = dataset.map(
      _input_pipeline_utils.prepare_text_for_image_fusion,
      fn_kwargs={
          "column_name": text_columns[0],
          "model_name": config.model_name,
          "config": config.spatial_merge_size_for_vit,
          "spatial_merge_size": config.spatial_merge_size_for_vit,
          "position_id_per_seconds": config.position_id_per_seconds,
          "use_audio_in_video": config.use_audio_in_video,
      },
  )

  dataset = _input_pipeline_utils.HFDataSource(
      dataset=dataset,
      dataloading_host_index=dataloading_host_index,
      dataloading_host_count=dataloading_host_count,
      num_threads=1,
      max_target_length=config.max_target_length,
      data_column_names=text_columns,
  )
  operations = []
  operations.append(
      _input_pipeline_utils.SFTPromptMaskingVision(
          query_column=text_columns[0],
          response_column=text_columns[1],
          max_target_length=config.max_target_length,
          unk_id=pad_id,
      )
  )
  # TODO(aireenmei, hengtaoguo): support packing
  operations.append(
      _input_pipeline_utils.PadOrTrimToMaxLength(
          config.max_target_length,
          pad_id,
          model_name=config.model_name,
          max_num_images_per_example=config.max_num_images_per_example,
          config=config,
      )
  )
  operations.append(_input_pipeline_utils.ExtractImagesAndMasks())
  operations.append(grain.Batch(batch_size=batch_size, drop_remainder=True))
  operations.append(_input_pipeline_utils.FoldImagesIntoBatch(model_name=config.model_name, config=config))
  operations.append(_input_pipeline_utils.ShiftData(ignored_ids=[pad_id], axis=1))
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
      worker_count=1,  # only supports <=1 for now, more workers results in duplicated data
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=batch_size * 4),
  )

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen


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
    chat_template_path="",
    add_bos=True,
    add_eos=True,
    packing=True,
    shift=True,
    num_threads=1,
    drop_remainder=True,
    generate_padding_batch=False,
    use_dpo=None,
    use_sft=None,
    sft_train_on_completion_only=True,
    grain_worker_count=1,  # only support 0 or 1
):
  """pipeline for preprocessing HF dataset"""

  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible by number of global devices."

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

    supported_columns = [["prompt", "completion"], ["messages"], ["question", "answer"]]
    assert any(
        set(data_column_names) == set(supported) for supported in supported_columns
    ), f"Dataset column names mismatch. Expected columns to match one of {supported_columns}, but got {data_column_names}"

    # convert instruction dataset to conversational format
    dataset, data_column_names = instruction_data_processing.convert_to_conversational_format(
        dataset=dataset, data_columns=data_column_names, chat_template_path=chat_template_path
    )

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
    operations.append(_input_pipeline_utils.PadOrTrimToMaxLength(max_target_length, pad_id))
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

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh, generate_padding_batch)

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
  if config.use_sft and config.use_multimodal:
    train_iter = vision_sft_preprocessing_pipeline(
        dataset=train_ds,
        config=config,
        dataloading_host_index=process_indices_train.index(jax.process_index()),
        dataloading_host_count=len(process_indices_train),
        global_mesh=global_mesh,
        text_columns=config.train_data_columns,
        image_column=config.train_image_column,
        global_batch_size=config.global_batch_size_to_load,
    )
  else:
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
        generate_padding_batch=config.generate_padding_batch_train,
        use_dpo=config.use_dpo,
        use_sft=config.use_sft,
        sft_train_on_completion_only=config.sft_train_on_completion_only,
        chat_template_path=config.chat_template_path,
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
  if config.use_sft and config.use_multimodal:
    eval_iter = vision_sft_preprocessing_pipeline(
        dataset=eval_ds,
        config=config,
        dataloading_host_index=process_indices_eval.index(jax.process_index()),
        dataloading_host_count=len(process_indices_eval),
        global_mesh=global_mesh,
        text_columns=config.eval_data_columns,
        image_column=config.eval_image_column,
        global_batch_size=config.global_batch_size_to_load_eval,
    )
  else:
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
        generate_padding_batch=config.generate_padding_batch_eval,
        use_dpo=config.use_dpo,
        use_sft=config.use_sft,
        sft_train_on_completion_only=config.sft_train_on_completion_only,
        chat_template_path=config.chat_template_path,
    )
  return eval_iter
