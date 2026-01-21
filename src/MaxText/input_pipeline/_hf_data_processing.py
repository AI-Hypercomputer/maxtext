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

from typing import List, Optional, Union

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
    dataset: datasets.Dataset,
    config: ml_collections.ConfigDict,
    dataloading_host_index: int,
    dataloading_host_count: int,
    global_mesh: jax.sharding.Mesh,
    text_columns: List[str],
    image_column: Union[str, List[str]],
    global_batch_size: int,
) -> multihost_dataloading.MultiHostDataLoadIterator:
  """Constructs a pipeline for multimodal SFT using Hugging Face datasets.

  Args:
    dataset: The Hugging Face dataset to process.
    config: Configuration object containing model and data settings.
    dataloading_host_index: Index of the current host among data loaders.
    dataloading_host_count: Total number of data loading hosts.
    global_mesh: The JAX global mesh used for distributed training.
    text_columns: List of column names containing text data (query, response).
    image_column: Column name or list of names containing image data.
    global_batch_size: The global batch size across all devices.

  Returns:
    A MultiHostDataLoadIterator that yields sharded batches of data.
  """

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
          "model_name": config.model_name,
      },
  )
  dataset = dataset.map(
      _input_pipeline_utils.reformat_response,
      fn_kwargs={"column": text_columns[1], "model_name": config.model_name},
  )

  dataset = dataset.map(
      _input_pipeline_utils.pre_process_image_sft,
      fn_kwargs={"image_column": "images", "model_name": config.model_name},
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
      fn_kwargs={"column_name": text_columns[0], "model_name": config.model_name},
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
      )
  )
  operations.append(_input_pipeline_utils.ExtractImagesAndMasks())
  operations.append(grain.Batch(batch_size=batch_size, drop_remainder=True))
  operations.append(_input_pipeline_utils.FoldImagesIntoBatch(model_name=config.model_name))
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


class GrainPipelineBuilder:
  """Builder pattern for constructing a Grain-based data loading pipeline from HF datasets.

  This builder manages the transformation of a Hugging Face dataset into a Grain dataloader,
  applying tokenization, shuffling, packing, and other preprocessing steps fluently.
  """

  def __init__(
      self,
      dataset: datasets.Dataset,
      global_mesh: jax.sharding.Mesh,
      dataloading_host_index: int,
      dataloading_host_count: int,
  ):
    """Initializes the builder.

    Args:
        dataset: The initial Hugging Face dataset.
        global_mesh: JAX mesh for distributing data.
        dataloading_host_index: Index of the current data loading host.
        dataloading_host_count: Total number of data loading hosts.
    """
    self._dataset = dataset
    self._global_mesh = global_mesh
    self._dataloading_host_index = dataloading_host_index
    self._dataloading_host_count = dataloading_host_count

    # State variables modified by methods
    self._operations: List[grain.Operation] = []
    self._tokenizer: Optional[transformers.PreTrainedTokenizer] = None
    self._pad_id: int = -1
    self._data_column_names: Optional[List[str]] = None
    self._max_target_length: int = 0
    self._global_batch_size: int = 0

  def add_shuffling(self, shuffle: bool, seed: int) -> "GrainPipelineBuilder":
    """Adds shuffling to the dataset if enabled.

    Args:
        shuffle: Whether to enable shuffling.
        seed: Random seed for shuffling.

    Returns:
        The builder instance for chaining.
    """
    if shuffle:
      self._dataset = self._dataset.shuffle(seed=seed)
    return self

  def add_tokenization(
      self,
      tokenizer_path: str,
      hf_access_token: str,
      tokenize: bool,
      data_column_names: List[str],
      max_target_length: int,
      use_sft: bool = False,
      chat_template_path: str = "",
      add_bos: bool = True,
      add_eos: bool = True,
  ) -> "GrainPipelineBuilder":
    """Configures tokenization, special tokens, and SFT-specific formatting.

    Loads the tokenizer, applies SFT chat templates and column mapping if requested,
    and applies the tokenization map to the dataset.

    Args:
        tokenizer_path: Path to the tokenizer model or Hub ID.
        hf_access_token: Hugging Face access token.
        tokenize: Whether to apply tokenization or assume pre-tokenized features.
        data_column_names: List of column names to process.
        max_target_length: Maximum sequence length.
        use_sft: Whether to apply Supervised Fine-Tuning specific logic (chat templates).
        chat_template_path: Path to a custom chat template definitions file.
        add_bos: Whether to add a Beginning-Of-Sequence token (pre-training only).
        add_eos: Whether to add an End-Of-Sequence token (pre-training only).

    Returns:
        The builder instance for chaining.
    """
    self._max_target_length = max_target_length

    self._tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_bos_token=add_bos if not use_sft else False,
        add_eos_token=add_eos if not use_sft else False,
        legacy=False,
        token=hf_access_token,
    )

    if use_sft:
      # Select columns and convert to conversational format if needed
      self._dataset = self._dataset.select_columns(data_column_names)

      supported_columns = [["prompt", "completion"], ["messages"], ["question", "answer"]]
      assert any(
          set(data_column_names) == set(supported) for supported in supported_columns
      ), f"Dataset column names mismatch. Expected columns match one of {supported_columns}, but got {data_column_names}"

      # Convert instruction dataset to conversational format
      dataset, data_column_names = instruction_data_processing.convert_to_conversational_format(
          dataset=self._dataset, data_columns=data_column_names, chat_template_path=chat_template_path
      )
      self._dataset = dataset

      assert _input_pipeline_utils.is_conversational(
          self._dataset.features, data_column_names
      ), "Dataset is not in conversational format."

      if len(data_column_names) > 1:
        combined_column_name = "messages"
        dataset_features = datasets.Features(
            {combined_column_name: [{"content": datasets.Value(dtype="string"), "role": datasets.Value(dtype="string")}]}
        )
        self._dataset = self._dataset.map(
            _input_pipeline_utils.combine_columns,
            fn_kwargs={"columns": data_column_names, "data_column": combined_column_name},
            remove_columns=data_column_names,
            features=dataset_features,
        )

      data_column_names = list(self._dataset.features.keys())
      self._dataset = self._dataset.map(
          _input_pipeline_utils.apply_chat_template,
          fn_kwargs={"tokenizer_model": self._tokenizer, "data_column_name": data_column_names[0]},
      )
    else:
      self._dataset = self._dataset.select_columns(data_column_names)

    # Determine PAD ID
    if self._tokenizer.pad_token_id is not None:
      self._pad_id = self._tokenizer.pad_token_id
    elif self._tokenizer.unk_token_id is not None:
      self._pad_id = self._tokenizer.unk_token_id
    else:
      self._pad_id = -1

    if tokenize:
      self._dataset = self._dataset.map(
          _input_pipeline_utils.tokenization,
          batched=True,
          fn_kwargs={
              "hf_tokenizer": self._tokenizer,
              "truncation": not use_sft,
              "max_length": self._max_target_length,
              "column_names": data_column_names,
          },
      )

    self._data_column_names = data_column_names
    return self

  def add_normalization(
      self,
      use_sft: bool = False,
      use_dpo: bool = False,
      sft_train_on_completion_only: bool = True,
      max_target_length: int = 0,
  ) -> "GrainPipelineBuilder":
    """Adds normalization operations to the Grain pipeline.

    Args:
        use_sft: Whether to add SFT-specific masking operations.
        use_dpo: Whether to add DPO formatting map operations.
        sft_train_on_completion_only: If True during SFT, mask prompts in target.
        max_target_length: Maximum sequence length.

    Returns:
        The builder instance for chaining.
    """
    if use_sft:
      self._operations.append(
          _input_pipeline_utils.SFTPromptMasking(
              text_column_name=self._data_column_names[0],
              completion_only=sft_train_on_completion_only,
              max_target_length=max_target_length,
              unk_id=self._pad_id,
          )
      )
      self._data_column_names = ("inputs", "targets")
    elif use_dpo:
      # For DPO, we convert lists/tuples to arrays
      def lists2array(x):
        """Convert lists/tuples to array"""
        return jax.tree.map(np.asarray, x, is_leaf=lambda y: isinstance(y, (list, tuple)))

      self._operations.append(grain.MapOperation(lists2array))
    else:
      assert len(self._data_column_names) == 1
      self._operations.append(_input_pipeline_utils.HFNormalizeFeatures(self._data_column_names[0]))
      self._data_column_names = ("inputs", "targets")
    return self

  def add_packing(
      self,
      packing: bool,
      global_batch_size: int,
      max_target_length: int,
      max_segments_per_seq: Optional[int] = None,
      drop_remainder: bool = True,
      use_dpo: bool = False,
  ) -> "GrainPipelineBuilder":
    """Adds packing or batching operations to the Grain pipeline.

    Args:
        packing: Whether to enable sequence packing.
        global_batch_size: Global batch size across all devices.
        max_target_length: Maximum sequence length used for packing bin size.
        max_segments_per_seq: Maximum sequences packed into one example.
        drop_remainder: Whether to drop the final partial batch.
        use_dpo: Whether using DPO (packing is disabled for DPO).

    Returns:
        The builder instance for chaining.
    """
    process_count = jax.process_count()
    assert (
        global_batch_size % process_count == 0
    ), f"Batch size {global_batch_size} should be divisible by number of global devices {process_count}."
    self._global_batch_size = global_batch_size

    if packing and not use_dpo:
      length_struct = {col: max_target_length for col in self._data_column_names}
      max_segments = max_segments_per_seq
      if max_segments is not None and max_segments <= 0:
        max_segments = None
      self._operations.append(
          grain.experimental.PackAndBatchOperation(
              batch_size=global_batch_size // process_count,
              length_struct=length_struct,
              max_sequences_per_bin=max_segments,
          )
      )
      self._operations.append(_input_pipeline_utils.ReformatPacking(self._data_column_names))
    else:
      self._operations.append(_input_pipeline_utils.PadOrTrimToMaxLength(max_target_length, self._pad_id))
      self._operations.append(grain.Batch(batch_size=global_batch_size // process_count, drop_remainder=drop_remainder))
    return self

  def add_shifting(self, shift: bool, use_dpo: bool) -> "GrainPipelineBuilder":
    """Adds shifting operation for autoregressive training.

    Args:
        shift: Whether to shift inputs and targets for teacher-forcing.
        use_dpo: Whether using DPO (shifting handled differently or not required).

    Returns:
        The builder instance for chaining.
    """
    if shift and not use_dpo:
      self._operations.append(
          _input_pipeline_utils.ShiftData(ignored_ids=[self._pad_id, self._tokenizer.bos_token_id], axis=1)
      )
    return self

  def build(
      self,
      num_threads: int,
      grain_worker_count: int,
      generate_padding_batch: bool,
  ) -> multihost_dataloading.MultiHostDataLoadIterator:
    """Finalizes the pipeline construction and returns the iterator.

    Args:
        num_threads: Number of threads for the HFDataSource.
        grain_worker_count: Number of worker processes for Grain DataLoader.
        generate_padding_batch: Whether to generate padding batches when exhausted.

    Returns:
        A MultiHostDataLoadIterator wrapping the configured Grain DataLoader.
    """
    # Create the data source wrapping the potentially mapped/filtered HF dataset
    dataset_source = _input_pipeline_utils.HFDataSource(
        dataset=self._dataset,
        dataloading_host_index=self._dataloading_host_index,
        dataloading_host_count=self._dataloading_host_count,
        num_threads=num_threads,
        max_target_length=self._max_target_length,
        data_column_names=self._data_column_names,
    )

    # Since HuggingFace IterableDataset does not support access through index
    # Indexes generated by dummy_index_sampler are not used.
    # dummy_index_sampler is used as an input place holder for grain.Dataloader
    dummy_index_sampler = grain.IndexSampler(
        num_records=len(dataset_source),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=self._dataloading_host_index,
            shard_count=self._dataloading_host_count,
            drop_remainder=False,
        ),
        shuffle=False,
        seed=0,
    )

    dataloader = grain.DataLoader(
        data_source=dataset_source,
        operations=self._operations,
        sampler=dummy_index_sampler,
        worker_count=grain_worker_count,  # only supports <=1 for now, more workers results in duplicated data
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=128),
    )

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, self._global_mesh, generate_padding_batch)

    return multihost_gen


def preprocessing_pipeline(
    dataloading_host_index: int,
    dataloading_host_count: int,
    global_mesh: jax.sharding.Mesh,
    dataset: datasets.Dataset,
    data_column_names: List[str],
    tokenize: bool,
    tokenizer_path: str,
    hf_access_token: str,
    global_batch_size: int,
    max_target_length: int,
    shuffle: bool,
    data_shuffle_seed: int,
    chat_template_path: str = "",
    add_bos: bool = True,
    add_eos: bool = True,
    packing: bool = True,
    shift: bool = True,
    num_threads: int = 1,
    drop_remainder: bool = True,
    generate_padding_batch: bool = False,
    use_dpo: Optional[bool] = None,
    use_sft: Optional[bool] = None,
    sft_train_on_completion_only: bool = True,
    grain_worker_count: int = 1,  # only support 0 or 1
    max_segments_per_seq: Optional[int] = None,
) -> multihost_dataloading.MultiHostDataLoadIterator:
  """Pipeline for preprocessing Hugging Face dataset.

  This function constructs a data loading pipeline that processes a Hugging Face
  dataset, applies tokenization, and formats it for training or evaluation.
  It delegates the construction logic to `GrainPipelineBuilder`.

  Args:
    dataloading_host_index: Index of the current host among data loaders.
    dataloading_host_count: Total number of data loading hosts.
    global_mesh: The JAX global mesh used for distributed training.
    dataset: The Hugging Face dataset to process.
    data_column_names: List of column names to process.
    tokenize: Whether to apply tokenization.
    tokenizer_path: Path to the tokenizer model.
    hf_access_token: Hugging Face access token.
    global_batch_size: The global batch size.
    max_target_length: Maximum target sequence length.
    shuffle: Whether to shuffle the dataset.
    data_shuffle_seed: Random seed for shuffling.
    chat_template_path: Path to a chat template file (optional).
    add_bos: Whether to add a BOS token.
    add_eos: Whether to add an EOS token.
    packing: Whether to enable sequence packing.
    shift: Whether to shift inputs for autoregressive training.
    num_threads: Number of threads for data loading.
    drop_remainder: Whether to drop the last partial batch.
    generate_padding_batch: Whether to generate padding batches if data runs out.
    use_dpo: Whether to use DPO preference optimization.
    use_sft: Whether to use Supervised Fine-Tuning.
    sft_train_on_completion_only: In SFT, mask prompt tokens.
    grain_worker_count: Number of worker processes for Grain.
    max_segments_per_seq: Max segments if using packing.

  Returns:
    A MultiHostDataLoadIterator that yields sharded, processed batches.
  """
  # Use the Builder pattern to construct the pipeline
  builder = GrainPipelineBuilder(
      dataset=dataset,
      global_mesh=global_mesh,
      dataloading_host_index=dataloading_host_index,
      dataloading_host_count=dataloading_host_count,
  )

  builder.add_shuffling(shuffle, data_shuffle_seed)

  builder.add_tokenization(
      tokenizer_path=tokenizer_path,
      hf_access_token=hf_access_token,
      tokenize=tokenize,
      data_column_names=data_column_names,
      max_target_length=max_target_length,
      use_sft=use_sft or False,
      chat_template_path=chat_template_path,
      add_bos=add_bos,
      add_eos=add_eos,
  )

  builder.add_normalization(
      use_sft=use_sft or False,
      use_dpo=use_dpo or False,
      sft_train_on_completion_only=sft_train_on_completion_only,
      max_target_length=max_target_length,
  )

  builder.add_packing(
      packing=packing,
      global_batch_size=global_batch_size,
      max_target_length=max_target_length,
      max_segments_per_seq=max_segments_per_seq,
      drop_remainder=drop_remainder,
      use_dpo=use_dpo or False,
  )

  builder.add_shifting(
      shift=shift,
      use_dpo=use_dpo or False,
  )

  return builder.build(
      num_threads=num_threads,
      grain_worker_count=grain_worker_count,
      generate_padding_batch=generate_padding_batch,
  )


def make_hf_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh: jax.sharding.Mesh,
    process_indices_train: List[int],
) -> multihost_dataloading.MultiHostDataLoadIterator:
  """Load, preprocess dataset and return iterators"""
  train_ds = datasets.load_dataset(
      config.hf_path,
      name=config.hf_name,
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
        max_segments_per_seq=config.max_segments_per_seq,
    )
  return train_iter


def make_hf_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh: jax.sharding.Mesh,
    process_indices_eval: List[int],
) -> multihost_dataloading.MultiHostDataLoadIterator:
  """Make Hugging Face evaluation iterator. Load and preprocess eval dataset: and return iterator."""
  eval_ds = datasets.load_dataset(
      config.hf_path,
      name=config.hf_name,
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
        max_segments_per_seq=config.max_segments_per_seq,
    )
  return eval_iter
