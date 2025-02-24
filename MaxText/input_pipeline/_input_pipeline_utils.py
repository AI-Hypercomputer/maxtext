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

"""Operations used by Grain"""

import dataclasses
import warnings
from typing import Dict
from threading import current_thread
import datasets
from datasets.distributed import split_dataset_by_node
import grain.python as grain
import numpy as np
import tensorflow as tf
import max_logging
import tokenizer

Features = Dict[str, tf.Tensor]
AUTOTUNE = tf.data.experimental.AUTOTUNE

########## Functions used by TFDS pipeline


def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}


def get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token=""):
  # Load tokenizer
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  return tokenizer_model


def truncate_to_max_allowable_length(x, max_length):
  return {k: v[:max_length] for k, v in x.items()}


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


def add_segmentation_and_position(x, data_columns):
  for data_column in data_columns:
    x[f"{data_column}_segmentation"] = tf.cast(x[data_column] != 0, tf.int32)
    x[f"{data_column}_position"] = tf.broadcast_to(
        tf.range(x[data_column].shape[-1], dtype=np.int32)[None, :], x[data_column].shape
    )
  return x


########## Functions used by HF pipeline


def combine_columns(example, columns):
  assert len(columns) > 1
  combined = []
  for i in range(len(example[columns[0]])):
    for c in columns:
      combined.append(example[c][i])
  example[columns[0]] = combined
  return example


def extract_messages_and_mask(example, data_column_name):
  example["is_prompt"] = [x["role"] == "user" for x in example[data_column_name]]
  example[data_column_name] = [x["content"] for x in example[data_column_name]]
  return example


def tokenization(example, hf_tokenizer, max_length, column_names):
  """Tokenize a HuggingFace dataset"""
  for column_name in column_names:
    if isinstance(example[column_name], list):
      example[column_name] = [
          hf_tokenizer(x, truncation=True, max_length=max_length)["input_ids"] for x in example[column_name]
      ]
    elif isinstance(example[column_name], str):
      example[column_name] = hf_tokenizer(example[column_name], truncation=True, max_length=max_length)["input_ids"]
  return example


@dataclasses.dataclass
class SFTPromptMasking(grain.MapTransform):

  def __init__(self, text_column_name, completion_only, max_target_length, add_bos, add_eos, bos_id=None, eos_id=None):
    self.text_column_name = text_column_name
    self.completion_only = completion_only
    self.max_target_length = max_target_length
    self.add_bos = add_bos
    self.add_eos = add_eos
    if self.add_bos:
      self.bos_id = bos_id
    if self.add_eos:
      self.eos_id = eos_id

  def map(self, features):
    inputs, targets = [], []
    for i, text in enumerate(features[self.text_column_name]):
      inputs += text
      targets += [0] * len(text) if self.completion_only and features["is_prompt"][i] else text
    if self.add_bos:
      inputs = [self.bos_id] + inputs
      targets = [self.bos_id] + targets
    if self.add_eos:
      inputs += [self.eos_id]
      targets += [self.eos_id]
    return {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
    }


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""

  def __init__(self, column_name):
    self.column_name = column_name

  def map(self, features):
    return {
        "inputs": np.asarray(features[self.column_name], dtype=np.int32),
        "targets": np.asarray(features[self.column_name], dtype=np.int32),
    }


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(
      self,
      dataset: datasets.IterableDataset,
      dataloading_host_index: int,
      dataloading_host_count: int,
      num_threads: int,
      generate_padding_example: bool,
      max_target_length: int,
      data_column_names: list[str],
  ):
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.generate_padding_example = generate_padding_example
    self.max_target_lenth = max_target_length
    self.data_column_names = data_column_names
    self.n_shards = dataset.n_shards
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []
    self.out_of_data = False

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(
          f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
          "smaller than number of host loading data. This is known to lead to inefficient dataloading. "
          "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
      )
      self.n_shards = self.dataloading_host_count * self.num_threads

  def _update_shard(self, idx):
    new_shard = self.dataset_shards[idx] + self.dataloading_host_count * self.num_threads
    if new_shard < self.n_shards:
      max_logging.log(f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}")
      max_logging.log(f"New shard is {new_shard}")
      self.dataset_shards[idx] = new_shard
      self.datasets[idx] = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx])
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      max_logging.log(
          f"Run out of shards on host {self.dataloading_host_index}, shard {self.dataset_shards[idx]} is not available"
      )
      self.out_of_data = True
      if self.generate_padding_example:
        max_logging.log(
            f"Host {self.dataloading_host_index} will start generating all-0 padding examples until step number is met."
        )

  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        if self.out_of_data:
          if self.generate_padding_example:
            return {column_name: np.zeros(self.max_target_lenth, dtype=np.int32) for column_name in self.data_column_names}
          else:
            return None
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)


########## Functions used by Grain pipeline


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized example"""

  def __init__(self, data_columns, tokenize):
    self.data_columns = data_columns
    if tokenize:
      self.dtype = tf.string
    else:
      self.dtype = tf.int64

  def map(self, features):
    def _parse(example):
      parsed = tf.io.parse_example(
          example,
          {col: tf.io.FixedLenSequenceFeature([], dtype=self.dtype, allow_missing=True) for col in self.data_columns},
      )
      return parsed

    return _parse(features)


@dataclasses.dataclass
class InputsTargetsFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_name):
    self.column_name = column_name

  def map(self, features):
    return {"inputs": features[self.column_name], "targets": features[self.column_name]}


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_names, tokenize):
    self.column_names = column_names
    self.tokenize = tokenize

  def map(self, features):
    if self.tokenize:
      return {col: features[col].numpy()[0].decode() for col in self.column_names}
    else:
      return {col: features[col].numpy() for col in self.column_names}


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def __init__(self, column_names):
    self.column_names = column_names

  def map(self, data):
    ret = {}
    for col in self.column_names:
      ret[f"{col}"] = data[0][col]
      ret[f"{col}_segmentation"] = data[1][col]
      ret[f"{col}_position"] = data[2][col]
    return ret


@dataclasses.dataclass
class PadToMaxLength(grain.MapTransform):
  """Pads each input to the specified length"""

  def __init__(self, max_length):
    self.max_length = max_length

  def map(self, data: dict[str, np.ndarray]):
    """map to each element"""

    def _pad(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
      return np.pad(x, pad_amount)

    data_columns = list(data.keys())
    for data_column in data_columns:
      data[f"{data_column}_segmentation"] = (data[data_column] != 0).astype(np.int32)
      data[f"{data_column}_position"] = np.arange(data[data_column].shape[0], dtype=np.int32)
    for key, _ in data.items():
      data[key] = _pad(data[key], self.max_length)
    return data


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_left(x, axis=1):
  """Shift to the left and pad."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(1, None)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_and_refine(x, bos_id, axis=1):
  """Shift inputs, set segmentation to 0 when target element is 0 or bos_id if provided"""
  x["targets"] = shift_left(x["targets"], axis=axis)
  if bos_id:
    x["targets_segmentation"] = np.where((x["targets"] != 0) & (x["targets"] != bos_id), x["targets_segmentation"], 0)
  else:
    x["targets_segmentation"] = np.where(x["targets"] != 0, x["targets_segmentation"], 0)

  x["inputs_segmentation"] = x["targets_segmentation"]

  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Shift inputs and refine annotations."""

  def __init__(self, bos_id=None, axis=1):
    self.axis = axis
    self.bos_id = bos_id

  def map(self, data):
    return shift_and_refine(data, bos_id=self.bos_id, axis=self.axis)
