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
from typing import Dict
from threading import current_thread
import datasets
from datasets.distributed import split_dataset_by_node
import grain.python as grain
import numpy as np
import tensorflow as tf
import max_logging

Features = Dict[str, tf.Tensor]


def tokenization(example, tokenizer, max_length):
  """Tokenize a HuggingFace dataset"""
  return tokenizer(example["text"], truncation=True, max_length=max_length)


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""

  def map(self, features):
    return {
        "inputs": np.asarray(features["input_ids"], dtype=np.int32),
        "targets": np.asarray(features["input_ids"], dtype=np.int32),
    }


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(self, dataset: datasets.IterableDataset, dataloading_host_index: int, dataloading_host_count: int, num_threads: int):
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.n_shards = dataset.n_shards
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []

  def _check_shard_count(self):
    if self.n_shards % (self.dataloading_host_count * self.num_threads) != 0:
      usable_shards = (
          self.n_shards
          // (self.dataloading_host_count * self.num_threads)
          * (self.dataloading_host_count * self.num_threads)
      )
      max_logging.log(f"Dataset contains {self.n_shards} shards, but only {usable_shards} shards will be used.")
      max_logging.log("Make (dataset shards) % (number of host loading data) == 0 to use all shards of data")

  def _update_shard(self, idx):
    max_logging.log(f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}")
    self.dataset_shards[idx] += self.dataloading_host_count * self.num_threads
    max_logging.log(f"New shard is {self.dataset_shards[idx]}")
    if self.dataset_shards[idx] > self.n_shards:
      raise ValueError(f"Run out of shards, shard {self.dataset_shards[idx]} is not available")
    self.datasets[idx] = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx])
    self.data_iters[idx] = iter(self.datasets[idx])

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
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized example"""

  def map(self, features):
    def _parse(example):
      parsed = tf.io.parse_example(example, {"text": tf.io.FixedLenFeature(shape=(), dtype=tf.string)})
      return parsed

    return _parse(features)


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def map(self, features):
    return {"inputs": features["text"].numpy().decode(), "targets": features["text"].numpy().decode()}


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def map(self, data):
    return {
        "inputs": data[0]["inputs"],
        "targets": data[0]["targets"],
        "inputs_segmentation": data[1]["inputs"],
        "targets_segmentation": data[1]["targets"],
        "inputs_position": data[2]["inputs"],
        "targets_position": data[2]["targets"],
    }


@dataclasses.dataclass
class PadToMaxLength(grain.MapTransform):
  """Pads each input to the specified length"""

  def __init__(self, max_length):
    self.max_length = max_length

  def map(self, data):
    """map to each element"""

    def _pad(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
      return np.pad(x, pad_amount)

    data["inputs_segmentation"] = np.ones(data["inputs"].shape, dtype=np.int32)
    data["inputs_position"] = np.arange(data["inputs"].shape[0], dtype=np.int32)
    data["targets_segmentation"] = np.ones(data["targets"].shape, dtype=np.int32)
    data["targets_position"] = np.arange(data["targets"].shape[0], dtype=np.int32)
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


def shift_and_refine(x, axis=1):
  """Shift inputs, set segmentation to 0 when target element is 0.
  Replace EOS by 0 for packed inputs."""
  x["inputs"] = shift_right(x["inputs"], axis=axis)
  targets_nonzero = x["targets"] != 0
  x["inputs_segmentation"] *= targets_nonzero
  x["targets_segmentation"] *= targets_nonzero
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  x["inputs"] *= x["inputs_segmentation"] == shift_right(x["inputs_segmentation"], axis=axis)

  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Shift inputs and refine annotations."""

  def __init__(self, axis=1):
    self.axis = axis

  def map(self, data):
    return shift_and_refine(data, axis=self.axis)
