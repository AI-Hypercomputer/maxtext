"""
Copyright 2024 Google LLC

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

import random
import abc
from typing import Iterable

import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq


import max_logging

import math

class ParquetIterableDataset(abc.ABC, IterableDataset):
  """A base class for different Parquet file read strategies.
  
  Implementers must override the `_iter_impl` method.
  """
  def __init__(self, allocated_parquet_files: Iterable[str], columns=None, batch_size=1000):
    max_logging.log(f'Using {self.__class__.__name__} strategy.')
    max_logging.log(f'Allocated with the following data files: {allocated_parquet_files}.')
    self.allocated_parquet_files = allocated_parquet_files
    self.columns = columns
    self.batch_size = batch_size

  @abc.abstractmethod
  def _iter_impl(self, assigned_parquet_files: Iterable[str]) -> Iterable:
    """Returns an iterable of data for the assigned Parquet files.

    This method will be called via a `yield from` by this class's __iter__
    function.
    
    Args:
      assigned_parquet_files: The parquest files whos data should be returned as
        an iterable.
    
    Returns:
      An iterable that provides the data using a read strategy described by the
      implementing class's name.
    """
    pass

  def __iter__(self) -> Iterable:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      print("Single-process data loading detected", flush=True)
      assigned_parquet_files = self.allocated_parquet_files
    else:
      print("Multi-process data loading detected", flush=True)
      per_worker = int(
          math.ceil(
              len(self.allocated_parquet_files) / float(worker_info.num_workers)
          )
      )
      worker_id = worker_info.id
      start = worker_id * per_worker
      end = min(start + per_worker, len(self.allocated_parquet_files))
      assigned_parquet_files = self.allocated_parquet_files[start:end]

      yield from self._iter_impl(assigned_parquet_files)


class FileParallelRandomRead(ParquetIterableDataset):
  "File Parallel, Random Read implementation for Parquet files."""

  def _iter_impl(self, assigned_parquet_files: Iterable[str]) -> Iterable:
    """File Parallel Random Read iterator.

    Notes:
      - Row groups that are not a multiple of the batch size will return partial batches
        on the last read of the row group.
      - The reading of the row groups within a parquet file is random.
    """
    for single_parquest_file in assigned_parquet_files:
      # Explicitly disable buffering to better simulate random reading of row groups.
      parquet_file = pq.ParquetFile(single_parquest_file, buffer_size = 0, pre_buffer = 0)
      # Randomize the row groups
      randomized_row_groups = self._random_iterator(range(parquet_file.num_row_groups))
      for current_row_group in randomized_row_groups:
        max_logging.log(f"row_group {current_row_group} has {parquet_file.metadata.row_group(current_row_group).num_rows} rows")
        for batch in parquet_file.iter_batches(
          row_groups = [current_row_group],
          batch_size=self.batch_size,
          columns=self.columns
        ):
          yield from batch.to_pylist()

  def _random_iterator(self, itr: Iterable):
    shuffled = list(itr) 
    random.shuffle(shuffled)
    for element in shuffled:
        yield element


class FileParallelSequentialRead(ParquetIterableDataset):
  """File Parallel, Sequential Read implementation for Parquet files."""

  def _iter_impl(self, assigned_parquet_files: Iterable[str]) -> Iterable:
    """File Parallel, Sequential Read iterator."""
    for each_parquet_file in assigned_parquet_files:
      table = pq.ParquetFile(each_parquet_file)
      for batch in table.iter_batches(
          batch_size=self.batch_size, columns=self.columns
      ):
        yield from batch.to_pylist()
