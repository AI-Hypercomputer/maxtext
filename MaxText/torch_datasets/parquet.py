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
from typing import Iterable

import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq


import max_logging

import math


class FileParallelRandomRead(IterableDataset):
  def __init__(self, allocated_parquet_files: Iterable[str], columns=None, batch_size=1000):
    super(ParquetDataset).__init__()
    max_logging.log("Using FileParallelRandomRead")
    # batch_size is basically chunk size
    self.allocated_parquet_files = allocated_parquet_files
    self.columns = columns
    self.batch_size = batch_size

  def __iter__(self):
    """File Parallel Random Read iterator.

    Some facts:
      - Parquet metadata for all row groups is at the beginning of the file which
        makes getting the metadata for row groups cheap.

    Some implementation details:
      - Row groups that are not a multiple of the batch size will return partial batches
        on the last read of the row group.
      - The reading of the row groups within a parquet file is random.
    """
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

    for single_parquest_file in assigned_parquet_files:
      parquet_file = pq.ParquetFile(single_parquest_file, buffer_size = 0, pre_buffer = 0)
      # Randomize the row groups
      randomized_row_groups = self._random_iterator(range(parquet_file.num_row_groups))
      for current_row_group in randomized_row_groups:
        max_logging.log(f"row_group {current_row_group} has {parquet_file.metadata.row_group(current_row_group).num_rows} rows")
        # TODO: Read batch size until all row groups have been read.
        # Currently, a partial batch may be returned for each row group.
        # Does this even matter since the method is returning a itr?
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



class ParquetDataset(IterableDataset):
  def __init__(self, allocated_parquet_files, columns=None, batch_size=1000):
    super(ParquetDataset).__init__()

    max_logging.log("Using ParquetDataset")

    self.allocated_parquet_files = allocated_parquet_files
    self.columns = columns
    self.batch_size = batch_size

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
      print("Single-process data loading detected", flush=True)
      # Single-process data loading.
      for each_parquet_file in self.allocated_parquet_files:
        table = pq.ParquetFile(each_parquet_file)
        for batch in table.iter_batches(
            batch_size=self.batch_size, columns=self.columns
        ):
          yield from batch.to_pylist()
    else:
      # Multi-process data loading.
      print("Multi-process data loading detected", flush=True)
      per_worker = int(
          math.ceil(
              len(self.allocated_parquet_files) / float(worker_info.num_workers)
          )
      )

      worker_id = worker_info.id
      start = worker_id * per_worker
      end = min(start + per_worker, len(self.allocated_parquet_files))

      for each_parquet_file in self.allocated_parquet_files[start:end]:
        table = pq.ParquetFile(each_parquet_file)
        for batch in table.iter_batches(
            batch_size=self.batch_size, columns=self.columns
        ):
          yield from batch.to_pylist()
