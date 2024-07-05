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

import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq

import math


class ParquetDataset(IterableDataset):
  def __init__(self, allocated_parquet_files, columns=None, batch_size=1000):
    super(ParquetDataset).__init__()

    self.allocated_parquet_files = allocated_parquet_files
    self.columns = columns
    self.batch_size = batch_size

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
      # Single-process data loading.
      print("Single-process data loading detected", flush=True)
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
