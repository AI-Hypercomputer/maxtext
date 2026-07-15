# Copyright 2023–2026 Google LLC
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

"""Unit tests for Bagz data processing and ElasticIterator compatibility."""

import os
import sys
import tempfile
import unittest
from absl import flags
try:
  import pytest
  cpu_only = pytest.mark.cpu_only
except ImportError:
  cpu_only = lambda x: x
import bagz
import grain.python as grain
from grain.experimental import ElasticIterator

from grain.python import BagzDataSource

flags.FLAGS(sys.argv)


@cpu_only
class GrainBagzProcessingTest(unittest.TestCase):
  """Test BagzDataSource and compatibility with Grain MapDataset and ElasticIterator."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()
    self.bagz_path = os.path.join(self.test_dir.name, "test_data.bagz")
    self.num_records = 20

    writer = bagz.Writer(self.bagz_path)
    for i in range(self.num_records):
      writer.write(f"record_{i}".encode("utf-8"))
    writer.close()

  def tearDown(self):
    self.test_dir.cleanup()
    super().tearDown()

  def test_bagz_data_source_random_access(self):
    source = BagzDataSource([self.bagz_path])
    self.assertEqual(len(source), self.num_records)
    self.assertEqual(source[0].decode("utf-8"), "record_0")
    self.assertEqual(source[self.num_records - 1].decode("utf-8"), f"record_{self.num_records - 1}")

  def test_bagz_map_dataset(self):
    source = BagzDataSource([self.bagz_path])
    ds = grain.MapDataset.source(source).map(lambda x: x.decode("utf-8"))
    self.assertEqual(len(ds), self.num_records)
    self.assertEqual(ds[5], "record_5")

  def test_bagz_elastic_iterator_single_process(self):
    source = BagzDataSource([self.bagz_path])
    ds = grain.MapDataset.source(source).map(lambda x: x.decode("utf-8"))

    iter_ds = ElasticIterator(
        ds,
        global_batch_size=4,
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
    )
    it = iter(iter_ds)
    batch1 = next(it)
    batch2 = next(it)
    self.assertEqual(len(batch1), 4)
    self.assertEqual(len(batch2), 4)
    self.assertEqual(list(batch1), ["record_0", "record_1", "record_2", "record_3"])
    self.assertEqual(list(batch2), ["record_4", "record_5", "record_6", "record_7"])

  def test_bagz_elastic_iterator_multi_process(self):
    source = BagzDataSource([self.bagz_path])
    ds = grain.MapDataset.source(source).map(lambda x: x.decode("utf-8"))

    mp_options = grain.MultiprocessingOptions(num_workers=2, per_worker_buffer_size=1)
    iter_ds = ElasticIterator(
        ds,
        global_batch_size=4,
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        multiprocessing_options=mp_options,
    )
    it = iter(iter_ds)
    batch1 = next(it)
    batch2 = next(it)
    self.assertEqual(len(batch1), 4)
    self.assertEqual(len(batch2), 4)

  def test_bagz_get_datasets_integration(self):
    from maxtext.input_pipeline import grain_data_processing
    train_ds = grain_data_processing.get_datasets(
        data_file_pattern=self.bagz_path,
        data_file_type="bagz",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=1,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        elastic=False,
    )
    records = list(train_ds)
    self.assertEqual(len(records), self.num_records)
    self.assertEqual(records[0], b"record_0")


if __name__ == "__main__":
  unittest.main()
