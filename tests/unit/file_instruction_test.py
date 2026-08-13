# Copyright 2023-2026 Google LLC
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

"""Tests validating the correctness and equivalence of FileInstruction in Grain."""

import json
import os
import tempfile
import unittest
from dataclasses import asdict, dataclass

import grain.python as grain
from array_record.python.array_record_module import ArrayRecordWriter
from maxtext.input_pipeline.grain_data_processing import FileInstruction, extract_file_instructions


class FileInstructionTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.num_shards = 4
    self.records_per_shard = 25
    self.file_paths = []
    self.expected_records = []

    # Generate synthetic arrayrecord shards
    for s in range(self.num_shards):
      fpath = os.path.join(self.temp_dir.name, f"shard_{s:03d}.array_record")
      self.file_paths.append(fpath)
      writer = ArrayRecordWriter(fpath, "group_size:1")
      for r in range(self.records_per_shard):
        rec = f"shard_{s}_record_{r:03d}".encode("utf-8")
        self.expected_records.append(rec)
        writer.write(rec)
      writer.close()

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_path_vs_file_instruction_equivalence(self):
    """Verify that ArrayRecordDataSource with FileInstruction produces identical data to path mode."""
    ds_path = grain.ArrayRecordDataSource(self.file_paths)
    self.assertEqual(len(ds_path), self.num_shards * self.records_per_shard)

    instructions = extract_file_instructions(self.file_paths)
    self.assertEqual(len(instructions), self.num_shards)

    ds_cached = grain.ArrayRecordDataSource(instructions)
    self.assertEqual(len(ds_cached), len(ds_path))

    for i in range(len(ds_path)):
      self.assertEqual(ds_path[i], ds_cached[i])
      self.assertEqual(ds_cached[i], self.expected_records[i])

  def test_json_serialization_roundtrip(self):
    """Verify serialization to/from JSON manifest."""
    instructions = extract_file_instructions(self.file_paths)

    json_str = json.dumps([inst.to_dict() for inst in instructions])
    loaded_dicts = json.loads(json_str)
    reconstructed_instructions = [FileInstruction.from_dict(d) for d in loaded_dicts]

    ds_reconstructed = grain.ArrayRecordDataSource(reconstructed_instructions)
    self.assertEqual(len(ds_reconstructed), len(self.expected_records))
    for i in range(len(ds_reconstructed)):
      self.assertEqual(ds_reconstructed[i], self.expected_records[i])

  def test_partial_shards_and_slices(self):
    """Verify FileInstructions with skip and take slices."""
    partial_instructions = [
        FileInstruction(filename=self.file_paths[0], skip=0, take=10, examples_in_shard=self.records_per_shard),
        FileInstruction(filename=self.file_paths[1], skip=5, take=10, examples_in_shard=self.records_per_shard),
    ]
    ds_partial = grain.ArrayRecordDataSource(partial_instructions)
    self.assertEqual(len(ds_partial), 20)

    expected_partial = (
        [f"shard_0_record_{r:03d}".encode("utf-8") for r in range(10)]
        + [f"shard_1_record_{r:03d}".encode("utf-8") for r in range(5, 15)]
    )
    for i in range(20):
      self.assertEqual(ds_partial[i], expected_partial[i])

  def test_grain_map_dataset_transforms(self):
    """Verify that grain.MapDataset works identically with FileInstruction."""
    instructions = [
        FileInstruction(filename=fp, skip=0, take=self.records_per_shard, examples_in_shard=self.records_per_shard)
        for fp in self.file_paths
    ]
    source = grain.ArrayRecordDataSource(instructions)
    map_ds = grain.MapDataset.source(source)

    # Apply batching and sharding
    sharded_ds_0 = map_ds[0::2]
    sharded_ds_1 = map_ds[1::2]
    self.assertEqual(len(sharded_ds_0), 50)
    self.assertEqual(len(sharded_ds_1), 50)

    # Verify records in sharded dataset
    for idx, orig_idx in enumerate(range(0, 100, 2)):
      self.assertEqual(sharded_ds_0[idx], self.expected_records[orig_idx])
    for idx, orig_idx in enumerate(range(1, 100, 2)):
      self.assertEqual(sharded_ds_1[idx], self.expected_records[orig_idx])

  def test_iter_dataset_batching_equivalence(self):
    """Verify that full pipeline with transforms and batching produces identical batches."""
    # Pipeline from path
    ds_path = grain.MapDataset.source(grain.ArrayRecordDataSource(self.file_paths))
    iter_path = (
        ds_path
        .map(lambda x: {"data": x.decode("utf-8")})
        .to_iter_dataset(read_options=grain.ReadOptions(prefetch_buffer_size=5))
        .batch(batch_size=8, drop_remainder=True)
    )

    # Pipeline from FileInstructions
    instructions = [
        FileInstruction(filename=fp, skip=0, take=self.records_per_shard, examples_in_shard=self.records_per_shard)
        for fp in self.file_paths
    ]
    ds_fi = grain.MapDataset.source(grain.ArrayRecordDataSource(instructions))
    iter_fi = (
        ds_fi
        .map(lambda x: {"data": x.decode("utf-8")})
        .to_iter_dataset(read_options=grain.ReadOptions(prefetch_buffer_size=5))
        .batch(batch_size=8, drop_remainder=True)
    )

    batches_path = list(iter_path)
    batches_fi = list(iter_fi)

    self.assertEqual(len(batches_path), len(batches_fi))
    for b1, b2 in zip(batches_path, batches_fi):
      self.assertEqual(list(b1["data"]), list(b2["data"]))


if __name__ == "__main__":
  unittest.main()
