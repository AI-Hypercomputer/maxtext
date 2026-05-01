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

"""Unit tests for input_pipeline_utils."""

import pytest
import unittest

from maxtext.input_pipeline.input_pipeline_utils import compute_file_sharding


@pytest.mark.cpu_only
class ComputeFileShardingNormalCaseTest(unittest.TestCase):
  """file_count >= host_count: disjoint file subsets, no row sharding."""

  def test_even_split(self):
    # 8 files, 4 hosts → interleaved assignment, 2 files each
    file_slice, files_per_host, _ = compute_file_sharding(8, host_index=0, host_count=4)
    self.assertEqual(list(range(8)[file_slice]), [0, 4])
    self.assertEqual(files_per_host, 2)

    file_slice, files_per_host, _ = compute_file_sharding(8, host_index=1, host_count=4)
    self.assertEqual(list(range(8)[file_slice]), [1, 5])
    self.assertEqual(files_per_host, 2)

    file_slice, files_per_host, _ = compute_file_sharding(8, host_index=2, host_count=4)
    self.assertEqual(list(range(8)[file_slice]), [2, 6])
    self.assertEqual(files_per_host, 2)

    file_slice, files_per_host, _ = compute_file_sharding(8, host_index=3, host_count=4)
    self.assertEqual(list(range(8)[file_slice]), [3, 7])
    self.assertEqual(files_per_host, 2)

  def test_uneven_split(self):
    # 5 files, 4 hosts → host 0 gets an extra file
    file_slice, _, _ = compute_file_sharding(5, host_index=0, host_count=4)
    self.assertEqual(list(range(5)[file_slice]), [0, 4])

    file_slice, _, _ = compute_file_sharding(5, host_index=1, host_count=4)
    self.assertEqual(list(range(5)[file_slice]), [1])

    file_slice, _, _ = compute_file_sharding(5, host_index=2, host_count=4)
    self.assertEqual(list(range(5)[file_slice]), [2])

    file_slice, _, _ = compute_file_sharding(5, host_index=3, host_count=4)
    self.assertEqual(list(range(5)[file_slice]), [3])

  def test_single_host_gets_all_files(self):
    file_slice, files_per_host, _ = compute_file_sharding(8, host_index=0, host_count=1)
    self.assertEqual(list(range(8)[file_slice]), [0, 1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(files_per_host, 8)

  def test_no_row_shard_in_normal_case(self):
    for host_index in range(4):
      _, _, row_shard = compute_file_sharding(8, host_index, host_count=4)
      self.assertIsNone(row_shard)


class ComputeFileShardingUndersizedCaseTest(unittest.TestCase):
  """file_count < host_count: multiple hosts share a file, split by row."""

  def test_single_file_four_hosts(self):
    # All 4 hosts read the same file, each gets a quarter of the rows
    _, _, row_shard = compute_file_sharding(1, host_index=0, host_count=4)
    self.assertEqual(row_shard, (0, 4))  # row index 0 of 4

    _, _, row_shard = compute_file_sharding(1, host_index=1, host_count=4)
    self.assertEqual(row_shard, (1, 4))  # row index 1 of 4

    _, _, row_shard = compute_file_sharding(1, host_index=2, host_count=4)
    self.assertEqual(row_shard, (2, 4))  # row index 2 of 4

    _, _, row_shard = compute_file_sharding(1, host_index=3, host_count=4)
    self.assertEqual(row_shard, (3, 4))  # row index 3 of 4

  def test_three_files_eight_hosts(self):
    # 8 hosts round-robin across 3 files:
    # hosts 0,3,6 → file 0 (3 readers); hosts 1,4,7 → file 1 (3 readers); hosts 2,5 → file 2 (2 readers)
    expected = {
        # host_index: (file_indices, row_shard)
        0: ([0], (0, 3)),
        1: ([1], (0, 3)),
        2: ([2], (0, 2)),
        3: ([0], (1, 3)),
        4: ([1], (1, 3)),
        5: ([2], (1, 2)),
        6: ([0], (2, 3)),
        7: ([1], (2, 3)),
    }
    for host_index, (exp_files, exp_row_shard) in expected.items():
      file_slice, _, row_shard = compute_file_sharding(3, host_index, host_count=8)
      self.assertEqual(list(range(3)[file_slice]), exp_files, f"host {host_index} file assignment")
      self.assertEqual(row_shard, exp_row_shard, f"host {host_index} row shard")

  def test_no_row_shard_when_only_one_reader(self):
    # 2 files, 3 hosts: file 1 has only one reader (host 1) → no row split needed
    _, _, row_shard = compute_file_sharding(2, host_index=1, host_count=3)
    self.assertIsNone(row_shard)


if __name__ == "__main__":
  unittest.main()
