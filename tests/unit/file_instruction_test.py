# Copyright 2026 Google LLC
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

"""Unit tests for Grain FileInstruction extraction and dataset integration."""

from types import SimpleNamespace
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized

from maxtext.input_pipeline.data_processing_utils import apply_multiprocessing_and_prefetch
from maxtext.input_pipeline.grain_data_processing import (
    FileInstruction,
    _parse_mixture,
    extract_file_instructions,
)


class FileInstructionTest(parameterized.TestCase):

  def test_file_instruction_dataclass(self):
    fi = FileInstruction(filename="gs://test-bucket/data.arrayrecord", skip=0, take=100, examples_in_shard=100)
    self.assertEqual(fi.filename, "gs://test-bucket/data.arrayrecord")
    self.assertEqual(fi.skip, 0)
    self.assertEqual(fi.take, 100)
    self.assertEqual(fi.examples_in_shard, 100)

    d = fi.to_dict()
    self.assertEqual(
        d,
        {
            "filename": "gs://test-bucket/data.arrayrecord",
            "skip": 0,
            "take": 100,
            "examples_in_shard": 100,
        },
    )

    restored = FileInstruction.from_dict(d)
    self.assertEqual(restored, fi)

  @mock.patch("maxtext.input_pipeline.grain_data_processing.find_data_files")
  @mock.patch("grain.python.ArrayRecordDataSource")
  def test_extract_file_instructions(self, mock_ds_cls, mock_find_files):
    mock_find_files.return_value = ["gs://bucket/shard_000.arrayrecord", "gs://bucket/shard_001.arrayrecord"]

    mock_ri1 = mock.MagicMock()
    mock_ri1.filename = "gs://bucket/shard_000.arrayrecord"
    mock_ri1.start = 0
    mock_ri1.num_records = 50

    mock_ri2 = mock.MagicMock()
    mock_ri2.filename = "gs://bucket/shard_001.arrayrecord"
    mock_ri2.start = 0
    mock_ri2.num_records = 75

    mock_ds_instance = mock.MagicMock()
    mock_ds_instance._read_instructions = [mock_ri1, mock_ri2]  # pylint: disable=protected-access
    mock_ds_cls.return_value = mock_ds_instance

    instructions = extract_file_instructions("gs://bucket/*.arrayrecord")
    self.assertEqual(len(instructions), 2)
    self.assertEqual(instructions[0].filename, "gs://bucket/shard_000.arrayrecord")
    self.assertEqual(instructions[0].take, 50)
    self.assertEqual(instructions[1].filename, "gs://bucket/shard_001.arrayrecord")
    self.assertEqual(instructions[1].take, 75)

  def test_extract_file_instructions_passthrough(self):
    existing = (
        FileInstruction("gs://bucket/shard_000.arrayrecord", 0, 50, 50),
        FileInstruction("gs://bucket/shard_001.arrayrecord", 0, 75, 75),
    )
    result = extract_file_instructions(existing)
    self.assertEqual(result, existing)

  @mock.patch("maxtext.input_pipeline.grain_data_processing.find_data_files")
  @mock.patch("grain.python.ArrayRecordDataSource")
  def test_extract_file_instructions_list_input(self, mock_ds_cls, mock_find_files):
    mock_find_files.side_effect = lambda x: [x] if isinstance(x, str) else list(x)

    mock_ri = mock.MagicMock()
    mock_ri.filename = "gs://bucket/shard_000.arrayrecord"
    mock_ri.start = 0
    mock_ri.num_records = 100

    mock_ds_instance = mock.MagicMock()
    mock_ds_instance._read_instructions = [mock_ri]  # pylint: disable=protected-access
    mock_ds_cls.return_value = mock_ds_instance

    instructions = extract_file_instructions(["gs://bucket/shard_000.arrayrecord"])
    self.assertEqual(len(instructions), 1)
    self.assertEqual(instructions[0].filename, "gs://bucket/shard_000.arrayrecord")

  def test_parse_mixture_string(self):
    mixture = _parse_mixture("gs://bucket/data1.arrayrecord,0.7;gs://bucket/data2.arrayrecord,0.3")
    self.assertIsNotNone(mixture)
    patterns, weights = mixture
    self.assertEqual(patterns, ["gs://bucket/data1.arrayrecord", "gs://bucket/data2.arrayrecord"])
    self.assertEqual(weights, [0.7, 0.3])

  def test_parse_mixture_pre_extracted_tuple(self):
    inst1 = [FileInstruction("f1.arrayrecord", 0, 10, 10)]
    inst2 = [FileInstruction("f2.arrayrecord", 0, 20, 20)]
    mixture = _parse_mixture(((inst1, inst2), [0.4, 0.6]))
    self.assertIsNotNone(mixture)
    patterns, weights = mixture
    self.assertEqual(patterns, [inst1, inst2])
    self.assertEqual(weights, [0.4, 0.6])

  def test_parse_mixture_single_pattern_returns_none(self):
    self.assertIsNone(_parse_mixture("gs://bucket/*.arrayrecord"))
    self.assertIsNone(_parse_mixture(["gs://bucket/f1.arrayrecord", "gs://bucket/f2.arrayrecord"]))
    self.assertIsNone(_parse_mixture(None))

  def test_colocated_python_unexpected_worker_count_raises(self):
    mock_dataset = mock.MagicMock()
    config = SimpleNamespace(
        grain_use_elastic_iterator=False,
        colocated_python_data_input=True,
        elastic_enabled=False,
    )
    with self.assertRaisesRegex(ValueError, "Colocated python data input only supports `grain_worker_count=1`"):
      apply_multiprocessing_and_prefetch(mock_dataset, config, grain_worker_count=2, grain_per_worker_buffer_size=1)

    with self.assertRaisesRegex(ValueError, "Colocated python data input only supports `grain_worker_count=1`"):
      apply_multiprocessing_and_prefetch(mock_dataset, config, grain_worker_count=-1, grain_per_worker_buffer_size=1)


if __name__ == "__main__":
  absltest.main()
