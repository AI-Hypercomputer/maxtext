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

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from maxtext.input_pipeline.input_pipeline_utils import (
    BlockDiffusionCorruption,
    compute_file_sharding,
    PadOrTrimToMaxLength,
    SFTPromptMasking,
)


class BlockDiffusionPaddingTest(unittest.TestCase):
  """Checks that tokenizer padding never becomes diffusion supervision."""

  def test_corruption_dataclass_serializes_configuration(self):
    transform = BlockDiffusionCorruption(
        block_size=4,
        mask_id=99,
        min_noise=0.05,
        logit_alignment="shifted",
        canvas_policy="seed_and_mask",
        axis=0,
    )
    other = BlockDiffusionCorruption(block_size=8, mask_id=100)

    self.assertEqual(
        tuple(field.name for field in dataclasses.fields(transform)),
        (
            "block_size",
            "mask_id",
            "min_noise",
            "logit_alignment",
            "canvas_policy",
            "axis",
            "completion_only",
        ),
    )
    self.assertNotEqual(dataclasses.asdict(transform), dataclasses.asdict(other))
    self.assertIn("block_size=4", repr(transform))

  def test_nonzero_token_pad_id_keeps_metadata_padding_zero(self):
    clean = PadOrTrimToMaxLength(
        max_length=6,
        pad_id=7,
        config=SimpleNamespace(training_objective="block_diffusion"),
    ).map(
        {
            "inputs": np.asarray([11, 12, 13], dtype=np.int32),
            "targets": np.asarray([11, 12, 13], dtype=np.int32),
        }
    )
    corrupted = BlockDiffusionCorruption(block_size=4, mask_id=99, min_noise=1.0, axis=0).random_map(
        clean, np.random.default_rng(0)
    )

    padding = np.arange(6) >= 3
    np.testing.assert_array_equal(clean["inputs"][padding], 7)
    np.testing.assert_array_equal(clean["inputs_segmentation"][padding], 0)
    np.testing.assert_array_equal(clean["targets_segmentation"][padding], 0)
    np.testing.assert_array_equal(clean["inputs_position"][padding], 0)
    np.testing.assert_array_equal(clean["targets_position"][padding], 0)
    np.testing.assert_array_equal(corrupted["corruption_mask"][padding], 0)
    np.testing.assert_array_equal(corrupted["targets_loss_mask"][padding], 0)
    np.testing.assert_array_equal(corrupted["inputs"][padding], 7)

  def test_pad_valued_source_token_remains_valid(self):
    clean = PadOrTrimToMaxLength(
        max_length=4,
        pad_id=7,
        config=SimpleNamespace(training_objective="block_diffusion"),
    ).map(
        {
            "inputs": np.asarray([11, 7], dtype=np.int32),
            "targets": np.asarray([11, 7], dtype=np.int32),
        }
    )

    np.testing.assert_array_equal(clean["inputs"], [11, 7, 7, 7])
    np.testing.assert_array_equal(clean["inputs_segmentation"], [1, 1, 0, 0])
    np.testing.assert_array_equal(clean["targets_segmentation"], [1, 1, 0, 0])

  def test_causal_padding_preserves_legacy_metadata_pad_value(self):
    clean = PadOrTrimToMaxLength(
        max_length=4,
        pad_id=7,
        config=SimpleNamespace(training_objective="causal_lm"),
    ).map(
        {
            "inputs": np.asarray([11, 7], dtype=np.int32),
            "targets": np.asarray([11, 7], dtype=np.int32),
        }
    )

    np.testing.assert_array_equal(clean["inputs_segmentation"], [1, 0, 7, 7])
    np.testing.assert_array_equal(clean["targets_segmentation"], [1, 0, 7, 7])
    np.testing.assert_array_equal(clean["inputs_position"], [0, 1, 7, 7])
    np.testing.assert_array_equal(clean["targets_position"], [0, 1, 7, 7])

  def test_corruption_rejects_mismatched_batch_shapes(self):
    with self.assertRaisesRegex(ValueError, "must have identical shapes"):
      BlockDiffusionCorruption(block_size=4, mask_id=99).random_map(
          {
              "inputs": np.asarray([11, 12], dtype=np.int32),
              "targets": np.asarray([11, 12, 13], dtype=np.int32),
              "targets_segmentation": np.ones(2, dtype=np.int32),
          },
          np.random.default_rng(0),
      )


class BlockDiffusionSFTInputTest(unittest.TestCase):
  """Checks completion roles remain separate from corruption and validity."""

  def _prepared_batch(self):
    """Builds a block-aligned conversation with two completion spans."""
    clean = SFTPromptMasking(
        text_column_name="messages",
        completion_only=True,
        max_target_length=8,
        unk_id=0,
        training_objective="block_diffusion",
    ).map(
        {
            "messages": [[11, 12], [21, 22], [31], [41]],
            "is_prompt": [True, False, True, False],
        }
    )
    return PadOrTrimToMaxLength(
        max_length=8,
        pad_id=7,
        config=SimpleNamespace(training_objective="block_diffusion"),
    ).map(clean)

  def test_role_mask_preserves_clean_targets_and_padding(self):
    batch = self._prepared_batch()

    np.testing.assert_array_equal(batch["inputs"], batch["targets"])
    np.testing.assert_array_equal(batch["completion_mask"], [0, 0, 1, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(batch["targets_segmentation"], [1, 1, 1, 1, 1, 1, 0, 0])
    self.assertNotIn("completion_mask_segmentation", batch)
    self.assertNotIn("completion_mask_position", batch)

  def test_prompt_masking_rejects_unsupported_objective(self):
    with self.assertRaisesRegex(ValueError, "Unsupported training objective"):
      SFTPromptMasking(
          text_column_name="messages",
          completion_only=True,
          max_target_length=8,
          training_objective="unsupported",
      )

  def test_completion_only_corruption_never_supervises_prompt(self):
    batch = self._prepared_batch()
    output = BlockDiffusionCorruption(
        block_size=4,
        mask_id=99,
        min_noise=1.0,
        completion_only=True,
        axis=0,
    ).random_map(batch, np.random.default_rng(0))

    completion = batch["completion_mask"] != 0
    self.assertFalse(output["corruption_mask"][~completion].any())
    self.assertFalse(output["targets_loss_mask"][~completion].any())
    np.testing.assert_array_equal(output["targets"], batch["targets"])

  def test_completion_only_corruption_requires_role_mask(self):
    batch = self._prepared_batch()
    del batch["completion_mask"]

    with self.assertRaisesRegex(ValueError, "explicit completion_mask"):
      BlockDiffusionCorruption(
          block_size=4,
          mask_id=99,
          completion_only=True,
          axis=0,
      ).random_map(batch, np.random.default_rng(0))

  def test_completion_only_corruption_rejects_none_role_mask(self):
    batch = self._prepared_batch()
    batch["completion_mask"] = None

    with self.assertRaisesRegex(ValueError, "non-None completion_mask"):
      BlockDiffusionCorruption(
          block_size=4,
          mask_id=99,
          completion_only=True,
          axis=0,
      ).random_map(batch, np.random.default_rng(0))

  def test_completion_only_corruption_rejects_mismatched_role_mask_shape(self):
    batch = self._prepared_batch()
    batch["completion_mask"] = batch["completion_mask"][:-1]

    with self.assertRaisesRegex(ValueError, "completion_mask must match inputs shape"):
      BlockDiffusionCorruption(
          block_size=4,
          mask_id=99,
          completion_only=True,
          axis=0,
      ).random_map(batch, np.random.default_rng(0))

  def test_completion_only_corruption_rejects_role_mask_on_padding(self):
    batch = self._prepared_batch()
    batch["completion_mask"][6] = 1

    with self.assertRaisesRegex(ValueError, "subset of valid target positions"):
      BlockDiffusionCorruption(
          block_size=4,
          mask_id=99,
          completion_only=True,
          axis=0,
      ).random_map(batch, np.random.default_rng(0))

  def test_completion_only_corruption_rejects_future_prompt_in_block(self):
    batch = self._prepared_batch()
    batch["completion_mask"] = np.asarray([0, 0, 1, 1, 1, 0, 0, 0], dtype=np.int32)

    with self.assertRaisesRegex(ValueError, "prompt token after a completion token"):
      BlockDiffusionCorruption(
          block_size=4,
          mask_id=99,
          completion_only=True,
          axis=0,
      ).random_map(batch, np.random.default_rng(0))

  def test_full_token_corruption_treats_none_role_mask_as_absent(self):
    batch = self._prepared_batch()
    batch["completion_mask"] = None

    output = BlockDiffusionCorruption(
        block_size=4,
        mask_id=99,
        min_noise=1.0,
        axis=0,
    ).random_map(batch, np.random.default_rng(0))

    self.assertTrue(output["corruption_mask"][:6].all())
    self.assertFalse(output["corruption_mask"][6:].any())

  def test_causal_sft_contract_is_unchanged(self):
    batch = SFTPromptMasking(
        text_column_name="messages",
        completion_only=True,
        max_target_length=4,
        unk_id=0,
    ).map({"messages": [[11, 12], [21, 22]], "is_prompt": [True, False]})

    np.testing.assert_array_equal(batch["inputs"], [11, 12, 21, 22])
    np.testing.assert_array_equal(batch["targets"], [0, 0, 21, 22])
    self.assertNotIn("completion_mask", batch)


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
