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

import numpy as np

from maxtext.input_pipeline.input_pipeline_utils import (
    BlockDiffusionCorruption,
    PadOrTrimToMaxLength,
    SFTPromptMasking,
    compute_file_sharding,
)


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


def _make_block_diffusion_batch():
  """Builds a padded, completion-only SFT batch with a partial final block."""
  inputs = np.asarray(
      [
          [11, 12, 21, 22, 23, 24, 25, 26, 27, 0],
          [31, 41, 42, 43, 44, 45, 0, 0, 0, 0],
      ],
      dtype=np.int32,
  )
  targets = np.asarray(
      [
          [11, 12, 21, 22, 23, 24, 25, 26, 27, 0],
          [31, 41, 42, 43, 44, 45, 0, 0, 0, 0],
      ],
      dtype=np.int32,
  )
  completion_mask = np.asarray(
      [
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
      ],
      dtype=np.int32,
  )
  positions = np.broadcast_to(np.arange(inputs.shape[1], dtype=np.int32), inputs.shape).copy()
  return {
      "inputs": inputs,
      "targets": targets,
      "inputs_segmentation": (inputs != 0).astype(np.int32),
      "targets_segmentation": (targets != 0).astype(np.int32),
      "inputs_position": positions.copy(),
      "targets_position": positions.copy(),
      "completion_mask": completion_mask,
  }


class _ControlledRng:
  """Supplies distinct noise rates while making token draws deterministic."""

  def __init__(self):
    self.uniform_calls = 0
    self.random_calls = 0

  def uniform(self, low, high, size):
    del low, high
    self.uniform_calls += 1
    return np.asarray([[[0.1], [0.9], [0.1]], [[0.1], [0.9], [0.1]]]).reshape(size)

  def random(self, size):
    self.random_calls += 1
    return np.full(size, 0.5)


@pytest.mark.cpu_only
class BlockDiffusionCorruptionTest(unittest.TestCase):
  """Tests aligned block-diffusion corruption metadata."""

  def _apply(self, seed=0):
    transform = BlockDiffusionCorruption(block_size=4, mask_id=99, min_noise=1.0e-3, completion_only=True)
    return transform.random_map(_make_block_diffusion_batch(), np.random.default_rng(seed))

  def test_preserves_targets_and_segmentation(self):
    clean = _make_block_diffusion_batch()
    output = self._apply()

    for key in (
        "targets",
        "inputs_segmentation",
        "targets_segmentation",
        "inputs_position",
        "targets_position",
    ):
      np.testing.assert_array_equal(output[key], clean[key])
    np.testing.assert_array_equal(output["completion_mask"], clean["completion_mask"])
    self.assertTrue(np.all(output["targets_loss_mask"] <= output["completion_mask"]))
    self.assertFalse(np.shares_memory(output["completion_mask"], output["targets_loss_mask"]))
    np.testing.assert_array_equal(output["corruption_mask"], output["targets_loss_mask"])

    loss_positions = output["targets_loss_mask"] != 0
    self.assertTrue(np.all(output["inputs"][loss_positions] == 99))
    np.testing.assert_array_equal(output["inputs"][~loss_positions], clean["inputs"][~loss_positions])

  def test_every_eligible_full_or_partial_block_has_loss(self):
    output = self._apply(seed=17)
    completion_mask = output["completion_mask"] != 0
    loss_mask = output["targets_loss_mask"] != 0

    for row in range(completion_mask.shape[0]):
      for start in range(0, completion_mask.shape[1], 4):
        stop = min(start + 4, completion_mask.shape[1])
        if np.any(completion_mask[row, start:stop]):
          self.assertTrue(np.any(loss_mask[row, start:stop]), (row, start, stop))
    self.assertTrue(loss_mask[0, 8])

  def test_corruption_rate_is_sampled_per_block(self):
    rng = _ControlledRng()
    transform = BlockDiffusionCorruption(block_size=4, mask_id=99, completion_only=True)

    output = transform.random_map(_make_block_diffusion_batch(), rng)

    self.assertEqual(rng.uniform_calls, 1)
    self.assertEqual(rng.random_calls, 2)
    self.assertEqual(int(output["targets_loss_mask"][0, :4].sum()), 1)
    self.assertEqual(int(output["targets_loss_mask"][0, 4:8].sum()), 4)

  def test_deterministic_for_fixed_seed(self):
    output_a = self._apply(seed=123)
    output_b = self._apply(seed=123)
    for key, value in output_a.items():
      np.testing.assert_array_equal(value, output_b[key])

  def test_invalid_parameters_raise(self):
    with self.assertRaises(ValueError):
      BlockDiffusionCorruption(block_size=0, mask_id=99)
    with self.assertRaises(ValueError):
      BlockDiffusionCorruption(block_size=4, mask_id=99, min_noise=0.0)
    with self.assertRaisesRegex(ValueError, "requires seed_first_token"):
      BlockDiffusionCorruption(block_size=4, mask_id=99, include_seed_in_loss=True)

  def test_completion_only_requires_explicit_role_mask(self):
    clean = _make_block_diffusion_batch()
    del clean["completion_mask"]

    with self.assertRaisesRegex(ValueError, "explicit completion_mask"):
      self._apply_transform(clean)

  def test_completion_mask_does_not_replace_full_validity(self):
    clean = _make_block_diffusion_batch()
    transform = BlockDiffusionCorruption(block_size=4, mask_id=99, min_noise=1.0, completion_only=False)

    output = transform.random_map(clean, np.random.default_rng(0))

    np.testing.assert_array_equal(output["targets_segmentation"], clean["targets_segmentation"])
    np.testing.assert_array_equal(output["completion_mask"], clean["completion_mask"])
    self.assertTrue(np.all(output["targets_loss_mask"][clean["inputs_segmentation"] != 0]))
    self.assertTrue(np.any(output["targets_loss_mask"] > output["completion_mask"]))

  def test_all_padding_has_finite_zero_masks(self):
    clean = _make_block_diffusion_batch()
    for key in (
        "inputs",
        "targets",
        "inputs_segmentation",
        "targets_segmentation",
        "completion_mask",
    ):
      clean[key] = np.zeros_like(clean[key])

    output = self._apply_transform(clean)

    self.assertFalse(output["corruption_mask"].any())
    self.assertFalse(output["targets_loss_mask"].any())

  def test_shifted_seed_canvas_keeps_anchors_clean_but_supervised(self):
    clean = _make_block_diffusion_batch()
    transform = BlockDiffusionCorruption(
        block_size=4,
        mask_id=99,
        min_noise=1.0,
        completion_only=False,
        seed_first_token=True,
        include_seed_in_loss=True,
    )

    output = transform.random_map(clean, np.random.default_rng(0))

    self.assertFalse(output["corruption_mask"][:, ::4].any())
    expected_loss = clean["targets_segmentation"].copy()
    expected_loss[:, 0] = 0
    np.testing.assert_array_equal(output["targets_loss_mask"], expected_loss)
    self.assertTrue(output["targets_loss_mask"][0, 4])
    self.assertTrue(output["targets_loss_mask"][0, 8])

  def _apply_transform(self, batch):
    transform = BlockDiffusionCorruption(block_size=4, mask_id=99, completion_only=True)
    return transform.random_map(batch, np.random.default_rng(0))


@pytest.mark.cpu_only
class BlockDiffusionSftMaskTest(unittest.TestCase):
  """Tests role metadata before randomized corruption."""

  def test_target_aligned_sft_preserves_clean_tokens_and_completion_roles(self):
    transform = SFTPromptMasking(
        text_column_name="messages",
        completion_only=True,
        max_target_length=8,
        unk_id=0,
        target_aligned=True,
        emit_completion_mask=True,
    )
    element = {
        "messages": [[11, 12], [21, 22], [31], [41, 42]],
        "is_prompt": [True, False, True, False],
    }

    output = transform.map(element)
    padded = PadOrTrimToMaxLength(max_length=8, pad_id=0).map(output)

    np.testing.assert_array_equal(padded["targets"], padded["inputs"])
    np.testing.assert_array_equal(padded["completion_mask"], [0, 0, 1, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(padded["targets_segmentation"], [1, 1, 1, 1, 1, 1, 1, 0])
    self.assertNotIn("completion_mask_segmentation", padded)
    self.assertNotIn("completion_mask_position", padded)


if __name__ == "__main__":
  unittest.main()
