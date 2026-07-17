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
    BlockDiffusionMasking,
    ShiftData,
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


_PAD_ID = 0
_MASK_ID = 99
_BD_SIZE = 4


def _make_sft_batch():
  """Synthetic post-PadOrTrim, post-Batch, pre-shift SFT batch (completion-only).

  Two examples, seq_len 8 (== 2 blocks of bd_size 4). Prompt and padding carry
  targets_segmentation == 0; the completion span carries targets_segmentation == 1.
  All token ids are nonzero (so pad_id 0 marks only prompt-mask/padding) and none
  equal _MASK_ID.
  """
  # ex0: prompt [11,12], response [21,22,23,24], pad [0,0]
  # ex1: prompt [31,32,33], response [41,42,43], pad [0,0]
  inputs = np.array(
      [
          [11, 12, 21, 22, 23, 24, 0, 0],
          [31, 32, 33, 41, 42, 43, 0, 0],
      ],
      dtype=np.int32,
  )
  # completion-only targets: prompt + padding zeroed, response = clean tokens (aligned).
  targets = np.array(
      [
          [0, 0, 21, 22, 23, 24, 0, 0],
          [0, 0, 0, 41, 42, 43, 0, 0],
      ],
      dtype=np.int32,
  )
  inputs_segmentation = (inputs != _PAD_ID).astype(np.int32)
  targets_segmentation = (targets != _PAD_ID).astype(np.int32)
  positions = np.broadcast_to(np.arange(inputs.shape[1], dtype=np.int32), inputs.shape).copy()
  return {
      "inputs": inputs,
      "targets": targets,
      "inputs_segmentation": inputs_segmentation,
      "targets_segmentation": targets_segmentation,
      "inputs_position": positions.copy(),
      "targets_position": positions.copy(),
  }


@pytest.mark.cpu_only
class BlockDiffusionMaskingTest(unittest.TestCase):
  """Tests for the masked-diffusion (CFT) data transform for block-diffusion SFT."""

  def _apply(self, seed):
    element = _make_sft_batch()
    transform = BlockDiffusionMasking(bd_size=_BD_SIZE, mask_id=_MASK_ID)
    return transform.random_map(element, np.random.default_rng(seed))

  def test_only_response_positions_are_masked(self):
    """(a) Prompt and padding are never masked; masks are a subset of the response span."""
    clean = _make_sft_batch()
    response = clean["targets_segmentation"] != 0
    for seed in range(50):
      out = self._apply(seed)
      masked = out["inputs"] == _MASK_ID
      # Every masked position is a response position.
      self.assertTrue(np.all(masked <= response), f"seed {seed}: masked outside response span")
      # Non-response inputs (prompt + padding) are byte-for-byte unchanged.
      np.testing.assert_array_equal(out["inputs"][~response], clean["inputs"][~response])

  def test_masked_inputs_equal_mask_id(self):
    """(b) Masked input positions hold mask_id, and some masking actually happens."""
    out = self._apply(seed=0)
    masked = out["targets_segmentation"] != 0
    self.assertGreater(int(masked.sum()), 0, "expected at least one masked position")
    self.assertTrue(np.all(out["inputs"][masked] == _MASK_ID))

  def test_targets_are_clean_and_aligned(self):
    """(c) Targets are the clean tokens, aligned (identical to input, not next-token shifted)."""
    clean = _make_sft_batch()
    for seed in range(10):
      out = self._apply(seed)
      np.testing.assert_array_equal(out["targets"], clean["targets"])
    # And explicitly NOT the AR next-token-shifted targets.
    shifted = np.array([row[1:].tolist() + [_PAD_ID] for row in clean["targets"]], dtype=np.int32)
    self.assertFalse(np.array_equal(out["targets"], shifted))

  def test_targets_segmentation_is_one_exactly_at_masked(self):
    """(d) targets_segmentation == 1 exactly at masked positions, 0 everywhere else."""
    out = self._apply(seed=0)
    masked = (out["inputs"] == _MASK_ID).astype(np.int32)
    np.testing.assert_array_equal(out["targets_segmentation"], masked)
    clean = _make_sft_batch()
    non_response = clean["targets_segmentation"] == 0
    self.assertTrue(np.all(out["targets_segmentation"][non_response] == 0))

  def test_inputs_segmentation_covers_real_tokens(self):
    """(e) inputs_segmentation is unchanged and covers all real prompt+response tokens."""
    clean = _make_sft_batch()
    out = self._apply(seed=0)
    np.testing.assert_array_equal(out["inputs_segmentation"], clean["inputs_segmentation"])
    # Masked positions are still attended to (segmentation stays 1 there).
    masked = out["inputs"] == _MASK_ID
    self.assertTrue(np.all(out["inputs_segmentation"][masked] == 1))

  def test_determinism_for_fixed_seed(self):
    """(f) A fixed RNG seed yields identical corruption; different seeds differ."""
    out_a = self._apply(seed=123)
    out_b = self._apply(seed=123)
    for key, value_a in out_a.items():
      np.testing.assert_array_equal(value_a, out_b[key], err_msg=f"non-deterministic key {key}")
    out_c = self._apply(seed=124)
    self.assertFalse(np.array_equal(out_a["inputs"], out_c["inputs"]))

  def test_flag_off_ar_path_shifts_targets(self):
    """(g) The AR path (ShiftData, used when enable_block_diffusion=False) shifts targets;

    block diffusion keeps them aligned. This confirms the two pipeline branches are
    genuinely different objectives and that leaving the flag off preserves the AR shift.
    """
    ar = ShiftData(ignored_ids=[_PAD_ID]).map(_make_sft_batch())
    clean = _make_sft_batch()
    # ShiftData shifts targets left by one (next-token prediction).
    np.testing.assert_array_equal(ar["targets"][:, :-1], clean["targets"][:, 1:])
    # Block diffusion leaves targets aligned (no shift).
    bd = self._apply(seed=0)
    np.testing.assert_array_equal(bd["targets"], clean["targets"])
    self.assertFalse(np.array_equal(ar["targets"], bd["targets"]))

  def test_invalid_bd_size_raises(self):
    """bd_size must be positive and must divide the sequence length."""
    with self.assertRaises(ValueError):
      BlockDiffusionMasking(bd_size=0, mask_id=_MASK_ID)
    with self.assertRaises(ValueError):
      # seq_len 8 is not divisible by bd_size 3.
      BlockDiffusionMasking(bd_size=3, mask_id=_MASK_ID).random_map(_make_sft_batch(), np.random.default_rng(0))


if __name__ == "__main__":
  unittest.main()
