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

from maxtext.input_pipeline.input_pipeline_utils import BlockDiffusionCorruption, compute_file_sharding, PadOrTrimToMaxLength
from maxtext.multimodal import utils as mm_utils

import pytest

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline.input_pipeline_utils import (
    MegatronSplitInputsTargets,
    megatron_min_segment_length,
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
        ("block_size", "mask_id", "min_noise", "logit_alignment", "canvas_policy", "axis"),
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


class GenerateDocSegmentIdsTest(unittest.TestCase):
  """Direct tests for mmap EOD-aware segmentation."""

  def test_reset_attention_mask_keeps_eod_in_previous_segment(self):
    transform = input_pipeline_utils.GenerateDocSegmentIds(eod_id=99, reset_attention_mask=True)
    result = transform.map({"inputs": np.array([10, 99, 20, 21, 99, 30], dtype=np.int32)})

    np.testing.assert_array_equal(result["inputs_segmentation"], np.array([1, 1, 2, 2, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(result["inputs_position"], np.array([0, 1, 0, 1, 2, 0], dtype=np.int32))

  def test_no_attention_reset_can_mask_eod_loss(self):
    transform = input_pipeline_utils.GenerateDocSegmentIds(
        eod_id=99,
        reset_attention_mask=False,
        eod_mask_loss=True,
    )
    result = transform.map({"targets": np.array([10, 99, 20, 99], dtype=np.int32)})

    np.testing.assert_array_equal(result["targets_segmentation"], np.array([1, 0, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(result["targets_position"], np.array([0, 1, 2, 3], dtype=np.int32))

  def test_short_eod_segments_are_merged(self):
    transform = input_pipeline_utils.GenerateDocSegmentIds(
        eod_id=99,
        reset_attention_mask=True,
        min_segment_length=3,
    )
    result = transform.map({"inputs": np.array([10, 99, 20, 99, 30, 31, 32], dtype=np.int32)})

    np.testing.assert_array_equal(result["inputs_segmentation"], np.array([1, 1, 1, 1, 2, 2, 2], dtype=np.int32))
    np.testing.assert_array_equal(result["inputs_position"], np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int32))

  def test_empty_columns_get_empty_annotations(self):
    transform = input_pipeline_utils.GenerateDocSegmentIds(eod_id=99)
    result = transform.map({"inputs": np.array([], dtype=np.int32)})

    self.assertEqual(result["inputs_segmentation"].shape, (0,))
    self.assertEqual(result["inputs_position"].shape, (0,))


class MegatronSplitInputsTargetsTest(unittest.TestCase):
  """Direct tests for mmap_npy L+1 sample splitting."""

  def test_split_preserves_final_real_target_and_positions(self):
    transform = input_pipeline_utils.MegatronSplitInputsTargets(eod_id=99)
    result = transform.map({"text": np.array([10, 99, 20, 21, 99, 30], dtype=np.int32)})

    np.testing.assert_array_equal(result["inputs"], np.array([10, 99, 20, 21, 99], dtype=np.int32))
    np.testing.assert_array_equal(result["targets"], np.array([99, 20, 21, 99, 30], dtype=np.int32))
    np.testing.assert_array_equal(result["inputs_segmentation"], np.array([1, 1, 2, 2, 2], dtype=np.int32))
    np.testing.assert_array_equal(result["targets_segmentation"], np.ones(5, dtype=np.int32))
    np.testing.assert_array_equal(result["inputs_position"], np.array([0, 1, 0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(result["targets_position"], result["inputs_position"])

  def test_eod_mask_loss_masks_by_input_token(self):
    transform = input_pipeline_utils.MegatronSplitInputsTargets(eod_id=99, eod_mask_loss=True)
    result = transform.map({"text": np.array([10, 99, 20, 99, 30], dtype=np.int32)})

    np.testing.assert_array_equal(result["targets_segmentation"], np.array([1, 0, 1, 0], dtype=np.int32))

  def test_no_attnmask_dataset_id_disables_reset_for_that_sample(self):
    transform = input_pipeline_utils.MegatronSplitInputsTargets(
        eod_id=99,
        reset_attention_mask=True,
        no_attnmask_dataset_ids={7},
    )
    result = transform.map({"text": np.array([10, 99, 20, 30], dtype=np.int32), "dataset_id": np.int32(7)})

    np.testing.assert_array_equal(result["inputs_segmentation"], np.ones(3, dtype=np.int32))
    np.testing.assert_array_equal(result["inputs_position"], np.array([0, 1, 2], dtype=np.int32))

  def test_short_segments_are_merged_after_split(self):
    transform = input_pipeline_utils.MegatronSplitInputsTargets(
        eod_id=99,
        reset_attention_mask=True,
        min_segment_length=3,
    )
    result = transform.map({"text": np.array([10, 99, 20, 99, 30, 31, 32, 40], dtype=np.int32)})

    np.testing.assert_array_equal(result["inputs_segmentation"], np.array([1, 1, 1, 1, 2, 2, 2], dtype=np.int32))
    np.testing.assert_array_equal(result["inputs_position"], np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int32))


def _cfg(reset_attention_mask: bool, divisor: int, max_target_length: int = 4096):
  return SimpleNamespace(
      reset_attention_mask=reset_attention_mask,
      packing_max_segments_per_sample=divisor,
      max_target_length=max_target_length,
  )


class TestMegatronMinSegmentLength:
  """``megatron_min_segment_length`` returns ``max_target_length // divisor`` only when merging is active."""

  def test_default_divisor_matches_prior_hardcoded_25(self):
    # Prior behavior used a hardcoded 25; the default config value preserves that.
    cfg = _cfg(reset_attention_mask=True, divisor=25, max_target_length=4096)
    assert megatron_min_segment_length(cfg) == 4096 // 25

  @pytest.mark.parametrize(
      "divisor,max_target_length,expected",
      [
          (50, 4096, 4096 // 50),
          (10, 8192, 8192 // 10),
          (1, 4096, 4096),
          (100, 4097, 4097 // 100),  # integer division (truncates)
      ],
  )
  def test_custom_divisor(self, divisor, max_target_length, expected):
    cfg = _cfg(reset_attention_mask=True, divisor=divisor, max_target_length=max_target_length)
    assert megatron_min_segment_length(cfg) == expected

  @pytest.mark.parametrize("divisor", [0, -1, -25])
  def test_non_positive_divisor_disables_merging(self, divisor):
    cfg = _cfg(reset_attention_mask=True, divisor=divisor)
    assert megatron_min_segment_length(cfg) == 0

  @pytest.mark.parametrize("divisor", [0, 25, 50])
  def test_reset_attention_mask_false_returns_zero(self, divisor):
    # reset_attention_mask=False short-circuits regardless of divisor.
    cfg = _cfg(reset_attention_mask=False, divisor=divisor)
    assert megatron_min_segment_length(cfg) == 0


class MegatronSplitDatasetIdTest(unittest.TestCase):

  def _element(self, with_id=True):
    el = {"text": np.arange(9, dtype=np.int32)}  # seq_len = 8 after split
    if with_id:
      el["dataset_id"] = np.int32(3)
    return el

  def test_emits_per_token_dataset_id_when_enabled(self):
    t = MegatronSplitInputsTargets(eod_id=0, emit_dataset_id=True)
    result = t.map(self._element())
    self.assertIn("dataset_id", result)
    self.assertEqual(result["dataset_id"].shape, (8,))
    self.assertTrue(np.all(result["dataset_id"] == 3))
    self.assertEqual(result["dataset_id"].dtype, np.int32)

  def test_omits_dataset_id_when_disabled(self):
    t = MegatronSplitInputsTargets(eod_id=0, emit_dataset_id=False)
    self.assertNotIn("dataset_id", t.map(self._element()))

  def test_omits_when_element_has_no_dataset_id(self):
    t = MegatronSplitInputsTargets(eod_id=0, emit_dataset_id=True)
    self.assertNotIn("dataset_id", t.map(self._element(with_id=False)))
class PadOrTrimToMaxLengthMultimodalTest(unittest.TestCase):
  """Unit tests for PadOrTrimToMaxLength image padding behaviors."""

  def test_qwen_vision_padding_bypass_and_validation(self):
    dummy_output = mm_utils.PreprocessorOutput(pixel_values=np.zeros((1, 3, 224, 224)), num_images=1)

    # Registered Qwen vision models bypass image padding
    for vb in ["qwen3_vl", "qwen3_omni", "qwen3_5"]:
      transform = PadOrTrimToMaxLength(
          max_length=128, pad_id=0, config=SimpleNamespace(vision_encoder_block=vb, use_multimodal=True)
      )
      self.assertIs(transform._pad_image_and_mask(dummy_output), dummy_output)  # pylint: disable=protected-access

    # Unregistered Qwen model raises ValueError when use_multimodal=True
    unreg_transform = PadOrTrimToMaxLength(
        max_length=128,
        pad_id=0,
        config=SimpleNamespace(vision_encoder_block="unregistered", model_name="qwen3-future", use_multimodal=True),
    )
    with self.assertRaisesRegex(ValueError, "registered in `PadOrTrimToMaxLength`"):
      unreg_transform._pad_image_and_mask(dummy_output)  # pylint: disable=protected-access

    # Text-only Qwen model (use_multimodal=False) bypasses error and returns preprocessed_image
    text_transform = PadOrTrimToMaxLength(
        max_length=128,
        pad_id=0,
        config=SimpleNamespace(vision_encoder_block=None, model_name="qwen3-0.6b", use_multimodal=False),
    )
    self.assertIs(text_transform._pad_image_and_mask(dummy_output), dummy_output)  # pylint: disable=protected-access

    # None pixel_values raises ValueError
    with self.assertRaisesRegex(ValueError, "must have pixel_values"):
      unreg_transform._pad_image_and_mask(mm_utils.PreprocessorOutput(pixel_values=None))  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
