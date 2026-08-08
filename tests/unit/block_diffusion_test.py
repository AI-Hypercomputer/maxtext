# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for model-independent block-diffusion primitives."""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.diffusion.block_diffusion import corruption
from maxtext.diffusion.block_diffusion import target_alignment


pytestmark = pytest.mark.cpu_only


def _make_batch():
  """Builds padded rows with a partial final diffusion block."""
  tokens = np.asarray(
      [
          [11, 12, 13, 14, 15, 16, 17, 18, 19, 0],
          [21, 22, 23, 24, 25, 26, 0, 0, 0, 0],
      ],
      dtype=np.int32,
  )
  positions = np.broadcast_to(np.arange(tokens.shape[1], dtype=np.int32), tokens.shape).copy()
  segmentation = (tokens != 0).astype(np.int32)
  return {
      "inputs": tokens.copy(),
      "targets": tokens.copy(),
      "inputs_segmentation": segmentation.copy(),
      "targets_segmentation": segmentation.copy(),
      "inputs_position": positions.copy(),
      "targets_position": positions.copy(),
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


class BlockDiffusionCorruptionTest(unittest.TestCase):

  def _apply(self, *, seed=0, **kwargs):
    """Applies corruption and reconstructs the input batch for assertions."""
    clean = _make_batch()
    result = corruption.corrupt_tokens(
        clean["inputs"],
        clean["targets_segmentation"] != 0,
        np.random.default_rng(seed),
        block_size=4,
        mask_id=99,
        **kwargs,
    )
    return clean | {
        "inputs": result.inputs,
        "corruption_mask": result.corruption_mask.astype(np.int32),
        "targets_loss_mask": result.targets_loss_mask.astype(np.int32),
    }

  def test_all_masked_contract_preserves_clean_targets_and_metadata(self):
    clean = _make_batch()
    output = self._apply(min_noise=1.0)

    for key in (
        "targets",
        "inputs_segmentation",
        "targets_segmentation",
        "inputs_position",
        "targets_position",
    ):
      np.testing.assert_array_equal(output[key], clean[key])
    np.testing.assert_array_equal(output["corruption_mask"], clean["targets_segmentation"])
    np.testing.assert_array_equal(output["targets_loss_mask"], output["corruption_mask"])
    self.assertTrue(np.all(output["inputs"][output["corruption_mask"] != 0] == 99))
    self.assertTrue(np.all(output["inputs"][clean["targets_segmentation"] == 0] == 0))

  def test_every_nonempty_partial_or_full_block_has_a_target(self):
    output = self._apply(seed=17)
    validity = output["targets_segmentation"] != 0
    loss_mask = output["targets_loss_mask"] != 0

    for row in range(validity.shape[0]):
      for start in range(0, validity.shape[1], 4):
        stop = min(start + 4, validity.shape[1])
        if np.any(validity[row, start:stop]):
          self.assertTrue(np.any(loss_mask[row, start:stop]), (row, start, stop))
    self.assertTrue(loss_mask[0, 8])

  def test_noise_rate_is_sampled_once_per_logical_block(self):
    rng = _ControlledRng()
    clean = _make_batch()
    result = corruption.corrupt_tokens(
        clean["inputs"],
        clean["targets_segmentation"] != 0,
        rng,
        block_size=4,
        mask_id=99,
    )

    self.assertEqual(rng.uniform_calls, 1)
    self.assertEqual(rng.random_calls, 2)
    self.assertEqual(int(result.targets_loss_mask[0, :4].sum()), 1)
    self.assertEqual(int(result.targets_loss_mask[0, 4:8].sum()), 4)

  def test_shifted_seed_contract_keeps_anchors_clean_and_supervises_later_anchors(self):
    output = self._apply(
        min_noise=1.0,
        logit_alignment="shifted",
        canvas_policy="seed_and_mask",
    )
    clean = _make_batch()

    self.assertFalse(output["corruption_mask"][:, ::4].any())
    expected_loss = clean["targets_segmentation"].copy()
    expected_loss[:, 0] = 0
    np.testing.assert_array_equal(output["targets_loss_mask"], expected_loss)
    np.testing.assert_array_equal(output["inputs"][:, ::4], clean["inputs"][:, ::4])

  def test_fixed_seed_is_deterministic(self):
    output_a = self._apply(seed=123)
    output_b = self._apply(seed=123)

    for key, value in output_a.items():
      np.testing.assert_array_equal(value, output_b[key])

  def test_all_invalid_row_has_no_corruption_or_loss(self):
    result = corruption.corrupt_tokens(
        np.zeros((1, 5), dtype=np.int32),
        np.zeros((1, 5), dtype=np.bool_),
        np.random.default_rng(0),
        block_size=4,
        mask_id=99,
    )

    self.assertFalse(result.corruption_mask.any())
    self.assertFalse(result.targets_loss_mask.any())
    np.testing.assert_array_equal(result.inputs, np.zeros((1, 5), dtype=np.int32))

  def test_single_token_blocks_are_supported_only_for_all_masked_canvas(self):
    all_masked = corruption.corrupt_tokens(
        np.asarray([[11, 12]], dtype=np.int32),
        np.ones((1, 2), dtype=np.bool_),
        np.random.default_rng(0),
        block_size=1,
        mask_id=99,
        min_noise=1.0,
    )

    np.testing.assert_array_equal(all_masked.inputs, [[99, 99]])
    with self.assertRaisesRegex(ValueError, "at least 2"):
      corruption.corrupt_tokens(
          np.asarray([[11, 12]], dtype=np.int32),
          np.ones((1, 2), dtype=np.bool_),
          np.random.default_rng(0),
          block_size=1,
          mask_id=99,
          logit_alignment="shifted",
          canvas_policy="seed_and_mask",
      )

  def test_invalid_configuration_or_shape_raises(self):
    invalid_kwargs = (
        {"block_size": 4, "mask_id": 99, "axis": 2},
        {"block_size": 0, "mask_id": 99},
        {"block_size": 4, "mask_id": -1},
        {"block_size": 4, "mask_id": 99, "min_noise": 0.0},
        {
            "block_size": 4,
            "mask_id": 99,
            "logit_alignment": "same_position",
            "canvas_policy": "seed_and_mask",
        },
    )
    for kwargs in invalid_kwargs:
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        corruption.corrupt_tokens(
            np.ones((1, 4), dtype=np.int32),
            np.ones((1, 4), dtype=np.bool_),
            np.random.default_rng(0),
            **kwargs,
        )

    batch = _make_batch()
    with self.assertRaisesRegex(ValueError, "identical shapes"):
      corruption.corrupt_tokens(
          batch["inputs"],
          batch["targets_segmentation"][:, :-1],
          np.random.default_rng(0),
          block_size=4,
          mask_id=99,
      )
    with self.assertRaisesRegex(ValueError, "nonempty"):
      corruption.corrupt_tokens(
          np.zeros((1, 0), dtype=np.int32),
          np.zeros((1, 0), dtype=np.bool_),
          np.random.default_rng(0),
          block_size=4,
          mask_id=99,
      )


class TargetAlignmentTest(unittest.TestCase):

  def test_same_position_is_identity_and_shifted_uses_previous_logit(self):
    logits = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)

    same_position = target_alignment.align_logits_to_targets(logits, "same_position")
    shifted = target_alignment.align_logits_to_targets(logits, "shifted")

    np.testing.assert_array_equal(same_position, logits)
    np.testing.assert_array_equal(shifted, logits[:, [0, 0, 1, 2], :])

  def test_shifted_alignment_follows_logical_positions_after_reordering(self):
    positions = jnp.asarray([[0, 3, 1, 2]], dtype=jnp.int32)
    logits = positions[..., None].astype(jnp.float32)

    shifted = target_alignment.align_logits_to_targets(
        logits,
        "shifted",
        positions=positions,
        validity_mask=jnp.ones_like(positions, dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(shifted[..., 0], [[0, 2, 0, 1]])

  def test_shifted_alignment_defaults_to_all_positions_valid(self):
    positions = jnp.asarray([[0, 3, 1, 2]], dtype=jnp.int32)
    logits = positions[..., None].astype(jnp.float32)

    shifted = target_alignment.align_logits_to_targets(logits, "shifted", positions=positions)

    np.testing.assert_array_equal(shifted[..., 0], [[0, 2, 0, 1]])

  def test_invalid_padding_positions_do_not_override_logical_position_zero(self):
    positions = jnp.asarray([[0, 3, 1, 2, 0, 0]], dtype=jnp.int32)
    validity = jnp.asarray([[1, 1, 1, 1, 0, 0]], dtype=jnp.bool_)
    logits = jnp.arange(6, dtype=jnp.float32).reshape(1, 6, 1)

    shifted = target_alignment.align_logits_to_targets(logits, "shifted", positions, validity)

    np.testing.assert_array_equal(shifted[0, :4, 0], [0, 3, 0, 2])

  def test_invalid_alignment_or_positions_raise(self):
    logits = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "Unsupported"):
      target_alignment.align_logits_to_targets(logits, "unknown")
    with self.assertRaisesRegex(ValueError, "positions must match"):
      target_alignment.align_logits_to_targets(logits, "shifted", positions=jnp.zeros((1, 3)))


if __name__ == "__main__":
  unittest.main()
