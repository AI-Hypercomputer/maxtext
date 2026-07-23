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

"""Tests explicit target-loss masking in the pre-training loss."""

# pylint: disable=too-many-positional-arguments

from dataclasses import dataclass
import unittest
from unittest import mock

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.trainers.pre_train import train as pre_train
from maxtext.diffusion import scoring as diffusion_scoring


@dataclass
class _Config:
  """Configuration subset consumed by pre-training loss_fn."""

  micro_batch_size_to_train_on: int = 2
  micro_batch_size_to_eval_on: int = 2
  vocab_size: int = 8
  z_loss_multiplier: float = 0.0
  enable_dropout: bool = False
  use_multimodal: bool = False
  use_indexer: bool = False
  indexer_sparse_training: bool = False
  indexer_loss_scaling_factor: float = 0.0
  num_vocab_tiling: int = 1
  num_experts: int = 1
  routed_bias: bool = False
  routed_bias_update_rate: float = 0.0
  mtp_num_layers: int = 0
  mtp_eval_target_module: int = 0
  use_qk_clip: bool = False
  use_tunix_gradient_accumulation: bool = False
  gradient_accumulation_steps: int = 1
  shard_mode: int = 0
  debug_sharding: bool = False
  weight_sparsity_n: int = 0
  weight_sparsity_m: int = 0
  attention_type: str = "global"
  training_objective: str = "causal_lm"
  block_diffusion_logit_alignment: str = "same_position"
  block_diffusion_canvas_policy: str = "all_masked"
  block_diffusion_block_size: int = 4


class _UniformNnxDecoder(nnx.Module):
  """Returns uniform logits through the NNX call contract."""

  def __init__(self, vocab_size):
    self.vocab_size = vocab_size
    self.mesh = jax.make_mesh((1, 1, 1, 1), ("data", "fsdp", "expert", "context"))

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      encoder_images=None,
      encoder_image_masks=None,
      enable_dropout=False,
      decoder_target_tokens=None,
      decoder_target_mask=None,
  ):
    del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks
    del enable_dropout, decoder_target_tokens, decoder_target_mask
    return jnp.zeros((*decoder_input_tokens.shape, self.vocab_size), dtype=jnp.float32)


class _UniformLinenDecoder(nn.Module):
  """Returns uniform logits through the Linen call contract."""

  vocab_size: int
  mesh: object

  @nn.compact
  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      encoder_images=None,
      encoder_image_masks=None,
      enable_dropout=False,
      decoder_target_tokens=None,
      decoder_target_mask=None,
  ):
    del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks
    del enable_dropout, decoder_target_tokens, decoder_target_mask
    return jnp.zeros((*decoder_input_tokens.shape, self.vocab_size), dtype=jnp.float32)


def _make_data(include_loss_mask=True):
  """Builds a batch whose explicit loss mask differs from segmentation."""
  data = {
      "inputs": jnp.zeros((2, 4), dtype=jnp.int32),
      "inputs_position": jnp.broadcast_to(jnp.arange(4), (2, 4)),
      "inputs_segmentation": jnp.ones((2, 4), dtype=jnp.int32),
      "targets": jnp.zeros((2, 4), dtype=jnp.int32),
      "targets_segmentation": jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=jnp.int32),
      "completion_mask": jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=jnp.int32),
      "corruption_mask": jnp.asarray([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=jnp.int32),
  }
  if include_loss_mask:
    data["targets_loss_mask"] = jnp.asarray([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=jnp.int32)
  return data


class PreTrainLossMaskTest(unittest.TestCase):
  """Checks explicit masks in both Linen and NNX loss branches."""

  def setUp(self):
    super().setUp()
    self.config = _Config()
    self.per_token_xent = jnp.arange(1, 9, dtype=jnp.float32).reshape(2, 4)
    self.per_token_z_loss = self.per_token_xent / 10.0

  def _cross_entropy_patch(self):
    return mock.patch.object(
        pre_train.max_utils,
        "cross_entropy_with_logits",
        return_value=(self.per_token_xent, self.per_token_z_loss),
    )

  def _use_block_diffusion(self):
    self.config.attention_type = "block_diffusion"
    self.config.training_objective = "block_diffusion"

  def _assert_explicit_mask_result(self, loss, aux):
    expected_mask = _make_data()["targets_loss_mask"] != 0
    expected_xent = jnp.sum(self.per_token_xent * expected_mask)
    expected_z_loss = jnp.sum(self.per_token_z_loss * expected_mask) / 3.0
    self.assertEqual(int(aux["total_weights"]), 3)
    self.assertAlmostEqual(float(aux["xent_sum"]), float(expected_xent))
    self.assertAlmostEqual(float(loss), float(expected_xent / 3.0))
    self.assertAlmostEqual(float(aux["z_loss"]), float(expected_z_loss))

  def test_nnx_loss_uses_targets_loss_mask(self):
    self._use_block_diffusion()
    model = _UniformNnxDecoder(self.config.vocab_size)
    with self._cross_entropy_patch():
      loss, aux = pre_train.loss_fn(model, self.config, _make_data(), None, None, is_train=True)

    self._assert_explicit_mask_result(loss, aux)

  def test_linen_loss_uses_targets_loss_mask(self):
    self._use_block_diffusion()
    mesh = jax.make_mesh((1, 1, 1, 1), ("data", "fsdp", "expert", "context"))
    model = _UniformLinenDecoder(vocab_size=self.config.vocab_size, mesh=mesh)
    data = _make_data()
    variables = model.init(
        jax.random.key(0),
        data["inputs"],
        data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    with self._cross_entropy_patch():
      loss, aux = pre_train.loss_fn(model, self.config, data, jax.random.key(1), variables, is_train=True)

    self._assert_explicit_mask_result(loss, aux)

  def test_causal_fallback_uses_targets_segmentation(self):
    data = _make_data(include_loss_mask=False)
    model = _UniformNnxDecoder(self.config.vocab_size)
    with self._cross_entropy_patch():
      loss, aux = pre_train.loss_fn(model, self.config, data, None, None, is_train=True)

    expected_mask = data["targets_segmentation"] != 0
    expected_xent = jnp.sum(self.per_token_xent * expected_mask)
    self.assertEqual(int(aux["total_weights"]), 7)
    self.assertAlmostEqual(float(aux["xent_sum"]), float(expected_xent))
    self.assertAlmostEqual(float(loss), float(expected_xent / 7.0))

  def test_zero_explicit_mask_has_finite_zero_loss(self):
    self._use_block_diffusion()
    data = _make_data()
    data["targets_loss_mask"] = jnp.zeros_like(data["targets_loss_mask"])
    model = _UniformNnxDecoder(self.config.vocab_size)
    with self._cross_entropy_patch():
      loss, aux = pre_train.loss_fn(model, self.config, data, None, None, is_train=True)

    self.assertEqual(int(aux["total_weights"]), 0)
    self.assertEqual(float(aux["xent_sum"]), 0.0)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertEqual(float(loss), 0.0)

  def test_causal_loss_rejects_block_bidirectional_attention(self):
    self.config.attention_type = "block_diffusion"
    model = _UniformNnxDecoder(self.config.vocab_size)

    with self.assertRaisesRegex(ValueError, "would leak"):
      pre_train.loss_fn(model, self.config, _make_data(), None, None, is_train=True)

  def test_block_diffusion_requires_all_explicit_masks(self):
    self._use_block_diffusion()
    model = _UniformNnxDecoder(self.config.vocab_size)

    for mask_name in ("completion_mask", "corruption_mask", "targets_loss_mask"):
      with self.subTest(mask_name=mask_name):
        data = _make_data()
        del data[mask_name]
        with self.assertRaisesRegex(ValueError, mask_name):
          pre_train.loss_fn(model, self.config, data, None, None, is_train=True)

  def test_logit_alignment_matches_runtime_contract(self):
    logits = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)

    same_position = diffusion_scoring.align_logits_to_targets(logits, "same_position")
    shifted = diffusion_scoring.align_logits_to_targets(logits, "shifted")

    np.testing.assert_array_equal(same_position, logits)
    np.testing.assert_array_equal(shifted, logits[:, [0, 0, 1, 2], :])

  def test_shifted_alignment_uses_logical_positions_after_reordering(self):
    positions = jnp.asarray([[0, 3, 1, 2]], dtype=jnp.int32)
    logits = positions[..., None].astype(jnp.float32)

    shifted = diffusion_scoring.align_logits_to_targets(
        logits,
        "shifted",
        positions=positions,
        validity_mask=jnp.ones_like(positions, dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(shifted[..., 0], [[0, 2, 0, 1]])

  def test_seeded_shifted_canvas_excludes_only_sequence_origin(self):
    self.config.attention_type = "block_diffusion"
    self.config.training_objective = "block_diffusion"
    self.config.block_diffusion_logit_alignment = "shifted"
    self.config.block_diffusion_canvas_policy = "seed_and_mask"
    data = _make_data()
    data["targets_loss_mask"] = jnp.ones_like(data["targets_segmentation"])
    model = _UniformNnxDecoder(self.config.vocab_size)

    with self._cross_entropy_patch():
      _, aux = pre_train.loss_fn(model, self.config, data, None, None, is_train=True)

    self.assertEqual(int(aux["total_weights"]), 5)


if __name__ == "__main__":
  unittest.main()
