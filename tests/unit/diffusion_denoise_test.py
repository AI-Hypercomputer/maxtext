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

"""Tests model-independent block-diffusion rollout transitions."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.diffusion import denoise


def _target_logits(targets, vocab_size=32, high=12.0):
  return jax.nn.one_hot(targets, vocab_size, dtype=jnp.float32) * high


class DiffusionDenoiseTest(absltest.TestCase):

  def test_same_position_generates_partial_blocks_and_preserves_prompt(self):
    initial = jnp.asarray([[7, 8, 1, 1, 1, 1, 1, 0]], dtype=jnp.int32)
    expected = jnp.asarray([[7, 8, 12, 13, 14, 15, 16, 0]], dtype=jnp.int32)
    positions = jnp.arange(8, dtype=jnp.int32)[None, :]
    validity = initial != 0
    completion = jnp.asarray([[0, 0, 1, 1, 1, 1, 1, 0]], dtype=jnp.bool_)

    generated = denoise.low_confidence_generate(
        lambda _: _target_logits(expected),
        initial,
        positions,
        validity,
        completion,
        block_size=4,
        mask_id=31,
        logit_alignment="same_position",
        canvas_policy="all_masked",
    )

    np.testing.assert_array_equal(generated, expected)

  def test_shifted_seed_generates_anchors_before_each_block(self):
    initial = jnp.asarray([[5, 6, 1, 1, 1, 1, 1, 0]], dtype=jnp.int32)
    expected = jnp.asarray([[5, 6, 9, 10, 11, 12, 13, 0]], dtype=jnp.int32)
    positions = jnp.arange(8, dtype=jnp.int32)[None, :]
    validity = initial != 0
    completion = jnp.asarray([[0, 0, 1, 1, 1, 1, 1, 0]], dtype=jnp.bool_)

    generated = denoise.low_confidence_generate(
        lambda _: _target_logits(expected),
        initial,
        positions,
        validity,
        completion,
        block_size=4,
        mask_id=31,
        logit_alignment="shifted",
        canvas_policy="seed_and_mask",
    )

    np.testing.assert_array_equal(generated, expected)
    self.assertEqual(int(generated[0, 4]), 11)

  def test_forced_argmax_progress_completes_low_confidence_rows(self):
    initial = jnp.asarray([[4, 1, 1, 1]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    completion = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_)

    generated = denoise.low_confidence_generate(
        lambda canvas: jnp.zeros((*canvas.shape, 8), dtype=jnp.float32),
        initial,
        positions,
        jnp.ones_like(completion),
        completion,
        block_size=4,
        mask_id=7,
        logit_alignment="same_position",
        canvas_policy="all_masked",
        confidence_threshold=0.99,
    )

    self.assertFalse(bool(jnp.any(generated == 7)))
    np.testing.assert_array_equal(generated[:, :1], initial[:, :1])

  def test_mask_token_is_never_committed(self):
    initial = jnp.asarray([[6, 1, 1, 1]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    completion = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_)

    def mask_favoring_logits(canvas):
      logits = jnp.zeros((*canvas.shape, 8), dtype=jnp.float32)
      return logits.at[..., 7].set(100.0).at[..., 5].set(10.0)

    generated = denoise.low_confidence_generate(
        mask_favoring_logits,
        initial,
        positions,
        jnp.ones_like(completion),
        completion,
        block_size=4,
        mask_id=7,
        logit_alignment="same_position",
        canvas_policy="all_masked",
    )

    np.testing.assert_array_equal(generated, [[6, 5, 5, 5]])

  def test_mask_token_is_excluded_from_shifted_anchor_proposals(self):
    initial = jnp.asarray([[2, 3, 4, 5, 1, 1]], dtype=jnp.int32)
    positions = jnp.arange(6, dtype=jnp.int32)[None, :]
    completion = jnp.asarray([[0, 0, 0, 0, 1, 1]], dtype=jnp.bool_)

    def mask_favoring_logits(canvas):
      logits = jnp.zeros((*canvas.shape, 8), dtype=jnp.float32)
      return logits.at[..., 7].set(100.0).at[..., 6].set(10.0)

    generated = denoise.low_confidence_generate(
        mask_favoring_logits,
        initial,
        positions,
        jnp.ones_like(completion),
        completion,
        block_size=4,
        mask_id=7,
        logit_alignment="shifted",
        canvas_policy="seed_and_mask",
    )

    np.testing.assert_array_equal(generated, [[2, 3, 4, 5, 6, 6]])

  def test_rollout_is_jittable(self):
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    completion = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_)
    expected = jnp.asarray([[4, 5, 6, 7]], dtype=jnp.int32)

    generate = jax.jit(
        lambda initial: denoise.low_confidence_generate(
            lambda _: _target_logits(expected),
            initial,
            positions,
            jnp.ones_like(completion),
            completion,
            block_size=4,
            mask_id=31,
            logit_alignment="same_position",
            canvas_policy="all_masked",
        )
    )

    np.testing.assert_array_equal(generate(jnp.asarray([[4, 1, 1, 1]], dtype=jnp.int32)), expected)

  def test_logical_positions_drive_blocks_after_reordering(self):
    positions = jnp.asarray([[0, 4, 1, 5, 2, 6, 3, 7]], dtype=jnp.int32)
    expected = positions + 10
    initial = jnp.where(positions < 2, expected, 1)
    completion = positions >= 2

    generated = denoise.low_confidence_generate(
        lambda _: _target_logits(expected),
        initial,
        positions,
        jnp.ones_like(completion),
        completion,
        block_size=4,
        mask_id=31,
        logit_alignment="same_position",
        canvas_policy="all_masked",
    )

    np.testing.assert_array_equal(generated, expected)

  def test_forced_ties_are_invariant_to_physical_reordering(self):
    completion = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_)

    def generate(physical_order):
      positions = jnp.asarray(physical_order, dtype=jnp.int32)[None, :]
      initial = jnp.take_along_axis(jnp.asarray([[4, 1, 1, 1]], dtype=jnp.int32), positions, axis=1)
      physical_completion = jnp.take_along_axis(completion, positions, axis=1)

      def low_confidence_logits(canvas):
        committed_count = jnp.sum((canvas != 31) & physical_completion, axis=1)
        proposed_token = jnp.broadcast_to((10 + committed_count)[:, None], canvas.shape)
        return _target_logits(proposed_token, vocab_size=32, high=0.1)

      output = denoise.low_confidence_generate(
          low_confidence_logits,
          initial,
          positions,
          jnp.ones_like(physical_completion),
          physical_completion,
          block_size=4,
          mask_id=31,
          logit_alignment="same_position",
          canvas_policy="all_masked",
          confidence_threshold=0.99,
      )
      inverse_order = jnp.argsort(positions, axis=1)
      return jnp.take_along_axis(output, inverse_order, axis=1)

    np.testing.assert_array_equal(generate([0, 1, 2, 3]), generate([0, 3, 1, 2]))

  def test_rejects_non_suffix_and_shifted_origin_completion(self):
    positions = np.arange(4, dtype=np.int32)[None, :]
    validity = np.ones((1, 4), dtype=bool)

    with self.assertRaisesRegex(ValueError, "contiguous suffix"):
      denoise.validate_completion_suffix(positions, validity, np.asarray([[0, 1, 0, 1]], dtype=bool))
    with self.assertRaisesRegex(ValueError, "position zero"):
      denoise.validate_completion_suffix(
          positions,
          validity,
          np.asarray([[1, 1, 1, 1]], dtype=bool),
          shifted_seed=True,
      )

  def test_rejects_missing_duplicate_or_out_of_range_positions(self):
    validity = np.ones((1, 3), dtype=bool)
    completion = np.asarray([[0, 1, 1]], dtype=bool)

    for positions in (
        np.asarray([[1, 2, 3]], dtype=np.int32),
        np.asarray([[0, 1, 1]], dtype=np.int32),
        np.asarray([[0, 1, 5]], dtype=np.int32),
    ):
      with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "logical positions"):
        denoise.validate_completion_suffix(positions, validity, completion)

  def test_rejects_unsupported_contract_and_short_step_cap(self):
    tokens = jnp.ones((1, 4), dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mask = jnp.ones_like(tokens, dtype=jnp.bool_)

    with self.assertRaisesRegex(ValueError, "supports only"):
      denoise.low_confidence_generate(
          lambda _: _target_logits(tokens),
          tokens,
          positions,
          mask,
          mask,
          block_size=4,
          mask_id=31,
          logit_alignment="same_position",
          canvas_policy="seed_and_mask",
      )
    with self.assertRaisesRegex(ValueError, "at least block_size"):
      denoise.low_confidence_generate(
          lambda _: _target_logits(tokens),
          tokens,
          positions,
          mask,
          mask,
          block_size=4,
          mask_id=31,
          logit_alignment="same_position",
          canvas_policy="all_masked",
          max_denoise_steps=3,
      )
