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

"""Tests model-independent block-diffusion denoising transitions."""

from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.diffusion.block_diffusion import denoise


pytestmark = pytest.mark.cpu_only


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

  def test_heterogeneous_rows_use_independent_fallback_and_finish_at_different_steps(self):
    initial = jnp.asarray(
        [
            [10, 11, 12, 13, 14, 1, 1, 0],
            [20, 21, 22, 23, 1, 1, 1, 1],
            [30, 31, 32, 33, 34, 35, 1, 1],
        ],
        dtype=jnp.int32,
    )
    expected = jnp.asarray(
        [
            [10, 11, 12, 13, 14, 40, 41, 0],
            [20, 21, 22, 23, 42, 43, 44, 45],
            [30, 31, 32, 33, 34, 35, 46, 47],
        ],
        dtype=jnp.int32,
    )
    positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), initial.shape)
    validity = initial != 0
    completion = jnp.asarray(
        [
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ],
        dtype=jnp.bool_,
    )

    def heterogeneous_logits(_):
      target_logits = jax.nn.one_hot(expected, 64, dtype=jnp.float32)
      row_strength = jnp.asarray([12.0, 0.1, 0.1], dtype=jnp.float32)[:, None, None]
      return target_logits * row_strength

    generated = denoise.low_confidence_generate(
        heterogeneous_logits,
        initial,
        positions,
        validity,
        completion,
        block_size=4,
        mask_id=63,
        logit_alignment="same_position",
        canvas_policy="all_masked",
        confidence_threshold=0.99,
    )

    np.testing.assert_array_equal(generated, expected)

  def test_confidence_is_shift_invariant_without_materializing_softmax(self):
    initial = jnp.asarray([[4, 1, 1, 1]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    completion = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_)

    def logits_fn(canvas):
      committed_count = jnp.sum((canvas != 7) & completion, axis=1)
      proposed_token = jnp.broadcast_to((2 + committed_count)[:, None], canvas.shape)
      logits = jnp.full((*canvas.shape, 8), 1e7 - 1, dtype=jnp.float32)
      return logits.at[..., proposed_token].set(1e7)

    with mock.patch.object(denoise.jax.nn, "softmax", side_effect=AssertionError("softmax materialized")):
      generated = denoise.low_confidence_generate(
          logits_fn,
          initial,
          positions,
          jnp.ones_like(completion),
          completion,
          block_size=4,
          mask_id=7,
          logit_alignment="same_position",
          canvas_policy="all_masked",
          confidence_threshold=0.34,
      )

    np.testing.assert_array_equal(generated, [[4, 2, 3, 4]])

  def test_nonfinite_confidence_still_makes_fallback_progress(self):
    initial = jnp.asarray([[4, 1, 1, 1]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    completion = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_)

    generated = denoise.low_confidence_generate(
        lambda canvas: jnp.full((*canvas.shape, 8), jnp.nan, dtype=jnp.float32),
        initial,
        positions,
        jnp.ones_like(completion),
        completion,
        block_size=4,
        mask_id=7,
        logit_alignment="same_position",
        canvas_policy="all_masked",
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

  def test_generate_is_jittable(self):
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

  def test_host_validation_converts_concrete_jax_arrays_but_skips_abstract_values(self):
    value = jnp.ones((1, 2))
    np.testing.assert_array_equal(denoise._concrete_numpy(value), np.ones((1, 2)))  # pylint: disable=protected-access
    self.assertIsNone(
        denoise._concrete_numpy(jax.ShapeDtypeStruct((1, 2), jnp.float32))  # pylint: disable=protected-access
    )

    def assert_tracer_is_skipped(tracer):
      self.assertIsNone(denoise._concrete_numpy(tracer))  # pylint: disable=protected-access
      return tracer

    jax.make_jaxpr(assert_tracer_is_skipped)(value)

  def test_concrete_jax_arrays_run_host_validation(self):
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    validity = jnp.ones((1, 4), dtype=jnp.bool_)

    with self.assertRaisesRegex(ValueError, "contiguous suffix"):
      denoise.validate_completion_suffix(
          positions,
          validity,
          jnp.asarray([[0, 1, 0, 1]], dtype=jnp.bool_),
      )
    with self.assertRaisesRegex(ValueError, "logical positions"):
      denoise.validate_completion_suffix(
          jnp.asarray([[0, 1, 1, 3]], dtype=jnp.int32),
          validity,
          jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_),
      )
    with self.assertRaisesRegex(ValueError, "subset of valid token positions"):
      denoise.validate_completion_suffix(
          positions,
          jnp.asarray([[1, 1, 0, 0]], dtype=jnp.bool_),
          jnp.asarray([[0, 1, 1, 0]], dtype=jnp.bool_),
      )

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

  def test_rejects_completion_outside_validity(self):
    with self.assertRaisesRegex(ValueError, "subset of valid token positions"):
      denoise.validate_completion_suffix(
          np.arange(4, dtype=np.int32)[None, :],
          np.asarray([[1, 1, 0, 0]], dtype=bool),
          np.asarray([[0, 1, 1, 0]], dtype=bool),
      )

  def test_rejects_missing_duplicate_or_out_of_range_positions(self):
    validity = np.ones((1, 3), dtype=bool)
    completion = np.asarray([[0, 1, 1]], dtype=bool)

    for positions in (
        np.asarray([[0, 1, 3]], dtype=np.int32),
        np.asarray([[0, 1, 1]], dtype=np.int32),
    ):
      with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "logical positions"):
        denoise.validate_completion_suffix(positions, validity, completion)

    for positions in (
        np.asarray([[-1, 0, 1]], dtype=np.int32),
        np.asarray([[0, 1, 5]], dtype=np.int32),
    ):
      with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "physical sequence length"):
        denoise.validate_completion_suffix(positions, validity, completion)

  def test_host_validation_accepts_empty_fully_padded_and_valid_shifted_rows(self):
    denoise.validate_completion_suffix(
        np.empty((0, 3), dtype=np.int32),
        np.empty((0, 3), dtype=bool),
        np.empty((0, 3), dtype=bool),
    )
    denoise.validate_completion_suffix(
        np.arange(4, dtype=np.int32)[None, :],
        np.ones((1, 4), dtype=bool),
        np.asarray([[0, 1, 1, 1]], dtype=bool),
        shifted_seed=True,
    )
    denoise.validate_completion_suffix(
        np.arange(3, dtype=np.int32)[None, :],
        np.zeros((1, 3), dtype=bool),
        np.zeros((1, 3), dtype=bool),
        shifted_seed=True,
    )

  def test_shifted_validation_accepts_fully_padded_rows_in_a_mixed_batch(self):
    denoise.validate_completion_suffix(
        np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32),
        np.asarray([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool),
        np.asarray([[0, 1, 1, 1], [0, 0, 0, 0]], dtype=bool),
        shifted_seed=True,
    )

  def test_rejects_invalid_shapes(self):
    with self.assertRaisesRegex(ValueError, r"\[batch, length\]"):
      denoise.low_confidence_generate(
          _target_logits,
          jnp.ones((2,), dtype=jnp.int32),
          jnp.ones((2,), dtype=jnp.int32),
          jnp.ones((2,), dtype=jnp.bool_),
          jnp.ones((2,), dtype=jnp.bool_),
          block_size=2,
          mask_id=31,
          logit_alignment="same_position",
          canvas_policy="all_masked",
      )

    tokens = jnp.ones((1, 2), dtype=jnp.int32)
    for name, positions, validity, completion in (
        ("positions", jnp.ones((1, 3)), jnp.ones_like(tokens), jnp.ones_like(tokens)),
        ("validity_mask", tokens, jnp.ones((1, 3)), jnp.ones_like(tokens)),
        ("completion_mask", tokens, jnp.ones_like(tokens), jnp.ones((1, 3))),
    ):
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
        denoise.low_confidence_generate(
            _target_logits,
            tokens,
            positions,
            validity,
            completion,
            block_size=2,
            mask_id=31,
            logit_alignment="same_position",
            canvas_policy="all_masked",
        )

  def test_rejects_invalid_scalar_arguments(self):
    tokens = jnp.ones((1, 2), dtype=jnp.int32)
    positions = jnp.arange(2, dtype=jnp.int32)[None, :]
    mask = jnp.ones_like(tokens, dtype=jnp.bool_)
    invalid_arguments = (
        ({"block_size": 0}, "block_size must be positive"),
        ({"mask_id": -1}, "mask_id must be nonnegative"),
        ({"confidence_threshold": -0.1}, "confidence_threshold"),
        ({"confidence_threshold": 1.1}, "confidence_threshold"),
        ({"temperature": 0.0}, "temperature must be positive"),
    )

    for overrides, message in invalid_arguments:
      arguments = {
          "block_size": 2,
          "mask_id": 31,
          "logit_alignment": "same_position",
          "canvas_policy": "all_masked",
          **overrides,
      }
      with self.subTest(arguments=overrides), self.assertRaisesRegex(ValueError, message):
        denoise.low_confidence_generate(
            _target_logits,
            tokens,
            positions,
            mask,
            mask,
            **arguments,
        )

  def test_rejects_invalid_logits(self):
    tokens = jnp.ones((1, 2), dtype=jnp.int32)
    positions = jnp.arange(2, dtype=jnp.int32)[None, :]
    mask = jnp.ones_like(tokens, dtype=jnp.bool_)

    def generate(logits_fn, mask_id=3):
      return denoise.low_confidence_generate(
          logits_fn,
          tokens,
          positions,
          mask,
          mask,
          block_size=2,
          mask_id=mask_id,
          logit_alignment="same_position",
          canvas_policy="all_masked",
      )

    invalid_logits = (
        (lambda _: jnp.zeros((1, 2)), r"\[batch, length, vocab\]"),
        (lambda _: jnp.zeros((2, 2, 4)), r"\[batch, length, vocab\]"),
        (lambda _: jnp.zeros((1, 2, 1)), "at least two vocabulary entries"),
    )
    for logits_fn, message in invalid_logits:
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
        generate(logits_fn)

    with self.assertRaisesRegex(ValueError, "mask_id must satisfy"):
      generate(lambda _: jnp.zeros((1, 2, 4)), mask_id=4)

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

  def test_rejects_seed_and_mask_with_unit_block(self):
    tokens = jnp.ones((1, 2), dtype=jnp.int32)

    with self.assertRaisesRegex(ValueError, "at least 2"):
      denoise.low_confidence_generate(
          lambda _: _target_logits(tokens),
          tokens,
          jnp.arange(2, dtype=jnp.int32)[None, :],
          jnp.ones_like(tokens, dtype=jnp.bool_),
          jnp.asarray([[0, 1]], dtype=jnp.bool_),
          block_size=1,
          mask_id=31,
          logit_alignment="shifted",
          canvas_policy="seed_and_mask",
      )


if __name__ == "__main__":
  absltest.main()
