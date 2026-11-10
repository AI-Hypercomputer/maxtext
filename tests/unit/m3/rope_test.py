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

"""Tests for `maxtext.m3.core.rope`.

Four layers of coverage:
  * Mathematical properties that define RoPE, which hold independently of any
    particular implementation.
  * Closed-form reference values, computed from the RoPE definition rather than
    from another implementation. Property tests alone cannot pin the numerics:
    a subtly wrong rotation can still preserve norms and relative positions.
  * Bit-exact parity against the legacy `RotaryEmbedding`, so migrated models
    reproduce legacy numbers.
  * Argument validation, since `apply_rope` is the one place a silent shape
    broadcast would corrupt a model quietly rather than loudly.

The parity layer has a finite lifetime -- it goes away when legacy
`maxtext.layers` does. The reference layer is what pins the numerics after
that, so it must not be written in terms of the legacy implementation.

The legacy import here is deliberate and one-directional: the *test* depends on
both stacks to compare them. `maxtext.m3` itself must never import legacy code.
"""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.core.rope import apply_rope
from maxtext.layers.embeddings import RotaryEmbedding

_BATCH = 2
_LENGTH = 6
_HEADS = 3
_HEAD_DIM = 8


def _random_x(dtype=jnp.float32, shape=(_BATCH, _LENGTH, _HEADS, _HEAD_DIM)):
  return jax.random.normal(jax.random.key(0), shape, dtype=dtype)


def _positions(batch=_BATCH, length=_LENGTH):
  return jnp.broadcast_to(jnp.arange(length), (batch, length))


class ApplyRopeShapeTest(parameterized.TestCase):
  """Shape and dtype behaviour."""

  @parameterized.named_parameters(
      ("float32", jnp.float32),
      ("bfloat16", jnp.bfloat16),
  )
  def test_preserves_shape_and_dtype(self, dtype):
    """RoPE is a rotation: it changes values, never shape or dtype."""
    x = _random_x(dtype)
    out = apply_rope(x, _positions())
    self.assertEqual(out.shape, x.shape)
    self.assertEqual(out.dtype, dtype)

  def test_broadcasts_across_heads(self):
    """All heads at a given position get the same rotation."""
    x = jnp.tile(_random_x(shape=(_BATCH, _LENGTH, 1, _HEAD_DIM)), (1, 1, _HEADS, 1))
    out = apply_rope(x, _positions())
    for head in range(1, _HEADS):
      np.testing.assert_array_equal(out[:, :, 0, :], out[:, :, head, :])


class ApplyRopePropertyTest(parameterized.TestCase):
  """Properties that define RoPE, independent of implementation."""

  def test_position_zero_is_identity(self):
    """At position 0 every angle is 0, so the rotation is the identity."""
    x = _random_x()
    positions = jnp.zeros((_BATCH, _LENGTH), dtype=jnp.int32)
    np.testing.assert_allclose(apply_rope(x, positions), x, rtol=1e-6, atol=1e-6)

  def test_preserves_pairwise_norm(self):
    """Rotating a 2D pair preserves its norm, so |x| is unchanged per pair."""
    x = _random_x()
    out = apply_rope(x, _positions())

    x1, x2 = jnp.split(x, 2, axis=-1)
    o1, o2 = jnp.split(out, 2, axis=-1)
    np.testing.assert_allclose(
        np.asarray(o1**2 + o2**2),
        np.asarray(x1**2 + x2**2),
        rtol=1e-5,
        atol=1e-5,
    )

  @parameterized.named_parameters(
      ("distance_2", (5, 3), (12, 10), True),
      ("distance_4", (7, 3), (20, 16), True),
      ("different_distance", (5, 3), (12, 9), False),
  )
  def test_attention_score_depends_only_on_relative_position(self, pair_a, pair_b, expect_equal):
    """The defining RoPE property.

    <rope(q, m), rope(k, n)> is a function of (m - n) alone. Two position pairs
    with the same gap must yield the same score; a different gap must not.
    """
    # Slot 0 holds the query vector, slot 1 the key vector.
    qk = _random_x(shape=(1, 2, 1, _HEAD_DIM))

    def score(m, n):
      out = apply_rope(qk, jnp.asarray([[m, n]], dtype=jnp.int32))
      return float(jnp.sum(out[0, 0, 0] * out[0, 1, 0]))

    score_a, score_b = score(*pair_a), score(*pair_b)
    if expect_equal:
      self.assertAlmostEqual(score_a, score_b, places=4)
    else:
      self.assertNotAlmostEqual(score_a, score_b, places=4)

  def test_packed_positions_restart_independently(self):
    """Packed sequences restart positions; the same position gives the same rotation."""
    x = jnp.tile(_random_x(shape=(1, 1, 1, _HEAD_DIM)), (1, 4, 1, 1))
    # Two packed segments: positions 0,1 then 0,1 again.
    out = apply_rope(x, jnp.asarray([[0, 1, 0, 1]], dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(out[0, 0]), np.asarray(out[0, 2]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out[0, 1]), np.asarray(out[0, 3]), rtol=1e-6)

  def test_max_timescale_changes_rotation(self):
    """`max_timescale` is load bearing: Qwen3 uses 1e6, not the 1e4 default."""
    x = _random_x()
    positions = _positions()
    default = apply_rope(x, positions)
    long_context = apply_rope(x, positions, max_timescale=1_000_000.0)
    self.assertFalse(np.allclose(np.asarray(default), np.asarray(long_context), rtol=1e-3))


class ApplyRopeReferenceTest(parameterized.TestCase):
  """Closed-form reference values, independent of any other implementation.

  These outlive the legacy parity tests. They are written from the RoPE
  definition directly so that removing `maxtext.layers` cannot leave
  `apply_rope` numerically unpinned.
  """

  def test_matches_closed_form_small_case(self):
    """Hand-checkable case: head_dim=4, positions 0 and 1, default timescale.

    head_dim=4 gives half_dim=2 and timescale = 10_000 ** [0, 0.5] = [1, 100],
    so position p has angles [p, p / 100]. The rotation is then
        first half : cos(a) - sin(a)
        second half: cos(a) + sin(a)
    for an all-ones input.
    """
    x = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    out = apply_rope(x, jnp.asarray([[0, 1]], dtype=jnp.int32))

    angles = np.array([1.0, 0.01])
    expected_pos_1 = np.concatenate([np.cos(angles) - np.sin(angles), np.cos(angles) + np.sin(angles)])

    # Position 0: all angles are zero, so the input passes through untouched.
    np.testing.assert_allclose(np.asarray(out[0, 0, 0]), np.ones(4), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out[0, 1, 0]), expected_pos_1, rtol=1e-6)

  @parameterized.named_parameters(
      ("default_timescale", 10_000.0),
      ("qwen3_timescale", 1_000_000.0),
  )
  def test_matches_closed_form_general_case(self, max_timescale):
    """Full reference computation in float64, over a realistic tensor shape."""
    x = _random_x(jnp.float32)
    positions = _positions()
    out = np.asarray(apply_rope(x, positions, max_timescale=max_timescale), dtype=np.float64)

    # Reference built straight from the definition, in numpy float64.
    half = _HEAD_DIM // 2
    timescale = max_timescale ** (2 * np.arange(half) / _HEAD_DIM)
    angles = np.asarray(positions, dtype=np.float64)[:, :, None, None] / timescale
    cos, sin = np.cos(angles), np.sin(angles)

    ref = np.asarray(x, dtype=np.float64)
    r1, r2 = ref[..., :half], ref[..., half:]
    expected = np.concatenate([r1 * cos - r2 * sin, r2 * cos + r1 * sin], axis=-1)

    # float32 compute against a float64 reference.
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

  def test_pins_sign_convention_and_frequency_spacing(self):
    """Rotates unit vectors so each pair's exact angle is directly observable.

    head_dim=4 gives two pairs with timescales [1, 100], so at position p they
    rotate by p and p/100 radians respectively. Starting each pair at the unit
    vector (1, 0) makes the output literally (cos, sin) per pair, which pins
    both the direction of rotation and the geometric spacing of the
    frequencies.

    Two bug classes survive every property test in this file and are caught
    only here: rotating by -theta instead of +theta, and an off-by-one factor
    in the `fraction` exponent. Note the second requires at least two pairs to
    detect -- pair 0 has timescale `max_timescale ** 0 == 1` whatever the
    exponent is, so a single-pair case is blind to it.
    """
    # x1 = (1, 1), x2 = (0, 0): each pair starts at the unit vector (1, 0).
    x = jnp.asarray([[[[1.0, 1.0, 0.0, 0.0]]]], dtype=jnp.float32)

    for position in (1, 2, 3):
      out = np.asarray(apply_rope(x, jnp.asarray([[position]], dtype=jnp.int32))[0, 0, 0])
      angles = np.array([position, position / 100.0])
      np.testing.assert_allclose(out[:2], np.cos(angles), rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(out[2:], np.sin(angles), rtol=1e-6, atol=1e-6)


class ApplyRopeLegacyParityTest(parameterized.TestCase):
  """Bit-exact agreement with the legacy `RotaryEmbedding`.

  Migration-scoped: delete alongside legacy `maxtext.layers.embeddings`.
  `ApplyRopeReferenceTest` is what pins the numerics afterwards.
  """

  @parameterized.named_parameters(
      ("default_timescale", 10_000),
      ("qwen3_timescale", 1_000_000),
  )
  def test_matches_legacy_float32(self, max_timescale):
    x = _random_x(jnp.float32)
    positions = _positions()

    legacy = RotaryEmbedding(
        min_timescale=1,
        max_timescale=max_timescale,
        mesh=None,
        embedding_dims=_HEAD_DIM,
        cast_as_fprop_dtype=True,
        fprop_dtype=jnp.float32,
    )
    np.testing.assert_array_equal(
        np.asarray(apply_rope(x, positions, max_timescale=max_timescale)),
        np.asarray(legacy(x, positions)),
    )

  def test_matches_legacy_bfloat16(self):
    """bf16 is the training dtype, so parity must hold there too."""
    x = _random_x(jnp.bfloat16)
    positions = _positions()

    legacy = RotaryEmbedding(
        min_timescale=1,
        max_timescale=10_000,
        mesh=None,
        embedding_dims=_HEAD_DIM,
        cast_as_fprop_dtype=True,
        fprop_dtype=jnp.bfloat16,
    )
    np.testing.assert_array_equal(
        np.asarray(apply_rope(x, positions)),
        np.asarray(legacy(x, positions)),
    )


class ApplyRopeValidationTest(parameterized.TestCase):
  """Bad arguments must fail loudly rather than broadcast silently."""

  def test_rejects_non_rank_4_input(self):
    x = _random_x(shape=(_BATCH, _LENGTH, _HEAD_DIM))
    with self.assertRaisesRegex(ValueError, "rank 4|batch, length, heads, head_dim"):
      apply_rope(x, _positions())

  def test_rejects_odd_head_dim(self):
    x = _random_x(shape=(_BATCH, _LENGTH, _HEADS, 7))
    with self.assertRaisesRegex(ValueError, "even head_dim"):
      apply_rope(x, _positions())

  @parameterized.named_parameters(
      ("wrong_length", (_BATCH, _LENGTH + 1)),
      ("wrong_batch", (_BATCH + 1, _LENGTH)),
      ("already_expanded", (_BATCH, _LENGTH, 1, 1)),
  )
  def test_rejects_mismatched_positions(self, positions_shape):
    """A silently broadcastable positions array is the dangerous failure mode."""
    x = _random_x()
    positions = jnp.zeros(positions_shape, dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "positions"):
      apply_rope(x, positions)


if __name__ == "__main__":
  absltest.main()
