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

"""In-tree tests for the Gemma-4 vision clipped-linears (PR #4790).

Pure mechanism + validation tests: no checkpoint, no TPU. Cover the merge-critical invariants:
clip math vs jnp.clip, dtype/no-op, NaN/Inf/ordering hard-fail, exact 112/448 counts, optimizer
freeze mask polarity, and fail-closed contract guards.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.models import gemma4_vision as gv

jax.config.update("jax_platform_name", "cpu")


def _finite_cb(in_lo=-2.5, in_hi=2.5, out_lo=-4.0, out_hi=4.0):
  cb = gv._make_clip_state()
  cb.input_min.value = jnp.asarray(in_lo, jnp.float32)
  cb.input_max.value = jnp.asarray(in_hi, jnp.float32)
  cb.output_min.value = jnp.asarray(out_lo, jnp.float32)
  cb.output_max.value = jnp.asarray(out_hi, jnp.float32)
  return cb


class ClipMathTest(unittest.TestCase):

  def test_clip_in_matches_jnp_clip(self):
    cb = _finite_cb()
    x = (jax.random.normal(jax.random.PRNGKey(0), (5, 17)) * 6.0).astype(jnp.float32)
    self.assertTrue(bool(jnp.array_equal(gv._clip_in(x, cb), jnp.clip(x, -2.5, 2.5))))

  def test_clip_out_matches_jnp_clip(self):
    cb = _finite_cb()
    x = (jax.random.normal(jax.random.PRNGKey(1), (5, 17)) * 6.0).astype(jnp.float32)
    self.assertTrue(bool(jnp.array_equal(gv._clip_out(x, cb), jnp.clip(x, -4.0, 4.0))))

  def test_none_is_exact_noop(self):
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 8)).astype(jnp.float32)
    self.assertTrue(bool(jnp.array_equal(gv._clip_in(x, None), x)))
    self.assertTrue(bool(jnp.array_equal(gv._clip_out(x, None), x)))

  def test_dtype_preserved(self):
    cb = _finite_cb()
    x = jax.random.normal(jax.random.PRNGKey(3), (4, 4)).astype(jnp.bfloat16)
    self.assertEqual(gv._clip_in(x, cb).dtype, jnp.bfloat16)

  def test_bounds_are_stop_gradient(self):
    # The clamp bound is read through jax.lax.stop_gradient at the use-site, so gradient wrt the bound
    # value is exactly zero even when the clamp is active. Probe _clip_bound_value directly with a
    # lightweight bound holder to avoid mutating an nnx Param inside a traced function.
    class _B:
      def __init__(self, v):
        self.value = v

    def loss(bound_val):
      x = jnp.ones((4,), jnp.float32) * 10.0  # above the bound -> clamp active
      hi = gv._clip_bound_value(_B(bound_val), jnp.float32)
      return jnp.sum(jnp.clip(x, -100.0, hi))

    g = jax.grad(loss)(jnp.asarray(2.5, jnp.float32))
    self.assertEqual(float(g), 0.0)


class ValidateBoundsTest(unittest.TestCase):

  def test_accepts_finite_ordered(self):
    gv.validate_clip_bounds(_finite_cb(), "ok")  # no raise

  def test_rejects_nan_sentinel(self):
    with self.assertRaises(ValueError):
      gv.validate_clip_bounds(gv._make_clip_state(), "nan")

  def test_rejects_inf(self):
    cb = _finite_cb()
    cb.input_max.value = jnp.asarray(np.inf, jnp.float32)
    with self.assertRaises(ValueError):
      gv.validate_clip_bounds(cb, "inf")

  def test_rejects_min_gt_max(self):
    cb = _finite_cb(in_lo=3.0, in_hi=-3.0)  # input_min > input_max
    with self.assertRaises(ValueError):
      gv.validate_clip_bounds(cb, "order")

  def test_none_is_noop(self):
    self.assertIsNone(gv.validate_clip_bounds(None))


class FreezeMaskTest(unittest.TestCase):

  def test_freeze_mask_polarity_and_count(self):
    tree = {
        "vision_encoder": {
            "layer_0": {
                "attention": {
                    "q_clip": {"input_min": 1.0, "input_max": 1.0, "output_min": 1.0, "output_max": 1.0},
                    "query": {"kernel": 1.0},
                },
                "mlp": {"gate_clip": {"input_min": 1.0, "output_max": 1.0}, "wi_0": {"kernel": 1.0}},
            }
        },
        "decoder": {"layer_0": {"norm": {"scale": 1.0}}},
    }
    mask = gv.clip_optimizer_freeze_mask(tree)
    flat = jax.tree_util.tree_flatten_with_path(mask)[0]
    n_clip = 0
    for path, v in flat:
      is_clip = gv._is_clip_bound_path(path)
      # clip leaves -> False (frozen); everything else -> True (trainable)
      self.assertEqual(bool(v), (not is_clip))
      if is_clip:
        n_clip += 1
    self.assertEqual(n_clip, 6)  # q_clip(4) + gate_clip(2)


class SymbolsTest(unittest.TestCase):

  def test_expected_counts_constants(self):
    self.assertEqual(gv.EXPECTED_CLIP_PROJECTIONS, 112)
    self.assertEqual(gv.EXPECTED_CLIP_BOUNDS, 448)

  def test_public_symbols_present(self):
    for sym in ("validate_all_vision_clip_bounds", "clip_optimizer_freeze_mask",
                "Gemma4ClippedMlpBlock", "Gemma4Attention"):
      self.assertTrue(hasattr(gv, sym), sym)


if __name__ == "__main__":
  unittest.main()


class OptionSInvariantsTest(unittest.TestCase):
  """M7: Option-S sentinel integrity + placeholder==pooled count invariants (pure validators)."""

  def test_valid_positions_pass(self):
    pos = np.array([[[0, 0], [1, 0], [-1, -1], [-1, -1]]], dtype=np.int32)  # 2 valid + 2 full-sentinel
    ok, n_bad = gv.option_s_position_sentinel_ok(pos)
    self.assertTrue(ok)
    self.assertEqual(n_bad, 0)
    gv.validate_option_s_positions(pos)  # must not raise

  def test_mixed_sentinel_row_fails(self):
    for bad_row in ([-1, 5], [5, -1]):
      pos = np.array([[[0, 0], bad_row, [-1, -1]]], dtype=np.int32)
      ok, n_bad = gv.option_s_position_sentinel_ok(pos)
      self.assertFalse(ok, f"mixed row {bad_row} should be flagged")
      self.assertEqual(n_bad, 1)
      with self.assertRaises(ValueError):
        gv.validate_option_s_positions(pos, where="test")

  def test_other_negative_coord_fails(self):
    pos = np.array([[[0, 0], [-2, -2], [-1, -1]]], dtype=np.int32)  # -2 is not the sentinel
    ok, n_bad = gv.option_s_position_sentinel_ok(pos)
    self.assertFalse(ok)

  def test_pooled_matches_placeholder(self):
    mask = np.array([True, True, True, False, False])  # 3 valid pooled tokens
    ok, n_valid = gv.option_s_pooled_matches_placeholder(mask, 3)
    self.assertTrue(ok)
    self.assertEqual(n_valid, 3)

  def test_pooled_placeholder_mismatch_detected(self):
    mask = np.array([True, True, True, False, False])  # 3 valid
    ok, n_valid = gv.option_s_pooled_matches_placeholder(mask, 4)  # placeholder count 4 != 3
    self.assertFalse(ok)
    self.assertEqual(n_valid, 3)
