# Copyright 2023-2026 Google LLC
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


"""Unit tests for distillation metrics.

Covers:
  - KL direction (forward KL = teacher || student)
  - KL self-equivalence (KL(p || p) == 0 at any temperature)
  - Cross-entropy / perplexity numerics
  - (sum, count) aggregator is unbiased under uneven masks
  - Sharded (multi-device) verification of the same property
  - Label mask correctness for pad_id == 0 and pad_id != 0
  - T^2 scaling of soft loss
  - Feature-mapping loss (beta > 0 path) and layer_indices slicing
"""

import pytest

pytest.importorskip("optax")
pytest.importorskip("tunix")

pytestmark = [pytest.mark.post_training]

import unittest
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.trainers.post_train.distillation import distillation_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    vocab_size: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    alpha: float = 0.5,
    beta_feature: float = 0.0,
    layer_indices: Optional[List[int]] = None,
):
  """Builds a CombinedDistillationStrategy with no-op forward fns (we feed outputs directly)."""

  def noop(*_args, **_kwargs):
    return distillation_utils.DistillationForwardOutput(logits=jnp.zeros((1, 1, vocab_size)))

  return distillation_utils.CombinedDistillationStrategy(
      student_forward_fn=noop,
      teacher_forward_fn=noop,
      pad_id=pad_id,
      temperature=temperature,
      alpha=alpha,
      beta_feature=beta_feature,
      layer_indices=layer_indices,
      vocab_size=vocab_size,
      max_steps=1,
  )


def _make_labels(targets: jnp.ndarray, vocab_size: int, pad_id: int = 0) -> distillation_utils.Labels:
  del vocab_size  # Integer labels do not depend on the vocabulary size.
  mask = jnp.not_equal(targets, pad_id).astype(jnp.float32)
  return distillation_utils.Labels(targets=jnp.where(mask > 0, targets, 0).astype(jnp.int32), mask=mask)


def _mean(pair):
  """Unpacks a (sum, count) metric tuple into a mean value."""
  s, c = pair
  c_val = float(c)
  return float(s) / c_val if c_val > 0 else float(s)


def _has_4_devices() -> bool:
  return len(jax.devices()) >= 4


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class DistillationMetricsTest(unittest.TestCase):
  """Unit tests for CombinedDistillationStrategy metrics."""

  # --- 1. KL direction --------------------------------------------------

  def test_kl_direction_matches_forward_kl_teacher_student(self):
    """optax.kl_divergence(log_pred, target) == KL(target || pred). We use it
    with log_pred=log_student, target=teacher_probs => KL(teacher || student)."""
    rng = np.random.default_rng(0)
    vocab_size = 8
    s_logits = jnp.asarray(rng.normal(size=(1, 1, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(1, 1, vocab_size)).astype(np.float32))

    log_s = jax.nn.log_softmax(s_logits, axis=-1)
    t_p = jax.nn.softmax(t_logits, axis=-1)
    optax_kl = float(optax.kl_divergence(log_s, t_p)[0, 0])

    # Reference: forward KL(teacher || student) = sum_i t_i (log t_i - log s_i).
    log_t = np.asarray(jax.nn.log_softmax(t_logits, axis=-1)[0, 0])
    ref_fwd = float(np.sum(np.asarray(t_p[0, 0]) * (log_t - np.asarray(log_s[0, 0]))))
    np.testing.assert_allclose(optax_kl, ref_fwd, rtol=1e-5, atol=1e-6)

    # Reverse KL(student || teacher) should NOT match — assert we didn't
    # accidentally compute it in the wrong direction.
    s_p = jax.nn.softmax(s_logits, axis=-1)
    log_t_jax = jax.nn.log_softmax(t_logits, axis=-1)
    ref_rev = float(np.sum(np.asarray(s_p[0, 0]) * (np.asarray(log_s[0, 0]) - np.asarray(log_t_jax[0, 0]))))
    self.assertFalse(np.isclose(ref_fwd, ref_rev, atol=1e-3), "test logits chosen badly: fwd == rev KL")

  def test_kl_self_distillation_is_zero_at_any_temperature(self):
    """KL(p || p) == 0; with student==teacher logits, soft_loss must be 0 at any T."""
    vocab_size, batch_size, seq_len = 8, 2, 3
    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    out = distillation_utils.DistillationForwardOutput(logits=logits, out_projection_activations=None)
    targets = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size)

    for temperature in [0.5, 1.0, 2.0, 8.0]:
      strategy = _make_strategy(vocab_size, temperature=temperature, alpha=1.0)  # alpha=1 -> total = soft only
      total_loss, metrics = strategy.compute_loss(out, out, labels, step=None)
      # soft_loss is scaled by T^2, so per-position float32 noise gets amplified
      # by T^2. Scale tolerances with T^2 so the assertion is meaningful across T.
      t_sq = max(1.0, float(temperature)) ** 2
      np.testing.assert_allclose(float(total_loss), 0.0, atol=1e-4 * t_sq)
      soft_sum, _ = metrics["distill/soft_loss"]
      np.testing.assert_allclose(float(soft_sum), 0.0, atol=1e-3 * t_sq)

  # --- 2. Perplexity / CE numerics --------------------------------------

  def test_hard_loss_matches_manual_cross_entropy_and_perplexity_relation(self):
    vocab_size = 4
    s_logits = jnp.array([[[2.0, 1.0, 0.5, 0.1], [0.0, 3.0, 1.0, 0.5]]], dtype=jnp.float32)
    # Target token 0 is the pad_id so it would be masked. Use non-zero targets.
    targets = jnp.array([[3, 1]], dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size, pad_id=0)
    out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)

    strategy = _make_strategy(vocab_size, pad_id=0, alpha=0.0)  # alpha=0 -> total = hard only
    total_loss, metrics = strategy.compute_loss(out, out, labels, step=None)

    log_p = jax.nn.log_softmax(s_logits, axis=-1)
    manual_ce = -float(log_p[0, 0, 3] + log_p[0, 1, 1]) / 2.0
    np.testing.assert_allclose(float(total_loss), manual_ce, rtol=1e-5)

    # Perplexity sanity: exp(hard_loss) finite and >= 1 for non-degenerate logits.
    ppl = float(np.exp(_mean(metrics["distill/hard_loss"])))
    self.assertGreaterEqual(ppl, 1.0)
    self.assertTrue(np.isfinite(ppl))

  # --- 3. Label mask: pad_id == 0 and pad_id != 0 -----------------------

  def test_label_mask_excludes_pad_tokens_pad0(self):
    self._run_label_mask_excludes_pad(pad_id=0)

  def test_label_mask_excludes_pad_tokens_pad7(self):
    self._run_label_mask_excludes_pad(pad_id=7)

  def _run_label_mask_excludes_pad(self, pad_id):
    """Asserts pad positions are zeroed in labels and excluded from hard loss."""
    vocab_size = 8
    strategy = _make_strategy(vocab_size, pad_id=pad_id, alpha=0.0)
    # second token is padding
    targets = jnp.array([[1, pad_id]], dtype=jnp.int32)
    labels = strategy.create_labels(targets)
    # the padded position must be masked out (and its target replaced by 0)
    np.testing.assert_array_equal(np.asarray(labels.mask), np.array([[1.0, 0.0]]))
    np.testing.assert_array_equal(np.asarray(labels.targets), np.array([[1, 0]]))

    s_logits = jnp.array([[[5.0, -5.0, 0, 0, 0, 0, 0, 0], [-5.0, 5.0, 0, 0, 0, 0, 0, 0]]], dtype=jnp.float32)
    out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    total_loss, metrics = strategy.compute_loss(out, out, labels, step=None)

    _, hard_cnt = metrics["distill/hard_loss"]
    np.testing.assert_allclose(float(hard_cnt), 1.0, atol=1e-6)  # only 1 valid token
    # Total loss must equal CE on token 0 only, ignoring padded position.
    log_p = jax.nn.log_softmax(s_logits, axis=-1)
    expected = -float(log_p[0, 0, 1])  # target = 1
    np.testing.assert_allclose(float(total_loss), expected, rtol=1e-5)

  def test_create_labels_masks_packed_segmentation(self):
    """Positions where targets_segmentation == 0 must be zeroed even when the target token is non-pad."""
    vocab_size = 8
    strategy = _make_strategy(vocab_size, pad_id=0, alpha=0.0)
    # Bin layout: doc1 at [0,1], doc2 at [2], in-bin pad at [3]. All targets non-pad.
    targets = jnp.array([[1, 2, 3, 1]], dtype=jnp.int32)
    targets_segmentation = jnp.array([[1, 1, 2, 0]], dtype=jnp.int32)

    labels_packed = strategy.create_labels(targets, targets_segmentation=targets_segmentation)
    labels_unpacked = strategy.create_labels(targets)

    self.assertEqual(float(labels_packed.mask[0, 3]), 0.0)
    self.assertEqual(float(labels_unpacked.mask[0, 3]), 1.0)
    for pos in (0, 1, 2):
      self.assertEqual(float(labels_packed.mask[0, pos]), float(labels_unpacked.mask[0, pos]))
      self.assertEqual(int(labels_packed.targets[0, pos]), int(labels_unpacked.targets[0, pos]))

  # --- 4. Temperature^2 scaling of soft loss ----------------------------

  def test_soft_loss_scales_with_temperature_squared_in_high_T_limit(self):
    """As T grows, soft_loss = T^2 * KL(softmax(t/T) || softmax(s/T)) should
    approach a finite non-zero value (Hinton scaling). Verify it doesn't collapse to 0."""
    vocab_size, batch_size, seq_len = 16, 2, 4
    rng = np.random.default_rng(2)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    t_out = distillation_utils.DistillationForwardOutput(logits=t_logits, out_projection_activations=None)
    targets = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size)

    losses = []
    for temperature in [1.0, 4.0, 16.0]:
      strategy = _make_strategy(vocab_size, temperature=temperature, alpha=1.0)
      _, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)
      losses.append(_mean(metrics["distill/soft_loss"]))

    # All positive and bounded; high-T value should not be drastically smaller than T=1.
    for value in losses:
      self.assertGreater(value, 0)
      self.assertTrue(np.isfinite(value))
    # T^2 scaling keeps high-T loss within an order of magnitude of T=1 in this regime.
    self.assertGreater(losses[-1], 0.1 * losses[0])

  def test_kl_T1_metric_is_temperature_invariant_when_logged(self):
    """distill/kl_div_T1 should be the same regardless of self.temperature."""
    vocab_size, batch_size, seq_len = 8, 1, 2
    rng = np.random.default_rng(3)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    t_out = distillation_utils.DistillationForwardOutput(logits=t_logits, out_projection_activations=None)
    targets = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size)

    values = []
    for temperature in [1.0, 4.0, 16.0]:
      strategy = _make_strategy(vocab_size, temperature=temperature, alpha=1.0)
      _, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)
      values.append(_mean(metrics["distill/kl_div_T1"]))
    np.testing.assert_allclose(values[0], values[1], rtol=1e-5)
    np.testing.assert_allclose(values[0], values[2], rtol=1e-5)

  # --- 5. (sum, count) aggregator: unbiased under uneven masks ----------

  def test_weighted_mean_aggregator_token_weighted(self):
    """The aggregator should compute sum(sums) / sum(counts), not mean of per-step means."""
    pairs = [
        (jnp.array(8.0), jnp.array(8.0)),  # 8 valid tokens, mean 1.0
        (jnp.array(5.0), jnp.array(1.0)),  # 1 valid token, value 5.0
        (jnp.array(8.0), jnp.array(4.0)),  # 4 valid tokens, mean 2.0
        (jnp.array(8.0), jnp.array(4.0)),  # 4 valid tokens, mean 2.0
    ]
    # mean-of-means would be (1+5+2+2)/4 = 2.5 (biased)
    # token-weighted: 29/17 ≈ 1.7058 (correct)
    np.testing.assert_allclose(distillation_utils.weighted_mean(pairs), 29.0 / 17.0, rtol=1e-6)

  def test_weighted_mean_aggregator_handles_empty_and_zero_count(self):
    self.assertEqual(distillation_utils.weighted_mean([]), 0.0)
    self.assertEqual(distillation_utils.weighted_mean([(jnp.array(0.0), jnp.array(0.0))]), 0.0)

  # --- 6. Multi-device sharded version ----------------------------------

  @unittest.skipIf(not _has_4_devices(), "Requires ≥4 devices.")
  def test_sum_count_unbiased_under_sharded_uneven_masks(self):
    """Across 4 devices with wildly different valid-token counts,
    mean-of-per-device-means is biased; token-weighted (sum/count) is correct."""
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ("data",))

    # Per-device (B=2, T=4). Layout: 4 shards along the data axis.
    # Device 0: 8 valid tokens, value 1.0  -> per-device mean 1.0
    # Device 1: 1 valid token,  value 5.0  -> per-device mean 5.0
    # Device 2: 4 valid tokens, value 2.0  -> per-device mean 2.0
    # Device 3: 4 valid tokens, value 2.0  -> per-device mean 2.0
    losses = jnp.array(
        [
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
            [[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    masks = jnp.array(
        [
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0]],
        ],
        dtype=jnp.float32,
    )

    sharding = NamedSharding(mesh, P("data"))
    losses = jax.device_put(losses, sharding)
    masks = jax.device_put(masks, sharding)

    @jax.jit
    def biased_mean_of_means(losses, masks):
      per_dev_sum = jnp.sum(losses * masks, axis=(1, 2))
      per_dev_cnt = jnp.maximum(jnp.sum(masks, axis=(1, 2)), 1.0)
      return jnp.mean(per_dev_sum / per_dev_cnt)

    @jax.jit
    def unbiased_token_weighted(losses, masks):
      return jnp.sum(losses * masks) / jnp.sum(masks)

    np.testing.assert_allclose(float(biased_mean_of_means(losses, masks)), 2.5, rtol=1e-5)
    np.testing.assert_allclose(float(unbiased_token_weighted(losses, masks)), 29.0 / 17.0, rtol=1e-5)

  @unittest.skipIf(not _has_4_devices(), "Requires ≥4 devices.")
  def test_compute_loss_sums_aggregate_correctly_across_shards(self):
    """Run compute_loss on a sharded batch; verify weighted_mean over per-shard
    (sum, count) outputs matches the global mean computed on the full batch."""
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ("data",))

    vocab_size, batch_size, seq_len = 8, 4, 4  # B=4 so each shard gets one example
    rng = np.random.default_rng(7)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    # Different valid-token counts per row (shard) to exercise the bias path.
    targets = jnp.array(
        [
            [1, 2, 3, 4],
            [1, 0, 0, 0],
            [1, 2, 0, 0],
            [1, 2, 3, 0],
        ],
        dtype=jnp.int32,
    )
    labels = _make_labels(targets, vocab_size, pad_id=0)

    strategy = _make_strategy(vocab_size, alpha=0.0)  # hard-loss only for clarity
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    t_out = distillation_utils.DistillationForwardOutput(logits=t_logits, out_projection_activations=None)

    # Global reference (single-host).
    _, global_metrics = strategy.compute_loss(s_out, t_out, labels, step=None)
    global_mean = _mean(global_metrics["distill/hard_loss"])

    # Sharded: compute per-shard (sum, count), aggregate via weighted_mean.
    sharding = NamedSharding(mesh, P("data"))
    s_sharded = jax.device_put(s_logits, sharding)
    t_sharded = jax.device_put(t_logits, sharding)
    l_sharded = jax.device_put(labels, sharding)

    @jax.jit
    def per_shard(s_l, t_l, lab):
      s_o = distillation_utils.DistillationForwardOutput(logits=s_l, out_projection_activations=None)
      t_o = distillation_utils.DistillationForwardOutput(logits=t_l, out_projection_activations=None)
      _, m = strategy.compute_loss(s_o, t_o, lab, step=None)
      return m["distill/hard_loss"]  # (sum, count) — sharded

    shard_sum, shard_cnt = per_shard(s_sharded, t_sharded, l_sharded)
    # Pull each shard's contribution as if it came from a different host.
    per_shard_pairs = [
        (jax.device_get(shard_sum.addressable_shards[i].data), jax.device_get(shard_cnt.addressable_shards[i].data))
        for i in range(len(devices))
    ]
    np.testing.assert_allclose(distillation_utils.weighted_mean(per_shard_pairs), global_mean, rtol=1e-5)

  # --- 7. Feature-mapping loss (beta > 0 path) --------------------------

  def test_feature_loss_zero_when_student_features_match_teacher(self):
    """With beta>0 and s_features == t_features, distill/out_proj_feature_loss
    must be 0 (cosine distance of identical vectors). Total loss then equals the
    base logit loss — verifies the feature path doesn't contribute spurious mass."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 2, 8
    num_layers = 4
    s_logits = jnp.array([[[2.0, 1.0, 0.5, 0.1], [0.0, 3.0, 1.0, 0.5]]], dtype=jnp.float32)
    rng = np.random.default_rng(11)
    features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=features)
    t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=features)

    targets = jnp.array([[3, 1]], dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size, pad_id=0)

    strategy = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 2])
    total_loss, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)

    feat_sum, _ = metrics["distill/out_proj_feature_loss"]
    np.testing.assert_allclose(float(feat_sum), 0.0, atol=1e-5)
    # alpha=0 + identical logits + zero feature loss -> total_loss is just hard CE.
    np.testing.assert_allclose(float(total_loss), _mean(metrics["distill/hard_loss"]), rtol=1e-5)

  def test_feature_loss_positive_when_student_features_differ_and_respects_layer_indices(self):
    """With beta>0 and s_features != t_features on the sliced layers, feature
    loss must be > 0; and restricting layer_indices to a subset must yield a
    different value than using all layers (slice is actually applied)."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 2, 8
    num_layers = 4
    s_logits = jnp.array([[[2.0, 1.0, 0.5, 0.1], [0.0, 3.0, 1.0, 0.5]]], dtype=jnp.float32)
    rng = np.random.default_rng(12)
    s_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    # Teacher features differ on layers 1 and 3; match on layers 0 and 2.
    t_features = s_features.at[1].set(s_features[1] + 1.0).at[3].set(s_features[3] - 1.0)
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s_features)
    t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t_features)

    targets = jnp.array([[3, 1]], dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size, pad_id=0)

    # Case A: include only layers where features match -> feature loss == 0.
    strategy_match = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 2])
    _, m_match = strategy_match.compute_loss(s_out, t_out, labels, step=None)
    feat_match, _ = m_match["distill/out_proj_feature_loss"]
    np.testing.assert_allclose(float(feat_match), 0.0, atol=1e-5)

    # Case B: include the differing layers -> feature loss > 0.
    strategy_diff = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[1, 3])
    _, m_diff = strategy_diff.compute_loss(s_out, t_out, labels, step=None)
    feat_diff, _ = m_diff["distill/out_proj_feature_loss"]
    self.assertGreater(float(feat_diff), 0.0)

  # --- 7b. Feature loss NaN-safety (pad mask + epsilon) -----------------

  def _features_with_zero_pad_positions(self, num_layers, batch_size, seq_len, hidden_dim, rng):
    """Returns [L, B, T, D] features where the last column of T is all zeros
    (matches what typically sits at padded positions after masked attention)."""
    feats = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    feats = feats.at[:, :, -1, :].set(0.0)
    return feats

  def test_feature_loss_stays_finite_when_padded_positions_are_zero_norm(self):
    """Before the patch, cosine distance at padded positions gave 0/0 = NaN
    and the mean leaked into the aggregated loss. The mask must zero those
    positions out before any reduction."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 2, 4, 8
    num_layers = 4
    rng = np.random.default_rng(21)
    # Padded positions (last token) have zero features for BOTH student and teacher.
    s_features = self._features_with_zero_pad_positions(num_layers, batch_size, seq_len, hidden_dim, rng)
    t_features = self._features_with_zero_pad_positions(num_layers, batch_size, seq_len, hidden_dim, rng)

    s_logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.float32)
    # Targets: last position is pad (pad_id=0); others are valid tokens 1..3.
    targets = jnp.array([[1, 2, 3, 0], [3, 2, 1, 0]], dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size, pad_id=0)

    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s_features)
    t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t_features)

    strategy = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 1, 2, 3])
    total_loss, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)

    feat_sum, _ = metrics["distill/out_proj_feature_loss"]
    self.assertTrue(np.isfinite(float(feat_sum)), f"feature loss not finite: {float(feat_sum)!r}")
    self.assertTrue(np.isfinite(float(total_loss)), f"total loss not finite: {float(total_loss)!r}")

  def test_feature_loss_is_invariant_to_padded_position_values(self):
    """The mask must make the feature loss independent of what sits at
    padded positions — i.e. rewriting pad features to garbage or zeros must
    not change the loss value."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 3, 8
    num_layers = 2
    rng = np.random.default_rng(22)
    s_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    t_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))

    # Mark the last position as padding in labels.
    targets = jnp.array([[1, 2, 0]], dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size, pad_id=0)
    s_logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.float32)

    def run(pad_filler):
      s = s_features.at[:, :, -1, :].set(pad_filler)
      t = t_features.at[:, :, -1, :].set(pad_filler)
      s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s)
      t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t)
      strategy = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 1])
      _, m = strategy.compute_loss(s_out, t_out, labels, step=None)
      return float(m["distill/out_proj_feature_loss"][0])

    # Replace pad positions with zeros, then with huge noise; both must give the same loss.
    loss_zero_pad = run(0.0)
    loss_huge_pad = run(1e6)
    np.testing.assert_allclose(loss_zero_pad, loss_huge_pad, rtol=1e-5, atol=1e-6)

  def test_feature_loss_safe_norm_handles_zero_valid_activations(self):
    """Even on valid (non-padded) positions, a zero-norm row must not blow up
    cosine distance — the optax epsilon in the patched fn floors the
    denominator. This guards against late-training activation collapse."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 2, 8
    num_layers = 2
    # BOTH student and teacher have zero-norm vectors on ALL positions
    # (including the valid ones). No mask can rescue us here — epsilon must.
    s_features = jnp.zeros((num_layers, batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    t_features = jnp.zeros((num_layers, batch_size, seq_len, hidden_dim), dtype=jnp.float32)

    targets = jnp.array([[1, 2]], dtype=jnp.int32)  # both valid
    labels = _make_labels(targets, vocab_size, pad_id=0)
    s_logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.float32)

    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s_features)
    t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t_features)

    strategy = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 1])
    total_loss, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)

    feat_sum, _ = metrics["distill/out_proj_feature_loss"]
    self.assertTrue(np.isfinite(float(feat_sum)))
    self.assertTrue(np.isfinite(float(total_loss)))

  def test_feature_loss_gradient_finite_under_degenerate_inputs(self):
    """The backward pass is where 0/0 actually bites in training: jax.grad
    of cosine on a zero-norm row would produce NaN without epsilon. Assert
    gradients flow cleanly through the patched feature loss."""
    num_layers, batch_size, seq_len, hidden_dim = 2, 1, 2, 4

    def _loss(s, t, mask):
      # Exercise the patched default feature_loss_fn in isolation.
      strategy = _make_strategy(vocab_size=4, alpha=0.0, beta_feature=1.0)
      return strategy.feature_loss_fn(s, t, mask)

    # Degenerate case: zero-norm rows everywhere.
    s = jnp.zeros((num_layers, batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    t = jnp.zeros((num_layers, batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

    grad_s = jax.grad(_loss, argnums=0)(s, t, mask)
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad_s))), f"grad not finite: {np.asarray(grad_s)!r}")

  def test_feature_loss_l2_respects_mask(self):
    """Same masking contract applies to the L2 branch — padded positions
    must not contribute to the loss."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 3, 8
    num_layers = 2
    rng = np.random.default_rng(23)
    s_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    t_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))

    targets = jnp.array([[1, 2, 0]], dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size, pad_id=0)
    s_logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.float32)

    def run(pad_filler):
      s = s_features.at[:, :, -1, :].set(pad_filler)
      t = t_features.at[:, :, -1, :].set(pad_filler)
      # Build an L2 strategy directly (constructor override).
      strategy = distillation_utils.CombinedDistillationStrategy(
          student_forward_fn=lambda *_a, **_k: distillation_utils.DistillationForwardOutput(logits=s_logits),
          teacher_forward_fn=lambda *_a, **_k: distillation_utils.DistillationForwardOutput(logits=s_logits),
          pad_id=0,
          temperature=1.0,
          alpha=0.0,
          beta_feature=1.0,
          layer_indices=[0, 1],
          feature_loss_type="l2",
          vocab_size=vocab_size,
          max_steps=1,
      )
      s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s)
      t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t)
      _, m = strategy.compute_loss(s_out, t_out, labels, step=None)
      return float(m["distill/out_proj_feature_loss"][0])

    loss_zero_pad = run(0.0)
    loss_huge_pad = run(1e6)
    np.testing.assert_allclose(loss_zero_pad, loss_huge_pad, rtol=1e-5, atol=1e-6)

  # --- 8. compute_eval_loss returns (sum, count) ------------------------

  def test_compute_eval_loss_returns_sum_count_pair(self):
    vocab_size = 4
    s_logits = jnp.array([[[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]], dtype=jnp.float32)
    targets = jnp.array([[0, 1]], dtype=jnp.int32)
    labels = _make_labels(targets, vocab_size)
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    strategy = _make_strategy(vocab_size)
    task_loss, metrics = strategy.compute_eval_loss(s_out, labels)
    self.assertIn("eval/hard_loss", metrics)
    np.testing.assert_allclose(_mean(metrics["eval/hard_loss"]), float(task_loss), rtol=1e-6)

  # --- 9. Integer labels, chunked losses and the offline top-K path -------

  def test_integer_label_losses_match_one_hot_reference(self):
    """Losses from integer labels equal the dense one-hot reference formulas."""
    vocab_size, batch_size, seq_len, temperature, alpha = 16, 2, 6, 2.0, 0.3
    rng = np.random.default_rng(10)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    targets = jnp.asarray(rng.integers(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32))
    targets = targets.at[0, -2:].set(0)  # padding
    strategy = _make_strategy(vocab_size, temperature=temperature, alpha=alpha)
    labels = strategy.create_labels(targets)
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits)
    t_out = distillation_utils.DistillationForwardOutput(logits=t_logits)
    total_loss, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)

    one_hot = jax.nn.one_hot(targets, vocab_size) * (targets != 0)[..., None]
    mask = (targets != 0).astype(jnp.float32)
    count = float(jnp.sum(mask))
    ce_s = float(jnp.sum(optax.softmax_cross_entropy(s_logits, one_hot) * mask))
    ce_t = float(jnp.sum(optax.softmax_cross_entropy(t_logits, one_hot) * mask))
    kl_t = optax.kl_divergence(jax.nn.log_softmax(s_logits / temperature), jax.nn.softmax(t_logits / temperature))
    kl_t = float(jnp.sum(kl_t * mask))
    kl_1 = float(jnp.sum(optax.kl_divergence(jax.nn.log_softmax(s_logits), jax.nn.softmax(t_logits)) * mask))
    np.testing.assert_allclose(_mean(metrics["distill/hard_loss"]), ce_s / count, rtol=1e-5)
    np.testing.assert_allclose(_mean(metrics["distill/teacher_loss"]), ce_t / count, rtol=1e-5)
    np.testing.assert_allclose(_mean(metrics["distill/kl_div_at_T"]), kl_t / count, rtol=1e-5)
    np.testing.assert_allclose(_mean(metrics["distill/kl_div_T1"]), kl_1 / count, rtol=1e-5)
    expected_total = alpha * kl_t * temperature**2 / count + (1 - alpha) * ce_s / count
    np.testing.assert_allclose(float(total_loss), expected_total, rtol=1e-5)

  def test_chunked_logit_losses_match_single_chunk(self):
    """Sequence-chunked loss computation matches the unchunked one (value and gradient)."""
    vocab_size, batch_size, seq_len = 32, 2, 16
    rng = np.random.default_rng(11)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    targets = jnp.asarray(rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32))

    def loss_and_grad(num_chunks, temperature, alpha):
      strategy = distillation_utils.CombinedDistillationStrategy(
          student_forward_fn=None,
          teacher_forward_fn=None,
          temperature=temperature,
          alpha=alpha,
          vocab_size=vocab_size,
          logit_loss_chunks=num_chunks,
      )
      labels = strategy.create_labels(targets)
      t_out = distillation_utils.DistillationForwardOutput(logits=t_logits)

      def f(s):
        loss, metrics = strategy.compute_loss(distillation_utils.DistillationForwardOutput(logits=s), t_out, labels)
        return loss, metrics

      (loss, metrics), grad = jax.jit(jax.value_and_grad(f, has_aux=True))(s_logits)
      return loss, metrics, grad

    for temperature, alpha in [(1.0, 1.0), (2.0, 0.5)]:
      ref_loss, ref_metrics, ref_grad = loss_and_grad(1, temperature, alpha)
      for num_chunks in (2, 4):
        loss, metrics, grad = loss_and_grad(num_chunks, temperature, alpha)
        np.testing.assert_allclose(float(loss), float(ref_loss), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(grad), np.asarray(ref_grad), rtol=1e-4, atol=1e-7)
        for key in ("distill/hard_loss", "distill/teacher_loss", "distill/kl_div_T1"):
          np.testing.assert_allclose(_mean(metrics[key]), _mean(ref_metrics[key]), rtol=1e-5)

  def test_chunked_logit_losses_reject_non_divisible_chunks(self):
    vocab_size = 8
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=None, teacher_forward_fn=None, vocab_size=vocab_size, logit_loss_chunks=4
    )
    logits = jnp.zeros((1, 6, vocab_size))
    out = distillation_utils.DistillationForwardOutput(logits=logits)
    with self.assertRaisesRegex(ValueError, "divisible"):
      strategy.compute_loss(out, out, strategy.create_labels(jnp.ones((1, 6), jnp.int32)))

  def test_top_k_offline_losses(self):
    """With teacher top-K logits, KL is computed over the teacher's top-K subset and teacher metrics are 0."""
    vocab_size, batch_size, seq_len, k, temperature = 16, 1, 3, 4, 2.0
    rng = np.random.default_rng(12)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    full_t = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_top_k, top_k_idx = jax.lax.top_k(full_t, k)
    targets = jnp.array([[3, 5, 0]], dtype=jnp.int32)
    strategy = _make_strategy(vocab_size, temperature=temperature, alpha=0.5)
    labels = strategy.create_labels(targets)
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits)
    t_out = distillation_utils.DistillationForwardOutput(logits=t_top_k, top_k_indices=top_k_idx)
    _, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)

    mask = np.array([[1.0, 1.0, 0.0]])
    s_sub = jnp.take_along_axis(s_logits, top_k_idx, axis=-1)
    t_p = jax.nn.softmax(t_top_k / temperature)
    kl = jnp.sum(t_p * (jax.nn.log_softmax(t_top_k / temperature) - jax.nn.log_softmax(s_sub / temperature)), -1)
    ce_s = optax.softmax_cross_entropy_with_integer_labels(s_logits, targets)
    np.testing.assert_allclose(_mean(metrics["distill/kl_div_at_T"]), float(jnp.sum(kl * mask)) / 2.0, rtol=1e-5)
    np.testing.assert_allclose(_mean(metrics["distill/hard_loss"]), float(jnp.sum(ce_s * mask)) / 2.0, rtol=1e-5)
    self.assertEqual(_mean(metrics["distill/teacher_loss"]), 0.0)
    self.assertEqual(_mean(metrics["distill/kl_div_T1"]), 0.0)

  # --- 10. Fused LM head ---------------------------------------------------

  @staticmethod
  def _reference_logits(hidden, embedding, head):
    """Mirrors the tied-embedding branch of the decoder's apply_output_head."""
    dot_dtype = jnp.float32 if head.logits_dot_in_fp32 else hidden.dtype
    logits = jnp.einsum("...e,ve->...v", hidden, embedding.astype(jnp.bfloat16), preferred_element_type=dot_dtype)
    if head.normalize_embedding_logits:
      logits = logits / jnp.sqrt(hidden.shape[-1])
    if head.final_logits_soft_cap:
      logits = jnp.tanh(logits / head.final_logits_soft_cap) * head.final_logits_soft_cap
    return logits

  def _fused_inputs(self, seq_len=8):
    rng = np.random.default_rng(13)
    batch_size, emb, vocab_size = 2, 16, 32
    s_hidden = jnp.asarray(rng.normal(size=(batch_size, seq_len, emb)).astype(np.float32))
    t_hidden = jnp.asarray(rng.normal(size=(batch_size, seq_len, emb)).astype(np.float32))
    s_emb = jnp.asarray(rng.normal(size=(vocab_size, emb)).astype(np.float32))
    t_emb = jnp.asarray(rng.normal(size=(vocab_size, emb)).astype(np.float32))
    targets = jnp.asarray(rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32))
    return s_hidden, t_hidden, s_emb, t_emb, targets, vocab_size

  def test_fused_lm_head_matches_unfused(self):
    """Fused (chunked) LM-head losses and gradients match losses on precomputed logits."""
    s_hidden, t_hidden, s_emb, t_emb, targets, vocab_size = self._fused_inputs()
    for normalize in (True, False):
      for soft_cap in (None, 3.0):
        student_head = distillation_utils.LMHeadSpec(normalize, soft_cap, logits_dot_in_fp32=True)
        # The teacher keeps its own settings (different soft cap) instead of reusing the student's.
        teacher_head = distillation_utils.LMHeadSpec(normalize, 5.0, logits_dot_in_fp32=True)
        kwargs = {"student_forward_fn": None, "teacher_forward_fn": None, "temperature": 2.0, "alpha": 0.5}
        fused = distillation_utils.CombinedDistillationStrategy(
            vocab_size=vocab_size,
            fused_lm_head_chunks=4,
            student_lm_head=student_head,
            teacher_lm_head=teacher_head,
            **kwargs,
        )
        unfused = distillation_utils.CombinedDistillationStrategy(vocab_size=vocab_size, **kwargs)
        labels = fused.create_labels(targets)
        t_logits = self._reference_logits(t_hidden, t_emb, teacher_head)

        def fused_loss(s_h, s_e, strategy=fused, labels=labels):
          s_out = distillation_utils.DistillationForwardOutput(logits=(s_h, s_e))
          t_out = distillation_utils.DistillationForwardOutput(logits=(t_hidden, t_emb))
          return strategy.compute_loss(s_out, t_out, labels)

        def unfused_loss(s_h, s_e, strategy=unfused, head=student_head, t_logits=t_logits, labels=labels):
          s_out = distillation_utils.DistillationForwardOutput(logits=self._reference_logits(s_h, s_e, head))
          t_out = distillation_utils.DistillationForwardOutput(logits=t_logits)
          return strategy.compute_loss(s_out, t_out, labels)

        def grad_fn(f):
          return jax.jit(jax.value_and_grad(f, argnums=(0, 1), has_aux=True))

        (loss_f, metrics_f), (gh_f, ge_f) = grad_fn(fused_loss)(s_hidden, s_emb)
        (loss_u, metrics_u), (gh_u, ge_u) = grad_fn(unfused_loss)(s_hidden, s_emb)
        msg = f"normalize={normalize}, soft_cap={soft_cap}"
        np.testing.assert_allclose(float(loss_f), float(loss_u), rtol=1e-5, err_msg=msg)
        for key in ("distill/hard_loss", "distill/teacher_loss", "distill/kl_div_T1"):
          np.testing.assert_allclose(_mean(metrics_f[key]), _mean(metrics_u[key]), rtol=1e-5, err_msg=msg)
        np.testing.assert_allclose(np.asarray(gh_f), np.asarray(gh_u), rtol=1e-4, atol=1e-6, err_msg=msg)
        # The student's (tied) embedding receives the LM-head gradient.
        self.assertGreater(float(jnp.max(jnp.abs(ge_f))), 0.0)
        # The embedding cotangent passes through the same bfloat16 cast as the decoder's LM head, so per-chunk
        # accumulation differs from the single-matmul gradient by bfloat16 rounding.
        np.testing.assert_allclose(np.asarray(ge_f), np.asarray(ge_u), rtol=2e-2, atol=1e-4, err_msg=msg)

  def test_fused_lm_head_eval_loss_matches_unfused(self):
    s_hidden, _, s_emb, _, targets, vocab_size = self._fused_inputs()
    head = distillation_utils.LMHeadSpec(True, 3.0, logits_dot_in_fp32=True)
    fused = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=None,
        teacher_forward_fn=None,
        vocab_size=vocab_size,
        fused_lm_head_chunks=2,
        student_lm_head=head,
        teacher_lm_head=head,
    )
    labels = fused.create_labels(targets)
    loss_f, _ = fused.compute_eval_loss(distillation_utils.DistillationForwardOutput(logits=(s_hidden, s_emb)), labels)
    s_logits = self._reference_logits(s_hidden, s_emb, head)
    loss_u, _ = fused.compute_eval_loss(distillation_utils.DistillationForwardOutput(logits=s_logits), labels)
    np.testing.assert_allclose(float(loss_f), float(loss_u), rtol=1e-5)

  def test_fused_lm_head_rejects_non_divisible_chunks(self):
    s_hidden, t_hidden, s_emb, t_emb, targets, vocab_size = self._fused_inputs(seq_len=6)
    head = distillation_utils.LMHeadSpec(True, None, logits_dot_in_fp32=True)
    fused = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=None,
        teacher_forward_fn=None,
        vocab_size=vocab_size,
        fused_lm_head_chunks=4,
        student_lm_head=head,
        teacher_lm_head=head,
    )
    s_out = distillation_utils.DistillationForwardOutput(logits=(s_hidden, s_emb))
    t_out = distillation_utils.DistillationForwardOutput(logits=(t_hidden, t_emb))
    with self.assertRaisesRegex(ValueError, "divisible"):
      fused.compute_loss(s_out, t_out, fused.create_labels(targets))

  def test_fused_lm_head_requires_both_models_fused(self):
    s_hidden, _, s_emb, _, targets, vocab_size = self._fused_inputs()
    head = distillation_utils.LMHeadSpec(True, None, logits_dot_in_fp32=True)
    fused = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=None,
        teacher_forward_fn=None,
        vocab_size=vocab_size,
        fused_lm_head_chunks=4,
        student_lm_head=head,
        teacher_lm_head=head,
    )
    s_out = distillation_utils.DistillationForwardOutput(logits=(s_hidden, s_emb))
    t_out = distillation_utils.DistillationForwardOutput(logits=jnp.zeros(targets.shape + (vocab_size,)))
    with self.assertRaisesRegex(ValueError, "both the student and the teacher"):
      fused.compute_loss(s_out, t_out, fused.create_labels(targets))


if __name__ == "__main__":
  absltest.main()
